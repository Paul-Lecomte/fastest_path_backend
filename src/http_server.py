# This module implements a simple HTTP server that provides a pathfinding API for transit networks.
from __future__ import annotations

import json
import logging
import time
import heapq
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .config import setup_logging
from .loader import NetworkLoader, build_mock_network, TransitNetwork, summarize_trip_profiles
from .solver import (
    build_path,
    build_path_dijkstra,
    run_dijkstra_fast,
    run_raptor,
    run_raptor_with_stats,
    run_astar_fast,
    numba_enabled,
)


logger = logging.getLogger("pathfinding.http")

DEFAULT_OFFSET_MINUTES: tuple[int, ...] = ()
DEFAULT_OPTION_COUNT = 3
DEFAULT_NEXT_OPTION_SEARCH_WINDOW_MINUTES = 90
DEFAULT_NEXT_OPTION_STEP_SECONDS = 300
DEFAULT_NEXT_OPTION_MAX_STEP_SECONDS = 1800
DEFAULT_NEXT_OPTION_MAX_EVALS = 3
DEFAULT_NEXT_OPTION_MAX_WALL_SECONDS = 8.0
DEFAULT_START_EXPANSION_HOPS = 2
DEFAULT_START_EXPANSION_MAX_STOPS = 256
DEFAULT_ORIGIN_RADIUS_M = 1000.0
DEFAULT_ORIGIN_MAX_CANDIDATES = 12
DEFAULT_ORIGIN_SEED_CANDIDATES = 2
DEFAULT_DESTINATION_SEED_CANDIDATES = 6
DEFAULT_WALK_SPEED_MPS = 1.4
UNIX_TIMESTAMP_MIN_SECONDS = 946684800
DEFAULT_RAPTOR_ROUNDS_MIN = 8
DEFAULT_RAPTOR_ROUNDS_MAX = 64
DEFAULT_ADAPTIVE_EXPANSION_MAX_TIERS = 4
DEFAULT_RAPTOR_TRANSFER_PENALTY_SECONDS = 3600
DEFAULT_OPTION_TRANSFER_PENALTY_SECONDS = 3600
DEFAULT_MAX_TRANSFERS = 4


def _snapshot_indices(values: list[int], limit: int = 64) -> list[int]:
    if len(values) <= limit:
        return [int(item) for item in values]
    return [int(item) for item in values[:limit]]


def _collect_candidate_counts(
    diagnostics: dict[str, Any] | None,
) -> tuple[set[int], set[int]]:
    start_counts: set[int] = set()
    end_counts: set[int] = set()
    if not diagnostics:
        return start_counts, end_counts

    for run in diagnostics.get("runs", []):
        try:
            start_counts.add(int(run.get("start_candidates", 0)))
            end_counts.add(int(run.get("end_candidates", 0)))
        except (TypeError, ValueError):
            continue
    return start_counts, end_counts


def _build_no_path_reason(
    diagnostics: dict[str, Any] | None,
    start_counts: set[int],
    end_counts: set[int],
) -> str:
    if not diagnostics or not diagnostics.get("runs"):
        return "no_attempt"
    if len(end_counts) > 1:
        return "no_path_after_destination_expansion"
    if len(start_counts) > 1:
        return "no_path_after_origin_expansion"
    return "no_path_within_candidate_budget"


def _merge_raptor_diagnostics(
    base: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "runs": [],
        "attempt_caps": [],
        "max_rounds_reached": False,
        "rounds_used_max": 0,
    }

    for payload in (base, current):
        if not payload:
            continue
        merged["runs"].extend(payload.get("runs", []))
        merged["attempt_caps"].extend(payload.get("attempt_caps", []))
        merged["max_rounds_reached"] = bool(merged["max_rounds_reached"] or payload.get("max_rounds_reached", False))
        payload_rounds = int(payload.get("rounds_used_max", 0))
        if payload_rounds > int(merged["rounds_used_max"]):
            merged["rounds_used_max"] = payload_rounds

    merged["attempt_caps"] = sorted(set(int(value) for value in merged["attempt_caps"]))
    return merged


def _candidate_tiers(
    count: int,
    start_count: int,
    max_tiers: int = DEFAULT_ADAPTIVE_EXPANSION_MAX_TIERS,
) -> tuple[int, ...]:
    if count <= 0 or start_count <= 0:
        return ()
    if count <= start_count:
        return (int(count),)
    tiers = []
    current = int(start_count)
    while current < count and len(tiers) + 1 < max_tiers:
        tiers.append(current)
        current = min(count, current * 2)
    tiers.append(count)
    return tuple(dict.fromkeys(int(value) for value in tiers))


def _sorted_stops_by_distance(
    network: TransitNetwork,
    lat: float,
    lon: float,
):
    valid_mask = np.isfinite(network.stop_lats) & np.isfinite(network.stop_lons)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    distances = _haversine_distance_m(
        lat,
        lon,
        network.stop_lats[valid_indices],
        network.stop_lons[valid_indices],
    )
    order = np.argsort(distances)
    return valid_indices[order], distances[order]


def _rank_origin_candidates_by_connectivity(
    network: TransitNetwork,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    walk_speed_mps: float,
) -> np.ndarray:
    if candidate_indices.size <= 1:
        return candidate_indices

    n_stops = int(network.stop_route_offsets.shape[0] - 1)
    ranked_payload: list[tuple[float, float, int]] = []
    for stop_idx, distance_m in zip(candidate_indices.tolist(), candidate_distances.tolist()):
        route_degree = 0
        if 0 <= int(stop_idx) < n_stops:
            row_start = int(network.stop_route_offsets[int(stop_idx)])
            row_end = int(network.stop_route_offsets[int(stop_idx) + 1])
            route_degree = max(0, row_end - row_start)
        walk_seconds = float(distance_m / max(0.5, float(walk_speed_mps)))
        utility = float(route_degree) * 600.0 - walk_seconds
        ranked_payload.append((utility, float(distance_m), int(stop_idx)))

    ranked_payload.sort(key=lambda item: (-item[0], item[1], item[2]))
    return np.array([item[2] for item in ranked_payload], dtype=np.int64)


def _expand_candidates_tiered(
    network: TransitNetwork,
    seed_indices: list[int],
    anchor: dict[str, Any] | None,
    candidate_type: str,
):
    unique_seed = [int(value) for value in dict.fromkeys(seed_indices)]
    if not unique_seed:
        return [unique_seed]

    if not isinstance(anchor, dict):
        return [unique_seed]

    lat = _to_float(anchor.get("lat"))
    lon = _to_float(anchor.get("lon"))
    if lat is None or lon is None:
        return [unique_seed]

    sorted_indices, _ = _sorted_stops_by_distance(network, lat, lon)
    if sorted_indices.size == 0:
        return [unique_seed]

    max_candidates = _to_int(anchor.get("max_candidates"), len(unique_seed))
    if max_candidates <= 0:
        max_candidates = len(unique_seed)

    tier_sets: list[list[int]] = [unique_seed]
    for size in _candidate_tiers(max_candidates, len(unique_seed)):
        selected = [int(value) for value in sorted_indices[:size].tolist()]
        if not selected:
            continue
        if selected == tier_sets[-1]:
            continue
        tier_sets.append(selected)

    logger.debug(
        "Adaptive %s tiers base=%s tiers=%s",
        candidate_type,
        len(unique_seed),
        [len(values) for values in tier_sets],
    )
    return tier_sets


def _count_transfers(segments) -> int:
    if not segments:
        return 0
    transfers = 0
    current_trip = None
    for trip_id, _, _ in segments:
        if int(trip_id) >= 0:
            current_trip = int(trip_id)
            break
    if current_trip is None:
        return 0

    for trip_id, _, _ in segments[1:]:
        trip_value = int(trip_id)
        if trip_value < 0:
            continue
        if trip_value != current_trip:
            transfers += 1
            current_trip = trip_value
    return transfers


def _score_segments_with_transfer_penalty(
    segments,
    base_score: int,
    transfer_penalty_seconds: int = DEFAULT_OPTION_TRANSFER_PENALTY_SECONDS,
    max_transfers: int = DEFAULT_MAX_TRANSFERS,
) -> int:
    transfer_count = int(_count_transfers(segments))
    if transfer_count > int(max_transfers):
        return int(2**62 - 1)
    return int(base_score) + transfer_count * int(transfer_penalty_seconds)


def _first_transit_trip_signature(option: dict[str, Any]) -> tuple[str, str | None]:
    segments = option.get("segments")
    if not isinstance(segments, list) or not segments:
        return ("NONE", None)
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        trip_id = segment.get("trip_id")
        if isinstance(trip_id, str) and trip_id and trip_id != "TRANSFER":
            stop_id = segment.get("stop_id")
            return (trip_id, stop_id if isinstance(stop_id, str) else None)
    return ("TRANSFER_ONLY", None)


def _summarize_option_trip_profile(network: TransitNetwork, segments) -> dict[str, Any]:
    if not segments:
        return {
            "trip_segment_count": 0,
            "distinct_trip_count": 0,
            "factor_counts": [],
            "route_type_counts": [],
            "unknown_route_type_share": 0.0,
        }

    distinct_trips = []
    seen = set()
    for trip_id, _, _ in segments:
        trip_idx = int(trip_id)
        if trip_idx < 0 or trip_idx in seen:
            continue
        seen.add(trip_idx)
        distinct_trips.append(trip_idx)

    if not distinct_trips:
        return {
            "trip_segment_count": 0,
            "distinct_trip_count": 0,
            "factor_counts": [],
            "route_type_counts": [],
            "unknown_route_type_share": 0.0,
        }

    factor_counts: dict[int, int] = {}
    route_type_counts: dict[int, int] = {}
    unknown_count = 0

    trip_cost_factors = network.trip_cost_factors
    trip_route_types = network.trip_route_types
    for trip_idx in distinct_trips:
        factor = int(trip_cost_factors[trip_idx]) if trip_idx < trip_cost_factors.shape[0] else 1500
        route_type = int(trip_route_types[trip_idx]) if trip_idx < trip_route_types.shape[0] else -1
        factor_counts[factor] = int(factor_counts.get(factor, 0) + 1)
        route_type_counts[route_type] = int(route_type_counts.get(route_type, 0) + 1)
        if route_type == -1:
            unknown_count += 1

    sorted_factor_counts = sorted(
        [(int(factor), int(count)) for factor, count in factor_counts.items()],
        key=lambda item: item[1],
        reverse=True,
    )
    sorted_route_type_counts = sorted(
        [(int(route_type), int(count)) for route_type, count in route_type_counts.items()],
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "trip_segment_count": int(sum(1 for trip_id, _, _ in segments if int(trip_id) >= 0)),
        "distinct_trip_count": int(len(distinct_trips)),
        "factor_counts": sorted_factor_counts,
        "route_type_counts": sorted_route_type_counts,
        "unknown_route_type_share": round(float(unknown_count / max(1, len(distinct_trips))), 6),
    }


def _departure_to_seconds(value: Any) -> int | None:
    def _from_unix_timestamp(timestamp_value: int) -> int:
        utc = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        return utc.hour * 3600 + utc.minute * 60 + utc.second

    def _normalize(value_seconds: int) -> int:
        if abs(value_seconds) >= UNIX_TIMESTAMP_MIN_SECONDS:
            return _from_unix_timestamp(value_seconds)
        return value_seconds

    if isinstance(value, int):
        return _normalize(value)
    if isinstance(value, float):
        return _normalize(int(value))
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return _normalize(int(text))
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _haversine_distance_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_earth_m = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    sin_half_dphi = np.sin(dphi / 2.0)
    sin_half_dlambda = np.sin(dlambda / 2.0)
    a = sin_half_dphi * sin_half_dphi + np.cos(phi1) * np.cos(phi2) * sin_half_dlambda * sin_half_dlambda
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_earth_m * c


def _stop_station_id(network: TransitNetwork, stop_id: int) -> int:
    mapping = getattr(network, "stop_station_ids", None)
    if isinstance(mapping, np.ndarray) and 0 <= int(stop_id) < mapping.shape[0]:
        return int(mapping[int(stop_id)])
    stop_text = network.stop_ids[int(stop_id)] if 0 <= int(stop_id) < len(network.stop_ids) else ""
    key = stop_text.split(":", 1)[0] if isinstance(stop_text, str) else ""
    if not key:
        return int(stop_id)
    station_keys = getattr(network, "station_keys", None)
    if isinstance(station_keys, list):
        try:
            return int(station_keys.index(key))
        except ValueError:
            return int(stop_id)
    return int(stop_id)


def _nearest_hub_station(network: TransitNetwork, station_id: int) -> int:
    hub_indices = getattr(network, "hub_station_indices", None)
    station_lats = getattr(network, "station_lats", None)
    station_lons = getattr(network, "station_lons", None)
    if not isinstance(hub_indices, np.ndarray) or hub_indices.size == 0:
        return int(station_id)
    if not isinstance(station_lats, np.ndarray) or not isinstance(station_lons, np.ndarray):
        return int(station_id)
    if 0 <= int(station_id) < hub_indices.shape[0] and int(station_id) in set(hub_indices.tolist()):
        return int(station_id)

    if not (0 <= int(station_id) < station_lats.shape[0]):
        return int(hub_indices[0])
    lat = float(station_lats[int(station_id)])
    lon = float(station_lons[int(station_id)])
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return int(hub_indices[0])

    hub_lats = station_lats[hub_indices]
    hub_lons = station_lons[hub_indices]
    valid = np.isfinite(hub_lats) & np.isfinite(hub_lons)
    if not np.any(valid):
        return int(hub_indices[0])
    valid_hubs = hub_indices[valid]
    distances = _haversine_distance_m(lat, lon, hub_lats[valid], hub_lons[valid])
    best = int(np.argmin(distances))
    return int(valid_hubs[best])


def _station_path_lexicographic(
    network: TransitNetwork,
    start_station: int,
    end_station: int,
    max_hops: int,
) -> list[int]:
    offsets = getattr(network, "station_adj_offsets", None)
    neighbors = getattr(network, "station_adj_neighbors", None)
    weights = getattr(network, "station_adj_weights", None)
    if not (isinstance(offsets, np.ndarray) and isinstance(neighbors, np.ndarray) and isinstance(weights, np.ndarray)):
        return []
    station_count = int(offsets.shape[0] - 1)
    if station_count <= 0:
        return []
    if not (0 <= int(start_station) < station_count and 0 <= int(end_station) < station_count):
        return []

    inf = int(2**62 - 1)
    best_hops = np.full(station_count, np.int64(inf), dtype=np.int64)
    best_time = np.full(station_count, np.int64(inf), dtype=np.int64)
    predecessor = np.full(station_count, -1, dtype=np.int64)

    start_station = int(start_station)
    end_station = int(end_station)
    best_hops[start_station] = 0
    best_time[start_station] = 0

    queue: list[tuple[int, int, int]] = [(0, 0, start_station)]
    while queue:
        hops, total_time, station = heapq.heappop(queue)
        if hops > int(best_hops[station]):
            continue
        if hops == int(best_hops[station]) and total_time > int(best_time[station]):
            continue
        if station == end_station:
            break
        if hops >= int(max_hops):
            continue

        row_start = int(offsets[station])
        row_end = int(offsets[station + 1])
        for idx in range(row_start, row_end):
            target = int(neighbors[idx])
            next_hops = hops + 1
            next_time = total_time + int(weights[idx])
            should_update = (
                next_hops < int(best_hops[target])
                or (next_hops == int(best_hops[target]) and next_time < int(best_time[target]))
            )
            if should_update:
                best_hops[target] = next_hops
                best_time[target] = next_time
                predecessor[target] = station
                heapq.heappush(queue, (next_hops, next_time, target))

    if int(best_hops[end_station]) >= inf:
        return []

    path = [end_station]
    current = end_station
    while current != start_station:
        current = int(predecessor[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return [int(value) for value in path]


def _compute_station_backbone(
    network: TransitNetwork,
    start_indices: list[int],
    end_indices: list[int],
    max_transfers: int,
) -> tuple[set[int] | None, list[int] | None]:
    if not start_indices or not end_indices:
        return None, None

    start_stations = {_stop_station_id(network, stop_id) for stop_id in start_indices}
    end_stations = {_stop_station_id(network, stop_id) for stop_id in end_indices}
    if not start_stations or not end_stations:
        return None, None

    max_hops = max(2, int(max_transfers) + 2)
    best_path: list[int] | None = None
    for start_station in start_stations:
        start_hub = _nearest_hub_station(network, int(start_station))
        for end_station in end_stations:
            end_hub = _nearest_hub_station(network, int(end_station))
            station_path = _station_path_lexicographic(network, start_hub, end_hub, max_hops=max_hops)
            if not station_path:
                continue
            if best_path is None or len(station_path) < len(best_path):
                best_path = station_path

    if not best_path:
        return None, None

    whitelist = set(int(station_id) for station_id in best_path)
    station_adj_offsets = getattr(network, "station_adj_offsets", None)
    station_adj_neighbors = getattr(network, "station_adj_neighbors", None)
    if isinstance(station_adj_offsets, np.ndarray) and isinstance(station_adj_neighbors, np.ndarray):
        for station_id in best_path:
            row_start = int(station_adj_offsets[int(station_id)])
            row_end = int(station_adj_offsets[int(station_id) + 1])
            for idx in range(row_start, row_end):
                whitelist.add(int(station_adj_neighbors[idx]))

    for station_id in start_stations:
        whitelist.add(int(station_id))
    for station_id in end_stations:
        whitelist.add(int(station_id))
    return whitelist, best_path


def _segments_follow_station_backbone(
    network: TransitNetwork,
    segments,
    whitelist: set[int] | None,
    max_off_path_stations: int = 2,
) -> bool:
    if not segments or not whitelist:
        return True
    off_path = set()
    for trip_id, stop_id, _ in segments:
        if int(trip_id) < 0:
            continue
        station_id = _stop_station_id(network, int(stop_id))
        if station_id not in whitelist:
            off_path.add(int(station_id))
            if len(off_path) > int(max_off_path_stations):
                return False
    return True


def _segment_coordinates(network: TransitNetwork, stop_id: int) -> tuple[float | None, float | None]:
    lat = float(network.stop_lats[stop_id])
    lon = float(network.stop_lons[stop_id])
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None, None
    return lat, lon


def _segment_payload(network: TransitNetwork, trip_id: int, stop_id: int, arrival_time: int) -> dict[str, Any]:
    lat, lon = _segment_coordinates(network, stop_id)
    trip_value = int(trip_id)
    if trip_value < 0:
        trip_label = "TRANSFER"
    else:
        trip_label = network.trip_ids[trip_value]
    return {
        "trip_id": trip_label,
        "stop_id": network.stop_ids[stop_id],
        "arrival_time": int(arrival_time),
        "lat": lat,
        "lon": lon,
    }


def _select_starts_from_origin(network: TransitNetwork, origin: Any):
    if not isinstance(origin, dict):
        return [], {}, "invalid_origin"

    lat = _to_float(origin.get("lat"))
    lon = _to_float(origin.get("lon"))
    if lat is None or lon is None:
        return [], {}, "invalid_origin"

    radius_m = float(origin.get("radius_m", DEFAULT_ORIGIN_RADIUS_M))
    if radius_m <= 0:
        radius_m = DEFAULT_ORIGIN_RADIUS_M

    max_candidates = _to_int(origin.get("max_candidates"), DEFAULT_ORIGIN_MAX_CANDIDATES)
    if max_candidates <= 0:
        max_candidates = DEFAULT_ORIGIN_MAX_CANDIDATES

    walk_speed_mps = float(origin.get("walk_speed_mps", DEFAULT_WALK_SPEED_MPS))
    if walk_speed_mps <= 0:
        walk_speed_mps = DEFAULT_WALK_SPEED_MPS

    valid_mask = np.isfinite(network.stop_lats) & np.isfinite(network.stop_lons)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return [], {}, "missing_stop_coordinates"

    distances = _haversine_distance_m(
        lat,
        lon,
        network.stop_lats[valid_indices],
        network.stop_lons[valid_indices],
    )
    order = np.argsort(distances)
    sorted_indices = valid_indices[order]
    sorted_distances = distances[order]

    within_radius = sorted_distances <= radius_m
    if np.any(within_radius):
        selected_indices_full = sorted_indices[within_radius][:max_candidates]
        selected_distances_full = sorted_distances[within_radius][:max_candidates]
    else:
        fallback_count = min(max_candidates, sorted_indices.size)
        selected_indices_full = sorted_indices[:fallback_count]
        selected_distances_full = sorted_distances[:fallback_count]

    ranked_indices = _rank_origin_candidates_by_connectivity(
        network,
        selected_indices_full,
        selected_distances_full,
        walk_speed_mps,
    )
    distance_by_stop = {
        int(stop_idx): float(distance_m)
        for stop_idx, distance_m in zip(selected_indices_full.tolist(), selected_distances_full.tolist())
    }
    selected_indices_full = ranked_indices

    seed_candidates = _to_int(origin.get("seed_candidates"), min(DEFAULT_ORIGIN_SEED_CANDIDATES, max_candidates))
    seed_candidates = max(1, min(seed_candidates, int(selected_indices_full.size)))
    selected_indices = selected_indices_full[:seed_candidates]

    start_indices = [int(value) for value in selected_indices.tolist()]
    start_penalties = {
        int(stop_idx): int(distance_by_stop[int(stop_idx)] / walk_speed_mps)
        for stop_idx in selected_indices_full.tolist()
    }

    metadata = {
        "origin": {
            "lat": float(lat),
            "lon": float(lon),
            "radius_m": float(radius_m),
            "seed_candidates": int(seed_candidates),
            "max_candidates": int(max_candidates),
        },
        "candidate_start_count": len(start_indices),
        "candidate_start_count_max": int(selected_indices_full.size),
    }
    return start_indices, start_penalties, metadata


def _select_ends_from_destination(network: TransitNetwork, destination: Any):
    if not isinstance(destination, dict):
        return [], {}, "invalid_destination"

    lat = _to_float(destination.get("lat"))
    lon = _to_float(destination.get("lon"))
    if lat is None or lon is None:
        return [], {}, "invalid_destination"

    radius_m = float(destination.get("radius_m", DEFAULT_ORIGIN_RADIUS_M))
    if radius_m <= 0:
        radius_m = DEFAULT_ORIGIN_RADIUS_M

    max_candidates = _to_int(destination.get("max_candidates"), DEFAULT_ORIGIN_MAX_CANDIDATES)
    if max_candidates <= 0:
        max_candidates = DEFAULT_ORIGIN_MAX_CANDIDATES

    walk_speed_mps = float(destination.get("walk_speed_mps", DEFAULT_WALK_SPEED_MPS))
    if walk_speed_mps <= 0:
        walk_speed_mps = DEFAULT_WALK_SPEED_MPS

    valid_mask = np.isfinite(network.stop_lats) & np.isfinite(network.stop_lons)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return [], {}, "missing_stop_coordinates"

    distances = _haversine_distance_m(
        lat,
        lon,
        network.stop_lats[valid_indices],
        network.stop_lons[valid_indices],
    )
    order = np.argsort(distances)
    sorted_indices = valid_indices[order]
    sorted_distances = distances[order]

    within_radius = sorted_distances <= radius_m
    if np.any(within_radius):
        selected_indices_full = sorted_indices[within_radius][:max_candidates]
        selected_distances_full = sorted_distances[within_radius][:max_candidates]
    else:
        fallback_count = min(max_candidates, sorted_indices.size)
        selected_indices_full = sorted_indices[:fallback_count]
        selected_distances_full = sorted_distances[:fallback_count]

    seed_candidates = _to_int(destination.get("seed_candidates"), min(DEFAULT_DESTINATION_SEED_CANDIDATES, max_candidates))
    seed_candidates = max(1, min(seed_candidates, int(selected_indices_full.size)))
    selected_indices = selected_indices_full[:seed_candidates]

    end_indices = [int(value) for value in selected_indices.tolist()]
    end_penalties = {
        int(stop_idx): int(distance_m / walk_speed_mps)
        for stop_idx, distance_m in zip(selected_indices_full, selected_distances_full)
    }

    metadata = {
        "destination": {
            "lat": float(lat),
            "lon": float(lon),
            "radius_m": float(radius_m),
            "seed_candidates": int(seed_candidates),
            "max_candidates": int(max_candidates),
        },
        "candidate_end_count": len(end_indices),
        "candidate_end_count_max": int(selected_indices_full.size),
    }
    return end_indices, end_penalties, metadata


def load_network() -> TransitNetwork:
    from .config import get_neo4j_config, get_network_cache_config
    from .loader import load_network_from_cache, save_network_to_cache

    config = get_neo4j_config()
    cache_config = get_network_cache_config()
    cache_path = cache_config["path"]

    if cache_config["enabled"] and not cache_config["force_refresh"]:
        cached_network = load_network_from_cache(cache_path, cache_config["max_age_seconds"])
        if cached_network is not None:
            logger.info("Using cached network from %s", cache_path)
            return cached_network

    loader = NetworkLoader(config["uri"], config["user"], config["password"])
    try:
        logger.info("Loading network from Neo4j at %s", config["uri"])
        network = loader.fetch_to_numpy()
        if cache_config["enabled"]:
            save_network_to_cache(cache_path, network)
        logger.info(
            "Loaded network stops=%s stop_times=%s routes=%s",
            network.stops.shape[0],
            network.stop_times.shape[0],
            network.routes.shape[0],
        )
        return network
    except Exception as exc:
        if cache_config["enabled"]:
            cached_network = load_network_from_cache(cache_path)
            if cached_network is not None:
                logger.warning("Neo4j load failed, using cached network path=%s error=%s", cache_path, exc)
                return cached_network
        logger.warning("Neo4j load failed, using mock network: %s", exc)
        return build_mock_network()
    finally:
        loader.close()


def _warmup_raptor(network: TransitNetwork) -> None:
    n_stops = int(network.stop_route_offsets.shape[0] - 1)
    if n_stops <= 0:
        return

    start_idx = 0
    end_idx = 0 if n_stops == 1 else 1
    started = time.perf_counter()
    run_raptor(
        network.stop_times,
        network.trip_offsets,
        network.route_stop_offsets,
        network.route_stops,
        network.route_trip_offsets,
        network.route_trips,
        network.route_board_offsets,
        network.route_board_times,
        network.route_board_monotonic,
        network.stop_route_offsets,
        network.stop_routes,
        network.trip_cost_factors,
        DEFAULT_RAPTOR_TRANSFER_PENALTY_SECONDS,
        network.transfer_offsets,
        network.transfer_neighbors,
        network.transfer_weights,
        start_idx,
        end_idx,
        0,
        max_rounds=1,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("RAPTOR warmup complete in %.2f ms", elapsed_ms)


def _raptor_round_budgets(network: TransitNetwork) -> tuple[int, ...]:
    n_stops = int(network.stop_route_offsets.shape[0] - 1)
    adaptive_cap = min(
        DEFAULT_RAPTOR_ROUNDS_MAX,
        max(DEFAULT_RAPTOR_ROUNDS_MIN, 24, int(np.sqrt(max(1, n_stops))) + 8),
    )
    budgets = []
    current = DEFAULT_RAPTOR_ROUNDS_MIN
    while current < adaptive_cap:
        budgets.append(current)
        current *= 2
    budgets.append(adaptive_cap)
    return tuple(dict.fromkeys(budgets))


def _algorithm_sequence(primary: str) -> tuple[str, ...]:
    return (primary,)


def _compute_segments(
    network: TransitNetwork,
    algorithm: str,
    start_idx: int,
    end_idx: int,
    departure_time: int,
):
    if algorithm == "raptor":
        attempt_summaries = []
        for max_rounds in _raptor_round_budgets(network):
            started = time.perf_counter()
            earliest, pred_stop, pred_trip, pred_time, rounds_used, marked_count, reached_target = run_raptor_with_stats(
                network.stop_times,
                network.trip_offsets,
                network.route_stop_offsets,
                network.route_stops,
                network.route_trip_offsets,
                network.route_trips,
                network.route_board_offsets,
                network.route_board_times,
                network.route_board_monotonic,
                network.stop_route_offsets,
                network.stop_routes,
                network.trip_cost_factors,
                DEFAULT_RAPTOR_TRANSFER_PENALTY_SECONDS,
                network.transfer_offsets,
                network.transfer_neighbors,
                network.transfer_weights,
                start_idx,
                end_idx,
                departure_time,
                max_rounds=max_rounds,
            )
            segments = build_path(
                network.stop_times,
                network.trip_offsets,
                end_idx,
                earliest,
                pred_stop,
                pred_trip,
                pred_time,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            attempt_summaries.append(
                f"cap={max_rounds} used={int(rounds_used)} reached={int(reached_target)} marked={int(marked_count)} ms={elapsed_ms:.1f} segs={len(segments)}"
            )
            if segments:
                if max_rounds > DEFAULT_RAPTOR_ROUNDS_MIN:
                    logger.info(
                        "RAPTOR long-trip success start=%s end=%s attempts=%s",
                        start_idx,
                        end_idx,
                        " | ".join(attempt_summaries),
                    )
                return segments

        logger.info(
            "RAPTOR no-path start=%s end=%s attempts=%s",
            start_idx,
            end_idx,
            " | ".join(attempt_summaries),
        )
        return []
    if algorithm == "dijkstra":
        dist, pred_stop, pred_trip = run_dijkstra_fast(
            network.adj_offsets,
            network.adj_neighbors,
            network.adj_weights,
            network.adj_trip_ids,
            start_idx,
            end_idx,
            departure_time,
        )
        return build_path_dijkstra(
            end_idx,
            dist,
            pred_stop,
            pred_trip,
        )
    if algorithm == "astar":
        heuristic = np.zeros(network.adj_offsets.shape[0] - 1, dtype=np.int64)
        dist, pred_stop, pred_trip = run_astar_fast(
            network.adj_offsets,
            network.adj_neighbors,
            network.adj_weights,
            network.adj_trip_ids,
            start_idx,
            end_idx,
            departure_time,
            heuristic,
        )
        return build_path_dijkstra(
            end_idx,
            dist,
            pred_stop,
            pred_trip,
        )
    return None


def _get_incoming_neighbors(network: TransitNetwork) -> list[set[int]]:
    cached = getattr(network, "_incoming_neighbors", None)
    if cached is not None:
        return cached

    n_stops = int(network.adj_offsets.shape[0] - 1)
    incoming = [set() for _ in range(n_stops)]
    for source_stop in range(n_stops):
        row_start = int(network.adj_offsets[source_stop])
        row_end = int(network.adj_offsets[source_stop + 1])
        for idx in range(row_start, row_end):
            target_stop = int(network.adj_neighbors[idx])
            incoming[target_stop].add(source_stop)

    setattr(network, "_incoming_neighbors", incoming)
    return incoming


def _expand_start_indices(
    network: TransitNetwork,
    start_indices: list[int],
    max_hops: int = DEFAULT_START_EXPANSION_HOPS,
    max_stops: int = DEFAULT_START_EXPANSION_MAX_STOPS,
) -> list[int]:
    if not start_indices:
        return []

    incoming_neighbors = _get_incoming_neighbors(network)
    expanded = list(dict.fromkeys(start_indices))
    visited = set(expanded)
    frontier = expanded[:]

    for _ in range(max_hops):
        if not frontier or len(expanded) >= max_stops:
            break
        next_frontier = []
        for stop_id in frontier:
            row_start = int(network.adj_offsets[stop_id])
            row_end = int(network.adj_offsets[stop_id + 1])
            for idx in range(row_start, row_end):
                neighbor = int(network.adj_neighbors[idx])
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                expanded.append(neighbor)
                next_frontier.append(neighbor)
                if len(expanded) >= max_stops:
                    break
            if len(expanded) >= max_stops:
                break

            for neighbor in incoming_neighbors[stop_id]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                expanded.append(neighbor)
                next_frontier.append(neighbor)
                if len(expanded) >= max_stops:
                    break
            if len(expanded) >= max_stops:
                break

        frontier = next_frontier

    return expanded


def _find_best_segments_for_starts(
    network: TransitNetwork,
    algorithm: str,
    start_indices: list[int],
    end_idx: int,
    departure_time: int,
    start_penalties: dict[int, int] | None = None,
    max_transfers: int = DEFAULT_MAX_TRANSFERS,
):
    if len(start_indices) == 1:
        only_start = start_indices[0]
        segments = _compute_segments(network, algorithm, only_start, end_idx, departure_time)
        if not segments:
            return [], None, None
        penalty = 0 if start_penalties is None else int(start_penalties.get(only_start, 0))
        score = _score_segments_with_transfer_penalty(
            segments,
            int(segments[-1][2]) + penalty,
            max_transfers=max_transfers,
        )
        if score >= int(2**62 - 1):
            return [], None, None
        return segments, only_start, score

    with ThreadPoolExecutor(max_workers=min(len(start_indices), 16)) as pool:
        futures = [
            pool.submit(
                _compute_segments,
                network,
                algorithm,
                start_idx,
                end_idx,
                departure_time,
            )
            for start_idx in start_indices
        ]
        results = [future.result() for future in futures]

    best_segments = []
    best_start = None
    best_score = None
    for start_idx, result in zip(start_indices, results):
        if not result:
            continue
        penalty = 0 if start_penalties is None else int(start_penalties.get(start_idx, 0))
        score = _score_segments_with_transfer_penalty(
            result,
            int(result[-1][2]) + penalty,
            max_transfers=max_transfers,
        )
        if score >= int(2**62 - 1):
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_segments = result
            best_start = int(start_idx)
    return best_segments, best_start, best_score


def _best_end_from_earliest(
    earliest,
    end_indices: list[int],
    end_penalties: dict[int, int] | None = None,
):
    inf = np.int64(2**62)
    best_end = None
    best_score = None
    for end_idx in end_indices:
        arrival = int(earliest[int(end_idx)])
        if arrival >= inf:
            continue
        egress_penalty = 0 if end_penalties is None else int(end_penalties.get(int(end_idx), 0))
        score = arrival + egress_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_end = int(end_idx)
    return best_end, best_score


def _find_best_segments_for_od_candidates_raptor(
    network: TransitNetwork,
    start_indices: list[int],
    end_indices: list[int],
    departure_time: int,
    start_penalties: dict[int, int] | None = None,
    end_penalties: dict[int, int] | None = None,
    max_transfers: int = DEFAULT_MAX_TRANSFERS,
):
    budget_caps = list(_raptor_round_budgets(network))

    def compute_for_start(start_idx: int):
        attempt_summaries = []
        attempt_records = []
        max_rounds_reached = False
        rounds_used_max = 0
        for max_rounds in budget_caps:
            started = time.perf_counter()
            earliest, pred_stop, pred_trip, pred_time, rounds_used, marked_count, _ = run_raptor_with_stats(
                network.stop_times,
                network.trip_offsets,
                network.route_stop_offsets,
                network.route_stops,
                network.route_trip_offsets,
                network.route_trips,
                network.route_board_offsets,
                network.route_board_times,
                network.route_board_monotonic,
                network.stop_route_offsets,
                network.stop_routes,
                network.trip_cost_factors,
                DEFAULT_RAPTOR_TRANSFER_PENALTY_SECONDS,
                network.transfer_offsets,
                network.transfer_neighbors,
                network.transfer_weights,
                int(start_idx),
                -1,
                departure_time,
                max_rounds=max_rounds,
            )
            best_end, best_end_score = _best_end_from_earliest(earliest, end_indices, end_penalties)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            rounds_value = int(rounds_used)
            rounds_used_max = max(rounds_used_max, rounds_value)
            if rounds_value >= int(max_rounds):
                max_rounds_reached = True
            attempt_summaries.append(
                f"cap={max_rounds} used={rounds_value} marked={int(marked_count)} ms={elapsed_ms:.1f} best_end={best_end}"
            )
            attempt_records.append(
                {
                    "cap": int(max_rounds),
                    "rounds_used": rounds_value,
                    "marked_count": int(marked_count),
                    "best_end_idx": None if best_end is None else int(best_end),
                    "elapsed_ms": round(float(elapsed_ms), 3),
                }
            )

            if best_end is None:
                continue

            segments = build_path(
                network.stop_times,
                network.trip_offsets,
                best_end,
                earliest,
                pred_stop,
                pred_trip,
                pred_time,
            )
            if not segments:
                continue

            start_penalty = 0 if start_penalties is None else int(start_penalties.get(int(start_idx), 0))
            total_score = _score_segments_with_transfer_penalty(
                segments,
                int(best_end_score) + start_penalty,
                max_transfers=max_transfers,
            )
            if total_score >= int(2**62 - 1):
                continue
            logger.info(
                "RAPTOR multi-end success start=%s end=%s attempts=%s",
                start_idx,
                best_end,
                " | ".join(attempt_summaries),
            )
            return (
                segments,
                int(start_idx),
                best_end,
                total_score,
                {
                    "success": True,
                    "start_idx": int(start_idx),
                    "end_idx": int(best_end),
                    "start_candidates": int(len(start_indices)),
                    "end_candidates": int(len(end_indices)),
                    "attempts": attempt_records,
                    "max_rounds_reached": bool(max_rounds_reached),
                    "rounds_used_max": int(rounds_used_max),
                },
            )

        logger.info("RAPTOR multi-end no-path start=%s attempts=%s", start_idx, " | ".join(attempt_summaries))
        return (
            [],
            None,
            None,
            None,
            {
                "success": False,
                "start_idx": int(start_idx),
                "end_idx": None,
                "start_candidates": int(len(start_indices)),
                "end_candidates": int(len(end_indices)),
                "attempts": attempt_records,
                "max_rounds_reached": bool(max_rounds_reached),
                "rounds_used_max": int(rounds_used_max),
            },
        )

    if len(start_indices) == 1:
        results = [compute_for_start(start_indices[0])]
    elif not numba_enabled():
        results = [compute_for_start(start_idx) for start_idx in start_indices]
    else:
        with ThreadPoolExecutor(max_workers=min(len(start_indices), 16)) as pool:
            futures = [pool.submit(compute_for_start, start_idx) for start_idx in start_indices]
            results = [future.result() for future in futures]

    best_segments = []
    best_start = None
    best_end = None
    best_score = None
    diagnostics_runs = []
    max_rounds_reached = False
    rounds_used_max = 0
    for segments, start_idx, end_idx, score, diag in results:
        if diag:
            diagnostics_runs.append(diag)
            max_rounds_reached = bool(max_rounds_reached or diag.get("max_rounds_reached", False))
            rounds_used_max = max(rounds_used_max, int(diag.get("rounds_used_max", 0)))
        if not segments or start_idx is None or end_idx is None or score is None:
            continue
        if best_score is None or score < best_score:
            best_score = int(score)
            best_segments = segments
            best_start = int(start_idx)
            best_end = int(end_idx)

    diagnostics = {
        "runs": diagnostics_runs,
        "attempt_caps": [int(value) for value in budget_caps],
        "max_rounds_reached": bool(max_rounds_reached),
        "rounds_used_max": int(rounds_used_max),
    }
    return best_segments, best_start, best_end, best_score, diagnostics


def _find_best_segments_for_od_candidates(
    network: TransitNetwork,
    algorithm: str,
    start_indices: list[int],
    end_indices: list[int],
    departure_time: int,
    start_penalties: dict[int, int] | None = None,
    end_penalties: dict[int, int] | None = None,
    max_transfers: int = DEFAULT_MAX_TRANSFERS,
):
    if algorithm == "raptor":
        return _find_best_segments_for_od_candidates_raptor(
            network,
            start_indices,
            end_indices,
            departure_time,
            start_penalties,
            end_penalties,
            max_transfers,
        )

    best_segments = []
    best_start = None
    best_end = None
    best_score = None

    for end_idx in end_indices:
        segments, start_idx, start_score = _find_best_segments_for_starts(
            network,
            algorithm,
            start_indices,
            int(end_idx),
            departure_time,
            start_penalties,
            max_transfers,
        )
        if not segments or start_idx is None or start_score is None:
            continue

        egress_penalty = 0 if end_penalties is None else int(end_penalties.get(int(end_idx), 0))
        total_score = int(start_score) + egress_penalty
        if best_score is None or total_score < best_score:
            best_score = total_score
            best_segments = segments
            best_start = int(start_idx)
            best_end = int(end_idx)

    return best_segments, best_start, best_end, best_score, None


def _build_option_response(
    network: TransitNetwork,
    algorithm: str,
    start_indices: list[int],
    end_indices: list[int],
    departure_time: int,
    start_penalties: dict[int, int] | None = None,
    end_penalties: dict[int, int] | None = None,
    origin_anchor: dict[str, Any] | None = None,
    destination_anchor: dict[str, Any] | None = None,
    max_transfers: int = DEFAULT_MAX_TRANSFERS,
) -> dict[str, Any]:
    algorithms = _algorithm_sequence(algorithm)
    station_whitelist, station_backbone_path = _compute_station_backbone(
        network,
        start_indices,
        end_indices,
        max_transfers,
    )

    segments = []
    best_start = None
    best_end = None
    resolved_algorithm = algorithm
    raptor_diagnostics = None
    for selected_algorithm in algorithms:
        segments, best_start, best_end, _, diagnostics = _find_best_segments_for_od_candidates(
            network,
            selected_algorithm,
            start_indices,
            end_indices,
            departure_time,
            start_penalties,
            end_penalties,
            max_transfers,
        )
        if segments and algorithm == "raptor":
            if not _segments_follow_station_backbone(network, segments, station_whitelist):
                segments = []
        if selected_algorithm == "raptor":
            raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, diagnostics)
            if raptor_diagnostics is not None and station_backbone_path:
                raptor_diagnostics["station_backbone_path"] = [int(value) for value in station_backbone_path]
                raptor_diagnostics["station_backbone_whitelist_size"] = int(len(station_whitelist or []))

            if not segments:
                destination_tiers = _expand_candidates_tiered(
                    network,
                    end_indices,
                    destination_anchor,
                    "destination",
                )
                for tier_idx, tier_end_indices in enumerate(destination_tiers[1:], start=1):
                    segments, best_start, best_end, _, tier_diagnostics = _find_best_segments_for_od_candidates(
                        network,
                        selected_algorithm,
                        start_indices,
                        tier_end_indices,
                        departure_time,
                        start_penalties,
                        end_penalties,
                        max_transfers,
                    )
                    raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, tier_diagnostics)
                    if segments and not _segments_follow_station_backbone(network, segments, station_whitelist):
                        segments = []
                    if segments:
                        logger.info(
                            "RAPTOR adaptive destination expansion hit tier=%s end_candidates=%s",
                            tier_idx,
                            len(tier_end_indices),
                        )
                        break

            if not segments:
                origin_tiers = _expand_candidates_tiered(
                    network,
                    start_indices,
                    origin_anchor,
                    "origin",
                )
                for tier_idx, tier_start_indices in enumerate(origin_tiers[1:], start=1):
                    segments, best_start, best_end, _, tier_diagnostics = _find_best_segments_for_od_candidates(
                        network,
                        selected_algorithm,
                        tier_start_indices,
                        end_indices,
                        departure_time,
                        start_penalties,
                        end_penalties,
                        max_transfers,
                    )
                    raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, tier_diagnostics)
                    if segments and not _segments_follow_station_backbone(network, segments, station_whitelist):
                        segments = []
                    if segments:
                        logger.info(
                            "RAPTOR adaptive origin expansion hit tier=%s start_candidates=%s",
                            tier_idx,
                            len(tier_start_indices),
                        )
                        break

            if raptor_diagnostics is not None:
                start_counts, end_counts = _collect_candidate_counts(raptor_diagnostics)
                raptor_diagnostics["attempted_start_candidates"] = sorted(start_counts)
                raptor_diagnostics["attempted_end_candidates"] = sorted(end_counts)
                raptor_diagnostics["start_indices_snapshot"] = _snapshot_indices(start_indices)
                raptor_diagnostics["end_indices_snapshot"] = _snapshot_indices(end_indices)
                if not segments:
                    raptor_diagnostics["no_path_reason"] = _build_no_path_reason(
                        raptor_diagnostics,
                        start_counts,
                        end_counts,
                    )

        if segments:
            resolved_algorithm = selected_algorithm
            break

    if segments is None:
        return {}
    option_trip_profile = _summarize_option_trip_profile(network, segments)
    return {
        "departure_time": int(departure_time),
        "resolver_algorithm": resolved_algorithm,
        "fallback_used": resolved_algorithm != algorithm,
        "transfers": _count_transfers(segments),
        "max_transfers": int(max_transfers),
        "duration_seconds": int(segments[-1][2] - departure_time) if segments else None,
        "start_stop_id": network.stop_ids[best_start] if best_start is not None else None,
        "access_walk_seconds": int(start_penalties.get(best_start, 0)) if (best_start is not None and start_penalties) else 0,
        "end_stop_id": network.stop_ids[best_end] if best_end is not None else None,
        "egress_walk_seconds": int(end_penalties.get(best_end, 0)) if (best_end is not None and end_penalties) else 0,
        "segments": [
            _segment_payload(network, trip_id, stop_id, arrival_time)
            for trip_id, stop_id, arrival_time in segments
        ],
        "itinerary_trip_profile": option_trip_profile,
        "raptor_diagnostics": raptor_diagnostics if algorithm == "raptor" else None,
    }


def build_multi_departure_response(
    network: TransitNetwork,
    algorithm: str,
    start_indices: list[int],
    end_idx: int | list[int],
    departure_time: int,
    offset_minutes: tuple[int, ...] = DEFAULT_OFFSET_MINUTES,
    start_penalties: dict[int, int] | None = None,
    end_penalties: dict[int, int] | None = None,
    metadata: dict[str, Any] | None = None,
    max_transfers: int = DEFAULT_MAX_TRANSFERS,
) -> dict[str, Any]:
    if isinstance(end_idx, list):
        end_indices = [int(value) for value in end_idx]
    else:
        end_indices = [int(end_idx)]

    origin_anchor = metadata.get("origin") if isinstance(metadata, dict) else None
    destination_anchor = metadata.get("destination") if isinstance(metadata, dict) else None

    options: list[dict[str, Any]] = []

    if offset_minutes:
        options = [None] * len(offset_minutes)
        use_parallel_offsets = not (algorithm == "raptor" and not numba_enabled())
        if use_parallel_offsets:
            with ThreadPoolExecutor(max_workers=min(len(offset_minutes), 8)) as pool:
                futures = {
                    pool.submit(
                        _build_option_response,
                        network,
                        algorithm,
                        start_indices,
                        end_indices,
                        departure_time + minutes * 60,
                        start_penalties,
                        end_penalties,
                        origin_anchor,
                        destination_anchor,
                        max_transfers,
                    ): idx
                    for idx, minutes in enumerate(offset_minutes)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    options[idx] = future.result()
        else:
            for idx, minutes in enumerate(offset_minutes):
                options[idx] = _build_option_response(
                    network,
                    algorithm,
                    start_indices,
                    end_indices,
                    departure_time + minutes * 60,
                    start_penalties,
                    end_penalties,
                    origin_anchor,
                    destination_anchor,
                    max_transfers,
                )
    else:
        current_departure = int(departure_time)
        next_option_started = time.perf_counter()
        next_option_evals = 0
        current_option = _build_option_response(
            network,
            algorithm,
            start_indices,
            end_indices,
            current_departure,
            start_penalties,
            end_penalties,
            origin_anchor,
            destination_anchor,
            max_transfers,
        )
        options.append(current_option)

        max_lookahead_seconds = int(DEFAULT_NEXT_OPTION_SEARCH_WINDOW_MINUTES * 60)
        step_seconds = max(60, int(DEFAULT_NEXT_OPTION_STEP_SECONDS))
        max_step_seconds = max(step_seconds, int(DEFAULT_NEXT_OPTION_MAX_STEP_SECONDS))
        while len(options) < int(DEFAULT_OPTION_COUNT):
            previous_signature = _first_transit_trip_signature(options[-1])
            previous_departure = int(options[-1].get("departure_time", current_departure))
            search_departure = previous_departure + step_seconds
            search_limit = previous_departure + max_lookahead_seconds
            found_next = False
            search_step_seconds = step_seconds
            same_signature_streak = 0

            while search_departure <= search_limit:
                if next_option_evals >= int(DEFAULT_NEXT_OPTION_MAX_EVALS):
                    break
                if (time.perf_counter() - next_option_started) >= float(DEFAULT_NEXT_OPTION_MAX_WALL_SECONDS):
                    break
                candidate = _build_option_response(
                    network,
                    algorithm,
                    start_indices,
                    end_indices,
                    search_departure,
                    start_penalties,
                    end_penalties,
                    origin_anchor,
                    destination_anchor,
                    max_transfers,
                )
                next_option_evals += 1
                candidate_signature = _first_transit_trip_signature(candidate)
                if candidate.get("segments") and candidate_signature != previous_signature:
                    options.append(candidate)
                    current_departure = search_departure
                    found_next = True
                    break
                same_signature_streak += 1
                if same_signature_streak >= 1:
                    search_step_seconds = min(max_step_seconds, search_step_seconds * 2)
                search_departure += search_step_seconds

            if not found_next:
                if next_option_evals >= int(DEFAULT_NEXT_OPTION_MAX_EVALS):
                    logger.info(
                        "Dynamic next-option search stopped by evaluation budget evals=%s options=%s",
                        next_option_evals,
                        len(options),
                    )
                elif (time.perf_counter() - next_option_started) >= float(DEFAULT_NEXT_OPTION_MAX_WALL_SECONDS):
                    logger.info(
                        "Dynamic next-option search stopped by wall-time budget elapsed=%.2fs options=%s evals=%s",
                        (time.perf_counter() - next_option_started),
                        len(options),
                        next_option_evals,
                    )
                break

    base = options[0] if options else {
        "departure_time": int(departure_time),
        "transfers": 0,
        "duration_seconds": None,
        "segments": [],
    }
    network_trip_profile = summarize_trip_profiles(network)
    return {
        "algorithm": algorithm,
        "candidate_start_count": len(start_indices),
        "candidate_end_count": len(end_indices),
        "max_transfers": int(max_transfers),
        "network_trip_profile": network_trip_profile,
        **(metadata or {}),
        "resolver_algorithm": base.get("resolver_algorithm", algorithm),
        "fallback_used": bool(base.get("fallback_used", False)),
        "transfers": base.get("transfers"),
        "duration_seconds": base.get("duration_seconds"),
        "start_stop_id": base.get("start_stop_id"),
        "access_walk_seconds": base.get("access_walk_seconds"),
        "end_stop_id": base.get("end_stop_id"),
        "egress_walk_seconds": base.get("egress_walk_seconds"),
        "segments": base.get("segments"),
        "raptor_diagnostics": base.get("raptor_diagnostics"),
        "options": options,
    }


class PathRequestHandler(BaseHTTPRequestHandler):
    network: TransitNetwork | None = None

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler signature
        if self.path != "/path":
            self._send_json(404, {"error": "not_found"})
            return

        payload = self._read_json()
        if payload is None:
            return

        start_stop_id = payload.get("start_stop_id")
        start_stop_ids = payload.get("start_stop_ids")
        origin = payload.get("origin")
        destination = payload.get("destination")
        end_stop_id = payload.get("end_stop_id")
        max_transfers_raw = payload.get("max_transfers")
        departure_raw = payload.get("departure_time")
        algorithm = payload.get("algorithm", "raptor")
        offset_minutes_payload = payload.get("offset_minutes")

        normalized_start_stop_ids = []
        if isinstance(start_stop_ids, list):
            normalized_start_stop_ids = [
                stop_id.strip()
                for stop_id in start_stop_ids
                if isinstance(stop_id, str) and stop_id.strip()
            ]
        if not normalized_start_stop_ids and isinstance(start_stop_id, str) and start_stop_id.strip():
            normalized_start_stop_ids = [start_stop_id.strip()]

        if not normalized_start_stop_ids and origin is None:
            self._send_json(400, {"error": "invalid_stop_id"})
            return

        if destination is None and (not isinstance(end_stop_id, str) or not end_stop_id.strip()):
            self._send_json(400, {"error": "invalid_stop_id"})
            return

        if not isinstance(algorithm, str):
            self._send_json(400, {"error": "invalid_algorithm"})
            return
        algorithm = algorithm.strip().lower()

        departure_time = _departure_to_seconds(departure_raw)
        if departure_time is None:
            self._send_json(400, {"error": "invalid_departure_time"})
            return

        max_transfers = DEFAULT_MAX_TRANSFERS
        if max_transfers_raw is not None:
            if isinstance(max_transfers_raw, (int, float)):
                max_transfers = int(max_transfers_raw)
            elif isinstance(max_transfers_raw, str) and max_transfers_raw.strip().lstrip("-").isdigit():
                max_transfers = int(max_transfers_raw.strip())
            else:
                self._send_json(400, {"error": "invalid_max_transfers"})
                return
        if max_transfers < 0:
            self._send_json(400, {"error": "invalid_max_transfers"})
            return

        if self.network is None:
            self._send_json(500, {"error": "network_not_loaded"})
            return

        metadata = None
        start_penalties = None
        end_penalties = None
        if origin is not None:
            start_indices, start_penalties, origin_result = _select_starts_from_origin(self.network, origin)
            if not start_indices:
                error_code = origin_result if isinstance(origin_result, str) else "invalid_origin"
                self._send_json(400, {"error": error_code})
                return
            if isinstance(origin_result, dict):
                metadata = origin_result
        else:
            start_indices = []
            for stop_id in normalized_start_stop_ids:
                stop_index = self.network.stop_id_index.get(stop_id)
                if stop_index is None:
                    self._send_json(404, {"error": "unknown_stop_id"})
                    return
                start_indices.append(stop_index)

            start_indices = list(dict.fromkeys(start_indices))

        if destination is not None:
            end_indices, end_penalties, destination_result = _select_ends_from_destination(self.network, destination)
            if not end_indices:
                error_code = destination_result if isinstance(destination_result, str) else "invalid_destination"
                self._send_json(400, {"error": error_code})
                return
            if isinstance(destination_result, dict):
                metadata = {**(metadata or {}), **destination_result}
        else:
            resolved_end_idx = self.network.stop_id_index.get(end_stop_id)
            if resolved_end_idx is None:
                self._send_json(404, {"error": "unknown_stop_id"})
                return
            end_indices = [resolved_end_idx]

        if algorithm not in {"raptor", "dijkstra", "astar"}:
            self._send_json(400, {"error": "unsupported_algorithm"})
            return

        offset_minutes = DEFAULT_OFFSET_MINUTES
        if isinstance(offset_minutes_payload, list):
            normalized_offsets = []
            for value in offset_minutes_payload:
                if isinstance(value, (int, float)):
                    normalized_offsets.append(int(value))
                elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
                    normalized_offsets.append(int(value.strip()))
                else:
                    self._send_json(400, {"error": "invalid_offset_minutes"})
                    return
            if not normalized_offsets:
                self._send_json(400, {"error": "invalid_offset_minutes"})
                return
            offset_minutes = tuple(normalized_offsets)

        response = build_multi_departure_response(
            self.network,
            algorithm,
            start_indices,
            end_indices,
            departure_time,
            offset_minutes=offset_minutes,
            start_penalties=start_penalties,
            end_penalties=end_penalties,
            metadata=metadata,
            max_transfers=max_transfers,
        )
        logger.info(
            "HTTP /path algorithm=%s starts=%s ends=%s departure=%s options=%s",
            algorithm,
            len(start_indices),
            len(end_indices),
            departure_time,
            len(response.get("options", [])),
        )
        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match base signature
        logger.info("%s - %s", self.address_string(), format % args)

    def _read_json(self) -> dict[str, Any] | None:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send_json(400, {"error": "missing_content_length"})
            return None
        try:
            length = int(length_header)
        except ValueError:
            self._send_json(400, {"error": "invalid_content_length"})
            return None

        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid_json"})
            return None
        if not isinstance(data, dict):
            self._send_json(400, {"error": "invalid_payload"})
            return None
        return data

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    setup_logging()
    server = ThreadingHTTPServer((host, port), PathRequestHandler)
    PathRequestHandler.network = load_network()
    trip_profile = summarize_trip_profiles(PathRequestHandler.network)
    logger.info(
        "Network trip profile trips=%s unknown_route_type_share=%s top_route_types=%s top_factors=%s",
        trip_profile.get("trip_count"),
        trip_profile.get("route_type_unknown_share"),
        trip_profile.get("route_type_counts_top"),
        trip_profile.get("factor_counts_top"),
    )
    logger.info("RAPTOR numba_jit_enabled=%s", numba_enabled())
    try:
        _warmup_raptor(PathRequestHandler.network)
    except Exception as exc:
        logger.warning("RAPTOR warmup skipped due to error: %s", exc)
    logger.info("HTTP server listening on %s:%s", host, port)
    return server


def main() -> None:
    server = serve()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down HTTP server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

