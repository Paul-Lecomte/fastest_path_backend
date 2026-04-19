# This module implements a simple HTTP server that provides a pathfinding API for transit networks.
from __future__ import annotations

import json
import logging
import time
import heapq
import math
import difflib
import re
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from datetime import datetime
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

# Cache repeated transfer walking paths across options/requests.
_WALKING_PATH_CACHE_MAX = 20000
_walking_path_cache: dict[tuple[int, int], list[tuple[float, float]] | None] = {}
_osrm_path_cache: dict[tuple[float, float, float, float], list[tuple[float, float]] | None] = {}
_osrm_failure_count = 0
_osrm_disabled_until = 0.0

DEFAULT_WALKING_PATH_SPEED_MPS = 1.4
DEFAULT_WALKING_PATH_BUDGET_FACTOR = 3.0
DEFAULT_WALKING_PATH_BUDGET_SLACK_SECONDS = 600
DEFAULT_WALKING_PATH_BUDGET_MAX_SECONDS = 3600
DEFAULT_OSRM_TIMEOUT_SECONDS = 1.2
DEFAULT_OSRM_FAILURE_BACKOFF_SECONDS = 60.0
DEFAULT_OSRM_FAILURE_THRESHOLD = 3

DEFAULT_OFFSET_MINUTES: tuple[int, ...] = ()
DEFAULT_OPTION_COUNT = 3
DEFAULT_NEXT_OPTION_SEARCH_WINDOW_MINUTES = 90
DEFAULT_NEXT_OPTION_STEP_SECONDS = 300
DEFAULT_NEXT_OPTION_MAX_STEP_SECONDS = 1800
DEFAULT_NEXT_OPTION_MAX_EVALS = 24
DEFAULT_NEXT_OPTION_MAX_WALL_SECONDS = 20.0
DEFAULT_START_EXPANSION_HOPS = 2
DEFAULT_START_EXPANSION_MAX_STOPS = 256
DEFAULT_ORIGIN_RADIUS_M = 1000.0
DEFAULT_ORIGIN_MAX_CANDIDATES = 12
DEFAULT_ORIGIN_SEED_CANDIDATES = 2
DEFAULT_DESTINATION_SEED_CANDIDATES = 6
DEFAULT_WALK_SPEED_MPS = 1.4
DEFAULT_WALK_TIME_MULTIPLIER = 1.0
UNIX_TIMESTAMP_MIN_SECONDS = 946684800
DEFAULT_RAPTOR_ROUNDS_MIN = 8
DEFAULT_RAPTOR_ROUNDS_MAX = 64
DEFAULT_ADAPTIVE_EXPANSION_MAX_TIERS = 4
DEFAULT_RAPTOR_TRANSFER_PENALTY_SECONDS = 3600
DEFAULT_OPTION_TRANSFER_PENALTY_SECONDS = 3600
DEFAULT_MAX_TRANSFERS = 4
DEFAULT_RAPTOR_BOARD_SCAN_LIMIT = 6
DEFAULT_RAPTOR_RESCUE_BOARD_SCAN_LIMIT = 18
DEFAULT_ACCESS_WALK_SCORE_CAP_SECONDS = 900
DEFAULT_EGRESS_WALK_SCORE_CAP_SECONDS = 900
DEFAULT_WALK_SCORE_WEIGHT_NUM = 1
DEFAULT_WALK_SCORE_WEIGHT_DEN = 2
DEFAULT_WALK_SEGMENT_PENALTY_SECONDS = 900
DEFAULT_NO_TRANSFER_PREFERENCE_SLACK_SECONDS = 480
DEFAULT_NO_TRANSFER_CLOSE_EGRESS_SECONDS = 720
DEFAULT_NO_TRANSFER_CLOSE_SLACK_SECONDS = 900
DEFAULT_LONG_DISTANCE_RESCUE_MIN_M = 35000.0
DEFAULT_LONG_DISTANCE_RESCUE_HUBS = 3
DEFAULT_LONG_DISTANCE_RESCUE_STOPS_PER_HUB = 6
DEFAULT_LONG_DISTANCE_RESCUE_EXTRA_TRANSFERS = 3
DEFAULT_LONG_DISTANCE_RESCUE_MAX_TRANSFERS = 8
DEFAULT_LONG_DISTANCE_BACKBONE_MAX_OFFPATH = 12
DEFAULT_GENERAL_RESCUE_MAX_CANDIDATES = 36


def _normalize_stop_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(part for part in normalized.split(" ") if part)


def _build_stop_lookup_cache(network: TransitNetwork) -> dict[str, Any]:
    cached = getattr(network, "_stop_lookup_cache", None)
    if isinstance(cached, dict):
        return cached

    id_exact: dict[str, int] = {}
    normalized_labels: list[tuple[str, int, str]] = []
    exact_norm_index: dict[str, int] = {}

    for stop_idx, stop_id in enumerate(network.stop_ids):
        stop_text = str(stop_id)
        id_exact[stop_text.lower()] = int(stop_idx)
        norm = _normalize_stop_text(stop_text)
        if norm:
            normalized_labels.append((norm, int(stop_idx), stop_text))
            if norm not in exact_norm_index:
                exact_norm_index[norm] = int(stop_idx)

    stop_names = getattr(network, "stop_names", None)
    if isinstance(stop_names, (list, np.ndarray)):
        for stop_idx in range(min(len(stop_names), len(network.stop_ids))):
            name = stop_names[stop_idx]
            if name is None:
                continue
            name_text = str(name)
            norm = _normalize_stop_text(name_text)
            if not norm:
                continue
            normalized_labels.append((norm, int(stop_idx), name_text))
            if norm not in exact_norm_index:
                exact_norm_index[norm] = int(stop_idx)

    cache = {
        "id_exact": id_exact,
        "normalized_labels": normalized_labels,
        "exact_norm_index": exact_norm_index,
    }
    setattr(network, "_stop_lookup_cache", cache)
    return cache


def _resolve_stop_query_to_index(network: TransitNetwork, query: str) -> tuple[int | None, dict[str, Any] | None]:
    raw = str(query).strip()
    if not raw:
        return None, None

    direct = network.stop_id_index.get(raw)
    if direct is not None:
        return int(direct), {"match_type": "exact_stop_id", "query": raw, "matched": raw}

    cache = _build_stop_lookup_cache(network)
    id_exact = cache["id_exact"]
    lower = raw.lower()
    if lower in id_exact:
        idx = int(id_exact[lower])
        return idx, {"match_type": "casefold_stop_id", "query": raw, "matched": network.stop_ids[idx]}

    norm_query = _normalize_stop_text(raw)
    if not norm_query:
        return None, None

    exact_norm_index = cache["exact_norm_index"]
    if norm_query in exact_norm_index:
        idx = int(exact_norm_index[norm_query])
        return idx, {"match_type": "normalized_exact", "query": raw, "matched": network.stop_ids[idx]}

    query_tokens = norm_query.split(" ")
    candidates: list[tuple[float, int, str]] = []
    for candidate_norm, stop_idx, candidate_label in cache["normalized_labels"]:
        if norm_query in candidate_norm:
            score = 0.97
        else:
            candidate_tokens = candidate_norm.split(" ")
            all_tokens_match = True
            for token in query_tokens:
                if not any(part.startswith(token) for part in candidate_tokens):
                    all_tokens_match = False
                    break
            if all_tokens_match:
                score = 0.95
            else:
                score = float(difflib.SequenceMatcher(None, norm_query, candidate_norm).ratio())
                if score < 0.80:
                    continue
        candidates.append((score, int(stop_idx), str(candidate_label)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
    best_score, best_idx, best_label = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else -1.0
    if best_score < 0.86:
        return None, None
    if second_score >= 0.86 and (best_score - second_score) < 0.02 and int(candidates[1][1]) != int(best_idx):
        return None, None

    return int(best_idx), {
        "match_type": "fuzzy",
        "query": raw,
        "matched": best_label,
        "confidence": round(float(best_score), 4),
    }


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
    walk_time_multiplier: float,
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
        walk_seconds = float(distance_m / max(0.5, float(walk_speed_mps))) * float(walk_time_multiplier)
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


def _count_walk_segments(segments) -> int:
    if not segments:
        return 0
    return int(sum(1 for trip_id, _, _ in segments if int(trip_id) < 0))


def _score_segments_with_transfer_penalty(
    segments,
    base_score: int,
    transfer_penalty_seconds: int = DEFAULT_OPTION_TRANSFER_PENALTY_SECONDS,
    walk_segment_penalty_seconds: int = DEFAULT_WALK_SEGMENT_PENALTY_SECONDS,
    max_transfers: int = DEFAULT_MAX_TRANSFERS,
) -> int:
    transfer_count = int(_count_transfers(segments))
    if transfer_count > int(max_transfers):
        return int(2**62 - 1)
    walk_segment_count = int(_count_walk_segments(segments))
    return (
        int(base_score)
        + transfer_count * int(transfer_penalty_seconds)
        + walk_segment_count * int(walk_segment_penalty_seconds)
    )


def _walk_score_penalty(raw_seconds: int, cap_seconds: int) -> int:
    bounded_raw = max(0, int(raw_seconds))
    bounded_cap = max(0, int(cap_seconds))
    bounded = min(bounded_raw, bounded_cap)
    return int((bounded * int(DEFAULT_WALK_SCORE_WEIGHT_NUM)) // max(1, int(DEFAULT_WALK_SCORE_WEIGHT_DEN)))


def _is_no_transfer_preferred(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_transfers = int(candidate.get("transfers", 0))
    incumbent_transfers = int(incumbent.get("transfers", 0))
    if candidate_transfers != 0 or incumbent_transfers == 0:
        return False

    candidate_arrival = int(candidate.get("arrival_time", 2**62 - 1))
    incumbent_arrival = int(incumbent.get("arrival_time", 2**62 - 1))
    if candidate_arrival <= incumbent_arrival + int(DEFAULT_NO_TRANSFER_PREFERENCE_SLACK_SECONDS):
        return True

    candidate_egress = int(candidate.get("egress_walk_seconds", 2**62 - 1))
    if candidate_egress <= int(DEFAULT_NO_TRANSFER_CLOSE_EGRESS_SECONDS):
        return candidate_arrival <= incumbent_arrival + int(DEFAULT_NO_TRANSFER_CLOSE_SLACK_SECONDS)
    return False


def _choose_better_candidate(current: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate

    if _is_no_transfer_preferred(candidate, current):
        return candidate
    if _is_no_transfer_preferred(current, candidate):
        return current

    # Lexicographic preference for user comfort: fewer transfers first.
    candidate_transfers = int(candidate.get("transfers", 2**62 - 1))
    current_transfers = int(current.get("transfers", 2**62 - 1))
    if candidate_transfers != current_transfers:
        return candidate if candidate_transfers < current_transfers else current

    candidate_arrival = int(candidate.get("arrival_time", 2**62 - 1))
    current_arrival = int(current.get("arrival_time", 2**62 - 1))
    if candidate_arrival != current_arrival:
        return candidate if candidate_arrival < current_arrival else current

    candidate_walk_segments = int(candidate.get("walk_segment_count", 2**62 - 1))
    current_walk_segments = int(current.get("walk_segment_count", 2**62 - 1))
    if candidate_walk_segments != current_walk_segments:
        return candidate if candidate_walk_segments < current_walk_segments else current

    candidate_score = int(candidate.get("score", 2**62 - 1))
    current_score = int(current.get("score", 2**62 - 1))
    if candidate_score != current_score:
        return candidate if candidate_score < current_score else current

    candidate_egress = int(candidate.get("egress_walk_seconds", 2**62 - 1))
    current_egress = int(current.get("egress_walk_seconds", 2**62 - 1))
    if candidate_egress != current_egress:
        return candidate if candidate_egress < current_egress else current

    return current


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
    def _from_datetime_local(dt_value: datetime) -> int:
        local_dt = dt_value.astimezone() if dt_value.tzinfo is not None else dt_value
        return local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second

    def _from_unix_timestamp(timestamp_value: int) -> int:
        local_dt = datetime.fromtimestamp(timestamp_value)
        return local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second

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
        return _from_datetime_local(parsed)
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


def _haversine_distance_point_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    values = _haversine_distance_m(
        float(lat1),
        float(lon1),
        np.asarray([float(lat2)], dtype=np.float64),
        np.asarray([float(lon2)], dtype=np.float64),
    )
    if values.size == 0:
        return 0.0
    return float(values[0])


def _build_node_buckets(node_lats: np.ndarray, node_lons: np.ndarray, cell_deg: float) -> dict[tuple[int, int], list[int]]:
    import math

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx in range(node_lats.shape[0]):
        lat = float(node_lats[idx])
        lon = float(node_lons[idx])
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        row = int(math.floor(lat / cell_deg))
        col = int(math.floor(lon / cell_deg))
        key = (row, col)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(int(idx))
    return buckets


def _nearest_node_index_from_buckets(
    lat: float,
    lon: float,
    node_lats: np.ndarray,
    node_lons: np.ndarray,
    buckets: dict[tuple[int, int], list[int]],
    cell_deg: float,
    max_ring: int = 3,
) -> int:
    import math

    row = int(math.floor(float(lat) / cell_deg))
    col = int(math.floor(float(lon) / cell_deg))

    best_idx = -1
    best_dist = float("inf")
    for ring in range(max_ring + 1):
        found_any = False
        for d_row in range(-ring, ring + 1):
            for d_col in range(-ring, ring + 1):
                for idx in buckets.get((row + d_row, col + d_col), []):
                    found_any = True
                    dist = _haversine_distance_point_m(
                        float(lat),
                        float(lon),
                        float(node_lats[idx]),
                        float(node_lons[idx]),
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = int(idx)
        if found_any and best_idx >= 0:
            return best_idx

    # Rare fallback when local buckets are empty.
    if node_lats.size == 0:
        return -1
    d_lat = node_lats - float(lat)
    d_lon = node_lons - float(lon)
    approx = d_lat * d_lat + d_lon * d_lon
    return int(np.argmin(approx))


def _ensure_runtime_walking_graph(network: TransitNetwork) -> bool:
    walking_node_ids = getattr(network, "walking_node_ids", np.zeros(0, dtype=object))
    walking_adj_offsets = getattr(network, "walking_adj_offsets", np.zeros(1, dtype=np.int64))
    stop_to_walking_node_idx = getattr(network, "stop_to_walking_node_idx", np.zeros(0, dtype=np.int32))

    has_graph = isinstance(walking_node_ids, np.ndarray) and walking_node_ids.size > 0 and isinstance(
        walking_adj_offsets, np.ndarray
    ) and walking_adj_offsets.size > 1
    has_mapping = isinstance(stop_to_walking_node_idx, np.ndarray) and stop_to_walking_node_idx.shape[0] == len(
        network.stop_ids
    )
    if has_graph and has_mapping:
        return True

    cache_path = ".cache/osm_walking_graph_compact.npz"
    try:
        payload = np.load(cache_path, allow_pickle=True)
    except Exception:
        return False

    required = {
        "node_ids",
        "node_lats",
        "node_lons",
        "adj_offsets",
        "adj_neighbors",
        "adj_weights",
    }
    if not required.issubset(set(payload.files)):
        return False

    node_ids = np.asarray(payload["node_ids"], dtype=object)
    node_lats = np.asarray(payload["node_lats"], dtype=np.float64)
    node_lons = np.asarray(payload["node_lons"], dtype=np.float64)
    adj_offsets = np.asarray(payload["adj_offsets"], dtype=np.int64)
    adj_neighbors = np.asarray(payload["adj_neighbors"], dtype=np.int32)
    adj_weights = np.asarray(payload["adj_weights"], dtype=np.int64)

    if node_ids.size == 0 or adj_offsets.size <= 1:
        return False

    network.walking_node_ids = node_ids
    network.walking_node_lats = node_lats
    network.walking_node_lons = node_lons
    network.walking_adj_offsets = adj_offsets
    network.walking_adj_neighbors = adj_neighbors
    network.walking_adj_weights = adj_weights

    if not has_mapping:
        cell_deg = 0.01
        buckets = _build_node_buckets(node_lats, node_lons, cell_deg)
        mapping = np.full(len(network.stop_ids), -1, dtype=np.int32)
        for idx in range(len(network.stop_ids)):
            lat = float(network.stop_lats[idx])
            lon = float(network.stop_lons[idx])
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            mapping[idx] = int(
                _nearest_node_index_from_buckets(
                    lat,
                    lon,
                    node_lats,
                    node_lons,
                    buckets,
                    cell_deg,
                )
            )
        network.stop_to_walking_node_idx = mapping

    logger.info(
        "Loaded runtime OSM walking graph from %s nodes=%s edges=%s",
        cache_path,
        int(network.walking_node_ids.shape[0]),
        int(network.walking_adj_neighbors.shape[0]),
    )
    return True


def _find_walking_path_via_astar(
    network: TransitNetwork,
    from_stop_idx: int,
    to_stop_idx: int,
    max_search_seconds: int | None = None,
) -> list[tuple[float, float]] | None:
    """
    Use A* on the OSM walking graph to find a detailed path between two stops.
    Returns list of (lat, lon) tuples representing the walking path.
    If OSM data unavailable or A* fails, returns None.
    """
    _ensure_runtime_walking_graph(network)

    budget_key = -1 if max_search_seconds is None else int(max_search_seconds)
    cache_key = (int(from_stop_idx), int(to_stop_idx), budget_key)
    cached = _walking_path_cache.get(cache_key)
    if cache_key in _walking_path_cache:
        return cached

    walking_node_ids = getattr(network, "walking_node_ids", np.zeros(0, dtype=object))
    walking_node_lats = getattr(network, "walking_node_lats", np.zeros(0, dtype=np.float64))
    walking_node_lons = getattr(network, "walking_node_lons", np.zeros(0, dtype=np.float64))
    walking_adj_offsets = getattr(network, "walking_adj_offsets", np.zeros(1, dtype=np.int64))
    walking_adj_neighbors = getattr(network, "walking_adj_neighbors", np.zeros(0, dtype=np.int32))
    walking_adj_weights = getattr(network, "walking_adj_weights", np.zeros(0, dtype=np.int64))
    stop_to_walking_node_idx = getattr(network, "stop_to_walking_node_idx", np.zeros(0, dtype=np.int32))

    # Check if OSM walking graph is available
    if (
        walking_node_ids.size == 0
        or walking_adj_offsets.size <= 1
        or stop_to_walking_node_idx.size == 0
    ):
        return None
    
    # Convert stop indices to OSM node indices
    if from_stop_idx < 0 or from_stop_idx >= stop_to_walking_node_idx.shape[0]:
        return None
    if to_stop_idx < 0 or to_stop_idx >= stop_to_walking_node_idx.shape[0]:
        return None
    
    from_node_idx = int(stop_to_walking_node_idx[from_stop_idx])
    to_node_idx = int(stop_to_walking_node_idx[to_stop_idx])
    
    if from_node_idx < 0 or to_node_idx < 0:
        return None
    
    # Run bounded dynamic A* without full-graph heuristic allocation.
    try:
        src = int(from_node_idx)
        dst = int(to_node_idx)
        if src == dst:
            result = [
                (float(walking_node_lats[src]), float(walking_node_lons[src])),
                (float(walking_node_lats[dst]), float(walking_node_lons[dst])),
            ]
            if len(_walking_path_cache) >= _WALKING_PATH_CACHE_MAX:
                _walking_path_cache.pop(next(iter(_walking_path_cache)))
            _walking_path_cache[cache_key] = result
            return result

        budget = DEFAULT_WALKING_PATH_BUDGET_MAX_SECONDS if max_search_seconds is None else int(max_search_seconds)
        budget = max(60, min(DEFAULT_WALKING_PATH_BUDGET_MAX_SECONDS, budget))

        target_lat = float(walking_node_lats[dst])
        target_lon = float(walking_node_lons[dst])

        dist: dict[int, int] = {src: 0}
        pred: dict[int, int] = {src: -1}

        def _h_seconds(node_idx: int) -> int:
            d_m = _haversine_distance_point_m(
                float(walking_node_lats[node_idx]),
                float(walking_node_lons[node_idx]),
                target_lat,
                target_lon,
            )
            return int(d_m / DEFAULT_WALKING_PATH_SPEED_MPS)

        heap: list[tuple[int, int, int]] = [(_h_seconds(src), 0, src)]
        found = False

        while heap:
            _, g, u = heapq.heappop(heap)
            if g != dist.get(u):
                continue
            if g > budget:
                continue
            if u == dst:
                found = True
                break

            row_start = int(walking_adj_offsets[u])
            row_end = int(walking_adj_offsets[u + 1])
            for edge_idx in range(row_start, row_end):
                v = int(walking_adj_neighbors[edge_idx])
                alt = int(g + int(walking_adj_weights[edge_idx]))
                if alt > budget:
                    continue
                prev = dist.get(v)
                if prev is None or alt < prev:
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(heap, (alt + _h_seconds(v), alt, v))

        if not found:
            result = None
        else:
            path_nodes: list[tuple[float, float]] = []
            current_idx = dst
            while current_idx != -1:
                lat = float(walking_node_lats[current_idx])
                lon = float(walking_node_lons[current_idx])
                if np.isfinite(lat) and np.isfinite(lon):
                    path_nodes.append((lat, lon))
                current_idx = int(pred.get(current_idx, -1))
            path_nodes.reverse()
            result = path_nodes if len(path_nodes) > 0 else None

        if len(_walking_path_cache) >= _WALKING_PATH_CACHE_MAX:
            _walking_path_cache.pop(next(iter(_walking_path_cache)))
        _walking_path_cache[cache_key] = result
        return result
    except Exception as exc:
        logger.warning(f"A* pathfinding failed: {exc}")
        return None


def _find_walking_path_via_osrm(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    timeout_seconds: float = 3.0,
) -> list[tuple[float, float]] | None:
    """Fallback to OSRM walking route when local OSM graph is unavailable."""
    global _osrm_failure_count, _osrm_disabled_until

    now = time.time()
    if now < _osrm_disabled_until:
        return None

    cache_key = (
        round(float(from_lat), 6),
        round(float(from_lon), 6),
        round(float(to_lat), 6),
        round(float(to_lon), 6),
    )
    if cache_key in _osrm_path_cache:
        return _osrm_path_cache[cache_key]

    try:
        coords = f"{float(from_lon):.7f},{float(from_lat):.7f};{float(to_lon):.7f},{float(to_lat):.7f}"
        query = urllib.parse.urlencode(
            {
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
            }
        )
        url = f"https://router.project-osrm.org/route/v1/foot/{coords}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "fastest-path-backend/1.0"})
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if not isinstance(payload, dict) or payload.get("code") != "Ok":
            return None
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            return None
        geometry = routes[0].get("geometry", {})
        coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            return None

        result: list[tuple[float, float]] = []
        for item in coordinates:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            lon = float(item[0])
            lat = float(item[1])
            if np.isfinite(lat) and np.isfinite(lon):
                result.append((lat, lon))
        if len(result) < 2:
            _osrm_path_cache[cache_key] = None
            return None
        if len(_osrm_path_cache) >= _WALKING_PATH_CACHE_MAX:
            _osrm_path_cache.pop(next(iter(_osrm_path_cache)))
        _osrm_path_cache[cache_key] = result
        _osrm_failure_count = 0
        return result
    except Exception:
        _osrm_path_cache[cache_key] = None
        _osrm_failure_count += 1
        if _osrm_failure_count >= DEFAULT_OSRM_FAILURE_THRESHOLD:
            _osrm_disabled_until = time.time() + DEFAULT_OSRM_FAILURE_BACKOFF_SECONDS
        return None


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


def _build_anchor_walking_segment_payload(
    *,
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    from_stop_id: str,
    to_stop_id: str,
    arrival_time: int,
    walk_duration_seconds: int,
    segment_type: str,
) -> dict[str, Any] | None:
    if not (
        np.isfinite(float(from_lat))
        and np.isfinite(float(from_lon))
        and np.isfinite(float(to_lat))
        and np.isfinite(float(to_lon))
    ):
        return None

    duration = max(0, int(walk_duration_seconds))
    walking_path = _find_walking_path_via_osrm(
        float(from_lat),
        float(from_lon),
        float(to_lat),
        float(to_lon),
        timeout_seconds=DEFAULT_OSRM_TIMEOUT_SECONDS,
    )

    if walking_path is not None and len(walking_path) > 0:
        routing_method = "osrm"
        coordinates = [[float(lon), float(lat)] for lat, lon in walking_path]
        total_distance_m = 0.0
        for idx in range(len(walking_path) - 1):
            lat1, lon1 = walking_path[idx]
            lat2, lon2 = walking_path[idx + 1]
            total_distance_m += _haversine_distance_point_m(lat1, lon1, lat2, lon2)
    else:
        routing_method = "straight_line"
        coordinates = [
            [float(from_lon), float(from_lat)],
            [float(to_lon), float(to_lat)],
        ]
        total_distance_m = _haversine_distance_point_m(
            float(from_lat),
            float(from_lon),
            float(to_lat),
            float(to_lon),
        )

    payload: dict[str, Any] = {
        "trip_id": "TRANSFER",
        "stop_id": str(to_stop_id),
        "from_stop_id": str(from_stop_id),
        "arrival_time": int(arrival_time),
        "lat": float(to_lat),
        "lon": float(to_lon),
        "walk_duration_seconds": int(duration),
        "walking_segment_type": str(segment_type),
        "walking_geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "walk_distance_m": float(total_distance_m),
        "routing_method": routing_method,
    }
    if duration > 0 and total_distance_m > 0:
        walk_speed_kmh = (float(total_distance_m) / float(duration)) * 3.6
        payload["walking_speed_kmh"] = round(float(walk_speed_kmh), 2)
    return payload


def _build_segment_payloads(
    network: TransitNetwork,
    segments,
    start_stop_idx: int | None,
    departure_time: int,
    end_stop_idx: int | None = None,
    origin_anchor: dict[str, Any] | None = None,
    destination_anchor: dict[str, Any] | None = None,
    access_walk_seconds: int = 0,
    egress_walk_seconds: int = 0,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for idx, (trip_id, stop_id, arrival_time) in enumerate(segments):
        payload = _segment_payload(network, trip_id, stop_id, arrival_time)

        prev_stop_idx: int | None = None
        prev_time = int(departure_time)
        if idx > 0:
            prev_stop_idx = int(segments[idx - 1][1])
            prev_time = int(segments[idx - 1][2])
        elif start_stop_idx is not None:
            prev_stop_idx = int(start_stop_idx)

        if prev_stop_idx is not None:
            payload["from_stop_id"] = network.stop_ids[int(prev_stop_idx)]
            # Add from_stop_name for user friendliness
            if hasattr(network, 'stop_names') and network.stop_names is not None:
                from_name = network.stop_names[int(prev_stop_idx)] if int(prev_stop_idx) < len(network.stop_names) else None
                if from_name:
                    payload["from_stop_name"] = str(from_name)

        if int(trip_id) < 0 and prev_stop_idx is not None:
            to_lat, to_lon = _segment_coordinates(network, int(stop_id))
            from_lat, from_lon = _segment_coordinates(network, int(prev_stop_idx))
            walk_duration_seconds = max(0, int(arrival_time) - int(prev_time))
            payload["walk_duration_seconds"] = int(walk_duration_seconds)
            
            # Add to_stop_name for walking segments
            if hasattr(network, 'stop_names') and network.stop_names is not None:
                to_name = network.stop_names[int(stop_id)] if int(stop_id) < len(network.stop_names) else None
                if to_name:
                    payload["to_stop_name"] = str(to_name)
            
            # Determine walking segment type
            if idx == 0 or (idx > 0 and int(segments[idx-1][1]) == int(start_stop_idx or prev_stop_idx)):
                segment_type = "access"
            elif idx == len(segments) - 1:
                segment_type = "egress"
            else:
                segment_type = "transfer"
            payload["walking_segment_type"] = segment_type
            
            if (
                from_lat is not None
                and from_lon is not None
                and to_lat is not None
                and to_lon is not None
            ):
                # Try to get detailed path from OSM walking graph using A*
                search_budget = min(
                    DEFAULT_WALKING_PATH_BUDGET_MAX_SECONDS,
                    int(
                        max(
                            DEFAULT_WALKING_PATH_BUDGET_SLACK_SECONDS,
                            float(walk_duration_seconds) * DEFAULT_WALKING_PATH_BUDGET_FACTOR
                            + DEFAULT_WALKING_PATH_BUDGET_SLACK_SECONDS,
                        )
                    ),
                )
                walking_path = _find_walking_path_via_astar(
                    network,
                    int(prev_stop_idx),
                    int(stop_id),
                    max_search_seconds=int(search_budget),
                )
                
                routing_method = "osm"
                if walking_path is None or len(walking_path) == 0:
                    walking_path = _find_walking_path_via_osrm(
                        float(from_lat),
                        float(from_lon),
                        float(to_lat),
                        float(to_lon),
                        timeout_seconds=DEFAULT_OSRM_TIMEOUT_SECONDS,
                    )
                    routing_method = "osrm" if walking_path is not None and len(walking_path) > 0 else "straight_line"
                
                if walking_path is not None and len(walking_path) > 0:
                    # Use the detailed walking path with all intermediate nodes
                    coordinates = [[float(lon), float(lat)] for lat, lon in walking_path]
                    payload["walking_geometry"] = {
                        "type": "LineString",
                        "coordinates": coordinates,
                    }
                    # Calculate total distance from path
                    total_distance_m = 0.0
                    for i in range(len(walking_path) - 1):
                        lat1, lon1 = walking_path[i]
                        lat2, lon2 = walking_path[i + 1]
                        total_distance_m += _haversine_distance_point_m(lat1, lon1, lat2, lon2)
                    payload["walk_distance_m"] = float(total_distance_m)
                else:
                    # Fall back to straight line if OSM graph unavailable or A* fails
                    routing_method = "straight_line"
                    payload["walking_geometry"] = {
                        "type": "LineString",
                        "coordinates": [
                            [float(from_lon), float(from_lat)],
                            [float(to_lon), float(to_lat)],
                        ],
                    }
                    payload["walk_distance_m"] = float(
                        _haversine_distance_point_m(from_lat, from_lon, to_lat, to_lon)
                    )
                
                # Add routing method indicator
                payload["routing_method"] = routing_method
                
                # Calculate and add walking speed in km/h
                walk_distance_m = payload.get("walk_distance_m", 0.0)
                if walk_duration_seconds > 0 and walk_distance_m > 0:
                    walk_speed_mps = float(walk_distance_m) / float(walk_duration_seconds)
                    walk_speed_kmh = walk_speed_mps * 3.6
                    payload["walking_speed_kmh"] = round(float(walk_speed_kmh), 2)


        payloads.append(payload)

    if payloads and start_stop_idx is not None and isinstance(origin_anchor, dict):
        origin_lat = _to_float(origin_anchor.get("lat"))
        origin_lon = _to_float(origin_anchor.get("lon"))
        start_lat, start_lon = _segment_coordinates(network, int(start_stop_idx))
        if origin_lat is not None and origin_lon is not None and start_lat is not None and start_lon is not None:
            access_payload = _build_anchor_walking_segment_payload(
                from_lat=float(origin_lat),
                from_lon=float(origin_lon),
                to_lat=float(start_lat),
                to_lon=float(start_lon),
                from_stop_id="ORIGIN",
                to_stop_id=str(network.stop_ids[int(start_stop_idx)]),
                arrival_time=int(departure_time) + max(0, int(access_walk_seconds)),
                walk_duration_seconds=max(0, int(access_walk_seconds)),
                segment_type="access",
            )
            if access_payload is not None:
                payloads = [access_payload, *payloads]

    if payloads and end_stop_idx is not None and isinstance(destination_anchor, dict):
        destination_lat = _to_float(destination_anchor.get("lat"))
        destination_lon = _to_float(destination_anchor.get("lon"))
        end_lat, end_lon = _segment_coordinates(network, int(end_stop_idx))
        if destination_lat is not None and destination_lon is not None and end_lat is not None and end_lon is not None:
            egress_payload = _build_anchor_walking_segment_payload(
                from_lat=float(end_lat),
                from_lon=float(end_lon),
                to_lat=float(destination_lat),
                to_lon=float(destination_lon),
                from_stop_id=str(network.stop_ids[int(end_stop_idx)]),
                to_stop_id="DESTINATION",
                arrival_time=int(segments[-1][2]) + max(0, int(egress_walk_seconds)),
                walk_duration_seconds=max(0, int(egress_walk_seconds)),
                segment_type="egress",
            )
            if egress_payload is not None:
                payloads.append(egress_payload)
    return payloads


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
    allow_far_fallback = bool(origin.get("allow_far_fallback", False))

    walk_speed_mps = float(origin.get("walk_speed_mps", DEFAULT_WALK_SPEED_MPS))
    if walk_speed_mps <= 0:
        walk_speed_mps = DEFAULT_WALK_SPEED_MPS
    walk_time_multiplier = float(origin.get("walk_time_multiplier", DEFAULT_WALK_TIME_MULTIPLIER))
    if walk_time_multiplier <= 0:
        walk_time_multiplier = DEFAULT_WALK_TIME_MULTIPLIER

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
    elif not allow_far_fallback:
        return [], {}, "no_nearby_origin_stop"
    else:
        fallback_count = min(max_candidates, sorted_indices.size)
        selected_indices_full = sorted_indices[:fallback_count]
        selected_distances_full = sorted_distances[:fallback_count]

    ranked_indices = _rank_origin_candidates_by_connectivity(
        network,
        selected_indices_full,
        selected_distances_full,
        walk_speed_mps,
        walk_time_multiplier,
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
        int(stop_idx): int((distance_by_stop[int(stop_idx)] / walk_speed_mps) * walk_time_multiplier)
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
    allow_far_fallback = bool(destination.get("allow_far_fallback", False))

    walk_speed_mps = float(destination.get("walk_speed_mps", DEFAULT_WALK_SPEED_MPS))
    if walk_speed_mps <= 0:
        walk_speed_mps = DEFAULT_WALK_SPEED_MPS
    walk_time_multiplier = float(destination.get("walk_time_multiplier", DEFAULT_WALK_TIME_MULTIPLIER))
    if walk_time_multiplier <= 0:
        walk_time_multiplier = DEFAULT_WALK_TIME_MULTIPLIER

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
    elif not allow_far_fallback:
        return [], {}, "no_nearby_destination_stop"
    else:
        fallback_count = min(max_candidates, sorted_indices.size)
        selected_indices_full = sorted_indices[:fallback_count]
        selected_distances_full = sorted_distances[:fallback_count]

    seed_candidates = _to_int(destination.get("seed_candidates"), min(DEFAULT_DESTINATION_SEED_CANDIDATES, max_candidates))
    seed_candidates = max(1, min(seed_candidates, int(selected_indices_full.size)))
    selected_indices = selected_indices_full[:seed_candidates]

    end_indices = [int(value) for value in selected_indices.tolist()]
    end_penalties = {
        int(stop_idx): int((distance_m / walk_speed_mps) * walk_time_multiplier)
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


def _nearest_station_indices(network: TransitNetwork, lat: float, lon: float, limit: int) -> list[int]:
    station_lats = getattr(network, "station_lats", None)
    station_lons = getattr(network, "station_lons", None)
    if not isinstance(station_lats, np.ndarray) or not isinstance(station_lons, np.ndarray):
        return []
    if station_lats.size == 0 or station_lons.size == 0:
        return []

    valid = np.isfinite(station_lats) & np.isfinite(station_lons)
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return []

    d = _haversine_distance_m(float(lat), float(lon), station_lats[valid_idx], station_lons[valid_idx])
    order = np.argsort(d)
    top = valid_idx[order][: max(1, int(limit))]
    return [int(item) for item in top.tolist()]


def _station_to_stop_indices(network: TransitNetwork, station_id: int, limit: int) -> list[int]:
    offsets = getattr(network, "station_stop_offsets", None)
    stops = getattr(network, "station_stops", None)
    if not isinstance(offsets, np.ndarray) or not isinstance(stops, np.ndarray):
        return []
    if station_id < 0 or station_id + 1 >= offsets.shape[0]:
        return []

    row_start = int(offsets[int(station_id)])
    row_end = int(offsets[int(station_id) + 1])
    if row_end <= row_start:
        return []
    station_stop_ids = [int(item) for item in stops[row_start:row_end].tolist()]
    return station_stop_ids[: max(1, int(limit))]


def _augment_candidates_with_hub_stops(
    network: TransitNetwork,
    base_indices: list[int],
    anchor: dict[str, Any] | None,
    max_hubs: int = DEFAULT_LONG_DISTANCE_RESCUE_HUBS,
    max_stops_per_hub: int = DEFAULT_LONG_DISTANCE_RESCUE_STOPS_PER_HUB,
) -> list[int]:
    if not isinstance(anchor, dict):
        return list(base_indices)
    lat = _to_float(anchor.get("lat"))
    lon = _to_float(anchor.get("lon"))
    if lat is None or lon is None:
        return list(base_indices)

    station_candidates = _nearest_station_indices(network, float(lat), float(lon), int(max_hubs))
    augmented = list(int(item) for item in base_indices)
    for station_id in station_candidates:
        for stop_id in _station_to_stop_indices(network, int(station_id), int(max_stops_per_hub)):
            augmented.append(int(stop_id))
    return list(dict.fromkeys(int(item) for item in augmented))


def _penalties_for_anchor(
    network: TransitNetwork,
    stop_indices: list[int],
    anchor: dict[str, Any] | None,
) -> dict[int, int]:
    if not isinstance(anchor, dict):
        return {}
    lat = _to_float(anchor.get("lat"))
    lon = _to_float(anchor.get("lon"))
    if lat is None or lon is None:
        return {}

    walk_speed_mps = float(anchor.get("walk_speed_mps", DEFAULT_WALK_SPEED_MPS))
    if walk_speed_mps <= 0:
        walk_speed_mps = DEFAULT_WALK_SPEED_MPS
    walk_time_multiplier = float(anchor.get("walk_time_multiplier", DEFAULT_WALK_TIME_MULTIPLIER))
    if walk_time_multiplier <= 0:
        walk_time_multiplier = DEFAULT_WALK_TIME_MULTIPLIER

    penalties: dict[int, int] = {}
    for stop_idx in stop_indices:
        stop_lat, stop_lon = _segment_coordinates(network, int(stop_idx))
        if stop_lat is None or stop_lon is None:
            continue
        distance_m = _haversine_distance_point_m(float(lat), float(lon), float(stop_lat), float(stop_lon))
        penalties[int(stop_idx)] = int((distance_m / walk_speed_mps) * walk_time_multiplier)
    return penalties


def _rescue_nearest_candidates(
    network: TransitNetwork,
    anchor: dict[str, Any] | None,
    target_count: int,
) -> list[int]:
    if not isinstance(anchor, dict):
        return []
    lat = _to_float(anchor.get("lat"))
    lon = _to_float(anchor.get("lon"))
    if lat is None or lon is None:
        return []
    sorted_indices, _ = _sorted_stops_by_distance(network, float(lat), float(lon))
    if sorted_indices.size == 0:
        return []
    count = max(1, min(int(target_count), int(sorted_indices.size)))
    return [int(value) for value in sorted_indices[:count].tolist()]


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


def _select_warmup_stop_pairs(network: TransitNetwork, limit: int = 3) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    offsets = getattr(network, "transfer_offsets", None)
    neighbors = getattr(network, "transfer_neighbors", None)

    if not isinstance(offsets, np.ndarray) or offsets.size <= 1:
        n_stops = int(network.stop_route_offsets.shape[0] - 1)
        if n_stops >= 2:
            return [(0, 1)]
        return []

    if not isinstance(neighbors, np.ndarray) or neighbors.size == 0:
        n_stops = int(offsets.shape[0] - 1)
        if n_stops >= 2:
            return [(0, 1)]
        return []

    n_stops = int(offsets.shape[0] - 1)
    for start_idx in range(n_stops):
        row_start = int(offsets[start_idx])
        row_end = int(offsets[start_idx + 1])
        if row_end <= row_start:
            continue
        end_idx = int(neighbors[row_start])
        if end_idx < 0 or end_idx >= n_stops or end_idx == start_idx:
            continue
        pairs.append((int(start_idx), int(end_idx)))
        if len(pairs) >= int(limit):
            break

    if pairs:
        return pairs
    if n_stops >= 2:
        return [(0, 1)]
    return []


def _warmup_request_pipeline(network: TransitNetwork) -> None:
    started = time.perf_counter()

    has_local_walking_graph = False
    try:
        has_local_walking_graph = bool(_ensure_runtime_walking_graph(network))
    except Exception as exc:
        logger.warning("Runtime walking graph prewarm skipped due to error: %s", exc)

    warmup_pairs = _select_warmup_stop_pairs(network, limit=3)
    if not warmup_pairs:
        logger.info("Startup request-pipeline warmup skipped (no candidate stop pairs)")
        return

    departure_time = int(time.time())
    warmed_route_cases = 0
    warmed_walk_cases = 0

    for start_idx, end_idx in warmup_pairs:
        try:
            _build_option_response(
                network,
                "raptor",
                [int(start_idx)],
                [int(end_idx)],
                departure_time,
                {int(start_idx): 0},
                {int(end_idx): 0},
                None,
                None,
                DEFAULT_MAX_TRANSFERS,
            )
            warmed_route_cases += 1
        except Exception as exc:
            logger.warning(
                "Startup route warmup case failed start=%s end=%s error=%s",
                start_idx,
                end_idx,
                exc,
            )

        if has_local_walking_graph:
            try:
                walking_path = _find_walking_path_via_astar(
                    network,
                    int(start_idx),
                    int(end_idx),
                    max_search_seconds=900,
                )
                if walking_path:
                    warmed_walk_cases += 1
            except Exception as exc:
                logger.warning(
                    "Startup walking-path warmup case failed start=%s end=%s error=%s",
                    start_idx,
                    end_idx,
                    exc,
                )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "Startup request-pipeline warmup complete in %.2f ms route_cases=%s walking_cases=%s local_walking_graph=%s",
        elapsed_ms,
        warmed_route_cases,
        warmed_walk_cases,
        has_local_walking_graph,
    )


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
    if primary == "raptor":
        return ("raptor", "astar", "dijkstra")
    if primary == "astar":
        return ("astar", "dijkstra")
    return ("dijkstra",)


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
                max_transfers=DEFAULT_MAX_TRANSFERS,
                board_scan_limit=DEFAULT_RAPTOR_BOARD_SCAN_LIMIT,
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
        penalty_raw = 0 if start_penalties is None else int(start_penalties.get(only_start, 0))
        penalty = _walk_score_penalty(penalty_raw, DEFAULT_ACCESS_WALK_SCORE_CAP_SECONDS)
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
    best_candidate = None
    for start_idx, result in zip(start_indices, results):
        if not result:
            continue
        penalty_raw = 0 if start_penalties is None else int(start_penalties.get(start_idx, 0))
        penalty = _walk_score_penalty(penalty_raw, DEFAULT_ACCESS_WALK_SCORE_CAP_SECONDS)
        score = _score_segments_with_transfer_penalty(
            result,
            int(result[-1][2]) + penalty,
            max_transfers=max_transfers,
        )
        if score >= int(2**62 - 1):
            continue
        candidate = {
            "segments": result,
            "start_idx": int(start_idx),
            "score": int(score),
            "arrival_time": int(result[-1][2]),
            "transfers": int(_count_transfers(result)),
            "walk_segment_count": int(_count_walk_segments(result)),
            "egress_walk_seconds": 0,
        }
        best_candidate = _choose_better_candidate(best_candidate, candidate)

    if best_candidate is None:
        return [], None, None
    best_segments = best_candidate["segments"]
    best_start = int(best_candidate["start_idx"])
    return best_segments, best_start, int(best_candidate["score"])


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
    board_scan_limit: int = DEFAULT_RAPTOR_BOARD_SCAN_LIMIT,
):
    budget_caps = list(_raptor_round_budgets(network))
    inf = int(2**62)

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
                max_transfers=max_transfers,
                board_scan_limit=board_scan_limit,
            )

            best_end = None
            best_end_score = None
            for end_idx in end_indices:
                arrival = int(earliest[int(end_idx)])
                if arrival >= inf:
                    continue
                egress_penalty_raw = 0 if end_penalties is None else int(end_penalties.get(int(end_idx), 0))
                egress_penalty = _walk_score_penalty(egress_penalty_raw, DEFAULT_EGRESS_WALK_SCORE_CAP_SECONDS)
                score = arrival + egress_penalty
                if best_end_score is None or score < best_end_score:
                    best_end_score = score
                    best_end = int(end_idx)

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

            best_candidate = None
            for end_idx in end_indices:
                arrival = int(earliest[int(end_idx)])
                if arrival >= inf:
                    continue

                segments = build_path(
                    network.stop_times,
                    network.trip_offsets,
                    int(end_idx),
                    earliest,
                    pred_stop,
                    pred_trip,
                    pred_time,
                )
                if not segments:
                    continue

                start_penalty_raw = 0 if start_penalties is None else int(start_penalties.get(int(start_idx), 0))
                egress_penalty_raw = 0 if end_penalties is None else int(end_penalties.get(int(end_idx), 0))
                start_penalty = _walk_score_penalty(start_penalty_raw, DEFAULT_ACCESS_WALK_SCORE_CAP_SECONDS)
                egress_penalty = _walk_score_penalty(egress_penalty_raw, DEFAULT_EGRESS_WALK_SCORE_CAP_SECONDS)
                total_score = _score_segments_with_transfer_penalty(
                    segments,
                    arrival + start_penalty + egress_penalty,
                    max_transfers=max_transfers,
                )
                if total_score >= int(2**62 - 1):
                    continue

                candidate = {
                    "segments": segments,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "score": int(total_score),
                    "arrival_time": int(arrival),
                    "transfers": int(_count_transfers(segments)),
                    "walk_segment_count": int(_count_walk_segments(segments)),
                    "egress_walk_seconds": int(egress_penalty_raw),
                }
                best_candidate = _choose_better_candidate(best_candidate, candidate)

            if best_candidate is None:
                continue

            logger.info(
                "RAPTOR multi-end success start=%s end=%s attempts=%s",
                start_idx,
                int(best_candidate["end_idx"]),
                " | ".join(attempt_summaries),
            )
            return (
                best_candidate["segments"],
                int(best_candidate["start_idx"]),
                int(best_candidate["end_idx"]),
                int(best_candidate["score"]),
                {
                    "success": True,
                    "start_idx": int(start_idx),
                    "end_idx": int(best_candidate["end_idx"]),
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
    best_candidate = None
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
        egress_penalty_raw = 0 if end_penalties is None else int(end_penalties.get(int(end_idx), 0))
        candidate = {
            "segments": segments,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "score": int(score),
            "arrival_time": int(segments[-1][2]),
            "transfers": int(_count_transfers(segments)),
            "walk_segment_count": int(_count_walk_segments(segments)),
            "egress_walk_seconds": int(egress_penalty_raw),
        }
        best_candidate = _choose_better_candidate(best_candidate, candidate)

    if best_candidate is not None:
        best_segments = best_candidate["segments"]
        best_start = int(best_candidate["start_idx"])
        best_end = int(best_candidate["end_idx"])
        best_score = int(best_candidate["score"])
    else:
        best_score = None

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
    board_scan_limit: int = DEFAULT_RAPTOR_BOARD_SCAN_LIMIT,
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
            board_scan_limit,
        )

    best_segments = []
    best_start = None
    best_end = None
    best_candidate = None

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

        egress_penalty_raw = 0 if end_penalties is None else int(end_penalties.get(int(end_idx), 0))
        egress_penalty = _walk_score_penalty(egress_penalty_raw, DEFAULT_EGRESS_WALK_SCORE_CAP_SECONDS)
        total_score = int(start_score) + egress_penalty
        candidate = {
            "segments": segments,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "score": int(total_score),
            "arrival_time": int(segments[-1][2]),
            "transfers": int(_count_transfers(segments)),
            "walk_segment_count": int(_count_walk_segments(segments)),
            "egress_walk_seconds": int(egress_penalty_raw),
        }
        best_candidate = _choose_better_candidate(best_candidate, candidate)

    if best_candidate is not None:
        best_segments = best_candidate["segments"]
        best_start = int(best_candidate["start_idx"])
        best_end = int(best_candidate["end_idx"])
        best_score = int(best_candidate["score"])
    else:
        best_score = None

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
    od_distance_m = None
    if isinstance(origin_anchor, dict) and isinstance(destination_anchor, dict):
        origin_lat = _to_float(origin_anchor.get("lat"))
        origin_lon = _to_float(origin_anchor.get("lon"))
        destination_lat = _to_float(destination_anchor.get("lat"))
        destination_lon = _to_float(destination_anchor.get("lon"))
        if (
            origin_lat is not None
            and origin_lon is not None
            and destination_lat is not None
            and destination_lon is not None
        ):
            od_distance_m = _haversine_distance_point_m(
                float(origin_lat),
                float(origin_lon),
                float(destination_lat),
                float(destination_lon),
            )

    backbone_max_offpath = 2
    if od_distance_m is not None and float(od_distance_m) >= float(DEFAULT_LONG_DISTANCE_RESCUE_MIN_M):
        backbone_max_offpath = int(DEFAULT_LONG_DISTANCE_BACKBONE_MAX_OFFPATH)

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
    raptor_board_scan_limit = int(DEFAULT_RAPTOR_BOARD_SCAN_LIMIT)
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
            raptor_board_scan_limit,
        )
        if segments and algorithm == "raptor":
            if not _segments_follow_station_backbone(
                network,
                segments,
                station_whitelist,
                max_off_path_stations=backbone_max_offpath,
            ):
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
                        raptor_board_scan_limit,
                    )
                    raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, tier_diagnostics)
                    if segments and not _segments_follow_station_backbone(
                        network,
                        segments,
                        station_whitelist,
                        max_off_path_stations=backbone_max_offpath,
                    ):
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
                        raptor_board_scan_limit,
                    )
                    raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, tier_diagnostics)
                    if segments and not _segments_follow_station_backbone(
                        network,
                        segments,
                        station_whitelist,
                        max_off_path_stations=backbone_max_offpath,
                    ):
                        segments = []
                    if segments:
                        logger.info(
                            "RAPTOR adaptive origin expansion hit tier=%s start_candidates=%s",
                            tier_idx,
                            len(tier_start_indices),
                        )
                        break

            if not segments and od_distance_m is not None:
                if od_distance_m >= float(DEFAULT_LONG_DISTANCE_RESCUE_MIN_M):
                        rescue_start_indices = _augment_candidates_with_hub_stops(
                            network,
                            start_indices,
                            origin_anchor,
                        )
                        rescue_end_indices = _augment_candidates_with_hub_stops(
                            network,
                            end_indices,
                            destination_anchor,
                        )
                        rescue_start_penalties = dict(start_penalties or {})
                        rescue_start_penalties.update(
                            _penalties_for_anchor(network, rescue_start_indices, origin_anchor)
                        )
                        rescue_end_penalties = dict(end_penalties or {})
                        rescue_end_penalties.update(
                            _penalties_for_anchor(network, rescue_end_indices, destination_anchor)
                        )

                        rescue_station_whitelist, _ = _compute_station_backbone(
                            network,
                            rescue_start_indices,
                            rescue_end_indices,
                            max_transfers,
                        )

                        segments, best_start, best_end, _, rescue_diagnostics = _find_best_segments_for_od_candidates(
                            network,
                            selected_algorithm,
                            rescue_start_indices,
                            rescue_end_indices,
                            departure_time,
                            rescue_start_penalties,
                            rescue_end_penalties,
                            max_transfers,
                            raptor_board_scan_limit,
                        )
                        raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, rescue_diagnostics)
                        if segments and not _segments_follow_station_backbone(
                            network,
                            segments,
                            rescue_station_whitelist,
                            max_off_path_stations=backbone_max_offpath,
                        ):
                            segments = []
                        if segments:
                            start_penalties = rescue_start_penalties
                            end_penalties = rescue_end_penalties
                            logger.info(
                                "RAPTOR long-distance rescue hit od_km=%.1f start_candidates=%s end_candidates=%s",
                                float(od_distance_m) / 1000.0,
                                len(rescue_start_indices),
                                len(rescue_end_indices),
                            )

                        if not segments:
                            relaxed_max_transfers = min(
                                int(DEFAULT_LONG_DISTANCE_RESCUE_MAX_TRANSFERS),
                                int(max_transfers) + int(DEFAULT_LONG_DISTANCE_RESCUE_EXTRA_TRANSFERS),
                            )
                            if relaxed_max_transfers > int(max_transfers):
                                segments, best_start, best_end, _, relaxed_diagnostics = _find_best_segments_for_od_candidates(
                                    network,
                                    selected_algorithm,
                                    rescue_start_indices,
                                    rescue_end_indices,
                                    departure_time,
                                    rescue_start_penalties,
                                    rescue_end_penalties,
                                    int(relaxed_max_transfers),
                                    raptor_board_scan_limit,
                                )
                                raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, relaxed_diagnostics)
                                if segments and not _segments_follow_station_backbone(
                                    network,
                                    segments,
                                    rescue_station_whitelist,
                                    max_off_path_stations=backbone_max_offpath,
                                ):
                                    segments = []
                                if segments:
                                    start_penalties = rescue_start_penalties
                                    end_penalties = rescue_end_penalties
                                    logger.info(
                                        "RAPTOR long-distance relaxed-transfer rescue hit od_km=%.1f max_transfers=%s->%s",
                                        float(od_distance_m) / 1000.0,
                                        int(max_transfers),
                                        int(relaxed_max_transfers),
                                    )

            # General fallback: if strict transfer cap finds no route, relax the cap once.
            # This avoids intermittent no-path outcomes for normal (non long-distance) requests.
            if not segments:
                relaxed_max_transfers = min(
                    int(DEFAULT_LONG_DISTANCE_RESCUE_MAX_TRANSFERS),
                    int(max_transfers) + 2,
                )
                if relaxed_max_transfers > int(max_transfers):
                    segments, best_start, best_end, _, relaxed_diagnostics = _find_best_segments_for_od_candidates(
                        network,
                        selected_algorithm,
                        start_indices,
                        end_indices,
                        departure_time,
                        start_penalties,
                        end_penalties,
                        int(relaxed_max_transfers),
                        raptor_board_scan_limit,
                    )
                    raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, relaxed_diagnostics)
                    if segments and not _segments_follow_station_backbone(
                        network,
                        segments,
                        station_whitelist,
                        max_off_path_stations=backbone_max_offpath,
                    ):
                        segments = []
                    if segments:
                        logger.info(
                            "RAPTOR relaxed-transfer fallback hit max_transfers=%s->%s",
                            int(max_transfers),
                            int(relaxed_max_transfers),
                        )

            # Final RAPTOR rescue: widen candidate boarding scan only when strict passes fail.
            if not segments:
                rescue_scan_limit = max(
                    int(DEFAULT_RAPTOR_RESCUE_BOARD_SCAN_LIMIT),
                    int(raptor_board_scan_limit),
                )
                if rescue_scan_limit > int(raptor_board_scan_limit):
                    segments, best_start, best_end, _, rescue_scan_diagnostics = _find_best_segments_for_od_candidates(
                        network,
                        selected_algorithm,
                        start_indices,
                        end_indices,
                        departure_time,
                        start_penalties,
                        end_penalties,
                        int(max_transfers),
                        int(rescue_scan_limit),
                    )
                    raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, rescue_scan_diagnostics)
                    if segments and not _segments_follow_station_backbone(
                        network,
                        segments,
                        station_whitelist,
                        max_off_path_stations=backbone_max_offpath,
                    ):
                        segments = []
                    if segments:
                        raptor_board_scan_limit = int(rescue_scan_limit)
                        logger.info(
                            "RAPTOR rescue scan fallback hit board_scan_limit=%s->%s",
                            int(DEFAULT_RAPTOR_BOARD_SCAN_LIMIT),
                            int(rescue_scan_limit),
                        )

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
                raptor_diagnostics["board_scan_limit"] = int(raptor_board_scan_limit)

        if selected_algorithm != "raptor":
            if not segments:
                destination_tiers = _expand_candidates_tiered(
                    network,
                    end_indices,
                    destination_anchor,
                    "destination",
                )
                for tier_idx, tier_end_indices in enumerate(destination_tiers[1:], start=1):
                    segments, best_start, best_end, _, _ = _find_best_segments_for_od_candidates(
                        network,
                        selected_algorithm,
                        start_indices,
                        tier_end_indices,
                        departure_time,
                        start_penalties,
                        end_penalties,
                        max_transfers,
                    )
                    if segments:
                        logger.info(
                            "%s adaptive destination expansion hit tier=%s end_candidates=%s",
                            selected_algorithm.upper(),
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
                    segments, best_start, best_end, _, _ = _find_best_segments_for_od_candidates(
                        network,
                        selected_algorithm,
                        tier_start_indices,
                        end_indices,
                        departure_time,
                        start_penalties,
                        end_penalties,
                        max_transfers,
                    )
                    if segments:
                        logger.info(
                            "%s adaptive origin expansion hit tier=%s start_candidates=%s",
                            selected_algorithm.upper(),
                            tier_idx,
                            len(tier_start_indices),
                        )
                        break

        # Final generic rescue for all algorithms: widen nearest OD candidate pools.
        if not segments and (isinstance(origin_anchor, dict) or isinstance(destination_anchor, dict)):
            rescue_start_target = max(int(DEFAULT_GENERAL_RESCUE_MAX_CANDIDATES), len(start_indices))
            rescue_end_target = max(int(DEFAULT_GENERAL_RESCUE_MAX_CANDIDATES), len(end_indices))
            rescue_start_indices = _rescue_nearest_candidates(network, origin_anchor, rescue_start_target) or list(start_indices)
            rescue_end_indices = _rescue_nearest_candidates(network, destination_anchor, rescue_end_target) or list(end_indices)

            rescue_start_penalties = dict(start_penalties or {})
            rescue_start_penalties.update(_penalties_for_anchor(network, rescue_start_indices, origin_anchor))
            rescue_end_penalties = dict(end_penalties or {})
            rescue_end_penalties.update(_penalties_for_anchor(network, rescue_end_indices, destination_anchor))

            rescue_max_transfers = int(max_transfers)
            if selected_algorithm == "raptor":
                rescue_max_transfers = min(
                    int(DEFAULT_LONG_DISTANCE_RESCUE_MAX_TRANSFERS),
                    int(max_transfers) + int(DEFAULT_LONG_DISTANCE_RESCUE_EXTRA_TRANSFERS),
                )

            segments, best_start, best_end, _, rescue_diagnostics = _find_best_segments_for_od_candidates(
                network,
                selected_algorithm,
                rescue_start_indices,
                rescue_end_indices,
                departure_time,
                rescue_start_penalties,
                rescue_end_penalties,
                rescue_max_transfers,
                raptor_board_scan_limit,
            )

            if selected_algorithm == "raptor":
                raptor_diagnostics = _merge_raptor_diagnostics(raptor_diagnostics, rescue_diagnostics)

            if segments:
                start_penalties = rescue_start_penalties
                end_penalties = rescue_end_penalties
                logger.info(
                    "%s final candidate rescue hit start_candidates=%s end_candidates=%s",
                    selected_algorithm.upper(),
                    len(rescue_start_indices),
                    len(rescue_end_indices),
                )

        if segments:
            resolved_algorithm = selected_algorithm
            break

    if segments is None:
        return {}
    option_trip_profile = _summarize_option_trip_profile(network, segments)
    access_walk_seconds = int(start_penalties.get(best_start, 0)) if (best_start is not None and start_penalties) else 0
    egress_walk_seconds = int(end_penalties.get(best_end, 0)) if (best_end is not None and end_penalties) else 0
    return {
        "departure_time": int(departure_time),
        "resolver_algorithm": resolved_algorithm,
        "fallback_used": resolved_algorithm != algorithm,
        "transfers": _count_transfers(segments),
        "max_transfers": int(max_transfers),
        "duration_seconds": int(segments[-1][2] - departure_time) if segments else None,
        "start_stop_id": network.stop_ids[best_start] if best_start is not None else None,
        "access_walk_seconds": access_walk_seconds,
        "end_stop_id": network.stop_ids[best_end] if best_end is not None else None,
        "egress_walk_seconds": egress_walk_seconds,
        "segments": _build_segment_payloads(
            network,
            segments,
            best_start,
            departure_time,
            end_stop_idx=best_end,
            origin_anchor=origin_anchor,
            destination_anchor=destination_anchor,
            access_walk_seconds=access_walk_seconds,
            egress_walk_seconds=egress_walk_seconds,
        ),
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
    option_count: int = DEFAULT_OPTION_COUNT,
    next_option_max_evals: int = DEFAULT_NEXT_OPTION_MAX_EVALS,
    next_option_max_wall_seconds: float = DEFAULT_NEXT_OPTION_MAX_WALL_SECONDS,
) -> dict[str, Any]:
    if isinstance(end_idx, list):
        end_indices = [int(value) for value in end_idx]
    else:
        end_indices = [int(end_idx)]

    origin_anchor = metadata.get("origin") if isinstance(metadata, dict) else None
    destination_anchor = metadata.get("destination") if isinstance(metadata, dict) else None

    options: list[dict[str, Any]] = []
    next_option_search_status = "not_requested"
    next_option_search_elapsed_seconds = 0.0
    next_option_search_evals = 0

    requested_option_count = max(1, int(option_count))
    eval_budget = max(0, int(next_option_max_evals))
    wall_budget_s = max(0.1, float(next_option_max_wall_seconds))

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
        next_option_search_status = "completed"
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
        while len(options) < requested_option_count:
            previous_signature = _first_transit_trip_signature(options[-1])
            previous_departure = int(options[-1].get("departure_time", current_departure))
            search_departure = previous_departure + step_seconds
            search_limit = previous_departure + max_lookahead_seconds
            found_next = False
            search_step_seconds = step_seconds
            same_signature_streak = 0

            while search_departure <= search_limit:
                if next_option_evals >= eval_budget:
                    break
                if (time.perf_counter() - next_option_started) >= wall_budget_s:
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
                if next_option_evals >= eval_budget:
                    next_option_search_status = "evaluation_budget"
                    logger.info(
                        "Dynamic next-option search stopped by evaluation budget evals=%s options=%s",
                        next_option_evals,
                        len(options),
                    )
                elif (time.perf_counter() - next_option_started) >= wall_budget_s:
                    next_option_search_status = "wall_time_budget"
                    logger.info(
                        "Dynamic next-option search stopped by wall-time budget elapsed=%.2fs options=%s evals=%s",
                        (time.perf_counter() - next_option_started),
                        len(options),
                        next_option_evals,
                    )
                else:
                    next_option_search_status = "search_window_exhausted"
                break

        next_option_search_elapsed_seconds = max(0.0, float(time.perf_counter() - next_option_started))
        next_option_search_evals = int(next_option_evals)

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
        "next_option_search_status": next_option_search_status,
        "next_option_search_elapsed_seconds": round(float(next_option_search_elapsed_seconds), 3),
        "next_option_search_evals": int(next_option_search_evals),
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
        option_count_raw = payload.get("option_count")
        next_option_max_evals_raw = payload.get("next_option_max_evals")
        next_option_max_wall_seconds_raw = payload.get("next_option_max_wall_seconds")
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

        option_count = int(DEFAULT_OPTION_COUNT)
        if option_count_raw is not None:
            if isinstance(option_count_raw, (int, float)):
                option_count = int(option_count_raw)
            elif isinstance(option_count_raw, str) and option_count_raw.strip().lstrip("-").isdigit():
                option_count = int(option_count_raw.strip())
            else:
                self._send_json(400, {"error": "invalid_option_count"})
                return
        if option_count < 1:
            self._send_json(400, {"error": "invalid_option_count"})
            return

        next_option_max_evals = int(DEFAULT_NEXT_OPTION_MAX_EVALS)
        if next_option_max_evals_raw is not None:
            if isinstance(next_option_max_evals_raw, (int, float)):
                next_option_max_evals = int(next_option_max_evals_raw)
            elif isinstance(next_option_max_evals_raw, str) and next_option_max_evals_raw.strip().lstrip("-").isdigit():
                next_option_max_evals = int(next_option_max_evals_raw.strip())
            else:
                self._send_json(400, {"error": "invalid_next_option_max_evals"})
                return
        if next_option_max_evals < 0:
            self._send_json(400, {"error": "invalid_next_option_max_evals"})
            return

        next_option_max_wall_seconds = float(DEFAULT_NEXT_OPTION_MAX_WALL_SECONDS)
        if next_option_max_wall_seconds_raw is not None:
            value = _to_float(next_option_max_wall_seconds_raw)
            if value is None:
                self._send_json(400, {"error": "invalid_next_option_max_wall_seconds"})
                return
            next_option_max_wall_seconds = float(value)
        if next_option_max_wall_seconds <= 0:
            self._send_json(400, {"error": "invalid_next_option_max_wall_seconds"})
            return

        if self.network is None:
            self._send_json(500, {"error": "network_not_loaded"})
            return

        metadata = None
        resolved_stop_queries: list[dict[str, Any]] = []
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
                stop_index, resolution = _resolve_stop_query_to_index(self.network, stop_id)
                if stop_index is None:
                    self._send_json(404, {"error": "unknown_stop_id"})
                    return
                start_indices.append(stop_index)
                if resolution is not None and resolution.get("match_type") != "exact_stop_id":
                    resolved_stop_queries.append(
                        {
                            "field": "start_stop_id",
                            **resolution,
                        }
                    )

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
            resolved_end_idx, resolution = _resolve_stop_query_to_index(self.network, end_stop_id)
            if resolved_end_idx is None:
                self._send_json(404, {"error": "unknown_stop_id"})
                return
            end_indices = [resolved_end_idx]
            if resolution is not None and resolution.get("match_type") != "exact_stop_id":
                resolved_stop_queries.append(
                    {
                        "field": "end_stop_id",
                        **resolution,
                    }
                )

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
            option_count=option_count,
            next_option_max_evals=next_option_max_evals,
            next_option_max_wall_seconds=next_option_max_wall_seconds,
        )
        if resolved_stop_queries:
            response["resolved_stop_queries"] = resolved_stop_queries
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
    try:
        _warmup_request_pipeline(PathRequestHandler.network)
    except Exception as exc:
        logger.warning("Request-pipeline warmup skipped due to error: %s", exc)
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

