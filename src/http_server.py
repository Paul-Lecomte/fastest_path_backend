# This module implements a simple HTTP server that provides a pathfinding API for transit networks.
from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .config import setup_logging
from .loader import NetworkLoader, build_mock_network, TransitNetwork
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

DEFAULT_OFFSET_MINUTES = (0, 10, 20, 30, 40)
DEFAULT_START_EXPANSION_HOPS = 2
DEFAULT_START_EXPANSION_MAX_STOPS = 256
DEFAULT_ORIGIN_RADIUS_M = 1000.0
DEFAULT_ORIGIN_MAX_CANDIDATES = 12
DEFAULT_WALK_SPEED_MPS = 1.4
UNIX_TIMESTAMP_MIN_SECONDS = 946684800
DEFAULT_RAPTOR_ROUNDS_MIN = 8
DEFAULT_RAPTOR_ROUNDS_MAX = 64


def _count_transfers(segments) -> int:
    if not segments:
        return 0
    transfers = 0
    current_trip = segments[0][0]
    for trip_id, _, _ in segments[1:]:
        if trip_id != current_trip:
            transfers += 1
            current_trip = trip_id
    return transfers


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


def _segment_coordinates(network: TransitNetwork, stop_id: int) -> tuple[float | None, float | None]:
    lat = float(network.stop_lats[stop_id])
    lon = float(network.stop_lons[stop_id])
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None, None
    return lat, lon


def _segment_payload(network: TransitNetwork, trip_id: int, stop_id: int, arrival_time: int) -> dict[str, Any]:
    lat, lon = _segment_coordinates(network, stop_id)
    return {
        "trip_id": network.trip_ids[trip_id],
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
        selected_indices = sorted_indices[within_radius][:max_candidates]
        selected_distances = sorted_distances[within_radius][:max_candidates]
    else:
        fallback_count = min(max_candidates, sorted_indices.size)
        selected_indices = sorted_indices[:fallback_count]
        selected_distances = sorted_distances[:fallback_count]

    start_indices = [int(value) for value in selected_indices.tolist()]
    start_penalties = {
        int(stop_idx): int(distance_m / walk_speed_mps)
        for stop_idx, distance_m in zip(selected_indices, selected_distances)
    }

    metadata = {
        "origin": {
            "lat": float(lat),
            "lon": float(lon),
            "radius_m": float(radius_m),
            "max_candidates": int(max_candidates),
        },
        "candidate_start_count": len(start_indices),
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
        selected_indices = sorted_indices[within_radius][:max_candidates]
        selected_distances = sorted_distances[within_radius][:max_candidates]
    else:
        fallback_count = min(max_candidates, sorted_indices.size)
        selected_indices = sorted_indices[:fallback_count]
        selected_distances = sorted_distances[:fallback_count]

    end_indices = [int(value) for value in selected_indices.tolist()]
    end_penalties = {
        int(stop_idx): int(distance_m / walk_speed_mps)
        for stop_idx, distance_m in zip(selected_indices, selected_distances)
    }

    metadata = {
        "destination": {
            "lat": float(lat),
            "lon": float(lon),
            "radius_m": float(radius_m),
            "max_candidates": int(max_candidates),
        },
        "candidate_end_count": len(end_indices),
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
        start_idx,
        end_idx,
        0,
        max_rounds=1,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("RAPTOR warmup complete in %.2f ms", elapsed_ms)


def _raptor_round_budgets(network: TransitNetwork) -> tuple[int, ...]:
    n_stops = int(network.stop_route_offsets.shape[0] - 1)
    adaptive_cap = min(DEFAULT_RAPTOR_ROUNDS_MAX, max(DEFAULT_RAPTOR_ROUNDS_MIN, int(np.sqrt(max(1, n_stops))) + 8))
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
):
    if len(start_indices) == 1:
        only_start = start_indices[0]
        segments = _compute_segments(network, algorithm, only_start, end_idx, departure_time)
        if not segments:
            return [], None, None
        penalty = 0 if start_penalties is None else int(start_penalties.get(only_start, 0))
        score = int(segments[-1][2]) + penalty
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
        score = int(result[-1][2]) + penalty
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
):
    def compute_for_start(start_idx: int):
        attempt_summaries = []
        for max_rounds in _raptor_round_budgets(network):
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
                int(start_idx),
                -1,
                departure_time,
                max_rounds=max_rounds,
            )
            best_end, best_end_score = _best_end_from_earliest(earliest, end_indices, end_penalties)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            attempt_summaries.append(
                f"cap={max_rounds} used={int(rounds_used)} marked={int(marked_count)} ms={elapsed_ms:.1f} best_end={best_end}"
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
            total_score = int(best_end_score) + start_penalty
            logger.info(
                "RAPTOR multi-end success start=%s end=%s attempts=%s",
                start_idx,
                best_end,
                " | ".join(attempt_summaries),
            )
            return segments, int(start_idx), best_end, total_score

        logger.info("RAPTOR multi-end no-path start=%s attempts=%s", start_idx, " | ".join(attempt_summaries))
        return [], None, None, None

    if len(start_indices) == 1:
        return compute_for_start(start_indices[0])

    if not numba_enabled():
        results = [compute_for_start(start_idx) for start_idx in start_indices]
    else:
        with ThreadPoolExecutor(max_workers=min(len(start_indices), 16)) as pool:
            futures = [pool.submit(compute_for_start, start_idx) for start_idx in start_indices]
            results = [future.result() for future in futures]

    best_segments = []
    best_start = None
    best_end = None
    best_score = None
    for segments, start_idx, end_idx, score in results:
        if not segments or start_idx is None or end_idx is None or score is None:
            continue
        if best_score is None or score < best_score:
            best_score = int(score)
            best_segments = segments
            best_start = int(start_idx)
            best_end = int(end_idx)

    return best_segments, best_start, best_end, best_score


def _find_best_segments_for_od_candidates(
    network: TransitNetwork,
    algorithm: str,
    start_indices: list[int],
    end_indices: list[int],
    departure_time: int,
    start_penalties: dict[int, int] | None = None,
    end_penalties: dict[int, int] | None = None,
):
    if algorithm == "raptor":
        return _find_best_segments_for_od_candidates_raptor(
            network,
            start_indices,
            end_indices,
            departure_time,
            start_penalties,
            end_penalties,
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

    return best_segments, best_start, best_end, best_score


def _build_option_response(
    network: TransitNetwork,
    algorithm: str,
    start_indices: list[int],
    end_indices: list[int],
    departure_time: int,
    start_penalties: dict[int, int] | None = None,
    end_penalties: dict[int, int] | None = None,
) -> dict[str, Any]:
    algorithms = _algorithm_sequence(algorithm)

    segments = []
    best_start = None
    best_end = None
    resolved_algorithm = algorithm
    for selected_algorithm in algorithms:
        segments, best_start, best_end, _ = _find_best_segments_for_od_candidates(
            network,
            selected_algorithm,
            start_indices,
            end_indices,
            departure_time,
            start_penalties,
            end_penalties,
        )
        if segments:
            resolved_algorithm = selected_algorithm
            break

    if not segments:
        expanded_start_indices = _expand_start_indices(network, start_indices)
        if len(expanded_start_indices) > len(start_indices):
            for selected_algorithm in algorithms:
                segments, best_start, best_end, _ = _find_best_segments_for_od_candidates(
                    network,
                    selected_algorithm,
                    expanded_start_indices,
                    end_indices,
                    departure_time,
                    start_penalties,
                    end_penalties,
                )
                if segments:
                    resolved_algorithm = selected_algorithm
                    logger.info(
                        "Route fallback used algorithm=%s resolved=%s start_expansion=%s original_starts=%s",
                        algorithm,
                        selected_algorithm,
                        len(expanded_start_indices),
                        len(start_indices),
                    )
                    break
                

    if segments is None:
        return {}
    return {
        "departure_time": int(departure_time),
        "resolver_algorithm": resolved_algorithm,
        "fallback_used": resolved_algorithm != algorithm,
        "transfers": _count_transfers(segments),
        "duration_seconds": int(segments[-1][2] - departure_time) if segments else None,
        "start_stop_id": network.stop_ids[best_start] if best_start is not None else None,
        "access_walk_seconds": int(start_penalties.get(best_start, 0)) if (best_start is not None and start_penalties) else 0,
        "end_stop_id": network.stop_ids[best_end] if best_end is not None else None,
        "egress_walk_seconds": int(end_penalties.get(best_end, 0)) if (best_end is not None and end_penalties) else 0,
        "segments": [
            _segment_payload(network, trip_id, stop_id, arrival_time)
            for trip_id, stop_id, arrival_time in segments
        ],
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
) -> dict[str, Any]:
    if isinstance(end_idx, list):
        end_indices = [int(value) for value in end_idx]
    else:
        end_indices = [int(end_idx)]

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
            )

    base = options[0] if options else {
        "departure_time": int(departure_time),
        "transfers": 0,
        "duration_seconds": None,
        "segments": [],
    }
    return {
        "algorithm": algorithm,
        "candidate_start_count": len(start_indices),
        "candidate_end_count": len(end_indices),
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

