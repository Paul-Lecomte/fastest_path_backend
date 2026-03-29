# This module defines the NetworkLoader class, which is responsible for loading transit network data from a Neo4j database and converting it into a format suitable for pathfinding algorithms. The loader fetches stop times, routes, and stops from the database, processes the data to build an adjacency list representation of the transit network, and returns a TransitNetwork object containing all the relevant information.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import logging
import os
import pickle
import time
from pathlib import Path
import numpy as np
from neo4j import GraphDatabase


logger = logging.getLogger("pathfinding.loader")


DEFAULT_TRANSFER_WALK_SPEED_MPS = 1.4
DEFAULT_TRANSFER_MIN_SECONDS = 30
DEFAULT_TRANSFER_FALLBACK_SECONDS = 120
DEFAULT_TRANSFER_MAX_DISTANCE_M = 250.0
DEFAULT_TRANSFER_NEARBY_MAX_DISTANCE_M = 120.0
DEFAULT_TRANSFER_NEARBY_MAX_NEIGHBORS = 4
DEFAULT_WALK_TIME_MULTIPLIER = 15.0
DEFAULT_TRANSFER_CACHE_PATH = ".cache/walk_transfers_osm.npz"
DEFAULT_TRANSFER_CACHE_FALLBACK_PATHS = (
    ".cache/transfer_graph_osrm.npz",
)

DEFAULT_ROUTE_TYPE_UNKNOWN = -1
DEFAULT_ROUTE_TYPE_COST_FACTORS = {
    -1: 1450,
    0: 1100,
    1: 1100,
    2: 1000,
    3: 2000,
    4: 1200,
    5: 1300,
    6: 1200,
    7: 2000,
    11: 2000,
    12: 2000,
    100: 1000,
    109: 1100,
    1000: 1200,
}


def parse_time_to_seconds(value) -> int:
    if value is None:
        raise ValueError("arrival_time is None")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        parts = text.split(":")
        if len(parts) == 2:
            hours, minutes = parts
            seconds = 0
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            raise ValueError(f"Unsupported time format: {value}")
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    raise ValueError(f"Unsupported time type: {type(value)}")


STOP_TIMES_DTYPE = np.dtype(
    [
        ("stop_id", np.int32),
        ("trip_id", np.int32),
        ("arrival_time", np.int64),
        ("stop_seq", np.int32),
    ]
)

ROUTES_DTYPE = np.dtype(
    [
        ("route_id", np.int32),
        ("trip_id", np.int32),
    ]
)

STOPS_DTYPE = np.dtype(
    [
        ("stop_id", np.int32),
        ("stop_seq", np.int32),
    ]
)


@dataclass
class TransitNetwork:
    stops: np.ndarray
    stop_times: np.ndarray
    routes: np.ndarray
    stop_id_index: Dict[str, int]
    trip_id_index: Dict[str, int]
    stop_ids: List[str]
    stop_lats: np.ndarray
    stop_lons: np.ndarray
    trip_ids: List[str]
    trip_offsets: np.ndarray
    adj_offsets: np.ndarray
    adj_neighbors: np.ndarray
    adj_weights: np.ndarray
    adj_trip_ids: np.ndarray
    route_stop_offsets: np.ndarray
    route_stops: np.ndarray
    route_trip_offsets: np.ndarray
    route_trips: np.ndarray
    route_board_offsets: np.ndarray
    route_board_times: np.ndarray
    route_board_monotonic: np.ndarray
    stop_route_offsets: np.ndarray
    stop_routes: np.ndarray
    trip_route_types: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    trip_cost_factors: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    transfer_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    transfer_neighbors: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    transfer_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    station_keys: List[str] = field(default_factory=list)
    stop_station_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    station_stop_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    station_stops: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    station_lats: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    station_lons: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    station_adj_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    station_adj_neighbors: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    station_adj_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    hub_station_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    # OSM walking graph for detailed pathfinding
    walking_node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    walking_node_lats: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    walking_node_lons: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    walking_adj_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    walking_adj_neighbors: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    walking_adj_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    stop_to_walking_node_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))


def _parse_route_type(value) -> int:
    if value is None:
        return DEFAULT_ROUTE_TYPE_UNKNOWN
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
        normalized = text.lower()
        if normalized in {"rail", "train", "intercity", "interregio", "ic", "ir", "rer", "sbahn", "s-bahn", "subway", "metro"}:
            return 2
        if normalized in {"tram", "streetcar", "light_rail", "light-rail"}:
            return 0
        if normalized in {"bus", "coach", "autobus"}:
            return 3
        if normalized in {"ferry", "boat"}:
            return 4
        if normalized in {"cable", "gondola", "funicular", "lift"}:
            return 6
        try:
            return int(float(normalized))
        except ValueError:
            return DEFAULT_ROUTE_TYPE_UNKNOWN
    return DEFAULT_ROUTE_TYPE_UNKNOWN


def _route_type_cost_factor(route_type: int) -> int:
    return int(DEFAULT_ROUTE_TYPE_COST_FACTORS.get(int(route_type), DEFAULT_ROUTE_TYPE_COST_FACTORS[DEFAULT_ROUTE_TYPE_UNKNOWN]))


def _trip_factor_from_speed_mps(avg_speed_mps: float) -> int:
    if avg_speed_mps >= 14.0:
        return 1000
    if avg_speed_mps >= 10.0:
        return 1150
    if avg_speed_mps >= 7.0:
        return 1450
    return 1900


def _estimate_trip_average_speed_mps(
    trip_id: int,
    stop_times: np.ndarray,
    trip_offsets: np.ndarray,
    stop_lats: np.ndarray,
    stop_lons: np.ndarray,
) -> float | None:
    start = int(trip_offsets[trip_id])
    end = int(trip_offsets[trip_id + 1])
    if end - start <= 1:
        return None

    total_distance_m = 0.0
    total_seconds = 0.0
    for idx in range(start + 1, end):
        prev_stop_id = int(stop_times[idx - 1][0])
        stop_id = int(stop_times[idx][0])
        prev_time = int(stop_times[idx - 1][2])
        arrival_time = int(stop_times[idx][2])
        delta_t = arrival_time - prev_time
        if delta_t <= 0:
            continue

        if prev_stop_id >= stop_lats.shape[0] or stop_id >= stop_lats.shape[0]:
            continue
        lat1 = float(stop_lats[prev_stop_id])
        lon1 = float(stop_lons[prev_stop_id])
        lat2 = float(stop_lats[stop_id])
        lon2 = float(stop_lons[stop_id])
        if not (np.isfinite(lat1) and np.isfinite(lon1) and np.isfinite(lat2) and np.isfinite(lon2)):
            continue

        distance_m = _haversine_m(lat1, lon1, lat2, lon2)
        if distance_m <= 0:
            continue
        total_distance_m += float(distance_m)
        total_seconds += float(delta_t)

    if total_seconds <= 0:
        return None
    return total_distance_m / total_seconds


def _build_trip_cost_factors(
    trip_route_types: np.ndarray,
    stop_times: np.ndarray | None = None,
    trip_offsets: np.ndarray | None = None,
    stop_lats: np.ndarray | None = None,
    stop_lons: np.ndarray | None = None,
) -> np.ndarray:
    factors = np.empty(trip_route_types.shape[0], dtype=np.int64)
    for idx in range(trip_route_types.shape[0]):
        route_type = int(trip_route_types[idx])
        if route_type != DEFAULT_ROUTE_TYPE_UNKNOWN:
            factors[idx] = _route_type_cost_factor(route_type)
            continue

        speed_mps = None
        if (
            stop_times is not None
            and trip_offsets is not None
            and stop_lats is not None
            and stop_lons is not None
        ):
            speed_mps = _estimate_trip_average_speed_mps(
                idx,
                stop_times,
                trip_offsets,
                stop_lats,
                stop_lons,
            )
        if speed_mps is None:
            factors[idx] = 1500
        else:
            factors[idx] = _trip_factor_from_speed_mps(float(speed_mps))
    return factors


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_earth_m = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    sin_half_dphi = np.sin(dphi / 2.0)
    sin_half_dlambda = np.sin(dlambda / 2.0)
    a = sin_half_dphi * sin_half_dphi + np.cos(phi1) * np.cos(phi2) * sin_half_dlambda * sin_half_dlambda
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(radius_earth_m * c)


def _station_key(stop_id: str) -> str:
    if not isinstance(stop_id, str):
        return ""
    text = stop_id.strip()
    if not text:
        return ""
    separator = text.find(":")
    if separator == -1:
        return text
    return text[:separator]


def _build_station_backbone(
    stop_ids: list[str],
    stop_times: np.ndarray,
    trip_offsets: np.ndarray,
    stop_lats: np.ndarray,
    stop_lons: np.ndarray,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_stops = len(stop_ids)
    if n_stops <= 0:
        return (
            [],
            np.zeros(0, dtype=np.int32),
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
        )

    station_index: Dict[str, int] = {}
    station_keys: list[str] = []
    station_members: List[List[int]] = []
    stop_station_ids = np.zeros(n_stops, dtype=np.int32)

    for stop_idx, stop_id in enumerate(stop_ids):
        key = _station_key(stop_id)
        if key not in station_index:
            station_index[key] = len(station_keys)
            station_keys.append(key)
            station_members.append([])
        station_id = station_index[key]
        stop_station_ids[stop_idx] = station_id
        station_members[station_id].append(stop_idx)

    station_count = len(station_keys)
    station_stop_offsets = np.zeros(station_count + 1, dtype=np.int64)
    station_stops = np.zeros(n_stops, dtype=np.int32)
    cursor = 0
    for station_id, members in enumerate(station_members):
        station_stop_offsets[station_id] = cursor
        for stop_idx in members:
            station_stops[cursor] = int(stop_idx)
            cursor += 1
    station_stop_offsets[station_count] = cursor

    station_lats = np.full(station_count, np.nan, dtype=np.float64)
    station_lons = np.full(station_count, np.nan, dtype=np.float64)
    for station_id, members in enumerate(station_members):
        member_array = np.array(members, dtype=np.int32)
        member_lats = stop_lats[member_array]
        member_lons = stop_lons[member_array]
        finite_lats = member_lats[np.isfinite(member_lats)]
        finite_lons = member_lons[np.isfinite(member_lons)]
        if finite_lats.size > 0:
            station_lats[station_id] = float(np.mean(finite_lats))
        if finite_lons.size > 0:
            station_lons[station_id] = float(np.mean(finite_lons))

    station_edges: List[Dict[int, int]] = [dict() for _ in range(station_count)]
    n_trips = int(trip_offsets.shape[0] - 1)
    for trip_id in range(n_trips):
        trip_start = int(trip_offsets[trip_id])
        trip_end = int(trip_offsets[trip_id + 1])
        if trip_end - trip_start <= 1:
            continue

        prev_station = int(stop_station_ids[int(stop_times[trip_start][0])])
        prev_time = int(stop_times[trip_start][2])
        for idx in range(trip_start + 1, trip_end):
            stop_id = int(stop_times[idx][0])
            station_id = int(stop_station_ids[stop_id])
            arrival_time = int(stop_times[idx][2])
            if station_id != prev_station:
                travel_time = arrival_time - prev_time
                if travel_time >= 0:
                    current = station_edges[prev_station].get(station_id)
                    if current is None or travel_time < current:
                        station_edges[prev_station][station_id] = int(travel_time)
                prev_station = station_id
                prev_time = arrival_time

    total_station_edges = sum(len(item) for item in station_edges)
    station_adj_offsets = np.zeros(station_count + 1, dtype=np.int64)
    station_adj_neighbors = np.zeros(total_station_edges, dtype=np.int32)
    station_adj_weights = np.zeros(total_station_edges, dtype=np.int64)
    cursor = 0
    for station_id in range(station_count):
        station_adj_offsets[station_id] = cursor
        for neighbor_station, weight in station_edges[station_id].items():
            station_adj_neighbors[cursor] = int(neighbor_station)
            station_adj_weights[cursor] = int(weight)
            cursor += 1
    station_adj_offsets[station_count] = cursor

    out_degree = np.array([len(station_edges[idx]) for idx in range(station_count)], dtype=np.int64)
    in_degree = np.zeros(station_count, dtype=np.int64)
    for source_station in range(station_count):
        for target_station in station_edges[source_station].keys():
            in_degree[int(target_station)] += 1
    station_degree = out_degree + in_degree

    if station_count <= 512:
        hub_station_indices = np.arange(station_count, dtype=np.int32)
    else:
        degree_order = np.argsort(-station_degree)
        hub_station_indices = degree_order[:512].astype(np.int32)

    return (
        station_keys,
        stop_station_ids,
        station_stop_offsets,
        station_stops,
        station_lats,
        station_lons,
        station_adj_offsets,
        station_adj_neighbors,
        station_adj_weights,
        hub_station_indices,
    )


def _build_transfers(
    stop_ids: list[str],
    stop_lats: np.ndarray,
    stop_lons: np.ndarray,
    walk_speed_mps: float = DEFAULT_TRANSFER_WALK_SPEED_MPS,
    min_seconds: int = DEFAULT_TRANSFER_MIN_SECONDS,
    fallback_seconds: int = DEFAULT_TRANSFER_FALLBACK_SECONDS,
    max_distance_m: float = DEFAULT_TRANSFER_MAX_DISTANCE_M,
    nearby_max_distance_m: float = DEFAULT_TRANSFER_NEARBY_MAX_DISTANCE_M,
    nearby_max_neighbors: int = DEFAULT_TRANSFER_NEARBY_MAX_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_stops = len(stop_ids)
    if n_stops <= 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)

    groups: Dict[str, List[int]] = {}
    for stop_idx, stop_id in enumerate(stop_ids):
        key = _station_key(stop_id)
        if not key:
            continue
        groups.setdefault(key, []).append(stop_idx)

    edges: List[Dict[int, int]] = [dict() for _ in range(n_stops)]
    speed = float(walk_speed_mps) if walk_speed_mps > 0 else DEFAULT_TRANSFER_WALK_SPEED_MPS
    min_walk = int(min_seconds) if min_seconds > 0 else DEFAULT_TRANSFER_MIN_SECONDS
    fallback_walk = int(fallback_seconds) if fallback_seconds > 0 else DEFAULT_TRANSFER_FALLBACK_SECONDS
    max_distance = float(max_distance_m) if max_distance_m > 0 else DEFAULT_TRANSFER_MAX_DISTANCE_M

    for members in groups.values():
        member_count = len(members)
        if member_count <= 1:
            continue

        for source in members:
            src_lat = float(stop_lats[source]) if source < stop_lats.shape[0] else np.nan
            src_lon = float(stop_lons[source]) if source < stop_lons.shape[0] else np.nan
            src_valid = np.isfinite(src_lat) and np.isfinite(src_lon)

            for target in members:
                if target == source:
                    continue
                travel_seconds = fallback_walk
                tgt_lat = float(stop_lats[target]) if target < stop_lats.shape[0] else np.nan
                tgt_lon = float(stop_lons[target]) if target < stop_lons.shape[0] else np.nan
                tgt_valid = np.isfinite(tgt_lat) and np.isfinite(tgt_lon)
                if src_valid and tgt_valid:
                    distance_m = _haversine_m(src_lat, src_lon, tgt_lat, tgt_lon)
                    if distance_m > max_distance:
                        continue
                    travel_seconds = max(min_walk, int(distance_m / speed))

                current = edges[source].get(target)
                if current is None or travel_seconds < current:
                    edges[source][target] = int(travel_seconds)

    nearby_distance_limit = float(nearby_max_distance_m) if nearby_max_distance_m > 0 else 0.0
    nearby_neighbor_cap = int(nearby_max_neighbors) if nearby_max_neighbors > 0 else 0
    if nearby_distance_limit > 0 and nearby_neighbor_cap > 0:
        lat_step_deg = nearby_distance_limit / 111320.0
        lon_step_deg = nearby_distance_limit / 111320.0
        if lat_step_deg > 0 and lon_step_deg > 0:
            buckets: Dict[Tuple[int, int], List[int]] = {}
            valid_stops: List[int] = []
            for stop_idx in range(n_stops):
                lat = float(stop_lats[stop_idx]) if stop_idx < stop_lats.shape[0] else np.nan
                lon = float(stop_lons[stop_idx]) if stop_idx < stop_lons.shape[0] else np.nan
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    continue
                row = int(np.floor(lat / lat_step_deg))
                col = int(np.floor(lon / lon_step_deg))
                buckets.setdefault((row, col), []).append(stop_idx)
                valid_stops.append(stop_idx)

            for source in valid_stops:
                src_lat = float(stop_lats[source])
                src_lon = float(stop_lons[source])
                src_row = int(np.floor(src_lat / lat_step_deg))
                src_col = int(np.floor(src_lon / lon_step_deg))

                candidates: List[Tuple[float, int]] = []
                for d_row in (-1, 0, 1):
                    for d_col in (-1, 0, 1):
                        for target in buckets.get((src_row + d_row, src_col + d_col), []):
                            if target == source:
                                continue
                            tgt_lat = float(stop_lats[target])
                            tgt_lon = float(stop_lons[target])
                            distance_m = _haversine_m(src_lat, src_lon, tgt_lat, tgt_lon)
                            if distance_m <= 0 or distance_m > nearby_distance_limit:
                                continue
                            candidates.append((distance_m, target))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: item[0])
                for distance_m, target in candidates[:nearby_neighbor_cap]:
                    travel_seconds = max(min_walk, int(distance_m / speed))
                    current = edges[source].get(target)
                    if current is None or travel_seconds < current:
                        edges[source][target] = int(travel_seconds)

    total_edges = sum(len(neighbors) for neighbors in edges)
    transfer_offsets = np.zeros(n_stops + 1, dtype=np.int64)
    transfer_neighbors = np.zeros(total_edges, dtype=np.int32)
    transfer_weights = np.zeros(total_edges, dtype=np.int64)

    cursor = 0
    for stop_idx in range(n_stops):
        transfer_offsets[stop_idx] = cursor
        for neighbor_idx, weight in edges[stop_idx].items():
            transfer_neighbors[cursor] = int(neighbor_idx)
            transfer_weights[cursor] = int(weight)
            cursor += 1
    transfer_offsets[n_stops] = cursor
    return transfer_offsets, transfer_neighbors, transfer_weights


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    text = raw.strip()
    if not text:
        return float(default)
    try:
        value = float(text)
    except ValueError:
        return float(default)
    if not np.isfinite(value) or value <= 0:
        return float(default)
    return float(value)


def _get_transfer_cache_path() -> str:
    raw = os.getenv("WALK_TRANSFER_CACHE_PATH", DEFAULT_TRANSFER_CACHE_PATH)
    configured = raw.strip() if raw and raw.strip() else DEFAULT_TRANSFER_CACHE_PATH
    if Path(configured).exists():
        return configured

    for fallback_path in DEFAULT_TRANSFER_CACHE_FALLBACK_PATHS:
        if Path(fallback_path).exists():
            logger.info(
                "Using fallback transfer cache path=%s because configured path is missing path=%s",
                fallback_path,
                configured,
            )
            return fallback_path
    return configured


def load_transfer_graph_from_cache(
    cache_path: str,
    stop_ids: list[str],
) -> dict | None:
    path = Path(cache_path)
    if not path.exists():
        return None

    try:
        payload = np.load(path, allow_pickle=True)
        keys = set(payload.files)
        required_core = {"transfer_offsets", "transfer_neighbors", "transfer_weights"}
        if not required_core.issubset(keys):
            logger.warning("Ignoring transfer cache without transfer arrays path=%s", path)
            return None

        expected_count = int(len(stop_ids))
        if "stop_ids" in keys:
            cached_stop_ids = payload["stop_ids"]
            expected_stop_ids = np.asarray(stop_ids)
            if cached_stop_ids.shape != expected_stop_ids.shape:
                logger.warning(
                    "Ignoring transfer cache due to stop count mismatch path=%s cache=%s network=%s",
                    path,
                    cached_stop_ids.shape[0] if cached_stop_ids.ndim == 1 else -1,
                    expected_stop_ids.shape[0] if expected_stop_ids.ndim == 1 else -1,
                )
                return None
            if not np.array_equal(cached_stop_ids, expected_stop_ids):
                logger.warning("Ignoring transfer cache due to stop id mismatch path=%s", path)
                return None
        elif "stop_count" in keys:
            cached_count = int(np.asarray(payload["stop_count"]).reshape(-1)[0])
            if cached_count != expected_count:
                logger.warning(
                    "Ignoring transfer cache due to stop_count mismatch path=%s cache=%s network=%s",
                    path,
                    cached_count,
                    expected_count,
                )
                return None
            logger.info(
                "Transfer cache path=%s validated by stop_count only (stop_ids not present)",
                path,
            )
        else:
            logger.warning("Ignoring transfer cache without stop_ids or stop_count metadata path=%s", path)
            return None

        transfer_offsets = np.asarray(payload["transfer_offsets"], dtype=np.int64)
        transfer_neighbors = np.asarray(payload["transfer_neighbors"], dtype=np.int32)
        transfer_weights = np.asarray(payload["transfer_weights"], dtype=np.int64)

        walk_time_multiplier = _parse_positive_float_env("WALK_TIME_MULTIPLIER", DEFAULT_WALK_TIME_MULTIPLIER)
        if abs(walk_time_multiplier - 1.0) > 1e-9:
            transfer_weights = np.maximum(
                1,
                np.rint(transfer_weights.astype(np.float64) * walk_time_multiplier).astype(np.int64),
            )

        if transfer_offsets.shape[0] != len(stop_ids) + 1:
            logger.warning("Ignoring transfer cache due to invalid offsets shape path=%s", path)
            return None
        if transfer_neighbors.shape[0] != transfer_weights.shape[0]:
            logger.warning("Ignoring transfer cache due to neighbors/weights mismatch path=%s", path)
            return None

        result = {
            'transfer_offsets': transfer_offsets,
            'transfer_neighbors': transfer_neighbors,
            'transfer_weights': transfer_weights,
        }
        
        # Load OSM walking graph data if available
        keys = set(payload.files)
        if 'osm_node_ids' in keys:
            result['walking_node_ids'] = np.asarray(payload['osm_node_ids'], dtype=object)
        if 'osm_node_lats' in keys:
            result['walking_node_lats'] = np.asarray(payload['osm_node_lats'], dtype=np.float64)
        if 'osm_node_lons' in keys:
            result['walking_node_lons'] = np.asarray(payload['osm_node_lons'], dtype=np.float64)
        if 'osm_adj_offsets' in keys:
            result['walking_adj_offsets'] = np.asarray(payload['osm_adj_offsets'], dtype=np.int64)
        if 'osm_adj_neighbors' in keys:
            result['walking_adj_neighbors'] = np.asarray(payload['osm_adj_neighbors'], dtype=np.int32)
        if 'osm_adj_weights' in keys:
            result['walking_adj_weights'] = np.asarray(payload['osm_adj_weights'], dtype=np.int64)
        if 'stop_to_node_idx' in keys:
            result['stop_to_walking_node_idx'] = np.asarray(payload['stop_to_node_idx'], dtype=np.int32)
        
        logger.info(
            "Loaded precomputed transfer cache path=%s stops=%s edges=%s",
            path,
            len(stop_ids),
            transfer_neighbors.shape[0],
        )
        return result
    except Exception as exc:
        logger.warning("Failed to load transfer cache path=%s error=%s", path, exc)
        return None


def save_transfer_graph_to_cache(
    cache_path: str,
    stop_ids: list[str],
    transfer_offsets: np.ndarray,
    transfer_neighbors: np.ndarray,
    transfer_weights: np.ndarray,
    osm_node_ids: np.ndarray | None = None,
    osm_node_lats: np.ndarray | None = None,
    osm_node_lons: np.ndarray | None = None,
    osm_adj_offsets: np.ndarray | None = None,
    osm_adj_neighbors: np.ndarray | None = None,
    osm_adj_weights: np.ndarray | None = None,
    stop_to_node_idx: np.ndarray | None = None,
) -> None:
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'stop_ids': np.asarray(stop_ids),
            'transfer_offsets': np.asarray(transfer_offsets, dtype=np.int64),
            'transfer_neighbors': np.asarray(transfer_neighbors, dtype=np.int32),
            'transfer_weights': np.asarray(transfer_weights, dtype=np.int64),
        }
        # Save OSM graph data if provided
        if osm_node_ids is not None:
            save_dict['osm_node_ids'] = np.asarray(osm_node_ids, dtype=object)
        if osm_node_lats is not None:
            save_dict['osm_node_lats'] = np.asarray(osm_node_lats, dtype=np.float64)
        if osm_node_lons is not None:
            save_dict['osm_node_lons'] = np.asarray(osm_node_lons, dtype=np.float64)
        if osm_adj_offsets is not None:
            save_dict['osm_adj_offsets'] = np.asarray(osm_adj_offsets, dtype=np.int64)
        if osm_adj_neighbors is not None:
            save_dict['osm_adj_neighbors'] = np.asarray(osm_adj_neighbors, dtype=np.int32)
        if osm_adj_weights is not None:
            save_dict['osm_adj_weights'] = np.asarray(osm_adj_weights, dtype=np.int64)
        if stop_to_node_idx is not None:
            save_dict['stop_to_node_idx'] = np.asarray(stop_to_node_idx, dtype=np.int32)
        
        np.savez_compressed(path, **save_dict)
        logger.info(
            "Saved transfer cache path=%s stops=%s edges=%s",
            path,
            len(stop_ids),
            int(np.asarray(transfer_neighbors).shape[0]),
        )
    except Exception as exc:
        logger.warning("Failed to save transfer cache path=%s error=%s", path, exc)


def _load_precomputed_transfer_graph_for_stop_ids(stop_ids: list[str]) -> dict | None:
    if not _parse_bool_env("WALK_TRANSFER_CACHE_ENABLED", True):
        return None
    cache_path = _get_transfer_cache_path()
    return load_transfer_graph_from_cache(cache_path, stop_ids)


def _ensure_transfer_graph(network: TransitNetwork) -> TransitNetwork:
    # Backward compatibility: old pickled TransitNetwork instances may miss
    # walking graph fields added in newer versions.
    if not hasattr(network, "walking_node_ids"):
        network.walking_node_ids = np.zeros(0, dtype=object)
    if not hasattr(network, "walking_node_lats"):
        network.walking_node_lats = np.zeros(0, dtype=np.float64)
    if not hasattr(network, "walking_node_lons"):
        network.walking_node_lons = np.zeros(0, dtype=np.float64)
    if not hasattr(network, "walking_adj_offsets"):
        network.walking_adj_offsets = np.zeros(1, dtype=np.int64)
    if not hasattr(network, "walking_adj_neighbors"):
        network.walking_adj_neighbors = np.zeros(0, dtype=np.int32)
    if not hasattr(network, "walking_adj_weights"):
        network.walking_adj_weights = np.zeros(0, dtype=np.int64)
    if not hasattr(network, "stop_to_walking_node_idx"):
        network.stop_to_walking_node_idx = np.zeros(0, dtype=np.int32)

    precomputed = _load_precomputed_transfer_graph_for_stop_ids(network.stop_ids)
    if precomputed is not None:
        network.transfer_offsets = precomputed['transfer_offsets']
        network.transfer_neighbors = precomputed['transfer_neighbors']
        network.transfer_weights = precomputed['transfer_weights']
        # Load OSM walking graph if available
        if 'walking_node_ids' in precomputed:
            network.walking_node_ids = precomputed['walking_node_ids']
        if 'walking_node_lats' in precomputed:
            network.walking_node_lats = precomputed['walking_node_lats']
        if 'walking_node_lons' in precomputed:
            network.walking_node_lons = precomputed['walking_node_lons']
        if 'walking_adj_offsets' in precomputed:
            network.walking_adj_offsets = precomputed['walking_adj_offsets']
        if 'walking_adj_neighbors' in precomputed:
            network.walking_adj_neighbors = precomputed['walking_adj_neighbors']
        if 'walking_adj_weights' in precomputed:
            network.walking_adj_weights = precomputed['walking_adj_weights']
        if 'stop_to_walking_node_idx' in precomputed:
            network.stop_to_walking_node_idx = precomputed['stop_to_walking_node_idx']
        return network

    n_stops = int(network.stops.shape[0])
    offsets = getattr(network, "transfer_offsets", None)
    neighbors = getattr(network, "transfer_neighbors", None)
    weights = getattr(network, "transfer_weights", None)

    if isinstance(offsets, np.ndarray) and isinstance(neighbors, np.ndarray) and isinstance(weights, np.ndarray):
        expected = n_stops + 1
        if offsets.shape[0] == expected:
            return network

    transfer_offsets, transfer_neighbors, transfer_weights = _build_transfers(
        network.stop_ids,
        network.stop_lats,
        network.stop_lons,
    )

    walk_time_multiplier = _parse_positive_float_env("WALK_TIME_MULTIPLIER", DEFAULT_WALK_TIME_MULTIPLIER)
    if abs(walk_time_multiplier - 1.0) > 1e-9 and transfer_weights.size > 0:
        transfer_weights = np.maximum(
            1,
            np.rint(transfer_weights.astype(np.float64) * walk_time_multiplier).astype(np.int64),
        )

    network.transfer_offsets = transfer_offsets
    network.transfer_neighbors = transfer_neighbors
    network.transfer_weights = transfer_weights
    return network


def _ensure_trip_cost_arrays(network: TransitNetwork) -> TransitNetwork:
    n_trips = int(network.trip_offsets.shape[0] - 1)
    route_types = getattr(network, "trip_route_types", None)
    factors = getattr(network, "trip_cost_factors", None)

    if not isinstance(route_types, np.ndarray) or route_types.shape[0] != n_trips:
        route_types = np.full(n_trips, DEFAULT_ROUTE_TYPE_UNKNOWN, dtype=np.int16)
        network.trip_route_types = route_types

    if not isinstance(factors, np.ndarray) or factors.shape[0] != n_trips:
        factors = _build_trip_cost_factors(
            network.trip_route_types,
            network.stop_times,
            network.trip_offsets,
            network.stop_lats,
            network.stop_lons,
        )
        network.trip_cost_factors = factors

    return network


def _ensure_station_backbone(network: TransitNetwork) -> TransitNetwork:
    n_stops = int(network.stops.shape[0])
    stop_station_ids = getattr(network, "stop_station_ids", None)
    station_adj_offsets = getattr(network, "station_adj_offsets", None)

    has_station_mapping = isinstance(stop_station_ids, np.ndarray) and stop_station_ids.shape[0] == n_stops
    has_station_graph = isinstance(station_adj_offsets, np.ndarray) and station_adj_offsets.shape[0] >= 1
    if has_station_mapping and has_station_graph:
        return network

    (
        station_keys,
        stop_station_ids,
        station_stop_offsets,
        station_stops,
        station_lats,
        station_lons,
        station_adj_offsets,
        station_adj_neighbors,
        station_adj_weights,
        hub_station_indices,
    ) = _build_station_backbone(
        network.stop_ids,
        network.stop_times,
        network.trip_offsets,
        network.stop_lats,
        network.stop_lons,
    )
    network.station_keys = station_keys
    network.stop_station_ids = stop_station_ids
    network.station_stop_offsets = station_stop_offsets
    network.station_stops = station_stops
    network.station_lats = station_lats
    network.station_lons = station_lons
    network.station_adj_offsets = station_adj_offsets
    network.station_adj_neighbors = station_adj_neighbors
    network.station_adj_weights = station_adj_weights
    network.hub_station_indices = hub_station_indices
    return network


def summarize_trip_profiles(network: TransitNetwork, top_k: int = 8) -> dict[str, object]:
    trip_route_types = getattr(network, "trip_route_types", np.zeros(0, dtype=np.int16))
    trip_cost_factors = getattr(network, "trip_cost_factors", np.zeros(0, dtype=np.int64))

    trip_count = int(trip_cost_factors.shape[0])
    if trip_count <= 0:
        return {
            "trip_count": 0,
            "route_type_unknown_share": 0.0,
            "route_type_counts_top": [],
            "factor_counts_top": [],
        }

    unknown_mask = trip_route_types == int(DEFAULT_ROUTE_TYPE_UNKNOWN)
    unknown_share = float(np.count_nonzero(unknown_mask) / max(1, trip_count))

    route_values, route_counts = np.unique(trip_route_types.astype(np.int64), return_counts=True)
    route_pairs = sorted(
        [(int(route_type), int(count)) for route_type, count in zip(route_values.tolist(), route_counts.tolist())],
        key=lambda item: item[1],
        reverse=True,
    )

    factor_values, factor_counts = np.unique(trip_cost_factors.astype(np.int64), return_counts=True)
    factor_pairs = sorted(
        [(int(factor), int(count)) for factor, count in zip(factor_values.tolist(), factor_counts.tolist())],
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "trip_count": trip_count,
        "route_type_unknown_share": round(unknown_share, 6),
        "route_type_counts_top": route_pairs[: max(1, int(top_k))],
        "factor_counts_top": factor_pairs[: max(1, int(top_k))],
    }


def load_network_from_cache(cache_path: str, max_age_seconds: int | None = None) -> TransitNetwork | None:
    path = Path(cache_path)
    if not path.exists():
        return None

    if max_age_seconds is not None:
        age_seconds = int(time.time() - path.stat().st_mtime)
        if age_seconds > max_age_seconds:
            logger.info(
                "Skipping network cache at %s because age=%ss exceeds max_age=%ss",
                path,
                age_seconds,
                max_age_seconds,
            )
            return None

    try:
        with path.open("rb") as handle:
            network = pickle.load(handle)
        if not isinstance(network, TransitNetwork):
            logger.warning("Ignoring invalid network cache payload at %s", path)
            return None
        logger.info(
            "Loaded network cache path=%s stops=%s stop_times=%s routes=%s",
            path,
            network.stops.shape[0],
            network.stop_times.shape[0],
            network.routes.shape[0],
        )
        network = _ensure_transfer_graph(network)
        network = _ensure_trip_cost_arrays(network)
        network = _ensure_station_backbone(network)
        return network
    except Exception as exc:
        logger.warning("Failed to load network cache path=%s error=%s", path, exc)
        return None


def save_network_to_cache(cache_path: str, network: TransitNetwork) -> None:
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved network cache to %s", path)
    except Exception as exc:
        logger.warning("Failed to save network cache path=%s error=%s", path, exc)


class NetworkLoader:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def fetch_to_numpy(self) -> TransitNetwork:
        query = (
            "MATCH (st:Stop_times)-[:AT_STOP]->(s:Stop) "
            "MATCH (st)-[:PART_OF_TRIP]->(t:Trip) "
            "RETURN s.stop_id AS stop_id, t.trip_id AS trip_id, "
            "coalesce(st.arrival_time, st.departure_time) AS arrival_time, "
            "st.stop_sequence AS stop_sequence, "
            "s.lat AS stop_lat, "
            "s.lon AS stop_lon, "
            "-1 AS route_type"
        )

        stop_ids: List[str] = []
        trip_ids: List[str] = []
        stop_id_index: Dict[str, int] = {}
        trip_id_index: Dict[str, int] = {}
        stop_coords: Dict[int, Tuple[float, float]] = {}
        trip_route_type_by_index: Dict[int, int] = {}
        rows: List[Tuple[int, int, int, int]] = []

        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                stop_id = record["stop_id"]
                trip_id = record["trip_id"]
                arrival_time = record["arrival_time"]
                stop_sequence = record["stop_sequence"]
                stop_lat = record["stop_lat"]
                stop_lon = record["stop_lon"]
                route_type = _parse_route_type(record.get("route_type"))
                if arrival_time is None or stop_sequence is None:
                    continue
                try:
                    arrival_time = parse_time_to_seconds(arrival_time)
                    stop_sequence = int(stop_sequence)
                except ValueError as exc:
                    logger.warning(
                        "Skipping stop_time stop_id=%s trip_id=%s arrival_time=%s stop_sequence=%s error=%s",
                        stop_id,
                        trip_id,
                        arrival_time,
                        stop_sequence,
                        exc,
                    )
                    continue

                if stop_id not in stop_id_index:
                    stop_id_index[stop_id] = len(stop_ids)
                    stop_ids.append(stop_id)
                if trip_id not in trip_id_index:
                    trip_id_index[trip_id] = len(trip_ids)
                    trip_ids.append(trip_id)

                trip_index = trip_id_index[trip_id]
                existing_route_type = trip_route_type_by_index.get(trip_index)
                if existing_route_type is None or existing_route_type == DEFAULT_ROUTE_TYPE_UNKNOWN:
                    trip_route_type_by_index[trip_index] = route_type

                stop_index = stop_id_index[stop_id]
                if stop_index not in stop_coords and stop_lat is not None and stop_lon is not None:
                    try:
                        stop_coords[stop_index] = (float(stop_lat), float(stop_lon))
                    except (TypeError, ValueError):
                        pass

                rows.append(
                    (
                        stop_id_index[stop_id],
                        trip_id_index[trip_id],
                        arrival_time,
                        stop_sequence,
                    )
                )

        rows.sort(key=lambda item: (item[1], item[3], item[2]))

        stops_array = np.zeros(len(stop_ids), dtype=STOPS_DTYPE)
        for i, _ in enumerate(stop_ids):
            stops_array[i] = (i, i)

        stop_lats = np.full(len(stop_ids), np.nan, dtype=np.float64)
        stop_lons = np.full(len(stop_ids), np.nan, dtype=np.float64)
        for stop_index, (lat, lon) in stop_coords.items():
            stop_lats[stop_index] = lat
            stop_lons[stop_index] = lon

        stop_times_array = np.zeros(len(rows), dtype=STOP_TIMES_DTYPE)
        for i, (stop_id, trip_id, arrival_time, stop_sequence) in enumerate(rows):
            stop_times_array[i] = (stop_id, trip_id, arrival_time, stop_sequence)

        routes_array = np.zeros(len(trip_ids), dtype=ROUTES_DTYPE)
        for i, _ in enumerate(trip_ids):
            routes_array[i] = (i, i)

        trip_offsets = np.zeros(len(trip_ids) + 1, dtype=np.int64)
        current_trip = 0
        trip_offsets[0] = 0
        for i, (_, trip_id, _, _) in enumerate(rows):
            while trip_id > current_trip:
                current_trip += 1
                trip_offsets[current_trip] = i
        for idx in range(current_trip + 1, len(trip_offsets)):
            trip_offsets[idx] = len(rows)

        trip_route_types = np.full(len(trip_ids), DEFAULT_ROUTE_TYPE_UNKNOWN, dtype=np.int16)
        for trip_index, route_type in trip_route_type_by_index.items():
            if 0 <= int(trip_index) < trip_route_types.shape[0]:
                trip_route_types[int(trip_index)] = int(route_type)
        trip_cost_factors = _build_trip_cost_factors(
            trip_route_types,
            stop_times_array,
            trip_offsets,
            stop_lats,
            stop_lons,
        )

        adj_offsets, adj_neighbors, adj_weights, adj_trip_ids = _build_adjacency(
            stop_times_array, trip_offsets, len(stop_ids)
        )

        (
            route_stop_offsets,
            route_stops,
            route_trip_offsets,
            route_trips,
            route_board_offsets,
            route_board_times,
            route_board_monotonic,
            stop_route_offsets,
            stop_routes,
        ) = _build_routes(stop_times_array, trip_offsets, len(stop_ids))

        precomputed = _load_precomputed_transfer_graph_for_stop_ids(stop_ids)
        if precomputed is not None:
            transfer_offsets, transfer_neighbors, transfer_weights = precomputed
        else:
            transfer_offsets, transfer_neighbors, transfer_weights = _build_transfers(
                stop_ids,
                stop_lats,
                stop_lons,
            )

        network = TransitNetwork(
            stops=stops_array,
            stop_times=stop_times_array,
            routes=routes_array,
            stop_id_index=stop_id_index,
            trip_id_index=trip_id_index,
            stop_ids=stop_ids,
            stop_lats=stop_lats,
            stop_lons=stop_lons,
            trip_ids=trip_ids,
            trip_offsets=trip_offsets,
            adj_offsets=adj_offsets,
            adj_neighbors=adj_neighbors,
            adj_weights=adj_weights,
            adj_trip_ids=adj_trip_ids,
            route_stop_offsets=route_stop_offsets,
            route_stops=route_stops,
            route_trip_offsets=route_trip_offsets,
            route_trips=route_trips,
            route_board_offsets=route_board_offsets,
            route_board_times=route_board_times,
            route_board_monotonic=route_board_monotonic,
            stop_route_offsets=stop_route_offsets,
            stop_routes=stop_routes,
            trip_route_types=trip_route_types,
            trip_cost_factors=trip_cost_factors,
            transfer_offsets=transfer_offsets,
            transfer_neighbors=transfer_neighbors,
            transfer_weights=transfer_weights,
        )
        return _ensure_station_backbone(network)


def _build_adjacency(stop_times: np.ndarray, trip_offsets: np.ndarray, n_stops: int):
    # Build a stop->stop graph using minimum in-trip travel times.
    edges = [dict() for _ in range(n_stops)]
    trip_edges = [dict() for _ in range(n_stops)]

    for trip_id in range(trip_offsets.shape[0] - 1):
        start = int(trip_offsets[trip_id])
        end = int(trip_offsets[trip_id + 1])
        if start >= end:
            continue
        prev_stop = int(stop_times[start][0])
        prev_time = int(stop_times[start][2])
        for i in range(start + 1, end):
            stop_id = int(stop_times[i][0])
            arrival_time = int(stop_times[i][2])
            travel_time = arrival_time - prev_time
            if travel_time >= 0:
                current = edges[prev_stop].get(stop_id)
                if current is None or travel_time < current:
                    edges[prev_stop][stop_id] = travel_time
                    trip_edges[prev_stop][stop_id] = trip_id
            prev_stop = stop_id
            prev_time = arrival_time

    total_edges = sum(len(item) for item in edges)
    adj_offsets = np.zeros(n_stops + 1, dtype=np.int64)
    adj_neighbors = np.zeros(total_edges, dtype=np.int32)
    adj_weights = np.zeros(total_edges, dtype=np.int64)
    adj_trip_ids = np.zeros(total_edges, dtype=np.int32)

    cursor = 0
    for stop_id in range(n_stops):
        adj_offsets[stop_id] = cursor
        for neighbor, weight in edges[stop_id].items():
            adj_neighbors[cursor] = neighbor
            adj_weights[cursor] = weight
            adj_trip_ids[cursor] = trip_edges[stop_id][neighbor]
            cursor += 1
    adj_offsets[n_stops] = cursor

    return adj_offsets, adj_neighbors, adj_weights, adj_trip_ids


def _build_routes(
    stop_times: np.ndarray,
    trip_offsets: np.ndarray,
    n_stops: int,
):
    route_id_by_stops: Dict[Tuple[int, ...], int] = {}
    route_stops_list: List[List[int]] = []
    route_trips_list: List[List[int]] = []

    n_trips = int(trip_offsets.shape[0] - 1)
    for trip_id in range(n_trips):
        start = int(trip_offsets[trip_id])
        end = int(trip_offsets[trip_id + 1])
        if start >= end:
            continue
        stops = [int(stop_times[i][0]) for i in range(start, end)]
        key = tuple(stops)
        route_id = route_id_by_stops.get(key)
        if route_id is None:
            route_id = len(route_stops_list)
            route_id_by_stops[key] = route_id
            route_stops_list.append(stops)
            route_trips_list.append([])
        route_trips_list[route_id].append(trip_id)

    for route_id, trips in enumerate(route_trips_list):
        trips.sort(key=lambda trip: int(stop_times[int(trip_offsets[trip])][2]))
        route_trips_list[route_id] = trips

    total_stops = sum(len(stops) for stops in route_stops_list)
    route_stop_offsets = np.zeros(len(route_stops_list) + 1, dtype=np.int64)
    route_stops = np.zeros(total_stops, dtype=np.int32)
    cursor = 0
    for route_id, stops in enumerate(route_stops_list):
        route_stop_offsets[route_id] = cursor
        for stop_id in stops:
            route_stops[cursor] = stop_id
            cursor += 1
    route_stop_offsets[len(route_stops_list)] = cursor

    total_trips = sum(len(trips) for trips in route_trips_list)
    route_trip_offsets = np.zeros(len(route_trips_list) + 1, dtype=np.int64)
    route_trips = np.zeros(total_trips, dtype=np.int32)
    cursor = 0
    for route_id, trips in enumerate(route_trips_list):
        route_trip_offsets[route_id] = cursor
        for trip_id in trips:
            route_trips[cursor] = trip_id
            cursor += 1
    route_trip_offsets[len(route_trips_list)] = cursor

    route_stop_trip_counts = [len(route_stops_list[route_id]) * len(route_trips_list[route_id]) for route_id in range(len(route_stops_list))]
    total_route_board_values = sum(route_stop_trip_counts)
    route_board_offsets = np.zeros(total_stops + 1, dtype=np.int64)
    route_board_times = np.zeros(total_route_board_values, dtype=np.int64)
    route_board_monotonic = np.ones(total_stops, dtype=np.uint8)

    cursor = 0
    for route_id, stops in enumerate(route_stops_list):
        trips = route_trips_list[route_id]
        route_stop_start = int(route_stop_offsets[route_id])
        for stop_local_index, _ in enumerate(stops):
            global_route_stop_index = route_stop_start + stop_local_index
            route_board_offsets[global_route_stop_index] = cursor
            prev_time = -1
            is_monotonic = 1
            for trip_id in trips:
                st_idx = int(trip_offsets[trip_id]) + stop_local_index
                arrival_time = int(stop_times[st_idx][2])
                if prev_time > arrival_time:
                    is_monotonic = 0
                route_board_times[cursor] = arrival_time
                prev_time = arrival_time
                cursor += 1
            route_board_monotonic[global_route_stop_index] = is_monotonic
    route_board_offsets[total_stops] = cursor

    stop_routes_list: List[List[int]] = [[] for _ in range(n_stops)]
    for route_id, stops in enumerate(route_stops_list):
        for stop_id in stops:
            stop_routes_list[stop_id].append(route_id)

    total_stop_routes = sum(len(routes) for routes in stop_routes_list)
    stop_route_offsets = np.zeros(n_stops + 1, dtype=np.int64)
    stop_routes = np.zeros(total_stop_routes, dtype=np.int32)
    cursor = 0
    for stop_id in range(n_stops):
        stop_route_offsets[stop_id] = cursor
        for route_id in stop_routes_list[stop_id]:
            stop_routes[cursor] = route_id
            cursor += 1
    stop_route_offsets[n_stops] = cursor

    return (
        route_stop_offsets,
        route_stops,
        route_trip_offsets,
        route_trips,
        route_board_offsets,
        route_board_times,
        route_board_monotonic,
        stop_route_offsets,
        stop_routes,
    )


def build_mock_network() -> TransitNetwork:
    stop_ids = ["A", "B", "C"]
    trip_ids = ["T1", "T2"]

    stops_array = np.zeros(len(stop_ids), dtype=STOPS_DTYPE)
    for i, _ in enumerate(stop_ids):
        stops_array[i] = (i, i)

    stop_times_array = np.array(
        [
            (0, 0, 1000, 1),
            (1, 0, 1050, 2),
            (1, 1, 1060, 1),
            (2, 1, 1100, 2),
        ],
        dtype=STOP_TIMES_DTYPE,
    )

    routes_array = np.zeros(len(trip_ids), dtype=ROUTES_DTYPE)
    routes_array[0] = (0, 0)
    routes_array[1] = (1, 1)

    stop_id_index = {"A": 0, "B": 1, "C": 2}
    stop_lats = np.array([48.8566, 48.8570, 48.8574], dtype=np.float64)
    stop_lons = np.array([2.3522, 2.3530, 2.3540], dtype=np.float64)
    trip_id_index = {"T1": 0, "T2": 1}
    trip_offsets = np.array([0, 2, 4], dtype=np.int64)
    trip_route_types = np.array([3, 2], dtype=np.int16)
    trip_cost_factors = _build_trip_cost_factors(
        trip_route_types,
        stop_times_array,
        trip_offsets,
        stop_lats,
        stop_lons,
    )

    adj_offsets, adj_neighbors, adj_weights, adj_trip_ids = _build_adjacency(
        stop_times_array, trip_offsets, len(stop_ids)
    )

    (
        route_stop_offsets,
        route_stops,
        route_trip_offsets,
        route_trips,
        route_board_offsets,
        route_board_times,
        route_board_monotonic,
        stop_route_offsets,
        stop_routes,
    ) = _build_routes(stop_times_array, trip_offsets, len(stop_ids))

    transfer_offsets, transfer_neighbors, transfer_weights = _build_transfers(
        stop_ids,
        stop_lats,
        stop_lons,
    )

    return TransitNetwork(
        stops=stops_array,
        stop_times=stop_times_array,
        routes=routes_array,
        stop_id_index=stop_id_index,
        trip_id_index=trip_id_index,
        stop_ids=stop_ids,
        stop_lats=stop_lats,
        stop_lons=stop_lons,
        trip_ids=trip_ids,
        trip_offsets=trip_offsets,
        adj_offsets=adj_offsets,
        adj_neighbors=adj_neighbors,
        adj_weights=adj_weights,
        adj_trip_ids=adj_trip_ids,
        route_stop_offsets=route_stop_offsets,
        route_stops=route_stops,
        route_trip_offsets=route_trip_offsets,
        route_trips=route_trips,
        route_board_offsets=route_board_offsets,
        route_board_times=route_board_times,
        route_board_monotonic=route_board_monotonic,
        stop_route_offsets=stop_route_offsets,
        stop_routes=stop_routes,
        trip_route_types=trip_route_types,
        trip_cost_factors=trip_cost_factors,
        transfer_offsets=transfer_offsets,
        transfer_neighbors=transfer_neighbors,
        transfer_weights=transfer_weights,
    )
