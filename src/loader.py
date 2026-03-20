# This module defines the NetworkLoader class, which is responsible for loading transit network data from a Neo4j database and converting it into a format suitable for pathfinding algorithms. The loader fetches stop times, routes, and stops from the database, processes the data to build an adjacency list representation of the transit network, and returns a TransitNetwork object containing all the relevant information.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
import numpy as np
from neo4j import GraphDatabase


logger = logging.getLogger("pathfinding.loader")


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
            "coalesce(s.stop_lat, s.latitude, s.lat) AS stop_lat, "
            "coalesce(s.stop_lon, s.longitude, s.lon) AS stop_lon"
        )

        stop_ids: List[str] = []
        trip_ids: List[str] = []
        stop_id_index: Dict[str, int] = {}
        trip_id_index: Dict[str, int] = {}
        stop_coords: Dict[int, Tuple[float, float]] = {}
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
        )


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
    )
