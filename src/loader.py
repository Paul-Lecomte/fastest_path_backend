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
    trip_ids: List[str]
    trip_offsets: np.ndarray


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
            "st.stop_sequence AS stop_sequence"
        )

        stop_ids: List[str] = []
        trip_ids: List[str] = []
        stop_id_index: Dict[str, int] = {}
        trip_id_index: Dict[str, int] = {}
        rows: List[Tuple[int, int, int, int]] = []

        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                stop_id = record["stop_id"]
                trip_id = record["trip_id"]
                arrival_time = record["arrival_time"]
                stop_sequence = record["stop_sequence"]
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

        return TransitNetwork(
            stops=stops_array,
            stop_times=stop_times_array,
            routes=routes_array,
            stop_id_index=stop_id_index,
            trip_id_index=trip_id_index,
            stop_ids=stop_ids,
            trip_ids=trip_ids,
            trip_offsets=trip_offsets,
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
    trip_id_index = {"T1": 0, "T2": 1}
    trip_offsets = np.array([0, 2, 4], dtype=np.int64)

    return TransitNetwork(
        stops=stops_array,
        stop_times=stop_times_array,
        routes=routes_array,
        stop_id_index=stop_id_index,
        trip_id_index=trip_id_index,
        stop_ids=stop_ids,
        trip_ids=trip_ids,
        trip_offsets=trip_offsets,
    )
