# This file contains smoke tests for the RAPTOR, Dijkstra, and A* algorithms on a mock transportation network.
import asyncio
from types import SimpleNamespace

import numpy as np

from src.loader import build_mock_network
from src.solver import build_path, build_path_dijkstra, run_dijkstra_fast, run_raptor, run_astar_fast
from src.http_server import build_multi_departure_response, _departure_to_seconds, _select_starts_from_origin
from src.server import _find_fastest_segments_parallel, _get_start_stop_ids


def test_smoke_mock_network():
    network = build_mock_network()
    earliest, pred_stop, pred_trip, pred_time = run_raptor(
        network.stop_times,
        network.trip_offsets,
        network.route_stop_offsets,
        network.route_stops,
        network.route_trip_offsets,
        network.route_trips,
        network.stop_route_offsets,
        network.stop_routes,
        0,
        2,
        900,
    )
    segments = build_path(
        network.stop_times,
        network.trip_offsets,
        2,
        earliest,
        pred_stop,
        pred_trip,
        pred_time,
    )

    assert segments
    assert segments[-1][2] == 1100


def test_smoke_mock_network_dijkstra():
    network = build_mock_network()
    dist, pred_stop, pred_trip = run_dijkstra_fast(
        network.adj_offsets,
        network.adj_neighbors,
        network.adj_weights,
        network.adj_trip_ids,
        0,
        2,
        900,
    )
    segments = build_path_dijkstra(
        2,
        dist,
        pred_stop,
        pred_trip,
    )

    assert segments
    assert segments[-1][2] == 990


def test_smoke_mock_network_astar():
    network = build_mock_network()
    heuristic = np.zeros(network.adj_offsets.shape[0] - 1, dtype=np.int64)
    dist, pred_stop, pred_trip = run_astar_fast(
        network.adj_offsets,
        network.adj_neighbors,
        network.adj_weights,
        network.adj_trip_ids,
        0,
        2,
        900,
        heuristic,
    )
    segments = build_path_dijkstra(
        2,
        dist,
        pred_stop,
        pred_trip,
    )

    assert segments
    assert segments[-1][2] == 990


def test_multi_departure_response_offsets():
    network = build_mock_network()
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["A"]],
        network.stop_id_index["C"],
        900,
    )

    assert response["options"]
    assert len(response["options"]) == 5
    assert response["options"][0]["departure_time"] == 900
    assert response["options"][1]["departure_time"] == 1500
    assert response["options"][2]["departure_time"] == 2100
    assert response["options"][3]["departure_time"] == 2700
    assert response["options"][4]["departure_time"] == 3300
    assert response["options"][0]["segments"]


def test_http_multi_start_selects_fastest_path():
    network = build_mock_network()
    response = build_multi_departure_response(
        network,
        "dijkstra",
        [network.stop_id_index["A"], network.stop_id_index["B"]],
        network.stop_id_index["C"],
        900,
        offset_minutes=(0,),
    )

    assert response["segments"]
    assert response["segments"][-1]["arrival_time"] == 940


def test_http_departure_parses_numeric_string_timestamp():
    assert _departure_to_seconds("1738580100") == 1738580100


def test_http_raptor_falls_back_to_dijkstra_when_no_schedule_path():
    network = build_mock_network()
    response = build_multi_departure_response(
        network,
        "raptor",
        [network.stop_id_index["A"]],
        network.stop_id_index["C"],
        2000,
        offset_minutes=(0,),
    )

    assert response["segments"]
    assert response["segments"][-1]["arrival_time"] == 2090


def test_origin_selects_candidate_starts_and_penalties():
    network = build_mock_network()
    start_indices, penalties, metadata = _select_starts_from_origin(
        network,
        {"lat": 48.8566, "lon": 2.3522, "radius_m": 300, "max_candidates": 2},
    )

    assert start_indices
    assert len(start_indices) <= 2
    assert isinstance(metadata, dict)
    assert metadata["candidate_start_count"] == len(start_indices)
    assert start_indices[0] == network.stop_id_index["A"]
    assert penalties[start_indices[0]] <= 1


def test_http_origin_route_without_explicit_start_ids():
    network = build_mock_network()
    start_indices, penalties, metadata = _select_starts_from_origin(
        network,
        {"lat": 48.8566, "lon": 2.3522, "radius_m": 1200, "max_candidates": 3},
    )
    response = build_multi_departure_response(
        network,
        "dijkstra",
        start_indices,
        network.stop_id_index["C"],
        900,
        offset_minutes=(0,),
        start_penalties=penalties,
        metadata=metadata,
    )

    assert response["segments"]
    assert response["start_stop_id"] in network.stop_id_index
    assert response["candidate_start_count"] == len(start_indices)


def test_get_start_stop_ids_prefers_repeated_field():
    request = SimpleNamespace(start_stop_id="A", start_stop_ids=[" B ", "A", "", "B"])
    assert _get_start_stop_ids(request) == ["B", "A"]


def test_parallel_multi_start_selects_fastest_path():
    network = build_mock_network()
    start_a = network.stop_id_index["A"]
    start_b = network.stop_id_index["B"]
    end_c = network.stop_id_index["C"]

    segments = asyncio.run(
        _find_fastest_segments_parallel(
            network,
            "dijkstra",
            [start_a, start_b],
            end_c,
            900,
        )
    )

    assert segments
    assert segments[-1][2] == 940
