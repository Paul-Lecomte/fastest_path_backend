# This file contains smoke tests for the RAPTOR, Dijkstra, and A* algorithms on a mock transportation network.
import numpy as np

from src.loader import build_mock_network
from src.solver import build_path, build_path_dijkstra, run_dijkstra_fast, run_raptor, run_astar_fast
from src.http_server import build_multi_departure_response


def test_smoke_mock_network():
    network = build_mock_network()
    earliest, pred_stop, pred_trip, pred_time = run_raptor(
        network.stop_times,
        network.trip_offsets,
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
        network.stop_id_index["A"],
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
