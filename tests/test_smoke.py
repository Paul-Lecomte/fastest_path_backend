from src.loader import build_mock_network
from src.solver import build_path, build_path_dijkstra, run_dijkstra, run_raptor


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
    dist, pred_stop, pred_trip = run_dijkstra(
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
    assert segments[-1][2] == 1050
