from src.loader import build_mock_network
from src.solver import run_raptor, build_path


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
