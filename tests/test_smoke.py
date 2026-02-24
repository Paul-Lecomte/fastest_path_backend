from src.loader import build_mock_network
from src.solver import run_raptor, build_path


def test_smoke_mock_network():
    network = build_mock_network()
    earliest = run_raptor(network.stop_times, 0, 2, 900)
    segments = build_path(network.stop_times, 2, earliest)

    assert segments
    assert segments[0][2] == 1100

