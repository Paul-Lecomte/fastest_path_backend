from __future__ import annotations

import argparse
import sys
import time

from src.http_server import load_network
from src.solver import run_raptor


def _measure_raptor(network, start_idx: int, end_idx: int, departure_time: int, repeat: int, warmup: bool) -> tuple[float, float]:
    def run_once() -> None:
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
            900,
            network.transfer_offsets,
            network.transfer_neighbors,
            network.transfer_weights,
            start_idx,
            end_idx,
            departure_time,
        )

    if warmup:
        run_once()

    durations = []
    for _ in range(repeat):
        started = time.perf_counter()
        run_once()
        durations.append(time.perf_counter() - started)

    first = durations[0] if durations else 0.0
    steady = sum(durations[1:]) / max(1, len(durations) - 1) if durations else 0.0
    return first, steady


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--departure", type=int, default=28800)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--max-first", type=float, default=5.0, help="Maximum allowed first-run seconds")
    parser.add_argument("--max-steady", type=float, default=2.0, help="Maximum allowed steady average seconds")
    args = parser.parse_args()

    network = load_network()
    if args.start is None or args.end is None:
        start_idx = 0
        end_idx = min(10, network.adj_offsets.shape[0] - 2)
    else:
        start_idx = network.stop_id_index.get(args.start)
        end_idx = network.stop_id_index.get(args.end)
        if start_idx is None or end_idx is None:
            print("Unknown stop_id")
            return 2

    first, steady = _measure_raptor(
        network,
        int(start_idx),
        int(end_idx),
        int(args.departure),
        int(args.repeat),
        bool(args.warmup),
    )

    print(f"raptor_gate first={first:.3f}s steady={steady:.3f}s")

    if first > args.max_first or steady > args.max_steady:
        print(
            f"FAILED threshold: first<={args.max_first:.3f}s steady<={args.max_steady:.3f}s",
            file=sys.stderr,
        )
        return 1

    print("PASSED threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
