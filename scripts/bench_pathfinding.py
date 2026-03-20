# Simple benchmark for pathfinding performance.
from __future__ import annotations

import argparse
import time

import numpy as np

from src.http_server import load_network
from src.solver import run_dijkstra_fast, run_astar_fast, run_raptor


def _bench(label, fn, repeat: int, warmup: bool) -> None:
    if warmup:
        warm_start = time.perf_counter()
        fn()
        warm_elapsed = time.perf_counter() - warm_start
        print(f"{label}: warmup={warm_elapsed:.3f}s")

    durations = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - started)

    total = sum(durations)
    avg = total / repeat if repeat > 0 else 0.0
    first = durations[0] if durations else 0.0
    steady = sum(durations[1:]) / max(1, len(durations) - 1) if durations else 0.0
    print(
        f"{label}: total={total:.3f}s avg={avg:.3f}s first={first:.3f}s steady={steady:.3f}s runs={len(durations)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--departure", type=int, default=28800)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    load_started = time.perf_counter()
    network = load_network()
    load_elapsed = time.perf_counter() - load_started
    print(f"network_load: {load_elapsed:.3f}s")
    if args.start is None or args.end is None:
        start_idx = 0
        end_idx = min(10, network.adj_offsets.shape[0] - 2)
    else:
        start_idx = network.stop_id_index.get(args.start)
        end_idx = network.stop_id_index.get(args.end)
        if start_idx is None or end_idx is None:
            raise SystemExit("Unknown stop_id")

    heuristic = np.zeros(network.adj_offsets.shape[0] - 1, dtype=np.int64)

    def run_dijkstra():
        run_dijkstra_fast(
            network.adj_offsets,
            network.adj_neighbors,
            network.adj_weights,
            network.adj_trip_ids,
            start_idx,
            end_idx,
            args.departure,
        )

    def run_astar():
        run_astar_fast(
            network.adj_offsets,
            network.adj_neighbors,
            network.adj_weights,
            network.adj_trip_ids,
            start_idx,
            end_idx,
            args.departure,
            heuristic,
        )

    def run_raptor_algo():
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
            args.departure,
        )

    _bench("dijkstra", run_dijkstra, args.repeat, args.warmup)
    _bench("astar", run_astar, args.repeat, args.warmup)
    _bench("raptor", run_raptor_algo, args.repeat, args.warmup)


if __name__ == "__main__":
    main()

