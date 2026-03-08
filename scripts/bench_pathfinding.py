# Simple benchmark for pathfinding performance.
from __future__ import annotations

import argparse
import time

import numpy as np

from src.http_server import load_network
from src.solver import run_dijkstra_fast, run_astar_fast, run_raptor


def _bench(label, fn, repeat: int) -> None:
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    elapsed = time.perf_counter() - start
    avg = elapsed / repeat
    print(f"{label}: total={elapsed:.3f}s avg={avg:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--departure", type=int, default=28800)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    network = load_network()
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
            start_idx,
            end_idx,
            args.departure,
        )

    _bench("dijkstra", run_dijkstra, args.repeat)
    _bench("astar", run_astar, args.repeat)
    _bench("raptor", run_raptor_algo, args.repeat)


if __name__ == "__main__":
    main()

