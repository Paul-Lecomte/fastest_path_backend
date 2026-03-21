from __future__ import annotations

import argparse
import random
import sys

from src.http_server import build_multi_departure_response, load_network


def _build_od_pairs(n_stops: int, sample_size: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []

    if n_stops <= 1:
        return pairs

    for _ in range(sample_size):
        start = rng.randrange(0, n_stops)
        end = rng.randrange(0, n_stops)
        while end == start:
            end = rng.randrange(0, n_stops)
        pairs.append((start, end))

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--departure", type=int, default=28800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-found-rate", type=float, default=0.95)
    parser.add_argument("--max-fallback-rate", type=float, default=0.80)
    args = parser.parse_args()

    network = load_network()
    n_stops = int(network.adj_offsets.shape[0] - 1)
    if n_stops <= 1:
        print("insufficient network stops to run reliability check", file=sys.stderr)
        return 2

    pairs = _build_od_pairs(n_stops, max(1, args.sample_size), args.seed)

    found = 0
    fallback_used = 0
    tested = 0
    for start_idx, end_idx in pairs:
        response = build_multi_departure_response(
            network,
            "raptor",
            [int(start_idx)],
            int(end_idx),
            int(args.departure),
            offset_minutes=(0,),
        )
        tested += 1

        segments = response.get("segments") or []
        if segments:
            found += 1
        if bool(response.get("fallback_used", False)):
            fallback_used += 1

    found_rate = found / tested if tested else 0.0
    fallback_rate = fallback_used / tested if tested else 0.0

    print(
        f"reliability tested={tested} found={found} found_rate={found_rate:.3f} "
        f"fallback_used={fallback_used} fallback_rate={fallback_rate:.3f}"
    )

    if found_rate < args.min_found_rate:
        print(
            f"FAILED found_rate threshold: {found_rate:.3f} < {args.min_found_rate:.3f}",
            file=sys.stderr,
        )
        return 1

    if fallback_rate > args.max_fallback_rate:
        print(
            f"FAILED fallback_rate threshold: {fallback_rate:.3f} > {args.max_fallback_rate:.3f}",
            file=sys.stderr,
        )
        return 1

    print("PASSED reliability thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
