from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    def njit(*_args, **_kwargs):
        def wrapper(func):
            return func

        return wrapper


@njit(cache=True)
def run_raptor(stop_times, start_stop_id: int, end_stop_id: int, departure_time: int):
    n_stops = 0
    for i in range(stop_times.shape[0]):
        if stop_times[i][0] + 1 > n_stops:
            n_stops = stop_times[i][0] + 1

    inf = np.int64(2**62)
    earliest = np.full(n_stops, inf, dtype=np.int64)
    earliest[start_stop_id] = departure_time

    # Minimal scan: pick the earliest arrival at the target stop after departure.
    best_arrival = inf
    for i in range(stop_times.shape[0]):
        stop_id = stop_times[i][0]
        arrival_time = stop_times[i][2]
        if stop_id == end_stop_id and arrival_time >= departure_time and arrival_time < best_arrival:
            best_arrival = arrival_time

    if best_arrival < earliest[end_stop_id]:
        earliest[end_stop_id] = best_arrival

    return earliest


def build_path(stop_times, end_stop_id: int, earliest):
    segments = []
    best_time = earliest[end_stop_id]
    if best_time <= 0 or best_time >= 2**61:
        return segments

    for row in stop_times:
        if row[0] == end_stop_id and row[2] == best_time:
            segments.append((row[1], row[0], row[2]))
            break

    return segments
