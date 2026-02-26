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
def run_raptor(stop_times, trip_offsets, start_stop_id: int, end_stop_id: int, departure_time: int, max_rounds: int = 6):
    n_stops = 0
    for i in range(stop_times.shape[0]):
        if stop_times[i][0] + 1 > n_stops:
            n_stops = stop_times[i][0] + 1

    inf = np.int64(2**62)
    earliest = np.full(n_stops, inf, dtype=np.int64)
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)
    pred_time = np.full(n_stops, -1, dtype=np.int64)

    earliest[start_stop_id] = departure_time

    for _round in range(max_rounds):
        improved = False
        for trip_id in range(trip_offsets.shape[0] - 1):
            start = trip_offsets[trip_id]
            end = trip_offsets[trip_id + 1]
            if start >= end:
                continue

            boarded = False
            for i in range(start, end):
                stop_id = stop_times[i][0]
                arrival_time = stop_times[i][2]
                if not boarded:
                    if earliest[stop_id] <= arrival_time:
                        boarded = True
                    else:
                        continue

                if arrival_time < earliest[stop_id]:
                    earliest[stop_id] = arrival_time
                    pred_stop[stop_id] = stop_times[i - 1][0] if i > start else stop_id
                    pred_trip[stop_id] = trip_id
                    pred_time[stop_id] = arrival_time
                    improved = True

        if not improved:
            break

    return earliest, pred_stop, pred_trip, pred_time


def build_path(stop_times, trip_offsets, end_stop_id: int, earliest, pred_stop, pred_trip, pred_time):
    segments = []
    best_time = earliest[end_stop_id]
    if best_time <= 0 or best_time >= 2**61:
        return segments

    current_stop = end_stop_id
    while current_stop != -1:
        trip_id = int(pred_trip[current_stop])
        arrival_time = int(pred_time[current_stop])
        if trip_id == -1:
            break
        segments.append((trip_id, current_stop, arrival_time))
        current_stop = int(pred_stop[current_stop])

    segments.reverse()
    return segments
