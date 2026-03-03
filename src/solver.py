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


@njit(cache=True)
def run_dijkstra(adj_offsets, adj_neighbors, adj_weights, adj_trip_ids, start_stop_id: int, end_stop_id: int, departure_time: int):
    n_stops = adj_offsets.shape[0] - 1
    inf = np.int64(2**62)
    dist = np.full(n_stops, inf, dtype=np.int64)
    visited = np.zeros(n_stops, dtype=np.uint8)
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)

    dist[start_stop_id] = departure_time

    for _ in range(n_stops):
        best = inf
        u = -1
        for i in range(n_stops):
            if visited[i] == 0 and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1 or u == end_stop_id:
            break
        visited[u] = 1

        start = adj_offsets[u]
        end = adj_offsets[u + 1]
        for idx in range(start, end):
            v = adj_neighbors[idx]
            w = adj_weights[idx]
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                pred_stop[v] = u
                pred_trip[v] = adj_trip_ids[idx]

    return dist, pred_stop, pred_trip


@njit(cache=True)
def run_astar(
    adj_offsets,
    adj_neighbors,
    adj_weights,
    adj_trip_ids,
    start_stop_id: int,
    end_stop_id: int,
    departure_time: int,
    heuristic,
):
    n_stops = adj_offsets.shape[0] - 1
    inf = np.int64(2**62)
    dist = np.full(n_stops, inf, dtype=np.int64)
    fscore = np.full(n_stops, inf, dtype=np.int64)
    visited = np.zeros(n_stops, dtype=np.uint8)
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)

    dist[start_stop_id] = departure_time
    fscore[start_stop_id] = departure_time + heuristic[start_stop_id]

    for _ in range(n_stops):
        best = inf
        u = -1
        for i in range(n_stops):
            if visited[i] == 0 and fscore[i] < best:
                best = fscore[i]
                u = i
        if u == -1 or u == end_stop_id:
            break
        visited[u] = 1

        start = adj_offsets[u]
        end = adj_offsets[u + 1]
        for idx in range(start, end):
            v = adj_neighbors[idx]
            w = adj_weights[idx]
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                fscore[v] = alt + heuristic[v]
                pred_stop[v] = u
                pred_trip[v] = adj_trip_ids[idx]

    return dist, pred_stop, pred_trip


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


def build_path_dijkstra(end_stop_id: int, dist, pred_stop, pred_trip):
    segments = []
    best_time = dist[end_stop_id]
    if best_time <= 0 or best_time >= 2**61:
        return segments

    current_stop = end_stop_id
    while current_stop != -1:
        trip_id = int(pred_trip[current_stop])
        if trip_id == -1:
            break
        segments.append((trip_id, current_stop, int(dist[current_stop])))
        current_stop = int(pred_stop[current_stop])

    segments.reverse()
    return segments

