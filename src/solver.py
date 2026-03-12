# This module implements the core algorithms for route planning, including RAPTOR, Dijkstra's algorithm, and A* search.
# The RAPTOR algorithm is implemented in the run_raptor function, which computes the earliest arrival times at each stop given a departure time and a transit network. The Dijkstra and A* algorithms are implemented in the run_dijkstra and run_astar functions, respectively, which compute the shortest path in a graph representation of the transit network. The build_path and build_path_dijkstra functions reconstruct the path from the predecessor information computed by the algorithms.
from __future__ import annotations

import numpy as np
import heapq

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when numba is unavailable
    _NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):
        def wrapper(func):
            return func

        return wrapper


@njit(cache=True)
def run_raptor(
    stop_times,
    trip_offsets,
    route_stop_offsets,
    route_stops,
    route_trip_offsets,
    route_trips,
    stop_route_offsets,
    stop_routes,
    start_stop_id: int,
    end_stop_id: int,
    departure_time: int,
    max_rounds: int = 6,
):
    n_stops = stop_route_offsets.shape[0] - 1
    n_routes = route_stop_offsets.shape[0] - 1

    inf = np.int64(2**62)
    earliest = np.full(n_stops, inf, dtype=np.int64)
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)
    pred_time = np.full(n_stops, -1, dtype=np.int64)

    marked = np.zeros(n_stops, dtype=np.uint8)
    new_marked = np.zeros(n_stops, dtype=np.uint8)
    route_marked = np.zeros(n_routes, dtype=np.uint8)

    earliest[start_stop_id] = departure_time
    marked[start_stop_id] = 1

    best_target = inf

    for _round in range(max_rounds):
        for stop_id in range(n_stops):
            if marked[stop_id] == 0:
                continue
            start = stop_route_offsets[stop_id]
            end = stop_route_offsets[stop_id + 1]
            for idx in range(start, end):
                route_id = stop_routes[idx]
                route_marked[route_id] = 1

        improved = False
        for route_id in range(n_routes):
            if route_marked[route_id] == 0:
                continue
            route_marked[route_id] = 0

            r_start = route_stop_offsets[route_id]
            r_end = route_stop_offsets[route_id + 1]
            t_start = route_trip_offsets[route_id]
            t_end = route_trip_offsets[route_id + 1]
            if r_start >= r_end or t_start >= t_end:
                continue

            current_trip_idx = -1
            for s_idx in range(r_end - r_start):
                stop_id = route_stops[r_start + s_idx]

                if marked[stop_id] != 0:
                    if best_target != inf and earliest[stop_id] >= best_target:
                        pass
                    else:
                        if current_trip_idx == -1:
                            cand_idx = t_start
                            while cand_idx < t_end:
                                trip_id = route_trips[cand_idx]
                                st_idx = trip_offsets[trip_id] + s_idx
                                if stop_times[st_idx][2] >= earliest[stop_id]:
                                    current_trip_idx = cand_idx
                                    break
                                cand_idx += 1

                if current_trip_idx == -1:
                    continue

                trip_id = route_trips[current_trip_idx]
                st_idx = trip_offsets[trip_id] + s_idx
                arrival_time = stop_times[st_idx][2]

                if best_target != inf and arrival_time >= best_target:
                    break

                if arrival_time < earliest[stop_id]:
                    earliest[stop_id] = arrival_time
                    pred_stop[stop_id] = route_stops[r_start + s_idx - 1] if s_idx > 0 else stop_id
                    pred_trip[stop_id] = trip_id
                    pred_time[stop_id] = arrival_time
                    new_marked[stop_id] = 1
                    improved = True
                    if stop_id == end_stop_id:
                        best_target = arrival_time

        if not improved:
            break

        for stop_id in range(n_stops):
            marked[stop_id] = new_marked[stop_id]
            new_marked[stop_id] = 0

    return earliest, pred_stop, pred_trip, pred_time


def _run_dijkstra_heap(
    adj_offsets,
    adj_neighbors,
    adj_weights,
    adj_trip_ids,
    start_stop_id: int,
    end_stop_id: int,
    departure_time: int,
):
    n_stops = adj_offsets.shape[0] - 1
    inf = 2**62
    dist = np.full(n_stops, inf, dtype=np.int64)
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)

    dist[start_stop_id] = departure_time
    heap = [(departure_time, start_stop_id)]

    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist != dist[u]:
            continue
        if u == end_stop_id:
            break

        start = adj_offsets[u]
        end = adj_offsets[u + 1]
        for idx in range(start, end):
            v = adj_neighbors[idx]
            alt = current_dist + adj_weights[idx]
            if alt < dist[v]:
                dist[v] = alt
                pred_stop[v] = u
                pred_trip[v] = adj_trip_ids[idx]
                heapq.heappush(heap, (alt, v))

    return dist, pred_stop, pred_trip


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


def run_dijkstra_fast(adj_offsets, adj_neighbors, adj_weights, adj_trip_ids, start_stop_id: int, end_stop_id: int, departure_time: int):
    if _NUMBA_AVAILABLE:
        return run_dijkstra(
            adj_offsets,
            adj_neighbors,
            adj_weights,
            adj_trip_ids,
            start_stop_id,
            end_stop_id,
            departure_time,
        )
    return _run_dijkstra_heap(
        adj_offsets,
        adj_neighbors,
        adj_weights,
        adj_trip_ids,
        start_stop_id,
        end_stop_id,
        departure_time,
    )


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


def _run_astar_heap(
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
    inf = 2**62
    dist = np.full(n_stops, inf, dtype=np.int64)
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)

    dist[start_stop_id] = departure_time
    heap = [(departure_time + int(heuristic[start_stop_id]), departure_time, start_stop_id)]

    while heap:
        fscore, current_dist, u = heapq.heappop(heap)
        if current_dist != dist[u]:
            continue
        if u == end_stop_id:
            break

        start = adj_offsets[u]
        end = adj_offsets[u + 1]
        for idx in range(start, end):
            v = adj_neighbors[idx]
            alt = current_dist + adj_weights[idx]
            if alt < dist[v]:
                dist[v] = alt
                pred_stop[v] = u
                pred_trip[v] = adj_trip_ids[idx]
                heapq.heappush(heap, (alt + int(heuristic[v]), alt, v))

    return dist, pred_stop, pred_trip


def run_astar_fast(
    adj_offsets,
    adj_neighbors,
    adj_weights,
    adj_trip_ids,
    start_stop_id: int,
    end_stop_id: int,
    departure_time: int,
    heuristic,
):
    if _NUMBA_AVAILABLE:
        return run_astar(
            adj_offsets,
            adj_neighbors,
            adj_weights,
            adj_trip_ids,
            start_stop_id,
            end_stop_id,
            departure_time,
            heuristic,
        )
    return _run_astar_heap(
        adj_offsets,
        adj_neighbors,
        adj_weights,
        adj_trip_ids,
        start_stop_id,
        end_stop_id,
        departure_time,
        heuristic,
    )


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

