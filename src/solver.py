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


def numba_enabled() -> bool:
    return bool(_NUMBA_AVAILABLE)


@njit(cache=True)
def _lower_bound_int64(values, start: int, end: int, target: int):
    left = start
    right = end
    while left < right:
        mid = (left + right) // 2
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


@njit(cache=True)
def run_raptor_with_stats(
    stop_times,
    trip_offsets,
    route_stop_offsets,
    route_stops,
    route_trip_offsets,
    route_trips,
    route_board_offsets,
    route_board_times,
    route_board_monotonic,
    stop_route_offsets,
    stop_routes,
    transfer_offsets,
    transfer_neighbors,
    transfer_weights,
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
    marked_list = np.empty(n_stops, dtype=np.int64)
    new_marked_list = np.empty(n_stops, dtype=np.int64)
    route_marked = np.zeros(n_routes, dtype=np.uint8)
    route_marked_list = np.empty(n_routes, dtype=np.int64)
    route_min_marked_time = np.full(n_routes, inf, dtype=np.int64)

    earliest[start_stop_id] = departure_time
    marked[start_stop_id] = 1
    marked_list[0] = np.int64(start_stop_id)
    marked_count = 1

    best_target = inf
    rounds_used = np.int64(0)

    for _round in range(max_rounds):
        rounds_used += 1
        if marked_count == 0:
            break

        transfer_seed_idx = 0
        while transfer_seed_idx < marked_count:
            from_stop = marked_list[transfer_seed_idx]
            transfer_seed_idx += 1
            base_time = earliest[from_stop]
            transfer_start = transfer_offsets[from_stop]
            transfer_end = transfer_offsets[from_stop + 1]
            for transfer_idx in range(transfer_start, transfer_end):
                to_stop = transfer_neighbors[transfer_idx]
                transfer_time = transfer_weights[transfer_idx]
                arrival_via_transfer = base_time + transfer_time
                if best_target != inf and arrival_via_transfer >= best_target:
                    continue
                if arrival_via_transfer < earliest[to_stop]:
                    earliest[to_stop] = arrival_via_transfer
                    pred_stop[to_stop] = from_stop
                    pred_trip[to_stop] = -2
                    pred_time[to_stop] = arrival_via_transfer
                    if marked[to_stop] == 0:
                        marked[to_stop] = 1
                        marked_list[marked_count] = to_stop
                        marked_count += 1
                    if to_stop == end_stop_id:
                        best_target = arrival_via_transfer

        route_marked_count = 0
        for i in range(marked_count):
            stop_id = marked_list[i]
            start = stop_route_offsets[stop_id]
            end = stop_route_offsets[stop_id + 1]
            for idx in range(start, end):
                route_id = stop_routes[idx]
                if route_marked[route_id] == 0:
                    route_marked[route_id] = 1
                    route_marked_list[route_marked_count] = route_id
                    route_marked_count += 1
                    route_min_marked_time[route_id] = earliest[stop_id]
                elif earliest[stop_id] < route_min_marked_time[route_id]:
                    route_min_marked_time[route_id] = earliest[stop_id]

        improved = False
        new_marked_count = 0
        for i in range(route_marked_count):
            route_id = route_marked_list[i]
            route_marked[route_id] = 0

            if best_target != inf and route_min_marked_time[route_id] >= best_target:
                route_min_marked_time[route_id] = inf
                continue

            route_min_marked_time[route_id] = inf

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
                        board_offset = route_board_offsets[r_start + s_idx]
                        board_end = route_board_offsets[r_start + s_idx + 1]
                        candidate_trip_idx = -1
                        candidate_arrival = inf
                        if route_board_monotonic[r_start + s_idx] != 0:
                            candidate = _lower_bound_int64(
                                route_board_times,
                                board_offset,
                                board_end,
                                earliest[stop_id],
                            )
                            if candidate < board_end:
                                candidate_trip_idx = t_start + (candidate - board_offset)
                                candidate_arrival = route_board_times[candidate]
                        else:
                            candidate = board_offset
                            while candidate < board_end:
                                value = route_board_times[candidate]
                                if value >= earliest[stop_id] and value < candidate_arrival:
                                    candidate_arrival = value
                                    candidate_trip_idx = t_start + (candidate - board_offset)
                                candidate += 1

                        if candidate_trip_idx != -1:
                            if current_trip_idx == -1:
                                current_trip_idx = candidate_trip_idx
                            else:
                                current_trip_id = route_trips[current_trip_idx]
                                current_arrival = stop_times[trip_offsets[current_trip_id] + s_idx][2]
                                if candidate_arrival < current_arrival:
                                    current_trip_idx = candidate_trip_idx

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
                    if new_marked[stop_id] == 0:
                        new_marked[stop_id] = 1
                        new_marked_list[new_marked_count] = stop_id
                        new_marked_count += 1
                    improved = True
                    if stop_id == end_stop_id:
                        best_target = arrival_time

        transfer_cursor = 0
        while transfer_cursor < new_marked_count:
            from_stop = new_marked_list[transfer_cursor]
            transfer_cursor += 1
            base_time = earliest[from_stop]
            transfer_start = transfer_offsets[from_stop]
            transfer_end = transfer_offsets[from_stop + 1]
            for transfer_idx in range(transfer_start, transfer_end):
                to_stop = transfer_neighbors[transfer_idx]
                transfer_time = transfer_weights[transfer_idx]
                arrival_via_transfer = base_time + transfer_time
                if best_target != inf and arrival_via_transfer >= best_target:
                    continue
                if arrival_via_transfer < earliest[to_stop]:
                    earliest[to_stop] = arrival_via_transfer
                    pred_stop[to_stop] = from_stop
                    pred_trip[to_stop] = -2
                    pred_time[to_stop] = arrival_via_transfer
                    if new_marked[to_stop] == 0:
                        new_marked[to_stop] = 1
                        new_marked_list[new_marked_count] = to_stop
                        new_marked_count += 1
                    improved = True
                    if to_stop == end_stop_id:
                        best_target = arrival_via_transfer

        if not improved:
            break

        for i in range(marked_count):
            stop_id = marked_list[i]
            marked[stop_id] = 0

        for i in range(new_marked_count):
            stop_id = new_marked_list[i]
            marked[stop_id] = 1
            new_marked[stop_id] = 0
            marked_list[i] = stop_id

        marked_count = new_marked_count

    reached_target = np.uint8(0)
    if end_stop_id >= 0 and end_stop_id < n_stops:
        reached_target = np.uint8(1 if earliest[end_stop_id] < inf else 0)
    return earliest, pred_stop, pred_trip, pred_time, rounds_used, marked_count, reached_target


def run_raptor(
    stop_times,
    trip_offsets,
    route_stop_offsets,
    route_stops,
    route_trip_offsets,
    route_trips,
    route_board_offsets,
    route_board_times,
    route_board_monotonic,
    stop_route_offsets,
    stop_routes,
    transfer_offsets,
    transfer_neighbors,
    transfer_weights,
    start_stop_id: int,
    end_stop_id: int,
    departure_time: int,
    max_rounds: int = 6,
):
    earliest, pred_stop, pred_trip, pred_time, _, _, _ = run_raptor_with_stats(
        stop_times,
        trip_offsets,
        route_stop_offsets,
        route_stops,
        route_trip_offsets,
        route_trips,
        route_board_offsets,
        route_board_times,
        route_board_monotonic,
        stop_route_offsets,
        stop_routes,
        transfer_offsets,
        transfer_neighbors,
        transfer_weights,
        start_stop_id,
        end_stop_id,
        departure_time,
        max_rounds,
    )
    return earliest, pred_stop, pred_trip, pred_time


@njit(cache=True)
def _heap_push2(keys, vals, size: int, key: int, val: int):
    i = size
    keys[i] = np.int64(key)
    vals[i] = np.int64(val)
    while i > 0:
        p = (i - 1) // 2
        if keys[p] <= keys[i]:
            break
        tk = keys[p]
        tv = vals[p]
        keys[p] = keys[i]
        vals[p] = vals[i]
        keys[i] = tk
        vals[i] = tv
        i = p
    return size + 1


@njit(cache=True)
def _heap_pop2(keys, vals, size: int):
    root_key = keys[0]
    root_val = vals[0]
    size -= 1
    if size > 0:
        keys[0] = keys[size]
        vals[0] = vals[size]
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            if left >= size:
                break
            child = left
            if right < size and keys[right] < keys[left]:
                child = right
            if keys[i] <= keys[child]:
                break
            tk = keys[i]
            tv = vals[i]
            keys[i] = keys[child]
            vals[i] = vals[child]
            keys[child] = tk
            vals[child] = tv
            i = child
    return root_key, root_val, size


@njit(cache=True)
def _heap_push3(keys, v1, v2, size: int, key: int, a: int, b: int):
    i = size
    keys[i] = np.int64(key)
    v1[i] = np.int64(a)
    v2[i] = np.int64(b)
    while i > 0:
        p = (i - 1) // 2
        if keys[p] <= keys[i]:
            break
        tk = keys[p]
        ta = v1[p]
        tb = v2[p]
        keys[p] = keys[i]
        v1[p] = v1[i]
        v2[p] = v2[i]
        keys[i] = tk
        v1[i] = ta
        v2[i] = tb
        i = p
    return size + 1


@njit(cache=True)
def _heap_pop3(keys, v1, v2, size: int):
    root_key = keys[0]
    root_a = v1[0]
    root_b = v2[0]
    size -= 1
    if size > 0:
        keys[0] = keys[size]
        v1[0] = v1[size]
        v2[0] = v2[size]
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            if left >= size:
                break
            child = left
            if right < size and keys[right] < keys[left]:
                child = right
            if keys[i] <= keys[child]:
                break
            tk = keys[i]
            ta = v1[i]
            tb = v2[i]
            keys[i] = keys[child]
            v1[i] = v1[child]
            v2[i] = v2[child]
            keys[child] = tk
            v1[child] = ta
            v2[child] = tb
            i = child
    return root_key, root_a, root_b, size


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
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)

    heap_keys = np.empty(n_stops, dtype=np.int64)
    heap_nodes = np.empty(n_stops, dtype=np.int64)
    heap_size = 0

    start = np.int64(start_stop_id)
    target = np.int64(end_stop_id)
    start_t = np.int64(departure_time)

    dist[start] = start_t
    heap_size = _heap_push2(heap_keys, heap_nodes, heap_size, start_t, start)

    while heap_size > 0:
        current_dist, u, heap_size = _heap_pop2(heap_keys, heap_nodes, heap_size)
        if current_dist != dist[u]:
            continue
        if u == target:
            break

        row_start = adj_offsets[u]
        row_end = adj_offsets[u + 1]
        for idx in range(row_start, row_end):
            v = adj_neighbors[idx]
            alt = current_dist + adj_weights[idx]
            if alt < dist[v]:
                dist[v] = alt
                pred_stop[v] = u
                pred_trip[v] = adj_trip_ids[idx]
                heap_size = _heap_push2(heap_keys, heap_nodes, heap_size, alt, v)

    return dist, pred_stop, pred_trip


def run_dijkstra_fast(adj_offsets, adj_neighbors, adj_weights, adj_trip_ids, start_stop_id: int, end_stop_id: int, departure_time: int):
    return run_dijkstra(
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
    pred_stop = np.full(n_stops, -1, dtype=np.int64)
    pred_trip = np.full(n_stops, -1, dtype=np.int64)

    heap_keys = np.empty(n_stops, dtype=np.int64)
    heap_g = np.empty(n_stops, dtype=np.int64)
    heap_nodes = np.empty(n_stops, dtype=np.int64)
    heap_size = 0

    start = np.int64(start_stop_id)
    target = np.int64(end_stop_id)
    start_t = np.int64(departure_time)

    dist[start] = start_t
    f0 = start_t + heuristic[start]
    fscore[start] = f0
    heap_size = _heap_push3(heap_keys, heap_g, heap_nodes, heap_size, f0, start_t, start)

    while heap_size > 0:
        _, current_dist, u, heap_size = _heap_pop3(heap_keys, heap_g, heap_nodes, heap_size)
        if current_dist != dist[u]:
            continue
        if u == target:
            break

        row_start = adj_offsets[u]
        row_end = adj_offsets[u + 1]
        for idx in range(row_start, row_end):
            v = adj_neighbors[idx]
            alt = current_dist + adj_weights[idx]
            if alt < dist[v]:
                dist[v] = alt
                fv = alt + heuristic[v]
                fscore[v] = fv
                pred_stop[v] = u
                pred_trip[v] = adj_trip_ids[idx]
                heap_size = _heap_push3(heap_keys, heap_g, heap_nodes, heap_size, fv, alt, v)

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
    h = np.ascontiguousarray(heuristic, dtype=np.int64)
    return run_astar(
        adj_offsets,
        adj_neighbors,
        adj_weights,
        adj_trip_ids,
        start_stop_id,
        end_stop_id,
        departure_time,
        h,
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

