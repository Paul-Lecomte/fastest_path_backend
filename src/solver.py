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
    trip_cost_factors,
    transfer_penalty_seconds: int,
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
    cost_scale = np.int64(1000)
    default_trip_factor = np.int64(1000)
    walk_factor_num = np.int64(3)
    walk_factor_den = np.int64(2)
    # Keep walk-feasibility caps aligned with inflated transfer edge weights.
    # Base policy (180s / 600s) is scaled by the same realism multiplier used in loader/http.
    walk_time_multiplier = np.int64(15)
    max_transfer_walk_seconds = np.int64(180) * walk_time_multiplier
    max_total_walk_seconds = np.int64(600) * walk_time_multiplier
    transfer_penalty_cost = np.int64(max(0, int(transfer_penalty_seconds))) * cost_scale
    transfer_walk_penalty_cost = transfer_penalty_cost
    transfer_board_buffer_seconds = np.int64(min(10, max(0, int(transfer_penalty_seconds)) // 60))
    earliest = np.full(n_stops, inf, dtype=np.int64)
    best_cost = np.full(n_stops, inf, dtype=np.int64)
    best_walk_seconds = np.full(n_stops, inf, dtype=np.int64)
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
    best_cost[start_stop_id] = np.int64(departure_time) * cost_scale
    best_walk_seconds[start_stop_id] = np.int64(0)
    marked[start_stop_id] = 1
    marked_list[0] = np.int64(start_stop_id)
    marked_count = 1

    best_target = inf

    # Initial one-hop access walk from the origin stop before route scanning.
    transfer_start = transfer_offsets[start_stop_id]
    transfer_end = transfer_offsets[start_stop_id + 1]
    for transfer_idx in range(transfer_start, transfer_end):
        to_stop = transfer_neighbors[transfer_idx]
        transfer_time = transfer_weights[transfer_idx]
        if transfer_time > max_transfer_walk_seconds:
            continue
        next_walk = np.int64(transfer_time)
        if next_walk > max_total_walk_seconds:
            continue

        weighted_walk_time = (np.int64(transfer_time) * walk_factor_num + walk_factor_den - 1) // walk_factor_den
        arrival_via_transfer = np.int64(departure_time) + np.int64(transfer_time)
        transfer_cost = (
            np.int64(departure_time) * cost_scale
            + weighted_walk_time * cost_scale
            + transfer_walk_penalty_cost
        )
        should_update = transfer_cost < best_cost[to_stop] or (
            transfer_cost == best_cost[to_stop] and (
                next_walk < best_walk_seconds[to_stop]
                or (next_walk == best_walk_seconds[to_stop] and arrival_via_transfer < earliest[to_stop])
            )
        )
        if should_update:
            earliest[to_stop] = arrival_via_transfer
            best_cost[to_stop] = transfer_cost
            best_walk_seconds[to_stop] = next_walk
            pred_stop[to_stop] = start_stop_id
            pred_trip[to_stop] = -2
            pred_time[to_stop] = arrival_via_transfer
            if marked[to_stop] == 0:
                marked[to_stop] = 1
                marked_list[marked_count] = to_stop
                marked_count += 1
            if to_stop == end_stop_id:
                best_target = arrival_via_transfer

    rounds_used = np.int64(0)

    for _round in range(max_rounds):
        rounds_used += 1
        if marked_count == 0:
            break

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

            route_min_marked_time[route_id] = inf

            r_start = route_stop_offsets[route_id]
            r_end = route_stop_offsets[route_id + 1]
            t_start = route_trip_offsets[route_id]
            t_end = route_trip_offsets[route_id + 1]
            if r_start >= r_end or t_start >= t_end:
                continue

            current_trip_idx = -1
            current_board_time = inf
            current_board_cost = inf
            current_board_walk_seconds = inf
            for s_idx in range(r_end - r_start):
                stop_id = route_stops[r_start + s_idx]

                if marked[stop_id] != 0:
                    board_ready_time = earliest[stop_id]
                    if _round > 0:
                        board_ready_time += transfer_board_buffer_seconds
                    board_offset = route_board_offsets[r_start + s_idx]
                    board_end = route_board_offsets[r_start + s_idx + 1]
                    candidate_trip_idx = -1
                    candidate_arrival = inf
                    candidate_board_cost = inf
                    candidate_board_walk_seconds = inf
                    search_cursor = board_offset
                    if route_board_monotonic[r_start + s_idx] != 0:
                        search_cursor = _lower_bound_int64(
                            route_board_times,
                            board_offset,
                            board_end,
                            board_ready_time,
                        )

                    candidate_scan_limit = 6
                    scanned = 0
                    while search_cursor < board_end and (route_board_monotonic[r_start + s_idx] == 0 or scanned < candidate_scan_limit):
                        value = route_board_times[search_cursor]
                        if value >= board_ready_time:
                            trip_idx = t_start + (search_cursor - board_offset)
                            trip_id_for_candidate = route_trips[trip_idx]
                            trip_factor = default_trip_factor
                            if trip_id_for_candidate >= 0 and trip_id_for_candidate < trip_cost_factors.shape[0]:
                                trip_factor = trip_cost_factors[trip_id_for_candidate]
                            wait_time = value - board_ready_time
                            board_cost = best_cost[stop_id] + np.int64(wait_time) * cost_scale
                            board_walk_seconds = best_walk_seconds[stop_id]
                            if _round > 0:
                                board_cost += transfer_penalty_cost

                            downstream_idx = s_idx + 1
                            if downstream_idx >= (r_end - r_start):
                                downstream_idx = s_idx
                            downstream_arrival = stop_times[trip_offsets[trip_id_for_candidate] + downstream_idx][2]
                            in_vehicle_preview = downstream_arrival - value
                            if in_vehicle_preview < 0:
                                in_vehicle_preview = 0
                            candidate_score = board_cost + np.int64(in_vehicle_preview) * np.int64(trip_factor)

                            best_candidate_score = inf
                            if candidate_trip_idx != -1:
                                chosen_trip_id = route_trips[candidate_trip_idx]
                                chosen_factor = default_trip_factor
                                if chosen_trip_id >= 0 and chosen_trip_id < trip_cost_factors.shape[0]:
                                    chosen_factor = trip_cost_factors[chosen_trip_id]
                                chosen_downstream = stop_times[trip_offsets[chosen_trip_id] + downstream_idx][2]
                                chosen_preview = chosen_downstream - candidate_arrival
                                if chosen_preview < 0:
                                    chosen_preview = 0
                                best_candidate_score = candidate_board_cost + np.int64(chosen_preview) * np.int64(chosen_factor)

                            if candidate_trip_idx == -1 or candidate_score < best_candidate_score:
                                candidate_trip_idx = trip_idx
                                candidate_arrival = value
                                candidate_board_cost = board_cost
                                candidate_board_walk_seconds = board_walk_seconds

                        search_cursor += 1
                        scanned += 1

                    if candidate_trip_idx != -1:
                        if current_trip_idx == -1:
                            current_trip_idx = candidate_trip_idx
                            current_board_time = candidate_arrival
                            current_board_cost = candidate_board_cost
                            current_board_walk_seconds = candidate_board_walk_seconds
                        else:
                            current_trip_id = route_trips[current_trip_idx]
                            current_trip_factor = default_trip_factor
                            if current_trip_id >= 0 and current_trip_id < trip_cost_factors.shape[0]:
                                current_trip_factor = trip_cost_factors[current_trip_id]
                            current_arrival = stop_times[trip_offsets[current_trip_id] + s_idx][2]
                            current_in_vehicle = current_arrival - current_board_time
                            if current_in_vehicle < 0:
                                current_in_vehicle = 0
                            current_score = current_board_cost + np.int64(current_in_vehicle) * np.int64(current_trip_factor)

                            candidate_trip_id = route_trips[candidate_trip_idx]
                            candidate_trip_factor = default_trip_factor
                            if candidate_trip_id >= 0 and candidate_trip_id < trip_cost_factors.shape[0]:
                                candidate_trip_factor = trip_cost_factors[candidate_trip_id]
                            candidate_in_vehicle = 0
                            candidate_score = candidate_board_cost + np.int64(candidate_in_vehicle) * np.int64(candidate_trip_factor)

                            if candidate_score < current_score:
                                current_trip_idx = candidate_trip_idx
                                current_board_time = candidate_arrival
                                current_board_cost = candidate_board_cost
                                current_board_walk_seconds = candidate_board_walk_seconds

                if current_trip_idx == -1:
                    continue

                trip_id = route_trips[current_trip_idx]
                st_idx = trip_offsets[trip_id] + s_idx
                arrival_time = stop_times[st_idx][2]

                trip_factor = default_trip_factor
                if trip_id >= 0 and trip_id < trip_cost_factors.shape[0]:
                    trip_factor = trip_cost_factors[trip_id]
                in_vehicle_time = arrival_time - current_board_time
                if in_vehicle_time < 0:
                    in_vehicle_time = 0
                arrival_cost = current_board_cost + np.int64(in_vehicle_time) * np.int64(trip_factor)
                arrival_walk_seconds = current_board_walk_seconds

                should_update = arrival_cost < best_cost[stop_id] or (
                    arrival_cost == best_cost[stop_id] and (
                        arrival_walk_seconds < best_walk_seconds[stop_id]
                        or (arrival_walk_seconds == best_walk_seconds[stop_id] and arrival_time < earliest[stop_id])
                    )
                )

                if should_update:
                    earliest[stop_id] = arrival_time
                    best_cost[stop_id] = arrival_cost
                    best_walk_seconds[stop_id] = arrival_walk_seconds
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
            # Only expand walking transfers from stops reached by in-vehicle transit.
            # This prevents chaining walk->walk edges across rounds.
            if pred_trip[from_stop] < 0:
                continue
            base_time = earliest[from_stop]
            transfer_start = transfer_offsets[from_stop]
            transfer_end = transfer_offsets[from_stop + 1]
            for transfer_idx in range(transfer_start, transfer_end):
                to_stop = transfer_neighbors[transfer_idx]
                transfer_time = transfer_weights[transfer_idx]
                if transfer_time > max_transfer_walk_seconds:
                    continue
                next_walk_seconds = best_walk_seconds[from_stop] + np.int64(transfer_time)
                if next_walk_seconds > max_total_walk_seconds:
                    continue
                arrival_via_transfer = base_time + transfer_time
                weighted_walk_time = (np.int64(transfer_time) * walk_factor_num + walk_factor_den - 1) // walk_factor_den
                transfer_cost = (
                    best_cost[from_stop]
                    + weighted_walk_time * cost_scale
                    + transfer_walk_penalty_cost
                )
                should_update = transfer_cost < best_cost[to_stop] or (
                    transfer_cost == best_cost[to_stop] and (
                        next_walk_seconds < best_walk_seconds[to_stop]
                        or (next_walk_seconds == best_walk_seconds[to_stop] and arrival_via_transfer < earliest[to_stop])
                    )
                )
                if should_update:
                    earliest[to_stop] = arrival_via_transfer
                    best_cost[to_stop] = transfer_cost
                    best_walk_seconds[to_stop] = next_walk_seconds
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
    trip_cost_factors,
    transfer_penalty_seconds: int,
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
        trip_cost_factors,
        transfer_penalty_seconds,
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

