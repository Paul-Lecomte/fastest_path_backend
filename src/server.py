# A gRPC server that provides a route search service using the pathfinding algorithms implemented in solver.py.
# The server loads the transit network data from a Neo4j database using the NetworkLoader class, and falls back to a mock network if the database load fails. The gRPC service defines a GetFastestPath method that accepts a PathRequest and returns a PathResponse containing the fastest path segments between the specified start and end stops at the given departure time, using the requested algorithm (RAPTOR, Dijkstra, or A*).
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import grpc
import numpy as np

from . import pathfinding_pb2, pathfinding_pb2_grpc
from .config import get_neo4j_config, setup_logging
from .loader import NetworkLoader, build_mock_network
from .solver import build_path, build_path_dijkstra, run_dijkstra_fast, run_raptor_with_stats, run_astar_fast


logger = logging.getLogger("pathfinding.server")

SUPPORTED_ALGORITHMS = {"raptor", "dijkstra", "astar"}
RAPTOR_ROUND_BUDGETS = (8, 16, 32, 64)


def _algorithm_sequence(primary: str) -> tuple[str, ...]:
    if primary == "raptor":
        return ("raptor", "astar", "dijkstra")
    if primary == "astar":
        return ("astar", "dijkstra")
    return ("dijkstra",)


def _compute_segments(network, algorithm: str, start_stop_id: int, end_stop_id: int, departure_time: int):
    if algorithm == "raptor":
        for max_rounds in RAPTOR_ROUND_BUDGETS:
            earliest, pred_stop, pred_trip, pred_time, _, _, _ = run_raptor_with_stats(
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
                network.transfer_offsets,
                network.transfer_neighbors,
                network.transfer_weights,
                start_stop_id,
                end_stop_id,
                departure_time,
                max_rounds=max_rounds,
            )

            segments = build_path(
                network.stop_times,
                network.trip_offsets,
                end_stop_id,
                earliest,
                pred_stop,
                pred_trip,
                pred_time,
            )
            if segments:
                return segments
        return []
    if algorithm == "dijkstra":
        dist, pred_stop, pred_trip = run_dijkstra_fast(
            network.adj_offsets,
            network.adj_neighbors,
            network.adj_weights,
            network.adj_trip_ids,
            start_stop_id,
            end_stop_id,
            departure_time,
        )
        return build_path_dijkstra(
            end_stop_id,
            dist,
            pred_stop,
            pred_trip,
        )
    heuristic = np.zeros(network.adj_offsets.shape[0] - 1, dtype=np.int64)
    dist, pred_stop, pred_trip = run_astar_fast(
        network.adj_offsets,
        network.adj_neighbors,
        network.adj_weights,
        network.adj_trip_ids,
        start_stop_id,
        end_stop_id,
        departure_time,
        heuristic,
    )
    return build_path_dijkstra(
        end_stop_id,
        dist,
        pred_stop,
        pred_trip,
    )


def _get_start_stop_ids(request) -> list[str]:
    start_stop_ids = [stop_id.strip() for stop_id in getattr(request, "start_stop_ids", []) if stop_id and stop_id.strip()]
    if not start_stop_ids:
        single_start = getattr(request, "start_stop_id", "")
        if single_start and single_start.strip():
            start_stop_ids = [single_start.strip()]

    unique_start_stop_ids = []
    seen = set()
    for stop_id in start_stop_ids:
        if stop_id in seen:
            continue
        seen.add(stop_id)
        unique_start_stop_ids.append(stop_id)
    return unique_start_stop_ids


async def _find_fastest_segments_parallel(network, algorithm: str, start_stop_ids: list[int], end_stop_id: int, departure_time: int):
    sequence = _algorithm_sequence(algorithm)
    if len(start_stop_ids) == 1:
        for selected_algorithm in sequence:
            segments = _compute_segments(network, selected_algorithm, start_stop_ids[0], end_stop_id, departure_time)
            if segments:
                return segments
        return []

    loop = asyncio.get_running_loop()
    results = []
    for selected_algorithm in sequence:
        tasks = [
            loop.run_in_executor(
                None,
                _compute_segments,
                network,
                selected_algorithm,
                start_stop_id,
                end_stop_id,
                departure_time,
            )
            for start_stop_id in start_stop_ids
        ]
        results = await asyncio.gather(*tasks)

        has_path = False
        for segments in results:
            if segments:
                has_path = True
                break
        if has_path:
            break

    best_segments = []
    best_arrival = None
    for segments in results:
        if not segments:
            continue
        arrival_time = int(segments[-1][2])
        if best_arrival is None or arrival_time < best_arrival:
            best_arrival = arrival_time
            best_segments = segments
    return best_segments


class RouteSearchServicer(pathfinding_pb2_grpc.RouteSearchServicer):
    def __init__(self, network):
        self.network = network

    async def GetFastestPath(self, request, context):
        logger.info(
            "GetFastestPath request start=%s starts=%s end=%s departure=%s",
            request.start_stop_id,
            len(getattr(request, "start_stop_ids", [])),
            request.end_stop_id,
            request.departure_time,
        )

        algorithm = getattr(request, "algorithm", "raptor") or "raptor"
        algorithm = algorithm.strip().lower()
        if algorithm not in SUPPORTED_ALGORITHMS:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported algorithm")

        request_start_stop_ids = _get_start_stop_ids(request)
        if not request_start_stop_ids:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing start_stop_id or start_stop_ids")

        start_stop_ids = []
        unknown_stops = []
        for stop_id in request_start_stop_ids:
            start_stop_id = self.network.stop_id_index.get(stop_id)
            if start_stop_id is None:
                unknown_stops.append(stop_id)
                continue
            start_stop_ids.append(start_stop_id)

        end_stop_id = self.network.stop_id_index.get(request.end_stop_id)
        if unknown_stops or end_stop_id is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Unknown stop_id")

        segments = await _find_fastest_segments_parallel(
            self.network,
            algorithm,
            start_stop_ids,
            end_stop_id,
            request.departure_time,
        )

        response = pathfinding_pb2.PathResponse()
        for trip_id, stop_id, arrival_time in segments:
            trip_value = int(trip_id)
            trip_label = "TRANSFER" if trip_value < 0 else self.network.trip_ids[trip_value]
            response.segments.add(
                trip_id=trip_label,
                stop_id=self.network.stop_ids[stop_id],
                arrival_time=int(arrival_time),
            )

        logger.info(
            "GetFastestPath response starts=%s segments=%s",
            len(start_stop_ids),
            len(segments),
        )
        return response


def load_network():
    from .config import get_network_cache_config
    from .loader import load_network_from_cache, save_network_to_cache

    config = get_neo4j_config()
    cache_config = get_network_cache_config()
    cache_path = cache_config["path"]

    if cache_config["enabled"] and not cache_config["force_refresh"]:
        cached_network = load_network_from_cache(cache_path, cache_config["max_age_seconds"])
        if cached_network is not None:
            logger.info("Using cached network from %s", cache_path)
            return cached_network

    loader = NetworkLoader(config["uri"], config["user"], config["password"])
    try:
        logger.info("Loading network from Neo4j at %s", config["uri"])
        network = loader.fetch_to_numpy()
        if cache_config["enabled"]:
            save_network_to_cache(cache_path, network)
        logger.info(
            "Loaded network stops=%s stop_times=%s routes=%s",
            network.stops.shape[0],
            network.stop_times.shape[0],
            network.routes.shape[0],
        )
        return network
    except Exception as exc:
        if cache_config["enabled"]:
            cached_network = load_network_from_cache(cache_path)
            if cached_network is not None:
                logger.warning("Neo4j load failed, using cached network path=%s error=%s", cache_path, exc)
                return cached_network
        logger.warning("Neo4j load failed, using mock network: %s", exc)
        return build_mock_network()
    finally:
        loader.close()


def serve():
    setup_logging()
    network = load_network()
    server = grpc.aio.server(ThreadPoolExecutor(max_workers=4))
    pathfinding_pb2_grpc.add_RouteSearchServicer_to_server(RouteSearchServicer(network), server)
    server.add_insecure_port("[::]:50051")
    logger.info("gRPC server listening on :50051")
    return server


async def main():
    server = serve()
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(main())
