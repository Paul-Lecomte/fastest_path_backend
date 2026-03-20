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
from .solver import build_path, build_path_dijkstra, run_dijkstra_fast, run_raptor, run_astar_fast


logger = logging.getLogger("pathfinding.server")

SUPPORTED_ALGORITHMS = {"raptor", "dijkstra", "astar"}


def _compute_segments(network, algorithm: str, start_stop_id: int, end_stop_id: int, departure_time: int):
    if algorithm == "raptor":
        earliest, pred_stop, pred_trip, pred_time = run_raptor(
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
            start_stop_id,
            end_stop_id,
            departure_time,
        )

        return build_path(
            network.stop_times,
            network.trip_offsets,
            end_stop_id,
            earliest,
            pred_stop,
            pred_trip,
            pred_time,
        )
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
    if len(start_stop_ids) == 1:
        return _compute_segments(network, algorithm, start_stop_ids[0], end_stop_id, departure_time)

    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            None,
            _compute_segments,
            network,
            algorithm,
            start_stop_id,
            end_stop_id,
            departure_time,
        )
        for start_stop_id in start_stop_ids
    ]
    results = await asyncio.gather(*tasks)

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
            response.segments.add(
                trip_id=self.network.trip_ids[trip_id],
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
    config = get_neo4j_config()
    loader = NetworkLoader(config["uri"], config["user"], config["password"])
    try:
        logger.info("Loading network from Neo4j at %s", config["uri"])
        network = loader.fetch_to_numpy()
        logger.info(
            "Loaded network stops=%s stop_times=%s routes=%s",
            network.stops.shape[0],
            network.stop_times.shape[0],
            network.routes.shape[0],
        )
        return network
    except Exception as exc:
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
