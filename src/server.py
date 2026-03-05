# A gRPC server that provides a route search service using the pathfinding algorithms implemented in solver.py.
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import grpc
import numpy as np

from . import pathfinding_pb2, pathfinding_pb2_grpc
from .config import get_neo4j_config, setup_logging
from .loader import NetworkLoader, build_mock_network
from .solver import build_path, build_path_dijkstra, run_dijkstra, run_raptor, run_astar


logger = logging.getLogger("pathfinding.server")


class RouteSearchServicer(pathfinding_pb2_grpc.RouteSearchServicer):
    def __init__(self, network):
        self.network = network

    async def GetFastestPath(self, request, context):
        logger.info(
            "GetFastestPath request start=%s end=%s departure=%s",
            request.start_stop_id,
            request.end_stop_id,
            request.departure_time,
        )
        start_stop_id = self.network.stop_id_index.get(request.start_stop_id)
        end_stop_id = self.network.stop_id_index.get(request.end_stop_id)

        if start_stop_id is None or end_stop_id is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Unknown stop_id")

        algorithm = getattr(request, "algorithm", "raptor") or "raptor"
        algorithm = algorithm.strip().lower()

        if algorithm == "raptor":
            earliest, pred_stop, pred_trip, pred_time = run_raptor(
                self.network.stop_times,
                self.network.trip_offsets,
                start_stop_id,
                end_stop_id,
                request.departure_time,
            )

            segments = build_path(
                self.network.stop_times,
                self.network.trip_offsets,
                end_stop_id,
                earliest,
                pred_stop,
                pred_trip,
                pred_time,
            )
        elif algorithm == "dijkstra":
            dist, pred_stop, pred_trip = run_dijkstra(
                self.network.adj_offsets,
                self.network.adj_neighbors,
                self.network.adj_weights,
                self.network.adj_trip_ids,
                start_stop_id,
                end_stop_id,
                request.departure_time,
            )
            segments = build_path_dijkstra(
                end_stop_id,
                dist,
                pred_stop,
                pred_trip,
            )
        elif algorithm == "astar":
            heuristic = np.zeros(self.network.adj_offsets.shape[0] - 1, dtype=np.int64)
            dist, pred_stop, pred_trip = run_astar(
                self.network.adj_offsets,
                self.network.adj_neighbors,
                self.network.adj_weights,
                self.network.adj_trip_ids,
                start_stop_id,
                end_stop_id,
                request.departure_time,
                heuristic,
            )
            segments = build_path_dijkstra(
                end_stop_id,
                dist,
                pred_stop,
                pred_trip,
            )
        else:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported algorithm")

        response = pathfinding_pb2.PathResponse()
        for trip_id, stop_id, arrival_time in segments:
            response.segments.add(
                trip_id=self.network.trip_ids[trip_id],
                stop_id=self.network.stop_ids[stop_id],
                arrival_time=int(arrival_time),
            )

        logger.info("GetFastestPath response segments=%s", len(segments))
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
