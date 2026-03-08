# This module implements a simple HTTP server that provides a pathfinding API for transit networks.
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from datetime import datetime

import numpy as np

from .config import setup_logging
from .loader import NetworkLoader, build_mock_network, TransitNetwork
from .solver import build_path, build_path_dijkstra, run_dijkstra, run_raptor, run_astar


logger = logging.getLogger("pathfinding.http")

DEFAULT_OFFSET_MINUTES = (0, 10, 20, 30, 40)


def _count_transfers(segments) -> int:
    if not segments:
        return 0
    transfers = 0
    current_trip = segments[0][0]
    for trip_id, _, _ in segments[1:]:
        if trip_id != current_trip:
            transfers += 1
            current_trip = trip_id
    return transfers


def _departure_to_seconds(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    return None


def load_network() -> TransitNetwork:
    from .config import get_neo4j_config

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


def _compute_segments(
    network: TransitNetwork,
    algorithm: str,
    start_idx: int,
    end_idx: int,
    departure_time: int,
):
    if algorithm == "raptor":
        earliest, pred_stop, pred_trip, pred_time = run_raptor(
            network.stop_times,
            network.trip_offsets,
            start_idx,
            end_idx,
            departure_time,
        )
        return build_path(
            network.stop_times,
            network.trip_offsets,
            end_idx,
            earliest,
            pred_stop,
            pred_trip,
            pred_time,
        )
    if algorithm == "dijkstra":
        dist, pred_stop, pred_trip = run_dijkstra(
            network.adj_offsets,
            network.adj_neighbors,
            network.adj_weights,
            network.adj_trip_ids,
            start_idx,
            end_idx,
            departure_time,
        )
        return build_path_dijkstra(
            end_idx,
            dist,
            pred_stop,
            pred_trip,
        )
    if algorithm == "astar":
        heuristic = np.zeros(network.adj_offsets.shape[0] - 1, dtype=np.int64)
        dist, pred_stop, pred_trip = run_astar(
            network.adj_offsets,
            network.adj_neighbors,
            network.adj_weights,
            network.adj_trip_ids,
            start_idx,
            end_idx,
            departure_time,
            heuristic,
        )
        return build_path_dijkstra(
            end_idx,
            dist,
            pred_stop,
            pred_trip,
        )
    return None


def _build_option_response(
    network: TransitNetwork,
    algorithm: str,
    start_idx: int,
    end_idx: int,
    departure_time: int,
) -> dict[str, Any]:
    segments = _compute_segments(network, algorithm, start_idx, end_idx, departure_time)
    if segments is None:
        return {}
    return {
        "departure_time": int(departure_time),
        "transfers": _count_transfers(segments),
        "duration_seconds": int(segments[-1][2] - departure_time) if segments else None,
        "segments": [
            {
                "trip_id": network.trip_ids[trip_id],
                "stop_id": network.stop_ids[stop_id],
                "arrival_time": int(arrival_time),
            }
            for trip_id, stop_id, arrival_time in segments
        ],
    }


def build_multi_departure_response(
    network: TransitNetwork,
    algorithm: str,
    start_idx: int,
    end_idx: int,
    departure_time: int,
    offset_minutes: tuple[int, ...] = DEFAULT_OFFSET_MINUTES,
) -> dict[str, Any]:
    options = []
    for minutes in offset_minutes:
        offset_departure = departure_time + minutes * 60
        option = _build_option_response(
            network,
            algorithm,
            start_idx,
            end_idx,
            offset_departure,
        )
        options.append(option)

    base = options[0] if options else {
        "departure_time": int(departure_time),
        "transfers": 0,
        "duration_seconds": None,
        "segments": [],
    }
    return {
        "algorithm": algorithm,
        "transfers": base.get("transfers"),
        "duration_seconds": base.get("duration_seconds"),
        "segments": base.get("segments"),
        "options": options,
    }


class PathRequestHandler(BaseHTTPRequestHandler):
    network: TransitNetwork | None = None

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler signature
        if self.path != "/path":
            self._send_json(404, {"error": "not_found"})
            return

        payload = self._read_json()
        if payload is None:
            return

        start_stop_id = payload.get("start_stop_id")
        end_stop_id = payload.get("end_stop_id")
        departure_raw = payload.get("departure_time")
        algorithm = payload.get("algorithm", "raptor")

        if not isinstance(start_stop_id, str) or not isinstance(end_stop_id, str):
            self._send_json(400, {"error": "invalid_stop_id"})
            return

        if not isinstance(algorithm, str):
            self._send_json(400, {"error": "invalid_algorithm"})
            return
        algorithm = algorithm.strip().lower()

        departure_time = _departure_to_seconds(departure_raw)
        if departure_time is None:
            self._send_json(400, {"error": "invalid_departure_time"})
            return

        if self.network is None:
            self._send_json(500, {"error": "network_not_loaded"})
            return

        start_idx = self.network.stop_id_index.get(start_stop_id)
        end_idx = self.network.stop_id_index.get(end_stop_id)
        if start_idx is None or end_idx is None:
            self._send_json(404, {"error": "unknown_stop_id"})
            return

        if algorithm not in {"raptor", "dijkstra", "astar"}:
            self._send_json(400, {"error": "unsupported_algorithm"})
            return

        response = build_multi_departure_response(
            self.network,
            algorithm,
            start_idx,
            end_idx,
            departure_time,
        )
        logger.info(
            "HTTP /path algorithm=%s start=%s end=%s departure=%s options=%s",
            algorithm,
            start_stop_id,
            end_stop_id,
            departure_time,
            len(response.get("options", [])),
        )
        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match base signature
        logger.info("%s - %s", self.address_string(), format % args)

    def _read_json(self) -> dict[str, Any] | None:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send_json(400, {"error": "missing_content_length"})
            return None
        try:
            length = int(length_header)
        except ValueError:
            self._send_json(400, {"error": "invalid_content_length"})
            return None

        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid_json"})
            return None
        if not isinstance(data, dict):
            self._send_json(400, {"error": "invalid_payload"})
            return None
        return data

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    setup_logging()
    server = ThreadingHTTPServer((host, port), PathRequestHandler)
    PathRequestHandler.network = load_network()
    logger.info("HTTP server listening on %s:%s", host, port)
    return server


def main() -> None:
    server = serve()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down HTTP server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

