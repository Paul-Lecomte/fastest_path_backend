<!-- PROJECT TITLE & BADGES -->
<p align="center">
  <img src="./public/fastest_path_logo.png" alt="FastestPath Logo" width="120" />
</p>
<h1 align="center">RAPTOR Pathfinding Microservice</h1>
<p align="center">
  <strong>High-performance transit route planning API (Python, RAPTOR)</strong><br>
  <img alt="Tech Stack" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
  <img alt="Tech Stack" src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white">
  <img alt="Tech Stack" src="https://img.shields.io/badge/Numba-00A3E0?logo=python&logoColor=white">
  <img alt="Tech Stack" src="https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white">
  <img alt="Tech Stack" src="https://img.shields.io/badge/gRPC-3A76F0?logo=google&logoColor=white">
  <img alt="License" src="https://img.shields.io/github/license/Paul-Lecomte/fastest_path_backend">
</p>

## What is RAPTOR Pathfinding Microservice?

This project provides a high-performance backend service for public transit routing using a RAPTOR-based algorithm.
It loads GTFS-like data (via Neo4j or a dummy dataset), runs fast earliest-arrival searches, and exposes results via HTTP and gRPC.

---

## Features

- RAPTOR route planning with NumPy-backed arrays
- Optional Neo4j storage for graph and stop times
- Numba-accelerated core loop (when available)
- HTTP endpoint for fastest-path queries
- Multiple departure options (+10/20/30/40 minutes) computed in parallel

---

## Tech Stack

- Language: Python 3.13
- Acceleration: NumPy, Numba
- Storage: Neo4j (optional)
- APIs: HTTP + gRPC

---

## Project Structure

```bash
fastest_path_backend/
├── proto/
│   └── pathfinding.proto
├── scripts/
│   └── bench_pathfinding.py
├── src/
│   ├── config.py
│   ├── http_server.py
│   ├── loader.py
│   ├── main.py
│   ├── pathfinding_pb2.py
│   ├── pathfinding_pb2_grpc.py
│   ├── server.py
│   └── solver.py
├── tests/
│   └── test_smoke.py
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## HTTP API

Single endpoint:

- `POST /path`
- JSON body (single start): `{"start_stop_id":"A","end_stop_id":"C","departure_time":900}`
- JSON body (multiple starts): `{"start_stop_ids":["A","B"],"end_stop_id":"C","departure_time":900}`
- JSON body (GPS origin): `{"origin":{"lat":48.8566,"lon":2.3522,"radius_m":1000,"max_candidates":12},"end_stop_id":"C","departure_time":900}`
- JSON body (GPS origin + destination): `{"origin":{"lat":48.8566,"lon":2.3522,"radius_m":1000,"max_candidates":12},"destination":{"lat":48.8574,"lon":2.3540,"radius_m":1000,"max_candidates":12},"departure_time":"1773924420"}`

Response:

```json
{
  "algorithm": "raptor",
  "resolver_algorithm": "raptor",
  "fallback_used": false,
  "transfers": 0,
  "duration_seconds": 200,
  "segments": [
    {"trip_id": "T1", "stop_id": "C", "arrival_time": 1100, "lat": 48.8574, "lon": 2.354}
  ],
  "options": [
    {
      "departure_time": 900,
      "transfers": 0,
      "duration_seconds": 200,
      "segments": [
        {"trip_id": "T1", "stop_id": "C", "arrival_time": 1100, "lat": 48.8574, "lon": 2.354}
      ]
    },
    {
      "departure_time": 1500,
      "transfers": 0,
      "duration_seconds": null,
      "segments": []
    }
  ]
}
```

`resolver_algorithm` indicates the algorithm that actually produced the path.
When RAPTOR cannot find a path for a query, the service automatically falls back
to A* then Dijkstra for reliability (`fallback_used=true`).

Segments can include multiple trips when a transfer is required.
When `start_stop_ids` is provided, all starts are evaluated and the fastest resulting path is returned.
When `origin` is provided, nearby stops are selected automatically and scored with walking access time.
When `destination` is provided, nearby destination stops are selected automatically and scored with walking egress time.
Large numeric `departure_time` values (Unix timestamps) are normalized to seconds-of-day.
If no path is found with the selected algorithm, the HTTP endpoint expands nearby start stops to improve route discovery.

## gRPC API

`PathRequest` now supports either a single start stop (`start_stop_id`) or multiple start stops (`start_stop_ids`).
When `start_stop_ids` is provided, the server evaluates all starts in parallel and returns the fastest resulting path.

```proto
message PathRequest {
  string start_stop_id = 1;
  string end_stop_id = 2;
  int64 departure_time = 3;
  repeated string start_stop_ids = 4;
}
```

---

## Prerequisites

- Python 3.13
- Optional: Neo4j running locally or remotely

---

## Setup

1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

2) Generate gRPC code

```powershell
python -m grpc_tools.protoc -I proto --python_out=src --grpc_python_out=src proto/pathfinding.proto
```

3) Run the server

```powershell
python -m src.main
```

---

## Configuration

Optional environment variables:

- `NEO4J_URI` (e.g., `neo4j://127.0.0.1:7687`)
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `LOG_LEVEL` (e.g., `DEBUG`, `INFO`)
- `NETWORK_CACHE_ENABLED` (`true`/`false`, default: `true`)
- `NETWORK_CACHE_PATH` (default: `.cache/transit_network.pkl`)
- `NETWORK_CACHE_FORCE_REFRESH` (`true`/`false`, default: `false`)
- `NETWORK_CACHE_MAX_AGE_SECONDS` (default: `0` meaning no TTL)

You can also create a `.env` file at the project root containing these variables.

Without Neo4j, the server loads a small dummy network for testing.
Neo4j stop times are parsed from `HH:MM:SS` or numeric values into seconds.
When cache is enabled, the first successful Neo4j load is persisted to disk and
subsequent starts reuse the cached network for much faster startup.

---

## Notes on Numba / Python 3.13

Numba does not yet support Python 3.13. The code runs without Numba (pure Python mode),
but for JIT acceleration, use Python 3.12 and install dependencies from `requirements.txt`
(includes `numba` for Python < 3.13).

---

## Benchmark

```powershell
python -m scripts.bench_pathfinding --start A --end C --departure 28800 --repeat 3 --warmup
```

The benchmark now reports:
- `network_load` time
- per-algorithm `warmup` time (optional)
- `first` run cost and `steady` average for measured runs

Optional RAPTOR performance gate:

```powershell
python -m scripts.check_bench --repeat 5 --warmup --max-first 5.0 --max-steady 2.0
```

Optional long-trip reliability gate:

```powershell
python -m scripts.check_reliability --sample-size 128 --min-found-rate 0.95 --max-fallback-rate 0.80
```

---

## Tests

```powershell
python -m pytest -q
```

---

## Roadmap

- [x] Working RAPTOR implementation with real data from Neo4j
- [x] Add more detailed logging (e.g., query times, pathfinding times)
- [ ] Implement returned data for the frontend and a function to fetch trip geometry
- [ ] Implement more comprehensive error handling (e.g., invalid input, database errors)
---

## License

This project is open source. See the [LICENSE](./LICENSE) file for details.
