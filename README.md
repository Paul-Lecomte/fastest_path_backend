# RAPTOR Pathfinding Microservice (Python)

High-performance pathfinding microservice using:

- Neo4j for graph storage
- NumPy for contiguous in-memory arrays
- Numba to compile the RAPTOR loop
- gRPC for low-latency communication

## HTTP API

Single endpoint:

- `POST /path`
- JSON body: `{"start_stop_id":"A","end_stop_id":"C","departure_time":900}`

Response:

```json
{"segments":[{"trip_id":"T1","stop_id":"C","arrival_time":1100}]}
```

`departure_time` accepts either:

- `int` seconds since midnight
- ISO-8601 datetime string, e.g. `"2026-02-25T08:13:00"`

## Quick start

1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

2) Generate gRPC code

```powershell
python -m grpc_tools.protoc -I proto --python_out=src --grpc_python_out=src proto/pathfinding.proto
```

3) Run the server (single command)

```powershell
python -m src.main
```

## Logs

The log level is configured via `LOG_LEVEL` (e.g., `DEBUG`, `INFO`).

## Notes on Numba / Python 3.13

Numba does not yet support Python 3.13. The code runs without Numba (pure Python mode),
but for JIT acceleration, use Python 3.12 and install `numba`.

## Configuration

Optional environment variables:

- `NEO4J_URI` (e.g., `neo4j://127.0.0.1:7687`)
- `NEO4J_USER`
- `NEO4J_PASSWORD`

You can also create a `.env` file at the project root containing these variables.

Without Neo4j, the server loads a small dummy network for testing.

Neo4j stop times are parsed from `HH:MM:SS` or numeric values into seconds.

## Tests

```powershell
python -m pytest -q
```

## Test with Postman

- Method: `POST`
- URL: `http://localhost:8080/path`
- Body: `raw` -> `JSON`

```json
{
  "start_stop_id": "A",
  "end_stop_id": "C",
  "departure_time": "2026-02-25T08:13:00"
}
```

## Test with curl

```powershell
curl -Method Post -Uri http://localhost:8080/path -ContentType application/json -Body '{"start_stop_id":"A","end_stop_id":"C","departure_time":900}'
```
