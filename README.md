# RAPTOR Pathfinding Microservice (Python)

Microservice de pathfinding haute performance utilisant:

- Neo4j pour le stockage graphe
- NumPy pour la mise en memoire contigue
- Numba pour compiler la boucle RAPTOR
- gRPC pour la communication faible latence

## API HTTP

Endpoint unique:

- `POST /path`
- Corps JSON: `{"start_stop_id":"A","end_stop_id":"C","departure_time":900}`

Reponse:

```json
{"segments":[{"trip_id":"T1","stop_id":"C","arrival_time":1100}]}
```

## Demarrage rapide

1) Installer les dependances

```powershell
python -m pip install -r requirements.txt
```

2) Generer le code gRPC

```powershell
python -m grpc_tools.protoc -I proto --python_out=src --grpc_python_out=src proto/pathfinding.proto
```

3) Lancer le serveur (commande unique)

```powershell
python -m src.main
```

## Logs

Le niveau de logs est configure via `LOG_LEVEL` (ex: `DEBUG`, `INFO`).

## Remarques sur Numba / Python 3.13

Numba ne supporte pas encore Python 3.13. Le code fonctionne sans Numba (mode Python pur),
mais pour l'acceleration JIT, utilisez Python 3.12 et installez `numba`.

## Configuration

Variables d'environnement optionnelles:

- `NEO4J_URI` (ex: `neo4j://127.0.0.1:7687`)
- `NEO4J_USER`
- `NEO4J_PASSWORD`

Vous pouvez aussi creer un fichier `.env` a la racine du projet avec ces variables.

Sans Neo4j, le serveur charge un petit reseau factice pour test.

## Tests

```powershell
python -m pytest -q
```

## Tester avec Postman

- Methode: `POST`
- URL: `http://localhost:8080/path`
- Body: `raw` -> `JSON`

```json
{
  "start_stop_id": "A",
  "end_stop_id": "C",
  "departure_time": 900
}
```

## Tester avec curl

```powershell
curl -Method Post -Uri http://localhost:8080/path -ContentType application/json -Body '{"start_stop_id":"A","end_stop_id":"C","departure_time":900}'
```
