from __future__ import annotations

import argparse
import heapq
import math
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.http_server import load_network
from src.loader import save_transfer_graph_to_cache


CHECKPOINT_VERSION = 1


def _save_checkpoint(
    checkpoint_path: Path,
    metadata: dict,
    edges_by_stop: list[dict[int, int]],
    done_source_node_ids: set,
) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "metadata": metadata,
        "edges_by_stop": edges_by_stop,
        "done_source_node_ids": list(done_source_node_ids),
        "saved_at": time.time(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_checkpoint(checkpoint_path: Path) -> dict | None:
    if not checkpoint_path.exists():
        return None
    try:
        with checkpoint_path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
            return None
        return payload
    except Exception:
        return None


def _load_osm_graph_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            return None
        if "graph" not in payload:
            return None
        return payload
    except Exception:
        return None


def _compute_stops_bbox(
    stop_lats: np.ndarray,
    stop_lons: np.ndarray,
    margin_m: float,
    trim_quantile: float = 0.005,
) -> list[float] | None:
    finite_mask = np.isfinite(stop_lats) & np.isfinite(stop_lons)
    if not np.any(finite_mask):
        return None

    lats = stop_lats[finite_mask]
    lons = stop_lons[finite_mask]
    q = max(0.0, min(0.2, float(trim_quantile)))
    if q > 0.0 and lats.size >= 100:
        low = q
        high = 1.0 - q
        min_lat = float(np.quantile(lats, low))
        max_lat = float(np.quantile(lats, high))
        min_lon = float(np.quantile(lons, low))
        max_lon = float(np.quantile(lons, high))
    else:
        min_lat = float(np.min(lats))
        max_lat = float(np.max(lats))
        min_lon = float(np.min(lons))
        max_lon = float(np.max(lons))

    margin_lat_deg = float(margin_m) / 111320.0
    mean_lat = 0.5 * (min_lat + max_lat)
    cos_lat = max(1e-6, math.cos(math.radians(mean_lat)))
    margin_lon_deg = float(margin_m) / (111320.0 * cos_lat)

    return [
        float(min_lon - margin_lon_deg),
        float(min_lat - margin_lat_deg),
        float(max_lon + margin_lon_deg),
        float(max_lat + margin_lat_deg),
    ]


def _save_osm_graph_cache(cache_path: Path, graph, nodes, edges) -> None:
    payload = {
        "graph": graph,
        "nodes": nodes,
        "edges": edges,
        "saved_at": time.time(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_compact_osm_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=True)
        required = {
            "node_ids",
            "node_lats",
            "node_lons",
            "adj_offsets",
            "adj_neighbors",
            "adj_weights",
        }
        if not required.issubset(set(payload.files)):
            return None
        return {
            "node_ids": payload["node_ids"],
            "node_lats": np.asarray(payload["node_lats"], dtype=np.float64),
            "node_lons": np.asarray(payload["node_lons"], dtype=np.float64),
            "adj_offsets": np.asarray(payload["adj_offsets"], dtype=np.int64),
            "adj_neighbors": np.asarray(payload["adj_neighbors"], dtype=np.int32),
            "adj_weights": np.asarray(payload["adj_weights"], dtype=np.int64),
        }
    except Exception:
        return None


def _save_compact_osm_cache(
    cache_path: Path,
    node_ids: list | np.ndarray,
    node_lats: np.ndarray,
    node_lons: np.ndarray,
    adj_offsets: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        node_ids=np.asarray(node_ids, dtype=object),
        node_lats=np.asarray(node_lats, dtype=np.float64),
        node_lons=np.asarray(node_lons, dtype=np.float64),
        adj_offsets=np.asarray(adj_offsets, dtype=np.int64),
        adj_neighbors=np.asarray(adj_neighbors, dtype=np.int32),
        adj_weights=np.asarray(adj_weights, dtype=np.int64),
    )


def _build_compact_osm_adjacency(graph, node_ids: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    osm_adj_edges = [dict() for _ in range(len(node_ids))]
    for u, v, data in graph.edges(data=True):
        if u not in node_id_to_idx or v not in node_id_to_idx:
            continue
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        walk_seconds = int(float(data.get("walk_seconds", 1.0)))
        if walk_seconds <= 0:
            walk_seconds = 1

        current_fw = osm_adj_edges[u_idx].get(v_idx)
        if current_fw is None or walk_seconds < current_fw:
            osm_adj_edges[u_idx][v_idx] = walk_seconds

        current_bw = osm_adj_edges[v_idx].get(u_idx)
        if current_bw is None or walk_seconds < current_bw:
            osm_adj_edges[v_idx][u_idx] = walk_seconds

    total_edges = sum(len(edges) for edges in osm_adj_edges)
    adj_offsets = np.zeros(len(node_ids) + 1, dtype=np.int64)
    adj_neighbors = np.zeros(total_edges, dtype=np.int32)
    adj_weights = np.zeros(total_edges, dtype=np.int64)

    cursor = 0
    for node_idx in range(len(node_ids)):
        adj_offsets[node_idx] = cursor
        for neighbor_idx, weight in osm_adj_edges[node_idx].items():
            adj_neighbors[cursor] = int(neighbor_idx)
            adj_weights[cursor] = int(weight)
            cursor += 1
    adj_offsets[len(node_ids)] = cursor
    return adj_offsets, adj_neighbors, adj_weights


def _single_source_dijkstra_csr(
    adj_offsets: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    source_idx: int,
    cutoff: float,
) -> dict[int, int]:
    cutoff_i = int(cutoff)
    dist: dict[int, int] = {int(source_idx): 0}
    heap: list[tuple[int, int]] = [(0, int(source_idx))]

    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > cutoff_i:
            break
        if current_dist != dist.get(node, None):
            continue

        row_start = int(adj_offsets[node])
        row_end = int(adj_offsets[node + 1])
        for edge_idx in range(row_start, row_end):
            nxt = int(adj_neighbors[edge_idx])
            alt = current_dist + int(adj_weights[edge_idx])
            if alt > cutoff_i:
                continue
            prev = dist.get(nxt)
            if prev is None or alt < prev:
                dist[nxt] = alt
                heapq.heappush(heap, (alt, nxt))
    return dist


def _configure_pandas_compatibility() -> None:
    """Reduce pyrosm incompatibility noise with pandas copy-on-write defaults."""
    try:
        from pandas.errors import ChainedAssignmentError

        # pyrosm currently triggers chained-assignment warnings under modern pandas.
        # We silence this known third-party warning during offline precompute.
        warnings.filterwarnings("ignore", category=ChainedAssignmentError)
    except Exception:
        # Best-effort only; script should continue even if pandas internals differ.
        return


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_earth_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    sin_half_dphi = math.sin(dphi / 2.0)
    sin_half_dlambda = math.sin(dlambda / 2.0)
    a = sin_half_dphi * sin_half_dphi + math.cos(phi1) * math.cos(phi2) * sin_half_dlambda * sin_half_dlambda
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(radius_earth_m * c)


def _station_key(stop_id: str) -> str:
    text = str(stop_id).strip()
    if not text:
        return ""
    separator = text.find(":")
    if separator < 0:
        return text
    return text[:separator]


def _build_bucket_index(lats: np.ndarray, lons: np.ndarray, cell_deg: float) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(lats.shape[0]):
        lat = float(lats[idx])
        lon = float(lons[idx])
        row = int(math.floor(lat / cell_deg))
        col = int(math.floor(lon / cell_deg))
        buckets[(row, col)].append(idx)
    return buckets


def _nearest_node_index(
    lat: float,
    lon: float,
    node_lats: np.ndarray,
    node_lons: np.ndarray,
    buckets: dict[tuple[int, int], list[int]],
    cell_deg: float,
    max_ring: int = 3,
) -> int:
    row = int(math.floor(lat / cell_deg))
    col = int(math.floor(lon / cell_deg))

    best_idx = -1
    best_dist = float("inf")
    for ring in range(max_ring + 1):
        found_any = False
        for d_row in range(-ring, ring + 1):
            for d_col in range(-ring, ring + 1):
                for idx in buckets.get((row + d_row, col + d_col), []):
                    found_any = True
                    dist = _haversine_m(lat, lon, float(node_lats[idx]), float(node_lons[idx]))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
        if found_any and best_idx >= 0:
            return best_idx

    # Global fallback if local buckets are empty.
    for idx in range(node_lats.shape[0]):
        dist = _haversine_m(lat, lon, float(node_lats[idx]), float(node_lons[idx]))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _collect_candidate_targets(
    src_idx: int,
    stop_lats: np.ndarray,
    stop_lons: np.ndarray,
    buckets: dict[tuple[int, int], list[int]],
    bucket_cell_deg: float,
    max_distance_m: float,
    max_neighbors: int,
) -> list[int]:
    src_lat = float(stop_lats[src_idx])
    src_lon = float(stop_lons[src_idx])
    row = int(math.floor(src_lat / bucket_cell_deg))
    col = int(math.floor(src_lon / bucket_cell_deg))

    candidates: list[tuple[float, int]] = []
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            for tgt_idx in buckets.get((row + d_row, col + d_col), []):
                if tgt_idx == src_idx:
                    continue
                tgt_lat = float(stop_lats[tgt_idx])
                tgt_lon = float(stop_lons[tgt_idx])
                distance_m = _haversine_m(src_lat, src_lon, tgt_lat, tgt_lon)
                if distance_m <= 0 or distance_m > max_distance_m:
                    continue
                candidates.append((distance_m, tgt_idx))

    candidates.sort(key=lambda item: item[0])
    if max_neighbors > 0:
        candidates = candidates[:max_neighbors]
    return [idx for _, idx in candidates]


def main() -> None:
    _configure_pandas_compatibility()

    parser = argparse.ArgumentParser(description="Precompute OSM-based walking transfers for transit stops")
    parser.add_argument("--osm-pbf", default="switzerland.osm.pbf", help="Path to OSM .pbf file")
    parser.add_argument("--output", default=".cache/walk_transfers_osm.npz", help="Output .npz transfer cache path")
    parser.add_argument("--max-distance-m", type=float, default=250.0, help="Max stop-to-stop walking distance")
    parser.add_argument("--max-neighbors", type=int, default=4, help="Max nearby stop targets per source stop")
    parser.add_argument("--walk-speed-mps", type=float, default=1.4, help="Walking speed for seconds conversion")
    parser.add_argument("--min-seconds", type=int, default=30, help="Minimum transfer walk seconds")
    parser.add_argument("--sample-rate", type=float, default=0.5, help="Fraction of stops to compute paths for (0.0-1.0)")
    parser.add_argument("--max-sources", type=int, default=0, help="Optional cap on unique OSM source nodes to process (0 = no cap)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint-path", default=".cache/walk_transfers_checkpoint.pkl", help="Checkpoint file path")
    parser.add_argument("--checkpoint-every", type=int, default=250, help="Save checkpoint every N unique source nodes")
    parser.add_argument("--keep-checkpoint", action="store_true", help="Keep checkpoint file after successful completion")
    parser.add_argument("--osm-graph-cache", default=".cache/osm_walking_graph.pkl", help="Path to cached parsed OSM walking graph")
    parser.add_argument("--skip-osm-graph-cache", action="store_true", help="Force re-parse OSM PBF and skip graph cache")
    parser.add_argument("--osm-compact-cache", default=".cache/osm_walking_graph_compact.npz", help="Path to compact OSM CSR cache")
    parser.add_argument("--skip-osm-compact-cache", action="store_true", help="Force rebuild compact OSM CSR cache")
    parser.add_argument("--bbox-from-stops", action="store_true", help="Load only OSM area around stop bbox (much faster first parse)")
    parser.add_argument("--bbox-margin-m", type=float, default=5000.0, help="Margin around stop bbox in meters when --bbox-from-stops is used")
    parser.add_argument(
        "--bbox-trim-quantile",
        type=float,
        default=0.005,
        help="Trim low/high coordinate tails when computing stop bbox (e.g. 0.005 trims 0.5% each side)",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Build/load OSM compact geometry cache only, then exit (no transfer Dijkstra)",
    )
    args = parser.parse_args()

    if sys.version_info >= (3, 13):
        raise SystemExit(
            "pyrosm currently fails to build on Python 3.13 (pyrobuf dependency).\n"
            "Run this precompute script in a Python 3.12 environment, then reuse the generated cache from your main env.\n"
            "Example (Windows):\n"
            "  py -3.12 -m venv .venv-precompute\n"
            "  .\\.venv-precompute\\Scripts\\python.exe -m pip install --upgrade pip\n"
            "  .\\.venv-precompute\\Scripts\\python.exe -m pip install pyrosm networkx\n"
            "  .\\.venv-precompute\\Scripts\\python.exe -m scripts.precompute_walk_transfers --osm-pbf switzerland.osm.pbf --output .cache/walk_transfers_osm.npz"
        )

    try:
        import networkx as nx
        from pyrosm import OSM
    except Exception as exc:
        raise SystemExit(
            "Missing optional dependencies. Install with: "
            "pip install pyrosm networkx\n"
            f"Import error: {exc}"
        )

    osm_path = Path(args.osm_pbf)
    if not osm_path.exists():
        raise SystemExit(f"OSM file not found: {osm_path}")

    network = load_network()
    stop_ids = network.stop_ids
    stop_lats = np.asarray(network.stop_lats, dtype=np.float64)
    stop_lons = np.asarray(network.stop_lons, dtype=np.float64)

    print(f"Loaded transit network stops={len(stop_ids)}")

    graph_cache_path = Path(str(args.osm_graph_cache))
    compact_cache_path = Path(str(args.osm_compact_cache))
    graph = None
    nodes = None
    edges = None
    compact = None
    node_ids = []
    node_lats = np.zeros(0, dtype=np.float64)
    node_lons = np.zeros(0, dtype=np.float64)
    osm_adj_offsets = np.zeros(1, dtype=np.int64)
    osm_adj_neighbors = np.zeros(0, dtype=np.int32)
    osm_adj_weights = np.zeros(0, dtype=np.int64)
    osm_load_started = time.perf_counter()

    if not args.skip_osm_compact_cache:
        compact = _load_compact_osm_cache(compact_cache_path)
        if compact is not None:
            node_ids = list(compact["node_ids"])
            node_lats = compact["node_lats"]
            node_lons = compact["node_lons"]
            osm_adj_offsets = compact["adj_offsets"]
            osm_adj_neighbors = compact["adj_neighbors"]
            osm_adj_weights = compact["adj_weights"]
            print(f"Loaded compact OSM cache from {compact_cache_path}")

    if compact is None and not args.skip_osm_graph_cache:
        cached = _load_osm_graph_cache(graph_cache_path)
        if cached is not None:
            graph = cached.get("graph")
            nodes = cached.get("nodes")
            edges = cached.get("edges")
            print(f"Loaded cached OSM walking graph from {graph_cache_path}")

    if compact is None and graph is None:
        print(f"Loading OSM walking graph from {osm_path} ...")
        osm = None
        if args.bbox_from_stops:
            bbox = _compute_stops_bbox(
                stop_lats,
                stop_lons,
                float(args.bbox_margin_m),
                float(args.bbox_trim_quantile),
            )
            if bbox is not None:
                print(
                    "Using stop bbox for OSM load "
                    f"(min_lon={bbox[0]:.4f}, min_lat={bbox[1]:.4f}, max_lon={bbox[2]:.4f}, max_lat={bbox[3]:.4f})"
                )
                try:
                    osm = OSM(str(osm_path), bounding_box=bbox)
                except Exception as exc:
                    print(f"BBox OSM load unavailable, falling back to full file: {exc}")
                    osm = None
        if osm is None:
            osm = OSM(str(osm_path))
        nodes, edges = osm.get_network(network_type="walking", nodes=True)
        graph = osm.to_graph(nodes, edges, graph_type="networkx")
        if not args.skip_osm_graph_cache:
            try:
                _save_osm_graph_cache(graph_cache_path, graph, nodes, edges)
                print(f"Saved OSM graph cache to {graph_cache_path}")
            except Exception as exc:
                print(f"OSM graph cache save skipped: {exc}")

    osm_load_elapsed = time.perf_counter() - osm_load_started
    print(f"OSM graph load completed in {osm_load_elapsed:.1f}s")

    if compact is None:
        if graph.number_of_nodes() == 0:
            raise SystemExit("OSM walking graph is empty")

        node_extract_started = time.perf_counter()
        node_ids = list(graph.nodes())
        node_lats = np.array([float(graph.nodes[node_id].get("y", np.nan)) for node_id in node_ids], dtype=np.float64)
        node_lons = np.array([float(graph.nodes[node_id].get("x", np.nan)) for node_id in node_ids], dtype=np.float64)
        valid_node_mask = np.isfinite(node_lats) & np.isfinite(node_lons)
        node_ids = [node_ids[idx] for idx in np.where(valid_node_mask)[0].tolist()]
        node_lats = node_lats[valid_node_mask]
        node_lons = node_lons[valid_node_mask]

        if node_lats.size == 0:
            raise SystemExit("OSM graph has no nodes with coordinates")

        # Ensure edge weights are in walking seconds.
        speed = max(0.5, float(args.walk_speed_mps))
        for u, v, data in graph.edges(data=True):
            length_m = float(data.get("length", 0.0))
            walk_seconds = max(1.0, length_m / speed)
            data["walk_seconds"] = walk_seconds

        osm_adj_offsets, osm_adj_neighbors, osm_adj_weights = _build_compact_osm_adjacency(graph, node_ids)
        if not args.skip_osm_compact_cache:
            try:
                _save_compact_osm_cache(
                    compact_cache_path,
                    node_ids,
                    node_lats,
                    node_lons,
                    osm_adj_offsets,
                    osm_adj_neighbors,
                    osm_adj_weights,
                )
                print(f"Saved compact OSM cache to {compact_cache_path}")
            except Exception as exc:
                print(f"Compact OSM cache save skipped: {exc}")

        node_extract_elapsed = time.perf_counter() - node_extract_started
        print(f"Extracted OSM node coordinates and built compact adjacency in {node_extract_elapsed:.1f}s")
    else:
        speed = max(0.5, float(args.walk_speed_mps))
        if node_lats.size == 0:
            raise SystemExit("Compact OSM cache has no nodes")

    if args.geometry_only:
        print(
            "Geometry cache ready: "
            f"nodes={len(node_ids)} edges={int(osm_adj_neighbors.shape[0])} "
            f"path={compact_cache_path}"
        )
        print("Exiting due to --geometry-only (skipped transfer precompute).")
        return

    node_bucket_cell = max(1e-4, float(args.max_distance_m) / 111320.0)
    node_buckets = _build_bucket_index(node_lats, node_lons, node_bucket_cell)

    stop_to_node_idx = np.full(len(stop_ids), -1, dtype=np.int64)
    print("Snapping stops to nearest OSM walking nodes ...")
    snap_started = time.perf_counter()
    for stop_idx in range(len(stop_ids)):
        lat = float(stop_lats[stop_idx])
        lon = float(stop_lons[stop_idx])
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        nearest_idx = _nearest_node_index(
            lat,
            lon,
            node_lats,
            node_lons,
            node_buckets,
            node_bucket_cell,
        )
        stop_to_node_idx[stop_idx] = int(nearest_idx)
    snap_elapsed = time.perf_counter() - snap_started
    print(f"Snapped stops to OSM nodes in {snap_elapsed:.1f}s")

    stop_bucket_cell = max(1e-4, float(args.max_distance_m) / 111320.0)
    stop_buckets = _build_bucket_index(stop_lats, stop_lons, stop_bucket_cell)

    station_groups: dict[str, list[int]] = defaultdict(list)
    for idx, stop_id in enumerate(stop_ids):
        station_groups[_station_key(stop_id)].append(idx)

    edges_by_stop: list[dict[int, int]] = [dict() for _ in range(len(stop_ids))]

    print("Computing OSM shortest-path walking transfers ...")
    max_walk_seconds = max(float(args.min_seconds), float(args.max_distance_m) / speed + 60.0)
    
    # Sample stops to speed up precomputation (default 50%)
    sample_rate = max(0.01, min(1.0, float(args.sample_rate)))
    import random
    all_src_indices = list(range(len(stop_ids)))
    if sample_rate < 1.0:
        num_samples = max(1, int(len(stop_ids) * sample_rate))
        src_indices_to_process = sorted(random.sample(all_src_indices, num_samples))
        print(f"Computing paths for sampled {len(src_indices_to_process)}/{len(stop_ids)} stops ({sample_rate*100:.0f}%) ...")
    else:
        src_indices_to_process = all_src_indices
        print(f"Computing paths for all {len(stop_ids)} stops ...")
    
    # Prepare all source->targets first, then reuse one Dijkstra per unique OSM source node.
    prepared_sources: dict[int, list[tuple[int, int]]] = {}
    source_groups: dict[int, list[int]] = defaultdict(list)
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    prep_started = time.perf_counter()
    for src_idx in src_indices_to_process:
        src_node_pos = int(stop_to_node_idx[src_idx])
        if src_node_pos < 0:
            continue
        src_node_id = node_ids[src_node_pos]
        src_node_idx = int(node_id_to_idx[src_node_id])

        candidate_targets = _collect_candidate_targets(
            src_idx,
            stop_lats,
            stop_lons,
            stop_buckets,
            stop_bucket_cell,
            float(args.max_distance_m),
            int(args.max_neighbors),
        )

        # Always keep same-station transfers to preserve platform changes.
        src_station_key = _station_key(stop_ids[src_idx])
        for tgt_idx in station_groups.get(src_station_key, []):
            if tgt_idx != src_idx:
                candidate_targets.append(int(tgt_idx))

        resolved_targets: list[tuple[int, int]] = []
        for tgt_idx in sorted(set(candidate_targets)):
            tgt_node_pos = int(stop_to_node_idx[tgt_idx])
            if tgt_node_pos < 0:
                continue
            tgt_node_id = node_ids[tgt_node_pos]
            tgt_node_idx = int(node_id_to_idx[tgt_node_id])
            resolved_targets.append((int(tgt_idx), tgt_node_idx))

        if not resolved_targets:
            continue

        prepared_sources[int(src_idx)] = resolved_targets
        source_groups[src_node_idx].append(int(src_idx))

    unique_sources = list(source_groups.items())
    if int(args.max_sources) > 0 and len(unique_sources) > int(args.max_sources):
        unique_sources = unique_sources[: int(args.max_sources)]
        print(f"Capping to {len(unique_sources)} unique source OSM nodes due to --max-sources")

    source_stop_counts: dict[int, int] = {int(src_node_id): len(src_group) for src_node_id, src_group in unique_sources}

    prep_elapsed = time.perf_counter() - prep_started
    print(
        f"Prepared {len(prepared_sources)} source stops grouped into {len(unique_sources)} unique OSM source nodes "
        f"in {prep_elapsed:.1f}s"
    )

    checkpoint_path = Path(str(args.checkpoint_path))
    checkpoint_every = max(1, int(args.checkpoint_every))

    run_metadata = {
        "stop_count": int(len(stop_ids)),
        "sample_rate": float(sample_rate),
        "max_distance_m": float(args.max_distance_m),
        "max_neighbors": int(args.max_neighbors),
        "min_seconds": int(args.min_seconds),
        "max_walk_seconds": float(max_walk_seconds),
        "unique_source_count": int(len(unique_sources)),
    }

    done_source_node_ids: set[int] = set()
    if args.resume:
        checkpoint_payload = _load_checkpoint(checkpoint_path)
        if checkpoint_payload is not None:
            checkpoint_meta = checkpoint_payload.get("metadata", {})
            meta_matches = (
                isinstance(checkpoint_meta, dict)
                and int(checkpoint_meta.get("stop_count", -1)) == run_metadata["stop_count"]
                and float(checkpoint_meta.get("sample_rate", -1.0)) == run_metadata["sample_rate"]
                and float(checkpoint_meta.get("max_distance_m", -1.0)) == run_metadata["max_distance_m"]
                and int(checkpoint_meta.get("max_neighbors", -1)) == run_metadata["max_neighbors"]
                and int(checkpoint_meta.get("min_seconds", -1)) == run_metadata["min_seconds"]
            )
            if meta_matches:
                loaded_edges = checkpoint_payload.get("edges_by_stop")
                loaded_done = checkpoint_payload.get("done_source_node_ids", [])
                if isinstance(loaded_edges, list) and len(loaded_edges) == len(stop_ids):
                    edges_by_stop = loaded_edges
                    done_source_node_ids = set(int(item) for item in loaded_done)
                    print(
                        f"Resumed checkpoint from {checkpoint_path} "
                        f"(done_unique_sources={len(done_source_node_ids)})"
                    )
                else:
                    print("Checkpoint ignored: invalid edges payload")
            else:
                print("Checkpoint ignored: run parameters differ")

    planned_source_stops = int(sum(source_stop_counts.values()))
    resumed_source_stops = int(sum(source_stop_counts.get(src_node_id, 0) for src_node_id in done_source_node_ids))
    print(
        f"Progress start: processed_source_stops={resumed_source_stops} "
        f"remaining_source_stops={max(0, planned_source_stops - resumed_source_stops)} "
        f"total_source_stops={planned_source_stops}"
    )

    search_started = time.perf_counter()
    processed_since_checkpoint = 0
    processed_unique_sources = 0
    processed_source_stops = resumed_source_stops
    for group_idx, (src_node_id, src_group) in enumerate(unique_sources, start=1):
        src_node_id = int(src_node_id)
        if src_node_id in done_source_node_ids:
            continue

        if group_idx % max(1, len(unique_sources) // 20) == 0:
            elapsed = time.perf_counter() - search_started
            remaining_source_stops = max(0, planned_source_stops - processed_source_stops)
            print(
                f"  Dijkstra progress: {group_idx}/{len(unique_sources)} unique sources ({elapsed:.1f}s) | "
                f"source_stops_done={processed_source_stops} source_stops_remaining={remaining_source_stops}"
            )

        # One bounded shortest-path search for all stops snapping to the same OSM node.
        path_lengths = _single_source_dijkstra_csr(
            osm_adj_offsets,
            osm_adj_neighbors,
            osm_adj_weights,
            src_node_id,
            max_walk_seconds,
        )

        for src_idx in src_group:
            for tgt_idx, tgt_node_idx in prepared_sources[src_idx]:
                seconds = path_lengths.get(int(tgt_node_idx))
                if seconds is None:
                    continue
                walk_seconds = max(int(args.min_seconds), int(seconds))
                if walk_seconds > max_walk_seconds:
                    continue
                current = edges_by_stop[src_idx].get(tgt_idx)
                if current is None or walk_seconds < current:
                    edges_by_stop[src_idx][tgt_idx] = int(walk_seconds)

        done_source_node_ids.add(src_node_id)
        processed_unique_sources += 1
        processed_since_checkpoint += 1
        processed_source_stops += int(len(src_group))
        if processed_since_checkpoint >= checkpoint_every:
            _save_checkpoint(
                checkpoint_path,
                run_metadata,
                edges_by_stop,
                done_source_node_ids,
            )
            processed_since_checkpoint = 0
            elapsed = time.perf_counter() - search_started
            remaining_source_stops = max(0, planned_source_stops - processed_source_stops)
            print(
                f"  Checkpoint saved ({len(done_source_node_ids)}/{len(unique_sources)} unique sources, {elapsed:.1f}s) | "
                f"source_stops_done={processed_source_stops} source_stops_remaining={remaining_source_stops}"
            )

    if processed_unique_sources > 0:
        _save_checkpoint(
            checkpoint_path,
            run_metadata,
            edges_by_stop,
            done_source_node_ids,
        )

    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink(missing_ok=True)

    print(
        f"Progress end: processed_source_stops={processed_source_stops} "
        f"remaining_source_stops={max(0, planned_source_stops - processed_source_stops)} "
        f"total_source_stops={planned_source_stops}"
    )

    total_edges = sum(len(item) for item in edges_by_stop)
    transfer_offsets = np.zeros(len(stop_ids) + 1, dtype=np.int64)
    transfer_neighbors = np.zeros(total_edges, dtype=np.int32)
    transfer_weights = np.zeros(total_edges, dtype=np.int64)

    cursor = 0
    for stop_idx in range(len(stop_ids)):
        transfer_offsets[stop_idx] = cursor
        for neighbor_idx, weight in edges_by_stop[stop_idx].items():
            transfer_neighbors[cursor] = int(neighbor_idx)
            transfer_weights[cursor] = int(weight)
            cursor += 1
    transfer_offsets[len(stop_ids)] = cursor

    # OSM adjacency already available from compact cache or built once from graph.
    print("Preparing OSM walking graph adjacency payload ...")
    osm_node_ids = np.array(node_ids, dtype=object)
    osm_node_lats = np.asarray(node_lats, dtype=np.float64)
    osm_node_lons = np.asarray(node_lons, dtype=np.float64)
    
    # Ensure stop_to_node mappings have proper type
    stop_to_node_idx_int = np.asarray(stop_to_node_idx, dtype=np.int32)

    save_transfer_graph_to_cache(
        args.output,
        stop_ids,
        transfer_offsets,
        transfer_neighbors,
        transfer_weights,
        osm_node_ids,
        osm_node_lats,
        osm_node_lons,
        osm_adj_offsets,
        osm_adj_neighbors,
        osm_adj_weights,
        stop_to_node_idx_int,
    )

    print(
        f"Saved OSM walk transfer cache to {args.output} "
        f"(stops={len(stop_ids)} edges={transfer_neighbors.shape[0]} "
        f"osm_nodes={len(node_ids)} osm_edges={osm_adj_neighbors.shape[0]})"
    )

    if args.keep_checkpoint:
        print(f"Checkpoint kept at {checkpoint_path}")


if __name__ == "__main__":
    main()
