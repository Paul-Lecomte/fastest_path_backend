"""Query-time walking transfer resolution using OSM graph with result caching."""
# This module provides the OSMWalkingTransfers class which loads a preprocessed OSM walking graph and computes walking transfer times on-demand. It uses a compact CSR adjacency format for efficient shortest path computation and caches results to speed up repeated queries. The class supports finding the nearest OSM node to given coordinates and computing walking times to multiple target locations within a specified maximum walking time.
from __future__ import annotations

import heapq
import math
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class OSMWalkingTransfers:
    """
    Loads OSM walking graph once and computes transfers on-demand.
    Results are cached to avoid recomputation for repeated queries.
    """

    def __init__(
        self,
        osm_graph_cache_path: Path | str = ".cache/osm_walking_graph.pkl",
        result_cache_size: int = 10000,
        walk_speed_mps: float = 1.4,
    ):
        self.osm_graph_cache_path = Path(osm_graph_cache_path)
        self.result_cache: Dict[Tuple[int, float], Dict[int, int]] = {}
        self.result_cache_size = result_cache_size
        self.walk_speed_mps = max(0.5, float(walk_speed_mps))

        # Graph data loaded on-demand
        self._graph = None
        self._node_ids = None
        self._node_lats = None
        self._node_lons = None
        self._adj_offsets = None
        self._adj_neighbors = None
        self._adj_weights = None
        self._node_id_to_idx = None
        self._node_buckets = None  # For fast spatial lookup

        self.load_time = 0.0
        self.is_loaded = False

    def _load_graph(self) -> bool:
        """Load OSM graph from cache if available. Returns True if loaded."""
        if self.is_loaded:
            return True

        if not self.osm_graph_cache_path.exists():
            return False

        try:
            start = time.perf_counter()
            with open(self.osm_graph_cache_path, "rb") as fh:
                payload = pickle.load(fh)

            if not isinstance(payload, dict):
                return False

            # Expect either full graph or compact CSR format
            if "graph" in payload:
                # NetworkX graph format (slower but works)
                graph = payload.get("graph")
                if graph is None or graph.number_of_nodes() == 0:
                    return False

                self._graph = graph
                self._node_ids = list(graph.nodes())
                self._node_lats = np.array(
                    [float(graph.nodes[nid].get("y", np.nan)) for nid in self._node_ids],
                    dtype=np.float64,
                )
                self._node_lons = np.array(
                    [float(graph.nodes[nid].get("x", np.nan)) for nid in self._node_ids],
                    dtype=np.float64,
                )

                # Build compact CSR adjacency
                self._build_csr_from_graph(graph)
            elif "adj_offsets" in payload:
                # Already compact CSR format
                self._node_ids = list(payload.get("node_ids", []))
                self._node_lats = np.asarray(payload.get("node_lats"), dtype=np.float64)
                self._node_lons = np.asarray(payload.get("node_lons"), dtype=np.float64)
                self._adj_offsets = np.asarray(payload.get("adj_offsets"), dtype=np.int64)
                self._adj_neighbors = np.asarray(payload.get("adj_neighbors"), dtype=np.int32)
                self._adj_weights = np.asarray(payload.get("adj_weights"), dtype=np.int64)
            else:
                return False

            if not self._node_ids or self._node_lats.size == 0:
                return False

            # Build lookup maps
            self._node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self._node_ids)}
            self._build_node_buckets()

            self.load_time = time.perf_counter() - start
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Failed to load OSM graph: {e}")
            return False

    def _build_csr_from_graph(self, graph) -> None:
        """Convert NetworkX graph to CSR adjacency format."""
        n = len(self._node_ids)
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self._node_ids)}

        # Build edge lists
        osm_adj_edges = [dict() for _ in range(n)]
        for u, v, data in graph.edges(data=True):
            if u not in node_id_to_idx or v not in node_id_to_idx:
                continue
            u_idx = node_id_to_idx[u]
            v_idx = node_id_to_idx[v]
            length_m = float(data.get("length", 1.0))
            walk_seconds = max(1, int(length_m / self.walk_speed_mps))

            # Bidirectional
            for src_idx, tgt_idx in [(u_idx, v_idx), (v_idx, u_idx)]:
                current = osm_adj_edges[src_idx].get(tgt_idx)
                if current is None or walk_seconds < current:
                    osm_adj_edges[src_idx][tgt_idx] = walk_seconds

        # Convert to CSR
        total_edges = sum(len(edges) for edges in osm_adj_edges)
        self._adj_offsets = np.zeros(n + 1, dtype=np.int64)
        self._adj_neighbors = np.zeros(total_edges, dtype=np.int32)
        self._adj_weights = np.zeros(total_edges, dtype=np.int64)

        cursor = 0
        for node_idx in range(n):
            self._adj_offsets[node_idx] = cursor
            for neighbor_idx, weight in osm_adj_edges[node_idx].items():
                self._adj_neighbors[cursor] = int(neighbor_idx)
                self._adj_weights[cursor] = int(weight)
                cursor += 1
        self._adj_offsets[n] = cursor

    def _build_node_buckets(self, cell_deg: float = 0.01) -> None:
        """Build spatial bucket index for fast node lookup."""
        from collections import defaultdict

        self._node_buckets = defaultdict(list)
        for idx in range(len(self._node_ids)):
            lat = float(self._node_lats[idx])
            lon = float(self._node_lons[idx])
            row = int(math.floor(lat / cell_deg))
            col = int(math.floor(lon / cell_deg))
            self._node_buckets[(row, col)].append(idx)

    def nearest_node_idx(
        self,
        lat: float,
        lon: float,
        max_ring: int = 3,
    ) -> Optional[int]:
        """Find nearest OSM walking node to given coordinate."""
        if not self.is_loaded:
            return None

        cell_deg = 0.01
        row = int(math.floor(lat / cell_deg))
        col = int(math.floor(lon / cell_deg))

        best_idx = -1
        best_dist = float("inf")

        for ring in range(max_ring + 1):
            found_any = False
            for d_row in range(-ring, ring + 1):
                for d_col in range(-ring, ring + 1):
                    for idx in self._node_buckets.get((row + d_row, col + d_col), []):
                        found_any = True
                        dist = self._haversine_m(lat, lon, float(self._node_lats[idx]), float(self._node_lons[idx]))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = idx

            if found_any and best_idx >= 0:
                return best_idx

        # Fallback: search all
        for idx in range(len(self._node_ids)):
            dist = self._haversine_m(lat, lon, float(self._node_lats[idx]), float(self._node_lons[idx]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        return best_idx if best_idx >= 0 else None

    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute distance in meters."""
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _dijkstra_csr(
        self,
        source_idx: int,
        cutoff: float,
    ) -> Dict[int, int]:
        """Single-source Dijkstra on CSR adjacency."""
        cutoff_i = int(cutoff)
        dist: Dict[int, int] = {source_idx: 0}
        heap: list[tuple[int, int]] = [(0, source_idx)]

        while heap:
            current_dist, node = heapq.heappop(heap)
            if current_dist > cutoff_i:
                break
            if current_dist != dist.get(node):
                continue

            row_start = int(self._adj_offsets[node])
            row_end = int(self._adj_offsets[node + 1])
            for edge_idx in range(row_start, row_end):
                nxt = int(self._adj_neighbors[edge_idx])
                alt = current_dist + int(self._adj_weights[edge_idx])
                if alt > cutoff_i:
                    continue
                if alt < dist.get(nxt, float("inf")):
                    dist[nxt] = alt
                    heapq.heappush(heap, (alt, nxt))

        return dist

    def get_transfers(
        self,
        src_lat: float,
        src_lon: float,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
        max_walk_seconds: float = 600,
        min_walk_seconds: int = 30,
    ) -> Dict[int, int]:
        """
        Compute walking times from (src_lat, src_lon) to targets.
        Returns dict[target_index] -> walk_seconds or empty dict if transfer impossible.
        """
        if not self.is_loaded:
            if not self._load_graph():
                return {}

        # Snap source to nearest OSM node
        src_node_idx = self.nearest_node_idx(src_lat, src_lon)
        if src_node_idx is None or src_node_idx < 0:
            return {}

        # Check result cache
        cache_key = (src_node_idx, max_walk_seconds)
        if cache_key in self.result_cache:
            cached_paths = self.result_cache[cache_key]
        else:
            # Compute shortest paths from source
            cached_paths = self._dijkstra_csr(src_node_idx, max_walk_seconds)

            # Cache result (with size limit)
            if len(self.result_cache) >= self.result_cache_size:
                # Simple eviction: remove first item
                first_key = next(iter(self.result_cache))
                del self.result_cache[first_key]
            self.result_cache[cache_key] = cached_paths

        # Map targets to nearest OSM nodes and look up distances
        result: Dict[int, int] = {}
        for target_idx in range(len(target_lats)):
            tgt_lat = float(target_lats[target_idx])
            tgt_lon = float(target_lons[target_idx])

            if not (np.isfinite(tgt_lat) and np.isfinite(tgt_lon)):
                continue

            tgt_node_idx = self.nearest_node_idx(tgt_lat, tgt_lon)
            if tgt_node_idx is None or tgt_node_idx < 0:
                continue

            walk_seconds = cached_paths.get(tgt_node_idx)
            if walk_seconds is None:
                continue

            walk_seconds = max(int(min_walk_seconds), int(walk_seconds))
            if walk_seconds <= max_walk_seconds:
                result[target_idx] = walk_seconds

        return result

    def clear_cache(self) -> None:
        """Clear result cache."""
        self.result_cache.clear()

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cached_results": len(self.result_cache),
            "cache_size_items": self.result_cache_size,
            "graph_loaded": self.is_loaded,
            "graph_load_time_s": self.load_time,
            "osm_nodes": len(self._node_ids) if self._node_ids else 0,
        }
