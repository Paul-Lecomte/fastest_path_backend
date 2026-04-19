"""Fast transfer computation using only Haversine distances (no OSM routing needed). command python.exe -m scripts.precompute_transfer_distances --output .cache/walk_transfers_osm.npz"""
import argparse
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

from src.http_server import load_network
from src.loader import save_transfer_graph_to_cache


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance in meters."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _station_key(stop_id: str) -> str:
    """Extract station key from stop ID."""
    text = str(stop_id).strip()
    if not text:
        return ""
    separator = text.find(":")
    if separator < 0:
        return text
    return text[:separator]


def _build_grid_index(stop_lats, stop_lons, cell_deg=0.02):
    """Build spatial grid index for fast neighbor lookup."""
    grid = defaultdict(list)
    for idx in range(len(stop_lats)):
        lat = float(stop_lats[idx])
        lon = float(stop_lons[idx])
        if np.isfinite(lat) and np.isfinite(lon):
            row = int(math.floor(lat / cell_deg))
            col = int(math.floor(lon / cell_deg))
            grid[(row, col)].append(idx)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast transfer computation using Haversine distances")
    parser.add_argument("--output", default=".cache/walk_transfers_osm.npz", help="Output .npz transfer cache path")
    parser.add_argument("--max-distance-m", type=float, default=250.0, help="Max transfer distance in meters")
    parser.add_argument("--max-neighbors", type=int, default=4, help="Max nearby stops per source")
    parser.add_argument("--walk-speed-mps", type=float, default=1.4, help="Walking speed (m/s)")
    parser.add_argument("--min-seconds", type=int, default=30, help="Minimum transfer time (seconds)")
    parser.add_argument("--walk-time-multiplier", type=float, default=1.0, help="Multiplier applied to geometric walk-time estimate")
    args = parser.parse_args()

    print("Loading transit network...")
    network = load_network()
    stop_ids = network.stop_ids
    stop_lats = np.asarray(network.stop_lats, dtype=np.float64)
    stop_lons = np.asarray(network.stop_lons, dtype=np.float64)
    
    print(f"Loaded {len(stop_ids)} transit stops")

    # Build transfer graph using Haversine distances with spatial indexing
    print("Building spatial grid index...")
    grid = _build_grid_index(stop_lats, stop_lons, cell_deg=0.02)
    
    print("Computing walking transfers using Haversine distances...")
    start = time.perf_counter()

    edges_by_stop = [dict() for _ in range(len(stop_ids))]
    station_groups = defaultdict(list)
    
    # Group stops by station
    for idx, stop_id in enumerate(stop_ids):
        station_groups[_station_key(stop_id)].append(idx)

    max_walk_seconds = float(args.max_distance_m) / max(0.5, float(args.walk_speed_mps)) * float(args.walk_time_multiplier)
    max_neighbors = max(0, int(args.max_neighbors))
    cell_deg = 0.02
    
    # Compute transfers for each source stop
    for src_idx in range(len(stop_ids)):
        src_lat = float(stop_lats[src_idx])
        src_lon = float(stop_lons[src_idx])
        
        if not (np.isfinite(src_lat) and np.isfinite(src_lon)):
            continue

        # Find nearby stops in adjacent grid cells only
        src_row = int(math.floor(src_lat / cell_deg))
        src_col = int(math.floor(src_lon / cell_deg))
        
        candidates = []
        # Only check stops in nearby grid cells (3x3)
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                for tgt_idx in grid.get((src_row + d_row, src_col + d_col), []):
                    if src_idx == tgt_idx:
                        continue
                    
                    tgt_lat = float(stop_lats[tgt_idx])
                    tgt_lon = float(stop_lons[tgt_idx])
                    
                    if not (np.isfinite(tgt_lat) and np.isfinite(tgt_lon)):
                        continue

                    distance_m = _haversine_m(src_lat, src_lon, tgt_lat, tgt_lon)
                    if distance_m > float(args.max_distance_m):
                        continue
                    
                    # Convert to walk seconds with multiplier
                    walk_seconds = int(distance_m / max(0.5, float(args.walk_speed_mps)) * float(args.walk_time_multiplier))
                    walk_seconds = max(int(args.min_seconds), walk_seconds)
                    
                    if walk_seconds <= max_walk_seconds:
                        candidates.append((distance_m, tgt_idx, walk_seconds))

        # Sort by distance and keep top N neighbors
        candidates.sort(key=lambda x: x[0])
        if max_neighbors > 0:
            candidates = candidates[:max_neighbors]
        
        # Add candidates to graph
        for _, tgt_idx, walk_seconds in candidates:
            edges_by_stop[src_idx][tgt_idx] = walk_seconds

        # Always add same-station transfers (for platform changes)
        src_station_key = _station_key(stop_ids[src_idx])
        for tgt_idx in station_groups.get(src_station_key, []):
            if tgt_idx != src_idx:
                # Same station = minimum transfer time
                edges_by_stop[src_idx][tgt_idx] = int(args.min_seconds)

        if (src_idx + 1) % max(1, len(stop_ids) // 10) == 0:
            elapsed = time.perf_counter() - start
            progress = 100 * (src_idx + 1) / len(stop_ids)
            print(f"  {progress:.0f}% ({src_idx + 1}/{len(stop_ids)}) - {elapsed:.1f}s")

    # Convert to CSR format for storage
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

    elapsed = time.perf_counter() - start
    print(f"Computed {total_edges} transfers in {elapsed:.1f}s")

    # Save cache
    output_path = Path(str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}...")
    
    # Save via loader function for compatibility
    save_transfer_graph_to_cache(
        str(output_path),
        stop_ids,
        transfer_offsets,
        transfer_neighbors,
        transfer_weights,
    )
    
    print(f"✓ Precompute complete in {elapsed:.1f}s")
    print(f"  Total transfers: {total_edges}")
    print(f"  Avg transfers per stop: {total_edges / len(stop_ids):.1f}")


if __name__ == "__main__":
    main()
