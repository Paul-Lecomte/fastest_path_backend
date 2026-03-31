#!/usr/bin/env python3
"""Diagnostic script to identify numba JIT compilation issues."""

import sys
import os

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Platform: {sys.platform}")

# Test numba import
print("\n=== NUMBA IMPORT ===")
try:
    import numba
    print(f"✓ numba {numba.__version__} imported successfully")
except Exception as e:
    print(f"✗ numba import failed: {e}")
    sys.exit(1)

# Test JIT decorator
print("\n=== NUMBA JIT DECORATOR ===")
try:
    from numba import njit
    
    @njit
    def test_jit():
        return sum(range(1000))
    
    result = test_jit()
    print(f"✓ Simple JIT function compiled and executed: {result}")
except Exception as e:
    print(f"✗ JIT compilation failed: {e}")
    import traceback
    traceback.print_exc()

# Test solver import
print("\n=== SOLVER IMPORT ===")
sys.path.insert(0, '/workspace' if os.path.exists('/workspace') else '.')
try:
    from src.solver import numba_enabled, run_raptor_with_stats
    enabled = numba_enabled()
    print(f"✓ Solver imported, numba_enabled()={enabled}")
    if not enabled:
        print("  WARNING: RAPTOR will run in pure Python mode (very slow)")
except Exception as e:
    print(f"✗ Solver import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test network load
print("\n=== NETWORK LOAD ===")
try:
    from src.http_server import load_network
    network = load_network()
    print(f"✓ Network loaded: {network.stop_times.shape[0]} stop_times")
    
    # Try a single RAPTOR run
    print("\n=== SINGLE RAPTOR RUN ===")
    import time
    started = time.perf_counter()
    
    earliest, _, _, _, _, _, _ = run_raptor_with_stats(
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
        network.trip_cost_factors,
        60,  # transfer_penalty_seconds
        network.transfer_offsets,
        network.transfer_neighbors,
        network.transfer_weights,
        500,  # start_stop_id
        -1,   # end_stop_id (search all)
        0,    # departure_time
        max_rounds=8,
    )
    elapsed = time.perf_counter() - started
    print(f"✓ RAPTOR executed in {elapsed*1000:.1f}ms")
    if elapsed > 1.0:
        print(f"  WARNING: This is very slow! Numba may not be working.")
        print(f"  Expected with JIT: ~50-200ms, without JIT: ~10-30s")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== SUMMARY ===")
if numba_enabled():
    print("✓ Numba JIT is ENABLED - server should perform well")
else:
    print("✗ Numba JIT is DISABLED - server will be very slow")
