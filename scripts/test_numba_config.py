import numba
import os
import traceback

print(f'Numba version: {numba.__version__}')

# Check environment
print(f'NUMBA_DISABLE_JIT: {os.environ.get("NUMBA_DISABLE_JIT", "not set")}')
print(f'NUMBA_CACHE_DIR: {os.environ.get("NUMBA_CACHE_DIR", "not set")}')

# Try simple JIT with cache disabled
from numba import njit

@njit(cache=False)
def test_func(x):
    return x + 1

try:
    result = test_func(5)
    print(f'✓ Simple JIT test passed: {result}')
except Exception as e:
    print(f'✗ Simple JIT test failed: {e}')
    traceback.print_exc()

# Now try the RAPTOR import
print("\n=== RAPTOR IMPORT TEST ===")
try:
    from src import solver
    # Check if the functions are actually compiled
    import inspect
    print(f'run_raptor_with_stats type: {type(solver.run_raptor_with_stats)}')
    print(f'run_raptor_with_stats module: {solver.run_raptor_with_stats.__module__}')
    
    # Try to get the compilation state
    if hasattr(solver.run_raptor_with_stats, 'py_func'):
        print('run_raptor_with_stats is a compiled Numba function')
    else:
        print('run_raptor_with_stats might not be compiled')
        
except Exception as e:
    print(f'Error: {e}')
    traceback.print_exc()
