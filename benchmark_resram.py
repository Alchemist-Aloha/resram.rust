import resram_core
from resram_core import load_input, param_init, raman_residual
import time
import numpy as np

def benchmark():
    print("Initializing ResRam benchmark...")
    
    # 1. Setup Data
    # Initialize the object and parameters
    fit_obj = load_input() 
    fit_switch = np.ones(len(fit_obj.delta)+7)
    params = param_init(fit_switch, fit_obj)
    
    # Iterations for averaging
    n_iters = 100
    
    print(f"\n--- Running Benchmark ({n_iters} iterations) ---")

    # 2. Benchmark Python Implementation
    resram_core.HAS_RUST = False
    print("\n[Python Path]")
    
    # Warm-up call
    raman_residual(params, fit_obj)
    
    start_py = time.perf_counter()
    for _ in range(n_iters):
        py_loss, py_sigma, py_mismatch = raman_residual(params, fit_obj)
    end_py = time.perf_counter()
    
    avg_py = (end_py - start_py) / n_iters
    print(f"Python Average Time: {avg_py:.6f} seconds")

    # 3. Benchmark Rust Implementation
    resram_core.HAS_RUST = True
    print("\n[Rust Path]")
    
    # Warm-up call
    raman_residual(params, fit_obj)
    
    start_rust = time.perf_counter()
    for _ in range(n_iters):
        rust_loss, rust_sigma, rust_mismatch = raman_residual(params, fit_obj)
    end_rust = time.perf_counter()
    
    avg_rust = (end_rust - start_rust) / n_iters
    print(f"Rust Average Time:   {avg_rust:.6f} seconds")

    # 4. Accuracy & Speed Comparison
    print("\n--- Final Comparison ---")
    print(f"Speedup: {avg_py / avg_rust:.2f}x faster with Rust")
    
    # Check numerical differences
    loss_diff = abs(py_loss - rust_loss)
    sigma_diff = abs(py_sigma - rust_sigma)
    mismatch_diff = abs(py_mismatch - rust_mismatch)
    
    print(f"\nNumerical Consistency Check:")
    print(f"Loss Difference:     {loss_diff:.2e}")
    print(f"Sigma Difference:    {sigma_diff:.2e}")
    print(f"Mismatch Difference: {mismatch_diff:.2e}%")
    
    if loss_diff < 1e-5:
        print("\nSUCCESS: Results are numerically consistent.")
    else:
        print("\nWARNING: Significant difference detected in results!")

if __name__ == "__main__":
    benchmark()
