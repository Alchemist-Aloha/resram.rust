import resram_core
from resram_core import load_input, cross_sections
import time
import numpy as np

def compare():
    print("Initializing cross_sections comparison...")
    
    # Initialize two identical objects
    obj_py = load_input()
    obj_rust = load_input()
    
    print(f"Modes: {len(obj_py.wg)}")
    print(f"Energy grid (EL): {len(obj_py.EL)} points")
    print(f"Convolved grid (convEL): {len(obj_py.convEL)} points")

    # 1. Execute Python Implementation
    resram_core.HAS_RUST = False
    print("\n[Running Python Implementation...]")
    start_py = time.perf_counter()
    abs_py, fl_py, ram_py, _, _ = cross_sections(obj_py)
    end_py = time.perf_counter()
    time_py = end_py - start_py
    print(f"Python Time: {time_py:.4f} seconds")

    # 2. Execute Rust Implementation
    resram_core.HAS_RUST = True
    print("\n[Running Rust Implementation...]")
    start_rust = time.perf_counter()
    abs_rust, fl_rust, ram_rust, _, _ = cross_sections(obj_rust)
    end_rust = time.perf_counter()
    time_rust = end_rust - start_rust
    print(f"Rust Time:   {time_rust:.4f} seconds")

    # 3. Numerical Comparison
    print("\n--- Numerical Consistency Analysis ---")
    
    def check_diff(name, py_arr, rust_arr):
        # Convert to real if necessary (spectra are real intensities)
        py_arr = np.real(py_arr)
        rust_arr = np.real(rust_arr)
        
        max_diff = np.max(np.abs(py_arr - rust_arr))
        mean_val = np.mean(np.abs(py_arr))
        rel_diff = max_diff / mean_val if mean_val > 0 else 0
        
        print(f"{name:12} | Max Abs Diff: {max_diff:.2e} | Rel Diff: {rel_diff:.2e}")
        return max_diff

    diff_abs = check_diff("Absorption", abs_py, abs_rust)
    diff_fl  = check_diff("Fluorescence", fl_py, fl_rust)
    diff_ram = check_diff("Raman REPs", ram_py, ram_rust)

    print("\n--- Summary ---")
    print(f"Speedup: {time_py / time_rust:.2f}x faster with Rust")
    
    # Spectroscopy calculations usually allow for minor float discrepancies
    threshold = 1e-6
    if diff_abs < threshold and diff_ram < threshold:
        print("\nSUCCESS: Rust and Python results match perfectly.")
    else:
        print(f"\nWARNING: Results differ by more than {threshold:.0e}!")
        if diff_ram > threshold:
            print("Note: Check Raman absolute square and convolution alignment.")

if __name__ == "__main__":
    compare()
