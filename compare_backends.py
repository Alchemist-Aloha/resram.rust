import os
import numpy as np
import subprocess
import shutil
import json
from resram_core import load_input, raman_residual, cross_sections

def create_dataset(base_dir, target_dir, params):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Copy essential data files
    files_to_copy = ["abs_exp.dat", "profs_exp.dat", "rpumps.dat", "rshift_exp.dat"]
    for f in files_to_copy:
        if os.path.exists(os.path.join(base_dir, f)):
            shutil.copy(os.path.join(base_dir, f), os.path.join(target_dir, f))
    
    # Handle freqs and deltas
    freqs = np.loadtxt(os.path.join(base_dir, "freqs.dat"))
    deltas = np.loadtxt(os.path.join(base_dir, "deltas.dat"))
    
    if "freq_scale" in params:
        freqs *= params["freq_scale"]
    if "delta_scale" in params:
        deltas *= params["delta_scale"]
    if "random_deltas" in params:
        np.random.seed(42)
        deltas = np.random.rand(len(freqs)) * params["random_deltas"]
    if "fewer_modes" in params:
        freqs = freqs[:params["fewer_modes"]]
        deltas = deltas[:params["fewer_modes"]]
        # If we reduce modes, we must also reduce profs_exp rows to match
        profs_exp = np.loadtxt(os.path.join(base_dir, "profs_exp.dat"))
        if profs_exp.ndim == 1: profs_exp = profs_exp.reshape(-1, 1)
        np.savetxt(os.path.join(target_dir, "profs_exp.dat"), profs_exp[:params["fewer_modes"], :], delimiter="\t")

    np.savetxt(os.path.join(target_dir, "freqs.dat"), freqs)
    np.savetxt(os.path.join(target_dir, "deltas.dat"), deltas)

    # Read base inp.txt
    with open(os.path.join(base_dir, "inp.txt"), "r") as f:
        lines = f.readlines()
    
    # Update parameters
    # 0: gamma, 1: theta, 2: E0, 3: kappa, 7: M, 13: Temp
    new_lines = []
    for i, line in enumerate(lines):
        comment = line.partition("#")[1] + line.partition("#")[2]
        val = line.partition("#")[0].strip()
        
        if i == 0 and "gamma" in params: val = str(params["gamma"])
        elif i == 1 and "theta" in params: val = str(params["theta"])
        elif i == 2 and "E0" in params: val = str(params["E0"])
        elif i == 3 and "kappa" in params: val = str(params["kappa"])
        elif i == 7 and "M" in params: val = str(params["M"])
        
        new_lines.append(f"{val} {comment}")
    
    with open(os.path.join(target_dir, "inp.txt"), "w") as f:
        f.writelines(new_lines)

def run_python_calc(data_dir):
    # Force HAS_RUST to False to use pure Python backend for comparison
    import resram_core
    resram_core.HAS_RUST = False
    
    obj = load_input(data_dir + "/")
    # We use raman_residual logic to get the loss exactly as defined
    from lmfit import Parameters
    params = Parameters()
    for i, d in enumerate(obj.delta):
        params.add(f"delta{i}", value=d)
    params.add("gamma", value=obj.gamma)
    params.add("transition_length", value=obj.M)
    params.add("kappa", value=obj.k)
    params.add("theta", value=obj.theta)
    params.add("E0", value=obj.E0)
    
    loss, sigma, mismatch = raman_residual(params, obj)
    
    # Also get the actual spectra arrays
    abs_calc = np.real(obj.abs_cross)
    fl_calc = np.real(obj.fl_cross)
    profs_calc = np.real(obj.raman_cross)
    
    return {
        "loss": loss,
        "sigma": sigma,
        "mismatch": mismatch,
        "abs": abs_calc,
        "fl": fl_calc,
        "profs": profs_calc
    }

def run_rust_calc(data_dir, cli_path):
    # First, clear any old output folders in the data_dir
    for item in os.listdir(data_dir):
        if item.startswith("data_") and os.path.isdir(os.path.join(data_dir, item)):
            shutil.rmtree(os.path.join(data_dir, item))

    cmd = [cli_path, "-d", data_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Find the output folder
    output_folder = None
    for item in os.listdir(data_dir):
        if item.startswith("data_") and os.path.isdir(os.path.join(data_dir, item)):
            output_folder = os.path.join(data_dir, item)
            break
    
    if not output_folder:
        print(f"Error: Rust CLI failed to produce output folder. Stderr: {result.stderr}")
        return None

    abs_rust = np.loadtxt(os.path.join(output_folder, "Abs.dat"))
    fl_rust = np.loadtxt(os.path.join(output_folder, "Fl.dat"))
    profs_rust = np.loadtxt(os.path.join(output_folder, "profs.dat")) 
    profs_rust_m_p = profs_rust.T
    
    # Need to load experimental data to calculate loss here
    abs_exp = np.loadtxt(os.path.join(data_dir, "abs_exp.dat"))
    rpumps = np.loadtxt(os.path.join(data_dir, "rpumps.dat"))
    if rpumps.ndim == 0: rpumps = np.array([rpumps])

    # Find the indices of the pump wavelengths in the energy grid
    el_rust = np.loadtxt(os.path.join(output_folder, "EL.dat"))
    rp_indices = []
    for pump in rpumps:
        idx = np.argmin(np.abs(el_rust - pump))
        rp_indices.append(idx)
    
    # Slice the Rust profiles at the pump wavelengths
    profs_rust_at_pumps = profs_rust_m_p[:, rp_indices]
    
    # Calculate Loss using the same logic
    # 1. Correlation
    abs_exp_interp = np.interp(el_rust, abs_exp[:, 0], abs_exp[:, 1])
    corr = np.corrcoef(abs_rust, abs_exp_interp)[0, 1]
    
    # 2. Raman Sigma
    profs_exp = np.loadtxt(os.path.join(data_dir, "profs_exp.dat"))
    if profs_exp.ndim == 1: profs_exp = profs_exp.reshape(-1, 1)
    
    sigma = 1e7 * np.sum((profs_rust_at_pumps - profs_exp)**2)
    loss = sigma + 30 * (1 - corr)
    
    return {
        "loss": loss,
        "sigma": sigma,
        "corr": corr,
        "abs": abs_rust,
        "fl": fl_rust,
        "profs": profs_rust_at_pumps
    }

def main():
    base_dir = "example"
    test_root = "backend_comparison"
    cli_path = os.path.abspath("resram_rust/target/release/resram-cli.exe")
    
    if not os.path.exists(test_root):
        os.makedirs(test_root)
    
    test_cases = [
        {"name": "baseline", "params": {}},
        {"name": "high_gamma", "params": {"gamma": 900.0}},
        {"name": "low_kappa", "params": {"kappa": 0.01}},
        {"name": "shift_e0", "params": {"E0": 17500.0}},
        {"name": "dipole_m", "params": {"M": 2.0}},
        {"name": "scaled_freqs", "params": {"freq_scale": 1.1}},
        {"name": "random_deltas", "params": {"random_deltas": 0.5}},
        {"name": "fewer_modes", "params": {"fewer_modes": 5}},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"Testing case: {case['name']}...")
        data_dir = os.path.join(test_root, case['name'])
        create_dataset(base_dir, data_dir, case['params'])
        
        py_res = run_python_calc(data_dir)
        rust_res = run_rust_calc(data_dir, cli_path)
        
        if rust_res is None: continue
        
        loss_diff = abs(py_res["loss"] - rust_res["loss"])
        loss_rel = loss_diff / py_res["loss"] if py_res["loss"] != 0 else 0
        
        # Spectral deviations
        abs_mae = np.mean(np.abs(py_res["abs"] - rust_res["abs"]))
        fl_mae = np.mean(np.abs(py_res["fl"] - rust_res["fl"]))
        
        print(f"  Loss: Py={py_res['loss']:.6f}, Rust={rust_res['loss']:.6f}, Diff={loss_diff:.2e}")
        print(f"  Abs MAE: {abs_mae:.2e}, Fl MAE: {fl_mae:.2e}")
        
        results.append({
            "case": case["name"],
            "py_loss": py_res["loss"],
            "rust_loss": rust_res["loss"],
            "loss_diff": loss_diff,
            "abs_mae": abs_mae,
            "fl_mae": fl_mae
        })

    print("\nSummary Table:")
    print(f"{'Case':<15} | {'Py Loss':<12} | {'Rust Loss':<12} | {'Rel Diff':<10} | {'Abs MAE':<10}")
    print("-" * 70)
    for r in results:
        rel = (r['loss_diff'] / r['py_loss']) if r['py_loss'] != 0 else 0
        print(f"{r['case']:<15} | {r['py_loss']:<12.4f} | {r['rust_loss']:<12.4f} | {rel:<10.2e} | {r['abs_mae']:<10.2e}")

if __name__ == "__main__":
    main()
