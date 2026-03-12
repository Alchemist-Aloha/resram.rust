import os
import numpy as np
import shutil
from resram_core import load_input, cross_sections, run_save, raman_residual, param_init
import matplotlib.pyplot as plt

def setup_mock_data(path):
    """Create a temporary directory with minimal valid input files."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    # Create mock freqs.dat
    freqs = np.array([500.0, 1000.0, 1500.0])
    np.savetxt(os.path.join(path, "freqs.dat"), freqs)
    
    # Create mock deltas.dat
    deltas = np.array([0.5, 0.3, 0.1])
    np.savetxt(os.path.join(path, "deltas.dat"), deltas)
    
    # Create mock inp.txt
    inp_content = [
        "732 # gamma linewidth parameter (cm^-1)",
        "1 # theta static inhomogeneous linewidth parameter (cm^-1)",
        "17250.0 # E0 (cm^-1)",
        "0.1 # kappa solvent parameter",
        "0.0005 # time step (ps)",
        "200.0 # number of time steps",
        "2000.0 # range plus and minus E0 to calculate lineshapes",
        "1.8 # transition length M (Angstroms)",
        "1.33 # refractive index n",
        "0 # start raman shift axis (cm^-1)",
        "2500 # end raman shift axis (cm^-1)",
        "10 # rshift axis step size (cm^-1)",
        "20 # raman spectrum resolution (cm^-1)",
        "298.0 # Temperature (K)",
        "1 # convergence for sums",
        "0 # Boltz Toggle"
    ]
    with open(os.path.join(path, "inp.txt"), "w") as f:
        f.write("\n".join(inp_content))
    
    # Create mock rpumps.dat
    rpumps = np.array([17000.0, 17500.0])
    np.savetxt(os.path.join(path, "rpumps.dat"), rpumps)

    # Mock experimental data
    dummy_spec = np.zeros((100, 2))
    dummy_spec[:, 0] = np.linspace(15000, 19000, 100)
    dummy_spec[:, 1] = np.exp(-((dummy_spec[:, 0] - 17250)**2) / 1000000)
    np.savetxt(os.path.join(path, "abs_exp.dat"), dummy_spec)
    
    profs_exp = np.ones((3, 2)) * 0.1
    np.savetxt(os.path.join(path, "profs_exp.dat"), profs_exp)

def run_tests():
    test_dir = "test_run_dir"
    setup_mock_data(test_dir)
    data_path = test_dir + "/"
    
    print("Running Tests...")
    
    # Test 1: Load Input
    print("Testing load_input...", end=" ")
    obj = load_input(dir=data_path)
    assert len(obj.wg) == 3
    assert obj.gamma == 732.0
    print("PASSED")
    
    # Test 2: Cross Sections
    print("Testing cross_sections...", end=" ")
    abs_c, fl_c, ram_c, _, _ = cross_sections(obj)
    assert len(abs_c) > 0
    assert ram_c.shape == (3, len(obj.convEL))
    print("PASSED")
    
    # Test 3: Run Save
    print("Testing run_save...", end=" ")
    timestamp = "smoke_test"
    run_save(obj, timestamp)
    save_path = timestamp + "_data" # It creates it in the CWD
    assert os.path.exists(save_path)
    assert os.path.exists(os.path.join(save_path, "Abs.dat"))
    print("PASSED")
    
    # Test 4: Residuals
    print("Testing raman_residual...", end=" ")
    fit_switch = [1] * 9
    params = param_init(fit_switch, obj=obj)
    loss, _, _ = raman_residual(params, fit_obj=obj)
    assert loss > 0
    print("PASSED")
    
    print("\nAll tests passed successfully!")
    
    # Cleanup
    shutil.rmtree(test_dir)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

if __name__ == "__main__":
    run_tests()
