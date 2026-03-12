import os
import numpy as np
import pytest
import shutil
from datetime import datetime
from resram_core import load_input, cross_sections, run_save, raman_residual, param_init

@pytest.fixture
def mock_data_dir(tmp_path):
    """
    Fixture to create a temporary directory with minimal valid input files.
    """
    d = tmp_path / "test_data"
    d.mkdir()
    
    # Create mock freqs.dat (3 modes)
    freqs = np.array([500.0, 1000.0, 1500.0])
    np.savetxt(d / "freqs.dat", freqs)
    
    # Create mock deltas.dat
    deltas = np.array([0.5, 0.3, 0.1])
    np.savetxt(d / "deltas.dat", deltas)
    
    # Create mock inp.txt based on existing format
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
    with open(d / "inp.txt", "w") as f:
        f.write("\n".join(inp_content))
    
    # Create mock rpumps.dat
    rpumps = np.array([17000.0, 17500.0])
    np.savetxt(d / "rpumps.dat", rpumps)

    # Create mock experimental data for residual tests
    # EL grid size is determined by load_input (usually 1000 or based on range)
    # We'll just create some dummy files
    dummy_spec = np.zeros((100, 2))
    dummy_spec[:, 0] = np.linspace(15000, 19000, 100)
    dummy_spec[:, 1] = np.exp(-((dummy_spec[:, 0] - 17250)**2) / 1000000)
    np.savetxt(d / "abs_exp.dat", dummy_spec)
    
    # Mock profs_exp.dat (modes x pumps) = (3 x 2)
    profs_exp = np.ones((3, 2)) * 0.1
    np.savetxt(d / "profs_exp.dat", profs_exp)

    return str(d) + "/"

def test_load_input(mock_data_dir):
    """Test if input files are loaded correctly."""
    obj = load_input(dir=mock_data_dir)
    assert len(obj.wg) == 3
    assert len(obj.delta) == 3
    assert obj.gamma == 732.0
    assert obj.E0 == 17250.0
    assert obj.T == 298.0
    assert hasattr(obj, 'abs_exp')
    assert obj.abs_exp.shape[1] == 2

def test_cross_sections(mock_data_dir):
    """Test the core cross section calculation."""
    obj = load_input(dir=mock_data_dir)
    abs_c, fl_c, ram_c, b_s, b_c = cross_sections(obj)
    
    # Check shapes
    assert len(abs_c) == len(obj.convEL)
    assert len(fl_c) == len(obj.convEL)
    # raman_cross shape: (modes, EL)
    assert ram_c.shape == (3, len(obj.convEL))
    
    # Basic value checks (should be real or complex but not NaN)
    assert not np.any(np.isnan(abs_c))
    assert not np.any(np.isnan(ram_c))

def test_run_save(mock_data_dir):
    """Test if output data is saved correctly."""
    obj = load_input(dir=mock_data_dir)
    timestamp = "test_run"
    data_obj = run_save(obj, timestamp)
    
    save_dir = mock_data_dir + timestamp + "_data"
    assert os.path.exists(save_dir)
    assert os.path.exists(os.path.join(save_dir, "Abs.dat"))
    assert os.path.exists(os.path.join(save_dir, "profs.dat"))
    assert os.path.exists(os.path.join(save_dir, "output.txt"))
    
    # Check if we can load it back
    assert data_obj.gamma == 732.0
    assert data_obj.E0 == 17250.0
    
    # Cleanup test folder
    shutil.rmtree(save_dir)

def test_raman_residual(mock_data_dir):
    """Test the residual function used for optimization."""
    obj = load_input(dir=mock_data_dir)
    
    # Create a switch array for param_init
    # deltas (3) + gamma + M + theta + kappa + E0 (total 3+5=8)
    # We'll just fix E0 at index 5 of the 5 global params
    fit_switch = [1, 1, 1, 1, 1, 1, 1, 1, 1] 
    params = param_init(fit_switch, obj=obj)
    
    loss, total_sigma, abs_mismatch = raman_residual(params, fit_obj=obj)
    
    assert loss > 0
    assert isinstance(total_sigma, (float, np.float64))
    assert isinstance(abs_mismatch, (float, np.float64))

def test_boltz_states(mock_data_dir):
    """Test Boltzmann state generation (even if toggle is 0, we can call it)."""
    obj = load_input(dir=mock_data_dir)
    states, coefs, energy = obj.boltz_states()
    assert len(coefs) > 0
    assert np.isclose(np.sum(coefs), 1.0)
