# ResRAM_NG: Resonance Raman Excitation Profile Analysis 🧪

Welcome! This program helps scientists calculate and "fit" (match) theoretical models to experimental data from **Resonance Raman Spectroscopy**. 

It implements the Independent Mode Displaced Harmonic Oscillator (IMDHO) formalism and Brownian oscillator theory to benchmark DFT functionals against experimental Femtosecond Stimulated Raman Spectroscopy (FSRS) data.

---

## 1. Quick Start (GUI) 🚀

The easiest way to use ResRAM is through the interactive graphical interface.

### Running from source
If you have `uv` installed, simply run:
```bash
uv run ResRamQt.py
```

### Pre-compiled Binaries
You can download standalone executables for **Windows**, **Linux**, and **macOS** from the [GitHub Releases](https://github.com/your-repo/releases) page. No Python installation is required for these versions.

> **Note for Linux users:** If you see an error about `libEGL.so.1`, install the following system libraries:
> `sudo apt-get install libegl1 libgl1-mesa-glx`

---

## 2. GUI Features ✨

The ResRAM GUI is optimized for a smooth, responsive experience:
*   **Real-time Visualization:** See Absorption, Fluorescence, and Raman Excitation Profiles update as you change parameters.
*   **Background Calculations:** Heavy spectroscopic simulations run in a separate thread, keeping the interface snappy.
*   **Debounced Updates:** Table edits are debounced (300ms) to prevent lag while typing.
*   **Interactive Control:** 
    *   Toggle individual vibrational modes to include/exclude them from plots.
    *   Select which parameters (deltas, gamma, E0, etc.) should be varied during the automated fitting process.
*   **Integrated Fitting:** Run the `lmfit` optimizer directly from the GUI and see the "best fit" results in real-time.

---

## 3. Installation & Development 🛠️

If you want to use the core library in your own scripts or notebooks:

### Using `uv` (Recommended)
```bash
uv pip install .
```

### Using `pip`
```bash
pip install .
```

### Required Python Libraries
If not using `uv`, manual installation:
```bash
pip install numpy matplotlib scipy lmfit PyQt6 pyqtgraph
```

---

## 4. Optional: Rust Acceleration ⚡

This project includes an optional **Rust backend** (`resram_rust`) that provides a **~4x performance speedup**.

### How to Compile the Rust Backend
1. **Install Rust:** [rustup.rs](https://rustup.rs/)
2. **Build:**
   ```bash
   cd resram_rust
   cargo build --release
   ```
3. **Copy the library** to the root project folder:
   * **Linux:** `cp resram_rust/target/release/libresram_rust.so ./resram_rust.so`
   * **macOS:** `cp resram_rust/target/release/libresram_rust.dylib ./resram_rust.so`
   * **Windows:** `copy resram_rust\target\release\resram_rust.dll .\resram_rust.pyd`

---

## 5. Input Data Format 📂

The program (GUI and Notebook) expects specific `.dat` files in the working directory:

*   **`inp.txt`**: Global settings (Temperature, Refractive Index, etc.).
*   **`freqs.dat`**: Vibrational frequencies (cm⁻¹).
*   **`deltas.dat`**: Initial dimensionless displacements.
*   **`abs_exp.dat`**: Experimental absorption spectrum.
*   **`fl_exp.dat`**: Experimental fluorescence spectrum.
*   **`profs_exp.dat`**: Experimental Raman excitation profiles.
*   **`rpumps.dat`**: Experimental laser excitation wavenumbers.

---

## 6. Notebook Usage (`.ipynb`) 📓

If you prefer a programmatic approach, use **`FSRSanalysis_v2.ipynb`**:
1. Open in VS Code or JupyterLab.
2. Run the cells sequentially to load data, perform calculations, and execute the fitting loop.
3. Results are saved into a time-stamped directory (e.g., `2026-03-12_data`).

---

## Acknowledgments 🙌
Developed by **Likun Cai**, based on theoretical frameworks and research from:
*   **Dr. Zachary Piontkowski** (University of Rochester)
*   **Dr. Juan S. Sandoval & Dr. David W. McCamant** (University of Rochester) - *J. Phys. Chem. A* 2023, 127, 39, 8238–8251.
*   **Mukamel et al.** - Brownian oscillator models for solvation.

Happy scientific computing!
