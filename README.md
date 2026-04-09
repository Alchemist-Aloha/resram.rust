# ResRam.rust: Resonance Raman Excitation Profile Analysis 🧪

ResRAM is a standalone, high-performance software tool for benchmarking DFT functionals against experimental Femtosecond Stimulated Raman Spectroscopy (FSRS) data. It implements the IMDHO (Independent Mode Displaced Harmonic Oscillator) model and Brownian oscillator theory.

This version is powered by an optimized **Rust** engine with a modern **Tauri** desktop interface.

---

## 1. Basic Usage 🚀

### Using the GUI 
The GUI provides a real-time, interactive environment for exploring parameters and fitting spectra.

1.  **Launch the Application:** Open the `resram-tauri` executable for your platform.
2.  **Open Workspace:** Click the **Folder Icon** in the sidebar and select a directory containing your input files (see [Input Files](#3-input--data)).
3.  **Explore:** Change parameters like **Gamma**, **Theta**, or **E0**. The plots will update automatically (300ms debounce).
4.  **Fit:** Select which parameters or vibrational modes to optimize using the "Fit?" checkboxes, then click **Start Fit**. Results are automatically saved to a new timestamped folder.

### Using the CLI
The `resram-cli` is ideal for headless calculations and batch processing.

```bash
# Run a single calculation in the current directory
./resram-cli

# Run an optimization (fit) using settings from fit.toml
./resram-cli --fit

# Specify a custom directory and skip saving results
./resram-cli --dir ./my_data --no-save
```

**Common Flags:**
- `-d, --dir <DIR>`: Root directory containing input files [default: .].
- `-f, --fit`: Trigger optimization (fitting) using `fit.toml`.
- `-i, --input <FILE>`: Custom input TOML file [default: `inp.toml`].
- `--no-save`: Skip saving results to a timestamped folder.

---

## 2. Key Features ✨

- **Optimized Rust Engine:** Integration loops are parallelized with Rayon and use Euler-method phase advancement, providing massive speedups over Python.
- **Automated Fitting:** Integrated global optimization algorithms (COBYLA, Powell, etc.) via `nlopt`.

- **Cross-Platform:** Native binaries available for Windows, Linux, and macOS (Intel & Silicon).

---

## 3. Input & Data 📂

The application requires a workspace directory containing:
*   **`inp.toml`**: Main simulation parameters (automatically migrated from legacy `inp.txt` if found).
*   **`freqs.dat`**: Vibrational frequencies ($cm^{-1}$).
*   **`deltas.dat`**: Initial dimensionless displacements.
*   **`abs_exp.dat`**: Experimental absorption spectrum (2 columns: $cm^{-1}$ and intensity).
*   **`profs_exp.dat`**: Experimental Raman excitation profiles.
*   **`rpumps.dat`**: Pump wavelengths ($cm^{-1}$) for Raman calculation.
*   **`fit.toml`** (Optional): Settings for the optimizer.

---

## 4. Building from Source 🛠️

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (latest stable)
- [Node.js](https://nodejs.org/) (v18+)
- **Linux (Ubuntu/Debian):**
  `sudo apt-get install libwebkit2gtk-4.1-dev libgtk-3-dev libappindicator3-dev librsvg2-dev patchelf libnlopt-dev pkg-config`

### Build the GUI (Tauri)
```bash
# Install frontend dependencies
npm install

# Build the release bundle (MSI, DMG, or AppImage)
npm run tauri build --release
```
The compiled installer will be located in `src-tauri/target/release/bundle/`.

### Build the CLI
```bash
cd resram_rust

# Build the standalone binary
cargo build --release --bin resram-cli
```
The binary will be located in `resram_rust/target/release/resram-cli`.

---

## Acknowledgments 🙌
Developed by **Likun Cai**, based on research from:
*   **Dr. Zachary Piontkowski, Dr. Juan S. Sandoval & Prof. David W. McCamant** (University of Rochester)
*   **Mukamel et al.** - Brownian oscillator models for solvation.

Happy scientific computing!
