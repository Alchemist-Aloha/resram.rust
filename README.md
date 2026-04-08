# ResRAM_NG: Resonance Raman Excitation Profile Analysis 🧪

ResRAM is a standalone, high-performance theoretical framework and software tool for benchmarking DFT functionals against experimental Femtosecond Stimulated Raman Spectroscopy (FSRS) data.

This version has been ported to **Tauri + Rust** for maximum performance and a modern desktop experience.

---

## 1. Quick Start 🚀

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install)
- [Node.js](https://nodejs.org/) (v18+)
- **Linux users:** Install system dependencies:
  `sudo apt-get install libwebkit2gtk-4.1-dev libgtk-3-dev libsoup-3.0-dev librsvg2-dev patchelf`

### Running the Desktop App
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start in development mode:
   ```bash
   npm run tauri dev
   ```

---

## 2. Key Features ✨

- **High-Performance Rust Engine:** Core physics and integration parallelized with Rayon, providing >4x speedup over Python.
- **Modern UI:** Built with **React** and **Plotly.js** for responsive, publication-quality scientific visualization.
- **Automated Fitting:** Integrated **Powell** and **COBYLA** global optimization algorithms via `nlopt`.
- **Real-time Interaction:** Debounced parameter updates allow you to see theoretical changes instantly as you type.
- **Standalone CLI:** Includes `resram-cli` for batch processing and headless calculations.

---

## 3. Architecture 🏗️

- **Backend (`resram_rust`):** A pure Rust library implementing the IMDHO (Independent Mode Displaced Harmonic Oscillator) model and Brownian oscillator theory.
- **Desktop Bridge (Tauri):** Securely connects the Rust engine to the web frontend.
- **Frontend (React):** Manages application state, parameter tables, and interactive spectral plots.

---

## 4. Input & Data 📂

The application uses a workspace-driven workflow. Open a folder containing:
*   **`inp.toml`**: Modernized configuration (automatically migrated from legacy `inp.txt`).
*   **`freqs.dat`**: Vibrational frequencies (cm⁻¹).
*   **`deltas.dat`**: Initial dimensionless displacements.
*   **`abs_exp.dat`**: Experimental absorption spectrum.
*   **`profs_exp.dat`**: Experimental Raman excitation profiles.

Outputs and optimized results are saved into a dedicated `rust_version/` subfolder.

---

## 5. CLI Usage 💻

For headless calculation or verification:
```bash
cd resram_rust
cargo run --bin resram-cli
```

---

## Acknowledgments 🙌
Developed by **Likun Cai**, based on research from:
*   **Dr. Zachary Piontkowski, Dr. Juan S. Sandoval & Prof. David W. McCamant** (University of Rochester)
*   **Mukamel et al.** - Brownian oscillator models for solvation.

Happy scientific computing!
