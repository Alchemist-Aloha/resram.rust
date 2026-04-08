# Design Specification: ResRAM Port to Tauri & Rust

**Date:** 2026-04-07
**Project:** ResRAM Spectroscopic Analysis Port
**Status:** Approved

## 1. Objective
Port the existing ResRAM (Resonance Raman) spectroscopic analysis tool from a Python/PySide6/Rust-extension hybrid to a standalone, high-performance desktop application using **Tauri**, **Rust**, and **React**.

## 2. Key Requirements
- **Performance:** Move all heavy computation (IMDHO logic and optimization loops) to a pure Rust backend.
- **Optimization:** Implement **Powell** and **COBYLA** algorithms by calling the `nlopt-rust` crate.
- **Modern UI:** Replace PySide6 with a React-based frontend and Plotly.js for scientific visualization.
- **Compatibility:** Maintain support for existing `.dat` files while transitioning `inp.txt` to `inp.toml`.
- **Workflow:** Generate a dedicated `rust_version/` folder for new outputs and configuration.

## 3. Architecture

### 3.1. Backend (Rust Engine-First)
- **`resram-engine` Library:** A standalone Rust library (`crate`) for the physics logic.
  - **IMDHO Model:** Ported from `resram_core.py` and `resram_rust`.
  - **Linear Algebra:** Uses `ndarray` for vectorization.
  - **Parallelism:** Uses `rayon` for multi-core spectral integration.
- **Optimization Loop:** 
  - Uses `nlopt-rust` for Powell and COBYLA.
  - Runs in a background thread to prevent UI blocking.
  - Emits Tauri events (throttled/debounced at ~200ms) with current iteration state (Loss, Correlation, Parameters).
- **Tauri Integration:** 
  - Commands for loading data, starting calculations, and initiating fits.
  - Event-based streaming of optimization progress to the frontend.

### 3.2. Frontend (React + Plotly.js)
- **Framework:** React with TypeScript.
- **Visualization:** Plotly.js for interactive, publication-quality plots (Abs/FL, REPs, Loss history).
- **State Management:** React hooks for managing parameter tables and plot data.
- **Interactivity:** 
  - Debounced (300ms) parameter updates to the backend for live "manual" fitting.
  - Structured sidebar for vibrational modes ($\Delta_k$) and global parameters ($\Gamma$, $\theta$, $E_0$, etc.).

### 3.3. Data Management
- **Input:** `inp.toml` for global and solver parameters. Legacy `.dat` files (`freqs.dat`, `deltas.dat`, etc.) remain the same.
- **Output:** All Rust-generated data saved into a `rust_version/` subfolder.
- **Migration:** Automatic conversion of `inp.txt` to `inp.toml` upon first project load in the Rust version.

## 4. Performance & Scaling
- **Target:** Achieve at least the performance of the current Rust extension (~4x speedup over Python) while eliminating Python/C++ interop overhead.
- **Concurrency:** Fully parallelized calculation of multiple vibrational modes and spectral grids.

## 5. Testing & Validation
- **Regression Suite:** Compare Rust output against existing Python/C++ "Ground Truth" for Bodipy datasets.
- **Numerical Precision:** Target $10^{-6}$ relative error between Python and Rust implementations.
- **Optimizer Benchmarks:** Verify that `nlopt` Powell/COBYLA converge to the same global minima as `lmfit`.

## 6. Implementation Strategy
1. **Phase 1: `resram-engine` Core:** Port IMDHO logic and `nlopt` integration to pure Rust.
2. **Phase 2: CLI/Scaffolding:** Build a CLI tool to verify the engine and `inp.toml` migration.
3. **Phase 3: Tauri Backend:** Implement Tauri commands and event emitters.
4. **Phase 4: React Frontend:** Build the Dashboard, Sidebar, and Plotly integration.
5. **Phase 5: Validation & Cleanup:** Final regression testing and standalone binary packaging.
