# ResRAM Port to Tauri & Rust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the ResRAM analysis tool to a high-performance Tauri desktop application with a pure Rust physics engine and React/Plotly.js frontend.

**Architecture:** A standalone Rust library (`resram-engine`) handles the IMDHO logic and `nlopt` integration, while Tauri provides the desktop bridge to a React-based frontend for parameter management and spectral visualization.

**Tech Stack:** Tauri, Rust (`ndarray`, `nlopt-rust`, `rayon`, `serde`, `toml`), React, TypeScript, Plotly.js.

---

### Phase 1: `resram-engine` Core (Rust)

#### Task 1: Update `Cargo.toml` and Setup Core Structure
- **Files:**
  - Modify: `resram_rust/Cargo.toml`
  - Create: `resram_rust/src/models.rs`
- [ ] **Step 1: Update dependencies in `Cargo.toml`**
  ```toml
  [package]
  name = "resram_rust"
  version = "0.2.0"
  edition = "2021"

  [lib]
  crate-type = ["cdylib", "rlib"]

  [dependencies]
  ndarray = "0.15"
  num-complex = "0.4"
  numpy = "0.21"
  pyo3 = { version = "0.21", features = ["extension-module"] }
  rayon = "1.8"
  serde = { version = "1.0", features = ["derive"] }
  toml = "0.8"
  nlopt = "0.7"
  anyhow = "1.0"
  thiserror = "1.0"
  ```
- [ ] **Step 2: Define the `Config` and `Result` data structures**
  ```rust
  use serde::{Serialize, Deserialize};
  use ndarray::Array1;

  #[derive(Debug, Serialize, Deserialize, Clone)]
  pub struct ResRamConfig {
      pub gamma: f64,
      pub theta: f64,
      pub e0: f64,
      pub kappa: f64,
      pub m: f64,
      pub n: f64,
      pub temp: f64,
      // ... other fields from inp.txt
  }
  ```
- [ ] **Step 3: Commit**
  ```bash
  git add resram_rust/Cargo.toml resram_rust/src/models.rs
  git commit -m "chore: setup resram_rust dependencies and models"
  ```

#### Task 2: Implement the `nlopt` Optimizer Wrapper
- **Files:**
  - Create: `resram_rust/src/optimizer.rs`
- [ ] **Step 1: Write a function to run Powell/COBYLA using `nlopt`**
- [ ] **Step 2: Implement the objective function callback**
- [ ] **Step 3: Add a progress callback for UI updates**
- [ ] **Step 4: Commit**
  ```bash
  git add resram_rust/src/optimizer.rs
  git commit -m "feat: add nlopt optimizer integration"
  ```

#### Task 3: Refactor `lib.rs` for standalone use
- **Files:**
  - Modify: `resram_rust/src/lib.rs`
- [ ] **Step 1: Decouple `calculate_cross_sections` from PyO3 types**
- [ ] **Step 2: Ensure it works with standard `ndarray` views**
- [ ] **Step 3: Commit**
  ```bash
  git add resram_rust/src/lib.rs
  git commit -m "refactor: decouple core logic from PyO3"
  ```

### Phase 2: CLI & Migration (Rust)

#### Task 4: Implement `inp.toml` and Migration
- **Files:**
  - Create: `resram_rust/src/config.rs`
- [ ] **Step 1: Write a parser for `inp.toml`**
- [ ] **Step 2: Write a migrator for `inp.txt` -> `inp.toml`**
- [ ] **Step 3: Commit**
  ```bash
  git add resram_rust/src/config.rs
  git commit -m "feat: add inp.toml support and migration"
  ```

#### Task 5: Create a Test CLI
- **Files:**
  - Create: `resram_rust/src/bin/resram-cli.rs`
- [ ] **Step 1: Implement a simple CLI to run a single calculation**
- [ ] **Step 2: Verify it produces consistent results with the Python version**
- [ ] **Step 3: Commit**
  ```bash
  git add resram_rust/src/bin/resram-cli.rs
  git commit -m "feat: add resram-cli for engine verification"
  ```

### Phase 3: Tauri Backend (Rust)

#### Task 6: Initialize Tauri Project
- **Files:**
  - Create: `src-tauri/` (via `npm create tauri-app`)
- [ ] **Step 1: Run `npm create tauri-app@latest`**
- [ ] **Step 2: Configure `tauri.conf.json`**
- [ ] **Step 3: Commit**
  ```bash
  git add .
  git commit -m "chore: initialize tauri project"
  ```

#### Task 7: Implement Tauri Commands
- **Files:**
  - Modify: `src-tauri/src/main.rs`
- [ ] **Step 1: Add `load_data` command**
- [ ] **Step 2: Add `run_fit` command (async with events)**
- [ ] **Step 3: Commit**
  ```bash
  git add src-tauri/src/main.rs
  git commit -m "feat: implement tauri commands"
  ```

### Phase 4: React Frontend (TypeScript)

#### Task 8: Setup React and Plotly.js
- [ ] **Step 1: Install dependencies (`plotly.js`, `react-plotly.js`, `lucide-react`)**
- [ ] **Step 2: Create basic layout with Sidebar and Plot area**
- [ ] **Step 3: Commit**

#### Task 9: Implement Parameter Sidebar
- [ ] **Step 1: Build the table for vibrational modes**
- [ ] **Step 2: Sync UI state with Tauri backend via events/commands**
- [ ] **Step 3: Commit**

#### Task 10: Implement Plotly Visualization
- [ ] **Step 1: Build the Absorption/FL plot**
- [ ] **Step 2: Build the REP plot**
- [ ] **Step 3: Implement debounced updates for live interaction**
- [ ] **Step 4: Commit**

### Phase 5: Validation & Cleanup

#### Task 11: End-to-End Testing
- [ ] **Step 1: Run the full Bodipy fit in the new app**
- [ ] **Step 2: Compare final results with `delta_fit.dat`**
- [ ] **Step 3: Commit**

#### Task 12: Final Cleanup and Documentation
- [ ] **Step 1: Update README.md**
- [ ] **Step 2: Prepare final binary release**
- [ ] **Step 3: Commit**
