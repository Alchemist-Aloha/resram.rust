# ResRAM: Resonance Raman Excitation Profile Analysis 🧪

Welcome! This program helps scientists calculate and "fit" (match) theoretical models to experimental data from **Resonance Raman Spectroscopy**. 

---

## 1. Installation 🛠️

You can now install this program as a Python package. This makes it easy to use `import resram_core` in any of your scripts or notebooks.

### Using `uv` (Recommended)
If you have `uv` installed, run:
```bash
uv pip install .
```

### Using `pip`
Or, use standard pip:
```bash
pip install .
```

## 2. Setting Up Your Computer 💻

Before running the code, you need a few things installed:

1.  **Python:** Download and install the latest version from [python.org](https://www.python.org/).
2.  **Required Libraries:** Open your terminal (or Command Prompt) and run this command to install the "math and plotting" tools the program needs:
    ```bash
    pip install numpy matplotlib scipy lmfit ipykernel tqdm
    ```
3.  **Jupyter Notebook:** This program is designed to be run inside a "Notebook" (a file ending in `.ipynb`). Most people use **VS Code** or **JupyterLab** to open them.

---

## 3. Optional: Rust Acceleration ⚡

This project includes an optional **Rust backend** (`resram_rust`) that drastically speeds up the rigorous integration calculations required for resonance Raman cross sections. When compiled, the Rust extension provides a **~4x performance speedup** while maintaining numerical consistency with the Python implementation.

### How to Compile the Rust Backend

1. **Install Rust:** Download and install Rust via [rustup.rs](https://rustup.rs/):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. **Build the extension:** Navigate to the `resram_rust` directory and compile:
   ```bash
   cd resram_rust
   cargo build --release
   cd ..
   ```
3. **Copy the shared library:** Move the compiled library into your working directory where the notebook or scripts will run.
   * **Linux:** `cp resram_rust/target/release/libresram_rust.so ./resram_rust.so`
   * **macOS:** `cp resram_rust/target/release/libresram_rust.dylib ./resram_rust.so`
   * **Windows:** `copy resram_rust\target\release\resram_rust.dll .\resram_rust.pyd`

Once the compiled file is in the root project folder, the Python engine (`resram_core.py`) will automatically detect it and seamlessly offload calculations to the faster Rust engine!

---

## 4. What does this program actually do? 🤔

Imagine you have a molecule (like Bodipy). When you hit it with a laser, it vibrates. This program:
1.  **Calculates** how that molecule should look in an Absorption or Raman spectrum based on "Displacements" (how much the molecule's shape changes when it's excited).
2.  **Compares** that calculation to your real experimental data.
3.  **Adjusts** the variables automatically until the calculation matches the experiment as closely as possible.

---

## 5. The Important Files 📂

The program looks for specific files in its folder. Here is what you need to know:

*   **`resram_core.py`**: The "Engine." This contains all the complex math. You usually don't need to change this.
*   **`inp.txt`**: The "Settings." This is where you set the Temperature, Refractive Index, and initial guesses for the math.
*   **`freqs.dat`**: A list of vibrational frequencies (wavenumber cm⁻¹).
*   **`deltas.dat`**: A list of initial guesses for the "Displacements" (how far the atoms move).
*   **`abs_exp.dat`**: Your real experimental absorption spectrum (vs. wavenumber).
*   **`fl_exp.dat`**: Your real experimental fluorescence spectrum (vs. wavenumber).
*   **`profs_exp.dat`**: Your real experimental Raman extinction profiles.
*   **`rpumps.dat`**: Your real experimental Raman laser energies (wavenumber cm⁻¹).

---

## 6. How to Run It 🚀

1.  Open the file **`FSRSanalysis_v2.ipynb`** in your Notebook editor.
2.  The notebook is divided into "Cells." You can run a cell by clicking the **Play** button next to it.
3.  **Step 1:** The first cells load the engine (`resram_core.py`).
4.  **Step 2:** Follow the instructions in the notebook to load your data.
5.  **Step 3:** Run the "Fitting" cell. You will see the program try different numbers to make the graphs match!

---

## 7. Understanding the Results 📊

When the program finishes, it creates a folder named with the current date (e.g., `2026-03-12_data`). Inside, you will find:
*   **`Abs.dat` / `Fl.dat`**: The calculated Absorption and Fluorescence spectra.
*   **`profs.dat`**: The Raman Excitation Profiles.
*   **`output.txt`**: A summary of the final "best" numbers found by the program.

---

## 💡 Pro-Tips for Beginners
*   **File Names:** Don't rename the `.dat` files unless you also change the code. The program is specifically looking for names like `freqs.dat`.
*   **Errors:** If you see a "FileNotFoundError," it usually means the terminal is looking in the wrong folder. Make sure your terminal path matches where the files are saved.

Happy scientific computing!

## Acknowledgments 🙌
This program was developed by Likun Cai and is based on the work of Dr. Zachary Piontkowski, Dr Juan S. Sandoval and many others in the field of Raman spectroscopy. For more details, check out their research:

Piontkowski, Z. (2020). Excited state torsions and electron transfer in dye-sensitizers for light harvesting and photodynamic therapy. University of Rochester.

Sandoval, J. S., & McCamant, D. W. (2023). The best models of Bodipy’s electronic excited state: comparing predictions from various DFT functionals with measurements from femtosecond stimulated Raman spectroscopy. The Journal of Physical Chemistry A, 127(39), 8238-8251.

Li, B., Johnson, A. E., Mukamel, S., & Myers, A. B. (1994). The Brownian oscillator model for solvation effects in spontaneous light emission and their relationship to electron transfer. Journal of the American Chemical Society, 116(24), 11039-11047.
