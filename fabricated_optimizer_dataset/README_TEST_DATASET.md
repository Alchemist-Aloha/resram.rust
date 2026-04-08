Fabricated optimizer test dataset

Purpose
- Test optimizer behavior (live parameter refresh, convergence, and save flow).

How this dataset was built
- Started from sample_data experimental targets:
  - abs_exp.dat
  - profs_exp.dat
  - rpumps.dat
  - freqs.dat
- Intentionally offset the starting guess:
  - deltas.dat was perturbed mode-by-mode.
  - inp.toml / inp.txt were shifted away from target values:
    - gamma: 910 (target-like sample was 732)
    - theta: 4.5 (target-like sample was 1.0)
    - e0: 16980 (target-like sample was 17250)
    - kappa: 0.22 (target-like sample was 0.1)
    - m: 1.52 (target-like sample was ~1.81)

Expected behavior
- Before fit: mismatch in Abs/REPs.
- During fit: deltas and selected globals should move continuously.
- After fit: lower loss and better overlap with abs_exp/profs_exp.
