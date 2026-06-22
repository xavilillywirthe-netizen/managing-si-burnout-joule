# Code Workflows

This folder contains the analysis and plotting workflows for the silicon/graphite material-diagnostic study. The root README explains the repository-level structure; this file is a compact navigation guide for running the code.

## Recommended Order

1. `msmr_si_ocp_identification/silicon_ocp_identification_with_msmr.ipynb`

   Identifies the effective silicon OCP from the measured anode and graphite OCP references using the MSMR model.

2. `pathway_dependent_evolution/F2_BOL_EOL_CurrentShare.ipynb`

   Computes material-resolved `dQ/dV`, silicon current share, and transition SoC for BOL and pathway-dependent EOL examples. The companion MATLAB script `F2_plot_v2.m` renders the Figure 2 panels.

3. `diagnostic_algorithm_lifetime_crate/A01_cell_lifetime_diagnostics.ipynb`

   Runs lifetime eSOH diagnostics for selected cells and writes per-cell outputs under `diagnostic_algorithm_lifetime_crate/batch_results/A01_cell_lifetime_diagnostics`.

4. `diagnostic_algorithm_lifetime_crate/A02_crates_diagnostics.ipynb`

   Runs BOL or EOL C-rate diagnostics from `../data/bol_eol_crate_data`.

5. `diagnostic_algorithm_lifetime_crate/A03_append_transitionSoC_to_all_csv.ipynb`

   Appends silicon-dominant transition SoC to lifetime and C-rate diagnostic summary CSV files.

6. `diagnostic_algorithm_lifetime_crate/A04a_run_initial_charge_discard_si_ocp_study.py`
   and `diagnostic_algorithm_lifetime_crate/A04b_evaluate_initial_charge_discard_si_ocp_study.py`

   Evaluate diagnostic robustness versus discarded initial charge data, silicon-OCP treatment, and kinetic compensation.

7. `error_bound_analysis/error_bound_analysis_with_CRB.ipynb`

   Generates the CRB-based sampling-frequency and SoC-window sensitivity outputs used in Figure 5.

8. `plotting/`

   Renders final figure panels from saved workflow outputs. Each figure folder contains scripts and, where useful, a local README.

## Figure Folder Map

- `plotting/f1`: OCP and material-current-share schematic panels.
- `plotting/f2`: points to `pathway_dependent_evolution`, which is the single source for Figure 2.
- `plotting/f3`: silicon-OCP deformation and boundary-case panels.
- `plotting/f4`: lifetime diagnostic accuracy and representative voltage reconstruction.
- `plotting/f5`: C-rate, sampling-frequency, and SoC-window panels.
- `plotting/f6`: lifetime group trends and transition-SoC map.

## Notes

- Create the Python environment from `../environment.yml` before running the notebooks.
- If the data package is distributed separately, download it from
  https://doi.org/10.5281/zenodo.20259651 and place the extracted `data` folder
  at the repository root before running these workflows.
- MATLAB R2023b was used for the MATLAB figure-rendering scripts.
- Notebooks locate the repository root automatically and use paths relative to the repo.
- Several notebooks default to representative cells or selected BOL/EOL cases. Edit the configuration cells to run the full dataset.
- Raw data are read from `../data`; derived outputs are written inside the relevant workflow folders.
- Some output tables keep historical internal column names for compatibility:
  `x100` corresponds to `x_n,100`, `y100` corresponds to `x_p,100`,
  `si_scale_a` corresponds to `s_V`, and `si_shift_b` corresponds to
  `U_off` in the manuscript. These columns should not be renamed unless the
  downstream notebooks and MATLAB plotting scripts are updated at the same
  time.
- Cache folders are generated artifacts and can be deleted when a clean rerun is needed.
- For updates and future releases, please see https://github.com/zhiwen-wan.
