# Managing Silicon Burn-Out

This repository contains the code used for the Joule article:

**Wan et al., _Managing silicon burn-out via onboard material diagnostics for durable high-energy density batteries_, Joule (2026).**  
https://doi.org/10.1016/j.joule.2026.102531

The workflows implement material-specific diagnostics for NMC622/SiO-graphite pouch cells, estimate silicon and graphite contributions to electrode capacity, infer the silicon-dominant transition state of charge (SoC), and reproduce the main analysis and figure-generation pipelines used in the article.

## Repository Layout

```text
.
├── code/
│   ├── msmr_si_ocp_identification/
│   ├── pathway_dependent_evolution/
│   ├── diagnostic_algorithm_lifetime_crate/
│   ├── error_bound_analysis/
│   └── plotting/
└── data/
    ├── cell_ocp/
    ├── cell_lifetime_data/
    └── bol_eol_crate_data/
```

- `code/msmr_si_ocp_identification`: identifies the effective silicon OCP used by the multi-species multi-reaction (MSMR) model.
- `code/pathway_dependent_evolution`: computes material-resolved incremental-capacity contributions and silicon current share for BOL/EOL pathway examples.
- `code/diagnostic_algorithm_lifetime_crate`: runs lifetime and C-rate eSOH diagnostics, appends transition SoC, and evaluates rate-dependent robustness.
- `code/error_bound_analysis`: evaluates sampling-frequency and SoC-window requirements using Cramer-Rao-bound analysis.
- `code/plotting`: contains figure-specific plotting scripts and lightweight figure READMEs.
- `data`: contains OCP references, lifetime cell data, the test matrix, and BOL/EOL C-rate files. See [data/README.md](data/README.md).

## Data

The full dataset is distributed separately on Zenodo:

https://doi.org/10.5281/zenodo.20259651

After download and extraction, the dataset is expected to be placed at the repository root as `data/`. The included `cell_lifetime_data/test_matrix.xlsx` lists all 60 lifetime cells and their cycling or calendar-aging conditions. The lifetime data are split into `type1` and `type2` folders because the processed Voltaiq export schema changed across the dataset; the diagnostic loader selects the correct schema from the cell ID.

The lifetime-aging trends represented by this dataset were previously discussed in:

Wan, Zhiwen, et al. "Degradation and expansion of lithium-ion batteries with silicon/graphite anodes: Impact of pretension, temperature, C-rate and state-of-charge window." *eTransportation* 24 (2025): 100416.

This repository uses that dataset to develop and demonstrate onboard material diagnostics and silicon-current-share analysis for the Joule article above.

## Requirements

The analysis workflows use Python 3.11.5 notebooks/scripts and MATLAB plotting scripts. A conda environment file is provided:

```text
conda env create -f environment.yml
conda activate managing-si-burnout
python -m ipykernel install --user --name managing-si-burnout --display-name "Python (managing-si-burnout)"
```

MATLAB R2023b was used for the final figure-rendering scripts. The code was organized to use repository-relative paths; no user-specific absolute paths should be required after cloning the repository and placing the `data` folder at the root.

## Quick Start

1. Clone or download this repository.
2. Place the downloaded `data` folder at the repository root if it is distributed separately.
3. Create and activate the conda environment listed above.
4. Start Jupyter from anywhere inside the repository.
5. Follow the workflow map in [code/README.md](code/README.md).

For a compact diagnostic run, start with:

```text
code/diagnostic_algorithm_lifetime_crate/A01_cell_lifetime_diagnostics.ipynb
code/diagnostic_algorithm_lifetime_crate/A02_crates_diagnostics.ipynb
code/diagnostic_algorithm_lifetime_crate/A03_append_transitionSoC_to_all_csv.ipynb
```

Some notebooks default to representative cells or representative BOL/EOL datasets to keep runtime manageable. Edit the configuration cells in each notebook to run the full dataset.

## Figure Workflows

The figure-generation code is organized under `code/plotting` and, where appropriate, points back to the analysis workflow that generates the data.

- Figure 1: `code/plotting/f1`
- Figure 2: `code/pathway_dependent_evolution`
- Figure 3: `code/plotting/f3`
- Figure 4: `code/plotting/f4`
- Figure 5: `code/plotting/f5` and `code/error_bound_analysis`
- Figure 6: `code/plotting/f6`

Each figure folder contains either the plotting scripts directly or a README explaining where the source workflow lives.

## Outputs And Cache Files

Most workflows write derived files under local `outputs`, `figures`, `batch_results`, or cache folders inside `code/`. These files can be regenerated from the provided data. Raw files under `data/` should not be edited in place.

Common generated folders include:

- `code/msmr_si_ocp_identification/outputs`
- `code/pathway_dependent_evolution/outputs`
- `code/diagnostic_algorithm_lifetime_crate/batch_results`
- `code/error_bound_analysis/outputs`
- `code/plotting/*/figures`

Cache folders can be deleted and rebuilt if a workflow needs to be rerun from raw data.

## Updates

For updates and future releases, please see:

https://github.com/zhiwen-wan

## License

The software source code in this repository is released under the BSD 3-Clause
License; see [LICENSE](LICENSE).

The dataset files under `data/` are released under the Creative Commons
Attribution 4.0 International License (CC BY 4.0); see
[data/LICENSE](data/LICENSE).

## Citation

Citation metadata for this repository are provided in [CITATION.cff](CITATION.cff):

**Wan et al., _Managing silicon burn-out via onboard material diagnostics for durable high-energy density batteries_, Joule (2026).**  
https://doi.org/10.1016/j.joule.2026.102531

The archived code package is available from Zenodo:

https://doi.org/10.5281/zenodo.20259383

For background on the lifetime-aging dataset and trends, cite:

Wan, Zhiwen, et al. "Degradation and expansion of lithium-ion batteries with silicon/graphite anodes: Impact of pretension, temperature, C-rate and state-of-charge window." *eTransportation* 24 (2025): 100416.
