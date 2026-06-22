# Figure 4

Figure 4 summarizes the lifetime diagnostic accuracy and shows a representative
voltage/electrode-potential reconstruction example.

The plotting scripts read prepared inputs from:

```text
data/lifetime_summaries
data/voltage_fit_example
```

The repository includes the summary CSV files needed to reproduce the figure.
For full regeneration of the model-accessible capacity column `C_est`, the
notebook reads raw lifetime data from:

```text
../../../data/cell_lifetime_data
```

Run the Figure 4 workflow in this order:

1. `F4_get_model_capacity.ipynb`

   Recomputes `C_est` from the measured C/20 charge voltage endpoints and the
   fitted OCP model. The notebook reads raw lifetime data from
   `../../../data/cell_lifetime_data` and writes the augmented lifetime summary
   CSV files in:

   ```text
   data/lifetime_summaries
   ```

   If a local data copy is intentionally incomplete, existing `C_est` values are
   preserved and the repo-local fallback table is used only when it is
   consistent with the measured charge capacity:

   ```text
   data/model_capacity_from_voltage_limits.csv
   ```

2. `F4_1.m`

   Generates the lifetime diagnostic accuracy panels:

   ```text
   figures/F4_1_capacity_error_pct.svg
   figures/F4_1_voltage_rmse.svg
   figures/F4_1_dvdq_rmse.svg
   ```

3. `F4_2.m`

   Generates the Figure 4b electrode-level validation panels using a prepared
   three-electrode comparison table. The table contains measured full-cell,
   anode, and cathode potentials from a 0.01C three-electrode charge, together
   with the corresponding potentials reconstructed from the fitted eSOH
   parameters:

   ```text
   data/voltage_fit_example/rpt_02_electrode_potential_comparison.csv
   ```

   The outputs are:

   ```text
   figures/F4_2_1.svg
   figures/F4_2_2.svg
   figures/F4_2_3.svg
   ```

The MATLAB scripts are portable and write all figure files to:

```text
figures
```
