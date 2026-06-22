# Figure 1

This folder contains the MATLAB scripts and prepared CSV inputs used to render
the Figure 1 material OCP and silicon-current-share panels.

- `F1_1.m` plots the graphite and effective silicon OCP curves together with
  the differential stoichiometry response.
- `F1_ThermaldQdV_Flowchart_v2.m` plots the BOL material-resolved
  incremental-capacity traces and silicon current share.

The scripts read figure-ready CSV files from:

```text
data
```

and write SVG outputs to:

```text
figures
```
