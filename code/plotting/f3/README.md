# Figure 3

This folder contains the MATLAB scripts and prepared CSV inputs used to render
the Figure 3 silicon-OCP deformation and room-temperature RPT progression
panels.

- `F3_1.m` plots the effective silicon-OCP deformation cases and the
  corresponding voltage and derivative reconstructions.
- `F3_2.m` plots the room-temperature RPT voltage and dV/dQ progression for
  cell064.

The scripts read figure-ready CSV files from:

```text
data
```

and write SVG outputs to:

```text
figures
```
