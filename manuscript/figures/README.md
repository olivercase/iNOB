# Figures of the paper

The generators of every figure in the iNOB paper, drawn to Nature's figure
standard (Helvetica/Arial 7 pt, panel letters 8 pt bold, 89/120/183 mm wide,
TrueType fonts only). `src/style/` is a copy of the house style the paper's
own repository uses, so these scripts run from a plain checkout.

| Script | Figures | Needs |
|---|---|---|
| `src/make_pipeline.py` | 1 (pipeline) | `rsvg-convert` |
| `src/make_anatomy.py` | 2 (anatomy) | meshes (Git LFS), pyvista |
| `src/make_torso.py` | 3 (field on the skin) | saved MXG and whole-body EXG leadfields |
| `src/make_data_figures.py [name ...]` | 4-10 (cord example) | saved leadfields |

Run from this folder with the package's Python:

```sh
python src/make_data_figures.py            # all data figures
python src/make_data_figures.py trials     # one of them
```

Both noise floors come from `configs/spine.yaml` (SQUID magnetoneurograph,
280 fT over 100-5,000 Hz; surface electrodes, 2.19 uV over 30-500 Hz). The
stand-off panel reads `outputs/paper/standoff.json` from
`scripts/sensor_sweep.py --json` when it exists and otherwise uses the values
of the paper's sweep, and says which. Figure 5b of the validation plots
published values (Miyano 2020; Hashimoto 2021), entered in the script.

Output (PDF, PNG, SVG) is written next to `src/`; only the PDFs are kept in
the repository.
