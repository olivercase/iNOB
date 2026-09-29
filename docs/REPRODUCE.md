# Reproducing the paper's figures

Everything in the paper comes from the meshes in `data/`, the configuration
in `configs/default.yaml`, and the commands below. Runs are seeded
(`reproducibility.seed`), so repeated runs give identical arrays; different
DUNEuro builds or BLAS libraries can differ in the last digits.

## What you need

- iNOB installed (`pip install -e .`) with a DUNEuro build (`inob doctor`
  must show `DUNEuro (duneuropy) importable`). `docs/QUICKSTART.md` §3.
- The meshes (`git lfs pull`).
- Time: the three forward solves are the cost, and it scales with the
  number of dipoles (`forward.source_spacing_mm`) and sensors. The solve
  splits the array across every core (`forward.local_workers: 0`); the
  muscle target at 15 mm spacing took 7 minutes on the development laptop,
  the 5 mm vagus and spine targets longer. `cluster/` has the SGE array-job
  version for the full-resolution runs.

## One command

```bash
bash scripts/run_all_targets.sh          # vagus, spine, muscle in turn
```

This is `scripts/run_target.sh` for each of the three worked examples:
geometry and FEM once, the OPM array once, then per target the MEG and EEG
solves and every figure the target supports. Leadfields land in
`outputs/forward/duneuro_leadfield_<target>.npz` (MEG, fT per nA·m) and
`duneuro_eeg_leadfield_<target>.npz` (EEG, µV per nA·m); figures under
`outputs/`.

Already have the leadfields (from the archived results deposit, or a previous
run)? Regenerate the figures alone:

```bash
bash scripts/regenerate_figures.sh
```

## Figure by figure

| Paper section | Command(s) | Output |
| --- | --- | --- |
| Anatomical inputs, FEM | `inob visualise` | `outputs/geometry/geometry.png`, `outputs/fem/fem.png` |
| Sensor arrays | `inob sensors`, `inob electrodes` | `outputs/sensors/*.png` |
| Calibration to absolute units | `inob calibrate` (`--meg` for the MEG sphere) | `outputs/calibration/calibration.json`, `meg_sphere_validation.json` |
| Sarvas vs full-FEM benchmark | `inob sarvas --out FILE.png`, `inob ladder --json` | benchmark figure, ladder table on stdout |
| Cervical vagus | `inob topoplot --source-target vagus`, `inob detect --source-target vagus`, `inob physiology --source-target vagus`, `inob sensor-field --source-target vagus` | `outputs/*_vagus.png` |
| Spinal cord | same with `--source-target spine`, plus `inob cross`, `inob cap-compare`, `inob location` | `outputs/*_spine.png` |
| Neck muscles | `inob topoplot --source-target muscle --target meg`, `inob cap-compare --source-target muscle`, `python scripts/muscle_field_sarvas.py` | `outputs/*_muscle.png`, `outputs/muscle_skin_topoplot.png` |
| Heart (peak-QRS dipole) | `inob forward --source-target heart --set forward.source_spacing_mm=10`, then `inob detect --source-target heart` | `outputs/*_heart.png` |
| Brain (cortical evoked dipole) | same with `--source-target brain` | `outputs/*_brain.png` |
| Whole-body electrode array (coverage comparisons, torso panel b) | `inob electrodes --source-target T --set electrodes.shape=whole_body`, then `inob eeg` with the same flags (muscle: add `--set forward.source_spacing_mm=15`) | `outputs/forward/duneuro_eeg_leadfield_wholebody_T.npz` |
| Field on the body (Figs 3-5) | `inob torso --source-target T --esg-npz outputs/forward/duneuro_eeg_leadfield_wholebody_T.npz` | `outputs/torso_topoplot_T.png` |
| Stand-off and noise level | `python scripts/sensor_sweep.py` (one DUNEuro solve per stand-off) | table on stdout |
| Array design | `inob placement --source-target T`, `inob placement --source-target T --optimise` | table on stdout, `outputs/placement_comparison_T.png` |
| EEG unit calibration across spheres | `python scripts/eeg_calibration_sweep.py` | table on stdout, `outputs/calibration/sweep/` |
| Every number in the worked examples | `python scripts/paper_numbers.py` (reads the leadfields above; no solve) | stdout |
| Conductivity sensitivity | `inob sensitivity` (re-solves DUNEuro per perturbation) | `outputs/forward/sensitivity/` |

## Archiving what you produced

`scripts/package_results.sh` collects the leadfields, meshes, sensor arrays
and figures into one tarball with a SHA-256 manifest, ready for a Zenodo or
OSF deposit. The solved leadfields are 20–170 MB each, above the journal's
100 MB supplementary-material limit, so they go in a repository with a DOI
rather than alongside the manuscript.
