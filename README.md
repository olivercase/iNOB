# Forward Model — Vagus Nerve

A **dual-modality** torso-scale forward model for non-invasive recording of
the cervical vagus nerve. From a single subject anatomy and a single source
model we predict, in one pass:

* **MEG** — magnetic field at a triaxial OPM array (radial + 2 tangents per
  position, 8 190 channels in the default array).
* **EEG** — surface potential at a high-density (PEDOT:PSS-style) electrode
  patch over the cervical vagus (8 × 16 = 128 contacts at 5 mm pitch).

Both leadfields come from the *same* tetrahedral FEM, the *same*
conductivities, and the *same* dipole sources, so MEG-vs-EEG SNR /
localisability / array-design comparisons are head-to-head.

```
data/{bone,torso,vagus}/*.stl
        │
        │  build_geom         watertighten + shrinkwrap → outputs/geometry/vagus_geometry.mat
        ▼
  outputs/geometry/vagus_geometry.mat
        │
        │  build_fem          iso2mesh + CGAL multi-region → outputs/fem/fem_vagus.mat
        ▼                                ▲                ▲
  outputs/fem/fem_vagus.mat              │                │
        │              sensors / generate_sensors    electrodes / generate_electrodes
        │              (OPM triaxial array)          (HD-EMG patch on skin)
        │                                ▼                ▼
        ├────────► forward (MEG)  ─────────────────────────►  outputs/forward/duneuro_leadfield_vagus.npz
        ├────────► forward (EEG)  ─────────────────────────►  outputs/forward/duneuro_eeg_leadfield_vagus.npz
        ▼
  Analyses (analysis/{snr, sensitivity, analytic_sphere}, sources/cap)
  Topoplots (viz/topoplot)  →  outputs/dual_topoplot.png
```

**Headline figure.** `vagus-fm-topoplot --target dual` renders a four-panel
Nature Reviews-styled comparison (anatomy, MEG topoplot, EEG topoplot, and
amplitude distributions) for a single source along the cervical vagus, in
units of fT and µV per 1 nA·m source.

## Quickstart

```bash
git clone <this repo>
cd Forward_Model_Vagus_Nerve

python3 -m pip install -e .[dev]      # editable install, with dev deps
make test                              # 80+ tests, no DUNEuro required

# Build everything except the forward solve (which needs DUNEuro):
make pipeline                          # geom → fem → sensors → viz

# Run the local DUNEuro forward solve (needs duneuropy + a built FEM):
python3 -m vagus_fm.cli.run_forward \
    --set forward.duneuro_path=/path/to/duneuro-py/src
```

The `--config` flag (default `configs/default.yaml`) is the single source
of truth for paths, mesh sizes, sensor params, conductivities, and solver
settings. Override individual fields with `--set key.path=value`.

## Pipeline orchestrator

```bash
vagus-fm-pipeline                      # run all stages, skip those whose outputs exist
vagus-fm-pipeline --stages geom,fem    # just two
vagus-fm-pipeline --force              # ignore existing outputs
vagus-fm-pipeline --skip-viz           # everything but the PNGs
```

Each stage validates its output against a schema before declaring success;
a failed stage drops a `.<name>.FAILED` marker so reruns retry it.

## Per-stage CLIs

| Command | What it does |
|---|---|
| `vagus-fm-build-geom`        | STL → watertight geometry HDF5 |
| `vagus-fm-build-fem`         | Geometry → CGAL multi-tissue tet mesh |
| `vagus-fm-sensors`           | Skin → triaxial OPM array |
| `vagus-fm-electrodes`        | Skin + FEM → HD electrode patch over the vagus |
| `vagus-fm-forward`           | FEM + OPMs → MEG leadfield (DUNEuro) |
| `vagus-fm-eeg`               | FEM + electrodes → EEG leadfield (DUNEuro) |
| `vagus-fm-snr`               | Predicted per-source SNR for either modality |
| `vagus-fm-topoplot --target dual\|meg\|eeg\|montage` | Nature-styled topoplots |
| `vagus-fm-visualise --target geom\|fem\|all`        | Geometry + FEM PNGs |

All of them accept `--config / --set / --project-root / --log-level / --log-file`.

## Cluster execution (UCL Myriad / Kathleen)

See [cluster/README.md](cluster/README.md). Tldr:

```bash
# Locally:
make pipeline                                            # build inputs
CLUSTER_PROFILE=myriad bash cluster/stage.sh             # rsync to cluster
# On the cluster:
CLUSTER_PROFILE=myriad bash cluster/build_duneuro.sh     # one-time
CLUSTER_PROFILE=myriad bash cluster/submit.sh array      # forward solve (32 chunks)
CLUSTER_PROFILE=myriad bash cluster/submit.sh reduce     # waits on the array job
# Locally:
scp myriad:~/Scratch/vagus_fm/duneuro_leadfield_vagus.npz outputs/forward/
```

Switching to Kathleen is `CLUSTER_PROFILE=kathleen` — same scripts.

## Configuration

`configs/default.yaml` controls the whole pipeline. The fields that change
most often:

| Field | Meaning |
|---|---|
| `geometry.shrinkwrap.<tissue>.{pitch,n_samples,close_iter,decimate_target,smooth_iter}` | Per-tissue voxel-shrinkwrap params |
| `fem.{pitch_mm,radbound,maxvol}` | CGAL meshing params |
| `fem.tissues` | Which tissues to include (subset of vagus_left, vagus_right, bone, skin) |
| `sensors.{resolution_mm,depth_mm,z_crop_low_factor}` | OPM grid resolution + stand-off + chest crop |
| `forward.conductivities_sm` | Per-tissue σ (S/m); single source of truth, used locally + on cluster |
| `forward.source_spacing_mm` | Dipole spacing along the vagus nerve |
| `forward.duneuro_path` | Optional shim path for source-built duneuropy |
| `cluster.n_chunks` | How many array tasks the forward solve splits into |

## What this pipeline does that ASCENT / Sim4Life / SimNIBS don't

| Capability                                                   | this pipeline | ASCENT | Sim4Life | SimNIBS |
| :----------------------------------------------------------- | :-----------: | :----: | :------: | :-----: |
| Vagus-nerve-specific torso geometry (skin / bone / vagus)    | ✓             | ✓      | partial  | ✗       |
| MEG forward (OPM-class biomagnetic field)                    | ✓             | ✗      | ✓        | ✗       |
| EEG forward (cm-scale surface potentials, HD-EMG arrays)     | ✓             | ✗      | ✓        | ✓       |
| **Both modalities from one FEM + one source model**          | ✓             | ✗      | partial  | ✗       |
| Configurable HD electrode patch on a real torso              | ✓             | ✗      | manual   | ✗       |
| Configurable triaxial OPM array (cylindrical raycast)        | ✓             | ✗      | manual   | ✗       |
| Propagating CAP source model (fibre-CV dispersion)           | ✓             | ✓      | ✗        | ✗       |
| Conductivity sensitivity sweep (built-in)                    | ✓             | partial | ✗      | ✗       |
| OPM / HD-EMG noise model + per-source SNR predictor          | ✓             | ✗      | ✗        | ✗       |
| Sarvas / Berg-Scherg analytic validation tests               | ✓             | ✗      | partial  | ✓       |
| Open-source, scriptable Python pipeline                      | ✓             | ✓      | ✗ (commercial) | ✓ |

ASCENT is the gold-standard for *intra-fascicular* vagus modelling but it
cannot produce non-invasive forward solutions. Sim4Life can solve EM and EEG
forwards but is commercial and not built around a vagal source. SimNIBS is
the EEG/TMS standard for the head and doesn't ship a torso pipeline. None
solve both MEG and EEG from the same FEM, which is the primary
contribution here.

## Cross-modality coupling: predict MEG from EEG, given a known source

Both leadfields share the same FEM and the same source space. **For a
known source position** (anatomical landmark on the cervical-vagus
polyline), the EEG response is `V = L_E q` and the MEG response is
`B = L_M q` with the same dipole moment `q`. Given a measured EEG topo
and 32 ≫ 3 sensor channels, the moment `q` is over-determined and you can
plug it back into `L_M` to predict what the MEG topo *should* look like.

This is **not** the inverse problem — we do not localise `r₀` from the
EEG observation alone. The figure illustrates the *forward consistency*
of the dual-modality model conditioned on an MR-/anatomy-derived source
position, with realistic HD-EMG noise added to the EEG observation so the
inversion is meaningfully tested rather than being exact-by-construction.

Run `vagus-fm-cross` to render the four-panel diagnostic: per-source
amplitude scatter (MEG vs EEG, coloured by Z position), the observed EEG
topo, the predicted MEG topo derived from the (noisy) EEG observation, and
the actual FEM MEG topo with the prediction RMS error overlaid.

## Multi-tissue divergence from Sarvas: why FEM ≠ single-sphere

The `vagus-fm-sarvas` benchmark shows FEM peak fields larger than the
homogeneous-sphere Sarvas (Sarvas 1987) prediction by a factor of ~5–10×
in the cervical-axis literature band (40 mm source-axis, 58.5 mm
sensor-axis; Bu et al. 2024). This divergence reflects the
secondary-current contribution to the magnetic field, captured by
Geselowitz reciprocity but absent from any single-sphere model:

* In a homogeneous conductor, the magnetic field from a current dipole is
  independent of conductivity (Sarvas 1987; Geselowitz 1970).
* In a *multi-tissue* conductor, conductivity contrasts (skin vs bone vs
  vagus) reroute the volume currents. Those secondary currents themselves
  generate additional magnetic field via Biot–Savart.
* For cervical/spinal MEG specifically, O'Neill et al. 2025 (Sci Rep)
  show that **bone significantly attenuates lateral (left-right and
  anterior-posterior) currents** while longitudinal (superior-inferior)
  currents are nearly conductor-invariant. The implication: our FEM
  amplification is concentrated in particular geometries (close to the
  spine, with the right tangent direction) and the headline ratio
  depends sensitively on which (source, coil) pairs are sampled.

**Numerical headline** (current run, Q = 1 nA·m, longitudinal moment
along the vagus, literature-band coils only):

| quantity                      | value         |
|-------------------------------|---------------|
| Sarvas peak \|B\|             | 41 fT/nAm     |
| Sarvas peak at Q = 70 nA·m    | 2.9 pT  ✓ matches Bu 2024 1–4 pT range |
| FEM peak \|B\|                | 283 fT/nAm    |
| FEM peak at Q = 70 nA·m       | 19.8 pT       |
| FEM/Sarvas ratio (peak)       | ~6.8×         |

The 19.8 pT FEM peak is a single best-aligned coil — averaged over the
literature-band geometry (40/58.5 mm) the median is much smaller, and
**the Sarvas band peak (2.9 pT) sits squarely inside Bu et al.'s
"1–4 pT consistently across subjects" measurement range.**

**Practical consequence for the dual-modality paper.** The amplification
is real but tissue-conductivity-dependent — exactly the conductivity
sensitivity that `vagus-fm`'s `analysis.sensitivity` sweep is designed to
quantify. Reporting Sarvas (analytic baseline) and FEM (full
secondary-current solution) side-by-side with a conductivity-uncertainty
band is the rigorous thing to do.

### Citation list (verified May 2026)

| Citation | DOI |
|---|---|
| Sarvas J. 1987. *Phys Med Biol* 32:11–22. "Basic mathematical and electromagnetic concepts of the biomagnetic inverse problem." | [10.1088/0031-9155/32/1/004](https://doi.org/10.1088/0031-9155/32/1/004) |
| Hämäläinen M, Hari R, Ilmoniemi RJ, Knuutila J, Lounasmaa OV. 1993. *Rev Mod Phys* 65:413–497. "Magnetoencephalography — theory, instrumentation, and applications to noninvasive studies of the working human brain." | [10.1103/RevModPhys.65.413](https://doi.org/10.1103/RevModPhys.65.413) |
| Boto E, Holmes N, Leggett J, Roberts G, Shah V, Meyer SS, Muñoz LD, Mullinger KJ, Tierney TM, Bestmann S, Barnes GR, Bowtell R, Brookes MJ. 2018. *Nature* 555:657–661. "Moving magnetoencephalography towards real-world applications with a wearable system." | [10.1038/nature26147](https://doi.org/10.1038/nature26147) |
| Tierney TM, Holmes N, Mellor S, López JD, Roberts G, Hill RM, Boto E, Leggett J, Shah V, Brookes MJ, Bowtell R, Barnes GR. 2019. *NeuroImage* 199:598–608. "Optically pumped magnetometers: From quantum origins to multi-channel magnetoencephalography." | [10.1016/j.neuroimage.2019.05.063](https://doi.org/10.1016/j.neuroimage.2019.05.063) |
| Tierney TM, Mellor S, O'Neill GC, Holmes N, Boto E, Roberts G, Hill RM, Leggett J, Bowtell R, Brookes MJ, Barnes GR. 2020. *Sci Rep* 10:21609. "Pragmatic spatial sampling for wearable MEG arrays." | [10.1038/s41598-020-77589-8](https://doi.org/10.1038/s41598-020-77589-8) |
| O'Neill GC, Spedden ME, Schmidt M, Mellor S, Stenroos M, Barnes GR. 2025. *Sci Rep* 15:26258. "Volume conductor models for magnetospinography." | [10.1038/s41598-025-10770-z](https://doi.org/10.1038/s41598-025-10770-z) |
| Bu Y et al. 2024. *Comm Biol* 7:893. "Non-invasive ventral cervical magnetoneurography as a proxy of in vivo lipopolysaccharide-induced inflammation." (Source of the 1–4 pT cervical-vagus measurement range and the Sarvas concentric-circles geometry.) | [10.1038/s42003-024-06435-8](https://doi.org/10.1038/s42003-024-06435-8) |
| Zuo Y et al. 2022. *Front Physiol* 13:798376. "Peripheral nerve magnetoneurography with optically pumped magnetometers." | [10.3389/fphys.2022.798376](https://doi.org/10.3389/fphys.2022.798376) |
| Geselowitz DB. 1970. *IEEE Trans Magn* 6(2):346–347. "On the magnetic field generated outside an inhomogeneous volume conductor by internal current sources." | [10.1109/TMAG.1970.1066765](https://doi.org/10.1109/TMAG.1970.1066765) |

(Note: the earlier "Tierney et al. 2020 NeuroImage on dense-array OPM
forward solutions" framing was wrong on two counts — the paper is
*Scientific Reports*, not *NeuroImage*, and its subject is array sampling
density / spatial discrimination, not FEM forward modelling per se. The
forward-modelling-specific reference for body MEG is the 2025 O'Neill et
al. magnetospinography paper.)

## Source-strength convention — one unified rule

**Every leadfield, every figure, every CLI output is reported per 1 nA·m
source moment, in fT (MEG) or µV (EEG).** This is the leadfield calibration
that every other plot in this repo uses.

* MEG axes / colour bars: `fT  (1 nA·m source)`.
* EEG axes / colour bars: `µV  (1 nA·m source)`.
* Sarvas vs FEM benchmark (`vagus-fm-sarvas`): defaults to `--Q-nAm 1`,
  same convention.

The value 70 nA·m only appears as a *physiological scaling* (Hämäläinen
1993 summation over the A + C fibre population at full activation) when
you want to predict the actual measured pT-scale real-CAP signal. To do
so, run `vagus-fm-sarvas --Q-nAm 70` — the y-axis numbers can then be read
as pT directly (1 fT × 70 = 70 fT = 0.07 pT … per the linear scaling).
The output JSON reports both conventions side-by-side
(`*_fT_per_nAm` and `*_pT_at_Q70`).

**No conflict with the previous "Q = 70" framing.** The earlier Sarvas
default was Q = 70 nA·m to match the literature 1–4 pT range directly. We
have switched to the unified Q = 1 nA·m default so every figure speaks the
same units. The literature comparison is now an opt-in flag.

## Dual-modality story (paper outline)

The pipeline supports — and is intended to enable — a head-to-head
comparison of OPM magnetoneurography vs HD-EMG-style surface electrodes
for non-invasive cervical-vagus recording, on the same anatomy and source
model:

1. **Source.** `vagus_fm.sources.cap.cap_signal` synthesises a propagating
   compound action potential along the vagus polyline with a
   fibre-diameter-distributed conduction velocity (Hursh / Pelot 2017).
2. **Forward solutions.** `forward.solve` (MEG) and `forward.eeg` (EEG)
   compute their leadfields from the same `fem_vagus.mat`.
3. **SNR.** `analysis.snr` predicts per-source SNR given the
   `cfg.noise.opm_intrinsic_fT_sqrtHz` and HD-EMG amplifier + Johnson noise
   floors (configurable).
4. **Sensitivity.** `analysis.sensitivity` perturbs `cfg.forward.conductivities_sm`
   (typically bone, skin) ± a few × 10 % and reports per-channel relative
   change. Expected finding: MEG ≪ EEG sensitivity to bone/skin σ.
5. **Validation.** `analysis.analytic_sphere` provides Sarvas (MEG) and
   Berg-Scherg (EEG) closed-form references in homogeneous spheres for
   cross-checking the FEM driver in tests/.
6. **Visualisation.** `viz.topoplot.render_dual_topoplot` produces the
   Nature Reviews-styled headline figure.

## Layout

```
src/vagus_fm/        — package: config, io, mesh, geometry, fem, sensors, sources, forward, viz, cli
cluster/             — profile-driven Myriad / Kathleen submission scripts
configs/             — YAML configs (default.yaml + tiny_test.yaml for tests)
data/{bone,torso,vagus}/  — raw STL inputs (74 bones + 1 skin + 2 vagus trunks)
outputs/             — generated artefacts (gitignored): geometry/, fem/, sensors/, forward/, logs/
tests/               — pytest suite (~86 tests; DUNEuro smoke auto-skips)
scripts/             — small utilities (e.g. inob_obj2stl.py)
```

## Development

```bash
make install-dev    # editable install + pytest + ruff
make lint           # ruff
make test           # pytest
make clean          # remove build/, __pycache__, etc.
make clean-outputs  # nuke outputs/
```

CI (.github/workflows/ci.yml) runs ruff + pytest on Python 3.11 / 3.12.
DUNEuro tests auto-skip when `duneuropy` is unavailable.

## Troubleshooting

* **`duneuropy could not be imported`** — install via `cluster/build_duneuro.sh`,
  or set `forward.duneuro_path` in your config to the duneuro-py source dir.
* **Geometry validation fails (`not watertight`)** — run
  `vagus-fm-build-geom --shrinkwrap-only` to force the voxel-shrinkwrap
  pipeline for every tissue.
* **`STLLoadError: implausible for mm`** — input was probably authored in
  metres or centimetres; convert before feeding into the pipeline (or pass
  `check_units_mm=False` if you're sure).
* **`MeshQualityError: min mesh quality`** — increase `fem.pitch_mm` or
  `fem.maxvol`, or relax `fem.validate.min_mesh_quality` in the config.

## Licensing

* **Code** (Python, build scripts, configs) — MIT, see [`LICENSE`](LICENSE).
* **Anatomical mesh data** — every mesh under `data/` and `internal_meshes/`,
  and all FEM / leadfield / figure artifacts derived from them, comes from the
  **BodyParts3D** atlas (© The Database Center for Life Science, DBCLS) and is
  licensed under **CC BY-SA 2.1 Japan**, not MIT. See
  [`LICENSE-DATA`](LICENSE-DATA). If you redistribute the meshes or our
  derivatives, you must attribute DBCLS and share alike. Cite Mitsuhashi *et
  al.*, *Nucleic Acids Research* 2009 (doi:10.1093/nar/gkn613).
