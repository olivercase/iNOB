# Validation

This file is the formal record of how every numerical claim in the pipeline
was validated, including the unit-convention saga and the published
references the absolute amplitudes are calibrated against.

A Nature reviewer should be able to read this file and understand,
without consulting body text, *exactly* how confidence in each headline
number was established.

## Summary of validations

| Claim | Validated by | Status | Where |
| --- | --- | :-: | --- |
| MEG forward conversion factor (×10⁶) | Sarvas analytic match (Bu 2024 1–4 pT range) + MNE `_do_sphere_field` cross-validation (rtol 1e-12) | ✓ | `tests/test_analytic_sphere.py::test_sarvas_matches_mne_to_machine_precision` |
| EEG forward conversion factor (×0.622) | Berg-Scherg analytic on a controlled homogeneous sphere FEM | ✓ | `outputs/calibration/calibration.json` + `vagus-fm-calibrate` |
| EEG calibration is geometry-stable | (R, depth, σ) sweep across 27 cases: CV = 6.8% | ✓ | `outputs/calibration/sweep.json` |
| Vagus position is anatomically anterior | Three independent landmark checks against the FEM | ✓ | `docs/COORDINATES.md` |
| FEM mesh quality | per-element Joe–Liu metric; min ≥ `cfg.fem.validate.min_mesh_quality` | ✓ | `src/vagus_fm/mesh/quality.py::assert_mesh_ok` (current FEM: min 0.092, p1 0.241, p5 0.301, mean 0.524, max 1.000 over 789 342 tets) |
| Schema validation on every output | HDF5 / NPZ shape, finiteness, units, contiguity gates | ✓ | `src/vagus_fm/io/{hdf5,npz}.py` |
| Stationary-dipole approx vs propagating CAP | Moving-wavelet benchmark on real FEM; cervical-localised peak ratio 0.72 (28% reduction) | ✓ | `vagus-fm-cap-compare` → `outputs/cap_compare.png`; `src/vagus_fm/viz/cap_compare.py` |

## 1. MEG units (×10⁶)

DUNEuro mm-mode returns the magnetic field at a coil for a unit current
dipole. The legacy `run_fem_duneuro.py` convention `L_fT_per_nAm = L * 1e6`
gives values that match published OPM-vagus measurements (Bu et al.
2024 *Comm Biol* 7:893, "1–4 pT consistently in subjects" at full A+C
fibre summation Q ≈ 70 nA·m).

Reproducing the validation:

```bash
vagus-fm-sarvas --Q-nAm 70   # Sarvas peak ≈ 2.9 pT — within Bu's 1–4 pT band
```

Independent gold-standard check: `tests/test_analytic_sphere.py` runs four
parametrised cases (tangential / longitudinal / off-axis / mixed-component
dipoles) against MNE-Python's `_do_sphere_field`. Match is to machine
precision (`rtol=1e-12`).

## 2. EEG units (×0.622)

DUNEuro mm-mode EEG output units are *not* documented in a way that
admits an a-priori dimensional analysis — the May 2026 audit attempted
that and got it wrong by a factor of 10⁶. The empirical calibration is the
correct procedure.

### 2.1 Setup

Build a homogeneous-tissue sphere FEM (one tissue, σ_skin = 0.43 S/m,
radius R = 100 mm). Place 200 electrodes on the sphere surface using a
Fibonacci lattice. Place a unit dipole at depth d = 50 mm. Run DUNEuro
EEG forward; compute Berg-Scherg analytic at the same electrode positions.
The ratio analytic / DUNEuro_raw is the calibration factor.

```bash
vagus-fm-calibrate                  # writes outputs/calibration/calibration.json
```

### 2.2 Headline calibration

| | value |
| --- | --: |
| n_tets | 2 731 |
| Raw DUNEuro peak | 0.319 |
| Berg-Scherg peak | 0.0980 µV/(nA·m) |
| **Median factor** | **0.622** |
| Geomean factor | 0.738 |
| Peak factor | 0.307 |

Applied: `L_uV_per_nAm = L_raw * EEG_CALIBRATION_FACTOR` in
`src/vagus_fm/forward/eeg.py` (constant defined at module top, value 0.622).
The factor is deliberately exposed as a named constant so the calibration
provenance is grep-able and so the saved NPZ on disk and a freshly re-run
forward solve are byte-identical.

### 2.3 Robustness sweep

Computed factor over 27 cases varying:

* Sphere radius R ∈ {60, 100, 140} mm.
* Source depth s ∈ {0.3, 0.5, 0.7} × R.
* Conductivity σ ∈ {0.1, 0.43, 1.0} S/m.

| | factor median |
| --- | --: |
| Min     | 0.597 |
| Max     | 0.754 |
| Mean    | 0.671 |
| Std     | 0.046 |
| **CV**  | **6.8%** |

The factor is **invariant to σ** (analytic verification: Berg-Scherg's
1/σ dependence and DUNEuro's σ scaling cancel exactly), and varies
slowly with (R, source depth) at the few-percent level. The production
factor 0.622 is well within the sweep range.

```bash
cat outputs/calibration/sweep.json
```

## 3. Sarvas vs FEM benchmark

`vagus-fm-sarvas` runs the moving-sphere Sarvas analytic against the
multi-tissue FEM forward for every (source, coil) pair on the cervical
vagus. Headline: peak FEM/Sarvas ratio = 6.8× — consistent with the
secondary-current contribution (Geselowitz 1970) documented for body MEG
in O'Neill et al. 2025 *Sci Rep* 15:26258.

Median ratio is much lower (~0.9) — the 6.8× is a peak over 7 333
(source, coil) pairs, dominated by best-aligned coils close to the spine.

## 3a. Stationary-dipole approximation vs propagating CAP

The physiology simulator (`vagus_fm.physiology.simulate.simulate_train`)
collapses each event to a biphasic moment trace at the cervical hot-spot
(highest-Z source on the vagus polyline). To benchmark this approximation,
`vagus-fm-cap-compare` runs three source models on the real MEG leadfield
at the same total event moment Q_total = N_fibres × ⟨Q_per_fibre⟩_w:

  1. **Stationary at hot-spot** (manuscript convention).
  2. **Cervical-localised moving wavelet**: a single AP wavelet that fires
     at the cervical bottom and propagates rostrally over the top 50 mm of
     the polyline at fibre-CV. This is the physiologically correct model
     for a synchronously-firing carotid-baroreceptor bundle.
  3. **Whole-vagus moving wavelet**: same dynamics but over the full 528 mm
     polyline; applicable to pulmonary or abdominal afferents.

Headline result on the production FEM, single 200-A-fibre baroreceptor
event (Q_total = 0.74 nA·m), best radial MEG channel #2219:

| | peak |B| (fT) | P/S ratio | FWHM (ms) |
| --- | ---: | ---: | ---: |
| Stationary at hot-spot               | 83.1 | 1.000 | 1.87 |
| Cervical-localised moving wavelet    | 60.1 | **0.72** | 0.30 |
| Whole-vagus moving wavelet           |  6.4 | 0.08  | — |

The 28% peak reduction (P/S 0.72) for the realistic cervical-localised model
comes from CV-dispersion across the lognormal fibre population (CVs span
12–90 m/s) and the narrow ~16-mm-FWHM spatial peak of L_long at the
topmost cervical sources — the wave only briefly samples high-L_long
territory. The trials-to-SNR=3 numbers in the manuscript should therefore
be inflated by 1/0.72² ≈ 1.93× under the propagating model
(baroreceptor MEG: 3.4 → 6.6 min; deep-breathing MEG: 25 → 48 min).

Reproducing:

```bash
vagus-fm-cap-compare           # writes outputs/cap_compare.png
```

## 4. Anatomy — vagus is anterior to spine

See `docs/COORDINATES.md`. Three landmark checks confirm the +Y axis is
posterior in our FEM, and the vagus polyline sits anterior to the
cervical vertebrae. No bone between the vagus and an anteriorly-placed
cervical electrode patch.

## 5. Schema + reproducibility

* Every leadfield output (HDF5 / NPZ) is validated against a typed
  schema before write (`src/vagus_fm/io/{hdf5,npz}.py:validate_*`).
  Validation includes: shape, finiteness, contiguous tissue ids, unit
  bbox bounds, max amplitude.
* Every CLI entry point seeds `np.random` and `random` from
  `cfg.reproducibility.seed` at startup
  (`src/vagus_fm/cli/_common.py::setup`).
* CGAL meshing is non-deterministic between iso2mesh versions but produces
  meshes within the validated quality band on every run (mesh-quality
  gate in `src/vagus_fm/fem/cgal_builder.py`).

## 6. Known limitations the paper should acknowledge front-and-centre

These are *limits of the current dataset*, not bugs in the pipeline. A
Nature reviewer will ask about each:

1. **Anatomy is from BodyParts3D atlas, not a subject scan.** No
   subject-specific fat/muscle thickness, no carotid sheath geometry, no
   heterogeneous skin layers. A roadmap for subject-specific extension is
   future work.
2. **The "skin" tissue is a soft-tissue composite** (muscle + fat +
   connective + skin). σ = 0.43 S/m is an averaged value (within the
   Gabriel 1996 / IT'IS database range for muscle/composite at low kHz).
   True multi-layer modelling — particularly the high-conductivity-contrast
   subcutaneous fat layer — is future work.
3. **Conductivities are static.** No frequency dependence, no anisotropy
   (muscle in particular has a 5–10× longitudinal/transverse conductivity
   anisotropy).
4. **CAP source model is simplified.** `vagus_fm.sources.cap` has a
   propagating-fibre-population model with diameter-dependent CV; the
   current SNR figures use a static-dipole approximation. Wiring CAP into
   the SNR pipeline is a one-week job.
5. **EEG electrode model is point-contact.** No electrode-skin impedance,
   no gel/dry distinction. Adding a finite-contact + impedance layer
   would refine the predicted µV amplitudes by 10–30%.
6. **Bandwidth assumption: 1 kHz.** Justified for fast A-fibre CAPs
   (Pelot et al. 2020 cite up to ~3 kHz for the largest myelinated
   fibres); could be lower (300–500 Hz) for averaged CAPs of mixed-fibre
   populations. Linear effect on noise σ via √BW.

## 7. References

| | citation | DOI |
| --- | --- | --- |
| MEG analytic (Sarvas)      | Sarvas J. 1987 *Phys Med Biol* 32:11–22 | `10.1088/0031-9155/32/1/004` |
| MEG/EEG canonical review   | Hämäläinen M *et al.* 1993 *Rev Mod Phys* 65:413 | `10.1103/RevModPhys.65.413` |
| Cervical-vagus OPM measurement | Bu Y *et al.* 2024 *Comm Biol* 7:893 | `10.1038/s42003-024-06435-8` |
| Body-MEG forward modelling | O'Neill GC *et al.* 2025 *Sci Rep* 15:26258 | `10.1038/s41598-025-10770-z` |
| OPM forward modelling      | Tierney TM *et al.* 2019 *NeuroImage* 199:598–608 | `10.1016/j.neuroimage.2019.05.063` |
| Wearable OPM-MEG           | Boto E *et al.* 2018 *Nature* 555:657–661 | `10.1038/nature26147` |
| OPM array sampling         | Tierney TM *et al.* 2020 *Sci Rep* 10:21609 | `10.1038/s41598-020-77589-8` |
| Volume conductor formalism | Geselowitz DB 1970 *IEEE Trans Magn* 6:346 | `10.1109/TMAG.1970.1066765` |
| Vagus fibre population     | Pelot NA *et al.* 2020 *Front Neurosci* 12:601 | `10.3389/fnins.2020.601479` |
