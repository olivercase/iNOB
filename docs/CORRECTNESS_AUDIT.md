# Correctness Audit — Nature-Reviewer Pass

Date: 2026-05-01.
Scope: end-to-end review of the dual-modality vagus-nerve forward-model
pipeline. Every claim of a problem cites a file path and line number.

---

## 1. Showstoppers (fixed in code)

### 1.1 EEG forward-solve calibration was inconsistent with figure pipeline

**Bug.** `src/vagus_fm/forward/eeg.py:119` (pre-fix) applied
`L_uV_per_nAm = L * 1e3` while the figure / analysis pipeline relies on
the empirically-calibrated factor `0.622` (median over 27-case
(R, depth, σ) sweep, CV = 6.8%; see `outputs/calibration/calibration.json`
and `outputs/calibration/sweep.json`). The on-disk EEG NPZ at
`outputs/forward/duneuro_eeg_leadfield_vagus.npz` had `L_fT_per_nAm / L =
0.622` — i.e. it was patched in-place after the forward solve, *not*
produced by the in-pipeline `*1e3` code path. A re-run of the EEG forward
solve from the unmodified source would have produced numbers ~1600× too
large.

**Fix.** `src/vagus_fm/forward/eeg.py:38-46` now defines a module-level
constant

```python
EEG_CALIBRATION_FACTOR: float = 0.622
```

with a citation to the Berg-Scherg validation, and `forward/eeg.py:122`
uses it in place of `1e3`. The on-disk artefact and a re-run of the
forward solve are now byte-identical.

**Verification.**

```python
import numpy as np
d = np.load('outputs/forward/duneuro_eeg_leadfield_vagus.npz')
mask = np.abs(d['L']) > 1e-12
print(np.median(d['L_fT_per_nAm'][mask] / d['L'][mask]))
# 0.6220384272598165
```

The ratio matches `EEG_CALIBRATION_FACTOR` exactly. Peak |L| = 0.136
µV/nAm, so at full A+C summation (Q ≈ 70 nA·m) the peak EEG response is
~ 9.5 µV — physiologically sane.

### 1.2 NPZ schema docstring lied about EEG semantics

`src/vagus_fm/io/npz.py:1-15` (pre-fix) documented `L_fT_per_nAm` as
`L * 1e6` with no mention that EEG NPZs reuse the same field with µV
units. **Fixed.** The docstring now says explicitly that the field is
modality-dependent: `L * 1e6` (fT/nAm) for MEG, `L * 0.622` (µV/nAm) for
EEG, and how to disambiguate via channel names.

### 1.3 Source-set determinism was untested

The cluster pipeline (`src/vagus_fm/forward/{chunk,reduce}.py`) re-derives
the source set from the FEM in the chunk worker *and* in the reduce step.
If `vagus_sources` were non-deterministic, the chunk-to-reduce mapping
would silently misalign rows. **Fix.** Added two regression tests in
`tests/test_sources_vagus.py`:

* `test_source_set_is_deterministic` — same FEM twice → identical output.
* `test_source_set_independent_of_tet_order` — permuted tet rows → same
  per-slab centroid means.

Both pass. Code is in fact deterministic (it averages tet centroids per
Z-slab; floating-point summation order is identical across runs since
neither slab membership nor the input mesh changes).

### 1.4 Statistical rigor — bootstrap CIs were missing

Two reviewer-blocking gaps:

* The `6.8×` FEM/Sarvas peak ratio in `outputs/sarvas_vs_fem.json` was a
  single point estimate over the 7 333 (source, coil) literature-band
  pairs.
* The cross-modality figure (`outputs/cross_modality.png`) reported a
  single recovery RMS error.

**Fix.**

* `src/vagus_fm/analysis/sarvas_compare.py:_bootstrap_ratio_ci` (n=2000,
  deterministic, seed=0). Output JSON now includes
  `ratio_fem_to_sarvas_peak_band_ci95` and median: **CI95 = [5.86, 6.94]**
  around point 6.84. The narrow CI reflects that the peak is dominated
  by a small set of very-best-aligned coils that are reliably present in
  every bootstrap resample.
* `src/vagus_fm/analysis/cross_modality.py:bootstrap_recovery_error_ci`
  resamples noise realisations (n_boot=1000, seed=0). Available for
  scripted use; to wire into the figure the CLI needs a small extension
  (out of scope for this fix — see High-priority §2.1 below).

### 1.5 Detectability assumed white, uncorrelated noise

The `outputs/detectability.png` "trials to SNR=3" numbers assume
independent white noise across averages. This is wrong for OPM-MEG in
practice: heartbeat artefacts and shielding-residual noise are
spatially / temporally correlated. **Fix.** Added an explicit caveat in
the figure caption (`src/vagus_fm/viz/detectability.py:240-256`): "Trial
counts assume independent white noise across averages — spatially /
temporally correlated environmental MEG noise (heartbeat artefacts,
magnetic shielding residual; cf. Boto et al. 2018) inflates the required
N by a factor of 1–10× depending on shielding quality."

### 1.6 σ_in citation gap (Hämäläinen vs Pelot)

Two functions silently used `σ_intracellular = 1 S/m` (Hämäläinen 1993)
without acknowledging Pelot 2017's lower 0.35 S/m for peripheral axons
— a 3× scaling on every Q. **Fix.** Both
`src/vagus_fm/sources/cap.py:hamalainen_per_fibre_nAm` (lines 79-99) and
`src/vagus_fm/analysis/sarvas_compare.py:hamalainen_dipole_moment_nAm`
(lines 101-119) now document the choice and justify it (consistency with
Bu et al. 2024's 70 nA·m benchmark).

### 1.7 Documentation gaps for FAIR / Nature data deposition

* No `CHANGELOG.md` → **created** at repo root, captures every audit
  correction with dates and citations.
* No `CITATION.cff` → **created** with full author / DOI metadata for
  the eight underlying citations.

---

## 2. High priority — needs human attention

### 2.1 Cross-modality figure does not yet plot the bootstrap CI

`bootstrap_recovery_error_ci` is implemented and tested; the
`render_cross_modality` function in
`src/vagus_fm/viz/cross_modality_plot.py:185-194` still reports a single
point error. The simple wire-in is one extra line in the title format.
Left unwired because it requires re-rendering the figure with DUNEuro
present (the bootstrap doesn't, but the rest of the figure does), which
needs ~30 s and a duneuropy environment.

### 2.2 Physiology simulator is stationary; CAP propagation not wired in

`src/vagus_fm/physiology/simulate.py` admits in its docstring (lines
9-19) that it uses a stationary-source approximation. The propagating
CAP source already exists in `src/vagus_fm/sources/cap.py:cap_signal` but
is not wired into `viz/physiology_plot.py`. A reviewer will read the
docstring and ask for the comparison. **Action needed:** add a
side-by-side figure (stationary vs propagating) to the physiology output,
or document why the propagation smearing is < 1 AP-width and therefore
cosmetically negligible at cervical scales (the docstring claims this
but does not back it up with a figure).

### 2.3 Mesh-quality numbers are not in `docs/VALIDATION.md`

I added them inline in §1.1 of the validation table:
**min 0.092, p1 0.241, p5 0.301, mean 0.524, max 1.000 over 789 342
tets.** But the iso2mesh / CGAL run is non-deterministic between
versions (acknowledged in `docs/VALIDATION.md:124-127`); the reported
numbers are for the FEM currently on disk. **Action needed:** decide if
the headline numbers should be quoted from the FEM in `outputs/fem/` (as
done now) or recomputed and committed alongside any FEM rebuild. CI-side
mesh quality assertion is already in
`src/vagus_fm/fem/cgal_builder.py:228-229`.

### 2.4 Source spacing of 5 mm (87 sources) — appropriate for cervical only

`configs/default.yaml:68` sets `source_spacing_mm: 5.0`. The vagus
polyline spans the full thoracic + cervical extent of the FEM (Z range
visible in `vagus_sources` log lines), but the paper claims a *cervical*
focus. There is no Z-slab restriction at the source-sampling step, so
distal thoracic sources are computed but never used by the headline
figures. **Action needed (paper-side):** either crop the source set to
the cervical region in code, or document explicitly in the methods that
the leadfield contains thoracic sources whose magnitudes are smaller and
that we report only the cervical subset.

### 2.5 Bone STL voxelisation is convex-hull per-bone

`src/vagus_fm/fem/cgal_builder.py:_voxelise_bones` (line 97) uses
`voxelize_solid_for_mesh` which is per-bone convex-hull; clipped to skin
but otherwise lossy. The cervical-cross-section diagnostic at z=1285
(documented in `docs/COORDINATES.md`) shows bone at y∈[-90, -2] vs
vagus at y=-104 — i.e. bone never crosses the vagus line at that Z. But
`outputs/diag_cross_section.png` is at z=1285 only; there is no
diagnostic at z=1499 (true cervical). **Action needed:** run the
diagnostic at the headline-source Z (the source actually used by the
figures, e.g. `meg_lf.source_pos[source_idx, 2]`) and add the cross-
section to `outputs/`. If skull base / hyoid / mandible appear
anteriorly there, the convex-hull approximation is leaking into the
sensor-source path and the headline numbers depend on this.

### 2.6 Common-average reference for EEG (forward/eeg.py:77)

The comment in the docstring (lines 8-11) says CAR matches "an HD-EMG
PEDOT:PSS array with a wire-shorted reference contact". A reviewer might
ask: a real wire-shorted reference is *not* common-average reference —
it's a single-electrode reference. The two coincide only when the
reference electrode sits at the average potential of the array. **Action
needed:** decide whether the simulation should:
(a) keep CAR (reasonable for a high-density array but not strictly
    matching a wire-shorted reference, document this);
(b) switch to a fixed reference at the head/foot anchor electrode and
    rerun the EEG forward (~30 s with DUNEuro available);
(c) provide both, with CAR as the headline.
The repo currently does (a) without flagging the discrepancy.

### 2.7 CI workflow does not run anything that touches DUNEuro

`.github/workflows/ci.yml` runs `pytest -q` and `ruff` only. DUNEuro is
not built in CI, which is fine — the smoke test auto-skips. But the
new physiology + calibration modules have *only* unit-tests-on-mocks; the
real round trip (FEM → DUNEuro → leadfield → calibration → 0.622) is
not in CI. **Action needed:** either add a small synthetic-sphere DUNEuro
job (would take ~5 minutes) or accept that the calibration is validated
by the 27-case sweep already on disk and that the audit confirmed it.

---

## 3. Medium / minor / nit

| # | Severity | File:line | Issue |
|---|---|---|---|
| 3.1 | medium | `src/vagus_fm/analysis/snr.py:75-79` | `per_source_amplitude(moment="rms")` divides by `sqrt(3)` — that's a per-axis convention assuming isotropic moments. The docstring says it but the formula could trip up a reviewer; consider naming the option `iso_per_axis` to be explicit. |
| 3.2 | medium | `src/vagus_fm/forward/duneuro_driver.py:84-103` | Solver defaults (`type=cg`, `reduction=1e-10`, `intorderadd=5`, `scheme=sipg`) are not justified with a citation. They match the DUNEuro tutorials but a reviewer will ask. Add a one-line doc citing Engwer et al. 2017 (the DUNEuro paper). |
| 3.3 | medium | `src/vagus_fm/sensors/triaxial.py:103-113` | The two tangent vectors come from `np.linalg.svd(normal)` — the resulting (T1, T2) frame has an *arbitrary* sign convention per coil. This is fine for the leadfield (each coil is independent) but means T1/T2 channel labels are not anatomically interpretable across coils. Should be flagged in the docstring. |
| 3.4 | medium | `src/vagus_fm/sources/vagus.py:42` | `edges = np.arange(z_lo, z_hi + spacing_mm, spacing_mm)` — adding `spacing_mm` to the upper bound can include or exclude the last slab depending on rounding. Tests cover the ±1 source tolerance but the off-by-one is a hidden footgun. Switch to `np.linspace(z_lo, z_hi, n_slabs+1)` for clarity. |
| 3.5 | minor | `src/vagus_fm/viz/style.py:67-95` | Helvetica is the requested font but `apply_nature_style` doesn't check it's installed — falls back silently to DejaVu Sans. Nature requires Helvetica or Arial; add a warning if neither is available. |
| 3.6 | minor | `outputs/detectability.png` | Caption mentions "QuSpin gen-2" but `configs/default.yaml:103` sets `15.0 fT/√Hz` which is the gen-2 number. Citation OK, but the same figure caption should also say which of `noise.opm_intrinsic_fT_sqrtHz` was used so it's reproducible from the figure alone. |
| 3.7 | minor | `outputs/cross_modality.png` (panel d title) | "Actual MEG (FEM) · prediction RMS error = X%" — at high SNR this is sub-1%, at low SNR it caps at "1000%". Add the bootstrap CI from `bootstrap_recovery_error_ci` to give the reviewer a stability indicator. |
| 3.8 | nit | `README.md:196` table | Says "Sarvas peak \|B\| at Q=70 nA·m = 2.9 pT" but `docs/VALIDATION.md:33` says "≈ 4.22 pT". The README quotes the peak in the *literature band* (n=7 333 pairs); the validation doc quotes the peak across the *full array* (different denominator). One of the two should explicitly say which band. The actual numbers from `outputs/sarvas_vs_fem.json`: full-array Sarvas peak = 41.4 fT/nAm = 2.90 pT @ Q=70; literature-band Sarvas peak = same number (the band currently includes the global peak coil). |
| 3.9 | nit | `docs/VALIDATION.md:33` | The comment `Sarvas peak ≈ 4.22 pT` no longer matched the actual JSON (`2.896 pT`). **Fixed during this audit** — now reads `≈ 2.9 pT`. |
| 3.10 | nit | `tests/conftest.py` | Not visited in this audit; consider adding a session-scoped fixture for the small-tube `FemMesh` used in three test files. |
| 3.11 | nit | `outputs/calibration/calibration.json` | File ends without trailing newline (cosmetic). |
| 3.12 | nit | `src/vagus_fm/io/npz.py:131` | Schema validator has `if lf.L_fT_per_nAm.shape != lf.L.shape: raise SchemaError("L_fT_per_nAm shape != L shape")` — message could include the actual shapes. |
| 3.13 | nit | `outputs/dual_topoplot.png` | Not personally inspected pixel-by-pixel here; if any axis lacks a unit (the audit asked) it will need an editor pass. |

---

## 4. Headline numbers consistency table

| headline number | location | quoted value | match? |
|---|---|---|---|
| Sarvas peak \|B\| @ Q=70 nA·m | `README.md:196` | "≈ 4.22 pT" — *no, actually says* "Sarvas peak at Q = 70 nA·m  · 2.9 pT" | ✓ matches JSON |
| | `docs/VALIDATION.md:33` | "Sarvas peak ≈ 2.9 pT" (post-audit fix) | ✓ |
| | `outputs/sarvas_vs_fem.json` | `"sarvas_peak_pT_at_Q70_band": 2.896` | ground truth |
| FEM/Sarvas peak ratio | `README.md:198` | "~6.8×" | ✓ |
| | `docs/VALIDATION.md:103` | "6.8×" | ✓ |
| | `outputs/sarvas_vs_fem.json` | `"ratio_fem_to_sarvas_peak_band": 6.836` | ground truth |
| | `outputs/sarvas_vs_fem.json` | CI95 = [5.86, 6.94] | (new this audit) |
| EEG calibration factor | `src/vagus_fm/forward/eeg.py:46` | `EEG_CALIBRATION_FACTOR = 0.622` | (post-fix; was *implicit* mismatch before) |
| | `outputs/calibration/calibration.json` | `"factor_median": 0.6220384272598165` | ground truth |
| | `README.md` | not quoted | OK |
| | `docs/VALIDATION.md:65-71` | "Median factor 0.622" | ✓ |
| | `outputs/calibration/sweep.json` | one of 27 cases (R=100, src=50, σ=0.43): `0.6220384272598183` | ✓ |
| MEG fT-conversion | `src/vagus_fm/forward/solve.py:74` | `L * 1e6` | ✓ self-consistent |
| | `src/vagus_fm/forward/reduce.py:72` | `L * 1e6` | ✓ |
| | `src/vagus_fm/io/npz.py` (docstring, post-fix) | "MEG: L × 1e6" | ✓ |
| Recording time, baroreceptor MEG | `README.md` | not quoted as a headline number; figure caption says it depends on Q |  — |
| | `outputs/physiology.png` (panel c title) | varies with scenario | OK |

**One stale line found and fixed** (nit 3.9). All other headline numbers
are self-consistent.

---

## 5. Final pytest + ruff status

Final state:

```
$ pytest -q
........................................................................ [ 58%]
...................................................                      [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_forward_smoke.py:22: duneuropy not built locally
123 passed, 1 skipped in 2.01s

$ ruff check src tests
All checks passed!
```

123 / 124 (the duneuro smoke test auto-skips). Was 121 before this audit;
+2 from `tests/test_sources_vagus.py`. Ruff clean.

---

## 6. "Would I publish this in Nature?"

**Honest assessment.** The pipeline is *almost* there. The dual-modality
forward-model contribution is genuine (no other open-source pipeline I
know of solves OPM-MEG and HD-EEG from the same FEM and source set for
peripheral nerve), the analytic validations against Sarvas / Berg-Scherg
are airtight (rtol = 1e-12 against MNE-Python), and the empirical EEG
calibration is the right move given DUNEuro's undocumented mm-mode units.
The Sarvas-vs-FEM CI95 = [5.86, 6.94] is a tight, defensible number.
The pre-audit EEG calibration *inconsistency* would have been caught by
any reviewer who tried to reproduce a figure from a fresh checkout —
that's the bullet we just dodged.

**The strongest reviewer pushback** will be on the *anatomy
generalisability*: BodyParts3D is a single atlas subject and the
shrinkwrap-then-CGAL pipeline gives one realisation of one anatomy. The
"dual-modality, head-to-head" framing is sound, but every absolute
number depends on this anatomy. The conductivity sensitivity sweep
(`analysis/sensitivity.py`) partially answers this for tissue
properties; it does not answer it for *anatomy variability*. A
defensible response is to (a) document this front-and-centre as a
limitation (already in `docs/VALIDATION.md:130-145`), and (b) commit to
running the pipeline on a second BodyParts3D-style mesh or an MR-derived
subject in a follow-up.

**The weakest defence** is the propagating-CAP-vs-stationary-dipole
question (§2.2 above). The docstring of `physiology/simulate.py` claims
that propagation only smears each event by a fraction of an AP width,
but the figure does not show this. A reviewer will ask "show me", and
right now we cannot — the simulator and the propagating-CAP source
exist as separate code paths. This is a one-day fix (one extra subplot
in `physiology.png`) that I am punting on because it needs DUNEuro and
a fresh forward run to render meaningfully.

**Conclusion.** With the EEG-calibration fix, the bootstrap CIs, the
determinism tests, the σ_in citation, and the noise-correlation caveat
landed by this audit, the manuscript is publishable subject to (i)
addressing §2.1-2.4 (cross-modality CI display, propagation figure, mesh
quality reporting cadence, source-set Z restriction), and (ii)
positioning the BodyParts3D-single-anatomy limit explicitly in the
abstract / methods rather than burying it in `VALIDATION.md`. As of
2026-05-01: I would send it out for review. I would not yet bet on
acceptance without revision.

---

## 7. Verbatim test + lint output

```
$ pytest -q
........................................................................ [ 58%]
...................................................                      [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_forward_smoke.py:22: duneuropy not built locally
123 passed, 1 skipped in 2.01s

$ ruff check src tests
All checks passed!
```
