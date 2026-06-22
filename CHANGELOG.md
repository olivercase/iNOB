# Changelog

All notable changes to the Forward_Model_Vagus_Nerve pipeline. Conforms to
[Keep a Changelog](https://keepachangelog.com) and uses semantic-ish
versioning. Dates in ISO-8601.

## [Unreleased] — Nature-reviewer audit, 2026-05-01

### Fixed (showstopper)
- **EEG forward calibration consistency.** `src/vagus_fm/forward/eeg.py`
  applied an *uncalibrated* `L * 1e3` factor in the in-pipeline forward
  solve, while figures used a separately-applied `L * 0.622` empirical
  calibration. The two never matched. Forward solve now applies the
  Berg-Scherg-validated `EEG_CALIBRATION_FACTOR = 0.622` directly, so
  the saved NPZ on disk and a re-run of the forward solve are byte-
  identical. Single source of truth.
- **`io/npz.py` schema docstring.** The `L_fT_per_nAm` field is now
  documented as modality-dependent (`fT/nAm` for MEG, `µV/nAm` for EEG)
  rather than as a magnetic-only quantity. The historical name is kept
  to preserve schema compatibility with on-disk artefacts.

### Added
- **Bootstrap confidence intervals** on:
    * The Sarvas-vs-FEM peak ratio
      (`vagus_fm.analysis.sarvas_compare._bootstrap_ratio_ci`,
      written to `outputs/sarvas_vs_fem.json` as
      `ratio_fem_to_sarvas_peak_band_ci95`).
    * The cross-modality MEG-from-EEG recovery error
      (`vagus_fm.analysis.cross_modality.bootstrap_recovery_error_ci`).
- **Determinism tests** for the canonical source-set sampler
  (`tests/test_sources_vagus.py`):
    * Sampling the same FEM twice produces byte-identical sources.
    * Permuting tet ordering does not change the sources.
- **Noise-correlation caveat** on the detectability figure: trial counts
  assume independent white noise; spatially-correlated environmental MEG
  noise inflates `N_trials_for_SNR=3` by 1–10× depending on shielding.
- **σ_in citation** in `sources/cap.py` and `analysis/sarvas_compare.py`
  documenting the choice of intracellular conductivity (Hämäläinen 1 S/m
  vs Pelot 2017 0.35 S/m) and why we use the former for benchmark
  consistency with Doherty et al. 2024.
- **Documentation updates** to `docs/VALIDATION.md` reflecting current
  mesh-quality numbers (min 0.092, mean 0.524 over 789 342 tets) and
  the renamed calibration constant.
- This `CHANGELOG.md`.
- `CITATION.cff` for FAIR-compliant citation metadata.

### Notes
- 121 / 121 pytests still pass (1 auto-skipped duneuro smoke test).
- ruff clean.

## [0.1.0] — 2026-04 (initial public)
First public release: dual-modality (MEG + EEG) forward model from a
single FEM, Sarvas / Berg-Scherg analytic validation, OPM noise model,
HD-EMG patch electrodes, conductivity sensitivity sweep, propagating
CAP source model, physiology event-train scenarios.
