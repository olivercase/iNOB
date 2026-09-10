# Changelog

All notable changes to iNOB. Conforms to
[Keep a Changelog](https://keepachangelog.com) and uses semantic-ish
versioning. Dates in ISO-8601.

## [Unreleased]

### Added
- **Engineering hygiene, measured against a sibling project.** Pre-commit
  hooks (`.pre-commit-config.yaml`, `make hooks`): whitespace, YAML/TOML/JSON,
  large files, private keys, ruff, gitleaks, and a guard against new
  `TODO`s in shipped code. CI gains a hooks backstop, a full-history secret
  scan, a coverage artifact and a package-build check; every action is
  pinned to a commit SHA, every workflow declares read-only permissions,
  superseded runs are cancelled, every job has a timeout. Dependabot for
  pip, npm and actions. `SECURITY.md`. `CLAUDE.md` with the physics gate
  and the rules an agent must follow here. `make ci` runs the same gate
  locally.

- **`ruff format` across the repository, enforced from here on** (hook, CI
  and `make ci`). One commit, no logic change; it is listed in
  `.git-blame-ignore-revs` so `git blame` looks through it.

### Fixed
- **Six `except Exception: pass` sites now say something.** Five decorative
  skin backdrops in the topoplot and cross-modality figures log at debug
  with the traceback instead of vanishing; the GUI capability report logs
  an unreadable config. The two that stay silent (a logging handler, a
  websocket close at teardown) say why.

## [1.1.0] — 2026-09-10

The help text is the manual, and the repository is packaged for review.

An end-to-end audit of everything a user reads. `--help` is the only
documentation most people will open, and fifteen commands were printing the
maintainer's module docstring at them instead — Sphinx roles, double
backticks, internal attribute names and a leading `CLI:` label included.

### Changed
- **Every command now has its own `description=`,** written for the person
  running it rather than the person maintaining it. Module docstrings stay as
  they were; they are simply no longer the help text. Gone with them:
  unrendered ``markup``, `:mod:` roles, `cfg.outputs.fem_mat`-style internals,
  "Returns the absolute output path", and the two examples that still taught
  the legacy `inob-physiology` spelling.
- **Help fits an 80-column terminal.** `inob run`'s description was one
  161-character line; `inob sensor-field --level` spilled all 24 vertebral
  levels into the usage block for want of a `metavar` the other two `--level`
  flags already had.
- **Example blocks line their descriptions up again.** Six commands had drifted
  to two or three different columns, and two had a single space where the gap
  should be, so the command ran into its own description.
- **`inob --help` documents `--debug` and `--version`,** and no longer claims
  `--config` is unavailable to `doctor` and `status`, which both take it.
- **"Did you mean" pads its suggestions** on the plain command name — the ANSI
  colour codes were being counted as width, leaving the list ragged.
- **README's command table lists all 26 commands.** It was missing `ladder`,
  `torso`, `sensor-field`, `source-models`, `volume-field` and
  `muscle-sources`, and still advertised "140 tests".

### Fixed
- **Config warnings are part of the report, not a wall of text above it.**
  Loading the config is what says where the log file goes, so warnings raised
  while loading it arrived through logging's last-resort handler: unformatted,
  unwrapped, ahead of the command's own output, and absent from the log
  entirely. `doctor` now folds them into the Config check as a warning with
  hints; `status` gives them a `Note` section and a `notes` key in `--json`;
  and `logging_setup.begin_capture` buffers them so they reach the log file in
  the order they happened.
- **GUI: "halving it halves the trials needed by a factor of four"** — noise
  and trials-to-detect were being described as halving and quartering in the
  same breath. Trials go as the square of the noise; the copy now says so.

### Added
- **Submission-ready packaging (2026-09-10).** `docs/QUICKSTART.md`,
  `docs/REPRODUCE.md`, `docs/COMPARISON.md` and `docs/CITING.md`;
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and pull-request templates;
  `.zenodo.json` so a tagged release archives with the right metadata; a
  `release` workflow that builds the sdist and wheel, checks the version
  against the tag, attaches them to the GitHub release and publishes to PyPI
  by trusted publishing; `scripts/package_results.sh` to bundle solved
  leadfields and figures with a SHA-256 manifest for a results deposit.
  `pyproject.toml` gains project URLs, keywords and the MIT classifier;
  `LICENSE` is the bare MIT text again so GitHub detects it (the data note
  lives in `LICENSE-DATA` and the README); `CHANGELOG` rolls the dated
  development notes up under `1.0.0`.
- **`--help` is tested like the published prose it is:** no Sphinx markup, no
  `CLI:` label, no legacy binary names, nothing wider than 79 columns, and one
  description column per example block. Plus a test that the README's command
  table and `inob --help` cannot disagree about which commands exist or which
  group they belong to.

## [1.0.0] — 2026-08-30

First tagged release (`v1.0.0`). The dated development notes below were
written as the work landed and are kept as they were.

### sensor bandwidth is half of a sensor spec, 2026-08-21

A magnetometer is two numbers. Quoting the noise density without the bandwidth
is how a QuSpin QZFM-3 ended up credited with a 30–500 Hz recording band it
cannot deliver — integrating noise it never sees, and paying nothing for the
CAP energy its 135 Hz pole cannot pass.

#### Added
- **`inob.sensors.opm_presets`** — OPM presets carrying noise density *and*
  3-dB bandwidth as a pair: `quspin_qzfm3` (7 fT/√Hz, 135 Hz — the default),
  `quspin_qzfm2` (15, 135), `fieldline_v3` (15, 500), `he4_wideband`
  (30, 2000). `noise.opm_sensor` names one; either number can still be
  overridden individually. The last two carry `verified=False` and log a
  warning: manufacturer-class figures, not checked against a current spec
  sheet by anyone here.
- **`--set noise.opm_bandwidth_hz=…`** and `NoiseCfg.opm_noise_bandwidth_hz`,
  the equivalent noise bandwidth of the recording band seen through the
  sensor pole: `f₃dB·[atan(hi/f₃dB) − atan(lo/f₃dB)]`.

#### Fixed
- **The sensor now rolls off signal and noise alike.** One pole, applied both
  ways: the noise integral saturates above the pole, and the CAP is scaled by
  `1/√(1 + (f_CAP/f₃dB)²)` at its spectral peak `f_CAP = 1/(2πσ)` — exact for
  the Gaussian-derivative AP shape in `inob.sources.cap`. Folded into the MEG
  σ returned by `compute_noise_floors`, so every consumer inherits it and none
  of them can reward a narrow-band sensor for the noise its own bandwidth
  removes. **Narrowing the band can no longer improve SNR** (tested).
- **The QZFM-3's effective floor is 217 fT, not 152 fT**, for the 0.5 ms vagal
  CAP: it passes 39% of a 318 Hz signal. Note what this does *not* do — it does
  not hand back the 6.7× fewer trials that dropping 1 kHz → 150 Hz appears to
  promise on the noise side alone. Against the old broadband 1 kHz figure
  (221 fT) the honest number moves by 2%, because the band the arithmetic
  removed was band the signal was living in.
- **The GUI has a light theme.** Every colour in `globals.css` is a token now,
  including the sheens and shadows that were hard-coded `rgba(255,255,255,…)`
  and `rgba(10,11,10,…)`; `<html data-theme>` picks the palette, `?theme=` or
  the top-right toggle sets it, and `@media print` forces light regardless, so
  a printed page is not a black slab. The 3-D well follows via `useTheme`.

### like-for-like modality comparison, 2026-07-29

`inob source-models` answers the spine question that does *not* need the source
strength to be settled: how do OPMs and surface electrodes compare at measuring
the same cord activity, stationary versus ascending?

#### Added
- **`inob.analysis.source_models`** — three source models at one moment, one
  orientation and one reduction per modality, so only the spatiotemporal model
  varies: `synchronous` (whole cord in phase — the naive upper bound),
  `stationary` (one segment, lumped) and `ascending` (the same moment
  travelling at fibre conduction velocity).
- **Every ratio it reports is independent of the assumed source strength.**
  Both modalities' amplitudes are linear in `Q`, so the modality gap, the
  propagation penalty and the coherence penalty cancel it exactly — tested, not
  asserted. This is the point: the cord's absolute moment is contested, and
  none of the conclusions here depend on resolving it.
  Absolute amplitudes are still reported, flagged as scaling with the assumed
  `Q` rather than as predictions.
- `inob.viz.source_models_plot` — SNR, averaging cost, and a panel of just the
  `Q`-independent ratios.

#### Result
At C7 with `Q = 5.11 nA·m` per active source:

| source model | OPM | electrodes | OPM/elec SNR |
| :--- | ---: | ---: | ---: |
| synchronous (whole cord) | 5,698 fT, 1 trial | 7.54 µV, 1 trial | 10.9× |
| stationary (one segment) | 210 fT, 5 trials | 0.32 µV, 411 trials | 9.4× |
| ascending (volley) | 102 fT, 20 trials | 0.12 µV, 2,871 trials | 12.0× |

The modality gap is **9–12× in SNR (≈90–140× in trials) regardless of source
model** — propagation costs both modalities within a small factor of each other
(OPM ×0.49, electrodes ×0.38), so it changes the averaging budget but not which
instrument to choose. Treating the cord as synchronous inflates the OPM
amplitude ~56× over the travelling volley, which is the size of the error the
vagus-style stationary assumption makes on a 475 mm structure.

Cross-checks out against `inob detect` (210 vs 215 fT, 411 vs 418 trials — the
small gap is the longitudinal projection used here versus the best moment
component used there).

### the cervical N13 removed, 2026-07-31

The N13 line of work is dropped. The spine model is back to one generator, the
ascending dorsal-column volley, and the question the package answers for the
cord is the source-model comparison above: stationary versus ascending, in
ratios that do not depend on an absolute source strength. The N13 required a
second, inverted calibration whose two anchors disagreed by ~22x, and closing
that gap needed cluster work that is not the point of this study.

#### Removed
- `inob.physiology.n13` and the `inob n13` command, including
  `PostsynapticGenerator`, `estimate_n13` and `SPINE_PROFILE.postsynaptic`.
- `electrodes.shape: ssep_montage` and its `montage_levels` / `bone_dir`
  params, which existed only to build the Cv6-AC pair the clinical N13
  amplitude is defined at.
- `inob.anatomy.skin_point_along` and the `ANTERIOR` / `POSTERIOR` constants,
  which had no remaining callers. `docs/COORDINATES.md` remains the reference
  for the +Y-is-posterior convention.

### spine EEG detectability, 2026-07-29

Investigating why the spine EEG amplitudes looked small. Most of the smallness
turned out to be real physics about patch arrays; what was wrong was how the
observable, the noise floor, and the source strength were being reported.

#### Fixed
- **EEG signal is now the best bipolar pair, not the best single channel.**
  `inob.analysis.snr.per_source_best_bipolar`. A surface potential exists only
  as a difference between contacts, so a per-channel figure depends entirely on
  the reference — and the saved leadfield's common-average reference removes
  most of a deep source's amplitude across a patch-sized footprint. The bipolar
  amplitude is reference-invariant (tested) and runs ~1.7× above the referenced
  peak on the cervical spine patch. Recoverable from existing leadfields; no
  re-solve needed.
- **Noise floor.** `eeg_amplifier_uV_sqrtHz` 0.5 → 0.1 (the 0.3–1 µV/√Hz range
  is a system-level figure that double-counts interface noise already modelled
  separately), and the noise-integration band is now the recording passband
  `noise.band_lo_hz`…`band_hi_hz`, defaulting to the 30–500 Hz evoked-potential
  band rather than broadband 0–1000 Hz. Together these take σ_eeg from 15.8 µV
  to 2.2 µV. The legacy scalar `noise.bandwidth_hz` still overrides the pair.
- **The spine planned against the vagus's source strength.** `scenarios_for_target`
  returned the vagal-CAP Q ladder (topping out at Bu et al.'s 70 nA·m) for spine
  targets — an order of magnitude above anything reported for the cord. Spine
  now gets its own ladder built around `SPINE_PROFILE.default_strength_nAm`
  (5.11 nA·m, Kawabata 2002 / Sasaki 2008 magnetospinography), pulled from the
  profile so the detectability and time-domain figures cannot drift apart.
- **Single-source panels quoted the wrong source.** `default_source_idx` now
  picks the source nearest the electrode array centroid instead of the midpoint
  of the source list. On the spine the midpoint is mid-thoracic, 146 mm from a
  C7 patch instead of 57 mm, which made panels a/c/d/f ~700× pessimistic and
  inconsistent with panels b/e beside them.
- **detectability and cap_compare contradicted each other.** `cap_compare_spine.png`
  printed "stationary approximation INVALID" and measured a propagating/stationary
  ratio of 0.298, while the detectability figure beside it computed signal as
  leadfield × Q — i.e. as if the ratio were 1. The wave simulation moved out of
  `viz/cap_compare.py` into `inob.analysis.propagation`; both figures now call
  it, so they cannot drift apart. `cap_compare` output is byte-identical after
  the refactor.

#### Added
- **Clinical averaging budget** (`CLINICAL_AVERAGE_BUDGET`, 500–2000 averages,
  Cruccu et al. 2008) drawn on the detectability panels for evoked targets, and
  `EEG_within_clinical_budget` in the summary. For a stimulus-locked paradigm
  the feasibility question is whether the answer fits a protocol that already
  exists, not the raw µV.
- **Both source models plotted.** Where a profile has `stationary_ok=False`,
  every detectability panel now carries a dashed propagation-corrected family
  alongside the solid stationary one (an upper bound), with the factor in the
  legend and the summary (`*_propagating` keys). The vagus, whose lumping is
  defensible, is unchanged — no second family is drawn.
  Two things had to be got right for the correction to compose:
    * the stationary reference must be the source the panel quotes, not the
      polyline's rostral end. Measured against the wrong reference the spine
      patch gave 8.0 — propagation apparently *amplifying* the signal.
    * the reduction must be each modality's own observable (best channel for
      MEG, best bipolar for EEG), or the factor and the signal it scales are
      defined on different quantities.
  The resulting factors (0.49 MEG, 0.38 EEG) are legitimately different from
  cap_compare's 0.298, which uses its own lump position and best-radial-channel
  reduction.
- **Guard on unordered source sets** (`is_ordered_polyline`). The propagating
  model advances along cumulative arc length, which is meaningless for a
  volume-fill point cloud. The combined spine+muscle target produced a factor
  of 0.006 (a 170× attenuation) purely from point ordering; it is now skipped
  with a warning, and those panels show the stationary bound alone. Measured
  tortuosity: cord 1.06, muscle 57, spine+muscle 45 — threshold 3.

#### Result
At the C7 source with the literature-anchored Q = 5.11 nA·m, the 32-contact
patch needs ~420 averages under the stationary model (inside the clinical
500–2000 budget) and ~2,900 under the propagating model (outside it). The
propagating figure is the honest one for a travelling volley. The 1000-contact
whole-body array needs ~250 stationary.

#### Known-stale artefact (pre-existing, unrelated)
`inob detect --source-target muscle` fails: `duneuro_leadfield_muscle.npz` has
736 sources and `duneuro_eeg_leadfield_muscle.npz` has 729, so they are from
different solves. Needs a muscle EEG re-solve.

### Nature-reviewer audit, 2026-05-01

#### Fixed (showstopper)
- **EEG forward calibration consistency.** `src/inob/forward/eeg.py`
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

#### Added
- **Bootstrap confidence intervals** on:
    * The Sarvas-vs-FEM peak ratio
      (`inob.analysis.sarvas_compare._bootstrap_ratio_ci`,
      written to `outputs/sarvas_vs_fem.json` as
      `ratio_fem_to_sarvas_peak_band_ci95`).
    * The cross-modality MEG-from-EEG recovery error
      (`inob.analysis.cross_modality.bootstrap_recovery_error_ci`).
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

#### Notes
- 121 / 121 pytests still pass (1 auto-skipped duneuro smoke test).
- ruff clean.

## [0.1.0] — 2026-04 (initial public)
First public release: dual-modality (MEG + EEG) forward model from a
single FEM, Sarvas / Berg-Scherg analytic validation, OPM noise model,
HD-EMG patch electrodes, conductivity sensitivity sweep, propagating
CAP source model, physiology event-train scenarios.
