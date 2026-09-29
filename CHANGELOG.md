# Changelog

All notable changes to iNOB. Conforms to
[Keep a Changelog](https://keepachangelog.com) and uses semantic-ish
versioning. Dates in ISO-8601.

## [Unreleased]

### Fixed — physics
- **The Berg-Scherg reference series under-read the surface potential by
  0.4-0.9x, and that is where the EEG calibration factor of 0.622 came
  from.** `homogeneous_sphere_eeg_potential` divided the radial term of the
  series by n and the tangential term by n(n+1); the correct coefficients,
  the radial and tangential gradient of the Neumann point-source series, are
  (2n+1) and (2n+1)/n. A centred tangential dipole read exactly half the
  textbook 3 q·r̂ / (4π σ R²). The symmetry tests could not see an amplitude
  error, and the MNE cross-checks never ran (below). The corrected series
  now matches the textbook centred dipole, an independent numerical Neumann
  solve (to 1e-12) and MNE's sphere EEG (to 1e-5, MNE's Berg approximation).

  Re-running the sphere calibration against DUNEuro with the corrected
  reference gives a median ratio of **0.98-1.03** across radius (80, 100 mm),
  depth (30-70 mm), conductivity (0.2, 0.43 S/m) and pitch (2, 3 mm): in
  mm-mode DUNEuro's EEG output is already µV per nA·m, as dimensional
  analysis says it should be, and the "units or source-model discrepancy we
  have not identified" was this bug. **`EEG_CALIBRATION_FACTOR` is now 1.0**
  (was 0.622), the dimensional-analysis value, which `inob calibrate` now
  reproduces: across 12 spheres (R 80/100 mm, depth 30-70 mm, σ 0.2/0.43)
  the median ratio is 0.975-1.028 (CV 1.8%). The least-squares ratio ranges
  0.72-1.02; it is dominated by the electrodes nearest the source, where a
  3 mm mesh is least accurate for a shallow source.
  Every electrode amplitude rises by 1/0.622 = 1.61x and every electrode
  trial count falls by 2.6x; electrode leadfields solved before this carry
  the old factor and must be re-solved. MG is unaffected.
- **The EEG sphere calibration compared two different references.** The
  DUNEuro potentials were common-average referenced and the analytic ones
  were not, so the median ratio was biased by the analytic mean. Both are now
  referenced, and a least-squares factor is reported beside the median.
- **The MNE cross-checks skipped on every current MNE.** They imported
  `_do_sphere_field` from `mne.forward._compute_forward`; MNE has since moved
  it to `_compute_forward_numba`. Both paths are now tried. With MNE
  installed the four Sarvas comparisons pass at rtol 1e-12, and a new EEG
  comparison joins them. MNE is still not installed in CI, so they skip
  there.
- **A figure could pair a MEG source with a different electrode source.**
  `inob torso` and the surface topoplot read the electrode leadfield at the
  MEG source's index. The two are solved separately and need not share a
  source list — the shipped muscle electrode solve has 7,563 sources against
  MEG's 730 — so the muscle ESG panel painted a dipole ~110 mm from the one
  in the MEG panel. Both now match by position
  (`inob.viz.detectability.matching_eeg_source_idx`), warn when the nearest
  electrode source is not the same point, and refuse when none is within
  10 mm.
- **An imported array's coils under the chin were reported as inside the
  skin.** The shipped skin is consistently wound but open at its cut ends,
  so the import oriented each face away from the body's centroid instead of
  trusting the winding — and wherever the surface faces back towards the
  centroid, a coil 10 mm out read as 10 mm in. A consistent winding is now
  trusted up to one global sign taken by majority; the centroid rule is kept
  for skins with no consistent winding. Exporting the shipped array and
  importing it again now raises no warning.
- **`inob location` looked for an untagged whole-body array.** Its leadfield
  default carried the target tag and its electrode-array default did not, so
  it only worked while a stray untagged file happened to exist. Both now
  follow the tagged names `--set electrodes.shape=whole_body` writes.
- **`scripts/run_target.sh muscle` always reported a failure.** It ran `inob
  physiology`, which refuses muscle by design (a single evoked volley, no
  event-train scenarios); the step is skipped for muscle targets.
- The MEG mm-mode scale comment cited a `calibrate_meg_factor` that does not
  exist. The factor of 0.1 is dimensional analysis; what checks it is the
  2% sphere agreement in `tests/test_meg_sphere_validation.py`.

### Added
- **`inob placement --optimise` designs an array for a whole structure**
  (Stephanie Mellor, after Roos, Hämäläinen & Iivanainen's cerebellar OPM
  work). Positions are added one at a time, each the one that adds the most
  total information (½ log2 det(I + L C Lᵀ/σ²), Iivanainen et al. 2021)
  about every source in the target given those already chosen, under the
  same spacing rule as the single-source comparison; the table sets it
  against the positions nearest the structure. `--write-array opt.npz`
  saves the chosen positions for `inob sensors --from`. Works for
  triaxial, single-axis and imported arrays. On the shipped anatomy the
  optimised array carries 14-17% more information than the positions nearest
  the spinal cord for 4-16 sensors (1% at 64) and <=1% more for muscle; for
  the vagus it does 5-6% worse than nearest-first, because greedy selection
  under a spacing rule has no optimality guarantee. Proximity is an adequate
  placement rule for these targets, as the single-source comparison found.
- **`scripts/paper_numbers.py` — every number the worked examples quote,
  from the leadfields on disk.** The counterpart of `paper_stats.py` for the
  physics: noise floors per target, FEM/Sarvas ratios with their CIs, the
  three spinal source models on the patch and the whole-body array, vagal
  coupling along the nerve, the Q sweep, the propagation ratio and FWHM, the
  muscle MUAP, and both placement results. Each comes from the function the
  corresponding command uses. The draft had carried most of these numbers
  forward from superseded solves; run this after any re-solve instead.
- **A whole-body electrode solve no longer overwrites the target's patch.**
  With `--set electrodes.shape=whole_body`, `--source-target T` now writes
  `electrode_array_wholebody_T.mat` and `duneuro_eeg_leadfield_wholebody_T.npz`
  beside the patch files instead of over them. The coverage comparisons and
  torso panel b need both, and until now the second could only be made by
  renaming files by hand. `scripts/brain_sanity.py` reads the tagged brain
  file.
- **`scripts/eeg_calibration_sweep.py`** runs `inob calibrate` across radius,
  depth, conductivity and pitch and reports the spread. It replaces the
  "27-case sweep" the manuscript cited, which was not in the repository.
- **`scripts/sensor_sweep.py` — what noise and stand-off cost.** Rebuilds the
  magnetometer array at each stand-off, solves one source with DUNEuro, and
  reports amplitude and trials under each noise preset (default: spine C7
  source; 6.5, 10, 20, 30 mm; QZFM-3 against QZFM-2).

### Changed
- **Triaxial tangential coils are oriented about a fixed axis, not per
  sensor.** T1 and T2 used to come from the SVD null space of each sensor's
  own normal, which is orthonormal but leaves the split between them to
  LAPACK: on the shipped 2,730-position array T1 wandered up to 30 degrees
  out of horizontal (median 5) and jumped between neighbouring sensors by
  several times their angular separation, so a tangential field map carried
  steps the field does not have. Both tangents are now built about +Z —
  `T1 = up x n`, `T2 = n x T1` — making T1 the circumferential axis and T2
  the head-foot one at every sensor. `sensors.tangent_basis: local_svd`
  restores the old frame for reproducing leadfields solved before this.

  The radial channel is bit-identical and the tangent plane the two axes
  span is identical to machine precision, so no field and no leadfield
  magnitude changes. What does move is `per_source_peak`, which maxes over
  channels and so depends on where inside that plane the axes sit: on the
  solved leadfields the array peak shifts by -0.1% (vagus), +2.9% (spine)
  and +0.1% (muscle), with individual sources moving up to 40%. Re-run
  `inob sensors` and re-solve to pick this up; an existing leadfield is
  unaffected until you do.
- **`inob torso` is two panels, not four.** The unrolled θ–z map and the
  patch-frame ESG topography are gone; what is left is MEG and ESG on the
  body, side by side. Both are painted on the same mesh, in the same camera,
  under the same orthographic projection, the same scale bar and the same
  symmetric colour rule, so the quarter-turn between the magnetic and
  electric pattern of one dipole reads straight off the figure. The two
  colour bars stay separate — the units differ by four orders of magnitude —
  but the rule that sets them is shared.
- **The ESG panel can be painted from a whole-torso electrode array.**
  `inob torso --esg-npz LF.npz` takes an electrode leadfield other than the
  target's HD patch, which is what makes panel b cover the body panel a
  covers; solve one with `inob electrodes --set electrodes.shape=whole_body`
  followed by `inob eeg`. The default source is still chosen from the
  configured patch, so pointing this elsewhere changes what is drawn, not
  which source the figure is about. The interpolation kernel and the
  "measured" mask now come from the array's own contact spacing rather than
  `electrodes.contact_pitch_mm`, which describes only the paddle.

### Fixed
- **The exploded anatomy figure sets its type from the render height.** Label
  size was a fixed pixel count chosen for a 2,600 px render, so asking for a
  taller figure silently shrank the captions. It is now a fraction of the
  height, and set larger; leader lines are thicker and darker, which they had
  to be to survive reduction to a column width.
- **A system's label no longer lands on its neighbour's mesh.** Column packing
  tested mesh extents alone, so two systems that merely stack (lower-limb
  muscle under the spinal cord) shared a column and the lower caption was
  drawn across the upper mesh. Packing now reserves the label's headroom,
  which surfaced as soon as the type was large enough to read.
- **The body-surface field maps show the array that produced them.**
  `inob torso` draws the sensor positions over the painted field, so the map
  reads as the Gaussian interpolation of a 30 mm-spaced array sampled at the
  stand-off that it is, rather than as a continuous measurement on the skin —
  the interpolation is wide enough to smooth away the spacing it was built
  from. `--no-sensors` restores the clean surface.
- **`inob --help` described `torso` as a four-panel figure.** It has been two
  panels since the unrolled and patch-frame maps went; the summary now says
  what it draws, MEG beside ESG.

### Added
- **Bring your own array; single-axis arrays; a SQUID noise preset**
  (Stephanie Mellor, Harry Cook, Nic Alexander). `inob sensors --from FILE`
  (or `sensors.from_file`) imports a magnetometer array — CSV/TXT
  `x y z nx ny nz` in mm, an NPZ with `coil_pos`/`coil_orient`, or a
  FieldTrip `grad` .mat (`coilpos`/`coilori`, else `chanpos`/`chanori`;
  `unit` m/cm/mm converted; v7 and v7.3; gradiometer `tra` refused) — and
  `inob electrodes --from FILE` (or `electrodes.from_file`) does the same for
  electrodes (CSV `x y z`, NPZ, FieldTrip `elec` with `elecpos`/`chanpos`).
  Both write the file `forward`/`eeg` already read. Positions are taken as
  body frame (docs/COORDINATES.md) and registering them is the user's job;
  the import warns when coils sit inside the skin or > 100 mm from it
  (electrodes: > 10 mm), and refuses zero-length or non-finite values.
  Three orthogonal coils per position are stored in the R/T1/T2 block layout
  with R the coil nearest the skin normal; one coil per position is stored
  as single-axis `-R`; any other grouping keeps file order as `-A1`, `-A2`…
  `sensors.axes: radial` (default `triaxial`, unchanged) generates one coil
  per position along the outward skin normal, the usual SQUID arrangement.
  Figures that took the first third of the channels as radial now read the
  labels, and the placement comparison refuses a non-triaxial array with a
  message instead of reading the wrong channels. New noise preset
  `megin_triux_mag` (5 fT/√Hz, 3.3 kHz; Elekta TRIUX technical manual,
  `verified=False`); the default stays `quspin_qzfm3`.
- **Figure titles name the vertebral level, not just a height.** "z = 1393
  mm" meant nothing to a reader without the model open (Maike Schmidt,
  Stephanie Mellor). `inob.anatomy.vertebral_level` reads the segmented
  vertebra meshes and names the level of the axial plane through a source —
  "body of C6", "C4/C5 intervertebral gap" — from the anterior quarter of
  each bone, which is its body: whole-bone bounds overlap by a level because
  the spinous processes slope down. Titles now read "source #72, z = 1393 mm
  (C6)" in `torso`, `topoplot`, `sensor-field`, `cross`, `location` and
  `source-models`. The level is of the plane, not of the skin: the spine
  patch is centred on the C7 bone (spinous process included), and the cord
  source nearest it lies at the level of the C6 body. Without the bone
  meshes the suffix is simply left off.
- **`inob torso` says what is drawn and from which way.** An arrow shows the
  dipole's moment where there was only a star; the target's modelled sources
  (the cord, the vagus, the muscle cloud) are drawn through the skin so the
  dipole is seen inside its structure; a labelled x/y/z triad names the axes
  (anterior is −y) for the shared camera; a legend names each mark; and a
  subtitle states that both panels come from the full-body FEM, not the
  sphere or Biot–Savart models (Nic Alexander, Stephanie Mellor).
- **Electrode re-referencing and a fixed dipole orientation, for `inob
  detect` and `inob snr`.** Asked whether ESG loses to OPM because of the
  common-average reference, and how a dipole pointing out of the back
  compares (Stephanie Mellor). `--eeg-reference average|electrode:N|farthest|
  nearest:x,y,z` re-references the stored leadfield (linear on its rows, no
  re-solve; only contacts in the array can serve). In `inob detect` it adds a
  single-contact `EEG_referenced_*` figure beside the bipolar one, which is
  reference-free and does not move. `--orientation x|y|z|posterior|anterior|
  vx,vy,vz` fixes the moment; the default "best" is now defined in `--help`
  as the maximum over channels (or contact pairs) and over the x, y, z axes.
  Both default to the old behaviour and leave the output unchanged. At the
  C7 source (#72) with Q = 5.11 nA·m, the 32-contact patch reads 0.19 µV
  average-referenced, 0.33 µV far-referenced (402 trials, equal to the
  best pair); the 1200-contact whole-body array reaches 0.64 µV (104
  trials); the OPM needs 6. A posterior (+Y) dipole is harder for both, and
  worse for the electrodes: 0.05 µV on the patch (~17,000 trials) and 0.11
  µV on the whole-body array (~3,600), against 179 fT (8 trials).
- **`noise.eeg_contact_temperature_K`** (default 310.15 K, unchanged) sets
  the temperature in the electrode Johnson term. Skin-like 305 K lowers the
  electrode noise floor by 0.014%, because the amplifier dominates it.
- **`scripts/brain_sanity.py` — MG against EG for the one target where both
  are well measured.** Asked why electrography comes out so far behind
  magnetography at the neck (Stephanie Mellor), the package should first get
  the brain right. On the shipped whole-body FEM a cortical source 20-30 mm
  under the skin gives a median 134 fT/nA·m at the OPM and 0.22 µV/nA·m
  between the best contact pair — 1.3 pT and 2.2 µV for a 10 nA·m evoked
  dipole, which is where on-scalp OPM and scalp EEG evoked responses sit. The
  leadfields are not the asymmetry. The noise floors are: under the 217 fT
  and 2.19 µV floors every other target is judged against, EG's SNR is
  0.14-0.31 of MG's for the brain, where MEG and EEG evoked SNR are
  empirically comparable (Goldenholz et al. 2009). Both floors are
  instrument-only — no environmental interference for the OPM, no EMG, ECG
  or background EEG for the electrodes — and that, not the forward model, is
  where the MG advantage in the worked examples comes from. The defaults are
  unchanged; a noise model calibrated against empirical brain SNR is an open
  question.
- **`scripts/paper_stats.py` — the counts the manuscript makes about the
  package, measured rather than remembered.** Four of the six were wrong by
  submission: 605 tests against 948, 18,400 lines of Python against 22,066,
  28,000 with tests against 33,789, and 13,000 lines of TypeScript against
  9,610. (The two Nic Alexander queried, the subcommand and binary counts,
  were the two that were right.) Run it and paste the output at each
  revision.
- **`inob placement` — does the forward model beat a tape measure?** Imaging
  Neuroscience asks a software toolbox to compare itself against existing
  practice, and the practice here is putting the sensors as close to the
  target as they will go. The command places `--n-sensors` positions two
  ways from one solved leadfield — by predicted coupling, and by proximity
  to the source — holds both to the same minimum spacing, and scores both
  by the multichannel matched-filter amplitude.

  On the shipped anatomy the answer is that it does not: the two arms land
  within 1% (vagus), 1% (spine) and 9% (muscle) of each other, because
  under a convex skin surface the best-coupled positions are very nearly the
  nearest ones (Spearman rho 0.88-0.99 between coupling and proximity; 19-20
  of the top 20 positions shared). That holds out to a 50 mm error in where
  the experimenter thinks the source is. The model's value is in the
  amplitude and the trial count, which proximity cannot supply at all, and
  in which region to cover — not in picking sensors within a covering array.
  `--min-spacing-mm` and `--source-idx` vary the comparison;
  `inob placement --help` states the result so nobody has to rediscover it.
- **`per_source_triaxial_peak`** in `inob.analysis.snr`: the field magnitude
  over a position's three axes, rather than its largest single axis. A
  triaxial sensor measures all three at once, so any orientation is
  recoverable afterwards and the available signal is the vector magnitude —
  which, unlike `per_source_peak`, does not depend on the tangential
  convention. It is not wired into the reported figures; `per_source_peak`
  remains what `inob detect` uses.
- **Per-tissue mesh quality at FEM build time.** `inob build-fem` now logs
  the Joe-Liu distribution for each compartment, not just one figure for the
  whole mesh, via `inob.mesh.quality.quality_by_tissue`. On the shipped
  whole-body mesh the worst of 294,031 elements scores 0.0989 against the
  0.05 gate and the median is 0.585, so the threshold is never binding: it
  is a tripwire for a mesher failure, not a tuning knob, and that is now
  stated where it is enforced.
- **Heart, brain and lungs are modelled tissues, not just figure anatomy.**
  The FEM gains three compartments — myocardium 0.20 S/m (Gabriel 1996),
  homogeneous brain 0.33 S/m (Haueisen 2002) and inflated lung 0.10 S/m
  (Gabriel 1996) — and `--source-target` gains `heart` and `brain`, so the
  cardiac and cortical generators can be solved on the same body, the same
  array and the same noise floor as the vagal one. The heart compartment is
  myocardium throughout (chamber blood is not separate) and the brain has no
  grey/white split and no CSF; both are stated in the config and in the
  physiology profiles rather than left for the reader to discover. Lungs are
  a volume conductor only — 3.2 litres of low-conductivity tissue flanking
  the heart, which is what shapes the cardiac field seen at the neck.
- **Physiology profiles for the two new targets, both marked PROVISIONAL.**
  `heart` carries the ~2 µA·m peak-QRS equivalent dipole (Baule & McFee 1963;
  Wikswo 1980) and an 80 ms QRS width; `brain` carries the conventional
  10 nA·m cortical evoked ECD (Hämäläinen 1993) and a 30 ms response width.
  Both are measured equivalent dipoles, so `PhysiologyProfile` gained
  `event_moment_nAm_override`: a generator that is not a volley of myelinated
  axons now states its event moment instead of deriving a wrong one from the
  Hämäläinen per-fibre formula. Neither profile has an activation-sequence or
  cortical-patch model, so both answer "how large is the field here" and not
  "what does the waveform look like".
- **A labelled exploded view of the anatomy the model has to represent**
  (`scripts/exploded_anatomy.py` → `outputs/anatomy/exploded_anatomy.png`):
  brain, vertebral column, spinal cord, vagus nerve, heart, lungs,
  gastrointestinal tract, great vessels and lower-limb muscle, each drawn in
  place inside a translucent body and again pulled out to its own column with
  a leader line back. Nothing is hand-placed: systems are packed into columns
  by height (two may share one only if they do not overlap vertically),
  columns are dealt outward narrowest-first to whichever side has room, and
  labels that would collide are lifted apart — so adding a system re-flows
  the figure instead of breaking it. Three-point lighting and per-tissue
  gloss, because one specular setting for all nine makes them look moulded
  from the same plastic.
- **Selective BodyParts3D fetch** (`scripts/fetch_bodyparts3d_subset.py`).
  The existing downloader takes the whole 3,210-mesh 4.3 object set; this one
  takes the same manifest and endpoint and asks only for named groups — 222
  meshes and ~110 MB for the figure above. `--list` prints exactly which
  concepts a group's patterns select before anything is downloaded, and the
  written `MANIFEST.csv` records what actually arrived rather than what was
  requested. Same provenance as the rest: BodyParts3D/Anatomography 4.3,
  CC BY-SA 2.1 JP.
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
- **The torso scale bar could be drawn off the panel.** It is laid along the
  screen-horizontal axis, which reverses as the camera swings past ±90°, but
  it always started from the same corner — so on the spine views the bar and
  its label ran out of the frame and were clipped.
- **The physics validation could not fetch its sources.** Since 2026-08-24
  `gitlab.dune-project.org` has answered GitHub's runners with 403, so every
  scheduled `duneuro-validation` run failed before compiling anything and the
  sphere check had not actually run for two and a half weeks. The Dockerfile
  now takes the DUNE 2.10 modules, duneuro (pinned commit) and duneuro-py
  from source tarballs on the `olivercase/duneuro-build` v1.1.0 release,
  verified against its `SHA256SUMS`, with the GitLab clone kept as a fallback.
  Same sources for everyone, and a build that works where GitLab does not.
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
