# Working on iNOB

Read by Claude Code (and any agent following the AGENTS.md convention) at
the start of every session. Short on purpose; the code and `--help` are the
reference.

## What this is

A forward-modelling and sensor-planning package for MEG/EEG of peripheral
and autonomic nerves. Its output is a number a physicist will act on — a
field in fT, a trials-to-detect count — so a wrong magnitude is a worse bug
than a crash. Published as a Software Toolbox paper; reviewers run it.

## Before editing

- `inob doctor` says what this machine can run. Without `duneuropy` every
  stage except `forward`/`eeg`/`volume-field`/`calibrate` works; the DUNEuro
  tests skip and say so.
- `CHANGELOG.md` `[Unreleased]` is the recent-decisions record. Read it
  before re-deciding something.
- `docs/COORDINATES.md` fixes the axis convention. Anterior is −Y. Don't
  re-derive it.

## The physics gate

Anything that can move a leadfield magnitude — `src/inob/forward/`,
`analysis/sphere_calibration.py`, `analysis/analytic_sphere.py`,
`analysis/sarvas_compare.py`, conductivity handling in `config.py`, sensor
geometry in `sensors/` — must leave `tests/test_meg_sphere_validation.py`
and `tests/test_venant_sphere_validation.py` passing against a real DUNEuro
build. Locally: `pytest -m duneuro`. In CI: the `duneuro-validation`
workflow, which treats a skipped DUNEuro test as a failure. If you can't run
it, say so in the summary; don't assume.

Do not change a calibration constant, a `sigma_unit_scale`, or a unit label
to make a test pass. Find out which side is wrong.

## Rules

- `make lint` and `make test` clean before you stop. `make ci` adds the
  package build.
- Every behaviour change has a test. Every user-visible change has a line
  under `[Unreleased]` in `CHANGELOG.md`.
- `--help` text is for the person running the command, not the maintainer:
  no Sphinx markup, no internal attribute names, ≤79 columns. Tested in
  `tests/test_cli_help.py`.
- No `except Exception: pass`. Log it (`logger.debug(..., exc_info=True)`)
  or state in a comment why swallowing is right (teardown, logging handlers).
- No new `TODO`/`FIXME` in shipped code; put it in the changelog or an
  issue. `PHYSIOLOGY-TODO` is the one sanctioned marker and is grepped for.
- Comments say why, not what. Match the density and voice of the file.
- No new runtime dependency without saying so; the install story is part of
  the paper.
- Meshes are Git LFS and BodyParts3D-derived (CC BY-SA 2.1 JP). Never add a
  mesh whose provenance you can't state.
- Don't commit or push unless asked. Don't tag; tags mint a Zenodo DOI.

## Stop and ask when

- The task needs a new tissue type, a new conductivity default, or a change
  to `configs/default.yaml` semantics — those are scientific choices.
- A DUNEuro build is failing and the fix is in `scripts/build_duneuro_local.sh`
  or the `duneuro-build` patch: it affects the cluster and Docker builds too.
- Anything touches `CITATION.cff`, `.zenodo.json`, `LICENSE*` or authorship.

## Ending a task

Summarise: what changed and why, what was tested and how (paste the pytest
tail), what was not run and why, open questions. Don't pad.
