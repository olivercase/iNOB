# Contributing to iNOB

Thanks for looking. iNOB is research software; bug reports about the physics
and the numbers are as welcome as reports about crashes.

## Reporting a problem

Open an issue at https://github.com/olivercase/iNOB/issues with:

- the output of `inob doctor` and `inob --version`;
- the config you ran (or the `--set` overrides) and the command;
- what you expected and what you got. For a wrong *number*, say which
  analytic or published value you compared against.

## Working on the code

```bash
git lfs install && git clone https://github.com/olivercase/iNOB.git && cd iNOB
python3 -m pip install -e .[dev]
make lint      # ruff, whole repo
make test      # pytest; DUNEuro tests auto-skip without duneuropy
```

Before opening a pull request:

- `make lint` and `make test` pass;
- new behaviour has a test. Anything that can move a leadfield magnitude
  must keep `tests/test_meg_sphere_validation.py` and
  `tests/test_venant_sphere_validation.py` passing — those run for real in
  the weekly `duneuro-validation` workflow;
- `--help` text is written for the user, not the maintainer
  (`tests/test_cli_help.py` enforces the rules);
- add a line to `CHANGELOG.md` under `[Unreleased]`.

Commit messages follow the existing style: a `type(scope): summary` line
that says why, then detail if it helps.

## Adding anatomy

Meshes are BodyParts3D derivatives and must stay under `LICENSE-DATA`
(CC BY-SA 2.1 JP). Put STLs under `data/<tissue>/`, keep the original
BodyParts3D file identifiers in the file name, and never commit a mesh whose
provenance you cannot state.

## Code of conduct

Participation is governed by `CODE_OF_CONDUCT.md`.
