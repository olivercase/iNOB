# Quickstart

Fifteen minutes from a fresh clone to a calibrated leadfield and a
trials-to-detect number. Nothing here is a scientific configuration; it
proves the pipeline runs on your machine. Swap `configs/quickstart.yaml` for
`configs/default.yaml` when you want a result you would report.

## 1. Install (2 minutes)

```bash
git lfs install                       # meshes are Git LFS objects
git clone https://github.com/olivercase/iNOB.git && cd iNOB
python3 -m pip install -e .[dev]      # Python 3.11+
inob doctor                           # every check but DUNEuro should be ✓
```

## 2. Anatomy → FEM → sensors (about 1 minute, no DUNEuro)

```bash
inob run --config configs/quickstart.yaml --stages geom,fem,sensors
```

What you get, all under `outputs/quickstart/`:

| Stage | Output | What it is |
| --- | --- | --- |
| geom | `geometry/geometry.mat` | watertight skin + vagus surfaces from the BodyParts3D STLs |
| fem | `fem/fem.mat` | two-tissue tetrahedral mesh (≈52 k tets at 4 mm) |
| sensors | `sensors/sensor_array.mat` | triaxial OPM array on the skin, 50 mm pitch |

`inob status --config configs/quickstart.yaml` shows what is built and what
comes next. Draw the meshes with `inob visualise --config configs/quickstart.yaml`.

## 3. The forward solve (DUNEuro)

```bash
conda env create -f environment.yml && conda activate inob
bash scripts/build_duneuro_local.sh   # ~30 min, once
pip install -e .
inob run --config configs/quickstart.yaml          # MEG leadfield, fT per nA·m
inob eeg --config configs/quickstart.yaml          # EEG leadfield, µV per nA·m
```

No local build? The Docker image has one:

```bash
docker build -t inob:duneuro .
docker run --rm -v "$PWD/outputs:/work/outputs" -v "$PWD/data:/work/data" \
    inob:duneuro inob run --config configs/quickstart.yaml
```

## 4. Read the answer

```bash
inob detect --config configs/quickstart.yaml       # trials to detect each source
inob snr    --config configs/quickstart.yaml       # predicted SNR per source
inob topoplot --config configs/quickstart.yaml     # field map on the skin
```

`detect` combines the leadfield with the noise floor in `noise:` (an OPM
preset plus an EEG amplifier and electrode-skin term) and reports how many
averaged trials each dipole needs at the configured SNR threshold.

## 5. Check the physics

Two validations run against closed-form solutions and need no anatomy:

```bash
inob sarvas      # FEM vs the Sarvas single-sphere MEG solution
inob ladder      # Biot–Savart → Sarvas → FEM, rung by rung
```

`tests/test_meg_sphere_validation.py` and
`tests/test_venant_sphere_validation.py` run the same comparison in CI every
week with a real DUNEuro build.

## Next

- `inob --help` lists all 26 commands; `inob <command> --help` documents one.
- `configs/default.yaml` is the reference configuration, every field commented.
- `docs/REPRODUCE.md` regenerates the figures in the paper.
- `make gui` starts the browser interface.
