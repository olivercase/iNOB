# Cluster pipeline (UCL Myriad / Kathleen)

A profile-driven SGE submission flow. The same body scripts run on either
cluster; the only thing that changes is `CLUSTER_PROFILE`.

## One-time setup (per-user)

1. **SSH aliases.** Add entries for the clusters you use to `~/.ssh/config`
   (host names like `myriad`, `kathleen`).
2. **Create `~/.inob.env`** (optional) to override defaults without editing
   anything in this repo:
   ```bash
   # ~/.inob.env
   INOB_LOCAL_DIR=$HOME/code/Forward_Model_Vagus_Nerve
   INOB_REMOTE_HOST=myriad
   # INOB_REMOTE_BASE=$HOME/Scratch/inob   # default
   ```
   Defaults work for any cluster account: `${HOME}/Scratch/inob`.

## Workflow

### 1. Build inputs locally (run once or whenever inputs change)

```bash
make pipeline                   # builds geometry → FEM → sensors locally
```

### 2. Stage to the cluster

```bash
CLUSTER_PROFILE=myriad bash cluster/stage.sh
# or:
CLUSTER_PROFILE=kathleen bash cluster/stage.sh --dry-run   # preview only
```

This rsyncs `outputs/{fem,sensors}/*.mat`, the `inob` package, configs,
and the cluster scripts to `${HOME}/Scratch/inob` on the cluster.

### 3. Build DUNEuro on the cluster (one-time)

```bash
ssh myriad
cd ~/Scratch/inob/code

# Interactive compute node (Myriad shape):
qrsh -pe smp 4 -l mem=8G,h_rt=2:00:00 -now no
cd ~/Scratch/inob/code
CLUSTER_PROFILE=myriad bash cluster/build_duneuro.sh

# Or as a batch job:
CLUSTER_PROFILE=myriad bash cluster/submit.sh build
```

### 4. Submit the forward solve

```bash
CLUSTER_PROFILE=myriad bash cluster/submit.sh array
CLUSTER_PROFILE=myriad bash cluster/submit.sh reduce   # waits on vagus_fwd
```

### 5. Pull the leadfield back

```bash
scp myriad:~/Scratch/inob/duneuro_leadfield_vagus.npz outputs/forward/
```

## How profiles work

`cluster/profiles/<name>.env` declares cluster-shape variables (PE, walltimes,
memory, modules, BLAS thread fan-out, chunk fan-out). `cluster/lib.sh` sources
the chosen profile and exports the variables; `cluster/submit.sh` translates
them into `qsub` flags so each body script (`run_array.sh`, `run_build.sh`,
`run_reduce.sh`) is profile-agnostic.

To add a new cluster: drop a new `profiles/<name>.env`, then run with
`CLUSTER_PROFILE=<name>` — no body script changes needed.

## What changed vs the legacy layout

| Before | After |
|---|---|
| `submit_array.sh` + `submit_array_kathleen.sh` | one `run_array.sh` + profile |
| `submit_build.sh` + `submit_build_kathleen.sh` | one `run_build.sh` + profile |
| `submit_reduce.sh` + `submit_reduce_kathleen.sh` | one `run_reduce.sh` + profile |
| `stage.sh` + `stage_kathleen.sh` | one `stage.sh` + profile |
| `cluster/run_chunk.py` (duplicated logic) | `python -m inob.forward.chunk` |
| `cluster/reduce.py` (missing `source_pos`) | `python -m inob.forward.reduce` (schema-correct) |
| Hardcoded `rmgpohk` username, `~olivercase` path | `${HOME}/Scratch/...` + env var overrides |
