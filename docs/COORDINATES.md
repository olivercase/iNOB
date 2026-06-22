# Coordinate system

**Definitive reference for the BodyParts3D-derived FEM used by this pipeline.**

A reviewer-level question — and one that caused real confusion during
development — is: which way is anterior? This document fixes the convention
once and answers it from the data.

## Convention

The pipeline uses an **LPS-style** axis convention inherited from the
BodyParts3D source data (with an arbitrary origin shift in mm):

| axis | direction        | sign in our coords     |
| :--: | :--------------- | :-------------------- |
| **+X** | subject's left   | left vagus is at X ≈ +20 mm; right vagus at X ≈ –20 mm |
| **+Y** | posterior        | scapulae sit at Y ≈ –40 to 0; vagus at Y ≈ –75 to –115 mm (anterior to spine) |
| **+Z** | superior         | feet ≈ –78 mm; head crown ≈ +1642 mm |

**Anterior = more negative Y.** **Posterior = more positive Y.**

This is the only place where the convention is recorded. **Do not change it
without re-checking every figure** — at least one earlier audit got the
direction wrong.

## How we know

Three independent landmarks confirm +Y is posterior:

1. **Scapulae position.** At z = 1285 mm (upper chest), the bone tets form
   two lateral ovals at Y ∈ [–80, –10] (positive Y from the body centroid).
   Scapulae are anatomically *posterior*, so positive Y is posterior.
2. **Vagus relative to spine.** The cervical vagus runs in the carotid
   sheath, *anterior* to the cervical vertebrae. In our FEM the vagus is at
   Y ≈ –75 to –115 mm (more negative) and the bone (vertebrae + ribs) is at
   Y ≈ –90 to 0 mm (less negative). Vagus more negative ⇒ vagus more
   anterior ⇒ +Y is posterior.
3. **Skin-shape vs Z.** The skin centroid Y at the upper chest (z ≈ 1300)
   is at Y ≈ –60, vs Y ≈ –92 at lower z. The chest is wider anteriorly than
   the legs are; if +Y were anterior, the upper-chest centroid would be
   *more positive* than the legs, not more negative.

Together these are unambiguous.

## Practical consequence: tissue path from a cervical source to the front
## of the neck

Source #43 at (X, Y, Z) = (23, –104, 1285) mm. Casting rays from the source
in each cardinal direction and recording the first tissue change:

| direction      | what we hit |
| :------------- | :---------- |
| **anterior** (–Y) | vagus@0 → skin@7 mm → air@99 mm   *(no bone)* |
| **posterior** (+Y) | vagus@0 → skin@4 mm → bone@71 mm → skin@85 mm → air@140 mm |
| left lateral   | vagus@0 → skin@6 mm → multiple skin/air bands (arm) |
| right lateral  | vagus@0 → skin@4 mm → multiple skin/air bands (arm) |

**There is no bone between an anteriorly-placed cervical electrode and
the vagus.** The cervical vertebrae are ~70 mm posterior to the source —
irrelevant for an anteriorly-sited HD-EMG patch. This rules out
"bone shielding" as an explanation for low EEG amplitudes.

## Where this is used in the code

* `src/inob/sensors/electrodes.py` — cervical paddle is positioned over
  the vagus polyline projected to the *anterior* skin (more negative Y).
* `src/inob/viz/topoplot.py:_format_3d_axis` — viewing angles assume
  this convention (elev=14°, azim=42° gives a roughly anterolateral view).
* `src/inob/analysis/sarvas_compare.py:estimate_cervical_axis_xy` —
  sphere centre is biased toward the bone XY-centroid (i.e. posterior) so
  that "source-to-axis distance" reflects the anterior offset of the
  vagus.

## Reproducing the diagnostic

```bash
python3 - <<'PY'
import h5py, numpy as np
with h5py.File("outputs/fem/fem_vagus.mat", "r") as f:
    pos = f["pos"][:]; tet = f["tet"][:]; tissue = f["tissue"][:].ravel()
    labels = [bytes(np.asarray(s).tobytes()).rstrip(b"\x00").decode()
              for s in f["tissue_labels"][:]]
centroids = pos[tet].mean(axis=1)
slab = (centroids[:, 2] > 1280) & (centroids[:, 2] < 1290)
for tid, lbl in enumerate(labels, start=1):
    sel = (tissue == tid) & slab
    if sel.any():
        c = centroids[sel]
        print(f"  {lbl:<12}: x=[{c[:,0].min():.0f},{c[:,0].max():.0f}]  "
              f"y=[{c[:,1].min():.0f},{c[:,1].max():.0f}]")
PY
```
