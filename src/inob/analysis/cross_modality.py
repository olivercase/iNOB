"""Cross-modality coupling between MEG and EEG forward solutions.

Both leadfields are computed from the *same* FEM and the *same* discrete
source space (one dipole position per cervical-vagus polyline node, three
orthogonal moments per position). That shared structure means the EEG
leadfield ``L_E`` and MEG leadfield ``L_M`` are coupled: any source moment
``q`` induces both ``V = L_E q`` and ``B = L_M q``.

Two questions:

  Q1.  *Given a measured EEG topo from a **known** source position, what
       should the MEG topo look like for the same source?*
       For a single source position with three unknown moment components,
       inverting the 32-channel EEG observation for ``q`` is a stable
       least-squares problem (32 ≫ 3). Plugging that ``q`` into ``L_M``
       gives a clean predicted MEG topo.

       This is *not* the inverse problem: localising the source in 3-D
       from the EEG alone is ill-posed and not what we do here. The
       construction assumes the source position has already been picked
       (e.g. anatomically, from MR-guided knowledge of the vagus polyline).

  Q2.  *Across all source positions, how correlated are the two
       modalities' amplitudes?*
       For each source, take the per-modality amplitude (RMS over moments
       and channels). High correlation ⇒ a strong EEG response implies a
       strong MEG response from the same location; a flat scatter ⇒ the two
       modalities probe different parts of the source space.

References
----------
* Sarvas J. (1987). "Basic mathematical and electromagnetic concepts of the
  biomagnetic inverse problem." Phys Med Biol 32:11.
  https://doi.org/10.1088/0031-9155/32/1/004
* Hämäläinen M, Hari R, Ilmoniemi RJ, Knuutila J, Lounasmaa OV (1993).
  "Magnetoencephalography — theory, instrumentation, and applications to
  noninvasive studies of the working human brain." Rev Mod Phys 65:413.
  https://doi.org/10.1103/RevModPhys.65.413
* Tierney TM, Holmes N, Mellor S, López JD, Roberts G, Hill RM, Boto E,
  Leggett J, Shah V, Brookes MJ, Bowtell R, Barnes GR (2019). "Optically
  pumped magnetometers: From quantum origins to multi-channel
  magnetoencephalography." NeuroImage 199:598-608.
  https://doi.org/10.1016/j.neuroimage.2019.05.063
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── per-source amplitudes ──────────────────────────────────────────────────

def per_source_amplitude(L: np.ndarray) -> np.ndarray:
    """RMS amplitude across (channels × moments) for each source position.

    Input shape (C, 3*S) → output shape (S,). Units match those of ``L``.
    """
    C, three_S = L.shape
    if three_S % 3 != 0:
        raise ValueError(f"L second dim {three_S} not divisible by 3")
    S = three_S // 3
    L3 = L.reshape(C, S, 3)
    return np.sqrt(np.mean(L3 ** 2, axis=(0, 2)))


# ── single-source forward prediction (Q1) ──────────────────────────────────

def fit_moment_from_eeg(
    L_eeg_src: np.ndarray, V_eeg_observed: np.ndarray,
) -> np.ndarray:
    """Least-squares fit a 3-moment dipole given a per-source EEG leadfield.

    Parameters
    ----------
    L_eeg_src       (C_e, 3) EEG leadfield columns for one source.
    V_eeg_observed  (C_e,)   measured surface potentials.
    """
    q, *_ = np.linalg.lstsq(L_eeg_src, V_eeg_observed, rcond=None)
    return q


def predict_meg_from_eeg(
    L_meg_src: np.ndarray, L_eeg_src: np.ndarray, V_eeg_observed: np.ndarray,
) -> np.ndarray:
    """Predict the MEG topo for a single source given its EEG observation.

    ``B_predicted = L_meg_src @ pinv_lsq(L_eeg_src) @ V_eeg_observed``.
    """
    q = fit_moment_from_eeg(L_eeg_src, V_eeg_observed)
    return L_meg_src @ q


def bootstrap_recovery_error_ci(
    L_meg_src: np.ndarray,
    L_eeg_src: np.ndarray,
    V_eeg_clean: np.ndarray,
    *,
    noise_uV: float,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI on the relative-RMS error of MEG-from-EEG recovery.

    For each bootstrap iteration we draw a fresh Gaussian noise realisation
    with std ``noise_uV``, add it to ``V_eeg_clean``, run the lstsq inversion
    + forward-prediction round trip, and record the relative RMS error
    against the FEM-truth ``B_meg_true = L_meg_src[:, 2]`` (the longitudinal
    moment column convention used by the cross-modality figure).

    Returns ``(low, median, high)`` of the relative error at confidence ``ci``.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(int(seed))
    B_true = L_meg_src[:, 2]
    B_true_rms = float(np.sqrt(np.mean(B_true ** 2)))
    if B_true_rms <= 0:
        return float("nan"), float("nan"), float("nan")
    errs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        V_obs = V_eeg_clean + rng.normal(0.0, float(noise_uV), V_eeg_clean.shape)
        q = fit_moment_from_eeg(L_eeg_src, V_obs)
        B_pred = L_meg_src @ q
        res = B_pred - B_true
        errs[b] = float(np.sqrt(np.mean(res ** 2))) / B_true_rms
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(errs, 100 * alpha)),
        float(np.median(errs)),
        float(np.percentile(errs, 100 * (1.0 - alpha))),
    )


# ── cross-modality summary (Q2) ────────────────────────────────────────────

@dataclass(frozen=True)
class CrossModalityStats:
    pearson_r: float
    spearman_rho: float
    log_log_slope: float    # slope of log(MEG_amp) vs log(EEG_amp), should be ~1 for ideal coupling
    n_sources: int


def amplitude_correlation(
    meg_L: np.ndarray, eeg_L: np.ndarray,
) -> CrossModalityStats:
    """Correlate per-source MEG and EEG amplitudes across the source space."""
    a_meg = per_source_amplitude(meg_L)
    a_eeg = per_source_amplitude(eeg_L)
    if len(a_meg) != len(a_eeg):
        raise ValueError(
            f"source-space size mismatch: MEG {len(a_meg)} vs EEG {len(a_eeg)}"
        )
    a_meg = np.maximum(a_meg, 1e-30)
    a_eeg = np.maximum(a_eeg, 1e-30)
    r = float(np.corrcoef(a_meg, a_eeg)[0, 1])
    rank_meg = a_meg.argsort().argsort()
    rank_eeg = a_eeg.argsort().argsort()
    rho = float(np.corrcoef(rank_meg, rank_eeg)[0, 1])
    log_meg = np.log(a_meg)
    log_eeg = np.log(a_eeg)
    slope, _ = np.polyfit(log_eeg, log_meg, 1)
    return CrossModalityStats(
        pearson_r=r, spearman_rho=rho, log_log_slope=float(slope),
        n_sources=len(a_meg),
    )


# ── singular-mode coupling (advanced) ──────────────────────────────────────

def shared_singular_modes(
    meg_L: np.ndarray, eeg_L: np.ndarray, *, n_modes: int = 6,
) -> dict[str, np.ndarray]:
    """Compute the dominant source-space singular modes for each modality.

    Each modality's leadfield ``L (C, 3S)`` has an SVD ``L = U Σ V^T``. The
    right-singular vectors ``V`` are 3S-dim source patterns. Comparing
    them across modalities tells you which spatial source patterns each
    modality is sensitive to. Equal modes ↔ identical sensitivity profiles.
    """
    _, sm, vm_T = np.linalg.svd(meg_L, full_matrices=False)
    _, se, ve_T = np.linalg.svd(eeg_L, full_matrices=False)
    n = min(n_modes, vm_T.shape[0], ve_T.shape[0])
    overlap = np.abs(vm_T[:n] @ ve_T[:n].T)         # |<v_i^MEG, v_j^EEG>|
    return {
        "meg_singular_values": sm[:n],
        "eeg_singular_values": se[:n],
        "mode_overlap": overlap,                     # (n, n) — diagonal ≈ 1 ⇒ same modes
    }
