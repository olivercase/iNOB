"""Factor levels and simulation constants for the precision x movement 2x2.

Port of ``cfa_2x2_design.m`` from the CFA replication pack, with the constants
that only the FEM build needs (compartment conductivities, source counts) added
alongside. Every module in :mod:`inob.cfa` reads its levels from here, so the
design cannot drift between the engine, the ANOVA, the bootstrap, the detection
curves and the rotation sweep.

FACTOR 1 -- PRECISION: how well the sensor array is characterised. Two error
channels are applied to the DATA while every estimator uses the NOMINAL array: a
per-channel gain error and a per-channel tilt of the sensitive axis. They are not
redundant. A gain error is a length change of the sensitive axis and can only
rescale what a channel already sees; a tilt mixes in field components the forward
model treats that channel as blind to, and it is tilt that caps the shielding
factor of harmonic interference models (Taulu & Simola 2006; Tierney et al.
2024). 3 deg is the order of the sensor-axis uncertainty of an array registered
by optical scan; 0.05 deg stands for one characterised against a reference coil
set.

FACTOR 2 -- MOVEMENT: head-orientation diversity across the recording. Low is a
still head; high is wide rotation whose per-beat orientation is tracked. Rotation
is multi-axis because yaw alone leaves the vertical component of the cardiac
leadfield untouched, so its orientation-average never approaches zero.

Beats are held constant across the movement factor (``N_BEATS`` in both levels,
split over that cell's orientations), so movement is not confounded with
recording length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Design", "DESIGN", "CONDUCTIVITY_SM"]


#: Compartment conductivities, S/m. The head values are the SimBio/DUNEuro
#: four-compartment defaults (Vorwerk et al. 2014); the torso values follow the
#: cardiac-modelling literature (Horacek 1973; Klepfer et al. 1997).
#:
#: ``bone`` is one label for the whole skeleton -- skull, ribs, sternum, spine.
#: 0.0042 S/m is compact-bone; the pack's torso BEM has no bone at all, and bone
#: is precisely where the cardiac return current does not go.
CONDUCTIVITY_SM: dict[str, float] = {
    "brain":       0.33,
    "insula":      0.33,     # same tissue as brain; a separate label only so the
                             # source can be placed by anatomy rather than by an
                             # MNI coordinate. Never given a different sigma.
    "csf":         1.79,
    "bone":        0.0042,
    #: "skin" is the label the mesher gives the BACKGROUND compartment -- every
    #: voxel inside the body that no organ claims. For a head-only model iNOB's
    #: 0.43 S/m (wet skin) is right, because there the label really is the scalp.
    #: Here it is the whole torso's generic soft tissue -- muscle, fat, connective
    #: tissue, viscera -- whose accepted value is 0.22-0.24 S/m (Horacek 1973;
    #: Klepfer et al. 1997; Gabriel et al. 1996). Using 0.43 nearly doubles the
    #: background conductivity of the entire body, which changes the volume-current
    #: return paths and hence how much cardiac field reaches the head array.
    "skin":        0.23,
    "muscle":      0.35,
    "heart":       0.62,     # blood-filled chambers dominate the whole-organ value
    "myocardium":  0.25,     # cardiac muscle, isotropic mean (Roth 1997)
    "lung_left":   0.05,
    "lung_right":  0.05,
    "spinal_cord": 0.23,
}


@dataclass(frozen=True)
class Design:
    """The 2x2 design. Frozen: a run may not mutate its own factor levels."""

    # --- Factor 1: precision -------------------------------------------------
    #: [gain SD (fractional), sensitive-axis tilt SD (deg)]
    prec_high: tuple[float, float] = (0.001, 0.05)
    prec_low: tuple[float, float] = (0.10, 3.00)

    # --- Factor 2: movement --------------------------------------------------
    #: Head still.
    ang_low: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 3), float))
    #: 8 tracked orientations, [yaw pitch roll] in degrees.
    ang_high: np.ndarray = field(default_factory=lambda: np.array([
        [-100.0, -12.0,   8.0],
        [ -70.0,  10.0,  -6.0],
        [ -45.0,  -8.0,  10.0],
        [ -15.0,  12.0,   5.0],
        [  15.0, -12.0,  -5.0],
        [  45.0,   8.0, -10.0],
        [  70.0, -10.0,   6.0],
        [ 100.0,  12.0,  -8.0],
    ]))

    #: (precision, movement) per cell, and the array each cell stands for.
    cells: tuple[tuple[str, str], ...] = (
        ("high", "low"), ("high", "high"), ("low", "low"), ("low", "high"))
    cell_labels: tuple[str, ...] = (
        "SQUID", "future OPM", "worst case", "current OPM")

    # --- Replication ---------------------------------------------------------
    n_rep: int = 16          # Monte-Carlo calibration draws per cell
    rng_seed: int = 42

    # --- Recording length ----------------------------------------------------
    n_beats: int = 500       # ~8.3 min at 60 bpm, identical across movement levels

    # --- Sensor noise --------------------------------------------------------
    #: i.i.d. Gaussian per channel per beat. The array is TRIAXIAL -- 85 helmet
    #: sites x 3 orthogonal magnetometers = 255 channels -- and 15 fT/sqrtHz is
    #: the conservative end of the triaxial OPM range (single-axis reaches
    #: 7-12; triaxial sensors trade some floor for the two extra axes). The HEF
    #: is a low-frequency evoked field, so the per-beat per-channel amplitude is
    #: density * sqrt(bandwidth). Noise is independent per channel including
    #: between the three axes of one site: they are separate readout channels.
    noise_density_ft_sqrthz: float = 15.0
    bandwidth_hz: float = 40.0

    # --- Software arm --------------------------------------------------------
    #: Order 1 is standard HFC -- a uniform field, 3 regressors -- which is what
    #: spm_opm_hfc implements. Orders 2-3 are the AMM-family extension and are
    #: where the replication pack stopped, because 15 regressors against 85
    #: single-axis channels is already a real cost. A triaxial array has 255
    #: channels, so order 5 (35 regressors) is cheap and well conditioned, and
    #: the near-field cardiac artefact is exactly the case where the
    #: higher-degree terms still carry energy. Each order is a separate arm.
    hfc_orders: tuple[int, ...] = (1, 2, 3, 5)
    lambda_reg: float = 0.05     # LCMV covariance shrinkage, fraction of trace/ns

    # --- Detection -----------------------------------------------------------
    snr_target: float = 3.0
    hr_bpm: float = 60.0

    # --- Cardiac source ------------------------------------------------------
    #: Number of dipoles distributed through the myocardial wall. The estimator's
    #: model stays 6-dof (two equivalent dipoles), so the distributed truth *is*
    #: the forward-model mismatch -- it replaces the pack's hand-set 10.2 mm
    #: position offset with a mismatch that has a physiological cause.
    n_myocardial_dipoles: int = 400
    #: Residual position error of the equivalent-dipole model, metres. Retained
    #: on top of the distributed source because a real template is registered to
    #: the torso surface and carries a position error of this order regardless of
    #: how well the source's shape is modelled.
    heart_offset_m: tuple[float, float, float] = (0.008, -0.004, 0.005)

    @property
    def noise_ft(self) -> float:
        """Per-channel per-beat white-noise SD, fT (~94.9)."""
        return self.noise_density_ft_sqrthz * np.sqrt(self.bandwidth_hz)

    def precision(self, level: str) -> tuple[float, float]:
        return {"high": self.prec_high, "low": self.prec_low}[level]

    def angles(self, level: str) -> np.ndarray:
        return {"high": self.ang_high, "low": self.ang_low}[level]

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def orientations(self) -> np.ndarray:
        """Every distinct head orientation the design needs, still first.

        The forward solve is per orientation and is by far the most expensive
        stage, so the engine asks for this union once rather than per cell.
        """
        return np.vstack([self.ang_low, self.ang_high])


DESIGN = Design()
