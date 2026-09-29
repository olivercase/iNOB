"""Does the forward model earn its keep when choosing where sensors go?

The package's central claim is that picking sensor positions from a forward
model beats pointing them at the target by eye. That is an empirical claim
and this module tests it, because a reviewer is entitled to ask what the
whole pipeline buys over a tape measure.

Both arms are drawn from *one* solved leadfield over the full array, so the
comparison is a choice of subset and nothing else — same volume conductor,
same source, same noise floor, no second solve to disagree about:

``informed``
    the ``n_sensors`` positions with the largest predicted coupling to the
    target source. This is what iNOB is for.

``naive``
    the ``n_sensors`` positions closest to the source in straight-line
    distance, which is what you do with a tape measure and no model.
    ``naive_aim`` displaces the point it aims at, so the arm can be given
    the localisation error a real experimenter works with rather than
    perfect knowledge of where the source is.

Both are held to the same minimum spacing, so neither wins by crowding
sensors into a spot the other was not allowed to use.

What this measures, and what it does not
----------------------------------------
On the shipped anatomy the two arms come out within a few percent of each
other, because under a convex skin surface the best-coupled positions are
very nearly the nearest ones (Spearman rho 0.88-0.99 between coupling and
proximity across the three worked examples). That remains true out to a
50 mm aiming error. The result is a real property of the problem, not a
failure of the comparison, and it says something worth stating: for a
compact source under a triaxial array, the forward model is not what tells
you which sensors to use.

It is a narrow question by construction. Choosing *which region of the body*
to cover, and how far along an extended structure the array should sit, is a
different decision and one where coupling varies by more than an order of
magnitude; that is the sweep over ``source_idx``, not this comparison. Nor
does either arm say what amplitude or trial count to expect, which is what
the forward model supplies and a tape measure cannot.

The figure of merit is the multichannel matched-filter amplitude,
``sqrt(sum_c L_c^2)``, the optimal detection statistic for a known spatial
pattern in spatially white noise. Summing in quadrature (rather than taking
the best channel) is what makes the count of useful sensors matter, which is
the question being asked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlacementArm:
    """One array's worth of chosen positions and what it achieves."""

    name: str
    position_idx: np.ndarray  # (n,) indices into the array's unique positions
    amplitude: float  # matched-filter |L| over the chosen channels
    snr: float  # amplitude / noise floor, single trial
    trials_to_detect: float  # averages needed to reach the SNR threshold
    median_distance_mm: float  # chosen positions to the source


@dataclass(frozen=True)
class PlacementComparison:
    informed: PlacementArm
    naive: PlacementArm
    source_idx: int
    source_pos: np.ndarray  # (3,) mm
    n_sensors: int
    min_spacing_mm: float
    noise_floor: float
    snr_threshold: float

    @property
    def amplitude_ratio(self) -> float:
        """How much more signal the informed array sees."""
        return float(self.informed.amplitude / self.naive.amplitude)

    @property
    def trials_ratio(self) -> float:
        """How many times more averages the naive array needs.

        Trials go as 1/SNR^2, so this is the amplitude ratio squared — the
        number that matters when deciding whether a study is affordable.
        """
        return float(self.naive.trials_to_detect / self.informed.trials_to_detect)


def _greedy_spaced(order: np.ndarray, positions: np.ndarray, n: int, spacing: float) -> np.ndarray:
    """Take the first ``n`` of ``order`` that stay ``spacing`` apart.

    Walking a preference order and rejecting anything too close to an
    already-accepted position is the same rule for both arms, so the spacing
    constraint cannot favour either.
    """
    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= n:
            break
        if chosen and np.min(np.linalg.norm(positions[chosen] - positions[idx], axis=1)) < spacing:
            continue
        chosen.append(int(idx))
    if len(chosen) < n:
        logger.warning(
            "only %d of %d requested positions satisfy the %.0f mm spacing",
            len(chosen),
            n,
            spacing,
        )
    return np.asarray(chosen, dtype=int)


def compare_placement(
    L_human_units: np.ndarray,
    coil_pos: np.ndarray,
    source_pos: np.ndarray,
    *,
    source_idx: int,
    n_sensors: int = 20,
    min_spacing_mm: float = 30.0,
    noise_floor: float,
    snr_threshold: float = 3.0,
    naive_aim: np.ndarray | None = None,
) -> PlacementComparison:
    """Model-informed against naive sensor placement for one source.

    ``L_human_units`` is the (C, 3S) leadfield in fT/nA·m (or µV/nA·m), with
    the triaxial block layout this package writes: all N radial channels,
    then all N T1, then all N T2. Each of the N positions therefore
    contributes three channels, and selecting a position takes all three.

    ``naive_aim`` is the point the naive arm aims at; it defaults to the true
    source position, which is the most generous assumption available to that
    arm. Pass a displaced point to give it a realistic localisation error.
    Both arms are always *scored* against the true source.
    """
    if noise_floor <= 0.0:
        raise ValueError(f"noise_floor must be positive, got {noise_floor}")
    channels, three_s = L_human_units.shape
    if channels % 3 != 0:
        raise ValueError(f"channel count {channels} not divisible by 3 (needs R/T1/T2)")
    n_pos = channels // 3
    n_src = three_s // 3
    if not 0 <= source_idx < n_src:
        raise ValueError(f"source_idx {source_idx} outside 0..{n_src - 1}")
    if n_sensors < 1:
        raise ValueError(f"n_sensors must be >= 1, got {n_sensors}")
    if n_sensors > n_pos:
        raise ValueError(f"asked for {n_sensors} of only {n_pos} available positions")

    positions = np.asarray(coil_pos, dtype=float)[:n_pos]
    target = np.asarray(source_pos, dtype=float)[source_idx]

    # Best-aligned moment for this source, so neither arm is penalised by an
    # arbitrary choice of dipole direction.
    block = L_human_units[:, 3 * source_idx : 3 * source_idx + 3]  # (C, 3)
    per_moment = np.sqrt((block.reshape(3, n_pos, 3) ** 2).sum(axis=0))  # (n_pos, 3)
    moment = int(np.argmax((per_moment**2).sum(axis=0)))
    coupling = per_moment[:, moment]  # (n_pos,) field magnitude per position

    distance = np.linalg.norm(positions - target, axis=1)
    aim = target if naive_aim is None else np.asarray(naive_aim, dtype=float)
    aim_distance = np.linalg.norm(positions - aim, axis=1)

    arms: dict[str, PlacementArm] = {}
    for name, order in (
        ("informed", np.argsort(coupling)[::-1]),
        ("naive", np.argsort(aim_distance)),
    ):
        picked = _greedy_spaced(order, positions, n_sensors, min_spacing_mm)
        amplitude = float(np.sqrt((coupling[picked] ** 2).sum()))
        snr = amplitude / noise_floor
        trials = float((snr_threshold / snr) ** 2) if snr > 0 else np.inf
        arms[name] = PlacementArm(
            name=name,
            position_idx=picked,
            amplitude=amplitude,
            snr=snr,
            trials_to_detect=trials,
            median_distance_mm=float(np.median(distance[picked])),
        )

    result = PlacementComparison(
        informed=arms["informed"],
        naive=arms["naive"],
        source_idx=source_idx,
        source_pos=target,
        n_sensors=n_sensors,
        min_spacing_mm=min_spacing_mm,
        noise_floor=noise_floor,
        snr_threshold=snr_threshold,
    )
    logger.debug(
        "placement @ source %d: informed %.1f (SNR %.2f, %.0f trials) vs "
        "naive %.1f (SNR %.2f, %.0f trials) — %.2fx amplitude, %.2fx trials",
        source_idx,
        result.informed.amplitude,
        result.informed.snr,
        result.informed.trials_to_detect,
        result.naive.amplitude,
        result.naive.snr,
        result.naive.trials_to_detect,
        result.amplitude_ratio,
        result.trials_ratio,
    )
    return result


def sweep_sensor_counts(
    L_human_units: np.ndarray,
    coil_pos: np.ndarray,
    source_pos: np.ndarray,
    *,
    source_idx: int,
    counts: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
    min_spacing_mm: float = 30.0,
    noise_floor: float,
    snr_threshold: float = 3.0,
) -> list[PlacementComparison]:
    """Run :func:`compare_placement` across array sizes.

    The gap between the two arms is not constant in the number of sensors: a
    naive patch catches up once it is large enough to cover the positions the
    model would have picked. Where that happens is the useful output, because
    it says how big an array has to be before the model stops mattering.
    """
    return [
        compare_placement(
            L_human_units,
            coil_pos,
            source_pos,
            source_idx=source_idx,
            n_sensors=n,
            min_spacing_mm=min_spacing_mm,
            noise_floor=noise_floor,
            snr_threshold=snr_threshold,
        )
        for n in counts
        if n <= L_human_units.shape[0] // 3
    ]


def sweep_aiming_error(
    L_human_units: np.ndarray,
    coil_pos: np.ndarray,
    source_pos: np.ndarray,
    *,
    source_idx: int,
    errors_mm: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 50.0),
    n_draws: int = 200,
    n_sensors: int = 20,
    min_spacing_mm: float = 30.0,
    noise_floor: float,
    snr_threshold: float = 3.0,
    seed: int = 0,
) -> dict[float, float]:
    """Mean informed/naive amplitude ratio against the naive arm's aiming error.

    The zero-error entry is the most favourable case for the naive arm — it
    is told exactly where the source is. Anything above 1.0 at that point is
    what the forward model buys a perfectly-informed experimenter; the
    remaining entries say how fast that changes when they are guessing.

    Displacements are drawn isotropically, so the result averages over
    aiming in a helpful direction and an unhelpful one.
    """
    rng = np.random.default_rng(seed)
    target = np.asarray(source_pos, dtype=float)[source_idx]
    out: dict[float, float] = {}
    for err in errors_mm:
        ratios = []
        for _ in range(n_draws if err > 0.0 else 1):
            offset = rng.standard_normal(3)
            offset *= err / np.linalg.norm(offset)
            ratios.append(
                compare_placement(
                    L_human_units,
                    coil_pos,
                    source_pos,
                    source_idx=source_idx,
                    n_sensors=n_sensors,
                    min_spacing_mm=min_spacing_mm,
                    noise_floor=noise_floor,
                    snr_threshold=snr_threshold,
                    naive_aim=target + offset,
                ).amplitude_ratio
            )
        out[float(err)] = float(np.mean(ratios))
    return out


def format_sweep_table(comparisons: list[PlacementComparison]) -> str:
    """Render a sensor-count sweep as a table for the terminal.

    "What could I do with 64 sensors?" is the question an investigator
    actually arrives with, and it wants a number rather than a curve.
    """
    if not comparisons:
        return "(no array sizes to report)"
    head = comparisons[0]
    lines = [
        f"Source {head.source_idx} at "
        f"({head.source_pos[0]:.0f}, {head.source_pos[1]:.0f}, {head.source_pos[2]:.0f}) mm, "
        f"noise floor {head.noise_floor:.0f}, SNR threshold {head.snr_threshold:g}, "
        f"sensors at least {head.min_spacing_mm:.0f} mm apart",
        "",
        f"{'sensors':>8}  {'informed':>10}  {'trials':>8}  "
        f"{'naive':>10}  {'trials':>8}  {'gain':>6}",
    ]
    for comparison in comparisons:
        lines.append(
            f"{comparison.n_sensors:>8}  "
            f"{comparison.informed.amplitude:>10.1f}  "
            f"{comparison.informed.trials_to_detect:>8.1f}  "
            f"{comparison.naive.amplitude:>10.1f}  "
            f"{comparison.naive.trials_to_detect:>8.1f}  "
            f"{comparison.amplitude_ratio:>6.2f}x"
        )
    lines += [
        "",
        "Amplitudes are matched-filter fT per nA·m; trials are averages to",
        "reach the threshold at a 1 nA·m source. 'gain' is informed/naive.",
    ]
    return "\n".join(lines)


# ── array optimisation over a whole target region ──────────────────────────
#
# The comparison above asks which sensors to use for one source, and finds
# that proximity already answers it. Designing an array for a *structure* is a
# different question: the positions have to serve every source along the
# nerve or cord at once, and what one more sensor adds depends on what the
# others already see. Total information capacity (Schneiderman 2014;
# Iivanainen et al. 2021) is the objective that captures both, and selecting
# it greedily is the farthest-point strategy of Roos, Hämäläinen & Iivanainen
# (Aalto; cerebellar OPM arrays, BIOMAG 2026).


@dataclass(frozen=True)
class OptimisedArray:
    """A chosen subset of positions and the information each step added."""

    position_idx: np.ndarray  # (n,) indices into ``positions``, in pick order
    positions: np.ndarray  # (P, 3) mm, every candidate position
    channel_idx: np.ndarray  # channels of the chosen positions, in pick order
    info_bits: np.ndarray  # (n,) total information after each pick
    full_info_bits: float  # the whole candidate array, for scale
    n_sources: int
    moment_nAm: float
    noise_floor: float


def position_groups(coil_pos: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unique sensor positions and the channels that sit at each.

    Grouping by position rather than assuming three channels per sensor lets
    the same optimiser choose from a triaxial, single-axis or imported array.
    """
    rounded = np.round(np.asarray(coil_pos, dtype=float), 6)
    uniq, first, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)  # keep the file's order of positions
    groups = [np.flatnonzero(inverse.ravel() == k) for k in order]
    return uniq[order], groups


def _logdet_spd(M: np.ndarray) -> float:
    """log det of a symmetric positive-definite matrix, by Cholesky.

    Every matrix here is I plus a Gram matrix, so its eigenvalues are at least
    one. Cholesky is exact for that case, and unlike slogdet's LU it does not
    raise spurious divide-by-zero warnings under macOS Accelerate.
    """
    return float(2.0 * np.log(np.diag(np.linalg.cholesky(M))).sum())


def _total_info_bits(X: np.ndarray) -> float:
    """½ log2 det(I + XᵀX) for an SNR-scaled leadfield X (channels, 3S)."""
    return 0.5 * _logdet_spd(np.eye(X.shape[1]) + X.T @ X) / np.log(2.0)


def optimise_array(
    L_human_units: np.ndarray,
    coil_pos: np.ndarray,
    *,
    n_sensors: int,
    noise_floor: float,
    moment_nAm: float,
    source_idx: np.ndarray | None = None,
    min_spacing_mm: float = 30.0,
) -> OptimisedArray:
    """Choose ``n_sensors`` positions that maximise total information.

    Each source contributes three orthogonal moments of variance
    ``moment_nAm²``, noise is white at ``noise_floor`` per channel, and the
    information of a channel set S is ``½ log2 det(I + L_S C L_Sᵀ / σ²)``.
    Positions are added one at a time, each the one that adds the most given
    those already chosen — so a second sensor over the same patch of field is
    worth less than the first, and coverage of the rest of the structure wins
    once it is. The determinant is kept in source space and updated by
    Woodbury, so each step is one matrix product over the candidates.

    ``source_idx`` restricts the region served (default: every source in the
    leadfield). The spacing rule is the one :func:`compare_placement` uses.
    """
    if noise_floor <= 0.0 or moment_nAm <= 0.0:
        raise ValueError("noise_floor and moment_nAm must be positive")
    positions, groups = position_groups(coil_pos)
    if not 1 <= n_sensors <= len(positions):
        raise ValueError(f"n_sensors must be in 1..{len(positions)}, got {n_sensors}")
    cols = np.arange(L_human_units.shape[1]).reshape(-1, 3)
    if source_idx is not None:
        cols = cols[np.asarray(source_idx, dtype=int)]
    X = np.asarray(L_human_units, dtype=float)[:, cols.ravel()] * (moment_nAm / noise_floor)

    blocks = [X[g] for g in groups]  # (channels at position, 3S)
    Minv = np.eye(X.shape[1])
    available = np.ones(len(positions), dtype=bool)
    chosen: list[int] = []
    info: list[float] = []
    total = 0.0
    for _ in range(n_sensors):
        best, best_gain = -1, -np.inf
        for p in np.flatnonzero(available):
            B = blocks[p]
            gain = _logdet_spd(np.eye(len(B)) + B @ Minv @ B.T)
            if gain > best_gain:
                best, best_gain = int(p), gain
        if best < 0:
            logger.warning(
                "only %d of %d positions satisfy the %.0f mm spacing",
                len(chosen),
                n_sensors,
                min_spacing_mm,
            )
            break
        B = blocks[best]
        BM = B @ Minv
        Minv = Minv - BM.T @ np.linalg.solve(np.eye(len(B)) + BM @ B.T, BM)
        total += 0.5 * best_gain / np.log(2.0)
        chosen.append(best)
        info.append(total)
        available &= np.linalg.norm(positions - positions[best], axis=1) >= min_spacing_mm

    idx = np.asarray(chosen, dtype=int)
    return OptimisedArray(
        position_idx=idx,
        positions=positions,
        channel_idx=np.concatenate([groups[p] for p in idx]) if len(idx) else idx,
        info_bits=np.asarray(info),
        full_info_bits=_total_info_bits(X),
        n_sources=len(cols),
        moment_nAm=float(moment_nAm),
        noise_floor=float(noise_floor),
    )


def nearest_array_info(
    L_human_units: np.ndarray,
    coil_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    n_sensors: int,
    noise_floor: float,
    moment_nAm: float,
    source_idx: np.ndarray | None = None,
    min_spacing_mm: float = 30.0,
) -> float:
    """Information of the tape-measure array: nearest to the structure, spaced.

    The baseline an optimised array has to beat, built the way
    :func:`compare_placement`'s naive arm is but aimed at the whole structure:
    positions ranked by distance to the nearest target source.
    """
    positions, groups = position_groups(coil_pos)
    d = np.min(np.linalg.norm(positions[:, None, :] - target_pos[None, :, :], axis=2), axis=1)
    picked = _greedy_spaced(np.argsort(d), positions, n_sensors, min_spacing_mm)
    cols = np.arange(L_human_units.shape[1]).reshape(-1, 3)
    if source_idx is not None:
        cols = cols[np.asarray(source_idx, dtype=int)]
    X = np.asarray(L_human_units, dtype=float)[:, cols.ravel()] * (moment_nAm / noise_floor)
    return _total_info_bits(X[np.concatenate([groups[p] for p in picked])])
