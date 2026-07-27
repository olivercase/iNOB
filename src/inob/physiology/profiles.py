"""Per-target physiology profiles.

The forward model is target-agnostic — same FEM, same solver, same leadfield
units — but the *physiology* driving the sources is not. A vagal baroreceptor
burst and a spinal somatosensory volley differ in fibre population, conduction
velocity, action-potential width, where the activity originates, how far it
propagates, and what experimental paradigm evokes it. This module makes those
differences explicit and selectable, so a figure titled "spine" is generated
from spinal physiology rather than from vagus physiology with a relabelled axis.

A :class:`PhysiologyProfile` is looked up from the ``--source-target`` slug via
:func:`profile_for_tag`, which every time-domain figure goes through.

Profile status
--------------
``vagus`` and ``spine`` are modelled from their own literature.
``muscle`` remains **provisional**: it has an accurate static forward model
(fibre-aligned anisotropy) and its single-event fibre/CV/amplitude parameters
are now muscle-specific (not borrowed from vagus), but it still has no
MUAP/motor-unit/recruitment model, so it cannot represent voluntary
interference EMG or build an event-train scenario. See the PHYSIOLOGY-TODO in
:mod:`inob.physiology.scenarios`.

References
----------
* Hämäläinen M *et al.* 1993 *Rev Mod Phys* 65:413 — Q = π·d²·σ_in·ΔV/4.
* Bu Y *et al.* 2024 *Comm Biol* 7:893 — cervical-vagus baroreceptor OPM recording.
* Pelot NA *et al.* 2017 *Front Neurosci* 12:601 — vagal fibre populations.
* Cruccu G *et al.* 2008 *Clin Neurophysiol* 119:1705 — SSEP recording standards
  (stimulation rates, cervical N13 generator, montages).
* Desmedt JE & Cheron G 1980 *Electroencephalogr Clin Neurophysiol* 50:382 —
  cervical N13 and the ascending dorsal-column volley.
* Kawabata S *et al.* 2002 *Clin Neurophysiol* 113:1874; Sasaki S *et al.* 2008
  *Spine* 33:E836 — magnetospinography of the propagating cord volley, from
  which the ~55–70 m/s conduction velocity and single-nA·m equivalent current
  dipole used below are taken.
* Lexell J, Taylor CC & Sjöström M 1988 *J Neurol Sci* 84:275 — human skeletal
  muscle fibre diameter (Type I/II, ~40–80 µm).
* McComas AJ 1977 *Neuromuscular Function and Disorders* — muscle-fibre
  intracellular AP amplitude and duration.
* Farina D & Merletti R 2004 *IEEE Rev Biomed Eng* — single-fibre and MUAP
  conduction velocity (skeletal-muscle fibre CV ≈ 3–5 m/s, weakly diameter-
  dependent, unlike saltatory conduction in myelinated nerve).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from inob.sources.cap import (
    FibreDistribution,
    conduction_velocity_m_per_s,
    hamalainen_per_fibre_nAm,
    lognormal_fibre_distribution,
)

logger = logging.getLogger(__name__)


# ── fibre populations ──────────────────────────────────────────────────────

def a_fibre_population(
    *, mean_um: float = 8.0, sigma_log: float = 0.30, n_bins: int = 20,
) -> FibreDistribution:
    """Vagal A-myelinated afferents (mean d ≈ 8 µm, mean CV ≈ 47 m/s).

    Re-exported here so both profiles pull their populations from one place;
    :mod:`inob.physiology.scenarios` keeps the original name for compatibility.
    """
    return lognormal_fibre_distribution(
        mean_um=mean_um, sigma_log=sigma_log, n_bins=n_bins,
        lo_um=2.0, hi_um=15.0,
    )


def dorsal_column_population(
    *, mean_um: float = 10.0, sigma_log: float = 0.25, n_bins: int = 20,
) -> FibreDistribution:
    """Dorsal-column ascending afferents (mean d ≈ 10 µm, mean CV ≈ 59 m/s).

    The cuneate/gracile fasciculi carry large-myelinated Aβ cutaneous and
    proprioceptive afferents — coarser and faster than the vagal A-fibre
    population, and it is their fastest components that dominate the evoked
    volley. The resulting mean CV of ≈ 59 m/s sits in the 55–70 m/s range
    measured for the human cervical cord by magnetospinography (Kawabata 2002,
    Sasaki 2008), which is the observable this population is tuned to match.
    """
    return lognormal_fibre_distribution(
        mean_um=mean_um, sigma_log=sigma_log, n_bins=n_bins,
        lo_um=3.0, hi_um=16.0,
    )


def muscle_fibre_population(
    *, mean_um: float = 60.0, sigma_log: float = 0.20, n_bins: int = 20,
) -> FibreDistribution:
    """Skeletal-muscle fibres (mean d ≈ 60 µm, range 40–80 µm).

    An order of magnitude coarser than any myelinated-axon population above:
    these are the sarcolemmal fibres themselves, not axons. Diameter range
    per Lexell et al. 1988 (human vastus lateralis, Type I/II fibres).
    Conduction velocity does *not* follow the myelinated-axon CV≈k·D law —
    see :data:`MUSCLE_PROFILE`'s ``cv_kwargs``.
    """
    return lognormal_fibre_distribution(
        mean_um=mean_um, sigma_log=sigma_log, n_bins=n_bins,
        lo_um=40.0, hi_um=80.0,
    )


# ── profile ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhysiologyProfile:
    """Everything about a target that is physiology rather than geometry."""

    name: str
    label: str
    validated: bool
    """False marks output that must carry a 'provisional physiology' banner."""

    fibres: FibreDistribution
    ap_width_ms: float
    ap_amplitude_mV: float
    sigma_in_Sm: float
    n_fibres: int

    default_strength_nAm: float
    """Default per-source strength for the trials-to-detect question, nA·m.

    Deliberately *not* the same quantity as :attr:`event_moment_nAm`. The event
    moment is what one modelled synchronous event produces; this is the source
    strength the planning question should assume, which for a given target may
    be a different, better-attested figure from the literature. Keeping them
    separate stops a change to the fibre model from silently rescaling every
    detectability result.
    """

    propagation_span_mm: float | None
    """Arc length over which one event's activity travels.

    ``None`` means the whole modelled polyline. This is the single most
    important physiological difference between the targets: a vagal
    baroreceptor burst is generated by afferents localised to the cervical
    bundle, whereas a spinal volley enters at one segment and ascends the
    entire imaged cord.
    """

    stationary_ok: bool
    """Whether lumping the event into one stationary dipole is defensible.

    True when transit time over :attr:`propagation_span_mm` is comparable to
    :attr:`ap_width_ms`; False when the wave sweeps far enough that the
    stationary approximation misrepresents both amplitude and time course.
    """

    generator: str
    """Human-readable description of where the activity originates."""

    paradigm: str
    """The experimental paradigm this physiology corresponds to."""

    notes: str = ""
    scenario_builder: Callable[..., tuple] | None = field(
        default=None, repr=False, compare=False,
    )
    cv_kwargs: dict | None = None
    """Override for :func:`inob.sources.cap.conduction_velocity_m_per_s`.

    ``None`` uses that function's defaults — the myelinated-axon law
    CV ≈ k·D (k = 6 m/s/µm), correct for nerve (vagus, spine). Muscle fibres
    conduct via continuous sarcolemmal excitation, not saltatory conduction,
    so CV is roughly diameter-independent at ≈ 3–5 m/s; that law does not
    apply and must be overridden.
    """

    # ── derived quantities ────────────────────────────────────────────────

    @property
    def mean_cv_m_per_s(self) -> float:
        """Fibre-population-weighted mean conduction velocity."""
        cv = conduction_velocity_m_per_s(
            self.fibres.diameters_um, **(self.cv_kwargs or {}),
        )
        return float(np.sum(cv * self.fibres.weights))

    @property
    def per_fibre_nAm(self) -> float:
        """Population-mean Hämäläinen dipole moment for one fibre."""
        return hamalainen_per_fibre_nAm(
            self.fibres,
            action_potential_mV=self.ap_amplitude_mV,
            sigma_intracellular_S_per_m=self.sigma_in_Sm,
        )

    @property
    def event_moment_nAm(self) -> float:
        """Total dipole moment of one modelled synchronous event, nA·m.

        Derived from the fibre model (n_fibres × per-fibre Hämäläinen moment).
        For the *planning* default used by trials-to-detect see
        :attr:`default_strength_nAm`, which is a separate, independently
        sourced figure.
        """
        return self.per_fibre_nAm * self.n_fibres

    def transit_ms(self, span_mm: float) -> float:
        """Time for the mean-CV wavefront to cross ``span_mm``."""
        return span_mm * 1e-3 / max(self.mean_cv_m_per_s, 1e-12) * 1000.0

    def scenarios(self, **kwargs) -> tuple:
        """Build this target's event-train scenarios."""
        if self.scenario_builder is None:
            raise NotImplementedError(
                f"physiology profile {self.name!r} defines no event-train "
                "scenarios; only single-event (cap_compare) output is available."
            )
        return self.scenario_builder(**kwargs)

    def describe(self) -> str:
        """One-block summary for figure panels and logs."""
        span = ("whole polyline" if self.propagation_span_mm is None
                else f"{self.propagation_span_mm:.0f} mm")
        return (
            f"{self.label} physiology — {self.paradigm}\n"
            f"Generator: {self.generator}\n"
            f"Fibres: n={self.n_fibres}, mean CV {self.mean_cv_m_per_s:.0f} m/s, "
            f"AP width {self.ap_width_ms:.2f} ms\n"
            f"Propagation span: {span}\n"
            f"Event moment: {self.event_moment_nAm:.2f} nA·m"
        )


# ── the profiles ───────────────────────────────────────────────────────────

def _scenario_pair(*prefixed: tuple[str, str]):
    """Build a ``scenario_builder`` from ``(prefix, scenario-function name)`` pairs.

    Callers pass one flat kwargs dict for all of a profile's scenarios and
    route each keyword by prefix — ``baro_duration_s=6.0`` reaches
    ``baroreceptor_scenario(duration_s=6.0)``.
    """
    def build(**kwargs) -> tuple:
        from inob.physiology import scenarios as _sc
        out = []
        for prefix, fn_name in prefixed:
            sub = {k[len(prefix) + 1:]: v for k, v in kwargs.items()
                   if k.startswith(f"{prefix}_")}
            out.append(getattr(_sc, fn_name)(**sub))
        return tuple(out)
    return build


_vagus_scenarios = _scenario_pair(
    ("baro", "baroreceptor_scenario"), ("resp", "respiratory_scenario"),
)
_spine_scenarios = _scenario_pair(
    ("median", "median_nerve_ssep_scenario"), ("tibial", "tibial_nerve_ssep_scenario"),
)


VAGUS_PROFILE = PhysiologyProfile(
    name="vagus",
    label="vagus",
    validated=True,
    fibres=a_fibre_population(),
    ap_width_ms=0.5,
    ap_amplitude_mV=70.0,
    sigma_in_Sm=1.0,
    n_fibres=200,
    # Unchanged from the package's long-standing default: the ~70 nA.m
    # cervical-vagus reference for full A+C fibre summation (Bu et al. 2024,
    # derived with Hamalainen sigma_in = 1 S/m). This is a larger, better-
    # attested quantity than one 200-fibre baroreceptor burst (~0.74 nA.m),
    # which is why the two are separate fields.
    default_strength_nAm=70.0,
    # Carotid-sinus baroreceptor afferents are anatomically localised to the
    # cervical bundle; they do not fire along the whole 500 mm nerve at once.
    propagation_span_mm=50.0,
    # 50 mm at 47 m/s ≈ 1.1 ms, comparable to the 0.5 ms AP width, so lumping
    # the event at the cervical hot-spot is defensible. This is the claim the
    # cap_compare figure exists to verify.
    stationary_ok=True,
    generator="carotid-sinus baroreceptor afferents, cervical bundle",
    paradigm="spontaneous cardiac- and respiratory-locked afferent traffic",
    notes=(
        "200 synchronous A-fibres per cardiac burst follows the Bu et al. 2024 "
        "cervical-vagus baroreceptor estimate."
    ),
    scenario_builder=_vagus_scenarios,
)


SPINE_PROFILE = PhysiologyProfile(
    name="spine",
    label="spine",
    validated=True,
    fibres=dorsal_column_population(),
    # The cervical cord evoked response (N13 and its magnetic counterpart) has
    # a component duration around 1 ms — broader than a single peripheral-nerve
    # AP because the ascending volley is already dispersed across CVs.
    ap_width_ms=0.7,
    # Large myelinated axons; intracellular AP amplitude toward the top of the
    # 70-100 mV range rather than the 70 mV used for the finer vagal fibres.
    ap_amplitude_mV=80.0,
    sigma_in_Sm=1.0,
    # NOTE: this is a *calibration* count, not an anatomical one. Supramaximal
    # median-nerve stimulation recruits O(10^4) myelinated afferents, but the
    # naive product (all fibres x full intracellular AP) overestimates the
    # measured equivalent current dipole by one to two orders of magnitude:
    # the volley disperses across conduction velocities and opposing membrane
    # currents partially cancel. 800 fibres reproduces an event moment of
    # ~5 nA.m, the single-nA.m range reported for cervical cord responses by
    # magnetospinography (Kawabata 2002, Sasaki 2008), which is the quantity
    # the forward model actually needs.
    n_fibres=800,
    # For the spine the modelled event and the planning default coincide: the
    # SSEP volley IS the thing being detected, and its ~5 nA.m moment is the
    # magnetospinography-derived figure (see n_fibres above).
    default_strength_nAm=5.11,
    # The defining difference from the vagus: the volley enters at one spinal
    # segment and ascends the entire imaged cord. There is no localised
    # generator to lump it into.
    propagation_span_mm=None,
    # ~450 mm of modelled cord at 59 m/s is a 7.6 ms transit — an order of
    # magnitude longer than the 0.7 ms AP width. The stationary approximation
    # that is fine for the vagus is simply wrong here.
    stationary_ok=False,
    generator="dorsal-column ascending volley from the stimulated segment",
    paradigm="somatosensory evoked potentials/fields (peripheral nerve stimulation)",
    notes=(
        "Evoked rather than spontaneous: the relevant planning question is "
        "trials-to-detect for an averaged SSEP, which is exactly the paradigm "
        "clinical SSEP and magnetospinography already use (500-2000 averages)."
    ),
    scenario_builder=_spine_scenarios,
)


MUSCLE_PROFILE = PhysiologyProfile(
    name="muscle",
    label="muscle",
    # Static forward model is sound (fibre-aligned anisotropy) and the
    # single-event fibre/CV/amplitude parameters below are now muscle-
    # specific. Still provisional: there is no MUAP/motor-unit/recruitment
    # model, so this can only represent one synchronous fibre volley (the
    # evoked-M-wave case), not voluntary interference EMG, and no
    # event-train scenario can be built. See PHYSIOLOGY-TODO in scenarios.py.
    validated=False,
    fibres=muscle_fibre_population(),
    # Single muscle-fibre intracellular AP duration is longer than a nerve
    # AP (continuous sarcolemmal excitation vs. saltatory conduction);
    # ap_width_ms is the sigma of the biphasic shape, so 1.5 ms gives a
    # FWHM of a few ms, consistent with single-fibre AP durations reported
    # in surface/needle EMG (Farina & Merletti 2004).
    ap_width_ms=1.5,
    # Skeletal-muscle fibre AP amplitude (McComas 1977), slightly larger
    # overshoot than the vagal/spinal axon figures used above.
    ap_amplitude_mV=90.0,
    sigma_in_Sm=1.0,
    # Calibration count (see SPINE_PROFILE.n_fibres for the same convention):
    # chosen so the modelled event's total moment lands on the "single MUAP"
    # Q ~= 10 nA.m estimate in inob.viz.detectability.MUSCLE_SCENARIOS,
    # which cites the same Hamalainen formula at fibre-population scale
    # (Cohen & Givler 1972). This also reconciles the two figures' magnitude
    # assumptions, which previously used different d/sigma_in/AP-amplitude.
    n_fibres=40,
    # Matches the "evoked compound M-wave" scenario in MUSCLE_SCENARIOS —
    # the closest muscle analogue to the spine profile's "the modelled event
    # IS the thing being detected" (a synchronous, stimulus-locked volley
    # rather than spontaneous/voluntary activity, which is not modelled).
    default_strength_nAm=1000.0,
    # NOTE: unlike vagus/spine, muscle source positions are a 3-D volume-fill
    # of the whole muscle belly, not a 1-D ordered path — "arc length along
    # the source list" is not a real anatomical distance here the way it is
    # for a nerve trunk or cord. Using ``None`` ("whole polyline", as for
    # spine) is unsafe: the cumulative arc length over an unordered point
    # cloud is enormous and physically meaningless (~tens of metres for a
    # ~300 mm muscle), which blows up cap_compare's transit-time window to
    # tens of seconds of modelled signal. Bounding the span keeps the figure
    # computable; it does not make the propagating-wavefront model
    # physically correct for muscle (see PHYSIOLOGY-TODO — the source
    # geometry itself needs a fibre-ordered path, not volume-fill points,
    # before propagation along "arc length" means anything for this target).
    propagation_span_mm=50.0,
    # At CV ~= 4 m/s, 50 mm gives a 12.5 ms transit — far longer than the
    # 1.5 ms AP width — so the stationary lumped-dipole approximation used
    # for vagus (transit ~= AP width, 50 mm at ~47 m/s ~= 1 ms) is not
    # defensible here even at this bounded span.
    stationary_ok=False,
    # Muscle-fibre CV is set by sarcolemmal membrane kinetics, not axon
    # diameter, and is roughly constant across the fibre population
    # (Farina & Merletti 2004, ~3-5 m/s) — the opposite of the myelinated
    # CV~=k*D law used for vagus/spine. Push every fibre diameter below the
    # "myelinated" threshold so conduction_velocity_m_per_s returns the flat
    # c_unmyelinated rate instead of extrapolating the nerve law to a 60 um
    # fibre (which would wrongly imply CV ~= 360 m/s).
    cv_kwargs={"myelinated_threshold_um": 1000.0, "c_unmyelinated": 4.0},
    generator="motor-endplate junction, propagating bidirectionally to tendons",
    paradigm="PROVISIONAL — single synchronous fibre volley (evoked-M-wave "
             "analogue); no MUAP/motor-unit/recruitment model",
    notes=(
        "Fibre diameter, AP amplitude, and conduction velocity (~4 m/s, "
        "diameter-independent) are now muscle-specific rather than reused "
        "from vagus. Still missing: motor-unit/MUAP structure, recruitment "
        "and rate-coding (size principle, 8-30 Hz), and the resulting "
        "asynchronous interference-EMG waveform for voluntary contraction — "
        "only a single synchronous evoked volley is representable."
    ),
    scenario_builder=None,
)


PROFILES: dict[str, PhysiologyProfile] = {
    "vagus": VAGUS_PROFILE,
    "spine": SPINE_PROFILE,
    "spine_vagus": SPINE_PROFILE,
    "muscle": MUSCLE_PROFILE,
    # Same gap as spine_vagus -> SPINE_PROFILE above: a combined forward solve
    # (spinal_cord + muscle tissue) still only gets spine's time-domain
    # physiology for any figure. No combined-target physiology model exists;
    # this is a placeholder that keeps every SOURCE_TARGET resolvable rather
    # than silently falling back to vagus.
    "spine_muscle": SPINE_PROFILE,
}


def profile_for_tag(tag: str) -> PhysiologyProfile:
    """Physiology profile for a ``--source-target`` slug.

    An empty tag (an untagged legacy ``duneuro_leadfield.npz``) falls back to
    the vagus profile, matching the historical behaviour of the package. An
    unrecognised tag also falls back to vagus but is marked unvalidated, so the
    figure says so rather than quietly asserting physiology it does not have.
    """
    if not tag:
        return VAGUS_PROFILE
    if tag in PROFILES:
        return PROFILES[tag]
    logger.warning(
        "no physiology profile for source-target %r; falling back to vagus "
        "physiology and marking the output provisional", tag,
    )
    from dataclasses import replace
    return replace(
        VAGUS_PROFILE, name=tag, label=tag.replace("_", " + "), validated=False,
        generator=f"PROVISIONAL — no profile for {tag!r}; reusing vagus",
        paradigm="PROVISIONAL — physiology not modelled for this target",
    )


def profile_for(cfg) -> PhysiologyProfile:
    """Physiology profile implied by ``cfg``'s leadfield filename."""
    from inob.config import source_target_tag
    return profile_for_tag(source_target_tag(cfg))
