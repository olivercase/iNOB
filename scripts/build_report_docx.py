#!/usr/bin/env python3
"""Rebuild forward_modelling_vagus.docx as a ~2000-word Nature Methods report
for the iNOB software package. British English; numbers verified against
src/inob and the production leadfield artefacts (6 tissues, 88 sources,
8190 OPM channels). iNOB is framed as a configurable, interactive simulator."""
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL

doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10)

TITLE, SEC, SUB, BODY, SMALL = 18, 14, 11, 10, 9


def para(text, size=BODY, bold=False, italic=False, align=AL.JUSTIFY,
         space_before=2, space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    return p


def heading(text, size=SEC):
    return para(text, size=size, bold=True, align=AL.LEFT,
                space_before=10, space_after=4)


def sub(text):
    return para(text, size=SUB, bold=True, align=AL.LEFT,
                space_before=8, space_after=2)


# ----- Title block -----
para("iNOB: imaging neuroscience outside the brain — an open software package "
     "for whole-body bioelectromagnetic forward modelling and sensor planning",
     size=TITLE, bold=True, align=AL.CENTER, space_after=8)

para("Oliver Case¹ (0009-0007-4499-0682)*, Sarah N. Garfinkel² "
     "(0000-0002-5961-1012), Pier D. Lambiase¹, Gareth R. Barnes³ "
     "(0000-0002-5396-7712)", align=AL.CENTER, space_after=4)
para("¹ UCL Centre for Neurocardiology, Institute of Cardiovascular "
     "Science, University College London, London, United Kingdom",
     size=SMALL, align=AL.CENTER, space_after=1)
para("² Institute of Cognitive Neuroscience, University College London, "
     "London, United Kingdom", size=SMALL, align=AL.CENTER, space_after=1)
para("³ Department of Imaging Neuroscience, UCL Queen Square Institute of "
     "Neurology, University College London, London, United Kingdom",
     size=SMALL, align=AL.CENTER, space_after=4)
para("*Corresponding author: Oliver Case — oliver.case.25@ucl.ac.uk",
     size=SMALL, align=AL.CENTER, space_after=8)

# ----- Abstract -----
heading("Abstract")
para(
    "Bioelectromagnetic source imaging has remained almost exclusively focused "
    "on the brain, yet the peripheral and autonomic nervous systems generate "
    "fields that, in principle, are recordable at the body surface. We present "
    "iNOB (imaging neuroscience outside the brain), an open-source software "
    "package for whole-body forward modelling and sensor planning. The user "
    "selects a target structure and source configuration (location, number and "
    "strength of dipoles), chooses which anatomical meshes to include and their "
    "conductivities, generates an optically pumped magnetometer (OPM) and/or a "
    "surface-electrode array, and runs the whole pipeline — from triangulated "
    "atlas meshes through a multi-tissue tetrahedral finite-element model (FEM) "
    "to calibrated MEG and EEG leadfields — from a single command or a browser "
    "interface. The headline output answers a planning question directly: given "
    "a sensor noise floor, how many averaged trials are needed to detect a given "
    "source? Both leadfields are computed from the same FEM, under one "
    "conductivity assignment, using DUNEuro, and are calibrated to absolute "
    "units against analytic solutions. Demonstrated on the cervical vagus nerve, "
    "iNOB predicts that a triaxial OPM array detects cardiac-locked baroreceptor "
    "compound action potentials within roughly 1.5 minutes of averaging, whereas "
    "surface electrography stays below threshold under any practical recording "
    "duration — a single-trial sensitivity gap of about 390-fold. A whole-body "
    "electrode sweep rules out array placement, and a tissue-path analysis rules "
    "out bone shielding, as the dominant limitation. The complete package, unit "
    "tests and figure scripts are released under the MIT licence.")

p = para("Keywords: ", bold=True, space_after=10)
p.add_run(
    "magnetoencephalography; optically pumped magnetometers; vagus nerve; "
    "forward modelling; finite-element method; electroneurography; autonomic "
    "nervous system; DUNEuro.").font.size = Pt(BODY)

# ----- Introduction -----
heading("Introduction")
para(
    "For the better part of a century, bioelectromagnetic instrumentation has "
    "been directed almost entirely at a single organ, the brain. Yet the "
    "viscera, the peripheral nerves and the spinal cord each generate "
    "bioelectric and biomagnetic fields that, in principle, carry physiological "
    "information at a fidelity comparable to that which has made cortical "
    "magnetoencephalography (MEG) and electroencephalography (EEG) productive. "
    "What has not existed is a common, open tool for asking, of any structure in "
    "the body: what would a sensor on the skin surface see, and how many trials "
    "would be required to see it?")
para(
    "The question is most pressing for the autonomic nervous system. Vagal "
    "afferent traffic — baroreceptor signalling, pulmonary stretch-receptor "
    "activity and cytokine-modulated inflammatory-reflex signals — is directly "
    "relevant to cardiovascular risk, interoception and post-viral syndromes "
    "[1–5]. Direct recording from the vagus has so far required invasive cuff or "
    "intraneural electrodes, or ultrasound-guided microneurography [6], none of "
    "which suits ambulatory monitoring. Two non-invasive modalities have "
    "recently emerged: wearable OPM arrays, which exploit the magnetic field's "
    "relative immunity to volume-conduction attenuation [7–9], and flexible "
    "high-density PEDOT:PSS surface-electrode patches over the anterior neck "
    "[10,11]. Which modality can detect physiological vagal signals under "
    "realistic conditions, and by how much, requires a forward model to answer.")
para(
    "Forward solvers optimised for brain MEG/EEG [12–15] and the body-MEG "
    "solver for magnetospinography [16] are individually well validated but "
    "address one modality at a time; ASCENT [17] models intrafascicular vagus "
    "anatomy in detail but cannot produce non-invasive forward solutions. No "
    "published platform delivers a unified MEG and EEG forward from a single FEM "
    "and a single source model — the prerequisite for any head-to-head modality "
    "comparison — nor an atlas-scale, reconfigurable tool for arbitrary body "
    "structures. iNOB fills this gap as a software package that any investigator "
    "can install, run, audit and extend.")

# ----- Framework -----
heading("The iNOB framework")
para(
    "iNOB is configuration-driven: a single YAML file is the source of truth for "
    "the target structure, the meshes included in the volume conductor, their "
    "conductivities, the source space (location, spacing, number and strength of "
    "dipoles), the sensor arrays and the noise floors, with any field "
    "overridable from the command line. From these inputs the package (i) "
    "maintains a curated atlas of watertight triangulated meshes derived from "
    "BodyParts3D; (ii) builds a multi-tissue tetrahedral FEM by voxel shrinkwrap "
    "and CGAL meshing; (iii) places triaxial OPM and surface-electrode arrays "
    "automatically on the relevant body surface; (iv) computes both MEG and EEG "
    "leadfields from the same FEM with DUNEuro, calibrated to absolute units; and "
    "(v) estimates detectability — predicted signal-to-noise ratio (SNR) and the "
    "number of averaged trials required — for a chosen source and noise floor. "
    "The pipeline runs end to end from one orchestrator command (eleven "
    "command-line tools cover the individual stages) or from a browser-based web "
    "application (a Python API backend with a modern JavaScript front-end) in "
    "which the user edits the configuration, previews the meshes, presses run, "
    "and watches each stage stream its log live. Every stage validates its "
    "output against a schema "
    "before the next begins, and every run is seeded for reproducibility, so the "
    "cervical-vagus study below is one instantiation of a general tool rather "
    "than a bespoke analysis. The package is approximately 4,000 lines of "
    "MIT-licensed Python with a unit-test suite run in continuous integration.")

# ----- Results -----
heading("Worked example: results for the cervical vagus")
para(
    "Both forward paths reproduce their analytic gold standards before any "
    "biological prediction. The MEG leadfield matches the Sarvas / MNE-Python "
    "homogeneous-sphere solution to machine precision (rtol = 10⁻¹² across four "
    "dipole orientations) [18], and the EEG leadfield is calibrated to the "
    "Berg–Scherg series [25] with a geometry-stable factor (coefficient of "
    "variation 6.8% across 27 conditions). On the cervical anatomy the full "
    "multi-tissue FEM exceeds the homogeneous-sphere prediction at best-aligned "
    "coils by a peak factor of 6.84 (95% CI 5.86–6.94; B = 2,000 bootstrap), "
    "consistent with secondary-current contributions at conductivity interfaces "
    "[16,20]; rescaled to full A+C-fibre summation (Q = 70 nA·m) the Sarvas peak "
    "is 2.9 pT, within the 1–4 pT range measured by Bu et al. [11]. The peak MEG "
    "leadfield at the OPM array is 283 fT (nA·m)⁻¹, against 0.024 µV (nA·m)⁻¹ "
    "at the cervical electrode paddle. With the noise floors below, this gives a "
    "single-trial OPM/HD-EMG sensitivity gap of ≈390×. Under the "
    "stationary-dipole model a QuSpin Gen-3 OPM reaches the Rose criterion "
    "(SNR = 3) for cardiac-locked baroreceptor traffic in ≈46 s and for "
    "slow-breathing respiratory afferents in ≈5 min; a propagating-CAP "
    "correction (peak ratio 0.72) inflates these to ≈1.5 min and ≈10 min. "
    "Surface electrography exceeds 10⁹ trials under all noise models. A "
    "1,000-contact whole-body electrode sweep improves amplitude by at most "
    "2.0× over the standard paddle, and a tissue-path analysis confirms no bone "
    "lies between the anterior cervical vagus and the skin: neither placement nor "
    "bone shielding is limiting. The constraint is 1/r² attenuation of the "
    "surface potential through ≈50 mm of conductive soft tissue, which electrode "
    "repositioning cannot overcome.")

# ----- Methods -----
heading("Methods")

sub("Anatomical inputs and tissue-surface preparation.")
para(
    "Surface meshes were derived from the BodyParts3D open atlas (version 4.3; "
    "Database Center for Life Science, DBCLS; CC BY-SA 2.1 Japan) [30]: one "
    "whole-body skin surface, 74 individual bones spanning the cervical–thoracic "
    "skeleton, neck and upper-thoracic skeletal muscle, the carotid-sheath "
    "vasculature (carotid artery and internal jugular vein), and the left and "
    "right cervical vagus trunks. Per-tissue surfaces were processed by a "
    "layered watertightening pipeline (inob.geometry.builder): (i) trimesh "
    "repair (vertex merging, removal of degenerate and duplicate faces, normal "
    "repair); (ii) boolean union of overlapping connected components; (iii) "
    "voxel shrinkwrap, which rasterised each compartment at a per-tissue voxel "
    "pitch (0.5–3 mm), applied morphological closing and exterior flood-fill, "
    "re-extracted the surface via marching cubes, and finished with quadric "
    "decimation and Taubin smoothing; and (iv) pymeshfix as a final hole-closer. "
    "Each compartment was gated on watertightness, winding consistency and Euler "
    "number = 2; the build raised rather than persisted any failure.")

sub("Tetrahedral FEM construction.")
para(
    "The multi-tissue tetrahedral mesh was built on a single global voxel grid "
    "(3 mm pitch, 50 mm padding) covering all compartments "
    "(inob.fem.cgal_builder). Each tissue was rasterised by the strategy "
    "appropriate to its input: convex-hull union per bone clipped to the skin "
    "envelope; ray-traced and flood-filled for the watertight vagus tubes (with "
    "a one-voxel dilation so the ≈2 mm nerve is represented on the 3 mm grid); "
    "and surface-sample splatting with morphological closing and exterior flood "
    "for the skin, which accommodates anatomical openings (mouth, nostrils) that "
    "defeat naive interior fills. Tissues were composited into a single labelled "
    "image, painted outermost-first (skin → muscle → bone → vessel → vagus "
    "right → vagus left) so inner labels overwrote outer; CGAL Mesh_3 was then "
    "invoked through iso2mesh.cgalv2m (radbound 2.5, maximum element volume "
    "20 voxel³). Joe–Liu mean-ratio quality was gated at min ≥ 0.05 "
    "(inob.mesh.quality). The production mesh comprised 209,927 nodes and "
    "789,342 elements across six tissues (vagus_left, vagus_right, blood_vessel, "
    "muscle, bone, skin); minimum quality 0.092, fifth percentile 0.301, mean "
    "0.524. Isotropic, frequency-independent conductivities were assigned (S/m): "
    "vagus 0.30, blood vessel 0.70, muscle 0.35, bone 0.0042, skin 0.43 (IT'IS "
    "Database v4.1 [27]).")

sub("OPM and surface-electrode arrays.")
para(
    "A triaxial OPM array was generated by inward-radial cylindrical raycasting "
    "against the skin surface (inob.sensors.triaxial). The cylinder radius was "
    "0.75× max(X-extent, Y-extent); axial and angular spacing were matched at "
    "30 mm; glancing hits (>90° between ray and outward normal) were rejected. "
    "Each surface hit emitted one position offset by a 10 mm stand-off along the "
    "outward normal, carrying three orthogonal coil orientations (radial plus "
    "two SVD-derived tangents) in canonical FieldTrip grad layout: 8,190 "
    "channels (2,730 positions × 3 axes). For surface electrography, a "
    "32-contact PEDOT:PSS paddle (inob.sensors.electrodes; one head contact, a "
    "6 × 5 body grid and one foot contact at 5–6 mm pitch) was placed over the "
    "anterior cervical vagus, replicating the Bu et al. hardware [10]; a "
    "1,000-contact Fibonacci-lattice whole-body array was used for the placement "
    "sweep.")

sub("Forward solves with DUNEuro.")
para(
    "Both leadfields were computed by DUNEuro 2.10 [13] (Python bindings via "
    "duneuropy). The volume conductor used mm-mode (σ in S/mm) under the single "
    "conductivity assignment above. The MEG path (inob.forward.solve) attached "
    "OPMs as point coils and used the transfer-matrix API "
    "(computeMEGTransferMatrix, then applyMEGTransfer) for three orthogonal "
    "moments per source. Solver settings were CG, reduction 10⁻¹⁰, SIPG scheme, "
    "integration-order add 5, with subtract_mean and post_process_meg enabled. "
    "The EEG path (inob.forward.eeg) attached electrodes with the "
    "closest-subentity-centre realisation (codims = 1), used "
    "computeEEGTransferMatrix and applyEEGTransfer, and common-average "
    "referenced the surface potentials. Source positions were sampled at 5 mm "
    "spacing along the vagus polyline by Z-slab averaging of vagus_left tet "
    "centroids (inob.sources.vagus.vagus_sources), yielding 88 cervical sources "
    "by default. Output was scaled to fT (nA·m)⁻¹ (MEG) and µV (nA·m)⁻¹ (EEG).")

sub("Calibration to absolute units.")
para(
    "DUNEuro's mm-mode raw output is dimensionally ambiguous, so both modalities "
    "were calibrated against analytic gold standards. For MEG, the conversion "
    "factor (×10⁶ to fT per nA·m) was verified against the Sarvas closed-form "
    "solution [18] and MNE-Python's _do_sphere_field reference to machine "
    "precision (rtol = 10⁻¹²; four dipole orientations). For EEG, no a-priori "
    "dimensional analysis was reliable, so we calibrated empirically "
    "(inob.analysis.sphere_calibration): a homogeneous-tissue sphere FEM "
    "(R = 100 mm, σ = 0.43 S/m, voxel pitch 4 mm, ≈2,700 elements) was meshed "
    "identically to production; 200 surface electrodes were placed on a "
    "Fibonacci lattice; the EEG forward was computed with a unit dipole at 50 mm "
    "depth; and the per-(electrode, moment) ratio between the Berg–Scherg "
    "analytic series [25] and the DUNEuro raw output was taken as the conversion "
    "factor, using the median for robustness to near-zero analytic values. The "
    "headline factor was 0.622 (geomean 0.738). A 27-case sweep across radius "
    "{60, 100, 140} mm × depth {0.3, 0.5, 0.7}·R × conductivity {0.1, 0.43, "
    "1.0} S/m gave median 0.671, s.d. 0.046, CV = 6.8%; the factor was "
    "σ-invariant by construction (DUNEuro's σ-scaling and Berg–Scherg's 1/σ "
    "dependence cancel exactly [26]). The constant EEG_CALIBRATION_FACTOR = 0.622 "
    "is set in inob/forward/eeg.py.")

sub("Sarvas versus full-FEM benchmark.")
para(
    "The cervical-vagus geometry was approximated by the concentric-circle "
    "parameters of Bu et al. (source–axis distance 40 mm; sensor–axis distance "
    "58.5 mm, including the QuSpin 6.5 mm stand-off). We used a moving sphere "
    "centre per source — at the same Z as each source, on the cervical axis — "
    "rather than a single fixed sphere, which would inflate Sarvas amplitudes for "
    "distant sources (inob.analysis.sarvas_compare). The cervical-axis (X, Y) was "
    "a 2:1 weighted average of bone tet centroids and source-polyline XY within "
    "the source Z-band. For each (source, coil) pair the FEM leadfield was "
    "projected onto the local vagus tangent and compared with the Sarvas "
    "prediction at the same orientation and moment; 95% confidence intervals on "
    "the peak ratio were obtained by non-parametric bootstrap (B = 2,000). The "
    "amplification reflects secondary-current contributions captured by "
    "Geselowitz reciprocity [20] but absent from any single-sphere model [16].")

sub("Source model and physiological scenarios.")
para(
    "Per-fibre dipole moments followed the Hämäläinen formulation "
    "Q = π·d²·σ_in·ΔV/4 [21], with σ_in = 1.0 S/m (canonical brain-MEG "
    "convention; switchable to the peripheral-axoplasm value 0.35 S/m of Pelot "
    "et al. [19]) and action-potential amplitudes ΔV = 70 mV (myelinated A) or "
    "80 mV (unmyelinated C). Conduction velocity followed Hursh's law "
    "(CV ≈ 6 m·s⁻¹·µm⁻¹) for myelinated fibres and 0.5 m·s⁻¹ for "
    "unmyelinated [24]. Two scenarios were implemented "
    "(inob.physiology.scenarios): (i) cardiac-locked baroreceptor activity at "
    "70 bpm (1.17 Hz), ≈200 A-fibres per burst (lognormal diameter centred on "
    "8 µm) at 150 ms post-R-wave with ±5 ms jitter; and (ii) slow/deep breathing "
    "at 6 breaths·min⁻¹ (0.10 Hz), with a phasic rapidly-adapting-receptor "
    "burst of ≈600 mixed pulmonary fibres at inspiration onset and a "
    "slowly-adapting tonic train of ≈80 fibres at 50 Hz over 45% of the cycle. "
    "The simulator (inob.physiology.simulate) generated per-channel time series "
    "at fs = 10 kHz by superposing biphasic-Gaussian compound action potentials "
    "(AP width σ = 0.5 ms) at the cervical hot-spot; a propagating-CAP control "
    "traversed the full vagus polyline at fibre-diameter-dependent CV.")

sub("Noise floors and detectability.")
para(
    "Per-channel noise was modelled as quadrature-summed white noise integrated "
    "over a 1 kHz analysis bandwidth (inob.analysis.snr). For OPM-MEG, "
    "σ_n = 7 fT/√Hz × √1000 Hz = 221 fT (QuSpin Gen-3, QZFM-3; the Gen-2 value "
    "of 15 fT/√Hz is available for comparison). For HD-EMG, the amplifier term "
    "(0.5 µV/√Hz) was summed in quadrature with the Johnson voltage noise of "
    "the skin contact, v_n = √(4 k_B T R), at T = 310.15 K and R = 10 kΩ "
    "(≈0.013 µV/√Hz; negligible against the amplifier at PEDOT:PSS impedances), "
    "giving σ_n ≈ 15.8 µV. Per-source signal amplitude was the channel-RMS of "
    "the orthogonal-moment-averaged leadfield; trials-to-detect at the Rose "
    "criterion (SNR = 3) was inverted from SNR ∝ √N for white uncorrelated "
    "noise. Spatially correlated environmental MEG noise inflates "
    "trials-to-detect by 1–10×, and LCMV/SAM beamforming reduces it by a "
    "comparable factor [23]; neither is modelled here.")

# ----- Statements -----
heading("Data and code availability")
para(
    "All BodyParts3D-derived STL inputs, validated leadfield arrays (NPZ), "
    "calibration records (JSON) and figure scripts will be deposited on Zenodo "
    "on acceptance. Raw BodyParts3D meshes are available from the DBCLS "
    "Anatomography portal under CC BY-SA 2.1 Japan. The full iNOB package is "
    "released under the MIT licence; DUNEuro is available under the LGPL, with "
    "local and cluster build scripts provided.")

heading("Funding")
para(
    "This research is funded by the British Heart Foundation (BHF), grant codes "
    "SI/F/24/21170016 and RG/F/25/110148, and by the Discovery Research Platform "
    "for Naturalistic Neuroimaging, funded by Wellcome (grant code "
    "226793/Z/22/Z).")

heading("Conflict of interest")
para(
    "The authors declare that the research was conducted in the absence of any "
    "commercial or financial relationships that could be construed as a "
    "potential conflict of interest.")

heading("Ethics statement")
para(
    "No new human or animal data were collected. All anatomy derives from the "
    "open BodyParts3D atlas; ethical approval was therefore not required.")

heading("CRediT author statement")
para(
    "Oliver Case: Conceptualisation; Methodology; Software; Formal analysis; "
    "Writing – original draft; Visualisation. Sarah N. Garfinkel: "
    "Conceptualisation; Writing – review & editing; Supervision. Gareth R. "
    "Barnes: Conceptualisation; Writing – review & editing; Supervision; Funding "
    "acquisition. Pier D. Lambiase: Writing – review & editing; Funding "
    "acquisition.")

# ----- References -----
heading("References")
refs = [
    "Tracey, K. J. The inflammatory reflex. Nature 420, 853–859 (2002).",
    "Zanos, T. P. et al. Identification of cytokine-specific sensory neural signals by decoding murine vagus nerve activity. Proc. Natl Acad. Sci. USA 115, E4843–E4852 (2018).",
    "Marmerstein, J. T., McCallum, G. A. & Durand, D. M. Direct measurement of vagal tone in rats does not show correlation to HRV. Sci. Rep. 11, 1210 (2021).",
    "Sevcencu, C., Nielsen, T. N. & Struijk, J. J. An intraneural electrode for bioelectronic medicines for treatment of hypertension. Neuromodulation 21, 777–786 (2018).",
    "Vallone, F. et al. Simultaneous decoding of cardiovascular and respiratory functional changes from pig intraneural vagus nerve signals. J. Neural Eng. 18, 0460a8 (2021).",
    "Ottaviani, M. M., Wright, L., Dawood, T. & Macefield, V. G. In vivo recordings from the human vagus nerve using ultrasound-guided microneurography. J. Physiol. 598, 3569–3576 (2020).",
    "Boto, E. et al. Moving magnetoencephalography towards real-world applications with a wearable system. Nature 555, 657–661 (2018).",
    "Tierney, T. M. et al. Optically pumped magnetometers: from quantum origins to multi-channel magnetoencephalography. NeuroImage 199, 598–608 (2019).",
    "Tierney, T. M. et al. Pragmatic spatial sampling for wearable MEG arrays. Sci. Rep. 10, 21609 (2020).",
    "Bu, Y. et al. A flexible adhesive surface electrode array capable of cervical electroneurography during a sequential autonomic stress challenge. Sci. Rep. 12, 19467 (2022).",
    "Bu, Y. et al. Non-invasive ventral cervical magnetoneurography as a proxy of in vivo lipopolysaccharide-induced inflammation. Commun. Biol. 7, 893 (2024).",
    "Vorwerk, J. et al. A guideline for head-volume-conductor modeling in EEG and MEG. NeuroImage 100, 590–607 (2014).",
    "Vorwerk, J. et al. DUNEuro — a software toolbox for forward modeling in bioelectromagnetism. PLOS ONE 16, e0252431 (2021).",
    "Saturnino, G. B. et al. SimNIBS 2.1: a comprehensive pipeline for individualized electric-field modelling for transcranial brain stimulation. In Brain and Human Body Modeling (Springer, 2019).",
    "Tadel, F. et al. Brainstorm: a user-friendly application for MEG/EEG analysis. Comput. Intell. Neurosci. 2011, 879716 (2011).",
    "O'Neill, G. C. et al. Volume conductor models for magnetospinography. Sci. Rep. 15, 25649 (2025).",
    "Musselman, E. D., Pelot, N. A. & Grill, W. M. ASCENT: a peripheral nervous system simulation pipeline. PLOS Comput. Biol. 17, e1009285 (2021).",
    "Sarvas, J. Basic mathematical and electromagnetic concepts of the biomagnetic inverse problem. Phys. Med. Biol. 32, 11–22 (1987).",
    "Pelot, N. A. et al. Quantified morphology of the cervical and subdiaphragmatic vagus nerves of human, pig, and rat. Front. Neurosci. 14, 601479 (2020).",
    "Geselowitz, D. B. On the magnetic field generated outside an inhomogeneous volume conductor by internal current sources. IEEE Trans. Magn. 6, 346–347 (1970).",
    "Hämäläinen, M. et al. Magnetoencephalography — theory, instrumentation, and applications to noninvasive studies of the working human brain. Rev. Mod. Phys. 65, 413–497 (1993).",
    "Gabriel, S., Lau, R. W. & Gabriel, C. The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues. Phys. Med. Biol. 41, 2271–2293 (1996).",
    "Sekihara, K. & Nagarajan, S. S. Adaptive Spatial Filters for Electromagnetic Brain Imaging (Springer, 2008).",
    "Hursh, J. B. Conduction velocity and diameter of nerve fibers. Am. J. Physiol. 127, 131–139 (1939).",
    "Berg, P. & Scherg, M. A fast method for forward computation of multiple-shell spherical head models. Electroencephalogr. Clin. Neurophysiol. 90, 58–64 (1994).",
    "Wolters, C. H. et al. Influence of tissue conductivity anisotropy on EEG/MEG field and return current computation in a realistic head model. NeuroImage 30, 813–826 (2006).",
    "Hasgall, P. A. et al. IT'IS Database for thermal and electromagnetic parameters of biological tissues, version 4.1 (2022).",
    "Rivnay, J. et al. Organic electrochemical transistors. Nat. Rev. Mater. 3, 17086 (2018).",
    "Pas, J. et al. A bilayer flexible PEDOT:PSS-coated electrode for chronic neural recordings. J. Neural Eng. 15, 065001 (2018).",
    "Mitsuhashi, N. et al. BodyParts3D: 3D structure database for anatomical concepts. Nucleic Acids Res. 37, D782–D785 (2009).",
]
for i, r in enumerate(refs, 1):
    para(f"{i}. {r}", size=SMALL, align=AL.LEFT, space_before=0, space_after=2)

import os
_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs", "forward_modelling_vagus.docx")
doc.save(_out)

import re
body = " ".join(p.text for p in doc.paragraphs if not re.match(r"^\d+\. ", p.text))
print("Saved. Main-text words (excl. refs):", len(re.findall(r"\w+", body)))
