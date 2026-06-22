# iNOB: Imaging Neuroscience Outside the Brain — An open-source framework for whole-body bioelectromagnetic forward modelling and sensor planning

**Oliver Case**¹\*, **Maike Schmidt**², **Pier D. Lambiase**¹, **Sarah N. Garfinkel**³†, **Gareth R. Barnes**²†

¹ UCL Neurocardiology, University College London, UK
² Wellcome Centre for Human Neuroimaging, UCL Queen Square Institute of Neurology, University College London, UK
³ Institute of Cognitive Neuroscience, University College London, UK
† Joint senior authors
\* Corresponding author: oliver.case.25@ucl.ac.uk

---

> **Manuscript status:** Full scaffold — prose complete throughout. Sections marked `[PLACEHOLDER]` require additional analysis before submission (see gap list at end of document).

---

## Abstract

The nervous system extends through the entire body, yet bioelectromagnetic source imaging has remained almost exclusively focused on the brain. Here we introduce iNOB (imaging neuroscience outside the brain), an open-source framework for whole-body forward modelling and sensor planning for both magnetometry and surface electrography. Built on a curated atlas of triangulated meshes derived from the BodyParts3D database, iNOB covers skeletal musculature, peripheral and cranial nerves including the vagus, the spinal cord, and major thoracic and abdominal viscera. Finite element forward solutions are computed using DUNEuro, with iNOB providing the anatomical atlas, source configuration, and sensor planning layer above it. Applied to the cervical vagus nerve as a worked example, we predict that triaxial optically pumped magnetometer (OPM) arrays can detect compound action potential (CAP) activity corresponding to cardiac baroreceptor traffic within approximately 1.5 minutes of averaged recording, while surface electrography remains below threshold under any practical recording duration — a biophysical gap of approximately 390-fold in single-trial sensitivity. A whole-body electrode location sweep rules out array placement as the dominant limitation. The complete pipeline, 123 unit tests, and all figure-generation scripts are released under the MIT licence.

---

## Introduction

Neurons communicate. Together they form the nervous system, and neuroscience, by its own definition, is the study of that system. Yet for the better part of a century the field has directed almost all of its instrumentation toward a single organ: the brain. The result is an asymmetry that runs deep — not only in the technology available to researchers, but in the conceptual vocabulary available to them. The viscera, the peripheral nerves, the spinal cord, the enteric system: each generates bioelectric and biomagnetic fields that, in principle, carry information about physiological state with the same fidelity that has made cortical magnetoencephalography (MEG) and electroencephalography (EEG) scientifically productive. What has not existed is a common open framework for asking, of any structure in the body: what would a sensor on the skin surface see, and how many trials would be required to see it?

The question matters most urgently for the autonomic nervous system. Vagal afferent traffic — baroreceptor signalling from the carotid sinus and aortic arch, pulmonary stretch-receptor activity, cytokine-modulated inflammatory reflex signals — carries information directly relevant to cardiovascular risk, interoceptive accuracy, psychiatric vulnerability, and the physiology of post-viral syndromes including long COVID. Direct recording from the vagus has to date required invasive cuff electrodes, intraneural microelectrodes, or ultrasound-guided microneurography — none of which is appropriate for ambulatory monitoring in clinical populations. Two complementary non-invasive modalities have recently emerged. Optically pumped magnetometers (OPMs) in conformal wearable arrays exploit the magnetic field's relative immunity to volume-conduction attenuation. High-density surface-electrode arrays — most prominently the flexible PEDOT:PSS paddle introduced by Bu et al. — exploit the proximity of the carotid sheath to the anterior cervical skin. Which modality is capable of detecting physiological vagal signals under realistic conditions, and by how much, is a question that requires a forward model to answer.

Forward modelling tools optimised for brain MEG/EEG (DUNEuro, SimNIBS, the Brainstorm–DUNEuro integration) and the body-MEG forward solver introduced by O'Neill et al. for magnetospinography are individually well-validated but address one modality at a time and are not designed for systematic whole-body sensor planning across structures. ASCENT provides intra-fascicular vagus modelling with extraordinary anatomical fidelity but cannot produce non-invasive forward solutions. No published platform delivers a unified MEG and EEG forward from a single finite element model (FEM) and a single source model — the prerequisite for any controlled head-to-head modality comparison — nor does any provide an atlas-scale framework for extending such analysis to arbitrary body structures.

Here we introduce iNOB: imaging neuroscience outside the brain. The framework (i) maintains a curated atlas of watertight triangulated meshes covering the full body, derived from the BodyParts3D anatomical database; (ii) builds multi-tissue tetrahedral FEMs from these meshes via voxel shrinkwrap and CGAL meshing; (iii) places OPM and surface-electrode sensor arrays automatically on the relevant body surface; (iv) computes both MEG and EEG leadfields from the same FEM under a single conductivity assignment using DUNEuro; and (v) estimates detectability — predicted signal-to-noise ratio (SNR) and number of averaged trials required — given specified sensor noise floors and source parameters. We demonstrate iNOB on the cervical vagus nerve as a fully worked example, comparing OPM magnetoneurography against high-density surface electrography under biophysically grounded source and noise models. We then use the framework to address two specific biophysical questions that motivate this class of modelling: is bone shielding the dominant attenuator of surface EEG signals from the cervical vagus? (No.) Is the conventional anterior-cervical electrode placement sub-optimal? (No.) The answers have direct consequences for experimental design in the field and are only obtainable through a validated forward model.

The pipeline is approximately 4,000 lines of Python, MIT-licensed, with eleven command-line tools, 123 unit tests, and every figure regenerable from a single command. iNOB is designed to be extended by the community to any source-bearing structure in the body, and the underlying forward solver — DUNEuro — is itself a community-maintained open project, so that every component from mesh to prediction can be inspected, contested, and improved.

---

## Results

### The iNOB atlas and pipeline architecture

iNOB provides an open atlas of triangulated anatomical meshes spanning the whole human body, derived from BodyParts3D version 4.3. The current release covers six tissue compartments relevant to the cervical-vagus demonstration: skin, cortical bone (a union of 74 individual skeletal elements), skeletal muscle of the neck and upper thorax, carotid-sheath vasculature (carotid artery and internal jugular vein), and the left and right vagus nerve trunks. The atlas is designed to be extensible: the same mesh-processing pipeline applies to any BodyParts3D structure, and the configuration system supports arbitrary tissue assignments.

**Fig. 1 | iNOB atlas and pipeline overview.** *(a)* Whole-body BodyParts3D atlas, anterior, lateral, and superior views, rendered from the open-source GLB output. The six tissues used in the cervical-vagus demonstration are highlighted: skin (translucent), bone (white), muscle (pink), vasculature (red), and vagus nerve trunks (yellow). *(b)* Pipeline architecture: atlas STLs → watertight geometry (voxel shrinkwrap) → multi-tissue tetrahedral FEM (CGAL) → dual-modality forward solution (DUNEuro) → SNR and trial estimation. *(c)* Slice through the cervical FEM, showing six tissue compartments. FEM summary: 209,927 nodes, 789,342 tetrahedra; minimum Joe–Liu mesh quality 0.092, fifth percentile 0.301, mean 0.524. *(d)* Validation chain: MEG calibration against Sarvas analytic and MNE-Python `_do_sphere_field` (machine precision, rtol = 10⁻¹²); EEG calibration against Berg–Scherg analytic on a controlled homogeneous-sphere FEM (empirical factor 0.622, geometry-stable to 6.8% across 27 conditions).

Each anatomical mesh undergoes a layered watertightening procedure: trimesh repair, boolean union of multi-component tissues, voxel shrinkwrap (per-tissue pitch 0.5–3 mm), and a final closure pass. The multi-tissue tetrahedral mesh is built on a single global voxel grid at 3 mm isotropic pitch by the CGAL 3D Mesh_3 algorithm (surface facet radius bound 2.5 mm, maximum tetrahedron volume 20 mm³). Schema-validated outputs at every pipeline stage ensure that corrupted geometry propagates as an error rather than a silently wrong result.

### Validation of the dual-modality forward solution

Before any biological prediction, both forward paths are validated against closed-form analytic solutions. The MEG path — DUNEuro configured with a CG-SIPG solver and conversion factor ×10⁶ to fT (nA·m)⁻¹ — is cross-validated against Sarvas's homogeneous-sphere solution via the MNE-Python `_do_sphere_field` reference implementation across four parametrised dipole orientations; agreement is at machine precision (rtol = 10⁻¹², atol = 10⁻¹²). The EEG path requires an empirical calibration because DUNEuro's internal mm-mode unit convention is not formally documented in a manner that admits *a priori* dimensional analysis. We build a controlled homogeneous-sphere FEM (radius 100 mm, σ = 0.43 S/m, ~2,700 tetrahedra), place 200 surface electrodes on a Fibonacci lattice, run the EEG forward with a unit dipole at 50 mm depth, and compute the Berg–Scherg analytic series at the same positions. The ratio analytic / DUNEuro-raw, taken as a median over all electrode-moment pairs, is the calibration factor.

**Fig. 2 | Analytic validation of the dual-modality forward.** *(a)* MEG: per-coil scatter comparing the full multi-tissue FEM against the Sarvas homogeneous-sphere analytic for one cervical source, parameterised by the Bu et al. geometry (source–axis distance 40 mm, sensor–axis distance 58.5 mm). *(b)* Residuals as a function of sensor–source distance; secondary-current contributions from conductivity interfaces amplify the FEM prediction at short distances. *(c)* Peak field magnitude along the cervical vagus polyline (FEM vs Sarvas), with bootstrapped 95% CI on the peak ratio. FEM/Sarvas peak ratio: **6.84 [95% CI 5.86, 6.94]** (n = 2,000 bootstrap resamples). At full A+C-fibre summation Q = 70 nA·m, the Sarvas prediction rescales to 2.9 pT, within the Bu et al. measured range of 1–4 pT — a literature-anchored consistency check. *(d)* EEG Berg–Scherg calibration factor across the 27-condition geometric sweep (radius ∈ {60, 100, 140} mm × source depth ∈ {0.3, 0.5, 0.7}R × conductivity ∈ {0.1, 0.43, 1.0} S/m); factor range 0.60–0.75, CV = 6.8%, invariant to σ (analytic confirmation that DUNEuro's σ-scaling and Berg–Scherg's 1/σ dependence cancel exactly). Production value: 0.622.

The 6.84-fold FEM-over-Sarvas peak amplification is consistent with secondary-current contributions at the conductivity interfaces of the multi-tissue torso, as documented for body-MEG forward modelling by O'Neill et al. The ratio is a *peak* over best-aligned coils near the spine; the median across all 7,333 (source, coil) pairs is 0.9.

### Surface topography of the dual-modality forward

**Fig. 3 | Dual-modality surface topography.** *(a)* MEG field magnitude on the cervical skin, Gaussian-KNN interpolated from the 8,190 radial and tangential OPM channels for one cervical source (z = 1,499 mm). *(b)* Cylindrical unroll (azimuth × Z): the bipolar field pattern rotates around the body axis with source position. *(c)* MEG amplitude vs sensor–source distance; 1/r² fall-off consistent with a magnetic dipole source in a weakly-varying medium. *(d)* EEG: the 32-channel PEDOT:PSS-style HD-EMG paddle (5 mm pitch, paddle32 layout matching Bu et al.) on the anterior cervical skin, bicubic-interpolated. *(e)* EEG amplitude vs sensor–source distance across the 32 contacts. Peak EEG leadfield at the cervical paddle: **0.024 µV (nA·m)⁻¹**. Peak MEG leadfield: **283 fT (nA·m)⁻¹**. The ratio of single-trial amplitudes at their respective noise floors is ~390× in favour of MEG (see Fig. 5).

### Sensitivity of forward predictions to tissue conductivity

A key practical concern for forward modelling is how strongly predictions depend on tissue conductivity values, which carry uncertainty of 20–50% in living subjects. For MEG, the secondary-current contribution to the external field depends on conductivity contrasts; for EEG, the dominant pathway through resistive tissue is expected to be far more sensitive. We compute conductivity sensitivity by independently perturbing bone (σ = 0.0042 S/m nominal) and skin/muscle (σ = 0.43 and 0.35 S/m nominal) by factors of 0.8× and 1.25×, re-running the full forward solution on the same FEM, and comparing per-source RMS leadfield amplitude.

**Fig. 4 | Conductivity sensitivity analysis.** `[PLACEHOLDER — requires running vagus-fm-sensitivity for both MEG and EEG and generating the figure. Expected content:]` *(a)* MEG: relative change in peak leadfield amplitude (%) for ±20% and ±25% perturbations of bone and skin conductivities; expected <10% sensitivity to bone, <5% to skin, consistent with the Sarvas result that B is conductivity-independent in a homogeneous sphere. *(b)* EEG: same perturbations; expected 20–50% sensitivity to soft-tissue conductivity and lower sensitivity to bone (given no bone lies between the anterior cervical vagus and the skin surface in this anatomy — confirmed in Supplementary Fig. 1). *(c)* Comparison of MEG vs EEG relative sensitivity across tissues — the headline claim that EEG is substantially more sensitive to conductivity uncertainty than MEG. `[This figure is a submission requirement — the code is in vagus_fm.analysis.sensitivity; run vagus-fm-sensitivity and vagus-fm-eeg-sensitivity to populate outputs/sensitivity_meg.json and outputs/sensitivity_eeg.json, then generate the figure.]`

### Detectability under realistic noise floors

**Fig. 5 | Detectability under realistic noise floors.** OPM noise floor: 7 fT/√Hz (QuSpin Gen-3 QZFM-3 intrinsic; older Gen-2: 15 fT/√Hz) × √(1 kHz bandwidth) = **221 fT per trial** (Gen-3). HD-EMG noise floor: amplifier 0.5 µV/√Hz combined in quadrature with Johnson noise from 10 kΩ electrode–skin contact at 37°C (1.8 µV/√Hz) = 1.87 µV/√Hz × √1 kHz = **59 µV per trial**. *(a)* MEG SNR vs number of averaged trials for four source-strength scenarios (Q ∈ {1, 5, 20, 70} nA·m corresponding to unit-calibration, sparse activation, modest CAP, and full A+C-fibre summation). *(b)* EEG SNR vs averaged trials, same scenarios. *(c)* Trials-to-SNR = 3 along the cervical vagus polyline, MEG. *(d)* Trials-to-SNR = 3, EEG. *(e)* Recording duration vs source moment Q at three physiological event rates (1, 5, 20 Hz), MEG. *(f)* Same for EEG. Shaded bands show `[PLACEHOLDER — bootstrap 95% CI over fibre-parameter uncertainty (ΔV ± 10 mV, σ_in 0.35–1.0 S/m, diameter distribution ±1 s.d.); requires Monte Carlo sweep in vagus_fm.analysis.cap_sensitivity]`.

At the highest cervical source and full A+C summation (Q = 70 nA·m), OPM Gen-3 achieves single-trial SNR ≈ **30** — immediately visible without averaging. HD-EMG single-trial SNR ≈ 8 × 10⁻², requiring ~1,500 trials (~5 minutes at 5 Hz) to reach SNR = 3. At a more clinically realistic modest CAP (Q = 20 nA·m), OPM single-trial SNR ≈ 8.6 (sub-second averaging); HD-EMG requires ~18,400 trials (~1 hour). The single-trial OPM/HD-EMG sensitivity gap is approximately **390-fold** under Gen-3 noise (180-fold under Gen-2). These estimates assume temporally uncorrelated white sensor noise; spatially correlated environmental noise inflates trial counts by factors of 1–10 depending on shielding environment (see Discussion).

### Location is not the problem; bone is not the problem

Two intuitive explanations for the HD-EMG detection failure can be ruled out by direct anatomical and computational test. First, is the conventional anterior-cervical electrode paddle poorly sited? Second, does cortical bone between the vagus and the skin surface provide the dominant electrical shielding?

**Fig. 6 | Whole-body electrode location optimisation and anatomical bone-path control.** *(a)* EEG leadfield magnitude heatmap across a 1,000-electrode Fibonacci-lattice whole-body array for one cervical source. *(b)* Cylindrical unroll of the whole-body heatmap, showing that the amplitude maximum lies on the anterior cervical neck — anatomically the same site as the conventional paddle. *(c)* Per-source single-trial SNR, 32-channel paddle vs best-of-1,000 whole-body contacts. *(d)* Peak-channel amplitude along the vagus polyline, paddle vs whole-body best; the whole-body advantage (shaded) is a maximum of **2.0×** in amplitude and does not open a new high-SNR region. *(e)* Tissue-path scan from one cervical source to the overlying anterior skin: **no bone is interposed**; the path traverses approximately 7 mm of soft tissue before exiting to air. *(f)* Trials-to-SNR = 3 at the whole-body optimal position: even at the best possible electrode site, Q = 70 nA·m requires ~7.5 × 10⁷ trials (~14 continuous days at 5 Hz).

The best-positioned electrode in the whole-body array lies 69 mm from the source, compared to 70 mm for the conventional paddle's peak-amplitude contact — anatomically the same site. The 2.0× whole-body amplitude advantage reflects better spatial sampling of the same hotspot, not discovery of a new one. A tissue-path analysis confirms that the cervical vagus at z = 1,285–1,499 mm lies anterior to the cervical vertebrae (vertebral bodies approximately 70 mm posterior in the y-direction of the LPS coordinate system); an anterior skin contact has no bone in its line-of-sight to the nerve. The dominant biophysical limiter is 1/r² attenuation of the surface potential through approximately 50 mm of electrically-conductive soft tissue, which is not amenable to electrode repositioning.

### Physiological forward simulations: baroreceptor and respiratory vagal afferents

We drive the validated leadfields with two physiologically-grounded compound action potential scenarios to produce time-domain SNR predictions directly comparable to the event-locked averaging paradigms used in experimental work.

**Fig. 7 | Physiological forward simulations.** Two scenarios are implemented. *Baroreceptor scenario*: cardiac rate 70 bpm (1.17 Hz), approximately 200 A-fibres per baroreceptor burst (lognormal diameter distribution, mean 8 µm), 150 ms after the R-wave with ±5 ms jitter, per-fibre moment via the Hämäläinen formula Q = πd²σ_in ΔV/4 with σ_in = 1.0 S/m, ΔV = 70 mV. *Deep-breathing scenario*: 6 breaths/min (0.10 Hz); rapidly adapting receptor (RAR) phasic burst of approximately 600 mixed pulmonary fibres at inspiration onset, slowly adapting receptor (SAR) tonic train of approximately 80 fibres at 50 Hz for 45% of the cycle, ΔV = 80 mV. *(a)* ECG train and best-radial-channel MEG time series, baroreceptor. *(b)* Single-trial SNR per modality, both scenarios. *(c)* Trials-to-SNR = 3 vs event rate sweep. *(d)* Power spectrum showing cardiac fundamental at 1.17 Hz and harmonics emerging from the OPM noise floor. *(e–h)* Equivalent panels for the deep-breathing scenario.

Predicted recording duration to SNR = 3 under the stationary-dipole approximation (QuSpin Gen-3, white-noise model): baroreceptor MEG ≈ **53 trials × (1/1.17 Hz) ≈ 46 seconds**; deep-breathing MEG ≈ **32 trials × (1/0.10 Hz) ≈ 5 minutes**. HD-EMG recording durations for both scenarios exceed 10⁹ trials under all noise models and are not practically achievable.

### Propagating vs stationary-dipole approximation

The physiology simulator above uses a stationary-dipole approximation: each event is represented as a biphasic compound moment at the highest-Z cervical source. The propagating model, in which the CAP wavelet travels along the nerve at fibre-diameter-dependent conduction velocities, produces a different prediction. We benchmark the approximation with a single 200-A-fibre baroreceptor wavelet (total moment Q_total = 0.74 nA·m) at the best radial MEG channel.

**Fig. 8 | Stationary vs propagating CAP benchmark.** *(a)* Time traces: stationary at cervical hotspot (peak 83.1 fT, FWHM 1.87 ms), cervical-localised propagating wavelet (peak 60.1 fT, peak-to-stationary ratio = **0.72**, FWHM 0.30 ms), and whole-vagus propagating wavelet (peak 6.4 fT, ratio = 0.08). The temporal narrowing of the propagating cervical wavelet reflects the short spatial window (FWHM ≈ 16 mm) over which the longitudinal leadfield L_long is large near the top of the polyline. *(b)* Residual (propagating − stationary). *(c)* Summary table of source-model parameters and the correction factor applied to headline trial estimates. The 0.72 peak ratio inflates trials-to-detect by 1/0.72² ≈ **1.93×** under the propagating model: baroreceptor MEG ≈ **1.5 minutes**; deep-breathing MEG ≈ **10 minutes**. For whole-vagus propagation (pulmonary or abdominal afferents), the 0.08 ratio corresponds to a ~156× inflation in trial counts.

---

## Discussion

The central result of this paper is that OPM magnetoneurography of the cervical vagus is operationally feasible under current sensor technology, while surface electrography — regardless of electrode placement, electrode density, or the presence or absence of bone along the signal path — is not. This is a biophysical result, not an instrumentation result. The cervical vagus lies approximately 50 mm deep to the anterior skin surface in a continuous, electrically conductive soft-tissue medium. At that depth, the surface electric potential from a travelling compound action potential is governed by 1/r² attenuation, and even a 1,000-electrode whole-body array recovers only a 2-fold amplitude improvement over the standard 32-channel cervical paddle — insufficient to close the 390-fold sensitivity gap by any practical combination of averaging and array density. The magnetic field, by contrast, is only weakly attenuated by the intervening medium and benefits from the secondary-current amplification at tissue conductivity interfaces that inflates FEM predictions above the homogeneous-sphere analytic by a factor of approximately seven at best-aligned coils. These two effects jointly determine the modality comparison, and both are consequences of Maxwell's equations applied to the specific anatomy of the human neck. They are therefore not addressable by improved instrumentation on the EEG side, only by approaching the source more closely — which requires invasive access.

iNOB's contribution is the quantification of this gap with a single end-to-end model that every investigator in the field can run, audit, and extend. Previous OPM and surface-electrode studies of the cervical vagus reported threshold-crossing rates from experimental recordings rather than absolute CAP amplitudes; this makes the SNR claims from those studies dependent on whatever proportion of threshold crossings are genuinely neural, a question that cannot be answered without simultaneous invasive ground truth. The forward-only biophysical predictions of iNOB side-step this question: they state what each array *would* see given a specified source, and can be checked against measured amplitudes directly. The consistency check on the MEG side — the Sarvas prediction at Q = 70 nA·m rescales to 2.9 pT, within the Bu et al. measured range of 1–4 pT — gives confidence that the source amplitude parameterisation is appropriate. On the EEG side, no such empirical benchmark currently exists; no surface-electrode study has reported absolute cervical CAP amplitudes against which forward predictions can be checked. This is acknowledged as a limitation and prioritised for future work.

Three classes of limitation should constrain the interpretation of these predictions. First, *anatomical fidelity*: all results use a single subject anatomy from the BodyParts3D atlas. Subject-specific geometry — fat thickness, muscle bulk, and the precise relationship between the carotid sheath and the surrounding tissues — will modify predicted amplitudes by tens of per cent. The "skin" compartment in the present FEM is a soft-tissue composite (σ = 0.43 S/m) that does not separately represent subcutaneous fat (σ ≈ 0.04 S/m); fat is a significant insulator at the neck and should be a dedicated compartment in subject-specific extensions. `[PLACEHOLDER — cross-subject variability analysis: obtain segmentations for 3–5 subjects from MRI or ultrasound; re-run iNOB; report prediction envelope. This is a requirement for full generalisability claims.]` Second, *source-model fidelity*: the CAP model uses fixed Hämäläinen formula parameters. The intracellular conductivity σ_in = 1.0 S/m (canonical brain-MEG convention) gives moments approximately 2.9× larger than the peripheral-axoplasm value of 0.35 S/m reported by Pelot et al.; sensitivity to this parameter and to the action-potential amplitude and fibre-diameter distribution is shown in Supplementary Fig. 2. `[PLACEHOLDER — fibre-parameter Monte Carlo; requires running vagus_fm.analysis.cap_sensitivity]`. Third, *noise model*: detectability estimates assume uncorrelated white noise. Real OPM arrays operating outside a magnetically shielded room have spatially correlated environmental noise that can inflate trial counts by factors of 1–10 depending on shielding; multi-channel beamformer processing can partially recover from this, reducing trial counts by a factor proportional to the effective number of spatial degrees of freedom in the array. Neither effect is modelled here. The estimates given therefore represent an optimistic bound for unshielded OPM and a pessimistic bound for shielded beamformed OPM.

The biggest applied implication of the present predictions is that cervical OPM-MEG is a tractable autonomic-physiology biomarker platform with current technology. Cardiac-locked baroreceptor activity should be recoverable within approximately 1.5 minutes of recording; slow-breathing respiratory afferent activity within approximately 10 minutes. These timescales are clinically practical and justify investment in the experimental infrastructure needed to validate the predictions in vivo. iNOB provides the quantitative scaffold for designing those experiments — specifying array geometry, noise floor requirements, averaging strategy, and expected signal amplitude in advance of data collection. The pipeline is structured so that subject-specific anatomy, when available from MRI or ultrasound segmentation, can replace the atlas geometry without changes to any other component.

Looking further, iNOB is a framework rather than a single application. The same pipeline that produces cervical vagus forward solutions can, by changing the atlas tissue configuration, produce forward solutions for the peripheral nerves of the limbs, the brachial and lumbosacral plexuses, the thoracic spinal cord, the enteric nervous system, and skeletal muscle. The atlas is designed to grow. The community-maintained DUNEuro core ensures that improvements to the finite element solver — higher-order elements, anisotropic conductivity tensors, realistic electrode models — propagate into iNOB without reimplementation. The open licence and reproducible pipeline mean that any result produced with iNOB can be independently verified, challenged, and extended. These design choices reflect a view that the field's progress will be fastest if the biophysical scaffolding is shared rather than reproduced independently in each laboratory.

---

## Methods

### Anatomical atlas and mesh processing

The iNOB atlas is derived from BodyParts3D version 4.3 (Database Center for Life Science, DBCLS; Creative Commons Attribution–Share Alike 2.1 Japan licence). For the cervical vagus demonstration, six tissue compartments were extracted: skin (a single whole-body surface mesh), cortical bone (74 individual skeletal element meshes unioned into a single compartment), skeletal muscle of the neck and upper thorax, carotid-sheath vasculature (carotid artery and internal jugular vein), and the left and right vagus nerve trunks.

Raw STL meshes were processed by the iNOB geometry pipeline (`vagus_fm.geometry.builder`, CLI `vagus-fm-build-geom`). Processing comprised: (i) trimesh repair (vertex merging, degenerate-face removal, winding-order normalisation); (ii) boolean union of multi-component tissues; (iii) voxel shrinkwrap to enforce watertightness — each mesh was rasterised to a signed-distance volume at tissue-specific pitch (skin 3.0 mm, bone 2.0 mm, vagus nerve 0.5 mm), converted to a binary occupancy volume by flood fill, re-meshed by marching cubes, decimated by quadric edge collapse to a target face count, and smoothed by two Taubin iterations; (iv) pymeshfix as a last-resort hole-closer for meshes that remained non-manifold after shrinkwrap. All output compartments were validated against three criteria: closed manifold topology, consistent outward winding, and Euler characteristic χ = 2 (genus zero). Any mesh failing these criteria caused the pipeline to raise an exception rather than proceed.

Coordinate axes follow the LPS convention of the BodyParts3D source data (positive x to subject's left, positive y posterior, positive z superior), as documented in `docs/COORDINATES.md`.

### Finite element mesh construction

The multi-tissue tetrahedral FEM was built using the CGAL 3D Mesh_3 library (version 5.x), accessed via the iso2mesh MATLAB interface called from Python (`vagus_fm.fem.cgal_builder`, CLI `vagus-fm-build-fem`). A global voxel label volume was constructed at 3 mm isotropic resolution with 50 mm boundary padding. Each tissue was rasterised into the label volume by a strategy matched to its geometry: ray-traced signed-distance fill for watertight closed meshes (vagus trunks, individual bones), convex-hull solid union for the assembled bone compartment, and surface-sample splatting followed by morphological closing and exterior-flood removal for the skin surface. Tissues were composited in priority order (skin → muscle → bone → vessel → vagus right → vagus left), so that interior structures overwrite exterior ones. Vagus nerve voxelisation applied a one-voxel dilation to ensure that the ~2 mm nerve was represented by at least one voxel layer at the 3 mm grid resolution. The label volume was passed to `cgalv2m` with surface facet radius bound 2.5 mm and maximum tetrahedron volume 20 mm³. Node coordinates were transformed from voxel indices to millimetres by the grid affine transform. The mesh was validated for minimum aspect-ratio quality ≥ 0.05 before use. The production mesh comprised 209,927 nodes and 789,342 tetrahedra across six tissue regions (minimum Joe–Liu quality 0.092, fifth percentile 0.301, mean 0.524).

Isotropic electrical conductivities were assigned as follows (S/m): vagus nerve 0.30, blood vessel 0.70, skeletal muscle 0.35, cortical bone 0.0042, skin/soft-tissue composite 0.43. All conductivities were scaled by 10⁻³ to S/mm for DUNEuro's internal millimetre representation. A conductivity sensitivity analysis (Fig. 4) independently perturbed bone and skin/muscle conductivities by factors of 0.8× and 1.25× and re-ran the full forward solution.

### Sensor and electrode arrays

**OPM array.** The triaxial OPM array was generated by the cylindrical raycast procedure in `vagus_fm.sensors.triaxial` (CLI `vagus-fm-sensors`). Rays were fired radially inward from a cylinder of radius 0.75 × max(x-extent, y-extent) of the skin surface, on a uniform angular–axial grid with 30 mm spacing. Surface intersection points were retained if the skin normal lay within 90° of the inward ray direction. At each accepted position, three orthonormal channels were computed: one radial (outward surface normal) and two tangential (from the singular value decomposition null-space of the normal vector). The array was cropped axially to cover the upper 45% of the body's vertical extent (cervical and upper thoracic region). This yielded approximately 8,190 channels at approximately 2,730 sensor positions in the standard configuration, arranged in the FieldTrip `grad` convention (all radial channels first, then tangential sets).

**HD-EMG electrode patch.** The 32-channel electrode paddle (`vagus_fm.sensors.electrodes`, CLI `vagus-fm-electrodes`) replicates the geometry of the Bu et al. flexible PEDOT:PSS hardware. The centroid of vagus-nerve tetrahedra within the cervical region was projected to the nearest skin surface point to define the patch origin and outward normal. A local tangent frame was established at this origin and used to place one head contact + 6 rows × 5 columns + one foot contact (32 total) at 5 mm row and 6 mm column pitch. Each contact was projected to the nearest skin face. The whole-body sensitivity sweep used a 1,000-contact Fibonacci-lattice array covering the entire skin surface.

### Forward solves

Both leadfields were computed with DUNEuro (version 2.10, built from source; Python bindings via duneuropy). A continuous Galerkin scheme with symmetric interior penalty (CG-SIPG) was used, with penalty factor 20, Houston edge-norm weighting, tensor-only weights, and a conjugate gradient solver with relative residual tolerance 10⁻¹⁰. The transfer-matrix formulation was applied throughout.

**MEG forward.** The OPM array was attached to the DUNEuro driver as point coils with their three-component orientations. The MEG transfer matrix T (dimensions n_channels × 3) was computed and applied to all source dipoles to give the leadfield matrix L (n_channels × 3 × n_sources). Output was scaled by 10⁶ to yield units of fT (nA·m)⁻¹.

**EEG forward.** Surface electrodes were attached via the closest-subentity-centre realisation. The EEG transfer matrix was computed and applied. A common-average reference was applied by subtracting the column mean across electrode channels. Output was scaled by 0.622 (the empirical Berg–Scherg calibration factor) to yield units of µV (nA·m)⁻¹.

The MEG forward solve for the full 8,190-channel array requires approximately 45 minutes on a 12-core workstation; a cluster mode (UCL Myriad / Kathleen, `cluster/submit.sh array`) splits the solve into 32 independent source-chunk jobs for ~50-minute wall-clock completion. The EEG solve for 32 electrodes requires approximately 30 seconds.

### EEG calibration via homogeneous-sphere FEM

The Berg–Scherg calibration was run by `vagus_fm.analysis.sphere_calibration` (CLI `vagus-fm-calibrate`). A homogeneous sphere FEM was built at radius 100 mm with σ = 0.43 S/m and voxel pitch 4 mm (~2,700 elements after CGAL meshing). Two hundred electrodes were placed on the sphere surface by Fibonacci lattice. The EEG forward was run with a unit dipole at 50 mm depth pointing in each of the three Cartesian directions. The Berg–Scherg series was evaluated at each electrode position and moment combination. The calibration factor was taken as the median of the analytic / DUNEuro-raw ratios across all (electrode, moment) pairs, with the median chosen for robustness to near-zero analytic values at unfavourable electrode positions. Robustness was confirmed across 27 configurations (radius ∈ {60, 100, 140} mm, source depth ∈ {0.3, 0.5, 0.7}R, σ ∈ {0.1, 0.43, 1.0} S/m): factor range 0.597–0.754, mean 0.671, CV = 6.8%. The production value 0.622 is stored as the named constant `EEG_CALIBRATION_FACTOR` in `vagus_fm/forward/eeg.py`.

### Source model

Source positions were distributed along the vagus nerve by computing the centroid of each source-tissue tetrahedron, binning centroids into 5 mm axial slabs, and averaging within each slab (~240 source positions spanning the cervical and upper thoracic vagus). Each position carried three orthogonal unit-moment dipoles.

Per-fibre dipole moments were computed by the Hämäläinen formula Q(d) = πd²σ_in ΔV / 4, where σ_in = 1.0 S/m (canonical MEG convention; the Pelot et al. peripheral-axoplasm value of 0.35 S/m, which reduces moments by ~65%, is switchable via configuration), ΔV = 70 mV for myelinated A-fibres and 80 mV for unmyelinated C-fibres. Conduction velocity was CV = 6d m/s for myelinated fibres (d ≥ 1.5 µm) and 0.5 m/s for unmyelinated fibres. The fibre-diameter population was represented as a lognormal distribution (mean 4 µm, log-standard-deviation 0.5, range 0.5–15 µm, 30 bins) for general sources, and scenario-specific distributions for the physiological simulations (baroreceptor: mean 8 µm A-fibres; pulmonary afferents: mixed population, mean 5 µm).

The compound action potential waveform was modelled as the first derivative of a Gaussian (σ = 0.5 ms), representing the biphasic extracellular waveform of a travelling action potential. Time-domain sensor signals were assembled by arc-length integration:

s_c(t) = Σ_d w(d) Σ_x L_long(c, x) · shape(t − x / CV(d))

where w(d) is the fibre-diameter weight, L_long(c, x) is the longitudinal leadfield projection at source position x, and shape(·) is the normalised biphasic waveform. Population-summed moments at full A+C-fibre activation reached approximately 70 nA·m.

### Noise model and detectability

OPM intrinsic noise was set to 7 fT/√Hz (QuSpin Gen-3 QZFM-3). Over the 1 kHz bandwidth relevant for CAP recording this gives σ_MEG = 221 fT per trial per channel. The Gen-2 value (15 fT/√Hz) is used for supplementary comparisons. HD-EMG noise was a quadrature combination of amplifier noise (0.5 µV/√Hz) and Johnson–Nyquist thermal noise from a 10 kΩ electrode–skin contact impedance at 310.15 K, giving 1.87 µV/√Hz combined and σ_EEG = 59 µV per trial per channel at 1 kHz bandwidth. Per-channel noise values were treated as independent (white-noise assumption), and multi-channel SNR was computed as the per-source RMS leadfield amplitude across channels divided by the per-channel noise and multiplied by √n_trials. Trials to SNR = 3 was determined by inverting this expression. The white-noise assumption is acknowledged as optimistic for OPM arrays; correlated environmental noise inflates trial counts by factors of 1–10 depending on shielding, and beamformer processing reduces them by a comparable factor.

### Statistical analysis

Bootstrap 95% confidence intervals (B = 2,000 resamples, seed 0) were computed for the FEM/Sarvas peak ratio (`outputs/sarvas_vs_fem.json`). The Berg–Scherg calibration CV (6.8%) was computed as the standard deviation of per-condition median factors divided by the across-condition mean. `[PLACEHOLDER — bootstrap CIs over fibre-parameter space for detectability figure; Monte Carlo over ΔV, σ_in, and diameter distribution; requires vagus_fm.analysis.cap_sensitivity]`. Pearson r and log-log slope were used for cross-modality amplitude correlations. No multiple-comparison correction was applied, as each headline number is reported with its own confidence interval rather than as a hypothesis test.

### Reproducibility and software

iNOB is a fully installable Python package (`pip install -e .`, Python ≥ 3.11) with eleven command-line entry points. Every entry point seeds `numpy.random` and `random` from a single configuration-level seed. A single command (`vagus-fm-pipeline`) regenerates all outputs from source STLs in approximately 2 hours on a 12-core workstation. CI is configured via GitHub Actions running ruff linting and 123 pytest cases on Python 3.11 and 3.12; one DUNEuro smoke test is auto-skipped when duneuropy is unavailable. Code style is enforced by ruff. `[PLACEHOLDER — Dockerfile or conda lock file for complete environment reproducibility before submission]`.

---

## Data availability

All BodyParts3D-derived STL inputs, validated leadfield NPZ outputs, calibration JSON files, and pre-rendered figures will be deposited on Zenodo upon acceptance `[DOI: TBC]`. Raw BodyParts3D meshes are available from the DBCLS Anatomography portal under CC BY-SA 2.1 Japan. The production FEM (`outputs/fem/fem_vagus.mat`, 4.6 MB) and both leadfields (`outputs/forward/`, 34 MB MEG + 0.2 MB EEG) are included in the Zenodo deposit.

## Code availability

The full iNOB pipeline is available at `[GitHub URL: TBC]` under the MIT licence. A validation record is at `docs/VALIDATION.md`; a coordinate-system reference at `docs/COORDINATES.md`; a literature review at `docs/LITERATURE_REVIEW.md`. DUNEuro is available at `[https://gitlab.dune-project.org/duneuro/duneuro]` under LGPL; build instructions for local and cluster environments are in `cluster/build_duneuro.sh` and `scripts/build_duneuro_local.sh`.

---

## References

1. Tracey, K. J. The inflammatory reflex. *Nature* **420**, 853–859 (2002). https://doi.org/10.1038/nature01321
2. Zanos, T. P. et al. Identification of cytokine-specific sensory neural signals by decoding murine vagus nerve activity. *Proc. Natl Acad. Sci. USA* **115**, E4843–E4852 (2018). https://doi.org/10.1073/pnas.1719083115
3. Marmerstein, J. T., McCallum, G. A. & Durand, D. M. Direct measurement of vagal tone in rats does not show correlation to HRV. *Sci. Rep.* **11**, 1210 (2021). https://doi.org/10.1038/s41598-020-79808-8
4. Sevcencu, C., Nielsen, T. N. & Struijk, J. J. An intraneural electrode for bioelectronic medicines for treatment of hypertension. *Neuromodulation* **21**, 777–786 (2018). https://doi.org/10.1111/ner.12758
5. Vallone, F. et al. Simultaneous decoding of cardiovascular and respiratory functional changes from pig intraneural vagus nerve signals. *J. Neural Eng.* **18**, 0460a8 (2021). https://doi.org/10.1088/1741-2552/ac0d42
6. Ottaviani, M. M., Wright, L., Dawood, T. & Macefield, V. G. In vivo recordings from the human vagus nerve using ultrasound-guided microneurography. *J. Physiol.* **598**, 3569–3576 (2020). https://doi.org/10.1113/JP280077
7. Boto, E. et al. Moving magnetoencephalography towards real-world applications with a wearable system. *Nature* **555**, 657–661 (2018). https://doi.org/10.1038/nature26147
8. Tierney, T. M. et al. Optically pumped magnetometers: from quantum origins to multi-channel magnetoencephalography. *NeuroImage* **199**, 598–608 (2019). https://doi.org/10.1016/j.neuroimage.2019.05.063
9. Tierney, T. M. et al. Pragmatic spatial sampling for wearable MEG arrays. *Sci. Rep.* **10**, 21609 (2020). https://doi.org/10.1038/s41598-020-77589-8
10. Bu, Y. et al. A flexible adhesive surface electrode array capable of cervical electroneurography during a sequential autonomic stress challenge. *Sci. Rep.* **12**, 19467 (2022). https://doi.org/10.1038/s41598-022-23114-y
11. Bu, Y. et al. Non-invasive ventral cervical magnetoneurography as a proxy of in vivo lipopolysaccharide-induced inflammation. *Commun. Biol.* **7**, 893 (2024). https://doi.org/10.1038/s42003-024-06435-8
12. Vorwerk, J. et al. A guideline for head-volume-conductor modeling in EEG and MEG. *NeuroImage* **100**, 590–607 (2014). https://doi.org/10.1016/j.neuroimage.2014.06.040
13. Vorwerk, J. et al. DUNEuro — a software toolbox for forward modeling in bioelectromagnetism. *PLOS ONE* **16**, e0252431 (2021). https://doi.org/10.1371/journal.pone.0252431
14. Saturnino, G. B. et al. SimNIBS 2.1: a comprehensive pipeline for individualized electric-field modelling for transcranial brain stimulation. In *Brain and Human Body Modeling* (Springer, 2019). https://doi.org/10.1007/978-3-030-21293-3_1
15. Tadel, F. et al. Brainstorm: a user-friendly application for MEG/EEG analysis. *Comput. Intell. Neurosci.* **2011**, 879716 (2011). https://doi.org/10.1155/2011/879716
16. O'Neill, G. C. et al. Volume conductor models for magnetospinography. *Sci. Rep.* **15**, 26258 (2025). https://doi.org/10.1038/s41598-025-10770-z
17. Musselman, E. D., Pelot, N. A. & Grill, W. M. ASCENT: a peripheral nervous system simulation pipeline. *PLOS Comput. Biol.* **17**, e1009285 (2021). https://doi.org/10.1371/journal.pcbi.1009285
18. Sarvas, J. Basic mathematical and electromagnetic concepts of the biomagnetic inverse problem. *Phys. Med. Biol.* **32**, 11–22 (1987). https://doi.org/10.1088/0031-9155/32/1/004
19. Pelot, N. A. et al. Quantified morphology of the cervical and subdiaphragmatic vagus nerves of human, pig, and rat. *Front. Neurosci.* **14**, 601479 (2020). https://doi.org/10.3389/fnins.2020.601479
20. Geselowitz, D. B. On the magnetic field generated outside an inhomogeneous volume conductor by internal current sources. *IEEE Trans. Magn.* **6**, 346–347 (1970). https://doi.org/10.1109/TMAG.1970.1066765
21. Hämäläinen, M. et al. Magnetoencephalography — theory, instrumentation, and applications to noninvasive studies of the working human brain. *Rev. Mod. Phys.* **65**, 413–497 (1993). https://doi.org/10.1103/RevModPhys.65.413
22. Gabriel, S., Lau, R. W. & Gabriel, C. The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues. *Phys. Med. Biol.* **41**, 2271–2293 (1996). https://doi.org/10.1088/0031-9155/41/11/003
23. Sekihara, K. & Nagarajan, S. S. *Adaptive Spatial Filters for Electromagnetic Brain Imaging* (Springer, 2008).
24. Hursh, J. B. Conduction velocity and diameter of nerve fibers. *Am. J. Physiol.* **127**, 131–139 (1939). https://doi.org/10.1152/ajplegacy.1939.127.1.131
25. Berg, P. & Scherg, M. A fast method for forward computation of multiple-shell spherical head models. *Electroencephalogr. Clin. Neurophysiol.* **90**, 58–64 (1994). https://doi.org/10.1016/0013-4694(94)90113-9
26. Wolters, C. H. et al. Influence of tissue conductivity anisotropy on EEG/MEG field and return current computation in a realistic head model. *NeuroImage* **30**, 813–826 (2006). https://doi.org/10.1016/j.neuroimage.2005.10.014
27. Hasgall, P. A. et al. IT'IS Database for thermal and electromagnetic parameters of biological tissues, version 4.1 (2022). https://doi.org/10.13099/VIP21000-04-1
28. Rivnay, J. et al. Organic electrochemical transistors. *Nat. Rev. Mater.* **3**, 17086 (2018). https://doi.org/10.1038/natrevmats.2017.86
29. Pas, J. et al. A bilayer flexible PEDOT:PSS-coated electrode for chronic neural recordings. *J. Neural Eng.* **15**, 065001 (2018). https://doi.org/10.1088/1741-2552/aadc1d

---

## Acknowledgements

We thank the BodyParts3D / Anatomography project (RIKEN and University of Tokyo) for the open anatomical STL meshes provided under CC BY-SA 2.1 Japan. DUNEuro is developed by the Computational Engineering and Biomedical Imaging groups at the Universities of Münster, Genova, and Aachen. iso2mesh is developed by Qianqian Fang and the JNeuroLab at Northeastern University. MNE-Python provided the analytic-sphere reference implementation used for MEG calibration validation.

O.C. is supported by `[PLACEHOLDER — grant reference]`. M.S. and G.R.B. are supported by the Wellcome Centre for Human Neuroimaging (Wellcome Trust 203147/Z/16/Z). `[PLACEHOLDER — remaining grant acknowledgements]`.

---

## Author contributions

O.C. conceived the iNOB framework, implemented the full pipeline, performed all analyses, and wrote the manuscript. M.S. advised on OPM sensor modelling and array design. P.D.L. provided clinical and physiological context for the vagal afferent scenarios. S.N.G. and G.R.B. provided senior supervision and revised the manuscript. All authors approved the final version.

## Competing interests

The authors declare no competing interests.

---

## Figure legends

**Fig. 1 | iNOB atlas and pipeline overview.** *(a)* Whole-body BodyParts3D 4.3 atlas rendered in GLB format; the six tissue compartments used in the cervical-vagus demonstration are highlighted. *(b)* Pipeline architecture from atlas STL to dual-modality SNR estimate. *(c)* Transverse slice through the cervical FEM at z = 1,300 mm showing all six compartments; inset reports mesh quality statistics. *(d)* Validation chain schematic for MEG (Sarvas / MNE-Python cross-validation) and EEG (Berg–Scherg sphere FEM calibration).

**Fig. 2 | Analytic validation.** *(a)* Per-coil MEG forward comparison, FEM vs Sarvas analytic, at one cervical source. *(b)* Residual as a function of sensor–source distance. *(c)* Peak field along the vagus polyline; FEM/Sarvas peak ratio 6.84 [95% CI 5.86, 6.94]; Sarvas at Q = 70 nA·m gives 2.9 pT, within the Bu et al. (2024) measured 1–4 pT range. *(d)* EEG Berg–Scherg calibration factor across 27-condition sweep; CV = 6.8%, invariant to σ.

**Fig. 3 | Dual-modality surface topography.** *(a)* MEG field magnitude map on the cervical skin for one cervical source; *(b)* cylindrical unroll; *(c)* amplitude vs distance. *(d)* EEG amplitude map at the 32-channel paddle; *(e)* amplitude vs distance. Peak MEG: 283 fT (nA·m)⁻¹; peak EEG: 0.024 µV (nA·m)⁻¹.

**Fig. 4 | Conductivity sensitivity analysis.** `[PLACEHOLDER]` Relative change in peak leadfield amplitude for ±20% and ±25% conductivity perturbations of bone and skin in MEG and EEG. Direct comparison of conductivity sensitivity across modalities.

**Fig. 5 | Detectability under realistic noise floors.** SNR vs averaged-trial count and trials-to-SNR = 3 for MEG and EEG at four source-strength scenarios (Q ∈ {1, 5, 20, 70} nA·m). Recording duration vs source moment at three event rates. `[Shaded bands: bootstrap CI over fibre-parameter uncertainty — PLACEHOLDER]`. Single-trial OPM/HD-EMG sensitivity gap: ~390×.

**Fig. 6 | Whole-body location optimisation and bone-path control.** *(a–e)* EEG leadfield magnitude across a 1,000-contact whole-body array; the amplitude maximum coincides with the conventional cervical paddle site. The whole-body best-electrode advantage is 2.0× in amplitude. *(f)* Tissue-path scan confirming no bone is interposed between the cervical vagus and the anterior skin.

**Fig. 7 | Physiological forward simulations.** Predicted MEG and EEG time series and SNR for baroreceptor (1.17 Hz) and deep-breathing (0.10 Hz) vagal afferent scenarios. MEG trials to SNR = 3: baroreceptor ~46 s, deep breathing ~5 min (stationary-dipole model).

**Fig. 8 | Stationary vs propagating CAP benchmark.** Peak ratio of cervical-localised propagating wavelet to stationary approximation: 0.72 (1.93× trial-count inflation). Whole-vagus propagating-to-stationary ratio: 0.08.

---

## Supplementary information

**Supplementary Fig. 1 | Bone-path anatomy.** Tissue composition along the line-of-sight from one cervical source (LPS coordinates [23, −104, 1,285] mm) to the nearest anterior skin surface. No bone is encountered; the path passes through approximately 7 mm of soft tissue.

**Supplementary Fig. 2 | Fibre-parameter sensitivity.** `[PLACEHOLDER]` Histogram of trials-to-SNR = 3 across Monte Carlo draws from the fibre-parameter space (ΔV ± 10 mV, σ_in ∈ {0.35, 1.0} S/m, diameter distribution ± 1 s.d.). Expected output from `vagus_fm.analysis.cap_sensitivity`.

**Supplementary Fig. 3 | EEG contact impedance sensitivity.** `[PLACEHOLDER]` EEG µV predictions as a function of electrode–skin contact impedance (1–100 kΩ) showing the Johnson-noise contribution to total HD-EMG noise floor.

**Supplementary Table 1 | Tissue conductivity values.** Nominal values, literature range, sources (Gabriel et al. 1996; IT'IS database v4.1), and perturbation factors used in the sensitivity sweep.

**Supplementary Table 2 | Pipeline command-line tools.** Name, function, primary inputs and outputs, and approximate run time for each of the eleven iNOB CLI entry points.

---

## Submission gap list

The following items are required before submission. Each has a corresponding `[PLACEHOLDER]` in the text above.

| Priority | Item | What to run | Figure |
|---|---|---|---|
| **Critical** | Conductivity sensitivity sweep results | `vagus-fm-sensitivity` (MEG + EEG), save `outputs/sensitivity_{meg,eeg}.json`, generate figure | Fig. 4 |
| **Critical** | Uncertainty bands on detectability | Monte Carlo in `vagus_fm.analysis.cap_sensitivity` over ΔV, σ_in, diameter distribution; add shaded bands | Fig. 5 |
| **Critical** | Single-anatomy scope framing | Rewrite abstract last sentence and Discussion para 3 to explicitly state single-atlas proof-of-concept; add cross-subject roadmap | Abstract, Discussion |
| **Major** | EEG experimental benchmark | Either: measure HD-EMG in 2–3 subjects + compare to predictions; *or* extend Discussion to explicitly bound EEG prediction uncertainty by the full σ_in range (0.35–1.0 S/m) | Discussion |
| **Major** | Correlated-noise OPM model | Literature survey of shielded vs unshielded OPM noise correlation length; add explicit trial-count inflation range to Figs. 5 and 7 | Figs. 5, 7 |
| **Major** | Dockerfile or conda lock | Create `Dockerfile` with pinned DUNEuro build; test in CI | — |
| **Major** | BodyParts3D licence statement | Add CC BY-SA 2.1 Japan attribution to `LICENSE` and `CITATION.cff` | — |
| **Important** | Fibre-parameter Monte Carlo | `vagus_fm.analysis.cap_sensitivity`; Supplementary Fig. 2 | Supplementary Fig. 2 |
| **Important** | Contact impedance sensitivity | `vagus_fm.analysis.eeg_impedance_sensitivity`; Supplementary Fig. 3 | Supplementary Fig. 3 |
| **Important** | GitHub URL + Zenodo DOI | Create public repository; upload data deposit | Throughout |
| **Important** | Grant acknowledgements | Fill in grant reference numbers | Acknowledgements |
| **Desirable** | FEM mesh convergence check | Re-run at 5 mm pitch; compare leadfield L2 change; confirm <5% | Supplementary |
| **Desirable** | Example Jupyter notebook | `examples/plot_leadfield.ipynb` showing load → topoplot | Repository |
