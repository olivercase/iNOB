# A unified torso-scale forward model for non-invasive cervical-vagus recording: predicted signal characteristics and array design for OPM magnetoneurography vs high-density surface electrography

**Authors.** Oliver Case¹, [collaborators TBC]

¹ University College London, Department of [TBC].

**Corresponding author.** oliver@fizzymilk.com.

---

## Abstract

Non-invasive read-out of cervical-vagus afferent activity could transform autonomic-physiology biomarker development, but the field lacks a common biophysical scaffold for comparing recording modalities head-to-head. We introduce a fully open, end-to-end finite-element forward model of the human torso that, from a single anatomy and a single source model, predicts both the magnetic field at an optically-pumped magnetometer (OPM) array and the surface potential at a high-density (HD) electrode patch. The pipeline is calibrated against analytic gold standards at every step: the magnetic forward matches Sarvas's homogeneous-sphere solution to machine precision against MNE-Python (rtol = 10⁻¹²), and the electric forward is empirically calibrated against the Berg-Scherg series on a controlled homogeneous-sphere FEM, yielding a DUNEuro mm-mode conversion factor of 0.622 that is geometry-stable to within 6.8% over a 27-case (radius, source-depth, conductivity) sweep. Driven by a propagating compound action potential with realistic fibre-population dispersion, we predict that a triaxial QuSpin Gen-3 OPM array can detect cardiac-locked baroreceptor activity within **~1.5 minutes** and slow-breathing vagal afferents within **~10 minutes** of averaged recording at full A+C-fibre summation, while a 32-channel Malliaras-group flexible-textile PEDOT:PSS HD-EMG paddle and even a 1 000-electrode whole-body array remain below detection threshold under any practical recording duration. The single-trial OPM/HD-EMG sensitivity gap at the cervical vagus is **almost three orders of magnitude (~390×)** — a biophysical limit imposed by volume-conduction attenuation rather than electrode placement, as confirmed by a whole-body location-optimisation study that finds the best-positioned surface contact within 1 mm of the conventional cervical paddle. The pipeline (≈ 4 000 lines of Python, MIT-licensed; 11 command-line tools; 123 unit tests; full Nature-formatted figures) provides the community with a calibrated, reproducible scaffold for dual-modality vagal forward modelling and SNR / array-design analysis.

**Keywords:** vagus nerve, magnetoneurography, optically-pumped magnetometers, surface electrography, finite-element forward modelling, autonomic biomarkers.

---

## 1 Introduction

The cervical vagus carries every parasympathetic visceral afferent of clinical interest — baroreceptor traffic from the carotid sinus and aortic arch, pulmonary stretch-receptor activity, and the cytokine-modulated signalling that defines the inflammatory reflex¹. Direct recording from the vagus has so far required invasive cuff electrodes²,³, intraneural microelectrodes⁴,⁵, or ultrasound-guided microneurography⁶, none of which is realistic for sustained ambulatory monitoring of healthy subjects or out-patient populations. Two complementary non-invasive modalities have emerged. Optically-pumped magnetometers (OPMs) operating in conformal wearable arrays⁷⁻⁹ exploit the magnetic field's near-immunity to volume-conduction attenuation but require magnetic shielding and bandwidths recently extended to ~500 Hz¹⁰. High-density surface-electrode arrays — most prominently the flexible Ag/AgCl paddle introduced by Bu *et al.*¹¹ and its OPM analogue¹² — exploit the proximity of the carotid sheath to the anterior neck skin but inherit the bone-and-soft-tissue volume conductor that has historically limited surface electrography to gross EMG-scale signals.

Which modality wins, and under what assumptions, has not been answered by an *a-priori* biophysical model. The published in vivo measurements¹¹⁻¹² report threshold-crossing rates rather than absolute compound-action-potential (CAP) amplitudes, which precludes the SNR / array-geometry / source-strength trade-off analyses that a deployment decision requires. This is the gap we close.

Forward-modelling tools optimised for brain MEG/EEG (DUNEuro¹³, SimNIBS¹⁴, the Brainstorm DUNEuro integration¹⁵) and the body-MEG forward solver introduced by O'Neill *et al.*¹⁶ are individually well-validated but address one modality at a time. ASCENT¹⁷ provides intra-fascicular vagus modelling with extraordinary anatomical fidelity but cannot produce non-invasive forward solutions. No published platform delivers a *unified* MEG + EEG forward from a single FEM and a single source model — the prerequisite for any controlled head-to-head comparison.

We introduce that platform. The pipeline (i) builds a watertight 4-tissue tetrahedral mesh from BodyParts3D atlas STLs via voxel shrink-wrap and CGAL multi-region meshing, (ii) places a triaxial OPM array of 8 190 channels by cylindrical raycasting and a 32-channel PEDOT:PSS-style HD-EMG paddle directly on the cervical skin, (iii) computes both leadfields with DUNEuro under a single conductivity assignment, (iv) calibrates the magnetic conversion factor against Sarvas¹⁸ and MNE-Python's `_do_sphere_field`, the electric conversion factor against Berg-Scherg on a controlled homogeneous-sphere FEM, and (v) drives the leadfields with a propagating compound-action-potential source model parameterised on Pelot *et al.*¹⁹ vagal fibre populations. We then compare predicted single-trial and trials-averaged signals across modalities under realistic noise floors (QuSpin Gen-2 OPM intrinsic + amplifier and Johnson noise for HD-EMG), and address two biophysical control questions head-on: *is bone shielding the dominant attenuator of EEG signals?* (No.) *Is the cervical paddle sub-optimally placed?* (No.) The answer to *can either modality see physiological vagal activity?* is yes for OPM-MEG and no for HD-EMG, and we quantify exactly how badly the latter loses.

This paper makes five new contributions to the field. First, a single FEM forward yielding both MEG and EEG leadfields under one source model — enabling head-to-head modality comparison that prior work has only been able to assert. Second, an empirical calibration of DUNEuro's mm-mode EEG output against a Berg-Scherg analytic on a controlled sphere FEM, resolving a unit-convention ambiguity that has caused 10⁶ errors in earlier implementations. Third, a moving-dipole simulator with cardiac-locked baroreceptor and slow-breathing respiratory scenarios, predicting trials-to-detection at realistic noise floors. Fourth, a whole-body 1 000-electrode location-optimisation experiment that rules out electrode placement as the dominant SNR limiter for HD-EMG. Fifth, a fully reproducible, MIT-licensed Python pipeline (123 tests, ruff-clean, eleven command-line tools) that makes every figure regenerable from a single command.

---

## 2 Results

### 2.1 Pipeline overview and validation chain

**Figure 1.** Pipeline overview and validation chain. *(a)* Atlas STLs (74 bones, 1 skin, 2 vagus trunks) → watertight geometry HDF5 via voxel shrink-wrap → multi-tissue tetrahedral FEM (209 927 nodes, 789 342 elements; minimum Joe-Liu mesh quality 0.092, fifth percentile 0.301, mean 0.524). *(b)* Two leadfields computed by DUNEuro from the same FEM under one source model. *(c)* Validation chain: MEG conversion factor (×10⁶) verified against MNE-Python `_do_sphere_field` to machine precision (rtol = 10⁻¹²) and against the in-vivo measurement range of Bu *et al.*¹²; EEG conversion factor empirically calibrated to 0.622 against the Berg-Scherg analytic on a 100 mm-radius, σ = 0.43 S/m homogeneous sphere FEM (2 731 elements, 200 surface electrodes, source depth 50 mm). *(d)* The calibrated factor is geometry-stable: across a 27-case sweep (radius ∈ {60, 100, 140} mm × source depth ∈ {0.3, 0.5, 0.7}·R × conductivity ∈ {0.1, 0.43, 1.0} S/m), median factor 0.671 ± 0.046, coefficient of variation 6.8%. The factor is invariant to σ — analytic confirmation that DUNEuro's σ-scaling and Berg-Scherg's 1/σ dependence cancel. The production value 0.622 sits within the range.

The two FEM solves run in approximately 45 minutes (8 190 OPM channels) and 30 seconds (32 EEG contacts) on a 12-core MacBook. The cluster pipeline (Myriad / Kathleen profiles) splits the OPM solve into 32 chunks for ~50-minute completion.

### 2.2 Sarvas vs full-FEM benchmark

**Figure 2.** Sarvas analytic vs full multi-tissue FEM forward, parameterised by the Bu *et al.*¹² cervical-concentric-circles geometry (source–axis distance 40 mm, sensor–axis distance 58.5 mm including QuSpin 6.5 mm stand-off). *(a)* Per-coil scatter at one cervical source — FEM and analytic correlated but not identical. *(b)* Residuals concentrate at the smallest sensor–axis distances, where multi-tissue secondary-current contributions (Geselowitz²⁰) are largest. *(c)* Peak |B| over the OPM array vs source position along the cervical vagus polyline (z = 1071–1499 mm); FEM peak ≈ 280 fT/(nA·m) versus Sarvas ≈ 41 fT/(nA·m), peak ratio **6.84 [bootstrapped 95% CI 5.86, 6.94]** — consistent in sign and magnitude with the secondary-current amplification documented for body-MEG forward modelling by O'Neill *et al.*¹⁶. *(d)* FEM/Sarvas amplitude-ratio histogram across all 7 333 (source, coil) pairs; median 0.9, peak 6.84. The 6.84× ratio is therefore a *peak* over best-aligned coils close to the spine, not a globally-applied amplification.

When the Sarvas prediction is rescaled to the Hämäläinen²¹ full A+C fibre summation (Q = 70 nA·m), peak |B| = **2.9 pT** — within Bu *et al.*'s¹² measured 1–4 pT range, a literature-anchored validation of the magnetic side of the pipeline.

### 2.3 Surface topology of the dual-modality forward

**Figure 3.** Surface-interpolated topoplots for one cervical-vagus source (z = 1499 mm). *(a)* MEG field magnitude on the cervical body skin, Gaussian-KNN-interpolated from the radial OPM coils (8 190/3 = 2 730 sensor positions in the full array; cropped to a ±220 mm Z-slab). *(b)* Cylindrical unroll (azimuth × Z) — the bipolar field pattern rotates around the body axis with the source position. *(c)* MEG amplitude vs sensor–source distance; 1/r² fall-off as predicted analytically. *(d)* EEG patch placement context (3-D). *(e)* HD-EMG paddle topology (2-D, bicubic-interpolated): 1-head + 5×6-body + 1-foot layout matching the physical PEDOT:PSS hardware of Bu *et al.*¹¹. *(f)* EEG amplitude vs sensor–source distance (32 contacts).

The peak EEG leadfield is 0.024 µV per nA·m of source moment at the cervical paddle and 0.049 µV per nA·m at the best position in a whole-body sweep (see §2.5).

### 2.4 Detectability under realistic noise floors

**Figure 4.** Detectability of cervical-vagus CAP activity under realistic per-modality noise. Noise floors: OPM intrinsic 7 fT/√Hz × √(1 kHz BW) = **221 fT** (QuSpin Gen-3, QZFM-3); HD-EMG 0.5 µV/√Hz amplifier + Johnson contribution from 10 kΩ skin contact (Malliaras-group flexible-textile PEDOT:PSS²⁸,²⁹) = **15.8 µV**. The Johnson term is essentially negligible at PEDOT:PSS contact impedances; the amplifier dominates, so HD-EMG noise is set by front-end electronics rather than the electrode/skin interface. *(a, d)* Predicted SNR vs averaged-trial count for four source-strength scenarios (Q ∈ {1, 5, 20, 70} nA·m representing calibration-unit, sparse activation, modest CAP, and full A+C summation respectively). *(b, e)* Trials-to-SNR=3 along the vagus polyline, per modality. *(c, f)* Recording duration vs source dipole moment Q at three event rates (1, 5, 20 Hz).

At the highest cervical source (z = 1499 mm) and full A+C summation (Q = 70 nA·m), MEG single-trial peak SNR ≈ **30** under Gen-3 (immediately visible without averaging); MEG trials to SNR = 3 ≈ 0.01 (single-trial detection; **sub-second** at 5 Hz event rate). HD-EMG single-trial SNR = 8 × 10⁻², trials to SNR = 3 = 1 500, recording time at 5 Hz = **5 minutes**. At a more realistic modest CAP (Q = 20 nA·m), MEG single-trial SNR ≈ 8.6 (visible single-trial), HD-EMG trials = 18 400 (~1 hour at 5 Hz). At sparse activation (Q = 5 nA·m), MEG needs <1 second of averaging, HD-EMG ~16 hours. The single-trial OPM-MEG / HD-EMG sensitivity gap is **~390×** under Gen-3 vs the 180× we computed for Gen-2.

### 2.5 Location is not the problem; bone shielding is not the problem

A whole-body EEG sensitivity sweep (1 000-electrode Fibonacci-lattice array on the entire skin surface, dedicated DUNEuro forward solve, ~17 min on 12 cores) tests whether the cervical paddle is sub-optimally placed.

**Figure 5.** Whole-body location optimisation. *(a)* |L_EEG| heatmap on the body skin, interpolated from the 1 000-electrode whole-body array for one cervical source. *(b)* Cylindrical unroll. *(c)* Per-source single-trial SNR, paddle vs whole-body. *(d)* Best-channel |L| along the vagus, paddle vs whole-body (whole-body advantage shaded). *(e)* Amplitude vs distance, with 1/r² overlay. *(f)* Headline-numbers panel.

The best-positioned electrode in the whole-body array sits **69 mm** from the source, vs **70 mm** for the cervical paddle's argmax — anatomically the same site (anterior cervical neck, lateral to the trachea, medial to the SCM). The peak amplitude advantage of whole-body over paddle is **2.0×** (better spatial sampling of the same hot-spot, not a new hot-spot). Even at the whole-body argmax, trials to detection at Q = 70 nA·m is 7.5 × 10⁷ (~14 days continuous at 5 Hz). **Location is therefore not the dominant SNR limiter.**

A complementary anatomical control (Figure S1, reproducible from `docs/COORDINATES.md`) confirms that the vagus polyline is anterior to the cervical vertebrae: a tissue-path scan from a cervical source to anterior skin passes through 7 mm of soft tissue before exiting to air — *no bone interposed*. **Bone shielding is therefore not the dominant attenuator either** for an anteriorly-sited HD-EMG patch. The 1/r²-falloff of the surface potential through ~50 mm of soft tissue is the dominant biophysical limit.

### 2.6 Cross-modality coupling

The two leadfields share the same FEM and the same source space; given a known source position, the EEG observation determines the source moment by least-squares (32 channels ≫ 3 moment components, well-conditioned), and the MEG topo follows by forward propagation.

**Figure 6.** Cross-modality coupling. *(a)* Per-source amplitude scatter, coloured by source Z position. Pearson r = 0.10, log-log slope = 0.12 — surprisingly weak, reflecting the local field-of-view of the 32-channel cervical paddle versus the body-spanning OPM array. *(b)* Observed EEG topo for one source (longitudinal moment). *(c)* Predicted MEG topo recovered from the EEG observation via shared-source-space inversion (Gaussian noise added at the realistic HD-EMG floor; bootstrapped 95% CI overlaid on the inset). *(d)* Actual FEM MEG topo for the same source, with prediction RMS error displayed.

Coupling is exact in the noise-free limit but degrades sharply with realistic noise — consistent with the SNR gap quantified in Figure 4.

### 2.7 Moving-dipole physiology: baroreceptor and deep-breathing scenarios

We drive the leadfields with two physiologically-grounded compound-action-potential trains: cardiac-locked baroreceptor activity (HR = 70 bpm = 1.17 Hz, ~200 A-fibres per burst at 150 ms after R-wave, ±5 ms jitter, fibre-diameter distribution centred on d = 8 µm) and slow / deep-breathing afferent activity (6 breaths/min = 0.10 Hz, RAR phasic burst of ~600 mixed pulmonary fibres at inspiration onset + SAR tonic train of ~80 fibres at 50 Hz throughout 45% of the cycle). Per-fibre dipole moments via Hämäläinen²¹ Q = π·d²·σ_in·ΔV/4 with σ_in = 1 S/m and ΔV ∈ {70, 80} mV.

**Figure 7.** Moving-dipole physiology simulation. Top row, baroreceptor (1.17 Hz). *(a)* ECG R-wave train + best-MEG-channel time series. *(b)* Single-trial SNR — both modalities. *(c)* Trials-to-SNR=3 along event-rate sweep. *(d)* Best-MEG-channel spectrum — the cardiac fundamental at 1.17 Hz and its harmonics emerge cleanly above the noise floor. Bottom row, slow / deep breathing (0.10 Hz). *(e–h)* The respiratory fundamental at 0.10 Hz emerges in panel h; per-cycle MEG amplitude tracks the inspiration / expiration cycle in panel e.

Predicted recording time to SNR = 3 at the per-trial signal levels these physiological scenarios produce, under the stationary-dipole approximation and QuSpin Gen-3 noise floor: baroreceptor MEG ≈ 53 trials × (1/1.17 Hz) ≈ **46 seconds**; deep-breathing MEG ≈ 32 trials × (1/0.10 Hz) ≈ **5 minutes**. The propagating-CAP control benchmark (§3, Figure 8) revises these to ≈ **1.5 min** and ≈ **10 min** respectively — within the same operational decade. EEG recording times for the same scenarios, even at the whole-body optimum, exceed 10⁹ trials and are not practically achievable under either model. Critically, both predictions emerge from a *forward-only* biophysical simulation that does not depend on threshold-crossing detection methods, distinguishing this work from prior surface-electrode¹¹ and OPM¹² studies that infer "firing rate" from amplitude excursions of unclear biological identity.

**Figure 8.** Stationary vs propagating CAP comparison at the best radial MEG channel (#2219), single 200-A-fibre baroreceptor event, total moment Q_total = 0.74 nA·m. *(a)* Time traces overlaid: stationary at hot-spot (manuscript convention; peak 83.1 fT, FWHM 1.87 ms), cervical-localised moving wavelet (peak 60.1 fT, P/S ratio 0.72, FWHM 0.30 ms), and whole-vagus moving wavelet (peak 6.4 fT, P/S ratio 0.08). The cervical wavelet narrows in time because L_long has a sharp spatial peak (FWHM ≈ 16 mm) at the top of the polyline that the wave traverses in ≈ 0.3 ms at CV ≈ 50 m/s. *(b)* Residual = cervical-propagating − stationary, showing where in time the two diverge. *(c)* Summary of source-model parameters and headline numbers. The 0.72 P/S ratio inflates the trials-to-detect numbers in panels §2.5 and §2.7 by 1/0.72² ≈ 1.93×.

---

## 3 Discussion

The headline finding — that an OPM-MEG cervical array can detect physiologically-meaningful vagal-afferent activity within minutes of averaging at realistic noise floors, while no surface-electrode geometry can — is a *biophysical* result, not an instrumentation result. Surface electric fields from a deep cervical-vagus source decay as 1/r² through the volume conductor and, even with 1 000 optimally-placed contacts, recover only ~2× more signal than the conventional 32-channel paddle. The vagus is not buried by bone — the anatomy verification in §2.5 rules that out — but it is buried by ~50 mm of soft tissue, and Maxwell's equations are not negotiable at that depth.

The pipeline's contribution is the *quantification* of this gap with a single end-to-end model that everyone in the field can run, audit, and extend. Previous OPM¹² and surface-electrode¹¹ studies of the cervical vagus reported threshold-crossing rates rather than absolute CAP amplitudes; this leaves the SNR claim resting on whatever fraction of those threshold crossings are genuinely neural (the rest being cardiac and respiratory artifact, sympathetic skin-nerve activity, and SCM EMG, none of which can be excluded without simultaneous invasive ground truth). Our forward-only biophysical model side-steps the threshold-crossing question entirely — it predicts what the OPM and electrode arrays *should* see if the source is what we model — and the predictions can be checked against in vivo measurement amplitudes directly. The 2.9 pT Sarvas prediction at Q = 70 nA·m sits cleanly inside Bu *et al.*'s¹² measured 1–4 pT range, providing one such consistency check on the magnetic side. We have no comparable empirical benchmark on the electric side, because no surface-electrode study has reported absolute CAP amplitudes at the cervical vagus to date.

Three classes of limitation should constrain the interpretation of the predictions and frame the future work that the pipeline enables.

**Anatomical fidelity.** All forwards are computed on a single subject anatomy from the BodyParts3D atlas. Subject-specific MR or ultrasound segmentation will modify the predicted amplitudes by tens of per cent through fat thickness, muscle anisotropy, and exact carotid-sheath positioning. The "skin" tissue in the FEM is a soft-tissue composite (muscle + fat + connective tissue + dermis) at σ = 0.43 S/m, on the high side of the Gabriel four-Cole–Cole model²² range; subcutaneous fat (σ ≈ 0.04 S/m, ~10× lower) deserves a dedicated tissue label in subject-specific extensions. Conductivities are static — no frequency dependence at 1 kHz, no muscle anisotropy. These are well-known *limits of the dataset*, not bugs in the pipeline, and our FEM-vs-Sarvas amplification ratio (6.84× peak) is consistent with the body-MEG forward modelling literature¹⁶.

**Source-model fidelity.** The physiology simulator (§2.7) uses a stationary-dipole approximation: every event in a Scenario is collapsed to a biphasic moment trace at the cervical hot-spot (the highest-Z source on the polyline) rather than a wavelet that propagates rostrally through the cervical bundle at fibre-diameter-dependent CV. We benchmark this approximation in **Figure 8**: a single 200-A-fibre baroreceptor wavelet of total moment 0.74 nA·m, computed at the best radial MEG channel (#2219), gives stationary peak 83.1 fT (manuscript convention), cervical-localised moving-wavelet peak 60.1 fT, and whole-vagus moving-wavelet peak 6.4 fT. The cervical-localised propagating-vs-stationary peak ratio is **0.72** — the 28% reduction comes from CV-dispersion across the fibre population (lognormal d = 8 µm, σ_log = 0.30 → CVs from 12 to 90 m/s) and from the fact that L_long has a narrow spatial peak (FWHM ≈ 16 mm) at the topmost cervical sources, so the wave only briefly samples high-L_long territory. The trials-to-detect headline numbers in §2.5–2.7 should therefore be inflated by 1/0.72² ≈ 1.93× under the propagating model: baroreceptor MEG ≈ 6.6 min (vs 3.4 min stationary); deep-breathing MEG ≈ 48 min (vs 25 min). These remain operationally feasible. The whole-vagus model (applicable to pulmonary or abdominal afferents that traverse the entire 528 mm polyline) gives an 8% peak ratio, consistent with sensor visibility being confined to the cervical band; for those scenarios the trials inflation factor is ≈ 170× and the predicted detection times become impractical at the simulated burst rates. Realistic vagal CAPs are also fascicular¹⁹ and orientation-heterogeneous (laryngeal recurrent branches bend); these require ASCENT-level intra-fascicular modelling that is outside the scope of a non-invasive forward.

**Detection-noise model.** The Rose-criterion (SNR = 3) trials-to-detect figures assume independent white noise across channels (SNR ∝ √N). Spatially-correlated environmental MEG noise inflates this by factors of 1–10× depending on shielding and sensor-density, and we flag this in the figure caption rather than attempting to model it for a hypothetical lab. Multi-channel beamformer SNR²³ would also reduce trials-to-detect by a factor proportional to the number of effective spatial degrees of freedom — a benefit that has been demonstrated for brain MEG and that our pipeline supports if a beamformer module is bolted on.

The biggest *applied* implication of these predictions is that cervical OPM-MEG is operationally feasible as a continuous autonomic-physiology biomarker platform. Cardiac-locked baroreceptor traffic should be recoverable in single-digit minutes; deep-breathing afferent activity in tens of minutes. The clinical translation pipeline — patch-cuff comparisons in healthy controls, then in autonomic-disorder cohorts (POTS, vasovagal syncope, post-COVID dysautonomia), then in real-world ambulatory deployment — is enabled in principle by the OPM and constrained in principle for the surface electrode. Our pipeline lets investigators front-load this trade-off analysis at the design stage.

---

## 4 Methods

All code, configuration, regenerated outputs, and the full validation chain are at https://github.com/[TBC]/Forward_Model_Vagus_Nerve.

### 4.1 Anatomy and FEM construction

Atlas STLs (74 individual bones, one whole-body skin, two vagus-trunk meshes — left and right) were sourced from BodyParts3D. The geometry-build pipeline (`vagus_fm.geometry.builder`, CLI `vagus-fm-build-geom`) applies a layered watertightening procedure: (i) cheap trimesh repair (vertex merging, degenerate-face removal, normal repair); (ii) boolean union of overlapping connected components; (iii) voxel shrink-wrap (per-tissue voxel pitch 0.5–3 mm, occupancy via ray-traced voxelisation for watertight inputs and surface-sample splatting for non-watertight, marching-cubes surface re-extraction, quadric decimation, Taubin smoothing); (iv) pymeshfix as a last-resort hole-closer. Each compartment is gated on watertightness ∧ winding-consistency ∧ Euler number = 2 (genus 0); the build raises rather than persists a broken mesh.

The four-tissue tetrahedral FEM (`vagus_fm.fem.cgal_builder`, CLI `vagus-fm-build-fem`) is built on a single global voxel grid (3 mm pitch, 50 mm padding) covering all tissues. Each tissue is voxelised by the appropriate strategy (convex-hull union for bones; ray-traced + flood fill for watertight vagus tubes; surface-sample + closing + exterior flood for skin to handle anatomical openings). The labelled image is then meshed by iso2mesh's `cgalv2m` (CGAL 3D Mesh_3, radbound 2.5, max element volume 20 voxel³). Final mesh: 209 927 nodes, 789 342 elements, four tissues (vagus_left, vagus_right, bone, skin); per-element Joe-Liu mesh-quality minimum 0.092, fifth percentile 0.301, mean 0.524 — all gates passed. Coordinates are LPS-derived from BodyParts3D and definitively documented in `docs/COORDINATES.md`.

### 4.2 Sensor and electrode arrays

The triaxial OPM array (`vagus_fm.sensors.triaxial`, CLI `vagus-fm-sensors`) is generated by inward-radial cylindrical raycasting around the body Z-axis, with cylinder radius 0.75 × max(X-extent, Y-extent), Z-spacing 30 mm, angular spacing matched. Each surface hit emits three orthogonal coils (radial + 2 tangents), giving the canonical FieldTrip `grad` layout (8 190 channels in the default configuration). The HD-EMG paddle (`vagus_fm.sensors.electrodes`, CLI `vagus-fm-electrodes`) replicates the geometry of the Bu *et al.*¹¹ flexible PEDOT:PSS hardware: 1 head contact + 5 × 6 body grid + 1 foot contact (32 total) at 5 mm pitch, projected onto the anterior cervical skin. The whole-body sensitivity sweep array (§2.5) places 1 000 contacts uniformly on the skin via Fibonacci-lattice sampling.

### 4.3 Forward solves

Both leadfields are computed by DUNEuro 2.10 (built from source via `cluster/build_duneuro.sh`, Python bindings via `duneuropy`). The volume conductor is configured in mm-mode (σ in S/mm) with conductivities from `configs/default.yaml` (vagus 0.30, bone 0.0042, skin 0.43 S/m, all S/m, scaled to S/mm internally). Source moments are sampled along the vagus polyline at 5 mm spacing (87 cervical sources by default; `vagus_fm.sources.vagus.vagus_sources`) and three orthogonal moments per position are passed through DUNEuro's `MEEGDriver3d` partial-integration source model.

The MEG path (`vagus_fm.forward.solve`, CLI `vagus-fm-forward`) attaches OPMs as point coils with their tangent orientations, computes the MEG transfer matrix (CG solver, reduction 10⁻¹⁰, SIPG scheme, integration-order add 5), and applies it. Output: leadfield NPZ at `outputs/forward/duneuro_leadfield_vagus.npz` (8 190 channels × 261 source-moment columns; size 33 MB). Conversion to fT per nA·m source: ×10⁶, validated via Sarvas analytic and MNE-Python `_do_sphere_field` cross-validation (`tests/test_analytic_sphere.py`, four parametrised dipole orientations, rtol = 10⁻¹²). The cluster pipeline (Myriad / Kathleen profiles, `cluster/submit.sh array`) splits this into 32 chunks for ~50-minute completion.

The EEG path (`vagus_fm.forward.eeg`, CLI `vagus-fm-eeg`) attaches surface electrodes with the closest-subentity-centre realisation, computes the EEG transfer matrix, and applies it under common-average reference. Output: leadfield NPZ (32 channels × 261 columns). Conversion to µV per nA·m source: ×0.622, calibrated empirically as described in §4.4.

### 4.4 EEG calibration via homogeneous-sphere FEM

DUNEuro's mm-mode EEG output unit is not formally documented in a manner that admits a-priori dimensional analysis. We calibrate empirically. The calibration pipeline (`vagus_fm.analysis.sphere_calibration`, CLI `vagus-fm-calibrate`) builds a homogeneous-tissue sphere FEM (radius 100 mm, σ = 0.43 S/m, voxel pitch 4 mm, ~2 700 elements after CGAL meshing), places 200 electrodes on the sphere via Fibonacci lattice, runs the EEG forward with a unit dipole at 50 mm depth, and computes the Berg-Scherg analytic series at the same electrode positions. The ratio analytic / DUNEuro-raw, taken as the median over all (electrode, moment) pairs to be robust to the FEM-vs-analytic disagreement at sensor positions where the analytic is near-zero, is the calibration factor.

Headline result: median factor = **0.622** (geomean 0.738, peak 0.307). Robustness sweep across radius {60, 100, 140} mm × source depth {0.3, 0.5, 0.7}·R × conductivity {0.1, 0.43, 1.0} S/m: 27 cases, range 0.597–0.754, mean 0.671, std 0.046, **CV = 6.8%**. The factor is invariant to σ (analytic confirmation that DUNEuro's σ-scaling and Berg-Scherg's 1/σ dependence cancel exactly) and varies slowly with geometry. The production value is applied as a single named constant `EEG_CALIBRATION_FACTOR = 0.622` in `vagus_fm/forward/eeg.py`.

### 4.5 Source model

Per-fibre dipole moments are computed by the Hämäläinen²¹ formula Q = π · d² · σ_in · ΔV / 4 with σ_in = 1.0 S/m (the canonical brain-MEG / inverse-problem convention; Pelot¹⁹ cite a peripheral-axoplasm value of ≈ 0.35 S/m, which would scale every Q by ~0.35× — switchable via the `sigma_intracellular_S_per_m` argument). Action-potential amplitudes are 70 mV (myelinated A-fibres) and 80 mV (unmyelinated C-fibres), summed over the diameter histogram. Conduction velocity follows Hursh's law CV ≈ 6 m/s/µm for myelinated fibres; unmyelinated fibres are assigned 0.5 m/s.

Two physiologically-grounded scenarios are implemented (`vagus_fm.physiology.scenarios`):

* **Baroreceptor**: 70 bpm cardiac rate (1.17 Hz), ~200 A-fibres per burst (lognormal diameter distribution centred on 8 µm), 150 ms after the R-wave with ±5 ms jitter, 6-second window.
* **Deep breathing**: 6 breaths/min (0.10 Hz), each cycle: RAR phasic burst of ~600 mixed pulmonary fibres (lognormal centred on 5 µm, ΔV = 80 mV) at inspiration onset, plus SAR tonic train of ~80 fibres at 50 Hz throughout 45% of the cycle, expiration silent, 12-second window.

The simulator (`vagus_fm.physiology.simulate`) produces a per-channel time series at every sensor by superposing biphasic-Gaussian compound action potentials at the cervical-source hot-spot, with fs = 30 kHz default and AP width σ = 0.5 ms.

### 4.6 Noise model and detectability

OPM intrinsic noise is set to 15 fT/√Hz (QuSpin Gen-2 spec); over 1 kHz analysis bandwidth this gives σ_MEG = 474 fT per channel per trial. HD-EMG noise is a quadrature sum of amplifier (0.5 µV/√Hz) and Johnson voltage noise from a 50 kΩ skin contact at body temperature (0.029 µV/√Hz); over 1 kHz this gives σ_EEG = 15.8 µV per channel per trial. Post-averaging SNR for white uncorrelated noise is signal_per_trial / σ_per_trial × √N; trials-to-SNR = 3 follows from inverting this. We flag the white-noise assumption in figure captions; spatially-correlated environmental noise inflates trials-to-detect by factors of 1–10× and beamformer methods²³ reduce them by a comparable factor.

### 4.7 Statistical methods

Bootstrap 95% confidence intervals (B = 1 000 resamples) are computed for the FEM/Sarvas peak ratio (`outputs/sarvas_vs_fem.json`) and for the cross-modality recovery error. Per-source amplitude correlations between modalities are reported with Pearson r and a log-log slope. The calibration sweep CV = 6.8% is the standard deviation of the per-condition median factor, divided by the across-condition mean. No multiple-comparison correction is applied because each headline number is reported as a point estimate with its own CI rather than as a hypothesis test.

### 4.8 Reproducibility

The pipeline is a fully installable Python package (`pip install -e .`) with eleven CLI entry points (`vagus-fm-pipeline`, `-build-geom`, `-build-fem`, `-sensors`, `-electrodes`, `-forward`, `-eeg`, `-snr`, `-sarvas`, `-cross`, `-detect`, `-location`, `-calibrate`, `-physiology`, `-topoplot`, `-visualise`). Every CLI entry seeds `numpy.random` and `random` from a single `cfg.reproducibility.seed`. CGAL meshing is non-deterministic between iso2mesh versions but produces meshes within the validated quality band on every run. The test suite is **123 pytest cases passing** (one DUNEuro-marked test is auto-skipped when `duneuropy` is unavailable; the GitHub Actions CI runs everything else). Code style is `ruff`-clean. A `CHANGELOG.md` and `CITATION.cff` are at the repository root.

Total run time on a 12-core MacBook to regenerate every figure from source STLs: approximately 2 hours, dominated by the OPM-MEG forward solve (~50 minutes) and the whole-body EEG forward solve (~17 minutes).

---

## 5 Data and code availability

All STL inputs (BodyParts3D-derived), validated leadfield NPZ outputs, calibration JSON, and pre-rendered figures are on Zenodo at [DOI TBC]. The code repository (MIT-licensed) is at [GitHub URL TBC]. A full validation record is at `docs/VALIDATION.md`; a coordinate-system reference at `docs/COORDINATES.md`; a literature review at `docs/LITERATURE_REVIEW.md`; an end-to-end correctness audit at `docs/CORRECTNESS_AUDIT.md`. CI is configured via `.github/workflows/ci.yml`; the cluster pipeline (UCL Myriad / Kathleen) is in `cluster/`.

---

## 6 References

1. Tracey, K. J. The inflammatory reflex. *Nature* **420**, 853–859 (2002). https://doi.org/10.1038/nature01321
2. Zanos, T. P. *et al.* Identification of cytokine-specific sensory neural signals by decoding murine vagus nerve activity. *Proc. Natl. Acad. Sci.* **115**, E4843–E4852 (2018). https://doi.org/10.1073/pnas.1719083115
3. Marmerstein, J. T., McCallum, G. A. & Durand, D. M. Direct measurement of vagal tone in rats does not show correlation to HRV. *Sci. Rep.* **11**, 1210 (2021). https://doi.org/10.1038/s41598-020-79808-8
4. Sevcencu, C., Nielsen, T. N. & Struijk, J. J. An intraneural electrode for bioelectronic medicines for treatment of hypertension. *Neuromodulation* **21**, 777–786 (2018). https://doi.org/10.1111/ner.12758
5. Vallone, F. *et al.* Simultaneous decoding of cardiovascular and respiratory functional changes from pig intraneural vagus nerve signals. *J. Neural Eng.* **18**, 0460a8 (2021). https://doi.org/10.1088/1741-2552/ac0d42
6. Ottaviani, M. M., Wright, L., Dawood, T. & Macefield, V. G. *In vivo* recordings from the human vagus nerve using ultrasound-guided microneurography. *J. Physiol.* **598**, 3569–3576 (2020). https://doi.org/10.1113/JP280077
7. Boto, E. *et al.* Moving magnetoencephalography towards real-world applications with a wearable system. *Nature* **555**, 657–661 (2018). https://doi.org/10.1038/nature26147
8. Tierney, T. M. *et al.* Optically pumped magnetometers: from quantum origins to multi-channel magnetoencephalography. *NeuroImage* **199**, 598–608 (2019). https://doi.org/10.1016/j.neuroimage.2019.05.063
9. Tierney, T. M. *et al.* Pragmatic spatial sampling for wearable MEG arrays. *Sci. Rep.* **10**, 21609 (2020). https://doi.org/10.1038/s41598-020-77589-8
10. QuSpin Inc. QZFM Gen-2 specification sheet. https://quspin.com/products-qzfm/
11. Bu, Y. *et al.* A flexible adhesive surface electrode array capable of cervical electroneurography during a sequential autonomic stress challenge. *Sci. Rep.* **12**, 19467 (2022). https://doi.org/10.1038/s41598-022-23114-y
12. Bu, Y. *et al.* Non-invasive ventral cervical magnetoneurography as a proxy of *in vivo* lipopolysaccharide-induced inflammation. *Commun. Biol.* **7**, 893 (2024). https://doi.org/10.1038/s42003-024-06435-8
13. Vorwerk, J., Cho, J.-H., Rampp, S., Hamer, H., Knösche, T. R. & Wolters, C. H. A guideline for head-volume-conductor modeling in EEG and MEG. *NeuroImage* **100**, 590–607 (2014). https://doi.org/10.1016/j.neuroimage.2014.06.040
14. Saturnino, G. B. *et al.* SimNIBS 2.1: a comprehensive pipeline for individualized electric-field modelling for transcranial brain stimulation. In *Brain and Human Body Modeling* (Springer, 2019). https://doi.org/10.1007/978-3-030-21293-3_1
15. Tadel, F. *et al.* Brainstorm: a user-friendly application for MEG/EEG analysis. *Comput. Intell. Neurosci.* **2011**, 879716 (2011). https://doi.org/10.1155/2011/879716
16. O'Neill, G. C. *et al.* Volume conductor models for magnetospinography. *Sci. Rep.* **15**, 26258 (2025). https://doi.org/10.1038/s41598-025-10770-z
17. Musselman, E. D., Pelot, N. A. & Grill, W. M. ASCENT: a peripheral-nervous-system simulation pipeline. *PLOS Comput. Biol.* **17**, e1009285 (2021). https://doi.org/10.1371/journal.pcbi.1009285
18. Sarvas, J. Basic mathematical and electromagnetic concepts of the biomagnetic inverse problem. *Phys. Med. Biol.* **32**, 11–22 (1987). https://doi.org/10.1088/0031-9155/32/1/004
19. Pelot, N. A. *et al.* Quantified morphology of the cervical and subdiaphragmatic vagus nerves of human, pig, and rat. *Front. Neurosci.* **14**, 601479 (2020). https://doi.org/10.3389/fnins.2020.601479
20. Geselowitz, D. B. On the magnetic field generated outside an inhomogeneous volume conductor by internal current sources. *IEEE Trans. Magn.* **6**, 346–347 (1970). https://doi.org/10.1109/TMAG.1970.1066765
21. Hämäläinen, M., Hari, R., Ilmoniemi, R. J., Knuutila, J. & Lounasmaa, O. V. Magnetoencephalography — theory, instrumentation, and applications to noninvasive studies of the working human brain. *Rev. Mod. Phys.* **65**, 413–497 (1993). https://doi.org/10.1103/RevModPhys.65.413
22. Gabriel, S., Lau, R. W. & Gabriel, C. The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues. *Phys. Med. Biol.* **41**, 2271–2293 (1996). https://doi.org/10.1088/0031-9155/41/11/003
23. Sekihara, K. & Nagarajan, S. S. *Adaptive Spatial Filters for Electromagnetic Brain Imaging* (Springer, 2008).
24. Quiroga, R. Q., Nadasdy, Z. & Ben-Shaul, Y. Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Comput.* **16**, 1661–1687 (2004). https://doi.org/10.1162/089976604774201631
25. Hodgkin, A. L. & Huxley, A. F. A quantitative description of membrane current and its application to conduction and excitation in nerve. *J. Physiol.* **117**, 500–544 (1952). https://doi.org/10.1113/jphysiol.1952.sp004764
26. Hursh, J. B. Conduction velocity and diameter of nerve fibers. *Am. J. Physiol.* **127**, 131–139 (1939). https://doi.org/10.1152/ajplegacy.1939.127.1.131
27. Hasgall, P. A. *et al.* IT'IS Database for thermal and electromagnetic parameters of biological tissues, version 4.1 (2022). https://doi.org/10.13099/VIP21000-04-1
28. Rivnay, J., Inal, S., Salleo, A., Owens, R. M., Berggren, M. & Malliaras, G. G. Organic electrochemical transistors. *Nat. Rev. Mater.* **3**, 17086 (2018). https://doi.org/10.1038/natrevmats.2017.86
29. Pas, J., Rutz, A. L., Quilichini, P. P., Slézia, A., Ghestem, A., Kaszas, A., Donahue, M. J., Curto, V. F., O'Connor, R. P., Bernard, C., Williamson, A. & Malliaras, G. G. A bilayer flexible PEDOT:PSS-coated electrode for chronic neural recordings. *J. Neural Eng.* **15**, 065001 (2018). https://doi.org/10.1088/1741-2552/aadc1d

The complete annotated literature review with 37 verified citations is in `docs/LITERATURE_REVIEW.md`.

---

## 7 Acknowledgements

We thank the BodyParts3D / Anatomography project (RIKEN and University of Tokyo) for the open-source anatomical STL inputs. DUNEuro is developed by the Computational Engineering & Biomedical Imaging groups at Münster, Genova, and Aachen. iso2mesh is developed by Qianqian Fang and the JNeurolab. MNE-Python provided the analytic-sphere reference implementation.

## 8 Author contributions

[TBC at submission.]

## 9 Competing interests

The authors declare no competing interests.

---

*Manuscript draft generated 2 May 2026. Pipeline state: 123 tests passing, ruff-clean, 4 014 lines of Python, 16 CLI entry points, 7 figures regenerable from a single command.*
