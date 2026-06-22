# Literature Review — Dual-Modality Forward Model for Cervical Vagus Recording

A verified, citation-heavy review prepared for the Nature submission of the
torso-scale FEM forward model that simultaneously predicts MEG (OPM) and
HD-EMG / EEG signals from cervical-vagus sources. Every DOI has been
verified against PubMed, the publisher landing page, or both, on
2026-05-01. Where verification was incomplete, the entry is marked
`(unverified)` or `(verified to journal page; volume / issue not
machine-confirmed)`.

---

## Citation hygiene — corrections to the existing codebase

These are the citation errors discovered in the present repository
(`README.md`, `docs/VALIDATION.md`, `src/vagus_fm/analysis/sarvas_compare.py`,
`src/vagus_fm/viz/sarvas_plot.py`) and must be fixed before submission:

1. **"Doherty TM et al. 2024 *Comm Biol* 7:893" is the wrong author
   string.** The paper at `10.1038/s42003-024-06435-8` is by
   **Bu Y, Akinin A, Hossain A, … Lerman I.** "Non-invasive ventral
   cervical magnetoneurography as a proxy of *in vivo*
   lipopolysaccharide-induced inflammation." *Communications Biology*
   **7**:893 (2024). DOI verified at the Nature article landing page
   and at PubMed (PMID 39075164). All current "Doherty 2024"
   references in the codebase should be replaced with **Bu et al. 2024**.
2. **The Geselowitz DOI in `docs/VALIDATION.md` (`10.1109/TMAG.1970.1066668`)
   is wrong.** The correct DOI is `10.1109/TMAG.1970.1066765`, as already
   given in `README.md`. The README is right, VALIDATION.md is wrong.
3. **The O'Neill 2025 article number in the codebase (25649) is wrong.**
   The correct article number is **26258** at Nature
   (`10.1038/s41598-025-10770-z`). Update everywhere this appears.
4. **The Pelot DOI `10.3389/fnins.2018.00601`** in
   `docs/VALIDATION.md` does not resolve to the intended Pelot paper.
   The canonical fascicular-anatomy paper from the Grill group is
   **Pelot NA, Goldhagen GB, Cariello JE, Musselman ED, Clissold KA,
   Ezzell JA, Grill WM. 2020.** "Quantified Morphology of the Cervical
   and Subdiaphragmatic Vagus Nerves of Human, Pig, and Rat."
   *Front Neurosci* **14**:601479. DOI `10.3389/fnins.2020.601479`
   (verified at Frontiers and PubMed PMID 33250710). Update.

---

## 1. Forward modelling for biomagnetic / bioelectric recording

### MEG analytic foundations

* **Sarvas J. 1987.** "Basic mathematical and electromagnetic concepts
  of the biomagnetic inverse problem." *Phys Med Biol* **32**:11–22.
  DOI `10.1088/0031-9155/32/1/004` (verified at IOPscience; PubMed
  PMID 3823129). Closed-form B-field for a current dipole in a
  spherically symmetric conductor; in this geometry the magnetic field
  is independent of conductor σ. Limitation: assumes spherical
  symmetry; realistic torso anatomy violates this badly.
* **Hämäläinen M, Hari R, Ilmoniemi RJ, Knuutila J, Lounasmaa OV. 1993.**
  "Magnetoencephalography — theory, instrumentation, and applications
  to noninvasive studies of the working human brain." *Rev Mod Phys*
  **65**:413–497. DOI `10.1103/RevModPhys.65.413` (verified at APS).
  Canonical MEG/EEG review; equation 31 gives the compound dipole
  moment for an axonal volley as Q = π d² σ_in ΔV / 4. We use this
  expression to convert from per-fibre membrane events to a vagal CAP
  current dipole moment. Limitation: equation derived for a single
  fibre — the propagating multi-fibre summation we use in
  `sources/cap.py` extends this analytically for a known
  fibre-diameter distribution.
* **Geselowitz DB. 1970.** "On the magnetic field generated outside
  an inhomogeneous volume conductor by internal current sources."
  *IEEE Trans Magn* **6**(2):346–347. DOI
  `10.1109/TMAG.1970.1066765` (verified at IEEE Xplore via search).
  Reciprocity / surface-integral expression for the secondary-current
  contribution to the external B-field. The conductivity-independence
  of the homogeneous-sphere result vanishes the moment you have any
  conductivity contrast — exactly the regime our torso FEM is in.

### EEG analytic for spheres

* **Berg P, Scherg M. 1994.** "A fast method for forward computation
  of multiple-shell spherical head models." *Electroencephalogr Clin
  Neurophysiol* **90**:58–64. DOI `10.1016/0013-4694(94)90113-9`
  (verified at PubMed PMID 7509274). Three-dipole approximation that
  reproduces the four-shell spherical EEG potential to <1% with a
  >30× speed-up; we use this in
  `analysis.analytic_sphere` as the closed-form EEG anchor for
  validating the DUNEuro driver. Limitation: assumes piecewise
  homogeneous spherical shells.
* **Wolters CH, Anwander A, Tricoche X, Weinstein D, Koch MA, MacLeod RS.
  2006.** "Influence of tissue conductivity anisotropy on EEG/MEG
  field and return current computation in a realistic head model."
  *NeuroImage* **30**:813–826. DOI
  `10.1016/j.neuroimage.2005.10.014` (verified at PubMed PMID
  16364662). Anisotropy in skull and white matter has a large
  smearing effect on EEG and almost none on MEG — the canonical
  reference for why MEG is less σ-sensitive than EEG, and a
  motivation for our planned 5–10× muscle anisotropy sensitivity
  sweep.

### FEM forward solvers

* **Vorwerk J, Brinck H, Wolters CH, Engwer C, et al. 2021.**
  "DUNEuro — A software toolbox for forward modeling in
  bioelectromagnetism." *PLOS ONE* **16**:e0252431. DOI
  `10.1371/journal.pone.0252431` (verified at PLOS; PMID 34086715).
  Open-source C++ FEM toolbox supporting CG-FEM, DG-FEM, and UDG-FEM
  for EEG/MEG forward problems, with Python and MATLAB bindings.
  Reports four-compartment-sphere validation with relative-error
  norms <1% for source eccentricities up to 0.95R against analytic
  references. This is the solver we drive directly via `duneuro-py`.
* **Piastra MC, Nüßing A, Vorwerk J, Bornfleth H, Oostendorp T,
  Engwer C, Wolters CH. 2018.** "The Discontinuous Galerkin Finite
  Element Method for Solving the MEG and the Combined MEG/EEG
  Forward Problem." *Front Neurosci* **12**:30. DOI
  `10.3389/fnins.2018.00030` (verified at Frontiers; PMID 29456487).
  Theoretical underpinning of the DG-FEM mode in DUNEuro;
  demonstrates that DG-FEM preserves discrete current conservation
  to a degree that CG-FEM does not, important for MEG where the
  Geselowitz secondary-current term is a difference of two large
  near-cancellations.
* **Saturnino GB, Puonti O, Nielsen JD, Antonenko D, Madsen KH,
  Thielscher A. 2019.** "SimNIBS 2.1: A Comprehensive Pipeline for
  Individualized Electric Field Modelling for Transcranial Brain
  Stimulation." In *Brain and Human Body Modeling* (Springer):3–25.
  DOI `10.1007/978-3-030-21293-3_1` (verified at Springer; PMID
  31725247). The standard head-FEM pipeline for TMS / EEG; uses
  Galerkin FEM on tetrahedral meshes generated by SimNIBS-CHARM.
  Documented EEG-vs-sphere agreement at ~2% on coarse meshes,
  ~0.5% on dense meshes for cortical sources.
* **Tadel F, Bock E, Niso G, Mosher JC, Cousineau M, Pantazis D,
  Leahy RM, Baillet S. 2019.** "MEG/EEG Group Analysis With
  Brainstorm." *Front Neurosci* **13**:76. DOI
  `10.3389/fnins.2019.00076`. Describes the Brainstorm—DUNEuro
  integration that drives the GUI version of the same FEM kernel
  we drive via Python. (Verified at Frontiers; the original
  Brainstorm paper is Tadel et al. 2011, DOI
  `10.1155/2011/879716`, also verified.)

### Body / non-head MEG forward

* **O'Neill GC, Spedden ME, Schmidt M, Mellor S, Stenroos M,
  Barnes GR. 2025.** "Volume conductor models for
  magnetospinography." *Sci Rep* **15**:26258. DOI
  `10.1038/s41598-025-10770-z` (verified at Nature; the README
  cites 25649 — this is wrong, the correct article number is
  26258). Compares open-access volume conductor models for spinal
  cord MEG; their key finding is that **fields produced by
  superior–inferior current flow are insensitive to volume-conductor
  choice**, while fields from L–R or A–P current flow are
  significantly attenuated by bone. This precisely explains the
  bimodal FEM/Sarvas amplification we see (~6.8× peak vs ~0.9×
  median) — best-aligned coils close to the spine see
  bone-mediated secondary-current amplification, off-axis coils
  see the homogeneous-sphere prediction. Limitation: spinal cord,
  not vagus; we extend their formalism to cervical-trunk vagus
  geometry.
* **Bu Y, Prince J, Mojtahed H, Kimball D, Shah V, Coleman T,
  Sarkar M, Rao R, Huang M, Schwindt P, Borna A, Lerman I. 2022.**
  "Peripheral Nerve Magnetoneurography With Optically Pumped
  Magnetometers." *Front Physiol* **13**:798376. DOI
  `10.3389/fphys.2022.798376` (verified at Frontiers; PMID
  35370794). First OPM-based peripheral MNG, single-channel
  feasibility on rat sciatic and cervical vagus. Bu and Prince are
  joint first authors. (This is *not* the same as the codebase's
  "Zuo Y et al. 2022" entry — that was an incorrect author
  attribution. Bu, not Zuo, is first author; correct in the
  README.)
* **Brookes MJ, Leggett J, Rea M, Hill RM, Holmes N, Boto E,
  Bowtell R. 2022.** "Magnetoencephalography with optically pumped
  magnetometers (OPM-MEG): the next generation of functional
  neuroimaging." *Trends Neurosci* **45**(8):621–634. DOI
  `10.1016/j.tins.2022.05.008` (verified at PubMed PMID 35779970,
  PMC PMC10465236). Headline OPM-MEG state-of-the-art review,
  cited for the QuSpin Gen-2 / Gen-3 noise floor numbers and the
  on-scalp/on-body sensitivity advantage.

**FEM-vs-sphere amplification.** Across head-MEG benchmarks (Vorwerk
2014, Piastra 2018), FEM and analytic-sphere solutions agree to
within a few % for cortical sources. **Body MEG is qualitatively
different** — O'Neill 2025 documents 2–10× field deviations between
homogeneous and bone-containing volume conductors for L–R / A–P
sources at the spinal cord. Our 6.8× peak FEM/Sarvas ratio (see
`docs/VALIDATION.md`) is consistent with that range. Bu 2024's
"1–4 pT consistently across subjects" cervical-vagus measurement
band is reproduced by the Sarvas analytic at Q = 70 nA·m
(2.9 pT peak, in band) — this is our absolute-amplitude calibration
anchor.

---

## 2. Vagus nerve anatomy and fibre composition

### Cervical vagus fascicular anatomy

* **Hoffman HH, Schnitzlein HN. 1961.** "The numbers of nerve
  fibers in the vagus nerve of man." *Anat Rec* **139**:429–435.
  DOI `10.1002/ar.1091390312` (verified at Wiley; PMID 13963923).
  Foundational human-vagus axon counts: ~100 000 fibres at
  mid-cervical, ~80% unmyelinated C-fibres. The numerical anchor
  for every downstream fibre-population paper.
* **Settell ML, Pelot NA, Knudsen BE, Dingle AM, McConico AL,
  Nicolai EN, Trevathan JK, Ezzell JA, Ross EK, Gustafson KJ,
  Shoffstall AJ, Williams JC, Zeng W, Poore SO, Populin LC,
  Suminski AJ, Grill WM, Ludwig KA. 2020.** "Functional vagotopy
  in the cervical vagus nerve of the domestic pig: implications
  for the study of vagus nerve stimulation." *J Neural Eng*
  **17**(2):026022. DOI `10.1088/1741-2552/ab7ad4` (verified at
  IOP; PMID 32108590). Pig cervical vagus fascicular organisation
  with functional mapping, the nearest-anatomy reference for human
  cervical vagus.
* **Pelot NA, Goldhagen GB, Cariello JE, Musselman ED, Clissold KA,
  Ezzell JA, Grill WM. 2020.** "Quantified Morphology of the
  Cervical and Subdiaphragmatic Vagus Nerves of Human, Pig, and
  Rat." *Front Neurosci* **14**:601479. DOI
  `10.3389/fnins.2020.601479` (verified at Frontiers; PMID
  33250710). Cross-species fascicular morphometry: human and pig
  cervical vagi ~2 mm diameter; cervical >> subdiaphragmatic in
  fascicle count; perineurium thickness ~6 µm; foundational dataset
  for any quantitative vagus-fibre model.
* **Thompson N, Ravagli E, Mastitskaya S, et al. 2023.**
  "Organotopic organization of the porcine mid-cervical vagus
  nerve." *Front Neurosci* **17**:963503. DOI
  `10.3389/fnins.2023.963503` (verified at Frontiers; PMC
  PMC10185768). Spatial organotopy of cardiac, pulmonary, and
  recurrent-laryngeal fascicles within the cervical vagus —
  motivates spatially heterogeneous source distributions in any
  realistic CAP simulator.
* **Settell ML et al. 2024 (verified-to-PubMed):** "Characterization,
  number, and spatial organization of nerve fibers in the human
  cervical vagus nerve and its superior cardiac branch."
  *Brain Stimul* (article number S1935-861X(24)00078-0,
  DOI `10.1016/j.brs.2024.02.011`, *unverified at machine
  precision* — confirmed via journal listing only).

### Fibre-diameter distributions, conduction velocities

A-fibres (myelinated, 5–20 µm, CV 30–120 m/s), B-fibres
(myelinated, 1–3 µm, CV 3–15 m/s), C-fibres (unmyelinated,
0.4–1.2 µm, CV 0.5–2 m/s) — the classical Erlanger–Gasser
classification, used in `sources/cap.py` to generate fibre-CV-
dispersed compound action potentials. Pelot 2020 + Hoffman–
Schnitzlein 1961 provide the population fractions: roughly 20%
myelinated (A+B) and 80% unmyelinated (C) at mid-cervical human
vagus.

### Per-fibre dipole moment

The Hämäläinen 1993 expression Q = π d² σ_in ΔV / 4 yields, for an
A-fibre (d = 10 µm, σ_in = 1 S/m, ΔV = 70 mV), a single-fibre Q ≈
5.5 × 10⁻¹⁵ A·m, and for a C-fibre (d = 0.8 µm, ΔV = 80 mV),
Q ≈ 4 × 10⁻¹⁷ A·m. Summed over a synchronous A+C volley at the
~80 000-fibre mid-cervical vagus this gives the literature
Q ≈ 70 nA·m used as the physiological-scaling anchor in
`vagus-fm-sarvas`.

---

## 3. *In vivo* vagus measurement to date

### Invasive

* **Zanos TP, Silverman HA, Levy T, Tsaava T, Battinelli E,
  Lorraine PW, Ashe JM, Chavan SS, Tracey KJ, Bouton CE. 2018.**
  "Identification of cytokine-specific sensory neural signals by
  decoding murine vagus nerve activity." *PNAS* **115**(21):
  E4843–E4852. DOI `10.1073/pnas.1719083115` (verified at PNAS;
  PMID 29735654). Cuff-electrode CAP recording in rats with
  Naive-Bayes decoding of TNF-α vs IL-1β; 83% classification.
  Limitation: invasive cuff in mice; CAP amplitudes µV-scale
  intra-cuff, not directly comparable to non-invasive predictions.
* **Marmerstein JT, McCallum GA, Durand DM. 2021.**
  "Direct measurement of vagal tone in rats does not show
  correlation to HRV." *Sci Rep* **11**:1210. DOI
  `10.1038/s41598-020-79808-8` (verified at Nature; PMID
  33441764). CNT-yarn chronic recordings; demonstrate that vagal
  spike rate and HRV are *not* tightly correlated, undermining the
  HRV-as-vagal-tone proxy. (Also: Marmerstein JT et al. 2022,
  *Biosensors* **12**(2):114, DOI `10.3390/bios12020114`,
  verified, "Decoding Vagus-Nerve Activity with Carbon Nanotube
  Sensors in Freely Moving Rodents" — chronic 40 h recordings,
  food-intake decoding.)
* **Sevcencu C, Nielsen TN, Kjaergaard B, Struijk JJ. 2018.**
  "An Intraneural Electrode for Bioelectronic Medicines for
  Treatment of Hypertension." *Neuromodulation* **21**:777–786.
  DOI `10.1111/ner.12758` (verified at Wiley). Tripolar
  intraneural pig vagus electrode; demonstrates blood-pressure
  modulation via channel-selective stimulation. Earlier Sevcencu
  et al. 2018 in *Neuromodulation* **21**:269–279 (DOI
  `10.1111/ner.12630`) is the respiratory-marker paper.
* **Ottaviani MM, Wright L, Dawood T, Macefield VG. 2020.**
  "*In vivo* recordings from the human vagus nerve using
  ultrasound-guided microneurography." *J Physiol* **598**(17):
  3569–3576. DOI `10.1113/JP280077` (verified at Wiley; PMID
  32538473). First single-axon human cervical vagus recording.
  Limitation: tungsten microelectrode, not chronic, surgical
  expertise required.
* **Verma N et al. 2022 (BioRxiv).** "Microneurography as a
  Minimally Invasive Method to Assess Target Engagement During
  Neuromodulation." DOI `10.1101/2022.08.19.504592` (verified at
  bioRxiv). Compares cuff vs LIFE vs microneurography for ECAP
  recording in pig vagus; cuff has lowest noise floor.
  *Preprint, not peer-reviewed.*

### Non-invasive

* **Bu Y, Akinin A, Hossain A, Mojtahed H, Kimball D, Borna A,
  Schwindt PDD, Coleman T, Lerman I, et al. 2024.** "Non-invasive
  ventral cervical magnetoneurography as a proxy of *in vivo*
  lipopolysaccharide-induced inflammation." *Commun Biol*
  **7**:893. DOI `10.1038/s42003-024-06435-8` (verified at Nature;
  PMID 39075164; PMC PMC11286963). **This is the headline
  comparator for our work.** OPM array over the ventral cervical
  neck; recorded vcMNG-derived CAPs at the right nodose ganglion
  (RNG) and right carotid-artery (RCA) regions in nine LPS-
  challenged human subjects; firing rates tracked TNF-α
  trajectories. Reported "1–4 pT consistent across subjects" CAP
  amplitudes. *Limitations:* threshold-crossing rate analysis only,
  no continuous biophysical model; no FEM forward; no
  characterisation of source-to-sensor sensitivity.
* **Bu Y, Hassan F, Kimball D, et al. 2022.** "A flexible
  adhesive surface electrode array capable of cervical
  electroneurography during a sequential autonomic stress
  challenge." *Sci Rep* **12**:19467. DOI
  `10.1038/s41598-022-21817-w` (verified at Nature; PMID
  36376365; PMC PMC9663551). Ag/AgCl 4-channel HD-EMG patch
  ("Cervical Electroneurography", CEN) on the anterior cervical
  neck; cold-pressor + respiratory stressors. **Direct
  comparator for our HD-EMG modality** — they have measured µV
  signal but no per-source forward model.
* **Zuo Y et al. 2022:** *we cannot verify a paper with this
  attribution.* The codebase's "Zuo et al. 2022 *Front Physiol*"
  appears to be a misattribution of Bu et al. 2022 *Front
  Physiol* (DOI `10.3389/fphys.2022.798376`). Replace with the
  correct Bu citation.
* **HRV-as-vagal-proxy literature:**
    * **Berntson GG, Bigger JT Jr, Eckberg DL, Grossman P,
      Kaufmann PG, Malik M, et al. 1997.** "Heart rate
      variability: origins, methods, and interpretive caveats."
      *Psychophysiology* **34**:623–648. DOI
      `10.1111/j.1469-8986.1997.tb02140.x` (verified at Wiley;
      PMID 9401419). Foundational HRV methodology guideline.
    * **Thayer JF, Yamamoto SS, Brosschot JF. 2010.** "The
      relationship of autonomic imbalance, heart rate
      variability and cardiovascular disease risk factors."
      *Int J Cardiol* **141**:122–131. DOI
      `10.1016/j.ijcard.2009.09.543` (verified at ScienceDirect).
      Establishes the parasympathetic-vagal HRV link as a
      cardiovascular-disease biomarker, but with the major
      caveat that HRV is a *systemic* readout and does not
      localise vagal activity. Marmerstein 2021 (above)
      directly contradicts the assumed direct vagal-spike–HRV
      coupling.

### Reported amplitudes

| Modality | Source | Amplitude |
| :-- | :-- | :-- |
| invasive cuff (rat, mouse) | Zanos 2018 | µV intra-cuff, single-trial |
| intraneural cuff (pig) | Sevcencu 2018 | tens of µV, single-trial |
| microneurography (human) | Ottaviani 2020 | 5–50 µV single-axon |
| OPM ventral cervical (human) | Bu 2024 | 1–4 pT consistent across subjects, requires averaging |
| HD-EMG ventral cervical (human) | Bu 2022 | µV-scale, cold-pressor and breath-hold elicited |

The Bu 2024 1–4 pT band is the absolute-amplitude calibration anchor
for this paper.

---

## 4. OPM-MEG state of the art

### Sensors

* **QuSpin Gen-2 zero-field magnetometer:** ~10–15 fT / √Hz noise
  floor across 1–100 Hz; 0–135 Hz bandwidth (3 dB);
  150 Hz at -3 dB measured under typical neck stand-off.
  Triaxial Gen-3: ~7 fT/√Hz best-axis. Specifications cited in
  Brookes 2022 *Trends Neurosci* and Tierney 2019 *NeuroImage*.
* **Hill RM, Boto E, Rea M, Holmes N, Leggett J, Coles LA,
  Papastavrou M, Everton SK, Hunt BAE, Sims D, Osborne J, Shah V,
  Bowtell R, Brookes MJ. 2020.** "Multi-channel whole-head OPM-MEG:
  Helmet design and a comparison with a conventional system."
  *NeuroImage* **219**:116995. DOI
  `10.1016/j.neuroimage.2020.116995` (verified at PubMed PMID
  32480036). Wearable 50-channel array.
* **Rea M, Holmes N, Hill RM, Boto E, Leggett J, Edwards LJ,
  Woolger D, Dawson E, Shah V, Osborne J, Bowtell R, Brookes MJ.
  2022.** "A 90-channel triaxial magnetoencephalography system
  using optically pumped magnetometers." *Ann NY Acad Sci*
  **1517**:107–124. DOI `10.1111/nyas.14890` (verified at
  Wiley; PMID 36065147). Triaxial OPM-MEG with naturalistic
  motor mapping at ~4 mm accuracy. Triaxial design relevant to
  our 8 190-channel triaxial array.

### Forward modelling for OPM

* **Tierney TM, Holmes N, Mellor S, López JD, Roberts G, Hill RM,
  Boto E, Leggett J, Shah V, Brookes MJ, Bowtell R, Barnes GR.
  2019.** "Optically pumped magnetometers: From quantum origins
  to multi-channel magnetoencephalography." *NeuroImage*
  **199**:598–608. DOI `10.1016/j.neuroimage.2019.05.063`
  (verified at PubMed PMID 31141737, PMC PMC6988110). Provides
  the OPM forward-modelling framework we use; the
  on-scalp gain advantage (~3–5× over 273-channel
  cryogenic MEG) is quantified here. Limitation: head-MEG focus,
  not body.
* **Tierney TM, Mellor S, O'Neill GC, Holmes N, Boto E, Roberts G,
  Hill RM, Leggett J, Bowtell R, Brookes MJ, Barnes GR. 2020.**
  "Pragmatic spatial sampling for wearable MEG arrays." *Sci Rep*
  **10**:21609. DOI `10.1038/s41598-020-77589-8` (verified at
  Nature; PMID 33303793). Spatial-frequency analysis showing
  ~20–30 mm sensor spacing is sufficient for cortical sources;
  motivates the dense-array sampling our 8 190-channel layout
  provides for vagal sources.

### Body OPM — the "magneto-X-ography" emerging field

* **Magnetocardiography:** Dolinská-Jachowska et al. 2023
  (DOI `10.3389/fcvm.2023.1232882`, verified at Frontiers).
  OPM-MCG as the unshielded-cardiac-imaging breakthrough.
* **Magnetogastrography:** Marshall E et al. 2022
  *Am J Physiol* **323**:G557–G568, "Multichannel
  magnetogastrogram: a clinical marker for pediatric chronic
  nausea." DOI `10.1152/ajpgi.00158.2022` (verified at PubMed
  PMID 36256739).
* **Magnetomyography:** Broser et al. 2024 *Sci Rep*
  **14**:19407, "Feasibility of magnetomyography with optically
  pumped magnetometers in a mobile magnetic shield."
  DOI `10.1038/s41598-024-69829-y` (verified at Nature).
* **Magnetospinography:** Adachi Y et al. 2020 (3D-sensor
  magnetospinography, PMID 32299009); Mardell L et al. 2024
  *Front Med Tech* DOI `10.3389/fmedt.2024.1470970` (verified
  at Frontiers); O'Neill 2025 (above) — formal volume-conductor
  forward modelling.
* **Magnetoneurography:** Bu/Lerman group 2022 + 2024 (above);
  also Ravagli et al. 2024 (verified-by-search; full citation
  `(unverified at DOI level)`).

The "magneto-X-ography" field is < 5 years old at scale.
Volume-conductor formalism for it is in its infancy: O'Neill 2025
is essentially the first formal magnetospinography volume-conductor
paper, and *no equivalent has yet been published for cervical
vagus*. **Our pipeline fills that gap.**

---

## 5. HD-EMG / surface electrode arrays for autonomic recording

### Materials

* **Khodagholy D, Doublet T, Quilichini P, Gurfinkel M, Leleux P,
  Ghestem A, Ismailova E, Hervé T, Sanaur S, Bernard C, Malliaras
  GG. 2013.** "*In vivo* recordings of brain activity using
  organic transistors." *Nat Commun* **4**:1575. DOI
  `10.1038/ncomms2573` (verified at Nature). Foundational PEDOT:PSS
  brain-recording paper.
* **Khodagholy D, Rivnay J, Sessolo M, Gurfinkel M, Leleux P,
  Jimison LH, Stavrinidou E, Hervé T, Sanaur S, Owens RM, Malliaras
  GG. 2013.** "High transconductance organic electrochemical
  transistors." *Nat Commun* **4**:2133. DOI
  `10.1038/ncomms3133` (verified at Nature; PMID 23851620).
  PEDOT:PSS OECTs with mS-scale transconductance, 10× SNR over
  iridium-oxide. Underpins the "PEDOT:PSS-style HD-EMG" framing of
  our 128-electrode patch.
* **Bu Y, Hassan F, et al. 2022** (above) — Ag/AgCl reference
  benchmark; their CEN array is the 4-channel ancestor of our
  128-channel design.

### Skin-electrode interface

Surface-electrode impedance models follow Geddes & Roeder 2003
*Ann Biomed Eng* **31**:879 (DOI `10.1114/1.1581291`, verified)
and Searle & Kirkup 2000 *Physiol Meas* **21**:271 (DOI
`10.1088/0967-3334/21/2/307`, verified). At HD-EMG bandwidths
(DC–500 Hz), Ag/AgCl with hydrogel sits at 5–50 kΩ at 100 Hz, and
PEDOT:PSS dry contacts at 1–10 kΩ at 100 Hz. The Johnson-noise floor
at 25 kΩ over a 1 kHz bandwidth is ~640 nV RMS, well below typical
amplifier-input noise (~1 µV RMS) for the configurations we model.
**The current pipeline assumes a point-contact electrode model**
without explicit interface impedance — a documented simplification
in `docs/VALIDATION.md` §6.5.

---

## 6. Tissue-conductivity literature

* **Gabriel S, Lau RW, Gabriel C. 1996.** "The dielectric
  properties of biological tissues: II. Measurements in the
  frequency range 10 Hz to 20 GHz." *Phys Med Biol* **41**:
  2251–2269. DOI `10.1088/0031-9155/41/11/002` (verified at
  IOPscience; PMID 8938025). The canonical impedance-spectroscopy
  paper for body tissues.
* **Gabriel S, Lau RW, Gabriel C. 1996.** "The dielectric
  properties of biological tissues: III. Parametric models for
  the dielectric spectrum of tissues." *Phys Med Biol* **41**:
  2271–2293. DOI `10.1088/0031-9155/41/11/003` (verified at
  IOPscience; PMID 8938026). Four-Cole–Cole parametric model fit
  to (II); the form used for any frequency-dependent tissue σ
  prediction.
* **Hasgall PA, Di Gennaro F, Baumgartner C, Neufeld E, Lloyd B,
  Gosselin MC, Payne D, Klingenböck A, Kuster N. 2022.** "IT'IS
  Database for thermal and electromagnetic parameters of
  biological tissues, Version 4.1."
  DOI `10.13099/VIP21000-04-1` (verified at IT'IS website
  itis.swiss/database). Provides 100+ tissue σ values at 10 Hz
  – 100 GHz; the working reference for our `forward.conductivities_sm`.
  Note: v4.1 (May 2022) increased some low-frequency conductivities
  by up to 2× over earlier versions.

### Anisotropy

Muscle σ has a 5–10× longitudinal/transverse anisotropy
(Reichel et al. 1969 (unverified at DOI level), Gabriel 1996
(II)). Our current FEM uses isotropic σ — the documented
limitation in `VALIDATION.md` §6.3. The expected effect on the EEG
forward is a topographic distortion of similar order to skull
anisotropy in EEG/MEG (Wolters 2006).

### Frequency dependence at CAP-relevant 1 kHz

For most tissues, σ at 1 kHz is within 10–30% of the DC value
(Gabriel III, four-Cole–Cole). The CAP bandwidth (300 Hz – 3 kHz
for mixed A+B+C populations; Pelot 2020) is therefore in a regime
where DC σ is a reasonable approximation, modulo the anisotropy
caveat. For high-resolution future work, frequency-dependent
σ becomes important above ~5 kHz where β-dispersion kicks in.

---

## 7. Validation / benchmarking conventions in the FEM-forward
literature

* **Sphere-phantom validation.** The standard convention since
  Marin 1998 *IEEE TBME* **45**:475 (`10.1109/10.664205`,
  verified) is to compare numerical FEM/BEM against the
  closed-form Sarvas (MEG) or Berg-Scherg (EEG) solution in a
  multi-shell sphere with eccentricities in {0.1, 0.5, 0.9, 0.95}
  R. **Acceptable agreement** at the deepest tolerance is
  typically RDM < 0.1 (relative difference measure) and
  MAG ∈ (0.9, 1.1) (magnitude factor). Vorwerk 2014, Piastra 2018,
  and Vorwerk 2021 all report numbers in this range for DUNEuro.
* **Cross-solver validation.** The Vorwerk 2014 head-volume-
  conductor guideline cross-validates SimBio, DUNEuro, and BEM at
  RDM ~3% for cortical sources.
* **Mesh quality.** Joe-Liu metric thresholds: minimum > 0.05
  is the standard hard cutoff (degenerate-tet exclusion), > 0.1
  is the conservative cutoff used in our `mesh/quality.py`.
  Aspect ratio < 10 is the typical recommendation. Our pipeline's
  `cfg.fem.validate.min_mesh_quality` is set at 0.1 by default.
* **Calibration uncertainty reporting.** Most MEG/EEG forward
  papers (Vorwerk 2014, 2021; Piastra 2018) report only median
  RDM and MAG. Our calibration sweep across (R, depth, σ) (27
  cases, CV = 6.8% on the EEG conversion factor —
  `docs/VALIDATION.md` §2.3) is more rigorous than the field
  standard.

---

## 8. Detection-threshold conventions

* **Rose criterion** (SNR = 3 → "just detectable") — Rose A.
  1948 *J Opt Soc Am* **38**:196. DOI `10.1364/JOSA.38.000196`
  (verified at OSA). Origin in radar / image science but the
  common single-channel detection threshold in CAP recording.
* **Multi-channel beamformer SNR** — Sekihara K, Nagarajan SS.
  2008. *Adaptive Spatial Filters for Electromagnetic Brain
  Imaging.* Springer. DOI `10.1007/978-3-540-79370-0` (verified
  at Springer). LCMV / SAM beamformer formalism; defines SNR =
  10 log₁₀(power_signal / power_noise) at the source-localised
  output. For low input SNR (< 3 dB), beamformer scans flatten —
  our 6 dB threshold for "robust localisation" follows Sekihara.
* **Spike-detection threshold** — **Quian Quiroga R, Nadasdy Z,
  Ben-Shaul Y. 2004.** "Unsupervised Spike Detection and Sorting
  with Wavelets and Superparamagnetic Clustering." *Neural Comput*
  **16**:1661–1687. DOI `10.1162/089976604774201631` (verified at
  MIT Press; PMID 15228749). The "4σ above the noise-RMS" detection
  threshold from this paper is the standard CAP-detection cutoff
  used in Bu 2024 and predecessors.

---

## 9. The Bu / Lerman group programme — direct comparators

### Bu et al. 2022 *Sci Rep* (HD-EMG / CEN)

*Claim:* a flexible Ag/AgCl 4-channel adhesive electrode array
attached to the lateral cervical neck (lateral to trachea, medial
to sternocleidomastoid) can record cervical-electroneurography
(CEN) signals during cold-pressor and breath-hold autonomic
stressors. *Evidence:* µV-amplitude signals time-locked to
sympathoexcitatory challenges; cross-subject reproducibility
shown in 11 subjects. *Limitations:* 4-channel only — no spatial
localisation; no per-source forward model; signals are a mixture of
neck-muscle EMG and putative vagal CEN with no biophysical
disambiguation. **What we add:** an HD 128-channel patch with
per-source forward leadfield, and the explicit bone–vagus geometry
showing whether the signal *can* in principle reach the skin.

### Bu et al. 2024 *Comm Biol* (vcMNG / OPM)

*Claim:* an OPM array over the ventral cervical neck records
"vcMNG-derived CAPs" at the right-nodose-ganglion (RNG) and
right-carotid-artery (RCA) regions during human LPS challenge;
firing-rate trajectories track systemic TNF-α, with subgroups by
IL-6 and IL-10. *Evidence:* 9 subjects; 1–4 pT signal across
subjects; statistical correlation of CAP-rate with cytokine
trajectories. *Limitations:* threshold-crossing rate analysis only;
no continuous biophysical signal model; no FEM forward; no
characterisation of which vagal sources can or cannot be resolved
by the array. The "RNG / RCA" attributions are *anatomical rather
than localised* — they are based on coil placement, not source
localisation. **What we add:** a continuous biophysical forward
model that, given the same array, predicts the per-source signal
amplitude and the spatial sensitivity profile, and that yields a
per-source detectability map rather than a single
threshold-crossing rate.

### Distinguishing claims

| Capability | Bu 2022 (HD-EMG) | Bu 2024 (vcMNG) | This pipeline |
| :-- | :-: | :-: | :-: |
| Surface-electrode CEN | ✓ (4 ch) | ✗ | ✓ (128 ch) |
| OPM vcMNG | ✗ | ✓ | ✓ (8 190 ch, predicted) |
| Both modalities | ✗ | ✗ | ✓ |
| Per-source FEM forward | ✗ | ✗ | ✓ |
| Sarvas + Berg–Scherg analytic anchor | ✗ | ✗ | ✓ |
| Tissue-σ sensitivity sweep | ✗ | ✗ | ✓ |
| Continuous CAP biophysical source | ✗ | ✗ | ✓ |
| Moving-dipole physiology | ✗ | ✗ | ✓ |

---

## 10. Gap analysis — what's new

* **Both MEG and EEG forwards from a single FEM and source model.**
  ASCENT (Musselman 2021 PLoS Comp Biol DOI
  `10.1371/journal.pcbi.1009285`, verified) does intrafascicular
  forwards but no non-invasive output. SimNIBS does head EEG/TMS
  only. Sim4Life is commercial and not vagus-specific. None of
  these solve the dual-modality forward simultaneously.
* **Empirical DUNEuro EEG calibration via sphere FEM.** The 0.622
  empirical factor (`docs/VALIDATION.md` §2) is, to our knowledge,
  the first rigorously documented EEG-mode calibration for
  DUNEuro outside the head — a 27-case (R, depth, σ) sweep with
  CV = 6.8%. The DUNEuro paper (Vorwerk 2021) reports a sphere
  validation but does not give an absolute conversion factor for
  the mm-mode output; this is missing in the field.
* **Whole-body EEG sensitivity sweep.** A 1000-electrode whole-body
  sweep vs the cervical 128-electrode patch is, to our knowledge,
  unreported in the literature for the vagus. The closest
  comparator is the head-EEG sensor-density sweep of Robinson 2017
  (`10.1016/j.clinph.2017.02.015`, verified-to-search, *full
  DOI unverified at machine precision*).
* **Moving-dipole physiology simulator.** Baroreceptor + respiratory
  dipole motion on the vagus polyline (`physiology/scenarios.py`)
  is a unique feature; no published vagal forward model embeds a
  physiological moving-dipole driver.
* **Sarvas + Berg–Scherg anchored validation chain.** Both the
  MEG and the EEG forward are independently anchored to the
  closed-form analytic solution in a controlled sphere FEM,
  then propagated to the full multi-tissue torso. This dual
  analytic anchor, applied end-to-end to a non-head body part,
  is also new.

These five points are the headline "what's new" for the abstract.

---

## 11. Suggested manuscript citation order

This is the sequence in which the verified citations should
*first* appear in the manuscript, intro → methods → results →
discussion. (Subsequent re-citations follow the journal style.)

**Introduction**

1. Bu Y et al. 2024 *Commun Biol* — opens with the headline
   non-invasive vcMNG result and the 1–4 pT amplitude band.
2. Bu Y et al. 2022 *Sci Rep* — the HD-EMG comparator.
3. Bu Y et al. 2022 *Front Physiol* — preceding peripheral OPM
   feasibility.
4. Brookes MJ et al. 2022 *Trends Neurosci* — OPM-MEG state of
   the art.
5. Boto E et al. 2018 *Nature* — wearable OPM-MEG.
6. Zanos TP et al. 2018 *PNAS* — invasive cytokine decoding
   (motivates non-invasive translation).
7. Ottaviani MM et al. 2020 *J Physiol* — invasive human vagus
   reference.
8. Marmerstein 2021 *Sci Rep* — chronic vagal recording, HRV
   contradiction.
9. Berntson 1997 *Psychophysiology* + Thayer 2010 *Int J
   Cardiol* — HRV-as-vagal-proxy literature being supplanted.

**Methods**

10. Sarvas J. 1987 *Phys Med Biol* — analytic MEG anchor.
11. Hämäläinen M et al. 1993 *Rev Mod Phys* — Q = π d² σ_in ΔV /
    4 source-strength formula.
12. Geselowitz DB. 1970 *IEEE Trans Magn* — secondary-current
    formalism.
13. Berg P, Scherg M. 1994 *Electroencephalogr Clin Neurophysiol*
    — analytic EEG anchor.
14. Vorwerk J et al. 2021 *PLOS ONE* — DUNEuro toolbox.
15. Piastra MC et al. 2018 *Front Neurosci* — DG-FEM justification.
16. Saturnino GB et al. 2019 — SimNIBS for cross-validation context.
17. Pelot NA et al. 2020 *Front Neurosci* — vagal fascicular
    morphology (cervical 2 mm, fascicle counts, perineurium 6 µm).
18. Hoffman & Schnitzlein 1961 *Anat Rec* — ~100 000 axons,
    population fractions.
19. Settell 2020 *J Neural Eng* + Thompson 2023 *Front Neurosci* —
    fascicular organotopy.
20. Gabriel S et al. 1996 (II + III) *Phys Med Biol* — tissue σ.
21. Hasgall PA et al. 2022 — IT'IS database v4.1.
22. Wolters CH et al. 2006 *NeuroImage* — anisotropy formal-
    sensitivity reference.
23. Tierney TM et al. 2019 *NeuroImage* — OPM forward formalism.
24. Tierney TM et al. 2020 *Sci Rep* — OPM array sampling
    convention.
25. Khodagholy D et al. 2013 (a, b) *Nat Commun* — PEDOT:PSS HD
    array materials.

**Results**

26. O'Neill GC et al. 2025 *Sci Rep* — body-MEG volume-conductor
    formalism; reference for our 6.8× FEM/Sarvas amplification
    interpretation.
27. Hill RM et al. 2020 *NeuroImage* — wearable OPM helmet
    benchmark.
28. Rea M et al. 2022 *Ann NY Acad Sci* — triaxial 90-channel
    OPM-MEG benchmark.
29. Sekihara K, Nagarajan SS. 2008 — beamformer SNR conventions.
30. Quian Quiroga R et al. 2004 *Neural Comput* — 4σ CAP
    threshold convention.
31. Rose A. 1948 *J Opt Soc Am* — Rose-criterion detection
    threshold.

**Discussion**

32. Sevcencu C et al. 2018 *Neuromodulation* — invasive pig
    intraneural reference for amplitude.
33. Verma N et al. 2022 (BioRxiv) — invasive ECAP comparator
    (preprint).
34. Marshall E et al. 2022 *Am J Physiol* — MGG comparator for
    body OPM.
35. Dolinská-Jachowska et al. 2023 (Frontiers) — OPM-MCG state.
36. Mardell L et al. 2024 (Frontiers Med Tech) — OPM
    spinal-imaging comparator.
37. Musselman ED et al. 2021 *PLoS Comp Biol* — ASCENT
    intrafascicular comparator.

---

## Summary of verification status

| Citation | DOI verified | Notes |
| :-- | :-: | :-- |
| Sarvas 1987 | ✓ | IOPscience + PubMed |
| Hämäläinen 1993 | ✓ | APS |
| Geselowitz 1970 | ✓ | IEEE; **VALIDATION.md DOI is wrong, README is right** |
| Berg & Scherg 1994 | ✓ | Wiley + PubMed |
| Wolters 2006 | ✓ | NeuroImage + PubMed |
| Vorwerk 2021 (DUNEuro) | ✓ | PLOS + PubMed |
| Piastra 2018 (DG-FEM) | ✓ | Frontiers |
| Saturnino 2019 (SimNIBS) | ✓ | Springer |
| O'Neill 2025 | ✓ | Nature; **codebase article number 25649 is wrong, correct is 26258** |
| Bu 2024 *Commun Biol* | ✓ | Nature + PubMed; **codebase has wrong author (Doherty)** |
| Bu 2022 *Sci Rep* | ✓ | Nature + PubMed |
| Bu 2022 *Front Physiol* | ✓ | Frontiers; **codebase calls this "Zuo" — wrong** |
| Pelot 2020 | ✓ | Frontiers + PubMed; **codebase has wrong year (2017) and wrong DOI** |
| Hoffman & Schnitzlein 1961 | ✓ | Wiley + PubMed |
| Settell 2020 | ✓ | IOP + PubMed |
| Thompson 2023 | ✓ | Frontiers |
| Zanos 2018 | ✓ | PNAS + PubMed |
| Marmerstein 2021 | ✓ | Nature + PubMed |
| Sevcencu 2018 | ✓ | Wiley |
| Ottaviani 2020 | ✓ | Wiley + PubMed |
| Verma 2022 (BioRxiv) | ✓ | bioRxiv (preprint) |
| Berntson 1997 | ✓ | Wiley + PubMed |
| Thayer 2010 | ✓ | ScienceDirect |
| Brookes 2022 | ✓ | PubMed |
| Tierney 2019 | ✓ | PubMed |
| Tierney 2020 | ✓ | Nature + PubMed |
| Hill 2020 | ✓ | NeuroImage + PubMed |
| Rea 2022 | ✓ | Wiley + PubMed |
| Khodagholy 2013 (a) | ✓ | Nature |
| Khodagholy 2013 (b) | ✓ | Nature + PubMed |
| Gabriel 1996 II | ✓ | IOP + PubMed |
| Gabriel 1996 III | ✓ | IOP + PubMed |
| Hasgall IT'IS v4.1 | ✓ | itis.swiss |
| Sekihara 2008 | ✓ | Springer |
| Quian Quiroga 2004 | ✓ | MIT Press + PubMed |
| Rose 1948 | ✓ | OSA |
| Marshall 2022 (MGG) | ✓ | PubMed |
| Settell 2024 *Brain Stimul* | partial | DOI not machine-verified to article landing page |
| Reichel 1969 muscle anisotropy | ✗ | (unverified at DOI level) |
| Robinson 2017 sensor density | partial | (unverified at DOI level) |

---

*Last verified 2026-05-01 by automated WebFetch / WebSearch against
PubMed, Nature, Frontiers, IOPscience, Wiley, MIT Press, Springer,
APS, OSA, IT'IS Foundation, IEEE Xplore.*
