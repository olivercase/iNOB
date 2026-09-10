# iNOB and existing software

What iNOB reuses, what it sits beside, and where it differs. Written to be
checked against the tools' own documentation; version numbers are those
current at the time of writing.

## The one-line difference

Every established bioelectromagnetic forward-modelling package models the
**head**. iNOB models the **body**: a multi-tissue FEM of the neck, trunk and
limbs, with the source in a peripheral or autonomic nerve, the spinal cord or
a muscle, and the sensors (OPMs, SQUIDs, surface electrodes) placed
automatically on the skin around the target. The forward solver itself is
not new — it is DUNEuro — and iNOB does not claim otherwise.

## Dependencies iNOB is built on

| Component | Used for | Licence |
| --- | --- | --- |
| [DUNEuro](https://www.duneuro.org) (Schrader et al., 2021) via `duneuropy` | the MEG and EEG FEM solves (partial-integration / Venant source models, CG–FEM and DG–FEM) | LGPL |
| [DUNE](https://www.dune-project.org) 2.10 (Bastian et al., 2021) | the grid and solver framework under DUNEuro | GPL-2 with runtime exception |
| [iso2mesh](https://iso2mesh.sourceforge.net) (Fang & Boas, 2009) with CGAL | volumetric multi-domain tetrahedral meshing from the label image | GPL-2 / CGAL |
| [pymeshfix](https://github.com/pyvista/pymeshfix) (MeshFix, Attene, 2010) | closing and repairing the atlas surfaces | GPL-3 |
| [trimesh](https://trimesh.org), [PyVista](https://docs.pyvista.org) (Sullivan & Kaszynski, 2019), scikit-image | surface handling, voxel shrink-wrap, rendering | MIT |
| [BodyParts3D](https://lifesciencedb.jp/bp3d/) (Mitsuhashi et al., 2009) | the anatomy | CC BY-SA 2.1 JP |
| NumPy, SciPy, h5py, matplotlib, PyYAML | numerics, I/O, figures | BSD / MIT |

`docs/CITING.md` lists the references to cite when these are used through
iNOB.

## Head-modelling toolboxes

| | Anatomy | Volume conductor | Forward solver | Sensors | Where iNOB differs |
| --- | --- | --- | --- | --- | --- |
| **MNE-Python** (Gramfort et al., 2013) | subject MRI via FreeSurfer | 1- or 3-layer BEM, sphere | BEM (own, OpenMEEG), sphere; DUNEuro via FieldTrip only | MEG helmets, EEG montages | head only; BEM cannot carry anisotropic muscle or thin bone shells around the neck |
| **FieldTrip** (Oostenveld et al., 2011) | subject MRI, SPM segmentation | BEM, FEM (SimBio/DUNEuro), sphere | BEM, FEM | MEG, EEG | head templates and segmentation; DUNEuro FEM is exposed but there is no body geometry, body sensor placement or peripheral source model |
| **Brainstorm** (Tadel et al., 2011) | subject MRI, CAT/FreeSurfer | BEM (OpenMEEG), FEM (DUNEuro) | BEM, FEM | MEG, EEG, sEEG | as FieldTrip; MATLAB, GUI-first |
| **SimNIBS** (Thielscher et al., 2015; Saturnino et al., 2019) | subject MRI, `charm` segmentation | multi-tissue head FEM | own FEM (getDP, then native) | TMS coils, tDCS electrodes | forward problem for *stimulation*, not recording; head only |
| **OpenMEEG** (Gramfort et al., 2010) | any nested surfaces | BEM | symmetric BEM | EEG, MEG | a solver, not a pipeline; nested-surface BEM does not describe a torso with disjoint bone and muscle compartments |
| **DUNEuro** (Schrader et al., 2021) | any tetra/hex mesh | FEM, any number of tissues | CG/DG-FEM, several source models | user-supplied coils and electrodes | iNOB *is* a DUNEuro application: it supplies the body mesh, the conductivities, the automatic sensor placement, calibration to absolute units, and the noise and detectability layer |

## Peripheral-nerve and body modelling

| | Purpose | Where iNOB differs |
| --- | --- | --- |
| **ASCENT** (Musselman et al., 2021) | FEM of *stimulation* of a peripheral nerve by a cuff electrode: nerve cross-section, fascicles, fibre models in NEURON; needs COMSOL | opposite direction (nerve as source, skin as sensor), whole-body rather than a nerve segment, open-source solver |
| **Sim4Life / SEMCAD, COMSOL, ANSYS** with the IT'IS Virtual Population | general electromagnetic simulation on detailed body models | commercial licences for solver and anatomy; no MEG/EEG leadfield, noise or detectability workflow |
| **Magnetospinography volume-conductor studies** (O'Neill et al., 2025) | analytic and BEM models of the spine for OPM-MSG | published models rather than a package; iNOB reproduces the spine case as one of three worked targets and adds vagus and muscle |
| **Magnetoneurography** (Doherty et al., 2024; earlier SQUID MNG work) | measurements of nerve fields | no forward model distributed with them; iNOB is the planning tool those experiments lacked |

## What iNOB does not do

- Subject-specific anatomy. The volume conductor is a single atlas
  (BodyParts3D). Nothing prevents replacing the STLs with segmented MRI, but
  no segmentation step is shipped.
- Frequency-dependent or nerve-anisotropic conductivity. Anisotropy is
  implemented for muscle only.
- Inverse modelling. iNOB produces leadfields and detectability estimates;
  source reconstruction is left to MNE-Python, FieldTrip or Brainstorm, whose
  leadfield formats the NPZ output maps onto.
- Fibre-level physiology. Sources are current dipoles and propagating
  compound action potentials built from them, not NEURON fibre models.

## References

- Attene M (2010). A lightweight approach to repairing digitized polygon meshes. *Vis Comput* 26:1393–1406.
- Bastian P et al. (2021). The DUNE framework: basic concepts and recent developments. *Comput Math Appl* 81:75–112.
- Doherty TM et al. (2024). Non-invasive ventral cervical magnetoneurography as a proxy of in vivo lipopolysaccharide-induced inflammation. *Commun Biol* 7:893.
- Fang Q, Boas DA (2009). Tetrahedral mesh generation from volumetric binary and grayscale images. *ISBI 2009*, 1142–1145.
- Gramfort A et al. (2010). OpenMEEG: opensource software for quasistatic bioelectromagnetics. *Biomed Eng Online* 9:45.
- Gramfort A et al. (2013). MEG and EEG data analysis with MNE-Python. *Front Neurosci* 7:267.
- Mitsuhashi N et al. (2009). BodyParts3D: 3D structure database for anatomical concepts. *Nucleic Acids Res* 37:D782–D785.
- Musselman ED et al. (2021). ASCENT: a pipeline for modeling stimulation of peripheral nerves. *PLoS Comput Biol* 17:e1009285.
- O'Neill GC et al. (2025). Volume conductor models for magnetospinography. *Sci Rep* 15:25649.
- Oostenveld R et al. (2011). FieldTrip: open source software for advanced analysis of MEG, EEG, and invasive electrophysiological data. *Comput Intell Neurosci* 2011:156869.
- Saturnino GB et al. (2019). SimNIBS 2.1: a comprehensive pipeline for individualized electric field modelling for the brain. In *Brain and Human Body Modeling*, Springer.
- Schrader S et al. (2021). DUNEuro — a software toolbox for forward modeling in bioelectromagnetism. *PLoS ONE* 16:e0252431.
- Sullivan CB, Kaszynski AA (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit. *J Open Source Softw* 4:1450.
- Tadel F et al. (2011). Brainstorm: a user-friendly application for MEG/EEG analysis. *Comput Intell Neurosci* 2011:879716.
- Thielscher A, Antunes A, Saturnino GB (2015). Field modeling for transcranial magnetic stimulation: a useful tool to understand the physiological effects of TMS? *IEEE EMBC 2015*, 222–225.
