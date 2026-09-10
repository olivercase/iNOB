# Citing iNOB and what it stands on

## iNOB

`CITATION.cff` at the repository root carries the metadata (GitHub shows a
"Cite this repository" button from it). Software DOI:
[10.17605/OSF.IO/U4MDS](https://doi.org/10.17605/OSF.IO/U4MDS). Cite the
exact version you used (`inob --version`); each tagged release is archived.

## Dependencies you are also using

A result from iNOB is a result from these packages too. The journal
guidelines for software papers ask that reuse be cited both in the paper and
by users of the software, so please cite:

**Always**

- DUNEuro, for every leadfield — Schrader S, Antonakakis M, Rampp S, Engwer C,
  Wolters CH (2021). DUNEuro — a software toolbox for forward modeling in
  bioelectromagnetism. *PLoS ONE* 16(6):e0252431.
  https://doi.org/10.1371/journal.pone.0252431
- DUNE — Bastian P et al. (2021). The DUNE framework: basic concepts and
  recent developments. *Computers & Mathematics with Applications* 81:75–112.
  https://doi.org/10.1016/j.camwa.2020.06.007
- BodyParts3D, for the anatomy — Mitsuhashi N et al. (2009). BodyParts3D: 3D
  structure database for anatomical concepts. *Nucleic Acids Research*
  37:D782–D785. https://doi.org/10.1093/nar/gkn613. Credit
  "BodyParts3D, © The Database Center for Life Science", CC BY-SA 2.1 JP
  (see `LICENSE-DATA`).
- iso2mesh / CGAL, for the tetrahedral mesh — Fang Q, Boas DA (2009).
  Tetrahedral mesh generation from volumetric binary and grayscale images.
  *IEEE ISBI 2009*, 1142–1145. https://doi.org/10.1109/ISBI.2009.5193259.
  The CGAL Project (2024). *CGAL User and Reference Manual*. https://www.cgal.org

**When the relevant stage was used**

- Surface repair (`build-geom`) — Attene M (2010). A lightweight approach to
  repairing digitized polygon meshes. *The Visual Computer* 26:1393–1406.
  https://doi.org/10.1007/s00371-010-0416-3
- Figures (`visualise`, `torso`, `topoplot`) — Sullivan CB, Kaszynski AA
  (2019). PyVista. *JOSS* 4(37):1450. https://doi.org/10.21105/joss.01450;
  Hunter JD (2007). Matplotlib. *Comput Sci Eng* 9:90–95.
- Analytic validation (`sarvas`, `ladder`, `calibrate`) — Sarvas J (1987).
  Basic mathematical and electromagnetic concepts of the biomagnetic inverse
  problem. *Phys Med Biol* 32:11–22. https://doi.org/10.1088/0031-9155/32/1/004;
  Berg P, Scherg M (1994). A fast method for forward computation of
  multiple-shell spherical head models. *Electroencephalogr Clin Neurophysiol*
  90:58–64.
- OPM noise presets (`noise.opm_sensor`) — the manufacturer specification named
  in `src/inob/sensors/opm_presets.py` for the preset you selected.

## Sample data

The STL meshes under `data/` are the sample data for every example in the
paper and the documentation, and are BodyParts3D derivatives under
`LICENSE-DATA`. Redistribute derivatives under the same CC BY-SA 2.1 JP
licence with the attribution above.
