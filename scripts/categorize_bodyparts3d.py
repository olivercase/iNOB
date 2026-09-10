#!/usr/bin/env python3
"""Sort the downloaded BodyParts3D 4.3 OBJ meshes into anatomical-system folders.

Method (in priority order, per mesh):
  1. **FMA IS-A walk** — build a child→parent map from ``isa_inclusion_relation_list.txt``
     and walk each mesh's FMA concept *upward* to the NEAREST system-root concept.
     This is the principled route: e.g. "Left splenius cervicis" has no word "muscle"
     in its name but IS-A …→ muscle organ (FMA5022) → muscle bucket.
  2. **Name heuristics** — fallback for FMA ids not connected to a root in the tree.
  3. ``other/`` — anything still unresolved (never dropped).

Reads from ``<src>`` (default ``data/bodyparts3d/raw_4.3``):
  ``objs/*.obj`` and ``metadata/isa_inclusion_relation_list.txt``.
Writes copies into ``<dst>/<system>/`` (default ``internal_meshes/``) plus
``MANIFEST.csv`` and ``README.md``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import sys
from collections import deque
from pathlib import Path

logger = logging.getLogger("bp3d_cat")

# Manual overrides from a hand-audit of 100 sampled meshes + the full residual
# bucket (2026-05-26). Keyed by FMA id; checked FIRST, so they always win. These
# are structures the IS-A walk and name rules misfiled (PART-OF-only links,
# irregular names). See README "How each mesh was classified".
MANUAL_OVERRIDES: dict[str, str] = {
    # CNS structures linked to brain only via part-of
    "FMA72976": "brain",
    "FMA72975": "brain",  # occipital lobe L/R
    "FMA72672": "brain",
    "FMA72671": "brain",  # superior parietal lobule L/R
    "FMA72940": "brain",
    "FMA61974": "brain",  # stria terminalis L/—
    "FMA62033": "brain",  # pineal body
    "FMA62327": "brain",  # tuber cinereum
    # respiratory / digestive / oral organ subparts
    "FMA7396": "organ",
    "FMA68418": "organ",  # main bronchus L/R
    "FMA11338": "organ",  # ileocecal junction
    "FMA59816": "organ",  # lip
    "FMA52781": "organ",  # external ear
    "FMA14643": "organ",
    "FMA16549": "organ",  # mesentery, mesoappendix
    "FMA14647": "organ",  # transverse mesocolon
    "FMA63120": "organ",  # parenchyma of pancreas
    "FMA15044": "organ",
    "FMA15042": "organ",
    "FMA15043": "organ",  # taeniae coli
    # bone
    "FMA7486": "bone",  # manubrium (of sternum)
    "FMA7488": "bone",  # xiphoid process
    # muscle (laryngeal)
    "FMA46605": "muscle",
    "FMA46604": "muscle",  # aryepiglotticus L/R
    # blood vessel (arterial arches)
    "FMA22840": "blood_vessel",
    "FMA22839": "blood_vessel",  # deep palmar arch L/R
    "FMA43944": "blood_vessel",
    "FMA43943": "blood_vessel",  # plantar arch L/R
    # fibrous raphes / membranes
    "FMA55620": "ligament_tendon",
    "FMA55619": "ligament_tendon",  # pterygomandibular raphe
    "FMA55077": "ligament_tendon",  # pharyngeal raphe
    "FMA55252": "ligament_tendon",
    "FMA55251": "ligament_tendon",  # conus elasticus L/R
    # cartilage
    "FMA55130": "cartilage",  # epiglottis (elastic cartilage)
    # skin / integument appendages
    "FMA54237": "skin",
    "FMA54241": "skin",
    "FMA54319": "skin",  # eyebrow, hair, pubic hair
}

# FMA concept id → system bucket. Nearest match wins during the upward walk,
# so specific roots (muscle organ) beat the generic "organ".
SYSTEM_ROOTS: dict[str, str] = {
    "FMA5018": "bone",  # bone organ
    "FMA55107": "cartilage",  # cartilage organ
    "FMA5022": "muscle",  # muscle organ
    "FMA50720": "blood_vessel",  # artery
    "FMA50723": "blood_vessel",  # vein
    "FMA3710": "blood_vessel",  # vascular tree
    "FMA14284": "blood_vessel",  # venous tree organ
    "FMA65132": "nerve",  # nerve
    "FMA5884": "nerve",  # ganglion
    "FMA9721": "ligament_tendon",  # tendon
    "FMA21496": "ligament_tendon",  # ligament organ
    "FMA7163": "skin",  # skin
    "FMA67498": "organ",  # organ (generic catch-all — lowest specificity)
}

# Fallback name heuristics (ordered; FIRST match wins). Applied only when the
# IS-A walk yields nothing — mostly organ subparts / CNS that connect to their
# system via PART-OF rather than IS-A. Vessel rules run first (e.g. "pulmonary
# trunk", "renal artery" must beat the respiratory/urinary organ rules).
NAME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(artery|arteries|arterial|aorta|arteriole)\b", re.I), "blood_vessel"),
    (re.compile(r"\b(vein|veins|venous|vena|venule|hepatovenous|portal)\b", re.I), "blood_vessel"),
    (re.compile(r"\b(vascular|vessel|capillary|perforating)\b", re.I), "blood_vessel"),
    (
        re.compile(
            r"\b(celiac|coeliac|pulmonary|brachiocephalic|thyrocervical|"
            r"costocervical) trunk\b",
            re.I,
        ),
        "blood_vessel",
    ),
    # CNS / brain (nervous tissue, distinct from peripheral nerve for FEM).
    (
        re.compile(
            r"\b(cerebr|cerebell|cortex|cortical|gyrus|gyri|sulcus|colliculus|"
            r"callosum|commissure|peduncle|thalam|hypothalam|\bpons\b|medulla "
            r"oblongata|fornix|forebrain|midbrain|hindbrain|diencephalon|"
            r"telencephalon|brainstem|brain stem|putamen|pallidum|caudate nucleus|"
            r"amygdala|hippocamp|insula|operculum|lobe of cerebr|white matter|"
            r"gr[ae]y matter|choroid plexus|lateral ventricle|fourth ventricle|"
            r"third ventricle|cerebral aqueduct|spinal cord|olfactory|optic chiasm|"
            r"habenula|internal capsule|external capsule|claustrum|globus pallidus|"
            r"substantia nigra|red nucleus|lentiform|striatum|tegmentum|\btectum\b|"
            r"subthalam|mammillary|septum pellucidum|dentate|vermis|uncus|"
            r"geniculate body|optic tract|interpeduncular|lamina terminalis)",
            re.I,
        ),
        "brain",
    ),
    (
        re.compile(
            r"\b(nerve|nervous|ganglion|ganglia|plexus|\bnucleus\b|nerve tract|"
            r"rootlet|ramus)\b",
            re.I,
        ),
        "nerve",
    ),
    (re.compile(r"\b(cartilage|meniscus|intervertebral disc|articular disc)\b", re.I), "cartilage"),
    (
        re.compile(
            r"\b(ligament|tendon|tendinous|aponeurosis|retinaculum|fascia|"
            r"iliotibial tract|linea alba)\b",
            re.I,
        ),
        "ligament_tendon",
    ),
    # Named skeletal muscles that the sparse IS-A tree didn't chain to FMA5022.
    (
        re.compile(
            r"\b(muscle|muscular|part of (left|right) |levatores|levator|"
            r"vastus|rectus (femoris|abdominis)|gracilis|sartorius|gastrocnemius|"
            r"soleus|plantaris|popliteus|tibialis|peroneus|fibularis|gemellus|"
            r"piriformis|obturator|quadratus|iliococcygeus|pubococcygeus|"
            r"puborectalis|coccygeus|interossei|interosseus|interossea|lumbric|"
            r"deltoid|biceps|triceps|brachialis|brachioradialis|supinator|"
            r"pronator|palmaris|extensor|flexor|abductor|adductor|opponens|"
            r"semimembranosus|semitendinosus|gluteus|psoas|iliacus|pectineus|"
            r"masseter|temporalis|pterygoid|buccinator|orbicularis|zygomaticus|"
            r"platysma|sternocleidomastoid|scalene|trapezius|rhomboid|serratus|"
            r"intercostal|diaphragm|transversus|\boblique\b|latissimus|"
            r"infraspinatus|supraspinatus|subscapularis|\bteres\b|coracobrachialis|"
            r"anconeus|hyoglossus|genioglossus|styloglossus|mylohyoid|digastric|"
            r"omohyoid|sternohyoid|thyrohyoid|cricothyroid|cricoarytenoid|"
            r"arytenoid|stylohyoid|geniohyoid|sphincter|detrusor)\b",
            re.I,
        ),
        "muscle",
    ),
    # Muscle-group stems (unanchored: catch plurals like "lumbricals", "rotatores").
    (
        re.compile(
            r"(lumbrical|intertransversari|interspinal|rotatores|multifidus|"
            r"semispinalis|splenius|longissimus|iliocostalis|spinalis|"
            r"longus (colli|capitis)|rectus capitis|scalenus|interossei)",
            re.I,
        ),
        "muscle",
    ),
    (re.compile(r"\b(adipose|fatty|fat pad|\bfat\b)\b", re.I), "fat"),
    (
        re.compile(
            r"\b(gland|glandular|pancrea|thyroid|adrenal|parotid|"
            r"submandibular|pituitary|hypophysis)\b",
            re.I,
        ),
        "gland",
    ),
    (re.compile(r"\b(lymph|lymphatic|lymphoid|tonsil|thymus|spleen|splenic)\b", re.I), "lymphatic"),
    (re.compile(r"\b(skin|epidermis|dermis|integument)\b", re.I), "skin"),
    (
        re.compile(
            r"\b(bone|vertebra|rib\b|sternum|mandible|maxilla|phalanx|phalange|"
            r"femur|tibia|fibula|humerus|radius|ulna|scapula|clavicle|"
            r"patella|carpal|tarsal|metacarpal|metatarsal|sacrum|coccyx|ilium|"
            r"ischium|pubis|cranium|skull|\bos |ossicle|hyoid|malleus|incus|"
            r"stapes|calcaneus|talus|navicular|cuboid|cuneiform)\b",
            re.I,
        ),
        "bone",
    ),
    # Viscera / organ subparts (digestive, respiratory, cardiac, urinary, repro, eye/ear).
    (
        re.compile(
            r"\b(colon|ileum|jejunum|duodenum|cecum|caecum|rectum|stomach|"
            r"intestin|liver|hepatic|biliary|\bbile\b|gallbladder|gall bladder|"
            r"esophagus|oesophagus|appendix|pylor|cardia\b|omentum|mesenter|"
            r"peritone)\b",
            re.I,
        ),
        "organ",
    ),
    (
        re.compile(
            r"\b(bronch|lung|trachea|alveol|pleura|larynx|laryngeal|pharyn|"
            r"epiglott|glottis|nasal cavity|paranasal|nostril|nasopharyn)\b",
            re.I,
        ),
        "organ",
    ),
    (
        re.compile(
            r"\b(valve|cusp|leaflet|atrium|atrial|ventricle|ventricular|"
            r"myocard|pericard|endocard|cardiac|chamber of (left|right)|"
            r"interventricular|interatrial|chordae|papillary|auricle)\b",
            re.I,
        ),
        "organ",
    ),
    (
        re.compile(
            r"\b(kidney|renal|ureter|bladder|urethra|urinary|prostat|uterus|"
            r"uterine|ovar|testis|testicular|penis|penile|corpus cavernosum|"
            r"corpus spongiosum|vagina|seminal|epididymis|scrotum|vas deferens|"
            r"fallopian|oviduct|clitoris|labium)\b",
            re.I,
        ),
        "organ",
    ),
    (
        re.compile(
            r"\b(eyeball|cornea|retina|\blens\b|sclera|\biris\b|vitreous|"
            r"aqueous|chamber of (left|right) eyeball|cochlea|vestibul|tympan|"
            r"auditory|pinna|eardrum|semicircular|labyrinth|conjunctiva|eyelid|"
            r"choroid|ciliaris|ciliary|lacrimal|gingiva|gum\b|palate|uvula|"
            r"tongue|lingual|tooth|teeth|dental)\b",
            re.I,
        ),
        "organ",
    ),
    (
        re.compile(
            r"\b(heart|trunk of|\bduct\b|sinus|tree|wall of|cavity of|"
            r"segment of|lobe of|root of|head of|body of|tail of|neck of)\b",
            re.I,
        ),
        "organ",
    ),
    # -- supplementary rules (appended; only reached when nothing above matched) --
    # These clear residual names whose stems the \b-anchored rules above miss
    # (e.g. "bronchus", "peritoneum", "prostate") plus specifically-named parts.
    # Ordered so vessel/nerve/brain/cartilage/membrane win before the broad
    # muscle/bone/organ stems.
    (re.compile(r"\b(arteria|palpebral arch)\b", re.I), "blood_vessel"),
    (re.compile(r"\bcauda equina\b", re.I), "nerve"),
    (re.compile(r"\b(stria terminalis|precuneus|cuneus)\b", re.I), "brain"),
    (re.compile(r"\bintervertebral dis[ck]\b", re.I), "cartilage"),
    (re.compile(r"\binterosseous membrane\b", re.I), "ligament_tendon"),
    (
        re.compile(
            r"\b(rectus|interosseous|tensor|pectoralis|subclavius|rotator|"
            r"obliquus|sternothyroid|palatopharyngeus|salpingopharyngeus|"
            r"stylopharyngeus|pharyngeal constrictor|veli palatini|vocalis)\b",
            re.I,
        ),
        "muscle",
    ),
    (
        re.compile(
            r"\b(trapezoid|trapezium|capitate|hamate|lunate|pisiform|scaphoid|"
            r"triquetral|atlas|axis|vomer|ethmoid|nasal concha)\b",
            re.I,
        ),
        "bone",
    ),
    (re.compile(r"\b(prostate|peritoneum|bronchus|parenchyma|bronchopulmonary)\b", re.I), "organ"),
]


def load_isa_parents(meta: Path) -> dict[str, set[str]]:
    """child FMA → {parent FMA} from isa_inclusion_relation_list.txt.

    Optional: the 4.3 download set does not ship this file (it is a dbarchive
    artifact). When absent we classify from the 4.3 manifest's FMA ids via the
    name rules + manual overrides alone, which resolve 100% of the 4.3 set."""
    path = meta / "isa_inclusion_relation_list.txt"
    if not path.exists():
        logger.info("no isa_inclusion_relation_list.txt -> name+override classification only")
        return {}
    parents: dict[str, set[str]] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if i == 0:
            continue  # header: parent id, parent name, child id, child name
        c = line.split("\t")
        if len(c) >= 3 and c[0].startswith("FMA") and c[2].startswith("FMA"):
            parents.setdefault(c[2], set()).add(c[0])
    return parents


def classify_by_isa(fma: str, parents: dict[str, set[str]]) -> str | None:
    """Walk upward (BFS by level) → bucket of the NEAREST system root, else None."""
    if fma in SYSTEM_ROOTS:
        return SYSTEM_ROOTS[fma]
    seen = {fma}
    frontier = deque([fma])
    while frontier:
        # process current level, collect any root hits, nearest level wins
        level: list[str] = []
        for _ in range(len(frontier)):
            node = frontier.popleft()
            for par in parents.get(node, ()):
                if par in seen:
                    continue
                seen.add(par)
                level.append(par)
        hits = [SYSTEM_ROOTS[p] for p in level if p in SYSTEM_ROOTS]
        if hits:
            # priority if several at the same distance (specific before generic)
            for b in (
                "blood_vessel",
                "nerve",
                "muscle",
                "bone",
                "cartilage",
                "ligament_tendon",
                "skin",
                "gland",
                "lymphatic",
                "fat",
                "organ",
            ):
                if b in hits:
                    return b
            return hits[0]
        frontier.extend(level)
    return None


def classify_by_name(name: str) -> str | None:
    for pat, bucket in NAME_RULES:
        if pat.search(name):
            return bucket
    return None


_FNAME = re.compile(r"^(FJ[0-9A-Za-z]+)_(BP[0-9]+)_(FMA[0-9]+)_(.+)\.obj$", re.I)


def parse_name(fn: str) -> tuple[str, str, str, str] | None:
    m = _FNAME.match(fn)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def obj_counts(path: Path) -> tuple[int, int]:
    """Cheap vertex/face counts by streaming the OBJ (no trimesh dependency)."""
    v = f = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                v += 1
            elif line.startswith("f "):
                f += 1
    return v, f


def run(src: Path, dst: Path, *, copy: bool, counts: bool) -> None:
    objs_dir, meta = src / "objs", src / "metadata"
    parents = load_isa_parents(meta)
    logger.info("IS-A edges loaded for %d child concepts", len(parents))

    dst.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    by_system: dict[str, int] = {}
    method_count = {"manual": 0, "isa": 0, "name": 0, "other": 0}

    files = sorted(objs_dir.glob("*.obj"))
    logger.info("classifying %d meshes…", len(files))
    for i, path in enumerate(files):
        parsed = parse_name(path.name)
        if not parsed:
            fj, bp, fma, label = "", "", "", path.stem
        else:
            fj, bp, fma, label = parsed
        readable = label.replace("_", " ")

        system = MANUAL_OVERRIDES.get(fma)
        method = "manual"
        if system is None:
            system = classify_by_isa(fma, parents) if fma else None
            method = "isa"
        if system is None:
            system = classify_by_name(readable)
            method = "name"
        if system is None:
            system, method = "other", "other"
        method_count[method] += 1

        out_dir = dst / system
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name
        if copy:
            shutil.copy2(path, out_path)
        v = f = -1
        if counts:
            v, f = obj_counts(path)
        rows.append(
            {
                "fma_id": fma,
                "fj_id": fj,
                "bp_id": bp,
                "name": readable,
                "system": system,
                "method": method,
                "vertices": v,
                "faces": f,
                "path": str(out_path.relative_to(dst.parent)) if copy else "",
                "data_version": "4.3",
            }
        )
        by_system[system] = by_system.get(system, 0) + 1
        if (i + 1) % 400 == 0:
            logger.info("  %d/%d", i + 1, len(files))

    # MANIFEST.csv
    man = dst / "MANIFEST.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", man, len(rows))

    write_readme(dst, by_system, method_count, len(rows))
    logger.info(
        "DONE — %d meshes; by system: %s",
        len(rows),
        ", ".join(f"{k}={v}" for k, v in sorted(by_system.items(), key=lambda x: -x[1])),
    )


def write_readme(dst: Path, by_system: dict[str, int], method: dict[str, int], total: int) -> None:
    lines = [
        "# internal_meshes — BodyParts3D 4.3, sorted by anatomical system",
        "",
        f"{total} full-resolution OBJ meshes from BodyParts3D / Anatomography "
        "**version 4.3**, sorted into per-system subfolders.",
        "",
        "**Provenance:** downloaded 2026-05-26 from the Anatomography viewer's "
        "internal download API (`lifesciencedb.jp/bp3d`) — the full-res 4.3 set is "
        "not available from the public bulk archive (which only ships 4.0, "
        "polygon-reduced). See `scripts/download_bodyparts3d_4.3.py`.",
        "",
        "**License:** BodyParts3D, © Database Center for Life Science (DBCLS), "
        "CC-BY-SA 2.1 Japan. Attribute DBCLS / BodyParts3D on redistribution.",
        "",
        "**Filename:** `FJ<file>_BP<rep>_FMA<concept>_<name>.obj` — the FMA id ties "
        "each mesh to the Foundational Model of Anatomy concept.",
        "",
        "## Counts by system",
        "",
        "| System | Meshes |",
        "|--------|-------:|",
    ]
    for k, v in sorted(by_system.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    lines += [
        f"| **total** | **{total}** |",
        "",
        "## How each mesh was classified",
        "",
        f"- `isa` (FMA IS-A hierarchy walk to nearest system root): {method['isa']}",
        f"- `name` (name-keyword fallback): {method['name']}",
        f"- `other` (unresolved): {method['other']}",
        "",
        "See `MANIFEST.csv` for the per-mesh fma_id / system / method / vertex+face counts.",
    ]
    (dst / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("/Volumes/UCL/Forward_Model_Vagus_Nerve/data/bodyparts3d/raw_4.3"),
    )
    ap.add_argument(
        "--dst", type=Path, default=Path("/Volumes/UCL/Forward_Model_Vagus_Nerve/internal_meshes")
    )
    ap.add_argument(
        "--no-copy", action="store_true", help="classify + manifest only; don't copy files"
    )
    ap.add_argument("--no-counts", action="store_true", help="skip vertex/face counting (faster)")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    run(args.src, args.dst, copy=not args.no_copy, counts=not args.no_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
