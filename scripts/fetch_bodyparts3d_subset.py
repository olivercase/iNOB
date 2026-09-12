#!/usr/bin/env python3
"""Download only the BodyParts3D 4.3 meshes a figure needs, grouped by system.

``download_bodyparts3d_4.3.py`` fetches the whole 3,210-mesh object set, which
is the right thing when the question is "what is in 4.3?" and the wrong thing
when the question is "show me a brain, a gut and a leg". This script takes the
same authoritative 4.3 manifest and the same download endpoint, but asks for a
named subset — a few hundred meshes, four requests instead of sixty-five.

Groups are defined in :data:`GROUPS` by FMA *name* pattern, matched against the
``FMA Name`` column of the 4.3 ``obj2FMA`` table. Names are the catalogue's own,
so a group is auditable: ``--list`` prints exactly which meshes a pattern
selected, before anything is downloaded.

Output layout (under ``--out``, default ``data/bodyparts3d/subset``)::

    chunks/     chunk_<group>_<n>.zip     raw downloads, resumable
    objs/<group>/FJ..._BP..._FMA..._<name>.obj
    MANIFEST.csv  group, fj_id, bp_id, fma_id, name, faces, verts, bytes

Provenance: BodyParts3D/Anatomography, version 4.3, © The Database Center for
Life Science, licensed CC BY-SA 2.1 JP. Same source and licence as the meshes
already in ``data/``; see that directory's provenance note.

Usage::

    python3 scripts/fetch_bodyparts3d_subset.py --list
    python3 scripts/fetch_bodyparts3d_subset.py
    python3 scripts/fetch_bodyparts3d_subset.py --group brain --group gut
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import re
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger("bp3d_subset")

#: Which meshes make up each system, as case-insensitive regexes over the FMA
#: name. Deliberately explicit rather than an IS-A walk: this is a figure's
#: cast list, not a classification, and a reader of the figure should be able
#: to read here exactly what "gut" was taken to mean.
GROUPS: dict[str, list[str]] = {
    # No single whole-brain mesh exists in 4.3 — the cerebrum is supplied as
    # gyri. Taking the gyri plus cerebellum and brainstem reconstitutes it.
    "brain": [
        r"^(left|right) .*gyrus$",
        r"^(anterior|posterior) part of (left|right) .*gyrus$",
        r"^(left|right) occipital lobe$",
        r"^(left|right) (thalamus|insula|cuneus|precuneus|hippocampus)$",
        r"^cerebellum$",
        r"^(pons|medulla oblongata)$",
    ],
    "spine": [
        r"^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"eleventh|twelfth) (cervical|thoracic|lumbar) vertebra$",
        r"^(sacrum|coccyx)$",
    ],
    "spinal_cord": [r"^neural tissue of spinal cord$"],
    "vagus_nerve": [r"^trunk of (left|right) vagus nerve$"],
    "gut": [
        r"^stomach$",
        r"^duodenum$",
        r"^(proximal|middle|distal) part of (jejunum|ileum)$",
        r"^(ascending|transverse|descending) colon$",
        r"^sigmoid colon$",
        r"^rectum$",
    ],
    # 4.3 has no single "heart" mesh: the organ is supplied as its chamber
    # walls, which together give the outer form.
    "heart": [
        r"^wall of ventricle$",
        r"^wall of (left|right) atrium$",
    ],
    # Nor a single lung — the parenchyma comes per bronchopulmonary segment.
    "lungs": [r"^parenchyma of .*bronchopulmonary segment$"],
    "blood_vessel": [
        r"^(ascending aorta|arch of aorta|descending aorta)$",
        r"^(superior|inferior) vena cava$",
        r"^trunk of (left|right) common carotid artery$",
        r"^(left|right) internal jugular vein$",
        r"^pulmonary trunk$",
    ],
    "leg_muscle": [
        r"^(left|right) (gluteus maximus|gluteus medius|gluteus minimus)$",
        r"^(left|right) (rectus femoris|vastus lateralis|vastus medialis|vastus intermedius)$",
        r"^(long|short) head of (left|right) biceps femoris$",
        r"^(left|right) (semitendinosus|semimembranosus|sartorius|gracilis)$",
        r"^(lateral|medial) head of (left|right) gastrocnemius$",
        r"^(left|right) (soleus|tibialis anterior|tibialis posterior)$",
    ],
    "skin": [r"^skin$"],
}


def _load_downloader() -> object:
    """Import the full-set downloader for its session, manifest and fetch code.

    Its filename carries a dot ("...4.3.py") so it is not importable by name;
    this is the supported way in, and keeps one implementation of the endpoint
    protocol rather than a second copy that can drift from it.
    """
    path = Path(__file__).with_name("download_bodyparts3d_4.3.py")
    spec = importlib.util.spec_from_file_location("bp3d_download", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_catalogue(meta_dir: Path) -> list[dict[str, str]]:
    """Rows of ``{fj, bp, fma, name}`` from the 4.3 obj2FMA table."""
    html = (meta_dir / "obj2FMA.html").read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    for row in re.findall(r"<tr.*?</tr>", html, re.DOTALL):
        cells = [
            re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL)
        ]
        if len(cells) >= 7 and cells[1].startswith("FJ") and cells[2]:
            rows.append({"fj": cells[1], "bp": cells[2], "fma": cells[3], "name": cells[4]})
    return rows


def select(rows: list[dict[str, str]], groups: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Group -> the catalogue rows its patterns match, de-duplicated by FJ id."""
    out: dict[str, list[dict]] = {}
    for group, patterns in groups.items():
        rx = [re.compile(p, re.IGNORECASE) for p in patterns]
        seen: set[str] = set()
        picked = []
        for r in rows:
            if r["fj"] in seen:
                continue
            if any(p.match(r["name"]) for p in rx):
                seen.add(r["fj"])
                picked.append(r)
        out[group] = picked
    return out


def fetch_group(
    dl: object,
    group: str,
    picked: list[dict[str, str]],
    out: Path,
    cookies: Path,
    *,
    chunk_size: int,
) -> None:
    """Download a group's meshes and unzip them under ``objs/<group>/``."""
    chunks_dir, objs_dir = out / "chunks", out / "objs" / group
    chunks_dir.mkdir(parents=True, exist_ok=True)
    objs_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(picked), chunk_size):
        batch = picked[i : i + chunk_size]
        zpath = chunks_dir / f"chunk_{group}_{i // chunk_size:03d}.zip"
        if not dl._zip_ok(zpath):
            dl.download_zip(
                [r["fj"] for r in batch],
                sorted({r["bp"] for r in batch}),
                zpath,
                cookies,
            )
            if not dl._zip_ok(zpath):
                raise RuntimeError(f"{zpath} is not a readable zip")
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.lower().endswith(".obj"):
                    continue
                dst = objs_dir / Path(name).name
                if not dst.exists():
                    dst.write_bytes(z.read(name))
        logger.info("  %s chunk %d: %d ids", group, i // chunk_size, len(batch))


def delivered(out: Path, group: str) -> dict[str, Path]:
    """FJ id -> the OBJ that actually landed for it, by filename prefix.

    The endpoint accepts an id list and returns whatever geometry it holds; a
    few catalogue concepts are grouping nodes with no mesh of their own and
    come back silently absent. Reconciling requested against delivered is what
    keeps the manifest a record of the files on disk rather than of an
    intention.
    """
    objs = (out / "objs" / group).glob("*.obj")
    return {p.name.split("_", 1)[0]: p for p in objs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/bodyparts3d/subset"))
    ap.add_argument(
        "--meta",
        type=Path,
        default=Path("data/bodyparts3d/raw_4.3/metadata"),
        help="Where the 4.3 manifests live (fetched there if absent).",
    )
    ap.add_argument(
        "--group",
        action="append",
        default=[],
        choices=sorted(GROUPS),
        help="Only these groups (repeatable). Default: all of them.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print what each group selects and exit, downloading nothing.",
    )
    ap.add_argument("--chunk-size", type=int, default=50)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    dl = _load_downloader()
    args.meta.mkdir(parents=True, exist_ok=True)
    cookies = args.meta.parent / "_session_cookies.txt"
    if not (args.meta / "obj2FMA.html").exists():
        dl.get_session(cookies)
        dl.fetch_fma2obj(args.meta, cookies)
        dl.fetch_obj2fma(args.meta, cookies)

    rows = read_catalogue(args.meta)
    wanted = {g: p for g, p in GROUPS.items() if not args.group or g in args.group}
    chosen = select(rows, wanted)

    for group, picked in chosen.items():
        logger.info("%s: %d meshes", group, len(picked))
        if args.list:
            for r in picked:
                logger.info("    %-10s %-12s %s", r["fj"], r["fma"], r["name"])
    if args.list:
        return 0

    empty = [g for g, p in chosen.items() if not p]
    if empty:
        # A pattern that matches nothing is a silent hole in the figure, not a
        # smaller download: fail loudly rather than render a missing system.
        raise SystemExit(f"no meshes matched for: {', '.join(empty)}")

    args.out.mkdir(parents=True, exist_ok=True)
    dl.get_session(cookies)
    manifest: list[dict[str, object]] = []
    missing: dict[str, list[str]] = {}
    for group, picked in chosen.items():
        logger.info("downloading %s (%d)", group, len(picked))
        fetch_group(dl, group, picked, args.out, cookies, chunk_size=args.chunk_size)
        have = delivered(args.out, group)
        for r in picked:
            obj = have.get(r["fj"])
            if obj is None:
                missing.setdefault(group, []).append(f"{r['fj']} {r['name']}")
                continue
            manifest.append(
                {
                    "group": group,
                    **r,
                    "obj": obj.relative_to(args.out).as_posix(),
                    "bytes": obj.stat().st_size,
                }
            )

    for group, names in missing.items():
        logger.warning(
            "%s: %d of %d requested meshes were not served (grouping concepts "
            "with no geometry of their own): %s",
            group,
            len(names),
            len(chosen[group]),
            ", ".join(names),
        )
    short = [g for g in chosen if not any(m["group"] == g for m in manifest)]
    if short:
        raise SystemExit(f"no geometry arrived for: {', '.join(short)}")

    csv_path = args.out / "MANIFEST.csv"
    # Written from what is on disk, so the manifest and the objs/ tree cannot
    # disagree about what this subset contains.
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, ["group", "fj", "bp", "fma", "name", "obj", "bytes"])
        w.writeheader()
        w.writerows(manifest)
    logger.info("wrote %s (%d meshes)", csv_path, len(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
