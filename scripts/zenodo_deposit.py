#!/usr/bin/env python3
"""Deposit the results bundle on Zenodo: reserve a DOI now, publish later.

The software itself is archived by the GitHub-Zenodo integration on each
tagged release. The solved leadfields are too large for that (hundreds of MB
each) and are not in git, so they go in a separate dataset record that
cites the software. This script does that in two deliberately separate
steps, because a published DOI cannot be withdrawn:

    # 1. create a draft, upload the bundle, reserve the DOI (reversible:
    #    a draft can be deleted on zenodo.org)
    python3 scripts/zenodo_deposit.py draft outputs/deposit/inob-results-*.tar.gz

    # 2. once the record looks right on zenodo.org, make it permanent
    python3 scripts/zenodo_deposit.py publish DEPOSITION_ID

The token (a Zenodo personal access token with deposit:write and
deposit:actions) is read from $ZENODO_TOKEN or ~/.config/zenodo/token.
Metadata come from .zenodo-results.json; nothing is sent without `draft`.
`--sandbox` uses sandbox.zenodo.org, whose DOIs are not real.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA = REPO_ROOT / ".zenodo-results.json"
TOKEN_FILE = Path.home() / ".config" / "zenodo" / "token"


def _token() -> str:
    tok = os.environ.get("ZENODO_TOKEN") or (
        TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else ""
    )
    if not tok:
        sys.exit(f"no Zenodo token: set $ZENODO_TOKEN or write it to {TOKEN_FILE}")
    return tok


def _call(method: str, url: str, token: str, body=None):
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as r:
        payload = r.read()
    return json.loads(payload) if payload else {}


def _upload(bucket: str, path: Path, token: str) -> None:
    # Streamed, so a 1 GB bundle is not held in memory.
    size = path.stat().st_size
    with path.open("rb") as fh:
        req = urllib.request.Request(
            f"{bucket}/{path.name}",
            data=fh,
            method="PUT",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
            },
        )
        with urllib.request.urlopen(req) as r:
            r.read()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft", help="create a draft, upload files, reserve the DOI")
    d.add_argument("files", nargs="+", type=Path)
    p = sub.add_parser("publish", help="publish a draft (permanent)")
    p.add_argument("deposition_id")
    args = ap.parse_args(argv)

    api = f"https://{'sandbox.' if args.sandbox else ''}zenodo.org/api/deposit/depositions"
    token = _token()
    if args.cmd == "draft":
        missing = [f for f in args.files if not f.exists()]
        if missing:
            sys.exit(f"missing: {', '.join(map(str, missing))}")
        if not METADATA.exists():
            sys.exit(f"no deposit metadata: write {METADATA.name} (see .zenodo.json for the shape)")
        meta = json.loads(METADATA.read_text())
        meta["prereserve_doi"] = True
        dep = _call("POST", api, token, {"metadata": meta})
        for f in args.files:
            print(f"uploading {f.name} ({f.stat().st_size / 1e6:.0f} MB)…", flush=True)
            _upload(dep["links"]["bucket"], f, token)
        doi = dep["metadata"]["prereserve_doi"]["doi"]
        print(f"draft {dep['id']}: reserved DOI {doi}")
        print(f"review it at {dep['links']['html']}, then: zenodo_deposit.py publish {dep['id']}")
    else:
        dep = _call("POST", f"{api}/{args.deposition_id}/actions/publish", token)
        print(f"published {dep['id']}: https://doi.org/{dep['doi']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
