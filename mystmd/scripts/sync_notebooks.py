"""Sync companion notebooks into the MyST project (issue #32).

Copies every ``lectures/lecture_*/code/*.ipynb`` into
``mystmd/notebooks/<lecture_folder>/`` so the book's MyST build can render
each notebook as its own page. The destination is gitignored — ``lectures/``
stays the single source of truth and the copies exist only at build time,
mirroring how ``_tools/`` and ``tmp/`` are handled.

Run it before ``myst build``:

    python3 mystmd/scripts/sync_notebooks.py

It also runs as part of ``bash mystmd/convert.sh`` (book-side step 5) and as
a CI step in .github/workflows/deploy-myst.yml. Idempotent: each copy
mirrors its source's mtime, so a source whose mtime matches its copy is
skipped, and files in the destination whose source has disappeared are
pruned — a notebook rename upstream cannot leave a stale page behind.

Only ``.ipynb`` files are carried. Verified against the current tree: no
notebook references a local image or data file from a markdown cell (the one
``<img>`` book-wide points at a remote URL), and the build never executes
notebooks, so runtime data files are irrelevant to rendering.

Each copy gets a leading frontmatter cell (MyST consumes it; it never
renders) that points the page's ``source_url`` at the real
``lectures/…/code/…`` file and suppresses ``edit_url`` — without this the
theme's source/edit buttons target the gitignored ``mystmd/notebooks/``
path, which does not exist on GitHub. No source notebook carries a leading
YAML cell of its own (verified), so prepending is safe.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

MYSTMD_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MYSTMD_DIR.parent
LECTURES_DIR = REPO_ROOT / "lectures"
DEST_DIR = MYSTMD_DIR / "notebooks"


def github_base() -> str:
    m = re.search(
        r"^\s*github:\s*(\S+)",
        (MYSTMD_DIR / "myst.yml").read_text(encoding="utf-8"),
        re.M,
    )
    return m.group(1).rstrip("/") if m else ""


def copy_with_frontmatter(src: Path, dest: Path, gh: str) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    relpath = src.relative_to(REPO_ROOT).as_posix()
    fm = f"---\nsource_url: {gh}/blob/main/{relpath}\nedit_url: null\n---"
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": fm.splitlines(keepends=True),
    }
    # nbformat 4.5+ requires an `id` on every cell; earlier minors do not
    # define the key. 44 of the 68 source notebooks are 4.5, the rest
    # 4.0–4.4, so the id is added only where the schema expects it.
    if nb.get("nbformat", 0) >= 4 and nb.get("nbformat_minor", 0) >= 5:
        cell["id"] = "synced-frontmatter"
    nb.setdefault("cells", []).insert(0, cell)
    dest.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    # Mirror the source mtime so an unchanged source is skipped next run.
    stat = src.stat()
    os.utime(dest, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def source_notebooks() -> dict[Path, Path]:
    """Map destination path -> source path for every companion notebook."""
    mapping: dict[Path, Path] = {}
    for src in sorted(LECTURES_DIR.glob("lecture_*/code/*.ipynb")):
        lecture_folder = src.parent.parent.name
        mapping[DEST_DIR / lecture_folder / src.name] = src
    return mapping


def main() -> int:
    if not LECTURES_DIR.is_dir():
        print(f"ERROR: {LECTURES_DIR} not found", file=sys.stderr)
        return 1

    mapping = source_notebooks()
    if not mapping:
        print(f"ERROR: no notebooks found under {LECTURES_DIR}", file=sys.stderr)
        return 1

    gh = github_base()
    copied = skipped = 0
    for dest, src in mapping.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_mtime_ns == src.stat().st_mtime_ns:
            skipped += 1
            continue
        copy_with_frontmatter(src, dest, gh)
        copied += 1

    pruned = 0
    if DEST_DIR.is_dir():
        for stale in DEST_DIR.rglob("*"):
            if stale.is_file() and stale not in mapping:
                stale.unlink()
                pruned += 1
        # Remove directories left empty by pruning (deepest first).
        for d in sorted(
            (p for p in DEST_DIR.rglob("*") if p.is_dir()), reverse=True
        ):
            if not any(d.iterdir()):
                d.rmdir()

    print(
        f"sync_notebooks: {len(mapping)} notebooks — "
        f"{copied} copied, {skipped} up to date, {pruned} stale pruned"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
