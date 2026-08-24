"""Companion-notebook appendix + in-text linkification (issue #32).

The manuscript names its companion notebooks ~109 times across 13 chapters
but never links to any of them. This script closes that gap in three
committed artifacts, all generated from the ``lectures/`` tree as the single
source of truth:

1. **appG_notebooks.md** — a web-only "Companion Notebooks" appendix listing
   every notebook (including the ones the manuscript never mentions) with
   its role, the chapters that reference it, and a GitHub source link. This
   page is a deliberate exception to the high-fidelity mapping with the PDF,
   which has appendices A–F only.

2. **myst.yml** — the "Companion Notebooks" TOC section between the
   ``BEGIN/END generated-notebook-toc`` markers, one entry per notebook so
   each renders as its own page (from stored outputs; the build never
   executes). The .ipynb files themselves are synced into ``notebooks/`` at
   build time by scripts/sync_notebooks.py and are gitignored.

3. **The generated chapter .md files** — every `` `…ipynb` `` code-span
   mention is rewritten to an internal link to the notebook's page, which
   ``myst build`` then validates. Chapter files are regenerated from the
   LaTeX source by convert.sh, so this rewrite runs as a book-side
   post-conversion step (convert.sh step 5), never by hand.

Mentions resolve against the tree by suffix match (the prose uses short
forms like ``04_Loss_Normalization.ipynb`` for
``lecture_05_04_Loss_Normalization.ipynb``); resolution must be unique and
failures are fatal. Mentions of files that live in OTHER repositories are
declared in EXTERNAL_MENTIONS and left untouched — see issue #32 for why a
naive "every mention must resolve" gate is wrong.

Usage:
    python3 mystmd/scripts/update_appendix.py           # regenerate in place
    python3 mystmd/scripts/update_appendix.py --check   # CI drift guard:
        exit 1 if any artifact differs from what the tree implies, or if a
        committed notebook link points at a notebook that no longer exists.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MYSTMD_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MYSTMD_DIR.parent
LECTURES_DIR = REPO_ROOT / "lectures"
APPENDIX_PATH = MYSTMD_DIR / "appG_notebooks.md"
MYST_YML_PATH = MYSTMD_DIR / "myst.yml"

TOC_BEGIN = "# BEGIN generated-notebook-toc"
TOC_END = "# END generated-notebook-toc"

# Mentions that name notebooks in OTHER repositories. They read like local
# companion notebooks but are external citations; leave them untouched.
#   ch06_ha_youngs.md: "adapted from the upstream tutorial
#   `01_KrusellSmith_Tutorial_CPU.ipynb` in the companion code repository …
#   github.com/azinoma/DeepLearningInTheSequenceSpace"
EXTERNAL_MENTIONS = {
    "01_KrusellSmith_Tutorial_CPU.ipynb",
}

# Role letters come from the Execution Map (front matter): c = core,
# e = exercise, s = solution, x = extension/self-study. Exercise/solution
# are recoverable from filenames; the extensions are declared here because
# nothing in the filename marks them. Keep in step with execution_map.md.
EXTENSION_NOTEBOOKS = {
    "lecture_10_KrusellSmith_Tutorial_CPU.ipynb",
    "lecture_14_09_Deep_Active_Subspace_Ridge.ipynb",
    "lecture_14_10_Deep_AS_vs_Linear_AS_Borehole.ipynb",
}

# Files at the mystmd/ root that the linkifier must not touch: the appendix
# itself (already generated with links), and pipeline docs where an .ipynb
# code span is not book prose.
LINKIFY_EXCLUDE = {APPENDIX_PATH.name, "README.md", "VALIDATION.md"}

MENTION_RE = re.compile(r"`([^`\n]*?\.ipynb)`")
EXISTING_TARGET_RE = re.compile(r"\]\((notebooks/[^)\s]+)\)")


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

class Notebook:
    def __init__(self, src: Path):
        self.src = src
        self.name = src.name
        self.lecture = src.parent.parent.name  # lecture_NN_topic
        self.relpath = src.relative_to(REPO_ROOT).as_posix()
        self.page = f"notebooks/{self.lecture}/{self.name}"
        self.title = self._first_h1()
        self.role = self._role()
        self.referenced_in: list[str] = []  # chapter stems, discovery order

    def _first_h1(self) -> str:
        try:
            nb = json.loads(self.src.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return self.name
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            for line in "".join(cell.get("source", [])).splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
        return self.name

    def _role(self) -> str:
        if self.name in EXTENSION_NOTEBOOKS:
            return "extension"
        lowered = self.name.lower()
        if "solution" in lowered:
            return "solution"
        if "exercise" in lowered or "blanks" in lowered:
            return "exercise"
        return "core"


def build_manifest() -> list[Notebook]:
    sources = sorted(LECTURES_DIR.glob("lecture_*/code/*.ipynb"))
    if not sources:
        sys.exit(f"ERROR: no notebooks found under {LECTURES_DIR}")
    manifest = [Notebook(s) for s in sources]
    names = [n.name for n in manifest]
    dupes = {x for x in names if names.count(x) > 1}
    if dupes:
        # MyST flattens page slugs to the basename, so duplicates collide.
        sys.exit(f"ERROR: duplicate notebook basenames (slug collision): {dupes}")
    return manifest


def resolve(mention: str, manifest: list[Notebook]) -> Notebook | None:
    """Resolve a prose mention to a manifest entry, or None for external.

    Raises ValueError when a local-looking mention resolves to zero or to
    several notebooks — both mean the manuscript and the tree disagree.
    """
    m = mention.lstrip("./")
    if m in EXTERNAL_MENTIONS:
        return None
    if "/" in m:
        cands = [
            n for n in manifest
            if n.relpath == m
            or (n.relpath.endswith("/" + m))
        ]
    else:
        cands = [n for n in manifest if n.name == m]
        if not cands:
            cands = [n for n in manifest if n.name.endswith("_" + m)]
    if len(cands) == 1:
        return cands[0]
    kind = "ambiguous" if cands else "unresolved"
    raise ValueError(
        f"{kind} notebook mention `{mention}`"
        + (f" -> {[n.relpath for n in cands]}" if cands else "")
    )


# --------------------------------------------------------------------------
# Linkification
# --------------------------------------------------------------------------

# Directive fences whose bodies are literal content rather than nested MyST
# markdown — a notebook mention inside one is code or math, not prose to
# link. Everything else in the {…} fence form ({table}, {exercise},
# {prf:*}, …) nests markdown and must stay linkifiable: the chapter
# exercises mention notebooks from inside those fences.
LITERAL_DIRECTIVES = {
    "code",
    "code-block",
    "code-cell",
    "literalinclude",
    "raw",
    "math",
}


def linkifiable_regions(text: str) -> list[tuple[int, int]]:
    """Spans of *text* that are prose, i.e. not inside literal fences.

    Directive fences (```{table} …) contain nested MyST markdown, so their
    bodies stay linkifiable — except LITERAL_DIRECTIVES, whose bodies are
    code or math. Plain/language fences are literal code. One refinement:
    mystmd renders a directive's ``:caption:`` option as markdown even when
    the directive body is literal (verified on ch02's ``{code-block}``
    caption, which renders its notebook link as a real ``<a>``), so caption
    option lines inside a literal directive stay linkifiable.
    """
    regions: list[tuple[int, int]] = []
    # (tick count, is_prose, is_directive)
    fence_stack: list[tuple[int, bool, bool]] = []
    pos = 0
    run_start: int | None = 0

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        m = re.match(r"(`{3,})(?:\{([^}\s]+)\})?", stripped)
        if m and m.group(1):
            ticks = len(m.group(1))
            if (
                fence_stack
                and fence_stack[-1][0] == ticks
                and not stripped[ticks:].strip()
            ):
                fence_stack.pop()
            else:
                name = m.group(2)
                is_prose = name is not None and name not in LITERAL_DIRECTIVES
                fence_stack.append((ticks, is_prose, name is not None))
            line_prose = False  # the delimiter line itself is never prose
        elif not any(not is_prose for _, is_prose, _ in fence_stack):
            line_prose = True
        else:
            # Inside literal content. Linkifiable only if it is a :caption:
            # option of a directly enclosing literal DIRECTIVE (not a plain
            # code fence) with nothing but prose fences above it.
            _, innermost_prose, innermost_directive = fence_stack[-1]
            line_prose = (
                not innermost_prose
                and innermost_directive
                and all(p for _, p, _ in fence_stack[:-1])
                and stripped.startswith(":caption:")
            )
        if line_prose and run_start is None:
            run_start = pos
        elif not line_prose and run_start is not None:
            regions.append((run_start, pos))
            run_start = None
        pos += len(line)
    if run_start is not None:
        regions.append((run_start, pos))
    return regions


def process_file(
    path: Path, manifest: list[Notebook], mention_log: list[tuple[str, Notebook]]
) -> tuple[str, int, list[str]]:
    """Return (new_text, rewrites, errors) and log resolved mentions."""
    text = path.read_text(encoding="utf-8")
    regions = linkifiable_regions(text)
    out: list[str] = []
    errors: list[str] = []
    rewrites = 0
    cursor = 0

    def in_region(i: int) -> bool:
        return any(a <= i < b for a, b in regions)

    for m in MENTION_RE.finditer(text):
        if not in_region(m.start()):
            continue
        mention = m.group(1)
        try:
            nb = resolve(mention, manifest)
        except ValueError as e:
            line = text.count("\n", 0, m.start()) + 1
            errors.append(f"{path.name}:{line}: {e}")
            continue
        if nb is None:
            continue  # declared external
        mention_log.append((path.stem, nb))

        already_linked = m.start() > 0 and text[m.start() - 1] == "["
        if already_linked:
            t = EXISTING_TARGET_RE.match(text, m.end())
            if t and t.group(1) == nb.page:
                continue  # correct link already in place
            if t:  # stale target (e.g. notebook renamed) — rewrite it
                out.append(text[cursor : m.end()])
                out.append(f"]({nb.page})")
                cursor = t.end()
                rewrites += 1
                continue
            continue  # linked to something unexpected; leave alone
        out.append(text[cursor : m.start()])
        out.append(f"[`{mention}`]({nb.page})")
        cursor = m.end()
        rewrites += 1

    out.append(text[cursor:])
    return "".join(out), rewrites, errors


# --------------------------------------------------------------------------
# Appendix page
# --------------------------------------------------------------------------

def lecture_title(lecture_folder: str) -> str:
    readme = LECTURES_DIR / lecture_folder / "README.md"
    if readme.is_file():
        first = readme.read_text(encoding="utf-8").splitlines()[0]
        if first.startswith("# "):
            return first[2:].strip()
    return lecture_folder.replace("_", " ")


def chapter_label(stem: str) -> str:
    m = re.match(r"ch(\d+)_", stem)
    if m:
        return f"Ch. {int(m.group(1))}"
    m = re.match(r"app([A-Z])_", stem)
    if m:
        return f"App. {m.group(1)}"
    return stem.replace("_", " ").capitalize()


def chapter_sort_key(stem: str):
    m = re.match(r"ch(\d+)_", stem)
    if m:
        return (0, int(m.group(1)), "")
    m = re.match(r"app([A-Z])_", stem)
    if m:
        return (1, 0, m.group(1))
    return (2, 0, stem)


def github_base() -> str:
    m = re.search(
        r"^\s*github:\s*(\S+)", MYST_YML_PATH.read_text(encoding="utf-8"), re.M
    )
    return m.group(1).rstrip("/") if m else ""


def render_appendix(manifest: list[Notebook]) -> str:
    gh = github_base()
    total = len(manifest)
    mentioned = sum(1 for n in manifest if n.referenced_in)
    lines = [
        "---",
        'title: "Companion Notebooks"',
        "label: companion-notebooks",
        "---",
        "",
        "<!-- GENERATED FILE — do not hand-edit. -->",
        "<!-- Regenerated by scripts/update_appendix.py, which runs as part of",
        "     `bash mystmd/convert.sh` (book-side step 5). See issue #32. -->",
        "",
        "This appendix exists only in the web edition of the book — the PDF has",
        "appendices A–F. It lists every companion notebook in the repository's",
        f"[`lectures/`]({gh}/tree/main/lectures) tree ({total} notebooks, of",
        f"which {mentioned} are referenced from the text), rendered here as",
        "browsable pages showing the outputs the authors committed. Nothing is",
        "re-executed for the web build; to run a notebook yourself, see the",
        "seed, `RUN_MODE`, and hardware conventions in",
        "[Appendix E](appE_reproducibility.md).",
        "",
        "Role letters follow the [Execution Map](execution_map.md): core",
        "notebooks accompany the chapter text, exercise/solution pairs support",
        "the end-of-chapter problems, and extensions are self-study material.",
        "",
    ]
    for lecture in sorted({n.lecture for n in manifest}):
        lines.append(f"## {lecture_title(lecture)}")
        lines.append("")
        lines.append("| Notebook | Role | Referenced in | Source |")
        lines.append("|---|---|---|---|")
        for n in [x for x in manifest if x.lecture == lecture]:
            refs = (
                ", ".join(
                    f"[{chapter_label(s)}]({s}.md)"
                    for s in sorted(set(n.referenced_in), key=chapter_sort_key)
                )
                or "—"
            )
            src = f"[GitHub]({gh}/blob/main/{n.relpath})" if gh else "—"
            lines.append(f"| [`{n.name}`]({n.page}) | {n.role} | {refs} | {src} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# myst.yml TOC block
# --------------------------------------------------------------------------

def render_toc_block(manifest: list[Notebook]) -> str:
    lines = [
        "    - title: Companion Notebooks",
        "      children:",
    ]
    for lecture in sorted({n.lecture for n in manifest}):
        lines.append(f'        - title: "{lecture_title(lecture)}"')
        lines.append("          children:")
        for n in [x for x in manifest if x.lecture == lecture]:
            lines.append(f"            - file: {n.page}")
    return "\n".join(lines) + "\n"


def splice_toc(yml_text: str, block: str) -> str:
    lines = yml_text.splitlines(keepends=True)
    begin = end = None
    for i, line in enumerate(lines):
        if TOC_BEGIN in line:
            begin = i
        elif TOC_END in line:
            end = i
    if begin is None or end is None or end < begin:
        sys.exit(
            f"ERROR: {TOC_BEGIN!r}/{TOC_END!r} markers not found in myst.yml — "
            "cannot splice the notebook TOC."
        )
    return "".join(lines[: begin + 1]) + block + "".join(lines[end:])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    check = "--check" in sys.argv[1:]
    manifest = build_manifest()

    md_files = sorted(
        p for p in MYSTMD_DIR.glob("*.md") if p.name not in LINKIFY_EXCLUDE
    )
    mention_log: list[tuple[str, Notebook]] = []
    pending: dict[Path, str] = {}
    errors: list[str] = []
    total_mentions = 0
    total_rewrites = 0
    for path in md_files:
        new_text, rewrites, errs = process_file(path, manifest, mention_log)
        errors.extend(errs)
        if rewrites:
            pending[path] = new_text
            total_rewrites += rewrites

    total_mentions = len(mention_log)
    for stem, nb in mention_log:
        if stem not in nb.referenced_in:
            nb.referenced_in.append(stem)

    # Stale-link scan on the post-rewrite text: any surviving notebook link
    # must point at a notebook that exists (process_file already repairs the
    # ones that sit next to a resolvable mention).
    valid_pages = {n.page for n in manifest}
    for path in md_files:
        text = pending.get(path, path.read_text(encoding="utf-8"))
        for m in EXISTING_TARGET_RE.finditer(text):
            if m.group(1) not in valid_pages:
                line = text.count("\n", 0, m.start()) + 1
                errors.append(
                    f"{path.name}:{line}: link target {m.group(1)} has no "
                    "matching notebook on disk"
                )

    if errors:
        print("update_appendix: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    appendix = render_appendix(manifest)
    yml_new = splice_toc(
        MYST_YML_PATH.read_text(encoding="utf-8"), render_toc_block(manifest)
    )

    drift: list[str] = []
    if (
        not APPENDIX_PATH.is_file()
        or APPENDIX_PATH.read_text(encoding="utf-8") != appendix
    ):
        drift.append(str(APPENDIX_PATH.relative_to(REPO_ROOT)))
    if MYST_YML_PATH.read_text(encoding="utf-8") != yml_new:
        drift.append("mystmd/myst.yml (generated-notebook-toc block)")
    drift.extend(str(p.relative_to(REPO_ROOT)) for p in pending)

    if check:
        if drift:
            print("update_appendix --check: FAILED — committed artifacts are", file=sys.stderr)
            print("out of step with the lectures/ tree. Stale:", file=sys.stderr)
            for d in drift:
                print(f"  {d}", file=sys.stderr)
            print(
                "Run `python3 mystmd/scripts/update_appendix.py` (or the full "
                "`bash mystmd/convert.sh`) and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(
            f"update_appendix --check: OK — {len(manifest)} notebooks, "
            f"{total_mentions} in-text mentions all resolved and linked."
        )
        return 0

    for path, new_text in pending.items():
        path.write_text(new_text, encoding="utf-8")
    APPENDIX_PATH.write_text(appendix, encoding="utf-8")
    MYST_YML_PATH.write_text(yml_new, encoding="utf-8")
    print(
        f"update_appendix: {len(manifest)} notebooks; {total_mentions} mentions "
        f"resolved ({total_rewrites} newly linkified); "
        f"{sum(1 for n in manifest if not n.referenced_in)} notebooks never "
        "mentioned in the text (listed in the appendix)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
