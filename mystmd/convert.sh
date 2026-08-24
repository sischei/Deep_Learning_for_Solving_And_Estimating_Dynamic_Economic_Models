#!/bin/bash
# =============================================================================
# convert.sh — Vendored entry point for the claude-latex-to-myst pipeline.
# =============================================================================
#
# This script lives in the BOOK repository, not in claude-latex-to-myst.
# It fetches the tool into mystmd/_tools/claude-latex-to-myst (gitignored,
# self-managed) at the version pinned in .tool-version, then runs its
# convert pipeline against this directory's config.yaml. That path is
# resolved from the script's own location, not the caller's working
# directory — the documented invocation is `bash mystmd/convert.sh` from
# the repo root, so a leading `./` here would name the wrong directory.
#
# Everything the pipeline fetches or generates stays inside this folder —
# _tools/, tmp/ and _build/ are covered by mystmd/.gitignore — so the
# conversion adds exactly one directory to the book repo and leaves the
# repo-root .gitignore untouched.
#
# Pinning:
#   .tool-version contains a git ref — a tag (``v0.1.0``), a branch
#   (``main``), or a full SHA. Edit the file to upgrade/pin/downgrade.
#
# Usage:
#   bash mystmd/convert.sh                  # convert all chapters
#   bash mystmd/convert.sh --build          # convert + run `myst build --html`
#   bash mystmd/convert.sh ch_intro         # convert just one chapter
#
# Renderer:
#   Building this output needs the QuantEcon mystmd FORK
#   (github.com/QuantEcon/mystmd) at `qe-v10` or later — not upstream npm
#   `mystmd`. Both report `v1.10.1`; only the `(qe-v10)` suffix in
#   `myst --version` tells them apart. On an older renderer a starred
#   `\section*` renders its `{.unnumbered}` marker as literal text in the
#   heading title, so a stale pin corrupts pages rather than merely
#   forfeiting a fix. `--build` warns if the version on PATH is too old.
#
# Env overrides:
#   CLAUDE_LATEX_TO_MYST_URL    Alternate clone URL (forks, mirrors).
#   CLAUDE_LATEX_TO_MYST_TOOLS  Override the tool-checkout parent directory
#                               (default: mystmd/_tools). Useful to share one
#                               checkout across several books.
#
# Book-side post-conversion steps:
#   Add commands at the bottom of this script (after the tool delegates).
#   Common cases: render TikZ figures, generate llms.txt artifacts, run
#   project-specific validators. The wrapper deliberately does NOT use
#   `exec` to delegate, so anything you append after that line runs as
#   part of the normal `bash mystmd/convert.sh` invocation.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPO_URL="${CLAUDE_LATEX_TO_MYST_URL:-https://github.com/QuantEcon/claude-latex-to-myst.git}"
TOOLS_DIR="${CLAUDE_LATEX_TO_MYST_TOOLS:-$SCRIPT_DIR/_tools}"
TOOL_DIR="$TOOLS_DIR/claude-latex-to-myst"

VERSION_FILE="$SCRIPT_DIR/.tool-version"
if [[ -f "$VERSION_FILE" ]]; then
  VERSION="$(head -n1 "$VERSION_FILE" | tr -d '[:space:]')"
else
  VERSION="main"
fi

# ── 1. Ensure the tool checkout exists ──────────────────────────────────────
if [[ ! -d "$TOOL_DIR/.git" ]]; then
  echo "Cloning $REPO_URL → $TOOL_DIR"
  mkdir -p "$TOOLS_DIR"
  git clone --quiet "$REPO_URL" "$TOOL_DIR"
fi

# ── 2. Resolve the pinned ref and check out if needed ───────────────────────
resolve_ref() {
  # Try local ref first (tag / branch / SHA already known to the local repo).
  (cd "$TOOL_DIR" && git rev-parse --verify --quiet "$1^{commit}" 2>/dev/null) || true
}

CURRENT="$(cd "$TOOL_DIR" && git rev-parse HEAD)"
TARGET="$(resolve_ref "$VERSION")"

if [[ -z "$TARGET" ]]; then
  # Unknown locally — fetch from origin and retry.
  echo "Fetching origin to resolve '$VERSION'..."
  (cd "$TOOL_DIR" && git fetch --quiet --tags origin)
  TARGET="$(resolve_ref "$VERSION")"
  if [[ -z "$TARGET" ]]; then
    TARGET="$(resolve_ref "origin/$VERSION")"
  fi
fi

if [[ -z "$TARGET" ]]; then
  echo "ERROR: cannot resolve tool version '$VERSION' in $TOOL_DIR" >&2
  echo "       Check .tool-version; valid refs are tags, branches, or SHAs." >&2
  exit 1
fi

# For branches like ``main``, the pin is "wherever the branch points right
# now" — fast-forward to origin even if HEAD already points at *some*
# version of that branch, so we always pick up upstream updates.
if (cd "$TOOL_DIR" && git show-ref --verify --quiet "refs/remotes/origin/$VERSION"); then
  (cd "$TOOL_DIR" && git fetch --quiet origin "$VERSION")
  TARGET="$(resolve_ref "origin/$VERSION")"
fi

if [[ "$CURRENT" != "$TARGET" ]]; then
  echo "Checking out claude-latex-to-myst@$VERSION ($TARGET)"
  (cd "$TOOL_DIR" && git -c advice.detachedHead=false checkout --quiet "$TARGET")
fi

# ── 3. Announce and delegate ────────────────────────────────────────────────
SHORT="$(cd "$TOOL_DIR" && git rev-parse --short HEAD)"
echo "Using claude-latex-to-myst @ $VERSION ($SHORT)"
echo ""

# ── 4. Pre-conversion: render inline TikZ figures ──────────────────────────
# Discover \begin{figure} blocks (TikZ + \includegraphics) and emit
# mystmd/tikz_overrides.py BEFORE the upstream pipeline runs — the
# postprocess reads that file at startup to swap admonition placeholders
# for {figure} directives. mtime-cached, so a no-op re-run is fast.
#
# Uses `uv run --no-project python` to match the book-dp1/dp2 convention:
# avoids relying on whatever `python3` happens to be on PATH (macOS ships
# 3.9, which is too old for modern type-annotation syntax). uv resolves
# to a recent Python automatically. Falls back to plain `python3` if uv
# isn't installed — the render script uses `from __future__ import
# annotations` so it still runs on 3.9+.
if command -v uv >/dev/null 2>&1; then
  uv run --no-project python "$SCRIPT_DIR/scripts/render_tikz.py"
else
  python3 "$SCRIPT_DIR/scripts/render_tikz.py"
fi

bash "$TOOL_DIR/scripts/convert.sh" \
  --config "$SCRIPT_DIR/config.yaml" "$@"

# ── 5. Book-side post-conversion steps ──────────────────────────────────────
# Add any project-specific steps below — they run after the shared pipeline
# but as part of the same `bash mystmd/convert.sh` invocation, so a single
# command covers the whole regen workflow.

# Companion notebooks (issue #32): sync lectures/**/code/*.ipynb into
# notebooks/ (gitignored; needed by `myst build`), then regenerate the
# Appendix G listing + myst.yml notebook TOC and linkify the in-text
# `*.ipynb` mentions in the freshly converted chapters. Must run after the
# pipeline — the linkifier edits generated files. Same uv/python3 fallback
# rationale as render_tikz.py above.
if command -v uv >/dev/null 2>&1; then
  uv run --no-project python "$SCRIPT_DIR/scripts/sync_notebooks.py"
  uv run --no-project python "$SCRIPT_DIR/scripts/update_appendix.py"
else
  python3 "$SCRIPT_DIR/scripts/sync_notebooks.py"
  python3 "$SCRIPT_DIR/scripts/update_appendix.py"
fi
