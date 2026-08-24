# MyST Markdown rendering

This directory holds the MyST Markdown rendering of the lecture script, generated from the LaTeX source in [`../lecture_script/`](../lecture_script/) by [`QuantEcon/claude-latex-to-myst`](https://github.com/QuantEcon/claude-latex-to-myst).

The committed `*.md` files are **conversion output**, not human-authored. Edits should land in the LaTeX source (and propagate via re-conversion); editing a `.md` directly will be overwritten on the next run.

## Regenerating the output

```bash
bash mystmd/convert.sh                 # all chapters
bash mystmd/convert.sh ch01_intro      # one stem
bash mystmd/convert.sh --build         # convert + myst build --html
```

Prerequisites: [`uv`](https://github.com/astral-sh/uv) (used to bootstrap a recent Python — falls back to `python3` if absent) and `pandoc >= 3.0`. The wrapper script clones `claude-latex-to-myst` into `_tools/claude-latex-to-myst/` **inside this directory** (gitignored, self-managed) at the SHA pinned in [`.tool-version`](.tool-version). Override the location with `CLAUDE_LATEX_TO_MYST_TOOLS` if you want to share one checkout across books.

## Local preview

```bash
cd mystmd
python3 scripts/sync_notebooks.py      # once per checkout / after lectures/ changes
myst start                             # dev server on http://localhost:3000
myst build --html                      # static site into _build/html/
```

The sync step copies the companion notebooks from `../lectures/` into the gitignored `notebooks/` folder — the TOC lists one page per notebook, so a build without it fails on missing files. It is idempotent and also runs inside `convert.sh` and the CI workflow.

Requires the [mystmd CLI](https://mystmd.org/guide/installing). The repo also ships a GitHub Actions workflow ([`../.github/workflows/deploy-myst.yml`](../.github/workflows/deploy-myst.yml)) that builds and publishes to GitHub Pages on every push to **`main`** touching `mystmd/**`, `lectures/**` (the companion notebooks render as pages), or the workflow file itself, plus manual runs via `workflow_dispatch`. (The workflow is in its own path filter because a change to the renderer pin lives there, not under `mystmd/`, and must still trigger a rebuild.)

Publishing is deliberately tied to `main` alone: the published site should mirror the book's released state, and a feature branch publishing to the same Pages target would make the live site reflect whichever branch pushed last. While conversion work is in flight on a branch, verify with a local `myst build --html` or dispatch the workflow manually against that ref.

With GitHub Pages set to build from GitHub Actions, the site publishes to `https://<owner>.github.io/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models/`. It can also be published on demand from **Actions → MyST GitHub Pages Deploy → Run workflow**.

## What's in this directory

| Path | Role | Tracked? |
|---|---|---|
| `convert.sh` | Entry point — bootstraps the tool checkout, runs preprocess + pandoc + postprocess | ✅ |
| `.tool-version` | Pins `claude-latex-to-myst` to a specific SHA | ✅ |
| `config.yaml` | Per-book conversion config (chapter list, preprocess rewrites, TikZ map filename, extra envs, …) | ✅ |
| `myst.yml` | mystmd renderer config (project layout, KaTeX math macros) | ✅ |
| `tikz_overrides.py` | Map from `fig-…` label → rendered SVG path; written by `scripts/render_tikz.py` and read by the upstream postprocess | ✅ |
| `scripts/render_tikz.py` | Discovers `\begin{tikzpicture}` blocks in source, compiles to SVG via pdflatex + pdf2svg, updates `tikz_overrides.py` | ✅ |
| `scripts/sync_notebooks.py` | Copies `../lectures/**/code/*.ipynb` into `notebooks/` (with a frontmatter cell pointing source links back at `lectures/`) so the build renders each as a page | ✅ |
| `scripts/update_appendix.py` | Generates `appG_notebooks.md` + the notebook TOC block in `myst.yml`, and linkifies in-text `*.ipynb` mentions in the converted chapters; `--check` mode is the CI drift guard (issue #32) | ✅ |
| `appG_notebooks.md` | Web-only Appendix G listing all companion notebooks — generated, do not hand-edit | ✅ |
| `notebooks/` | Build-time copies of the companion notebooks, one rendered page each | gitignored |
| `figures/` | Compiled SVGs / curated raster figures | ✅ |
| `references.bib` | Bibliography (mirror of `../readings/bibliography.bib`, copied during convert) | ✅ |
| `index.md`, `preface.md`, `notation.md`, `ch??_*.md`, `appA…F_*.md` | Conversion output — 23 files (12 chapters + 6 appendices + 5 frontmatter) | ✅ |
| `VALIDATION.md` | Per-round validation report (structural counts, build warnings, round-to-round deltas) | ✅ |
| `.gitignore` | Ignore rules for everything the pipeline fetches or generates — keeps them out of the repo-root `.gitignore` | ✅ |
| `tmp/` | Per-chapter `.tex` slices produced by `preprocess.split:`, plus pandoc intermediate markdown | gitignored |
| `_build/` | mystmd HTML output | gitignored |
| `_tools/claude-latex-to-myst/` | Tool checkout, cloned at the pinned SHA | gitignored |

**Everything above lives inside this one directory.** The only file this conversion adds outside `mystmd/` is [`../.github/workflows/deploy-myst.yml`](../.github/workflows/deploy-myst.yml), which GitHub requires to sit in `.github/workflows/`. Nothing in `../lecture_script/` is modified — source-side problems are routed to issues rather than patched here — and the repo-root `.gitignore` is untouched. Adding the conversion to a book is one new directory plus one workflow file; removing it is `rm -rf mystmd/`.

## Bumping the tool version

Edit [`.tool-version`](.tool-version) to the new SHA (or a branch name like `main`), re-run `bash mystmd/convert.sh`, and check the diff. Each round in [`VALIDATION.md`](VALIDATION.md) documents one such bump with the resulting build state and any new issues filed.

> **These two pins move together.** `.tool-version` (which selects the converter that generates the markdown) and the `git checkout qe-v10` in [`../.github/workflows/deploy-myst.yml`](../.github/workflows/deploy-myst.yml) (which selects the renderer that builds the site) are coupled. Check the renderer floor whenever you move `.tool-version`. There are now two couplings in force, and they fail differently:
>
> - **`qe-v9` (Round 26, forgiving).** Since [QuantEcon/claude-latex-to-myst#201](https://github.com/QuantEcon/claude-latex-to-myst/pull/201) the converter emits non-starred `align` verbatim and relies on the fork's per-row equation numbering ([QuantEcon/mystmd#81](https://github.com/QuantEcon/mystmd/pull/81)). An older renderer does **not** break the site — it silently renders the *previous* equation numbering while the converter's changelog claims the numbering matches the printed PDF.
> - **`qe-v10` (Round 28, strict).** Since [QuantEcon/claude-latex-to-myst#209](https://github.com/QuantEcon/claude-latex-to-myst/pull/209) the converter emits `{.unnumbered}` on starred sections, which needs the fork's heading attribute blocks ([QuantEcon/mystmd#89](https://github.com/QuantEcon/mystmd/pull/89)). An older renderer **corrupts pages**: the block renders as literal braces in the heading title and pollutes its auto-slug.
>
> So: diff the built equation numbers against the PDF rather than against the previous build, and check that no heading title contains a literal `{`. To tell the two builds apart, read the **parenthesised fork tag** at the end of `myst --version` — `(qe-v10)` or newer is what you want, and upstream npm `mystmd` prints no tag at all. The numeric version ahead of it is not the check and should not be relied on (both lines happen to read `v1.10.1` today). `convert.sh --build` now warns when the tag is older than the floor or missing entirely.

## Known follow-ups

None blocks use of the rendering. [`VALIDATION.md`](VALIDATION.md) describes each in context.

| What | Waiting on |
|---|---|
| Ten `(a)`/`(i)` enumerate markers are escaped as literal text, because the stock book theme drops fancy-list `style`/`delimiter` and would render them `1./2./3.` | a theme that honours those attributes — [QuantEcon/mystmd#74](https://github.com/QuantEcon/mystmd/issues/74) |
| `fig:attention`'s legend is carried in its caption rather than beside the diagram | nothing; 1 of 88 figures has that structure, so a general fix is not warranted |
| An optional CI check that fails when the committed `.md` drifts from what the converter produces | nothing — deferred as optional |

Two upstream converter issues are open with no action needed here: [QuantEcon/claude-latex-to-myst#207](https://github.com/QuantEcon/claude-latex-to-myst/issues/207), where a bridging comma is dropped when a labelled `align` row is split out, so ch11's three sibling first-order conditions end `= 0` / `= 0,` / `= 0.`; and [QuantEcon/claude-latex-to-myst#210](https://github.com/QuantEcon/claude-latex-to-myst/issues/210), where a trailing brace group in a heading *title* is consumed if it parses as attributes — latent here, with zero occurrences book-wide.

The conversion's full development history, including every issue raised and resolved along the way, is in the [conversion fork](https://github.com/mmcky/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models/tree/mystmd-conversion/mystmd).
