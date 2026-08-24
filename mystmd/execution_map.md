---
title: "Execution Map"
label: execution-map
---

The following table maps each manuscript chapter to its companion slide deck(s) and Jupyter notebooks. All paths are relative to the repository root; the short names are intentionally compact so the table remains readable.

````{table}

Execution map: manuscript chapters, slides, and notebooks

| **Ch.** | **Topic** | **Lecture folder & deck** | **Notebooks (role)** |
|:---:|---|---|---|
| 1 | Intro to ML & DL | L02: `intro_deep_learning` | L02 `01`–`09` (c$\times$9) |
| 2 | Deep Equilibrium Nets | L03: `deep_equilibrium_nets`; L07: `autodiff_for_deqns` | L03 `01`–`02` (c), `03`–`04` (e/s), `05` (c); L07 `01`–`04` (c) |
| 3 | IRBC Model | L04: `irbc` | L04 `01`–`02` (c) |
| 4 | NAS & Loss Norm. | L05: `neural_architecture_search`, `loss_normalization` | L05 `02`–`04` (c), `05` (e) |
| 5 | OLG Models | L08: `olg_models_deqns` | L08 `07`–`10` (c), `11` (e) |
| 6 | HA, Young, Seq. Space | L09: `heterogeneous_agents_youngs`; L10: `sequence_space_deqns` | L09 `10`–`12` (c); L10 `05`, `05b`, `06` (c), `KrusellSmith_Tutorial_CPU` (x) |
| 7 | PINNs | L11: `pinns` | L11 `01`–`05` (c) |
| 8 | CT Het. Agents | L12: `continuous_time_ha_theory`; L13: `continuous_time_ha_numerics` | L13 `08` (c) |
| 9 | Surrogates, GPs, DKL | L14: `surrogates_and_gps` | L14 `01`, `02`, `04`–`08` (c), `09`, `10` (x) |
| 10 | Structural Estimation | L15: `structural_estimation_smm` | L15 `03`, `03b` (c) |
| 11 | Climate & Deep UQ | L16: `climate_economics_iams`; L17: `deep_uq_pareto_policy` | L16 `01`–`03` (c); L17 `09_DICE_2P_UQ_Analysis` (c) plus 4 `.py` pipeline drivers |
| 12 | Synthesis & Outlook | L18: `wrap_up` | — |
````

**Path conventions.** Lecture folders are `lectures/lecture_`$NN$`_*/`, where `L`$NN$ in the table is the lecture number; the slide sources live in that folder's `slides/` subfolder and the notebooks in `code/`. Names in the table are abbreviated: deck names drop the `lecture_`$NN$`_` prefix that every slide file carries, and a notebook is cited by the number that follows that prefix, so `L03 01` is [`lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb). Notebook role letters: `c` = core, `e` = exercise, `s` = solution (paired with an exercise notebook), `x` = extension/self-study. See the README for complete file names and direct links.

**Lectures without a chapter.** Two lectures have no manuscript chapter of their own. L01 (`lecture_01_python_primer`) is a self-contained Python primer for readers who want a refresher before {ref}`ch-intro`. L06 (`lecture_06_agentic_programming`) is a hands-on workshop on agentic programming (using AI agents as coding partners); because this field is evolving quickly, it is presented through slides, two Python helper scripts, and exercise prompts rather than as a fixed manuscript chapter.

**Reproducibility.** Random-seed conventions, the `RUN_MODE` budget split, hardware and software pins, and GPU-determinism flags used by every notebook in the table above are documented in Appendix {ref}`app-reproducibility`. Worked solutions and guidance for the end-of-chapter exercises are collected in Appendix {ref}`app-solutions`.
