---
title: "Neural Architecture Search and Loss Normalization"
label: ch-nas
---

The DEQN models of {ref}`ch-deqn`–{ref}`ch-irbc` involve several hyperparameters (network depth, width, activation functions, learning rate) and multi-component loss functions whose relative scales can differ by orders of magnitude. To fix ideas, even a modest sweep over depth $\in\{1,\ldots,10\}$ and width $\in\{16, 32, 64, 128, 256, 512\}$ on the companion NAS regression task already spans $10\times 6 = 60$ configurations, and the production sweep below (six axes) reaches $\sim 3{,}000$; on that task (illustrative numbers), the best mean absolute error ($\approx 3\times 10^{-3}$) is attained at a $5\times 256$ network, while $10\times 512$ overfits by almost an order of magnitude ($\approx 2\times 10^{-2}$). Hand-tuning at this scale is infeasible. This chapter addresses both challenges. We first survey hyperparameter-search methods (random search, Bayesian optimization, Hyperband, and BOHB, which combines TPE with Hyperband, {cite:p}`bergstra2012random,snoek2012practical,jamieson2016nonstochastic,li2018hyperband,falkner2018bohb,garnett2023bayesian`, and then develop a classroom-friendly version of the ReLoBRaLo algorithm {cite:p}`bischof2025relobralo` for adaptive multi-objective loss balancing. In the notebooks, this is implemented as a deterministic convex blend of step-wise and baseline loss comparisons, which keeps the code compact while preserving the balancing intuition.

A terminology note before we begin: in this chapter we use "NAS" loosely to cover both *hyperparameter optimization* (HPO; choosing widths, depths, activations, learning rates from a fixed parameterization) and "true" NAS in the sense of {cite:t}`elsken2019neural`, where the network's wiring graph itself is searched (e.g. regularized evolution {cite:p}`real2019regularized`). The two literatures share methodology (Random / Bayesian / Hyperband-style search) but differ in scope. All four methods discussed here are HPO; for graph-level NAS, the textbook reference is {cite:t}`hutter2019automl`. {cite:t}`elsken2019neural` provide the canonical survey; the local copy in `readings/` is recommended as the first deep-dive reference.

**Hands-on notebooks for this chapter.** Two NAS walkthroughs are provided alongside the ReLoBRaLo notebook, plus the IRBC exercise notebook that doubles as the entry point to this chapter. All four live in the NAS chapter's code folder:

- [`02_NAS_Random_Search_10D.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_02_NAS_Random_Search_10D.ipynb): a library-free Random Search loop (model in TF/Keras) on a 10-dimensional analytical regression task, used to illustrate the projection argument of {cite:t}`bergstra2012random` in its cleanest form.

- [`03_NAS_RandomSearch_Hyperband.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb): from-scratch Random Search and Successive Halving (Hyperband's inner loop) on a two-dimensional Genz Gaussian, written in $\sim$25 lines of plain Python so the algorithms in this chapter are visible without library abstraction; after the first run, the cached records in `nas_results/` short-circuit re-runs for instant re-inspection.

- [`04_Loss_Normalization.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_04_Loss_Normalization.ipynb): the classroom ReLoBRaLo implementation, matched to the notation below.

- [`05_IRBC_Exercise.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_05_IRBC_Exercise.ipynb): the IRBC exercise notebook (closed-form steady-state comparative statics and inverse-loss weighting on a multi-component IRBC residual); it is the notebook referenced by {ref}`ch-irbc` {prf:ref}`ex-ch3-6`–{prf:ref}`ex-ch3-7`, and it reuses the loss-balancing ideas of this chapter on a deliberately small, library-free example.

## The Hyperparameter Space

The performance of a neural network depends sensitively on its architecture (number of layers, neurons per layer, activation functions) and training configuration (learning rate, batch size, optimizer, regularization). Choosing these hyperparameters by hand is tedious and often suboptimal.

To appreciate the scale of the search problem, consider as a stylized example a typical DEQN setup where we must select: the number of hidden layers ($L \in \{2, 3, 4, 5\}$), neurons per layer ($n \in \{32, 64, 128, 256\}$), activation function (ReLU, Swish, Tanh), learning rate ($\eta \in \{10^{-4}, 5\times 10^{-4}, 10^{-3}, 5\times 10^{-3}\}$), batch size ($B \in \{64, 128, 256, 512\}$), and weight decay ($\lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}\}$). The total number of configurations is $4 \times 4 \times 3 \times 4 \times 4 \times 4 = 3{,}072$. If each configuration requires 30 minutes to train, exhaustive evaluation would take 64 days on a single GPU. With additional choices (optimizer type, learning rate schedule, dropout rate), the space easily exceeds $10^4$ configurations. (The slide deck uses a slightly larger illustrative space, $5 \times 8 \times 3 \times 20 \times 4 = 9{,}600$ configurations, for the same point.)

## Grid Search

The simplest approach is to define a grid of values for each hyperparameter and evaluate all combinations. For $d$ hyperparameters, each taking $k$ values, the cost is $\mathcal{O}(k^d)$, the same exponential scaling that plagues grid-based PDE solvers. Grid search is deterministic and easy to implement, but it has a fundamental flaw: it allocates the same density of evaluations to all hyperparameters, including those to which performance is insensitive. If only 2 out of 6 hyperparameters matter (which is typical in practice), the remaining 4 dimensions contribute only wasted computation.

(sec-nas_random_search)=
## Random Search
{cite:t}`bergstra2012random` demonstrated that random sampling of hyperparameter configurations often outperforms grid search, particularly when only a few hyperparameters are important. The key insight is a *projection argument*: when a random configuration is projected onto any single hyperparameter axis, the marginal distribution covers the entire range densely, regardless of how many other hyperparameters exist. In contrast, a grid with the same total number of evaluations provides only $k = N^{1/d}$ distinct values per axis, which can be very coarse in high dimensions.

For example, with a budget of $N = 60$ evaluations in $d = 6$ dimensions, a grid provides only $60^{1/6} \approx 2$ values per hyperparameter, while random search provides 60 distinct values per hyperparameter (in the marginal sense). This makes random search much more likely to find good values for the hyperparameters that matter most. {numref}`fig-grid_vs_random` shows the same projection argument in two dimensions.

```{figure} figures/fig-grid_vs_random.svg
:name: fig-grid_vs_random

Why random search beats grid search when only one hyperparameter matters. Both designs spend the same budget of nine evaluations on a two-dimensional space in which only the horizontal axis affects performance; the vertical axis is a "nuisance" hyperparameter. Project each design onto that important axis (the strip below each panel): the $3\times 3$ grid stacks its nine points into only three columns, so it probes just three distinct values of the parameter that matters, whereas the random design lands on nine distinct values. Equivalently, with a budget of $N$ points in $d$ dimensions a grid resolves only $N^{1/d}$ values per axis no matter which axes matter, while a random design resolves $N$ values per axis in the marginal sense. Since in practice only two or three of many hyperparameters typically drive performance {cite:p}`bergstra2012random`, the random design extracts far more information about the dimensions that count for the same compute.
```

## Bayesian Optimization

Bayesian optimization (BO) treats the validation loss $\ell(\bm{h})$ as an expensive black-box function of the hyperparameter vector $\bm{h}$, and uses a probabilistic surrogate, fitted to the configurations evaluated so far, to decide where to evaluate next {cite:p}`snoek2012practical`. The standard surrogate is a Gaussian process (GP). The next subsection gives a self-contained primer in the notation that {ref}`ch-gp` will reuse, so the reader can follow the BO logic without flipping ahead; the full kernel zoo, marginal-likelihood hyperparameter learning, and Bayesian active learning are developed there. Throughout this section the GP input is the hyperparameter vector itself, $\x \equiv \bm{h}$, and the kernel length-scale symbol $\ell$ that appears in {eq}`eq-nas_kse` below is the standard GP notation of {ref}`ch-gp`; it is distinct from the validation-loss symbol $\ell(\bm{h})$ and the two are always clear from context.

(sec-nas_gp_primer)=
### A Brief Introduction to Gaussian Processes
A Gaussian process is a probability distribution over real-valued functions, equivalently, a random function $f$ whose values at any finite collection of inputs $\x_1, \ldots, \x_n$ are jointly Gaussian {cite:p}`Rasmussen:2005:GPM:1162254`. It is fully specified by a mean function $\mu(\x)$ and a covariance (kernel) function $k(\x, \x')$ that says how correlated the function's values at two inputs are:

$$
f(\x) \sim \mathcal{GP}\bigl(\mu(\x),\; k(\x, \x')\bigr).
$$

This means that, for any finite test set $\x_1, \ldots, \x_n$, the vector $\bm{f} = (f(\x_1), \ldots, f(\x_n))^\top$ is distributed as $\bm{f} \sim \mathcal{N}(\bm{\mu}, K)$ with kernel matrix $K_{ij} = k(\x_i, \x_j)$. For concreteness, we use the squared-exponential (RBF) kernel throughout this section:

$$
k_\mathrm{SE}(\x, \x') = \sigma_f^2\,\exp\!\left(-\frac{\|\x - \x'\|^2}{2\ell^2}\right),
$$ (eq-nas_kse)

where $\ell$ is the kernel length scale (the distance in input space over which two function values remain correlated) and $\sigma_f^2$ is the signal variance (the typical squared amplitude of the function). Both are kernel hyperparameters that one would normally fit by marginal-likelihood maximization; here we treat them as fixed at sensible values, since BO calls the GP once per outer iteration and modest mis-tuning only changes the candidate ranking marginally.

**Posterior conditioning, in two formulas.** Given $n$ training configurations $\mathcal{D} = \{(\x_i, y_i)\}_{i=1}^n$ with possibly noisy outputs $y_i = f(\x_i) + \varepsilon_i$ and $\varepsilon_i \sim \mathcal{N}(0, \sigma_y^2)$, write $\bm{y} = (y_1, \ldots, y_n)^\top$. Three matrices and one vector enter the posterior: the kernel matrix $K \in \mathbb{R}^{n \times n}$ with $K_{ij} = k(\x_i, \x_j)$; its noise-augmented version $K_y = K + \sigma_y^2 I$; the prior-mean vector $\bm{\mu}_X$ whose $i$-th entry is $\mu(\x_i)$; and the cross-covariance $\bm{k}_* \in \mathbb{R}^n$ whose $i$-th entry is $k(\x_*, \x_i)$, where $\x_*$ is any query point. Conditioning the joint Gaussian $(\bm{y}, f(\x_*))$ on the observations yields a Gaussian posterior over the latent value $f(\x_*)$, with closed-form mean and variance:

$$
\bar{f}_* = \mu(\x_*) + \bm{k}_*^\top K_y^{-1}\bigl(\bm{y} - \bm{\mu}_X\bigr)
$$ (eq-nas_gp_mean)

$$
\sigma_{f,*}^2 = k(\x_*, \x_*) - \bm{k}_*^\top K_y^{-1} \bm{k}_*.
$$ (eq-nas_gp_var)

These two equations carry all of the GP intuition we will use below. In the noiseless limit $\sigma_y^2 \to 0$, the posterior mean $\bar{f}_*$ passes exactly through every observation, so the GP is an interpolator, and the posterior variance $\sigma_{f,*}^2$ collapses to zero at observation points and grows in the gaps between them. The pair $\bar{f}_* \pm 2\sigma_{f,*}$ therefore traces an honest error bar: tight where the data are dense and loose where they are sparse. {ref}`ch-gp` derives {eq}`eq-nas_gp_mean`–{eq}`eq-nas_gp_var` step by step, works a hand-traceable 1D example, and explains the marginal-likelihood Occam's razor that selects $(\ell, \sigma_f, \sigma_y)$ from data; for the BO logic that follows, {eq}`eq-nas_gp_mean`–{eq}`eq-nas_gp_var` are sufficient.

(sec-nas_ei)=
### Expected Improvement
We now plug the GP posterior into a decision rule for choosing the next configuration to evaluate. Specialize the GP input to the hyperparameter vector, $\x = \bm{h}$, and treat each observation $y_i = \ell_i = \ell(\bm{h}_i)$ as the validation loss at the $i$-th evaluated configuration. At any untried $\bm{h}$ the GP returns a posterior mean $\bar{f}(\bm{h})$ and posterior standard deviation $\sigma_f(\bm{h})$, where we drop the $*$ subscript when $\bm{h}$ is understood as the query point. We score each candidate $\bm{h}$ by how much, in expectation under the GP, its loss would beat the best loss seen so far, $\ell^\star = \min_i \ell_i$:

$$
\mathrm{EI}(\bm{h}) = \E{\max\bigl(\ell^\star - \ell(\bm{h}),\; 0\bigr)}.
$$

Reading this as "the GP believes the loss at $\bm{h}$ is roughly $\bar{f}(\bm{h})$ plus a Gaussian fluctuation of size $\sigma_f(\bm{h})$, and we credit only the improvement, not the deterioration," gives the intuition. Under the GP posterior the expectation evaluates in closed form:

$$
\mathrm{EI}(\bm{h}) = \bigl(\ell^\star - \bar{f}(\bm{h})\bigr)\,\Phi(Z) + \sigma_f(\bm{h})\,\phi(Z),
\qquad Z = \frac{\ell^\star - \bar{f}(\bm{h})}{\sigma_f(\bm{h})},
$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF, respectively {cite:p}`garnett2023bayesian`. The first term rewards *exploitation* (the predicted mean $\bar{f}$ lies below the current best $\ell^\star$); the second rewards *exploration* ($\sigma_f$ is large where no data constrain the loss). EI peaks where the two conspire, which is exactly the place a sensible researcher would probe by hand. {numref}`fig-bayesopt` illustrates one iteration of this rule in one dimension, and the algorithmic box below collects it into a four-step recipe. Bayesian optimization is particularly effective when the number of hyperparameters is moderate ($d \leq 20$) and each evaluation is expensive, which describes many economic applications well.

```{figure} figures/fig-bayesopt.svg
:name: fig-bayesopt

Bayesian optimization in one dimension; the same setup is reproduced in the companion notebook. *Top:* after five evaluations (black dots), the GP posterior mean $\bar f(h)$ (solid blue) interpolates the observations and the $\bar f(h) \pm 2\sigma_f(h)$ credible band (shaded) pinches to zero at each observation and widens in the gaps; this is the picture predicted by {eq}`eq-nas_gp_mean`–{eq}`eq-nas_gp_var`. The dashed red curve is the (in practice unknown) true loss; the horizontal gray line marks $\ell^\star$, the best observation so far (near $h\approx 2.3$). *Bottom:* Expected Improvement $\mathrm{EI}(h)$ is essentially zero at the existing data, where there is nothing to learn, rises in the unexplored gaps, and peaks at $h \approx 3.75$, which combines a predicted mean already below $\ell^\star$ with substantial residual uncertainty. The maximizer (red arrow) is selected as the next configuration to evaluate; EI thus balances exploitation against exploration automatically, and here it steers the search at the neighborhood of the hidden true minimum.
```

```{prf:definition} Bayesian Optimization with Expected Improvement

- **Input:** initial design $\mathcal{D}_{n_0} = \{(\bm{h}_i, \ell_i)\}_{i=1}^{n_0}$, total budget $N$
- **for** $n = n_0, n_0+1, \ldots, N-1$ **do**
  - Fit the GP posterior $(\bar{f}, \sigma_f)$ to $\mathcal{D}_n$ using {eq}`eq-nas_gp_mean`–{eq}`eq-nas_gp_var`
  - Evaluate $\mathrm{EI}(\bm{h})$ on a fine candidate grid (or via a local optimizer) over the search space
  - Select $\bm{h}_{n+1} = \arg\max_{\bm{h}} \mathrm{EI}(\bm{h})$
  - Evaluate the validation loss $\ell_{n+1} = \ell(\bm{h}_{n+1})$ and append to $\mathcal{D}_{n+1} = \mathcal{D}_n \cup \{(\bm{h}_{n+1}, \ell_{n+1})\}$
- **end**
- **Output:** best observed configuration $\arg\min_i \ell_i$
```


(sec-hyperband)=
## Hyperband and Successive Halving
{cite:t}`li2018hyperband` proposed an entirely different approach based on *adaptive resource allocation*, building on the Successive Halving Algorithm popularized by {cite:t}`jamieson2016nonstochastic`. The key observation is that poor configurations can often be identified early in training, without running them to completion.

The Successive Halving Algorithm (SHA) turns this observation into a concrete schedule. Start with $n_0$ configurations, give each a small initial budget of $r_0$ epochs, train, then keep only the top $1/\eta$ fraction of survivors and multiply the per-candidate budget by $\eta$ for the next round. Two facts about this schedule do most of the explanatory work below. First, the per-round compute is approximately constant: round $s$ runs $n_s$ survivors for $r_s$ epochs each, and the rule "$\eta\times$ fewer candidates at $\eta\times$ the budget" keeps the product $n_s r_s$ fixed. Second, the cumulative cost across the cascade therefore scales with the *number of rounds* rather than with the worst-case "train every candidate to the final budget" benchmark. Both points are made concrete in the worked example that follows the pseudocode.

```{prf:definition} Successive Halving Algorithm

- **Input:** initial candidates $n_0$, initial per-candidate budget $r_0$ (epochs), reduction factor $\eta = 3$
- Draw a set $S_0$ of $n_0$ configurations from the search space
- **for** round $s = 0, 1, \ldots, \lceil\log_\eta n_0\rceil - 1$ **do**
  - Train each of the $n_s$ survivors in $S_s$ for $r_s$ epochs and record their validation losses
  - Keep the top $1/\eta$ fraction of $S_s$ to form $S_{s+1}$
  - Set $n_{s+1} = \lceil n_s / \eta \rceil$ and $r_{s+1} = \eta\, r_s$
- **end**
- **Output:** best surviving configuration
```


To see the per-round invariant on a concrete schedule, take $n_0 = 81$, $r_0 = r$, and $\eta = 3$. Round 0 trains all 81 candidates for $r$ epochs and keeps the top 27; round 1 trains those 27 for $3r$ epochs and keeps the top 9; round 2 trains the 9 for $9r$ and keeps 3; round 3 trains those 3 for $27r$ and selects the winner. At each round the round-level compute is $n_s\, r_s = 81\,r$ (the $1/3$ shrink in survivors is exactly offset by the $3\times$ growth in budget), so the four rounds together cost $4 \cdot 81\,r = 324\,r$, equivalent to four full $R = 81r$-epoch training runs rather than the $81$ that naive parallel evaluation would require. {numref}`fig-hyperband` visualizes this resource-allocation cascade.

```{figure} figures/fig-hyperband.svg
:name: fig-hyperband

Successive Halving with 81 initial candidates and reduction factor $\eta = 3$. Each round trains the surviving configurations for $\eta$ times the previous budget, then discards the bottom $(1-1/\eta)$ fraction. Total compute per bracket is only $\mathcal{O}(B)$ rather than $\mathcal{O}(nB)$ for training every candidate to completion. Hyperband runs several such brackets in parallel with different $(n,r)$ trade-offs to hedge against unknown early-vs-late performance correlations {cite:p}`li2018hyperband`.
```

Hyperband extends SHA by running not one but several SHA cascades, each starting from a different trade-off between the number of candidates and the per-candidate initial budget; the intuition is hedging. A bracket that starts with many cheap candidates (large $s$) is well suited to problems where the eventual winner reveals itself early, while a bracket that starts with fewer well-trained candidates (small $s$) is better when ranks shuffle late in training, and Hyperband simply runs both kinds. The cascade shown in {numref}`fig-hyperband`, $(81,1) \to (27,3) \to (9,9) \to (3,27) \to (1,81)$, is the most exploratory of the brackets ($s = s_{\max} = 4$); {numref}`tab-hyperband_brackets` unrolls the full canonical schedule for maximum resource $R = 81$ and reduction factor $\eta = 3$.

````{table}
:name: tab-hyperband_brackets

Hyperband bracket schedule for $R = 81$, $\eta = 3$, following Table 1 of {cite:t}`li2018hyperband`. Each bracket $s$ is a standalone SHA cascade with $n_s$ initial candidates and per-candidate initial budget $r_s$ epochs; the survivors are then halved according to the rule of Algorithm {numref}`fig-hyperband`. Large $s$ probes many cheap candidates and halves them aggressively (exploration); small $s$ trains a handful of candidates to the full $R$-epoch budget from the start (exploitation). Per-bracket totals $B_s$ are reported in units of $R$ and are not monotone in $s$ because $n_s$ is integer-rounded. Hyperband runs all five brackets and returns the best surviving configuration across them.

| **Bracket $s$** | **$(n_s, r_s)$** | **Unrolled schedule (candidates @ epochs)** | **Total cost $B_s$** |
|:---:|:---:|---|:---:|
| $4$ | $(81, 1)$ | $81$@$1 \to 27$@$3 \to 9$@$9 \to 3$@$27 \to 1$@$81$ | $5R$ |
| $3$ | $(27, 3)$ | $27$@$3 \to 9$@$9 \to 3$@$27 \to 1$@$81$ | $4R$ |
| $2$ | $(9, 9)$ | $9$@$9 \to 3$@$27 \to 1$@$81$ | $3R$ |
| $1$ | $(6, 27)$ | $6$@$27 \to 2$@$81$ | $4R$ |
| $0$ | $(5, 81)$ | $5$@$81$ | $5R$ |
````

The companion notebook [`03_NAS_RandomSearch_Hyperband.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb) implements the SHA inner loop only; the full Hyperband schedule is a straightforward outer loop that iterates this inner loop across the five $(n_s, r_s)$ starting points in {numref}`tab-hyperband_brackets`.

## Method Comparison

{numref}`tab-nas_methods` contrasts the four hyperparameter-search strategies covered above on three dimensions that matter in practice: the cost of $N$ objective evaluations, the degree to which the evaluations can be parallelized, and the sample efficiency (how much of the budget actually improves the best-so-far value).

````{table}
:name: tab-nas_methods

Comparison of hyperparameter-search methods. Grid search scales exponentially in the number of hyperparameters $d$; random search and Hyperband scale linearly in the chosen evaluation/resource budget and parallelize well; Bayesian optimization has the highest per-evaluation information gain but adds surrogate-fitting overhead and is partly sequential.

| **Method** | **Cost** | **Parallelizable** | **Sample efficiency** | **Best for** |
|---|:---:|:---:|:---:|:---:|
| Grid search | $\mathcal{O}(k^d)$ evals | fully | low | $d \leq 3$ |
| Random search | $N$ evals | fully | moderate | general use |
| Bayesian opt. | $N$ evals + GP fit | limited | high | expensive evals |
| Hyperband | $N$ resource units | within brackets | moderate | cheap evals |
````

For the DEQN and PINN applications in this course, random search or Bayesian optimization are typically the most practical choices. Hyperband is attractive when training is relatively cheap and many configurations need to be screened quickly.

(sec-nas-implementation)=
## Implementing the Search in Practice
To keep the algorithms transparent, the companion notebook [`03_NAS_RandomSearch_Hyperband.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb) implements both Random Search (§ {ref}`sec-nas_random_search`) and the Successive Halving Algorithm (§ {ref}`sec-hyperband`) directly in plain Python, with no hyperparameter-search library involved. The search space is encoded as an ordinary dict (number of hidden layers $\in \{1,\ldots,5\}$, units per layer $\in \{32, 64, \ldots, 256\}$, activation function $\in \{\texttt{relu}, \texttt{tanh}, \texttt{swish}\}$, and learning rate log-uniform in $[10^{-4}, 10^{-2}]$), and a single `sample_config(rng)` function draws candidates from it. Random Search is then a $30$-iteration loop that builds, trains, and scores each candidate; Successive Halving is the same loop wrapped in a halving schedule ($n_0 = 27$ candidates at $r_0 = 8$ epochs $\to$ $9$ at $24$ $\to$ $3$ at $72$ $\to$ winner, with $\eta = 3$). Both implementations fit on a single slide and reproduce the qualitative finding of {cite:t}`li2018hyperband` that Successive Halving reaches comparable accuracy to Random Search at substantially lower compute: in the notebook run, the same MAE is recovered with $\sim 2.3\times$ less compute (648 SHA config-epochs vs. 1500 for 30 Random Search trials at 50 epochs each) at a comparable number of architectures (27 vs. 30). The precise multipliers are notebook-specific; the magnitudes reported in Li et al. vary by benchmark.

**Production tooling (footnote).** Real projects rarely hand-roll the search loop. Several established libraries wrap (and parallelize) the same algorithms behind uniform APIs: `KerasTuner`[^1] (Random, Bayesian, Hyperband; tight Keras integration), `Optuna`[^2] (TPE, CMA-ES, Hyperband, NSGA-II; framework-agnostic), `Ray Tune`[^3] (all of the above plus ASHA and population-based training, distributed by design), `Hyperopt`[^4] (the original TPE reference), `Ax` / `BoTorch`[^5] (PyTorch-native multi-objective Bayesian optimization), `NNI`[^6] (Microsoft; full graph-NAS support), and `AutoKeras`[^7] (full AutoML pipeline). We deliberately teach the algorithms rather than the wrappers because library APIs change every few years; the underlying search procedures (Random, SHA / Hyperband, GP+EI, TPE) do not. The notebook additionally compares the best NAS-found architecture to a hand-tuned baseline, which makes the pedagogical value of automated search concrete.

## Multi-Component Losses: The Scale Problem

In many applications, including DEQNs and PINNs, the loss function is a weighted sum of several components:

$$
\ell = \sum_{k=1}^{K} w_k \, \ell_k,
$$

where $\ell_k \ge 0$ is the $k$-th *loss component* (one individual residual, e.g. a per-country Euler equation, a market-clearing identity, a complementarity-condition residual in an OLG model, or a boundary or initial-condition penalty in a PINN), $w_k \ge 0$ is its scalar weight, and $K$ is the total number of components. In the DEQN setting of this script, each $\ell_k$ is typically the mean-squared residual of a particular equilibrium equation evaluated on the training mini-batch, so $\ell_k = 0$ at a solution and $\ell$ is jointly minimized over the network parameters.

From a multi-objective-optimization standpoint, the vector $(\ell_1, \dots, \ell_K)$ is the object of interest: the equilibrium is defined by all $K$ residuals being zero, and any nonzero weight vector $\bm{w}$ picks a particular scalarization of the same underlying Pareto problem. When the components are on comparable scales, uniform weights work; when they are not, the scalarized problem is dominated by a single component and the optimizer effectively ignores the others. Adaptive weighting methods (discussed below) can be seen as online strategies for traversing the Pareto frontier rather than committing to a single scalarization up front. If the individual components $\ell_k$ differ in magnitude by several orders of magnitude, training can become unstable or converge slowly. Consider a concrete example from the IRBC model with $N=10$ countries: at initialization, the Euler equation residual for country 1 might be $\ell_1 \approx 50$, while for country 10 it might be $\ell_{10} \approx 0.05$, a ratio of $10^3$. With uniform weights, the gradient is dominated by $\nabla \ell_1$, and the optimizer essentially ignores $\ell_{10}$ until $\ell_1$ is nearly converged. This sequential convergence pattern can be $5$–$10\times$ slower than balanced convergence.

The fundamental difficulty is that the gradient of the total loss $\nabla \ell = \sum_k w_k \nabla \ell_k$ is dominated by the components with the largest $|w_k \nabla \ell_k|$. Even if all components are equally important for the economic solution, the optimizer cannot "see" the small components through the noise of the large ones.

### Inverse-Loss Weighting

A simple first approach is to set $w_k = 1/\bar{\ell}_k$, where $\bar{\ell}_k$ is an exponential moving average of $\ell_k$. This normalizes each component to have approximately unit magnitude. While straightforward, this method can be unstable when loss components change rapidly.

### SoftAdapt

{cite:t}`heydari2019softadapt` proposed a more principled approach based on the *rates of change* of the loss components. Define $\Delta_k^{(t)} = \ell_k^{(t)} - \ell_k^{(t-1)}$.[^8] The SoftAdapt weights are:

$$
w_k^{(t)} = \frac{\exp(\Delta_k^{(t)}/\tau)}{\sum_{j=1}^K \exp(\Delta_j^{(t)}/\tau)},
$$

where $\tau > 0$ is a temperature parameter. Components that are decreasing slowly (or increasing) receive higher weight, directing the optimizer's attention to the lagging components. In practice, SoftAdapt uses smoothed rates (averaged over a window of recent iterations) for stability. SoftAdapt is discussed here for context; the companion notebook [`04_Loss_Normalization.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_04_Loss_Normalization.ipynb) implements equal-, inverse-loss-, and ReLoBRaLo-weighting, and {prf:ref}`ex-ch4-6` asks you to add a GradNorm balancer to the same testbed.

(sec-relobralo)=
### ReLoBRaLo: Adaptive Loss Balancing
The *Relative Loss Balancing with Random Lookback* (ReLoBRaLo) algorithm of {cite:t}`bischof2025relobralo` motivates the deterministic classroom implementation used here. In the notebooks, we use a convex blend of the same ingredients, which is easier to follow while preserving the balancing logic.

**High-level intuition.** ReLoBRaLo combines two complementary signals into a single weight per loss component. The *step-wise* signal asks "which component lagged the most since the last iteration?" and rewards it with more weight; this is fast and reactive but noisy. The *baseline* signal asks "which component lagged the most since the start of training?" and is slow but globally aware. The two are then averaged with a one-step smoother to dampen oscillations. Concretely the algorithm stacks three pieces (Components 1–3 below); only the temperature $T$ usually needs tuning, while the smoothing parameters $\alpha,\rho$ work at their textbook defaults.

**Component 1: Relative balancing.** At iteration $t$, compute relative losses with respect to the previous iteration:

$$
\begin{aligned}
r_{k,\mathrm{step}}^{(t)}
&= \frac{\ell_k^{(t)}}{T\,\ell_k^{(t-1)}+\varepsilon_{\mathrm{num}}},\\
\hat{w}_{k,\mathrm{step}}^{(t)}
&= K \cdot
\frac{\exp\!\bigl(r_{k,\mathrm{step}}^{(t)}\bigr)}
{\sum_{j=1}^{K}\exp\!\bigl(r_{j,\mathrm{step}}^{(t)}\bigr)}.
\end{aligned}
$$

This upweights components whose relative loss increased (lagging behind) and downweights those that improved. The small $\varepsilon_{\mathrm{num}}$ prevents division by zero; in code the softmax is evaluated after subtracting the largest ratio for numerical stability.

**Component 2: Baseline comparison.** To maintain a global perspective, compare the current losses to their initial values at $t=0$:

$$
\begin{aligned}
r_{k,\mathrm{base}}^{(t)}
&= \frac{\ell_k^{(t)}}{T\,\ell_k^{(0)}+\varepsilon_{\mathrm{num}}},\\
\hat{w}_{k,\mathrm{base}}^{(t)}
&= K \cdot
\frac{\exp\!\bigl(r_{k,\mathrm{base}}^{(t)}\bigr)}
{\sum_{j=1}^{K}\exp\!\bigl(r_{j,\mathrm{base}}^{(t)}\bigr)}.
\end{aligned}
$$

This baseline comparison provides robustness to non-monotone loss trajectories and prevents the algorithm from losing sight of overall training progress.

**Component 3: Smoothed combination.** The final weight blends historical weights, baseline weights, and step-wise weights:

$$
w_k^{(t)} =
\alpha\bigl[\rho\, w_k^{(t-1)} + (1-\rho)\, \hat{w}_{k,\mathrm{base}}^{(t)}\bigr]
+ (1-\alpha)\, \hat{w}_{k,\mathrm{step}}^{(t)},
$$ (eq-relobralo_full)

where $\alpha \in [0,1]$ is a smoothing parameter controlling how much to trust historical weights versus the current step-wise signal, and $\rho \in [0,1]$ is a baseline-mix coefficient controlling the relative importance of the previous weights versus the initial-loss comparison. Equivalently, $w_k^{(t)}$ is a convex combination of $\{w_k^{(t-1)}, \hat{w}_{k,\mathrm{base}}^{(t)}, \hat{w}_{k,\mathrm{step}}^{(t)}\}$ with mixture weights $(\alpha\rho,\, \alpha(1-\rho),\, 1-\alpha)$, which sum to 1. (In the original ReLoBRaLo formulation, $\rho$ governs a stochastic Bernoulli lookback mechanism; here and in the notebooks we use a deterministic convex blend, which is simpler and easier to reproduce.) Typical values are collected in {numref}`tab-relobralo_hp`.

````{table}
:name: tab-relobralo_hp

ReLoBRaLo hyperparameters used in the companion notebook. Default ranges follow {cite:t}`bischof2025relobralo`. $T$ is the only one that usually needs tuning; $\alpha$ and $\rho$ at their defaults give slow, stable adaptation that works across a wide range of multi-component problems.

| **Hyperparameter** | **Role** | **Typical value** |
|---|---|---|
| $T$ (temperature) | Controls weight concentration (softmax sharpness) | $0.5$–$2.0$ |
| $\alpha$ (smoothing) | History vs. new information | $0.99$–$0.999$ |
| $\rho$ (mixing coefficient) | Initial-loss baseline vs. historical weight | $0.99$–$0.999$ |
````

### GradNorm

An alternative approach proposed by {cite:t}`chen2018gradnorm` directly normalizes the gradient magnitudes rather than the loss values. GradNorm adjusts the weights so that $\|w_k \nabla \ell_k\|$ is approximately equal across all components, using the ratio of each component's training rate to the average training rate as a signal. While more computationally expensive than ReLoBRaLo (it requires computing per-component gradient norms), GradNorm can be effective when gradient magnitudes are a better proxy for training difficulty than loss magnitudes.

```{figure} figures/fig-multi_component_loss.svg
:name: fig-multi_component_loss

*Stylized* sketch of the multi-component loss-scale problem, drawn to mimic what one typically sees early in a two-country IRBC training run; this is *not* measured data. The three curves are hand-picked exponentials $a_k\,e^{-t/\tau_k}$ (with $a_1{=}50,\tau_1{=}150$; $a_2{=}0.5,\tau_2{=}750$; $a_3{=}5,\tau_3{=}200$), chosen only to make the mechanism visible: at initialization the residuals differ by about two orders of magnitude, and under *uniform* weighting the optimizer drives the largest component $\ell_1$ (blue) down fastest because it dominates the summed gradient, while the smaller-scale but equally important country-2 Euler residual $\ell_2$ (red) decays roughly five times more slowly and is left all but flat next to the others. Adaptive loss balancing such as ReLoBRaLo re-weights the components so that all three decrease at comparable rates. For the *actual* recorded trajectories on this problem, see the companion notebook [`04_Loss_Normalization.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_04_Loss_Normalization.ipynb).
```

{numref}`fig-multi_component_loss` illustrates the typical behavior: without adaptive reweighting, the optimizer focuses almost exclusively on $\ell_1$ (the largest component), allowing $\ell_2$ to stagnate; with adaptive loss balancing (e.g., ReLoBRaLo, GradNorm), all components converge at comparable rates. As a concrete reference, an unweighted run of the two-country IRBC training loop in the companion notebook typically prints something like the trace below (numbers indicative, seed-dependent):

```{code-block} text
:name: lst-irbc_residual_trace
:caption: Indicative residual log from an unweighted two-country IRBC run; the largest component falls quickly, the smaller component stalls.

epoch    0:  ell_1=49.700  ell_2=0.510  ell_arc=4.820
epoch  200:  ell_1=0.0123  ell_2=0.494  ell_arc=0.041
epoch  500:  ell_1=8.2e-4  ell_2=0.470  ell_arc=3.5e-3
```
The pathology is immediate: $\ell_1$ drops four orders of magnitude while $\ell_2$ barely moves. Replacing the equal weights with ReLoBRaLo ({ref}`sec-relobralo`) typically produces a trace in which all three components decay together; see the companion notebook for the actual ReLoBRaLo trace on the same seed. Reported convergence-speed improvements vary across schemes and benchmarks; multi-physics PINN benchmarks have shown substantial gains with ReLoBRaLo {cite:p}`bischof2025relobralo`, while gains on DEQN-style Euler-equation losses tend to be smaller and problem-specific (the multi-component scale gap there is usually one to two orders of magnitude rather than the four to six common in PINN systems). A complementary line uses Neural-Tangent-Kernel diagnostics to choose the weights {cite:p}`wang2022when`, and an older multi-task baseline weights losses by their predictive uncertainty {cite:p}`kendall2018multi`.

### Summary of Balancing Methods

{numref}`tab-balancing_methods` compares the four balancing strategies on the two dimensions that reliably matter in practice: runtime overhead per step and the number of hyperparameters the user must set.

````{table}
:name: tab-balancing_methods

Summary of adaptive loss-balancing methods. Overhead is a per-step wall-clock cost (additional softmaxes for SoftAdapt/ReLoBRaLo; per-component gradient norms for GradNorm). Quantitative speedups depend strongly on the problem; see the companion notebook [`04_Loss_Normalization.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_04_Loss_Normalization.ipynb) for problem-specific measurements.

| **Method** | **Overhead** | **Hyperparameters** |
|---|:---:|:---:|
| Uniform weights | none | 0 |
| Inverse-loss | negligible | 1 (smoothing) |
| Uncertainty weighting {cite:p}`kendall2018multi` | negligible | $K$ (one log-variance per loss) |
| SoftAdapt {cite:p}`heydari2019softadapt` | negligible | 2 ($\tau$, window) |
| ReLoBRaLo {cite:p}`bischof2025relobralo` | negligible | 3 ($T$, $\alpha$, $\rho$) |
| GradNorm {cite:p}`chen2018gradnorm` | moderate | 1 ($\alpha$) |
| NTK-based {cite:p}`wang2022when` | moderate–high | 0 (data-driven) |
````

Quantitative speedup claims depend on the specific problem (PDE vs. Euler residual, number of components, imbalance ratio), the baseline (uniform vs. manually tuned), and the success criterion. The companion notebook [`04_Loss_Normalization.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_04_Loss_Normalization.ipynb) runs the four methods on a shared multi-scale regression task so that the reader can generate problem-specific numbers rather than rely on headline speedup factors from unrelated benchmarks.

```{prf:remark} Practical guidance

 When implementing ReLoBRaLo:

- Set $T \in [0.5, 2.0]$; higher values yield more uniform weighting. In the limit, $T \to 0$ approximates a winner-take-all scheme that concentrates all weight on the single most-lagging component, while $T \to \infty$ recovers uniform weighting regardless of loss dynamics.

- Start with $\alpha = \rho = 0.999$ and reduce if weights change too slowly.

- ReLoBRaLo adds negligible computational overhead (one softmax per iteration) but can dramatically improve convergence for multi-component losses; GradNorm and SoftAdapt make analogous trade-offs.

- For PINN applications ({ref}`ch-pinn`), adaptive loss balancing in general (ReLoBRaLo, GradNorm, NTK-based schemes) is particularly effective at balancing PDE residual terms against boundary condition penalties.

- In DSGE applications, multi-component losses arise naturally: a model with $N$ countries has $N$ Euler equations, an aggregate resource constraint, and $N$ complementarity conditions, often differing by several orders of magnitude. Without loss balancing, the optimizer focuses on the largest component and ignores smaller but economically important residuals (see {ref}`ch-irbc`).
```


```{prf:remark} Chapter Summary

- Architecture and hyperparameter choice are first-order: random search dominates grid search whenever only a few hyperparameters matter, and Bayesian optimization dominates random search whenever each evaluation is expensive.

- Hyperband-style multi-fidelity scheduling is the modern compromise: cheap evaluations of many candidates, with budget concentrated on the survivors.

- Multi-component DEQN/PINN losses suffer from scale imbalance; ReLoBRaLo and related adaptive schemes restore balance by reweighting components on the fly.

- All three families (NAS, Bayesian optimization, adaptive loss balancing) are implementation details, but they make the difference between a DEQN that converges on a coffee break and one that diverges overnight.
```


## Further Reading {.unnumbered}
- {cite:t}`bergstra2012random`, the original case for random over grid search.

- {cite:t}`snoek2012practical`, foundational reference for Bayesian optimization in ML.

- {cite:t}`li2018hyperband`, Hyperband and successive halving.

- {cite:t}`bischof2025relobralo`, ReLoBRaLo loss-balancing scheme used throughout the PINN chapter.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch4-1

**[Core\] Random vs. grid.** Reproduce the random-vs-grid figure of {cite:t}`bergstra2012random`. With $9$ evaluations and only one "important" axis out of two, suppose the near-optimal region occupies a fraction $p$ of that axis and its location relative to the grid is unknown. What is the hit probability for a $3\times 3$ grid, and what is the hit probability for random search?
```

```{exercise}
:label: ex-ch4-2

**[Computational\] Bayesian optimization toy problem.** Implement a Gaussian-process-based BO loop on the 1D function $f(x) = -\sin(3x) - x^2 + 0.7x$ over $x \in [-1,2]$. How many BO evaluations are typically needed before it matches the best value found by a grid with step size $0.01$?
```

```{exercise}
:label: ex-ch4-3

**[Core\] Hyperband budget allocation.** For a Hyperband run with maximum resource $R=81$ and $\eta=3$, list the brackets $(n_i, r_i)$ used and the total resource consumed. Compare with a naive "train each candidate to $R$" strategy at fixed candidate count $n_0 = 27$.
```

```{exercise}
:label: ex-ch4-4

**[Core\] Loss balancing.** In a multi-component PINN loss with three terms of magnitude $10^0$, $10^{-2}$, $10^{-4}$, write down what fixed weights $\lambda_i$ would be needed to equalize their gradient contributions. Why does this become impractical when the gradients are correlated?
```

```{exercise}
:label: ex-ch4-5

**[Core\] Pareto frontier geometry.** Consider the toy two-component loss $\mathcal{L}(\theta;\lambda) = \lambda\,(\theta - a)^2 + (1-\lambda)(\theta - b)^2$ with $\theta \in \mathbb{R}$, $a < b$, and $\lambda \in [0,1]$. (i) Solve for the minimizer $\theta^\star(\lambda)$ in closed form. (ii) Compute the per-component residuals $\ell_1^\star(\lambda) = (\theta^\star - a)^2$ and $\ell_2^\star(\lambda) = (\theta^\star - b)^2$. (iii) Eliminate $\lambda$ to express the Pareto frontier in the $(\ell_1, \ell_2)$ plane and show it is the curve $\sqrt{\ell_1} + \sqrt{\ell_2} = b - a$ for $\ell_1, \ell_2 \ge 0$, hence convex. (iv) Sketch the frontier and identify which point on it corresponds to the equal-weight choice $\lambda = 1/2$. (v) In higher-dimensional parameter spaces, explain why nonzero gradient inner products $\langle\nabla \ell_1, \nabla \ell_2\rangle$ make fixed scalar weights fragile. Contrast ReLoBRaLo's relative-loss progress rule with GradNorm's direct gradient-norm balancing.
```

```{exercise}
:label: ex-ch4-6

**[Computational\] ReLoBRaLo vs. GradNorm.** In notebook `04_Loss_Normalization`, swap the ReLoBRaLo balancer for a GradNorm balancer ({cite:t}`chen2018gradnorm`; see the chapter for the per-component gradient-norm equalization rule). Run both on the same multi-scale regression target with components of magnitude $10^0, 10^{-2}, 10^{-4}$. Report (i) wall-clock training time per epoch, (ii) final residual on each component, (iii) the gradient-balance ratio $\|\nabla\ell_k\|/\sum_j \|\nabla\ell_j\|$ at convergence. Comment on the cost–benefit trade-off: under what circumstances is the extra per-step cost of GradNorm worth it?
```

```{exercise}
:label: ex-ch4-7

**[Core\] HPO vs. full NAS decision.** You have access to either (a) a single consumer GPU (RTX 3060, $\sim 12$ GB) or (b) one A100 ($80$ GB), for one week of compute. Your search problem is either (i) a fixed-topology MLP with unknown depth $\in \{1,\ldots,5\}$, width $\in \{32,\ldots,512\}$, activation $\in \{\mathrm{ReLU}, \mathrm{Swish}, \tanh\}$, learning rate (log-uniform), or (ii) a flexible network that can be MLP / RNN / shallow Transformer with unknown layer connectivity (graph-level NAS). For each of the four (hardware $\times$ problem) cells, recommend in three to five sentences whether to use Random Search with Successive Halving, Bayesian Optimization, or full graph-level NAS. Justify by referencing the per-method overhead and search-space size.
```

[^1]: <https://keras.io/keras_tuner/>

[^2]: <https://optuna.org/>

[^3]: <https://docs.ray.io/en/latest/tune/>

[^4]: <http://hyperopt.github.io/hyperopt/>

[^5]: <https://ax.dev/>

[^6]: <https://nni.readthedocs.io/>

[^7]: <https://autokeras.com/>

[^8]: The raw $\Delta_k^{(t)}$ is dimensional, so a component at scale $10^3$ produces fluctuations that swamp one at scale $10^{-3}$; in practice one rescales before the softmax, e.g. $\tilde\Delta_k^{(t)} = \Delta_k^{(t)} / (\ell_k^{(t)} + \varepsilon)$, so the rule reacts to *relative* progress, the same idea that underlies ReLoBRaLo's loss *ratios* below.
