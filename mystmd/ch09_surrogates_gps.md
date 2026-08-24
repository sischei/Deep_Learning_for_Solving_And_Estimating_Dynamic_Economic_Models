---
title: "Gaussian Processes"
label: ch-gp
---

{ref}`sec-nas_gp_primer` introduced Gaussian processes at the picture level, with just enough machinery to follow the Bayesian-optimization argument of {ref}`ch-nas`: the definition $f(\x) \sim \mathcal{GP}(\mu(\x), k(\x, \x'))$, the squared-exponential kernel {eq}`eq-nas_kse`, and the closed-form posterior mean and variance {eq}`eq-nas_gp_mean`–{eq}`eq-nas_gp_var`. Those formulas took the kernel and its hyperparameters $(\ell, \sigma_f, \sigma_y)$ as given. The present chapter removes both shortcuts and develops the full machinery: it samples whole functions from the GP *prior* ({ref}`sec-gp_regression`), learns $(\ell, \sigma_f, \sigma_y)$ from data via the marginal-likelihood Occam's razor ({ref}`sec-gp_kernels`), expands the kernel zoo beyond the squared-exponential to Matérn and composite kernels ({ref}`sec-matern`), uses the posterior variance to choose where to evaluate next via Bayesian Active Learning ({ref}`sec-bal`), scales the whole machinery to high-dimensional inputs through active subspaces ({ref}`sec-active_subspaces`), and embeds GPs inside value-function iteration for high-dimensional dynamic programming ({ref}`sec-gp_dp`). Two applications-driven chapters then put the machinery to work: structural estimation via surrogate-driven SMM ({ref}`ch-estimation`) and climate-policy uncertainty quantification ({ref}`ch-climate`). Methodological foundations are the GP textbook of {cite:t}`Rasmussen:2005:GPM:1162254` and the Bayesian-active-learning ideas dating back to {cite:t}`mackay1992information` and {cite:t}`krause2008near`. Embedding GPs inside dynamic programming was pioneered by {cite:t}`deisenroth2009gaussian` (GPDP) and {cite:t}`engel2005reinforcement` (GP temporal-difference learning); within economics, applications include high-dimensional growth models {cite:p}`SCHEIDEGGER201968`, dynamic incentive problems {cite:p}`rennerscheidegger_2018`, and deep uncertainty quantification for integrated assessment models {cite:p}`friedlDeep2023`. Sparse approximations {cite:p}`titsias2009variational,hensman2013gaussian` extend GPs beyond their cubic scaling limit; deep kernel learning combines neural-network feature extraction with GP inference and closes the chapter.

(sec-gp_regression)=
## Gaussian Process Regression
**Sampling from the GP prior.** Recall from {ref}`sec-nas_gp_primer` that for any finite set of test inputs $\x_1, \ldots, \x_n$, the function values $\bm{f} = (f(\x_1), \ldots, f(\x_n))^\top$ are jointly Gaussian with prior mean $\bm{\mu}$ (the entries $\mu(\x_i)$) and covariance matrix $K_{ij} = k(\x_i, \x_j)$. To draw a function from the GP prior at those test inputs, evaluate $K$, compute its Cholesky decomposition $K = L L^\top$, draw $\bm{u} \sim \mathcal{N}(\bm{0}, I)$, and set $\bm{f} = \bm{\mu} + L \bm{u}$. The resulting paths have the smoothness, amplitude, and length scale dictated entirely by the kernel.

**Kernel choice.** The squared-exponential kernel {eq}`eq-nas_kse` of {ref}`sec-nas_gp_primer` remains the right default for genuinely smooth targets, e.g. in the unconstrained interior of an ergodic set. In economic applications, however, the target function (value, policy, equilibrium price) often has *kinks* from occasionally binding constraints, and the Matérn-$3/2$ kernel introduced in {ref}`sec-matern` below is then a better default than RBF, since it does not oversmooth at non-differentiable features.

```{figure} figures/fig-rbf_length_scale.svg
:name: fig-rbf_length_scale

Squared-exponential kernel as a function of distance for three length scales. Small $\ell$ makes correlations decay quickly and produces rougher, more local fits; large $\ell$ couples distant points and imposes smoother functions.
```

{numref}`fig-rbf_length_scale` shows how the RBF length scale controls the distance over which observations remain informative. For any training dataset $\mathcal{D} = \{(\x_i, y_i)\}_{i=1}^n$, the GP posterior at a test point $\x_*$ has the closed-form mean $\bar{f}_*$ and variance $\sigma_{f,*}^2$ stated in {eq}`eq-nas_gp_mean`–{eq}`eq-nas_gp_var`, with $K_{ij} = k(\x_i,\x_j)$ the kernel matrix on training inputs, $K_y = K + \sigma_y^2 I$ its noise-augmented version, $\bm{\mu}_X$ the prior-mean vector at training inputs, and $\bm{k}_*$ the cross-covariance whose $i$-th entry is $k(\x_*, \x_i)$. For a noisy future observation $y_*$ the predictive variance is $\sigma_{y,*}^2 = \sigma_{f,*}^2 + \sigma_y^2$, and the common zero-mean formulas are recovered by centering outputs or setting $\mu \equiv 0$.

**A hand-traceable 1D example.** To make {eq}`eq-nas_gp_mean`–{eq}`eq-nas_gp_var` concrete, take $f(x) = \sin x$, observe $f$ noiselessly at $x_1 = 0$ and $x_2 = \pi$ (so $y_1 = y_2 = 0$), and query at $x_\star = \pi/2$. Use the kernel $k(x, x') = \exp\!\bigl(-(x - x')^2/2\bigr)$ and a tiny noise floor $\sigma_y^2 = 10^{-6}$ for numerical stability. The training kernel matrix is

```{math}
:enumerated: false

K + \sigma_y^2 I \;=\;
\begin{pmatrix} 1 & e^{-\pi^2/2} \\ e^{-\pi^2/2} & 1 \end{pmatrix}
\;\approx\;
\begin{pmatrix} 1.000 & 0.00719 \\ 0.00719 & 1.000 \end{pmatrix},
```

where the off-diagonal $e^{-\pi^2/2} \approx 0.00719$ is small because $0$ and $\pi$ are far apart relative to $\ell = 1$. The cross-covariance vector is

```{math}
:enumerated: false

\bm k_\star \;=\;
\begin{pmatrix} \exp(-(\pi/2)^2/2) \\ \exp(-(\pi/2)^2/2) \end{pmatrix}
\;\approx\;
\begin{pmatrix} 0.2910 \\ 0.2910 \end{pmatrix},
```

since $x_\star = \pi/2$ is equidistant from $0$ and $\pi$. Because $\bm y = (0,0)^\top$, the posterior mean {eq}`eq-nas_gp_mean` is exactly $\bar f_\star = 0$. For the variance,

```{math}
:enumerated: false

(K + \sigma_y^2 I)^{-1} \bm k_\star \;\approx\; \tfrac{0.2910}{1 + 0.00719}\,(1, 1)^\top \;\approx\; (0.2890, 0.2890)^\top,
```

so $\bm k_\star^\top (K + \sigma_y^2 I)^{-1} \bm k_\star \approx 2 \cdot 0.2910 \cdot 0.2890 \approx 0.1682$, giving $\sigma_\star^2 \approx 1 - 0.1682 \approx 0.832$ and a posterior standard deviation $\sigma_\star \approx 0.91$. The GP predicts zero at the midpoint, with substantial residual uncertainty, consistent with the fact that $\sin(\pi/2) = 1$ is not pinned down by the two boundary observations under this length scale.

{numref}`fig-gp_prior_posterior` illustrates the GP prior and posterior for a simple one-dimensional regression problem. Before observing data, the GP prior has constant mean and uniform uncertainty. After conditioning on five observations, the posterior mean interpolates the data and the uncertainty bands collapse near the observations while remaining wide in unexplored regions.

```{figure} figures/fig-gp_prior_posterior.svg
:name: fig-gp_prior_posterior

Gaussian-process prior and posterior on a 1D regression problem. *Left:* the prior has constant mean (here zero) and uniform uncertainty; the shaded bands show the $68\%$ and $95\%$ credible intervals, and the thin gray curves are three sample paths drawn from the prior. *Right:* after conditioning on five observations (black dots), the posterior mean (red curve) interpolates the data exactly, and the credible band collapses near the observed points while widening in unexplored regions away from the data, giving the GP its built-in uncertainty quantification.
```

```{prf:remark} Built-in uncertainty quantification

 The posterior variance $\sigma_*^2$ is small near observed data (the GP is confident) and large far from observed data (the GP is uncertain). This property is the foundation of Bayesian active learning.
```


(sec-gp_kernels)=
## Kernel Functions and Hyperparameter Learning
The kernel hyperparameters $\bm{\vartheta} = (\ell, \sigma_f, \sigma_y)$ are learned by maximizing the log marginal likelihood:

$$
\log p(\bm{y} | \X, \bm{\vartheta}) = -\frac{1}{2}(\bm{y}-\bm{\mu}_X)^\top K_y^{-1}(\bm{y}-\bm{\mu}_X) - \frac{1}{2}\log|K_y| - \frac{n}{2}\log 2\pi,
$$

where $K_y = K + \sigma_y^2 I$ and $\bm{\mu}_X$ is the prior mean evaluated on the training inputs. In the zero-mean convention used elsewhere in this section, set $\bm{\mu}_X = 0$ (or center the outputs).

**Why marginal likelihood?** The log evidence $\log p(\bm y \mid \X, \bm\vartheta)$ encodes both data fit *and* an automatic complexity penalty in a single closed-form expression. The quadratic form $-\tfrac{1}{2}\bm y^\top K_y^{-1}\bm y$ rewards hyperparameters that explain the centered observations with a small inverse-covariance norm, while the log-determinant term $-\tfrac{1}{2}\log|K_y|$ penalizes overly flexible kernels that admit too many possible functions, giving Bayesian Occam's razor {cite:p}`Rasmussen:2005:GPM:1162254`. Compared with cross-validated MSE, this approach requires no held-out split, makes use of all $n$ observations, and exposes a closed-form gradient with respect to $\bm\vartheta$, which is essential for L-BFGS-style optimization in scikit-learn / GPyTorch. The maximum is reached at the kernel that is just expressive enough to fit the data but no more ({numref}`fig-occam_marginal_likelihood`).

```{figure} figures/fig-occam_marginal_likelihood.svg
:name: fig-occam_marginal_likelihood

Marginal-likelihood Occam's razor for a GP. As the kernel becomes more flexible (smaller length scale $\ell$), the data-fit term improves but the $\log|K_y|$ complexity penalty grows linearly. Their sum, the log evidence, peaks at an interior optimum that is just expressive enough to explain the data, automatically. No held-out validation set is required.
```

**A free held-out diagnostic.** Marginal likelihood is not the only validation tool. The leave-one-out (LOO) predictive error of a GP admits a closed form using the same Cholesky factor already computed for posterior inference, so it costs nothing extra; we develop the formula in {ref}`sec-gp_loo`, where it serves as an iteration-by-iteration health check inside GP-based VFI, and reuse it in {ref}`sec-smm_gp_moments` for the GP layer over the SMM moment map.

### Kernel Composition

More complex covariance structures can be built by composing simpler kernels. The sum of two kernels is again a valid kernel, as is the product. For example:

- $k_\mathrm{SE} + k_\mathrm{periodic}$: captures a smooth trend plus periodic oscillations.

- $k_\mathrm{SE}(\ell_1) \cdot k_\mathrm{SE}(\ell_2)$: models interactions between two length scales.

(sec-matern)=
##### The Matérn kernel family.
The Matérn kernel is parameterized by a smoothness parameter $\nu > 0$:

$$
k_{\mathrm{Mat\acute{e}rn}}(r) = \sigma_f^2\,\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\, r}{\ell}\right)^\nu K_\nu\!\left(\frac{\sqrt{2\nu}\, r}{\ell}\right),
$$

where $r = \|\x - \x'\|$, $\ell$ is the length scale, and $K_\nu$ is the modified Bessel function of the second kind. The smoothness parameter $\nu$ controls the regularity of sample paths: a Matérn-$\nu$ GP is $k$-times mean-square differentiable for every integer $k < \nu$, with the RBF kernel recovered in the limit $\nu \to \infty$. Important special cases:

- $\nu = 1/2$: the Ornstein–Uhlenbeck kernel $k(r) = \sigma_f^2 \exp(-r/\ell)$; sample paths are continuous but nowhere differentiable.

- $\nu = 3/2$: $k(r) = \sigma_f^2(1 + \sqrt{3}\,r/\ell)\exp(-\sqrt{3}\,r/\ell)$; once differentiable.

- $\nu = 5/2$: $k(r) = \sigma_f^2(1 + \sqrt{5}\,r/\ell + 5r^2/(3\ell^2))\exp(-\sqrt{5}\,r/\ell)$; twice differentiable.

- $\nu \to \infty$: recovers the squared exponential (RBF) kernel; infinitely differentiable.

For economic applications where the target function may have kinks (e.g., due to occasionally binding constraints), the Matérn-3/2 kernel is often a better choice than the infinitely smooth RBF kernel, as it does not oversmooth near non-differentiable features {cite:p}`rennerscheidegger_2018`.

(sec-bal)=
## Bayesian Active Learning for Sample-Efficient Training
When model evaluations are expensive, we wish to select training points that provide maximal information. Bayesian Active Learning (BAL) uses the GP posterior variance to guide this selection. The information-theoretic foundations go back to {cite:t}`mackay1992information`, with submodular guarantees for variance-based sensor placement established by {cite:t}`krause2008near` and the widely used upper-confidence-bound formulation of {cite:t}`srinivas2010gaussian`.

**Connection to Bayesian optimization ({ref}`ch-nas`).** BAL is the active-learning twin of the Bayesian-optimization (BO) recipe introduced in {ref}`ch-nas`: same GP surrogate, same acquisition-function machinery. The difference is the target. BO seeks a *scalar optimum* of the surrogate (e.g. Expected Improvement steers samples toward $\argmax \hat f$), so its acquisition trades exploration against the chance of beating the current best. BAL instead targets *global function approximation*: it allocates samples wherever posterior variance is largest, irrespective of the predicted value. Both reduce to the same primitive (fit GP, maximize acquisition, evaluate, refit), but the choice of acquisition reflects whether one wants the best point or the best surrogate.

```{prf:definition} BAL Acquisition Function

$$
U(\x) = w_{\mathrm{obj}} \cdot \mu(\x) + \frac{w_{\mathrm{var}}}{2}\log\sigma^2(\x),
$$
where $\mu(\x)$ and $\sigma^2(\x)$ are the GP posterior mean and variance, $w_{\mathrm{obj}}$ is the objective-following (or Bayesian-optimization) weight, and $w_{\mathrm{var}}$ controls exploration. This logarithmic-variance form is the one used by {cite:t}`rennerscheidegger_2018` in economic applications; it is a non-standard relative of GP-UCB {cite:p}`srinivas2010gaussian`. When the goal is global surrogate accuracy rather than optimization, integrated posterior variance or expected error reduction can be more natural objectives. The weights $w_{\mathrm{obj}}$ and $w_{\mathrm{var}}$ are local to the BAL context and distinct from the discount factor $\beta$ and shock-persistence notation used elsewhere in the script.
```


```{prf:definition} Algorithm: Bayesian Active Learning

- **Input:** Initial design $\mathcal{D}_0 = \{(\x_i, y_i)\}_{i=1}^{n_0}$, budget $N$, kernel $k$, acquisition weights $(w_{\mathrm{obj}}, w_{\mathrm{var}})$
- Fit GP on $\mathcal{D}_0$: learn hyperparameters via marginal likelihood
- **for** $n = n_0+1, \ldots, N$ **do**
  - Compute acquisition function: $U(\x) = w_{\mathrm{obj}} \cdot \mu_{n-1}(\x) + \frac{w_{\mathrm{var}}}{2}\log\sigma_{n-1}^2(\x)$
  - Select next point: $\x_n = \argmax_{\x} U(\x)$
  - Evaluate expensive model: $y_n = f(\x_n)$
  - Update dataset: $\mathcal{D}_n = \mathcal{D}_{n-1} \cup \{(\x_n, y_n)\}$
  - Re-fit GP on $\mathcal{D}_n$ (update hyperparameters every $k$ iterations)
- **end**
- **Output:** Trained GP surrogate $\hat{f}(\cdot)$ with posterior mean $\mu_N$ and variance $\sigma_N^2$
```


The BAL algorithm concentrates training points near kinks, boundary layers, and other regions where the function is hardest to approximate, achieving the same accuracy as uniform sampling with far fewer evaluations {cite:p}`rennerscheidegger_2018`. The exploration–exploitation trade-off controlled by $(w_{\mathrm{obj}}, w_{\mathrm{var}})$ ensures that the algorithm balances refining the approximation in already well-sampled regions against exploring uncharted territory.

The intuition behind the acquisition function is as follows. The first term $w_{\mathrm{obj}} \cdot \mu(\x)$ favors regions where the predicted function value is large (exploitation: sample where the function is interesting). The second term $\frac{w_{\mathrm{var}}}{2}\log\sigma^2(\x)$ favors regions of high uncertainty (exploration: sample where we know least). This acquisition function belongs to the Upper Confidence Bound (UCB) family {cite:p}`srinivas2010gaussian`: with $w_{\mathrm{obj}} = 0$ it has the same maximizers as pure posterior-variance sampling, while the logarithmic variance weighting provides a more conservative exploration bonus than the standard-deviation weighting used in GP-UCB when it is combined with exploitation. By adjusting $w_{\mathrm{obj}}$ and $w_{\mathrm{var}}$, the practitioner can control the balance. For economic applications where the entire domain is relevant (e.g., approximating a policy function), a pure exploration strategy ($w_{\mathrm{obj}} = 0$, $w_{\mathrm{var}} > 0$) is often appropriate, reducing BAL to an uncertainty sampling scheme that minimizes the integrated posterior variance. {numref}`fig-bal-iterations` illustrates two BAL iterations on a 1D toy.

```{figure} figures/gp_active_learning.pdf
:name: fig-bal-iterations
:width: 100%

Bayesian Active Learning in action. **(a)** Starting from three initial observations (red dots), the GP posterior mean (blue line) deviates from the true function (dashed black) in the data-sparse region, where the 95% credible band (blue shading) is wide. **(b)** The acquisition function selects the point of maximum posterior variance (green diamond); after evaluation, the posterior tightens locally and the mean improves. **(c)** A second active-learning iteration fills the remaining gap. With only five strategically chosen points, the GP posterior closely tracks the true function across the entire domain.
```

(sec-gp_vs_dnn)=
## When to Use GPs vs. DNNs
Now that the GP machinery (posterior inference, kernel design, and BAL) has been introduced, we can give a more detailed comparison than the overview in {numref}`tab-surrogate_strategy_comparison`. Active subspaces, introduced in {ref}`sec-active_subspaces`, are the main tool for pushing GP surrogates beyond the moderate-dimensional regime. {numref}`tab-gp_vs_dnn_surrogates` extends {numref}`tab-surrogate_strategy_comparison` with the GP-specific items (LOO diagnostics, marginal-likelihood Occam, BAL).

````{table}
:name: tab-gp_vs_dnn_surrogates

GP and DNN surrogate trade-offs after the basic GP machinery has been introduced. GPs are sample-efficient and uncertainty-aware but expensive in the number of training points; DNNs scale to much larger datasets and dimensions but need more data and separate machinery for calibrated uncertainty.\
$\dagger$ Sparse-GP via inducing points reduces $\mathcal{O}(N^3)$ to $\mathcal{O}(Nm^2 + m^3)$ for $m \ll N$ inducing inputs {cite:p}`titsias2009variational,hensman2013gaussian`; see the inducing-point remarkbox below.

| **Criterion** | **Gaussian Processes** | **Deep Neural Networks** |
|---|---|---|
| Training cost | $\mathcal{O}(N^3)$; exact for $N$ in the low thousands^$\dagger$^ | $\mathcal{O}(N)$ per epoch; scales to $N \sim 10^6$ |
| Prediction cost | $\mathcal{O}(N)$ posterior mean, $\mathcal{O}(N^2)$ posterior variance, per test point | $\mathcal{O}(\text{weights})$ per forward pass; independent of $N$ |
| Gradient access | closed-form from posterior | autodiff (exact, any order) |
| Non-smooth features | Matérn-$3/2$ adapts well | excellent with enough data |
| High-dim. extension | active subspaces ($d \gg 10$) | native |
````

**Practical guidelines.**

- **Choose a GP** when each model evaluation is expensive (minutes to hours per solve), the effective dimensionality is moderate ($d \lesssim 15$, or higher with active subspaces), and uncertainty quantification on the surrogate is needed, e.g., for reporting confidence intervals on estimated parameters or for guiding further data collection via BAL.

- **Choose a DNN** when training data is cheap to generate, the input dimension is high ($d \gg 10$), and the downstream task requires millions of fast evaluations, e.g., Monte Carlo simulation, grid search over a large parameter space, or real-time inference.

- **Combine both** when the problem has a natural two-stage structure: use a GP with BAL to build a small, high-quality training set with uncertainty estimates, then train a DNN on the resulting dataset for fast large-scale deployment. The active subspace approach of {cite:t}`SCHEIDEGGER201968` is particularly effective in this setting, extending GP methods far beyond their naïve scaling limits.

### GP Regression in Practice

In code, GP regression is a one-liner once the kernel is chosen. In scikit-learn the standard pattern is to assemble a kernel as the sum of an RBF (`RBF(length_scale=...)`) and a noise term (`WhiteKernel(noise_level=...)`), pass it to `GaussianProcessRegressor`, call `.fit(X_train, y_train)`, and obtain posterior mean and standard deviation from `.predict(X_test, return_std=True)`; the kernel hyperparameters (`length_scale`, `noise_level`, output amplitude) are optimized by maximizing the marginal likelihood, with `n_restarts_optimizer` controlling robustness to local optima. The companion notebook [`02_GP_and_BAL.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_02_GP_and_BAL.ipynb) provides a full worked example fitting noisy observations of $\sin(x)$ on $[-2,2]$.

**Application: GP surrogates for option pricing.** GPs are particularly well suited as surrogates for derivative pricing models. For example, one can train a GP on as few as 5–50 Black–Scholes option prices (evaluated at different spot prices or parameter configurations) and obtain a surrogate that accurately reproduces the pricing surface with calibrated uncertainty bands. The posterior variance immediately quantifies the interpolation uncertainty at each query point. This idea extends naturally to stochastic volatility models such as Heston, where the analytical pricing formula is expensive to evaluate. Furthermore, because GP predictions are linear in the training targets, the uncertainty of a *portfolio* of GP-priced instruments propagates analytically: for a linear portfolio $\sum_i w_i \hat{V}_i$ with vector of weights $\bm{w}$ and joint posterior covariance $\Sigma_{\hat{V}}$, $\mathrm{Var}(\bm{w}^\top \hat{V}) = \bm{w}^\top \Sigma_{\hat{V}} \bm{w}$. When the surrogate errors are independent across instruments, $\Sigma_{\hat{V}}$ is diagonal with entries $\sigma_i^2$ and the formula reduces to $\sum_i w_i^2 \sigma_i^2$; otherwise the off-diagonal cross-instrument covariances must be retained, e.g. via a multi-output GP. Either way the assessment is instant.

```{prf:remark} Computational scaling of GPs

 The main limitation of GPs is their $\mathcal{O}(n^3)$ training cost, arising from the inversion of the $n \times n$ kernel matrix. For $n > 10{,}000$, exact GP inference becomes impractical. Approximate methods, such as variational sparse GPs with inducing points {cite:p}`titsias2009variational`, stochastic-variational GPs {cite:p}`hensman2013gaussian`, random Fourier features, and structured kernel interpolation, can extend the range to $n \sim 10^5$, but for truly large-scale problems, deep neural networks remain the method of choice. The BAL framework mitigates this limitation by keeping $n$ small through intelligent sample selection.

The *inducing-point* idea ({numref}`fig-inducing_points`) is simple: instead of carrying the full $n\times n$ kernel matrix, summarize the dataset with a much smaller set of $m \ll n$ pseudo-inputs $\bm Z = \{\bm z_1,\dots,\bm z_m\}$ and replace $K_{nn}$ by the Nyström-style low-rank approximation $K_{nm}K_{mm}^{-1}K_{mn}$. Training and prediction then cost about $\mathcal{O}(nm^2 + m^3)$ rather than $\mathcal{O}(n^3)$, with the $m^3$ term coming from the small inducing-point block. The variational formulation of {cite:t}`titsias2009variational` additionally treats $\bm Z$ as parameters to be optimized against the marginal likelihood, so the inducing inputs migrate to wherever the GP actually needs resolution.
```


```{figure} figures/fig-inducing_points.svg
:name: fig-inducing_points

Inducing-point intuition. Exact GP inference conditions on all $n=21$ observations and uses the full $n\times n$ kernel matrix. Sparse variational GP methods introduce $m\ll n$ inducing inputs $\bm Z$ and approximate the posterior through the low-rank structure induced by $K_{nm}K_{mm}^{-1}K_{mn}$. The top panel compares the exact posterior mean and uncertainty band with a sparse approximation using $m=4$ pseudo-inputs; the bottom panel shows the absolute difference between the two means. In this smooth one-dimensional illustration the approximation is close, but its quality depends on $m$, the kernel hyperparameters, and the placement of $\bm Z$. The dominant training cost falls from $\mathcal{O}(n^3)$ to $\mathcal{O}(nm^2 + m^3)$; the variational formulation of {cite:t}`titsias2009variational` optimizes $\bm Z$ jointly with the kernel hyperparameters.
```

(sec-active_subspaces)=
## Scaling GPs to High Dimensions: Active Subspaces
A common criticism of GP methods is that they are limited to moderate-dimensional problems ($d \leq 10$). While this is true for naïve implementations, where the number of training points needed to cover the input space grows exponentially in $d$, {cite:t}`SCHEIDEGGER201968` show that *active subspace* methods can mitigate this barrier *when the target function exhibits a clear spectral gap in the gradient outer-product matrix $C$ of {eq}`eq-active_subspace` below*, i.e., when its variation is concentrated on a low-dimensional linear subspace of the input space. If $f$ depends comparably on all $d$ coordinates, the spectral gap is absent and the technique offers little gain; sufficient conditions are in {cite:t}`constantine2015active`. For a textbook treatment of the active subspace framework and references to its origins, see {cite:t}`constantine2015active`; we summarize the key ideas and their application to dynamic economic models below.

**Intuition: why most directions don't matter.** Consider a value function $V(\x)$ in a dynamic stochastic economy with $d$ state variables. Although $V$ is formally a function of all $d$ inputs, it often responds primarily to a few linear combinations of them. For example, in a multi-country model, the value function may depend mostly on *aggregate* capital $\sum_j k^j$ and a measure of *inequality* $\mathrm{Var}(k^j)$, rather than on each country's capital individually. If we can identify these important directions, we can project the $d$-dimensional input onto a much lower-dimensional subspace and fit the GP there, substantially mitigating the practical grid-based curse of dimensionality.

**The gradient outer product matrix.** The key diagnostic tool is the *gradient outer product matrix* {cite:p}`constantine2015active`. Given a function $f\colon \R^d \to \R$ with input distribution $\pi(\x)$, define:

$$
C = \int \nabla f(\x) \, \nabla f(\x)^\top \, \pi(\x)\, d\x \;\in\; \R^{d \times d}.
$$ (eq-active_subspace)

The matrix $C$ is symmetric and positive semi-definite. Its $(i,j)$-entry measures how much $f$ co-varies along input directions $i$ and $j$, averaged over the input distribution. In particular:

- If $\partial f / \partial x_i$ is consistently large across the input space, the $i$-th diagonal entry of $C$ is large, meaning direction $i$ is important.

- If two directions always change $f$ together, the corresponding off-diagonal entry is large, as they form part of the same important linear combination.

- If $\partial f / \partial x_i \approx 0$ everywhere, then direction $i$ contributes nothing to $C$, so it is irrelevant and can be projected away.

**Worked toy example.** Take $f(x_1, x_2) = x_1^2$ on $\x \sim \mathrm{Unif}([-1,1]^2)$. Then $\nabla f = (2x_1, 0)^\top$, and integrating against the uniform measure gives $C = \mathrm{diag}(4/3,\,0)$. The eigenvalues are $\lambda_1 = 4/3$ and $\lambda_2 = 0$; the first eigenvector is $(1,0)^\top$, the second is $(0,1)^\top$; the spectral gap is infinite. The active subspace is the $x_1$-axis, and $f$ is recovered exactly as $f = g(x_1)$ with $g(u) = u^2$. This is the cleanest instance of the projection in {eq}`eq-linear_as_ansatz` below.

**Eigendecomposition and the spectral gap.** Let $C = U \Lambda U^\top$ be the eigendecomposition, with eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$. The eigenvectors reveal the directions of maximal average variation; the eigenvalues quantify how much variation each direction captures. Each diagonal entry of $C$ is the average squared sensitivity of $f$ along that input direction, so eigenvectors of $C$ are directions of *correlated* sensitivity: when the top $m$ eigenvalues capture most of the trace, $f$ varies primarily along their span and is approximately constant in the perpendicular $(d-m)$-dimensional complement. If there is a clear *spectral gap*, i.e., $\lambda_m \gg \lambda_{m+1}$, then the first $m$ eigenvectors $U_m = [u_1, \ldots, u_m]$ span the *active subspace*, and $f$ is well approximated by a function of the reduced coordinates $\tilde{\x} = U_m^\top \x \in \R^m$ alone.

$$
f(\x) \;\approx\; g(U_m^\top \x),
\qquad U_m = [u_1,\ldots,u_m].
$$ (eq-linear_as_ansatz)

```{figure} figures/fig-active_subspace_spectrum.svg
:name: fig-active_subspace_spectrum

Spectral decay of the active-subspace eigenvalues for a schematic example with $d=12$ state variables. The first three eigenvalues are orders of magnitude larger than the rest; the dashed red line marks the spectral gap after $\lambda_3$, which indicates that $f$ effectively lives on a 3-dimensional active subspace (green brace) and reduces the GP regression problem from $\R^{12}$ to $\R^3$. The remaining nine directions form the inactive subspace (gray brace), along which $f$ is nearly constant on average.
```

**Computing the gradient.** In practice, the integral in {eq}`eq-active_subspace` is estimated from a finite sample of gradient evaluations {cite:p}`constantine2015active` (this is a one-time pre-processing cost, $\mathcal{O}(N_g\, d)$ with $N_g \sim 2d$–$10d$, far smaller than the surrogate training set of size $N$):

$$
\hat{C} = \frac{1}{N_g}\sum_{i=1}^{N_g} \nabla f(\x_i)\,\nabla f(\x_i)^\top, \qquad \x_i \sim \pi(\x).
$$ (eq-empirical_C)

The gradient $\nabla f(\x_i)$ can be obtained via:

- **Automatic differentiation** through the model solver (if differentiable, e.g., a DEQN or PINN).

- **Finite differences**: evaluate $f$ at $d+1$ perturbed inputs per sample point. This costs $(d+1)\cdot N_g$ model evaluations, but $N_g$ can be modest (typically $N_g \sim 2d$ to $10d$ suffices for a reliable estimate of the leading eigenspace).

The eigendecomposition of $\hat{C}$ is a standard $d \times d$ matrix operation and is cheap relative to the model evaluations. When $\{\x_i\}$ are i.i.d. from $\pi$, $\hat{C} \to C$ almost surely as $N_g \to \infty$ by the law of large numbers, so the eigenvectors of $\hat{C}$ are consistent estimates of the population active subspace; finite-sample bias is documented in {cite:t}`constantine2015active`.

**Practical workflow.** The full dimension-reduction pipeline consists of three steps:

1.  **Gradient sampling:** Draw $N_g$ points $\x_i \sim \pi(\x)$ and compute $\nabla f(\x_i)$ at each. Form the empirical estimate $\hat{C}$ via {eq}`eq-empirical_C`.

2.  **Eigendecomposition and dimension selection:** Compute $\hat{C} = \hat{U}\hat{\Lambda}\hat{U}^\top$. Inspect the eigenvalue decay, as in {numref}`fig-active_subspace_spectrum`, and choose $m$ at the spectral gap.

3.  **GP in reduced space:** Project all training inputs onto the active subspace: $\tilde{\x}_i = \hat{U}_m^\top \x_i \in \R^m$. Fit a standard GP on the $m$-dimensional projected data.

When combined with BAL, the GP adaptively selects training points in the reduced space, further improving sample efficiency.

```{figure} figures/fig-active_subspace_pipeline.svg
:name: fig-active_subspace_pipeline

Linear active-subspace pipeline. Gradient samples identify the dominant eigenspace of the gradient outer-product matrix, all simulator inputs are projected to the reduced coordinates $\tilde{\x}=U_m^\top \x$, and the GP/BAL loop is then run in the low-dimensional active subspace.
```

{numref}`fig-active_subspace_pipeline` summarizes the linear active-subspace workflow used before fitting the GP.

**Application to dynamic stochastic economies.** {cite:t}`SCHEIDEGGER201968` apply this pipeline to high-dimensional dynamic programming problems arising in neoclassical growth models. Their key findings:

- Models with up to $d = 500$ continuous state variables can be solved, far beyond the reach of grid-based methods or naïve GP implementations.

- The active subspace typically reduces the effective dimension to $m = 2$–$5$, even when $d$ is in the hundreds. The value function's dependence on the full state vector collapses onto a few aggregate quantities (e.g., mean capital, dispersion of productivity).

- The state space is partitioned into clusters via Bayesian Gaussian mixture models, and a separate GP with BAL is fit within each cluster. This local approximation strategy handles non-stationarities in the value function (e.g., different curvature in high- vs. low-capital regions).

- Parallel computing distributes the gradient evaluations and GP fits across multiple processors, enabling the method to scale to large-scale models on high-performance computing clusters.

**Anchor on the multi-sector growth benchmark.** In the multi-sector neoclassical growth benchmark of {cite:t}`SCHEIDEGGER201968`, the active-subspace dimension collapses to $m = 1$ even at $D = 500$, so the entire ASGP pipeline operates on a one-dimensional projected coordinate. On models with other configurations the active subspace can be larger; the spectral-gap test is what decides.

(sec-deep_as)=
### Nonlinear Generalization: Deep Active Subspaces
The linear pipeline of the previous section assumes that the important directions are *linear* combinations of the input variables. That is an inductive bias: the first $m$ columns of $U$ always span a linear subspace of $\R^d$, so if the response $f$ actually varies along a *curved* low-dimensional manifold (a ridge, a product of two linear features, a radial coordinate, a hidden regime indicator), the linear projection has to carry enough directions to cover the whole manifold, not the manifold itself. Concretely, if $f(\xi) = \varphi\!\bigl((w_1^\top \xi)^2 + (w_2^\top \xi)^2\bigr)$ is a radial function of two linear features, the gradient $\nabla f = 2\,\varphi'(\cdot)\,(s_1 w_1 + s_2 w_2)$ spans a *two*-dimensional space when averaged over $\xi$, so linear AS returns $m = 2$; yet the intrinsic "interesting" coordinate is the scalar aggregate $r^2 = s_1^2 + s_2^2$, which is one-dimensional. {cite:t}`tripathy2018deep` remove the linearity bias by replacing the linear projection with a learned nonlinear encoder and thereby recover the one-dimensional structure.

**The deep active-subspace ansatz.** Tripathy and Bilionis parametrize the surrogate as a composition of two small neural networks,

$$
\hat f(\xi) \;=\; g\bigl(h(\xi)\bigr),
\qquad
h\colon \R^D \!\to\! \R^d,
\qquad
g\colon \R^d \!\to\! \R,
$$ (eq-deep_as_ansatz)

where $h$ is a multilayer perceptron (MLP) that plays the role of the projection matrix $U_m^\top$ in the linear case, and $g$ is a second MLP that plays the role of the GP link. Setting $h(\xi) = U_m^\top \xi$ and taking $g$ to be a polynomial recovers the linear active-subspace surrogate {eq}`eq-linear_as_ansatz`, so {eq}`eq-deep_as_ansatz` generalizes the linear-AS ansatz rather than the gradient covariance matrix {eq}`eq-active_subspace` itself. The two networks are trained *jointly* to minimize the residual $\hat f(\xi_i) - y_i$ on an input-output sample, so that the bottleneck $h(\xi) \in \R^d$ adapts to the specific response surface rather than being fixed in advance by the spectrum of $C$.

**Architecture choices.** Three design choices make {eq}`eq-deep_as_ansatz` trainable with a few hundreds of samples:

- **Exponentially decaying encoder widths** {cite:p}`tripathy2018deep`. For an encoder with $L$ layers mapping $\R^D$ to $\R^d$, choose widths

$$
d_k \;=\; \bigl\lceil D \, e^{\eta_{\mathrm{w}} k} \bigr\rceil, \qquad \eta_{\mathrm{w}} = \tfrac{1}{L}\log(d/D), \qquad k = 0, 1, \ldots, L,
$$ (eq-tb_widths)

so that the bottleneck closes smoothly from $D$ to $d$ without a brittle hyperparameter choice. For $D = 20,\, L = 3,\, d = 1$ this gives widths $[20, 8, 3, 1]$; for $d = 2$, widths $[20, 10, 5, 2]$ – a recipe rather than a search.

- **Swish activation** {cite:p}`tripathy2018deep`,

$$
\sigma(z) \;=\; \frac{z}{1 + e^{-\gamma z}}, \qquad \gamma = 1,
$$ (eq-swish)

which is smooth everywhere (unlike ReLU) and non-saturating (unlike $\tanh$); smoothness matters because we will want to differentiate through $\hat f$ for sensitivity analysis and because the elastic-net penalty below otherwise has to fight the sharp corners of ReLU.

- **Elastic-net regularization on every weight matrix** {cite:p}`tripathy2018deep`. Writing $\theta = (\theta_h, \theta_g)$ for all encoder and link parameters, the training loss is

$$
\mathcal{L}(\theta) \;=\; \frac{1}{N}\sum_{i=1}^N \bigl(\hat f_\theta(\xi_i) - y_i\bigr)^2
  \;+\; \lambda_1 \lVert \theta \rVert_1
  \;+\; \lambda_2 \lVert \theta \rVert_2^2,
$$ (eq-tb_loss)

with small $\lambda_1, \lambda_2$; the $\ell_1$ term encourages sparse weights and input usage, the $\ell_2$ term controls the overall smoothness of $\hat f_\theta$ and prevents the loss landscape from becoming pathological as $N$ shrinks.

Crucially, *no gradient samples of $f$* are required: $\theta$ is learned from the input-output pairs $\{(\xi_i, y_i)\}$ alone. The orthogonality constraint $U^\top U = I$ that is implicit in the linear case becomes unnecessary, because the learned encoder is free to pick any smooth low-dimensional parameterization of the active manifold.

**Choosing the latent dimension $d$.** The spectral gap of $C$ is no longer available – the encoder is nonlinear – so $d$ is chosen by a *validation-MSE elbow*: hold out an independent fraction of the sample, train a small family of models with $d = 1, 2, 3, \ldots$, and pick the smallest $d$ beyond which held-out error no longer drops significantly. An operational rule of thumb is to stop at the first $d$ for which the MSE improvement from $d$ to $d+1$ is less than a factor of two: smaller gains are typically driven by optimization slack, not by new latent structure. On curved problems this elbow lies *strictly below* the linear-AS spectral gap: the deep encoder collapses two linear features into a single nonlinear aggregate (notebook `09_Deep_Active_Subspace_Ridge` gives a reproducible instance in $D = 20$). On nearly-linear problems the two criteria agree qualitatively, and at small training-set sizes a polynomial link on top of a two-dimensional linear AS can in fact be more data-efficient than the deep encoder (notebook `10_Deep_AS_vs_Linear_AS_Borehole`, the canonical borehole benchmark with $D = 8$ and $N = 500$). {numref}`fig-deep_as_elbow` contrasts the elbow rule with the linear-AS spectral gap on the radial-ridge target.

```{figure} figures/fig-deep_as_elbow.svg
:name: fig-deep_as_elbow

Stylized comparison of the two selection criteria for the radial-ridge target $y(\xi) = \exp(-[(w_1^\top\xi)^2 + (w_2^\top\xi)^2])$ in $D = 20$ (see notebook `09`). The linear-AS eigenvalue spectrum has two dominant directions, so the spectral-gap criterion picks $d = 2$ (right green dashed line). The deep-AS validation MSE, by contrast, is already at its plateau at $d = 1$ (left green dashed line): the learned encoder $h_\theta$ represents the nonlinear aggregate $r^2 = s_1^2 + s_2^2$ as a *scalar*. The curves are stylized; the orders of magnitude track the notebook (linear-AS eigenvalues $\approx 0.16$ for $i = 1, 2$ then a sharp drop; deep-AS validation MSE $\approx 10^{-4}$ at $d = 1$ and roughly flat thereafter), and the elbow-at-$d=1$ versus spectral-gap-at-$d=2$ contrast is reproduced.
```

**Training recipe.** A practical recipe that reproduces the experiments in notebooks `09` and `10`:

1.  Sample $N$ input-output pairs $(\xi_i, y_i)$; split $80/20$ into train and validation sets. Standardize inputs to unit variance and center outputs.

2.  For each candidate $d \in \{1, 2, 3, \ldots\}$: build $h_\theta$ with widths {eq}`eq-tb_widths` and $g_\theta$ with two hidden layers ($16$–$32$ units); train on loss {eq}`eq-tb_loss` with Adam ($\text{lr} = 5 \times 10^{-3}$), a cosine learning-rate schedule over $10^3$–$2 \times 10^3$ epochs, and $\lambda_1 = 10^{-5},\, \lambda_2 = 10^{-4}$.

3.  Record the validation MSE, apply the elbow rule, and deploy $\hat f_\theta$ with the chosen $d$.

Sample-budget rule of thumb: $N \approx 50\, d_{\mathrm{nl}}$ to *find* the bottleneck, inflated to $N \approx 200\, d_{\mathrm{nl}}$ for a deployment surrogate. Two post-training sanity checks are worth doing: (i) the validation curve should be monotone-then-flat in $d$; (ii) a scatter of $h_\theta(\xi_i)$ against the top linear-AS coordinate $U_1^\top \xi_i$ reveals whether the encoder has actually gone nonlinear (a non-monotone relation is the fingerprint).

```{figure} figures/fig-deep_as_pipeline.svg
:name: fig-deep_as_pipeline

Deep active-subspace pipeline. Input–output pairs $(\xi_i, y_i)$ are drawn directly from the simulator (no gradient samples needed); a nonlinear encoder $h_\theta$ compresses the high-dimensional input into a $d$-dimensional latent code, a small link network $g_\theta$ maps the latent code to the response, and an elastic-net / validation-MSE elbow chooses $d$. The trained composition $\hat f_\theta = g_\theta \circ h_\theta$ is the deployed surrogate. Compared with the linear active-subspace pipeline ({numref}`fig-active_subspace_pipeline`), the encoder + link boxes replace the eigendecomposition + linear projection, the elbow box replaces the spectral-gap search, and the gradient-sampling step disappears entirely.
```

{numref}`fig-deep_as_pipeline` should be compared with the linear pipeline in {numref}`fig-active_subspace_pipeline`: the two blue encoder / link boxes replace "eigendecomposition + project", the "elastic-net + elbow" box replaces "find spectral gap", and the gradient-sampling step is gone entirely. The economic pay-off is the same, a cheap surrogate in $d$ variables where the original has $D$, but the modeling assumption is weaker: curved active manifolds are now admissible.

**When is the extra machinery worth it?** Deep AS pays off when (i) no gradient samples are available (e.g., black-box simulators, noisy observations from a lab experiment, or calibration targets returned by a proprietary solver), (ii) the spectral gap of $C$ is ambiguous or the response surface genuinely depends on *nonlinear* aggregates of the inputs (ridges, radial coordinates, thresholded features, piecewise regimes), or (iii) the input dimension is large ($D \gg 10$) and a linear projection is too restrictive to capture the active manifold. When none of these conditions apply, linear AS plus a polynomial or a GP link is usually more data-efficient and should be fit first as a baseline. The borehole experiment in notebook `10` is the honest diagnostic: on a function this close to a $d = 2$ ridge, the two curves *cross*, and a cubic polynomial on two linear features beats the deep encoder at all $d \ge 2$. Deep AS's advantage materializes precisely at $d = 1$, where a single linear feature cannot span the active direction. This perspective also connects naturally to deep kernel learning ({ref}`sec-dkl`), which composes a learned feature map with a GP head; deep AS is the same idea for the feature map, with an MLP link in place of the GP.

**Applications in economics and finance.** The deep-AS architecture is particularly attractive in settings where the state variable is high-dimensional but the economic mechanism acts through a few aggregate quantities that are not linear in the state. Examples include (i) heterogeneous-agent models whose cross-sectional distribution of wealth enters the equilibrium law of motion only through a few *moments* (mean, dispersion, tail mass), the Krusell-Smith aggregation logic of {ref}`ch-young`, where the map from the raw distribution to those moments is a learned, nonlinear encoder; (ii) multi-region climate-economy integrated assessment models with hundreds of regional capital stocks, in which damages and optimal carbon taxes depend on nonlinear aggregates of the underlying state (total radiative forcing, regional inequality indices); and (iii) dynamic games and mean-field games in which the relevant action depends on a curved functional of the opponent distribution. In each case the original space is too large for a direct GP and the linear active subspace is too rigid, while the deep encoder provides a trainable compromise. {cite:t}`Bilionis:2016wc` and {cite:t}`tripathy2018deep` provide the general UQ and deep-active-subspace methodology; for integrated assessment models specifically, {cite:t}`friedlDeep2023` is the economics-side reference used in this course.

(sec-gp_dp)=
## Dynamic Programming with Gaussian Processes
Returning to the second oracle announced at the start of this chapter and developed formally in {ref}`sec-gp_dp_supervised_view` below, we now embed the GP and active-subspace machinery developed above into a *value function iteration* (VFI) algorithm, with the Bellman operator $TV$ playing the role of the expensive label generator. The acronym *ASGP* used in the section titles and figure captions below stands for *Active-Subspace Gaussian Process*: the dimensionality reducer of {ref}`sec-active_subspaces` stacked under the GP of {ref}`sec-gp_kernels`.

The idea of representing the value function as a GP inside a DP recursion was introduced under the name *Gaussian process dynamic programming* (GPDP) by {cite:t}`deisenroth2009gaussian`, who used it for continuous-state, continuous-action optimal control problems and combined it with a variance-based active selection of support points. Closely related work {cite:p}`engel2005reinforcement` embeds GPs into temporal-difference learning, replacing tabular value functions with a GP posterior. These ideas later inspired model-based policy search methods such as PILCO {cite:p}`deisenroth2011pilco`, which propagate GP uncertainty through long horizons. In economics, {cite:t}`SCHEIDEGGER201968` show that pairing this paradigm with active subspaces and Bayesian Gaussian mixture models for the ergodic set scales the framework to economies with up to 500 continuous state variables; {cite:t}`rennerscheidegger_2018` apply it to dynamic incentive problems, {cite:t}`gaegauf2023portfolio` to dynamic portfolio choice, and {cite:t}`chen2025private` to dynamic private asset allocation. The remainder of this section presents the algorithm in this combined form.

(sec-gp_dp_supervised_view)=
### Why GPs for DP: a Supervised-Learning View of the Bellman Operator
Before stating the algorithm, it helps to see VFI through a supervised-learning lens. At iteration $s$, given an incumbent value function $V^{s-1}$, we generate a training set

```{math}
:enumerated: false

\mathcal{D}^s = \bigl\{(\x^{(i)},\, t_i^s)\bigr\}_{i=1}^{n^s}, \qquad t_i^s = (TV^{s-1})(\x^{(i)}),
```

and fit a regressor $V^s \approx (TV^{s-1})$ to it. The label $t_i^s$ is not free. Each evaluation of the Bellman operator at a state $\x^{(i)}$ requires solving a constrained nonlinear program over controls $\xi$ subject to the law of motion and any feasibility / market-clearing constraints; quadrature over $\x'$ is taken inside. In textbook problems each oracle call costs $10^{-2}$–$10^{0}$ seconds, and in higher-dimensional models with adjustment costs, occasionally binding constraints, or aggregator nonlinearities, it can run into minutes. Because the calls are independent across $\x^{(i)}$, the design is embarrassingly parallel {cite:p}`SCHEIDEGGER201968`, but each individual label remains expensive.

This is exactly the regime where Gaussian processes shine. Three properties make them the natural surrogate class:

1.  **Sample efficiency under an expensive oracle.** The marginal-likelihood Occam's razor of {ref}`sec-gp_kernels` delivers calibrated fits at $n \sim 10^2$–$10^3$ design points, well below the $n \gg 10^4$ regime in which deep-network surrogates start to dominate ({numref}`tab-gp_vs_bnn`).

2.  **Built-in uncertainty quantification.** The GP posterior variance $\sigma_\mathrm{GP}^2(\x)$ tells us, at every state, how much the current surrogate trusts its own prediction. This is the input that turns a passive interpolant into an *adaptive* one ({ref}`sec-gp_dp_bal_inside`).

3.  **Cheap held-out diagnostics.** The leave-one-out predictive error is available in closed form from the same Cholesky factor used for posterior inference ({ref}`sec-gp_loo`), so we can monitor surrogate health every iteration without spending oracle calls on a held-out split.

The same logic applies, with a different oracle, in the structural estimation chapter: {ref}`ch-estimation` stacks a GP over the moment map $m(\theta)$ where each label is a forward simulation plus moment computation under a candidate parameter, again expensive to evaluate and cheap to store. Both settings are instances of one pattern: *supervised regression on the output of a costly numerical procedure*. The next subsection writes down the formal DP problem; the subsequent subsections develop the GP-based interpolant and the two ingredients (LOO and active learning) that make it sample-efficient at modest design size.

### The Dynamic Programming Problem

Consider an infinite-horizon, discrete-time stochastic optimal decision problem. A representative agent chooses a sequence of controls $\{\xi_t\}_{t=0}^\infty$ to maximize the expected discounted sum of returns:

$$
V(\x_0) = \max_{\{\xi_t\}} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t\, r(\x_t, \xi_t),
$$ (eq-dp_objective)

subject to the law of motion $\x_{t+1} \sim F(\cdot\,|\,\x_t, \xi_t)$, where $\x_t \in \mathcal{X} \subset \R^D$ is the state, $\xi_t \in \Lambda(\x_t)$ is the control, $r$ is the per-period return function, and $\beta \in (0,1)$ is the discount factor.

By the *principle of optimality* {cite:p}`bellman1957`, the infinite-dimensional problem {eq}`eq-dp_objective` reduces to the *Bellman equation*:

$$
V(\x) = \max_{\xi \in \Lambda(\x)} \bigl\{ r(\x, \xi) + \beta\, \mathbb{E}\bigl[V(\x')\bigr] \bigr\},
$$ (eq-bellman)

where the expectation is over the stochastic transition. The *Bellman operator* $T$ maps value functions to value functions:

```{math}
:enumerated: false

(TV)(\x) = \max_{\xi \in \Lambda(\x)} \bigl\{ r(\x, \xi) + \beta\, \mathbb{E}\bigl[V(\x')\bigr] \bigr\}.
```

Under standard regularity conditions (bounded returns, $\beta < 1$, monotonicity, discounting), $T$ is a *contraction mapping* on the Banach space of bounded continuous functions with sup-norm, with modulus $\beta$. By the Banach fixed-point theorem (Appendix {ref}`app-fixed_points`), $T$ admits a unique fixed point $V^*$, and iterating $V^{s+1} = TV^s$ from any initial guess $V^0$ converges geometrically: $\|V^s - V^*\|_\infty \le \beta^s \|V^0 - V^*\|_\infty$. The classical references in economics are {cite:t}`stokeylucas1989` [Chs. 3–4] for the formal contraction-and-fixed-point theory of the Bellman operator and {cite:t}`judd1998numerical` [Ch. 12] for the numerical implementation in continuous-state settings; {cite:t}`ljungqvist2018recursive` [Chs. 3–4] and {cite:t}`sargentstachurski2026dp` provide modern macroeconomic treatments. In the operations-research and optimal-control literature, {cite:t}`Bertsekas:2000:DPO:517430` is the canonical reference, with extensive coverage of approximate dynamic programming. The contraction modulus $\beta$ applies to the *exact* Bellman operator; convergence of the GP-fitted iterates additionally requires controlling the per-step interpolation error of the surrogate, otherwise the approximate operator can fail to contract globally even though the exact one does. This is the standard textbook VFI loop: guess a value function, apply the Bellman operator, interpolate the updated values, and repeat until convergence. In the present chapter, the only change is the interpolant: a GP replaces the grid or polynomial basis when the state space is irregular or moderately high-dimensional. For a comprehensive treatment of dynamic programming in economics, see also the open-source QuantEcon lectures.[^1]

**The computational challenge.** At every VFI iteration, we must *approximate* $V^{s+1}$ as a function of the full state vector $\x \in \R^D$. Traditional approaches (finite grids, Smolyak sparse grids, tensor-product polynomial bases) suffer from the curse of dimensionality: the number of grid points grows exponentially in $D$, and they require hypercubic state-space geometries. For $D > 20$, these methods become infeasible.

(sec-growth_model)=
### The Stochastic Optimal Growth Model
The workhorse test case for GP-based dynamic programming is the multi-sector stochastic optimal growth model. Assume $D$ sectors, each with a capital stock $k_j$, so the state is $\bm{k} = (k_1, \ldots, k_D) \in \R^D_+$. A representative household chooses consumption $\bm{c}$, labor supply $\bm{l}$, and investment $\bm{i}$ to maximize:

$$
V_0(\bm{k}_0) = \max_{\{\bm{c}_t, \bm{l}_t, \bm{i}_t\}} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t\, u(\bm{c}_t, \bm{l}_t),
$$ (eq-growth_bellman)

subject to the capital law of motion $k_{j,t+1} = (1-\delta)\,k_{j,t} + i_{j,t}$, a sector-by-sector resource constraint $c_{j,t} + i_{j,t} + \Gamma_{j,t} = \exp(z_{j,t})\,A\,k_{j,t}^\psi\,l_{j,t}^{1-\psi}$ with convex adjustment cost $\Gamma_{j,t}$, and Cobb–Douglas production $f(k_j, l_j) = A\,k_j^\psi\,l_j^{1-\psi}$. Productivity shocks $z_{j,t}$ follow a stationary process and the dimension $D$ can be scaled from 1 (the textbook Brock–Mirman model of {ref}`ch-deqn`) to 500 without changing the algorithmic structure.

(sec-asgp_vfi)=
### GP-Based Value Function Iteration (ASGP)
The key idea, common to {cite:t}`deisenroth2009gaussian` in the control literature and {cite:t}`SCHEIDEGGER201968` in the economics literature, is to use a *Gaussian process as the interpolation scheme* inside VFI, replacing grid-based methods entirely. At each VFI step $s$:

1.  Generate $n^s$ training inputs $\X = \{\x^{(1)}, \ldots, \x^{(n^s)}\} \subset [\underline{\bm{k}}, \bar{\bm{k}}]^D$.

2.  Evaluate the Bellman operator at each training point: $t_i^s = (TV^{s-1})(\x^{(i)})$.

3.  Learn a GP (or ASGP) surrogate $V_\mathrm{surr}$ from the training data $\{\X, \bm{t}\}$.

4.  Set $V^s = V_\mathrm{surr}$ (the GP posterior mean).

5.  Compute the convergence error $e^s = \|V^s - V^{s-1}\|_\infty / \Delta_V$, where $\Delta_V$ is a normalizing value-function range.

6.  If $e^s < \bar{e}$, stop; otherwise continue.

**Advantages over grid-based methods.** This GP-VFI approach offers several structural advantages:

- **Arbitrary geometries:** GPs require no tensor-product structure, so training points can be placed anywhere in the state space, including on irregularly shaped ergodic sets.

- **Adaptive training:** Points where the Bellman operator fails to converge (e.g., near constraints) can simply be excluded and retried later; adding or removing points is trivial.

- **Built-in uncertainty quantification:** The GP posterior variance provides interpolation uncertainty at every state point, at every iteration, "free" UQ.

- **Active subspace integration:** When $D \gg 10$, the GP operates on the projected inputs $\tilde{\x} = U_m^\top \x \in \R^m$ discovered via the active subspace pipeline ({ref}`sec-active_subspaces`), reducing the effective dimensionality from hundreds to a handful.

**Parallelization.** The most expensive part of the algorithm, evaluating the Bellman operator at $n^s$ training points, is *embarrassingly parallel*: each evaluation is independent. In the MPI implementation of {cite:t}`SCHEIDEGGER201968`, the current value function surrogate is broadcast to all workers, each worker evaluates the Bellman operator at $n^s/n_\mathrm{cpu}$ points, and the results are gathered at the master for GP fitting. Communication cost is negligible relative to the Bellman evaluations.

(sec-gp_loo)=
### Leave-One-Out Error: a Held-out Diagnostic for Free
How do we know that the GP surrogate at iteration $s$ is good enough to take another Bellman step? The marginal-likelihood objective of {ref}`sec-gp_kernels` fits the kernel, but it does not, on its own, certify pointwise predictive accuracy on the design. A held-out validation split would, but every held-out point is one fewer expensive Bellman label going into the GP.

The standard escape route in GP regression is the leave-one-out (LOO) predictive error, and it has the pleasant feature that for a Gaussian process it admits a closed form using the same Cholesky factor already computed for the posterior {cite:p}`Rasmussen:2005:GPM:1162254`:

$$
\mu_{-i}(\x^{(i)}) - y_i = -\frac{\alpha_i}{\bigl[K_y^{-1}\bigr]_{ii}},
\qquad
\sigma_{-i}^2(\x^{(i)}) = \frac{1}{\bigl[K_y^{-1}\bigr]_{ii}},
$$ (eq-gp_loo)

where $\mu_{-i}$ and $\sigma_{-i}^2$ denote the posterior mean and variance after removing the $i$-th observation, and $\alpha = K_y^{-1}(\bm y-\bm\mu_X)$ (for centered outputs, $\bm\mu_X=0$). Since $K_y^{-1}$ is recovered from the Cholesky factor of $K_y$ in $\mathcal{O}(n^2)$ once the factorization has been done, computing the full $n$-vector of LOO residuals is essentially free relative to the $\mathcal{O}(n^3)$ already paid for posterior inference.

**What the LOO RMSE tells us.** Tracking

```{math}
:enumerated: false

\mathrm{LOO\text{-}RMSE}(\mathcal{D}^s) \;=\; \sqrt{\frac{1}{n^s}\sum_{i=1}^{n^s}\bigl(\mu_{-i}(\x^{(i)}) - t_i^s\bigr)^2}
```

across VFI iterations is a cheap surrogate-health metric that is independent of the Bellman residual:

- A flat-then-rising LOO curve at the same design size signals *kernel mis-specification* (length scale collapsing, noise variance hitting a bound) and tells us to revisit the kernel choice or hyperparameter bounds before adding more design points.

- A high LOO RMSE at small $n^s$ that decays as the design grows is the expected behavior and tells us that the surrogate simply needs more labels.

- A small LOO RMSE coexisting with a large Bellman residual points the finger at the *operator*, not the surrogate: the iterate may be far from the fixed point even though the GP fits the current $TV^{s-1}$ well.

The notebook [`04_GP_Value_Function_Iteration.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_04_GP_Value_Function_Iteration.ipynb) computes {eq}`eq-gp_loo` via `scipy.linalg.cho_solve` (function `gp_loo_rmse`) and reports it alongside the Bellman residual at every VFI iteration, separating these two failure modes. The same diagnostic reappears in {ref}`sec-smm_gp_moments` for the GP layer over the SMM moment map.

(sec-gp_dp_bal_inside)=
### Active Learning Inside the VFI Loop
The Bayesian active-learning machinery of {ref}`sec-bal` carries over almost verbatim once the VFI loop is in place, but with one important adjustment: the goal is now *uniform interpolation accuracy* of the value function on the relevant state-space region, not maximization of a payoff or minimization of a loss. The right acquisition function is therefore pure exploration, the GP posterior standard deviation, rather than a UCB or expected-improvement criterion that trades off exploitation and exploration:

$$
\x^{\mathrm{next}} \in \argmax_{\x \in \mathcal{X}^\mathrm{cand}} \sigma_\mathrm{GP}^s(\x).
$$ (eq-bal_vfi)

A practical implementation, used in notebook [`04_GP_Value_Function_Iteration.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_04_GP_Value_Function_Iteration.ipynb), runs the VFI iterations with a frozen design for a few steps, then *enriches* the design every $N$ iterations:

1.  Evaluate $\sigma_\mathrm{GP}^s$ on a dense candidate set $\mathcal{X}^\mathrm{cand}$ (a Latin-hypercube draw over the current state-space region).

2.  Pick the top-$n_\mathrm{add}$ candidates by posterior standard deviation, subject to a minimum-spacing constraint $\|\x^{(\mathrm{new})} - \x^{(j)}\| \ge \delta_\mathrm{spacing}$ against existing design points to avoid clustering.

3.  Evaluate the Bellman operator at the new points (one expensive oracle call each, in parallel) and append them to $\mathcal{D}^s$.

4.  Refit the GP and continue iterating.

**Why pure exploration here.** A UCB-style acquisition would bias the design toward states with high *value*, which is not what we want when the surrogate is a building block of an iteration. We want the GP posterior to be uniformly tight wherever the Bellman operator might be evaluated, so that the contraction modulus of the *approximate* operator stays close to $\beta$. This is structurally different from the optimization setting of Bayesian optimization, where exploitation is a feature.

**Empirical impact.** The companion notebook compares a same-budget fixed Latin-hypercube design with an active design inside the one-dimensional GP-VFI loop. Both designs use Bellman labels; the active design starts from a small initial set and adds states by maximizing the GP posterior standard deviation {eq}`eq-bal_vfi` subject to a spacing rule. At the same final number of labels, the active design lowers posterior uncertainty and achieves a comparable or smaller dense-grid Bellman residual ({numref}`fig-gp_vfi_active_1d`). The figure is one-dimensional by design: the goal is to show active enrichment inside a genuine Bellman iteration, not to use a separable interpolation toy as a proxy for multidimensional dynamic programming.

```{figure} figures/gp_vfi_active_learning_1d.png
:name: fig-gp_vfi_active_1d
:width: 100%

Same-budget active enrichment inside one-dimensional GP value-function iteration. The left panel compares the GP posterior means from a fixed Latin-hypercube design and an active design against a reference GP-VFI solution. The middle panel shows the posterior standard deviation and the states added by the active rule {eq}`eq-bal_vfi`, marked as triangles on the horizontal axis. The right panel reports the dense-grid Bellman residual. Unlike the previously used two-dimensional separable interpolation benchmark, every plotted training value here is generated by a Bellman maximization. Generated by notebook [`lecture_14_04_GP_Value_Function_Iteration.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_04_GP_Value_Function_Iteration.ipynb).
```

### Results: From One to 500 Dimensions

The companion notebook deliberately stops at the one-dimensional Bellman problem, where every numerical object can be inspected directly. The high-dimensional claims below come from the ASGP implementation of {cite:t}`SCHEIDEGGER201968`, where the GP operates on an active subspace rather than on the full tensor-product state space; we report their findings as a literature summary, not as something the notebook itself demonstrates.

{cite:t}`SCHEIDEGGER201968` apply the ASGP-VFI algorithm to the stochastic optimal growth model {eq}`eq-growth_bellman` across a range of dimensions, from $D = 1$ (the textbook Brock–Mirman case) to $D = 500$ continuous state variables. The key findings are:

1.  **Convergence is dimension-independent in the active subspace:** all models converge to a normalized error below $10^{-4}$ within approximately 70 VFI iterations regardless of $D$, where the iteration count is dimension-independent in the $m$-dimensional active subspace on which the GP actually operates rather than in the full ambient $\R^D$.

2.  **Low-dimensional structure:** the active subspace dimension is $m = 1$ in all cases tested, so the value function depends on a single linear combination of the $D$ capital stocks, confirming that the high-dimensional problem has intrinsic low-dimensional structure.

3.  **Sub-exponential scaling:** CPU time grows sub-exponentially with $D$ (concave on a log scale when parallelized across compute nodes); on the benchmarks reported in {cite:t}`SCHEIDEGGER201968`, $D = 500$ is solved at sub-exponential cost in $D$ when distributed across a compute cluster. This avoids the explicit grid-based curse but does not eliminate all high-dimensional sample/optimization costs.

4.  **Beyond the reach of grids:** grid-based methods such as Smolyak sparse grids cannot handle $D > 100$, making ASGP one demonstrated grid-free approach for models at this scale; neural residual methods such as DEQNs are complementary alternatives when the equilibrium conditions are easier to express directly.

### Uncertainty Quantification and Ergodic Sets

**Free UQ from the GP posterior.** Because the value function at each VFI iteration is a GP, the posterior variance $\sigma^2(\x)$ provides a built-in measure of approximation uncertainty. This gives pointwise credible bands for the interpolated value function essentially for free. Policy uncertainty is not automatic in the same sense: it must be propagated through the Bellman maximization, for example by evaluating policies under posterior draws or local approximations to the GP posterior.

**Parameter uncertainty propagation.** Uncertain model parameters can be treated as additional pseudo-states. For example, extending the state to $\tilde{S} = (\bm{k}, \gamma)$ where $\gamma \in [\underline{\gamma}, \bar{\gamma}]$ is the risk aversion parameter allows solving the model once on the extended state space and then extracting univariate effects and global sensitivity indices *in a single computation*, avoiding the traditional approach of re-solving the model for each parameter value. {cite:t}`harenberg2019uncertainty` give a general introduction to uncertainty quantification for economic models, covering univariate-effect plots and Sobol' sensitivity indices.

**Learning ergodic sets.** Many economic models have irregularly shaped ergodic sets (ellipsoids or manifolds, not hypercubes). {cite:t}`SCHEIDEGGER201968` propose learning the ergodic distribution via Bayesian Gaussian mixture models: (i) solve the model on a large initial domain; (ii) simulate the economy forward to generate capital paths $\{\bm{k}_t^+\}$; (iii) fit a mixture of Gaussians $\hat{\rho}_\mathrm{ergodic}$ to the simulated paths; (iv) re-solve on the ergodic set by sampling training points from $\hat{\rho}_\mathrm{ergodic}$ instead of the full hypercube. GPs handle this naturally because they require no grid structure.

### Comparison: GPs vs. DEQNs for Solving Dynamic Models

Both GPs (this chapter) and DEQNs ({ref}`ch-deqn`–{ref}`ch-olg`) solve dynamic stochastic models, but with different trade-offs ({numref}`tab-gp_vs_deqn_dynamic_models`):

````{table}
:name: tab-gp_vs_deqn_dynamic_models

GP/ASGP and DEQN solvers for dynamic models. The table distinguishes exact fixed-point theory from the numerical approximation actually used: GP-VFI inherits the Bellman contraction only up to interpolation error, while DEQNs are judged by residual and simulation diagnostics.

| **Criterion** | **GP / ASGP** | **DEQNs** |
|---|---|---|
| Solution method | Value function iteration | Euler equation residuals |
| Dimensionality | Up to $\sim$500 (with AS) | $>$1000 feasible |
| UQ built-in | Yes (GP posterior variance) | No (needs extra work) |
| Hardware | CPU clusters (MPI) | GPUs (TensorFlow/PyTorch) |
| Irregular domains | Yes (grid-free) | Yes (mesh-free) |
| Sensitivity analysis | Cheap after pseudo-state augmentation | Requires pseudo-states or re-training |
| Convergence diagnostics | Exact Bellman operator is a contraction; GP interpolation error must be controlled | Euler residuals and simulation tests; no generic global proof |
````

**When to use which:** GPs when the effective dimension is moderate ($D \lesssim 15$, or a few hundred only when active-subspace structure is strong) and uncertainty quantification or sensitivity analysis is required; DEQNs when $D$ is very large, GPU hardware is available, or the model involves complicated market-clearing conditions that are more naturally expressed as Euler equation residuals than as Bellman maximization.

```{prf:remark} Hands-on

 Notebook [`04_GP_Value_Function_Iteration.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_04_GP_Value_Function_Iteration.ipynb) implements GP-based VFI for the one-dimensional stochastic growth model. It shows convergence of the GP-VFI outer loop, posterior credible bands for the value-function interpolant, Cholesky-based LOO-RMSE {eq}`eq-gp_loo`, dense-grid Bellman residuals, active enrichment of the Bellman design {eq}`eq-bal_vfi`, policy recovery, and a deterministic full-depreciation verification ($\delta = 1$ and $\sigma = 0$) against the closed-form Brock–Mirman solution. The multidimensional ASGP extension is discussed in the literature summary above and illustrated separately by the active-subspace notebooks, [`05_Active_Subspace_2D.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_05_Active_Subspace_2D.ipynb), [`06_Active_Subspace_10D.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_06_Active_Subspace_10D.ipynb), and [`07_Active_Subspace_Nonlinear.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_07_Active_Subspace_Nonlinear.ipynb), on 2D, 10D, and nonlinear test functions.
```


(sec-dkl)=
## Deep Kernel Learning
Standard GP kernels such as the RBF or Matérn operate on *raw* input features: the covariance $k(\x, \x')$ depends on $\|\x - \x'\|$, which may be a poor measure of similarity when the inputs are high-dimensional or when the function's structure depends on complex nonlinear interactions among variables. *Deep Kernel Learning* (DKL) {cite:p}`wilson2016deep` addresses this limitation by combining the representation-learning power of deep neural networks with the probabilistic framework of GPs.

**The DKL architecture.** A DKL model consists of two components:

1.  A **feature extractor** $\phi(\x; \bm{\theta}_\mathrm{NN})\colon \R^D \to \R^d$, typically a multi-layer perceptron that maps the raw input $\x$ to a learned representation of (usually much lower) dimension $d$.

2.  A **GP layer** that places a GP prior on the function $g(\z) = g(\phi(\x; \bm{\theta}_\mathrm{NN}))$, with a standard base kernel $k_\mathrm{base}(\z, \z')$ (e.g., RBF or Matérn) operating in the learned feature space.

The *deep kernel* is thus:

$$
k_\mathrm{DKL}(\x, \x') = k_\mathrm{base}\bigl(\phi(\x; \bm{\theta}_\mathrm{NN}),\; \phi(\x'; \bm{\theta}_\mathrm{NN})\bigr).
$$ (eq-dkl_kernel)

Unlike a standard kernel that computes distance in the input space, the DKL kernel computes distance in a *learned* feature space. If the neural network learns to map functionally similar inputs close together, the GP can exploit this structure for better predictions and more calibrated uncertainty estimates.

**Joint training.** The parameters of both the neural network ($\bm{\theta}_\mathrm{NN}$) and the GP hyperparameters ($\ell, \sigma_f, \sigma_y$ of $k_\mathrm{base}$) are optimized jointly by maximizing the GP marginal likelihood:

$$
\max_{\bm{\theta}_\mathrm{NN},\, \bm{\vartheta}} \; \log p(\bm{y}\,|\,\Phi(\X; \bm{\theta}_\mathrm{NN}),\, \bm{\vartheta}),
$$

where $\Phi(\X; \bm{\theta}_\mathrm{NN})$ is the matrix of transformed features. This end-to-end training procedure automatically learns features that are useful for the GP, without requiring manual feature engineering. In practice, DKL is implemented via GPyTorch {cite:p}`gardner2018gpytorch`, which provides efficient GPU-accelerated GP inference and integrates seamlessly with PyTorch for the neural network component.

**When DKL helps.** DKL is most beneficial when:

- The input space is high-dimensional but the function has low-dimensional structure that is *nonlinearly* embedded (active subspaces find *linear* structure; DKL can find nonlinear features).

- Sufficient training data is available to train the feature extractor without overfitting (DKL has more parameters than a standard GP).

- Uncertainty quantification is important, but a standard GP kernel is insufficiently expressive to capture the true function's covariance structure.

The trade-off relative to a standard GP is clear: DKL offers greater expressiveness at the cost of more parameters and the risk of overfitting with very small datasets. For the moderate-data regimes typical of economic surrogate models ($N \sim 100$–$1000$), DKL can offer significant improvements over standard kernels, particularly when the target function has complex multi-scale behavior. {cite:t}`chen2025private` apply learned-feature-map GP architectures in a dynamic model of private asset allocation, where the feature map captures the complex nonlinear interactions between illiquidity, portfolio composition, and optimal rebalancing.

**Illustrative examples.** Two simple examples highlight when DKL provides a qualitative advantage over standard kernels:

- **Step functions.** Consider approximating a 1D function with a sharp discontinuity. A standard GP with an RBF kernel necessarily oversmooths near the jump and yields poorly calibrated uncertainty bands. A DKL model, by contrast, can learn a feature map that "compresses" the input near the discontinuity, effectively sharpening the GP's resolution where it matters most. This is directly relevant to economic applications with occasionally binding constraints, where policy functions exhibit kinks or jumps.

- **Anisotropic boundaries in 2D.** Standard stationary kernels (RBF, Matérn) are isotropic: they measure distance with circular level sets. When the target function has a discontinuity along a *diagonal* or curved boundary, common in portfolio problems with no-trade regions or in models with regime-dependent policies, the isotropic kernel cannot adapt. DKL learns a nonlinear coordinate transformation that aligns the kernel's smoothness assumptions with the function's actual structure, capturing diagonal and curved boundaries that would require an impractical number of training points with a standard kernel.

```{prf:remark} Hands-on

 The companion notebook [`08_Deep_Kernel_Learning.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_08_Deep_Kernel_Learning.ipynb) implements a simplified DKL pipeline (a supervised feature extractor stacked with a scikit-learn GP head) and compares the learned deep kernel against standard RBF and Matérn GPs on function approximation tasks; the full GPyTorch joint marginal-likelihood training of {eq}`eq-dkl_kernel` is left as an extension.
```


(sec-bayesian_dl_compare)=
## GPs Among Their Bayesian Cousins
Gaussian processes are the most analytically transparent way to attach uncertainty to a non-parametric regressor, but they are not the only one. For completeness, this section briefly situates the GP machinery against two neural-network-based alternatives that come up frequently in modern uncertainty-aware deep learning.

**Monte-Carlo dropout.** {cite:t}`gal2016dropout` show that a deep network trained with dropout and *evaluated* with dropout still active can be interpreted as approximate variational inference over a particular Bayesian neural network. Predictive uncertainty is obtained by averaging $T$ stochastic forward passes; the cost is one extra forward pass per sample. Calibration is rougher than a GP's, but the method requires no architectural change and scales to deep nets and large $n$.

**Deep ensembles.** {cite:t}`lakshminarayanan2017simple` train $E$ independent neural networks with different random seeds and combine them into a Gaussian-mixture predictor. Empirically this is one of the most robust uncertainty-quantification recipes available, often beating MC dropout and approaching the calibration of GPs, at $E$ times the training cost.

````{table}
:name: tab-gp_vs_bnn

Three uncertainty-quantification recipes compared on the dimensions that drive method choice in economic surrogate work. This script uses GPs in {ref}`ch-gp` and {ref}`ch-climate` because the typical regime ($n \lesssim 10^3$, expensive simulator) plays to their strengths; readers operating in larger-data regimes should consider MC dropout or deep ensembles before resorting to a sparse GP.

|  | **Gaussian process** | **MC dropout** | **Deep ensembles** |
|---|---|---|---|
| Calibration | exact under model | approximate | strong empirically |
| Training cost | $\mathcal{O}(n^3)$ | one network | $E\times$ one network |
| Inference cost | $\mathcal{O}(n^2)$ | $T$ forward passes | $E$ forward passes |
| Sample efficiency | best at small $n$ | needs much more data | needs much more data |
| Best when | $n \lesssim 10^4$, low $d$ | cheap UQ on existing nets | willing to pay $E\times$ for top calibration |
| Reference | {cite:t}`Rasmussen:2005:GPM:1162254` | {cite:t}`gal2016dropout` | {cite:t}`lakshminarayanan2017simple` |
````

**A decision rule for practice.** In our experience the right method follows from the application: *(i)* plain GPs for moderate $d$ ($\lesssim 10$–$20$) with a smooth target and an expensive simulator, when calibrated uncertainty is the goal; *(ii)* deep kernels (Wilson & al., {cite:year}`wilson2016deep`) when the input geometry is non-trivial (regime switches, manifold structure, image-like inputs); *(iii)* deep ensembles or MC dropout for high-$d$ regression where calibrated uncertainty is desirable but exact GP inference is infeasible; *(iv)* sparse GPs ({cite:t}`titsias2009variational` {cite}`hensman2013gaussian`) for $n \gtrsim 10^4$ when the target stays smooth. The summary frame in the companion deck ("Toolbox: When to Use What") gives the same decomposition visually.

```{prf:remark} Chapter Summary

- Gaussian processes attach calibrated uncertainty to non-parametric regression at $\mathcal{O}(n^3)$ cost; the marginal likelihood implements an automatic Occam's razor that chooses model complexity without held-out validation.

- Active subspaces collapse $d$-dimensional inputs to $m \ll d$ via the gradient outer product; this is the trick that makes GPs viable past $d \sim 10$.

- Bayesian active learning closes the surrogate loop: train, evaluate uncertainty, request next design point where it's largest, repeat. {cite:t}`SCHEIDEGGER201968` provide one canonical example.

- Deep kernel learning composes a NN feature extractor with a GP head; an alternative to plain GPs and to deep ensembles.
```


## Further Reading {.unnumbered}
- {cite:t}`Rasmussen:2005:GPM:1162254`, the standard GP textbook.

- {cite:t}`constantine2015active`, the active-subspaces monograph.

- {cite:t}`rennerscheidegger_2018` {cite}`SCHEIDEGGER201968`, GP+BAL methodology and applications in economics.

- {cite:t}`wilson2016deep`, deep kernel learning.

- {cite:t}`titsias2009variational` {cite}`hensman2013gaussian`, sparse-GP scaling.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch9-1

**[Core\] Posterior on three points.** Fit a GP with RBF kernel ($\ell=1$, $\sigma_f=1$, $\sigma_y=0.1$) to three points $(0,0), (1,0.8), (2,0.3)$. Compute the posterior mean and variance at $x^\star = 1.5$ in closed form.
```

```{exercise}
:label: ex-ch9-2

**[Core\] Marginal likelihood Occam.** For the same three points, plot the log marginal likelihood as a function of the length scale $\ell \in [0.1, 5]$. Identify the optimum and explain it in terms of the data-fit / complexity decomposition.
```

```{exercise}
:label: ex-ch9-3

**[Core\] Active subspace by hand.** For $f(\x) = (x_1 + x_2 + x_3)^2 + 0.01(x_1 - x_2)^2$ on $[-1,1]^3$, compute the gradient outer product matrix $\hat C$ on a uniform sample. Show that the leading eigenvector identifies the "aggregate" direction $(1,1,1)/\sqrt 3$.
```

```{exercise}
:label: ex-ch9-4

**[Computational\] Deep vs. linear active subspace on a radial ridge.** Take the target $y(\xi) = \exp\!\bigl(-[(w_1^\top\xi)^2 + (w_2^\top\xi)^2]\bigr)$ with $\xi \sim \mathcal{N}(0, I_{20})$ and $w_1, w_2 \in \R^{20}$ fixed orthonormal. (i) Compute $\hat C$ from $4\,000$ samples and confirm that it has two nonzero eigenvalues. (ii) Train the Tripathy–Bilionis surrogate {eq}`eq-deep_as_ansatz` with widths {eq}`eq-tb_widths`, Swish activation {eq}`eq-swish`, and elastic-net loss {eq}`eq-tb_loss` for $d \in \{1,2,3,4\}$; plot held-out MSE on a log scale and identify the elbow. (iii) Explain why linear AS needs $d \ge 2$ while deep AS already captures the response at $d = 1$ (cf. notebook `09_Deep_Active_Subspace_Ridge`).
```

```{exercise}
:label: ex-ch9-5

**[Computational\] BAL on a 2D function.** Modify notebook `02_GP_and_BAL` so that one acquisition uses pure variance, $U_{\mathrm{var}}(\x)=\sigma^2(\x)$, and another uses the mixed log-variance score $U_{\mathrm{mix}}(\x)=w_{\mathrm{obj}}\mu(\x)+\tfrac{w_{\mathrm{var}}}{2}\log\sigma^2(\x)$ with $w_{\mathrm{obj}}>0$. Compare the resulting designs and comment on which gives smoother domain coverage. Explain why pure $\sigma^2(\x)$ and pure $\log\sigma^2(\x)$ have identical maximizers.
```

```{exercise}
:label: ex-ch9-6

**[Computational\] Sobol sensitivity with a GP surrogate.** Write a short script, or extend notebook [`02_GP_and_BAL.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_02_GP_and_BAL.ipynb), to train a GP surrogate on the $4$-dimensional Genz product-peak function $f(\bm x) = \prod_{i=1}^{4}(c_i^{-2} + (x_i - w_i)^2)^{-1}$ on $[0,1]^4$ with $c = (1, 2, 0.5, 1.5)$, $w = (0.4, 0.6, 0.3, 0.7)$, using $N \in \{50, 100, 200\}$ training points. For each $N$, compute the Sobol first-order and total-effect indices in two ways: (i) direct on the true $f$ via $10{,}000$ Monte Carlo samples (reference), (ii) on the GP surrogate via $10^6$ samples (cheap thanks to the surrogate). Plot the relative error in each Sobol index against $N$. Verify that the surrogate-based Sobol estimates converge to the reference as $N$ grows, and identify the $N$ at which all four first-order indices match the reference within $5\%$. Discuss why surrogate-based sensitivity analysis is the workhorse for expensive simulators (climate models, structural macro models) where direct $10^6$-sample Monte Carlo is infeasible.
```

```{exercise}
:label: ex-ch9-7

**[Core\] Prior-driven RBF-GP extrapolation outside the training domain.** Train a GP with RBF kernel on $20$ points sampled uniformly from $[0, 1]$ from the function $f(x) = \sin(2\pi x)\,e^{-x}$. (i) Plot the posterior mean and $\pm 2$ standard deviation band on the training interval $[0,1]$; verify the posterior tracks the true function and the band is narrow. (ii) Now extrapolate to $x \in [1.5, 3.0]$ and plot the posterior mean and band over the extended interval. Show analytically that for an RBF kernel with length scale $\ell$, the posterior at $x$ far from any training point ($|x - x_i| \gg \ell$ for all $i$) reverts to the prior: posterior mean $\to 0$, posterior variance $\to \sigma_f^2$. (iii) Verify numerically: at $x = 3$, the posterior mean is essentially zero (independent of the training data), and the posterior standard deviation has expanded back to the prior $\sigma_f$. (iv) Discuss the implication: far from the training domain the posterior reverts to the prior, so the variance band there reflects the learned hyperparameters rather than the data, and the band is overconfident only when the prior variance or the learned length scale is itself misleading. Naive Bayesian active learning that relies on posterior variance may therefore fail to acquire informative samples outside the convex hull when the prior variance is no larger than the inside-hull noise scale. Mitigations: use a Matérn-$\nu$ kernel with smaller $\nu$ (heavier-tailed), bound the analysis to the convex hull of the training data, or use a boundary-aware acquisition function.
```

[^1]: <https://dp.quantecon.org/>
