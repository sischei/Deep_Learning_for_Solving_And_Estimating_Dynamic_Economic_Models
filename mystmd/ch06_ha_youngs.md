---
title: "Heterogeneous Agents and Young's Method"
label: ch-young
---

The OLG models of {ref}`ch-olg` featured a finite number of agent types, so the cross-sectional state was simply a vector $(k^1,\ldots,k^A)$. Many important macroeconomic applications instead require a *continuum* of agents subject to idiosyncratic shocks. In the {cite:t}`krusell1998income` economy, an aggregate productivity shock additionally moves the cross section in a stochastic way, and the entire wealth distribution $\mu_t$ then becomes part of the aggregate state. The {cite:t}`aiyagari1994uninsured` model is the special case without aggregate risk: $\mu_t$ is fixed in stationary equilibrium and evolves deterministically along transitions, so it is a parameter of the equilibrium rather than a stochastic state variable. Incomplete markets prevent full insurance in both, making explicit distributional tracking essential, but the "master-equation" challenge of treating $\mu_t$ as a high-dimensional aggregate state arises only once aggregate risk is added on top of Aiyagari. Why represent $\mu_t$ as a histogram on a discrete grid rather than as a *Monte Carlo panel* (a finite simulated sample of $N$ agents whose empirical cross-section stands in for $\mu_t$)? Two reasons motivate the choice up front: a histogram is *deterministic*, so re-running the same equilibrium gives identical aggregates and the loss is a smooth function of the network weights, and it is *noise-free*, so Euler-equation residuals reflect approximation error rather than $\mathcal{O}(1/\sqrt N)$ Monte Carlo sampling noise ({numref}`fig-young_vs_mc` makes this contrast quantitative). This chapter develops Young's non-stochastic simulation method {cite:p}`young2010` for representing $\mu_t$ as a histogram and shows how to embed that method within the DEQN framework of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` to solve heterogeneous-agent economies with neural network policy functions.

**How this chapter maps onto the slides and notebooks.** The heterogeneous-agent material of this chapter, together with the companion deck in `lecture_09_heterogeneous_agents_youngs_method/slides/`, can be read independently of the sequence-space material in {ref}`sec-sequence_space`; readers already comfortable with Krusell–Smith may skip straight there on a first pass. Two notebooks in `lecture_09_heterogeneous_agents_youngs_method/code/` accompany {ref}`sec-young_method`–{ref}`sec-young_deqn`: [`lecture_09_10_Youngs_Method_Examples.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_10_Youngs_Method_Examples.ipynb) isolates Young's redistribution operator on toy examples, and [`lecture_09_11_Continuum_of_Agents_DEQN.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_11_Continuum_of_Agents_DEQN.ipynb) embeds the same operator inside the Appendix A.5 endowment-economy DEQN. {ref}`sec-ks_alternatives` on alternative deep-learning approaches is paired with [`lecture_09_12_KrusellSmith_DeepLearning.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_12_KrusellSmith_DeepLearning.ipynb), a classroom-scale all-in-one DL solver in the spirit of {cite:t}`maliar2021deep`.

**The Bewley–Huggett–Aiyagari lineage.** The continuum-agent framework that this chapter operationalizes has three foundational sources. {cite:t}`bewley1986stationary` introduced stationary monetary equilibrium with a continuum of agents subject to iid endowment shocks; the explicit self-insurance-through-borrowing-constraints mechanism that defines the modern incomplete-markets workhorse is due to {cite:t}`imrohoroglu1989cost`, {cite:t}`huggett1993riskfree`, and {cite:t}`aiyagari1994uninsured`. {cite:t}`huggett1993riskfree` cast the idea as a tractable endowment economy with a single risk-free asset, focusing on the equilibrium interest rate. {cite:t}`aiyagari1994uninsured` added neoclassical production, closing the model in general equilibrium with capital accumulation; this is the canonical incomplete-markets economy. {cite:t}`krusell1998income` layered aggregate productivity shocks on top, producing the modern Krusell–Smith economy that this chapter targets, and its continuous-time reformulation is developed by {cite:t}`achdou2022income` and revisited in {ref}`ch-ct_theory`.

```{figure} figures/fig-bewley_lineage.svg
:name: fig-bewley_lineage

Genealogy of the heterogeneous-agent models treated in this script. This chapter targets the Krusell–Smith branch (right) by combining a DEQN policy with Young's histogram update; the continuous-time branch (Achdou–Han–Lasry–Lions–Moll) is taken up in {ref}`ch-ct_theory`.
```

(sec-young_motivation)=
## From Representative to Heterogeneous Agents
In the representative-agent models of {ref}`ch-deqn`–{ref}`ch-irbc`, aggregate capital $K_t$ is a sufficient statistic for the economy's state. In reality, agents differ in wealth, income, and employment status, and incomplete markets prevent full insurance against idiosyncratic risk. This heterogeneity matters for policy analysis:

- Fiscal stimulus: agents near the borrowing constraint have high marginal propensities to consume; wealthy agents save most of a windfall.

- Monetary policy affects borrowers and savers differently.

- The shape of the wealth distribution feeds back into aggregate demand and equilibrium prices.

The mathematical challenge depends on whether the economy carries aggregate risk. Without it (the Aiyagari case), the cross-section $\mu_t$ is fixed in stationary equilibrium and evolves deterministically along transitions, so it parameterizes the equilibrium rather than serving as a stochastic state variable. With aggregate risk (the Krusell–Smith case that this chapter targets), an aggregate shock moves the cross-section in a stochastic way, and the entire distribution $\mu_t$ then becomes part of the aggregate state. Since $\mu_t$ is in either case a measure (an infinite-dimensional object), it cannot be placed on a grid without encountering the curse of dimensionality.

**Two approaches.** There are two main strategies for handling heterogeneous agents in discrete time. *Finite-agent methods* keep a high-dimensional but finite vector $(k_1,\ldots,k_N)$ of individual states, which is exact in $\mu_t$ but pays a permutation-symmetry cost that scales poorly with $N$ (the OLG models of {ref}`ch-olg`, and the finite-agent Monte-Carlo *implementation* of the continuum-agent Krusell–Smith model by {cite:t}`maliarmaliarvalli2010`, fall in this category). *Continuum-agent methods* replace the agent labels with the distributional state $\mu_t(\cdot)$, which is infinite-dimensional but eliminates sampling noise and the permutation problem, at the price of needing to approximate $\mu_t$ ({cite:t}`krusell1998income` and {cite:t}`young2010` are the canonical references). This chapter pursues the second approach: tracking the distribution via a histogram, with Young's histogram method {cite:p}`young2010` providing the distribution update.

(sec-ks_economy)=
## The Krusell–Smith (1998) Economy
The canonical heterogeneous-agent model with aggregate uncertainty is due to {cite:t}`krusell1998income`. Its key features are:

- A *continuum* of ex ante identical, infinitely-lived agents.

- *Incomplete markets:* no insurance against idiosyncratic labor income risk.

- *Aggregate uncertainty:* productivity shocks that affect all agents.

- A single asset (capital/bonds) with a borrowing constraint.

Each agent must forecast future prices to make optimal savings decisions. But prices depend on the entire wealth distribution, which is infinite-dimensional. The key insight of {cite:t}`krusell1998income` is that the *mean* of the wealth distribution is a nearly sufficient statistic for forecasting: a log-linear forecasting rule

$$
\hat{H}(m_1, a) = \exp\!\bigl(A(a) + B(a)\log m_1\bigr),
$$ (eq-ks_forecast)

where $\hat{H}(m_1,a)$ is the approximate next-period aggregate capital (the price-forecasting function), $A(a)$ and $B(a)$ are the OLS intercept and slope coefficients (each a function of the aggregate shock state $a$), and $m_1 = \int k\,d\mu$ is mean capital; this rule achieves $R^2 > 0.9999$ in practice.

```{prf:remark} Approximate aggregation: what it means and what it does *not* mean

 The empirical observation that a single moment $m_1 = \int k\,d\mu$ is sufficient for price forecasting is known as *approximate aggregation*. Conceptually, it says that the equilibrium price mapping $r_{t+1}, w_{t+1}$ depends on the cross-sectional distribution *almost only through its mean*, so the high-dimensional state $\mu_t$ collapses to a one-dimensional summary for forecasting purposes.

Importantly, {cite:t}`krusell1998income` did not restrict themselves to the one-moment specification. Their original paper explicitly reports results for two- and three-moment forecasting rules (adding the variance, and further adding the skewness of the wealth distribution) and finds that additional moments provide only marginal improvement: $R^2$ barely changes. The one-moment benchmark is therefore an *empirical* finding, not a modeling assumption.

Two cautions are essential. First, this is a property of the *Krusell–Smith model class* (one risky asset, modest wealth dispersion, moderate income risk) and is not a general theorem; in models with multiple assets, large MPC heterogeneity, or occasionally binding constraints that bite on a non-trivial mass of agents, the forecasting rule typically requires higher moments or a more flexible function approximator ({ref}`ch-gp`, {ref}`ch-climate`). Second, $R^2 > 0.9999$ on simulated mean dynamics does *not* imply that the distribution itself is well summarized by its mean: tails, percentiles, and the borrowing-constrained mass continue to matter for welfare and policy analysis. Approximate aggregation is therefore best read as a *computational* result, and {ref}`sec-young_method`–{ref}`sec-young_deqn` accordingly track the full histogram and use the mean only as one of the network's input features rather than as a state-space substitute.
```


**Equilibrium.** A recursive competitive equilibrium consists of:

1.  Individual optimization: each agent chooses savings $k'$ to maximize utility given the forecasting rule for aggregate prices (equivalently, for next-period aggregate capital $K'$).

2.  Market clearing: aggregate demand equals supply.

3.  Consistency between individual policies and the law of motion for the distribution: $\mu_{t+1} = T(\mu_t, a_t)$ given the optimal savings policies just derived.

4.  Rational expectations: agents' forecasts are self-confirming.

The question is: how do we simulate the distribution forward to compute the realized mean and check the forecasting rule?

**The traditional Krusell–Smith algorithm in detail.** Before turning to the distribution-update step, it is useful to write the canonical KS algorithm explicitly. It is a nested fixed-point iteration with an *outer* loop over forecasting-rule coefficients and an *inner* loop that solves the individual household's Bellman equation given those coefficients.

```{prf:definition} Krusell–Smith (1998), Traditional Algorithm

- **Input:** Initial forecast coefficients $(A^{(0)}(a), B^{(0)}(a))$ from Eq.~{eq}`eq-ks_forecast`; capital grid $\mathcal{K}$; tolerances $\varepsilon_{\mathrm{inner}},\varepsilon_{\mathrm{outer}}$; simulation length $T_{\mathrm{sim}}$.
- *Key notation:* $a$: aggregate TFP shock; $\varepsilon$: idiosyncratic employment shock; $m_1 = \int k\,d\mu$: mean capital; $\beta$: household discount factor; $u(c)$: period utility; $c = w(\hat H)\,y(\varepsilon) + (1+r(\hat H))\,k - k'$: consumption (wages $w$ and return $r$ come from the forecasting rule $\hat H$); $k'$: end-of-period savings (the policy choice); $\varepsilon', a'$: next-period shocks; $R^2$: OLS coefficient of determination.
- **for** outer iteration $\ell = 0, 1, 2, \ldots$ until $\|A^{(\ell+1)}-A^{(\ell)}\| + \|B^{(\ell+1)}-B^{(\ell)}\| < \varepsilon_{\mathrm{outer}}$ **do**
  - **(a) Solve individual Bellman equation** given $\hat H(m_1, a) = \exp\!\bigl(A^{(\ell)}(a) + B^{(\ell)}(a)\log m_1\bigr)$.
  - \quad Iterate $V^{(n+1)}(k, \varepsilon, m_1, a) = u(c) + \beta\, \E{V^{(n)}(k', \varepsilon', \hat H(m_1,a), a')}$ until $\|V^{(n+1)}-V^{(n)}\| < \varepsilon_{\mathrm{inner}}$.
  - \quad Return policy $k'(k, \varepsilon, m_1, a)$.
  - **(b) Simulate the cross-section forward** for $T_{\mathrm{sim}}$ periods with (say) $10{,}000$ agents, starting from an initial distribution and drawing idiosyncratic shocks independently.
  - **(c) Update forecasting rule** by OLS on the simulated path: regress $\log m_1^{t+1}$ on $\log m_1^t$ within each aggregate-shock state to obtain $(A^{(\ell+1)}(a), B^{(\ell+1)}(a))$.
  - **(d) Convergence check:** retain the rule if $R^2 \geq 0.9999$ and the coefficient update $\|A^{(\ell+1)}-A^{(\ell)}\| + \|B^{(\ell+1)}-B^{(\ell)}\|$ is below $\varepsilon_{\mathrm{outer}}$.
- **end**
```


The remarkable empirical finding of {cite:t}`krusell1998income` is that a log-linear rule in $m_1$ alone achieves $R^2 > 0.9999$ and very small forecast errors: the "approximate aggregation" result. Textbook implementations (e.g., the `econ-ark/KrusellSmith` REMARK) typically converge in $\sim 20$ outer iterations with standard parameters $\beta = 0.99$, $\alpha = 0.36$, log utility, aggregate shocks on $\{\text{good},\text{bad}\}$ with persistence $\approx 0.875$, and unemployment rates $4\%$ (good) vs. $10\%$ (bad).

**Bottleneck.** The inner Bellman solve on a two-dimensional $(k, m_1)$ grid is the expensive step. It scales exponentially if one tries to add additional moments (e.g., tracking variance or skewness), which is why the KS algorithm as stated caps out at one moment. Extensions that need more moments must use more powerful function approximators. This is where the Young-histogram DEQN of {ref}`sec-young_deqn` (and the alternatives in {ref}`sec-ks_alternatives`) come in.

(sec-young_method)=
## Young's (2010) Non-Stochastic Simulation
{cite:t}`young2010` proposed a deterministic method for propagating the wealth distribution that avoids Monte Carlo sampling entirely.

**Core idea.** Represent the wealth distribution at time $t$ as a *histogram* over discrete bins, using two ingredients:

- a fixed capital grid $\{k_1, k_2, \ldots, k_{N_k}\}$, where $N_k$ is the number of bins and $i \in \{1,\ldots,N_k\}$ indexes individual grid points $k_i$;

- a finite set of idiosyncratic-shock states $\{\varepsilon_1,\ldots,\varepsilon_{N_\varepsilon}\}$, indexed by $j \in \{1,\ldots,N_\varepsilon\}$.

The histogram value $G_t(k_i, \varepsilon_j) \in [0,1]$ is the probability mass of agents sitting at capital bin $k_i$ in shock state $\varepsilon_j$ at time $t$; by construction, $\sum_{i,j} G_t(k_i, \varepsilon_j) = 1$.

Given the household policy function $k' = g(k, \varepsilon, m_1, a)$, where:

- $k$ is the individual's current capital and $\varepsilon$ their current idiosyncratic shock;

- $m_1 = \int k\,d\mu$ is aggregate mean capital (the summary statistic from Krusell–Smith);

- $a$ is the aggregate productivity shock (the same $a$ that drives prices in the KS setup above);

the key operation is *mass redistribution*. Evaluating $g$ at a bin $(k_i,\varepsilon_j)$ produces the savings target $k' = g(k_i, \varepsilon_j, m_1, a)$, which generically lies *off* the grid. Let $n = n(i,j) \in \{1,\ldots,N_k-1\}$ denote the bracketing index defined by $k_n \leq k' < k_{n+1}$, and let

```{math}
:enumerated: false

m \equiv G_t(k_i,\varepsilon_j)
```

be the probability mass currently sitting at bin $(k_i,\varepsilon_j)$. This mass $m$ is then split between the two bracketing grid points $k_n$ and $k_{n+1}$ using linear-interpolation weights:

$$
\omega = 1 - \frac{k' - k_n}{k_{n+1} - k_n}, \qquad
\text{mass } \omega\cdot m \text{ to } k_n, \quad (1-\omega)\cdot m \text{ to } k_{n+1}.
$$ (eq-young_weights)

{numref}`fig-young_interp` illustrates this operation. A mass $m$ sitting at the off-grid savings target $k'$ is split between the two bracketing grid points $k_n$ and $k_{n+1}$, with the weight $\omega$ proportional to the proximity of $k'$ to $k_n$ (so $\omega \to 1$ when $k' \to k_n$ and $\omega \to 0$ when $k' \to k_{n+1}$).

```{figure} figures/fig-young_interp.svg
:name: fig-young_interp

Linear interpolation in Young's method. Mass $m$ at off-grid point $k'$ is redistributed to the two bracketing grid points $k_n$ and $k_{n+1}$ using weights $\omega$ and $1-\omega$. Closer proximity to a grid point yields a larger share of the mass.
```

**Why exactly this weight?** The lottery weight $\omega$ in Eq. {eq}`eq-young_weights` is not an arbitrary choice: it is the *unique* value for which the two-point split has conditional mean equal to the off-grid policy choice $k'$. Solving the equation $\omega\cdot k_n + (1-\omega)\cdot k_{n+1} = k'$ for $\omega$ recovers Eq. {eq}`eq-young_weights` in one line:

$$
\omega\cdot k_n + (1-\omega)\cdot k_{n+1} = k_n + (1-\omega)(k_{n+1} - k_n) = k_n + (k' - k_n) = k'.
$$ (eq-young_meanpreserve)

By linearity of expectation, this conditional mean equality at every grid bin extends to the full distribution: the unconditional mean of $G_{t+1}$ equals the unconditional mean of the policy-implied next-period capital. Higher moments (variance, percentiles) are only approximated; the leading error is of order $(\Delta k)^2$ on smooth densities, so a finer grid improves higher-moment fidelity at no cost to mean preservation.

```{prf:remark} Lottery interpretation: what makes this "non-stochastic"

 Equation {eq}`eq-young_weights` can equivalently be read as a *fair lottery*: every agent at $(k, \varepsilon)$ whose policy choice lands at off-grid $k'$ is reassigned to $k_n$ with probability $\omega$ and to $k_{n+1}$ with probability $1-\omega$. In a Monte Carlo panel of size $N$, each agent draws this lottery once and the empirical histogram converges to its expectation at rate $\mathcal{O}(N^{-1/2})$. Young's method instead computes that expectation in closed form, which is equivalent to "running" an infinite number of agents through the lottery and integrating out the result. This is why the procedure is called non-stochastic: the lottery is still there, but its outcome is computed analytically rather than sampled. The same logic, applied to the discrete shock transitions $\pi_{\varepsilon'\mid\varepsilon}$, sends each piece of mass into all reachable next-period $\varepsilon'$ in proportion to $\pi$ rather than drawing one realization.
```


**Worked example.** We illustrate the full histogram update with a small grid of $N_k=4$ capital levels $\{1.0,\, 2.0,\, 3.0,\, 4.0\}$ and two idiosyncratic states $\varepsilon \in \{\text{low},\, \text{high}\}$.

**Step 1, Setup.** The initial histogram $G_0$ (masses summing to 1):

|  | $k=1.0$ | $k=2.0$ | $k=3.0$ | $k=4.0$ | Row sum |
|---|:---:|:---:|:---:|:---:|:---:|
| $\varepsilon = \text{low}$ | 0.10 | 0.20 | 0.10 | 0.05 | 0.45 |
| $\varepsilon = \text{high}$ | 0.05 | 0.15 | 0.20 | 0.15 | 0.55 |

The mean capital is $\bar{k}_0 = \sum_{i,j} G_0(k_i,\varepsilon_j)\cdot k_i = 0.10(1) + 0.20(2) + 0.10(3) + 0.05(4) + 0.05(1) + 0.15(2) + 0.20(3) + 0.15(4) = 2.55$.

**Step 2, Policy evaluation.** Let $y(\varepsilon)$ denote the dollar productivity associated with shock state $\varepsilon$, with $y_{\text{low}}=1$ and $y_{\text{high}}=3$ (here $\varepsilon$ is the *state index*, $y$ is its numerical value). Using a simple linear savings rule $k' = 0.4 k + 0.5\,y(\varepsilon)$:

|  | $k=1.0$ | $k=2.0$ | $k=3.0$ | $k=4.0$ |
|---|:---:|:---:|:---:|:---:|
| $\varepsilon = \text{low}$: $k'$ | 0.9 | 1.3 | 1.7 | 2.1 |
| $\varepsilon = \text{high}$: $k'$ | 1.9 | 2.3 | 2.7 | 3.1 |

**Step 3, Interpolation weights.** Since the grid is equispaced with $\Delta k = 1.0$, each off-grid $k'$ is bracketed by $[k_n, k_{n+1}]$ with weight $\omega = 1 - (k' - k_n)/\Delta k$ (it is the equispacing, not the unit value, that makes this expression linear in the bracket index):

|  | $k=1.0$ | $k=2.0$ | $k=3.0$ | $k=4.0$ |
|---|:---:|:---:|:---:|:---:|
| $\varepsilon = \text{low}$: bracket, $\omega$ | $[1,1]$, clip | $[1,2]$, 0.7 | $[1,2]$, 0.3 | $[2,3]$, 0.9 |
| $\varepsilon = \text{high}$: bracket, $\omega$ | $[1,2]$, 0.1 | $[2,3]$, 0.7 | $[2,3]$, 0.3 | $[3,4]$, 0.9 |

When $k' < k_1 = 1.0$ (here $k' = 0.9$ for the low-state, lowest-capital agents), all mass is assigned to $k_1$ (clipped to the boundary).

**Step 4, Mass redistribution.** For simplicity, assume the shock transition is the identity ($\varepsilon' = \varepsilon$), so mass stays in its current $\varepsilon$-state. Building $G_1$ by accumulating the redistributed mass from each source bin (mass is conserved bin-by-bin: e.g. the $\varepsilon=\text{low}$, $k=2$ source of mass $0.20$ contributes $0.7\cdot0.20 = 0.14$ to $k=1$ and $0.3\cdot0.20 = 0.06$ to $k=2$):

|  | $k=1.0$ | $k=2.0$ | $k=3.0$ | $k=4.0$ | Row sum |
|---|:---:|:---:|:---:|:---:|:---:|
| $\varepsilon = \text{low}$ | 0.270 | 0.175 | 0.005 | 0.000 | 0.450 |
| $\varepsilon = \text{high}$ | 0.005 | 0.210 | 0.320 | 0.015 | 0.550 |
| total | 0.275 | 0.385 | 0.325 | 0.015 | 1.000 |

**Step 5, Mean verification.** The mean of $G_1$ is $\bar{k}_1 = 0.275(1)+0.385(2)+0.325(3)+0.015(4) = 2.08$. The unclipped, policy-implied mean is $\sum_{i,j} G_0(k_i,\varepsilon_j)\,k'(k_i,\varepsilon_j) = 2.07$, so boundary clipping at $k' = 0.9$ raises the mean by only $0.01$. Mean preservation is exact for source bins whose policy choice $k'$ lies strictly inside the grid; clipping at the boundary slightly biases the mean, here upward, because mass that would have landed at $k'=0.9$ is forced to $k=1$. In general, boundary clipping biases the mean in the *direction of the violated boundary*: clipping at $k_{\min}$ biases the mean upward (mass is pulled in from below the grid), clipping at $k_{\max}$ biases it downward. With a wider grid ($k_{\min} < 0.9$) the mean would be preserved exactly.

```{prf:remark} Practical takeaway

 This small example shows why the grid must extend beyond the range of the policy function. In production code, a safeguard checks that boundary bins contain negligible mass (typically $<10^{-6}$).
```


**The full picture: one cell splits into four.** The worked example above used identity shock transitions to keep the arithmetic visible. The general case combines the capital lottery with a *shock fork*: each piece of mass split between $k_J$ and $k_{J+1}$ is then split again across the reachable next-period $\varepsilon'$ values according to $\pi_{\varepsilon'\mid\varepsilon, a}$. With two shocks $\varepsilon \in \{L, H\}$ this produces *four* destination bins for every source bin, with weights given by the product $\{\omega, 1-\omega\} \times \{\pi_{\varepsilon L}, \pi_{\varepsilon H}\}$. {numref}`fig-young_cascade` reproduces this two-stage cascade (essentially Fig. 1 of {cite:t}`young2010`), annotated with a concrete numerical case.

```{figure} figures/fig-young_cascade.svg
:name: fig-young_cascade

Verification: $0.027 + 0.003 + 0.018 + 0.002 = 0.05 = m$ $✓$.The conditional mean of next-period capital is $\omega k_J + (1-\omega) k_{J+1} = k'$, by Eq. {eq}`eq-young_meanpreserve`.

Young's cascade for one source bin (essentially Fig. 1 of {cite:t}`young2010`). Mass $m$ at $(k, \varepsilon{=}L)$ flows in two stages: the capital lottery ($\omega$ vs. $1-\omega$) sends it to the bracketing grid points $k_J$ and $k_{J+1}$, and the shock fork ($\pi_{LL}$ vs. $\pi_{LH}$) splits each piece across the reachable next-period idiosyncratic states. Each of the four leaves receives the product of its stage-1 and stage-2 weights times the source mass; the four leaf masses sum back to $m$. Repeating the cascade for every active source bin and accumulating the leaves yields the new histogram $G_{t+1}$.
```

A reader implementing the method should recognize three properties from the figure: (i) mass is conserved bin-by-bin because $\omega + (1-\omega) = 1$ and $\sum_{\varepsilon'} \pi_{\varepsilon\varepsilon'} = 1$; (ii) the capital lottery's expected next-period $k$ equals the policy choice $k'$ by Eq. {eq}`eq-young_meanpreserve`; (iii) the entire update is *linear* in $G_t$: $G_{t+1} = T(\rho)\,G_t$, where the transition operator $T$ depends on the current policy. Linearity is what enables differentiation through the histogram step; separately, $T$ is *sparse* (each column has at most $2\,|\mathcal{E}|$ non-zeros from the bracket lottery times the shock transitions), which is what makes the matrix-vector product cheap to evaluate in practice. This linearity is what makes the histogram update differentiable in the policy values almost everywhere, conditional on the interpolation brackets, when we embed it inside a neural-network training loop in {ref}`sec-young_deqn`. Index changes at bin boundaries and clipping at domain edges are nondifferentiable; standard implementations either ignore these measure-zero events, smooth the assignment, or stop gradients through the index-selection step, and in practice none of these choices materially affects training in the calibrations covered in this chapter.

**The histogram update algorithm.** At each time step, the histogram is updated deterministically:

```{prf:algorithm} Young's histogram update at time $t$
:label: alg-young

- **Input:** Current histogram $G_t$, policy $g(\cdot)$, transition matrix $\pi_{\varepsilon'|\varepsilon,a}$, grid $\{k_j\}$
- Initialize $G_{t+1}(k_j, \varepsilon') = 0$ for all $(j, \varepsilon')$
- **for** each $(k_i, \varepsilon_j)$ with $G_t(k_i, \varepsilon_j) > 0$ **do**
  - Compute optimal savings: $k' = g(k_i, \varepsilon_j, m_1, a_t)$
  - Find bracketing grid points: $k' \in [k_J, k_{J+1}]$
  - Compute weight: $\omega = 1 - (k' - k_J)/(k_{J+1} - k_J)$
  - **for** each next-period shock $\varepsilon'$ **do**
    - $G_{t+1}(k_J, \varepsilon') \mathrel{+}= \omega \cdot \pi_{\varepsilon'|\varepsilon_j, a_t}\cdot G_t(k_i, \varepsilon_j)$
    - $G_{t+1}(k_{J+1}, \varepsilon') \mathrel{+}= (1-\omega) \cdot \pi_{\varepsilon'|\varepsilon_j, a_t}\cdot G_t(k_i, \varepsilon_j)$
  - **end**
- **end**
- **return** Updated histogram $G_{t+1}$
```

**Implementation cheatsheet.** The pseudocode above translates into a few lines of `NumPy`; the inner double loop can also be vectorised with `np.add.at` for production use.

```python
import numpy as np

def young_step(G, kp, pi_eps, k_grid):
    """One Young (2010) histogram update on a uniform k-grid.
    G[i,j]      mass at (k_grid[i], eps_j),       sums to 1
    kp[i,j]     policy choice k'(k_i, eps_j),     possibly off-grid
    pi_eps[j,jp] = Pr(eps_{t+1}=jp | eps_t=j)
    """
    G_next = np.zeros_like(G)
    dk     = k_grid[1] - k_grid[0]                       # uniform spacing
    n_k    = len(k_grid)
    for j in range(G.shape[1]):                          # current shock
        for i in range(G.shape[0]):                      # current capital bin
            x = kp[i, j]
            # Boundary handling: clip BOTH the bracket index J AND the
            # lottery weight omega.  Otherwise an off-grid x produces
            # omega < 0 or omega > 1 and hence negative or super-unit mass.
            if x <= k_grid[0]:
                J, omega = 0, 1.0                        # all mass to the floor
            elif x >= k_grid[-1]:
                J, omega = n_k - 2, 0.0                  # all mass to the cap
            else:
                J     = int((x - k_grid[0]) // dk)
                J     = min(max(J, 0), n_k - 2)
                omega = (k_grid[J + 1] - x) / dk         # in [0, 1]
            for jp in range(G.shape[1]):                 # next-period shock
                w = pi_eps[j, jp] * G[i, j]              # shock fork x source mass
                G_next[J,     jp] += omega       * w
                G_next[J + 1, jp] += (1 - omega) * w
    return G_next
```
The four scatter-add lines correspond exactly to the four leaves of {numref}`fig-young_cascade`: each leaf receives `omega` or `(1-omega)` from the capital lottery, multiplied by `pi_eps[j, jp]` from the shock fork, multiplied by the source mass. The Krusell–Smith JAX tutorial in [`lectures/lecture_10_sequence_space_deqns/code/lecture_10_KrusellSmith_Tutorial_CPU.ipynb`](notebooks/lecture_10_sequence_space_deqns/lecture_10_KrusellSmith_Tutorial_CPU.ipynb) implements this same operation as `distribution_step`, vectorised over the grid via `jax.vmap` and accumulated with `.at[ ].add( )`.

````{prf:remark} Closed-form bracketing on GPUs, and how to keep it on log-spaced grids

 The expensive sub-step of Young's update (and of any piecewise-linear interpolation) is the *bracketing index*

```{math}
:enumerated: false

J(k') \;=\; \max\{\,n : k_n \le k'\,\},\qquad k' \in [k_J,\, k_{J+1}).
```

The textbook implementation uses a binary search (e.g. `numpy.searchsorted` or `jnp.searchsorted`): $\mathcal{O}(\log N_k)$ per query, with data-dependent branches. On a CPU this is essentially free; on a GPU under `XLA`/`CUDA` it is one of the worst patterns one can write, because (i) SIMT threads of the same warp take different branches (warp divergence), (ii) the resulting gathers are irregular, and (iii) the opaque search op breaks operator fusion with the surrounding arithmetic, forcing extra kernel launches.

For an *equidistant* grid the search collapses to a single fused multiply–add and a cast,

```{math}
:enumerated: false

J(k') \;=\; \mathrm{clip}\!\left(\Big\lfloor \tfrac{k' - k_0}{\Delta k} \Big\rfloor,\;0,\;N_k-2\right),
```

which is exactly what the line `J = int((x - k_grid[0]) // dk)` in the cheatsheet does. This is branch-free, uniform across threads, and fuses with the lottery weight $\omega = (k_{J+1}-k')/\Delta k$ into a single GPU kernel.

**Log-spaced grids without losing $\mathcal{O}(1)$ lookup.** Uniform $k$-grids waste resolution where the consumption policy is flat (the right tail) and starve resolution where it is steepest (near the borrowing constraint $k\!\to\!0$). A simple fix preserves the closed-form bracketing: place equidistant knots in a *transformed* coordinate $x = \phi(k)$ that maps the desired refinement region uniformly. For a log-spaced grid with shift $c>0$,

```{math}
:enumerated: false

x_n \,=\, x_0 + n\,\Delta x,\qquad k_n \,=\, e^{x_n} - c,\qquad n=0,\dots,N_k,
```

the bracketing index becomes

```{math}
:enumerated: false

J(k') \;=\; \mathrm{clip}\!\left(\Big\lfloor \tfrac{\log(c+k') - x_0}{\Delta x}\Big\rfloor,\;0,\;N_k-2\right),
```

again $\mathcal{O}(1)$ per query and branch-free. Crucially, the *index* is computed in $x$-space but the lottery *weights* are taken in the original $k$-space (using $k_J$, $k_{J+1}$ from the table), so the interpolated policy remains piecewise linear in $k$, consistent with the on-grid consumption values and with the mean-preserving lottery of Eq. {eq}`eq-young_meanpreserve`. More generally, any bijective $\phi$ for which $\phi^{-1}$ is cheap admits the same trick. See `interpolate()` and `distribution_step()` in the Krusell–Smith JAX tutorial of {cite:t}`azinovicyangzemlicka2025sequencespace` for a production implementation.
````


{numref}`fig-young_forward` visualizes the five stages of a single forward step.

```{figure} figures/fig-young_forward.svg
:name: fig-young_forward

Flow diagram for one forward step of Young's histogram update ({prf:ref}`alg-young`). Starting from $G_t$, the policy function is evaluated at every active bin, the resulting off-grid savings are interpolated back onto the grid, and idiosyncratic shock transitions redistribute mass across $\varepsilon$-states to produce $G_{t+1}$.
```

**Comparison with Monte Carlo.** Young's method produces *zero sampling noise* (deterministic), preserves the mean *exactly*, requires only $\sim$100–5,000 grid points (versus $>$50,000 agents for Monte Carlo), and is fully reproducible. The trade-off is of a different kind than Monte Carlo's: the mean-preserving lottery introduces a deterministic *discretization bias* into higher moments (it tends to flatten cross-sectional kurtosis at coarse grids and is sensitive to the choice of bracket scheme), whereas a Monte Carlo panel is unbiased in all moments but pays $\mathcal{O}(N^{-1/2})$ sampling noise on each. Both methods also require a grid (Young) or an empirical range (Monte Carlo) that is wide enough to contain all mass. The following table summarizes the comparison:

|  | **Young's method** | **Panel simulation** |
|---|:---:|:---:|
| Sampling noise | None | $\mathcal{O}(1/\sqrt{N})$ |
| Mean preservation | Exact | Approximate |
| Typical size | 100–5,000 grid points | $>$50,000 agents |
| Reproducibility | Deterministic | Seed-dependent |
| Higher moments | Approximated | Approximated |

{numref}`fig-young_vs_mc` contrasts the two approaches visually: the histogram method yields a smooth, noise-free distribution, while a Monte Carlo panel of comparable size exhibits visible sampling noise.

```{figure} figures/fig-young_vs_mc.svg
:name: fig-young_vs_mc

Young's histogram (left) versus Monte Carlo panel simulation (right). Both approximate the same underlying wealth density (dashed). The histogram method is deterministic and smooth; the Monte Carlo panel exhibits $\mathcal{O}(1/\sqrt{N})$ sampling noise that contaminates downstream OLS regressions in the Krusell–Smith algorithm. *The bars in this figure are a TikZ schematic illustrating the two regimes; for the actual histograms produced by the algorithm see notebook `lecture_09_10_Youngs_Method_Examples` in the Lecture-09 `code/` folder.*
```

The absence of sampling noise matters for the Krusell–Smith algorithm: Monte Carlo noise in the realized mean contaminates the OLS regression that updates the forecasting rule, potentially destabilizing convergence.

**Grid design.** Two separate grids are used in practice. The *value function grid* (typically $\sim$150 irregularly spaced points) is finer near the borrowing constraint where the policy function has high curvature and coarser for large $k$ where behavior is smooth. The *simulation grid* for Young's histogram (typically $\sim$1,000–5,000 uniformly spaced points) uses uniform spacing to produce smooth histograms without artifacts. The upper bound $k_{\max}$ must be chosen large enough that no mass reaches the boundary; if $k' > k_{\max}$ for any agent, all mass is assigned to the last grid point, which violates mean preservation. A practical safeguard is to run a preliminary simulation and verify that the boundary bins contain negligible mass.

**The full Krusell–Smith algorithm.** Combining value function iteration (VFI) with Young's simulation yields:

1.  Initialize forecasting coefficients $A(a), B(a)$.

2.  Solve the household problem via VFI given the forecasting rule $\hat{H}$.

3.  Forward-simulate the distribution for $T$ periods via Young's method, recording realized means $m_1^t$.

4.  Re-estimate $A(a), B(a)$ by OLS on simulated $(m_1^t, m_1^{t+1}, a_t)$.

5.  Check convergence; if not converged, return to step 2.

Young's non-stochastic simulation makes step 3 essentially noise-free and fast relative to a large Monte Carlo panel. The full outer-loop iteration count and wall-clock time, however, remain implementation- and tolerance-dependent because the VFI solve and forecasting-rule fixed point are still present; the speedup applies to the simulation step, not as a generic wall-clock guarantee for the traditional KS workflow.

**Convergence and accuracy.** A remarkable empirical finding is that the log-linear forecasting rule {eq}`eq-ks_forecast` achieves $R^2 > 0.9999$ in the standard Krusell–Smith economy: the first moment of the wealth distribution is a nearly sufficient statistic for predicting next-period prices. Adding higher moments (variance, skewness) to the forecasting rule barely improves the fit. This "approximate aggregation" result does not hold universally; it relies on the specific features of the Krusell–Smith calibration (small aggregate shocks, moderate borrowing constraint), but it is a useful benchmark against which richer models can be compared.

(sec-young_deqn)=
## DEQN with a Continuum of Agents
The traditional Krusell–Smith algorithm provides the benchmark logic for this chapter, but the classroom DEQN notebook used in the course is the simpler Bewley endowment economy of Appendix A.5 in {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`. This distinction is pedagogically useful. Krusell–Smith explains *why* distribution tracking matters and *why* Young's method is valuable inside an outer forecasting-rule loop. Appendix A.5 then shows *how* the same histogram machinery enters a neural-equilibrium implementation once one replaces the forecasting rule by a price network. By combining Young's histogram method with neural network policies, the DEQN approach overcomes both main limitations of the traditional KS workflow: the network can condition on the *full histogram*, and there is no need for a separate forecasting rule because the price network directly takes the distribution as input.

**What {ref}`sec-young_deqn` inherits from Appendix A.5.** Five features of the Appendix-A.5 teaching model anchor the rest of this section and the companion notebook [`11_Continuum_of_Agents_DEQN.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_11_Continuum_of_Agents_DEQN.ipynb); they are the distinguishing departures from the canonical Krusell–Smith calibration of §{ref}`sec-ks_economy`–{ref}`sec-young_method`:

- **Endowment economy, not production.** Aggregate output is exogenous, $Y_t = w(a_t)$, instead of $Y_t = z_t K_t^\alpha L_t^{1-\alpha}$; there is no capital and no firm problem.

- **Bonds in unit net supply.** Households trade a single one-period bond at endogenous price $p_t$, and the market-clearing condition is $\int b_{t+1}\,d\mu_t = \bar B = 1$.

- **Epstein–Zin recursive utility.** Risk aversion ($\sigma=8$) and inverse IES ($\rho=2$) are separated, with discount factor $\beta=0.95$. The KS benchmark in {ref}`sec-ks_economy` instead used log utility with $\beta=0.99$.

- **Six-state aggregate shock.** $a_t \in \{0,\ldots,5\}$ encodes a $2$-state uncertainty regime crossed with a $3$-state income level, replacing the $2$-state TFP shock of canonical KS.

- **Two idiosyncratic productivity types.** Labour endowment $\eta_t \in \{0.8, 1.2\}$ on a transition matrix $\Pi_\eta$, in place of the employed/unemployed two-state Markov chain of canonical KS.

**Histogram encoding.** The aggregate state is encoded as a vector containing three aggregate-shock index entries plus $N_b$ histogram values for each idiosyncratic shock type. In the Appendix A.5 notebook these three entries are the combined six-state aggregate index, the income-level index, and the uncertainty-regime index:

$$
x_t^{agg} = \bigl(\underbrace{z_t^{\mathrm{idx}},\,\mathrm{inc}_t^{\mathrm{idx}},\,\mathrm{unc}_t^{\mathrm{idx}}}_{3\text{ shock-index entries}},\; \underbrace{h_t^{\eta=0.8}(b_1),\ldots,h_t^{\eta=0.8}(b_{N_b})}_{N_b\text{ values}},\;\underbrace{h_t^{\eta=1.2}(b_1),\ldots,h_t^{\eta=1.2}(b_{N_b})}_{N_b\text{ values}}\bigr).
$$ (eq-young_hist_state)

For $N_b = 100$ grid points and 2 idiosyncratic states, the aggregate state has dimension $3 + 200 = 203$. The full input to the policy network adds the individual state $(b_t, \eta_t)$, giving total dimension $205$. This is the `production` setting; the checked-in `smoke` and `teaching` runs use $N_b = 50$, so the aggregate state has dimension $103$ and the policy input $105$. {numref}`fig-young_encoding` shows how the histogram vector and individual state are assembled and fed into the two networks.

**How the notebooks fit together.** The companion notebook sequence mirrors this decomposition. [`10_Youngs_Method_Examples.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_10_Youngs_Method_Examples.ipynb) isolates Young's redistribution operator on toy examples, first in one dimension and then with idiosyncratic shocks. [`11_Continuum_of_Agents_DEQN.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_11_Continuum_of_Agents_DEQN.ipynb) then reuses the same logic inside the full aggregate state vector {eq}`eq-young_hist_state`. In other words, the second notebook is not a new distributional device; it is the first notebook's histogram update embedded in a larger equilibrium-learning loop.

```{figure} figures/fig-young_encoding.svg
:name: fig-young_encoding

Histogram encoding and neural network architecture. The individual state $(\eta_t, b_t)$ (blue boxes / blue arrows) and the aggregate state $(z_t^{\mathrm{idx}},\mathrm{inc}_t^{\mathrm{idx}},\mathrm{unc}_t^{\mathrm{idx}}, h_t^{\eta=0.8}, h_t^{\eta=1.2})$ (orange box for the three shock-index entries, red boxes for the two histograms) are concatenated as input to the policy network $\mathcal{N}_{pol}$; each input arrow is colored to match its source box and enters the policy-input layer at a distinct horizontal offset, so the five arrows are uniquely identifiable at a glance. The price network $\mathcal{N}_{price}$ receives only the aggregate state. Both networks use softplus output activations.
```

**Neural network architecture.** Two networks are trained jointly:

- **Policy network** $\mathcal{N}_{pol}$: takes individual + aggregate state ($\mathbb{R}^{5+2N_b}$) $\to$ outputs savings $b'$, borrowing multiplier $\lambda$, and value $V$ (3 outputs with softplus activation).

- **Price network** $\mathcal{N}_{price}$: takes only aggregate state ($\mathbb{R}^{3+2N_b}$) $\to$ outputs the bond price $p$ (1 output with softplus).

Both networks use two hidden layers with 500 units (`production`) or 128 (the checked-in `smoke`/`teaching` runs).

**Equilibrium conditions as loss.** The loss function comprises five terms, all of which should be zero in equilibrium:

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\left[\mathrm{EE}_i^2 + \mathrm{BE}_i^2 + \left(n_Z\,\mathrm{MC}_i\right)^2 + \mathrm{KKT}_i^2 + \mathrm{CB}_i^2\right],
$$ (eq-young_loss)

where EE is the Euler equation residual, BE is the Bellman equation consistency condition, MC is the bond market clearing condition, KKT is the borrowing complementarity, and CB penalizes negative consumption. The market-clearing residual is rescaled by $n_Z$ (the number of aggregate-shock states) before squaring, which puts the single market-clearing residual on the same scale as the per-state Euler, Bellman, and complementarity residuals (themselves evaluated in relative terms, divided through by consumption).

**Market clearing via histogram.** A key advantage of the histogram representation is that market clearing can be computed *exactly* (no Monte Carlo). The equilibrium condition is $\sum_{\eta,j} h_t(\eta,b_j)\,b'(\eta,b_j,x_t^{agg}) = \bar B$; the residual that enters the loss is its deviation from zero,

$$
\mathrm{MC} \;=\; \sum_{\eta}\sum_{j=1}^{N_b} h_t(\eta, b_j)\cdot b'(\eta, b_j, x_t^{agg}) \;-\; \bar B \;=\; 0 \text{ at equilibrium},
$$ (eq-young_mc)

where $h_t(\eta, b_j)$ is the histogram mass, $b'(\cdot)$ is the policy network output, and $\bar B$ is aggregate net bond supply. We write the *signed* residual here because it enters the loss {eq}`eq-young_loss` squared, which is also how the notebook computes it. The Appendix A.5 notebook normalizes $\bar B=1$, which is the source of the "$-1$" term sometimes seen in code and slides.

**Young's method inside the training loop.** During each training episode, the histogram is propagated forward using Young's method ({prf:ref}`alg-young`) with the current neural network providing the policy function. This creates a sequence of aggregate states $(x_0^{agg}, x_1^{agg}, \ldots, x_T^{agg})$ on which the equilibrium residuals {eq}`eq-young_loss` are evaluated. Young's redistribution operator is differentiable almost everywhere (linear interpolation conditional on fixed brackets), so the *one-step* histogram update that enters the market-clearing residual at each sampled state carries gradients back to the policy network. The longer simulated path of aggregate states is treated as data: it is regenerated each episode from the current network and held fixed inside the gradient tape, rather than backpropagated through end to end. Even so the distribution co-evolves with the network: early in training, when the policy network is inaccurate, the simulated path will be "wrong," but the market-clearing residuals evaluated along it provide corrective feedback through the loss, and as the network improves the distribution converges toward the ergodic steady state.

(sec-young_results)=
## Results and Discussion
In the Appendix A.5 teaching model, the DEQN with histogram encoding achieves competitive accuracy: Euler equation errors are of order $10^{-3}$, and market-clearing residuals are comparably small. These figures come from the checked-in `teaching`/`smoke` configuration of the companion notebook (a small network trained for a modest number of episodes); the `production` configuration tightens them further. The broader lesson for the Krusell–Smith benchmark is conceptual rather than model-specific. Compared to the traditional KS algorithm, the DEQN approach has two advantages: (i) the neural network can condition on the *full* distribution rather than just its mean, providing a richer approximation that can capture situations where higher moments of the distribution matter for prices, and (ii) there is no need for a separate outer loop to update forecasting coefficients, since equilibrium conditions are enforced directly through the loss function.

```{prf:remark} Connection to continuous time

 The discrete-time histogram approach presented here has a natural continuous-time counterpart. In {ref}`ch-ct_theory`, we will see that the Kolmogorov forward equation (KFE; Fokker–Planck) describes the evolution of the wealth *density* in continuous-time heterogeneous-agent models such as {cite:t}`achdou2022income`. While the mathematical formulation differs (histogram update vs. PDE), the economic question is the same: how does the wealth distribution evolve, and how does it feed back into equilibrium prices?
```


(sec-ks_alternatives)=
## Alternative Deep-Learning Approaches to Krusell–Smith
Before turning to the deep-learning alternatives, it is useful to set the histogram-DEQN method in the broader landscape of solution techniques for heterogeneous-agent equilibria with aggregate shocks. {numref}`tab-ha_methods_landscape` compares classical and modern approaches along four dimensions that drive method choice in practice.

````{table}
:name: tab-ha_methods_landscape

Heterogeneous-agent solution methods at a glance. The first three rows are classical or finite-difference; the last three are the modern deep-learning families compared in detail in {numref}`tab-ks_dl_comparison` and the rest of this section.

| **Method** | **Distribution rep.** | **Aggregate state** | **Solution principle** | **Best when** |
|---|---|---|---|---|
| Classical KS {cite:p}`krusell1998income` | Panel of $N \sim 10^4$ agents | First moment(s) | Bounded-rationality fixed point of forecasting rule | Standard incomplete-markets, low-dim aggregate state |
| Reiter (back-loaded) {cite:p}`reiter2009` | Histogram on a fixed grid | First-order perturbation around stationary | Linearize after solving the steady state | Small aggregate shocks, smooth policies |
| Continuous-time Achdou et al. {cite:p}`achdou2022income` | Density solving a KFE PDE | In the limit, the entire density | Coupled HJB+KFE finite differences | PDE-friendly model, smooth densities |
| Histogram-DEQN {cite:p}`azinovicDEEPEQUILIBRIUMNETS2022` | Young histogram on grid | Histogram entries (or moments thereof) | SGD on equilibrium residuals | Strong nonlinearities, occasionally binding constraints |
| All-in-one DL {cite:p}`maliar2021deep` | Panel of $N \gtrsim 10^3$ agents | Agent-level states, policy and aggregate together | SGD on stacked Euler+market-clearing residuals | Many states per agent, GPU available |
| DeepHAM {cite:p}`han2023deepham` | Permutation-invariant set encoder (DeepSets) | Learned $M\!\ll\!N$ generalized moments | Cumulative utility along simulated paths (policy-gradient with structural individual dynamics) | Want a low-dim aggregate state without committing to a moment *a priori* |
````

Whereas {numref}`tab-ha_methods_landscape` is panoramic (classical and DL methods on common axes), {numref}`tab-ks_dl_comparison` drills into the DL trio along the axes that matter when choosing among them. Histogram-DEQN is not the only deep-learning approach to heterogeneous-agent equilibria, and it is pedagogically useful to see how the space of deep-learning strategies decomposes. Three broad families have emerged in the literature. Two informative axes organize them: how the cross-sectional distribution is represented as input to the network, and what objective is optimized. Histogram-DEQN and All-in-One DL minimize residuals of the structural equilibrium equations; DeepHAM instead maximizes cumulative utility along simulated paths and uses Bellman residuals as a validation diagnostic.

````{table}
:name: tab-ks_dl_comparison

Three deep-learning approaches to heterogeneous-agent equilibria, all applied to the Krusell–Smith benchmark. They differ on two axes: how the cross-sectional distribution is encoded as input to the network, and what objective is optimized. Histogram-DEQN and All-in-One DL minimize squared structural residuals (Euler, Bellman, market clearing); DeepHAM instead maximizes cumulative utility along simulated paths, with the individual law of motion embedded in the computational graph and Bellman residuals tracked as a validation diagnostic.

```{list-table}
:header-rows: 2
:enumerated: false

* - 
  - **Histogram DEQN**
  - **All-in-one DL**
  - **DeepHAM**
* - 
  - {cite:p}`azinovicDEEPEQUILIBRIUMNETS2022`
  - {cite:p}`maliar2021deep`
  - {cite:p}`han2023deepham`
* - State distribution representation
  - Explicit histogram vector on a fixed grid
  - Explicit panel of $N$ agents' states
  - Learned generalized moments via a permutation-invariant encoder
* - Dimension of aggregate state seen by the network
  - $\mathcal{O}(N_b)$ histogram entries
  - $\mathcal{O}(N)$ agent states
  - $M \ll N_b$ learned scalars
* - Interpretability of distributional state
  - High (histogram readable)
  - Low (permutation-dependent)
  - High in the economic sense (the learned basis is interpretable, e.g. concave in assets, linking the moment to MPCs and redistribution)
* - Permutation invariance
  - Automatic (histogram is invariant)
  - Requires careful architecture / data augmentation
  - Baked into the encoder by DeepSets structure
* - Training objective
  - Euler + Bellman + market clearing + FB residuals (squared)
  - Euler + Bellman + market clearing residuals (squared)
  - Cumulative utility along simulated paths plus value-function error; Bellman residual is a validation diagnostic only
* - Reported accuracy (baseline KS)
  - Euler errors $\sim 10^{-3}$–$10^{-4}$
  - Approximation errors $< 1\%$ with $>10^3$ agents
  - Bellman residual (used as a validation diagnostic) reduced by $68.9\%$ vs. KS with one learned generalized moment
```
````

(sec-mmw_ks)=
### All-in-One Deep Learning (Maliar, Maliar & Winant, 2021)
{cite:t}`maliar2021deep` propose what they call *all-in-one deep learning*. The key observation is that every dynamic economic model can be cast, at its core, as a collection of *expectation conditions* (optimality, feasibility, market clearing) that vanish at the true solution. They rewrite these conditions as non-linear regression equations with zero dependent variable, parameterize the policy and value functions by deep neural networks, and minimize the expected-squared residual by stochastic gradient descent on simulated paths.

For the Krusell–Smith benchmark with aggregate shocks and a continuum of agents, {cite:t}`maliar2021deep` formulate the following components of the loss. An alternative deep-learning route to the same problem is the symmetry-exploiting parameterization of {cite:t}`kahou2021exploiting`, which uses a permutation-invariant aggregation of agent-level features; the modern perturbation route is the refined Reiter implementation of {cite:t}`bayer2020solving`. Both are useful complements to the Young/DEQN combination developed below, with different scaling profiles and code-complexity tradeoffs.

- **Euler residual.** The household's consumption-saving first-order condition:

$$
\mathrm{EE}(s) \;=\; u'(c_t(s)) - \beta\,\mathbb{E}_{t}\!\left[u'(c_{t+1}(s'))\,R_{t+1}\right],
$$

where $s = (k_t,\varepsilon_t; \bm{S}_t)$ is the individual-plus-aggregate state and $\bm{S}_t$ is an $N$-agent panel of capital holdings. The policy network outputs $c_t = \pi_\rho(s)$; the Euler residual is evaluated at simulated states and the expectation by Monte Carlo over next-period idiosyncratic and aggregate shocks. With the borrowing constraint $k'\ge 0$, the plain Euler residual is insufficient: at a binding constraint the equation can be slack, so the loss must combine EE with a complementarity condition via Fischer–Burmeister or KKT (see below).

- **Bellman residual / value-function error** for off-policy learning of a value function, used in the "lifetime reward" formulation.

- **Market-clearing residual:** $\sum_i k_{t+1}^i = \bar K_{t+1}$ summed across the $N$ agents in the panel.

- **Borrowing constraint** non-negativity enforced architecturally (softplus on savings); the complementarity optimality condition still enters the loss via a Fischer–Burmeister term regardless.

A central computational contribution is the *all-in-one integration operator*: a single Monte Carlo realization of the next-period state is used to estimate the combined expectation across all residual terms simultaneously, rather than evaluating separate quadrature nodes for each conditional expectation. This reduces the cost of computing the conditional expectations from $\mathcal{O}(Q^m)$ (with $Q$ nodes per shock and $m$ shocks) to $\mathcal{O}(1)$ per state: a single draw of the next-period shocks plays the role of the full quadrature grid in the training loss.

**Scale reported by the authors.** {cite:t}`maliar2021deep` demonstrate the approach across several increasingly demanding setups: a Krusell–Smith variant with $N \geq 1{,}000$ explicit agents and $2{,}001$ state variables (two per agent plus one aggregate) in the baseline parameterization, scaled up to $N = 10{,}000$ agents in a moments-reduced variant where the cross section is summarized by $10$–$20$ aggregate moments; and on a one-agent consumption-savings problem with kinked policies they report approximation errors of at most a fraction of one percentage point. A Python/TensorFlow replication is available at <https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith> {cite:p}`marcmaliar2022dlks`; it is the basis for the companion notebook [`lectures/lecture_09_heterogeneous_agents_youngs_method/code/lecture_09_12_KrusellSmith_DeepLearning.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_12_KrusellSmith_DeepLearning.ipynb) (described below).

**Why this approach?** All-in-one DL is the closest cousin of the DEQN framework of {ref}`ch-deqn`. The only conceptual difference is that the aggregate state is represented by an *explicit panel* of $N$ agents (a large vector $\bm{S}_t \in \R^N$) rather than by a histogram on a fixed grid. Stochastic mini-batches draw both individual state and aggregate state, and the law of large numbers delivers permutation-invariance in the limit $N \to \infty$ without requiring a permutation-invariant architecture. The appeal is conceptual simplicity; the cost is that the input dimension grows with $N$ and the learner must rediscover permutation symmetry from the data.

(sec-deepham_ks)=
### DeepHAM: Generalized Moments via DeepSets (Han, Yang & E, 2024)
{cite:t}`han2023deepham` ask a more ambitious question: can the network *learn* which summary statistics of the cross-sectional distribution are relevant for individual decisions, rather than having the researcher commit to tracking a histogram or the first moment? They propose replacing the distribution $\mu_t$ with a small set of *generalized moments* obtained by averaging a flexible neural feature over the cross-section,

$$
m_t^\ell \;=\; \frac{1}{N}\sum_{i=1}^{N} \phi_\theta^{\,\ell}(s_t^i) \quad\xrightarrow{N\to\infty}\quad \int \phi_\theta^{\,\ell}(s)\, d\mu_t(s), \qquad \ell = 1,\ldots,M,
$$ (eq-deepham_moments)

with $\bm{m}_t = (m_t^1,\ldots,m_t^M)$ and $\phi_\theta^{\,\ell}: \R^{d} \to \R$ a neural feature encoder trained jointly with the policy and value networks. Equation {eq}`eq-deepham_moments` is the canonical DeepSets architecture of permutation-invariant set functions {cite:p}`zaheer2017deep`: averaging (or, in the continuum limit, integration against $\mu_t$) makes $\bm{m}_t$ invariant to permutations of the agents, and the $\phi_\theta^{\,\ell}$ encoders are flexible enough (by universal approximation on the permutation-invariant functions) to represent any fixed-arity moment.

The individual's value and policy functions then take the form

$$
V_t = V_\rho(s_t^i; \bm{m}_t, a_t), \qquad k_{t+1}^i = \pi_\rho(s_t^i; \bm{m}_t, a_t),
$$

and all networks $(V_\rho, \pi_\rho, \phi_\theta)$ are trained jointly. The primitive training objective is to *maximize* cumulative utility along simulated paths,

$$
J(\theta,\rho) \;=\; \mathbb{E}\!\left[\sum_{t=0}^{T-1} \beta^{t}\,u\!\bigl(c_\rho(s_t^i; \bm{m}_t, a_t)\bigr) \;+\; \beta^{T}\, V_\psi(s_T^i; \bm{m}_T, a_T)\right],
$$ (eq-deepham_objective)

where the expectation is taken over simulated idiosyncratic and aggregate histories generated under the current policy. Squared Bellman residuals are reported as a validation diagnostic, not as the optimization target. Because the individual law of motion, the budget constraint, and the transition structure are known economic objects, they are written directly into the computational graph: gradients of $J$ flow through these structural dynamics, in contrast to model-free reinforcement learning where transitions are observed only as samples {cite:p}`yang2025structural`. Because the generalized moments are themselves parameters of the optimization (not hyperparameters), the method automatically discovers the minimal set of distributional statistics required for equilibrium pricing.

A practical consequence of the policy-gradient formulation is that DeepHAM is well suited to problems where first-order conditions are difficult to write down or inconvenient to use, including constrained-efficiency problems with aggregate shocks, optimal-policy design, and behavioral macro questions; the headline application in {cite:t}`han2023deepham` is precisely a constrained-efficiency problem solved by simulating the economy under candidate allocation rules and updating the rules to improve social welfare.

**Reported accuracy (validation diagnostics).** The numbers that follow are taken from {cite:t}`han2023deepham`; they have not been independently replicated by the companion notebooks of this chapter, and should be read as reported results rather than as figures we can vouch for from our own runs. On the baseline Krusell–Smith model, the authors report the following Bellman-error reductions, computed *ex post* as validation measures of the converged solution rather than as the training objective:

- DeepHAM with *only* the first moment in the state vector reduces the Bellman residual by **61.6%** compared to the classical KS algorithm.

- DeepHAM with *one* learned generalized moment (on top of, or replacing, the first moment) reduces the residual by **68.9%**.

- The method extends to HA models with multiple shocks, multiple endogenous states, large shocks (risky steady state), and nonlinearities (e.g., ZLB) where both KS and the local perturbation method of {cite:t}`reiter2009` break down.

Conceptually, DeepHAM bridges two traditions: the approximate-aggregation insight of {cite:t}`krusell1998income` (a small number of moments can suffice) and the deep-function-approximator philosophy of {cite:t}`maliar2021deep` and {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` (the NN need not be restricted to pre-specified basis functions). The official reference implementation, including replication code for the Krusell–Smith benchmarks above, is available at <https://github.com/frankhan91/DeepHAM>.

**Interpretability of the learned moments.** Although the encoders $\phi_\theta^{\,\ell}$ are flexible neural networks, the moments they produce are interpretable in the economic sense relevant to heterogeneous-agent macro: they summarize how the cross-section affects welfare and policy. In the Krusell–Smith application of {cite:t}`han2023deepham`, the learned basis function is concave in assets, so a marginal asset held by a poor household contributes more to the moment than a marginal asset held by a rich household. This links the learned distributional representation directly to marginal propensities to save, redistribution, and welfare effects, which is the natural object of interest in HA models. Generalized moments are therefore not just a flexible compression of $\mu_t$; they double as a device for reading economic content out of the trained equilibrium.

**From learned moments to learned state spaces.** Once equilibrium computation is written as policy-gradient optimization over simulated paths, one can ask a sharper question: do agents need the full distribution as a state variable at all? In Walrasian heterogeneous-agent models, agents care about $\mu_t$ only indirectly, through equilibrium prices. {cite:t}`yang2025structural` exploit this in their *structural reinforcement learning* (SRL) framework: agents' policy functions take low-dimensional prices, or short price histories, as the aggregate state, while $\mu_t$ remains part of the simulated environment used to clear markets. This sidesteps the master equation entirely for a substantial class of HA economies and produces a natural conceptual arc: KS chooses moments *a priori* ({ref}`sec-young_method`); DeepHAM *learns* moments from the equilibrium objective (this section); SRL replaces moments with prices when the model permits, and lets ML help *define* a tractable equilibrium concept rather than only solving a fixed one.

(sec-deepsam_ks)=
### Beyond Walrasian Markets: DeepSAM and Search Frictions
A natural follow-up question is whether the DeepHAM machinery extends to economies in which the cross-sectional distribution affects decisions through channels other than aggregate prices. In standard heterogeneous-agent models the distribution is felt only through equilibrium prices: $\mu_t$ enters individual problems by setting $r_t = r(K_t)$ and $w_t = w(K_t)$. In labor markets, search-and-matching, and other non-Walrasian settings, the distribution enters more directly: through the matching technology, the type composition on each side of the market, outside options, and bargained transfers. This makes the equilibrium mapping intrinsically non-Walrasian and forecloses simple price-only summaries of the cross-section.

{cite:t}`payne2025deepsam` address this case with *DeepSAM*, a deep-learning solver for search-and-matching models with two-sided heterogeneity and aggregate shocks. The architecture inherits the DeepHAM idea of a permutation-invariant set encoder, applied to each side of the market separately, and feeds the resulting type-composition summaries into networks that approximate value functions, matching surplus, and policy. The training objective is again policy-improvement on simulated paths, with the matching technology and bargaining rule embedded structurally. In Walrasian HAM the cross-sectional asset/wealth/income distribution enters individual problems only *indirectly*, through equilibrium prices $r_t,w_t$; in SAM the type composition on each side of the matching market enters *directly*, through the matching technology, outside options, and bargained transfers. That is why DeepHAM and DeepSAM, despite sharing a permutation-invariant set-encoder architecture, treat the distribution differently.

### Which Method, When?

The three deep-learning approaches (Histogram DEQN, {ref}`sec-young_deqn`; All-in-One DL, {ref}`sec-mmw_ks`; and DeepHAM, {ref}`sec-deepham_ks`) are complements rather than substitutes. {numref}`tab-ks_dl_comparison` summarizes the practical trade-offs:

- For *teaching* purposes, the Histogram DEQN is the cleanest: the network input is an interpretable distribution vector, and the training loop directly mirrors the DEQN template introduced in {ref}`ch-deqn`.

- For *research* problems where the number of agents is the natural state dimension (e.g., overlapping generations with many cohorts, or finite-agent social-planner problems), All-in-One DL is often more convenient because it requires no grid design.

- For *policy analysis in richer HA environments* (risky steady states, multiple endogenous states, ZLB), DeepHAM's learned-moment representation pays off both in accuracy and in interpretability, because the learned moments can be plotted and analyzed as functions of the distribution.

{numref}`tab-ks_dl_chooser` distils the same trade-offs into a quick decision aid: when the model fits the row's \"When it shines\" column, the matching method is the first one to try.

````{table}
:name: tab-ks_dl_chooser

Practical chooser for the three DL approaches to Krusell–Smith. When several rows look applicable, the recommended ordering is: start with Histogram DEQN if a clean teaching narrative is the goal; switch to All-in-One DL if grid design is awkward; switch to DeepHAM if the cumulative-utility objective or learned moments are first-order to the question. Adapted from the L09 deck's *Head-to-Head* slide.

| **Method** | **Pros** | **When it shines** |
|---|---|---|
| Histogram DEQN {cite:p}`azinovicDEEPEQUILIBRIUMNETS2022` | Interpretable state; exact market clearing; reuses the DEQN template | Teaching; $N_b$ moderate; smooth policies |
| All-in-One DL {cite:p}`maliar2021deep` | No grid design; large-$N$ panels; single optimizer for all residuals | Large-$N$ research problems; OLG-many-cohort extensions |
| DeepHAM {cite:p}`han2023deepham` | Learned moments; cumulative-utility objective; risky steady state; ZLB | Rich HA macro-finance; constrained-efficiency / optimal-policy design |
````

**Notebook: a classroom-scale all-in-one KS solver.** The accompanying Jupyter notebook [`lecture_09_12_KrusellSmith_DeepLearning.ipynb`](notebooks/lecture_09_heterogeneous_agents_youngs_method/lecture_09_12_KrusellSmith_DeepLearning.ipynb) (introduced in {ref}`sec-young_results` and described in detail in the README) implements a classroom-scale version of the all-in-one DL approach of {cite:t}`maliar2021deep` on the Krusell–Smith benchmark with the parameters of {cite:t}`krusell1998income`. It uses a single policy network $\pi_\rho(k, \varepsilon, K, a)$ parameterized by TensorFlow/Keras and an explicit panel of $N$ agents as the distributional input; the loss is the squared Euler residual, augmented with a Fischer–Burmeister complementarity term at the borrowing constraint, and aggregate consistency is imposed by construction (next-period capital is the cross-sectional mean of the panel's savings choices) rather than through a separate market-clearing penalty. This is a deliberate simplification of the full all-in-one formulation of {ref}`sec-mmw_ks`, which also carries a value network and an explicit market-clearing residual. The notebook is annotated cell-by-cell with the correspondence to the equations in this chapter, and is designed to converge in under ten minutes on a standard CPU. For the production-scale counterpart (up to $N = 10^4$ agents, GPU acceleration, richer shock structure), we refer the reader to the replication repository of {cite:t}`marcmaliar2022dlks`.

**Benchmarks and replication pointers.** For readers who want to benchmark any of the deep-learning approaches against the traditional KS algorithm, the canonical reference implementation is `econ-ark/KrusellSmith` {cite:p}`econark2020ks`, which implements the forecasting-rule-update algorithm of {cite:t}`krusell1998income` and reports $R^2 > 0.9999$ within about 20 outer iterations under standard parameters ($\beta = 0.99$, $\alpha = 0.36$, log utility, two aggregate states). Any deep-learning method must at least match that accuracy on the baseline model; the advantages are supposed to appear when the model is extended in directions KS cannot handle cleanly (more moments, many endogenous states, risky steady state).

(sec-sequence_space)=
## Extension: Deep Learning in the Sequence Space
The histogram-based DEQN above is transparent because it feeds a direct approximation of the endogenous cross-sectional distribution into the neural network. The price of that transparency is dimensionality: in richer heterogeneous-agent economies, the aggregate state can contain hundreds of histogram entries. {cite:t}`azinovicyangzemlicka2025sequencespace` propose a different representation of the aggregate state. Instead of feeding the *current endogenous state* to the network, they feed a *truncated history of exogenous aggregate shocks*. The equilibrium logic does not change: one still enforces Euler equations, market clearing, and occasionally binding constraints inside the loss. What changes is the object that summarizes the aggregate state for the network. {numref}`fig-sequence_space_compare` contrasts the two views.

```{figure} figures/fig-sequence_space_compare.svg
:name: fig-sequence_space_compare

Two ways to encode the aggregate state in deep equilibrium learning. Each pipeline reads top-to-bottom: the upper (colored) box is the *input* the user gives to the same neural network $\mathcal{N}_\rho$, the middle (colored) box is the network's *output* (policy and price objects), and the green box is the equilibrium loss that consumes those outputs. Histogram DEQNs (left, blue) feed an endogenous state representation $(A_t, \mu_t)$; sequence-space DEQNs (right, red) feed a truncated exogenous shock history $z_t^T$. Crucially, the network and the residual-based training loss are identical across the two pipelines, only the input changes.
```

**The sequence-space representation.** Let $z_t^T := (z_{t-T+1}, \ldots, z_t) \in \R^T$ denote the last $T$ realizations of the exogenous aggregate shock. The key claim is that, in an ergodic economy, this history is an *approximate sufficient statistic* for the endogenous aggregate state. In the Brock–Mirman warm-up notebook, the network maps the shock history to a bounded savings rate, from which next-period capital follows by the resource constraint,

```{math}
:enumerated: false

s_t = \sigma\!\bigl(\mathcal{N}_\rho(z_t^T)\bigr) \in (0,1), \qquad K_{t+1} = s_t\, z_t K_t^\alpha,
```

where $\sigma$ is the logistic squashing that keeps $K_{t+1}$ feasible. In the richer heterogeneous-agent version, the network instead maps the same history to higher-level equilibrium objects such as policy-function coefficients or pricing objects. This connects the method to the MIT-shock and sequence-space Jacobian literature of {cite:t}`boppart2018exploiting` and {cite:t}`auclert2021using`, but replaces local linear approximations with a global residual-based neural approximation.

````{prf:definition} State-space vs sequence-space: the equilibrium operator

 The two formulations can be written symmetrically. Let $y_t$ denote the equilibrium objects of interest (policies, prices) at date $t$, $x_t$ the endogenous aggregate state, and $\varepsilon_t$ the exogenous shock.

**State-space recursion.** The decision rule is a function $f$ of the current state, the state evolves through a known transition $H$, and equilibrium is the functional equation

```{math}
:enumerated: false

y_t = f(x_t), \qquad x_{t+1} = H(x_t, y_t, \varepsilon_{t+1}), \qquad G(f, x) = 0 \;\;\forall x.
```

**Sequence-space formulation.** Let $\mathcal{E}_t = (\varepsilon_t, \varepsilon_{t-1}, \ldots)$ denote the full shock history. The decision rule is now a function $\Psi$ of the history (with initial condition $x_0$), and the state $x_t$ is recovered by iterating the same law of motion under $\Psi$:

```{math}
:enumerated: false

y_t = \Psi(\mathcal{E}_t \mid x_0), \qquad x_t = \mathcal{H}(\mathcal{E}_t, x_0 \mid \Psi), \qquad G(\Psi, \mathcal{E}, x_0) = 0 \;\;\forall \mathcal{E}, \forall x_0.
```

Both formulations describe the *same* equilibrium. What differs is the *domain of approximation*: $f$ lives on the (potentially infinite-dimensional) endogenous state space, while $\Psi$ lives on the (also infinite-dimensional but exogenously driven) space of shock histories. In an ergodic economy the partial derivative $\partial\Psi/\partial\varepsilon_{t-\tau}$ vanishes as $\tau\to\infty$, so $\Psi$ admits a finite-history truncation $\widehat\Psi(z_t^T)$ with controllable error. This truncation step, developed in the next paragraph, is what makes the sequence-space formulation computable.
````


**Intuition first.** The easiest way to think about the method is as a *memory compression* device. A positive aggregate shock today raises output and therefore raises tomorrow's capital. That extra capital still matters the period after, but only through the production elasticity $\alpha$, so its influence is smaller. One more period later it is smaller again. In other words, the current aggregate state stores a decaying memory of past shocks. The sequence-space idea is to feed that shock history directly to the network rather than feeding the current endogenous state itself.

```{figure} figures/fig-sequence_space_decay.svg
:name: fig-sequence_space_decay

Intuition for sequence space in Brock–Mirman. $\log K_t$ depends on past shocks with weights that decay like $\alpha^j$ in the lag $j$ (here $\alpha = 0.36$, the standard capital share). Already at $j = 3$ the weight has fallen to $\sim\!0.05$, so a finite history of recent shocks summarizes the relevant aggregate information; very old shocks matter little.
```

**Brock–Mirman: what changes relative to {ref}`ch-deqn`?** The Brock–Mirman warm-up is useful because the change can be written down exactly. In {ref}`ch-deqn`, the state-space DEQN uses the current state as input,

```{math}
:enumerated: false

x_t^{\mathrm{state}} = (K_t, z_t), \qquad C_t = \mathcal{N}_\rho(K_t, z_t), \qquad K_{t+1} = z_t K_t^\alpha - C_t.
```

In the sequence-space version, the *economic model* is unchanged, but the network sees a different input:

```{math}
:enumerated: false

x_t^{\mathrm{seq}} = z_t^T = (z_{t-T+1}, \ldots, z_t), \qquad s_t = \sigma\!\bigl(\mathcal{N}_\rho(z_t^T)\bigr), \qquad K_{t+1} = s_t\, z_t K_t^\alpha, \qquad C_t = (1-s_t)\, z_t K_t^\alpha.
```

The Euler residual is the same object as before,

```{math}
:enumerated: false

G_t = 1 - \beta \,\frac{C_t}{C_{t+1}} \,\alpha z_{t+1} K_{t+1}^{\alpha-1},
```

so the economics are unchanged. What changes is the computational representation:

- the **network input** changes from the current state $(K_t, z_t)$ to the recent history $z_t^T$;

- the **network output** changes from current consumption $C_t$ to a bounded savings rate $s_t\in(0,1)$ in the warm-up notebook, so that $K_{t+1}=s_t z_t K_t^\alpha$ is feasible by construction;

- the **current capital stock** is no longer fed directly into the network, but is generated recursively from the initial condition and previously predicted capital choices;

- the **training samples** are overlapping shock histories rather than pointwise states $(K_t, z_t)$.

This distinction is important conceptually. For Brock–Mirman, sequence space is *not* a dimensionality reduction, since $(K_t, z_t)$ is only two-dimensional whereas a history of length $T=25$ is larger. The Brock–Mirman notebook is therefore a pedagogical demonstration of the idea that histories can stand in for endogenous states. The dimensionality gain appears only in richer heterogeneous-agent models, where the relevant alternative is a large histogram or other high-dimensional distributional summary.

**Intermediate bridge: sequence-space IRBC.** Between the one-shock Brock–Mirman warm-up and the infinite-dimensional Krusell–Smith state, the companion notebook [`lectures/lecture_10_sequence_space_deqns/code/lecture_10_05b_SequenceSpace_IRBC.ipynb`](notebooks/lecture_10_sequence_space_deqns/lecture_10_05b_SequenceSpace_IRBC.ipynb) re-trains the two-country IRBC model of {ref}`ch-irbc` under sequence-space inputs: the policy network reads the last $T=80$ shock vectors (a $240$-dimensional history with $\rho_z^T \approx 1.7\times 10^{-2}$ truncation error) instead of the four-dimensional current state. The $2N+1$ equilibrium residuals (Euler, ARC, Fischer–Burmeister), the Gauss–Hermite quadrature, and the cloud-method sampler are literally unchanged from nb 01; only the input domain changes. Because the current capital stock is no longer an input, we parametrize the output head around the steady state, $k'_j = k_{ss}\exp(\tanh z^k_j)$ and $\lambda = \lambda_{ss}\exp(\tanh z^\lambda)$, which keeps gradients lively at the target policy and prevents the cold-start divergence that plagued a naive softplus head. This notebook is a *pedagogical bridge* rather than a computational win, at a four-dimensional state the history is much larger, not smaller, but it shows that the same template handles a multi-equation system with multiple independent shock channels before we hand the method over to Krusell–Smith, where the dimensionality gain is real.

**Training logic.** The computational pattern is also close to the rest of this chapter. One samples an exogenous shock path, constructs overlapping history windows $z_t^T$, evaluates the network on those windows, and then uses the resulting decisions to simulate the endogenous economy forward. In the Brock–Mirman warm-up this produces the capital sequence directly; in the Krusell–Smith tutorial it produces policy-function objects, while Young's method still propagates the cross-sectional distribution inside the simulator. Residuals are then evaluated on the simulated path and backpropagated through the full pipeline. {numref}`fig-sequence_space_training` summarizes this workflow.

```{figure} figures/fig-sequence_space_training.svg
:name: fig-sequence_space_training

Training flow for sequence-space DEQNs. The exogenous shock history is the network input, but the forward simulator still produces endogenous objects such as prices, aggregate capital, or cross-sectional distributions needed for residual evaluation.
```

````{prf:remark} Worked example: what the network input looks like

 In the companion notebook [`KrusellSmith_Tutorial_CPU.ipynb`](notebooks/lecture_10_sequence_space_deqns/lecture_10_KrusellSmith_Tutorial_CPU.ipynb), the helper function `encode_Z_history` represents a discrete shock history as a one-hot block concatenated with the corresponding realized levels. Suppose $N_Z = 2$ (so $Z_t \in \{Z_L, Z_H\}$ with $Z_L = 0.93$, $Z_H = 1.07$) and the truncated history of length $H = 3$ is $(Z_L, Z_H, Z_L)$. `encode_Z_history` then returns

```{math}
:enumerated: false

\underbrace{\bigl[\,\underbrace{1,0}_{Z_L},\ \underbrace{0,1}_{Z_H},\ \underbrace{1,0}_{Z_L}\,\bigr]}_{\text{one-hot block, length }H \cdot N_Z}
\;\bigm\Vert\;
\underbrace{\bigl[\,0.93,\ 1.07,\ 0.93\,\bigr]}_{\text{level block, length }H},
```

a single vector of length $H \cdot (N_Z + 1) = 9$ that is fed to the MLP. In the Krusell–Smith tutorial, $H = 50$ and $N_Z = 2$, giving an input of length $150$. The corresponding histogram-based input, by contrast, would have hundreds of bins from the wealth distribution alone.
````


**Why truncated histories can work.** The Brock–Mirman warm-up makes the logic especially transparent. With full depreciation ($\delta = 1$) and log utility, recursive substitution shows that the capital stock depends on the last $T$ shocks up to an error of order $\alpha^T \log(K_{t-T})$. Since $\alpha < 1$ (typically $\alpha \approx 0.36$), this error vanishes exponentially: for $T = 25$, the truncation error is of order $10^{-11}$. More generally, in ergodic economies with persistent aggregate shocks, the approximation error decays at roughly $\max\{|\varrho|, |\alpha|\}^T$, where $\varrho$ denotes the persistence of the aggregate productivity shock (as in Ch. {ref}`ch-deqn`). In richer heterogeneous-agent models this is no longer an exact algebraic statement, so the history length $T$ becomes an empirical accuracy choice rather than a theorem.

**Why this is useful in heterogeneous-agent models.** Two advantages are worth separating. First, *as a network input*, a history of $T \approx 25$ shocks can be much smaller than a histogram with hundreds of bins. Second, exogenous shock histories are sampled from a fixed distribution. This removes one source of instability in residual-based training: the set of network inputs is anchored by model primitives even though the endogenous simulator still evolves with the current policy network. In the Krusell–Smith tutorial, this means that the network is conditioned on shock histories, while Young's method remains responsible for propagating the distribution used in market-clearing calculations.

````{prf:remark} Why sequence-space training is more stable: the feedback loop

 The second advantage above deserves to be unpacked, because it is empirically the single biggest source of stability gains reported in {cite:t}`azinovicyangzemlicka2025sequencespace`. In a state-space deep-learning HA solver the network reads the endogenous distribution $\mu_t$ as part of its input, and the training set of $\mu$'s is generated by simulating the economy under the *current* policy network. This creates a self-amplifying loop:

```{math}
:enumerated: false

\begin{aligned}
\rho^{(k)} \;\longrightarrow\; \pi_{\rho^{(k)}} \;\longrightarrow\; \{\mu_t\}_{\rho^{(k)}} \;\longrightarrow\;& \text{input distribution shifts}\\[-2pt]
\;\longrightarrow\; \text{out-of-distribution evaluations} \;\longrightarrow\;& \text{large residual gradient} \;\longrightarrow\; \rho^{(k+1)}\;\text{overshoots},
\end{aligned}
```

and the next outer iteration starts from inputs the network has never seen before, often producing even larger shifts. In sequence space the network input is the truncated shock history $z_t^T$, drawn from the *exogenous* law of motion of the aggregate shock. That distribution is fixed by model primitives and *does not move* with the policy update. The feedback loop is broken at its first link: training inputs are stationary even when the policy is far from optimal. Empirically this often turns a calibration that fails to converge in the state-space formulation (across random seeds and learning rates) into one that converges robustly in sequence space.
````


**Shape-preserving operator learning.** A second contribution of {cite:t}`azinovicyangzemlicka2025sequencespace` is to let the network output *policy-function objects* rather than a single scalar choice. In particular, they construct architectures that guarantee monotonicity and concavity of the predicted consumption rule by representing it with an I-spline basis and non-negative coefficients. In the Krusell–Smith tutorial, the network maps the shock history to these coefficients; the resulting policy can then be evaluated at all idiosyncratic states on the wealth grid. This operator-learning view pairs naturally with the endogenous grid method (EGM) of {cite:t}`carroll2006method` and avoids ad hoc penalties for monotonicity or concavity.

**Explicit I-spline MPC parameterization.** Having seen *why* a shape-preserving output head matters (above), we now write the construction down concretely; this is the most technical paragraph of the section and a reader who already accepts the monotonicity/concavity guarantees can skip to "Fischer–Burmeister KKT loss" below. Let $\{k_n\}_{n=0}^N$ be a fixed log-spaced wealth grid and let $B \in \mathbb{R}^{J \times (N+1)}$ be a precomputed I-spline basis evaluated on it,

```{math}
:enumerated: false

B_{j,n} \;=\; I_j\!\bigl(\log(\eta + k_n)\bigr), \qquad j = 1,\ldots,J,\; n=0,\ldots,N,
```

where $\eta>0$ is a small numerical shift (the `BASIS_SHIFT` constant in the notebook) and each $I_j$ is an integrated B-spline that is monotonically increasing from $0$ to $1$. For each idiosyncratic state $\varepsilon$, the network outputs two objects: a boundary marginal propensity to consume $\alpha(\varepsilon)\in(0,1)$ (sigmoid head) and non-negative weights $\widetilde w_j(\varepsilon)\ge 0$ with $\sum_j \widetilde w_j(\varepsilon) < 1$ (a "phantom-zero" softmax head). The grid MPC is

$$
\mathrm{MPC}_{\varepsilon,n} \;=\; \alpha(\varepsilon)\Bigl(1 - \sum_{j=1}^J \widetilde w_j(\varepsilon)\, B_{j,n}\Bigr),
$$ (eq-ispline_mpc)

which is decreasing in $n$ by construction (positive weights times an increasing basis, subtracted off a constant), bounded in $[0, \alpha(\varepsilon)] \subset [0,1]$, and continuous in the network parameters. Consumption is then recovered on the grid by cumulation of the MPC schedule along cash-on-hand $m = w\varepsilon + Rk$,

$$
c(\varepsilon, k_0) \;=\; \mathrm{MPC}_{\varepsilon,0}\, m(\varepsilon, k_0), \qquad
c(\varepsilon, k_n) \;=\; c(\varepsilon, k_{n-1}) + \mathrm{MPC}_{\varepsilon,n}\, R\, (k_n - k_{n-1}),
$$ (eq-ispline_cumulation)

and off-grid evaluation uses piecewise-linear interpolation. Equations {eq}`eq-ispline_mpc`–{eq}`eq-ispline_cumulation` guarantee, by construction and without any auxiliary penalty, that the consumption rule is non-negative, monotonically increasing in $k$, concave in $k$, and feasible ($c \le m$). In code, $B$ is the matrix `ispline_basis`, $\alpha$ and $\widetilde w$ come from the two heads of `actor_c_grid`, and the cumulation is the closing block of that same function.

**Fischer–Burmeister KKT loss.** Households face a borrowing constraint $k_{t+1} \ge 0$. The Karush–Kuhn–Tucker conditions of the household problem split into two regimes: at an interior optimum the Euler equation holds with equality, while at a binding constraint the Euler equation can be slack but next-period capital is zero. Define the (relative) Euler residual and the (relative) savings slack (in this section $g$ is reused as the Euler residual, matching the tutorial code's variable name; it is *not* the household policy function $g(k,\varepsilon,\ldots)$ of {ref}`sec-young_method`)

```{math}
:enumerated: false

g \;=\; \frac{c_{\text{Euler}} - c}{c}, \qquad s \;=\; \frac{k'}{c},
```

where $c_{\text{Euler}} = (u')^{-1}\!\bigl(\beta\,\mathbb{E}_t [R'\,u'(c')]\bigr)$ is the consumption level implied by the Euler equation given the network's continuation policy. The KKT pair is then

```{math}
:enumerated: false

g = 0,\ s \ge 0 \quad\text{(interior)} \qquad\text{or}\qquad g \ge 0,\ s = 0 \quad\text{(constrained)},
```

which is a complementarity condition. The Fischer–Burmeister envelope, in the same sign convention used in Ch. {ref}`ch-irbc` and Ch. {ref}`ch-olg`,

$$
\mathrm{FB}(g, s) \;=\; g \;+\; s \;-\; \sqrt{g^2 + s^2 + \epsilon_{\text{fb}}}
$$ (eq-fb)

is smooth and satisfies $\mathrm{FB}(g,s) = 0$ if and only if $\min(g, s) = 0$ with both non-negative; the small constant $\epsilon_{\text{fb}}$ (set to $10^{-12}$ in the notebooks) is a numerical stabilizer for the square root. The upstream JAX tutorial code uses the negative-sign variant $\sqrt{g^2+s^2+\epsilon}-g-s$, which has the same zero set when squared. The training loss is the buffer-and-grid average of $\mathrm{FB}^2$,

```{math}
:enumerated: false

\mathcal{L}(\rho) \;=\; \mathbb{E}_{(z^H,\mu)\sim\mathcal{B}}\biggl[\, \frac{1}{N_\varepsilon (N+1)} \sum_{\varepsilon,n} \mathrm{FB}\!\bigl(g_{\varepsilon,n}(\rho), s_{\varepsilon,n}(\rho)\bigr)^2 \,\biggr],
```

so that one differentiable scalar simultaneously enforces the Euler equation in the interior region and the complementarity condition at the borrowing constraint, without case splits or shadow-price augmentation. This reuses the smooth complementarity construction of {cite:t}`fischer1992special` that is standard in nonlinear programming, applied here to a heterogeneous-agent equilibrium loss.

**Putting the pieces together: the HA training loop.** The Krusell–Smith tutorial assembles the encoder, the I-spline policy head, Young's distribution step, and the Fischer–Burmeister loss into a single replay-buffer training loop. {prf:ref}`algo-ks_seqspace` states it explicitly.

```{prf:definition} Algorithm: Sequence-Space DEQN with Young's method (KS tutorial)
:label: algo-ks_seqspace

- **Input:** network $\mathcal{N}_\rho$ (I-spline MPC heads), I-spline basis $B$, transition matrices $\Pi_\varepsilon, \Pi_Z$, prices $R(K,L,Z), w(K,L,Z)$
- **Hyperparameters:** history length $H{=}50$, roll-out $T{=}100$, buffer cap $C{=}128$, $N_{\text{agg}}{=}8$, FB steps per epoch $S{=}10$, mini-batch $B_{\text{fb}}{=}16$, learning rate $\alpha{=}5\!\times\!10^{-5}$, gradient clip $\|\cdot\|_2 \le 1$
- **Initialize** replay buffer $\mathcal{B}$ with copies of $(\mathbf{0}_H,\, \mu_0)$, where $\mu_0$ is centered at the deterministic-RA reference $K_{ss}$
- **for** epoch $e = 1, 2, \ldots$ **do**
  - Draw $N_{\text{agg}}$ replay states $\{(z^H_i, \mu_i)\}$ from $\mathcal{B}$
  - Draw fresh shock paths $\{Z_{i,1:T}\}$ with $\Pi_Z$ from each terminal $z^H_i[-1]$
  - **for** $i = 1, \ldots, N_{\text{agg}}$ **do**
    - **Forward roll (no grad):** for $t=1,\ldots,T$ compute $K_t,L_t$ from $\mu_{i,t-1}$, prices $R_t,w_t$, MPC and $c_t$ from $\mathcal{N}_\rho$ on the running history, advance $\mu_{i,t}$ via the Young step
    - Append $(z^H_{i,T}, \mu_{i,T})$ to $\mathcal{B}$; evict oldest if $|\mathcal{B}| > C$
  - **end**
  - **for** $s = 1, \ldots, S$ **do**
    - Sample mini-batch $\mathcal{M}\subset\mathcal{B}$ of size $B_{\text{fb}}$
    - Compute Fischer–Burmeister loss $\mathcal{L}(\rho)$ on $\mathcal{M}$ using {eq}`eq-fb`
    - Update $\rho \leftarrow \mathrm{Adam}\bigl(\rho,\, \nabla_\rho \mathcal{L}(\rho)\bigr)$ with global-norm clipping
  - **end**
- **end**
- **Output:** trained $\mathcal{N}_{\rho^\star}$ that maps a shock history to grid-MPC coefficients
```


Three implementation choices in {prf:ref}`algo-ks_seqspace` are worth flagging. First, the forward roll is wrapped in `stop_gradient` so that gradients only flow through the FB residual evaluated on the buffer, not through long simulation chains; this is what makes training tractable for $T = 100$ horizons. Second, because Young's step gives *exact* aggregate sums, the only stochasticity in the gradient comes from buffer mini-batching, which acts as standard SGD noise rather than a Monte-Carlo aggregation noise floor. Third, the buffer simultaneously decouples training-state drawing from the current policy and lets the network see distributions $\mu$ generated by earlier policies, which improves coverage of the ergodic set during early training.

**Two training algorithms: residual minimization vs. time iteration.** {prf:ref}`algo-ks_seqspace` is one of two families of training schemes used in the sequence-space DL literature. In *direct residual minimization* (the version above and in our notebook), the network is trained by gradient descent on the squared equilibrium residual itself. In *time iteration with EGM*, the network is trained on a sequence of supervised regression problems: at each outer iteration, one (i) uses the current network to construct next-period policies, (ii) backs out implied current-period policies via the endogenous-grid method of {cite:t}`carroll2006method`, and (iii) updates the network by minimizing the squared error against those EGM targets. Time iteration is more involved and requires a per-batch root-finding step, but it is more flexible: it tolerates non-trivial market clearing and, crucially, handles *non-convex* choices (e.g., a discrete retirement decision) and non-monotone Laffer curves where the Euler equation has multiple roots that direct residual minimization cannot disambiguate. In practice, residual minimization is the simpler entry point on smooth, convex problems; switch to time iteration when convergence stalls, when the model contains discrete choices, or when the continuation value has convex regions that admit multiple optimal savings.

```{prf:remark} Practical heuristics for sequence-space DEQNs

 Four operational rules of thumb, distilled from {cite:t}`azinovicyangzemlicka2025sequencespace`:

- **Choosing the truncation length $T$.** Three sensible heuristics, in increasing order of conservatism: (i) for OLG models, set $T$ to roughly two life-cycles, so that all shocks experienced by any household alive today are inside the window; (ii) start short and iteratively increase $T$, monitoring the equilibrium residual; (iii) "overkill", set $T$ such that $\varrho_{\text{shock}}^T \le \text{tol}$ with $\text{tol} \in [10^{-8}, 10^{-6}]$, since long histories are cheap to feed.

- **Innovations or realizations?** For an AR(1) shock $z_t = \varrho z_{t-1} + \varepsilon_t$, feeding the history of *realizations* $z_t^T$ to the network typically allows shorter truncation than feeding the history of *innovations* $\varepsilon_t^T$, because the levels already integrate the persistence. Feeding both is the most accurate option in practice.

- **What to approximate.** Smooth, bounded equilibrium objects (savings rates, MPCs) are easier to learn and yield better Euler accuracy than raw policy levels (consumption, savings). This is the deeper reason the I-spline MPC parameterization in {eq}`eq-ispline_mpc`–{eq}`eq-ispline_cumulation` pays off so much.

- **When the method breaks.** The truncation argument relies on *ergodicity* of the underlying dynamics, $\partial\Psi/\partial\varepsilon_{t-\tau} \to 0$ as $\tau \to \infty$. Models with stable limit cycles or deterministic chaos violate this assumption; in those exotic settings the sequence-space approach is not guaranteed to converge to the equilibrium policy.
```


**Applications and notebooks.** The paper applies this framework to three demanding models: (i) an OLG economy with portfolio choice and aggregate risk, (ii) an economy with a continuum of heterogeneous firms and households featuring idiosyncratic and aggregate shocks, and (iii) a lifecycle model with a discrete early-retirement choice that introduces local convexities. Mean Euler equation errors are below 0.2% in all cases. The two TensorFlow 2 companion notebooks are intentionally complementary: [`05_SequenceSpace_BrockMirman.ipynb`](notebooks/lecture_10_sequence_space_deqns/lecture_10_05_SequenceSpace_BrockMirman.ipynb) is a Brock–Mirman warm-up that isolates the history-to-policy logic in the simplest possible environment, while [`06_SequenceSpace_KrusellSmith.ipynb`](notebooks/lecture_10_sequence_space_deqns/lecture_10_06_SequenceSpace_KrusellSmith.ipynb) is a compact teaching implementation that combines sequence-space inputs, I-spline policies, and Young's method in a heterogeneous-agent setting.

A third notebook, [`KrusellSmith_Tutorial_CPU.ipynb`](notebooks/lecture_10_sequence_space_deqns/lecture_10_KrusellSmith_Tutorial_CPU.ipynb), is a JAX/optax port of the upstream pedagogical tutorial released by the paper's authors. It exposes the same shape-preserving I-spline MPC parameterization, the same Young step inside the simulator, and the same Fischer–Burmeister KKT loss as the TensorFlow notebook, but in the original JAX form. It is adapted from the upstream tutorial `01_KrusellSmith_Tutorial_CPU.ipynb` in the companion code repository {cite:p}`azinovicyangzemlicka2025sequencespacecode`, available at <https://github.com/azinoma/DeepLearningInTheSequenceSpace>; the local adaptation adds an explicit shape-guarantee diagnostic and additional inline commentary, leaving the algorithm unchanged.

````{prf:definition} Math-to-code glossary for the KS sequence-space tutorial

```{list-table}
:header-rows: 1

* - **Symbol**
  - **Meaning**
  - **Notebook name(s)**
* - $\rho$
  - Policy-network parameters
  - `psi`
* - $z_t^T$ (or $z^H$)
  - Truncated shock history of length $H$
  - `z_history`, `z_hist`
* - encoded $z^H$
  - One-hot $\Vert$ value vector, length $H(N_Z{+}1)$
  - output of `encode_Z_history`
* - $\mathcal{N}_\rho$
  - MLP mapping history to MPC heads
  - `actor_c_grid`
* - $B_{j,n}$
  - Precomputed I-spline basis matrix
  - `ispline_basis`
* - $\alpha(\varepsilon)$
  - Boundary MPC head (sigmoid output)
  - `alpha`
* - $\widetilde w_j(\varepsilon)$
  - Non-negative I-spline weights (phantom-zero softmax)
  - `w_tilde`
* - $\mathrm{MPC}_{\varepsilon,n}$, $c$
  - Decreasing MPC and cumulated grid consumption
  - `mpc`, `c_grid`
* - $\mu(\varepsilon, k)$
  - Cross-sectional distribution on the $(\varepsilon,k)$ grid
  - `mu`
* - $K_t, L_t$
  - Exact aggregates from $\mu$
  - `distribution_aggregates`
* - Young step
  - Non-stochastic distribution update $\mu \mapsto \mu'$
  - `distribution_step`
* - $g$, $s$
  - Relative Euler residual and savings slack
  - `g`, `s`
* - $\mathrm{FB}(g,s)$
  - Smooth complementarity envelope
  - `fb` in `fb_loss_one_state`
* - $\mathcal{B}$
  - Replay buffer of $(z^H, \mu)$ pairs
  - `buffer_z`, `buffer_mu`
```
````


```{prf:remark} Chapter Summary

 Continuum-agent equilibria require explicit distribution tracking; Young's (2010) histogram is a deterministic mass-redistribution scheme on a fixed grid that converges to the ergodic distribution exactly as the grid is refined. Embedding Young's update inside a DEQN training loop yields fully differentiable heterogeneous-agent solutions, with gradients flowing through the histogram and the policy network simultaneously. The sequence-space DEQN of {cite:t}`azinovicyangzemlicka2025sequencespace` is the natural alternative when the entire path of aggregate variables, rather than the cross-section, is the state of interest. Bewley–Huggett–Aiyagari–Krusell–Smith is the foundational lineage; {cite:t}`achdou2022income` supply the continuous-time counterpart taken up in {ref}`ch-ct_theory`.
```


## Further Reading {.unnumbered}
- {cite:t}`young2010`, the original non-stochastic histogram paper.

- {cite:t}`krusell1998income`, the canonical heterogeneous-agent benchmark with aggregate shocks.

- {cite:t}`azinovicyangzemlicka2025sequencespace`, sequence-space DEQNs, the natural extension.

- {cite:t}`achdou2022income`, the continuous-time treatment that {ref}`ch-ct_theory` builds on.

- {cite:t}`maliar2021deep` {cite}`han2023deepham`, alternative deep-learning approaches to KS, contrasted in {ref}`sec-ks_alternatives`.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch6-1

**[Core\] Mean-preserving lottery.** Show that the unique two-point split that places probability $\omega$ at $k_n$ and $1-\omega$ at $k_{n+1}$ such that $\omega k_n + (1-\omega) k_{n+1} = k'$ is given by $\omega = (k_{n+1} - k')/(k_{n+1} - k_n)$. Verify mass conservation.
```

```{exercise}
:label: ex-ch6-2

**[Computational\] Closed-form bracketing on log-spaced grids.** Implement the $\mathcal{O}(1)$ bracketing index for a log-spaced grid $k_n = e^{x_0 + n\Delta x} - c$ in five lines of NumPy. Verify against `numpy.searchsorted` on a random batch of queries; measure the relative speed-up.
```

```{exercise}
:label: ex-ch6-3

**[Core\] Approximate aggregation, scope.** Construct an example economy in which the KS log-linear forecasting rule fails (e.g. multiple assets with switching liquidity). Why does adding higher moments not always rescue the rule?
```

```{exercise}
:label: ex-ch6-4

**[Computational\] Sequence-space vs. histogram DEQN.** Use Notebook 11 as the histogram-state baseline and compare it with the sequence-space implementation in Notebook 06 or the JAX tutorial. In the sequence-space notebook, vary the history length $T$ (or $H$ in the code) and compare the residual after training. Comment on which formulation generalizes better to a much longer test horizon and why.
```

```{exercise}
:label: ex-ch6-5

**[Core\] DeepSets permutation invariance.** Consider the DeepHAM aggregator from {eq}`eq-deepham_moments`, $\bm{m}_t = \bigl(\sum_i g_\theta^1(s_t^i), \ldots, \sum_i g_\theta^M(s_t^i)\bigr)$. (i) Show that for any permutation $\pi$ of the agents, $\bm{m}_t(\pi \cdot s) = \bm{m}_t(s)$, so the encoder is permutation-invariant by construction. (ii) Show that the policy $\pi_\rho(s_t^i; \bm{m}_t, a_t)$ is consequently equivariant in the agent index $i$: if we permute the agents, each agent's policy moves with its own index but the dependence on the population through $\bm{m}_t$ is unchanged. (iii) State the converse ({cite:t}`zaheer2017deep`): any continuous permutation-invariant function on $\mathbb{R}^d$-valued sets of fixed cardinality can be written in the form $\rho(\sum_i g(s^i))$. Use this to argue why the DeepHAM architecture can in principle replace tracking the entire histogram with a finite vector $\bm{m}_t$ of learned moments, provided the dimension $M$ is large enough.
```

```{exercise}
:label: ex-ch6-6

**[Computational\] Histogram noise vs. Monte Carlo sampling.** In the Krusell–Smith tutorial notebook [`KrusellSmith_Tutorial_CPU.ipynb`](notebooks/lecture_10_sequence_space_deqns/lecture_10_KrusellSmith_Tutorial_CPU.ipynb), fix one aggregate shock path and one initial distribution. Run Young's histogram with $N_\mathrm{grid} = 100$ wealth grid points once, and run $R$ repeated Monte Carlo panels under the same aggregate path for $N \in \{100, 1000, 10\,000\}$ agents. At each $t = 1, \dots, 200$, record aggregate capital $K_t^{(r)}$ for every Monte Carlo replication $r$ and the corresponding Young aggregate $K_t^Y$. Plot the cross-replication standard deviation $\mathrm{sd}_r(K_t^{(r)})$ over time, or its time average, rather than the time-series standard deviation of $K_t$. Verify (i) Young's method has zero sampling variance conditional on the aggregate path, (ii) MC sampling variance scales as $1/\sqrt{N}$, (iii) at $N \approx N_\mathrm{grid}^2 = 10^4$, MC's stochastic noise becomes comparable to Young's discretization error on a smooth statistic like $K_t$. Comment on why the discretization-error-vs-MC-error trade-off depends on which functional of $\mu_t$ is targeted (mean, variance, tail mass).
```

```{exercise}
:label: ex-ch6-7

**[Computational\] Two-moment forecasting rule.** In the same notebook, replace the Krusell–Smith log-linear forecasting rule $\log K_{t+1} = a_0 + a_1 \log K_t + a_2 \mathbb{1}[z_t = z_g]$ with a two-moment extension that also conditions on cross-sectional dispersion: $\log K_{t+1} = a_0 + a_1 \log K_t + a_2 \mathbb{1}[z_t = z_g] + a_3 \log V_t$, where $V_t = \mathrm{Var}_{\mu_t}(k)$. Re-fit the rule and the network jointly. Report (i) the forecasting $R^2$ for $\log K_{t+1}$ before and after, (ii) the maximum Euler residual after training, (iii) by how much the second moment "buys" you in absolute residual terms. Connect the result to the chapter's discussion of *approximate aggregation*: in the standard Krusell–Smith calibration the marginal information in the second moment is small, but it becomes material once you introduce features (e.g., binding borrowing constraints in a non-trivial fraction of the population, multiple assets) that break aggregation.
```
