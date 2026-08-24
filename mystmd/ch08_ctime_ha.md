---
title: "Heterogeneous Agent Models in Continuous Time"
label: ch-ct_theory
---

Building on the PINN foundations of {ref}`ch-pinn`, this chapter develops the full continuous-time heterogeneous-agent framework: the coupled system of the Hamilton–Jacobi–Bellman equation (for individual optimization) and the Kolmogorov forward equation (for the stationary wealth distribution), closed by a market-clearing condition. Without aggregate risk, this coupled HJB–KFE system is the continuous-time analogue of the Aiyagari economy treated in {ref}`ch-young`; once aggregate risk is added, the natural object becomes the *master equation* ({ref}`sec-master_eq`), which is the continuous-time analogue of the Krusell–Smith model. The primary reference is {cite:t}`achdou2022income`; for pedagogical background on the continuous-time methods, see also Moll's lecture notes.[^1]

In the previous chapter, we applied PINNs to individual PDEs (ODEs, the Poisson equation, the HJB for cake-eating, and the Black–Scholes equation). This chapter makes the leap to *equilibrium systems*: coupled PDEs that arise when a continuum of heterogeneous agents interact through prices. The theoretical framework draws on the Bewley–Huggett–Aiyagari tradition, formulated in continuous time following {cite:t}`achdou2022income`. We derive the two core PDEs, the Hamilton–Jacobi–Bellman (HJB) equation for individual optimization and the Kolmogorov forward equation (KFE) for the cross-sectional density, and show how they are coupled through market clearing. The chapter culminates with the *master equation*, a single (infinite-dimensional) PDE that encapsulates the full equilibrium, and with EMINNs, introduced by {cite:t}`gu2024masterequations`, which solve it using deep learning.

**Companion notebook.** One notebook accompanies this chapter and the Lecture 13 numerical deck: [`lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb`](notebooks/lecture_13_continuous_time_ha_numerics/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb). It first computes the stationary Aiyagari equilibrium with an upwind finite-difference solver, then freezes those equilibrium prices and trains a PINN for the coupled HJB–KFE system at them (with normalization, boundary-flux, state-constraint, and aggregate-capital diagnostics) – so the PINN stage is an equilibrium-consistency and representation exercise, not a price-discovery algorithm. For transparency the notebook specializes the general CRRA/switching equations below to log utility ($\gamma=1$) and symmetric two-state Poisson switching; the formulas in this chapter are the general case. The detailed upwind finite-difference scheme and PINN losses are developed in the accompanying Lecture 13 deck; this chapter focuses on the continuous-time equilibrium equations and the conceptual bridge to master-equation methods. (A partial-equilibrium HJB on its own is just a single-PDE PINN problem of the kind treated in {ref}`ch-pinn`; here we go straight to the coupled equilibrium system.)

**Where does $g$ come from?** Before turning to the PDEs themselves, it is worth fixing the economic picture. The model has a continuum of agents, each indexed by an idiosyncratic state $(a,z)$ comprising wealth $a$ and a labor or productivity component $z$. Each agent solves its own HJB equation taking the prices $(r,w)$ as given, which yields an optimal savings policy that pushes mass through wealth space. The Kolmogorov forward equation tracks how this aggregate mass evolves, and its stationary solution $g^\star(a,z)$ is the cross-sectional density that the general-equilibrium clearing equations integrate against to obtain aggregate capital, $K = \int a\, g^\star(a,z)\,da\,dz$, and aggregate labor, $L = \int z\, g^\star(a,z)\,da\,dz$.

(sec-ct_why)=
## Why Continuous Time?
{ref}`ch-deqn`–{ref}`ch-young` formulated heterogeneous-agent models in discrete time. Working in continuous time offers several complementary advantages:

- **Analytical tractability.** FOCs hold with equality in the interior of the state space, and borrowing constraints appear as boundary conditions on the HJB/KFE rather than as occasionally binding inequality constraints inside the recursion; the separation between the backward HJB and the forward KFE is also sharper than in discrete time.

- **Differential operators in place of conditional expectations.** Itô calculus turns conditional expectations into infinitesimal generators, so the HJB and KFE are PDEs rather than integral equations; this removes numerical integration over shock distributions from the inner loop.

- **Powerful numerical methods.** Finite differences, PINNs, and deep learning methods (EMINNs) can be applied directly to the PDE system.

- **Connection to mean field games.** The coupled HJB–KFE system is precisely a *mean field game* (MFG) in the sense of {cite:t}`lasry2007mean`: each atomistic agent solves an HJB taking the cross-sectional density as given, while the density itself evolves via a KFE driven by those individual best responses. Equilibrium is the fixed point of this two-way coupling. Recasting the problem in MFG language gives access to a large mathematical literature on existence, uniqueness, and numerical analysis {cite:p}`carmona2018probabilistic,cardaliaguet2019master`.

**Historical context.** The models in this chapter build on a long tradition: {cite:t}`bewley1986stationary` introduced precautionary savings with borrowing constraints; {cite:t}`huggett1993riskfree` studied endowment economies with incomplete markets; {cite:t}`aiyagari1994uninsured` added production and general equilibrium; and {cite:t}`krusell1998income` incorporated aggregate uncertainty. {cite:t}`achdou2022income` reformulated these models in continuous time and demonstrated that finite-difference methods can solve the coupled HJB–KFE system efficiently. More recently, {cite:t}`gu2024masterequations` introduced EMINNs to solve the master equation globally, enabling treatment of aggregate shocks without moment-based approximations.

(sec-stoch_calc)=
## Stochastic Calculus Refresher
We briefly review the stochastic calculus tools needed for continuous-time models; for a standard finance-oriented textbook treatment, see {cite:t}`shreve2004stochasticii`.

**Quick reference.** Appendix {ref}`app-stoch` contains a one-page summary of the same material (Brownian motion, Itô's lemma, ergodicity in one paragraph) for readers who want a compact reminder rather than the full exposition below.

### Brownian Motion

```{prf:definition} Standard Brownian Motion

 A stochastic process $\{B_t\}_{t \geq 0}$ is a *standard Brownian motion* (Wiener process) if: (i) $B_0 = 0$; (ii) it has independent increments: $B_t - B_s \perp B_s - B_r$ for $r < s < t$; (iii) increments are Gaussian: $B_t - B_s \sim \mathcal{N}(0, t-s)$; and (iv) paths $t \mapsto B_t$ are continuous almost surely.
```


Key properties include $\E{B_t} = 0$, $\mathrm{Var}(B_t) = t$, nowhere-differentiable paths, and quadratic variation $\langle B \rangle_t = t$. Brownian motion arises as the scaling limit of a random walk: if $X_{t+\Delta t} = X_t + \sqrt{\Delta t}\,\varepsilon_t$ with $\varepsilon_t \in \{-1,+1\}$ equiprobably, then $X^{\Delta t} \xrightarrow{d} B_t$ as $\Delta t \to 0$ (Donsker's theorem). The $\sqrt{\Delta t}$ scaling ensures that $\mathrm{Var}(X_t) = t$ in the limit. {numref}`fig-brownian_paths` shows three discretized sample paths with the same variance scaling.

```{figure} figures/fig-brownian_paths.svg
:name: fig-brownian_paths

Three simulated standard Brownian sample paths $\{B_t\}_{t\in[0,1]}$, generated with discretization step $\Delta t = 0.05$ and Gaussian increments $B_{t+\Delta t} = B_t + \sqrt{\Delta t}\,\varepsilon_t$, $\varepsilon_t \sim \mathcal{N}(0,1)$. Paths are jagged at the chosen $\Delta t$ and the polylines linearly interpolate between the sample points, so the figure is an approximate visualization rather than a true Brownian path: limiting Brownian paths are continuous almost surely but nowhere differentiable, with $\mathrm{Var}(B_t) = t$, and therefore cannot be drawn exactly at any finite resolution.
```

### Itô Processes and SDEs

An *Itô process* $X_t$ satisfies the stochastic differential equation (SDE):

$$
dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dB_t,
$$ (eq-ito_sde)

where $\mu(\cdot)$ is the *drift* (deterministic trend) and $\sigma(\cdot)$ is the *diffusion coefficient* (volatility). Readers unfamiliar with Itô calculus will benefit from {cite:t}`shreve2004stochasticii` as a prerequisite; the key fact $(dB_t)^2 = dt$ is what forces the second-order term in Itô's lemma below. Two important special cases recur throughout this chapter:

- **Geometric Brownian motion (GBM):** $dS_t = \mu S_t\,dt + \sigma S_t\,dB_t$ (stock prices, GDP).

- **Ornstein–Uhlenbeck (OU) process:** $dZ_t = \eta(\bar{Z} - Z_t)\,dt + \sigma\,dB_t$ (productivity, interest rates). The OU process is mean-reverting with stationary distribution $Z_\infty \sim \mathcal{N}(\bar{Z}, \sigma^2/(2\eta))$. It will model aggregate TFP in the Krusell–Smith economy below.

For discrete income switching, we use a *continuous-time Markov chain*: a labor state $n_t \in \{n_1, n_2\}$ that switches with Poisson intensities $\lambda_1$ (from $n_1$ to $n_2$) and $\lambda_2$ (from $n_2$ to $n_1$), yielding ergodic probabilities $\pi_1 = \lambda_2/(\lambda_1 + \lambda_2)$ and $\pi_2 = \lambda_1/(\lambda_1 + \lambda_2)$.

### Itô's Lemma

```{prf:definition} Itô's Lemma

 Let $X_t$ follow $dX_t = \mu\,dt + \sigma\,dB_t$ and let $f \in C^2$. Then:

$$
df(X_t) = \left[f'(X_t)\,\mu + \tfrac{1}{2}f''(X_t)\,\sigma^2\right]dt + f'(X_t)\,\sigma\,dB_t.
$$ (eq-ito_lemma)
```


The key difference from ordinary calculus is the second-order correction $\tfrac{1}{2}f''\sigma^2\,dt$, which arises because $(dB_t)^2 = dt \neq 0$. The differential algebra rules are: $dt \cdot dt = 0$, $dt \cdot dB_t = 0$, $dB_t \cdot dB_t = dt$.

**Worked example.** For GBM $dX_t = \mu X_t\,dt + \sigma X_t\,dB_t$, applying Itô's lemma to $f(x) = \ln x$ gives $d(\ln X_t) = (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\,dB_t$, so $X_t = X_0\exp[(\mu - \tfrac{1}{2}\sigma^2)t + \sigma B_t]$. The $-\tfrac{1}{2}\sigma^2$ Itô correction explains why the expected log return differs from the expected return for volatile assets.

**Time-dependent version.** For $V(X_t, t)$ where $dX_t = \mu\,dt + \sigma\,dB_t$:

$$
dV = \left[\partial_t V + \mu\,\partial_x V + \tfrac{1}{2}\sigma^2\,\partial_{xx} V\right]dt + \sigma\,\partial_x V\,dB_t.
$$ (eq-ito_multi)

With Poisson jumps of intensity $\lambda$ and jump size $\Delta X$, an additional term $[V(X_{t^-} + \Delta X, t) - V(X_{t^-}, t)]\,dN_t$ appears.

(sec-kfe_theory)=
## The Kolmogorov Forward Equation
The *Kolmogorov forward equation* (KFE), also known as the *Fokker–Planck equation*, describes how the probability density of a stochastic process evolves over time. It was introduced by Kolmogorov (1931) and, independently in the physics literature, by Fokker (1914) and Planck (1917) to describe diffusion of particles. In stationary equilibrium, $\partial_t g = 0$ and the KFE reduces to an elliptic PDE for the cross-sectional density (used throughout {ref}`sec-ct_equilibrium`–{ref}`sec-ct_pinn`); when aggregate shocks make prices time-varying, the parabolic time-dependent form returns and motivates the master-equation reformulation of {ref}`sec-master_eq`.

### Derivation from First Principles

Consider a population of particles, each independently following $dX_t = \mu\,dt + \sigma\,dB_t$ with constant coefficients. If the initial density is $g_0(x)$, what is the density $g(x,t)$ at time $t > 0$?

The derivation proceeds in four steps. (i) For any smooth test function $\varphi(x)$, $\E{\varphi(X_t)} = \int \varphi(x)\,g(x,t)\,dx$. (ii) Differentiate with respect to $t$ and apply Itô's lemma: $\tfrac{d}{dt}\int \varphi\,g\,dx = \int [\mu\,\varphi' + \tfrac{\sigma^2}{2}\varphi'']\,g\,dx$. (iii) Integrate by parts. Take $\varphi$ to be *compactly supported*: this kills the boundary terms cleanly and costs nothing here, since the identity must hold for every such $\varphi$. We obtain $\int \mu\,\varphi'\,g\,dx = -\int \varphi\,\mu\,\partial_x g\,dx$ and $\int \tfrac{\sigma^2}{2}\varphi''\,g\,dx = \int \varphi\,\tfrac{\sigma^2}{2}\partial_{xx}g\,dx$. (The natural-looking assumption $g \to 0$ at $\pm\infty$ is automatic for any continuous density, but it is not by itself enough: what is needed is that the product $g\,\varphi$ vanishes at infinity, which compact support of $\varphi$ delivers for free. Spatial boundaries, e.g. a borrowing constraint, require separate treatment in the next subsection.) (iv) Since the identity holds for all test functions $\varphi$, we obtain:

$$
\boxed{\frac{\partial g}{\partial t}(x,t) = -\mu\frac{\partial g}{\partial x}(x,t) + \frac{\sigma^2}{2}\frac{\partial^2 g}{\partial x^2}(x,t).}
$$ (eq-kfe_const)

The two terms on the right encode competing effects: $-\mu\,\partial_x g$ is *advection* (drift transports the density), and $\tfrac{\sigma^2}{2}\partial_{xx}g$ is *diffusion* (noise spreads the density).

**General form.** For state-dependent coefficients $dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dB_t$, the KFE in *divergence form* is:

$$
\partial_t g = -\partial_x\!\left[\mu(x)\,g(x,t)\right] + \tfrac{1}{2}\partial_{xx}\!\left[\sigma(x)^2\,g(x,t)\right] = -\partial_x J(x,t),
$$ (eq-kfe_general)

where $J = \mu g - \tfrac{1}{2}\partial_x(\sigma^2 g)$ is the probability flux. The identity $\partial_t g + \partial_x J = 0$ is a conservation law: probability is neither created nor destroyed. When $\sigma$ depends on $x$, the diffusion coefficient must be written *inside* the second derivative; in particular, $\tfrac{1}{2}\partial_{xx}[\sigma(x)^2\,g] \neq \tfrac{\sigma(x)^2}{2}\partial_{xx}g$, with the two expressions differing by $(\sigma^2)'\,\partial_x g + \tfrac{1}{2}(\sigma^2)''\,g$. This distinction is invisible in the constant-coefficient case but is what keeps the operator self-adjoint and probability-conserving when $\sigma$ varies.

### Examples

**Heat equation.** With $\mu = 0$ and constant $\sigma$, the KFE reduces to $\partial_t g = \tfrac{\sigma^2}{2}\partial_{xx}g$, the classical heat equation. Starting from a point mass $g(x,0) = \delta(x-x_0)$, the solution is a Gaussian with variance growing linearly: $g(x,t) = (2\pi\sigma^2 t)^{-1/2}\exp[-(x-x_0)^2/(2\sigma^2 t)]$.

**Ornstein–Uhlenbeck process.** For $dX_t = \eta(\bar{x} - X_t)\,dt + \sigma\,dB_t$, the KFE is $\partial_t g = \eta\,\partial_x[(x - \bar{x})\,g] + \tfrac{\sigma^2}{2}\partial_{xx}g$. Setting $\partial_t g = 0$ and trying a Gaussian ansatz $g^\star(x) = (2\pi s^2)^{-1/2}\exp\!\bigl[-(x-\bar x)^2/(2s^2)\bigr]$, the advection and diffusion terms balance when $s^2 = \sigma^2/(2\eta)$, giving the stationary distribution $g^*(x) = \mathcal{N}(\bar{x}, \sigma^2/(2\eta))$: mean reversion concentrates mass around $\bar{x}$, while diffusion spreads it, and the balance produces a Gaussian steady state.

**Reading the SDE qualitatively.** It is worth pausing to read the drift sign by eye, because students are often trained to compute with differential equations but rarely to look at them. When $X_t < \bar{x}$, the drift $\eta(\bar{x} - X_t)$ is positive and pushes $X_t$ upward; when $X_t > \bar{x}$, the drift is negative and pushes $X_t$ downward, hence the name *mean-reverting*. Flipping the sign ($\eta < 0$) gives a *mean-repelling*, unstable process that drifts away from $\bar{x}$ without bound. Closely related cubic SDEs make the same point sharply. The process $dX_t = -X_t^3\,dt + \sigma\,dB_t$ uses the cubic well as a restoring force and is well-posed with a stationary distribution, while $dX_t = +X_t^3\,dt + \sigma\,dB_t$ blows up in finite time, and a naive Euler discretization produces `NaN`s almost immediately. Throughout this chapter the same qualitative reading is what carries intuition over from SDEs to KFEs and HJBs.

### From Physics to Economics: A Continuum of Agents

Consider a continuum of agents $i \in [0,1]$, each independently following $dX_t^i = \mu(X_t^i)\,dt + \sigma(X_t^i)\,dB_t^i$ with independent idiosyncratic Brownian motions $B_t^i$. By the law of large numbers at the population level, the cross-sectional density $g(x,t)$ evolves *deterministically*, even though each individual path is random. The density satisfies the KFE {eq}`eq-kfe_general`.

```{prf:remark} Physical analogy

 Each molecule of air moves randomly (Brownian motion), but the temperature distribution in a room evolves deterministically (heat equation). In economics: each household's income is random, but the wealth distribution evolves deterministically (KFE).
```


**KFE with income switching.** When agent $i$ has wealth $a_t^i$ and income state $n_t^i \in \{n_1, n_2\}$ switching with Poisson intensities, and wealth evolves as $da_t^i = s(a_t^i, n_t^i)\,dt$ with $a_t^i \geq \underline{a}$, the cross-sectional density $g_t(a,n)$ satisfies:

$$
\partial_t g_t(a,n) = -\partial_a\!\left[s(a,n)\,g_t(a,n)\right] - \lambda(n)\,g_t(a,n) + \lambda(\hat{n})\,g_t(a,\hat{n}),
$$ (eq-kfe_econ)

where $\hat{n}$ denotes the complementary income state. The three terms represent: flow of agents along the wealth axis (savings), agents leaving state $n$, and agents entering state $n$ from $\hat{n}$.

**Boundaries and mass points.** At the borrowing constraint $a = \underline{a}$, the no-flux boundary condition $s(\underline{a},n)\,g_t(\underline{a},n) = 0$ prevents the absolutely continuous density from flowing below $\underline{a}$. If households are constrained at this boundary, a boundary atom may be needed in addition to the interior density; finite-difference implementations represent that atom as mass in the first grid cell. This is the continuous-time counterpart of the characteristic left spike observed in empirical wealth distributions.

**When does the KFE become stochastic?** With purely idiosyncratic shocks (Aiyagari), the KFE is deterministic. When *aggregate shocks* are present (Krusell–Smith), prices $r_t, w_t$ depend on the aggregate state $Z_t$, making the drift stochastic and the density $g_t$ a stochastic process adapted to the filtration $\mathcal{F}_t^0$ generated by the aggregate Brownian motion. This is the "master equation" setting discussed in {ref}`sec-master_eq`.

```{prf:remark} KFE validation: more than the pointwise PDE residual

 A small pointwise KFE residual $|R_{\text{KFE}}|$ is a necessary check, but pointwise differentiation of a noisy density $g_\phi$ is fragile and the residual can sit near zero while the implied density is visibly wrong. Notebook `08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch` therefore monitors a small panel of complementary diagnostics: (i) total mass $\sum_n\int g_\phi\,da \to 1$, (ii) no-flux boundary $J(\underline a, n), J(\bar a, n) \to 0$, (iii) aggregate-capital error $|K^{\text{PINN}} - K^{\text{FD}}|$, (iv) CDF distance (Kolmogorov–Smirnov) per income state, and (v) an integrated flux-balance residual that replaces $\partial_a$ with quadrature. In practice, mass and aggregate $K$ converge first; the local KFE residual and the KS distance close last. Reporting all five together makes PINN quality transparent and avoids reading a small pointwise residual as a guarantee of a small density error.
```


(sec-hjb_theory)=
## The Hamilton–Jacobi–Bellman Equation
Individual optimality in continuous-time heterogeneous-agent models is characterized by the HJB equation. This section gives the full five-step derivation from Itô's lemma; readers who saw the motivating overview of the continuous-time heterogeneous-agent setting in {ref}`sec-ct_hank` can treat the present section as its formal counterpart.

**The agent's problem.** Agent $i$ with state $x_t^i = (a_t^i, n_t^i)$ chooses consumption $c_t^i$ to maximize:

$$
\max_{\{c_t^i\}} \;\E{\int_0^\infty e^{-\rho t}\,u(c_t^i)\,dt}
$$ (eq-hjb_agent)

subject to $da_t^i = (wn_t^i + ra_t^i - c_t^i)\,dt$ and $a_t^i \geq \underline{a}$, where $u(c) = c^{1-\gamma}/(1-\gamma)$ is CRRA utility, $\rho > 0$ is the discount rate, and $(r,w)$ are factor prices. The standard transversality condition $\lim_{t\to\infty} e^{-\rho t}\,V(a_t^i, n_t^i) = 0$ along optimal paths, the continuous-time no-Ponzi requirement, is imposed throughout.

**The value function.** The value function $V(a,n)$ records the maximum expected discounted utility starting from state $(a,n)$. It is increasing and concave in $a$, and $V(a,n_2) > V(a,n_1)$ when $n_2 > n_1$.

**Deriving the HJB.** The derivation follows five steps; each unpacks one ingredient of equation {eq}`eq-hjb_full` below.

*Step (i): Dynamic programming principle.* For small $h > 0$,

```{math}
:enumerated: false

V(a,n) \;=\; \max_c\Big\{\int_0^h e^{-\rho t}\,u(c_t)\,dt \;+\; e^{-\rho h}\,\E{V(a_h, n_h)}\Big\}.
```

This is Bellman's principle of optimality applied to a continuous-time problem.

*Step (ii): Itô's lemma between income jumps.* Conditional on the income state $n_t$ not changing on $[0, h]$, wealth follows the *deterministic* ODE $da_t = s(c_t, a_t, n_t)\,dt$ (there is no Brownian forcing on wealth in this model), so

```{math}
:enumerated: false

dV = V'(a,n)\,s(c,a,n)\,dt.
```

The second-order Itô correction $\tfrac{1}{2}V''\sigma^2\,dt$ vanishes because the wealth diffusion is zero between jumps; this is the most subtle step and is what distinguishes the income-switching HJB from a standard diffusion HJB.

*Step (iii): Account for Poisson jumps in expectation.* Adding the Poisson-jump contribution and taking expectations,

```{math}
:enumerated: false

\E{dV} = \Big[V'(a,n)\,s + \lambda(n)\bigl(V(a,\hat{n}) - V(a,n)\bigr)\Big]\,dt,
```

where $\lambda(n)$ is the intensity of switching out of state $n$ and $\hat{n}$ is the complementary state.

*Step (iv): Substitute into the DPP and let $h \to 0$.* Plugging the expectation back into the Bellman expression, dividing by $h$, and taking $h \to 0$ yields a flow equation in which the discount $\rho V$ on the left balances the flow utility plus expected change on the right.

*Step (v): Optimize over $c$.* Imposing the first-order condition over consumption gives the HJB equation:

$$
\boxed{\rho V(a,n) = \max_c\left\{u(c) + V'(a,n)\cdot(wn + ra - c) + \lambda(n)\!\left(V(a,\hat{n}) - V(a,n)\right)\right\}.}
$$ (eq-hjb_full)

**Interpretation.** The HJB is an *asset pricing equation*: the left side $\rho V$ is the "required return" on the value function (discount rate times "asset value"), and the right side is the "total return" consisting of the flow dividend $u(c)$, the capital gain $V' \cdot s$ from saving, and the expected gain/loss $\lambda(n)(\Delta V)$ from income switching.

**Optimal policy.** The first-order condition $u'(c^*) = V'(a,n)$ gives the consumption function:

$$
c^*(a,n) = \left[V'(a,n)\right]^{-1/\gamma}.
$$ (eq-hjb_foc)

Substituting back eliminates the maximization, yielding a nonlinear PDE in $V$. The savings function is $s(a,n) = wn + ra - c^*(a,n)$.

**Boundary conditions.** At the borrowing constraint $a = \underline{a}$, consumption must keep the drift feasible: $s(\underline{a},n)=wn+r\underline{a}-c \geq 0$, or equivalently $c \leq wn+r\underline{a}$. The boundary HJB is therefore the constrained maximization

```{math}
:enumerated: false

\begin{aligned}
\rho V(\underline{a},n)
&=\max_{0<c\leq wn+r\underline{a}}
\Big\{u(c)+V_a(\underline{a},n)(wn+r\underline{a}-c)\\
&\qquad\qquad\qquad\qquad
+\lambda(n)\bigl(V(\underline{a},\hat n)-V(\underline{a},n)\bigr)\Big\}.
\end{aligned}
```

For CRRA utility this gives the boundary policy $c^*(\underline{a},n)=\min\{[V_a(\underline{a},n)]^{-1/\gamma},\,wn+r\underline{a}\}$ and $s(\underline{a},n)\geq 0$. This is the state-constraint form of the borrowing limit; numerical solvers usually impose it with one-sided derivatives, a constrained policy rule, or a penalty on negative boundary drift.

**Boundary atoms in the stationary distribution.** When the borrowing constraint binds on a positive mass of agents, the stationary measure is not absolutely continuous with respect to Lebesgue measure: it carries a Dirac atom at $a=\underline a$. The decomposition is $g(a,n) = g_{\mathrm{ac}}(a,n) + \alpha(n)\,\delta(a-\underline a)$, where $g_{\mathrm{ac}}$ is the absolutely-continuous interior density and $\alpha(n)\geq 0$ is the constrained mass for income state $n$. In the sense of distributions (acting on smooth test functions against which the atom is integrated), equation {eq}`eq-kfe_econ` remains valid for the full measure $g$. The explicit split into a PDE for the interior part $g_{\mathrm{ac}}$ and a separate flux-balance equation for $\alpha(n)$ (equating inflows from the no-flux boundary condition with the income-driven outflow back into the interior) is the *numerical* device used by finite-difference and PINN implementations, which cannot resolve a delta on a regular grid. Finite-difference implementations typically represent $\alpha(n)$ as the mass in the first grid cell, and PINN implementations either absorb the atom implicitly into a smooth density approximation (with corresponding accuracy loss near $\underline a$) or explicitly parameterize $\alpha(n)$ alongside the interior network.

**Variant: continuous (diffusion) income.** A natural variant replaces the two-state Poisson process with a continuously distributed earnings state $z_t$ following an Ornstein–Uhlenbeck diffusion, $dz_t = \eta(\bar z - z_t)\,dt + \sigma\,dB_t^z$ (with idiosyncratic Brownian motion $B_t^z$). The agent's state is then $(a,z)$, the value function $V(a,z)$ is smooth in both arguments, and Itô's lemma along the $z$-diffusion produces a genuine second-order term. After substituting the first-order condition $c^\star=(V_a)^{-1/\gamma}$ the HJB becomes the elliptic PDE

$$
\rho V(a,z) = u(c^\star) + V_a\,(wz + ra - c^\star) + \eta(\bar z - z)\,V_z + \tfrac{1}{2}\sigma^2\,V_{zz},
$$ (eq-hjb_diffusion)

with the borrowing-constraint state condition $s(\underline a,z)\geq 0$ at $a=\underline a$, a Neumann (FOC) condition $V_a(\bar a,z)=u'(w z + r\bar a)$ on the truncated upper wealth boundary, and reflecting conditions $V_z=0$ at the $z$-boundaries. This is the smallest model in the chapter that carries the diffusion second-order term $\tfrac{1}{2}\sigma^2 V_{zz}$ in the *individual* HJB: the income-switching HJB {eq}`eq-hjb_full` has none, because wealth carries no Brownian forcing between jumps, while the master equation {eq}`eq-master_eq` carries $\tfrac{1}{2}(\sigma^z)^2 V_{zz}$ only for the *aggregate* TFP state $z$.

(sec-ct_equilibrium)=
## Competitive Equilibrium: Huggett and Aiyagari
The HJB gives individual optimal behavior for given prices; the KFE tracks the resulting distribution. Market clearing pins down prices, closing the system. {numref}`fig-hjb_kfe_market_loop` summarizes this fixed-point structure.

### The Coupled HJB–KFE System

```{figure} figures/fig-hjb_kfe_market_loop.svg
:name: fig-hjb_kfe_market_loop

Stationary continuous-time heterogeneous-agent equilibrium as a coupled HJB–KFE–market-clearing loop. Given prices, the HJB determines optimal savings; the KFE maps that policy into a stationary density; aggregating the density updates capital, labor, and therefore prices.
```

The solution method for the stationary equilibrium (no aggregate shocks) iterates: guess $r$ $\to$ solve HJB for $V, c^*$ $\to$ solve KFE for $g^*$ $\to$ compute $K = \sum_n \int a\,g^*\,da$ $\to$ update $r$ from the firm FOC. Under standard regularity conditions on preferences, technology, and the income process, aggregate capital supply $K^s(r)$ is monotone decreasing in $r$ over the relevant range, which makes the bisection (or fixed-point iteration) on $r$ well-posed; see {cite:t}`achdou2022income` [§2] for the full statement.

### Huggett (1993): Endowment Economy

Huggett's model is an endowment economy with idiosyncratic income $y_t \in \{y_1, y_2\}$, a single risk-free bond $b_t \geq \underline{b}$ paying instantaneous return $r$, and bonds in zero net supply. The HJB is $\rho V = \max_c\{u(c) + V_b(rb + y - c) + \lambda(y)(\Delta V)\}$, and the KFE determines the stationary density $g^*(b,y)$. An equilibrium return $r^*$ is a value satisfying the asset-market-clearing condition $\sum_y \int b\,g^*(b,y)\,db = 0$; uniqueness requires standard monotonicity assumptions on aggregate bond demand {cite:p}`achdou2022income`.

The mechanism is that the risk-free rate is pinned down by *precautionary self-insurance demand*, not by a production FOC: agents desire to hold positive bond positions for precautionary reasons, and the return must adjust downward until bond demand equals zero net supply.

### Aiyagari (1994): Production Economy

Aiyagari's model adds a representative firm with Cobb–Douglas production $Y = K^\alpha L^{1-\alpha}$. The household asset is now a claim on aggregate capital, and the firm's FOCs yield $r = \alpha K^{\alpha-1}L^{1-\alpha} - \delta$ and $w = (1-\alpha)K^\alpha L^{-\alpha}$. The equilibrium condition is $K^s(r) = K^d(r)$, where $K^s(r) = \sum_z \int a\,g^*(a,z)\,da$ is capital supplied by households (via HJB + KFE) and $K^d(r)$ is capital demanded by the firm (inverse of the firm FOC). {numref}`tab-huggett_aiyagari_ct` highlights the key distinction between the endowment and production versions.

**What changes from Huggett to Aiyagari?** Both models share the same household problem (HJB) and the same cross-sectional law of motion (KFE). What differs is the equilibrium concept. In Huggett the equilibrium is the single price $r^\star$ that clears a zero-net-supply bond market, $\sum_y\int b\,g^\star(b,y)\,db = 0$; the rate is pinned down by precautionary self-insurance demand alone, with no production side. In Aiyagari the equilibrium is a fixed point in $r$ that matches household supply $K^s(r)$ to firm demand $K^d(r) = (\alpha/(r+\delta))^{1/(1-\alpha)} L$, with prices determined jointly by household savings and the firm FOCs. Numerically, both reduce to a one-dimensional root-finding problem in $r$, but the economic mechanism (precautionary saving vs. marginal-product-of-capital pinning) is qualitatively different.

**Adding aggregate TFP.** When aggregate TFP $Z_t$ is allowed to vary (e.g., the OU process introduced in {ref}`sec-master_eq` below for Krusell–Smith), the firm FOCs generalize to $r_t = \alpha\, e^{Z_t} K_t^{\alpha-1}L_t^{1-\alpha} - \delta$ and $w_t = (1-\alpha)\, e^{Z_t} K_t^\alpha L_t^{-\alpha}$. Aiyagari is recovered when $Z_t \equiv 0$; the same expression covers both calibrations, which is convenient for the master-equation analysis below.

````{table}
:name: tab-huggett_aiyagari_ct

Huggett and Aiyagari as two continuous-time incomplete-markets benchmarks. Huggett clears a zero-net-supply bond market by adjusting the bond return; Aiyagari clears a positive capital market with prices pinned down by firm first-order conditions.

|  | **Huggett (1993)** | **Aiyagari (1994)** |
|---|:---:|:---:|
| Economy | Endowment | Production ($Y = K^\alpha L^{1-\alpha}$) |
| Asset | Bond $b$ | Capital claim $a$ |
| Net supply | Zero ($\int b\,g = 0$) | Positive ($\int a\,g = K > 0$) |
| Price determined by | Bond return $q$ | Interest rate $r$, wage $w$ |
| Wealth distribution | Centered around $0$, mass at $\underline{b}$ | Right-skewed, long right tail |
````

{numref}`fig-huggett_aiyagari_densities` contrasts the two stationary densities visually. In Huggett (left), bonds are in zero net supply, so the cross-sectional density is centred near $b\!=\!0$, with a Dirac atom at the borrowing limit $\underline{b}$ carried entirely by constrained low-productivity households; high-productivity households are smoothly distributed and never bind. In Aiyagari (right), agents hold positive capital in equilibrium, so the same atom now sits at $\underline{a}\!=\!0$ but the bulk of the mass is shifted right with a long upper tail.

```{figure} figures/fig-huggett_aiyagari_densities.svg
:name: fig-huggett_aiyagari_densities

Stationary cross-sectional densities $g^*$ in the two benchmarks, by productivity type $n\in\{n_1,n_2\}$ (low and high). In both economies, only the constrained low-productivity type $n_1$ supports a Dirac atom at the borrowing constraint (blue spike): high-productivity households are not bound. *Left:* Huggett, bonds with limit $\underline{b}=-2$ and zero net supply, so the bulk of mass sits around $b=0$. *Right:* Aiyagari, capital with limit $\underline{a}=0$ and positive aggregate $K$, shifting the unconstrained mass to the right with a long upper tail. The blue spikes visualize the Dirac atom $\alpha(n_1)\,\delta(a-\underline a)$ in the decomposition $g = g_{\mathrm{ac}} + \alpha(n)\,\delta(a-\underline a)$ introduced in the boundary-atom paragraph above. These curves are schematic TikZ illustrations of the qualitative contrast (zero-net-supply bonds versus positive capital), not direct exports; the exact densities depend on calibration and boundary treatment.
```

**Connection to HANK.** These models are the building blocks of Heterogeneous Agent New Keynesian (HANK) models {cite:p}`kaplan2018monetary`, which replace the representative agent in New Keynesian frameworks with an Aiyagari economy. Adding nominal rigidities allows monetary policy to affect consumption *heterogeneously*: agents near the borrowing constraint have high marginal propensities to consume and respond strongly to fiscal stimulus, while wealthy agents absorb shocks through savings.

(sec-ct_pinn)=
## A PINN Solver for the Stationary Huggett–Aiyagari System
The stationary equilibrium of {ref}`sec-ct_equilibrium` couples five conditions: the HJB equation for the value function $V(a,z)$, the consumption first-order condition $c^\star(a,z) = \big(V'(a,z)\big)^{-1/\gamma}$, the KFE for the stationary density $g(a,z)$ with savings drift $s(a,z) = wz + ra - c^\star(a,z)$, the no-flux boundary condition $s(\underline a, z)\,g(\underline a, z) = 0$ at the borrowing constraint $\underline a$, mass normalization $\sum_z\int g(a,z)\,da = 1$, and the market-clearing condition that pins down prices. The traditional solver iterates a fixed point over $r$ (guess $r$ $\to$ solve HJB for $V, c^\star$ $\to$ solve KFE for $g$ $\to$ aggregate capital $\to$ update $r$ from the firm FOC); a PINN instead trains all of them jointly.

A PINN implementation uses two networks, trained jointly:

- $\hat{V}_\theta(a,z)$: approximates the value function. Its derivative $\hat{V}_a$ is computed via automatic differentiation, and the implied consumption policy is $\hat c^\star(a,z) = \big(\hat{V}_a(a,z)\big)^{-1/\gamma}$.

- $\hat{g}_\psi(a,z)$: approximates the stationary density, with a positivity transform (e.g., softplus or $\exp$) ensuring $\hat{g} > 0$.

The joint loss has four components:

$$
\ell = w_{\mathrm{HJB}}\,\ell_{\mathrm{HJB}} + w_{\mathrm{KFE}}\,\ell_{\mathrm{KFE}} + w_{\mathrm{flux}}\,\ell_{\mathrm{flux}} + w_{\mathrm{mass}}\left(\sum_z\int \hat{g}_\psi(a,z)\,da - 1\right)^{\!2},
$$ (eq-ct_loss)

where $\ell_{\mathrm{HJB}}$ and $\ell_{\mathrm{KFE}}$ are mean squared PDE residuals computed on collocation points, $\ell_{\mathrm{flux}}$ enforces the no-flux boundary conditions $s(\underline a, z)\,\hat g_\psi(\underline a, z) = 0$, and the mass term enforces normalization. The integral is evaluated numerically (quadrature or Monte Carlo on the collocation points). The aggregate-flux identity $\sum_z s(a,z)\,g(a,z) = 0$ at each interior $a$ – which follows from the no-flux boundary and total-mass conservation within each $a$-slice – is a free consistency check at solution time and is sometimes added as an auxiliary loss term.

Balancing the four loss components is critical: the HJB and KFE residuals can differ by orders of magnitude, so adaptive loss balancing (ReLoBRaLo, {ref}`ch-nas`) is strongly recommended. Practical considerations include using separate learning-rate schedules for the two networks, concentrating collocation points near the borrowing constraint where the density can become sharply peaked, and verifying that the consumption policy $\hat c^\star$ satisfies $\hat c^\star > 0$ everywhere. If the true stationary distribution contains an atom at the borrowing constraint (as in both Huggett and Aiyagari, {numref}`fig-huggett_aiyagari_densities`), a continuous density network alone cannot represent it; one must add a separate boundary-mass variable or use a discretization that permits a point mass.

```{prf:remark} Why a Fourier basis is a poor test-function set

 A natural-looking alternative to pointwise (collocation) residuals is to project the KFE residual onto a set of test functions $\{\psi_n\}$ and minimize the resulting projection coefficients. The pitfall is that for the obvious choice of a Fourier basis, $\psi_n(a) = e^{i 2\pi n a / L}$, the coefficients $c_n$ of any smooth $g$ decay to zero by the Riemann–Lebesgue lemma; the smoother $g$ is, and the more it vanishes near the boundary, the faster the decay. In practice, beyond $n \approx 5$ the $c_n$ are essentially zero *regardless* of how well the residual is solved: they no longer measure error, they only measure smoothness. Hat functions, low-order polynomials on local patches, or learned neural test functions (the WAN, or weak-adversarial-networks, idea) probe the residual far more honestly.
```


This continuous-time PINN approach and the discrete-time DEQN with Young's method ({ref}`ch-young`) address the *same* economic question – how heterogeneous agents interact through prices when markets are incomplete – but differ in the mathematical formulation summarized in {numref}`tab-discrete_continuous_ha`.

````{table}
:name: tab-discrete_continuous_ha

Discrete- and continuous-time formulations of incomplete-markets heterogeneous-agent models. The economic object is the same in both columns, but the numerical residual changes from a discrete-time law of motion plus Euler equation to a coupled HJB–KFE PDE system.

|  | **Discrete time (Ch. {ref}`ch-young`)** | **Continuous time (this chapter)** |
|---|---|---|
| Distribution | Histogram $G_t(k_i, \varepsilon_j)$ | Density $g(a,z)$ |
| Evolution | Young's redistribution | KFE PDE |
| Individual opt. | Euler equation | HJB PDE |
| Solver | DEQN (TensorFlow) | PINN (PyTorch) |
| Key reference | {cite:t}`krusell1998income` | {cite:t}`achdou2022income` |
````

Both approaches can incorporate aggregate shocks, multiple assets, and general equilibrium; the choice between them depends primarily on whether the underlying economic model is formulated in discrete or continuous time. For the aggregate-shock case the natural continuous-time object is the *master equation* ({ref}`sec-master_eq`), solved with EMINNs ({ref}`sec-eminn`).

(sec-master_eq)=
## The Master Equation
When aggregate TFP $Z_t$ follows an OU process $dZ_t = \eta(\bar{Z} - Z_t)\,dt + \sigma^z\,dB_t^0$ (with mean-reversion speed $\eta>0$; not the OLG TFP shock or the network learning rate of earlier chapters), the value function depends on the *entire wealth distribution*: $V = V(a, n, z, g)$. The HJB becomes a PDE with a functional argument, and the "curse of infinite dimensionality" strikes: the functional derivative $\delta V/\delta g$ makes the problem intractable by standard methods. The master equation approach, which originated in the mean field games literature {cite:p}`lasry2007mean` and is developed systematically in the monograph of {cite:t}`cardaliaguet2019master`, reformulates this coupled HJB–KFE system as a single PDE on the extended state space $(a, n, z, g)$ that retains the distribution argument explicitly but is amenable to neural network approximation.

**Why collapse the coupled system?** Without aggregate shocks, solving the stationary equilibrium as a fixed point in $r$ over the coupled HJB–KFE system is straightforward ({ref}`sec-ct_equilibrium`). With aggregate shocks, however, every realization of $Z_t$ would in principle require its own coupled solve, and no parametric guess for $V(a,n,z,g)$ can be informed by the cross-section unless the cross-section is treated as an explicit argument. The master equation reformulation lifts $g$ into the state space so a *single* PDE in $(a,n,z,g)$ encodes the entire economy, which makes a global neural-network ansatz feasible ({ref}`sec-eminn`). The price of this convenience is the appearance of the functional-derivative term $\delta V/\delta g$, which the rest of this section unpacks.

**The master equation.** The key idea is to substitute the KFE, market clearing, and belief consistency *directly into* the HJB, collapsing the coupled system into a single PDE:

$$
\begin{split}
0 &= -\rho V(a,n,z,g) + u(c^*) + V_a\left[w(z,g)n + r(z,g)a - c^*\right] \\
&\quad + \lambda(n)\!\left(V(a,\hat{n},z,g) - V(a,n,z,g)\right) + V_z\,\mu^z(z) + \tfrac{1}{2}(\sigma^z)^2 V_{zz} \\
&\quad + \sum_{j} \int_{\underline{a}}^{\infty} \frac{\delta V}{\delta g_j}(a,n,z,g)(y)\;\mu^g_j(y,z,g)\,dy,
\end{split}
$$ (eq-master_eq)

where $\mu^g_j$ is the KFE drift computed from the optimal savings policy, and the last line encodes how changes in the distribution affect the value function through prices. The kernel $\delta V/\delta g_j(y)$ is the infinite-dimensional analogue of a gradient: it measures how the value function at the individual state $(a,n,z)$ responds to an infinitesimal redistribution of probability mass to wealth level $y$ in the cross-section; the remark below makes this precise. Mass conservation guarantees $\int_{\underline a}^{\infty}\mu^g_j(y,z,g)\,dy = 0$ (the KFE flux integrates to zero under no-flux boundary conditions), so the integrand of the last line pairs $\delta V/\delta g_j$ with a mean-zero test perturbation in exactly the sense required by the functional-derivative remark below.

**The envelope condition.** Following {cite:t}`gu2024masterequations`, it is more convenient to work with $W(a,n,z,g) := \partial_a V(a,n,z,g)$, which directly gives the consumption policy via $c^* = W^{-1/\gamma}$. The master equation for $W$ takes the form:

$$
0 = (r(z,g) - \rho)\,W + \underbrace{\mathcal{L}_x W}_{\text{individual}} + \underbrace{\mathcal{L}_z W}_{\text{aggregate TFP}} + \underbrace{\mathcal{L}_g W}_{\text{distribution}},
$$ (eq-master_W)

where $\mathcal{L}_x W$ captures individual state dynamics (savings, income switching), $\mathcal{L}_z W = \partial_z W \cdot \mu^z + \frac{1}{2}(\sigma^z)^2\partial_{zz}W$ captures TFP dynamics, and $\mathcal{L}_g W = \sum_n \int (\delta W/\delta g_n)\cdot\mu^g_n\,dy$ captures distribution dynamics.

```{prf:remark} One equation to rule them all

 The master equation subsumes the HJB, KFE, and market clearing into a *single* PDE. Its advantage is conceptual clarity and amenability to neural network approximation (via EMINNs). Its challenge is infinite dimensionality: the argument $g$ is a measure, requiring finite-dimensional approximation.
```


````{prf:remark} A user's note on the functional derivative $\delta W/\delta g$

 The objects $\delta V/\delta g$ and $\delta W/\delta g$ deserve a moment of attention because they are the only piece of mathematics in this chapter that is not standard PDE calculus. $V$ takes *a function* (the density $g$) as one of its arguments; differentiating $V$ with respect to $g$ therefore returns not a number but again a function. (Strictly speaking, the Fréchet derivative is the linear functional $\zeta \mapsto \int (\delta V/\delta g)\,\zeta\,dy$ itself, and the kernel $\delta V/\delta g$ is what is properly called the *functional* or *variational* derivative. We follow common usage and call the kernel the Fréchet derivative below.) Concretely, if $g_\varepsilon = g + \varepsilon\,\zeta$ for some test perturbation $\zeta(y)$ with $\int \zeta\, dy = 0$, then

```{math}
:enumerated: false

\frac{d}{d\varepsilon}\, V(\cdot, g_\varepsilon)\Big|_{\varepsilon=0}
  \;=\; \int \frac{\delta V}{\delta g}(y)\,\zeta(y)\,dy.
```

The kernel $\delta V/\delta g$ is the *Fréchet* (or *linear functional*) derivative of $V$ in the $g$-argument; it measures how the value of an agent at a given $(a,n,z)$ responds to an infinitesimal redistribution of mass at point $y$ in the cross-section. In the master-equation literature this object is more precisely the *Lions derivative* $\partial_\mu V$ with respect to the mean-field measure $\mu$; when $\mu$ admits a density $g$ it reduces to the functional derivative used here. The infinite-dimensionality of the master equation is the price of this extra argument; the EMINN approach in the next section makes it tractable by replacing $g$ with a finite-dimensional surrogate $\hat\varphi$ and then differentiating through $\hat\varphi$ via the chain rule and standard automatic differentiation. For the underlying mean-field-games calculus see {cite:t}`cardaliaguet2019master`.
````


(sec-eminn)=
## EMINNs: Solving the Master Equation with Deep Learning
Economic Model Informed Neural Networks (EMINNs), introduced by {cite:t}`gu2024masterequations`, solve the master equation by (i) approximating the infinite-dimensional distribution $g$ by a finite-dimensional object $\hat{\varphi}$, and (ii) parameterizing $W$ by a neural network trained to minimize the master equation residual. A teaching-scale companion notebook for EMINNs is forthcoming; in the present edition, the master-equation discussion is text-only and the Aiyagari notebook ([`lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb`](notebooks/lecture_13_continuous_time_ha_numerics/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb)) is the closest computational reference.

(subsec-three_approx)=
### Three Approximation Approaches for the Distribution
The infinite-dimensional distribution $g$ must be replaced by a finite-dimensional approximation $\hat{\varphi} \in \R^d$ so that $V(a,n,z,g) \approx \hat{V}(a,n,z,\hat{\varphi})$. {numref}`tab-eminn_distribution_approximations` summarizes the three approaches used in {cite:t}`gu2024masterequations`.

````{table}
:name: tab-eminn_distribution_approximations

Finite-dimensional representations of the cross-sectional distribution in EMINNs. Here $\delta_{\bullet}$ denotes a Dirac measure centered at $\bullet$, not the depreciation rate of {ref}`ch-deqn`–{ref}`ch-young`.

|  | **Finite population** | **Discrete state** | **Projection** |
|---|:---:|:---:|:---:|
| $\hat{\varphi}_t$ | $\{(a_t^i, n_t^i)\}_{i=1}^N$ | Masses on grid $\{a_1,\ldots,a_I\}$ | Basis coefficients |
| $\hat{g}_t$ | $\frac{1}{N}\sum_i \delta_{\hat{\varphi}_t^i}$ | $\sum_{i,j} \hat{\varphi}_{ij}\,\delta_{(a_i,n_j)}$ | $b_0 + \sum_i \hat{\varphi}_i b_i(a,n)$ |
| Dimension | $\sim 40$ | $\sim 200$ | $\sim 5$ |
````

**Finite population ($N \approx 40$).** Replace the continuum by $N$ agents with states $\hat{\varphi}_t = \{(a_t^1, n_t^1),\ldots,(a_t^N,n_t^N)\}$. Aggregate capital is $K_t = N^{-1}\sum_i a_t^i$. Sampling individual states and distribution states *separately* during training keeps $N$ manageable; the law of large numbers provides accurate aggregate capital even with 40 agents.

**Discrete state ($\sim$200 grid points).** Discretize wealth on a grid $\{a_1,\ldots,a_I\}$ and represent the distribution as masses $\hat{\varphi}_{i,j}$ at each $(a_i, n_j)$. The KFE becomes a finite-difference mass evolution, and the functional derivative becomes a partial derivative $\partial_{\hat{\varphi}_{m,j}}\hat{W}$.

**Projection ($\sim$5 components).** Project $g$ onto eigenfunctions of the steady-state KFE operator $\bar{\mathcal{L}}^{KF}$. These are the most *persistent* density components, carrying the most price-relevant information. Only $\sim$5 basis functions suffice, yielding the lowest-dimensional representation, but the setup requires computing eigenfunctions and choosing appropriate test functions for the KFE evolution.

### The EMINN Algorithm

A neural network $\hat{W}_\Theta(\omega)$ with $\omega = (a, n, z, \hat{\varphi})$ parameterizes the marginal value of wealth. The output uses a softplus activation to ensure $\hat{W} > 0$, and consumption follows directly from the envelope condition: $c^* = \hat{W}^{-1/\gamma}$. {prf:ref}`alg-eminn` gives the resulting residual-minimization loop.

```{prf:algorithm} EMINN training for the master equation
:label: alg-eminn

- **Input:** Initial parameters $\Theta$, tolerance $\varepsilon$, loss weights $\kappa^e$, $\kappa^s$; initialize $\mathcal{E}=\infty$
- **while** $\mathcal{E} > \varepsilon$ **do**
  - Sample $M$ collocation points: $(a_m, n_m, z_m, \hat{\varphi}_m)_{m=1}^M$
  - Compute distribution-drift coefficients $\mu_{\hat\varphi,n}(\hat\varphi_m)$ (method-specific; see \S\ref{subsec:three_approx} and remark below)
  - Compute master equation residual $\hat{\mathcal{L}}(\cdot;\Theta)$ via automatic differentiation through $\hat\varphi$
  - Compute shape penalty $\mathcal{E}^s(\cdot;\Theta)$ (concavity, monotonicity)
  - Total loss: $\mathcal{E} = \kappa^e\,M^{-1}\sum_m |\hat{\mathcal{L}}_m|^2 + \kappa^s\,\mathcal{E}^s$
  - Update: $\Theta \leftarrow \Theta - \alpha_{\mathrm{opt}}\,\nabla_\Theta \mathcal{E}$ \;*($\alpha_{\mathrm{opt}}$: optimizer learning rate; distinct from the OU mean-reversion $\eta$ of \S\ref{sec:master_eq})*
- **end**
```

This is precisely a PINN applied to the master equation: the "physics" is the economic equilibrium structure, and all derivatives, including those with respect to the distribution parameters $\hat{\varphi}$, are computed by automatic differentiation.

The master equation residual decomposes as:

$$
0 = (r(z,\hat{\varphi}) - \rho)\,\hat{W} + \hat{\mathcal{L}}_x\hat{W} + \hat{\mathcal{L}}_z\hat{W} + \hat{\mathcal{L}}_g\hat{W},
$$

where $\hat{\mathcal{L}}_x\hat{W} = s(\cdot)\partial_a\hat{W} + \lambda(n)(\hat{W}(\hat{n}) - \hat{W}(n))$ captures savings and income switching, $\hat{\mathcal{L}}_z\hat{W} = \partial_z\hat{W}\cdot\mu^z + \frac{1}{2}(\sigma^z)^2\partial_{zz}\hat{W}$ captures the OU aggregate shock, and $\hat{\mathcal{L}}_g\hat{W} = \sum_n (\partial\hat{W}/\partial\hat{\varphi}_n)\cdot\mu_{\hat{\varphi},n}$ captures distribution evolution.

**Computing the drift coefficients $\mu_{\hat{\varphi},n}$.** The coefficients $\mu_{\hat{\varphi},n}$ describing how the finite-dimensional approximation $\hat{\varphi}$ of the cross-sectional density evolves in time are not given for free. Recovering them is a genuine algorithmic step, and it is the place where the three approximation choices in Section {ref}`subsec-three_approx` differ most sharply:

- **Finite-population approximation.** $\hat{\varphi}$ is the empirical measure of a finite particle system, so $\mu_{\hat{\varphi},n}$ is the SDE drift of particle $n$, available in closed form from the underlying individual problem.

- **Discrete-state (grid) approximation.** $\hat{\varphi}_{i,j}$ is the mass at $(a_i, n_j)$; the discretized KFE evaluated at that node returns $\mu_{\hat{\varphi},(i,j)}$ directly. This is also where an upwind scheme reappears (see the remark after {eq}`eq-cake_expand_V`): the side of $\partial_a$ used in the discretization is selected by the sign of the drift $s$.

- **Projection / basis expansion.** $\hat{\varphi}(a) = \sum_n c_n \psi_n(a)$. The KFE returns $\partial_t g$ as a function of $a$; one then recovers $\mu_{\hat{\varphi},n} = \dot c_n$ by least-squares projection of $\partial_t g$ onto $\{\psi_n\}$. This is the only step of EMINN that genuinely depends on the choice of approximation, and it is where most of the practical art lives.

Concretely, between the forward pass and the residual evaluation in {prf:ref}`alg-eminn`, an additional sub-step computes $\mu_{\hat{\varphi},n}$ for the current $\hat{\varphi}$; the chain rule through $\hat{\varphi}$ then propagates these coefficients into $\hat{\mathcal{L}}_g\hat{W}$.

### Shape Constraints and Training Stability

A key practical challenge in training EMINNs is that neural networks may converge to "cheat solutions" (constant, non-increasing, or non-concave value functions that produce small residuals but are economically meaningless). Shape penalties are added to the loss to enforce economic structure:

- **Concavity:** penalize non-concavity (i.e., violations of the standard $V_{aa}\leq 0$ condition) via $\mathcal{E}^{\text{concav}} = M^{-1}\sum_m \max(0, \partial_{aa}V_m)^2$, which is positive only when $\partial_{aa}V > 0$ at some collocation point.

- **Monotonicity:** in calibrations where higher TFP lowers the marginal value of liquid wealth, penalize violations of the model-specific restriction $\partial_z V_a < 0$.

- **Initialization:** set $W(a,\cdot) = e^{-a}$ as a simple decreasing initial marginal-value profile.

- **Architectural encoding:** if the chosen formulation has known boundary asymptotics for $V$ or $W$, multiply the network by an appropriate boundary factor rather than asking the optimizer to learn the singular shape from scratch.

- **Active sampling:** concentrate collocation points where the residual is large.

### Results and Method Comparison

For the Aiyagari model (no aggregate shocks), {cite:t}`gu2024masterequations` report master-equation residuals of order $10^{-5}$ and mean squared errors against finite-difference benchmarks of order $10^{-5}$, with close agreement in consumption policies, marginal value functions, and stationary distributions across the finite-population and discrete-state approaches.

For the Krusell–Smith model (with aggregate shocks), their reported results show master-equation training losses of order $10^{-5}$ across all three approximation approaches and similar time paths for aggregate variables (capital, interest rate, wage). In that experiment, the projection approach achieves the lowest reported loss ($8.5 \times 10^{-6}$) with only 5 distribution parameters, while the finite-population approach ($3.0 \times 10^{-5}$) offers the simplest implementation.

{numref}`tab-ct_method_comparison` gives the practical method comparison for this chapter.

````{table}
:name: tab-ct_method_comparison

Finite differences, PINNs, and EMINNs for the continuous-time heterogeneous-agent problems covered in this chapter. The entries are practical guidance, not universal impossibility results: finite differences remain the benchmark in low dimension, while EMINNs target the global master-equation setting with the distribution as a state.

|  | **Finite differences** | **PINN** | **EMINN** |
|---|---|---|---|
| Stationary equilibrium | Benchmark method | Works, validate carefully | Works, but overkill |
| Aggregate shocks | Local/low-dimensional only | Coupled system, not full master equation | Designed for global master equation |
| Grid required | Yes | No state grid | No state grid |
| High-dimensional scaling | Limited by grids | Better, optimization-limited | Better, distribution-state approximation needed |
| Handles $g$ as state | Not in standard stationary FD | No | Yes, through $\hat{\varphi}$ |
| Convergence theory | Strong for low-dimensional monotone schemes | Limited | Limited |
| Low-dimensional accuracy | Often $\sim 10^{-6}$ | Problem-dependent; validate | Reported $\sim 10^{-5}$ |
````

For stationary, low-dimensional problems, finite differences remain fast and reliable and should be used as the benchmark. For models with aggregate shocks and high-dimensional state spaces, EMINNs are, among the methods surveyed here, one of the few approaches demonstrated on global master-equation solutions for this class of benchmarks. PINNs serve as a useful intermediate step: they share the same code philosophy as EMINNs (automatic differentiation of PDE residuals) but apply to the coupled HJB–KFE system rather than the master equation.

```{prf:remark} Chapter Summary

- In continuous time, the heterogeneous-agent equilibrium is a coupled HJB + KFE system, recast as a mean field game in the sense of Lasry–Lions.

- Closing with aggregate shocks gives the master equation: a single PDE in $(x, g)$ where the second argument is a measure. EMINNs solve it by finite-dimensional approximation of $g$ and a neural-network ansatz for the value function.

- The functional derivative $\delta V/\delta g$ is the only piece of mathematics not standard in PDEs; it measures sensitivity to infinitesimal cross-section perturbations.

- Achdou et al. (2022) finite differences and EMINNs are complementary: FD wins in low dimension, while EMINNs are designed for aggregate-shock master-equation models.
```


## Further Reading {.unnumbered}
- {cite:t}`achdou2022income`, the canonical continuous-time HA reference.

- {cite:t}`gu2024masterequations`, the EMINN paper.

- {cite:t}`lasry2007mean` {cite}`carmona2018probabilistic,cardaliaguet2019master`, mean-field-games foundations.

- {cite:t}`shreve2004stochasticii`, stochastic-calculus textbook.

- Moll's online lecture notes (<https://benjaminmoll.com/lectures/>), pedagogical complement.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch8-1

**[Core\] Itô on GBM.** Apply Itô's lemma to $f(x) = \ln x$ with $X_t$ following geometric Brownian motion to derive $X_t = X_0 \exp[(\mu - \tfrac{1}{2}\sigma^2)t + \sigma B_t]$. Discuss why "volatility drag" lowers the expected log return below the expected return.
```

```{exercise}
:label: ex-ch8-2

**[Core\] KFE for an OU process.** Write the Kolmogorov forward equation for the Ornstein–Uhlenbeck process $dX_t = \eta(\bar X - X_t)\,dt + \sigma\,dB_t$ and derive the stationary density $\mathcal{N}(\bar X, \sigma^2/(2\eta))$ by setting $\partial_t g = 0$.
```

```{exercise}
:label: ex-ch8-3

**[Core\] Functional derivative.** For the master-equation value function $V(a, g)$, compute $\delta V/\delta g$ for the toy specification $V(a,g) = \int u(c(a, y))\,g(y)\,dy$ where $c$ is a fixed consumption rule. Interpret the result.
```

```{exercise}
:label: ex-ch8-4

**[Computational\] HJB residual.** In notebook [`lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb`](notebooks/lecture_13_continuous_time_ha_numerics/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb), compute the maximum HJB residual on a 50-point test grid after a fixed PINN training budget. Repeat with a larger collocation batch and report the actual scaling you observe.
```

```{exercise}
:label: ex-ch8-5

**[Core\] Closed Aiyagari system in continuous time.** Combine the HJB equation {eq}`eq-hjb_full` for $V(a,n)$ and the KFE {eq}`eq-kfe_econ` for $g(a,n)$ into the closed Aiyagari general-equilibrium system. (i) Add the firm's first-order conditions: with Cobb–Douglas production $Y = A K^\alpha L^{1-\alpha}$, write $r = \alpha A K^{\alpha-1} L^{1-\alpha} - \delta$ and $w = (1-\alpha) A K^\alpha L^{-\alpha}$. (ii) Add the market-clearing conditions $K = \sum_n\int_{\underline a}^{\infty} a\,g(a,n)\,da$ and $L = \sum_n n\int_{\underline a}^{\infty}g(a,n)\,da$. (iii) Show that the coupled system pins down $(V,g,K,L,r,w)$, with $L$ often fixed by the stationary income shares and $w$ implied by $(K,L)$. (iv) Briefly explain why solving this system for the stationary equilibrium is commonly reduced to a fixed point in $r$, and why the inner HJB and KFE problems must be solved consistently at each candidate $r$.
```

```{exercise}
:label: ex-ch8-6

**[Advanced/project\] Stationary-distribution convergence.** In notebook [`lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb`](notebooks/lecture_13_continuous_time_ha_numerics/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb), take the optimal policy $s^\star(a,n)$ from the converged HJB solver (i.e., fix the policy) and run the KFE {eq}`eq-kfe_econ` forward in time starting from a non-stationary initial density (e.g., uniform on $[\underline a, a_\mathrm{max}]$). Plot snapshots of $g_t(a, n)$ at $t \in \{1, 5, 25, 100, 500\}$ years and the running $L^2$ distance $\|g_t - g^\star\|_{L^2}$ to the stationary distribution $g^\star$. Check whether convergence is approximately exponential by fitting a line to $\log\|g_t - g^\star\|$ over the linear region, and relate the fitted rate to the spectral gap of the KFE operator.
```

```{exercise}
:label: ex-ch8-7

**[Advanced/project\] FD upwind vs. PINN benchmark.** In the same notebook, the chapter ships both a finite-difference upwind solver (deterministic, grid-based) and a PINN trained on HJB and KFE residuals. Run both on the same $1$-asset Aiyagari calibration and report (i) wall-clock time for your chosen residual target, (ii) the final residual on a fine $200$-point test grid, and (iii) peak memory, including GPU memory if you run the PINN on a GPU. As an extension, sweep the discount rate $\rho$ over $\{0.04, 0.05, 0.06\}$ and compare cold-start FD runs with warm-started PINN runs. Discuss when each is preferred: FD for low-dimensional, fixed-parameter computations; PINN for parametric sweeps and higher-dimensional extensions.
```

[^1]: <https://benjaminmoll.com/lectures/>
