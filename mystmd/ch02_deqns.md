---
title: "Deep Equilibrium Nets"
label: ch-deqn
---

This chapter introduces Deep Equilibrium Nets (DEQNs), the central computational framework of this script. Rather than generating labeled training data, a DEQN embeds the equilibrium conditions of a dynamic stochastic model (Euler equations, budget constraints, complementarity conditions) directly into the loss function of a neural network. The network then learns policy functions by minimizing the violation of these conditions via gradient descent. We develop the methodology on the classical Brock–Mirman growth model, for which an analytical solution exists and serves as a benchmark. The primary reference is {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`.

**Notation transition.** {ref}`ch-intro` used $\bm{\theta}$ for the parameter vector of a generic supervised model, following the textbook convention. From this chapter onward we switch to $\rho$ (so that the network is denoted $\mathcal{N}_\rho(\x)$), matching the notation of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` and most of the subsequent DEQN literature. The two symbols denote the same object; we use $\rho$ for parameters wherever a network appears, and reserve $\bm{\theta}$ for structural economic parameters that the network does not see (Chs. {ref}`ch-estimation`–{ref}`ch-climate`).

(sec-curse_of_dim)=
## The Curse of Dimensionality in Economics
A central challenge in computational economics is that many models of interest, such as overlapping-generations (OLG) economies, heterogeneous-agent models, and international business cycle models, feature state spaces of high dimension. In an OLG model with $N$ cohorts and idiosyncratic risk, the state vector may include individual capital holdings, productivity levels, and wealth shares for each cohort, leading to state spaces of dimension $d \gg 10$. Traditional grid-based solution methods, such as value function iteration on a Cartesian grid with $n$ nodes per dimension, require $n^d$ grid points; this is the *curse of dimensionality*, the term coined by {cite:t}`bellman1961adaptive` to describe the exponential growth of computational cost with the number of state variables. A concrete back-of-the-envelope makes the bottleneck plain: at the modest resolution of $n=100$ nodes per dimension, a $d=5$ state space already requires $100^5 = 10^{10}$ grid points, which is infeasible under any reasonable compute budget; even more modest resolutions of $n=20$–$30$ become prohibitive once $d$ exceeds ten.

{numref}`tab-curse_of_dim` illustrates the curse concretely. Even with a modest $n=10$ grid points per dimension, the total number of grid points grows astronomically.

````{table}
:name: tab-curse_of_dim

Size of an $n = 10$ Cartesian grid and the 64-bit memory required to store one floating-point value per grid point, as a function of state-space dimension $d$. Grid-based methods are comfortable only at low dimension; by $d = 10$ even storing one scalar per grid point is already borderline before policies, interpolation objects, residuals, and simulation output are added.

| $d$ | Grid points ($10^d$) | Memory (64-bit) | Feasibility |
|---:|---:|---:|---|
| 1 | 10 | 80 B | trivial |
| 2 | 100 | 800 B | trivial |
| 5 | $10^5$ | 800 KB | feasible |
| 10 | $10^{10}$ | 80 GB | borderline |
| 20 | $10^{20}$ | $8 \times 10^{11}$ GB | impossible |
| 50 | $10^{50}$ | — | absurd |
| 100 | $10^{100}$ | — | a googol |
````

**The volume paradox.** A complementary, and more geometric, perspective on the curse of dimensionality starts from a single ratio. The volume of the unit $d$-ball relative to the smallest cube that contains it, $[-1,1]^d$, is

$$
\frac{V_\mathrm{ball}(d)}{V_\mathrm{cube}(d)} = \frac{\pi^{d/2}}{2^d\,\Gamma(d/2+1)}.
$$ (eq-volume_ratio)

This ratio collapses with $d$ ({numref}`fig-volume_paradox`): it is $\pi/4 \approx 0.785$ at $d=2$ and $\pi/6 \approx 0.524$ at $d=3$, has fallen to $\approx 2.5\times 10^{-3}$ by $d=10$, and is below $10^{-70}$ at $d=100$. The geometry behind the numbers is sharp. The inscribed ball touches the center of each of the $2d$ faces, at distance $1$ from the center, but its surface never gets within $\sqrt{d}-1$ of any of the $2^d$ corners, which sit at distance $\sqrt{d}$. As $d$ grows the corners march off to infinity while the ball stays put, so essentially all of the cube's volume drains into its corners, far from the center. A box-shaped object in high dimensions is, to a first approximation, all corners and no middle.

Why should an economist care about a fact about cubes and balls? Because solving a dynamic stochastic model means producing a *global* solution[^1], and the region over which that solution actually has to be accurate is the model's *ergodic set*, the states the economy visits under its own law of motion. That set is typically a thin, curved, low-volume sliver of any hypercube $[\x_{\min},\x_{\max}]^d$ that one would draw around it, often well under one percent of the bounding box already at moderate $d$ {cite:p}`judd2011numerically`. Yet most of the high-dimensional approximation machinery on offer, tensor-product grids, adaptive sparse grids, tensor-product quadrature, is built on the hypercube; by the volume paradox an exponentially growing fraction of those nodes lands in corners the model never visits, so most of the computational effort is spent representing the function where it does not matter. {numref}`fig-ergodic_vs_grid` visualizes this mismatch. This is exactly the waste the DEQN approach is designed to avoid: instead of tiling a box, it trains the network on states sampled from the model's own simulated trajectories, putting the approximation effort where the economics lives.

Distance itself is the second casualty of high dimensions, and it is worth flagging now because it returns later with teeth. Draw a point uniformly from the unit $d$-ball: its radius has cumulative distribution $r^d$ and density $d\,r^{d-1}$ on $[0,1]$, which for large $d$ piles essentially all of its mass against the shell $r\to 1$. Random points sit on the pulp, not in the core, and they do so with overwhelming probability. Relatedly, pairwise Euclidean distances among random points concentrate so tightly that the nearest and the farthest neighbor of a query point become almost equidistant, so "distance" loses most of its power to discriminate {cite:p}`aggarwal2001surprising` ({numref}`fig-distance_concentration`). This is not a curio: any method that judges similarity through $\|\x-\x'\|$, from $k$-nearest-neighbors to kernel ridge regression to the Gaussian-process surrogates of {ref}`ch-gp`, whose RBF and Matérn kernels are functions of $\|\x-\x'\|$ alone, loses resolution as $d$ grows. It is one of the reasons {ref}`ch-gp` reaches for dimension reduction, active subspaces and deep kernels, before fitting a GP in a high-dimensional input space.

```{figure} figures/fig-volume_paradox.svg
:name: fig-volume_paradox

\(a\) the unit ball inscribed in $[-1,1]^d$

\(b\) ratio {eq}`eq-volume_ratio` versus $d$ (log scale)

The volume paradox behind the curse of dimensionality. *(a)* The largest ball that fits inside the cube $[-1,1]^d$ touches the center of every face, at distance $1$ from the center, but its surface stays $\sqrt{d}-1$ away from each of the $2^d$ corners, which lie at distance $\sqrt{d}$. As $d$ grows the corners recede while the ball does not, so almost all of the cube's volume ends up in the corners (tinted), far from the center. *(b)* The ball-to-cube volume ratio of equation {eq}`eq-volume_ratio` on a logarithmic scale: $\pi/4$ at $d=2$, about $2.5\times10^{-3}$ at $d=10$, and below $10^{-70}$ at $d=100$. A grid or quadrature rule built on the bounding hypercube therefore spends an exponentially growing share of its nodes in corners that the model's ergodic set never reaches.
```

```{figure} figures/fig-ergodic_vs_grid.svg
:name: fig-ergodic_vs_grid

Grid-based vs. simulation-based state sampling. A Cartesian grid (left) allocates effort uniformly over the hypercube and places most of its $n^d$ points far from the model's *ergodic set*, the small subset of the state space that the economy actually visits under its own dynamics (red band). The DEQN algorithm samples states along simulated trajectories (right), concentrating training points exactly on that set. As $d$ grows, the fraction of the cube reached by the ergodic set shrinks rapidly, so the grid's relative waste grows exponentially.
```

```{figure} figures/fig-distance_concentration.svg
:name: fig-distance_concentration

\(a\) blue core and red shell each hold half a $d$-ball's volume; the shell thins as $d$ grows

\(b\) the radius CDF piles up at $r\to1$ as $d$ grows

Why a "random" point in high dimensions lives on the shell, not in the core. *(a)* For each $d$, the blue inner disk and the red outer shell each hold half of the $d$-ball's volume; the shell's fractional thickness $1-(1/2)^{1/d}$ shrinks from $\approx0.29$ at $d=2$ to $\approx0.07$ at $d=10$ to $\approx0.014$ at $d=50$, so half of the mass crowds into an ever thinner rim near the surface. *(b)* Equivalently, the radius of a uniform draw has cumulative distribution $r^{d}$, which for large $d$ stays near zero until $r$ is almost $1$ and then jumps: the radius is essentially deterministic at $1$. The companion fact is that pairwise Euclidean distances among random points concentrate, so a query point's nearest and farthest neighbors become nearly indistinguishable {cite:p}`aggarwal2001surprising`; this is why distance-based methods, including the Gaussian-process kernels of {ref}`ch-gp`, lose resolution in high dimensions.
```

## From Supervised Learning to Self-Supervised Equilibrium Solving

A word on terminology before we start, recalling the taxonomy of {ref}`sec-supervised_vs_unsupervised`. A learning method is *supervised* when its loss compares the network's output to an externally given target label; it is *unsupervised* when there are no target labels at all (clustering, density estimation, dimension reduction); and it is *self-supervised* when, although no human-provided labels exist, the method manufactures its own supervisory target out of the available structure. Deep Equilibrium Nets are unsupervised in the narrow "no labels" sense, but the sharper and more useful description is *equation-based self-supervision*: at every state where we evaluate the model, the equilibrium conditions tell us what the correct relationship between today's policy and tomorrow's must be, namely a residual of zero, and that residual plays exactly the role a label plays in supervised learning. The DEQN objective introduced below is therefore a genuine regression loss, only against a target the model itself dictates rather than one handed to us as data.

With that in mind, the key conceptual shift introduced by {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` is to replace the labeled training data of standard supervised learning with the structural equations of the economic model. In a generic dynamic stochastic economic model, the state of the economy at time $t$ is summarized by a vector $\x_t \in \R^d$, and agents choose policy (or decision) variables $p_t = p(\x_t) \in \R^m$. The equilibrium is characterized by a system of functional equations

$$
G\bigl(\x_t,\, p(\x_t),\, \mathbb{E}_t[H(\x_{t+1}, p(\x_{t+1}))]\bigr) = 0, \qquad \forall\, \x_t,
$$

where $G$ encodes optimality conditions (e.g., Euler equations) and $\mathbb{E}_t[\cdot]$ is the conditional expectation over next-period shocks via the transition law $\x_{t+1} = T(\x_t, p(\x_t), \varepsilon_{t+1})$. The fundamental challenge is that this is a *functional equation*: we seek functions $p:\R^d \to \R^m$ rather than finite-dimensional parameter vectors. The DEQN approach parameterizes $p$ as a neural network and solves this functional equation via stochastic optimization ({numref}`fig-supervised_vs_deqn`).

```{figure} figures/fig-supervised_vs_deqn.svg
:name: fig-supervised_vs_deqn

Supervised learning (left) versus DEQN training (right). Both paradigms train a parameter vector by minimizing a residual loss with SGD; the difference is the *source* of the training signal, labeled data $(\x_i, y_i)$ in the supervised case, structural equilibrium equations $G(\x, p(\x)) = 0$ in the DEQN case. No labeled solution data are required for DEQNs. For Brock–Mirman the right-hand side specializes to $G(K, z) = 1 - \beta\,(C/C') \cdot \alpha z' K'^{\alpha-1}$ with $C = \mathcal{N}_\rho(K, z)$ and $K' = z K^\alpha - C$; the network learns the policy by driving the squared mean of this residual to zero on simulated trajectories.
```

In the standard (supervised) paradigm, one minimizes the discrepancy between the network's predictions and known target values. In the DEQN framework, the loss function is instead the squared norm of the equilibrium residual:

$$
\boxed{
\ell_\rho = \frac{1}{N} \sum_{i=1}^{N} \bigl\| G\bigl(\x_i,\, \mathcal{N}_\rho(\x_i)\bigr) \bigr\|^2,
}
$$ (eq-deqn_loss)

where $\mathcal{N}_\rho$ denotes a deep neural network with parameters $\rho$ that maps states to policies. If the residual loss is zero on a sufficiently rich set of states and the conditional expectations inside $G$ are evaluated accurately (rather than with a single shock realization), the network satisfies the equilibrium conditions on those states; global accuracy must always be assessed on independent validation trajectories and, where available, against benchmark solutions. The continuous-time analogue is the residual-loss formulation of {cite:t}`raissi2019physics`, which inspired the PINN literature developed in {ref}`ch-pinn`. The training points $\x_i$ are drawn not from an exogenous dataset, but from the model's own simulated dynamics: the network learns on the ergodic distribution of the economy, concentrating computational effort on economically relevant regions of the state space. The idea of using simulated paths as the support for projection goes back to the parameterized-expectations algorithm of {cite:t}`marcet1988pea` and {cite:t}`marcet_marshall:94`, which the DEQN literature reframed as ergodic-set sampling to focus computation on economically relevant states; the numerical stability of stochastic-simulation projection was further developed by {cite:t}`judd2011numerically`.

**Connection to the modern overparameterized regime.** Note that DEQNs operate squarely in the "modern regime" of {ref}`sec-generalization`: the network often has $p \gg n$ parameters relative to the mini-batch size $n$. Unlike supervised learning with a fixed dataset, however, the training distribution is *renewable*: each episode can generate fresh state samples from the model's own simulation. This changes the fixed-dataset double-descent analogy, but it does not remove the need for validation: $p$ can still exceed the effective sample budget seen during training, and a flexible network can fit residuals on a narrow simulated region. The practical lesson is to combine overparameterized networks and SGD with independent validation trajectories, held-out Euler residuals, and, where available, closed-form benchmarks.

(sec-deqn_algo)=
## The DEQN Training Algorithm
```{prf:definition} Algorithm: Deep Equilibrium Net Training

- **Input:** Initial state $\x_0$, network $\mathcal{N}_\rho$, learning rate $\eta$, episodes $E$, simulation horizon $T_{\mathrm{sim}}$, training steps $T_{\mathrm{train}}$, expectation rule $\mathcal{Q}$ (path-average or quadrature)
- **[NEW]** Burn-in: draw the first episode's states from a broad prior (uniform box around the deterministic steady state) and maintain a small replay buffer of early states.
- **[NEW]** Independent validation trajectory $\x^{\mathrm{val}}_{0:T_{\mathrm{val}}}$ for out-of-sample residual diagnostics, simulated once with frozen $\rho$ at each checkpoint.
- **for** episode $e = 1, \ldots, E$ **do**
  - **Simulate path:** $\x_0 \to \x_1 \to \cdots \to \x_{T_{\mathrm{sim}}}$ using $\mathcal{N}_\rho$ and transition law; guard infeasible states (e.g.\ clip $C\le 0$).
  - **for** gradient step $t = 1, \ldots, T_{\mathrm{train}}$ **do**
    - Draw mini-batch $\mathcal{B} \subset \{\x_0, \ldots, \x_{T_{\mathrm{sim}}}\} \cup \mathrm{replay}$
    - Compute loss:~$\ell_\rho = \frac{1}{|\mathcal{B}|}\sum_{\x_i \in \mathcal{B}} \|G(\x_i, \mathcal{N}_\rho(\x_i); \mathcal{Q})\|^2$ **[NEW: $\mathcal{Q}$ is the chosen path-average or Gauss–Hermite / monomial / QMC rule]**
    - Update:~$\rho \leftarrow \rho - \eta \cdot \nabla_\rho \ell_\rho$ **\[NEW: wrap the per-step kernel in `@tf.function` / `torch.compile` / `@jax.jit` for $5$–$50\times$ speed-up]**
  - **end**
- **end**
- **Output:** Trained network $\mathcal{N}_{\rho^\star}$ approximating the policy function; report Euler residuals on $\x^{\mathrm{val}}$.
```


Several features of this algorithm deserve emphasis:

- **Endogenous training distribution.** As the policy network improves, the simulated paths become more accurate, and the training points concentrate on the ergodic set of the true equilibrium. This is fundamentally different from supervised learning, where the training distribution is fixed. Early in training, however, the simulated distribution is generated by a poor and rapidly changing policy: it can be unstable, infeasible, or far too narrow. Practical implementations therefore (i) burn in with broad sampling from the prior or a uniform box around the deterministic steady state, (ii) clip or guard against infeasible states (e.g. negative consumption), and (iii) often maintain a small replay buffer so that early states are revisited as the policy improves. Out-of-sample validation on an independent trajectory remains essential.

- **No labeled data.** The algorithm requires no "ground truth" solutions, only the structural equations of the model. The loss function has direct economic interpretation: it measures the violation of equilibrium conditions.

- **Stochastic optimization.** Mini-batch SGD naturally handles the stochasticity of the model: different mini-batches sample different states, providing implicit exploration of the state space.

- **Scalability.** The DEQN avoids explicit tensor-product state grids entirely: the per-iteration cost scales with network size, batch size, the number of equations, and the cost of integrating the conditional expectation; the input/output layer widths and the simulation step also depend on the state and shock dimensions, but only linearly. This favorable empirical scaling does not eliminate all high-dimensional costs (sample complexity, expectation accuracy, optimization difficulty), but it does mitigate the practical grid-based curse of dimensionality.

- **JIT-compile every gradient step.** In any production or classroom-scale implementation, the per-batch gradient step (lines 5–6 inside the inner loop) should be wrapped in `@tf.function` (TensorFlow), `torch.compile` (PyTorch), or `@jax.jit` (JAX). The speed-up over an un-traced Python loop is typically $5$–$50\times$ for the per-step kernel, and the trace cost is amortized after the first call. All companion notebooks in this course follow this convention; un-traced Python loops are simply too slow for classroom use, let alone for research-scale runs.

**Episode length $T_{\mathrm{sim}}$.** The simulated-trajectory length $T_{\mathrm{sim}}$ controls the effective size of the training set within an episode. {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` use $T_{\mathrm{sim}} = 10{,}000$ time steps (split into many short mini-batches) in their 113-dimensional 56-agent OLG benchmark. Shorter episodes (e.g., $T_{\mathrm{sim}} = 1000$) speed up early training because the distribution is re-drawn more frequently, at the cost of higher variance; longer episodes produce smoother gradient estimates but run the risk of stale data if the policy has drifted during training. A pragmatic rule is to start short, lengthen $T_{\mathrm{sim}}$ as the loss plateaus, and check out-of-sample accuracy on an *independent* simulated trajectory.

**Connection to continuous-time methods.** The DEQN residual loss is the discrete-time analogue of the Deep Galerkin Method (DGM) of {cite:t}`sirignano2018dgm`, which minimizes a PDE residual on neural-network-parameterized PDE solutions in continuous time, and of the deep BSDE solver of {cite:t}`han2018solving` (Han–Jentzen–E), which formulates the same problem as a backward stochastic differential equation. Within the discrete-time deep-learning toolkit, {cite:t}`maliar2021deep` unify lifetime-reward, Bellman-equation, and Euler-equation training into a single framework, paired with an "all-in-one" integration operator that estimates all conditional expectations from a single Monte Carlo realization; DEQN as developed here is the Euler-equation branch of that taxonomy. {ref}`ch-pinn` and {ref}`ch-ct_theory` develop the continuous-time analogues explicitly; the present chapter should be read as introducing the discrete-time machinery on which those continuous-time methods are built.

(sec-bm)=
## The Brock–Mirman Benchmark
Having laid out the generic training loop, we now put it to work on a concrete model. To validate the DEQN methodology, {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` begin with the stochastic growth model of {cite:t}`brock1972optimal`, which admits a closed-form solution and therefore serves as an ideal test case.

The social planner solves:

$$
\max_{\{C_t,\, K_{t+1}\}_{t=0}^{\infty}} \E{\sum_{t=0}^{\infty} \beta^t \ln(C_t)}
\quad\text{s.t.}\quad
K_{t+1} + C_t = z_t K_t^\alpha, \quad \ln z_{t+1} = \varrho \ln z_t + \sigma_z \epsilon_{t+1},
$$ (eq-bm_problem)

where $\beta \in (0,1)$ is the discount factor, $\alpha \in (0,1)$ the capital share, $\varrho \in [0,1)$ the shock persistence, $\sigma_z > 0$ the shock volatility, and $\epsilon_{t+1} \sim \mathcal{N}(0,1)$ i.i.d. innovations. Depreciation is full ($\delta = 1$).

**Derivation of the Euler equation.** To derive the optimality conditions, form the Lagrangian by attaching a multiplier $\beta^t \lambda_t$ to the resource constraint at each date $t$:

$$
\mathcal{L} = \E{\sum_{t=0}^{\infty} \beta^t \Bigl[\ln(C_t) + \lambda_t \bigl(z_t K_t^\alpha - C_t - K_{t+1}\bigr)\Bigr]}.
$$ (eq-bm_lagrangian)

Since the planner chooses $C_t$ and $K_{t+1}$ at each date, we take partial derivatives with respect to each:

**FOC w.r.t. $C_t$:**

$$
\frac{\partial \mathcal{L}}{\partial C_t} = \beta^t\!\left(\frac{1}{C_t} - \lambda_t\right) = 0
\qquad\Longrightarrow\qquad
\lambda_t = \frac{1}{C_t}.
$$ (eq-bm_foc_c)

**FOC w.r.t. $K_{t+1}$:** The variable $K_{t+1}$ appears in the date-$t$ constraint (with coefficient $-1$) and in the date-$(t\!+\!1)$ constraint via output $z_{t+1}K_{t+1}^\alpha$. Differentiating:

$$
\frac{\partial \mathcal{L}}{\partial K_{t+1}}
= \beta^t\bigl(-\lambda_t\bigr) + \beta^{t+1}\,\mathbb{E}_t\!\bigl[\lambda_{t+1}\,\alpha\, z_{t+1}\, K_{t+1}^{\alpha-1}\bigr] = 0.
$$ (eq-bm_foc_k)

Dividing both sides by $\beta^t$ yields:

$$
\lambda_t = \beta\,\mathbb{E}_t\!\bigl[\lambda_{t+1}\,\alpha\, z_{t+1}\, K_{t+1}^{\alpha-1}\bigr].
$$

Finally, substituting $\lambda_t = 1/C_t$ from {eq}`eq-bm_foc_c` into this expression gives the Euler equation:

$$
\frac{1}{C_t}
= \beta\,\mathbb{E}_t\!\left[
\frac{\alpha\, z_{t+1}\, K_{t+1}^{\alpha-1}}{C_{t+1}}
\right].
$$ (eq-bm_euler)

The economic intuition is transparent: the left-hand side is the marginal utility cost of saving one additional unit today; the right-hand side is the expected discounted marginal utility gain from the extra output $\alpha\, z_{t+1}\, K_{t+1}^{\alpha-1}$ that unit produces tomorrow. At the optimum, the planner is indifferent between consuming and saving at the margin.

**Analytical solution.** When depreciation is full ($\delta = 1$), this model admits a closed-form solution. One can verify by direct substitution into the Euler equation {eq}`eq-bm_euler` that the optimal consumption policy is:

$$
C_t^\star = (1 - \beta\alpha)\, z_t K_t^\alpha.
$$ (eq-bm_analytical)

The derivation proceeds by guessing that the value function takes the form $V(K,z) = V_0 + B\ln K + D\ln z$ (with the constant $V_0$ written as $V_0$ rather than $A$ to avoid clashing with the TFP / cohort-count uses of $A$ elsewhere in the script) and using the envelope theorem. Substituting this guess into the Bellman equation $V(K,z) = \max_C \{\ln C + \beta \E{V(K',z')}\}$ with $K' = zK^\alpha - C$ yields a system of equations for $V_0$, $B$, and $D$. Matching coefficients on $\ln K$ pins down $B = \alpha/(1-\alpha\beta)$; matching on $\ln z$ pins down $D = 1/[(1-\alpha\beta)(1-\beta\varrho)]$ (so $D$, unlike $B$, depends on the shock persistence); and the constant $V_0$ then absorbs the remaining terms. The first-order condition $1/C_t = \beta\,B/K_{t+1}$ together with $C_t + K_{t+1} = z_t K_t^\alpha$ delivers the linear savings rule $K_{t+1} = \beta\alpha\, z_t K_t^\alpha$, from which {eq}`eq-bm_analytical` follows immediately via the resource constraint. Notably, this closed-form solution holds regardless of the shock persistence $\varrho$ for the savings rule itself, because $z_{t+1}$ cancels in the ratio $z_{t+1}/C_{t+1}$ under the linear consumption rule: $C_{t+1} \propto z_{t+1}$, so $z_{t+1}/C_{t+1} = 1/[(1-\beta\alpha) K_{t+1}^\alpha]$ no longer depends on the next-period shock.

**Persistent shocks.** When $\varrho > 0$, productivity shocks are autocorrelated and the state of the economy becomes the pair $(K_t, z_t)$, since the current shock level $z_t$ now carries information about future productivity. Under log utility with Cobb–Douglas production *and full depreciation ($\delta = 1$)*, the analytical solution {eq}`eq-bm_analytical` continues to hold regardless of the persistence $\varrho$. The reason fits in one line: under the linear consumption rule $C_t = (1-\beta\alpha)\,z_t K_t^\alpha$ we have $z_{t+1}/C_{t+1} = 1/[(1-\beta\alpha)K_{t+1}^\alpha]$, so $z_{t+1}$ cancels and the conditional expectation in the Euler equation {eq}`eq-bm_euler` no longer depends on the next-period shock; the constant $D$ in the value function still depends on $\varrho$, but $D$ enters $V$ and cancels in the FOC for the savings rule. However, for more general preferences (e.g., CRRA with $\gamma \neq 1$), partial depreciation ($\delta < 1$), or non-Cobb–Douglas production, the closed-form solution breaks down and numerical methods, and the DEQN approach in particular, become essential: the policy function $C(K_t, z_t)$ must be approximated rather than derived analytically. (Concretely: the deterministic companion notebook uses $\delta = 1$ so that the closed form applies and can be used as a benchmark; the stochastic companion notebook switches to $\delta = 0.1$, where it does not, and the policy is genuinely learned. Only the loss-kernel notebook of {ref}`sec-loss_kernels` returns to $\delta = 1$, precisely so that the closed-form savings rate $s^\star = \alpha\beta$ is available as ground truth.)

**DEQN formulation.** We parametrize the consumption policy as $C_t = \mathcal{N}_\rho(K_t, z_t)$ and define the residual

$$
G(K_t, z_t) = 1 - \beta \, \frac{C_t}{C_{t+1}} \cdot \alpha z_{t+1} K_{t+1}^{\alpha-1},
$$

where $z_{t+1}$ denotes a single realization of the next-period shock and $K_{t+1} = z_t K_t^\alpha - \mathcal{N}_\rho(K_t, z_t)$. In the code listing below the network actually emits a *savings share* $s_t\in(0,1)$ through a sigmoid head rather than $C_t$ directly, which guarantees $C_t>0$ *and* $K_{t+1}>0$ at every training step; {ref}`sec-deqn_hard_soft` explains the hard-vs-soft constraint split this exemplifies. Note that the Euler equation {eq}`eq-bm_euler` involves the conditional expectation $\mathbb{E}_t[\cdot]$, whereas the residual $G$ above is written for a single realization of $z_{t+1}$. Two approaches are common for handling this expectation:

1.  **Path averaging:** draw a single $z_{t+1}$ per state, compute $G$ for that draw, and average $G^2$ over many states along the simulated path. This minimizes $\mathbb{E}_{(K_t,z_t,z_{t+1})}[G^2]$ – the squared *pathwise* residual. It is an unbiased Monte Carlo objective for that stronger loss, and Jensen's inequality gives $\bigl(\mathbb{E}_t[G]\bigr)^2 \leq \mathbb{E}_t[G^2]$. Thus $G=0$ almost surely implies the conditional Euler equation, but the converse need not hold: a policy can have $\mathbb{E}_t[G]=0$ while $G$ varies with the shock. In the log/full-depreciation Brock–Mirman benchmark the closed-form policy happens to drive the pathwise residual itself to zero, so this stronger target is harmless; in richer models it is a modeling choice. This is the approach used in the Brock–Mirman code listing below. Note that the pathwise-residual-zero coincidence is special to $\delta = 1$: for $\delta < 1$ (the calibration of the stochastic notebook) a converged policy has $\mathbb{E}_\varepsilon[G\mid x]=0$ but $G$ itself is nonzero realization-by-realization, so path averaging then minimizes a strictly stronger objective than the conditional Euler residual.

2.  **Quadrature:** for each state $(K_t, z_t)$, approximate $\mathbb{E}_t[\cdot]$ explicitly via deterministic nodes and weights ({ref}`sec-quadrature_rules`), form the residual from the estimated expectation, and then square. This targets $\bigl(\mathbb{E}_t[G]\bigr)^2$ directly and is the approach used for the IRBC model of {ref}`ch-irbc`, where accurate expectations are critical for convergence.

Both approaches recover the analytical solution {eq}`eq-bm_analytical` to high accuracy, providing a rigorous validation of the methodology. Because convergence curves depend on the exact training run, random seed, and solver configuration, this manuscript does *not* include a hand-drawn convergence plot. In practice, one should report diagnostics from the *actual notebook run*: residual trajectories, held-out Euler errors, and the gap between the learned policy and the analytical benchmark. {numref}`fig-bm_convergence_schematic` sketches the qualitative shape one should expect to see.

```{figure} figures/fig-bm_convergence_schematic.svg
:name: fig-bm_convergence_schematic

Schematic, not measured: the qualitative convergence behavior typical of a successful Brock–Mirman DEQN run. An early high-residual phase reflects an untrained network feeling out the state space; a mid phase descends roughly exponentially as the policy locks onto the equilibrium structure; a late phase plateaus near the irreducible quadrature/training-noise floor. The dotted line marks the analytical benchmark below which the residual cannot be driven without higher-precision quadrature. For the actual numbers on a specific seed, consult the companion notebook.
```

**Relative Euler equation error.** Following {cite:t}`judd1998numerical` and {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` (Eq. 43), the standard accuracy diagnostic for a DEQN is the *relative* Euler error, which measures the percentage consumption error implied by a violation of the Euler equation. For Brock–Mirman with log utility, the relative error at state $\x_j = (K_j, z_j)$ is

$$
e^{\mathrm{REE}}_{\x_j}(\rho)
\;=\;
\frac{
  \bigl(\beta\,\mathbb{E}[\alpha\, z' (K_j')^{\alpha-1}/C'_\rho(\x'_j)\mid \x_j]\bigr)^{-1}
}{C_\rho(\x_j)}
\;-\; 1,
$$ (eq-ree_bm)

where $C_\rho = \mathcal{N}_\rho$, $K'_j$ is next-period capital under the network policy, the expectation conditions on $\x_j$, and the $^{-1}$ inverts the marginal-utility relation $u'(c) = 1/c$. The value $e^{\mathrm{REE}} = 10^{-4}$ means the agent's optimal consumption is mispriced by $0.01\%$, independent of units or utility scale. This is the metric reported in Table 3 of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, where the trained DEQN achieves mean relative Euler errors of order $10^{-4}$ on the 113-dimensional 56-agent OLG benchmark.

```{prf:remark} Why does this work so well?

 The success of the DEQN approach rests on three pillars. First, neural networks are universal function approximators that can represent the smooth policy functions arising in most economic models. Second, the training distribution is endogenous: the network learns on the model's own ergodic distribution, concentrating computational effort precisely where it matters. Third, stochastic gradient descent operates directly on the economic equilibrium conditions, so the loss function has a clear economic interpretation: a pointwise relative Euler error of $10^{-4}$ means the consumption level implied by the Euler equation differs from the network's consumption by about $0.01\%$.
```


```{code-block} text
:caption: Representative DEQN loss for Brock--Mirman with path averaging. The network outputs a savings share $s \in (0,1)$ via a sigmoid, which jointly enforces $C > 0$ \emph{and} $K' > 0$; softplus on $C$ alone would not, since $C > z K^\alpha$ would yield $K' < 0$ and an undefined $K'^{\,\alpha-1}$.

def deqn_loss(model, K, z, beta, alpha, z_next):
    Y       = z * K**alpha                   # output today
    s       = model(K, z)                    # savings share in (0,1) via sigmoid
    C       = (1.0 - s) * Y                  # consumption today  (>0)
    K_next  = s * Y                          # capital tomorrow   (>0)
    Y_next  = z_next * K_next**alpha
    s_next  = model(K_next, z_next)
    C_next  = (1.0 - s_next) * Y_next        # consumption tomorrow
    # z_next: single draw; expectation via path averaging
    G = 1.0 - beta * (C / C_next) * alpha * z_next * K_next**(alpha-1)
    return tf.reduce_mean(G**2)
```
A Gauss–Hermite variant (developed formally in {ref}`sec-gh_tensor_product`) replaces the single shock draw `z_next` by a $Q$-node weighted sum $\mathrm{E}[\,\alpha z' K'^{\alpha-1}/C'\mid z\,] \approx \sum_{q=1}^{Q} w_q\,\alpha z_q' K'^{\alpha-1}/C'(K',z_q')$ with $z_q' = z^\varrho\exp(\sigma_z\varepsilon_q)$ at the rescaled Hermite nodes $\varepsilon_q$; in practice $Q=5$ already drives the quadrature error below the training error. The autodiff companion notebook [`03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb`](notebooks/lecture_07_autodiff_for_deqns/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb) implements this variant explicitly.

(sec-deqn_hard_soft)=
## Encoding Equilibrium Conditions: Hard vs. Soft Constraints
The Brock–Mirman DEQN loss above already used one design choice without dwelling on it: the network emits a savings share through a sigmoid rather than consumption directly. This is an instance of a general principle. {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` §4.2.2 do not treat every equilibrium equation symmetrically in the loss {eq}`eq-deqn_loss`; they split the conditions into two groups. We make the split explicit on the Brock–Mirman model of the previous section and then indicate how it generalizes.

Hard constraints, encoded in the architecture (never in the loss).
: Some equations can be satisfied exactly for every state $\x = (K,z)$:

  - The *resource / state-transition* equation $K_{t+1} = z_t K_t^\alpha - C_t$ defines next-period capital from the network's consumption policy. It is *not* minimized; it is evaluated as a closed-form function of $\x_t$ and $C_t$.

  - The economic requirements $C_t > 0$ *and* $K_{t+1} > 0$ are jointly imposed by parameterizing the network's output as a *savings share* $s_t \in (0,1)$ via a *sigmoid* activation, $\mathrm{sigmoid}(z) = 1/(1+e^{-z})$, and recovering both quantities in closed form from the resource constraint, $C_t = (1-s_t)\,z_t K_t^\alpha$ and $K_{t+1} = s_t\,z_t K_t^\alpha$. A softplus head on $C_t$ alone would guarantee $C_t>0$ but not $K_{t+1}>0$ (the network could output $C_t > z_t K_t^\alpha$). This sigmoid-savings parameterization removes an entire class of infeasible candidate policies before training begins; see the code listing in {ref}`sec-bm` above.

Soft constraint, minimized in the loss.
: The only equilibrium condition that cannot be enforced analytically is the *Euler equation* {eq}`eq-bm_euler`. The squared relative Euler error {eq}`eq-ree_bm` is averaged over the mini-batch and driven toward zero by stochastic gradient descent.


This split is pedagogically important for three reasons: (i) only the genuinely non-closed-form conditions enter the loss, which speeds up training; (ii) it eliminates a family of bad local minima in which the network produces, e.g., slightly negative consumption or infeasible states; and (iii) it explains why the loss typically converges to a small but nonzero value even at the optimum, since the Euler residual is intrinsic and cannot be removed by re-parameterization. {numref}`fig-hard_soft` summarizes this construction for Brock–Mirman.

```{figure} figures/fig-hard_soft.svg
:name: fig-hard_soft

Hard vs. soft constraints in the DEQN architecture for Brock–Mirman. The network $\mathcal{N}_\rho$ reads the state $(K_t,z_t)$ and emits a *savings share* $s_t \in (0,1)$ via a *sigmoid* head. Both consumption $C_t = (1-s_t)\,z_t K_t^\alpha$ and next-period capital $K_{t+1} = s_t\,z_t K_t^\alpha$ are then defined in closed form (green, top), guaranteeing $C_t>0$ *and* $K_{t+1}>0$ simultaneously; a softplus head on $C_t$ alone could not, since $C_t > z_t K_t^\alpha$ would push $K_{t+1}<0$ and make $K_{t+1}^{\alpha-1}$ undefined. Only the relative Euler-equation residual (red, bottom) is squared, averaged over the mini-batch, and minimized by SGD. This figure matches the code listing in {ref}`sec-bm` above. In richer models (OLG, {ref}`ch-olg`) the same split extends to firm first-order conditions, household budget constraints, and KKT multipliers with softplus heads and Fischer–Burmeister complementarity residuals.
```

**How the split generalizes.** In models with more equilibrium objects (the OLG benchmark of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, the IRBC model of {ref}`ch-irbc`, the HA economies of {ref}`ch-young`), the same logic extends: firm first-order conditions for prices $(w_t,r_t)$ are closed-form in the aggregate state, the household budget constraint algebraically determines consumption given savings, and each inequality constraint contributes a softplus-headed KKT multiplier paired with a Fischer–Burmeister complementarity residual in the soft part of the loss ({ref}`ch-irbc`). The Brock–Mirman case above is the minimal instance of the pattern.

**Market-clearing layers.** One can push this design philosophy further by encoding the *market-clearing condition itself* into the network as a dedicated output layer rather than minimizing a market-clearing residual in the loss: the network outputs unnormalized cohort savings (or shares) and a final layer rescales them so that aggregate clearing holds by construction. {cite:t}`azinoviczemlicka_2024` use such a "market-clearing layer" in an OLG economy with rare disasters; this is the discrete-time counterpart of the PINN-style practice of baking conservation laws directly into the network output ({ref}`ch-pinn`). Sigmoid heads, softplus heads, Fischer–Burmeister residuals, and market-clearing rescaling layers thus form a small toolbox of architecture-level encodings that each move part of the equilibrium structure from the soft loss into the hard part of the network.

(sec-quadrature_rules)=
## Quadrature Rules for Conditional Expectations
(sec-integration_primer)=
### Why We End Up Integrating, and What Numerical Integration Is
**Why an integral always shows up.** Sooner or later, every dynamic stochastic model in this script presents the same algorithmic bottleneck: a forward-looking agent has to form an expectation over future shocks before any decision can be made. The Euler equation {eq}`eq-bm_euler` hinges on $\mathbb{E}_t[\,\alpha z_{t+1} K_{t+1}^{\alpha-1}/C_{t+1}\,]$; the Bellman operator behind a value-function iteration evaluates $\mathbb{E}[V(\bm{s}')\mid\bm{s},a]$ at every state and action; the relative Euler error {eq}`eq-ree_bm` embeds the same expectation in the diagnostic; the integrated-assessment models of {ref}`ch-climate` require expected discounted utility under climate-shock distributions; and the structural-estimation moments of {ref}`ch-estimation` are themselves expectations over simulated paths. Outside a handful of conjugate or fully-linear-quadratic-Gaussian special cases, none of these expectations admit closed-form expressions, so the integral $\mathbb{E}[g(\bm\varepsilon)] = \int g(\bm\varepsilon)\mu(\bm\varepsilon)\,d\bm\varepsilon$ has to be approximated numerically by a finite, cheap-to-evaluate, ideally differentiable, weighted sum $\sum_q w_q\, g(\bm\varepsilon_q)$. Choosing the nodes $\bm\varepsilon_q$ and weights $w_q$ well is the entire content of *quadrature theory*. The path-averaging recipe used in the Brock–Mirman code above is one such choice (a Monte Carlo sample average with one node per state along a simulated trajectory); the rest of this section makes the deterministic and quasi-random alternatives explicit and explains when each is preferable.

**Picture 1: a definite integral as area, and the Riemann/midpoint sum.** Strip away the economics. A definite integral $\int_a^b f(x)\,dx$ is the (signed) area between the graph of $f$ and the $x$-axis on $[a,b]$. The simplest deterministic numerical rule, the *midpoint rule*, replaces this area by a stack of $N$ rectangles of equal width $\Delta x = (b-a)/N$, each one as tall as $f$ at the midpoint of its base ({numref}`fig-integration_primer`, left). Adding the rectangle areas yields

$$
I_N = \Delta x \sum_{i=1}^{N} f\bigl(x_i^{\text{mid}}\bigr)
\;\xrightarrow[N \to \infty]{}\; \int_a^b f(x)\,dx,
\qquad \text{with error } |I_N - I| = \mathcal{O}(N^{-2})
$$

for smooth $f$. Trapezoid and Simpson rules refine the same idea by replacing each rectangle with a trapezoid or a parabolic arc and reach $\mathcal{O}(N^{-2})$ and $\mathcal{O}(N^{-4})$ respectively; Gauss–Hermite, the workhorse of {ref}`sec-gh_tensor_product`, is the optimal rule when the integrand is multiplied by the Gaussian weight $e^{-x^2}$ and reaches degree-$(2Q-1)$ exactness using only $Q$ nodes. The same tile-the-area idea generalizes to higher dimensions by laying out a Cartesian grid. With $N$ nodes per coordinate the cost is $N^d$; equivalently, with $M=N^d$ total nodes the midpoint rate becomes $\mathcal{O}(M^{-2/d})$. Notice that it is the *exponent* of the rate, not just the constant, that degrades from $-2$ in 1D to $-2/d$ in $d$-D Cartesian, the curse of dimensionality made quantitative. This is the explicit form of the curse of dimensionality flagged in {ref}`sec-curse_of_dim`, and it motivates both the Stroud monomial rules of {ref}`sec-monomial_cubature` and the randomized methods of {ref}`sec-qmc_cdf`.

**Picture 2: throwing darts to estimate $\pi$.** An entirely different idea is to abandon the grid and approximate the integral by a *sample average*: draw $N$ random points $\bm{u}_1, \ldots, \bm{u}_N$ uniformly from the integration domain $\Omega$ and report $I_N = \mathrm{vol}(\Omega)\cdot \tfrac{1}{N}\sum_{i=1}^{N} f(\bm{u}_i)$. This is plain Monte Carlo (MC), and its error decays as $\mathcal{O}(N^{-1/2})$ with a rate that is *independent of the dimension $d$*, although the variance constant can still worsen with dimension. This is why MC dominates deterministic grids when $d$ is large. The cleanest illustration uses the unit square $[0,1]^2$ and the indicator function of a quarter-disc:

$$
\frac{\pi}{4} \;=\; \int_0^1\!\!\int_0^1 \mathbf{1}\bigl[x^2 + y^2 \leq 1\bigr]\, dx\, dy
\;\approx\; \frac{N_{\mathrm{in}}}{N},
\qquad
\widehat{\pi}_N \;=\; 4\cdot \frac{N_{\mathrm{in}}}{N},
$$ (eq-pi_mc)

where $N_{\mathrm{in}}$ counts how many of the $N$ uniform "darts" land inside the quarter-circle $x^2+y^2 \leq 1$ ({numref}`fig-integration_primer`, right). With $N=100$ a typical run gives $\widehat{\pi}_{100} \approx 3.04$ (about 3% off); with $N=10^6$ a typical run gives $\widehat{\pi}_{10^6} \approx 3.1417$ (about $10^{-4}$ off). The error shrinks as $\mathcal{O}(1/\sqrt{N})$, requiring a hundredfold increase in $N$ to gain one extra decimal of accuracy, which is glacially slow compared to $\mathcal{O}(N^{-4})$ for Simpson's rule on a smooth 1D integrand. But the MC *rate* has no dependence on the dimension of the domain: replacing the quarter-disc by a $d$-dimensional unit ball would leave the rate untouched, while the deterministic grid would suffer the $N^d$ cost explosion of the curse of dimensionality. This is what makes MC and its quasi-random refinement (QMC, {ref}`sec-qmc_cdf`) the natural tools for the conditional expectations encountered in DEQNs at $d \gtrsim 10$.

```{figure} figures/fig-integration_primer.svg
:name: fig-integration_primer

Two paradigms for numerical integration that underlie every rule in this section. Deterministic tiling (left) is exact and highly accurate at low dimension but suffers exponential cost growth in $d$. Random sampling (right) is dimension-independent in its error rate but slow to converge in any single dimension. The Gauss–Hermite, monomial, and QMC rules of §{ref}`sec-gh_tensor_product`–{ref}`sec-qmc_cdf` are sophisticated descendants of the two ideas, designed to combine deterministic accuracy with manageable cost in $d$.
```

**Where the rest of this section is going.** With this picture in hand, the design of every quadrature rule in the literature can be read as an answer to two questions: (i) where do we place the nodes $\bm\varepsilon_q$, and (ii) what weights $w_q$ do we attach to them? Tensor-product Gauss–Hermite ({ref}`sec-gh_tensor_product`) places nodes deterministically on a Cartesian grid of Hermite roots; the Stroud-3 monomial rule of {ref}`sec-monomial_cubature` places only $2d$ nodes on the principal axes and accepts a controlled bias on fourth-order moments in exchange for linear-in-$d$ scaling; the QMC construction of {ref}`sec-qmc_cdf` places nodes from a low-discrepancy sequence in the unit cube and pulls them back through the inverse CDF. The cost–accuracy trade-offs differ dramatically with $d$, as {numref}`tab-quadrature_costs` makes concrete. {ref}`ch-irbc` returns to these numbers when scaling DEQNs to multi-country economies.

(sec-gh_tensor_product)=
### Tensor-Product Gauss–Hermite
The classical Gauss–Hermite rule approximates integrals against the weight function $e^{-x^2}$:

$$
\int_{-\infty}^{\infty} f(x)\, e^{-x^2}\, dx \approx \sum_{q=1}^{Q} \tilde{w}_q\, f(x_q),
$$

where the $Q$ nodes $\{x_q\}$ are the roots of the $Q$-th Hermite polynomial $H_Q(x)$ and the weights are $\tilde{w}_q = 2^{Q-1} Q!\sqrt{\pi}/(Q^2 [H_{Q-1}(x_q)]^2)$. For expectations under the standard normal distribution, we apply the change of variables $\varepsilon = \sqrt{2}\, x$ to obtain:

$$
\E{h(\varepsilon)} = \int_{-\infty}^{\infty} h(\varepsilon)\frac{e^{-\varepsilon^2/2}}{\sqrt{2\pi}}\,d\varepsilon \approx \sum_{q=1}^{Q} w_q\, h(\varepsilon_q),
\qquad w_q = \frac{\tilde{w}_q}{\sqrt{\pi}},\quad \varepsilon_q = \sqrt{2}\,x_q.
$$ (eq-gh_1d)

The 1D rule {eq}`eq-gh_1d` is exact for univariate polynomials in $\varepsilon$ of degree at most $2Q-1$, an attractive accuracy guarantee for smooth integrands.

**Worked example: Brock–Mirman Euler expectation.** In the Brock–Mirman model, the conditional expectation in the Euler equation {eq}`eq-bm_euler` is one-dimensional:

$$
\mathbb{E}_t\!\left[
\frac{\alpha\, z_{t+1}\,K_{t+1}^{\alpha-1}}{C_{t+1}}
\right],
\qquad
z_{t+1}=z_t^{\varrho}\exp(\sigma_z\varepsilon_{t+1}),
\quad
\varepsilon_{t+1}\sim\mathcal{N}(0,1).
$$

Replacing path averaging by Gauss–Hermite with $Q=5$ nodes turns the residual {eq}`eq-bm_euler` into a deterministic function of $(K_t, z_t)$ and the network parameters, eliminating the shock noise inside the loss while costing only five extra forward passes per state. This is the simplest concrete instance of the general construction below and a useful sanity check: at $Q=5$ the quadrature error is negligible relative to the training error in this benchmark, so a well-trained network should match the analytical Brock–Mirman policy to numerical tolerance.

For shock dimension $d > 1$, the multivariate expectation is computed via a tensor product of one-dimensional rules. Each multi-dimensional quadrature node is a $d$-tuple $(\varepsilon_1^{q_1}, \ldots, \varepsilon_d^{q_d})$ with weight $\prod_{j=1}^{d} w_{q_j}$. The total number of quadrature nodes grows as $Q^{d}$, which becomes prohibitive once $d$ is large {cite:p}`judd1998numerical`. For $d \geq 10$, the monomial rule of {ref}`sec-monomial_cubature` or *sparse-grid* (Smolyak) alternatives can substantially reduce the computational cost: {cite:t}`gerstner1998numerical` introduce the construction in numerical analysis, {cite:t}`heiss2008likelihood` apply it to econometric likelihoods, and {cite:t}`ECTA:ECTA1716` develop adaptive sparse grids tailored to dynamic-programming and equilibrium computations. The polynomial-versus-exponential gap depends on smoothness assumptions and the effective dimension of the integrand: sparse grids attain polynomial rates for sufficiently regular functions; QMC achieves close-to-$\mathcal{O}(1/M)$ rates under randomized constructions and finite-variation / low-effective-dimension conditions {cite:p}`sobol1967distribution,niederreiter1992random,owen1995randomly,caflisch1998monte,novak2008tractability`, but neither method is universally polynomial.

(sec-monomial_cubature)=
### Monomial (Stroud-3) Cubature with Linear Scaling
The exponential cost $Q^{d}$ of the previous subsection is the price paid for being exact in the highest-degree univariate polynomial along *each* coordinate. In equilibrium-residual applications, however, the integrand $h(\bm\varepsilon')$ obtained by composing the policy network with model primitives is typically only mildly nonlinear in the shock vector, and reproducing every cross-coordinate monomial of total degree up to $2Q-1$ is overkill. Once one is willing to give up that high degree, dramatically cheaper rules become available.

The simplest such rule is the *Stroud degree-3 monomial cubature* {cite:p}`stroud1971approximate`, a cornerstone of the toolkit reviewed in {cite:t}`judd1998numerical` [§7.5] and {cite:t}`Maliar2014325` [§5.3] for large-scale dynamic models. For a $d$-dimensional standard normal expectation it uses only $2d$ nodes, all on the principal axes:

$$
\E{h(\bm\varepsilon')} \;\approx\; \frac{1}{2d} \sum_{k=1}^{d}\Bigl[\, h\bigl(+\sqrt{d}\,\bm{e}_k\bigr) \;+\; h\bigl(-\sqrt{d}\,\bm{e}_k\bigr)\,\Bigr],
$$ (eq-stroud3)

where $\bm{e}_k$ is the $k$-th standard basis vector in $\R^{d}$. All weights are equal to $1/(2d)$, and the nodes lie at the common radius $\sqrt{d}$ from the origin (so they expand outward as the dimension grows, consistent with the fact that the bulk of a high-dimensional Gaussian sits in a thin spherical shell, recalling the dimension-of-volumes discussion in {ref}`sec-curse_of_dim`). At $d=1$, the rule collapses to the two-point cubature $\{+1, -1\}$ with weights $\{\tfrac12, \tfrac12\}$, exact for polynomials in $\varepsilon$ up to degree three.

**Why it works (and where it stops).** Rule {eq}`eq-stroud3` integrates every monomial in $\bm\varepsilon'$ of total degree at most three exactly: the first moments vanish by $\pm$-symmetry, the variances $\E{\varepsilon_i'^{\,2}}$ all evaluate to $1$ (each axis contributes $\tfrac{1}{2d} \cdot 2 \cdot d = 1$), the cross-moments $\E{\varepsilon_i' \varepsilon_j'}$ ($i \neq j$) vanish because each node has only one nonzero coordinate, and the third moments vanish again by symmetry. Higher moments are not exact: $\E{\varepsilon_i'^{\,4}}$ is reproduced as $d$ rather than the truth $3$, so the rule introduces a controlled bias proportional to the integrand's fourth-order shock content. In the IRBC class of {ref}`ch-irbc` this bias is empirically negligible at the Euler-equation tolerance one cares about ($\sim 10^{-3}$); see {cite:t}`pichler2011` for a careful demonstration on a multi-country RBC and {cite:t}`Maliar2014325` [Table 5] for a broader cost–accuracy benchmark. When higher accuracy is required, Stroud's degree-5 monomial rule with $2d^2 + 1$ nodes still scales polynomially, not exponentially {cite:p}`judd1998numerical`.

**Cost comparison.** At $Q=3$ nodes per dimension, the tensor-product rule of {ref}`sec-gh_tensor_product` costs $3^{d}$ evaluations per state, while {eq}`eq-stroud3` costs $2d$. {numref}`tab-quadrature_costs` contrasts the two for the dimensionalities encountered later in the script. The monomial rule is the default integration scheme behind the production DEQN code for IRBC-type models in {ref}`ch-irbc` once the shock dimension exceeds about five, dropped in by replacing a single quadrature kernel.

````{table}
:name: tab-quadrature_costs

Quadrature cost per residual evaluation, tensor-product Gauss–Hermite ($Q=3$) versus the Stroud-3 monomial rule of equation {eq}`eq-stroud3`, as a function of the shock dimension $d$. The tensor-product cost grows exponentially in $d$; the monomial cost grows linearly. Brock–Mirman ($d=1$) is the simplest case where the two rules agree to within a factor of $1.5$; the gap explodes once the shock vector is multi-dimensional.

| shock dim. $d$ | where it appears | tensor-product $3^{d}$ | monomial $2d$ | speed-up |
|---:|---|---:|---:|---|
| $1$ | Brock–Mirman (Ch. {ref}`ch-deqn`) | $3$ | $2$ | $\sim 1.5\times$ |
| $3$ | 2-country IRBC (Ch. {ref}`ch-irbc`) | $27$ | $6$ | $\sim 5\times$ |
| $6$ | 5-country IRBC | $729$ | $12$ | $\sim 60\times$ |
| $11$ | 10-country IRBC | $177{,}147$ | $22$ | $\sim 8{,}000\times$ |
| $21$ | 20-country IRBC | $\sim 10^{10}$ | $42$ | $\sim 2 \times 10^8\times$ |
| $51$ | 50-country IRBC | $\sim 10^{24}$ | $102$ | $\sim 10^{22}\times$ |
````

(sec-qmc_cdf)=
### Inverse-CDF Transform and Quasi–Monte Carlo
For very high-dimensional expectations ($d \geq 10$), even sparse-grid rules can become expensive, and, when the integrand contains material fourth-order shock content, the bias of the monomial rule {eq}`eq-stroud3` starts to dominate. An effective alternative is Quasi-Monte Carlo (QMC) integration combined with the inverse-CDF transform {cite:p}`judd1998numerical`. This method maps the unbounded domain of normal shocks $\R^{d}$ to the unit hypercube $[0,1]^{d}$ using the inverse normal CDF:

$$
\E{f(\bm{\varepsilon})} \approx \frac{1}{M} \sum_{m=1}^{M} f\bigl(\Phi^{-1}(\bm{u}_m)\bigr),
$$

where $\bm{\varepsilon} \in \R^{d}$ is the vector of i.i.d. standard normal shocks, $\Phi^{-1}$ is the element-wise inverse standard normal CDF, and $\bm{u}_m$ are points from a low-discrepancy sequence (e.g., Sobol) on $[0,1]^{d}$. For sufficiently smooth integrands, randomized QMC often achieves error rates close to $\mathcal{O}(1/M)$ up to logarithmic factors, compared with the $\mathcal{O}(1/\sqrt{M})$ rate of standard Monte Carlo. This makes QMC highly effective in many high-dimensional applications, including the multi-country IRBC of {ref}`ch-irbc`. The companion notebook [`07_Genz_Approximation_and_Loss_Functions.ipynb`](notebooks/lecture_02_intro_deep_learning/lecture_02_07_Genz_Approximation_and_Loss_Functions.ipynb) explores QMC integration using the Genz test functions, a standard suite of integrands for benchmarking high-dimensional quadrature methods.

(sec-autodiff)=
## Automatic Differentiation for DEQNs
The quadrature rules of the previous section answer the question "how do we evaluate the expectation in the Euler equation?" This section answers the companion question that every DEQN application faces: "how do we evaluate the *derivatives* in the Euler equation, the marginal utility $u'(C_t)$, the marginal product $\partial Y / \partial K$, and the derivative of the value function $V'(K)$?" For the textbook Brock–Mirman model of {ref}`sec-bm` we derived all of these by hand. But every variation of the model, whether CRRA utility in place of log, CES production in place of Cobb–Douglas, a capital adjustment cost, or a borrowing constraint, would force us to redo pages of algebra before writing the loss. Automatic differentiation (AD) removes that tax: the user writes only the *period payoff* $\Pi$ of the model once, and the first-order conditions, the envelope theorem, and all higher-order sensitivities are computed exactly from the code that evaluates $\Pi$. The result is not a numerical approximation of the Euler equation: it is the textbook Euler equation, produced by a different mechanism.

(sec-ad_three_methods)=
### Three ways to take a derivative
There are three generic ways to compute a derivative of a programmable function $f:\R^n \to \R$ at a point $x_0$: symbolic, finite differences, and autodiff. Each has a distinct profile.

**Symbolic differentiation** uses a computer algebra system (Mathematica, SymPy) to manipulate the algebraic expression for $f$ into an algebraic expression for $f'$. Its output is exact and human-readable, and for clean analytic formulas it is ideal. Its two weaknesses are (i) *expression swell*: the chain rule applied to a deeply composed expression can produce a result with many more terms than the original, sometimes enough to defeat further symbolic manipulation; and (ii) it cannot handle arbitrary code with loops and conditional branches.

**Finite differences (FD)** approximates the derivative by the definition itself. For the common central-difference variant,

$$
\widehat{f'}(x_0) \;=\; \frac{f(x_0 + h) - f(x_0 - h)}{2\,h}
\;=\; f'(x_0) \;+\; \tfrac{1}{6}\,f'''(x_0)\,h^2 \;+\; \mathcal{O}(h^4).
$$ (eq-fd_central)

The geometric content of this formula is the mundane fact that the slope of a smooth curve at a point can be approximated by the slope of a nearby *secant line*. {numref}`fig-fd_geometry` makes this concrete: the true tangent at $x_0$ (red) is approximated by the secant connecting $(x_0 - h, f(x_0-h))$ to $(x_0+h, f(x_0+h))$ (blue dashed). As $h$ shrinks the secant rotates toward the tangent, and its slope converges to $f'(x_0)$; taking $h$ too small, however, forces the numerator $f(x_0+h) - f(x_0-h)$ to become the difference of two nearly equal floating-point numbers, at which point catastrophic cancellation dominates.

```{figure} figures/fig-fd_geometry.svg
:name: fig-fd_geometry

Geometric meaning of a central finite difference. The derivative $f'(x_0)$ is the slope of the tangent to $f$ at $x_0$ (red). Finite differences replace this slope by the slope of the secant through the two nearby points $(x_0 - h, f(x_0-h))$ and $(x_0 + h, f(x_0+h))$ (blue, dashed). For small $h$, the secant approaches the tangent at rate $\mathcal{O}(h^2)$; for $h$ too small, the subtraction in the numerator becomes unreliable in finite-precision arithmetic and the approximation degrades. {numref}`fig-fd_ucurve` plots the resulting error against $h$.
```

Two sources of error compete as $h$ changes. The *truncation error* $\tfrac{1}{6}f'''(x_0)\,h^2$ shrinks as $h$ decreases; the *rounding error* associated with computing the near-zero subtraction $f(x_0+h)-f(x_0-h)$ in floating-point arithmetic grows as $\epsilon/h$, where $\epsilon \approx 2.2\cdot 10^{-16}$ is machine precision in double precision. The total error is minimized at the step size that balances the two,

$$
h^\star \sim \epsilon^{1/3},\qquad\text{minimum error}\quad \sim \epsilon^{2/3} \approx 10^{-11}.
$$

The practical consequence is that central FD loses roughly *five digits of precision* relative to the input precision. For second derivatives the loss is worse: Hessians obtained by FD are usually unusable for precision-critical work. FD has one undeniable advantage: it treats $f$ as a black box, requiring only the ability to evaluate $f$. For debugging or for a one-off comparative-statics check it is fine. For *training* a DEQN, where the gradient is used millions of times, it is badly suboptimal. {numref}`fig-fd_ucurve` makes the trade-off visible.

```{figure} figures/fig-fd_ucurve.svg
:name: fig-fd_ucurve

The classic U-curve of central finite differences, here for $f(x) = \mathrm{e}^x$ at $x_0 = 1$. Truncation error (red, dashed) falls as $h^2$; roundoff error (green, dashed) grows as $\epsilon / h$. Their sum (blue) has a minimum at $h^\star \approx 5 \cdot 10^{-6}$, where the achievable precision is around $10^{-11}$, roughly five digits worse than the input precision. Autodiff, by contrast, attains machine precision at no step-size tuning.
```

**Automatic differentiation** exploits the fact that every piece of Python (or any other imperative language) that evaluates a numerical function is, at run-time, a composition of *elementary operations* ($+$, $-$, $\times$, $/$, $\exp$, $\log$, $\sin$, $\cos$, $\ldots$) whose derivatives are known in closed form. The chain rule composes these local derivatives into the derivative of the whole program, exactly. The cost is a small multiple of one function evaluation, typically $2$–$4\times$ in reverse mode, and is *independent of the number of inputs*. There is no step-size $h$ to tune, and the result is exact to machine precision. For ML-scale problems with $10^6$ parameters and a scalar loss, AD is the *only* feasible method. The classic computer-science references are {cite:t}`baydin2018automatic` and {cite:t}`margossian2019review`; both motivate AD from machine-learning workloads and lay out the forward/reverse-mode taxonomy used below.

**Where do local derivatives come from? A tiny built-in table.** It is worth pausing on a question that beginners reasonably ask: *"How does the computer know that the derivative of $\sin(x)$ is $\cos(x)$?"* The answer is deflating but honest: it does not compute this from first principles. Every autodiff framework ships with a short, hand-written table of derivatives for the elementary operations it supports, a few dozen entries such as those in {numref}`tab-ad_primitives`. When your program executes $y = \sin(x)$, the framework looks up the rule "gradient of $\sin$ is $\cos$" in that table and records $\cos(x)$ on the edge of the computational graph. The chain rule then composes those table entries. *Autodiff is not a theorem prover*; it is a library of primitive rules plus an automatic application of the chain rule. This explains both its power (every program expressible in terms of these primitives is differentiable) and its limits (genuinely new functions require extending the table).

````{table}
:name: tab-ad_primitives

A representative slice of the primitive-derivative table that every autodiff framework ships with. TensorFlow, PyTorch, and JAX each include a few hundred such entries: one per elementary operation they support. User code composes these primitives; the chain rule composes their derivatives. When an economist writes the Brock–Mirman period payoff $\Pi(K_t, K_{t+1}) = \log(K_t^{\alpha} + (1-\delta)K_t - K_{t+1})$, only entries from this table are invoked; no calculus is performed at runtime.

| **Primitive** | **Local derivative (built in)** | **Where to find it** |
|---|---|---|
| $y = x + z$ | $\partial y/\partial x = 1,\ \partial y/\partial z = 1$ | `tf.math.add` |
| $y = x \cdot z$ | $\partial y/\partial x = z,\ \partial y/\partial z = x$ | `tf.math.multiply` |
| $y = x^{\alpha}$ | $\partial y/\partial x = \alpha\,x^{\alpha-1}$ | `tf.math.pow` |
| $y = \exp(x)$ | $\partial y/\partial x = \exp(x)$ | `tf.math.exp` |
| $y = \log(x)$ | $\partial y/\partial x = 1/x$ | `tf.math.log` |
| $y = \sin(x)$ | $\partial y/\partial x = \cos(x)$ | `tf.math.sin` |
| $y = \cos(x)$ | $\partial y/\partial x = -\sin(x)$ | `tf.math.cos` |
| $y = \mathrm{ReLU}(x)$ | $\partial y/\partial x = \mathbf{1}\{x > 0\}$ | `tf.nn.relu` |
````

This explanation also makes the *limits* of AD concrete. Operations absent from the table (a black-box solver call, a non-differentiable step like sorting or `argmax`) require either a custom rule (via `@tf.custom_gradient`) or a redesign of the code. The pitfalls in {ref}`sec-ad_pitfalls` are essentially cases where either the table's entry is degenerate at a point (kinks, $\partial|x|/\partial x$ at $x=0$) or the composed gradient is numerically unstable even though each local rule is correct.

(sec-ad_modes)=
### Computational graph; forward and reverse modes
Every numerical function, once evaluated, corresponds to a directed acyclic *computational graph* whose nodes are elementary operations and whose edges carry intermediate values. Take the toy example $y = f(x) = x^2 + \sin(x)$ evaluated at $x_0=2$:

```{math}
:enumerated: false

x \;\to\; v_1 = x^2 \;\to\; y = v_1 + v_2,\qquad
x \;\to\; v_2 = \sin(x) \;\to\; y = v_1 + v_2.
```

The values along the graph are $v_1=4$, $v_2=\sin(2) \approx 0.909$, $y \approx 4.909$. The *edges* carry local derivatives: $\partial v_1/\partial x = 2x = 4$, $\partial v_2/\partial x = \cos(x) \approx -0.416$, $\partial y/\partial v_1 = \partial y/\partial v_2 = 1$. AD combines the edge derivatives by the chain rule in one of two traversal orders. {numref}`fig-ad_graph` shows the graph together with both traversals.

```{figure} figures/fig-ad_graph.svg
:name: fig-ad_graph

The two modes of autodiff on $y = x^2 + \sin(x)$ at $x = 2$. *Top:* forward mode carries a derivative tag $\dot v = \partial v / \partial x$ alongside each value and reads $\dot y = dy/dx$ at the output. *Bottom:* reverse mode evaluates $f$ forward, stores the graph, then walks backwards with $\bar v = \partial y / \partial v$ and reads $\bar x = dy/dx$ at the input. Both deliver $3.584$, equal to $f'(2) = 2 \cdot 2 + \cos(2)$ at machine precision. Forward mode scales linearly with the number of *inputs*; reverse mode scales linearly with the number of *outputs*, which is why it wins for DEQN training (scalar loss, many parameters).
```

**Forward mode** carries a derivative tag $\dot v$ next to every value $v$ and updates both in a single forward pass through the graph. One seeds $\dot x = 1$ at the input; by the chain rule, $\dot v_1 = 4$, $\dot v_2 = -0.416$, and $\dot y = \dot v_1 + \dot v_2 = 3.584$, which is exactly $f'(2) = 2x + \cos x$ evaluated at $x=2$. The cost is one extra float per variable and one extra pass through the graph. For a function with $n$ inputs and $m$ outputs, the cost of the full Jacobian is $\mathcal{O}(n)$ forward passes; forward mode is therefore attractive when *the number of inputs is small*.

**Reverse mode** first evaluates $f$ forward, storing the computational graph and all intermediate values. It then walks the graph backwards, carrying a sensitivity $\bar v = \partial y / \partial v$ along each edge. Seeded with $\bar y = 1$ at the output, the backward pass accumulates

```{math}
:enumerated: false

\bar v_i \;=\; \sum_{j:\, v_i \to v_j} \bar v_j \cdot \frac{\partial v_j}{\partial v_i},
```

yielding at the input node $\bar x = \bar v_1 \cdot 4 + \bar v_2 \cdot (-0.416) = 1\cdot 4 + 1\cdot(-0.416) = 3.584$. The computational cost is $2$–$4\times$ a single forward pass, and, crucially, *one reverse pass produces the full gradient with respect to all inputs simultaneously*. Reverse mode is what `tf.GradientTape`, `torch.autograd`, and `jax.grad` use by default. The price paid is memory: the entire forward graph must be stored for the backward pass. For extremely long simulations this can be a binding constraint; we return to the point in the pitfalls below.

**Which mode for DEQNs.** The DEQN loss is a *scalar* mean-squared Euler residual, computed from a network with $n \sim 10^3$ to $10^5$ parameters. Reverse mode produces the full gradient with respect to all network parameters in $2$–$4\times$ one forward pass, a speedup of $n/4$ over forward mode. This is the same reason reverse mode is the workhorse of deep-learning training: it exactly fits the "few outputs, many inputs" regime.

(sec-ad_euler)=
### The autodiff Euler residual
We now apply autodiff to the very object we spent {ref}`sec-bm` deriving by hand: the Euler equation of the Brock–Mirman planner. Define the *period payoff* as a function of the current state and the current choice:

$$
\Pi(K_{\text{in}}, K_{\text{out}}, z_{\text{in}}) \;=\; u\!\left(\,z_{\text{in}} K_{\text{in}}^{\alpha} + (1-\delta)K_{\text{in}} \;-\; K_{\text{out}}\,\right),
$$ (eq-ad_pi)

where the three argument *slots* are the state $K_{\text{in}}$, the choice $K_{\text{out}}$, and the exogenous shock $z_{\text{in}}$ (the same slot names the code listing below uses); applying $\Pi$ at a date $t$ means plugging $K_t$ into slot 1, $K_{t+1}$ into slot 2, and $z_t$ into slot 3. This is the only place the model enters. Change $u$, $Y$, or $\delta$ and nothing else in what follows needs to move. The Bellman equation of {ref}`sec-bm` then reads

$$
V(K_t, z_t) \;=\; \max_{K_{t+1}}\;\Pi(K_t, K_{t+1}, z_t) + \beta\,\mathbb{E}\!\left[V(K_{t+1}, z_{t+1})\,\big|\,z_t\right].
$$

**Notation: what do $\partial_1\Pi$ and $\partial_2\Pi$ mean?** Throughout this section the subscript names the *slot* of $\Pi$ being differentiated, not a time index. With the three-slot definition $\Pi(K_{\text{in}},K_{\text{out}},z_{\text{in}})$ of {eq}`eq-ad_pi`, we write

```{math}
:enumerated: false

\partial_1\Pi \;\equiv\; \frac{\partial\Pi}{\partial K_{\text{in}}}, \qquad \partial_2\Pi \;\equiv\; \frac{\partial\Pi}{\partial K_{\text{out}}}.
```

The first slot $K_{\text{in}}$ is the *state*, the second slot $K_{\text{out}}$ is the *choice*, the third slot $z_{\text{in}}$ is an exogenous parameter and is not differentiated. An expression like $\partial_2\Pi(K_t,K_{t+1},z_t)$ therefore denotes the derivative of $\Pi$ in its second slot, evaluated with $K_t$ plugged into slot 1, $K_{t+1}$ into slot 2, and $z_t$ into slot 3, so it equals $\partial\Pi/\partial K_{t+1}$. The expression $\partial_1\Pi(K_{t+1},K_{t+2},z_{t+1})$ denotes the derivative in the *first* slot, evaluated at the time-$(t+1)$ state pair, so it also equals $\partial\Pi/\partial K_{t+1}$ but for a different physical reason (because $K_{t+1}$ now sits in the state slot of the period-$t+1$ problem). This overloading is the whole point: the same partial expression handles the FOC in one period and the envelope term in the next.

Two facts, both derived in {ref}`sec-bm`, are now worth restating in terms of $\Pi$:

FOC w.r.t.\ the choice $K_{t+1}$:
: ```{math}
  :enumerated: false

  \partial_2 \Pi(K_t,K_{t+1},z_t)
  \;+\;
  \beta\,\mathbb{E}_{z_{t+1}\mid z_t}[\,V'(K_{t+1},z_{t+1})\,]
  \;=\;0.
  ```

Envelope at the optimum, evaluated at the state $K_t$:
: ```{math}
  :enumerated: false

  V'(K_t, z_t)
  \;=\;
  \partial_1 \Pi(K_t, g(K_t, z_t), z_t),
  ```

  where $g$ is the optimal policy.


Substituting the envelope (evaluated one period ahead, so that $K_{t+1}$ sits in the *state* slot) into the FOC, the familiar Euler equation becomes

$$
\boxed{
\begin{aligned}
0
= {}& \partial_2\,\Pi(K_t, K_{t+1}, z_t)\\
&+ \beta\,\mathbb{E}_{z_{t+1}\mid z_t}\!\left[
\partial_1\,\Pi(K_{t+1}, K_{t+2}, z_{t+1})
\right].
\end{aligned}}
$$ (eq-ad_euler)

Here $K_{t+2}=g(K_{t+1},z_{t+1})$. Both terms in {eq}`eq-ad_euler` are derivatives with respect to the same physical variable $K_{t+1}$, but one treats $K_{t+1}$ as the *choice* of period $t$ (slot 2) and the other treats $K_{t+1}$ as the *state* of period $t+1$ (slot 1). Every term is therefore a *partial derivative of the same function $\Pi$*. Neither $u'$ nor the marginal product of capital, nor the envelope formula $V'(K) = u'(C)(\alpha K^{\alpha-1} + 1 - \delta)$ need to be written out by hand. `tf.GradientTape` records the forward evaluation of $\Pi$ and produces both partials on demand; the expectation is approximated by any of the quadrature rules of {ref}`sec-quadrature_rules`. The code is shorter than the pen-and-paper counterpart and, more importantly, entirely *model-agnostic*: swapping log for CRRA utility means editing the body of `Pi` and nothing else.

```{code-block} text
:name: lst-autodiff_euler
:caption: Autodiff Euler residual for Brock--Mirman. The function \texttt{Pi} is the only model-specific code; the rest is generic. A full implementation lives in the autodiff chapter's code folder, notebook [`02_Brock_Mirman_AutoDiff_DEQN.ipynb`](notebooks/lecture_07_autodiff_for_deqns/lecture_07_02_Brock_Mirman_AutoDiff_DEQN.ipynb).

def Pi(K_in, K_out, z_in):
    Y = z_in * K_in ** alpha
    C = Y + (1.0 - delta) * K_in - K_out
    return tf.math.log(C)

def euler_residual(K_t, z_t, K_tp1, K_tp2, z_tp1):
    # FOC term:       d Pi / d K_{t+1} at (K_t, K_{t+1}, z_t)
    with tf.GradientTape() as t1:
        t1.watch(K_tp1)
        pi_t = Pi(K_t, K_tp1, z_t)
    d2 = t1.gradient(pi_t, K_tp1)
    # Envelope term:  d Pi / d K_t at (K_{t+1}, K_{t+2}, z_{t+1})
    with tf.GradientTape() as t2:
        t2.watch(K_tp1)
        pi_tp1 = Pi(K_tp1, K_tp2, z_tp1)
    d1 = t2.gradient(pi_tp1, K_tp1)
    return d2 + beta * d1
```
**Cross-checking the autodiff loss.** An important pedagogical point is that the autodiff residual {eq}`eq-ad_euler` and the hand-derived residual of {ref}`sec-bm` are the same mathematical object. The companion autodiff notebooks implement both expressions on the same network and report the maximum absolute difference, which in our (seeded) runs is of order $10^{-6}$–$10^{-7}$ in the deterministic case and $10^{-6}$–$10^{-5}$ in the stochastic case with Gauss–Hermite quadrature (float32 arithmetic; float64 tightens this by roughly seven orders of magnitude). In every case the residual is consistent with finite-precision arithmetic, graph ordering, and quadrature accumulation rather than with any difference in the underlying mathematics. Under full depreciation ($\delta = 1$) the trained policy from the autodiff loss matches the analytical closed-form $K_{t+1} = \alpha\beta K_t^{\alpha}$ (respectively $\alpha\beta z_t K_t^{\alpha}$) to mean relative error $\sim 10^{-4}$ in the deterministic case and $\sim 10^{-3}$ on a coarse classroom grid in the stochastic case (the residual rises with the quadrature footprint), confirming that minimizing the autodiff residual recovers the true policy when one is available.

(sec-ad_pitfalls)=
### Pitfalls of autodiff
AD is exact and cheap but it is not magical. Five pitfalls are worth naming; each has a concrete workaround.

**Non-differentiable kinks.** Operations such as $\max(0,x)$ (ReLU), $|x|$, $\min$, $\max$, `argmax`, $\mathrm{sort}$, and indicators are non-differentiable at isolated points. Frameworks return a *subgradient* at the kink – in TensorFlow and PyTorch, $\partial \mathrm{ReLU}/\partial x\,|_{x=0} = 0$ by convention. If the loss repeatedly lands on such a kink, SGD can receive an uninformative gradient and stall. The cure is to either *smooth* the kink ($\mathrm{softplus}$ for ReLU, $\sqrt{x^2 + \varepsilon^2}$ for $|x|$, log-sum-exp for $\max$) or, in complementarity problems, to replace the non-smooth indicator by a Fischer–Burmeister residual ({ref}`ch-irbc`, {ref}`sec-irbc_fischer_burmeister`).

**Boundary singularities.** The Brock–Mirman loss contains $\log C_t$ and hence $1/C_t$. If the network's output happens to drive $C_t$ close to zero during training, AD dutifully returns a very large gradient; SGD then takes a very large step and the network explodes or produces `NaN`. Two cures of different quality are in wide use:

- *Reparameterize so the domain is respected by architecture.* In the Brock–Mirman notebooks, the network outputs the *savings share* $s \in (0,1)$ through a sigmoid; this guarantees $C_t > 0$ *and* $K_{t+1} > 0$ simultaneously, at every training iteration, with no penalty term. This is the hard/soft constraint split of {numref}`fig-hard_soft`.

- *Use numerically stable primitives.* Prefer `tf.math.log1p(x)` to `log(1+x)`, `tf.math.softplus` to a hand-coded $\log(1 + e^x)$, and `tf.math.xlogy(a,b)` to $a\log b$ when the $a=0$ convention matters. AD does not see the cancellations these functions implement; the human must.

**Reverse-mode memory.** Reverse mode stores the entire forward graph so it can walk it backwards. Unrolling a $10{,}000$-step simulation and back-propagating through all of it has memory cost $\mathcal{O}(T \times \mathrm{dim}\,\text{state})$ and can exhaust GPU memory on non-trivial models. Standard mitigations are (i) *gradient checkpointing* (`tf.recompute_grad`), which recomputes parts of the graph on the backward pass in exchange for memory, (ii) *truncated back-propagation through time* for long recurrences, and (iii) path-averaging mini-batches of trajectory segments rather than full trajectories (notebook 02 of Day 2 already does this).

**Loss of structural insight.** AD returns a number, not an algebraic expression. Useful structural facts – the cancellation $u'(C_t) - \beta u'(C_{t+1}) R$ has the sign of the Euler wedge, the Euler residual is homogeneous of a known degree in productivity, and so on – are invisible to AD and come from the hand derivation. In practice one should retain the ability to derive the FOC on a toy version of the model, precisely as we did in {ref}`sec-bm`: AD scales that derivation; it does not replace the understanding it provides.

**`stop_gradient` and the envelope theorem.** A subtle point worth emphasizing. The envelope identity $V'(K) = \partial_1 \Pi(K, g(K))$ holds at the optimum because the FOC kills the term $(-u'(C) + \beta V'(g(K)))\cdot g'(K)$ that otherwise contributes to the total derivative of the composed continuation value. During training we are not yet at the optimum. Two natural codings of the next-period term therefore give different numbers: one differentiates *through* the policy $g$ and computes a total derivative of a rollout, while the other treats $K'=g(K)$ as a fixed choice and computes the partial derivative used in the Euler FOC. The autodiff residual {eq}`eq-ad_euler` uses the latter, which is what falls out of `t2.gradient(pi_tp1, K_tp1)` in Listing {ref}`lst-autodiff_euler`. This is the correct residual for checking the Euler equation; differentiating through the policy is a different object. The two codings agree as the FOC residual vanishes, but off the optimum they can disagree.

**Companion notebooks.** Three notebooks (in the autodiff chapter's code folder) put the above into practice, with a fourth provided as self-study:

- `01_AutoDiff_Analytical_Examples`: warm-up ($y=x^2 + \sin x$), the FD U-curve regenerated numerically, CRRA utility's first and second derivative, Cobb–Douglas production's 2-D gradient field, the capital adjustment cost $\Gamma(K,K')$ with AD vs hand partials side by side, and the Hessian of Cobb–Douglas via a nested `GradientTape`.

- `02_Brock_Mirman_AutoDiff_DEQN`: the loss of {eq}`eq-ad_euler` implemented on the deterministic model of {ref}`sec-bm`; cross-check against the hand residual at float32 tolerance; cross-check of the trained policy against $K_{t+1} = \alpha\beta K_t^\alpha$.

- `03_Brock_Mirman_Uncertainty_AutoDiff_DEQN`: the same, with AR(1) productivity and Gauss–Hermite quadrature; cross-check against the stochastic hand residual and the closed-form $K_{t+1} = \alpha\beta z_t K_t^\alpha$ in a full-depreciation side-experiment.

- `04_IRBC_AutoDiff_DEQN` (self-study): the same $\partial_2\Pi + \beta\,\mathbb{E}[\partial_1\Pi]$ template scaled up to the multi-country IRBC model of {ref}`ch-irbc` (per-country planner Lagrangian, Fischer–Burmeister irreversibility, tensor-product Gauss–Hermite), with a machine-precision cross-check against the {ref}`ch-irbc` hand-derived residual.

## Data Parallelization with MPI

For very large-scale applications (e.g., $N \geq 50$ countries in the IRBC model of {ref}`ch-irbc`), training can be accelerated by distributing the gradient computation across multiple GPUs or compute nodes. The standard approach uses synchronous data parallelism via MPI `Allreduce`; in the DEQN paper the corresponding implementation is built on *Horovod* {cite:p}`sergeev2018horovod`, which wraps `Allreduce` into a drop-in replacement for the training optimizer.

```{prf:definition} Data-Parallel DEQN Training

- **Input:** $P$ workers, each with local copy of $\mathcal{N}_\rho$
- **for** each training iteration **do**
  - Each worker $p$ draws local mini-batch $\mathcal{B}_p$ from simulation
  - Each worker computes local gradient: $\bm{g}_p = \nabla_\rho \ell(\mathcal{B}_p)$
  - **MPI\_Allreduce:** $\bar{\bm{g}} = \frac{1}{P}\sum_{p=1}^P \bm{g}_p$
  - Each worker updates: $\rho \leftarrow \rho - \eta\,\bar{\bm{g}}$
- **end**
```


Since each worker processes an independent mini-batch and the `Allreduce` operation averages the gradients, the effective batch size scales linearly with the number of workers. In practice, this yields near-linear speedup for moderate numbers of workers ($P \leq 32$), with communication overhead becoming significant only for very large clusters. The key advantage for economics applications is that each worker can simulate its own trajectory of the economy, naturally exploring different regions of the state space and improving the diversity of the training distribution.

(sec-loss_kernels)=
## Choice of Loss Kernel: Beyond Mean-Squared Loss
Every DEQN derivation in this script ends with the same step: take the equilibrium residual $r(\bm{x})$, square it, average over a mini-batch, and minimize. The squared average, mean-squared error (MSE), is the canonical default, but it is only one of many reasonable reductions of a residual vector to a scalar. Different reductions emphasize different parts of the residual distribution and have visibly different convergence properties, even on the same model with the same network and data. This section makes the trade-off concrete on the stochastic Brock–Mirman model with full depreciation, where the optimal savings rate is known in closed form ($s^\star = \alpha\beta$) so that the deviation between learned and optimal policy can be measured exactly.

**Six kernels.** We compare:

- **MSE**: $\ell = \tfrac1B \sum r^2$. Quadratic everywhere; large residuals dominate the gradient.

- **MAE**: $\ell = \tfrac1B \sum |r|$. Robust to outliers, but the gradient has *constant magnitude*, so the optimizer takes the same step size regardless of how close to zero a residual already is. This is fine for learning a coarse fit, but it means MAE training plateaus at a finite "noise floor" that is several orders of magnitude above what MSE attains at the same training budget.

- **Huber**: quadratic for $|r| \le \delta$, linear for $|r| > \delta$. Smooth hybrid that combines MSE's fine convergence with MAE's tail robustness, but the transition at $|r|=\delta$ is a kink in the gradient, which can introduce small training-loop oscillations as residuals migrate across the threshold.

- **Quantile pinball** at level $\tau$: $\ell = \tfrac1B \sum \max(\tau r, (\tau-1)r)$. Targets the signed $\tau$-quantile of the residual distribution; high $\tau$ emphasizes positive residual tails, low $\tau$ emphasizes negative tails, and all observations contribute with asymmetric weights.

- **CVaR-style** at confidence $\alpha$: $\ell =$ mean of the top $(1{-}\alpha)$ fraction of $|r|$. Explicitly trains the worst-case states.

- **LogCosh**: $\ell = \tfrac1B \sum \log\cosh(r)$. This kernel is $C^\infty$ everywhere and combines MSE-like behavior near $r=0$ ($\log\cosh r \approx \tfrac12 r^2$) with MAE-like saturation in the tails ($\log\cosh r \approx |r| - \log 2$, gradient saturates at $\pm 1$). Crucially, the transition between the two regimes is smooth: there is no kink at any threshold.

**Convergence behavior.** {numref}`fig-loss_kernels` shows the mean, $p_{90}$, and $p_{99}$ of the absolute relative Euler error on a fixed evaluation grid as a function of the training episode, for each of the six kernels. Three patterns are immediate.

```{figure} figures/loss_kernel_convergence.png
:name: fig-loss_kernels
:width: 100%

Convergence of the relative Euler-error distribution under six different loss kernels on the stochastic Brock–Mirman model with full depreciation. Same network (2$\times$32 swish, sigmoid head), same Adam optimizer, same CRN training stream; only the loss kernel changes. Left: mean $|r|$. Middle: $p_{90}\,|r|$. Right: $p_{99}\,|r|$. All on a log scale. Closed-form benchmark: $s^\star = \alpha\beta$.
```

*First*, MAE clearly stalls at a noise floor an order of magnitude above the other kernels, exactly the constant-gradient pathology described above; the curve is monotone but flattens out and refuses to drop further. *Second*, MSE and log-cosh achieve essentially the same final mean residual ($\sim 1.4\!\times\!10^{-4}$), and both pull the $p_{99}$ down with comparable efficacy. *Third*, the tail-aware kernels (Huber, Quantile, CVaR) typically produce the narrowest spread between mean and max residual but pay a small constant cost in the bulk; the practical pay-off shows up in problems where rare-but-consequential states matter.

**Why log-cosh tends to converge most smoothly.** Of the six kernels, log-cosh is the one whose convergence curve is essentially monotone with no spikes. This is not an accident: $\log\cosh(r)$ is a convenient smooth interpolant between $\tfrac12 r^2$ near zero and $|r|$ in the tails, with no kink at any threshold. Near the optimum its gradient is $\tanh(r) \approx r$ (linear, like MSE), so it inherits MSE's fine-grained convergence. Far from the optimum its gradient saturates at $\pm 1$ (like MAE), so an unusually large residual cannot dominate the mini-batch gradient. In effect, log-cosh is what one would design if asked for "MSE near zero and MAE in the tails, smoothly". For practical DEQN settings where the residual distribution is unknown in advance, log-cosh is therefore a useful default to compare against MSE rather than a mere robustness afterthought.

**Why this is an *economic* comparison.** The relative Euler-equation residual already has a direct economic interpretation: a $1\%$ relative Euler error translates approximately into a $1\%$ per-period consumption error along the simulated path {cite:p}`judd1998numerical`. The mean / $p_{90}$ / $p_{99}$ panels of {numref}`fig-loss_kernels` therefore measure, in interpretable units, the average and tail consumption mistakes that each loss kernel leaves behind. The point of the experiment is that *the training-loss ranking is not the same as the ranking on this economic metric*: a kernel that achieves a slightly higher *mean* residual but a much narrower *tail* produces smaller worst-case consumption errors, which is what matters when rare states are economically consequential (occasionally binding constraints, fat-tailed shocks, tipping risks). The training loss is the instrument; the relative Euler error along simulated paths is the criterion. Choosing the kernel with that hierarchy in mind is part of the modeling decision. The companion notebook pushes this one step further by collapsing the per-period error distribution into a single consumption-equivalent welfare loss against $s^\star$; readers who want a single-number summary of the trade-off should consult that table directly.

The full experiment, including a path-residual histogram, a policy-error heatmap on the $(z, \log K)$ plane, and a CE-welfare-loss summary table, is in the companion notebook [`05_StochasticBM_LossComparison.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_05_StochasticBM_LossComparison.ipynb).

**Extensions of the basic DEQN template.** The Brock–Mirman model establishes the core DEQN recipe, but it is only the starting point. Later chapters modify the same template in several directions: {ref}`ch-olg` adds lifecycle structure and complementarity constraints, {ref}`ch-young` replaces a finite vector of states by a histogram representation of the cross-sectional distribution, {ref}`sec-sequence_space` then shows that one can also feed *shock histories* to the network rather than the current endogenous aggregate state, and {ref}`ch-climate` extends the method to nonstationary climate-economy models with pseudo-states. The equilibrium logic is the same in all cases: choose a network parameterization, simulate the model forward, evaluate equilibrium residuals, and update the network by gradient descent.

**Code examples.** The following Jupyter notebooks implement and extend the material in this chapter. Notebooks 01 and 02 illustrate the *sampling progression* that is pedagogically central to the method: 01 uses uniform random states to isolate the loss-and-architecture mechanics, while 02 replaces the exogenous grid by simulated trajectories and learns on the model's ergodic set ({ref}`sec-deqn_algo`); both derive the FOC and apply the envelope theorem on paper before writing the loss. Notebooks 03 and 04 introduce additional techniques (endogenous labor, a KKT-constrained labor-time ceiling encoded via a *Fischer–Burmeister* complementarity function, and a six-period OLG extension) that anticipate material developed formally in {ref}`ch-irbc` and {ref}`ch-olg`. Three further notebooks that re-solve the same two Brock–Mirman models with an *autodiff* loss (illustrating the template of {ref}`sec-autodiff`) are placed in the autodiff-chapter code folder as the autodiff primer: [`01_AutoDiff_Analytical_Examples.ipynb`](notebooks/lecture_07_autodiff_for_deqns/lecture_07_01_AutoDiff_Analytical_Examples.ipynb), [`02_Brock_Mirman_AutoDiff_DEQN.ipynb`](notebooks/lecture_07_autodiff_for_deqns/lecture_07_02_Brock_Mirman_AutoDiff_DEQN.ipynb), [`03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb`](notebooks/lecture_07_autodiff_for_deqns/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb).

- [`01_Brock_Mirman_1972_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb): deterministic Brock–Mirman; exogenous uniform sampling; hand-derived FOC + envelope.

- [`02_Brock_Mirman_Uncertainty_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb): stochastic Brock–Mirman ($\varrho > 0$, $\sigma_z > 0$); simulation-based sampling on the ergodic set; hand-derived FOC + envelope.

- [`03_DEQN_Exercises_Blanks.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_03_DEQN_Exercises_Blanks.ipynb): four guided exercises (endogenous labor, KKT + Fischer–Burmeister, simple OLG extension); blank version.

- [`04_DEQN_Exercises_Solutions.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_04_DEQN_Exercises_Solutions.ipynb): the same four exercises with complete solutions.

- [`05_StochasticBM_LossComparison.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_05_StochasticBM_LossComparison.ipynb): the stochastic Brock–Mirman is re-solved six times with identical network, optimizer, and CRN training data, and only the loss kernel changes (MSE, MAE, Huber, quantile pinball, CVaR, log-cosh). The notebook switches to full depreciation $\delta=1$ so that the closed-form optimal savings rate $s^\star=\alpha\beta$ is available, then evaluates each trained policy on the *economic* metric: the relative Euler-equation error along a simulated path, plus the consumption-equivalent welfare loss against $s^\star$. The exercise makes concrete that the choice of training loss and the convergence of the metric we ultimately care about are not the same thing, and that tail-aware kernels (Huber, CVaR, quantile pinball) trade a small loss in the bulk for a much cleaner $p_{99}$.

```{prf:remark} Chapter Summary

- DEQNs solve dynamic-equilibrium models by treating the equilibrium conditions $G(\x_t, p(\x_t), \mathbb{E}_t[H(\cdot)]) = 0$ as a residual loss; SGD on this loss replaces the traditional fixed-point iteration {cite:p}`azinovicDEEPEQUILIBRIUMNETS2022`.

- The hard/soft split is the key design pattern: state transitions and budget constraints are encoded *exactly* in the network architecture (e.g. a sigmoid savings-share head); only the Euler residual is minimized in the loss. This eliminates infeasible policies and accelerates training.

- The Brock–Mirman benchmark with closed-form solution validates the methodology and is the reference example reused throughout the script.

- Pathwise and conditional-expectation residuals target different objectives: path averaging is cheap and can work well in benchmarks such as Brock–Mirman, while explicit quadrature directly targets the conditional Euler equation and is preferred when expectation accuracy is central.
```


## Further Reading {.unnumbered}
- {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, the foundational DEQN paper; required reading.

- {cite:t}`maliar2021deep`, "all-in-one" deep learning, an alternative formulation discussed in {ref}`ch-young`.

- {cite:t}`fernandezvillaverde2024taming`, broad survey of deep learning in macro, situating DEQNs against PINNs and other approaches.

- {cite:t}`judd1998numerical`, the classical numerical-methods reference whose toolkit DEQNs supplement (rather than replace); especially Chapters 7 and 7.5 on Gauss–Hermite quadrature and monomial rules underlying {ref}`sec-quadrature_rules`.

- {cite:t}`stroud1971approximate`, the canonical reference for the monomial cubature formulas of {ref}`sec-monomial_cubature`.

- {cite:t}`pichler2011` and {cite:t}`Maliar2014325`, monomial integration in large-scale dynamic economic models.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch2-1

**[Core\] Closed-form Brock–Mirman.** Verify that for log utility, $\delta = 1$, and AR(1) productivity $\ln z_{t+1} = \varrho \ln z_t + \sigma_z \varepsilon_{t+1}$, the optimal savings share is the constant $s^\star = \alpha\beta$. Use this to check that a converged DEQN's average sigmoid output should equal $\alpha\beta$.
```

```{exercise}
:label: ex-ch2-2

**[Core\] Hard vs. soft constraints.** Consider a softplus head on $C_t$ alone (so $C_t > 0$ but $K_{t+1}$ unconstrained). Construct an explicit input $(K_t, z_t)$ for which a randomly initialized network would predict $K_{t+1} < 0$. Explain why the sigmoid-savings parameterization eliminates this failure mode.
```

```{exercise}
:label: ex-ch2-3

**[Core\] Path averaging vs. conditional expectation.** Let $G_\rho(x_t,\varepsilon_{t+1})$ denote the one-shock Euler residual under network parameters $\rho$. Show that, under ergodicity, the path average of $G_\rho^2$ converges almost surely to $\mathbb{E}_{\mu,\varepsilon}[G_\rho(x,\varepsilon)^2]$ as $T_{\text{sim}} \to \infty$. Compare this target with the conditional residual objective $\mathbb{E}_{\mu}[(\mathbb{E}[G_\rho(x,\varepsilon)\mid x])^2]$. Use Jensen's inequality to explain why the pathwise squared objective is generally stronger, and discuss the finite-sample variance trade-off.
```

```{exercise}
:label: ex-ch2-4

**[Core\] Brock–Mirman with Gauss–Hermite.** In the Brock–Mirman Euler equation {eq}`eq-bm_euler`, replace the single-shock pathwise residual by a $Q=5$ Gauss–Hermite expectation ({ref}`sec-gh_tensor_product`). Write out the resulting deterministic loss in closed form, and check on a few representative $(K_t, z_t)$ that the value matches a Monte Carlo estimate with $10^4$ shock draws to four significant digits.
```

```{exercise}
:label: ex-ch2-5

**[Core\] Monomial rule by hand.** Take the Stroud-3 rule of equation {eq}`eq-stroud3` for shock dimension $d=4$ (so $8$ nodes). Verify by direct computation that the rule reproduces $\E{\varepsilon_i'}=0$, $\E{\varepsilon_i'^{\,2}}=1$, $\E{\varepsilon_i'\varepsilon_j'}=0$ for $i\neq j$, and $\E{\varepsilon_i'^{\,3}}=0$ exactly, but gives $\E{\varepsilon_i'^{\,4}} = d = 4$ instead of the true value $3$. Show that the relative bias on the fourth moment grows linearly in $d$, and discuss when this matters for an Euler-equation residual.
```

```{exercise}
:label: ex-ch2-6

**[Core\] Loss-kernel selection.** Three application scenarios are described below; match each to the most appropriate loss kernel from the menu {MSE, MAE, Huber($\delta$), pinball loss at $\tau$, CVaR at $\alpha$, log-cosh} and justify the choice in two or three sentences. (a) "Quadrature noise occasionally produces a few large Euler residuals; we want a smooth loss that behaves quadratically near zero so gradients vanish at the optimum, but only linearly in the tails so that a single noisy point cannot dominate the gradient." (b) "A regulator will inspect the worst $1\%$ of residuals; we want the optimizer to drive the conditional mean above the $99$th percentile down to tolerance, not the average." (c) "We care that the *median* Euler residual is small. Tails are an artefact of badly conditioned states near the borrowing constraint and should not pull the gradient."
```

```{exercise}
:label: ex-ch2-7

**[Computational\] Multivariate shock scaling.** Extend notebook [`lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb) to $d$ independent productivity shocks, $d \in \{1, 2, 4, 8\}$ (e.g., add country-level TFP terms to the production technology, all i.i.d. standard normal). For each $d$, train the network twice: once with tensor-product Gauss–Hermite at $Q=3$ ($3^d$ nodes per residual), once with the Stroud-3 rule of {eq}`eq-stroud3` ($2d$ nodes). Plot training time per epoch and final relative Euler error against $d$ on the same axes. Confirm that the Stroud-3 cost grows linearly while Gauss–Hermite is exponential, and identify the $d$ at which Gauss–Hermite becomes impractical on a single GPU.
```

```{exercise}
:label: ex-ch2-8

**[Computational\] Implementation.** Modify notebook [`lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb) to use a tanh activation instead of Swish. Does training still converge? How does the time-to-converge change?
```

[^1]: In the sense of {cite:t}`ECTA:ECTA1716`: an approximation of the equilibrium policy (or value) functions over the entire economically relevant region of the state space, in particular over the model's ergodic set, as opposed to a *local* (perturbation) solution that is accurate only in a neighborhood of the deterministic steady state.
