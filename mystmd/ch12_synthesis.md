---
title: "Synthesis and Outlook"
label: ch-outlook
---

This final chapter steps back from the individual methods and applications to identify the unifying themes that connect them. We distill the common computational paradigm, embedding economic structure in a neural network loss function and optimizing via gradient descent, and provide a practical decision guide for choosing among DEQNs, PINNs, deep surrogates, and GP + BAL based on the problem at hand. We then discuss open challenges at the frontier and offer practical tips distilled from the experience of applying these methods to real research problems. The chapter is intentionally high-level: it emphasizes synthesis and practical judgment rather than introducing new technical machinery.

## The Unifying Paradigm

The methods presented in this course, namely DEQNs, PINNs, deep surrogates, and GP + BAL, do not all use the same estimator, but they share a common computational workflow: a flexible function approximator, a deep network or a Gaussian-process kernel, encodes the economic object of interest; structural information enters through residuals, hard constraints, model-generated training data, or an acquisition rule; and a differentiable or Bayesian inference machinery delivers training, sensitivity analysis, and uncertainty quantification.

```{figure} figures/fig-unifying_paradigm.svg
:name: fig-unifying_paradigm

The shared computational workflow of the four methods of this course (DEQNs, PINNs, deep surrogates, and GP + BAL): a flexible function approximator (a deep network or a Gaussian-process kernel) plays the role of the unknown economic object; structural information enters through equilibrium residuals and hard constraints (DEQNs, PINNs), through model-generated training data and parameter pseudo-states (deep surrogates), or through the prior, the simulator, and the acquisition rule (GP + BAL); and a differentiable or Bayesian inference layer supplies the gradients, the posterior, and the design rule needed for training, sensitivity analysis, and uncertainty quantification.
```

{numref}`fig-unifying_paradigm` makes the common template explicit. Structural knowledge of the economic or financial model is built directly into the training objective or the training design rather than learned from labeled data: DEQNs minimize equilibrium-residual losses, PINNs minimize PDE residuals, deep surrogates fit model-generated labels (often with parameter pseudo-states), and GP + BAL combines a kernel prior with an acquisition rule that decides where to sample next. Automatic differentiation supplies the gradients needed for network training, and the mesh-free nature of neural networks and kernels avoids explicit tensor-product state grids, mitigating the practical curse of dimensionality without eliminating the underlying high-dimensional sample-complexity, approximation, optimization, and integration costs.

## Decision Guide: When to Use Which Method

The four method families covered in this course occupy partly overlapping but distinguishable niches. {numref}`tab-method_decision_guide` summarizes the key distinctions along four practical criteria: the natural *time domain* (discrete vs. continuous), the kind of *equation* the method is designed to solve, the *key advantage* that distinguishes it from the others, and the *typical use cases* encountered in this course.

````{table}
:name: tab-method_decision_guide

Decision guide: when to use which method.

| **Criterion** | **DEQN** | **PINN** | **Deep surrogate** | **GP / BAL** |
|---|---|---|---|---|
| Time domain | Discrete | Continuous | Either | Either |
| Equation type | Algebraic / expectations | PDEs (HJB, KFE, BS) | Black-box model | Black-box model |
| Key advantage | Up to $d>100$ in selected models {cite:p}`azinovicDEEPEQUILIBRIUMNETS2022` | AD derivatives of the network approximation | Instant re-evaluation | Built-in UQ + active learning |
| Typical use | DSGE, OLG, IRBC | HJB, option pricing | Estimation, UQ, policy | Small-$N$ surrogates, BAL design |
````

A few practical takeaways from the table. **DEQNs** are the workhorse for *discrete-time* general-equilibrium models with high-dimensional state spaces, where Euler equations and market-clearing residuals admit a clean residual-loss formulation. **PINNs** are the natural analogue in *continuous time*, replacing finite-difference HJB or BSDE solvers when one needs exact derivatives of the trained network by automatic differentiation rather than finite-difference derivatives of an interpolant. **Deep surrogates** differ in kind: they do not solve the model; they learn a fast emulator of the already-solved mapping from parameters to economic quantities, which is what makes structural estimation, sensitivity analysis, and policy design tractable at scale. **Gaussian processes with Bayesian active learning** occupy the small-data, high-uncertainty corner: the right tool when each evaluation is expensive, the input dimension is moderate ($d \lesssim 10$ for naïve GPs, extending to higher dimensions via active subspaces and deep AS; cf. {ref}`ch-gp`), and posterior uncertainty is itself part of the answer. Within the PINN family, EMINNs ({ref}`ch-ct_theory`) are a specialization to continuous-time heterogeneous-agent master equations: they keep the PDE-residual training logic but encode the cross-sectional distribution itself as part of the network input, which is one of several routes to global solutions of Krusell–Smith-style models with aggregate shocks, complementary to the set-encoder and price-as-state approaches discussed in {ref}`sec-open_challenges`.

The categories are not mutually exclusive. Many production pipelines combine them, e.g., a DEQN to solve the equilibrium and a deep surrogate or GP that treats parameters as pseudo-states for estimation ({ref}`ch-estimation`), or a PINN with a surrogate wrapper for policy counterfactuals. The right question is rarely "which of the four?" but "which combination?".

(sec-running_case_bm)=
## Running Benchmarks Through Every Lens
The course repeatedly returns to a small set of canonical benchmarks: Brock–Mirman for discrete-time stochastic growth, cake-eating for continuous-time HJBs, and parameterized Brock–Mirman variants for surrogates and SMM. {numref}`tab-bm_running_case` reads each benchmark through the lens of one course method, exposing what changes when the methodology changes and what stays invariant.

````{table}
:name: tab-bm_running_case

Canonical benchmarks treated through each course method. The economic content differs by row, three rows are Brock–Mirman and the PINN row is cake-eating, but the table exposes the invariant computational pattern: choose an object to approximate, encode the model restrictions in the objective or training design, and validate with an economically interpretable diagnostic. Notebook filenames in the last column omit the standard `lecture_NN_` prefix; the four notebooks live, in row order, in the `code/` subdirectories of lectures 03, 11, 14, and 15.

| **Method** | **What is approximated** | **Loss / objective** | **Diagnostic** | **Notebook** |
|---|---|---|---|---|
| DEQN | Savings share $s(K,z)\in(0,1)$ via sigmoid head | Squared rel. Euler residual on simulated trajectory | $s \to \alpha\beta$ at $\delta\!=\!1$, log utility | [`01_Brock_Mirman_1972_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb) |
| PINN, cake-eating | Value function $\hat V(a)$ on a 1D interval | HJB residual $\rho V-\max_c[u(c)+V'(a)(ra-c)]$ | $\hat V$ matches closed form $V^\star$ (CRRA) | [`04_Cake_Eating_HJB_PINN.ipynb`](notebooks/lecture_11_pinns/lecture_11_04_Cake_Eating_HJB_PINN.ipynb) |
| GP $+$ BAL | $\hat V(K)$ on Brock–Mirman ergodic set | GP marginal likelihood; BAL picks next $K_*$ | 95% pred. band covers true $V$ | [`04_GP_Value_Function_Iteration.ipynb`](notebooks/lecture_14_surrogates_and_gps/lecture_14_04_GP_Value_Function_Iteration.ipynb) |
| SMM $+$ surrogate | Policy $s(K,z,\varrho)$ pseudo-state on $\varrho$ | SMM moment objective in $\varrho$ | $\hat\varrho$ within MCSE of truth | [`lecture_15_03_Structural_Estimation_BM.ipynb`](notebooks/lecture_15_structural_estimation_smm/lecture_15_03_Structural_Estimation_BM.ipynb) |
````

The takeaway is that no single methodology is the "right" one for these benchmarks. Rather, the four lenses answer different research questions: a DEQN gives the policy function on Brock–Mirman, a PINN gives a value function on cake-eating with AD derivatives of the network approximation, a GP gives uncertainty-quantified value-function estimates on Brock–Mirman with guidance for the next sample point, and the surrogate-SMM pipeline estimates the deep parameter $\varrho$ (productivity persistence) on a Brock–Mirman variant from data; the joint $(\beta,\varrho)$ extension lives in the companion [`lecture_15_03b_Structural_Estimation_BM_Joint.ipynb`](notebooks/lecture_15_structural_estimation_smm/lecture_15_03b_Structural_Estimation_BM_Joint.ipynb), where the imperfect identification of $\beta$ from macro moments shows up as a visible ridge in the criterion surface. Real research applications typically combine at least two of these, and the implementation guidance throughout the chapters is designed to make those combinations cheap.

## Bridges Between Methods

The four method families do not sit in isolated boxes. {numref}`fig-unified_view_methods` maps the bridges between them: which methods share a state representation, which methods share an inference machinery, and where one method layers naturally on top of another. Discrete- and continuous-time formulations connect DEQNs and PINNs; deep surrogates and GP + BAL share the pseudo-state and uncertainty-quantification idea; and the lower row of the figure can be read as a pipeline (solve $\to$ amortize $\to$ quantify) rather than as four parallel choices.

```{figure} figures/fig-unified_view_methods.svg
:name: fig-unified_view_methods

Bridges between the four method families. The core box restates the shared workflow: a flexible approximator (network or kernel), with economic structure encoded in the objective or training design, trained or updated through a differentiable or Bayesian pipeline. The dashed gray arrows mark the two main bridges, between discrete- and continuous-time residual solvers (DEQN $\leftrightarrow$ PINN) and between deep surrogates and the GP + BAL uncertainty layer that sits on top of them.
```

Reading the figure column by column gives a one-line gloss of each family: DEQNs (blue) handle discrete-time Euler equations in high-dimensional state spaces; PINNs (red) handle continuous-time HJB, Kolmogorov-forward, and Black–Scholes PDEs through PDE residuals; deep surrogates (orange) make model parameters part of the input space and replace the expensive solve with an instant forward pass for estimation and uncertainty quantification; GP + BAL (green) layers calibrated uncertainty and sample-efficient active learning on top of the surrogate. The bridges in the figure are not exhaustive; production pipelines often chain three families at once, for example a DEQN solver followed by a deep surrogate, followed by a GP for sensitivity analysis around the estimated optimum.

(sec-open_challenges)=
## Open Challenges and Future Directions
Despite the considerable progress surveyed in this course, several open challenges remain at the frontier of deep learning for economics:

1.  **Convergence guarantees.** While empirical performance is strong, theoretical convergence guarantees for DEQNs and PINNs in the context of economic equilibrium computation remain limited. Establishing rates of convergence as a function of network width, depth, and training budget is an active area of research. The gap between theory and practice remains wide.

2.  **Combining global and local methods.** Deep Kernel Learning {cite:p}`wilson2016deep,chen2025private`, which uses a DNN as a feature map inside a GP kernel, is one principled step toward integrating the two paradigms. In principle, such hybrids could retain some of the GP's calibrated uncertainty estimates while benefiting from the DNN's representational flexibility. In practice, however, the UQ properties of deep kernel models require careful validation, their scalability advantages over pure GPs are problem-dependent, and locating the kinks and boundary layers where local refinement would help most is itself a hard unsolved problem. Hybrid methods therefore remain a promising but largely open research direction rather than a deployable solution.

3.  **Real-time policy analysis.** Deep surrogates enable near-instant policy evaluation, potentially transforming how policy makers conduct scenario analysis and stress testing. {cite:t}`kubler2025using` demonstrate that constrained optimal policy rules can be computed over high-dimensional parameter spaces using deep surrogates, opening the door to real-time policy counterfactuals. For central banks and finance ministries, this could mean running complex DSGE scenarios with substantial speedups (seconds rather than minutes-to-hours, depending on the underlying solver), enabling more responsive and comprehensive policy analysis during fast-moving economic crises.

4.  **Estimation of rich structural models.** Combining DEQNs or PINNs with surrogate-based estimation pipelines allows researchers to estimate models with many parameters from micro and macro data simultaneously {cite:p}`friedlDeep2023,chen2026Deep`. {cite:t}`kase2022estimating` use neural networks to estimate nonlinear heterogeneous-agent models, opening the door to estimation of models that were previously intractable. As micro-level administrative datasets become increasingly available to central banks, the ability to estimate high-dimensional structural models from rich data sources represents a major frontier.

5.  **Climate–economy integrated assessment.** Integrated assessment models with tipping points, regional heterogeneity, and multiple sources of risk represent a natural application of the methods surveyed here. {cite:t}`fernandezvillaverde2025climate` provide a comprehensive overview of the state of the art. The DEQN methodology is particularly well suited to these models because climate–economy interactions introduce additional state variables (carbon concentrations, temperature, damages) that exacerbate the curse of dimensionality.

6.  **Heterogeneous-agent models and beyond.** Extending the DEQN and PINN frameworks to heterogeneous-agent models with aggregate shocks, where the cross-sectional distribution of agents is itself part of the aggregate state, is a major frontier ({ref}`ch-young`). {cite:t}`han2023deepham` propose DeepHAM, which represents the distribution via learned generalized moments and trains networks by maximizing cumulative utility along simulated paths ({ref}`sec-deepham_ks`); {cite:t}`payne2025deepsam` extend the same set-encoder idea to non-Walrasian search-and-matching models, where the distribution affects decisions through the matching technology and bargaining rather than through prices ({ref}`sec-deepsam_ks`); and {cite:t}`yang2025structural` show that for many Walrasian HA models, equilibrium prices alone, rather than the full distribution, can serve as the aggregate state in agents' policy functions, sidestepping the master equation altogether. Together these approaches demonstrate that deep learning can tackle the infinite-dimensional state spaces inherent in heterogeneous-agent economies, and can also help *define* new equilibrium concepts for them.

7.  **Operator learning.** Rather than learning a single function (as in DEQNs or PINNs), operator learning frameworks such as DeepONet {cite:p}`lu2021learning` and the Fourier Neural Operator {cite:p}`li2021fourier` learn mappings between function spaces, for example, learning the map from a forcing function to the PDE solution, or from a model specification to its equilibrium. This "learn the solver" paradigm could enable even faster model evaluation by amortizing the cost of solving across many model instances.

8.  **Error certification and benchmark protocols.** A small residual on the training distribution is not the same as a small policy or welfare error on the economically relevant state space. A major frontier is the development of a-posteriori error bounds, benchmark suites with classical baselines, and standardized reporting conventions: residual heat maps, Euler-error distributions, market-clearing errors, transversality checks, and consumption-equivalent welfare losses. Without such certification, deep solvers are hard to compare across papers and hard to audit in policy environments.

9.  **Distribution shift, tails, and rare events.** Simulation-based training concentrates samples on the ergodic set. This is efficient for average performance but dangerous when the research question concerns crises, tail risk, climate tipping points, binding collateral constraints, or off-policy counterfactuals. Robust sampling schemes that combine ergodic states, boundary states, adversarial states, and economically meaningful stress scenarios are still underdeveloped, and they are precisely the regime where small training residuals can coexist with large welfare errors.

(sec-when_not_to_use_dl)=
## When *Not* to Use Deep Learning
A pedagogical script risks selling the methods it teaches. Symmetry requires an explicit statement of the regimes in which the deep-learning toolkit covered here is *not* the right answer; recognizing those regimes is part of using the toolkit well.

1.  **The state space is genuinely low-dimensional.** For models with $d \lesssim 5$ continuous state variables and smooth policies, classical methods, projection on Chebyshev polynomials, value-function iteration on a fine grid, or perturbation around the steady state, typically deliver higher accuracy at lower cost than a DEQN {cite:p}`judd1998numerical`. The curse of dimensionality is real, but for smooth low-dimensional models it is rarely the binding constraint; nonsmoothness, occasionally binding constraints, and tail events can make even low-dimensional problems hard.

2.  **The model is locally linear and the question is local.** If you only need first- or second-order responses around the deterministic steady state, perturbation methods give analytical impulse responses in seconds; the standard implementation in macroeconomics is {cite:t}`adjemian2024dynare`. Use a DEQN or a PINN only when global nonlinearity, large shocks, occasionally binding constraints, or the ergodic distribution itself is the object of interest.

3.  **You need likelihood-based inference and the model is small.** When closed-form or quadrature-based likelihoods are available, full-information ML and Bayesian estimation with a particle filter dominate any simulation-based approach in efficiency. The deep-surrogate route in {ref}`ch-estimation` pays off only when the likelihood is intractable or the simulator is much more expensive than evaluating the network.

4.  **Sample size is small and uncertainty is the answer.** At $n \lesssim 10^3$ design points and moderate input dimension, a well-specified GP with active learning ({ref}`ch-gp`) often provides stronger uncertainty quantification and competitive point prediction relative to a deep network of comparable complexity; the advantage can disappear when the dimension is high, the kernel is misspecified, or the response is strongly nonstationary. Default to GPs when the simulator is expensive and posterior uncertainty matters.

5.  **Reproducibility and audit-ability are paramount.** Deep-learning solutions depend on initialization seeds, optimizer tuning, and floating-point order of operations on the GPU. In central-bank or regulatory contexts where round-off-stable, configuration-light reproducibility is required, a deterministic finite-difference scheme remains preferable; bit-exactness across heterogeneous BLAS/LAPACK back-ends is not guaranteed for either approach, but FD typically has fewer moving parts to pin. At minimum, pin every random seed and report hardware/software versions (cf. the reproducibility appendix).

6.  **Training instability is a model-misspecification signal.** If a DEQN refuses to converge, the diagnosis is rarely "add more layers", it is usually a poorly specified loss, a missing hard constraint, an unstable simulation, or a model that admits no equilibrium near the trial states. Spending a day debugging a divergent training run is often the discovery that the model itself needs to change.

The chapters of this script argue that, in the right regime, deep learning unlocks problems that are out of reach of any other tool. The point of this short list is the converse: there is a long catalog of standard problems where it is the wrong tool, and recognizing them keeps the methodology honest.

## Practical Tips and Common Pitfalls

The following guidelines distill recurring lessons from the applications covered in this course:

1.  **Start simple.** Begin with a small network (2–3 hidden layers, 32–64 neurons) and a low-dimensional test case with a known solution (e.g., Brock–Mirman for DEQNs, the 1D ODE for PINNs). Only increase complexity once the simple case works.

2.  **Normalize states, controls, and residuals.** Many failed training runs are scaling failures rather than approximation failures. Work with logs, ratios, standardized states, or economically natural units, and scale residuals so that one equation does not dominate the loss only because it is measured in larger units. In climate and OLG models, nondimensionalization is often as important as the architecture.

3.  **Build in hard constraints where possible.** Positivity of consumption, bounds on savings or abatement, no-arbitrage restrictions, and budget feasibility should be enforced by output transformations or hard ansatzes whenever they can be. Penalty terms are useful, but a network that cannot violate a constraint is usually easier to train and easier to trust; this is also where monotonicity and concavity restrictions on policy or value functions are best imposed.

4.  **Use independent validation states and shocks.** Report residuals on held-out state points, independent shock sequences, boundary states, and economically important stress scenarios. A solver that performs well only on the training simulation path has not solved the global model; this is the practical counterpart to the distribution-shift frontier discussed in {ref}`sec-open_challenges`.

5.  **Monitor all loss components.** In multi-term losses (PDE residual $+$ boundary penalties, or Euler residuals $+$ complementarity), log each component separately. A declining total loss can mask a rising boundary term.

6.  **Use adaptive loss balancing.** Manual tuning of penalty weights $\lambda$ is fragile. ReLoBRaLo ({ref}`ch-nas`) and related schemes automatically balance competing loss terms; in the benchmarks reported by {cite:t}`bischof2025relobralo` they substantially improve accuracy over fixed weights, with the magnitude of the gain depending on the problem.

7.  **Check economic diagnostics, not just the loss.** Verify that policy functions satisfy economic intuition (e.g., consumption increasing in wealth), that market clearing holds, and that the relative Euler error is economically small (e.g., median or mean below $10^{-3}$). Pick one convention (relative Euler error, $\log_{10}$ Euler error, or consumption-equivalent error) and use it consistently across diagnostics.

8.  **Choose activations deliberately.** In strong-form PINNs that involve second-order PDEs, use $C^\infty$ activations such as $\tanh$, Swish, or softplus, since ReLU has $u'' = 0$ a.e.; weak-form formulations and methods that handle nonsmooth solutions explicitly can still use ReLU. Swish is a strong default for DEQNs. Avoid mixing activation types across layers unless there is a specific reason.

9.  **Use common random numbers (CRN).** When comparing model outputs across parameter values (e.g., in surrogate-based estimation), always fix the shock sequence. CRN can reduce variance substantially in surrogate-SMM pipelines (often by an order of magnitude or more), making gradient-based optimization feasible.

**Common failure modes.** {numref}`tab-dl_failure_modes` summarizes recurring failure modes and their mitigations:

````{table}
:name: tab-dl_failure_modes

Recurring failure modes encountered when training the deep-learning solvers of this course (DEQNs, PINNs, deep surrogates, GP + BAL), together with the practical mitigations that work most reliably across the companion notebooks. The list is short by design: most production failures fall into one of these categories, and the corresponding fix is usually the first thing to try before invoking heavier diagnostics.

| **Failure Mode** | **Mitigation** |
|---|---|
| Loss decreases but solution is wrong | Check individual loss components; use economic diagnostics |
| One residual dominates all others | Adaptive loss balancing (Ch. {ref}`ch-nas`); inspect per-component gradients; normalize residuals to comparable units |
| Solver looks fine on simulation path, fails off path | Add held-out validation states; sample corners and stress states; use residual-based adaptive resampling |
| Policy network outputs collapse to zero (e.g., $c \equiv 0$ or $s \equiv 0$) | Improve initialization; use simulation-based training; check that hard constraints are not forcing the trivial solution |
| NaN in loss | Clamp inputs to GP/network; check for log of negative values |
| Euler errors large in corners of state space | Increase sampling near boundaries; use hard constraints |
| Surrogate gives a precise but biased SMM optimum | Validate the surrogate near the estimated optimum; add active-learning points around the criterion minimum; cross-check with direct DEQN re-evaluation |
| GP posterior variance too large | Add training points via BAL; check kernel hyperparameters |
| Training succeeds only for one seed | Report multi-seed dispersion; reduce learning rate; tighten scaling; check residual signs |
````

**Implementation checklist.** For each new model, we recommend the following workflow:

1.  Solve a simplified version with a known analytical solution (e.g., Brock–Mirman).

2.  Verify that all equilibrium conditions are correctly encoded in the loss.

3.  Train with a small network and monitor all loss components separately.

4.  Check economic diagnostics: policy function shapes, market clearing, Euler errors.

5.  Scale up: increase network size, dimension, and training budget.

6.  Document the final configuration and report reproducibility information.

## Concluding Remarks

The methods presented in this course are not merely computational tricks: they expand the feasible frontier of quantitative economic modeling when state spaces are high-dimensional, equilibrium restrictions are differentiable, and repeated model evaluation is valuable. They do not replace classical numerical economics; they add a flexible amortized layer that is most powerful when combined with economic structure, careful diagnostics, and classical benchmarks.

The convergence of abundant compute, mature software ecosystems (TensorFlow, PyTorch, JAX), and increasingly complex economic questions ensures that deep learning methods will play an expanding role in economic research and central bank practice in the years ahead. We hope that this course has provided the conceptual foundations and practical tools needed to engage with this frontier.

For further reading, we refer to the comprehensive survey by {cite:t}`fernandezvillaverde2024taming`, the methodological foundations laid in {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, and the applications in {cite:t}`rennerscheidegger_2018`, {cite:t}`friedlDeep2023`, {cite:t}`han2023deepham`, {cite:t}`payne2025deepsam`, {cite:t}`kase2022estimating`, {cite:t}`chen2026Deep`, {cite:t}`kubler2025using`, and {cite:t}`fernandezvillaverde2025climate`. The list of references in these notes is necessarily incomplete; for a full bibliography, we refer the reader to the cited papers and the references therein.

```{prf:remark} Chapter Summary

- All four method families share one paradigm: *neural network as function approximator* $+$ *economic structure in the loss* $+$ *automatic differentiation for training*.

- Knowing when *not* to use deep learning is part of using it well: classical projection or perturbation often dominates in low dimension; exact likelihood beats simulation when both are available; a well-specified GP is often the better choice at small $n$ and moderate dimension, when uncertainty is the deliverable.

- Open frontiers: convergence theory, hybrid global/local methods, real-time policy analysis, large structural estimation, climate–economy IAMs, full HA models with aggregate shocks, operator learning.

- The right question for a research project is rarely "which of these four methods?" but "which combination?", e.g. DEQN $+$ surrogate for estimation, PINN $+$ GP for sensitivity, or all of the above for climate policy.
```


## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch12-1

**[Core\] Method-choice scenario.** You are handed (a) a 4-state monetary-policy DSGE with smooth shocks, (b) a 200-agent OLG with progressive taxation, (c) an option-pricing problem on an irregularly shaped exotic payoff, and (d) a climate-IAM where uncertainty in the SCC is the deliverable. Justify which method you would use for each, and what hybrid you would consider. For each case, state the classical baseline you would compare against before reaching for a deep-learning method.
```

```{exercise}
:label: ex-ch12-2

**[Core\] When NOT to use deep learning.** Pick one of the six regimes from {ref}`sec-when_not_to_use_dl` and write a one-page note for a colleague explaining why a classical method dominates.
```

```{exercise}
:label: ex-ch12-3

**[Computational\] Reproducibility audit.** Take any notebook from this script and document hardware, software versions, and seeds. Re-run end-to-end and report the maximum deviation in any reported number.
```

```{exercise}
:label: ex-ch12-4

**[Advanced/project\] Open-ended.** Identify one frontier from {ref}`sec-open_challenges` (operator learning, hybrid global/local, real-time policy, large structural estimation, climate-economy IAMs, HA with aggregate shocks) and sketch a 6-month research project that combines two methods from this script.
```

```{exercise}
:label: ex-ch12-5

**[Advanced/project\] Hybrid pipeline: DEQN $+$ GP $+$ SMM end-to-end.** Build a complete estimation pipeline on the Brock–Mirman model with two parameters $(\beta, \varrho)$, matching the parameter pair worked in the companion notebook [`lecture_15_03b_Structural_Estimation_BM_Joint.ipynb`](notebooks/lecture_15_structural_estimation_smm/lecture_15_03b_Structural_Estimation_BM_Joint.ipynb). As that notebook documents, $\beta$ is only *partially* identified by macro moments (a visible ridge along the $\beta$ axis); $\varrho$ has the cleaner identification map. The exercise below is a controlled-difficulty version of that pipeline.

1.  Train a DEQN with *pseudo-state augmentation*: extend the network input from $(K_t, z_t)$ to $(K_t, z_t, \beta, \varrho)$.

2.  On a $20\times 20$ grid of $(\beta, \varrho) \in [0.92, 0.99] \times [0.50, 0.99]$, simulate the model and compute four moments: mean savings rate, mean consumption growth, $\mathrm{Var}(\log C_t)$, and $\mathrm{Var}(K_t)$.

3.  Fit a GP surrogate $(\beta, \varrho) \mapsto \bm m$ on the resulting $400$-point dataset.

4.  Generate observed moments at $(\beta^\star,\varrho^\star)=(0.96,0.90)$ from one $T=200$ simulation and solve the GP-based SMM problem.

Report the wall-clock time of each step, total pipeline time, estimation errors, a comparison to a naive SMM that re-solves the DEQN at each candidate $(\beta,\varrho)$, the hold-out surrogate error on $(\beta,\varrho)$ pairs outside the $20\times 20$ grid, a contour plot of the GP-based SMM criterion that exposes the $\beta$ ridge, and a direct DEQN re-evaluation at the GP optimum to test surrogate bias. Evaluate whether the surrogate-based optimization is faster after accounting for the one-time GP fitting overhead and whether its accuracy is comparable. Bonus: bootstrap confidence intervals for $(\hat\beta,\hat\varrho)$ using the parametric bootstrap of {prf:ref}`ex-ch10-6`.
```

````{exercise}
:label: ex-ch12-6

**[Advanced/project\] DeepONet for a parameterized HJB.** Consider the cake-eating HJB

```{math}
:enumerated: false

\rho V(a; \gamma) = \max_c \left[\frac{c^{1-\gamma}}{1-\gamma} + V'(a; \gamma)(ra - c)\right],
\qquad \gamma \in [1.5,5],
```

This is deliberately a low-dimensional operator-learning toy: since the branch input is the scalar $\gamma$, a parameter-conditioned PINN would be an equally natural baseline. The purpose of the exercise is to expose the DeepONet branch–trunk decomposition before moving to function-valued branch inputs (e.g., a state-dependent drift $r(a)$ or discount rate $\rho(a)$ observed at sensor points).

1.  Sketch the DeepONet architecture for learning the operator $\mathcal{G}: \gamma \mapsto V(\cdot;\gamma)$: identify the branch-net input and the trunk-net input, and write the predicted output as the inner product of branch and trunk outputs.

2.  State the operator-learning universal approximation theorem of {cite:t}`lu2021learning` and explain why a DeepONet of moderate width can approximate any continuous operator on a compact set.

3.  Suppose you want $V(\cdot;\gamma)$ for $N=50$ values of $\gamma$. Compare $N$ independent PINN runs at cost $C_\mathrm{PINN}$ each with one DeepONet run at cost $C_\mathrm{DON}$. At what ratio $C_\mathrm{DON}/C_\mathrm{PINN}$ does operator learning win, and how does the break-even ratio scale with $N$?

4.  Discuss two limitations: extrapolation outside $[1.5,5]$ and preservation of structural properties such as concavity of $V$ in $a$.
````
