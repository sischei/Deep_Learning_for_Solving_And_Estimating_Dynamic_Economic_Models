---
title: "The International Real Business Cycle Model"
label: ch-irbc
---

Having established the DEQN framework on the one-dimensional Brock–Mirman model in {ref}`ch-deqn`, we now scale it to the multi-country international real business cycle (IRBC) model of {cite:t}`backus1992international`. This model features $N$ countries with heterogeneous productivity, complete markets, irreversible investment, and convex capital adjustment costs. It is the standard testbed for high-dimensional solution methods in macroeconomics, and applying DEQNs to it illustrates how the framework handles high-dimensional state spaces, multiple equilibrium conditions, and complementarity constraints.

(sec-irbc_motivation)=
## Why IRBC for Macro-Finance Research?
Beyond its computational-testbed role, the IRBC model is the workhorse framework for *open-economy asset pricing* and *international risk sharing*. Several first-order questions in macro-finance can be posed sharply within it:

- **International risk sharing.** Under complete markets and homogeneous preferences, the planner's allocation implies perfectly correlated consumption growth across countries, whereas the data show consumption correlations that are lower than output correlations (the *consumption-correlation puzzle* of {cite:t}`backus1992international`): the benchmark BKK model predicts a cross-country consumption correlation close to 1, while the empirical correlation between the US and a typical industrial country is in the range 0.3–0.5 and is systematically below the corresponding output correlation. IRBC extensions with incomplete markets, frictions, or heterogeneous preferences are designed precisely to close this gap.

- **Asset-market structure and welfare.** {cite:t}`heathcote2002financial` use an IRBC-style setup to quantify the welfare cost of moving from complete markets to one-bond economies ("financial autarky"), obtaining a welfare cost of the same order of magnitude as the business cycle itself. More generally, variants of this model are the standard laboratory for comparing complete vs. incomplete market structures.

- **Capital flows and current-account dynamics.** With intertemporal savings and heterogeneous productivity, the IRBC delivers persistent current-account imbalances as an equilibrium outcome rather than as a reduced-form residual. This is the starting point for the modern open-economy DSGE literature.

- **Home bias.** The frictionless benchmark of near-unit consumption correlation is the anchor against which observed portfolio home bias must be explained; {cite:t}`heathcote2013international` show that accounting for nontraded goods and labor-income hedging substantially narrows the gap between theory and observed portfolios.

- **Macro-financial transmission.** Adjustment costs, borrowing constraints, and Pareto weights become levers for studying how financial frictions propagate across borders. This is the class of extensions that motivates the DEQN treatment: once frictions are added, the policy functions acquire kinks and nonlinearities that are hard to handle with traditional grid-based methods.

The IRBC model is therefore an interesting substantive object, not merely a scaling test. Its combination of a clean complete-markets benchmark and rich, realistic frictions makes it a natural next step after the one-country Brock–Mirman benchmark of {ref}`ch-deqn`.

**A calibration caveat for the puzzles above.** The shock decomposition of {ref}`sec-irbc_setup` below, $z^{j\prime} = \rho_z z^j + \sigma_e(\varepsilon^j + \varepsilon^{\mathrm{agg}})$, hard-wires a cross-country innovation correlation of exactly $1/2$ for any number of countries $N$. The consumption-correlation and Backus–Smith puzzles cited in the bullets above should therefore be read as statements about this specific calibration: richer correlation structures, country-specific factor loadings, or fewer aggregate factors would change the quantitative bite of the puzzles in this model. This is calibration, not theory.

(sec-irbc_setup)=
## Model Setup
````{table}
:name: tab-irbc_symbols

Symbol cheat-sheet for the IRBC model. Note the IES-vs-CRRA convention: here $\gamma_j$ is the intertemporal elasticity, and the implied risk aversion is $1/\gamma_j$; later chapters on continuous-time HA models and climate use $\gamma$ for CRRA and $\psi$ for IES.

| **Symbol** | **Role** | **Range / sign** | **Calibration** |
|---|---|---|---|
| \_j | IES of country $j$ (*not* CRRA) | $>0$ | $[0.25, 1.0]$ linearly spaced |
| \^j | Pareto weight on country $j$ | $>0$ | $(A_{\mathrm{tfp}}-\delta)^{1/\gamma_j}$ |
| \_t | Aggregate resource-constraint multiplier | $>0$ | $\lambda_{\mathrm{ss}} = 1$ |
| \_t\^j | Irreversibility KKT multiplier on $I^j \ge 0$ | $\ge 0$ | $0$ in slack regime |
| A\_ | TFP normalization constant | $>0$ | $\approx 0.0559$ |
|  | Capital share in Cobb–Douglas | $\in (0,1)$ | $0.36$ |
| \^j | Quadratic adjustment-cost level | $\ge 0$ | $\kappa=0.50$ |
| \_z | TFP persistence | $\in [0,1)$ | $0.95$ |
| \_e | Innovation s.d. per component | $>0$ | $0.01$ |
| \^j | Idiosyncratic innovation | $\mathcal N(0,1)$ | i.i.d. across $j,t$ |
| \^ | Aggregate innovation | $\mathcal N(0,1)$ | common factor |
|  | Adjustment-cost intensity | $\ge 0$ | $0.50$ |
````

The international real business cycle (IRBC) model, introduced by {cite:t}`backus1992international`, extends the single-country growth model to $N$ heterogeneous countries, each endowed with country-specific capital $k^j$ and total factor productivity $z^j$. The model features complete markets, irreversible investment, and convex capital adjustment costs, and serves as the workhorse test case for high-dimensional solution methods {cite:p}`ECTA:ECTA1716`. Here, we apply the DEQN methodology of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` to this setting.

**Preferences.** Each country $j$ has CRRA utility

$$
u^j(c) \;=\; \begin{cases}
\dfrac{c^{1-1/\gamma_j} - 1}{1 - 1/\gamma_j}, & \gamma_j \neq 1,\\[6pt]
\ln c, & \gamma_j = 1,
\end{cases}
$$

where the intertemporal elasticity of substitution (IES) $\gamma_j$ is heterogeneous across countries; risk aversion under this CRRA specification equals $1/\gamma_j$. **Notation warning:** this chapter uses $\gamma$ for the IES, while later chapters on continuous-time HA models and climate use $\gamma$ for CRRA risk aversion and $\psi$ for the IES. The convention is stated explicitly at the start of each chapter. A social planner maximizes

$$
\max \; \sum_{t=0}^{\infty} \beta^t \, \E{\sum_{j=1}^{N} \tau^j \, u^j(c_t^j)}
$$ (eq-irbc_utility)

with Pareto weights $\tau^j > 0$.

**Production.** Country $j$ produces $Y^j = A_\mathrm{tfp} \exp(z^j)(k^j)^\zeta$, where the total factor productivity constant $A_\mathrm{tfp}$ is calibrated to normalize the steady-state capital stock to unity. In steady state (where $z^j = 0$, $k^j = 1$, and $k^{j\prime} = 1$), the Euler equation implies:

$$
A_\mathrm{tfp} = \frac{1/\beta - 1 + \delta}{\zeta}.
$$ (eq-atfp)

This normalization ensures that the deterministic steady state lies at $(k^\star, z^\star) = (1, 0)$ for all countries, which simplifies the network's learning task and provides a natural center for the training distribution.

**TFP process.** Log productivity follows an AR(1) with common and idiosyncratic shocks:

$$
z^{j\prime} = \rho_z z^j + \sigma_e(\varepsilon^j + \varepsilon^\mathrm{agg}), \qquad \varepsilon^j, \varepsilon^\mathrm{agg} \sim \mathcal{N}(0,1)\text{ i.i.d.}, \qquad |\rho_z| < 1
$$ (eq-irbc_tfp)

The persistence restriction $|\rho_z|<1$ guarantees stationarity of the TFP process, which in turn underlies the existence of an ergodic distribution on which DEQN training samples ({ref}`sec-deqn_algo`). Here $\sigma_e$ is the per-component standard deviation, so the marginal innovation variance for country $j$ is $2\sigma_e^2$ and the cross-country innovation covariance is $\sigma_e^2$. These two facts imply a fixed cross-country innovation correlation of $1/2$ regardless of $N$, a direct consequence of the equal-weighted aggregate-shock decomposition $\varepsilon^j + \varepsilon^\mathrm{agg}$. Asset-pricing implications (in particular the international consumption-correlation puzzle and the cyclicality of trade balances) inherit this hard-wired common-factor structure: results below should be interpreted with that calibration choice in mind. If a desired total innovation scale $\bar\sigma$ is targeted instead, set $\sigma_e = \bar\sigma/\sqrt{2}$.

**Adjustment costs and irreversibility.** Changing the capital stock incurs a quadratic adjustment cost:

$$
\Gamma^j = \frac{\kappa}{2}\, k^j \left(\frac{k^{j\prime}}{k^j} - 1\right)^{\!2},
$$ (eq-irbc_adjcost)

with marginal derivatives that appear in the Euler equations: 

(eq-irbc_adjcost_derivs)=

$$
\begin{aligned}
\frac{\partial \Gamma^j}{\partial k^{j\prime}} &= \kappa\left(\frac{k^{j\prime}}{k^j} - 1\right), &
\frac{\partial \Gamma^j}{\partial k^j} &= \frac{\kappa}{2}\left(1 - \left(\frac{k^{j\prime}}{k^j}\right)^{\!2}\right).
\end{aligned}
$$

Note that $\partial\Gamma^j/\partial k^j$ is *negative* whenever $k^{j\prime} > k^j$, i.e. in expanding states. Consequently the term $-\partial\Gamma^j/\partial k^j$ that appears in the marginal product of capital below {eq}`eq-irbc_mpk` *raises* MPK in expansion phases; a reader who plugs in $|\partial\Gamma/\partial k|$ here will introduce a sign error. Investment is irreversible: $I^j = k^{j\prime} - (1-\delta)k^j \geq 0$.

**Pareto-weight calibration.** With heterogeneous IES $\gamma_j$, a symmetric deterministic steady state is most easily obtained by choosing the Pareto weights as

$$
\tau^j \;=\; \bigl(A_\mathrm{tfp} - \delta\bigr)^{1/\gamma_j}, \qquad j=1,\ldots,N.
$$ (eq-pareto_calibration)

The derivation is a two-step inversion of the planner's first-order condition. The consumption-sharing condition {eq}`eq-irbc_consumption` (derived in the next section from the FOC for $c^j_t$) reads $\tau^j (c^j_t)^{-1/\gamma_j} = \lambda_t$, so $c^j_t = (\lambda_t / \tau^j)^{-\gamma_j}$. In the deterministic steady state with the normalizations $\lambda_\mathrm{ss} = 1$ and $k^j_\mathrm{ss} = 1$ we want every country to consume the same amount $c^j_\mathrm{ss} = A_\mathrm{tfp} - \delta$ implied by the resource constraint $c^j_\mathrm{ss} = Y^j_\mathrm{ss} - I^j_\mathrm{ss}$. Setting $(1/\tau^j)^{-\gamma_j} = A_\mathrm{tfp} - \delta$ and solving for $\tau^j$ gives Eq. {eq}`eq-pareto_calibration`. The symmetric steady state thus serves as a natural anchor for training: the network's initial predictions need only match this point to avoid infeasible economies during the early simulated trajectories.

**Reference calibration.** Throughout the companion notebooks [`lecture_04_01_IRBC_DEQN_smooth.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_01_IRBC_DEQN_smooth.ipynb) and [`lecture_04_02_IRBC_DEQN_irreversible.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_02_IRBC_DEQN_irreversible.ipynb), we use the quarterly calibration summarized in {numref}`tab-irbc_params`. The implied total factor productivity and deterministic steady-state quantities can then be computed analytically.

````{table}
:name: tab-irbc_params

Reference IRBC calibration used in the companion notebook. Countries' IES values $\gamma_j$ are linearly spaced in $[\gamma_{\min}, \gamma_{\max}]$. Pareto weights are computed from {eq}`eq-pareto_calibration`.

| **Symbol** | **Name** | **Value** | **Description** |
|---|---|---:|---|
| $\beta$ | Discount factor | 0.99 | Quarterly |
| $\zeta$ | Capital share | 0.36 | Cobb–Douglas |
| $\delta$ | Depreciation | 0.01 | Low quarterly rate |
| $\rho_z$ | TFP persistence | 0.95 | Highly persistent |
| $\sigma_e$ | Shock std. dev. | 0.01 | Small innovations |
| $\kappa$ | Adjustment-cost intensity | 0.50 | Moderate frictions |
| $\gamma_{\min}$ | Min IES | 0.25 | Risk aversion $=4$ |
| $\gamma_{\max}$ | Max IES | 1.00 | Log utility |
| $k^\star$ | Steady-state capital | 1.00 | Normalization |
````

**Worked steady state.** Equation {eq}`eq-atfp` is most compactly written as $A_\mathrm{tfp}=(1/\beta - 1 + \delta)/\zeta$; multiplying numerator and denominator by $\beta$ gives the algebraically equivalent form $A_\mathrm{tfp}=(1-\beta(1-\delta))/(\zeta\beta)$ used below. Substituting the reference values:

$$
\begin{align}
A_\mathrm{tfp} &= \frac{1-\beta(1-\delta)}{\zeta\,\beta}
               = \frac{1 - 0.99 \cdot 0.99}{0.36 \cdot 0.99}
               \;\approx\; 0.0559, \\
Y^\star_j      &= A_\mathrm{tfp}\,(k^\star)^\zeta \approx 0.0559, \qquad
I^\star_j      = \delta\,k^\star = 0.01, \qquad
c^\star_j      = Y^\star_j - I^\star_j \approx 0.0459.
\end{align}
$$

The aggregate resource constraint {eq}`eq-irbc_arc` is then satisfied country by country, $Y^\star_j - I^\star_j - c^\star_j = 0$, as a trivial check. These numbers provide a baseline against which the trained network's predictions on an out-of-sample simulation can be compared.

(sec-irbc_planner_problem)=
## The Planner's Problem and Equilibrium Conditions
**The planner's problem.** The social planner maximizes the weighted sum of utilities across all $N$ countries, subject to the aggregate resource constraint {eq}`eq-irbc_arc`, the irreversibility constraints, the production technology, and the TFP process {eq}`eq-irbc_tfp`:

$$
\max_{\{c_t^j,\, k_{t+1}^j\}_{j,t}} \; \sum_{t=0}^{\infty} \beta^t \, \E{\sum_{j=1}^{N} \tau^j \, u^j(c_t^j)}
$$ (eq-irbc_planner)

with Pareto weights $\tau^j > 0$ and discount factor $\beta \in (0,1)$.

**The Lagrangian.** Following the same approach as in {ref}`sec-bm` for the Brock–Mirman model, we form the Lagrangian by attaching discounted multipliers to each constraint. Let $\beta^t \lambda_t$ be the multiplier on the aggregate resource constraint at date $t$, and $\beta^t \mu_t^j$ the multiplier on the irreversibility constraint for country $j$ at date $t$. The Lagrangian is:

$$
\begin{split}
\mathcal{L} = \mathbb{E}\Biggl[\sum_{t=0}^{\infty} \beta^t \Biggl(
&\sum_{j=1}^{N} \tau^j \, \frac{(c_t^j)^{1-1/\gamma_j} - 1}{1-1/\gamma_j}
+ \lambda_t \sum_{j=1}^{N} \bigl(Y_t^j + (1-\delta)k_t^j - k_{t+1}^j - \Gamma_t^j - c_t^j\bigr) \\
&+ \sum_{j=1}^{N} \mu_t^j \bigl(k_{t+1}^j - (1-\delta)k_t^j\bigr)
\Biggr)\Biggr].
\end{split}
$$ (eq-irbc_lagrangian)

The planner chooses $c_t^j$ and $k_{t+1}^j$ for each country $j$ and each date $t$. The complementary slackness conditions require $\mu_t^j \geq 0$, $I_t^j \geq 0$, and $\mu_t^j \cdot I_t^j = 0$. Two notation reminders before we differentiate. First, the irreversibility multiplier is $\mu_t^j$, not the resource-constraint multiplier $\lambda_t$; the two play different roles ($\lambda_t$ shadow-prices the aggregate goods market; $\mu_t^j$ shadow-prices country $j$'s individual investment floor) and they enter the FOCs through entirely different channels. Second, $\mu_t^j \geq 0$ is the standard KKT sign: the multiplier on a $\geq$-constraint is non-negative at the optimum, and the Fischer–Burmeister residual constructed below packages this sign restriction together with the slackness condition into a single smooth squared term that is compatible with SGD.

**FOC w.r.t. $c_t^j$:** Differentiating the Lagrangian with respect to $c_t^j$:

$$
\frac{\partial \mathcal{L}}{\partial c_t^j}
= \beta^t \bigl[\tau^j (c_t^j)^{-1/\gamma_j} - \lambda_t\bigr] = 0
\qquad\Longrightarrow\qquad
\tau^j (c_t^j)^{-1/\gamma_j} = \lambda_t.
$$ (eq-irbc_foc_c)

This is the *consumption-sharing condition*: the planner equates the Pareto-weighted marginal utility of consumption across all countries to a common shadow price $\lambda_t$. Solving {eq}`eq-irbc_foc_c` for $c_t^j$:

$$
c_t^j = \left(\frac{\lambda_t}{\tau^j}\right)^{-\gamma_j}.
$$ (eq-irbc_consumption)

This shows that all $N$ consumption levels are determined by the single variable $\lambda_t$: a higher shadow price (resources are scarcer) lowers consumption in every country. Countries with a higher IES $\gamma_j$ respond more elastically to changes in $\lambda_t$.

**FOC w.r.t. $k_{t+1}^j$:** The variable $k_{t+1}^j$ appears in three places in the Lagrangian: (i) the date-$t$ resource constraint with coefficient $-\lambda_t(1 + \partial\Gamma_t^j/\partial k_{t+1}^j)$, (ii) the date-$t$ irreversibility constraint with coefficient $+\mu_t^j$, and (iii) the date-$(t\!+\!1)$ terms via output $Y_{t+1}^j$, depreciated capital $(1-\delta)k_{t+1}^j$, adjustment costs $\Gamma_{t+1}^j$, and the irreversibility constraint. Differentiating and collecting terms:

$$
\frac{\partial \mathcal{L}}{\partial k_{t+1}^j}
= \beta^t \!\left[-\lambda_t\!\left(1 + \frac{\partial \Gamma_t^j}{\partial k_{t+1}^j}\right) + \mu_t^j\right] \\
+ \beta^{t+1}\,\mathbb{E}_t\!\left[\lambda_{t+1}\!\left(\frac{\partial Y_{t+1}^j}{\partial k_{t+1}^j} + (1-\delta) - \frac{\partial \Gamma_{t+1}^j}{\partial k_{t+1}^j}\right) - \mu_{t+1}^j(1-\delta)\right] = 0.
$$ (eq-irbc_foc_k_raw)

Now define the *marginal product of capital* (inclusive of depreciation and adjustment cost effects):

$$
\mathrm{MPK}^j \;\equiv\; 1-\delta + \zeta A_\mathrm{tfp}\exp(z^j)(k^j)^{\zeta-1} - \frac{\partial \Gamma^j}{\partial k^j},
$$ (eq-irbc_mpk)

and note from {eq}`eq-irbc_adjcost_derivs` that $\partial \Gamma_t^j / \partial k_{t+1}^j = \kappa(k_{t+1}^j/k_t^j - 1)$. Dividing {eq}`eq-irbc_foc_k_raw` by $\beta^t$ and substituting the MPK definition:

$$
\lambda_t\!\left(1 + \frac{\partial \Gamma_t^j}{\partial k_{t+1}^j}\right) - \mu_t^j
= \beta\,\mathbb{E}_t\!\bigl[\lambda_{t+1}\,\mathrm{MPK}_{t+1}^j - (1-\delta)\,\mu_{t+1}^j\bigr].
$$ (eq-irbc_euler_level)

This is the *Euler equation* for country $j$. The left-hand side is the cost of investing one more unit in country $j$'s capital: the shadow price $\lambda_t$ of the resources used (scaled by the marginal adjustment cost) minus the value $\mu_t^j$ of relaxing the irreversibility constraint. The right-hand side is the expected discounted benefit: next period's shadow price times the marginal product of capital, minus the option-value loss from tightening next period's irreversibility constraint.

**Relative error form.** For numerical purposes, regroup {eq}`eq-irbc_euler_level` so that the cost-of-investment term $\lambda_t(1+\partial\Gamma_t^j/\partial k_{t+1}^j)$ stands alone on the left, $\lambda_t(1+\partial\Gamma_t^j/\partial k_{t+1}^j) = \beta\,\mathbb{E}_t[\lambda_{t+1}\,\mathrm{MPK}_{t+1}^j - (1-\delta)\mu_{t+1}^j] + \mu_t^j$, and divide through by it. This gives a scale-free formulation:

$$
\frac{\beta\,\mathbb{E}_t\!\left[\lambda' \cdot \mathrm{MPK}^{j\prime} - (1-\delta)\mu^{j\prime}\right] + \mu^j}{\lambda(1+\partial\Gamma^j/\partial k^{j\prime})} - 1 = 0, \qquad j=1,\ldots,N.
$$ (eq-irbc_euler_relerr)

This ensures that all $N$ Euler equations are dimensionless and residuals can be interpreted directly as percentage deviations from optimality.

**Aggregate resource constraint.** All output is allocated to consumption, investment, and adjustment costs:

$$
\sum_{j=1}^{N}\bigl[Y^j + (1-\delta)k^j - k^{j\prime} - \Gamma^j - c^j\bigr] = 0.
$$ (eq-irbc_arc)

**Summary of equilibrium conditions.** The complete system consists of three blocks:

1.  **Consumption sharing** {eq}`eq-irbc_consumption`: determines all $N$ consumption levels from $\lambda_t$.

2.  **Euler equations** {eq}`eq-irbc_euler_relerr`: $N$ intertemporal optimality conditions, one per country.

3.  **Aggregate resource constraint** {eq}`eq-irbc_arc`: closes the model by equating world supply and demand.

In addition, the $N$ irreversibility constraints are enforced via complementary slackness ($\mu^j \geq 0$, $I^j \geq 0$, $\mu^j I^j = 0$).

(sec-irbc_fischer_burmeister)=
##### Fischer–Burmeister complementarity.
The irreversibility constraint is enforced via a smoothed Fischer–Burmeister residual:

$$
\mathrm{FB}_\varepsilon(\mu^j, I^j) = \mu^j + I^j - \sqrt{(\mu^j)^2 + (I^j)^2 + \varepsilon^2} = 0.
$$

The exact Fischer–Burmeister map is the limiting case $\mathrm{FB}_0(\mu,I)=\mu+I-\sqrt{\mu^2+I^2}$. Its zero set coincides with the positive axes in the $(\mu, I)$-plane, ensuring $\mu^j \geq 0$, $I^j \geq 0$, and $\mu^j \cdot I^j = 0$ ({numref}`fig-fb_zeroset`). The smoothed version with $\varepsilon > 0$ rounds the corner at the origin and is differentiable there, improving numerical conditioning at the cost of a slight relaxation of exact complementarity. The companion notebooks use $\varepsilon = 10^{-4}$ as the default; tighter values ($10^{-6}$–$10^{-5}$) are sometimes preferred when complementarity must hold to higher accuracy, at the cost of stiffer gradients near the origin.

```{figure} figures/fig-fb_zeroset.svg
:name: fig-fb_zeroset

The Fischer–Burmeister complementarity function, drawn in the investment–multiplier plane: investment $I^j$ on the horizontal axis, the irreversibility multiplier $\mu^j$ on the vertical axis. The exact map $\mathrm{FB}_0(\mu,I)=\mu+I-\sqrt{\mu^2+I^2}$ packs the three Karush–Kuhn–Tucker conditions $\mu\ge 0$, $I\ge 0$, $\mu I=0$ into a single smooth equation: $\mathrm{FB}_0=0$ holds *exactly* on the two heavy blue half-axes and nowhere else. The horizontal half-axis ($\mu=0$, $I>0$) is the *investing* regime, where the country invests a strictly positive amount, the irreversibility constraint is slack, and its shadow price $\mu$ is therefore zero. The vertical half-axis ($I=0$, $\mu>0$) is the *constrained* regime, where the constraint binds, investment is pinned at zero, and $\mu>0$ measures how much the planner would pay to relax it; the origin is the knife-edge where both hold with equality. The open interior of the first quadrant ($\mu>0$ *and* $I>0$ together) is *infeasible* because it violates complementarity, and there $\mathrm{FB}_0>0$ strictly (since $\mu+I>\sqrt{\mu^2+I^2}$ whenever both are positive). This is exactly what makes the function useful as a loss term: when the network's predicted $(\mu^j,I^j)$ lands in that forbidden region, the squared residual $\mathrm{FB}_\varepsilon^2$ is positive and its negative gradient $-\nabla\mathrm{FB}_0$ (green arrow) pushes the iterate back toward the nearest feasible half-axis, so the network learns which regime applies at each state without any explicit regime switch. The exact map has a single kink, at the origin; the smoothed version $\mathrm{FB}_\varepsilon(\mu,I)=\mu+I-\sqrt{\mu^2+I^2+\varepsilon^2}$ actually used in the code rounds that corner, restoring differentiability everywhere at the price of an $\mathcal{O}(\varepsilon)$ relaxation of exact complementarity.
```

The complementarity conditions $\mu^j \geq 0$, $I^j \geq 0$, $\mu^j \cdot I^j = 0$ have a natural economic interpretation: when investment is strictly positive ($I^j > 0$), the irreversibility constraint is slack and the multiplier is zero ($\mu^j = 0$); conversely, when the constraint binds ($I^j = 0$), the multiplier is positive, reflecting the shadow value of the binding constraint. The FB function smoothly encodes both regimes, allowing the neural network to learn which regime applies for each state without explicit regime switching.

```{prf:remark} Why Fischer–Burmeister works so well in DEQNs

 The squared FB residual converts a discrete regime-switching problem (constraint slack vs binding) into a smooth gradient field that SGD can navigate. Three properties matter: (i) the zero set of $\mathrm{FB}_0$ *exactly* coincides with the KKT complementarity axes, so a converged network satisfies the constraint structure to whatever tolerance the loss is driven; (ii) the residual is smooth everywhere away from the origin, so backpropagation through it is well behaved; and (iii) the $\varepsilon^2$ smoothing rounds the single remaining kink at the origin, restoring differentiability there at the price of an $\mathcal{O}(\varepsilon)$ relaxation of exact complementarity. In the IRBC context, the network learns which states fall on the "investing" axis and which on the "constrained" axis without ever being told which regime applies, a major saving over methods that require manual regime indicators.
```


## DEQN Formulation

**From Brock–Mirman to IRBC.** It is useful to see the IRBC as the natural extension of the one-country benchmark of {ref}`ch-deqn`. {numref}`tab-bm_vs_irbc` summarizes what changes.

````{table}
:name: tab-bm_vs_irbc

The DEQN template is the same in both cases; only the input/output dimensions, the number of loss terms, and the presence of complementarity constraints change.

|  | **Brock–Mirman (Ch. {ref}`ch-deqn`)** | **IRBC (this chapter)** |
|---|---|---|
| Countries | 1 | $N$ |
| States | $(K, z)$ | $(k^1,\ldots,k^N, z^1,\ldots,z^N)$ |
| Policies | $C$ | $(k^{1\prime},\ldots,k^{N\prime}, \lambda, \mu^1,\ldots,\mu^N)$ |
| Loss terms | 1 Euler | $N$ Euler $+$ 1 ARC $+$ $N$ Fischer–Burmeister |
| Constraints | none | irreversibility, convex adjustment costs |
| Shocks per period | 1 | $N+1$ (one idiosyncratic per country + one aggregate) |
| Output activation | softplus or sigmoid | softplus |
| Analytical solution | yes (log utility, $\delta=1$) | no |
````

The full system of equations comprises $N$ Euler equations, $N$ Fischer–Burmeister conditions, and 1 aggregate resource constraint, totaling $2N+1$ equations. {numref}`tab-irbc_scalability` summarizes how the problem dimensions scale with $N$.

````{table}
:name: tab-irbc_scalability

Scaling of the IRBC state, policy, equation, and quadrature dimensions with the number of countries $N$. The state, policy, and equation counts grow linearly. Tensor-product Gauss–Hermite quadrature grows as $Q^{N+1}$, while the Stroud-3 monomial rule uses only $2(N+1)$ nodes; this is why the notebook uses Gauss–Hermite only for the two-country classroom case and switches to monomial or QMC rules in larger IRBC applications.

| $N$ | States | Policies | Equations | Shock dim. | GH nodes ($Q=3$) | Stroud-3 nodes |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 5 | 5 | 3 | $27$ | $6$ |
| 5 | 10 | 11 | 11 | 6 | $729$ | $12$ |
| 10 | 20 | 21 | 21 | 11 | $1.8\times 10^5$ | $22$ |
| 50 | 100 | 101 | 101 | 51 | $\sim 2.2\times 10^{24}$ | $102$ |
| 100 | 200 | 201 | 201 | 101 | $\sim 1.5\times 10^{48}$ | $202$ |
````

```{figure} figures/fig-irbc_quad_cost.svg
:name: fig-irbc_quad_cost

Quadrature-cost crossover for the IRBC model as a function of the number of countries $N$. Tensor-product Gauss–Hermite (red) grows exponentially in $N$ and becomes infeasible by $N=10$; the Stroud-3 monomial rule (blue) grows linearly and stays well under $10^3$ nodes even at $N=100$. This is the operational reason every IRBC application beyond the classroom $N=2$ case uses monomial or QMC integration.
```

The neural network maps the full state vector $\bm{s} = (k^1,\ldots,k^N, z^1,\ldots,z^N) \in \R^{2N}$ to all $2N+1$ policy variables $(k^{1\prime},\ldots,k^{N\prime}, \lambda, \mu^1,\ldots,\mu^N)$ simultaneously through the small Swish–softplus network in {numref}`fig-irbc_nn_arch`.

```{figure} figures/fig-irbc_nn_arch.svg
:name: fig-irbc_nn_arch

Reference network architecture used for the $N$-country IRBC model. The diagram shows the irreversible companion notebook ([`lecture_04_02_IRBC_DEQN_irreversible.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_02_IRBC_DEQN_irreversible.ipynb)): two hidden layers of 64 Swish units mapping the $2N$-dimensional state to a $2N+1$-dimensional output ($N$ capital choices, the resource-constraint multiplier $\lambda$, and the $N$ irreversibility multipliers $\mu^j$); softplus on the $\lambda$ and $\mu^j$ heads enforces non-negativity, and capital choices use the bounded growth head described below. The smooth-benchmark companion ([`lecture_04_01_IRBC_DEQN_smooth.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_01_IRBC_DEQN_smooth.ipynb)) drops the $\mu^j$ block, leaving an $N+1$-dimensional output head and no Fischer–Burmeister residual; in both notebooks the capital head is parameterized as the bounded log-growth $k_{t+1}^j = k_t^j\exp\{\bar g\,\tanh r_j(\bm s)\}$ (smooth) or the additive form $k_{t+1}^j = (1-\delta)k_t^j + \mathrm{softplus}(r_j)$ (irreversible), both of which keep $k_{t+1}^j > 0$ by construction.
```

The hidden layers use the Swish activation $\mathrm{swish}(x) = x \cdot \sigma(x)$, while the output layer employs the softplus function $\ln(1+e^x)$ to keep the multipliers and capital choice positive. Two approximation caveats deserve emphasis. First, $\mathrm{softplus}(x) > 0$ for all $x$, so the multipliers $\mu^j$ are strictly positive rather than exactly zero when the constraint is slack; complementarity is enforced only approximately. Second, irreversibility requires $I^j = k^{j\prime} - (1-\delta)k^j \geq 0$; a softplus on $k^{j\prime}$ alone does *not* enforce this, since the network can output a positive $k^{j\prime}$ that nonetheless implies negative investment. A cleaner alternative is to output investment directly via $I^j = \mathrm{softplus}(r^j)$ and set $k^{j\prime} = (1-\delta)k^j + I^j$, which hard-enforces the constraint by construction.

The total DEQN loss aggregates the equilibrium conditions. In the smooth benchmark (companion notebook [`lecture_04_01_IRBC_DEQN_smooth.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_01_IRBC_DEQN_smooth.ipynb)) only the Euler and aggregate-resource-constraint residuals appear:

$$
\ell^{\mathrm{smooth}}_\rho = \frac{1}{N_s} \sum_{i=1}^{N_s} \left[
\sum_{j=1}^{N} \bigl(\mathrm{Euler}^j(\bm{s}_i)\bigr)^2
+ \bigl(\mathrm{ARC}(\bm{s}_i)\bigr)^2
\right].
$$ (eq-irbc_loss_smooth)

The irreversibility extension (companion notebook [`lecture_04_02_IRBC_DEQN_irreversible.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_02_IRBC_DEQN_irreversible.ipynb)) augments {eq}`eq-irbc_loss_smooth` with the Fischer–Burmeister complementarity block:

$$
\ell^{\mathrm{irrev}}_\rho = \ell^{\mathrm{smooth}}_\rho \;+\; \frac{1}{N_s}\sum_{i=1}^{N_s} \sum_{j=1}^{N} \bigl(\mathrm{FB}^j(\bm{s}_i)\bigr)^2,
$$ (eq-irbc_loss)

where $N_s$ is the number of training states. When the individual loss components differ in magnitude across countries (which is typical when countries differ in size or calibration), an adaptive loss-balancing scheme from {ref}`ch-nas` (e.g., ReLoBRaLo, SoftAdapt, GradNorm) can be applied to reweight the components during training.

**Representative implementation.** The architecture is a 2-hidden-layer Swish network with a softplus output head. In the smooth benchmark the head has dimension $N + 1$ (the $N$ capital choices and the resource-constraint multiplier $\lambda$); in the irreversible extension the head expands to $2N + 1$, adding the irreversibility multipliers $\mu^j \ge 0$ (softplus enforces non-negativity by construction). Only the irreversible loss carries a non-textbook line, the Fischer–Burmeister smoothing of the complementarity $0 \le \mu^j \perp I^j \ge 0$:

```{code-block} text
:name: lst-irbc_fb
:caption: Fischer--Burmeister smoothing of $\mu \perp I$ (irreversible companion notebook only).

def fischer_burmeister(mu, I, eps=1e-4):
    return mu + I - tf.sqrt(mu**2 + I**2 + eps**2)
```
This residual is then squared elementwise and averaged across the mini-batch and across the $N$ countries, in line with the squared-residual treatment of the Euler and ARC blocks; that elementwise square is what makes the gradient field push iterates toward the complementarity axes (see {numref}`fig-fb_zeroset`). Inside the per-batch cost function of the irreversible notebook, this residual is squared and averaged alongside the Euler-equation residual (whose conditional expectation is handled by the Stroud-3 monomial rule of {ref}`sec-monomial_cubature` – $2(N+1)$ nodes for the $N$ idiosyncratic and one aggregate shock) and the aggregate-resource-constraint residual. The smooth companion implements the same `compute_cost` pipeline with the $\mu^j$ outputs and the FB block removed.

(sec-irbc_persistent_simulation)=
## Persistent-Simulation Training
The companion notebooks train the IRBC DEQN with a single training pipeline: a continuing ensemble of stochastic trajectories that evolves alongside the policy network. There is no Phase 1 / Phase 2 switch and no reset to the steady state between training segments.

```{prf:definition} Persistent-Simulation Training

 Maintain a vector of $M$ stochastic trajectory heads $\bm{X}^{(1)}_t,\ldots,\bm{X}^{(M)}_t$. Each *training segment* simulates these heads forward for $T$ stochastic periods under the *current* policy network, flattens the simulated states into a training cloud of size $M\cdot T$, performs a fixed number of SGD passes on that cloud, and then continues from the segment's terminal states $\bm{X}^{(m)}_{t+T}$. The trajectory ensemble therefore co-evolves with the policy and is never reset to the steady state.
```


What makes the single-pipeline approach feasible is that both companion notebooks parameterize the policy so that capital cannot leave the feasible set, even at random initialization. In the smooth notebook the network outputs a bounded log-growth term, $k_{t+1}^j = k_t^j\exp\{\bar g\,\tanh r_j(\bm{s})\}$, which keeps $k_{t+1}^j$ strictly positive and per-period capital growth bounded by $\exp\{\pm\bar g\}$. In the irreversible notebook the policy network outputs an investment fraction shaped by a sigmoid head and the law of motion $k_{t+1}^j = (1-\delta)k_t^j + I^j$ is hard-coded with $I^j \ge 0$. Either choice removes the reason historical implementations needed a uniform-sampling burn-in: the simulation cannot diverge.

A `SAMPLING_MODE` switch (`simulation` vs `exogenous`) is exposed for ablation studies and debugging, exogenous sampling on a wide box can be useful to confirm that a finding is not an artefact of the ergodic set, but the default `simulation` mode runs for the entire training horizon without a phase change.

A typical schedule on the two-country benchmark uses $M = 10$ trajectories of length $T = 256$ per segment, a batch size of $256$, and one or a small number of optimizer passes per segment, with Adam at learning rate $\eta \sim 10^{-3}$ and a cosine decay; convergence is read off the diagnostics of the next section rather than off a phase-transition criterion. As a budgeting reference, the companion notebooks typically run on the order of $200$–$500$ training segments before mean Euler errors drop below $10^{-3}$ on a held-out trajectory.

```{prf:remark} Why persistent-simulation training works

 Three properties make the single-pipeline approach robust. First, the bounded capital-growth heads of the previous section keep the simulation feasible at every weight setting, so there is no need for a separate uniform-sampling warm-up phase to prevent the trajectories from diverging. Second, because the training cloud co-evolves with the policy, the network is always trained on states drawn from the current policy's ergodic distribution, which is the same distribution out-of-sample evaluation will face; there is no train/test distributional shift. Third, the lack of a phase-transition criterion makes the protocol model-agnostic: scaling from $N=2$ to $N=10$ requires only changing the network's input dimension and the number of equations in the loss, not redesigning the training schedule. The trade-off is that early-training states reflect a poor and rapidly changing policy, so a small replay buffer or generous mini-batch size is helpful to keep the gradient signal stable.
```


## Results and Scalability

The DEQN approach has been successfully applied to IRBC models with up to $N=100$ countries (200 state variables, 201 policy outputs), producing equilibrium errors below $10^{-3}$ in all Euler equations, a level comparable to the best existing solution methods at a fraction of the computational cost, while substantially mitigating curse-of-dimensionality effects in practice.

**Convergence diagnostics.** The quality of the DEQN solution is assessed using several complementary diagnostics:

1.  **Euler equation errors:** For each country $j$, compute $\max_{\bm{s} \in \mathcal{S}_\mathrm{test}} |\mathrm{Euler}^j(\bm{s})|$. Errors below $10^{-3}$ indicate that the optimality condition is violated by less than 0.1% of consumption, an acceptable tolerance for most applications.

2.  **Resource constraint residual:** Verify that $|\mathrm{ARC}(\bm{s})| < 10^{-4}$ on the test set.

3.  **Complementarity check (irreversible companion only):** Confirm that $\mathrm{FB}^j \approx 0$ and that the multiplier $\mu^j$ is positive only when investment is at its lower bound.

4.  **Economic diagnostics:** Verify that the ergodic distribution of capital, output, and consumption has sensible properties (e.g., positive trade balances for productive countries, capital flowing to high-productivity states).

5.  **Policy-drift / time-invariance check:** Evaluate the policy on a fixed anchor cloud `X_anchor` after each monitoring interval and report `policy_drift_rms` and `policy_drift_max`. The architecture has no calendar-time input, so any fixed weight vector is a stationary recursive policy by construction; the empirical question is whether SGD has stopped moving the policy function. The run is treated as time-invariant once both drift statistics fall below the prescribed tolerances `TIME_INVARIANCE_TOL_RMS` and `TIME_INVARIANCE_TOL_MAX`.

6.  **Zero-shock stochastic steady state (SSS):** Iterate the learned policy from `ZERO_SHOCK_N_STARTS` dispersed feasible starts with all shocks set to zero. A well-trained policy converges to a common point with $I^j \approx \delta\,k^j$ and (in the irreversible case) $\mu^j \approx 0$; the SSS is a fixed point of the learned stochastic policy that is not imposed during training.

### Validation Protocol {.unnumbered}
To keep the manuscript self-contained, we summarize here the validation diagnostics used for the IRBC model:

1.  **Held-out residual table.** Evaluate mean and max absolute residuals on an out-of-sample test set for each equation block (Euler and ARC always; FB only in the irreversible companion). In the two-country benchmark, typical values are mean $\sim 10^{-4}$ and max $\sim 10^{-3}$ for Euler/ARC, with smaller FB residuals.

2.  **Euler-side comparison.** Compare left and right sides of the Euler equation directly on the test set (scatter around the 45-degree line). Target thresholds are mean relative error below $10^{-3}$ and max relative error below $10^{-2}$.

3.  **Constraint diagnostics (irreversible companion only).** Verify $I^j \ge 0$ everywhere and that $(\mu^j, I^j)$ lies close to the complementarity axes ($\mu^j \approx 0$ when $I^j > 0$).

4.  **Economic sanity checks.** Confirm market-wide accounting identities (e.g., trade balances summing to zero), sensible consumption-sharing behavior, and stable ergodic state distributions around economically plausible regions.

5.  **Policy-drift / time-invariance check.** Track `policy_drift_rms` and `policy_drift_max` on a fixed anchor cloud across training segments; flag the run as time-invariant once both drop below the prescribed tolerances. This check distinguishes "the policy has stabilized" from "the residuals are small"; both are needed for a trustworthy recursive solution.

6.  **Zero-shock stochastic steady state.** Simulate the learned policy with all shocks set to zero from several dispersed feasible initial states. Convergence to a single point with $I^j \approx \delta\,k^j$ (and $\mu^j \approx 0$ in the irreversible case) is a coordinate-free sanity check that complements the held-out residual table.

This protocol makes solution quality auditable and comparable across model sizes and network configurations.

**Policy function properties.** The learned policy functions exhibit the expected economic properties. Consumption sharing follows the Pareto-weight and IES structure in {eq}`eq-irbc_consumption`: holding the common shadow price $\lambda_t$ fixed, a higher Pareto weight raises country $j$'s consumption, and with heterogeneous IES the consumption ratio varies with $\lambda_t$. Productivity affects consumption only indirectly through the equilibrium shadow price and the resource constraint, not through a mechanical bilateral ratio $z^j/z^k$. This is the textbook complete-markets prediction: the cross-country consumption ratio depends on the Pareto weights $\tau^j/\tau^k$ and the IES gap, not on the productivity differential. The empirical failure of this prediction is the consumption-correlation puzzle introduced in {ref}`sec-irbc_motivation`; a closely related but distinct failure is the Backus–Smith puzzle, which concerns the correlation between relative consumption growth and the real exchange rate, predicted to be near one under complete markets but empirically near zero or even negative. Any model that aims to reproduce either puzzle has to break some of the assumptions used here (e.g. by restricting the asset menu, {cite:t}`heathcote2002financial`, or adding non-traded goods, {cite:t}`heathcote2013international`). Investment responds procyclically to productivity shocks: a high realization of $z^j$ raises the marginal product of capital in country $j$, triggering increased investment. When the irreversibility constraint binds ($I^j = 0$), capital cannot be disinvested and the multiplier $\mu^j$ becomes positive; the network learns this regime-switching behavior smoothly through the Fischer–Burmeister loss. Trade balances adjust to channel resources toward productive countries: positive trade balances (net exports of goods) correspond to countries whose current productivity exceeds the average, and the implied capital flows are consistent with standard international macroeconomic theory.

The key advantage of the DEQN approach is its scaling behavior: while traditional Cartesian grid-based methods {cite:p}`judd1998numerical` exhibit exponential growth in computation time as $N$ increases, and even adaptive sparse grid methods {cite:p}`ECTA:ECTA1716`, which significantly mitigate the curse of dimensionality, become computationally demanding for $N > 10$, DEQN runtimes in our implementations are reported close to linear in $N$ over a broad range of model sizes (see {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, Table 2 and surrounding discussion, for timings across $N \in \{2, \ldots, 100\}$). This favorable empirical scaling arises because the network's parameter count grows roughly linearly (more input/output neurons), while each SGD step avoids state-space grids. The companion notebooks ([`lecture_04_01_IRBC_DEQN_smooth.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_01_IRBC_DEQN_smooth.ipynb) and [`lecture_04_02_IRBC_DEQN_irreversible.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_02_IRBC_DEQN_irreversible.ipynb)) only run the $N=2$ case, so the linear-scaling claim cannot be reproduced from the in-class material; readers who wish to verify it directly should consult the published timings or replicate the larger-$N$ runs from the Azinovic et al. codebase.

**Comparison with adaptive sparse grids.** The approach of {cite:t}`ECTA:ECTA1716` handles kinks in the policy function (e.g., those induced by the irreversibility constraint) by *refining* the grid locally around the kink using hierarchical surplus indicators. This keeps the method accurate but the grid remains anchored to a hypercube, so computation still scales poorly once the number of active kinks or the dimensionality grows. DEQNs do not represent kinks by grid refinement; instead, they fit a smooth approximator (Swish/softplus network) to the Fischer–Burmeister-regularized problem, which produces a globally smooth policy that tracks the true piecewise structure without needing localized grid points. The two methods are therefore complementary: adaptive sparse grids give deterministic error bounds on a hypercube; DEQNs give simulation-based error bounds on the ergodic set with no grid at all. From a theoretical perspective, {cite:t}`montanelli2019deep` establish error bounds showing that deep ReLU networks can approximate functions on sparse grids without the exponential growth in parameters that afflicts classical polynomial methods, providing formal underpinning for why deep learning can mitigate (though not eliminate) the high-dimensional approximation cost. Exact runtimes depend on architectural choices, quadrature design, and hardware; the robust finding is that the DEQN formulation avoids explicit tensor-product state grids and remains computationally viable in dimensions where standard methods become prohibitively expensive.

Beyond the IRBC setting, closely related neural-equilibrium methods have been applied to other policy-relevant problems. {cite:t}`nuno2024monetary` use DEQNs to compute optimal *monetary policy rules* under persistent supply shocks, replacing the linearization step around steady state with a globally trained policy network. {cite:t}`bretscherRicardianBusinessCycles2022` apply DEQN to multi-country international real business cycles with comparative advantage. Most recently, {cite:t}`azinovicyangzemlicka2025sequencespace` replace the endogenous cross-sectional state with a *truncated history of exogenous aggregate shocks* (the sequence-space representation), so that the network's input dimension scales with the truncation horizon rather than with the number of agents, which is the heterogeneous-agent extension developed in {ref}`ch-young`.

```{prf:remark} Chapter Summary

- The IRBC model is the standard testbed for high-dimensional global solution methods: $N$ countries each with capital and TFP push the state dimension to $2N$ in the planner formulation used here, and the irreversibility constraint introduces non-smooth kinks via Karush–Kuhn–Tucker complementarity.

- Fischer–Burmeister smoothing converts the non-differentiable irreversibility complementarity $0 \le \mu^j \perp I^j \ge 0$ into a smooth squared residual $\Phi_\varepsilon^2$ that is compatible with SGD.

- Persistent-simulation training, a single continuing ensemble of stochastic trajectories, never reset to steady state, is the recipe used in the companion notebooks; bounded policy parameterizations ($k_{t+1}^j = k_t^j\exp\{\bar g\tanh r_j(\bm{s})\}$ in the smooth notebook, $k_{t+1}^j = (1-\delta)k_t^j + I^j$ with $I^j \ge 0$ in the irreversible notebook) keep the simulation feasible without a separate burn-in phase.

- Time-invariance via policy drift on a fixed anchor cloud, and a zero-shock stochastic-steady-state check from dispersed feasible starts, are the two convergence diagnostics specific to recursive DEQN training; both are run inside the companion notebooks.

- Gauss–Hermite tensor-product quadrature ({ref}`sec-gh_tensor_product`, introduced in the previous chapter) handles expectations in low-dimensional shock spaces; once $N \gtrsim 5$ the linear-scaling Stroud-3 monomial rule ({ref}`sec-monomial_cubature`) is the workhorse for IRBC, and is also what the companion notebooks use for $N=2$, with QMC ({ref}`sec-qmc_cdf`) and sparse grids reserved for high-accuracy or very high-dimensional cases.
```


## Further Reading {.unnumbered}
- {cite:t}`ECTA:ECTA1716`, adaptive sparse grids for IRBC, the classical-method benchmark this chapter contrasts with.

- {cite:t}`pichler2011`, an IRBC-specific application of the monomial rule of {ref}`sec-monomial_cubature`, useful as a sanity check for the multi-country setting.

- {cite:t}`niederreiter1992random`, the standard reference for quasi-Monte Carlo and low-discrepancy sequences for the high-dimensional integrals encountered at large $N$.

- {cite:t}`nuno2024monetary`, a recent DEQN application to optimal monetary policy.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch3-1

**[Core\] Fischer–Burmeister.** (a) Show that $\Phi(a,b) = a + b - \sqrt{a^2+b^2} = 0 \iff a\ge 0,\, b\ge 0,\, ab=0$. (b) Plot the level set $\Phi(a,b)=0$ and a smoothed variant $\Phi_\varepsilon(a,b) = a + b - \sqrt{a^2+b^2+\varepsilon^2}$ for $\varepsilon=10^{-3}$ in the $(a,b)$ plane. (c) Compute the gradient $\nabla\Phi(a,b) = (1 - a/\sqrt{a^2+b^2},\, 1 - b/\sqrt{a^2+b^2})$ and evaluate it at $(a,b) = (1,1)$. Note that the raw gradient $\nabla\Phi$ at $(1,1)$ equals $(1-1/\sqrt 2,\, 1-1/\sqrt 2) > 0$ and therefore points *away* from the complementarity axes (northeast). Now show that gradient descent on the *squared* residual $\Phi^2$ instead points toward the zero set: in the open positive quadrant $\Phi(a,b) > 0$, so $-\nabla\Phi^2 = -2\Phi\,\nabla\Phi$ points southwest, back to the L-shape. (d) Explain in two sentences why replacing $\Phi$ by $-\Phi$ has no effect on the squared loss: $\Phi^2 = (-\Phi)^2$, so the gradient field of $\Phi^2$ is unchanged. In particular, the sign convention $a+b-\sqrt{a^2+b^2}$ versus $\sqrt{a^2+b^2}-a-b$ is irrelevant once the residual is squared, and the operative quantity for SGD is $-\nabla(\Phi^2)$ rather than $-\nabla\Phi$.
```

```{exercise}
:label: ex-ch3-2

**[Core\] State-space scaling.** For an IRBC with $N$ symmetric countries, write down the state, policy, and equation dimensions. At what $N$ does a tensor-product Gauss–Hermite rule with $Q=3$ nodes per dimension exceed $10^4$ evaluations per Euler equation? At what $N$ does the Stroud-3 monomial rule of {ref}`sec-monomial_cubature` stay below $100$ nodes?
```

```{exercise}
:label: ex-ch3-3

**[Core\] Persistent-simulation feasibility.** The companion notebooks parameterize next-period capital as $k_{t+1}^j = k_t^j\exp\{\bar g\,\tanh r_j(\bm{s}_t)\}$ in the smooth model and $k_{t+1}^j = (1-\delta)k_t^j + I^j$ with $I^j\ge 0$ in the irreversible model. (a) Show that under either parameterization the simulated capital stock cannot leave the feasible set $\{k>0\}$ even when $r_j$ is generated by a randomly initialized network. (b) Contrast with the naive head $k_{t+1}^j = \mathrm{softplus}(r_j)$: construct a $\bm{s}_t$ at which a random initialization can drive the simulation arbitrarily far from any economically sensible region in one step. (c) Explain in two sentences why feasibility-by-construction is what allows the notebooks to dispense with a separate uniform-sampling burn-in phase.
```

```{exercise}
:label: ex-ch3-4

**[Core\] Adjustment-cost partials and Tobin's Q.** Starting from the quadratic adjustment cost {eq}`eq-irbc_adjcost`, $\Gamma^j = (\kappa/2)\,k^j (k^{j\prime}/k^j - 1)^2$, derive the partial derivatives $\partial\Gamma^j/\partial k^{j\prime}$ and $\partial\Gamma^j/\partial k^j$ shown in {eq}`eq-irbc_adjcost_derivs`. Show that both partials vanish at the steady state $k^{j\prime} = k^j$, so the deterministic steady-state allocation is identical to the one of the frictionless model. Define the *net investment rate* $g^j = k^{j\prime}/k^j - 1$ and re-express the marginal adjustment cost $\partial\Gamma^j/\partial k^{j\prime}$ as $\kappa\,g^j$; this is the Tobin's-Q-style wedge that the planner must balance against the marginal product of capital in the Euler equation. Discuss qualitatively how raising $\kappa$ changes the speed of convergence to the steady state under a positive productivity shock.
```

```{exercise}
:label: ex-ch3-5

**[Core\] Complete-markets risk sharing.** Use the consumption-sharing condition {eq}`eq-irbc_consumption` to derive the cross-country consumption ratio $c_t^i / c_t^j$ as a function of the Pareto weights $(\tau^i, \tau^j)$, the IES parameters $(\gamma_i, \gamma_j)$, and the shadow price $\lambda_t$. (i) Show that with homogeneous preferences ($\gamma_i = \gamma_j = \gamma$), the ratio $c_t^i/c_t^j$ is *constant in time*, reproducing the perfect-risk-sharing result of {cite:t}`backus1992international`. (ii) Show that with heterogeneous IES, the ratio fluctuates with $\lambda_t$, but consumption growth rates are still scalar multiples of the same aggregate shadow-price growth when all $\gamma_j>0$. What does this imply for cross-country consumption-growth correlations in this planner allocation? (iii) Explain in two sentences why heterogeneous IES alone does not resolve the empirical consumption-correlation puzzle.
```

```{exercise}
:label: ex-ch3-6

**[Core\] Notebook: steady-state comparative statics.** Open [`lecture_05_05_IRBC_Exercise.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_05_IRBC_Exercise.ipynb) (it lives in the Lecture-05 code folder, since it doubles as the entry point to the NAS / loss-normalization material of {ref}`ch-nas`). Working from the closed-form deterministic steady state, in which the Euler condition pins $\mathrm{MPK} = 1/\beta$ and hence $k_\mathrm{ss} = \bigl[(1/\beta - 1 + \delta)/(\zeta A_\mathrm{tfp})\bigr]^{1/(\zeta-1)}$, compute $k_\mathrm{ss}$ and $c_\mathrm{ss}$ for (a) higher depreciation $\delta = 0.05$, (b) a more impatient household $\beta = 0.95$, and (c) a lower capital share $\zeta = 0.30$, each relative to the baseline. Predict the sign of each change before computing, and explain why $k_\mathrm{ss}$ falls in every case. (The notebook deliberately uses a standalone calibration $A_\mathrm{tfp} = 1$; the IRBC training notebook [`lecture_04_01_IRBC_DEQN_smooth.ipynb`](notebooks/lecture_04_irbc_with_deqns/lecture_04_01_IRBC_DEQN_smooth.ipynb) instead pins $A_\mathrm{tfp}$ so that $k_\mathrm{ss} = 1$, a rescaling that does not change any of the qualitative conclusions.)
```

```{exercise}
:label: ex-ch3-7

**[Computational\] Notebook: loss-component weighting.** In the same notebook, take a snapshot of the IRBC training loss with five components (two Euler residuals, the aggregate resource constraint, and two Fischer–Burmeister complementarity residuals) whose magnitudes span several orders ($\ell_\mathrm{ARC} \sim 5$, $\ell_\mathrm{FB} \sim 10^{-4}$). Replace the equal weights $w_i = 1$ with inverse-loss weights $w_i = (1/\ell_i)/\sum_j (1/\ell_j)$, verify that every weighted contribution $w_i \ell_i$ is then equal, and identify which component receives the most weight (and why). Finally, on the supplied pair of (synthetic) equal-weight vs. inverse-weight training curves, plot both on a log axis and quantify the convergence speed-up. In which regime do you expect inverse-loss weighting to help most, and when can it hurt; think of a component that is small because it is genuinely satisfied by construction, e.g., a hard-coded resource constraint?
```
