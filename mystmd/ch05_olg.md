---
title: "Overlapping Generations Models with DEQNs"
label: ch-olg
---

In {ref}`ch-deqn`–{ref}`ch-irbc` all agents were infinitely lived. We now extend the DEQN framework to *overlapping generations* (OLG) models {cite:p}`diamond1965national`, where $A$ finitely-lived cohorts coexist in every period. OLG models introduce lifecycle savings, intergenerational transfers, age-dependent heterogeneity, and inequality constraints on portfolio choices, phenomena that are central to fiscal policy analysis, pension reform, and demographic modeling. We proceed in two stages. We first solve a deliberately small 6-agent OLG that admits a *closed-form solution* {cite:p}`Krueger20041411`, which gives a clean ground truth against which to validate the neural-network solver. We then scale up to the 56-agent research benchmark of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, where the no-short-sale-of-capital constraint $k'^h\ge 0$ binds on a non-trivial slice of the ergodic set; that constraint introduces a kink, the main new computational challenge of the benchmark, and we handle it by combining softplus output activations (for non-negativity) with squared product residuals for the orthogonality conditions in the loss. The model also carries a collateral constraint $k'^h+\kappa\,b'^h\ge 0$ that the current notebook parameterization of $\hat q^h$ keeps slack on the learned ergodic set; we develop both constraints below so that the architecture is in place when a future calibration makes the collateral side bind.

(sec-olg_why)=
## Why Overlapping Generations?
In the Brock–Mirman and IRBC models of {ref}`ch-deqn`–{ref}`ch-irbc`, all agents are infinitely lived. Picture instead a photograph of the economy taken at a single instant: it contains a twenty-something just entering the workforce with no savings, a forty-something at peak earnings putting money aside, and a retiree drawing down a lifetime of accumulated wealth, all making decisions in the same period and all linked through the prices that their collective saving determines. The infinitely-lived-agent assumption collapses this picture and rules out several economically important phenomena:

- **Lifecycle savings.** Agents accumulate wealth when young, draw it down in old age.

- **Intergenerational transfers.** Pensions, social security, and bequests cannot be studied without age structure.

- **Age-dependent heterogeneity.** Labor endowments, risk preferences, and portfolio composition vary systematically over the lifecycle.

An OLG economy consists of $A$ cohorts that coexist in each period: a new cohort of age 1 is born, the oldest cohort of age $A$ dies, and everyone else ages by one period. Crucially, the number of agent types is *finite*, so the cross-sectional distribution has only $A$ entries and the state space remains finite-dimensional, in contrast to the continuum-of-agents models treated in {ref}`ch-young`. The mechanism that ties the three phenomena above together is consumption smoothing over a hump-shaped earnings path ({numref}`fig-olg_lifecycle`): because labor income rises and then falls over the lifecycle while agents prefer a steady consumption stream, they accumulate assets in their high-earning years and run them down afterwards, and the equilibrium interest rate is whatever clears the resulting demand for savings against the economy's capital stock.

```{figure} figures/fig-olg_lifecycle.svg
:name: fig-olg_lifecycle

Stylized lifecycle profiles in an OLG economy (schematic, *not* a solution of the model). Labor income (blue) is hump-shaped, peaking in mid-career, while agents prefer a roughly flat consumption path (green); so they accumulate assets out of income during their high-earning years and run them down near the end of life. The asset profile (red, dashed) is therefore a hump that starts near zero for the newborn cohort, peaks toward the end of working life, and returns to zero for the oldest cohort, which consumes everything. The 6-agent analytic model of {ref}`sec-olg_analytic` is a stripped-down version of this picture (only the youngest cohort earns labor income); the 56-agent benchmark of {ref}`sec-olg_56` reproduces the full hump.
```

We develop the OLG framework in two stages. {ref}`sec-olg_analytic` works through the 6-agent model with a closed-form solution, maps it to a DEQN ({ref}`sec-olg_deqn`), and validates the trained network against the analytical savings rates; {ref}`sec-olg_fb` then explains how binding borrowing and collateral constraints are encoded, and {ref}`sec-olg_56` solves the 56-agent research benchmark with exactly the same training loop.

(sec-olg_analytic)=
## The 6-Agent Analytic OLG Model

```{prf:remark} End-to-end vignette: a 6-agent OLG in five steps

 Before stepping through the formal model, it is worth seeing the full DEQN pipeline in one breath. The same five steps apply to every model in this script.

1.  **State and policy.** Stack the 6 cohort capital holdings, the aggregate $K_t$, prices $(r_t, w_t)$, the TFP shock $\eta_t$, and the depreciation shock $\delta_t$ into a state vector $\x_t$. A single MLP $\mathcal{N}_\theta(\x_t) \to \R^{5}$ outputs the savings of cohorts 1–5 (cohort 6 saves nothing).

2.  **Loss.** Stack 5 Euler-equation residuals into a single mean-square loss; a sigmoid savings-fraction head makes each $k'^h \in [0,\mathrm{inc}^h]$, so $k'^h \ge 0$ and $c^h \ge 0$ both hold exactly, and capital-market clearing is satisfied by construction (aggregate $K_{t+1}$ is read off as the sum of cohort savings). Small additive penalties on negative $c$ and $K$ remain in the loss as numerical backstops but are inactive with this head.

3.  **Training distribution.** At each segment, construct a state cloud: in the persistent notebooks the cloud is generated by rolling parallel trajectories forward under the current policy; in the exogenous companions it is drawn from broad feasible boxes. Mini-batches are sampled from the current cloud.

4.  **Optimization.** Adam with the standard moment coefficients $(0.9,0.999)$ (avoiding the $\beta$ symbol of the household discount factor) and learning rate $\sim 10^{-4}$ in the short presets, decaying to $10^{-5}$ in the production preset of the analytic-OLG notebook. In the 56-agent benchmark of {ref}`sec-olg_56`, the production schedule is $10^{-5}$ for the first $60{,}000$ episodes followed by $10^{-6}$ for the remaining episodes.

5.  **Diagnostics.** At convergence, check (i) the average savings rate by cohort against the closed-form $\beta_h$ in Eq. {eq}`eq-olg_savings_rate` below, (ii) Euler residuals across the simulated ergodic set, (iii) (in the 56-agent benchmark only) the bond-market-clearing residual, and (iv) policy-drift / time-invariance on a fixed anchor cloud: evaluate the policy on `X_anchor` after each monitoring interval and flag the run as *time-invariant* once `policy_drift_rms` and `policy_drift_max` fall below `TIME_INVARIANCE_TOL_RMS` and `TIME_INVARIANCE_TOL_MAX`. In the 6-agent analytic model, capital-market clearing is satisfied by construction, since aggregate next-period capital is taken as the sum of cohort savings.

The notebook [`lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb) ships with a `RUN_MODE` switch: `"smoke"` ($\sim$30 s, sanity check), `"teaching"` ($\sim$5 min, savings close to the closed form and mean Euler residuals $\sim 10^{-3}$ on the simulated cloud), and `"production"` (longer run, Table 3-level Euler accuracy on the full ergodic set); a feedback-free exogenous-sampling companion is available as [`lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb). A fifth notebook, [`lecture_08_11_OLG_Exercise.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_11_OLG_Exercise.ipynb), is a self-contained warm-up (closed-form savings rates, a single-cohort lifecycle simulation, and a discount-factor comparison) on the same analytic model. The 56-agent benchmark of {ref}`sec-olg_56` (notebook [`lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb), with the exogenous-sampling companion [`lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb)) runs in a few hours on a GPU with the *same* code template.
```


{cite:t}`Krueger20041411` proposed a deliberately simple OLG model with a *closed-form solution*, making it an ideal validation benchmark for the DEQN approach. We develop it here as the first of the two OLG instances of this chapter; {ref}`sec-olg_deqn` maps it to a DEQN and validates the trained network against the closed form derived below.

We instantiate the OLG environment with $A=6$ overlapping cohorts, indexed by age $h \in \{1,\ldots,6\}$. Time is discrete and infinite. The model equations below are written for general $A$ and specialized to $A=6$ in the calibration that follows.

**Household problem.** An agent of age $h$ at time $t$ maximizes expected lifetime utility:

$$
\max_{\{c_{t+j}^{h+j},\, k_{t+j+1}^{h+j+1}\}_{j=0}^{A-h}} \;\mathbb{E}_t\!\left[\sum_{j=0}^{A-h} \beta^j\, u(c_{t+j}^{h+j})\right],
$$ (eq-olg_hh)

subject to the period budget constraint

$$
c_t^h + k_{t+1}^{h+1} = r_t \, k_t^h + w_t \, \ell^h \equiv \mathrm{inc}_t^h,
$$ (eq-olg_budget)

where $k_t^h$ denotes capital holdings, $r_t$ is the gross return on capital, $w_t$ is the wage, $\ell^h$ is an age-dependent labor endowment, and $\mathrm{inc}_t^h$ is total income.

**Boundary conditions.**

- Newborns have no initial wealth: $k_t^1 = 0$.

- The oldest cohort consumes everything: $k_{t+1}^{A+1} = 0$.

- Borrowing is not permitted: $k_{t+1}^{h+1} \geq 0$ for all $h$.

**Euler equations.** The first-order conditions yield $A-1$ Euler equations (for ages $h=1,\ldots,A-1$):

$$
u'(c_t^h) = \beta\,\mathbb{E}_t\!\left[r_{t+1}\, u'(c_{t+1}^{h+1})\right].
$$ (eq-olg_euler)

**Firm problem and market clearing.** A representative firm operates a Cobb–Douglas technology with value added $F_t = \eta_t K_t^\alpha L_t^{1-\alpha}$, where $\eta_t$ is a TFP shock and $L_t = \sum_{h=1}^A \ell^h$; the gross resource available to households is $Y_t = F_t + (1-\delta_t)K_t = r_t K_t + w_t L_t$ (it is $Y_t$, not $F_t$, that the notebook passes as an engineered feature). Competitive factor markets imply:

$$
r_t = \alpha\,\eta_t\, K_t^{\alpha-1}L_t^{1-\alpha} + (1-\delta_t), \qquad
w_t = (1-\alpha)\,\eta_t\, K_t^\alpha L_t^{-\alpha},
$$ (eq-olg_prices)

where $\delta_t$ is the depreciation rate (potentially stochastic). Market clearing requires that aggregate capital at $t+1$ is the sum of holdings across cohorts:

$$
\sum_{h=2}^{A} k_{t+1}^{h} = K_{t+1},
$$ (eq-olg_mc)

with $k_{t+1}^{1}=0$ as a newborn boundary condition (cohort 1 enters life with no assets), and where $k_{t+1}^{h}$ for $h=2,\dots,A$ is the savings of cohort $h-1$ at date $t$ (which becomes the date-$(t+1)$ holdings of the cohort once it has aged by one period).

**Calibration.** The model has $A=6$ agents with log utility ($\gamma = 1$), Cobb–Douglas production ($\alpha = 0.3$), and discount factor $\beta = 0.7$. Only agent 1 works ($\ell = (1,0,0,0,0,0)$); this stripped-down labor profile is what gives the closed form below, not a realistic lifecycle assumption, and the 56-agent benchmark of {ref}`sec-olg_56` restores a hump-shaped endowment. Four exogenous shock states combine TFP $\eta \in \{0.95,1.05\}$ and depreciation $\delta \in \{0.5,0.9\}$, with i.i.d. transitions ($\pi_{ss'} = 0.25$).

**Analytical solution.** With log utility and i.i.d. shocks, the optimal savings rate has a closed form. Define the age-dependent savings rate:

$$
\beta_h = \beta \cdot \frac{1 - \beta^{A-h}}{1 - \beta^{A-h+1}}, \qquad h = 1, \ldots, A-1.
$$ (eq-olg_savings_rate)

The optimal policy is then $k'^h = \beta_h \cdot \mathrm{inc}^h$: each agent saves a *fixed fraction* of total income, regardless of the shock. Two features of the calibration drive this clean form. First, under log utility the income and substitution effects of a return shock exactly cancel, so the savings *rate* is invariant to $(r_t, w_t)$. Second, because the shocks are i.i.d. there is nothing about the future to forecast, so the rate does not depend on the current shock either; only the horizon matters. The fraction $\beta_h$ therefore declines with age: cohort $h$ has only $A-h$ remaining periods over which to spread its future income, so the marginal incentive to carry resources forward weakens as $h$ grows. For $A=6$, $\beta=0.7$, {numref}`tab-olg6_savings_rates` reports the resulting savings rates.

````{table}
:name: tab-olg6_savings_rates

Closed-form age-specific savings rates in the 6-agent analytic OLG with log utility and $\beta=0.7$.

| Age $h$ | 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\beta_h$ | 0.660 | 0.639 | 0.605 | 0.543 | 0.412 |
````

Young agents save more (more periods ahead); old agents save less; {numref}`fig-olg6_savings` plots the same numbers across $h$. This vector is the validation target: at convergence, the trained network's average sigmoid output should reproduce $\beta_h$ cohort by cohort.

```{figure} figures/fig-olg6_savings.svg
:name: fig-olg6_savings

Closed-form savings rates $\beta_h$ from {numref}`tab-olg6_savings_rates` for the 6-agent analytic OLG ($\beta=0.7$, log utility). The monotone decline with age reflects the shrinking forward horizon: cohort $h$ has only $A-h$ remaining periods over which to consume future income, so the marginal incentive to save weakens as $h$ grows. This is the validation target the trained DEQN's average sigmoid output should match cohort by cohort.
```

(sec-olg_deqn)=
## Mapping the Analytic OLG to a DEQN
The mapping follows the same "states $\to$ network $\to$ loss" structure as Brock–Mirman ({ref}`ch-deqn`). We now write each block explicitly for the 6-agent analytic model just set up; this is exactly what slides II.7–II.9 of `lectures/lecture_08_olg_models_deqns/slides/lecture_08_olg_models_deqns.tex` render in pictures. The 56-agent benchmark of {ref}`sec-olg_56` extends the same template with two extra policy blocks (multipliers, bond price) and an additional market-clearing residual; we write that version out there.

**State $\x_t$ entering the network.** What does the network actually need to know? The *informational* state of the analytic model is just the pair

$$
\bigl(z_t,\,\bm{k}_t\bigr) \;\in\; \{1,\ldots,4\}\times\R^A
\qquad\text{where}\qquad \bm{k}_t = (k_t^1,\ldots,k_t^A),
$$ (eq-olg_state_min)

the current shock index plus the cross-sectional capital distribution. This is the minimal vector that pins down the equilibrium, and it is what slide II.8 displays in the FREE signature. Everything else, the aggregate capital $K_t=\sum_h k_t^h$, the prices $(r_t,w_t)$, output $Y_t$, each cohort's income, the row of next-period transition probabilities, is a deterministic function of $(z_t,\bm{k}_t)$. The network could in principle re-derive all of it from the raw pair, but there is no reason to make it: we hand the network those quantities pre-computed, which is a pure change of input coordinates that leaves the equilibrium map untouched and frees the network's capacity for the one genuinely hard thing it has to learn, the savings policy. Concretely the notebook feeds an *extended state* of dimension $16+4A$,

$$
\x_t \;=\; \bigl(\,\underbrace{z_t,\,\mathbf{1}\{z_t\},\,\eta_t,\delta_t,K_t,L_t,r_t,w_t,Y_t}_{12\text{ aggregate}},\;
\underbrace{k_t^{1:A},\,\mathrm{fw}_t^{1:A},\,\mathrm{linc}_t^{1:A},\,\mathrm{inc}_t^{1:A}}_{4A\text{ per-agent}},\;
\underbrace{\pi(z_t,\cdot)}_{4\text{ transition probs}}\bigr) \;\in\; \R^{16+4A},
$$ (eq-olg_state)

with $\mathbf{1}\{z_t\}$ the 4-state one-hot of the current shock, $K_t=\sum_h k_t^h$, $L_t=\sum_h \ell^h$, $(r_t,w_t)$ from {eq}`eq-olg_prices`, $Y_t$ the gross resource $\eta_t K_t^\alpha L_t^{1-\alpha} + (1-\delta_t)K_t$, and the per-agent blocks $\mathrm{fw}_t^h = r_t k_t^h$ (capital income), $\mathrm{linc}_t^h = w_t\,\ell^h$ (labor income), $\mathrm{inc}_t^h = \mathrm{fw}_t^h + \mathrm{linc}_t^h$ (total income). Since the map $(z_t,\bm{k}_t)\mapsto\x_t$ is deterministic, {eq}`eq-olg_state_min` and {eq}`eq-olg_state` carry exactly the same information. For $A=6$ this is $16+4\cdot 6 = 40$ inputs (the notebook constant `FEATURE_DIM`).

**Policies approximated by the network.** A single multilayer perceptron with a *sigmoid savings-fraction* output head approximates the equilibrium policy as a function of the state. (Throughout this OLG chapter we use $\theta$ for the network parameters rather than the $\rho$ of {ref}`ch-deqn`–{ref}`ch-irbc`; both refer to the same object, and the switch follows the convention of the public OLG reference implementation.)

$$
\boxed{\;\mathcal{N}_\theta:\;\R^{16+4A} \;\longrightarrow\; \R^{A-1},\qquad
\mathcal{N}_\theta(\x_t) \;=\; \bigl(\hat\beta^1(\x_t),\ldots,\hat\beta^{A-1}(\x_t)\bigr),\qquad
\hat a^h(\x_t) := \hat\beta^h(\x_t)\,\mathrm{inc}_t^h\;}
$$ (eq-olg_policy)

where the network output $\hat\beta^h(\x_t)\in(0,1)$ is cohort $h$'s *savings rate* and $\hat a^h(\x_t)$ its savings level (slide II.9, output column). This parameterization mirrors the closed-form solution's structure (each cohort saves a fixed fraction of income, Eq. {eq}`eq-olg_savings_rate`). Cohort $A$ saves nothing by terminal boundary, so the network has $A-1$ outputs rather than $A$. Three by-construction guarantees follow:

- Non-negativity of savings $\hat a^h \ge 0$ holds at every iteration, so the borrowing constraint {eq}`eq-olg_kkt` is satisfied without an explicit Lagrange multiplier (in this calibration $\lambda^h \equiv 0$ on the ergodic set; see {ref}`sec-olg_fb` for the multiplier-based variant used in the 56-agent benchmark).

- Non-negativity of consumption $\hat c^h = (1-\hat\beta^h)\,\mathrm{inc}_t^h \ge 0$ also holds by construction, so the soft penalty on $\max(-\hat c^h,0)$ in the loss (next paragraph) is a dead backstop.

- Capital-market clearing $K_{t+1} = \sum_{h=2}^{A} k_{t+1}^h$ also holds by construction, since aggregate next-period capital is read off as the sum of the network's savings outputs together with the newborn boundary $k_{t+1}^1 = 0$.

**Equilibrium residual.** Each cohort $h\in\{1,\ldots,A-1\}$ contributes one *relative* Euler-equation residual, built from three quantities. First, the implied current consumption, read off from the budget {eq}`eq-olg_budget` as $\hat c^h(\x_t) := \mathrm{inc}_t^h - \hat a^h(\x_t)$. Second, the next-state map $\Phi$, which combines the current policy with a fresh shock $z_{t+1}$ to produce next period's extended state $\hat\x_{t,+}=\Phi(\x_t,z_{t+1};\theta)$ (the construction of $\Phi$ is spelled out in the next paragraph). Third, the implied next-period consumption $\hat c^{h+1}(\hat\x_{t,+})$ of the cohort that has just aged from $h$ to $h+1$. The relative Euler-equation residual is then

$$
e_{\mathrm{REE}}^h(\x_t)
\;:=\;
\frac{(u')^{-1}\!\Bigl(\beta\,\E{\,r(\hat\x_{t,+})\,u'\!\bigl(\hat c^{h+1}(\hat\x_{t,+})\bigr)\,}\Bigr)}{\hat c^h(\x_t)} \;-\; 1,
\qquad h=1,\ldots,A-1,
$$ (eq-olg_ree)

with $u(c)=\ln c$ in the analytic model so $(u')^{-1}(y) = 1/y$. Equation {eq}`eq-olg_ree` is the unit-free residual of the standard Euler equation {eq}`eq-olg_euler`: a value of $10^{-3}$ means cohort $h$'s implied consumption is mispriced by $0.1\%$ relative to the conditional certainty equivalent. This is the residual displayed in slide II.7.

**Sampling the conditional expectation.** The expectation in {eq}`eq-olg_ree` is over the next-period shock $z_{t+1}$. Because the analytic-model shock has only four states with i.i.d. transition $\pi_{ss'} = 1/4$, the expectation is computed *exactly* (no Monte Carlo) by summing over the four next-period shocks: $\E{\,r(\hat\x_{t,+})\,u'(\hat c^{h+1})\,} = \tfrac14\,\sum_{s'=1}^{4} r(\Phi(\x_t,s';\theta))\,u'\!\bigl(\hat c^{h+1}(\Phi(\x_t,s';\theta))\bigr).$ For each candidate $s'$ the next-state map $\Phi$ ages the cross-section by one period, sets the newborn to $k^1=0$, evaluates the firm prices {eq}`eq-olg_prices` at $K_{t+1}$, and produces the next-period extended state $\hat\x_{t,+}$ on which the network is evaluated again to obtain $\hat a^h(\hat\x_{t,+})$ and hence $\hat c^{h+1}(\hat\x_{t,+})$. When the shock has more states or is continuous, the same construction is replaced by a sample of $z_{t+1}\sim\pi(\cdot|z_t)$ inside the mini-batch (see {ref}`sec-olg_56`).

**The DEQN loss for the analytic OLG.** Given a mini-batch $D_{\mathrm{train}}\subset\{\x_j\}_{j=1}^{N}$ sampled from the ergodic set of the current policy, the loss is the mean-squared relative Euler residual averaged across cohorts and states:

$$
\boxed{\;\mathcal{L}_{D_{\mathrm{train}}}(\theta)
\;=\;
\frac{1}{|D_{\mathrm{train}}|}\;\frac{1}{A-1}\,
\sum_{\x_j\in D_{\mathrm{train}}}\;\sum_{h=1}^{A-1}
\bigl(e_{\mathrm{REE}}^h(\x_j)\bigr)^2\;}
$$ (eq-olg_loss)

*(matching slide II.7).* Two small barrier-style additive penalties on rescaled negative-consumption and negative-aggregate-capital hinges are summed in alongside {eq}`eq-olg_loss` to keep training numerically robust away from convergence; in the notebook they carry the weight `PENALTY_WEIGHT` $=10$ and act on terms such as $\max(-\hat c^h,0)/(1+|\hat c^h|)$ rather than the raw squared hinge. With the sigmoid savings-fraction head described above these hinges are in fact identically zero (savings stay in $[0,\mathrm{inc}^h]$, so $\hat c^h\ge 0$ and $K_{t+1}\ge 0$ always), so the penalties are pure backstops and do not bias the solution.[^1]

**DEQN architecture and training.** The network takes a 40-dimensional input (the extended state {eq}`eq-olg_state`, $16 + 4 \times 6$) and outputs 5 savings *rates* $\hat\beta^h$ via a $40 \to 100 \to 50 \to 5$ architecture with ReLU hidden layers and a sigmoid savings-fraction output ($\approx 9{,}400$ parameters). Training uses the episode-based procedure from {ref}`ch-deqn`: the current network generates a capital path (episode), equilibrium residuals are computed and used for SGD updates, and a new episode is simulated periodically. The companion notebook exposes a `RUN_MODE` switch with three calibrated budgets: `"smoke"` ($\sim$25 training segments, $\sim$30 s on CPU; a code-path sanity check, well short of convergence), `"teaching"` ($\sim$500 segments, $\sim$5 min on CPU; savings rates match the closed form to a few parts in $10^{4}$ and mean relative Euler errors are already $\sim 10^{-3}$ on the simulated cloud, though larger off-trajectory), and `"production"` ($\sim$10,000 segments with longer trajectories, several hours on CPU; mean Euler errors $\sim 10^{-3}$ or below, matching Table 3 of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`). Adam is used throughout (learning rate $\sim 3\times 10^{-4}$ in the short presets, $10^{-5}$ in the production preset); the analogous decay to $10^{-6}$ used by the 56-agent benchmark ({ref}`sec-olg_56`) is not needed at the analytic model's scale.

```{prf:remark} Validation principle

 The 6-agent analytic OLG provides a known ground truth against which the DEQN can be validated exactly. This validation step gives confidence that the same framework will produce reliable results for models without closed-form solutions.
```


(sec-olg_fb)=
## Inequality Constraints and KKT Complementarity
The 6-agent calibration above is deliberately frictionless: the no-short-sale-of-capital constraint never binds on its ergodic set, so we could solve it with a plain sigmoid-savings head and no multipliers. Realistic OLG economies are not so kind. The 56-agent benchmark of the next section carries a no-short-sale-of-capital constraint that binds on a non-trivial slice of states and a collateral constraint that the current notebook parameterization keeps slack on the learned ergodic set; binding inequality constraints in general bring in Karush–Kuhn–Tucker (KKT) complementarity, with its characteristic non-smooth orthogonality condition. This section sets out how the DEQN framework encodes that complementarity; the next section puts it to work.

The no-short-sale-of-capital constraint $k'^h \geq 0$ introduces a complementarity condition via the Karush–Kuhn–Tucker (KKT) system:

$$
k'^h \geq 0, \qquad \lambda^h \geq 0, \qquad k'^h \cdot \lambda^h = 0,
$$ (eq-olg_kkt)

where $\lambda^h$ is the KKT multiplier on the constraint. In a generic non-linear program, the orthogonality condition $k'^h \cdot \lambda^h = 0$ is non-smooth at the origin and cannot be differentiated through naively.

The DEQN setup of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` sidesteps the kink by splitting enforcement across the architecture and the loss:

- **Hard side (architecture).** The savings $k'^h$ and the multiplier $\lambda^h$ are both produced by the network through a softplus activation, so the inequalities $k'^h \geq 0$ and $\lambda^h \geq 0$ hold by construction at every iteration.

- **Soft side (loss).** With non-negativity already guaranteed, the orthogonality $k'^h \cdot \lambda^h = 0$ is enforced by adding the squared product residual $(k'^h \lambda^h)^2$ to the loss.

This product form is what the public reference implementation accompanying {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` uses, and what we adopt in the *56-agent benchmark* of {ref}`sec-olg_56` (Notebook [`lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb)). As noted above, in the 6-agent analytic calibration of {ref}`sec-olg_analytic` the no-short-sale-of-capital constraint is non-binding everywhere on the ergodic set, so $\lambda^h \equiv 0$ and the multipliers (and the KKT residual) drop out of both the network output and the loss; that is why the mapping there was the simpler $\mathcal{N}_\theta: \mathbb{R}^{16+4A} \to \mathbb{R}^{A-1}$ of {ref}`sec-olg_deqn` above, with no multiplier outputs. The smoother *Fischer–Burmeister* (FB) reformulation, $\Phi(a,b) = a + b - \sqrt{a^2 + b^2}$, is an alternative used in the IRBC notebook of {ref}`ch-irbc` for the investment-irreversibility constraint.

**When to choose product form vs. Fischer–Burmeister.** The product form $(k'^h \lambda^h)^2$ is simpler, gradient-cheaper, and sufficient whenever the constraint is rarely active on the ergodic set, since the optimizer just needs to verify slackness in expectation. The Fischer–Burmeister residual $\Phi(a,b)^2$ keeps gradient information *on both sides* of the active set: when the constraint is frequently binding (e.g. the IRBC irreversibility constraint on a non-trivial fraction of states), product-form gradients vanish whenever the constraint is locally inactive, which can stall training; FB does not have this pathology. As a rule of thumb: product form for occasionally-binding KKT, FB for frequently-binding KKT. In the OLG benchmark of {ref}`sec-olg_56` the no-short-sale-of-capital constraint binds on a thin slice of the ergodic set, so the product form was sufficient; the IRBC application of the previous chapter binds more often and benefits from FB.

**The two OLG models we solve, side by side.** We have now built and solved the first of the two OLG instances that anchor the rest of the manuscript: the 6-agent analytic model used to validate the DEQN against a closed form ({ref}`sec-olg_analytic`–{ref}`sec-olg_deqn`). The second is the 56-agent benchmark of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, developed in the next section. {numref}`tab-olg_6_vs_56` summarizes the structural and computational gap between them before we turn to it.

````{table}
:name: tab-olg_6_vs_56

The two OLG models solved in this chapter, side by side. The economic richness of the 56-agent benchmark adds two assets, an effectively binding no-short-sale-of-capital constraint (the collateral constraint is kept slack by the $\hat q^h$ parameterization), persistent shocks, lifecycle labor, and adjustment costs, raising the network input dimension from 40 to 240 and the output dimension from 5 to 221. The DEQN training loop is structurally identical in both cases. Each variant additionally ships with a feedback-free exogenous-sampling companion notebook ([`lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb), [`lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb)) that exercises the same model under a non-co-evolving training cloud.

|  | **6-agent analytic ({ref}`sec-olg_analytic`)** | **56-agent benchmark ({ref}`sec-olg_56`)** |
|---|---|---|
| Cohorts $A$ | 6 (childhood-style) | 56 (ages 25–80, one period = one year) |
| Utility | Log ($\gamma=1$) | CRRA ($\gamma=2$) |
| Shocks | i.i.d. TFP & depreciation, 4 states | Persistent Markov on $(\eta,\delta)$ |
| Labor profile | Only youngest cohort works | Hump-shaped lifecycle endowment $\ell^h$ |
| Assets | Capital only | Capital $+$ bonds |
| Constraints | None binding in calibration | No-short-sale of capital $k'^h\!\ge\!0$ binds; collateral $k'^h\!+\!\kappa b'^h\!\ge\!0$ kept slack by the $\hat q^h$ parameterization |
| Adjustment cost | None | Quadratic $\tfrac{\zeta}{2}(k'^h\!-\!rk^h)^2$ |
| Network input dim | 40 (extended; minimal 7) | 240 (extended; minimal 113) |
| Output dim | 5 (savings rates of cohorts 1–5) | 221 ($4(A-1)+1$: policies, multipliers, price) |
| Loss terms | 5 Euler $+$ market clearing by construction | 221: $4(A-1)$ Euler/KKT $+$ 1 bond clearing |
| Network | Input(40) $\to$ 100 $\to$ 50 $\to$ 5 | Input(240) $\to$ 128 $\to$ 128 $\to$ 221 (teaching) / Input(240) $\to$ 1000 $\to$ 1000 $\to$ 221 (production) |
| Validation target | Closed-form $\beta_h$ of {cite:t}`Krueger20041411` | Mean Euler residual on simulated trajectory |
| Notebook | [`lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb) | [`lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb) |
````

(sec-olg_56)=
## The 56-Agent Benchmark

{numref}`tab-olg_6_vs_56` above previewed the gap; we now develop the second model in full. The benchmark of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022` scales the OLG framework to $A = 56$ agents (ages 25–80) with several realistic features:

- **CRRA utility** with $\gamma = 2$ (replacing log utility).

- **Two assets:** capital $k$ and one-period bonds $b$, with bond price $p$ determined in equilibrium.

- **Hump-shaped labor endowment** $e^h$ peaking in the early 50s.

- **No-short-sale of capital:** $k'^h \geq 0$ (the constraint historically labeled the "borrowing constraint" in this literature; we keep the more precise name to free "borrowing" for the bond side).

- **Collateral constraint:** $k'^h + \kappa\, b'^h \geq 0$, where $\kappa = 1/(1-\delta_{\max})$.

- **Capital adjustment costs:** $\Psi^h = \frac{\zeta}{2}(k'^h - r\cdot k^h)^2$.

- **Persistent shocks:** a 4-state Markov chain for TFP $\times$ depreciation (contrast with i.i.d. in the analytic model).

**Lifecycle labor endowments.** The labor endowment profile $e^h$ follows {cite:t}`BKS1`. In the implementation used here, $e^h$ is a quadratic in age that rises from $0.60$ at age $25$, peaks at $\approx 1.36$ around age $53$, then decays linearly between ages $\sim 62$ and $\sim 70$ to a flat post-retirement floor of $\approx 0.64$. {numref}`tab-olg56_labor_profile` lists the values produced by the notebook formula at a few representative ages.

````{table}
:name: tab-olg56_labor_profile

Representative points on the lifecycle labor-endowment profile in the 56-agent benchmark.

| Age | 25 | 30 | 40 | 48 | 53 | 65 | 80 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $e^h$ | 0.60 | 0.85 | 1.20 | 1.34 | 1.36 | 1.04 | 0.64 |
````

This hump-shaped profile ensures realistic savings heterogeneity: young agents with low labor income and no initial wealth are borrowing-constrained, mid-career agents with high earnings accumulate both capital and bonds, and older agents gradually decumulate toward the end of life.

**Persistent aggregate shocks.** The 4-state Markov chain combines TFP $\eta$ and depreciation $\delta$ into the pairs $(\eta,\delta) \in \{(0.978, 0.08),\, (1.022, 0.08),\, (0.978, 0.11),\, (1.022, 0.11)\}$. The transition matrix is persistent (diagonal entries $\sim$0.63–0.88), in contrast to the i.i.d. shocks in the analytic model. This persistence creates richer dynamics in capital accumulation: a sequence of bad TFP draws can push young agents deep into their borrowing constraint, producing endogenous amplification that a single-period shock would not generate.

**Budget constraint.** Each agent of age $h$ faces:

$$
c^h + k'^h + p\cdot b'^h + \Psi^h = r\cdot k^h + b^h + w\cdot e^h.
$$

The collateral constraint $k'^h + \kappa\,b'^h \geq 0$ acts as a margin requirement: it limits bond borrowing ($b'^h < 0$) relative to capital holdings. Since $\kappa = 1/(1-\delta_{\max})$, the constraint tightens when depreciation is high, precisely when agents are most likely to seek insurance through borrowing.

**State $\x_t$ entering the network.** The informational state of the benchmark is the triple $(z_t,\,\bm{k}_t,\,\bm{b}_t) \in \{1,\ldots,4\}\times\R^A\times\R^A$, where $\bm{k}_t = (k_t^1,\ldots,k_t^A)$ and $\bm{b}_t = (b_t^1,\ldots,b_t^A)$ are the cross-sectional capital and bond distributions, so the minimal state has dimension $1+2A = 113$. As in the analytic case, the notebook feeds the network an extended state of the same $16+4A$ form – twelve aggregate scalars (shock index and its one-hot, $\eta_t$, $\delta_t$, $K_t$, $L_t$, $r_t$, $w_t$, and the gross resource $Y_t = \eta_t K_t^\alpha L_t^{1-\alpha} + (1-\delta_t)K_t$), four per-agent blocks ($k_t^h$, financial income $r_t k_t^h + b_t^h$, labor income $w_t e^h$, and cash $r_t k_t^h + b_t^h + w_t e^h$ – the bond holdings $b_t^h$ are recoverable from financial income and are *not* passed as a separate block, and the bond price $\hat p_t$ is an output, not an input), and the row of next-period transition probabilities $\pi(z_t,\cdot)$ (used by the conditional-expectation block of the loss); concretely $240 = 12 + 4\times 56 + 4$ (the notebook constant `FEATURE_DIM`). This is the analogue of slide III.8.

**Policies approximated by the network.** A single network $\mathcal{N}_\theta$ with softplus output produces a $4(A-1)+1$-dimensional vector that is sliced into five economic blocks (slide III.9):

$$
\boxed{\;\mathcal{N}_\theta:\;\R^{240} \;\longrightarrow\; \R^{4(A-1)+1},\qquad
\mathcal{N}_\theta(\x_t) \;=\; \bigl(\hat k'^{1:A-1},\;\hat \lambda_b^{1:A-1},\;\hat q^{1:A-1},\;\hat \mu^{1:A-1},\;\hat p\bigr)(\x_t)\;}
$$ (eq-olg56_policy)

where $\hat k'^h$ are capital savings, $\hat\lambda_b^h$ the no-short-sale-of-capital multipliers, $\hat q^h \equiv \hat k'^h + \kappa\,\hat b'^h$ the collateral requirement (from which bond holdings are recovered as $\hat b'^h = (\hat q^h - \hat k'^h)/\kappa$), $\hat\mu^h$ the collateral-constraint multipliers, and $\hat p$ the equilibrium bond price. Each raw output is mapped to an admissible value: softplus for the multipliers, and a bounded-exponential map around a baseline for the positive levels. Concretely, writing $z^k_h$, $z^q_h$, and $z^p$ for the raw network outputs, the heads are

$$
\hat k'^h \;=\; k^h_{\mathrm{baseline}}\,\exp(\tanh z^k_h), \quad
\hat q^h \;=\; q^h_{\mathrm{baseline}}\,\exp(\tanh z^q_h), \quad
\hat p \;=\; p_{\mathrm{baseline}}\,\exp(\tanh z^p),
$$ (eq-olg56_heads)

so the four non-negativity inequalities $\hat k'^h\ge 0$, $\hat\lambda_b^h\ge 0$, $\hat q^h\ge 0$, $\hat\mu^h\ge 0$ hold *by construction*, leaving the orthogonality conditions of the KKT systems to be enforced softly in the loss (next paragraph).[^2] The production network uses $1000\times 1000$ hidden units ($\sim\!1.5$M parameters); the teaching version uses $128\times 128$.

**Equilibrium residuals.** Each cohort $h\in\{1,\ldots,A-1\}$ contributes *four* residuals, one per equilibrium condition (slide III.6). To keep the displayed form compact, introduce *numerator/denominator shorthands* for the two Euler conditions:

```{math}
:enumerated: false

\begin{aligned}
\mathcal{N}^h_k(\x_t) &:= \beta\,\E{r_{t+1}\,\mathcal{D}^{h+1}_k(\hat\x_{t,+})\,u'(\hat c^{h+1}_{t+1})} + \hat\lambda_b^h + \hat\mu^h, \quad &
\mathcal{D}^h_k(\x_t) &:= 1 + \zeta\bigl(\hat k'^h - r_t k_t^h\bigr), \\[2pt]
\mathcal{N}^h_b(\x_t) &:= \beta\,\E{u'(\hat c^{h+1}_{t+1})} + \kappa\,\hat\mu^h, &
\mathcal{D}^h_b(\x_t) &:= \hat p.
\end{aligned}
```

Here $\mathcal{D}^h_k$ is the marginal-adjustment-cost wedge from $\Psi^h = \tfrac{\zeta}{2}(k'^h - r_t k^h)^2$: the capital Euler equation in envelope form reads $u'(c_t^h)\,\mathcal{D}^h_k = \beta\,\E{r_{t+1}\,\mathcal{D}^{h+1}_k\,u'(c_{t+1}^{h+1})} + \lambda_b^h + \mu^h$, so the same wedge appears next period on the marginal return to capital (this is the factor `adj_factor_next` in the notebook). With $\zeta = 0$ it collapses to the textbook Euler equation. The bond Euler reduces to the textbook stochastic-discount-factor form $\hat p = \beta\,\mathbb{E}[u'(c')]/u'(c)$ *only when the collateral constraint is slack* ($\hat\mu^h = 0$); whenever $\hat\mu^h > 0$, the bond price carries an additional shadow-value term $\kappa\hat\mu^h/u'(\hat c^h)$ that captures the value of relaxing the collateral constraint. The four per-cohort residuals are then

$$
\begin{aligned}
e_{\mathrm{REE},k}^h(\x_t) &:= \frac{(u')^{-1}\bigl(\mathcal{N}^h_k(\x_t)\,/\,\mathcal{D}^h_k(\x_t)\bigr)}{\hat c^h(\x_t)} - 1, & \text{(Euler, } k \text{)}\\[2pt]
e_{\mathrm{REE},b}^h(\x_t) &:= \frac{(u')^{-1}\bigl(\mathcal{N}^h_b(\x_t)\,/\,\mathcal{D}^h_b(\x_t)\bigr)}{\hat c^h(\x_t)} - 1, & \text{(Euler, } b \text{)}\\[2pt]
e_{\mathrm{KKT},b}^h(\x_t) &:= \hat\lambda_b^h \cdot \hat k'^h, & \text{(borrowing complementarity)}\\
e_{\mathrm{KKT},c}^h(\x_t) &:= \hat\mu^h \cdot \bigl(\hat k'^h + \kappa\,\hat b'^h\bigr) = \hat\mu^h \cdot \hat q^h. & \text{(collateral complementarity)}
\end{aligned}
$$ (eq-olg56_residuals)

On top of these per-agent residuals the bond market must clear: bonds are in zero net supply, so the residual is the cross-sectional sum of bond holdings against the target $\bar B = 0$,

$$
e_{\mathrm{MC},b}(\x_t) \;:=\; \sum_{h=1}^{A} \hat b'^h(\x_t) \;-\; \bar B \;=\; \frac{1}{\kappa}\sum_{h=1}^{A-1}\bigl(\hat q^h - \hat k'^h\bigr), \qquad \bar B = 0.
$$ (eq-olg56_mc)

Capital-market clearing $K_{t+1} = \sum_{h=2}^{A} k_{t+1}^h$ is once again satisfied by construction and does *not* appear as a residual. The conditional expectation in the two Euler equations is computed exactly as in {eq}`eq-olg_ree`: by summing over the four next-period shocks weighted by the persistent-Markov transition probabilities $\pi(z_t,\cdot)$.

**The DEQN loss for the 56-agent benchmark.** Stack the four per-cohort residuals into one squared-cohort term $R^h(\x)^2 := (e_{\mathrm{REE},k}^h)^2 + (e_{\mathrm{REE},b}^h)^2 + (e_{\mathrm{KKT},b}^h)^2 + (e_{\mathrm{KKT},c}^h)^2,$ then add the bond-market-clearing residual. The mini-batch loss is

$$
\boxed{\;\mathcal{L}_{D_{\mathrm{train}}}(\theta)
\;=\;
\frac{1}{|D_{\mathrm{train}}|}\;\frac{1}{4(A-1)+1}\,
\sum_{\x_j\in D_{\mathrm{train}}}\!\Biggl[\;\sum_{h=1}^{A-1} R^h(\x_j)^2 \;+\; \bigl(e_{\mathrm{MC},b}(\x_j)\bigr)^2\;\Biggr]\;}
$$ (eq-olg56_loss)

*(matching slide III.6).* With $A=56$ this is $4\times 55 + 1 = 221$ squared residuals per training state. Each residual enters with weight one: no adaptive loss balancing (cf. {ref}`ch-nas`) is applied because the relative-Euler convention {eq}`eq-olg_ree` already homogenizes the per-cohort Euler scales, and the product-form KKT residuals are unit-free under the softplus head; ReLoBRaLo or GradNorm would be the natural next step if a future calibration broke this homogeneity. Comparison with {eq}`eq-olg_loss`: the analytic case is the special instance of {eq}`eq-olg56_loss` in which the no-short-sale-of-capital constraint never binds (so $\lambda_b^h\equiv 0$), there are no bonds (so all $b$- and collateral-related blocks drop out), and $4(A-1)+1$ collapses to $A-1$. The two losses are the same template instantiated at different complexity. {numref}`tab-olg56_residual_count` unpacks the residual blocks.

````{table}
:name: tab-olg56_residual_count

Residual blocks entering the 56-agent benchmark loss for one training state.

| **Component** | **Symbol** | **Count** |
|---|---|:---:|
| Euler (capital) | $e_{\mathrm{REE},k}^h$ | 55 |
| Euler (bonds) | $e_{\mathrm{REE},b}^h$ | 55 |
| KKT (borrowing) | $e_{\mathrm{KKT},b}^h = \hat\lambda_b^h\,\hat k'^h$ | 55 |
| KKT (collateral) | $e_{\mathrm{KKT},c}^h = \hat\mu^h\,\hat q^h$ | 55 |
| Market clearing (bonds) | $e_{\mathrm{MC},b} = \sum_h \hat b'^h$ | 1 |
| **Total residuals** |  | **221** |
````

**Training and results.** Production training uses 60,000 episodes at lr $= 10^{-5}$ followed by 140,000 episodes at lr $= 10^{-6}$, with runtime of several hours on GPU. The teaching version ($\sim$200 segments, 128-128 hidden units) runs in a few minutes on CPU and is meant to show the mechanics and qualitative lifecycle patterns, not final accuracy. The loss trajectory typically exhibits oscillations, caused by re-simulation of the capital path at each episode, but the overall trend is steadily downward.

**Lifecycle diagnostics.** The trained model produces economically plausible lifecycle patterns. Capital savings $k'^h$ follow a hump shape that mirrors the labor income profile: young agents save little (borrowing constraint binds), mid-career agents accumulate rapidly, and older agents decumulate. Bond holdings $b'^h$ are initially negative (young agents borrow against future income) and increase with age as agents shift from illiquid capital to liquid bonds. Bond prices vary across shock states, with higher prices in high-TFP states reflecting stronger demand for savings. In the teaching run the Euler residuals are still large enough to treat the output as diagnostic; in production runs the mean Euler equation errors are of order $10^{-4}$–$10^{-3}$ for both capital and bond equations (matching Table 3 of {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`), corresponding to a $\sim$0.01%–0.1% deviation in consumption. Market clearing residuals are comparably small. Convergence is also confirmed by the policy-drift check on the fixed anchor cloud: the run is treated as time-invariant once `policy_drift_rms` and `policy_drift_max` fall below their prescribed tolerances.

```{prf:remark} Scalability of the DEQN framework

 The *same algorithm* that solved the 6-agent model scales to 56 agents with two assets and inequality constraints. Only the network size and training duration change; the DEQN framework itself is unchanged. The 56-agent benchmark has a 113-dimensional minimal state and a 240-dimensional engineered network input, both infeasible for traditional grid-based methods.
```


```{prf:remark} Chapter Summary

 OLG models discretize the cross-section into a finite number of cohorts $A$, so the minimal state vector contains the aggregate shock together with the cohort asset holdings and has dimension $\mathcal{O}(A)$. Market clearing $\sum_h k_{t+1}^h = K_{t+1}$ closes the model and is the natural place to encode aggregation exactly via a market-clearing output layer ({ref}`ch-deqn`, footnote on Azinovic–Yang & Žemlička 2024). No-short-sale-of-capital and collateral constraints introduce KKT complementarity, which we enforce by combining softplus output activations (for non-negativity) with squared product residuals $(a\cdot b)^2$ in the loss. Across both instances, the 6-agent analytic OLG and the 56-agent IER benchmark, the *same* training loop covers the spectrum from textbook closed-form validation to research-scale scalability.
```


## Further Reading {.unnumbered}
- {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`, the IER paper that established the 56-agent benchmark.

- {cite:t}`auerbach1987dynamic`, the classical OLG reference.

- {cite:t}`azinoviczemlicka_2024`, market-clearing output layer in OLG with rare disasters.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch5-1

**[Core\] OLG market clearing for $A=3$.** Write the budget constraint and Euler equation for each of three cohorts (young, middle, old) and the market-clearing condition that closes the model. Verify that the count of equations matches the count of unknown policies.
```

```{exercise}
:label: ex-ch5-2

**[Core\] KKT residuals: product form vs. Fischer–Burmeister.** For a borrowing-constrained OLG agent, write the KKT conditions and rewrite them as *both* (i) a product-form residual $(k'^h\cdot\lambda^h)^2$ and (ii) a Fischer–Burmeister residual $\Phi(\lambda^h, k'^h)^2$ with $\Phi(a,b)=a+b-\sqrt{a^2+b^2}$. In each case, show that the loss vanishes *exactly* at any KKT point, not just approximately. Using the rule of thumb in {ref}`sec-olg_fb`, argue which formulation you would choose for the *6-agent analytic OLG* ({ref}`sec-olg_analytic`, where the borrowing constraint is non-binding on the ergodic set) versus the *investment-irreversibility constraint* of {ref}`ch-irbc` (which binds on a non-trivial fraction of states).
```

```{exercise}
:label: ex-ch5-3

**[Computational\] Hump-shaped lifecycle.** Argue why a hump-shaped labor-income profile (rising through working years, falling after retirement) implies a hump in the savings policy $k'^h$. Sketch the expected shape and check it against the trained-network output in notebook [`lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb).
```

```{exercise}
:label: ex-ch5-4

**[Computational\] Hard aggregation layer.** The analytic notebook currently predicts cohort savings and then defines aggregate next-period capital as their sum, so capital-market clearing is already exact. Implement an alternative architecture in notebook [`lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb) in which the network outputs a scalar $\widehat K_{t+1}>0$ and unnormalized cohort scores $(z^2,\ldots,z^A)$, then sets $s^h=\mathrm{softmax}(z^h)$ and $k_{t+1}^h=\widehat K_{t+1}s^h$. Verify that $\sum_{h=2}^{A} k_{t+1}^h=\widehat K_{t+1}$ is at machine precision (below $10^{-12}$) at every training step. Compare the Euler residual and runtime against the current "sum-of-savings" implementation, and explain why the hard layer is useful mainly when aggregate $K_{t+1}$ is a separate policy head.
```

````{exercise}
:label: ex-ch5-5

**[Core\] Bond pricing in equilibrium.** Assume the borrowing/collateral constraint is slack for cohort $h$ throughout this exercise. In the 56-agent OLG ({ref}`sec-olg_56`) cohort $h$ holds capital $k^h$ and one-period riskless bonds $b^h$; capital pays the stochastic gross return $R_{t+1}$, bonds pay one unit of consumption next period and trade at price $p_t$. Write the agent's Euler equations for capital and for bonds separately. Eliminate the marginal utilities to derive the equilibrium bond-pricing equation

```{math}
:enumerated: false

p_t \;=\; \frac{\beta\,\mathbb{E}_t\!\bigl[u'(c^{h+1}_{t+1})\bigr]}{u'(c^h_t)},
```

and show that the same expression equals $\mathbb{E}_t[M_{t,t+1}]$ for the stochastic discount factor $M_{t,t+1}:=\beta u'(c^{h+1}_{t+1})/u'(c^h_t)$. Show that absence of arbitrage between bonds and capital implies $1/p_t = \mathbb{E}_t[R_{t+1}] + \mathrm{Cov}_t(M_{t,t+1}, R_{t+1})/p_t$, equivalently $\mathbb{E}_t[R_{t+1}] - 1/p_t = -\mathrm{Cov}_t(M_{t,t+1}, R_{t+1})/p_t$, the standard risk-premium decomposition. Then explain qualitatively how the bond-pricing equation changes when the collateral multiplier $\mu^h_t > 0$ is positive (i.e., the constraint binds), and identify which cohorts are most likely to be affected. Briefly explain why the 6-agent analytic OLG of {ref}`sec-olg_analytic` does not need an explicit bond-market residual in its DEQN loss.
````

```{exercise}
:label: ex-ch5-6

**[Advanced/project\] Collateral sensitivity.** In the 56-agent benchmark notebook [`lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb), the collateral constraint is $k'^h + \kappa b'^h \ge 0$ with reference $\kappa = 1/(1-\delta_{\max}) \approx 1.1236$. Sweep $\kappa \in \{0.8, 1.0, 1.1236, 1.3, 1.5\}$, retraining the network for each. Report (i) the fraction of ergodic states at which the collateral constraint binds, measured by $\hat q^h < 10^{-4}$ together with a positive multiplier $\hat\mu^h > 10^{-4}$, (ii) the cross-cohort dispersion of bond holdings $b^h$ at the deterministic steady state, (iii) the equilibrium bond price $p_\mathrm{ss}$. Plot all three against $\kappa$ and explain qualitatively how tightening the collateral requirement (larger $\kappa$, hence less negative admissible bond positions for a fixed $k'^h$) changes portfolio choice.
```

```{exercise}
:label: ex-ch5-7

**[Advanced/project\] KKT-binding frequency under aggregate volatility.** Still in the 56-agent benchmark, vary the standard deviation of the aggregate productivity shock $\sigma_z \in \{0.005, 0.01, 0.02, 0.04\}$ (the reference calibration uses $\sigma_z \approx 0.01$). For each, retrain and report the fraction of ergodic-set draws on which (a) the borrowing constraint $k'^h \ge 0$ binds and (b) the collateral constraint $k'^h + \kappa b'^h \ge 0$ binds, broken out by cohort age. Use the same small-slack/positive-multiplier convention as in {prf:ref}`ex-ch5-6`. Show that the binding fraction grows roughly linearly in $\sigma_z$ for the youngest cohorts but stays near zero for older cohorts, and connect this to the chapter's recommendation ({ref}`sec-olg_fb`) that product-form KKT residuals suffice when the constraint binds rarely.
```

[^1]: The 56-agent benchmark of {ref}`sec-olg_56` adds two genuine extras to {eq}`eq-olg_loss`: KKT product residuals (because the borrowing and collateral constraints actually bind) and an explicit bond-market-clearing residual (because the network outputs each agent's bond holding independently). An orthogonal extension is to encode capital-market clearing *exactly* via a dedicated output layer that rescales unnormalized cohort savings so that $\sum_{h=2}^{A} k_{t+1}^{h} = K_{t+1}$ holds by construction; {cite:t}`azinoviczemlicka_2024` adopt this design in an OLG economy with rare disasters.

[^2]: In the current notebook implementation $\hat q^h$ is parameterized *relative* to $\hat k'^h$, so it cannot fall to zero while $\hat k'^h>0$; the collateral-complementarity residual is then satisfied by $\hat\mu^h\to 0$, and the collateral constraint is effectively non-binding on the learned ergodic set, consistent with the chapter-opening note. Allowing it to bind exactly requires a free positive slack output (a softplus head on $\hat q^h$); the architecture above already accommodates this swap.
