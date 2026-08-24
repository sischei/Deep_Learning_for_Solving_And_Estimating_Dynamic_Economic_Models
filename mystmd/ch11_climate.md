---
title: "Climate Economics and Deep Uncertainty Quantification"
label: ch-climate
---

This chapter brings together the methods developed throughout this script and applies them to one of the most consequential computational challenges in economics: *climate change policy*. Integrated assessment models (IAMs) couple economic growth with the carbon cycle, temperature dynamics, and climate damages, creating high-dimensional nonstationary dynamic programming problems that are ideal candidates for the DEQN and surrogate methods we have developed. We present the CDICE model of {cite:t}`Folini_2021`, solve it with DEQNs, and then use GP surrogates and Bayesian active learning, first for deep uncertainty quantification {cite:p}`friedlDeep2023`, and then, applying the same surrogate-then-optimize machinery to a different OLG model and a different surrogate, to *search over policy parameters* and derive constrained Pareto-improving carbon tax rules in an OLG–IAM with deep uncertainty {cite:p}`kubler2025using`. This last step illustrates a general use of surrogates that goes beyond estimation and UQ: once the structural model has been mapped into a fast, differentiable surrogate, the costly outer loop of an optimal-policy search in a dynamic stochastic heterogeneous-agent economy collapses into a small optimization on the surrogate. For broader overviews of climate economics and IAMs, see {cite:t}`hassler2016environmental` on environmental macroeconomics, {cite:t}`DIETZ20241` on IAMs, {cite:t}`fernandezvillaverde2025climate` on the intersection of climate economics and deep learning, and {cite:t}`vanderploegrezai2026climate` on the macroeconomics of climate policy.

(sec-climate_motivation)=
## The Macroeconomics of Climate Change
Climate change is a global externality: the emissions of each agent affect the welfare of all agents, including future generations that have no say in current decisions. Unlike standard market failures, the climate externality operates across centuries, involves deep scientific uncertainty, and couples the macroeconomy with the earth system in both directions. Recent overviews include {cite:t}`hassler2016environmental` on environmental macroeconomics, {cite:t}`DIETZ20241` on IAMs, {cite:t}`fernandezvillaverde2025climate` on climate economics and deep learning, and {cite:t}`vanderploegrezai2026climate` on the macroeconomics of climate policy.

**Integrated assessment models.** Integrated assessment models (IAMs) formalize this coupling. The economy produces output and emissions; emissions accumulate in the atmosphere and raise global temperature; temperature increases cause damages that reduce output. The feedback loop is closed ({numref}`fig-iam_feedback_loop`):

```{figure} figures/fig-iam_feedback_loop.svg
:name: fig-iam_feedback_loop

The integrated-assessment feedback loop. The economy produces output and CO$_2$ emissions; emissions accumulate in the atmosphere and raise global mean temperature ($\Delta T$); higher temperatures generate damages that reduce output and consumption, which in turn shape the path of future emissions. An IAM closes this loop and uses it to quantify the welfare cost of additional emissions, summarized by the social cost of carbon {eq}`eq-scc`.
```

The central output of an IAM is the **social cost of carbon** (SCC): the marginal welfare cost of one additional unit of CO$_2$ emissions, measured in consumption-equivalent units. When emissions are measured in GtC (gigatons of carbon), the SCC has units of consumption per GtC. Conversion to USD per tCO$_2$ requires first applying the consumption-to-USD numeraire and then converting the carbon mass unit: one tCO$_2$ contains $12/44$ tons of carbon, so a price expressed per ton of carbon is divided by $44/12$ to obtain the corresponding price per ton of CO$_2$ (and a GtC price is also divided by $10^9$). Formally,

$$
\mathrm{SCC}_t = -\frac{\partial V_t / \partial E_t}{\partial V_t / \partial C_t},
$$ (eq-scc)

where $V_t$ is the value function, $E_t$ is contemporaneous emissions, and $C_t$ is consumption. The flow form is linked to the stock-based form $\mathrm{SCC}^M_t = -(\partial V_t/\partial M^{\mathrm{AT}}_t)/(\partial V_t/\partial C_t)$, derived in {ref}`sec-dice_deqn`, by the chain rule

```{math}
:enumerated: false

\frac{\partial V_t}{\partial E_t}
\;=\;
\frac{\partial V_{t+1}}{\partial M^{\mathrm{AT}}_{t+1}}\,\frac{\partial M^{\mathrm{AT}}_{t+1}}{\partial E_t}
```

together with the carbon-to-CO$_2$ unit conversion noted above. In a first-best allocation, the optimal carbon tax equals the SCC {cite:p}`golosov2014optimal`. The SCC is high when climate damages are steep, the climate response is strong, discounting is low, and tipping risks are material {cite:p}`caiSocialCostCarbon2019,DIETZ20241`.

**From surrogates to climate IAMs.** {ref}`ch-gp` introduced surrogates and Bayesian active learning as fast approximators for repeated model evaluations. Climate IAMs are the natural application: each parameter configuration (climate sensitivity, damage curvature, discount rate) is expensive to solve, yet policy questions require evaluating thousands of configurations to map out tail risks and Pareto-improving rules. The DEQN approach of {ref}`ch-deqn`–{ref}`ch-irbc`, combined with the GP and active-learning toolkit of {ref}`ch-gp`, is therefore the natural workhorse for climate-policy uncertainty quantification.

**Why computation matters.** Solving IAMs globally, as opposed to linearization or certainty equivalence, is computationally demanding for several reasons:

- **Nonstationarity:** TFP, population, emissions intensity, and radiative forcing all change exogenously over time, so the policy function cannot be time-invariant.

- **Coupled dynamics:** the economy and climate interact in both directions through emissions and damages.

- **Long horizon:** welfare effects unfold over 100–300 years, requiring stable numerical solutions far from the steady state.

- **Curse of dimensionality:** multiple climate state variables (carbon stocks, temperature layers), stochastic shocks, and uncertain parameters raise the dimension of the state space well beyond what standard grid-based methods can handle.

The deep learning toolkit developed in this course (DEQNs, deep surrogates, and GP-based uncertainty quantification) is therefore particularly well suited to climate economics.

**The three movements of this chapter.** The remainder of this chapter has three movements. Movement 1 ({ref}`sec-iam_nonstationarity`–{ref}`sec-nsdeqn_algo`) makes precise what changes when we ask the Deep Equilibrium Network of {ref}`ch-deqn` to solve a non-stationary IAM, and presents the modified training algorithm in one labeled box. Movement 2 ({ref}`sec-dice_deqn`–{ref}`sec-dice_to_stochastic_iam`) puts that algorithm to work on a concrete stochastic DICE economy. Movement 3 ({ref}`sec-bayesian_learning`–{ref}`sec-pareto_carbon_tax`) sketches the four extensions that matter for serious climate-finance research: Bayesian learning on the climate sensitivity, recursive Epstein–Zin preferences, global uncertainty quantification of the social cost of carbon, and constrained Pareto-improving carbon-tax design in a heterogeneous-agent IAM.

(sec-dice)=
## The DICE Model
(sec-iam_landscape)=
### The IAM Landscape
DICE is the workhorse of this chapter, but it is one of several integrated assessment models in active use. The list below summarizes the active landscape; each model trades global parsimony for regional or sectoral granularity, and computational tractability for fidelity of the climate physics.

DICE.
: Global aggregate; 3-box carbon cycle and 2-layer energy-balance model; one-sector Ramsey planner. The standard benchmark for SCC and integrated policy analysis {cite:p}`nordhausRevisitingSocialCost2017`.

RICE.
: Twelve-region extension of DICE with trade {cite:p}`nordhaus1996regional`. Used for regional SCC and equity questions.

CDICE.
: A global DICE-2016 recalibration tailored to deep-learning solution methods, with Epstein–Zin preferences and OLG variants. The model used in {ref}`sec-dice_deqn` below {cite:p}`Folini_2021`.

ACE.
: Analytic Climate Economy: log-linear approximations to the carbon cycle, temperature dynamics, and damages yield a *closed-form* optimal carbon tax {cite:p}`traeger2018ace`. Acts as an analytic benchmark for the numerical SCC computed below.

FaIR / MAGICC.
: Reduced-complexity climate emulators that take emissions as input and produce temperature responses; widely used to translate IPCC scenarios to economic models.

WITCH / REMIND.
: Multi-region IAMs with full energy-system modules; standard for mitigation-pathway and technology-portfolio studies. Outside the scope of this script.


CDICE is the model we solve in this chapter. ACE provides a useful analytic shadow for it, in particular a closed-form SCC that decomposes transparently into structural parameters; we do not derive that closed form here, but {prf:ref}`ex-ch11-3` asks the reader to compute it from {cite:t}`traeger2018ace` and compare against the DEQN-trained CDICE solution as an external sanity check.

The *Dynamic Integrated model of Climate and the Economy* (DICE), developed by {cite:t}`nordhaus1994managing`, is the most influential IAM; in this chapter we follow the variant of {cite:t}`nordhausRevisitingSocialCost2017` as recalibrated by {cite:t}`Folini_2021`. It couples a neoclassical growth model with a reduced-form climate module in a single global framework. The remainder of this section builds the model up block by block, in increasing complexity: first the macro-economic backbone ({ref}`sec-dice_ramsey`), then the emissions and abatement technology ({ref}`sec-dice_emissions`), then the climate physics ({ref}`sec-dice_carbon_cycle`–{ref}`sec-dice_temperature`), and finally the damage feedback ({ref}`sec-dice_damages`) that closes the loop. A consolidated calibration is given in {numref}`tab-dice_calibration`.

**Time-step convention.** Following {cite:t}`Folini_2021` we calibrate CDICE on an *annual* time step, $\Delta_t = 1$ year, so that all rates in {numref}`tab-dice_calibration` (capital depreciation $\delta$, pure rate of time preference $\rho$, the decay rates $g^{\sigma}_0, \delta^{\sigma}, g^{\mathrm{back}}, \delta^{\mathrm{Land}}$, the carbon-cycle transfer rates $b_{12}, b_{23}$, and the temperature-block coefficients $c_1, c_3, c_4$) are read directly as annual values; the original DICE-2016 calibration of {cite:t}`nordhausRevisitingSocialCost2017`, by contrast, hard-wires a 5-year time step into its coefficients. Growth rates of TFP and population are written as annual log changes, $g^A_t := \ln(A_{t+1}/A_t)$ and $g^L_t := \ln(L_{t+1}/L_t)$, and the dynamics {eq}`eq-carbon_cycle`, {eq}`eq-temp_at`–{eq}`eq-temp_oc` and the FOC residuals of {ref}`sec-dice_deqn` therefore carry no $\Delta_t$ multipliers; emissions $E_t$ entering {eq}`eq-carbon_cycle` are the annual total. Switching to a non-annual $\Delta_t$ amounts to reinserting the multiplications $\Delta_t \cdot \{g^{\sigma}_0, \delta^{\sigma}, g^{\mathrm{back}}, \delta^{\mathrm{Land}}, b_{12}, b_{23}, c_1, c_3, c_4\}$ in the obvious places, the time-step-generic form discussed in Online Appendix D of {cite:t}`Folini_2021`.

(sec-dice_ramsey)=
### Production and the Ramsey–Cass–Koopmans backbone
Strip away the climate block and DICE is just a neoclassical growth model with population and TFP growth. A single representative firm produces gross output with Cobb–Douglas technology in capital and effective labor,

$$
Y^{\mathrm{gross}}_t \;=\; K_t^{\alpha}\,(A_t L_t)^{1-\alpha},
$$ (eq-gross_output)

where $\alpha\in(0,1)$ is the capital share, $A_t$ is total factor productivity, and $L_t$ is population. Both $A_t$ and $L_t$ follow deterministic but time-varying paths: $A_t$ trends because of exogenous productivity growth, and $L_t$ follows the calibrated demographic projection of {cite:t}`nordhausRevisitingSocialCost2017`. The capital stock evolves under the standard accumulation law

$$
K_{t+1} \;=\; (1-\delta) K_t + I_t,
$$ (eq-capital_accumulation)

with depreciation rate $\delta$ and gross investment $I_t$. The economy's resource constraint, written in terms of net (after-damages, after-abatement) output that we develop in {ref}`sec-dice_emissions`–{ref}`sec-dice_damages`, is $C_t + I_t \le Y^{\mathrm{net}}_t$, where $C_t$ is aggregate consumption.

A benevolent planner picks $(C_t,\, I_t,\, \mu_t)_{t\ge 0}$ to maximize a discounted CRRA-IES felicity sum,

$$
V_0 \;=\; \sum_{t=0}^{\infty} \beta_t\, L_t\,\frac{(C_t/L_t)^{1-1/\psi}-1}{1-1/\psi},
\qquad
\beta_t \;=\; \exp(-\rho\,\Delta_t \cdot t),
$$ (eq-planner_obj_crra)

with intertemporal-elasticity-of-substitution parameter $\psi>0$ and pure rate of time preference $\rho$. This is the time-additive aggregator of the standard Ramsey–Cass–Koopmans growth model; we replace it with the recursive Epstein–Zin form once stochastic risk enters the picture ({ref}`sec-ez_layer`). The planner controls $\mu_t$, the emissions abatement rate, in addition to the savings–consumption split; we develop the cost of abatement next.

(sec-dice_emissions)=
### Industrial emissions, abatement, and the backstop technology
Industrial production is a CO$_2$-emitting activity. Let $\sigma_t$ denote the *carbon intensity* of gross output, expressed in CDICE's working units of $10^3$ GtC of emissions per unit of gross output (a $10^3$ GtC normalization on the carbon stocks improves the conditioning of the climate side; see {numref}`tab-dice_calibration`). Industrial emissions are then $\sigma_t Y^{\mathrm{gross}}_t$ before any mitigation effort; with abatement rate $\mu_t \in [0,1]$ the planner can scale these emissions down,

$$
E_{\mathrm{ind},t} \;=\; (1-\mu_t)\,\sigma_t\, Y^{\mathrm{gross}}_t \;=\; (1-\mu_t)\,\sigma_t\, K_t^{\alpha}(A_t L_t)^{1-\alpha}.
$$ (eq-emissions_ind)

Carbon intensity is itself an exogenous decreasing time path. DICE-2016 calibrates a closed-form decay,

$$
\sigma_t \;=\; \sigma_0\,\exp\!\left[\frac{g^{\sigma}_0}{\log(1+\delta^{\sigma})}\bigl((1+\delta^{\sigma})^{t}-1\bigr)\right],
$$ (eq-sigma_decay)

with initial intensity $\sigma_0$, initial growth rate $g^{\sigma}_0<0$ (so emissions per dollar of output fall over time), and second-derivative parameter $\delta^{\sigma}>0$ that bends the path further down at long horizons. Equation {eq}`eq-sigma_decay` captures the steady decarbonization that even *unabated* world output undergoes through ongoing technological change; the planner's $\mu_t$ is the additional mitigation effort *on top of* that baseline.

Abatement is not free. In the spirit of an aggregate marginal-abatement-cost curve, DICE assumes the abatement-cost share of gross output is a power function of $\mu_t$,

$$
\Theta(\mu_t) \;=\; \theta_{1,t}\,\mu_t^{\theta_2},
$$ (eq-abat_cost)

with curvature parameter $\theta_2>1$ (a typical calibration is $\theta_2=2.6$). The level coefficient $\theta_{1,t}$ is not a free parameter: it is pinned down by the cost of the *backstop technology*, the cleanest large-scale abatement technology available at any given time (e.g. direct air capture). Let $p^{\mathrm{back}}_t$ denote the cost per unit of CO$_2$ avoided when the backstop is fully deployed, and assume an exogenous declining path,

$$
p^{\mathrm{back}}_t \;=\; p^{\mathrm{back}}_0\,\exp(-g^{\mathrm{back}}\,t),
$$ (eq-backstop_price)

reflecting steady cost reductions in clean technologies. Setting the marginal abatement cost at $\mu_t=1$ equal to the backstop price (multiplied by carbon-to-CO$_2$ conversion $\mathrm{c2co2}$ to keep mass units consistent, and by $10^3$ to convert $\sigma_t$ from $10^3$ GtC working units back to GtC) yields the calibration identity

$$
\theta_{1,t} \;=\; \frac{p^{\mathrm{back}}_t \cdot 10^3 \cdot \mathrm{c2co2} \cdot \sigma_t}{\theta_2}.
$$ (eq-theta1_calibration)

Equation {eq}`eq-theta1_calibration` is what makes $\Theta(\mu)$ *economically* meaningful rather than a fitted polynomial: the abatement-cost function inherits its level from the backstop price and its curvature from the assumption $\theta_2=2.6$. The $10^3$ factor matches Equation (11) of Online Appendix D of {cite:t}`Folini_2021` and the corresponding factor of `1000` in the companion implementation. As the backstop becomes cheaper ($p^{\mathrm{back}}_t \downarrow$), full mitigation becomes cheaper too, which is one of the channels that makes the deterministic optimal $\mu_t$ rise toward 1 over the 21st century.

The bound $\mu_t \in [0,1]$ deserves a comment. $\mu_t = 0$ means business-as-usual emissions; $\mu_t = 1$ means full deployment of the backstop, eliminating all industrial emissions. Values $\mu_t > 1$ would correspond to net-negative industrial emissions (e.g. aggressive direct air capture beyond the firm's own footprint), which DICE forbids; we will impose the upper bound as a Kuhn–Tucker constraint, smoothed by a Fischer–Burmeister term, in {ref}`sec-dice_deqn`.

(sec-dice_landuse)=
### Land-use emissions and net output
The atmosphere does not distinguish between an industrial flow and a non-industrial flow of carbon. In DICE, total emissions therefore comprise an industrial component {eq}`eq-emissions_ind` and an exogenous land-use-change component,

$$
E_{\mathrm{Land},t} \;=\; E_{\mathrm{Land},0}\,\exp(-\delta^{\mathrm{Land}}\,t),
$$ (eq-land_emissions)

which decays smoothly toward zero as deforestation slows. Total emissions feeding the atmosphere are

$$
E_t \;=\; E_{\mathrm{ind},t} + E_{\mathrm{Land},t}.
$$ (eq-total_emissions)

Closing the production block requires accounting for two additional drains on gross output: climate damages, governed by atmospheric temperature $T^{\mathrm{AT}}_t$ via a damage fraction $\Omega(T^{\mathrm{AT}}_t)$ developed in {ref}`sec-dice_damages`, and abatement spending {eq}`eq-abat_cost`. Net output is therefore

$$
Y^{\mathrm{net}}_t \;=\; \bigl(1 - \Omega(T^{\mathrm{AT}}_t) - \Theta(\mu_t)\bigr)\,Y^{\mathrm{gross}}_t,
$$ (eq-net_output)

which is what is available for consumption and investment. The additive form is the convention adopted in CDICE and used by the production-grade DEQN library port; an alternative multiplicative form $(1-\Omega^{\mathrm{ret}})(1-\Theta)$ with retained-output factor $\Omega^{\mathrm{ret}}$ appears in {cite:t}`nordhaus2008question`.

The planner's controls and exogenous trends are now all named. The endogenous economic state is the capital stock $K_t$. The exogenous trends are TFP $A_t$, population $L_t$, carbon intensity $\sigma_t$, land-use emissions $E_{\mathrm{Land},t}$, and (added below) the non-CO$_2$ component of radiative forcing $F^{\mathrm{EX}}_t$. The planner controls the consumption–investment split (equivalently, the savings rate $s_t$) and the abatement rate $\mu_t \in [0,1]$. All that remains is the climate side: the carbon cycle that turns total emissions $E_t$ into atmospheric concentration, the energy balance that turns concentration into temperature, and the damage function that turns temperature back into output loss.

(sec-dice_carbon_cycle)=
### Carbon cycle
DICE represents the global carbon cycle as a three-reservoir linear system: an atmospheric box, an upper (mixed-layer) ocean box, and a lower (deep) ocean box. Carbon flows between reservoirs at calibrated rates, and total emissions $E_t$ from {eq}`eq-total_emissions` enter directly into the atmospheric reservoir. Stacking concentrations as $M_t = (M^{\mathrm{AT}}_t,\, M^{\mathrm{UO}}_t,\, M^{\mathrm{LO}}_t)^\top$, the transition is

$$
M_{t+1} \;=\; (I + B)\, M_t \;+\; \bm{e}_1\,E_t,
$$ (eq-carbon_cycle)

where $\bm{e}_1 = (1,0,0)^\top$ injects emissions into the atmosphere alone, $E_t$ is the per-period emissions total, and the transfer matrix

$$
B \;=\; \begin{pmatrix}
-b_{12} & b_{12}\,M^{\mathrm{AT}}_{\mathrm{eq}}/M^{\mathrm{UO}}_{\mathrm{eq}} & 0 \\
b_{12} & -b_{12}\,M^{\mathrm{AT}}_{\mathrm{eq}}/M^{\mathrm{UO}}_{\mathrm{eq}} - b_{23} & b_{23}\,M^{\mathrm{UO}}_{\mathrm{eq}}/M^{\mathrm{LO}}_{\mathrm{eq}} \\
0 & b_{23} & -b_{23}\,M^{\mathrm{UO}}_{\mathrm{eq}}/M^{\mathrm{LO}}_{\mathrm{eq}}
\end{pmatrix}
$$ (eq-carbon_B_matrix)

encodes the two atmosphere–upper-ocean exchange rates ($b_{12}$ in either direction) and the two upper-ocean–lower-ocean exchange rates ($b_{23}$ in either direction). The off-diagonal scaling by the equilibrium-mass ratios $M^{\mathrm{AT}}_{\mathrm{eq}}/M^{\mathrm{UO}}_{\mathrm{eq}}$ and $M^{\mathrm{UO}}_{\mathrm{eq}}/M^{\mathrm{LO}}_{\mathrm{eq}}$ guarantees that, under zero net emissions, the system relaxes to the calibrated pre-industrial equilibrium $M_{\mathrm{eq}} = (M^{\mathrm{AT}}_{\mathrm{eq}},\, M^{\mathrm{UO}}_{\mathrm{eq}},\, M^{\mathrm{LO}}_{\mathrm{eq}})^\top$. Calibrated values for $b_{12}, b_{23}$, and $M_{\mathrm{eq}}$ in CDICE are listed in {numref}`tab-dice_calibration`. The lecture slides for this chapter sometimes write the same transition with four directional rates $\phi_{12},\phi_{21},\phi_{23},\phi_{32}$ in place of $b_{12}$ and $b_{23}$; the two parameterizations are identical under $\phi_{12} = b_{12}$, $\phi_{21} = b_{12}\,M^{\mathrm{AT}}_{\mathrm{eq}}/M^{\mathrm{UO}}_{\mathrm{eq}}$, $\phi_{23} = b_{23}$, $\phi_{32} = b_{23}\,M^{\mathrm{UO}}_{\mathrm{eq}}/M^{\mathrm{LO}}_{\mathrm{eq}}$, i.e. the slide form makes the equilibrium-mass scaling absorbed into $B$ explicit at the cost of two extra symbols.

Equation {eq}`eq-carbon_cycle` is a pulse-and-decay system: a unit pulse of emissions raises atmospheric carbon by one unit instantaneously, and that anomaly then bleeds into the upper ocean over decades and into the deep ocean over centuries. {numref}`fig-restud_bau_emissions` shows the implied BAU emissions trajectory under nine alternative climate-module calibrations; the spread is mostly driven by the equilibrium climate sensitivity (developed in {ref}`sec-dice_temperature`), not by the carbon cycle, which is tightly disciplined by the pulse and step tests of {ref}`sec-cdice_recalibration`.

```{figure} figures/restud_fig11a.png
:name: fig-restud_bau_emissions
:width: 85%

Business-as-usual industrial emissions in CDICE (in GtCO$_2$/yr) under the nine combinations of three carbon-cycle calibrations (MMM, MESMO, LOVECLIM) and three temperature calibrations (MMM, HadGEM2-ES, GISS-E2-R); the thin CDICE curves overlap visually, confirming that the BAU emissions path is essentially insensitive to the climate-module calibration because $\sigma_t$ and $A_t$ are exogenous. The thick red and orange curves are the RCP 8.5 and RCP 6.0 scenarios, included as climate-policy reference paths. Reproduced from {cite:t}`Folini_2021`, Figure 11(a).
```

(sec-dice_temperature)=
### Two-layer energy balance and radiative forcing
A two-layer energy balance model links carbon concentrations to temperature:

$$
T^{\mathrm{AT}}_{t+1} = T^{\mathrm{AT}}_t + c_1 \bigl(F_t - \lambda\, T^{\mathrm{AT}}_t - c_3(T^{\mathrm{AT}}_t - T^{\mathrm{OC}}_t)\bigr)
$$ (eq-temp_at)

$$
T^{\mathrm{OC}}_{t+1} = T^{\mathrm{OC}}_t + c_4 \bigl(T^{\mathrm{AT}}_t - T^{\mathrm{OC}}_t\bigr)
$$ (eq-temp_oc)

where radiative forcing is

$$
F_t = F_{\mathrm{2\times CO_2}} \frac{\log(M^{\mathrm{AT}}_t / M^{\mathrm{AT}}_{\mathrm{PI}})}{\log 2} + F^{\mathrm{EX}}_t.
$$ (eq-forcing)

{numref}`fig-cdice_climate_topology` summarizes the full topology of the climate side: industrial emissions enter the atmospheric carbon stock, leak into the upper and lower ocean reservoirs at calibrated rates, raise radiative forcing through the logarithmic CO$_2$ term, and warm the atmospheric and ocean temperature layers through the two-layer energy balance.

```{figure} figures/fig-cdice_climate_topology.svg
:name: fig-cdice_climate_topology

Topology of the CDICE climate side. Total emissions $E_t$ enter the atmospheric carbon box $M^{\mathrm{AT}}_t$, leak into the upper- and lower-ocean carbon boxes at exchange rates $b_{12}$ and $b_{23}$, and drive radiative forcing $F_t$ through the logarithmic CO$_2$ relation. The two-layer energy balance maps $F_t$ into the atmospheric temperature $T^{\mathrm{AT}}_t$ via $c_1$, with $c_3, c_4$ governing the heat exchange between atmosphere and ocean. The dashed arrow closes the loop through the damage function back into output (developed in {ref}`sec-dice_damages`). Five climate states $(M^{\mathrm{AT}}, M^{\mathrm{UO}}, M^{\mathrm{LO}}, T^{\mathrm{AT}}, T^{\mathrm{OC}})$ form the climate-side block of the DEQN state vector {eq}`eq-iam_state`.
```

The parameter $\lambda = F_{\mathrm{2\times CO_2}} / \Delta T_{\mathrm{AT},\times 2}$ is determined by the *equilibrium climate sensitivity* (ECS), defined as the long-run atmospheric warming from a doubling of CO$_2$ concentration. We treat $\lambda$ as a deterministic constant in the baseline model; {ref}`sec-bayesian_learning` promotes it to a learnable Gaussian parameter, with the additive feedback term $\varphi_{1C}\tilde f_{t+1} T^{\mathrm{AT}}_t$ entering the right-hand side of {eq}`eq-temp_at` and the coefficient $\varphi_{1C}$ defined in that subsection. ECS is one of the most consequential and uncertain parameters in climate science {cite:p}`roe2007climate,knutti2017beyond`. Observational and model-based estimates place ECS in a *likely* (66 %) range of 2.5°C–4°C and a *very likely* (90 %) range of 2°C–5°C, with a best estimate of approximately 3°C {cite:p}`calvinIPCC2023Climate2023a`; ECS uncertainty is one of the largest single sources of variance in the SCC.

(sec-dice_damages)=
### Damage function: closing the climate–economy loop
The damage function is what turns a temperature anomaly back into an output loss, and so it is what closes the economy–climate–damages feedback loop drawn schematically in {numref}`fig-iam_feedback_loop`. Following the convention in {cite:t}`Folini_2021`, Online Appendix D, we treat $\Omega(T_{\mathrm{AT}})$ as the *damage fraction* of gross output (the fraction lost to climate damages, increasing in $T_{\mathrm{AT}}$), and the abatement-cost fraction $\Theta(\mu)$ from {eq}`eq-abat_cost` as a separate output drain. The two enter additively in net output {eq}`eq-net_output`; an alternative multiplicative form $(1-\Omega^{\mathrm{ret}})(1-\Theta)$ with retained-output factor $\Omega^{\mathrm{ret}}$ is used by {cite:t}`nordhaus2008question`.

The workhorse specification is {cite:t}`nordhaus2008question`'s quadratic,

$$
\Omega^N(T_{\mathrm{AT}}) \;=\; \pi_1\, T_{\mathrm{AT}} + \pi_2\, T_{\mathrm{AT}}^2,
$$ (eq-damage_nordhaus)

which is relatively benign for moderate warming and is what we use in the deterministic CDICE solve below. Calibrated values $(\pi_1, \pi_2)$ are listed in {numref}`tab-dice_calibration`. The damage function {eq}`eq-damage_nordhaus` is the most contested object in the IAM literature: at $T_{\mathrm{AT}}=3\,{}^\circ\mathrm{C}$ above pre-industrial, Nordhaus–quadratic damages amount to roughly $2\%$ of gross output, which several recent empirical literatures argue is far below realistic central estimates. We therefore treat the damage curvature $\pi_2$ as one of the two key uncertain parameters in the deep-UQ analysis of {ref}`sec-deep_uq` (the other being the equilibrium climate sensitivity).

For the tipping-point branch of the literature, {cite:t}`weitzman2012ghg` argued that catastrophic thresholds require a steeper damage function,

$$
\Omega^W(T_{\mathrm{AT}}) \;=\; 1 \;-\; \frac{1}{1 + \bigl(\tfrac{1}{\psi_1} T_{\mathrm{AT}}\bigr)^2 + \bigl(\tfrac{1}{2\, TP} T_{\mathrm{AT}}\bigr)^{6.754}},
$$ (eq-damage_weitzman)

where $TP$ is a stochastic tipping-point threshold. We do not solve a Weitzman damage variant in the baseline CDICE-DEQN, but the OLG-IAM of {ref}`sec-pareto_carbon_tax` introduces a stylized tipping risk in the same spirit; the degree of convexity of the damage function is one of the most important determinants of the optimal carbon tax.

(sec-cdice_recalibration)=
### CDICE: recalibration of the climate module
A key contribution of {cite:t}`Folini_2021` is a systematic recalibration of the DICE climate module against benchmarks from climate science model archives (CMIP). Their CDICE framework retains the same functional forms as DICE but fits parameters to the four-test protocol summarized in {numref}`tab-cdice_tests`.

````{table}
:name: tab-cdice_tests

CDICE climate-module calibration protocol. The first two tests discipline the carbon-cycle and temperature-response blocks directly; the last two check whether the calibrated reduced-form module remains accurate on out-of-sample and historically realistic forcing paths.

| **Test** | **Target** | **Use** |
|---|---|---|
| 1. Carbon pulse (100 GtC) | Atmospheric retention path | Calibrate carbon cycle |
| 2. $4\times$CO$_2$ step | Temperature impulse response | Calibrate temperature block |
| 3. 1% CO$_2$/year | Transient climate response | Out-of-sample validation |
| 4. Historical + RCP | Realistic forcing paths | End-to-end validation |
````

This calibration ensures that the reduced-form climate module is consistent with state-of-the-art earth system models. CDICE also introduces a transparent time-step formulation, $X_{t+\Delta t} = X_t + \Delta t \cdot f(X_t, u_t; \theta)$, that allows coherent implementation at annual, 5-year, or 10-year resolution within a single generic framework. {numref}`fig-restud_bau_mat` illustrates how much the climate-cycle calibration matters even before the planner makes any decision: under business-as-usual, DICE-2016 and CDICE produce visibly different atmospheric carbon trajectories, and the gap propagates into temperature, damages, and ultimately the SCC.

```{figure} figures/restud_fig15a.png
:name: fig-restud_bau_mat
:width: 85%

Atmospheric carbon $M^{\mathrm{AT}}_t$ along the BAU path (in GtC, over 200 years from 2015) under the three CDICE carbon-cycle calibrations (CDICE = MMM, CDICE-MESMO, CDICE-LOVECLIM) and the legacy DICE-2016 carbon cycle. Only the carbon-cycle block is varied here; the temperature block is held at the CDICE MMM calibration, since the BAU carbon-stock path does not depend on the temperature calibration to first order. The DICE-2016 path lies systematically above the CMIP-disciplined paths, reflecting that the original DICE carbon cycle overstates atmospheric retention; CDICE-MESMO and CDICE-LOVECLIM bracket the CDICE baseline on the slow-removal and fast-removal sides, respectively. Reproduced from {cite:t}`Folini_2021`, Figure 15(a).
```

(sec-dice_calibration)=
### Calibration and initial conditions, in one place
The block-by-block model description above introduces a fairly large set of parameters. {numref}`tab-dice_calibration` consolidates the calibration we use throughout the rest of the chapter, lifted from the Online Appendix of {cite:t}`Folini_2021`. Two CMIP5 alternatives (HadGEM2-ES and GISS-E2-R) are shown alongside the multi-model mean (MMM) so that the deep-UQ analysis of {ref}`sec-deep_uq` has a concrete distribution to draw from. We follow the CDICE convention of expressing all carbon quantities in $10^3$ GtC working units: equilibrium and initial carbon stocks $M_{\mathrm{eq}}$ and $M_0$, the initial carbon intensity $\sigma_0$, and the initial land-use emissions $E_{\mathrm{Land},0}$ are all on the same scale, which keeps the numerical conditioning of the carbon-cycle and emissions states under control. The factor $10^3$ appears explicitly in the abatement-cost calibration {eq}`eq-theta1_calibration` to convert $\sigma_t$ back to GtC when it is multiplied by the backstop price; a reader comparing values against raw DICE-2016 numbers (e.g. $\sim 2.6$ GtC/yr land-use emissions, $\sim 851$ GtC atmospheric carbon in 2015) should multiply the table entries by $10^3$ first.

````{table}
:name: tab-dice_calibration

CDICE baseline calibration used in the deterministic CDICE-DEQN solve. Parameter values follow the Online Appendix of {cite:t}`Folini_2021` and are stated on an annual time step ($\Delta_t = 1$ yr). All carbon quantities ($M_{\mathrm{eq}}, M_0, \sigma_0, E_{\mathrm{Land},0}$) are in CDICE's $10^3$ GtC working units; multiply by $10^3$ to recover GtC. Two alternative climate calibrations (CDICE-HadGEM2-ES, CDICE-GISS-E2-R) are listed in the temperature block, with their full free-parameter sets $\{c_1, c_3, c_4, F_{\mathrm{2\times CO_2}}, \lambda\}$ and corresponding ECS, since simply varying ECS while holding the rest of the temperature block fixed is *not* equivalent to using the full CMIP5 calibration {cite:p}`Folini_2021`. Initial state is for year 2015.

| **Block** | **Parameter** | **Value** | **Meaning** |
|---|---|---|---|
| Economy | $\alpha$ | $0.30$ | Capital share in Cobb–Douglas output |
|  | $\delta$ | $0.10$/yr | Capital depreciation rate |
|  | $\rho$ | $0.015$/yr | Pure rate of time preference |
|  | $\psi$ | $0.69$ | Intertemporal elasticity of substitution |
| Emissions & | $\sigma_0$ | $9.556\!\times\!10^{-5}$ ($10^3$ GtC)/USD | Initial carbon intensity |
| abatement | $g^{\sigma}_0$ | $-0.0152$/yr | Initial decay rate of $\sigma_t$ |
|  | $\delta^{\sigma}$ | $0.001$/yr | Curvature of $\sigma_t$ decay |
|  | $p^{\mathrm{back}}_0$ | $0.55$ thUSD/tCO$_2$ | Initial backstop price |
|  | $g^{\mathrm{back}}$ | $0.005$/yr | Decay rate of backstop price |
|  | $\theta_2$ | $2.6$ | Curvature of $\Theta(\mu)$ |
|  | $\mathrm{c2co2}$ | $3.666$ | Carbon-to-CO$_2$ mass conversion |
| Land use | $E_{\mathrm{Land},0}$ | $7.09\!\times\!10^{-4}$ ($10^3$ GtC)/yr | Initial land-use emissions |
|  | $\delta^{\mathrm{Land}}$ | $0.023$/yr | Decay rate of $E_{\mathrm{Land},t}$ |
| Carbon cycle | $b_{12}$ | $0.054$/yr | Atm.–upper-ocean transfer rate |
|  | $b_{23}$ | $0.0082$/yr | Upper-ocean–lower-ocean transfer rate |
|  | $M_{\mathrm{eq}}$ | $(0.607, 0.489, 1.281)$ ($10^3$ GtC) | Pre-industrial equilibrium masses |
| Temperature | $c_1$ (MMM) | $0.137$/yr | Atmospheric heat-capacity inverse |
|  | $c_3$ (MMM) | $0.73$/yr | Atm.–ocean coupling |
|  | $c_4$ (MMM) | $0.00689$/yr | Ocean heat-capacity inverse |
|  | $F_{\mathrm{2\times CO_2}}$ (MMM) | $3.45$ W/m$^2$ | Forcing from CO$_2$ doubling |
|  | $\lambda$ (MMM) | $1.06$ W/m$^2$/K | Climate feedback parameter |
|  | ECS (MMM) | $\approx 3.25\,{}^\circ$C | Equilibrium climate sensitivity |
|  | HadGEM2-ES | $(c_1,c_3,c_4)=(0.154,0.55,0.00671)$/yr | High-end CMIP5 calibration |
|  |  | $F_{\mathrm{2\times CO_2}}=2.95$, $\lambda=0.65$, ECS$\approx 4.55$ $^\circ$C |  |
|  | GISS-E2-R | $(c_1,c_3,c_4)=(0.213,1.16,0.00921)$/yr | Low-end CMIP5 calibration |
|  |  | $F_{\mathrm{2\times CO_2}}=3.65$, $\lambda=1.70$, ECS$\approx 2.15$ $^\circ$C |  |
| Damages | $\pi_1$ | $0.0$ | Linear damage coefficient |
|  | $\pi_2$ | $0.00236$ | Quadratic damage coefficient |
| Initial state | $K_0$ | $223$ T USD | Capital, year 2015 |
|  | $M_0$ | $(0.851, 0.628, 1.323)$ ($10^3$ GtC) | Atm./upper/lower carbon, 2015 |
|  | $T_0$ | $(1.10, 0.27)\,{}^\circ$C | Atm./ocean temp. above pre-industrial, 2015 |
````

(sec-dice_summary)=
### The full IAM, summarized
Pulling the previous subsections together, CDICE is a deterministic dynamical system on a finite-dimensional state vector that the planner steers with two controls. The endogenous state at date $t$ is the sextuple

$$
\bm{X}^{\mathrm{end}}_t \;=\; \bigl(K_t,\; M^{\mathrm{AT}}_t,\; M^{\mathrm{UO}}_t,\; M^{\mathrm{LO}}_t,\; T^{\mathrm{AT}}_t,\; T^{\mathrm{OC}}_t\bigr),
$$ (eq-dice_endo_state)

the exogenous-trend vector is

$$
\bm{X}^{\mathrm{exo}}_t \;=\; \bigl(A_t,\; L_t,\; \sigma_t,\; E_{\mathrm{Land},t},\; F^{\mathrm{EX}}_t\bigr),
$$

and the planner's controls are $(C_t,\, \mu_t)$ (equivalently $(K_{t+1},\, \mu_t)$, since investment is determined by the resource constraint $C_t + I_t = Y^{\mathrm{net}}_t$ together with {eq}`eq-capital_accumulation`). The transitions are: capital from {eq}`eq-capital_accumulation` with $I_t = Y^{\mathrm{net}}_t - C_t$; total emissions from {eq}`eq-total_emissions`, fed into the carbon cycle {eq}`eq-carbon_cycle`; temperature from {eq}`eq-temp_at`–{eq}`eq-temp_oc` with forcing {eq}`eq-forcing`; and net output, hence the resource constraint, from {eq}`eq-net_output`. The objective is the discounted CRRA-IES felicity sum {eq}`eq-planner_obj_crra` subject to $\mu_t \in [0,1]$.

That is the entire deterministic IAM. Every primitive named above has a closed-form expression and a calibrated parameter ({numref}`tab-dice_calibration`); the only thing left is to find the optimal policy $(C_t, \mu_t)_{t\ge 0}$. The model is intrinsically non-stationary. {ref}`sec-iam_nonstationarity` makes that observation precise; the stationary DEQN of {ref}`ch-deqn` needs to be amended before we can solve this system.

(sec-iam_nonstationarity)=
## Why DICE Breaks the Stationary DEQN
This is the technical pivot of the chapter. The stationary DEQN of {ref}`ch-deqn` was designed for models whose policy function is a fixed point of a Bellman operator on an ergodic state space. IAMs satisfy neither premise. Three structural features break the stationarity assumption simultaneously, and each must be addressed before the DEQN can be trained at all.

**Time-varying state distributions with no ergodic limit.** The endogenous state of a stationary DSGE is the projection of a recurrent Markov chain onto a finite-dimensional vector; the policy function lives on its stationary distribution. In an IAM the analogue object does not exist within the planning horizon. Atmospheric carbon $M^{\mathrm{AT}}_t$ rises from a pre-industrial baseline of $\sim 600$ GtC to a peak of $\sim 1500$ GtC over a century, then decays over millennia; atmospheric temperature $T^{\mathrm{AT}}_t$ follows with a multi-decade lag and a multi-century relaxation. Within the 300 years the planner cares about, neither variable ever returns to a state it has been in before. The state visited at $t = 100$ is therefore not exchangeable with the state visited at $t = 200$, and a time-invariant policy function $\bm p(\bm X_t)$ that depends only on the endogenous state misses the whole point of the exercise: the optimal mitigation effort at a given $(M^{\mathrm{AT}}, T^{\mathrm{AT}})$ depends on whether that state was reached on the way up or on the way down. Cf. the curse-of-dimensionality discussion in {ref}`sec-curse_of_dim`: it is not the size of the state space that breaks the DEQN here, it is the lack of recurrence.

**Deterministic drift through exogenous trends.** Even setting the carbon and temperature stocks aside, the IAM is drifting deterministically. Total factor productivity $A_t$ trends up at a calibrated, time-varying rate; population $L_t$ follows the demographic projection of {cite:t}`nordhausRevisitingSocialCost2017`; carbon intensity $\sigma_t$ falls along the closed-form decay {eq}`eq-sigma_decay`; land-use emissions $E_{\mathrm{Land},t}$ decay smoothly {eq}`eq-land_emissions`; the backstop price $p^{\mathrm{back}}_t$ falls {eq}`eq-backstop_price`; the exogenous non-CO$_2$ forcing $F^{\mathrm{EX}}_t$ follows a fitted RCP trajectory; and the abatement-cost level $\theta_{1,t}$ inherits the time dependence of $\sigma_t$ and $p^{\mathrm{back}}_t$ through {eq}`eq-theta1_calibration`. Seven exogenous trends drive the model even before a shock is introduced. A time-invariant policy can never see them, and replacing them with their long-run averages is exactly the certainty-equivalence move that defeats the purpose of solving the model globally.

**Finite calendar-time horizon.** A stationary DEQN trains under a transversality condition: as $t \to \infty$, the discounted shadow price of capital goes to zero, and the iterative-projection loss inherits that fixed-point structure for free. An IAM is not solved on $[0, \infty)$. The planning horizon is a finite calendar date (the notebooks of {ref}`sec-iam_dequ_loss` run roughly three centuries from a 2015 start), so transversality is not available and the policy is solved over a finite forward sweep instead.

**Putting it together.** These features compound and explain why a time-invariant DEQN of {ref}`ch-deqn` cannot be used here without modification. The next two sections operationalize the response: {ref}`sec-nsdeqn_setup` reorganizes the network inputs to include calendar time, and {ref}`sec-nsdeqn_algo` states the resulting training algorithm as a labeled diff against the stationary DEQN box of {ref}`sec-deqn_algo`.

(sec-nsdeqn_setup)=
## What Changes in the DEQN Setup
We now translate this into one concrete design choice for the network inputs. The autodiff machinery, the squared-residual structure, and the rest of the training loop of {ref}`ch-deqn` carry over unchanged; this is a refactor of what the network sees, not a new solver.

(sec-nsdeqn_setup_time)=
### Time and trends as states
Calendar time itself enters as a state. Because neural networks prefer bounded inputs, we use the monotone time rescaling $\tau_t = 1 - \exp(-\vartheta\, t) \in [0, 1)$ of Eq. {eq}`eq-time_transform`. Every exogenous trend ($A_t, L_t, \sigma_t, E_{\mathrm{Land},t}, F^{\mathrm{EX}}_t, p^{\mathrm{back}}_t, \theta_{1,t}$) is then a deterministic function of $\tau_t$, so passing $\tau_t$ to the network is informationally equivalent to passing the entire trend bundle. Training trajectories begin from the calibrated 2015 state and run forward over the planner's horizon.

(sec-nsdeqn_algo)=
## The Non-Stationary DEQN Algorithm
The design choice of {ref}`sec-nsdeqn_setup` translates into a single training algorithm. The body below is a literal diff against the stationary DEQN of {ref}`sec-deqn_algo`: unchanged lines are grayed, new or modified lines are bolded.

```{prf:algorithm} Non-Stationary DEQN Training
:label: alg-nsdeqn

- **Input:** Network $\mathcal{N}_\rho$, learning rate $\eta$, episodes $E$, training steps $T_{\mathrm{train}}$; \\ **[NEW]** calibrated initial state $\bm x_0$ (e.g., the 2015 state) and a planning horizon $T_{\max}$
- **for** episode $e = 1, \ldots, E$ **do**
  - **[CHANGED] Simulate $K$ forward trajectories from $\bm x_0$ over $[0, T_{\max}]$ under the current policy, and collect the time-stamped states $(\tau_t, \bm x_t)$ into $\mathcal D$**
  - **for** gradient step $t = 1, \ldots, T_{\mathrm{train}}$ **do**
    - Draw mini-batch $\mathcal B \subset \mathcal D$
    - Compute loss:~$\ell_\rho = \frac{1}{|\mathcal B|}\sum_{\bm x_i \in \mathcal B}\|G(\bm x_i, \mathcal N_\rho(\bm x_i))\|^2$
    - Update:~$\rho \leftarrow \rho - \eta \cdot \nabla_\rho \ell_\rho$
  - **end**
- **end**
- **Output:** Trained network $\mathcal{N}_{\rho^\star}$ approximating the policy function
```

One delta against the stationary DEQN box. The simulation step starts from a calibrated initial state $\bm x_0$ and integrates $K$ trajectories forward through calendar time, so the pool $\mathcal D$ contains time-stamped states $(\tau_t, \bm x_t)$ along finite-horizon trajectories rather than draws from an ergodic distribution. With $\tau_t$ in the input the network learns a time-dependent policy; every other line of the box is the stationary DEQN of {ref}`sec-deqn_algo` unchanged.

**What replaces transversality.** Because the pool $\mathcal D$ is built from $K$ forward simulations of length $T_{\max}$ that all start at the same $\bm x_0$, every trajectory visits the full calendar window $[0, T_{\max}]$ and a uniform mini-batch draw from $\mathcal D$ is therefore stratified across calendar time by construction. The missing transversality condition of {ref}`sec-iam_nonstationarity` is absorbed numerically by choosing the horizon long enough that the discounted contribution of the terminal state falls below the training-noise floor: at the CDICE calibration $\rho = 0.015$/yr and the notebooks' default $T_{\max} = 300$ years, $\hat\beta_t^{\,T_{\max}} \approx \exp(-\rho\,T_{\max}) \approx 0.011$, which is one to two orders of magnitude below the achievable residual root-mean-square at convergence. When the horizon must be short (e.g., the 1D toy of {prf:ref}`ex-ch11-10`), one instead adds an explicit terminal residual $\lambda_T\,\|\bm x_{T_{\max}} - \bm x^{\mathrm{ref}}_{T_{\max}}\|^2$ to the loss; both options are standard in the finite-horizon DEQN literature.

```{prf:remark} The non-stationary DEQN in one line

 One modification, one solver: time enters as a state. The autodiff backbone, the residual structure, the sampling rule, and the update rule carry over from the stationary DEQN of {ref}`ch-deqn`.
```


(sec-dice_deqn)=
## The Planner's Lagrangian and FOCs

Movement 2 puts the non-stationary DEQN of {ref}`sec-nsdeqn_algo` to work on the deterministic CDICE economy of {ref}`sec-dice`. Solving this system with the algorithm of {ref}`sec-nsdeqn_algo` amounts to writing the planner's Lagrangian, deriving the first-order and envelope conditions, normalizing them, treating each FOC as a residual, and minimizing the sum of squared residuals on the time-stamped state pool generated by the forward simulation of {ref}`sec-nsdeqn_algo`. This section follows {cite:t}`friedlDeep2023` and Online Appendix D of {cite:t}`Folini_2021`.

**Detrending and state vector, in compact form.** The model-rendering choices already named in {ref}`sec-nsdeqn_setup_time` carry over verbatim. Variables that grow with the productivity–population product $A_t L_t$ are rescaled to per-effective-capita units:

$$
c_t \;:=\; \frac{C_t}{A_t\,L_t}, \qquad
k_t \;:=\; \frac{K_t}{A_t\,L_t}.
$$ (eq-iam_detrend)

Calendar time enters through the bounded rescaling (compatible with the dynamic-programming convention of {cite:t}`traeger4StatedDICEQuantitatively2014`),

$$
\tau \;=\; 1 - \exp(-\vartheta\, t) \;\in\; [0,1),
\qquad\text{with inverse}\quad
t \;=\; -\frac{\ln(1-\tau)}{\vartheta},
$$ (eq-time_transform)

with compression parameter $\vartheta > 0$. The full DEQN state vector then collects the detrended endogenous CDICE states, the bounded time index, the Bayesian-belief states $(\mu_{f,t}, S_{f,t})$ used in {ref}`sec-bayesian_learning`, and a slot for pseudo-state parameters $\theta$ used in the UQ analysis of {ref}`sec-deep_uq`:

$$
\bm{X}_t \;=\; \bigl[\underbrace{k_t,\, M^{\mathrm{AT}}_t,\, M^{\mathrm{UO}}_t,\, M^{\mathrm{LO}}_t,\, T^{\mathrm{AT}}_t,\, T^{\mathrm{OC}}_t,\, \mu_{f,t},\, S_{f,t},\, \tau_t}_{9\text{ endogenous, exogenous, and time states}};\; \underbrace{\theta}_{N\text{ pseudo-state parameters}}\bigr].
$$ (eq-iam_state)

In the deterministic core developed in this and the next section, only the six endogenous-state entries plus $\tau_t$ are active, i.e. a seven-dimensional input vector; $(\mu_{f,t}, S_{f,t})$ and $\theta$ are appended only in the extensions of Movement 3.

**The Lagrangian.** We now derive the equilibrium conditions that the DEQN will be trained against. The derivation follows the standard Lagrangian approach in CRRA-IES form, working directly with the deterministic CDICE primitives of {ref}`sec-dice` (the recursive Epstein–Zin refinement is layered on in {ref}`sec-ez_layer`). Write the Lagrangian with multiplier $\lambda_t$ for the budget constraint $C_t + I_t = Y^{\mathrm{net}}_t$, multipliers $\nu^{\mathrm{AT}}_t,\nu^{\mathrm{UO}}_t,\nu^{\mathrm{LO}}_t$ for the three carbon-reservoir transitions {eq}`eq-carbon_cycle`, multipliers $\eta^{\mathrm{AT}}_t,\eta^{\mathrm{OC}}_t$ for the temperature dynamics {eq}`eq-temp_at`–{eq}`eq-temp_oc`, and KKT multiplier $\lambda^\mu_t \ge 0$ for the abatement bound $\mu_t \le 1$. The derivation produces ten equilibrium conditions: a consumption FOC, an abatement FOC, the capital Euler equation, the budget/resource constraint, three carbon-stock envelope conditions, two temperature-envelope conditions, and the abatement upper-bound complementarity. Two of these are enforced algebraically (the static consumption and abatement FOCs); the remaining eight become the DEQN residuals of {ref}`sec-iam_dequ_loss`.

**Qualitative overview.** Taking derivatives of the Lagrangian with respect to the controls yields:

- **w.r.t. $C_t$:** the marginal utility of consumption equals the shadow price of the budget, $\partial V^{1-1/\psi}/\partial C_t = \lambda_t$.

- **w.r.t. $K_{t+1}$:** the shadow value of capital today equals the discounted expected marginal value tomorrow, $\xi_t = e^{-\rho}\,\partial \mathbb{E}_t[V_{t+1}^{1-\gamma}]^{(1-1/\psi)/(1-\gamma)} / \partial K_{t+1}$.

- **w.r.t. $\mu_t$:** the marginal abatement cost equals the shadow value of reduced emissions (plus the complementarity term if $\mu_t = 1$).

**Envelope theorem.** Since the FOC for $K_{t+1}$ involves $\partial V/\partial K_{t+1}$, which cannot be computed analytically, we apply the envelope theorem. It provides derivatives of the value function with respect to *current* states, in particular $\partial V/\partial k_t$, $\partial V/\partial M_{\mathrm{AT},t}$, $\partial V/\partial T^{\mathrm{AT}}_t$, which are then shifted forward one period and substituted back into the FOCs.

**Capital Euler equation.** Combining the FOCs and envelope conditions yields:

$$
1 = e^{-\rho}\,\mathbb{E}_t\!\left[\left(\frac{V_{t+1}}{\bigl(\mathbb{E}_t[V_{t+1}^{1-\gamma}]\bigr)^{1/(1-\gamma)}}\right)^{1/\psi - \gamma} \cdot \frac{(C_{t+1}/L_{t+1})^{-1/\psi}}{(C_t/L_t)^{-1/\psi}} \cdot R^K_{t+1}\right],
$$ (eq-iam_euler)

where $R^K_{t+1}$ is the return on capital inclusive of climate damages. The SCC also appears through the shadow price of atmospheric carbon:

$$
\mathrm{SCC}^{M}_t = -\frac{\partial V_t / \partial M_{\mathrm{AT},t}}{\partial V_t / \partial C_t}.
$$

This is a shadow value per unit of atmospheric carbon stock. The emissions-based SCC in {eq}`eq-scc` additionally includes the marginal loading of a unit of emissions into $M_{\mathrm{AT},t}$ and the carbon-to-CO$_2$ unit conversion. At the optimum, the marginal abatement cost equals the carbon tax equals the emissions SCC after these conversions.

**Normalization of multipliers.** Over a 300-year horizon, $A_t$ and $L_t$ can move the natural scale of marginal utilities and multipliers substantially, with the direction and magnitude depending on the IES through $A_t^{1-1/\psi}L_t$. Such scale drift makes network outputs and gradients harder to optimize stably. Following the detrending logic of {eq}`eq-iam_detrend`, all multipliers, the budget multiplier, the abatement-bound multiplier, and the five climate envelope multipliers alike, are divided by $A_t^{1-1/\psi}\,L_t$. The argument for the climate multipliers tracks the budget-multiplier case via the envelope conditions of {ref}`sec-dice_deqn` and is spelled out in Online Appendix D of {cite:t}`Folini_2021`; we adopt the result here:

$$
\hat{\lambda}_t := \frac{\lambda_t}{A_t^{1-1/\psi}\,L_t},\quad
\hat{\lambda}^\mu_t := \frac{\lambda^\mu_t}{A_t^{1-1/\psi}\,L_t},\quad
\hat{\nu}^{\mathrm{AT}}_t := \frac{\nu^{\mathrm{AT}}_t}{A_t^{1-1/\psi}\,L_t},\quad
\hat{\nu}^{\mathrm{UO}}_t := \frac{\nu^{\mathrm{UO}}_t}{A_t^{1-1/\psi}\,L_t},\quad \ldots
$$ (eq-iam_norm_mult)

and analogously for the remaining multipliers $\hat{\nu}^{\mathrm{LO}}_t$, $\hat{\eta}^{\mathrm{AT}}_t$, and $\hat{\eta}^{\mathrm{OC}}_t$. The normalization induces an *effective discount factor* that absorbs the trend growth in the per-effective-capita Euler equation,

$$
\hat{\beta}_t \;:=\; \exp\!\left(-\rho + \left(1-\frac{1}{\psi}\right) g^A_t + g^L_t\right),
$$ (eq-iam_eff_beta)

where $g^A_t := \ln(A_{t+1}/A_t)$ and $g^L_t := \ln(L_{t+1}/L_t)$ are annual log changes. Equation {eq}`eq-iam_eff_beta` mirrors Equation (38) of Online Appendix D of {cite:t}`Folini_2021`: the population term enters linearly because $L_t$ enters the felicity weight $L_t (C_t/L_t)^{1-1/\psi}$ linearly, while the productivity term inherits the $1-1/\psi$ exponent from the per-effective-capita rescaling of consumption. All intertemporal equations below use $\hat{\beta}_t$ in place of $e^{-\rho}$. For a non-annual time step, replace $\rho$ by $\rho\Delta_t$ and $g^A_t, g^L_t$ by their per-period analogues.

**Sign convention for the climate multipliers.** We adopt the value-derivative convention throughout the script: each climate multiplier $\hat{\nu}^{\mathrm{AT}}_t,\, \hat{\nu}^{\mathrm{UO}}_t,\, \hat{\nu}^{\mathrm{LO}}_t,\, \hat{\eta}^{\mathrm{AT}}_t,\, \hat{\eta}^{\mathrm{OC}}_t$ is the (normalized) partial derivative of the value function with respect to the corresponding climate state. Because extra atmospheric carbon lowers welfare, $\hat{\nu}^{\mathrm{AT}}_t$ is non-positive at the optimum, which is why the stock SCC carries a minus sign, $\mathrm{SCC}^M_t = -\hat{\nu}^{\mathrm{AT}}_t/\hat{\lambda}_t$. The companion implementation in `dice_2p_surrogate_lib.py` stores the *positive marginal damage* $-\hat{\nu}^{\mathrm{AT}}_t$ as a network output for numerical conditioning and flips the sign explicitly inside each residual; the algebra below uses the script convention, so the reader who compares the equations to the code will see one extra sign flip per carbon-multiplier term.

**Symbol cheat-sheet for the multipliers.** Before writing the FOCs and the loss, {numref}`tab-cdice_multipliers` collects the multipliers that the DEQN learns and their role; subsequent equations use the hat-normalized form throughout.

````{table}
:name: tab-cdice_multipliers

Normalized Lagrange multipliers in the CDICE–DEQN. All values are divided by $A_t^{1-1/\psi}\,L_t$ relative to the raw multipliers, so the hatted versions inherit the per-effective-capita scale that the network outputs see. The atmospheric carbon multiplier carries the SCC up to the marginal-utility denominator: $\mathrm{SCC}^M_t = -\hat\nu^{\mathrm{AT}}_t / \hat\lambda_t$.

| **Symbol** | **Multiplier on** | **Sign at optimum** | **Network output?** |
|---|---|:---:|---|
| \_t | Budget constraint $C_t + I_t = Y^{\mathrm{net}}_t$ | $> 0$ | yes (softplus) |
| \^\_t | Abatement upper bound $\mu_t \le 1$ | $\ge 0$ | no (implied, Eq. {eq}`eq-iam_lambdamu_implied`) |
| \^\_t | Atmospheric carbon transition $M^{\mathrm{AT}}_{t+1}=\ldots$ | $\le 0$ | yes (stored as $-\hat\nu^{\mathrm{AT}}_t > 0$ via softplus) |
| \^\_t | Upper-ocean carbon transition | $\le 0$ | yes (linear) |
| \^\_t | Lower-ocean carbon transition | $\le 0$ | yes (linear) |
| \^\_t | Atmospheric temperature transition | $\le 0$ | yes (linear) |
| \^\_t | Ocean temperature transition | $\le 0$ | yes (linear) |
````

**Key FOCs in normalized form.** After normalization, the most important first-order conditions become (see Online Appendix D of {cite:t}`Folini_2021` for the complete set of 14 equations):

$$
\frac{\partial \mathcal{L}}{\partial c_t} = 0 \; \Leftrightarrow\;
c_t^{-1/\psi}\,A_t^{1-1/\psi}\,L_t - \hat{\lambda}_t = 0
$$ (eq-iam_foc_c)

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial k_{t+1}} = 0 \;&\Leftrightarrow\;
\exp\!\bigl(g^A_t + g^L_t\bigr)\,\hat{\lambda}_t - \hat{\beta}_t\Bigl\{\hat{\lambda}_{t+1}\bigl[\bigl(1-\Omega(T_{\mathrm{AT},t+1}) - \Theta(\mu_{t+1})\bigr)\alpha k_{t+1}^{\alpha-1} + (1-\delta)\bigr] \\
&\quad + \hat{\nu}^{\mathrm{AT}}_{t+1}\,\sigma_{t+1}(1-\mu_{t+1})A_{t+1}L_{t+1}\alpha k_{t+1}^{\alpha-1}\Bigr\} = 0,
\end{aligned}
$$ (eq-iam_foc_k)

$$
\frac{\partial \mathcal{L}}{\partial \mu_t} = 0 \; \Leftrightarrow\;
\hat{\lambda}_t\,\Theta'(\mu_t)\,k_t^\alpha + \hat{\lambda}^\mu_t + \hat{\nu}^{\mathrm{AT}}_t\,\sigma_t\,A_t\,L_t\,k_t^\alpha = 0.
$$ (eq-iam_foc_mu)

Equation {eq}`eq-iam_foc_k` is the capital Euler equation: it equates the marginal cost of saving one additional unit today (left) to the discounted marginal benefit tomorrow (right), which now includes a term from the atmospheric carbon envelope ($\hat{\nu}^{\mathrm{AT}}_{t+1}$) because higher capital increases output and hence emissions.

**Envelope conditions.** *Convention reminder.* As stated in {ref}`sec-iam_landscape`, CDICE is calibrated on an annual time step and the coefficients $b_{12}, b_{23}, c_1, c_3, c_4$ in {numref}`tab-dice_calibration` are annual rates; consequently no $\Delta_t$ multipliers appear in either the dynamics {eq}`eq-carbon_cycle`, {eq}`eq-temp_at`–{eq}`eq-temp_oc` or in the FOC residuals below.

Differentiating the Lagrangian with respect to *state* variables and shifting forward one period yields the shadow prices of the climate stocks. For example, the atmospheric carbon envelope is:

$$
\frac{\partial \mathcal{L}}{\partial M_{\mathrm{AT},t+1}} = 0 \;\Leftrightarrow\;
\hat{\nu}^{\mathrm{AT}}_t - \hat{\beta}_t\!\left[\hat{\nu}^{\mathrm{AT}}_{t+1}(1-b_{12}) + \hat{\nu}^{\mathrm{UO}}_{t+1}\,b_{12} + \hat{\eta}^{\mathrm{AT}}_{t+1}\,c_1\,F_{\mathrm{2\times CO_2}}\,\frac{1}{\ln 2\,M_{\mathrm{AT},t+1}}\right] = 0.
$$ (eq-iam_env_mat)

This equation says that the current shadow price of atmospheric carbon ($\hat{\nu}^{\mathrm{AT}}_t$) must equal the discounted future effects through three channels: persistence in the atmosphere ($b_{12}$ term), diffusion into the upper ocean ($\hat{\nu}^{\mathrm{UO}}_{t+1}$ term), and radiative forcing on temperature ($\hat{\eta}^{\mathrm{AT}}_{t+1}$ term). It is the existence of these climate multipliers that distinguishes the IAM from the purely economic models of {ref}`ch-deqn`–{ref}`ch-nas`.

**Fischer–Burmeister complementarity for abatement.** The abatement rate is bounded above by 1 (full abatement), giving the KKT condition:

$$
1 - \mu_t \;\geq\; 0 \quad\perp\quad \hat{\lambda}^\mu_t \;\geq\; 0,
$$ (eq-iam_kkt_mu)

which is non-smooth at $\mu_t = 1$. As in the borrowing-constraint treatment of {ref}`ch-olg` ({ref}`sec-olg_fb`), we replace it with the Fischer–Burmeister smooth approximation:

$$
\Psi^{\mathrm{FB}}\!\bigl(\hat{\lambda}^\mu_t,\; 1-\mu_t\bigr)
\;=\; \hat{\lambda}^\mu_t + (1-\mu_t) - \sqrt{(\hat{\lambda}^\mu_t)^2 + (1-\mu_t)^2 + \varepsilon_{\mathrm{FB}}} \;=\; 0,
$$ (eq-iam_fb)

with the same regularization parameter $\varepsilon_{\mathrm{FB}} \geq 0$ used in {ref}`ch-irbc`–{ref}`ch-olg`. In CDICE-DEQN we take $\varepsilon_{\mathrm{FB}} = 10^{-6}$, equivalent to the IRBC chapter's $\varepsilon = 10^{-3}$ under its $\varepsilon^2$ convention; the trained policy is insensitive to the choice in the range $10^{-10}$ to $10^{-4}$. At $\varepsilon_{\mathrm{FB}} = 0$ the zero set of $\Psi^{\mathrm{FB}}$ coincides with the positive axes in the $(\hat{\lambda}^\mu_t,\, 1-\mu_t)$-plane, enforcing the original complementarity exactly but the function is non-differentiable at the origin; with $\varepsilon_{\mathrm{FB}} > 0$ the function is differentiable everywhere at the cost of a slight relaxation of exact complementarity.

(sec-iam_dequ_loss)=
## From FOCs to a Single Loss
The Lagrangian of {ref}`sec-dice_deqn` produces ten equilibrium conditions: the consumption FOC {eq}`eq-iam_foc_c`, the capital Euler {eq}`eq-iam_foc_k`, the abatement FOC {eq}`eq-iam_foc_mu`, the budget/resource constraint $C_t + I_t = Y^{\mathrm{net}}_t$, the three carbon-stock envelopes (one of which is the atmospheric-carbon envelope {eq}`eq-iam_env_mat`), the two temperature-layer envelopes, and the Fischer–Burmeister abatement complementarity {eq}`eq-iam_fb`. In the DEQN solver two of these ten are enforced *exactly* by algebraic recovery rather than as squared residuals: the consumption FOC is inverted to yield $c_t$ from $\hat{\lambda}_t$, and the abatement FOC is solved for $\hat{\lambda}^\mu_t$ and the resulting *implied multiplier* is fed straight into the Fischer–Burmeister condition. What remains is an eight-residual sum-of-squares loss with eight network outputs, structurally identical to the stationary DEQN of {ref}`ch-deqn`–{ref}`ch-irbc`. The only substantive difference is that the network must learn the shadow prices of all five climate state variables (three carbon stocks and two temperature layers) in addition to the economic choices, so that the planner has a gradient signal for how today's decisions propagate through the carbon cycle and the energy balance into future damages.

**Policy network specification.** The policy function approximated by the neural network outputs an eight-dimensional vector,

$$
\mathcal{N}_\rho(\bm{x}_t) \;\in\; \mathbb{R}^{8}
\;:=\; \bigl(k_{t+1},\; \mu_t,\; \hat{\lambda}_t,\; \hat{\nu}^{\mathrm{AT}}_t,\; \hat{\nu}^{\mathrm{UO}}_t,\; \hat{\nu}^{\mathrm{LO}}_t,\; \hat{\eta}^{\mathrm{AT}}_t,\; \hat{\eta}^{\mathrm{OC}}_t\bigr),
$$ (eq-iam_nn_output)

comprising two choice variables ($k_{t+1}$, $\mu_t$), the consumption shadow price $\hat{\lambda}_t$, and the five normalized climate multipliers. Note that the abatement KKT multiplier $\hat{\lambda}^\mu_t$ is *not* a network output: it is recovered algebraically inside the loss (see below). A key difference from the stationary DEQN of {ref}`ch-deqn`–{ref}`ch-irbc` is that the network must learn the shadow prices of all climate constraints, not just the economic choices. Without the climate multipliers, the planner would have no gradient signal about how today's decisions propagate through the carbon cycle and temperature dynamics into future damages.

**Bounds and positivity.** The output activations of $\mathcal{N}_\rho$ are chosen so that the bound and positivity constraints of the model hold for every input, eliminating the need for additional residuals. The capital level $k_{t+1}$, the consumption shadow $\hat{\lambda}_t$, and the abatement rate $\mu_t$ are each passed through a softplus, which guarantees $k_{t+1} > 0$, $\hat{\lambda}_t > 0$ (so consumption recovered via {eq}`eq-iam_c_recovery` is positive), and $\mu_t \ge 0$ exactly. The upper bound $\mu_t \le 1$ is enforced jointly by the Fischer–Burmeister condition $l_8$ at the implied multiplier {eq}`eq-iam_lambdamu_implied` and by a small quadratic upper-bound penalty $\propto \mathbb{E}[\max(\mu_t - 1, 0)^2]$ added to the training loss. The atmospheric-carbon shadow $\hat{\nu}^{\mathrm{AT}}_t$ is stored in the implementation as a positive marginal damage (see the sign-convention note in {ref}`sec-dice_deqn`) and is output through a softplus; the remaining climate multipliers $\hat{\nu}^{\mathrm{UO}}_t, \hat{\nu}^{\mathrm{LO}}_t, \hat{\eta}^{\mathrm{AT}}_t, \hat{\eta}^{\mathrm{OC}}_t$ are unconstrained and use linear output activations.

**How is consumption $c_t$ determined?** The consumption FOC {eq}`eq-iam_foc_c` is enforced exactly by inversion rather than as a residual: given the network's prediction of $\hat{\lambda}_t$, consumption is recovered algebraically as

$$
c_t \;=\; \bigl(\hat{\lambda}_t \cdot A_t^{1/\psi - 1}\,L_t^{-1}\bigr)^{-\psi},
$$ (eq-iam_c_recovery)

so $c_t$ is not itself a network output. Positivity of $c_t$ is guaranteed because the implementation passes $\hat{\lambda}_t$ through a softplus activation, so $\hat{\lambda}_t > 0$ for every input.

**How is the abatement multiplier $\hat{\lambda}^\mu_t$ determined?** The same trick handles the abatement FOC {eq}`eq-iam_foc_mu`: rather than have the network output $\hat{\lambda}^\mu_t$ and impose the FOC as a separate residual, we *solve* the FOC for $\hat{\lambda}^\mu_t$ and treat the resulting implied multiplier as a deterministic function of the other network outputs. Setting $\partial\mathcal{L}/\partial\mu_t = 0$ in {eq}`eq-iam_foc_mu` yields

$$
\hat{\lambda}^{\mu,\mathrm{impl}}_t \;=\; -\hat{\lambda}_t\,\Theta'(\mu_t)\,k_t^{\alpha} \;-\; \hat{\nu}^{\mathrm{AT}}_t\,\sigma_t\, A_t\, L_t\,k_t^{\alpha}.
$$ (eq-iam_lambdamu_implied)

Plugged into the Fischer–Burmeister condition {eq}`eq-iam_fb`, this is the residual $l_8$ below. Two facts come for free. First, whenever $l_8 = 0$ holds and the smoothing parameter $\varepsilon_{\mathrm{FB}}$ is small, the abatement FOC *also* holds automatically, because $l_8$ couples the implied multiplier to the slack $1-\mu_t$. Second, the network output dimension drops from nine to eight, which improves training stability: the network no longer has to discover that $\hat{\lambda}^\mu_t$ is exactly the right algebraic combination of $\hat{\lambda}_t,\, \mu_t,\, \hat{\nu}^{\mathrm{AT}}_t$.

The network architecture uses two hidden layers with 1024 units each, SELU activation, and the Adam optimizer with learning rate $10^{-5}$. Training alternates between broad sampling (Phase 1) and endogenous simulation (Phase 2), as described in {ref}`ch-irbc`.

**The 8 loss components.** Each remaining equilibrium condition from {ref}`sec-dice_deqn` becomes a residual $l_m = 0$, and the network is asked to drive every $l_m$ to zero simultaneously along simulated paths. The mapping is one-for-one: $l_1$ is the capital-Euler FOC {eq}`eq-iam_foc_k`; $l_2$ is the budget constraint that closes {eq}`eq-capital_accumulation`; $l_3$, $l_4$, $l_5$ are the three carbon-reservoir envelope conditions, of which $l_3$ is {eq}`eq-iam_env_mat`; $l_6$ and $l_7$ are the two temperature-layer envelopes; and $l_8$ is the Fischer–Burmeister smoothing {eq}`eq-iam_fb` of the KKT slack on $\mu_t \le 1$, evaluated at the implied multiplier {eq}`eq-iam_lambdamu_implied`. The consumption FOC {eq}`eq-iam_foc_c` and the abatement FOC {eq}`eq-iam_foc_mu` are enforced *exactly* via the inversions in {eq}`eq-iam_c_recovery` and {eq}`eq-iam_lambdamu_implied`, which is why the loss list contains eight entries instead of nine. Written out, the eight components are:

```{math}
:label: eq-iam_l1
:enumerator: capital Euler

\begin{aligned}
l_1 &:= \exp\!\bigl(g^A_t + g^L_t\bigr)\,\hat{\lambda}_t - \hat{\beta}_t\Bigl\{\hat{\lambda}_{t+1}\bigl[\bigl(1-\Omega(T_{\mathrm{AT},t+1}) - \Theta(\mu_{t+1})\bigr)\alpha k_{t+1}^{\alpha-1} + (1-\delta)\bigr] \\
&\quad + \hat{\nu}^{\mathrm{AT}}_{t+1}\,\sigma_{t+1}(1-\mu_{t+1})A_{t+1}L_{t+1}\alpha k_{t+1}^{\alpha-1}\Bigr\}
\end{aligned}
```

```{math}
:label: eq-iam_l2
:enumerator: budget

l_2 := \bigl(1-\Omega(T_{\mathrm{AT},t}) - \Theta(\mu_t)\bigr)\,k_t^\alpha + (1-\delta)\,k_t - c_t - \exp\!\bigl(g^A_t + g^L_t\bigr)\,k_{t+1}
```

```{math}
:label: eq-iam_l3
:enumerator: atm. carbon

l_3 := \hat{\nu}^{\mathrm{AT}}_t - \hat{\beta}_t\!\left[\hat{\nu}^{\mathrm{AT}}_{t+1}(1-b_{12}) + \hat{\nu}^{\mathrm{UO}}_{t+1}\,b_{12} + \hat{\eta}^{\mathrm{AT}}_{t+1}\,c_1\,F_{\mathrm{2\times CO_2}}\,\tfrac{1}{\ln 2\,M_{\mathrm{AT},t+1}}\right]
```

```{math}
:label: eq-iam_l4
:enumerator: upper ocean C

l_4 := \hat{\nu}^{\mathrm{UO}}_t - \hat{\beta}_t\!\Bigl[\hat{\nu}^{\mathrm{AT}}_{t+1}\,b_{12}\,\tfrac{M^{\mathrm{AT}}_{\mathrm{EQ}}}{M^{\mathrm{UO}}_{\mathrm{EQ}}} + \hat{\nu}^{\mathrm{UO}}_{t+1}\!\Bigl(1-b_{12}\tfrac{M^{\mathrm{AT}}_{\mathrm{EQ}}}{M^{\mathrm{UO}}_{\mathrm{EQ}}}-b_{23}\Bigr) + \hat{\nu}^{\mathrm{LO}}_{t+1}\,b_{23}\Bigr]
```

```{math}
:label: eq-iam_l5
:enumerator: lower ocean C

l_5 := \hat{\nu}^{\mathrm{LO}}_t - \hat{\beta}_t\!\Bigl[\hat{\nu}^{\mathrm{UO}}_{t+1}\,b_{23}\,\tfrac{M^{\mathrm{UO}}_{\mathrm{EQ}}}{M^{\mathrm{LO}}_{\mathrm{EQ}}} + \hat{\nu}^{\mathrm{LO}}_{t+1}\!\Bigl(1-b_{23}\,\tfrac{M^{\mathrm{UO}}_{\mathrm{EQ}}}{M^{\mathrm{LO}}_{\mathrm{EQ}}}\Bigr)\Bigr]
```

```{math}
:label: eq-iam_l6
:enumerator: atm. temp.

l_6 := \hat{\eta}^{\mathrm{AT}}_t - \hat{\beta}_t\!\Bigl[-\hat{\lambda}_{t+1}\,\Omega'(T_{\mathrm{AT},t+1})\,k_{t+1}^\alpha + \hat{\eta}^{\mathrm{AT}}_{t+1}\!\Bigl(1-c_1\,\tfrac{F_{\mathrm{2\times CO_2}}}{\Delta T_{\mathrm{AT},\times 2}}-c_1 c_3\Bigr) + \hat{\eta}^{\mathrm{OC}}_{t+1}\,c_4\Bigr]
```

```{math}
:label: eq-iam_l7
:enumerator: ocean temp.

l_7 := \hat{\eta}^{\mathrm{OC}}_t - \hat{\beta}_t\!\left[\hat{\eta}^{\mathrm{AT}}_{t+1}\,c_1 c_3 + \hat{\eta}^{\mathrm{OC}}_{t+1}(1-c_4)\right]
```

```{math}
:label: eq-iam_l8
:enumerator: Fischer–Burmeister, implied multiplier

l_8 := \hat{\lambda}^{\mu,\mathrm{impl}}_t + (1-\mu_t) - \sqrt{(\hat{\lambda}^{\mu,\mathrm{impl}}_t)^2 + (1-\mu_t)^2 + \varepsilon_{\mathrm{FB}}}
```

Loss components $l_1$–$l_2$ enforce intertemporal optimality and feasibility, $l_3$–$l_7$ are the envelope conditions that price the five climate state variables, and $l_8$ jointly enforces the abatement FOC (via the implied multiplier) and the upper-bound complementarity $\mu_t \le 1$.

**Total loss.** The DEQN loss aggregates all residuals along a simulated path:

$$
\ell_\rho \;:=\; \frac{1}{N_{\text{path}}} \sum_{\bm{x}_t\,\text{on sim.\ path}} \;\sum_{m=1}^{8}\; \bigl(l_m(\bm{x}_t,\, \mathcal{N}_\rho(\bm{x}_t))\bigr)^2.
$$ (eq-iam_loss)

This is the same sum-of-squared-residuals structure as the $N$-country IRBC model of {ref}`ch-irbc`, but with 8 equations per time step instead of the IRBC's $2N+1$ ($N$ Euler equations, $N$ Fischer–Burmeister conditions, and one aggregate resource constraint).

**State evolution.** To evaluate the loss along a simulated path, the full state vector {eq}`eq-iam_state` must be propagated forward. In CDICE the next-period state is:

$$
\bm{x}_{t+1} = \bigl(k_{t+1},\; M^{\mathrm{AT}}_{t+1},\; M^{\mathrm{UO}}_{t+1},\; M^{\mathrm{LO}}_{t+1},\; T^{\mathrm{AT}}_{t+1},\; T^{\mathrm{OC}}_{t+1},\; \mu_{f,t+1},\; S_{f,t+1},\; \tau_{t+1};\; \theta\bigr)^T,
$$ (eq-iam_state_evol)

where:

- $k_{t+1}$ comes from the network output {eq}`eq-iam_nn_output` (choice variable);

- $M^{\mathrm{AT}}_{t+1}$, $M^{\mathrm{UO}}_{t+1}$, $M^{\mathrm{LO}}_{t+1}$, $T^{\mathrm{AT}}_{t+1}$, $T^{\mathrm{OC}}_{t+1}$ are computed from the transition equations of {ref}`sec-dice` (carbon cycle and temperature dynamics);

- the Bayesian belief states $\mu_{f,t+1}$ and $S_{f,t+1}$ are updated via the conjugate posterior {eq}`eq-bayes_mean`–{eq}`eq-bayes_var` once the period-$t$ temperature observation is realized;

- the bounded time index advances as $\tau_{t+1} = 1 - \exp\!\bigl(-\vartheta\,(t+\Delta_t)\bigr)$, the image of the calendar increment $t \mapsto t+\Delta_t$ under the time-rescaling {eq}`eq-time_transform`;

- the pseudo-state parameters $\theta$ are held fixed within an episode and re-sampled across episodes ({ref}`sec-deep_uq`).

All deterministic transitions are differentiable; stochastic shock draws are handled via reparameterization / common random numbers, so the simulate-then-backpropagate loop can be executed end-to-end with automatic differentiation.

```{prf:remark} Recipe: mapping a non-stationary model onto the DEQN framework

1.  Detrend variables that grow with $A_t L_t$ (Eq. {eq}`eq-iam_detrend`).

2.  Map unbounded time to $[0,1)$ via $\tau = 1 - e^{-\vartheta t}$ (Eq. {eq}`eq-time_transform`).

3.  Normalize all Lagrange multipliers by $A_t^{1-1/\psi}\,L_t$ (Eq. {eq}`eq-iam_norm_mult`).

4.  Derive FOCs from the Lagrangian, both economic *and* climate constraints (Eqs. {eq}`eq-iam_foc_c`–{eq}`eq-iam_env_mat`).

5.  Enforce static FOCs by inversion: invert the consumption FOC for $c_t$ (Eq. {eq}`eq-iam_c_recovery`) and solve the abatement FOC for the implied multiplier $\hat{\lambda}^{\mu,\mathrm{impl}}_t$ (Eq. {eq}`eq-iam_lambdamu_implied`).

6.  Smooth the upper-bound KKT complementarity via Fischer–Burmeister, evaluated at the implied multiplier (Eq. {eq}`eq-iam_fb`).

7.  Form the loss as the sum of squared residuals over the 8 remaining conditions (Eq. {eq}`eq-iam_loss`).

8.  Train as in the stationary DEQN: simulate $\to$ record loss $\to$ backprop $\to$ repeat.
```


This is the deterministic CDICE-DEQN solver in its entirety. Companion notebook [`02_DICE_DEQN_Library_Port.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_02_DICE_DEQN_Library_Port.ipynb) trains it against the CDICE library reference solution of {cite:t}`Folini_2021`; the verification gate inside that notebook is the natural stopping point for a reader who wants only the deterministic core.

(sec-dice_to_stochastic_iam)=
## From CDICE to Stochastic IAMs
The deterministic CDICE-DEQN of {ref}`sec-iam_dequ_loss`, together with the AR(1) productivity extension developed in the remarkbox below, is the right pedagogical anchor because it contains every mechanical component of an integrated assessment model: capital accumulation, emissions, carbon diffusion, temperature dynamics, damages, abatement costs, and the SCC as a shadow price. It is not yet the object one wants for quantitative climate-policy research. Three features are still missing.

First, climate policy is an intrinsically stochastic problem. Productivity, carbon intensity, damages, climate feedbacks, and tipping thresholds are not known constants. Once they are stochastic, a carbon tax is not a path but a state-contingent policy. Second, long-run climate risk makes time-additive CRRA preferences too restrictive: the intertemporal elasticity of substitution and risk aversion should be separate parameters. Third, climate policy is distributional. The representative-agent SCC answers a marginal pricing question, but an implementable policy also asks which cohorts pay the tax and which cohorts receive the transfers. This is the point at which the chapter moves from representative-agent DICE to stochastic overlapping-generations IAMs.

The transition is smooth if one keeps the computational object fixed. In every case the neural network approximates a policy map

$$
\begin{aligned}
    u_t &= \mathcal N_\rho(\tilde{\bm x}_t), \\
    \tilde{\bm x}_t &= (\text{economic states},\ \text{climate states},\ \text{beliefs},\ \text{parameters},\ \text{policy-rule coefficients}),
\end{aligned}
$$

and the loss is still a sum of normalized equilibrium residuals. The only changes are the variables appended to $\tilde{\bm x}_t$ and the conditional expectations appearing in the residuals. {numref}`tab-climate_frontier_map` summarizes the sequence.

````{table}
:name: tab-climate_frontier_map

The layers of the climate-economy pipeline used in the remainder of the chapter. Each layer is a small extension of the previous one; no new numerical paradigm is introduced after the deterministic CDICE-DEQN.

| **Layer** | **Economic question** | **Computational change** |
|---|---|---|
| Deterministic CDICE ({ref}`sec-iam_dequ_loss`) | What is the globally optimal abatement path and SCC at the baseline calibration? | Time-stamped DEQN; eight residuals; horizon $T_{\max}$ chosen so discounting absorbs transversality. |
| Stochastic DICE (AR(1) + GH quadrature, see the productivity-shock remarkbox below) | How do shocks alter the SCC distribution? | Add shock states; replace future terms by Gauss–Hermite expectations. |
| Bayesian learning on ECS ({ref}`sec-bayesian_learning`) | How does learning about climate sensitivity alter the SCC distribution? | Add belief mean and belief variance as states; one signal equation; conjugate Gaussian update. |
| Epstein–Zin DICE ({ref}`sec-ez_layer`) | How do risk aversion and IES separately price climate tails? | Add the value level as a network output; add one recursion residual and an EZ continuation-value weight. |
| Deep UQ ({ref}`sec-deep_uq`) | Which uncertain parameters drive SCC variation? | Treat parameters as pseudo-states; fit a GP surrogate for the QoI; compute Sobol, Shapley, and univariate effects. |
| Stochastic OLG-IAM ({ref}`sec-pareto_carbon_tax`) | Can carbon taxes be welfare improving and Pareto improving across cohorts? | Treat tax coefficients and transfer shares as pseudo-states; fit GP surrogates for cohort welfare; solve constrained policy design on the surrogate. |
````

```{prf:remark} Adding aggregate shocks: AR(1) productivity and Gauss–Hermite quadrature

 Real climate–economy interactions are shot through with stochastic shocks. The minimal stochastic extension that already lets us reproduce the qualitative SCC fan-chart structure of {cite:t}`caiSocialCostCarbon2019` on a laptop adds an AR(1) shock to log TFP:

$$
z_{t+1} = \rho_z\, z_t + \sigma_z\, \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,1),
$$ (eq-tfp_ar1)
with effective TFP $A_t \exp(z_t)$. The state vector {eq}`eq-iam_state` acquires a new entry,

$$
\tilde{\bm{x}}_t = \bigl(k_t,\; M^{\mathrm{AT}}_t,\; M^{\mathrm{UO}}_t,\; M^{\mathrm{LO}}_t,\; T^{\mathrm{AT}}_t,\; T^{\mathrm{OC}}_t,\; \tau_t,\; z_t\bigr)^\top,
$$ (eq-iam_state_stoch)
and each forward-looking residual {eq}`eq-iam_l1`–{eq}`eq-iam_l7` acquires a conditional expectation over $\varepsilon_{t+1}$; the capital Euler, for example, becomes

$$
\begin{aligned}
e^{g^A_t + g^L_t}\,\hat{\lambda}_t \;=\; \hat{\beta}_t\,\mathbb{E}_t\Bigl[\;
&\hat{\lambda}_{t+1}\bigl(\bigl(1-\Omega(T^{\mathrm{AT}}_{t+1}) - \Theta(\mu_{t+1})\bigr)\alpha\,k_{t+1}^{\alpha-1} + (1-\delta)\bigr) \\
&\;+\; \hat{\nu}^{\mathrm{AT}}_{t+1}\,\sigma_{t+1}(1-\mu_{t+1})\,A_{t+1}L_{t+1}\,\alpha\,k_{t+1}^{\alpha-1}\,\Bigr].
\end{aligned}
$$ (eq-iam_euler_stoch)
With $\varepsilon_{t+1}$ Gaussian, the conditional expectation is evaluated with a small number of Gauss–Hermite nodes $\{(\xi_q, w_q)\}_{q=1}^Q$,

$$
\mathbb{E}_t[f(\varepsilon_{t+1})] \;\approx\; \frac{1}{\sqrt{\pi}}\sum_{q=1}^{Q} w_q\, f\bigl(\sqrt{2}\,\xi_q\bigr),
$$ (eq-gh_quadrature)
and each residual is replaced by its stochastic counterpart

$$
l_m^{\mathrm{stoch}}(\tilde{\bm{x}}_t,\, \mathcal{N}_\rho) \;=\; \frac{1}{\sqrt{\pi}}\sum_{q=1}^{Q} w_q\, l_m\bigl(\tilde{\bm{x}}_t,\, \mathcal{N}_\rho;\, \varepsilon_{t+1} = \sqrt{2}\,\xi_q\bigr),
$$ (eq-loss_gh)
which the total loss {eq}`eq-iam_loss` then aggregates as before. In practice $Q = 5$ nodes drive the quadrature error well below the training-noise floor; the GH evaluation is fully differentiable, so the autodiff backward pass is unchanged. When several independent shock dimensions appear simultaneously (productivity shock $\varepsilon_{t+1}$, learning innovation $\tilde\epsilon_{T,t+1}$, and the EZ certainty-equivalent integrand of {ref}`sec-ez_layer`), each conditional expectation is taken under a tensor product of one-dimensional GH rules at $Q$ nodes per dimension, i.e. $Q^d$ total nodes for $d$ shock dimensions; the autodiff backward pass traverses the quadrature unchanged. Forward-simulating $N_{\mathrm{MC}}$ trajectories of the AR(1) shock produces a Monte-Carlo SCC fan chart whose right-tail mass is the channel of {cite:author}`caiSocialCostCarbon2019`'s headline result. Companion notebook [`03_Stochastic_DICE_DEQN.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_03_Stochastic_DICE_DEQN.ipynb) trains this stochastic extension end-to-end and is the natural anchor for {prf:ref}`ex-ch11-7`.
```


```{prf:remark} The unifying trick

 The same pseudo-state idea appears three times: uncertain structural parameters are pseudo-states for UQ, policy-rule coefficients are pseudo-states for constrained optimal policy, and Bayesian posterior beliefs are endogenous pseudo-states for learning. The network does not care which interpretation is attached to an input coordinate; it only learns a differentiable equilibrium map on the enlarged domain. The methodological cost of each layer is small: at most one additional state or output and one additional term in the loss; the algorithm of {ref}`sec-nsdeqn_algo` is unchanged.
```


(sec-bayesian_learning)=
## Bayesian Learning About Climate Sensitivity
**Why ECS is the natural learning state.** The equilibrium climate sensitivity (ECS), defined as the long-run atmospheric warming from a doubling of CO$_2$, is the single most consequential and most uncertain parameter in the climate side of an IAM. Observational, paleoclimate, and model-based estimates place ECS in a *likely* (66%) range of roughly 2.5–4°C and a *very-likely* (90%) range of 2–5°C {cite:p}`sherwood2020assessment,knutti2017beyond,roe2007climate`, and ECS uncertainty is the largest single contributor to SCC dispersion across model variants. Crucially, ECS is partially identified from temperature realizations conditional on emissions and forcing: a Bayesian planner who observes temperature paths can therefore update her posterior period by period, and the policy that maximizes ex-ante welfare conditions on the current posterior rather than on a fixed point estimate.

**How learning enters the state.** Promote the climate-feedback parameter $\lambda$ in {eq}`eq-temp_at` to a stochastic object by adding the feedback term $\varphi_{1C}\,\tilde f_{t+1}\,T^{\mathrm{AT}}_t$ to the right-hand side, where $\varphi_{1C}$ is a calibrated coupling coefficient (taken from {cite:t}`friedlDeep2023`) and $\tilde f_{t+1} \sim \mathcal N(\mu_{f,t}, S_{f,t})$ is a per-period draw under the planner's posterior over the unknown climate-feedback deviation. The unknown itself is time-invariant; the subscript $t{+}1$ indexes the period in which the subjective draw enters the temperature equation, and the planner's posterior moments $(\mu_{f,t}, S_{f,t})$ shift over time as new temperature observations arrive. The planner observes the temperature-residual signal

$$
y_{t+1} \;:=\; \varphi_{1C}\,T^{\mathrm{AT}}_t\,\tilde f_{t+1} \;+\; \tilde\epsilon_{T,t+1},\qquad \tilde\epsilon_{T,t+1} \sim \mathcal N(0, S_{\epsilon_T}),
$$ (eq-bayes_signal)

and conjugate Gaussian–Gaussian updating delivers the posterior

$$
\mu_{f,t+1} = \frac{S_{\epsilon_T}\,\mu_{f,t} + \varphi_{1C}\,T^{\mathrm{AT}}_t\,S_{f,t}\,y_{t+1}}{S_{\epsilon_T} + (\varphi_{1C}\,T^{\mathrm{AT}}_t)^2\,S_{f,t}}
$$ (eq-bayes_mean)

$$
S_{f,t+1} = \frac{S_{\epsilon_T} \cdot S_{f,t}}{S_{\epsilon_T} + (\varphi_{1C}\,T^{\mathrm{AT}}_t)^2\,S_{f,t}}
$$ (eq-bayes_var)

which the planner takes as two additional laws of motion for the belief states $(\mu_{f,t}, S_{f,t})$. These two states occupy the slots already reserved in the augmented state vector {eq}`eq-iam_state`. Equations {eq}`eq-bayes_mean`–{eq}`eq-bayes_var` are the Kalman update for a scalar linear-Gaussian state-space model with observation gain $\varphi_{1C}\,T^{\mathrm{AT}}_t$ and noise variance $S_{\epsilon_T}$; cf. {cite:t}`bishop2006` [§ 13.3] for the generic derivation. The DEQN algorithm of {ref}`sec-nsdeqn_algo` is unchanged: the network simply receives two more inputs and learns a richer policy.

**Where this sits in the literature.** Bayesian learning about climate parameters in an integrated assessment frame has a long pedigree. {cite:t}`kellyBayesianLearningGrowth1999` and {cite:t}`kellyLearningClimateFeedbacks2015` establish the basic Kelly–Kolstad result that learning takes decades to centuries in calibrated DICE-like settings, and that the tradeoff between mitigation (which lowers temperature variance) and information (which requires informative temperature paths) is sharp. {cite:t}`leachClimateChangeLearning2007` and {cite:t}`websterLearningClimateChange2008` sharpen the slow-learning result and quantify the policy errors induced by treating uncertainty as resolved too quickly. On the dynamic-programming side, {cite:t}`caiSocialCostCarbon2019` solve a stochastic-DICE variant with tipping-point hazards and recursive preferences. The robust-control program of {cite:t}`anderson2014Uncertainty`, {cite:t}`barnett2023climate`, {cite:t}`barnett2023deep`, and {cite:t}`Barnett2020` addresses a complementary question (planner ambiguity over the data-generating process), and modern deep-learning solutions are the natural computational companion because tensor-product grids over belief states are infeasible at realistic state-vector sizes.

**Headline result from the UQ literature.** {cite:t}`friedlDeep2023` solve the joint stochastic-DICE–Bayesian-learning DEQN with the methodology of this chapter and find two qualitative features that survive across the calibration cloud.[^1] First, ECS uncertainty is largely resolved within roughly ten years of optimal policy: the posterior variance $S_{f,t}$ shrinks by an order of magnitude over the first decade of the planner's horizon, even though the absolute posterior mean takes longer to settle. Second, the SCC under learning is roughly half the no-learning SCC for moderate true ECS values, and roughly the same as the no-learning SCC at the upper tail of the ECS distribution; learning is a strong substitute for precautionary mitigation in the moderate-ECS regime, and a weak substitute in the tail-ECS regime. The asymmetry is policy-relevant: the value of waiting to learn falls sharply once the planner suspects she is in the tail. The broader teaching point is that uncertainty is not automatically a reason to abate more: its policy effect depends on whether the uncertainty is static, learnable, or associated with irreversible tail risk. {numref}`fig-bayes_learning_schematic` illustrates the two qualitative features.

```{figure} figures/fig-bayes_learning_schematic.svg
:name: fig-bayes_learning_schematic

Schematic of the two qualitative features reported by {cite:t}`friedlDeep2023`. Left: posterior variance $S_{f,t}$ relative to its prior value, on a logarithmic scale. The variance falls by roughly an order of magnitude over the first decade, mirroring the Kelly–Kolstad slow-learning result but accelerated by the deeper signal–noise ratio of the modern climate calibration. Right: $\mathrm{SCC}_0$ as a function of the true ECS, with and without Bayesian learning. Learning approximately halves the SCC at moderate ECS values where uncertainty is the dominant driver of precautionary abatement, but converges to the no-learning curve at the upper tail where the underlying physical damage dominates. Curves are illustrative; the magnitudes are those quoted in the body text.
```

(sec-ez_layer)=
## Epstein–Zin Preferences
**Why recursive preferences for climate.** The time-additive CRRA-IES aggregator {eq}`eq-planner_obj_crra` ties risk aversion and intertemporal substitution together. Climate policy is exactly the environment in which this restriction is least attractive. A planner may want a high IES $\psi$ to govern intertemporal substitution across long horizons, and a separate high coefficient of relative risk aversion $\gamma_u$ to price low-probability climate disasters. Recursive Kreps–Porteus preferences, following {cite:t}`epstein1989Substitution` and {cite:t}`weil1989Equity`, implement this separation.[^2]

Working with the normalized per-capita value $v_t$ and per-capita consumption $c_t = C_t/L_t$, and writing $\beta^{\mathrm{EZ}}_t := \exp(-\rho\,\Delta_t)$ for the one-period Epstein–Zin discount factor, the recursion is

$$
v_t
    =
    \left[
    (1-\beta^{\mathrm{EZ}}_t)\, c_t^{1-1/\psi}
    +
    \beta^{\mathrm{EZ}}_t
    \left(\mathbb E_t\!\left[v_{t+1}^{1-\gamma_u}\right]\right)^{\frac{1-1/\psi}{1-\gamma_u}}
    \right]^{\frac{1}{1-1/\psi}},
$$ (eq-ez_iam)

with the usual logarithmic limits when $\psi = 1$ or $\gamma_u = 1$, subject to the same budget constraint, capital-accumulation law, and climate dynamics as before.

**What changes in the DEQN loss.** The value level $v_t$ becomes an additional network output, paired with a ninth residual that enforces the recursion {eq}`eq-ez_iam`:

$$
\mathcal R^{\mathrm{EZ}}_t
    =
    v_t
    -
    \left[
    (1-\beta^{\mathrm{EZ}}_t)\, c_t^{1-1/\psi}
    +
    \beta^{\mathrm{EZ}}_t
    \left(\mathbb E_t\!\left[v_{t+1}^{1-\gamma_u}\right]\right)^{\frac{1-1/\psi}{1-\gamma_u}}
    \right]^{\frac{1}{1-1/\psi}}.
$$ (eq-ez_residual)

In the deterministic CRRA-IES core of {ref}`sec-iam_dequ_loss`, $v_t$ never appears explicitly, which is why eight residuals suffice there. The Euler and costate residuals of {ref}`sec-dice_deqn` keep their deterministic form but receive a Bansal–Yaron certainty-equivalent weighting inside each conditional expectation. It is convenient to write the one-step recursive-pricing kernel as

$$
\mathcal M^{\mathrm{EZ}}_{t,t+1}
    =
    \hat\beta_t
    \left(
    \frac{v_{t+1}}{\left(\mathbb E_t[v_{t+1}^{1-\gamma_u}]\right)^{1/(1-\gamma_u)}}
    \right)^{1/\psi - \gamma_u}
    \left(\frac{c_{t+1}}{c_t}\right)^{-1/\psi},
$$ (eq-ez_kernel)

where $\hat\beta_t$ inherits the deterministic growth normalization of {eq}`eq-iam_eff_beta`. In the code, {eq}`eq-ez_kernel` is just a multiplicative weight on next-period marginal-value terms; the certainty-equivalent operator inside the Kreps–Porteus aggregator becomes a second nested expectation. The DEQN loss inherits one extra Gauss–Hermite quadrature step and one extra network output, but no new algorithmic ingredient.

**Interpretation for the SCC.** {cite:t}`crost2013Optimal` and {cite:t}`crost2014Optimal` establish the analytic baseline: in a deterministic IAM, decoupling risk aversion from the IES changes the optimal carbon tax only when stochastic risk is present, but the change can be quantitatively large once it is. {cite:t}`jensenOptimalClimateChange2014` and {cite:t}`traeger2018ace` {cite}`traeger2021ACE` extend the result to closed-form ACE-class settings and show that for reasonable risk aversion above $1/\psi$, the SCC roughly doubles relative to CRRA; {cite:t}`caiSocialCostCarbon2019` reach the same conclusion in a fully stochastic DICE variant. Intuitively, recursive preferences change the SCC because carbon emissions affect the distribution of long-run consumption, not only its mean: if damages create low-consumption tail states, a high $\gamma_u$ raises the SCC through the disaster-insurance channel. The sign of the IES effect depends on which shock dominates: in TFP-driven economies higher $\psi$ dampens the SCC because consumption smoothing absorbs the productivity risk, whereas in temperature-driven economies higher $\psi$ amplifies the SCC because the planner cares more about late-horizon consumption losses. {cite:t}`bansalKikuOchoa2016` make the asset-pricing case for the same channel: long-run temperature shifts price into expected returns through the EZ aggregator, and ignoring them understates the welfare cost of carbon emissions. This is why stochastic DICE with Epstein–Zin preferences is a better teaching object than deterministic DICE for climate-finance questions: it connects welfare, tail risk, and asset-pricing logic in a single equilibrium loss.

(sec-deep_uq)=
## Deep Uncertainty Quantification via Surrogates
Deep UQ answers a different question from solving one stochastic IAM. The object is now a scalar quantity of interest,

$$
q(\theta) = \mathrm{SCC}_{2100}(\theta), \qquad \theta\in\Theta\subset\mathbb R^{d_\theta},
$$ (eq-uq_qoi)

where $\theta$ collects uncertain structural parameters: the ECS or its prior mean $\mu_{f,0}$, the prior variance $S_{f,0}$, the pure rate of time preference $\rho$, the IES $\psi$, risk aversion $\gamma_u$, the damage curvature $\pi_2$, and any tipping parameters included in the experiment. Direct global sensitivity analysis would require solving the IAM thousands of times. Deep UQ replaces this infeasible outer loop by two amortizations.

**Amortization 1: parameters as pseudo-states.** The pseudo-state trick of {cite:t}`friedlDeep2023` collapses the outer loop into a single DEQN training pass. Uncertain parameters $\theta$ are appended to the network's input,

$$
\tilde{\bm{x}}_t = \bigl(\underbrace{\bm x_t}_{\text{economic + climate states}},\; \underbrace{\theta}_{\text{uncertain parameters}}\bigr),\qquad u_t = \mathcal N_\rho(\tilde{\bm x}_t),
$$ (eq-pseudo_state)

held fixed within each simulation episode and re-sampled across episodes from a design distribution $\mathcal D_\theta$. One trained network therefore approximates the policy function for every $\theta$ in $\mathcal D_\theta$; evaluating any new $\theta$ requires only a forward pass. For very large pseudo-state dimensions the active-subspace methods of {ref}`sec-active_subspaces` compress $\theta$ before the next step. This is the same idea as the parameterized policy networks in {ref}`ch-estimation`; here the target is not an SMM criterion but an SCC distribution.

**Amortization 2: a GP for the quantity of interest.** After training, the DEQN is evaluated at a design set $\{\theta_i\}_{i=1}^n$ and the corresponding QoI values $q_i = q(\theta_i)$ are computed by forward simulation. Fit a Gaussian-process surrogate

$$
q(\theta) = m(\theta) + \varepsilon(\theta), \qquad m(\theta)\mid \{(\theta_i,q_i)\}_{i=1}^n \sim \mathcal{GP}\bigl(\mu_n(\theta),\, k_n(\theta,\theta')\bigr).
$$ (eq-uq_gp)

The GP is cheap enough to evaluate millions of times, so the expensive IAM is no longer called inside Sobol, Shapley, or univariate-effect estimators. Bayesian active learning improves the design by adding points where the GP posterior uncertainty is largest or where integrated posterior variance is most reduced, following the toolkit of {ref}`ch-gp` (see {numref}`fig-surrogate_outer_loop` and {numref}`tab-surrogate_strategy_comparison`).

**Sobol, Shapley, univariate effects.** Three complementary global sensitivity indices answer different questions about how $\theta$ drives the SCC. The first-order Sobol index $S_i$ of {cite:t}`sobolGlobalSensitivityIndices2001` measures the share of output variance explained by $\theta_i$ alone,

$$
S_i = \frac{\mathrm{Var}\bigl(\mathbb E[q(\theta)\mid\theta_i]\bigr)}{\mathrm{Var}(q(\theta))},
$$ (eq-sobol_first)

and the total-effect index captures both direct and interaction effects,

$$
S_i^{\mathrm{tot}} = 1 - \frac{\mathrm{Var}\bigl(\mathbb E[q(\theta)\mid\theta_{-i}]\bigr)}{\mathrm{Var}(q(\theta))}.
$$ (eq-sobol_total)

For independent inputs the $\{S_i\}$ sum to at most one, while the $\{S_i^{\mathrm{tot}}\}$ can exceed one in the presence of interactions; equality $\sum_i S_i^{\mathrm{tot}} = 1$ characterizes additive models. Shapley effects, introduced into sensitivity analysis by {cite:t}`owen2014sobol` and developed further by {cite:t}`songShapleyEffectsGlobal2016` and {cite:t}`ioossShapleyEffectsSensitivity2019`, allocate $\mathrm{Var}(q)$ across parameters via cooperative-game averaging over all subsets of other parameters {cite:p}`shapley1953value`, sum exactly to $\mathrm{Var}(q)$ (raw) or one (normalized), and handle correlated inputs cleanly. Univariate-effect plots show the conditional mean $\mathbb E[q(\theta)\mid\theta_i]$ as $\theta_i$ varies and capture the directional response that Sobol indices average over. {cite:t}`saltelli2010Sensitivity` and {cite:t}`saltelli2008global` give the standard estimators and best-practice warnings.

```{prf:remark} Deep UQ implementation recipe

1.  Choose a parameter domain $\Theta$ and a sampling law $\mathcal D_\theta$ for the uncertain climate, damage, and preference parameters.

2.  Train the stochastic CDICE-DEQN on $(\bm x_t, z_t, \mu_{f,t}, S_{f,t}, \theta)$, resampling $\theta$ across episodes and holding it fixed within an episode.

3.  Generate $n$ design evaluations $\{(\theta_i, q_i)\}_{i=1}^n$ from the trained network, where $q_i$ is typically $\mathrm{SCC}_{2100}$ or an expected welfare functional.

4.  Fit a GP surrogate $\theta \mapsto q(\theta)$, validate by leave-one-out cross-validation (target: LOO $R^2 \ge 0.95$ or LOO RMSE below $5\%$ of the QoI standard deviation), and enrich the design with Bayesian active learning if the threshold is not met.

5.  Compute Sobol, Shapley, and univariate effects on the GP surrogate, not on the structural model.
```


The reason this pipeline is the only feasible route is computational: direct Monte Carlo on Sobol or Shapley indices requires $O(10^4)$ to $O(10^6)$ evaluations of the structural model at fresh $\theta$ draws. Even at one DEQN solve per parameter vector, that price tag is several core-decades. The DEQN-with-pseudo-states amortizes one loop, and the GP surrogate amortizes the other; the sensitivity indices are then computed on the GP rather than on the IAM.

**Empirical headline.** {cite:t}`friedlDeep2023` apply the pipeline to a stochastic DICE variant with Epstein–Zin preferences and Bayesian learning, and find that two ingredients dominate the SCC variance across 2020–2100: the mean of the ECS belief (roughly 50–70% of the total-effect Sobol share) and the curvature parameter of the damage function (roughly 15–25%).[^3] Together these account for 70–90% of the SCC variance. Risk aversion contributes a few percentage points; the pure rate of time preference and the IES contribute negligibly once damage curvature is conditioned on. The policy lesson is that under deep uncertainty the SCC should be reported as a distribution, not a point estimate, and that climate-policy design should target tail insurance against the upper ECS–damage corner rather than precision over the central calibration. {numref}`fig-sobol_shares_schematic` sketches the resulting variance decomposition.

```{figure} figures/fig-sobol_shares_schematic.svg
:name: fig-sobol_shares_schematic

Schematic of the total-effect Sobol shares of $\mathrm{SCC}_{2100}$ variance reported by {cite:t}`friedlDeep2023`. Midpoints reflect the ranges quoted in the text (ECS mean 50–70%, damage curvature 15–25%), with horizontal error bars on the two leading parameters indicating the spread across horizon dates and damage-function specifications. The shape, two parameters carrying almost the entire variance, is what motivates the tail-insurance framing in the closing paragraph.
```

(sec-pareto_carbon_tax)=
## Constrained Pareto-Improving Carbon Tax in OLG-IAMs

The SCC analysis of {ref}`sec-deep_uq` is still the marginal welfare cost of one extra ton of carbon *to a representative agent*. Climate policy, however, redistributes welfare across cohorts: today's workers pay abatement costs while tomorrow's households inherit a cooler planet. A *Pareto-improving* carbon tax must transfer enough revenue back to current cohorts that no generation is worse off than under business-as-usual. This section closes Movement 3 by walking through the constrained-Pareto OLG-IAM of {cite:t}`kubler2025using`, reusing the DEQN-with-pseudo-states machinery of {ref}`sec-deep_uq` and the GP surrogate of {ref}`ch-gp`. The Pareto-improvement criterion is closely related to the social-security reform literature {cite:p}`krueger2006pareto`, to recent work on intergenerational climate policy {cite:p}`Karp_Peri_Rezai_2024,kotlikoff2020ParetoImproving`, and to the constrained-optimal-tax frontier of {cite:t}`douenne_hummel_pedroni_2024`.

**Notation reset for this section.**

The OLG-IAM uses different conventions than the representative-agent CDICE block of {ref}`sec-dice`, following {cite:t}`kubler2025using`, and we summarize the differences here so the reader is not surprised. $\Omega_t(T_t)$ now denotes the *retained-output* factor, so net output is $\Omega_t \Phi K^\alpha L^{1-\alpha}$ rather than $(1-\Omega-\Theta)Y^{\mathrm{gross}}$. $p^{\mathrm{tax}}_t$ is the carbon tax (a per-tCO$_2$ price); to avoid clashing with the transformed-time variable $\tau_t$ of {ref}`sec-iam_nonstationarity`, this section uses $p^{\mathrm{tax}}_t$ for the tax throughout, in line with the price-level interpretation. $e_t$ denotes the per-period emissions *flow* (in GtC), and $E_t = \sum_{s\le t} e_s$ is *cumulative* emissions through date $t$; this is the convention of the climate-emulator literature {cite:p}`dietz2019cumulative` and of the companion paper, and it is the source of the section's frequent "cumulative-emissions tax" phrasing. Finally, the policy vector that the planner ultimately optimizes over is $\vartheta = (\vartheta_{\mathrm{tax}}, \omega)$, the joint vector of tax-rule coefficients and cohort transfer shares defined in Step 1 below.

**From CDICE to a TCRE emulator.** The OLG-IAM uses a much simpler climate side than the 5-state CDICE module of {ref}`sec-dice` (three carbon stocks plus two temperature layers). Once the planner's horizon is converted to cumulative-emissions form $E_t = \sum_{s\le t} e_s$, the linear Transient Climate Response to cumulative carbon Emissions (TCRE) approximation collapses the carbon-cycle and energy-balance machinery to a single algebraic relation $T^{\mathrm{AT}}_t \approx \sigma_{\mathrm{CCR}}\,E_t$ {cite:p}`dietz2019cumulative`, which removes five climate states from the planner's optimization. The simplification is essential: it is what makes the OLG state space (12 cohort assets + 5 climate / shock states + $\vartheta$ pseudo-states) tractable end-to-end on a GPU. The reader who finds the change abrupt should treat the TCRE relation as a reduced-form summary of the same physics that drove {ref}`sec-dice_carbon_cycle`–{ref}`sec-dice_temperature`, fitted directly to long-run paths rather than block-by-block. {numref}`fig-cdice_vs_tcre` contrasts the two climate sides.

```{figure} figures/fig-cdice_vs_tcre.svg
:name: fig-cdice_vs_tcre

Climate side of CDICE versus TCRE. The 5-state CDICE module on the left, in which atmospheric carbon, two ocean carbon reservoirs, atmospheric temperature, and ocean temperature all enter the planner's state, is collapsed in the OLG-IAM to a single algebraic relation between cumulative emissions and atmospheric temperature, $T^{\mathrm{AT}}_t \approx \sigma_{\mathrm{CCR}}\,E_t$. The simplification trades fidelity to short-run climate dynamics for tractability of the 12-cohort heterogeneous-agent state space and is what makes the bilevel policy search of {ref}`sec-pareto_carbon_tax` end-to-end feasible.
```

(sec-olg_iam)=
### The OLG-IAM Model
The model features $A=12$ overlapping generations of selfish agents (ages 20–80 in 5-year periods), a competitive firm, and a simplified, cumulative-emissions climate module in the spirit of {cite:t}`dietz2019cumulative`:

- **Technology:** Output is $Y_t = \Omega_t(T_t)\,\Phi(\mu_t)\,K_t^\alpha L_t^{1-\alpha}$ with retained-output damage factor $\Omega_t$ and net-of-abatement-cost factor $\Phi(\mu_t) = 1 - \theta_1\mu_t^{\theta_2}$; emissions are $e_t = (1-\mu_t)\kappa_t Y_t$ with stochastic carbon intensity $\kappa_t$; the period resource constraint is $C_t + K_{t+1} = Y_t + (1-\delta)K_t$.

- **Households:** Each agent maximizes $\mathbb{E}_t\sum_{j=1}^{A}\beta^{j-1}\, C_{t+j-1,j}^{1-\sigma_u}/(1-\sigma_u)$ (where $\sigma_u$ is the household CRRA risk-aversion coefficient, distinct from the climate-chapter notation $\sigma_t$ for emissions intensity) subject to the budget constraint $C_{t,j} + a_{t+1,j+1} = (1+r_t)\,a_{t,j} + w_t\,l_j + \mathbb{T}_{t,j}$, where $\mathbb{T}_{t,j}$ is the transfer from carbon tax revenue and $j$ runs over all $A=12$ cohorts alive at $t$ (newborns included).

- **Climate:** The climate emulator imposes a near-linear relationship between cumulative emissions and atmospheric temperature, $T^{\mathrm{AT}}_t \approx \sigma_{\mathrm{CCR}}\,E_t$ where $E_t = \sum_{s\le t} e_s$ is cumulative carbon, augmented by a stochastic tipping mechanism: damages depend on $T^{\mathrm{AT}}_t$ relative to a threshold $TP_t$ via a Weitzman-type retained-output factor that becomes steeply convex once $T^{\mathrm{AT}}_t$ approaches $TP_t$.

- **Stochastic shocks:** Carbon intensity $\kappa_t$ follows an AR(1) with time-varying persistence; the tipping threshold $TP_t$ follows a bounded random walk that becomes absorbing once it has been crossed.

The household Euler equation takes the standard form $C_{t,j}^{-\sigma_u} = \beta\,\mathbb{E}_t[(1+r_{t+1})\,C_{t+1,j+1}^{-\sigma_u}]$ for $j = 1,\ldots,A-1$, and market clearing requires that aggregate savings equal the capital stock: $\sum_j a_{t,j} = K_t$. {numref}`fig-bau_olg_baseline` simulates this model without policy intervention; it fixes the business-as-usual (BAU) baseline against which every Pareto-improving policy below is benchmarked, and supplies the cohort-by-cohort participation constraints for the constrained policy search.

```{figure} figures/jpe_bau_baseline.png
:name: fig-bau_olg_baseline
:width: 92%

Business-as-usual baseline for the 12-cohort stochastic OLG-IAM of {cite:t}`kubler2025using`. Without policy intervention the median warming reaches roughly $3\,{}^\circ$C over the 150-year horizon, and the upper tail of damages is substantially larger than the mean. Every Pareto-improving policy below is benchmarked against this baseline, which also supplies the participation constraints for the constrained policy search. Figure extracted from {cite:t}`kubler2025using`.
```

### The 3-Step ML Pipeline

Finding an optimal carbon tax rule in this OLG economy is a bilevel optimization problem: the outer level searches over tax parameters, and the inner level solves the full stochastic general equilibrium for each candidate tax. {cite:t}`kubler2025using` decompose this into three steps, summarized in {numref}`fig-olg_iam_pipeline`:

```{figure} figures/fig-olg_iam_pipeline.svg
:name: fig-olg_iam_pipeline

Three-step machine-learning pipeline for constrained carbon-tax design. The DEQN amortizes equilibrium solution across tax parameters, the GP surrogate maps policy parameters to welfare and cohort utilities, and the final optimization imposes the Pareto constraints on the surrogate.
```

**Step 1: DEQN with pseudo-states.** The tax-rule coefficients $\vartheta_{\mathrm{tax}}$ and the $A=12$ transfer shares $\omega = (\omega_1,\ldots,\omega_{12})$ are appended to the state of the neural network as pseudo-states. The transfer shares are non-negative weights satisfying $\sum_{j=1}^{A} \omega_j = 1$, with cohort $j$'s lump-sum transfer given by $\mathbb{T}_{t,j} = \omega_j\,p^{\mathrm{tax}}_t\,e_t$ from the government's resource constraint $\sum_j \mathbb{T}_{t,j} = p^{\mathrm{tax}}_t\,e_t$. The simplex constraint $\omega \in \Delta^{A-1}$ is enforced by sampling unconstrained logits and applying a softmax before feeding $\omega$ into the network, so the DEQN never sees an infeasible transfer profile. All cohorts alive at $t$, including the newborn cohort, receive a transfer. The number of tax parameters depends on the rule: a simple linear rule on cumulative emissions has $\vartheta_{\mathrm{tax}} = (\vartheta_0,\vartheta_E) \in \mathbb{R}^2$ (so a 14-dimensional pseudo-state vector together with the 12 transfer shares), and a richer rule that adds dependence on carbon intensity and tipping has $\vartheta_{\mathrm{tax}} \in \mathbb{R}^4$ (a 16-dimensional pseudo-state vector with the 12 shares). The DEQN learns the equilibrium for *all* candidate tax-and-transfer configurations at once, so that simulating any $(\vartheta_{\mathrm{tax}}, \omega)$ requires only a forward pass. The network architecture, optimizer schedule, and training-pool design follow {cite:t}`kubler2025using` verbatim; the exact configuration is documented in the companion repository linked at the end of this section.

**Step 2: GP surrogate.** At each design point $\vartheta = (\vartheta_{\mathrm{tax}}, \omega)$, the trained DEQN is simulated to obtain Monte-Carlo estimates of expected lifetime utility for the 40 tracked cohorts (12 alive at $t=0$ plus 28 future cohorts born during the planner's 150-year horizon). Independent GPs are then fitted to map $\vartheta$ to expected aggregate welfare $\mathcal{W}(\vartheta)$ and to each of the 40 cohort welfares $\tilde{U}_t(\vartheta)$. The design itself uses Latin-hypercube sampling augmented with Bayesian active learning: the size scales with the dimension of $\vartheta$, with roughly 500 points sufficient for the 14-dimensional "linear-in-$E$ + transfers" specification (Section 5.3 of {cite:t}`kubler2025using`) and roughly 800 points for the 16-dimensional "richer rule + transfers" specification (Section 5.4). {numref}`fig-gp_welfare_contour` shows the resulting welfare surface for the two-parameter linear-in-cumulative-emissions rule, with transfer shares held at the Pareto-optimal solution: the contour exposes the low-dimensional ridge along which intercept and slope trade off cleanly, and on which the Step-3 optimizer searches.

```{figure} figures/jpe_gp_welfare_contour.png
:name: fig-gp_welfare_contour
:width: 55%

Gaussian-process welfare surrogate over the two-dimensional tax-parameter slice $(\vartheta_0, \vartheta_E)$ of the linear-in-cumulative-emissions rule, with transfer shares $\omega$ held at the Pareto-optimal solution. The contour exposes the low-dimensional welfare surface on which the constrained optimizer of Eq. {eq}`eq-pareto_opt` searches once the DEQN has amortized the equilibrium solve. Figure extracted from {cite:t}`kubler2025using`.
```

**Step 3: Constrained optimization.** The planner solves

$$
\vartheta^* = \argmax_{\vartheta = (\vartheta_{\mathrm{tax}}, \omega)}\;\mathcal{W}(\vartheta) \qquad \text{s.t.}\quad \tilde{U}_t(\vartheta) \geq U_t \;\;\forall\, t,\;\; \omega \in \Delta^{A-1},
$$ (eq-pareto_opt)

where $U_t$ is the business-as-usual (BAU) welfare of cohort $t$ and $\Delta^{A-1}$ is the standard simplex on $A=12$ shares. The Pareto constraint ensures that *no generation is worse off*; whenever the welfare-maximizing $\vartheta^\ast$ lies strictly inside the feasible polytope (which is the case in every scenario reported below) it is also strictly Pareto-improving for at least one cohort, so the weak constraint $\tilde U_t \ge U_t$ and the textbook strict-improvement requirement coincide at the optimum. Because each evaluation of $\mathcal{W}$ and $\tilde U_t$ is a forward pass through the trained GP rather than a fresh DEQN simulation, the constrained search reduces to a sequence of small SLSQP problems (the paper uses 500 random restarts of `scipy.optimize.minimize`) that complete in seconds. By contrast, replacing the surrogate with brute-force re-solves of the full SOLG IAM at every candidate $\vartheta$ would require on the order of tens of thousands of core-hours per candidate, which is the comparison the paper draws against traditional methods.

### Results: Why Transfers Matter

The unconstrained welfare-maximizing cumulative-emissions tax is the natural benchmark. With a linear rule $p^{\mathrm{tax}}_t = \vartheta_0 + \vartheta_E\,E_t$ and a fixed declining transfer scheme $\omega = \bar\omega$, the policy cuts emissions aggressively, stabilizes mean warming around $2.7\,{}^{\circ}\mathrm C$, and raises aggregate social welfare by about $1.6\%$ in consumption-equivalent terms. But it imposes losses of up to roughly $5\%$ on initial generations: it is therefore welfare-improving in the social-welfare-function sense, but *not* Pareto improving. {numref}`fig-unconstrained_linear_tax` shows the failure: the welfare-gains panel records the losses for transition generations that the social-welfare-function aggregate hides.

```{figure} figures/jpe_unconstrained_linear_tax.png
:name: fig-unconstrained_linear_tax
:width: 95%

Welfare-improving but not Pareto-improving cumulative-emissions tax with a fixed exogenous transfer scheme. The policy strongly reduces climate risk and raises aggregate welfare by about $1.6\%$ in consumption-equivalent terms, but the welfare-gains panel shows losses for transition generations. Figure extracted from {cite:t}`kubler2025using`.
```

Endogenizing the transfer shares changes the conclusion. With the same simple tax base and an optimized transfer simplex, {cite:t}`kubler2025using` report the optimized coefficients[^4]

$$
p^{\mathrm{tax}}_t = \vartheta_0 + \vartheta_E\,E_t,
    \qquad
    (\vartheta_0,\, \vartheta_E) = (-0.186,\, 0.225),
$$ (eq-pareto_linear_tax_coefficients)

together with transfer shares

$$
\omega
=
(0.128,\, 0.051,\, 0.058,\, 0.089,\, 0.149,\, 0.090,\, 0.066,\, 0.143,\, 0.076,\, 0.048,\, 0.039,\, 0.061),
$$ (eq-pareto_linear_transfer_shares)

which sum to one up to rounding. {numref}`fig-pareto_transfer_profile` plots this transfer profile against cohort index; the non-monotone shape is what allows a less aggressive cumulative-emissions tax to satisfy the Pareto constraint at every age, and it is the single most informative graphical summary of the constrained-optimal-policy step. The negative intercept $\vartheta_0 = -0.186$ is not a subsidy in practice: the planner's horizon starts well into the industrial era at a strictly positive cumulative-emissions stock $E_0 > 0$, so the effective tax $\vartheta_0 + \vartheta_E\,E_t$ is positive for every relevant $E_t$ along the optimum. The negative intercept simply registers that the linear-in-$E$ rule undershoots a constant carbon price near $E = 0$ and ramps up roughly proportionally to cumulative emissions thereafter. The combined policy makes every tracked cohort weakly better off than under BAU. The aggregate welfare gain is more modest than under the unconstrained optimum, at about $0.42\%$ in consumption-equivalent terms, but the right tail of damages is truncated: the 99th percentile of damages falls to roughly $7\%$ of output rather than about $9\%$ under BAU. {numref}`fig-pareto_tax_main` reports the full result. Comparing its welfare-gains panel with that of {numref}`fig-unconstrained_linear_tax` is the section's headline: a lower, simpler tax combined with an optimized transfer system shifts every cohort weakly into the gains region.

```{figure} figures/jpe_pareto_linear_tax.png
:name: fig-pareto_tax_main
:width: 95%

Pareto-improving cumulative-emissions tax with optimized intergenerational transfers, at the coefficients of {eq}`eq-pareto_linear_tax_coefficients`–{eq}`eq-pareto_linear_transfer_shares`. The tax is less aggressive than the unconstrained rule, but the optimized transfer system shields current cohorts while preserving climate-risk reduction for future cohorts. Aggregate welfare rises by about $0.42\%$. Figure extracted from {cite:t}`kubler2025using`.
```

```{figure} figures/fig-pareto_transfer_profile.svg
:name: fig-pareto_transfer_profile

Optimized transfer-share profile $\omega_j$ across the 12 cohorts alive at $t = 0$, drawn directly from {eq}`eq-pareto_linear_transfer_shares`. The profile is decidedly non-monotone: the largest shares go to cohorts 1 (oldest), 5, and 8, which are precisely the cohorts the participation constraint $\tilde U_t \ge U_t$ binds most tightly for under the un-transferred tax of {numref}`fig-unconstrained_linear_tax`. The non-monotone shape is what allows a less aggressive cumulative-emissions tax to satisfy Pareto improvement at every age.
```

The richer rule of {ref}`sec-pareto_carbon_tax` adds carbon intensity and a tipping-state statistic,

$$
p^{\mathrm{tax}}_t = \vartheta_0 + \vartheta_E\,E_t + \vartheta_\kappa\,\kappa_t + \vartheta_{TP}(1-D_t),
$$

where $D_t$ is the climate-tipping state of the model (built from the proximity of $T^{\mathrm{AT}}_t$ to the stochastic threshold $TP_t$ and the absorbed-tipping flag). Its optimized coefficients are

$$
(\vartheta_0,\, \vartheta_E,\, \vartheta_\kappa,\, \vartheta_{TP}) = (-0.237,\, 0.203,\, 0.037,\, 0.012),
$$ (eq-pareto_full_tax_coefficients)

with the associated aggregate welfare gain rising only from about $0.42\%$ to about $0.45\%$. The cohort-by-cohort welfare profile (not plotted; see {cite:t}`kubler2025using` for the figure) again keeps every cohort weakly above its BAU baseline, and the marginal welfare improvement from the extra two policy-state coefficients is small. This is the substantive headline of {cite:t}`kubler2025using`: once intergenerational transfers are optimized, the simple cumulative-emissions tax captures most of the feasible Pareto-improving welfare gain. More policy-state variables improve the fit to climate risk, but the participation constraints bind tightly enough that the marginal welfare benefit of policy complexity is small. $D_t$ is a deterministic function of variables already in the SOLG state, so it can be evaluated inside each forward pass; the exact functional form is in the paper.

**Runtime in numbers.** On a standard laptop (Apple M1), the OLG DEQN trains in roughly four wall-clock hours; on a high-end accelerator such as an NVIDIA GH200, training drops to the order of minutes {cite:p}`kubler2025using`. Adding the GP fits over 500 (resp. 800) design points and the constrained Step-3 optimization keeps the entire pipeline within the same order of magnitude, while the comparable brute-force re-solve of the SOLG model at *every* candidate $\vartheta$ would dominate by orders of magnitude (the paper reports tens of thousands of core-hours for one fixed-parameter calibration, which would have to be repeated for every Step-3 candidate vector).

**Companion code.** The full production OLG-IAM solver, including the DEQN training loop with $(\vartheta_{\mathrm{tax}}, \omega)$ pseudo-states and the bilevel policy search, is hosted in the companion repository [`sischei/JPE_Macro_Using_ML_to_compute_constrained_optimal_carbon_tax_rules`](https://github.com/sischei/JPE_Macro_Using_ML_to_compute_constrained_optimal_carbon_tax_rules), which accompanies {cite:t}`kubler2025using`. The classroom notebook in Lecture 17 of this course exposes a reduced surrogate-only version that loads pre-trained GP surrogates and reproduces the constrained-optimization step (Step 3) interactively, but does not retrain the OLG DEQN end-to-end; readers who want the full pipeline should clone the companion repository.

```{prf:remark} The policy-design message

 Deep learning is used here not because a neural network is fashionable, but because the relevant object is a high-dimensional map from states and policy-rule coefficients to equilibrium allocations and cohort welfare. Once that map is learned, constrained optimal policy becomes a small optimization problem on a surrogate. The economic result is equally important: carbon taxes need transfers. Without transfers, the welfare-maximizing tax creates transition losers; with optimized transfers, a simpler and lower tax can be Pareto improving. The main welfare channel is disaster-risk reduction (tail insurance), not maximal average abatement.
```


(sec-climate_discussion)=
## Discussion and Outlook
The combination of DEQNs, pseudo-states, and GP surrogates provides a scalable and transparent framework for climate economics that overcomes key limitations of traditional methods.

**Comparison with traditional IAM solutions.** Standard IAMs (such as the GAMS implementation of DICE) rely on shooting methods or nonlinear programming solvers that find deterministic optimal paths. These approaches struggle with stochastic extensions: Monte Carlo integration over shocks is expensive, and certainty equivalence (replacing random variables with their means) misses the welfare cost of tail risks. The DEQN approach approximates the stochastic recursive solution over the chosen training distribution and state/pseudo-state domain (with Bayesian learning and recursive Epstein–Zin utility) in a single training run.

**Limitations.** Several limitations should be noted. First, the CDICE climate module, while calibrated to CMIP benchmarks, remains a reduced-form emulator and cannot capture spatial heterogeneity or regional climate impacts. Second, the OLG-IAM treats each generation as identical within a cohort; within-cohort heterogeneity (e.g., geographic exposure to climate damages) would require further extensions along the lines of {ref}`ch-young`. Third, the linear tax rules are interpretable and implementable but may leave welfare gains on the table relative to fully nonlinear rules.

**Extensions.** Active research frontiers include: multi-region IAMs with trade and carbon leakage {cite:p}`nordhaus1996regional`; richer damage specifications including tipping cascades; endogenous technical change in abatement technology; and embedding climate modules in continuous-time heterogeneous-agent models ({ref}`ch-ct_theory`) to study the joint dynamics of climate risk and wealth inequality. The methodological toolkit developed in this course (DEQNs for equilibrium computation, PINNs for continuous-time PDEs, deep surrogates for uncertainty quantification, and Young's method for distribution tracking) provides the computational infrastructure for these extensions.

**The three movements, in one synthesis.** Movement 1 established that solving an IAM by DEQN requires three modifications relative to the stationary toolkit of {ref}`ch-deqn`: time enters as a state, the training pool is built by simulating $K$ forward trajectories from a calibrated initial state rather than by sampling an ergodic distribution, and the missing transversality is absorbed numerically by choosing the horizon $T_{\max}$ long enough that discounting suffices (or, on short horizons, by adding an explicit terminal residual). Movement 2 put that algorithm to work on a worked stochastic DICE economy, producing the eight-residual loss whose minimization delivers the deterministic policy and, with one extra Gauss–Hermite layer, the AR(1) SCC fan chart. Movement 3 layered four extensions onto the same spine: Bayesian learning over the climate sensitivity, recursive Epstein–Zin preferences, global UQ of the SCC via pseudo-states and GP surrogates, and constrained Pareto-improving carbon-tax design in a heterogeneous-agent OLG-IAM. {ref}`ch-outlook` threads these into the broader synthesis with the rest of the course.

```{prf:remark} Chapter Summary

- Climate-economy IAMs are the natural showcase for the methodological stack: a moderately high-dimensional non-stationary DSGE solved by DEQN, a GP+BAL surrogate for SCC sensitivity, and Sobol/Shapley decomposition for deep uncertainty quantification.

- DICE (and the CDICE recalibration) provide the workhorse model; {prf:ref}`ex-ch11-3` asks the reader to use the closed-form ACE expression of {cite:t}`traeger2018ace` as an analytic benchmark for the DEQN-trained CDICE solution.

- Pareto-improving carbon-tax rules {cite:p}`kubler2025using` demonstrate that the surrogate machinery has direct policy relevance: linear, intergenerationally fair, implementable rules can be designed via constrained optimization on the surrogate.

- Deep UQ matters because the SCC distribution under climate, damage, and preference uncertainty is wider than any pointwise estimate suggests; reporting the distribution rather than a number is the responsible default.
```


## Further Reading {.unnumbered}
- {cite:t}`nordhausRevisitingSocialCost2017`, the canonical DICE update.

- {cite:t}`Folini_2021`, the CDICE recalibration used in the deep-learning solution.

- {cite:t}`traeger2018ace`, the analytic ACE benchmark.

- {cite:t}`DIETZ20241` {cite}`fernandezvillaverde2025climate,vanderploegrezai2026climate`, recent surveys on climate macroeconomics.

- {cite:t}`kubler2025using`, Pareto-improving carbon-tax design.

- {cite:t}`friedlDeep2023`, deep uncertainty quantification methodology.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

```{exercise}
:label: ex-ch11-1

**[Computational\] ECS sensitivity.** Holding all other DICE parameters at calibration, vary the equilibrium climate sensitivity over the {cite:t}`sherwood2020assessment` likely range (2.6–3.9°C) and report the implied SCC at $t=0$. Then repeat over the broader IPCC AR6 very-likely range (2–5°C). How much of the SCC variation is driven by the high-ECS tail? State your classical baseline (e.g., a Nordhaus-2017 grid-search SCC at the central ECS calibration) before reaching for the DEQN.
```

```{exercise}
:label: ex-ch11-2

**[Core\] Sobol decomposition.** For a function $q(\theta_1, \theta_2, \theta_3) = \theta_1\theta_2 + \theta_3^2$ with i.i.d. uniform inputs $\theta_i \sim \mathcal{U}[0,1]$, compute the Sobol first- and total-effect indices analytically. Verify with a $10\,000$-sample SALib estimate.
```

```{exercise}
:label: ex-ch11-3

**[Computational\] ACE benchmark.** Using the closed-form ACE expression for the optimal carbon tax, compute the SCC at calibration and compare with the DEQN-trained DICE solution. Quantify the discrepancy in % terms. State your classical baseline (the ACE closed form itself, evaluated at the chapter's calibration) before reaching for the DEQN.
```

```{exercise}
:label: ex-ch11-4

**[Advanced/project\] Pareto-improving tax design.** Building on the OLG-IAM of {ref}`sec-olg_iam`, search over linear tax-and-transfer rules $\vartheta = (\vartheta_{\mathrm{tax}}, \omega)$ with $\vartheta_{\mathrm{tax}} = (\vartheta_0, \vartheta_E)$ on cumulative emissions and transfer shares $\omega \in \Delta^{A-1}$, $A=12$, such that $\tilde U_t(\vartheta) \ge U_t$ for all cohorts. How tight is the Pareto frontier? Then repeat with $\omega$ held at the BAU/declining benchmark $\bar\omega$ and observe how much welfare is left on the table when transfers are not endogenous.
```

```{exercise}
:label: ex-ch11-5

**[Computational\] Carbon-cycle warm-up.** Run notebook [`01_Climate_Exercise.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_01_Climate_Exercise.ipynb) end-to-end. (a) Report the avoided warming and avoided damages at year 2100 under a 50% mitigation rule relative to BAU. (b) Inspect the carbon-cycle transition matrix and identify which reservoir (atmospheric, upper ocean, lower ocean) has the longest timescale. (c) Explain in two sentences why a quadratic damage-loss function $D(T) = \pi_2 T^2$ is insufficient for tail-risk assessment.
```

```{exercise}
:label: ex-ch11-6

**[Advanced/project\] Deterministic CDICE-DEQN reproduction.** Open notebook [`02_DICE_DEQN_Library_Port.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_02_DICE_DEQN_Library_Port.ipynb) and train the deterministic solver to convergence. Verify the following quantities against the reference solution, using the tolerances stated in the notebook's verification gate: $T^{\mathrm{AT}}(2100)$, $M^{\mathrm{AT}}(2100)$, $\mu(2100)$, and the SCC at 2015, 2100, and 2300.
```

```{exercise}
:label: ex-ch11-7

**[Advanced/project\] Stochastic SCC fan chart.** In notebook [`03_Stochastic_DICE_DEQN.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_03_Stochastic_DICE_DEQN.ipynb), vary the AR(1) productivity volatility $\sigma_z$ over $\{0.005, 0.01, 0.02\}$ and plot the resulting SCC fan charts. How does the right-tail of the SCC distribution at 2100 scale with $\sigma_z$? Compare your finding with the qualitative message of {cite:t}`caiSocialCostCarbon2019`. State your classical baseline (e.g., a deterministic-DICE SCC at the same calibration, or a certainty-equivalent SCC) before reaching for the DEQN.
```

```{exercise}
:label: ex-ch11-8

**[Advanced/project\] Tipping-point regime-switching damages.** This exercise temporarily switches from the additive damage-fraction convention $\Omega^N(T) = \pi_1 T + \pi_2 T^2$ used in the chapter body and in notebook 02 (Eq. {eq}`eq-damage_nordhaus`) to the multiplicative *retained-output* convention $\Omega^{\mathrm{ret}}(T) = 1/(1 + \pi_2 T^2)$, the form discussed in {ref}`sec-dice_damages` as an alternative, so that the regime-switching modification below has a single multiplicative knob. First make this substitution in notebook [`02_DICE_DEQN_Library_Port.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_02_DICE_DEQN_Library_Port.ipynb) (which ships with the additive form $\Omega^N$) and re-calibrate $\pi_2$ so that the retained-output form matches the additive baseline at $T = 2.5\,{}^\circ\mathrm{C}$, i.e. choose $\pi_2$ such that $1 - 1/(1+\pi_2 T^2) = \pi_2^{N} T^2$ at $T = 2.5\,{}^\circ\mathrm{C}$ with $\pi_2^N = 0.00236$ from {numref}`tab-dice_calibration`. Then replace the smooth retained-output factor $\Omega^{\mathrm{ret}}(T) = 1/(1 + \pi_2 T^2)$ with a regime-switching specification. At each step, with hazard rate $\lambda_\mathrm{TP}(T) = \lambda_0 + \lambda_1\max(0, T - T_\mathrm{thresh})$, an irreversible tipping event occurs. If the event has occurred, multiply the damage term in the denominator by $D_\mathrm{TP}=1.5$, so retained output becomes $\Omega^{\mathrm{TP}}(T)=1/(1+D_\mathrm{TP}\pi_2T^2)$. Calibrate $\lambda_0 = 0.001$, $\lambda_1 = 0.05$, $T_\mathrm{thresh} = 2.0\,{}^\circ\mathrm{C}$. Retrain the DEQN solver and report (i) SCC at $t = 0$ under the regime-switching specification vs. the smooth baseline; (ii) the time path of optimal abatement $\mu_t$ in both cases; (iii) the unconditional probability of a tipping event by 2100. Sweep $T_\mathrm{thresh}$ over $\{1.5, 2.0, 2.5\}\,{}^\circ\mathrm{C}$ and plot the SCC against the threshold. Discuss the policy implications: a lower threshold raises the SCC by what factor, and what does this imply for near-term tax design under deep uncertainty about $T_\mathrm{thresh}$ itself?
```

```{exercise}
:label: ex-ch11-9

**[Advanced/project\] Real options value of waiting.** Consider a stylized two-period climate–policy decision. At $t = 0$ the planner does not know the equilibrium climate sensitivity $\mathrm{ECS}$, with prior $\mathrm{ECS} \sim \mathcal{U}([\mathrm{ECS}_L, \mathrm{ECS}_H])$ where $\mathrm{ECS}_L = 2$, $\mathrm{ECS}_H = 5$ $^\circ\mathrm{C}$. Two choices: (a) **act now**, abate at level $\mu \in [0,1]$ with cost $\Theta(\mu) = \theta\mu^2$; (b) **wait**, abate at $t=1$ after observing a noisy signal $\widehat{\mathrm{ECS}} = \mathrm{ECS} + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)$. Damages are $D(\mu, \mathrm{ECS}) = \alpha\,\mathrm{ECS}\cdot(1 - \mu)$. The planner minimizes expected total cost. (i) Derive closed-form expressions for the optimal $\mu^\star$ and the expected total cost under each strategy. (ii) Define the *value of waiting* as the cost difference: $\mathrm{VoW} = \mathbb{E}[\mathrm{cost}_\mathrm{wait}] - \mathbb{E}[\mathrm{cost}_\mathrm{now}]$. Show that as the signal becomes informative ($\sigma_\varepsilon \to 0$), $\mathrm{VoW}$ becomes negative (waiting is preferred): more information allows better decisions. (iii) Now add an irreversibility wedge $\eta\,\mu_1^2$ paid only on the wait branch, penalizing deferred action (e.g., capital stock accumulates carbon faster while waiting, so the wait-period abatement is more costly than an equivalent abatement at $t = 0$). Show that for sufficiently large $\eta$, $\mathrm{VoW}$ is positive (act now is preferred), even with substantial learning. Connect this trade-off to the Bayesian-learning section of the chapter and to the broader literature on real options in climate policy.
```

```{exercise}
:label: ex-ch11-10

**[Core\] Non-stationary DEQN on a 1D toy.** Implement the three modifications of {ref}`sec-nsdeqn_algo` on a one-dimensional non-stationary problem. The toy economy: a planner picks $u_t \in \mathbb R$ to minimize $\sum_{t=0}^{T_{\max}-1} \bigl[(x_t - x^\ast_t)^2 + r u_t^2\bigr] + \lambda_T (x_{T_{\max}} - x^\ast_{T_{\max}})^2$ subject to the linear-Gaussian law of motion $x_{t+1} = \alpha\,x_t + u_t + g_t + \sigma\,\varepsilon_{t+1}$, $\varepsilon_{t+1}\sim\mathcal N(0,1)$, with deterministic drift $g_t = g_0 + g_1 t$ and a calendar-time target path $x^\ast_t = a + b t$. Set $T_{\max} = 50$, $\alpha = 0.95$, $g_0 = 0.02$, $g_1 = 0.001$, $r = 0.1$, $\sigma = 0.05$, $\lambda_T = 5$, $a = 0$, $b = 0.04$. (i) Derive the closed-form LQ-Riccati policy and state your classical baseline before reaching for the DEQN. (ii) Train a small neural network $u_t = \mathcal N_\rho(x_t, \tau_t)$ with $\tau_t = 1 - \exp(-\vartheta\,t)$, sampling initial states from a uniform prior on $[a - 1, a + 1]$, stratifying the mini-batch over ten calendar-time bins of $[0, T_{\max}]$, and adding the terminal penalty $\lambda_T (x_{T_{\max}} - x^\ast_{T_{\max}})^2$ to the loss. (iii) Verify that the trained network reproduces the LQ-Riccati baseline within 1% in the mean-squared-action metric. (iv) Ablate each of the three modifications in turn (drop $\tau_t$ from the input; remove stratification; remove the terminal penalty) and report which ablation hurts most as a function of $T_{\max}$.
```

[^1]: The numerical claims in this paragraph quote the headline results of {cite:t}`friedlDeep2023`; consult that paper for the precise figures and the underlying calibration grid.

[^2]: **Notation note.** We use $\psi$ for the IES and $\gamma_u$ for risk aversion throughout this section, following {cite:t}`friedlDeep2023`. The IRBC chapter ({ref}`ch-irbc`) used $\gamma$ for the IES under the bundled CRRA-IES convention; here in the Epstein–Zin block we deliberately decouple the two parameters, so the symbol switch is intentional. The CRRA limit is recovered at $\gamma_u = 1/\psi$.

[^3]: Variance-share ranges quoted from {cite:t}`friedlDeep2023`; the spread reflects different points along the planner's horizon and different damage-function specifications.

[^4]: All numerical coefficients, welfare gains, and damage-quantile values in this subsection are quoted from {cite:t}`kubler2025using`; consult the paper for the source tables and figures.
