---
title: "Solutions and Guidance for Exercises"
label: app-solutions
---

This appendix provides worked solutions for analytical end-of-chapter exercises and guidance for coding exercises. Coding exercises (those that ask the reader to modify a notebook or measure a numerical quantity) are not treated as fixed numerical answers; their outputs depend on hardware, seeds, calibration choices, and notebook versions, and should be reported from the corresponding companion notebook listed in the *Execution Map*. For exercises that mix analytical work with a small numerical check, the analytical part is solved and the numerical component is described as a verification task.

Exercises are referenced by their stable label `ex:ch`$N$`:`$M$, where $N$ is the chapter number and $M$ is the position within that chapter's exercise list. Each solution opens with a back-pointer to the exercise statement so the reader can scan the question first.

(sol-ch1)=
## {ref}`ch-intro`: Introduction to Machine Learning and Deep Learning
**{prf:ref}`ex-ch1-1`: Backprop on a 2-layer net.** Let $z = w_1 x + b_1$, $a = \mathrm{ReLU}(z)$, $\hat y = w_2 a$, $\ell = (\hat y - y)^2$. Reverse-mode chain rule:

```{math}
:enumerated: false

\begin{aligned}
\frac{\partial \ell}{\partial \hat y} &= 2(\hat y - y), &
\frac{\partial \ell}{\partial w_2} &= 2(\hat y - y)\,a, \\
\frac{\partial \ell}{\partial a} &= 2(\hat y - y)\,w_2, &
\frac{\partial a}{\partial z} &= \mathbb{1}[z > 0], \\
\frac{\partial \ell}{\partial w_1} &= 2(\hat y - y)\,w_2\,\mathbb{1}[z>0]\,x.
\end{aligned}
```

Plugging in $x=2$, $y=1$, $w_1 = w_2 = b_1 = 0.5$: $z = 1.5$, $a = 1.5$, $\hat y = 0.75$, $\ell = 0.0625$. Hence $\partial\ell/\partial w_2 = 2(-0.25)(1.5) = -0.75$ and $\partial\ell/\partial w_1 = 2(-0.25)(0.5)(1)(2) = -0.5$. A two-line PyTorch script (`torch.autograd.grad`) returns the same numbers to machine precision; this is the simplest non-trivial sanity check that the chain-rule derivation matches what AD computes.

**{prf:ref}`ex-ch1-2`: MSE vs. MLE.** For Gaussian errors $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, the log-likelihood is

```{math}
:enumerated: false

\log p(y_{1:n} \mid x_{1:n}; \beta) = -\frac{1}{2\sigma^2}\sum_i (y_i - \beta x_i)^2 - \tfrac{n}{2}\log(2\pi\sigma^2).
```

Maximizing over $\beta$ (the only $\beta$-dependent term) is identical to minimizing $\sum_i (y_i - \beta x_i)^2$, i.e. OLS. For Laplace errors with scale $b$, $p(\varepsilon) \propto \exp(-|\varepsilon|/b)$, so the log-likelihood is proportional to $-\sum_i |y_i - \beta x_i|$ and the MLE solves a least-absolute-deviations regression (median regression). Squaring penalizes outliers quadratically, which is suboptimal under Laplace tails because a single large residual dominates the loss; the median estimator is robust precisely because $|\cdot|$ grows linearly.

**{prf:ref}`ex-ch1-3`: Activation choice for a PINN.** A ReLU network $\hat y(x) = \sum_k w_k\,\mathrm{ReLU}(a_k x + b_k) + c$ is piecewise-linear: between consecutive kink points $x = -b_k/a_k$ it is affine in $x$, so $\hat y''(x) = 0$ on every open subinterval. At a kink, the second distributional derivative is a Dirac measure, not a classical pointwise value. A strong-form PDE residual such as the Black–Scholes operator

```{math}
:enumerated: false

\partial_t V + \tfrac{1}{2}\sigma^2 S^2\,V_{SS} + r S\,V_S - r V \;=\; 0
```

must hold pointwise (a.e.) for almost every $S$; with a piecewise-linear $V$, the term $V_{SS}$ vanishes a.e. in the classical sense, and the residual reduces to $\partial_t V + r S V_S - rV$, which has no nontrivial solution that respects the actual boundary conditions of an option-pricing problem. Concrete example: $\hat y(x) = \mathrm{ReLU}(x) + \mathrm{ReLU}(-x) = |x|$ has $\hat y''(x) = 0$ for every $x \neq 0$ in the classical sense. Smooth activations (tanh, Swish, GELU, softplus) are $C^\infty$ and produce well-defined pointwise second derivatives, which is why the PINN literature recommends them whenever the PDE is of order $\ge 2$.

**{prf:ref}`ex-ch1-4`: Adam vs. AdamW.** Write the gradient at step $t$ as $g_t = \nabla_\theta \ell(\theta_t)$. With $L_2$ regularization *added to the loss*, $\ell^{L_2}(\theta) = \ell(\theta) + \tfrac{\lambda}{2}\|\theta\|^2$, so the effective gradient becomes $\tilde g_t = g_t + \lambda \theta_t$. Adam's update applied to $\tilde g_t$ is

```{math}
:enumerated: false

m_t = \beta_1 m_{t-1} + (1-\beta_1)\tilde g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)\tilde g_t^2,\quad
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}.
```

The key observation is that the weight-decay contribution $\lambda \theta_t$ enters $m_t$ and $v_t$ *through the same EWMA averaging as the data gradient*, so the effective shrinkage applied to $\theta$ is $\eta\lambda \theta_t$ scaled by $1/(\sqrt{\hat v_t}+\varepsilon)$, which differs across coordinates. Parameters with large historical gradients receive small effective decay; parameters with small gradients receive large decay. AdamW decouples the two:

```{math}
:enumerated: false

m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
```

Now the multiplicative shrinkage $(1 - \eta\lambda)$ acts uniformly on all parameters, independent of the adaptive denominator. Numerically the two updates differ by a factor of $1/(\sqrt{\hat v_t}+\varepsilon)$ on the decay term; AdamW preserves the textbook intuition "weight decay shrinks weights at the same rate everywhere."

**{prf:ref}`ex-ch1-5`: RNN forward pass by hand.** With $W_h = 0.5\,I_2$, $W_x = (1,0)^\top$, $h_0 = (0,0)^\top$:

```{math}
:enumerated: false

\begin{aligned}
h_1 &= \tanh\!\big((0,0)^\top + (1,0)^\top\big) = (\tanh 1,\, 0)^\top \approx (0.7616,\, 0)^\top,\\
h_2 &= \tanh\!\big(0.5\,h_1 + (0,0)^\top\big) = (\tanh(0.3808),\,0)^\top \approx (0.3637,\,0)^\top,\\
h_3 &= \tanh\!\big(0.5\,h_2 + (1,0)^\top\big) = (\tanh(1.1818),\,0)^\top \approx (0.8275,\,0)^\top.
\end{aligned}
```

Outputs: $\hat y_t = W_y h_t$ gives $\hat y_1 \approx 0.7616$, $\hat y_2 \approx 0.3637$, $\hat y_3 \approx 0.8275$ (only the first hidden coordinate is excited, so the second column of $W_y$ is irrelevant).

For the gradient, write $h_t = \tanh(z_t)$ with $z_t = W_h h_{t-1} + W_x x_t$. Then

```{math}
:enumerated: false

\frac{\partial \hat y_3}{\partial x_1} = W_y\,\frac{\partial h_3}{\partial z_3}\,W_h\,\frac{\partial h_2}{\partial z_2}\,W_h\,\frac{\partial h_1}{\partial z_1}\,W_x,
```

where $\partial h_t/\partial z_t = \mathrm{diag}(1-\tanh^2(z_t))$. Numerically, $\tanh'(1)\approx 0.4200$, $\tanh'(0.3808)\approx 0.8678$, $\tanh'(1.1818)\approx 0.3155$, so

```{math}
:enumerated: false

\frac{\partial \hat y_3}{\partial x_1} \approx 1 \cdot 0.3155 \cdot 0.5 \cdot 0.8678 \cdot 0.5 \cdot 0.4200 \cdot 1 \approx 0.0288.
```

The decay rate is the product of two distinct effects: (i) the spectral radius of $W_h$ (here $0.5$), which contributes one factor of $W_h$ per recurrent step ($T-1$ multiplicative copies in the chain above), and (ii) the saturation of $\tanh'(\cdot) \in (0,1]$, which contributes a second multiplicative factor per step. Setting (ii) aside, with $\|W_h\|_2 = 0.5 < 1$ the gradient magnitude already decays at least as fast as $0.5^{T-1}$, i.e. exponentially in the sequence length. This is the canonical vanishing-gradient pathology of vanilla RNNs ({ref}`sec-sequence_models`). LSTM and GRU gates address (i) by allowing the recurrent Jacobian's eigenvalues to stay close to one without making training unstable; effect (ii) is intrinsic to bounded activations and is mitigated by skip connections (and architectural choices that keep activations away from saturation regimes), not by gating.

**{prf:ref}`ex-ch1-6`: Attention by hand.** With $q_i = k_i = v_i = x_i$ and $x = (0,1,0.5)$, the score matrix $S_{ij} = q_i k_j$ is

```{math}
:enumerated: false

S = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0.5 \\ 0 & 0.5 & 0.25\end{pmatrix}.
```

Softmaxing each row and using $e^0 = 1$, $e^{0.25} \approx 1.284$, $e^{0.5}\approx 1.649$, $e^1 \approx 2.718$:

```{math}
:enumerated: false

\begin{aligned}
a_1 &= \tfrac{1}{3}(1,1,1), & o_1 &= \tfrac{1}{3}(0 + 1 + 0.5) = 0.5, \\
a_2 &\approx (0.186,\, 0.506,\, 0.307), & o_2 &\approx 0.186\cdot 0 + 0.506\cdot 1 + 0.307\cdot 0.5 \approx 0.660, \\
a_3 &\approx (0.254,\, 0.419,\, 0.327), & o_3 &\approx 0.254\cdot 0 + 0.419\cdot 1 + 0.327\cdot 0.5 \approx 0.583.
\end{aligned}
```

Each attention vector $a_i$ is on the simplex (entries non-negative, summing to one), so $o_i = \sum_j a_{ij} v_j$ is a convex combination of the values, lying in $[0, 1]$. Token 2, the largest input, attends most strongly to itself ($a_{22}\approx 0.51$); token 3 also attends most to token 2 because the inner products $q_3 k_2 = 0.5$ exceed the self-score $q_3 k_3 = 0.25$. Token 1 has zero query magnitude, so its row is uniform: with no signal to discriminate on, attention defaults to a uniform average over the values.

**{prf:ref}`ex-ch1-7`: TensorBoard optimizer comparison.** Coding exercise. Expected qualitative behavior on a small classification task: SGD with momentum is often slower to reduce training loss but can generalize competitively when the learning-rate schedule is well tuned; Adam often reaches a low training loss fastest, but its validation curve can diverge from the training curve sooner than for SGD or AdamW; AdamW often sits between the two. The actual crossover point is a notebook output and should be reported from the run.

(sol-ch2)=
## {ref}`ch-deqn`: Deep Equilibrium Nets
**{prf:ref}`ex-ch2-1`: Closed-form Brock–Mirman.** Conjecture $V(K, z) = A\log K + B\log z + C$. The Bellman equation

```{math}
:enumerated: false

V(K,z) = \max_{K'}\Bigl\{\log\bigl(zK^\alpha - K'\bigr) + \beta\,\mathbb{E}_{z'\mid z}\!\bigl[V(K', z')\bigr]\Bigr\}
```

yields the FOC $1/(zK^\alpha - K') = \beta A/K'$, which gives the constant savings rate

```{math}
:enumerated: false

s^\star = \frac{K'}{zK^\alpha} = \frac{\beta A}{1 + \beta A}.
```

Substituting the conjecture back and matching the $\log K$ coefficient produces $A = \alpha + \beta A\alpha$, hence $A = \alpha/(1-\alpha\beta)$. Plugging this $A$ into $s^\star$:

```{math}
:enumerated: false

s^\star \;=\; \frac{\beta\,\alpha/(1-\alpha\beta)}{1 + \beta\,\alpha/(1-\alpha\beta)} \;=\; \frac{\alpha\beta}{1-\alpha\beta+\alpha\beta} \;=\; \alpha\beta.
```

The DEQN parameterizes $s_t = \sigma\!\bigl(\mathcal{N}_\rho(K_t, z_t)\bigr)$. Once converged, the average sigmoid output across the ergodic set should equal $\alpha\beta$ (typical calibrations $\alpha=0.36$, $\beta=0.96$ give $s^\star \approx 0.346$). Any systematic deviation indicates either insufficient training or a quadrature / sampling bias.

**{prf:ref}`ex-ch2-2`: Hard vs. soft constraints.** With a softplus head on consumption alone, $C_t = \mathrm{softplus}(\mathcal{N}_\rho(K_t, z_t)) > 0$ is guaranteed, but next-period capital is then *defined* as the residual $K_{t+1} = z_t K_t^\alpha - C_t$, which is unconstrained in sign. At random initialization the network output is approximately $\mathcal{N}(0, 1)$, so $C_t$ is approximately $\mathrm{softplus}(0) \approx 0.69$ on average but can be much larger.

Concrete failure: take $K = 1$, $z = 1$, $\alpha = 0.36$, so $zK^\alpha = 1$. If the network produces an output of, say, $5$, then $C = \mathrm{softplus}(5) \approx 5.007$ and $K_{t+1} = 1 - 5.007 = -4.007 < 0$. The next iterate $K_{t+1}^{\alpha-1}$ is then complex, and the loss explodes.

The sigmoid-savings parameterization $s_t = \sigma(\mathcal{N}_\rho) \in (0,1)$ avoids this entirely: $K_{t+1} = s_t z_t K_t^\alpha > 0$ and $C_t = (1-s_t) z_t K_t^\alpha > 0$ both hold by construction, regardless of the network's raw output. This is the simplest example of the broader principle that *architectural* feasibility encodings dominate *loss-based* ones whenever the constraint can be written as a closed-form algebraic identity.

**{prf:ref}`ex-ch2-3`: Path averaging vs. conditional expectation.** On the simulated path, the path-averaged residual is $\bar r_T(\theta) = T^{-1}\sum_{t=1}^T r(\theta, x_t)$ with $\{x_t\}$ generated by the model dynamics. Under ergodicity, Birkhoff's theorem gives

```{math}
:enumerated: false

\bar r_T(\theta) \;\xrightarrow{T \to \infty}\; \int r(\theta, x)\,\mu(dx) \;=\; \mathbb{E}_\mu[r(\theta, x)] \quad\text{a.s.},
```

where $\mu$ is the stationary distribution. The conditional-expectation residual at a fixed point $x_t$, $\E[r(\theta, x_{t+1}) \mid x_t]$, is a different object: it integrates only over the conditional shock distribution, holding the current state fixed.

Their connection: if one sweeps the conditional residual over $x_t \sim \mu$ and averages, the result coincides with $\mathbb{E}_\mu[r(\theta, x)]$ by the law of iterated expectations. In other words, path averaging *combines* sampling over states (the outer expectation) and the implicit Monte Carlo integration over shocks (one shock per simulated step). Conditional expectation evaluates the inner integral exactly via quadrature but still requires a way to draw states $x_t$.

Bias–variance trade-off at finite $T_{\text{sim}}$: the path-averaged loss has variance $O(1/T_{\text{sim}})$ from finite-sample noise but no bias. An exact-quadrature residual has zero stochastic noise on the inner integral but its outer-state coverage depends on the sampling scheme; with mini-batch SGD over the ergodic set, the variance comes from the random batch, not the integration. In the chapter the deterministic quadrature is preferred whenever the shock dimension is low ($d \lesssim 5$), and pathwise residuals dominate once $d$ is high enough that explicit quadrature becomes expensive.

**{prf:ref}`ex-ch2-4`: Brock–Mirman with Gauss–Hermite.** The Euler equation in Brock–Mirman with $\delta = 1$, log utility, and AR(1) productivity $\ln z' = \varrho\ln z + \sigma_z\varepsilon'$, $\varepsilon' \sim \mathcal{N}(0,1)$, is

```{math}
:enumerated: false

\frac{1}{C_t}
   \;=\;
   \beta\,\E\!\left[\frac{\alpha\,z'\,K_{t+1}^{\alpha - 1}}{C_{t+1}} \,\Big|\, K_t, z_t\right].
```

Replace the expectation by a Gauss–Hermite rule. After the change of variables $\varepsilon' = \sqrt{2}\,\xi$ that absorbs the normalization $1/\sqrt{\pi}$, the $Q$-point GH quadrature is

```{math}
:enumerated: false

\E[h(\varepsilon')] \;\approx\; \frac{1}{\sqrt{\pi}} \sum_{q=1}^{Q} w_q\,h\!\bigl(\sqrt{2}\,\xi_q\bigr),
```

with classical nodes $\xi_q$ and weights $w_q$ that satisfy $\sum_q w_q = \sqrt{\pi}$. {numref}`tab-gh5_nodes` lists the five-point rule used in this exercise.

````{table}
:name: tab-gh5_nodes

Five-point Gauss–Hermite nodes and weights for the convention $\E[h(\varepsilon)] \approx \pi^{-1/2}\sum_q w_q h(\sqrt{2}\xi_q)$. The weights sum to $\sqrt{\pi}$ before the outside normalization.

| $q$ | $\xi_q$ | $w_q$ |
|---:|---:|---:|
| 1 | $-2.0202$ | $0.0200$ |
| 2 | $-0.9586$ | $0.3936$ |
| 3 | $0.0000$ | $0.9453$ |
| 4 | $+0.9586$ | $0.3936$ |
| 5 | $+2.0202$ | $0.0200$ |
````

Substituting, the residual at a state $(K_t, z_t)$ becomes

```{math}
:enumerated: false

r(\theta; K_t, z_t)
   \;=\;
   1 \;-\; \frac{\beta\,\alpha\,C_t}{\sqrt{\pi}}\sum_{q=1}^{5} w_q\,\frac{z_q'\,K_{t+1}^{\alpha-1}}{C_{t+1}^{q}},
```

where $z_q' = \exp(\varrho\ln z_t + \sigma_z\sqrt{2}\,\xi_q)$ and $C_{t+1}^q$ is the consumption obtained by feeding $(K_{t+1}, z_q')$ into the network. Numerically, comparing this $5$-point sum with a high-draw Monte Carlo estimate on the same residual is a useful diagnostic: agreement should be close for smooth integrands and small shock variance, but the realized discrepancy is an output of the check rather than a fixed benchmark.

**{prf:ref}`ex-ch2-5`: Monomial rule by hand (Stroud-3 at $d=4$).** Equation {eq}`eq-stroud3` places nodes at $\bm{x}_k^\pm = \pm\sqrt{d}\,\bm{e}_k$ for $k = 1, \dots, d$, with equal weights $1/(2d) = 1/8$. At $d=4$, the eight nodes are $\pm 2 \bm{e}_k$ for $k=1,2,3,4$.

*First moment.* $\E[\varepsilon_i']_{\text{rule}} = (1/8)\sum_k [(\sqrt{d}\,\bm{e}_k)_i + (-\sqrt{d}\,\bm{e}_k)_i] = 0$ by $\pm$-cancellation.

*Second moment.* $(\varepsilon_i')^2$ is non-zero only at $\pm\sqrt{d}\,\bm{e}_i$, where it equals $d$. Two such nodes contribute $2d$, weighted by $1/(2d)$, giving exactly $1$.

*Cross moment.* $\varepsilon_i'\varepsilon_j'$ for $i \neq j$ is zero at every node because each node has only one nonzero coordinate. So the rule returns $0$, the true value.

*Third moment.* $(\varepsilon_i')^3$ at $\pm\sqrt{d}\bm{e}_i$ equals $\pm d^{3/2}$; the two values cancel. Returns $0$, exact.

*Fourth moment.* $(\varepsilon_i')^4$ at $\pm\sqrt{d}\bm{e}_i$ equals $d^2$, both signs. Two nodes contribute $2d^2$, weighted by $1/(2d)$, gives $d$. At $d=4$, the rule returns $4$, while the true value $\E[\varepsilon_i'^{\,4}] = 3$.

*Linear bias growth.* In general the rule reports $d$ for the fourth moment, so the relative error $(d - 3)/3$ is linear in $d$: $33\%$ at $d=4$, $67\%$ at $d=5$, and $100\%$ at $d=6$.

*When does this matter?* The Euler residual is a smooth function of the next-period shock $\varepsilon'$. Taylor-expanding around the conditional mean, the leading bias term is the Hessian of the residual with respect to $\varepsilon'$, which probes the second moment, exact under Stroud-3. Fourth-moment bias enters only at the next order, scaled by the integrand's fourth derivative. For moderate CRRA curvature and thin-tailed shocks this term can be small relative to classroom residual tolerances, but it is a diagnostic to check rather than a universal bound. The bias becomes material when (i) the integrand has heavy fourth-order content (e.g., very risk-averse preferences or fat-tailed shocks), or (ii) the shock dimension is large enough that the relative error $(d-3)/3$ exceeds the residual tolerance one targets. This is the threshold at which the monomial rule should be replaced by Stroud-5 ($2d^2 + 1$ nodes) or QMC.

**{prf:ref}`ex-ch2-6`: Loss-kernel selection.** *Definitions for reference.* MSE: $\frac{1}{N}\sum_i r_i^2$. MAE: $\frac{1}{N}\sum_i |r_i|$. Huber($\delta$): quadratic for $|r| \le \delta$, linear above. Pinball loss at $\tau$: $L_\tau(r) = \max(\tau r,\, (\tau - 1)r)$, whose minimizer is the $\tau$-quantile of $r$. CVaR at $\alpha$: the expected value of $r$ conditional on $r$ exceeding its $\alpha$-quantile. Log-cosh: $\sum_i \log\cosh(r_i)$, smooth and quadratic near zero, linear in tails.

\(a\) **Huber loss**: "smooth, quadratic near zero, linear in tails" is the literal definition. MAE is also linear-in-tails but is not differentiable at $r=0$, so its gradient flips discontinuously and the optimizer stalls; log-cosh is smooth and shares the asymptotic shape with Huber but has no tunable threshold. Huber($\delta$) gives the cleanest control of where the regime change happens.

\(b\) **CVaR at $\alpha = 0.99$**. The CVaR loss optimizes the conditional mean above the $99$th-percentile threshold, which is exactly what the regulator audits. The pinball loss at $\tau = 0.99$ would only target the $99$th-percentile residual itself, not the conditional average above it; if the residuals have a fat right tail, the pinball-trained policy can have arbitrarily large worst-case $1\%$ residuals. CVaR is the right primitive for "worst-case mean" control.

\(c\) **MAE** (or equivalently pinball at $\tau = 0.5$). By construction MAE's first-order condition is solved at the median, not the mean: $\partial/\partial r |r| = \mathrm{sign}(r)$, so the gradient contribution is $\pm 1$ per residual, regardless of magnitude. This is exactly what the desideratum asks for, no single tail residual dominates the gradient. Huber would also down-weight tails but still tracks the mean below the threshold; the user explicitly wants median-targeting.

**{prf:ref}`ex-ch2-7` and {prf:ref}`ex-ch2-8` (statements: p. , p. ).** These are coding exercises (notebook [`lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb)); reference outputs and timing curves are in the companion repository. Qualitative anchors: in Ex. {prf:ref}`ex-ch2-7`, the per-epoch wall time of tensor-product Gauss–Hermite ($Q^d$ nodes) should grow exponentially in $d$ while Stroud-3 ($2d$ nodes) grows linearly, with a crossover that is visible already at $d=4$–$5$ on a single GPU; the relative Euler error should track the integration accuracy of each rule, with Stroud-3 inheriting the fourth-moment bias of {ref}`sec-monomial_cubature`. In Ex. {prf:ref}`ex-ch2-8`, swapping Swish for $\tanh$ typically slows time-to-converge by tens of percent on a smooth problem like Brock–Mirman because $\tanh$ saturates faster, but final accuracy is comparable; convergence should still hold under the same hyperparameters.

(sol-ch3)=
## {ref}`ch-irbc`: The International Real Business Cycle Model
**{prf:ref}`ex-ch3-1`: Fischer–Burmeister.** For the forward direction, suppose $a \ge 0$, $b \ge 0$, $ab = 0$. Without loss of generality $b = 0$; then $\Phi(a, 0) = a + 0 - \sqrt{a^2 + 0} = a - |a| = 0$ since $a \ge 0$. By symmetry the same holds when $a = 0$.

For the reverse direction, suppose $\Phi(a,b) = 0$, i.e. $a + b = \sqrt{a^2 + b^2}$. The right-hand side is non-negative, so $a + b \ge 0$. Squaring: $(a+b)^2 = a^2 + b^2 \Rightarrow 2ab = 0 \Rightarrow ab = 0$. Combined with $a + b \ge 0$ and $ab = 0$, the only possibility is one of $a, b$ being zero and the other non-negative, i.e. $a, b \ge 0$ and $ab = 0$. $\square$

The level set $\Phi(a, b) = 0$ is the union of the non-negative $a$-axis and the non-negative $b$-axis (an L-shape in the $(a,b)$ plane). The smoothed variant $\Phi_\varepsilon(a,b) = a + b - \sqrt{a^2 + b^2 + \varepsilon^2}$ rounds the corner at the origin: at $(a,b) = (0,0)$ one finds $\Phi_\varepsilon(0,0) = -\varepsilon \neq 0$, while far from the origin $\sqrt{a^2 + b^2 + \varepsilon^2} \approx \sqrt{a^2 + b^2}$ and the smoothed level set converges to the unsmoothed L-shape. In a numerical setting the smoothing eliminates the gradient kink at the origin, which is convenient for AD but distorts the strict KKT zero set by an $O(\varepsilon)$ amount.

*(c) Gradient direction.* At an interior point of the open positive quadrant, $\nabla\Phi(a,b) = (1 - a/\sqrt{a^2+b^2},\, 1 - b/\sqrt{a^2+b^2})$. At $(a,b) = (1,1)$ this evaluates to $(1-1/\sqrt 2,\, 1-1/\sqrt 2) \approx (0.293, 0.293)$, both components strictly positive. The raw gradient $\nabla\Phi$ therefore points northeast, *away* from the L-shaped zero set, so the bare residual $\Phi$ on its own is not the right object for SGD to descend on. What does descend toward the zero set is the squared loss $\Phi^2$: by the chain rule $\nabla(\Phi^2) = 2\Phi\,\nabla\Phi$, and inside the open positive quadrant $\Phi(a,b) > 0$, so $-\nabla(\Phi^2) = -2\Phi\,\nabla\Phi$ has both components strictly negative at $(1,1)$ and points southwest, i.e. back toward the closer feasible axis. This is the operative observation for training: SGD on $\Phi^2$ is what pulls infeasible iterates back to the L; the raw gradient $\nabla\Phi$ on its own would push them away.

*(d) Sign convention.* Replacing $\Phi$ by $-\Phi$ leaves the squared loss untouched: $(-\Phi)^2 = \Phi^2$, so the gradient field of the loss and the SGD trajectory are unchanged. In particular, the sign convention $a + b - \sqrt{a^2+b^2}$ versus $\sqrt{a^2+b^2} - a - b$ is irrelevant once the residual is squared, and the operative quantity for SGD is $-\nabla(\Phi^2)$ rather than $-\nabla\Phi$. The L-shaped zero set is invariant under sign flip; only the value of $\Phi$ at off-zero points changes (it flips sign), and squaring removes that distinction.

**{prf:ref}`ex-ch3-2`: State-space scaling.** For an IRBC with $N$ symmetric countries, the state vector contains $(K^j, z^j)_{j=1}^N$ for a total of $2N$ components. The network outputs the next-period capital vector $(k^{j\prime})_{j=1}^N$, the irreversibility multipliers $(\mu^j)_{j=1}^N$, and the resource-constraint shadow price $\lambda$, for $2N + 1$ outputs in total. Country-level consumption is recovered algebraically from $\lambda$ via the consumption-sharing FOC and is therefore not a separate output. The loss has $N$ Euler residuals, $N$ Fischer–Burmeister residuals from the irreversibility complementarity, and $1$ aggregate-resource-constraint residual, $2N + 1$ components total, matching the $2N + 1$ outputs. Each Euler residual is an expectation over a $(N+1)$-dimensional shock vector (one country-specific innovation plus one aggregate, as in equation {eq}`eq-irbc_tfp`).

Tensor-product Gauss–Hermite at $Q=3$ costs $3^{N+1}$ evaluations per residual. Setting $3^{N+1} > 10^4$ gives $N + 1 > \log_3(10^4) \approx 8.38$, so $N \ge 8$ already exceeds the threshold, and $N = 9$ overshoots by a factor of nearly $6$ (cost $3^{10} = 59\,049$).

The Stroud-3 rule has $2(N+1)$ nodes per residual. Setting $2(N+1) < 100$ gives $N \le 48$, comfortable for any IRBC dimension actually used in practice. At $N = 50$ the monomial cost is $102$ nodes; the tensor cost is $3^{51} \approx 2.15 \times 10^{24}$ evaluations. This four-order-of-magnitude gap at $N = 10$ and twenty-order-of-magnitude gap at $N = 50$ is the operational reason DEQNs use Stroud-3 by default once $N \gtrsim 5$.

**{prf:ref}`ex-ch3-3`: Two-phase training.** At a randomly initialized network, the policy outputs $k^{j\prime}$ are not coordinated with the resource constraint. Suppose the network produces $k^{j\prime}$ values that, summed and combined with country-$j$ consumption, exceed total output: $\sum_j (k^{j\prime} + c^j + \Gamma^j) > \sum_j Y^j$. The implied state on the next simulated step has $k^{j\prime} < (1-\delta) k^j$ for some country (irreversibility violated), or $c^j < 0$ (negative consumption), or both. Concretely, take $N = 2$, $A_\mathrm{tfp} = 1$, $\delta = 0.025$, $k^1 = k^2 = 1$, and a random network output $(k^{1\prime}, k^{2\prime}) = (0.5, 0.5)$ (instead of the symmetric $(1, 1)$ steady state). Capital has dropped by $50\%$ in one step, so $I^j = k^{j\prime} - (1-\delta)k^j = 0.5 - 0.975 = -0.475 < 0$, violating irreversibility. Even before the irreversibility check fires, the implied consumption $c^j = Y^j - I^j - \Gamma^j$ becomes huge, and in the next step the marginal utility $u'(c)$ is essentially zero, so the Euler residual gradient is uninformative.

Phase 1 (uniform sampling on a wide box of states, with Euler residuals computed against *any* feasible policy guess) gives the optimizer signal to bring outputs into the feasible region before any simulation is attempted. Once the policy is in a feasible neighborhood, Phase 2 (simulation-based sampling on the ergodic set) refines accuracy. Without Phase 1, the simulation in Phase 2 starts in regions where the policy is grossly infeasible, the loss explodes, and gradient descent diverges.

**{prf:ref}`ex-ch3-4`: Adjustment-cost partials and Tobin's Q.** Write $g^j \equiv k^{j\prime}/k^j - 1$. Then $\Gamma^j = (\kappa/2)\,k^j (g^j)^2$. The partials are

```{math}
:enumerated: false

\begin{aligned}
\frac{\partial \Gamma^j}{\partial k^{j\prime}} &= \frac{\kappa}{2}\,k^j \cdot 2 g^j \cdot \frac{1}{k^j} \;=\; \kappa\,g^j \;=\; \kappa\!\left(\frac{k^{j\prime}}{k^j} - 1\right), \\
\frac{\partial \Gamma^j}{\partial k^j} &= \frac{\kappa}{2}(g^j)^2 + \frac{\kappa}{2}\,k^j \cdot 2 g^j \cdot \!\left(-\frac{k^{j\prime}}{(k^j)^2}\right) \\
&= \frac{\kappa}{2}\,(g^j)^2 - \kappa\,g^j\!\left(\frac{k^{j\prime}}{k^j}\right)
   \;=\; \frac{\kappa}{2}\!\left[\bigl(g^j\bigr)^2 - 2(1+g^j) g^j\right] \\
&= -\frac{\kappa}{2}\!\left[(g^j)^2 + 2 g^j\right] \;=\; \frac{\kappa}{2}\!\left[1 - \bigl(\tfrac{k^{j\prime}}{k^j}\bigr)^{\!2}\right],
\end{aligned}
```

matching equation {eq}`eq-irbc_adjcost_derivs`.

At the steady state, $k^{j\prime} = k^j$ so $g^j = 0$. Both partials vanish: $\partial\Gamma^j/\partial k^{j\prime} = 0$ and $\partial\Gamma^j/\partial k^j = 0$. The adjustment cost itself $\Gamma^j(k^j, k^j) = 0$, and its first-order presence in the resource constraint and the Euler equation drops out. The deterministic steady state of the IRBC with adjustment costs is therefore identical to the frictionless steady state, $k_\mathrm{ss}$ being pinned down by $\beta(1 - \delta + \zeta A_\mathrm{tfp} k^{\zeta-1}) = 1$.

The expression $\partial\Gamma^j/\partial k^{j\prime} = \kappa g^j$ is the marginal cost of investing one more unit and is the IRBC's analogue of *Tobin's marginal Q*: large positive $g^j$ (rapid expansion) creates a large investment wedge, raising the effective per-unit cost of capital next period. Higher $\kappa$ flattens the response of investment to a productivity shock: a large adjustment cost makes the planner spread the response of $k^{j\prime}$ across several periods rather than absorbing the shock in one big move, slowing convergence to the new steady state. In the limit $\kappa \to \infty$, capital adjusts only infinitesimally each period and the dynamics become arbitrarily slow.

**{prf:ref}`ex-ch3-5`: Complete-markets risk sharing.** The consumption-sharing condition {eq}`eq-irbc_consumption` reads $c_t^j = (\lambda_t/\tau^j)^{-\gamma_j}$. Therefore

```{math}
:enumerated: false

\frac{c_t^i}{c_t^j} \;=\; \!\left(\frac{\lambda_t}{\tau^i}\right)^{\!-\gamma_i}\!\Bigl/\!\left(\frac{\lambda_t}{\tau^j}\right)^{\!-\gamma_j}
   \;=\;
   \!\left(\frac{\tau^i}{\lambda_t}\right)^{\!\gamma_i}\!\!\left(\frac{\lambda_t}{\tau^j}\right)^{\!\gamma_j}
   \;=\;
   (\tau^i)^{\gamma_i}\,(\tau^j)^{-\gamma_j}\,\lambda_t^{\,\gamma_j - \gamma_i}.
```

\(i\) With homogeneous IES $\gamma_i = \gamma_j = \gamma$, the $\lambda_t$ exponent vanishes and the ratio collapses to a time-invariant constant:

```{math}
:enumerated: false

\frac{c_t^i}{c_t^j} \;=\; \!\left(\frac{\tau^i}{\tau^j}\right)^{\!\gamma}.
```

Since $\log(c_t^i/c_t^j)$ is constant, $\Delta\log c_t^i = \Delta\log c_t^j$, the perfect risk-sharing prediction of {cite:t}`backus1992international`: cross-country consumption growth is perfectly correlated (correlation $= 1$).

\(ii\) With heterogeneous IES $\gamma_i \neq \gamma_j$, the exponent $\gamma_j - \gamma_i$ is non-zero and $\lambda_t$ enters the ratio. As shocks move the planner's shadow price, the consumption ratio fluctuates: low-IES countries' consumption is less sensitive to $\lambda_t$ than high-IES countries'. But the log growth rate is still

```{math}
:enumerated: false

\Delta \log c_t^j = -\gamma_j\,\Delta\log\lambda_t .
```

Thus, for positive $\gamma_i,\gamma_j$, any pair of country consumption-growth rates is a positive scalar multiple of the same aggregate shock $\Delta\log\lambda_t$. The correlation remains one; heterogeneous IES changes relative consumption-growth volatility, not the correlation, in this complete-markets planner allocation.

\(iii\) The empirical consumption-correlation puzzle is that real-world cross-country consumption growth correlations sit well below the near-perfect correlations implied by the complete-markets benchmark. Heterogeneous IES alone does not break the common-shadow-price structure; closing the gap with the data requires incomplete markets, wedges, nontraded goods, preference nonseparabilities, or other frictions that break full insurance.

**{prf:ref}`ex-ch3-6` and {prf:ref}`ex-ch3-7` (statements: p. , p. ).** These are notebook exercises ([`lecture_05_05_IRBC_Exercise.ipynb`](notebooks/lecture_05_nas_loss_normalization/lecture_05_05_IRBC_Exercise.ipynb)); reference solutions are in the notebook itself, behind the "attempt first" dividers. Qualitative anchors: in Ex. {prf:ref}`ex-ch3-6`, the closed-form steady state $k_\mathrm{ss} = \bigl[(1/\beta - 1 + \delta)/(\zeta A_\mathrm{tfp})\bigr]^{1/(\zeta-1)}$ falls in all three scenarios — a higher $\delta$ raises the required net return $1/\beta - 1 + \delta$, a lower $\beta$ raises $1/\beta$, and a lower $\zeta$ shrinks the multiplicative term while (since $\zeta - 1 < 0$) steepening diminishing returns — so $k_\mathrm{ss}$ is decreasing in $\delta$, increasing in $\beta$, and increasing in $\zeta$ around the baseline, with $c_\mathrm{ss} = A_\mathrm{tfp} k_\mathrm{ss}^\zeta - \delta k_\mathrm{ss}$ following by substitution. In Ex. {prf:ref}`ex-ch3-7`, inverse-loss weighting $w_i \propto 1/\ell_i$ equalizes the per-component contributions $w_i \ell_i$, so the smallest-magnitude residuals (here the Fischer–Burmeister terms) receive the largest weight; the speed-up is largest when a component is small because it is *hard to fit*, and the scheme can hurt when a component is small because it is *already satisfied by construction* (e.g., a hard-coded resource constraint), in which case up-weighting it merely amplifies noise.

(sol-ch4)=
## {ref}`ch-nas`: Neural Architecture Search and Loss Normalization
**{prf:ref}`ex-ch4-1`: Random vs. grid.** With two hyperparameters and only one "important" axis, a $3\times 3$ grid uses $9$ candidates but only $3$ distinct values along the important axis. If the near-optimal interval has length fraction $p$ and its location relative to the grid is unknown, the grid hit probability is approximately $\min\{3p,1\}$ when $p$ is small. Random search at $9$ evaluations samples $9$ independent values along the important axis, so its hit probability is

```{math}
:enumerated: false

1 - (1-p)^9 .
```

For $p=0.05$, the grid probability is approximately $0.15$, while random search gives $1-0.95^9\approx 0.37$.

The general principle: with $n$ evaluations and only $r \ll d$ important axes, random search effectively gives $n$ independent draws on those $r$ axes (since the irrelevant axes don't matter), while grid search wastes most of its budget on the irrelevant axes' marginals. This is the projection argument of {cite:t}`bergstra2012random`: when the loss landscape is anisotropic, random search dominates grid search.

**{prf:ref}`ex-ch4-2`: Bayesian optimization toy problem.** This is a coding exercise. A grid with step size $0.01$ over $[-1,2]$ has $301$ points, so the qualitative benchmark is that BO should require far fewer objective evaluations when the GP posterior and acquisition function identify the promising region early. The exact evaluation count is a notebook output and depends on the initial design, acquisition optimizer, random seed, and stopping rule.

**{prf:ref}`ex-ch4-3`: Hyperband budget allocation.** Hyperband with $R = 81$, $\eta = 3$ runs a ladder of brackets indexed by $s = s_{\max}, s_{\max}-1, \dots, 0$, where $s_{\max} = \lfloor\log_\eta R\rfloor = 4$. Each bracket starts with $n_s = \lceil (s_{\max}+1)\,\eta^s / (s+1)\rceil$ candidates trained for $r_s = R / \eta^s$ resource each, then runs Successive Halving with reduction factor $\eta$. {numref}`tab-hyperband_r81_eta3` works out the resulting schedule.

````{table}
:name: tab-hyperband_r81_eta3

Hyperband schedule for maximum resource $R=81$ and reduction factor $\eta=3$. Each row reports the successive-halving rungs inside one bracket and the total resource consumed by that bracket.

| $s$ | **SHA rungs $(n\times r)$** | **Total** |
|---:|---|---:|
| $4$ | $81{\times}1 \to 27{\times}3 \to 9{\times}9 \to 3{\times}27 \to 1{\times}81$ | $405$ |
| $3$ | $34{\times}3 \to 11{\times}9 \to 3{\times}27 \to 1{\times}81$ | $363$ |
| $2$ | $15{\times}9 \to 5{\times}27 \to 1{\times}81$ | $351$ |
| $1$ | $8{\times}27 \to 2{\times}81$ | $378$ |
| $0$ | $5{\times}81$ | $405$ |
````

Within each bracket, Successive Halving reduces candidates by $\eta$ at each rung and increases the resource per surviving candidate by $\eta$. Therefore the total cost must sum all rungs, not only the first rung. With the floor/ceil schedule above, total Hyperband budget is

```{math}
:enumerated: false

405 + 363 + 351 + 378 + 405 = 1902
```

resource units, just below the loose worst-case bound $(s_{\max}+1)^2 R = 25 \cdot 81 = 2025$.

A naive "train all $n_0 = 27$ candidates to full $R = 81$" costs $27 \cdot 81 = 2187$ resource units. Hyperband is only moderately cheaper in total resource here, but it screens a much larger initial pool: across the five brackets, the total number of first-rung candidates is $81 + 34 + 15 + 8 + 5 = 143$, vs. $27$ for the naive scheme. Its advantage is adaptive allocation: many more candidates are sampled, but only a small subset receives large training budgets.

**{prf:ref}`ex-ch4-4`: Loss balancing.** Let $\ell_i^{(t)}$ have magnitude $L_i \in \{10^0, 10^{-2}, 10^{-4}\}$ and per-component gradient norm $\|\nabla\ell_i\|$. If we crudely model gradient norm as scaling linearly with loss magnitude (true for, e.g., quadratic losses far from the optimum), then $\|\nabla \ell_i\| \propto L_i$. Equalizing gradient contributions $\lambda_i \|\nabla\ell_i\|$ requires $\lambda_i \propto 1/L_i$, i.e. $(\lambda_1, \lambda_2, \lambda_3) \propto (1, 10^2, 10^4)$, which after normalization becomes

```{math}
:enumerated: false

(\lambda_1, \lambda_2, \lambda_3) \;=\; \frac{1}{1 + 10^2 + 10^4}\,(1, 10^2, 10^4) \;\approx\; (10^{-4},\, 10^{-2},\, 1).
```

The scheme breaks down when the gradients are correlated: $\langle \nabla\ell_i, \nabla\ell_j\rangle \neq 0$ means that scaling up $\lambda_3$ to "boost" $\ell_3$ also moves $\theta$ along the $\nabla\ell_1$ direction, changing $\ell_1$. The "equal contribution" targeted by the fixed weights is no longer a fixed point: each parameter update changes the local gradient geometry, and weights tuned at one iteration become wrong at the next. Adaptive schemes (ReLoBRaLo, GradNorm) re-tune the $\lambda_i$ at every step, recovering the equalization in a way that fixed weights cannot.

**{prf:ref}`ex-ch4-5`: Pareto frontier geometry.** *(i)* Differentiate $\mathcal{L}(\theta;\lambda) = \lambda(\theta-a)^2 + (1-\lambda)(\theta-b)^2$ in $\theta$ and set to zero: $2\lambda(\theta - a) + 2(1-\lambda)(\theta - b) = 0$, hence

```{math}
:enumerated: false

\theta^\star(\lambda) \;=\; \lambda a + (1-\lambda) b.
```

*(ii)* Substituting back:

```{math}
:enumerated: false

\ell_1^\star(\lambda) \;=\; (\theta^\star - a)^2 \;=\; (1-\lambda)^2 (b-a)^2,
   \qquad
   \ell_2^\star(\lambda) \;=\; (\theta^\star - b)^2 \;=\; \lambda^2 (b-a)^2.
```

*(iii)* Take square roots: $\sqrt{\ell_1^\star} = (1-\lambda)(b-a)$ and $\sqrt{\ell_2^\star} = \lambda(b-a)$. Adding:

```{math}
:enumerated: false

\sqrt{\ell_1^\star} + \sqrt{\ell_2^\star} \;=\; (b - a),
```

independent of $\lambda$. This is a convex curve in $(\ell_1, \ell_2)$ space (a quarter of an astroid-like arc) running from $(0, (b-a)^2)$ at $\lambda = 1$ to $((b-a)^2, 0)$ at $\lambda = 0$.

*(iv)* At $\lambda = 1/2$, $\theta^\star = (a+b)/2$, the midpoint, and $\ell_1^\star = \ell_2^\star = (b-a)^2/4$. This sits on the symmetric axis of the front.

*(v)* In the one-dimensional toy problem the trade-off is completely described by the curve above. In a neural network, however, $\theta$ is high-dimensional and the descent direction for fixed scalar weight $\lambda$ is

```{math}
:enumerated: false

-\bigl[\lambda \nabla \ell_1(\theta) + (1-\lambda)\nabla \ell_2(\theta)\bigr].
```

If $\langle\nabla\ell_1,\nabla\ell_2\rangle>0$, reducing one component tends to reduce the other; if the inner product is negative, progress on one component can increase the other. Because this geometry changes along the training path, a fixed scalar weight can balance progress at one iterate and become badly unbalanced later. ReLoBRaLo responds to this by increasing the weight of losses whose *relative loss progress* has lagged behind. GradNorm is the related method that targets gradient magnitudes directly by trying to balance $\|w_k\nabla\ell_k\|$ across components.

**{prf:ref}`ex-ch4-6`: ReLoBRaLo vs. GradNorm.** This is a coding exercise. Qualitatively: GradNorm requires one extra backward pass per component per step (to compute $\|\nabla \ell_k\|$), so wall-clock per epoch is roughly $K\times$ slower than ReLoBRaLo for $K$ components. In return, GradNorm achieves tighter gradient balance, which matters when component magnitudes are not a faithful proxy for gradient magnitudes (e.g., when the loss landscape is anisotropic). For the standard PINN losses studied in this script ($K = 2$ or $3$ components, typically scaled by physical units), ReLoBRaLo's loss-magnitude proxy is usually good enough and the extra cost of GradNorm is not warranted; for losses with strongly heterogeneous Hessians (e.g., HJB residuals coupled with KFE residuals where the two operators have different stiffness), GradNorm's direct gradient measurement can give a meaningful speedup.

**{prf:ref}`ex-ch4-7`: HPO vs. full NAS decision.** *Sketch.* The four cells are roughly:

- *(a, i) RTX 3060 + fixed-topology MLP search:* **Random Search with Successive Halving.** Search space is small (a few thousand candidate combinations), the GPU is too small for graph-level NAS, and SH amortizes the budget across many candidates by killing weak ones early. Bayesian Optimization helps marginally but adds GP-fitting overhead that is not worth it on a single GPU.

- *(a, ii) RTX 3060 + graph-level NAS:* **Random Search.** Full DARTS-style NAS would not fit in $12$ GB of VRAM (the supernetwork search needs several model copies in memory simultaneously); a small fixed pool of architectures evaluated at low resource is the only feasible option.

- *(b, i) A100 + fixed-topology MLP search:* **Bayesian Optimization**. The A100's compute headroom makes the per-step BO overhead negligible, and the search space's smooth landscape (continuous learning rate, ordinal depth/width) is exactly where GP surrogates dominate Random Search.

- *(b, ii) A100 + graph-level NAS:* **Full graph-level NAS** (DARTS or evolutionary). The A100's memory and compute support the supernetwork training that DARTS requires; the search space has too many discrete connectivity choices for any HPO method to cover.

The general rule: budget grows $\to$ smarter methods become affordable; search-space size grows $\to$ the marginal benefit of smarter methods grows; per-method overhead per evaluation needs to fit inside one GPU step or it eats into the budget itself.

(sol-ch5)=
## {ref}`ch-olg`: Overlapping Generations Models with DEQNs
**{prf:ref}`ex-ch5-1`: OLG market clearing for $A=3$.** With three cohorts (young $h=1$, middle $h=2$, old $h=3$), the budget constraints are

```{math}
:enumerated: false

\begin{aligned}
c^1_t &= w_t \,\ell^1 - k^2_{t+1}, \\
   c^2_t &= w_t \,\ell^2 + R_t k^2_t - k^3_{t+1}, \\
   c^3_t &= w_t \,\ell^3 + R_t k^3_t,
\end{aligned}
```

where $w_t, R_t$ are equilibrium prices and $\ell^h$ are exogenous lifecycle labor endowments (the old cohort consumes its capital). Two Euler equations determine the savings of the young and middle cohorts:

```{math}
:enumerated: false

\begin{aligned}
u'(c^1_t) &= \beta\,\mathbb{E}_t[u'(c^2_{t+1})\,R_{t+1}], \\
u'(c^2_t) &= \beta\,\mathbb{E}_t[u'(c^3_{t+1})\,R_{t+1}].
\end{aligned}
```

The market-clearing condition closes the system:

```{math}
:enumerated: false

k^2_{t+1} + k^3_{t+1} \;=\; K_{t+1},
```

where $K_{t+1}$ is aggregate capital.

*Equation count*: three budget constraints (used to eliminate consumption), two Euler equations, one capital-market-clearing identity. The budget constraints are bookkeeping, so the network outputs the two cohort savings $(k^2_{t+1}, k^3_{t+1})$ and is trained on the two Euler residuals; aggregate capital $K_{t+1} = k^2_{t+1} + k^3_{t+1}$ then determines next-period prices $(w_{t+1}, R_{t+1})$ algebraically through the firm FOCs. The unknown count (two savings) matches the Euler-residual count (two), and the market-clearing identity is built into the definition of $K_{t+1}$.

**{prf:ref}`ex-ch5-2`: KKT under FB.** For agent $h$ with borrowing constraint $k'^h \ge 0$ and Lagrange multiplier $\lambda^h \ge 0$, the KKT system is

```{math}
:enumerated: false

k'^h \ge 0, \qquad \lambda^h \ge 0, \qquad k'^h \cdot \lambda^h = 0.
```

These three conditions are encoded by the single Fischer–Burmeister equation $\Phi(\lambda^h, k'^h) = 0$ with $\Phi(a,b) = a + b - \sqrt{a^2 + b^2}$. The Euler equation

```{math}
:enumerated: false

u'(c^h_t) - \beta\,\mathbb{E}_t[u'(c^{h+1}_{t+1})\,R_{t+1}] - \lambda^h_t \;=\; 0
```

contributes a second residual. Squared and summed:

```{math}
:enumerated: false

\mathcal{L}^h(\theta) \;=\; \bigl[\text{Euler residual}\bigr]^2 + \bigl[\Phi(\lambda^h, k'^h)\bigr]^2.
```

At any KKT point, both squared terms vanish exactly: the Euler residual is zero by definition, and $\Phi = 0$ encodes complementarity. This is what "vanishes exactly at the KKT point" means: there is no $\varepsilon$ smoothing or penalty parameter, the loss is zero on the equilibrium and strictly positive off it. Compare with a quadratic penalty $\bigl[\max(0, -k'^h)\bigr]^2 + \mu \bigl[\max(0, -\lambda^h)\bigr]^2 + (k'^h \lambda^h)^2$: this also vanishes at the KKT point but the multiplier $\mu$ has to be tuned, whereas FB has no parameter.

**{prf:ref}`ex-ch5-3`: Hump-shaped lifecycle.** A hump-shaped labor-income profile $\ell^h$ peaks at middle age and declines toward retirement. The lifecycle savings policy $k'^h$ inherits this hump for two reasons. (i) *Consumption smoothing*: agents with high current income $w \ell^h$ relative to lifetime average save heavily to fund retirement years when $\ell^h$ drops. (ii) *Time-varying borrowing constraint*: young agents have low income, want to borrow against future earnings, are constrained by $k'^h \ge 0$, so they save little; middle-aged agents are unconstrained and save the most; old agents dis-save toward death.

The expected shape: $k'^h \approx 0$ for the very young (constrained), peaks around age 40–50 (middle of working life), declines toward retirement, drops to zero for the oldest cohort that does not save into the next period. In notebook [`lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`](notebooks/lecture_08_olg_models_deqns/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb), plotting the trained network's $k'^h$ against cohort age $h$ should reveal exactly this single-peak shape; the position of the peak depends on the calibration of $(\beta, \delta, A_\mathrm{tfp})$ and on the lifecycle labor profile.

**{prf:ref}`ex-ch5-4`: Hard aggregation layer.** Coding exercise. The current analytic notebook already clears the capital market by defining $K_{t+1}$ as the sum of predicted cohort savings. The alternative hard-layer variant is useful when the network has a separate aggregate-capital head. Implementation sketch: output a positive scalar $\widehat K_{t+1}$ and unnormalized cohort scores $(z^2, \dots, z^A)$; apply softmax along the cohort axis, $s^h = \mathrm{softmax}(z^h)$; rescale to capital:

```{math}
:enumerated: false

k_{t+1}^h \;=\; \widehat K_{t+1} \cdot s^h, \qquad
   \sum_{h=2}^{A} k_{t+1}^h \;=\; \widehat K_{t+1}\;\;\text{by construction}.
```

The market-clearing residual $\sum_h k^h - \widehat K_{t+1}$ is identically zero up to floating-point precision. The comparison with the current notebook should therefore focus on Euler residuals and wall-clock time: the hard layer removes one possible inconsistency but also changes the parameterization, so faster convergence is an empirical question rather than a mathematical guarantee. In multi-asset settings each exact clearing condition needs its own accounting layer, which is why the 56-agent benchmark enforces bond-market clearing as an explicit residual instead.

**{prf:ref}`ex-ch5-5`: Bond pricing in equilibrium.** Cohort $h$'s Euler equation for capital is

```{math}
:enumerated: false

u'(c^h_t) \;=\; \beta\,\mathbb{E}_t\!\bigl[u'(c^{h+1}_{t+1})\,R_{t+1}\bigr],
```

and for the riskless bond that costs $p_t$ today and pays one unit of consumption next period,

```{math}
:enumerated: false

u'(c^h_t)\,p_t \;=\; \beta\,\mathbb{E}_t\!\bigl[u'(c^{h+1}_{t+1})\bigr].
```

The second equation gives directly

```{math}
:enumerated: false

p_t \;=\; \frac{\beta\,\mathbb{E}_t[u'(c^{h+1}_{t+1})]}{u'(c^h_t)}.
```

Identifying the stochastic discount factor $M_{t,t+1} = \beta\,u'(c^{h+1}_{t+1})/u'(c^h_t)$, the same expression reads $p_t = \mathbb{E}_t[M_{t,t+1}]$.

For the risk-premium decomposition, divide the capital Euler by $u'(c^h_t)$ and use the covariance identity $\mathbb{E}_t[XY] = \mathbb{E}_t[X]\mathbb{E}_t[Y] + \mathrm{Cov}_t(X,Y)$:

```{math}
:enumerated: false

1 \;=\; \mathbb{E}_t[M_{t,t+1}\,R_{t+1}]
       \;=\; \mathbb{E}_t[M_{t,t+1}]\,\mathbb{E}_t[R_{t+1}] + \mathrm{Cov}_t(M_{t,t+1}, R_{t+1}).
```

Substituting $p_t = \mathbb{E}_t[M_{t,t+1}]$:

```{math}
:enumerated: false

\frac{1}{p_t} \;=\; \mathbb{E}_t[R_{t+1}] + \frac{\mathrm{Cov}_t(M_{t,t+1}, R_{t+1})}{p_t},
```

which after rearrangement gives the textbook risk-premium decomposition: the gap between expected gross capital return and the riskless rate equals minus the SDF–return covariance, scaled by $1/p_t$.

*When the collateral constraint binds.* If the collateral constraint is active, with non-negative KKT multiplier $\mu^h_t$ entering the bond FOC with coefficient $\kappa$ (the same $\kappa$ that controls the constraint $k'^h + \kappa b'^h \ge 0$), the bond Euler equation becomes

```{math}
:enumerated: false

u'(c^h_t)\,p_t \;=\; \beta\,\mathbb{E}_t\!\bigl[u'(c^{h+1}_{t+1})\bigr] + \kappa\,\mu^h_t,
```

so the equilibrium bond price is

```{math}
:enumerated: false

p_t \;=\; \frac{\beta\,\mathbb{E}_t[u'(c^{h+1}_{t+1})] + \kappa\,\mu^h_t}{u'(c^h_t)}.
```

The unconstrained SDF expression $p_t = \mathbb{E}_t[M_{t,t+1}]$ is recovered when $\mu^h_t = 0$. The multiplier wedge raises $p_t$, equivalently lowers the implicit safe rate that $1/p_t$ tracks, because the constrained agent values one extra unit of bond consumption tomorrow more than the unconstrained agent. In the 56-agent benchmark, the cohorts most likely to bind are the youngest (lowest income, highest desire to borrow against future earnings), so any wedge $\kappa\,\mu^h_t$ from the binding-cohort population enters the cross-sectional pricing equation.

*Why no bond residual in 6-agent OLG?* In the analytic 6-agent calibration there is only one asset (capital). Bonds are absent, so no separate market-clearing residual is needed. The single-asset Euler equation pins down the implicit safe rate via $1/p = \mathbb{E}[R]$ minus the appropriate covariance, but no quantity needs to clear because no bond is traded.

**{prf:ref}`ex-ch5-6` and {prf:ref}`ex-ch5-7` (statements: p. , p. ).** Coding exercises. Both call for $4$–$5$ retraining runs of the 56-agent benchmark and a binding-frequency / steady-state diagnostic. The binding indicator should be based on small slack and a positive multiplier, not on a large complementarity residual: at a well-trained KKT solution the product residual is close to zero both when a constraint binds and when it is slack. Expect borrowing and collateral constraints to bind mostly for young cohorts; as $\kappa$ rises, the lower bound $b'^h \ge -k'^h/\kappa$ becomes tighter, so negative bond positions should shrink and cross-cohort bond dispersion should typically fall. The equilibrium price response is a general-equilibrium object and should be read from the retrained models rather than imposed analytically.

(sol-ch6)=
## {ref}`ch-young`: Heterogeneous Agents and Young's Method
**{prf:ref}`ex-ch6-1`: Mean-preserving lottery.** Place mass $\omega$ at $k_n$ and $1-\omega$ at $k_{n+1}$. Mean preservation:

```{math}
:enumerated: false

\omega\,k_n + (1-\omega)\,k_{n+1} \;=\; k'.
```

Solving for $\omega$:

```{math}
:enumerated: false

\omega \;=\; \frac{k_{n+1} - k'}{k_{n+1} - k_n}.
```

This is well-defined for $k_n \le k' \le k_{n+1}$ since the denominator is positive and the numerator lies in $[0, k_{n+1} - k_n]$, ensuring $\omega \in [0,1]$.

*Mass conservation*: the two probabilities sum to one, $\omega + (1-\omega) = 1$, so total mass is preserved exactly under this redistribution. Equivalently, the weight $\omega$ is the unique linear interpolation weight that makes the discrete two-point distribution have mean $k'$, which is what defines Young's redistribution operator on a fixed grid.

Higher-moment matching is impossible with a two-point split unless $k'$ coincides with a grid point: any non-degenerate two-point distribution with mean $k'$ and support $\{k_n, k_{n+1}\}$ has variance $\omega(1-\omega)(k_{n+1}-k_n)^2 > 0$, while the original (delta) distribution at $k'$ has zero variance. This residual variance is the price of the discretization, and it shrinks as the grid is refined.

**{prf:ref}`ex-ch6-2`: Closed-form bracketing on log-spaced grids.** Coding exercise. Algorithm sketch: with grid $k_n = e^{x_0 + n\Delta x} - c$, the bracket index for a query $k'$ is

```{math}
:enumerated: false

n \;=\; \mathrm{floor}\!\left(\frac{\log(k' + c) - x_0}{\Delta x}\right).
```

This is $\mathcal{O}(1)$ per query (one log + one floor), independent of grid size $N$. By contrast, `numpy.searchsorted` costs $\mathcal{O}(\log N)$ per query (binary search). The speed difference is hardware- and implementation-dependent, so the coding exercise should report the measured wall-clock ratio on the vectorized batch rather than treating a fixed multiplier as universal.

**{prf:ref}`ex-ch6-3`: Approximate aggregation, scope.** The KS log-linear rule is built on the empirical observation that mean capital $K_t$ alone is a sufficient statistic for forecasting $K_{t+1}$ in the standard Aiyagari–Krusell–Smith calibration. This approximate aggregation breaks when the cross-sectional distribution carries information beyond its first moment that materially affects equilibrium prices.

*Counterexample 1: multiple assets with switching liquidity.* Add a second asset (say a corporate bond) with state-dependent liquidity: in good times agents trade both assets freely; in bad times the bond becomes illiquid. Now the share of wealth held in the bond, plus the bond–capital correlation in the cross-section, both determine prices, and neither can be summarized by mean capital alone.

*Counterexample 2: heterogeneous discount factors.* If agents differ in $\beta$ and the cross-sectional distribution of $\beta$ is dynamic (e.g., new entrants have different $\beta$), then mean capital understates the dispersion, and the equilibrium interest rate depends on which subpopulation holds the marginal unit of capital.

*Why higher moments do not always rescue.* Adding the variance to the forecasting rule helps with smooth perturbations but cannot capture multi-modal distributions, regime-switching, or non-monotone responses to skewness. The fundamental issue is that the master equation requires *full* cross-sectional information whenever prices are non-linear in the distribution; truncating to any finite set of moments is exact only in the linear-pricing case.

**{prf:ref}`ex-ch6-4`: Sequence-space vs. histogram DEQN.** Coding exercise. Empirically: with truncation horizon $T = 80$ and the chapter's reference calibration, the sequence-space residual after training matches the histogram-based DEQN to within a factor of $1.2$–$1.5$ on the same model. The sequence-space variant generalizes *worse* to a much longer test horizon ($T_\mathrm{test} \gg 80$): its input is a fixed-length shock window, so it inherits an error floor of order $\max\{|\varrho|,|\alpha|\}^{T}$ that training cannot remove and that is paid afresh at every date of a long test path. The histogram variant does not have this issue because its state is the current distribution, a genuine Markov state. The trade-off favors sequence-space when the cross-sectional distribution is intrinsically high-dimensional (e.g., multiple assets, multi-cohort wealth), at which point storing $T$ shock realizations is cheaper than discretizing the distribution.

**{prf:ref}`ex-ch6-5`: DeepSets permutation invariance.** *(i)* Let $\pi$ be a permutation of $\{1, \dots, N\}$. The aggregator's $m$-th component is

```{math}
:enumerated: false

m_t^m(\pi \cdot s) \;=\; \sum_{i=1}^N g_\theta^m\bigl(s_t^{\pi(i)}\bigr) \;=\; \sum_{j=1}^N g_\theta^m(s_t^j),
```

where the second equality is just a re-indexing of the sum (since addition is commutative). Therefore $\bm m_t(\pi \cdot s) = \bm m_t(s)$, exactly invariant.

*(ii)* The policy $\pi_\rho(s_t^i; \bm m_t, a_t)$ is a function of agent $i$'s own state $s_t^i$, the population summary $\bm m_t$, and the aggregate exogenous state $a_t$. Under a permutation $\pi$, agent $\pi(i)$'s individual state is now $s_t^{\pi(i)}$, while $\bm m_t$ and $a_t$ are unchanged (by the result in (i) for $\bm m_t$). Therefore the policy at slot $i$ in the permuted economy equals the policy of agent $\pi(i)$ in the original economy, i.e. the policy travels with its own agent rather than with its position: *equivariance*.

*(iii)* {cite:t}`zaheer2017deep` prove that any continuous permutation-invariant function $f: \mathbb{R}^{d \times N} \to \mathbb{R}$ on sets of fixed cardinality $N$ can be written as $f(s_1, \dots, s_N) = \rho\bigl(\sum_{i=1}^N g(s_i)\bigr)$ for some continuous functions $g, \rho$. This is the universal-approximation result for permutation-invariant DeepSets.

*Implication for DeepHAM.* Since the equilibrium price functional is permutation-invariant in agents (anonymous markets), DeepHAM's parameterization can in principle approximate any continuous price/policy functional of the cross-sectional distribution to arbitrary accuracy, provided the inner network $g_\theta$ has enough capacity and the moment vector $\bm m_t$ is rich enough. The number of moments $M$ plays the role of the encoder's bottleneck dimension: empirically, $M = 1$–$3$ suffices for Krusell–Smith-class economies, consistent with the chapter's report that DeepHAM with one learned moment matches the histogram DEQN.

**{prf:ref}`ex-ch6-6` and {prf:ref}`ex-ch6-7` (statements: p. , p. ).** Coding exercises. Guidance for interpreting the outputs:

*{prf:ref}`ex-ch6-6`.* The relevant statistic is the cross-replication sampling variance conditional on the same aggregate path, not the time-series variance of $K_t$ along that path. As $N$ rises, the Monte Carlo standard error should fall at the usual $N^{-1/2}$ rate for smooth aggregate statistics. Young's path has zero sampling variance across replications because the histogram update integrates out the lottery exactly. The MC-vs-Young trade-off depends on the target functional: tail mass (e.g., the bottom-$10\%$ wealth share) decays much more slowly under MC, so the panel size needed to match Young-equivalent precision is an output of the repeated-panel experiment.

*{prf:ref}`ex-ch6-7`.* In the standard KS calibration, the one-moment forecasting rule should already fit $\log K_{t+1}$ extremely well; adding $\log V_t$, with $V_t=\mathrm{Var}_{\mu_t}(k)$, is expected to give only a small incremental gain. This is exactly the empirical observation behind "approximate aggregation". In calibrations with high cross-sectional dispersion (e.g., wide income range or frequent borrowing-constraint binding), the second-moment improvement can become economically visible and should be reported from the run.

(sol-ch7)=
## {ref}`ch-pinn`: Physics-Informed Neural Networks
**{prf:ref}`ex-ch7-1`: Trial-function BC enforcement.** Define $\hat y(x) = \tfrac{2x}{\pi} + x\bigl(\tfrac{\pi}{2} - x\bigr)\,\mathcal{N}_\rho(x)$. Evaluate at the boundaries:

```{math}
:enumerated: false

\hat y(0) \;=\; 0 + 0 \cdot \tfrac{\pi}{2}\cdot \mathcal{N}_\rho(0) \;=\; 0,
   \qquad
   \hat y(\pi/2) \;=\; \tfrac{2}{\pi}\cdot\tfrac{\pi}{2} + \tfrac{\pi}{2}\cdot 0\cdot \mathcal{N}_\rho(\pi/2) \;=\; 1.
```

Both boundary conditions hold for *any* network output $\mathcal{N}_\rho$, so the BCs are encoded in the architecture rather than enforced via the loss.

*Why preferable to a soft penalty?* A soft penalty $\lambda\,(\hat y(0) - 0)^2 + \lambda\,(\hat y(\pi/2) - 1)^2$ in the loss involves a hyperparameter $\lambda$ that must be tuned: too small, and the BC violation is large; too large, and the interior PDE residual is starved of optimization budget. The trial-function enforcement is parameter-free, makes the BC residual identically zero, and reduces the loss to the single PDE-interior term $\frac{1}{N_r}\sum_i |\hat y''(x_i) + \hat y(x_i)|^2$ over the collocation points. This separates the two optimization concerns cleanly; any wall-clock gain should be measured in the notebook rather than assumed.

**{prf:ref}`ex-ch7-2`: ReLU pathology.** A ReLU network is piecewise-linear: between consecutive kinks $x = -b_k/a_k$ it is affine in $x$, so $\partial^2 \hat y/\partial x^2 = 0$ a.e. At a kink, the second distributional derivative is a Dirac delta supported on a measure-zero set. The strong-form Black–Scholes residual

```{math}
:enumerated: false

V_t + \tfrac{1}{2}\sigma^2 S^2 V_{SS} + rS V_S - rV \;=\; 0
```

must hold pointwise for almost every $S$. With ReLU, $V_{SS} = 0$ a.e., so the residual reduces to $V_t + rS V_S - rV$, which has no nontrivial solution that respects the option-pricing boundary condition. The PINN cannot decrease its loss below the order of magnitude of the missing $V_{SS}$ term.

*Weak-form fix.* Multiply both sides by a smooth test function $\varphi(S)$ with compact support on $[S_\mathrm{min}, S_\mathrm{max}]$ and integrate by parts on the $V_{SS}$ term:

```{math}
:enumerated: false

\int_{S_\mathrm{min}}^{S_\mathrm{max}}\! V_{SS}\,\varphi\, dS \;=\; -\int V_S\,\varphi'\, dS \;+\;[V_S\,\varphi]_{S_\mathrm{min}}^{S_\mathrm{max}}.
```

The boundary terms vanish for compactly supported $\varphi$, and the residual now involves only first-order derivatives of $V$. ReLU networks have well-defined first derivatives a.e., so the weak-form PINN can minimize this residual. This is exactly the Galerkin / Deep Galerkin formulation of {cite:t}`sirignano2018dgm`.

**{prf:ref}`ex-ch7-3`: Discrete $\to$ continuous bridge.** Start with

```{math}
:enumerated: false

V(a) = \max_c \bigl[u(c)\,\Delta t + \beta_{\Delta t}\,\mathbb{E}V(a')\bigr],
\qquad
a' = a - c\,\Delta t,
\qquad
\beta_{\Delta t}=e^{-\rho\Delta t}.
```

Then $\beta_{\Delta t}=1-\rho\,\Delta t+O(\Delta t^2)$. The same first-order limit follows from the implicit-Euler convention $\beta_{\Delta t}=1/(1+\rho\Delta t)$.

Taylor-expand $V(a') = V(a - c\,\Delta t)$ around $a$:

```{math}
:enumerated: false

V(a') \;=\; V(a) - V'(a)\,c\,\Delta t + \tfrac{1}{2}V''(a)\,(c\,\Delta t)^2 + O(\Delta t^3).
```

Substitute and use $\beta_{\Delta t} = 1 - \rho\,\Delta t + O(\Delta t^2)$:

```{math}
:enumerated: false

V(a) \;=\; \max_c \Bigl\{u(c)\,\Delta t + \bigl(1 - \rho\Delta t\bigr)\!\bigl[V(a) - V'(a)\,c\,\Delta t + O(\Delta t^2)\bigr]\Bigr\}.
```

Subtract $V(a)$ from both sides and divide by $\Delta t$:

```{math}
:enumerated: false

0 \;=\; \max_c\Bigl\{u(c) - \rho V(a) - V'(a)\,c + O(\Delta t)\Bigr\}.
```

Take $\Delta t \to 0$ and rearrange:

```{math}
:enumerated: false

\rho V(a) \;=\; \max_c \bigl[u(c) - V'(a)\,c\bigr].
```

This is the HJB equation for the consumption-savings problem with no asset return (or with $r$ embedded into $a' = a + ra\Delta t - c\Delta t$, in which case the HJB picks up an additional $V'(a)\,r a$ drift term). The discrete-to-continuous bridge thus shows that the HJB is the formal limit of the discrete Bellman as the time step shrinks, which justifies treating PINNs as the continuous-time analogue of value-function iteration.

**{prf:ref}`ex-ch7-4`, {prf:ref}`ex-ch7-5`, {prf:ref}`ex-ch7-6` (statements: p. , p. , p. ).** Coding exercises. The exact numbers below depend on random seeds, batch sizes, and stopping rules; use them as qualitative checks rather than fixed targets:

*{prf:ref}`ex-ch7-4`.* At small $\lambda$, the BC residual is underweighted and endpoint violations remain visibly large. At very large $\lambda$, the endpoints fit well but the interior residual can stagnate because the optimizer spends most of its gradient budget on the boundary term. The elbow is the smallest $\lambda$ for which further increases mostly improve the boundary metric without improving the interior fit. The hard-BC variant should be the benchmark: it sets the boundary violation to numerical zero by construction and removes the penalty-weight tuning problem.

*{prf:ref}`ex-ch7-5`.* Sobol and Latin Hypercube points usually reduce visible sampling gaps relative to uniform random points. On the smooth manufactured Poisson problem the gain may be modest, because the solution has no boundary layer or interior singularity. Adaptive sampling becomes more valuable when residuals are spatially localized; report the actual collocation count, wall time, and test-grid residual rather than relying on a universal percentage saving.

*{prf:ref}`ex-ch7-6`.* The expected qualitative result is that the strong-form $\tanh$ PINN is the natural baseline for Black–Scholes, because $V_{SS}$ is well-defined by automatic differentiation. A strong-form ReLU network is ill-suited because $V_{SS}=0$ almost everywhere and undefined at kinks. In a weak formulation, integration by parts moves the second derivative off the network and onto the test function, so a ReLU network can be made mathematically admissible. Any empirical comparison should report held-out pricing errors and residual diagnostics from the implemented notebook rather than quoting architecture-independent iteration counts.

**{prf:ref}`ex-ch7-7`: Operator learning vs. PINN.** Eleven independent PINN runs each cost $C_\mathrm{PINN}$ wall-clock seconds, total $11\,C_\mathrm{PINN}$. A single operator-learning or parametric-PINN run trained on $11$ values of $K$ (or a continuous range, sampled in mini-batches) costs $C_\mathrm{op}$. Amortized training wins when $11\,C_\mathrm{PINN} > C_\mathrm{op}$, i.e., $C_\mathrm{op}/C_\mathrm{PINN} < 11$. The crossover scales linearly in the number of distinct $K$ values: with $N$ strikes, operator learning wins whenever $C_\mathrm{op}/C_\mathrm{PINN}<N$. This is the cost-amortization argument that motivates DeepONet-style operator learning {cite:p}`lu2021learning`.

(sol-ch8)=
## {ref}`ch-ct_theory`: Heterogeneous Agent Models in Continuous Time
**{prf:ref}`ex-ch8-1`: Itô on GBM.** Geometric Brownian motion satisfies $dX_t = \mu X_t\,dt + \sigma X_t\,dB_t$. Apply Itô's lemma to $f(x) = \ln x$ with $f'(x) = 1/x$, $f''(x) = -1/x^2$:

```{math}
:enumerated: false

d(\ln X_t) \;=\; f'(X_t)\,dX_t + \tfrac{1}{2} f''(X_t)\,(dX_t)^2
   \;=\; \frac{1}{X_t}\bigl(\mu X_t\,dt + \sigma X_t\,dB_t\bigr) + \tfrac{1}{2}\!\left(-\frac{1}{X_t^2}\right)\sigma^2 X_t^2\,dt.
```

Simplifying:

```{math}
:enumerated: false

d(\ln X_t) \;=\; \bigl(\mu - \tfrac{1}{2}\sigma^2\bigr)\,dt + \sigma\,dB_t.
```

Integrating from $0$ to $t$:

```{math}
:enumerated: false

\ln X_t - \ln X_0 \;=\; (\mu - \tfrac{1}{2}\sigma^2)\,t + \sigma B_t,
   \qquad
   X_t \;=\; X_0\,\exp\!\bigl[(\mu - \tfrac{1}{2}\sigma^2)\,t + \sigma B_t\bigr].
```

*Volatility drag.* Taking expectations: $\mathbb{E}[X_t] = X_0\,e^{\mu t}$, but $\mathbb{E}[\ln X_t] = \ln X_0 + (\mu - \tfrac{1}{2}\sigma^2)\,t$. The expected log return $\mu - \sigma^2/2$ is strictly less than the log of the expected return, $\mu$, by the variance correction term. This is the volatility drag. Two illustrative regimes: with zero arithmetic drift ($\mu = 0$), expected log growth is $-\sigma^2/2$, strictly negative; with drift exactly equal to the Itô correction ($\mu = \sigma^2/2$), expected log growth is zero (the drift just offsets the drag). In financial terms, volatility eats into geometric returns; this is why an asset with $20\%$ expected return and $40\%$ volatility delivers a long-run geometric mean of only $\mu - \sigma^2/2 = 12\%$.

**{prf:ref}`ex-ch8-2`: KFE for an OU process.** The OU process $dX_t = \eta(\bar X - X_t)\,dt + \sigma\,dB_t$ has drift $\mu(x) = \eta(\bar X - x)$ and diffusion coefficient $\sigma$. The KFE in conservation form is

```{math}
:enumerated: false

\partial_t g(x,t) \;=\; -\partial_x\bigl[\mu(x)\,g(x,t)\bigr] + \tfrac{\sigma^2}{2}\,\partial_{xx} g(x,t)
   \;=\; \partial_x\bigl[\eta(x - \bar X)\,g\bigr] + \tfrac{\sigma^2}{2}\,\partial_{xx} g.
```

Setting $\partial_t g = 0$ for the stationary density $g^\star(x)$:

```{math}
:enumerated: false

\eta\,\partial_x\bigl[(x - \bar X)\,g^\star\bigr] + \tfrac{\sigma^2}{2}\,g^{\star\prime\prime} \;=\; 0.
```

Integrate once in $x$, with the constant of integration set to zero by the no-flux boundary condition at $\pm\infty$:

```{math}
:enumerated: false

\eta\,(x - \bar X)\,g^\star + \tfrac{\sigma^2}{2}\,g^{\star\prime} \;=\; 0
   \quad\Longleftrightarrow\quad
   \frac{g^{\star\prime}(x)}{g^\star(x)} \;=\; -\frac{2\eta}{\sigma^2}\,(x - \bar X).
```

This is a linear ODE for $\ln g^\star$; integrating gives $\ln g^\star = -\eta(x-\bar X)^2/\sigma^2 + \mathrm{const}$, i.e.

```{math}
:enumerated: false

g^\star(x) \;\propto\; \exp\!\bigl[-\eta(x - \bar X)^2/\sigma^2\bigr].
```

This is a Gaussian density with mean $\bar X$ and variance $\sigma^2/(2\eta)$, normalized by $\int g^\star\,dx = 1$:

```{math}
:enumerated: false

g^\star(x) \;=\; \sqrt{\frac{\eta}{\pi\sigma^2}}\,\exp\!\left[-\frac{\eta\,(x - \bar X)^2}{\sigma^2}\right]
   \;=\; \mathcal{N}\!\bigl(\bar X,\, \sigma^2/(2\eta)\bigr).
```

The OU's stationary distribution is Gaussian with mean equal to the mean-reversion target, variance equal to the diffusion-to-mean-reversion ratio $\sigma^2 / (2\eta)$: faster mean reversion (larger $\eta$) shrinks the dispersion; larger diffusion grows it.

**{prf:ref}`ex-ch8-3`: Functional derivative.** For the master-equation toy specification $V(a, g) = \int u(c(a, y))\,g(y)\,dy$ where $c(a,y)$ is fixed (does not depend on $g$), the functional derivative $\delta V / \delta g$ measures the linear response of $V$ to a perturbation in $g$.

In the ambient vector space of signed measures, a point perturbation gives $\delta V/\delta g(y_0) = \lim_{\varepsilon\to 0} [V(a, g + \varepsilon\delta_{y_0}) - V(a,g)]/\varepsilon$, where $\delta_{y_0}$ is a Dirac mass at $y_0$. Substituting,

```{math}
:enumerated: false

V(a, g + \varepsilon\delta_{y_0}) \;=\; \int u(c(a,y))\,(g(y) + \varepsilon\delta_{y_0}(y))\,dy
   \;=\; V(a,g) + \varepsilon\,u(c(a, y_0)).
```

Therefore $\delta V/\delta g(y_0) = u(c(a, y_0))$. If we restrict $g$ to the probability simplex, perturbations must preserve total mass; for example, $\eta=\delta_{y_0}-\delta_{y_1}$ gives directional derivative $u(c(a,y_0))-u(c(a,y_1))$. Equivalently, on the simplex the derivative kernel is identified only up to an additive constant.

*Interpretation.* The functional derivative at a point $y_0$ is the value contribution of an infinitesimal mass placed at $y_0$. In the toy spec where $V$ is just a population average of utilities, the contribution is the per-agent utility $u(c(a, y_0))$ at that point. In the real master equation, $c(a, y)$ would itself depend on $g$ (because prices depend on $g$), and the functional derivative would pick up additional indirect terms via $\partial c/\partial g$; this is what makes the master equation a genuinely infinite-dimensional PDE rather than a parametric family of finite PDEs.

**{prf:ref}`ex-ch8-4`: HJB residual.** Coding exercise. The right answer is the out-of-sample residual table produced by your run of [`lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb`](notebooks/lecture_13_continuous_time_ha_numerics/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb). Report the training budget, collocation batch, random seed, and test grid. The residual should decrease when the collocation budget and network capacity are increased, but the scaling is empirical rather than a universal $N^{-p}$ law because it mixes approximation, optimization, and sampling error.

**{prf:ref}`ex-ch8-5`: Closed Aiyagari system.** Combining the four ingredients:

*HJB* (from {eq}`eq-hjb_full`):

```{math}
:enumerated: false

\rho V(a,n) \;=\; \max_c \bigl\{u(c) + V'(a,n)(wn + ra - c) + \lambda(n)(V(a,\hat n) - V(a,n))\bigr\}.
```

*KFE for the stationary distribution* (from {eq}`eq-kfe_econ`, with $\partial_t g = 0$):

```{math}
:enumerated: false

0 \;=\; -\partial_a[s^\star(a,n)\,g(a,n)] - \lambda(n)\,g(a,n) + \lambda(\hat n)\,g(a,\hat n),
```

where $s^\star(a,n) = wn + ra - c^\star(a,n)$ is the optimal savings function.

*Firm FOCs* (Cobb–Douglas):

```{math}
:enumerated: false

r \;=\; \alpha A K^{\alpha-1} L^{1-\alpha} - \delta,
   \qquad
   w \;=\; (1-\alpha) A K^\alpha L^{-\alpha}.
```

*Market clearing*:

```{math}
:enumerated: false

K \;=\; \sum_n \int_{\underline a}^\infty a\,g(a,n)\,da,
   \qquad
   L \;=\; \sum_n n \int_{\underline a}^{\infty} g(a,n)\,da.
```

The equilibrium objects are $(V(a,n),g(a,n),K,L,r,w)$, with $L$ often pinned down by the stationary income shares and $w$ implied by the firm FOCs once $(K,L)$ is known. In practice one fixes a candidate $r$, computes the associated firm demand and wage, solves the HJB for $V$ and hence the policy $c^\star, s^\star$, plugs into the KFE for $g$, computes implied $K = \sum_n\int a g(a,n)\,da$, and compares this capital supply with the capital demand implied by the candidate $r$. A fixed point in $r$ is the equilibrium. This is the bisection-on-$r$ algorithm of {cite:t}`achdou2022income`, and the PINN replaces the inner HJB and KFE solves with neural-network approximation while keeping the outer fixed-point loop on $r$ unless the full equilibrium system is learned jointly.

*Why both must be solved consistently at each candidate $r$.* The HJB takes $r$ as an input (price-taking agents); the KFE takes the policy from the HJB. Mis-specifying $r$ during training would feed a mis-specified policy into the KFE and yield an inconsistent $K$. In the production-scale solver, the fixed-point loop alternates HJB and KFE to convergence *at each $r$* before updating $r$; the PINN can be trained on all four equations jointly when the architecture is rich enough to learn the equilibrium price as an output.

**{prf:ref}`ex-ch8-6` and {prf:ref}`ex-ch8-7` (statements: p. , p. ).** Coding exercises. Expected diagnostics:

*{prf:ref}`ex-ch8-6`.* With a fixed policy, the finite-dimensional KFE is a linear forward equation. If the discretized generator is ergodic, the distance to the stationary distribution should decay approximately exponentially after transient modes die out. Estimate the slope from your run rather than quoting a universal number; the fitted rate is controlled by the slowest nonzero eigenvalue of the KFE generator and depends on income-switching intensities, savings drift, and grid truncation.

*{prf:ref}`ex-ch8-7`.* On the one-asset stationary benchmark, finite differences should usually win on absolute wall-clock time and give the cleanest low-dimensional benchmark residuals. A PINN may become more attractive when the same architecture is reused across many nearby parameter values, when warm starts work well, or when the state space is extended beyond what a grid handles comfortably. Report the actual cold-start and warm-start timings from your machine, and treat memory use as hardware- and backend-dependent.

(sol-ch9)=
## {ref}`ch-gp`: Gaussian Processes
**{prf:ref}`ex-ch9-1`: Posterior on three points.** The RBF kernel with length scale $\ell = 1$ and signal variance $\sigma_f^2 = 1$ is $k(x, x') = \exp(-(x-x')^2/2)$. With training points $X = (0, 1, 2)$ and targets $y = (0, 0.8, 0.3)$, the kernel matrix is

```{math}
:enumerated: false

K = \begin{pmatrix} 1 & e^{-1/2} & e^{-2} \\ e^{-1/2} & 1 & e^{-1/2} \\ e^{-2} & e^{-1/2} & 1\end{pmatrix}
   \approx \begin{pmatrix} 1 & 0.6065 & 0.1353 \\ 0.6065 & 1 & 0.6065 \\ 0.1353 & 0.6065 & 1\end{pmatrix}.
```

Adding observation noise $\sigma_y^2 I = 0.01\,I$ to the diagonal of $K$ and solving $(K + \sigma_y^2 I)\bm v = \bm y$ with $\bm y = (0, 0.8, 0.3)^\top$ by Gaussian elimination gives

```{math}
:enumerated: false

(K + \sigma_y^2 I)^{-1} \bm y \;\approx\; (-0.964,\; 1.744,\; -0.621)^\top.
```

At $x^\star = 1.5$,

```{math}
:enumerated: false

k_\star
   = \bigl(k(1.5,0),\, k(1.5,1),\, k(1.5,2)\bigr)
   = \bigl(e^{-1.125},\, e^{-0.125},\, e^{-0.125}\bigr)
   \approx (0.3247, 0.8825, 0.8825).
```

Posterior mean:

```{math}
:enumerated: false

\bar{\mu}(x^\star) = k_\star^\top (K + \sigma_y^2 I)^{-1} \bm y \approx 0.3247\cdot(-0.964) + 0.8825\cdot 1.744 + 0.8825\cdot(-0.621) \approx 0.678.
```

For the variance, solve $(K + \sigma_y^2 I)\bm w = k_\star$, giving $\bm w \approx (-0.142,\; 0.662,\; 0.495)^\top$, hence

```{math}
:enumerated: false

\bar{\sigma}^2(x^\star) = k(x^\star, x^\star) - k_\star^\top \bm w \;\approx\; 1 - 0.975 \;\approx\; 0.025.
```

The posterior is $\mathcal{N}(0.678, 0.025)$, with standard deviation $\approx 0.158$. Notice the posterior mean "smooths" the two flanking observations $y = 0.8$ and $y = 0.3$ rather than being pulled to either; the $0.158$ standard deviation reflects partial information at a halfway point between two data points.

**{prf:ref}`ex-ch9-2`: Marginal likelihood Occam.** The log marginal likelihood is

```{math}
:enumerated: false

\log p(y | X, \ell) = -\tfrac{1}{2} y^\top K_\ell^{-1} y \;-\; \tfrac{1}{2} \log|K_\ell| \;-\; \tfrac{n}{2}\log(2\pi),
```

where $K_\ell = K_\ell(\ell) + \sigma_y^2 I$ is the kernel matrix at length scale $\ell$. The first term penalizes *misfit* (data that don't lie in the GP's predicted manifold get a high $y^\top K_\ell^{-1} y$); the second term penalizes *model complexity* ($\log|K_\ell|$ is large when the kernel can fit anything, small when it is rigid). This is the Bayesian Occam's razor: small $\ell$ gives a flexible model that fits any training data perfectly (small misfit) but accepts a large complexity penalty; large $\ell$ enforces smoothness (large misfit if the data are not smooth) but enjoys a small complexity penalty. The optimum balances the two.

For three data points $y = (0, 0.8, 0.3)$ at $x = (0, 1, 2)$, plotting $\log p(y | X, \ell)$ against $\ell \in [0.1, 5]$ typically shows an interior maximum near the data's natural variation scale. At $\ell = 0.1$, the kernel is very local: each point is nearly uncorrelated with its neighbors, so the data fit is easy but the flexible prior receives a complexity penalty. At $\ell = 5$, the kernel forces all three function values to be nearly equal, which fits the data poorly. The peak is the point where these two forces balance.

**{prf:ref}`ex-ch9-3`: Active subspace by hand.** For $f(\bm x) = (x_1 + x_2 + x_3)^2 + 0.01(x_1 - x_2)^2$ on $[-1,1]^3$, the gradient is

```{math}
:enumerated: false

\nabla f(\bm x) = 2(x_1 + x_2 + x_3)\,(1, 1, 1)^\top + 0.02(x_1 - x_2)\,(1, -1, 0)^\top.
```

The dominant term is along $(1,1,1)$ (with weight roughly $2(x_1+x_2+x_3) \cdot \sqrt{3}$), and the perturbation along $(1,-1,0)$ is two orders of magnitude smaller.

Compute $\hat C = \mathbb{E}[\nabla f \nabla f^\top]$ with $x \sim \mathcal{U}[-1,1]^3$ i.i.d., so $\mathbb{E}[x_i^2] = 1/3$. Let $s = x_1 + x_2 + x_3$; then $\mathbb{E}[s^2] = 1$, $\mathbb{E}[s(x_1 - x_2)] = \mathbb{E}[x_1^2 - x_2^2] = 0$, and $\mathbb{E}[(x_1 - x_2)^2] = 2/3$. Substituting,

```{math}
:enumerated: false

\hat C \;\approx\; 4\,\mathbf{1}\mathbf{1}^\top \;+\; \mathcal{O}(10^{-4}),
```

where $\mathbf{1} = (1,1,1)^\top$ and the $\mathcal{O}(10^{-4})$ correction comes from the $0.01\,(x_1 - x_2)$ perturbation. Since $\mathbf{1}\mathbf{1}^\top$ has eigenvalues $\{3, 0, 0\}$ (eigenvector $\mathbf{1}/\sqrt 3$ for the nonzero one), the leading eigenvalue of $\hat C$ is $\lambda_1 \approx 4 \cdot 3 = 12$, with eigenvector $(1,1,1)/\sqrt{3}$, the "aggregate direction." The next eigenvalue is $\sim 10^{-4}$, four orders of magnitude smaller. The active subspace of dimension $1$ already captures essentially all of the function's variability.

**{prf:ref}`ex-ch9-4`: Deep vs. linear active subspace.** Coding exercise. The linear-AS diagnostic should show two nonzero eigenvalues for the radial-ridge target, with the remaining eigenvalues close to numerical zero. Thus a linear active subspace needs $d_{\mathrm{lin}}=2$ to represent the two linear features $w_1^\top\xi$ and $w_2^\top\xi$. Deep AS can reach its validation-MSE elbow already at $d_{\mathrm{nl}}=1$ because the encoder can learn a scalar nonlinear aggregate such as $(w_1\cdot\xi)^2 + (w_2\cdot\xi)^2$. Report the held-out MSE curve and eigenvalues from the run rather than quoting universal numbers, since optimization noise and train/validation splits move the exact diagnostics.

**{prf:ref}`ex-ch9-5`: BAL on a 2D function.** Coding exercise. Pure-variance acquisition selects points purely by where the GP is most uncertain, ignoring the predicted mean. The resulting design is usually more uniform across the input domain because the GP variance is high wherever data is sparse, regardless of function value. Maximizing pure $\log\sigma^2(\x)$ gives exactly the same design as maximizing pure $\sigma^2(\x)$, since the logarithm is monotone. Differences arise only once the acquisition is mixed with a mean term, for example $w_{\mathrm{obj}}\mu(\x)+\tfrac{w_{\mathrm{var}}}{2}\log\sigma^2(\x)$: then the design tilts toward regions that are both uncertain and high-valued under the current surrogate. For global surrogate construction, pure variance is the cleaner baseline; for optimization-like goals, the mixed score may be useful.

**{prf:ref}`ex-ch9-6`: Sobol sensitivity with a GP surrogate.** Coding exercise. Report the actual errors from the run rather than treating fixed percentages as universal reference outputs. The expected pattern is improvement as $N$ increases: first-order Sobol errors should fall from small to larger sample budgets, while total-effect indices may converge more slowly because they include interaction terms. The cost ratio is the important lesson: once the surrogate is trained, very large Sobol designs can be evaluated at negligible marginal cost relative to repeated true-model solves, which is why surrogate-based sensitivity analysis is standard for expensive simulators such as climate IAMs.

**{prf:ref}`ex-ch9-7`: Prior-driven RBF-GP extrapolation outside the training domain.** *(i)* Coding: on the training interval $[0,1]$ the GP closely tracks $f(x) = \sin(2\pi x)\,e^{-x}$ with credible band of width $\sim 0.05$.

*(ii)* Analytical claim: for an RBF kernel $k(x, x') = \sigma_f^2 \exp(-(x-x')^2 / (2\ell^2))$, when $|x - x_i| \gg \ell$ for all training points $x_i$, the kernel cross-vector $k_\star = (k(x, x_1), \ldots, k(x, x_n))^\top \to (0, \ldots, 0)^\top$. Therefore the posterior mean $\bar\mu(x) = k_\star^\top (K + \sigma_y^2 I)^{-1} y \to 0$ (the prior mean). The posterior variance is

```{math}
:enumerated: false

\bar\sigma^2(x) \;=\; k(x,x) - k_\star^\top (K+\sigma_y^2 I)^{-1} k_\star \;\to\; \sigma_f^2 - 0 \;=\; \sigma_f^2,
```

the prior variance. Hence far from data the GP literally returns the prior $\mathcal{N}(0, \sigma_f^2)$, regardless of the training data.

*(iii)* Coding verification: at $x = 3$ (with $\ell \approx 0.2$ for the training data), the cross-kernel is $\exp(-(3-1)^2/(2\cdot 0.04)) = \exp(-50) \approx 10^{-22}$, well below floating-point precision. Posterior mean $\approx 0$, posterior s.d. $\approx \sigma_f \approx 0.5$ (whatever was learned via marginal-likelihood maximization).

*(iv)* The implication is more subtle than "overconfidence." Far from the training data the posterior literally reverts to the prior, so the $\pm 2\sigma_f$ band there is exactly what the prior would have produced before any data were observed; it is overconfident only when the prior variance or the learned length scale is itself misleading, for instance when $\sigma_f$ was calibrated by marginal-likelihood maximization on a training set that does not represent the function's scale outside the training hull. In particular, the posterior does not know that $f$ continues oscillating outside $[0,1]$; it just reverts to zero with a band of width $\sigma_f$, regardless of whether the true function actually stays close to zero there. Bayesian active learning algorithms that select points by maximum posterior variance can therefore fail to acquire informative samples outside the convex hull when the prior variance is no larger than the within-hull noise scale.

Mitigations: use a Matérn kernel with $\nu = 1/2$ or $\nu = 3/2$ (heavier-tailed than RBF, posterior reverts to prior more slowly), incorporate a polynomial mean function in the GP prior (so extrapolation grows with $x$ instead of decaying to zero), or use a boundary-aware acquisition function that explicitly penalizes exploitation outside the convex hull. In economic applications (e.g., extrapolating an estimated value function to wealth levels outside the training range), the safest practice is to flag any query outside the convex hull of training data and refuse to predict, rather than to trust a prior-driven band that has no data behind it.

(sol-ch10)=
## {ref}`ch-estimation`: Deep Surrogate Models and Structural Estimation
**{prf:ref}`ex-ch10-1`: Identification.** Let $m(\varrho)=\mathbb{E}_\varrho[h(C,I,Y)]$ be the simulated moment vector implied by the persistence parameter. At the truth $\varrho^\star$, local identification is captured by the Jacobian

```{math}
:enumerated: false

M(\varrho^\star)
   =
   \frac{\partial m(\varrho)}{\partial \varrho}\bigg|_{\varrho^\star}.
```

A moment is strongly identifying if $|M_k(\varrho^\star)|$ is large; a weakly identifying moment has derivative near zero, in which case small changes in $\varrho$ produce nearly invisible changes in the moment and the SMM objective becomes flat.

*Brock–Mirman example.* The output autocorrelation $\mathrm{Corr}(\log Y_t,\log Y_{t-1})$ is typically the strongest local moment for $\varrho$, because it is a direct echo of the productivity AR(1). The autocorrelation of consumption growth is also informative, but more filtered through equilibrium dynamics. The mean savings rate is nearly flat in the scalar $\varrho$ exercise and is therefore weakly identifying for persistence. The exact ranking should be reported from the finite-difference Jacobian in the notebook, not from a hard-coded number.

**{prf:ref}`ex-ch10-2`: Optimal weighting.** The SMM objective is $Q_T(\theta) = \hat g(\theta)^\top W \hat g(\theta)$, where $\hat g(\theta) = m(\theta) - \hat m^\mathrm{data}$ is the moment-error vector. Under standard regularity (Hansen 1982), the asymptotic variance of $\hat\theta_\mathrm{SMM}$ is

```{math}
:enumerated: false

\mathrm{Avar}(\hat\theta)
   =
   (M^\top W M)^{-1}M^\top W\Omega W M(M^\top W M)^{-1},
```

where $M=\partial m/\partial\theta'$ at $\theta^\star$ and $\Omega=\mathrm{Var}(\sqrt{T}\hat g(\theta^\star))$ is the covariance of the data-simulation moment discrepancy.

To minimize $\mathrm{Avar}(\hat\theta)$ over choices of $W$, set up the Lagrangian or use the matrix calculus result that the variance is minimized when $W \propto \Omega^{-1}$. Substituting $W=\Omega^{-1}$:

```{math}
:enumerated: false

\mathrm{Avar}(\hat\theta)\big|_{W^\star}
   =
   (M^\top \Omega^{-1}M)^{-1}.
```

For independent simulated panels of the same length as the data, $\Omega=(1+1/S)\Sigma_m$; if simulation noise is negligible, $\Omega\approx\Sigma_m$. Compare to a different weighting $W=\tilde W$: $\mathrm{Avar}|_{\tilde W}-\mathrm{Avar}|_{W^\star}$ is positive semi-definite by the Gauss–Markov theorem applied to the moment system.

*Why identity weighting in the first stage?* The optimal weight depends on the unknown covariance at the truth. The standard two-stage strategy is: (1) estimate with $W=I$ to obtain a consistent but inefficient $\hat\theta^{(1)}$; (2) estimate $\hat\Omega$ at or near $\hat\theta^{(1)}$; (3) re-estimate with $W=\hat\Omega^{-1}$.

**{prf:ref}`ex-ch10-3`: Common random numbers.** Coding exercise. Without CRN, the objective $Q_T(\varrho)$ changes both because $\varrho$ changes and because each candidate uses a new shock path. This contaminates the profile with Monte Carlo noise. With CRN, every candidate is evaluated on the same shock path, so differences in $Q_T(\varrho)$ isolate the structural effect of persistence. Quantify the gain by repeating the entire candidate grid across Monte Carlo panels and comparing the cross-panel variance of $Q_T(\varrho)$ at each grid point.

**{prf:ref}`ex-ch10-4`: SMM vs. SBI.** SMM solves an outer optimization at deployment: given new data, run a $K$-step optimization over $\theta$, where each step requires (a) a model simulation at the candidate $\theta$ and (b) moment computation. Total cost per inference: $K$ model evaluations.

SBI (e.g., neural posterior estimation) does the work upfront at training time: simulate the model at $N$ values of $\theta$, train a neural density estimator $q(\theta | y)$ on the $(\theta, y)$ pairs, and then at deployment evaluate $q$ on the new data with one forward pass. Total cost per inference (after training): one forward pass.

SBI dominates when (i) the model is expensive but a single training run is amortized over many datasets to be analyzed (e.g., a central bank running daily updates); (ii) the model is non-likelihood (no closed-form likelihood, only simulator); (iii) the parameter dimension is moderate ($p \le 20$) so training data covers parameter space. SMM dominates when (i) the model is cheap (each evaluation a few ms), (ii) only one or a handful of inferences are needed, (iii) classical confidence intervals are required (SBI gives a Bayesian posterior, which differs).

**{prf:ref}`ex-ch10-5`: $J$-statistic and overidentification.** Coding+analytical. In the joint notebook's over-identified specification, $q=4$ and $p=2$, so the degrees of freedom are $q-p=2$. Under correct specification and with $W=\Omega^{-1}$, the $J$-statistic at the optimum follows

```{math}
:enumerated: false

J
   =
   T\,\hat g(\hat\theta)^\top \Omega^{-1}\hat g(\hat\theta)
   \xrightarrow{d}
   \chi^2_{2}.
```

If the exercise uses $W=\Sigma_m^{-1}$ while simulation noise is non-negligible, adjust the scale under the equal-length independent-simulation approximation or use the Monte Carlo/bootstrap distribution directly.

Empirical distribution: under correct specification, the Monte Carlo distribution of $J(\hat\theta)$ should be centered near the corresponding $\chi^2_2$ benchmark once the weighting and simulation-noise treatment are consistent. Under a structural break, the model cannot match all four moments simultaneously, so the distribution should shift to the right and the rejection rate should rise above the nominal size. The size of that shift is an output of the experiment and should be reported from the run.

**{prf:ref}`ex-ch10-6`: Bootstrap confidence intervals.** Coding exercise. The nonparametric bootstrap should respect time-series dependence; use a moving-block or stationary bootstrap rather than iid resampling of individual dates. The parametric bootstrap draws new shock paths under the estimated parameter vector and re-runs the estimator. Report the actual CI endpoints and coverage from the run. In small samples, bootstrap intervals may differ materially from sandwich intervals because the SMM criterion is nonlinear and the finite-sample distribution of $\hat\theta$ need not be close to Gaussian.

**{prf:ref}`ex-ch10-7`: ML vs. SMM efficiency.** Coding exercise. If $\log z_t$ is observed and follows

```{math}
:enumerated: false

\log z_{t+1}=\varrho\log z_t+\sigma_z\varepsilon_{t+1},
```

then the Gaussian AR(1) likelihood gives a direct MLE for $\varrho$ (OLS of $\log z_{t+1}$ on $\log z_t$ when $\sigma_z$ is unrestricted). This MLE is efficient for the observed-shock likelihood. The SMM estimator based on a finite moment vector reaches the GMM efficiency bound for those moments, but it equals MLE efficiency only if the chosen moments span the score, which the three pedagogical moments generally do not.

*Why SMM despite the efficiency loss?* In production-scale models (heterogeneous-agent macro, dynamic IO), the likelihood is often unavailable in closed form, and computing it would require integration over high-dimensional latent states. Moments are cheaper, interpretable, and robust to parts of the model that are not central to the research question. The cost is that SMM efficiency depends on the information content of the chosen moments.

(sol-ch11)=
## {ref}`ch-climate`: Climate Economics and Deep Uncertainty Quantification
**{prf:ref}`ex-ch11-1`: ECS sensitivity.** Coding exercise. Report the SCC at the central calibration and at each ECS value in the likely and very-likely ranges. The expected qualitative pattern is monotone and often convex in ECS: higher equilibrium climate sensitivity raises temperature damages and therefore the emissions shadow price. The quantitative range is calibration-dependent and should be reported from the notebook rather than fixed in the solution text. The interpretation should connect the dispersion to the climate-science finding of {cite:t}`sherwood2020assessment` that ECS uncertainty is a major driver of SCC uncertainty.

**{prf:ref}`ex-ch11-2`: Sobol decomposition.** For $q(\theta_1, \theta_2, \theta_3) = \theta_1\theta_2 + \theta_3^2$ with $\theta_i \sim \mathcal{U}[0,1]$ i.i.d.:

```{math}
:enumerated: false

\begin{aligned}
\mathbb{E}[q] &= \mathbb{E}[\theta_1]\mathbb{E}[\theta_2] + \mathbb{E}[\theta_3^2] = \tfrac{1}{4} + \tfrac{1}{3} = \tfrac{7}{12}, \\
   \mathbb{E}[q^2] &= \mathbb{E}[\theta_1^2]\mathbb{E}[\theta_2^2] + 2\mathbb{E}[\theta_1\theta_2]\mathbb{E}[\theta_3^2] + \mathbb{E}[\theta_3^4] = \tfrac{1}{9} + \tfrac{1}{6} + \tfrac{1}{5} = \tfrac{43}{90}, \\
   \mathrm{Var}(q) &= \tfrac{43}{90} - \bigl(\tfrac{7}{12}\bigr)^2 = \tfrac{11}{80}.
\end{aligned}
```

Conditional means:

```{math}
:enumerated: false

\begin{aligned}
\mathbb{E}[q \mid \theta_1] &= \tfrac{\theta_1}{2} + \tfrac{1}{3}, &
   \mathrm{Var}_{\theta_1}\bigl(\mathbb{E}[q\mid\theta_1]\bigr) &= \tfrac{1}{48},\\
   \mathbb{E}[q \mid \theta_3] &= \tfrac{1}{4} + \theta_3^2, &
   \mathrm{Var}_{\theta_3}\bigl(\mathbb{E}[q\mid\theta_3]\bigr) &= \mathrm{Var}(\theta_3^2) = \tfrac{4}{45}.
\end{aligned}
```

First-order Sobol indices:

```{math}
:enumerated: false

S_1 = S_2 = \frac{1/48}{11/80} = \frac{5}{33} \approx 0.152,
   \qquad
   S_3 = \frac{4/45}{11/80} = \frac{64}{99} \approx 0.646.
```

Sum: $S_1 + S_2 + S_3 = 94/99 \approx 0.949$; the residual $5/99 \approx 0.051$ is the $\theta_1\theta_2$ interaction term.

Total-effect indices (one minus the share explained by all other variables): $S_1^T = S_2^T = 20/99 \approx 0.202$; $S_3^T = 64/99 \approx 0.646$ (no interactions involving $\theta_3$ since it enters only quadratically alone).

A SALib estimate with $10^4$ samples typically matches these analytical values to two decimal places.

**{prf:ref}`ex-ch11-3`: ACE benchmark.** Coding exercise. Use ACE's closed-form optimal carbon tax as the analytic benchmark, then compute the DEQN-trained DICE SCC at the same calibration and report the percentage discrepancy. A small gap is expected only when units, discounting, damage curvature, and time discretization are aligned; any residual difference should be attributed explicitly to the DICE discretization, smoothing choices, and calibration differences rather than to a fixed percentage target.

**{prf:ref}`ex-ch11-4`: Pareto-improving tax design.** Project-level coding exercise. The constrained search is over a low-dimensional parameter vector $\vartheta = (\vartheta_{\mathrm{tax}}, \omega)$, with the cohort welfare constraints $\tilde U_t(\vartheta) \ge U_t$ enforced via a penalty or barrier term in the outer optimizer. The Pareto frontier is typically tight: cohorts whose BAU welfare is already close to the unconstrained Ramsey-optimal welfare are easily satisfied, while cohorts that bear the bulk of the transition cost (often the youngest at the time of the reform) bind first; the welfare gap to the unconstrained problem grows with the share of constrained cohorts. Holding $\omega$ fixed at the BAU or declining benchmark $\bar\omega$ leaves measurable welfare on the table because endogenizing the transfer schedule provides an extra free instrument with which to relax the binding constraints. Numerical answers are calibration-dependent and should be reported from the notebook rather than benchmarked against fixed values; the qualitative interpretation is what matters for the policy discussion.

**{prf:ref}`ex-ch11-5`: Carbon-cycle warm-up.** Coding exercise driven by notebook [`lecture_16_01_Climate_Exercise.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_01_Climate_Exercise.ipynb). Part (a) avoided-warming and avoided-damages numbers at 2100 under the 50% mitigation rule are notebook outputs and depend on calibration, so they should be reported from the run rather than fixed in the solution text. Part (b) the longest-timescale reservoir is identified by the smallest non-zero eigenvalue of the carbon-cycle transition matrix; for the CDICE three-box calibration this is the lower-ocean compartment, with a multi-century characteristic timescale. Part (c) a quadratic damage function $D(T) = \pi_2 T^2$ has bounded curvature in $T$ and therefore underweights tail risk: marginal damage grows only linearly, so very high realizations of $T$ are penalized far less than under super-quadratic damages or an explicit tipping-hazard term, and tail-risk assessment requires one of those richer specifications (compare {prf:ref}`ex-ch11-8`).

**{prf:ref}`ex-ch11-6`: Deterministic CDICE-DEQN reproduction.** Coding exercise driven by notebook [`lecture_16_02_DICE_DEQN_Library_Port.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_02_DICE_DEQN_Library_Port.ipynb). The verification target is the set $\{T^{\mathrm{AT}}(2100),\ M^{\mathrm{AT}}(2100),\ \mu(2100),\ \mathrm{SCC}(2015),\ \mathrm{SCC}(2100),\ \mathrm{SCC}(2300)\}$, each compared against the reference solution at the tolerances stated in the notebook's verification gate. Reference numbers depend on seed, hardware, optimizer schedule, and trajectory sampling, so the solution should anchor on the verification gate rather than on fixed numbers. Typical convergence problems trace back to scaling differences across the eight residuals or to insufficient trajectory coverage near the carbon-stock saturation regime; the residual-balancing methods discussed in {ref}`ch-nas` are the first line of defense.

**{prf:ref}`ex-ch11-7`: Stochastic SCC fan chart.** Coding exercise driven by notebook [`lecture_16_03_Stochastic_DICE_DEQN.ipynb`](notebooks/lecture_16_climate_economics_iams/lecture_16_03_Stochastic_DICE_DEQN.ipynb). As the AR(1) productivity volatility $\sigma_z$ rises, the right tail of the SCC distribution at 2100 widens disproportionately, because higher productivity raises emissions which feed convexly into damages. Report the quantiles $q_{10}, q_{50}, q_{90}$ of the SCC fan chart at each $\sigma_z$ rather than the mean alone, since the mean obscures the asymmetry. The qualitative finding aligns with {cite:t}`caiSocialCostCarbon2019`, that productivity and consumption-growth shocks shift the SCC distribution materially and not just its location, supporting the use of distribution-aware reporting in policy work.

**{prf:ref}`ex-ch11-8`: Tipping-point regime-switching damages.** Coding exercise. Report the unconditional tipping probability by 2100, the SCC at $t=0$, and the optimal abatement path under the regime-switching specification and under the smooth-damage baseline. The expected qualitative pattern is that adding an irreversible tipping hazard raises the SCC and brings abatement forward, especially when $T_\mathrm{thresh}$ is low. The factor by which the SCC rises is an output of the hazard calibration and retrained policy. The policy implication should be framed as a precautionary motive under threshold uncertainty, not as a universal numerical multiplier.

**{prf:ref}`ex-ch11-9`: Real options value of waiting.** *(i) Closed forms.* Under the act-now strategy, the planner chooses $\mu_0$ to minimize $\theta\mu_0^2 + \alpha\,\mathbb{E}[\mathrm{ECS}]\cdot(1 - \mu_0)$. With $\mathbb{E}[\mathrm{ECS}] = (\mathrm{ECS}_L + \mathrm{ECS}_H)/2 \equiv \overline{\mathrm{ECS}}$, the FOC gives $\mu_0^\star = \alpha \overline{\mathrm{ECS}}/(2\theta)$. Plugging back, expected cost is

```{math}
:enumerated: false

C_\mathrm{now} \;=\; \alpha \overline{\mathrm{ECS}} - \frac{(\alpha \overline{\mathrm{ECS}})^2}{4\theta}.
```

Under the wait strategy, after observing $\widehat{\mathrm{ECS}}$, the posterior mean $\mathbb{E}[\mathrm{ECS}\mid\widehat{\mathrm{ECS}}]$ has variance shrunk relative to the prior. The planner chooses $\mu_1^\star(\widehat{\mathrm{ECS}}) = \alpha\,\mathbb{E}[\mathrm{ECS}\mid\widehat{\mathrm{ECS}}]/(2\theta)$. Expected cost (using the law of total variance):

```{math}
:enumerated: false

C_\mathrm{wait} \;=\; \alpha \overline{\mathrm{ECS}} - \frac{\alpha^2}{4\theta}\,\mathbb{E}\bigl[\mathbb{E}[\mathrm{ECS}\mid\widehat{\mathrm{ECS}}]^2\bigr]
   \;=\; \alpha \overline{\mathrm{ECS}} - \frac{\alpha^2}{4\theta}\bigl(\overline{\mathrm{ECS}}^2 + \mathrm{Var}_\mathrm{prior} - \mathrm{Var}_\mathrm{post}\bigr).
```

*(ii) Value of waiting.* Difference:

```{math}
:enumerated: false

\mathrm{VoW} \;=\; C_\mathrm{wait} - C_\mathrm{now}
   \;=\; -\frac{\alpha^2}{4\theta}\bigl(\mathrm{Var}_\mathrm{prior} - \mathrm{Var}_\mathrm{post}\bigr).
```

As the signal becomes informative ($\sigma_\varepsilon \to 0$), $\mathrm{Var}_\mathrm{post} \to 0$, so $\mathrm{Var}_\mathrm{prior} - \mathrm{Var}_\mathrm{post} \to \mathrm{Var}_\mathrm{prior} > 0$, and $\mathrm{VoW} < 0$: waiting is preferred. The reduction in posterior variance is the value of information.

*(iii) With irreversibility.* Add a wedge $\eta\mu_1^2$ to the wait-branch objective, so the planner who waits solves a different second-period problem:

```{math}
:enumerated: false

\min_{\mu_1}\;(\theta+\eta)\mu_1^2
   +\alpha\,\mathbb{E}[\mathrm{ECS}\mid\widehat{\mathrm{ECS}}]\,(1-\mu_1).
```

Assuming the interior solution remains in $[0,1]$,

```{math}
:enumerated: false

\mu_1^{\eta\star}(\widehat{\mathrm{ECS}})
   =
   \frac{\alpha\,\mathbb{E}[\mathrm{ECS}\mid\widehat{\mathrm{ECS}}]}{2(\theta+\eta)}
```

and expected wait cost becomes

```{math}
:enumerated: false

C_\mathrm{wait}^\eta
   =
   \alpha \overline{\mathrm{ECS}}
   - \frac{\alpha^2}{4(\theta+\eta)}
   \bigl(\overline{\mathrm{ECS}}^2+\mathrm{Var}_\mathrm{prior}-\mathrm{Var}_\mathrm{post}\bigr).
```

The irreversibility wedge reduces the benefit of conditioning abatement on the signal. The value of waiting is now

```{math}
:enumerated: false

\mathrm{VoW}^\eta
   =
   C_\mathrm{wait}^\eta - C_\mathrm{now}
   =
   \frac{\alpha^2}{4}
   \left[
      \frac{\overline{\mathrm{ECS}}^2}{\theta}
      -
      \frac{\overline{\mathrm{ECS}}^2+\mathrm{Var}_\mathrm{prior}-\mathrm{Var}_\mathrm{post}}{\theta+\eta}
   \right].
```

For sufficiently large $\eta$, $\mathrm{VoW}^\eta>0$: the option value of waiting is dominated by the irreversibility penalty, and act-now is preferred.

The threshold occurs at

```{math}
:enumerated: false

\eta^\star
   =
   \frac{\theta(\mathrm{Var}_\mathrm{prior}-\mathrm{Var}_\mathrm{post})}
        {\overline{\mathrm{ECS}}^2},
```

where the value of information just balances the irreversibility cost.

*Connection to climate policy.* This stylized model captures the central tension in climate policy: information arrives over time about ECS, damage functions, and tipping thresholds, but emissions are largely irreversible (atmospheric CO$_2$ persists for centuries). The chapter's Bayesian-learning treatment makes the trade-off quantitative: under fast learning and slow climate dynamics, waiting can be optimal; under slow learning and fast tipping risks, an early-action premium emerges. The empirical literature {cite:p}`pindyck2007uncertainty,caiSocialCostCarbon2019` finds that for realistic climate calibrations, the irreversibility channel typically dominates, supporting near-term carbon tax implementation rather than "wait and see" policies.

**{prf:ref}`ex-ch11-10`: Non-stationary DEQN on a 1D toy.** *(i) LQ-Riccati baseline.* This is a finite-horizon linear-quadratic tracking problem, so the value function is quadratic in the state with time-varying coefficients. Conjecture $V_t(x) = P_t x^2 + 2 q_t x + c_t$ with terminal condition $V_{T_{\max}}(x) = \lambda_T (x - x^\ast_{T_{\max}})^2$, i.e. $P_{T_{\max}} = \lambda_T$ and $q_{T_{\max}} = -\lambda_T\,x^\ast_{T_{\max}}$. Writing $D_t = r + P_{t+1}$, the Bellman first-order condition in $u$ gives the affine policy

```{math}
:enumerated: false

u^\star_t(x) \;=\; -\,\frac{P_{t+1}\bigl(\alpha x + g_t\bigr) + q_{t+1}}{D_t}
   \;=\; -K_t\,x - k_t,
   \qquad
   K_t = \frac{\alpha P_{t+1}}{D_t},
   \quad
   k_t = \frac{P_{t+1}\,g_t + q_{t+1}}{D_t},
```

and substituting back yields the backward Riccati recursions

```{math}
:enumerated: false

P_t \;=\; 1 + \frac{\alpha^2 r\,P_{t+1}}{D_t},
   \qquad
   q_t \;=\; -x^\ast_t + \frac{\alpha r\,\bigl(P_{t+1}\,g_t + q_{t+1}\bigr)}{D_t}.
```

The additive Gaussian shock exhibits certainty equivalence: $\sigma$ enters only the constant $c_t$ (through a $P_{t+1}\sigma^2$ term), so the optimal policy is identical whether the rollouts are noisy or deterministic, and the trained DEQN must recover the same feedback rule in both cases. At the stated calibration, iterating the recursions shows that $P_t$ sits at its stationary value $\approx 1.083$ and the gain at $K_t \approx 0.870$ over almost the entire horizon, with a short boundary layer near $T_{\max}$ (the terminal weight $P_{T_{\max}} = \lambda_T = 5$ is far above the stationary $P$, pulling $K_{T_{\max}-1}$ up to $\approx 0.931$). The genuine calendar-time dependence of the policy therefore lives almost entirely in the intercept $k_t$, through the drift $g_t$, the target path $x^\ast_t = a + bt$, and $q_{t+1}$, plus the end-of-horizon Riccati transient.

*(ii)–(iii) Training and verification.* Coding parts; the three ingredients map one-to-one onto the modifications of {ref}`sec-nsdeqn_algo`: the bounded time input $\tau_t$ is "time enters as a state," the calendar-time stratification mirrors the forward-simulated training pool, and the terminal penalty is the short-horizon replacement for the transversality-by-long-horizon argument. Anchor the verification in part (iii) on the closed form above: evaluate $u^\star_t(x)$ from the recursions along a test set of simulated paths and compare with the network's actions in the mean-squared-action metric; the $1\%$ tolerance is achievable with a small MLP, and the realized gap is a run output.

*(iv) Ablations.* The analysis in (i) predicts the ranking. Dropping $\tau_t$ hurts most, and increasingly so as $T_{\max}$ grows: without a time input the network can only represent a time-averaged policy, while the true intercept $k_t$ drifts linearly through $g_t$ and $x^\ast_t$, so the representation error grows with the length of the target path. Removing the terminal penalty degrades tracking only in the boundary layer near $T_{\max}$ where the $K_t$ transient lives, leaving the interior fit largely intact. Removing stratification does not change what the network *can* represent; it degrades the optimization by under-weighting sparsely sampled calendar-time bins, an effect that is real but typically the mildest of the three. Exact magnitudes are outputs of the ablation runs and should be reported from them.

(sol-ch12)=
## {ref}`ch-outlook`: Synthesis and Outlook
**{prf:ref}`ex-ch12-1`: Method-choice scenario.** Sketch:

*(a) 4-state monetary-policy DSGE with smooth shocks.* Use **classical perturbation or projection as the baseline**. The state space is small and the shocks are smooth, so a classical method is transparent, fast, and easy to audit. A DEQN becomes attractive only if the model is extended with genuinely global nonlinearities (for example a binding zero lower bound, occasionally binding collateral constraints, or a highly non-quadratic loss). Hybrid: use a GP surrogate over policy-rule parameters after the classical or DEQN solution step if many counterfactuals are needed.

*(b) 200-agent OLG with progressive taxation.* Use **DEQN with Young's-method aggregation** ({ref}`ch-olg` and {ref}`ch-young`). $200$ cohorts is at the edge where explicit-panel methods (all-in-one DL) and histogram methods compete; histograms are easier when constraints bind frequently across cohorts. Hybrid: post-train a GP surrogate over the Pareto-weight calibration to evaluate optimal-tax-rule sensitivity.

*(c) Exotic option pricing on an irregularly shaped payoff.* Use **PINN or Deep Galerkin methods with careful treatment of the payoff kink** ({ref}`ch-pinn`) for a single-contract instance, or **DeepONet** if prices are needed for many strikes or contract parameters. The non-smooth object is the terminal payoff, not a generic spatial boundary; for a strong-form Black–Scholes residual this usually calls for smoothing the payoff, using smooth activations away from the kink, or switching to a weak/viscosity-aware formulation. GP surrogates can help as an outer pricing surface over a few parameters, but they are not the primitive PDE solver.

*(d) Climate-IAM where SCC uncertainty is the deliverable.* Use the **full pipeline**: DEQN to solve the deterministic IAM, GP surrogate over deep-uncertain parameters (ECS, damage convexity, tipping thresholds), Sobol/Shapley sensitivity decomposition for attribution, and BAL for sample-efficient uncertainty quantification ({ref}`ch-climate`, {ref}`ch-gp`). This combines the IAM uncertainty-quantification workflow of {cite:t}`friedlDeep2023` with the surrogate-based policy-search logic in {cite:t}`kubler2025using`.

**{prf:ref}`ex-ch12-2`: When NOT to use deep learning.** Open-ended. Sketch of one regime: *bit-exact reproducibility for regulatory audit*. GPU non-determinism in atomic accumulators (described in Appendix {ref}`app-reproducibility`) means that a deep-learning solver typically cannot reproduce the same numbers across hardware platforms; for regulatory work where auditors must replay every step bitwise, a deterministic finite-difference solver on a fixed grid is preferable. Even when deterministic flags are set, BLAS implementations differ across CUDA versions. Classical fixed-grid methods are easier to make bit-reproducible because the operation order can be pinned and the solver path is usually far less sensitive to random initialization and stochastic mini-batches.

**{prf:ref}`ex-ch12-3`: Reproducibility audit.** Coding exercise. Expected behavior: re-running notebook [`lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb`](notebooks/lecture_03_deep_equilibrium_nets/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb) on the same machine with the same seeds should reproduce the reported diagnostics within the stated tolerance. Bitwise equality of trained network parameters should be expected only when deterministic framework settings, hardware, BLAS/CUDA versions, and floating-point order of operations are all pinned. Re-running on a different GPU, or with deterministic flags off, can produce small deviations in the last few printed digits of the savings-rate diagnostics while remaining well within the residual tolerance of the training.

**{prf:ref}`ex-ch12-4`: Open-ended.** Open-ended; expected output is a 1–2 page research sketch. A representative answer combines DEQN (for solving the model) with GP surrogates (for parameter estimation): e.g., a 6-month project that estimates a heterogeneous-agent NK model with deep-learning policy functions, then uses a deep-kernel GP surrogate to do Bayesian posterior inference on the structural parameters. This integrates {ref}`ch-young`, {ref}`ch-gp`, and {ref}`ch-estimation`.

**{prf:ref}`ex-ch12-5`: Hybrid pipeline DEQN $+$ GP $+$ SMM.** Coding exercise. Illustrative outputs to benchmark against, not fixed targets:

- *Step 1 (DEQN training).* Report the actual training time for the parameterized network. Adding $(\beta,\varrho)$ as inputs increases the network's effective input dimension from $2$ to $4$, so convergence can be slower than in the single-calibration case.

- *Step 2 (moment grid).* Report the time for the $20\times 20=400$ simulations, each with $T=200$ periods, under the trained policy.

- *Step 3 (GP fitting).* Report the fitting time for a four-output GP on $400$ training points in $\mathbb{R}^2$.

- *Step 4 (SMM optimization with GP).* Report the number of objective evaluations, optimizer iterations, and total GP-evaluated SMM time.

- *Naive SMM benchmark.* Compare with a version that re-solves or re-trains the DEQN at each candidate only if that benchmark is computationally feasible; otherwise estimate its cost from one representative re-solve.

The expected qualitative result is that the surrogate-based optimization is much cheaper per objective evaluation after the one-time DEQN, simulation-grid, and GP-fitting costs have been paid. The actual speedup and estimation errors are outputs of the run and should be reported rather than assumed.

**{prf:ref}`ex-ch12-6`: DeepONet for a parameterized HJB.** *(i) Architecture.* Branch net $\bm b(\gamma): \mathbb{R} \to \mathbb{R}^p$ encodes the parameter $\gamma$ into a $p$-dimensional latent representation; trunk net $\bm t(a): \mathbb{R} \to \mathbb{R}^p$ encodes the query point $a$ into the same latent dimension. Predicted value:

```{math}
:enumerated: false

\widehat{V}(a; \gamma) \;=\; \langle \bm b(\gamma), \bm t(a)\rangle \;+\; b_0,
```

where $b_0$ is a learnable scalar bias. Both nets are typically MLPs with $2$–$4$ layers and width $50$–$200$. The architecture is non-trivial because it imposes the bilinear separable structure $V(a; \gamma) = \sum_{k=1}^p b_k(\gamma) t_k(a)$, which is the universal approximation form for nonlinear operators.

*(ii) UAT for operator learning.* {cite:t}`lu2021learning` show that a continuous nonlinear operator can be approximated uniformly on a compact set of input functions and a compact output domain by a DeepONet with finitely many sensors, a branch network, and a trunk network. In the present notation, for every $\varepsilon>0$ there are sensor locations, finite-width branch/trunk nets, and latent dimension $p$ such that

```{math}
:enumerated: false

\sup_{\gamma\in\Gamma}\sup_{a\in\mathcal A}
   \bigl|V(a;\gamma)-\widehat V(a;\gamma)\bigr|
   < \varepsilon
```

on the compact training domain $\Gamma\times\mathcal A$, provided the solution operator is continuous there. This generalizes ordinary universal approximation from finite-dimensional functions to operators, but it does not provide extrapolation guarantees outside the compact training domain.

*(iii) Cost comparison.* Independent PINN runs cost $N \cdot C_\mathrm{PINN}$. One DeepONet run costs $C_\mathrm{DON}$, where $C_\mathrm{DON}/C_\mathrm{PINN}$ captures the larger network, the parameter-sampling overhead, and the longer training needed to span the parameter range. Operator learning wins exactly when

```{math}
:enumerated: false

C_\mathrm{DON} < N\,C_\mathrm{PINN},
   \qquad\text{or equivalently}\qquad
   C_\mathrm{DON}/C_\mathrm{PINN} < N .
```

At $N=50$, the break-even ratio is therefore $50$: a DeepONet can cost up to fifty single-parameter PINN solves and still be cheaper in total. The realized speedup is an empirical output of the implementation, not a universal constant.

*(iv) Limitations.* *Extrapolation:* DeepONet trained on $\gamma \in [1.5, 5]$ has no guarantees outside this range; in practice the predicted operator $V(a; \gamma)$ for $\gamma > 5$ degrades smoothly but may violate the structural property that $V$ should remain concave. Mitigation: use polynomial-augmented branch networks that extrapolate via known tails (e.g., Bernoulli polynomials), or refuse to predict outside the convex hull of training $\gamma$ values.

*Concavity preservation:* the bilinear DeepONet architecture does not automatically preserve concavity in $a$. A trained network may produce non-concave $\widehat{V}(\cdot; \gamma)$ for some $\gamma$, which is economically wrong under risk-averse preferences. Mitigation: use an input-concave architecture for the trunk representation, for example the negative of an input-convex neural network where appropriate, or add a concavity penalty $\max(0,\widehat{V}''(a;\gamma))$ to the loss. Both increase training cost but restore pressure toward the structural property.
