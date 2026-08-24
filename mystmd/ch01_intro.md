---
title: "Introduction to Machine Learning and Deep Learning"
label: ch-intro
---

The Preface argued that quantitative economics needs new tools because the models of interest, heterogeneous-agent economies, OLG models with aggregate risk, integrated assessment models, all share state spaces too large for traditional grid-based methods. This chapter takes that argument as given and supplies the technical machinery the rest of the manuscript will build on. Readers who want a fuller pitch for *why* deep learning is the right response should re-read the Preface; readers who already accept the premise can dive directly into the machinery below.

We begin with a brief refresher on the motivation, mostly to fix vocabulary and citations, then survey the three broad paradigms of machine learning (supervised, unsupervised, and reinforcement learning), and then develop the core technical machinery: neural network architectures, loss functions, gradient-based optimization, backpropagation, weight initialization, activation functions, and the modern theory of generalization including the double descent phenomenon. Readers already comfortable with these topics may skim this chapter and proceed to {ref}`ch-deqn`, where the economic applications begin.

The foundational references for the material in this chapter include {cite:t}`mcculloch1943logical`, {cite:t}`hebb1949organization` and {cite:t}`rosenblatt1958perceptron` for the historical origins of artificial neurons and the first learning rules, {cite:t}`cybenko1989approximation` and {cite:t}`hornik1989multilayer` for universal approximation, {cite:t}`rumelhart1986learning` for backpropagation, {cite:t}`robbins1951stochastic` for the stochastic-approximation roots of SGD, {cite:t}`geman1992biasvariance` for the bias/variance dilemma that underpins modern generalization theory, {cite:t}`kingma2015adam` for the Adam optimizer, {cite:t}`ioffe2015batch` and {cite:t}`srivastava2014dropout` for batch normalization and dropout, and {cite:t}`goodfellow2016deep` for a comprehensive textbook treatment, complemented by the historical survey of {cite:t}`schmidhuber2015deep`.

## Motivation and Applications

The past decade has witnessed a remarkable convergence of three developments that have transformed machine learning from a niche academic pursuit into a practical tool of extraordinary power: the availability of large-scale datasets, the advent of massively parallel hardware (GPUs and TPUs), and algorithmic innovations in training deep neural networks ({numref}`fig-dl_enablers`). While much of the public attention has focused on applications such as image recognition, natural language processing, and game playing, the implications for economics and finance are equally profound.

```{figure} figures/fig-dl_enablers.svg
:name: fig-dl_enablers

The three enablers of the deep-learning revolution: large-scale data, massively parallel compute, and algorithmic improvements. None of the three alone is sufficient; their co-availability since the early 2010s is what has turned neural networks from a niche academic curiosity into a workhorse scientific and industrial tool.
```

Deep learning has already demonstrated its potential across a broad range of economic applications. In macroeconomics, neural networks serve as global approximators of policy and value functions in high-dimensional equilibria that classical grid-based methods cannot reach {cite:p}`maliar2021deep,azinovicDEEPEQUILIBRIUMNETS2022`; heterogeneous-agent extensions encode cross-sectional distributions via histograms or permutation-invariant moment networks {cite:p}`young2010,han2023deepham,yang2025structural`. Search-and-matching models with aggregate shocks {cite:p}`payne2025deepsam` and continuous-time macro-finance settings requiring HJB approximation or deep-BSDE solvers {cite:p}`gopalakrishna2024aliens,duarte2024ml,han2018solving` add further coverage, as do optimal monetary policy rules under persistent supply shocks {cite:p}`nuno2024monetary`. In climate economics, surrogate-based workflows solve integrated assessment models and derive Pareto-improving carbon-tax rules in OLG–IAMs with deep uncertainty {cite:p}`kubler2025using,Folini_2021,fernandezvillaverde2025climate`. In finance, surrogate models accelerate option pricing, sovereign-default computation, and portfolio optimization {cite:p}`hutchinson1994nonparametric,scheideggertreccani_2018,arellano2008default,gaegauf2023portfolio,chen2025private`. For structural estimation, neural-network surrogates make inference tractable and enable global uncertainty quantification for IAMs {cite:p}`kase2022estimating,friedlDeep2023,chen2026Deep`. Finally, these methods connect to an earlier generation of neural-network function approximation in economic computation: adaptive learning {cite:p}`chenwhite1998adaptive`, derivative pricing {cite:p}`hutchinson1994nonparametric`, and parameterized expectations {cite:p}`duffy2001approximating`, with {cite:t}`valaitisvilla2024` showing how the parameterized-expectations approach extends to contemporary deep architectures; structural discrete-choice estimation rounds out the historical picture {cite:p}`norets2012structural`.

Two themes cut across these application areas and motivate the rest of the script. First, every application area listed there involves a state space whose dimension grows with the number of agents, assets, shocks, or climate states; tensor-product grids become infeasible long before the modeling questions become uninteresting. Second, neural networks are universal approximators with cost that scales with parameter count rather than with $N^d$, so they are the natural replacement function class once the problem becomes high-dimensional. Subsequent chapters take this through-line and develop it: {ref}`ch-deqn` introduces the DEQN methodology that all macro / heterogeneous-agent / search applications above share; {ref}`ch-irbc` scales it to a 100+-country benchmark; {ref}`ch-pinn`–{ref}`ch-ct_theory` develop the continuous-time analogue; {ref}`ch-gp` develops the Gaussian-process methodology, and {ref}`ch-estimation`–{ref}`ch-climate` put the deep-surrogate and GP machinery to work on structural estimation and integrated assessment models.

In this course, we focus on the recent advances enabled by *deep* neural networks, modern hardware, and algorithmic innovations in training.

For central banks, in particular, these tools address pressing practical needs. Many of the models listed above, such as heterogeneous-agent New Keynesian (HANK) models, search and matching models with aggregate shocks, overlapping-generations (OLG) economies with aggregate risk, and integrated assessment models coupling climate and economic dynamics, involve state spaces of dimension $d \gg 5$, where traditional grid-based numerical methods are computationally infeasible: a tensor-product grid with $N$ points per dimension requires $N^{d}$ nodes, the canonical *curse of dimensionality* of {cite:t}`bellman1961adaptive`. Deep learning provides a mesh-free function approximator. For Barron-class functions, two-layer networks achieve dimension-independent rates in the number of hidden units (often stated as squared $L^2$ error of order $\mathcal{O}(1/n_1)$ under a bounded Barron norm), whereas tensor-product grids for generic smooth functions suffer rates such as $\mathcal{O}(M^{-2/d})$ in the total number $M=N^d$ of grid nodes. The point is not that every economic object satisfies a Barron bound, but that neural approximators avoid the mechanical tensor-product explosion. {ref}`ch-deqn` returns to this comparison with concrete numbers for a six-shock DSGE.

Specifically, this course focuses on using deep neural networks as computational tools for: (i) solving high-dimensional dynamic stochastic general equilibrium (DSGE) models, (ii) approximating value and policy functions in continuous-time settings via physics-informed neural networks, (iii) constructing fast and accurate surrogate models for parameter estimation and uncertainty quantification, and (iv) leveraging Gaussian processes for sample-efficient Bayesian active learning {cite:p}`azinovicDEEPEQUILIBRIUMNETS2022,friedlDeep2023,chen2026Deep`. Supporting techniques include neural architecture search and adaptive multi-objective loss balancing (one option, ReLoBRaLo {cite:p}`bischof2025relobralo`, is developed in the notebooks; alternatives include SoftAdapt and GradNorm). For a comprehensive textbook treatment of the foundations, we refer the reader to {cite:t}`goodfellow2016deep` and {cite:t}`chollet2017deeplearning`; for concise surveys we recommend {cite:t}`lecun2015deep`.

## Types of Machine Learning

Before diving into the technical details, it is useful to recall the three broad paradigms of machine learning, each defined by the nature of the available data and the feedback signal provided to the algorithm.

### Supervised Learning

Given a set of *labeled* input–output pairs $\{(\x^{(i)}, y^{(i)})\}_{i=1}^{m}$, the goal is to learn a mapping $h:\mathcal{X}\to\mathcal{Y}$ that generalizes to unseen inputs. The two main tasks are *regression* and *classification*.

**Regression** ($y \in \R$): predict a continuous target from input features. A simple linear model takes the form

```{math}
:enumerated: false

h_{\bm{\theta}}(x) = \theta_0 + \theta_1 x,
```

where the parameters $\bm{\theta} = (\theta_0, \theta_1)$ are learned from data. {numref}`fig-regression` illustrates regression on a house-price dataset: each dot is a training observation, and the line is the fitted model.

```{figure} figures/fig-regression.svg
:name: fig-regression

Supervised learning: regression. The model $h_{\bm{\theta}}(x) = \theta_0 + \theta_1 x$ (red line) is fitted to observed house prices (blue dots).
```

**Classification.** ($y \in \{0,1,\dots,K\}$): assign an input $\x$ to one of $K$ discrete categories. A linear classifier predicts class 1 whenever $\w^\top \x + b > 0$ and class 0 otherwise. {numref}`fig-classification` shows a credit-scoring example: applicants are classified as low-risk or high-risk based on income and savings, and the dashed line is the learned decision boundary.

```{figure} figures/fig-classification.svg
:name: fig-classification

Supervised learning: classification. A linear decision boundary separates low-risk (blue circles) from high-risk (red crosses) applicants in the income–savings feature space.
```

### Unsupervised Learning

Given only *unlabeled* data $\{\x^{(i)}\}_{i=1}^{m}$, the goal is to discover hidden structure without any target signal. Two common tasks are:

- **Clustering:** partitioning data into groups of similar observations. *Example:* segmenting firms into peer groups based on financial characteristics such as size, leverage, and profitability.

- **Dimensionality reduction:** compressing features into fewer dimensions while preserving important variation. *Example:* principal component analysis of yield curves, where three factors (level, slope, curvature) capture most of the cross-sectional variation.

{numref}`fig-clustering` illustrates a clustering task: unlabeled data points in two dimensions are partitioned into three clusters, each indicated by a different color and centroid marker.

```{figure} figures/fig-clustering.svg
:name: fig-clustering

Unsupervised learning: clustering. Unlabeled data points are grouped into three clusters; the $+$ markers indicate cluster centroids. No target labels are used; the algorithm discovers the grouping from the data alone.
```

### Reinforcement Learning

In reinforcement learning, an *agent* interacts with an *environment* over a sequence of time steps. At each step $t$, the agent observes a state $s_t$, selects an action $a_t = \pi(s_t)$ according to its policy $\pi$, and receives a reward $r_t$ from the environment. The goal is to learn a policy that maximizes the expected cumulative discounted return:

```{math}
:enumerated: false

\max_{\pi}\; \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t \, r_t\right], \qquad \gamma \in [0,1),
```

where $\mathbb{E}_{\pi}[\,\cdot\,]$ is taken over the trajectory distribution induced jointly by the policy $\pi$ and the (possibly stochastic) environment dynamics, starting from a given initial-state distribution. {numref}`fig-rl-loop` illustrates this agent–environment interaction loop.

```{figure} figures/fig-rl-loop.svg
:name: fig-rl-loop

Reinforcement learning: the agent–environment loop. The agent observes a state, takes an action, and receives a reward signal. Over time, it learns a policy $\pi$ that maximizes cumulative discounted reward.
```

*Example:* an algorithmic trader learning an execution strategy by optimizing realized profit over sequences of order placements; or a central bank learning an interest-rate rule by maximizing a welfare criterion over simulated macroeconomic trajectories.

(sec-supervised_vs_unsupervised)=
## Course Focus: Supervised vs. Unsupervised Learning
This course begins with the *supervised learning* paradigm, which provides the essential building blocks: choosing a parameterized model, defining a loss function, and minimizing it via gradient descent.

The core methods of this course, DEQNs and PINNs, are not supervised in the classical labeled-data sense. More precisely, they are *self-supervised residual methods*: the economic equilibrium conditions or governing equations generate the training signal. To see why, recall the key distinction: in supervised learning, the loss function measures the discrepancy between the network's prediction $\hat{y}^{(i)}$ and a known target label $y^{(i)}$, for example, a mean squared error $\frac{1}{m}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$. This requires a dataset of correct input–output pairs.

In DEQNs and PINNs, *no such labels exist*. Consider the key differences:

- **DEQNs:** A neural network approximates the unknown policy function of a dynamic economic model. The loss function is the *Euler equation residual*, which measures how much the network's output violates the model's optimality conditions at sampled state points. The "correct" policy values are never provided; instead, the network discovers the equilibrium by driving these residuals to zero.

- **PINNs:** A neural network approximates the unknown solution to a partial differential equation (PDE). The loss function is the *PDE residual*, which evaluates how well the network satisfies the differential equation at randomly sampled collocation points. Again, the true solution is not available as training data; the network learns it by enforcing the PDE constraint.

In both cases, the training data consists only of *input locations* (sampled states or collocation points) with no associated output labels. The loss is defined entirely by the structure of the economic model or the governing equation, not by example solutions. They are therefore unsupervised in the narrow sense of using no labels, but the more informative term is equation-based self-supervision.

Despite this fundamental difference, the optimization machinery is shared: these approaches define a loss $J(\bm{\theta})$ over trainable parameters and minimize it via (stochastic) gradient descent. This is why we introduce the supervised learning pipeline first in the next section: it establishes the model–loss–optimizer framework that DEQNs and PINNs then adapt by replacing the data-driven loss with a physics-based one.

## The Supervised Learning Pipeline

Every supervised learning algorithm follows the same three-step recipe, regardless of whether the model is a linear regression, a random forest, or a deep neural network ({numref}`fig-ml_recipe`). Understanding this pipeline is essential because the DEQN and PINN methods in later chapters modify step 2 (replacing data-driven losses with physics-based residuals) while keeping steps 1 and 3 intact.

```{figure} figures/fig-ml_recipe.svg
:name: fig-ml_recipe

The three-step supervised-learning recipe that underpins every model in this course. Choose a parametric hypothesis $h(\x;\bm{\theta})$, measure its misfit on a labeled dataset via a loss $J(\bm{\theta})$, and minimize $J$ over the parameter vector $\bm{\theta}$. DEQNs and PINNs modify step 2 (replacing the data-driven loss with an equilibrium or PDE residual) while keeping steps 1 and 3 identical.
```

Given a training set $\{(\x^{(i)}, y^{(i)})\}_{i=1}^{m}$ of input–output pairs, we seek a hypothesis $h:\mathcal{X}\to\mathcal{Y}$ parameterized by $\bm{\theta}$ that minimizes the empirical risk. For regression problems, the default choice is the mean squared error (MSE):

$$
J(\bm{\theta}) = \frac{1}{m}\sum_{i=1}^{m}\big(h_{\bm{\theta}}(\x^{(i)}) - y^{(i)}\big)^{2}.
$$ (eq-mse)

This loss is not chosen arbitrarily. If the data are generated by

```{math}
:enumerated: false

y^{(i)} = h_{\bm{\theta}}(\x^{(i)}) + \varepsilon^{(i)}, \qquad \varepsilon^{(i)} \sim \mathcal{N}(0,\sigma^2),
```

then the log-likelihood of the sample is, up to constants, proportional to $-\sum_i (h_{\bm{\theta}}(\x^{(i)}) - y^{(i)})^2$. Minimizing MSE is therefore equivalent to maximum likelihood under homoskedastic Gaussian observation noise. This is one reason why squared error is the natural benchmark loss for regression; see {cite:t}`bishop2006` {cite}`goodfellow2016deep,deisenroth2020mathematics`.

For classification tasks, the model must output class probabilities. In the binary case ($K=2$), the simplest approach passes a single scalar score $z$ through the *sigmoid* function,

```{math}
:enumerated: false

p \;=\; \sigma(z) \;=\; \frac{1}{1+e^{-z}} \;\in\; (0,1),
```

and assigns class 1 whenever $p > 0.5$, equivalently whenever $z > 0$ ({numref}`fig-sigmoid_decision`). For $K>2$ classes the natural generalization maps a raw score vector $\bm{z}\in\mathbb{R}^K$ onto the probability simplex via *softmax*: $\hat{y}_k = e^{z_k}/\sum_j e^{z_j}$. In both cases the misfit between predicted probabilities and true labels is measured by the cross-entropy loss:

$$
J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_k^{(i)}\log \hat{y}_k^{(i)},
$$ (eq-cross_entropy)

where $\hat{y}_k^{(i)}$ is the predicted probability that observation $i$ belongs to class $k$. If the target is encoded as a one-hot vector, exactly one component of $(y_1^{(i)},\dots,y_K^{(i)})$ equals one and all others are zero, so for an observation whose true class is $c$ the loss contribution reduces to $-\log \hat{y}_c^{(i)}$. The model is rewarded for assigning high probability to the correct class and penalized heavily when that probability is near zero.

The origin of cross-entropy is again likelihood theory. In binary classification, with label $y \in \{0,1\}$ and predicted success probability $p$, the Bernoulli log-likelihood is

```{math}
:enumerated: false

\log L = y \log p + (1-y)\log(1-p).
```

Negating and averaging this expression gives the binary cross-entropy. The $K$-class formula above is the corresponding negative log-likelihood for a categorical distribution with probabilities generated by sigmoid ($K=2$) or softmax ($K>2$). Cross-entropy is therefore the statistically natural loss whenever the model output is meant to represent class probabilities; see again {cite:t}`bishop2006` {cite}`deisenroth2020mathematics`.

```{figure} figures/fig-sigmoid_decision.svg
:name: fig-sigmoid_decision

Binary classification with a sigmoid output. A scalar score $z$ (the model's raw output) is mapped to a probability $p = \sigma(z) \in (0,1)$. The prediction rule assigns class 1 whenever $p > 0.5$, equivalently whenever $z > 0$. The dashed lines mark the decision threshold; no neural-network architecture is assumed—any model that produces a real-valued score can be combined with this mapping and the binary cross-entropy loss.
```

```{figure} figures/fig-classification_losses.svg
:name: fig-classification_losses

Binary cross-entropy and mean squared error as functions of the predicted class probability $p$. Cross-entropy rises much more sharply near confident mistakes, which is why it is usually better aligned with probabilistic classification.
```

{numref}`fig-classification_losses` makes the practical difference visible. If the true label is $y=1$ but the model predicts a very small $p$, then the cross-entropy loss explodes because $-\log p \to \infty$ as $p \downarrow 0$. The same holds symmetrically when $y=0$ and $p \uparrow 1$. Mean squared error does penalize mistakes, but it does so much more mildly near the boundaries. For probabilistic classification, that weaker penalty is usually undesirable because it does not strongly discourage overconfident wrong predictions.

The optimization is performed via gradient descent or one of its stochastic variants, which we discuss in {ref}`sec-training`.

(sec-robust_losses)=
### Beyond MSE: Robust and Asymmetric Losses
MSE is optimal under Gaussian noise, but real-world economic and financial data often contain outliers and heavy tails that inflate the squared penalty disproportionately. Two classical alternatives are useful in this course and beyond.

**Huber loss.** Introduced by {cite:t}`huber1964robust` in the context of robust location estimation, the Huber loss behaves like MSE near the origin and like $L_1$ in the tails, capping the influence of any single observation:

$$
\ell_\delta(r) = \begin{cases}
\tfrac{1}{2} r^2 & \text{if } |r| \leq \delta, \\
\delta\,(|r| - \tfrac{1}{2}\delta) & \text{otherwise,}
\end{cases}
\qquad r \equiv y - \hat{y}.
$$ (eq-huber)

The threshold $\delta$ controls the transition and is typically chosen to be a few times the noise scale. Huber loss retains the smoothness needed for gradient-based optimization while reducing the weight of extreme residuals, which makes it the default choice for regression problems with suspected outliers.

**Quantile (pinball) loss.** {cite:t}`koenker1978regression` proposed the *check function*

$$
\ell_\tau(r) = \max\!\bigl(\tau\, r,\; (\tau-1)\, r\bigr), \qquad \tau \in (0,1),
$$

whose minimizer is the conditional $\tau$-quantile of $y$ given $\x$ rather than the conditional mean. Setting $\tau = 0.5$ recovers the median (absolute-error) regression; setting $\tau = 0.05$ or $\tau = 0.95$ targets the lower or upper tail. In financial risk management this is precisely the statistic of interest: $\tau = 0.05$ yields a neural-network estimator of the lower-tail $5\%$-quantile of returns, which corresponds to Value-at-Risk (VaR) at the conventional $5\%$ level, and averaging the pinball loss over many quantiles traces out the full conditional distribution of returns, an approach known as quantile regression or distributional regression.

```{prf:remark} Why this matters in economics and finance

 Tail risk is often more important than average performance. The Huber and quantile losses let the network focus explicitly on robustness to outliers and on worst-case outcomes, respectively. A single quantile loss gives a Value-at-Risk estimator at the chosen probability level; Expected Shortfall requires additional structure, such as averaging lower-tail quantiles, fitting several quantiles jointly, or using a dedicated joint VaR–ES scoring rule. Notebook `07_Genz_Approximation_and_Loss_Functions` compares MSE, Huber, and quantile losses on a common regression task.
```


## From Perceptrons to Deep Networks

The building block of every neural network is the artificial neuron, first proposed by {cite:t}`mcculloch1943logical`. A companion question, how the synaptic weights $w_i$ themselves should adapt with experience, was first addressed by {cite:t}`hebb1949organization`, whose rule "neurons that fire together, wire together" ($\Delta w_{ij} = \eta\, x_i y_j$) is the conceptual ancestor of all gradient-based learning rules discussed below. {cite:t}`rosenblatt1958perceptron` then introduced the Perceptron, the first trainable binary classifier built on these ideas. A single neuron computes a weighted linear combination of its inputs, adds a bias term, and passes the result through a nonlinear activation function:

$$
\hat{y} = g\!\big(w_0 + \x^\top \w\big),
$$

where $\w = (w_1, \dots, w_d)^\top$ are the synaptic weights, $w_0$ is the bias, and $g(\cdot)$ is the activation function ({numref}`fig-artificial_neuron`).

```{figure} figures/fig-artificial_neuron.svg
:name: fig-artificial_neuron

An artificial neuron in the McCulloch–Pitts lineage. Inputs $x_i$ are multiplied by synaptic weights $w_i$, summed into a pre-activation $z$, and passed through a nonlinear activation $g(\cdot)$ to yield the output $\hat{y}$. The original {cite:t}`mcculloch1943logical` unit used a binary threshold for $g$; the modern artificial neuron generalizes this to arbitrary smooth activations, and all deep networks are compositions of neurons of this form.
```

Common choices for $g$ include the sigmoid $\sigma(z) = (1+e^{-z})^{-1}$, the hyperbolic tangent $\tanh(z)$, and the rectified linear unit $\mathrm{ReLU}(z) = \max(0,z)$ {cite:p}`nair2010rectified,glorot2011deep`. Without a nonlinear activation, any composition of linear layers collapses to a single affine map, a mathematical fact of fundamental importance: $\W_2(\W_1\x + \bb_1) + \bb_2 = \W_2\W_1\x + (\W_2\bb_1 + \bb_2)$.

**From a single neuron to a layer.** The single-neuron equation $\hat y = g(w_0 + \x^\top \w)$ produces a scalar output. In practice we want vector-valued outputs (and, more importantly, vector-valued *intermediate features*). A *layer* of $n$ parallel neurons, each with its own weights $\w^{(j)}$ and bias $w_0^{(j)}$, is a vector-valued generalization

```{math}
:enumerated: false

\bm{a} \;=\; g\!\big(\W \x + \bb\big), \qquad \W \in \R^{n \times d},\ \bb \in \R^{n},\ \bm{a} \in \R^{n},
```

where the nonlinearity $g$ is applied componentwise. Each row of $\W$ is the weight vector of one neuron; stacking $n$ of them gives the matrix $\W$ at once, so the layer evaluates $n$ neurons in a single matrix–vector product.

**From one layer to a deep composition.** A single hidden layer is already a universal approximator ({cite:t}`cybenko1989approximation,hornik1989multilayer`; the universal-approximation theorem stated in the next subsection), but its hidden-layer width can grow exponentially in the input dimension to attain a target accuracy. Stacking layers on top of one another reuses earlier features as inputs to later neurons; the resulting compositional representation is dramatically more efficient for many functions of interest {cite:p}`telgarsky2016benefits,barron1993universal`. A *deep feedforward network* with $L$ layers is therefore a nested composition of layer maps:

$$
f(\x;\bm{\theta}) = g_L\!\Big(\W^{(L)} g_{L-1}\!\big(\cdots g_1\!\big(\W^{(1)}\x + \bb^{(1)}\big)\cdots + \bb^{(L-1)}\big) + \bb^{(L)}\Big).
$$ (eq-dnn)

The architecture that implements {eq}`eq-dnn` is sketched in {numref}`fig-deep_ff_net`.

```{figure} figures/fig-deep_ff_net.svg
:name: fig-deep_ff_net

An $L$-layer deep feedforward network. Each layer applies an affine map followed by a pointwise nonlinearity; the composition realizes Eq. {eq}`eq-dnn`. Depth (rather than width) is what gives neural networks their efficient representational power for compositionally structured functions.
```

A useful geometric intuition, popularized by {cite:t}`chollet2017deeplearning`, is that each layer of the network performs a nonlinear coordinate transformation, successively "untangling" the manifold on which the data lies. In the input space, the data may be entangled in complex ways (e.g., two classes forming concentric spirals); each hidden layer warps the space so that the data become progressively more linearly separable. By the final hidden layer, a simple linear readout suffices. This perspective, formalized also by {cite:t}`goodfellow2016deep`, explains why depth is so powerful: each layer adds an additional coordinate transformation, and the composition of many simple transformations can represent very complex mappings with far fewer parameters than a single, wide layer would require.

**Other architectures.** This course focuses almost entirely on feedforward networks of the form {eq}`eq-dnn`, because DEQNs and PINNs operate on unstructured state vectors $\x \in \R^d$ for which feedforward maps are the natural choice. For structured inputs other architecture families exist and are used widely elsewhere: convolutional networks {cite:p}`lecun2015deep` for image data, graph neural networks for relational data, and Transformers ({ref}`sec-transformers`) for sequences. We mention them so that readers who encounter these models in the empirical-finance or applied-ML literatures know where they fit; none are required for the methods developed in later chapters.

The *universal approximation theorem* {cite:p}`cybenko1989approximation,hornik1989multilayer` guarantees that even a single hidden layer with sufficiently many neurons can approximate any continuous function on a compact set to arbitrary precision. {cite:t}`leshno1993multilayer` sharpen the hypothesis on the activation function to non-polynomiality, which covers every activation used in this script. However, in practice, deep (multi-layer) networks achieve the same accuracy with exponentially fewer parameters than wide (single-layer) ones, which motivates the use of depth; {cite:t}`telgarsky2016benefits` makes this precise by exhibiting compositional functions that a depth-$L$ network can represent in $\mathrm{poly}(d,L)$ parameters but for which any depth-$(L-1)$ network requires width exponential in $L$. {cite:t}`barron1993universal` provides classical dimension-independent approximation rates for Barron-class targets, often stated as squared $L^2$ error of order $\mathcal{O}(1/n_1)$ in the hidden width, whereas tensor-product methods for generic smooth functions scale poorly in the total number of grid nodes. This qualified comparison is the formal version of the "deep learning can beat grids" argument that motivates DEQNs in {ref}`ch-deqn`.

```{prf:definition} Universal Approximation Theorem

 Let $g:\R\to\R$ be a bounded, non-constant, continuous activation function. For any continuous function $f:\mathcal{K}\to\R$ on a compact set $\mathcal{K}\subset\R^d$ and any $\varepsilon>0$, there exists a single-hidden-layer network $F(\x) = \sum_{j=1}^{n_1} v_j\, g(\w_j^\top\x + b_j)$ such that $\sup_{\x\in\mathcal{K}} |F(\x) - f(\x)| < \varepsilon$.
```


(sec-training)=
## Training: Loss Functions, Gradient Descent, and Backpropagation
Given a loss function $J(\bm{\theta})$, training proceeds by iteratively updating the parameters in the direction of steepest descent:

$$
\bm{\theta} \leftarrow \bm{\theta} - \eta\,\nabla_{\!\bm{\theta}} J(\bm{\theta}),
$$ (eq-gd)

where $\eta > 0$ is the learning rate. In this introductory chapter $\bm{\theta}$ denotes generic trainable model parameters; in later chapters, when structural parameters and neural-network weights appear together, the script uses $\bm{\theta}$ for the structural parameters and $\rho$ for the network weights. Computing the gradient $\nabla_{\!\bm{\theta}} J$ for a deep network is achieved through *backpropagation* {cite:p}`rumelhart1986learning`, an efficient application of the chain rule that propagates error signals from the output layer back to the input layer. Appendix {ref}`app-ad` collects the matrix-calculus identities and the one-paragraph reverse-mode AD summary used throughout the script.

To see why the chain rule is central, consider a network with a single hidden layer. Let $\z = \W^{(1)}\x + \bb^{(1)}$, $\a = g(\z)$, and $\hat{y} = \w^{(2)\top}\a + b^{(2)}$ with loss $J = \tfrac{1}{2}(\hat{y}-y)^2$. Define the "delta" at the hidden layer:

$$
\bm{\delta}^{(1)} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \a} \cdot \frac{\partial \a}{\partial \z} = (\hat{y}-y)\,\w^{(2)} \odot g'(\z).
$$

Then the weight gradient follows immediately:

$$
\frac{\partial J}{\partial \W^{(1)}} = \bm{\delta}^{(1)}\,\x^\top.
$$

The key insight of {cite:t}`rumelhart1986learning` is that this chain rule application can be organized as a single backward pass through the network, reusing intermediate quantities (the $\bm{\delta}$ vectors) from the forward pass. The resulting algorithm has computational cost proportional to the forward pass; there is no need for finite differences or other expensive gradient approximations.

In practice, evaluating the full gradient over all $m$ training examples is expensive. *Stochastic gradient descent* (SGD), whose roots go back to the stochastic-approximation scheme of {cite:t}`robbins1951stochastic`, replaces the full sum with a random mini-batch $\mathcal{B} \subset \{1, \dots, m\}$, yielding an unbiased estimate of the empirical-risk gradient at much lower cost per iteration:

$$
\widehat{\nabla_{\!\bm{\theta}} J}_{\mathcal{B}}
= \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \nabla_{\!\bm{\theta}}\,\ell\bigl(h_{\bm{\theta}}(\x^{(i)}),\, y^{(i)}\bigr).
$$

With mini-batch sizes of 32–256, SGD achieves both computational efficiency and an implicit regularization effect from gradient noise. Two strands of theoretical work explain why this matters in deep nets specifically: the loss landscape of a deep network is dominated by saddle points rather than isolated bad local minima {cite:p}`dauphin2014identifying`, and SGD's gradient noise tends to bias training toward *flat* rather than *sharp* regions of the loss surface, which often generalize better {cite:p}`keskar2017large`. Even on linearly separable data, gradient descent on the logistic loss converges to the maximum-margin solution, an instance of the broader principle that the optimizer itself imposes an *implicit bias* that contributes to generalization {cite:p}`soudry2018implicit`. For a comprehensive modern review of stochastic optimization for large-scale learning (including convergence rates, adaptive methods, and variance reduction), see {cite:t}`bottou2018optimization`.

### The Adam Optimizer

Modern optimizers such as Adam {cite:p}`kingma2015adam` adapt the learning rate for each parameter based on running averages of the first and second moments of the gradient:

$$
\begin{align}
\bm{m}_t &= \beta_1\,\bm{m}_{t-1} + (1-\beta_1)\,\nabla J_t, \\
\bm{v}_t &= \beta_2\,\bm{v}_{t-1} + (1-\beta_2)\,(\nabla J_t)^2, \\
\hat{\bm{m}}_t &= \bm{m}_t/(1-\beta_1^t), \qquad \hat{\bm{v}}_t = \bm{v}_t/(1-\beta_2^t), \\
\bm{\theta}_{t+1} &= \bm{\theta}_t - \eta \cdot \hat{\bm{m}}_t / (\sqrt{\hat{\bm{v}}_t} + \varepsilon).
\end{align}
$$

The bias-corrected first moment $\hat{\bm{m}}_t$ provides momentum (smoothing out gradient noise), while the second moment $\hat{\bm{v}}_t$ provides per-parameter adaptive learning rates (parameters with large gradients receive smaller effective steps). The default hyperparameters $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$ work well across a wide range of problems, including all the economic applications in this course.

(sec-optimizer_zoo)=
### The Optimizer Family Tree: Momentum, RMSprop, Adam, AdamW
Adam did not appear out of thin air; it inherits from a family of refinements to plain SGD whose interactions are worth being explicit about for readers who will tune optimizers in practice. {numref}`tab-optimizer_family` traces the lineage; each row is a one-line modification of the row above it.

````{table}
:name: tab-optimizer_family

Lineage from plain SGD to AdamW. Each row introduces exactly one new ingredient: momentum buffers gradient noise; RMSprop adds a per-parameter learning-rate scaling by the running second moment; Adam combines the two with bias correction; AdamW separates weight decay from the gradient step so that the implicit $L_2$ regularizer does not interact with the adaptive denominator. PINN training in continuous-time chapters ({ref}`ch-pinn`–{ref}`ch-ct_theory`) often uses Adam or AdamW; DEQNs in {ref}`ch-deqn`–{ref}`ch-young` use plain Adam with the default $(\beta_1,\beta_2)=(0.9, 0.999)$ as in {cite:t}`azinovicDEEPEQUILIBRIUMNETS2022`.

| **Optimizer** | **Update rule (one parameter)** | **Reference** |
|---|---|---|
| SGD | $\theta \leftarrow \theta - \eta\,g_t$ | {cite:t}`robbins1951stochastic` |
| SGD + momentum | $v_t = \mu v_{t-1} + g_t$;  $\theta \leftarrow \theta - \eta\,v_t$ | {cite:t}`sutskever2013importance` |
| RMSprop | $s_t = \rho s_{t-1} + (1-\rho)g_t^2$;  $\theta \leftarrow \theta - \eta\,g_t/\sqrt{s_t+\varepsilon}$ | {cite:t}`tieleman2012rmsprop` |
| Adam | momentum on $g_t$ and on $g_t^2$ with bias correction (Eqs. above) | {cite:t}`kingma2015adam` |
| AdamW | Adam plus *decoupled* weight decay $\theta \leftarrow (1-\eta\lambda)\theta - \eta\,\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$ | {cite:t}`loshchilov2019decoupled` |
````

The Adam-vs-AdamW distinction is sharper than the one-line table entry suggests, so it is worth writing out both rules side by side. With $\hat m_t$, $\hat v_t$ the bias-corrected first and second moment of the gradient and $\lambda$ the weight-decay rate, Adam-with-$L_2$ (i.e. Adam applied to the loss $J + \tfrac{\lambda}{2}\|\bm\theta\|^2$) updates

```{math}
:enumerated: false

\bm\theta_{t+1} \;=\; \bm\theta_t \;-\; \eta\,\frac{\hat m_t + \lambda\,\bm\theta_t}{\sqrt{\hat v_t}+\varepsilon},
```

so the implicit regularizer is itself rescaled by the adaptive denominator $\sqrt{\hat v_t}+\varepsilon$. AdamW separates the two:

```{math}
:enumerated: false

\bm\theta_{t+1} \;=\; (1-\eta\lambda)\,\bm\theta_t \;-\; \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon},
```

so the weight-decay term shrinks every parameter by the same proportional factor regardless of gradient magnitude. This is why AdamW recovers the textbook intuition "weight decay shrinks weights uniformly" that Adam-with-$L_2$ loses.

{numref}`fig-optimizer_trajectories` gives a schematic comparison of the qualitative convergence patterns behind this optimizer family tree.

```{figure} figures/fig-optimizer_trajectories.svg
:name: fig-optimizer_trajectories

Schematic loss-trajectory comparison on a moderately ill-conditioned objective. Momentum and adaptive rescaling often improve early convergence relative to plain SGD, and Adam/AdamW are therefore useful defaults in the notebooks. The ranking is illustrative rather than universal: on some objectives, carefully tuned SGD or RMSprop can match or beat Adam-family methods.
```

### Learning Rate Schedules

The choice of learning rate $\eta$ is arguably the single most important hyperparameter in deep learning. Too large, and the optimizer diverges; too small, and convergence is impractically slow. A popular schedule is *cosine annealing* {cite:p}`loshchilov2017sgdr`, which smoothly decays the learning rate according to:

$$
\eta(t) = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\bigl(1 + \cos(\pi\, t / T)\bigr),
$$

where $T$ is the total number of training iterations. {numref}`fig-lr_schedules` compares the three learning-rate strategies used most often in practice.

```{figure} figures/fig-lr_schedules.svg
:name: fig-lr_schedules

Three common learning-rate schedules. A constant rate is simple but often converges slowly in the fine-tuning phase; exponential decay shrinks monotonically; cosine annealing {cite:p}`loshchilov2017sgdr` provides a smooth warm-to-cold transition that empirically performs well across a wide range of problems. DEQNs and PINNs typically use exponential decay or cosine annealing to polish the solution after the initial coarse-grained phase.
```

In practice, decaying schedules such as exponential decay or cosine annealing tend to refine solutions in the later stages of training, once the optimizer has found a good basin of attraction.

```{prf:remark} Loss function landscape

 The loss surface of a deep network is high-dimensional and non-convex, containing saddle points, plateaus, and sharp minima. Stochastic optimization methods navigate this landscape effectively because the noise from mini-batch sampling helps escape shallow local minima and saddle points. In economic applications, the loss function has direct economic interpretation, whether an Euler equation residual, a PDE residual, or a moment matching criterion, which provides a natural metric for assessing convergence quality.
```


### Backpropagation: The Chain Rule at Scale

For a network with $L$ layers, denote the pre-activation at layer $l$ as $\z^{(l)} = \W^{(l)}\a^{(l-1)} + \bb^{(l)}$ and the activation as $\a^{(l)} = g(\z^{(l)})$. If the final layer is linear, set $g'(z)=1$ at $l=L$ and interpret $\a^{(L)}$ as the prediction $\hat y$. The backpropagation algorithm computes $\partial J / \partial \W^{(l)}$ for all layers simultaneously by propagating a "delta" vector backward:

$$
\begin{align}
\bm{\delta}^{(L)} &= \nabla_{\a^{(L)}} J \odot g'(\z^{(L)}), \\
\bm{\delta}^{(l)} &= \bigl((\W^{(l+1)})^\top \bm{\delta}^{(l+1)}\bigr) \odot g'(\z^{(l)}), \qquad l = L-1, \ldots, 1,
\end{align}
$$

where $\odot$ denotes element-wise multiplication. The parameter gradients are then $\partial J / \partial \W^{(l)} = \bm{\delta}^{(l)} (\a^{(l-1)})^\top$ and $\partial J / \partial \bb^{(l)} = \bm{\delta}^{(l)}$. The computational cost is linear in the number of layers and the total number of parameters, a remarkable efficiency that enables training networks with millions of parameters. {numref}`fig-backprop_passes` shows the forward and backward passes side by side.

```{figure} figures/fig-backprop_passes.svg
:name: fig-backprop_passes

Backpropagation as forward and backward passes through the network. The forward pass (blue) produces and caches every layer's pre-activations $\z^{(l)}$ and activations $\a^{(l)}$; the backward pass (red, dashed) propagates the "delta" vectors $\bm{\delta}^{(l)}$ from the loss back to the inputs, reusing the cached quantities to compute all parameter gradients in a single sweep.
```

In modern deep learning frameworks such as TensorFlow and PyTorch, backpropagation is implemented automatically through computational graph tracing ("autograd"). The user only needs to define the forward computation; the framework handles the differentiation. This same automatic differentiation capability is what makes PINNs ({ref}`ch-pinn`) possible: the PDE residual requires derivatives of the network output with respect to its inputs, which autograd provides exactly and efficiently.

(sec-weight_init)=
## Weight Initialization
The initialization of network weights has a profound effect on training dynamics. If weights are initialized too large, activations explode through the layers; if too small, they vanish to zero. In both cases, the gradient signal degrades and training stalls. The key principle is to choose initial weights so that the variance of activations remains approximately constant across layers.

**Xavier/Glorot initialization.** {cite:t}`glorot2010understanding` derived the following rule for networks with symmetric activations (such as $\tanh$). For a layer with $n_\mathrm{in}$ input neurons and $n_\mathrm{out}$ output neurons, initialize weights as:

$$
W_{ij} \sim \mathcal{U}\!\left[-\sqrt{\frac{6}{n_\mathrm{in}+n_\mathrm{out}}},\; \sqrt{\frac{6}{n_\mathrm{in}+n_\mathrm{out}}}\right]
\qquad\text{or equivalently}\qquad
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_\mathrm{in}+n_\mathrm{out}}\right).
$$ (eq-xavier)

This ensures $\mathrm{Var}[a^{(l)}] \approx \mathrm{Var}[a^{(l-1)}]$ under the assumption that activations are in the linear regime of $\tanh$.

**He initialization.** For ReLU activations, {cite:t}`he2015delving` showed that the weight variance should be doubled relative to the forward fan-in rule:

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_\mathrm{in}}\right).
$$ (eq-he_init)

The justification is a *second-moment-preserving* calculation, not a variance one. For a centered symmetric pre-activation $z$,

```{math}
:enumerated: false

\E{\mathrm{ReLU}(z)^2}
    \;=\; \int_0^\infty z^2 \, p(z) \, dz
    \;=\; \tfrac{1}{2}\,\E{z^2},
```

because $p(z)$ is symmetric about zero, so the negative half of the integrand is killed and the positive half is preserved. Doubling the input weight variance therefore preserves the second moment $\E{(z^{(l)})^2}$ across layers under ReLU. Strictly speaking $\E{\mathrm{ReLU}(z)} > 0$, so the *variance* (centered second moment) is slightly smaller than the second moment, and the factor of 2 is an approximation rather than an identity; in practice the approximation is excellent and He initialization is the default for ReLU-family networks throughout this course.

```{prf:remark} Practical guidance

 The applications in this course use different activations depending on the task: **(i)** the introductory DEQN examples (Brock–Mirman, {ref}`ch-deqn`, {ref}`sec-bm`) use *ReLU* for its simplicity and fast training; **(ii)** the multi-country IRBC model ({ref}`ch-irbc`) and the deep surrogate ({ref}`ch-estimation`) use *Swish* ($z\sigma(z)$) for its smooth gradients; **(iii)** all PINN examples ({ref}`ch-pinn`) use *$\tanh$*, whose $C^\infty$ smoothness is essential when the loss involves second-order derivatives. For ReLU-family networks, He initialization (`kaiming_normal_` in PyTorch, `he_normal` in Keras) is the natural default. For $\tanh$ and sigmoid networks, Xavier/Glorot initialization is usually the cleaner starting point; Swish and related smooth non-monotone activations often work well with either He-style or Xavier-style scaling, so the notebooks state the chosen initializer explicitly when it matters.
```


(sec-activations)=
## Activation Functions in Depth
Beyond the three classical choices (sigmoid, tanh, ReLU), several modern activation functions address specific shortcomings. {numref}`tab-activations` summarizes the options used in this course.

````{table}
:name: tab-activations

Activation functions used throughout the course. Origin papers: ReLU {cite:p}`nair2010rectified`, Leaky ReLU {cite:p}`maas2013rectifier`, ELU {cite:p}`clevert2016elu`, Swish {cite:p}`ramachandran2017swish`, GELU {cite:p}`hendrycks2016gelu`, Mish {cite:p}`misra2019mish`. *Range* is the set of output values for $z \in \R$. *Smoothness* matters when derivatives of the network output are needed: sigmoid, tanh, Swish, GELU, Mish, and softplus are $C^\infty$; ReLU is only $C^0$; Leaky ReLU is piecewise linear; ELU is piecewise $C^\infty$ and is $C^1$ at the origin only for $\alpha=1$. Smooth activations are required for PINN applications that involve second-order derivatives ({ref}`ch-pinn`).

| **Activation** | **Formula** | **Range** | **Key property** |
|---|---|---|---|
| Sigmoid | $\sigma(z) = (1+e^{-z})^{-1}$ | $(0,1)$ | Smooth, saturates |
| Tanh | $\tanh(z)$ | $(-1,1)$ | Zero-centered, saturates |
| ReLU | $\max(0,z)$ | $[0,\infty)$ | Non-saturating for $z>0$ |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha\!=\!0.1$ | $(-\infty,\infty)$ | No dead neurons |
| ELU | $z\text{ if }z>0;\ \alpha(e^z-1)\text{ else}$ | $[-\alpha,\infty)$ | Negative saturation; $C^1$ if $\alpha=1$ |
| Swish | $z \cdot \sigma(z)$ | $\approx [-0.28,\infty)$ | Smooth, non-monotone |
| GELU | $z\cdot\Phi(z)$ | $\approx [-0.17,\infty)$ | Smooth, default in BERT / GPT |
| Mish | $z\cdot\tanh(\ln(1+e^z))$ | $\approx [-0.31,\infty)$ | Smooth, used in YOLOv4 |
| Softplus | $\ln(1+e^z)$ | $(0,\infty)$ | Smooth ReLU approximation |
````

Leaky ReLU and ELU address the dying-neuron issue by providing a small but nonzero gradient for negative inputs. The Swish activation $\mathrm{swish}(z) = z\sigma(z)$ {cite:p}`ramachandran2017swish`, which is used extensively in the DEQN and IRBC implementations of this course, combines the benefits of ReLU (non-saturating for large $z$) with smoothness at the origin. Its derivative $\mathrm{swish}'(z) = \sigma(z) + z\sigma(z)(1-\sigma(z))$ is smooth everywhere and bounded between approximately $-0.1$ and $1.1$, which can improve optimization stability.

For PDE applications ({ref}`ch-pinn`), the choice of activation function is particularly important because the PINN loss involves derivatives of the network output. Since $\mathrm{ReLU}''(z) = 0$ almost everywhere, a ReLU network cannot represent second-order PDE residuals faithfully. Smooth activations such as $\tanh$ ($C^\infty$) or Swish are therefore required for PINN applications involving second-order PDEs. {numref}`fig-activations` plots seven representative activations from {numref}`tab-activations`.

```{figure} figures/fig-activations.svg
:name: fig-activations

Seven representative activation functions from {numref}`tab-activations`. Sigmoid and tanh saturate at large $|z|$ (vanishing gradients); ReLU is non-saturating but kinked at the origin; Leaky ReLU and ELU repair the dead-neuron problem with a small negative response; Swish and Softplus are everywhere $C^\infty$, which the PINN chapter ({ref}`ch-pinn`) requires.
```

(sec-vanishing)=
## Vanishing and Exploding Gradients
A central obstacle to training deep networks is that the gradient signal reaching early layers can become either astronomically small or astronomically large as it is back-propagated through many layers. The backward recursion derived above for the "delta" vector is

$$
\bm{\delta}^{(l)} = \bigl((\W^{(l+1)})^\top \bm{\delta}^{(l+1)}\bigr) \odot g'(\z^{(l)}),
$$

so the magnitude of $\bm{\delta}^{(l)}$ is governed, roughly, by the product of derivatives $\prod_{k=l}^{L} g'(\z^{(k)})$ and the norms $\|\W^{(k)}\|$. Two symmetric failure modes follow:

- **Vanishing gradients.** For the sigmoid activation, $|g'(z)| \leq 1/4$; for $\tanh$, $|g'(z)| \leq 1$; and both derivatives approach zero when $|z|$ is large. In the worst sigmoid case each layer shrinks the gradient by a factor close to $1/4$, so after $L=10$ layers the signal at the first layer can be attenuated by roughly $(1/4)^{10}\approx 10^{-6}$. Early layers stop learning.

- **Exploding gradients.** If $|g'(z)| > 1$ or if $\|\W^{(k)}\|$ is large, the product grows instead of shrinking. Updates become huge, parameters diverge, and the loss "blows up".

Three ingredients, each already introduced separately, combine to tame these problems:

1.  **Non-saturating activations.** ReLU has $g'(z) = 1$ for $z > 0$, eliminating the $(1/4)^L$ decay; Swish and tanh avoid vanishing when activations remain in a moderate range.

2.  **Variance-preserving initialization.** Xavier/Glorot {cite:p}`glorot2010understanding` and He {cite:p}`he2015delving` pick $\mathrm{Var}[W] \propto 1/n_\mathrm{in}$ precisely so that $\mathrm{Var}[\z^{(l)}]$ is constant across layers, keeping activations in the useful range of $g$ ({ref}`sec-weight_init`).

3.  **Batch normalization** {cite:p}`ioffe2015batch`. Re-centering and re-scaling the pre-activations of each mini-batch prevents them from drifting toward the saturated tails of $g$ during training and allows much larger learning rates. Its affine parameters $(\gamma, \beta)$ are learned.

A practical complement is *gradient clipping*: if $\|\nabla_{\bm{\theta}} J\|$ exceeds a threshold, rescale it to the threshold. This eliminates the most damaging exploding-gradient events at negligible cost and is standard in RNN training; it is occasionally useful in DEQNs and PINNs when the residual magnitudes are highly unbalanced across collocation points.

```{prf:remark} Course-wide implication

 Everywhere in this course where we train a network of non-trivial depth ({ref}`ch-deqn`–{ref}`ch-pinn`), the combination of *He/Xavier initialization*, a *smooth non-saturating activation* (ReLU, Swish, tanh), and Adam's *per-parameter adaptive step* keeps the gradient flow well conditioned. Batch normalization is used when depth exceeds roughly ten layers or when the input distribution shifts substantially during training.
```


(sec-batchnorm)=
### Batch Normalization
Among the three mitigations listed above, batch normalization {cite:p}`ioffe2015batch` (BN) deserves a closer look because it is simple to state, surprisingly effective in practice, and has become a default building block in supervised deep networks. At its core BN is the standardization trick familiar from regression, re-centering each variable to mean zero and rescaling it to unit variance, applied separately to every layer's pre-activations and recomputed on every mini-batch of training data.

Let $z_1,\dots,z_B$ denote the pre-activations at one neuron over the $B$ examples in a mini-batch $\mathcal B$. Batch normalization replaces them by

$$
\mu_{\mathcal B} = \frac{1}{B}\sum_{i=1}^{B} z_i,
\quad
\sigma^2_{\mathcal B} = \frac{1}{B}\sum_{i=1}^{B} (z_i-\mu_{\mathcal B})^2,
\quad
\hat z_i = \frac{z_i-\mu_{\mathcal B}}{\sqrt{\sigma^2_{\mathcal B}+\varepsilon}},
\quad
y_i = \gamma\,\hat z_i + \beta,
$$ (eq-batchnorm)

where $\varepsilon$ is a small constant for numerical stability and $(\gamma,\beta)$ are *learnable* scalar parameters specific to that neuron. The transformed activation $y_i$ is what the next layer sees. In the standard recipe BN is inserted between the linear map $\W^{(\ell)}\bm a^{(\ell-1)}+\bm b^{(\ell)}$ and the elementwise nonlinearity $g(\cdot)$.

**Why standardization, layer by layer.** Without BN, the input distribution to a hidden layer $\ell$ depends on every weight in layers $1,\dots,\ell-1$. As earlier weights update during gradient descent, the distribution faced by layer $\ell$ *drifts* from one optimization step to the next: each layer therefore chases a moving target, a phenomenon {cite:t}`ioffe2015batch` called *internal covariate shift*. BN pins the input distribution of every layer to mean zero and unit variance at every step ({numref}`fig-batchnorm_intuition`). Gradients become better conditioned, and substantially larger learning rates become safe.

**The role of the affine parameters.** At first glance the learnable shift and scale $(\gamma,\beta)$ seem to undo the normalization that BN just imposed. This is exactly the point. If a layer happens to prefer non-standard inputs, for example a tanh layer that needs slightly negative pre-activations to operate in its linear regime, the network is free to recover them via $(\gamma,\beta)$. BN therefore never reduces the network's representational capacity; it merely shifts to a parameterization in which the optimization trajectory is easier to follow.

```{figure} figures/fig-batchnorm_intuition.svg
:name: fig-batchnorm_intuition

Distribution of pre-activations at one hidden neuron, sampled at three points during training. *Left:* without BatchNorm, the distribution drifts in mean and in variance as earlier layers update, each layer chases a moving target. *Right:* with BatchNorm, the affine pre-normalization transformation pins the inputs to $\mathcal{N}(0,1)$ at every training step, before the learned scale $\gamma$ and shift $\beta$ are applied. The downstream layer always operates on inputs of the same scale, and the gradient signal flowing back is well conditioned.
```

**Why higher learning rates work.** A precise Lipschitz bound depends on the operator norms of the surrounding weight matrices, the activation derivatives, and the learned affine scale $\gamma$. The useful intuition is that BN reduces sensitivity to shifts and rescalings of intermediate activations, making the local optimization problem better conditioned and allowing step sizes that would otherwise cause divergence. {cite:t}`santurkar2018how` argue that loss-landscape smoothing, rather than the original "internal covariate shift" interpretation, better explains why BN helps optimization; the two views are complementary, but the smoothing perspective is the more directly testable one.

**At inference time.** During training, BN uses the current mini-batch to compute $\mu_{\mathcal B},\sigma^2_{\mathcal B}$. At inference the mini-batch may be a single example, in which case those statistics would be ill-defined. Implementations therefore maintain a running average of $\mu$ and $\sigma^2$ across training mini-batches and use these fixed estimates at test time, so the network's output is deterministic at deployment.

(sec-normalization_variants)=
### Normalization Variants Beyond BatchNorm
Batch normalization is the original normalization trick, but for several common deep-learning settings, small mini-batches, recurrent / sequence models, generative models, transformers, it is not the right one. All variants share the form $\hat z = (z-\mu)/\sigma$ followed by a learned affine $\gamma\hat z + \beta$; they differ only in *which axes* the statistics $(\mu,\sigma)$ are pooled over:

- **LayerNorm** {cite:p}`ba2016layer`: pool across all features of a single example. Decouples training from batch size and is the de-facto choice for RNNs and transformers; it can also be useful in small-batch residual methods, though PINNs more commonly rely on careful input/output scaling and smooth activations.

- **GroupNorm** {cite:p}`wu2018group`: pool over a group of channels of a single example. Interpolates between LayerNorm ($G\!=\!1$) and InstanceNorm ($G$ = channels); the default in object detection and small-batch CNNs.

- **WeightNorm** {cite:p}`salimans2016weight`: reparameterize $\bm w = (g/\|\bm v\|)\,\bm v$ so the activation statistics never enter the gradient. Avoids any batch-size dependence at the cost of slightly less representation power.

For the methods in this script, the practical guidance is: use BatchNorm for large supervised datasets ({ref}`ch-intro`), LayerNorm for recurrent or attention-based architectures and as an option when the effective batch size is small, and skip normalization entirely for small DEQN or PINN MLPs when input/output scaling plus Adam already condition the problem well.

(sec-generalization)=
## Generalization: Overfitting, Regularization, and Double Descent
(sec-traintest)=
### Train / Validation / Test Split
Before discussing overfitting formally, it is essential to fix the experimental protocol that every supervised-learning study should follow. The available data are partitioned into three disjoint subsets:

- **Training set** (typically $\sim 70\%$): used to fit the model parameters $\bm{\theta}$ by minimizing the loss.

- **Validation set** (typically $\sim 20\%$): used for *model selection*, choosing hyperparameters, comparing architectures, deciding when to stop training. The model's performance on this set guides these decisions but its parameters are never trained on it.

- **Test set** (typically $\sim 10\%$): touched *once*, at the very end, to report an unbiased estimate of generalization performance.

The key discipline is that no decision about the model (not hyperparameter tuning, not architecture, not early-stopping patience) may be informed by the test set. Using the test set multiple times turns it into an implicit validation set and invalidates its role as a measure of out-of-sample error. For small datasets, $k$-fold cross-validation replaces the fixed train/validation split: the training data are partitioned into $k$ equal folds; for each fold, the model is trained on the other $k-1$ folds and evaluated on the held-out fold; the $k$ resulting validation scores are averaged. Common choices are $k = 5$ or $k = 10$; the test set is always held separately. In DEQNs and PINNs ({ref}`ch-deqn` and {ref}`ch-pinn`), training and validation points are drawn from the same state distribution, and "generalization" is measured against an *independently simulated test trajectory* rather than a held-out labeled set.

A model that memorizes the training data but fails on unseen examples is said to *overfit*. To understand overfitting precisely, consider the following thought experiment. The decomposition below is the classical bias/variance analysis of {cite:t}`geman1992biasvariance`, which provided the canonical framework for thinking about generalization in neural networks long before modern overparameterized regimes were studied. Suppose we draw many independent training sets $\mathcal{D}$, each of size $n$, from the same data-generating process $y = f(\x) + \varepsilon$, where $\varepsilon$ is zero-mean noise with variance $\sigma^2$. On each training set we fit our model, obtaining a predictor $\hat{f}_{\mathcal{D}}$. Conditioning on a fixed test input $\x_0$ and averaging over both the training set and the new test noise $\varepsilon_0$, the squared prediction error decomposes into exactly three terms:

$$
\begin{align}
\mathbb{E}_{\mathcal{D},\varepsilon_0}
\!\bigl[(y_0 - \hat{f}_{\mathcal{D}}(\x_0))^2\bigr]
&=
\underbrace{\bigl(f(\x_0) - \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(\x_0)]\bigr)^2}_{\text{Bias}^2}
\notag\\
&\quad+
\underbrace{\mathbb{E}_{\mathcal{D}}\!\bigl[(\hat{f}_{\mathcal{D}}(\x_0) - \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(\x_0)])^2\bigr]}_{\text{Variance}}
\;+\;
\underbrace{\sigma^2}_{\text{Irreducible noise}}.
\label{eq-bias-variance}
\end{align}
$$

Each term captures a distinct source of error:

- **Bias$^2$** measures the systematic error: how far the *average* prediction $\mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(\x_0)]$ is from the true function $f(\x_0)$. A model that is too simple (e.g., a linear function fit to a nonlinear target) will have high bias regardless of how much data it sees.

- **Variance** measures the sensitivity of the prediction to the particular training set drawn. A highly flexible model (e.g., a large neural network) may fit each training set well, but the predictions can differ wildly across draws, which is overfitting.

- **Irreducible noise** $\sigma^2$ is the error that no model can eliminate, because it stems from randomness in the data-generating process itself.

In classical statistics, there is a fundamental trade-off: reducing bias requires more flexible models, which increases variance. However, modern deep networks challenge this picture. {cite:t}`zhang2017understanding` showed empirically that standard architectures can perfectly fit randomly labeled data, implying a VC-style capacity far beyond what classical bounds predict, yet still generalize well on real data. {cite:t}`belkin2019reconciling` subsequently demonstrated that with sufficient overparameterization, models exhibit a *double descent* phenomenon: test error first increases as the model becomes more complex (classical regime), but then decreases again as the number of parameters greatly exceeds the number of data points (interpolation regime). {cite:t}`nakkiran2020deep` showed that this phenomenon extends to deep networks and occurs not only as a function of model size, but also as a function of training time ("epoch-wise double descent") and dataset size.

The key techniques for preventing overfitting in neural networks are:

1.  **Early stopping** {cite:p}`prechelt1998early`: monitor validation loss and stop training when it begins to rise.

2.  **Weight decay ($L_2$ regularization)** {cite:p}`krogh1991simple`: add $\frac{\lambda}{2}\|\bm{\theta}\|^2$ to the loss.

3.  **Dropout** {cite:p}`srivastava2014dropout`: randomly drop a fraction $p$ of activations at every training step. Two implementation conventions exist. The original convention drops units during training and multiplies the outgoing weights or activations by the keep probability $1-p$ at test time, so that the expected activation matches the training-time expectation. The now-standard *inverted-dropout* convention divides the retained activations by $1-p$ during training, so no rescaling is needed at test time. Either way, the mechanism is equivalent to training, on each mini-batch, a different sub-network drawn from an exponentially large ensemble that shares weights; the final network approximates the ensemble average. Typical values are $p=0.5$ for hidden layers and $p=0.1$–$0.2$ for inputs. Dropout is less commonly used in DEQN and PINN applications, where the loss is already noisy (stochastic collocation) and regularization is often supplied implicitly by the state-space sampling scheme.

4.  **Data augmentation:** synthetically enlarge the training set via transformations.

5.  **Batch normalization** {cite:p}`ioffe2015batch`: normalize activations within each mini-batch to stabilize training; its mini-batch statistics also act as a mild regularizer.

```{figure} figures/fig-double_descent.svg
:name: fig-double_descent

Schematic of the double-descent phenomenon. In the classical regime ($p < n$) test error follows the standard bias–variance U-curve; around the interpolation threshold $p \approx n$ test error can peak sharply because the fitted function is highly sensitive to noise; in the modern overparameterized regime ($p \gg n$) test error decreases again {cite:p}`belkin2019reconciling,nakkiran2020deep`. In some linearized, kernel, max-margin, or least-norm settings, gradient methods exhibit an implicit bias toward particular low-complexity interpolants; in nonlinear finite-width networks this bias depends on architecture, data, optimizer, initialization, and training protocol. Axes are unitless; the qualitative shape, not the scale, is the point. The curve is illustrative, not a measurement.
```

{numref}`fig-double_descent` illustrates why classical bias–variance intuition breaks down for modern deep networks. In the classical regime ($p < n$), increasing model capacity beyond a point leads to overfitting. At the interpolation threshold ($p \approx n$), the model has just enough parameters to perfectly fit the training data, and the resulting solution can be extremely sensitive to noise. In the modern regime ($p \gg n$), test error often decreases again because optimization and architecture bias select comparatively regular interpolating solutions rather than arbitrary ones {cite:p}`belkin2019reconciling`.

This phenomenon has been documented across many architectures and datasets by {cite:t}`nakkiran2020deep`, who showed that it persists even when controlling for effective model complexity. The implications for computational economics are substantial but should not be overstated. In DEQN and PINN applications, the practitioner controls both the network size (number of parameters $p$) and the amount of training data (number of collocation points $n$), and those collocation points are often resampled rather than fixed once and for all. Overparameterized networks can therefore be useful and sometimes necessary, but their credibility must be checked by independent residual diagnostics, simulated trajectories, and benchmark comparisons rather than by parameter counting alone.

The double descent phenomenon can be explored interactively in the companion notebook [`03_Double_Descent.ipynb`](notebooks/lecture_02_intro_deep_learning/lecture_02_03_Double_Descent.ipynb), which reproduces the spirit of the double-descent curve in a small kernel-regression / random-Fourier-features setting ({cite:t}`belkin2019reconciling`); see also {cite:t}`nakkiran2020deep` for the deep-network / CIFAR-10 version of the experiment, which the notebook does *not* attempt to replicate at scale.

(sec-ntk_pointer)=
### A Pointer to the Theory: Neural Tangent Kernel (NTK) and Benign Overfitting
Why *does* an over-parameterized network with $p \gg n$ generalize instead of merely memorizing? Two complementary lines of theory have emerged.

**Neural Tangent Kernel (NTK).** In the limit of infinite width and a particular initialization scaling, gradient-descent training of a deep network is described by kernel gradient descent with a fixed, deterministic kernel, the *Neural Tangent Kernel* of {cite:t}`jacot2018neural`. In that lazy-training limit the network's function evolves almost linearly around its initialization, which explains why first-order optimization can be well behaved for very wide networks even though the finite-parameter objective is non-convex.

**Benign overfitting.** In linear and kernel settings, gradient methods often select a minimum-norm interpolating solution, which can behave much like a ridge-regularized least-squares estimator and inherit good generalization properties. {cite:t}`bartlett2020benign` make this precise for linear regression, showing that interpolation can be benign provided the spectrum of the input covariance has heavy enough tails. Subsequent work has extended both stories beyond the simplest settings; for the practitioner the takeaway is that the NTK regime helps explain *why training succeeds*, while benign-overfitting theory explains why interpolation need not imply poor test error under additional assumptions.

These two threads are not the final word: finite-width deviations from the NTK matter for feature learning, and benign overfitting requires conditions on the data covariance. They nevertheless help explain why the rest of this script can use networks substantially wider than a classical degrees-of-freedom calculation would recommend, provided the numerical residuals are validated out of sample.

(sec-sequence_models)=
## Sequence Models: RNNs, LSTMs, and Attention
```{prf:remark} Optional section

 This section and the in-context AR(1) aside ({ref}`sec-incontext_ar1`) survey sequence architectures that the rest of the script does not use: {ref}`ch-deqn`–{ref}`ch-climate` operate on unstructured state vectors and rely entirely on feedforward MLPs. Readers focused on DEQN, PINN, and the structural-estimation chapters can skip directly to the Chapter Summary without loss of continuity. The material below is included for completeness and as a reference for readers who later encounter Transformers in empirical-finance or applied-ML work.
```


The chapters that follow rely almost entirely on feedforward networks, but many economic and financial datasets are intrinsically sequential. Before closing this introduction, it is therefore useful to briefly survey the main architectures for sequence data. We represent a sequence as *tokens* $\x_1, \dots, \x_n$, where each token is a vector. The word "token" is borrowed from natural-language processing, where a token is typically a word or sub-word piece; in the economic and financial context we use throughout this course a token is simply one element of the sequence, for example a scalar return, a vector of macroeconomic variables at one quarter, or a price-volume pair at one tick. The algorithms below are agnostic to this choice; they only need the sequence elements to be real-valued vectors of a fixed dimension.

### Recurrent Neural Networks

The traditional approach to sequence data is the *Recurrent Neural Network* (RNN) {cite:p}`elman1990finding`. Unlike an MLP, which maps an input to an output in a single forward pass, an RNN maintains an internal *hidden state* $\h_t$ that acts as a memory of past information. At each time step $t$, the network updates this state using the current input $\x_t$ and the previous state $\h_{t-1}$:

$$
\h_t = \sigma(\Wh \h_{t-1} + \Wx \x_t + \bb),
$$

where $\sigma$ is an activation function. Concretely: for a scalar time series $\x_t$ is a scalar (e.g. log-return at date $t$), $\h_t \in \R^d$ is a $d$-dimensional hidden vector summarizing everything the network has seen so far, and $\Wh,\Wx,\bb$ are learnable parameters. The same update is applied at every time step, so this recursive structure lets the network process sequences of arbitrary length with a fixed parameter budget. {numref}`fig-rnn` shows the resulting unrolled computation graph.

```{figure} figures/fig-rnn.svg
:name: fig-rnn

An unrolled Recurrent Neural Network. The same parameters $\Wh, \Wx$ are reused at every time step, allowing the hidden state $\h_t$ to accumulate historical information.
```

**Training: Backpropagation Through Time (BPTT).** Because the unrolled RNN *is* a feedforward graph of depth $T$ with *shared* weights, one can apply ordinary backpropagation and then *sum* the weight-gradients across time. Concretely, let $\mathcal{L}_T = \sum_{t=1}^{T}\ell(\hat{\y}_t, \y_t)$ denote the total loss. For the recurrence above, the forward single-step Jacobian is

```{math}
:enumerated: false

J_t \equiv \frac{\partial \h_t}{\partial \h_{t-1}}
= \mathrm{diag}\!\bigl(\sigma'(\Wh\h_{t-1}+\Wx\x_t+\bb)\bigr)\Wh.
```

With column gradients, the backward pass multiplies by $J_t^\top$. Differentiating with respect to an early hidden state $\h_k$ therefore yields the schematic product

$$
\nabla_{\h_k}\mathcal{L}_T
\;\approx\;
J_{k+1}^{\top}J_{k+2}^{\top}\cdots J_T^{\top}\,\nabla_{\h_T}\mathcal{L}_T,
$$ (eq-bptt)

so the gradient picks up $T-k$ products of matrices containing the same recurrent weight matrix $\Wh$. The relevant singular values or spectral radii of these factors determine the asymptotics. If they are mostly below one, the gradient *vanishes* exponentially in $T-k$ and the network cannot learn dependencies that span many steps; if they are mostly above one, it *explodes*, producing NaNs during training. This is the *vanishing/exploding gradient problem*, originally analyzed by {cite:t}`hochreiter1991untersuchungen` and developed formally in {cite:t}`bengio1994learning` {cite}`hochreiter2001gradient`, and revisited with a modern optimization-theoretic view by {cite:t}`pascanu2013difficulty`.

Three practical remedies partially alleviate the pathology without changing the architecture. *Gradient clipping* {cite:p}`pascanu2013difficulty` rescales the parameter-gradient whenever its norm exceeds a threshold; it eliminates the worst exploding events at negligible cost and is now a standard training default for any recurrent model. *Orthogonal or identity-like initialization* of $\Wh$ places its singular values near $1$ so that the Jacobian product preserves norms at the start of training. *Truncated BPTT* unrolls only $K$ steps back at each update, capping both the memory footprint and the effective gradient horizon at $K$. These help but do not cure the problem: the structural fix is to change the recurrence itself, which is the move made by *gated cells*.

(sec-lstm)=
### Long Short-Term Memory (LSTM) and GRUs
Before writing equations, it helps to keep a plain-language picture in mind. A vanilla RNN asks one hidden state $\h_t$ to do three jobs at once: retain useful old information, absorb new information, and expose the relevant part to the next layer. This is a fragile design. The LSTM separates these tasks. It keeps a dedicated *memory lane* $\bm{C}_t$ flowing through time and at each date makes three soft decisions: what to keep, what to write, and what to reveal. That is the intuition behind the gate equations below.

The LSTM cell {cite:p}`hochreiter1997long` replaces the single-state recurrence $\h_t = \sigma(\Wh\h_{t-1} + \Wx\x_t)$ by a pair of states: a *cell state* $\bm{C}_t$ that flows along the top of the cell with *additive* updates, and a *hidden state* $\h_t$ that is read off it. Three learned sigmoid gates, each depending on the concatenation $[\h_{t-1}, \x_t]$, control what information flows through:

$$
\bm{f}_t = \sigma\!\big(\W_f\,[\h_{t-1}, \x_t] + \bb_f\big)  \text{(forget gate)}
$$ (eq-lstm_f)

$$
\bm{i}_t = \sigma\!\big(\W_i\,[\h_{t-1}, \x_t] + \bb_i\big)  \text{(input gate)}
$$

$$
\tilde{\bm{C}}_t = \tanh\!\big(\W_C\,[\h_{t-1}, \x_t] + \bb_C\big)  \text{(candidate cell)}
$$

$$
\bm{C}_t = \bm{f}_t \odot \bm{C}_{t-1} + \bm{i}_t \odot \tilde{\bm{C}}_t  \text{(cell state update)}
$$ (eq-lstm_C)

$$
\bm{o}_t = \sigma\!\big(\W_o\,[\h_{t-1}, \x_t] + \bb_o\big)  \text{(output gate)}
$$

$$
\h_t = \bm{o}_t \odot \tanh(\bm{C}_t)  \text{(hidden state).}
$$

Each of $\bm{f}_t, \bm{i}_t, \bm{o}_t \in (0,1)^{d}$ acts as a soft switch applied element-wise. The crucial structural change is in equation {eq}`eq-lstm_C`: the cell state is *additively* corrected rather than multiplicatively overwritten. Along the direct memory path, differentiating $\bm{C}_t$ with respect to $\bm{C}_{t-1}$ contributes $\mathrm{diag}(\bm{f}_t)$ in place of a full recurrent matrix product. The full derivative also contains indirect terms because the gates depend on $\h_{t-1}$ and hence on earlier cell states, but the direct path is the constant-error-carousel intuition: when the cell judges information worth keeping, it can open the forget gate ($\bm{f}_t \approx \bm{1}$) and allow gradients to flow through *as if* the sequence were shorter. {numref}`fig-lstm_cell` sketches the resulting cell.

```{figure} figures/fig-lstm_cell.svg
:name: fig-lstm_cell

The LSTM cell. The green top lane is the *protected memory lane*: old memory can be kept, new information can be written additively, and the resulting state can later be revealed through the hidden output (*visible output*, blue bottom lane). This is the intuition behind the forget, input, candidate, and output components shown inside the cell.
```

The *Gated Recurrent Unit* (GRU) of {cite:t}`cho2014gru` is a lighter sibling that merges the forget and input gates into a single *update* gate $\bm{z}_t$ and drops the separate cell state in favor of the hidden state itself:

$$
\begin{align}
\bm{z}_t       &= \sigma(\W_z\,[\h_{t-1}, \x_t] + \bb_z), \\
\bm{r}_t       &= \sigma(\W_r\,[\h_{t-1}, \x_t] + \bb_r), \\
\tilde{\h}_t   &= \tanh\!\big(\W_h\,[\bm{r}_t \odot \h_{t-1}, \x_t] + \bb_h\big), \\
\h_t           &= (1 - \bm{z}_t) \odot \h_{t-1} + \bm{z}_t \odot \tilde{\h}_t.
\end{align}
$$

A GRU uses roughly $25\%$ fewer parameters than an LSTM of the same hidden size and performs comparably on many sequence tasks {cite:p}`chung2014empirical`, at the cost of a slightly less expressive memory channel.

LSTMs remain particularly effective for economic time-series tasks where capturing specialized temporal patterns is more important than massive scalability. For example, {cite:t}`holt2024detecting` use LSTMs to detect *Edgeworth cycles* in retail gasoline-price data. These cycles are asymmetric, high-frequency price movements that are difficult to identify with traditional spectral analysis or simple rule-based methods. In this setting the LSTM architecture excels at identifying the characteristic sawtooth patterns (sudden price jumps followed by slow decays) across thousands of retail stations, providing a robust tool for antitrust analysis and competition policy.

(sec-rnn_limits)=
### Limits of Recurrent Models
Even with gated cells, recurrent architectures retain two fundamental limitations that no amount of tuning can remove.

1.  **Inherently sequential.** Step $t$ cannot begin until step $t-1$ has finished, so the full computation takes $T$ serial wavefronts regardless of how many cores or tensor units are available. On a modern GPU that can evaluate thousands of MLP rows in parallel, this forced serialization makes RNN training slow and wall-clock expensive. Training a contemporary large language model on a trillion-token corpus with a plain LSTM would take years of wall-clock time rather than weeks.

2.  **Path-length-dependent decay.** Information from position $1$ must travel through all intermediate hidden states to influence position $T$, each hop applying a matrix and a nonlinearity. Gating mitigates but does not eliminate this decay, and empirically LSTM performance saturates on sequences well below the theoretical limit implied by the cell's capacity.

Both problems have the same structural cause: the recurrence forces a path of length $T$ between the two extreme positions of the sequence. The remedy is to drop recurrence entirely and let every position read every other position *directly*, in parallel. This is the Transformer.

(sec-transformers)=
### The Transformer Architecture
The *Transformer* architecture {cite:p}`vaswani2017attention` replaced recurrence entirely with *self-attention*. This allows the model to weigh the importance of all tokens in a sequence simultaneously, enabling massive parallelization and better capturing global dependencies.[^1]

For a first pass, four steps are enough. Add position information so the model knows order; project each token into a *query*, *key*, and *value*; compare each query to all keys; then take a weighted average of the values. Multi-heads, LayerNorm, residual connections, and pointwise MLPs are refinements of this core idea, not a different idea.

(sec-self_attention)=
#### The Self-Attention Mechanism
**Intuition: search, then retrieval.** Before writing the mechanism formally it is useful to see it as a *soft library lookup*. Picture a small library with $n$ shelves, one per position in the sequence. Each shelf $i$ carries three vectors: a *query* $q_i$ ("what am I looking for?"), a *key* $k_i$ ("what is printed on my label?"), and a *value* $v_i$ ("what do I actually contain?"). All three are linear projections of the same input token $\x_i$, and the projection matrices $\W_Q,\W_K,\W_V$ are *learned* during training; the librarian (queries and keys) and the books (values) are co-designed for whatever task the training objective encodes.

To produce the updated representation $\bm{o}_i$ at shelf $i$, the following four steps happen.

1.  **Score.** Shelf $i$'s query $q_i$ is compared against every shelf's key $k_j$; the dot product $q_i^{\!\top} k_j$ is a *similarity score*, high when shelf $j$'s label matches what shelf $i$ is looking for.

2.  **Normalize.** The $n$ scores are divided by $\sqrt{d_k}$ (to keep them in a numerically sane range) and pushed through a softmax, producing probability weights $\alpha_{ij}\in[0,1]$ with $\sum_j \alpha_{ij}=1$.

3.  **Retrieve.** The weights are applied to the values: $\bm{o}_i = \sum_j \alpha_{ij}\, v_j$ is a weighted average of the $n$ shelf contents.

4.  **Repeat.** Steps 1–3 happen in parallel for every shelf $i$, producing the whole output sequence $\bm{o}_1,\dots,\bm{o}_n$ in a single layer.

Shelves whose label matched the query contribute most to the output; shelves whose label was off-topic are almost ignored.

**Why this is useful: a worked example.** Consider the sentence *"The cat sat on the mat. It purred."* For the model to process "it" properly, it must first decide what "it" refers to, the cat or the mat. Self-attention performs exactly this disambiguation: the query vector at the "it" position probes the key vectors at every earlier position, and the softmax converts the raw similarity scores into probability weights that concentrate most of the mass on the correct antecedent. {numref}`fig-attention` below illustrates the resulting pattern: the bulk of the weight lands on "cat", and the updated representation at the "it" position is formed as a weighted average of the values, driven mostly by "cat". The same mechanism, run in parallel for every position, produces all of $\bm{o}_1,\dots,\bm{o}_n$ in a single layer.

**A small concrete example.** Suppose we have only three shelves and two-dimensional $q$'s and $k$'s. Take the query at shelf 3 to be $q_3 = (1,0)$ and the three keys $k_1 = (0.9,\,0.1),\ k_2 = (0.1,\,0.8),\ k_3 = (1,\,0).$ The raw similarity scores are $q_3^{\!\top}k_1 = 0.9$, $q_3^{\!\top}k_2 = 0.1$, $q_3^{\!\top}k_3 = 1.0$. Ignoring the $1/\sqrt{d_k}$ scaling to keep the arithmetic clean, the softmax weights come out to roughly $(0.405,\,0.182,\,0.448)$. Shelf 3's output $\bm{o}_3$ is therefore a blend of the three values in those proportions: most mass on shelf 3 itself and on shelf 1 (the closest match in label space); shelf 2, whose label $k_2$ points in an orthogonal direction, contributes about $18\%$. The softmax is "soft" precisely in this sense, a smoothed nearest-neighbor retrieval rather than a hard argmax over the most similar shelf.

The "self" in *self-*attention says that every shelf plays both roles simultaneously: every shelf's query probes every shelf's key, and every shelf receives an updated representation as output. A single attention layer therefore pairs all $n$ positions with all $n$ positions in parallel. Contrast this with an RNN or LSTM: to let position $t$ peek at information from position 1, the signal must be shuttled through every intermediate position via the hidden state, with each hop applying its own weight matrix and nonlinearity, so long-range information is either blurred or lost entirely. Attention short-circuits that chain: position $t$ reads position $1$ directly, with no intermediate hops and no path-length-dependent attenuation. This is the architectural source of the Transformer's advantage on long sequences, and the reason why attention-based models quickly displaced recurrent ones for language, code, and (increasingly) time-series forecasting.

Given a sequence of input vectors, stack the tokens as rows of $\X \in \R^{n\times d}$. The attention mechanism computes three projections for each token: a *Query* ($Q$), a *Key* ($K$), and a *Value* ($V$), using learned weight matrices $\W_Q, \W_K, \W_V$:

$$
Q = \X \W_Q, \qquad K = \X \W_K, \qquad V = \X \W_V.
$$

The output of the attention layer is a weighted sum of the values, where the weights are determined by the compatibility (dot product) of the queries with the keys:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

The scaling factor $\sqrt{d_k}$ (the dimensionality of the keys) prevents the dot products from growing too large in magnitude, which would otherwise make the softmax unstable.

**An econometric lens.** The attention layer is a *data-dependent, learnable kernel smoother*. Compare it to the Nadaraya–Watson estimator $\hat f(x) = \sum_i w_i(x)\,y_i$ with kernel-based weights $w_i(x) \propto k(x, x_i)$: attention has exactly this form, but the similarity $k(\cdot,\cdot)$ is the parametric bilinear form $(q,k)\mapsto q^{\!\top}k/\sqrt{d_k}$ and both $q$ and $k$ are themselves *learned* projections of the input. From this vantage point the Transformer's "magic" is less mysterious: it is a nonparametric smoother whose kernel the optimizer tunes to whatever task the training objective encodes. Self-attention further recovers the classical recurrence-free property that every pair of positions interacts in a single parallel layer, with no signal decay along the sequence.

{numref}`fig-attention` renders the attention pattern of the worked "cat/it" example on a compressed five-token version of the sentence. The output $\bm{o}_{\textit{it}}$ is the new representation at the "it" position, formed as a weighted average of the values, with most weight coming from "cat".

```{figure} figures/fig-attention.svg
:name: fig-attention

A worked self-attention pattern. Both $q_{\textit{it}}$ and $\bm{o}_{\textit{it}}$ sit above the "it" position: $q_{\textit{it}}$ is the query projection of "it" (blue), and $\bm{o}_{\textit{it}}$ is the updated representation at that same position (green). The blue arrows show how the query is built from "it" and then compared with every key. The softmax weights above each token sum to one; here the largest weight lands on "cat", so the green aggregation arrow starts above "cat" and curves up to $\bm{o}_{\textit{it}}$, indicating that the update at "it" is driven mainly by the value at "cat". *How to read the arrows.* (1) Blue down arrow: build the query $q_{\textit{it}}$ from the current token "it". (2) Blue bent arrow: compare that query with all keys in the sequence. (3) Numbers above tokens: $0.05 + 0.58 + 0.08 + 0.20 + 0.09 = 1.00$, as softmax weights must. (4) The weight $0.58$ above "cat" is the largest one. (5) Green arrows: the "cat" value enters $\bm{o}_{\textit{it}}$.
```

(sec-mha)=
#### Multi-Head Attention
A single attention layer implements *one* similarity pattern between positions. Linguistic and economic sequences, however, contain many relations that matter simultaneously: subject–verb agreement, coreference of pronouns, topic alignment, or, in a macro panel, sector co-movements, shock transmission lags, and autocorrelation structure. A single head is forced to compress all of them into one kernel, and typically does a poor job.

The fix is *multi-head attention*. Run $H$ attention layers in parallel, each with its *own* projection matrices $(\W_Q^{(h)}, \W_K^{(h)}, \W_V^{(h)})$ mapping the input to a lower-dimensional subspace of size $d_k = d/H$, compute $H$ attention outputs independently, then concatenate and linearly project back:

$$
\begin{align}
\mathrm{head}_h
&= \mathrm{Attention}\bigl(\X\W_Q^{(h)},\, \X\W_K^{(h)},\, \X\W_V^{(h)}\bigr), \\
\mathrm{MHA}(\X)
&= \big[\mathrm{head}_1; \,\mathrm{head}_2; \,\ldots;\,\mathrm{head}_H\big]\,\W_O .
\end{align}
$$

The total parameter count is essentially unchanged, because each head works on a $1/H$-dimensional slice, but the inductive bias is richer: different heads are free to specialize in different relations. Interpretability studies of trained Transformers routinely find heads that focus on the previous token, on the closing bracket matching an open one, on the subject of the current clause, or, in time-series models, on the most recent analogue of the current calendar month. Multi-head attention is therefore *structurally analogous to a mixture of learned kernels* in a nonparametric regression; the weights $\W_O$ are the mixing coefficients, and the softmax-scored pairs are the kernels themselves.

(sec-pos_enc)=
#### Positional Encoding
Self-attention treats its input as a *set*: permuting the positions of the tokens permutes the rows of $\X$ and permutes the rows of the output, but changes nothing else. This is problematic, because order matters in every sequence application ("dog bites man" vs. "man bites dog"; "inflation before the shock" vs. "after"). A model without a notion of position cannot distinguish them.

The fix that {cite:t}`vaswani2017attention` propose is to *add* a deterministic vector $\mathrm{PE}(t)\in\R^d$ to the input embedding at each position $t$, chosen so that the dot products $\mathrm{PE}(t)^\top \mathrm{PE}(t+k)$ encode the relative distance $k$. The original sinusoidal scheme is

$$
\mathrm{PE}(t, 2k) = \sin\!\big(t/10000^{2k/d}\big),
\qquad
\mathrm{PE}(t, 2k+1) = \cos\!\big(t/10000^{2k/d}\big),
\qquad k=0,\ldots,d/2-1.
$$

Three properties make this choice useful. First, each coordinate $2k$ is a sine wave of wavelength $2\pi \cdot 10000^{2k/d}$, so the encoding spans wavelengths from roughly $2\pi$ up to $2\pi \cdot 10000$; the model sees both fine local positions and coarse global ones at once. Second, for each sine/cosine pair, a fixed offset can be represented by a $2\times2$ rotation: $\mathrm{PE}(t+r)$ is a linear function of $\mathrm{PE}(t)$ with coefficients depending only on the relative offset $r$. This gives attention a convenient way to represent relative positions. Third, the encoding extrapolates: a model trained on sequences of length $512$ can be fed sequences of length $1024$ and the position code is still well-defined. Modern alternatives, rotary position embeddings (RoPE) and ALiBi, refine these properties but keep the same philosophy.

(sec-transformer_block)=
#### The Transformer Block
Single attention layers, even multi-headed, are not yet expressive enough: they are essentially linear in the values, with a nonlinear mixing pattern. A full *Transformer block* wraps one MHA layer and one pointwise MLP with residual connections and LayerNorm:

$$
\x^{+} = \x + \mathrm{MHA}\!\big(\mathrm{LN}(\x)\big)
$$ (eq-tblock1)

$$
\x^{\mathrm{out}} = \x^{+} + \mathrm{MLP}\!\big(\mathrm{LN}(\x^{+})\big).
$$ (eq-tblock2)

The LayerNorm steps {cite:p}`ba2016layer` standardize across feature coordinates; together with the residual additions they stabilize training of very deep stacks. Equations {eq}`eq-tblock1`–{eq}`eq-tblock2` describe the modern *pre-norm* variant (LN before each sub-block), which is easier to train than the original *post-norm* variant of {cite:t}`vaswani2017attention`. {numref}`fig-transformer_block` shows the architecture schematically.

```{figure} figures/fig-transformer_block.svg
:name: fig-transformer_block

One *Transformer block* in pre-norm form. Self-attention first mixes information across token positions, then the pointwise MLP transforms each token separately. The red skip paths are the residual connections that let deep stacks train stably. A full Transformer stacks $L$ such blocks; GPT-3, for instance, uses $L=96$.
```

**Encoder, decoder, causal masking.** The original Transformer of {cite:t}`vaswani2017attention` pairs an *encoder* stack (processes the source sequence) with a *decoder* stack (generates the target sequence), linked by a cross-attention layer. Most modern large language models are *decoder-only*: they use the same block as above, but the attention softmax is applied to a masked score matrix that sets all entries above the diagonal to $-\infty$. This *causal mask* forbids a position from attending to future positions, turning the Transformer into a left-to-right autoregressive predictor suitable for language modeling.

**Scaling and parallelism.** For day 1 the key engineering fact is simpler than the modern LLM discussion: attention is parallel, recurrence is not. Because every sub-block is pointwise in time and the only cross-position operation (attention) is fully parallelizable, a Transformer with $L$ blocks, $H$ heads, and hidden dimension $d$ can be trained on accelerators at roughly the theoretical peak throughput. This is why modern foundation models, GPT-$n$, BERT, ViT, Claude, are Transformers rather than RNNs. Empirical studies of *scaling laws* {cite:p}`kaplan2020scaling,hoffmann2022training` document that test loss decreases as a power law in parameters, data, and compute, giving rise to the compute-optimal prescriptions used by the largest labs. For an economist, the relevant takeaway is that the marginal cost of additional capability is governed by a smooth, quantifiable, and *very large* compute bill.

#### At a glance: RNN vs LSTM vs Transformer

{numref}`tab-seq_compare` summarizes the three architectures along the dimensions most relevant to a practitioner's choice.

````{table}
:name: tab-seq_compare

Comparison of the three sequence architectures. Transformers trade a quadratic-in-$T$ attention cost for full parallelism and unit-length paths between any pair of positions, which is an excellent trade on modern accelerators and for the long sequences typical in language and high-frequency finance.

|  | **RNN** | **LSTM / GRU** | **Transformer** |
|---|---|---|---|
| Hidden state | single $\h_t$ | $\h_t$ and $\bm{C}_t$ | none per step; the residual stream across layers is an implicit state |
| Path length $1 \to T$ | $\mathcal{O}(T)$ | $\mathcal{O}(T)$ | $\mathcal{O}(1)$ |
| Parallelism over $t$ | none | none | full (all positions at once) |
| Compute per layer | $\mathcal{O}(T\,d^2)$ | $\mathcal{O}(T\,d^2)$ | $\mathcal{O}(T^2 d + T\,d^2)$ |
| Memory per layer | $\mathcal{O}(T\,d)$ | $\mathcal{O}(T\,d)$ | $\mathcal{O}(T^2 + T\,d)$ |
| Training stability | gradient decay/blow-up | much better, gated | stable with LN + residuals |
| Sweet spot | short sequences | mid-length, niche patterns | long context, massive parallelism |
````

A practical rule of thumb follows immediately. If the task is a specialized time-series problem with moderate history length and limited data, an LSTM remains a strong baseline. If context is long and accelerator-friendly parallelism matters, one should usually start with a Transformer.

(sec-incontext_ar1)=
### Advanced Aside: In-Context Learning of an AR(1) Process
The material up to this point is the core {ref}`ch-intro` message; readers comfortable with the RNN $\to$ LSTM $\to$ Transformer summary may skip directly to the Chapter Summary. This subsection is an optional but illuminating detour: it shows why economists often find attention intuitive once they see it through a regression lens, and it is the analytical companion to notebook `09_Transformer_InContext_AR1`.

A remarkable emergent property of large Transformers is *in-context learning*: the ability to solve a task *at inference time*, given only examples in the prompt, with no weight updates. A model pretrained on a universe of sequences can be shown a fresh series it has never seen and produce sensible next-step forecasts. For an economist this is striking: the Transformer behaves as though it *runs a regression inside its forward pass*, with the prompt playing the role of the training sample and the final token playing the role of the test point. {cite:t}`vonoswald2023transformers` make this formal by showing that self-attention layers can implement gradient-based optimization internally, so a stack of such layers iteratively reduces an implicit loss.

Consider the simplest concrete setting, an autoregressive process of order 1:

$$
x_t = \varrho \, x_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2).
$$

If we provide a Transformer with a sequence $(x_1, \dots, x_t)$, it can predict $x_{t+1}$ by implicitly estimating $\varrho$ from the history. There are two closely related but distinct perspectives. First, under suitable linear-attention parameterizations, self-attention layers can implement gradient-descent-like updates on least-squares objectives {cite:p}`vonoswald2023transformers`; specialized to the AR(1) target above, this reads as descent on $\min_\varrho \sum_{i=2}^{t} (x_i - \varrho\,x_{i-1})^2$. Second, with ordinary softmax attention, *one* natural Q/K/V assignment behaves like a kernel smoother:

- **Query:** the current state $Q_t = x_t$ ("where am I now?"),

- **Key:** the lagged state $K_i = x_{i-1}$ ("past states that preceded each outcome"),

- **Value:** the realized successor $V_i = x_i$ ("what came next").

This is one particular parameterization that works because the AR(1) target is linear in the lagged state; with this choice the softmax-attention output becomes

$$
\hat{x}_{t+1} = \sum_{i=2}^{t} \alpha_i\, x_i \approx \hat{\varrho} \, x_t,
$$ (eq-incontext_kernel_smoother)

where the softmax weights $\alpha_i \propto \exp(x_t\,x_{i-1})$ concentrate mass on those past states $x_{i-1}$ that *look like* the current state $x_t$.

**Why this approximates $\varrho\,x_t$, in three short steps.** Equation {eq}`eq-incontext_kernel_smoother` is exactly a Nadaraya–Watson kernel regression of $x_i$ on $x_{i-1}$, evaluated at $x_{i-1} = x_t$, with kernel $K(x_*, x_{i-1}) = \exp(x_*\,x_{i-1})$. The intuition is then standard:

1.  *Population fact.* The AR(1) data-generating process implies $\E{x_i \mid x_{i-1} = x_*} = \varrho\, x_*$ exactly. So the population conditional mean we are trying to estimate at $x_* = x_t$ is just $\varrho\, x_t$.

2.  *Kernel smoother.* The softmax attention output $\sum_i \alpha_i x_i$ with $\alpha_i \propto K(x_t, x_{i-1})$ is a kernel-weighted average of the values $x_i$ at past time steps $i$, with the weights peaked where the lagged state $x_{i-1}$ is closest to $x_t$. Provided enough past observations land near $x_t$ (so the kernel concentrates around $x_t$), this is the empirical Nadaraya–Watson estimator of $\E{x \mid x_{i-1} = x_t}$.

3.  *Conclusion.* Combining the two, $\sum_i \alpha_i x_i \approx \E{x_i \mid x_{i-1} = x_t} = \varrho\, x_t$. The shock variance $\sigma^2$ controls how much the realized $x_i$'s scatter around the conditional mean and therefore the variance of the kernel estimate, but it does not enter the Nadaraya–Watson *location* at first order.

Note that the unscaled inner product $x_t\, x_{i-1}$ used in the kernel is dimension-1, so the standard $1/\sqrt{d_k}$ scaling of multi-head attention plays no role here: even without it, the softmax concentrates around the past $x_{i-1}$'s closest to $x_t$ as soon as the prompt is long enough.

For the econometrician's mental library the closest classical objects are the Nadaraya–Watson and local-linear estimators; the novelty is that the kernel is not hand-chosen but jointly learned with the data representation, over a large corpus of related tasks. This is a form of *meta-learning*: the network learned "how to regress" during pretraining, and at inference it runs that regression on a brand-new series. Self-attention can therefore implement optimization-like or regression-like computations internally, not merely local pattern matching.

**Code examples.** The following Jupyter notebooks implement and extend the material in this chapter:

- `01_BasicML_intro`: linear regression, classification, and loss functions.

- `02_GradientDescent_and_SGD`: implementing and visualizing optimizers.

- `03_Double_Descent`: the modern generalization regime.

- `04_Gentle_DNN`: building a simple DNN from scratch.

- `05_Tensorboard`: monitoring training progress.

- `06_PyTorch_intro`: introduction to PyTorch fundamentals.

- `07_Genz_Approximation_and_Loss_Functions`: high-dimensional integration using Genz test functions and robust losses.

- `08_MLP_LSTM_Transformer_Edgeworth_Cycles`: a three-way comparison on the same Edgeworth-cycle task. An MLP that sees only $x_t$ collapses near the cycle mean (no memory); an LSTM with a 32-unit hidden state tracks the sawtooth almost exactly via its gated memory; a tiny encoder-only Transformer ($d_{\text{model}}\!=\!16$, two layers, four heads, $\sim\!4.7$k parameters) attends to the full window in parallel and beats the MLP by an order of magnitude, while remaining slightly behind the LSTM on this small, highly periodic, low-data signal – a deliberate illustration that architectural inductive bias matters as much as flexibility on small problems. Quantitatively, the test-set MAE ranking that the notebook prints in its summary cell is, in order, LSTM $<$ Transformer $\ll$ MLP, with the Transformer typically within a small multiple of the LSTM and the MLP an order of magnitude worse; readers should consult the notebook's printed table for the seed-specific numbers.

- `09_Transformer_InContext_AR1`: advanced / optional notebook. A tiny 2-layer Transformer learns *how to regress* across many AR(1) draws; at inference it recovers $\hat{\varrho}$ in-context without weight updates, reproducing the analytical prediction above.

```{prf:remark} Chapter Summary
:label: sec-ch1_summary

- Deep networks compose simple nonlinear coordinate transformations: sufficiently wide shallow networks already attain universal approximation {cite:p}`cybenko1989approximation,hornik1989multilayer`, but depth gives provably more efficient representations for compositional functions {cite:p}`telgarsky2016benefits,barron1993universal`.

- Training rests on three pillars: He/Xavier initialization, smooth non-saturating activations, and adaptive optimizers (SGD with momentum $\to$ RMSprop $\to$ Adam $\to$ AdamW); backpropagation makes the gradient cost essentially the same as the forward pass {cite:p}`rumelhart1986learning,baydin2018automatic`.

- Generalization in modern over-parameterized regimes is described by the double-descent curve and explained theoretically via the Neural Tangent Kernel and benign-overfitting results, not by classical bias–variance {cite:p}`jacot2018neural,belkin2019reconciling,nakkiran2020deep,bartlett2020benign`.

- Sequence architectures (RNN $\to$ LSTM $\to$ Transformer) are mentioned for completeness; the rest of the script uses feed-forward MLPs because DEQNs and PINNs operate on unstructured state vectors.
```


## Further Reading {.unnumbered}
- {cite:t}`goodfellow2016deep`, the standard graduate textbook covering everything in this chapter at greater depth.

- {cite:t}`schmidhuber2015deep`, a historical survey tracing the deep-learning lineage; useful for context on LSTMs, Highway Networks, and Fast Weight Programmers.

- {cite:t}`bishop2006`, pattern recognition from a Bayesian/statistical viewpoint; the natural complement for econometric readers.

- {cite:t}`kingma2015adam` {cite}`loshchilov2019decoupled`, the canonical references on Adam and AdamW; read together they explain modern optimizer tuning.

- {cite:t}`bottou2018optimization`, a comprehensive survey of stochastic optimization for large-scale learning, including convergence rates.

## Exercises {.unnumbered}
Worked solutions and guidance for these exercises appear in Appendix {ref}`app-solutions`.

**Workload labels.** Throughout the script, every exercise carries one of three workload tags inside its title. *[Core\]* marks short analytical or pencil-and-paper questions suitable for a weekly problem set. *[Computational\]* marks notebook-based exercises that involve running or modifying companion code; allow yourself a long evening or a weekend with verification gates and starter code in hand. *[Advanced/project\]* marks longer, research-style assignments that may require a multi-day investment, a proper compute budget, or a small term-project plan. The labels are advisory rather than prescriptive: students with prior exposure can promote a [Computational\] exercise to a quick warm-up, while those new to the material can treat several [Advanced/project\] entries as inspiration for term work.

```{exercise}
:label: ex-ch1-1

**[Core\] Backprop on a 2-layer net.** Take a single hidden layer with ReLU activation: $\hat y = w_2\,\sigma(w_1 x + b_1)$ and squared loss $\ell = (\hat y - y)^2$. Derive $\partial \ell / \partial w_1$ and $\partial \ell / \partial w_2$ by hand and compare with what `torch.autograd` returns on a worked numerical example ($x=2$, $y=1$, all weights $0.5$).
```

```{exercise}
:label: ex-ch1-2

**[Core\] MSE vs. MLE.** Show that the maximum-likelihood estimator of the slope of $y_i = \beta x_i + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$, coincides with the OLS estimator and with the minimizer of $\sum_i (y_i - \beta x_i)^2$. Then repeat with $\varepsilon_i$ Laplace-distributed and discuss why the squared loss is no longer optimal.
```

```{exercise}
:label: ex-ch1-3

**[Core\] Activation choice for a PINN.** Argue carefully why ReLU networks cannot be used to solve a second-order PDE in strong form. Construct an explicit example where the second derivative of the network output is identically zero except on a measure-zero set.
```

```{exercise}
:label: ex-ch1-4

**[Core\] Adam vs. AdamW.** In a regression with $L_2$ regularization, write down the per-parameter update rule for Adam (with $L_2$ added to the loss) and for AdamW (decoupled). Show that the two updates do *not* coincide and explain which one preserves the intuition that "weight decay shrinks weights uniformly."
```

```{exercise}
:label: ex-ch1-5

**[Core\] RNN forward pass by hand.** Take a vanilla recurrent unit with hidden dimension $H=2$, scalar input, and update $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$, scalar readout $\hat y_t = W_y h_t$. Use $W_h = 0.5\,I_2$, $W_x = (1,0)^\top$, $b = (0,0)^\top$, $W_y = (1,1)$, $h_0 = (0,0)^\top$, and the input sequence $x_{1:3} = (1, 0, 1)$. (i) Compute $h_t$ and $\hat y_t$ for $t = 1, 2, 3$. (ii) Derive a closed-form expression for $\partial \hat y_3 / \partial x_1$. (iii) Show that with $\|W_h\|_2 < 1$ this gradient decays exponentially in the sequence length, and explain how this connects to the vanishing-gradient problem ({ref}`sec-sequence_models`).
```

```{exercise}
:label: ex-ch1-6

**[Core\] Attention by hand.** Consider a single attention head on a 3-token scalar sequence $x_{1:3} = (0.0,\, 1.0,\, 0.5)$ with identity projections $W_Q = W_K = W_V = 1$ (so $q_i = k_i = v_i = x_i$) and scaling $\sqrt{d_k} = 1$. Compute the attention weights $a_{ij} = \mathrm{softmax}_j(q_i k_j)$ and the output $o_i = \sum_j a_{ij} v_j$ for $i = 1, 2, 3$. Verify that each $o_i$ is a convex combination of $\{v_1, v_2, v_3\}$ and identify which token attends most to which.
```

```{exercise}
:label: ex-ch1-7

**[Computational\] Notebook.** In `05_Tensorboard`, plot the train and validation loss for three optimizer choices (SGD, Adam, AdamW) on a small classification task. Comment on which converges fastest, which generalizes best, and where the curves diverge.
```

[^1]: The idea that a network can compute its own weights from its inputs has a long history: the *Fast Weight Programmers* of {cite:t}`schmidhuber1992learning` use one network to write the weights of another from context, which is widely viewed as a conceptual precursor of attention. {cite:t}`schlag2021linear` make the formal equivalence explicit, showing that linear-attention Transformers *are* Fast Weight Programmers.
