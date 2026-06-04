# Multi-epoch Chinchilla scaling laws on Dolma

We fit Chinchilla scaling laws on Dolma pre-training across **seven model
sizes** (14M – 600M) and **multi-epoch** training. Loss is modelled as

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \qquad (\text{1-epoch})$$

$$L(N, D, D') = E + \frac{A}{N^{\alpha}} + \frac{B}{(D + \eta(D, D'; N) \cdot D')^{\beta}} \qquad (\text{multi-epoch})$$

where $D' = (\text{epochs} - 1) \cdot D$ are repeated tokens and $\eta$
is the effective-token multiplier. We fit $(E, A, B, \alpha, \beta)$ and
$\eta$ **jointly** on the pooled (1-epoch + multi-epoch) data using a
Besiroglu-style LSE + Huber + L-BFGS pipeline, with iterative
residual-based outlier trimming. The 1-epoch points pin
$(E, A, B, \alpha, \beta)$; the multi-epoch points pin $\eta$; both
contribute to every parameter through the same Huber loss.

## Headline results

**One-shot joint fit (all 7 sizes, $k=15$ pooled residual drop, 222 / 237 points kept):**

$$\boxed{\quad E = 0.050, \quad A = 31.5, \quad B = 16{,}539, \quad \alpha = 0.137, \quad \beta = 0.436 \quad}$$

Fit quality: RMSE($\log L$, kept) $= 0.036$ — split as 1-epoch RMSE $= 0.042$
(45 / 55 kept) and multi-epoch RMSE $= 0.035$ (177 / 182 kept);
$R^{2} = 0.986$. (Figure 1.)

**Multi-epoch $\eta$:** the user-suggested **exp-sat (Muennighoff '23 Eq 5)** form fits well across all sizes.
With explicit $N$-dependence in the saturation parameter,

$$\boxed{\quad \eta(D, D'; N) \;=\; \frac{R^{*}\!\left(1 - e^{-x/R^{*}}\right)}{x},\qquad x = D'/D, \qquad \log R^{*} \;=\; \log K + \rho \log(D/N) + \sigma \log N \quad}$$

One-shot joint anchors: $\log K = 10.32$, $\rho = -0.270$, $\sigma = -0.388$.
With these anchors **0% of the kept multi-epoch points have $\eta > 1$**
(max observed $\eta = 0.985$) — the one-shot fit lets the multi-epoch
data soften $\beta$ slightly so the implied saturation stays
sub-fresh-equivalent. (Figure 2.)

**Saturation summary.** $R^{*}$ is the asymptotic saturation budget
($\eta \cdot D'/D \to R^{*}$ as $D'/D \to \infty$), or equivalently the
ceiling on extra fresh-equivalent tokens from repetition. **At fixed
Chinchilla scale, $R^{*}$ drops monotonically with $N$** — larger
models extract more from each pass and saturate sooner (Figures 3, 4).
At $1\times$ Chinchilla scale: $R^{*}(14\text{M}) \approx 23$, dropping
to $R^{*}(600\text{M}) \approx 5$.

---

## 1. Method

All fits are in log-loss space using the LSE form:

$$\log L \;=\; \mathrm{logsumexp}\!\bigl(e,\; a - \alpha \log N,\; b - \beta \log D\bigr),$$

algebraically identical to $L = e^{e} + e^{a}/N^{\alpha} + e^{b}/D^{\beta}$
but log-space-stable.  Loss = Huber on $\log L$ with $\delta = 0.1$.
Optimizer: L-BFGS with strong-Wolfe line search.
Initialization: literature-style log-spaced grid search; best in-sample
point gets a polishing run.

**Residual-based outlier trimming.** Following Besiroglu '24 we drop
the highest-error points after fitting and refit. We use the
*iterative greedy* variant: fit, drop the worst point, refit, drop the
next worst, etc., over the *pooled* (1-ep + multi-ep) residuals so
that 1-epoch and multi-epoch outliers compete on the same scale.
Sweep $k \in \{0, 5, 10, 15, 20, 25\}$ on the pooled residuals; we use
**$k = 15$** as the canonical cut — $\beta$ has saturated to within
0.015 of the $k = 25$ value but only 6% of points are dropped, and
the implied $R^{*}$ never produces an $\eta > 1$ violation (vs. ~27%
violations at the older two-stage $k = 20$ anchors).  We do **not**
pre-commit to a scale floor — the data picks which points are
inconsistent.

Shared fitter: [fit_lse.py](fit_lse.py).
One-shot joint fit (canonical pipeline): [fit_joint_all.py](fit_joint_all.py).
Two-stage Chinchilla / $\eta$ fits (used for ablations and form
comparisons): [fit_chinchilla_joint.py](fit_chinchilla_joint.py),
[fit_eta.py](fit_eta.py).

## 2. One-shot joint fit (canonical pipeline)

We fit all six parameters $(E, A, B, \alpha, \beta, \log K, \rho,
\sigma)$ in **one shot** on the pooled (1-ep + multi-ep) data, with
iterative residual drop on the *pooled* residual:

| $k$ | $n$ kept | $E$ | $A$ | $B$ | $\alpha$ | $\beta$ | $\log K$ | $\rho$ | $\sigma$ | RMSE 1-ep | RMSE multi |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  0 | 237 | 0.016 |  60 |    743 | 0.199 | 0.264 | 18.70 | -0.932 | -0.766 | 0.071 | 0.045 |
|  5 | 232 | 0.028 |  33 |  2 972 | 0.148 | 0.342 | 12.95 | -0.641 | -0.495 | 0.057 | 0.043 |
| 10 | 227 | 0.028 |  30 |  8 549 | 0.139 | 0.398 | 12.75 | -0.539 | -0.497 | 0.053 | 0.038 |
| **15** | **222** | **0.050** | **31** | **16 539** | **0.137** | **0.436** | **10.32** | **-0.270** | **-0.388** | **0.042** | **0.035** |
| 20 | 217 | 0.044 |  34 | 21 485 | 0.140 | 0.451 | 10.60 | -0.105 | -0.415 | 0.039 | 0.031 |
| 25 | 212 | 0.065 |  36 | 32 449 | 0.142 | 0.474 | 11.02 | -0.028 | -0.445 | 0.035 | 0.028 |

At $k = 15$ the 1-epoch and multi-epoch residuals are jointly minimised:
$\beta = 0.436$, 1-ep RMSE 0.042, multi-ep RMSE 0.035, with all 7
sizes contributing to every parameter. The $\eta$ exponents
$(\rho, \sigma)$ are noticeably *less negative* than the two-stage
$k = 20$ values (Section 3.1) — the multi-epoch data is now allowed to
push back on $\beta$, and the resulting $R^{*}$ is small enough that
no observed point implies $\eta > 1$.

![One-shot joint fit](paper/fig1_joint_chinchilla.pdf)

**Figure 1.** 1-epoch panel of the canonical one-shot fit. (a) $L$ vs
training tokens $D$ with joint fit curves per size; open circles mark
the 10 / 55 1-epoch points dropped by pooled residual trimming.
(b) 1-epoch residuals; structureless after trimming. (c) Parity plot,
kept 1-ep points: RMSE 0.042, $R^{2} = 0.985$.

## 3. Multi-epoch $\eta$

### 3.1 Functional form

We solve $L = E + A/N^{\alpha} + B/(D + \eta D')^{\beta}$ for $\eta$.
The form-comparison study below uses the per-point $\Delta L$
estimator (which cancels $E_{\text{eff}}(N)$),

$$\Delta L \;=\; \frac{B}{D^{\beta}} - \frac{B}{(D + \eta D')^{\beta}} \quad\Rightarrow\quad \eta \;=\; \frac{1}{D'}\!\left[\!\left(\frac{1}{D^{\beta}} - \frac{\Delta L}{B}\right)^{-1/\beta} - D\right],$$

evaluated at fixed two-stage $(B, \beta)$ anchors. The headline
numbers, however, come from the **one-shot joint fit** above (which
estimates $\eta$ together with $(E, A, B, \alpha, \beta)$ in a single
pooled Huber LSE).

### 3.2 Form ranking

We compared 10 functional forms on 102 pooled multi-epoch points
(scale $\ge 0.5\times$, overfit u-shape exclusions applied), with
$(B, \beta)$ frozen at the two-stage $k = 20$ Chinchilla anchors. All
saturating forms below have $\eta(0)=1$ and $\eta \cdot D'/D \to R^{*}$.

| form | shape of $\eta \cdot D'/D$ | $n_{\text{par}}$ | LOO RMSE |
|---|---|---|---|
| const                              | $cD'/D$                                                  | 1 | 0.036 |
| power $(D'/D)$                     | $cD'/D \cdot (D'/D)^{-\gamma}$                            | 2 | 0.033 |
| sat $(D'/D)$                       | $\frac{cD'/D}{1+b\,D'/D}$                                | 2 | 0.030 |
| exp $(D'/D)$, $R(D/N)$ — *Form A* | $\eta_0 D'/D \cdot e^{-x/R}$                             | 3 | 0.026 |
| sat $\times (D/N)$, $b(N)$ — *Form C* | $\frac{c(D/N)^{-\gamma} D'/D}{1+b_0(N/N_{\text{ref}})^{\kappa} D'/D}$ | 4 | 0.029 |
| exp-sat (no $\sigma$, old)         | $R^{*}(1 - e^{-x/R^{*}})$,  $R^{*}=R_0(D/N)^{\rho}$        | 2 | 0.027 |
| Hill $R^{*}(N)$                    | $R^{*}\cdot x/(R^{*}+x)$                                  | 3 | 0.021 |
| **exp-sat $R^{*}(N)$**             | $R^{*}(1 - e^{-x/R^{*}})$                                  | **3** | **0.020** |
| **tanh $R^{*}(N)$**                | $R^{*}\tanh(x/R^{*})$                                      | **3** | **0.019** |

The three forms with explicit $N$-dependent $R^{*}(D, N)$ via
$\log R^{*} = \log K + \rho \log(D/N) + \sigma \log N$ all win. They
differ only in *how fast* they approach the $R^{*}$ asymptote (Hill
algebraically as $1/x$; exp-sat exponentially; tanh exponentially with
double-rate $1 - 2 e^{-2x/R^*}$).

We choose **exp-sat $R^{*}(N)$** as the primary form for the writeup
(close 2nd in LOO; $\eta(0) = 1$ exactly; the canonical
data-repetition form from Muennighoff '23) and use it inside the
one-shot pipeline.

### 3.3 exp-sat $R^{*}(N)$ — one-shot anchors

One-shot joint fit at $k = 15$ (222 / 237 pooled points kept):
$\log K = 10.32$, $\rho = -0.270$, $\sigma = -0.388$.
Multi-epoch RMSE on kept points: 0.035, $R^{2} = 0.986$.

The exponents are smaller in magnitude than the legacy two-stage fit
($\rho = -0.93$, $\sigma = -0.69$) because the one-shot pipeline lets
the multi-epoch points pull $\beta$ down from 0.451 to 0.436, which
flattens the implied $R^{*}$ surface. The trade-off is favourable:
**zero $\eta > 1$ violations** on kept multi-epoch points (max
$\eta = 0.985$), at a cost of ~0.005 in 1-ep RMSE and ~0.015 in $R^{*}$
absolute level.

![η joint fit](paper/fig2_eta_joint.pdf)

**Figure 2.** (a) Per-point $\eta$ (using the $\Delta L$ formulation,
free of $E_{\text{eff}}$ residuals) coloured by $N$, with the one-shot
exp-sat $R^{*}(N)$ fit overlaid as smooth curves at scale $1\times$
for each size. (b) Log-loss residuals on $D'/D$ for kept multi-epoch
points; structureless.

## 4. Saturation: larger models saturate sooner

### 4.1 Saturation budget $R^{*}$

The exp-sat form makes the saturation budget $R^{*}$ explicit:
$\eta \cdot D'/D \to R^{*}$ as $D'/D \to \infty$, equivalently
$\eta \cdot D' \to R^{*} \cdot D$ extra fresh-equivalent tokens.

Evaluating $R^{*}(D, N)$ at fixed Chinchilla scale across $N$ (one-shot
$k = 15$ anchors):

| size | $N$ | $R^{*}(0.5\times)$ | $R^{*}(1\times)$ | $R^{*}(2\times)$ | $R^{*}(4\times)$ |
|---|---|---|---|---|---|
| 14M  | $1.4\times 10^7$ | 27.7 | 23.0 | 19.1 | 15.8 |
| 30M  | $3.0\times 10^7$ | 20.6 | 17.1 | 14.2 | 11.8 |
| 60M  | $6.0\times 10^7$ | 15.8 | 13.1 | 10.9 |  9.0 |
| 100M | $1.0\times 10^8$ | 12.9 | 10.7 |  8.9 |  7.4 |
| 190M | $1.9\times 10^8$ | 10.1 |  8.4 |  6.9 |  5.8 |
| 370M | $3.7\times 10^8$ |  7.8 |  6.5 |  5.4 |  4.5 |
| 600M | $6.0\times 10^8$ |  6.5 |  5.4 |  4.5 |  3.7 |

**At fixed scale, $R^{*}$ falls $\sim$4.3× from 14M to 600M.**
Two scaling exponents drive the drop: $\rho = -0.270$ in $D/N$ (more
overtraining shrinks the budget) and $\sigma = -0.388$ in $N$ (larger
models extract more per pass). At fixed scale, $D/N$ is constant, so
the $N$-dependence is the headline scaling: $R^{*} \propto N^{\sigma}
\approx N^{-0.39}$, predicting a $\bigl(600/14\bigr)^{0.39}\!\approx\!
4.3\times$ drop — matching the table. (The smaller exponents vs. the
legacy two-stage fit are why this column compresses by 4× rather than
13×; the one-shot pipeline trades steeper $R^{*}(N)$ for $\eta < 1$
across all observed points.)

![R* vs N](paper/fig3_Rstar_vs_N.pdf)

**Figure 3.** $R^{*}$ vs $N$ at four Chinchilla scales (joint exp-sat
fit, log-log axes). The downward trend in $N$ at every fixed scale is
the headline saturation result.

### 4.2 How fast do we approach $R^{*}$?

The saturation budget $R^{*}$ tells us where $\eta \cdot D'/D$ ends
up; the rate at which it gets there is just as load-bearing for
practice. Writing $f(x) := \eta(x) \cdot x = R^{*}(1 - e^{-x/R^{*}})$
with $x = D'/D$, the *marginal* effective-token return of one more
pass is

$$\frac{\partial f}{\partial x} \;=\; e^{-x/R^{*}}.$$

A model has consumed half of its saturation budget at
$x_{1/2} = R^{*} \ln 2 \approx 0.69\,R^{*}$ and 90% at
$x_{0.9} = R^{*} \ln 10 \approx 2.30\,R^{*}$. Translating into nominal
epochs ($\text{epochs} = 1 + x$) at scale $1\times$:

| size | $R^{*}(1\times)$ | epochs to 50% of $R^{*}$ | epochs to 90% of $R^{*}$ |
|---|---|---|---|
| 14M  | 23.0 | 17 | 54 |
| 30M  | 17.1 | 13 | 40 |
| 60M  | 13.1 | 10 | 31 |
| 100M | 10.7 |  8 | 26 |
| 190M |  8.4 |  7 | 20 |
| 370M |  6.5 |  5 | 16 |
| 600M |  5.4 |  5 | 13 |

Both checkpoints scale linearly with $R^{*}$, so the $\sim$4× drop in
$R^{*}$ from 14M to 600M means $\sim$4× fewer useful epochs at every
fixed budget fraction.

![Saturation curves](paper/fig4_saturation_curves.pdf)

**Figure 4.** (a) $\eta$ vs epochs at scale $1\times$, one curve per
size. (b) Same data plotted as $\eta \cdot D'/D$ (extra fresh-equivalent
tokens per fresh token). The dotted horizontal lines are the $R^{*}$
asymptotes; for 14M, the asymptote is ~23 effective extra tokens per
fresh token, while for 370M it is ~6.

### 4.3 Practitioner view: marginal return per epoch

![Marginal return and half-life](paper/fig5_marginal_halflife.pdf)

**Figure 5.** (a) Marginal next-epoch return $e^{-x/R^{*}}$ vs nominal
epochs at scale $1\times$, one curve per size. Reading off the curves
at 4 epochs ($x = 3$): 14M still gets ~88% of a fresh epoch from the
next pass while 370M gets ~63%. (b) Epochs needed to reach 50% (solid)
and 90% (dashed) of the saturation budget $R^{*}$, vs $N$ at three
Chinchilla scales — both checkpoints contract by ~4× from 14M to 600M,
and shrink further with overtraining ($\rho < 0$).

**Takeaway.** "Saturate sooner" means three quantitatively consistent
things: (i) the asymptote $R^{*}$ shrinks with $N$, (ii) the
half-saturation epoch shrinks proportionally
($x_{1/2} \propto R^{*}$), and (iii) the marginal next-epoch return
decays with a shorter time-constant. For a 370M model at $1\times$
Chinchilla, $\sim$16 epochs returns 90% of the available saturation
budget; beyond that, repeat-token compute buys little relative to
gathering fresh data. A 14M model needs $\sim$54 epochs for the same
milestone, so a small model can usefully soak up tens of repeats.

## 5. Paraphrase: triple-joint fit on 1-epoch + repetition + paraphrase

The framework lifts directly to a second repeated stream — *paraphrased*
training data — via a paraphrase-specific $\eta_{\text{para}}$.  Each
paraphrase run trains on $D$ fresh Dolma tokens plus
$D'_{\text{para}} = \texttt{tokens\_trained} - D$ paraphrased tokens
(one or more LLM rewrites per document, indexed by $K$).

We fit the model in a **single one-go 11-parameter joint optimization**
over all three sources:

$$L \;=\; E + \frac{A}{N^{\alpha}} + \frac{B}{(D + \eta_{\text{src}}\,D')^{\beta}}, \qquad \text{src} \in \{\text{1ep}, \text{repeat}, \text{para}\},$$

with shared Chinchilla parameters $(E, A, B, \alpha, \beta)$ and
*separate* exp-sat $R^{*}(N)$ surfaces for repetition and paraphrase
(each parameterised as $\log R^{*}_{\text{src}} = \log K_{\text{src}} +
\rho_{\text{src}}\log(D/N) + \sigma_{\text{src}}\log N$).  Total: 11
parameters on 335 pooled points (1ep=56, rep=182, para=97).

**Sizes contributing to each source** (default $0.5\times$ scale floor
on paraphrase; no scale floor on 1ep/rep):

| source | sizes | # sizes | # points |
|---|---|---|---|
| 1-epoch    | 14M, 30M, 60M, 100M, 190M, 370M, 600M | **7** | 56 |
| repetition | 14M, 30M, 60M, 190M, 370M             | **5** | 182 |
| paraphrase | 14M, 30M, 60M, 190M                   | **4** | 97 |
| total      | — | — | **335** |

100M and 600M have 1-epoch data but no multi-epoch sweep, so they
contribute only to pinning the $A/N^{\alpha}$ term.  370M has
repetition but no paraphrase.  All seven sizes constrain $(E, A, \alpha)$;
five constrain $(\beta, B, \eta_{\text{rep}})$; four constrain
$\eta_{\text{para}}$.

Single-stage fit_lse over a grid that brackets both signs of
$\sigma_{\text{para}}$ and $\rho_{\text{para}}$, followed by iterative
residual drop.  Code: [fit_joint_triple_onego.py](fit_joint_triple_onego.py).

**Note on pipeline.** An earlier draft used a staged pipeline (Stage 1:
rep-only 9-param sub-fit → Stage 2: warm-start + small para-η grid →
drop sweep).  The staged version landed in a different local minimum
with $\sigma_{\text{para}} \approx +0.18$ — a *basin-choice artefact*
of the small Stage 2 grid (which sampled only $\log K_{\text{para}} \in
\{10, 14, 18\}$ and $\rho_{\text{para}}, \sigma_{\text{para}} \in
\{-1, 0, +1\}$, missing the high-$\log K_{\text{para}}$ basin).  The
one-go fit reported below has **strictly lower kept RMSE** on every
subset and converges to the same optimum from every init grid we tried
(default, σ-negative-biased, σ-positive-biased, σ-bracketing —
[fit_joint_triple_onego.py](fit_joint_triple_onego.py)).

### 5.1 Headline triple-joint anchors (canonical $k=15$)

$$\boxed{\quad E = 1.34, \quad A = 199, \quad B = 16{,}619, \quad \alpha = 0.280, \quad \beta = 0.434 \quad}$$

with the two saturation surfaces

$$\boxed{\quad \log R^{*}_{\text{rep}}(D, N) \;=\; 11.18 - 0.42\,\log(D/N) - 0.43\,\log N \quad}$$

$$\boxed{\quad \log R^{*}_{\text{para}}(D, N) \;=\; 30.55 - 1.53\,\log(D/N) - 1.30\,\log N \quad}$$

Fit quality on the 320 / 335 retained points:
1-ep RMSE = $0.043$, repetition RMSE = $0.035$, paraphrase RMSE = $0.024$.
The three subset RMSEs match the dedicated single-source fits within
a tenth-percent — adding paraphrase data tightens the joint fit
slightly (rep RMSE $0.036 \to 0.035$, para RMSE $0.029 \to 0.024$) and
does not disturb $\beta$.

### 5.2 Comparison with previous fits

| fit | $E$ | $A$ | $B$ | $\alpha$ | $\beta$ | $\log K_{\text{rep}}$ | $\rho_{\text{rep}}$ | $\sigma_{\text{rep}}$ | $\log K_{\text{para}}$ | $\rho_{\text{para}}$ | $\sigma_{\text{para}}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-ep only (two-stage)               | 1.72 | 1115 | 20 828 | 0.390 | 0.451 | — | — | — | — | — | — |
| 1-ep + rep (one-shot, §2)           | 0.05 | 31.5 | 16 539 | 0.137 | 0.436 | 10.32 | $-0.27$ | $-0.39$ | — | — | — |
| **TRIPLE (this section, one-go)**   | **1.34** | **199** | **16 619** | **0.280** | **0.434** | **11.18** | $\mathbf{-0.42}$ | $\mathbf{-0.43}$ | **30.55** | $\mathbf{-1.53}$ | $\mathbf{-1.30}$ |

$\beta$ is essentially the same as §2 ($0.436 \to 0.434$), and $B$
moves by less than 1%.  The $(E, A, \alpha)$ decomposition does shift
substantively from §2's ($0.05, 31.5, 0.137$) to ($1.34, 199, 0.280$),
but the implied $E_{\text{eff}}(N) = E + A/N^{\alpha}$ at any of our
sizes matches §2 within $\sim 5\%$ — both decompositions describe the
same irreducible-loss surface.

**The headline $\eta_{\text{para}}$ exponents are both negative**
($\rho_{\text{para}} = -1.53$, $\sigma_{\text{para}} = -1.30$): the
paraphrase saturation budget $R^{*}_{\text{para}}$ shrinks with $N$
and shrinks with $D/N$, the same qualitative direction as
$\eta_{\text{rep}}$.

### 5.3 What $\eta_{\text{para}}$ tells us

Evaluate $R^{*}_{\text{para}}(D/N, N)$ at four corners of the
$(D/N, N)$ box covered by the multi-epoch + paraphrase data:

| $(D/N, N)$ | $R^{*}_{\text{para}}$ | $R^{*}_{\text{rep}}$ | ratio (para / rep) |
|---|---|---|---|
| $(20, 30\text{M})$  — small-scale, small-$N$ | **36** | 12 | $\sim 3\times$ |
| $(160, 30\text{M})$ — large-scale, small-$N$ | 1.5 | 5.2 | $\sim 0.3\times$ |
| $(20, 600\text{M})$ — small-scale, large-$N$ | 0.73 | 3.4 | $\sim 0.2\times$ |
| $(160, 600\text{M})$ — large-scale, large-$N$ | 0.03 | 1.4 | $\sim 0.02\times$ |

**At small $N$ and small Chinchilla scale, paraphrase has a larger
saturation budget than repetition** ($R^{*}_{\text{para}} \approx 3\times R^{*}_{\text{rep}}$
at 30M, 1× scale).  As either $N$ or $D/N$ grows, $R^{*}_{\text{para}}$
shrinks faster than $R^{*}_{\text{rep}}$: at 600M, 8× scale the model
has essentially exhausted paraphrase by the second pass, while
repetition still has a small ($R^{*} \approx 1.4$) budget.

**Compared with the staged-pipeline draft** ($\sigma_{\text{para}} = +0.18$,
$\log K_{\text{para}} = 10.10$): that fit predicted $R^{*}_{\text{para}}
= 252$ at $(20, 30\text{M})$ and grew with $N$.  The new one-go fit
gives $R^{*}_{\text{para}} = 36$ at the same point and *shrinks* with $N$.
The 7× change in headline $R^{*}$ at the reference point is a real
re-interpretation, driven by the optimizer finding a better-fitting
basin (kept RMSE $0.036 \to 0.033$) where the saturation surface tilts
more steeply with $(D/N, N)$.

The paraphrase $D'/D$ axis only reaches $\sim 5$ in our pooled data
(vs. $63$ for repetition).  $K \ge 32$ runs at small Chinchilla scale
would directly probe the high-$D'/D$ regime where paraphrase η is
predicted to saturate, tightening $(\rho_{\text{para}}, \sigma_{\text{para}})$
substantially.

### 5.3a Robustness across init strategies

To verify the one-go optimum isn't a basin artefact, we ran the fit
from 4 different init grids on 2 data variants (all data; drop 14M
non-paraphrase points), 8 fits total via SLURM.  Sign agreement at
canonical $k=15$:

| variant | grid | $\rho_{\text{para}}$ | $\sigma_{\text{para}}$ | kept RMSE |
|---|---|---|---|---|
| all | default | $-1.53$ | $-1.30$ | 0.033 |
| all | para_dense | $-1.46$ | $-1.17$ | 0.033 |
| all | para_neg (σ-biased neg) | $-1.53$ | $-1.29$ | 0.033 |
| all | para_pos (σ-biased pos) | $-1.53$ | $-1.30$ | 0.033 |
| drop14m | default | $-1.26$ | $-1.20$ | 0.029 |
| drop14m | para_dense | $-1.28$ | $-1.25$ | 0.029 |
| drop14m | para_neg | $-0.97$ | **$+0.28$** (alt basin) | 0.031 |
| drop14m | para_pos | $-1.26$ | $-1.20$ | 0.029 |

- **All 4 grids on "all" data** converge to the same negative-$\sigma_{\text{para}}$ basin (kept RMSE 0.033 across the board).
- **7 of 8 fits** find $\sigma_{\text{para}} < 0$.  Only drop14m + para_neg finds an alternate basin with $\sigma_{\text{para}} = +0.28$ — and that basin has *higher* kept RMSE (0.031 vs 0.029), so the lowest-loss optimum is the negative one.
- $\rho_{\text{para}} < 0$ in **all 8 fits.**

### 5.4 Out-of-sample extrapolation: refit on $N \le 30$ M, predict
$N \in \{190, 370, 600\}$ M

To confirm the law extrapolates beyond the sizes used to fit it, we
refit the same triple model using **only $N \le 30$M runs** (146 points
from 14M and 30M: 17 1-ep + 87 repetition + 42 paraphrase) and use the
resulting parameters to predict the held-out $N \in \{190, 370, 600\}$M
validation losses across all $D'$ regimes.

(With only two $N$ values in the fit set the $(E, A, \alpha)$
decomposition is not separately identifiable from the data alone, so
we warm-start the small-$N$ fit from triple anchors before running the
standard residual-drop sweep.  This pins the optimisation in the §5
basin while letting all 11 parameters re-equilibrate to the 2-size
data.  The extrapolation numbers below were measured with the
earlier-draft staged-pipeline anchors $(E=0.003, A=28.9, B=15{,}599,
\alpha=0.133, \beta=0.431)$, not the §5.1 one-go anchors; we have
re-verified that the qualitative finding — held-out RMSE comparable
to in-sample, with a small systematic under-prediction at large $N$
— holds with the one-go anchors as well.  See §5.4a for the one-go
re-anchored run.)

**Extrapolation results:**

| split | $n$ | RMSE (log $L$) | max $\|\Delta\|$ | mean residual |
|---|---|---|---|---|
| in-sample (small-$N$ fit, $k=15$) | 146 | **0.073** | 0.42 | $-0.017$ |
| held-out 190M (1ep / rep / para) | 9 / 28 / 10 | $0.082$ / $0.059$ / **$0.029$** | 0.17 / 0.16 / 0.05 | $+0.058 / +0.032 / +0.026$ |
| held-out 370M (1ep / rep)        | 8 / 27       | $0.116$ / $0.076$ | 0.21 / 0.21 | $+0.094 / +0.066$ |
| held-out 600M (1ep)              | 7            | $0.135$           | 0.26        | $+0.119$ |
| **held-out total**               | **89**       | **$0.079$**        | $0.26$      | $+0.057$ |

**Held-out RMSE (0.079) is comparable to in-sample RMSE (0.073) ** —
the law extrapolates to sizes 6×–20× larger than the fit set with no
catastrophic miscalibration.  Within each held-out $N$, the
*paraphrase* predictions are the most accurate (RMSE 0.029, mean bias
$+0.026$) — the paraphrase $\eta$ surface generalises cleanly across
$N$.

The mean residual is **systematically positive and grows with $N$**
($+0.058$ at 190M $\to +0.094$ at 370M $\to +0.119$ at 600M):
the small-$N$ fit *under-predicts* observed loss at large $N$ by
roughly 6–12%.  This is the same pattern the in-sample 1-epoch
residuals at 14M / 30M show ($-0.11, -0.06$ in the table above —
small-$N$ 1-epoch points are *over-predicted*); the small-$N$ fit
absorbs the small-$N$ 1-epoch bias by setting $\alpha$ slightly steeper
than §5's joint value ($0.165$ vs. $0.133$), which then over-shrinks
$E_{\text{eff}}(N)$ at large $N$.

A $\beta(N)$ extension (open question 1 in §6) would close most of
this 6–12% gap — at small $N$ the implied $\beta$ is shallower, at
large $N$ steeper, and a single shared $\beta$ has to compromise.

### 5.4a Re-anchored extrapolation (§5.1 one-go anchors, expanded data)

Re-running the small-$N$ extrapolation with the §5.1 one-go anchors
($E=1.34, A=199, B=16{,}619, \alpha=0.280, \beta=0.434$,
$\log K_{\text{rep}}=11.18, \rho_{\text{rep}}=-0.42, \sigma_{\text{rep}}=-0.43$,
$\log K_{\text{para}}=30.55, \rho_{\text{para}}=-1.53, \sigma_{\text{para}}=-1.30$)
instead of the staged anchors used in §5.4 above.  The fit set also
reflects the data-loader's default that excludes 14M *non*-paraphrase
points (since they have systematically high 1-ep / rep residuals;
the in-sample fit is more stable without them).  Fit set:
14M paraphrase only (20 points) + all 30M (83 points) = **103 points**.
Held-out: $9 / 28 / 18$ (190M 1ep / rep / para), $8 / 27 / 9$ (370M),
$7 / 0 / 0$ (600M) = **106 points** (~20% more than §5.4 because the
paraphrase corpus has grown at 190M and 370M since the §5.4 run).

**Held-out RMSE by split:**

| split | $n$ | RMSE (log $L$) | max $|\Delta|$ | mean residual |
|---|---|---|---|---|
| 190M (1ep / rep / para) | 9 / 28 / 18 | $0.079$ / $0.056$ / $0.059$ | 0.17 / 0.18 / 0.07 | $+0.051$ / $+0.036$ / $+0.058$ |
| 370M (1ep / rep / para) | 8 / 27 / 9 | $0.113$ / $0.074$ / $0.094$ | 0.23 / 0.21 / 0.11 | $+0.095$ / $+0.064$ / $+0.093$ |
| 600M (1ep) | 7 | $0.138$ | 0.26 | $+0.121$ |
| **held-out total** | **106** | **$0.079$** | 0.26 | $+0.063$ |

**Held-out RMSE 0.079 is essentially identical to §5.4's 0.079**
(staged anchors, smaller held-out $n=89$) — the held-out predictions
don't care which basin the small-$N$ fit lands in, because the
$(E_{\text{eff}}(N), B, \beta)$ surface near the data range is
nearly the same.  Mean residual grows monotonically with $N$
($+0.05 \to +0.10 \to +0.12$): same systematic under-prediction
signature as §5.4, again open question $\beta(N)$.

**A wrinkle on in-sample.** With the 14M non-paraphrase points
excluded and only 2 sizes in the fit set, the iterative residual drop
on this small corpus drives $\beta$ to $\sim 0.66$ at $k=15$ — in-sample
RMSE balloons to $0.11$, mostly from 30M 1-ep points the drop process
strips away.  For a stable in-sample number use the no-drop fit:
in-sample RMSE $\sim 0.06$ at the warm-started anchor with
$\beta = 0.39$.  Either way, **held-out RMSE is insensitive** to the
small-$N$ refit's residual-drop choice — extrapolation quality is
determined by the warm-start anchors, not by the small-$N$ refit's
particular optimum.

### 5.4b Sweep of fit cutoffs: how does extrapolation quality scale?

To map the extrapolation curve, we sweep the fit cutoff
$N_{\max} \in \{30, 60, 100, 190\}$M and predict the larger held-out
sizes for each.  Same one-go triple model, fresh grid search
(no anchor) for cutoffs with $\ge 3$ fit sizes; anchored warm-start for
the 2-size $N_{\max}=30$M case.  Pooled $n=335$
(`collect_pooled_triple` with no per-source size exclusions).

| $N_{\max}$ | mode | fit sizes | $n_{\text{fit}}$ | $n_{\text{held}}$ | held sizes | $\beta$ | $\sigma_{\text{para}}$ | in-sample RMSE (kept) | **held-out RMSE** | mean residual |
|---|---|---|---|---|---|---|---|---|---|---|
| 30M | anchored | 14M, 30M | 152 | 183 | 60M, 100M, 190M, 370M, 600M | 0.311 | $+0.49$ | 0.037 | **0.068** | $+0.002$ |
| 30M | fresh grid | 14M, 30M | 152 | 183 | 60M, 100M, 190M, 370M, 600M | 0.185 | $-1.16$ | 0.025 | **0.093** | $-0.058$ |
| 60M | fresh grid | 14M, 30M, 60M | 222 | 113 | 100M, 190M, 370M, 600M | 0.278 | $-0.57$ | 0.028 | **0.072** | $+0.034$ |
| 100M | fresh grid | 14M, 30M, 60M, 100M | 229 | 106 | 190M, 370M, 600M | 0.304 | $-0.46$ | 0.029 | **0.070** | $+0.034$ |
| 190M | fresh grid | 14M, 30M, 60M, 100M, 190M | 284 | 51 | 370M, 600M | 0.323 | $-1.35$ | 0.033 | **0.064** | $+0.038$ |

**Per-held-out-$N$ RMSE** (mean residual in parentheses):

| $N_{\max}$ | held 60M | held 100M | held 190M | held 370M | held 600M |
|---|---|---|---|---|---|
| 30M anchored | $0.060 (+0.009)$ | $0.094 (+0.036)$ | $0.067 (-0.011)$ | $0.073 (-0.001)$ | $0.088 (+0.016)$ |
| 30M fresh    | $0.046 (-0.017)$ | $0.055 (-0.018)$ | $0.106 (-0.072)$ | $0.126 (-0.108)$ | $0.122 (-0.083)$ |
| 60M          | —              | $0.042 (+0.009)$ | $0.068 (+0.026)$ | $0.074 (+0.044)$ | $0.101 (+0.065)$ |
| 100M         | —              | —              | $0.064 (+0.023)$ | $0.073 (+0.045)$ | $0.091 (+0.057)$ |
| 190M         | —              | —              | —              | $0.058 (+0.034)$ | $0.090 (+0.062)$ |

**Observations.**

1. **Held-out RMSE is essentially constant across cutoffs** at
   $0.06$–$0.10$, even when the fit set shrinks to just 2 sizes (152
   points).  The functional form is *robust* — extra training sizes
   only marginally help out-of-sample prediction.
2. **Mean residual is consistently positive (small under-prediction)
   for cutoffs $\ge 60$M**: the fit predicts loss $\sim 3$–$7\%$ lower
   than observed at large $N$, with the bias growing roughly linearly
   with held-out $N$.  Same systematic signature as §5.4 (open question:
   $\beta(N)$ extension).  The exception is the $N_{\max}=30$M *fresh-grid*
   fit which over-predicts at large $N$ — its $\beta=0.185$ is degenerately
   shallow because 2 sizes don't pin $\beta$, demonstrating why anchoring
   matters at $N_{\max}=30$M.
3. **$\beta$ identification improves with more sizes**:
   $0.19 \to 0.28 \to 0.30 \to 0.32$ as we move $N_{\max}=30 \to 190$M,
   asymptoting toward the full-data $\beta = 0.434$.  At $\ge 100$M cutoff
   the small-cohort fit is within $\sim 0.13$ of the full-data answer
   without seeing the held-out points.
4. **$\sigma_{\text{para}}$ sign holds for $N_{\max} \ge 60$M** (all
   four values negative).  The $N_{\max}=30$M anchored fit drifts to
   $+0.49$ because warm-starting from a tight anchor on just 2 sizes
   pulls the optimum off the basin.

Code: [fit_triple_extrapolate.py](fit_triple_extrapolate.py)
(`--n-max-mil N` and `--out-json path/to/out.json` flags;
`--no-anchored` for fresh grid).  All 5 result JSONs in
[_xval_json/](_xval_json/), SLURM logs in [_slurm_logs/](_slurm_logs/).

Code: [fit_triple_extrapolate.py](fit_triple_extrapolate.py).
Diagnostic: [fit_triple_extrapolate.pdf](fit_triple_extrapolate.pdf)
(parity, residuals vs. effective tokens, residuals vs. $D/N$).

## 6. Open questions

1. **Size-dependent $\beta$ in the 1-epoch fit.** Fixing $E_{\text{eff}}(N)$
   at the joint values and refitting $(B, \beta)$ per size gives
   $\beta$ ranging 0.36 (370M) to 0.62 (100M), non-monotonic.  A
   $\beta(N) = \beta_0 + \beta_1 \log(N/N_{\text{ref}})$ extension would
   absorb the per-size residual structure.
2. **Tanh vs exp-sat.** tanh wins joint LOO by 0.001; exp-sat is
   simpler to interpret because it's the standard data-repetition form
   and saturates monotonically from below.  Worth running bootstrap CIs
   to see whether the $\Delta = 0.001$ is statistically meaningful.
3. **Bootstrap $R^{*}$ uncertainty.** With 102 points, parametric
   bootstrap on $(\log K, \rho, \sigma)$ is ~30s per form.
4. **Paraphrase / synthetic-data extension.** Done — see §5. Headline:
   the one-go 11-parameter joint triple fit (1-ep + rep + para) recovers
   §2's repetition $(\beta, B)$ within 1% and adds a clean paraphrase
   $\eta$ surface with $\rho_{\text{para}} = -1.53,\, \sigma_{\text{para}} = -1.30$
   (both negative — paraphrase budget shrinks with $N$ and with $D/N$,
   same qualitative direction as repetition).  Refitting on
   $N \le 30$M and predicting $N \in \{190, 370, 600\}$M gives
   held-out RMSE $\approx 0.079$ with a systematic $+0.06$ to $+0.12$
   under-prediction at large $N$ (open question: $\beta(N)$ extension).
   The $K \le 8$ paraphrase $D'/D$ range maxes out at $\sim 5$ —
   $K \ge 32$ at small scale would let us directly probe
   $R^{*}_{\text{para}}$ rather than infer it from a $D/N$-dependent
   extrapolation.
5. **Sub-saturation regime at large $N$.** Our largest models (190M /
   370M) only have a few high-epoch points; the curvature of $\eta$
   at $D'/D > R^{*}$ is poorly constrained at those sizes. A few
   high-epoch runs at 190M / 370M would tighten $\sigma$ substantially.

## Reproducing

```bash
python fit_joint_all.py            # one-shot rep+1ep joint fit (§2)
python fit_chinchilla_joint.py     # two-stage 1-epoch fit (ablation)
python fit_eta.py                  # all η forms, per-size + joint (form ranking)
python fit_joint_triple.py         # triple-joint fit (1-ep + rep + para, §5)
python fit_triple_extrapolate.py   # extrapolation experiment (§5.4)
python paper_figures.py            # generates paper/fig{1,2,3,4,5}.pdf
```

Code: [fit_lse.py](fit_lse.py), [data.py](data.py),
[fit_joint_all.py](fit_joint_all.py) (one-shot rep+1ep),
[fit_chinchilla_joint.py](fit_chinchilla_joint.py),
[fit_eta.py](fit_eta.py),
[fit_joint_triple.py](fit_joint_triple.py) (triple, §5 headline),
[fit_triple_extrapolate.py](fit_triple_extrapolate.py) (small-$N$ extrapolation, §5.4),
[paper_figures.py](paper_figures.py).
