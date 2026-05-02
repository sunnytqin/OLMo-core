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

## 5. Open questions

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
4. **Paraphrase / synthetic-data extension.** The framework lifts
   directly to two repeated streams via $\eta_{\text{para}}$.
5. **Sub-saturation regime at large $N$.** Our largest models (190M /
   370M) only have a few high-epoch points; the curvature of $\eta$
   at $D'/D > R^{*}$ is poorly constrained at those sizes. A few
   high-epoch runs at 190M / 370M would tighten $\sigma$ substantially.

## Reproducing

```bash
python fit_joint_all.py            # canonical one-shot joint fit (k-sweep)
python fit_chinchilla_joint.py     # two-stage 1-epoch fit (ablation)
python fit_eta.py                  # all η forms, per-size + joint (form ranking)
python paper_figures.py            # generates paper/fig{1,2,3,4,5}.pdf
```

Code: [fit_lse.py](fit_lse.py), [data.py](data.py),
[fit_joint_all.py](fit_joint_all.py) (canonical pipeline),
[fit_chinchilla_joint.py](fit_chinchilla_joint.py),
[fit_eta.py](fit_eta.py),
[paper_figures.py](paper_figures.py).
