# Chinchilla Scaling-Law Fits on Dolma

Multi-epoch Chinchilla scaling-law fit on Dolma across **seven model
sizes** (14M → 600M). The classic Chinchilla form is

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

and multi-epoch training is modelled as

$$L = E_{\text{eff}}(N) + \frac{B}{(D + \eta \cdot D')^{\beta}}, \qquad D' = (\text{epochs}-1) \cdot D,$$

where $E_{\text{eff}}(N) = E + A/N^{\alpha}$ and $\eta$ is the
effective-token multiplier for repeated data.

## Current best fit (all sizes, $\delta=0.1$ Huber on $\log L$)

**Joint Chinchilla anchors** (1-epoch, all 7 sizes; residual-based
top-$k$ outlier drop, Besiroglu-style — see §1.1 for the canonical-$k$
selection):

| param | value |
|---|---|
| $E$ | 1.72 |
| $A$ | 1115 |
| $B$ | 20 828 |
| $\alpha$ | 0.390 |
| $\beta$ | 0.451 |

Fit on the kept 35 points (out of 55 total): RMSE$(\log L) = 0.019$,
$R^2 = 0.997$. The legacy hard-cut "scale $\ge 0.5\times$" fit (34
points) gave $\beta=0.459$ at RMSE 0.032; the residual-drop version
arrives at the same $\beta$ at lower RMSE without committing to a
fixed scale boundary.

We carry two strong $\eta$ forms forward.  Both are reported throughout
the writeup so their behaviours can be compared directly.

**Form A — exp$(D'/D)$ with TTP-dep $R$** (the user-proposed exponential
form, current best on the joint pool):

$$\boxed{\quad \eta_A(D, D'; N) \;=\; \eta_0 \cdot \exp\!\left(-\frac{D'/D}{R_0 \cdot (D/N)^{\rho}}\right),\qquad \eta_0 = 0.946,\;R_0 = 471,\;\rho = -0.755 \quad}$$

Joint LOO RMSE $= 0.026$, $R^2 = 0.986$. 3 free parameters.
$\eta_0 \le 1$ implies $\eta \le 1$ everywhere by construction.

**Form B — Muennighoff '23 Eq 5** (close 2nd, $\eta(0)=1$ exactly):

$$\boxed{\eta_A(D, D'; N) = \eta_0 \cdot \exp\left(-\frac{D'/D}{R_0 \cdot (D/N)^{\rho}}\right), \qquad \eta_0 = 0.946,\; R_0 = 471,\; \rho = -0.755}$$

Joint LOO RMSE $= 0.027$. Only 2 free parameters. Algebraically forces
$\eta \to 1$ as $D' \to 0$ and $\eta \to 0$ as $D' \to \infty$.

**Form C — Saturating MM with $N$-dependent $b$** (previous champion):

$$\boxed{\quad \eta_C(D, D'; N) \;=\; \frac{c \cdot (D/N)^{-\gamma}}{1 + b_0 \cdot (N/N_{\text{ref}})^{\kappa} \cdot D'/D},\qquad c=3.51,\;\gamma=0.317,\;b_0=0.062,\;\kappa=0.54,\;N_{\text{ref}}=30\,\text{M} \quad}$$

Joint LOO RMSE $= 0.029$. 4 parameters; not constrained to $\eta \le 1$,
which is *worse* on physical-cleanness grounds but lets it fit the
joint-anchor $\eta > 1$ artefacts at 190M / 370M (§3.4).

**Joint form ranking (LOO RMSE on 102 pooled points,
new joint Chinchilla anchors):**

| form | $n_{\text{par}}$ | RMSE | LOO |
|---|---|---|---|
| **A. exp $(D'/D)$, $R(D/N)$** | 3 | **0.026** | **0.026** |
| **B. Muennighoff Eq 5** | 2 | 0.027 | 0.027 |
| **C. sat $\times$ $(D/N)$, $b(N)$** | 4 | 0.029 | 0.029 |
| sat $\times$ $(D/N)$ | 3 | 0.029 | 0.030 |
| sat $(D'/D)$ | 2 | 0.029 | 0.030 |
| exp $(D'/D)$ | 2 | 0.029 | 0.029 |
| double power | 3 | 0.032 | 0.033 |
| const | 1 | 0.035 | 0.036 |

Per-size LOO RMSE for each contender:

| size | $n$ | **A. exp, $R(D/N)$** | **B. Muennighoff** | **C. sat × $(D/N)$, $b(N)$** |
|---|---|---|---|---|
| 14M  | 23 | 0.022 | **0.015** | 0.015 |
| 30M  | 28 | 0.022 | **0.019** | 0.031 |
| 60M  | 23 | 0.027 | 0.025 | 0.034 |
| 190M | 16 | **0.015** | 0.015 | **0.011** |
| 370M | 12 | 0.009 | 0.012 | **0.007** |

A and B are very close on the joint pool; they differ in shape — A allows
$\eta_0 < 1$ (e.g., here $\eta_0 = 0.95$, so the very first repeated
token is worth ~95% of fresh), while B forces $\eta(D'\!\to\!0) = 1$
exactly. C wins at the two largest sizes because it can fit the
joint-anchor $\eta > 1$ artefacts (§3.4); A and B cannot.

---

## 1. Fitting procedure

All fits use the LSE parameterization and the same optimizer, to match the
Hoffmann / Besiroglu methodology:

- **Form:** $\log L = \mathrm{logsumexp}(e,\; b - \beta \cdot \log D)$,
  algebraically identical to $L = e^{e} + e^{b}/D^{\beta}$, but numerically
  stable and log-space-friendly.
- **Loss:** Huber on $\log L$ with threshold $\delta$ (swept; see §1.2).
- **Optimizer:** L-BFGS with strong-Wolfe line search.
- **Init:** grid-search over literature-informed log-spaced values for
  $(e, b, \beta)$; each grid point is briefly optimized, best in-sample
  loss wins, then polished.
- **Diagnostics:** in-sample RMSE, LOO cross-validation RMSE (honest OOB),
  warm-started bootstrap ($n_{\text{boot}}=500$) for parameter 95% CIs.

Shared fitter: [fit_lse.py](fit_lse.py) — ~130 LOC, reusable by every form.

### 1.1 Outlier handling: residual-based top-$k$ drop

Following [Besiroglu 2024](Analyzing_Chinchilla_data.ipynb), we drop
the $k$ highest-error points after fitting and refit. We use the
**iterative greedy** variant (also known as IRLS-style trimming):

1. Fit on all data.
2. Drop the single point with the largest $|\Delta \log L|$.
3. Refit, recompute residuals, drop the next worst.
4. Repeat $k$ times.

This handles the case where many points coherently bias the fit (a
one-shot rank-and-drop gives the same answer as a fixed scale-cut and
has trouble seeing past a cluster). The earlier 30M-only analysis used
a *scale-based* cut (drop scales $\le 0.25\times$); the multi-size
joint fit uses **residual-based** drop on all 55 1-epoch points (no
scale floor).

**Drop sweep on the joint Chinchilla** (all 55 1-epoch points, $\delta=0.1$):

| $k$ | $n$ | $E$ | $A$ | $B$ | $\alpha$ | $\beta$ | RMSE | $R^2$ |
|---|---|---|---|---|---|---|---|---|
| 0  | 55 | 0.082 | 51.5 | 498 | 0.205 | 0.240 | 0.067 | 0.970 |
| 5  | 50 | 0.051 | 23.7 | 2 863 | 0.135 | 0.337 | 0.053 | 0.978 |
| 8  | 47 | 0.000 | 22.4 | 8 680 | 0.123 | 0.396 | 0.043 | 0.984 |
| 12 | 43 | 1.097 | 105 | 11 199 | 0.238 | 0.413 | 0.035 | 0.990 |
| 16 | 39 | 1.523 | 363 | 20 641 | 0.320 | 0.448 | 0.027 | 0.995 |
| **20** | **35** | **1.72** | **1115** | **20 828** | **0.390** | **0.451** | **0.019** | **0.997** |
| 25 | 30 | 1.83 | 2034 | 26 180 | 0.427 | 0.464 | 0.014 | 0.998 |

**Canonical $k$ selection.** $\beta$ stops moving (consecutive $|\Delta\beta|<0.01$)
between $k=16 \to 20$; we commit to $k=20$. By that point the fit has
removed roughly the same set of small-scale / high-loss points that the
legacy "scale $\ge 0.5\times$" cut removed, and the result is in
agreement: legacy fit gave $\beta=0.459$ at RMSE 0.032; residual-drop
gives $\beta=0.451$ at RMSE 0.019.

**What gets dropped (k=20).** The 20 worst-residual points are heavily
concentrated at small scales (0.05x, 0.1x — across all sizes), with a
few at very large scales (16x@190M, 8x@370M). Specifically these
points were dropped by the iterative greedy: 30m/0.05x, 14m/0.05x,
14m/0.1x, 14m/0.1x, 600m/0.05x, 30m/0.1x, 14m/0.25x, 60m/0.05x,
370m/0.1x, 190m/0.1x, 100m/0.25x, 190m/0.25x, 60m/1x, 100m/0.5x,
600m/0.1x, 100m/0.1x, 190m/0.05x, 30m/1x, ... Note this is not a clean
"scale cut" — points at 1x and 0.5x at certain sizes are also
inconsistent with the joint shape and get pruned.

**Takeaway.** Don't pre-commit to a scale floor. Fit on all data, drop
top-$k$ residuals, sweep $k$, pick the value where $\beta$ stabilizes.
This both retains more data (35 vs 34) and gives a tighter fit
(RMSE 0.019 vs 0.032) than a hard scale cut.

For the 30M single-size analysis (kept for historical context below),
we still report the older scale-cut $k$-sweep — the qualitative
picture is the same.

### 1.2 Huber $\delta$ sensitivity

Fitting on the $k=3$ set, we sweep $\delta$:

| $\delta$ | regime | $E$ | $\beta$ | RMSE $_{\text{in}}$ | RMSE $_{\text{LOO}}$ |
|---|---|---|---|---|---|
| $1.0$ / $0.1$ | $L_2$-like | **3.01** | **0.486** | 0.025 | 0.071 |
| $0.03$ | mixed | 2.96 | 0.466 | 0.025 | — |
| $0.01$ | transition | 2.85 | 0.416 | 0.029 | — |
| $10^{-3}$ (Besiroglu) | $L_1$-like | 2.54 | 0.355 | 0.031 | **0.045** |
| $10^{-5}$ | pure $L_1$ | 2.94 | 0.426 | 0.031 | 0.061 |

Initially we picked $\delta=10^{-3}$ (lowest LOO RMSE). Then the $\eta$
sanity-check (§2.3) revealed that $\beta=0.355$ is *too shallow* for
multi-epoch extrapolation, producing $\eta > 1$ for many points.
**Switching to $\delta=0.1$ gives $\beta=0.486$**, which extrapolates
cleanly to multi-epoch with $\eta < 1$ almost everywhere. Tradeoff:
1-epoch LOO RMSE rises from $0.045 \to 0.071$, but the multi-epoch $\eta$
fit gets better (LOO $0.040 \to 0.034$) and physically consistent.

**Why 1-epoch LOO picks the wrong fit .**

LOO holds out each of the 6 fit points ($D \in [3\times 10^8,\, 9.6 \times 10^9]$)
and predicts it from the others. That's an *interpolation* test within a
narrow $D$ range; the held-out point always has neighbours on both sides.
Both candidate fits interpolate well (max $|\Delta \log L| < 0.07$), so
LOO can't tell them apart on the *fit* range.

But multi-epoch training doesn't live in the 1-epoch range. Effective
tokens are $D_{\text{eff}} = D + \eta \cdot D'$; at $0.5\times$ 64ep
with a reasonable $\eta$ we're already at $D_{\text{eff}} \sim 10^{10}$,
past the max $D$ in our 1-epoch fit. Heavier multi-epoch pushes well
beyond. Small differences in $\beta$ barely matter *inside* the fit
range and become large differences *outside* it:

| fit | at $D = 9.6 \times 10^9$ (in-range) | at $D_{\text{eff}} = 2 \times 10^{10}$ (extrapolation) |
|---|---|---|
| $\delta=10^{-3}$ ($\beta=0.355$) | $L = 3.68$ | $L = 3.21$ |
| $\delta=0.1$ ($\beta=0.486$) | $L = 3.68$ | $L = 3.45$ |

Both fits agree inside the data range and disagree by $\sim 0.24$ in raw
loss ($\sim 0.07$ in $\log L$) precisely where we need them to agree —
which is the regime multi-epoch training actually hits.

Multi-epoch data tells us which extrapolation is right. Invert
$L = E + B/D_{\text{eff}}^{\beta}$ at a clean multi-epoch observation like
$0.5\times$ 4ep ($L = 4.87$, $D = 3 \times 10^8$, $D' = 9 \times 10^8$):

- $\delta=10^{-3}$ fit ⟹ $D_{\text{eff}} = 1.29 \times 10^9$ ⟹ $\eta = 1.10$ (non-physical)
- $\delta=0.1$ fit ⟹ $D_{\text{eff}} = 1.16 \times 10^9$ ⟹ $\eta = 0.96$ ✓

The shallower $\beta$ requires "more effective tokens than $D + D'$" to
explain the observed loss, i.e. $\eta > 1$ — the $\eta < 1$ constraint
is acting as evidence about $\beta$ at the large-$D$ extrapolation.

**General lesson.** LOO is the right selection criterion when the
quantity of interest is interpolation. Here we care about extrapolation
to a different regime, and six in-range points leave that extrapolation
underdetermined. Cross-validating against an *independent* data source
(the multi-epoch set, via the $\eta < 1$ physical constraint) is
strictly more informative for picking $\beta$.

### 1.3 Chosen 1-epoch fit

$$\boxed{\quad E = 3.010, \quad B = 47\,022, \quad \beta = 0.4863 \quad}$$

(k=3 cut, $\delta = 0.1$.)

Fit diagnostics at other cuts (for reference):
- [fit_1epoch_30m_min0_0x.pdf](fit_1epoch_30m_min0_0x.pdf) — $k=0$ (all 9 pts; broken)
- [fit_1epoch_30m_min0_1x.pdf](fit_1epoch_30m_min0_1x.pdf) — $k=1$ (drop $0.05\times$)
- [fit_1epoch_30m_min0_25x.pdf](fit_1epoch_30m_min0_25x.pdf) — $k=2$ (drop $0.05\times$, $0.1\times$)
- [fit_1epoch_30m_min0_5x.pdf](fit_1epoch_30m_min0_5x.pdf) — **$k=3$ (our choice)**

Cut-sweep diagnostics (bootstrap CIs, LOO residuals):
- [tune_1epoch_30m_delta_0_001.pdf](tune_1epoch_30m_delta_0_001.pdf) — $\delta = 10^{-3}$
- [tune_1epoch_30m_delta_0_1.pdf](tune_1epoch_30m_delta_0_1.pdf) — $\delta = 0.1$

Code: [fit_1epoch_30m.py](fit_1epoch_30m.py), [tune_1epoch_30m.py](tune_1epoch_30m.py).

---

## 2. $\eta$ fitting

With $(E, B, \beta)$ fixed, we fit $\eta$ on multi-epoch data using the
same machinery: log-$L$ Huber + L-BFGS + grid-search init.

**Data:** multi-epoch points (epochs $> 1$) at scale $\ge 0.5\times$
(consistent with the 1-epoch cut), excluding u-shape overfit pairs
($0.05\times$ / $128$ ep, $0.1\times$ / $128$ ep). 21 points covering scales
$\{0.5, 1, 2, 4, 8\}\times$.

### 2.1 Candidate forms

Since the functional form is not known, we try six candidates:

| name | form | $n_{\text{par}}$ |
|---|---|---|
| const | $\eta = c$ | 1 |
| power($D/N$) | $\eta = c \cdot (D/N)^{-\gamma}$ | 2 |
| power($D'/D$) | $\eta = c \cdot (D'/D)^{-\gamma}$ | 2 |
| sat($D'/D$) | $\eta = c \,/\, (1 + b \cdot D'/D)$ | 2 |
| **sat $\times$ ($D/N$)** | $\boldsymbol{\eta = c \cdot (D/N)^{-\gamma} \,/\, (1 + b \cdot D'/D)}$ | **3** |
| double power | $\eta = c \cdot (D/N)^{-\gamma_1} \cdot (D'/D)^{-\gamma_2}$ | 3 |

"sat" = saturating (Michaelis–Menten): $\eta$ grows linearly at small
$D'/D$ and flattens at large $D'/D$. In $\eta \cdot D'$ terms, repetition
asymptotes to a finite ceiling of $c \cdot D / b$ extra fresh-equivalent
tokens.

### 2.2 Results (1-epoch anchor: $\delta=0.1$, $\beta=0.486$)

| form | $n_{\text{par}}$ | RMSE $_{\text{in}}$ | $\max \|\Delta \log L\|$ | $R^2$ | **RMSE $_{\text{LOO}}$** |
|---|---|---|---|---|---|
| **sat $\times$ ($D/N$)** | 3 | **0.030** | 0.102 | **0.920** | **0.034** |
| sat($D'/D$) | 2 | 0.031 | 0.055 | 0.913 | 0.036 |
| power($D'/D$) | 2 | 0.038 | 0.075 | 0.871 | 0.045 |
| double power | 3 | 0.042 | 0.160 | 0.842 | 0.047 |
| power($D/N$) | 2 | 0.048 | 0.103 | 0.792 | 0.051 |
| const | 1 | 0.053 | 0.093 | 0.755 | 0.057 |

**Winner:** $\eta = c \cdot (D/N)^{-\gamma} / (1 + b \cdot D'/D)$ with
$c=6.66$, $\gamma=0.48$, $b=0.10$.

Observations:
- Both ingredients matter: the $(D/N)$ dependence and the $(D'/D)$
  saturation each contribute significantly over the other alone.
- Saturation (MM) beats power-law in $D'/D$ — repetition genuinely
  diminishes, doesn't just decay.
- Double power is worse than sat $\times$ $(D/N)$ despite equal
  $n_{\text{par}}$: the saturation shape in $D'/D$ matches the data
  better than a second power law.

Figure: [fit_eta_30m_form_comparison.pdf](fit_eta_30m_form_comparison.pdf)
— 2-row grid with all 6 forms sorted by LOO RMSE; top row shows $\eta$
vs $D'$ with per-point $\eta$ (dots) and fitted curves, bottom row shows
residuals.

Best-form detail: [fit_eta_30m_sat_×_DoverN.pdf](fit_eta_30m_sat_×_DoverN.pdf)
— $\eta$ vs $D'$ by scale, loss collapse onto the 1-epoch curve,
residuals vs $D'$.

### 2.3 Sanity check: $\eta < 1$

Per-point analytical $\eta$: min $0.14$, median $0.62$, max $1.23$.
**3 / 21 points have $\eta > 1$** — all at small $D$, few epochs
($0.5\times$ @ 2–4ep, $1\times$ @ 4ep). Down from 6/21 with
$\delta=10^{-3}$.

Remaining $\eta > 1$ points are at the regime boundary where the
1-epoch extrapolation is still imperfect. They're small deviations
($\max \eta = 1.23$, within $\sim 25\%$ of 1.0). The parametric
sat $\times$ $(D/N)$ form predicts $\eta < 1$ everywhere for this data
range; the per-point excursions above 1 are residuals of the Chinchilla
form itself, not signals the $\eta$ form is wrong.

### 2.4 Chosen $\eta$ fit

$$\boxed{\quad \eta(D, D'; N) = \frac{c \cdot (D/N)^{-\gamma}}{1 + b \cdot D'/D},
\qquad c = 6.66,\; \gamma = 0.48,\; b = 0.10 \quad}$$

Saturation ceiling (fresh-equivalent multiplier of $D$ as $D' \to \infty$):

$$\lim_{D' \to \infty} \frac{\eta \cdot D'}{D} \;=\; \frac{c \cdot (D/N)^{-\gamma}}{b}$$

| scale | $D/N$ | ceiling |
|---|---|---|
| $0.5\times$ | 10 | $\sim 22\times$ |
| $16\times$ | 320 | $\sim 4.2\times$ |

Matches the DCLM writeup's finding of $\sim 20\times$ ceiling at small
$D$, $\sim 4\times$ at large $D$.

Code: [fit_eta_30m.py](fit_eta_30m.py).

---

## 3. Multi-size: joint Chinchilla + per-size $\eta$

Data now spans seven sizes: 14M, 30M, 60M, 100M, 190M, 370M, 600M.
100M and 600M are 1-epoch only; the other five have multi-epoch sweeps.

### 3.1 Joint 1-epoch Chinchilla fit

Fitting the 5-parameter form on all 55 1-epoch points (no scale floor,
$\delta=0.1$), using residual-based iterative top-$k$ drop with the
canonical $k=20$ (see §1.1):

$$\log L = \mathrm{logsumexp}(e,\; a - \alpha \log N,\; b - \beta \log D)$$

$$\boxed{\quad E = 1.72,\quad A = 1115,\quad B = 20{,}828,\quad \alpha = 0.390,\quad \beta = 0.451 \quad}$$

Global RMSE $(\log L) = 0.019$, $R^2 = 0.997$ on the 35 retained
points. Per-size residuals at the canonical-$k$ fit:

| size | $n_{\text{kept}}$ | RMSE | max $abs(\Delta \log)$ | implied $E_{\text{eff}}(N)$ |
|---|---|---|---|---|
| 14M  | 6 | 0.054 | 0.125 | 2.635 |
| 30M  | 7 | 0.029 | 0.054 | 2.383 |
| 60M  | 8 | 0.054 | 0.113 | 2.175 |
| 100M | 7 | 0.049 | 0.081 | 2.034 |
| 190M | 9 | 0.056 | 0.086 | 1.870 |
| 370M | 8 | 0.070 | 0.105 | 1.714 |
| 600M | 5 | 0.037 | 0.061 | 1.609 |

Compared to the legacy "$\ge 0.5\times$" hard-cut fit ($\beta=0.459$,
RMSE 0.032), residual drop arrives at the same $\beta$ ($0.451$) at
lower RMSE on more points (35 vs 34). The two fits implicitly remove
similar points but the residual-drop set isn't a clean scale floor —
e.g. at $k=20$ both 0.05x@14M *and* 1x@30M get pruned alongside many
small-scale points.

Figure: [fit_chinchilla_joint.pdf](fit_chinchilla_joint.pdf) — $L$ vs $D$
coloured by $N$ with fit curves per size, residuals, and parity plot.

Code: [fit_chinchilla_joint.py](fit_chinchilla_joint.py).

### 3.2 Per-size $\eta$ fits (shared anchors)

Fixing $(E, A, B, \alpha, \beta)$ at the joint-Chinchilla values, we fit
$\eta$ separately on each size's multi-epoch data. We report both
contenders side-by-side.

**Form B — sat × (D/N) per size** (3 params, the original):

| size | $n$ | $c$ | $\gamma$ | $b$ | LOO RMSE | η>1 |
|---|---|---|---|---|---|---|
| 14M  | 23 | 1.72 | 0.23 | 0.022 | 0.019 | 0/23 |
| 30M  | 28 | 8.77 | 0.55 | 0.110 | 0.033 | 4/28 |
| 60M  | 23 | 5.65 | 0.48 | 0.091 | 0.031 | 3/23 |
| 190M | 16 | 6.89 | 0.45 | 0.197 | 0.012 | 6/16 |
| 370M | 12 | 9.40 | 0.60 | 0.426 | 0.008 | 3/12 |

Trends: $b$ rises with $N$ (0.022 → 0.43) — larger models saturate
repetition faster. The ceiling $(D/N)^{-\gamma}/b$ therefore shrinks at
large $N$, matching the intuition that big models extract all the signal
from a single pass. $\gamma$ sits around 0.5 at mid sizes; the extremes
are noisier. The 4-parameter $b(N)$ generalization with
$b_0=0.071,\;\kappa=0.35$ collapses these per-size $b$ values onto a
single $b_{\text{eff}}(N) = 0.071 \cdot (N/30\text{M})^{0.35}$.

**Form A — Muennighoff Eq 5 per size** (2 params):

| size | $n$ | $R_0$ | $\rho$ | LOO RMSE |
|---|---|---|---|---|
| 14M  | 23 | 288  | $-0.90$ | **0.009** |
| 30M  | 28 | 558  | $-1.14$ | **0.020** |
| 60M  | 23 | 191  | $-0.90$ | **0.018** |
| 190M | 16 | 236  | $-1.13$ | 0.025 |
| 370M | 12 | 100  | $-1.16$ | 0.015 |

$\rho$ is roughly $-1$ across all sizes (the saturation scale $R^*$
varies inversely with $D/N$, as one would expect for a fixed total
saturation budget per parameter). Per-size LOO is materially better
than form B at 14M / 30M / 60M (often $\sim 2\times$); slightly worse at
190M / 370M because of the $\eta > 1$ artefact those sizes carry — see
§3.4.

**Figures:**
- [fit_eta_per_size.pdf](fit_eta_per_size.pdf) — grid of the 4 top
  forms (rows) $\times$ 5 sizes (columns) on $\eta$ vs $D'/D$ log-log,
  with per-point $\eta$ (grey) and parametric $\eta$ (blue), plus LOO
  RMSE annotated in each panel. Top row = `sat × (D/N)` (3-param),
  bottom row = Muennighoff Eq 5 (2-param).
- [fit_eta_AvsB_per_size.pdf](fit_eta_AvsB_per_size.pdf) — head-to-head
  per size: Muennighoff (red, solid) vs `sat × (D/N), b(N)` (blue,
  dashed) curves overlaid on per-point $\eta$ (grey). Title of each
  panel reports the per-size LOO RMSE for both. Visually:
  Muennighoff hugs the data and stays $\le 1$ at small/mid $N$; B is
  free to climb above 1 at 190M/370M and matches the artefact points.

### 3.3 Joint $\eta$ fits (pooled across sizes)

Pooling all 102 multi-epoch points, we fit each candidate form to a
single $(c, \gamma, ...)$ shared across sizes. Two new forms, inspired
by Muennighoff '23 (Eq 5 of the data-repetition paper), bake in the
physical constraint $\eta \le 1$:

- **exp $(D'/D)$**: $\eta = \eta_0 \cdot \exp\!\left(-(D'/D)/R\right)$.
  At $D' = 0$, $\eta = \eta_0$; if $\eta_0 \le 1$, then $\eta \le 1$
  everywhere.
- **Muennighoff Eq 5**: $\eta = R^*(1 - e^{-x/R^*}) / x$, $x = D'/D$.
  Algebraically forces $\eta \to 1$ as $x \to 0$ (the first repeated
  token = fresh) and $\eta \to 0$ as $x \to \infty$ (full saturation).
  Only 1 parameter without TTP-dep, 2 with $R^* = R_0 \cdot (D/N)^{\rho}$.

Joint LOO RMSE comparison:

| form | $n_{\text{par}}$ | RMSE | LOO | $R^2$ | params |
|---|---|---|---|---|---|
| const | 1 | 0.045 | 0.046 | 0.956 | $c=0.72$ |
| power $(D/N)$ | 2 | 0.040 | 0.040 | 0.966 | $c=2.08$, $\gamma=0.35$ |
| power $(D'/D)$ | 2 | 0.038 | 0.040 | 0.970 | $c=1.17$, $\gamma=0.31$ |
| sat $(D'/D)$ | 2 | 0.035 | 0.036 | 0.973 | $c=1.02$, $b=0.049$ |
| exp $(D'/D)$ | 2 | 0.036 | 0.036 | 0.973 | $\eta_0=0.94$, $R=41$ |
| sat $\times$ $(D/N)$ | 3 | 0.027 | 0.028 | 0.984 | $c=4.00$, $\gamma=0.41$, $b=0.068$ |
| exp $(D'/D)$, $R(D/N)$ | 3 | 0.025 | 0.026 | 0.986 | $\eta_0=0.93$, $R_0=530$, $\rho=-0.83$ |
| double power | 3 | 0.030 | 0.034 | 0.980 | $c=3.54$, $\gamma_1=0.33$, $\gamma_2=0.34$ |
| sat $\times$ $(D/N)$, $b(N)$ | 4 | 0.026 | 0.028 | 0.986 | $c=4.88$, $\gamma=0.45$, $b_0=0.071$, $\kappa=0.35$ |
| **Muennighoff Eq 5** | **2** | **0.023** | **0.023** | **0.989** | $R_0=275$, $\rho=-0.97$ |

**Two contenders carried forward:**

- **A. Muennighoff Eq 5** ($n_{\text{par}}=2$): wins the joint pool
  (LOO 0.023) and 14M / 30M / 60M individually. $\eta \le 1$ enforced
  by construction. Equivalent compact form (using $\rho \approx -1$):
  $R^{*} \approx R_{0} / (D/N)$, so $R^{*} \cdot (D/N) \approx 275$
  across all sizes. Saturation hits at roughly the same total-token
  count $R^{*} \cdot D \approx 275 \cdot N$ regardless of scale — a
  clean physical statement.
- **B. sat $\times$ $(D/N)$, $b(N)$** ($n_{\text{par}}=4$): the
  previous champion. Worse on the joint pool (LOO 0.028) but stays the
  best at 190M / 370M because it can fit the $\eta > 1$ joint-anchor
  artefacts there (see §3.4) — a worse property in principle, but
  empirically lower RMSE on those sizes. Has two derived statements
  worth keeping: $b_{\text{eff}}(N) = 0.071 \cdot (N/30\text{M})^{0.35}$
  (saturation grows with $N$), and a closed-form ceiling
  $\eta \cdot D' / D \to c \cdot (D/N)^{-\gamma} / b_{\text{eff}}(N)$
  (e.g., $\sim 22\times$ at $D/N=10$, $\sim 4\times$ at $D/N=320$).

**The user's own exponential form** $\eta = \eta_0 \exp(-(D'/D)/R(D/N))$
also outperforms the sat-family with one fewer parameter ($n=3$, joint
LOO 0.026) — it sits between A and B on every diagnostic.

**Figures:**
- [fit_eta_joint.pdf](fit_eta_joint.pdf) — best joint form (Muennighoff
  Eq 5): $\eta$ (per-point + fitted) and residuals vs $D'/D$, coloured
  by size.
- [fit_eta_AvsB_joint.pdf](fit_eta_AvsB_joint.pdf) — 2$\times$2 grid:
  top row = $\eta$ scatter (per-point dots, fitted ×) for forms A and
  B; bottom row = log-loss residuals vs $D'/D$ for each. Lets you see
  where each form deviates from the data — A's residuals are tighter
  on the small/mid-$N$ population, B's are tighter on the large-$N$
  outliers.

Code: [fit_eta.py](fit_eta.py). Shared data loader:
[data.py](data.py).

### 3.4 Non-physical $\eta > 1$: a joint-anchor artefact

Per-point analytical $\eta$ produces some non-physical values
($\eta > 1$, occasionally $\eta < 0$):

| size | $\eta > 1$ | per-point $\eta$ range | per-size η LOO |
|---|---|---|---|
| 14M  | 0/23  | [0.20, 1.00] | 0.019 |
| 30M  | 4/28  | [0.04, 1.29] | 0.033 |
| 60M  | 3/23  | [0.12, 1.21] | 0.031 |
| 190M | 6/16  | [−0.27, 2.00] | 0.012 |
| 370M | 3/12  | [0.16, 1.48] | 0.008 |

190M / 370M look the worst on $\eta > 1$ counts but actually have the
*best* per-size η fit quality (LOO = 0.012, 0.008). The non-physical
$\eta$ values are a property of inverting the joint 1-epoch anchors at
those sizes, not a property of the η form.

**Mechanism.** Recall

$$\eta = \frac{(B/(L-E_{\text{eff}}))^{1/\beta} - D}{D'}.$$

If the joint 1-epoch curve over-predicts loss at a given $N$, the
multi-epoch loss looks better-than-expected and the inversion returns a
$D_{\text{eff}}$ larger than the actual (total tokens), giving
$\eta > 1$. The 16x 2ep point at 190M flips the other way at very
large $D$ (joint under-predicts there) and produces $\eta < 0$.

**Direct evidence — fitting $\beta$ per size with $E_{\text{eff}}$
fixed from joint** (so the 3-param fit can't absorb $E$-mismatch into
$\beta$):

| size | $\beta_{3p}$ (free $E$) | $\beta_{2p}$ ($E$ from joint) |
|---|---|---|
| 14M  | 0.20 | 0.39 |
| 30M  | 0.49 | 0.48 |
| 60M  | 0.59 | 0.55 |
| 100M | 0.81 | 0.62 |
| 190M | 0.76 | **0.43** |
| 370M | 0.80 | **0.36** |
| 600M | 0.42 | 0.37 |

Joint $\beta = 0.460$. The 2-param column shows $\beta$ is
**non-monotonic in $N$** — rises from 14M to 100M, then drops at 190M+.
The free-$E$ 3-param column is misleading: with 4–6 points per size it
interpolates (RMSE $< 0.002$) and absorbs $E$-mismatch by inflating
$\beta$.

So 190M / 370M actually want a *shallower* $\beta$ than joint, not
steeper. The joint compromise overshoots at large $N$, producing the
$\eta > 1$ pattern.

See [plot_beta_by_N.pdf](plot_beta_by_N.pdf) for the log-log diagnostic
and [fit_eta_per_size.pdf](fit_eta_per_size.pdf) for the per-size η fits
(where the joint-anchor mis-extrapolation has no effect — those fits
have their own implicit anchors).

**Ways forward:**
1. Pick 1-epoch parameters such that $\eta$ gets reasonable values.
<!-- 1. **Add $\beta(N)$ to the joint 1-epoch fit.** A 6-parameter form
   $L = E + A/N^{\alpha} + B/D^{\beta(N)}$ with $\beta(N) = \beta_0 +
   \beta_1 \log(N/N_{\text{ref}})$ adds 1 parameter and should kill
   most of the per-size residual structure.
2. **Accept per-size η parameters.** Form B's per-size fit gives LOO
   RMSE 0.008–0.033 across sizes; form A is even tighter at small/mid -->
   <!-- $N$ (0.009–0.020). Both already beat the corresponding joint form. -->

---
<!-- 
## 4. Dolma-30M data summary

Per-scale data availability (excluding u-shape overfit points):

| scale | 1-ep? | multi-ep epochs | flagged |
|---|---|---|---|
| $0.05\times$ | yes | 2,4,8,16,32,64 | drop 128ep (overfit); excluded from fits |
| $0.1\times$ | yes | 2,4,8,16,32,64 | drop 128ep; excluded |
| $0.25\times$ | yes | 2,4,8,16,32,64 | excluded |
| **$0.5\times$** | yes | 2,4,8,16,32,64 | ✓ |
| **$1\times$** | yes | 4,8,16,32,64 | ✓ (missing 2ep) |
| **$2\times$** | yes | 4,8,16,32,64 | ✓ (missing 2ep) |
| **$4\times$** | yes | 4,8,16,32 | ✓ (missing 2ep, 64ep) |
| **$8\times$** | yes | 2 | ✓ (only 2ep available) |
| **$16\times$** | yes | — | ✓ 1-ep only |

Data requests that would materially help:
- Fill in $8\times$ multi-epoch (currently only 1ep + 2ep). A run at
  $8\times$ 4/8/16ep would anchor the large-$D$ saturation of $\eta$.
- Fill in $16\times$ 2/4ep. Would let us verify $\eta$ still saturates
  at $D/N = 320$.
- 2ep at $1\times$/$2\times$/$4\times$ is minor; helps shape at low
  epoch counts.

Overfit points (u-shape in loss vs epochs) excluded from all fits:
$\{(0.05\times, 128\text{ep}),\; (0.1\times, 128\text{ep})\}$.

--- -->

## 4. Open questions

1. **Size-dependent $\beta$ in the 1-epoch fit.** The joint fit uses
   a single $\beta = 0.460$. With $E_{\text{eff}}(N)$ fixed from joint,
   per-size 2-param fits give $\beta$ ranging 0.36 (370M) to 0.62
   (100M), non-monotonic in $N$. A 6-parameter joint form
   $L = E + A/N^{\alpha} + B/D^{\beta(N)}$ with
   $\beta(N) = \beta_0 + \beta_1 \log(N/N_{\text{ref}})$ would absorb
   most of the per-size residual without overfitting (only 1 extra
   parameter on 34 points). With $\beta(N)$ in place, the
   $\eta > 1$ artefact at 190M / 370M should largely disappear, and
   the Muennighoff form should win those sizes too.
2. **Bootstrap CIs for $\eta$ across sizes.** Warm-started LOO already
   handles stability diagnostics; a full bootstrap would give CIs on
   $(R_0, \rho)$. ~30 s per size.
3. **Paraphrase / synthetic data extension.** The framework extends
   directly via $\eta_{\text{para}}$ on the second token stream. Not
   currently in the Dolma data.
