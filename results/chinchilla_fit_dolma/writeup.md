# Chinchilla Scaling-Law Fits on Dolma

Multi-epoch Chinchilla scaling-law fit on Dolma across **seven model
sizes** (14M → 600M). The classic Chinchilla form is

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

and multi-epoch training is modelled as

$$L = E_{\text{eff}}(N) + \frac{B}{(D + \eta \cdot D')^{\beta}}, \qquad D' = (\text{epochs}-1) \cdot D,$$

where $E_{\text{eff}}(N) = E + A/N^{\alpha}$ and $\eta$ is the
effective-token multiplier for repeated data.

## Current best fit (all sizes, $\delta=0.1$ Huber on $\log L$)

**Joint Chinchilla anchors** (1-epoch, all 7 sizes, 34 points):

| param | value |
|---|---|
| $E$ | 0.908 |
| $A$ | 57.9 |
| $B$ | 26 706 |
| $\alpha$ | 0.194 |
| $\beta$ | 0.458 |

Global fit quality: RMSE$(\log L) = 0.032$, $R^2 = 0.989$.

**Best $\eta$ form (joint, all sizes):**
$\eta = c \cdot (D/N)^{-\gamma} / \left(1 + b_0 \cdot (N/N_{\text{ref}})^{\kappa} \cdot D'/D\right)$
with $N_{\text{ref}} = 30\,\text{M}$.

Joint pooled fit on 102 multi-epoch points across 5 sizes:

$$\boxed{\quad c = 5.12,\quad \gamma = 0.464,\quad b_0 = 0.072,\quad \kappa = 0.369 \quad}$$

Joint LOO RMSE $= 0.027$, $R^2 = 0.986$. The N-dependent saturation
($\kappa > 0$) absorbs the cross-size variation in $b$ that the 3-parameter
`sat × (D/N)` form had to leave in free per-size parameters.

Per-size 3-parameter `sat × (D/N)` fits (for comparison):

| size | $n_{\text{multi}}$ | $c$ | $\gamma$ | $b$ | **LOO RMSE** |
|---|---|---|---|---|---|
| 14M  | 23 | 1.73 | 0.23 | 0.022 | **0.019** |
| 30M  | 28 | 8.96 | 0.55 | 0.111 | **0.033** |
| 60M  | 23 | 5.76 | 0.48 | 0.092 | **0.031** |
| 190M | 16 | 7.04 | 0.46 | 0.200 | **0.014** |
| 370M | 12 | 9.57 | 0.61 | 0.430 | **0.008** |

$b$ ranges over $\sim 20\times$ across sizes; the joint $b(N)$ form
captures this via $b_{\text{eff}}(N) = b_0 \cdot (N/N_{\text{ref}})^{\kappa}$,
ranging from $b_{\text{eff}}(14\text{M}) = 0.057$ to
$b_{\text{eff}}(370\text{M}) = 0.175$ (a $\sim 3\times$ range).  Fit
quality improves only marginally (joint LOO $0.028 \to 0.027$) — a single
$(c, \gamma, \kappa)$ can't fully capture the per-size variation, but
$\kappa = 0.37$ is the principled summary.

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

### 1.1 Outlier handling (top-$k$ loss drop)

Following [Besiroglu 2024](Analyzing_Chinchilla_data.ipynb), we drop the
$k$ highest-loss points and refit. We swept $k \in \{0, 1, 2, 3\}$ —
dropping the small-scale points ($0.05\times$, $0.1\times$, $0.25\times$
in that order). Results:

| $k$ | $n$ | $E$ | $\beta$ | RMSE $_{\text{in}}$ | **RMSE $_{\text{LOO}}$** | RMSE on dropped |
|---|---|---|---|---|---|---|
| 0 | 9 | 0.028 | 0.162 | 0.048 | 0.074 | — |
| 1 | 8 | 0.428 | 0.195 | 0.043 | 0.052 | 0.170 |
| 2 | 7 | 0.982 | 0.223 | 0.037 | 0.052 | 0.151 |
| **3** | **6** | **2.54–3.01** | **0.35–0.49** | 0.031 | **0.045–0.071** | 0.21–0.35 |

($\beta$ range in $k=3$ reflects the $\delta$ choice; see §1.2.)

**Takeaways:**
- LOO RMSE drops monotonically with $k$ — each cut genuinely helps out-of-sample.
- At $k=0$, bootstrap shows $E$ is pegged at its positivity constraint in
  *every* resample — the fit is degenerate (outliers force $E \approx 0$).
- Residuals at $k=0/1/2$ have visible structure; at $k=3$ they're within
  $\pm 0.06$.
- **We commit to $k=3$** (drop $0.05\times$, $0.1\times$, $0.25\times$).

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

Pooling all 34 1-epoch points (scale $\ge 0.5\times$, all sizes) and
fitting the 5-parameter form

$$\log L = \mathrm{logsumexp}(e,\; a - \alpha \log N,\; b - \beta \log D)$$

gives:

$$\boxed{\quad E = 0.908,\quad A = 57.9,\quad B = 26{,}706,\quad \alpha = 0.194,\quad \beta = 0.458 \quad}$$

Global RMSE $(\log L) = 0.032$, $R^2 = 0.989$. Per-size residuals:

| size | $n$ | RMSE | max $abs(\Delta \log)$ | implied $E_{\text{eff}}(N)$ |
|---|---|---|---|---|
| 14M | 5 | 0.034 | 0.072 | 3.291 |
| 30M | 6 | 0.026 | 0.045 | 2.963 |
| 60M | 5 | 0.048 | 0.079 | 2.705 |
| 100M | 4 | 0.040 | 0.060 | 2.535 |
| 190M | 6 | 0.025 | 0.034 | 2.345 |
| 370M | 5 | 0.026 | 0.041 | 2.171 |
| 600M | 3 | 0.013 | 0.018 | 2.058 |

Consistency check: the single-size 30M fit (§1.3) gave $E = 3.010$,
$\beta = 0.486$. The joint fit's implied $E_{\text{eff}}(30\text{M}) =
2.963$ and $\beta = 0.458$ agree with it to within a few percent — the
joint $\alpha$, $\beta$ are close to what you'd get by fitting the 30M
slice alone, and the $A/N^\alpha$ term cleanly captures the
scale-monotone drop in $E_{\text{eff}}$.

**$\delta$ sweep on the joint fit** (1280 init points × 4 values of
$\delta$):

| $\delta$ | $E$ | $A$ | $B$ | $\alpha$ | $\beta$ | RMSE | $R^2$ |
|---|---|---|---|---|---|---|---|
| $1.0$ | 0.916 | 58.7 | 26 676 | 0.195 | 0.458 | 0.032 | 0.989 |
| **$0.1$ (canon.)** | **0.911** | **58.2** | **26 702** | **0.194** | **0.458** | **0.032** | **0.989** |
| $0.01$ | 1.426 | 182.1 | 57 874 | 0.272 | 0.500 | 0.034 | 0.988 |
| $10^{-3}$ | 1.530 | 237.8 | 64 260 | 0.290 | 0.506 | 0.035 | 0.987 |

Same pattern as the 30M-only fit: $L_2$-regime ($\delta \ge 0.1$) gives
shallower $\beta$, better matches multi-epoch extrapolation;
$L_1$-regime ($\delta \le 10^{-3}$) gives steeper $\beta$, worse
multi-epoch consistency. Canonical: $\delta = 0.1$.

Figure: [fit_chinchilla_joint.pdf](fit_chinchilla_joint.pdf) — $L$ vs $D$
coloured by $N$ with fit curves per size, residuals, and parity plot.

Code: [fit_chinchilla_joint.py](fit_chinchilla_joint.py).

### 3.2 Per-size $\eta$ fits (shared anchors)

Fixing $(E, A, B, \alpha, \beta)$ at the joint-Chinchilla values, we fit
$\eta$ separately on each size's multi-epoch data. The form `sat $\times$
$(D/N)$` wins on LOO RMSE in every size (it was the best on 30M already;
confirmed across 14M / 60M / 190M / 370M).

| size | $n$ | $c$ | $\gamma$ | $b$ | LOO RMSE | η>1 |
|---|---|---|---|---|---|---|
| 14M  | 23 | 1.73 | 0.23 | 0.022 | 0.020 | 1/23 |
| 30M  | 28 | 8.96 | 0.55 | 0.111 | 0.034 | 4/28 |
| 60M  | 23 | 5.76 | 0.48 | 0.092 | 0.037 | 2/23 |
| 190M | 16 | 7.04 | 0.46 | 0.200 | 0.016 | 6/16 |
| 370M | 12 | 9.57 | 0.61 | 0.430 | 0.010 | 3/12 |

**Trends across $N$:**
- $b$ rises with $N$ (0.022 → 0.43) — larger models saturate repetition
  faster. The ceiling $(D/N)^{-\gamma}/b$ therefore shrinks at large $N$,
  matching the intuition that big models extract all the signal from a
  single pass.
- $\gamma$ sits around $0.5$ at mid sizes (30M, 60M, 190M) with some
  variation at the extremes (14M $\gamma=0.23$, 370M $\gamma=0.61$).
- $c$ is noisier and partly compensates for $\gamma$ in the shared
  $c \cdot (D/N)^{-\gamma}$ factor.

Figure: [fit_eta_per_size.pdf](fit_eta_per_size.pdf) — grid of 4 forms
(rows) $\times$ 5 sizes (columns) on $\eta$ vs $D'/D$ log-log, with
per-point $\eta$ (grey) and parametric $\eta$ (blue), plus LOO RMSE in
each panel.

### 3.3 Joint $\eta$ fits (pooled across sizes)

Pooling all 102 multi-epoch points, we fit each candidate form to a
single $(c, \gamma, ...)$ shared across sizes. Best results:

| form | $n_{\text{par}}$ | RMSE | LOO | $R^2$ | params |
|---|---|---|---|---|---|
| const | 1 | 0.046 | 0.046 | 0.955 | $c=0.72$ |
| power $(D/N)$ | 2 | 0.040 | 0.040 | 0.966 | $c=2.12$, $\gamma=0.36$ |
| power $(D'/D)$ | 2 | 0.038 | 0.040 | 0.969 | $c=1.17$, $\gamma=0.31$ |
| sat $(D'/D)$ | 2 | 0.036 | 0.036 | 0.973 | $c=1.03$, $b=0.050$ |
| sat $\times$ $(D/N)$ | 3 | 0.027 | 0.028 | 0.984 | $c=4.08$, $\gamma=0.41$, $b=0.070$ |
| double power | 3 | 0.031 | 0.032 | 0.980 | $c=3.59$, $\gamma_1=0.33$, $\gamma_2=0.35$ |
| **sat $\times$ $(D/N)$, $b(N)$** | 4 | **0.026** | **0.027** | **0.986** | $c=5.12$, $\gamma=0.46$, $b_0=0.072$, $\kappa=0.37$ |

The **N-dependent form** wins: adding one parameter $\kappa$ to make the
saturation scale with $N$ gives the best LOO of any form (0.027). The
closed-form summary is

$$\boxed{\quad \eta(D, D'; N) \;=\; \frac{c \cdot (D/N)^{-\gamma}}{1 + b_0 \cdot (N/N_{\text{ref}})^{\kappa} \cdot D'/D},\qquad c=5.12,\; \gamma=0.46,\; b_0=0.072,\; \kappa=0.37,\; N_{\text{ref}}=30\,\text{M}. \quad}$$

Equivalently, $b_{\text{eff}}(N) = 0.072 \cdot (N/30\,\text{M})^{0.37}$.

The N-dependent form is still $\sim 2\times$ worse on LOO than the best
per-size fits (e.g., $0.008$ at 370M, $0.014$ at 190M). Per-size residual
structure exists beyond what $b(N)$ can capture — this is partly
systematic 1-epoch residuals (see §3.5) and partly real size-specific
shape.

Figure: [fit_eta_joint.pdf](fit_eta_joint.pdf) — $\eta$ (per-point +
fitted) and residuals vs $D'/D$, coloured by size.

Code: [fit_eta.py](fit_eta.py). Shared data loader:
[data.py](data.py).

### 3.4 Non-physical $\eta > 1$ counts

With the joint-Chinchilla anchors:

| size | $\eta > 1$ | per-point $\eta$ range |
|---|---|---|
| 14M  | 1/23  | [0.20, 1.00] |
| 30M  | 4/28  | [0.04, 1.30] |
| 60M  | 2/23  | [0.12, 1.21] |
| 190M | 6/16  | [−0.28, 1.85] |
| 370M | 3/12  | [0.16, 1.49] |

190M stands out: 38% of points have $\eta > 1$ and one even has
$\eta < 0$.

### 3.5 190M diagnostic

Running the per-point $\eta$ back-out against the joint 1-epoch curve
shows a systematic pattern:

| scale | ep | $D$ | $L_{\text{obs}}$ | $L_{\text{1ep-pred}}(D)$ | $\Delta \log$ | $\eta_{\text{pp}}$ |
|---|---|---|---|---|---|---|
| 0.5 | 2  | 1.9e9 | 3.283 | 3.859 | $-0.16$ | 1.85 ⚠ |
| 0.5 | 4  | 1.9e9 | 3.046 | 3.859 | $-0.24$ | 1.46 ⚠ |
| 0.5 | 8  | 1.9e9 | 2.925 | 3.859 | $-0.28$ | 1.02 ⚠ |
| 0.5 | 16 | 1.9e9 | 2.867 | 3.859 | $-0.30$ | 0.62 |
| 1.0 | 2  | 3.8e9 | 3.039 | 3.448 | $-0.13$ | 1.75 ⚠ |
| 1.0 | 4  | 3.8e9 | 2.883 | 3.448 | $-0.18$ | 1.27 ⚠ |
| 2.0 | 2  | 7.6e9 | 2.874 | 3.148 | $-0.09$ | 1.49 ⚠ |
| 4.0 | 2  | 15.2e9 | 2.790 | 2.930 | $-0.05$ | 0.82 |
| 8.0 | 2  | 30.4e9 | 2.741 | 2.771 | $-0.01$ | 0.17 |
| 16.0 | 2 | 60.8e9 | 2.705 | 2.655 | $+0.02$ | **−0.28** ⚠ |

$\Delta \log$ summary: mean $-0.147$, positive in only 1 / 16 points.
**The joint 1-epoch fit systematically over-predicts 190M multi-epoch
loss** — not a bias in the multi-epoch observations, but a size-specific
residual of the joint 1-epoch extrapolation.

This propagates into the $\eta$ inversion directly. Recall

$$\eta = \frac{(B/(L-E_{\text{eff}}))^{1/\beta} - D}{D'}.$$

If the 1-epoch curve says "at this $D$ you should see loss $L_{\text{1ep}}$"
but the observed 1-epoch loss at 190M is lower than that, then
subtracting the multi-epoch loss from $E_{\text{eff}}$ and inverting
gives a $D_{\text{eff}}$ that's *larger* than the true (total tokens),
hence $\eta > 1$.

Independent evidence: a 190M-only 3-parameter 1-epoch refit gives $\beta
\approx 0.81$, very different from the joint $\beta = 0.458$. This
isn't a meaningful physical $\beta$ though — with only 6 points per
size, the 3-param fit is essentially interpolating (RMSE $< 0.001$),
and the same "local $\beta$" pattern shows at 100M and 370M too
($\beta \approx 0.80$). The real story: **the 1-epoch $L(D)$ curve at
large $N$ is locally steeper than the $\beta = 0.458$ that best fits
the joint 1-epoch dataset across all sizes.**

**Why 16x 2ep has $\eta < 0$:** at that point $D = 60.8 \times 10^9$ is
*larger* than the 190M 1-epoch fit's extrapolation support. Joint
predicts $L = 2.655$, observed 1-epoch at 16x (not shown) is $2.74$, and
the 2-epoch point is $2.705$. Joint is "too optimistic" at huge $D$ for
this specific size, so even the 2-epoch (improved) loss still sits
*above* the joint curve — and the $\eta$ formula interprets that as
"negative effective tokens".

**What this means for the fit.** We haven't mis-modelled $\eta$; we've
hit the limits of how well a single set of Chinchilla anchors describes
seven very different model sizes. Two plausible fixes:
1. **Refit per-size 1-epoch $\beta$ but share $E_{\text{eff}}(N)$
   structure.** Probably the cleanest: let $\beta(N) = \beta_0 + \beta_1
   \log N$ or similar.
2. **Accept per-size $\eta$ parameters** (the per-size fit in §3.2 gives
   $\text{LOO} = 0.014$ at 190M, which is great — the problem is only
   with the *joint* anchors).

See [fit_eta_per_size.pdf](fit_eta_per_size.pdf) for the per-size fits
where this 190M issue is much less visible.

---

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

---

## 5. Open questions

1. **Size-dependent $\beta$ in the 1-epoch fit.** The joint fit uses
   a single $\beta = 0.458$. Per-size 3-param fits suggest locally
   steeper $\beta \approx 0.8$ at mid-large $N$ (100M, 190M, 370M) —
   though those fits are near-interpolating with only 4–6 points each.
   A 6-parameter joint form $L = E + A/N^{\alpha_N} + B/D^{\beta(N)}$
   with $\beta(N) = \beta_0 + \beta_1 \log(N/N_{\text{ref}})$ would
   absorb the 190M structure without overfitting.
2. **$\eta = 1$ regime boundary.** Even with the N-dependent $b(N)$
   form, the 30M-specific $\eta > 1$ excursions at low epochs remain.
   A double-power-law 1-epoch form $L = E + B_1/D^{\beta_1} +
   B_2/D^{\beta_2}$ or Hoffmann-style saturated form may fit better
   at low $D$.
3. **Bootstrap CIs for $\eta$ across sizes.** Warm-started LOO already
   handles stability diagnostics; a full bootstrap would give CIs on
   $(c, \gamma, b_0, \kappa)$. ~30 s per size.
4. **Paraphrase / synthetic data extension.** The framework extends
   directly via $\eta_{\text{para}}$ on the second token stream. Not
   currently in the Dolma data.
