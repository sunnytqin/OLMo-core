# Chinchilla Scaling-Law Fits on Dolma-30M

Multi-epoch scaling-law fit for a 30M-parameter model on Dolma data.
Model size is fixed, so the classic Chinchilla reduces to $L = E + B/D^{\beta}$,
where $D$ is training tokens. We additionally model multi-epoch training as

$$L = E + \frac{B}{(D + \eta \cdot D')^{\beta}}, \qquad D' = (\text{epochs}-1) \cdot D,$$

and fit $\eta$, the effective-token multiplier for repeated data.

**Current best fit (Dolma-30M, scales $\ge 0.5\times$, $\delta=0.1$ on $\log L$):**

| stage | params | LOO RMSE |
|---|---|---|
| 1-epoch | $E=3.01$, $B=47022$, $\beta=0.486$ | 0.071 |
| $\eta$ (`sat × (D/N)`) | $c=6.66$, $\gamma=0.48$, $b=0.10$ | **0.034** |

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

## 3. Dolma-30M data summary

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

## 4. Open questions

1. **Joint fit across model sizes.** The same pipeline applied to 60M /
   190M / 370M should reveal whether $(\gamma, b)$ transfer across $N$,
   and whether the $(D/N)^{-\gamma}$ dependence is the right encoding of
   the $N$-dimension.
2. **$\eta = 1$ regime boundary.** The 3 remaining $\eta > 1$ points
   suggest the Chinchilla $L = E + B/D^{\beta}$ form itself starts
   breaking near the 1-epoch data range. Worth checking whether a
   double-power-law $L = E + B_1/D^{\beta_1} + B_2/D^{\beta_2}$ or a
   Hoffmann-style saturated form fits the 1-epoch data better — this
   would obviate the need to argue about $\delta$.
3. **Bootstrap CIs for $\eta$ params.** Easy to add once we're happy
   with the chosen form; 21 points $\times$ 3 params, warm-started,
   $\sim 30\,\text{s}$ total.
4. **Paraphrase / synthetic data extension.** Once $\eta$ for pure
   repeat is nailed down, the same framework handles paraphrase via an
   $\eta_{\text{para}}$ on the second token stream. Not in current
   Dolma-30M data.
