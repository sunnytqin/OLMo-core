# Response: joint fit, backbone drift, and η_para uncertainty

> *"Could you report the joint (non-staged) fit and quantify how far the
> stage-1 backbone fit drifts when paraphrase data is included, with
> uncertainty on the η_para parameters?"*

We agree this is the right check, and we have run it. All numbers below are
on a single consistent corpus (n = 370: 56 one-epoch + 197 repetition + 117
paraphrase points) using the same residual-drop policy (k = 15) for both
pipelines, so that the *only* difference between them is whether the backbone
is frozen or fit jointly.

## 1. The joint (non-staged) fit

We fit all 11 parameters simultaneously on the pooled one-epoch + repetition
+ paraphrase data. The backbone is no longer frozen; this is now the headline
fit. Its parameters are:

| block | parameters |
|---|---|
| Chinchilla backbone | E = 1.41, A = 236, B = 17 900, α = 0.292, β = 0.438 |
| η_rep | log K = 11.02, ρ = −0.42, σ = −0.42 |
| η_para | log K = 27.07, ρ = −1.31, σ = −1.16 |

## 2. How far the backbone drifts — and why the raw drift is misleading

Comparing the stage-1 backbone (fit on one-epoch + repetition only, no
paraphrase) against the joint fit above, the raw Chinchilla parameters move
substantially, most visibly a +236% change in A:

| param | stage-1 (no paraphrase) | joint (+paraphrase) | rel. change |
|---|---|---|---|
| E | 0.94 | 1.41 | +50% |
| A | 70.3 | 236 | +236% |
| α | 0.204 | 0.292 | +43% |
| β | 0.441 | 0.438 | −0.6% |
| B | 18 390 | 17 900 | −2.6% |

This drift is misleading, because E, A and α are not individually
identifiable over our range of model sizes — only the combination they form
is. That combination is the quantity E_eff(N). The backbone predicts the
one-epoch loss of a model of size N in the large-data limit, where the data
term B/D^β vanishes as D grows, leaving the size-dependent loss floor

    E_eff(N) = E + A / N^α.

E_eff(N) is the physically meaningful part of the backbone: it is the loss a
model of size N reaches with unlimited one-epoch data, and E, A and α exist
only to shape this one curve. The remaining two backbone parameters, B and β,
describe how loss *approaches* that floor as data grows; the table above shows
they are essentially unchanged (Δβ = −0.6%, ΔB = −2.6%).

The raw parameters drift while E_eff(N) does not because our model sizes span
only 14M–600M, about 1.6 decades of N. Over such a limited range many
different (E, A, α) triples trace almost the same curve E + A/N^α: a larger
constant E trades against a smaller, more steeply decaying A/N^α term with
negligible change at any N we actually observe. The three parameters are
therefore jointly identified but individually degenerate, and adding
paraphrase data simply moves the optimizer to a different point along this
near-degenerate valley — producing large swings in the raw numbers while
leaving the prediction intact.

The correct way to quantify drift is therefore to evaluate E_eff(N) at the
model sizes we have, rather than to compare E, A and α in isolation. Doing so
shows the backbone prediction moves by at most 2.8% at any size when
paraphrase data is added:

| N | E_eff (stage-1) | E_eff (joint) | drift |
|---|---|---|---|
| 14M | 3.368 | 3.335 | −0.97% |
| 30M | 3.017 | 2.950 | −2.23% |
| 60M | 2.742 | 2.667 | −2.75% |
| 100M | 2.563 | 2.492 | −2.78% |
| 190M | 2.363 | 2.306 | −2.42% |
| 370M | 2.181 | 2.146 | −1.60% |
| 600M | 2.064 | 2.049 | −0.74% |

The joint fit thus does not drift the backbone in any way that changes what it
predicts for one-epoch training. The apparent drift lives entirely in the
unidentified (E, A, α) reparameterization and cancels out of every observable
quantity.

## 3. η_para from both pipelines, with 95% confidence intervals

To test the concern directly — does freezing the backbone distort η_para? —
we compare η_para from the two pipelines on the identical kept-set, with
95% CIs from a parametric bootstrap (B = 200; the frozen pipeline resamples
the paraphrase rows and refits η_para with the backbone held at its stage-1
value; the joint pipeline resamples all rows and refits all 11 parameters):

| parameter | staged (backbone frozen) | joint (non-staged) |
|---|---|---|
| log K_para | +30.4  [+25.7, +36.6] | +27.1  [+26.6, +27.5] |
| ρ_para | −1.36  [−1.63, −1.12] | −1.31  [−1.46, −1.14] |
| σ_para | −1.32  [−1.61, −1.11] | −1.16  [−1.20, −1.13] |

**The two pipelines agree.** Both σ_para intervals are firmly negative and
overlap; ρ_para is nearly identical; the log K_para intervals overlap. The
staged intervals are wider only because that pipeline uses just the 117
paraphrase rows to constrain η_para, whereas the joint fit uses all 355 kept
points. Freezing the backbone therefore does **not** relocate misfit into
η_para: fitting η_para against a frozen backbone recovers the same values
(within uncertainty) as fitting everything jointly.

## 4. Correction to an earlier draft

An earlier version of this section reported the staged pipeline giving
σ_para ≈ +0.2 (paraphrase behaving as fresh data), in apparent conflict with
the joint fit's σ_para < 0. We have traced that discrepancy: the earlier
staged fit froze the backbone to an *outdated* stage-1 solution (from before
the additional paraphrase runs were added). When the backbone is frozen to
the current stage-1 fit, the staged pipeline gives σ_para = −1.32, in
agreement with the joint fit. The earlier σ_para > 0 was an artifact of a
stale frozen backbone, not a genuine property of the freezing procedure, and
we have removed that claim.

## Summary

- The joint (non-staged) fit is now the headline; the backbone is not frozen.
- The backbone's raw (E, A, α) reparameterize when paraphrase is added, but
  its actual prediction E_eff(N) is stable to ≤ 2.8% at every model size, and
  β and B are unchanged.
- η_para is the same, within bootstrapped 95% CIs, whether the backbone is
  frozen or fit jointly — so freezing does not distort the paraphrase
  saturation parameters. σ_para < 0 (paraphrase value saturates with scale)
  is robust to the choice of pipeline.

*Reproduction:* `fit_pipeline_compare.py --k 15 --B 200`;
results in `_onego_json/pipeline_compare_n370_k15.json`.
