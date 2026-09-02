#!/usr/bin/env python3
"""
Practical implications of the §6 headline triple scaling-law fit.

Uses one-go triple fit (1-epoch + repetition + paraphrase, k=15):
  Chinchilla:  E=1.34  A=187  B=17375  α=0.276  β=0.437
  Repetition:  log R* = 10.96 − 0.42·log(D/N) − 0.42·log(N)
  Paraphrase:  log R* = 31.02 − 1.54·log(D/N) − 1.33·log(N)

Answers three practitioner questions:
  Q1. Multi-epoch: how many epochs to run, and what is the fresh-data equiv?
  Q2. Paraphrase: how many paraphrase passes (K) to generate, and fresh-data equiv?
  Q3. When to prefer paraphrase over multi-epoch (depends on N and scale)?

Run:
  python practical_implications.py
"""

import math
import numpy as np

# ─── §6 headline triple fit (one-go, k=15) ───────────────────────────────────
E, A, B, alpha, beta = 1.34, 187.0, 17_375.0, 0.276, 0.437

# η form: exp-sat (Muennighoff Eq 5) with R*(D,N)
#   η · D'/D = R*(1 − exp(−x/R*)),   x = D'/D
#   log R* = log_K + ρ·log(D/N) + σ·log(N)      (natural logs)

LOG_K_REP,  RHO_REP,  SIGMA_REP  = 10.96, -0.42, -0.42
LOG_K_PARA, RHO_PARA, SIGMA_PARA = 31.02, -1.54, -1.33

# Paraphrase produces ≈ 0.322·D extra tokens per generation pass (K=1)
# (empirically: K=32 gives D'/D ≈ 10.3; 10.3/32 ≈ 0.322)
TOKENS_PER_PARA = 0.322

TTP = 20  # Chinchilla tokens-to-params ratio: D_chinchilla = scale · TTP · N

# Bootstrap parameter vectors (B=200) from §6.6 of writeup.md — used to put
# confidence bands on R*.  Each sample is a full 11-parameter fit on a
# resampled kept set; propagating them through R_star() gives correlation-aware
# CIs rather than naive marginal-parameter intervals.
BOOTSTRAP_JSON = "_onego_json/bootstrap_all_B200_seed42.json"


# ─── Core functions ──────────────────────────────────────────────────────────

def R_star(log_K: float, rho: float, sigma: float, D_over_N: float, N: float) -> float:
    return math.exp(log_K + rho * math.log(D_over_N) + sigma * math.log(N))


def eta_expsat(x: float, Rstar: float) -> float:
    """η(x, R*) for the exp-sat form; x = D'/D."""
    if x <= 0:
        return 1.0
    return Rstar * (1.0 - math.exp(-x / Rstar)) / x


def D_eff_rep(D: float, N: float, epochs: float) -> float:
    """Effective fresh-token count for multi-epoch training."""
    Dp = (epochs - 1.0) * D
    x  = epochs - 1.0
    Rs = R_star(LOG_K_REP, RHO_REP, SIGMA_REP, D / N, N)
    return D + eta_expsat(x, Rs) * Dp


def D_eff_para(D: float, N: float, K: float) -> float:
    """Effective fresh-token count for K paraphrase passes."""
    Dp = K * TOKENS_PER_PARA * D
    x  = K * TOKENS_PER_PARA          # = D'/D
    Rs = R_star(LOG_K_PARA, RHO_PARA, SIGMA_PARA, D / N, N)
    return D + eta_expsat(x, Rs) * Dp


def loss_from_D_eff(N: float, D_eff: float) -> float:
    return (E + A / N**alpha) + B / D_eff**beta


def epochs_at_budget_fraction(f: float, Rstar: float) -> float:
    """Epochs needed to consume fraction f of the repetition budget R*."""
    x = -Rstar * math.log(1.0 - f)   # solve 1 − exp(−x/R*) = f
    return 1.0 + x


def K_at_budget_fraction(f: float, Rstar: float) -> float:
    """Paraphrase passes K needed to consume fraction f of the para budget R*."""
    x = -Rstar * math.log(1.0 - f)
    return x / TOKENS_PER_PARA


# ─── Print helpers ───────────────────────────────────────────────────────────

def fmt_N(N: float) -> str:
    if N >= 1e9:
        return f"{N/1e9:.0f}B"
    return f"{N/1e6:.0f}M"


def fmt_tokens(D: float) -> str:
    if D >= 1e12:
        return f"{D/1e12:.1f}T"
    if D >= 1e9:
        return f"{D/1e9:.1f}B"
    return f"{D/1e6:.0f}M"


def hr(char="─", width=100):
    print(char * width)


# ─── Model sizes and scales ───────────────────────────────────────────────────
# Fit range: 14M–600M.  1B+ are extrapolations (flagged in output).
SIZES = [30e6, 60e6, 190e6, 370e6, 600e6, 1e9, 3e9, 7e9]
FIT_MAX_N = 600e6  # largest size in the fit

SCALES = [0.5, 1.0, 2.0, 4.0]
SCALE_LABELS = {0.5: "0.5× (D/N=10)", 1.0: "1× (Chinchilla)", 2.0: "2× overtr.", 4.0: "4× overtr."}

EPOCH_GRID   = [2, 4, 8, 16, 32, 64]
K_GRID       = [1, 2, 4, 8, 16, 32]
BUDGET_FRACS = [0.5, 0.75, 0.9]


# ═══════════════════════════════════════════════════════════════════════════════
# Q1 — MULTI-EPOCH: recommended epoch counts and fresh-data equivalents
# ═══════════════════════════════════════════════════════════════════════════════

def q1_multi_epoch():
    hr("═")
    print("Q1 — MULTI-EPOCH: epoch recommendations")
    hr("═")
    print()
    print("Notation:")
    print("  R*         = saturation budget: limit of (D_eff − D)/D as epochs → ∞")
    print("  Ceiling    = 1 + R* = limit of D_eff/D as epochs → ∞")
    print("  Useful ep  = Ep@90% = 1 + ln(10)·R* ≈ 1 + 2.3·R*  (solve 1−exp(−x/R*)=0.9 → x=ln(10)·R*)")
    print("  D_eff/D    at useful ep = 1 + 0.9·R*  (what you realise at that cutoff)")
    print()
    print("  For each (model size, D/N) pair the table shows R*, the useful-epoch")
    print("  cutoff, the realised D_eff/D at that cutoff, and the ceiling.")
    print()

    SIZES_Q1 = [30e6, 60e6, 190e6, 370e6, 600e6, 1e9, 3e9]

    for scale in SCALES:
        D_over_N = scale * TTP
        hr()
        print(f"D/N = {D_over_N:.0f}  ({scale}× Chinchilla)")
        hr("-", 72)
        print(f"{'Model':>7}  {'D':>8}  {'R*':>5}  {'Useful ep':>10}  {'D_eff/D':>10}  {'Ceiling':>9}")
        print(f"{'':>7}  {'':>8}  {'':>5}  {'(=1+2.3R*)':>10}  {'@ useful ep':>10}  {'(=1+R*)':>9}")
        hr("-", 72)
        for N in SIZES_Q1:
            D      = D_over_N * N
            Rs     = R_star(LOG_K_REP, RHO_REP, SIGMA_REP, D_over_N, N)
            ep90   = epochs_at_budget_fraction(0.9, Rs)
            deff90 = 1.0 + 0.9 * Rs
            ceil_  = 1.0 + Rs
            extrap = "*" if N > FIT_MAX_N else " "
            print(f"{fmt_N(N):>6}{extrap}  {fmt_tokens(D):>8}  {Rs:>5.1f}  "
                  f"{ep90:>10.1f}  {deff90:>10.1f}×  {ceil_:>9.1f}×")
        print()

    print("* = extrapolated beyond fit range (≤ 600M)")
    print()

    # ── Budget fraction at common epoch counts ────────────────────────────────
    hr()
    print("Budget fraction consumed at common epoch counts  (D/N = 20, i.e. 1× Chinchilla)")
    hr("-", 100)
    col_header = f"{'Model':>7}  {'R*':>5}"
    for e in EPOCH_GRID:
        col_header += f"  {str(e)+'ep':>11}"
    print(col_header)
    hr("-", 100)
    for N in SIZES_Q1:
        D      = TTP * N
        Rs     = R_star(LOG_K_REP, RHO_REP, SIGMA_REP, TTP, N)
        extrap = "*" if N > FIT_MAX_N else " "
        row    = f"{fmt_N(N):>6}{extrap}  {Rs:>5.1f}"
        for e in EPOCH_GRID:
            x    = float(e - 1)
            frac = 1.0 - math.exp(-x / Rs)
            deff = 1.0 + frac * Rs
            row += f"  {frac*100:>4.0f}%({deff:.1f}×)"
        print(row)
    print()
    print("  frac% = fraction of R* budget consumed  |  Deff/D in parentheses")
    print()

    # ── Same table but fixed N=1B, varying D/N ───────────────────────────────
    hr()
    print("Budget fraction consumed at common epoch counts  (fixed N = 1B*, varying D/N)")
    hr("-", 100)
    col_header = f"{'D/N':>6}  {'D':>8}  {'R*':>5}"
    for e in EPOCH_GRID:
        col_header += f"  {str(e)+'ep':>11}"
    print(col_header)
    hr("-", 100)
    N_fixed = 1e9
    for D_over_N in [5, 10, 20, 40, 80, 160]:
        D   = D_over_N * N_fixed
        Rs  = R_star(LOG_K_REP, RHO_REP, SIGMA_REP, D_over_N, N_fixed)
        row = f"{D_over_N:>6}  {fmt_tokens(D):>8}  {Rs:>5.1f}"
        for e in EPOCH_GRID:
            x    = float(e - 1)
            frac = 1.0 - math.exp(-x / Rs)
            deff = 1.0 + frac * Rs
            row += f"  {frac*100:>4.0f}%({deff:.1f}×)"
        print(row)
    print()
    print("  * 1B is extrapolated beyond the 30M–600M fit range")
    print("  frac% = fraction of R* budget consumed  |  Deff/D in parentheses")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 — PARAPHRASE: recommended K and fresh-data equivalents
# ═══════════════════════════════════════════════════════════════════════════════

def q2_paraphrase():
    hr("═")
    print("Q2 — PARAPHRASE: K-pass recommendations and fresh-data equivalents")
    hr("═")
    print()
    print("Notation:")
    print("  R*         = saturation budget: limit of (D_eff − D)/D as K → ∞")
    print("  Ceiling    = 1 + R* = limit of D_eff/D as K → ∞")
    print("  Useful K   = K@90% = ln(10)·R*/0.322 ≈ 7.15·R*  (each pass adds 0.322·D tokens;")
    print("               solve 1−exp(−K·0.322/R*)=0.9 → K=ln(10)·R*/0.322)")
    print("  D_eff/D    at useful K = 1 + 0.9·R*  (what you realise at that cutoff)")
    print()
    print("  For each (model size, D/N) pair the table shows R*, the useful-K")
    print("  cutoff, the realised D_eff/D at that cutoff, and the ceiling.")
    print()

    SIZES_Q2 = [30e6, 60e6, 190e6, 370e6, 600e6, 1e9, 3e9]

    for scale in SCALES:
        D_over_N = scale * TTP
        hr()
        print(f"D/N = {D_over_N:.0f}  ({scale}× Chinchilla)")
        hr("-", 72)
        print(f"{'Model':>7}  {'D':>8}  {'R*':>5}  {'Useful K':>10}  {'D_eff/D':>10}  {'Ceiling':>9}")
        print(f"{'':>7}  {'':>8}  {'':>5}  {'(=7.15R*)':>10}  {'@ useful K':>10}  {'(=1+R*)':>9}")
        hr("-", 72)
        for N in SIZES_Q2:
            D      = D_over_N * N
            Rs     = R_star(LOG_K_PARA, RHO_PARA, SIGMA_PARA, D_over_N, N)
            K90    = K_at_budget_fraction(0.9, Rs)
            deff90 = 1.0 + 0.9 * Rs
            ceil_  = 1.0 + Rs
            extrap = "*" if N > FIT_MAX_N else " "
            print(f"{fmt_N(N):>6}{extrap}  {fmt_tokens(D):>8}  {Rs:>5.1f}  "
                  f"{K90:>10.1f}  {deff90:>10.1f}×  {ceil_:>9.1f}×")
        print()

    print("* = extrapolated beyond fit range (≤ 600M)")
    print()

    # ── Budget fraction at common K values ────────────────────────────────────
    hr()
    print("Budget fraction consumed at common K values  (D/N = 20, i.e. 1× Chinchilla)")
    hr("-", 100)
    col_header = f"{'Model':>7}  {'R*':>5}"
    for k in K_GRID:
        col_header += f"  {'K='+str(k):>11}"
    print(col_header)
    hr("-", 100)
    for N in SIZES_Q2:
        D      = TTP * N
        Rs     = R_star(LOG_K_PARA, RHO_PARA, SIGMA_PARA, TTP, N)
        extrap = "*" if N > FIT_MAX_N else " "
        row    = f"{fmt_N(N):>6}{extrap}  {Rs:>5.1f}"
        for k in K_GRID:
            x    = k * TOKENS_PER_PARA
            frac = 1.0 - math.exp(-x / Rs)
            deff = 1.0 + frac * Rs
            row += f"  {frac*100:>4.0f}%({deff:.1f}×)"
        print(row)
    print()
    print("  frac% = fraction of R* budget consumed  |  Deff/D in parentheses")
    print()

    # ── Same table but fixed N=1B, varying D/N ───────────────────────────────
    hr()
    print("Budget fraction consumed at common K values  (fixed N = 1B*, varying D/N)")
    hr("-", 100)
    col_header = f"{'D/N':>6}  {'D':>8}  {'R*':>5}"
    for k in K_GRID:
        col_header += f"  {'K='+str(k):>11}"
    print(col_header)
    hr("-", 100)
    N_fixed = 1e9
    for D_over_N in [5, 10, 20, 40, 80, 160]:
        D   = D_over_N * N_fixed
        Rs  = R_star(LOG_K_PARA, RHO_PARA, SIGMA_PARA, D_over_N, N_fixed)
        row = f"{D_over_N:>6}  {fmt_tokens(D):>8}  {Rs:>5.1f}"
        for k in K_GRID:
            x    = k * TOKENS_PER_PARA
            frac = 1.0 - math.exp(-x / Rs)
            deff = 1.0 + frac * Rs
            row += f"  {frac*100:>4.0f}%({deff:.1f}×)"
        print(row)
    print()
    print("  * 1B is extrapolated beyond the 30M–600M fit range")
    print("  frac% = fraction of R* budget consumed  |  Deff/D in parentheses")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — PARAPHRASE vs. MULTI-EPOCH: which to prefer and by how much?
# ═══════════════════════════════════════════════════════════════════════════════

def crossover_N(scale: float) -> float:
    """Model size N where R*_para = R*_rep at the given chinchilla scale."""
    D_over_N = scale * TTP
    # log R*_para = log R*_rep
    # ΔlogK + Δρ·log(D/N) + Δσ·log(N) = 0
    delta_logK = LOG_K_PARA - LOG_K_REP            # 20.06
    delta_rho  = RHO_PARA   - RHO_REP              # -1.12
    delta_sig  = SIGMA_PARA  - SIGMA_REP            # -0.91
    # delta_logK + delta_rho·log(D/N) + delta_sig·log(N) = 0
    log_N = -(delta_logK + delta_rho * math.log(D_over_N)) / delta_sig
    return math.exp(log_N)


def q3_comparison():
    hr("═")
    print("Q3 — PARAPHRASE vs. MULTI-EPOCH: when to prefer which?")
    hr("═")
    print()
    print(
        "Key insight: at the same extra-token budget D', paraphrase vs. multi-epoch\n"
        "gives a larger D_effective exactly when R*_para > R*_rep  (larger saturation\n"
        "budget → higher η → more effective tokens per repeated/paraphrased token).\n"
        "\n"
        "Crossover N where R*_para = R*_rep  (below → para better; above → rep better):"
    )
    hr("-", 60)
    print(f"  {'Scale':>20}   Crossover N")
    hr("-", 60)
    for scale in SCALES:
        N_co = crossover_N(scale)
        print(f"  {SCALE_LABELS[scale]:>20}   {fmt_N(N_co)}")
    print()
    N_co_05 = crossover_N(0.5)
    N_co_1  = crossover_N(1.0)
    N_co_4  = crossover_N(4.0)
    print(
        f"Interpretation:\n"
        f"  • At 1× scale: paraphrase is preferred below ~{fmt_N(N_co_1)} params, multi-epoch above.\n"
        f"  • More overtraining (higher scale) → crossover shifts to smaller N:\n"
        f"    at 4× scale the crossover is ~{fmt_N(N_co_4)}, so most practically-sized models\n"
        f"    should prefer multi-epoch when heavily overtrained.\n"
        f"  • With limited fresh data (0.5× scale) → crossover shifts to ~{fmt_N(N_co_05)};\n"
        f"    models below that size benefit from paraphrase even in the data-scarce regime.\n"
    )

    # Quantitative advantage at 1× scale
    hr()
    print("Ratio R*_para / R*_rep at 1× (Chinchilla) scale")
    hr("-", 80)
    print(f"{'Model':>6}  {'R*_rep':>8}  {'R*_para':>8}  {'ratio':>7}  {'winner':>12}  {'advantage'}  ")
    hr("-", 80)
    for N in SIZES:
        D_over_N = TTP
        Rs_rep  = R_star(LOG_K_REP,  RHO_REP,  SIGMA_REP,  D_over_N, N)
        Rs_para = R_star(LOG_K_PARA, RHO_PARA, SIGMA_PARA, D_over_N, N)
        ratio   = Rs_para / Rs_rep
        winner  = "paraphrase" if ratio > 1 else "multi-epoch"
        extrap  = "*" if N > FIT_MAX_N else " "
        print(f"{fmt_N(N):>5}{extrap}  {Rs_rep:>8.2f}  {Rs_para:>8.2f}  {ratio:>7.2f}  {winner:>12}  "
              f"(D_eff ceiling {max(ratio,1/ratio):.1f}× more budget)")
    print()

    # Full comparison table: for a fixed D' budget, how much D_eff do you get?
    hr()
    print("D_effective / D for the SAME extra-token budget D'/D — 1× scale")
    print("(rows = model size; cols = extra token ratio D'/D = epochs−1 = K×0.322)")
    hr("-", 100)
    extras = [1.0, 2.0, 4.0, 8.0, 16.0]   # D'/D
    K_equiv = [x / TOKENS_PER_PARA for x in extras]

    header = f"{'Model':>6}  {'D_fresh':>8}"
    for x in extras:
        header += f"  {'D/D='+str(int(x)):>15}"
    print(header)
    header2 = f"{'':>6}  {'(1ep)':>6}"
    for k in K_equiv:
        header2 += f"  {'rep / para':>15}"
    print(header2)
    hr("-", 100)

    for N in SIZES:
        D       = TTP * N
        D_over_N = TTP
        extrap  = "*" if N > FIT_MAX_N else " "
        row = f"{fmt_N(N):>5}{extrap}  {'1.00':>6}"
        for x, k in zip(extras, K_equiv):
            Dp      = x * D
            # multi-epoch: epochs = 1 + x
            Rs_r    = R_star(LOG_K_REP, RHO_REP, SIGMA_REP, D_over_N, N)
            Def_r   = D + eta_expsat(x, Rs_r) * Dp
            # paraphrase: K = x / TOKENS_PER_PARA
            Rs_p    = R_star(LOG_K_PARA, RHO_PARA, SIGMA_PARA, D_over_N, N)
            Def_p   = D + eta_expsat(x, Rs_p) * Dp
            row += f"  {Def_r/D:.2f}x / {Def_p/D:.2f}x"
        print(row)
    print()
    print("Format: rep_Deff/D / para_Deff/D.  Bold the larger number to see the winner.")
    print()

    # Practical decision table — compute crossovers at all scales
    N_crossovers = {s: crossover_N(s) for s in SCALES}

    hr()
    print("Practical decision guide")
    hr("-", 90)
    print(f"  {'Model size':>12}  |  {'Chinchilla scale':>18}  |  Recommendation")
    hr("-", 90)
    for scale in SCALES:
        N_co = N_crossovers[scale]
        slab = SCALE_LABELS[scale]
        print(f"  {'< '+fmt_N(N_co):>12}  |  {slab:>18}  |  Paraphrase  (R*_para > R*_rep)")
        print(f"  {'≈ '+fmt_N(N_co):>12}  |  {slab:>18}  |  About equal; either works")
        print(f"  {'> '+fmt_N(N_co):>12}  |  {slab:>18}  |  Multi-epoch (R*_rep  > R*_para)")
        hr("-", 90)
    print(
        "\n  Note: 'recommendation' refers to D_eff per extra training token, not wall-clock.\n"
        "  Paraphrase also has a data-generation cost (LLM inference); weigh that\n"
        "  against the D_eff gain vs. simply running more epochs on the same data.\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Bonus: loss predictions at a glance
# ═══════════════════════════════════════════════════════════════════════════════

def bonus_loss_table():
    hr("═")
    print("BONUS — predicted val loss at 1× scale for different strategies")
    hr("═")
    print()

    STRATEGIES = [
        ("1ep (baseline)",  lambda N, D: D),
        ("ep×2",            lambda N, D: D_eff_rep(D, N, 2)),
        ("ep×4",            lambda N, D: D_eff_rep(D, N, 4)),
        ("ep×8",            lambda N, D: D_eff_rep(D, N, 8)),
        ("ep×16",           lambda N, D: D_eff_rep(D, N, 16)),
        ("K=4 para",        lambda N, D: D_eff_para(D, N, 4)),
        ("K=8 para",        lambda N, D: D_eff_para(D, N, 8)),
        ("K=16 para",       lambda N, D: D_eff_para(D, N, 16)),
        ("K=32 para",       lambda N, D: D_eff_para(D, N, 32)),
    ]

    hr("-", 110)
    header = f"{'Model':>6}  {'L(1ep)':>8}"
    for name, _ in STRATEGIES[1:]:
        header += f"  {name:>10}"
    print(header)
    hr("-", 110)

    for N in SIZES:
        D      = TTP * N
        L1ep   = loss_from_D_eff(N, D)
        extrap = "*" if N > FIT_MAX_N else " "
        row = f"{fmt_N(N):>5}{extrap}  {L1ep:>8.4f}"
        for name, fn in STRATEGIES[1:]:
            Def = fn(N, D)
            L   = loss_from_D_eff(N, Def)
            row += f"  {L:>10.4f}"
        print(row)

    print()
    print("* = extrapolated beyond fit range (≤600M)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figures(save_dir="."):
    """
    Two stand-alone figures designed to stack vertically and align column-wise:

      heatmap_rstar.pdf  (TOP)    : R* — Repetition | Paraphrase  (+ boundary legend)
      heatmap_useful.pdf (BOTTOM) : Recommended epochs | Recommended K passes

    Both share an identical *horizontal* layout (panel + colorbar x-positions),
    a single muted colormap (cividis) and the y-axis (TTP).  The top figure hides
    its x-axis (model-size) ticks/label so the bottom figure carries the shared
    x-axis when the two are placed one above the other.

    X-axis = model size N  (discrete columns),  Y-axis = D/N  (discrete rows).

    NOTE: we deliberately do NOT use bbox_inches='tight' — that crops each figure
    to its own content and would break column alignment between the two files.
    Both figures use the same width and the same fractional column positions, so
    scaling each to the same width (e.g. \\linewidth) keeps the columns aligned.
    """
    import glob
    import os
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import font_manager
    import seaborn as sns  # registers the 'mako' / 'rocket' colormaps

    # Match paper font (Palatino Linotype, same as isoloss_contour.py)
    _font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'fonts')
    for _f in glob.glob(os.path.join(_font_dir, 'palatinolinotype_*.ttf')):
        font_manager.fontManager.addfont(_f)
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Palatino Linotype', 'P052', 'Palatino', 'serif'],
        'mathtext.fontset': 'cm',
        'font.size':        13,
    })

    # The two stacked figures share one colormap (YlGnBu) split in half, so the
    # pair reads as a single coherent, low-contrast family rather than two
    # competing palettes:
    #   R* (top)     -> darker half (green -> deep blue), white annotations.
    #   useful (bot) -> lighter half (pale yellow -> green), black annotations.
    # Each half is internally consistent enough that a single fixed text color
    # stays legible across all its cells (no per-cell text switching, which could
    # read as a spurious signal).
    def _trunc(cmap, lo, hi):
        return mcolors.LinearSegmentedColormap.from_list(
            cmap.name + "_t", cmap(np.linspace(lo, hi, 256)))

    _ylgnbu = plt.get_cmap("YlGnBu")
    CMAP_R = _trunc(_ylgnbu, 0.50, 1.00)   # darker half  (top)    + white text
    CMAP_U = _trunc(_ylgnbu, 0.00, 0.50)   # lighter half (bottom) + black text

    GRID_N   = [30e6, 60e6, 190e6, 370e6, 600e6, 1e9, 3e9, 7e9]
    GRID_DoN = [3, 5, 10, 20, 40, 80, 160]
    N_labels   = [fmt_N(N) for N in GRID_N]
    DoN_labels = [str(int(d)) for d in GRID_DoN]

    nN, nDoN = len(GRID_N), len(GRID_DoN)

    # rows = D/N index, cols = N index; origin='lower' → row 0 at bottom of plot
    R_rep_grid  = np.array([[R_star(LOG_K_REP,  RHO_REP,  SIGMA_REP,  d, N)
                              for N in GRID_N] for d in GRID_DoN])
    R_para_grid = np.array([[R_star(LOG_K_PARA, RHO_PARA, SIGMA_PARA, d, N)
                              for N in GRID_N] for d in GRID_DoN])

    Ep_grid = 1.0 + math.log(10) * R_rep_grid
    K_grid  = math.log(10) * R_para_grid / TOKENS_PER_PARA

    # ── Annotation formatters ─────────────────────────────────────────────────
    def fmt_R(v):
        if v >= 10:  return f"{round(v)}"
        if v >= 1:   return f"{v:.1f}"
        return f"{v:.2f}"

    def fmt_ep(v):
        r = round(v)
        return f"{r}" if r >= 2 else "~1"

    def fmt_K(v):
        if v < 0.5:  return "<1"
        return f"{round(v)}"

    # ── Shared horizontal geometry (figure fractions, width-relative) ─────────
    # Identical in both figures so columns line up when stacked.
    LEFT, PW, CB_GAP, CBW, COL_GAP = 0.075, 0.350, 0.010, 0.014, 0.072
    X_P1  = LEFT
    X_CB1 = X_P1 + PW + CB_GAP
    X_P2  = X_CB1 + CBW + COL_GAP
    X_CB2 = X_P2 + PW + CB_GAP

    # ── Shared grid-drawing helper ────────────────────────────────────────────
    def draw_grid(ax, data, cmap, norm, fmt_fn, title,
                  hide_ylabels=False, hide_xlabels=False, text_color="white"):
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", origin="lower",
                       extent=[-0.5, nN - 0.5, -0.5, nDoN - 0.5])
        # Cell dividers
        for x in np.arange(0.5, nN - 0.5, 1):
            ax.axvline(x, color="white", lw=0.8)
        for y in np.arange(0.5, nDoN - 0.5, 1):
            ax.axhline(y, color="white", lw=0.8)
        # Per-cell annotations
        for i in range(nDoN):
            for j in range(nN):
                v = data[i, j]
                ax.text(j, i, fmt_fn(v), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)
        ax.set_xticks(range(nN))
        if hide_xlabels:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(N_labels, fontsize=13)
            ax.set_xlabel("Model size N", fontsize=16)
        ax.set_yticks(range(nDoN))
        if hide_ylabels:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(DoN_labels, fontsize=13)
            ax.set_ylabel("TPP (D/N)", fontsize=16)
        ax.set_title(title, fontsize=16, pad=5)
        return im

    # ══ Figure 1: R* grids (TOP) — shared colorbar + bottom legend ════════════
    # x-axis shown (each figure is self-contained); leave room below for the
    # x-labels and the boundary legend.
    fig1 = plt.figure(figsize=(12, 4.4))
    Y0_1, PH_1 = 0.27, 0.56
    ax1 = fig1.add_axes([X_P1, Y0_1, PW, PH_1])
    ax2 = fig1.add_axes([X_P2, Y0_1, PW, PH_1])

    # Single shared LogNorm spanning both grids so panels are directly comparable
    vmin_r = max(min(R_rep_grid.min(), R_para_grid.min()), 0.01)
    vmax_r = max(R_rep_grid.max(), R_para_grid.max())
    norm_r = mcolors.LogNorm(vmin=vmin_r, vmax=vmax_r)

    draw_grid(ax1, R_rep_grid,  CMAP_R, norm_r, fmt_R, r"$R^*$ — Repetition")
    m2 = draw_grid(ax2, R_para_grid, CMAP_R, norm_r, fmt_R, r"$R^*$ — Paraphrase",
                   hide_ylabels=True)

    # Single shared colorbar (right column slot) — default log ticks
    cbar_ax1 = fig1.add_axes([X_CB2, Y0_1, CBW, PH_1])
    fig1.colorbar(m2, cax=cbar_ax1, label=r"$R^*$")

    # Two polygon outlines: "para better" (lower-left) vs "rep better" (upper-right)
    from matplotlib.patches import Polygon as MplPolygon

    para_better = R_para_grid > R_rep_grid  # (nDoN, nN)

    # Rightmost para column per row; -0.5 means no para cells in that row
    bx = [float(np.where(para_better[i, :])[0].max()) + 0.5
          if para_better[i, :].any() else -0.5
          for i in range(nDoN)]

    # Para polygon: bottom-left → right to row-0 boundary → staircase up-left → close
    para_verts = [(-0.5, -0.5), (bx[0], -0.5)]
    for i in range(nDoN):
        if bx[i] <= -0.5:
            para_verts.append((-0.5, i - 0.5))
            break
        para_verts.append((bx[i], i + 0.5))
        nxt = bx[i + 1] if i + 1 < nDoN else -0.5
        para_verts.append((max(nxt, -0.5), i + 0.5))
        if nxt <= -0.5:
            break
    else:
        para_verts.append((-0.5, nDoN - 0.5))
    para_verts.append((-0.5, -0.5))  # close along left edge

    # Rep polygon: reverse staircase → across bottom → around grid perimeter
    stair_mid = para_verts[2:-1]       # staircase portion only
    rep_verts  = list(reversed(stair_mid))
    rep_verts.append((bx[0], -0.5))   # bottom of grid at rightmost para boundary
    rep_verts += [(nN - 0.5, -0.5), (nN - 0.5, nDoN - 0.5), (-0.5, nDoN - 0.5)]
    rep_verts.append(rep_verts[0])     # close

    # The non-staircase edges of both polygons run along the grid perimeter
    # (x/y at ±0.5 boundaries), so they sit right on the axes spine and get
    # half-clipped → nearly invisible. Nudge any vertex on a boundary inward by
    # `INSET` cells so the outline is clearly drawn *inside* the grid. Shared
    # staircase vertices are insetted identically in both polygons, so the two
    # outlines stay aligned along the diagonal.
    INSET = 0.09
    _xmin, _xmax, _ymin, _ymax = -0.5, nN - 0.5, -0.5, nDoN - 0.5

    def _inset_perimeter(verts):
        out = []
        for x, y in verts:
            if abs(x - _xmin) < 1e-9:   x = _xmin + INSET
            elif abs(x - _xmax) < 1e-9: x = _xmax - INSET
            if abs(y - _ymin) < 1e-9:   y = _ymin + INSET
            elif abs(y - _ymax) < 1e-9: y = _ymax - INSET
            out.append((x, y))
        return out

    para_verts = _inset_perimeter(para_verts)
    rep_verts  = _inset_perimeter(rep_verts)

    # Accent outlines chosen to stay legible on cividis (blue → yellow):
    #   para region is bright (high R*) → vermillion reads on yellow;
    #   rep region is dark (low R*)     → reddish-purple reads on dark blue.
    # Both are Okabe-Ito colorblind-safe and also show on a white legend.
    PARA_COLOR = "#F2C200"   # golden yellow (reads on the dark R* cells)
    REP_COLOR  = "#CC79A7"   # reddish-purple

    from matplotlib.lines import Line2D
    for ax in (ax1, ax2):
        ax.add_patch(MplPolygon(para_verts, closed=True,
                                fill=False, edgecolor=PARA_COLOR, lw=6,   linestyle="-",
                                zorder=3))
        ax.add_patch(MplPolygon(rep_verts,  closed=True,
                                fill=False, edgecolor=REP_COLOR,  lw=5,   linestyle="--",
                                zorder=4))
        ax.set_xlim(-0.5, nN - 0.5)
        ax.set_ylim(-0.5, nDoN - 0.5)

    # Legend centered under the two panels (within the reserved bottom margin)
    legend_handles1 = [
        Line2D([0], [0], color=PARA_COLOR, lw=3, linestyle="-",
               label=r"Paraphrase better  ($R^*_{\mathrm{para}} > R^*_{\mathrm{rep}}$)"),
        Line2D([0], [0], color=REP_COLOR,  lw=3, linestyle="--",
               label=r"Repetition better  ($R^*_{\mathrm{rep}} > R^*_{\mathrm{para}}$)"),
    ]
    panel_center = (X_P1 + X_P2 + PW) / 2.0
    fig1.legend(handles=legend_handles1,
                loc="lower center", bbox_to_anchor=(panel_center, 0.015),
                ncol=2, fontsize=15, frameon=False, borderpad=0.6, handlelength=2.5)

    fig1.savefig(f"{save_dir}/heatmap_rstar.pdf")
    fig1.savefig(f"{save_dir}/heatmap_rstar.png", dpi=150)
    plt.close(fig1)
    print(f"Saved {save_dir}/heatmap_rstar.{{pdf,png}}")

    # ══ Figure 2: Recommended ep / Recommended K grids (BOTTOM) ═══════════════
    # Same horizontal geometry; x-axis shown (shared axis for the stack).
    fig2 = plt.figure(figsize=(12, 3.9))
    Y0_2, PH_2 = 0.20, 0.64
    ax3 = fig2.add_axes([X_P1, Y0_2, PW, PH_2])
    ax4 = fig2.add_axes([X_P2, Y0_2, PW, PH_2])

    im3 = draw_grid(ax3, Ep_grid, CMAP_U,
                    mcolors.LogNorm(vmin=max(Ep_grid.min(), 1.001), vmax=Ep_grid.max()),
                    fmt_ep, "Recommended epochs (Ep@90%)", text_color="black")
    im4 = draw_grid(ax4, K_grid, CMAP_U,
                    mcolors.LogNorm(vmin=max(K_grid.min(), 0.1), vmax=K_grid.max()),
                    fmt_K,  "Recommended $K$ passes (K@90%)",
                    hide_ylabels=True, text_color="black")

    # Two colorbars (different scales) in the shared right-of-panel slots
    cbar_ax3 = fig2.add_axes([X_CB1, Y0_2, CBW, PH_2])
    fig2.colorbar(im3, cax=cbar_ax3, label="Epochs (log)")
    cbar_ax4 = fig2.add_axes([X_CB2, Y0_2, CBW, PH_2])
    fig2.colorbar(im4, cax=cbar_ax4, label="$K$ passes (log)")

    fig2.savefig(f"{save_dir}/heatmap_useful.pdf")
    fig2.savefig(f"{save_dir}/heatmap_useful.png", dpi=150)
    plt.close(fig2)
    print(f"Saved {save_dir}/heatmap_useful.{{pdf,png}}")

    # ══ Combined figure: both rows stacked tightly in one plot ════════════════
    # Same horizontal geometry as above, so all four panels share columns; the
    # top row hides its x-axis (truly shared with the bottom row).
    figc = plt.figure(figsize=(12, 6.6))
    PH_C    = 0.355
    TOP_Y0  = 0.515     # top row (R*)
    BOT_Y0  = 0.085     # bottom row (recommendations)

    # Combined-only horizontal geometry: the middle colorbar slot is omitted
    # here, so pull the two columns together (tight gap) and widen the panels to
    # use the reclaimed space. Only the far-right R* colorbar remains.
    C_MID = 0.040                       # gap between the two panels
    C_PW  = 0.378                       # panel width (wider than the shared PW)
    C_X1  = LEFT
    C_X2  = C_X1 + C_PW + C_MID
    C_CB2 = C_X2 + C_PW + CB_GAP
    C_center = (C_X1 + C_X2 + C_PW) / 2.0

    # ── Top row: R* (shared colorbar, x-axis hidden) ──────────────────────────
    tax1 = figc.add_axes([C_X1, TOP_Y0, C_PW, PH_C])
    tax2 = figc.add_axes([C_X2, TOP_Y0, C_PW, PH_C])
    draw_grid(tax1, R_rep_grid,  CMAP_R, norm_r, fmt_R, r"$R^*$ — Repetition",
              hide_xlabels=True)
    tm2 = draw_grid(tax2, R_para_grid, CMAP_R, norm_r, fmt_R, r"$R^*$ — Paraphrase",
                    hide_ylabels=True, hide_xlabels=True)
    figc.colorbar(tm2, cax=figc.add_axes([C_CB2, TOP_Y0, CBW, PH_C]), label=r"$R^*$")

    for ax in (tax1, tax2):
        ax.add_patch(MplPolygon(para_verts, closed=True, fill=False,
                                edgecolor=PARA_COLOR, lw=6, linestyle="-", zorder=3))
        ax.add_patch(MplPolygon(rep_verts,  closed=True, fill=False,
                                edgecolor=REP_COLOR,  lw=5, linestyle="--", zorder=4))
        ax.set_xlim(-0.5, nN - 0.5)
        ax.set_ylim(-0.5, nDoN - 0.5)

    # ── Bottom row: recommended ep / K (two colorbars, x-axis shown) ──────────
    bax3 = figc.add_axes([C_X1, BOT_Y0, C_PW, PH_C])
    bax4 = figc.add_axes([C_X2, BOT_Y0, C_PW, PH_C])
    bm3 = draw_grid(bax3, Ep_grid, CMAP_U,
                    mcolors.LogNorm(vmin=max(Ep_grid.min(), 1.001), vmax=Ep_grid.max()),
                    fmt_ep, "Recommended epochs (Ep@90%)", text_color="black")
    bm4 = draw_grid(bax4, K_grid, CMAP_U,
                    mcolors.LogNorm(vmin=max(K_grid.min(), 0.1), vmax=K_grid.max()),
                    fmt_K,  "Recommended $K$ passes (K@90%)",
                    hide_ylabels=True, text_color="black")
    # Bottom-row colorbars intentionally omitted: every cell is already annotated,
    # so the swatch is redundant and the rows can stack more tightly.

    # Crossover legend moved to the top (above the R* row) so the two rows stack
    # closely without a legend wedged between them.
    figc.legend(handles=legend_handles1, loc="upper center",
                bbox_to_anchor=(C_center, 0.985), ncol=2, fontsize=15,
                frameon=False, handlelength=2.5)

    figc.savefig(f"{save_dir}/heatmap_combined.pdf")
    figc.savefig(f"{save_dir}/heatmap_combined.png", dpi=150)
    plt.close(figc)
    print(f"Saved {save_dir}/heatmap_combined.{{pdf,png}}")


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-sections with confidence intervals
#
# The heatmaps above show the full R*(N, D/N) surface for repetition and
# paraphrase but cannot show uncertainty.  Here we take two 1-D slices through
# that surface and overlay 95% bootstrap CIs (§6.6 of writeup.md), so the
# para-vs-rep crossover can be read off *with* its sampling uncertainty.
#
#   Slice 1: fix N = 600M, sweep TTP (= D/N).  Crossover in D/N.
#   Slice 2: fix TTP = 20 (1× Chinchilla), sweep model size N.  Crossover in N.
# ═══════════════════════════════════════════════════════════════════════════════

def _load_bootstrap_samples(path=BOOTSTRAP_JSON):
    """Load the list of bootstrap parameter dicts; returns (samples, anchor)."""
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    full = path if os.path.isabs(path) else os.path.join(here, path)
    with open(full) as fh:
        d = json.load(fh)
    return d["bootstrap_samples"], d["anchor_params"]


def _rstar_ci(samples, source, D_over_N_arr, N_arr, q=(2.5, 97.5)):
    """
    R* with a bootstrap CI band along a 1-D sweep.

    D_over_N_arr, N_arr broadcast to a common length-M sweep (one is the swept
    axis, the other a fixed scalar broadcast by numpy).  Returns
    (central, lo, hi), each length M.  `central` uses the §6 headline anchors
    (the same values the heatmaps use); the band is the 2.5/97.5 percentile of
    R* evaluated under each of the B bootstrap parameter draws.
    """
    D_over_N_arr = np.broadcast_to(D_over_N_arr, np.broadcast(D_over_N_arr, N_arr).shape)
    N_arr        = np.broadcast_to(N_arr,        D_over_N_arr.shape)
    logD = np.log(D_over_N_arr)
    logN = np.log(N_arr)

    if source == "rep":
        kK, kr, ks = "log_K_rep", "rho_rep", "sigma_rep"
        central = np.exp(LOG_K_REP + RHO_REP * logD + SIGMA_REP * logN)
    else:
        kK, kr, ks = "log_K_para", "rho_para", "sigma_para"
        central = np.exp(LOG_K_PARA + RHO_PARA * logD + SIGMA_PARA * logN)

    boot = np.array([
        np.exp(s[kK] + s[kr] * logD + s[ks] * logN) for s in samples
    ])  # (B, M)
    lo, hi = np.percentile(boot, q, axis=0)
    return central, lo, hi


def _nice_log_ticks(lo, hi, candidates):
    """Subset of `candidates` lying within [lo, hi] (inclusive of a small pad)."""
    lo, hi = min(lo, hi), max(lo, hi)
    return [c for c in candidates if lo * 0.999 <= c <= hi * 1.001]


def _fmt_num(v):
    """Plain human-readable number: 0.1, 0.3, 3, 30, 100 (no 1e1 / 10^1)."""
    return f"{v:g}"


def _style_log_axis(ax, axis, candidates):
    """Put plain-number ticks at `candidates` within the current axis range and
    suppress the unlabeled minor ticks, so a log axis reads like 3, 10, 30."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    ticks = _nice_log_ticks(lo, hi, candidates)
    locator   = FixedLocator(ticks)
    formatter = FixedFormatter([_fmt_num(t) for t in ticks])
    if axis == "x":
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_locator(NullLocator())


def _interp_zero(x, diff):
    """First x where `diff` crosses zero (linear interp in log-x), or None."""
    sign = np.sign(diff)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    lx0, lx1 = math.log(x[i]), math.log(x[i + 1])
    d0, d1 = diff[i], diff[i + 1]
    lx = lx0 + (0.0 - d0) * (lx1 - lx0) / (d1 - d0)
    return math.exp(lx)


def _crossover_x(x, r_para, r_rep):
    """First x where r_para crosses r_rep (linear interp in log-log), or None."""
    return _interp_zero(x, np.log(r_para) - np.log(r_rep))


def _overlap_region(x, para_lo, para_hi, rep_lo, rep_hi):
    """
    Boundaries of the region where the paraphrase and repetition 95% CIs
    *overlap* — i.e. neither strategy significantly beats the other.

    Outside this region one band sits entirely above the other (a significant
    win).  Returns (x_lo, x_hi): the swept-x values where the significance
    switches.  Either endpoint may be None if the CIs never separate on that
    side within the swept range.

        para strictly above rep   <=>  para_lo > rep_hi
        rep  strictly above para  <=>  rep_lo  > para_hi
    """
    x_para_sig = _interp_zero(x, np.log(para_lo) - np.log(rep_hi))  # para-wins edge
    x_rep_sig  = _interp_zero(x, np.log(rep_lo)  - np.log(para_hi))  # rep-wins  edge
    lo, hi = sorted([e for e in (x_para_sig, x_rep_sig) if e is not None]) \
        if (x_para_sig is not None and x_rep_sig is not None) \
        else (x_para_sig, x_rep_sig)
    return lo, hi


def _mark_crossover(ax, x_arr, xco, x_lo, x_hi, fmt_fn, crossover_prefix, ann_fs=14):
    """
    Annotate a cross-section panel with the significance structure:

      • grey vertical band over [x_lo, x_hi] where the two 95% CIs overlap
        (neither strategy significantly beats the other);
      • a thin dotted line at the central crossover `xco`.

    x_lo / x_hi may be None if the CIs do not separate within the swept range;
    the band then extends to the corresponding edge of the swept range `x_arr`.
    """
    band_lo = x_lo if x_lo is not None else float(np.min(x_arr))
    band_hi = x_hi if x_hi is not None else float(np.max(x_arr))
    ax.axvspan(band_lo, band_hi, color="0.55", alpha=0.14, lw=0, zorder=0)

    if xco is not None:
        ax.axvline(xco, color="0.4", ls=":", lw=1.3, zorder=1)

    # Single label at the bottom of the band: crossover value + "CIs overlap"
    label = f"crossover {crossover_prefix} ≈ {fmt_fn(xco)}\nCIs overlap" \
        if xco is not None else "CIs overlap"
    ax.text(math.sqrt(band_lo * band_hi), 0.03, label,
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=ann_fs - 3, color="0.30", style="italic", zorder=2)


def plot_cross_sections_ci(save_dir="."):
    """
    Two-panel cross-section of the R* surface with 95% bootstrap CIs.

      Panel (a): TTP = 20 fixed; R* vs model size N.
      Panel (b): N = 600M fixed; R* vs TTP (D/N).

    Both panels overlay repetition and paraphrase R* with shaded 95% CI bands
    and mark the para↔rep crossover.
    """
    import glob
    import os
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    _font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'fonts')
    for _f in glob.glob(os.path.join(_font_dir, 'palatinolinotype_*.ttf')):
        font_manager.fontManager.addfont(_f)
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Palatino Linotype', 'P052', 'Palatino', 'serif'],
        'mathtext.fontset': 'cm',
        'font.size':        16,
    })
    LBL_FS, TIT_FS, TICK_FS, LEG_FS, ANN_FS = 17, 17, 15, 15, 14

    samples, _anchor = _load_bootstrap_samples()
    B = len(samples)

    PARA_COLOR = "#E69F00"   # amber  (matches heatmap_rstar)
    REP_COLOR  = "#CC79A7"   # mauve

    fig, (axN, axT) = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.subplots_adjust(left=0.09, right=0.98, wspace=0.26, top=0.91, bottom=0.15)

    # ── Panel (a): fix TTP = 20, sweep N ─────────────────────────────────────
    Nsw  = np.logspace(math.log10(14e6), math.log10(3e9), 200)
    ttpv = np.full_like(Nsw, float(TTP))

    rep_cN, rep_loN, rep_hiN = _rstar_ci(samples, "rep",  ttpv, Nsw)
    par_cN, par_loN, par_hiN = _rstar_ci(samples, "para", ttpv, Nsw)

    axN.fill_between(Nsw, rep_loN, rep_hiN, color=REP_COLOR,  alpha=0.22, lw=0)
    axN.fill_between(Nsw, par_loN, par_hiN, color=PARA_COLOR, alpha=0.22, lw=0)
    axN.plot(Nsw, rep_cN, color=REP_COLOR,  lw=2.4, label="Repetition")
    axN.plot(Nsw, par_cN, color=PARA_COLOR, lw=2.4, label="Paraphrase")

    xcoN = _crossover_x(Nsw, par_cN, rep_cN)
    oloN, ohiN = _overlap_region(Nsw, par_loN, par_hiN, rep_loN, rep_hiN)
    _mark_crossover(axN, Nsw, xcoN, oloN, ohiN, fmt_N, crossover_prefix="N")

    axN.set_xscale("log"); axN.set_yscale("log")
    axN.set_xlabel("Model size  N", fontsize=LBL_FS)
    axN.set_ylabel(r"$R^*$  (saturation budget)", fontsize=LBL_FS)
    axN.set_title(r"(a)  TPP $= 20$ (1× Chinchilla)  —  $R^*$ vs $N$",
                  fontsize=TIT_FS, pad=6)
    axN.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    xt = [30e6, 60e6, 190e6, 600e6, 1e9, 3e9]
    axN.set_xticks(xt); axN.set_xticklabels([fmt_N(x) for x in xt])
    axN.xaxis.set_minor_locator(NullLocator())
    _style_log_axis(axN, axis="y",
                    candidates=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300])
    axN.tick_params(labelsize=TICK_FS)
    axN.legend(fontsize=LEG_FS, frameon=False, loc="upper right")

    # ── Panel (b): fix N = 600M, sweep TTP = D/N ─────────────────────────────
    N_FIX = 600e6
    ttp = np.logspace(math.log10(3), math.log10(160), 200)
    Nv  = np.full_like(ttp, N_FIX)

    rep_cT, rep_loT, rep_hiT = _rstar_ci(samples, "rep",  ttp, Nv)
    par_cT, par_loT, par_hiT = _rstar_ci(samples, "para", ttp, Nv)

    axT.fill_between(ttp, rep_loT, rep_hiT, color=REP_COLOR,  alpha=0.22, lw=0)
    axT.fill_between(ttp, par_loT, par_hiT, color=PARA_COLOR, alpha=0.22, lw=0)
    axT.plot(ttp, rep_cT, color=REP_COLOR,  lw=2.4, label="Repetition")
    axT.plot(ttp, par_cT, color=PARA_COLOR, lw=2.4, label="Paraphrase")

    xcoT = _crossover_x(ttp, par_cT, rep_cT)
    oloT, ohiT = _overlap_region(ttp, par_loT, par_hiT, rep_loT, rep_hiT)
    _mark_crossover(axT, ttp, xcoT, oloT, ohiT, lambda v: f"{v:.0f}",
                    crossover_prefix="D/N")

    axT.set_xscale("log"); axT.set_yscale("log")
    axT.set_xlabel("TPP  (D/N)", fontsize=LBL_FS)
    axT.set_ylabel(r"$R^*$  (saturation budget)", fontsize=LBL_FS)
    axT.set_title(r"(b)  $N = 600$M  —  $R^*$ vs TPP", fontsize=TIT_FS, pad=6)
    axT.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    _style_log_axis(axT, axis="x", candidates=[3, 5, 10, 20, 40, 80, 160])
    _style_log_axis(axT, axis="y",
                    candidates=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300])
    axT.tick_params(labelsize=TICK_FS)

    fig.savefig(f"{save_dir}/cross_section_rstar_ci.pdf", bbox_inches="tight")
    fig.savefig(f"{save_dir}/cross_section_rstar_ci.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {save_dir}/cross_section_rstar_ci.{{pdf,png}}")
    if xcoN is not None:
        print(f"  Panel (a) TTP=20: para↔rep crossover at N ≈ {fmt_N(xcoN)}")
    if xcoT is not None:
        print(f"  Panel (b) N=600M: para↔rep crossover at D/N ≈ {xcoT:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 100)
    print("PRACTICAL SCALING-LAW IMPLICATIONS — §6 Triple Fit")
    print("Fit range: 30M–600M params; sizes above 600M are extrapolations (*)")
    print("=" * 100)
    print()
    print(f"Fitted parameters (triple one-go, k=15):")
    print(f"  Chinchilla:  E={E}  A={A}  B={B}  α={alpha}  β={beta}")
    print(f"  Rep  η:  log R* = {LOG_K_REP} + ({RHO_REP})·log(D/N) + ({SIGMA_REP})·log(N)")
    print(f"  Para η:  log R* = {LOG_K_PARA} + ({RHO_PARA})·log(D/N) + ({SIGMA_PARA})·log(N)")
    print(f"  Tokens per paraphrase pass: {TOKENS_PER_PARA:.3f}·D")
    print()

    q1_multi_epoch()
    q2_paraphrase()
    q3_comparison()
    bonus_loss_table()
    plot_figures(save_dir=".")
    plot_cross_sections_ci(save_dir=".")
