"""
LSE-style scaling-law fitter.

Fits log L = logsumexp(e, b - beta * log D, ...) in log space via
L-BFGS + Huber loss, with a grid search over initial parameter values.

The LSE form is algebraically identical to the additive Chinchilla form
    L = exp(e) + exp(b) / D^beta + ...
but fitting in log space with Huber loss matches the procedure used in
the scaling-laws literature (Hoffmann '22, Besiroglu '24).

This module is loss-agnostic: the caller supplies a forward_fn that maps
a dict of trainable tensors to predicted log L. Features (log D etc.)
are captured in the caller's closure.
"""

import itertools
from typing import Callable, Dict, List

import torch


# ──────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────

def logsumexp_stable(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Numerically stable log-sum-exp along `dim`."""
    m = torch.max(x, dim=dim, keepdim=True).values
    return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m), dim=dim))


def expand_grid(grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """Cartesian product of per-axis grids."""
    keys = list(grid.keys())
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


# ──────────────────────────────────────────────────────────────────────
# Core fitter
# ──────────────────────────────────────────────────────────────────────

ForwardFn = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


def _lbfgs(params: Dict[str, torch.Tensor], forward_fn: ForwardFn,
           log_L_obs: torch.Tensor, *, delta: float, max_iter: int,
           outer_iter: int, lr: float,
           weights: torch.Tensor = None) -> float:
    """Run L-BFGS + Huber in log space; returns final (weighted) Huber loss.
    If weights is given (1-D tensor, same length as log_L_obs), per-row Huber
    is multiplied by weights and summed."""
    opt = torch.optim.LBFGS(
        list(params.values()), lr=lr, max_iter=max_iter,
        history_size=50, line_search_fn="strong_wolfe",
    )
    if weights is None:
        huber = torch.nn.HuberLoss(delta=delta, reduction="sum")

        def closure():
            opt.zero_grad()
            loss = huber(forward_fn(params), log_L_obs)
            loss.backward()
            return loss
    else:
        huber_per = torch.nn.HuberLoss(delta=delta, reduction="none")

        def closure():
            opt.zero_grad()
            per_elem = huber_per(forward_fn(params), log_L_obs)
            loss = (per_elem * weights).sum()
            loss.backward()
            return loss

    final = float("inf")
    for _ in range(outer_iter):
        final = opt.step(closure).item()
    return final


def fit_lse(forward_fn: ForwardFn,
            log_L_obs: torch.Tensor,
            init_grid: List[Dict[str, float]],
            *,
            delta: float = 1e-3,
            max_iter: int = 2000,
            outer_iter: int = 10,
            grid_max_iter: int = 50,
            grid_outer_iter: int = 1,
            lr: float = 1.0,
            weights: torch.Tensor = None,
            verbose: bool = False) -> Dict:
    """Grid search + L-BFGS refine; returns dict with fitted params.

    Each grid point is briefly optimized; the best-loss point is then
    polished with a longer L-BFGS run.

    Returns:
        params:     dict of optimized parameters (python scalars)
        loss:       final sum-Huber loss on log L
        rmse_logL:  sqrt(mean((pred - log L)^2))
        r2_logL:    1 - SS_res / SS_tot on log L
        best_init:  winning grid point (dict)
    """
    best_loss, best_state, best_init = float("inf"), None, None
    for init in init_grid:
        params = {k: torch.tensor(float(v), dtype=log_L_obs.dtype, requires_grad=True)
                  for k, v in init.items()}
        _lbfgs(params, forward_fn, log_L_obs,
               delta=delta, max_iter=grid_max_iter,
               outer_iter=grid_outer_iter, lr=lr, weights=weights)
        with torch.no_grad():
            if weights is None:
                loss = torch.nn.HuberLoss(delta=delta, reduction="sum")(
                    forward_fn(params), log_L_obs).item()
            else:
                per_elem = torch.nn.HuberLoss(delta=delta, reduction="none")(
                    forward_fn(params), log_L_obs)
                loss = (per_elem * weights).sum().item()
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.detach().clone() for k, v in params.items()}
            best_init = init

    if best_state is None:
        raise RuntimeError("Grid search produced no valid fit.")

    # Polish
    params = {k: v.detach().clone().requires_grad_(True) for k, v in best_state.items()}
    final_loss = _lbfgs(params, forward_fn, log_L_obs,
                        delta=delta, max_iter=max_iter,
                        outer_iter=outer_iter, lr=lr, weights=weights)

    with torch.no_grad():
        pred = forward_fn(params)
        resid = pred - log_L_obs
        rmse = torch.sqrt(torch.mean(resid ** 2)).item()
        ss_tot = torch.sum((log_L_obs - log_L_obs.mean()) ** 2).item()
        r2 = 1.0 - torch.sum(resid ** 2).item() / ss_tot if ss_tot > 0 else float("nan")

    if verbose:
        print(f"[fit_lse] loss={final_loss:.5g}  R²(logL)={r2:.5f}  "
              f"RMSE(logL)={rmse:.5f}  init={best_init}")

    return {
        "params": {k: v.detach().item() for k, v in params.items()},
        "loss": final_loss,
        "rmse_logL": rmse,
        "r2_logL": r2,
        "best_init": best_init,
    }
