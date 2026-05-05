import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d as _interp1d
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# ---- Select model size ----
MODEL_SIZE = 'dolma_30m'

if MODEL_SIZE == '30m':
    import dclm_30m as model_data
elif MODEL_SIZE == '370m':
    import data_370m as model_data
elif MODEL_SIZE == 'dolma_14m':
    import dolma_14m as model_data
elif MODEL_SIZE == 'dolma_30m':
    import dolma_30m as model_data
elif MODEL_SIZE == 'dolma_60m':
    import dolma_60m as model_data
elif MODEL_SIZE == 'dolma_100m':
    import dolma_100m as model_data
elif MODEL_SIZE == 'dolma_190m':
    import dolma_190m as model_data
elif MODEL_SIZE == 'dolma_370m':
    import dolma_370m as model_data
elif MODEL_SIZE == 'dolma_600m':
    import dolma_600m as model_data
else:
    raise ValueError(f'Unknown MODEL_SIZE: {MODEL_SIZE}')

selfdistill_datasets = getattr(model_data, 'selfdistill_datasets', None)
parap_datasets = getattr(model_data, 'parap_datasets', None)

all_datasets = model_data.ALL_DATASETS


def _best_per_chin(datasets):
    """Return three lists (chin, val_loss, flops) — one entry per chin scale (lowest loss)."""
    by_chin = {}
    for data in datasets:
        chin = data['chinchilla_scale'][0]
        for f, l in zip(data['flops_multiplier'], data['validation_loss']):
            if np.isnan(l):
                continue
            if chin not in by_chin or l < by_chin[chin][0]:
                by_chin[chin] = (l, f)
    items = sorted(by_chin.items())
    chins = [c for c, _ in items]
    losses = [v[0] for _, v in items]
    flops = [v[1] for _, v in items]
    return chins, losses, flops


sd_best_chin, sd_best_loss, sd_best_flops = (
    _best_per_chin(selfdistill_datasets) if selfdistill_datasets else ([], [], [])
)

# ---- Plot: Data Size vs Loss, scatter colored by FLOPs ----

# Collect all multi-epoch points as flat arrays
all_chin = []
all_loss = []
all_flops = []

for data in all_datasets:
    chin = data['chinchilla_scale'][0]
    for f, l in zip(data['flops_multiplier'], data['validation_loss']):
        if not np.isnan(l):
            all_chin.append(chin)
            all_loss.append(l)
            all_flops.append(f)

all_chin = np.array(all_chin)
all_loss = np.array(all_loss)
all_flops = np.array(all_flops)

# Colormap: YlOrRd — yellow = low FLOPs (cheap), red = high FLOPs (expensive)
# Shared norm across multi-epoch and paraphrasing
flops_min = all_flops.min()
flops_max = all_flops.max()
if parap_datasets is not None:
    for data in parap_datasets:
        for f, l in zip(data['flops_multiplier'], data['validation_loss']):
            if not np.isnan(l):
                flops_min = min(flops_min, f)
                flops_max = max(flops_max, f)
if selfdistill_datasets is not None:
    for data in selfdistill_datasets:
        for f, l in zip(data['flops_multiplier'], data['validation_loss']):
            if not np.isnan(l):
                flops_min = min(flops_min, f)
                flops_max = max(flops_max, f)

norm = plt.Normalize(vmin=np.log2(flops_min), vmax=np.log2(flops_max))
cmap = plt.cm.YlOrRd

FONT_LABEL = 15
FONT_TITLE = 17
FONT_LEGEND = 13
FONT_TICK = 13

fig, ax = plt.subplots(figsize=(10, 6))

# Multi-epoch scatter (circles), shifted 8% left
X_SHIFT_LEFT = 0.92
all_ttp = all_chin * 20 * X_SHIFT_LEFT
sc = ax.scatter(all_ttp, all_loss, c=np.log2(all_flops), cmap=cmap, norm=norm,
                s=30, edgecolors='k', linewidths=0.4, zorder=10, label='Multi-epoch')

# Paraphrasing: full per-flops scatter with diamond markers, unshifted (middle position)
if parap_datasets is not None:
    parap_chin = []
    parap_loss = []
    parap_flops = []
    for data in parap_datasets:
        chin = data['chinchilla_scale'][0]
        for f, l in zip(data['flops_multiplier'], data['validation_loss']):
            if not np.isnan(l):
                parap_chin.append(chin)
                parap_loss.append(l)
                parap_flops.append(f)
    parap_chin = np.array(parap_chin)
    parap_loss = np.array(parap_loss)
    parap_flops = np.array(parap_flops)
    parap_ttp = parap_chin * 20  # no shift — sits between circles (left) and crosses (right)

    sc_parap = ax.scatter(parap_ttp, parap_loss, c=np.log2(parap_flops), cmap=cmap, norm=norm,
                          marker='D', s=30, edgecolors='k', linewidths=0.4, zorder=12,
                          label='Paraphrasing')

# Self-distillation: scatter ALL (chin, K) points at 8%-right-shifted x position
X_SHIFT_RIGHT = 1.08
if selfdistill_datasets is not None:
    sd_chin = []
    sd_loss = []
    sd_flops = []
    for data in selfdistill_datasets:
        chin = data['chinchilla_scale'][0]
        for f, l in zip(data['flops_multiplier'], data['validation_loss']):
            if not np.isnan(l):
                sd_chin.append(chin)
                sd_loss.append(l)
                sd_flops.append(f)
    sd_chin = np.array(sd_chin)
    sd_loss = np.array(sd_loss)
    sd_flops = np.array(sd_flops)
    ax.scatter(sd_chin * 20 * X_SHIFT_RIGHT, sd_loss,
               c=np.log2(sd_flops), cmap=cmap, norm=norm,
               marker='X', s=80, edgecolors='k', linewidths=0.5,
               label='Self-distillation', zorder=20)

# Three Data Optimal lines — same color family (blues), darkest = self-distill
COLOR_OPT_MULTI  = '#9ecae1'  # light blue
COLOR_OPT_SYN    = '#3182bd'  # medium blue
COLOR_OPT_SELF   = '#08519c'  # dark blue

# Data Optimal (multi-epoch): min loss per chinchilla scale from multi-epoch dots
chin_unique = sorted(set(all_chin))
min_loss_per_chin = [all_loss[all_chin == c].min() for c in chin_unique]
ax.plot([c * 20 for c in chin_unique], min_loss_per_chin,
        linestyle='--', color=COLOR_OPT_MULTI, linewidth=2.5, alpha=0.9,
        label='Data Optimal (multi-epoch)', zorder=5)

# Data Optimal (paraphrase): min loss per chinchilla scale from paraphrasing
if parap_datasets is not None:
    parap_chin_all = np.array([data['chinchilla_scale'][0]
                                for data in parap_datasets
                                for l in data['validation_loss']
                                if not np.isnan(l)])
    parap_loss_all = np.array([l
                                for data in parap_datasets
                                for l in data['validation_loss']
                                if not np.isnan(l)])
    parap_chin_unique = sorted(set(parap_chin_all))
    min_loss_parap = [parap_loss_all[parap_chin_all == c].min() for c in parap_chin_unique]
    ax.plot([c * 20 for c in parap_chin_unique], min_loss_parap,
            linestyle='--', color=COLOR_OPT_SYN, linewidth=2.5, alpha=0.9,
            label='Data Optimal (paraphrase)', zorder=5)

# Data Optimal (self-distill): best-per-chin envelope across self-distill scatter
if selfdistill_datasets is not None and sd_best_chin:
    ax.plot([c * 20 for c in sd_best_chin], sd_best_loss,
            linestyle='--', color=COLOR_OPT_SELF, linewidth=2.5, alpha=0.9,
            label='Data Optimal (self-distill)', zorder=5)

# Compute Optimal: connect 1-epoch points across all data scales
compute_opt_chin = []
compute_opt_loss = []
for data in all_datasets:
    epochs = np.array(data['epochs'])
    loss = np.array(data['validation_loss'], dtype=float)
    idx_1ep = np.where(epochs == 1)[0]
    if len(idx_1ep) > 0 and not np.isnan(loss[idx_1ep[0]]):
        compute_opt_chin.append(data['chinchilla_scale'][0])
        compute_opt_loss.append(loss[idx_1ep[0]])
ax.plot([c * 20 for c in compute_opt_chin], compute_opt_loss,
        linestyle='--', color='black', linewidth=2.5, alpha=0.7,
        label='Compute Optimal', zorder=5)

# === Shade regions ===
# Derive axis limits and shading floor from data
_all_min_loss = all_loss.min()  # min across ALL runs for this model size
_FLOOR = np.floor(_all_min_loss * 10) / 10 - 0.1  # floor slightly below min loss
_chin_max = max(chin_unique)
_XLEFT = min(chin_unique) * 20 * 0.5
_XRIGHT = _chin_max * 20 * 1.5
_MODEL_BOUND_Y = _all_min_loss  # model-bound = best achievable loss for this model size

# Build interpolators in log2(x) space — matches matplotlib's log-scale line rendering exactly.
_co_pairs = sorted(zip([c * 20 for c in compute_opt_chin], compute_opt_loss))
_co_xs = np.array([x for x, y in _co_pairs])
_co_ys = np.array([y for x, y in _co_pairs])
_co_fn = _interp1d(np.log2(_co_xs), _co_ys, kind='linear', fill_value='extrapolate')
def _co(x): return _co_fn(np.log2(np.asarray(x, dtype=float)))

# Minimum envelope of all three blue data-optimal curves
_blue_fns = []
_me_pairs = sorted(zip([c * 20 for c in chin_unique], min_loss_per_chin))
_me_fn = _interp1d(np.log2([x for x, y in _me_pairs]), [y for x, y in _me_pairs],
                   kind='linear', fill_value='extrapolate')
_blue_fns.append(_me_fn)
if parap_datasets is not None:
    _pa_pairs = sorted(zip([c * 20 for c in parap_chin_unique], min_loss_parap))
    _pa_fn = _interp1d(np.log2([x for x, y in _pa_pairs]), [y for x, y in _pa_pairs],
                       kind='linear', fill_value='extrapolate')
    _blue_fns.append(_pa_fn)
if selfdistill_datasets is not None and sd_best_chin:
    _sd_pairs = sorted(zip([c * 20 for c in sd_best_chin], sd_best_loss))
    _sd_fn = _interp1d(np.log2([x for x, y in _sd_pairs]), [y for x, y in _sd_pairs],
                       kind='linear', fill_value='extrapolate')
    _blue_fns.append(_sd_fn)

def _blue_min(x):
    log2x = np.log2(np.asarray(x, dtype=float))
    return np.minimum.reduce([fn(log2x) for fn in _blue_fns])

# Darkest-blue interpolator (last appended = highest priority) for the data-bound region
_do_fn = _blue_fns[-1]
def _do(x): return _do_fn(np.log2(np.asarray(x, dtype=float)))

# Compute-bound (yellow): above max(blue_min, _MODEL_BOUND_Y), up to compute-optimal (black) line
_x_cb = np.geomspace(_XLEFT, _XRIGHT, 500)
_co_y_cb = _co(_x_cb)
_blue_bot = _blue_min(_x_cb)
_y_bot_cb = np.maximum(_blue_bot, _MODEL_BOUND_Y)
ax.fill_between(_x_cb, _y_bot_cb, _co_y_cb,
                where=_co_y_cb > _y_bot_cb,
                color='#ffcc44', alpha=0.25, zorder=1, label='Compute-bound')

# Data-bound: below darkest blue, above _MODEL_BOUND_Y, from y-spine to end of blue
_do_xs = np.array([x for x, y in _sd_pairs]) if (selfdistill_datasets is not None and sd_best_chin) else \
         np.array([x for x, y in _pa_pairs]) if parap_datasets is not None else \
         np.array([x for x, y in _me_pairs])
_x_db = np.geomspace(_XLEFT, _do_xs.max(), 300)
ax.fill_between(_x_db, _MODEL_BOUND_Y, _do(_x_db),
                where=_do(_x_db) > _MODEL_BOUND_Y,
                color='#6baed6', alpha=0.25, zorder=2, label='Data-bound')

# Model-bound (green): strictly y <= _MODEL_BOUND_Y across full x range
ax.fill_between([_XLEFT, _XRIGHT], _FLOOR, _MODEL_BOUND_Y,
                color='#74c476', alpha=0.25, zorder=3, label='Model-bound')

ax.set_xlabel('Fresh Data Size,  TTP (Chinchilla X)', fontsize=FONT_LABEL)
ax.set_ylabel('Validation Loss', fontsize=FONT_LABEL)
# ax.set_title(f'Validation Loss vs Data Size for Different FLOPs Budgets ({MODEL_SIZE.upper()})',
#              fontsize=FONT_TITLE)
ax.set_xscale('log', base=2)
ax.set_xlim(_XLEFT, _XRIGHT)
ax.set_ylim(bottom=_FLOOR)
ax.tick_params(axis='both', labelsize=FONT_TICK)

# TTP tick values with chinchilla scale in parentheses (use unshifted positions)
ttp_values = [c * 20 for c in chin_unique]
ax.set_xticks(ttp_values)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}\n({x/20:g}x)'))
ax.xaxis.set_minor_formatter(plt.NullFormatter())

ax.grid(True, alpha=0.3)

# Legend in specified order: multi-epoch, paraphrasing, self-distillation, data-optimal, compute-optimal
all_handles, all_labels = ax.get_legend_handles_labels()
handles_map = dict(zip(all_labels, all_handles))
legend_order = ['Multi-epoch', 'Paraphrasing', 'Self-distillation',
                'Data Optimal (multi-epoch)', 'Data Optimal (paraphrase)', 'Data Optimal (self-distill)',
                'Compute Optimal',
                'Compute-bound', 'Model-bound', 'Data-bound']
ordered_handles = [handles_map[lbl] for lbl in legend_order if lbl in handles_map]
ordered_labels = [lbl for lbl in legend_order if lbl in handles_map]
ax.legend(ordered_handles, ordered_labels, fontsize=FONT_LEGEND,
          loc='upper right')

# Colorbar with FLOPs tick labels
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('FLOPs (Chinchilla Optimal = 1X)', fontsize=FONT_LABEL)
cbar.ax.tick_params(labelsize=FONT_TICK)
flops_ticks = [0.05, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
cbar.set_ticks([np.log2(v) for v in flops_ticks])
cbar.set_ticklabels([f'{v:g}' for v in flops_ticks])

plt.tight_layout()
out_path = f'results/data_optimal_scaling_{MODEL_SIZE}.pdf'
plt.savefig(out_path, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.show()
