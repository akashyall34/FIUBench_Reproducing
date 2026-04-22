"""
Generate Figure 2: Performance of various baselines under LLaVA-Phi over different unlearning steps.

Usage:
  !python plot_figure2.py \
      --eval_dir /content/drive/MyDrive/fiubench_checkpoints/step_eval \
      --out /content/drive/MyDrive/fiubench_checkpoints/figure2.pdf

Expects one JSON per (method, step) saved by eval_step_level.py:
  {method}_step{step:04d}.json  e.g.  ga_step0006.json

JSON format:
  {"method": "ga", "step": 6, "rouge_l": 0.91, "gpt_eval": 0.78,
   "exact_match": 0.42, "mia_mink": 0.55}
"""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--eval_dir', default='/content/drive/MyDrive/fiubench_checkpoints/step_eval')
parser.add_argument('--out',      default='/content/drive/MyDrive/fiubench_checkpoints/figure2.pdf')
args = parser.parse_args()

# ── load all JSONs ────────────────────────────────────────────────────────────
data = {}   # {method: {step: {metric: value}}}
for p in sorted(Path(args.eval_dir).glob('*.json')):
    try:
        d = json.loads(p.read_text())
        method = d['method']
        step   = int(d['step'])
        data.setdefault(method, {})[step] = d
    except Exception as e:
        print(f"Skip {p.name}: {e}")

if not data:
    raise RuntimeError(f"No JSON files found in {args.eval_dir}")

methods_found = sorted(data.keys())
print(f"Methods found: {methods_found}")
for m in methods_found:
    print(f"  {m}: steps {sorted(data[m].keys())}")

# ── style matching Figure 2 ──────────────────────────────────────────────────
METHOD_STYLE = {
    'kl':     dict(label='KL Minimization',       color='#2ca02c', marker='D', lw=2.0),
    'gd':     dict(label='Gradient Difference',   color='#ff7f0e', marker='s', lw=2.0),
    'ga':     dict(label='Gradient Ascent',       color='#1f77b4', marker='o', lw=2.0),
    'po':     dict(label='Preference Optimization', color='#d62728', marker='^', lw=2.0),
    'retain': dict(label='Retain Model',          color='purple',  marker='*', lw=2.0),
}

PLOTS = [
    ('rouge_l',     'Rouge-L',       'Model Utility'),
    ('gpt_eval',    'GPT Score',     'Model Utility'),
    ('exact_match', 'Exact Match',   'Forget Quality'),
    ('mia_mink',    'MIA',           'Forget Quality'),
]

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.subplots_adjust(bottom=0.22, wspace=0.32)

for ax, (metric, ylabel, title_type) in zip(axes, PLOTS):
    ax.set_title(f'{ylabel}  ({title_type})', fontsize=11, pad=6)
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # plot in fixed order so legend matches paper (KL, GD, GA, PO)
    for method in ['kl', 'gd', 'ga', 'po']:
        if method not in data:
            continue
        style = METHOD_STYLE[method]
        steps = sorted(data[method].keys())
        vals  = [data[method][s].get(metric, float('nan')) for s in steps]
        ax.plot(steps, vals,
                color=style['color'], marker=style['marker'],
                linewidth=style['lw'], markersize=6,
                label=style['label'])

    ax.tick_params(labelsize=9)

# ── shared legend below all subplots ────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color=METHOD_STYLE[m]['color'],
                  marker=METHOD_STYLE[m]['marker'],
                  linewidth=2, markersize=7,
                  label=METHOD_STYLE[m]['label'])
    for m in ['kl', 'gd', 'ga', 'po']
    if m in data
]
fig.legend(handles=handles, loc='lower center', ncol=len(handles),
           fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, 0.02))

fig.suptitle('', y=1.0)  # remove if you don't want a super-title

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=150, bbox_inches='tight')

# also save as PNG alongside
png_out = out.with_suffix('.png')
fig.savefig(str(png_out), dpi=150, bbox_inches='tight')

print(f"\n✅ Figure saved → {out}")
print(f"✅ PNG saved     → {png_out}")
plt.close()
