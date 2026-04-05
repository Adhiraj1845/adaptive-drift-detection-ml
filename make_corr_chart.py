"""Generate market correlation bar chart for all 21 experiments."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, static_corr, adaptive_corr)
# Original 14 v2 runs (dji_exp excluded — static corr NaN)
original = [
    ("NASDAQ",      0.157, 0.889),
    ("AAPL",        0.242, 0.895),
    ("BTC",         0.234, 0.852),
    ("GSPC",        0.404, 0.877),
    ("TLT",         0.444, 0.844),
    ("MSFT",        0.462, 0.906),
    ("AAPL-exp",    0.458, 0.902),
    ("GSPC-exp",    0.348, 0.898),
    ("TSLA",        0.596, 0.857),
    ("XOM",         0.610, 0.884),
    ("GLD",         0.719, 0.892),
    ("DJI",         0.702, 0.851),
    ("GOLD",        0.831, 0.881),
]

# 7 new experiments
new_exps = [
    ("NVDA",        0.279, 0.867),
    ("ETH",         0.210, 0.859),
    ("NFLX",        0.483, 0.843),
    ("AMZN",        0.447, 0.900),
    ("SMH",         0.476, 0.872),
    ("META",        0.770, 0.885),
    ("ARKK",        0.854, 0.864),
]

all_exps = original + new_exps

# Sort by gap descending
all_exps.sort(key=lambda x: x[2] - x[1], reverse=True)

labels   = [e[0] for e in all_exps]
static   = np.array([e[1] for e in all_exps])
adaptive = np.array([e[2] for e in all_exps])
gaps     = adaptive - static

# Colour bars differently for new experiments
new_labels = {e[0] for e in new_exps}
bar_colors_s = ["#CC3333"] * len(labels)
bar_colors_a = ["#1A7A3A"] * len(labels)

n = len(labels)
x = np.arange(n)
w = 0.35

fig, ax = plt.subplots(figsize=(17, 7), facecolor="white")
ax.set_facecolor("white")

for i in range(n):
    ax.bar(x[i] - w/2, static[i],   w, color="#CC3333", zorder=3,
           alpha=0.75 if labels[i] in new_labels else 1.0)
    ax.bar(x[i] + w/2, adaptive[i], w, color="#1A7A3A", zorder=3,
           alpha=0.75 if labels[i] in new_labels else 1.0)

# Gap annotations
for i, (a, g) in enumerate(zip(adaptive, gaps)):
    ax.text(x[i] + w/2, a + 0.005, f"+{g:.2f}", ha="center", va="bottom",
            fontsize=7, color="#1A7A3A", fontweight="bold")

# Tick labels: italicise new experiments
ax.set_xticks(x)
tick_labels = [f"{l}*" if l in new_labels else l for l in labels]
ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=9.5)

ax.set_ylabel("Market Correlation (Pearson r)", fontsize=12)
ax.set_title("Market Correlation: Static vs Adaptive — 20 Experiments  (* = new)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.08)
ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", zorder=2)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#CC3333", label="Static"),
    Patch(facecolor="#1A7A3A", label="Adaptive"),
]
ax.legend(handles=legend_elements, fontsize=11)
ax.grid(axis="y", color="lightgrey", linewidth=0.5, zorder=1)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
out = "results/chart_market_corr_21experiments.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
