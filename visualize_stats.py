"""Visualize encoder and decoder pipeline statistics."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
enc = pd.read_csv("encoder_stats.csv")
dec = pd.read_csv("decoder_stats.csv")

# Merge on chunk_index
df = enc.merge(dec, on="chunk_index", suffixes=("_enc", "_dec"))

# Compute derived columns
df["total_latency"] = df["encode_time_s"] + df["serialize_time_s"] + df["latency_s"] + df["decode_time_s"]
df["network_latency"] = df["latency_s"]

# ── Style setup ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
})

COLORS = {
    "encode": "#58a6ff",
    "serialize": "#bc8cff",
    "network": "#f0883e",
    "decode": "#3fb950",
    "total": "#f778ba",
}

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Encoder / Decoder Pipeline Stats", fontsize=18, fontweight="bold", y=0.97)

gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.30,
                      left=0.08, right=0.95, top=0.92, bottom=0.06)

# ── 1. Stacked area – per-chunk time breakdown ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
chunks = df["chunk_index"]
ax1.stackplot(
    chunks,
    df["encode_time_s"] * 1000,
    df["serialize_time_s"] * 1000,
    df["network_latency"] * 1000,
    df["decode_time_s"] * 1000,
    labels=["Encode", "Serialize", "Network", "Decode"],
    colors=[COLORS["encode"], COLORS["serialize"], COLORS["network"], COLORS["decode"]],
    alpha=0.85,
)
ax1.set_title("Per-Chunk Time Breakdown", fontsize=14, pad=10)
ax1.set_xlabel("Chunk Index")
ax1.set_ylabel("Time (ms)")
ax1.legend(loc="upper right", framealpha=0.6, edgecolor="#30363d")
ax1.set_xlim(chunks.min(), chunks.max())
ax1.grid(axis="y", linewidth=0.4)

# ── 2. Encode time per chunk ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(chunks, df["encode_time_s"] * 1000, color=COLORS["encode"], width=0.7, alpha=0.9)
mean_enc = df["encode_time_s"].mean() * 1000
ax2.axhline(mean_enc, color=COLORS["encode"], ls="--", lw=1.2, alpha=0.7)
ax2.text(chunks.max() * 0.6, mean_enc + 0.3, f"mean = {mean_enc:.2f} ms",
         color=COLORS["encode"], fontsize=10)
ax2.set_title("Encode Time per Chunk", fontsize=13, pad=8)
ax2.set_xlabel("Chunk Index")
ax2.set_ylabel("Time (ms)")
ax2.grid(axis="y", linewidth=0.4)

# ── 3. Decode time per chunk ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(chunks, df["decode_time_s"] * 1000, color=COLORS["decode"], width=0.7, alpha=0.9)
mean_dec = df["decode_time_s"].mean() * 1000
ax3.axhline(mean_dec, color=COLORS["decode"], ls="--", lw=1.2, alpha=0.7)
ax3.text(chunks.max() * 0.6, mean_dec + 0.5, f"mean = {mean_dec:.2f} ms",
         color=COLORS["decode"], fontsize=10)
ax3.set_title("Decode Time per Chunk", fontsize=13, pad=8)
ax3.set_xlabel("Chunk Index")
ax3.set_ylabel("Time (ms)")
ax3.grid(axis="y", linewidth=0.4)

# ── 4. Network latency per chunk ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(chunks, df["network_latency"] * 1000, color=COLORS["network"],
         marker="o", markersize=4, linewidth=1.8)
ax4.fill_between(chunks, 0, df["network_latency"] * 1000,
                 color=COLORS["network"], alpha=0.15)
mean_net = df["network_latency"].mean() * 1000
ax4.axhline(mean_net, color=COLORS["network"], ls="--", lw=1.2, alpha=0.7)
ax4.text(chunks.max() * 0.55, mean_net + 3, f"mean = {mean_net:.2f} ms",
         color=COLORS["network"], fontsize=10)
ax4.set_title("Network Latency per Chunk", fontsize=13, pad=8)
ax4.set_xlabel("Chunk Index")
ax4.set_ylabel("Latency (ms)")
ax4.grid(axis="y", linewidth=0.4)

# ── 5. Summary table ────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis("off")

summary_data = [
    ["Encode",    f"{df['encode_time_s'].mean()*1000:.2f}",
                  f"{df['encode_time_s'].std()*1000:.2f}",
                  f"{df['encode_time_s'].min()*1000:.2f}",
                  f"{df['encode_time_s'].max()*1000:.2f}"],
    ["Serialize", f"{df['serialize_time_s'].mean()*1000:.2f}",
                  f"{df['serialize_time_s'].std()*1000:.2f}",
                  f"{df['serialize_time_s'].min()*1000:.2f}",
                  f"{df['serialize_time_s'].max()*1000:.2f}"],
    ["Network",   f"{df['network_latency'].mean()*1000:.2f}",
                  f"{df['network_latency'].std()*1000:.2f}",
                  f"{df['network_latency'].min()*1000:.2f}",
                  f"{df['network_latency'].max()*1000:.2f}"],
    ["Decode",    f"{df['decode_time_s'].mean()*1000:.2f}",
                  f"{df['decode_time_s'].std()*1000:.2f}",
                  f"{df['decode_time_s'].min()*1000:.2f}",
                  f"{df['decode_time_s'].max()*1000:.2f}"],
    ["Total",     f"{df['total_latency'].mean()*1000:.2f}",
                  f"{df['total_latency'].std()*1000:.2f}",
                  f"{df['total_latency'].min()*1000:.2f}",
                  f"{df['total_latency'].max()*1000:.2f}"],
]
col_labels = ["Stage", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)"]

table = ax5.table(
    cellText=summary_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.6)

# Style table
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#30363d")
    if row == 0:
        cell.set_facecolor("#161b22")
        cell.set_text_props(fontweight="bold", color="#c9d1d9")
    else:
        cell.set_facecolor("#0d1117")
        cell.set_text_props(color="#c9d1d9")
        # Color the stage name column
        if col == 0:
            stage = summary_data[row - 1][0].lower()
            color = COLORS.get(stage, "#c9d1d9")
            cell.set_text_props(color=color, fontweight="bold")

ax5.set_title("Summary Statistics", fontsize=13, pad=12)

# ── Save & show ──────────────────────────────────────────────────────────────
output_path = "pipeline_stats.png"
fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
print(f"Saved visualization to {output_path}")
plt.show()
