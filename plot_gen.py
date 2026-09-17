import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CARICAMENTO DATI
# -----------------------------------------------------------------------------
df = pd.read_csv("benchmark_results.csv")
t_seq = df[df["Platform"] == "CPU_Sequential"]["Mean_Time_s"].values[0]

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

# -----------------------------------------------------------------------------
# 2. GRAFICO 1: CPU STRONG SCALING & EFFICIENCY
# -----------------------------------------------------------------------------
df_cpu = df[df["Platform"] == "CPU_Multiprocessing"].copy()
workers = df_cpu["Workers/Units"].astype(int).values
cpu_mean = df_cpu["Mean_Time_s"].values
cpu_std = df_cpu["Std_Time_s"].values
cpu_speedup = df_cpu["Speedup"].values
cpu_eff = df_cpu["Efficiency"].values * 100.0
cpu_speedup_err = cpu_speedup * (cpu_std / cpu_mean)

fig_cpu, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Subplot 1: Speedup CPU
ax1.plot(
    workers,
    workers,
    "k--",
    label="Ideal Linear ($S=p$)",
    linewidth=1.5,
    alpha=0.7,
)
ax1.errorbar(
    workers,
    cpu_speedup,
    yerr=cpu_speedup_err,
    fmt="-o",
    color="#1f77b4",
    ecolor="#d62728",
    elinewidth=1.8,
    capsize=4,
    capthick=1.5,
    markersize=6,
    linewidth=2,
    label="Observed Speedup",
)
for x, y in zip(workers, cpu_speedup):
    ax1.annotate(
        f"{y:.2f}x",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax1.set_title(
    "CPU Strong Scaling: Speedup vs Workers",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
ax1.set_xlabel("Number of Processes ($p$)", fontsize=10)
ax1.set_ylabel(r"Speedup $S(p) = T_{seq} / T(p)$", fontsize=10)
ax1.set_xticks(workers)
ax1.set_ylim(0, max(workers) + 0.5)
ax1.legend(loc="upper left", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.6)

# Subplot 2: Efficienza CPU
ax2.axhline(
    100,
    color="k",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Ideal Efficiency (100%)",
)
ax2.plot(
    workers,
    cpu_eff,
    "-s",
    color="#2ca02c",
    linewidth=2,
    markersize=6,
    label="Observed Efficiency",
)
for x, y in zip(workers, cpu_eff):
    ax2.annotate(
        f"{y:.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax2.set_title(
    "CPU Parallel Efficiency vs Workers", fontsize=11, fontweight="bold", pad=12
)
ax2.set_xlabel("Number of Processes ($p$)", fontsize=10)
ax2.set_ylabel(r"Efficiency $E(p) = S(p)/p$ (%)", fontsize=10)
ax2.set_xticks(workers)
ax2.set_ylim(0, 115)
ax2.legend(loc="lower left", frameon=True)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
fig_cpu.savefig("cpu_scaling_benchmarks.png", dpi=300, bbox_inches="tight")
plt.close(fig_cpu)
print("✅ Immagine generata: 'cpu_scaling_benchmarks.png'")

# -----------------------------------------------------------------------------
# 3. GRAFICO 2: GPU BLOCK DIMENSION TUNING
# -----------------------------------------------------------------------------
df_gpu = df[df["Platform"].str.contains("GPU")].copy()
block_sizes = df_gpu["Workers/Units"].astype(int).values
gpu_ms = df_gpu["Mean_Time_s"].values * 1000.0
gpu_std_ms = df_gpu["Std_Time_s"].values * 1000.0
gpu_speedup = df_gpu["Speedup"].values
best_idx = np.argmin(gpu_ms)
x_indices = np.arange(len(block_sizes))

fig_gpu, (ax_g1, ax_g2) = plt.subplots(1, 2, figsize=(12, 4.8))

# Subplot 1: Latenza GPU
bars = ax_g1.bar(
    x_indices,
    gpu_ms,
    yerr=gpu_std_ms,
    capsize=4,
    color="#e67e22",
    alpha=0.85,
    edgecolor="black",
    width=0.55,
)
bars[best_idx].set_color("#27ae60")  # Evidenzia la configurazione migliore

for idx, (bar, val) in enumerate(zip(bars, gpu_ms)):
    label = f"{val:.3f} ms" + ("\n(Best)" if idx == best_idx else "")
    ax_g1.annotate(
        label,
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

ax_g1.set_title(
    "GPU Kernel Latency vs Block Dimension",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
ax_g1.set_xlabel("Threads per Block (Block Size)", fontsize=10)
ax_g1.set_ylabel("Execution Time (ms)", fontsize=10)
ax_g1.set_xticks(x_indices)
ax_g1.set_xticklabels(block_sizes)
ax_g1.set_ylim(min(gpu_ms) * 0.92, max(gpu_ms) * 1.12)
ax_g1.grid(True, linestyle="--", alpha=0.6, axis="y")

# Subplot 2: Speedup GPU
ax_g2.plot(
    x_indices,
    gpu_speedup,
    "-o",
    color="#2980b9",
    linewidth=2,
    markersize=7,
    label="Observed Speedup",
)
ax_g2.plot(
    best_idx,
    gpu_speedup[best_idx],
    "o",
    color="#27ae60",
    markersize=10,
    label=f"Optimal: {block_sizes[best_idx]} TPB",
)

for idx, (x, sp) in enumerate(zip(x_indices, gpu_speedup)):
    ax_g2.annotate(
        f"{sp:,.0f}x",
        (x, sp),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
    )

ax_g2.set_title(
    "GPU Speedup vs Block Dimension", fontsize=11, fontweight="bold", pad=12
)
ax_g2.set_xlabel("Threads per Block (Block Size)", fontsize=10)
ax_g2.set_ylabel(r"Speedup $S = T_{seq} / T_{gpu}$", fontsize=10)
ax_g2.set_xticks(x_indices)
ax_g2.set_xticklabels(block_sizes)
ax_g2.set_ylim(min(gpu_speedup) * 0.95, max(gpu_speedup) * 1.05)
ax_g2.legend(loc="lower right", frameon=True)
ax_g2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
fig_gpu.savefig("gpu_block_tuning.png", dpi=300, bbox_inches="tight")
plt.close(fig_gpu)
print("✅ Immagine generata: 'gpu_block_tuning.png'")

# -----------------------------------------------------------------------------
# 4. GRAFICO 3: CONFRONTO GLOBALE A BARRE (SCALA LOGARITMICA)
# -----------------------------------------------------------------------------
best_cpu = df_cpu.loc[df_cpu["Mean_Time_s"].idxmin()]
best_gpu = df_gpu.loc[df_gpu["Mean_Time_s"].idxmin()]

labels = [
    "CPU Seq\n(1 Core)",
    f"CPU Multi\n({int(best_cpu['Workers/Units'])} Cores)",
    f"GPU CUDA\n({int(best_gpu['Workers/Units'])} TPB)",
]
times = [t_seq, best_cpu["Mean_Time_s"], best_gpu["Mean_Time_s"]]
speedups = [1.0, best_cpu["Speedup"], best_gpu["Speedup"]]
colors = ["#4a5568", "#2b6cb0", "#27ae60"]

fig_comp, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar(
    labels, times, color=colors, width=0.5, edgecolor="black", linewidth=1
)
ax.set_yscale("log")
ax.set_ylabel(
    "Mean Execution Time (seconds, log scale)", fontsize=10, fontweight="bold"
)
ax.set_title(
    "Global Performance Comparison (Log Scale)",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
ax.grid(True, linestyle="--", alpha=0.6, axis="y")

for bar, t, sp in zip(bars, times, speedups):
    t_label = f"{t:.2f} s" if t >= 1.0 else f"{t*1000:.3f} ms"
    sp_label = f"Speedup: {sp:.2f}x" if sp < 1000 else f"Speedup: {sp:,.0f}x"
    ax.annotate(
        f"{t_label}\n({sp_label})",
        xy=(bar.get_x() + bar.get_width() / 2, t),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax.set_ylim(0.0001, 1000)
plt.tight_layout()
fig_comp.savefig("global_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig_comp)
print("✅ Immagine generata: 'global_comparison.png'")