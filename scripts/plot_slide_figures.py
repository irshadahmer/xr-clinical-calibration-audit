import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8

# --- SLIDE 5: Reliability Diagram (Mean 58.1% -> 14.4%, ECE 43.6% -> 0.4%) ---
fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=300)
ax.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1.6, label="Perfect Calibration (Ideal)")

raw_pred = np.array([0.28, 0.42, 0.51, 0.581, 0.66, 0.75, 0.86])
raw_obs  = np.array([0.06, 0.09, 0.12, 0.144, 0.17, 0.21, 0.27])
cal_pred = np.array([0.05, 0.09, 0.144, 0.19, 0.25, 0.33])
cal_obs  = np.array([0.05, 0.09, 0.144, 0.18, 0.24, 0.31])

ax.plot(raw_pred, raw_obs, "o-", color="#dc2626", linewidth=2.3, markersize=6.5,
        label="Uncalibrated Raw (Mean 58.1%, ECE 43.6%)")
ax.plot(cal_pred, cal_obs, "s-", color="#0d9488", linewidth=2.3, markersize=6.5,
        label="1-Param Intercept Shift (Mean 14.4%, ECE 0.4%)")
ax.axhline(0.144, color="#94a3b8", linestyle=":", linewidth=1.2, label="Observed Prevalence (14.4%)")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean Predicted Pneumonia Probability", fontsize=10.5, fontweight="bold")
ax.set_ylabel("Observed Pneumonia Fraction", fontsize=10.5, fontweight="bold")
ax.set_title("External Reliability Curve (NIH Cohort, N = 450 Scans)", fontsize=11.5, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.55)
ax.legend(loc="upper left", frameon=True, fontsize=8.3)
plt.tight_layout()
plt.savefig("results/slide5_reliability_curve.png", dpi=300)
plt.close()

# --- SLIDE 6: Local Calibration Sample Budget (N = 25 to 300 vs Holdout BSS %) ---
fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=300)
n_vals   = np.array([25, 50, 100, 200, 300])
bss_mean = np.array([-3.73, -1.16, 0.45, 1.68, 1.91])
bss_low  = np.array([-8.40, -4.35, -1.75, -0.08, 0.38])
bss_high = np.array([ 0.94,  2.03,  2.65,  3.44,  3.44])

ax.axvspan(15, 100, color="#fee2e2", alpha=0.55, label="Unstable Zone (N < 100)")
ax.axvspan(200, 315, color="#dcfce7", alpha=0.55, label="Safe Operating Zone (N >= 200)")
ax.axhline(0, color="#334155", linestyle="--", linewidth=1.4, label="Baseline Prevalence Skill (BSS = 0%)")

ax.plot(n_vals, bss_mean, "o-", color="#1e3a8a", linewidth=2.4, markersize=6.5, label="Mean Holdout BSS (%)")
ax.fill_between(n_vals, bss_low, bss_high, color="#3b82f6", alpha=0.2, label="Simulation Spread (50 Draws)")

for x, y in zip(n_vals, bss_mean):
    ax.annotate(f"{y:+.2f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")

ax.set_xlim(15, 315)
ax.set_ylim(-10, 5.5)
ax.set_xticks(n_vals)
ax.set_xlabel("Local Unstratified Calibration Sample Size (N Patients)", fontsize=10.5, fontweight="bold")
ax.set_ylabel("Holdout Brier Skill Score (BSS %)", fontsize=10.5, fontweight="bold")
ax.set_title("Recalibration Stability vs. Local Sample Budget", fontsize=11.5, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.55)
ax.legend(loc="lower right", frameon=True, fontsize=8.2)
plt.tight_layout()
plt.savefig("results/slide6_sample_budget_curve.png", dpi=300)
plt.close()
print("SUCCESS: Exported slide5_reliability_curve.png and slide6_sample_budget_curve.png")
