import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv("results/chexpert_nih_realistic_controls.csv")
y_true = df["ground_truth"].values

def compute_ece(y_t, y_p, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i+1]
        m = (y_p >= low) & (y_p <= high) if i == n_bins - 1 else (y_p >= low) & (y_p < high)
        if np.sum(m) > 0:
            ece += (np.sum(m) / len(y_t)) * np.abs(np.mean(y_t[m]) - np.mean(y_p[m]))
    return ece

ece_raw = compute_ece(y_true, df["raw_prob"].values)
ece_int = compute_ece(y_true, df["prob_intercept"].values)
ece_platt = compute_ece(y_true, df["prob_platt"].values)
auc_val = roc_auc_score(y_true, df["raw_prob"].values)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

pt_raw, pp_raw = calibration_curve(y_true, df["raw_prob"], n_bins=10)
pt_int, pp_int = calibration_curve(y_true, df["prob_intercept"], n_bins=10)
pt_platt, pp_platt = calibration_curve(y_true, df["prob_platt"], n_bins=10)

ax1.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Ideal Calibration")
ax1.plot(pp_raw, pt_raw, "s-", color="#d95f02", lw=1.8, label=f"Uncalibrated (ECE = {ece_raw*100:.1f}%)")
ax1.plot(pp_int, pt_int, "^-", color="#7570b3", lw=1.8, label=f"Intercept-Only (ECE = {ece_int*100:.1f}%)")
ax1.plot(pp_platt, pt_platt, "o-", color="#1b9e77", lw=1.8, label=f"Platt Scaled (ECE = {ece_platt*100:.1f}%)")

ax1.axvline(df["raw_prob"].mean(), color="#d95f02", linestyle=":", alpha=0.6, label=f"Mean Pred Risk ({df['raw_prob'].mean()*100:.1f}%)")
ax1.axhline(y_true.mean(), color="#333333", linestyle=":", alpha=0.6, label=f"Observed Prev ({y_true.mean()*100:.1f}%)")

ax1.set_title(f"A: External Calibration (AUROC = {auc_val:.3f})", fontsize=11, fontweight="bold")
ax1.set_xlabel("Mean Predicted Probability", fontsize=10)
ax1.set_ylabel("Observed Fraction of Positives", fontsize=10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.grid(alpha=0.25)
ax1.legend(loc="upper left", fontsize=8)

m_young = df["Patient Age"] < 50
m_old = df["Patient Age"] >= 50
auc_y = roc_auc_score(df.loc[m_young, "ground_truth"], df.loc[m_young, "raw_prob"])
auc_o = roc_auc_score(df.loc[m_old, "ground_truth"], df.loc[m_old, "raw_prob"])

pt_y, pp_y = calibration_curve(df.loc[m_young, "ground_truth"], df.loc[m_young, "prob_intercept"], n_bins=6)
pt_o, pp_o = calibration_curve(df.loc[m_old, "ground_truth"], df.loc[m_old, "prob_intercept"], n_bins=6)

ax2.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Ideal Calibration")
ax2.plot(pp_y, pt_y, "^-", color="#2b83ba", lw=1.8, label=f"Age < 50 (AUROC = {auc_y:.3f})")
ax2.plot(pp_o, pt_o, "v-", color="#d7191c", lw=1.8, label=f"Age >= 50 (AUROC = {auc_o:.3f})")

ax2.set_title("B: Exploratory Age Stratification (Intercept-Adjusted)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Mean Predicted Probability", fontsize=10)
ax2.set_ylabel("Observed Fraction of Positives", fontsize=10)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.grid(alpha=0.25)
ax2.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("results/figure1_calibration_audit.png")
print("Saved final publication figures to results/figure1_calibration_audit.png")
