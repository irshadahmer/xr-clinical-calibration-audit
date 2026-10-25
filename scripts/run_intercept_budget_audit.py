import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

df = pd.read_csv("results/chexpert_nih_realistic_controls.csv")
y_true = df["ground_truth"].values
logits = df["logit"].values
patient_ids = df["Patient ID"].values
unique_pids = np.unique(patient_ids)
pid_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_pids}

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def bss(y_t, y_p):
    bs = brier_score_loss(y_t, y_p)
    base = np.mean(y_t)
    bs_ref = np.mean((base - y_t) ** 2)
    return 1.0 - (bs / bs_ref)

print("=" * 70)
print("INTERCEPT-ONLY PATIENT BUDGET AUDIT (50 Resamples Each)")
print(f"{'Budget (N)':<12} {'Mean BSS':<12} {'BSS 95% Interval':<24} {'% Resamples BSS < 0'}")
print("-" * 70)

rng = np.random.RandomState(42)
budgets = [25, 50, 100, 150, 200, 300]

for b in budgets:
    bsses = []
    for _ in range(50):
        shuffled_pids = rng.permutation(unique_pids)
        cal_idx = []
        for p in shuffled_pids:
            cal_idx.extend(pid_to_indices[p])
            if len(cal_idx) >= b:
                cal_idx = np.array(cal_idx[:b])
                break

        test_idx = np.setdiff1d(np.arange(len(y_true)), cal_idx)

        def nll(b_val, x_tr, y_tr):
            p = np.clip(sigmoid(x_tr + b_val), 1e-15, 1 - 1e-15)
            return -np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p))

        res = minimize(nll, x0=[0.0], args=(logits[cal_idx], y_true[cal_idx]), method='Nelder-Mead')
        p_test = sigmoid(logits[test_idx] + res.x[0])
        bsses.append(bss(y_true[test_idx], p_test))

    pct_neg = np.mean(np.array(bsses) < 0) * 100
    m_bss = np.mean(bsses)
    lo_b, hi_b = np.percentile(bsses, 2.5), np.percentile(bsses, 97.5)
    print(f"{b:<12} {m_bss*100:6.2f}%      [{lo_b*100:5.2f}%, {hi_b*100:5.2f}%]        {pct_neg:5.1f}%")
