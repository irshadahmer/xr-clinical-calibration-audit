import os
import numpy as np
import pandas as pd
import torch
import skimage.io
import skimage.transform
import torchxrayvision as xrv
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

img_dir = "data/nih/images"
csv_path = "data/nih/Data_Entry_2017.csv"

if not os.path.exists(csv_path):
    print("Notice: Raw NIH dataset not found locally. Loading cached results.")
    if os.path.exists("results/chexpert_nih_realistic_controls.csv"):
        cohort = pd.read_csv("results/chexpert_nih_realistic_controls.csv")
    else:
        raise FileNotFoundError("Neither raw data nor cached results CSV found.")
else:
    df_meta = pd.read_csv(csv_path)
    local_files = set(os.listdir(img_dir))
    df_local = df_meta[df_meta["Image Index"].isin(local_files)].copy()

    pneu_df = df_local[df_local["Finding Labels"].str.contains("Pneumonia", na=False)].copy()
    pneu_df["ground_truth"] = 1

    neg_pool = df_local[~df_local["Finding Labels"].str.contains("Pneumonia", na=False)].copy()
    neg_df = neg_pool.sample(n=385, random_state=42).copy()
    neg_df["ground_truth"] = 0

    cohort = pd.concat([pneu_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device)
    model.eval()
    pneu_idx = model.pathologies.index("Pneumonia")

    logits, raw_probs = [], []
    with torch.no_grad():
        for _, row in cohort.iterrows():
            img_p = os.path.join(img_dir, row["Image Index"])
            img = skimage.io.imread(img_p)
            if len(img.shape) > 2:
                img = img.mean(axis=2)
            img = skimage.transform.resize(img, (224, 224), preserve_range=True, anti_aliasing=True)
            img = xrv.datasets.normalize(img, 255)
            img_t = torch.from_numpy(img[None, None, ...]).float().to(device)
            out = model(img_t)
            logits.append(out[0, pneu_idx].item())
            raw_probs.append(torch.sigmoid(out[0, pneu_idx]).item())

    cohort["logit"] = logits
    cohort["raw_prob"] = raw_probs

y_true = cohort["ground_truth"].values
p_raw = cohort["raw_prob"].values
logits_arr = cohort["logit"].values
patient_ids = cohort["Patient ID"].values

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
p_platt = np.zeros_like(p_raw)
p_intercept = np.zeros_like(p_raw)

for train_idx, test_idx in sgkf.split(logits_arr.reshape(-1, 1), y_true, groups=patient_ids):
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(logits_arr[train_idx].reshape(-1, 1), y_true[train_idx])
    p_platt[test_idx] = lr.predict_proba(logits_arr[test_idx].reshape(-1, 1))[:, 1]

    def nll_intercept(b, x_tr, y_tr):
        p = np.clip(sigmoid(x_tr + b), 1e-15, 1 - 1e-15)
        return -np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p))

    res = minimize(nll_intercept, x0=[0.0], args=(logits_arr[train_idx], y_true[train_idx]), method='Nelder-Mead')
    p_intercept[test_idx] = sigmoid(logits_arr[test_idx] + res.x[0])

cohort["prob_platt"] = p_platt
cohort["prob_intercept"] = p_intercept
os.makedirs("results", exist_ok=True)
cohort.to_csv("results/chexpert_nih_realistic_controls.csv", index=False)

def compute_ece(y_t, y_p, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i+1]
        m = (y_p >= low) & (y_p <= high) if i == n_bins - 1 else (y_p >= low) & (y_p < high)
        if np.sum(m) > 0:
            ece += (np.sum(m) / len(y_t)) * np.abs(np.mean(y_t[m]) - np.mean(y_p[m]))
    return ece

def bss(y_t, y_p):
    bs = brier_score_loss(y_t, y_p)
    base = np.mean(y_t)
    bs_ref = np.mean((base - y_t) ** 2)
    return 1.0 - (bs / bs_ref)

print("=" * 70)
print(f"Sample N = {len(y_true)} | Observed Prevalence = {y_true.mean()*100:.1f}%")
print(f"AUROC:               {roc_auc_score(y_true, p_raw):.4f}")
print(f"Uncalibrated ECE:    {compute_ece(y_true, p_raw):.4f} | BSS: {bss(y_true, p_raw)*100:.2f}%")
print(f"Intercept-Only ECE:  {compute_ece(y_true, p_intercept):.4f} | BSS: {bss(y_true, p_intercept)*100:.2f}%")
print(f"Platt Scaled ECE:    {compute_ece(y_true, p_platt):.4f} | BSS: {bss(y_true, p_platt)*100:.2f}%")
print("=" * 70)
