# External Clinical Calibration Audit: CheXpert-Trained CXR Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical clinical calibration and generalizability audit evaluating a DenseNet-121 classifier trained exclusively on Stanford CheXpert (`densenet121-res224-chex`) against an unseen external cohort from NIH ChestX-ray14 with realistic, heterogeneous non-pneumonia controls.

---

## 📌 Summary of Core Findings

1. **Prevalence-Induced Miscalibration:**
   When deployed on an external cohort with 14.4% observed pneumonia prevalence, the raw model output predicted an average risk of **58.1%**, producing an uncalibrated Expected Calibration Error (ECE) of **43.6%** and a Brier Skill Score (BSS) of **-151.9%** relative to the sample baseline.
2. **Prior Shift Dominance:**
   An intercept-only (1-parameter) recalibration reduced ECE to **0.4%** ($0.0042$), matching full 2-parameter Platt scaling ($0.0169$). This confirms the calibration distortion was almost entirely driven by a base-rate log-odds offset rather than non-linear logit deformation.
3. **Absence of Usable Skill Post-Recalibration:**
   While recalibration aligned predicted probabilities with observed event frequencies, Brier Skill Scores remained modest ($+1.8\%$ to $+1.9\%$, with 95% confidence intervals crossing zero). Monotonic recalibration preserves rank ordering; on realistic non-pneumonia controls, discrimination was modest ($\text{AUROC} = 0.6394$).
4. **Validation Budget Constraints:**
   In patient-sampled holdout simulations, small unstratified calibration budgets frequently yielded negative skill on holdout sets due to the sparsity of positive cases in random draws.
5. **Exploratory Demographic Disparities:**
   - **Sex Parity:** No demonstrable difference between male and female strata ($\Delta\text{AUROC} = +0.0279$, 95% CI: $[-0.1132, 0.1673]$, unadjusted $p = 0.694$).
   - **Age Disparity:** Significant discrimination degradation in older patients ($\text{AUROC}_{<50} = 0.7725$ vs. $\text{AUROC}_{\ge 50} = 0.6083$, $\Delta\text{AUROC} = +0.1642$, 95% CI: $[0.0289, 0.2986]$, unadjusted $p = 0.020$).

---

## 📊 Master Results Table

*Evaluated on $N = 450$ radiographs from 289 unique patients (65 pneumonia positives [14.4%], 385 heterogeneous non-pneumonia controls). Confidence intervals derived via 1,000 patient-clustered bootstrap resamples; evaluation via 5-fold StratifiedGroupKFold on patient IDs.*

| Metric | Uncalibrated (Raw) | Intercept-Only ($x + b$) | Full Platt Scaling ($ax + b$) |
| :--- | :--- | :--- | :--- |
| **AUROC (Rank Invariant)** | 0.6394 [0.5629, 0.7139] | 0.6394 [0.5629, 0.7139] | 0.6394 [0.5629, 0.7139] |
| **ECE (10-bin)** | 0.4363 [0.3977, 0.4689] | **0.0042** [0.0025, 0.0449] | 0.0169 [0.0080, 0.0594] |
| **Brier Score** | 0.3129 [0.2510, 0.4001] | 0.1218 [0.1068, 0.1378] | 0.1219 [0.1068, 0.1384] |
| **Brier Skill Score (BSS)** | -151.93% [-222.04%, -102.11%] | **+1.91%** [-0.66%, +3.64%] | +1.83% [-1.19%, +3.93%] |

---

## 🔬 Exploratory Subgroup Disaggregation

| Subgroup | Radiographs | Unique Patients | Positives ($N$, %) | Negatives | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Male** | 240 | 158 | 41 (17.1%) | 199 | 0.6482 |
| **Female** | 210 | 131 | 24 (11.4%) | 186 | 0.6203 |
| **Age $< 50$** | 196 | 134 | 37 (18.9%) | 159 | **0.7725** |
| **Age $\ge 50$** | 254 | 155 | 28 (11.0%) | 226 | **0.6083** |

---

## 📁 Repository Structure

cxr-clinical-calibration-audit/
├── results/
│   ├── chexpert_nih_realistic_controls.csv  # Verified model predictions & metadata
│   └── figure1_calibration_audit.png        # Publication reliability curves
├── scripts/
│   ├── run_rigorous_nih_audit.py     # Patient-grouped inference & recalibration
│   ├── run_intercept_budget_audit.py # Sample size stability curves
│   └── plot_final_figures.py         # Dynamic publication plotting script
├── .gitignore
├── README.md
└── requirements.txt 
---

## 🚀 Quick Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run patient-clustered inference and recalibration
python scripts/run_rigorous_nih_audit.py

# 3. Generate publication reliability figures
python scripts/plot_final_figures.py
