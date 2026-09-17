# Transaction Fraud Risk Scoring

**Business question:** How should a fraud team set a review threshold to 
minimize total operational cost, while identifying the transaction and 
account characteristics that drive fraud risk?

## Overview

This project builds an interpretable fraud risk scoring model on the 
IEEE-CIS Fraud Detection dataset (590K+ transactions, 3.5% fraud rate), 
and translates model output into an operational decision framework — not 
just a classifier, but a recommendation a fraud strategy team could act on.

## What's inside

- **`notebooks/fraud-risk-monitoring.ipynb`** — full analysis: EDA, feature 
  engineering, modeling, validation, and cost-based threshold optimization
- **`docs/`** — saved charts (EDA summary, precision-recall curve, cost 
  threshold curve)

## Key findings

- **Product category and card type (credit vs. debit) are the strongest 
  fraud drivers** — Product C shows ~12% fraud vs. 3-4x lower for other 
  products; credit transactions show 6.68% fraud vs. 2.43% for debit.
- **Logistic regression achieved 0.746 ROC-AUC**, benchmarked against a 
  Random Forest (0.880 ROC-AUC) to quantify the interpretability/performance 
  tradeoff of choosing a linear, explainable model.
- **Cost-based threshold optimization** identified 0.40 as the operating 
  point minimizing total expected cost, reducing cost by $19,560 (1.6%) 
  versus a naive 0.5 threshold on the test sample — a modest percentage 
  driven by the model's precision ceiling (~19% even at strict thresholds), 
  motivating a tiered review framework over a single flag/no-flag rule.

Full findings, methodology, and the Operational Recommendation Memo are in 
the notebook.

## Methodology notes

- Train/test split performed **before** engineering account-history features 
  (`amt_deviation`, `card1_txn_count_total`) to avoid data leakage — these 
  are computed from training data only and mapped onto the test set.
- `card1` used as a proxy for account identity (no true account ID exists in 
  this dataset); its highly skewed transaction-count distribution suggests 
  high-frequency values may represent shared BIN ranges rather than 
  individual cards — a documented limitation, not a design flaw.
- The 339 anonymized Vesta `V`-features were excluded from the interpretable 
  feature set (undocumented semantics) but implicitly available to the 
  Random Forest benchmark for comparison.
- Illustrative costs ($500/missed fraud, $15/false positive) are placeholders 
  from the project brief, not validated business figures.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Download the IEEE-CIS Fraud Detection dataset from Kaggle 
(`train_transaction.csv`, `train_identity.csv`) into a local `data/` folder 
(not included in this repo due to size).
