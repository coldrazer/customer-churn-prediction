# Predicting Telecom Customer Churn, and Deciding Who to Call

Gradient boosting vs. a linear baseline, SHAP explanations, and a profit based decision threshold on 7,043 telecom customers (27% churn rate).

## Headline results

* Tuned LightGBM: **ROC-AUC 0.847, PR-AUC 0.664** on the held out test set (logistic regression: 0.841 / 0.633)
* Profit maximizing threshold of **0.38**: contact 32% of customers, reach **70% of churners**, expected profit **$50,863**
* Value aware targeting rule (probability x revenue at risk minus offer cost): **$52,959** expected profit, beating any flat threshold
* SHAP explanations for both global drivers and individual customers

## Approach

1. **Honest metrics first.** With 27% positives, always predicting "stay" scores 73% accuracy, so accuracy is discarded. ROC-AUC for ranking quality, PR-AUC for the minority class, Brier score for calibration.
2. **Model comparison with uncertainty.** 5 fold CV with confidence intervals; LightGBM vs. logistic regression. The intervals are wide enough that small AUC gaps are not over interpreted.
3. **From probabilities to profit.** The retention offer costs $100 per contacted customer. The threshold is chosen to maximize expected profit on out of fold training predictions, never on the test set, and compared against the naive 0.5 default.
4. **Sensitivity analysis.** Key cost assumptions are varied to show where the recommendation holds and where it breaks.

## What the model found

Target retention offers by expected profit, not by a flat probability cutoff. The value aware rule contacts slightly more customers (36%) but earns more ($52,959) by prioritizing high revenue accounts at risk.

## Contents

* `customer_churn_prediction.ipynb` — the full analysis, end to end

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb
```

Random seed 42 throughout. Test set held out until final evaluation.

## Author

Christopher Guerrero, October 2026.
