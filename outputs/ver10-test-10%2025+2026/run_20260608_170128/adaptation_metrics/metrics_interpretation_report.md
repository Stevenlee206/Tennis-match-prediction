# Metrics Interpretation Report

Use this file first. The CSV files are detailed evidence tables; this report turns them into research conclusions.

## How To Read The New Metrics

- Accuracy/F1/Balanced Accuracy answer hard-label winner prediction.
- Log Loss/Brier/ROC AUC/PR AUC answer probabilistic forecasting quality.
- Adaptation Gain answers whether a mode improved over Static for the same model.
- Streaming Advantage answers whether PCN beats NN/ResNet in Ultimate Streaming.
- Adaptation Gain Advantage answers whether PCN benefits more from streaming than NN/ResNet.
- ECE and overconfident-wrong rate answer calibration and confidence reliability.
- Elo/Rank disagreement accuracy answers whether the model does more than copy heuristics.

## Executive Conclusions

### Does PCN improve hard-label winner prediction in Ultimate Streaming?

**Evidence:** PCN US Acc=68.02%, NN US Acc=67.20%; PCN Static->US Gain=+1.10 pp.

**Conclusion:** PCN has higher Ultimate Streaming accuracy.

**Recommended claim:** Claim PCN is better for streaming hard-label prediction.

### Does PCN benefit more from sequential updates than the baseline?

**Evidence:** PCN Static->US Gain=+1.10 pp; NN Static->US Gain=-0.21 pp; Gain Advantage=+1.31 pp.

**Conclusion:** PCN benefits more from streaming adaptation.

**Recommended claim:** Use this as the main PCN adaptation claim.

### Is PCN also better as a probabilistic forecaster?

**Evidence:** US LogLoss: PCN=0.6083, NN=0.5990; Brier: PCN=0.2103, NN=0.2068; ROC AUC: PCN=0.7328, NN=0.7395.

**Conclusion:** NN remains stronger on at least part of probabilistic forecasting.

**Recommended claim:** Present PCN as a hard-label/streaming adaptation model, not necessarily best calibrated.

### Is calibration a limitation?

**Evidence:** US ECE: PCN=0.0344, NN=0.0154. Lower is better.

**Conclusion:** PCN calibration is weaker in this run.

**Recommended claim:** Consider temperature scaling, Platt scaling, or isotonic regression.

### What is the direct PCN streaming advantage over the baseline?

**Evidence:** Accuracy advantage=+0.83 pp; LogLoss advantage=-0.0092; Brier advantage=-0.0035. Positive means PCN is better.

**Conclusion:** This row is the direct head-to-head Ultimate Streaming comparison.

**Recommended claim:** Use it as the compact model-comparison table in reports.

### Does PCN learn beyond Elo/Rank heuristics?

**Evidence:** Accuracy when disagreeing with Elo: PCN=57.67%, NN=54.27%.

**Conclusion:** PCN is stronger on Elo-disagreement cases.

**Recommended claim:** Use Elo-disagreement accuracy to support or limit the temporal-pattern claim.

### Is PCN update-efficient?

**Evidence:** No update_time/update_time_ms columns were available.

**Conclusion:** Efficiency cannot be concluded from this run.

**Recommended claim:** Add update-time logging before making compute-efficiency claims.
