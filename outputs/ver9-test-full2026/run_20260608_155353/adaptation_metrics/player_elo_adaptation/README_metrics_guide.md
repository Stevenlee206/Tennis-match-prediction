# Player-Centric Elo Adaptation Metrics Guide

Read this file before interpreting the CSV tables in this directory.

## Research Question

These metrics test whether PCN learns changing player dynamics better than NN/ResNet during Finetune, Online, and Ultimate Streaming updates.

High Elo reliance alone is not evidence of continual adaptation. Strong evidence requires improvement for rising, declining, high-Elo-change, volatile, or Elo-disagreement cases.

## Direction Convention

- Higher is better: Accuracy, F1, Elo Residual Gain, Adaptation Gain, trend correlation, directional alignment, PCN advantage.
- Lower is better: Log Loss, Brier Score, ECE, samples-to-catch-up.
- Gain values are normalized so positive generally means improvement.
- In `pcn_elo_continual_advantage_table`, positive always means PCN is better.

## Player Elo Long Format

File: `player_elo_long_format.csv`

Each original match becomes two rows, one from each player's perspective.

- `player_win`: whether the current player won.
- `player_win_prob`: model probability that the current player wins.
- `player_pred_win`: model hard-label prediction for the current player.
- `player_elo`, `opponent_elo`: pre-match Elo ratings.
- `elo_diff_player`: player Elo minus opponent Elo.
- `player_p_elo`: Elo baseline probability that the current player wins.
- `player_elo_delta`: change in the player's Elo since their previous match.
- `abs_elo_delta`: magnitude of the Elo change.
- `player_elo_quantile`: low, mid-low, mid-high, or high Elo group.
- `player_trend_group`: rising, declining, or stable based on Elo change direction.
- `high_delta_group`: whether the Elo change is in the top 25%.
- `volatility_group`: whether recent Elo-change volatility is in the top 25%.

This is a diagnostic dataset. It should not be used as model training input.

## Elo-Conditioned Player Adaptation Gain

File: `elo_conditioned_adaptation_gain_table.csv`

Measures whether an adaptive mode improves over Static inside each Elo-related player group.

- `static_accuracy`: Static accuracy in the group.
- `adaptive_accuracy`: Finetune, Online, or Ultimate Streaming accuracy.
- `elo_pag_accuracy = adaptive_accuracy - static_accuracy`.
- `elo_pag_logloss = static_logloss - adaptive_logloss`.
- `elo_pag_brier = static_brier - adaptive_brier`.
- `group_type`: which grouping produced the row, such as trend, Elo quantile, high delta, or volatility.
- `elo_group`: the specific group value.

Interpretation:

- Positive gain means the adaptive mode improved over Static.
- PCN has stronger player-centric adaptation when its gains are larger than NN's, especially for rising, declining, high-delta, and high-volatility groups.

## Elo Residual Gain

File: `elo_residual_gain_table.csv`

Measures whether the model predicts better probabilities than the Elo baseline.

- `LL_Elo`: Elo baseline Log Loss.
- `LL_Model`: model Log Loss.
- `EloResidualGain_LL = LL_Elo - LL_Model`.
- `Brier_Elo`: Elo baseline Brier Score.
- `Brier_Model`: model Brier Score.
- `EloResidualGain_Brier = Brier_Elo - Brier_Model`.

Interpretation:

- Positive residual gain means the model improves on Elo.
- Negative residual gain means Elo alone is a better probabilistic predictor.
- Compare PCN and NN within the same mode and group.

## High Elo-Change Metrics

File: `high_elo_change_metrics_table.csv`

Evaluates matches involving players whose absolute Elo change is at or above the 75th percentile.

- `accuracy`, `logloss`, `brier`, `ece`: performance inside the group.
- `adaptation_gain_accuracy`: mode accuracy minus Static accuracy.
- `relative_adaptation_gain_accuracy`: percentage improvement relative to Static accuracy.
- `adaptation_gain_logloss`: Static Log Loss minus mode Log Loss.
- `adaptation_gain_brier`: Static Brier minus mode Brier.

Interpretation:

- Positive adaptation gains indicate successful adaptation to rapidly changing players.
- Lower ECE indicates more reliable confidence.

## Elo Disagreement Adaptation

File: `elo_disagreement_adaptation_table.csv`

Tests whether the model learns when to deviate from Elo.

- `elo_disagreement_rate`: fraction of predictions that disagree with Elo.
- `acc_agree_elo`: accuracy when model and Elo agree.
- `acc_disagree_elo`: accuracy when model and Elo disagree.
- `ll_disagree_elo`, `brier_disagree_elo`: probability quality on disagreement cases.
- `disagree_gain_accuracy`: adaptive disagreement accuracy minus Static disagreement accuracy.
- `disagree_gain_logloss`: Static disagreement Log Loss minus adaptive disagreement Log Loss.
- `disagree_gain_brier`: Static disagreement Brier minus adaptive disagreement Brier.

Interpretation:

- High disagreement rate alone is not good.
- Strong evidence requires positive disagreement gain and good `acc_disagree_elo`.

## Elo Trend Alignment

File: `elo_trend_alignment_table.csv`

Checks whether changes in model win probability follow changes in player Elo.

- `trend_corr`: correlation between probability change and Elo change.
- `directional_alignment`: fraction where probability and Elo move in the same direction.
- `n_observations`: number of valid change pairs.

Interpretation:

- Higher values mean model probabilities track Elo dynamics more closely.
- This does not automatically mean better match prediction; inspect it together with Accuracy and Elo Residual Gain.

## Samples-to-Catch-Up

File: `samples_to_catch_up_elo_table.csv`

Measures how many subsequent player matches an adaptive mode needs to outperform Static after a rising, declining, or high-delta event.

- `mean_samples_to_catch_up_acc`, `median_samples_to_catch_up_acc`: matches needed to achieve the Accuracy gain threshold.
- `percentage_players_caught_up_acc`: fraction of players that reached the threshold.
- Log Loss columns provide the equivalent probabilistic catch-up measurement.

Interpretation:

- Lower samples-to-catch-up is better.
- Higher percentage caught up is better.
- A low average based on very few players should not be treated as strong evidence.

## Elo Volatility Robustness

File: `elo_volatility_robustness_table.csv`

Evaluates robustness for players with unstable recent Elo changes.

- `accuracy`, `logloss`, `brier`: performance in each volatility group.
- `adaptation_gain_accuracy`, `adaptation_gain_logloss`: improvement over Static.
- `elo_residual_gain_logloss`: improvement over Elo baseline.
- `acc_disagree_elo`: accuracy when disagreeing with Elo.

Interpretation:

- PCN is stronger for volatile players only when it has positive adaptation gain and compares favorably against both NN and Elo.

## PCN vs NN Continual Advantage

File: `pcn_elo_continual_advantage_table.csv`

This is the primary compact comparison table.

- `PCN value`: PCN metric value.
- `NN value`: NN/ResNet metric value.
- `PCN advantage`: direction-normalized difference.

Interpretation:

- Positive `PCN advantage` always means PCN is better.
- Negative means NN/ResNet is better.
- Use group and mode columns to identify where the advantage occurs.

## Recommended Conclusion Logic

Claim stronger PCN player-centric continual adaptation only when several of these are true:

1. PCN has larger Elo-conditioned adaptation gains than NN.
2. PCN has positive and larger Elo Residual Gain.
3. PCN improves on high Elo-change or high-volatility players.
4. PCN improves Accuracy when disagreeing with Elo.
5. PCN tracks player Elo trends better.
6. PCN needs fewer samples to catch up after player trend changes.

If PCN improves hard-label Accuracy but NN has lower Log Loss, Brier, or ECE, conclude that PCN is stronger for hard-label continual adaptation while NN remains better calibrated.
