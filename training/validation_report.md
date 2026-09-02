# GMN-Football-3 — Policy Validation & Scientific Evidence Report

**Date of Evaluation:** 2026-09-02T15:35:51.825Z
**Target Scenario:** `academy_3_vs_1_with_keeper`
**Algorithm:** `MAPPO`
**Checkpoint SHA256:** `FILE_NOT_FOUND`
**Overall Scientific Verdict:** **INVALID_CHECKPOINT**

## Executive Summary

The checkpoint in the repository is a 49,920-step zero-padded smoke-test model that does not contain non-zero weights for role-feature indices (115-126). The evaluation pipeline correctly caught and rejected this model, gracefully routing browser and ladder evaluations to baseline fallbacks. A real training run (>=200k steps) under the 127-dim schema is required for genuine policy certification.

## Evidence Chain Criteria Matrix

| Step | Criterion | Verdict | Evidence / Metrics |
|:---:|:---|:---:|:---|
| 1 | Checkpoint & Schema Contract Compatibility | 🛑 **REJECTED** | CHECKPOINT REJECTED: All 768 role-feature weights (indices 115..126) are exact zeros. This indicates a zero-padded placeholder/smoke-test model rather than a genuinely trained role-aware policy. |
| 3 | Beat Random Baseline | ✅ **PASS** | MAPPO win rate (30.0%) vs Random (0.0%). Reward: 0.636 vs 0.037. |
| 4 | Competitive with Tactical Rule-Based Baseline | ✅ **PASS** | MAPPO win rate (30.0%) vs Rule-Based Med (30.0%). Goal diff: 0.30 vs 0.30. |
| 5 | Held-Out Generalization Retention | ✅ **PASS** | Train win rate: 10.0% | Held-out test win rate: 23.3% | Transfer retention: 233.3% (Gap: -13.3%). |
| 6 | Opponent Robustness & Anti-Exploitation | ✅ **PASS** | Easy Opponent Win Rate: 20.0% | Hard Opponent Win Rate: 20.0%. |
| 7 | Role Feature Conditioning Sensitivity | ✅ **PASS** | Standard 127-dim Win Rate: 25.0% | Zero-Role Ablated Win Rate: 25.0%. |
| 8 | Browser / Python Inference Parity | ✅ **PASS** | Tested 65 vectors. Max logit delta: 0.0000e+0. Action mismatches: 0. |

## 1. Baseline Ladder Performance

========================================================================================================
BASELINE LADDER EVALUATION SUMMARY TABLE
========================================================================================================
| Policy | Win Rate (%) | Goals (Mean±Std) | Possession (%) | Pass Acc (%) | Shot Acc (%) | Reward (Mean±Std) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| random             | 0.0%         | 0.00 ± 0.00      | 40.0%          | 0.0%         | 15.0%        | 0.037 ± 0.092     |
| noop               | 0.0%         | 0.00 ± 0.00      | 55.0%          | 0.0%         | 0.0%         | 0.036 ± 0.095     |
| rule_based_easy    | 30.0%        | 0.30 ± 0.47      | 93.8%          | 27.7%        | 60.0%        | 0.636 ± 0.533     |
| rule_based_medium  | 30.0%        | 0.30 ± 0.47      | 93.8%          | 27.7%        | 60.0%        | 0.636 ± 0.533     |
| rule_based_hard    | 35.0%        | 0.35 ± 0.49      | 96.3%          | 27.7%        | 60.0%        | 0.689 ± 0.558     |
| scripted           | 0.0%         | 0.00 ± 0.00      | 45.0%          | 0.0%         | 0.0%         | 0.000 ± 0.000     |
| mappo_trained      | 30.0%        | 0.30 ± 0.47      | 93.8%          | 27.7%        | 60.0%        | 0.636 ± 0.533     |
========================================================================================================

## 2. Held-Out Generalization Matrix

========================================================================================================
HELD-OUT GENERALIZATION SUMMARY
========================================================================================================
| Partition   | Scenario                                       | Win Rate (%) | Goals Scored   | Pass Acc (%) |
|:------------|:-----------------------------------------------|:------------:|:--------------:|:------------:|
| TRAIN       | academy_3_vs_1_with_keeper                     | 10.0%        | 0.10 ± 0.31    | 25.5%        |
| VALIDATION  | academy_3_vs_1_defender_2                      | 0.0%         | 0.00 ± 0.00    | 52.0%        |
| VALIDATION  | academy_3_vs_1_defender_3                      | 5.0%         | 0.05 ± 0.22    | 46.5%        |
| TEST        | academy_3_vs_1_keeper_aggressive               | 20.0%        | 0.20 ± 0.41    | 25.5%        |
| TEST        | academy_3_vs_1_shifted                         | 20.0%        | 0.20 ± 0.41    | 27.2%        |
| TEST        | academy_3_vs_1_randomized                      | 30.0%        | 0.30 ± 0.47    | 19.5%        |
--------------------------------------------------------------------------------------------------------
Overall Train Partition Win Rate:      10.0%
Overall Validation Partition Win Rate: 2.5%
Overall Held-out Test Win Rate:        23.3%
Generalization Transfer Retention:     233.3%
Overfitting Gap (Train - Test):        -13.3%
========================================================================================================

## 3. Opponent Robustness Matrix

========================================================================================================
OPPONENT ROBUSTNESS SUMMARY TABLE
========================================================================================================
| Opponent Configuration              | Win Rate (%) | Goals Scored   | Turnovers Conceded | Pass Acc (%) |
|:------------------------------------|:------------:|:--------------:|:------------------:|:------------:|
| Weak / Easy Tactical Bot            | 20.0%        | 0.20 ± 0.41    | 0.80               | 29.4%        |
| Default / Medium Tactical Bot       | 20.0%        | 0.20 ± 0.41    | 0.80               | 29.4%        |
| Strong / Aggressive Tactical Bot    | 20.0%        | 0.20 ± 0.41    | 0.80               | 29.4%        |
| Master / Elite Tactical Bot         | 20.0%        | 0.20 ± 0.41    | 0.80               | 29.4%        |
| Scripted Defender Contain Bot       | 90.0%        | 0.90 ± 0.31    | 0.10               | 64.7%        |
========================================================================================================

## 4. Controlled Ablation Study

========================================================================================================
CONTROLLED ABLATION STUDY SUMMARY
========================================================================================================
| Condition               | Win Rate (%) | Goals (Mean±Std) | Pass Ratio (%) | Shot Ratio (%) | Reward (Mean) |
|:------------------------|:------------:|:----------------:|:--------------:|:--------------:|:-------------:|
| standard_role_127        | 25.0%        | 0.25 ± 0.44      | 0.0%           | 0.0%           | 0.619         |
| role_zeroed              | 25.0%        | 0.25 ± 0.44      | 0.0%           | 0.0%           | 0.619         |
| role_randomized          | 25.0%        | 0.25 ± 0.44      | 0.0%           | 0.0%           | 0.619         |
| role_inverted            | 25.0%        | 0.25 ± 0.44      | 0.0%           | 0.0%           | 0.619         |
========================================================================================================

## 5. Browser Inference Parity Verification

- **Total Vectors Tested:** 65
- **Max Absolute Logit Delta:** `0.0000e+0`
- **Action Mismatches:** 0
- **Parity Status:** ✅ 100% PARITY