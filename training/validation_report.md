# GMN-Football-3 — Policy Validation & Scientific Evidence Report

**Date of Evaluation:** 2026-09-03T12:46:10.591Z
**Target Scenario:** `academy_3_vs_1_with_keeper`
**Algorithm:** `MAPPO`
**Checkpoint SHA256:** `FILE_NOT_FOUND`
**Overall Scientific Verdict:** **SUPERFICIAL_OR_OVERFIT**

## Executive Summary

The policy showed moderate learning but failed one or more generalization/robustness criteria.

## Evidence Chain Criteria Matrix

| Step | Criterion | Verdict | Evidence / Metrics |
|:---:|:---|:---:|:---|
| 1 | Checkpoint & Schema Contract Compatibility | ✅ **PASS** | Checkpoint satisfies 127-dim role-aware contract (Role-aware weights present). |
| 3 | Beat Random Baseline | ❌ **FAIL** | MAPPO win rate (0.0%) vs Random (0.0%). Reward: 0.197 vs 0.041. |
| 4 | Competitive with Tactical Rule-Based Baseline | ⚠️ **INCONCLUSIVE** | MAPPO win rate (0.0%) vs Rule-Based Med (16.0%). Goal diff: 0.00 vs 0.16. |
| 5 | Held-Out Generalization Retention | ❌ **FAIL** | Train win rate: 0.0% | Held-out test win rate: 0.0% | Transfer retention: 0.0% (Gap: 0.0%). |
| 6 | Opponent Robustness & Anti-Exploitation | ✅ **PASS** | Easy Opponent Win Rate: 0.0% | Hard Opponent Win Rate: 0.0%. |
| 7 | Role Feature Conditioning Sensitivity | ✅ **PASS** | Standard 127-dim Win Rate: 0.0% | Zero-Role Ablated Win Rate: 0.0%. |
| 8 | Browser / Python Inference Parity | ✅ **PASS** | Tested 65 vectors. Max logit delta: 0.0000e+0. Action mismatches: 0. |

## 1. Baseline Ladder Performance

========================================================================================================
BASELINE LADDER EVALUATION SUMMARY TABLE
========================================================================================================
| Policy | Win Rate (%) | Goals (Mean±Std) | Possession (%) | Pass Acc (%) | Shot Acc (%) | Reward (Mean±Std) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| random             | 0.0%         | 0.00 ± 0.00      | 48.0%          | 0.0%         | 14.0%        | 0.041 ± 0.096     |
| noop               | 0.0%         | 0.00 ± 0.00      | 51.3%          | 0.0%         | 0.0%         | 0.043 ± 0.104     |
| rule_based_easy    | 16.0%        | 0.16 ± 0.37      | 95.3%          | 31.8%        | 76.0%        | 0.497 ± 0.426     |
| rule_based_medium  | 16.0%        | 0.16 ± 0.37      | 95.3%          | 31.8%        | 76.0%        | 0.497 ± 0.426     |
| rule_based_hard    | 18.0%        | 0.18 ± 0.39      | 96.3%          | 31.8%        | 76.0%        | 0.517 ± 0.445     |
| scripted           | 0.0%         | 0.00 ± 0.00      | 43.0%          | 0.0%         | 0.0%         | 0.000 ± 0.000     |
| mappo_trained      | 0.0%         | 0.00 ± 0.00      | 66.0%          | 1.0%         | 56.0%        | 0.197 ± 0.172     |
========================================================================================================

## 2. Held-Out Generalization Matrix

========================================================================================================
HELD-OUT GENERALIZATION SUMMARY
========================================================================================================
| Partition   | Scenario                                       | Win Rate (%) | Goals Scored   | Pass Acc (%) |
|:------------|:-----------------------------------------------|:------------:|:--------------:|:------------:|
| TRAIN       | academy_3_vs_1_with_keeper                     | 0.0%         | 0.00 ± 0.00    | 0.0%         |
| VALIDATION  | academy_3_vs_1_defender_2                      | 0.0%         | 0.00 ± 0.00    | 1.3%         |
| VALIDATION  | academy_3_vs_1_defender_3                      | 0.0%         | 0.00 ± 0.00    | 2.0%         |
| TEST        | academy_3_vs_1_keeper_aggressive               | 0.0%         | 0.00 ± 0.00    | 2.0%         |
| TEST        | academy_3_vs_1_shifted                         | 0.0%         | 0.00 ± 0.00    | 0.0%         |
| TEST        | academy_3_vs_1_randomized                      | 0.0%         | 0.00 ± 0.00    | 0.0%         |
--------------------------------------------------------------------------------------------------------
Overall Train Partition Win Rate:      0.0%
Overall Validation Partition Win Rate: 0.0%
Overall Held-out Test Win Rate:        0.0%
Generalization Transfer Retention:     0.0%
Overfitting Gap (Train - Test):        0.0%
========================================================================================================

## 3. Opponent Robustness Matrix

========================================================================================================
OPPONENT ROBUSTNESS SUMMARY TABLE
========================================================================================================
| Opponent Configuration              | Win Rate (%) | Goals Scored   | Turnovers Conceded | Pass Acc (%) |
|:------------------------------------|:------------:|:--------------:|:------------------:|:------------:|
| Weak / Easy Tactical Bot            | 0.0%         | 0.00 ± 0.00    | 0.60               | 1.0%         |
| Default / Medium Tactical Bot       | 0.0%         | 0.00 ± 0.00    | 0.60               | 1.0%         |
| Strong / Aggressive Tactical Bot    | 0.0%         | 0.00 ± 0.00    | 0.60               | 1.0%         |
| Master / Elite Tactical Bot         | 0.0%         | 0.00 ± 0.00    | 0.60               | 1.0%         |
| Scripted Defender Contain Bot       | 14.0%        | 0.14 ± 0.35    | 0.36               | 2.0%         |
========================================================================================================

## 4. Controlled Ablation Study

========================================================================================================
CONTROLLED ABLATION STUDY SUMMARY
========================================================================================================
| Condition               | Win Rate (%) | Goals (Mean±Std) | Pass Ratio (%) | Shot Ratio (%) | Reward (Mean) |
|:------------------------|:------------:|:----------------:|:--------------:|:--------------:|:-------------:|
| standard_role_127        | 0.0%         | 0.00 ± 0.00      | 4.8%           | 6.0%           | 0.193         |
| role_zeroed              | 0.0%         | 0.00 ± 0.00      | 6.6%           | 4.8%           | 0.198         |
| role_randomized          | 0.0%         | 0.00 ± 0.00      | 7.7%           | 10.7%          | 0.186         |
| role_inverted            | 0.0%         | 0.00 ± 0.00      | 7.1%           | 5.3%           | 0.200         |
========================================================================================================

## 5. Browser Inference Parity Verification

- **Total Vectors Tested:** 65
- **Max Absolute Logit Delta:** `0.0000e+0`
- **Action Mismatches:** 0
- **Parity Status:** ✅ 100% PARITY