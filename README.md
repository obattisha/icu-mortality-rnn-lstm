# icu-mortality-rnn-lstm

Deep learning project predicting 30-day hospital mortality for ICU patients using time-series vital sign measurements and static comorbidity features.

Built as a machine learning course final project (Battisha & Bhavani).

## Problem

Early and accurate prediction of ICU patient mortality is a critical clinical challenge. This project frames it as a binary classification task and compares two sequence models on a real patient dataset.

## Models

**Conditional RNN** (primary model)
Uses the [`cond_rnn`](https://github.com/philipperemy/cond_rnn) library to jointly process time-varying vital signs and time-invariant patient characteristics in a single recurrent network, avoiding the need to replicate static features across timesteps.

**LSTM** (baseline)
Standard long short-term memory network on the same data for direct comparison.

## Data

~2,580 ICU patients with:
- **Time-varying features** (hourly, 25-hour window): heart rate, respiratory rate, systolic BP, diastolic BP, body temperature, oxygen saturation
- **Static comorbidities** (8 binary flags): COPD, heart failure, renal disease, and others
- **Outcome**: 30-day mortality (binary)

## Evaluation

- ROC-AUC
- Confusion matrix
- Calibration plots
- 5-fold stratified cross-validation

## Files

| File | Description |
|------|-------------|
| `Data_Cleaning.ipynb` | Preprocessing: forward/backward fill of missing vitals, standardization, reshape to 3D tensor |
| `Conditional_RNN-LSTM.ipynb` | Model training and evaluation for both architectures |
| `Final Report.pdf` | Full written report |
| `proposal/Proposal.pdf` | Original project proposal |
| `ml_standardized.csv` / `ml_fixed.csv` | Processed patient data |

## Requirements

Python 3 with: `tensorflow`, `keras`, `cond_rnn`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`
