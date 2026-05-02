# Drift Report

This report compares a reference training batch to a current synthetic cohort.

## Numeric Drift

| Column | Reference Mean | Current Mean | Mean Difference | Reference Std | Current Std |
| --- | ---: | ---: | ---: | ---: | ---: |
| age | 57.8782 | 57.7634 | -0.1148 | 17.4665 | 17.4819 |
| local_resistance_rate | 0.3424 | 0.3420 | -0.0004 | 0.1709 | 0.1707 |
| comorbidity_score | 1.8046 | 1.8172 | 0.0126 | 1.3464 | 1.3501 |

## Categorical Drift

| Column | New Categories | Max Distribution Delta |
| --- | --- | ---: |
| pathogen | [] | 0.0009 |
| antibiotic | [] | 0.0035 |
| ward_type | [] | 0.0037 |

## Target Rate

| Reference Susceptible Rate | Current Susceptible Rate | Delta |
| ---: | ---: | ---: |
| 0.3975 | 0.3976 | 0.0001 |

## Alerts

- Numeric drift alert: age mean changed by -0.115, above threshold 0.100.

## Disclaimer

This project is an educational prototype. It is not a medical device, not clinically validated, and must not be used for clinical decision-making.
