# Evaluation Report

This report was generated from the held-out synthetic test split.

## Global Metrics

| Metric | Value |
| --- | ---: |
| roc_auc | 0.7207 |
| accuracy | 0.6571 |
| precision | 0.5565 |
| recall | 0.6809 |
| f1 | 0.6124 |
| brier_score | 0.2147 |

## Threshold Analysis

Recommendation levels in the API are threshold-based demonstration labels. This table shows how standard classification metrics change across candidate thresholds.

| Threshold | Accuracy | Precision | Recall | F1 | Predicted Positive Rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 0.6127 | 0.5084 | 0.8059 | 0.6235 | 0.6307 |
| 0.50 | 0.6571 | 0.5565 | 0.6809 | 0.6124 | 0.4868 |
| 0.60 | 0.6815 | 0.6206 | 0.5133 | 0.5619 | 0.3291 |
| 0.80 | 0.6265 | 0.6885 | 0.1117 | 0.1922 | 0.0646 |

## Calibration Summary

Calibration is summarized by comparing average predicted probability to observed susceptible rate in probability bins. This is a synthetic-data diagnostic, not clinical validation.

| Probability Bin | Count | Mean Predicted Probability | Observed Susceptible Rate |
| --- | ---: | ---: | ---: |
| 0.0-0.1 | 28 | 0.0793 | 0.1071 |
| 0.1-0.2 | 94 | 0.1594 | 0.1383 |
| 0.2-0.3 | 105 | 0.2511 | 0.1905 |
| 0.3-0.4 | 122 | 0.3455 | 0.3033 |
| 0.4-0.5 | 136 | 0.4500 | 0.3456 |
| 0.5-0.6 | 149 | 0.5482 | 0.4228 |
| 0.6-0.7 | 127 | 0.6541 | 0.5276 |
| 0.7-0.8 | 123 | 0.7494 | 0.6829 |
| 0.8-0.9 | 56 | 0.8385 | 0.6964 |
| 0.9-1.0 | 5 | 0.9157 | 0.6000 |

## Slice Evaluation


### ward_type

| Slice | Support | Low Support | Target Rate | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| outpatient | 272 | no | 0.3971 | 0.7313 | 0.6801 | 0.5778 | 0.7222 | 0.6420 |
| medical | 246 | no | 0.4106 | 0.6820 | 0.6382 | 0.5476 | 0.6832 | 0.6079 |
| emergency | 205 | no | 0.4146 | 0.7458 | 0.6780 | 0.6022 | 0.6588 | 0.6292 |
| surgery | 151 | no | 0.3841 | 0.7238 | 0.6159 | 0.5000 | 0.6724 | 0.5735 |
| ICU | 71 | no | 0.3380 | 0.7376 | 0.6620 | 0.5000 | 0.5833 | 0.5385 |

### pathogen

| Slice | Support | Low Support | Target Rate | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E. coli | 372 | no | 0.4839 | 0.6823 | 0.6048 | 0.5642 | 0.8056 | 0.6636 |
| Klebsiella pneumoniae | 190 | no | 0.3000 | 0.7209 | 0.7158 | 0.5283 | 0.4912 | 0.5091 |
| Staphylococcus aureus | 173 | no | 0.4277 | 0.7307 | 0.6532 | 0.5897 | 0.6216 | 0.6053 |
| Pseudomonas aeruginosa | 118 | no | 0.2627 | 0.7412 | 0.7034 | 0.4474 | 0.5484 | 0.4928 |
| Enterococcus faecalis | 92 | no | 0.3696 | 0.6552 | 0.6957 | 0.5882 | 0.5882 | 0.5882 |

### antibiotic

| Slice | Support | Low Support | Target Rate | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nitrofurantoin | 200 | no | 0.4550 | 0.6957 | 0.5950 | 0.5338 | 0.8681 | 0.6611 |
| ciprofloxacin | 167 | no | 0.2635 | 0.6824 | 0.6826 | 0.3636 | 0.2727 | 0.3117 |
| gentamicin | 127 | no | 0.4173 | 0.7022 | 0.6772 | 0.6304 | 0.5472 | 0.5859 |
| ceftriaxone | 121 | no | 0.3140 | 0.6893 | 0.6777 | 0.4857 | 0.4474 | 0.4658 |
| meropenem | 115 | no | 0.5217 | 0.7703 | 0.5913 | 0.5644 | 0.9500 | 0.7081 |
| vancomycin | 109 | no | 0.6422 | 0.4546 | 0.6147 | 0.6458 | 0.8857 | 0.7470 |
| amoxicillin | 106 | no | 0.1887 | 0.5006 | 0.8019 | 0.0000 | 0.0000 | 0.0000 |

### prior_antibiotic_exposure

| Slice | Support | Low Support | Target Rate | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 667 | no | 0.4393 | 0.7071 | 0.6357 | 0.5641 | 0.7509 | 0.6442 |
| 1 | 278 | no | 0.2986 | 0.7302 | 0.7086 | 0.5143 | 0.4337 | 0.4706 |

### comorbidity_bucket

| Slice | Support | Low Support | Target Rate | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-2 | 527 | no | 0.4042 | 0.7164 | 0.6452 | 0.5474 | 0.7042 | 0.6160 |
| 3-4 | 232 | no | 0.3793 | 0.7160 | 0.6509 | 0.5347 | 0.6136 | 0.5714 |
| 0 | 144 | no | 0.4306 | 0.7223 | 0.6806 | 0.6053 | 0.7419 | 0.6667 |
| 5-6 | 42 | no | 0.3095 | 0.8037 | 0.7619 | 0.6667 | 0.4615 | 0.5455 |

## Interpretation Notes

- ROC-AUC may be `n/a` for small slices containing only one target class.
- Brier score is included as a probability-quality metric; lower is better.
- Slice metrics are included to demonstrate healthcare ML monitoring discipline, not to claim subgroup safety.

## Disclaimer

This project is an educational prototype. It is not a medical device, not clinically validated, and must not be used for clinical decision-making.
