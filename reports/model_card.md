# Model Card: Synthetic Treatment-Response Baseline

## Model Name

MedMLOps synthetic treatment-response logistic-regression baseline.

## Model Type

Scikit-learn pipeline with numeric scaling, categorical one-hot encoding, and logistic regression. Training can be tracked with MLflow for experiment metadata, metrics, and artifacts.

## Intended Use

This model is intended for portfolio demonstration of healthcare ML engineering workflows: synthetic data generation, validation, cohort construction, training, evaluation, experiment tracking, serving, explainability, monitoring, and documentation.

## Not Intended Use

This model must not be used for clinical decision-making, antibiotic selection, patient care, prescribing, antimicrobial stewardship decisions, or any real-world medical workflow.

## Dataset

The dataset is generated locally and synthetically. It simulates antimicrobial resistance-like records with patient demographics, care setting, infection site, pathogen, antibiotic, prior exposure, comorbidity score, local resistance rate, sample date, and synthetic susceptibility label.

No private clinical data, patient records, credentialed datasets, or PHI are used.

## Target

`susceptible`, where 1 means synthetic susceptibility and 0 means synthetic resistant or intermediate.

## Features

Numeric features:

- age
- comorbidity_score
- local_resistance_rate
- prior_antibiotic_exposure

Categorical features:

- sex
- ward_type
- infection_site
- pathogen
- antibiotic

## Metrics

Training writes evaluation metrics to `reports/evaluation_report.md`, including ROC-AUC, accuracy, precision, recall, F1, and Brier score.

The evaluation report also includes:

- threshold tradeoff table
- calibration summary by probability bin
- per-slice metrics by ward, pathogen, antibiotic, prior exposure, and comorbidity bucket

These diagnostics are intended to demonstrate engineering rigor, not clinical validity.

## Experiment Tracking

When MLflow is installed, training logs:

- model and split parameters
- global metrics
- trained model artifact
- evaluation report
- test predictions
- reference profile
- model card

## Explainability

The project exposes global logistic-regression coefficients after preprocessing and a lightweight local heuristic explanation for key clinical-context variables. This is not a clinical explanation and does not establish causal treatment effect.

## Calibration

The project reports Brier score and a calibration summary by probability bin. This is a synthetic-data diagnostic only. A real healthcare model would require calibration validation on representative external datasets and ongoing calibration monitoring.

## Ethical And Clinical Limitations

- Synthetic data does not represent real patient populations.
- The model has not been externally validated.
- The model omits many clinically necessary variables such as allergy, renal function, dose, route, infection severity, source control, local formulary, and susceptibility breakpoints.
- The model is not calibrated for real-world use.
- No subgroup fairness, safety, or prospective impact assessment has been performed.
- The model does not implement organism-specific breakpoints, allergy checks, renal dosing, drug interactions, formulary constraints, or stewardship policy.
- Recommendation levels are threshold-based demo labels and must not be interpreted as clinical recommendations.

## Deployment Considerations

The FastAPI and Streamlit interfaces are included for engineering demonstration. A real deployment would require clinical governance, secure data pipelines, access controls, audit logging, model registry, reproducible release process, monitoring, rollback, and human-factors review.

The current API includes health, metadata, model-info, and prediction endpoints. The model artifact is stored locally as a joblib file for simplicity.

## Monitoring Needs

The MVP includes drift monitoring for numeric means, categorical distributions, new categories, missingness, and target-rate changes. It writes both markdown and JSON reports with severity labels.

Real monitoring should add calibration monitoring, subgroup performance, data freshness, operational alerts, clinical safety review, and governance sign-off.

## Disclaimer

This project is an educational prototype. It is not a medical device, not clinically validated, and must not be used for clinical decision-making.
