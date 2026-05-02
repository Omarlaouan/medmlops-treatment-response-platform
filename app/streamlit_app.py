"""Streamlit demo for the MedMLOps treatment-response platform."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.explainability.explain_prediction import explain_input, get_global_feature_importance
from src.inference.predict import DISCLAIMER, predict_one

WARD_TYPES = ["emergency", "outpatient", "medical", "surgery", "ICU"]
INFECTION_SITES = ["urinary", "respiratory", "bloodstream", "skin_soft_tissue"]
PATHOGENS = [
    "E. coli",
    "Klebsiella pneumoniae",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Enterococcus faecalis",
]
ANTIBIOTICS = [
    "amoxicillin",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "meropenem",
    "vancomycin",
    "nitrofurantoin",
]


def _load_model_metadata(path: Path = Path("models/model_metadata.json")) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _write_factor_list(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def main() -> None:
    st.set_page_config(page_title="MedMLOps Treatment Response Platform", layout="wide")
    st.title("MedMLOps Treatment Response Platform")
    st.warning(DISCLAIMER)

    st.sidebar.header("Patient and microbiology inputs")
    payload = {
        "age": st.sidebar.slider("Age", min_value=18, max_value=95, value=67),
        "sex": st.sidebar.selectbox("Sex", options=["F", "M"]),
        "ward_type": st.sidebar.selectbox("Ward type", options=WARD_TYPES, index=2),
        "infection_site": st.sidebar.selectbox("Infection site", options=INFECTION_SITES),
        "pathogen": st.sidebar.selectbox("Pathogen", options=PATHOGENS),
        "antibiotic": st.sidebar.selectbox("Antibiotic", options=ANTIBIOTICS, index=6),
        "prior_antibiotic_exposure": int(
            st.sidebar.checkbox("Prior antibiotic exposure", value=False)
        ),
        "comorbidity_score": st.sidebar.slider(
            "Comorbidity score", min_value=0, max_value=6, value=2
        ),
        "local_resistance_rate": st.sidebar.slider(
            "Local resistance rate",
            min_value=0.0,
            max_value=1.0,
            value=0.12,
            step=0.01,
        ),
    }

    model_path = Path("models/treatment_response_model.joblib")
    if not model_path.exists():
        st.error("Model artifact not found. Run `make all` or `make train` before using the demo.")
        st.stop()

    metadata = _load_model_metadata()
    prediction = predict_one(payload, model_path=str(model_path))
    explanation = explain_input(payload)
    probability = prediction["probability_of_susceptibility"]

    prediction_tab, drivers_tab, architecture_tab, limitations_tab = st.tabs(
        ["Prediction", "Model Drivers", "MLOps Architecture", "Limitations"]
    )

    with prediction_tab:
        metric_col, label_col, model_col = st.columns(3)
        metric_col.metric("Probability of susceptibility", f"{probability:.1%}")
        label_col.metric("Recommendation level", prediction["recommendation_level"])
        model_col.metric("Model status", "artifact loaded")
        st.progress(probability, text="Susceptibility probability")

        st.subheader("Local explanation")
        pos_col, neg_col = st.columns(2)
        with pos_col:
            st.write("Positive factors")
            _write_factor_list(explanation["positive_factors"])
        with neg_col:
            st.write("Negative factors")
            _write_factor_list(explanation["negative_factors"])
        st.write("Context")
        _write_factor_list(explanation["contextual_factors"])

        st.subheader("Model status")
        status_col, rows_col, metric_status_col = st.columns(3)
        status_col.write(f"Artifact updated: `{_model_timestamp(model_path)}`")
        if metadata:
            rows_col.write(f"Training rows: `{metadata.get('n_train_rows', 'n/a')}`")
            roc_auc = metadata.get("metrics", {}).get("roc_auc", "n/a")
            metric_status_col.write(
                f"ROC-AUC: `{roc_auc:.3f}`" if isinstance(roc_auc, float) else "ROC-AUC: `n/a`"
            )
        else:
            rows_col.write("Training rows: `n/a`")
            metric_status_col.write("ROC-AUC: `n/a`")

    with drivers_tab:
        st.subheader("Global model drivers")
        try:
            importance = get_global_feature_importance(model_path)
            positive_col, negative_col = st.columns(2)
            with positive_col:
                st.write("Top positive drivers")
                st.dataframe(importance["top_positive_drivers"], width="stretch")
            with negative_col:
                st.write("Top negative drivers")
                st.dataframe(importance["top_negative_drivers"], width="stretch")
        except Exception as exc:  # pragma: no cover - UI-only safeguard
            st.info(f"Global feature importance is unavailable: {exc}")

    with architecture_tab:
        st.subheader("Why this is an MLOps project, not a clinical tool")
        st.write(
            "This demo emphasizes the operational wrapper around a healthcare ML model: "
            "validated data, reproducible cohort construction, training, API serving, "
            "explainability, monitoring, tests, Docker packaging, CI, and a model card."
        )
        arch_cols = st.columns(3)
        arch_cols[0].markdown(
            "**Data**\n- synthetic generation\n- schema validation\n- cohort construction"
        )
        arch_cols[1].markdown(
            "**Modeling**\n- preprocessing pipeline\n- baseline classifier\n- evaluation slices"
        )
        arch_cols[2].markdown(
            "**Operations**\n- FastAPI serving\n- Streamlit demo\n- drift monitoring"
        )

        if metadata:
            st.subheader("Training metadata")
            st.json(
                {
                    "model_type": metadata.get("model_type"),
                    "numeric_features": metadata.get("numeric_features"),
                    "categorical_features": metadata.get("categorical_features"),
                    "mlflow_run_id": metadata.get("mlflow_run_id"),
                }
            )

    with limitations_tab:
        st.subheader("Clinical and ethical limitations")
        st.markdown(
            """
            - Synthetic data only; no real patient records or PHI.
            - No clinical validation, prospective validation, or stewardship review.
            - No dosing, allergy, renal function, breakpoint, formulary, or source-control logic.
            - Recommendation levels are threshold-based demo labels, not treatment guidance.
            - A real system would require governance, calibration monitoring, subgroup analysis, and clinician-in-the-loop validation.
            """
        )
        st.caption(prediction["warning"])


if __name__ == "__main__":
    main()
