"""Streamlit demo for the MedMLOps treatment-response platform."""

from __future__ import annotations

from pathlib import Path

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
        "comorbidity_score": st.sidebar.slider("Comorbidity score", min_value=0, max_value=6, value=2),
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

    prediction = predict_one(payload, model_path=str(model_path))
    explanation = explain_input(payload)

    metric_col, label_col = st.columns(2)
    metric_col.metric(
        "Probability of susceptibility",
        f"{prediction['probability_of_susceptibility']:.1%}",
    )
    label_col.metric("Recommendation level", prediction["recommendation_level"])

    st.subheader("Explanation")
    st.write("Positive factors")
    st.write(explanation["positive_factors"])
    st.write("Negative factors")
    st.write(explanation["negative_factors"])
    st.write("Context")
    st.write(explanation["contextual_factors"])

    st.subheader("Global model drivers")
    try:
        importance = get_global_feature_importance(model_path)
        st.write("Top positive drivers")
        st.dataframe(importance["top_positive_drivers"], use_container_width=True)
        st.write("Top negative drivers")
        st.dataframe(importance["top_negative_drivers"], use_container_width=True)
    except Exception as exc:  # pragma: no cover - UI-only safeguard
        st.info(f"Global feature importance is unavailable: {exc}")

    st.subheader("Why this is an MLOps project, not a clinical tool")
    st.write(
        "This demo emphasizes the operational wrapper around a healthcare ML model: validated data, "
        "reproducible cohort construction, training, API serving, explainability, monitoring, tests, "
        "Docker packaging, and a model card. It does not implement clinical validation or treatment guidance."
    )

    st.subheader("Architecture components")
    st.markdown(
        """
        - data validation
        - cohort construction
        - training
        - API
        - explainability
        - monitoring
        - model card
        """
    )
    st.caption(prediction["warning"])


if __name__ == "__main__":
    main()
