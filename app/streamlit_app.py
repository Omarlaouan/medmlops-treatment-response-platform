"""Streamlit demo for the MedMLOps treatment-response platform."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
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

MODEL_PATH = Path("models/treatment_response_model.joblib")
MODEL_METADATA_PATH = Path("models/model_metadata.json")
EVALUATION_JSON_PATH = Path("reports/evaluation_report.json")
DRIFT_JSON_PATH = Path("reports/drift_report.json")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _format_percent(value: Any, decimals: int = 1) -> str:
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: Any, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _recommendation_class(level: str) -> str:
    return {
        "strong candidate": "strong",
        "candidate": "candidate",
        "uncertain": "uncertain",
        "not recommended": "not-recommended",
    }.get(level, "uncertain")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        h1, h2, h3 {
            color: #172033;
            letter-spacing: 0;
        }
        .product-bar {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            padding: 0.95rem 1.1rem;
            background: #ffffff;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .product-title {
            font-size: 1.45rem;
            font-weight: 750;
            line-height: 1.2;
            color: #172033;
        }
        .product-subtitle {
            margin-top: 0.2rem;
            color: #526173;
            font-size: 0.93rem;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            justify-content: flex-end;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            min-height: 1.65rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            color: #334155;
            font-size: 0.78rem;
            font-weight: 650;
            white-space: nowrap;
        }
        .chip.teal {
            border-color: #99f6e4;
            background: #ecfdf5;
            color: #0f766e;
        }
        .chip.amber {
            border-color: #fde68a;
            background: #fffbeb;
            color: #92400e;
        }
        .chip.red {
            border-color: #fecaca;
            background: #fef2f2;
            color: #991b1b;
        }
        .disclaimer {
            padding: 0.78rem 0.9rem;
            border-radius: 8px;
            border: 1px solid #fcd34d;
            background: #fffbeb;
            color: #78350f;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            min-height: 122px;
            padding: 0.9rem 1rem;
            background: #ffffff;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02rem;
        }
        .metric-value {
            color: #172033;
            font-size: 2rem;
            font-weight: 760;
            line-height: 1.15;
            margin-top: 0.45rem;
        }
        .metric-help {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }
        .tier {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.62rem;
            border-radius: 6px;
            font-size: 1.25rem;
            font-weight: 760;
            margin-top: 0.45rem;
        }
        .tier.strong {
            color: #065f46;
            background: #d1fae5;
        }
        .tier.candidate {
            color: #0f766e;
            background: #ccfbf1;
        }
        .tier.uncertain {
            color: #92400e;
            background: #fef3c7;
        }
        .tier.not-recommended {
            color: #991b1b;
            background: #fee2e2;
        }
        .probability-card {
            padding: 1rem;
            background: #ffffff;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .probability-rail {
            position: relative;
            height: 46px;
            display: flex;
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
        }
        .probability-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.74rem;
            font-weight: 720;
            color: #172033;
            border-right: 1px solid rgba(255,255,255,0.95);
            text-align: center;
        }
        .seg-red { background: #fecaca; width: 40%; }
        .seg-amber { background: #fde68a; width: 20%; }
        .seg-teal { background: #99f6e4; width: 20%; }
        .seg-green { background: #bbf7d0; width: 20%; border-right: 0; }
        .probability-marker {
            position: absolute;
            top: -7px;
            bottom: -7px;
            width: 3px;
            border-radius: 3px;
            background: #172033;
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.18);
        }
        .section-card {
            padding: 1rem;
            background: #ffffff;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .factor-row {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.72rem 0.8rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            margin-bottom: 0.5rem;
        }
        .factor-dot {
            width: 0.72rem;
            min-width: 0.72rem;
            height: 0.72rem;
            border-radius: 999px;
            margin-top: 0.2rem;
        }
        .factor-dot.positive { background: #0f766e; }
        .factor-dot.negative { background: #dc2626; }
        .factor-dot.context { background: #2563eb; }
        .factor-title {
            color: #172033;
            font-size: 0.92rem;
            font-weight: 650;
        }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
        }
        .workflow-step {
            min-height: 98px;
            padding: 0.85rem;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            background: #ffffff;
        }
        .workflow-index {
            color: #0f766e;
            font-weight: 800;
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
        }
        .workflow-title {
            color: #172033;
            font-weight: 730;
            margin-bottom: 0.2rem;
        }
        .workflow-copy {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.35;
        }
        .empty-state {
            padding: 1rem;
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            color: #64748b;
            background: #f8fafc;
        }
        @media (max-width: 900px) {
            .product-bar {
                flex-direction: column;
            }
            .chip-row {
                justify-content: flex-start;
            }
            .workflow-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(metadata: dict[str, Any] | None, model_exists: bool) -> None:
    artifact_status = "Model loaded" if model_exists else "Model missing"
    artifact_class = "teal" if model_exists else "red"
    timestamp = _model_timestamp(MODEL_PATH) if model_exists else "run make all"
    row_count = metadata.get("n_rows") if metadata else "n/a"

    st.markdown(
        f"""
        <div class="product-bar">
          <div>
            <div class="product-title">MedMLOps Treatment Response Platform</div>
            <div class="product-subtitle">
              Synthetic AMR treatment-response workbench for ML engineering review.
            </div>
          </div>
          <div class="chip-row">
            <span class="chip amber">Educational prototype</span>
            <span class="chip {artifact_class}">{artifact_status}</span>
            <span class="chip teal">Synthetic data</span>
            <span class="chip red">Not clinical use</span>
            <span class="chip">Rows: {html.escape(str(row_count))}</span>
            <span class="chip">Updated: {html.escape(timestamp)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="disclaimer">{html.escape(DISCLAIMER)}</div>', unsafe_allow_html=True)


def _render_sidebar_inputs() -> dict[str, Any]:
    st.sidebar.title("Input Panel")
    st.sidebar.caption("Synthetic patient and microbiology context")

    st.sidebar.markdown("### Patient Context")
    age = st.sidebar.slider("Age", min_value=18, max_value=95, value=67)
    sex = st.sidebar.selectbox("Sex", options=["F", "M"])
    comorbidity_score = st.sidebar.slider("Comorbidity score", min_value=0, max_value=6, value=2)

    st.sidebar.markdown("### Care Setting")
    ward_type = st.sidebar.selectbox("Ward type", options=WARD_TYPES, index=2)
    infection_site = st.sidebar.selectbox("Infection site", options=INFECTION_SITES)

    st.sidebar.markdown("### Microbiology")
    pathogen = st.sidebar.selectbox("Pathogen", options=PATHOGENS)
    antibiotic = st.sidebar.selectbox("Antibiotic", options=ANTIBIOTICS, index=6)

    st.sidebar.markdown("### Resistance Context")
    prior_antibiotic_exposure = int(st.sidebar.toggle("Prior antibiotic exposure", value=False))
    local_resistance_rate = st.sidebar.slider(
        "Local resistance rate",
        min_value=0.0,
        max_value=1.0,
        value=0.12,
        step=0.01,
    )

    return {
        "age": age,
        "sex": sex,
        "ward_type": ward_type,
        "infection_site": infection_site,
        "pathogen": pathogen,
        "antibiotic": antibiotic,
        "prior_antibiotic_exposure": prior_antibiotic_exposure,
        "comorbidity_score": comorbidity_score,
        "local_resistance_rate": local_resistance_rate,
    }


def _metric_card(label: str, value: str, help_text: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{html.escape(label)}</div>
      <div class="metric-value">{html.escape(value)}</div>
      <div class="metric-help">{html.escape(help_text)}</div>
    </div>
    """


def _render_prediction_workbench(
    prediction: dict[str, Any],
    probability: float,
    explanation: dict[str, list[str]],
    metadata: dict[str, Any] | None,
) -> None:
    level = prediction["recommendation_level"]
    tier_class = _recommendation_class(level)

    metric_col, tier_col, model_col = st.columns([1.1, 1.1, 1.0])
    metric_col.markdown(
        _metric_card(
            "Probability of susceptibility",
            _format_percent(probability),
            "Model-estimated synthetic susceptibility probability",
        ),
        unsafe_allow_html=True,
    )
    tier_col.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Recommendation tier</div>
          <div class="tier {tier_class}">{html.escape(level)}</div>
          <div class="metric-help">Threshold-based demo label</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    roc_auc = metadata.get("metrics", {}).get("roc_auc") if metadata else None
    brier = metadata.get("metrics", {}).get("brier_score") if metadata else None
    model_col.markdown(
        _metric_card(
            "Model quality snapshot",
            f"AUC {_format_number(roc_auc)}",
            f"Brier {_format_number(brier)}",
        ),
        unsafe_allow_html=True,
    )

    marker_left = max(0.0, min(100.0, probability * 100.0))
    st.markdown(
        f"""
        <div class="probability-card">
          <div class="metric-label">Threshold Bands</div>
          <div class="probability-rail">
            <div class="probability-segment seg-red">&lt;0.40<br>not recommended</div>
            <div class="probability-segment seg-amber">0.40-0.60<br>uncertain</div>
            <div class="probability-segment seg-teal">0.60-0.80<br>candidate</div>
            <div class="probability-segment seg-green">&ge;0.80<br>strong candidate</div>
            <div class="probability-marker" style="left: {marker_left:.2f}%"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    positive_col, negative_col, context_col = st.columns(3)
    with positive_col:
        _render_factor_group("Positive Drivers", explanation["positive_factors"], "positive")
    with negative_col:
        _render_factor_group("Negative Drivers", explanation["negative_factors"], "negative")
    with context_col:
        _render_factor_group("Context", explanation["contextual_factors"], "context")


def _render_factor_group(title: str, items: list[str], factor_class: str) -> None:
    rows = "\n".join(
        f"""
        <div class="factor-row">
          <div class="factor-dot {factor_class}"></div>
          <div class="factor-title">{html.escape(item)}</div>
        </div>
        """
        for item in items
    )
    st.markdown(
        f"""
        <div class="section-card">
          <div class="metric-label">{html.escape(title)}</div>
          <div style="height: .65rem"></div>
          {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_model_operations(metadata: dict[str, Any] | None) -> None:
    st.subheader("Model Operations")
    if not metadata:
        st.markdown(
            '<div class="empty-state">Model metadata not found. Run <code>make all</code>.</div>',
            unsafe_allow_html=True,
        )
        return

    feature_count = len(metadata.get("numeric_features", [])) + len(
        metadata.get("categorical_features", [])
    )
    cols = st.columns(5)
    values = [
        ("Model type", metadata.get("model_type", "n/a"), "baseline classifier"),
        ("Train rows", str(metadata.get("n_train_rows", "n/a")), "reference batch"),
        ("Test rows", str(metadata.get("n_test_rows", "n/a")), "held-out split"),
        ("Feature groups", str(feature_count), "numeric + categorical"),
        ("MLflow run", metadata.get("mlflow_run_id") or "not logged", "local tracking"),
    ]
    for column, (label, value, help_text) in zip(cols, values, strict=False):
        column.markdown(_metric_card(label, str(value), help_text), unsafe_allow_html=True)


def _render_model_drivers() -> None:
    st.subheader("Model Drivers")
    try:
        importance = get_global_feature_importance(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - UI-only safeguard
        st.markdown(
            f'<div class="empty-state">Global model drivers unavailable: {html.escape(str(exc))}</div>',
            unsafe_allow_html=True,
        )
        return

    positive_col, negative_col = st.columns(2)
    with positive_col:
        st.markdown("#### Top Positive Coefficients")
        st.dataframe(pd.DataFrame(importance["top_positive_drivers"]), width="stretch")
    with negative_col:
        st.markdown("#### Top Negative Coefficients")
        st.dataframe(pd.DataFrame(importance["top_negative_drivers"]), width="stretch")


def _render_evaluation(evaluation: dict[str, Any] | None, metadata: dict[str, Any] | None) -> None:
    st.subheader("Evaluation")
    if not evaluation:
        if metadata and metadata.get("metrics"):
            metric_df = _metrics_dataframe(metadata["metrics"])
            st.dataframe(metric_df, width="stretch", hide_index=True)
        else:
            st.markdown(
                '<div class="empty-state">Evaluation artifact not found. Run <code>make all</code>.</div>',
                unsafe_allow_html=True,
            )
        return

    metric_col, threshold_col = st.columns([0.8, 1.2])
    with metric_col:
        st.markdown("#### Global Metrics")
        st.dataframe(
            _metrics_dataframe(evaluation.get("global_metrics", {})),
            width="stretch",
            hide_index=True,
        )
    with threshold_col:
        st.markdown("#### Threshold Tradeoffs")
        threshold_df = pd.DataFrame(evaluation.get("threshold_metrics", []))
        if not threshold_df.empty:
            st.dataframe(threshold_df, width="stretch", hide_index=True)

    calibration_df = pd.DataFrame(evaluation.get("calibration_summary", []))
    if not calibration_df.empty:
        st.markdown("#### Calibration Summary")
        calibration_df = calibration_df.assign(
            probability_bin=calibration_df.apply(
                lambda row: f"{row['bin_lower']:.1f}-{row['bin_upper']:.1f}",
                axis=1,
            )
        )[
            [
                "probability_bin",
                "count",
                "mean_predicted_probability",
                "observed_susceptible_rate",
            ]
        ]
        st.dataframe(calibration_df, width="stretch", hide_index=True)

    slice_metrics = evaluation.get("slice_metrics", {})
    if slice_metrics:
        st.markdown("#### Slice Evaluation")
        selected_slice = st.selectbox(
            "Slice dimension",
            options=list(slice_metrics.keys()),
            index=0,
        )
        slice_df = pd.DataFrame(slice_metrics[selected_slice])
        st.dataframe(slice_df, width="stretch", hide_index=True)


def _metrics_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": key, "value": _format_number(value, decimals=4)}
            for key, value in metrics.items()
        ]
    )


def _render_monitoring(drift: dict[str, Any] | None) -> None:
    st.subheader("Monitoring")
    if not drift:
        st.markdown(
            '<div class="empty-state">Drift artifact not found. Run <code>make monitor</code>.</div>',
            unsafe_allow_html=True,
        )
        return

    alerts = drift.get("alerts", [])
    severity_counts: dict[str, int] = {}
    for alert in alerts:
        severity = alert.get("severity", "info")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    alert_col, target_col, rows_col = st.columns(3)
    alert_col.markdown(
        _metric_card("Active drift alerts", str(len(alerts)), ", ".join(severity_counts) or "none"),
        unsafe_allow_html=True,
    )
    target = drift.get("target_summary") or {}
    target_col.markdown(
        _metric_card(
            "Current target rate",
            _format_percent(target.get("current_target_rate")),
            f"Delta {_format_number(target.get('target_rate_delta'), 4)}",
        ),
        unsafe_allow_html=True,
    )
    rows_col.markdown(
        _metric_card(
            "Reference target rate",
            _format_percent(target.get("reference_target_rate")),
            "training reference batch",
        ),
        unsafe_allow_html=True,
    )

    if alerts:
        for alert in alerts:
            st.warning(f"{alert.get('severity', 'info')}: {alert.get('message', '')}")
    else:
        st.success("No drift alerts triggered by the current simple monitor.")

    table_tabs = st.tabs(["Numeric", "Categorical", "Missingness"])
    with table_tabs[0]:
        st.dataframe(
            pd.DataFrame(drift.get("numeric_summary", [])), width="stretch", hide_index=True
        )
    with table_tabs[1]:
        st.dataframe(
            pd.DataFrame(drift.get("categorical_summary", [])),
            width="stretch",
            hide_index=True,
        )
    with table_tabs[2]:
        st.dataframe(
            pd.DataFrame(drift.get("missingness_summary", [])),
            width="stretch",
            hide_index=True,
        )


def _render_architecture() -> None:
    st.subheader("MLOps Architecture")
    steps = [
        ("01", "Generate", "Synthetic AMR-like records with reproducible signal."),
        ("02", "Validate", "Schema, ranges, categories, missingness, and target checks."),
        ("03", "Cohort", "Urinary cohort construction with pathogen-antibiotic controls."),
        ("04", "Train", "Scikit-learn pipeline with evaluation and MLflow logging."),
        ("05", "Serve", "FastAPI prediction, metadata, and model-info endpoints."),
        ("06", "Explain", "Global coefficients and local heuristic drivers."),
        ("07", "Monitor", "Reference batch drift checks with markdown and JSON outputs."),
        ("08", "Govern", "Model card, clinical safety note, disclaimer, and CI."),
    ]
    cards = "\n".join(
        f"""
        <div class="workflow-step">
          <div class="workflow-index">{index}</div>
          <div class="workflow-title">{html.escape(title)}</div>
          <div class="workflow-copy">{html.escape(copy)}</div>
        </div>
        """
        for index, title, copy in steps
    )
    st.markdown(f'<div class="workflow-grid">{cards}</div>', unsafe_allow_html=True)


def _render_governance() -> None:
    st.subheader("Governance and Limitations")
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(
            """
            **Boundary conditions**

            - Synthetic data only; no real patient records or PHI.
            - Not a medical device.
            - Not clinically validated.
            - No dosing, allergy, renal function, breakpoint, formulary, or source-control logic.
            - Recommendation levels are demo thresholds, not treatment guidance.
            """
        )
    with right_col:
        st.markdown(
            """
            **Real-world validation checklist**

            - External site validation.
            - Calibration and subgroup analysis.
            - Stewardship and clinical governance review.
            - Silent-mode prospective evaluation.
            - Monitoring thresholds and escalation paths.
            """
        )
    st.info(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="MedMLOps Treatment Response Platform", layout="wide")
    _inject_css()

    metadata = _load_json(MODEL_METADATA_PATH)
    evaluation = _load_json(EVALUATION_JSON_PATH)
    drift = _load_json(DRIFT_JSON_PATH)
    _render_header(metadata, MODEL_PATH.exists())
    payload = _render_sidebar_inputs()

    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Run `make all` before using the workbench.")
        return

    prediction = predict_one(payload, model_path=str(MODEL_PATH))
    explanation = explain_input(payload)
    probability = float(prediction["probability_of_susceptibility"])

    (
        workbench_tab,
        drivers_tab,
        evaluation_tab,
        monitoring_tab,
        architecture_tab,
        governance_tab,
    ) = st.tabs(["Workbench", "Drivers", "Evaluation", "Monitoring", "Architecture", "Governance"])

    with workbench_tab:
        _render_prediction_workbench(prediction, probability, explanation, metadata)
        _render_model_operations(metadata)

    with drivers_tab:
        _render_model_drivers()

    with evaluation_tab:
        _render_evaluation(evaluation, metadata)

    with monitoring_tab:
        _render_monitoring(drift)

    with architecture_tab:
        _render_architecture()

    with governance_tab:
        _render_governance()


if __name__ == "__main__":
    main()
