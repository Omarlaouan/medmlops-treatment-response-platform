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

CASE_PRESETS: dict[str, dict[str, Any]] = {
    "balanced_review": {
        "label": "Balanced urinary review",
        "description": "A common urinary cohort case used as the default product walkthrough.",
        "values": {
            "age": 67,
            "sex": "F",
            "ward_type": "medical",
            "infection_site": "urinary",
            "pathogen": "E. coli",
            "antibiotic": "nitrofurantoin",
            "prior_antibiotic_exposure": 0,
            "comorbidity_score": 2,
            "local_resistance_rate": 0.12,
        },
    },
    "icu_high_resistance": {
        "label": "ICU high-resistance case",
        "description": "A high-risk synthetic profile with prior exposure and elevated local resistance.",
        "values": {
            "age": 78,
            "sex": "M",
            "ward_type": "ICU",
            "infection_site": "bloodstream",
            "pathogen": "Pseudomonas aeruginosa",
            "antibiotic": "ceftriaxone",
            "prior_antibiotic_exposure": 1,
            "comorbidity_score": 5,
            "local_resistance_rate": 0.47,
        },
    },
    "gram_negative_vanco_mismatch": {
        "label": "Gram-negative vancomycin mismatch",
        "description": "A synthetic antimicrobial mismatch scenario that should surface negative drivers.",
        "values": {
            "age": 58,
            "sex": "F",
            "ward_type": "emergency",
            "infection_site": "urinary",
            "pathogen": "Klebsiella pneumoniae",
            "antibiotic": "vancomycin",
            "prior_antibiotic_exposure": 0,
            "comorbidity_score": 1,
            "local_resistance_rate": 0.24,
        },
    },
    "meropenem_stewardship_review": {
        "label": "Meropenem stewardship review",
        "description": "A high-activity antibiotic context with enough risk to keep the output nuanced.",
        "values": {
            "age": 72,
            "sex": "M",
            "ward_type": "medical",
            "infection_site": "respiratory",
            "pathogen": "Klebsiella pneumoniae",
            "antibiotic": "meropenem",
            "prior_antibiotic_exposure": 1,
            "comorbidity_score": 4,
            "local_resistance_rate": 0.31,
        },
    },
}


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


def _html_block(markup: str) -> str:
    """Compact generated HTML so Streamlit Markdown does not render it as code."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


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
        p, li {
            color: #334155;
            line-height: 1.5;
        }
        [data-testid="stTabs"] button {
            font-weight: 650;
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
        .section-intro {
            color: #526173;
            font-size: 0.95rem;
            margin: -0.25rem 0 1rem 0;
            max-width: 920px;
        }
        .case-summary {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.35rem 0 1rem 0;
        }
        .case-item {
            min-height: 84px;
            padding: 0.82rem 0.9rem;
            background: #ffffff;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
        }
        .case-value {
            color: #172033;
            font-weight: 740;
            margin-top: 0.32rem;
            overflow-wrap: anywhere;
        }
        .insight-panel {
            padding: 0.9rem 1rem;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            background: #ffffff;
            margin: 1rem 0;
        }
        .insight-title {
            color: #172033;
            font-weight: 740;
            margin-bottom: 0.35rem;
        }
        .insight-copy {
            color: #526173;
            font-size: 0.92rem;
        }
        .metric-card {
            min-height: 122px;
            padding: 0.9rem 1rem;
            background: #ffffff;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            overflow: hidden;
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
            overflow-wrap: anywhere;
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
            padding: 0.72rem 0;
            border-bottom: 1px solid #e2e8f0;
        }
        .factor-row:last-child {
            border-bottom: 0;
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
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 760;
            white-space: nowrap;
        }
        .status-ok {
            color: #065f46;
            background: #d1fae5;
            border: 1px solid #a7f3d0;
        }
        .status-watch {
            color: #92400e;
            background: #fef3c7;
            border: 1px solid #fde68a;
        }
        .status-alert {
            color: #991b1b;
            background: #fee2e2;
            border: 1px solid #fecaca;
        }
        .alert-row {
            padding: 0.8rem 0.9rem;
            border: 1px solid #fde68a;
            border-radius: 8px;
            background: #fffbeb;
            color: #78350f;
            margin-bottom: 0.55rem;
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
        .artifact-grid,
        .governance-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-bottom: 1rem;
        }
        .governance-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .artifact-card,
        .governance-panel {
            min-height: 118px;
            padding: 0.9rem 1rem;
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            background: #ffffff;
        }
        .artifact-title,
        .governance-title {
            color: #172033;
            font-weight: 740;
            margin-bottom: 0.35rem;
        }
        .artifact-copy,
        .governance-copy {
            color: #526173;
            font-size: 0.88rem;
            line-height: 1.42;
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
            .case-summary,
            .artifact-grid,
            .governance-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 620px) {
            .case-summary,
            .workflow-grid,
            .artifact-grid,
            .governance-grid {
                grid-template-columns: 1fr;
            }
            .metric-value {
                font-size: 1.65rem;
            }
            .probability-segment {
                font-size: 0.66rem;
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


def _apply_case_preset(preset_key: str) -> None:
    """Write the selected case preset into Streamlit widget state."""
    values = CASE_PRESETS[preset_key]["values"]
    for field, value in values.items():
        if field == "prior_antibiotic_exposure":
            st.session_state[f"input_{field}"] = bool(value)
        else:
            st.session_state[f"input_{field}"] = value


def _render_sidebar_inputs() -> tuple[dict[str, Any], str]:
    st.sidebar.title("Input Panel")
    st.sidebar.caption("Synthetic patient and microbiology context")

    preset_options = list(CASE_PRESETS)
    selected_preset = st.sidebar.selectbox(
        "Synthetic case preset",
        options=preset_options,
        format_func=lambda key: CASE_PRESETS[key]["label"],
        key="selected_case_preset",
    )
    if st.session_state.get("_applied_case_preset") != selected_preset:
        _apply_case_preset(selected_preset)
        st.session_state["_applied_case_preset"] = selected_preset

    st.sidebar.caption(CASE_PRESETS[selected_preset]["description"])
    if st.sidebar.button("Reset inputs to selected case", use_container_width=True):
        _apply_case_preset(selected_preset)

    st.sidebar.markdown("### Patient Context")
    age = st.sidebar.slider("Age", min_value=18, max_value=95, key="input_age")
    sex = st.sidebar.selectbox(
        "Sex",
        options=["F", "M"],
        key="input_sex",
    )
    comorbidity_score = st.sidebar.slider(
        "Comorbidity score",
        min_value=0,
        max_value=6,
        key="input_comorbidity_score",
    )

    st.sidebar.markdown("### Care Setting")
    ward_type = st.sidebar.selectbox(
        "Ward type",
        options=WARD_TYPES,
        key="input_ward_type",
    )
    infection_site = st.sidebar.selectbox(
        "Infection site",
        options=INFECTION_SITES,
        key="input_infection_site",
    )

    st.sidebar.markdown("### Microbiology")
    pathogen = st.sidebar.selectbox(
        "Pathogen",
        options=PATHOGENS,
        key="input_pathogen",
    )
    antibiotic = st.sidebar.selectbox(
        "Antibiotic",
        options=ANTIBIOTICS,
        key="input_antibiotic",
    )

    st.sidebar.markdown("### Resistance Context")
    prior_antibiotic_exposure = int(
        st.sidebar.toggle("Prior antibiotic exposure", key="input_prior_antibiotic_exposure")
    )
    local_resistance_rate = st.sidebar.slider(
        "Local resistance rate",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="input_local_resistance_rate",
    )

    payload = {
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
    return payload, selected_preset


def _metric_card(label: str, value: str, help_text: str = "") -> str:
    return _html_block(
        f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value">{html.escape(value)}</div>
          <div class="metric-help">{html.escape(help_text)}</div>
        </div>
        """
    )


def _status_badge(label: str, status: str) -> str:
    status_class = {
        "ok": "status-ok",
        "watch": "status-watch",
        "alert": "status-alert",
    }.get(status, "status-watch")
    return f'<span class="status-badge {status_class}">{html.escape(label)}</span>'


def _prediction_interpretation(level: str, probability: float) -> str:
    if level == "strong candidate":
        return (
            "Within this synthetic benchmark, the selected antibiotic lands in the highest "
            "demo threshold band. Treat this as an ML systems output, not treatment guidance."
        )
    if level == "candidate":
        return (
            "Within this synthetic benchmark, the selected antibiotic clears the candidate "
            "threshold but remains subject to calibration, drift, and clinical-validation limits."
        )
    if level == "uncertain":
        return (
            "The model places this case in the uncertainty band. In a real workflow this would "
            "need additional clinical, microbiology, and stewardship review before any action."
        )
    return (
        "The selected antibiotic falls below the demo candidate threshold in this synthetic "
        f"model run at {_format_percent(probability)} susceptibility probability."
    )


def _render_case_summary(payload: dict[str, Any], selected_preset: str) -> None:
    preset = CASE_PRESETS[selected_preset]
    prior_exposure = "Yes" if payload["prior_antibiotic_exposure"] else "No"
    items = [
        ("Preset", preset["label"]),
        ("Patient context", f"{payload['age']} years, sex {payload['sex']}"),
        ("Care setting", f"{payload['ward_type']} / {payload['infection_site']}"),
        ("Microbiology", f"{payload['pathogen']} + {payload['antibiotic']}"),
        ("Prior exposure", prior_exposure),
        ("Comorbidity", f"score {payload['comorbidity_score']} of 6"),
        ("Local resistance", _format_percent(payload["local_resistance_rate"], 0)),
        ("Use boundary", "synthetic review only"),
    ]
    cards = "".join(
        _html_block(
            f"""
            <div class="case-item">
              <div class="metric-label">{html.escape(label)}</div>
              <div class="case-value">{html.escape(str(value))}</div>
            </div>
            """
        )
        for label, value in items
    )
    st.markdown(f'<div class="case-summary">{cards}</div>', unsafe_allow_html=True)


def _render_prediction_workbench(
    prediction: dict[str, Any],
    probability: float,
    explanation: dict[str, list[str]],
    metadata: dict[str, Any] | None,
    payload: dict[str, Any],
    selected_preset: str,
) -> None:
    level = prediction["recommendation_level"]
    tier_class = _recommendation_class(level)

    st.subheader("Treatment Response Workbench")
    st.markdown(
        """
        <div class="section-intro">
          Explore synthetic patient, microbiology, and resistance inputs; review the model output;
          and inspect the operational evidence that would surround a healthcare ML service.
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_case_summary(payload, selected_preset)

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

    st.markdown(
        f"""
        <div class="insight-panel">
          <div class="insight-title">Model output interpretation</div>
          <div class="insight-copy">{html.escape(_prediction_interpretation(level, probability))}</div>
          <div class="insight-copy" style="margin-top: .45rem"><strong>Warning:</strong>
            {html.escape(prediction["warning"])}
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
    rows = "".join(
        _html_block(
            f"""
            <div class="factor-row">
              <div class="factor-dot {factor_class}"></div>
              <div class="factor-title">{html.escape(item)}</div>
            </div>
            """
        )
        for item in items
    )
    st.markdown(
        _html_block(
            f"""
            <div class="section-card">
              <div class="metric-label">{html.escape(title)}</div>
              <div style="height: .65rem"></div>
              {rows}
            </div>
            """
        ),
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
    mlflow_run_id = metadata.get("mlflow_run_id")
    mlflow_display = f"{mlflow_run_id[:8]}..." if mlflow_run_id else "not logged"
    cols = st.columns(5)
    values = [
        ("Model type", metadata.get("model_type", "n/a"), "baseline classifier"),
        ("Train rows", str(metadata.get("n_train_rows", "n/a")), "reference batch"),
        ("Test rows", str(metadata.get("n_test_rows", "n/a")), "held-out split"),
        ("Feature groups", str(feature_count), "numeric + categorical"),
        ("MLflow run", mlflow_display, "local tracking metadata"),
    ]
    for column, (label, value, help_text) in zip(cols, values, strict=False):
        column.markdown(_metric_card(label, str(value), help_text), unsafe_allow_html=True)


def _clean_feature_name(feature: str) -> str:
    return (
        feature.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .replace("ward type", "ward")
        .replace("local resistance rate", "local resistance")
    )


def _importance_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return pd.DataFrame(
        {
            "driver": df["feature"].map(_clean_feature_name),
            "coefficient": df["coefficient"].map(lambda value: round(float(value), 4)),
            "absolute_weight": df["coefficient"].map(lambda value: round(abs(float(value)), 4)),
        }
    )


def _render_model_drivers() -> None:
    st.subheader("Model Drivers")
    st.markdown(
        """
        <div class="section-intro">
          Global drivers come from logistic-regression coefficients after preprocessing.
          They are useful for ML review and debugging, not causal clinical evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        st.dataframe(
            _importance_dataframe(importance["top_positive_drivers"]),
            width="stretch",
            hide_index=True,
        )
    with negative_col:
        st.markdown("#### Top Negative Coefficients")
        st.dataframe(
            _importance_dataframe(importance["top_negative_drivers"]),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        """
        <div class="insight-panel">
          <div class="insight-title">Explainability boundary</div>
          <div class="insight-copy">
            The local explanation in the workbench is intentionally heuristic. It highlights
            engineered clinical-risk concepts and known synthetic pair effects without claiming
            patient-level causal attribution.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metrics_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": key, "value": _format_number(value, decimals=4)}
            for key, value in metrics.items()
        ]
    )


def _metric_display(metric_name: str, value: Any) -> str:
    if metric_name == "brier_score":
        return _format_number(value, 3)
    return _format_percent(value, 1)


def _threshold_dataframe(thresholds: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(thresholds)
    if df.empty:
        return df
    display_df = df.copy()
    display_df["threshold"] = display_df["threshold"].map(lambda value: f"{float(value):.2f}")
    for column in ["accuracy", "precision", "recall", "f1", "predicted_positive_rate"]:
        if column in display_df:
            display_df[column] = display_df[column].map(lambda value: _format_percent(value, 1))
    return display_df


def _calibration_dataframe(calibration: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(calibration)
    if df.empty:
        return df
    return df.assign(
        probability_bin=df.apply(
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


def _render_evaluation(evaluation: dict[str, Any] | None, metadata: dict[str, Any] | None) -> None:
    st.subheader("Evaluation")
    st.markdown(
        """
        <div class="section-intro">
          This tab turns the generated evaluation artifact into an ML review surface:
          global metrics, threshold tradeoffs, calibration, and slice behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )
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

    metrics = evaluation.get("global_metrics", {})
    metric_cards = [
        ("ROC-AUC", "roc_auc", metrics.get("roc_auc"), "ranking quality"),
        ("Accuracy", "accuracy", metrics.get("accuracy"), "0.50 threshold"),
        ("Precision", "precision", metrics.get("precision"), "candidate reliability"),
        ("Recall", "recall", metrics.get("recall"), "susceptible capture"),
        ("F1", "f1", metrics.get("f1"), "precision-recall balance"),
        ("Brier", "brier_score", metrics.get("brier_score"), "probability calibration"),
    ]
    metric_columns = st.columns(6)
    for column, (label, metric_key, value, help_text) in zip(
        metric_columns, metric_cards, strict=False
    ):
        column.markdown(
            _metric_card(label, _metric_display(metric_key, value), help_text),
            unsafe_allow_html=True,
        )

    threshold_col, calibration_col = st.columns([1.05, 0.95])
    with threshold_col:
        st.markdown("#### Threshold Tradeoffs")
        threshold_df = _threshold_dataframe(evaluation.get("threshold_metrics", []))
        if not threshold_df.empty:
            st.dataframe(threshold_df, width="stretch", hide_index=True)
        st.caption("Thresholds create demo recommendation bands; they are not clinical policies.")

    with calibration_col:
        st.markdown("#### Calibration Summary")
        calibration_df = _calibration_dataframe(evaluation.get("calibration_summary", []))
        if not calibration_df.empty:
            chart_df = calibration_df.set_index("probability_bin")[
                ["mean_predicted_probability", "observed_susceptible_rate"]
            ]
            st.line_chart(chart_df)
            st.dataframe(
                calibration_df.assign(
                    mean_predicted_probability=calibration_df[
                        "mean_predicted_probability"
                    ].map(lambda value: _format_percent(value, 1)),
                    observed_susceptible_rate=calibration_df[
                        "observed_susceptible_rate"
                    ].map(lambda value: _format_percent(value, 1)),
                ),
                width="stretch",
                hide_index=True,
            )

    slice_metrics = evaluation.get("slice_metrics", {})
    if slice_metrics:
        st.markdown("#### Slice Evaluation")
        selected_slice = st.selectbox(
            "Slice dimension",
            options=list(slice_metrics.keys()),
            index=0,
            key="evaluation_slice_dimension",
        )
        slice_df = pd.DataFrame(slice_metrics[selected_slice])
        if not slice_df.empty:
            for column in ["accuracy", "precision", "recall", "f1"]:
                if column in slice_df:
                    slice_df[column] = slice_df[column].map(lambda value: _format_percent(value, 1))
            st.dataframe(slice_df, width="stretch", hide_index=True)


def _numeric_drift_dataframe(drift: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in drift.get("numeric_summary", []):
        mean_delta = float(item.get("mean_difference", 0.0))
        threshold = float(item.get("threshold", 0.0))
        rows.append(
            {
                "column": item.get("column"),
                "reference_mean": _format_number(item.get("reference_mean"), 3),
                "current_mean": _format_number(item.get("current_mean"), 3),
                "mean_delta": _format_number(mean_delta, 3),
                "threshold": _format_number(threshold, 3),
                "status": "Alert" if abs(mean_delta) > threshold else "OK",
            }
        )
    return pd.DataFrame(rows)


def _categorical_drift_dataframe(drift: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in drift.get("categorical_summary", []):
        new_categories = item.get("new_categories", [])
        max_delta = float(item.get("max_distribution_delta", 0.0))
        rows.append(
            {
                "column": item.get("column"),
                "new_categories": ", ".join(new_categories) if new_categories else "none",
                "max_distribution_delta": _format_percent(max_delta, 1),
                "status": "Alert" if new_categories else "Watch" if max_delta > 0.05 else "OK",
            }
        )
    return pd.DataFrame(rows)


def _missingness_dataframe(drift: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in drift.get("missingness_summary", []):
        delta = float(item.get("missing_rate_delta", 0.0))
        rows.append(
            {
                "column": item.get("column"),
                "reference_missing_rate": _format_percent(item.get("reference_missing_rate"), 1),
                "current_missing_rate": _format_percent(item.get("current_missing_rate"), 1),
                "missing_rate_delta": _format_percent(delta, 1),
                "status": "Alert" if abs(delta) > 0.05 else "Watch" if abs(delta) > 0.01 else "OK",
            }
        )
    return pd.DataFrame(rows)


def _render_monitoring(drift: dict[str, Any] | None) -> None:
    st.subheader("Monitoring")
    st.markdown(
        """
        <div class="section-intro">
          The monitor compares the current cohort against the training reference batch and
          turns simple drift checks into operational review signals.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not drift:
        st.markdown(
            '<div class="empty-state">Drift artifact not found. Run <code>make monitor</code>.</div>',
            unsafe_allow_html=True,
        )
        return

    alerts = drift.get("alerts", [])
    has_alert = bool(alerts)
    overall_status = "alert" if has_alert else "ok"
    status_label = "Action needed" if has_alert else "Healthy"

    alert_col, target_col, rows_col = st.columns(3)
    alert_col.markdown(
        _metric_card("Monitor status", status_label, f"{len(alerts)} active alert(s)"),
        unsafe_allow_html=True,
    )
    target = drift.get("target_summary") or {}
    target_delta = float(target.get("target_rate_delta", 0.0))
    target_col.markdown(
        _metric_card(
            "Current target rate",
            _format_percent(target.get("current_target_rate")),
            f"Delta {_format_percent(target_delta, 2)}",
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

    st.markdown(
        f"""
        <div class="insight-panel">
          <div class="insight-title">Operational status {_status_badge(status_label, overall_status)}</div>
          <div class="insight-copy">
            Target-rate drift is {_format_percent(target_delta, 2)} against a simple alert threshold of
            5 percentage points. This is a lightweight MVP monitor, not a replacement for production
            observability or clinical governance.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if alerts:
        for alert in alerts:
            severity = str(alert.get("severity", "watch")).lower()
            status = "alert" if severity in {"high", "critical", "alert"} else "watch"
            st.markdown(
                f"""
                <div class="alert-row">
                  {_status_badge(severity.upper(), status)}
                  <span style="margin-left: .45rem">{html.escape(alert.get("message", ""))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("No drift alerts triggered by the current simple monitor.")

    table_tabs = st.tabs(["Numeric", "Categorical", "Missingness"])
    with table_tabs[0]:
        st.dataframe(_numeric_drift_dataframe(drift), width="stretch", hide_index=True)
    with table_tabs[1]:
        st.dataframe(_categorical_drift_dataframe(drift), width="stretch", hide_index=True)
    with table_tabs[2]:
        st.dataframe(_missingness_dataframe(drift), width="stretch", hide_index=True)


def _render_architecture() -> None:
    st.subheader("MLOps Architecture")
    st.markdown(
        """
        <div class="section-intro">
          The app is the visible layer of a reproducible healthcare ML system: validated synthetic
          data, a trained model artifact, serving code, monitoring reports, and governance docs.
        </div>
        """,
        unsafe_allow_html=True,
    )
    artifacts = [
        ("Data pipeline", "CLI modules generate, validate, cohort, split, and materialize batches."),
        ("Model lifecycle", "Training writes model, metadata, metrics, predictions, and MLflow traces."),
        ("Serving layer", "FastAPI exposes health, metadata, model-info, and prediction endpoints."),
        ("Demo interface", "Streamlit calls local inference and renders explainability + operations."),
        ("Monitoring", "Reference/current batch comparison emits markdown and JSON drift artifacts."),
        ("Governance", "Model card and safety note document limitations and intended boundaries."),
    ]
    artifact_cards = "".join(
        _html_block(
            f"""
            <div class="artifact-card">
              <div class="artifact-title">{html.escape(title)}</div>
              <div class="artifact-copy">{html.escape(copy)}</div>
            </div>
            """
        )
        for title, copy in artifacts
    )
    st.markdown(f'<div class="artifact-grid">{artifact_cards}</div>', unsafe_allow_html=True)

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
    cards = "".join(
        _html_block(
            f"""
            <div class="workflow-step">
              <div class="workflow-index">{index}</div>
              <div class="workflow-title">{html.escape(title)}</div>
              <div class="workflow-copy">{html.escape(copy)}</div>
            </div>
            """
        )
        for index, title, copy in steps
    )
    st.markdown(f'<div class="workflow-grid">{cards}</div>', unsafe_allow_html=True)


def _render_governance() -> None:
    st.subheader("Governance and Limitations")
    panels = [
        (
            "Intended use",
            "Portfolio demonstration of healthcare ML engineering: reproducible synthetic data, "
            "training, serving, explainability, monitoring, tests, and documentation.",
        ),
        (
            "Not intended use",
            "No clinical decision-making, diagnosis, treatment selection, dosing, allergy review, "
            "breakpoint interpretation, or patient-level care workflow.",
        ),
        (
            "Clinical limitations",
            "Synthetic data only; no PHI, no external validation, no clinical calibration, no subgroup "
            "safety analysis, and no stewardship governance approval.",
        ),
        (
            "Production readiness gap",
            "A real deployment would require credentialed datasets, FHIR or OMOP integration, "
            "monitoring escalation paths, external validation, and prospective clinical review.",
        ),
    ]
    panel_markup = "".join(
        _html_block(
            f"""
            <div class="governance-panel">
              <div class="governance-title">{html.escape(title)}</div>
              <div class="governance-copy">{html.escape(copy)}</div>
            </div>
            """
        )
        for title, copy in panels
    )
    st.markdown(f'<div class="governance-grid">{panel_markup}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-panel">
          <div class="insight-title">Documentation artifacts</div>
          <div class="insight-copy">
            Review <code>reports/model_card.md</code>, <code>reports/clinical_safety_note.md</code>,
            <code>reports/evaluation_report.md</code>, and <code>reports/drift_report.md</code>
            for the repo-level governance story.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="MedMLOps Treatment Response Platform", layout="wide")
    _inject_css()

    metadata = _load_json(MODEL_METADATA_PATH)
    evaluation = _load_json(EVALUATION_JSON_PATH)
    drift = _load_json(DRIFT_JSON_PATH)
    _render_header(metadata, MODEL_PATH.exists())
    payload, selected_preset = _render_sidebar_inputs()

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
        _render_prediction_workbench(
            prediction,
            probability,
            explanation,
            metadata,
            payload,
            selected_preset,
        )
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
