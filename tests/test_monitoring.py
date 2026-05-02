from __future__ import annotations

import json

import pandas as pd

from src.monitoring.drift_report import generate_drift_report


def test_drift_report_writes_markdown_and_json(tmp_path) -> None:
    reference = pd.DataFrame(
        {
            "age": [40, 50, 60],
            "local_resistance_rate": [0.1, 0.2, 0.3],
            "comorbidity_score": [0, 1, 2],
            "pathogen": ["E. coli", "E. coli", "Klebsiella pneumoniae"],
            "antibiotic": ["nitrofurantoin", "ceftriaxone", "ceftriaxone"],
            "ward_type": ["outpatient", "medical", "medical"],
            "susceptible": [1, 1, 0],
        }
    )
    current = reference.copy()
    current.loc[0, "pathogen"] = "Pseudomonas aeruginosa"

    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    output_path = tmp_path / "drift_report.md"
    json_path = tmp_path / "drift_report.json"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    report = generate_drift_report(reference_path, current_path, output_path, json_path)

    assert output_path.exists()
    assert json_path.exists()
    assert "alerts" in report
    assert "numeric_summary" in json.loads(json_path.read_text(encoding="utf-8"))
