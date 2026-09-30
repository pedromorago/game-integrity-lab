"""The report writer, the regression gate and the command line."""

import copy
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from gil.cli import main
from gil.gate import Tolerances, compare
from gil.report import to_json, to_markdown

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _reports(tmp: str) -> tuple[dict, dict]:
    out = Path(tmp)
    assert main(["generate", "--config", "small", "--seed", "5", "--out", str(out / "data")]) == 0
    assert main(["evaluate", "--data", str(out / "data"), "--out", str(out / "base"), "--no-scenarios"]) == 0
    assert main(["evaluate", "--data", str(out / "data"), "--out", str(out / "cand"), "--no-scenarios",
                 "--exclude", "accuracy"]) == 0
    return (json.loads((out / "base" / "report.json").read_text()),
            json.loads((out / "cand" / "report.json").read_text()))


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    return _reports(str(tmp_path_factory.mktemp("cli")))


def test_report_is_strict_json_and_marked_synthetic(reports):
    base, _ = reports
    assert base["synthetic"] is True
    json.loads(to_json(base))  # parses back without NaN or Infinity
    md = to_markdown(base)
    assert "synthetic" in md and "never proof of cheating" in md
    assert "\u2014" not in md


def test_gate_passes_on_identical_reports(reports):
    base, _ = reports
    result = compare(base, copy.deepcopy(base))
    assert result.passed and result.findings


def test_gate_fails_without_accuracy_features(reports):
    base, cand = reports
    result = compare(base, cand)
    assert not result.passed
    assert any(f.where.startswith("player_task/gradient_boosting") for f in result.failures())


def test_gate_catches_one_segment_getting_worse(reports):
    base, _ = reports
    cand = copy.deepcopy(base)
    seg = cand["player_task"]["models"]["rules"]["segments"]["regular"]
    seg["rate"] += 0.05  # overall numbers untouched
    failed = compare(base, cand).failures()
    assert [(f.where, f.metric) for f in failed] == [("player_task/rules/segment regular", "flag rate")]


def test_gate_refuses_reports_on_different_data(reports):
    base, _ = reports
    cand = copy.deepcopy(base)
    cand["data"]["fingerprint"] = "0" * 16
    result = compare(base, cand)
    assert not result.passed and result.errors


def test_gate_config_file_loads_and_rejects_unknown_keys(tmp_path):
    assert Tolerances.load(ROOT / "configs" / "gate.json").max_pr_auc_drop == 0.03
    bad = tmp_path / "bad.json"
    bad.write_text('{"max_pr_auc_dorp": 0.1}')
    with pytest.raises(ValueError):
        Tolerances.load(bad)


def test_gate_exit_codes(reports, tmp_path):
    base, cand = reports
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    b.write_text(json.dumps(base))
    c.write_text(json.dumps(cand))
    assert main(["gate", str(b), str(b), "--quiet"]) == 0
    assert main(["gate", str(b), str(c), "--markdown", str(tmp_path / "g.md")]) == 1
    assert "FAIL" in (tmp_path / "g.md").read_text()
    assert main(["gate", str(b), str(tmp_path / "missing.json")]) == 2


def test_python_dash_m_runs():
    r = subprocess.run([sys.executable, "-m", "gil", "--help"], capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0 and "generate" in r.stdout
