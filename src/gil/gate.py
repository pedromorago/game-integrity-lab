"""Regression gate: compare a candidate report with a baseline report.

A model change can raise the headline number and still flag more innocent
players in one segment. The gate compares, per task and per model, the
overall numbers and the per-segment flag rates, each against a tolerance,
and fails when any comparison is out of bounds. Tolerances live in a JSON
file (configs/gate.json).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .data import LEGITIMATE

TASKS = ("player_task", "pair_task")
LEGITIMATE_BY_TASK = {"player_task": LEGITIMATE, "pair_task": ("friends", "other")}


@dataclass(frozen=True)
class Tolerances:
    max_pr_auc_drop: float = 0.03
    max_recall_drop: float = 0.05
    max_fpr_increase: float = 0.005
    # per segment, on the test split
    max_legit_segment_rate_increase: float = 0.01
    max_positive_segment_recall_drop: float = 0.10
    # frozen models on the robustness scenarios
    max_scenario_pr_auc_drop: float = 0.05

    @classmethod
    def load(cls, path: Path | None) -> "Tolerances":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known - {"_comment"}
        if unknown:
            raise ValueError(f"unknown tolerance(s) in {path}: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in raw.items() if k in known})


@dataclass(frozen=True)
class Finding:
    where: str      # e.g. "player_task/gradient_boosting/segment regular"
    metric: str
    baseline: float
    candidate: float
    limit: str      # human-readable tolerance
    failed: bool


@dataclass
class GateResult:
    findings: list[Finding]
    errors: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors and not any(f.failed for f in self.findings)

    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.failed]


def _drop(where: str, metric: str, b: float, c: float, tol: float) -> Finding:
    return Finding(where, metric, b, c, f"drop <= {tol:g}", b - c > tol + 1e-12)


def _rise(where: str, metric: str, b: float, c: float, tol: float) -> Finding:
    return Finding(where, metric, b, c, f"increase <= {tol:g}", c - b > tol + 1e-12)


def compare(baseline: dict, candidate: dict, tol: Tolerances = Tolerances()) -> GateResult:
    findings: list[Finding] = []
    errors: list[str] = []
    for name, rep in (("baseline", baseline), ("candidate", candidate)):
        if rep.get("synthetic") is not True:
            errors.append(f"the {name} report is not marked synthetic; this gate compares reports made by this tool")
    if baseline["data"]["fingerprint"] != candidate["data"]["fingerprint"]:
        errors.append("the reports were computed on different data (fingerprints "
                      f"{baseline['data']['fingerprint']} and {candidate['data']['fingerprint']}); "
                      "a comparison would mix a data change with a model change")
    if baseline["config"]["budget"] != candidate["config"]["budget"]:
        errors.append(f"the false-positive budgets differ ({baseline['config']['budget']} and "
                      f"{candidate['config']['budget']})")
    if errors:
        return GateResult(findings, errors)

    for task in TASKS:
        bt, ct = baseline.get(task), candidate.get(task)
        if not bt:
            continue
        if not ct:
            errors.append(f"{task} is in the baseline and missing from the candidate")
            continue
        legit = LEGITIMATE_BY_TASK[task]
        for model, b in bt["models"].items():
            c = ct["models"].get(model)
            where = f"{task}/{model}"
            if c is None:
                errors.append(f"{where} is in the baseline and missing from the candidate")
                continue
            findings.append(_drop(where, "PR-AUC", b["pr_auc"], c["pr_auc"], tol.max_pr_auc_drop))
            findings.append(_drop(where, "recall", b["recall"], c["recall"], tol.max_recall_drop))
            findings.append(_rise(where, "FPR", b["fpr"], c["fpr"], tol.max_fpr_increase))
            for seg, bs in b["segments"].items():
                cs = c["segments"].get(seg)
                if cs is None:
                    errors.append(f"{where}: segment {seg} is missing from the candidate")
                    continue
                w = f"{where}/segment {seg}"
                if seg in legit:
                    findings.append(_rise(w, "flag rate", bs["rate"], cs["rate"], tol.max_legit_segment_rate_increase))
                else:
                    findings.append(_drop(w, "flag rate", bs["rate"], cs["rate"], tol.max_positive_segment_recall_drop))

    notes: list[str] = []
    for scen, models in (baseline.get("robustness") or {}).items():
        cm = (candidate.get("robustness") or {}).get(scen)
        if cm is None:
            notes.append(f"robustness scenario {scen} is not in the candidate report, so it was not compared")
            continue
        for model, b in models.items():
            if model in cm:
                findings.append(_drop(f"robustness/{scen}/{model}", "PR-AUC", b["pr_auc"], cm[model]["pr_auc"],
                                      tol.max_scenario_pr_auc_drop))
    return GateResult(findings, errors, notes)


def to_markdown(result: GateResult, show_all: bool = False) -> str:
    lines = [f"# Regression gate: {'PASS' if result.passed else 'FAIL'}", ""]
    for e in result.errors:
        lines.append(f"- error: {e}")
    for n in result.notes:
        lines.append(f"- note: {n}")
    if result.errors or result.notes:
        lines.append("")
    rows = result.findings if show_all else result.failures()
    n_fail = len(result.failures())
    lines.append(f"{len(result.findings)} comparisons, {n_fail} out of tolerance.")
    if rows:
        lines += ["", "| Where | Metric | Baseline | Candidate | Change | Tolerance | Result |",
                  "|---|---|---|---|---|---|---|"]
        for f in rows:
            lines.append(f"| {f.where} | {f.metric} | {f.baseline:.4f} | {f.candidate:.4f} | "
                         f"{f.candidate - f.baseline:+.4f} | {f.limit} | {'FAIL' if f.failed else 'ok'} |")
    lines.append("")
    return "\n".join(lines)


def to_dict(result: GateResult) -> dict:
    return {"passed": result.passed, "errors": result.errors, "notes": result.notes, "findings": [asdict(f) for f in result.findings]}
