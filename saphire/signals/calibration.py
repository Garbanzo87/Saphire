"""Judge calibration: how well do model-graded (or learned) scores agree with human labels?

Pairs scores on the same rollout/trace and reports agreement, Cohen's kappa, precision/recall of the judge treated
as a classifier, the best decision threshold, and a reliability table (judge-score bins vs human positive rate).
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session


def calibrate(judge: list[float], human: list[float], threshold: float = 0.5, human_threshold: float = 0.5) -> dict[str, Any]:
    n = len(judge)
    if n == 0 or n != len(human):
        return {"n": 0, "note": "no paired labels"}
    jb = [1 if j >= threshold else 0 for j in judge]
    hb = [1 if h >= human_threshold else 0 for h in human]
    agree = sum(1 for a, b in zip(jb, hb) if a == b) / n
    pj, ph = sum(jb) / n, sum(hb) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    tp = sum(1 for a, b in zip(jb, hb) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(jb, hb) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(jb, hb) if a == 0 and b == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    # best threshold by agreement
    best_t, best_a = threshold, agree
    for t in sorted(set(judge)):
        a = sum(1 for j, h in zip(judge, hb) if (1 if j >= t else 0) == h) / n
        if a > best_a:
            best_t, best_a = t, a
    bins: dict[str, dict[str, float]] = {}
    for j, h in zip(judge, hb):
        k = f"{min(int(j * 5), 4) / 5:.1f}-{min(int(j * 5), 4) / 5 + 0.2:.1f}"
        b = bins.setdefault(k, {"n": 0, "human_positive": 0})
        b["n"] += 1
        b["human_positive"] += h
    reliability = {k: {"n": v["n"], "human_positive_rate": v["human_positive"] / v["n"]} for k, v in sorted(bins.items())}
    return {"n": n, "agreement": agree, "kappa": kappa, "precision": prec, "recall": rec, "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "threshold": threshold, "best_threshold": best_t, "best_agreement": best_a, "reliability": reliability}


def calibrate_from_db(db: Session, project_id: str, judge_name: str = "judge_score", human_sources: tuple[str, ...] = ("human", "sdk", "api", "product"),
                      human_name: Optional[str] = None) -> dict[str, Any]:
    from ..server import db as D

    judge = {}
    for s in db.query(D.Score).filter_by(project_id=project_id, name=judge_name):
        if s.rollout_id:
            judge[s.rollout_id] = s.value
    human: dict[str, list[float]] = {}
    q = db.query(D.Score).filter(D.Score.project_id == project_id, D.Score.source.in_(human_sources))
    if human_name:
        q = q.filter(D.Score.name == human_name)
    for s in q:
        if s.rollout_id:
            human.setdefault(s.rollout_id, []).append(s.value)
    keys = [k for k in judge if k in human]
    return calibrate([judge[k] for k in keys], [sum(human[k]) / len(human[k]) for k in keys]) | {"judge": judge_name, "paired_rollouts": len(keys)}
