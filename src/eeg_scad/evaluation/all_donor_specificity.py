"""All-owner specificity summaries with participant as the biological unit."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Any

import numpy as np


def donor_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["recipient"]))].append(row)
    result = []
    for (method, recipient), values in sorted(groups.items()):
        risk = {str(row["donor"]): float(row["risk"]) for row in values}
        if recipient not in risk or len(risk) != 15:
            raise RuntimeError(f"incomplete donor family {method}/{recipient}: {len(risk)}")
        correct = risk[recipient]
        wrong = np.asarray([value for donor, value in risk.items() if donor != recipient])
        ordered = sorted(risk, key=lambda donor: (risk[donor], donor))
        result.append({
            "method": method, "recipient": recipient, "correct_risk": correct,
            "mean_wrong_risk": float(wrong.mean()), "median_wrong_risk": float(np.median(wrong)),
            "best_wrong_risk": float(wrong.min()), "worst_wrong_risk": float(wrong.max()),
            "correct_rank": ordered.index(recipient) + 1, "correct_is_best": int(ordered[0] == recipient),
            "correct_minus_mean_wrong_utility": float(wrong.mean() - correct),
            "correct_minus_best_wrong_utility": float(wrong.min() - correct),
        })
    return result


def group_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows); result = []
    for method in sorted({str(row["method"]) for row in rows}):
        chosen = [row for row in rows if row["method"] == method]
        ranks = np.asarray([float(row["correct_rank"]) for row in chosen])
        effects = np.asarray([float(row["correct_minus_mean_wrong_utility"]) for row in chosen])
        rng = np.random.Generator(np.random.PCG64DXSM(20260931))
        boot = np.asarray([rng.choice(effects, len(effects), replace=True).mean() for _ in range(20000)])
        result.append({
            "method": method, "participants": len(chosen), "mean_correct_rank": float(ranks.mean()),
            "median_correct_rank": float(np.median(ranks)), "correct_top1": int(np.sum(ranks == 1)),
            "correct_top3": int(np.sum(ranks <= 3)), "mean_correct_minus_mean_wrong": float(effects.mean()),
            "bootstrap_low": float(np.quantile(boot, .025)), "bootstrap_high": float(np.quantile(boot, .975)),
        })
    return result


__all__ = ["donor_summary", "group_summary"]

