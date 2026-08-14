"""U0-b hierarchical EB operator posterior and held-out block coverage.

Prior: per (session, task) group, entrywise N(pop, τ²) with τ² the scatter of
fold-TRAIN owners' 120-s operators around the population operator (the V43
gated mean's own hierarchy).  Observation: sub-block fits ~ N(C, v) with v the
within-cell block scatter.  Coverage: fit on 3 of 4 blocks, check the 80%
predictive interval on the held-out block, entrywise, leave-one-block-out.
"""
from __future__ import annotations

import numpy as np

Z80 = 1.2815515655446004  # 80% central normal interval (known-variance reference)


def block_coverage(pop: np.ndarray, tau2: np.ndarray, blocks: np.ndarray,
                   nominal: float = 0.80) -> float:
    """Mean entrywise coverage over the 4 leave-one-block-out rotations.

    The within-cell variance is estimated from only len(kept) blocks, so the
    predictive interval uses the exact t(df = len(kept) - 1) quantile rather
    than the normal one (a z plug-in undercovers badly at df = 2)."""
    from scipy import stats

    blocks = np.asarray(blocks, np.float64)
    count = len(blocks)
    coverages = []
    for held in range(count):
        kept = np.delete(blocks, held, axis=0)
        v = kept.var(axis=0, ddof=1).clip(1e-12)
        quantile = float(stats.t.ppf(0.5 + nominal / 2.0, df=len(kept) - 1))
        observation = kept.mean(axis=0)
        posterior_var = 1.0 / (1.0 / np.maximum(tau2, 1e-12) + len(kept) / v)
        posterior_mean = posterior_var * (pop / np.maximum(tau2, 1e-12)
                                          + len(kept) * observation / v)
        predictive_sd = np.sqrt(posterior_var + v)
        inside = np.abs(blocks[held] - posterior_mean) <= quantile * predictive_sd
        coverages.append(float(inside.mean()))
    return float(np.mean(coverages))


__all__ = ["Z80", "block_coverage"]
