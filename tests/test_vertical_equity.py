import numpy as np
import pandas as pd

from openavmkit.vertical_equity_study import get_vertical_equity_scores


def _make(cheap_mean, cheap_sd, exp_mean, exp_sd, n=20, seed=42):
    """Two clean price tiers (cheap vs expensive) with controllable ratio spread.

    Bimodal sale price -> the market_proxy used internally splits the rows
    cleanly into a low tier and a high tier, so we can dial the per-tier ratio
    confidence intervals to be either well-separated or overlapping.
    """
    rng = np.random.default_rng(seed)
    price = np.concatenate([np.full(n, 100_000.0), np.full(n, 500_000.0)])
    ratio = np.concatenate([
        rng.normal(cheap_mean, cheap_sd, n),
        rng.normal(exp_mean, exp_sd, n),
    ])
    return pd.DataFrame({"sale": price, "val": ratio * price})


def test_vei_significance_distinguishes_significant_from_not():
    # Tight, non-overlapping tier CIs -> the regressivity is statistically real.
    sig = get_vertical_equity_scores(_make(1.20, 0.02, 0.80, 0.02), "sale", "val")
    # Wide, overlapping tier CIs -> regressive point estimate but NOT significant.
    nsig = get_vertical_equity_scores(_make(1.02, 0.35, 0.98, 0.35), "sale", "val")

    # Both are regressive in the point estimate (cheap tier over-valued).
    assert sig["vei"] < 0
    assert nsig["vei"] < 0

    # vei_significance must distinguish the two: it keeps vei's (negative) sign
    # only when the extreme tiers' CIs do not overlap, and flips positive when
    # they overlap. (The pre-fix outer-bounds formula stayed negative in both
    # cases and could not tell them apart.)
    assert sig["vei_significance"] < 0, "non-overlapping tier CIs should read as significant"
    assert nsig["vei_significance"] > 0, "overlapping tier CIs should read as not significant"
