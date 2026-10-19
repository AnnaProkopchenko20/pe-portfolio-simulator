"""Optimizer tests.

The first block uses an ARTIFICIAL MATHEMATICAL TEST PORTFOLIO, clearly
separate from the investment case: ownership 100%, multiple 1x, zero baseline
and debt, instant implementation and a single certain scenario, so that fund
proceeds = EBITDA uplift = a known function of investment and the true optimum
can be derived by hand.
"""
import math

import numpy as np
import pytest

from assumptions import (
    COMPANY_KEYS,
    CaseCompany,
    CompanyAssumptions,
    FocusSpec,
    LiquiditySpec,
    ModelAssumptions,
    ResponseSpec,
    ScenarioInputs,
    TimingSpec,
    illustrative_assumptions,
)
from financial_model import evaluate_portfolio
from optimizer import (
    breakeven_response_scale,
    company_grid,
    marginal_values,
    one_way_sensitivity,
    optimize,
    sweep_values,
    two_way_sensitivity,
)


# ---------------------------------------------------------------------------
# Artificial test portfolio (NOT the case)
# ---------------------------------------------------------------------------
def toy(responses: dict[str, ResponseSpec], budget=15.0, caps=(10.0, 8.0, 15.0),
        nonfocus_mult=1.0, nonfocus_cap=15.0) -> ModelAssumptions:
    case = {k: CaseCompany(k, f"Toy {k}", f"Toy {k}", 1, 1, 1, 1, 1, 0.0, 1.0, cap)
            for k, cap in zip(COMPANY_KEYS, caps)}
    instant = TimingSpec(start_month=0, implementation_months=0, benefit_lag_months=0, ramp_months=0)
    sc = {s: ScenarioInputs(baseline_ebitda=0.0, exit_multiple=1.0, exit_net_debt=0.0) for s in
          ("downside", "base", "upside")}
    comps = {k: CompanyAssumptions(responses[k], instant, {s: ScenarioInputs(**vars(v)) for s, v in sc.items()})
             for k in COMPANY_KEYS}
    return ModelAssumptions(
        companies=comps, probabilities={"downside": 0.0, "base": 100.0, "upside": 0.0},
        valuation_month=24, focus=FocusSpec(1.0, nonfocus_mult, nonfocus_cap, 0.0),
        liquidity=LiquiditySpec(requirement=0.0), budget=budget, case=case)


def lin(slope_per_m, X):
    return ResponseSpec(shape="linear", full_effect_investment=X, max_ebitda_uplift=slope_per_m * X)


def test_toy_linear_known_optimum():
    # Proceeds per $1M: A 2.0, B 1.5, C 0.5. Net per $1M: +1.0, +0.5, -0.5.
    # Budget 15: fill A to its cap 10, then B with the remaining 5. C gets nothing.
    a = toy({"ecom": lin(2.0, 10), "it": lin(1.5, 8), "sec": lin(0.5, 15)})
    r = optimize(a, "expected_net", focus="ecom")
    assert r.best.allocation == {"ecom": 10.0, "it": 5.0, "sec": 0.0}
    assert r.best.expected_net_gain == pytest.approx(10 * 1.0 + 5 * 0.5)


def test_toy_concave_known_optimum():
    # A: g(x) = A0 (1 - e^{-kx}) / c with c = 1 - e^{-kX}. Optimum where g'(x) = 1:
    #    x* = ln(A0 k / c) / k   (interior; budget not binding).
    A0, k, X = 12.0, 0.4, 10.0
    c = 1 - math.exp(-k * X)
    x_star = math.log(A0 * k / c) / k
    zero = ResponseSpec(shape="linear", full_effect_investment=1, max_ebitda_uplift=0)
    a = toy({"ecom": ResponseSpec(shape="diminishing", full_effect_investment=X, curvature=k, max_ebitda_uplift=A0),
             "it": zero, "sec": zero})
    r = optimize(a, "expected_net", focus="ecom", step=0.25)
    assert abs(r.best.allocation["ecom"] - x_star) <= 0.125 + 1e-9
    assert r.best.allocation["it"] == 0 and r.best.allocation["sec"] == 0


def test_toy_threshold_found_by_global_search():
    # A pays 20 only once 6 is invested (nothing before). B pays 1.5 per $1M up to 8.
    # Budget 10: A=6 (+14 net) and B=4 (+2 net) = 16 beats B=8 alone (+4).
    # A local/greedy method starting at zero would never cross A's threshold.
    thr = ResponseSpec(shape="threshold", full_effect_investment=6, threshold=6, jump=1.0, max_ebitda_uplift=20)
    zero = ResponseSpec(shape="linear", full_effect_investment=1, max_ebitda_uplift=0)
    a = toy({"ecom": thr, "it": lin(1.5, 8), "sec": zero}, budget=10)
    r = optimize(a, "expected_net", focus="ecom")
    assert r.best.allocation == {"ecom": 6.0, "it": 4.0, "sec": 0.0}
    assert r.best.expected_net_gain == pytest.approx(16.0)


def test_toy_focus_selection():
    # Only the focus company creates value (non-focus multiplier 0). C is best as focus.
    a = toy({"ecom": lin(1.2, 10), "it": lin(1.4, 8), "sec": lin(1.3, 15)}, nonfocus_mult=0.0)
    r = optimize(a, "expected_net", focus=None)
    # C: 15 x 0.3 = 4.5 > A: 10 x 0.2 = 2.0 > B: 8 x 0.4 = 3.2
    assert r.best.focus == "sec"
    assert r.best.allocation == {"ecom": 0.0, "it": 0.0, "sec": 15.0}


def test_toy_nonfocus_cap_respected():
    a = toy({"ecom": lin(2.0, 10), "it": lin(1.5, 8), "sec": lin(1.4, 15)}, nonfocus_cap=2.0)
    r = optimize(a, "expected_net", focus="sec")
    assert r.best.allocation["ecom"] <= 2.0 and r.best.allocation["it"] <= 2.0
    assert r.best.allocation == {"ecom": 2.0, "it": 2.0, "sec": 11.0}


def test_toy_tie_prefers_less_capital():
    # B returns exactly its cost: investing or not is a tie -> keep the cash.
    a = toy({"ecom": lin(2.0, 10), "it": lin(1.0, 8), "sec": lin(0.5, 15)})
    r = optimize(a, "expected_net", focus="ecom")
    assert r.best.allocation == {"ecom": 10.0, "it": 0.0, "sec": 0.0}
    assert r.n_near_optimal > 1          # the tie is reported, not hidden


def test_toy_all_negative_keeps_reserve():
    a = toy({"ecom": lin(0.5, 10), "it": lin(0.9, 8), "sec": lin(0.1, 15)})
    r = optimize(a, "expected_net", focus=None)
    assert r.best.invested == 0.0


def test_roi_objective_excludes_tiny_allocations():
    a = toy({"ecom": ResponseSpec(shape="diminishing", full_effect_investment=10, curvature=1.0, max_ebitda_uplift=20),
             "it": lin(0.0, 8), "sec": lin(0.0, 15)})
    r = optimize(a, "roi", focus="ecom", min_roi_investment=1.0)
    assert r.best.invested >= 1.0 - 1e-9
    assert np.isfinite(r.best.score)
    assert any("ROI" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Constraints on the actual case grid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("objective", ["expected_net", "roi", "downside_net", "robust"])
def test_constraints_respected(objective):
    a = illustrative_assumptions()
    r = optimize(a, objective, focus=None)
    for c in [r.best] + r.near_optimal:
        assert c.invested <= 15.0 + 1e-9
        for k in COMPANY_KEYS:
            assert 0 <= c.allocation[k] <= a.case[k].request + 1e-9
            if k != c.focus:
                assert c.allocation[k] <= a.focus.nonfocus_cap + 1e-9


def test_grid_respects_requests_and_step():
    a = illustrative_assumptions()
    g = company_grid(a, "ecom", "ecom", 0.25)
    assert g[0] == 0 and g[-1] == 10.0 and len(g) == 41
    g = company_grid(a, "sec", "ecom", 0.25)          # non-focus: capped
    assert g[-1] == a.focus.nonfocus_cap


def test_optimum_beats_every_grid_point_bruteforce():
    """Cross-check the broadcast search with a slow explicit loop (coarse grid)."""
    a = illustrative_assumptions()
    r = optimize(a, "expected_net", focus="it", step=1.0)
    best = -1e18
    for x1 in range(0, 8):
        for x2 in range(0, 9):
            for x3 in range(0, 8):
                if x1 + x2 + x3 > 15:
                    continue
                s = evaluate_portfolio(a, {"ecom": x1, "it": x2, "sec": x3}, "it").summary()
                best = max(best, s["net_cash_returned"])
    assert r.best.score == pytest.approx(best)


def test_deterministic():
    a = illustrative_assumptions()
    r1, r2 = optimize(a), optimize(a)
    assert r1.best == r2.best


def test_optimum_changes_with_assumptions():
    a = illustrative_assumptions()
    r0 = optimize(a)
    b = a.copy()
    b.companies["sec"].response.max_multiple_uplift *= 2.5
    r1 = optimize(b)
    assert r1.best.allocation["sec"] > r0.best.allocation["sec"]


def test_marginal_values_and_sensitivity_run():
    a = illustrative_assumptions()
    r = optimize(a)
    mv = marginal_values(a, r.best.allocation, r.best.focus)
    assert set(mv) == set(COMPANY_KEYS)
    rows = one_way_sensitivity(a, "sec:response_magnitude", sweep_values("sec:response_magnitude", 5))
    assert len(rows) == 5 and all(row["ecom"] + row["it"] + row["sec"] <= 15 + 1e-9 for row in rows)
    grid = two_way_sensitivity(a, "ecom:exit_multiple", [0.8, 1.2], "prob:downside", [10, 40], step=1.0)
    assert len(grid) == 2 and len(grid[0]) == 2
    sc = breakeven_response_scale(a, "sec")
    assert sc is None or sc >= 0
