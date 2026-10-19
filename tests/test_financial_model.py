"""Tests for case data, valuation chain, response functions and accounting identities."""
import numpy as np
import pytest

from assumptions import (
    CASE_BUDGET,
    CASE_COMPANIES,
    CASE_TOTAL_REQUEST,
    COMPANY_KEYS,
    SCENARIOS,
    ResponseSpec,
    illustrative_assumptions,
    parse_custom_points,
)
from financial_model import (
    AssumptionError,
    company_values,
    evaluate_portfolio,
    expected_bridge,
    response_fraction,
    timing_effect,
    validate_probabilities,
)


# 1. Original case-data accuracy -------------------------------------------
def test_case_data_matches_datapack():
    e, i, s = CASE_COMPANIES["ecom"], CASE_COMPANIES["it"], CASE_COMPANIES["sec"]
    assert (e.prev_revenue, e.latest_revenue, e.prev_ebitda, e.latest_ebitda) == (700, 600, 35, 12)
    assert (e.latest_margin, e.net_debt, e.ownership, e.request) == (0.02, 35, 0.30, 10)
    assert (i.prev_revenue, i.latest_revenue, i.prev_ebitda, i.latest_ebitda) == (200, 180, 36, 21.6)
    assert (i.latest_margin, i.net_debt, i.ownership, i.request) == (0.12, 0, 0.30, 8)
    assert (s.prev_revenue, s.latest_revenue, s.prev_ebitda, s.latest_ebitda) == (250, 325, 50, 65)
    assert (s.latest_margin, s.net_debt, s.ownership, s.request) == (0.20, 10, 0.20, 15)
    assert CASE_BUDGET == 15 and CASE_TOTAL_REQUEST == 33
    assert sum(c.request for c in CASE_COMPANIES.values()) == CASE_TOTAL_REQUEST
    # stated margins are consistent with the figures (rounded as in the case)
    for c in CASE_COMPANIES.values():
        assert round(c.latest_ebitda / c.latest_revenue, 2) == c.latest_margin


# 4. Zero-investment baseline ----------------------------------------------
def test_zero_investment_equals_counterfactual():
    a = illustrative_assumptions()
    r = evaluate_portfolio(a, {k: 0.0 for k in COMPANY_KEYS}, "ecom")
    s = r.summary()
    assert s["incremental_proceeds"] == pytest.approx(0.0)
    assert s["net_gain"] == pytest.approx(0.0)
    assert s["roi"] is None
    assert r.reserve == pytest.approx(CASE_BUDGET)


# 5-7. EV, equity, ownership -----------------------------------------------
def test_valuation_chain_by_hand():
    a = illustrative_assumptions()
    sc = a.companies["it"].scenarios["base"]
    v = company_values(a, "it", "base", [0.0], is_focus=False)
    ev = sc.baseline_ebitda * sc.exit_multiple
    eq = max(ev - sc.exit_net_debt, 0)
    assert v["ev"][0] == pytest.approx(ev)
    assert v["equity"][0] == pytest.approx(eq)
    assert v["proceeds"][0] == pytest.approx(eq * 0.30)


def test_ownership_fixed_and_equity_mode():
    a = illustrative_assumptions()
    v = company_values(a, "sec", "base", [5.0], True)
    assert v["ownership"][0] == pytest.approx(0.20)
    a.financing_mode = "equity"
    a.companies["sec"].pre_money_equity = 100.0
    v = company_values(a, "sec", "base", [5.0], True)
    assert v["ownership"][0] == pytest.approx((0.2 * 100 + 5) / 105)
    v0 = company_values(a, "sec", "base", [0.0], True)
    assert v0["ownership"][0] == pytest.approx(0.20)


# 8 + 16. Net benefit and no double counting --------------------------------
def test_net_benefit_identity_and_no_double_counting():
    a = illustrative_assumptions()
    alloc = {"ecom": 5.0, "it": 3.0, "sec": 7.0}
    r = evaluate_portfolio(a, alloc, "sec")
    for scen in list(SCENARIOS) + [None]:
        s = r.summary(scen)
        assert s["net_gain"] == pytest.approx(s["total_proceeds"] - s["baseline_proceeds"] - 15.0)
        assert s["net_cash_returned"] == pytest.approx(s["total_proceeds"] - 15.0)


def test_capital_with_no_effect_is_pure_cost():
    """Capital spent on a programme with zero effect must not reappear as exit cash."""
    a = illustrative_assumptions()
    for k in COMPANY_KEYS:
        a.companies[k].response.max_ebitda_uplift = 0.0
        a.companies[k].response.max_multiple_uplift = 0.0
    r = evaluate_portfolio(a, {"ecom": 4.0, "it": 2.0, "sec": 3.0}, "ecom")
    s = r.summary()
    assert s["incremental_proceeds"] == pytest.approx(0.0)
    assert s["net_gain"] == pytest.approx(-9.0)
    for scen in SCENARIOS:
        for k in COMPANY_KEYS:
            assert r.by_scenario[scen][k]["net_debt"] == pytest.approx(r.baseline[scen][k]["net_debt"])


def test_value_bridge_adds_up():
    a = illustrative_assumptions()
    a.liquidity.requirement = 3.0
    r = evaluate_portfolio(a, {"ecom": 5.0, "it": 3.0, "sec": 7.0}, "sec")
    for k in COMPANY_KEYS:
        b = expected_bridge(r, k)
        parts = b["ebitda"] + b["multiple"] + b["net_debt"] + b["ownership"]
        assert parts == pytest.approx(b["total"])
        inc = r.company_expected(k, "proceeds") - r.company_expected(k, "proceeds", True)
        assert b["total"] == pytest.approx(inc)


# 9. Scenario probabilities ---------------------------------------------------
def test_probabilities_must_sum_to_100():
    assert validate_probabilities({"downside": 25, "base": 50, "upside": 25})["base"] == pytest.approx(0.5)
    with pytest.raises(AssumptionError):
        validate_probabilities({"downside": 30, "base": 50, "upside": 25})
    with pytest.raises(AssumptionError):
        validate_probabilities({"downside": -10, "base": 85, "upside": 25})
    a = illustrative_assumptions()
    a.probabilities = {"downside": 10, "base": 10, "upside": 10}
    with pytest.raises(AssumptionError):
        evaluate_portfolio(a, {k: 0.0 for k in COMPANY_KEYS}, "ecom")


def test_expected_value_is_probability_weighted():
    a = illustrative_assumptions()
    r = evaluate_portfolio(a, {"ecom": 5.0, "it": 3.0, "sec": 7.0}, "ecom")
    manual = 0.25 * r.proceeds("downside") + 0.5 * r.proceeds("base") + 0.25 * r.proceeds("upside")
    assert r.summary()["total_proceeds"] == pytest.approx(manual)


# 10-11. Response functions ----------------------------------------------------
X = np.linspace(0, 12, 481)


@pytest.mark.parametrize("spec", [
    ResponseSpec(shape="linear", full_effect_investment=10),
    ResponseSpec(shape="diminishing", full_effect_investment=10, curvature=0.4),
    ResponseSpec(shape="threshold", full_effect_investment=9, threshold=4, jump=0.5, below_threshold=0.05),
    ResponseSpec(shape="s_curve", full_effect_investment=10, midpoint=5, steepness=1.2),
    ResponseSpec(shape="custom", custom_points=[(2, 0.1), (6, 0.8), (10, 1.0)]),
])
def test_response_basic_properties(spec):
    f = response_fraction(X, spec)
    assert f[0] == 0.0
    assert np.all(np.diff(f) >= -1e-12)          # monotone non-decreasing
    assert np.all((f >= 0) & (f <= 1 + 1e-12))
    assert response_fraction([10.0], spec)[0] == pytest.approx(1.0)


def test_linear_is_proportional():
    spec = ResponseSpec(shape="linear", full_effect_investment=8)
    assert response_fraction([2.0, 4.0], spec) == pytest.approx([0.25, 0.5])


def test_diminishing_is_concave():
    spec = ResponseSpec(shape="diminishing", full_effect_investment=10, curvature=0.5)
    f = response_fraction(np.linspace(0, 10, 101), spec)
    assert np.all(np.diff(f, 2) <= 1e-12)
    assert f[50] > 0.5                              # more than half of the effect from half the money


def test_threshold_behaviour():
    spec = ResponseSpec(shape="threshold", full_effect_investment=9, threshold=6, jump=0.6, below_threshold=0.0)
    f = response_fraction([2.0, 5.99, 6.0, 7.5, 9.0, 10.0], spec)
    assert f[0] == 0 and f[1] == 0                  # nothing below the threshold
    assert f[2] == pytest.approx(0.6)               # jump at the threshold
    assert f[3] == pytest.approx(0.8)
    assert f[4] == pytest.approx(1.0) and f[5] == pytest.approx(1.0)


def test_s_curve_is_convex_then_concave():
    spec = ResponseSpec(shape="s_curve", full_effect_investment=10, midpoint=5, steepness=1.0)
    f = response_fraction(np.linspace(0, 10, 101), spec)
    d2 = np.diff(f, 2)
    assert np.all(d2[:40] > 0) and np.all(d2[-40:] < 0)


def test_custom_points_parse_and_interpolate():
    pts = parse_custom_points("2:10, 6:80, 10:100")
    spec = ResponseSpec(shape="custom", custom_points=pts)
    assert response_fraction([4.0], spec)[0] == pytest.approx(0.45)
    assert response_fraction([15.0], spec)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        parse_custom_points("2-10")


def test_zero_investment_forced_to_zero_even_with_zero_threshold():
    spec = ResponseSpec(shape="threshold", full_effect_investment=5, threshold=0, jump=0.5)
    assert response_fraction([0.0], spec)[0] == 0.0
    assert response_fraction([0.01], spec)[0] > 0.5


# Timing -------------------------------------------------------------------------
def test_timing_realisation():
    from assumptions import TimingSpec
    t = TimingSpec(start_month=1, implementation_months=12, benefit_lag_months=2, ramp_months=6, partial_credit=0.5)
    r = timing_effect(t, delay=0, valuation_month=21)
    assert r.ready == 13 and r.benefit_start == 15
    assert r.ebitda_realisation == pytest.approx(1.0)
    assert r.multiple_credit == 1.0
    r = timing_effect(t, delay=4, valuation_month=21)     # ready 17, benefits 19
    assert r.ebitda_realisation == pytest.approx(2 / 6)
    r = timing_effect(t, delay=12, valuation_month=21)    # ready 25: incomplete
    assert r.ebitda_realisation == 0.0
    assert r.multiple_credit == pytest.approx(0.5 * 20 / 24)


def test_delay_reduces_value():
    a = illustrative_assumptions()
    v0 = company_values(a, "it", "base", [6.0], True)["proceeds"][0]
    a.companies["it"].scenarios["base"].delay_months = 6
    v1 = company_values(a, "it", "base", [6.0], True)["proceeds"][0]
    assert v1 < v0


# 15. Zero / negative outcomes ---------------------------------------------------
def test_equity_floored_at_zero():
    a = illustrative_assumptions()
    sc = a.companies["ecom"].scenarios["downside"]
    sc.baseline_ebitda = -5.0
    v = company_values(a, "ecom", "downside", [0.0, 2.0], False)
    assert np.all(v["equity"] == 0.0)
    assert np.all(v["proceeds"] == 0.0)
    sc.baseline_ebitda = 5.0
    sc.exit_net_debt = 500.0
    assert company_values(a, "ecom", "downside", [0.0], False)["proceeds"][0] == 0.0


# Liquidity --------------------------------------------------------------------
def test_liquidity_penalty_only_when_specified_and_unmet():
    a = illustrative_assumptions()
    base = company_values(a, "ecom", "base", [0.0, 2.0], True)
    assert np.all(base["haircut"] == 0)                   # requirement not specified
    a.liquidity.requirement = 3.0
    v = company_values(a, "ecom", "base", [2.0, 3.0], True)
    assert v["haircut"][0] == pytest.approx(0.25) and v["haircut"][1] == 0
    assert v["net_debt"][0] == pytest.approx(a.companies["ecom"].scenarios["base"].exit_net_debt + 10)
    a.companies["ecom"].timing.start_month = 7             # after the 6-month deadline
    v = company_values(a, "ecom", "base", [5.0], True)
    assert v["haircut"][0] == pytest.approx(0.25)
    r = evaluate_portfolio(a, {"ecom": 5.0, "it": 0, "sec": 0}, "ecom")
    assert any("LIQUIDITY WARNING" in w for w in r.warnings)


# 2-3, 13-14. Feasibility flags ------------------------------------------------
def test_feasibility_flags():
    a = illustrative_assumptions()
    r = evaluate_portfolio(a, {"ecom": 10.0, "it": 8.0, "sec": 0.0}, "ecom")
    assert not r.feasible and any("budget" in m for m in r.feasibility)
    r = evaluate_portfolio(a, {"ecom": 11.0, "it": 0.0, "sec": 0.0}, "ecom")
    assert any("request" in m for m in r.feasibility)
    a.focus.nonfocus_cap = 5.0
    r = evaluate_portfolio(a, {"ecom": 5.0, "it": 3.0, "sec": 7.0}, "ecom")
    assert any("not the management focus" in m for m in r.feasibility)
    r = evaluate_portfolio(a, {"ecom": 5.0, "it": 3.0, "sec": 7.0}, "sec")
    assert r.feasible


def test_reserve_preserved():
    a = illustrative_assumptions()
    r = evaluate_portfolio(a, {"ecom": 4.0, "it": 2.0, "sec": 3.5}, "ecom")
    assert r.invested + r.reserve == pytest.approx(CASE_BUDGET)
    assert r.reserve == pytest.approx(5.5)
    # reserve is not deducted from net cash returned
    assert r.summary()["net_cash_returned"] == pytest.approx(r.summary()["total_proceeds"] - 9.5)


def test_focus_changes_effectiveness():
    a = illustrative_assumptions()
    on = company_values(a, "it", "base", [6.0], True)["ebitda_uplift"][0]
    off = company_values(a, "it", "base", [6.0], False)["ebitda_uplift"][0]
    assert off == pytest.approx(on * a.focus.nonfocus_multiplier / a.focus.focus_multiplier)
