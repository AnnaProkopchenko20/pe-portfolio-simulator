"""UI-logic tests (no browser): slider capping, value round-trip, case-facts mode."""
import pytest

import app
from assumptions import illustrative_assumptions


def vals():
    return app.assumptions_to_values(illustrative_assumptions())


def test_values_round_trip():
    a = illustrative_assumptions()
    b = app.values_to_assumptions(vals())
    assert b.companies["ecom"].response == a.companies["ecom"].response
    assert b.companies["sec"].scenarios == a.companies["sec"].scenarios
    assert b.probabilities == a.probabilities and b.focus == a.focus and b.liquidity == a.liquidity


def test_slider_capped_at_remaining_budget_without_touching_others():
    h = app.make_slider_handler(0)                      # E-commerce slider moved
    out = h(app.MODE_SIM, 10.0, 3.0, 7.0, "ecom", "base", None, *vals())
    upd_x1, upd_x2, upd_x3 = out[0], out[1], out[2]
    assert upd_x1["value"] == pytest.approx(5.0)        # 15 - 3 - 7
    assert "value" not in upd_x2 and "value" not in upd_x3


def test_slider_not_capped_when_within_budget():
    h = app.make_slider_handler(2)
    out = h(app.MODE_SIM, 5.0, 3.0, 7.0, "ecom", "base", None, *vals())
    assert all("value" not in u for u in out[:3])


def test_case_facts_mode_shows_no_returns():
    out = app.compute_main(app.MODE_FACTS, 5, 3, 7, "ecom", "base", None, vals())
    assert "CASE FACTS ONLY" in out[0]
    assert "Expected fund proceeds" not in out[1]
    opt = app.run_optimizer(app.MODE_FACTS, 5, 3, 7, "ecom", "expected_net", True, 0.25, 1.0, *vals())
    assert "No optimum from case facts alone" in opt[0] and opt[4] is None


def test_invalid_input_reported_not_crashing():
    v = vals()
    v[len(app.CFIELDS) * 3 + 1] = 90                    # P(downside) -> sum != 100
    out = app.compute_main(app.MODE_SIM, 5, 3, 7, "ecom", "base", None, v)
    assert "sum to 100%" in out[0]


def test_success_criterion_flow():
    """5/3/7 -> move $2M from Security to E-commerce -> results change and are explained -> optimise."""
    v = vals()
    o1 = app.compute_main(app.MODE_SIM, 5, 3, 7, "ecom", "base", None, v)
    o2 = app.compute_main(app.MODE_SIM, 7, 3, 5, "ecom", "base", o1[12], v)
    assert o1[12]["net"] != o2[12]["net"]
    assert "What changed since your previous allocation" in o2[5]
    opt = app.run_optimizer(app.MODE_SIM, 7, 3, 5, "ecom", "expected_net", True, 0.25, 1.0, *v)
    assert "OPTIMAL INVESTMENT ALLOCATION" in opt[0] and opt[4] is not None
    # change an assumption -> optimum moves
    i_sec_maxM = len(app.CFIELDS) * 2 + app.CFIELDS.index("maxM")
    v2 = list(v)
    v2[i_sec_maxM] = 4.0
    opt2 = app.run_optimizer(app.MODE_SIM, 7, 3, 5, "ecom", "expected_net", True, 0.25, 1.0, *v2)
    assert opt2[4] != opt[4]
