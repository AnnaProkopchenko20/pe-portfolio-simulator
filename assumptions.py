"""
Case facts and model assumptions.

Two clearly separated layers:

1. CASE FACTS  - hardcoded figures from "Case Challenge 3.0" (SKELAR x Horizon
   Capital x Mastercard). These are illustrative case-company numbers, not the
   performance of any real company. They are never modified by the model.

2. ASSUMPTIONS - everything else (exit EBITDA, multiples, response curves,
   timing, focus effects, scenario probabilities ...). None of these values is
   provided by the case. The ILLUSTRATIVE preset below exists only to
   demonstrate how the model works. Every value in it is labelled:
   "ILLUSTRATIVE ASSUMPTION - NOT PROVIDED BY THE CASE".

All monetary values are in $ millions.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

ILLUSTRATIVE_LABEL = "ILLUSTRATIVE ASSUMPTION — NOT PROVIDED BY THE CASE"

COMPANY_KEYS: tuple[str, ...] = ("ecom", "it", "sec")
SCENARIOS: tuple[str, ...] = ("downside", "base", "upside")
SCENARIO_LABELS = {"downside": "Downside", "base": "Base case", "upside": "Upside"}
RESPONSE_SHAPES: tuple[str, ...] = ("linear", "diminishing", "threshold", "s_curve", "custom")
SHAPE_LABELS = {
    "linear": "Linear",
    "diminishing": "Diminishing returns",
    "threshold": "Threshold",
    "s_curve": "S-curve",
    "custom": "Custom (piecewise)",
}


# ---------------------------------------------------------------------------
# 1. CASE FACTS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseCompany:
    key: str
    name: str
    short: str
    prev_revenue: float
    latest_revenue: float
    prev_ebitda: float
    latest_ebitda: float
    latest_margin: float          # as stated in the case (fraction)
    net_debt: float               # current net debt
    ownership: float              # fund ownership (fraction)
    request: float                # funding request = investment cap
    challenges: tuple[str, ...] = ()


CASE_COMPANIES: dict[str, CaseCompany] = {
    "ecom": CaseCompany(
        key="ecom", name="E-commerce retailer", short="E-commerce",
        prev_revenue=700.0, latest_revenue=600.0,
        prev_ebitda=35.0, latest_ebitda=12.0, latest_margin=0.02,
        net_debt=35.0, ownership=0.30, request=10.0,
        challenges=(
            "Recovering from a major missile strike and logistics disruption",
            "40% of storage capacity lost",
            "Delivery time 2 → 5 days",
            "Liquidity needed within 6 months",
        ),
    ),
    "it": CaseCompany(
        key="it", name="IT engineering company", short="IT Engineering",
        prev_revenue=200.0, latest_revenue=180.0,
        prev_ebitda=36.0, latest_ebitda=21.6, latest_margin=0.12,
        net_debt=0.0, ownership=0.30, request=8.0,
        challenges=(
            "Declining demand and pricing pressure as AI reshapes software development",
            "~3,000 specialists",
            "Billable utilization 85% → 70%",
            "Repositioning: 12–18 months",
        ),
    ),
    "sec": CaseCompany(
        key="sec", name="Security hardware company", short="Security Hardware",
        prev_revenue=250.0, latest_revenue=325.0,
        prev_ebitda=50.0, latest_ebitda=65.0, latest_margin=0.20,
        net_debt=10.0, ownership=0.20, request=15.0,
        challenges=(
            "Growing strongly",
            "95% international sales",
            "75% of manufacturing in Ukraine (buyer concern)",
            "Overseas expansion: 12–18 months",
        ),
    ),
}

CASE_BUDGET = 15.0            # available fund capital
CASE_TOTAL_REQUEST = 33.0     # sum of funding requests
CASE_EXIT_DEADLINE = 24       # months
CASE_ECOM_LIQUIDITY_DEADLINE = 6  # months (the case gives the deadline, NOT the amount)


# ---------------------------------------------------------------------------
# 2. ASSUMPTION STRUCTURES
# ---------------------------------------------------------------------------
@dataclass
class ResponseSpec:
    """How investment x ($M) translates into a 0..1 'programme progress' f(x).

    f(x) is then scaled by the maximum EBITDA uplift (run-rate, $M) and the
    maximum exit-multiple uplift (turns of EBITDA). f(0) = 0 always, so zero
    investment is exactly the no-investment counterfactual.
    """
    shape: str = "linear"
    full_effect_investment: float = 10.0   # X_full: investment at which f reaches 1
    curvature: float = 0.3                 # diminishing returns: k (per $M)
    threshold: float = 0.0                 # threshold: minimum viable investment T
    jump: float = 0.5                      # threshold: share of full effect unlocked at T
    below_threshold: float = 0.0           # threshold: share achieved just below T (linear leak)
    midpoint: float = 5.0                  # S-curve: investment at the steepest point
    steepness: float = 1.0                 # S-curve: steepness (per $M)
    custom_points: list[tuple[float, float]] = field(default_factory=list)  # (x $M, f in 0..1)
    max_ebitda_uplift: float = 0.0         # run-rate EBITDA uplift at f = 1 ($M)
    max_multiple_uplift: float = 0.0       # exit-multiple uplift at f = 1 (x EBITDA)


@dataclass
class TimingSpec:
    start_month: float = 0.0          # when capital is committed / implementation starts
    implementation_months: float = 6.0
    benefit_lag_months: float = 2.0   # from operational readiness to first EBITDA effect
    ramp_months: float = 6.0          # from first effect to full run-rate
    partial_credit: float = 0.0       # buyer credit for multiple uplift if not yet complete (0..1)


@dataclass
class ScenarioInputs:
    baseline_ebitda: float            # exit EBITDA with NO additional investment
    exit_multiple: float              # EV / EBITDA multiple without investment effects
    exit_net_debt: float              # net debt at exit (excluding investment spend)
    effectiveness: float = 1.0        # scales the investment response (1 = as designed)
    delay_months: float = 0.0         # extra implementation delay in this scenario


@dataclass
class CompanyAssumptions:
    response: ResponseSpec
    timing: TimingSpec
    scenarios: dict[str, ScenarioInputs]
    pre_money_equity: float = 100.0   # only used in the new-equity financing mode


@dataclass
class FocusSpec:
    focus_multiplier: float = 1.0       # effectiveness multiplier for the focus company
    nonfocus_multiplier: float = 0.6    # effectiveness multiplier for the other two
    nonfocus_cap: float = 7.5           # max capital a non-focus company can absorb ($M)
    focus_time_saving: float = 0.0      # months by which focus shortens implementation


@dataclass
class LiquiditySpec:
    """E-commerce liquidity requirement. The case gives the 6-month deadline only.
    requirement = 0 means 'not specified' -> no penalty is applied, a notice is shown."""
    requirement: float = 0.0
    deadline_month: float = CASE_ECOM_LIQUIDITY_DEADLINE
    ebitda_haircut: float = 0.25        # share of exit EBITDA lost if requirement unmet
    extra_net_debt: float = 10.0        # rescue financing / payables stretch if unmet ($M)


@dataclass
class ModelAssumptions:
    companies: dict[str, CompanyAssumptions]
    probabilities: dict[str, float]     # in percent, must sum to 100
    valuation_month: float = 21.0       # month whose run-rate EBITDA buyers price
    sale_launch_month: float = 15.0
    financing_mode: str = "fixed"       # "fixed" (non-dilutive) | "equity" (new equity)
    focus: FocusSpec = field(default_factory=FocusSpec)
    liquidity: LiquiditySpec = field(default_factory=LiquiditySpec)
    budget: float = CASE_BUDGET
    case: dict[str, CaseCompany] = field(default_factory=lambda: dict(CASE_COMPANIES))

    def copy(self) -> "ModelAssumptions":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# 3. ILLUSTRATIVE PRESET  (every number below: ILLUSTRATIVE ASSUMPTION)
# ---------------------------------------------------------------------------
# Design intent: show each response shape and each value source, without
# rigging a winner. E-commerce value comes mostly from EBITDA recovery
# (threshold-type warehouse rebuild), IT from utilization/EBITDA over a long
# repositioning (diminishing returns), Security Hardware mostly from risk
# reduction (multiple re-rating, S-curve plant build-out). With these
# illustrative numbers the three companies produce comparable value; small
# changes in assumptions change the optimizer's choice.
def illustrative_assumptions() -> ModelAssumptions:
    companies = {
        "ecom": CompanyAssumptions(
            response=ResponseSpec(
                shape="threshold", full_effect_investment=10.0, threshold=6.0,
                jump=0.65, below_threshold=0.0, curvature=0.3, midpoint=5.0,
                steepness=1.0,
                custom_points=[(2.0, 0.05), (4.0, 0.5), (6.0, 0.75), (9.0, 1.0)],
                max_ebitda_uplift=13.0, max_multiple_uplift=0.0,
            ),
            timing=TimingSpec(start_month=0, implementation_months=4,
                              benefit_lag_months=2, ramp_months=6, partial_credit=0.0),
            scenarios={
                "downside": ScenarioInputs(0.0, 4.0, 35.0, 0.7, 3),
                "base":     ScenarioInputs(12.0, 5.0, 35.0, 1.0, 0),
                "upside":   ScenarioInputs(18.0, 6.0, 35.0, 1.1, 0),
            },
            pre_money_equity=37.0,   # = 12 x 6.0 - 35
        ),
        "it": CompanyAssumptions(
            response=ResponseSpec(
                shape="diminishing", full_effect_investment=8.0, curvature=0.35,
                threshold=2.0, jump=0.3, below_threshold=0.0, midpoint=4.0,
                steepness=1.0,
                custom_points=[(2.0, 0.4), (4.0, 0.65), (8.0, 1.0)],
                max_ebitda_uplift=1.5, max_multiple_uplift=0.0,
            ),
            timing=TimingSpec(start_month=1, implementation_months=12,
                              benefit_lag_months=1, ramp_months=6, partial_credit=0.25),
            scenarios={
                "downside": ScenarioInputs(14.0, 6.0, 0.0, 0.6, 4),
                "base":     ScenarioInputs(19.0, 7.0, 0.0, 1.0, 0),
                "upside":   ScenarioInputs(23.0, 8.0, 0.0, 1.15, 0),
            },
            pre_money_equity=151.0,  # = 21.6 x 7.0 - 0
        ),
        "sec": CompanyAssumptions(
            response=ResponseSpec(
                shape="custom", full_effect_investment=15.0, midpoint=8.0,
                steepness=0.8, curvature=0.3, threshold=6.0, jump=0.4,
                below_threshold=0.0,
                custom_points=[(2.75, 0.0), (3.0, 0.35), (7.5, 0.35), (12.0, 0.65), (15.0, 1.0)],
                max_ebitda_uplift=0.0, max_multiple_uplift=1.2,
            ),
            timing=TimingSpec(start_month=1, implementation_months=15,
                              benefit_lag_months=2, ramp_months=6, partial_credit=0.5),
            scenarios={
                "downside": ScenarioInputs(60.0, 6.0, 10.0, 0.6, 4),
                "base":     ScenarioInputs(75.0, 7.5, 10.0, 1.0, 0),
                "upside":   ScenarioInputs(88.0, 9.0, 10.0, 1.15, 0),
            },
            pre_money_equity=477.5,  # = 65 x 7.5 - 10
        ),
    }
    return ModelAssumptions(
        companies=companies,
        probabilities={"downside": 25.0, "base": 50.0, "upside": 25.0},
        valuation_month=21.0,
        sale_launch_month=15.0,
        financing_mode="fixed",
        focus=FocusSpec(focus_multiplier=1.0, nonfocus_multiplier=0.6,
                        nonfocus_cap=7.5, focus_time_saving=0.0),
        liquidity=LiquiditySpec(requirement=0.0, deadline_month=6.0,
                                ebitda_haircut=0.25, extra_net_debt=10.0),
    )


def parse_custom_points(text: str) -> list[tuple[float, float]]:
    """Parse 'x:pct, x:pct' (pct in % of full effect) into [(x, fraction)]."""
    pts: list[tuple[float, float]] = []
    if not text or not text.strip():
        return pts
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Custom point '{chunk}' must look like 'investment:percent', e.g. '6:60'.")
        a, b = chunk.split(":", 1)
        try:
            x, pct = float(a), float(b)
        except ValueError as exc:
            raise ValueError(f"Custom point '{chunk}' is not numeric.") from exc
        if x < 0 or pct < 0:
            raise ValueError(f"Custom point '{chunk}': values must be non-negative.")
        pts.append((x, pct / 100.0))
    return pts


def format_custom_points(points: list[tuple[float, float]]) -> str:
    return ", ".join(f"{x:g}:{f * 100:g}" for x, f in points)
