"""
Financial model: investment -> operational progress -> EBITDA / multiple / net
debt -> enterprise value -> equity value -> cash proceeds to the fund.

All monetary values in $ millions. Pure functions, no UI code.

Valuation chain per company, scenario s, investment x:

    f(x)            = response progress in [0, 1]          (shape chosen by user)
    m               = effectiveness_s x focus multiplier
    rE(T)           = EBITDA realisation at valuation month T (timing model)
    rM(T)           = buyer credit for risk reduction at T   (timing model)

    Exit EBITDA     = Baseline EBITDA_s + f(x) * maxEBITDAuplift * m * rE(T)
    Exit multiple   = Base multiple_s   + f(x) * maxMultipleUplift * m * rM(T)
    EV              = Exit EBITDA x Exit multiple
    Equity          = max(EV - Exit net debt, 0)
    Fund proceeds   = Equity x Fund ownership (after dilution in equity mode)

Net benefit = (Proceeds(x) - Proceeds(0)) - x
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from assumptions import (
    COMPANY_KEYS,
    SCENARIOS,
    CompanyAssumptions,
    ModelAssumptions,
    ResponseSpec,
    TimingSpec,
)

EPS = 1e-9


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class AssumptionError(ValueError):
    """Raised when user-entered assumptions are invalid."""


def validate_probabilities(probs: dict[str, float]) -> dict[str, float]:
    """Return probabilities as fractions. Probabilities are entered in percent
    and must sum to exactly 100 (tolerance 0.01). No silent normalisation."""
    for s in SCENARIOS:
        p = probs.get(s)
        if p is None or not np.isfinite(p):
            raise AssumptionError(f"Probability for '{s}' is missing.")
        if p < 0:
            raise AssumptionError("Scenario probabilities cannot be negative.")
    total = sum(probs[s] for s in SCENARIOS)
    if abs(total - 100.0) > 0.01:
        raise AssumptionError(
            f"Scenario probabilities must sum to 100%. They currently sum to {total:.2f}%."
        )
    return {s: probs[s] / 100.0 for s in SCENARIOS}


def validate_response(spec: ResponseSpec, name: str = "") -> None:
    tag = f" ({name})" if name else ""
    if spec.shape not in ("linear", "diminishing", "threshold", "s_curve", "custom"):
        raise AssumptionError(f"Unknown response shape '{spec.shape}'{tag}.")
    if spec.shape != "custom" and spec.full_effect_investment <= 0:
        raise AssumptionError(f"Full-effect investment must be > 0{tag}.")
    if spec.shape == "diminishing" and spec.curvature < 0:
        raise AssumptionError(f"Curvature must be ≥ 0{tag}.")
    if spec.shape == "threshold":
        if not (0 <= spec.threshold <= spec.full_effect_investment):
            raise AssumptionError(f"Threshold must lie between 0 and the full-effect investment{tag}.")
        if not (0 <= spec.jump <= 1) or not (0 <= spec.below_threshold <= 1):
            raise AssumptionError(f"Threshold jump and below-threshold share must be 0–100%{tag}.")
        if spec.below_threshold > spec.jump + EPS:
            raise AssumptionError(f"Below-threshold share cannot exceed the jump at the threshold{tag}.")
    if spec.shape == "s_curve" and spec.steepness <= 0:
        raise AssumptionError(f"S-curve steepness must be > 0{tag}.")
    if spec.shape == "custom" and len([p for p in spec.custom_points if p[0] > 0]) == 0:
        raise AssumptionError(f"Custom response needs at least one point with investment > 0{tag}.")
    if spec.max_ebitda_uplift < 0 or spec.max_multiple_uplift < 0:
        raise AssumptionError(
            f"Maximum uplifts must be ≥ 0{tag}. Model value destruction through scenarios instead."
        )


def validate_timing(t: TimingSpec, name: str = "") -> None:
    tag = f" ({name})" if name else ""
    for label, v in (("Start month", t.start_month), ("Implementation", t.implementation_months),
                     ("Benefit lag", t.benefit_lag_months), ("Ramp-up", t.ramp_months)):
        if v < 0 or not np.isfinite(v):
            raise AssumptionError(f"{label} must be ≥ 0{tag}.")
    if t.start_month > 24:
        raise AssumptionError(f"Start month must be within the 24-month horizon{tag}.")
    if not (0 <= t.partial_credit <= 1):
        raise AssumptionError(f"Partial credit must be 0–100%{tag}.")


def validate_assumptions(a: ModelAssumptions) -> dict[str, float]:
    """Validate everything; return probabilities as fractions."""
    probs = validate_probabilities(a.probabilities)
    if a.financing_mode not in ("fixed", "equity"):
        raise AssumptionError("Financing mode must be 'fixed' or 'equity'.")
    if not (0 < a.valuation_month <= 24):
        raise AssumptionError("Valuation month must be within (0, 24].")
    f = a.focus
    if f.focus_multiplier < 0 or f.nonfocus_multiplier < 0:
        raise AssumptionError("Focus multipliers must be ≥ 0.")
    if f.nonfocus_cap < 0:
        raise AssumptionError("Non-focus capital cap must be ≥ 0.")
    if f.focus_time_saving < 0:
        raise AssumptionError("Focus time saving must be ≥ 0.")
    liq = a.liquidity
    if liq.requirement < 0 or liq.extra_net_debt < 0 or not (0 <= liq.ebitda_haircut <= 1):
        raise AssumptionError("Liquidity inputs: requirement and extra debt ≥ 0, haircut 0–100%.")
    for k in COMPANY_KEYS:
        ca = a.companies[k]
        name = a.case[k].short
        validate_response(ca.response, name)
        validate_timing(ca.timing, name)
        for s in SCENARIOS:
            sc = ca.scenarios[s]
            if sc.exit_multiple < 0:
                raise AssumptionError(f"Exit multiple must be ≥ 0 ({name}, {s}).")
            if sc.effectiveness < 0:
                raise AssumptionError(f"Effectiveness must be ≥ 0 ({name}, {s}).")
            if sc.delay_months < 0:
                raise AssumptionError(f"Delay must be ≥ 0 ({name}, {s}).")
        if a.financing_mode == "equity" and ca.pre_money_equity <= 0:
            raise AssumptionError(f"Pre-money equity value must be > 0 in equity mode ({name}).")
    return probs


# ---------------------------------------------------------------------------
# Investment-response functions  f: [0, inf) -> [0, 1], f(0) = 0
# ---------------------------------------------------------------------------
def response_fraction(x, spec: ResponseSpec) -> np.ndarray:
    """Vectorised programme progress f(x) in [0, 1] for investment x ($M)."""
    x = np.asarray(x, dtype=float)
    X = float(spec.full_effect_investment)
    shape = spec.shape

    if shape == "linear":
        f = x / X

    elif shape == "diminishing":
        # f = (1 - e^{-k x}) / (1 - e^{-k X}): concave, marginal return falls as x rises.
        k = float(spec.curvature)
        xc = np.minimum(x, X)
        if k < 1e-6:
            f = xc / X                                  # limit k -> 0 is linear
        else:
            f = (1.0 - np.exp(-k * xc)) / (1.0 - np.exp(-k * X))

    elif shape == "threshold":
        # Below T: small linear 'leak' up to below_threshold.
        # At T: jump to `jump`. Between T and X: linear to 1. Above X: saturated.
        T, j, b = float(spec.threshold), float(spec.jump), float(spec.below_threshold)
        below = b * x / T if T > 0 else np.zeros_like(x)
        span = X - T
        if span <= EPS:
            above = np.ones_like(x)
        else:
            above = j + (1.0 - j) * np.clip((x - T) / span, 0.0, 1.0)
        f = np.where(x < T, below, above)

    elif shape == "s_curve":
        # Logistic L(x) = 1/(1+e^{-s(x-m)}), rescaled so f(0)=0 and f(X)=1.
        s, m = float(spec.steepness), float(spec.midpoint)

        def L(v):
            return 1.0 / (1.0 + np.exp(-s * (v - m)))

        L0, LX = L(0.0), L(X)
        xc = np.minimum(x, X)
        f = (L(xc) - L0) / (LX - L0) if LX - L0 > EPS else xc / X

    elif shape == "custom":
        # Piecewise-linear through (0,0) and the user's points; flat beyond the last point.
        pts = sorted((float(px), float(pf)) for px, pf in spec.custom_points if px > 0)
        xs = np.array([0.0] + [p[0] for p in pts])
        fs = np.array([0.0] + [p[1] for p in pts])
        f = np.interp(x, xs, fs)

    else:
        raise AssumptionError(f"Unknown response shape '{shape}'.")

    f = np.clip(f, 0.0, 1.0 if shape != "custom" else np.inf)
    # Zero investment is, by construction, the no-investment counterfactual.
    return np.where(x <= 0, 0.0, f)


# ---------------------------------------------------------------------------
# Timing model
# ---------------------------------------------------------------------------
@dataclass
class TimingResult:
    start: float
    ready: float          # operational readiness month
    benefit_start: float  # first EBITDA effect
    full_run_rate: float  # full run-rate reached
    ebitda_realisation: float   # share of run-rate uplift buyers see at valuation month
    multiple_credit: float      # share of multiple uplift buyers credit at valuation month


def timing_effect(t: TimingSpec, delay: float, valuation_month: float,
                  time_saving: float = 0.0) -> TimingResult:
    impl = max(t.implementation_months + delay - time_saving, 0.0)
    ready = t.start_month + impl
    benefit_start = ready + t.benefit_lag_months
    full = benefit_start + t.ramp_months
    T = valuation_month
    if t.ramp_months > EPS:
        r_e = float(np.clip((T - benefit_start) / t.ramp_months, 0.0, 1.0))
    else:
        r_e = 1.0 if T >= benefit_start else 0.0
    if T >= ready:
        r_m = 1.0
    elif impl > EPS:
        r_m = t.partial_credit * float(np.clip((T - t.start_month) / impl, 0.0, 1.0))
    else:
        r_m = 0.0
    return TimingResult(t.start_month, ready, benefit_start, full, r_e, r_m)


# ---------------------------------------------------------------------------
# Company valuation
# ---------------------------------------------------------------------------
def focus_multiplier(a: ModelAssumptions, is_focus: bool) -> float:
    return a.focus.focus_multiplier if is_focus else a.focus.nonfocus_multiplier


def liquidity_met(a: ModelAssumptions, x_ecom) -> np.ndarray:
    """True where the e-commerce liquidity requirement is covered before the deadline.
    requirement == 0 means 'not specified' -> treated as met (no penalty)."""
    x_ecom = np.asarray(x_ecom, dtype=float)
    liq = a.liquidity
    if liq.requirement <= 0:
        return np.ones_like(x_ecom, dtype=bool)
    in_time = a.companies["ecom"].timing.start_month <= liq.deadline_month
    return (x_ecom + EPS >= liq.requirement) & in_time


def fund_ownership(a: ModelAssumptions, key: str, x) -> np.ndarray:
    """Fund ownership at exit. Fixed mode: unchanged. Equity mode: new money
    buys shares at the pre-money value: (o*V + x) / (V + x)."""
    x = np.asarray(x, dtype=float)
    o = a.case[key].ownership
    if a.financing_mode == "fixed":
        return np.full_like(x, o)
    V = a.companies[key].pre_money_equity
    return (o * V + x) / (V + x)


def company_values(a: ModelAssumptions, key: str, scenario: str, x, is_focus: bool) -> dict:
    """Vectorised valuation of one company for investment array x."""
    x = np.asarray(x, dtype=float)
    ca: CompanyAssumptions = a.companies[key]
    sc = ca.scenarios[scenario]
    m = sc.effectiveness * focus_multiplier(a, is_focus)
    tm = timing_effect(ca.timing, sc.delay_months, a.valuation_month,
                       a.focus.focus_time_saving if is_focus else 0.0)
    f = response_fraction(x, ca.response)

    ebitda_uplift = f * ca.response.max_ebitda_uplift * m * tm.ebitda_realisation
    multiple_uplift = f * ca.response.max_multiple_uplift * m * tm.multiple_credit
    ebitda = sc.baseline_ebitda + ebitda_uplift
    net_debt = np.full_like(x, sc.exit_net_debt)

    haircut = np.zeros_like(x)
    if key == "ecom":
        unmet = ~liquidity_met(a, x)
        haircut = np.where(unmet, a.liquidity.ebitda_haircut, 0.0)
        net_debt = net_debt + np.where(unmet, a.liquidity.extra_net_debt, 0.0)
    ebitda = ebitda * (1.0 - haircut)

    multiple = sc.exit_multiple + multiple_uplift
    ev = ebitda * multiple
    equity = np.maximum(ev - net_debt, 0.0)        # common equity floored at zero
    own = fund_ownership(a, key, x)
    proceeds = equity * own
    return {
        "x": x, "f": f, "m": m, "timing": tm,
        "ebitda_uplift": ebitda_uplift, "multiple_uplift": multiple_uplift,
        "haircut": haircut, "ebitda": ebitda, "multiple": multiple,
        "ev": ev, "net_debt": net_debt, "equity": equity,
        "ownership": own, "proceeds": proceeds,
    }


# ---------------------------------------------------------------------------
# Portfolio evaluation
# ---------------------------------------------------------------------------
@dataclass
class PortfolioResult:
    allocation: dict[str, float]
    focus: str
    probs: dict[str, float]
    by_scenario: dict[str, dict[str, dict]]          # scenario -> company -> scalar values
    baseline: dict[str, dict[str, dict]]             # same, with x = 0
    invested: float
    reserve: float
    feasibility: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.feasibility

    # --- aggregates -------------------------------------------------------
    def proceeds(self, scenario: str) -> float:
        return sum(self.by_scenario[scenario][k]["proceeds"] for k in COMPANY_KEYS)

    def baseline_proceeds(self, scenario: str) -> float:
        return sum(self.baseline[scenario][k]["proceeds"] for k in COMPANY_KEYS)

    def expected(self, fn) -> float:
        return sum(self.probs[s] * fn(s) for s in SCENARIOS)

    def company_expected(self, key: str, field_: str, baseline: bool = False) -> float:
        src = self.baseline if baseline else self.by_scenario
        return sum(self.probs[s] * src[s][key][field_] for s in SCENARIOS)

    def summary(self, scenario: str | None = None) -> dict[str, float | None]:
        if scenario is None:
            P = self.expected(self.proceeds)
            B = self.expected(self.baseline_proceeds)
        else:
            P, B = self.proceeds(scenario), self.baseline_proceeds(scenario)
        inc = P - B
        net_gain = inc - self.invested
        return {
            "total_proceeds": P,
            "baseline_proceeds": B,
            "incremental_proceeds": inc,
            "invested": self.invested,
            "net_cash_returned": P - self.invested,
            "net_gain": net_gain,
            "roi": net_gain / self.invested if self.invested >= 0.25 - EPS else None,
        }


def _scalarise(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = float(v.reshape(-1)[0])
        else:
            out[k] = v
    return out


def check_feasibility(a: ModelAssumptions, allocation: dict[str, float], focus: str) -> list[str]:
    issues = []
    total = sum(allocation.values())
    if total > a.budget + EPS:
        issues.append(f"Total allocation ${total:.2f}M exceeds the ${a.budget:.0f}M budget.")
    for k in COMPANY_KEYS:
        x = allocation[k]
        if x < -EPS:
            issues.append(f"{a.case[k].short}: allocation cannot be negative.")
        if x > a.case[k].request + EPS:
            issues.append(f"{a.case[k].short}: ${x:.2f}M exceeds its ${a.case[k].request:.0f}M request.")
        if k != focus and x > a.focus.nonfocus_cap + EPS:
            issues.append(
                f"{a.case[k].short} is not the management focus: ${x:.2f}M exceeds the "
                f"${a.focus.nonfocus_cap:.2f}M cap for limited/defensive support."
            )
    return issues


def evaluate_portfolio(a: ModelAssumptions, allocation: dict[str, float], focus: str) -> PortfolioResult:
    if focus not in COMPANY_KEYS:
        raise AssumptionError("Select exactly one primary management focus.")
    probs = validate_assumptions(a)
    by_s, base_s = {}, {}
    for s in SCENARIOS:
        by_s[s], base_s[s] = {}, {}
        for k in COMPANY_KEYS:
            is_f = k == focus
            by_s[s][k] = _scalarise(company_values(a, k, s, [allocation[k]], is_f))
            base_s[s][k] = _scalarise(company_values(a, k, s, [0.0], is_f))
    invested = float(sum(allocation.values()))
    res = PortfolioResult(
        allocation=dict(allocation), focus=focus, probs=probs,
        by_scenario=by_s, baseline=base_s, invested=invested,
        reserve=max(a.budget - invested, 0.0),
    )
    res.feasibility = check_feasibility(a, allocation, focus)
    res.warnings = model_warnings(a, allocation, focus)
    return res


def model_warnings(a: ModelAssumptions, allocation: dict[str, float], focus: str) -> list[str]:
    w = []
    liq = a.liquidity
    ecom_t = a.companies["ecom"].timing
    if liq.requirement <= 0:
        w.append("E-commerce liquidity requirement is not specified. The case states liquidity is "
                 "needed within 6 months but gives no amount. Enter one in Financial Assumptions "
                 "to activate the liquidity test.")
    else:
        if ecom_t.start_month > liq.deadline_month:
            w.append(f"LIQUIDITY WARNING: e-commerce funding starts in month {ecom_t.start_month:g}, "
                     f"after the month-{liq.deadline_month:g} liquidity deadline.")
        if allocation["ecom"] + EPS < liq.requirement:
            w.append(f"LIQUIDITY WARNING: e-commerce receives ${allocation['ecom']:.2f}M, below the "
                     f"assumed ${liq.requirement:.2f}M liquidity requirement. Distress penalty applied "
                     f"(EBITDA −{liq.ebitda_haircut:.0%}, net debt +${liq.extra_net_debt:.1f}M).")
    for k in COMPANY_KEYS:
        tm = timing_effect(a.companies[k].timing, a.companies[k].scenarios["base"].delay_months,
                           a.valuation_month, a.focus.focus_time_saving if k == focus else 0.0)
        if allocation[k] > 0 and tm.ebitda_realisation < 1 and a.companies[k].response.max_ebitda_uplift > 0:
            w.append(f"{a.case[k].short}: in the base case only {tm.ebitda_realisation:.0%} of the run-rate "
                     f"EBITDA uplift is visible by valuation month {a.valuation_month:g} "
                     f"(full run-rate in month {tm.full_run_rate:g}).")
        if tm.ready > 24 and allocation[k] > 0:
            w.append(f"{a.case[k].short}: implementation completes after month 24 in the base case.")
    return w


# ---------------------------------------------------------------------------
# Value bridge (why did proceeds change?)
# ---------------------------------------------------------------------------
def value_bridge(base: dict, new: dict) -> dict[str, float]:
    """Sequential attribution of the change in fund proceeds for one company:
    EBITDA -> multiple -> net debt -> ownership. Each step uses the floored
    equity formula, so the steps add up exactly to the total change.
    (Sequential attribution is order-dependent; the order is stated in the UI.)"""
    def P(E, M, N, o):
        return max(E * M - N, 0.0) * o

    E0, M0, N0, o0 = base["ebitda"], base["multiple"], base["net_debt"], base["ownership"]
    E1, M1, N1, o1 = new["ebitda"], new["multiple"], new["net_debt"], new["ownership"]
    p0 = P(E0, M0, N0, o0)
    pa = P(E1, M0, N0, o0)
    pb = P(E1, M1, N0, o0)
    pc = P(E1, M1, N1, o0)
    pd = P(E1, M1, N1, o1)
    return {"ebitda": pa - p0, "multiple": pb - pa, "net_debt": pc - pb,
            "ownership": pd - pc, "total": pd - p0}


def expected_bridge(res: PortfolioResult, key: str) -> dict[str, float]:
    out = {"ebitda": 0.0, "multiple": 0.0, "net_debt": 0.0, "ownership": 0.0, "total": 0.0}
    for s in SCENARIOS:
        b = value_bridge(res.baseline[s][key], res.by_scenario[s][key])
        for k2 in out:
            out[k2] += res.probs[s] * b[k2]
    return out


# ---------------------------------------------------------------------------
# Calculation breakdown (text)
# ---------------------------------------------------------------------------
def fmt(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}"


def calculation_breakdown(a: ModelAssumptions, res: PortfolioResult, scenario: str) -> str:
    """Markdown step-by-step calculation for every company in one scenario."""
    lines = []
    for k in COMPANY_KEYS:
        c = a.case[k]
        r = res.by_scenario[scenario][k]
        b = res.baseline[scenario][k]
        ca = a.companies[k]
        sc = ca.scenarios[scenario]
        tm: TimingResult = r["timing"]
        focus_tag = "PRIMARY FOCUS" if k == res.focus else "non-focus (limited support)"
        x = res.allocation[k]
        lines.append(f"### {c.short} — ${x:.2f}M invested · {focus_tag}")
        lines.append("| Step | Calculation | With investment | No investment |")
        lines.append("|---|---|---:|---:|")
        lines.append(f"| Programme progress f(x) | {ca.response.shape} curve at x = {x:.2f} | {r['f']:.1%} | 0.0% |")
        lines.append(f"| Effectiveness × focus | {sc.effectiveness:.2f} × {focus_multiplier(a, k == res.focus):.2f} | {r['m']:.2f} | – |")
        lines.append(f"| Timing at month {a.valuation_month:g} | EBITDA realised {tm.ebitda_realisation:.0%}; risk-reduction credit {tm.multiple_credit:.0%} | | |")
        lines.append(f"| Baseline exit EBITDA | assumption | {fmt(sc.baseline_ebitda)} | {fmt(sc.baseline_ebitda)} |")
        lines.append(f"| + EBITDA uplift | {r['f']:.3f} × {fmt(ca.response.max_ebitda_uplift)} × {r['m']:.2f} × {tm.ebitda_realisation:.2f} | {fmt(r['ebitda_uplift'])} | 0.00 |")
        if k == "ecom" and (r["haircut"] > 0 or b["haircut"] > 0):
            lines.append(f"| − Liquidity distress haircut | unmet liquidity requirement | {r['haircut']:.0%} | {b['haircut']:.0%} |")
        lines.append(f"| **= Exit EBITDA** | | **{fmt(r['ebitda'])}** | **{fmt(b['ebitda'])}** |")
        lines.append(f"| Exit multiple | {fmt(sc.exit_multiple)} + {r['f']:.3f} × {fmt(ca.response.max_multiple_uplift)} × {r['m']:.2f} × {tm.multiple_credit:.2f} | {fmt(r['multiple'])}x | {fmt(b['multiple'])}x |")
        lines.append(f"| **= Enterprise value** | Exit EBITDA × multiple | **{fmt(r['ev'])}** | **{fmt(b['ev'])}** |")
        lines.append(f"| − Exit net debt | assumption{' + liquidity penalty' if k == 'ecom' and r['net_debt'] != sc.exit_net_debt else ''} | {fmt(r['net_debt'])} | {fmt(b['net_debt'])} |")
        lines.append(f"| **= Equity value** | max(EV − net debt, 0) | **{fmt(r['equity'])}** | **{fmt(b['equity'])}** |")
        lines.append(f"| × Fund ownership | {'unchanged (non-dilutive)' if a.financing_mode == 'fixed' else '(o·V + x)/(V + x)'} | {r['ownership']:.2%} | {b['ownership']:.2%} |")
        lines.append(f"| **= Fund exit proceeds** | | **{fmt(r['proceeds'])}** | **{fmt(b['proceeds'])}** |")
        inc = r["proceeds"] - b["proceeds"]
        lines.append(f"| Additional proceeds from intervention | {fmt(r['proceeds'])} − {fmt(b['proceeds'])} | {fmt(inc)} | |")
        lines.append(f"| − New fund capital | | {fmt(x)} | |")
        lines.append(f"| **= Incremental net benefit** | | **{fmt(inc - x)}** | |")
        lines.append("")
    return "\n".join(lines)
