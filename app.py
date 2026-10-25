"""
Private Equity Portfolio Optimization Simulator
Case Challenge 3.0 (SKELAR x Horizon Capital x Mastercard)
"""
from __future__ import annotations

import html
import os
import traceback
from pathlib import Path

# Disable Gradio's experimental Node.js SSR server (it crashes on Hugging Face Spaces).
os.environ["GRADIO_SSR_MODE"] = "false"

import gradio as gr
import numpy as np
import pandas as pd

# --- Gradio slider fix ---------------------------------------------------------
_orig_slider_preprocess = gr.Slider.preprocess


def _slider_preprocess(self, payload):
    if isinstance(payload, str):
        try:
            payload = float(payload)
        except ValueError:
            payload = None
    return _orig_slider_preprocess(self, payload)


gr.Slider.preprocess = _slider_preprocess
# -------------------------------------------------------------------------------

import charts
from assumptions import (
    CASE_BUDGET,
    CASE_COMPANIES,
    CASE_TOTAL_REQUEST,
    COMPANY_KEYS,
    ILLUSTRATIVE_LABEL,
    SCENARIO_LABELS,
    SCENARIOS,
    SHAPE_LABELS,
    CompanyAssumptions,
    FocusSpec,
    LiquiditySpec,
    ModelAssumptions,
    ResponseSpec,
    ScenarioInputs,
    TimingSpec,
    format_custom_points,
    illustrative_assumptions,
    parse_custom_points,
)
from financial_model import (
    EPS,
    AssumptionError,
    PortfolioResult,
    calculation_breakdown,
    evaluate_portfolio,
    expected_bridge,
    validate_assumptions,
)
from optimizer import (
    OBJECTIVES,
    SENS_PARAMS,
    breakeven_response_scale,
    marginal_values,
    one_way_sensitivity,
    optimize,
    standalone_best,
    sweep_values,
    two_way_sensitivity,
)

NAMES = charts.NAMES
MODE_FACTS = "Case facts only"
MODE_SIM = "Illustrative simulation"
FOCUS_CHOICES = [(NAMES[k], k) for k in COMPANY_KEYS]
SHAPE_CHOICES = [(v, k) for k, v in SHAPE_LABELS.items()]
SCEN_COLUMNS = ["Company", "Scenario", "Baseline exit EBITDA ($M)", "Exit multiple (x)",
                "Exit net debt ($M)", "Effectiveness (%)", "Extra delay (months)"]
CFIELDS = ["shape", "X", "k", "T", "jump", "below", "mid", "steep", "custom",
           "maxE", "maxM", "start", "impl", "lag", "ramp", "partial", "premoney"]
GFIELDS = ["p_down", "p_base", "p_up", "val_month", "sale_month", "financing",
           "focus_mult", "nonfocus_mult", "nonfocus_cap", "time_saving",
           "liq_req", "liq_deadline", "haircut", "extra_debt"]
FIN_CHOICES = [("Fixed ownership / non-dilutive support — illustrative assumption", "fixed"),
               ("New equity at an adjustable pre-money valuation (dilution)", "equity")]


def scenario_frame(a: ModelAssumptions) -> pd.DataFrame:
    rows = []
    for k in COMPANY_KEYS:
        for s in SCENARIOS:
            sc = a.companies[k].scenarios[s]
            rows.append([NAMES[k], SCENARIO_LABELS[s], sc.baseline_ebitda, sc.exit_multiple,
                         sc.exit_net_debt, round(sc.effectiveness * 100, 4), sc.delay_months])
    return pd.DataFrame(rows, columns=SCEN_COLUMNS)


def assumptions_to_values(a: ModelAssumptions) -> list:
    vals = []
    for k in COMPANY_KEYS:
        c = a.companies[k]
        r, t = c.response, c.timing
        vals += [r.shape, r.full_effect_investment, r.curvature, r.threshold, r.jump * 100,
                 r.below_threshold * 100, r.midpoint, r.steepness, format_custom_points(r.custom_points),
                 r.max_ebitda_uplift, r.max_multiple_uplift, t.start_month, t.implementation_months,
                 t.benefit_lag_months, t.ramp_months, t.partial_credit * 100, c.pre_money_equity]
    vals.append(scenario_frame(a))
    p = a.probabilities
    vals += [p["downside"], p["base"], p["upside"], a.valuation_month, a.sale_launch_month,
             a.financing_mode, a.focus.focus_multiplier, a.focus.nonfocus_multiplier,
             a.focus.nonfocus_cap, a.focus.focus_time_saving, a.liquidity.requirement,
             a.liquidity.deadline_month, a.liquidity.ebitda_haircut * 100, a.liquidity.extra_net_debt]
    return vals


def _num(v, label: str) -> float:
    if v is None or (isinstance(v, str) and not v.strip()):
        raise AssumptionError(f"'{label}' is empty.")
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise AssumptionError(f"'{label}' must be a number.") from exc
    if not np.isfinite(f):
        raise AssumptionError(f"'{label}' must be finite.")
    return f


def values_to_assumptions(vals: list) -> ModelAssumptions:
    n = len(CFIELDS)
    companies = {}
    scen_df = vals[3 * n]
    if not isinstance(scen_df, pd.DataFrame):
        scen_df = pd.DataFrame(scen_df["data"] if isinstance(scen_df, dict) else scen_df, columns=SCEN_COLUMNS)
    if len(scen_df) != 9:
        raise AssumptionError("The scenario table must keep its 9 rows (3 companies × 3 scenarios).")
    for ci, k in enumerate(COMPANY_KEYS):
        v = dict(zip(CFIELDS, vals[ci * n:(ci + 1) * n]))
        nm = NAMES[k]
        resp = ResponseSpec(
            shape=str(v["shape"]),
            full_effect_investment=_num(v["X"], f"{nm}: full-effect investment"),
            curvature=_num(v["k"], f"{nm}: curvature"),
            threshold=_num(v["T"], f"{nm}: threshold"),
            jump=_num(v["jump"], f"{nm}: jump") / 100,
            below_threshold=_num(v["below"], f"{nm}: below-threshold share") / 100,
            midpoint=_num(v["mid"], f"{nm}: S-curve midpoint"),
            steepness=_num(v["steep"], f"{nm}: S-curve steepness"),
            custom_points=parse_custom_points(v["custom"] or ""),
            max_ebitda_uplift=_num(v["maxE"], f"{nm}: max EBITDA uplift"),
            max_multiple_uplift=_num(v["maxM"], f"{nm}: max multiple uplift"),
        )
        tim = TimingSpec(
            start_month=_num(v["start"], f"{nm}: start month"),
            implementation_months=_num(v["impl"], f"{nm}: implementation months"),
            benefit_lag_months=_num(v["lag"], f"{nm}: benefit lag"),
            ramp_months=_num(v["ramp"], f"{nm}: ramp-up months"),
            partial_credit=_num(v["partial"], f"{nm}: partial credit") / 100,
        )
        scen = {}
        for si, s in enumerate(SCENARIOS):
            row = scen_df.iloc[ci * 3 + si]
            lab = f"{nm} / {SCENARIO_LABELS[s]}"
            scen[s] = ScenarioInputs(
                baseline_ebitda=_num(row.iloc[2], f"{lab}: baseline EBITDA"),
                exit_multiple=_num(row.iloc[3], f"{lab}: exit multiple"),
                exit_net_debt=_num(row.iloc[4], f"{lab}: exit net debt"),
                effectiveness=_num(row.iloc[5], f"{lab}: effectiveness") / 100,
                delay_months=_num(row.iloc[6], f"{lab}: delay"),
            )
        companies[k] = CompanyAssumptions(resp, tim, scen, _num(v["premoney"], f"{nm}: pre-money equity"))
    g = dict(zip(GFIELDS, vals[3 * n + 1:]))
    return ModelAssumptions(
        companies=companies,
        probabilities={"downside": _num(g["p_down"], "P(downside)"), "base": _num(g["p_base"], "P(base)"),
                       "upside": _num(g["p_up"], "P(upside)")},
        valuation_month=_num(g["val_month"], "valuation month"),
        sale_launch_month=_num(g["sale_month"], "sale launch month"),
        financing_mode=str(g["financing"]),
        focus=FocusSpec(_num(g["focus_mult"], "focus multiplier"), _num(g["nonfocus_mult"], "non-focus multiplier"),
                        _num(g["nonfocus_cap"], "non-focus cap"), _num(g["time_saving"], "focus time saving")),
        liquidity=LiquiditySpec(_num(g["liq_req"], "liquidity requirement"), _num(g["liq_deadline"], "liquidity deadline"),
                                _num(g["haircut"], "distress haircut") / 100, _num(g["extra_debt"], "distress extra debt")),
    )


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def money(v: float | None, d: int = 2, sign: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{'+' if sign and v > 0 else ''}{'−' if v < 0 else ''}${abs(v):,.{d}f}M"


def notice(kind: str, items: list[str]) -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{html.escape(i)}</li>" for i in items)
    return f'<div class="notice {kind}"><ul>{lis}</ul></div>'


def banner(mode: str) -> str:
    if mode == MODE_FACTS:
        return ('<div class="banner facts"><b>CASE FACTS ONLY.</b> The case gives current financials, funding '
                'requests and constraints, but not how investment changes EBITDA, risk or exit multiples. '
                'An optimal allocation cannot be derived from case facts alone, so no returns are shown. '
                'Switch to <i>Illustrative simulation</i> to load an editable example parameter set.</div>')
    return ('<div class="banner sim"><b>ILLUSTRATIVE SIMULATION.</b> Every projection depends on '
            f'assumptions labelled <i>{ILLUSTRATIVE_LABEL}</i>. Results are CONDITIONAL ON ASSUMPTIONS, '
            'not statistically validated forecasts. Edit them in the Financial Assumptions tab.</div>')


def clamp_alloc(x1, x2, x3) -> dict[str, float]:
    return {"ecom": float(x1 or 0), "it": float(x2 or 0), "sec": float(x3 or 0)}


N_MAIN = 14


def compute_main(mode, x1, x2, x3, focus, bscen, prev, vals):
    alloc = clamp_alloc(x1, x2, x3)
    total = sum(alloc.values())
    reserve = CASE_BUDGET - total

    def facts_outputs(msg_html: str):
        a0 = illustrative_assumptions()
        cards = (kpi("Total capital allocated", money(total), f"of ${CASE_BUDGET:.0f}M available")
                 + kpi("Capital remaining (reserve)", money(reserve), "kept by the fund")
                 + kpi("Funding requested", f"${CASE_TOTAL_REQUEST:.0f}M", "vs $15M available", "warn"))
        empty = charts.empty_figure("Projections require explicit assumptions.\nSwitch to 'Illustrative simulation'.")
        facts_df = pd.DataFrame([{
            "Company": NAMES[k], "Investment ($M)": alloc[k], "Request ($M)": c.request,
            "Latest EBITDA ($M)": c.latest_ebitda, "Current net debt ($M)": c.net_debt,
            "Fund ownership": f"{c.ownership:.0%}"} for k, c in CASE_COMPANIES.items()])
        return (msg_html, f'<div class="kpi-row">{cards}</div>', facts_df, pd.DataFrame(),
                "*No calculation breakdown in case-facts mode: the case does not specify exit EBITDA, "
                "multiples or investment responses.*", "", charts.allocation_chart(a0, alloc),
                empty, empty, empty, empty, empty, prev,
                '<div class="notice info">Case-facts mode: assumptions are not used.</div>')

    if mode == MODE_FACTS:
        return facts_outputs(banner(mode))

    try:
        a = values_to_assumptions(vals)
        validate_assumptions(a)
        res = evaluate_portfolio(a, alloc, focus)
    except AssumptionError as e:
        msg = banner(mode) + notice("error", [f"Assumption error: {e}"])
        out = list(facts_outputs(msg))
        out[13] = notice("error", [str(e)])
        return tuple(out)
    except Exception as e:
        traceback.print_exc()
        out = list(facts_outputs(banner(mode) + notice("error", [f"Unexpected error: {e}"])))
        return tuple(out)

    s = res.summary()
    sd, su = res.summary("downside"), res.summary("upside")
    feas_html = (notice("ok", ["Allocation is feasible: within the $15M budget, within each funding "
                               "request and within the non-focus support caps."]) if res.feasible
                 else notice("error", ["INFEASIBLE ALLOCATION:"] + res.feasibility))
    head = banner(mode) + feas_html + notice("warn", res.warnings)

    roi_txt = f"{s['roi']:.0%}" if s["roi"] is not None else "n/a"
    cards = "".join([
        kpi("Total capital allocated", money(total), f"of ${a.budget:.0f}M budget", "" if res.feasible else "bad"),
        kpi("Capital remaining (reserve)", money(res.reserve), "unallocated, kept by fund"),
        kpi("Expected fund proceeds", money(s["total_proceeds"]), f"no-investment: {money(s['baseline_proceeds'])}"),
        kpi("Additional proceeds", money(s["incremental_proceeds"], sign=True), "created by the intervention"),
        kpi("Net gain after new capital", money(s["net_gain"], sign=True), "additional proceeds − capital",
            "good" if s["net_gain"] >= 0 else "bad"),
        kpi("Incremental ROI", roi_txt, "net gain / new capital" if s["roi"] is not None else "needs ≥ $0.25M invested"),
        kpi("Downside net cash", money(sd["net_cash_returned"]), f"net gain {money(sd['net_gain'], sign=True)}"),
        kpi("Upside net cash", money(su["net_cash_returned"]), f"net gain {money(su['net_gain'], sign=True)}"),
    ])
    kpi_html = f'<div class="kpi-row">{cards}</div><div class="kpi-foot">Expected = probability-weighted ' \
               f'({res.probs["downside"]:.0%} / {res.probs["base"]:.0%} / {res.probs["upside"]:.0%}). ' \
               'Net cash returned = total fund proceeds − new capital. All values $M.</div>'

    rows = []
    for k in COMPANY_KEYS:
        ce = lambda f, b=False: res.company_expected(k, f, b)
        inc = ce("proceeds") - ce("proceeds", True)
        rows.append({
            "Company": NAMES[k] + (" ★" if k == focus else ""),
            "Investment ($M)": round(alloc[k], 2),
            "Exit EBITDA ($M)": round(ce("ebitda"), 2),
            "  of which uplift": round(ce("ebitda") - ce("ebitda", True), 2),
            "Exit multiple (x)": round(ce("multiple"), 2),
            "Enterprise value ($M)": round(ce("ev"), 1),
            "Exit net debt ($M)": round(ce("net_debt"), 1),
            "Equity value ($M)": round(ce("equity"), 1),
            "Fund ownership": f"{ce('ownership'):.2%}",
            "Fund proceeds ($M)": round(ce("proceeds"), 2),
            "No-investment proceeds ($M)": round(ce("proceeds", True), 2),
            "Additional proceeds ($M)": round(inc, 2),
            "Net benefit ($M)": round(inc - alloc[k], 2),
        })
    tot = {"Company": "TOTAL", "Investment ($M)": round(total, 2)}
    for col in ("Fund proceeds ($M)", "No-investment proceeds ($M)", "Additional proceeds ($M)", "Net benefit ($M)"):
        tot[col] = round(sum(r[col] for r in rows), 2)
    rows.append(tot)
    results_df = pd.DataFrame(rows).fillna("")

    scen_rows = []
    for s_ in list(SCENARIOS) + [None]:
        sm = res.summary(s_)
        scen_rows.append({
            "Scenario": SCENARIO_LABELS[s_] if s_ else "Probability-weighted",
            "Probability": f"{res.probs[s_]:.0%}" if s_ else "100%",
            "Fund proceeds ($M)": round(sm["total_proceeds"], 2),
            "No-investment ($M)": round(sm["baseline_proceeds"], 2),
            "Additional ($M)": round(sm["incremental_proceeds"], 2),
            "New capital ($M)": round(sm["invested"], 2),
            "Net cash returned ($M)": round(sm["net_cash_returned"], 2),
            "Net gain ($M)": round(sm["net_gain"], 2),
        })
    scen_df = pd.DataFrame(scen_rows)

    breakdown = calculation_breakdown(a, res, bscen or "base")

    br_lines = ["#### Value bridge vs no-investment counterfactual (expected, fund share, $M)",
                "Attribution order: EBITDA → multiple → net debt → ownership.",
                "| Company | EBITDA effect | Multiple (risk) effect | Net-debt effect | Ownership effect | = Additional proceeds | − Capital | = Net benefit |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in COMPANY_KEYS:
        b = expected_bridge(res, k)
        br_lines.append(f"| {NAMES[k]} | {b['ebitda']:+.2f} | {b['multiple']:+.2f} | {b['net_debt']:+.2f} | "
                        f"{b['ownership']:+.2f} | {b['total']:+.2f} | {alloc[k]:.2f} | {b['total'] - alloc[k]:+.2f} |")
    snap = {"alloc": alloc, "focus": focus,
            "co": {k: {"proceeds": res.company_expected(k, "proceeds"), "ebitda": res.company_expected(k, "ebitda"),
                       "multiple": res.company_expected(k, "multiple")} for k in COMPANY_KEYS},
            "net": s["net_gain"]}
    if prev and isinstance(prev, dict) and "alloc" in prev and (prev["alloc"] != alloc or prev["focus"] != focus):
        br_lines += ["", "#### What changed since your previous allocation",
                     f"Previous: E-com ${prev['alloc']['ecom']:.2f}M / IT ${prev['alloc']['it']:.2f}M / Security "
                     f"${prev['alloc']['sec']:.2f}M, focus {NAMES[prev['focus']]}  →  now: E-com ${alloc['ecom']:.2f}M / "
                     f"IT ${alloc['it']:.2f}M / Security ${alloc['sec']:.2f}M, focus {NAMES[focus]}.",
                     "| Company | Δ Capital | Δ Exit EBITDA | Δ Exit multiple | Δ Fund proceeds | Δ Net (proceeds − capital) |",
                     "|---|---:|---:|---:|---:|---:|"]
        for k in COMPANY_KEYS:
            dc = alloc[k] - prev["alloc"][k]
            d = {f: snap["co"][k][f] - prev["co"][k][f] for f in ("proceeds", "ebitda", "multiple")}
            br_lines.append(f"| {NAMES[k]} | {dc:+.2f} | {d['ebitda']:+.2f} | {d['multiple']:+.3f} | "
                            f"{d['proceeds']:+.2f} | {d['proceeds'] - dc:+.2f} |")
        br_lines.append(f"\n**Portfolio net gain changed by {snap['net'] - prev['net']:+.2f} $M** "
                        f"({prev['net']:.2f} → {snap['net']:.2f}). A company's Δ net is positive when the extra "
                        "fund proceeds exceed the extra capital it received.")
    bridge_md = "\n".join(br_lines)

    zero = evaluate_portfolio(a, {k: 0.0 for k in COMPANY_KEYS}, focus)
    try:
        opt = optimize(a, "expected_net", focus=None, step=0.25, max_list=1)
        ores = evaluate_portfolio(a, opt.best.allocation, opt.best.focus)
        comp = [("No investment", zero), ("Current", res), ("Optimized", ores)]
    except Exception:
        comp = [("No investment", zero), ("Current", res)]
    return (head, kpi_html, results_df, scen_df, breakdown, bridge_md,
            charts.allocation_chart(a, alloc), charts.investment_value_chart(a, res),
            charts.portfolio_chart(comp), charts.scenario_chart(comp),
            charts.timeline_chart(a, focus, alloc), charts.response_preview(a), snap,
            notice("ok", ["Assumptions valid. " + ILLUSTRATIVE_LABEL + " (all values in this tab)."]))


def on_change(mode, x1, x2, x3, focus, bscen, prev, *vals):
    return compute_main(mode, x1, x2, x3, focus, bscen, prev, list(vals))


def make_slider_handler(idx: int):
    def handler(mode, x1, x2, x3, focus, bscen, prev, *vals):
        xs = [float(x1 or 0), float(x2 or 0), float(x3 or 0)]
        others = sum(xs) - xs[idx]
        upd = [gr.update(), gr.update(), gr.update()]
        if sum(xs) > CASE_BUDGET + EPS:
            capped = float(max(np.floor((CASE_BUDGET - others) / 0.25 + EPS) * 0.25, 0.0))
            xs[idx] = capped
            upd[idx] = gr.update(value=capped)
        out = compute_main(mode, *xs, focus, bscen, prev, list(vals))
        return (*upd, *out)
    return handler


def reset_illustrative(mode, x1, x2, x3, focus, bscen, prev):
    vals = assumptions_to_values(illustrative_assumptions())
    return (MODE_SIM, *vals, *compute_main(MODE_SIM, x1, x2, x3, focus, bscen, prev, vals))


def reset_facts(x1, x2, x3, focus, bscen, prev, *vals):
    return (MODE_FACTS, *compute_main(MODE_FACTS, x1, x2, x3, focus, bscen, prev, list(vals)))


def run_optimizer(mode, x1, x2, x3, focus, objective, opt_focus, step, min_roi, *vals):
    if mode == MODE_FACTS:
        msg = ('<div class="banner facts"><b>No optimum from case facts alone.</b> The case does not say how '
               'investment changes EBITDA, risk or exit multiples, so every allocation is equally '
               '"optimal" on facts alone. Switch to <i>Illustrative simulation</i> (or enter your own '
               'assumptions) to optimise.</div>')
        return msg, "", pd.DataFrame(), charts.empty_figure("No optimisation in case-facts mode."), None
    try:
        a = values_to_assumptions(list(vals))
        step = float(step)
        r = optimize(a, objective, focus=None if opt_focus else focus, step=step,
                     min_roi_investment=float(min_roi or 1.0))
        cur = evaluate_portfolio(a, clamp_alloc(x1, x2, x3), focus)
        best = evaluate_portfolio(a, r.best.allocation, r.best.focus)
    except (AssumptionError, ValueError) as e:
        return notice("error", [str(e)]), "", pd.DataFrame(), charts.empty_figure(str(e)), None

    b = r.best
    bs, cs = best.summary(), cur.summary()
    bd, cd = best.summary("downside"), cur.summary("downside")
    al = b.allocation
    reserve = a.budget - b.invested
    res_html = f"""
<div class="banner sim"><b>Optimal UNDER YOUR CURRENT ASSUMPTIONS</b>, not a real-world optimum derived from the
case. Objective: <i>{OBJECTIVES[objective]}</i>. Exhaustive search over {r.n_evaluated:,} feasible allocations
on a ${step:g}M grid{' across all three focus choices' if opt_focus else f', focus fixed to {NAMES[focus]}'}.</div>
<div class="opt-box">
  <div class="opt-title">OPTIMAL INVESTMENT ALLOCATION</div>
  <table class="opt-table">
    <tr><td>E-commerce</td><td>{money(al['ecom'])}</td></tr>
    <tr><td>IT Engineering</td><td>{money(al['it'])}</td></tr>
    <tr><td>Security Hardware</td><td>{money(al['sec'])}</td></tr>
    <tr><td>Reserve (unallocated)</td><td>{money(reserve)}</td></tr>
    <tr class="tot"><td>TOTAL</td><td>{money(b.invested)} ≤ ${a.budget:.0f}M</td></tr>
    <tr><td>Intensive management focus</td><td>★ {NAMES[b.focus]}</td></tr>
  </table>
</div>
<div class="kpi-row">
  {kpi('Expected fund proceeds', money(bs['total_proceeds']), f"current: {money(cs['total_proceeds'])}")}
  {kpi('Net benefit after investment', money(bs['net_gain'], sign=True), f"current: {money(cs['net_gain'], sign=True)}", 'good' if bs['net_gain'] >= 0 else 'bad')}
  {kpi('Downside net cash', money(bd['net_cash_returned']), f"current: {money(cd['net_cash_returned'])}")}
  {kpi('vs your allocation', money(bs['net_cash_returned'] - cs['net_cash_returned'], sign=True), 'expected net cash difference')}
</div>"""
    reasons = explain_optimum(a, r, best, cur)
    near = pd.DataFrame([{
        "Rank": i + 1, "Focus": NAMES[c.focus], "E-com ($M)": c.allocation["ecom"], "IT ($M)": c.allocation["it"],
        "Security ($M)": c.allocation["sec"], "Reserve ($M)": round(a.budget - c.invested, 2),
        "Objective score": round(c.score, 4), "Expected net gain ($M)": round(c.expected_net_gain, 2),
        "Downside net cash ($M)": round(c.downside_net, 2), "Worst-scenario gain ($M)": round(c.worst_gain, 2)}
        for i, c in enumerate(r.near_optimal)])
    zero = evaluate_portfolio(a, {k: 0.0 for k in COMPANY_KEYS}, focus)
    fig = charts.portfolio_chart([("No investment", zero), ("Current", cur), ("Optimized", best)])
    return res_html, reasons, near, fig, {"alloc": b.allocation, "focus": b.focus}


def explain_optimum(a: ModelAssumptions, r, best: PortfolioResult, cur: PortfolioResult) -> str:
    b = r.best
    L = ["### Why this allocation"]
    mv = marginal_values(a, b.allocation, b.focus, r.grid_step)
    st = r.grid_step
    for k in COMPANY_KEYS:
        x = b.allocation[k]
        inc = best.company_expected(k, "proceeds") - best.company_expected(k, "proceeds", True)
        ca = a.companies[k]
        line = f"- **{NAMES[k]}: {money(x)}.** "
        if x > EPS:
            line += f"Adds {money(inc)} of expected fund proceeds ({inc / x:.2f}× the capital). "
        else:
            line += "No capital: at every tested amount the extra fund proceeds did not beat the alternatives. "
        m = mv[k]
        if m["next_step"] is not None:
            line += f"The next ${st:g}M would add {money(m['next_step'])} of proceeds "
            line += "(less than its cost). " if m["next_step"] < st else "(more than its cost, but budget is better used elsewhere or exhausted). "
        else:
            cap = a.case[k].request if k == b.focus else min(a.case[k].request, a.focus.nonfocus_cap)
            line += f"It is at its cap (${cap:g}M: {'request' if cap == a.case[k].request else 'non-focus support cap'}). "
        if ca.response.shape == "threshold" and abs(x - ca.response.threshold) < st / 2 + EPS:
            line += f"Investment sits exactly at the assumed ${ca.response.threshold:g}M threshold. "
        L.append(line)
    reserve = a.budget - b.invested
    if reserve > EPS:
        L.append(f"- **Reserve {money(reserve)}:** no company returns more than ${st:g}M of extra expected proceeds for "
                 f"the next ${st:g}M invested (or it is capped), so holding the cash is better under this objective.")
    else:
        L.append("- **Budget fully used:** the $15M constraint binds; the last dollars go where marginal proceeds are highest.")
    L.append("\n### Focus comparison (best allocation under each focus)")
    L.append("| Focus | E-com | IT | Security | Objective score | Expected net gain |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for fc, c in r.per_focus_best.items():
        L.append(f"| {'**' if fc == b.focus else ''}{NAMES[fc]}{'**' if fc == b.focus else ''} | {c.allocation['ecom']:.2f} | "
                 f"{c.allocation['it']:.2f} | {c.allocation['sec']:.2f} | {c.score:.3f} | {c.expected_net_gain:+.2f} |")
    L.append("\n### Key assumptions determining this result")
    for k in COMPANY_KEYS:
        ca = a.companies[k]
        rs = ca.response
        base = ca.scenarios["base"]
        shape = {"linear": f"linear to ${rs.full_effect_investment:g}M",
                 "diminishing": f"diminishing returns, k = {rs.curvature:g}, full at ${rs.full_effect_investment:g}M",
                 "threshold": f"threshold ${rs.threshold:g}M (jump {rs.jump:.0%}), full at ${rs.full_effect_investment:g}M",
                 "s_curve": f"S-curve, midpoint ${rs.midpoint:g}M, steepness {rs.steepness:g}",
                 "custom": "custom piecewise curve"}[rs.shape]
        L.append(f"- {NAMES[k]}: {shape}; max EBITDA uplift ${rs.max_ebitda_uplift:g}M, max multiple uplift "
                 f"{rs.max_multiple_uplift:g}x; base multiple {base.exit_multiple:g}x; ready in month "
                 f"{ca.timing.start_month + ca.timing.implementation_months:g} (base). ")
    L.append(f"- Focus multiplier {a.focus.focus_multiplier:g} vs non-focus {a.focus.nonfocus_multiplier:g}; "
             f"non-focus cap ${a.focus.nonfocus_cap:g}M; valuation month {a.valuation_month:g}; probabilities "
             f"{a.probabilities['downside']:g}/{a.probabilities['base']:g}/{a.probabilities['upside']:g}%.")
    if a.liquidity.requirement <= 0:
        L.append("- E-commerce liquidity requirement not specified → no distress penalty in any allocation.")
    for n in r.notes:
        L.append(f"\n> {n}")
    L.append("\n*Change any of these in Financial Assumptions and re-run: the optimum may move. "
             "Use Risk & Sensitivity to find the switching points.*")
    return "\n".join(L)


def apply_optimal(opt_state, mode, x1, x2, x3, focus, bscen, prev, *vals):
    if not opt_state:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                *compute_main(mode, x1, x2, x3, focus, bscen, prev, list(vals)))
    al, fc = opt_state["alloc"], opt_state["focus"]
    e, i, c = float(al["ecom"]), float(al["it"]), float(al["sec"])
    return (e, i, c, fc,
            *compute_main(mode, e, i, c, fc, bscen, prev, list(vals)))


SENS_CHOICES = [(v["label"], k) for k, v in SENS_PARAMS.items()]


def sens_defaults(param):
    lo, hi = SENS_PARAMS[param]["rng"]
    return lo, hi


def run_one_way(mode, focus, param, lo, hi, npts, objective, opt_focus, *vals):
    if mode == MODE_FACTS:
        return charts.empty_figure("Sensitivity needs assumptions (Illustrative simulation)."), pd.DataFrame(), ""
    try:
        a = values_to_assumptions(list(vals))
        xs = sweep_values(param, int(npts), float(lo), float(hi))
        rows = one_way_sensitivity(a, param, xs, objective, None if opt_focus else focus, 0.25)
    except (AssumptionError, ValueError) as e:
        return charts.empty_figure(str(e)), pd.DataFrame(), notice("error", [str(e)])
    lab = SENS_PARAMS[param]["label"]
    fig = charts.one_way_chart(rows, lab, OBJECTIVES[objective].split(":")[0])
    df = pd.DataFrame([{"Value": r["value"], "Optimal focus": NAMES[r["focus"]], "E-com ($M)": r["ecom"],
                        "IT ($M)": r["it"], "Security ($M)": r["sec"], "Reserve ($M)": round(r["reserve"], 2),
                        "Expected net gain ($M)": round(r["expected_net_gain"], 2),
                        "Downside net cash ($M)": round(r["downside_net"], 2)} for r in rows])
    sw = []
    for r0, r1 in zip(rows, rows[1:]):
        if r0["focus"] != r1["focus"]:
            sw.append(f"Optimal focus switches from {NAMES[r0['focus']]} to {NAMES[r1['focus']]} between "
                      f"{r0['value']:g} and {r1['value']:g}.")
        big0 = max(COMPANY_KEYS, key=lambda k: r0[k]); big1 = max(COMPANY_KEYS, key=lambda k: r1[k])
        if big0 != big1 and r0[big0] > 0 and r1[big1] > 0:
            sw.append(f"Largest allocation moves from {NAMES[big0]} to {NAMES[big1]} between {r0['value']:g} and {r1['value']:g}.")
    if not sw:
        sw = ["Within this range the optimal focus and the company receiving the most capital do not change: "
              "the recommendation is robust to this assumption alone."]
    txt = notice("info", sw + ["Conditional calculations, not proof of the future. Grid step $0.25M."])
    return fig, df, txt


def run_two_way(mode, focus, p1, p2, objective, opt_focus, *vals):
    if mode == MODE_FACTS:
        return charts.empty_figure("Sensitivity needs assumptions (Illustrative simulation).")
    if p1 == p2:
        return charts.empty_figure("Choose two different parameters.")
    try:
        a = values_to_assumptions(list(vals))
        v1, v2 = sweep_values(p1, 7), sweep_values(p2, 7)
        grid = two_way_sensitivity(a, p1, v1, p2, v2, objective, None if opt_focus else focus, step=0.5)
    except (AssumptionError, ValueError) as e:
        return charts.empty_figure(str(e))
    return charts.heatmap_chart(grid, SENS_PARAMS[p1]["label"], SENS_PARAMS[p2]["label"])


def run_breakeven(mode, *vals):
    if mode == MODE_FACTS:
        return charts.empty_figure("Break-even analysis needs assumptions."), ""
    try:
        a = values_to_assumptions(list(vals))
        probs = validate_assumptions(a)
    except (AssumptionError, ValueError) as e:
        return charts.empty_figure(str(e)), notice("error", [str(e)])
    lines = ["### Under what conditions does each company become the best single focus?",
             "Each company is tested as the only investment and focus (all budget available to it, others $0). "
             "The *break-even scale* is the factor by which its investment response (max EBITDA **and** multiple "
             "uplift) must be multiplied for its best standalone net benefit to match the best rival.",
             "", "| Company | Best standalone net benefit | at investment | Break-even response scale | Reading |",
             "|---|---:|---:|---:|---|"]
    for k in COMPANY_KEYS:
        g, x = standalone_best(a, k)
        sc = breakeven_response_scale(a, k)
        if sc is None:
            reading = "Cannot lead even at 10× its response (check timing / ownership / caps)."
            sc_txt = "> 10×"
        elif sc <= 1.0 + 1e-6:
            reading = "Already the strongest standalone choice." if sc < 1.0 - 1e-3 or g >= 0 else "Ties."
            sc_txt = f"{sc:.2f}×"
        else:
            reading = f"Leads only if its programme delivers {sc - 1:.0%} more than assumed."
            sc_txt = f"{sc:.2f}×"
        lines.append(f"| {NAMES[k]} | {money(g, sign=True)} | {money(x)} | {sc_txt} | {reading} |")
    lines.append("\nRelated levers: fund ownership (E-com 30%, IT 30%, Security 20%) means $1M of company equity "
                 "value is worth $0.30M / $0.30M / $0.20M to the fund; a Security Hardware improvement must "
                 "therefore be 1.5× larger in company value to match the same fund cash.")
    return charts.standalone_curves(a, probs), "\n".join(lines)


def overview_table() -> pd.DataFrame:
    rows = []
    for k, c in CASE_COMPANIES.items():
        rows.append({
            "Company": c.name, "Previous revenue": c.prev_revenue, "Latest revenue": c.latest_revenue,
            "Revenue change": f"{c.latest_revenue / c.prev_revenue - 1:+.1%}",
            "Previous EBITDA": c.prev_ebitda, "Latest EBITDA": c.latest_ebitda,
            "EBITDA change": f"{c.latest_ebitda / c.prev_ebitda - 1:+.1%}",
            "Latest EBITDA margin": f"{c.latest_margin:.0%}", "Current net debt": c.net_debt,
            "Fund ownership": f"{c.ownership:.0%}", "Funding request": c.request})
    return pd.DataFrame(rows)


def challenge_cards() -> str:
    out = []
    for k, c in CASE_COMPANIES.items():
        lis = "".join(f"<li>{html.escape(x)}</li>" for x in c.challenges)
        out.append(f'<div class="co-card" style="border-top:4px solid {charts.COLORS[k]}">'
                   f'<div class="co-name">{c.name}</div><div class="co-req">Requests ${c.request:.0f}M · '
                   f'fund owns {c.ownership:.0%}</div><ul>{lis}</ul></div>')
    return f'<div class="co-row">{"".join(out)}</div>'


try:
    METHODOLOGY = (Path(__file__).parent / "METHODOLOGY.md").read_text(encoding="utf-8")
except Exception:
    METHODOLOGY = "## Methodology\nRefer to case documentation for detailed calculations."

GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

CSS = """
.gradio-container {max-width: 1400px !important; margin: 0 auto;}
.app-title h1 {font-size: 1.55rem; margin-bottom: 0;}
.app-title p {color: #475569; margin-top: 4px;}
.banner {padding: 10px 14px; border-radius: 8px; margin: 6px 0; font-size: 0.92rem; line-height: 1.45;}
.banner.sim {background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a;}
.banner.facts {background: #fefce8; border: 1px solid #fde68a; color: #713f12;}
.notice {padding: 6px 12px; border-radius: 8px; margin: 6px 0; font-size: 0.88rem;}
.notice ul {margin: 2px 0 2px 16px; padding: 0;}
.notice.ok {background: #f0fdf4; border: 1px solid #bbf7d0; color: #14532d;}
.notice.warn {background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12;}
.notice.error {background: #fef2f2; border: 1px solid #fecaca; color: #7f1d1d;}
.notice.info {background: #f8fafc; border: 1px solid #e2e8f0; color: #334155;}
.kpi-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 8px 0;}
.kpi {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 12px;}
.kpi-label {font-size: 0.74rem; text-transform: uppercase; letter-spacing: .04em; color: #64748b;}
.kpi-value {font-size: 1.35rem; font-weight: 650; color: #0f172a; font-variant-numeric: tabular-nums;}
.kpi-sub {font-size: 0.78rem; color: #64748b;}
.kpi.good .kpi-value {color: #15803d;} .kpi.bad .kpi-value {color: #b91c1c;} .kpi.warn .kpi-value {color: #b45309;}
.kpi-foot {font-size: 0.78rem; color: #64748b;}
.opt-box {border: 2px solid #0f172a; border-radius: 10px; padding: 12px 16px; margin: 10px 0; max-width: 520px; background: #fff;}
.opt-title {font-weight: 700; letter-spacing: .05em; margin-bottom: 6px; color: #0f172a;}
.opt-table {width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums;}
.opt-table td {padding: 3px 4px; border: none; color: #0f172a;} .opt-table td:last-child {text-align: right; font-weight: 600;}
.opt-table tr.tot td {border-top: 1px solid #94a3b8; font-weight: 700;}
.co-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px;}
.co-card {background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; color: #0f172a;}
.co-name {font-weight: 650;} .co-req {color: #64748b; font-size: 0.85rem;}
.co-card ul {margin: 6px 0 0 16px; padding: 0; font-size: 0.9rem;}
.assume-tag {font-size: 0.78rem; font-weight: 600; color: #b45309; letter-spacing: .03em;}
"""


THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")


def build_app() -> gr.Blocks:
    A0 = illustrative_assumptions()
    V0 = assumptions_to_values(A0)
    G0 = dict(zip(GFIELDS, V0[-len(GFIELDS):]))

    blocks_kwargs = {} if GRADIO_6 else {"css": CSS, "theme": THEME}
    with gr.Blocks(title="PE Portfolio Optimization Simulator", **blocks_kwargs) as demo:
        gr.HTML('<div class="app-title"><h1>Private Equity Portfolio Optimization Simulator</h1>'
                '<p>Case Challenge 3.0 · SKELAR × Horizon Capital × Mastercard · $15M of capital, '
                '$33M of requests, one intensive programme, exit within 24 months. All values in $ millions. '
                'Case figures are illustrative and do not represent real companies.</p></div>')
        with gr.Row():
            mode = gr.Radio([MODE_SIM, MODE_FACTS], value=MODE_SIM, label="Model state", scale=3,
                            info="Case facts only = no projections. Illustrative simulation = editable example assumptions.")
            btn_reset_facts = gr.Button("Reset to case facts", scale=1)
            btn_reset_ill = gr.Button("Reset illustrative assumptions", scale=1)

        prev_state = gr.State(None)
        opt_state = gr.State(None)

        with gr.Tab("1 · Portfolio overview"):
            gr.Markdown("#### Case facts (hardcoded from the case datapack, $M)")
            gr.Dataframe(overview_table(), interactive=False, wrap=True)
            gr.HTML(challenge_cards())
            gr.Plot(charts.overview_chart(A0))
            gr.Markdown(
                "**Constraints from the case:** $15M available vs $33M requested · only ONE intensive "
                "value-creation programme · no additional funding committed · board seats in all three; "
                "major changes need founder support (assumed available) · exit ALL companies within 24 months.  \n"
                "**Objective:** maximise cash returned to the fund within 24 months, net of execution risk and "
                "additional capital invested.  \n"
                "**What the case does *not* give:** how investment changes EBITDA or risk, exit multiples, exit-date "
                "net debt, the e-commerce liquidity amount, or the financing instrument. These are the assumptions "
                "the simulator makes explicit.")

        with gr.Tab("2 · Capital allocation simulator"):
            head_html = gr.HTML()
            with gr.Row():
                with gr.Column(scale=4, min_width=320):
                    gr.Markdown("#### Allocate the $15M")
                    x1 = gr.Slider(0, 10, value=10, step=0.25, label="E-commerce ($M, request $10M)")
                    x2 = gr.Slider(0, 8, value=0, step=0.25, label="IT Engineering ($M, request $8M)")
                    x3 = gr.Slider(0, 15, value=3, step=0.25, label="Security Hardware ($M, request $15M)")
                    focus = gr.Radio(FOCUS_CHOICES, value="ecom", label="PRIMARY MANAGEMENT FOCUS (only one)",
                                     info="The focus company gets the focus effectiveness multiplier; the other two "
                                          "receive limited/defensive support up to the non-focus cap.")
                    gr.Markdown("<span class='assume-tag'>If a slider would push the total above $15M it is capped "
                                "at the remaining budget. Other allocations are never reduced. Unallocated capital "
                                "is kept as a reserve.</span>")
                with gr.Column(scale=6):
                    kpi_html = gr.HTML()
                    plot_alloc = gr.Plot()
            results_df = gr.Dataframe(label="Projected results by company (probability-weighted, $M; ★ = focus)",
                                      interactive=False, wrap=True)
            plot_inv = gr.Plot()
            with gr.Row():
                plot_port = gr.Plot()
            plot_scen = gr.Plot()
            scen_df = gr.Dataframe(label="Downside / base / upside results ($M)", interactive=False, wrap=True)
            plot_time = gr.Plot()
            with gr.Accordion("Inspect the calculation: why did the results change?", open=True):
                bridge_md = gr.Markdown()
                bscen = gr.Radio([(SCENARIO_LABELS[s], s) for s in SCENARIOS], value="base",
                                 label="Step-by-step calculation for scenario")
                breakdown_md = gr.Markdown()

        comps: dict[str, list] = {}
        with gr.Tab("3 · Financial assumptions"):
            assume_status = gr.HTML()
            gr.Markdown(f"<span class='assume-tag'>Every value on this tab is an {ILLUSTRATIVE_LABEL}. "
                        "Changes apply immediately to every tab.</span>")
            with gr.Accordion("Scenarios: exit EBITDA without investment, exit multiple, exit net debt, effectiveness, delay", open=True):
                gr.Markdown("*Baseline exit EBITDA* = EBITDA at exit with **no** additional fund capital. "
                            "*Exit multiple* = EV/EBITDA before any risk-reduction uplift (not market data). "
                            "*Exit net debt* excludes the new capital, which is assumed spent on the programme. "
                            "*Effectiveness* scales how much of the designed response is delivered. "
                            "*Extra delay* adds months to implementation in that scenario.")
                scen_table = gr.Dataframe(value=V0[3 * len(CFIELDS)], headers=SCEN_COLUMNS, interactive=True,
                                          label="Scenario assumptions")
                with gr.Row():
                    p_down = gr.Number(value=G0["p_down"], label="P(Downside) %", minimum=0, maximum=100)
                    p_base = gr.Number(value=G0["p_base"], label="P(Base) %", minimum=0, maximum=100)
                    p_up = gr.Number(value=G0["p_up"], label="P(Upside) %", minimum=0, maximum=100)
                gr.Markdown("Probabilities are your judgement, not empirical success rates. They must sum to 100%.")
            preview = gr.Plot()
            for ci, k in enumerate(COMPANY_KEYS):
                with gr.Accordion(f"{NAMES[k]}: investment response, timing, financing", open=(ci == 0)):
                    v = dict(zip(CFIELDS, V0[ci * len(CFIELDS):(ci + 1) * len(CFIELDS)]))
                    with gr.Row():
                        shape = gr.Dropdown(SHAPE_CHOICES, value=v["shape"], label="Response shape",
                                            info="How capital turns into programme progress f(x), 0–100%.")
                        X = gr.Number(value=v["X"], label="Full-effect investment ($M)", info="f(x) reaches 100% here")
                        maxE = gr.Number(value=v["maxE"], label="Max EBITDA uplift ($M run-rate)",
                                         info="Operating improvement at 100% progress")
                        maxM = gr.Number(value=v["maxM"], label="Max multiple uplift (x)",
                                         info="Risk reduction / buyer re-rating at 100% progress")
                    with gr.Row():
                        kk = gr.Number(value=v["k"], label="Diminishing: curvature k (1/$M)",
                                       info="Higher = faster early gains, earlier saturation")
                        T = gr.Number(value=v["T"], label="Threshold: minimum viable investment ($M)")
                        jump = gr.Number(value=v["jump"], label="Threshold: effect unlocked at threshold (%)")
                        below = gr.Number(value=v["below"], label="Threshold: effect just below threshold (%)")
                    with gr.Row():
                        mid = gr.Number(value=v["mid"], label="S-curve: midpoint ($M)", info="Steepest point")
                        steep = gr.Number(value=v["steep"], label="S-curve: steepness (1/$M)")
                        custom = gr.Textbox(value=v["custom"], label="Custom: points 'investment:% of full effect'",
                                            info="e.g. 2:5, 4:50, 6:75, 9:100 — (0,0) is added; flat after last point")
                    with gr.Row():
                        start = gr.Number(value=v["start"], label="Implementation start (month)")
                        impl = gr.Number(value=v["impl"], label="Implementation duration (months)")
                        lag = gr.Number(value=v["lag"], label="Time until benefits begin after readiness (months)")
                        ramp = gr.Number(value=v["ramp"], label="Ramp-up to full run-rate (months)")
                        partial = gr.Number(value=v["partial"], label="Buyer credit for unfinished risk reduction (%)")
                        premoney = gr.Number(value=v["premoney"], label="Pre-money equity value ($M, equity mode only)")
                    comps[k] = [shape, X, kk, T, jump, below, mid, steep, custom, maxE, maxM,
                                start, impl, lag, ramp, partial, premoney]
            with gr.Accordion("Management attention, valuation timing, financing, e-commerce liquidity", open=True):
                with gr.Row():
                    focus_mult = gr.Number(value=G0["focus_mult"], label="Focus effectiveness multiplier (×)")
                    nonfocus_mult = gr.Number(value=G0["nonfocus_mult"], label="Non-focus effectiveness multiplier (×)")
                    nonfocus_cap = gr.Number(value=G0["nonfocus_cap"], label="Non-focus capital cap ($M)",
                                             info="Max limited/defensive support without the intensive programme")
                    time_saving = gr.Number(value=G0["time_saving"], label="Focus shortens implementation by (months)")
                with gr.Row():
                    val_month = gr.Slider(1, 24, value=G0["val_month"], step=1, label="Valuation month",
                                          info="Run-rate EBITDA and risk profile buyers can verify at this month")
                    sale_month = gr.Slider(0, 24, value=G0["sale_month"], step=1, label="Sale process launch (month)")
                    financing = gr.Radio(FIN_CHOICES, value=G0["financing"], label="Financing convention")
                with gr.Row():
                    liq_req = gr.Number(value=G0["liq_req"], label="E-commerce liquidity requirement ($M; 0 = not specified)",
                                        info="The case gives the 6-month deadline, not the amount")
                    liq_deadline = gr.Number(value=G0["liq_deadline"], label="Liquidity deadline (month)")
                    haircut = gr.Number(value=G0["haircut"], label="If unmet: exit EBITDA haircut (%)")
                    extra_debt = gr.Number(value=G0["extra_debt"], label="If unmet: extra net debt ($M)")

        globals_ = [p_down, p_base, p_up, val_month, sale_month, financing, focus_mult, nonfocus_mult,
                    nonfocus_cap, time_saving, liq_req, liq_deadline, haircut, extra_debt]
        assumption_inputs = comps["ecom"] + comps["it"] + comps["sec"] + [scen_table] + globals_

        with gr.Tab("4 · Optimization"):
            gr.HTML('<div class="banner sim">The optimizer finds the best allocation <b>under the assumptions you '
                    'have entered</b>. It cannot find a real-world objectively optimal allocation from the case data '
                    'alone.</div>')
            with gr.Row():
                objective = gr.Radio([(v, k) for k, v in OBJECTIVES.items()], value="expected_net", label="Objective", scale=3)
                with gr.Column(scale=2):
                    opt_focus = gr.Checkbox(value=True, label="Also optimise which ONE company gets intensive focus")
                    step = gr.Dropdown([("$0.25M", 0.25), ("$0.50M", 0.5), ("$1.00M", 1.0)], value=0.25, label="Grid step")
                    min_roi = gr.Number(value=1.0, label="ROI objective: minimum capital considered ($M)")
            btn_opt = gr.Button("FIND OPTIMAL ALLOCATION", variant="primary", size="lg")
            opt_html = gr.HTML()
            btn_apply = gr.Button("Apply optimal allocation to the simulator sliders")
            with gr.Row():
                with gr.Column(scale=5):
                    opt_md = gr.Markdown()
                with gr.Column(scale=5):
                    opt_plot = gr.Plot()
            near_df = gr.Dataframe(label="Near-optimal allocations (within tolerance of the best score)",
                                   interactive=False, wrap=True)

        with gr.Tab("5 · Risk & sensitivity"):
            gr.HTML('<div class="banner sim">Sensitivity results are conditional financial calculations, not proof '
                    'of the future. They show which assumptions change the recommendation.</div>')
            plot_scen2 = gr.Plot()
            gr.Markdown("#### One-way sensitivity: re-optimise as one assumption changes")
            with gr.Row():
                s_param = gr.Dropdown(SENS_CHOICES, value="sec:response_magnitude", label="Assumption", scale=3)
                s_lo = gr.Number(value=0.0, label="From")
                s_hi = gr.Number(value=2.0, label="To")
                s_n = gr.Slider(3, 21, value=11, step=1, label="Points")
            with gr.Row():
                s_obj = gr.Dropdown([(v, k) for k, v in OBJECTIVES.items()], value="expected_net", label="Objective", scale=3)
                s_focus = gr.Checkbox(value=True, label="Optimise focus too")
                btn_s1 = gr.Button("Run one-way sensitivity", variant="primary")
            s1_plot = gr.Plot()
            s1_txt = gr.HTML()
            s1_df = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("#### Two-way sensitivity heatmap (7 × 7, $0.50M grid)")
            with gr.Row():
                h1 = gr.Dropdown(SENS_CHOICES, value="ecom:response_magnitude", label="X axis")
                h2 = gr.Dropdown(SENS_CHOICES, value="sec:response_magnitude", label="Y axis")
                btn_h = gr.Button("Run heatmap", variant="primary")
            h_plot = gr.Plot()
            gr.Markdown("#### Break-even: when does Company A beat B or C?")
            btn_be = gr.Button("Run break-even analysis", variant="primary")
            with gr.Row():
                be_plot = gr.Plot()
                be_md = gr.Markdown()

        with gr.Tab("6 · Methodology"):
            gr.Markdown(METHODOLOGY)

        main_outputs = [head_html, kpi_html, results_df, scen_df, breakdown_md, bridge_md, plot_alloc,
                        plot_inv, plot_port, plot_scen, plot_time, preview, prev_state, assume_status]
        assert len(main_outputs) == N_MAIN
        ctx = [mode, x1, x2, x3, focus, bscen, prev_state]

        def with_scen2(fn):
            def wrapped(*args):
                out = fn(*args)
                return (*out, out[-N_MAIN + 9])
            return wrapped

        all_out = main_outputs + [plot_scen2]
        for comp in [mode, focus, bscen] + assumption_inputs:
            comp.input(with_scen2(on_change), inputs=ctx + assumption_inputs, outputs=all_out,
                       trigger_mode="always_last", show_progress="hidden")
        for i, sl in enumerate([x1, x2, x3]):
            sl.input(with_scen2(make_slider_handler(i)), inputs=ctx + assumption_inputs,
                     outputs=[x1, x2, x3] + all_out, trigger_mode="always_last", show_progress="hidden")
        btn_reset_ill.click(with_scen2(reset_illustrative), inputs=ctx, outputs=[mode] + assumption_inputs + all_out)
        btn_reset_facts.click(with_scen2(reset_facts), inputs=ctx[1:] + assumption_inputs, outputs=[mode] + all_out)
        btn_opt.click(run_optimizer, inputs=[mode, x1, x2, x3, focus, objective, opt_focus, step, min_roi] + assumption_inputs,
                      outputs=[opt_html, opt_md, near_df, opt_plot, opt_state])
        btn_apply.click(with_scen2(apply_optimal), inputs=[opt_state] + ctx + assumption_inputs,
                        outputs=[x1, x2, x3, focus] + all_out)
        s_param.change(sens_defaults, inputs=s_param, outputs=[s_lo, s_hi])
        btn_s1.click(run_one_way, inputs=[mode, focus, s_param, s_lo, s_hi, s_n, s_obj, s_focus] + assumption_inputs,
                     outputs=[s1_plot, s1_df, s1_txt])
        btn_h.click(run_two_way, inputs=[mode, focus, h1, h2, s_obj, s_focus] + assumption_inputs, outputs=h_plot)
        btn_be.click(run_breakeven, inputs=[mode] + assumption_inputs, outputs=[be_plot, be_md])
        demo.load(with_scen2(on_change), inputs=ctx + assumption_inputs, outputs=all_out)
    return demo


demo = build_app()
if __name__ == "__main__":
    import os
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,::1"
    
    port = int(os.environ.get("PORT", 10000))
    launch_kwargs = {"theme": THEME, "css": CSS} if GRADIO_6 else {}
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, **launch_kwargs)