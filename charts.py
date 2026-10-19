"""Plotly figures. Every monetary axis is labelled in $M."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assumptions import COMPANY_KEYS, SCENARIO_LABELS, SCENARIOS, ModelAssumptions
from financial_model import PortfolioResult, company_values, response_fraction, timing_effect

COLORS = {"ecom": "#2563eb", "it": "#d97706", "sec": "#059669", "reserve": "#94a3b8", "none": "#cbd5e1"}
NAMES = {"ecom": "E-commerce", "it": "IT Engineering", "sec": "Security Hardware", "reserve": "Reserve", "none": "No investment"}
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def _layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
        font=dict(family=FONT, size=12, color="#1e293b"),
        paper_bgcolor="white", plot_bgcolor="white", height=height,
        margin=dict(l=60, r=24, t=64, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", linecolor="#cbd5e1", zerolinecolor="#94a3b8")
    return fig


def empty_figure(msg: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color="#64748b"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, "", height)


# CHART 1 -----------------------------------------------------------------
def allocation_chart(a: ModelAssumptions, alloc: dict[str, float]) -> go.Figure:
    reserve = max(a.budget - sum(alloc.values()), 0.0)
    keys = list(COMPANY_KEYS) + ["reserve"]
    vals = [alloc[k] for k in COMPANY_KEYS] + [reserve]
    fig = go.Figure()
    fig.add_bar(x=[NAMES[k] for k in keys], y=vals, marker_color=[COLORS[k] for k in keys],
                text=[f"${v:.2f}M" for v in vals], textposition="outside", name="Allocated",
                hovertemplate="%{x}: $%{y:.2f}M<extra></extra>")
    fig.add_scatter(x=[NAMES[k] for k in COMPANY_KEYS], y=[a.case[k].request for k in COMPANY_KEYS],
                    mode="markers", marker=dict(symbol="line-ew", size=38, line=dict(width=2, color="#0f172a")),
                    name="Funding request (cap)", hovertemplate="Request: $%{y:.0f}M<extra></extra>")
    fig.update_yaxes(title="Capital ($M)", range=[0, max(16, max(vals) * 1.2)])
    return _layout(fig, "Chart 1 · Capital allocation of the $15M budget", 340)


# CHART 2 -----------------------------------------------------------------
def investment_value_chart(a: ModelAssumptions, res: PortfolioResult) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Exit EBITDA uplift vs investment (company level, expected, $M)",
        "Incremental expected fund proceeds vs investment ($M)"))
    probs = res.probs
    for k in COMPANY_KEYS:
        cap = a.case[k].request
        g = np.linspace(0, cap, 161)
        is_f = k == res.focus
        e_up = sum(probs[s] * (company_values(a, k, s, g, is_f)["ebitda"] - company_values(a, k, s, [0.0], is_f)["ebitda"][0]) for s in SCENARIOS)
        p = sum(probs[s] * company_values(a, k, s, g, is_f)["proceeds"] for s in SCENARIOS)
        inc = p - p[0]
        x0 = res.allocation[k]
        i0 = int(np.argmin(np.abs(g - x0)))
        fig.add_scatter(x=g, y=e_up, mode="lines", line=dict(color=COLORS[k], width=2.5), name=NAMES[k],
                        legendgroup=k, row=1, col=1, hovertemplate="$%{x:.2f}M → EBITDA +$%{y:.2f}M<extra>" + NAMES[k] + "</extra>")
        fig.add_scatter(x=g, y=inc, mode="lines", line=dict(color=COLORS[k], width=2.5), name=NAMES[k],
                        legendgroup=k, showlegend=False, row=1, col=2,
                        hovertemplate="$%{x:.2f}M → fund +$%{y:.2f}M<extra>" + NAMES[k] + "</extra>")
        for col, arr in ((1, e_up), (2, inc)):
            fig.add_scatter(x=[x0], y=[arr[i0]], mode="markers", marker=dict(size=11, color=COLORS[k], line=dict(width=2, color="white")),
                            showlegend=False, legendgroup=k, row=1, col=col, hovertemplate="Current: $%{x:.2f}M → $%{y:.2f}M<extra></extra>")
    g = np.linspace(0, 15, 2)
    fig.add_scatter(x=g, y=g, mode="lines", line=dict(color="#64748b", dash="dash", width=1.5),
                    name="Break-even (proceeds = capital)", row=1, col=2, hoverinfo="skip")
    fig.update_xaxes(title_text="Investment ($M)")
    fig.update_yaxes(title_text="EBITDA uplift ($M)", row=1, col=1)
    fig.update_yaxes(title_text="Additional fund proceeds ($M)", row=1, col=2)
    _layout(fig, "Chart 2 · Investment vs financial value (dots = current allocation; current focus applied)", 420)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5), margin=dict(b=90))
    fig.update_annotations(font_size=12)
    return fig


# CHART 3 -----------------------------------------------------------------
def portfolio_chart(rows: list[tuple[str, PortfolioResult]]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=(
        "Expected fund proceeds by company ($M)", "Incremental net benefit after new capital ($M)"))
    labels = [r[0] for r in rows]
    for k in COMPANY_KEYS:
        fig.add_bar(x=labels, y=[r.company_expected(k, "proceeds") for _, r in rows], name=NAMES[k],
                    marker_color=COLORS[k], row=1, col=1, hovertemplate="%{x}: $%{y:.2f}M<extra>" + NAMES[k] + "</extra>")
    nets = [r.summary()["net_gain"] for _, r in rows]
    fig.add_bar(x=labels, y=nets, marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in nets],
                text=[f"${v:.2f}M" for v in nets], textposition="outside", showlegend=False, row=1, col=2,
                hovertemplate="%{x}: $%{y:.2f}M<extra>net benefit</extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="$M", row=1, col=1)
    fig.update_yaxes(title_text="$M", row=1, col=2)
    _layout(fig, "Chart 3 · Portfolio proceeds: no investment vs current vs optimized", 400)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5), margin=dict(b=90))
    fig.update_annotations(font_size=12)
    return fig


# CHART 4 -----------------------------------------------------------------
def scenario_chart(rows: list[tuple[str, PortfolioResult]]) -> go.Figure:
    fig = go.Figure()
    palette = ["#94a3b8", "#2563eb", "#059669", "#d97706"]
    xs = [SCENARIO_LABELS[s] for s in SCENARIOS] + ["Probability-weighted"]
    for i, (lab, r) in enumerate(rows):
        ys = [r.summary(s)["net_cash_returned"] for s in SCENARIOS] + [r.summary()["net_cash_returned"]]
        fig.add_bar(x=xs, y=ys, name=lab, marker_color=palette[i % len(palette)],
                    text=[f"{v:.1f}" for v in ys], textposition="outside",
                    hovertemplate="%{x}: $%{y:.2f}M<extra>" + lab + "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Net cash returned to fund ($M)")
    return _layout(fig, "Chart 4 · Scenario comparison: fund proceeds minus new capital", 380)


# TIMELINE ----------------------------------------------------------------
def timeline_chart(a: ModelAssumptions, focus: str, alloc: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    ylabels = []
    phase_colors = {"Implementation": "#93c5fd", "Benefit lag": "#fde68a", "EBITDA ramp-up": "#86efac",
                    "Exit preparation": "#e2e8f0", "Sale process": "#c4b5fd"}
    shown = set()

    def bar(y, start, end, phase, extra=""):
        start, end = max(start, 0), min(end, 30)
        if end <= start:
            return
        fig.add_bar(y=[y], x=[end - start], base=[start], orientation="h", marker_color=phase_colors[phase],
                    name=phase, showlegend=phase not in shown, legendgroup=phase,
                    hovertemplate=f"{phase}: month {start:g} → {end:g}{extra}<extra>{y}</extra>")
        shown.add(phase)

    for k in COMPANY_KEYS:
        t = a.companies[k].timing
        tm = timing_effect(t, a.companies[k].scenarios["base"].delay_months, a.valuation_month,
                           a.focus.focus_time_saving if k == focus else 0.0)
        y = f"{NAMES[k]}{' ★' if k == focus else ''} (${alloc[k]:.2f}M)"
        ylabels.append(y)
        bar(y, tm.start, tm.ready, "Implementation")
        bar(y, tm.ready, tm.benefit_start, "Benefit lag")
        bar(y, tm.benefit_start, tm.full_run_rate, "EBITDA ramp-up")
        fig.add_scatter(y=[y], x=[tm.start], mode="markers", marker=dict(symbol="diamond", size=11, color="#0f172a"),
                        name="Capital commitment", showlegend=k == "ecom", hovertemplate="Commitment: month %{x:g}<extra></extra>")
        fig.add_scatter(y=[y], x=[tm.ready], mode="markers", marker=dict(symbol="star", size=12, color="#2563eb"),
                        name="Operational readiness", showlegend=k == "ecom", hovertemplate="Ready: month %{x:g}<extra></extra>")
    fund = "Fund: exit process"
    bar(fund, max(a.sale_launch_month - 6, 0), a.sale_launch_month, "Exit preparation")
    bar(fund, a.sale_launch_month, 24, "Sale process")
    fig.add_vline(x=24, line=dict(color="#dc2626", width=2), annotation_text="Exit deadline (24)", annotation_position="top")
    fig.add_vline(x=a.valuation_month, line=dict(color="#7c3aed", dash="dot"), annotation_text=f"Valuation month ({a.valuation_month:g})",
                  annotation_position="bottom right")
    fig.add_vline(x=a.liquidity.deadline_month, line=dict(color="#ea580c", dash="dash"), annotation_text="E-com liquidity deadline",
                  annotation_position="top left")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Month", range=[0, 30], tickvals=[0, 6, 12, 18, 24, 30])
    fig.update_yaxes(categoryorder="array", categoryarray=[fund] + ylabels[::-1])
    return _layout(fig, "24-month implementation timeline (base case; ★ = management focus)", 360)


# RESPONSE CURVE PREVIEW --------------------------------------------------
def response_preview(a: ModelAssumptions) -> go.Figure:
    fig = go.Figure()
    for k in COMPANY_KEYS:
        cap = a.case[k].request
        g = np.linspace(0, cap, 241)
        try:
            f = response_fraction(g, a.companies[k].response)
        except Exception:
            continue
        fig.add_scatter(x=g, y=f * 100, mode="lines", line=dict(color=COLORS[k], width=2.5),
                        name=f"{NAMES[k]} ({a.companies[k].response.shape})",
                        hovertemplate="$%{x:.2f}M → %{y:.1f}% of full effect<extra></extra>")
    fig.update_xaxes(title="Investment ($M)")
    fig.update_yaxes(title="Programme progress f(x), % of full effect")
    return _layout(fig, "Investment-response functions (before effectiveness, focus and timing)", 360)


# SENSITIVITY -------------------------------------------------------------
def one_way_chart(rows: list[dict], label: str, objective_label: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    xs = [f"{r['value']:g}" for r in rows]
    for k in list(COMPANY_KEYS) + ["reserve"]:
        fig.add_bar(x=xs, y=[r[k] for r in rows], name=NAMES[k], marker_color=COLORS[k],
                    hovertemplate="%{x}: $%{y:.2f}M<extra>" + NAMES[k] + "</extra>", secondary_y=False)
    fig.add_scatter(x=xs, y=[r["expected_net_gain"] for r in rows], mode="lines+markers", name="Expected net benefit ($M)",
                    line=dict(color="#0f172a", width=2), secondary_y=True)
    fig.add_scatter(x=xs, y=[15.8] * len(rows), mode="markers", marker=dict(symbol="star", size=11,
                    color=[COLORS[r["focus"]] for r in rows]), name="Optimal focus (★ colour)",
                    text=[NAMES[r["focus"]] for r in rows], hovertemplate="Focus: %{text}<extra></extra>", secondary_y=False)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title=label)
    fig.update_yaxes(title_text="Optimal allocation ($M)", range=[0, 16.5], secondary_y=False)
    fig.update_yaxes(title_text="Expected incremental net benefit ($M)", secondary_y=True, showgrid=False)
    return _layout(fig, f"Chart 5 · Optimal allocation as an assumption changes ({objective_label}; ★ = optimal focus)", 430)


def heatmap_chart(grid: list[list[dict]], lab1: str, lab2: str) -> go.Figure:
    order = ["none", "ecom", "it", "sec"]
    z, text = [], []
    for row in grid:
        zr, tr = [], []
        for c in row:
            vals = {k: c[k] for k in COMPANY_KEYS}
            dom = max(vals, key=vals.get) if max(vals.values()) > 1e-9 else "none"
            zr.append(order.index(dom))
            tr.append(f"{NAMES[dom][:4]}<br>{c['ecom']:.1f}/{c['it']:.1f}/{c['sec']:.1f}")
        z.append(zr)
        text.append(tr)
    cs = []
    for i, k in enumerate(order):
        cs += [[i / 4, COLORS[k]], [(i + 1) / 4, COLORS[k]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c['x']:.2f}" for c in grid[0]], y=[f"{r[0]['y']:.2f}" for r in grid],
        zmin=-0.5, zmax=3.5, colorscale=cs, text=text, texttemplate="%{text}", textfont=dict(size=9, color="white"),
        showscale=False, hovertemplate=f"{lab1}: %{{x}}<br>{lab2}: %{{y}}<br>%{{text}}<extra></extra>"))
    fig.update_xaxes(title=lab1)
    fig.update_yaxes(title=lab2)
    return _layout(fig, "Two-way sensitivity: company receiving the most capital (cells: E-com/IT/Security $M)", 480)


def standalone_curves(a: ModelAssumptions, probs: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    for k in COMPANY_KEYS:
        g = np.linspace(0, a.case[k].request, 161)
        p = sum(probs[s] * company_values(a, k, s, g, True)["proceeds"] for s in SCENARIOS)
        net = p - p[0] - g
        fig.add_scatter(x=g, y=net, mode="lines", line=dict(color=COLORS[k], width=2.5), name=NAMES[k],
                        hovertemplate="$%{x:.2f}M → net $%{y:.2f}M<extra>" + NAMES[k] + "</extra>")
    fig.add_hline(y=0, line=dict(color="#64748b", dash="dash"), annotation_text="break-even")
    fig.update_xaxes(title="Investment in this company as management focus ($M)")
    fig.update_yaxes(title="Expected incremental net benefit ($M)")
    return _layout(fig, "Return curves: net benefit if each company were the focus (others get $0)", 400)


# OVERVIEW ----------------------------------------------------------------
def overview_chart(a: ModelAssumptions) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Revenue ($M)", "EBITDA ($M)", "Funding request vs budget ($M)"),
                        horizontal_spacing=0.08)
    names = [NAMES[k] for k in COMPANY_KEYS]
    c = a.case
    fig.add_bar(x=names, y=[c[k].prev_revenue for k in COMPANY_KEYS], name="Previous", marker_color="#cbd5e1", row=1, col=1)
    fig.add_bar(x=names, y=[c[k].latest_revenue for k in COMPANY_KEYS], name="Latest", marker_color="#334155", row=1, col=1)
    fig.add_bar(x=names, y=[c[k].prev_ebitda for k in COMPANY_KEYS], name="Previous", marker_color="#cbd5e1", showlegend=False, row=1, col=2)
    fig.add_bar(x=names, y=[c[k].latest_ebitda for k in COMPANY_KEYS], name="Latest", marker_color="#334155", showlegend=False, row=1, col=2)
    fig.add_bar(x=names, y=[c[k].request for k in COMPANY_KEYS], marker_color=[COLORS[k] for k in COMPANY_KEYS],
                showlegend=False, text=[f"${c[k].request:.0f}M" for k in COMPANY_KEYS], textposition="outside", row=1, col=3)
    fig.add_hline(y=a.budget, line=dict(color="#dc2626", dash="dash"), annotation_text="$15M available", row=1, col=3)
    fig.update_layout(barmode="group")
    _layout(fig, "Case facts: previous vs latest performance and funding requests", 380)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5), margin=dict(b=90))
    fig.update_annotations(font_size=12)
    return fig
