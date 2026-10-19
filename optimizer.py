"""
Global optimiser by exhaustive grid search.

Decision variables  x1 (E-commerce), x2 (IT), x3 (Security), $M
Constraints         0 <= xi <= request_i ; non-focus xi <= non-focus cap
                    x1 + x2 + x3 <= budget   (unallocated capital = reserve)

Why exhaustive search: response curves may have thresholds (discontinuities)
and S-shapes (non-concave), so gradient methods can stop at local optima.
With three companies and a $0.25M grid there are at most 41 x 33 x 61 points
per focus choice. Because fund proceeds are additive across companies (each
company's value depends only on its own investment, the focus choice and the
scenario), we evaluate each company once per grid value and combine them by
NumPy broadcasting. The search is exact on the grid and fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from assumptions import COMPANY_KEYS, SCENARIOS, ModelAssumptions
from financial_model import EPS, company_values, validate_assumptions

OBJECTIVES = {
    "expected_net": "Max expected net cash returned (proceeds − new capital)",
    "roi": "Max incremental ROI on new capital",
    "downside_net": "Max downside-case net cash returned",
    "robust": "Robust: max worst-scenario incremental net benefit",
}

MIN_ROI_INVESTMENT_DEFAULT = 1.0


@dataclass
class Candidate:
    allocation: dict[str, float]
    focus: str
    score: float
    expected_net: float           # E[proceeds] - invested
    expected_net_gain: float      # E[proceeds - baseline] - invested
    downside_net: float
    worst_gain: float
    invested: float


@dataclass
class OptimizationResult:
    objective: str
    best: Candidate
    near_optimal: list[Candidate]
    n_near_optimal: int
    tolerance: float
    per_focus_best: dict[str, Candidate]
    grid_step: float
    n_evaluated: int
    notes: list[str] = field(default_factory=list)


def company_grid(a: ModelAssumptions, key: str, focus: str, step: float) -> np.ndarray:
    cap = min(a.case[key].request, a.budget)
    if key != focus:
        cap = min(cap, a.focus.nonfocus_cap)
    n = int(np.floor(cap / step + EPS))
    g = np.round(np.arange(n + 1) * step, 6)
    return g


def _focus_tensors(a: ModelAssumptions, focus: str, step: float):
    """Return grids and per-scenario proceeds tensors for one focus choice."""
    grids = [company_grid(a, k, focus, step) for k in COMPANY_KEYS]
    P = {}
    B = {}
    for s in SCENARIOS:
        v = [company_values(a, k, s, g, k == focus)["proceeds"] for k, g in zip(COMPANY_KEYS, grids)]
        # Broadcast additively: P[i,j,k] = v1[i] + v2[j] + v3[k]
        P[s] = v[0][:, None, None] + v[1][None, :, None] + v[2][None, None, :]
        B[s] = float(sum(vi[0] for vi in v))     # grid starts at 0 -> baseline
    inv = grids[0][:, None, None] + grids[1][None, :, None] + grids[2][None, None, :]
    return grids, P, B, inv


def _score(objective: str, P, B, inv, probs, min_roi_inv: float):
    EP = sum(probs[s] * P[s] for s in SCENARIOS)
    EB = sum(probs[s] * B[s] for s in SCENARIOS)
    exp_net = EP - inv
    exp_gain = EP - EB - inv
    down_net = P["downside"] - inv
    worst_gain = np.minimum.reduce([P[s] - B[s] - inv for s in SCENARIOS])
    if objective == "expected_net":
        score = exp_net
    elif objective == "roi":
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(inv >= min_roi_inv - EPS, exp_gain / np.maximum(inv, EPS), -np.inf)
    elif objective == "downside_net":
        score = down_net
    elif objective == "robust":
        score = worst_gain
    else:
        raise ValueError(f"Unknown objective '{objective}'")
    return score, exp_net, exp_gain, down_net, worst_gain


def optimize(a: ModelAssumptions, objective: str = "expected_net", focus: str | None = None,
             step: float = 0.25, tolerance: float | None = None,
             min_roi_investment: float = MIN_ROI_INVESTMENT_DEFAULT,
             max_list: int = 12) -> OptimizationResult:
    """Exhaustive search. focus=None -> also optimise over the focus company."""
    probs = validate_assumptions(a)
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    focuses = list(COMPANY_KEYS) if focus is None else [focus]

    all_cands: list[tuple[float, float, int, Candidate]] = []
    per_focus: dict[str, Candidate] = {}
    best_overall: Candidate | None = None
    n_eval = 0
    stash = []
    for fi, fc in enumerate(focuses):
        grids, P, B, inv = _focus_tensors(a, fc, step)
        feasible = inv <= a.budget + EPS
        score, exp_net, exp_gain, down_net, worst = _score(objective, P, B, inv, probs, min_roi_investment)
        score = np.where(feasible, score, -np.inf)
        n_eval += int(feasible.sum())
        if not np.isfinite(score).any():
            continue
        stash.append((fc, grids, score, exp_net, exp_gain, down_net, worst, inv))
        best_val = float(np.max(score))
        # Tie-break: among (numerically) equal scores prefer the smallest capital use.
        ties = np.argwhere(score >= best_val - 1e-9)
        tie_inv = inv[tuple(ties.T)]
        idx = tuple(ties[int(np.argmin(tie_inv))])
        cand = _make_cand(fc, grids, idx, score, exp_net, exp_gain, down_net, worst, inv)
        per_focus[fc] = cand
        if best_overall is None or cand.score > best_overall.score + 1e-9 or (
                abs(cand.score - best_overall.score) <= 1e-9 and cand.invested < best_overall.invested - EPS):
            best_overall = cand

    if best_overall is None:
        raise ValueError("No feasible allocation satisfies the objective constraints "
                         "(for ROI, the minimum investment may be too high).")

    # Near-optimal set: everything within tolerance of the best score.
    if tolerance is None:
        # Default: within 1% of the value created by the best allocation (min $0.05M);
        # for ROI, within 0.5 percentage points.
        tolerance = (max(0.05, 0.01 * abs(best_overall.expected_net_gain))
                     if objective != "roi" else 0.005)
    n_near = 0
    for fc, grids, score, exp_net, exp_gain, down_net, worst, inv in stash:
        mask = score >= best_overall.score - tolerance
        n_near += int(mask.sum())
        idxs = np.argwhere(mask)
        # keep the best few per focus to bound list size
        order = np.argsort(-score[tuple(idxs.T)], kind="stable")[: max_list * 3]
        for o in order:
            idx = tuple(idxs[o])
            all_cands.append((-float(score[idx]), float(inv[idx]), len(all_cands),
                              _make_cand(fc, grids, idx, score, exp_net, exp_gain, down_net, worst, inv)))
    all_cands.sort(key=lambda t: (t[0], t[1], t[2]))
    near = [c for *_, c in all_cands[:max_list]]

    notes = []
    if objective == "roi":
        notes.append(
            f"ROI = (incremental expected proceeds − new capital) / new capital. Allocations below "
            f"${min_roi_investment:.2f}M are excluded: as capital approaches zero the ratio becomes "
            "unstable and can be maximised by investing almost nothing. A high ROI on a small cheque "
            "can still create less absolute value than a lower ROI on a large one."
        )
    if n_near > 1:
        notes.append(f"{n_near:,} grid allocations score within {tolerance:.3g} of the best. "
                     "Treat the optimum as a region, not a single point.")
    return OptimizationResult(objective, best_overall, near, n_near, tolerance, per_focus, step, n_eval, notes)


def _make_cand(fc, grids, idx, score, exp_net, exp_gain, down_net, worst, inv) -> Candidate:
    alloc = {k: float(grids[n][idx[n]]) for n, k in enumerate(COMPANY_KEYS)}
    return Candidate(alloc, fc, float(score[idx]), float(exp_net[idx]), float(exp_gain[idx]),
                     float(down_net[idx]), float(worst[idx]), float(inv[idx]))


# ---------------------------------------------------------------------------
# Marginal analysis (used to explain the optimum)
# ---------------------------------------------------------------------------
def marginal_values(a: ModelAssumptions, allocation: dict[str, float], focus: str,
                    step: float = 0.25) -> dict[str, dict[str, float | None]]:
    """Expected change in fund proceeds from adding / removing one grid step
    at each company, holding the others fixed."""
    probs = validate_assumptions(a)
    out = {}
    for k in COMPANY_KEYS:
        x = allocation[k]
        pts = np.array([max(x - step, 0.0), x, x + step])
        ev = sum(probs[s] * company_values(a, k, s, pts, k == focus)["proceeds"] for s in SCENARIOS)
        up_cap = min(a.case[k].request, a.budget if k == focus else min(a.budget, a.focus.nonfocus_cap))
        out[k] = {
            "next_step": float(ev[2] - ev[1]) if x + step <= up_cap + EPS else None,
            "last_step": float(ev[1] - ev[0]) if x > EPS else None,
        }
    return out


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
def _scale_company(a, key, attr_path, factor=None, add=None):
    b = a.copy()
    ca = b.companies[key]
    if attr_path == "exit_multiple":
        for s in SCENARIOS:
            ca.scenarios[s].exit_multiple *= factor
    elif attr_path == "baseline_ebitda":
        for s in SCENARIOS:
            ca.scenarios[s].baseline_ebitda *= factor
    elif attr_path == "max_ebitda_uplift":
        ca.response.max_ebitda_uplift *= factor
    elif attr_path == "max_multiple_uplift":
        ca.response.max_multiple_uplift *= factor
    elif attr_path == "response_magnitude":
        ca.response.max_ebitda_uplift *= factor
        ca.response.max_multiple_uplift *= factor
    elif attr_path == "effectiveness":
        for s in SCENARIOS:
            ca.scenarios[s].effectiveness *= factor
    elif attr_path == "cost":
        # Programme cost: the same effect needs `factor` times more capital.
        r = ca.response
        r.full_effect_investment *= factor
        r.threshold *= factor
        r.midpoint *= factor
        r.curvature /= factor
        r.custom_points = [(px * factor, pf) for px, pf in r.custom_points]
    elif attr_path == "delay":
        for s in SCENARIOS:
            ca.scenarios[s].delay_months += add
    elif attr_path == "net_debt":
        for s in SCENARIOS:
            ca.scenarios[s].exit_net_debt += add
    else:
        raise ValueError(attr_path)
    return b


SENS_PARAMS: dict[str, dict] = {}
for _k, _n in (("ecom", "E-commerce"), ("it", "IT Engineering"), ("sec", "Security Hardware")):
    SENS_PARAMS[f"{_k}:exit_multiple"] = dict(label=f"{_n} — exit multiple (× scale)", kind="scale", key=_k, attr="exit_multiple", rng=(0.6, 1.4))
    SENS_PARAMS[f"{_k}:response_magnitude"] = dict(label=f"{_n} — investment response size (× scale)", kind="scale", key=_k, attr="response_magnitude", rng=(0.0, 2.0))
    SENS_PARAMS[f"{_k}:effectiveness"] = dict(label=f"{_n} — implementation effectiveness (× scale)", kind="scale", key=_k, attr="effectiveness", rng=(0.4, 1.6))
    SENS_PARAMS[f"{_k}:cost"] = dict(label=f"{_n} — programme cost / thresholds (× scale)", kind="scale", key=_k, attr="cost", rng=(0.5, 1.5))
    SENS_PARAMS[f"{_k}:delay"] = dict(label=f"{_n} — execution delay (+ months)", kind="add", key=_k, attr="delay", rng=(0.0, 9.0))
    SENS_PARAMS[f"{_k}:net_debt"] = dict(label=f"{_n} — exit net debt (+ $M)", kind="add", key=_k, attr="net_debt", rng=(-20.0, 20.0))
SENS_PARAMS["prob:downside"] = dict(label="Downside probability (taken from base case, %)", kind="prob", rng=(0.0, 60.0))
SENS_PARAMS["global:liquidity"] = dict(label="E-commerce liquidity requirement ($M)", kind="liquidity", rng=(0.0, 10.0))


def apply_sensitivity(a: ModelAssumptions, param: str, value: float) -> ModelAssumptions:
    spec = SENS_PARAMS[param]
    if spec["kind"] == "scale":
        return _scale_company(a, spec["key"], spec["attr"], factor=value)
    if spec["kind"] == "add":
        return _scale_company(a, spec["key"], spec["attr"], add=value)
    b = a.copy()
    if spec["kind"] == "prob":
        # Move probability between downside and base, keep upside fixed.
        up = b.probabilities["upside"]
        value = min(value, 100.0 - up)
        b.probabilities = {"downside": value, "base": 100.0 - up - value, "upside": up}
        return b
    if spec["kind"] == "liquidity":
        b.liquidity.requirement = value
        return b
    raise ValueError(param)


def sweep_values(param: str, n: int = 11, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    r = SENS_PARAMS[param]["rng"]
    lo = r[0] if lo is None else lo
    hi = r[1] if hi is None else hi
    return np.round(np.linspace(lo, hi, n), 4)


def one_way_sensitivity(a: ModelAssumptions, param: str, values, objective: str = "expected_net",
                        focus: str | None = None, step: float = 0.25) -> list[dict]:
    rows = []
    for v in values:
        b = apply_sensitivity(a, param, float(v))
        r = optimize(b, objective, focus=focus, step=step, max_list=1)
        best = r.best
        rows.append({
            "value": float(v), "focus": best.focus, **{k: best.allocation[k] for k in COMPANY_KEYS},
            "reserve": a.budget - best.invested, "score": best.score,
            "expected_net_gain": best.expected_net_gain, "downside_net": best.downside_net,
        })
    return rows


def dominant_company(row: dict) -> str:
    """Company receiving the largest allocation ('none' if nothing invested)."""
    vals = {k: row[k] for k in COMPANY_KEYS}
    k = max(vals, key=lambda kk: vals[kk])
    return k if vals[k] > EPS else "none"


def two_way_sensitivity(a: ModelAssumptions, p1: str, v1, p2: str, v2,
                        objective: str = "expected_net", focus: str | None = None,
                        step: float = 0.5) -> list[list[dict]]:
    grid = []
    for y in v2:
        row = []
        for x in v1:
            b = apply_sensitivity(apply_sensitivity(a, p1, float(x)), p2, float(y))
            r = optimize(b, objective, focus=focus, step=step, max_list=1)
            row.append({"x": float(x), "y": float(y), "focus": r.best.focus,
                        **r.best.allocation, "score": r.best.score})
        grid.append(row)
    return grid


# ---------------------------------------------------------------------------
# Break-even: when does company A become the best single focus?
# ---------------------------------------------------------------------------
def standalone_best(a: ModelAssumptions, key: str, step: float = 0.25) -> tuple[float, float]:
    """Best expected incremental net benefit if ALL capital options are limited to
    this one company (as focus). Returns (best net gain, investment)."""
    probs = validate_assumptions(a)
    g = company_grid(a, key, key, step)
    ev = sum(probs[s] * company_values(a, key, s, g, True)["proceeds"] for s in SCENARIOS)
    gain = ev - ev[0] - g
    i = int(np.argmax(gain))
    return float(gain[i]), float(g[i])


def breakeven_response_scale(a: ModelAssumptions, key: str, step: float = 0.25,
                             hi: float = 10.0, tol: float = 0.01) -> float | None:
    """Smallest scale factor on `key`'s response magnitude (EBITDA and multiple
    uplift) at which its standalone net benefit matches the best rival's.
    None if not reached within [0, hi]. Returns <1 if it already leads."""
    rivals = [standalone_best(a, k, step)[0] for k in COMPANY_KEYS if k != key]
    target = max(rivals + [0.0])

    def gap(fac):
        return standalone_best(_scale_company(a, key, "response_magnitude", factor=fac), key, step)[0] - target

    if gap(hi) < 0:
        return None
    lo_, hi_ = 0.0, hi
    if gap(0.0) >= 0:
        return 0.0
    while hi_ - lo_ > tol:
        mid = 0.5 * (lo_ + hi_)
        if gap(mid) >= 0:
            hi_ = mid
        else:
            lo_ = mid
    return hi_
