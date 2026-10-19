## Methodology

All monetary values are in **$ millions**. The case figures are illustrative case-company data, not the performance of real companies. Every other number in the simulator is an assumption.

### 1. Facts vs assumptions

| From the case (hardcoded) | Not in the case (assumptions you control) |
|---|---|
| Revenue, EBITDA (previous and latest), margin, current net debt, fund ownership, funding requests | Exit EBITDA with and without investment |
| $15M available vs $33M requested; one intensive programme; 24-month exit | Exit EV/EBITDA multiples, exit net debt |
| E-commerce: 40% storage lost, delivery 2→5 days, liquidity needed within 6 months | How capital turns into improvement (response curves) |
| IT: ~3,000 specialists, utilization 85%→70%, repositioning 12–18 months | Implementation timing, effectiveness, focus effects |
| Security: 95% international sales, 75% manufacturing in Ukraine, expansion 12–18 months | Scenario probabilities, liquidity amount, financing terms |

In **Case facts only** mode the simulator shows no returns: with facts alone every allocation is equally "optimal", because nothing links capital to value.

### 2. Calculation chain (one company, one scenario)

```
f(x)              = programme progress in [0, 1] from the response curve, f(0) = 0
m                 = scenario effectiveness × (focus multiplier or non-focus multiplier)
rE                = share of run-rate EBITDA uplift visible at the valuation month (timing)
rM                = share of risk reduction buyers credit at the valuation month (timing)

Exit EBITDA       = Baseline exit EBITDA + f(x) × Max EBITDA uplift × m × rE
Exit multiple     = Base exit multiple   + f(x) × Max multiple uplift × m × rM
Enterprise value  = Exit EBITDA × Exit multiple
Equity value      = max(Enterprise value − Exit net debt, 0)
Fund proceeds     = Equity value × Fund ownership
```

Portfolio: **Total fund proceeds** = sum over the three companies. **Additional proceeds** = proceeds with investment − proceeds with no investment (same scenario). **Net gain** = additional proceeds − new capital. **Net cash returned** = total proceeds − new capital. **Incremental ROI** = net gain / new capital (shown only when at least $0.25M is invested).

**Expected value** = P(Downside) × Downside value + P(Base) × Base value + P(Upside) × Upside value. Probabilities must sum to 100%; they are judgements, not empirical success rates.

### 3. Investment-response functions

Each shape maps investment *x* to progress *f(x)* between 0% and 100%. *X* = full-effect investment.

| Shape | Formula | Parameters |
|---|---|---|
| Linear | f = min(x / X, 1) | X |
| Diminishing returns | f = (1 − e^(−k·x)) / (1 − e^(−k·X)), x ≤ X | k: higher = faster early gains |
| Threshold | x < T: f = b·x/T; x ≥ T: f = j + (1 − j)·(x − T)/(X − T) | T minimum viable investment, j effect unlocked at T, b effect just below T |
| S-curve | f = (L(x) − L(0)) / (L(X) − L(0)), L(x) = 1/(1 + e^(−s(x − m))) | m midpoint (steepest point), s steepness |
| Custom | piecewise-linear through (0, 0) and your points, flat after the last point | points "investment:% of full effect" |

The presets are labelled illustrative. The case does not establish any threshold or curve shape.

### 4. Two separate sources of value

* **Operating performance** (EBITDA): warehouse recovery, utilization, margin. Enters through *Max EBITDA uplift*.
* **Risk reduction / valuation** (multiple): manufacturing diversification, lower geopolitical exposure, buyer appetite. Enters through *Max multiple uplift*.

**Avoid double counting:** if a buyer would pay more only *because* EBITDA is higher, put the effect in EBITDA, not in the multiple. Use the multiple uplift only for lower risk or a better buyer universe at the same EBITDA. The value bridge shows each effect separately (attribution order: EBITDA → multiple → net debt → ownership; sequential attribution is order-dependent).

Other value sources that are kept separate: **exit net debt** (scenario input and the e-commerce distress penalty) and **ownership** (changes only in the equity financing mode).

### 5. Financing conventions

* **Fixed ownership / non-dilutive support (default, illustrative assumption).** The fund pays for the programme, ownership stays unchanged, and the capital is deducted once when computing net benefit. The money is assumed to be spent on the programme, so it is **not** also added as cash on the exit balance sheet (no double counting). This is a modelling convention, not a claim about the actual instrument.
* **New equity at a pre-money valuation V.** Fund ownership after the round = (o·V + x) / (V + x). A lower V gives the fund more ownership for the same cheque. The cash is again assumed spent, so exit net debt is unchanged.
* **Not modelled:** shareholder loans or preferred instruments. A loan would be repaid before equity (higher fund recovery, lower common equity), and a liquidation preference would change downside proceeds. Results could differ materially under those terms.

### 6. Timing (24-month horizon)

```
Operational readiness = start + implementation + scenario delay − focus time saving
Benefits begin        = readiness + benefit lag
EBITDA realisation rE = clip((valuation month − benefits begin) / ramp-up, 0, 1)
Risk-reduction credit rM = 1 if ready by the valuation month, otherwise
                           partial credit × share of implementation completed
```

Buyers are assumed to price the **run-rate EBITDA** and risk profile verifiable at the *valuation month* (default 21, inside the 24-month deadline). An improvement that is still ramping up is only partly paid for; a plant that is half-built receives only the partial credit you set.

**E-commerce liquidity.** The case gives the 6-month deadline, not the amount. Enter the amount (0 = not specified, no penalty). If e-commerce funding is below the requirement, or starts after the deadline, the assumed distress penalty is applied: exit EBITDA is cut by the haircut % and exit net debt rises by the stated amount. A warning appears in the simulator.

### 7. Management attention

Only one company can be the **primary focus**. Its response is scaled by the focus multiplier (and its implementation can be shortened). The other two use the non-focus multiplier and can absorb at most the non-focus capital cap ("limited or defensive support"). Allocations above the cap are flagged infeasible in the simulator and excluded by the optimizer. All of these values are editable.

### 8. Optimization

Decision variables x₁ (E-commerce), x₂ (IT), x₃ (Security), $M. Constraints: 0 ≤ x₁ ≤ 10, 0 ≤ x₂ ≤ 8, 0 ≤ x₃ ≤ 15, x₁ + x₂ + x₃ ≤ 15, non-focus xᵢ ≤ cap. Unallocated capital is a reserve.

Method: **exhaustive search on a grid** ($0.25M by default) for each focus choice. Thresholds and S-curves make the problem non-smooth and non-concave, so a gradient method could stop at a local optimum; the grid search cannot. Because each company's value depends only on its own investment, the three per-company value vectors are combined by NumPy broadcasting (up to ~80,000 allocations per focus, evaluated in milliseconds). Ties are broken in favour of using less capital. Results are deterministic.

| Objective | Definition | Note |
|---|---|---|
| Expected net cash returned (primary) | E[total proceeds from all three companies] − new capital | Same ranking as expected net gain |
| Incremental ROI | (E[additional proceeds] − capital) / capital | Allocations below the minimum capital are excluded; otherwise ROI can be maximised by investing almost nothing. High ROI ≠ high absolute value. |
| Downside net cash | Downside proceeds − capital | Protects the bad case |
| Robust | max over allocations of min over scenarios of (additional proceeds − capital) | Maximin on incremental benefit |

The optimizer reports the **near-optimal set**: all allocations within a tolerance of the best score. When many allocations score similarly, the optimum is a region, not a point.

### 9. Sensitivity and break-even

* **One-way:** change one assumption over a range and re-optimise at each point; see where the focus or the largest allocation switches.
* **Two-way heatmap:** the company receiving the most capital for each combination of two assumptions.
* **Break-even response scale:** for each company, the factor by which its investment response must be multiplied for its best standalone net benefit to equal the best rival's. This answers "under what conditions does A become a better investment than B or C?".

### 10. Terminology

* **EBITDA** — earnings before interest, taxes, depreciation and amortisation. **EBITDA is not cash flow:** it ignores capex, working capital, taxes and interest. A warehouse rebuild is cash out even if EBITDA recovers.
* **Enterprise value (EV)** — value of the whole business to all capital providers. **Equity value** = EV − net debt. **Fund proceeds** = the fund's share of equity value.
* **EV/EBITDA multiple** — a shortcut valuation. It is not always the best method: distressed firms (e-commerce at 2% margin) may be valued on revenue, assets or a DCF; high-growth firms may command growth premia. The multiples here are inputs, not market data.
* **Net debt** — debt minus cash.
* **ROI** — a ratio; it is not absolute profit.

### 11. Limitations

* Results are **conditional on assumptions**, not statistically validated forecasts. No Monte Carlo simulation is included; the three-scenario model makes every probability explicit instead of drawing arbitrary random numbers.
* No discounting within the 24 months; no taxes, transaction fees, earn-outs or deal structures.
* Exit values assume a sale at the modelled value by month 24; buyer availability in wartime is not modelled beyond your multiple and scenario inputs.
* Each company's value depends only on its own capital (no cross-portfolio synergies); management bandwidth is modelled only through the focus multipliers and cap.
* Common equity is floored at zero; creditors' recoveries are not modelled.
