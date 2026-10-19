---
title: PE Portfolio Optimization Simulator
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Capital allocation simulator for Case Challenge 3.0
---

# Private Equity Portfolio Optimization Simulator

An interactive financial simulator built on the **SKELAR × Horizon Capital × Mastercard Case Challenge 3.0** case: a PE fund with **$15M** of capital, **$33M** of funding requests, three portfolio companies (E-commerce, IT Engineering, Security Hardware), capacity for **one** intensive value-creation programme and a **24-month** exit deadline.

You move capital between the companies and immediately see projected EBITDA, exit enterprise value, equity value and fund proceeds. You can inspect every step of the calculation, and a global optimizer finds the best allocation **under your assumptions**.

> The case figures are illustrative and do not represent real companies. Every other number is an assumption that you can edit, labelled *ILLUSTRATIVE ASSUMPTION — NOT PROVIDED BY THE CASE*. Results are conditional on assumptions, not forecasts.

## Features

* **Two model states:** *Case facts only* (no projections, because none can be derived) and *Illustrative simulation* (a complete, editable example parameter set).
* **Allocation sliders** with budget enforcement: a slider that would push the total above $15M is capped at the remaining budget, the other sliders are never changed, and unallocated capital is shown as a reserve.
* **Five investment-response shapes:** linear, diminishing returns, threshold, S-curve and custom piecewise. All parameters are adjustable and plotted.
* **Transparent valuation chain:** investment → progress → EBITDA and multiple → EV → equity → fund proceeds → net benefit, with the no-investment counterfactual alongside.
* **Separate value sources:** operating EBITDA vs risk-reduction multiple vs net debt vs ownership, shown in a value bridge.
* **Timing model:** start, duration, benefit lag, ramp-up, valuation month, plus an e-commerce liquidity test (the amount is entered by the user, since the case gives only the 6-month deadline).
* **Management attention constraint:** exactly one focus company; editable focus and non-focus multipliers and a cap on non-focus capital.
* **Financing conventions:** fixed ownership with non-dilutive support (default), or new equity at an adjustable pre-money valuation.
* **Scenarios:** downside, base and upside with user probabilities that must sum to 100%.
* **Optimizer:** exhaustive $0.25M grid search, optionally over the focus choice. Objectives: expected net cash, ROI, downside, robust (maximin). It also reports near-optimal alternatives and explains its choice.
* **Sensitivity:** one-way re-optimisation sweeps, two-way heatmaps and break-even response scales ("when does A beat B or C?").

## Project structure

```
app.py                 Gradio UI (tabs, events, explanations)
financial_model.py     Response functions, timing, valuation chain, breakdown
optimizer.py           Grid optimizer, objectives, sensitivity, break-even
assumptions.py         Case facts (hardcoded) + illustrative assumption preset
charts.py              Plotly figures
METHODOLOGY.md         Formulas, definitions, limitations (shown in tab 6)
requirements.txt
tests/                 pytest suite (financial model, optimizer, UI logic)
```

## Run locally

```bash
pip install -r requirements.txt
python app.py              # open http://127.0.0.1:7860
python -m pytest -q        # run the tests
```

## Deploy to Hugging Face Spaces

1. Sign in at huggingface.co, then go to **New → Space**.
2. Enter a name, choose a licence, and select **Gradio** as the SDK. The free CPU hardware is enough.
3. Upload all project files, keeping the folder structure (`app.py`, the other `.py` files, `METHODOLOGY.md`, `requirements.txt`, `README.md`, `tests/`). You can drag and drop them in **Files → Add file → Upload files**, or use git:
   ```bash
   git clone https://huggingface.co/spaces/<user>/<space-name>
   cp -r portfolio-simulator/* <space-name>/
   cd <space-name> && git add . && git commit -m "Add simulator" && git push
   ```
4. Spaces reads the YAML header of this README (`sdk: gradio`, `app_file: app.py`), installs `requirements.txt` automatically and starts the app. The build log is under **Logs**.
5. When the status shows **Running**, the simulator is live at `https://huggingface.co/spaces/<user>/<space-name>`.

If the Space reports that `sdk_version` is unavailable, change it in the header to a Gradio version listed in the Space settings. The app uses standard Gradio components that work on Gradio 5 and 6.

No API keys, external data or model inference are needed; every calculation runs locally.

## How to use

1. **Portfolio overview:** the case facts and constraints.
2. **Capital allocation simulator:** move the sliders and choose the primary focus. The KPI cards, tables and charts update immediately. Open *Inspect the calculation* to see the step-by-step maths, the value bridge and **what changed since your previous allocation**.
3. **Financial assumptions:** edit the scenario table (baseline exit EBITDA, multiple, net debt, effectiveness, delay), the probabilities, each company's response curve and timing, the focus effects, the financing convention and the liquidity test. *Reset illustrative assumptions* restores the example.
4. **Optimization:** choose an objective and click **FIND OPTIMAL ALLOCATION**. Read the reasons and the near-optimal alternatives, then *Apply* the result to the sliders.
5. **Risk & sensitivity:** find the assumptions that flip the recommendation.
6. **Methodology:** every formula and limitation.

### Interpreting results

* **Fund proceeds** are the fund's share of exit equity, not company value.
* **Net gain** = additional proceeds created by the intervention − new capital. This is the economic value of the decision.
* **ROI** is a ratio. A small cheque can have a high ROI and still create little value.
* The optimum is optimal **only under the current assumptions**. Use sensitivity analysis before drawing conclusions.
