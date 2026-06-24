# Delhi-NCR Air Quality Analytics

## What this is
An analytics project on Delhi-NCR air quality that goes beyond basic EDA:
long-run trends, seasonal decomposition, event analysis (Diwali, stubble-burning
season, COVID lockdown), a short-horizon AQI forecast, pollutant source
attribution, and a multi-city comparison. Each view surfaces a plain-English
insight, not just a chart. Goal: quantify *why* and *when* Delhi's air gets bad,
not just show that it does.

## Dataset
Kaggle "Air Quality Data in India (2015-2024)" (or the 2015-2020 version):
city_day.csv with columns like Date, City, PM2.5, PM10, NO, NO2, NOx, NH3, CO,
SO2, O3, Benzene, Toluene, AQI, AQI_Bucket. REAL data — never replace with
synthetic. Filter to Delhi and NCR cities for the core analysis; keep other
major cities available for the comparison phase. Raw data lives in data/ and is
git-ignored (large); never commit it.

## Tech stack
- Python 3.11+, pandas, Plotly, Streamlit
- statsmodels for seasonal decomposition + SARIMA forecast (or Prophet)
- Deploy: Streamlit Community Cloud

## Structure
- data/ — raw data (git-ignored)
- src/data_load.py — load + clean + filter
- src/analysis.py — pure functions: trends, seasonal decomposition, event
  comparisons, source attribution, city comparison. No Streamlit imports here.
- src/forecast.py — the forecasting model, isolated
- app.py — Streamlit UI only; imports from src/
- notebooks/ — optional EDA scratch

## Phased roadmap
Build and verify each phase before starting the next. Commit at the end of each.
- Phase 0-1: setup + trustworthy data layer; long-run trend + seasonal
  decomposition of PM2.5/AQI.
- Phase 2: event analysis — quantify Diwali week, the Oct-Nov stubble-burning
  season, and the 2020 COVID lockdown vs baseline, with actual numbers.
- Phase 3: short-horizon AQI forecast (SARIMA or Prophet) with error metrics.
- Phase 4: Streamlit dashboard tying it together, plain-English insights,
  README, deploy. THIS IS THE FINISH LINE for the core project — get it fully
  working and deployed before starting stretch phases.
- Phase 5 (stretch): pollutant source attribution — which pollutant drives bad
  days in each season (e.g. PM2.5 in winter vs O3 in summer).
- Phase 6 (stretch): multi-city comparison — Delhi vs Mumbai, Kolkata,
  Bengaluru, etc. Is Delhi uniquely bad, or seasonally bad?

## Future work (README only, not building now)
Weather correlation (wind/temp vs dispersion), model comparison (SARIMA vs
Prophet vs ML), policy-impact analysis (odd-even, GRAP), health-impact layer
(days exceeding WHO limits). Listing these signals product thinking; do not
start them.

## Key constraints
- Correctness first: sanity-check totals, date range, and per-pollutant null
  rates before trusting any aggregate.
- Missing data is significant here: AQI/pollutant columns have real gaps. Handle
  them explicitly (report null rates, choose interpolation vs drop deliberately,
  document the choice) — never silently fill.
- Dates: parse the Date column to datetime and validate the range up front.
- Every dashboard section must show at least one plain-English insight derived
  from the data.
- Keep analysis/forecast logic as pure functions separate from the UI; cache
  expensive computations with @st.cache_data.
- This dataset is widely used — prioritize differentiated analysis (event
  quantification, decomposition, forecasting, source attribution) over generic
  EDA plots.
- Do not start stretch phases (5-6) until the Phase 4 core is deployed and
  working. Avoid scope creep.

## Conventions
- Type hints on functions. Small, single-purpose functions.
- Commit at logical checkpoints with clear messages.

## Workflow notes
- If a feature's requirements are unclear, ask me clarifying questions first.
- When I correct a mistake, add a rule to this file so it doesn't repeat.
