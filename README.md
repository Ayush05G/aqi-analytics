# 🌫️ Delhi-NCR Air Quality Analytics

An analytics project on Delhi-NCR air quality that goes **beyond basic EDA**. It
quantifies *why* and *when* Delhi's air gets bad — long-run trends, the seasonal
cycle, the impact of specific events (Diwali, stubble burning, the COVID
lockdown), and a short-horizon AQI forecast — and surfaces a plain-English
insight for every view, not just a chart.

Built on **real CPCB data** (2015–2020) via Kaggle. Streamlit dashboard with
analysis/forecast logic kept as pure, tested functions.

## What it shows

| Section | Question it answers | Headline finding (Delhi) |
|---|---|---|
| **Long-run trend** | Is the air getting better? | Mean AQI fell **~22%** from 2015 (297) to 2019 (232) — a genuine multi-year decline. |
| **Seasonality** | When is it worst? | PM2.5 peaks in **November (~238)** and bottoms in **August (~43)** — a ~5× swing, confirmed by seasonal decomposition. |
| **Events** | What drives bad-air days? | **Stubble burning** lifts Oct–Nov PM2.5 to **1.8× the rest of the year**; **Diwali** adds **+85%** over the prior fortnight; the **2020 lockdown** cut Apr–May PM2.5 by **~50%**. |
| **Forecast** | Can we predict AQI? | A **SARIMA(1,1,2)(0,1,1,7)** model beats naive baselines on a 30-day holdout (MAE 24.7 vs 29.6). |

## Methodology notes (why the numbers are trustworthy)

- **Missing data is handled explicitly, never silently filled.** Aggregates use
  observed days only and report day counts; the forecast interpolates only short
  daily gaps (≤7 days) and reports how many.
- **Event baselines control for confounders.** Diwali is compared to the *prior
  fortnight* (isolating fireworks from the stubble haze already building); the
  stubble season is compared *within the same year* (controlling for the
  downward trend); the lockdown is compared to the *same calendar months* in
  2015–2019 (controlling for seasonality).
- **The forecast is validated, not asserted.** It's backtested on a held-out
  window and benchmarked against persistence and seasonal-naive baselines.

## Project structure

```
src/data_load.py     load + clean + filter to Delhi-NCR (pure)
src/analysis.py      trends, seasonal decomposition, event analysis (pure)
src/forecast.py      SARIMA backtest + forecast (pure, isolated)
src/download_data.py Kaggle fetch / runtime bootstrap
app.py               Streamlit UI only; imports from src/
data/                raw data (git-ignored, never committed)
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows ( source .venv/bin/activate on macOS/Linux )
pip install -r requirements.txt

# Fetch the dataset (needs a Kaggle API token in ~/.kaggle/; see below)
python -m src.download_data

# Sanity-check the data + analysis from the CLI
python -m src.data_load
python -m src.analysis
python -m src.forecast

# Launch the dashboard
streamlit run app.py
```

### Kaggle credentials
Create a token at <https://www.kaggle.com/settings/api>. Save the newer `KGAT_`
token to `~/.kaggle/access_token`, or place the classic `kaggle.json` in
`~/.kaggle/`.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (the dataset stays git-ignored).
2. On <https://share.streamlit.io>, create an app pointing at `app.py`.
3. In **App settings → Secrets**, add your Kaggle credentials so the app can
   fetch `city_day.csv` on first load:
   ```toml
   KAGGLE_USERNAME = "your_username"
   KAGGLE_KEY = "your_key"
   ```
   (Use a classic `kaggle.json` username/key pair here.) The app downloads only
   `city_day.csv` (~2.6 MB), not the full archive.

## Data

Kaggle — [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india),
`city_day.csv`. Source: India's Central Pollution Control Board (CPCB).

## Future work

Deliberately out of scope for now, but natural extensions:

- **Pollutant source attribution** — which pollutant drives bad days per season
  (PM2.5 in winter vs O₃ in summer).
- **Multi-city comparison** — Delhi vs Mumbai, Kolkata, Bengaluru: is Delhi
  uniquely bad, or seasonally bad?
- **Weather correlation** — wind/temperature vs dispersion.
- **Model comparison** — SARIMA vs Prophet vs ML.
- **Policy-impact analysis** — odd-even, GRAP interventions.
- **Health-impact layer** — days exceeding WHO limits.

---
*Analysis/forecast logic is separated from the UI as pure functions; expensive
computations are cached with `@st.cache_data`.*
