# 🌫️ Delhi-NCR Air Quality Analytics

**🔗 Live app: https://aqi-analytics-in.streamlit.app**

An analytics project on Delhi-NCR air quality that goes **beyond basic EDA**. It
quantifies *why* and *when* Delhi's air gets bad — long-run trends, the seasonal
cycle, the impact of specific events (Diwali, stubble burning, the COVID
lockdown), and a short-horizon AQI forecast — and surfaces a plain-English
insight for every view, not just a chart.

Built on **real CPCB data, 2015–2026**, stitched from three sources (each
validated where they overlap): hourly CPCB stations rebuilt to daily city values
(2015–2023, r≈0.97 vs the original Kaggle `city_day`), the official CPCB daily
**AQI bulletin** (2023–2025), and the **OpenAQ** API (2025→today). AQI is
continuous to the present; pollutant *concentrations* (PM2.5 µg/m³ etc.) have a
2023–2025 gap where no public concentration source exists (handled explicitly in
the UI). Streamlit dashboard with analysis/forecast logic kept as pure functions.

## What it shows

| Section | Question it answers | Headline finding (Delhi) |
|---|---|---|
| **Long-run trend** | Is the air getting better? | Mean AQI eased from **~254 (2015)** to **~200 (2025)**, with a sharp **2020 lockdown dip (190)**, a **post-COVID rebound**, then a plateau around 200 — visible now the series runs to 2026. |
| **Seasonality** | When is it worst? | PM2.5 peaks in **November (~238)** and bottoms in **August (~43)** — a ~5× swing, confirmed by seasonal decomposition. |
| **Events** | What drives bad-air days? | **Stubble burning** lifts Oct–Nov PM2.5 to **1.8× the rest of the year**; **Diwali** adds **+85%** over the prior fortnight; the **2020 lockdown** cut Apr–May PM2.5 by **~50%**. |
| **Forecast** | Can we predict AQI? | A **SARIMA(1,1,2)(0,1,1,7)** model beats naive baselines on a 30-day holdout (MAE 24.7 vs 29.6). |
| **Sources** | Which pollutant drives bad days? | Recomputed CPCB sub-indices (validated at **r=0.93** vs the dataset's AQI) show Delhi's bad days are **overwhelmingly particulate** — PM2.5 in winter (**86%**), PM10 rising with summer/monsoon dust. |
| **Compare cities** | Is Delhi uniquely bad? | Delhi's mean AQI (**259**) is **~1.8×** other major cities and **65%** of its days are Poor+ — but the winter spike is a **shared North-India pattern** (Patna, Lucknow, Kolkata similar), while coastal/southern cities stay clean and flat. |
| **Health** | What is it doing to people? | Delhi's PM2.5 averages **117 µg/m³ — 23× the WHO annual guideline**; **~100%** of days breach the WHO 24h limit and **~0%** are clean, equivalent to **~5 cigarettes/day (~1,900/year)** from breathing. |

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
src/analysis.py      trends, decomposition, events, source attribution,
                     multi-city comparison, health impact (pure)
src/forecast.py      SARIMA backtest + forecast (pure, isolated)
src/download_data.py Kaggle fetch (2015-2020 fallback) / runtime bootstrap
src/build_city_day.py  rebuild 2015-2023 daily data from hourly CPCB stations
src/fetch_openaq.py    fetch recent (2025+) daily concentrations from OpenAQ
src/extend_dataset.py  splice stations + CPCB bulletin + OpenAQ -> 2015-2026 file
app.py               Streamlit UI only; imports from src/
data/                raw data: hourly stations, bulletin, OpenAQ (git-ignored)
data_processed/      committed derived files: city_day_2015_2026.csv (app loads
                     this) + the 2015_2023 stations-only base
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows ( source .venv/bin/activate on macOS/Linux )
pip install -r requirements.txt

# The extended 2015-2023 dataset is committed at data_processed/, so the app
# runs out of the box. To (re)build it from raw hourly CPCB stations, or to
# fetch the original 2015-2020 Kaggle file, you need a Kaggle token (see below):
python -m src.build_city_day     # rebuild extended file (downloads ~1 GB raw)
python -m src.download_data      # or just the 2015-2020 Kaggle city_day.csv

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

1. Push this repo to GitHub.
2. On <https://share.streamlit.io>, create an app pointing at `app.py`.

That's it — the extended dataset (`data_processed/city_day_2015_2023.csv`) is
committed, so the app loads it directly with **no Kaggle secret required**. (If
that file is ever removed, the app falls back to downloading the 2015-2020
`city_day.csv` from Kaggle, which would then need a `KAGGLE_API_TOKEN` secret.)

## Data

The app loads `data_processed/city_day_2015_2026.csv`, spliced from three real
sources (priority for AQI: stations → bulletin → OpenAQ):

1. **2015–2023 concentrations + AQI** — rebuilt from raw hourly CPCB stations
   ([Time Series Air Quality Data of India 2010–2023](https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023))
   via `src/build_city_day.py`; AQI = max CPCB sub-index (8h-max O₃/CO).
   Validated **r≈0.97** vs the original [rohanrao city_day](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
2. **2023–2025 AQI** — official [CPCB daily AQI bulletin](https://www.kaggle.com/datasets/saikiranudayana/india-air-quality-index-aqi-dataset-20232025)
   (AQI + prominent pollutant only). Validated **r≈0.97** on the 2022–23 overlap.
3. **2025→today concentrations + AQI** — the [OpenAQ](https://openaq.org) v3 API
   (`src/fetch_openaq.py`); PM2.5/PM10/NO2/O3 in µg/m³ (CO/SO₂ dropped — OpenAQ's
   India gas units are inconsistent).

**Known gap:** pollutant *concentrations* are missing 2023-04 → 2024-12 (no public
source); **AQI is continuous** throughout. Source: India's CPCB.

## Future work

Deliberately out of scope for now, but natural extensions:

- **Weather correlation** — wind/temperature vs dispersion (needs an external
  weather source; not in this dataset).
- **8-hour-max sub-indices** — recompute O₃/CO attribution from the hourly file
  so ozone isn't understated (the Sources tab uses daily means).
- **Model comparison** — SARIMA vs Prophet vs an ML baseline.
- **Policy-impact analysis** — odd-even, GRAP interventions.
- **Model comparison** — SARIMA vs Prophet vs ML.
- **Policy-impact analysis** — odd-even, GRAP interventions.
- **Health-impact layer** — days exceeding WHO limits.

---
*Analysis/forecast logic is separated from the UI as pure functions; expensive
computations are cached with `@st.cache_data`.*
