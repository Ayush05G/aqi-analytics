"""Delhi-NCR Air Quality Analytics — Streamlit dashboard (UI only).

All analysis/forecast logic lives in src/; this file just wires it to widgets,
caches the expensive bits, and surfaces a plain-English insight per section.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    diwali_effect,
    lockdown_effect,
    lockdown_summary,
    monthly_aqi_trend,
    monthly_climatology,
    seasonal_decomposition,
    stubble_season_effect,
    yearly_aqi_trend,
)
from src.data_load import load_clean
from src.download_data import configure_credentials, ensure_city_day
from src.forecast import backtest, forecast_future, naive_persistence, naive_seasonal

st.set_page_config(page_title="Delhi-NCR Air Quality Analytics", page_icon="🌫️", layout="wide")

# AQI category colour bands (CPCB) for chart context.
AQI_BANDS = [
    (0, 50, "Good", "#9ccc65"),
    (50, 100, "Satisfactory", "#d4e157"),
    (100, 200, "Moderate", "#ffee58"),
    (200, 300, "Poor", "#ffa726"),
    (300, 400, "Very Poor", "#ef5350"),
    (400, 600, "Severe", "#b71c1c"),
]


# --------------------------------------------------------------------------- #
# Data + cached computations
# --------------------------------------------------------------------------- #
def _apply_kaggle_secrets() -> None:
    """Push Kaggle creds from st.secrets into env vars / token file, with logging.

    Accepts the token as a top-level KAGGLE_API_TOKEN, a classic username/key
    pair, or a nested [kaggle] section (api_token / token / key+username).
    """
    try:
        secrets = st.secrets
    except Exception as exc:
        print(f"[app] no st.secrets available (local dev?): {exc}", flush=True)
        return

    try:
        top_keys = list(secrets.keys())
    except Exception as exc:
        print(f"[app] could not read st.secrets keys: {exc}", flush=True)
        return
    print(f"[app] st.secrets top-level keys: {top_keys}", flush=True)

    if "KAGGLE_API_TOKEN" in secrets:
        configure_credentials(api_token=str(secrets["KAGGLE_API_TOKEN"]))
    elif "KAGGLE_USERNAME" in secrets and "KAGGLE_KEY" in secrets:
        configure_credentials(str(secrets["KAGGLE_USERNAME"]), str(secrets["KAGGLE_KEY"]))
    elif "kaggle" in secrets:
        sec = secrets["kaggle"]
        configure_credentials(
            username=sec.get("username"),
            key=sec.get("key"),
            api_token=sec.get("api_token") or sec.get("token") or sec.get("KAGGLE_API_TOKEN"),
        )
    else:
        print("[app] WARNING: no recognized Kaggle secret found in st.secrets.", flush=True)

    print(
        "[app] kaggle creds applied: "
        f"api_token={'set' if os.environ.get('KAGGLE_API_TOKEN') else 'unset'}, "
        f"username={'set' if os.environ.get('KAGGLE_USERNAME') else 'unset'}",
        flush=True,
    )


@st.cache_data(show_spinner="Loading air-quality data…")
def get_data() -> pd.DataFrame:
    """Ensure city_day.csv exists (download on cloud) then load + clean it."""
    print("[app] get_data: applying secrets + ensuring dataset…", flush=True)
    _apply_kaggle_secrets()
    path = ensure_city_day()
    df = load_clean(path)
    print(f"[app] get_data: loaded {len(df):,} rows for {sorted(df['City'].unique())}", flush=True)
    return df


@st.cache_data(show_spinner="Decomposing the seasonal cycle…")
def cached_decomposition(city: str, col: str):
    return seasonal_decomposition(get_data(), city, col)


@st.cache_data(show_spinner="Fitting the SARIMA forecast…")
def cached_backtest(city: str, horizon: int):
    return backtest(get_data(), city, "AQI", horizon=horizon)


@st.cache_data(show_spinner="Projecting the forecast forward…")
def cached_future(city: str, horizon: int):
    return forecast_future(get_data(), city, "AQI", horizon=horizon)


def insight(text: str) -> None:
    """Render a plain-English, data-derived takeaway."""
    st.info(f"**Insight —** {text}")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
try:
    df = get_data()
except Exception as exc:  # pragma: no cover - surfaced in the UI
    st.error(
        "Couldn't load the dataset. Locally, run `python -m src.download_data`. "
        "On Streamlit Cloud, add your Kaggle `KAGGLE_USERNAME` and `KAGGLE_KEY` "
        f"to the app secrets.\n\nDetails: {exc}"
    )
    st.stop()

cities = sorted(df["City"].unique())

st.title("🌫️ Delhi-NCR Air Quality Analytics")
st.caption(
    "Beyond basic EDA: long-run trends, the seasonal cycle, event impacts "
    "(Diwali, stubble burning, COVID lockdown), and a short-horizon AQI forecast — "
    "each with a plain-English takeaway. Data: CPCB via Kaggle, 2015–2020."
)

with st.sidebar:
    st.header("Controls")
    city = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
    span = df.loc[df["City"] == city, "Date"]
    n_rows = int((df["City"] == city).sum())
    st.metric("Days of data", f"{n_rows:,}")
    st.caption(f"{span.min().date()} → {span.max().date()}")
    st.caption(
        "Missing data is preserved, not silently filled. Aggregates use observed "
        "days only; the forecast interpolates a handful of short daily gaps."
    )

tab_trend, tab_season, tab_events, tab_forecast = st.tabs(
    ["📉 Long-run trend", "🗓️ Seasonality", "🎆 Events", "🔮 Forecast"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Long-run trend
# --------------------------------------------------------------------------- #
with tab_trend:
    st.subheader(f"How has {city}'s air changed, 2015–2020?")
    monthly = monthly_aqi_trend(df, city, "AQI")
    yearly = yearly_aqi_trend(df, city, "AQI")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"], y=monthly["mean"], mode="lines",
            name="Monthly mean AQI", line=dict(color="#5c6bc0", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"], y=monthly["mean"].rolling(12, min_periods=6).mean(),
            mode="lines", name="12-month rolling mean",
            line=dict(color="#e53935", width=3),
        )
    )
    fig.update_layout(
        height=420, yaxis_title="AQI", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    full_years = yearly[yearly["n_days"] >= 350]
    if len(full_years) >= 2:
        first, last = full_years.iloc[0], full_years.iloc[-1]
        delta = last["mean"] - first["mean"]
        pct = 100 * delta / first["mean"]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{int(first['year'])} mean AQI", f"{first['mean']:.0f}")
        c2.metric(f"{int(last['year'])} mean AQI", f"{last['mean']:.0f}", f"{pct:+.0f}%")
        c3.metric("Worst year", f"{int(yearly.loc[yearly['mean'].idxmax(), 'year'])}")
        direction = "improved" if delta < 0 else "worsened"
        insight(
            f"{city}'s mean AQI {direction} from **{first['mean']:.0f}** in "
            f"{int(first['year'])} to **{last['mean']:.0f}** in {int(last['year'])} "
            f"(**{pct:+.0f}%** across full-data years). The rolling line shows the "
            "drop is a genuine multi-year trend, not a one-off year."
        )
    st.caption("Partial years (e.g. 2020 ends mid-year) are excluded from the headline comparison.")


# --------------------------------------------------------------------------- #
# Tab 2 — Seasonality
# --------------------------------------------------------------------------- #
with tab_season:
    st.subheader(f"When in the year is {city}'s air worst?")
    pollutant = st.selectbox("Pollutant", ["PM2.5", "AQI", "PM10", "NO2", "O3"], index=0)
    clim = monthly_climatology(df, city, pollutant)

    fig = go.Figure(
        go.Bar(
            x=clim["month"], y=clim["mean"],
            marker_color=clim["mean"], marker_colorscale="YlOrRd",
            text=clim["mean"].round(0), textposition="outside",
        )
    )
    fig.update_layout(
        height=420, yaxis_title=f"Mean {pollutant}", xaxis_title=None, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, width="stretch")

    peak = clim.loc[clim["mean"].idxmax()]
    trough = clim.loc[clim["mean"].idxmin()]
    ratio = peak["mean"] / trough["mean"]
    insight(
        f"{city}'s {pollutant} peaks in **{peak['month']} ({peak['mean']:.0f})** and "
        f"bottoms out in **{trough['month']} ({trough['mean']:.0f})** — a "
        f"**{ratio:.1f}×** swing. Winter traps pollutants under cool, stagnant air "
        "while the summer monsoon washes them out."
    )

    if pollutant in ("PM2.5", "AQI", "PM10"):
        try:
            dec = cached_decomposition(city, pollutant)
            sc = dec.seasonal_cycle
            fig2 = go.Figure(
                go.Bar(
                    x=sc["month"], y=sc["effect"],
                    marker_color=["#ef5350" if v > 0 else "#42a5f5" for v in sc["effect"]],
                )
            )
            fig2.update_layout(
                height=320, yaxis_title=f"Seasonal effect on {pollutant}",
                xaxis_title=None, margin=dict(t=30, b=10),
                title="Isolated seasonal component (additive decomposition)",
            )
            st.plotly_chart(fig2, width="stretch")
            st.caption(
                f"Decomposition on {len(dec.observed)} monthly points "
                f"({dec.n_interpolated} interpolated). The seasonal component repeats "
                "each year; red months push pollution up, blue months pull it down."
            )
        except ValueError as exc:
            st.warning(f"Not enough clean data to decompose {pollutant} for {city}: {exc}")


# --------------------------------------------------------------------------- #
# Tab 3 — Events
# --------------------------------------------------------------------------- #
with tab_events:
    st.subheader(f"What actually drives {city}'s bad-air days?")

    st.markdown("#### 🎆 Diwali fireworks")
    diw = diwali_effect(df, city, "PM2.5")
    if not diw.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=diw["year"], y=diw["baseline_mean"], name="Prior fortnight", marker_color="#90a4ae"))
        fig.add_trace(go.Bar(x=diw["year"], y=diw["festival_mean"], name="Diwali window", marker_color="#ef5350"))
        fig.update_layout(height=320, barmode="group", yaxis_title="Mean PM2.5", xaxis_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        avg_pct = diw["pct_change"].mean()
        worst = diw.loc[diw["pct_change"].idxmax()]
        insight(
            f"Across {len(diw)} Diwalis, PM2.5 in the festival window averaged "
            f"**{avg_pct:+.0f}%** vs the two weeks before — worst in "
            f"**{int(worst['year'])} ({worst['pct_change']:+.0f}%)**. Comparing to the "
            "pre-Diwali fortnight isolates fireworks from the stubble haze already building."
        )
    else:
        st.warning(f"No Diwali windows with both festival + baseline data for {city}.")

    st.markdown("#### 🔥 Stubble-burning season (Oct–Nov)")
    stub = stubble_season_effect(df, city, "PM2.5")
    if not stub.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stub["year"], y=stub["rest_mean"], name="Rest of year", marker_color="#90a4ae"))
        fig.add_trace(go.Bar(x=stub["year"], y=stub["season_mean"], name="Oct–Nov", marker_color="#ff7043"))
        fig.update_layout(height=320, barmode="group", yaxis_title="Mean PM2.5", xaxis_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        insight(
            f"Every year, Oct–Nov PM2.5 runs **{stub['multiplier'].mean():.1f}×** the "
            f"rest of the year (**{stub['pct_change'].mean():+.0f}%** on average). This "
            "post-monsoon stubble-burning window, not Diwali, is the dominant driver of "
            f"{city}'s worst stretch."
        )

    st.markdown("#### 🦠 COVID-19 lockdown (Apr–May 2020)")
    lock = lockdown_effect(df, city, "PM2.5")
    if (lock["is_lockdown"]).any() and (~lock["is_lockdown"]).any():
        summ = lockdown_summary(lock)
        colors = ["#ef5350" if flag else "#90a4ae" for flag in lock["is_lockdown"]]
        fig = go.Figure(go.Bar(x=lock["year"], y=lock["mean"], marker_color=colors,
                               text=lock["mean"].round(0), textposition="outside"))
        fig.update_layout(height=320, yaxis_title="Apr–May mean PM2.5", xaxis_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        insight(
            f"During the strict Apr–May 2020 lockdown, {city}'s PM2.5 fell to "
            f"**{summ['lockdown_mean']:.0f}** vs a {summ['n_baseline_years']}-year "
            f"baseline of **{summ['baseline_mean']:.0f}** for the same months — a "
            f"**{summ['pct_change']:+.0f}%** drop. A rough ceiling on what halting "
            "traffic and industry alone can achieve."
        )
    else:
        st.warning(f"Not enough Apr–May coverage to compare lockdown vs baseline for {city}.")


# --------------------------------------------------------------------------- #
# Tab 4 — Forecast
# --------------------------------------------------------------------------- #
with tab_forecast:
    st.subheader(f"Can we forecast {city}'s AQI a few weeks out?")
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14, step=1)

    # Fitting SARIMA is the heaviest step (a small order search + two refits) and
    # would otherwise run on every page load. Gate it behind a button so the other
    # tabs stay fast and the cloud instance isn't pinned on each rerun.
    if st.button("▶ Run forecast", type="primary"):
        st.session_state["run_forecast"] = True

    if not st.session_state.get("run_forecast"):
        st.info("Click **Run forecast** to fit the SARIMA model and backtest it "
                "(~30–90s on the cloud; cached afterwards).")
        st.stop()

    try:
        result = cached_backtest(city, horizon)
        fut_mean, fut_ci = cached_future(city, horizon)
    except Exception as exc:
        st.warning(f"Couldn't fit a forecast for {city}: {exc}")
        st.stop()

    recent_train = result.train.iloc[-90:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_train.index, y=recent_train.values, name="History", line=dict(color="#90a4ae")))
    fig.add_trace(go.Scatter(x=result.test.index, y=result.test.values, name="Actual (held out)", line=dict(color="#1e88e5", width=2)))
    fig.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast.values, name="SARIMA forecast", line=dict(color="#e53935", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(result.conf_int.index) + list(result.conf_int.index[::-1]),
        y=list(result.conf_int.iloc[:, 1]) + list(result.conf_int.iloc[:, 0][::-1]),
        fill="toself", fillcolor="rgba(229,57,53,0.15)", line=dict(width=0),
        name="95% interval", hoverinfo="skip",
    ))
    fig.update_layout(height=420, yaxis_title="AQI", xaxis_title=None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    m = result.metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("SARIMA MAE", f"{m['sarima']['mae']:.1f}")
    c2.metric("vs persistence", f"{m['persistence']['mae']:.1f}", f"{m['sarima']['mae'] - m['persistence']['mae']:+.1f}", delta_color="inverse")
    c3.metric("vs seasonal-naive", f"{m['seasonal_naive']['mae']:.1f}", f"{m['sarima']['mae'] - m['seasonal_naive']['mae']:+.1f}", delta_color="inverse")

    metrics_table = pd.DataFrame(m).T[["mae", "rmse", "mape"]].round(1)
    metrics_table.columns = ["MAE", "RMSE", "MAPE %"]
    st.dataframe(metrics_table, width="stretch")

    best_naive = min(m["persistence"]["mae"], m["seasonal_naive"]["mae"])
    beats = m["sarima"]["mae"] < best_naive
    insight(
        f"On a {horizon}-day holdout ({result.test.index.min().date()} → "
        f"{result.test.index.max().date()}), **SARIMA{result.order}×"
        f"{result.seasonal_order}** scored MAE **{m['sarima']['mae']:.1f}** "
        f"(MAPE {m['sarima']['mape']:.0f}%), which "
        f"{'**beats**' if beats else 'does **not** beat'} the best naive baseline "
        f"({best_naive:.1f}). Daily AQI is persistent, so even a few weeks out is "
        "tractable — though the annual cycle limits longer horizons."
    )
    st.caption(
        f"Forward projection ({fut_mean.index.min().date()} → {fut_mean.index.max().date()}) "
        "is shown for context; with data ending mid-2020 it cannot be validated."
    )
