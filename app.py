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
    INDIA_PM25_24H,
    INDIA_PM25_ANNUAL,
    MAJOR_CITIES,
    SEASON_ORDER,
    WHO_PM25_24H,
    WHO_PM25_ANNUAL,
    city_summary,
    compute_sub_indices,
    diwali_effect,
    dominant_pollutant_by_season,
    health_exceedance,
    health_summary,
    lockdown_effect,
    lockdown_summary,
    monthly_aqi_trend,
    monthly_climatology,
    seasonal_attribution,
    seasonal_decomposition,
    seasonality_index,
    stubble_season_effect,
    validate_computed_aqi,
    yearly_aqi_trend,
)
from src.data_load import load_clean, load_raw
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


@st.cache_data(show_spinner="Loading all cities for comparison…")
def get_all_data() -> pd.DataFrame:
    """Full all-city dataset (for the comparison tab), not NCR-filtered."""
    _apply_kaggle_secrets()
    return load_raw(ensure_city_day())


@st.cache_data(show_spinner="Attributing pollution sources…")
def cached_attribution(city: str, bad_only: bool):
    df_ = get_data()
    return (
        seasonal_attribution(df_, city, bad_only=bad_only),
        dominant_pollutant_by_season(df_, city, bad_only=bad_only),
        validate_computed_aqi(compute_sub_indices(df_, city)),
    )


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

(tab_trend, tab_season, tab_events, tab_forecast,
 tab_sources, tab_compare, tab_health) = st.tabs(
    ["📉 Long-run trend", "🗓️ Seasonality", "🎆 Events", "🔮 Forecast",
     "🧪 Sources", "🏙️ Compare cities", "🫁 Health"]
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
def render_forecast_tab(city: str) -> None:
    """Forecast tab body, factored out so early returns don't halt the whole app."""
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
        return

    try:
        result = cached_backtest(city, horizon)
        fut_mean, fut_ci = cached_future(city, horizon)
    except Exception as exc:
        st.warning(f"Couldn't fit a forecast for {city}: {exc}")
        return

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


with tab_forecast:
    render_forecast_tab(city)


# --------------------------------------------------------------------------- #
# Tab 5 — Sources (pollutant attribution)
# --------------------------------------------------------------------------- #
with tab_sources:
    st.subheader(f"Which pollutant drives {city}'s bad-air days, by season?")
    bad_only = st.toggle("Bad-air days only (computed AQI > 200)", value=True)

    try:
        att, dom, val = cached_attribution(city, bad_only)
    except Exception as exc:
        st.warning(f"Couldn't run source attribution for {city}: {exc}")
        st.stop()

    # Stacked bar: pollutant share per season.
    pollutants = sorted(att["responsible"].unique())
    palette = {
        "PM2.5": "#e53935", "PM10": "#fb8c00", "NO2": "#8e24aa",
        "O3": "#1e88e5", "CO": "#6d4c41", "SO2": "#00897b", "NH3": "#7cb342",
    }
    fig = go.Figure()
    for p in pollutants:
        sub = att[att["responsible"] == p].set_index("season")["share_pct"]
        sub = sub.reindex(SEASON_ORDER)
        fig.add_trace(go.Bar(x=list(SEASON_ORDER), y=sub.values, name=p,
                             marker_color=palette.get(p, "#90a4ae")))
    fig.update_layout(
        height=420, barmode="stack", yaxis_title="% of days pollutant is responsible",
        xaxis_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    c1.metric("Computed-AQI vs dataset AQI", f"r = {val['correlation']:.2f}")
    c2.metric("Days analysed", f"{val['n']:,}")

    dom_map = {r["season"]: (r["responsible"], r["share_pct"]) for _, r in dom.iterrows()}
    w = dom_map.get("Winter")
    s = dom_map.get("Summer")
    parts = []
    if w:
        parts.append(f"**winter** is {w[0]} ({w[1]:.0f}%)")
    if s:
        parts.append(f"**summer** is {s[0]} ({s[1]:.0f}%)")
    scope = "worst" if bad_only else "all"
    insight(
        f"The pollutant responsible for {city}'s {scope} days "
        + (" and ".join(parts) if parts else "varies by season")
        + ". Delhi's burden is overwhelmingly **particulate** — fine PM2.5 in the "
        "cold months, coarser PM10 as summer/monsoon dust rises — rather than "
        "gaseous pollutants. (Validated: recomputed CPCB AQI tracks the dataset's "
        f"AQI at r={val['correlation']:.2f}.)"
    )
    st.caption(
        "Attribution recomputes each day's CPCB sub-index per pollutant from raw "
        "concentrations; the responsible pollutant is the one with the highest "
        "sub-index. Note: O₃ is understated here because city_day.csv provides "
        "daily-mean O₃, whereas CPCB's ozone sub-index uses the 8-hour maximum."
    )


# --------------------------------------------------------------------------- #
# Tab 6 — Compare cities
# --------------------------------------------------------------------------- #
with tab_compare:
    st.subheader("Is Delhi uniquely bad, or part of a regional pattern?")

    df_all = get_all_data()
    available = sorted(df_all["City"].unique())
    default = [c for c in MAJOR_CITIES if c in available]
    picked = st.multiselect("Cities to compare", available, default=default)
    if len(picked) < 2:
        st.info("Pick at least two cities to compare.")
        st.stop()

    picked_t = tuple(picked)
    summary = city_summary(df_all, picked_t, "AQI")
    seas = seasonality_index(df_all, picked_t, "PM2.5")

    # Mean AQI per city, Delhi highlighted.
    colors = ["#e53935" if c == "Delhi" else "#90a4ae" for c in summary["city"]]
    fig = go.Figure(go.Bar(
        x=summary["city"], y=summary["mean"], marker_color=colors,
        text=summary["mean"].round(0), textposition="outside",
        customdata=summary["pct_bad"].round(0),
        hovertemplate="%{x}: mean AQI %{y:.0f}<br>%{customdata:.0f}% days Poor+<extra></extra>",
    ))
    fig.update_layout(height=380, yaxis_title="Mean AQI (all years)", xaxis_title=None, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    # Annual cycle (PM2.5) per city — shows whether the winter spike is shared.
    fig2 = go.Figure()
    for c in picked:
        clim = monthly_climatology(df_all, c, "PM2.5")
        fig2.add_trace(go.Scatter(
            x=clim["month"], y=clim["mean"], mode="lines+markers", name=c,
            line=dict(width=3 if c == "Delhi" else 1.5),
        ))
    fig2.update_layout(
        height=400, yaxis_title="Mean PM2.5", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title="Annual PM2.5 cycle by city", margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig2, width="stretch")

    if "Delhi" in picked:
        delhi_mean = summary.loc[summary["city"] == "Delhi", "mean"].iloc[0]
        delhi_bad = summary.loc[summary["city"] == "Delhi", "pct_bad"].iloc[0]
        others = summary.loc[summary["city"] != "Delhi", "mean"]
        ratio = delhi_mean / others.mean() if not others.empty else float("nan")
        most_seasonal = seas.iloc[0]
        delhi_seas = seas.loc[seas["City"] == "Delhi", "winter_monsoon_ratio"]
        delhi_seas_v = delhi_seas.iloc[0] if not delhi_seas.empty else float("nan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Delhi mean AQI", f"{delhi_mean:.0f}", f"{ratio:.1f}× others")
        c2.metric("Delhi days Poor+", f"{delhi_bad:.0f}%")
        c3.metric("Delhi winter/monsoon PM2.5", f"{delhi_seas_v:.1f}×")
        insight(
            f"Delhi is uniquely **polluted** — mean AQI **{delhi_mean:.0f}**, about "
            f"**{ratio:.1f}×** the other selected cities, with **{delhi_bad:.0f}%** of days "
            "Poor-or-worse. But the winter spike is a **regional North-India pattern**, "
            f"not Delhi's alone: its winter PM2.5 is **{delhi_seas_v:.1f}×** its monsoon "
            f"level, comparable to other Gangetic-plain cities (most seasonal here: "
            f"{most_seasonal['City']} {most_seasonal['winter_monsoon_ratio']:.1f}×). Coastal "
            "and southern cities stay far cleaner and flatter year-round."
        )
    st.caption(
        "Means use observed days only; cities differ in coverage (e.g. Mumbai/Kolkata "
        "have fewer AQI days). Ahmedabad is excluded from the default set — its series "
        "looks anomalously high — but can be added manually."
    )


# --------------------------------------------------------------------------- #
# Tab 7 — Health impact
# --------------------------------------------------------------------------- #
with tab_health:
    st.subheader(f"What is {city}'s air doing to the people who breathe it?")

    hs = health_summary(df, city, "PM2.5")
    he = health_exceedance(df, city, "PM2.5")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean PM2.5", f"{hs['mean_pm25']:.0f} µg/m³")
    c2.metric("× WHO annual limit", f"{hs['x_who_annual']:.0f}×")
    c3.metric("Days over WHO 24h", f"{hs['pct_over_who']:.0f}%")
    c4.metric("Cigarettes / day", f"{hs['avg_cigarettes_day']:.1f}")

    # Per-year breakdown of day categories vs the 24h limits.
    he = he.assign(
        moderate=he["days_over_who"] - he["days_over_india"],  # WHO < x ≤ India
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=he["year"], y=he["clean_days"], name=f"Clean (≤{WHO_PM25_24H:.0f})", marker_color="#66bb6a"))
    fig.add_trace(go.Bar(x=he["year"], y=he["moderate"], name=f"Over WHO (≤{INDIA_PM25_24H:.0f})", marker_color="#ffca28"))
    fig.add_trace(go.Bar(x=he["year"], y=he["days_over_india"], name=f"Over India ({INDIA_PM25_24H:.0f}+)", marker_color="#ef5350"))
    fig.update_layout(
        height=380, barmode="stack", yaxis_title="Days in year", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    # Mean PM2.5 per year with guideline reference lines.
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=he["year"], y=he["mean_pm25"], marker_color="#5c6bc0",
                          text=he["mean_pm25"].round(0), textposition="outside", name="Annual mean PM2.5"))
    fig2.add_hline(y=INDIA_PM25_ANNUAL, line_dash="dash", line_color="#ef5350",
                   annotation_text=f"India annual ({INDIA_PM25_ANNUAL:.0f})", annotation_position="top left")
    fig2.add_hline(y=WHO_PM25_ANNUAL, line_dash="dot", line_color="#2e7d32",
                   annotation_text=f"WHO annual ({WHO_PM25_ANNUAL:.0f})", annotation_position="bottom left")
    fig2.update_layout(height=340, yaxis_title="Annual mean PM2.5 (µg/m³)", xaxis_title=None, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, width="stretch")

    insight(
        f"{city}'s air averages **{hs['mean_pm25']:.0f} µg/m³** PM2.5 — "
        f"**{hs['x_who_annual']:.0f}×** the WHO annual guideline and "
        f"**{hs['x_india_annual']:.1f}×** India's. About **{hs['pct_over_who']:.0f}%** of "
        f"days breach the WHO 24-hour limit and **{hs['pct_over_india']:.0f}%** breach "
        f"India's; only **{hs['pct_clean']:.0f}%** of days are 'clean'. In cigarette terms "
        f"that's roughly **{hs['avg_cigarettes_day']:.1f} cigarettes a day** — about "
        f"**{hs['cigarettes_year']:.0f} a year** — from breathing alone."
    )
    st.caption(
        f"Limits: WHO 2021 PM2.5 — 24h {WHO_PM25_24H:.0f}, annual {WHO_PM25_ANNUAL:.0f} µg/m³; "
        f"India NAAQS — 24h {INDIA_PM25_24H:.0f}, annual {INDIA_PM25_ANNUAL:.0f} µg/m³. "
        "Cigarette equivalence uses Berkeley Earth's ≈22 µg/m³·day ≈ 1 cigarette; it is a "
        "communication heuristic, not a clinical dose. 2020 is a partial year."
    )
