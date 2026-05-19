"""
dashboard.py
============
Interactive Streamlit dashboard for Finland Flood Analytics.
Works locally AND on Streamlit Cloud (auto-fetches data if missing).

Run:
    streamlit run src/dashboard.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Page config — must come before any other st.* call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Finland Flood Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-fetch data if missing (handles Streamlit Cloud cold start)
# ---------------------------------------------------------------------------
def _data_ready() -> bool:
    return (RAW / "water_levels.csv").exists() and (PROC / "flood_events.csv").exists()

if not _data_ready():
    st.info("⏳ First run — fetching and analysing Finnish flood data (~60 sec)...")
    bar = st.progress(0, text="Running fetch_data.py ...")
    try:
        subprocess.run([sys.executable, str(SRC / "fetch_data.py")],
                       check=True, capture_output=True, text=True)
        bar.progress(50, text="Running analysis.py ...")
        subprocess.run([sys.executable, str(SRC / "analysis.py")],
                       check=True, capture_output=True, text=True)
        bar.progress(100, text="Done!")
        st.success("✅ Data ready — loading dashboard...")
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"Setup failed:\n{e.stderr}")
        st.stop()

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.kpi {
    background: linear-gradient(135deg,#003580,#1a5bb0);
    color:white; padding:1rem 1.4rem;
    border-radius:10px; margin-bottom:0.5rem;
}
.kpi h2{margin:0;font-size:2rem;}
.kpi p {margin:0;opacity:.85;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABELS = {
    "kemijoki_rovaniemi":   "Kemijoki / Rovaniemi",
    "kemijoki_isohaara":    "Kemijoki / Isohaara",
    "kokemaenjoki_pori":    "Kokemäenjoki / Pori",
    "kokemaenjoki_tampere": "Nokianvirta / Nokia",
    "kymijoki_kouvola":     "Kymijoki / Kouvola",
    "vuoksi_imatra":        "Vuoksi / Imatra",
    "oulujoki_oulu":        "Oulujoki / Oulu",
    "aurajoki_turku":       "Aurajoki / Turku",
}
RISK_CLR = {"very_high":"#C0392B","high":"#E67E22","medium":"#003580","low":"#4A90D9"}

def lbl(s): return LABELS.get(s, s)

# ---------------------------------------------------------------------------
# Load data — no cache so it always reads fresh files
# ---------------------------------------------------------------------------
def load_all() -> dict:
    def rd(name, dates=None):
        p = (RAW if name in ("water_levels","discharge","precipitation",
                              "flood_risk_areas","flood_return_levels",
                              "stations_metadata") else PROC)
        path = p / f"{name}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=dates or False)

    return {
        "wl":       rd("water_levels",       ["datetime"]),
        "q":        rd("discharge",           ["datetime"]),
        "precip":   rd("precipitation",       ["date"]),
        "events":   rd("flood_events",        ["start_date","end_date"]),
        "trends":   rd("trends"),
        "seasonal": rd("seasonal_stats"),
        "rp":       rd("return_periods"),
        "risk":     rd("flood_risk_areas"),
        "rl":       rd("flood_return_levels"),
        "stations": rd("stations_metadata"),
    }

d = load_all()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌊 Finland Flood Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📈 Water Levels",
    "🌧️ Precipitation & Discharge",
    "🔴 Flood Events",
    "📅 Seasonal Analysis",
    "📉 Return Periods",
    "🗺️ Risk Map",
    "📋 Trend Analysis",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Open Data Sources (CC BY 4.0)**")
st.sidebar.markdown("- [SYKE Hydrology API](https://www.syke.fi/en-US/Open_information)")
st.sidebar.markdown("- [FMI Open Data](https://en.ilmatieteenlaitos.fi/open-data)")
st.sidebar.markdown("- [SYKE Flood Risk Areas](https://luontotieto.syke.fi)")
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 Data Status"):
    for k, df in d.items():
        st.caption(f"{'✅' if not df.empty else '❌'} {k}: {len(df):,} rows")

# ===========================================================================
# PAGE 1 — Overview
# ===========================================================================
if page == "📊 Overview":
    st.title("🌊 Finland Flood Analytics")
    st.markdown("Real Finnish hydrology data from **SYKE** and **FMI** — CC BY 4.0.")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{len(d['wl']):,}",                    "Water Level Observations"),
        (c2, str(d['wl']['station'].nunique() if not d['wl'].empty else 0), "Monitoring Stations"),
        (c3, str(len(d['events'])),                  "Flood Events Detected"),
        (c4, str(len(d['risk'])),                    "Official Risk Areas"),
    ]:
        col.markdown(f'<div class="kpi"><h2>{val}</h2><p>{label}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("📍 Monitoring Stations")
        if not d["stations"].empty:
            st.dataframe(
                d["stations"][["station_key","river","city","lat","lon"]].rename(
                    columns={"station_key":"Station","river":"River",
                             "city":"City","lat":"Lat","lon":"Lon"}),
                use_container_width=True, height=300)

    with right:
        st.subheader("🔴 Top 10 Flood Events")
        if not d["events"].empty:
            top = d["events"].nlargest(10,"peak_level_m")[
                ["station","start_date","duration_days","peak_level_m","excess_m"]].copy()
            top["station"] = top["station"].map(LABELS).fillna(top["station"])
            top.columns = ["Station","Start","Duration (d)","Peak (m)","Excess (m)"]
            st.dataframe(top, use_container_width=True, height=300)

    if not d["risk"].empty:
        st.subheader("Risk Level Distribution — 22 EU Flood Risk Areas")
        cnt = d["risk"]["risk"].value_counts().reset_index()
        cnt.columns = ["Risk","Count"]
        fig = px.pie(cnt, names="Risk", values="Count",
                     color="Risk", color_discrete_map=RISK_CLR, hole=0.4)
        fig.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2 — Water Levels
# ===========================================================================
elif page == "📈 Water Levels":
    st.title("📈 Water Level Time Series")

    if d["wl"].empty:
        st.error("No water level data — check Data Status in sidebar.")
        st.stop()

    stations = sorted(d["wl"]["station"].unique())
    labels   = [lbl(s) for s in stations]
    sel_lbl  = st.selectbox("Station", labels)
    sel_st   = stations[labels.index(sel_lbl)]

    wl = d["wl"][d["wl"]["station"] == sel_st].sort_values("datetime").copy()

    ctrl, chart = st.columns([1,3])
    with ctrl:
        show_roll  = st.checkbox("30-day rolling mean", True)
        show_p95   = st.checkbox("P95 flood threshold", True)
        dr = st.date_input("Date Range",
                           value=(wl["datetime"].min().date(),
                                  wl["datetime"].max().date()))

    if len(dr) == 2:
        wl = wl[(wl["datetime"].dt.date >= dr[0]) &
                (wl["datetime"].dt.date <= dr[1])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl["datetime"], y=wl["water_level_m"],
                              mode="lines", name="Water Level",
                              line=dict(color="#4A90D9",width=1),
                              fill="tozeroy", fillcolor="rgba(74,144,217,0.08)"))
    if show_roll and len(wl) > 30:
        roll = wl.set_index("datetime")["water_level_m"].rolling("30D").mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values,
                                  mode="lines", name="30-day mean",
                                  line=dict(color="#003580",width=2.5)))
    if show_p95:
        p95 = wl["water_level_m"].quantile(0.95)
        fig.add_hline(y=p95, line_dash="dash", line_color="#C0392B",
                      annotation_text=f"P95 = {p95:.2f} m")

    fig.update_layout(title=f"Water Level — {sel_lbl}",
                      xaxis_title="Date", yaxis_title="Water Level (m)",
                      height=430, template="plotly_white",
                      legend=dict(orientation="h",y=1.05))
    with chart:
        st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Mean", f"{wl['water_level_m'].mean():.3f} m")
    c2.metric("Max",  f"{wl['water_level_m'].max():.3f} m")
    c3.metric("Min",  f"{wl['water_level_m'].min():.3f} m")
    c4.metric("Std",  f"{wl['water_level_m'].std():.3f} m")
    c5.metric("P95",  f"{wl['water_level_m'].quantile(0.95):.3f} m")


# ===========================================================================
# PAGE 3 — Precipitation & Discharge
# ===========================================================================
elif page == "🌧️ Precipitation & Discharge":
    st.title("🌧️ Precipitation & River Discharge")
    t1, t2 = st.tabs(["River Discharge", "Precipitation Climatology"])

    with t1:
        if d["q"].empty:
            st.error("No discharge data.")
        else:
            stations = sorted(d["q"]["station"].unique())
            sel = st.selectbox("Station", [lbl(s) for s in stations])
            st_key = stations[[lbl(s) for s in stations].index(sel)]
            q = d["q"][d["q"]["station"] == st_key].sort_values("datetime")
            fig = px.area(q, x="datetime", y="discharge_m3s",
                          title=f"River Discharge — {sel}",
                          labels={"discharge_m3s":"Discharge (m³/s)","datetime":"Date"},
                          color_discrete_sequence=["#003580"])
            fig.update_layout(height=410, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Mean", f"{q['discharge_m3s'].mean():.1f} m³/s")
            c2.metric("Peak", f"{q['discharge_m3s'].max():.1f} m³/s")
            c3.metric("Min",  f"{q['discharge_m3s'].min():.1f} m³/s")

    with t2:
        if d["precip"].empty:
            st.error("No precipitation data.")
        else:
            city = st.selectbox("City", sorted(d["precip"]["place"].unique()))
            p = d["precip"][d["precip"]["place"] == city].copy()
            p["month"] = p["date"].dt.month
            mn = p.groupby("month").agg(
                mean_precip=("precipitation_mm","mean"),
                mean_snow=("snow_depth_cm","mean"),
                mean_temp=("temperature_c","mean"),
            ).reset_index()
            mlabels = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(x=mn["month"],y=mn["mean_precip"],
                                  name="Precipitation (mm/day)",
                                  marker_color="#4A90D9"), secondary_y=False)
            fig.add_trace(go.Scatter(x=mn["month"],y=mn["mean_snow"],
                                      mode="lines+markers",name="Snow (cm)",
                                      line=dict(color="#85C1E9",width=2)), secondary_y=True)
            fig.add_trace(go.Scatter(x=mn["month"],y=mn["mean_temp"],
                                      mode="lines+markers",name="Temp (°C)",
                                      line=dict(color="#C0392B",dash="dash",width=2)),
                           secondary_y=True)
            fig.update_layout(title=f"Monthly Climatology — {city}",
                              xaxis=dict(tickvals=list(range(1,13)),ticktext=mlabels),
                              height=430, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 4 — Flood Events
# ===========================================================================
elif page == "🔴 Flood Events":
    st.title("🔴 Detected Flood Events")

    if d["events"].empty:
        st.error("No events — run analysis.py first.")
        st.stop()

    ev = d["events"].copy()
    ev["station_label"] = ev["station"].map(LABELS).fillna(ev["station"])
    ev["start_date"]    = pd.to_datetime(ev["start_date"])
    ev["end_date"]      = pd.to_datetime(ev["end_date"])

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Events",   len(ev))
    c2.metric("Avg Duration",   f"{ev['duration_days'].mean():.1f} days")
    c3.metric("Longest Event",  f"{ev['duration_days'].max()} days")

    st.subheader("Events per Station")
    cnt = ev.groupby("station_label").size().reset_index(name="Events")
    fig = px.bar(cnt.sort_values("Events"), x="Events", y="station_label",
                  orientation="h", color="Events", color_continuous_scale="Blues",
                  title="Flood Events per Station (P95 threshold, min 2 days)")
    fig.update_layout(height=350, template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Event Timeline")
    fig2 = px.timeline(ev, x_start="start_date", x_end="end_date",
                        y="station_label", color="peak_level_m",
                        color_continuous_scale="Reds",
                        hover_data=["duration_days","excess_m"],
                        labels={"peak_level_m":"Peak (m)"})
    fig2.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All Events")
    show = ev[["station_label","start_date","end_date",
               "duration_days","peak_level_m","excess_m"]].copy()
    show.columns = ["Station","Start","End","Duration (d)","Peak (m)","Excess (m)"]
    st.dataframe(show, use_container_width=True)


# ===========================================================================
# PAGE 5 — Seasonal Analysis
# ===========================================================================
elif page == "📅 Seasonal Analysis":
    st.title("📅 Seasonal Flood Patterns")

    if d["seasonal"].empty:
        st.error("No seasonal data — run analysis.py first.")
        st.stop()

    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    df = d["seasonal"].copy()
    df["month_name"] = pd.Categorical(df["month_name"],
                                       categories=month_order, ordered=True)
    stations = sorted(df["station"].unique())
    labels   = [lbl(s) for s in stations]
    sel      = st.multiselect("Stations", labels, default=labels[:4])
    if not sel:
        st.warning("Select at least one station.")
        st.stop()
    sel_st = [stations[labels.index(l)] for l in sel]
    filt   = df[df["station"].isin(sel_st)]

    fig = go.Figure()
    for station in sel_st:
        g = filt[filt["station"] == station].sort_values("month")
        fig.add_trace(go.Scatter(
            x=g["month_name"], y=g["mean_level_m"],
            mode="lines+markers", name=lbl(station),
            error_y=dict(type="data", array=g["std_level_m"].values, visible=True),
        ))
    fig.update_layout(title="Monthly Mean Water Level (±1σ)",
                      xaxis_title="Month", yaxis_title="Water Level (m)",
                      height=440, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 Finnish rivers show two flood peaks per year — "
            "spring (April–May, *kevättulva*) from snowmelt, "
            "and autumn (October) from rainfall.")


# ===========================================================================
# PAGE 6 — Return Periods
# ===========================================================================
elif page == "📉 Return Periods":
    st.title("📉 Flood Frequency Analysis")
    t1, t2 = st.tabs(["Gumbel Fitted Curves", "Official SYKE Hazard Levels"])

    with t1:
        if d["rp"].empty:
            st.error("No return period data — run analysis.py first.")
        else:
            stations = sorted(d["rp"]["station"].unique())
            labels   = [lbl(s) for s in stations]
            sel      = st.multiselect("Stations", labels, default=labels[:3])
            sel_st   = [stations[labels.index(l)] for l in sel]
            fig = go.Figure()
            for station in sel_st:
                g = d["rp"][d["rp"]["station"]==station].sort_values("return_period_years")
                fig.add_trace(go.Scatter(x=g["return_period_years"],
                                          y=g["estimated_level_m"],
                                          mode="lines+markers", name=lbl(station)))
            fig.add_vline(x=100, line_dash="dash", line_color="#C0392B",
                           annotation_text="1/100a design event")
            fig.update_xaxes(type="log",
                              tickvals=[2,5,10,50,100,500,1000],
                              ticktext=["2","5","10","50","100","500","1000"])
            fig.update_layout(title="Gumbel EV-I Flood Frequency",
                              xaxis_title="Return Period (years)",
                              yaxis_title="Estimated Water Level (m)",
                              height=450, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Gumbel EV-I fitted to annual spring flood maxima.")

    with t2:
        if d["rl"].empty:
            st.error("No return level data.")
        else:
            rl = d["rl"].copy()
            rp_map = {"1/2a":2,"1/10a":10,"1/50a":50,
                      "1/100a":100,"1/250a":250,"1/1000a":1000}
            rl["rp_years"] = rl["return_period"].map(rp_map)
            loc = st.selectbox("Location", sorted(rl["location"].unique()))
            g   = rl[rl["location"]==loc].sort_values("rp_years")
            fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Water Depth (m)", "Flooded Area (km²)"])
            fig.add_trace(go.Scatter(x=g["rp_years"],y=g["depth_m"],
                                      mode="lines+markers",name="Depth",
                                      line=dict(color="#003580")), row=1,col=1)
            fig.add_trace(go.Scatter(x=g["rp_years"],y=g["area_km2"],
                                      mode="lines+markers",name="Area",
                                      line=dict(color="#C0392B")), row=1,col=2)
            fig.update_xaxes(type="log",
                              tickvals=[2,5,10,50,100,250,1000],
                              ticktext=["2","5","10","50","100","250","1000"])
            fig.update_layout(title=f"SYKE Official Flood Hazard — {loc}",
                              height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 7 — Risk Map
# ===========================================================================
elif page == "🗺️ Risk Map":
    st.title("🗺️ Flood Risk Areas — Finland")

    if d["risk"].empty:
        st.error("No risk area data.")
        st.stop()

    ra = d["risk"].copy()
    fig = px.scatter_mapbox(
        ra, lat="lat", lon="lon", size="pop_at_risk",
        color="risk", color_discrete_map=RISK_CLR,
        hover_name="name",
        hover_data={"river":True,"type":True,"pop_at_risk":True,
                    "lat":False,"lon":False},
        zoom=4.5, center={"lat":64.5,"lon":26},
        mapbox_style="carto-positron",
        title="22 Designated Flood Risk Areas (EU Floods Directive 2007/60/EC)",
        labels={"risk":"Risk Level","pop_at_risk":"Population at Risk"},
    )
    fig.update_layout(height=580, margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Population at Risk by Area (1/100a flood event)")
    fig2 = px.bar(ra.sort_values("pop_at_risk",ascending=True),
                   x="pop_at_risk", y="name", orientation="h",
                   color="risk", color_discrete_map=RISK_CLR,
                   labels={"pop_at_risk":"Population at Risk","name":""})
    fig2.update_layout(height=620, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Details Table")
    show = ra[["name","river","type","risk","pop_at_risk"]].rename(
        columns={"name":"Area","river":"River","type":"Flood Type",
                 "risk":"Risk Level","pop_at_risk":"Population at Risk"})
    st.dataframe(show, use_container_width=True)


# ===========================================================================
# PAGE 8 — Trend Analysis
# ===========================================================================
elif page == "📋 Trend Analysis":
    st.title("📋 Long-Term Trend Analysis (Mann-Kendall)")

    if d["trends"].empty:
        st.error("No trend data — run analysis.py first.")
        st.stop()

    tr = d["trends"].copy()
    tr["label"] = tr["station"].map(LABELS).fillna(tr["station"])
    tr["color"] = tr["trend"].map({"increasing":"#C0392B",
                                    "decreasing":"#27AE60",
                                    "no_significant_trend":"#888888"})

    c1,c2,c3 = st.columns(3)
    c1.metric("📈 Increasing",           int((tr["trend"]=="increasing").sum()))
    c2.metric("📉 Decreasing",           int((tr["trend"]=="decreasing").sum()))
    c3.metric("➡️ No significant trend",  int((tr["trend"]=="no_significant_trend").sum()))

    fig = go.Figure()
    for _, row in tr.iterrows():
        fig.add_trace(go.Bar(x=[row["label"]],
                              y=[row["sen_slope_per_year"] * 10],
                              marker_color=row["color"],
                              showlegend=False))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="Sen's Slope — Water Level Change (m/decade)",
                      xaxis_title="Station", yaxis_title="m/decade",
                      height=410, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Statistics")
    show = tr[["label","n_years","mean_level_m","std_level_m",
               "mk_statistic","p_value","sen_slope_per_year","trend"]].copy()
    show["sen_slope_per_year"] = (show["sen_slope_per_year"]*1000).round(3)
    show.columns = ["Station","Years","Mean (m)","Std (m)",
                    "MK Z","p-value","Slope (mm/yr)","Trend"]
    st.dataframe(show, use_container_width=True)
    st.caption("Mann-Kendall non-parametric test. p < 0.05 = statistically significant.")

# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Data CC BY 4.0 — SYKE & FMI  |  Code MIT License")
