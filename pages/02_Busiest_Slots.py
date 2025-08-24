# pages/03_Busiest_Slots.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Busiest Time Slots — Avoid Congestion", layout="wide")
st.title("Busiest Time Slots — Avoid Congestion")

# ---------------- Paths & Loading ----------------
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "flight_data_clean.csv"

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(
        CSV_PATH,
        parse_dates=["Date","STD","ATD","STA","ATA"],
    )
    df.columns = [c.strip() for c in df.columns]
    # Derived
    df["dep_hour"] = df["STD"].dt.hour
    df["arr_hour"] = df["STA"].dt.hour
    df["dow"] = df["Date"].dt.dayofweek   # 0=Mon
    df["weekday"] = df["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
    if "dep_delay_min" in df.columns:
        df["dep_delay_pos"] = df["dep_delay_min"].clip(lower=0)
    else:
        df["dep_delay_pos"] = np.nan
    if "arr_delay_min" in df.columns:
        df["arr_delay_pos"] = df["arr_delay_min"].clip(lower=0)
    else:
        df["arr_delay_pos"] = np.nan
    df["DateOnly"] = df["Date"].dt.date
    return df

df = load_data()
if df.empty:
    st.error("No rows loaded from CSV.")
    st.stop()

# ---------------- Filters (main page) ----------------
st.markdown("### Filters")

dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
c1, c2, c3 = st.columns([2,2,1])
with c1:
    date_range = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
with c2:
    day_filter = st.multiselect(
        "Days of week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        default=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    )
with c3:
    slot_width = st.selectbox("Slot width (min)", [15, 30, 60], index=2)

c4, c5, c6, c7 = st.columns([1,1,2,2])
with c4:
    hour_start = st.number_input("Hour start", 0, 23, 6, 1)
with c5:
    hour_end = st.number_input("Hour end (inclusive)", 0, 23, 11, 1)
with c6:
    min_flights = st.slider("Min flights per slot", 1, 100, 5, 1)
with c7:
    delay_blend = st.slider("Congestion index: weight of delay (%)", 0, 100, 25, 5)

c8, c9 = st.columns(2)
from_opt = ["(All)"] + (sorted(df["From"].dropna().unique().tolist()) if "From" in df.columns else [])
to_opt   = ["(All)"] + (sorted(df["To"].dropna().unique().tolist())   if "To"   in df.columns else [])
with c8:
    sel_from = st.selectbox("Origin (From)", from_opt, index=0)
with c9:
    sel_to   = st.selectbox("Destination (To)", to_opt, index=0)

# Apply filters
mask = df["Date"].dt.date.between(date_range[0], date_range[-1])
mask &= df["weekday"].isin(day_filter)
if sel_from != "(All)" and "From" in df.columns:
    mask &= (df["From"] == sel_from)
if sel_to != "(All)" and "To" in df.columns:
    mask &= (df["To"] == sel_to)
df_f = df.loc[mask].copy()

# Guard rail
if df_f.empty:
    st.warning("No rows after applying filters.")
    st.stop()

# ---------------- Helpers ----------------
def slot_floor(ts: pd.Series, width_min: int) -> pd.Series:
    """Floor timestamps to slot width (min)."""
    return ts.dt.floor(f"{width_min}min")

def slot_label(ts: pd.Series) -> pd.Series:
    """Return HH:MM label as time-of-day, independent of date."""
    return ts.dt.strftime("%H:%M")

# replace your current normalize_per_day with this:
def normalize_per_day(counts_by_slot_and_date: pd.DataFrame, label_col: str) -> pd.Series:
    """
    counts_by_slot_and_date: has columns ['DateOnly', <label_col>, 'count']
    Returns average flights per day per slot label (indexed by <label_col>).
    """
    if counts_by_slot_and_date.empty:
        return pd.Series(dtype=float)
    return (
        counts_by_slot_and_date
        .groupby(label_col)["count"]
        .mean()    # average over distinct days because we counted per day already
        .sort_index()
    )

def build_congestion_index(volume: pd.Series, delay: pd.Series, weight_delay_pct: int) -> pd.Series:
    """
    Blend normalized volume and normalized delay into a single congestion score.
    weight_delay_pct: 0..100 (how much weight to give to delay vs volume)
    """
    # Align indices
    idx = volume.index.union(delay.index)
    v = volume.reindex(idx).fillna(0)
    d = delay.reindex(idx).fillna(0)

    # Normalize (0..1)
    v_norm = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v*0
    d_norm = (d - d.min()) / (d.max() - d.min()) if d.max() > d.min() else d*0

    w = weight_delay_pct / 100.0
    return (1 - w) * v_norm + w * d_norm

def top_and_bottom(df_metric: pd.Series, k: int = 10):
    """Return top-k and bottom-k slots with values."""
    s = df_metric.dropna().sort_values(ascending=False)
    return s.head(k), s.tail(k)

# Build slot columns for departures & arrivals
df_f["dep_slot_dt"] = slot_floor(df_f["STD"], slot_width)
df_f["arr_slot_dt"] = slot_floor(df_f["STA"], slot_width)
df_f["dep_slot_label"] = slot_label(df_f["dep_slot_dt"])
df_f["arr_slot_label"] = slot_label(df_f["arr_slot_dt"])

# Limit to hour window (on slot *start* hour)
df_f = df_f[(df_f["dep_slot_dt"].dt.hour.between(hour_start, hour_end)) |
            (df_f["arr_slot_dt"].dt.hour.between(hour_start, hour_end))].copy()

# ---------------- Tabs ----------------
tab_dep, tab_arr, tab_maps = st.tabs(["🛫 Departures", "🛬 Arrivals", "🗺 Heatmaps"])

# ===================== DEPARTURES =====================
with tab_dep:
    st.subheader("Busiest Departure Slots")

    dep = df_f.dropna(subset=["dep_slot_dt"]).copy()
    if dep.empty:
        st.info("No departures in the filtered window.")
    else:
        dep_counts_day = (
            dep.groupby(["DateOnly", "dep_slot_label"])
            .size().rename("count").reset_index()
        )
        dep_avg_per_slot = normalize_per_day(dep_counts_day, "dep_slot_label")

        dep_delay_per_slot = dep.groupby("dep_slot_label")["dep_delay_pos"].mean()

        # --- define dep_congestion here ---
        dep_congestion = build_congestion_index(dep_avg_per_slot, dep_delay_per_slot, delay_blend)

        keep = dep_avg_per_slot[dep_avg_per_slot >= min_flights]
        if keep.empty:
            st.warning("No slots meet the 'min flights per slot' threshold.")
        else:
            dep_avg_per_slot = dep_avg_per_slot.loc[keep.index]
            dep_congestion = dep_congestion.loc[keep.index]  # <--- safe now
            dep_delay_per_slot = dep_delay_per_slot.reindex(keep.index)

            # Leaderboards etc …
            # Leaderboards
            top_busy, low_busy = top_and_bottom(dep_avg_per_slot, k=10)
            top_cong, low_cong = top_and_bottom(dep_congestion, k=10)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Busiest dep slot (by volume)",
                          f"{top_busy.index[0]}",
                          help=f"~{top_busy.iloc[0]:.1f} flights/day")
            with k2:
                st.metric("Worst dep slot (congestion score)",
                          f"{top_cong.index[0]}",
                          help=f"score={top_cong.iloc[0]:.2f}")
            with k3:
                st.metric("Best dep slot (lowest congestion)",
                          f"{low_cong.index[-1]}",
                          help=f"score={low_cong.iloc[-1]:.2f}")

            # Charts
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Bar(x=dep_avg_per_slot.index, y=dep_avg_per_slot.values,
                                     name="Avg flights / day"))
                fig.add_trace(go.Scatter(x=dep_congestion.index, y=dep_congestion.values,
                                         mode="lines+markers", name=f"Congestion (vol {100-delay_blend}% + delay {delay_blend}%)",
                                         yaxis="y2"))
                fig.update_layout(
                    height=360,
                    xaxis_title="Departure slot (HH:MM)",
                    yaxis=dict(title="Avg flights / day"),
                    yaxis2=dict(title="Congestion score (0..1)", overlaying="y", side="right"),
                    margin=dict(l=10,r=10,t=30,b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            except ModuleNotFoundError:
                st.bar_chart(dep_avg_per_slot, height=240)
                st.info("Install Plotly for the congestion overlay: `pip install plotly`")

            st.markdown("#### Departure Slot Leaderboards")
            colA, colB = st.columns(2)
            with colA:
                st.write("Top 10 busiest (avg flights/day)")
                st.dataframe(top_busy.reset_index().rename(columns={"index":"slot","count":"avg_flights_per_day"}), use_container_width=True)
            with colB:
                st.write("Top 10 worst congestion (score)")
                st.dataframe(top_cong.reset_index().rename(columns={"index":"slot",0:"congestion_score"}), use_container_width=True)

            st.download_button("Download departure slot table (CSV)",
                               dep_avg_per_slot.reset_index().rename(columns={"index":"slot",0:"avg_flights_per_day"}).to_csv(index=False).encode("utf-8"),
                               file_name="departure_slot_volume.csv",
                               mime="text/csv")

# ===================== ARRIVALS =====================
with tab_arr:
    st.subheader("Busiest Arrival Slots")

    arr = df_f.dropna(subset=["arr_slot_dt"]).copy()
    if arr.empty:
        st.info("No arrivals in the filtered window.")
    else:
        arr_counts_day = (
            arr.groupby(["DateOnly", "arr_slot_label"])
            .size().rename("count").reset_index()
        )
        arr_avg_per_slot = normalize_per_day(arr_counts_day, "arr_slot_label")

        arr_delay_per_slot = arr.groupby("arr_slot_label")["arr_delay_pos"].mean()

        # --- define arr_congestion here ---
        arr_congestion = build_congestion_index(arr_avg_per_slot, arr_delay_per_slot, delay_blend)

        keep = arr_avg_per_slot[arr_avg_per_slot >= min_flights]
        if keep.empty:
            st.warning("No slots meet the 'min flights per slot' threshold.")
        else:
            arr_avg_per_slot = arr_avg_per_slot.loc[keep.index]
            arr_congestion = arr_congestion.loc[keep.index]  # <--- safe now
            arr_delay_per_slot = arr_delay_per_slot.reindex(keep.index)

            # Leaderboards etc …

            top_busy, low_busy = top_and_bottom(arr_avg_per_slot, k=10)
            top_cong, low_cong = top_and_bottom(arr_congestion, k=10)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Busiest arr slot (by volume)",
                          f"{top_busy.index[0]}",
                          help=f"~{top_busy.iloc[0]:.1f} flights/day")
            with k2:
                st.metric("Worst arr slot (congestion score)",
                          f"{top_cong.index[0]}",
                          help=f"score={top_cong.iloc[0]:.2f}")
            with k3:
                st.metric("Best arr slot (lowest congestion)",
                          f"{low_cong.index[-1]}",
                          help=f"score={low_cong.iloc[-1]:.2f}")

            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Bar(x=arr_avg_per_slot.index, y=arr_avg_per_slot.values,
                                     name="Avg flights / day"))
                fig.add_trace(go.Scatter(x=arr_congestion.index, y=arr_congestion.values,
                                         mode="lines+markers", name=f"Congestion (vol {100-delay_blend}% + delay {delay_blend}%)",
                                         yaxis="y2"))
                fig.update_layout(
                    height=360,
                    xaxis_title="Arrival slot (HH:MM)",
                    yaxis=dict(title="Avg flights / day"),
                    yaxis2=dict(title="Congestion score (0..1)", overlaying="y", side="right"),
                    margin=dict(l=10,r=10,t=30,b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            except ModuleNotFoundError:
                st.bar_chart(arr_avg_per_slot, height=240)
                st.info("Install Plotly for the congestion overlay: `pip install plotly`")

            st.markdown("#### Arrival Slot Leaderboards")
            colA, colB = st.columns(2)
            with colA:
                st.write("Top 10 busiest (avg flights/day)")
                st.dataframe(top_busy.reset_index().rename(columns={"index":"slot","count":"avg_flights_per_day"}), use_container_width=True)
            with colB:
                st.write("Top 10 worst congestion (score)")
                st.dataframe(top_cong.reset_index().rename(columns={"index":"slot",0:"congestion_score"}), use_container_width=True)

            st.download_button("Download arrival slot table (CSV)",
                               arr_avg_per_slot.reset_index().rename(columns={"index":"slot",0:"avg_flights_per_day"}).to_csv(index=False).encode("utf-8"),
                               file_name="arrival_slot_volume.csv",
                               mime="text/csv")

# ===================== HEATMAPS =====================
with tab_maps:
    st.subheader("Heatmaps & Patterns")
    st.caption("Spot patterns across days/hours and by airport (darker = higher).")

    try:
        import plotly.express as px

        # Hour x Weekday (departures)
        dep_hm = df_f.groupby(["weekday","dep_hour"]).size().rename("flights").reset_index()
        dep_hm["weekday"] = pd.Categorical(dep_hm["weekday"],
                                           ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], ordered=True)
        fig1 = px.imshow(
            dep_hm.pivot(index="weekday", columns="dep_hour", values="flights"),
            aspect="auto", color_continuous_scale="Blues", origin="lower",
            labels=dict(color="Departures (count)")
        )
        fig1.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), title="Departures — Hour × Weekday")
        st.plotly_chart(fig1, use_container_width=True)

        # Hour x Weekday (arrivals)
        arr_hm = df_f.groupby(["weekday","arr_hour"]).size().rename("flights").reset_index()
        arr_hm["weekday"] = pd.Categorical(arr_hm["weekday"],
                                           ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], ordered=True)
        fig2 = px.imshow(
            arr_hm.pivot(index="weekday", columns="arr_hour", values="flights"),
            aspect="auto", color_continuous_scale="Greens", origin="lower",
            labels=dict(color="Arrivals (count)")
        )
        fig2.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), title="Arrivals — Hour × Weekday")
        st.plotly_chart(fig2, use_container_width=True)

        # Airport × Hour heatmaps (top 12 keeps it readable)
        axis_choice = st.radio("Airport heatmap dimension", ["Origin (From)", "Destination (To)"], horizontal=True)
        if axis_choice.startswith("Origin") and "From" in df_f.columns:
            base = (df_f.groupby(["From","dep_hour"]).size().rename("flights").reset_index())
            key = "From"
            title = "Departures — Hour × Origin"
        elif "To" in df_f.columns:
            base = (df_f.groupby(["To","arr_hour"]).size().rename("flights").reset_index()
                    .rename(columns={"To":"Dest","arr_hour":"dep_hour"}))
            base = base.rename(columns={"Dest":"To"})   # align names
            key = "To"
            title = "Arrivals — Hour × Destination"
        else:
            base = None

        if base is not None and not base.empty:
            top_keys = base.groupby(key)["flights"].sum().sort_values(ascending=False).head(12).index
            base = base[base[key].isin(top_keys)]
            fig3 = px.imshow(
                base.pivot(index=key, columns="dep_hour", values="flights").sort_index(),
                aspect="auto", color_continuous_scale="Viridis", origin="lower",
                labels=dict(color="Flights (count)")
            )
            fig3.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), title=title)
            st.plotly_chart(fig3, use_container_width=True)

    except ModuleNotFoundError:
        st.info("Install Plotly for interactive heatmaps: `pip install plotly`")

# ---------------- Footer ----------------
st.caption(
    "Busiest slots: average flights per slot per day. Congestion index blends volume and average delay. "
    "Use slot width (15/30/60) and filters to find low-traffic windows for scheduling."
)
