# pages/01_Best_Hours.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Best Hours — Takeoff & Landing", layout="wide")
st.title("Best Time to Takeoff / Land — Pro")

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "flight_data_clean.csv"

# ---------- Load ----------
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
    df["dow"] = df["Date"].dt.dayofweek  # 0=Mon
    df["weekday"] = df["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
    if "dep_delay_min" in df.columns:
        df["dep_delay_pos"] = df["dep_delay_min"].clip(lower=0)
    else:
        df["dep_delay_pos"] = np.nan
    if "arr_delay_min" in df.columns:
        df["arr_delay_pos"] = df["arr_delay_min"].clip(lower=0)
    else:
        df["arr_delay_pos"] = np.nan
    return df

df = load_data()
if df.empty:
    st.error("No rows loaded from CSV.")
    st.stop()

# ---------- Filters on main page ----------
st.markdown("### Filters")
dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    date_range = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
with c2:
    day_filter = st.multiselect(
        "Days of week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        default=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    )
with c3:
    agg_choice = st.selectbox("Delay metric", ["Mean", "Median", "P90"], index=0)

c4, c5, c6, c7 = st.columns([1,1,2,2])
with c4:
    hour_start = st.number_input("Hour start", 0, 23, 6, 1)
with c5:
    hour_end = st.number_input("Hour end (inclusive)", 0, 23, 11, 1)
with c6:
    trim_q = st.slider("Trim outliers (winsorize upper %)", 90, 100, 99, 1)
with c7:
    min_flights = st.slider("Min flights per hour", 5, 200, 20, 5)

c8, c9 = st.columns(2)
from_opt = ["(All)"] + (sorted(df["From"].dropna().unique().tolist()) if "From" in df.columns else [])
to_opt   = ["(All)"] + (sorted(df["To"].dropna().unique().tolist())   if "To"   in df.columns else [])
with c8:
    sel_from = st.selectbox("Origin (From)", from_opt, index=0)
with c9:
    sel_to   = st.selectbox("Destination (To)", to_opt, index=0)

# ---------- Apply filters ----------
mask = df["Date"].dt.date.between(date_range[0], date_range[-1])
mask &= df["weekday"].isin(day_filter)
if sel_from != "(All)" and "From" in df.columns:
    mask &= (df["From"] == sel_from)
if sel_to != "(All)" and "To" in df.columns:
    mask &= (df["To"] == sel_to)
df_f = df.loc[mask].copy()

# ---------- Helpers ----------
def winsorize_upper(s: pd.Series, upper_q: float) -> pd.Series:
    if s.isna().all(): return s
    cap = s.quantile(upper_q/100.0)
    return s.clip(upper=cap)

def agg_series(s: pd.Series, how: str):
    if how == "Median": return s.median()
    if how == "P90":    return s.quantile(0.90)
    return s.mean()

def mean_ci(s: pd.Series):
    s = s.dropna()
    n = len(s)
    if n < 2:
        m = s.mean() if n else np.nan
        return m, np.nan, np.nan
    m = s.mean()
    sd = s.std(ddof=1)
    se = sd/np.sqrt(n)
    ci = 1.96 * se
    return m, m-ci, m+ci

def build_ci_table(grouped, index_name: str):
    """
    Returns a flat DataFrame indexed by hour with columns ['mean','lo','hi'].
    Collapses duplicate hours (if any) and normalizes column names.
    """
    ci = (
        grouped.apply(lambda s: pd.Series(mean_ci(s), index=["mean","lo","hi"]))
        .reset_index()
        .rename(columns={index_name: "hour"})
        .groupby("hour", as_index=True)   # collapse any accidental duplicates
        .first()
        .sort_index()
    )

    # Normalize column names in case pandas gave [0,1,2] or partials
    if ci.shape[1] == 3:
        ci.columns = ["mean","lo","hi"]
    elif ci.shape[1] == 2:
        # Rare: only mean and one bound showed up
        cols = list(ci.columns)
        mapping = {}
        for c in cols:
            mapping[c] = {"0":"mean","1":"lo","2":"hi"}.get(str(c), str(c))
        ci = ci.rename(columns=mapping)
        for needed in ["mean","lo","hi"]:
            if needed not in ci.columns:
                ci[needed] = np.nan
        ci = ci[["mean","lo","hi"]]
    elif ci.shape[1] == 1:
        # Only mean present
        only = ci.columns[0]
        ci = ci.rename(columns={only: "mean"})
        ci["lo"] = np.nan
        ci["hi"] = np.nan
    else:
        # Unexpected shape: build empty shell with proper columns
        ci = pd.DataFrame(index=ci.index, data={"mean": np.nan, "lo": np.nan, "hi": np.nan})

    return ci

# -----------------------------------------------------------------------------
#                     Tabs — Departures / Arrivals / Heatmaps
# -----------------------------------------------------------------------------
tab_dep, tab_arr, tab_maps = st.tabs(["🛫 Departures", "🛬 Arrivals", "🗺 Heatmaps"])

# ====================== DEPARTURES ======================
with tab_dep:
    st.subheader("Best Departure Hours")

    dep = df_f[df_f["dep_hour"].between(hour_start, hour_end)]
    if dep["dep_delay_pos"].notna().sum() == 0:
        st.warning("No departure delay data for current filters.")
    else:
        dep = dep.copy()
        dep["dep_delay_trim"] = winsorize_upper(dep["dep_delay_pos"], trim_q)

        grouped = dep.groupby("dep_hour")["dep_delay_trim"]
        metric = grouped.apply(lambda s: agg_series(s, agg_choice)).rename("delay_min")
        counts = grouped.size().rename("flights")
        stdv   = grouped.std(ddof=1).rename("std_min")

        # Flat CI table, unique index -> avoids duplicate-label reindex errors
        conf = build_ci_table(grouped, "dep_hour")

        # Keep only hours with sufficient support
        hour_range = list(range(hour_start, hour_end + 1))
        valid_hours = counts[counts >= min_flights].index
        metric = metric.reindex(hour_range).loc[valid_hours]
        counts = counts.reindex(metric.index)
        stdv   = stdv.reindex(metric.index)
        conf   = conf.reindex(metric.index)   # safe

        if metric.empty:
            st.warning("No hours meet the minimum flights threshold. Reduce the threshold or widen filters.")
        else:
            # Composite score: lower delay & lower std better; support bonus
            z_delay = (metric - metric.mean()) / (metric.std(ddof=0) or 1)
            z_std   = (stdv   - stdv.mean())  / (stdv.std(ddof=0)   or 1)
            support = np.sqrt(counts / counts.max())
            score = (-0.65*z_delay.fillna(0)) + (-0.25*z_std.fillna(0)) + (0.10*support.fillna(0))

            board = pd.DataFrame({
                "hour": metric.index,
                f"{agg_choice} delay (min)": metric.values.round(2),
                "flights": counts.values.astype(int),
                "std dev (min)": stdv.values.round(2),
                "score": score.values.round(3)
            }).sort_values("score", ascending=False)

            # KPIs
            best_row = board.iloc[0]
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("⭐ Best departure hour", f"{int(best_row['hour']):02d}:00")
            with k2: st.metric(f"{agg_choice} delay @ best (min)", f"{best_row[f'{agg_choice} delay (min)']:.2f}")
            with k3: st.metric("Flights @ best hour", f"{int(best_row['flights'])}")


            # Plot with CI band
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metric.index, y=metric.values, mode="lines+markers",
                                         name=f"{agg_choice} delay"))
                if not conf.empty:
                    fig.add_trace(go.Scatter(x=conf.index, y=conf["hi"], mode="lines", line=dict(width=0),
                                             showlegend=False))
                    fig.add_trace(go.Scatter(x=conf.index, y=conf["lo"], mode="lines", line=dict(width=0),
                                             fill="tonexty", name="~95% CI (mean)", opacity=0.2))
                fig.update_layout(height=340, xaxis_title="Departure hour",
                                  yaxis_title="Delay (min)", margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
            except ModuleNotFoundError:
                st.info("Install Plotly for the CI chart: `pip install plotly`")

            # Flights per hour
            st.caption("Flights per hour (data support)")
            st.bar_chart(counts, height=200)

            # Leaderboard
            st.markdown("#### Best hours leaderboard")
            st.dataframe(board.reset_index(drop=True), use_container_width=True)
            st.download_button("Download departure leaderboard (CSV)",
                               board.to_csv(index=False).encode("utf-8"),
                               file_name="best_departure_hours.csv",
                               mime="text/csv")

# ====================== ARRIVALS ======================
with tab_arr:
    st.subheader("Best Arrival Hours")

    arr = df_f[df_f["arr_hour"].between(hour_start, hour_end)]
    if arr["arr_delay_pos"].notna().sum() == 0:
        st.warning("No arrival delay data for current filters.")
    else:
        arr = arr.copy()
        arr["arr_delay_trim"] = winsorize_upper(arr["arr_delay_pos"], trim_q)

        grouped = arr.groupby("arr_hour")["arr_delay_trim"]
        metric = grouped.apply(lambda s: agg_series(s, agg_choice)).rename("delay_min")
        counts = grouped.size().rename("flights")
        stdv   = grouped.std(ddof=1).rename("std_min")
        conf   = build_ci_table(grouped, "arr_hour")

        hour_range = list(range(hour_start, hour_end + 1))
        valid_hours = counts[counts >= min_flights].index
        metric = metric.reindex(hour_range).loc[valid_hours]
        counts = counts.reindex(metric.index)
        stdv   = stdv.reindex(metric.index)
        conf   = conf.reindex(metric.index)

        if metric.empty:
            st.warning("No hours meet the minimum flights threshold. Reduce the threshold or widen filters.")
        else:
            z_delay = (metric - metric.mean()) / (metric.std(ddof=0) or 1)
            z_std   = (stdv   - stdv.mean())  / (stdv.std(ddof=0)   or 1)
            support = np.sqrt(counts / counts.max())
            score = (-0.65*z_delay.fillna(0)) + (-0.25*z_std.fillna(0)) + (0.10*support.fillna(0))

            board = pd.DataFrame({
                "hour": metric.index,
                f"{agg_choice} delay (min)": metric.values.round(2),
                "flights": counts.values.astype(int),
                "std dev (min)": stdv.values.round(2),
                "score": score.values.round(3)
            }).sort_values("score", ascending=False)

            best_row = board.iloc[0]
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("⭐ Best arrival hour", f"{int(best_row['hour']):02d}:00")
            with k2: st.metric(f"{agg_choice} delay @ best (min)", f"{best_row[f'{agg_choice} delay (min)']:.2f}")
            with k3: st.metric("Flights @ best hour", f"{int(best_row['flights'])}")

            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metric.index, y=metric.values, mode="lines+markers",
                                         name=f"{agg_choice} delay"))
                if not conf.empty:
                    fig.add_trace(go.Scatter(x=conf.index, y=conf["hi"], mode="lines", line=dict(width=0),
                                             showlegend=False))
                    fig.add_trace(go.Scatter(x=conf.index, y=conf["lo"], mode="lines", line=dict(width=0),
                                             fill="tonexty", name="~95% CI (mean)", opacity=0.2))
                fig.update_layout(height=340, xaxis_title="Arrival hour",
                                  yaxis_title="Delay (min)", margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
            except ModuleNotFoundError:
                st.info("Install Plotly for the CI chart: `pip install plotly`")

            st.caption("Flights per hour (data support)")
            st.bar_chart(counts, height=200)

            st.markdown("#### Best hours leaderboard")
            st.dataframe(board.reset_index(drop=True), use_container_width=True)
            st.download_button("Download arrival leaderboard (CSV)",
                               board.to_csv(index=False).encode("utf-8"),
                               file_name="best_arrival_hours.csv",
                               mime="text/csv")

# ====================== HEATMAPS ======================
with tab_maps:
    st.subheader("Heatmaps")
    st.caption("Helps spot patterns across days and hours (darker = higher delay).")

    try:
        import plotly.express as px
        # Hour × Weekday (departures)
        dep_hm = df_f.groupby(["weekday","dep_hour"])["dep_delay_pos"].mean().reset_index()
        dep_hm["weekday"] = pd.Categorical(dep_hm["weekday"],
                                           categories=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                           ordered=True)
        dep_hm = dep_hm.sort_values(["weekday","dep_hour"])
        fig1 = px.imshow(
            dep_hm.pivot(index="weekday", columns="dep_hour", values="dep_delay_pos"),
            aspect="auto", color_continuous_scale="RdYlGn_r", origin="lower",
            labels=dict(color="Mean dep delay (min)")
        )
        fig1.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), title="Departures — Hour × Weekday")
        st.plotly_chart(fig1, use_container_width=True)

        # Hour × Origin or Destination (choose)
        axis_choice = st.radio("Heatmap dimension", ["Origin (From)", "Destination (To)"], horizontal=True)
        if axis_choice.startswith("Origin") and "From" in df_f.columns:
            by = "From"
            hm = df_f.groupby([by,"dep_hour"])["dep_delay_pos"].mean().reset_index()
            title = "Departures — Hour × Origin"
        elif "To" in df_f.columns:
            by = "To"
            hm = df_f.groupby([by,"arr_hour"])["arr_delay_pos"].mean().reset_index()
            hm = hm.rename(columns={"arr_hour":"dep_hour","arr_delay_pos":"dep_delay_pos"})
            title = "Arrivals — Hour × Destination"
        else:
            st.info("No 'From'/'To' columns available to build airport heatmap.")
            hm = None

        if hm is not None and not hm.empty:
            counts_by = df_f[by].value_counts().head(12).index
            hm = hm[hm[by].isin(counts_by)]
            fig2 = px.imshow(
                hm.pivot(index=by, columns="dep_hour", values="dep_delay_pos").sort_index(),
                aspect="auto", color_continuous_scale="RdYlGn_r", origin="lower",
                labels=dict(color="Mean delay (min)")
            )
            fig2.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), title=title)
            st.plotly_chart(fig2, use_container_width=True)

    except ModuleNotFoundError:
        st.info("Install Plotly for interactive heatmaps: `pip install plotly`")

# ---------- Footer ----------
st.caption(
    "Composite score balances lower delay, stability (low std dev), and data support (flights). "
    "Confidence bands use trimmed means (winsorized at the selected percentile)."
)
