# pages/01_Best_Hours.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

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
        # Compute if possible
        if {"ATD","STD"}.issubset(df.columns):
            df["dep_delay_min"] = (df["ATD"] - df["STD"]).dt.total_seconds()/60
            df["dep_delay_pos"] = df["dep_delay_min"].clip(lower=0)
        else:
            df["dep_delay_pos"] = np.nan
    if "arr_delay_min" in df.columns:
        df["arr_delay_pos"] = df["arr_delay_min"].clip(lower=0)
    else:
        if {"ATA","STA"}.issubset(df.columns):
            df["arr_delay_min"] = (df["ATA"] - df["STA"]).dt.total_seconds()/60
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

# Allow future selection: up to +60 days beyond dataset
dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
future_max = dmax + timedelta(days=60)

# Normalize Streamlit date_input (tuple vs list safety)
def normalize_date_range(dr):
    """Return (start_date, end_date) as datetime.date objects."""
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        a, b = dr
        # Streamlit returns datetime.date; just ensure order
        if b < a:
            a, b = b, a
        return a, b
    # Fallback to dataset span
    return dmin, dmax

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    date_range_raw = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=future_max)
    sel_start, sel_end = normalize_date_range(date_range_raw)

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

# Detect forecast mode: if any selected day > dataset max
forecast_mode = sel_end > dmax or sel_start > dmax
if forecast_mode:
    st.info("**Forecast Mode (Experimental)** — selected date range extends beyond available data. "
            "Predictions are simulated from a model trained on the historical week.")

# ---------- Apply filters (for historical mode) ----------
mask = df["Date"].dt.date.between(sel_start, sel_end)
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
    if set(ci.columns) >= {"mean","lo","hi"}:
        return ci[["mean","lo","hi"]]
    # Fill missing columns safely
    out = pd.DataFrame(index=ci.index)
    out["mean"] = ci[ci.columns[0]] if len(ci.columns) > 0 else np.nan
    out["lo"]   = ci[ci.columns[1]] if len(ci.columns) > 1 else np.nan
    out["hi"]   = ci[ci.columns[2]] if len(ci.columns) > 2 else np.nan
    return out

# ---------- Forecast utilities ----------
def train_delay_model(df_train: pd.DataFrame):
    """Train a small RandomForest on historical rows (dep delay)."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
    except ModuleNotFoundError:
        return None, None

    dft = df_train.dropna(subset=["STD"]).copy()
    # Need a target; compute if absent
    if "dep_delay_min" not in dft.columns:
        if {"ATD","STD"}.issubset(dft.columns):
            dft["dep_delay_min"] = (dft["ATD"] - dft["STD"]).dt.total_seconds()/60
        else:
            return None, None
    dft["y"] = dft["dep_delay_min"].clip(lower=0)
    dft = dft.dropna(subset=["y"])
    if len(dft) < 50:
        return None, None

    dft["dow"] = dft["Date"].dt.dayofweek
    dft["dep_hour"] = dft["STD"].dt.hour
    dft["is_weekend"] = (dft["dow"] >= 5).astype(int)

    # Categorical features (fill missing with sentinel)
    for c in ["From","To","Airline"]:
        if c not in dft.columns:
            dft[c] = "UNK"
        else:
            dft[c] = dft[c].fillna("UNK").astype(str)

    cat_cols = [c for c in ["From","To","Airline"] if c in dft.columns]
    num_cols = ["dep_hour","dow","is_weekend"]

    X = dft[cat_cols + num_cols]
    y = dft["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])
    model = Pipeline([
        ("pre", pre),
        ("rf",  RandomForestRegressor(n_estimators=300, random_state=42))
    ])
    model.fit(Xtr, ytr)
    r2 = float(model.score(Xte, yte))
    meta = {"cat_cols": cat_cols, "num_cols": num_cols, "r2": r2}
    return model, meta

def simulate_predictions_for_range(model, meta, start_d: date, end_d: date,
                                   hours: tuple[int,int],
                                   from_val: str | None,
                                   to_val: str | None,
                                   airline_hint: str | None,
                                   df_hist_for_modes: pd.DataFrame):
    """Build a grid for selected dates/hours and predict delays."""
    if model is None:  # safety
        return None

    h0, h1 = hours
    all_dates = pd.date_range(start=start_d, end=end_d, freq="D")
    # Choose representative values if not provided
    def mode_or(val, col):
        if val and val != "(All)":
            return val
        if col in df_hist_for_modes.columns and not df_hist_for_modes[col].dropna().empty:
            return df_hist_for_modes[col].dropna().astype(str).mode().iloc[0]
        return "UNK"

    sim_rows = []
    for d in all_dates:
        dow = d.dayofweek
        is_weekend = int(dow >= 5)
        for h in range(h0, h1+1):
            sim_rows.append({
                "Date": d,
                "dow": dow,
                "is_weekend": is_weekend,
                "dep_hour": h,
                "From": mode_or(from_val, "From"),
                "To": mode_or(to_val, "To"),
                "Airline": mode_or(airline_hint, "Airline"),
            })
    sim = pd.DataFrame(sim_rows)
    # Predict
    X = sim[meta["cat_cols"] + meta["num_cols"]]
    sim["pred_delay_min"] = model.predict(X)
    return sim

# ---------------------------------------------------------------------
#                     Tabs — Departures / Arrivals / Heatmaps
# ---------------------------------------------------------------------
tab_dep, tab_arr, tab_maps = st.tabs(["🛫 Departures", "🛬 Arrivals", "🗺 Heatmaps"])

# ====================== DEPARTURES ======================
with tab_dep:
    st.subheader("Best Departure Hours")

    if not forecast_mode:
        # Historical mode
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
                    if conf is not None and not conf.empty:
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
    else:
        # Forecast mode (Experimental) — Predict per hour on selected dates
        model, meta = train_delay_model(df)
        if model is None:
            st.warning("Forecast requires scikit-learn and >=50 rows with dep delays. Showing historical mode instead.")
        else:
            # Use historical rows filtered by From/To (for representative modes)
            hist_subset = df.copy()
            if sel_from != "(All)" and "From" in df.columns:
                hist_subset = hist_subset[hist_subset["From"] == sel_from]
            if sel_to != "(All)" and "To" in df.columns:
                hist_subset = hist_subset[hist_subset["To"] == sel_to]

            sim = simulate_predictions_for_range(
                model, meta, sel_start, sel_end,
                (int(hour_start), int(hour_end)),
                sel_from if sel_from != "(All)" else None,
                sel_to   if sel_to   != "(All)" else None,
                airline_hint=None,
                df_hist_for_modes=hist_subset if not hist_subset.empty else df
            )
            if sim is None or sim.empty:
                st.warning("Could not simulate predictions for the selected range.")
            else:
                # Aggregate to hour → predicted delay
                pred_hourly = sim.groupby("dep_hour")["pred_delay_min"].mean().reindex(range(hour_start, hour_end+1))
                best_h = int(pred_hourly.idxmin())
                k1,k2,k3 = st.columns(3)
                with k1: st.metric("⭐ Predicted best dep hour", f"{best_h:02d}:00")
                with k2: st.metric("Predicted delay @ best (min)", f"{pred_hourly.min():.2f}")
                with k3: st.metric("Model R² (holdout)", f"{meta['r2']:.3f}")

                st.line_chart(pred_hourly.rename("Predicted delay (min)"))

                st.caption("Forecast is experimental and based on patterns learned from the historical week.")

# ====================== ARRIVALS ======================
with tab_arr:
    st.subheader("Best Arrival Hours")

    # Historical arrivals only (we don't forecast STA vs ATA here; keep logic as-is)
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
                if conf is not None and not conf.empty:
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
    st.caption("Historical patterns across days and hours. Forecast heatmap appears automatically for future selections.")

    try:
        import plotly.express as px

        # Historical — Hour × Weekday (departures)
        if not df_f.empty:
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
            fig1.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), title="Departures — Hour × Weekday (Historical)")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No historical rows for the selected filters/date range.")

        # Historical — Hour × Origin/Destination
        axis_choice = st.radio("Heatmap dimension", ["Origin (From)", "Destination (To)"], horizontal=True)
        if not df_f.empty:
            if axis_choice.startswith("Origin") and "From" in df_f.columns:
                by = "From"
                hm = df_f.groupby([by,"dep_hour"])["dep_delay_pos"].mean().reset_index()
                title = "Departures — Hour × Origin (Historical)"
            elif "To" in df_f.columns:
                by = "To"
                hm = df_f.groupby([by,"arr_hour"])["arr_delay_pos"].mean().reset_index()
                hm = hm.rename(columns={"arr_hour":"dep_hour","arr_delay_pos":"dep_delay_pos"})
                title = "Arrivals — Hour × Destination (Historical)"
            else:
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

        # -------- Predicted Heatmap — Experimental --------
        if forecast_mode:
            st.markdown("#### Predicted Heatmap — Experimental")
            model, meta = train_delay_model(df)
            if model is None:
                st.info("Forecast requires scikit-learn and enough historical rows. Please install or widen filters.")
            else:
                hist_subset = df.copy()
                if sel_from != "(All)" and "From" in df.columns:
                    hist_subset = hist_subset[hist_subset["From"] == sel_from]
                if sel_to != "(All)" and "To" in df.columns:
                    hist_subset = hist_subset[hist_subset["To"] == sel_to]

                sim = simulate_predictions_for_range(
                    model, meta, sel_start, sel_end,
                    (int(hour_start), int(hour_end)),
                    sel_from if sel_from != "(All)" else None,
                    sel_to   if sel_to   != "(All)" else None,
                    airline_hint=None,
                    df_hist_for_modes=hist_subset if not hist_subset.empty else df
                )
                if sim is not None and not sim.empty:
                    # Build a (date × hour) heatmap
                    sim["date_str"] = sim["Date"].dt.strftime("%Y-%m-%d")
                    grid = sim.pivot(index="date_str", columns="dep_hour", values="pred_delay_min")
                    if grid.shape[0] == 1:
                        title = f"Predicted Delay by Hour — {grid.index[0]}"
                    else:
                        title = "Predicted Delay: Date × Hour"
                    figp = px.imshow(
                        grid,
                        aspect="auto", color_continuous_scale="RdYlGn_r", origin="lower",
                        labels=dict(color="Pred delay (min)")
                    )
                    figp.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10), title=title)
                    st.plotly_chart(figp, use_container_width=True)
                else:
                    st.info("No predicted grid could be built for the selected range.")

    except ModuleNotFoundError:
        st.info("Install Plotly for interactive heatmaps: `pip install plotly`")

# ---------- Footer ----------
st.caption(
    "Historical KPIs use trimmed means (winsorized) plus stability and data-support. "
    "Forecast Mode uses a RandomForest trained on your historical week to simulate future hourly delays. "
    "Predictions are experimental and meant for planning ‘what-if’ exploration."
)
