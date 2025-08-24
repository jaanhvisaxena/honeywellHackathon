# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Flight Ops Insights (6–12)", layout="wide")
st.title("Flight Ops Insights (6–12)")

# ---------- Config ----------
CSV_PATH = r"data/flight_data_clean.csv"
REQUIRED_DATE_COLS = ["Date","STD","ATD","STA","ATA"]
# Optional columns (used if present)
OPT_COLS = ["Flight Number","From","To","Airline","Tail","dep_delay_min","arr_delay_min"]

@st.cache_data(show_spinner=True)
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date","STD","ATD","STA","ATA"])
    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df

def have(cols, df):
    return all(c in df.columns for c in cols)

# ---------- Load ----------
try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load CSV from:\n{CSV_PATH}\n\n{e}")
    st.stop()

st.success(f"Loaded {len(df):,} records")
missing = [c for c in REQUIRED_DATE_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Precompute safe fields
df["dep_hour"] = df["STD"].dt.hour
df["arr_hour"] = df["STA"].dt.hour
if "dep_delay_min" in df.columns:
    df["dep_delay_pos"] = df["dep_delay_min"].clip(lower=0)
else:
    df["dep_delay_pos"] = np.nan
if "arr_delay_min" in df.columns:
    df["arr_delay_pos"] = df["arr_delay_min"].clip(lower=0)
else:
    df["arr_delay_pos"] = np.nan

tab1, tab2, tab3, tab4 = st.tabs(["Best Hours", "Busiest Slots", "What-If", "Cascades"])

# ---------- Tab 1: Best Hours ----------
with tab1:
    st.subheader("Best Departure / Arrival Hours (6–12 window)")
    # Filter to 6–11 for hour bins
    dep_morn = df[df["dep_hour"].between(6,11)]
    arr_morn = df[df["arr_hour"].between(6,11)]

    if dep_morn.empty or arr_morn.empty:
        st.warning("No rows in the 6–12 window. Check your timestamps.")
    else:
        dep_avg = dep_morn.groupby("dep_hour")["dep_delay_pos"].mean().rename("Avg Dep Delay (min)")
        arr_avg = arr_morn.groupby("arr_hour")["arr_delay_pos"].mean().rename("Avg Arr Delay (min)")

        left, right = st.columns(2)
        with left:
            st.write("**Departure** (mean delay, clipped at 0)")
            st.bar_chart(dep_avg)
            st.write("Best dep hour:", int(dep_avg.idxmin()), ":00 —", round(dep_avg.min(), 2), "min")
        with right:
            st.write("**Arrival** (mean delay, clipped at 0)")
            st.bar_chart(arr_avg)
            st.write("Best arr hour:", int(arr_avg.idxmin()), ":00 —", round(arr_avg.min(), 2), "min")

        st.divider()
        st.write("Raw summaries")
        st.divider()
        st.write("Raw summaries")

        # Departure summary table
        dep_table = pd.DataFrame({
            "dep_hour": dep_avg.index,
            "avg_dep_delay_min": dep_avg.values
        }).round(2)

        # Arrival summary table
        arr_table = pd.DataFrame({
            "arr_hour": arr_avg.index,
            "avg_arr_delay_min": arr_avg.values
        }).round(2)

        # Show side-by-side (no merge needed)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Departure summary")
            st.dataframe(dep_table, use_container_width=True)
        with col2:
            st.write("Arrival summary")
            st.dataframe(arr_table, use_container_width=True)

# ---------- Tab 2: Busiest Slots ----------
with tab2:
    st.subheader("Busiest Departure Slots (6–12)")
    if "Flight Number" not in df.columns:
        st.warning("Column 'Flight Number' not found; counting rows instead.")
        counts = df[df["dep_hour"].between(6,11)].groupby("dep_hour").size().rename("Flights")
    else:
        counts = df[df["dep_hour"].between(6,11)].groupby("dep_hour")["Flight Number"].count().rename("Flights")

    if counts.empty:
        st.warning("No departures between 6–12 found.")
    else:
        st.bar_chart(counts)
        st.write("Counts")
        st.dataframe(counts.reset_index().rename(columns={"dep_hour":"Hour"}), use_container_width=True)

    # Optional heatmap-style pivot if 'From' exists
    if "From" in df.columns:
        st.markdown("**Hour × Origin table**")
        pivot = (df[df["dep_hour"].between(6,11)]
                 .pivot_table(index="dep_hour", columns="From",
                              values="Flight Number" if "Flight Number" in df.columns else None,
                              aggfunc="count", fill_value=0))
        st.dataframe(pivot, use_container_width=True)

# ---------- Tab 3: What-If (delay prediction demo) ----------
with tab3:
    st.subheader("What-If: Shift a Flight and Predict Delay")
    st.caption("Baseline demo using RandomForest; requires scikit-learn. Works with morning data; model quality depends on features present.")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        # Minimal features that likely exist
        df_feat = df.dropna(subset=["STD"]).copy()
        df_feat["dow"] = df_feat["Date"].dt.dayofweek
        df_feat["is_weekend"] = (df_feat["dow"] >= 5).astype(int)
        df_feat["dep_hour"] = df_feat["STD"].dt.hour
        df_feat["y"] = df_feat.get("dep_delay_min", pd.Series(np.nan, index=df_feat.index)).clip(lower=0)
        df_feat = df_feat.dropna(subset=["y"])

        cat_cols = [c for c in ["From","To","Airline"] if c in df_feat.columns]
        num_cols = ["dep_hour","dow","is_weekend"]
        X = df_feat[cat_cols + num_cols]
        y = df_feat["y"]

        if len(df_feat) < 50:
            st.warning("Not enough rows with delay target to train a demo model.")
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                     ("num", "passthrough", num_cols)])
            model = Pipeline([("pre", pre),
                              ("rf", RandomForestRegressor(n_estimators=300, random_state=42))])
            model.fit(Xtr, ytr)
            r2 = model.score(Xte, yte)
            st.success(f"Model trained. Holdout R² = {r2:.3f}")

            # Pick a sample
            sample_idx = st.number_input("Sample row index", min_value=0, max_value=len(df_feat)-1, value=0, step=1)
            row = df_feat.iloc[int(sample_idx)]

            shift = st.slider("Shift STD (minutes)", -120, 120, 0, step=15)
            shifted = row.copy()
            shifted_std = row["STD"] + pd.Timedelta(minutes=int(shift))
            shifted["dep_hour"] = shifted_std.hour
            shifted["dow"] = shifted_std.dayofweek
            shifted["is_weekend"] = int(shifted["dow"] >= 5)

            x = shifted[cat_cols + num_cols].to_frame().T
            pred0 = float(model.predict(row[cat_cols + num_cols].to_frame().T)[0])
            predS = float(model.predict(x)[0])

            st.write(f"**Predicted delay @ baseline**: {pred0:.1f} min")
            st.write(f"**Predicted delay @ shift {shift:+d} min**: {predS:.1f} min")
            st.write(f"**Δ change**: {predS - pred0:+.1f} min")

    except ModuleNotFoundError as e:
        st.error(f"Missing ML dependency: {e}. Install with `pip install scikit-learn`.")

# ---------- Tab 4: Cascades (Network) ----------
with tab4:
    st.subheader("Cascading Impact Flights")
    st.caption("Builds a flight dependency graph and ranks flights by outgoing influence. Requires `networkx`.")

    try:
        import networkx as nx
        # Require both STD & ATA to compute gap
        need = ["STD","ATA"]
        df_chain = df.dropna(subset=need).copy()

        # Identity keys
        use_tail = "Tail" in df_chain.columns
        KEYS = ["Tail"] if use_tail else [c for c in ["Airline","From"] if c in df_chain.columns]

        TURN_MIN, TURN_MAX = 45, 180  # minutes window
        df_chain["STD_min"] = df_chain["STD"]
        df_chain["ATA_min"] = df_chain["ATA"]

        df_chain["flight_id"] = (df_chain["Date"].dt.strftime("%Y-%m-%d") + "_" +
                                 df_chain.get("Flight Number", pd.Series(["?"]*len(df_chain))).astype(str))

        G = nx.DiGraph()
        for _, group in df_chain.groupby(KEYS if KEYS else ["Date"], dropna=False):
            grp = group.sort_values("ATA_min")
            for i in range(len(grp)):
                ai = grp.iloc[i]
                for j in range(i+1, len(grp)):
                    bj = grp.iloc[j]
                    gap = (bj["STD_min"] - ai["ATA_min"]).total_seconds() / 60.0
                    if gap < TURN_MIN:
                        continue
                    if gap > TURN_MAX:
                        break
                    a_delay = max(0.0, float(ai.get("dep_delay_min", 0) or 0))
                    weight = min(1.0, a_delay / 60.0)
                    if weight > 0:
                        G.add_edge(ai["flight_id"], bj["flight_id"], weight=weight)

        out_strength = {n: sum(d["weight"] for _,_,d in G.out_edges(n, data=True)) for n in G.nodes()}
        top_spreaders = sorted(out_strength.items(), key=lambda x: x[1], reverse=True)[:10]

        if not top_spreaders:
            st.warning("No cascading edges were formed. Try widening TURN_MIN/MAX or ensure delays exist.")
        else:
            # Table
            rows = []
            for fid, score in top_spreaders:
                row = df_chain[df_chain["flight_id"] == fid].iloc[0]
                rows.append({
                    "flight_id": fid,
                    "Flight Number": row.get("Flight Number","?"),
                    "Route": f"{row.get('From','?')}→{row.get('To','?')}",
                    "dep_delay_min": round(float(row.get("dep_delay_min", np.nan)), 1),
                    "influence": round(float(score), 2),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Small network view (top nodes and neighbors)
            import matplotlib.pyplot as plt
            top_nodes = [fid for fid, _ in top_spreaders]
            neighbors = set()
            for n in top_nodes:
                neighbors.update(G.successors(n))
                neighbors.update(G.predecessors(n))
            subG = G.subgraph(top_nodes + list(neighbors))

            node_colors = ["red" if n in top_nodes else "lightgray" for n in subG.nodes()]
            max_out = max(out_strength.values()) if out_strength else 1.0
            node_sizes = [300 + 800*out_strength.get(n,0)/max_out for n in subG.nodes()]

            fig = plt.figure(figsize=(10,7))
            pos = nx.spring_layout(subG, k=0.4, seed=42)
            nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)
            nx.draw_networkx_edges(subG, pos, arrowsize=8, alpha=0.3)
            # label only the top influencers to avoid clutter
            labels = {n: df_chain.loc[df_chain["flight_id"]==n,"Flight Number"].iloc[0]
                      for n in top_nodes}
            nx.draw_networkx_labels(subG, pos, labels=labels, font_size=9)
            plt.title("Cascading Delay Network (Top Influencers in Red)")
            plt.axis("off")
            st.pyplot(fig)

    except ModuleNotFoundError as e:
        st.error(f"Missing graph dependency: {e}. Install with `pip install networkx matplotlib`.")
    except Exception as e:
        st.exception(e)
