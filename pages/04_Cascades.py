# pages/04_Cascades.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Cascading Impacts — Delay Network", layout="wide")
st.title("Cascading Impacts — Delay Network & Simulation")

# ---------------- Paths & Load ----------------
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "flight_data_clean.csv"

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(
        CSV_PATH,
        parse_dates=["Date","STD","ATD","STA","ATA"],
    )
    df.columns = [c.strip() for c in df.columns]

    # Derive delays if not present
    if "dep_delay_min" not in df.columns and {"ATD","STD"}.issubset(df.columns):
        df["dep_delay_min"] = (df["ATD"] - df["STD"]).dt.total_seconds() / 60
    if "arr_delay_min" not in df.columns and {"ATA","STA"}.issubset(df.columns):
        df["arr_delay_min"] = (df["ATA"] - df["STA"]).dt.total_seconds() / 60

    df["dep_delay_pos"] = df["dep_delay_min"].clip(lower=0)
    df["arr_delay_pos"] = df["arr_delay_min"].clip(lower=0)

    # Handy formatted label
    fn = df.get("Flight Number", pd.Series(["?"]*len(df))).astype(str)
    df["flight_id"] = df["Date"].dt.strftime("%Y-%m-%d") + "_" + fn
    return df

df = load_data()
if df.empty:
    st.error("CSV returned no rows.")
    st.stop()

# ---------------- Controls ----------------
st.markdown("### Build the dependency graph (turnaround logic)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    key_choice = st.selectbox(
        "Turnaround identity",
        options=[
            "Tail (best if available)",
            "Airline + From (proxy)",
            "Airline only",
            "From only",
        ],
        index=0,
        help="Flights sharing this identity can connect if the next STD is within the turn window after previous ATA."
    )
with c2:
    turn_min = st.number_input("TURN_MIN (min)", 15, 360, 45, step=5,
                               help="Earliest plausible ground time to service the next flight.")
with c3:
    turn_max = st.number_input("TURN_MAX (min)", 30, 720, 180, step=10,
                               help="Latest time after arrival to still consider it part of a chain.")
with c4:
    min_delay_considered = st.number_input("Min dep delay to propagate (min)", 0, 240, 5, step=5,
                                           help="Ignore tiny upstream delays below this threshold.")

c5, c6, c7, c8 = st.columns(4)
with c5:
    influence_cap = st.slider("Cap edge influence at (min)", 10, 180, 60, step=5,
                              help="Edge weight ~ dep_delay / cap (clipped to 1.0).")
with c6:
    score_kind = st.selectbox("Impact score",
                              ["Outgoing influence sum", "Betweenness centrality", "PageRank"],
                              index=0,
                              help="Different centralities to rank cascading impact.")
with c7:
    top_k = st.number_input("Top-K to display", 5, 50, 10, step=1)
with c8:
    viz_limit = st.number_input("Max nodes in visualization", 50, 600, 220, step=20,
                                help="Subgraph larger than this can be slow to render.")

st.markdown("### Optional filters")
c9, c10, c11 = st.columns(3)
with c9:
    if "From" in df.columns:
        from_air = st.selectbox("Origin filter", ["(All)"] + sorted(df["From"].dropna().unique().tolist()), index=0)
    else:
        from_air = "(All)"
with c10:
    if "To" in df.columns:
        to_air = st.selectbox("Destination filter", ["(All)"] + sorted(df["To"].dropna().unique().tolist()), index=0)
    else:
        to_air = "(All)"
with c11:
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)

# Apply filters
mask = df["Date"].dt.date.between(date_range[0], date_range[-1])
if "From" in df.columns and from_air != "(All)":
    mask &= (df["From"] == from_air)
if "To" in df.columns and to_air != "(All)":
    mask &= (df["To"] == to_air)

df_f = df.loc[mask].dropna(subset=["STD","ATA"]).copy()
st.caption(f"Working set size: {len(df_f):,} rows")

# ---------------- Build graph ----------------
try:
    import networkx as nx
except ModuleNotFoundError:
    st.error("Missing dependency: networkx. Install with `pip install networkx`.")
    st.stop()

# Choose grouping keys
if key_choice == "Tail (best if available)" and "Tail" in df_f.columns:
    KEYS = ["Tail"]
elif key_choice == "Airline + From (proxy)":
    KEYS = [c for c in ["Airline", "From"] if c in df_f.columns]
elif key_choice == "Airline only" and "Airline" in df_f.columns:
    KEYS = ["Airline"]
elif key_choice == "From only" and "From" in df_f.columns:
    KEYS = ["From"]
else:
    KEYS = ["Date"]  # fallback to avoid cross-day links

df_f["STD_min"] = df_f["STD"]
df_f["ATA_min"] = df_f["ATA"]

G = nx.DiGraph()
cap = float(influence_cap)

# Build edges per group
for _, group in df_f.groupby(KEYS, dropna=False):
    grp = group.sort_values("ATA_min")
    for i in range(len(grp)):
        ai = grp.iloc[i]
        a_delay = max(0.0, float(ai.get("dep_delay_min", 0) or 0))
        if a_delay < min_delay_considered:
            continue
        for j in range(i+1, len(grp)):
            bj = grp.iloc[j]
            gap = (bj["STD_min"] - ai["ATA_min"]).total_seconds() / 60.0
            if gap < turn_min:
                continue
            if gap > turn_max:
                break
            weight = min(1.0, a_delay / cap)  # 0..1
            if weight > 0:
                G.add_edge(ai["flight_id"], bj["flight_id"], weight=weight)

n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
st.markdown(f"**Graph size**: {n_nodes:,} nodes, {n_edges:,} edges")
if n_edges == 0:
    st.warning("No cascading edges were formed. Try lowering TURN_MIN, raising TURN_MAX, or lowering 'Min dep delay'.")
    st.stop()

# ---------------- Scores ----------------
# outgoing influence (direct)
out_strength = {n: sum(d["weight"] for _,_,d in G.out_edges(n, data=True)) for n in G.nodes()}

if score_kind == "Outgoing influence sum":
    score_series = pd.Series(out_strength, name="score")
elif score_kind == "Betweenness centrality":
    score_series = pd.Series(nx.betweenness_centrality(G, weight="weight", normalized=True), name="score")
else:
    score_series = pd.Series(nx.pagerank(G, weight="weight"), name="score")

# Merge flight info
meta_cols = ["Date","STD","ATD","STA","ATA","Flight Number","From","To","Airline","Tail","dep_delay_min","arr_delay_min"]
meta_cols = [c for c in meta_cols if c in df_f.columns]
info_map = df_f.set_index("flight_id")[meta_cols]

scores_df = (
    score_series.sort_values(ascending=False).head(top_k).to_frame()
    .join(pd.Series(out_strength, name="out_influence"), how="left")
    .join(info_map, how="left")
    .reset_index().rename(columns={"index":"flight_id"})
)

# Pretty up
for c in ("dep_delay_min","arr_delay_min","out_influence","score"):
    if c in scores_df.columns:
        scores_df[c] = scores_df[c].round(2)

st.markdown("### Top cascading-impact flights")
st.dataframe(scores_df, use_container_width=True)
st.download_button(
    "Download top flights (CSV)",
    scores_df.to_csv(index=False).encode("utf-8"),
    file_name="cascading_impact_flights.csv",
    mime="text/csv"
)

# ---------------- Visualization ----------------
st.markdown("### Network view (top influencers & neighbors)")
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.info("Install matplotlib for the network chart: `pip install matplotlib`")
    plt = None

if plt is not None:
    # Focused subgraph: top nodes + their 1-hop neighbors (truncated)
    top_nodes = scores_df["flight_id"].tolist()
    neighbors = set()
    for n in top_nodes:
        neighbors.update(G.successors(n))
        neighbors.update(G.predecessors(n))
    nodes_of_interest = list(dict.fromkeys(top_nodes + list(neighbors)))  # preserve order, unique
    if len(nodes_of_interest) > viz_limit:
        nodes_of_interest = nodes_of_interest[:viz_limit]
    subG = G.subgraph(nodes_of_interest).copy()

    max_out = max(out_strength.values()) if out_strength else 1.0
    node_sizes = [300 + 900 * (out_strength.get(n, 0) / max_out) for n in subG.nodes()]
    node_colors = ["tab:red" if n in top_nodes else "tab:gray" for n in subG.nodes()]

    labels = {}
    for n in subG.nodes():
        if n in info_map.index and "Flight Number" in info_map.columns:
            labels[n] = str(info_map.loc[n, "Flight Number"])
        else:
            labels[n] = n

    fig = plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(subG, k=0.4, seed=42, weight="weight")
    nx.draw_networkx_edges(subG, pos, arrows=True, arrowsize=8, alpha=0.25, width=1.0)
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=0.5, edgecolors="black")
    # label only top nodes
    top_labels = {n: labels[n] for n in top_nodes if n in subG.nodes()}
    nx.draw_networkx_labels(subG, pos, labels=top_labels, font_size=9)
    plt.title("Cascading Delay Network — Top Influencers (red)")
    plt.axis("off")
    st.pyplot(fig)

# ---------------- Knock-on Simulation ----------------
st.markdown("## Knock-on Delay Simulation")
st.caption("Pick a seed flight, add +X minutes upstream, and propagate across edges with per-hop decay.")

def simulate_knock_on(G, start_fid, added_delay_min, decay=0.65, depth_limit=4, min_effect=0.1):
    """
    Propagate extra delay along edges with attenuation.
    Each edge contributes: bump = incoming_delay * edge_weight * (decay ** depth).
    """
    start_fid = str(start_fid)
    extra = {start_fid: float(added_delay_min)}
    queue = [(start_fid, float(added_delay_min), 0)]
    while queue:
        node, dmin, depth = queue.pop(0)
        if depth >= depth_limit:
            continue
        for _, nxt, data in G.out_edges(node, data=True):
            w = float(data.get("weight", 0.0))
            bump = dmin * w * (decay ** depth)
            if bump <= min_effect:
                continue
            extra[nxt] = extra.get(nxt, 0.0) + bump
            queue.append((nxt, bump, depth+1))
    return extra

# UI controls
sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
with sim_col1:
    seed_flight = st.selectbox("Seed flight (from Top table)", scores_df["flight_id"])
with sim_col2:
    added = st.slider("Add upstream delay (min)", 5, 180, 30, step=5)
with sim_col3:
    decay = st.slider("Per-hop decay", 0.3, 0.95, 0.65, step=0.05)
with sim_col4:
    hops = st.number_input("Max hops", 1, 8, 4, step=1)

if st.button("Run simulation"):
    impacts = simulate_knock_on(G, seed_flight, added, decay=decay, depth_limit=hops)
    sim_df = (
        pd.Series(impacts, name="added_delay_min")
          .sort_values(ascending=False)
          .to_frame()
          .join(info_map, how="left")
          .reset_index().rename(columns={"index":"flight_id"})
    )
    sim_df["added_delay_min"] = sim_df["added_delay_min"].round(2)
    st.markdown("### Simulation result (most affected first)")
    st.dataframe(sim_df.head(50), use_container_width=True)
    st.download_button(
        "Download simulation CSV",
        sim_df.to_csv(index=False).encode("utf-8"),
        file_name="knock_on_simulation.csv",
        mime="text/csv"
    )

# ---------------- Explainers ----------------
with st.expander("ℹ️ Methodology"):
    st.markdown(
        """
- **Edges (A → B)** exist if two flights share the chosen identity (Tail / Airline+From / Airline / From)
  **and** `B.STD` is within `[TURN_MIN, TURN_MAX]` minutes after `A.ATA`.
- **Edge weight** ≈ `min(1.0, dep_delay_A / CAP)` — proxy for how much A's delay could constrain B.
- **Scores**:
  - *Outgoing influence*: sum of outgoing weights → direct spread potential.
  - *Betweenness centrality*: bridging power on shortest paths.
  - *PageRank*: global influence in the network.
- **Simulation**: we inject `+X` minutes at a seed flight, and propagate through edges.
  Each hop is attenuated by the edge weight and a per-hop `decay` factor.
        """
    )

with st.expander("✅ Tips"):
    st.markdown(
        """
- Prefer **Tail** if available; otherwise start with **Airline + From**.
- Start with `TURN_MIN=45`, `TURN_MAX=180`, `Min dep delay=5`, `Cap=60`.
- Zoom via date/origin/destination filters for base-specific insights.
        """
    )
