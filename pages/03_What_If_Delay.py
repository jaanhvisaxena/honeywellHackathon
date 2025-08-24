# pages/03_What_If_Delay.py
import streamlit as st, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from what_if_core import train_model, predict_delay_for_shift

st.set_page_config(page_title="What-If Delay", layout="wide")
st.title("What-If: Shift a Flight and Predict Delay")

# Resolve CSV relative to project root (parent of /pages)
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "flight_data_clean.csv"

@st.cache_data(show_spinner=True)
def load_data():
    return pd.read_csv(CSV_PATH, parse_dates=["Date","STD","ATD","STA","ATA"])

@st.cache_resource(show_spinner=True)
def get_model(_df):
    return train_model(_df)

df = load_data()
model, meta = get_model(df)
st.success(f"Model trained. R²={meta['r2']:.3f} | RMSE={meta['rmse']:.1f} min | MAE={meta['mae']:.1f} min")

# ---------- Choose flight by NAME (Flight Number | From→To | Date HH:MM) ----------
df_valid = df.dropna(subset=["Date","STD"]).copy()

def mk_label(r: pd.Series) -> str:
    fn = r.get("Flight Number", "?")
    frm = r.get("From", "?")
    to  = r.get("To", "?")
    d   = r["Date"].date().isoformat()
    t   = r["STD"].strftime("%H:%M")
    return f"{fn} | {frm}→{to} | {d} {t}"

df_valid["label"] = df_valid.apply(mk_label, axis=1)

selected_idx = st.selectbox(
    "Select a flight",
    options=list(df_valid.index),
    format_func=lambda i: df_valid.loc[i, "label"]
)
row = df.loc[selected_idx]
flight_label = df_valid.loc[selected_idx, "label"]

st.markdown("**Chosen flight**")
info_cols = [c for c in ["Date","STD","From","To","Airline","Flight Number"] if c in df.columns]
st.write(pd.DataFrame(row[info_cols]).T)

# ---------- Shift control & predictions ----------
shift = st.slider("Shift minutes (−120 to +120)", -120, 120, 0, step=15)
base = predict_delay_for_shift(row, 0, model, meta)
pred = predict_delay_for_shift(row, shift, model, meta)

st.metric("Baseline predicted delay (min)", f"{base:.1f}")
st.metric(f"Predicted delay @ shift {shift:+d} min (min)", f"{pred:.1f}", delta=f"{pred - base:+.1f}")

# ---------- What-If curve controls ----------
c1, c2, c3 = st.columns(3)
with c1: lo = st.number_input("Min shift", -240, 0, -120, step=15)
with c2: hi = st.number_input("Max shift", 0, 240, 120, step=15)
with c3: stp = st.number_input("Step", 5, 60, 15, step=5)

shifts = list(range(int(lo), int(hi) + 1, int(stp)))
preds = [predict_delay_for_shift(row, s, model, meta) for s in shifts]
delta = [p - base for p in preds]
dd = pd.DataFrame(
    {"shift_min": shifts, "pred_delay_min": preds, "delta_vs_base_min": delta}
).round(1)

# ---------- Plot with flight label in title ----------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dd["shift_min"], dd["pred_delay_min"], marker="o")
ax.axvline(0, linestyle="--", alpha=0.6)
ax.set_xlabel("Shift (min, + = later)")
ax.set_ylabel("Predicted Delay (min)")
ax.set_title(f"Predicted Delay vs Shift — {flight_label}")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("#### What-If table")
st.dataframe(dd, use_container_width=True)
