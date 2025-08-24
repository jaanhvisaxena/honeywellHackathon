# pages/00_AI_Query.py
import os, re, json, time
from pathlib import Path
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

st.set_page_config(page_title="AI Query — NLP Prompts", layout="wide")
st.title("🔎 Ask in Plain English (AI Query)")

# ---------------------------------------------------------------------
# .env (secrets) + Gemini status badge
# ---------------------------------------------------------------------
load_dotenv()  # loads GEMINI_API_KEY if present
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def status_badge(enabled: bool, model: str | None):
    color = "#16a34a" if enabled else "#ef4444"
    text  = f"Gemini: {'ON' if enabled else 'OFF'}" + (f" · {model}" if enabled and model else "")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:6px 10px;border-radius:8px;background:{color};color:white;font-weight:600;">
        {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Data loading (local analytics are 100% open-source)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "flight_data_clean.csv"

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(
        CSV_PATH,
        parse_dates=["Date","STD","ATD","STA","ATA"],
    )
    df.columns = [c.strip() for c in df.columns]

    # Derived fields (safe if missing)
    if "dep_delay_min" not in df.columns and {"ATD","STD"}.issubset(df.columns):
        df["dep_delay_min"] = (df["ATD"] - df["STD"]).dt.total_seconds()/60
    if "arr_delay_min" not in df.columns and {"ATA","STA"}.issubset(df.columns):
        df["arr_delay_min"] = (df["ATA"] - df["STA"]).dt.total_seconds()/60

    df["dep_delay_pos"] = df.get("dep_delay_min", pd.Series(np.nan, index=df.index)).clip(lower=0)
    df["arr_delay_pos"] = df.get("arr_delay_min", pd.Series(np.nan, index=df.index)).clip(lower=0)

    # Time parts
    for col in ("STD","STA"):
        if col in df.columns:
            hh = "dep_hour" if col == "STD" else "arr_hour"
            df[hh] = df[col].dt.hour
    df["dow"] = df["Date"].dt.dayofweek
    df["weekday_name"] = df["dow"].map({0:"mon",1:"tue",2:"wed",3:"thu",4:"fri",5:"sat",6:"sun"})

    # Ensure optional cols exist
    for c in ("From","To","Airline","Tail","Flight Number"):
        if c not in df.columns:
            df[c] = np.nan

    return df

df = load_data()
if df.empty:
    st.error(f"No data found at: {CSV_PATH}")
    st.stop()

# Vocab from your data (help both parsers)
AIRPORTS = sorted(set(pd.concat([df["From"].dropna(), df["To"].dropna()]).unique().tolist()))
IATA_CODES = sorted(set(re.findall(r"\(([A-Z]{3})\)", " ".join(AIRPORTS))))
AIRLINES = sorted(set(df["Airline"].dropna().unique().tolist()))
FLIGHTS = sorted(set(df["Flight Number"].dropna().astype(str).unique().tolist()))

# ---------------------------------------------------------------------
# Parser engine controls
# ---------------------------------------------------------------------
st.sidebar.header("Parser Engine")
use_gemini = st.sidebar.toggle("Use Gemini for NLP parsing", value=bool(GEMINI_API_KEY),
                               help="If off (or key missing), an offline parser is used.")
model_name = st.sidebar.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)

gemini = None
if use_gemini and GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini = genai.GenerativeModel(model_name)
        status_badge(True, model_name)
    except Exception as e:
        st.sidebar.error(f"Gemini init failed: {e}")
        use_gemini = False
        status_badge(False, None)
else:
    status_badge(False, None)

# ---------------------------------------------------------------------
# Schema + validation
# ---------------------------------------------------------------------
# Intents now include a CSV Q&A lane: "csv"
ALLOWED_INTENTS = {"best_hours","busiest_slots","what_if","cascades","csv"}

def coerce_request(d: dict) -> dict:
    """Coerce/validate parser output to safe types with defaults."""
    safe = {}

    # intent
    intent = str(d.get("intent", "best_hours")).strip()
    safe["intent"] = intent if intent in ALLOWED_INTENTS else "best_hours"

    # common fields
    metric = str(d.get("metric","Mean")).strip().title()
    if metric not in {"Mean","Median","P90"}: metric = "Mean"
    safe["metric"] = metric
    safe["origin"] = (d.get("origin") or "").strip()
    safe["destination"] = (d.get("destination") or "").strip()

    # hours
    try:
        hours = d.get("hours", [6,11])
        if not isinstance(hours, (list,tuple)) or len(hours) != 2:
            hours = [6,11]
        h0, h1 = int(hours[0]), int(hours[1])
        h0 = max(0, min(23, h0)); h1 = max(0, min(23, h1))
        if h1 < h0: h0, h1 = h1, h0
        safe["hours"] = [h0, h1]
    except Exception:
        safe["hours"] = [6,11]

    # dates
    try:
        dr = d.get("dates", [])
        if not isinstance(dr, (list,tuple)) or len(dr) != 2:
            dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
            dr = [str(dmin), str(dmax)]
        safe["dates"] = dr
    except Exception:
        dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
        safe["dates"] = [str(dmin), str(dmax)]

    # busiest
    sw = int(d.get("slot_width", 60) or 60)
    if sw not in {15,30,60}: sw = 60
    safe["slot_width"] = sw

    # what-if
    safe["flight_number"] = str(d.get("flight_number","")).strip()
    safe["shift_min"] = int(d.get("shift_min", 0) or 0)

    # cascades
    safe["turn_min"] = int(d.get("turn_min", 45) or 45)
    safe["turn_max"] = int(d.get("turn_max", 180) or 180)
    ident = str(d.get("identity","Tail")).strip()
    if ident not in {"Tail","Airline+From","Airline","From"}:
        ident = "Tail"
    safe["identity"] = ident
    safe["top_k"] = max(1, min(50, int(d.get("top_k", 10) or 10)))

    # CSV Q&A fields
    safe["csv_action"] = str(d.get("csv_action","")).strip().lower()  # schema|head|sample|describe|value_counts|filter|count
    safe["csv_column"] = str(d.get("csv_column","")).strip()
    safe["csv_limit"]  = max(1, min(1000, int(d.get("csv_limit", 50) or 50)))
    safe["csv_where"]  = str(d.get("csv_where","")).strip()

    return safe

# ---------------------------------------------------------------------
# Offline parser (regex-based)
# ---------------------------------------------------------------------
def parse_offline(text: str) -> dict:
    t = text.lower()

    # CSV Q&A?
    csv_action = ""
    if any(k in t for k in ["columns","schema"]): csv_action = "schema"
    elif re.search(r"\bhead\s+\d+\b", t):         csv_action = "head"
    elif "head" in t:                             csv_action = "head"
    elif re.search(r"\bsample\s+\d+\b", t):       csv_action = "sample"
    elif "sample" in t:                           csv_action = "sample"
    elif "describe" in t:                         csv_action = "describe"
    elif "value counts" in t or "valuecounts" in t or "vc" in t: csv_action = "value_counts"
    elif "count rows" in t or "row count" in t:   csv_action = "count"
    elif "where " in t:                           csv_action = "filter"

    if csv_action:
        # column (if any)
        col = ""
        m = re.search(r"(describe|value\s*counts)\s+([A-Za-z _\-]+)", t)
        if m: col = m.group(2).strip()
        # limit
        lim = 50
        m = re.search(r"(head|sample)\s+(\d+)", t)
        if m: lim = int(m.group(2))
        m = re.search(r"limit\s+(\d+)", t)
        if m: lim = int(m.group(1))
        # where
        where_text = ""
        m = re.search(r"where\s+(.+)$", t)
        if m: where_text = m.group(1).strip()

        return coerce_request({
            "intent": "csv",
            "csv_action": csv_action,
            "csv_column": col,
            "csv_limit": lim,
            "csv_where": where_text,
            "hours": [6,11],
            "dates": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        })

    # main 4 intents
    if any(k in t for k in ["busiest","congestion","slot","traffic"]):
        intent = "busiest_slots"
    elif any(k in t for k in ["what-if","what if","shift","predict","tune schedule"]):
        intent = "what_if"
    elif any(k in t for k in ["cascade","cascading","impact","ripple","network"]):
        intent = "cascades"
    else:
        intent = "best_hours"

    # metric
    metric = "Mean"
    if "median" in t: metric = "Median"
    if "p90" in t or "90th" in t: metric = "P90"

    # slot width
    sw = 60
    m = re.search(r"\b(15|30|60)\s*(min|minutes)?\b", t)
    if m: sw = int(m.group(1))

    # hours
    h0, h1 = 6, 11
    m = re.search(r"\b(\d{1,2})\s*[-to]\s*(\d{1,2})\b", t)
    if m:
        h0, h1 = int(m.group(1)), int(m.group(2))
    elif "morning" in t:
        h0, h1 = 6, 11

    # dates
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    if "today" in t:
        dmin = dmax = date.today()
    elif "tomorrow" in t:
        dmin = dmax = date.fromordinal(date.today().toordinal()+1)

    # airports
    def match_airport(tt):
        codes = [c for c in re.findall(r"\b([A-Z]{3})\b", tt.upper()) if c in IATA_CODES]
        out = [ap for ap in AIRPORTS if ap.lower() in tt]
        return list(dict.fromkeys(codes + out))

    aps = match_airport(t)
    origin = aps[0] if len(aps) >= 1 else ""
    destination = aps[1] if len(aps) >= 2 else ""

    # --- Route listing shortcut: "flights from X to Y" ---
    # If user asks to list flights on a route, route it to CSV filter intent.
    m_route = re.search(r"\bflights?\s+from\s+(.+?)\s+to\s+(.+)$", t)
    if m_route:
        # Prefer parsed airports if we got them; otherwise use raw capture
        o = origin or m_route.group(1).strip()
        d = destination or m_route.group(2).strip()
        # Build a simple CSV where-clause; our apply_where() supports "contains"
        return coerce_request({
            "intent": "csv",
            "csv_action": "filter",
            "csv_where": f"From contains {o} and To contains {d}",
            "csv_limit": 500,
            "hours": [0, 23],
            "dates": [str(df['Date'].min().date()), str(df['Date'].max().date())],
        })

    # flight
    flt = ""
    cand = [c.replace(" ","") for c in re.findall(r"\b[A-Z0-9]{2,6}\s?\d{1,4}\b", t.upper())]
    for c in cand:
        if c in FLIGHTS: flt = c; break

    # shift
    m = re.search(r"shift.*?([+-]?\d+)\s*min", t)
    shift = int(m.group(1)) if m else 0

    # cascades tuning
    tm, tx = 45, 180
    m = re.search(r"turn[_\s-]?min\s*[:=]?\s*(\d+)", t);  tm = int(m.group(1)) if m else tm
    m = re.search(r"turn[_\s-]?max\s*[:=]?\s*(\d+)", t);  tx = int(m.group(1)) if m else tx
    ident = "Tail"
    if "airline + from" in t or "airline and from" in t: ident = "Airline+From"
    elif "airline only" in t: ident = "Airline"
    elif "from only" in t:   ident = "From"

    # top_k
    m = re.search(r"top\s*(\d+)", t)
    topk = int(m.group(1)) if m else 10

    return coerce_request({
        "intent": intent, "metric": metric,
        "origin": origin, "destination": destination,
        "hours": [h0, h1], "dates": [str(dmin), str(dmax)],
        "slot_width": sw, "flight_number": flt, "shift_min": shift,
        "turn_min": tm, "turn_max": tx, "identity": ident, "top_k": topk
    })

# ---------------------------------------------------------------------
# Gemini parser (JSON-only; fallback to offline if anything goes wrong)
# ---------------------------------------------------------------------
GEMINI_SYSTEM = """You convert a user question about flights into a STRICT JSON object.
Fields and allowed values:
- intent: one of ["best_hours","busiest_slots","what_if","cascades","csv"]
- metric: one of ["Mean","Median","P90"]
- origin: airport name or IATA like "Mumbai(BOM)" or "BOM" (may be empty "")
- destination: same style (may be empty "")
- hours: [start_hour_int, end_hour_int] 0..23 inclusive
- dates: ["YYYY-MM-DD","YYYY-MM-DD"] inclusive range; if absent, use the dataset range
- slot_width: one of [15,30,60]
- flight_number: like "6E1185" (may be "")
- shift_min: integer minutes (may be negative)
- turn_min: integer minutes >=0
- turn_max: integer minutes >= turn_min
- identity: one of ["Tail","Airline+From","Airline","From"]
- top_k: integer (1..50)
- csv_action: one of ["schema","head","sample","describe","value_counts","filter","count"] (may be "")
- csv_column: string (may be "")
- csv_limit: integer (default 50)
- csv_where: free text condition like 'Flight Number = 6E1185' or 'From contains BOM and Date between 2025-07-20 and 2025-07-25' (may be "")

Rules:
- If the user asks for “flights from X to Y” or “list flights”, set intent="csv", csv_action="filter",
  and set csv_where to "From contains <X> and To contains <Y>" using IATA or airport names where possible.
- If ambiguous, choose sensible defaults: hours [6,11], metric "Mean", slot_width 60, identity "Tail".
Return ONLY JSON. No prose. Choose reasonable defaults if ambiguous.
"""

def parse_with_gemini(prompt_text: str) -> dict:
    if not gemini:
        return parse_offline(prompt_text)
    exemplar_vocab = {"iata_codes": IATA_CODES[:50], "airlines": AIRLINES[:50], "flights": FLIGHTS[:100]}
    try:
        resp = gemini.generate_content(
            [
                {"role": "user", "parts": GEMINI_SYSTEM},
                {"role": "user", "parts": json.dumps(exemplar_vocab)},
                {"role": "user", "parts": prompt_text.strip()},
            ],
        )
        txt = (resp.text or "{}").strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            txt = re.sub(r"^json", "", txt).strip()
        return coerce_request(json.loads(txt))
    except Exception:
        time.sleep(0.4)
        try:
            resp = gemini.generate_content(prompt_text.strip())
            return coerce_request(json.loads(resp.text or "{}"))
        except Exception:
            return parse_offline(prompt_text)

# ---------------------------------------------------------------------
# Helpers: filters + aggregations
# ---------------------------------------------------------------------
def apply_filters(df_in: pd.DataFrame, req: dict) -> tuple[pd.DataFrame, tuple[int,int]]:
    d0, d1 = req["dates"]
    try:
        d0 = pd.to_datetime(d0).date()
        d1 = pd.to_datetime(d1).date()
    except Exception:
        d0, d1 = df_in["Date"].min().date(), df_in["Date"].max().date()
    mask = df_in["Date"].dt.date.between(d0, d1)

    if req["origin"]:
        mask &= df_in["From"].fillna("").str.contains(re.escape(req["origin"]), case=False, regex=True)
    if req["destination"]:
        mask &= df_in["To"].fillna("").str.contains(re.escape(req["destination"]), case=False, regex=True)
    return df_in.loc[mask].copy(), (req["hours"][0], req["hours"][1])

def agg_series(s: pd.Series, how: str):
    if how == "Median": return s.median()
    if how == "P90":    return s.quantile(0.90)
    return s.mean()

# ---------------------------------------------------------------------
# CSV Q&A utilities
# ---------------------------------------------------------------------
def canonical_col_map(df_in: pd.DataFrame) -> dict:
    # map normalized key -> actual column
    norm = lambda x: re.sub(r"[^a-z0-9]+", "", x.lower())
    return {norm(c): c for c in df_in.columns}

def find_column(df_in: pd.DataFrame, user_text: str) -> str | None:
    norm = lambda x: re.sub(r"[^a-z0-9]+", "", x.lower())
    cmap = canonical_col_map(df_in)
    # try direct match
    key = norm(user_text)
    if key in cmap: return cmap[key]
    # fuzzy contains
    for k, v in cmap.items():
        if key and key in k: return v
    return None

def apply_where(df_in: pd.DataFrame, where_text: str) -> pd.DataFrame:
    """Very simple parser: supports
       - 'Flight Number = 6E1185'
       - 'From contains BOM'
       - 'Date between 2025-07-20 and 2025-07-25'
       - chain with 'and'
    """
    if not where_text: return df_in
    parts = [p.strip() for p in re.split(r"\band\b", where_text, flags=re.I)]
    out = df_in.copy()
    for p in parts:
        # between
        m = re.search(r"^([A-Za-z _\-]+)\s+between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})$", p, re.I)
        if m:
            col = find_column(out, m.group(1))
            if col and pd.api.types.is_datetime64_any_dtype(out[col]):
                d0 = pd.to_datetime(m.group(2))
                d1 = pd.to_datetime(m.group(3))
                out = out[(out[col] >= d0) & (out[col] <= d1)]
            continue
        # contains
        m = re.search(r"^([A-Za-z _\-]+)\s+contains\s+(.+)$", p, re.I)
        if m:
            col = find_column(out, m.group(1)); val = m.group(2).strip()
            if col:
                out = out[out[col].astype(str).str.contains(re.escape(val), case=False, na=False)]
            continue
        # equals
        m = re.search(r"^([A-Za-z _\-]+)\s*=\s*(.+)$", p)
        if m:
            col = find_column(out, m.group(1)); val = m.group(2).strip()
            if col:
                # try exact vs str match
                if pd.api.types.is_numeric_dtype(out[col]):
                    try:
                        num = float(val); out = out[out[col] == num]; continue
                    except: pass
                out = out[out[col].astype(str).str.fullmatch(re.escape(val), na=False)]
            continue
    return out

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.markdown("Type a question. Examples:")
st.code("""best departure hour for BOM on weekdays (median)
busiest 30 min arrival slots at BOM
what-if shift 6E1185 by +30 min (predict delay)
cascades at BOM with Tail identity, top 10
columns of the CSV
head 20 where Flight Number = 6E1185
describe dep_delay_min
value counts Airline limit 15""")

q = st.text_input("Your question", value="best departure hour for BOM (median)")
if st.button("Run"):
    req = parse_with_gemini(q) if use_gemini else parse_offline(q)
    st.caption(f"Parsed request → {req}")

    # Route by intent
    intent = req["intent"]
    metric = req["metric"]

    # CSV Q&A lane -----------------------------------------------------
    if intent == "csv":
        action = req["csv_action"] or "schema"
        col    = req["csv_column"]
        limit  = req["csv_limit"]
        where  = req["csv_where"]

        dff = apply_where(df, where) if where else df

        if action == "schema":
            nonnull = dff.notna().mean().mul(100).round(1).astype(str) + "%"
            info = pd.DataFrame({"dtype": dff.dtypes.astype(str), "nonnull%": nonnull})
            st.dataframe(info, use_container_width=True)
        elif action in {"head","sample","filter"}:
            if action == "sample":
                show = dff.sample(n=min(limit, len(dff))) if len(dff) else dff.head(0)
            else:
                show = dff.head(limit)
            st.dataframe(show, use_container_width=True)
            st.download_button("Download CSV", show.to_csv(index=False).encode("utf-8"),
                               file_name="csv_query.csv", mime="text/csv")
        elif action == "describe":
            c = find_column(dff, col) if col else None
            if not c:
                st.warning("Which column to describe? e.g., `describe dep_delay_min`")
            else:
                if pd.api.types.is_numeric_dtype(dff[c]):
                    st.dataframe(dff[c].describe().to_frame().T, use_container_width=True)
                else:
                    vc = dff[c].value_counts(dropna=False).head(limit)
                    st.dataframe(vc.to_frame("count"), use_container_width=True)
        elif action == "value_counts":
            c = find_column(dff, col) if col else None
            if not c:
                st.warning("Which column for value counts? e.g., `value counts Airline`")
            else:
                vc = dff[c].value_counts(dropna=False).head(limit)
                st.dataframe(vc.to_frame("count"), use_container_width=True)
        elif action == "count":
            st.metric("Row count", f"{len(dff):,}")
        else:
            st.info("Unknown CSV action. Try: schema, head, sample, filter, describe, value counts, count.")
        st.stop()

    # Main 4 tasks -----------------------------------------------------
    df_f, (h0, h1) = apply_filters(df, req)
    if df_f.empty:
        st.warning("No rows after applying filters.")
        st.stop()

    # Task 1: Best Hours
    if intent == "best_hours":
        t1, t2 = st.tabs(["🛫 Departures", "🛬 Arrivals"])
        with t1:
            dep = df_f[df_f["dep_hour"].between(h0, h1)]
            if dep["dep_delay_pos"].notna().sum() == 0:
                st.info("No departure delay data for this query.")
            else:
                g = dep.groupby("dep_hour")["dep_delay_pos"]
                m = g.apply(lambda s: agg_series(s, metric)).rename("delay_min")
                c = g.size().rename("flights")
                if not m.empty:
                    best_h = int(m.idxmin())
                    k1,k2,k3 = st.columns(3)
                    with k1: st.metric("⭐ Best dep hour", f"{best_h:02d}:00")
                    with k2: st.metric(f"{metric} delay @ best (min)", f"{m.min():.2f}")
                    with k3: st.metric("Flights @ best hour", f"{int(c.loc[best_h])}")
                    st.line_chart(m.rename(f"{metric} delay (min)"))
                    st.bar_chart(c.rename("Flights"))
                    st.dataframe(pd.DataFrame({"hour": m.index, f"{metric} delay (min)": m.values,
                                               "flights": c.reindex(m.index).values}), use_container_width=True)
        with t2:
            arr = df_f[df_f["arr_hour"].between(h0, h1)]
            if arr["arr_delay_pos"].notna().sum() == 0:
                st.info("No arrival delay data for this query.")
            else:
                g = arr.groupby("arr_hour")["arr_delay_pos"]
                m = g.apply(lambda s: agg_series(s, metric)).rename("delay_min")
                c = g.size().rename("flights")
                if not m.empty:
                    best_h = int(m.idxmin())
                    k1,k2,k3 = st.columns(3)
                    with k1: st.metric("⭐ Best arr hour", f"{best_h:02d}:00")
                    with k2: st.metric(f"{metric} delay @ best (min)", f"{m.min():.2f}")
                    with k3: st.metric("Flights @ best hour", f"{int(c.loc[best_h])}")
                    st.line_chart(m.rename(f"{metric} delay (min)"))
                    st.bar_chart(c.rename("Flights"))
                    st.dataframe(pd.DataFrame({"hour": m.index, f"{metric} delay (min)": m.values,
                                               "flights": c.reindex(m.index).values}), use_container_width=True)

    # Task 2: Busiest Slots
    elif intent == "busiest_slots":
        sw = req["slot_width"]
        st.write(f"Slot width: **{sw} min**")
        dep = df_f.copy()
        dep["dep_slot"] = dep["STD"].dt.floor(f"{sw}min")
        dep = dep[dep["dep_slot"].dt.hour.between(h0, h1)]
        if dep.empty:
            st.info("No departures in this window.")
        else:
            dep["slot_label"] = dep["dep_slot"].dt.strftime("%H:%M")
            counts_day = dep.groupby([dep["Date"].dt.date, "slot_label"]).size().rename("count").reset_index()
            avg_per_slot = counts_day.groupby("slot_label")["count"].mean().sort_index()
            st.bar_chart(avg_per_slot.rename("Avg flights/day"))
            st.dataframe(avg_per_slot.rename("avg_flights_per_day").reset_index(), use_container_width=True)

    # Task 3: What-If
    elif intent == "what_if":
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.pipeline import Pipeline
        except ModuleNotFoundError:
            st.error("Install scikit-learn for What-If: pip install scikit-learn")
            st.stop()

        dfw = df_f.dropna(subset=["STD","dep_delay_min"]).copy()
        if dfw.empty or len(dfw) < 50:
            st.info("Not enough rows with dep delays to train a model for this query.")
        else:
            dfw["dow"] = dfw["Date"].dt.dayofweek
            dfw["dep_hour"] = dfw["STD"].dt.hour
            dfw["is_weekend"] = (dfw["dow"] >= 5).astype(int)
            cat = [c for c in ["From","To","Airline"] if c in dfw.columns]
            num = ["dep_hour","dow","is_weekend"]

            X = dfw[cat+num]; y = dfw["dep_delay_min"].clip(lower=0)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                     ("num", "passthrough", num)])
            model = Pipeline([("pre", pre),
                              ("rf", RandomForestRegressor(n_estimators=300, random_state=42))])
            model.fit(Xtr, ytr)
            r2 = model.score(Xte, yte)
            st.success(f"RandomForest trained. R²={r2:.3f}")

            # pick row: by flight_number if specified
            row = None
            flt = req["flight_number"]
            if flt:
                cand = dfw[dfw["Flight Number"].astype(str) == flt]
                if not cand.empty:
                    row = cand.iloc[0]
            if row is None:
                row = dfw.iloc[0]

            st.write("Sample row:", row[["Date","STD","From","To","Airline","Flight Number"]])

            # Sweep around requested shift
            base_shift = int(req["shift_min"])
            shifts = list(range(base_shift-120, base_shift+121, 15))
            preds = []
            for s in shifts:
                std2 = row["STD"] + pd.Timedelta(minutes=s)
                feat = row.copy()
                feat["dep_hour"] = std2.hour
                feat["dow"] = std2.dayofweek
                feat["is_weekend"] = int(feat["dow"] >= 5)
                x = feat[cat+num].to_frame().T
                preds.append(float(model.predict(x)[0]))
            base_pred = float(model.predict(row[cat+num].to_frame().T)[0])

            out = pd.DataFrame({"shift_min": shifts, "pred_delay_min": preds})
            out["delta_vs_base_min"] = (out["pred_delay_min"] - base_pred).round(1)
            st.line_chart(out.set_index("shift_min")["pred_delay_min"])
            st.dataframe(out, use_container_width=True)

    # Task 4: Cascades
    elif intent == "cascades":
        try:
            import networkx as nx
        except ModuleNotFoundError:
            st.error("Install networkx: pip install networkx")
            st.stop()

        # choose identity keys
        ident = req["identity"]
        if ident == "Tail" and "Tail" in df_f.columns:
            KEYS = ["Tail"]
        elif ident == "Airline+From":
            KEYS = [c for c in ["Airline","From"] if c in df_f.columns] or ["Date"]
        elif ident == "Airline" and "Airline" in df_f.columns:
            KEYS = ["Airline"]
        elif ident == "From" and "From" in df_f.columns:
            KEYS = ["From"]
        else:
            KEYS = ["Date"]

        turn_min, turn_max = req["turn_min"], req["turn_max"]
        cap = 60.0

        dff = df_f.dropna(subset=["STD","ATA"]).copy()
        dff["flight_id"] = dff["Date"].dt.strftime("%Y-%m-%d") + "_" + dff["Flight Number"].astype(str)

        G = nx.DiGraph()
        for _, grp in dff.groupby(KEYS, dropna=False):
            grp = grp.sort_values("ATA")
            for i in range(len(grp)):
                ai = grp.iloc[i]
                a_delay = max(0.0, float(ai.get("dep_delay_min", 0) or 0))
                if a_delay <= 0:  # ignore non-positive
                    continue
                for j in range(i+1, len(grp)):
                    bj = grp.iloc[j]
                    gap = (bj["STD"] - ai["ATA"]).total_seconds()/60.0
                    if gap < turn_min: continue
                    if gap > turn_max: break
                    w = min(1.0, a_delay/cap)
                    if w > 0:
                        G.add_edge(ai["flight_id"], bj["flight_id"], weight=w)

        if G.number_of_edges() == 0:
            st.info("No cascading edges formed with current settings.")
            st.stop()

        out_strength = {n: sum(d["weight"] for _,_,d in G.out_edges(n, data=True)) for n in G.nodes()}
        top = sorted(out_strength.items(), key=lambda x: x[1], reverse=True)[:req["top_k"]]

        rows = []
        for fid, sc in top:
            r = dff[dff["flight_id"] == fid].iloc[0]
            rows.append({
                "flight_id": fid,
                "Flight Number": r.get("Flight Number","?"),
                "Route": f"{r.get('From','?')}→{r.get('To','?')}",
                "dep_delay_min": round(float(r.get("dep_delay_min", np.nan)),1),
                "influence": round(float(sc),2)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
