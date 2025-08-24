# what_if_core.py
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

CANDIDATE_CATS = ["From","To","Airline"]

def build_hour_load_reference(df: pd.DataFrame):
    D = df.copy()
    D["dep_hour"] = D["STD"].dt.hour
    D["DateOnly"] = D["Date"].dt.date
    count_col = "Flight Number" if "Flight Number" in D.columns else None

    if "From" in D.columns:
        ref_daily = (D.groupby(["DateOnly","From","dep_hour"])[count_col].count().rename("cnt").reset_index()
                     if count_col else
                     D.groupby(["DateOnly","From","dep_hour"]).size().rename("cnt").reset_index())
        days = max(1, D["DateOnly"].nunique())
        ref_hist = (D.groupby(["From","dep_hour"])[count_col].count().div(days).rename("avg_cnt").reset_index()
                    if count_col else
                    D.groupby(["From","dep_hour"]).size().div(days).rename("avg_cnt").reset_index())
    else:
        ref_daily = (D.groupby(["DateOnly","dep_hour"])[count_col].count().rename("cnt").reset_index()
                     if count_col else
                     D.groupby(["DateOnly","dep_hour"]).size().rename("cnt").reset_index())
        days = max(1, D["DateOnly"].nunique())
        ref_hist = (D.groupby(["dep_hour"])[count_col].count().div(days).rename("avg_cnt").reset_index()
                    if count_col else
                    D.groupby(["dep_hour"]).size().div(days).rename("avg_cnt").reset_index())
    return ref_daily, ref_hist

def get_hour_load(row: pd.Series, shifted_hour: int, ref_daily: pd.DataFrame, ref_hist: pd.DataFrame) -> float:
    date_key = row["Date"].date()
    if "From" in row.index and "From" in ref_daily.columns:
        m = (ref_daily["DateOnly"] == date_key) & (ref_daily["From"] == row["From"]) & (ref_daily["dep_hour"] == shifted_hour)
        if m.any(): return float(ref_daily.loc[m, "cnt"].iloc[0])
        m2 = (ref_hist["From"] == row["From"]) & (ref_hist["dep_hour"] == shifted_hour)
        if m2.any(): return float(ref_hist.loc[m2, "avg_cnt"].iloc[0])
    else:
        m = (ref_daily["DateOnly"] == date_key) & (ref_daily["dep_hour"] == shifted_hour)
        if m.any(): return float(ref_daily.loc[m, "cnt"].iloc[0])
        m2 = (ref_hist["dep_hour"] == shifted_hour)
        if m2.any(): return float(ref_hist.loc[m2, "avg_cnt"].iloc[0])
    return 0.0

def train_model(df: pd.DataFrame):
    D = df.dropna(subset=["Date","STD","dep_delay_min"]).copy()
    D["dep_delay_pos"] = D["dep_delay_min"].clip(lower=0)
    D["dep_hour"] = D["STD"].dt.hour
    D["dow"] = D["Date"].dt.dayofweek
    D["is_weekend"] = (D["dow"] >= 5).astype(int)

    ref_daily, ref_hist = build_hour_load_reference(D)
    D["hour_load"] = [get_hour_load(r, int(r["dep_hour"]), ref_daily, ref_hist) for _, r in D.iterrows()]

    cat_cols = [c for c in CANDIDATE_CATS if c in D.columns]
    num_cols = ["dep_hour","dow","is_weekend","hour_load"]
    X, y = D[cat_cols + num_cols], D["dep_delay_pos"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                             ("num", "passthrough", num_cols)])
    model = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=300, random_state=42))]).fit(Xtr, ytr)

    r2 = model.score(Xte, yte)
    y_pred = model.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
    mae  = float(mean_absolute_error(yte, y_pred))

    meta = {"cat_cols": cat_cols, "num_cols": num_cols, "r2": float(r2), "rmse": rmse, "mae": mae,
            "ref_daily": ref_daily, "ref_hist": ref_hist}
    return model, meta

def predict_delay_for_shift(row: pd.Series, shift_minutes: int, model, meta) -> float:
    shifted_std = row["STD"] + pd.Timedelta(minutes=int(shift_minutes))
    x = {"dep_hour": shifted_std.hour,
         "dow": shifted_std.dayofweek,
         "is_weekend": int(shifted_std.dayofweek >= 5),
         "hour_load": get_hour_load(row, shifted_std.hour, meta["ref_daily"], meta["ref_hist"])}
    for c in meta["cat_cols"]:
        x[c] = row.get(c, np.nan)
    X = pd.DataFrame([x])[meta["cat_cols"] + meta["num_cols"]]
    return float(model.predict(X)[0])
