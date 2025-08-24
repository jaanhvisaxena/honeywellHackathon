# what_if_delay.py
# Full What-If schedule tuning demo for flight delays.
# Requires: pandas, numpy, scikit-learn. (matplotlib optional for plotting)

import argparse
import sys
import numpy as np
import pandas as pd

# ==== CONFIG ====
CSV_PATH = r"C:\college\honeywell\data\flight_data_clean.csv"  # <-- change if needed
DATE_COLS = ["Date","STD","ATD","STA","ATA"]
TARGET_COL = "dep_delay_min"   # we predict departure delay (clipped at 0)
CANDIDATE_CATS = ["From","To","Airline"]  # use whichever exist
SHIFT_RANGE = range(-120, 121, 15)        # minutes

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[c for c in DATE_COLS if c in pd.read_csv(csv_path, nrows=1).columns])
    # Normalize/trim column names
    df.columns = [c.strip() for c in df.columns]
    # Sanity check required columns
    missing = [c for c in ["Date","STD"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Target: clip at 0 (no negative delay for early departures)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")
    df["y"] = df[TARGET_COL].clip(lower=0)
    # Basic features
    df["dep_hour"] = df["STD"].dt.hour
    df["dow"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def build_hour_load_reference(DF: pd.DataFrame):
    """Build per-day + historical hour-load references used as congestion proxy."""
    df = DF.copy()
    df["dep_hour"] = df["dep_hour"] if "dep_hour" in df.columns else df["STD"].dt.hour
    df["DateOnly"] = df["Date"].dt.date

    # We count "flights" per (date, origin, hour). If Flight Number missing, count rows.
    count_col = "Flight Number" if "Flight Number" in df.columns else None

    if "From" in df.columns:
        if count_col:
            ref_daily = (df.groupby(["DateOnly","From","dep_hour"])[count_col]
                           .count().rename("cnt").reset_index())
        else:
            ref_daily = (df.groupby(["DateOnly","From","dep_hour"])
                           .size().rename("cnt").reset_index())

        # Historical average per (From, dep_hour)
        days = df["DateOnly"].nunique() if df["DateOnly"].nunique() > 0 else 1
        if count_col:
            ref_hist = (df.groupby(["From","dep_hour"])[count_col].count()
                        .div(days).rename("avg_cnt").reset_index())
        else:
            ref_hist = (df.groupby(["From","dep_hour"]).size()
                        .div(days).rename("avg_cnt").reset_index())
    else:
        if count_col:
            ref_daily = (df.groupby(["DateOnly","dep_hour"])[count_col]
                           .count().rename("cnt").reset_index())
        else:
            ref_daily = (df.groupby(["DateOnly","dep_hour"])
                           .size().rename("cnt").reset_index())

        days = df["DateOnly"].nunique() if df["DateOnly"].nunique() > 0 else 1
        if count_col:
            ref_hist = (df.groupby(["dep_hour"])[count_col].count()
                        .div(days).rename("avg_cnt").reset_index())
        else:
            ref_hist = (df.groupby(["dep_hour"]).size()
                        .div(days).rename("avg_cnt").reset_index())

    return ref_daily, ref_hist

def get_hour_load(row: pd.Series, shifted_hour: int, ref_daily: pd.DataFrame, ref_hist: pd.DataFrame) -> float:
    """Return congestion proxy for row's (date, from, hour), with historical fallback."""
    date_key = row["Date"].date()
    if "From" in row.index and "From" in ref_daily.columns:
        m = (ref_daily["DateOnly"] == date_key) & (ref_daily["From"] == row["From"]) & (ref_daily["dep_hour"] == shifted_hour)
        if m.any():
            return float(ref_daily.loc[m, "cnt"].iloc[0])
        m2 = (ref_hist["From"] == row["From"]) & (ref_hist["dep_hour"] == shifted_hour)
        if m2.any():
            return float(ref_hist.loc[m2, "avg_cnt"].iloc[0])
    else:
        m = (ref_daily["DateOnly"] == date_key) & (ref_daily["dep_hour"] == shifted_hour)
        if m.any():
            return float(ref_daily.loc[m, "cnt"].iloc[0])
        m2 = (ref_hist["dep_hour"] == shifted_hour)
        if m2.any():
            return float(ref_hist.loc[m2, "avg_cnt"].iloc[0])
    return 0.0

def train_model(df: pd.DataFrame):
    """Train RandomForest on available features; return (pipeline, feature names)."""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline

    # Keep rows with target
    data = df.dropna(subset=["y", "STD", "Date"]).copy()

    # Select categoricals that actually exist
    cat_cols = [c for c in CANDIDATE_CATS if c in data.columns]
    # hour_load computed later on-the-fly for what-if; but include the observed load during training
    # Build observed hour_load for training set (using same-day same-origin)
    ref_daily, ref_hist = build_hour_load_reference(data)
    data["hour_load"] = [
        get_hour_load(row, int(row["dep_hour"]), ref_daily, ref_hist) for _, row in data.iterrows()
    ]

    num_cols = ["dep_hour","dow","is_weekend","hour_load"]
    use_cols = cat_cols + num_cols

    X = data[use_cols]
    y = data["y"]
    if len(X) < 50:
        raise ValueError("Not enough training rows with target 'y'. Need at least 50.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
    ])
    model.fit(Xtr, ytr)

    # Evaluation
    r2 = model.score(Xte, yte)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    y_pred = model.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
    mae  = float(mean_absolute_error(yte, y_pred))

    metrics = {"R2": float(r2), "RMSE_min": rmse, "MAE_min": mae}

    # Also return references for what-if reuse
    return model, cat_cols, num_cols, ref_daily, ref_hist, metrics

def predict_delay_for_shift(row: pd.Series, shift_minutes: int,
                            model, cat_cols, num_cols,
                            ref_daily: pd.DataFrame, ref_hist: pd.DataFrame) -> float:
    """Predict delay (min) when shifting STD by `shift_minutes` for a given row."""
    shifted_std = row["STD"] + pd.Timedelta(minutes=int(shift_minutes))
    x = {
        "dep_hour": shifted_std.hour,
        "dow": shifted_std.dayofweek,
        "is_weekend": int(shifted_std.dayofweek >= 5),
        "hour_load": get_hour_load(row, shifted_std.hour, ref_daily, ref_hist),
    }
    for c in cat_cols:
        x[c] = row.get(c, np.nan)
    # arrange in the order model expects (cat before num as in training)
    feature_order = cat_cols + [c for c in num_cols]  # num_cols already contains hour_load etc.
    X = pd.DataFrame([x])[feature_order]
    pred = float(model.predict(X)[0])
    return max(0.0, pred)  # clip negative just in case

def main():
    parser = argparse.ArgumentParser(description="What-If: shift a flight and predict delay.")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to cleaned CSV")
    parser.add_argument("--row", type=int, default=0, help="Row index to simulate on (0-based)")
    parser.add_argument("--min", dest="minshift", type=int, default=-120, help="Min shift (minutes)")
    parser.add_argument("--max", dest="maxshift", type=int, default=120, help="Max shift (minutes)")
    parser.add_argument("--step", type=int, default=15, help="Step (minutes)")
    parser.add_argument("--plot", action="store_true", help="Plot the What-If curve (requires matplotlib)")
    args = parser.parse_args()

    print(f"Loading data from: {args.csv}")
    df = load_data(args.csv)

    # Train model
    print("Training model...")
    model, cat_cols, num_cols, ref_daily, ref_hist, metrics = train_model(df)
    print(f"Model metrics: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE_min']:.2f} min, MAE={metrics['MAE_min']:.2f} min")

    # Pick sample row
    if args.row < 0 or args.row >= len(df):
        print(f"Row index out of range 0..{len(df)-1}", file=sys.stderr)
        sys.exit(1)

    row = df.iloc[args.row]
    if pd.isna(row["STD"]) or pd.isna(row["Date"]):
        print("Selected row is missing STD/Date; choose another row.", file=sys.stderr)
        sys.exit(1)

    # Baseline prediction (no shift)
    base_pred = predict_delay_for_shift(row, 0, model, cat_cols, num_cols, ref_daily, ref_hist)
    print(f"\nSample row #{args.row}:")
    show_cols = ["Date","STD","From","To","Airline","Flight Number"]  # print if present
    available = [c for c in show_cols if c in df.columns]
    print(row[available].to_string())
    print(f"Baseline predicted delay: {base_pred:.1f} min")

    # What-If over range
    shifts = list(range(args.minshift, args.maxshift + 1, args.step))
    preds = [predict_delay_for_shift(row, s, model, cat_cols, num_cols, ref_daily, ref_hist) for s in shifts]
    out = pd.DataFrame({"shift_min": shifts, "pred_delay_min": preds})
    out["delta_vs_base_min"] = out["pred_delay_min"] - base_pred

    print("\nWhat-If (shift vs predicted delay):")
    print(out.to_string(index=False, justify="center", col_space=14, formatters={
        "pred_delay_min": "{:.1f}".format, "delta_vs_base_min": "{:+.1f}".format
    }))

    # Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,5))
            plt.plot(out["shift_min"], out["pred_delay_min"], marker="o")
            plt.axvline(0, linestyle="--")
            plt.title("Predicted Delay vs Departure Time Shift")
            plt.xlabel("Shift (minutes, + = later STD)")
            plt.ylabel("Predicted Delay (min)")
            plt.grid(True, alpha=0.3)
            plt.show()
        except ModuleNotFoundError:
            print("matplotlib not installed; skipping plot. Install via: pip install matplotlib")

if __name__ == "__main__":
    main()
