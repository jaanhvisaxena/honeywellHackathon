#!/usr/bin/env python3
"""
Generate fixed notebook code for Flight_Scheduling_Analysis.ipynb
"""

fixed_code = '''# =======================
# PHASE 1 — CLEAN PIPELINE (FIXED - No Warnings, All Sheets)
# =======================
import pandas as pd, numpy as np, re
from pathlib import Path

FILE = Path("../data/Flight_Data.xlsx")   # adjust if needed
OUT_CSV = FILE.parent / "flight_data_clean.csv"

# --- regex for time extraction ---
clock12 = re.compile(r"(\\d{1,2}:\\d{2}\\s*[AP]M)", flags=re.I)
clock24 = re.compile(r"(\\d{1,2}:\\d{2})(?::\\d{2})?")

def extract_clock_str(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = clock12.search(s)
    if m: return pd.to_datetime(m.group(1)).strftime("%H:%M:%S")
    m = clock24.search(s)
    if m: return pd.to_datetime(m.group(1)).strftime("%H:%M:%S")
    try:
        return pd.to_datetime(s, errors="raise").strftime("%H:%M:%S")
    except Exception:
        return np.nan

def force_date_with_time(date_s, time_s):
    dpart = pd.to_datetime(date_s, errors="coerce").dt.strftime("%Y-%m-%d")
    tpart = time_s.apply(extract_clock_str)
    combo = dpart + " " + tpart.fillna("00:00:00")
    out = pd.to_datetime(combo, errors="coerce")
    out[tpart.isna()] = pd.NaT
    return out

# --- Load both sheets ---
xls = pd.ExcelFile(FILE)
print(f"📊 Found {len(xls.sheet_names)} sheets: {xls.sheet_names}")

frames = []
for sh in xls.sheet_names:
    df = pd.read_excel(FILE, sheet_name=sh)
    
    # Fix the date column issue - rename Unnamed: 2 to Date if needed
    if 'Unnamed: 2' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'Unnamed: 2': 'Date'})
        print(f"✅ Renamed 'Unnamed: 2' to 'Date' in sheet: {sh}")
    
    df["__sheet__"] = sh
    frames.append(df)
    print(f"📋 Sheet {sh}: {len(df)} rows")

df = pd.concat(frames, ignore_index=True)
print(f"🔗 Combined total: {len(df)} rows")

# --- Clean & forward-fill ---
# Clean strings (fixed applymap deprecation warning)
df = df.map(lambda x: str(x).replace("\\xa0", "").strip() if isinstance(x,str) else x)

# Forward-fill Date and Flight Number *before* parsing
if "Date" in df.columns:
    df["Date"] = df["Date"].replace({"": np.nan, "nan": np.nan}).ffill()
if "Flight Number" in df.columns:
    df["Flight Number"] = df["Flight Number"].replace({"": np.nan, "nan": np.nan}).ffill()

# Parse Date properly
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# Clean ATA like "Landed 11:02 AM"
if "ATA" in df.columns:
    df["ATA"] = df["ATA"].astype(str).str.replace(r"(?i)^\\s*landed\\s*", "", regex=True).replace({"": np.nan})

# Build datetime columns
for col in ["STD","ATD","STA","ATA"]:
    if col in df.columns:
        df[col] = force_date_with_time(df["Date"], df[col])

# Compute delays
df["dep_delay_min"] = (df["ATD"] - df["STD"]).dt.total_seconds()/60
df["arr_delay_min"] = (df["ATA"] - df["STA"]).dt.total_seconds()/60

# Final clean
df = df[df["Date"].notna()].copy()

# Show summary
print(f"\\n📊 FINAL SUMMARY:")
print(f"Total rows: {len(df)}")
print(f"Sheets processed: {df['__sheet__'].nunique()}")
print(f"Rows per sheet:")
print(df.groupby('__sheet__').size())

# Save
df.to_csv(OUT_CSV, index=False)
print(f"\\n✅ Saved clean CSV with {len(df)} rows → {OUT_CSV}")
'''

print("Copy this code into your notebook to fix all issues:")
print("=" * 60)
print(fixed_code)
print("=" * 60)
print("This fixed version:")
print("1. ✅ Handles both sheets properly")
print("2. ✅ Renames 'Unnamed: 2' to 'Date' for the first sheet")
print("3. ✅ Fixes the applymap deprecation warning")
print("4. ✅ Shows detailed processing information")
print("5. ✅ Should process all ~1000+ rows from both sheets")
