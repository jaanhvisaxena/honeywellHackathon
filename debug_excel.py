#!/usr/bin/env python3
"""
Debug script to examine Excel file and see why only 3 rows are processed
"""

import pandas as pd
from pathlib import Path

def debug_excel():
    file_path = Path("data/Flight_Data.xlsx")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"🔍 Examining Excel file: {file_path}")
    print("=" * 50)
    
    # Read Excel file info
    xls = pd.ExcelFile(file_path)
    print(f"📊 Found {len(xls.sheet_names)} sheets: {xls.sheet_names}")
    
    for sheet_name in xls.sheet_names:
        print(f"\n📋 Sheet: {sheet_name}")
        print("-" * 30)
        
        # Read raw data (no header)
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        print(f"Raw rows: {len(raw_df)}")
        print(f"Raw columns: {len(raw_df.columns)}")
        
        # Show first 5 rows
        print("\nFirst 5 rows (raw):")
        print(raw_df.head().to_string())
        
        # Try to find header row
        print(f"\n🔍 Looking for header row...")
        for i in range(min(10, len(raw_df))):
            row_values = [str(v).lower() for v in raw_df.iloc[i].tolist()]
            # Check if this row contains flight-related keywords
            keywords = ["flight", "from", "to", "std", "atd", "sta", "ata", "date"]
            matches = sum(1 for val in row_values for kw in keywords if kw in val)
            print(f"Row {i}: {matches} matches - {row_values[:5]}")
            
            if matches >= 3:
                print(f"✅ Found header at row {i}")
                break
        
        # Try reading with different header positions
        print(f"\n📖 Testing different header positions:")
        for header_row in [0, 1, 2, 3]:
            try:
                test_df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
                print(f"Header row {header_row}: {len(test_df)} rows, columns: {list(test_df.columns)[:5]}")
            except Exception as e:
                print(f"Header row {header_row}: Error - {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    debug_excel()
