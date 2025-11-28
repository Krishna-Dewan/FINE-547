#!/usr/bin/env python3
"""
Label ret_sample_usa.parquet with ACQUISITION events only
using verified data from ma_bankruptcy_events_merged_v2.csv

CRITICAL REQUIREMENTS:
1. Only use ACQUISITIONS (not mergers)
2. Exclude ALL December 2014 events
3. Drop duplicates (same gvkey + actual_date)
4. Label the closest month BEFORE actual_date as is_acquired=1
5. All subsequent months for that stock remain 0 (firm disappears after acquisition)
"""

import pandas as pd
import numpy as np


def load_data():
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    # Load target panel data
    panel = pd.read_parquet(
        "/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa.parquet"
    )
    print(f"✓ Loaded panel: {panel.shape[0]:,} rows")

    # Load verified merged events file
    events = pd.read_csv(
        "/Users/krishna_dewan/Desktop/FINE 547 M&A/Find acquisitions/ma_bankruptcy_events_merged_v3.csv"
    )
    print(f"✓ Loaded events: {events.shape[0]:,} rows")

    return panel, events


def prepare_events(events):
    print("\n" + "=" * 80)
    print("FILTERING VERIFIED ACQUISITIONS ONLY")
    print("=" * 80)

    print(f"\nInitial events: {len(events):,}")

    # Convert verified to boolean
    events["verified"] = events["verified"].astype(str).str.upper() == "TRUE"

    # Filter 1: Verified and acquisition only (NOT merger)
    valid = events[
        (events["verified"] == True) &
        (events["actual_event_type"].str.lower() == "acquisition")
        ].copy()

    print(f"✓ After verified==True & actual_event_type=='acquisition': {len(valid):,}")

    # Convert actual_date to datetime
    valid["actual_date"] = pd.to_datetime(valid["actual_date"], errors="coerce")

    # Filter 2: Must have actual_date
    valid = valid[valid["actual_date"].notna()].copy()
    print(f"✓ After actual_date not null: {len(valid):,}")

    # Filter 3: Exclude ALL of December 2014
    dec_2014_mask = (
            (valid["actual_date"].dt.year == 2014) &
            (valid["actual_date"].dt.month == 12)
    )

    excluded_dec_2014 = dec_2014_mask.sum()
    valid = valid[~dec_2014_mask].copy()
    print(f"✓ Excluded {excluded_dec_2014:,} events from December 2014")
    print(f"✓ After excluding Dec 2014: {len(valid):,}")

    # Filter 4: Drop duplicates (same gvkey + actual_date)
    before_dedup = len(valid)
    valid = valid.drop_duplicates(subset=["gvkey", "actual_date"], keep="first")
    duplicates_removed = before_dedup - len(valid)

    if duplicates_removed > 0:
        print(f"✓ Removed {duplicates_removed:,} duplicate events (same gvkey + actual_date)")

    print(f"\n✅ FINAL clean acquisition events: {len(valid):,}")

    # Keep only needed fields
    valid = valid[[
        "gvkey",
        "actual_date",
        "actual_event_type",
        "acquirer",
        "acquisition_price"
    ]].copy()

    # Show distribution by year
    print(f"\n📊 Acquisitions by year:")
    year_dist = valid["actual_date"].dt.year.value_counts().sort_index()
    for year, count in year_dist.items():
        if pd.notna(year):
            print(f"  {int(year)}: {count:,}")

    print("\n📋 Sample verified events:")
    print(valid.head(10).to_string())

    return valid


def label_panel(panel, events):
    print("\n" + "=" * 80)
    print("LABELING PANEL WITH ACQUISITIONS")
    print("=" * 80)

    df = panel.copy()

    # Convert date to datetime
    print("\nConverting dates...")
    df["date"] = df["date"].astype(int).astype(str)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # Sort by firm and date
    df = df.sort_values(['gvkey', 'date'])

    # Initialize new columns
    df["is_acquired"] = 0
    df["acquisition_date"] = pd.NaT
    df["days_to_acquisition"] = np.nan
    df["acquirer"] = None
    df["acquisition_price"] = np.nan
    df["event_type_verified"] = None

    print(f"\nProcessing {len(events):,} acquisition events...")

    events_matched = 0
    events_no_match = 0

    # Process each verified acquisition event
    for idx, ev in events.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(events)} events...")

        gv = ev["gvkey"]
        acq_date = ev["actual_date"]

        # Get all rows for this firm
        firm_rows = df[df["gvkey"] == gv].copy()

        if firm_rows.empty:
            events_no_match += 1
            continue

        # Compute days until acquisition
        # Positive = days in the FUTURE (before acquisition)
        # Negative = days in the PAST (after acquisition)
        days_diff = (acq_date - firm_rows["date"]).dt.days

        # ONLY keep rows BEFORE the acquisition
        mask_before = days_diff > 0

        if mask_before.sum() == 0:
            events_no_match += 1
            continue

        events_matched += 1

        # Get indices of all months BEFORE acquisition
        before_indices = firm_rows.index[mask_before]

        # Fill metadata for ALL months before acquisition
        df.loc[before_indices, "days_to_acquisition"] = days_diff[mask_before].values
        df.loc[before_indices, "acquisition_date"] = acq_date
        df.loc[before_indices, "acquirer"] = ev["acquirer"]
        df.loc[before_indices, "acquisition_price"] = ev["acquisition_price"]
        df.loc[before_indices, "event_type_verified"] = ev["actual_event_type"]

        # Find the CLOSEST month BEFORE acquisition
        # This is the month we'll label as is_acquired=1
        min_days_idx = days_diff[mask_before].argmin()
        closest_idx = before_indices[min_days_idx]

        # Mark ONLY this closest month as acquired
        df.loc[closest_idx, "is_acquired"] = 1

    print(f"\n✓ Labeling complete")
    print(f"  Events matched to panel: {events_matched:,} / {len(events):,}")
    print(f"  Events not in panel: {events_no_match:,}")

    # Summary statistics
    total_acquired = (df['is_acquired'] == 1).sum()
    print(f"\n📊 Label Summary:")
    print(f"  Total months labeled is_acquired=1: {total_acquired:,}")
    print(f"  Total months is_acquired=0: {(df['is_acquired'] == 0).sum():,}")
    print(f"  Positive rate: {total_acquired / len(df) * 100:.4f}%")

    # Distribution by year
    df_temp = df[df['is_acquired'] == 1].copy()
    df_temp['year'] = df_temp['date'].dt.year

    print(f"\n📅 Acquisitions labeled by year:")
    acq_by_year = df_temp.groupby('year').size().sort_index()
    for year, count in acq_by_year.items():
        if pd.notna(year):
            print(f"  {int(year)}: {count:,}")

    # Sanity check: Each firm should have at most 1 acquisition
    firms_with_multiple = df[df['is_acquired'] == 1].groupby('gvkey').size()
    multiple_acqs = (firms_with_multiple > 1).sum()

    if multiple_acqs > 0:
        print(f"\n⚠️  WARNING: {multiple_acqs} firms have multiple is_acquired=1 labels")
        print(f"   This might indicate:")
        print(f"   - Same firm acquired multiple times (rare but possible)")
        print(f"   - Duplicate events not properly cleaned")
        print(f"\n   Top firms with multiple acquisitions:")
        for gvkey, count in firms_with_multiple.nlargest(10).items():
            print(f"      GVKEY {gvkey}: {count} acquisitions")
    else:
        print(f"\n✓ Sanity check passed: Each firm has at most 1 acquisition")

    return df


def save(df):
    print("\n" + "=" * 80)
    print("SAVING LABELED DATASET")
    print("=" * 80)

    out_parquet = "/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa_labeled_3.parquet"
    out_csv = ("/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa_labeled_3"
               ".csv")

    # Save parquet (efficient format)
    print(f"💾 Saving parquet...")
    df.to_parquet(out_parquet, index=False)
    print(f"✓ Saved parquet: {out_parquet}")

    # Save CSV (this will take longer for large files)
    print(f"\n💾 Saving CSV (this may take a minute for {len(df):,} rows)...")
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved CSV: {out_csv}")

    # Save a summary
    summary = {
        'total_rows': int(len(df)),
        'total_acquired': int((df['is_acquired'] == 1).sum()),
        'positive_rate': float((df['is_acquired'] == 1).sum() / len(df) * 100),
        'years_covered': [int(y) for y in sorted(df[df['is_acquired'] == 1]['date'].dt.year.unique())],
        'filters_applied': [
            'Only acquisitions (not mergers)',
            'Excluded all December 2014',
            'Dropped duplicates (gvkey + actual_date)',
            'Labeled closest month before acquisition',
            'Only verified events with actual_date'
        ]
    }

    import json
    summary_path = "/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/labeling_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")

    print(f"\n📁 Output files:")
    print(f"  • {out_parquet}")
    print(f"  • {out_csv}")
    print(f"  • {summary_path}")


def main():
    print("\n" + "=" * 80)
    print("M&A PANEL LABELING - ACQUISITIONS ONLY")
    print("=" * 80)

    # Load data
    panel, events = load_data()

    # Filter to clean acquisition events
    clean_events = prepare_events(events)

    # Label panel
    labeled = label_panel(panel, clean_events)

    # Save results (both parquet and CSV)
    save(labeled)

    print("\n" + "=" * 80)
    print("✅ DONE - Dataset labeled with ACQUISITIONS only!")
    print("=" * 80)
    print("\nKEY FILTERS APPLIED:")
    print("  1. ✓ Only acquisitions (not mergers)")
    print("  2. ✓ Excluded all December 2014")
    print("  3. ✓ Dropped duplicates (same gvkey + actual_date)")
    print("  4. ✓ Labeled closest month BEFORE actual_date")
    print("  5. ✓ Only verified events with actual_date")
    print("\nNEXT STEPS:")
    print("  - Rebuild your model with this clean data")
    print("  - Use lagged features (not contemporaneous)")
    print("  - Expect realistic performance metrics")


if __name__ == "__main__":
    main()