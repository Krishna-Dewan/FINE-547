"""
Main script to run missing data analysis on sample.csv
"""
import pandas as pd
import os
from data_loader_helper import DataLoader, run_complete_workflow_on_your_data

# Load API key from environment variable (recommended)
claude_api_key = os.getenv('CLAUDE_API_KEY')
if claude_api_key:
    print("✓ Claude API key loaded from environment")
else:
    print("⚠️ No Claude API key found - will run detection only")
    # Uncomment and add your key directly here if not using environment variable:
    # claude_api_key = 'sk-ant-your-api-key-here'


# ============================================================================
# OPTION 1: Efficient Single-Pass Detection (NEW!)
# ============================================================================

print("Starting efficient single-pass event detection...")

# Use the new efficient detector
from efficient_single_pass_detector import run_efficient_detection

results = run_efficient_detection(
    data_file='/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa.parquet',
    claude_api_key=claude_api_key,  # For compatibility
    bankruptcy_threshold=55,        # Much more selective for bankruptcy
    merger_threshold=50,            # Much more selective for M&A
    halt_threshold=999,             # Disabled (set very high)
    save_every_n_stocks=999999,    # Disable progress saving - only save final result
    verbose=True,
    output_file='ma_bankruptcy_events.csv'  # Better filename for M&A/bankruptcy focus
)

print("\n✅ M&A and Bankruptcy detection complete!")
print(f"📁 Results: ma_bankruptcy_events.csv")
if len(results) > 0:
    print(f"🎯 Found {len(results)} M&A/bankruptcy events (one per stock max)")
    print(f"📊 Ready for your claude_verifier!")
    # Show breakdown by event type
    print("\n📊 Event breakdown:")
    for event_type, count in results['event_type'].value_counts().items():
        print(f"   {event_type}: {count}")
else:
    print(f"✅ No events flagged - thresholds working correctly!")


# ============================================================================
# OPTION 2: Step-by-Step (Manual Control)
# ============================================================================

# Uncomment below if you want more control:

"""
# Step 1: Load your data
loader = DataLoader('sample.csv')
raw_data = loader.load_raw_data()

# Step 2: Process it
processed_data = loader.process_for_pipeline()

# Step 3: Check what features you have
feature_check = loader.check_feature_availability()
print(feature_check)

# Step 4: Analyze missing data patterns
patterns = loader.detect_missing_data_patterns()
print(patterns)

# Step 5: Get statistics
stats = loader.get_sample_statistics()
print(stats)

# Step 6: Run detection (when ready)
from complete_pipeline import CompleteMissingDataPipeline

pipeline = CompleteMissingDataPipeline(
    price_data=processed_data,
    claude_api_key=None,  # Add later
    dataset_start_date=processed_data['date'].min()
)

detected = pipeline.step1_detect_all_events()
detected.to_csv('detected_events.csv', index=False)
"""

