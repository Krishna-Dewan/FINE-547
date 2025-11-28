import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import os


class DataLoader:
    """
    Loads and prepares your specific CSV format for the missing data pipeline.

    Your format:
    - Column A: Index (no header)
    - Columns B-L: Identifiers and dates
    - Columns M+: Features
    """

    def __init__(self, filepath: str = '/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa.parquet', max_memory_gb: float = 1.0):
        """
        Initialize loader with your parquet file.

        Args:
            filepath: Path to your ret_sample.parquet
            max_memory_gb: Maximum memory to use (GB). Triggers chunked processing if exceeded.
        """
        self.filepath = filepath
        self.max_memory_gb = max_memory_gb
        self.raw_data = None
        self.processed_data = None
        self.is_large_file = False
        self.chunk_results = []

    def load_raw_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load the raw CSV/Parquet file with smart chunking for large files.
        """
        if verbose:
            print(f"Loading data from: {self.filepath}")

        # Read file - handle both parquet and CSV formats
        import os
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        file_size_gb = os.path.getsize(self.filepath) / (1024**3)
        if verbose:
            print(f"File size: {file_size_gb:.2f} GB")

        # Auto-detect if we need chunked processing for large files
        if file_size_gb > self.max_memory_gb:
            if verbose:
                print(f"🔄 Large file detected (>{self.max_memory_gb}GB) - using chunked processing")
            return self.load_raw_data_chunked(verbose=verbose)

        # Determine file type and read accordingly
        if self.filepath.lower().endswith('.csv'):
            if verbose:
                print("Reading CSV file...")
            self.raw_data = pd.read_csv(self.filepath)

        elif self.filepath.lower().endswith('.parquet'):
            if verbose:
                print(f"Reading parquet file...")

            try:
                # For smaller files, try direct reading first
                if file_size_gb < 0.5:  # Less than 500MB
                    self.raw_data = pd.read_parquet(self.filepath)
                else:
                    # For files 500MB-1GB, use batch reading to be safe
                    if verbose:
                        print("Using batch reading for safety...")
                    return self._load_parquet_in_batches(verbose=verbose)

            except Exception as e:
                if verbose:
                    print(f"Direct reading failed: {e}")
                    print("Falling back to batch reading...")
                return self._load_parquet_in_batches(verbose=verbose)

        else:
            raise ValueError(f"Unsupported file format: {self.filepath}. Use .csv or .parquet files.")

        if verbose:
            print(f"✓ Loaded {len(self.raw_data):,} rows")
            print(f"✓ Columns: {len(self.raw_data.columns)}")

        return self.raw_data

    def _load_parquet_in_batches(self, batch_size: int = 500000, verbose: bool = True) -> pd.DataFrame:
        """
        Load parquet file in row-based batches (safer for large files).
        """
        if verbose:
            print(f"Loading parquet in batches of {batch_size:,} rows...")

        import pyarrow.parquet as pq

        try:
            parquet_file = pq.ParquetFile(self.filepath)

            if verbose:
                print(f"Total rows: {parquet_file.metadata.num_rows:,}")
                print(f"Columns: {len(parquet_file.schema)}")

            # Read in batches
            all_batches = []
            total_rows = 0

            for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
                batch_df = batch.to_pandas()
                all_batches.append(batch_df)
                total_rows += len(batch_df)

                if verbose and (i + 1) % 5 == 0:  # Progress every 5 batches
                    print(f"  Loaded batch {i+1}: {total_rows:,} rows so far...")

            # Combine all batches
            if verbose:
                print("Combining all batches...")

            self.raw_data = pd.concat(all_batches, ignore_index=True)

            if verbose:
                print(f"✓ Successfully loaded {len(self.raw_data):,} rows")
                print(f"✓ Columns: {len(self.raw_data.columns)}")

        except Exception as e:
            if verbose:
                print(f"Batch reading failed: {e}")
                print("This file may be corrupted or inaccessible.")
            raise FileNotFoundError(
                f"Cannot read parquet file: {self.filepath}. "
                f"File may be corrupted, on slow storage, or access denied. "
                f"Error: {e}"
            )

        return self.raw_data

    def load_raw_data_chunked(self, max_stocks_per_chunk: int = 100, verbose: bool = True) -> pd.DataFrame:
        """
        Load large parquet file in chunks by stock (gvkey) for memory efficiency.

        Args:
            max_stocks_per_chunk: Number of stocks to process per chunk
            verbose: Print progress

        Returns:
            Concatenated DataFrame of all chunks
        """
        if verbose:
            print(f"Loading large file in chunks: {self.filepath}")

        import pyarrow.parquet as pq

        try:
            # First, get unique gvkeys without loading full data
            if verbose:
                print("Getting list of unique stocks...")
            gvkeys_table = pq.read_table(self.filepath, columns=['gvkey'])
            unique_gvkeys = gvkeys_table.to_pandas()['gvkey'].unique()

            if verbose:
                print(f"Found {len(unique_gvkeys)} unique stocks")
                print(f"Processing in chunks of {max_stocks_per_chunk} stocks")

            # Process in chunks
            all_chunks = []
            total_rows = 0

            for i in range(0, len(unique_gvkeys), max_stocks_per_chunk):
                chunk_gvkeys = unique_gvkeys[i:i+max_stocks_per_chunk]

                if verbose:
                    print(f"Loading chunk {i//max_stocks_per_chunk + 1}/{(len(unique_gvkeys)-1)//max_stocks_per_chunk + 1}: "
                          f"stocks {i+1}-{min(i+max_stocks_per_chunk, len(unique_gvkeys))}")

                # Read only data for this chunk of stocks
                filters = [('gvkey', 'in', chunk_gvkeys.tolist())]
                chunk_data = pq.read_table(self.filepath, filters=filters).to_pandas()

                all_chunks.append(chunk_data)
                total_rows += len(chunk_data)

                if verbose:
                    print(f"  Loaded {len(chunk_data):,} rows ({total_rows:,} total so far)")

            # Combine all chunks
            if verbose:
                print("Combining all chunks...")
            self.raw_data = pd.concat(all_chunks, ignore_index=True)
            self.is_large_file = True

            if verbose:
                print(f"✓ Successfully loaded {len(self.raw_data):,} rows from {len(unique_gvkeys)} stocks")
                print(f"✓ Columns: {len(self.raw_data.columns)}")

        except Exception as e:
            if verbose:
                print(f"Chunked loading failed: {e}")
                print("Falling back to regular loading...")
            # Fallback to regular loading
            return self.load_raw_data(verbose=verbose)

        return self.raw_data

    def process_for_pipeline(self, verbose: bool = True) -> pd.DataFrame:
        """
        Process raw data into format needed for the missing data pipeline.

        Pipeline expects:
        - ticker: Stock identifier
        - date: Trading date
        - price: Stock price (we'll derive from returns)
        - volume: Trading volume
        - All your features

        Returns:
            DataFrame ready for CompleteMissingDataPipeline
        """
        if self.raw_data is None:
            self.load_raw_data(verbose=verbose)

        if verbose:
            print("\nProcessing data for pipeline...")

        df = self.raw_data.copy()

        # Create ticker from gvkey + iid (unique identifier)
        df['ticker'] = df['gvkey'].astype(str) + '_' + df['iid'].astype(str)

        # Use date column - check if date exists first, then char_date
        if 'date' in df.columns:
            if verbose:
                print("✓ Using 'date' column for dates")
            # Handle decimal dates by converting to int first, then string
            df['date'] = pd.to_datetime(df['date'].astype(int).astype(str), format='%Y%m%d')
        elif 'char_date' in df.columns:
            if verbose:
                print("✓ Using 'char_date' column for dates")
            df['date'] = pd.to_datetime(df['char_date'].astype(int).astype(str), format='%Y%m%d')
        else:
            raise ValueError("No date column found. Expected 'date' or 'char_date'")

        # Handle price column - use existing if available, otherwise compute from returns
        if 'price' in df.columns or 'prc' in df.columns:
            if verbose:
                print("✓ Using existing price column")
            # Use existing price column (handle both 'price' and 'prc' naming)
            if 'prc' in df.columns and 'price' not in df.columns:
                df['price'] = df['prc']
        elif 'stock_ret' in df.columns:
            if verbose:
                print("⚠️ No price column found - computing prices from returns...")

            # Create price from returns (cumulative product)
            # Start at 100 for each stock and apply monthly returns
            price_data = []
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker].sort_values('date').copy()

                # Cumulative returns to create price series starting at 100
                ticker_data['price'] = 100 * (1 + ticker_data['stock_ret']).cumprod()

                price_data.append(ticker_data)

            df = pd.concat(price_data, ignore_index=True)
        else:
            if verbose:
                print("⚠️ No price or return data found - price analysis will be limited")

        # Handle volume column - use existing if available, otherwise create proxy
        if 'volume' not in df.columns:
            if 'stock_ret' in df.columns:
                if verbose:
                    print("⚠️ No volume column found - creating proxy from returns")
                # We can use the variability in returns as a proxy for activity
                df['volume'] = np.abs(df['stock_ret']) * 1000000  # Rough proxy
            else:
                if verbose:
                    print("⚠️ No volume or return data found - using placeholder volume")
                df['volume'] = 1000000  # Simple placeholder
        else:
            if verbose:
                print("✓ Using existing volume column")

        # Rename features to match expected names (if needed)
        # Your features are already named, but let's standardize some key ones
        feature_mapping = {
            # Add mappings here if your column names differ
            # Example: 'your_column_name': 'expected_column_name'
        }

        df = df.rename(columns=feature_mapping)

        # Select essential columns + all features
        essential_cols = ['ticker', 'date', 'price', 'volume', 'stock_ret',
                          'gvkey', 'iid', 'excntry', 'year', 'month']

        # Get all feature columns (everything not in essential cols)
        feature_cols = [col for col in df.columns if col not in essential_cols]

        # Final dataframe
        self.processed_data = df[essential_cols + feature_cols]

        if verbose:
            print(f"✓ Processed data ready!")
            print(f"  Unique tickers: {self.processed_data['ticker'].nunique()}")
            print(f"  Date range: {self.processed_data['date'].min()} to {self.processed_data['date'].max()}")
            print(f"  Total features: {len(feature_cols)}")
            print(f"\nFeature columns available:")
            print(feature_cols[:20], "..." if len(feature_cols) > 20 else "")

        return self.processed_data

    def check_feature_availability(self) -> pd.DataFrame:
        """
        Check which critical features are available in your data.

        Returns:
            DataFrame showing which features exist and their coverage
        """
        if self.processed_data is None:
            self.process_for_pipeline(verbose=False)

        # Critical features for detection (using actual column names from your dataset)
        critical_features = {
            'Bankruptcy Detection': [
                'z_score',  # Altman Z-score
                'o_score',  # Ohlson O-score
                'kz_index',  # Kaplan-Zingales index
                'ni_be',  # Return on equity
                'ocf_at',  # Operating cash flow to assets
                'zero_trades_252d',  # Number of zero trades (12 months)
                'ami_126d',  # Amihud Measure
                'at_be'  # Book leverage
            ],
            'Merger Detection': [
                'dolvol_126d',  # Dollar trading volume
                'bidaskhl_21d',  # The high-low bid-ask spread
                'ni_be',  # Return on equity
                'eqnpo_me'  # Net payout yield
            ],
            'Liquidity Metrics': [
                'zero_trades_21d',  # Number of zero trades (1 month)
                'zero_trades_126d',  # Number of zero trades (6 months)
                'zero_trades_252d',  # Number of zero trades (12 months)
                'turnover_126d',  # Share turnover
                'ami_126d',  # Amihud Measure
                'dolvol_var_126d'  # Coefficient of variation for dollar trading volume
            ],
            'Quality Metrics': [
                'f_score',  # Pitroski F-score
                'ni_ar1',  # Earnings persistence
                'ni_ivol'  # Earnings volatility
            ]
        }

        results = []

        for category, features in critical_features.items():
            for feature in features:
                # Try different possible column names
                possible_names = [
                    feature,
                    feature.replace('_', ' ').title().replace(' ', ''),
                    feature.lower(),
                    feature.upper()
                ]

                found = False
                actual_name = None
                coverage = 0

                for name in possible_names:
                    if name in self.processed_data.columns:
                        found = True
                        actual_name = name
                        coverage = (self.processed_data[name].notna().sum() /
                                    len(self.processed_data) * 100)
                        break

                results.append({
                    'category': category,
                    'feature': feature,
                    'available': '✓' if found else '✗',
                    'column_name': actual_name if found else 'NOT FOUND',
                    'coverage_pct': f"{coverage:.1f}%" if found else "N/A"
                })

        return pd.DataFrame(results)

    def get_sample_statistics(self) -> Dict:
        """
        Get basic statistics about your dataset.
        """
        if self.processed_data is None:
            self.process_for_pipeline(verbose=False)

        df = self.processed_data

        stats = {
            'total_rows': len(df),
            'unique_tickers': df['ticker'].nunique(),
            'unique_stocks': df['gvkey'].nunique(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'years_covered': df['year'].nunique(),
            'countries': df['excntry'].unique().tolist() if 'excntry' in df.columns else [],
            'avg_observations_per_stock': len(df) / df['ticker'].nunique(),
            'total_features': len([c for c in df.columns if c not in
                                   ['ticker', 'date', 'price', 'volume', 'stock_ret',
                                    'gvkey', 'iid', 'excntry', 'year', 'month']])
        }

        return stats

    def detect_missing_data_patterns(self) -> pd.DataFrame:
        """
        Analyze missing data patterns before running detection.

        This helps you understand what you're dealing with.
        """
        if self.processed_data is None:
            self.process_for_pipeline(verbose=False)

        df = self.processed_data

        patterns = []

        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('date')

            # Basic info
            first_date = ticker_data['date'].min()
            last_date = ticker_data['date'].max()
            n_observations = len(ticker_data)

            # Date gaps
            ticker_data['date_diff'] = ticker_data['date'].diff().dt.days
            gaps = ticker_data[ticker_data['date_diff'] > 35]  # More than ~1 month
            n_gaps = len(gaps)
            max_gap = ticker_data['date_diff'].max() if len(ticker_data) > 1 else 0

            # Ending status
            days_since_last = (pd.Timestamp.now() - last_date).days

            if days_since_last > 365:
                status = 'likely_delisted'
            elif days_since_last > 90:
                status = 'possibly_delisted'
            elif n_gaps > 5:
                status = 'many_gaps'
            else:
                status = 'active'

            patterns.append({
                'ticker': ticker,
                'first_date': first_date,
                'last_date': last_date,
                'n_observations': n_observations,
                'n_gaps': n_gaps,
                'max_gap_days': max_gap,
                'days_since_last': days_since_last,
                'status': status
            })

        return pd.DataFrame(patterns)


# ============================================================================
# COMPANY NAMES HELPER FUNCTIONS
# ============================================================================

def create_combined_company_mapping(usa_file: str = "cik_gvkey_linktable_USA_only.csv",
                                   global_file: str = "GlobalGVKEY.csv") -> Dict[str, str]:
    """
    Create a combined company mapping from both USA and Global GVKEY files.

    Args:
        usa_file: Path to USA-only linktable file
        global_file: Path to GlobalGVKEY file

    Returns:
        Dictionary mapping gvkey to company name
    """
    company_names = {}

    try:
        # Load USA-only linktable first
        if os.path.exists(usa_file):
            names_df = pd.read_csv(usa_file)
            names_df = names_df.sort_values(['gvkey', 'datadate']).groupby('gvkey').last()
            company_names = dict(zip(names_df.index, names_df['conm']))
            usa_count = len(company_names)
            print(f"✓ Loaded {usa_count} company names from USA linktable")
        else:
            print(f"⚠️ USA linktable not found: {usa_file}")

        # Load GlobalGVKEY.csv for additional international names
        if os.path.exists(global_file):
            global_df = pd.read_csv(global_file)

            # Clean and process GlobalGVKEY data
            if 'datadate' in global_df.columns:
                global_df['datadate'] = pd.to_datetime(global_df['datadate'], errors='coerce')

            # Filter out entries with missing company names
            global_df = global_df.dropna(subset=['conm'])
            global_df = global_df[global_df['conm'].str.strip() != '']

            if len(global_df) > 0:
                # Get most recent company name for each gvkey
                if 'datadate' in global_df.columns:
                    global_df = global_df.sort_values(['gvkey', 'datadate']).groupby('gvkey').last()
                else:
                    global_df = global_df.groupby('gvkey').last()

                # Add global names for gvkeys not already in USA data
                global_names = dict(zip(global_df.index, global_df['conm']))
                new_names = {gvkey: name for gvkey, name in global_names.items()
                           if gvkey not in company_names}

                company_names.update(new_names)
                print(f"✓ Added {len(new_names)} additional company names from GlobalGVKEY")
            else:
                print("⚠️ No valid company names found in GlobalGVKEY.csv")
        else:
            print(f"ℹ️ GlobalGVKEY.csv not found at: {global_file}")

        print(f"✓ Total company names available: {len(company_names)}")

    except Exception as e:
        print(f"⚠️ Error creating company mapping: {e}")

    return company_names


# ============================================================================
# COMPLETE WORKFLOW FOR YOUR DATA
# ============================================================================

def run_complete_workflow_on_your_data(
        csv_filepath: str = '/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample.parquet',
        claude_api_key: str = None,
        verify_batch_size: int = 100,
        use_haiku: bool = True
):
    """
    Complete end-to-end workflow customized for your data format.

    Steps:
    1. Load your ret_sample.parquet
    2. Process into pipeline format
    3. Check feature availability
    4. Run detection rules
    5. Verify with Claude (when you have API key)
    6. Create labeled dataset
    7. Export results

    Args:
        csv_filepath: Path to your ret_sample.parquet
        claude_api_key: Your Claude API key (skip verification if None)
        verify_batch_size: How many events to verify
        use_haiku: Use cheaper Haiku model (recommended)
    """

    print("=" * 70)
    print("COMPLETE MISSING DATA PIPELINE - YOUR DATA FORMAT")
    print("=" * 70)

    # STEP 1: Load your data
    print("\n" + "=" * 70)
    print("STEP 1: LOAD YOUR DATA")
    print("=" * 70)

    # Use smaller memory limit for 3GB files to trigger chunked processing
    loader = DataLoader(csv_filepath, max_memory_gb=1.0)  # Trigger chunking for files >1GB
    processed_data = loader.process_for_pipeline(verbose=True)

    # STEP 2: Check features
    print("\n" + "=" * 70)
    print("STEP 2: CHECK FEATURE AVAILABILITY")
    print("=" * 70)

    feature_check = loader.check_feature_availability()
    print("\nCritical Features Status:")
    print(feature_check.to_string(index=False))

    # STEP 3: Analyze missing data patterns
    print("\n" + "=" * 70)
    print("STEP 3: ANALYZE MISSING DATA PATTERNS")
    print("=" * 70)

    patterns = loader.detect_missing_data_patterns()
    print("\nMissing Data Summary:")
    print(f"Total stocks: {len(patterns)}")
    print(f"\nBy status:")
    print(patterns['status'].value_counts())
    print(f"\nStocks with gaps: {(patterns['n_gaps'] > 0).sum()}")
    print(f"Stocks likely delisted: {(patterns['status'] == 'likely_delisted').sum()}")

    # STEP 4: Get dataset statistics
    print("\n" + "=" * 70)
    print("STEP 4: DATASET STATISTICS")
    print("=" * 70)

    stats = loader.get_sample_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # STEP 5: Run detection (if user wants to continue)
    print("\n" + "=" * 70)
    print("STEP 5: RUN DETECTION RULES")
    print("=" * 70)

    proceed = input("\nProceed with event detection? (yes/no): ")

    if proceed.lower() != 'yes':
        print("\nStopping here. Your processed data is ready in loader.processed_data")
        return loader

    # Import the pipeline
    from CompleteMissingDataPipeline import CompleteMissingDataPipeline

    pipeline = CompleteMissingDataPipeline(
        price_data=processed_data,
        claude_api_key=claude_api_key or "dummy_key",  # Placeholder if no key yet
        dataset_start_date=processed_data['date'].min(),
        company_names_file="../cik_gvkey_linktable_USA_only.csv"  # Add company names mapping
    )

    # Run detection only (skip verification if no API key)
    print("\n💾 Note: Results will be saved after detection completes")
    detected_events = pipeline.step1_detect_all_events(verbose=True)

    # Save immediately after detection
    print(f"\n💾 Saving {len(detected_events)} detected events...")
    detected_events.to_csv('detected_events_backup.csv', index=False)
    print("✓ Backup saved to: detected_events_backup.csv")

    # STEP 6: Verify with Claude (if API key provided)
    if claude_api_key:
        print("\n" + "=" * 70)
        print("STEP 6: VERIFY WITH CLAUDE")
        print("=" * 70)

        verified_events = pipeline.step2_verify_with_llm(
            batch_size=verify_batch_size,
            use_haiku=use_haiku,
            verbose=True
        )

        # Create labeled dataset
        labeled_data = pipeline.step3_create_labeled_dataset(verbose=True)

        # Export everything
        pipeline.step4_export_results(output_dir='results', verbose=True)

    else:
        print("\n" + "=" * 70)
        print("NO API KEY PROVIDED")
        print("=" * 70)
        print("\nSkipping Claude verification.")
        print("Detected events saved for later verification.")

        # Save detected events
        detected_events.to_csv('new_events_to_verify.csv', index=False)
        print(f"\n✓ Saved {len(detected_events)} detected events to: new_events_to_verify.csv")
        print("\nNext steps:")
        print("1. Get Claude API key from: console.anthropic.com")
        print("2. Run again with: run_complete_workflow_on_your_data(claude_api_key='your-key')")

    return loader, pipeline


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  MISSING DATA EXPLANATION PIPELINE                ║
    ║                      Customized for Your Data                     ║
    ╚══════════════════════════════════════════════════════════════════╝

    This will:
    1. Load your ret_sample.parquet with the specific format
    2. Check which features are available
    3. Analyze missing data patterns
    4. Run aggressive detection rules
    5. (Optional) Verify with Claude when you have API key

    """)

    # Run workflow
    # WITHOUT API key (just detection):
    loader, pipeline = run_complete_workflow_on_your_data(
        csv_filepath='/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample.parquet',
        claude_api_key=None  # Set to your key when you have it
    )

    # WITH API key (full pipeline):
    # loader, pipeline = run_complete_workflow_on_your_data(
    #     csv_filepath='/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample.parquet',
    #     claude_api_key='sk-ant-...',  # Your actual key
    #     verify_batch_size=50,
    #     use_haiku=True  # Cheaper model
    # )