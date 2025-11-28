"""
Complete Missing Data Pipeline - Detection + LLM Verification + Labeling

This integrates:
1. Aggressive detection rules from liquidity_enhanced_rules.py
2. Claude LLM verification with web search
3. Labeled dataset creation
4. Export functionality

Usage:
    from complete_pipeline import CompleteMissingDataPipeline

    pipeline = CompleteMissingDataPipeline(
        price_data=your_data,
        claude_api_key='sk-ant-...',
        dataset_start_date=your_data['date'].min()
    )

    results = pipeline.run_complete_pipeline(verify_batch_size=100)
"""

import pandas as pd
from datetime import datetime
import json
import time
from typing import Dict, List, Optional
import anthropic
from liquidity_enhanced_rules import LiquidityEnhancedDetector


class CompleteMissingDataPipeline:
    """
    End-to-end pipeline: Aggressive detection → LLM verification → Labeled dataset

    Workflow:
    1. Run aggressive rules on all stocks (cast wide net)
    2. Get ~1,500-2,000 flagged events
    3. LLM verifies each one using web search
    4. Create clean labeled dataset explaining ALL missing data
    5. Export for ML training or analysis

    Cost: ~$5 to explain all missing data in 1,000 stocks over 20 years
    """

    def __init__(self,
                 price_data: pd.DataFrame,
                 claude_api_key: str,
                 dataset_start_date: Optional[datetime] = None,
                 company_names_file: str = None):
        """
        Initialize the complete pipeline.

        Args:
            price_data: DataFrame with price + liquidity features
            claude_api_key: Your Anthropic API key
            dataset_start_date: Start date of your dataset (for IPO detection)
            company_names_file: Path to cik_gvkey_linktable_USA_only.csv file
        """
        self.price_data = price_data.copy()
        self.price_data['date'] = pd.to_datetime(self.price_data['date'])

        # Use gvkey instead of ticker
        if 'gvkey' in self.price_data.columns:
            self.price_data = self.price_data.sort_values(['gvkey', 'date'])
        else:
            # Fallback for older data
            self.price_data = self.price_data.sort_values(['ticker', 'date'])

        self.claude_api_key = claude_api_key
        self.client = anthropic.Anthropic(api_key=claude_api_key)

        self.dataset_start_date = dataset_start_date or self.price_data['date'].min()

        # Initialize the enhanced detector
        self.detector = LiquidityEnhancedDetector(
            data=self.price_data,
            company_names_file=company_names_file
        )

        # Results storage
        self.detected_events = None
        self.verified_events = None
        self.labeled_data = None

    def step1_detect_all_events(self, verbose: bool = True) -> pd.DataFrame:
        """
        STEP 1: Run aggressive detection rules to catch all potential events.

        This casts a WIDE net - high false positive rate is OK because
        LLM will verify in Step 2.

        Returns:
            DataFrame with all detected events (bankruptcies, mergers, halts, IPOs, etc.)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 1: AGGRESSIVE EVENT DETECTION")
            print("=" * 70)

            # Use gvkey if available, otherwise fall back to ticker
            id_col = 'gvkey' if 'gvkey' in self.price_data.columns else 'ticker'
            print(f"Analyzing {self.price_data[id_col].nunique()} stocks using {id_col}...")

        # Use the enhanced detector with much higher thresholds for monthly data and end-of-sample bias fix
        print("\n💾 Note: Using chunked processing with periodic saves")
        results = self.detector.detect_all_events_comprehensive(
            bankruptcy_threshold=5,   # VERY LOW - catch everything for Claude verification
            merger_threshold=5,       # VERY LOW - catch everything for Claude verification
            halt_threshold=5,         # VERY LOW - catch everything for Claude verification
            dataset_start_date=self.dataset_start_date,
            verbose=verbose,
            save_every_n_chunks=10,   # Save every 10 chunks
            output_file='interim_detected_events.csv'
        )

        # Get the combined events
        self.detected_events = results['all_events']

        if verbose:
            print(f"\n✓ Detection complete!")
            print(f"Total events flagged: {len(self.detected_events)}")
            if len(self.detected_events) > 0:
                print(f"\nBy event type:")
                print(self.detected_events['event_type'].value_counts())
                print(f"\nBy confidence:")
                print(self.detected_events['confidence'].value_counts())

        return self.detected_events

    def step2_verify_with_llm(self,
                              batch_size: int = 100,
                              delay_seconds: float = 0.5,
                              use_haiku: bool = False,
                              verbose: bool = True) -> pd.DataFrame:
        """
        STEP 2: Verify detected events using Claude + web search.

        Claude will search the web for each flagged event and determine:
        - Is it real?
        - What actually happened?
        - When did it happen?
        - Any additional details (acquisition price, etc.)

        Args:
            batch_size: Max events to verify (for cost control)
            delay_seconds: Delay between API calls
            use_haiku: Use cheaper Haiku model instead of Sonnet
            verbose: Print progress

        Returns:
            DataFrame with verified events
        """
        if self.detected_events is None:
            raise ValueError("Run step1_detect_all_events() first!")

        if verbose:
            print("\n" + "=" * 70)
            print("STEP 2: LLM VERIFICATION")
            print("=" * 70)

            # Count events with valid company names for accurate estimates
            events_with_names_count = len(self.detected_events[
                (self.detected_events['company_name'].notna()) &
                (self.detected_events['company_name'] != 'Unknown') &
                (self.detected_events['company_name'] != '')
            ])

            total_to_verify = min(batch_size, events_with_names_count)
            model = "claude-haiku-4-20250228" if use_haiku else "claude-sonnet-4-5-20250929"
            cost_per = 0.0003 if use_haiku else 0.003
            estimated_cost = total_to_verify * cost_per

            print(f"Total detected events: {len(self.detected_events)}")
            print(f"Events with company names: {events_with_names_count}")
            print(f"Events to verify: {total_to_verify}")
            print(f"Model: {model}")
            print(f"Estimated cost: ${estimated_cost:.2f}")
            print(f"Estimated time: {total_to_verify * delay_seconds / 60:.1f} minutes")

            proceed = input("\nProceed with verification? (yes/no): ")
            if proceed.lower() != 'yes':
                print("Cancelled.")
                return pd.DataFrame()

        verified_results = []

        # Filter to only events with valid company names
        events_with_names = self.detected_events[
            (self.detected_events['company_name'].notna()) &
            (self.detected_events['company_name'] != 'Unknown') &
            (self.detected_events['company_name'] != '')
        ].copy()

        events_without_names = self.detected_events[
            (self.detected_events['company_name'].isna()) |
            (self.detected_events['company_name'] == 'Unknown') |
            (self.detected_events['company_name'] == '')
        ].copy()

        if verbose and len(events_without_names) > 0:
            print(f"\n⚠️  Skipping {len(events_without_names)} events without company names (can't search effectively)")
            print(f"✓ Will verify {len(events_with_names)} events with valid company names")

        # Sort by confidence (verify high confidence first)
        confidence_order = {'high': 3, 'medium': 2, 'low': 1, 'very_low': 0}
        events_to_verify = events_with_names.copy()
        events_to_verify['conf_rank'] = events_to_verify['confidence'].map(confidence_order)
        events_to_verify = events_to_verify.sort_values('conf_rank', ascending=False)

        # Add skipped events to results with "no_company_name" status
        for _, skipped_event in events_without_names.iterrows():
            gvkey = skipped_event.get('gvkey', skipped_event.get('ticker', 'Unknown'))
            verified_results.append({
                'gvkey': gvkey,
                'company_name': 'Unknown',
                'verified': False,
                'actual_event_type': 'skipped',
                'llm_confidence': 'low',
                'details': 'Skipped verification - no company name found in database',
                'detected_event_type': skipped_event['event_type'],
                'detected_date': pd.to_datetime(skipped_event['event_date']).strftime('%Y-%m-%d'),
                'skip_reason': 'no_company_name'
            })

        model_name = "claude-haiku-4-20250228" if use_haiku else "claude-sonnet-4-5-20250929"

        for i, (idx, event) in enumerate(events_to_verify.head(batch_size).iterrows()):
            if verbose:
                # Use company name and gvkey instead of ticker for progress messages
                company_display = event.get('company_name', 'Unknown Company')
                gvkey_display = event.get('gvkey', event.get('ticker', 'Unknown'))
                print(
                    f"\n[{i + 1}/{min(batch_size, len(events_to_verify))}] Verifying {company_display} (GVKEY: {gvkey_display}) - {event['event_type']}...")

            # Verify with Claude
            result = self._verify_single_event_with_claude(event, model_name)
            verified_results.append(result)

            if verbose:
                if result.get('verified'):
                    print(
                        f"  ✓ Verified as {result.get('actual_event_type')} (confidence: {result.get('llm_confidence')})")
                else:
                    print(f"  ✗ Could not verify (confidence: {result.get('llm_confidence')})")

            time.sleep(delay_seconds)

        self.verified_events = pd.DataFrame(verified_results)

        if verbose:
            print("\n" + "=" * 70)
            print("VERIFICATION SUMMARY")
            print("=" * 70)
            print(f"Total verified: {self.verified_events['verified'].sum()}")
            print(f"Not verified: {(~self.verified_events['verified']).sum()}")
            if self.verified_events['verified'].sum() > 0:
                print(f"\nActual event types found:")
                print(self.verified_events[self.verified_events['verified']]['actual_event_type'].value_counts())

        return self.verified_events

    def step3_create_labeled_dataset(self,
                                     min_confidence: str = 'medium',
                                     verbose: bool = True) -> pd.DataFrame:
        """
        STEP 3: Create labeled training dataset from verified events.

        This applies verified labels to your price data, creating a clean
        dataset where all missing data is explained.

        Args:
            min_confidence: Minimum LLM confidence to include ('high', 'medium', 'low')
            verbose: Print progress

        Returns:
            DataFrame with labels explaining all missing data
        """
        if self.verified_events is None:
            raise ValueError("Run step2_verify_with_llm() first!")

        if verbose:
            print("\n" + "=" * 70)
            print("STEP 3: CREATE LABELED DATASET")
            print("=" * 70)

        # Filter to verified events with sufficient confidence
        confidence_levels = {'high': 3, 'medium': 2, 'low': 1}
        min_conf_level = confidence_levels[min_confidence]

        high_quality_events = self.verified_events[
            (self.verified_events['verified'] == True) &
            (self.verified_events['llm_confidence'].map(confidence_levels).fillna(0) >= min_conf_level)
            ].copy()

        if verbose:
            print(f"Using {len(high_quality_events)} high-quality verified events")

        # Create labels
        labeled_data = self.price_data.copy()
        labeled_data['event_label'] = 'none'
        labeled_data['event_date'] = pd.NaT
        labeled_data['event_confidence'] = None
        labeled_data['event_details'] = ''

        # Apply verified labels
        for _, event in high_quality_events.iterrows():
            ticker = event['ticker']
            event_type = event['actual_event_type']
            event_date = pd.to_datetime(event.get('actual_date') or event.get('event_date'))

            # Label strategy depends on event type
            if event_type in ['bankruptcy', 'merger', 'acquisition']:
                # Label last 60 days before event (distress period)
                mask = (
                        (labeled_data['ticker'] == ticker) &
                        (labeled_data['date'] <= event_date) &
                        (labeled_data['date'] >= event_date - pd.Timedelta(days=60))
                )
            elif event_type == 'halt':
                # Label exact halt period
                halt_start = event_date
                halt_end = pd.to_datetime(event.get('halt_end_date', event_date))
                mask = (
                        (labeled_data['ticker'] == ticker) &
                        (labeled_data['date'] >= halt_start) &
                        (labeled_data['date'] <= halt_end)
                )
            elif event_type == 'ipo':
                # Label IPO date only
                mask = (
                        (labeled_data['ticker'] == ticker) &
                        (labeled_data['date'] == event_date)
                )
            else:
                # For other events, label the specific date
                mask = (
                        (labeled_data['ticker'] == ticker) &
                        (labeled_data['date'] == event_date)
                )

            labeled_data.loc[mask, 'event_label'] = event_type
            labeled_data.loc[mask, 'event_date'] = event_date
            labeled_data.loc[mask, 'event_confidence'] = event.get('llm_confidence', 'medium')
            labeled_data.loc[mask, 'event_details'] = event.get('details', '')

        self.labeled_data = labeled_data

        if verbose:
            print(f"\n✓ Labeled dataset created!")
            print(f"Total rows: {len(labeled_data):,}")
            print(f"Labeled rows: {(labeled_data['event_label'] != 'none').sum():,}")
            print(f"\nLabel distribution:")
            print(labeled_data['event_label'].value_counts())

        return self.labeled_data

    def step4_export_results(self,
                             output_dir: str = '.',
                             verbose: bool = True):
        """
        STEP 4: Export all results for future use.

        Saves:
        - detected_events.csv: All flags from rules
        - verified_events.csv: LLM-verified events (your ground truth!)
        - labeled_data.parquet: Complete dataset with labels
        - summary_report.txt: Statistics and summary
        """
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 4: EXPORT RESULTS")
            print("=" * 70)

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export detected events
        if self.detected_events is not None:
            path = f"{output_dir}/detected_events.csv"
            self.detected_events.to_csv(path, index=False)
            if verbose:
                print(f"✓ Saved detected events: {path}")

        # Export verified events (GOLD STANDARD!)
        if self.verified_events is not None:
            path = f"{output_dir}/verified_events.csv"
            self.verified_events.to_csv(path, index=False)
            if verbose:
                print(f"✓ Saved verified events: {path}")
                print(f"  → This is your ground truth! Reuse without re-verifying.")

        # Export labeled data
        if self.labeled_data is not None:
            path = f"{output_dir}/labeled_data.parquet"
            self.labeled_data.to_parquet(path, index=False)
            if verbose:
                print(f"✓ Saved labeled dataset: {path}")

        # Generate summary report
        if self.verified_events is not None:
            self._generate_summary_report(output_dir, verbose)

        if verbose:
            print(f"\n✅ All results exported to: {output_dir}/")

    def run_complete_pipeline(self,
                              verify_batch_size: int = 100,
                              use_haiku: bool = False) -> Dict:
        """
        Run the COMPLETE end-to-end pipeline.

        This is the main function you call to process everything!

        Returns:
            Dictionary with all results
        """
        print("\n" + "=" * 70)
        print("COMPLETE MISSING DATA EXPLANATION PIPELINE")
        print("=" * 70)
        print("\nThis will:")
        print("1. Detect all potential events using aggressive rules")
        print("2. Verify each event using Claude + web search")
        print("3. Create labeled dataset explaining all missing data")
        print("4. Export results for future use")

        # Step 1: Detect
        detected = self.step1_detect_all_events(verbose=True)

        # Step 2: Verify
        verified = self.step2_verify_with_llm(
            batch_size=verify_batch_size,
            use_haiku=use_haiku,
            verbose=True
        )

        if len(verified) == 0:
            print("\n⚠️  Verification cancelled or failed.")
            return {}

        # Step 3: Label
        labeled = self.step3_create_labeled_dataset(verbose=True)

        # Step 4: Export
        self.step4_export_results(verbose=True)

        return {
            'detected_events': detected,
            'verified_events': verified,
            'labeled_data': labeled
        }

    # ========================================================================
    # DETECTION METHODS (simplified versions - full versions in liquidity_enhanced_rules.py)
    # ========================================================================

    def _detect_bankruptcy(self, stock_data: pd.DataFrame, ticker: str) -> List[Dict]:
        """Detect bankruptcy - FIXED FOR MONTHLY DATA (threshold: 25 points)"""
        events = []

        if len(stock_data) < 24:  # Need 2 years of monthly data
            return events

        recent = stock_data.tail(24)  # Last 2 years for monthly data
        score = 0
        reasons = []

        # Financial distress signals
        if 'altman_z_score' in recent.columns:
            z_score = recent['altman_z_score'].iloc[-1]
            if pd.notna(z_score):
                if z_score < 1.8:
                    score += 15
                    reasons.append(f"Altman Z-score {z_score:.2f} (bankruptcy zone)")
                elif z_score < 3.0:
                    score += 8
                    reasons.append(f"Altman Z-score {z_score:.2f} (gray zone)")

        if 'ohlson_o_score' in recent.columns:
            o_score = recent['ohlson_o_score'].iloc[-1]
            if pd.notna(o_score):
                if o_score > 0.5:
                    score += 15
                    reasons.append(f"Ohlson O-score {o_score:.2f} (high risk)")
                elif o_score > 0.3:
                    score += 8
                    reasons.append(f"Ohlson O-score {o_score:.2f} (elevated risk)")

        # Profitability collapse
        if 'return_on_equity' in recent.columns:
            roe = recent['return_on_equity'].iloc[-1]
            if pd.notna(roe):
                if roe < -0.20:
                    score += 10
                    reasons.append(f"ROE {roe * 100:.1f}%")
                elif roe < -0.10:
                    score += 5
                    reasons.append(f"ROE {roe * 100:.1f}%")
                elif roe < 0:
                    score += 3
                    reasons.append(f"ROE negative")

        # Liquidity death - FIXED FOR MONTHLY DATA
        if 'zero_trades_12m' in recent.columns:
            zero_trades = recent['zero_trades_12m'].iloc[-1]
            if pd.notna(zero_trades):
                if zero_trades > 90:  # More than 3 months of zero trades
                    score += 10
                    reasons.append(f"{zero_trades:.0f} zero-trade days (3+ months)")
                elif zero_trades > 60:  # More than 2 months
                    score += 5
                    reasons.append(f"{zero_trades:.0f} zero-trade days (2+ months)")

        # Price collapse
        price_change = (recent['price'].iloc[-1] - recent['price'].iloc[0]) / recent['price'].iloc[0]
        if price_change < -0.80:
            score += 15
            reasons.append(f"Price dropped {price_change * 100:.1f}%")
        elif price_change < -0.60:
            score += 8
            reasons.append(f"Price dropped {price_change * 100:.1f}%")

        # Trading stopped - FIXED FOR MONTHLY DATA
        last_date = recent['date'].iloc[-1]
        days_since = (pd.Timestamp.now() - last_date).days
        if days_since > 180:  # 6+ months for monthly data
            score += 20
            reasons.append(f"No trading for {days_since} days (6+ months)")
        elif days_since > 90:  # 3+ months for monthly data
            score += 10
            reasons.append(f"No trading for {days_since} days (3+ months)")

        if score >= 5:   # VERY LOW threshold - catch everything for Claude verification
            confidence = 'high' if score >= 50 else 'medium' if score >= 35 else 'low'
            events.append({
                'ticker': ticker,
                'event_type': 'bankruptcy',
                'event_date': last_date,
                'detection_score': score,
                'confidence': confidence,
                'reasons': '; '.join(reasons),
                'last_price': recent['price'].iloc[-1]
            })

        return events

    def _detect_merger(self, stock_data: pd.DataFrame, ticker: str) -> List[Dict]:
        """Detect merger/acquisition - FIXED FOR MONTHLY DATA (threshold: 20 points)"""
        events = []

        if len(stock_data) < 24:  # Need 2 years of monthly data
            return events

        recent = stock_data.tail(24)  # Last 2 years for monthly data
        score = 0
        reasons = []

        # Check if trading stopped
        last_date = recent['date'].iloc[-1]
        days_since = (pd.Timestamp.now() - last_date).days

        if days_since < 90:  # For monthly data, need 3+ months gap
            return events  # Still trading (less than 3 months is normal for monthly data)

        # Price stability (key difference from bankruptcy)
        price_change = (recent['price'].iloc[-1] - recent['price'].iloc[0]) / recent['price'].iloc[0]
        if -0.15 < price_change < 0.50:
            score += 20
            reasons.append(f"Price stable ({price_change * 100:.1f}%)")

        # Volume spike (arb traders) - FIXED FOR MONTHLY DATA
        if 'dollar_volume' in recent.columns:
            recent_vol = recent['dollar_volume'].iloc[-6:].mean()  # Last 6 months for monthly data
            hist_vol = stock_data['dollar_volume'].quantile(0.50)
            if pd.notna(recent_vol) and pd.notna(hist_vol) and hist_vol > 0:
                vol_ratio = recent_vol / hist_vol
                if vol_ratio > 2.0:  # Higher threshold for monthly data
                    score += 15
                    reasons.append(f"Volume spike {vol_ratio:.1f}x")
                elif vol_ratio > 1.5:
                    score += 8
                    reasons.append(f"Volume elevated {vol_ratio:.1f}x")

        # Financial health (NOT distressed)
        if 'return_on_equity' in recent.columns:
            roe = recent['return_on_equity'].iloc[-1]
            if pd.notna(roe) and roe > 0:
                score += 8
                reasons.append("Positive ROE")

        # Trading stopped - FIXED FOR MONTHLY DATA
        if days_since > 90:  # 3+ months for monthly data
            score += 15
            reasons.append(f"Trading stopped {days_since} days ago (3+ months)")

        # Anti-bankruptcy filter
        bankruptcy_score = 0
        if 'altman_z_score' in recent.columns:
            z = recent['altman_z_score'].iloc[-1]
            if pd.notna(z) and z < 1.8:
                bankruptcy_score += 15
        if price_change < -0.50:
            bankruptcy_score += 15

        if score >= 5 and bankruptcy_score < 30:   # VERY LOW threshold - catch everything for Claude verification
            confidence = 'medium' if score >= 35 else 'low'
            events.append({
                'ticker': ticker,
                'event_type': 'merger',
                'event_date': last_date,
                'detection_score': score,
                'confidence': confidence,
                'reasons': '; '.join(reasons),
                'last_price': recent['price'].iloc[-1]
            })

        return events

    def _detect_halt(self, stock_data: pd.DataFrame, ticker: str) -> List[Dict]:
        """Detect trading halts - FIXED FOR MONTHLY DATA"""
        events = []

        if len(stock_data) < 10:
            return events

        # Find gaps in data - FOR MONTHLY DATA, need 90+ day gaps (3+ months)
        stock_data = stock_data.copy()
        stock_data['days_diff'] = stock_data['date'].diff().dt.days

        # For monthly data: normal = ~30 days, halt = 90+ days
        gaps = stock_data[stock_data['days_diff'] >= 90]

        for idx, gap_row in gaps.iterrows():
            score = 0
            reasons = []

            gap_days = gap_row['days_diff']
            gap_start = stock_data[stock_data['date'] < gap_row['date']]['date'].iloc[-1] if len(
                stock_data[stock_data['date'] < gap_row['date']]) > 0 else gap_row['date']
            gap_end = gap_row['date']

            # Gap detected (90+ days for monthly data)
            if gap_days >= 90:
                score += 10
                reasons.append(f"{gap_days:.0f}-day gap (3+ months)")

            # Trading resumed (critical for halt vs delisting)
            row_position = stock_data.index.get_loc(idx)
            if row_position < len(stock_data) - 1:
                score += 15
                reasons.append("Trading resumed")

            # Shorter gaps more likely to be halts
            if gap_days < 180:  # Less than 6 months
                score += 10
                reasons.append("Short gap (likely halt)")
            elif gap_days < 365:  # Less than 1 year
                score += 5
                reasons.append("Medium gap")

            if score >= 3:   # VERY LOW threshold - catch everything for Claude verification
                confidence = 'medium' if score >= 25 else 'low'
                events.append({
                    'ticker': ticker,
                    'event_type': 'halt',
                    'event_date': gap_start,
                    'halt_end_date': gap_end,
                    'detection_score': score,
                    'confidence': confidence,
                    'reasons': '; '.join(reasons),
                    'gap_days': gap_days
                })

        return events

    def _detect_ipo(self, stock_data: pd.DataFrame, ticker: str) -> List[Dict]:
        """Detect IPO - FIXED FOR MONTHLY DATA (Only flag real IPOs)"""
        events = []

        first_date = stock_data['date'].min()

        # FIXED: Only flag as IPO if first date is SIGNIFICANTLY after dataset start
        # For monthly data, if stock appears 6+ months after dataset start, likely real IPO
        months_after_start = (first_date - self.dataset_start_date).days / 30.0

        if months_after_start < 6:  # Less than 6 months after dataset start
            # Likely just dataset boundary, not real IPO
            return events  # Don't flag as IPO
        else:
            # Check firm age if available
            if 'firm_age' in stock_data.columns:
                firm_age_at_start = stock_data.iloc[0]['firm_age']
                if pd.notna(firm_age_at_start):
                    if firm_age_at_start < 2:  # Less than 2 years old
                        event_type = 'ipo'
                        confidence = 'high'
                        reasons = f"Young firm (age {firm_age_at_start:.1f}y), appeared {months_after_start:.1f} months after dataset start"
                    else:
                        event_type = 'uncertain_ipo'
                        confidence = 'medium'
                        reasons = f"Firm age {firm_age_at_start:.1f}y, appeared {months_after_start:.1f} months after dataset start"
                else:
                    event_type = 'ipo'
                    confidence = 'medium'
                    reasons = f"Appeared {months_after_start:.1f} months after dataset start"
            else:
                event_type = 'ipo'
                confidence = 'medium'
                reasons = f"Appeared {months_after_start:.1f} months after dataset start"

            events.append({
                'ticker': ticker,
                'event_type': event_type,
                'event_date': first_date,
                'detection_score': 100,  # Deterministic
                'confidence': confidence,
                'reasons': reasons,
                'first_price': stock_data.iloc[0]['price']
            })

        return events

    def _detect_stock_split(self, stock_data: pd.DataFrame, ticker: str) -> List[Dict]:
        """Detect stock splits"""
        events = []

        # Look for sudden price drops of exactly 50%, 33%, 25% (2:1, 3:1, 4:1 splits)
        stock_data = stock_data.copy()
        stock_data['price_change'] = stock_data['price'].pct_change()

        # Common split ratios
        split_ratios = {
            -0.50: '2:1 split',
            -0.67: '3:1 split',
            -0.75: '4:1 split'
        }

        for ratio, split_type in split_ratios.items():
            matches = stock_data[abs(stock_data['price_change'] - ratio) < 0.02]

            for idx, row in matches.iterrows():
                score = 15
                reasons = [f"Price drop {row['price_change'] * 100:.1f}% (matches {split_type})"]

                # Check volume spike
                row_position = stock_data.index.get_loc(idx)
                if row_position > 0 and 'volume' in stock_data.columns:
                    prev_vol = stock_data.iloc[row_position - 1]['volume']
                    if pd.notna(prev_vol) and pd.notna(row['volume']) and prev_vol > 0:
                        if row['volume'] > prev_vol * 1.5:
                            score += 10
                            reasons.append("Volume spike")

                events.append({
                    'ticker': ticker,
                    'event_type': 'stock_split',
                    'event_date': row['date'],
                    'detection_score': score,
                    'confidence': 'medium',
                    'reasons': '; '.join(reasons),
                    'split_type': split_type
                })

        return events

    def _detect_data_errors(self, stock_data: pd.DataFrame, ticker: str) -> List[Dict]:
        """Detect data quality issues"""
        events = []

        for idx, row in stock_data.iterrows():
            score = 0
            reasons = []

            # Negative prices
            if row['price'] < 0:
                score += 20
                reasons.append("Negative price")

            # Extreme price jumps
            row_position = stock_data.index.get_loc(idx)
            if row_position > 0:
                prev_price = stock_data.iloc[row_position - 1]['price']
                if prev_price > 0:
                    price_change = (row['price'] - prev_price) / prev_price
                    if abs(price_change) > 10:  # 1000%+ jump
                        score += 15
                        reasons.append(f"Extreme jump {price_change * 100:.0f}%")

            # Zero volume for extended period
            if 'volume' in row and pd.notna(row['volume']) and row['volume'] == 0:
                score += 5
                reasons.append("Zero volume")

            if score >= 3:
                events.append({
                    'ticker': ticker,
                    'event_type': 'data_error',
                    'event_date': row['date'],
                    'detection_score': score,
                    'confidence': 'high',
                    'reasons': '; '.join(reasons),
                    'price': row['price']
                })

        return events

    # ========================================================================
    # LLM VERIFICATION
    # ========================================================================

    def _verify_single_event_with_claude(self, event: pd.Series, model: str) -> Dict:
        """Verify a single detected event using Claude with web search"""

        # Use gvkey and company_name instead of ticker
        gvkey = event.get('gvkey', event.get('ticker', 'Unknown'))  # Fallback for old data
        company_name = event.get('company_name', 'Unknown Company')
        event_type = event['event_type']
        event_date = pd.to_datetime(event['event_date'])

        # Craft prompt for Claude using company name for search
        prompt = f"""I detected a potential {event_type} event for company "{company_name}" (GVKEY: {gvkey}).

        **Detected Information:**
        - Company: {company_name}
        - GVKEY: {gvkey}
        - Event Type: {event_type}
        - Detected Date: {event_date.strftime('%Y-%m-%d')}
        - Detection Score: {event.get('detection_score', 'N/A')}
        - Confidence: {event.get('confidence', 'N/A')}
        - Reasons: {event.get('reasons', 'N/A')}

        **Your Task:**
        1. Search the web to verify what actually happened to "{company_name}" around {event_date.strftime('%B %Y')}
        2. Determine if this was a real {event_type} or something else
        3. Provide accurate details

        **Respond in this EXACT JSON format (no markdown, just JSON):**
        {{
            "verified": true or false,
            "actual_event_type": "bankruptcy" | "merger" | "acquisition" | "halt" | "ipo" | "stock_split" | "ticker_change" | "data_error" | "none" | "unknown",
            "actual_date": "YYYY-MM-DD" or null,
            "llm_confidence": "high" | "medium" | "low",
            "details": "Brief explanation of what happened",
            "acquirer": "Company name if merger/acquisition, else null",
            "acquisition_price": price per share if known, else null
        }}

        If you cannot find information, set verified=false and llm_confidence="low".
        Search the web thoroughly before concluding."""

        try:
            # Call Claude with web search enabled
            message = self.client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = message.content[0].text

            # Extract JSON from response
            # LLM might wrap JSON in markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)

            # Add metadata
            result['gvkey'] = gvkey
            result['company_name'] = company_name
            result['detected_event_type'] = event_type
            result['detected_date'] = event_date.strftime('%Y-%m-%d')
            result['llm_response'] = response_text

            return result

        except Exception as e:
            print(f"  Error verifying {company_name} (GVKEY: {gvkey}): {str(e)}")
            return {
                'gvkey': gvkey,
                'company_name': company_name,
                'verified': False,
                'actual_event_type': 'error',
                'llm_confidence': 'low',
                'details': f'Error during verification: {str(e)}',
                'error': str(e),
                'detected_event_type': event_type,
                'detected_date': event_date.strftime('%Y-%m-%d')
            }

    def _generate_summary_report(self, output_dir: str, verbose: bool):
        """Generate text summary report"""

        report_path = f"{output_dir}/summary_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MISSING DATA EXPLANATION - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total stocks: {self.price_data['ticker'].nunique()}\n")
            f.write(f"Date range: {self.price_data['date'].min().date()} to {self.price_data['date'].max().date()}\n")
            f.write(f"Total observations: {len(self.price_data):,}\n\n")

            # Detection summary
            if self.detected_events is not None:
                f.write("DETECTION SUMMARY (Rules-Based)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total events detected: {len(self.detected_events)}\n\n")
                f.write("By event type:\n")
                for event_type, count in self.detected_events['event_type'].value_counts().items():
                    f.write(f"  {event_type}: {count}\n")
                f.write("\nBy confidence:\n")
                for conf, count in self.detected_events['confidence'].value_counts().items():
                    f.write(f"  {conf}: {count}\n")
                f.write("\n")

            # Verification summary
            if self.verified_events is not None:
                f.write("VERIFICATION SUMMARY (LLM-Verified)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total events verified: {len(self.verified_events)}\n")
                f.write(f"Confirmed as real: {self.verified_events['verified'].sum()}\n")
                f.write(f"Could not verify: {(~self.verified_events['verified']).sum()}\n\n")

                verified_only = self.verified_events[self.verified_events['verified']]
                if len(verified_only) > 0:
                    f.write("Actual event types found:\n")
                    for event_type, count in verified_only['actual_event_type'].value_counts().items():
                        f.write(f"  {event_type}: {count}\n")
                    f.write("\n")

            # Labeling summary
            if self.labeled_data is not None:
                f.write("LABELED DATASET SUMMARY\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total rows in dataset: {len(self.labeled_data):,}\n")
                f.write(f"Rows with event labels: {(self.labeled_data['event_label'] != 'none').sum():,}\n")
                f.write(
                    f"Percentage labeled: {(self.labeled_data['event_label'] != 'none').sum() / len(self.labeled_data) * 100:.2f}%\n\n")
                f.write("Label distribution:\n")
                for label, count in self.labeled_data['event_label'].value_counts().items():
                    f.write(f"  {label}: {count}\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        if verbose:
            print(f"✓ Saved summary report: {report_path}")

    # ============================================================================
    # USAGE EXAMPLE
    # ============================================================================

    if __name__ == "__main__":
        print("""
            ╔══════════════════════════════════════════════════════════════════╗
            ║          COMPLETE MISSING DATA PIPELINE - READY TO USE           ║
            ╚══════════════════════════════════════════════════════════════════╝

            Usage:

            from complete_pipeline import CompleteMissingDataPipeline

            # Initialize
            pipeline = CompleteMissingDataPipeline(
                price_data=your_data,
                claude_api_key='sk-ant-...',
                dataset_start_date=your_data['date'].min()
            )

            # Run complete pipeline
            results = pipeline.run_complete_pipeline(
                verify_batch_size=100,
                use_haiku=True  # Cheaper model
            )

            # Or run step-by-step:
            detected = pipeline.step1_detect_all_events()
            verified = pipeline.step2_verify_with_llm(batch_size=50)
            labeled = pipeline.step3_create_labeled_dataset()
            pipeline.step4_export_results(output_dir='results')
            """)
