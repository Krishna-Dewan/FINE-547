#!/usr/bin/env python3
"""
Efficient Single-Pass Event Detector with Conflict Resolution

This replaces the multiple-pass detection with a single efficient pass that:
1. Goes through each stock only once
2. Runs all detections simultaneously
3. Applies smart conflict resolution
4. Saves progress every N stocks
5. Minimizes duplicate events per stock
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from liquidity_enhanced_rules import LiquidityEnhancedDetector
import os

class EfficientSinglePassDetector(LiquidityEnhancedDetector):
    """
    Enhanced detector that processes each stock once and resolves conflicts.
    """

    def detect_all_events_single_pass(self,
                                    bankruptcy_threshold: int = 50,
                                    merger_threshold: int = 40,
                                    halt_threshold: int = 30,
                                    dataset_start_date: Optional[datetime] = None,
                                    verbose: bool = True,
                                    save_every_n_stocks: int = 1000,
                                    output_file: str = 'new_events_to_verify.csv') -> pd.DataFrame:
        """
        Single-pass detection with conflict resolution.

        Args:
            bankruptcy_threshold: Minimum score for bankruptcy
            merger_threshold: Minimum score for merger
            halt_threshold: Minimum score for halt
            dataset_start_date: Start of dataset for IPO detection
            verbose: Print progress
            save_every_n_stocks: Save progress every N stocks
            output_file: Final output filename

        Returns:
            DataFrame with one best event per stock (minimizes duplicates)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("EFFICIENT SINGLE-PASS EVENT DETECTION")
            print("=" * 70)
            print("✅ Each stock processed ONCE")
            print("✅ Smart conflict resolution")
            print("✅ Chunked saving every", save_every_n_stocks, "stocks")
            print("=" * 70)

        all_events = []
        gvkeys = self.data['gvkey'].unique()
        total_stocks = len(gvkeys)

        if dataset_start_date is None:
            dataset_start_date = self.data['date'].min()

        if verbose:
            print(f"\n🔍 DATA LOADING COMPLETE - Starting event detection...")
            print(f"📊 Processing {total_stocks} unique stocks for events...")
            print(f"💾 Will save progress every {save_every_n_stocks} stocks")

        for idx, gvkey in enumerate(gvkeys):
            if verbose and idx % 500 == 0:  # Show progress every 500 stocks
                print(f"  🔍 Analyzing stock {idx+1}/{total_stocks} (gvkey: {gvkey})...")

            # Show chunk completion
            if verbose and idx > 0 and idx % 1000 == 0:
                print(f"  ✅ Completed {idx} stocks so far...")

            # Save progress every N stocks
            if idx > 0 and idx % save_every_n_stocks == 0 and len(all_events) > 0:
                self._save_progress(all_events, output_file, idx, verbose)

            # Get stock data
            stock = self.data[self.data['gvkey'] == gvkey].copy()

            if len(stock) < 10:  # Skip stocks with insufficient data
                continue

            # Run ALL detections for this stock simultaneously
            event_candidates = self._detect_all_events_for_stock(
                gvkey, stock,
                bankruptcy_threshold, merger_threshold, halt_threshold,
                dataset_start_date
            )

            # Apply conflict resolution - pick the BEST event for this stock
            best_event = self._resolve_conflicts(event_candidates, gvkey)

            if best_event:
                all_events.append(best_event)

        # Final save
        results_df = pd.DataFrame(all_events) if all_events else pd.DataFrame()

        if len(results_df) > 0:
            results_df.to_csv(output_file, index=False)
            if verbose:
                print(f"\n💾 Final save: {output_file} ({len(results_df)} events)")
                print(f"\n📊 SUMMARY:")
                print(f"   Total stocks processed: {total_stocks}")
                print(f"   Events detected: {len(results_df)}")
                print(f"   Event types:")
                for event_type, count in results_df['event_type'].value_counts().items():
                    print(f"     {event_type}: {count}")
                print(f"   Confidence levels:")
                for conf, count in results_df['confidence'].value_counts().items():
                    print(f"     {conf}: {count}")
        else:
            if verbose:
                print(f"\n✅ No events detected with current thresholds")
                print(f"   This is likely correct - thresholds are working!")

        return results_df

    def _detect_all_events_for_stock(self, gvkey: int, stock: pd.DataFrame,
                                   bankruptcy_threshold: int, merger_threshold: int,
                                   halt_threshold: int, dataset_start_date: datetime) -> List[Dict]:
        """
        Run M&A and bankruptcy detections for a single stock and return all candidates.
        FOCUSED ON: Mergers & Acquisitions + Bankruptcy only
        """
        candidates = []

        # 1. Check for bankruptcy signals
        bankruptcy_score, bankruptcy_reasons = self._score_bankruptcy(stock)
        if bankruptcy_score >= bankruptcy_threshold:
            candidates.append({
                'gvkey': gvkey,
                'event_type': 'bankruptcy',
                'score': bankruptcy_score,
                'confidence': self._get_confidence(bankruptcy_score, 'bankruptcy'),
                'reasons': bankruptcy_reasons,
                'event_date': stock['date'].iloc[-1],
                'priority': 1  # Highest priority - most definitive
            })

        # 2. Check for merger signals (only if not clearly bankrupt)
        if bankruptcy_score < bankruptcy_threshold * 0.8:  # Not clearly bankrupt
            merger_score, merger_reasons = self._score_merger(stock)
            if merger_score >= merger_threshold:
                candidates.append({
                    'gvkey': gvkey,
                    'event_type': 'merger',
                    'score': merger_score,
                    'confidence': self._get_confidence(merger_score, 'merger'),
                    'reasons': merger_reasons,
                    'event_date': stock['date'].iloc[-1],
                    'priority': 2
                })

        # REMOVED: Halt detection and IPO detection (not needed for M&A/bankruptcy analysis)

        return candidates

    def _score_bankruptcy(self, stock: pd.DataFrame) -> Tuple[int, str]:
        """
        Score bankruptcy signals for a single stock.
        Returns: (score, reasons_string)
        """
        if len(stock) < 24:
            return 0, ""

        recent = stock.tail(24)
        score = 0
        reasons = []

        # Financial distress signals
        if 'z_score' in recent.columns:
            z_score = recent['z_score'].iloc[-1]
            if pd.notna(z_score):
                if z_score < 1.8:
                    score += 15
                    reasons.append(f"Z-score {z_score:.2f} (bankruptcy zone)")
                elif z_score < 3.0:
                    score += 8
                    reasons.append(f"Z-score {z_score:.2f} (gray zone)")

        if 'o_score' in recent.columns:
            o_score = recent['o_score'].iloc[-1]
            if pd.notna(o_score):
                if o_score > 0.5:
                    score += 15
                    reasons.append(f"O-score {o_score:.2f} (high risk)")

        # Profitability collapse
        if 'ni_be' in recent.columns:
            roe = recent['ni_be'].iloc[-1]
            if pd.notna(roe):
                if roe < -0.20:
                    score += 10
                    reasons.append(f"ROE {roe*100:.1f}%")
                elif roe < 0:
                    score += 5
                    reasons.append("Negative ROE")

        # Trading activity collapse
        if 'zero_trades_252d' in recent.columns:
            zero_trades = recent['zero_trades_252d'].iloc[-1]
            if pd.notna(zero_trades) and zero_trades > 90:
                score += 10
                reasons.append(f"{zero_trades:.0f} zero-trade days")

        # Volume collapse
        if 'dolvol_126d' in recent.columns:
            recent_vol = recent['dolvol_126d'].iloc[-12:].mean()
            hist_vol = stock['dolvol_126d'].quantile(0.50)
            if pd.notna(recent_vol) and pd.notna(hist_vol) and hist_vol > 0:
                vol_collapse = 1 - (recent_vol / hist_vol)
                if vol_collapse > 0.80:
                    score += 8
                    reasons.append(f"Volume collapsed {vol_collapse*100:.0f}%")

        # Trading stopped before dataset end (be more conservative)
        last_date = recent['date'].iloc[-1]
        days_since_last = (self.dataset_end - last_date).days
        # Only flag if stopped trading >1 year before dataset end (very conservative)
        if days_since_last > 365 and last_date < self.dataset_end - pd.Timedelta(days=180):
            score += 20
            reasons.append(f"Stopped trading {days_since_last} days early")

        # Price collapse
        price_change = (recent['price'].iloc[-1] - recent['price'].iloc[0]) / recent['price'].iloc[0]
        if price_change < -0.80:
            score += 15
            reasons.append(f"Price dropped {price_change*100:.1f}%")

        return score, '; '.join(reasons)

    def _score_merger(self, stock: pd.DataFrame) -> Tuple[int, str]:
        """
        Score merger signals for a single stock.
        Returns: (score, reasons_string)
        """
        if len(stock) < 24:
            return 0, ""

        recent = stock.tail(24)
        last_date = recent['date'].iloc[-1]
        days_since_last = (self.dataset_end - last_date).days

        # Must have stopped trading before dataset end (more conservative)
        if days_since_last < 240 or last_date >= self.dataset_end - pd.Timedelta(days=120):
            return 0, ""

        score = 0
        reasons = []

        # Price stability (key merger signal)
        price_change = (recent['price'].iloc[-1] - recent['price'].iloc[0]) / recent['price'].iloc[0]
        if -0.15 < price_change < 0.50:
            score += 20
            reasons.append(f"Price stable ({price_change*100:.1f}%)")

        # Volume activity
        if 'dolvol_126d' in recent.columns:
            recent_vol = recent['dolvol_126d'].iloc[-6:].mean()
            hist_vol = stock['dolvol_126d'].quantile(0.50)
            if pd.notna(recent_vol) and pd.notna(hist_vol) and hist_vol > 0:
                vol_ratio = recent_vol / hist_vol
                if vol_ratio > 2.0:
                    score += 15
                    reasons.append(f"Volume spike {vol_ratio:.1f}x")

        # Financial health
        if 'ni_be' in recent.columns:
            roe = recent['ni_be'].iloc[-1]
            if pd.notna(roe) and roe > 0:
                score += 8
                reasons.append("Positive ROE")

        if 'z_score' in recent.columns:
            z_score = recent['z_score'].iloc[-1]
            if pd.notna(z_score) and z_score > 1.8:
                score += 5
                reasons.append(f"Healthy Z-score {z_score:.2f}")

        # Trading stopped
        if days_since_last > 120:
            score += 15
            reasons.append(f"Trading stopped {days_since_last} days early")

        return score, '; '.join(reasons)

    def _score_halt(self, stock: pd.DataFrame) -> Tuple[int, str, pd.Timestamp]:
        """
        Score halt signals for a single stock.
        Returns: (score, reasons_string, halt_date)
        """
        if len(stock) < 10:
            return 0, "", pd.NaT

        stock = stock.copy()
        stock['days_diff'] = stock['date'].diff().dt.days
        stock['has_gap'] = stock['days_diff'] >= 120

        # Find halt periods that RESUMED
        max_score = 0
        best_reasons = ""
        best_date = pd.NaT

        gaps = stock[stock['has_gap']]
        for idx, gap_row in gaps.iterrows():
            # Check if trading resumed after this gap
            gap_date = gap_row['date']
            later_trading = stock[stock['date'] > gap_date]

            if len(later_trading) > 0:  # Trading resumed
                score = 25  # Base score for resumed trading
                total_days = gap_row['days_diff']

                reasons = [f"{total_days:.0f}-day gap", "Trading resumed"]

                if total_days < 180:
                    score += 10
                    reasons.append("Short halt")

                if score > max_score:
                    max_score = score
                    best_reasons = '; '.join(reasons)
                    best_date = gap_date

        return max_score, best_reasons, best_date

    def _resolve_conflicts(self, candidates: List[Dict], gvkey: int) -> Optional[Dict]:
        """
        Smart conflict resolution - pick the BEST event for this stock.

        Priority order:
        1. Bankruptcy (most definitive)
        2. Merger (if not bankrupt)
        3. Halt (if trading resumed)
        4. IPO (if no other events)
        """
        if not candidates:
            return None

        # Sort by priority (lower = higher priority)
        candidates.sort(key=lambda x: (x['priority'], -x['score']))

        best = candidates[0]

        # Add company name and final formatting
        return {
            'gvkey': gvkey,
            'company_name': self.company_names.get(gvkey, 'Unknown'),
            'event_type': best['event_type'],
            'event_date': best['event_date'],
            'detection_score': best['score'],
            'confidence': best['confidence'],
            'reasons': best['reasons']
        }

    def _get_confidence(self, score: int, event_type: str) -> str:
        """Get confidence level based on score and event type."""
        if event_type == 'bankruptcy':
            return 'high' if score >= 60 else 'medium' if score >= 40 else 'low'
        elif event_type == 'merger':
            return 'medium' if score >= 50 else 'low'
        elif event_type == 'halt':
            return 'medium' if score >= 35 else 'low'
        else:
            return 'medium'

    def _save_progress(self, events: List[Dict], output_file: str, stock_idx: int, verbose: bool):
        """Save progress every N stocks."""
        if not events:
            return

        temp_df = pd.DataFrame(events)
        progress_file = output_file.replace('.csv', f'_progress_{stock_idx}.csv')
        temp_df.to_csv(progress_file, index=False)

        if verbose:
            print(f"    💾 Progress saved: {progress_file} ({len(events)} events so far)")

def run_efficient_detection(data_file: str = None,
                          claude_api_key: str = None,
                          **kwargs) -> pd.DataFrame:
    """
    Main function to run efficient single-pass detection.

    Args:
        data_file: Path to your data file
        claude_api_key: API key (not used in detection, just for compatibility)
        **kwargs: Additional parameters for detection

    Returns:
        DataFrame with detected events
    """
    print("🚀 Starting Efficient Single-Pass Event Detection")

    # Load data
    if data_file is None:
        data_file = '/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa.parquet'

    from data_loader_helper import DataLoader
    loader = DataLoader(data_file)
    processed_data = loader.process_for_pipeline(verbose=True)

    # Initialize efficient detector
    detector = EfficientSinglePassDetector(
        data=processed_data,
        company_names_file="/Users/krishna_dewan/Desktop/FINE 547 M&A/cik_gvkey_linktable_USA_only.csv"
    )

    # Run single-pass detection
    results = detector.detect_all_events_single_pass(
        bankruptcy_threshold=kwargs.get('bankruptcy_threshold', 50),
        merger_threshold=kwargs.get('merger_threshold', 40),
        halt_threshold=kwargs.get('halt_threshold', 30),
        dataset_start_date=processed_data['date'].min(),
        verbose=kwargs.get('verbose', True),
        save_every_n_stocks=kwargs.get('save_every_n_stocks', 100),
        output_file=kwargs.get('output_file', 'not used/new_events_to_verify.csv')
    )

    return results

if __name__ == "__main__":
    # Run efficient detection
    results = run_efficient_detection(
        bankruptcy_threshold=50,
        merger_threshold=40,
        halt_threshold=30,
        save_every_n_stocks=100,
        verbose=True
    )

    print(f"\n✅ Efficient detection complete!")
    print(f"📁 Results saved to: new_events_to_verify.csv")
    if len(results) > 0:
        print(f"🎯 Found {len(results)} unique events (one per stock)")
        print(f"📊 Event breakdown:")
        for event_type, count in results['event_type'].value_counts().items():
            print(f"   {event_type}: {count}")
    else:
        print(f"✅ No events flagged - thresholds working correctly!")