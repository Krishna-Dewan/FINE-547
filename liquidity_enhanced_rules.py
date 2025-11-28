import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class StockStatus(Enum):
    """Enum for different stock lifecycle states"""
    ACTIVE = "active"
    DELISTED_BANKRUPTCY = "delisted_bankruptcy"
    DELISTED_MERGER = "delisted_merger"
    HALTED = "halted"
    PRE_IPO = "pre_ipo"


class LiquidityEnhancedDetector:
    """
    Advanced corporate event detection using liquidity and trading features.

    These features are MUCH better than just price drops because they capture:
    - Trading activity drying up (bankruptcy signal)
    - Liquidity evaporating (delisting warning)
    - Abnormal bid-ask spreads (distress indicator)
    - Zero trading days (death spiral)

    Uses aggressive thresholds to catch ALL potential events (LLM filters false positives).
    """

    def __init__(self, data: pd.DataFrame, company_names_file: str = None):
        """
        Initialize with data that includes liquidity features.

        Expected columns:
        - date, gvkey, prc (or price), volume (optional)
        - Plus any of these features (more is better):
          * ami_126d (Amihud Measure)
          * dolvol_var_126d (Coefficient of variation for dollar trading volume)
          * dolvol_126d (Dollar trading volume)
          * aliq_at (Liquidity of book assets)
          * aliq_mat (Liquidity of market assets)
          * bidaskhl_21d (The high-low bid-ask spread)
          * turnover_var_126d (Coefficient of variation for share turnover)
          * zero_trades_252d, zero_trades_21d, zero_trades_126d (Number of zero trades)
          * turnover_126d (Share turnover)
          * z_score (Altman Z-score)
          * o_score (Ohlson O-score)
          * kz_index (Kaplan-Zingales index)
          * ni_be (Return on equity)
          * ocf_at (Operating cash flow to assets)
          * at_be (Book leverage)
          * debt_me (Debt-to-market)
          * f_score (Pitroski F-score)
          * And many more...

        Args:
            data: Stock data DataFrame
            company_names_file: Path to cik_gvkey_linktable_USA_only.csv file
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])

        # Handle price column - use 'prc' if available, otherwise 'price'
        if 'prc' in self.data.columns and 'price' not in self.data.columns:
            self.data['price'] = self.data['prc']

        self.data = self.data.sort_values(['gvkey', 'date'])

        # Calculate dataset boundaries for proper event detection
        self.dataset_start = self.data['date'].min()
        self.dataset_end = self.data['date'].max()
        print(f"Dataset range: {self.dataset_start.date()} to {self.dataset_end.date()}")

        # Load company names mapping from multiple sources
        self.company_names = {}
        if company_names_file:
            try:
                # Load USA-only linktable first
                names_df = pd.read_csv(company_names_file)
                # Create mapping from gvkey to most recent company name
                names_df = names_df.sort_values(['gvkey', 'datadate']).groupby('gvkey').last()
                self.company_names = dict(zip(names_df.index, names_df['conm']))
                usa_count = len(self.company_names)
                print(f"✓ Loaded {usa_count} company names from USA linktable")

                # Load GlobalGVKEY.csv for additional international names
                try:
                    import os
                    global_file = '/Users/krishna_dewan/Desktop/FINE 547 M&A/GlobalGVKEY.csv'
                    if os.path.exists(global_file):
                        global_df = pd.read_csv(global_file)

                        # Clean and process GlobalGVKEY data
                        # Convert datadate format if needed (31/1/06 -> datetime)
                        if 'datadate' in global_df.columns:
                            # Handle different date formats
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
                                       if gvkey not in self.company_names}

                            self.company_names.update(new_names)
                            print(f"✓ Added {len(new_names)} additional company names from GlobalGVKEY")
                        else:
                            print("⚠️ No valid company names found in GlobalGVKEY.csv")
                    else:
                        print("ℹ️ GlobalGVKEY.csv not found, using USA-only names")
                except Exception as e:
                    print(f"⚠️ Could not load GlobalGVKEY.csv: {e}")

                print(f"✓ Total company names available: {len(self.company_names)}")

            except Exception as e:
                print(f"⚠️ Could not load company names: {e}")
                self.company_names = {}

    def detect_bankruptcy_advanced(self,
                                   bankruptcy_threshold: int = 50,  # INCREASED: Monthly data needs higher thresholds
                                   verbose: bool = False,
                                   save_every_n_chunks: int = 10,
                                   output_file: str = 'bankruptcy_progress.csv') -> pd.DataFrame:
        """
        Advanced bankruptcy detection using liquidity signals.

        AGGRESSIVE MODE:
        Threshold = 15 points (catches ~95%+ of real bankruptcies)

        A stock heading toward bankruptcy shows these patterns:
        1. Price drops (but maybe only 60-80%, not 95%)
        2. Amihud illiquidity (ami_126d) SPIKES (nobody wants to trade it)
        3. Zero trading days (zero_trades_252d) increase dramatically
        4. Bid-ask spread (bidaskhl_21d) WIDENS (dealers abandon it)
        5. Share turnover (turnover_126d) COLLAPSES (activity dies)
        6. Altman Z-score (z_score) < 1.8 (bankruptcy zone)
        7. Negative profitability (ni_be, ocf_at)

        Args:
            bankruptcy_threshold: Minimum score to flag (default 15 - aggressive)
            verbose: Print progress

        Returns:
            DataFrame of detected bankruptcy candidates
        """
        bankruptcy_signals = []

        gvkeys = self.data['gvkey'].unique()
        total = len(gvkeys)
        chunk_size = save_every_n_chunks * 10  # Process 10 stocks per "chunk"

        for idx, gvkey in enumerate(gvkeys):
            if verbose and idx % 100 == 0:
                print(f"  Processing {idx}/{total} stocks...")

            # Save progress every chunk_size stocks
            if idx > 0 and idx % chunk_size == 0 and len(bankruptcy_signals) > 0:
                temp_df = pd.DataFrame(bankruptcy_signals)
                progress_file = output_file.replace('.csv', f'_bankruptcy_progress_{idx}.csv')
                temp_df.to_csv(progress_file, index=False)
                if verbose:
                    print(f"    💾 Saved bankruptcy progress: {progress_file} ({len(bankruptcy_signals)} events so far)")

            stock = self.data[self.data['gvkey'] == gvkey].copy()

            if len(stock) < 24:  # Need at least 2 years of monthly data
                continue

            # Get last 24 months (2 years) for monthly data
            recent = stock.tail(24)

            # === SCORING SYSTEM ===
            score = 0
            reasons = []

            # === SIGNAL 1: Financial Distress (40 points max) ===
            if 'z_score' in recent.columns:
                z_score = recent['z_score'].iloc[-1]
                if pd.notna(z_score):
                    if z_score < 1.8:
                        score += 15
                        reasons.append(f"Altman Z-score {z_score:.2f} (bankruptcy zone)")
                    elif z_score < 3.0:
                        score += 8
                        reasons.append(f"Altman Z-score {z_score:.2f} (gray zone)")

            if 'o_score' in recent.columns:
                o_score = recent['o_score'].iloc[-1]
                if pd.notna(o_score):
                    if o_score > 0.5:
                        score += 15
                        reasons.append(f"Ohlson O-score {o_score:.2f} (high risk)")
                    elif o_score > 0.3:
                        score += 8
                        reasons.append(f"Ohlson O-score {o_score:.2f} (elevated risk)")

            if 'kz_index' in recent.columns:
                kz = recent['kz_index'].iloc[-1]
                if pd.notna(kz) and kz < -3:
                    score += 10
                    reasons.append(f"KZ index {kz:.2f} (constrained)")

            # === SIGNAL 2: Profitability Collapse (30 points max) ===
            if 'ni_be' in recent.columns:
                roe = recent['ni_be'].iloc[-1]
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

            if 'ocf_at' in recent.columns:
                ocf = recent['ocf_at'].iloc[-1]
                if pd.notna(ocf):
                    if ocf < -0.05:
                        score += 10
                        reasons.append(f"Operating CF {ocf * 100:.1f}%")
                    elif ocf < 0:
                        score += 5
                        reasons.append(f"Operating CF negative")

            if 'ebitda_mev' in recent.columns:
                ebitda_ev = recent['ebitda_mev'].iloc[-1]
                if pd.notna(ebitda_ev) and ebitda_ev < 0:
                    score += 8
                    reasons.append("Negative EBITDA/EV")

            # === SIGNAL 3: Liquidity Death Spiral (30 points max) ===
            if 'zero_trades_252d' in recent.columns:
                zero_trades = recent['zero_trades_252d'].iloc[-1]
                if pd.notna(zero_trades):
                    if zero_trades > 90:  # More than 3 months for monthly data
                        score += 10
                        reasons.append(f"{zero_trades:.0f} zero-trade days (>3m)")
                    elif zero_trades > 60:  # More than 2 months
                        score += 5
                        reasons.append(f"{zero_trades:.0f} zero-trade days (>2m)")
                    elif zero_trades > 30:  # More than 1 month
                        score += 3
                        reasons.append(f"{zero_trades:.0f} zero-trade days (>1m)")

            if 'ami_126d' in recent.columns:
                recent_illiq = recent['ami_126d'].iloc[-12:].mean()  # Last 12 months for monthly data
                hist_illiq = stock['ami_126d'].quantile(0.50)
                if pd.notna(recent_illiq) and pd.notna(hist_illiq) and hist_illiq > 0:
                    illiq_spike = recent_illiq / hist_illiq
                    if illiq_spike > 5.0:  # Higher threshold for monthly data
                        score += 8
                        reasons.append(f"Illiquidity spike {illiq_spike:.1f}x")

            if 'bidaskhl_21d' in recent.columns:
                spread = recent['bidaskhl_21d'].iloc[-1]
                spread_85th = self.data['bidaskhl_21d'].quantile(0.85)
                if pd.notna(spread) and pd.notna(spread_85th) and spread > spread_85th:
                    score += 5
                    reasons.append(f"Wide bid-ask spread (>85th pct)")

            if 'turnover_126d' in recent.columns:
                turnover = recent['turnover_126d'].iloc[-12:].mean()  # Last 12 months for monthly data
                hist_turnover = stock['turnover_126d'].quantile(0.50)
                if pd.notna(turnover) and pd.notna(hist_turnover) and hist_turnover > 0:
                    turnover_decline = 1 - (turnover / hist_turnover)
                    if turnover_decline > 0.60:  # Higher threshold for monthly data
                        score += 5
                        reasons.append(f"Turnover declined {turnover_decline * 100:.0f}%")

            # === SIGNAL 4: Balance Sheet Stress (20 points max) ===
            if 'at_be' in recent.columns:
                leverage = recent['at_be'].iloc[-1]
                if pd.notna(leverage):
                    if leverage > 0.7:
                        score += 8
                        reasons.append(f"High leverage {leverage:.2f}")

            if 'debt_me' in recent.columns:
                dtm = recent['debt_me'].iloc[-1]
                if pd.notna(dtm) and dtm > 1.5:
                    score += 5
                    reasons.append(f"Debt-to-market {dtm:.2f}")

            # === SIGNAL 5: Quality Deterioration (15 points max) ===
            if 'f_score' in recent.columns:
                f_score = recent['f_score'].iloc[-1]
                if pd.notna(f_score):
                    if f_score < 3:
                        score += 10
                        reasons.append(f"Pitroski F-score {f_score:.0f}")
                    elif f_score < 4:
                        score += 5
                        reasons.append(f"Pitroski F-score {f_score:.0f}")

            if 'ni_ar1' in recent.columns:
                ep = recent['ni_ar1'].iloc[-1]
                if pd.notna(ep) and ep < 0:
                    score += 5
                    reasons.append("Negative earnings persistence")

            # === SIGNAL 6: Trading Stopped (30 points max) ===
            last_date = recent['date'].iloc[-1]
            # FIXED: Use dataset end instead of current date to avoid end-of-sample bias
            days_since_last = (self.dataset_end - last_date).days

            # Only flag if stock stopped trading BEFORE dataset end (not at the natural end)
            if days_since_last > 180 and last_date < self.dataset_end - pd.Timedelta(days=90):  # 6+ months gap AND stopped before dataset end
                score += 20
                reasons.append(f"Stopped trading {days_since_last} days before dataset end")
            elif days_since_last > 120 and last_date < self.dataset_end - pd.Timedelta(days=60):  # 4+ months gap
                score += 10
                reasons.append(f"Stopped trading {days_since_last} days before dataset end")

            # === SIGNAL 7: Price Collapse (20 points max) ===
            price_change = (recent['price'].iloc[-1] - recent['price'].iloc[0]) / recent['price'].iloc[0]
            if price_change < -0.80:
                score += 15
                reasons.append(f"Price dropped {price_change * 100:.1f}%")
            elif price_change < -0.60:
                score += 8
                reasons.append(f"Price dropped {price_change * 100:.1f}%")
            elif price_change < -0.40:
                score += 4
                reasons.append(f"Price dropped {price_change * 100:.1f}%")

            # === SIGNAL 8: Dollar Volume Collapse (15 points max) ===
            if 'dolvol_126d' in recent.columns:
                recent_vol = recent['dolvol_126d'].iloc[-12:].mean()  # Last 12 months for monthly data
                hist_vol = stock['dolvol_126d'].quantile(0.50)
                if pd.notna(recent_vol) and pd.notna(hist_vol) and hist_vol > 0:
                    vol_collapse = 1 - (recent_vol / hist_vol)
                    if vol_collapse > 0.80:  # Higher threshold for monthly data
                        score += 8
                        reasons.append(f"Volume collapsed {vol_collapse * 100:.0f}%")
                    elif vol_collapse > 0.70:
                        score += 4
                        reasons.append(f"Volume collapsed {vol_collapse * 100:.0f}%")

            # === DECISION: Flag if score >= threshold ===
            if score >= bankruptcy_threshold:
                confidence = 'high' if score >= 60 else 'medium' if score >= 40 else 'low'

                bankruptcy_signals.append({
                    'gvkey': gvkey,
                    'company_name': self.company_names.get(gvkey, 'Unknown'),
                    'event_type': 'bankruptcy',
                    'event_date': last_date,
                    'detection_score': score,
                    'confidence': confidence,
                    'reasons': '; '.join(reasons),
                    'last_price': recent['price'].iloc[-1],
                    'price_change_60d': price_change * 100
                })

        return pd.DataFrame(bankruptcy_signals)

    def detect_merger_advanced(self,
                               merger_threshold: int = 40,  # INCREASED: Monthly data needs higher thresholds
                               verbose: bool = False) -> pd.DataFrame:
        """
        Advanced merger detection using liquidity patterns.

        AGGRESSIVE MODE: Threshold = 10 points

        Key difference from bankruptcy: Price STABLE + Activity NORMAL until sudden stop

        Args:
            merger_threshold: Minimum score to flag (default 10 - aggressive)
            verbose: Print progress

        Returns:
            DataFrame of detected merger candidates
        """
        merger_signals = []

        gvkeys = self.data['gvkey'].unique()
        total = len(gvkeys)

        for idx, gvkey in enumerate(gvkeys):
            if verbose and idx % 100 == 0:
                print(f"  Processing {idx}/{total} stocks...")

            stock = self.data[self.data['gvkey'] == gvkey].copy()

            if len(stock) < 24:  # Need at least 2 years of monthly data
                continue

            recent = stock.tail(24)  # Last 2 years for monthly data

            # Check if stopped trading - FIXED: Use dataset end, not current date
            last_date = recent['date'].iloc[-1]
            days_since_last = (self.dataset_end - last_date).days

            # Only consider for merger if stopped trading BEFORE dataset naturally ends
            if days_since_last < 120 or last_date >= self.dataset_end - pd.Timedelta(days=60):
                continue  # Still trading or stopped at natural dataset end

            # === SCORING SYSTEM ===
            score = 0
            reasons = []

            # === SIGNAL 1: Price Stability (20 points max) ===
            price_change = (recent['price'].iloc[-1] - recent['price'].iloc[0]) / recent['price'].iloc[0]

            if -0.15 < price_change < 0.50:
                score += 20
                reasons.append(f"Price stable ({price_change * 100:.1f}%)")
            elif -0.25 < price_change < 1.00:
                score += 10
                reasons.append(f"Price moderately stable ({price_change * 100:.1f}%)")

            # === SIGNAL 2: Volume Activity (15 points max) ===
            if 'dolvol_126d' in recent.columns:
                recent_vol = recent['dolvol_126d'].iloc[-6:].mean()  # Last 6 months for monthly data
                hist_vol = stock['dolvol_126d'].quantile(0.50)
                if pd.notna(recent_vol) and pd.notna(hist_vol) and hist_vol > 0:
                    vol_ratio = recent_vol / hist_vol
                    if vol_ratio > 2.0:  # Higher threshold for monthly data
                        score += 15
                        reasons.append(f"Volume spike {vol_ratio:.1f}x")
                    elif vol_ratio > 1.5:
                        score += 8
                        reasons.append(f"Volume elevated {vol_ratio:.1f}x")

            # === SIGNAL 3: Liquidity Stays Healthy (10 points max) ===
            if 'bidaskhl_21d' in recent.columns:
                recent_spread = recent['bidaskhl_21d'].iloc[-6:].mean()  # Last 6 months for monthly data
                hist_spread = stock['bidaskhl_21d'].quantile(0.50)
                if pd.notna(recent_spread) and pd.notna(hist_spread) and hist_spread > 0:
                    spread_change = (recent_spread - hist_spread) / hist_spread
                    if spread_change < -0.20:  # Less stringent for monthly data
                        score += 10
                        reasons.append("Bid-ask spread tightened")

            if 'zero_trades_21d' in recent.columns:
                zero_trades = recent['zero_trades_21d'].iloc[-1]
                if pd.notna(zero_trades) and zero_trades < 5:
                    score += 8
                    reasons.append("Active trading (low zero-trade days)")

            # === SIGNAL 4: Financial Health (15 points max) ===
            if 'ni_be' in recent.columns:
                roe = recent['ni_be'].iloc[-1]
                if pd.notna(roe) and roe > 0:
                    score += 8
                    reasons.append("Positive ROE")

            if 'ocf_at' in recent.columns:
                ocf = recent['ocf_at'].iloc[-1]
                if pd.notna(ocf) and ocf > 0:
                    score += 8
                    reasons.append("Positive operating CF")

            if 'z_score' in recent.columns:
                z_score = recent['z_score'].iloc[-1]
                if pd.notna(z_score) and z_score > 1.8:
                    score += 5
                    reasons.append(f"Healthy Z-score {z_score:.2f}")

            # === SIGNAL 5: Payout Changes (10 points max) ===
            if 'eqnpo_me' in recent.columns:
                recent_payout = recent['eqnpo_me'].iloc[-5:].mean()
                hist_payout = stock['eqnpo_me'].iloc[:-5].mean()
                if pd.notna(recent_payout) and pd.notna(hist_payout):
                    if recent_payout > hist_payout:
                        score += 5
                        reasons.append("Payout increased")

            # === SIGNAL 6: Trading Stopped (15 points max) ===
            # Only award points if actually stopped before dataset end
            if days_since_last > 120 and last_date < self.dataset_end - pd.Timedelta(days=60):
                score += 15
                reasons.append(f"Trading stopped {days_since_last} days before dataset end")

            # === ANTI-BANKRUPTCY FILTER ===
            # Don't flag as merger if it looks like bankruptcy
            bankruptcy_score = 0

            if 'z_score' in recent.columns:
                z = recent['z_score'].iloc[-1]
                if pd.notna(z) and z < 1.8:
                    bankruptcy_score += 15

            if price_change < -0.50:
                bankruptcy_score += 15

            if 'ni_be' in recent.columns:
                roe = recent['ni_be'].iloc[-1]
                if pd.notna(roe) and roe < -0.10:
                    bankruptcy_score += 10

            # === DECISION: Flag if score >= threshold AND not bankruptcy ===
            if score >= merger_threshold and bankruptcy_score < 30:
                confidence = 'medium' if score >= 40 else 'low'

                merger_signals.append({
                    'gvkey': gvkey,
                    'company_name': self.company_names.get(gvkey, 'Unknown'),
                    'event_type': 'merger',
                    'event_date': last_date,
                    'detection_score': score,
                    'confidence': confidence,
                    'reasons': '; '.join(reasons),
                    'last_price': recent['price'].iloc[-1],
                    'price_change_60d': price_change * 100
                })

        return pd.DataFrame(merger_signals)

    def detect_halt_advanced(self,
                             halt_threshold: int = 30,  # INCREASED: Monthly data needs higher thresholds
                             verbose: bool = False) -> pd.DataFrame:
        """
        Advanced halt detection - gaps in data that resumed.

        AGGRESSIVE MODE: Threshold = 5 points (catches all gaps >=3 days)

        Key pattern: Gap → Trading RESUMES

        Args:
            halt_threshold: Minimum score to flag (default 5 - very aggressive)
            verbose: Print progress

        Returns:
            DataFrame of detected halt candidates
        """
        halt_signals = []

        gvkeys = self.data['gvkey'].unique()
        total = len(gvkeys)

        for idx, gvkey in enumerate(gvkeys):
            if verbose and idx % 100 == 0:
                print(f"  Processing {idx}/{total} stocks...")

            stock = self.data[self.data['gvkey'] == gvkey].copy()

            if len(stock) < 10:
                continue

            # Find gaps and merge consecutive ones - MONTHLY DATA NEEDS LARGER GAPS
            stock['days_diff'] = stock['date'].diff().dt.days
            stock['has_gap'] = stock['days_diff'] >= 120  # 4+ months gap for monthly data (more conservative)

            # Find halt periods (consecutive gaps)
            halt_periods = []
            current_halt = None

            for idx, row in stock.iterrows():
                if row['has_gap']:
                    if current_halt is None:
                        # Start new halt period
                        prev_date = stock[stock['date'] < row['date']]['date'].iloc[-1] if len(
                            stock[stock['date'] < row['date']]) > 0 else row['date']
                        current_halt = {
                            'start_date': prev_date,
                            'end_date': row['date'],
                            'total_gap_days': row['days_diff'],
                            'gap_count': 1
                        }
                    else:
                        # Extend current halt period
                        current_halt['end_date'] = row['date']
                        current_halt['total_gap_days'] += row['days_diff']
                        current_halt['gap_count'] += 1
                else:
                    # Trading resumed - close current halt if exists
                    if current_halt is not None:
                        current_halt['resumed'] = True
                        halt_periods.append(current_halt)
                        current_halt = None

            # Handle case where halt continues to end of data
            if current_halt is not None:
                current_halt['resumed'] = False
                halt_periods.append(current_halt)

            # Score each halt period
            for halt_period in halt_periods:
                score = 0
                reasons = []

                total_days = halt_period['total_gap_days']
                gap_count = halt_period['gap_count']

                # === SCORING ===
                # Gap detected - INCREASED threshold for monthly data
                if total_days >= 120:  # 4+ months for monthly data
                    score += 10
                    if gap_count > 1:
                        reasons.append(f"{total_days:.0f}-day halt period ({gap_count} consecutive gaps)")
                    else:
                        reasons.append(f"{total_days:.0f}-day gap")

                # Trading resumed (CRITICAL for halt vs delisting)
                if halt_period['resumed']:
                    score += 15
                    reasons.append("Trading resumed")

                # Length-based scoring - ADJUSTED for monthly data
                if total_days < 180:  # Less than 6 months
                    score += 10
                    reasons.append("Short halt (likely temporary)")
                elif total_days < 365:  # Less than 1 year
                    score += 5
                    reasons.append("Medium halt")

                # Multiple consecutive gaps indicate systematic halt
                if gap_count > 1:
                    score += 5
                    reasons.append(f"Multiple consecutive gaps ({gap_count})")

                if score >= halt_threshold:
                    confidence = 'medium' if score >= 20 else 'low'

                    halt_signals.append({
                        'gvkey': gvkey,
                        'company_name': self.company_names.get(gvkey, 'Unknown'),
                        'event_type': 'halt',
                        'event_date': halt_period['start_date'],
                        'halt_end_date': halt_period['end_date'],
                        'detection_score': score,
                        'confidence': confidence,
                        'reasons': '; '.join(reasons),
                        'total_gap_days': total_days,
                        'consecutive_gaps': gap_count
                    })

        return pd.DataFrame(halt_signals)

    def detect_ipo(self,
                   dataset_start_date: datetime,
                   verbose: bool = False) -> pd.DataFrame:
        """
        Detect IPOs - first appearance in dataset.

        This is DETERMINISTIC - no scoring needed.

        Args:
            dataset_start_date: Start date of your dataset
            verbose: Print progress

        Returns:
            DataFrame of detected IPOs
        """
        ipo_signals = []

        gvkeys = self.data['gvkey'].unique()
        total = len(gvkeys)

        for idx, gvkey in enumerate(gvkeys):
            if verbose and idx % 100 == 0:
                print(f"  Processing {idx}/{total} stocks...")

            stock = self.data[self.data['gvkey'] == gvkey].copy()

            first_date = stock['date'].min()

            # MUCH MORE RESTRICTIVE IPO DETECTION FOR MONTHLY DATA
            # Only flag as IPO if stock appears significantly AFTER dataset start
            # AND shows other IPO characteristics

            months_after_start = (first_date - dataset_start_date).days / 30.0

            # Skip if too close to dataset start (likely dataset boundary, not real IPO)
            if months_after_start < 18:  # Less than 1.5 years after dataset start (more conservative)
                continue  # Don't flag as any type of IPO

            # Check for young firm characteristics
            is_ipo = False
            confidence = 'low'

            if 'age' in stock.columns and len(stock) > 0:
                firm_age = stock.iloc[0]['age']
                if pd.notna(firm_age) and firm_age < 3:  # Very young firm
                    is_ipo = True
                    confidence = 'high'
                    reasons = f"Young firm (age {firm_age:.1f}y), appeared {months_after_start:.1f} months after dataset start"

            # Only flag if we have strong evidence of IPO
            if is_ipo:
                ipo_signals.append({
                    'gvkey': gvkey,
                    'company_name': self.company_names.get(gvkey, 'Unknown'),
                    'event_type': 'ipo',
                    'event_date': first_date,
                    'detection_score': 100,  # Deterministic
                    'confidence': confidence,
                    'reasons': reasons,
                    'first_price': stock.iloc[0]['price']
                })

        return pd.DataFrame(ipo_signals)

    def detect_stock_split(self, verbose: bool = False) -> pd.DataFrame:
        """
        Detect stock splits - sudden price drops of 50%, 33%, 25%

        Args:
            verbose: Print progress

        Returns:
            DataFrame of detected stock splits
        """
        split_signals = []

        gvkeys = self.data['gvkey'].unique()
        total = len(gvkeys)

        for idx, gvkey in enumerate(gvkeys):
            if verbose and idx % 100 == 0:
                print(f"  Processing {idx}/{total} stocks...")

            stock = self.data[self.data['gvkey'] == gvkey].copy()

            stock['price_change'] = stock['price'].pct_change()

            # Common split ratios
            split_ratios = {
                -0.50: '2:1 split',
                -0.67: '3:1 split',
                -0.75: '4:1 split'
            }

            for ratio, split_type in split_ratios.items():
                # Find price drops matching split ratio (within 2%)
                matches = stock[abs(stock['price_change'] - ratio) < 0.02]

                for match_idx, match_row in matches.iterrows():
                    score = 15
                    reasons = [f"Price drop {match_row['price_change'] * 100:.1f}% (matches {split_type})"]

                    # Check volume spike
                    row_position = stock.index.get_loc(match_idx)
                    if row_position > 0 and 'volume' in stock.columns:
                        prev_vol = stock.iloc[row_position - 1]['volume']
                        curr_vol = match_row['volume']
                        if pd.notna(prev_vol) and pd.notna(curr_vol) and prev_vol > 0:
                            if curr_vol > prev_vol * 1.5:
                                score += 10
                                reasons.append("Volume spike")

                    split_signals.append({
                        'gvkey': gvkey,
                        'company_name': self.company_names.get(gvkey, 'Unknown'),
                        'event_type': 'stock_split',
                        'event_date': match_row['date'],
                        'detection_score': score,
                        'confidence': 'medium',
                        'reasons': '; '.join(reasons),
                        'split_type': split_type
                    })

        return pd.DataFrame(split_signals)

    def detect_data_errors(self, verbose: bool = False) -> pd.DataFrame:
        """
        Detect data quality issues.

        Args:
            verbose: Print progress

        Returns:
            DataFrame of detected data errors
        """
        error_signals = []

        gvkeys = self.data['gvkey'].unique()
        total = len(gvkeys)

        for idx, gvkey in enumerate(gvkeys):
            if verbose and idx % 100 == 0:
                print(f"  Processing {idx}/{total} stocks...")

            stock = self.data[self.data['gvkey'] == gvkey].copy()

            for row_idx, row in stock.iterrows():
                score = 0
                reasons = []

                # Negative prices
                if row['price'] < 0:
                    score += 20
                    reasons.append("Negative price")

                # Extreme price jumps - penny stock tolerant
                row_position = stock.index.get_loc(row_idx)
                if row_position > 0:
                    prev_price = stock.iloc[row_position - 1]['price']
                    if prev_price > 0:
                        price_change = (row['price'] - prev_price) / prev_price
                        abs_change = abs(price_change)

                        # Dynamic thresholds based on price level
                        if prev_price >= 20:  # Large cap stocks ($20+)
                            jump_threshold = 5.0  # 500% max reasonable jump
                        elif prev_price >= 5:  # Mid-tier stocks ($5-$20)
                            jump_threshold = 10.0  # 1000% max reasonable jump
                        elif prev_price >= 1:  # Small stocks ($1-$5)
                            jump_threshold = 20.0  # 2000% max reasonable jump
                        elif prev_price >= 0.10:  # Penny stocks ($0.10-$1)
                            jump_threshold = 50.0  # 5000% max reasonable jump
                        else:  # Sub-penny stocks (< $0.10)
                            jump_threshold = 100.0  # 10000% max reasonable jump

                        if abs_change > jump_threshold:
                            score += 15
                            reasons.append(f"Extreme jump {price_change * 100:.0f}%")

                # Zero volume - penny stock tolerant
                if 'volume' in row and pd.notna(row['volume']) and row['volume'] == 0:
                    # Only penalize zero volume for higher-priced stocks
                    if row['price'] >= 5.0:  # $5+ stocks should trade daily
                        score += 8
                        reasons.append("Zero volume (unusual for higher-priced stock)")
                    elif row['price'] >= 1.0:  # $1-$5 stocks - mild penalty
                        score += 3
                        reasons.append("Zero volume (somewhat unusual)")
                    # Penny stocks < $1: Zero volume is normal, no penalty

                if score >= 10:
                    error_signals.append({
                        'gvkey': gvkey,
                        'company_name': self.company_names.get(gvkey, 'Unknown'),
                        'event_type': 'data_error',
                        'event_date': row['date'],
                        'detection_score': score,
                        'confidence': 'high',
                        'reasons': '; '.join(reasons),
                        'price': row['price']
                    })

        return pd.DataFrame(error_signals)

    def detect_all_events_comprehensive(self,
                                        bankruptcy_threshold: int = 15,
                                        merger_threshold: int = 10,
                                        halt_threshold: int = 5,
                                        dataset_start_date: Optional[datetime] = None,
                                        verbose: bool = True,
                                        save_every_n_chunks: int = 10,
                                        output_file: str = 'partial_detected_events.csv') -> Dict[str, pd.DataFrame]:
        """
        Run ALL detection methods and return comprehensive results.

        This is the MAIN function to use - runs everything!

        Args:
            bankruptcy_threshold: Minimum score for bankruptcy (default 15 - aggressive)
            merger_threshold: Minimum score for merger (default 10 - aggressive)
            halt_threshold: Minimum score for halt (default 5 - very aggressive)
            dataset_start_date: Start of your dataset (for IPO detection)
            verbose: Print progress

        Returns:
            Dictionary with different event types detected
        """
        if verbose:
            print("\n" + "=" * 70)
            print("BANKRUPTCY & M&A DETECTION")
            print("=" * 70)
            print(f"Analyzing {self.data['gvkey'].nunique()} unique gvkeys...")
            print("Focus: Bankruptcies and Mergers/Acquisitions only")

        results = {}

        # Detect bankruptcies
        if verbose:
            print("\n1. Detecting bankruptcies...")
        results['bankruptcies'] = self.detect_bankruptcy_advanced(
            bankruptcy_threshold=bankruptcy_threshold,
            verbose=verbose,
            save_every_n_chunks=save_every_n_chunks,
            output_file=output_file
        )
        if verbose:
            print(f"   Found: {len(results['bankruptcies'])} bankruptcy signals")

        # Detect mergers
        if verbose:
            print("\n2. Detecting mergers/acquisitions...")
        results['mergers'] = self.detect_merger_advanced(
            merger_threshold=merger_threshold,
            verbose=verbose
        )
        if verbose:
            print(f"   Found: {len(results['mergers'])} merger signals")

        # SKIPPING other event types - only detecting bankruptcies and mergers/acquisitions
        if verbose:
            print("\n✓ Skipping halts, IPOs, stock splits, and data errors")
            print("   Focus: Bankruptcies and M&A only")

        # Combine all events
        all_events_list = []

        for event_type, df in results.items():
            if len(df) > 0:
                df_copy = df.copy()
                # Ensure all have the same columns
                required_cols = ['gvkey', 'company_name', 'event_type', 'event_date', 'detection_score', 'confidence', 'reasons']
                for col in required_cols:
                    if col not in df_copy.columns:
                        df_copy[col] = None
                all_events_list.append(df_copy[required_cols])

        if all_events_list:
            results['all_events'] = pd.concat(all_events_list, ignore_index=True)

            # Save final results
            if verbose:
                print(f"\n💾 Saving final results to: {output_file}")
            results['all_events'].to_csv(output_file.replace('.csv', '_FINAL.csv'), index=False)

        else:
            results['all_events'] = pd.DataFrame(
                columns=['gvkey', 'company_name', 'event_type', 'event_date', 'detection_score', 'confidence', 'reasons'])

        if verbose:
            print("\n" + "=" * 70)
            print("DETECTION SUMMARY")
            print("=" * 70)
            print(f"Total events detected: {len(results['all_events'])}")
            if len(results['all_events']) > 0:
                print(f"\nBy type:")
                print(results['all_events']['event_type'].value_counts())
                print(f"\nBy confidence:")
                print(results['all_events']['confidence'].value_counts())

        return results

