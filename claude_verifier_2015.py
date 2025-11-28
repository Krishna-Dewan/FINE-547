#!/usr/bin/env python3
"""
2015-Only Event Verifier

This version ONLY re-verifies 2015 events:
- Re-checks all 2015 events (even if already verified)
- CLEARS actual_date if Claude cannot find exact date
- Only keeps dates that Claude can explicitly confirm

Usage:
    export CLAUDE_API_KEY=sk-ant-your-key-here
    cd "/Users/krishna_dewan/Desktop/FINE 547 M&A/Find acquisitions"
    python3 claude_verifier_2015_only.py
"""

import pandas as pd
import json
import time
import os
from typing import Dict, Tuple
import anthropic


class BudgetTracker:
    """Track API usage and enforce budget limits"""

    def __init__(self, max_budget: float = None):
        self.max_budget = max_budget
        self.total_spent = 0.0
        self.request_count = 0
        self.start_time = time.time()
        self.model_costs = {
            "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
        }

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def calculate_cost(self, model: str, input_text: str, output_text: str) -> float:
        if model not in self.model_costs:
            return 0.0003
        costs = self.model_costs[model]
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost

    def add_request(self, model: str, input_text: str, output_text: str) -> float:
        cost = self.calculate_cost(model, input_text, output_text)
        self.total_spent += cost
        self.request_count += 1
        return cost

    def check_budget(self) -> Tuple[bool, str]:
        if self.max_budget is None:
            return True, f"💰 Spent: ${self.total_spent:.4f}"

        if self.total_spent >= self.max_budget:
            return False, f"🚫 BUDGET LIMIT! ${self.total_spent:.4f} of ${self.max_budget:.2f}"

        remaining = self.max_budget - self.total_spent
        percent_used = (self.total_spent / self.max_budget) * 100

        if percent_used >= 90:
            return True, f"⚠️  {percent_used:.1f}% used"
        elif percent_used >= 75:
            return True, f"💰 {percent_used:.1f}% used"
        else:
            return True, f"💰 {percent_used:.1f}% used"

    def get_stats(self) -> str:
        elapsed = time.time() - self.start_time
        rate = self.request_count / elapsed * 60 if elapsed > 0 else 0
        return f"""
📊 USAGE STATISTICS:
   Requests: {self.request_count}
   Total: ${self.total_spent:.4f}
   Avg/request: ${self.total_spent / self.request_count:.4f}
   Rate: {rate:.1f} req/min
   Time: {elapsed / 60:.1f} min"""


def verify_event(event: pd.Series, client: anthropic.Anthropic,
                 model: str, budget_tracker: BudgetTracker = None) -> Dict:
    """Verify event and return actual_date ONLY if found"""

    gvkey = event.get('gvkey', 'Unknown')
    company_name = event.get('company_name', 'Unknown')
    event_type = event['event_type']
    event_date = pd.to_datetime(event['event_date'])

    prompt = f"""Verify this potential {event_type} for "{company_name}" (GVKEY: {gvkey}).

**Detected:**
- Company: {company_name}
- Detected as: {event_type}
- Detected date: {event_date.strftime('%Y-%m-%d')}

**CRITICAL TASK:**
Search the web for what ACTUALLY happened to "{company_name}" in 2015.

**Response format (JSON only, no markdown):**
{{
    "verified": true/false,
    "actual_event_type": "acquisition" | "merger" | "bankruptcy" | "none" | "unknown",
    "actual_date": "YYYY-MM-DD" OR null,
    "llm_confidence": "high" | "medium" | "low",
    "details": "Brief explanation",
    "acquirer": "Company name" or null,
    "acquisition_price": number or null
}}

**RULES:**
1. actual_date = null UNLESS you find EXACT date in web search
2. DO NOT use detected date unless you confirm it's correct
3. DO NOT guess dates
4. If no info found: verified=false, llm_confidence="low"
"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        if budget_tracker:
            budget_tracker.add_request(model, prompt, response_text)

        # Extract JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            start = response_text.find('{')
            if start != -1:
                brace_count = 0
                end = start
                for i, char in enumerate(response_text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                json_str = response_text[start:end]
            else:
                json_str = response_text.strip()

        result = json.loads(json_str)

        # Force actual_date to None if it's "null" string or empty
        if result.get('actual_date') in ['null', 'None', '', 'N/A', 'n/a']:
            result['actual_date'] = None

        return result

    except Exception as e:
        error_msg = str(e)
        if any(x in error_msg.lower() for x in ["rate_limit", "quota", "billing"]):
            return {
                'verified': False,
                'actual_event_type': 'error',
                'actual_date': None,
                'llm_confidence': 'low',
                'details': f'API error: {error_msg}',
                'stop_processing': True
            }
        else:
            return {
                'verified': False,
                'actual_event_type': 'error',
                'actual_date': None,
                'llm_confidence': 'low',
                'details': f'Error: {error_msg}'
            }


def main():
    csv_file = 'ma_bankruptcy_events_merged_v2.csv'

    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found")
        print(f"   Current directory: {os.getcwd()}")
        return

    print("=" * 80)
    print("2015-ONLY EVENT VERIFIER")
    print("=" * 80)
    print(f"\n📁 Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df):,} events")

    # Parse dates
    df['event_date_parsed'] = pd.to_datetime(df['event_date'], errors='coerce')
    df['event_year'] = df['event_date_parsed'].dt.year

    # ONLY 2015 events
    year_2015 = df['event_year'] == 2015
    events_to_process = df[year_2015].copy()

    print(f"\n🔍 Filtering to 2015 events ONLY...")
    print(f"   2015 events: {year_2015.sum():,}")

    if len(events_to_process) == 0:
        print("✅ No 2015 events found!")
        return

    # Filter to events with company names
    events_with_names = events_to_process[
        (events_to_process['company_name'].notna()) &
        (events_to_process['company_name'] != 'Unknown') &
        (events_to_process['company_name'] != '')
        ].copy()

    print(f"\n✅ Will process {len(events_with_names):,} 2015 events")

    # Cost estimate
    model = "claude-3-5-haiku-20241022"
    estimated_cost = len(events_with_names) * 0.0003
    print(f"\n💰 Estimated cost: ${estimated_cost:.2f}")
    print(f"⏱️  Estimated time: {len(events_with_names) * 0.5 / 60:.1f} minutes")

    # Check API key
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        print("\n❌ Error: Set CLAUDE_API_KEY environment variable")
        print("   export CLAUDE_API_KEY=sk-ant-...")
        return

    # Budget
    budget_input = input(f"\n💳 Budget limit? (press Enter for no limit, or amount like 1.00): ").strip()
    max_budget = None
    if budget_input and budget_input.replace('.', '').isdigit():
        max_budget = float(budget_input)
        print(f"✓ Budget: ${max_budget:.2f}")

    proceed = input("\n🤔 Proceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("❌ Cancelled")
        return

    # Initialize
    client = anthropic.Anthropic(api_key=api_key)
    budget_tracker = BudgetTracker(max_budget=max_budget)

    # Ensure columns exist
    for col in ['verified', 'actual_event_type', 'actual_date', 'llm_confidence',
                'details', 'acquirer', 'acquisition_price']:
        if col not in df.columns:
            df[col] = None

    print("\n" + "=" * 80)
    print("STARTING 2015 VERIFICATION")
    print("=" * 80)

    save_interval = 50
    resume_from = 613

    for i, (idx, event) in enumerate(events_with_names.iterrows()):
        if i < resume_from:
            continue

        company = event.get('company_name', 'Unknown')
        gvkey = event.get('gvkey', 'Unknown')
        event_date_str = event['event_date']

        print(f"\n[{i + 1}/{len(events_with_names)}] {company} ({gvkey})")
        print(f"  Detected: {event['event_type']} on {event_date_str}")

        # Check budget
        can_continue, budget_msg = budget_tracker.check_budget()
        if not can_continue:
            print(f"  {budget_msg}")
            print(f"  💾 Saving and stopping...")
            df.to_csv(csv_file, index=False)
            print(f"\n💳 BUDGET LIMIT REACHED at event {i + 1}")
            print(f"📊 To resume: Set resume_from = {i} in code")
            print(budget_tracker.get_stats())
            return

        # Verify
        result = verify_event(event, client, model, budget_tracker)

        if result.get('stop_processing'):
            print(f"  🚫 API error: {result.get('details')}")
            df.to_csv(csv_file, index=False)
            print(f"\n🚫 STOPPED at event {i + 1}")
            print(budget_tracker.get_stats())
            return

        # Update DataFrame
        df.loc[idx, 'verified'] = result.get('verified', False)
        df.loc[idx, 'actual_event_type'] = result.get('actual_event_type')
        df.loc[idx, 'llm_confidence'] = result.get('llm_confidence')
        df.loc[idx, 'details'] = result.get('details')
        df.loc[idx, 'acquirer'] = result.get('acquirer')
        df.loc[idx, 'acquisition_price'] = result.get('acquisition_price')

        # CRITICAL: Only set actual_date if Claude found one
        actual_date = result.get('actual_date')
        if actual_date and actual_date not in ['null', 'None', '', 'N/A']:
            df.loc[idx, 'actual_date'] = actual_date
            print(f"  ✅ {result.get('actual_event_type')} on {actual_date} ({result.get('llm_confidence')})")
        else:
            df.loc[idx, 'actual_date'] = None  # Clear it!
            print(f"  ⚠️  {result.get('actual_event_type')} but NO DATE FOUND (cleared)")

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  {budget_msg}")

        if (i + 1) % save_interval == 0:
            df.to_csv(csv_file, index=False)
            print(f"  💾 Saved progress ({i + 1}/{len(events_with_names)})")

        time.sleep(0.5)

    # Final save
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Final save: {csv_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - 2015 EVENTS")
    print("=" * 80)

    df_2015 = df[df['event_year'] == 2015]
    df_2015_verified = df_2015[df_2015['verified'] == True]
    df_2015_with_date = df_2015[df_2015['actual_date'].notna()]
    df_2015_no_date = df_2015[df_2015['actual_date'].isna()]

    print(f"\n2015 Events:")
    print(f"  Total in file: {len(df_2015):,}")
    print(f"  Verified: {len(df_2015_verified):,}")
    print(f"  With valid date: {len(df_2015_with_date):,}")
    print(f"  No date (cleared): {len(df_2015_no_date):,}")

    if len(df_2015_with_date) > 0:
        print(f"\n2015 Events with dates by type:")
        print(df_2015_with_date['actual_event_type'].value_counts())

    print(budget_tracker.get_stats())
    print(f"\n✅ Done!")
    print(f"\n💡 Next step: Re-run your labeling script to use the cleaned 2015 data")


if __name__ == "__main__":
    main()