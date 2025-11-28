# M&A Event Detection and Verification Pipeline

This repository contains a complete pipeline for detecting and verifying mergers, acquisitions, bankruptcies, and other corporate events using stock price, volume, and firm-level characteristics.

## Overview

The pipeline is structured to:

1. Detect potential corporate events from market and accounting signals  
2. Verify detected events using an AI model (Claude) and web search  
3. Evaluate and diagnose the detection and prediction models  

The goal is to construct a reliable, research-quality M&A event dataset that can be used for empirical or machine learning applications.

---

## Key Files

### Main Execution Scripts

#### `run_analysis.py`
**Purpose:** Primary script for event detection.  
- Loads stock data from:  
  `/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa.parquet`
- Runs a single-pass detector that flags potential events (M&A, bankruptcy, halts)
- Key thresholds:  
  - `bankruptcy_threshold=55`  
  - `merger_threshold=50`  
  - `halt_threshold=999` (disabled)  
- Output: `ma_bankruptcy_events.csv`

#### `claude_verifier.py`
**Purpose:** Verifies detected events using Claude + web search.  
- Confirms or rejects flagged events  
- Adds fields such as:  
  - `verified`  
  - `actual_event_type`  
  - `llm_confidence`  
  - `details`  
  - `acquirer`  
  - `acquisition_price`  
- Supports resumable processing  
- Tracks API usage and costs

#### `model_diagnostics_comprehensive.py`
**Purpose:** Full diagnostic analysis of the event detection and ML prediction system.  
- Compares RF embeddings vs baseline XGBoost  
- Detects temporal leakage and survivorship issues  
- Evaluates calibration, top-K performance, and temporal stability  
- Produces diagnostic plots and recommendations  

---

## Supporting Libraries

### `efficient_single_pass_detector.py`
- Core detection engine  
- Processes each firm once  
- Detects multiple event types simultaneously  
- Applies conflict resolution to ensure a single event per firm  
- Saves progress periodically for safe interruption  

### `data_loader_helper.py`
- Handles loading and preprocessing of large parquet/CSV files  
- Uses chunking for memory efficiency  
- Standardizes data format for detection pipeline  

### `liquidity_enhanced_rules.py` (referenced)
- Advanced rule-based detection logic  
- Uses liquidity patterns, volume shifts, and discontinuities  
- Provides scoring functions for different event types  

---

## How to Use

### 1. Run Event Detection
```bash
python run_analysis.py
