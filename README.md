# M&A Event Detection and Verification Pipeline

This repository contains a comprehensive pipeline for detecting and verifying M&A (mergers & acquisitions), bankruptcy, and other corporate events from stock price and volume data.

## 🎯 Overview

The pipeline consists of several Python scripts that work together to:
1. **Detect** potential corporate events from stock price anomalies
2. **Verify** detected events using AI (Claude) and web search
3. **Analyze** the quality and performance of the detection models

## 📁 Key Files

### Main Execution Scripts

#### `run_analysis.py`
**Purpose**: Main entry point for the detection pipeline
- Loads stock price data from `/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa.parquet`
- Uses efficient single-pass detection to find M&A/bankruptcy events
- Configurable thresholds for different event types:
  - `bankruptcy_threshold=55` (very selective)
  - `merger_threshold=50` (very selective)
  - `halt_threshold=999` (disabled)
- Outputs: `ma_bankruptcy_events.csv`

#### `claude_verifier.py`
**Purpose**: AI-powered verification of detected events
- Takes detected events and verifies them using Claude AI + web search
- Budget tracking and cost estimation
- Resumable processing (can stop/restart from specific points)
- Updates CSV files with verification results:
  - `verified` (true/false)
  - `actual_event_type` (bankruptcy/merger/acquisition/etc.)
  - `llm_confidence` (high/medium/low)
  - `details` (explanation of what happened)
  - `acquirer` (if acquisition)
  - `acquisition_price` (if known)

#### `model_diagnostics_comprehensive.py`
**Purpose**: Comprehensive model diagnostics and performance analysis
- Compares current RF embedding model vs simple XGBoost baseline
- Detects temporal leakage, data quality issues, survivorship bias
- Analyzes top-K precision, calibration, temporal stability
- Provides actionable recommendations for model improvement
- Outputs detailed diagnostic reports and visualizations

### Supporting Libraries

#### `efficient_single_pass_detector.py`
- Core detection engine that processes each stock once
- Runs multiple event detections simultaneously
- Smart conflict resolution to minimize duplicate events per stock
- Progress saving every N stocks for resumability

#### `data_loader_helper.py`
- Handles loading and preprocessing of parquet/CSV stock data
- Smart chunking for large files (>1GB)
- Memory management and optimization
- Data format standardization for the pipeline

#### `liquidity_enhanced_rules.py` (referenced)
- Advanced rule-based detection algorithms
- Incorporates liquidity, volume, and price patterns
- Multiple scoring methods for different event types

## 🚀 How to Use

### 1. Event Detection
```bash
python run_analysis.py
```
This will:
- Process your stock data
- Detect potential M&A/bankruptcy events
- Save results to `ma_bankruptcy_events.csv`
- Show summary statistics

### 2. Event Verification
```bash
export CLAUDE_API_KEY=sk-ant-your-api-key-here
python claude_verifier.py
```
This will:
- Load detected events from CSV
- Verify each event using Claude AI + web search
- Update the same CSV with verification results
- Track API costs and usage

### 3. Model Diagnostics
```bash
python model_diagnostics_comprehensive.py
```
This will:
- Analyze your ML model performance
- Compare RF embeddings vs baseline XGBoost
- Check for data quality issues and leakage
- Generate diagnostic plots and recommendations

## 📊 Output Files

- `ma_bankruptcy_events.csv` - Raw detected events
- `ma_bankruptcy_events_merged_v2.csv` - Verified events with AI analysis
- `diagnostics/` folder - Model diagnostic reports and plots
- `ml_outputs/` folder - Trained models and predictions

## 💰 Cost Estimation

The verification step uses Claude AI which costs approximately:
- ~$0.0003 per event verification
- For 1000 events: ~$0.30
- Budget limits can be set to control spending

## 🔧 Configuration

### Detection Thresholds
In `run_analysis.py`, adjust these for sensitivity:
```python
bankruptcy_threshold=55    # Higher = more selective
merger_threshold=50        # Higher = more selective
halt_threshold=999         # Set very high to disable
```

### API Settings
Set your Claude API key:
```bash
export CLAUDE_API_KEY=sk-ant-your-api-key-here
```

### File Paths
Update paths in scripts if your data is located elsewhere:
- Stock data: `/Users/krishna_dewan/Desktop/FINE 547 M&A/Data Files/ret_sample_usa.parquet`
- Output directory: Current folder

## 📈 Typical Workflow

1. **Detection**: Run `run_analysis.py` to find potential events
2. **Manual Review**: Check `ma_bankruptcy_events.csv` for obvious false positives
3. **Verification**: Run `claude_verifier.py` to verify events with AI
4. **Analysis**: Use verified data for research/trading strategies
5. **Model Improvement**: Run `model_diagnostics_comprehensive.py` for insights

## ⚠️ Important Notes

- The pipeline is designed for **research purposes** in M&A detection
- Detection thresholds are set conservatively to minimize false positives
- Verification adds significant accuracy but requires API costs
- Model diagnostics help improve detection quality over time
- All scripts support resumable processing for large datasets

## 🛠️ Dependencies

- pandas, numpy
- scikit-learn, xgboost
- anthropic (for Claude AI)
- matplotlib, seaborn (for diagnostics)

Install with: `pip install pandas numpy scikit-learn xgboost anthropic matplotlib seaborn`