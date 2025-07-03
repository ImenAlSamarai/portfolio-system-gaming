# Multi-Agent Portfolio Optimization System

## Overview
Quantitative trading system implementing multi-agent architecture for systematic portfolio management and risk assessment. Developed to explore the application of AI-driven decision making in institutional-style portfolio optimization.

## System Architecture

### Multi-Agent Framework
- **Evolving Agent Modules (EAMs)**: Independent agents implementing Conservative, Growth, and Balanced investment strategies
- **Strategic Agent Module (SAM)**: Central orchestrator managing portfolio allocation and risk controls
- **Asynchronous Processing**: Concurrent execution for market data retrieval and agent decision-making

### Risk Management Implementation
- **Statistical Validation**: Sharpe ratio, Sortino ratio, and Calmar ratio calculations
- **Value-at-Risk (VaR)**: 95% confidence interval risk assessment with Conditional VaR
- **Drawdown Analysis**: Maximum drawdown monitoring with position sizing controls
- **Performance Attribution**: Agent-level return decomposition and analysis

### Data Pipeline
- **Market Data Integration**: Real-time price feeds via yFinance API
- **Caching Layer**: Pickle-based serialization for performance optimization
- **Logging Framework**: Comprehensive execution tracking and audit trail
- **Output Generation**: Automated reporting with statistical summaries

## Technical Implementation

```python
# Core Dependencies
numpy>=1.21.0           # Numerical computations
pandas>=1.3.0           # Data manipulation and analysis
yfinance>=0.2.0         # Market data retrieval
anthropic>=0.3.0        # AI agent decision-making
scikit-learn>=1.0.0     # Statistical preprocessing
matplotlib>=3.5.0       # Visualization framework
scipy>=1.7.0            # Statistical analysis
```

## Backtesting Results

**Period**: January 2022 - January 2024 (511 trading days)  
**Initial Capital**: $100,000  
**Final Value**: $99,736.57  
**Total Return**: -0.26%  

**Risk Metrics**:
- Maximum Drawdown: -1.21%
- Annual Volatility: 0.70%
- Sharpe Ratio: -5.936
- 95% VaR (Daily): -0.06%

**Execution Statistics**:
- Total Trades: 303
- Average Trade Size: $3,557.82
- Win Rate: 28.4%

## File Structure
```
portfolio_system/
├── portfolio_system.py              # Main execution engine
├── requirements.txt                  # Dependencies specification
├── performance_report.txt            # Comprehensive analysis output
├── portfolio_backtest_results.csv    # Historical performance data
├── execution_log.json               # Transaction log
└── data_cache/                      # Cached market data
    └── market_data_SPY_max.pkl      # Serialized price data
```

## Research Objectives

This implementation serves as a framework for investigating:
- Multi-agent coordination in financial decision-making
- Risk-adjusted performance measurement methodologies
- Systematic approach to portfolio construction and rebalancing
- Statistical validation of algorithmic trading strategies

## Usage
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_api_key"
python portfolio_system.py
```

## Notes
The system demonstrates conservative risk management with limited alpha generation. Results indicate successful implementation of risk controls while highlighting the challenges of consistent outperformance in systematic strategies.