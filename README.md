# QuantStats Analysis API with FastAPI, MCP, and Web Frontend

A comprehensive portfolio analysis platform that combines FastAPI endpoints, Model Context Protocol (MCP) integration, and an interactive web frontend. The API endpoints can be used independently by other applications, while the MCP integration enables seamless AI/LLM interactions for automated portfolio analysis.

> **Important Note**: This project uses [quantstats-lumi](https://github.com/Lumiwealth/quantstats_lumi), the Lumiwealth fork of the original QuantStats library. This fork includes important bug fixes and improvements over the original package while maintaining API compatibility. The fork is actively maintained by the Lumiwealth team and addresses several issues present in the original library.

### Statcounter Note
The application includes a Statcounter web analytics code patch in `index.html`. This tracking code is linked to my personal account, so all analytics data will be sent there. Please replace it with your own Statcounter ID or other analytics tracking code, or remove it entirely if you don't need web analytics.

### AI Co-Analyst Platform

For additional data analysis capabilities, visit our AI Co-Analyst Platform at [rex.tigzig.com](https://rex.tigzig.com). For any questions, reach out to me at amar@harolikar.com


## How It Works
.
1. **Data Collection**: Historical price data is fetched from Yahoo Finance API and processed into return series
2. **Portfolio Analysis**: 
   - Calculates key performance metrics (Sharpe ratio, Sortino ratio, Maximum Drawdown)
   - Analyzes risk metrics (Value at Risk, correlation with benchmark)
   - Processes rolling statistics and return distributions
3. **Report Generation**: Creates comprehensive HTML reports with visualizations using QuantStats
4. **Integration Layer**: Returns URLs to access interactive reports via FastAPI or MCP endpoints

## Methodology & Date Calculations

Understanding how dates and returns are processed is crucial for interpreting the analysis results. This section explains the complete data flow and why report dates may differ from input dates.

### Date Processing Flow

The date processing involves multiple stages, each potentially affecting the final date range:

#### 1. Data Download Phase
- **Input**: User-specified start/end dates (e.g., June 8, 2015)
- **Yahoo Finance**: Downloads available market data
- **Weekend Adjustment**: Non-trading days are automatically excluded
- **Market Holidays**: No data available for market closures

#### 2. Return Calculation Phase
```python
# Convert prices to percentage returns
returns = price_data.pct_change().dropna()
```
- **Effect**: Always loses the first day (can't calculate return for first price)
- **Example**: June 8 input → Returns start June 9
- **Reason**: `pct_change()` creates NaN for first row, `dropna()` removes it

#### 3. QuantStats Date Alignment Phase
```python
# QuantStats _match_dates function
loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())
```
- **Purpose**: Find first day where both series have non-zero returns
- **Effect**: May lose additional 1-2 days
- **Reason**: Ensures statistical calculations use "meaningful" trading data

### Why Dates Get Shifted: Real Examples

#### Example 1: Normal 1-Day Shift
- **Input Date**: June 8, 2015
- **After pct_change()**: June 9, 2015 ✅
- **QuantStats Start**: June 9, 2015 ✅
- **Result**: 1 day lost (expected)

#### Example 2: Zero-Return Filtering (2-Day Shift)
- **Input Date**: June 8, 2015
- **After pct_change()**: June 9, 2015
- **S&P 500 June 9**: 0.000418... (treated as zero)
- **QuantStats Start**: June 10, 2015
- **Result**: 2 days lost (zero-return filtering)

#### Example 3: Actual Zero Returns
- **Input Date**: June 7, 2020
- **After pct_change()**: June 9, 2020
- **S&P 500 June 9**: 0.0 (actual zero return)
- **QuantStats Start**: June 10, 2020
- **Result**: 2 days lost (legitimate zero filtering)

### Zero-Return Filtering Logic

QuantStats automatically skips days with zero returns to improve statistical accuracy:

```python
# What QuantStats considers "zero"
returns.ne(0)  # Returns False for exact 0.0 values
benchmark.ne(0)  # Also checks benchmark for zeros

# Common causes of zero returns:
- Market holidays (no trading)
- Exact price matches (rare but possible)
- Data processing artifacts
- Weekend boundary effects
```

### Return Calculation Methods

#### Daily Returns
```python
daily_return = (price_today - price_yesterday) / price_yesterday
```

#### Annualized Returns
QuantStats uses different conventions for annualization:

- **Trading Days**: 252 days per year (most common)
- **Calendar Days**: 365 days per year
- **Business Days**: 260 days per year (some calculations)

```python
# Annualized return calculation
annualized_return = (1 + daily_return_mean) ** periods_per_year - 1
```

#### Compound vs Simple Returns

**Total Return Calculation**:
- QuantStats uses **compound returns** (not simple addition)
- Formula: `(1 + r1) × (1 + r2) × ... × (1 + rn) - 1`
- More accurate for long periods and volatile returns

```python
# Compound total return
total_return = (1 + returns).prod() - 1

# NOT simple addition:
# wrong_total = returns.sum()  # This would be incorrect
```

### Key Metrics Calculations

#### Sharpe Ratio
```python
sharpe_ratio = (mean_return - risk_free_rate) / return_volatility
```
- Uses annualized figures
- Risk-free rate adjusted to same frequency as returns

#### Maximum Drawdown
```python
# Peak-to-trough calculation
cumulative_returns = (1 + returns).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min()
```

#### Value at Risk (VaR)
- Uses empirical percentile method (default 5%)
- Based on actual return distribution
- No assumption of normal distribution

### Best Practices for Date Interpretation

1. **Expect 1-2 Day Shifts**: This is normal behavior, not an error
2. **Check Raw Data**: CSV exports show actual downloaded data
3. **Understand Zero Filtering**: Some days may be legitimately skipped
4. **Use Business Days**: Avoid weekend start/end dates
5. **Account for Holidays**: Major market holidays will create gaps

### Debugging Date Issues

If you see unexpected date shifts:

1. **Check CSV Files**: Examine raw downloaded data
2. **Look for Zero Returns**: Identify days with 0.0% returns
3. **Verify Market Hours**: Ensure markets were open
4. **Consider Data Quality**: Yahoo Finance occasionally has gaps

### Technical Notes

- **Timezone Handling**: All dates processed in market local time
- **Index Alignment**: Pandas automatically aligns mismatched date indices
- **Missing Data**: `dropna()` calls remove incomplete rows
- **Frequency Conversion**: Daily data is standard; other frequencies auto-converted

This methodology ensures robust, statistically sound analysis while explaining why input dates may differ from report header dates.

## Endpoints

The application exposes three types of endpoints:

- **Frontend**: `/` - Web-based interface for interactive analysis
- **API**: `/analyze` - FastAPI endpoint for direct programmatic access
- **MCP**: `/mcp` - Model Context Protocol endpoint for AI/LLM integration

## FastAPI Implementation

The FastAPI server is enhanced with MCP capabilities using the `fastapi-mcp` package. Key implementation features:

1. **MCP Integration**:
```python
mcp = FastApiMCP(
    app,
    name="QuantStats Analysis MCP API",
    description="MCP server for portfolio analysis endpoints",
    base_url=os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000"),
    include_operations=["create_portfolio_analysis"],
    describe_all_responses=True,
    describe_full_response_schema=True,
    http_client=httpx.AsyncClient(timeout=180.0)
)
mcp.mount()
```

2. **Enhanced Documentation**:
- Detailed docstrings for each endpoint
- Parameter descriptions using FastAPI's Query
- Clear response models with examples
- Operation IDs for MCP tool identification

3. **Structured Models**:
- Pydantic models for request/response validation
- Comprehensive parameter descriptions
- Type hints and validation rules

## Configuration Required

Create a `.env` file in the root directory with:

```env
# For Local Development. Set to 0 for production deployment
IS_LOCAL_DEVELOPMENT=1

# Base URL where this is deployed. Used to construct full paths for reports
BASE_URL_FOR_REPORTS=https://your-domain.com/
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up environment variables
4. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Usage

The API can be accessed in three ways:

1. **Web Interface**: Visit `/` for the interactive web frontend
2. **Direct API**: Make HTTP GET requests to `/analyze` with required parameters
3. **MCP/LLM**: Connect to `/mcp` using any MCP-compatible client (e.g., Cursor, Claude)

## Example Request

```
GET /analyze?symbols=AAPL&benchmark=^GSPC&start_date=2023-01-01&end_date=2024-01-01&risk_free_rate=5.0
```

## Response

The API returns a URL to access the generated HTML report:

```json
{
    "html_url": "https://your-domain.com/static/reports/AAPL_vs_^GSPC_20240315_123456.html"
}
```

## Dependencies

- FastAPI
- fastapi-mcp
- quantstats-lumi (Lumiwealth's fork of QuantStats)
- yfinance
- pandas & numpy
- matplotlib
- Jinja2 templates
- httpx
- Other requirements listed in requirements.txt

## License

MIT License
