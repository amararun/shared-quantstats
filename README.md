# QuantStats Analysis API with FastAPI, MCP, and Web Frontend

A comprehensive portfolio analysis platform that combines FastAPI endpoints, Model Context Protocol (MCP) integration, and an interactive web frontend. The API endpoints can be used independently by other applications, while the MCP integration enables seamless AI/LLM interactions for automated portfolio analysis.

> **Important Note**: This project uses [quantstats-lumi](https://github.com/Lumiwealth/quantstats_lumi), the Lumiwealth fork of the original QuantStats library. This fork includes important bug fixes and improvements over the original package while maintaining API compatibility. The fork is actively maintained by the Lumiwealth team and addresses several issues present in the original library.

### Statcounter Note
The application includes a Statcounter web analytics code patch in `index.html`. This tracking code is linked to my personal account, so all analytics data will be sent there. Please replace it with your own Statcounter ID or other analytics tracking code, or remove it entirely if you don't need web analytics.

## How It Works
.
1. **Data Collection**: Historical price data is fetched from Yahoo Finance API and processed into return series
2. **Portfolio Analysis**: 
   - Calculates key performance metrics (Sharpe ratio, Sortino ratio, Maximum Drawdown)
   - Analyzes risk metrics (Value at Risk, correlation with benchmark)
   - Processes rolling statistics and return distributions
3. **Report Generation**: Creates comprehensive HTML reports with visualizations using QuantStats
4. **Integration Layer**: Returns URLs to access interactive reports via FastAPI or MCP endpoints

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
