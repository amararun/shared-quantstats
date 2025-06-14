from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import quantstats_lumi as qs
import yfinance as yf
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import matplotlib
matplotlib.use('Agg')
from fastapi_mcp import FastApiMCP
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables for URL handling
IS_LOCAL_DEVELOPMENT = os.getenv('IS_LOCAL_DEVELOPMENT', '0') == '1'
BASE_URL_FOR_REPORTS = os.getenv('BASE_URL_FOR_REPORTS', 'https://quantstat-nextjs.vercel.app/')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Create a separate debug logger for detailed diagnostics
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('debug.log')
debug_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
debug_logger.addHandler(debug_handler)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Monkey patch numpy.product
if not hasattr(np, 'product'):
    np.product = np.prod

def cleanup_old_reports():
    """Clean up report files older than 3 days"""
    try:
        # Calculate cutoff time (3 days ago)
        cutoff_time = datetime.now() - timedelta(days=3)
        cutoff_timestamp = cutoff_time.timestamp()
        
        logger.info("Starting cleanup of old report files...")
        logger.info(f"Deleting files older than: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        deleted_count = 0
        total_size_freed = 0
        
        # Check if reports directory exists
        if os.path.exists(REPORTS_DIR):
            # Get all files in reports directory
            for filename in os.listdir(REPORTS_DIR):
                file_path = os.path.join(REPORTS_DIR, filename)
                
                # Skip if it's not a file
                if not os.path.isfile(file_path):
                    continue
                
                # Get file modification time
                file_mtime = os.path.getmtime(file_path)
                
                # Delete if older than 3 days
                if file_mtime < cutoff_timestamp:
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_count += 1
                        total_size_freed += file_size
                        logger.info(f"Deleted old report: {filename}")
                    except Exception as e:
                        logger.error(f"Error deleting file {filename}: {str(e)}")
        
        if deleted_count > 0:
            size_mb = total_size_freed / (1024 * 1024)
            logger.info(f"Cleanup completed: {deleted_count} files deleted, {size_mb:.2f} MB freed")
        else:
            logger.info("Cleanup completed: No old files found to delete")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        # Don't raise the error - cleanup failure shouldn't prevent server startup

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("=" * 40)
    logger.info("FastAPI server with MCP integration is starting!")
    
    # Run cleanup of old report files
    cleanup_old_reports()
    
    logger.info("MCP endpoint is available at: http://localhost:8007/mcp")
    logger.info("Using custom httpx client with 3-minute (180 second) timeout")
    
    # Log all available routes and their operation IDs
    logger.info("Available routes and operation IDs in FastAPI app:")
    fastapi_operations = []
    for route in app.routes:
        if hasattr(route, "operation_id"):
            logger.info(f"Route: {route.path}, Operation ID: {route.operation_id}")
            fastapi_operations.append(route.operation_id)
    
    yield  # This is where the FastAPI app runs
    
    # Shutdown code
    logger.info("=" * 40)
    logger.info("FastAPI server is shutting down")
    logger.info("=" * 40)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="QuantStats Analysis API",
    description="API for generating portfolio analysis reports using QuantStats and yfinance data",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Ensure reports directory exists
REPORTS_DIR = os.path.join('static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Ensure data directory exists for CSV exports
DATA_DIR = os.path.join('static', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Extend pandas
qs.extend_pandas()

# Pydantic models for API
class AnalysisRequest(BaseModel):
    """Request model for portfolio analysis. Requires exactly 2 Yahoo Finance symbols: one for the asset to analyze and one for benchmark comparison."""
    stock_code: str = Field(
        description="MANDATORY: Primary stock symbol to analyze (e.g., 'AAPL', 'GOOG', '^NSEI'). Must be a valid Yahoo Finance ticker symbol. This is the main asset being analyzed.",
        example="AAPL"
    )
    benchmark_code: str = Field(
        default="^GSPC",
        description="MANDATORY: Benchmark symbol for comparison (e.g., '^GSPC' for S&P 500, '^IXIC' for NASDAQ). Must be a valid Yahoo Finance ticker symbol. While it has a default value of '^GSPC', a benchmark is always required for analysis - you cannot analyze without a benchmark.",
        example="^GSPC"
    )
    start_date: str = Field(
        description="Start date for analysis in YYYY-MM-DD format. Should be at least 6 months before end_date for meaningful analysis.",
        example="2023-01-01"
    )
    end_date: str = Field(
        description="End date for analysis in YYYY-MM-DD format. Must be after start_date and not in the future.",
        example="2024-01-01"
    )
    risk_free_rate: float = Field(
        default=5.0,
        description="Risk-free rate as percentage (e.g., 5.0 for 5%). Used in Sharpe ratio calculations.",
        example=5.0,
        ge=0,
        le=100
    )

class AnalysisResponse(BaseModel):
    """Response model for portfolio analysis. Contains URL to the comprehensive comparative analysis report generated from exactly 2 Yahoo Finance symbols."""
    html_url: str = Field(
        description="Direct URL to access the comprehensive HTML report with visualizations, performance metrics, and risk analysis comparing the primary symbol against the benchmark",
        example="https://quantstat-nextjs.vercel.app/static/reports/AAPL_vs_^GSPC_20240315_123456.html"
    )

def process_stock_data(data, symbol):
    """Process stock data to get returns series with proper format"""
    try:
        debug_logger.debug(f"\nProcessing {symbol} data:")
        debug_logger.debug(f"Input data shape: {data.shape}")
        
        # Get the Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[('Close', symbol)]
        else:
            prices = data['Close']
        
        debug_logger.debug(f"Price series shape: {prices.shape}")
        debug_logger.debug(f"Price range: {prices.min():.2f} to {prices.max():.2f}")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Convert to Series if it's not already
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
            
        # Name the series
        returns.name = symbol
        
        debug_logger.debug(f"Returns series shape: {returns.shape}")
        debug_logger.debug(f"Returns range: {returns.min():.4f} to {returns.max():.4f}")
        
        return returns
        
    except Exception as e:
        debug_logger.error(f"Error processing {symbol} data: {str(e)}", exc_info=True)
        raise

def generate_quantstats_report(stock_returns, benchmark_returns, output_file, title, risk_free_rate=5.0):
    """Generate QuantStats report with error handling"""
    try:
        debug_logger.debug("\nGenerating QuantStats report:")
        debug_logger.debug(f"Stock returns: {len(stock_returns)} periods")
        debug_logger.debug(f"Benchmark returns: {len(benchmark_returns)} periods")
        debug_logger.debug(f"Risk-free rate: {risk_free_rate:.2f}%")
        
        # Ensure the returns are properly aligned
        start_date = max(stock_returns.index.min(), benchmark_returns.index.min())
        end_date = min(stock_returns.index.max(), benchmark_returns.index.max())
        
        stock_returns = stock_returns[start_date:end_date]
        benchmark_returns = benchmark_returns[start_date:end_date]
        
        debug_logger.debug(f"Aligned period: {start_date} to {end_date}")
        debug_logger.debug(f"Final series lengths - Stock: {len(stock_returns)}, Benchmark: {len(benchmark_returns)}")
        
        # NEW: Add detailed debugging for the first few dates and zero returns
        debug_logger.debug(f"Stock returns first 5 dates: {stock_returns.head().index.tolist()}")
        debug_logger.debug(f"Stock returns first 5 values: {stock_returns.head().tolist()}")
        debug_logger.debug(f"Benchmark returns first 5 dates: {benchmark_returns.head().index.tolist()}")
        debug_logger.debug(f"Benchmark returns first 5 values: {benchmark_returns.head().tolist()}")
        
        # Check for zero returns that might trigger QuantStats _match_dates filtering
        stock_zeros = stock_returns[stock_returns == 0.0]
        benchmark_zeros = benchmark_returns[benchmark_returns == 0.0]
        debug_logger.debug(f"Stock zero returns count: {len(stock_zeros)}")
        debug_logger.debug(f"Benchmark zero returns count: {len(benchmark_zeros)}")
        if len(stock_zeros) > 0:
            debug_logger.debug(f"Stock zero dates (first 5): {stock_zeros.head().index.tolist()}")
        if len(benchmark_zeros) > 0:
            debug_logger.debug(f"Benchmark zero dates (first 5): {benchmark_zeros.head().index.tolist()}")
        
        # Check what QuantStats _match_dates logic would do
        # Simulating: max(stock_returns.ne(0).idxmax(), benchmark_returns.ne(0).idxmax())
        stock_first_nonzero = stock_returns[stock_returns != 0.0].index.min() if len(stock_returns[stock_returns != 0.0]) > 0 else None
        benchmark_first_nonzero = benchmark_returns[benchmark_returns != 0.0].index.min() if len(benchmark_returns[benchmark_returns != 0.0]) > 0 else None
        debug_logger.debug(f"Stock first non-zero date: {stock_first_nonzero}")
        debug_logger.debug(f"Benchmark first non-zero date: {benchmark_first_nonzero}")
        if stock_first_nonzero and benchmark_first_nonzero:
            quantstats_start_date = max(stock_first_nonzero, benchmark_first_nonzero)
            debug_logger.debug(f"QuantStats _match_dates would start from: {quantstats_start_date}")
            debug_logger.debug(f"Days lost to zero-return filtering: {(quantstats_start_date - start_date).days}")
        
        # Convert annual rate to decimal
        rf_rate = risk_free_rate / 100.0
        
        # NEW: Debug exactly what we're passing to QuantStats
        debug_logger.debug(f"=== BEFORE QuantStats Processing ===")
        debug_logger.debug(f"Stock returns passed to QuantStats - Start: {stock_returns.index.min()}, End: {stock_returns.index.max()}")
        debug_logger.debug(f"Benchmark returns passed to QuantStats - Start: {benchmark_returns.index.min()}, End: {benchmark_returns.index.max()}")
        debug_logger.debug(f"Stock returns length: {len(stock_returns)}")
        debug_logger.debug(f"Benchmark returns length: {len(benchmark_returns)}")
        
        # NEW: Patch QuantStats _match_dates to see what it actually does
        original_match_dates = qs.reports._match_dates
        def debug_match_dates(returns, benchmark):
            debug_logger.debug(f"=== _match_dates CALLED ===")
            debug_logger.debug(f"Input returns start: {returns.index.min()}")
            debug_logger.debug(f"Input benchmark start: {benchmark.index.min()}")
            
            # NEW: Debug the exact .ne(0).idxmax() logic
            fix_instance = lambda x: x[x.columns[0]] if isinstance(x, pd.DataFrame) else x
            returns_fixed = fix_instance(returns)
            benchmark_fixed = fix_instance(benchmark)
            
            returns_ne_zero = returns_fixed.ne(0)
            benchmark_ne_zero = benchmark_fixed.ne(0)
            
            returns_first_nonzero_idx = returns_ne_zero.idxmax()
            benchmark_first_nonzero_idx = benchmark_ne_zero.idxmax()
            
            debug_logger.debug(f"Returns first few ne(0): {returns_ne_zero.head().tolist()}")
            debug_logger.debug(f"Benchmark first few ne(0): {benchmark_ne_zero.head().tolist()}")
            
            # NEW: Show exact values being compared
            debug_logger.debug(f"Returns first few values: {returns_fixed.head().tolist()}")
            debug_logger.debug(f"Benchmark first few values: {benchmark_fixed.head().tolist()}")
            debug_logger.debug(f"Benchmark == 0 first few: {(benchmark_fixed == 0).head().tolist()}")
            debug_logger.debug(f"Benchmark != 0 first few: {(benchmark_fixed != 0).head().tolist()}")
            
            debug_logger.debug(f"Returns .ne(0).idxmax(): {returns_first_nonzero_idx}")
            debug_logger.debug(f"Benchmark .ne(0).idxmax(): {benchmark_first_nonzero_idx}")
            
            loc = max(returns_first_nonzero_idx, benchmark_first_nonzero_idx)
            debug_logger.debug(f"max() result (loc): {loc}")
            
            # Call original function
            result_returns, result_benchmark = original_match_dates(returns, benchmark)
            
            debug_logger.debug(f"Output returns start: {result_returns.index.min()}")
            debug_logger.debug(f"Output benchmark start: {result_benchmark.index.min()}")
            debug_logger.debug(f"_match_dates dropped {(result_returns.index.min() - returns.index.min()).days} days")
            
            return result_returns, result_benchmark
        
        # Temporarily replace the function
        qs.reports._match_dates = debug_match_dates
        
        # Generate the report
        qs.reports.html(
            stock_returns, 
            benchmark_returns,
            rf=rf_rate,
            output=output_file,
            title=title
        )
        
        # Restore original function
        qs.reports._match_dates = original_match_dates
        
        # NEW: Debug what QuantStats actually used (check the generated HTML)
        debug_logger.debug(f"=== AFTER QuantStats Processing ===")
        try:
            # Read the generated HTML to extract the actual date range used
            with open(output_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
                # Look for the date range pattern in the HTML
                import re
                date_pattern = r'(\d{1,2}\s+\w{3},\s+\d{4})\s*-\s*(\d{1,2}\s+\w{3},\s+\d{4})'
                match = re.search(date_pattern, html_content)
                if match:
                    actual_start = match.group(1)
                    actual_end = match.group(2)
                    debug_logger.debug(f"QuantStats HTML shows date range: {actual_start} - {actual_end}")
                    debug_logger.debug(f"Expected start: 9 Jun, 2015 | Actual start: {actual_start}")
                else:
                    debug_logger.debug("Could not extract date range from generated HTML")
        except Exception as e:
            debug_logger.debug(f"Error reading HTML file: {str(e)}")
        
        # Add custom footer to the generated HTML
        add_custom_footer_to_html(output_file)
        
        return True
    except Exception as e:
        debug_logger.error("Error generating report:", exc_info=True)
        raise

def add_custom_footer_to_html(html_file_path):
    """Add custom footer to the generated QuantStats HTML report"""
    try:
        # Read the generated HTML file
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Define the custom header with REX branding and icon options
        custom_header = '''
        <style>
            body { margin: 0 !important; padding-top: 0 !important; }
        </style>
        <header style="
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
            color: white; 
            padding: 6px 0; 
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        ">
            <div style="max-width: 1200px; margin: 0 auto; padding: 0 16px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 12px; font-size: 16px; font-weight: 600;">
                    <!-- Icons first -->
                    <div style="display: flex; align-items: center; gap: 2px; font-size: 18px;">
                        <span style="font-family: 'Times New Roman', serif; font-style: italic; font-weight: 500;">f(x)</span>
                        <span style="font-family: monospace; font-weight: 500; letter-spacing: -1px;">&lt;/&gt;</span>
                    </div>
                    <!-- REX text with full hyperlink -->
                    <a href="https://rex.tigzig.com" target="_blank" rel="noopener noreferrer" style="color: #fbbf24; text-decoration: none; font-weight: 700;">
                        REX <span style="color: #ffffff;">: AI Co-Analyst</span>
                    </a>
                    <span style="color: #ffffff;">- Portfolio Analytics</span>
                </div>
            </div>
        </header>
        '''
        
        # Define the disclaimer note about QuantStats-LumiWealth Version
        disclaimer_note = '''
        <div style="
            background: rgba(255,248,220,0.8); 
            border: 1px solid #fbbf24; 
            border-radius: 6px;
            padding: 12px; 
            margin: 20px auto; 
            max-width: 1200px;
            font-size: 13px; 
            color: #92400e;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.5;
        ">
            Fine Print: All metrics in this report are generated using the open-source <a href="https://github.com/Lumiwealth/quantstats_lumi" target="_blank" rel="noopener noreferrer" style="color: #1d4ed8; text-decoration: none; font-weight: 500;">QuantStats-LumiWealth Version</a>. While widely used for performance and risk analytics, some calculations may rely on assumptions (e.g., trading days, compounding methods) or be subject to version-specific behavior (e.g., variations between QuantStats and FFN reports). Key metrics such as total return and CAGR have been manually reviewed for consistency, but users are encouraged to refer to the <a href="https://github.com/Lumiwealth/quantstats_lumi" target="_blank" rel="noopener noreferrer" style="color: #1d4ed8; text-decoration: none; font-weight: 500;">official documentation</a> for full methodology. Results should be interpreted in light of the intended use case - acceptable variation may differ depending on the analytical objective or decision context. Always validate outputs.
        </div>
        '''
        
        # Define the custom footer HTML with email added
        custom_footer = '''
        <footer style="
            background: rgba(255,255,255,0.5); 
            border-top: 1px solid #e0e7ff; 
            padding: 8px 0; 
            margin-top: 20px; 
            font-size: 12px; 
            color: #1e1b4b;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="max-width: 1200px; margin: 0 auto; padding: 0 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 4px;">
                    <div style="font-size: 12px; color: rgba(30, 27, 75, 0.7);">
                        Amar Harolikar <span style="margin: 0 6px; color: #c7d2fe;">•</span> 
                        Specialist - Decision Sciences & Applied Generative AI <span style="margin: 0 6px; color: #c7d2fe;">•</span>
                        <a href="mailto:amar@harolikar.com" style="color: #4338ca; text-decoration: none;">amar@harolikar.com</a>
                    </div>
                    <div style="display: flex; align-items: center; gap: 16px; font-size: 12px;">
                        <a href="https://www.linkedin.com/in/amarharolikar" target="_blank" rel="noopener noreferrer"
                           style="color: #4338ca; text-decoration: none;">
                            LinkedIn
                        </a>
                        <a href="https://github.com/amararun" target="_blank" rel="noopener noreferrer"
                           style="color: #4338ca; text-decoration: none;">
                            GitHub
                        </a>
                        <a href="https://rex.tigzig.com" target="_blank" rel="noopener noreferrer"
                           style="color: #4338ca; text-decoration: none;">
                            rex.tigzig.com
                        </a>
                        <a href="https://tigzig.com" target="_blank" rel="noopener noreferrer"
                           style="color: #4338ca; text-decoration: none;">
                            tigzig.com
                        </a>
                    </div>
                </div>
            </div>
        </footer>
        '''
        
        # Insert the header at the beginning after <body> tag
        if '<body' in html_content:
            # Find the end of the <body> tag (could have attributes)
            body_start = html_content.find('<body')
            body_tag_end = html_content.find('>', body_start) + 1
            html_content = html_content[:body_tag_end] + custom_header + html_content[body_tag_end:]
        else:
            # If no body tag, add at the beginning
            html_content = custom_header + html_content
        
        # Insert the disclaimer note and footer before the closing </body> tag
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', disclaimer_note + custom_footer + '\n</body>')
        else:
            # If no </body> tag found, append to the end
            html_content += disclaimer_note + custom_footer
        
        # Write the modified HTML back to the file
        with open(html_file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
            
        debug_logger.debug(f"Successfully added custom header, disclaimer note and footer with email to {html_file_path}")
        
    except Exception as e:
        debug_logger.error(f"Error adding header and footer to HTML file: {str(e)}", exc_info=True)
        # Don't raise the error - footer addition is not critical
        pass

@app.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request,
    error: str = None,
    success: str = None,
    report_path: str = None,
    symbols: str = None
):
    """Serve the web interface"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": error,
            "success": success,
            "report_generated": bool(report_path),
            "report_path": report_path,
            "symbols": symbols
        }
    )

def construct_report_url(filename: str) -> str:
    """Construct the appropriate URL for report files based on environment."""
    if IS_LOCAL_DEVELOPMENT:
        return f"/static/reports/{filename}"
    else:
        # Ensure BASE_URL_FOR_REPORTS ends with a slash
        base_url = BASE_URL_FOR_REPORTS.rstrip('/') + '/'
        return f"{base_url}static/reports/{filename}"

def sanitize_for_filename(symbol: str) -> str:
    """Convert symbol to a filename-safe format by replacing special characters."""
    # Common substitutions for stock indices
    common_replacements = {
        "^GSPC": "SP500",
        "^IXIC": "NASDAQ",
        "^DJI": "DOW",
        "^RUT": "RUSSELL2000"
    }
    
    # If it's a common index, use its friendly name
    if symbol in common_replacements:
        return common_replacements[symbol]
    
    # Otherwise, remove or replace special characters
    return symbol.replace("^", "").replace(".", "_").replace("/", "_").replace("\\", "_")

@app.get("/analyze", operation_id="create_portfolio_analysis", response_model=AnalysisResponse)
def analyze(
    symbols: str = Query(..., description="MANDATORY: Primary stock symbol to analyze (e.g., 'AAPL', 'MSFT', 'GOOGL'). Must be a valid Yahoo Finance ticker symbol. This is the main asset being analyzed. Note: Despite the plural name, only one symbol should be provided."),
    benchmark: str = Query(default="^GSPC", description="MANDATORY: Benchmark symbol for comparison (e.g., '^GSPC' for S&P 500, '^IXIC' for NASDAQ, '^DJI' for Dow Jones). Must be a valid Yahoo Finance ticker symbol. While this defaults to '^GSPC', a benchmark is ALWAYS used in analysis - this tool requires exactly 2 symbols for comparative analysis. If you don't specify a benchmark, '^GSPC' (S&P 500) will be used automatically."),
    start_date: str = Query(..., description="MANDATORY: Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="MANDATORY: End date in YYYY-MM-DD format"),
    risk_free_rate: float = Query(default=5.0, description="Risk-free rate as percentage (default: 5.0)"),
    request: Request = None
):
    """
    Generate a comprehensive portfolio analysis report using QuantStats.
    
    IMPORTANT: This tool requires EXACTLY 2 Yahoo Finance symbols:
    1. Primary symbol - the stock/asset you want to analyze
    2. Benchmark symbol - the comparison benchmark (e.g., ^GSPC for S&P 500)
    
    Both symbols are MANDATORY. You cannot perform analysis with just one symbol.
    
    This endpoint performs detailed portfolio analysis including:
    - Performance metrics (Sharpe ratio, Sortino ratio, CAGR, etc.)
    - Risk metrics (Value at Risk, Maximum Drawdown, etc.)
    - Return analysis and distribution
    - Rolling statistics and correlations
    - Comparative analysis against benchmark
    
    The analysis is returned as:
    - Interactive HTML report with visualizations
    - Direct URL to view the comprehensive report
    
    Parameters:
    - symbols: MANDATORY - Primary stock symbol to analyze (e.g., 'AAPL', 'MSFT', 'GOOGL'). Despite the plural name, provide only one symbol.
    - benchmark: MANDATORY - Benchmark symbol for comparison (e.g., '^GSPC', '^IXIC', '^DJI'). Defaults to '^GSPC' if not specified, but a benchmark is always used.
    - start_date: MANDATORY - Analysis start date (YYYY-MM-DD format)
    - end_date: MANDATORY - Analysis end date (YYYY-MM-DD format)
    - risk_free_rate: Risk-free rate percentage for calculations (default: 5.0)
    
    Returns:
    - AnalysisResponse with html_url to access the generated HTML report
    
    Note: The analysis may take up to 2-3 minutes depending on the date range and data availability.
    Both symbols must be valid Yahoo Finance ticker symbols.
    """
    try:
        # Validate and process the single symbol
        stock_code = symbols.strip().upper()
        benchmark_code = benchmark.strip().upper()
        
        if not stock_code:
            raise HTTPException(status_code=400, detail="Primary stock symbol is required")
        if not benchmark_code:
            raise HTTPException(status_code=400, detail="Benchmark symbol is required")
            
        logger.info(f"Processing request for stock: {stock_code} against benchmark: {benchmark_code}")
        debug_logger.debug(f"\nNew analysis request for {stock_code} vs {benchmark_code}")
        debug_logger.debug(f"Date range: {start_date} to {end_date}")
        debug_logger.debug(f"Risk-free rate: {risk_free_rate}%")
        
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_stock_code = sanitize_for_filename(stock_code)
        safe_benchmark_code = sanitize_for_filename(benchmark_code)
        report_filename = f'{safe_stock_code}_vs_{safe_benchmark_code}_{timestamp}.html'
        report_path = os.path.join(REPORTS_DIR, report_filename)
        
        # Parse and validate dates
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Add one day to end_date to include the last day in the analysis
            end = end + timedelta(days=1)
            
            if start >= end:
                raise HTTPException(status_code=400, detail="Start date must be before end date")
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
            
        debug_logger.debug(f"Downloading data for period: {start} to {end}")
        
        # Download stock data
        logger.info(f"Downloading data for {stock_code}")
        stock_data = yf.download(stock_code, start=start, end=end, progress=False)
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock {stock_code}")
        
        # Export raw stock data to CSV for validation
        stock_csv_filename = f'{safe_stock_code}_raw_data_{timestamp}.csv'
        stock_csv_path = os.path.join(DATA_DIR, stock_csv_filename)
        stock_data.to_csv(stock_csv_path)
        logger.info(f"Raw stock data exported to: {stock_csv_filename}")
        
        # Process stock returns
        stock_returns = process_stock_data(stock_data, stock_code)
        
        # Download benchmark data
        logger.info(f"Downloading data for benchmark {benchmark_code}")
        benchmark_data = yf.download(benchmark_code, start=start, end=end, progress=False)
        
        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {benchmark_code}")
        
        # Export raw benchmark data to CSV for validation
        benchmark_csv_filename = f'{safe_benchmark_code}_raw_data_{timestamp}.csv'
        benchmark_csv_path = os.path.join(DATA_DIR, benchmark_csv_filename)
        benchmark_data.to_csv(benchmark_csv_path)
        logger.info(f"Raw benchmark data exported to: {benchmark_csv_filename}")
        
        # Process benchmark returns
        benchmark_returns = process_stock_data(benchmark_data, benchmark_code)
        
        # Generate the report
        generate_quantstats_report(
            stock_returns,
            benchmark_returns,
            report_path,
            f'{stock_code} vs {benchmark_code} Analysis Report',
            risk_free_rate=risk_free_rate
        )
        
        logger.info(f"Successfully generated report for {stock_code} vs {benchmark_code}")
        
        # After successful report generation, construct the response
        report_url = construct_report_url(report_filename)
        
        # Return the standardized response with only html_url
        return AnalysisResponse(
            html_url=report_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        debug_logger.error(f"Detailed error for {symbols}:", exc_info=True)
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_api(analysis: AnalysisRequest):
    """API endpoint for analysis"""
    try:
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_stock_code = sanitize_for_filename(analysis.stock_code)
        safe_benchmark_code = sanitize_for_filename(analysis.benchmark_code)
        report_filename = f'{safe_stock_code}_vs_{safe_benchmark_code}_{timestamp}.html'
        report_path = os.path.join(REPORTS_DIR, report_filename)

        # Convert dates
        start = datetime.strptime(analysis.start_date, '%Y-%m-%d')
        end = datetime.strptime(analysis.end_date, '%Y-%m-%d') + timedelta(days=1)

        if start >= end:
            raise HTTPException(status_code=400, detail="Start date must be before end date")

        # Download data
        stock_data = yf.download(analysis.stock_code, start=start, end=end, progress=False)
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock {analysis.stock_code}")

        benchmark_data = yf.download(analysis.benchmark_code, start=start, end=end, progress=False)
        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {analysis.benchmark_code}")

        # Process data
        stock_returns = process_stock_data(stock_data, analysis.stock_code)
        benchmark_returns = process_stock_data(benchmark_data, analysis.benchmark_code)

        # Generate report
        generate_quantstats_report(
            stock_returns,
            benchmark_returns,
            report_path,
            f'{analysis.stock_code} vs {analysis.benchmark_code} Analysis Report',
            risk_free_rate=analysis.risk_free_rate
        )

        # Return success response with report URL
        return {
            "success": True,
            "message": f"Successfully generated report for {analysis.stock_code} vs {analysis.benchmark_code}",
            "report_url": f"/static/reports/{report_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{filename:path}")
async def serve_report(filename: str):
    """Serve report files"""
    return FileResponse(f"static/reports/{filename}")

# Create MCP server and include relevant endpoints
base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8007")
if not base_url.startswith(("http://", "https://")):
    base_url = f"http://{base_url}"

mcp = FastApiMCP(
    app,
    name="QuantStats Analysis MCP API",
    description="MCP server for portfolio analysis endpoints. Note: Operations may take up to 3 minutes due to data fetching and analysis requirements.",
    include_operations=[
        "create_portfolio_analysis"
    ],
    # Better schema descriptions
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Inject custom httpx client with proper configuration
    http_client=httpx.AsyncClient(
        timeout=180.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url=base_url
    )
)

# Mount the MCP server
mcp.mount()

# Log MCP operations
logger.info("Operations included in MCP server:")
for op in mcp._include_operations:
    logger.info(f"Operation '{op}' included in MCP")

logger.info("MCP server exposing portfolio analysis endpoints")
logger.info("=" * 40)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
