# FastAPI-MCP Upgrade Guide

## Issue
When using FastAPI-MCP 0.3.4, you might encounter this error:
```
httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol
```

## Solution

### 1. Upgrade FastAPI-MCP
Update your `requirements.txt` to use the latest version:

```diff
- fastapi-mcp==0.2.0
+ fastapi-mcp==0.3.4
```

Run:
```bash
pip install -r requirements.txt
```

### 2. Update FastApiMCP Initialization
The key change is to properly configure the base URL in the httpx client. Here's the complete working example:

```python
# Create MCP server and include relevant endpoints
base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
if not base_url.startswith(("http://", "https://")):
    base_url = f"http://{base_url}"

mcp = FastApiMCP(
    app,
    name="Your API Name",
    description="Your API Description",
    include_operations=[
        "your_operation"
    ],
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
```

### 3. Important Notes
- Always ensure your base_url includes the protocol (http:// or https://)
- Configure appropriate timeout values based on your operation requirements
- Use connection limits for better stability in production

## Verification
After making these changes:

1. Restart your FastAPI application
2. Check the logs for successful MCP server initialization:
```
[INFO] MCP server listening at /mcp
[INFO] Operations included in MCP server:
[INFO] Operation 'your_operation' included in MCP
```

## Additional Resources
- [FastAPI-MCP Documentation](https://fastapi-mcp.tadata.com/)
- [HTTPX Documentation](https://www.python-httpx.org/)
