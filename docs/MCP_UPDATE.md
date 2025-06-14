# FastAPI-MCP Complete Upgrade Guide

## Overview
This guide covers upgrading FastAPI-MCP from versions ≤ 0.2.0 to version 0.3.4. The main issues you'll encounter are related to parameter changes and URL protocol handling.

## Common Errors You'll See

1. **Base URL Parameter Error:**
```
TypeError: FastApiMCP.__init__() got an unexpected keyword argument 'base_url'
```

2. **Protocol Missing Error:**
```
httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol
```

## Step-by-Step Fix

### 1. Update requirements.txt
Change your FastAPI-MCP version:
```diff
- fastapi-mcp==0.2.0
+ fastapi-mcp==0.3.4
```

### 2. Identify Your Current MCP Initialization
Look for code that looks like this (OLD VERSION):
```python
# OLD - This will cause errors
mcp = FastApiMCP(
    app,
    name="Your MCP API",
    description="Your description",
    base_url="http://localhost:8000",  # ❌ This parameter is removed
    include_operations=["your_operation"],
    http_client=httpx.AsyncClient(timeout=180.0)  # ❌ Missing base_url here
)
```

### 3. Replace with New Configuration
Replace the old initialization with this pattern:

```python
# NEW - Working configuration
base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
# Ensure protocol is included
if not base_url.startswith(("http://", "https://")):
    base_url = f"http://{base_url}"

mcp = FastApiMCP(
    app,
    name="Your MCP API",
    description="Your description",
    include_operations=[
        "your_operation_name"  # Replace with your actual operation
    ],
    describe_all_responses=True,
    describe_full_response_schema=True,
    http_client=httpx.AsyncClient(
        timeout=180.0,  # Or your preferred timeout
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url=base_url  # ✅ base_url goes here now
    )
)

# Mount the MCP server
mcp.mount()
```

## Key Changes Summary

| What Changed | Old Way | New Way |
|--------------|---------|---------|
| Base URL | `FastApiMCP(..., base_url="...")` | `httpx.AsyncClient(..., base_url="...")` |
| Protocol Check | Not required | Must ensure http:// or https:// prefix |
| Connection Limits | Not configured | Recommended to add limits |

## Environment Variables
Make sure you handle different environments:
- **Local development:** `http://localhost:8000`
- **Production:** Use environment variable like `RENDER_EXTERNAL_URL`
- **Docker/Container:** May need container-specific URL

## Complete Working Example
Here's a complete working example you can adapt:

```python
import os
import httpx
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

# Your FastAPI routes here
@app.post("/your-endpoint")
async def your_endpoint():
    # Your endpoint logic
    pass

# MCP Configuration
base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
if not base_url.startswith(("http://", "https://")):
    base_url = f"http://{base_url}"

mcp = FastApiMCP(
    app,
    name="Your MCP API",
    description="Your MCP server description",
    include_operations=[
        "your_endpoint"  # Must match your FastAPI route operation_id
    ],
    describe_all_responses=True,
    describe_full_response_schema=True,
    http_client=httpx.AsyncClient(
        timeout=300.0,  # 5 minutes timeout
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url=base_url
    )
)

mcp.mount()
```

## Verification Steps
After making changes:

1. **Install updated requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test locally:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Check logs for these success messages:**
   ```
   [INFO] MCP server listening at /mcp
   [INFO] Operation 'your_operation' included in MCP
   [INFO] MCP server exposing ... endpoints
   ```

4. **Test MCP endpoint:**
   - Visit: `http://localhost:8000/mcp`
   - Should return MCP server info without errors

## Troubleshooting

### If you still get protocol errors:
- Check that your `base_url` includes `http://` or `https://`
- Verify environment variables are set correctly
- Ensure no trailing slashes in URLs

### If operations aren't recognized:
- Check that `include_operations` matches your FastAPI route `operation_id`
- Verify your FastAPI routes are defined before MCP initialization
- Check that routes have proper decorators and return types

### If timeout errors occur:
- Increase timeout value in httpx.AsyncClient
- Add connection limits as shown in the example
- Consider your operation's actual processing time

## Notes for AI Coders
When implementing these changes:
1. Always backup the original code first
2. Update requirements.txt before changing code
3. Look for the FastApiMCP initialization - it's usually near the end of main.py
4. Replace the entire MCP initialization block, don't just modify parameters
5. Test locally before deploying to production
6. Check logs carefully for any remaining errors
