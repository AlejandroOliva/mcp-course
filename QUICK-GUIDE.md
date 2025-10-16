# 🚀 MCP Quick Reference Guide

## 📋 MCP Protocol Basics

### Core Concepts
- **MCP Server**: Provides tools and resources to AI agents
- **MCP Client**: AI agent that consumes tools and resources
- **Tools**: Functions that AI agents can call
- **Resources**: Data sources that AI agents can access
- **Prompts**: Templates for AI agent interactions

## 🔧 Essential Commands

### Installation
```bash
# Python SDK
pip install mcp

# Node.js SDK
npm install @modelcontextprotocol/sdk

# Go SDK
go get github.com/modelcontextprotocol/go-sdk
```

### Development
```bash
# Run MCP server
python3 server.py

# Test server connection
mcp-client connect server.py

# Validate server implementation
mcp-validate server.py

# Generate server template
mcp-init my-server
```

## 📝 Common Patterns

### Basic Tool Definition (Python)
```python
from mcp.server import Server
from mcp.types import Tool

server = Server("my-server")

@server.tool("example_tool")
async def example_tool(param: str) -> str:
    """Example tool that does something useful."""
    return f"Processed: {param}"

if __name__ == "__main__":
    server.run()
```

### Basic Tool Definition (Node.js)
```javascript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: "my-server",
  version: "1.0.0"
});

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === "example_tool") {
    return {
      content: [{ type: "text", text: `Processed: ${args.param}` }]
    };
  }
  
  throw new Error(`Unknown tool: ${name}`);
});

const transport = new StdioServerTransport();
server.connect(transport);
```

### Resource Definition
```python
@server.resource("file://example.txt")
async def get_file_resource(uri: str) -> Resource:
    """Provide file content as a resource."""
    with open(uri.replace("file://", ""), "r") as f:
        content = f.read()
    
    return Resource(
        uri=uri,
        name="Example File",
        description="An example file resource",
        mimeType="text/plain",
        text=content
    )
```

## 🛠️ Tool Schema Examples

### Simple Tool
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      }
    },
    "required": ["location"]
  }
}
```

### Complex Tool
```json
{
  "name": "analyze_data",
  "description": "Analyze data with various options",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data": {
        "type": "array",
        "description": "Data to analyze"
      },
      "method": {
        "type": "string",
        "enum": ["statistical", "ml", "visualization"],
        "description": "Analysis method"
      },
      "options": {
        "type": "object",
        "description": "Additional analysis options"
      }
    },
    "required": ["data", "method"]
  }
}
```

## 📊 Resource Types

### Text Resource
```python
Resource(
    uri="text://example",
    name="Example Text",
    description="Sample text resource",
    mimeType="text/plain",
    text="This is example text content"
)
```

### JSON Resource
```python
Resource(
    uri="json://data",
    name="Example Data",
    description="Sample JSON data",
    mimeType="application/json",
    text=json.dumps({"key": "value"})
)
```

### Binary Resource
```python
Resource(
    uri="binary://image.png",
    name="Example Image",
    description="Sample image file",
    mimeType="image/png",
    blob=binary_data
)
```

## 🔐 Authentication Patterns

### API Key Authentication
```python
async def authenticate_request(request):
    api_key = request.headers.get("Authorization")
    if not api_key or not validate_api_key(api_key):
        raise AuthenticationError("Invalid API key")
    return True
```

### Token-based Authentication
```python
async def authenticate_with_token(request):
    token = request.headers.get("X-Auth-Token")
    if not token or not validate_token(token):
        raise AuthenticationError("Invalid token")
    return True
```

## ⚡ Error Handling

### Standard Error Responses
```python
from mcp.types import ErrorCode

# Tool execution error
raise ToolExecutionError(
    code=ErrorCode.INVALID_PARAMS,
    message="Invalid parameter provided"
)

# Resource access error
raise ResourceAccessError(
    code=ErrorCode.NOT_FOUND,
    message="Resource not found"
)

# Authentication error
raise AuthenticationError(
    code=ErrorCode.UNAUTHORIZED,
    message="Authentication required"
)
```

## 🚀 Deployment Patterns

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "server.py"]
```

### Environment Configuration
```python
import os

# Server configuration
SERVER_NAME = os.getenv("MCP_SERVER_NAME", "my-server")
SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")

# API configuration
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.example.com")
```

## 📈 Performance Tips

### Async Operations
```python
import asyncio
import aiohttp

async def fetch_data_async(urls):
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await resp.text() for resp in responses]
```

### Caching
```python
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def expensive_computation(param):
    """Cache expensive computations."""
    time.sleep(1)  # Simulate expensive operation
    return param * 2
```

### Streaming Responses
```python
async def stream_large_data():
    """Stream large data in chunks."""
    for chunk in large_data_generator():
        yield chunk
        await asyncio.sleep(0.01)  # Yield control
```

## 🔍 Debugging

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use in tools
logger.info(f"Tool called with params: {params}")
```

### Error Tracking
```python
import traceback

try:
    result = await risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    logger.error(traceback.format_exc())
    raise
```

## 📚 Common Libraries

### Python
- `mcp`: Official MCP SDK
- `aiohttp`: Async HTTP client
- `pydantic`: Data validation
- `fastapi`: Web framework (if needed)
- `sqlalchemy`: Database ORM

### Node.js
- `@modelcontextprotocol/sdk`: Official MCP SDK
- `axios`: HTTP client
- `joi`: Data validation
- `express`: Web framework (if needed)
- `prisma`: Database ORM

### Go
- `github.com/modelcontextprotocol/go-sdk`: Official MCP SDK
- `github.com/gin-gonic/gin`: Web framework
- `gorm.io/gorm`: Database ORM
- `github.com/go-playground/validator`: Data validation

## 🎯 Best Practices

1. **Always validate input parameters**
2. **Provide clear error messages**
3. **Use async/await for I/O operations**
4. **Implement proper logging**
5. **Handle edge cases gracefully**
6. **Test with real AI clients**
7. **Document your tools thoroughly**
8. **Use version control**
9. **Implement proper authentication**
10. **Monitor performance and errors**

---

*Keep this guide handy while developing MCP servers! 🚀*
