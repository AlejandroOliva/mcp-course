# Module 2: MCP Fundamentals

Welcome to Module 2! Now that you understand what MCP is, let's dive deep into the fundamental concepts that make MCP work. This module will teach you the core architecture, patterns, and best practices for building MCP servers and clients.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Understand MCP server-client architecture
- Define and implement tools with proper schemas
- Manage resources effectively
- Handle errors and edge cases
- Implement basic authentication
- Understand JSON-RPC communication

## 📚 Topics Covered

1. [Server-Client Architecture](#server-client-architecture)
2. [Tool Definitions and Schemas](#tool-definitions-and-schemas)
3. [Resource Management](#resource-management)
4. [Error Handling](#error-handling)
5. [JSON-RPC Communication](#json-rpc-communication)
6. [Authentication Basics](#authentication-basics)
7. [Exercises](#exercises)

---

## Server-Client Architecture

### MCP Architecture Overview

MCP follows a client-server architecture where:
- **MCP Server**: Provides tools and resources
- **MCP Client**: Consumes tools and resources (typically an AI agent)

```
┌─────────────────┐    JSON-RPC    ┌─────────────────┐
│   AI Client     │◄──────────────►│   MCP Server    │
│   (Claude, etc.)│                │   (Your Code)   │
└─────────────────┘                └─────────────────┘
                                           │
                                           ▼
                                   ┌─────────────────┐
                                   │ External Systems│
                                   │ (APIs, DBs, etc)│
                                   └─────────────────┘
```

### Server Responsibilities

1. **Tool Management**: Define, validate, and execute tools
2. **Resource Provision**: Provide access to data sources
3. **Security**: Handle authentication and authorization
4. **Error Handling**: Provide meaningful error responses
5. **Performance**: Optimize for production use

### Client Responsibilities

1. **Tool Discovery**: Find available tools and their schemas
2. **Tool Execution**: Call tools with proper parameters
3. **Resource Access**: Read and consume resources
4. **Error Handling**: Handle server errors gracefully
5. **Caching**: Optimize repeated operations

## Tool Definitions and Schemas

### Tool Schema Structure

Every MCP tool must have a well-defined schema that describes:
- **Name**: Unique identifier for the tool
- **Description**: What the tool does
- **Input Schema**: Parameters and their types
- **Output Schema**: Return value structure

### Python Tool Definition

```python
from mcp.server import Server
from mcp.types import Tool
from pydantic import BaseModel, Field

server = Server("example-server")

class UserInput(BaseModel):
    name: str = Field(..., description="User's name")
    age: int = Field(..., ge=0, le=120, description="User's age")
    email: str = Field(..., description="User's email address")

@server.tool("create_user")
async def create_user(user_data: UserInput) -> dict:
    """
    Create a new user account.
    
    Args:
        user_data: User information including name, age, and email
    
    Returns:
        Dictionary containing user ID and creation status
    """
    # Validate input (Pydantic handles this automatically)
    user_id = f"user_{len(users) + 1}"
    
    # Store user data
    users[user_id] = {
        "id": user_id,
        "name": user_data.name,
        "age": user_data.age,
        "email": user_data.email,
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "user_id": user_id,
        "status": "created",
        "message": f"User {user_data.name} created successfully"
    }
```

### Node.js Tool Definition

```javascript
import { z } from 'zod';

const UserSchema = z.object({
  name: z.string().describe("User's name"),
  age: z.number().min(0).max(120).describe("User's age"),
  email: z.string().email().describe("User's email address")
});

const tools = [
  {
    name: "create_user",
    description: "Create a new user account",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "User's name" },
        age: { 
          type: "number", 
          minimum: 0, 
          maximum: 120, 
          description: "User's age" 
        },
        email: { 
          type: "string", 
          format: "email", 
          description: "User's email address" 
        }
      },
      required: ["name", "age", "email"]
    }
  }
];

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === "create_user") {
    // Validate input
    const userData = UserSchema.parse(args);
    
    const userId = `user_${Object.keys(users).length + 1}`;
    
    users[userId] = {
      id: userId,
      ...userData,
      created_at: new Date().toISOString()
    };
    
    return {
      content: [{
        type: "text",
        text: JSON.stringify({
          user_id: userId,
          status: "created",
          message: `User ${userData.name} created successfully`
        })
      }]
    };
  }
});
```

### Advanced Tool Patterns

#### Async Operations

```python
import asyncio
import aiohttp

@server.tool("fetch_data")
async def fetch_data(urls: list[str]) -> dict:
    """
    Fetch data from multiple URLs concurrently.
    
    Args:
        urls: List of URLs to fetch
    
    Returns:
        Dictionary with URL and response data
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = session.get(url)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results[urls[i]] = {"error": str(response)}
            else:
                results[urls[i]] = await response.text()
        
        return results
```

#### Streaming Responses

```python
from mcp.types import TextContent

@server.tool("stream_data")
async def stream_data(count: int) -> list[TextContent]:
    """
    Stream data in chunks.
    
    Args:
        count: Number of items to stream
    
    Returns:
        List of text content chunks
    """
    chunks = []
    for i in range(count):
        chunk = TextContent(
            type="text",
            text=f"Chunk {i + 1}: Data content here"
        )
        chunks.append(chunk)
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
    
    return chunks
```

## Resource Management

### Resource Types

MCP supports different types of resources:
- **Text Resources**: Plain text content
- **JSON Resources**: Structured data
- **Binary Resources**: Files, images, etc.
- **Streaming Resources**: Real-time data

### Text Resource Example

```python
from mcp.types import Resource

@server.resource("config://settings")
async def get_settings() -> Resource:
    """Provide application settings as a resource."""
    settings = {
        "app_name": "My MCP Server",
        "version": "1.0.0",
        "debug_mode": False,
        "max_connections": 100
    }
    
    return Resource(
        uri="config://settings",
        name="Application Settings",
        description="Current application configuration",
        mimeType="application/json",
        text=json.dumps(settings, indent=2)
    )
```

### Dynamic Resource Example

```python
@server.resource("data://users/{user_id}")
async def get_user_resource(uri: str) -> Resource:
    """Provide user data as a resource."""
    # Extract user_id from URI
    user_id = uri.split("/")[-1]
    
    if user_id not in users:
        raise ResourceAccessError(
            code=ErrorCode.NOT_FOUND,
            message=f"User {user_id} not found"
        )
    
    user_data = users[user_id]
    
    return Resource(
        uri=uri,
        name=f"User: {user_data['name']}",
        description=f"User data for {user_data['name']}",
        mimeType="application/json",
        text=json.dumps(user_data, indent=2)
    )
```

### Resource Listing

```python
@server.list_resources()
async def list_resources() -> list[Resource]:
    """List all available resources."""
    resources = []
    
    # Static resources
    resources.append(Resource(
        uri="config://settings",
        name="Application Settings",
        description="Current application configuration",
        mimeType="application/json"
    ))
    
    # Dynamic resources
    for user_id, user_data in users.items():
        resources.append(Resource(
            uri=f"data://users/{user_id}",
            name=f"User: {user_data['name']}",
            description=f"User data for {user_data['name']}",
            mimeType="application/json"
        ))
    
    return resources
```

## Error Handling

### Error Types

MCP defines several error types:
- **ToolExecutionError**: Tool execution failed
- **ResourceAccessError**: Resource access failed
- **AuthenticationError**: Authentication failed
- **ValidationError**: Input validation failed

### Comprehensive Error Handling

```python
from mcp.types import ToolExecutionError, ResourceAccessError, ErrorCode
import logging

logger = logging.getLogger(__name__)

@server.tool("risky_operation")
async def risky_operation(param: str) -> str:
    """
    Demonstrate comprehensive error handling.
    
    Args:
        param: Input parameter
    
    Returns:
        Success message or error
    """
    try:
        # Input validation
        if not param or len(param) < 3:
            raise ToolExecutionError(
                code=ErrorCode.INVALID_PARAMS,
                message="Parameter must be at least 3 characters long"
            )
        
        # Business logic validation
        if param.lower() == "error":
            raise ToolExecutionError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Simulated business logic error"
            )
        
        # External service call (simulated)
        if param.lower() == "timeout":
            await asyncio.sleep(5)  # Simulate timeout
            raise ToolExecutionError(
                code=ErrorCode.TIMEOUT,
                message="Operation timed out"
            )
        
        # Success case
        logger.info(f"Operation completed successfully for param: {param}")
        return f"Operation completed successfully: {param}"
        
    except ToolExecutionError:
        # Re-raise MCP errors
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in risky_operation: {e}")
        raise ToolExecutionError(
            code=ErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error: {str(e)}"
        )
```

### Error Recovery Patterns

```python
@server.tool("resilient_operation")
async def resilient_operation(url: str, max_retries: int = 3) -> str:
    """
    Demonstrate error recovery with retries.
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
    
    Returns:
        Fetched content or error message
    """
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
        
        except asyncio.TimeoutError:
            if attempt < max_retries:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise ToolExecutionError(
                    code=ErrorCode.TIMEOUT,
                    message=f"Operation timed out after {max_retries} retries"
                )
        
        except aiohttp.ClientError as e:
            if attempt < max_retries:
                logger.warning(f"Client error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"External service error: {str(e)}"
                )
    
    # This should never be reached
    raise ToolExecutionError(
        code=ErrorCode.INTERNAL_ERROR,
        message="Unexpected error in retry logic"
    )
```

## JSON-RPC Communication

### Request/Response Format

MCP uses JSON-RPC 2.0 for all communication:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "create_user",
    "arguments": {
      "name": "John Doe",
      "age": 30,
      "email": "john@example.com"
    }
  }
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"user_id\": \"user_1\", \"status\": \"created\"}"
      }
    ]
  }
}
```

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "details": "Parameter 'name' is required"
    }
  }
}
```

## Authentication Basics

### API Key Authentication

```python
from mcp.types import AuthenticationError

class AuthenticatedServer(Server):
    def __init__(self, name: str, api_key: str):
        super().__init__(name)
        self.api_key = api_key
    
    async def authenticate_request(self, request):
        """Authenticate incoming requests."""
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            raise AuthenticationError(
                code=ErrorCode.UNAUTHORIZED,
                message="Authorization header required"
            )
        
        if not auth_header.startswith("Bearer "):
            raise AuthenticationError(
                code=ErrorCode.UNAUTHORIZED,
                message="Invalid authorization format"
            )
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        if token != self.api_key:
            raise AuthenticationError(
                code=ErrorCode.UNAUTHORIZED,
                message="Invalid API key"
            )
        
        return True

# Usage
server = AuthenticatedServer("secure-server", "your-secret-api-key")
```

### Token-based Authentication

```python
import jwt
from datetime import datetime, timedelta

class TokenAuthenticatedServer(Server):
    def __init__(self, name: str, secret_key: str):
        super().__init__(name)
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    async def authenticate_token(self, token: str) -> str:
        """Authenticate JWT token and return user_id."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload["user_id"]
        except jwt.ExpiredSignatureError:
            raise AuthenticationError(
                code=ErrorCode.UNAUTHORIZED,
                message="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise AuthenticationError(
                code=ErrorCode.UNAUTHORIZED,
                message="Invalid token"
            )
```

## Exercises

### Exercise 1: User Management System

Create an MCP server with comprehensive user management tools.

**Requirements:**
- `create_user`: Create new users with validation
- `get_user`: Retrieve user information
- `update_user`: Update user data
- `delete_user`: Remove users
- `list_users`: List all users
- Proper error handling for all operations
- Input validation using Pydantic models

### Exercise 2: File System Resource Provider

Create an MCP server that provides file system access as resources.

**Requirements:**
- Resource provider for text files
- Support for directory listing
- File content reading
- Proper error handling for file operations
- Security considerations (path traversal prevention)

### Exercise 3: Weather API Integration

Create an MCP server that integrates with a weather API.

**Requirements:**
- `get_current_weather`: Get current weather for a location
- `get_forecast`: Get weather forecast
- `get_weather_alerts`: Get weather alerts
- Proper error handling for API failures
- Rate limiting and caching

## 🎯 Module Summary

In this module, you learned:

- ✅ MCP server-client architecture and responsibilities
- ✅ How to define tools with proper schemas and validation
- ✅ Resource management patterns and best practices
- ✅ Comprehensive error handling strategies
- ✅ JSON-RPC communication format
- ✅ Basic authentication mechanisms

## 🚀 Next Steps

You're now ready to move on to **Module 3: Advanced MCP Concepts**, where you'll learn about:
- Custom tool development patterns
- Advanced resource providers
- Prompt template systems
- Streaming and async operations
- Security best practices

---

**Congratulations on completing Module 2! 🎉**

*Next: [Module 3: Advanced MCP Concepts](module-03-advanced-concepts/README.md)*
