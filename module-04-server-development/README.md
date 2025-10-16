# Module 4: Basic Server Development

Welcome to Module 4! Now that you understand advanced MCP concepts, let's focus on building your first real MCP servers. This module covers basic server patterns, tool implementation strategies, and simple configuration management.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Build functional MCP servers with multiple tools
- Implement proper error handling and validation
- Manage basic server configuration
- Create reusable server components
- Test your MCP servers effectively
- Understand server lifecycle management

## 📚 Topics Covered

1. [Basic Server Patterns](#basic-server-patterns)
2. [Tool Implementation Strategies](#tool-implementation-strategies)
3. [Configuration Management](#configuration-management)
4. [Error Handling Patterns](#error-handling-patterns)
5. [Testing Your Servers](#testing-your-servers)
6. [Exercises](#exercises)

---

## Basic Server Patterns

### Simple Server Structure

Let's start with a clean, simple server structure that you can build upon:

```python
#!/usr/bin/env python3
"""
Basic MCP Server Template
A simple structure for building MCP servers.
"""

import asyncio
import logging
from typing import Dict, Any, List
from mcp.server import Server
from mcp.types import ToolExecutionError, ErrorCode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicMCPServer:
    """Basic MCP server with common patterns."""
    
    def __init__(self, name: str):
        self.name = name
        self.server = Server(name)
        self.tools = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all server tools."""
        
        @self.server.tool("ping")
        async def ping() -> Dict[str, Any]:
            """Check if the server is responding."""
            return {
                "status": "pong",
                "server": self.name,
                "timestamp": asyncio.get_event_loop().time()
            }
        
        @self.server.tool("get_info")
        async def get_info() -> Dict[str, Any]:
            """Get server information."""
            return {
                "name": self.name,
                "tools": list(self.tools.keys()),
                "status": "running"
            }
    
    async def run(self):
        """Run the server."""
        logger.info(f"Starting {self.name}...")
        await self.server.run()

# Usage
if __name__ == "__main__":
    server = BasicMCPServer("basic-server")
    asyncio.run(server.run())
```

### Server with Data Storage

Here's a server that manages data with proper validation:

```python
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os

@dataclass
class User:
    """User data model."""
    id: str
    name: str
    email: str
    age: int
    created_at: str

class UserManagementServer:
    """MCP server for user management."""
    
    def __init__(self, name: str, data_file: str = "users.json"):
        self.name = name
        self.data_file = data_file
        self.users: Dict[str, User] = {}
        self.server = Server(name)
        self._load_data()
        self._register_tools()
    
    def _load_data(self):
        """Load user data from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for user_id, user_data in data.items():
                        self.users[user_id] = User(**user_data)
                logger.info(f"Loaded {len(self.users)} users")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save user data to file."""
        try:
            data = {user_id: asdict(user) for user_id, user in self.users.items()}
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _register_tools(self):
        """Register user management tools."""
        
        @self.server.tool("create_user")
        async def create_user(name: str, email: str, age: int) -> Dict[str, Any]:
            """Create a new user."""
            try:
                # Validate input
                if not name or len(name) < 2:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Name must be at least 2 characters"
                    )
                
                if not email or "@" not in email:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Valid email address required"
                    )
                
                if age < 0 or age > 150:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Age must be between 0 and 150"
                    )
                
                # Check if email already exists
                for user in self.users.values():
                    if user.email == email:
                        raise ToolExecutionError(
                            code=ErrorCode.INVALID_PARAMS,
                            message="Email already exists"
                        )
                
                # Create user
                user_id = f"user_{len(self.users) + 1}"
                user = User(
                    id=user_id,
                    name=name,
                    email=email,
                    age=age,
                    created_at=asyncio.get_event_loop().time()
                )
                
                self.users[user_id] = user
                self._save_data()
                
                return {
                    "status": "created",
                    "user_id": user_id,
                    "user": asdict(user)
                }
                
            except ToolExecutionError:
                raise
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to create user: {str(e)}"
                )
        
        @self.server.tool("get_user")
        async def get_user(user_id: str) -> Dict[str, Any]:
            """Get user by ID."""
            if user_id not in self.users:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"User {user_id} not found"
                )
            
            return {
                "status": "found",
                "user": asdict(self.users[user_id])
            }
        
        @self.server.tool("list_users")
        async def list_users() -> Dict[str, Any]:
            """List all users."""
            return {
                "status": "success",
                "count": len(self.users),
                "users": [asdict(user) for user in self.users.values()]
            }
    
    async def run(self):
        """Run the server."""
        logger.info(f"Starting {self.name}...")
        await self.server.run()
```

## Tool Implementation Strategies

### Tool with External API Integration

```python
import aiohttp
from typing import Optional

class WeatherServer:
    """MCP server for weather information."""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.server = Server(name)
        self._register_tools()
    
    def _register_tools(self):
        """Register weather tools."""
        
        @self.server.tool("get_weather")
        async def get_weather(location: str) -> Dict[str, Any]:
            """Get current weather for a location."""
            try:
                # Mock weather data (in real implementation, call weather API)
                weather_data = {
                    "location": location,
                    "temperature": 22,
                    "condition": "sunny",
                    "humidity": 65,
                    "wind_speed": 10
                }
                
                return {
                    "status": "success",
                    "weather": weather_data
                }
                
            except Exception as e:
                logger.error(f"Weather API error: {e}")
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Weather service unavailable: {str(e)}"
                )
        
        @self.server.tool("get_forecast")
        async def get_forecast(location: str, days: int = 3) -> Dict[str, Any]:
            """Get weather forecast."""
            try:
                if days < 1 or days > 7:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Days must be between 1 and 7"
                    )
                
                # Mock forecast data
                forecast = []
                for i in range(days):
                    forecast.append({
                        "day": i + 1,
                        "temperature": 20 + (i * 2),
                        "condition": ["sunny", "cloudy", "rainy"][i % 3]
                    })
                
                return {
                    "status": "success",
                    "location": location,
                    "forecast": forecast
                }
                
            except ToolExecutionError:
                raise
            except Exception as e:
                logger.error(f"Forecast error: {e}")
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Forecast unavailable: {str(e)}"
                )
```

### Tool with File Operations

```python
import os
from pathlib import Path

class FileServer:
    """MCP server for file operations."""
    
    def __init__(self, name: str, base_path: str = "."):
        self.name = name
        self.base_path = Path(base_path).resolve()
        self.server = Server(name)
        self._register_tools()
    
    def _register_tools(self):
        """Register file operation tools."""
        
        @self.server.tool("read_file")
        async def read_file(file_path: str) -> Dict[str, Any]:
            """Read a text file."""
            try:
                # Security check - prevent path traversal
                full_path = (self.base_path / file_path).resolve()
                if not str(full_path).startswith(str(self.base_path)):
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Invalid file path"
                    )
                
                if not full_path.exists():
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"File not found: {file_path}"
                    )
                
                if not full_path.is_file():
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"Path is not a file: {file_path}"
                    )
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return {
                    "status": "success",
                    "file_path": file_path,
                    "content": content,
                    "size": len(content)
                }
                
            except ToolExecutionError:
                raise
            except Exception as e:
                logger.error(f"File read error: {e}")
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to read file: {str(e)}"
                )
        
        @self.server.tool("list_files")
        async def list_files(directory: str = ".") -> Dict[str, Any]:
            """List files in a directory."""
            try:
                dir_path = (self.base_path / directory).resolve()
                
                # Security check
                if not str(dir_path).startswith(str(self.base_path)):
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Invalid directory path"
                    )
                
                if not dir_path.exists():
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"Directory not found: {directory}"
                    )
                
                if not dir_path.is_dir():
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"Path is not a directory: {directory}"
                    )
                
                files = []
                for item in dir_path.iterdir():
                    files.append({
                        "name": item.name,
                        "type": "file" if item.is_file() else "directory",
                        "size": item.stat().st_size if item.is_file() else None
                    })
                
                return {
                    "status": "success",
                    "directory": directory,
                    "files": files,
                    "count": len(files)
                }
                
            except ToolExecutionError:
                raise
            except Exception as e:
                logger.error(f"Directory listing error: {e}")
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to list directory: {str(e)}"
                )
```

## Configuration Management

### Simple Configuration System

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class ServerConfig:
    """Basic server configuration."""
    name: str
    debug: bool = False
    log_level: str = "INFO"
    data_file: Optional[str] = None
    api_key: Optional[str] = None

class ConfigurableServer:
    """MCP server with configuration management."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.server = Server(config.name)
        self._setup_logging()
        self._register_tools()
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.config.debug:
            logger.info("Debug mode enabled")
    
    def _register_tools(self):
        """Register server tools."""
        
        @self.server.tool("get_config")
        async def get_config() -> Dict[str, Any]:
            """Get server configuration (excluding sensitive data)."""
            return {
                "name": self.config.name,
                "debug": self.config.debug,
                "log_level": self.config.log_level,
                "has_api_key": bool(self.config.api_key)
            }
        
        @self.server.tool("health_check")
        async def health_check() -> Dict[str, Any]:
            """Check server health."""
            return {
                "status": "healthy",
                "server": self.config.name,
                "debug_mode": self.config.debug
            }

# Usage
config = ServerConfig(
    name="configurable-server",
    debug=True,
    log_level="DEBUG",
    data_file="server_data.json"
)

server = ConfigurableServer(config)
```

## Error Handling Patterns

### Comprehensive Error Handling

```python
class RobustServer:
    """MCP server with robust error handling."""
    
    def __init__(self, name: str):
        self.name = name
        self.server = Server(name)
        self.request_count = 0
        self.error_count = 0
        self._register_tools()
    
    def _register_tools(self):
        """Register tools with error handling."""
        
        @self.server.tool("safe_operation")
        async def safe_operation(operation: str, data: str) -> Dict[str, Any]:
            """Perform a safe operation with comprehensive error handling."""
            self.request_count += 1
            
            try:
                # Input validation
                if not operation:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Operation parameter is required"
                    )
                
                if not data:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Data parameter is required"
                    )
                
                # Business logic validation
                if len(data) > 1000:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Data too large (max 1000 characters)"
                    )
                
                # Simulate different operations
                if operation == "uppercase":
                    result = data.upper()
                elif operation == "lowercase":
                    result = data.lower()
                elif operation == "reverse":
                    result = data[::-1]
                elif operation == "count":
                    result = str(len(data))
                else:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"Unknown operation: {operation}"
                    )
                
                logger.info(f"Operation '{operation}' completed successfully")
                return {
                    "status": "success",
                    "operation": operation,
                    "result": result,
                    "input_length": len(data)
                }
                
            except ToolExecutionError:
                self.error_count += 1
                raise
            except Exception as e:
                self.error_count += 1
                logger.error(f"Unexpected error in safe_operation: {e}")
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Operation failed: {str(e)}"
                )
        
        @self.server.tool("get_stats")
        async def get_stats() -> Dict[str, Any]:
            """Get server statistics."""
            error_rate = self.error_count / max(self.request_count, 1)
            
            return {
                "status": "success",
                "server": self.name,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": round(error_rate, 3),
                "uptime": "running"
            }
```

## Testing Your Servers

### Basic Testing Framework

```python
import pytest
import asyncio
from unittest.mock import Mock, patch

class TestMCPServer:
    """Test suite for MCP servers."""
    
    @pytest.fixture
    async def server(self):
        """Create test server instance."""
        server = BasicMCPServer("test-server")
        yield server
    
    @pytest.mark.asyncio
    async def test_ping_tool(self, server):
        """Test ping tool."""
        result = await server.server.tools["ping"]()
        
        assert result["status"] == "pong"
        assert result["server"] == "test-server"
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_get_info_tool(self, server):
        """Test get_info tool."""
        result = await server.server.tools["get_info"]()
        
        assert result["name"] == "test-server"
        assert result["status"] == "running"
        assert isinstance(result["tools"], list)

class TestUserManagementServer:
    """Test suite for user management server."""
    
    @pytest.fixture
    async def user_server(self, tmp_path):
        """Create test user server."""
        data_file = tmp_path / "test_users.json"
        server = UserManagementServer("test-user-server", str(data_file))
        yield server
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, user_server):
        """Test successful user creation."""
        result = await user_server.server.tools["create_user"](
            name="John Doe",
            email="john@example.com",
            age=30
        )
        
        assert result["status"] == "created"
        assert result["user"]["name"] == "John Doe"
        assert result["user"]["email"] == "john@example.com"
        assert result["user"]["age"] == 30
    
    @pytest.mark.asyncio
    async def test_create_user_validation_error(self, user_server):
        """Test user creation with validation error."""
        with pytest.raises(ToolExecutionError) as exc_info:
            await user_server.server.tools["create_user"](
                name="",  # Invalid name
                email="invalid-email",
                age=-1  # Invalid age
            )
        
        assert exc_info.value.code == ErrorCode.INVALID_PARAMS
    
    @pytest.mark.asyncio
    async def test_get_user_not_found(self, user_server):
        """Test getting non-existent user."""
        with pytest.raises(ToolExecutionError) as exc_info:
            await user_server.server.tools["get_user"]("nonexistent")
        
        assert exc_info.value.code == ErrorCode.NOT_FOUND
```

## Exercises

### Exercise 1: Task Management Server

Create an MCP server for task management with the following tools:

**Requirements:**
- `create_task`: Create a new task with title, description, and priority
- `get_task`: Get task by ID
- `update_task`: Update task status or details
- `list_tasks`: List all tasks with optional filtering
- `delete_task`: Remove a task
- Proper validation and error handling
- Data persistence to JSON file

**Hint:**
```python
@dataclass
class Task:
    id: str
    title: str
    description: str
    priority: str  # "low", "medium", "high"
    status: str    # "pending", "in_progress", "completed"
    created_at: str
    updated_at: str
```

### Exercise 2: Calculator Server

Create an MCP server with advanced calculator tools:

**Requirements:**
- `calculate`: Basic arithmetic operations
- `calculate_advanced`: Advanced operations (power, sqrt, log)
- `calculate_batch`: Process multiple calculations
- `get_history`: Get calculation history
- `clear_history`: Clear calculation history
- Input validation for all operations
- Error handling for invalid operations

### Exercise 3: File Manager Server

Create an MCP server for file management:

**Requirements:**
- `read_file`: Read text files safely
- `write_file`: Write content to files
- `list_directory`: List files and directories
- `create_directory`: Create new directories
- `delete_file`: Delete files safely
- `get_file_info`: Get file metadata
- Security checks to prevent path traversal
- Proper error handling for file operations

## 🎯 Module Summary

In this module, you learned:

- ✅ Basic server patterns and structures
- ✅ Tool implementation strategies
- ✅ Simple configuration management
- ✅ Comprehensive error handling patterns
- ✅ Testing methodologies for MCP servers
- ✅ Data persistence and validation

## 🚀 Next Steps

You're now ready to move on to **Module 5: Client Integration**, where you'll learn about:
- MCP client development patterns
- AI application integration strategies
- Performance optimization techniques
- Error handling and recovery

---

**Congratulations on completing Module 4! 🎉**

*Next: [Module 5: Client Integration](module-05-client-integration/README.md)*