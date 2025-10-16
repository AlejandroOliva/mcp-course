# Module 4: Server Development

Welcome to Module 4! Now that you understand advanced MCP concepts, let's focus on building production-ready MCP servers. This module covers server architecture patterns, complex tool implementations, configuration management, deployment strategies, and comprehensive testing.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Design and implement production-ready MCP servers
- Apply complex tool implementation strategies
- Manage configuration and environment settings
- Deploy MCP servers to various environments
- Implement comprehensive testing strategies
- Monitor and maintain MCP servers

## 📚 Topics Covered

1. [Server Architecture Patterns](#server-architecture-patterns)
2. [Complex Tool Implementation](#complex-tool-implementation)
3. [Configuration Management](#configuration-management)
4. [Deployment Strategies](#deployment-strategies)
5. [Testing Strategies](#testing-strategies)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Exercises](#exercises)

---

## Server Architecture Patterns

### Layered Architecture

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
import logging

class DataLayer(ABC):
    """Abstract data access layer."""
    
    @abstractmethod
    async def get_data(self, query: str) -> Any:
        """Get data from data source."""
        pass
    
    @abstractmethod
    async def save_data(self, data: Any) -> bool:
        """Save data to data source."""
        pass

class BusinessLogicLayer(ABC):
    """Abstract business logic layer."""
    
    def __init__(self, data_layer: DataLayer):
        self.data_layer = data_layer
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """Process business request."""
        pass

class PresentationLayer:
    """MCP server presentation layer."""
    
    def __init__(self, business_logic: BusinessLogicLayer):
        self.business_logic = business_logic
        self.logger = logging.getLogger(__name__)
    
    async def handle_tool_call(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Handle MCP tool calls."""
        try:
            self.logger.info(f"Handling tool call: {tool_name}")
            
            # Validate input
            validated_params = self._validate_params(tool_name, params)
            
            # Process through business logic
            result = await self.business_logic.process_request({
                "tool": tool_name,
                "params": validated_params
            })
            
            self.logger.info(f"Tool {tool_name} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in tool {tool_name}: {e}")
            raise ToolExecutionError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Tool execution failed: {str(e)}"
            )
    
    def _validate_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters."""
        # Implementation depends on your validation requirements
        return params

# Concrete implementations
class DatabaseDataLayer(DataLayer):
    """Database data access layer."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Initialize database connection
    
    async def get_data(self, query: str) -> Any:
        """Get data from database."""
        # Implement database query logic
        pass
    
    async def save_data(self, data: Any) -> bool:
        """Save data to database."""
        # Implement database save logic
        pass

class UserBusinessLogic(BusinessLogicLayer):
    """User management business logic."""
    
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """Process user-related requests."""
        tool = request["tool"]
        params = request["params"]
        
        if tool == "create_user":
            return await self._create_user(params)
        elif tool == "get_user":
            return await self._get_user(params)
        elif tool == "update_user":
            return await self._update_user(params)
        else:
            raise ValueError(f"Unknown tool: {tool}")
    
    async def _create_user(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user."""
        user_data = {
            "name": params["name"],
            "email": params["email"],
            "created_at": datetime.now().isoformat()
        }
        
        # Save to data layer
        success = await self.data_layer.save_data(user_data)
        
        if success:
            return {"status": "created", "user": user_data}
        else:
            raise Exception("Failed to create user")
    
    async def _get_user(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get user information."""
        user_id = params["user_id"]
        query = f"SELECT * FROM users WHERE id = {user_id}"
        
        user_data = await self.data_layer.get_data(query)
        
        if user_data:
            return {"status": "found", "user": user_data}
        else:
            raise Exception("User not found")
    
    async def _update_user(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update user information."""
        # Implementation for updating user
        pass

# MCP Server using layered architecture
class LayeredMCPServer(Server):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        
        # Initialize layers
        self.data_layer = DatabaseDataLayer(config["database_url"])
        self.business_logic = UserBusinessLogic(self.data_layer)
        self.presentation_layer = PresentationLayer(self.business_logic)
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools."""
        
        @self.tool("create_user")
        async def create_user(name: str, email: str) -> Dict[str, Any]:
            """Create a new user."""
            return await self.presentation_layer.handle_tool_call(
                "create_user", 
                {"name": name, "email": email}
            )
        
        @self.tool("get_user")
        async def get_user(user_id: str) -> Dict[str, Any]:
            """Get user information."""
            return await self.presentation_layer.handle_tool_call(
                "get_user", 
                {"user_id": user_id}
            )
```

### Microservices Architecture

```python
from typing import Dict, Any, List
import aiohttp
import asyncio

class ServiceRegistry:
    """Registry for microservices."""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
    
    def register_service(self, name: str, url: str, health_check: str = None):
        """Register a microservice."""
        self.services[name] = {
            "url": url,
            "health_check": health_check or f"{url}/health",
            "status": "unknown"
        }
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(service["health_check"], timeout=5) as response:
                    is_healthy = response.status == 200
                    service["status"] = "healthy" if is_healthy else "unhealthy"
                    return is_healthy
        except Exception:
            service["status"] = "unhealthy"
            return False
    
    async def get_healthy_services(self) -> List[str]:
        """Get list of healthy services."""
        healthy_services = []
        
        for service_name in self.services:
            if await self.check_service_health(service_name):
                healthy_services.append(service_name)
        
        return healthy_services

class MicroserviceClient:
    """Client for communicating with microservices."""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_service(self, service_name: str, endpoint: str, method: str = "GET", data: Any = None) -> Any:
        """Call a microservice endpoint."""
        if service_name not in self.service_registry.services:
            raise ValueError(f"Service {service_name} not registered")
        
        service = self.service_registry.services[service_name]
        
        if service["status"] != "healthy":
            if not await self.service_registry.check_service_health(service_name):
                raise Exception(f"Service {service_name} is not healthy")
        
        url = f"{service['url']}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    return await response.json()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    return await response.json()
            elif method.upper() == "PUT":
                async with self.session.put(url, json=data) as response:
                    return await response.json()
            elif method.upper() == "DELETE":
                async with self.session.delete(url) as response:
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        
        except Exception as e:
            raise Exception(f"Error calling service {service_name}: {e}")

class MicroserviceMCPServer(Server):
    def __init__(self, name: str):
        super().__init__(name)
        self.service_registry = ServiceRegistry()
        self._register_services()
    
    def _register_services(self):
        """Register microservices."""
        self.service_registry.register_service("user-service", "http://user-service:8080")
        self.service_registry.register_service("order-service", "http://order-service:8080")
        self.service_registry.register_service("payment-service", "http://payment-service:8080")
    
    @server.tool("create_order")
    async def create_order(user_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an order using microservices."""
        async with MicroserviceClient(self.service_registry) as client:
            # Get user information
            user = await client.call_service("user-service", f"users/{user_id}")
            
            # Create order
            order_data = {
                "user_id": user_id,
                "items": items,
                "total": sum(item["price"] * item["quantity"] for item in items)
            }
            
            order = await client.call_service("order-service", "orders", "POST", order_data)
            
            # Process payment
            payment_data = {
                "order_id": order["id"],
                "amount": order["total"],
                "user_id": user_id
            }
            
            payment = await client.call_service("payment-service", "payments", "POST", payment_data)
            
            return {
                "order": order,
                "payment": payment,
                "status": "completed"
            }
```

## Complex Tool Implementation

### Database Integration Tools

```python
import asyncio
import asyncpg
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    min_connections: int = 5
    max_connections: int = 20

class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections
        )
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a SELECT query."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute an INSERT/UPDATE/DELETE command."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(command, *args)
            return result
    
    async def execute_transaction(self, operations: List[tuple]) -> List[str]:
        """Execute multiple operations in a transaction."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                results = []
                for operation, args in operations:
                    if operation.startswith("SELECT"):
                        result = await conn.fetch(operation, *args)
                        results.append([dict(row) for row in result])
                    else:
                        result = await conn.execute(operation, *args)
                        results.append(result)
                return results

class DatabaseMCPServer(Server):
    def __init__(self, name: str, db_config: DatabaseConfig):
        super().__init__(name)
        self.db_manager = DatabaseManager(db_config)
        self._register_database_tools()
    
    def _register_database_tools(self):
        """Register database-related tools."""
        
        @self.tool("execute_query")
        async def execute_query(query: str, params: List[Any] = None) -> Dict[str, Any]:
            """Execute a database query."""
            try:
                # Validate query (basic SQL injection prevention)
                if not query.strip().upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
                    raise ValueError("Only SELECT, INSERT, UPDATE, DELETE queries allowed")
                
                if query.strip().upper().startswith('SELECT'):
                    results = await self.db_manager.execute_query(query, *(params or []))
                    return {
                        "type": "query",
                        "results": results,
                        "row_count": len(results)
                    }
                else:
                    result = await self.db_manager.execute_command(query, *(params or []))
                    return {
                        "type": "command",
                        "result": result
                    }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Database operation failed: {str(e)}"
                )
        
        @self.tool("batch_operations")
        async def batch_operations(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Execute multiple database operations in a transaction."""
            try:
                # Convert operations to tuple format
                db_operations = []
                for op in operations:
                    query = op["query"]
                    params = op.get("params", [])
                    db_operations.append((query, params))
                
                results = await self.db_manager.execute_transaction(db_operations)
                
                return {
                    "type": "batch",
                    "results": results,
                    "operation_count": len(results)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Batch operation failed: {str(e)}"
                )
        
        @self.tool("get_table_schema")
        async def get_table_schema(table_name: str) -> Dict[str, Any]:
            """Get schema information for a table."""
            try:
                # PostgreSQL specific query
                query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = $1
                ORDER BY ordinal_position
                """
                
                columns = await self.db_manager.execute_query(query, table_name)
                
                return {
                    "table": table_name,
                    "columns": columns,
                    "column_count": len(columns)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get table schema: {str(e)}"
                )
```

### API Integration Tools

```python
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration."""
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class APIClient:
    """Generic API client."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GET request."""
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise e
    
    async def post(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make POST request."""
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(url, json=data) as response:
                    if response.status in [200, 201]:
                        return await response.json()
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise e

class APIMCPServer(Server):
    def __init__(self, name: str, api_configs: Dict[str, APIConfig]):
        super().__init__(name)
        self.api_configs = api_configs
        self._register_api_tools()
    
    def _register_api_tools(self):
        """Register API integration tools."""
        
        @self.tool("api_request")
        async def api_request(service: str, method: str, endpoint: str, data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
            """Make a request to an external API."""
            if service not in self.api_configs:
                raise ValueError(f"Unknown service: {service}")
            
            config = self.api_configs[service]
            
            try:
                async with APIClient(config) as client:
                    if method.upper() == "GET":
                        result = await client.get(endpoint, params)
                    elif method.upper() == "POST":
                        result = await client.post(endpoint, data)
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    
                    return {
                        "service": service,
                        "method": method,
                        "endpoint": endpoint,
                        "result": result,
                        "status": "success"
                    }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"API request failed: {str(e)}"
                )
        
        @self.tool("batch_api_requests")
        async def batch_api_requests(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Make multiple API requests concurrently."""
            tasks = []
            
            for req in requests:
                task = self.api_request(
                    req["service"],
                    req["method"],
                    req["endpoint"],
                    req.get("data"),
                    req.get("params")
                )
                tasks.append(task)
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append({
                            "request_index": i,
                            "status": "error",
                            "error": str(result)
                        })
                    else:
                        processed_results.append({
                            "request_index": i,
                            "status": "success",
                            "result": result
                        })
                
                return processed_results
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Batch API requests failed: {str(e)}"
                )
```

## Configuration Management

### Environment-based Configuration

```python
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json

@dataclass
class ServerConfig:
    """MCP server configuration."""
    name: str
    version: str
    debug: bool = False
    log_level: str = "INFO"
    database: Optional[Dict[str, Any]] = None
    apis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """Configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[ServerConfig] = None
    
    def load_config(self) -> ServerConfig:
        """Load configuration from file and environment."""
        # Load from file if exists
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
        else:
            file_config = {}
        
        # Override with environment variables
        env_config = self._load_from_env()
        
        # Merge configurations
        merged_config = self._merge_configs(file_config, env_config)
        
        # Create ServerConfig object
        self.config = ServerConfig(**merged_config)
        return self.config
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Server settings
        if os.getenv("MCP_SERVER_NAME"):
            env_config["name"] = os.getenv("MCP_SERVER_NAME")
        
        if os.getenv("MCP_SERVER_VERSION"):
            env_config["version"] = os.getenv("MCP_SERVER_VERSION")
        
        if os.getenv("MCP_DEBUG"):
            env_config["debug"] = os.getenv("MCP_DEBUG").lower() == "true"
        
        if os.getenv("MCP_LOG_LEVEL"):
            env_config["log_level"] = os.getenv("MCP_LOG_LEVEL")
        
        # Database settings
        if os.getenv("DATABASE_URL"):
            env_config["database"] = {"url": os.getenv("DATABASE_URL")}
        
        # API settings
        api_keys = {}
        for key, value in os.environ.items():
            if key.startswith("API_KEY_"):
                service_name = key.replace("API_KEY_", "").lower()
                api_keys[service_name] = {"api_key": value}
        
        if api_keys:
            env_config["apis"] = api_keys
        
        # Security settings
        if os.getenv("JWT_SECRET"):
            env_config["security"] = {"jwt_secret": os.getenv("JWT_SECRET")}
        
        return env_config
    
    def _merge_configs(self, file_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge file and environment configurations."""
        merged = file_config.copy()
        
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(merged, env_config)
        return merged
    
    def save_config(self, config: ServerConfig, path: str):
        """Save configuration to file."""
        config_dict = {
            "name": config.name,
            "version": config.version,
            "debug": config.debug,
            "log_level": config.log_level,
            "database": config.database,
            "apis": config.apis,
            "security": config.security,
            "performance": config.performance
        }
        
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
    
    def validate_config(self, config: ServerConfig) -> List[str]:
        """Validate configuration."""
        errors = []
        
        if not config.name:
            errors.append("Server name is required")
        
        if not config.version:
            errors.append("Server version is required")
        
        if config.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("Invalid log level")
        
        if config.database and not config.database.get("url"):
            errors.append("Database URL is required if database is configured")
        
        return errors

# Usage in MCP server
class ConfigurableMCPServer(Server):
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Validate configuration
        errors = self.config_manager.validate_config(self.config)
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        super().__init__(self.config.name)
        
        # Initialize components based on configuration
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize server components based on configuration."""
        # Initialize database if configured
        if self.config.database:
            db_config = DatabaseConfig(**self.config.database)
            self.db_manager = DatabaseManager(db_config)
        
        # Initialize API clients if configured
        if self.config.apis:
            api_configs = {}
            for service, api_config in self.config.apis.items():
                api_configs[service] = APIConfig(**api_config)
            self.api_clients = api_configs
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized {self.config.name} v{self.config.version}")
```

## Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile for MCP server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["python", "server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MCP_SERVER_NAME=production-server
      - MCP_LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://user:password@db:5432/mcpdb
      - API_KEY_WEATHER=your-weather-api-key
    depends_on:
      - db
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=mcpdb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  labels:
    app: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: your-registry/mcp-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: MCP_SERVER_NAME
          value: "k8s-server"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: database-url
        - name: API_KEY_WEATHER
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: weather-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: mcp-secrets
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  weather-api-key: <base64-encoded-api-key>
```

## Testing Strategies

### Unit Testing

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from mcp.types import ToolExecutionError, ErrorCode

class TestMCPServer:
    """Test suite for MCP server."""
    
    @pytest.fixture
    async def server(self):
        """Create test server instance."""
        config = ServerConfig(
            name="test-server",
            version="1.0.0",
            debug=True
        )
        server = ConfigurableMCPServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.fixture
    def mock_database(self):
        """Mock database manager."""
        mock_db = Mock()
        mock_db.execute_query = AsyncMock(return_value=[{"id": 1, "name": "Test"}])
        mock_db.execute_command = AsyncMock(return_value="INSERT 1")
        return mock_db
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, server, mock_database):
        """Test successful user creation."""
        server.db_manager = mock_database
        
        result = await server.create_user("John Doe", "john@example.com")
        
        assert result["status"] == "created"
        assert result["user"]["name"] == "John Doe"
        assert result["user"]["email"] == "john@example.com"
        mock_database.execute_command.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_user_validation_error(self, server):
        """Test user creation with validation error."""
        with pytest.raises(ToolExecutionError) as exc_info:
            await server.create_user("", "invalid-email")
        
        assert exc_info.value.code == ErrorCode.INVALID_PARAMS
    
    @pytest.mark.asyncio
    async def test_database_connection_error(self, server):
        """Test database connection error handling."""
        mock_db = Mock()
        mock_db.execute_command = AsyncMock(side_effect=Exception("Connection failed"))
        server.db_manager = mock_db
        
        with pytest.raises(ToolExecutionError) as exc_info:
            await server.create_user("John Doe", "john@example.com")
        
        assert exc_info.value.code == ErrorCode.INTERNAL_ERROR
    
    @pytest.mark.asyncio
    async def test_api_request_success(self, server):
        """Test successful API request."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "success"})
            
            mock_session.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await server.api_request("test-service", "GET", "/test")
            
            assert result["status"] == "success"
            assert result["service"] == "test-service"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, server):
        """Test concurrent request handling."""
        tasks = []
        for i in range(10):
            task = server.create_user(f"User {i}", f"user{i}@example.com")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for result in results:
            assert result["status"] == "created"

class TestConfigurationManager:
    """Test suite for configuration manager."""
    
    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test-server
version: 1.0.0
debug: true
log_level: DEBUG
""")
        
        config_manager = ConfigManager(str(config_file))
        config = config_manager.load_config()
        
        assert config.name == "test-server"
        assert config.version == "1.0.0"
        assert config.debug is True
        assert config.log_level == "DEBUG"
    
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'MCP_SERVER_NAME': 'env-server',
            'MCP_SERVER_VERSION': '2.0.0',
            'MCP_DEBUG': 'true',
            'MCP_LOG_LEVEL': 'WARNING'
        }):
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            assert config.name == "env-server"
            assert config.version == "2.0.0"
            assert config.debug is True
            assert config.log_level == "WARNING"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Test valid config
        valid_config = ServerConfig(name="test", version="1.0.0")
        errors = config_manager.validate_config(valid_config)
        assert len(errors) == 0
        
        # Test invalid config
        invalid_config = ServerConfig(name="", version="", log_level="INVALID")
        errors = config_manager.validate_config(invalid_config)
        assert len(errors) > 0
```

### Integration Testing

```python
import pytest
import aiohttp
import asyncio
from mcp.client import MCPClient

class TestMCPIntegration:
    """Integration tests for MCP server."""
    
    @pytest.fixture
    async def server_process(self):
        """Start MCP server process."""
        import subprocess
        import time
        
        process = subprocess.Popen([
            "python", "server.py",
            "--config", "test_config.yaml"
        ])
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        yield process
        
        process.terminate()
        process.wait()
    
    @pytest.fixture
    async def mcp_client(self):
        """Create MCP client for testing."""
        client = MCPClient("http://localhost:8000")
        await client.connect()
        yield client
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_server_health_check(self, server_process):
        """Test server health endpoint."""
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                assert response.status == 200
                data = await response.json()
                assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_tool_discovery(self, mcp_client):
        """Test tool discovery."""
        tools = await mcp_client.list_tools()
        
        assert len(tools) > 0
        tool_names = [tool["name"] for tool in tools]
        assert "create_user" in tool_names
        assert "get_user" in tool_names
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, mcp_client):
        """Test tool execution."""
        result = await mcp_client.call_tool("create_user", {
            "name": "Integration Test User",
            "email": "integration@example.com"
        })
        
        assert result["status"] == "created"
        assert result["user"]["name"] == "Integration Test User"
    
    @pytest.mark.asyncio
    async def test_resource_access(self, mcp_client):
        """Test resource access."""
        resources = await mcp_client.list_resources()
        
        assert len(resources) > 0
        
        # Test accessing a specific resource
        resource = await mcp_client.read_resource(resources[0]["uri"])
        assert resource is not None
```

## Monitoring and Maintenance

### Health Monitoring

```python
import time
import psutil
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class HealthStatus:
    """Health status information."""
    status: str
    timestamp: float
    details: Dict[str, Any]

class HealthMonitor:
    """Health monitoring system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    def record_request(self):
        """Record a request."""
        self.request_count += 1
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
    
    async def check_health(self) -> HealthStatus:
        """Check overall system health."""
        health_details = {
            "uptime": time.time() - self.start_time,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(),
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        # Determine overall status
        if health_details["error_rate"] > 0.1:  # More than 10% error rate
            status = "unhealthy"
        elif health_details["memory_usage"] > 90:  # More than 90% memory usage
            status = "unhealthy"
        elif health_details["cpu_usage"] > 95:  # More than 95% CPU usage
            status = "unhealthy"
        else:
            status = "healthy"
        
        return HealthStatus(
            status=status,
            timestamp=time.time(),
            details=health_details
        )
    
    async def check_database_health(self, db_manager) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Simple query to test database connection
            await db_manager.execute_query("SELECT 1")
            return {"status": "healthy", "response_time": "< 100ms"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_api_health(self, api_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Check external API health."""
        api_status = {}
        
        for service_name, client in api_clients.items():
            try:
                # Test API connectivity
                async with client as api_client:
                    await api_client.get("/health")
                api_status[service_name] = {"status": "healthy"}
            except Exception as e:
                api_status[service_name] = {"status": "unhealthy", "error": str(e)}
        
        return api_status

class MonitoringMCPServer(Server):
    def __init__(self, name: str):
        super().__init__(name)
        self.health_monitor = HealthMonitor()
        self._register_monitoring_tools()
    
    def _register_monitoring_tools(self):
        """Register monitoring tools."""
        
        @self.tool("health_check")
        async def health_check() -> Dict[str, Any]:
            """Get system health status."""
            health_status = await self.health_monitor.check_health()
            
            return {
                "status": health_status.status,
                "timestamp": health_status.timestamp,
                "details": health_status.details
            }
        
        @self.tool("get_metrics")
        async def get_metrics() -> Dict[str, Any]:
            """Get system metrics."""
            return {
                "uptime": time.time() - self.health_monitor.start_time,
                "request_count": self.health_monitor.request_count,
                "error_count": self.health_monitor.error_count,
                "error_rate": self.health_monitor.error_count / max(self.health_monitor.request_count, 1),
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage('/').percent
            }
        
        @self.tool("check_dependencies")
        async def check_dependencies() -> Dict[str, Any]:
            """Check health of external dependencies."""
            dependencies = {}
            
            # Check database
            if hasattr(self, 'db_manager'):
                dependencies["database"] = await self.health_monitor.check_database_health(self.db_manager)
            
            # Check APIs
            if hasattr(self, 'api_clients'):
                dependencies["apis"] = await self.health_monitor.check_api_health(self.api_clients)
            
            return dependencies
```

## Exercises

### Exercise 1: Production-Ready Server

Build a production-ready MCP server with the following features:

**Requirements:**
- Layered architecture with proper separation of concerns
- Comprehensive configuration management
- Database integration with connection pooling
- API integration with retry logic and circuit breakers
- Health monitoring and metrics collection
- Comprehensive error handling and logging

### Exercise 2: Microservices Integration

Create an MCP server that integrates with multiple microservices:

**Requirements:**
- Service registry and discovery
- Load balancing across service instances
- Circuit breaker pattern for fault tolerance
- Distributed tracing and monitoring
- Configuration management for multiple services

### Exercise 3: Deployment Pipeline

Set up a complete deployment pipeline:

**Requirements:**
- Docker containerization
- Kubernetes deployment manifests
- CI/CD pipeline configuration
- Environment-specific configurations
- Automated testing and deployment
- Monitoring and alerting setup

## 🎯 Module Summary

In this module, you learned:

- ✅ Production-ready server architecture patterns
- ✅ Complex tool implementation strategies
- ✅ Comprehensive configuration management
- ✅ Deployment strategies for various environments
- ✅ Testing methodologies and best practices
- ✅ Monitoring and maintenance systems

## 🚀 Next Steps

You're now ready to move on to **Module 5: Client Integration**, where you'll learn about:
- MCP client development patterns
- AI application integration strategies
- Performance optimization techniques
- Error handling and recovery

---

**Congratulations on completing Module 4! 🎉**

*Next: [Module 5: Client Integration](module-05-client-integration/README.md)*
