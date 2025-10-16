# Module 8: Production and Deployment

Welcome to Module 8! This is the final module where you'll learn how to take your MCP servers from development to production. This module covers advanced server architecture, deployment strategies, monitoring, and maintenance for production environments.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Design production-ready MCP server architectures
- Implement advanced server patterns and middleware
- Deploy MCP servers using Docker and Kubernetes
- Set up monitoring and logging systems
- Implement security best practices
- Handle production scaling and performance

## 📚 Topics Covered

1. [Production Architecture Patterns](#production-architecture-patterns)
2. [Advanced Server Features](#advanced-server-features)
3. [Deployment Strategies](#deployment-strategies)
4. [Monitoring and Logging](#monitoring-and-logging)
5. [Security Best Practices](#security-best-practices)
6. [Performance Optimization](#performance-optimization)
7. [Exercises](#exercises)

---

## Production Architecture Patterns

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

## Advanced Server Features

### Middleware System

```python
from typing import Callable, Dict, Any
import time
import asyncio

class AdvancedMCPServer:
    """MCP server with middleware support."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.middleware = []
        self.metrics = {
            "request_count": 0,
            "error_count": 0,
            "total_time": 0
        }
        self.cache = {}
        self.rate_limits = {}
        self._register_tools()
    
    def add_middleware(self, func: Callable):
        """Add middleware function."""
        self.middleware.append(func)
        return func
    
    def tool(self, name: str, rate_limit: int = None):
        """Decorator for registering tools with rate limiting."""
        def decorator(func):
            if rate_limit:
                self.rate_limits[name] = rate_limit
            
            async def wrapper(*args, **kwargs):
                # Execute middleware
                for middleware in self.middleware:
                    result = await middleware(name, kwargs)
                    if result.get("block"):
                        return result.get("response")
                
                # Execute tool
                return await func(*args, **kwargs)
            
            setattr(self, f"tool_{name}", wrapper)
            return wrapper
        return decorator
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any], user_id: str = None) -> Any:
        """Call a tool with middleware and metrics."""
        start_time = time.time()
        self.metrics["request_count"] += 1
        
        try:
            # Rate limiting
            if tool_name in self.rate_limits:
                if not self._check_rate_limit(tool_name, user_id):
                    raise ToolExecutionError(
                        code=ErrorCode.RATE_LIMITED,
                        message="Rate limit exceeded"
                    )
            
            # Execute tool
            tool_func = getattr(self, f"tool_{tool_name}", None)
            if not tool_func:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Tool {tool_name} not found"
                )
            
            result = await tool_func(**params)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_time"] += execution_time
            
            return result
            
        except Exception as e:
            self.metrics["error_count"] += 1
            raise e
    
    def _check_rate_limit(self, tool_name: str, user_id: str) -> bool:
        """Check if user has exceeded rate limit for tool."""
        # Simple rate limiting implementation
        key = f"{tool_name}:{user_id or 'anonymous'}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old requests (older than 1 minute)
        self.rate_limits[key] = [
            req_time for req_time in self.rate_limits[key] 
            if now - req_time < 60
        ]
        
        # Check if limit exceeded
        limit = self.rate_limits.get(tool_name, 60)  # Default 60 requests per minute
        if len(self.rate_limits[key]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[key].append(now)
        return True
    
    def _register_tools(self):
        """Register server tools."""
        
        @self.tool("get_metrics")
        async def get_metrics() -> Dict[str, Any]:
            """Get server metrics."""
            avg_time = (
                self.metrics["total_time"] / max(self.metrics["request_count"], 1)
            )
            
            return {
                "request_count": self.metrics["request_count"],
                "error_count": self.metrics["error_count"],
                "error_rate": self.metrics["error_count"] / max(self.metrics["request_count"], 1),
                "average_response_time": round(avg_time, 3),
                "uptime": time.time() - self.metrics.get("start_time", time.time())
            }

# Middleware examples
@server.add_middleware
async def auth_middleware(tool_name: str, params: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """Authentication middleware."""
    if tool_name in ["admin_tool", "sensitive_tool"]:
        if not user_id:
            return {
                "block": True,
                "response": ToolExecutionError(
                    code=ErrorCode.UNAUTHORIZED,
                    message="Authentication required"
                )
            }
    return {"block": False}

@server.add_middleware
async def logging_middleware(tool_name: str, params: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """Logging middleware."""
    logger.info(f"Tool call: {tool_name} by user: {user_id or 'anonymous'}")
    return {"block": False}

@server.add_middleware
async def caching_middleware(tool_name: str, params: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """Caching middleware."""
    cache_key = f"{tool_name}:{hash(str(params))}"
    
    if cache_key in server.cache:
        logger.info(f"Cache hit for {tool_name}")
        return {
            "block": True,
            "response": server.cache[cache_key]
        }
    
    return {"block": False}
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

## Monitoring and Logging

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

## Security Best Practices

### Authentication and Authorization

```python
import jwt
from datetime import datetime, timedelta
from typing import Optional

class SecureMCPServer:
    """MCP server with security features."""
    
    def __init__(self, name: str, secret_key: str):
        self.name = name
        self.secret_key = secret_key
        self.server = Server(name)
        self.user_sessions = {}
        self._register_tools()
    
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
    
    def _register_tools(self):
        """Register secure tools."""
        
        @self.tool("login")
        async def login(username: str, password: str) -> Dict[str, Any]:
            """Authenticate user and return token."""
            # Simple authentication (in production, use proper password hashing)
            if username == "admin" and password == "admin123":
                token = self.generate_token(username)
                self.user_sessions[username] = {
                    "token": token,
                    "login_time": datetime.utcnow(),
                    "permissions": ["read", "write", "admin"]
                }
                
                return {
                    "status": "success",
                    "token": token,
                    "user": username,
                    "permissions": ["read", "write", "admin"]
                }
            else:
                raise AuthenticationError(
                    code=ErrorCode.UNAUTHORIZED,
                    message="Invalid credentials"
                )
        
        @self.tool("admin_action")
        async def admin_action(action: str, token: str) -> Dict[str, Any]:
            """Perform admin action (requires authentication)."""
            user_id = await self.authenticate_token(token)
            
            if user_id not in self.user_sessions:
                raise AuthenticationError(
                    code=ErrorCode.UNAUTHORIZED,
                    message="Session not found"
                )
            
            session = self.user_sessions[user_id]
            if "admin" not in session["permissions"]:
                raise AuthenticationError(
                    code=ErrorCode.FORBIDDEN,
                    message="Admin permission required"
                )
            
            return {
                "status": "success",
                "action": action,
                "user": user_id,
                "message": f"Admin action '{action}' completed"
            }
```

## Performance Optimization

### Caching and Connection Pooling

```python
import redis
import asyncio
from typing import Dict, Any, Optional
import json

class OptimizedMCPServer:
    """MCP server with performance optimizations."""
    
    def __init__(self, name: str, redis_url: str = "redis://localhost:6379"):
        self.name = name
        self.server = Server(name)
        self.redis_client = redis.from_url(redis_url)
        self.connection_pool = None
        self._register_tools()
    
    async def get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None
    
    async def set_cached_data(self, key: str, data: Any, ttl: int = 300):
        """Set data in cache."""
        try:
            self.redis_client.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _register_tools(self):
        """Register optimized tools."""
        
        @self.tool("get_data_cached")
        async def get_data_cached(data_id: str) -> Dict[str, Any]:
            """Get data with caching."""
            cache_key = f"data:{data_id}"
            
            # Try cache first
            cached_data = await self.get_cached_data(cache_key)
            if cached_data:
                return {
                    "status": "success",
                    "data": cached_data,
                    "source": "cache"
                }
            
            # Fetch from source (simulated)
            data = {
                "id": data_id,
                "content": f"Data for {data_id}",
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            await self.set_cached_data(cache_key, data, ttl=300)
            
            return {
                "status": "success",
                "data": data,
                "source": "database"
            }
        
        @self.tool("batch_operation")
        async def batch_operation(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Perform batch operations efficiently."""
            results = []
            
            # Process operations concurrently
            tasks = []
            for op in operations:
                task = self._process_operation(op)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "status": "success",
                "results": results,
                "count": len(results)
            }
    
    async def _process_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single operation."""
        # Simulate operation processing
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "operation": operation["type"],
            "result": f"Processed {operation['type']}",
            "success": True
        }
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
- ✅ Advanced server features and middleware
- ✅ Deployment strategies for various environments
- ✅ Monitoring and logging systems
- ✅ Security best practices
- ✅ Performance optimization techniques

## 🚀 Course Completion

**Congratulations! You have completed the MCP Course! 🎉**

You now have the knowledge and skills to:
- Build MCP servers from basic to production-ready
- Integrate MCP with AI applications
- Deploy and maintain MCP systems in production
- Apply best practices for security and performance

## 📚 Next Steps

Continue your MCP journey by:
- Building your own MCP tools and servers
- Contributing to the MCP ecosystem
- Exploring advanced use cases
- Sharing your knowledge with the community

---

**Thank you for completing the MCP Course! 🚀**

*Check out the [examples](../examples/) directory for reference implementations and inspiration for your own projects.*
