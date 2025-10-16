# Business Tools MCP Server Configuration

## Environment Variables

Set the following environment variables for production deployment:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/business_db
REDIS_URL=redis://localhost:6379

# API Keys
PAYMENT_API_KEY=your_payment_api_key_here
SHIPPING_API_KEY=your_shipping_api_key_here

# Server Configuration
MCP_SERVER_NAME=business-tools
MCP_SERVER_PORT=8000
MCP_SERVER_HOST=0.0.0.0

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here
API_RATE_LIMIT=1000

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/business-tools-mcp.log
```

## Docker Configuration

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "business_tools_server.py"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  business-tools-mcp:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/business_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/var/log

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=business_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Production Deployment

### Systemd Service
```ini
[Unit]
Description=Business Tools MCP Server
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/business-tools-mcp
ExecStart=/opt/business-tools-mcp/venv/bin/python business_tools_server.py
Restart=always
RestartSec=10
Environment=DATABASE_URL=postgresql://user:password@localhost/business_db
Environment=REDIS_URL=redis://localhost:6379

[Install]
WantedBy=multi-user.target
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring and Health Checks

### Health Check Endpoint
```python
@server.tool("health_check")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring."""
    try:
        # Check database connection
        db_status = "healthy"  # In production, check actual DB connection
        
        # Check Redis connection
        redis_status = "healthy"  # In production, check actual Redis connection
        
        # Check server metrics
        metrics = server.get_metrics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "redis": redis_status,
            "metrics": {
                "uptime": time.time() - server.start_time,
                "requests_total": metrics["requests_total"],
                "success_rate": metrics["success_rate"]
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('mcp_requests_total', 'Total MCP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('mcp_active_connections', 'Active connections')
```

## Security Best Practices

### Authentication Middleware
```python
@server.add_middleware
async def auth_middleware(tool_name: str, params: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """Enhanced authentication middleware."""
    if not user_id:
        raise Exception("Authentication required")
    
    # Validate JWT token
    # Check user permissions
    # Log access attempt
    
    return params
```

### Input Validation
```python
def validate_input(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate input data against schema."""
    # Implement JSON schema validation
    # Sanitize input data
    # Check for SQL injection attempts
    pass
```

## Performance Optimization

### Caching Strategy
```python
import redis
from functools import wraps

def cache_result(ttl: int = 300):
    """Cache decorator for expensive operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await redis_client.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator
```

### Database Connection Pooling
```python
import asyncpg

class DatabasePool:
    def __init__(self, database_url: str, min_connections: int = 10, max_connections: int = 100):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool = None
    
    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.min_connections,
            max_size=self.max_connections
        )
    
    async def execute_query(self, query: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
```

## Testing

### Unit Tests
```python
import pytest
from business_tools_server import server

@pytest.mark.asyncio
async def test_create_customer():
    result = await server.call_tool("create_customer", {
        "name": "Test Customer",
        "email": "test@example.com",
        "company": "Test Corp"
    })
    
    assert result["status"] == "success"
    assert result["customer"]["name"] == "Test Customer"
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_order_workflow():
    # Create customer
    customer_result = await server.call_tool("create_customer", {...})
    
    # Create order
    order_result = await server.call_tool("create_order", {...})
    
    # Process payment
    payment_result = await server.call_tool("process_payment", {...})
    
    assert payment_result["status"] == "success"
```

## Backup and Recovery

### Database Backup Script
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="business_db"

pg_dump $DATABASE_URL > $BACKUP_DIR/backup_$DATE.sql
gzip $BACKUP_DIR/backup_$DATE.sql

# Keep only last 30 days of backups
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

### Recovery Script
```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

gunzip -c $BACKUP_FILE | psql $DATABASE_URL
```
