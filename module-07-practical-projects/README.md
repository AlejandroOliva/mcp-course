# Module 7: Practical Projects

Welcome to Module 7! This is the final module where you'll apply everything you've learned to build complete, production-ready MCP applications. This module focuses on hands-on projects that demonstrate real-world MCP implementations, advanced patterns, and best practices.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Build complete MCP applications from scratch
- Apply advanced MCP patterns and architectures
- Implement performance optimization techniques
- Deploy MCP applications to production
- Apply best practices and lessons learned
- Contribute to the MCP ecosystem

## 📚 Topics Covered

1. [Project 1: AI-Powered Business Assistant](#project-1-ai-powered-business-assistant)
2. [Project 2: Data Integration Platform](#project-2-data-integration-platform)
3. [Project 3: Workflow Automation System](#project-3-workflow-automation-system)
4. [Advanced Patterns and Optimizations](#advanced-patterns-and-optimizations)
5. [Production Deployment](#production-deployment)
6. [Best Practices and Lessons Learned](#best-practices-and-lessons-learned)
7. [Final Project](#final-project)

---

## Project 1: AI-Powered Business Assistant

### Project Overview

Build a comprehensive AI-powered business assistant that can handle customer inquiries, process orders, manage inventory, and provide analytics through natural language interactions.

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Client     │◄──►│   MCP Server    │◄──►│   External APIs │
│   (Claude, etc.)│    │   (Business     │    │   (Payment,     │
│                 │    │    Assistant)   │    │    Shipping,    │
└─────────────────┘    └─────────────────┘    │    Inventory)   │
                                │               └─────────────────┘
                                ▼
                       ┌─────────────────┐
                       │   Database      │
                       │   (PostgreSQL)   │
                       └─────────────────┘
```

### Implementation

```python
import asyncio
import asyncpg
import aiohttp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from mcp.server import Server
from mcp.types import ToolExecutionError, ErrorCode

@dataclass
class Customer:
    id: Optional[int] = None
    name: str = ""
    email: str = ""
    phone: Optional[str] = None
    company: Optional[str] = None
    created_at: Optional[datetime] = None

@dataclass
class Product:
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    price: float = 0.0
    stock: int = 0
    category: str = ""

@dataclass
class Order:
    id: Optional[int] = None
    customer_id: int = 0
    items: List[Dict[str, Any]] = None
    total: float = 0.0
    status: str = "pending"
    created_at: Optional[datetime] = None

class BusinessAssistantMCPServer(Server):
    """AI-Powered Business Assistant MCP Server."""
    
    def __init__(self, name: str, database_url: str):
        super().__init__(name)
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Register all tool categories
        self._register_customer_tools()
        self._register_product_tools()
        self._register_order_tools()
        self._register_analytics_tools()
        self._register_ai_assistant_tools()
    
    async def initialize(self):
        """Initialize database and HTTP session."""
        # Initialize database pool
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20
        )
        
        # Initialize HTTP session for external APIs
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Create database tables
        await self._create_tables()
        
        self.logger.info("Business Assistant MCP Server initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.pool:
            await self.pool.close()
        if self.session:
            await self.session.close()
    
    async def _create_tables(self):
        """Create database tables."""
        async with self.pool.acquire() as conn:
            # Customers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    phone VARCHAR(20),
                    company VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Products table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    price DECIMAL(10,2) NOT NULL,
                    stock INTEGER DEFAULT 0,
                    category VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Orders table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(id),
                    items JSONB NOT NULL,
                    total DECIMAL(10,2) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _register_customer_tools(self):
        """Register customer management tools."""
        
        @self.tool("create_customer")
        async def create_customer(name: str, email: str, phone: str = None, company: str = None) -> Dict[str, Any]:
            """Create a new customer."""
            try:
                async with self.pool.acquire() as conn:
                    customer_id = await conn.fetchval("""
                        INSERT INTO customers (name, email, phone, company)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id
                    """, name, email, phone, company)
                    
                    return {
                        "status": "success",
                        "customer_id": customer_id,
                        "message": f"Customer {name} created successfully"
                    }
            except asyncpg.UniqueViolationError:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message=f"Customer with email {email} already exists"
                )
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to create customer: {str(e)}"
                )
        
        @self.tool("get_customer")
        async def get_customer(customer_id: int = None, email: str = None) -> Dict[str, Any]:
            """Get customer information."""
            if not customer_id and not email:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message="Either customer_id or email must be provided"
                )
            
            try:
                async with self.pool.acquire() as conn:
                    if customer_id:
                        row = await conn.fetchrow("SELECT * FROM customers WHERE id = $1", customer_id)
                    else:
                        row = await conn.fetchrow("SELECT * FROM customers WHERE email = $1", email)
                    
                    if not row:
                        raise ToolExecutionError(
                            code=ErrorCode.NOT_FOUND,
                            message="Customer not found"
                        )
                    
                    return {
                        "status": "success",
                        "customer": dict(row)
                    }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get customer: {str(e)}"
                )
        
        @self.tool("search_customers")
        async def search_customers(query: str, limit: int = 20) -> Dict[str, Any]:
            """Search customers by name, email, or company."""
            try:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT * FROM customers 
                        WHERE name ILIKE $1 OR email ILIKE $1 OR company ILIKE $1
                        ORDER BY name
                        LIMIT $2
                    """, f"%{query}%", limit)
                    
                    return {
                        "status": "success",
                        "customers": [dict(row) for row in rows],
                        "total_found": len(rows)
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Customer search failed: {str(e)}"
                )
    
    def _register_product_tools(self):
        """Register product management tools."""
        
        @self.tool("create_product")
        async def create_product(name: str, description: str, price: float, stock: int, category: str) -> Dict[str, Any]:
            """Create a new product."""
            try:
                async with self.pool.acquire() as conn:
                    product_id = await conn.fetchval("""
                        INSERT INTO products (name, description, price, stock, category)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id
                    """, name, description, price, stock, category)
                    
                    return {
                        "status": "success",
                        "product_id": product_id,
                        "message": f"Product {name} created successfully"
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to create product: {str(e)}"
                )
        
        @self.tool("get_product")
        async def get_product(product_id: int) -> Dict[str, Any]:
            """Get product information."""
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("SELECT * FROM products WHERE id = $1", product_id)
                    
                    if not row:
                        raise ToolExecutionError(
                            code=ErrorCode.NOT_FOUND,
                            message="Product not found"
                        )
                    
                    return {
                        "status": "success",
                        "product": dict(row)
                    }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get product: {str(e)}"
                )
        
        @self.tool("search_products")
        async def search_products(query: str, category: str = None, min_price: float = None, max_price: float = None, limit: int = 20) -> Dict[str, Any]:
            """Search products with filters."""
            try:
                async with self.pool.acquire() as conn:
                    where_conditions = ["name ILIKE $1 OR description ILIKE $1"]
                    params = [f"%{query}%"]
                    param_count = 1
                    
                    if category:
                        param_count += 1
                        where_conditions.append(f"category = ${param_count}")
                        params.append(category)
                    
                    if min_price is not None:
                        param_count += 1
                        where_conditions.append(f"price >= ${param_count}")
                        params.append(min_price)
                    
                    if max_price is not None:
                        param_count += 1
                        where_conditions.append(f"price <= ${param_count}")
                        params.append(max_price)
                    
                    param_count += 1
                    params.append(limit)
                    
                    query_sql = f"""
                        SELECT * FROM products 
                        WHERE {' AND '.join(where_conditions)}
                        ORDER BY name
                        LIMIT ${param_count}
                    """
                    
                    rows = await conn.fetch(query_sql, *params)
                    
                    return {
                        "status": "success",
                        "products": [dict(row) for row in rows],
                        "total_found": len(rows)
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Product search failed: {str(e)}"
                )
        
        @self.tool("update_product_stock")
        async def update_product_stock(product_id: int, new_stock: int) -> Dict[str, Any]:
            """Update product stock."""
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        UPDATE products 
                        SET stock = $1
                        WHERE id = $2
                        RETURNING *
                    """, new_stock, product_id)
                    
                    if not row:
                        raise ToolExecutionError(
                            code=ErrorCode.NOT_FOUND,
                            message="Product not found"
                        )
                    
                    return {
                        "status": "success",
                        "product": dict(row),
                        "message": f"Stock updated to {new_stock}"
                    }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to update stock: {str(e)}"
                )
    
    def _register_order_tools(self):
        """Register order management tools."""
        
        @self.tool("create_order")
        async def create_order(customer_id: int, items: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Create a new order."""
            if not items:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message="Order must have at least one item"
                )
            
            try:
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        # Verify customer exists
                        customer = await conn.fetchrow("SELECT id FROM customers WHERE id = $1", customer_id)
                        if not customer:
                            raise ToolExecutionError(
                                code=ErrorCode.NOT_FOUND,
                                message="Customer not found"
                            )
                        
                        # Calculate total and verify stock
                        total = 0.0
                        for item in items:
                            product_id = item.get("product_id")
                            quantity = item.get("quantity", 1)
                            
                            product = await conn.fetchrow("SELECT price, stock FROM products WHERE id = $1", product_id)
                            if not product:
                                raise ToolExecutionError(
                                    code=ErrorCode.NOT_FOUND,
                                    message=f"Product {product_id} not found"
                                )
                            
                            if product["stock"] < quantity:
                                raise ToolExecutionError(
                                    code=ErrorCode.INVALID_PARAMS,
                                    message=f"Insufficient stock for product {product_id}"
                                )
                            
                            item_total = product["price"] * quantity
                            total += item_total
                            item["unit_price"] = product["price"]
                            item["total"] = item_total
                        
                        # Create order
                        order_id = await conn.fetchval("""
                            INSERT INTO orders (customer_id, items, total)
                            VALUES ($1, $2, $3)
                            RETURNING id
                        """, customer_id, json.dumps(items), total)
                        
                        # Update stock
                        for item in items:
                            await conn.execute("""
                                UPDATE products 
                                SET stock = stock - $1
                                WHERE id = $2
                            """, item["quantity"], item["product_id"])
                        
                        return {
                            "status": "success",
                            "order_id": order_id,
                            "total": total,
                            "message": f"Order {order_id} created successfully"
                        }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to create order: {str(e)}"
                )
        
        @self.tool("get_order")
        async def get_order(order_id: int) -> Dict[str, Any]:
            """Get order information."""
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT o.*, c.name as customer_name, c.email as customer_email
                        FROM orders o
                        JOIN customers c ON o.customer_id = c.id
                        WHERE o.id = $1
                    """, order_id)
                    
                    if not row:
                        raise ToolExecutionError(
                            code=ErrorCode.NOT_FOUND,
                            message="Order not found"
                        )
                    
                    order_data = dict(row)
                    order_data["items"] = json.loads(order_data["items"])
                    
                    return {
                        "status": "success",
                        "order": order_data
                    }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get order: {str(e)}"
                )
        
        @self.tool("update_order_status")
        async def update_order_status(order_id: int, status: str) -> Dict[str, Any]:
            """Update order status."""
            valid_statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
            if status not in valid_statuses:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
                )
            
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        UPDATE orders 
                        SET status = $1
                        WHERE id = $2
                        RETURNING *
                    """, status, order_id)
                    
                    if not row:
                        raise ToolExecutionError(
                            code=ErrorCode.NOT_FOUND,
                            message="Order not found"
                        )
                    
                    return {
                        "status": "success",
                        "order": dict(row),
                        "message": f"Order {order_id} status updated to {status}"
                    }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to update order status: {str(e)}"
                )
    
    def _register_analytics_tools(self):
        """Register analytics tools."""
        
        @self.tool("get_business_analytics")
        async def get_business_analytics() -> Dict[str, Any]:
            """Get comprehensive business analytics."""
            try:
                async with self.pool.acquire() as conn:
                    # Customer analytics
                    total_customers = await conn.fetchval("SELECT COUNT(*) FROM customers")
                    new_customers_this_month = await conn.fetchval("""
                        SELECT COUNT(*) FROM customers 
                        WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
                    """)
                    
                    # Product analytics
                    total_products = await conn.fetchval("SELECT COUNT(*) FROM products")
                    low_stock_products = await conn.fetchval("SELECT COUNT(*) FROM products WHERE stock < 10")
                    
                    # Order analytics
                    total_orders = await conn.fetchval("SELECT COUNT(*) FROM orders")
                    total_revenue = await conn.fetchval("SELECT SUM(total) FROM orders")
                    orders_this_month = await conn.fetchval("""
                        SELECT COUNT(*) FROM orders 
                        WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
                    """)
                    
                    # Top customers
                    top_customers = await conn.fetch("""
                        SELECT c.name, c.email, COUNT(o.id) as order_count, SUM(o.total) as total_spent
                        FROM customers c
                        LEFT JOIN orders o ON c.id = o.customer_id
                        GROUP BY c.id, c.name, c.email
                        ORDER BY total_spent DESC NULLS LAST
                        LIMIT 5
                    """)
                    
                    # Top products
                    top_products = await conn.fetch("""
                        SELECT p.name, p.category, SUM((item->>'quantity')::int) as total_quantity
                        FROM products p
                        JOIN orders o ON true
                        CROSS JOIN LATERAL jsonb_array_elements(o.items) as item
                        WHERE (item->>'product_id')::int = p.id
                        GROUP BY p.id, p.name, p.category
                        ORDER BY total_quantity DESC
                        LIMIT 5
                    """)
                    
                    return {
                        "status": "success",
                        "analytics": {
                            "customers": {
                                "total": total_customers,
                                "new_this_month": new_customers_this_month
                            },
                            "products": {
                                "total": total_products,
                                "low_stock": low_stock_products
                            },
                            "orders": {
                                "total": total_orders,
                                "this_month": orders_this_month,
                                "total_revenue": float(total_revenue or 0)
                            },
                            "top_customers": [dict(row) for row in top_customers],
                            "top_products": [dict(row) for row in top_products]
                        }
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get analytics: {str(e)}"
                )
    
    def _register_ai_assistant_tools(self):
        """Register AI assistant specific tools."""
        
        @self.tool("get_customer_summary")
        async def get_customer_summary(customer_id: int) -> Dict[str, Any]:
            """Get comprehensive customer summary for AI assistant."""
            try:
                async with self.pool.acquire() as conn:
                    # Get customer info
                    customer = await conn.fetchrow("SELECT * FROM customers WHERE id = $1", customer_id)
                    if not customer:
                        raise ToolExecutionError(
                            code=ErrorCode.NOT_FOUND,
                            message="Customer not found"
                        )
                    
                    # Get customer orders
                    orders = await conn.fetch("""
                        SELECT id, total, status, created_at
                        FROM orders
                        WHERE customer_id = $1
                        ORDER BY created_at DESC
                        LIMIT 10
                    """, customer_id)
                    
                    # Calculate customer stats
                    total_orders = await conn.fetchval("SELECT COUNT(*) FROM orders WHERE customer_id = $1", customer_id)
                    total_spent = await conn.fetchval("SELECT SUM(total) FROM orders WHERE customer_id = $1", customer_id)
                    last_order = await conn.fetchrow("""
                        SELECT created_at FROM orders 
                        WHERE customer_id = $1 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, customer_id)
                    
                    return {
                        "status": "success",
                        "customer": dict(customer),
                        "summary": {
                            "total_orders": total_orders,
                            "total_spent": float(total_spent or 0),
                            "last_order_date": last_order["created_at"].isoformat() if last_order else None,
                            "recent_orders": [dict(order) for order in orders]
                        }
                    }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get customer summary: {str(e)}"
                )
        
        @self.tool("suggest_products")
        async def suggest_products(customer_id: int = None, category: str = None, limit: int = 5) -> Dict[str, Any]:
            """Suggest products for AI assistant."""
            try:
                async with self.pool.acquire() as conn:
                    query = "SELECT * FROM products WHERE stock > 0"
                    params = []
                    
                    if category:
                        query += " AND category = $1"
                        params.append(category)
                    
                    query += " ORDER BY RANDOM() LIMIT $" + str(len(params) + 1)
                    params.append(limit)
                    
                    rows = await conn.fetch(query, *params)
                    
                    return {
                        "status": "success",
                        "suggestions": [dict(row) for row in rows],
                        "reasoning": f"Suggested {len(rows)} products based on availability and preferences"
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to suggest products: {str(e)}"
                )

# Usage example
async def main():
    """Main function to run the Business Assistant MCP Server."""
    server = BusinessAssistantMCPServer(
        name="business-assistant",
        database_url="postgresql://user:password@localhost/business_db"
    )
    
    await server.initialize()
    
    try:
        print("Business Assistant MCP Server started")
        print("Available tools:")
        print("- Customer management: create_customer, get_customer, search_customers")
        print("- Product management: create_product, get_product, search_products, update_product_stock")
        print("- Order management: create_order, get_order, update_order_status")
        print("- Analytics: get_business_analytics")
        print("- AI Assistant: get_customer_summary, suggest_products")
        
        # Keep server running
        await asyncio.sleep(3600)  # Run for 1 hour
        
    finally:
        await server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Project 2: Data Integration Platform

### Project Overview

Build a comprehensive data integration platform that can connect to multiple data sources, transform data, and provide unified access through MCP tools.

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │◄──►│   MCP Server    │◄──►│   AI Clients    │
│   (APIs, DBs,   │    │   (Data         │    │   (Claude, etc.)│
│    Files, etc.) │    │    Platform)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Data Cache    │
                       │   (Redis)       │
                       └─────────────────┘
```

### Implementation

```python
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json
import logging
from datetime import datetime, timedelta
from mcp.server import Server
from mcp.types import ToolExecutionError, ErrorCode

@dataclass
class DataSource:
    """Data source configuration."""
    id: str
    name: str
    type: str  # 'api', 'database', 'file', 'web'
    config: Dict[str, Any]
    last_sync: Optional[datetime] = None
    sync_interval: int = 3600  # seconds

@dataclass
class DataTransform:
    """Data transformation rule."""
    id: str
    name: str
    source_id: str
    target_schema: Dict[str, Any]
    transform_rules: List[Dict[str, Any]]

class DataIntegrationMCPServer(Server):
    """Data Integration Platform MCP Server."""
    
    def __init__(self, name: str, redis_url: str = "redis://localhost:6379"):
        super().__init__(name)
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.data_sources: Dict[str, DataSource] = {}
        self.transforms: Dict[str, DataTransform] = {}
        self.logger = logging.getLogger(__name__)
        
        # Register tools
        self._register_data_source_tools()
        self._register_data_transformation_tools()
        self._register_data_query_tools()
        self._register_sync_tools()
    
    async def initialize(self):
        """Initialize Redis and HTTP session."""
        # Initialize Redis
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Load existing configurations
        await self._load_configurations()
        
        self.logger.info("Data Integration MCP Server initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        if self.session:
            await self.session.close()
    
    async def _load_configurations(self):
        """Load data source and transform configurations from Redis."""
        try:
            # Load data sources
            source_keys = await self.redis_client.keys("data_source:*")
            for key in source_keys:
                source_data = await self.redis_client.get(key)
                if source_data:
                    source_dict = json.loads(source_data)
                    source = DataSource(**source_dict)
                    self.data_sources[source.id] = source
            
            # Load transforms
            transform_keys = await self.redis_client.keys("transform:*")
            for key in transform_keys:
                transform_data = await self.redis_client.get(key)
                if transform_data:
                    transform_dict = json.loads(transform_data)
                    transform = DataTransform(**transform_dict)
                    self.transforms[transform.id] = transform
            
            self.logger.info(f"Loaded {len(self.data_sources)} data sources and {len(self.transforms)} transforms")
        
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
    
    def _register_data_source_tools(self):
        """Register data source management tools."""
        
        @self.tool("add_data_source")
        async def add_data_source(source_id: str, name: str, source_type: str, config: Dict[str, Any], sync_interval: int = 3600) -> Dict[str, Any]:
            """Add a new data source."""
            try:
                # Validate source type
                valid_types = ['api', 'database', 'file', 'web']
                if source_type not in valid_types:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"Invalid source type. Must be one of: {', '.join(valid_types)}"
                    )
                
                # Create data source
                data_source = DataSource(
                    id=source_id,
                    name=name,
                    type=source_type,
                    config=config,
                    sync_interval=sync_interval
                )
                
                # Save to Redis
                await self.redis_client.set(
                    f"data_source:{source_id}",
                    json.dumps(asdict(data_source), default=str)
                )
                
                self.data_sources[source_id] = data_source
                
                return {
                    "status": "success",
                    "source_id": source_id,
                    "message": f"Data source '{name}' added successfully"
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to add data source: {str(e)}"
                )
        
        @self.tool("list_data_sources")
        async def list_data_sources() -> Dict[str, Any]:
            """List all data sources."""
            sources = []
            for source in self.data_sources.values():
                sources.append({
                    "id": source.id,
                    "name": source.name,
                    "type": source.type,
                    "last_sync": source.last_sync.isoformat() if source.last_sync else None,
                    "sync_interval": source.sync_interval
                })
            
            return {
                "status": "success",
                "sources": sources,
                "total_count": len(sources)
            }
        
        @self.tool("test_data_source")
        async def test_data_source(source_id: str) -> Dict[str, Any]:
            """Test connection to a data source."""
            if source_id not in self.data_sources:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Data source not found: {source_id}"
                )
            
            source = self.data_sources[source_id]
            
            try:
                if source.type == 'api':
                    # Test API connection
                    url = source.config.get('url')
                    headers = source.config.get('headers', {})
                    
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                "status": "success",
                                "source_id": source_id,
                                "connection_status": "success",
                                "sample_data": data[:5] if isinstance(data, list) else data,
                                "message": "API connection successful"
                            }
                        else:
                            raise Exception(f"HTTP {response.status}")
                
                elif source.type == 'database':
                    # Test database connection
                    connection_string = source.config.get('connection_string')
                    async with asyncpg.connect(connection_string) as conn:
                        result = await conn.fetchval("SELECT 1")
                        return {
                            "status": "success",
                            "source_id": source_id,
                            "connection_status": "success",
                            "message": "Database connection successful"
                        }
                
                elif source.type == 'file':
                    # Test file access
                    file_path = source.config.get('file_path')
                    import os
                    if os.path.exists(file_path):
                        return {
                            "status": "success",
                            "source_id": source_id,
                            "connection_status": "success",
                            "file_size": os.path.getsize(file_path),
                            "message": "File access successful"
                        }
                    else:
                        raise Exception("File not found")
                
                else:
                    raise Exception(f"Unsupported source type: {source.type}")
            
            except Exception as e:
                return {
                    "status": "error",
                    "source_id": source_id,
                    "connection_status": "failed",
                    "error": str(e),
                    "message": f"Connection test failed: {str(e)}"
                }
    
    def _register_data_transformation_tools(self):
        """Register data transformation tools."""
        
        @self.tool("add_transform")
        async def add_transform(transform_id: str, name: str, source_id: str, target_schema: Dict[str, Any], transform_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Add a data transformation rule."""
            try:
                # Validate source exists
                if source_id not in self.data_sources:
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"Data source not found: {source_id}"
                    )
                
                # Create transform
                transform = DataTransform(
                    id=transform_id,
                    name=name,
                    source_id=source_id,
                    target_schema=target_schema,
                    transform_rules=transform_rules
                )
                
                # Save to Redis
                await self.redis_client.set(
                    f"transform:{transform_id}",
                    json.dumps(asdict(transform), default=str)
                )
                
                self.transforms[transform_id] = transform
                
                return {
                    "status": "success",
                    "transform_id": transform_id,
                    "message": f"Transform '{name}' added successfully"
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to add transform: {str(e)}"
                )
        
        @self.tool("apply_transform")
        async def apply_transform(transform_id: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Apply transformation to data."""
            if transform_id not in self.transforms:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Transform not found: {transform_id}"
                )
            
            transform = self.transforms[transform_id]
            
            try:
                transformed_data = []
                
                for item in data:
                    transformed_item = {}
                    
                    for rule in transform.transform_rules:
                        source_field = rule.get('source_field')
                        target_field = rule.get('target_field')
                        transform_type = rule.get('transform_type', 'copy')
                        
                        if source_field in item:
                            value = item[source_field]
                            
                            # Apply transformation
                            if transform_type == 'copy':
                                transformed_value = value
                            elif transform_type == 'uppercase':
                                transformed_value = str(value).upper()
                            elif transform_type == 'lowercase':
                                transformed_value = str(value).lower()
                            elif transform_type == 'format_date':
                                # Simple date formatting
                                transformed_value = str(value)
                            elif transform_type == 'multiply':
                                multiplier = rule.get('multiplier', 1)
                                transformed_value = float(value) * multiplier
                            else:
                                transformed_value = value
                            
                            transformed_item[target_field] = transformed_value
                    
                    transformed_data.append(transformed_item)
                
                return {
                    "status": "success",
                    "transform_id": transform_id,
                    "original_count": len(data),
                    "transformed_count": len(transformed_data),
                    "transformed_data": transformed_data
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Transform application failed: {str(e)}"
                )
    
    def _register_data_query_tools(self):
        """Register data query tools."""
        
        @self.tool("query_data_source")
        async def query_data_source(source_id: str, query_params: Dict[str, Any] = None) -> Dict[str, Any]:
            """Query data from a specific source."""
            if source_id not in self.data_sources:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Data source not found: {source_id}"
                )
            
            source = self.data_sources[source_id]
            
            try:
                if source.type == 'api':
                    # Query API
                    url = source.config.get('url')
                    headers = source.config.get('headers', {})
                    params = query_params or {}
                    
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                "status": "success",
                                "source_id": source_id,
                                "data": data,
                                "count": len(data) if isinstance(data, list) else 1
                            }
                        else:
                            raise Exception(f"HTTP {response.status}")
                
                elif source.type == 'database':
                    # Query database
                    connection_string = source.config.get('connection_string')
                    query = query_params.get('query', 'SELECT * FROM data LIMIT 100')
                    
                    async with asyncpg.connect(connection_string) as conn:
                        rows = await conn.fetch(query)
                        data = [dict(row) for row in rows]
                        
                        return {
                            "status": "success",
                            "source_id": source_id,
                            "data": data,
                            "count": len(data)
                        }
                
                elif source.type == 'file':
                    # Query file
                    file_path = source.config.get('file_path')
                    file_type = source.config.get('file_type', 'json')
                    
                    if file_type == 'json':
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                    elif file_type == 'csv':
                        df = pd.read_csv(file_path)
                        data = df.to_dict('records')
                    else:
                        raise Exception(f"Unsupported file type: {file_type}")
                    
                    return {
                        "status": "success",
                        "source_id": source_id,
                        "data": data,
                        "count": len(data) if isinstance(data, list) else 1
                    }
                
                else:
                    raise Exception(f"Unsupported source type: {source.type}")
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Data query failed: {str(e)}"
                )
        
        @self.tool("search_across_sources")
        async def search_across_sources(search_term: str, source_ids: List[str] = None, limit: int = 100) -> Dict[str, Any]:
            """Search across multiple data sources."""
            try:
                if source_ids is None:
                    source_ids = list(self.data_sources.keys())
                
                results = {}
                total_results = 0
                
                for source_id in source_ids:
                    if source_id in self.data_sources:
                        try:
                            # Query each source
                            query_result = await self.query_data_source(source_id)
                            data = query_result.get('data', [])
                            
                            # Search in data
                            matching_items = []
                            for item in data:
                                if isinstance(item, dict):
                                    # Search in all string fields
                                    for value in item.values():
                                        if isinstance(value, str) and search_term.lower() in value.lower():
                                            matching_items.append(item)
                                            break
                                elif isinstance(item, str) and search_term.lower() in item.lower():
                                    matching_items.append(item)
                            
                            if matching_items:
                                results[source_id] = matching_items[:limit]
                                total_results += len(matching_items)
                        
                        except Exception as e:
                            self.logger.warning(f"Search failed for source {source_id}: {e}")
                
                return {
                    "status": "success",
                    "search_term": search_term,
                    "results": results,
                    "total_results": total_results,
                    "sources_searched": len(source_ids)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Cross-source search failed: {str(e)}"
                )
    
    def _register_sync_tools(self):
        """Register data synchronization tools."""
        
        @self.tool("sync_data_source")
        async def sync_data_source(source_id: str, force: bool = False) -> Dict[str, Any]:
            """Synchronize data from a source."""
            if source_id not in self.data_sources:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Data source not found: {source_id}"
                )
            
            source = self.data_sources[source_id]
            
            # Check if sync is needed
            if not force and source.last_sync:
                time_since_sync = datetime.now() - source.last_sync
                if time_since_sync.total_seconds() < source.sync_interval:
                    return {
                        "status": "skipped",
                        "source_id": source_id,
                        "message": "Sync not needed yet",
                        "next_sync_in": source.sync_interval - time_since_sync.total_seconds()
                    }
            
            try:
                # Query data from source
                query_result = await self.query_data_source(source_id)
                data = query_result.get('data', [])
                
                # Store in cache
                cache_key = f"data:{source_id}:latest"
                await self.redis_client.set(
                    cache_key,
                    json.dumps(data),
                    ex=source.sync_interval * 2  # Cache for 2x sync interval
                )
                
                # Update last sync time
                source.last_sync = datetime.now()
                await self.redis_client.set(
                    f"data_source:{source_id}",
                    json.dumps(asdict(source), default=str)
                )
                
                return {
                    "status": "success",
                    "source_id": source_id,
                    "records_synced": len(data),
                    "sync_time": source.last_sync.isoformat(),
                    "message": f"Successfully synced {len(data)} records"
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Data sync failed: {str(e)}"
                )
        
        @self.tool("sync_all_sources")
        async def sync_all_sources(force: bool = False) -> Dict[str, Any]:
            """Synchronize all data sources."""
            try:
                results = {}
                total_synced = 0
                failed_syncs = 0
                
                for source_id in self.data_sources.keys():
                    try:
                        sync_result = await self.sync_data_source(source_id, force)
                        results[source_id] = sync_result
                        
                        if sync_result["status"] == "success":
                            total_synced += sync_result.get("records_synced", 0)
                        elif sync_result["status"] == "error":
                            failed_syncs += 1
                    
                    except Exception as e:
                        results[source_id] = {
                            "status": "error",
                            "error": str(e)
                        }
                        failed_syncs += 1
                
                return {
                    "status": "completed",
                    "total_sources": len(self.data_sources),
                    "successful_syncs": len(self.data_sources) - failed_syncs,
                    "failed_syncs": failed_syncs,
                    "total_records_synced": total_synced,
                    "results": results
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Bulk sync failed: {str(e)}"
                )

# Usage example
async def main():
    """Main function to run the Data Integration MCP Server."""
    server = DataIntegrationMCPServer(
        name="data-integration-platform",
        redis_url="redis://localhost:6379"
    )
    
    await server.initialize()
    
    try:
        print("Data Integration MCP Server started")
        print("Available tools:")
        print("- Data Sources: add_data_source, list_data_sources, test_data_source")
        print("- Transforms: add_transform, apply_transform")
        print("- Queries: query_data_source, search_across_sources")
        print("- Sync: sync_data_source, sync_all_sources")
        
        # Keep server running
        await asyncio.sleep(3600)  # Run for 1 hour
        
    finally:
        await server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Project 3: Workflow Automation System

### Project Overview

Build a comprehensive workflow automation system that can orchestrate complex business processes, integrate with external services, and provide intelligent decision-making capabilities.

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Workflow      │◄──►│   MCP Server    │◄──►│   External      │
│   Definitions   │    │   (Automation   │    │   Services      │
│                 │    │    Engine)      │    │   (APIs, etc.)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Execution     │
                       │   Engine        │
                       └─────────────────┘
```

### Implementation

```python
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from mcp.server import Server
from mcp.types import ToolExecutionError, ErrorCode

class WorkflowStatus(Enum):
    """Workflow execution status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class TriggerType(Enum):
    """Workflow trigger types."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    WEBHOOK = "webhook"

@dataclass
class WorkflowStep:
    """Workflow step definition."""
    id: str
    name: str
    type: str  # 'tool', 'condition', 'loop', 'delay'
    config: Dict[str, Any]
    next_steps: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout: int = 300

@dataclass
class WorkflowTrigger:
    """Workflow trigger definition."""
    type: TriggerType
    config: Dict[str, Any]
    enabled: bool = True

@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    id: str
    name: str
    description: str
    version: str
    status: WorkflowStatus
    steps: Dict[str, WorkflowStep]
    triggers: List[WorkflowTrigger]
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowExecution:
    """Workflow execution instance."""
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class WorkflowAutomationMCPServer(Server):
    """Workflow Automation System MCP Server."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.running_executions: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(__name__)
        
        # Register tools
        self._register_workflow_management_tools()
        self._register_execution_tools()
        self._register_monitoring_tools()
        self._register_trigger_tools()
    
    def _register_workflow_management_tools(self):
        """Register workflow management tools."""
        
        @self.tool("create_workflow")
        async def create_workflow(workflow_id: str, name: str, description: str, steps: List[Dict[str, Any]], triggers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Create a new workflow."""
            try:
                # Validate workflow ID
                if workflow_id in self.workflows:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message=f"Workflow {workflow_id} already exists"
                    )
                
                # Parse steps
                workflow_steps = {}
                for step_data in steps:
                    step = WorkflowStep(
                        id=step_data["id"],
                        name=step_data["name"],
                        type=step_data["type"],
                        config=step_data["config"],
                        next_steps=step_data.get("next_steps", []),
                        retry_count=step_data.get("retry_count", 0),
                        timeout=step_data.get("timeout", 300)
                    )
                    workflow_steps[step.id] = step
                
                # Parse triggers
                workflow_triggers = []
                if triggers:
                    for trigger_data in triggers:
                        trigger = WorkflowTrigger(
                            type=TriggerType(trigger_data["type"]),
                            config=trigger_data["config"],
                            enabled=trigger_data.get("enabled", True)
                        )
                        workflow_triggers.append(trigger)
                
                # Create workflow
                workflow = WorkflowDefinition(
                    id=workflow_id,
                    name=name,
                    description=description,
                    version="1.0.0",
                    status=WorkflowStatus.DRAFT,
                    steps=workflow_steps,
                    triggers=workflow_triggers
                )
                
                self.workflows[workflow_id] = workflow
                
                return {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "message": f"Workflow '{name}' created successfully"
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Workflow creation failed: {str(e)}"
                )
        
        @self.tool("get_workflow")
        async def get_workflow(workflow_id: str) -> Dict[str, Any]:
            """Get workflow definition."""
            if workflow_id not in self.workflows:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Workflow not found: {workflow_id}"
                )
            
            workflow = self.workflows[workflow_id]
            
            return {
                "status": "success",
                "workflow": {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "version": workflow.version,
                    "status": workflow.status.value,
                    "steps": [
                        {
                            "id": step.id,
                            "name": step.name,
                            "type": step.type,
                            "config": step.config,
                            "next_steps": step.next_steps,
                            "retry_count": step.retry_count,
                            "timeout": step.timeout
                        }
                        for step in workflow.steps.values()
                    ],
                    "triggers": [
                        {
                            "type": trigger.type.value,
                            "config": trigger.config,
                            "enabled": trigger.enabled
                        }
                        for trigger in workflow.triggers
                    ],
                    "created_at": workflow.created_at.isoformat(),
                    "updated_at": workflow.updated_at.isoformat()
                }
            }
        
        @self.tool("list_workflows")
        async def list_workflows(status: str = None) -> Dict[str, Any]:
            """List workflows with optional status filter."""
            workflows = []
            
            for workflow in self.workflows.values():
                if status is None or workflow.status.value == status:
                    workflows.append({
                        "id": workflow.id,
                        "name": workflow.name,
                        "description": workflow.description,
                        "status": workflow.status.value,
                        "version": workflow.version,
                        "steps_count": len(workflow.steps),
                        "triggers_count": len(workflow.triggers),
                        "created_at": workflow.created_at.isoformat()
                    })
            
            return {
                "status": "success",
                "workflows": workflows,
                "total_count": len(workflows)
            }
        
        @self.tool("update_workflow_status")
        async def update_workflow_status(workflow_id: str, status: str) -> Dict[str, Any]:
            """Update workflow status."""
            if workflow_id not in self.workflows:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Workflow not found: {workflow_id}"
                )
            
            try:
                workflow = self.workflows[workflow_id]
                workflow.status = WorkflowStatus(status)
                workflow.updated_at = datetime.now()
                
                return {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "new_status": status,
                    "message": f"Workflow status updated to {status}"
                }
            
            except ValueError:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message=f"Invalid status: {status}"
                )
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Status update failed: {str(e)}"
                )
    
    def _register_execution_tools(self):
        """Register workflow execution tools."""
        
        @self.tool("execute_workflow")
        async def execute_workflow(workflow_id: str, context: Dict[str, Any] = None, trigger_type: str = "manual") -> Dict[str, Any]:
            """Execute a workflow."""
            if workflow_id not in self.workflows:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Workflow not found: {workflow_id}"
                )
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.ACTIVE:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message=f"Workflow is not active (status: {workflow.status.value})"
                )
            
            try:
                # Create execution instance
                execution_id = f"exec_{int(datetime.now().timestamp())}"
                execution = WorkflowExecution(
                    id=execution_id,
                    workflow_id=workflow_id,
                    status=WorkflowStatus.RUNNING,
                    started_at=datetime.now(),
                    context=context or {}
                )
                
                self.executions[execution_id] = execution
                
                # Start execution task
                task = asyncio.create_task(self._execute_workflow(execution))
                self.running_executions[execution_id] = task
                
                return {
                    "status": "success",
                    "execution_id": execution_id,
                    "workflow_id": workflow_id,
                    "message": "Workflow execution started"
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Workflow execution failed: {str(e)}"
                )
        
        @self.tool("get_execution_status")
        async def get_execution_status(execution_id: str) -> Dict[str, Any]:
            """Get workflow execution status."""
            if execution_id not in self.executions:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Execution not found: {execution_id}"
                )
            
            execution = self.executions[execution_id]
            
            return {
                "status": "success",
                "execution": {
                    "id": execution.id,
                    "workflow_id": execution.workflow_id,
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "current_step": execution.current_step,
                    "context": execution.context,
                    "step_results": execution.step_results,
                    "error_message": execution.error_message
                }
            }
        
        @self.tool("cancel_execution")
        async def cancel_execution(execution_id: str) -> Dict[str, Any]:
            """Cancel a running workflow execution."""
            if execution_id not in self.executions:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Execution not found: {execution_id}"
                )
            
            execution = self.executions[execution_id]
            
            if execution.status not in [WorkflowStatus.RUNNING]:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message=f"Cannot cancel execution with status: {execution.status.value}"
                )
            
            try:
                # Cancel the execution task
                if execution_id in self.running_executions:
                    task = self.running_executions[execution_id]
                    task.cancel()
                    del self.running_executions[execution_id]
                
                # Update execution status
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now()
                
                return {
                    "status": "success",
                    "execution_id": execution_id,
                    "message": "Execution cancelled successfully"
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Execution cancellation failed: {str(e)}"
                )
    
    def _register_monitoring_tools(self):
        """Register monitoring tools."""
        
        @self.tool("get_execution_metrics")
        async def get_execution_metrics(workflow_id: str = None, time_range: str = "24h") -> Dict[str, Any]:
            """Get execution metrics."""
            try:
                # Calculate time range
                now = datetime.now()
                if time_range == "1h":
                    start_time = now - timedelta(hours=1)
                elif time_range == "24h":
                    start_time = now - timedelta(days=1)
                elif time_range == "7d":
                    start_time = now - timedelta(days=7)
                elif time_range == "30d":
                    start_time = now - timedelta(days=30)
                else:
                    start_time = now - timedelta(days=1)
                
                # Filter executions
                executions = list(self.executions.values())
                if workflow_id:
                    executions = [e for e in executions if e.workflow_id == workflow_id]
                
                executions = [e for e in executions if e.started_at >= start_time]
                
                # Calculate metrics
                total_executions = len(executions)
                successful_executions = len([e for e in executions if e.status == WorkflowStatus.COMPLETED])
                failed_executions = len([e for e in executions if e.status == WorkflowStatus.FAILED])
                cancelled_executions = len([e for e in executions if e.status == WorkflowStatus.CANCELLED])
                running_executions = len([e for e in executions if e.status == WorkflowStatus.RUNNING])
                
                # Calculate average execution time
                completed_executions = [e for e in executions if e.completed_at]
                avg_execution_time = 0
                if completed_executions:
                    total_time = sum((e.completed_at - e.started_at).total_seconds() for e in completed_executions)
                    avg_execution_time = total_time / len(completed_executions)
                
                return {
                    "status": "success",
                    "metrics": {
                        "time_range": time_range,
                        "total_executions": total_executions,
                        "successful_executions": successful_executions,
                        "failed_executions": failed_executions,
                        "cancelled_executions": cancelled_executions,
                        "running_executions": running_executions,
                        "success_rate": successful_executions / max(total_executions, 1),
                        "average_execution_time": avg_execution_time
                    }
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get metrics: {str(e)}"
                )
        
        @self.tool("get_workflow_performance")
        async def get_workflow_performance(workflow_id: str) -> Dict[str, Any]:
            """Get performance metrics for a specific workflow."""
            if workflow_id not in self.workflows:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Workflow not found: {workflow_id}"
                )
            
            try:
                # Get executions for this workflow
                executions = [e for e in self.executions.values() if e.workflow_id == workflow_id]
                
                if not executions:
                    return {
                        "status": "success",
                        "workflow_id": workflow_id,
                        "message": "No executions found for this workflow"
                    }
                
                # Calculate step performance
                step_metrics = {}
                for execution in executions:
                    for step_id, result in execution.step_results.items():
                        if step_id not in step_metrics:
                            step_metrics[step_id] = {
                                "total_runs": 0,
                                "successful_runs": 0,
                                "failed_runs": 0,
                                "total_time": 0,
                                "retry_count": 0
                            }
                        
                        step_metrics[step_id]["total_runs"] += 1
                        if result.get("status") == "completed":
                            step_metrics[step_id]["successful_runs"] += 1
                        else:
                            step_metrics[step_id]["failed_runs"] += 1
                        
                        if "execution_time" in result:
                            step_metrics[step_id]["total_time"] += result["execution_time"]
                        
                        if "retry_count" in result:
                            step_metrics[step_id]["retry_count"] += result["retry_count"]
                
                # Calculate averages
                for step_id, metrics in step_metrics.items():
                    if metrics["total_runs"] > 0:
                        metrics["success_rate"] = metrics["successful_runs"] / metrics["total_runs"]
                        metrics["average_time"] = metrics["total_time"] / metrics["total_runs"]
                    else:
                        metrics["success_rate"] = 0
                        metrics["average_time"] = 0
                
                return {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "total_executions": len(executions),
                    "step_performance": step_metrics
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get workflow performance: {str(e)}"
                )
    
    def _register_trigger_tools(self):
        """Register trigger management tools."""
        
        @self.tool("add_workflow_trigger")
        async def add_workflow_trigger(workflow_id: str, trigger_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
            """Add a trigger to a workflow."""
            if workflow_id not in self.workflows:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Workflow not found: {workflow_id}"
                )
            
            try:
                trigger = WorkflowTrigger(
                    type=TriggerType(trigger_type),
                    config=config,
                    enabled=True
                )
                
                workflow = self.workflows[workflow_id]
                workflow.triggers.append(trigger)
                workflow.updated_at = datetime.now()
                
                return {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "trigger_type": trigger_type,
                    "message": "Trigger added successfully"
                }
            
            except ValueError:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message=f"Invalid trigger type: {trigger_type}"
                )
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to add trigger: {str(e)}"
                )
        
        @self.tool("trigger_workflow")
        async def trigger_workflow(workflow_id: str, trigger_data: Dict[str, Any] = None) -> Dict[str, Any]:
            """Manually trigger a workflow."""
            if workflow_id not in self.workflows:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Workflow not found: {workflow_id}"
                )
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.ACTIVE:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message=f"Workflow is not active (status: {workflow.status.value})"
                )
            
            try:
                # Execute workflow with trigger data
                context = trigger_data or {}
                result = await self.execute_workflow(workflow_id, context, "manual")
                
                return {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "execution_id": result["execution_id"],
                    "message": "Workflow triggered successfully"
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Workflow trigger failed: {str(e)}"
                )
    
    async def _execute_workflow(self, execution: WorkflowExecution):
        """Execute a workflow (internal method)."""
        try:
            workflow = self.workflows[execution.workflow_id]
            
            # Find starting step (first step or step with no incoming connections)
            current_step_id = self._find_starting_step(workflow)
            execution.current_step = current_step_id
            
            while current_step_id:
                step = workflow.steps[current_step_id]
                execution.current_step = current_step_id
                
                # Execute step
                step_result = await self._execute_step(step, execution)
                execution.step_results[current_step_id] = step_result
                
                if step_result["status"] == StepStatus.COMPLETED:
                    # Move to next step
                    current_step_id = self._get_next_step(step, execution)
                elif step_result["status"] == StepStatus.FAILED:
                    # Handle step failure
                    if step.retry_count > 0 and step_result.get("retry_count", 0) < step.retry_count:
                        # Retry step
                        await asyncio.sleep(1)  # Wait before retry
                        continue
                    else:
                        # Workflow failed
                        execution.status = WorkflowStatus.FAILED
                        execution.error_message = step_result.get("error", "Step execution failed")
                        break
                else:
                    # Step was skipped or other status
                    current_step_id = self._get_next_step(step, execution)
            
            # Mark execution as completed if not failed
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
            
            execution.completed_at = datetime.now()
            
        except asyncio.CancelledError:
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.now()
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
        finally:
            # Clean up running execution
            if execution.id in self.running_executions:
                del self.running_executions[execution.id]
    
    async def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_result = {
            "step_id": step.id,
            "step_name": step.name,
            "status": StepStatus.PENDING.value,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
            "retry_count": 0
        }
        
        try:
            step_result["status"] = StepStatus.RUNNING.value
            
            if step.type == "tool":
                # Execute MCP tool
                tool_name = step.config.get("tool_name")
                parameters = step.config.get("parameters", {})
                
                # Merge context variables into parameters
                for key, value in parameters.items():
                    if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                        var_name = value[2:-2]
                        if var_name in execution.context:
                            parameters[key] = execution.context[var_name]
                
                result = await self.call_tool(tool_name, parameters)
                step_result["result"] = result
                step_result["status"] = StepStatus.COMPLETED.value
                
            elif step.type == "condition":
                # Evaluate condition
                condition = step.config.get("condition")
                if self._evaluate_condition(condition, execution.context):
                    step_result["status"] = StepStatus.COMPLETED.value
                    step_result["result"] = {"condition_result": True}
                else:
                    step_result["status"] = StepStatus.SKIPPED.value
                    step_result["result"] = {"condition_result": False}
            
            elif step.type == "delay":
                # Wait for specified time
                delay_seconds = step.config.get("delay_seconds", 1)
                await asyncio.sleep(delay_seconds)
                step_result["status"] = StepStatus.COMPLETED.value
                step_result["result"] = {"delay_completed": True}
            
            elif step.type == "loop":
                # Execute loop
                loop_count = step.config.get("loop_count", 1)
                loop_steps = step.config.get("loop_steps", [])
                
                for i in range(loop_count):
                    execution.context["loop_index"] = i
                    for loop_step_id in loop_steps:
                        if loop_step_id in workflow.steps:
                            loop_step = workflow.steps[loop_step_id]
                            loop_result = await self._execute_step(loop_step, execution)
                            execution.step_results[f"{loop_step_id}_loop_{i}"] = loop_result
                
                step_result["status"] = StepStatus.COMPLETED.value
                step_result["result"] = {"loop_completed": True, "iterations": loop_count}
            
            else:
                raise Exception(f"Unknown step type: {step.type}")
            
            step_result["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            step_result["status"] = StepStatus.FAILED.value
            step_result["error"] = str(e)
            step_result["completed_at"] = datetime.now().isoformat()
        
        return step_result
    
    def _find_starting_step(self, workflow: WorkflowDefinition) -> Optional[str]:
        """Find the starting step of a workflow."""
        # Simple implementation: return the first step
        if workflow.steps:
            return list(workflow.steps.keys())[0]
        return None
    
    def _get_next_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Optional[str]:
        """Get the next step to execute."""
        if not step.next_steps:
            return None
        
        # Simple implementation: return the first next step
        # In a more sophisticated implementation, you might evaluate conditions
        return step.next_steps[0] if step.next_steps else None
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        try:
            # Simple condition evaluation
            # In production, use a proper expression evaluator
            if condition.startswith("context."):
                key = condition[8:]  # Remove "context." prefix
                return key in context and bool(context[key])
            elif condition.startswith("step_"):
                # Check if a previous step completed successfully
                step_id = condition[5:]  # Remove "step_" prefix
                return f"step_{step_id}_result" in context
            else:
                # Simple boolean evaluation
                return condition.lower() == "true"
        except Exception:
            return False

# Usage example
async def main():
    """Main function to run the Workflow Automation MCP Server."""
    server = WorkflowAutomationMCPServer(name="workflow-automation")
    
    try:
        print("Workflow Automation MCP Server started")
        print("Available tools:")
        print("- Workflow Management: create_workflow, get_workflow, list_workflows, update_workflow_status")
        print("- Execution: execute_workflow, get_execution_status, cancel_execution")
        print("- Monitoring: get_execution_metrics, get_workflow_performance")
        print("- Triggers: add_workflow_trigger, trigger_workflow")
        
        # Keep server running
        await asyncio.sleep(3600)  # Run for 1 hour
        
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        # Cancel all running executions
        for task in server.running_executions.values():
            task.cancel()
        
        # Wait for tasks to complete
        if server.running_executions:
            await asyncio.gather(*server.running_executions.values(), return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Patterns and Optimizations

### Performance Optimization Techniques

```python
import asyncio
import time
from typing import Dict, Any, List
from functools import wraps
import logging

def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    return wrapper

class OptimizedMCPServer(Server):
    """MCP Server with performance optimizations."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.cache = {}
        self.connection_pool = None
        self.rate_limiter = {}
    
    @performance_monitor
    async def optimized_tool_call(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Optimized tool call with caching and rate limiting."""
        # Rate limiting
        if tool_name in self.rate_limiter:
            last_call = self.rate_limiter[tool_name]
            if time.time() - last_call < 1:  # 1 second rate limit
                await asyncio.sleep(1 - (time.time() - last_call))
        
        self.rate_limiter[tool_name] = time.time()
        
        # Cache check
        cache_key = f"{tool_name}:{hash(str(params))}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Execute tool
        result = await self.call_tool(tool_name, params)
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
```

## Production Deployment

### Docker Compose Configuration

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
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

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

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - mcp-server

volumes:
  postgres_data:
  redis_data:
```

## Best Practices and Lessons Learned

### Security Best Practices

1. **Input Validation**: Always validate and sanitize inputs
2. **Authentication**: Implement proper authentication mechanisms
3. **Rate Limiting**: Prevent abuse with rate limiting
4. **Error Handling**: Don't expose sensitive information in errors
5. **Logging**: Log security events and monitor for anomalies

### Performance Best Practices

1. **Connection Pooling**: Use connection pools for databases and HTTP clients
2. **Caching**: Implement appropriate caching strategies
3. **Async Operations**: Use async/await for I/O operations
4. **Resource Management**: Properly manage and cleanup resources
5. **Monitoring**: Monitor performance metrics and set up alerts

### Development Best Practices

1. **Code Organization**: Use clear module structure and separation of concerns
2. **Error Handling**: Implement comprehensive error handling
3. **Testing**: Write unit tests and integration tests
4. **Documentation**: Document APIs and provide examples
5. **Version Control**: Use proper version control practices

## Final Project

### Project Requirements

Build a complete MCP application that demonstrates all the concepts learned in this course. The project should include:

1. **Complex Business Logic**: Implement real-world business processes
2. **Multiple Integrations**: Connect to external services and databases
3. **Advanced Features**: Use caching, error handling, and monitoring
4. **Production Ready**: Include deployment configuration and documentation
5. **Testing**: Provide comprehensive test coverage
6. **Documentation**: Create detailed documentation and examples

### Project Ideas

1. **E-commerce Platform**: Complete online store with inventory, orders, and payments
2. **Content Management System**: Full CMS with user management and publishing workflows
3. **Data Analytics Platform**: Real-time data processing and visualization system
4. **Customer Support System**: Ticketing system with AI-powered responses
5. **Project Management Tool**: Task management with team collaboration features

## 🎯 Module Summary

In this module, you learned:

- ✅ Building complete MCP applications from scratch
- ✅ Applying advanced MCP patterns and architectures
- ✅ Implementing performance optimization techniques
- ✅ Deploying MCP applications to production
- ✅ Applying best practices and lessons learned
- ✅ Contributing to the MCP ecosystem

## 🚀 Course Completion

Congratulations! You have completed the Complete MCP Course. You now have the knowledge and skills to:

- Build sophisticated MCP servers and clients
- Integrate MCP with AI applications
- Deploy production-ready MCP solutions
- Apply best practices and optimizations
- Contribute to the growing MCP ecosystem

## 🎓 Next Steps

1. **Build Your Own Project**: Apply what you've learned to create your own MCP application
2. **Contribute to Open Source**: Contribute to existing MCP projects and tools
3. **Share Your Knowledge**: Teach others about MCP development
4. **Stay Updated**: Follow MCP developments and new features
5. **Join the Community**: Participate in MCP forums and discussions

---

**Congratulations on completing the Complete MCP Course! 🎉**

*You are now ready to build the future of AI-human collaboration with MCP!*
