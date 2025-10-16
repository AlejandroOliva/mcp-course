# Module 6: Real-World Applications

Welcome to Module 6! Now that you have a solid understanding of both MCP server and client development, let's apply this knowledge to build real-world applications. This module covers practical MCP implementations, external service integrations, business logic implementation, and production deployment strategies.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Build practical MCP applications for real-world scenarios
- Integrate MCP with external services and APIs
- Implement complex business logic using MCP
- Create production-ready MCP solutions
- Deploy and maintain MCP applications
- Optimize MCP applications for scale

## 📚 Topics Covered

1. [Database Integration Applications](#database-integration-applications)
2. [API Integration Applications](#api-integration-applications)
3. [File System Applications](#file-system-applications)
4. [Web Scraping Applications](#web-scraping-applications)
5. [Business Logic Applications](#business-logic-applications)
6. [Production Deployment](#production-deployment)
7. [Exercises](#exercises)

---

## Database Integration Applications

### Customer Management System

```python
import asyncio
import asyncpg
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class Customer:
    """Customer data model."""
    id: Optional[int] = None
    name: str = ""
    email: str = ""
    phone: Optional[str] = None
    address: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "active"

@dataclass
class Order:
    """Order data model."""
    id: Optional[int] = None
    customer_id: int = 0
    items: List[Dict[str, Any]] = None
    total_amount: float = 0.0
    status: str = "pending"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class CustomerManagementMCPServer(Server):
    """MCP server for customer management system."""
    
    def __init__(self, name: str, database_url: str):
        super().__init__(name)
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        self._register_customer_tools()
        self._register_order_tools()
        self._register_analytics_tools()
    
    async def initialize(self):
        """Initialize database connection."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20
        )
        
        # Create tables if they don't exist
        await self._create_tables()
    
    async def cleanup(self):
        """Cleanup database connection."""
        if self.pool:
            await self.pool.close()
    
    async def _create_tables(self):
        """Create database tables."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    phone VARCHAR(20),
                    address TEXT,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(id),
                    items JSONB NOT NULL,
                    total_amount DECIMAL(10,2) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _register_customer_tools(self):
        """Register customer management tools."""
        
        @self.tool("create_customer")
        async def create_customer(name: str, email: str, phone: str = None, address: str = None) -> Dict[str, Any]:
            """Create a new customer."""
            try:
                async with self.pool.acquire() as conn:
                    customer_id = await conn.fetchval("""
                        INSERT INTO customers (name, email, phone, address)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id
                    """, name, email, phone, address)
                    
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
                        row = await conn.fetchrow(
                            "SELECT * FROM customers WHERE id = $1", customer_id
                        )
                    else:
                        row = await conn.fetchrow(
                            "SELECT * FROM customers WHERE email = $1", email
                        )
                    
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
        
        @self.tool("update_customer")
        async def update_customer(customer_id: int, **updates) -> Dict[str, Any]:
            """Update customer information."""
            if not updates:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
                    message="No updates provided"
                )
            
            try:
                async with self.pool.acquire() as conn:
                    # Build dynamic update query
                    set_clauses = []
                    values = []
                    param_count = 1
                    
                    for field, value in updates.items():
                        if field in ['name', 'email', 'phone', 'address', 'status']:
                            set_clauses.append(f"{field} = ${param_count}")
                            values.append(value)
                            param_count += 1
                    
                    if not set_clauses:
                        raise ToolExecutionError(
                            code=ErrorCode.INVALID_PARAMS,
                            message="No valid fields to update"
                        )
                    
                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    values.append(customer_id)
                    
                    query = f"""
                        UPDATE customers 
                        SET {', '.join(set_clauses)}
                        WHERE id = ${param_count}
                        RETURNING *
                    """
                    
                    row = await conn.fetchrow(query, *values)
                    
                    if not row:
                        raise ToolExecutionError(
                            code=ErrorCode.NOT_FOUND,
                            message="Customer not found"
                        )
                    
                    return {
                        "status": "success",
                        "customer": dict(row),
                        "message": "Customer updated successfully"
                    }
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to update customer: {str(e)}"
                )
        
        @self.tool("list_customers")
        async def list_customers(status: str = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
            """List customers with optional filtering."""
            try:
                async with self.pool.acquire() as conn:
                    query = "SELECT * FROM customers"
                    params = []
                    
                    if status:
                        query += " WHERE status = $1"
                        params.append(status)
                    
                    query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
                    params.extend([limit, offset])
                    
                    rows = await conn.fetch(query, *params)
                    
                    # Get total count
                    count_query = "SELECT COUNT(*) FROM customers"
                    count_params = []
                    
                    if status:
                        count_query += " WHERE status = $1"
                        count_params.append(status)
                    
                    total_count = await conn.fetchval(count_query, *count_params)
                    
                    return {
                        "status": "success",
                        "customers": [dict(row) for row in rows],
                        "total_count": total_count,
                        "limit": limit,
                        "offset": offset
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to list customers: {str(e)}"
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
                # Calculate total amount
                total_amount = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
                
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        # Verify customer exists
                        customer = await conn.fetchrow(
                            "SELECT id FROM customers WHERE id = $1", customer_id
                        )
                        if not customer:
                            raise ToolExecutionError(
                                code=ErrorCode.NOT_FOUND,
                                message="Customer not found"
                            )
                        
                        # Create order
                        order_id = await conn.fetchval("""
                            INSERT INTO orders (customer_id, items, total_amount)
                            VALUES ($1, $2, $3)
                            RETURNING id
                        """, customer_id, json.dumps(items), total_amount)
                        
                        return {
                            "status": "success",
                            "order_id": order_id,
                            "total_amount": total_amount,
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
                        SET status = $1, updated_at = CURRENT_TIMESTAMP
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
        
        @self.tool("get_customer_analytics")
        async def get_customer_analytics() -> Dict[str, Any]:
            """Get customer analytics."""
            try:
                async with self.pool.acquire() as conn:
                    # Total customers
                    total_customers = await conn.fetchval("SELECT COUNT(*) FROM customers")
                    
                    # Active customers
                    active_customers = await conn.fetchval(
                        "SELECT COUNT(*) FROM customers WHERE status = 'active'"
                    )
                    
                    # New customers this month
                    new_customers = await conn.fetchval("""
                        SELECT COUNT(*) FROM customers 
                        WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
                    """)
                    
                    # Top customers by order count
                    top_customers = await conn.fetch("""
                        SELECT c.name, c.email, COUNT(o.id) as order_count, SUM(o.total_amount) as total_spent
                        FROM customers c
                        LEFT JOIN orders o ON c.id = o.customer_id
                        GROUP BY c.id, c.name, c.email
                        ORDER BY order_count DESC
                        LIMIT 10
                    """)
                    
                    return {
                        "status": "success",
                        "analytics": {
                            "total_customers": total_customers,
                            "active_customers": active_customers,
                            "new_customers_this_month": new_customers,
                            "top_customers": [dict(row) for row in top_customers]
                        }
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get analytics: {str(e)}"
                )
        
        @self.tool("get_order_analytics")
        async def get_order_analytics() -> Dict[str, Any]:
            """Get order analytics."""
            try:
                async with self.pool.acquire() as conn:
                    # Total orders
                    total_orders = await conn.fetchval("SELECT COUNT(*) FROM orders")
                    
                    # Total revenue
                    total_revenue = await conn.fetchval("SELECT SUM(total_amount) FROM orders")
                    
                    # Orders by status
                    orders_by_status = await conn.fetch("""
                        SELECT status, COUNT(*) as count
                        FROM orders
                        GROUP BY status
                    """)
                    
                    # Revenue by month
                    revenue_by_month = await conn.fetch("""
                        SELECT 
                            DATE_TRUNC('month', created_at) as month,
                            COUNT(*) as order_count,
                            SUM(total_amount) as revenue
                        FROM orders
                        GROUP BY DATE_TRUNC('month', created_at)
                        ORDER BY month DESC
                        LIMIT 12
                    """)
                    
                    return {
                        "status": "success",
                        "analytics": {
                            "total_orders": total_orders,
                            "total_revenue": float(total_revenue or 0),
                            "orders_by_status": [dict(row) for row in orders_by_status],
                            "revenue_by_month": [dict(row) for row in revenue_by_month]
                        }
                    }
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get order analytics: {str(e)}"
                )
```

## API Integration Applications

### E-commerce Integration System

```python
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

@dataclass
class Product:
    """Product data model."""
    id: str
    name: str
    price: float
    description: str
    category: str
    stock: int
    image_url: Optional[str] = None

@dataclass
class PaymentResult:
    """Payment result model."""
    transaction_id: str
    status: str
    amount: float
    currency: str
    payment_method: str

class EcommerceIntegrationMCPServer(Server):
    """MCP server for e-commerce integrations."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.payment_api_key = None
        self.shipping_api_key = None
        self.inventory_api_key = None
        self._register_product_tools()
        self._register_payment_tools()
        self._register_shipping_tools()
        self._register_inventory_tools()
    
    def configure_apis(self, payment_api_key: str, shipping_api_key: str, inventory_api_key: str):
        """Configure API keys for external services."""
        self.payment_api_key = payment_api_key
        self.shipping_api_key = shipping_api_key
        self.inventory_api_key = inventory_api_key
    
    def _register_product_tools(self):
        """Register product management tools."""
        
        @self.tool("search_products")
        async def search_products(query: str, category: str = None, min_price: float = None, max_price: float = None, limit: int = 20) -> Dict[str, Any]:
            """Search for products."""
            try:
                # This would integrate with a product search API
                # For demo purposes, we'll simulate the search
                products = [
                    Product(
                        id="1",
                        name="Wireless Headphones",
                        price=99.99,
                        description="High-quality wireless headphones",
                        category="Electronics",
                        stock=50,
                        image_url="https://example.com/headphones.jpg"
                    ),
                    Product(
                        id="2",
                        name="Smart Watch",
                        price=199.99,
                        description="Feature-rich smart watch",
                        category="Electronics",
                        stock=25,
                        image_url="https://example.com/smartwatch.jpg"
                    )
                ]
                
                # Filter products based on search criteria
                filtered_products = products
                
                if category:
                    filtered_products = [p for p in filtered_products if p.category.lower() == category.lower()]
                
                if min_price is not None:
                    filtered_products = [p for p in filtered_products if p.price >= min_price]
                
                if max_price is not None:
                    filtered_products = [p for p in filtered_products if p.price <= max_price]
                
                # Apply search query
                if query:
                    filtered_products = [
                        p for p in filtered_products 
                        if query.lower() in p.name.lower() or query.lower() in p.description.lower()
                    ]
                
                # Limit results
                filtered_products = filtered_products[:limit]
                
                return {
                    "status": "success",
                    "products": [asdict(p) for p in filtered_products],
                    "total_found": len(filtered_products)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Product search failed: {str(e)}"
                )
        
        @self.tool("get_product_details")
        async def get_product_details(product_id: str) -> Dict[str, Any]:
            """Get detailed product information."""
            try:
                # This would integrate with a product API
                # For demo purposes, we'll simulate the response
                product = Product(
                    id=product_id,
                    name="Sample Product",
                    price=99.99,
                    description="This is a sample product description",
                    category="Electronics",
                    stock=100,
                    image_url="https://example.com/product.jpg"
                )
                
                return {
                    "status": "success",
                    "product": asdict(product)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get product details: {str(e)}"
                )
    
    def _register_payment_tools(self):
        """Register payment processing tools."""
        
        @self.tool("process_payment")
        async def process_payment(amount: float, currency: str, payment_method: str, customer_info: Dict[str, Any]) -> Dict[str, Any]:
            """Process a payment."""
            if not self.payment_api_key:
                raise ToolExecutionError(
                    code=ErrorCode.CONFIGURATION_ERROR,
                    message="Payment API key not configured"
                )
            
            try:
                # This would integrate with a real payment API (Stripe, PayPal, etc.)
                # For demo purposes, we'll simulate the payment
                async with aiohttp.ClientSession() as session:
                    payment_data = {
                        "amount": amount,
                        "currency": currency,
                        "payment_method": payment_method,
                        "customer_info": customer_info
                    }
                    
                    headers = {
                        "Authorization": f"Bearer {self.payment_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Simulate API call
                    await asyncio.sleep(1)  # Simulate network delay
                    
                    # Simulate successful payment
                    payment_result = PaymentResult(
                        transaction_id=f"txn_{int(time.time())}",
                        status="succeeded",
                        amount=amount,
                        currency=currency,
                        payment_method=payment_method
                    )
                    
                    return {
                        "status": "success",
                        "payment": asdict(payment_result),
                        "message": "Payment processed successfully"
                    }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Payment processing failed: {str(e)}"
                )
        
        @self.tool("refund_payment")
        async def refund_payment(transaction_id: str, amount: float = None) -> Dict[str, Any]:
            """Refund a payment."""
            if not self.payment_api_key:
                raise ToolExecutionError(
                    code=ErrorCode.CONFIGURATION_ERROR,
                    message="Payment API key not configured"
                )
            
            try:
                # This would integrate with a real payment API
                # For demo purposes, we'll simulate the refund
                async with aiohttp.ClientSession() as session:
                    refund_data = {
                        "transaction_id": transaction_id,
                        "amount": amount
                    }
                    
                    headers = {
                        "Authorization": f"Bearer {self.payment_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Simulate API call
                    await asyncio.sleep(1)  # Simulate network delay
                    
                    return {
                        "status": "success",
                        "refund_id": f"refund_{int(time.time())}",
                        "amount": amount,
                        "message": "Refund processed successfully"
                    }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Refund processing failed: {str(e)}"
                )
    
    def _register_shipping_tools(self):
        """Register shipping tools."""
        
        @self.tool("calculate_shipping")
        async def calculate_shipping(items: List[Dict[str, Any]], destination: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate shipping costs."""
            if not self.shipping_api_key:
                raise ToolExecutionError(
                    code=ErrorCode.CONFIGURATION_ERROR,
                    message="Shipping API key not configured"
                )
            
            try:
                # This would integrate with a real shipping API (UPS, FedEx, etc.)
                # For demo purposes, we'll simulate the calculation
                total_weight = sum(item.get("weight", 1) * item.get("quantity", 1) for item in items)
                total_value = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
                
                # Simulate shipping calculation
                base_rate = 10.0
                weight_rate = total_weight * 0.5
                value_rate = total_value * 0.02
                
                shipping_cost = base_rate + weight_rate + value_rate
                
                return {
                    "status": "success",
                    "shipping_options": [
                        {
                            "service": "Standard",
                            "cost": shipping_cost,
                            "estimated_days": 5
                        },
                        {
                            "service": "Express",
                            "cost": shipping_cost * 1.5,
                            "estimated_days": 2
                        },
                        {
                            "service": "Overnight",
                            "cost": shipping_cost * 2,
                            "estimated_days": 1
                        }
                    ]
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Shipping calculation failed: {str(e)}"
                )
        
        @self.tool("create_shipment")
        async def create_shipment(items: List[Dict[str, Any]], destination: Dict[str, Any], service: str) -> Dict[str, Any]:
            """Create a shipment."""
            if not self.shipping_api_key:
                raise ToolExecutionError(
                    code=ErrorCode.CONFIGURATION_ERROR,
                    message="Shipping API key not configured"
                )
            
            try:
                # This would integrate with a real shipping API
                # For demo purposes, we'll simulate shipment creation
                shipment_id = f"ship_{int(time.time())}"
                
                return {
                    "status": "success",
                    "shipment_id": shipment_id,
                    "tracking_number": f"TRK{shipment_id}",
                    "service": service,
                    "message": "Shipment created successfully"
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Shipment creation failed: {str(e)}"
                )
    
    def _register_inventory_tools(self):
        """Register inventory management tools."""
        
        @self.tool("check_inventory")
        async def check_inventory(product_id: str, quantity: int = 1) -> Dict[str, Any]:
            """Check product inventory."""
            if not self.inventory_api_key:
                raise ToolExecutionError(
                    code=ErrorCode.CONFIGURATION_ERROR,
                    message="Inventory API key not configured"
                )
            
            try:
                # This would integrate with a real inventory API
                # For demo purposes, we'll simulate the check
                available_stock = 100  # Simulate available stock
                
                if available_stock >= quantity:
                    return {
                        "status": "success",
                        "available": True,
                        "stock": available_stock,
                        "message": f"Product {product_id} is available"
                    }
                else:
                    return {
                        "status": "success",
                        "available": False,
                        "stock": available_stock,
                        "message": f"Insufficient stock for product {product_id}"
                    }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Inventory check failed: {str(e)}"
                )
        
        @self.tool("reserve_inventory")
        async def reserve_inventory(product_id: str, quantity: int) -> Dict[str, Any]:
            """Reserve inventory for an order."""
            if not self.inventory_api_key:
                raise ToolExecutionError(
                    code=ErrorCode.CONFIGURATION_ERROR,
                    message="Inventory API key not configured"
                )
            
            try:
                # This would integrate with a real inventory API
                # For demo purposes, we'll simulate the reservation
                reservation_id = f"res_{int(time.time())}"
                
                return {
                    "status": "success",
                    "reservation_id": reservation_id,
                    "product_id": product_id,
                    "quantity": quantity,
                    "message": "Inventory reserved successfully"
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.EXTERNAL_ERROR,
                    message=f"Inventory reservation failed: {str(e)}"
                )
```

## File System Applications

### Document Management System

```python
import os
import shutil
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import json
from datetime import datetime

class DocumentManagementMCPServer(Server):
    """MCP server for document management."""
    
    def __init__(self, name: str, base_directory: str):
        super().__init__(name)
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self._register_document_tools()
        self._register_search_tools()
        self._register_metadata_tools()
    
    def _register_document_tools(self):
        """Register document management tools."""
        
        @self.tool("upload_document")
        async def upload_document(file_path: str, destination: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
            """Upload a document to the system."""
            try:
                source_path = Path(file_path)
                if not source_path.exists():
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"Source file not found: {file_path}"
                    )
                
                # Generate destination path
                if destination:
                    dest_path = self.base_directory / destination
                else:
                    # Use original filename
                    dest_path = self.base_directory / source_path.name
                
                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, dest_path)
                
                # Calculate file hash
                file_hash = self._calculate_file_hash(dest_path)
                
                # Save metadata
                metadata_file = dest_path.with_suffix(dest_path.suffix + '.meta')
                doc_metadata = {
                    "original_name": source_path.name,
                    "uploaded_at": datetime.now().isoformat(),
                    "file_size": dest_path.stat().st_size,
                    "file_hash": file_hash,
                    "mime_type": mimetypes.guess_type(str(dest_path))[0],
                    "custom_metadata": metadata or {}
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(doc_metadata, f, indent=2)
                
                return {
                    "status": "success",
                    "document_id": str(dest_path.relative_to(self.base_directory)),
                    "file_path": str(dest_path),
                    "metadata": doc_metadata,
                    "message": "Document uploaded successfully"
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Document upload failed: {str(e)}"
                )
        
        @self.tool("download_document")
        async def download_document(document_id: str, destination_path: str = None) -> Dict[str, Any]:
            """Download a document from the system."""
            try:
                doc_path = self.base_directory / document_id
                if not doc_path.exists():
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"Document not found: {document_id}"
                    )
                
                # Load metadata
                metadata_file = doc_path.with_suffix(doc_path.suffix + '.meta')
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Copy to destination if specified
                if destination_path:
                    dest_path = Path(destination_path)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(doc_path, dest_path)
                    
                    return {
                        "status": "success",
                        "document_id": document_id,
                        "destination": str(dest_path),
                        "metadata": metadata,
                        "message": "Document downloaded successfully"
                    }
                else:
                    # Return file content
                    with open(doc_path, 'rb') as f:
                        content = f.read()
                    
                    return {
                        "status": "success",
                        "document_id": document_id,
                        "content": content.decode('utf-8', errors='ignore'),
                        "metadata": metadata
                    }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Document download failed: {str(e)}"
                )
        
        @self.tool("delete_document")
        async def delete_document(document_id: str) -> Dict[str, Any]:
            """Delete a document from the system."""
            try:
                doc_path = self.base_directory / document_id
                if not doc_path.exists():
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"Document not found: {document_id}"
                    )
                
                # Delete document and metadata
                doc_path.unlink()
                metadata_file = doc_path.with_suffix(doc_path.suffix + '.meta')
                if metadata_file.exists():
                    metadata_file.unlink()
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "message": "Document deleted successfully"
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Document deletion failed: {str(e)}"
                )
        
        @self.tool("list_documents")
        async def list_documents(path: str = "", include_metadata: bool = False) -> Dict[str, Any]:
            """List documents in the system."""
            try:
                search_path = self.base_directory / path
                if not search_path.exists():
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"Path not found: {path}"
                    )
                
                documents = []
                for item in search_path.rglob("*"):
                    if item.is_file() and not item.name.endswith('.meta'):
                        doc_info = {
                            "document_id": str(item.relative_to(self.base_directory)),
                            "name": item.name,
                            "size": item.stat().st_size,
                            "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                        }
                        
                        if include_metadata:
                            metadata_file = item.with_suffix(item.suffix + '.meta')
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    doc_info["metadata"] = json.load(f)
                        
                        documents.append(doc_info)
                
                return {
                    "status": "success",
                    "documents": documents,
                    "total_count": len(documents)
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Document listing failed: {str(e)}"
                )
    
    def _register_search_tools(self):
        """Register document search tools."""
        
        @self.tool("search_documents")
        async def search_documents(query: str, file_types: List[str] = None, metadata_filters: Dict[str, Any] = None) -> Dict[str, Any]:
            """Search for documents."""
            try:
                results = []
                
                for doc_path in self.base_directory.rglob("*"):
                    if doc_path.is_file() and not doc_path.name.endswith('.meta'):
                        # Check file type filter
                        if file_types:
                            file_ext = doc_path.suffix.lower()
                            if file_ext not in file_types:
                                continue
                        
                        # Load metadata
                        metadata_file = doc_path.with_suffix(doc_path.suffix + '.meta')
                        metadata = {}
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                        
                        # Check metadata filters
                        if metadata_filters:
                            match = True
                            for key, value in metadata_filters.items():
                                if key not in metadata.get("custom_metadata", {}):
                                    match = False
                                    break
                                if metadata["custom_metadata"][key] != value:
                                    match = False
                                    break
                            if not match:
                                continue
                        
                        # Search in filename and metadata
                        if query.lower() in doc_path.name.lower():
                            results.append({
                                "document_id": str(doc_path.relative_to(self.base_directory)),
                                "name": doc_path.name,
                                "match_type": "filename",
                                "metadata": metadata
                            })
                        elif query.lower() in str(metadata.get("custom_metadata", {})):
                            results.append({
                                "document_id": str(doc_path.relative_to(self.base_directory)),
                                "name": doc_path.name,
                                "match_type": "metadata",
                                "metadata": metadata
                            })
                
                return {
                    "status": "success",
                    "query": query,
                    "results": results,
                    "total_found": len(results)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Document search failed: {str(e)}"
                )
    
    def _register_metadata_tools(self):
        """Register metadata management tools."""
        
        @self.tool("update_document_metadata")
        async def update_document_metadata(document_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
            """Update document metadata."""
            try:
                doc_path = self.base_directory / document_id
                if not doc_path.exists():
                    raise ToolExecutionError(
                        code=ErrorCode.NOT_FOUND,
                        message=f"Document not found: {document_id}"
                    )
                
                metadata_file = doc_path.with_suffix(doc_path.suffix + '.meta')
                
                # Load existing metadata
                existing_metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        existing_metadata = json.load(f)
                
                # Update custom metadata
                if "custom_metadata" not in existing_metadata:
                    existing_metadata["custom_metadata"] = {}
                
                existing_metadata["custom_metadata"].update(metadata)
                existing_metadata["updated_at"] = datetime.now().isoformat()
                
                # Save updated metadata
                with open(metadata_file, 'w') as f:
                    json.dump(existing_metadata, f, indent=2)
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "metadata": existing_metadata,
                    "message": "Document metadata updated successfully"
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Metadata update failed: {str(e)}"
                )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
```

## Web Scraping Applications

### Data Collection System

```python
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
import re
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime

class WebScrapingMCPServer(Server):
    """MCP server for web scraping applications."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.session: Optional[aiohttp.ClientSession] = None
        self._register_scraping_tools()
        self._register_data_extraction_tools()
        self._register_monitoring_tools()
    
    async def initialize(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
    
    async def cleanup(self):
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
    
    def _register_scraping_tools(self):
        """Register web scraping tools."""
        
        @self.tool("scrape_website")
        async def scrape_website(url: str, selectors: Dict[str, str] = None, extract_links: bool = False, max_depth: int = 1) -> Dict[str, Any]:
            """Scrape a website for data."""
            try:
                if not self.session:
                    raise ToolExecutionError(
                        code=ErrorCode.CONFIGURATION_ERROR,
                        message="HTTP session not initialized"
                    )
                
                # Validate URL
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ToolExecutionError(
                        code=ErrorCode.INVALID_PARAMS,
                        message="Invalid URL format"
                    )
                
                # Fetch page content
                async with self.session.get(url) as response:
                    if response.status != 200:
                        raise ToolExecutionError(
                            code=ErrorCode.EXTERNAL_ERROR,
                            message=f"HTTP {response.status}: Failed to fetch page"
                        )
                    
                    html_content = await response.text()
                
                # Parse HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract data based on selectors
                extracted_data = {}
                if selectors:
                    for key, selector in selectors.items():
                        elements = soup.select(selector)
                        if elements:
                            extracted_data[key] = [elem.get_text(strip=True) for elem in elements]
                        else:
                            extracted_data[key] = []
                
                # Extract links if requested
                links = []
                if extract_links:
                    link_elements = soup.find_all('a', href=True)
                    for link in link_elements:
                        href = link['href']
                        absolute_url = urljoin(url, href)
                        links.append({
                            "text": link.get_text(strip=True),
                            "url": absolute_url
                        })
                
                # Extract page metadata
                metadata = {
                    "title": soup.title.string if soup.title else "",
                    "description": "",
                    "keywords": "",
                    "url": url,
                    "scraped_at": datetime.now().isoformat()
                }
                
                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    metadata["description"] = meta_desc.get('content', '')
                
                # Extract meta keywords
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                if meta_keywords:
                    metadata["keywords"] = meta_keywords.get('content', '')
                
                return {
                    "status": "success",
                    "url": url,
                    "metadata": metadata,
                    "extracted_data": extracted_data,
                    "links": links[:max_depth * 10] if extract_links else [],
                    "message": "Website scraped successfully"
                }
            
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Web scraping failed: {str(e)}"
                )
        
        @self.tool("scrape_multiple_pages")
        async def scrape_multiple_pages(urls: List[str], selectors: Dict[str, str] = None, max_concurrent: int = 5) -> Dict[str, Any]:
            """Scrape multiple pages concurrently."""
            try:
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def scrape_single_page(url):
                    async with semaphore:
                        return await self.scrape_website(url, selectors)
                
                tasks = [scrape_single_page(url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                successful_results = []
                failed_results = []
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_results.append({
                            "url": urls[i],
                            "error": str(result)
                        })
                    else:
                        successful_results.append(result)
                
                return {
                    "status": "success",
                    "total_urls": len(urls),
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                    "results": successful_results,
                    "errors": failed_results
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Multiple page scraping failed: {str(e)}"
                )
    
    def _register_data_extraction_tools(self):
        """Register data extraction tools."""
        
        @self.tool("extract_emails")
        async def extract_emails(url: str) -> Dict[str, Any]:
            """Extract email addresses from a webpage."""
            try:
                # Scrape the page
                scrape_result = await self.scrape_website(url)
                html_content = scrape_result.get("extracted_data", {})
                
                # Extract emails using regex
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = set()
                
                # Search in all text content
                for key, values in html_content.items():
                    for value in values:
                        found_emails = re.findall(email_pattern, value)
                        emails.update(found_emails)
                
                return {
                    "status": "success",
                    "url": url,
                    "emails": list(emails),
                    "count": len(emails)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Email extraction failed: {str(e)}"
                )
        
        @self.tool("extract_phone_numbers")
        async def extract_phone_numbers(url: str) -> Dict[str, Any]:
            """Extract phone numbers from a webpage."""
            try:
                # Scrape the page
                scrape_result = await self.scrape_website(url)
                html_content = scrape_result.get("extracted_data", {})
                
                # Extract phone numbers using regex
                phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
                phone_numbers = set()
                
                # Search in all text content
                for key, values in html_content.items():
                    for value in values:
                        found_phones = re.findall(phone_pattern, value)
                        for phone in found_phones:
                            # Format phone number
                            formatted_phone = ''.join(phone)
                            phone_numbers.add(formatted_phone)
                
                return {
                    "status": "success",
                    "url": url,
                    "phone_numbers": list(phone_numbers),
                    "count": len(phone_numbers)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Phone number extraction failed: {str(e)}"
                )
        
        @self.tool("extract_social_media_links")
        async def extract_social_media_links(url: str) -> Dict[str, Any]:
            """Extract social media links from a webpage."""
            try:
                # Scrape the page with links
                scrape_result = await self.scrape_website(url, extract_links=True)
                links = scrape_result.get("links", [])
                
                # Social media patterns
                social_patterns = {
                    "facebook": r'facebook\.com',
                    "twitter": r'twitter\.com|x\.com',
                    "instagram": r'instagram\.com',
                    "linkedin": r'linkedin\.com',
                    "youtube": r'youtube\.com',
                    "tiktok": r'tiktok\.com'
                }
                
                social_links = {}
                
                for link in links:
                    link_url = link["url"].lower()
                    for platform, pattern in social_patterns.items():
                        if re.search(pattern, link_url):
                            if platform not in social_links:
                                social_links[platform] = []
                            social_links[platform].append(link)
                
                return {
                    "status": "success",
                    "url": url,
                    "social_links": social_links,
                    "platforms_found": list(social_links.keys())
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Social media link extraction failed: {str(e)}"
                )
    
    def _register_monitoring_tools(self):
        """Register monitoring tools."""
        
        @self.tool("monitor_website_changes")
        async def monitor_website_changes(url: str, selectors: Dict[str, str], previous_content: Dict[str, Any] = None) -> Dict[str, Any]:
            """Monitor website for changes."""
            try:
                # Scrape current content
                current_result = await self.scrape_website(url, selectors)
                current_content = current_result.get("extracted_data", {})
                
                if not previous_content:
                    return {
                        "status": "success",
                        "url": url,
                        "current_content": current_content,
                        "changes": [],
                        "message": "No previous content to compare"
                    }
                
                # Compare content
                changes = []
                for key, current_values in current_content.items():
                    previous_values = previous_content.get(key, [])
                    
                    # Check for new items
                    new_items = set(current_values) - set(previous_values)
                    if new_items:
                        changes.append({
                            "type": "added",
                            "selector": key,
                            "items": list(new_items)
                        })
                    
                    # Check for removed items
                    removed_items = set(previous_values) - set(current_values)
                    if removed_items:
                        changes.append({
                            "type": "removed",
                            "selector": key,
                            "items": list(removed_items)
                        })
                
                return {
                    "status": "success",
                    "url": url,
                    "current_content": current_content,
                    "changes": changes,
                    "has_changes": len(changes) > 0,
                    "change_count": len(changes)
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Website monitoring failed: {str(e)}"
                )
```

## Business Logic Applications

### Workflow Automation System

```python
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime, timedelta

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
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

@dataclass
class WorkflowStep:
    """Workflow step definition."""
    id: str
    name: str
    tool_name: str
    parameters: Dict[str, Any]
    condition: Optional[str] = None
    retry_count: int = 0
    timeout: int = 300

@dataclass
class WorkflowExecution:
    """Workflow execution instance."""
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps: List[Dict[str, Any]] = None
    context: Dict[str, Any] = None

class WorkflowAutomationMCPServer(Server):
    """MCP server for workflow automation."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self._register_workflow_tools()
        self._register_execution_tools()
        self._register_monitoring_tools()
    
    def _register_workflow_tools(self):
        """Register workflow management tools."""
        
        @self.tool("create_workflow")
        async def create_workflow(workflow_id: str, name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Create a new workflow."""
            try:
                # Validate workflow steps
                workflow_steps = []
                for step_data in steps:
                    step = WorkflowStep(
                        id=step_data["id"],
                        name=step_data["name"],
                        tool_name=step_data["tool_name"],
                        parameters=step_data["parameters"],
                        condition=step_data.get("condition"),
                        retry_count=step_data.get("retry_count", 0),
                        timeout=step_data.get("timeout", 300)
                    )
                    workflow_steps.append(step)
                
                self.workflows[workflow_id] = workflow_steps
                
                return {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "name": name,
                    "steps_count": len(workflow_steps),
                    "message": f"Workflow '{name}' created successfully"
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INVALID_PARAMS,
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
            
            workflow_steps = self.workflows[workflow_id]
            
            return {
                "status": "success",
                "workflow_id": workflow_id,
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "tool_name": step.tool_name,
                        "parameters": step.parameters,
                        "condition": step.condition,
                        "retry_count": step.retry_count,
                        "timeout": step.timeout
                    }
                    for step in workflow_steps
                ]
            }
        
        @self.tool("list_workflows")
        async def list_workflows() -> Dict[str, Any]:
            """List all workflows."""
            workflows = []
            for workflow_id, steps in self.workflows.items():
                workflows.append({
                    "workflow_id": workflow_id,
                    "steps_count": len(steps),
                    "created_at": "2024-01-01T00:00:00Z"  # Would be stored in real implementation
                })
            
            return {
                "status": "success",
                "workflows": workflows,
                "total_count": len(workflows)
            }
    
    def _register_execution_tools(self):
        """Register workflow execution tools."""
        
        @self.tool("execute_workflow")
        async def execute_workflow(workflow_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
            """Execute a workflow."""
            if workflow_id not in self.workflows:
                raise ToolExecutionError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Workflow not found: {workflow_id}"
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
                
                # Execute workflow steps
                workflow_steps = self.workflows[workflow_id]
                execution.steps = []
                
                for step in workflow_steps:
                    step_result = await self._execute_step(step, execution)
                    execution.steps.append(step_result)
                    
                    # Check if step failed and should stop workflow
                    if step_result["status"] == StepStatus.FAILED and step.retry_count == 0:
                        execution.status = WorkflowStatus.FAILED
                        break
                
                # Mark execution as completed if not failed
                if execution.status == WorkflowStatus.RUNNING:
                    execution.status = WorkflowStatus.COMPLETED
                
                execution.completed_at = datetime.now()
                
                return {
                    "status": "success",
                    "execution_id": execution_id,
                    "workflow_id": workflow_id,
                    "execution_status": execution.status.value,
                    "steps_executed": len(execution.steps),
                    "execution_time": (execution.completed_at - execution.started_at).total_seconds(),
                    "message": f"Workflow execution {execution.status.value}"
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
                "execution_id": execution_id,
                "workflow_id": execution.workflow_id,
                "execution_status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "steps": execution.steps or [],
                "context": execution.context
            }
    
    def _register_monitoring_tools(self):
        """Register monitoring tools."""
        
        @self.tool("get_workflow_metrics")
        async def get_workflow_metrics(workflow_id: str = None) -> Dict[str, Any]:
            """Get workflow execution metrics."""
            try:
                executions = list(self.executions.values())
                
                if workflow_id:
                    executions = [e for e in executions if e.workflow_id == workflow_id]
                
                total_executions = len(executions)
                successful_executions = len([e for e in executions if e.status == WorkflowStatus.COMPLETED])
                failed_executions = len([e for e in executions if e.status == WorkflowStatus.FAILED])
                
                # Calculate average execution time
                completed_executions = [e for e in executions if e.completed_at]
                avg_execution_time = 0
                if completed_executions:
                    total_time = sum((e.completed_at - e.started_at).total_seconds() for e in completed_executions)
                    avg_execution_time = total_time / len(completed_executions)
                
                return {
                    "status": "success",
                    "metrics": {
                        "total_executions": total_executions,
                        "successful_executions": successful_executions,
                        "failed_executions": failed_executions,
                        "success_rate": successful_executions / max(total_executions, 1),
                        "average_execution_time": avg_execution_time
                    }
                }
            
            except Exception as e:
                raise ToolExecutionError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Failed to get metrics: {str(e)}"
                )
    
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
            # Check condition if specified
            if step.condition:
                if not self._evaluate_condition(step.condition, execution.context):
                    step_result["status"] = StepStatus.SKIPPED.value
                    step_result["completed_at"] = datetime.now().isoformat()
                    return step_result
            
            # Execute step with retries
            for attempt in range(step.retry_count + 1):
                try:
                    step_result["retry_count"] = attempt
                    step_result["status"] = StepStatus.RUNNING.value
                    
                    # Call the tool
                    result = await self.call_tool(step.tool_name, step.parameters)
                    
                    step_result["status"] = StepStatus.COMPLETED.value
                    step_result["result"] = result
                    step_result["completed_at"] = datetime.now().isoformat()
                    
                    # Update execution context with step result
                    execution.context[f"step_{step.id}_result"] = result
                    
                    break
                
                except Exception as e:
                    if attempt < step.retry_count:
                        step_result["error"] = str(e)
                        await asyncio.sleep(1)  # Wait before retry
                        continue
                    else:
                        step_result["status"] = StepStatus.FAILED.value
                        step_result["error"] = str(e)
                        step_result["completed_at"] = datetime.now().isoformat()
                        break
        
        except Exception as e:
            step_result["status"] = StepStatus.FAILED.value
            step_result["error"] = str(e)
            step_result["completed_at"] = datetime.now().isoformat()
        
        return step_result
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a workflow condition."""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            # This is a basic implementation for demo purposes
            if condition.startswith("context."):
                key = condition[8:]  # Remove "context." prefix
                return key in context and context[key]
            elif condition.startswith("step_"):
                # Check if a previous step completed successfully
                step_id = condition[5:]  # Remove "step_" prefix
                return f"step_{step_id}_result" in context
            else:
                # Simple boolean evaluation
                return condition.lower() == "true"
        
        except Exception:
            return False
```

## Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile for production MCP server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
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
          value: "production-server"
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
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
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

## Exercises

### Exercise 1: Complete E-commerce System

Build a complete e-commerce MCP system with the following features:

**Requirements:**
- Customer management (CRUD operations)
- Product catalog management
- Order processing workflow
- Payment integration
- Inventory management
- Analytics and reporting
- Admin dashboard tools

### Exercise 2: Content Management System

Create a comprehensive CMS MCP server:

**Requirements:**
- Document upload and management
- Content versioning
- Search and filtering
- Metadata management
- Content approval workflow
- Publishing automation
- Analytics tracking

### Exercise 3: Data Pipeline System

Build a data pipeline MCP server:

**Requirements:**
- Data extraction from multiple sources
- Data transformation and cleaning
- Data validation and quality checks
- Automated data processing workflows
- Error handling and recovery
- Monitoring and alerting
- Data export and reporting

## 🎯 Module Summary

In this module, you learned:

- ✅ Building practical MCP applications for real-world scenarios
- ✅ Integrating MCP with external services and APIs
- ✅ Implementing complex business logic using MCP
- ✅ Creating production-ready MCP solutions
- ✅ Deploying and maintaining MCP applications
- ✅ Optimizing MCP applications for scale

## 🚀 Next Steps

You're now ready to move on to **Module 7: Practical Projects**, where you'll learn about:
- Building complete MCP applications
- Advanced project patterns
- Performance optimization
- Deployment strategies
- Best practices and lessons learned

---

**Congratulations on completing Module 6! 🎉**

*Next: [Module 7: Practical Projects](module-07-practical-projects/README.md)*
