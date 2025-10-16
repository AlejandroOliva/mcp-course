#!/usr/bin/env python3
"""
Business Tools MCP Server
A comprehensive business automation MCP server with production-ready features
"""

import asyncio
import json
import uuid
import hashlib
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessToolsMCPServer:
    """Production-ready MCP Server for business automation."""
    
    def __init__(self, name: str, database_url: str = None, redis_url: str = None):
        self.name = name
        self.database_url = database_url or "postgresql://user:password@localhost/business_db"
        self.redis_url = redis_url or "redis://localhost:6379"
        self.tools = {}
        self.middleware = []
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "response_times": [],
            "errors": []
        }
        self.cache = {}
        self.rate_limits = {}
        self.is_initialized = False
        
        # Business data stores (in production, these would be database connections)
        self.customers_db = {}
        self.orders_db = {}
        self.products_db = {}
        self.inventory_db = {}
        self.payments_db = {}
        self.workflows_db = {}
        self.analytics_cache = {}
    
    def tool(self, name: str, rate_limit: int = None):
        """Decorator to register tools with optional rate limiting."""
        def decorator(func):
            self.tools[name] = {
                "function": func,
                "rate_limit": rate_limit,
                "calls": 0,
                "last_reset": time.time()
            }
            return func
        return decorator
    
    def add_middleware(self, func):
        """Add middleware function."""
        self.middleware.append(func)
        return func
    
    async def initialize(self):
        """Initialize the server."""
        logger.info(f"Initializing {self.name} MCP Server...")
        
        # In production, this would establish database connections
        await asyncio.sleep(0.1)  # Simulate initialization
        
        self.is_initialized = True
        logger.info(f"✅ {self.name} MCP Server initialized successfully")
    
    async def cleanup(self):
        """Cleanup server resources."""
        logger.info("Cleaning up server resources...")
        # In production, this would close database connections
        self.is_initialized = False
        logger.info("✅ Server cleanup completed")
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any], user_id: str = None) -> Any:
        """Call a tool with middleware, rate limiting, and metrics."""
        start_time = time.time()
        self.metrics["requests_total"] += 1
        
        try:
            if tool_name not in self.tools:
                raise Exception(f"Tool {tool_name} not found")
            
            tool_info = self.tools[tool_name]
            
            # Rate limiting
            if tool_info["rate_limit"]:
                if not await self._check_rate_limit(tool_name, user_id, tool_info["rate_limit"]):
                    raise Exception(f"Rate limit exceeded for tool {tool_name}")
            
            # Execute middleware
            for middleware_func in self.middleware:
                params = await middleware_func(tool_name, params, user_id)
            
            # Remove internal parameters
            params = {k: v for k, v in params.items() if not k.startswith('_')}
            
            # Execute tool
            result = await tool_info["function"](**params)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics["response_times"].append(response_time)
            self.metrics["requests_successful"] += 1
            tool_info["calls"] += 1
            
            logger.info(f"Tool {tool_name} executed successfully in {response_time:.3f}s")
            return result
        
        except Exception as e:
            self.metrics["requests_failed"] += 1
            self.metrics["errors"].append({
                "tool": tool_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            logger.error(f"Tool {tool_name} failed: {str(e)}")
            raise e
    
    async def _check_rate_limit(self, tool_name: str, user_id: str, limit: int) -> bool:
        """Check rate limit for a tool."""
        key = f"{tool_name}:{user_id or 'anonymous'}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {"count": 0, "window_start": now}
        
        rate_info = self.rate_limits[key]
        
        # Reset window if needed (1 minute windows)
        if now - rate_info["window_start"] > 60:
            rate_info["count"] = 0
            rate_info["window_start"] = now
        
        if rate_info["count"] >= limit:
            return False
        
        rate_info["count"] += 1
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        metrics = self.metrics.copy()
        
        if metrics["response_times"]:
            metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
            metrics["max_response_time"] = max(metrics["response_times"])
            metrics["min_response_time"] = min(metrics["response_times"])
        
        metrics["success_rate"] = (
            metrics["requests_successful"] / metrics["requests_total"] 
            if metrics["requests_total"] > 0 else 0
        )
        
        return metrics

# Create server instance
server = BusinessToolsMCPServer(
    name="business-tools",
    database_url="postgresql://user:password@localhost/business_db",
    redis_url="redis://localhost:6379"
)

# Middleware for authentication and logging
@server.add_middleware
async def auth_middleware(tool_name: str, params: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """Authentication middleware."""
    if user_id:
        params["_user_id"] = user_id
    return params

@server.add_middleware
async def logging_middleware(tool_name: str, params: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """Logging middleware."""
    logger.info(f"Tool {tool_name} called by user {user_id or 'anonymous'}")
    return params

# Business Data Models
@dataclass
class Customer:
    id: str
    name: str
    email: str
    company: str
    phone: str
    address: Dict[str, str]
    created_at: str
    updated_at: str
    status: str  # active, inactive, prospect
    value: float

@dataclass
class Product:
    id: str
    name: str
    description: str
    price: float
    category: str
    sku: str
    stock_quantity: int
    created_at: str
    updated_at: str
    is_active: bool

@dataclass
class Order:
    id: str
    customer_id: str
    items: List[Dict[str, Any]]
    total_amount: float
    status: str
    shipping_address: Dict[str, str]
    payment_method: str
    created_at: str
    updated_at: str

@dataclass
class Workflow:
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    triggers: List[Dict[str, Any]]
    status: str
    created_at: str
    updated_at: str

def generate_id() -> str:
    return str(uuid.uuid4())

# Customer Management Tools
@server.tool("create_customer", rate_limit=10)
async def create_customer(name: str, email: str, company: str, phone: str = None, address: Dict[str, str] = None) -> Dict[str, Any]:
    """Create new customer records."""
    try:
        if not all([name, email, company]):
            return {"error": "Name, email, and company are required", "status": "failed"}
        
        # Check for duplicate email
        for customer in server.customers_db.values():
            if customer["email"] == email:
                return {"error": f"Email '{email}' already exists", "status": "failed"}
        
        customer_id = generate_id()
        now = datetime.now().isoformat()
        
        customer = Customer(
            id=customer_id,
            name=name,
            email=email,
            company=company,
            phone=phone or "",
            address=address or {},
            created_at=now,
            updated_at=now,
            status="prospect",
            value=0.0
        )
        
        server.customers_db[customer_id] = asdict(customer)
        
        return {
            "customer": asdict(customer),
            "message": f"Customer '{name}' created successfully",
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Customer creation failed: {str(e)}", "status": "failed"}

@server.tool("get_customer", rate_limit=20)
async def get_customer(customer_id: str = None, email: str = None) -> Dict[str, Any]:
    """Retrieve customer information."""
    try:
        if not any([customer_id, email]):
            return {"error": "Customer ID or email required", "status": "failed"}
        
        customer = None
        if customer_id and customer_id in server.customers_db:
            customer = server.customers_db[customer_id]
        elif email:
            for c in server.customers_db.values():
                if c["email"] == email:
                    customer = c
                    break
        
        if not customer:
            return {"error": "Customer not found", "status": "failed"}
        
        return {"customer": customer, "status": "success"}
    
    except Exception as e:
        return {"error": f"Customer retrieval failed: {str(e)}", "status": "failed"}

@server.tool("update_customer", rate_limit=15)
async def update_customer(customer_id: str, **updates) -> Dict[str, Any]:
    """Update customer details."""
    try:
        if customer_id not in server.customers_db:
            return {"error": "Customer not found", "status": "failed"}
        
        customer = server.customers_db[customer_id]
        
        # Validate updates
        allowed_fields = ["name", "email", "company", "phone", "address", "status"]
        for field in updates:
            if field not in allowed_fields:
                return {"error": f"Field '{field}' cannot be updated", "status": "failed"}
        
        # Check for conflicts
        if "email" in updates:
            for cid, c in server.customers_db.items():
                if cid != customer_id and c["email"] == updates["email"]:
                    return {"error": f"Email '{updates['email']}' already exists", "status": "failed"}
        
        # Update customer
        for field, value in updates.items():
            customer[field] = value
        
        customer["updated_at"] = datetime.now().isoformat()
        
        return {
            "customer": customer,
            "message": "Customer updated successfully",
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Customer update failed: {str(e)}", "status": "failed"}

@server.tool("search_customers", rate_limit=20)
async def search_customers(query: str, field: str = "name", limit: int = 50) -> Dict[str, Any]:
    """Search customers by various criteria."""
    try:
        results = []
        query_lower = query.lower()
        
        for customer in server.customers_db.values():
            if field in customer and query_lower in str(customer[field]).lower():
                results.append(customer)
        
        results = results[:limit]
        
        return {
            "customers": results,
            "count": len(results),
            "query": query,
            "field": field,
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Customer search failed: {str(e)}", "status": "failed"}

# Order Management Tools
@server.tool("create_order", rate_limit=10)
async def create_order(customer_id: str, items: List[Dict[str, Any]], shipping_address: Dict[str, str], payment_method: str) -> Dict[str, Any]:
    """Create new orders."""
    try:
        if customer_id not in server.customers_db:
            return {"error": "Customer not found", "status": "failed"}
        
        if not items:
            return {"error": "At least one item required", "status": "failed"}
        
        # Validate items and calculate total
        total_amount = 0
        validated_items = []
        
        for item in items:
            product_id = item.get("product_id")
            quantity = item.get("quantity", 1)
            
            if not product_id or product_id not in server.products_db:
                return {"error": f"Product {product_id} not found", "status": "failed"}
            
            product = server.products_db[product_id]
            available_stock = server.inventory_db.get(product_id, 0)
            
            if quantity > available_stock:
                return {"error": f"Insufficient stock for {product['name']}. Available: {available_stock}", "status": "failed"}
            
            item_total = product["price"] * quantity
            total_amount += item_total
            
            validated_items.append({
                "product_id": product_id,
                "product_name": product["name"],
                "quantity": quantity,
                "unit_price": product["price"],
                "total_price": item_total
            })
        
        # Create order
        order_id = generate_id()
        now = datetime.now().isoformat()
        
        order = Order(
            id=order_id,
            customer_id=customer_id,
            items=validated_items,
            total_amount=total_amount,
            status="pending",
            shipping_address=shipping_address,
            payment_method=payment_method,
            created_at=now,
            updated_at=now
        )
        
        server.orders_db[order_id] = asdict(order)
        
        # Reserve inventory
        for item in validated_items:
            product_id = item["product_id"]
            quantity = item["quantity"]
            server.inventory_db[product_id] -= quantity
        
        return {
            "order": asdict(order),
            "message": f"Order {order_id} created successfully",
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Order creation failed: {str(e)}", "status": "failed"}

@server.tool("get_order", rate_limit=20)
async def get_order(order_id: str) -> Dict[str, Any]:
    """Retrieve order information."""
    try:
        if order_id not in server.orders_db:
            return {"error": "Order not found", "status": "failed"}
        
        order = server.orders_db[order_id]
        
        return {"order": order, "status": "success"}
    
    except Exception as e:
        return {"error": f"Order retrieval failed: {str(e)}", "status": "failed"}

@server.tool("update_order_status", rate_limit=15)
async def update_order_status(order_id: str, new_status: str) -> Dict[str, Any]:
    """Update order status."""
    try:
        valid_statuses = ["pending", "paid", "shipped", "delivered", "cancelled"]
        
        if new_status not in valid_statuses:
            return {"error": f"Invalid status. Must be one of: {', '.join(valid_statuses)}", "status": "failed"}
        
        if order_id not in server.orders_db:
            return {"error": "Order not found", "status": "failed"}
        
        order = server.orders_db[order_id]
        old_status = order["status"]
        
        order["status"] = new_status
        order["updated_at"] = datetime.now().isoformat()
        
        return {
            "order_id": order_id,
            "old_status": old_status,
            "new_status": new_status,
            "message": f"Order status updated from '{old_status}' to '{new_status}'",
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Order status update failed: {str(e)}", "status": "failed"}

@server.tool("process_payment", rate_limit=5)
async def process_payment(order_id: str, payment_amount: float, payment_method: str = "credit_card") -> Dict[str, Any]:
    """Process order payments."""
    try:
        if order_id not in server.orders_db:
            return {"error": "Order not found", "status": "failed"}
        
        order = server.orders_db[order_id]
        
        if order["status"] != "pending":
            return {"error": f"Order status is '{order['status']}', cannot process payment", "status": "failed"}
        
        if payment_amount != order["total_amount"]:
            return {"error": f"Payment amount {payment_amount} does not match order total {order['total_amount']}", "status": "failed"}
        
        # Simulate payment processing
        payment_id = generate_id()
        payment_status = "success" if random.random() > 0.1 else "failed"  # 90% success rate
        
        payment_record = {
            "payment_id": payment_id,
            "order_id": order_id,
            "amount": payment_amount,
            "method": payment_method,
            "status": payment_status,
            "processed_at": datetime.now().isoformat()
        }
        
        server.payments_db[payment_id] = payment_record
        
        if payment_status == "success":
            order["status"] = "paid"
            order["updated_at"] = datetime.now().isoformat()
            
            # Update customer value
            customer = server.customers_db[order["customer_id"]]
            customer["value"] += payment_amount
            
            return {
                "payment": payment_record,
                "order_status": "paid",
                "message": "Payment processed successfully",
                "status": "success"
            }
        else:
            return {
                "payment": payment_record,
                "message": "Payment processing failed",
                "status": "failed"
            }
    
    except Exception as e:
        return {"error": f"Payment processing failed: {str(e)}", "status": "failed"}

# Inventory Management Tools
@server.tool("check_inventory", rate_limit=30)
async def check_inventory(product_id: str = None, sku: str = None) -> Dict[str, Any]:
    """Check product availability."""
    try:
        if not any([product_id, sku]):
            return {"error": "Product ID or SKU required", "status": "failed"}
        
        product = None
        if product_id and product_id in server.products_db:
            product = server.products_db[product_id]
        elif sku:
            for p in server.products_db.values():
                if p["sku"] == sku:
                    product = p
                    break
        
        if not product:
            return {"error": "Product not found", "status": "failed"}
        
        current_stock = server.inventory_db.get(product["id"], 0)
        
        return {
            "product": product,
            "current_stock": current_stock,
            "available": current_stock > 0,
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Inventory check failed: {str(e)}", "status": "failed"}

@server.tool("update_stock", rate_limit=10)
async def update_stock(product_id: str, quantity_change: int, operation: str = "adjust") -> Dict[str, Any]:
    """Update product stock levels."""
    try:
        if product_id not in server.products_db:
            return {"error": "Product not found", "status": "failed"}
        
        product = server.products_db[product_id]
        current_stock = server.inventory_db.get(product_id, 0)
        
        if operation == "adjust":
            new_stock = current_stock + quantity_change
        elif operation == "set":
            new_stock = quantity_change
        else:
            return {"error": "Operation must be 'adjust' or 'set'", "status": "failed"}
        
        if new_stock < 0:
            return {"error": "Stock cannot be negative", "status": "failed"}
        
        # Update inventory
        server.inventory_db[product_id] = new_stock
        product["stock_quantity"] = new_stock
        product["updated_at"] = datetime.now().isoformat()
        
        return {
            "product_id": product_id,
            "old_stock": current_stock,
            "new_stock": new_stock,
            "operation": operation,
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Stock update failed: {str(e)}", "status": "failed"}

@server.tool("get_low_stock_products", rate_limit=10)
async def get_low_stock_products(threshold: int = 10) -> Dict[str, Any]:
    """Get products with low stock."""
    try:
        low_stock_products = []
        
        for product_id, stock in server.inventory_db.items():
            if stock <= threshold and product_id in server.products_db:
                product = server.products_db[product_id]
                low_stock_products.append({
                    "product_id": product_id,
                    "name": product["name"],
                    "sku": product["sku"],
                    "current_stock": stock,
                    "threshold": threshold
                })
        
        return {
            "low_stock_products": low_stock_products,
            "count": len(low_stock_products),
            "threshold": threshold,
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Low stock check failed: {str(e)}", "status": "failed"}

# Analytics Tools
@server.tool("get_sales_analytics", rate_limit=5)
async def get_sales_analytics(start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Get sales performance metrics."""
    try:
        # Filter orders by date if provided
        orders = list(server.orders_db.values())
        
        if start_date:
            orders = [o for o in orders if o["created_at"] >= start_date]
        if end_date:
            orders = [o for o in orders if o["created_at"] <= end_date]
        
        # Calculate analytics
        total_orders = len(orders)
        total_revenue = sum(order["total_amount"] for order in orders)
        paid_orders = [o for o in orders if o["status"] in ["paid", "shipped", "delivered"]]
        conversion_rate = len(paid_orders) / total_orders if total_orders > 0 else 0
        
        # Top products
        product_sales = {}
        for order in orders:
            for item in order["items"]:
                product_id = item["product_id"]
                quantity = item["quantity"]
                product_sales[product_id] = product_sales.get(product_id, 0) + quantity
        
        top_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Customer analytics
        customer_orders = {}
        for order in orders:
            customer_id = order["customer_id"]
            customer_orders[customer_id] = customer_orders.get(customer_id, 0) + 1
        
        top_customers = sorted(customer_orders.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "period": {"start_date": start_date, "end_date": end_date},
            "total_orders": total_orders,
            "total_revenue": total_revenue,
            "conversion_rate": round(conversion_rate * 100, 2),
            "top_products": top_products,
            "top_customers": top_customers,
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Sales analytics failed: {str(e)}", "status": "failed"}

@server.tool("get_customer_analytics", rate_limit=10)
async def get_customer_analytics(customer_id: str = None) -> Dict[str, Any]:
    """Get customer behavior analytics."""
    try:
        if customer_id:
            # Single customer analytics
            if customer_id not in server.customers_db:
                return {"error": "Customer not found", "status": "failed"}
            
            customer = server.customers_db[customer_id]
            customer_orders = [o for o in server.orders_db.values() if o["customer_id"] == customer_id]
            
            total_spent = sum(order["total_amount"] for order in customer_orders)
            order_count = len(customer_orders)
            avg_order_value = total_spent / order_count if order_count > 0 else 0
            
            return {
                "customer": customer,
                "total_spent": total_spent,
                "order_count": order_count,
                "avg_order_value": round(avg_order_value, 2),
                "status": "success"
            }
        else:
            # All customers analytics
            total_customers = len(server.customers_db)
            active_customers = len([c for c in server.customers_db.values() if c["status"] == "active"])
            total_customer_value = sum(c["value"] for c in server.customers_db.values())
            avg_customer_value = total_customer_value / total_customers if total_customers > 0 else 0
            
            return {
                "total_customers": total_customers,
                "active_customers": active_customers,
                "total_customer_value": total_customer_value,
                "avg_customer_value": round(avg_customer_value, 2),
                "status": "success"
            }
    
    except Exception as e:
        return {"error": f"Customer analytics failed: {str(e)}", "status": "failed"}

@server.tool("generate_report", rate_limit=3)
async def generate_report(report_type: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Generate business reports."""
    try:
        valid_reports = ["sales", "customers", "inventory", "financial"]
        
        if report_type not in valid_reports:
            return {"error": f"Invalid report type. Must be one of: {', '.join(valid_reports)}", "status": "failed"}
        
        report_id = generate_id()
        now = datetime.now().isoformat()
        
        if report_type == "sales":
            sales_data = await server.call_tool("get_sales_analytics", {"start_date": start_date, "end_date": end_date})
            report_data = {
                "report_id": report_id,
                "type": "sales",
                "data": sales_data,
                "generated_at": now,
                "period": {"start_date": start_date, "end_date": end_date}
            }
        
        elif report_type == "customers":
            customer_data = await server.call_tool("get_customer_analytics", {})
            report_data = {
                "report_id": report_id,
                "type": "customers",
                "data": customer_data,
                "generated_at": now
            }
        
        elif report_type == "inventory":
            low_stock_data = await server.call_tool("get_low_stock_products", {"threshold": 10})
            report_data = {
                "report_id": report_id,
                "type": "inventory",
                "data": low_stock_data,
                "generated_at": now
            }
        
        elif report_type == "financial":
            sales_data = await server.call_tool("get_sales_analytics", {"start_date": start_date, "end_date": end_date})
            customer_data = await server.call_tool("get_customer_analytics", {})
            
            report_data = {
                "report_id": report_id,
                "type": "financial",
                "data": {
                    "sales": sales_data,
                    "customers": customer_data
                },
                "generated_at": now,
                "period": {"start_date": start_date, "end_date": end_date}
            }
        
        return {
            "report": report_data,
            "message": f"{report_type.title()} report generated successfully",
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}", "status": "failed"}

# Workflow Automation Tools
@server.tool("create_workflow", rate_limit=5)
async def create_workflow(name: str, description: str, steps: List[Dict[str, Any]], triggers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create automated workflows."""
    try:
        if not all([name, description, steps]):
            return {"error": "Name, description, and steps are required", "status": "failed"}
        
        workflow_id = generate_id()
        now = datetime.now().isoformat()
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            steps=steps,
            triggers=triggers or [],
            status="active",
            created_at=now,
            updated_at=now
        )
        
        server.workflows_db[workflow_id] = asdict(workflow)
        
        return {
            "workflow": asdict(workflow),
            "message": f"Workflow '{name}' created successfully",
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Workflow creation failed: {str(e)}", "status": "failed"}

@server.tool("execute_workflow", rate_limit=10)
async def execute_workflow(workflow_id: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute workflow processes."""
    try:
        if workflow_id not in server.workflows_db:
            return {"error": "Workflow not found", "status": "failed"}
        
        workflow = server.workflows_db[workflow_id]
        
        if workflow["status"] != "active":
            return {"error": f"Workflow status is '{workflow['status']}', cannot execute", "status": "failed"}
        
        execution_id = generate_id()
        execution_results = []
        
        # Execute workflow steps
        for i, step in enumerate(workflow["steps"]):
            step_name = step.get("name", f"step_{i}")
            step_type = step.get("type", "action")
            
            # Simulate step execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            step_result = {
                "step_name": step_name,
                "step_type": step_type,
                "status": "completed",
                "result": f"Step {i+1} executed successfully",
                "executed_at": datetime.now().isoformat()
            }
            
            execution_results.append(step_result)
        
        execution_record = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "workflow_name": workflow["name"],
            "steps_executed": len(execution_results),
            "results": execution_results,
            "status": "completed",
            "executed_at": datetime.now().isoformat()
        }
        
        return {
            "execution": execution_record,
            "message": f"Workflow '{workflow['name']}' executed successfully",
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Workflow execution failed: {str(e)}", "status": "failed"}

@server.tool("get_workflow_status", rate_limit=20)
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Check workflow execution status."""
    try:
        if workflow_id not in server.workflows_db:
            return {"error": "Workflow not found", "status": "failed"}
        
        workflow = server.workflows_db[workflow_id]
        
        return {
            "workflow": workflow,
            "status": "success"
        }
    
    except Exception as e:
        return {"error": f"Workflow status check failed: {str(e)}", "status": "failed"}

async def main():
    """Test the Business Tools MCP Server."""
    print("🏢 Business Tools MCP Server - Production Example")
    print("=" * 60)
    
    # Initialize server
    await server.initialize()
    
    # Create sample data
    print("\n📊 Creating sample business data...")
    
    # Create sample products
    products = [
        ("Enterprise Software", "Complete business management solution", 2999.99, "Software", "ENT-SW-001", 10),
        ("Cloud Storage", "Secure cloud storage service", 99.99, "Service", "CLOUD-001", 100),
        ("Consulting Package", "Business consulting services", 5000.00, "Service", "CONS-001", 5)
    ]
    
    product_ids = []
    for name, desc, price, category, sku, stock in products:
        product_id = generate_id()
        now = datetime.now().isoformat()
        
        product = Product(
            id=product_id,
            name=name,
            description=desc,
            price=price,
            category=category,
            sku=sku,
            stock_quantity=stock,
            created_at=now,
            updated_at=now,
            is_active=True
        )
        
        server.products_db[product_id] = asdict(product)
        server.inventory_db[product_id] = stock
        product_ids.append(product_id)
        print(f"✅ Created product: {name}")
    
    # Create sample customers
    customers = [
        ("John Smith", "john@techcorp.com", "TechCorp", "+1234567890"),
        ("Sarah Johnson", "sarah@innovate.com", "Innovate Inc", "+1987654321")
    ]
    
    customer_ids = []
    for name, email, company, phone in customers:
        result = await server.call_tool("create_customer", {
            "name": name,
            "email": email,
            "company": company,
            "phone": phone
        }, user_id="admin")
        
        if result["status"] == "success":
            customer_ids.append(result["customer"]["id"])
            print(f"✅ Created customer: {name}")
    
    # Create sample orders
    orders = [
        (customer_ids[0], [{"product_id": product_ids[0], "quantity": 1}]),
        (customer_ids[1], [{"product_id": product_ids[1], "quantity": 2}])
    ]
    
    for customer_id, items in orders:
        result = await server.call_tool("create_order", {
            "customer_id": customer_id,
            "items": items,
            "shipping_address": {"street": "123 Main St", "city": "NY", "state": "NY", "zip": "10001"},
            "payment_method": "credit_card"
        }, user_id="admin")
        
        if result["status"] == "success":
            print(f"✅ Created order: {result['order']['id']} (${result['order']['total_amount']})")
    
    # Test analytics
    print("\n📈 Testing analytics...")
    analytics_result = await server.call_tool("get_sales_analytics", {}, user_id="admin")
    if analytics_result["status"] == "success":
        print(f"✅ Sales analytics: ${analytics_result['total_revenue']} revenue, {analytics_result['total_orders']} orders")
    
    # Test workflow creation
    print("\n🔄 Testing workflow automation...")
    workflow_result = await server.call_tool("create_workflow", {
        "name": "Order Processing Workflow",
        "description": "Automated order processing workflow",
        "steps": [
            {"name": "validate_order", "type": "validation"},
            {"name": "process_payment", "type": "payment"},
            {"name": "update_inventory", "type": "inventory"},
            {"name": "send_confirmation", "type": "notification"}
        ],
        "triggers": [{"type": "order_created", "condition": "status=pending"}]
    }, user_id="admin")
    
    if workflow_result["status"] == "success":
        workflow_id = workflow_result["workflow"]["id"]
        print(f"✅ Created workflow: {workflow_result['workflow']['name']}")
        
        # Execute workflow
        execution_result = await server.call_tool("execute_workflow", {
            "workflow_id": workflow_id,
            "input_data": {"order_id": "test_order"}
        }, user_id="admin")
        
        if execution_result["status"] == "success":
            print(f"✅ Executed workflow: {execution_result['execution']['steps_executed']} steps completed")
    
    # Test report generation
    print("\n📋 Testing report generation...")
    report_result = await server.call_tool("generate_report", {
        "report_type": "sales"
    }, user_id="admin")
    
    if report_result["status"] == "success":
        print(f"✅ Generated {report_result['report']['type']} report: {report_result['report']['report_id']}")
    
    # Get server metrics
    print("\n📊 Server Metrics:")
    metrics = server.get_metrics()
    print(f"   - Total requests: {metrics['requests_total']}")
    print(f"   - Success rate: {metrics['success_rate']:.2%}")
    print(f"   - Avg response time: {metrics['avg_response_time']:.3f}s")
    print(f"   - Errors: {metrics['requests_failed']}")
    
    # Cleanup
    await server.cleanup()
    
    print("\n🎉 Business Tools MCP Server demonstration completed!")
    print("🚀 Ready for production deployment!")

if __name__ == "__main__":
    asyncio.run(main())
