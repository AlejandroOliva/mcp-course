# Business Tools MCP Server - Usage Examples

## Basic Usage

### Starting the Server
```python
import asyncio
from business_tools_server import BusinessToolsMCPServer

async def main():
    server = BusinessToolsMCPServer(
        name="business-tools",
        database_url="postgresql://user:password@localhost/business_db",
        redis_url="redis://localhost:6379"
    )
    
    await server.initialize()
    
    try:
        print("Business Tools MCP Server started")
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        await server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Customer Management Examples

### Create a New Customer
```python
# Create customer
result = await server.call_tool("create_customer", {
    "name": "Acme Corporation",
    "email": "contact@acme.com",
    "company": "Acme Corp",
    "phone": "+1-555-0123",
    "address": {
        "street": "123 Business St",
        "city": "New York",
        "state": "NY",
        "zip": "10001",
        "country": "USA"
    }
}, user_id="admin")

print(f"Customer created: {result['customer']['id']}")
```

### Search Customers
```python
# Search customers by company name
result = await server.call_tool("search_customers", {
    "query": "Acme",
    "field": "company",
    "limit": 10
}, user_id="admin")

for customer in result['customers']:
    print(f"Found: {customer['name']} ({customer['company']})")
```

### Update Customer Status
```python
# Update customer status to active
result = await server.call_tool("update_customer", {
    "customer_id": "customer-uuid-here",
    "status": "active"
}, user_id="admin")

print(f"Customer status updated: {result['message']}")
```

## Order Management Examples

### Create an Order
```python
# Create order with multiple items
result = await server.call_tool("create_order", {
    "customer_id": "customer-uuid-here",
    "items": [
        {"product_id": "product-uuid-1", "quantity": 2},
        {"product_id": "product-uuid-2", "quantity": 1}
    ],
    "shipping_address": {
        "street": "123 Business St",
        "city": "New York",
        "state": "NY",
        "zip": "10001"
    },
    "payment_method": "credit_card"
}, user_id="admin")

print(f"Order created: {result['order']['id']} (Total: ${result['order']['total_amount']})")
```

### Process Payment
```python
# Process payment for order
result = await server.call_tool("process_payment", {
    "order_id": "order-uuid-here",
    "payment_amount": 1299.99,
    "payment_method": "credit_card"
}, user_id="admin")

if result['status'] == 'success':
    print(f"Payment processed: {result['payment']['payment_id']}")
    print(f"Order status: {result['order_status']}")
else:
    print(f"Payment failed: {result['message']}")
```

### Update Order Status
```python
# Update order status to shipped
result = await server.call_tool("update_order_status", {
    "order_id": "order-uuid-here",
    "new_status": "shipped"
}, user_id="admin")

print(f"Order status updated: {result['message']}")
```

## Inventory Management Examples

### Check Product Inventory
```python
# Check inventory by SKU
result = await server.call_tool("check_inventory", {
    "sku": "ENT-SW-001"
}, user_id="admin")

if result['available']:
    print(f"Product available: {result['current_stock']} units")
else:
    print("Product out of stock")
```

### Update Stock Levels
```python
# Add stock to inventory
result = await server.call_tool("update_stock", {
    "product_id": "product-uuid-here",
    "quantity_change": 50,
    "operation": "adjust"
}, user_id="admin")

print(f"Stock updated: {result['old_stock']} → {result['new_stock']}")
```

### Get Low Stock Products
```python
# Get products with low stock
result = await server.call_tool("get_low_stock_products", {
    "threshold": 10
}, user_id="admin")

print(f"Found {result['count']} products with low stock:")
for product in result['low_stock_products']:
    print(f"  - {product['name']}: {product['current_stock']} units")
```

## Analytics Examples

### Get Sales Analytics
```python
# Get sales analytics for last 30 days
result = await server.call_tool("get_sales_analytics", {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
}, user_id="admin")

print(f"Sales Analytics:")
print(f"  Total Orders: {result['total_orders']}")
print(f"  Total Revenue: ${result['total_revenue']}")
print(f"  Conversion Rate: {result['conversion_rate']}%")
print(f"  Top Products: {result['top_products']}")
```

### Get Customer Analytics
```python
# Get analytics for specific customer
result = await server.call_tool("get_customer_analytics", {
    "customer_id": "customer-uuid-here"
}, user_id="admin")

print(f"Customer Analytics:")
print(f"  Total Spent: ${result['total_spent']}")
print(f"  Order Count: {result['order_count']}")
print(f"  Avg Order Value: ${result['avg_order_value']}")
```

### Generate Business Report
```python
# Generate comprehensive sales report
result = await server.call_tool("generate_report", {
    "report_type": "sales",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
}, user_id="admin")

print(f"Report generated: {result['report']['report_id']}")
print(f"Report type: {result['report']['type']}")
```

## Workflow Automation Examples

### Create Automated Workflow
```python
# Create order processing workflow
result = await server.call_tool("create_workflow", {
    "name": "Order Processing Workflow",
    "description": "Automated order processing and fulfillment",
    "steps": [
        {
            "name": "validate_order",
            "type": "validation",
            "action": "check_customer_and_items"
        },
        {
            "name": "process_payment",
            "type": "payment",
            "action": "charge_customer"
        },
        {
            "name": "update_inventory",
            "type": "inventory",
            "action": "reserve_items"
        },
        {
            "name": "send_confirmation",
            "type": "notification",
            "action": "email_customer"
        },
        {
            "name": "create_shipment",
            "type": "shipping",
            "action": "generate_shipping_label"
        }
    ],
    "triggers": [
        {
            "type": "order_created",
            "condition": "status=pending"
        }
    ]
}, user_id="admin")

print(f"Workflow created: {result['workflow']['id']}")
```

### Execute Workflow
```python
# Execute workflow with input data
result = await server.call_tool("execute_workflow", {
    "workflow_id": "workflow-uuid-here",
    "input_data": {
        "order_id": "order-uuid-here",
        "customer_id": "customer-uuid-here",
        "priority": "normal"
    }
}, user_id="admin")

print(f"Workflow executed: {result['execution']['execution_id']}")
print(f"Steps completed: {result['execution']['steps_executed']}")
```

### Check Workflow Status
```python
# Check workflow status and details
result = await server.call_tool("get_workflow_status", {
    "workflow_id": "workflow-uuid-here"
}, user_id="admin")

workflow = result['workflow']
print(f"Workflow: {workflow['name']}")
print(f"Status: {workflow['status']}")
print(f"Steps: {len(workflow['steps'])}")
print(f"Triggers: {len(workflow['triggers'])}")
```

## Advanced Usage Patterns

### Batch Operations
```python
# Create multiple customers in batch
customers_data = [
    {"name": "Customer 1", "email": "c1@example.com", "company": "Corp 1"},
    {"name": "Customer 2", "email": "c2@example.com", "company": "Corp 2"},
    {"name": "Customer 3", "email": "c3@example.com", "company": "Corp 3"}
]

created_customers = []
for customer_data in customers_data:
    result = await server.call_tool("create_customer", customer_data, user_id="admin")
    if result['status'] == 'success':
        created_customers.append(result['customer'])

print(f"Created {len(created_customers)} customers")
```

### Error Handling
```python
# Robust error handling example
async def safe_create_order(order_data):
    try:
        result = await server.call_tool("create_order", order_data, user_id="admin")
        
        if result['status'] == 'success':
            return result['order']
        else:
            print(f"Order creation failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# Usage
order = await safe_create_order({
    "customer_id": "customer-uuid",
    "items": [{"product_id": "product-uuid", "quantity": 1}],
    "shipping_address": {"street": "123 Main St", "city": "NY", "state": "NY", "zip": "10001"},
    "payment_method": "credit_card"
})
```

### Rate Limiting Awareness
```python
# Handle rate limiting gracefully
async def create_customer_with_retry(customer_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await server.call_tool("create_customer", customer_data, user_id="admin")
            return result
            
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise e
            else:
                raise e
    
    return None
```

## Integration Examples

### REST API Integration
```python
from fastapi import FastAPI
from business_tools_server import BusinessToolsMCPServer

app = FastAPI()
server = BusinessToolsMCPServer("business-tools")

@app.on_event("startup")
async def startup():
    await server.initialize()

@app.post("/api/customers")
async def create_customer_api(customer_data: dict):
    result = await server.call_tool("create_customer", customer_data, user_id="api")
    return result

@app.get("/api/analytics/sales")
async def get_sales_analytics_api(start_date: str = None, end_date: str = None):
    result = await server.call_tool("get_sales_analytics", {
        "start_date": start_date,
        "end_date": end_date
    }, user_id="api")
    return result
```

### Webhook Integration
```python
# Webhook handler for external system integration
async def handle_order_webhook(webhook_data):
    # Extract order data from webhook
    order_data = webhook_data.get('order')
    
    # Create order in our system
    result = await server.call_tool("create_order", {
        "customer_id": order_data['customer_id'],
        "items": order_data['items'],
        "shipping_address": order_data['shipping_address'],
        "payment_method": order_data['payment_method']
    }, user_id="webhook")
    
    if result['status'] == 'success':
        # Trigger automated workflow
        await server.call_tool("execute_workflow", {
            "workflow_id": "order-processing-workflow",
            "input_data": {"order_id": result['order']['id']}
        }, user_id="webhook")
    
    return result
```

## Monitoring and Debugging

### Get Server Metrics
```python
# Get comprehensive server metrics
metrics = server.get_metrics()

print("Server Metrics:")
print(f"  Total Requests: {metrics['requests_total']}")
print(f"  Success Rate: {metrics['success_rate']:.2%}")
print(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")
print(f"  Max Response Time: {metrics['max_response_time']:.3f}s")
print(f"  Min Response Time: {metrics['min_response_time']:.3f}s")
print(f"  Errors: {metrics['requests_failed']}")
```

### Debug Tool Calls
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Tool calls will now show detailed debug information
result = await server.call_tool("create_customer", {
    "name": "Debug Customer",
    "email": "debug@example.com",
    "company": "Debug Corp"
}, user_id="debug")
```
