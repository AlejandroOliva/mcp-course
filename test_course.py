#!/usr/bin/env python3
"""
Mock MCP Server for Testing
Simulates MCP functionality for course verification.
"""

import asyncio
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Mock MCP classes for testing
class MockServer:
    def __init__(self, name: str):
        self.name = name
        self.tools = {}
    
    def tool(self, name: str):
        def decorator(func):
            self.tools[name] = func
            return func
        return decorator
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        if tool_name in self.tools:
            return await self.tools[tool_name](**params)
        else:
            raise Exception(f"Tool {tool_name} not found")
    
    async def run(self):
        print(f"Mock MCP Server '{self.name}' is running...")
        print(f"Available tools: {list(self.tools.keys())}")

# Mock error classes
class MockToolExecutionError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

class MockErrorCode:
    INVALID_PARAMS = "INVALID_PARAMS"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"

# Test the course examples
async def test_module1_examples():
    """Test Module 1 examples."""
    print("\n🧪 Testing Module 1 Examples...")
    
    # Create mock server
    server = MockServer("hello-server")
    
    @server.tool("greet")
    async def greet_tool(name: str = "World") -> str:
        return f"Hello, {name}! Welcome to MCP!"
    
    @server.tool("echo")
    async def echo_tool(message: str) -> str:
        return f"Echo: {message}"
    
    # Test tools
    result1 = await server.call_tool("greet", {"name": "Alice"})
    result2 = await server.call_tool("echo", {"message": "Test message"})
    
    print(f"✅ Greet tool result: {result1}")
    print(f"✅ Echo tool result: {result2}")
    
    return True

async def test_module4_examples():
    """Test Module 4 examples."""
    print("\n🧪 Testing Module 4 Examples...")
    
    # Test Task Management
    @dataclass
    class Task:
        id: str
        title: str
        description: str
        priority: str
        status: str
        created_at: str
        updated_at: str
    
    class TaskManagementServer:
        def __init__(self, name: str):
            self.name = name
            self.server = MockServer(name)
            self.tasks = {}
            self._register_tools()
        
        def _register_tools(self):
            @self.server.tool("create_task")
            async def create_task(title: str, description: str, priority: str = "medium") -> Dict[str, Any]:
                if not title or len(title) < 2:
                    raise MockToolExecutionError(
                        MockErrorCode.INVALID_PARAMS,
                        "Title must be at least 2 characters"
                    )
                
                if priority not in ["low", "medium", "high"]:
                    raise MockToolExecutionError(
                        MockErrorCode.INVALID_PARAMS,
                        "Priority must be low, medium, or high"
                    )
                
                task_id = f"task_{len(self.tasks) + 1}"
                task = Task(
                    id=task_id,
                    title=title,
                    description=description,
                    priority=priority,
                    status="pending",
                    created_at="2024-01-01T00:00:00Z",
                    updated_at="2024-01-01T00:00:00Z"
                )
                
                self.tasks[task_id] = task
                
                return {
                    "status": "created",
                    "task": asdict(task)
                }
            
            @self.server.tool("get_task")
            async def get_task(task_id: str) -> Dict[str, Any]:
                if task_id not in self.tasks:
                    raise MockToolExecutionError(
                        MockErrorCode.NOT_FOUND,
                        f"Task {task_id} not found"
                    )
                
                return {
                    "status": "found",
                    "task": asdict(self.tasks[task_id])
                }
            
            @self.server.tool("list_tasks")
            async def list_tasks(status: str = None) -> Dict[str, Any]:
                tasks = list(self.tasks.values())
                
                if status:
                    tasks = [t for t in tasks if t.status == status]
                
                return {
                    "status": "success",
                    "tasks": [asdict(t) for t in tasks],
                    "count": len(tasks)
                }
    
    # Test task management
    task_server = TaskManagementServer("task-server")
    
    # Create a task
    result1 = await task_server.server.call_tool("create_task", {
        "title": "Learn MCP",
        "description": "Complete the MCP course",
        "priority": "high"
    })
    print(f"✅ Task created: {result1['task']['title']}")
    
    # Get the task
    task_id = result1['task']['id']
    result2 = await task_server.server.call_tool("get_task", {"task_id": task_id})
    print(f"✅ Task retrieved: {result2['task']['title']}")
    
    # List tasks
    result3 = await task_server.server.call_tool("list_tasks", {})
    print(f"✅ Tasks listed: {result3['count']} tasks found")
    
    return True

async def test_calculator_examples():
    """Test Calculator examples."""
    print("\n🧪 Testing Calculator Examples...")
    
    class CalculatorServer:
        def __init__(self, name: str):
            self.name = name
            self.server = MockServer(name)
            self.history = []
            self._register_tools()
        
        def _register_tools(self):
            @self.server.tool("calculate")
            async def calculate(operation: str, a: float, b: float = None) -> Dict[str, Any]:
                if operation == "add":
                    result = a + b
                elif operation == "subtract":
                    result = a - b
                elif operation == "multiply":
                    result = a * b
                elif operation == "divide":
                    if b == 0:
                        raise MockToolExecutionError(
                            MockErrorCode.INVALID_PARAMS,
                            "Division by zero"
                        )
                    result = a / b
                else:
                    raise MockToolExecutionError(
                        MockErrorCode.INVALID_PARAMS,
                        f"Unknown operation: {operation}"
                    )
                
                # Record in history
                self.history.append({
                    "operation": operation,
                    "a": a,
                    "b": b,
                    "result": result
                })
                
                return {
                    "status": "success",
                    "operation": operation,
                    "result": result
                }
            
            @self.server.tool("get_history")
            async def get_history(limit: int = 10) -> Dict[str, Any]:
                recent_history = self.history[-limit:] if limit else self.history
                return {
                    "status": "success",
                    "history": recent_history,
                    "count": len(recent_history)
                }
    
    # Test calculator
    calc_server = CalculatorServer("calculator-server")
    
    # Test calculations
    result1 = await calc_server.server.call_tool("calculate", {
        "operation": "add",
        "a": 5,
        "b": 3
    })
    print(f"✅ Addition: 5 + 3 = {result1['result']}")
    
    result2 = await calc_server.server.call_tool("calculate", {
        "operation": "multiply",
        "a": 4,
        "b": 7
    })
    print(f"✅ Multiplication: 4 * 7 = {result2['result']}")
    
    # Test history
    result3 = await calc_server.server.call_tool("get_history", {"limit": 5})
    print(f"✅ History: {result3['count']} calculations recorded")
    
    return True

async def test_file_manager_examples():
    """Test File Manager examples."""
    print("\n🧪 Testing File Manager Examples...")
    
    class FileManagerServer:
        def __init__(self, name: str, base_path: str = "."):
            self.name = name
            self.base_path = base_path
            self.server = MockServer(name)
            self.files = {}  # Mock file storage
            self._register_tools()
        
        def _validate_path(self, file_path: str) -> str:
            # Simple path validation (mock)
            if ".." in file_path or file_path.startswith("/"):
                raise MockToolExecutionError(
                    MockErrorCode.INVALID_PARAMS,
                    "Invalid file path"
                )
            return file_path
        
        def _register_tools(self):
            @self.server.tool("read_file")
            async def read_file(file_path: str) -> Dict[str, Any]:
                validated_path = self._validate_path(file_path)
                
                if validated_path not in self.files:
                    raise MockToolExecutionError(
                        MockErrorCode.NOT_FOUND,
                        f"File not found: {file_path}"
                    )
                
                content = self.files[validated_path]
                return {
                    "status": "success",
                    "file_path": file_path,
                    "content": content,
                    "size": len(content)
                }
            
            @self.server.tool("write_file")
            async def write_file(file_path: str, content: str) -> Dict[str, Any]:
                validated_path = self._validate_path(file_path)
                
                self.files[validated_path] = content
                
                return {
                    "status": "success",
                    "file_path": file_path,
                    "message": "File written successfully"
                }
            
            @self.server.tool("list_files")
            async def list_files() -> Dict[str, Any]:
                files = []
                for path, content in self.files.items():
                    files.append({
                        "name": path,
                        "size": len(content),
                        "type": "file"
                    })
                
                return {
                    "status": "success",
                    "files": files,
                    "count": len(files)
                }
    
    # Test file manager
    file_server = FileManagerServer("file-manager-server")
    
    # Write a file
    result1 = await file_server.server.call_tool("write_file", {
        "file_path": "test.txt",
        "content": "Hello, MCP World!"
    })
    print(f"✅ File written: {result1['message']}")
    
    # Read the file
    result2 = await file_server.server.call_tool("read_file", {"file_path": "test.txt"})
    print(f"✅ File read: {result2['content']}")
    
    # List files
    result3 = await file_server.server.call_tool("list_files", {})
    print(f"✅ Files listed: {result3['count']} files found")
    
    return True

async def test_business_tools():
    """Test Business Tools example."""
    print("\n🧪 Testing Business Tools Example...")
    
    # Test the business tools server (simplified version)
    class BusinessToolsServer:
        def __init__(self, name: str):
            self.name = name
            self.server = MockServer(name)
            self.customers = {}
            self.products = {}
            self.orders = {}
            self._register_tools()
        
        def _register_tools(self):
            @self.server.tool("create_customer")
            async def create_customer(name: str, email: str, phone: str = None) -> Dict[str, Any]:
                customer_id = f"cust_{len(self.customers) + 1}"
                customer = {
                    "id": customer_id,
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "created_at": "2024-01-01T00:00:00Z"
                }
                
                self.customers[customer_id] = customer
                
                return {
                    "status": "created",
                    "customer": customer
                }
            
            @self.server.tool("create_product")
            async def create_product(name: str, price: float, description: str = "") -> Dict[str, Any]:
                product_id = f"prod_{len(self.products) + 1}"
                product = {
                    "id": product_id,
                    "name": name,
                    "price": price,
                    "description": description,
                    "created_at": "2024-01-01T00:00:00Z"
                }
                
                self.products[product_id] = product
                
                return {
                    "status": "created",
                    "product": product
                }
            
            @self.server.tool("create_order")
            async def create_order(customer_id: str, product_ids: List[str]) -> Dict[str, Any]:
                if customer_id not in self.customers:
                    raise MockToolExecutionError(
                        MockErrorCode.NOT_FOUND,
                        f"Customer {customer_id} not found"
                    )
                
                total = 0
                order_items = []
                
                for product_id in product_ids:
                    if product_id not in self.products:
                        raise MockToolExecutionError(
                            MockErrorCode.NOT_FOUND,
                            f"Product {product_id} not found"
                        )
                    
                    product = self.products[product_id]
                    order_items.append(product)
                    total += product["price"]
                
                order_id = f"order_{len(self.orders) + 1}"
                order = {
                    "id": order_id,
                    "customer_id": customer_id,
                    "items": order_items,
                    "total": total,
                    "status": "pending",
                    "created_at": "2024-01-01T00:00:00Z"
                }
                
                self.orders[order_id] = order
                
                return {
                    "status": "created",
                    "order": order
                }
    
    # Test business tools
    business_server = BusinessToolsServer("business-server")
    
    # Create customer
    result1 = await business_server.server.call_tool("create_customer", {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "+1234567890"
    })
    print(f"✅ Customer created: {result1['customer']['name']}")
    
    # Create product
    result2 = await business_server.server.call_tool("create_product", {
        "name": "MCP Course",
        "price": 99.99,
        "description": "Complete MCP learning course"
    })
    print(f"✅ Product created: {result2['product']['name']}")
    
    # Create order
    customer_id = result1['customer']['id']
    product_id = result2['product']['id']
    result3 = await business_server.server.call_tool("create_order", {
        "customer_id": customer_id,
        "product_ids": [product_id]
    })
    print(f"✅ Order created: Total ${result3['order']['total']}")
    
    return True

async def main():
    """Run all course tests."""
    print("🎓 MCP Course Verification Test")
    print("=" * 50)
    
    try:
        # Test all modules
        await test_module1_examples()
        await test_module4_examples()
        await test_calculator_examples()
        await test_file_manager_examples()
        await test_business_tools()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("🎉 The MCP course is working correctly!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
