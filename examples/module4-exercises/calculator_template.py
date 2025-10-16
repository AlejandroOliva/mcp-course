#!/usr/bin/env python3
"""
Exercise 2: Calculator Server Template
A template for building an advanced calculator MCP server.

This is a starting template - complete the implementation following the module instructions.
"""

import asyncio
import math
from typing import Dict, Any, List
from mcp.server import Server
from mcp.types import ToolExecutionError, ErrorCode

class CalculatorServer:
    """MCP server for advanced calculator operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.server = Server(name)
        self.history: List[Dict[str, Any]] = []
        self._register_tools()
    
    def _register_tools(self):
        """Register calculator tools."""
        
        @self.server.tool("calculate")
        async def calculate(operation: str, a: float, b: float = None) -> Dict[str, Any]:
            """Perform basic arithmetic operations."""
            # TODO: Implement basic calculations
            # Requirements:
            # - Support: add, subtract, multiply, divide
            # - Validate operation parameter
            # - Handle division by zero
            # - Record in history
            # - Return result
            pass
        
        @self.server.tool("calculate_advanced")
        async def calculate_advanced(operation: str, value: float) -> Dict[str, Any]:
            """Perform advanced mathematical operations."""
            # TODO: Implement advanced calculations
            # Requirements:
            # - Support: power, sqrt, log, sin, cos, tan
            # - Validate operation parameter
            # - Handle invalid inputs (negative sqrt, etc.)
            # - Record in history
            # - Return result
            pass
        
        @self.server.tool("calculate_batch")
        async def calculate_batch(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Process multiple calculations."""
            # TODO: Implement batch processing
            # Requirements:
            # - Process list of operations
            # - Handle errors gracefully
            # - Return results for each operation
            # - Record all in history
            pass
        
        @self.server.tool("get_history")
        async def get_history(limit: int = 10) -> Dict[str, Any]:
            """Get calculation history."""
            # TODO: Implement history retrieval
            # Requirements:
            # - Return recent calculations
            # - Limit results if specified
            # - Include operation details
            pass
        
        @self.server.tool("clear_history")
        async def clear_history() -> Dict[str, Any]:
            """Clear calculation history."""
            # TODO: Implement history clearing
            # Requirements:
            # - Clear all history
            # - Return confirmation
            pass
    
    async def run(self):
        """Run the server."""
        print(f"Starting {self.name}...")
        await self.server.run()

# Example usage
if __name__ == "__main__":
    server = CalculatorServer("calculator-server")
    asyncio.run(server.run())
