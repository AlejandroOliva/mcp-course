#!/usr/bin/env python3
"""
Hello World MCP Server
A simple MCP server that demonstrates basic functionality.
"""

import asyncio
from mcp.server import Server

# Create the server instance
server = Server("hello-server")

@server.tool("greet")
async def greet_tool(name: str = "World") -> str:
    """
    A simple greeting tool.
    
    Args:
        name: The name to greet (default: "World")
    
    Returns:
        A greeting message
    """
    return f"Hello, {name}! Welcome to MCP!"

@server.tool("echo")
async def echo_tool(message: str) -> str:
    """
    Echo back the provided message.
    
    Args:
        message: The message to echo back
    
    Returns:
        The same message
    """
    return f"Echo: {message}"

async def main():
    """Main server function."""
    print("Starting Hello World MCP Server...")
    print("Available tools:")
    print("- greet: Greet someone")
    print("- echo: Echo a message")
    
    # Run the server
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
