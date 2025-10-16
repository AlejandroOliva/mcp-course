# Module 1: Introduction to MCP

Welcome to your first step in learning MCP (Model Context Protocol) development! This module will introduce you to MCP, its importance in AI development, and how to get started with building your first MCP server.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Understand what MCP is and why it's important
- Set up your MCP development environment
- Write and run your first MCP server
- Understand the basic MCP protocol flow
- Recognize MCP server structure

## 📚 Topics Covered

1. [What is MCP?](#what-is-mcp)
2. [Why Learn MCP?](#why-learn-mcp)
3. [Installing MCP SDK](#installing-mcp-sdk)
4. [Your First MCP Server](#your-first-mcp-server)
5. [Understanding the Protocol](#understanding-the-protocol)
6. [Basic Server Structure](#basic-server-structure)
7. [Exercises](#exercises)

---

## What is MCP?

MCP (Model Context Protocol) is a standardized protocol that enables AI agents to securely connect to data sources and tools. It acts as a bridge between AI models and external systems, allowing AI agents to perform actions and access information beyond their training data.

### Key Characteristics of MCP:

- **Standardized**: Provides a common interface for AI-tool communication
- **Secure**: Built-in authentication and authorization mechanisms
- **Extensible**: Supports custom tools and resources
- **Language Agnostic**: Works with multiple programming languages
- **Real-time**: Supports streaming and async operations
- **Production Ready**: Designed for enterprise deployment

### MCP's Role in AI Development:

MCP enables AI agents to:
- **Access External Data**: Connect to databases, APIs, and file systems
- **Perform Actions**: Execute tools and workflows
- **Maintain Context**: Share information across interactions
- **Scale Safely**: Handle complex operations securely

## Why Learn MCP?

### 1. **Future of AI Development**
MCP represents the next evolution in AI development, enabling AI agents to interact with the real world safely and effectively.

### 2. **Career Opportunities**
- **AI Engineer**: Building AI agent capabilities
- **Integration Developer**: Connecting AI with business systems
- **AI Tool Developer**: Creating specialized AI tools
- **Enterprise AI Architect**: Designing AI integration strategies

### 3. **Real-World Applications**
- **Business Automation**: AI-powered workflow automation
- **Data Analysis**: AI agents that can query and analyze data
- **Customer Service**: AI assistants with access to business systems
- **Content Creation**: AI tools that can access and process information

### 4. **Technical Benefits**
- **Standardized Interface**: Consistent way to build AI tools
- **Security First**: Built-in security and authentication
- **Performance**: Optimized for production use
- **Community**: Growing ecosystem of tools and resources

## Installing MCP SDK

### Prerequisites

Before installing MCP, ensure you have:

1. **Python 3.8+** (recommended) or **Node.js 16+**
2. **pip** (Python package manager) or **npm** (Node.js package manager)
3. **Git** (for version control)

### Python Installation

```bash
# Install MCP SDK for Python
pip install mcp

# Verify installation
python3 -c "import mcp; print('MCP SDK installed successfully')"
```

### Node.js Installation

```bash
# Install MCP SDK for Node.js
npm install @modelcontextprotocol/sdk

# Verify installation
node -e "console.log('MCP SDK installed successfully')"
```

### Go Installation

```bash
# Install MCP SDK for Go
go get github.com/modelcontextprotocol/go-sdk

# Verify installation
go run -c "fmt.Println('MCP SDK installed successfully')"
```

## Your First MCP Server

Let's create your first MCP server! We'll build a simple "Hello World" server that provides a basic tool.

### Python Example

Create a file called `hello_server.py`:

```python
#!/usr/bin/env python3
"""
Hello World MCP Server
A simple MCP server that demonstrates basic functionality.
"""

import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent

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
```

### Node.js Example

Create a file called `hello_server.js`:

```javascript
#!/usr/bin/env node
/**
 * Hello World MCP Server
 * A simple MCP server that demonstrates basic functionality.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

// Create the server instance
const server = new Server({
  name: "hello-server",
  version: "1.0.0"
});

// Define available tools
const tools = [
  {
    name: "greet",
    description: "A simple greeting tool",
    inputSchema: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "The name to greet",
          default: "World"
        }
      }
    }
  },
  {
    name: "echo",
    description: "Echo back the provided message",
    inputSchema: {
      type: "object",
      properties: {
        message: {
          type: "string",
          description: "The message to echo back"
        }
      },
      required: ["message"]
    }
  }
];

// Handle tool listing
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  switch (name) {
    case "greet":
      const name = args.name || "World";
      return {
        content: [{ 
          type: "text", 
          text: `Hello, ${name}! Welcome to MCP!` 
        }]
      };
      
    case "echo":
      return {
        content: [{ 
          type: "text", 
          text: `Echo: ${args.message}` 
        }]
      };
      
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// Start the server
async function main() {
  console.log("Starting Hello World MCP Server...");
  console.log("Available tools:");
  console.log("- greet: Greet someone");
  console.log("- echo: Echo a message");
  
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
```

### Running Your Server

```bash
# Python
python3 hello_server.py

# Node.js
node hello_server.js
```

## Understanding the Protocol

### MCP Protocol Basics

MCP uses JSON-RPC 2.0 for communication between clients and servers. The protocol defines several key concepts:

1. **Tools**: Functions that AI agents can call
2. **Resources**: Data sources that AI agents can access
3. **Prompts**: Templates for AI agent interactions
4. **Messages**: JSON-RPC requests and responses

### Protocol Flow

```
AI Client (Claude, ChatGPT, etc.)
    ↓ JSON-RPC requests
MCP Server (Your code)
    ↓ Tool execution
External Systems (APIs, databases, etc.)
    ↓ Results
MCP Server
    ↓ JSON-RPC responses
AI Client
```

### Message Types

- **tools/list**: List available tools
- **tools/call**: Execute a tool
- **resources/list**: List available resources
- **resources/read**: Read a resource
- **prompts/list**: List available prompts
- **prompts/get**: Get a prompt template

## Basic Server Structure

### Server Components

1. **Server Instance**: Main server object
2. **Tool Definitions**: Functions that can be called
3. **Resource Providers**: Data sources
4. **Error Handling**: Proper error responses
5. **Authentication**: Security mechanisms

### Tool Definition Structure

```python
@server.tool("tool_name")
async def tool_function(param1: str, param2: int = 10) -> str:
    """
    Tool description.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (optional)
    
    Returns:
        Description of return value
    """
    # Tool implementation
    return "result"
```

### Error Handling

```python
from mcp.types import ToolExecutionError, ErrorCode

try:
    result = await some_operation()
except Exception as e:
    raise ToolExecutionError(
        code=ErrorCode.INTERNAL_ERROR,
        message=f"Operation failed: {str(e)}"
    )
```

## Exercises

### Exercise 1: Basic Calculator Tool

Create an MCP server with a calculator tool that can perform basic arithmetic operations.

**Requirements:**
- Tool name: `calculate`
- Parameters: `operation` (add, subtract, multiply, divide), `a` (number), `b` (number)
- Return: Result of the calculation

**Hint:**
```python
@server.tool("calculate")
async def calculate_tool(operation: str, a: float, b: float) -> str:
    # Your implementation here
    pass
```

### Exercise 2: File Reader Tool

Create an MCP server with a tool that can read text files.

**Requirements:**
- Tool name: `read_file`
- Parameters: `file_path` (string)
- Return: File contents or error message
- Handle file not found errors

### Exercise 3: Weather Tool (Mock)

Create an MCP server with a mock weather tool that returns fake weather data.

**Requirements:**
- Tool name: `get_weather`
- Parameters: `location` (string)
- Return: Mock weather information (temperature, condition, humidity)

## 🎯 Module Summary

In this module, you learned:

- ✅ What MCP is and why it's important for AI development
- ✅ How to install MCP SDK for your preferred language
- ✅ How to create a basic MCP server
- ✅ Understanding of the MCP protocol flow
- ✅ Basic server structure and tool definitions

## 🚀 Next Steps

You're now ready to move on to **Module 2: MCP Fundamentals**, where you'll learn about:
- Core MCP concepts in detail
- Server-client architecture
- Resource management
- Error handling patterns

---

**Congratulations on completing Module 1! 🎉**

*Next: [Module 2: MCP Fundamentals](module-02-fundamentals/README.md)*
