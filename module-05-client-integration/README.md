# Module 5: Client Integration

Welcome to Module 5! Now that you can build production-ready MCP servers, let's focus on the client side. This module covers MCP client development, AI application integration, performance optimization, and error handling strategies for client applications.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Develop MCP clients for various use cases
- Integrate MCP with AI applications effectively
- Optimize client performance and reliability
- Handle errors and implement recovery strategies
- Build user-friendly AI interfaces
- Implement advanced client patterns

## 📚 Topics Covered

1. [MCP Client Development](#mcp-client-development)
2. [AI Application Integration](#ai-application-integration)
3. [Performance Optimization](#performance-optimization)
4. [Error Handling and Recovery](#error-handling-and-recovery)
5. [User Interface Patterns](#user-interface-patterns)
6. [Advanced Client Patterns](#advanced-client-patterns)
7. [Exercises](#exercises)

---

## MCP Client Development

### Basic MCP Client

```python
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging

@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str

class MCPClient:
    """Basic MCP client implementation."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Connect to MCP server."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(headers=headers)
        
        # Discover available tools and resources
        await self.discover_tools()
        await self.discover_resources()
        
        self.logger.info(f"Connected to MCP server at {self.server_url}")
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def discover_tools(self):
        """Discover available tools from the server."""
        try:
            response = await self._make_request("tools/list")
            tools_data = response.get("tools", [])
            
            for tool_data in tools_data:
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    input_schema=tool_data.get("inputSchema", {})
                )
                self.tools[tool.name] = tool
            
            self.logger.info(f"Discovered {len(self.tools)} tools")
        
        except Exception as e:
            self.logger.error(f"Failed to discover tools: {e}")
            raise
    
    async def discover_resources(self):
        """Discover available resources from the server."""
        try:
            response = await self._make_request("resources/list")
            resources_data = response.get("resources", [])
            
            for resource_data in resources_data:
                resource = MCPResource(
                    uri=resource_data["uri"],
                    name=resource_data["name"],
                    description=resource_data["description"],
                    mime_type=resource_data.get("mimeType", "text/plain")
                )
                self.resources[resource.uri] = resource
            
            self.logger.info(f"Discovered {len(self.resources)} resources")
        
        except Exception as e:
            self.logger.error(f"Failed to discover resources: {e}")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        try:
            response = await self._make_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            
            # Extract content from response
            content = response.get("content", [])
            if content:
                return content[0].get("text", "")
            else:
                return response
        
        except Exception as e:
            self.logger.error(f"Tool call failed for {tool_name}: {e}")
            raise
    
    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server."""
        if uri not in self.resources:
            raise ValueError(f"Unknown resource: {uri}")
        
        try:
            response = await self._make_request("resources/read", {
                "uri": uri
            })
            
            # Extract content from response
            content = response.get("content", [])
            if content:
                return content[0].get("text", "")
            else:
                return response
        
        except Exception as e:
            self.logger.error(f"Resource read failed for {uri}: {e}")
            raise
    
    async def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a JSON-RPC request to the MCP server."""
        if not self.session:
            raise RuntimeError("Client not connected")
        
        request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/rpc",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                result = await response.json()
                
                if "error" in result:
                    raise Exception(f"RPC Error: {result['error']}")
                
                return result.get("result", {})
        
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def list_tools(self) -> List[MCPTool]:
        """List all available tools."""
        return list(self.tools.values())
    
    def list_resources(self) -> List[MCPResource]:
        """List all available resources."""
        return list(self.resources.values())
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        return self.tools.get(name)
    
    def get_resource(self, uri: str) -> Optional[MCPResource]:
        """Get a specific resource by URI."""
        return self.resources.get(uri)
```

### Advanced MCP Client with Caching

```python
import time
import hashlib
from typing import Dict, Any, Optional
from functools import wraps

class CachedMCPClient(MCPClient):
    """MCP client with caching capabilities."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None, cache_ttl: int = 300):
        super().__init__(server_url, api_key)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl
    
    def _generate_cache_key(self, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        key_data = {"method": method, "params": params}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() < entry["expires_at"]:
                return entry["result"]
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result."""
        self.cache[cache_key] = {
            "result": result,
            "expires_at": time.time() + self.cache_ttl
        }
    
    async def call_tool_cached(self, tool_name: str, arguments: Dict[str, Any], use_cache: bool = True) -> Any:
        """Call a tool with optional caching."""
        if not use_cache:
            return await self.call_tool(tool_name, arguments)
        
        cache_key = self._generate_cache_key("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for tool {tool_name}")
            return cached_result
        
        # Call tool and cache result
        result = await self.call_tool(tool_name, arguments)
        self._cache_result(cache_key, result)
        
        return result
    
    async def read_resource_cached(self, uri: str, use_cache: bool = True) -> Any:
        """Read a resource with optional caching."""
        if not use_cache:
            return await self.read_resource(uri)
        
        cache_key = self._generate_cache_key("resources/read", {"uri": uri})
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for resource {uri}")
            return cached_result
        
        # Read resource and cache result
        result = await self.read_resource(uri)
        self._cache_result(cache_key, result)
        
        return result
    
    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if time.time() >= entry["expires_at"]
        )
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries
        }
```

## AI Application Integration

### LangChain Integration

```python
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Type, Optional
from pydantic import BaseModel, Field

class MCPToolInput(BaseModel):
    """Input schema for MCP tools."""
    tool_name: str = Field(description="Name of the MCP tool to call")
    arguments: dict = Field(description="Arguments to pass to the tool")

class LangChainMCPTool(BaseTool):
    """LangChain tool wrapper for MCP tools."""
    
    name: str = "mcp_tool"
    description: str = "Call tools on an MCP server"
    args_schema: Type[BaseModel] = MCPToolInput
    
    def __init__(self, mcp_client: MCPClient):
        super().__init__()
        self.mcp_client = mcp_client
    
    def _run(self, tool_name: str, arguments: dict) -> str:
        """Synchronous version of tool execution."""
        # This would need to be implemented with asyncio.run() in practice
        return asyncio.run(self._arun(tool_name, arguments))
    
    async def _arun(self, tool_name: str, arguments: dict) -> str:
        """Asynchronous version of tool execution."""
        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            return str(result)
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

class MCPAgent:
    """AI agent that uses MCP tools."""
    
    def __init__(self, mcp_client: MCPClient, openai_api_key: str):
        self.mcp_client = mcp_client
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # Create MCP tool wrapper
        self.mcp_tool = LangChainMCPTool(mcp_client)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to various tools through MCP (Model Context Protocol).

Available tools:
{tools}

Use these tools to help users accomplish their tasks. When you need to use a tool, call it with the appropriate arguments.

Tool descriptions:
{tool_descriptions}

Always explain what you're doing and provide clear, helpful responses."""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=[self.mcp_tool],
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[self.mcp_tool],
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def run(self, query: str) -> str:
        """Run the agent with a query."""
        try:
            # Get tool descriptions
            tools = self.mcp_client.list_tools()
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}" for tool in tools
            ])
            
            result = await self.agent_executor.ainvoke({
                "input": query,
                "tools": [tool.name for tool in tools],
                "tool_descriptions": tool_descriptions
            })
            
            return result["output"]
        
        except Exception as e:
            return f"Error running agent: {str(e)}"

# Usage example
async def main():
    """Example usage of MCP agent."""
    async with CachedMCPClient("http://localhost:8000") as client:
        agent = MCPAgent(client, "your-openai-api-key")
        
        response = await agent.run("Create a new user named John Doe with email john@example.com")
        print(response)
```

### Custom AI Interface

```python
import streamlit as st
import asyncio
from typing import Dict, Any, List
import json

class StreamlitMCPInterface:
    """Streamlit interface for MCP client."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.session_state = st.session_state
    
    def render_tool_interface(self):
        """Render tool selection and execution interface."""
        st.header("MCP Tools")
        
        # Tool selection
        tools = self.mcp_client.list_tools()
        tool_names = [tool.name for tool in tools]
        
        selected_tool = st.selectbox("Select Tool", tool_names)
        
        if selected_tool:
            tool = self.mcp_client.get_tool(selected_tool)
            st.write(f"**Description:** {tool.description}")
            
            # Render input form based on tool schema
            self._render_tool_input_form(tool)
    
    def _render_tool_input_form(self, tool: MCPTool):
        """Render input form for a tool."""
        st.subheader(f"Execute {tool.name}")
        
        # Parse input schema
        properties = tool.input_schema.get("properties", {})
        required_fields = tool.input_schema.get("required", [])
        
        form_data = {}
        
        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "string")
            field_description = field_schema.get("description", "")
            is_required = field_name in required_fields
            
            if field_type == "string":
                if field_schema.get("format") == "email":
                    value = st.text_input(
                        f"{field_name} {'*' if is_required else ''}",
                        help=field_description,
                        type="email"
                    )
                elif "enum" in field_schema:
                    value = st.selectbox(
                        f"{field_name} {'*' if is_required else ''}",
                        field_schema["enum"],
                        help=field_description
                    )
                else:
                    value = st.text_input(
                        f"{field_name} {'*' if is_required else ''}",
                        help=field_description
                    )
            elif field_type == "number":
                value = st.number_input(
                    f"{field_name} {'*' if is_required else ''}",
                    help=field_description
                )
            elif field_type == "boolean":
                value = st.checkbox(
                    f"{field_name} {'*' if is_required else ''}",
                    help=field_description
                )
            elif field_type == "array":
                value = st.text_area(
                    f"{field_name} {'*' if is_required else ''} (JSON array)",
                    help=field_description,
                    placeholder='["item1", "item2"]'
                )
                try:
                    value = json.loads(value) if value else []
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for array field")
                    value = []
            else:
                value = st.text_input(
                    f"{field_name} {'*' if is_required else ''}",
                    help=field_description
                )
            
            form_data[field_name] = value
        
        # Execute button
        if st.button("Execute Tool"):
            # Validate required fields
            missing_fields = [field for field in required_fields if not form_data.get(field)]
            
            if missing_fields:
                st.error(f"Missing required fields: {', '.join(missing_fields)}")
            else:
                # Execute tool
                with st.spinner("Executing tool..."):
                    try:
                        result = asyncio.run(self.mcp_client.call_tool(tool.name, form_data))
                        st.success("Tool executed successfully!")
                        st.json(result)
                    except Exception as e:
                        st.error(f"Tool execution failed: {str(e)}")
    
    def render_resource_interface(self):
        """Render resource access interface."""
        st.header("MCP Resources")
        
        resources = self.mcp_client.list_resources()
        
        if resources:
            resource_uris = [resource.uri for resource in resources]
            selected_resource = st.selectbox("Select Resource", resource_uris)
            
            if selected_resource:
                resource = self.mcp_client.get_resource(selected_resource)
                st.write(f"**Name:** {resource.name}")
                st.write(f"**Description:** {resource.description}")
                st.write(f"**MIME Type:** {resource.mime_type}")
                
                if st.button("Read Resource"):
                    with st.spinner("Reading resource..."):
                        try:
                            content = asyncio.run(self.mcp_client.read_resource(selected_resource))
                            st.success("Resource read successfully!")
                            
                            # Display content based on MIME type
                            if resource.mime_type == "application/json":
                                try:
                                    json_content = json.loads(content)
                                    st.json(json_content)
                                except json.JSONDecodeError:
                                    st.text(content)
                            else:
                                st.text(content)
                        except Exception as e:
                            st.error(f"Resource read failed: {str(e)}")
        else:
            st.info("No resources available")
    
    def render_chat_interface(self):
        """Render chat interface for AI interaction."""
        st.header("AI Chat with MCP Tools")
        
        # Initialize chat history
        if "messages" not in self.session_state:
            self.session_state.messages = []
        
        # Display chat messages
        for message in self.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to do?"):
            # Add user message
            self.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # This would integrate with your AI agent
                        response = asyncio.run(self._generate_response(prompt))
                        st.markdown(response)
                        self.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        self.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate AI response using MCP tools."""
        # This would integrate with your AI agent implementation
        # For now, return a simple response
        return f"I received your message: '{prompt}'. I can help you use the available MCP tools to accomplish your tasks."

def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="MCP Client Interface", layout="wide")
    
    # Initialize MCP client
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = None
    
    # Server connection
    st.sidebar.header("MCP Server Connection")
    server_url = st.sidebar.text_input("Server URL", "http://localhost:8000")
    api_key = st.sidebar.text_input("API Key (optional)", type="password")
    
    if st.sidebar.button("Connect"):
        try:
            st.session_state.mcp_client = CachedMCPClient(server_url, api_key)
            asyncio.run(st.session_state.mcp_client.connect())
            st.sidebar.success("Connected to MCP server!")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {str(e)}")
    
    if st.session_state.mcp_client:
        # Main interface
        interface = StreamlitMCPInterface(st.session_state.mcp_client)
        
        tab1, tab2, tab3 = st.tabs(["Tools", "Resources", "Chat"])
        
        with tab1:
            interface.render_tool_interface()
        
        with tab2:
            interface.render_resource_interface()
        
        with tab3:
            interface.render_chat_interface()
    else:
        st.info("Please connect to an MCP server to get started.")

if __name__ == "__main__":
    main()
```

## Performance Optimization

### Connection Pooling

```python
import asyncio
from typing import Dict, Any, List
import aiohttp
from asyncio import Semaphore

class PooledMCPClient(MCPClient):
    """MCP client with connection pooling."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None, max_connections: int = 10):
        super().__init__(server_url, api_key)
        self.max_connections = max_connections
        self.connection_semaphore = Semaphore(max_connections)
        self.connector: Optional[aiohttp.TCPConnector] = None
    
    async def connect(self):
        """Connect with connection pooling."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Create connector with connection pooling
        self.connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=self.connector,
            timeout=timeout
        )
        
        # Discover tools and resources
        await self.discover_tools()
        await self.discover_resources()
        
        self.logger.info(f"Connected to MCP server with {self.max_connections} max connections")
    
    async def disconnect(self):
        """Disconnect and cleanup connections."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def call_tool_batch(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple tool calls concurrently."""
        async def execute_tool_call(tool_call):
            async with self.connection_semaphore:
                return await self.call_tool(tool_call["name"], tool_call["arguments"])
        
        tasks = [execute_tool_call(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def read_resources_batch(self, uris: List[str]) -> List[Any]:
        """Read multiple resources concurrently."""
        async def read_resource(uri):
            async with self.connection_semaphore:
                return await self.read_resource(uri)
        
        tasks = [read_resource(uri) for uri in uris]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
```

### Request Batching

```python
from typing import Dict, Any, List, Optional
import time
from dataclasses import dataclass

@dataclass
class BatchedRequest:
    """Batched request item."""
    method: str
    params: Dict[str, Any]
    future: asyncio.Future

class BatchedMCPClient(MCPClient):
    """MCP client with request batching."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None, batch_size: int = 10, batch_timeout: float = 0.1):
        super().__init__(server_url, api_key)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[BatchedRequest] = []
        self.batch_task: Optional[asyncio.Task] = None
    
    async def _process_batch(self):
        """Process a batch of requests."""
        if not self.pending_requests:
            return
        
        # Take up to batch_size requests
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        
        # Create batch request
        batch_requests = []
        for i, req in enumerate(batch):
            batch_requests.append({
                "jsonrpc": "2.0",
                "id": i + 1,
                "method": req.method,
                "params": req.params
            })
        
        try:
            # Send batch request
            async with self.session.post(
                f"{self.server_url}/rpc",
                json=batch_requests,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                results = await response.json()
                
                # Process results
                for i, result in enumerate(results):
                    if i < len(batch):
                        req = batch[i]
                        if "error" in result:
                            req.future.set_exception(Exception(f"RPC Error: {result['error']}"))
                        else:
                            req.future.set_result(result.get("result", {}))
        
        except Exception as e:
            # Set exception for all requests in batch
            for req in batch:
                req.future.set_exception(e)
    
    async def _batch_processor(self):
        """Background task to process batches."""
        while True:
            try:
                if self.pending_requests:
                    await self._process_batch()
                else:
                    await asyncio.sleep(self.batch_timeout)
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(1)
    
    async def connect(self):
        """Connect and start batch processor."""
        await super().connect()
        self.batch_task = asyncio.create_task(self._batch_processor())
    
    async def disconnect(self):
        """Disconnect and stop batch processor."""
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        await super().disconnect()
    
    async def _make_request_batched(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Make a batched request."""
        if not self.session:
            raise RuntimeError("Client not connected")
        
        # Create future for result
        future = asyncio.Future()
        
        # Add to pending requests
        request = BatchedRequest(method, params or {}, future)
        self.pending_requests.append(request)
        
        # Process batch if it's full
        if len(self.pending_requests) >= self.batch_size:
            await self._process_batch()
        
        # Wait for result
        return await future
    
    async def call_tool_batched(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool using batched requests."""
        return await self._make_request_batched("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    async def read_resource_batched(self, uri: str) -> Any:
        """Read a resource using batched requests."""
        return await self._make_request_batched("resources/read", {"uri": uri})
```

## Error Handling and Recovery

### Circuit Breaker Pattern

```python
import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3

class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker transitioning to CLOSED")
        else:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker transitioning to OPEN after {self.failure_count} failures")

class ResilientMCPClient(MCPClient):
    """MCP client with circuit breaker and retry logic."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(server_url, api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Circuit breaker for server calls
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3
        )
        self.circuit_breaker = CircuitBreaker(circuit_config)
    
    async def call_tool_with_retry(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call tool with retry logic and circuit breaker."""
        async def _call_tool():
            return await self.call_tool(tool_name, arguments)
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self.circuit_breaker.call(_call_tool)
            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Tool call failed (attempt {attempt + 1}), retrying in {self.retry_delay}s: {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Tool call failed after {self.max_retries} retries: {e}")
                    raise e
    
    async def read_resource_with_retry(self, uri: str) -> Any:
        """Read resource with retry logic and circuit breaker."""
        async def _read_resource():
            return await self.read_resource(uri)
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self.circuit_breaker.call(_read_resource)
            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Resource read failed (attempt {attempt + 1}), retrying in {self.retry_delay}s: {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    self.logger.error(f"Resource read failed after {self.max_retries} retries: {e}")
                    raise e
```

## User Interface Patterns

### Command Line Interface

```python
import click
import asyncio
import json
from typing import Dict, Any, List

class MCPCLI:
    """Command line interface for MCP client."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
    
    @click.group()
    def cli():
        """MCP Client CLI"""
        pass
    
    @cli.command()
    @click.option('--server-url', default='http://localhost:8000', help='MCP server URL')
    @click.option('--api-key', help='API key for authentication')
    def connect(server_url: str, api_key: str):
        """Connect to MCP server."""
        async def _connect():
            client = MCPClient(server_url, api_key)
            await client.connect()
            click.echo(f"Connected to {server_url}")
            return client
        
        asyncio.run(_connect())
    
    @cli.command()
    @click.option('--server-url', default='http://localhost:8000', help='MCP server URL')
    @click.option('--api-key', help='API key for authentication')
    def list_tools(server_url: str, api_key: str):
        """List available tools."""
        async def _list_tools():
            async with MCPClient(server_url, api_key) as client:
                tools = client.list_tools()
                
                click.echo("Available Tools:")
                for tool in tools:
                    click.echo(f"  - {tool.name}: {tool.description}")
        
        asyncio.run(_list_tools())
    
    @cli.command()
    @click.option('--server-url', default='http://localhost:8000', help='MCP server URL')
    @click.option('--api-key', help='API key for authentication')
    def list_resources(server_url: str, api_key: str):
        """List available resources."""
        async def _list_resources():
            async with MCPClient(server_url, api_key) as client:
                resources = client.list_resources()
                
                click.echo("Available Resources:")
                for resource in resources:
                    click.echo(f"  - {resource.uri}: {resource.name}")
        
        asyncio.run(_list_resources())
    
    @cli.command()
    @click.option('--server-url', default='http://localhost:8000', help='MCP server URL')
    @click.option('--api-key', help='API key for authentication')
    @click.argument('tool_name')
    @click.argument('arguments', required=False)
    def call_tool(server_url: str, api_key: str, tool_name: str, arguments: str):
        """Call a tool."""
        async def _call_tool():
            async with MCPClient(server_url, api_key) as client:
                try:
                    # Parse arguments
                    if arguments:
                        args = json.loads(arguments)
                    else:
                        args = {}
                    
                    result = await client.call_tool(tool_name, args)
                    click.echo(json.dumps(result, indent=2))
                
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
        
        asyncio.run(_call_tool())
    
    @cli.command()
    @click.option('--server-url', default='http://localhost:8000', help='MCP server URL')
    @click.option('--api-key', help='API key for authentication')
    @click.argument('uri')
    def read_resource(server_url: str, api_key: str, uri: str):
        """Read a resource."""
        async def _read_resource():
            async with MCPClient(server_url, api_key) as client:
                try:
                    result = await client.read_resource(uri)
                    click.echo(result)
                
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
        
        asyncio.run(_read_resource())

if __name__ == '__main__':
    MCPCLI.cli()
```

## Advanced Client Patterns

### Plugin System

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import importlib
import os

class MCPClientPlugin(ABC):
    """Base class for MCP client plugins."""
    
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @abstractmethod
    async def initialize(self, client: 'MCPClient') -> None:
        """Initialize plugin with client."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

class ToolCachePlugin(MCPClientPlugin):
    """Plugin for caching tool results."""
    
    def name(self) -> str:
        return "tool_cache"
    
    def version(self) -> str:
        return "1.0.0"
    
    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.client: Optional[MCPClient] = None
    
    async def initialize(self, client: 'MCPClient'):
        self.client = client
        # Override client's call_tool method
        original_call_tool = client.call_tool
        client.call_tool = self._cached_call_tool
    
    async def cleanup(self):
        self.cache.clear()
    
    async def _cached_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Cached version of call_tool."""
        cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() < entry["expires_at"]:
                return entry["result"]
            else:
                del self.cache[cache_key]
        
        # Call original method
        result = await self.client.call_tool(tool_name, arguments)
        
        # Cache result
        self.cache[cache_key] = {
            "result": result,
            "expires_at": time.time() + self.cache_ttl
        }
        
        return result

class MetricsPlugin(MCPClientPlugin):
    """Plugin for collecting metrics."""
    
    def name(self) -> str:
        return "metrics"
    
    def version(self) -> str:
        return "1.0.0"
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "tool_calls": 0,
            "resource_reads": 0,
            "errors": 0,
            "response_times": []
        }
        self.client: Optional[MCPClient] = None
    
    async def initialize(self, client: 'MCPClient'):
        self.client = client
        # Override client methods to collect metrics
        original_call_tool = client.call_tool
        original_read_resource = client.read_resource
        
        client.call_tool = self._instrumented_call_tool
        client.read_resource = self._instrumented_read_resource
    
    async def cleanup(self):
        pass
    
    async def _instrumented_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Instrumented version of call_tool."""
        start_time = time.time()
        
        try:
            result = await self.client.call_tool(tool_name, arguments)
            self.metrics["tool_calls"] += 1
            return result
        except Exception as e:
            self.metrics["errors"] += 1
            raise e
        finally:
            response_time = time.time() - start_time
            self.metrics["response_times"].append(response_time)
    
    async def _instrumented_read_resource(self, uri: str) -> Any:
        """Instrumented version of read_resource."""
        start_time = time.time()
        
        try:
            result = await self.client.read_resource(uri)
            self.metrics["resource_reads"] += 1
            return result
        except Exception as e:
            self.metrics["errors"] += 1
            raise e
        finally:
            response_time = time.time() - start_time
            self.metrics["response_times"].append(response_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        metrics = self.metrics.copy()
        
        if metrics["response_times"]:
            metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
            metrics["max_response_time"] = max(metrics["response_times"])
            metrics["min_response_time"] = min(metrics["response_times"])
        
        return metrics

class PluginManager:
    """Manager for MCP client plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, MCPClientPlugin] = {}
        self.client: Optional[MCPClient] = None
    
    def register_plugin(self, plugin: MCPClientPlugin):
        """Register a plugin."""
        self.plugins[plugin.name()] = plugin
    
    def load_plugins_from_directory(self, directory: str):
        """Load plugins from a directory."""
        if not os.path.exists(directory):
            return
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"{directory}.{module_name}")
                    
                    # Look for plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, MCPClientPlugin) and 
                            attr != MCPClientPlugin):
                            plugin = attr()
                            self.register_plugin(plugin)
                
                except Exception as e:
                    print(f"Failed to load plugin {module_name}: {e}")
    
    async def initialize_plugins(self, client: MCPClient):
        """Initialize all plugins with client."""
        self.client = client
        
        for plugin in self.plugins.values():
            try:
                await plugin.initialize(client)
                print(f"Initialized plugin: {plugin.name()} v{plugin.version()}")
            except Exception as e:
                print(f"Failed to initialize plugin {plugin.name()}: {e}")
    
    async def cleanup_plugins(self):
        """Cleanup all plugins."""
        for plugin in self.plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                print(f"Failed to cleanup plugin {plugin.name()}: {e}")

class PluginAwareMCPClient(MCPClient):
    """MCP client with plugin support."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        super().__init__(server_url, api_key)
        self.plugin_manager = PluginManager()
    
    def register_plugin(self, plugin: MCPClientPlugin):
        """Register a plugin."""
        self.plugin_manager.register_plugin(plugin)
    
    def load_plugins_from_directory(self, directory: str):
        """Load plugins from directory."""
        self.plugin_manager.load_plugins_from_directory(directory)
    
    async def connect(self):
        """Connect and initialize plugins."""
        await super().connect()
        await self.plugin_manager.initialize_plugins(self)
    
    async def disconnect(self):
        """Disconnect and cleanup plugins."""
        await self.plugin_manager.cleanup_plugins()
        await super().disconnect()
```

## Exercises

### Exercise 1: Advanced MCP Client

Build a comprehensive MCP client with the following features:

**Requirements:**
- Connection pooling and request batching
- Circuit breaker pattern for fault tolerance
- Comprehensive caching system
- Plugin architecture for extensibility
- Metrics collection and monitoring
- Error handling and recovery strategies

### Exercise 2: AI Integration Platform

Create a platform that integrates MCP with AI applications:

**Requirements:**
- LangChain integration for tool usage
- Custom AI agent implementation
- Web interface for tool interaction
- Chat interface for natural language interaction
- Tool recommendation system
- Performance optimization

### Exercise 3: Enterprise MCP Client

Build an enterprise-grade MCP client:

**Requirements:**
- Multi-server support with load balancing
- Authentication and authorization
- Audit logging and compliance
- Configuration management
- Monitoring and alerting
- High availability and failover

## 🎯 Module Summary

In this module, you learned:

- ✅ MCP client development patterns and best practices
- ✅ AI application integration strategies
- ✅ Performance optimization techniques
- ✅ Error handling and recovery mechanisms
- ✅ User interface patterns and implementations
- ✅ Advanced client patterns and plugin systems

## 🚀 Next Steps

You're now ready to move on to **Module 6: Real-World Applications**, where you'll learn about:
- Building practical MCP applications
- Integrating with external services
- Implementing business logic
- Creating production solutions

---

**Congratulations on completing Module 5! 🎉**

*Next: [Module 6: Real-World Applications](module-06-real-world-applications/README.md)*
