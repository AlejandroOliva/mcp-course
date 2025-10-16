# Module 3: Advanced MCP Concepts

Welcome to Module 3! Now that you understand the fundamentals, let's explore advanced MCP concepts that will help you build sophisticated, production-ready MCP servers. This module covers custom tool development, advanced resource patterns, prompt templates, and security considerations.

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Develop custom tools with advanced patterns
- Implement sophisticated resource providers
- Use prompt templates effectively
- Handle streaming and async operations
- Implement comprehensive security measures
- Optimize performance and scalability

## 📚 Topics Covered

1. [Custom Tool Development](#custom-tool-development)
2. [Advanced Resource Providers](#advanced-resource-providers)
3. [Prompt Template System](#prompt-template-system)
4. [Streaming and Async Operations](#streaming-and-async-operations)
5. [Security Best Practices](#security-best-practices)
6. [Performance Optimization](#performance-optimization)
7. [Exercises](#exercises)

---

## Custom Tool Development

### Tool Composition Patterns

#### Chain of Responsibility Pattern

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ToolHandler(ABC):
    """Abstract base class for tool handlers."""
    
    def __init__(self, next_handler=None):
        self.next_handler = next_handler
    
    @abstractmethod
    async def can_handle(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if this handler can process the tool."""
        pass
    
    @abstractmethod
    async def handle(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Handle the tool execution."""
        pass
    
    async def process(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Process the tool or pass to next handler."""
        if await self.can_handle(tool_name, params):
            return await self.handle(tool_name, params)
        elif self.next_handler:
            return await self.next_handler.process(tool_name, params)
        else:
            raise ToolExecutionError(
                code=ErrorCode.METHOD_NOT_FOUND,
                message=f"No handler found for tool: {tool_name}"
            )

class DatabaseHandler(ToolHandler):
    """Handler for database operations."""
    
    async def can_handle(self, tool_name: str, params: Dict[str, Any]) -> bool:
        return tool_name.startswith("db_")
    
    async def handle(self, tool_name: str, params: Dict[str, Any]) -> Any:
        if tool_name == "db_query":
            return await self.execute_query(params["query"])
        elif tool_name == "db_insert":
            return await self.insert_data(params["table"], params["data"])
        # ... more database operations

class APIToolHandler(ToolHandler):
    """Handler for external API operations."""
    
    async def can_handle(self, tool_name: str, params: Dict[str, Any]) -> bool:
        return tool_name.startswith("api_")
    
    async def handle(self, tool_name: str, params: Dict[str, Any]) -> Any:
        if tool_name == "api_fetch":
            return await self.fetch_from_api(params["url"], params.get("headers", {}))
        # ... more API operations

# Usage in MCP server
class AdvancedMCPServer(Server):
    def __init__(self, name: str):
        super().__init__(name)
        self.tool_chain = APIToolHandler(
            DatabaseHandler(None)
        )
    
    @server.tool("dynamic_tool")
    async def dynamic_tool(tool_name: str, params: Dict[str, Any]) -> Any:
        """Dynamic tool dispatcher using chain of responsibility."""
        return await self.tool_chain.process(tool_name, params)
```

#### Factory Pattern for Tool Creation

```python
from typing import Type, Dict, Any
from abc import ABC, abstractmethod

class ToolFactory(ABC):
    """Abstract factory for creating tools."""
    
    @abstractmethod
    def create_tool(self, tool_type: str, config: Dict[str, Any]) -> Any:
        """Create a tool instance."""
        pass

class DatabaseToolFactory(ToolFactory):
    """Factory for database tools."""
    
    def create_tool(self, tool_type: str, config: Dict[str, Any]) -> Any:
        if tool_type == "query":
            return DatabaseQueryTool(config)
        elif tool_type == "insert":
            return DatabaseInsertTool(config)
        elif tool_type == "update":
            return DatabaseUpdateTool(config)
        else:
            raise ValueError(f"Unknown database tool type: {tool_type}")

class APIToolFactory(ToolFactory):
    """Factory for API tools."""
    
    def create_tool(self, tool_type: str, config: Dict[str, Any]) -> Any:
        if tool_type == "rest":
            return RESTAPITool(config)
        elif tool_type == "graphql":
            return GraphQLAPITool(config)
        elif tool_type == "websocket":
            return WebSocketAPITool(config)
        else:
            raise ValueError(f"Unknown API tool type: {tool_type}")

class ToolRegistry:
    """Registry for managing tool factories."""
    
    def __init__(self):
        self.factories: Dict[str, ToolFactory] = {}
    
    def register_factory(self, category: str, factory: ToolFactory):
        """Register a tool factory."""
        self.factories[category] = factory
    
    def create_tool(self, category: str, tool_type: str, config: Dict[str, Any]) -> Any:
        """Create a tool using the appropriate factory."""
        if category not in self.factories:
            raise ValueError(f"No factory registered for category: {category}")
        
        return self.factories[category].create_tool(tool_type, config)
```

### Advanced Tool Patterns

#### Middleware Pattern

```python
from typing import Callable, Any, Dict
import asyncio
import time

class ToolMiddleware:
    """Middleware for tool execution."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def __call__(self, tool_func: Callable, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute middleware logic."""
        raise NotImplementedError

class LoggingMiddleware(ToolMiddleware):
    """Middleware for logging tool execution."""
    
    async def __call__(self, tool_func: Callable, tool_name: str, params: Dict[str, Any]) -> Any:
        logger.info(f"Executing tool: {tool_name} with params: {params}")
        
        try:
            result = await tool_func(tool_name, params)
            logger.info(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

class TimingMiddleware(ToolMiddleware):
    """Middleware for timing tool execution."""
    
    async def __call__(self, tool_func: Callable, tool_name: str, params: Dict[str, Any]) -> Any:
        start_time = time.time()
        
        try:
            result = await tool_func(tool_name, params)
            execution_time = time.time() - start_time
            logger.info(f"Tool {tool_name} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool {tool_name} failed after {execution_time:.2f} seconds: {e}")
            raise

class RateLimitMiddleware(ToolMiddleware):
    """Middleware for rate limiting."""
    
    def __init__(self, name: str, max_calls: int, time_window: int):
        super().__init__(name)
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Dict[str, List[float]] = {}
    
    async def __call__(self, tool_func: Callable, tool_name: str, params: Dict[str, Any]) -> Any:
        client_id = params.get("client_id", "default")
        current_time = time.time()
        
        # Clean old calls
        if client_id in self.calls:
            self.calls[client_id] = [
                call_time for call_time in self.calls[client_id]
                if current_time - call_time < self.time_window
            ]
        else:
            self.calls[client_id] = []
        
        # Check rate limit
        if len(self.calls[client_id]) >= self.max_calls:
            raise ToolExecutionError(
                code=ErrorCode.RATE_LIMITED,
                message=f"Rate limit exceeded. Max {self.max_calls} calls per {self.time_window} seconds"
            )
        
        # Record this call
        self.calls[client_id].append(current_time)
        
        # Execute tool
        return await tool_func(tool_name, params)

class MiddlewareChain:
    """Chain of middleware to execute."""
    
    def __init__(self):
        self.middlewares: List[ToolMiddleware] = []
    
    def add_middleware(self, middleware: ToolMiddleware):
        """Add middleware to the chain."""
        self.middlewares.append(middleware)
    
    async def execute(self, tool_func: Callable, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute tool with all middleware."""
        async def execute_with_middleware(index: int):
            if index >= len(self.middlewares):
                return await tool_func(tool_name, params)
            
            middleware = self.middlewares[index]
            return await middleware(execute_with_middleware(index + 1), tool_name, params)
        
        return await execute_with_middleware(0)
```

## Advanced Resource Providers

### Dynamic Resource Discovery

```python
from typing import List, Dict, Any, Optional
import re

class ResourceProvider:
    """Base class for resource providers."""
    
    def __init__(self, base_uri: str):
        self.base_uri = base_uri
    
    async def list_resources(self) -> List[Resource]:
        """List all available resources."""
        raise NotImplementedError
    
    async def get_resource(self, uri: str) -> Resource:
        """Get a specific resource."""
        raise NotImplementedError
    
    def matches_uri(self, uri: str) -> bool:
        """Check if this provider handles the given URI."""
        return uri.startswith(self.base_uri)

class DatabaseResourceProvider(ResourceProvider):
    """Resource provider for database tables."""
    
    def __init__(self, base_uri: str, db_connection):
        super().__init__(base_uri)
        self.db = db_connection
    
    async def list_resources(self) -> List[Resource]:
        """List all database tables as resources."""
        resources = []
        
        # Get list of tables
        tables = await self.db.execute("SHOW TABLES")
        
        for table in tables:
            uri = f"{self.base_uri}/tables/{table['name']}"
            resources.append(Resource(
                uri=uri,
                name=f"Table: {table['name']}",
                description=f"Database table {table['name']}",
                mimeType="application/json"
            ))
        
        return resources
    
    async def get_resource(self, uri: str) -> Resource:
        """Get table data as a resource."""
        # Extract table name from URI
        match = re.match(rf"{re.escape(self.base_uri)}/tables/(.+)", uri)
        if not match:
            raise ResourceAccessError(
                code=ErrorCode.INVALID_PARAMS,
                message=f"Invalid table URI: {uri}"
            )
        
        table_name = match.group(1)
        
        # Get table schema
        schema = await self.db.execute(f"DESCRIBE {table_name}")
        
        # Get sample data (limit to 100 rows)
        data = await self.db.execute(f"SELECT * FROM {table_name} LIMIT 100")
        
        resource_data = {
            "table": table_name,
            "schema": schema,
            "data": data,
            "row_count": len(data)
        }
        
        return Resource(
            uri=uri,
            name=f"Table: {table_name}",
            description=f"Database table {table_name} with schema and sample data",
            mimeType="application/json",
            text=json.dumps(resource_data, indent=2, default=str)
        )

class FileSystemResourceProvider(ResourceProvider):
    """Resource provider for file system access."""
    
    def __init__(self, base_uri: str, root_path: str):
        super().__init__(base_uri)
        self.root_path = root_path
    
    async def list_resources(self) -> List[Resource]:
        """List files and directories as resources."""
        resources = []
        
        for root, dirs, files in os.walk(self.root_path):
            # Add directory resources
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                relative_path = os.path.relpath(dir_path, self.root_path)
                uri = f"{self.base_uri}/{relative_path.replace(os.sep, '/')}"
                
                resources.append(Resource(
                    uri=uri,
                    name=f"Directory: {dir_name}",
                    description=f"Directory {dir_name}",
                    mimeType="application/json"
                ))
            
            # Add file resources
            for file_name in files:
                file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(file_path, self.root_path)
                uri = f"{self.base_uri}/{relative_path.replace(os.sep, '/')}"
                
                # Determine MIME type
                mime_type = self._get_mime_type(file_path)
                
                resources.append(Resource(
                    uri=uri,
                    name=f"File: {file_name}",
                    description=f"File {file_name}",
                    mimeType=mime_type
                ))
        
        return resources
    
    async def get_resource(self, uri: str) -> Resource:
        """Get file content as a resource."""
        # Extract file path from URI
        relative_path = uri.replace(f"{self.base_uri}/", "")
        file_path = os.path.join(self.root_path, relative_path)
        
        # Security check - prevent path traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(self.root_path)):
            raise ResourceAccessError(
                code=ErrorCode.FORBIDDEN,
                message="Path traversal not allowed"
            )
        
        if not os.path.exists(file_path):
            raise ResourceAccessError(
                code=ErrorCode.NOT_FOUND,
                message=f"File not found: {relative_path}"
            )
        
        if os.path.isdir(file_path):
            # Return directory listing
            files = os.listdir(file_path)
            return Resource(
                uri=uri,
                name=f"Directory: {os.path.basename(file_path)}",
                description=f"Directory listing for {os.path.basename(file_path)}",
                mimeType="application/json",
                text=json.dumps({"files": files}, indent=2)
            )
        else:
            # Return file content
            mime_type = self._get_mime_type(file_path)
            
            if mime_type.startswith("text/"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return Resource(
                    uri=uri,
                    name=f"File: {os.path.basename(file_path)}",
                    description=f"File content for {os.path.basename(file_path)}",
                    mimeType=mime_type,
                    text=content
                )
            else:
                # For binary files, return metadata
                stat = os.stat(file_path)
                return Resource(
                    uri=uri,
                    name=f"File: {os.path.basename(file_path)}",
                    description=f"Binary file {os.path.basename(file_path)}",
                    mimeType=mime_type,
                    text=json.dumps({
                        "type": "binary",
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    }, indent=2)
                )
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for a file."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

class ResourceRegistry:
    """Registry for managing resource providers."""
    
    def __init__(self):
        self.providers: List[ResourceProvider] = []
    
    def register_provider(self, provider: ResourceProvider):
        """Register a resource provider."""
        self.providers.append(provider)
    
    async def list_all_resources(self) -> List[Resource]:
        """List resources from all providers."""
        all_resources = []
        
        for provider in self.providers:
            try:
                resources = await provider.list_resources()
                all_resources.extend(resources)
            except Exception as e:
                logger.error(f"Error listing resources from provider {provider.base_uri}: {e}")
        
        return all_resources
    
    async def get_resource(self, uri: str) -> Resource:
        """Get a resource from the appropriate provider."""
        for provider in self.providers:
            if provider.matches_uri(uri):
                try:
                    return await provider.get_resource(uri)
                except Exception as e:
                    logger.error(f"Error getting resource {uri} from provider {provider.base_uri}: {e}")
                    raise
        
        raise ResourceAccessError(
            code=ErrorCode.NOT_FOUND,
            message=f"No provider found for URI: {uri}"
        )
```

## Prompt Template System

### Template Engine

```python
from typing import Dict, Any, List, Optional
import re
from string import Template

class PromptTemplate:
    """Advanced prompt template system."""
    
    def __init__(self, template_id: str, template: str, variables: Dict[str, Any] = None):
        self.template_id = template_id
        self.template = template
        self.variables = variables or {}
        self.compiled_template = self._compile_template(template)
    
    def _compile_template(self, template: str) -> Template:
        """Compile template with custom syntax."""
        # Support for {{variable}} syntax
        template = re.sub(r'\{\{(\w+)\}\}', r'$\1', template)
        
        # Support for {{variable|default}} syntax
        template = re.sub(r'\{\{(\w+)\|([^}]+)\}\}', r'$\1 or "\2"', template)
        
        # Support for {{variable|filter}} syntax
        template = re.sub(r'\{\{(\w+)\|(\w+)\}\}', r'$\1', template)
        
        return Template(template)
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context."""
        try:
            # Merge template variables with context
            render_context = {**self.variables, **context}
            
            # Render the template
            rendered = self.compiled_template.safe_substitute(render_context)
            
            # Apply any filters
            rendered = self._apply_filters(rendered, render_context)
            
            return rendered
        except Exception as e:
            raise ValueError(f"Error rendering template {self.template_id}: {e}")
    
    def _apply_filters(self, text: str, context: Dict[str, Any]) -> str:
        """Apply template filters."""
        # Support for |upper, |lower, |title filters
        text = re.sub(r'\{\{(\w+)\|upper\}\}', lambda m: context.get(m.group(1), '').upper(), text)
        text = re.sub(r'\{\{(\w+)\|lower\}\}', lambda m: context.get(m.group(1), '').lower(), text)
        text = re.sub(r'\{\{(\w+)\|title\}\}', lambda m: context.get(m.group(1), '').title(), text)
        
        return text
    
    def get_required_variables(self) -> List[str]:
        """Get list of required variables."""
        variables = re.findall(r'\{\{(\w+)\}\}', self.template)
        return list(set(variables))

class PromptTemplateManager:
    """Manager for prompt templates."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
    
    def register_template(self, template: PromptTemplate):
        """Register a prompt template."""
        self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> PromptTemplate:
        """Get a prompt template."""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
        
        return self.templates[template_id]
    
    def render_template(self, template_id: str, context: Dict[str, Any]) -> str:
        """Render a template with context."""
        template = self.get_template(template_id)
        return template.render(context)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates."""
        return [
            {
                "id": template_id,
                "name": template.template_id,
                "description": template.template[:100] + "..." if len(template.template) > 100 else template.template,
                "required_variables": template.get_required_variables()
            }
            for template_id, template in self.templates.items()
        ]

# Usage in MCP server
class PromptMCPServer(Server):
    def __init__(self, name: str):
        super().__init__(name)
        self.template_manager = PromptTemplateManager()
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default prompt templates."""
        
        # Code generation template
        code_template = PromptTemplate(
            template_id="code_generation",
            template="""
Generate {{language}} code for the following requirements:

Requirements: {{requirements}}
Style: {{style|clean}}
Comments: {{comments|detailed}}

Please provide:
1. Complete, working code
2. Clear comments explaining the logic
3. Error handling where appropriate
4. Example usage if relevant

Code:
""",
            variables={"style": "clean", "comments": "detailed"}
        )
        self.template_manager.register_template(code_template)
        
        # Data analysis template
        analysis_template = PromptTemplate(
            template_id="data_analysis",
            template="""
Analyze the following data and provide insights:

Data: {{data}}
Analysis Type: {{analysis_type|comprehensive}}
Format: {{format|detailed}}

Please provide:
1. Key findings and patterns
2. Statistical insights
3. Visualizations recommendations
4. Actionable recommendations

Analysis:
""",
            variables={"analysis_type": "comprehensive", "format": "detailed"}
        )
        self.template_manager.register_template(analysis_template)
    
    @server.tool("generate_prompt")
    async def generate_prompt(template_id: str, context: Dict[str, Any]) -> str:
        """Generate a prompt using a template."""
        return self.template_manager.render_template(template_id, context)
    
    @server.tool("list_prompt_templates")
    async def list_prompt_templates() -> List[Dict[str, Any]]:
        """List all available prompt templates."""
        return self.template_manager.list_templates()
```

## Streaming and Async Operations

### Streaming Tool Responses

```python
from typing import AsyncGenerator, List
import asyncio
import json

class StreamingTool:
    """Tool that supports streaming responses."""
    
    async def stream_large_data(self, data_source: str, chunk_size: int = 1000) -> AsyncGenerator[str, None]:
        """Stream large data in chunks."""
        # Simulate reading from a large data source
        total_items = 10000  # Simulate large dataset
        
        for i in range(0, total_items, chunk_size):
            chunk = {
                "chunk_id": i // chunk_size,
                "data": list(range(i, min(i + chunk_size, total_items))),
                "progress": (i + chunk_size) / total_items * 100
            }
            
            yield json.dumps(chunk)
            
            # Yield control to allow other operations
            await asyncio.sleep(0.01)
    
    async def stream_file_processing(self, file_path: str) -> AsyncGenerator[str, None]:
        """Stream file processing results."""
        try:
            with open(file_path, 'r') as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    
                    # Process line
                    processed_line = self._process_line(line.strip())
                    
                    result = {
                        "line_number": line_number,
                        "original": line.strip(),
                        "processed": processed_line,
                        "status": "processed"
                    }
                    
                    yield json.dumps(result)
                    
                    # Yield control periodically
                    if line_number % 100 == 0:
                        await asyncio.sleep(0.01)
        
        except FileNotFoundError:
            yield json.dumps({"error": "File not found", "file_path": file_path})
        except Exception as e:
            yield json.dumps({"error": str(e), "file_path": file_path})
    
    def _process_line(self, line: str) -> str:
        """Process a single line of text."""
        # Simple processing example
        return line.upper().strip()

# Usage in MCP server
@server.tool("stream_data")
async def stream_data(data_source: str, chunk_size: int = 1000) -> List[str]:
    """Stream large data with progress updates."""
    streaming_tool = StreamingTool()
    results = []
    
    async for chunk in streaming_tool.stream_large_data(data_source, chunk_size):
        results.append(chunk)
    
    return results

@server.tool("process_file_stream")
async def process_file_stream(file_path: str) -> List[str]:
    """Process a file with streaming results."""
    streaming_tool = StreamingTool()
    results = []
    
    async for result in streaming_tool.stream_file_processing(file_path):
        results.append(result)
    
    return results
```

### Concurrent Operations

```python
import asyncio
from typing import List, Dict, Any
import aiohttp

class ConcurrentOperations:
    """Handle concurrent operations efficiently."""
    
    async def fetch_multiple_urls(self, urls: List[str], max_concurrent: int = 10) -> Dict[str, Any]:
        """Fetch multiple URLs concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    async with session.get(url, timeout=10) as response:
                        return {
                            "url": url,
                            "status": response.status,
                            "content": await response.text(),
                            "success": True
                        }
                except Exception as e:
                    return {
                        "url": url,
                        "error": str(e),
                        "success": False
                    }
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "total_urls": len(urls),
                "successful": sum(1 for r in results if isinstance(r, dict) and r.get("success", False)),
                "failed": sum(1 for r in results if isinstance(r, dict) and not r.get("success", False)),
                "results": results
            }
    
    async def process_database_queries(self, queries: List[Dict[str, Any]], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple database queries concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_query(query_info: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    # Simulate database query execution
                    await asyncio.sleep(0.1)  # Simulate query time
                    
                    return {
                        "query_id": query_info["id"],
                        "result": f"Query {query_info['id']} executed successfully",
                        "success": True
                    }
                except Exception as e:
                    return {
                        "query_id": query_info["id"],
                        "error": str(e),
                        "success": False
                    }
        
        tasks = [execute_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

# Usage in MCP server
@server.tool("concurrent_fetch")
async def concurrent_fetch(urls: List[str], max_concurrent: int = 10) -> Dict[str, Any]:
    """Fetch multiple URLs concurrently."""
    concurrent_ops = ConcurrentOperations()
    return await concurrent_ops.fetch_multiple_urls(urls, max_concurrent)

@server.tool("concurrent_queries")
async def concurrent_queries(queries: List[Dict[str, Any]], max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """Execute multiple database queries concurrently."""
    concurrent_ops = ConcurrentOperations()
    return await concurrent_ops.process_database_queries(queries, max_concurrent)
```

## Security Best Practices

### Input Validation and Sanitization

```python
from typing import Any, Dict, List
import re
import html
from urllib.parse import urlparse

class SecurityValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_sql_query(query: str) -> str:
        """Validate and sanitize SQL queries."""
        # Remove dangerous SQL keywords
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"Dangerous SQL keyword '{keyword}' not allowed")
        
        # Basic SQL injection prevention
        if any(char in query for char in [';', '--', '/*', '*/']):
            raise ValueError("Potentially dangerous SQL characters detected")
        
        return query.strip()
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> str:
        """Validate file paths to prevent path traversal."""
        # Normalize path
        normalized_path = os.path.normpath(file_path)
        
        # Check for path traversal attempts
        if '..' in normalized_path or normalized_path.startswith('/'):
            raise ValueError("Path traversal not allowed")
        
        # Check file extension if specified
        if allowed_extensions:
            file_ext = os.path.splitext(normalized_path)[1].lower()
            if file_ext not in allowed_extensions:
                raise ValueError(f"File extension '{file_ext}' not allowed")
        
        return normalized_path
    
    @staticmethod
    def validate_url(url: str, allowed_domains: List[str] = None) -> str:
        """Validate URLs to prevent SSRF attacks."""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                raise ValueError("Only HTTP and HTTPS URLs are allowed")
            
            # Check domain if specified
            if allowed_domains:
                if parsed.hostname not in allowed_domains:
                    raise ValueError(f"Domain '{parsed.hostname}' not allowed")
            
            # Check for localhost/private IPs
            if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                raise ValueError("Localhost URLs not allowed")
            
            return url
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}")
    
    @staticmethod
    def sanitize_html(html_content: str) -> str:
        """Sanitize HTML content."""
        # Escape HTML entities
        sanitized = html.escape(html_content)
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: URLs
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def validate_json_schema(data: Any, schema: Dict[str, Any]) -> Any:
        """Validate data against JSON schema."""
        # Simple schema validation (in production, use jsonschema library)
        for field, rules in schema.items():
            if field not in data:
                if rules.get('required', False):
                    raise ValueError(f"Required field '{field}' missing")
                continue
            
            value = data[field]
            field_type = rules.get('type')
            
            if field_type == 'string':
                if not isinstance(value, str):
                    raise ValueError(f"Field '{field}' must be a string")
                
                min_length = rules.get('minLength', 0)
                max_length = rules.get('maxLength', float('inf'))
                
                if len(value) < min_length:
                    raise ValueError(f"Field '{field}' too short (minimum {min_length})")
                if len(value) > max_length:
                    raise ValueError(f"Field '{field}' too long (maximum {max_length})")
            
            elif field_type == 'number':
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Field '{field}' must be a number")
                
                minimum = rules.get('minimum', float('-inf'))
                maximum = rules.get('maximum', float('inf'))
                
                if value < minimum:
                    raise ValueError(f"Field '{field}' too small (minimum {minimum})")
                if value > maximum:
                    raise ValueError(f"Field '{field}' too large (maximum {maximum})")
            
            elif field_type == 'array':
                if not isinstance(value, list):
                    raise ValueError(f"Field '{field}' must be an array")
                
                min_items = rules.get('minItems', 0)
                max_items = rules.get('maxItems', float('inf'))
                
                if len(value) < min_items:
                    raise ValueError(f"Field '{field}' has too few items (minimum {min_items})")
                if len(value) > max_items:
                    raise ValueError(f"Field '{field}' has too many items (maximum {max_items})")
        
        return data

# Usage in MCP server
class SecureMCPServer(Server):
    def __init__(self, name: str):
        super().__init__(name)
        self.validator = SecurityValidator()
    
    @server.tool("secure_query")
    async def secure_query(query: str) -> str:
        """Execute a secure database query."""
        # Validate SQL query
        validated_query = self.validator.validate_sql_query(query)
        
        # Execute query (implementation depends on your database)
        # result = await self.db.execute(validated_query)
        
        return f"Query executed: {validated_query}"
    
    @server.tool("secure_file_access")
    async def secure_file_access(file_path: str) -> str:
        """Securely access a file."""
        # Validate file path
        validated_path = self.validator.validate_file_path(
            file_path, 
            allowed_extensions=['.txt', '.json', '.csv']
        )
        
        # Read file
        with open(validated_path, 'r') as f:
            content = f.read()
        
        return content
    
    @server.tool("secure_url_fetch")
    async def secure_url_fetch(url: str) -> str:
        """Securely fetch content from a URL."""
        # Validate URL
        validated_url = self.validator.validate_url(
            url,
            allowed_domains=['api.example.com', 'data.example.com']
        )
        
        # Fetch content
        async with aiohttp.ClientSession() as session:
            async with session.get(validated_url) as response:
                return await response.text()
```

## Performance Optimization

### Caching Strategies

```python
from typing import Any, Dict, Optional
import time
import hashlib
import json
from functools import wraps

class CacheManager:
    """Advanced caching system for MCP tools."""
    
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key from tool name and parameters."""
        key_data = {
            "tool": tool_name,
            "params": params
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result."""
        key = self._generate_key(tool_name, params)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if entry has expired
            if time.time() < entry['expires_at']:
                return entry['value']
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, tool_name: str, params: Dict[str, Any], value: Any, ttl: Optional[int] = None) -> None:
        """Cache a result."""
        key = self._generate_key(tool_name, params)
        ttl = ttl or self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
    
    def invalidate(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Invalidate cached result."""
        key = self._generate_key(tool_name, params)
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if time.time() >= entry['expires_at']
        )
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "hit_rate": getattr(self, '_hit_rate', 0)
        }

def cached(cache_manager: CacheManager, ttl: Optional[int] = None):
    """Decorator for caching tool results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(tool_name: str, params: Dict[str, Any]) -> Any:
            # Check cache first
            cached_result = cache_manager.get(tool_name, params)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(tool_name, params)
            
            # Cache result
            cache_manager.set(tool_name, params, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Usage in MCP server
class OptimizedMCPServer(Server):
    def __init__(self, name: str):
        super().__init__(name)
        self.cache_manager = CacheManager(default_ttl=300)  # 5 minutes default
    
    @server.tool("expensive_calculation")
    @cached(cache_manager, ttl=600)  # Cache for 10 minutes
    async def expensive_calculation(data: List[int]) -> Dict[str, Any]:
        """Perform expensive calculation with caching."""
        # Simulate expensive operation
        await asyncio.sleep(2)
        
        result = {
            "sum": sum(data),
            "average": sum(data) / len(data),
            "max": max(data),
            "min": min(data),
            "count": len(data)
        }
        
        return result
    
    @server.tool("api_data")
    @cached(cache_manager, ttl=1800)  # Cache for 30 minutes
    async def api_data(endpoint: str) -> Dict[str, Any]:
        """Fetch data from external API with caching."""
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint) as response:
                data = await response.json()
        
        return data
```

## Exercises

### Exercise 1: Advanced Tool Framework

Create a comprehensive tool framework with the following features:

**Requirements:**
- Chain of responsibility pattern for tool handling
- Middleware support (logging, timing, rate limiting)
- Tool factory for dynamic tool creation
- Comprehensive error handling
- Input validation and sanitization

### Exercise 2: Multi-Provider Resource System

Build a resource system that supports multiple providers:

**Requirements:**
- Database resource provider
- File system resource provider
- API resource provider
- Dynamic resource discovery
- Resource caching and optimization
- Security controls

### Exercise 3: Streaming Data Processor

Create a streaming data processing system:

**Requirements:**
- Stream large datasets efficiently
- Real-time progress updates
- Error recovery and retry logic
- Concurrent processing
- Memory optimization

## 🎯 Module Summary

In this module, you learned:

- ✅ Advanced tool development patterns and architectures
- ✅ Sophisticated resource provider systems
- ✅ Prompt template management and rendering
- ✅ Streaming and async operation handling
- ✅ Comprehensive security best practices
- ✅ Performance optimization techniques

## 🚀 Next Steps

You're now ready to move on to **Module 4: Server Development**, where you'll learn about:
- Building production-ready MCP servers
- Complex tool implementation strategies
- Configuration and deployment systems
- Testing methodologies and best practices

---

**Congratulations on completing Module 3! 🎉**

*Next: [Module 4: Server Development](module-04-server-development/README.md)*
