#!/usr/bin/env python3
"""
Exercise 3: File Manager Server Template
A template for building a file management MCP server.

This is a starting template - complete the implementation following the module instructions.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from mcp.server import Server
from mcp.types import ToolExecutionError, ErrorCode

class FileManagerServer:
    """MCP server for file management operations."""
    
    def __init__(self, name: str, base_path: str = "."):
        self.name = name
        self.base_path = Path(base_path).resolve()
        self.server = Server(name)
        self._register_tools()
    
    def _validate_path(self, file_path: str) -> Path:
        """Validate and secure file path."""
        # TODO: Implement path validation
        # Requirements:
        # - Prevent path traversal attacks
        # - Ensure path is within base_path
        # - Return resolved Path object
        pass
    
    def _register_tools(self):
        """Register file operation tools."""
        
        @self.server.tool("read_file")
        async def read_file(file_path: str) -> Dict[str, Any]:
            """Read a text file safely."""
            # TODO: Implement file reading
            # Requirements:
            # - Validate file path
            # - Check file exists
            # - Check file is readable
            # - Read file content
            # - Return content and metadata
            pass
        
        @self.server.tool("write_file")
        async def write_file(file_path: str, content: str) -> Dict[str, Any]:
            """Write content to a file."""
            # TODO: Implement file writing
            # Requirements:
            # - Validate file path
            # - Create directory if needed
            # - Write content to file
            # - Return success confirmation
            pass
        
        @self.server.tool("list_directory")
        async def list_directory(directory: str = ".") -> Dict[str, Any]:
            """List files and directories."""
            # TODO: Implement directory listing
            # Requirements:
            # - Validate directory path
            # - List files and directories
            # - Include file metadata (size, type)
            # - Return structured list
            pass
        
        @self.server.tool("create_directory")
        async def create_directory(directory: str) -> Dict[str, Any]:
            """Create a new directory."""
            # TODO: Implement directory creation
            # Requirements:
            # - Validate directory path
            # - Create directory
            # - Handle existing directories
            # - Return success confirmation
            pass
        
        @self.server.tool("delete_file")
        async def delete_file(file_path: str) -> Dict[str, Any]:
            """Delete a file safely."""
            # TODO: Implement file deletion
            # Requirements:
            # - Validate file path
            # - Check file exists
            # - Delete file
            # - Return success confirmation
            pass
        
        @self.server.tool("get_file_info")
        async def get_file_info(file_path: str) -> Dict[str, Any]:
            """Get file metadata."""
            # TODO: Implement file info retrieval
            # Requirements:
            # - Validate file path
            # - Get file statistics
            # - Return metadata (size, modified time, etc.)
            pass
    
    async def run(self):
        """Run the server."""
        print(f"Starting {self.name}...")
        print(f"Base path: {self.base_path}")
        await self.server.run()

# Example usage
if __name__ == "__main__":
    server = FileManagerServer("file-manager-server", "/tmp")
    asyncio.run(server.run())
