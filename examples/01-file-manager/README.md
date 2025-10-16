# File Manager MCP Server

A comprehensive file management MCP server that provides tools for file operations, directory management, and file system monitoring.

## Features

- File upload and download
- Directory listing and navigation
- File search and filtering
- File metadata management
- File system monitoring
- Batch operations

## Installation

```bash
pip install mcp aiofiles
```

## Usage

```python
import asyncio
from file_manager_server import FileManagerMCPServer

async def main():
    server = FileManagerMCPServer(
        name="file-manager",
        base_directory="/path/to/base/directory"
    )
    
    await server.initialize()
    
    try:
        print("File Manager MCP Server started")
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        await server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Available Tools

- `upload_file`: Upload a file to the server
- `download_file`: Download a file from the server
- `list_files`: List files in a directory
- `search_files`: Search for files by name or content
- `get_file_info`: Get detailed file information
- `delete_file`: Delete a file
- `create_directory`: Create a new directory
- `monitor_directory`: Monitor directory for changes

## Configuration

Set the `BASE_DIRECTORY` environment variable to specify the base directory for file operations.

## Security

The server includes security measures to prevent path traversal attacks and unauthorized access.
