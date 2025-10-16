# Database Query Tool MCP Server

A powerful database query MCP server that provides tools for database operations, query execution, and data analysis.

## Features

- Database connection management
- SQL query execution
- Data analysis and reporting
- Database schema exploration
- Transaction management
- Query optimization suggestions

## Installation

```bash
pip install mcp asyncpg sqlalchemy pandas
```

## Usage

```python
import asyncio
from database_tool_server import DatabaseToolMCPServer

async def main():
    server = DatabaseToolMCPServer(
        name="database-tool",
        database_url="postgresql://user:password@localhost/database"
    )
    
    await server.initialize()
    
    try:
        print("Database Tool MCP Server started")
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        await server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Available Tools

- `execute_query`: Execute SQL queries
- `get_table_schema`: Get table schema information
- `list_tables`: List all tables in the database
- `analyze_data`: Perform data analysis
- `export_data`: Export data to various formats
- `optimize_query`: Get query optimization suggestions
- `backup_table`: Backup table data
- `restore_table`: Restore table from backup

## Configuration

Set the `DATABASE_URL` environment variable to specify the database connection string.

## Security

The server includes SQL injection prevention and query validation to ensure database security.
