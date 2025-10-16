#!/usr/bin/env python3
"""
Exercise 1: Task Management Server
A template for building a task management MCP server.

This is a starting template - complete the implementation following the module instructions.
"""

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
from mcp.server import Server
from mcp.types import ToolExecutionError, ErrorCode

@dataclass
class Task:
    """Task data model."""
    id: str
    title: str
    description: str
    priority: str  # "low", "medium", "high"
    status: str    # "pending", "in_progress", "completed"
    created_at: str
    updated_at: str

class TaskManagementServer:
    """MCP server for task management."""
    
    def __init__(self, name: str, data_file: str = "tasks.json"):
        self.name = name
        self.data_file = data_file
        self.tasks: Dict[str, Task] = {}
        self.server = Server(name)
        self._load_data()
        self._register_tools()
    
    def _load_data(self):
        """Load task data from file."""
        # TODO: Implement data loading
        pass
    
    def _save_data(self):
        """Save task data to file."""
        # TODO: Implement data saving
        pass
    
    def _register_tools(self):
        """Register task management tools."""
        
        @self.server.tool("create_task")
        async def create_task(title: str, description: str, priority: str = "medium") -> Dict[str, Any]:
            """Create a new task."""
            # TODO: Implement task creation
            # Requirements:
            # - Validate input parameters
            # - Check priority is valid ("low", "medium", "high")
            # - Generate unique task ID
            # - Create Task object
            # - Save to storage
            # - Return success response
            pass
        
        @self.server.tool("get_task")
        async def get_task(task_id: str) -> Dict[str, Any]:
            """Get task by ID."""
            # TODO: Implement task retrieval
            # Requirements:
            # - Check if task exists
            # - Return task data or error
            pass
        
        @self.server.tool("update_task")
        async def update_task(task_id: str, **updates) -> Dict[str, Any]:
            """Update task status or details."""
            # TODO: Implement task updates
            # Requirements:
            # - Validate task exists
            # - Validate update parameters
            # - Update task data
            # - Save changes
            # - Return updated task
            pass
        
        @self.server.tool("list_tasks")
        async def list_tasks(status: str = None, priority: str = None) -> Dict[str, Any]:
            """List all tasks with optional filtering."""
            # TODO: Implement task listing
            # Requirements:
            # - Filter by status if provided
            # - Filter by priority if provided
            # - Return list of tasks
            pass
        
        @self.server.tool("delete_task")
        async def delete_task(task_id: str) -> Dict[str, Any]:
            """Remove a task."""
            # TODO: Implement task deletion
            # Requirements:
            # - Check if task exists
            # - Remove from storage
            # - Return success confirmation
            pass
    
    async def run(self):
        """Run the server."""
        print(f"Starting {self.name}...")
        await self.server.run()

# Example usage
if __name__ == "__main__":
    server = TaskManagementServer("task-server")
    asyncio.run(server.run())
