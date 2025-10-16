#!/bin/bash

# Build script for MCP Course containers
# This script builds all module containers using Podman

set -e

echo "🐳 Building MCP Course Containers..."

# Check if Podman is available
if ! command -v podman &> /dev/null; then
    echo "❌ Podman not found. Please install Podman first."
    exit 1
fi

# Build base container
echo "📦 Building base container..."
podman build -f Dockerfile.base -t mcp-course-base ..

# Build module containers
echo "📦 Building Module 1 container..."
podman build -f Dockerfile.module1 -t mcp-course-module1 ..

echo "📦 Building Module 4 container..."
podman build -f Dockerfile.module4 -t mcp-course-module4 ..

# Create additional module containers (simplified versions)
echo "📦 Creating additional module containers..."

# Module 2
podman build -f Dockerfile.module1 -t mcp-course-module2 ..

# Module 3  
podman build -f Dockerfile.module1 -t mcp-course-module3 ..

# Module 5
podman build -f Dockerfile.module1 -t mcp-course-module5 ..

# Module 6
podman build -f Dockerfile.module1 -t mcp-course-module6 ..

# Module 7
podman build -f Dockerfile.module1 -t mcp-course-module7 ..

# Module 8
podman build -f Dockerfile.module1 -t mcp-course-module8 ..

echo "✅ All containers built successfully!"
echo ""
echo "🚀 Available containers:"
echo "  - mcp-course-module1 (Introduction)"
echo "  - mcp-course-module2 (Fundamentals)"
echo "  - mcp-course-module3 (Advanced Concepts)"
echo "  - mcp-course-module4 (Basic Server Development)"
echo "  - mcp-course-module5 (Client Integration)"
echo "  - mcp-course-module6 (Real-World Applications)"
echo "  - mcp-course-module7 (Practical Projects)"
echo "  - mcp-course-module8 (Production and Deployment)"
echo ""
echo "📖 Usage example:"
echo "  podman run -it --rm -v \$(pwd):/workspace mcp-course-module4"
