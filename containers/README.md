# 🐳 MCP Course Containers

This directory contains Docker/Podman containers for each module, providing pre-configured environments so students can focus on learning MCP concepts without worrying about infrastructure setup.

## 📚 Available Containers

### Module 1: Introduction
- **Container**: `mcp-course-module1`
- **Purpose**: Basic MCP development environment
- **Includes**: Python 3.11, MCP SDK, basic tools
- **Usage**: `podman run -it mcp-course-module1`

### Module 2: Fundamentals  
- **Container**: `mcp-course-module2`
- **Purpose**: MCP fundamentals with validation tools
- **Includes**: Pydantic, validation libraries, example data
- **Usage**: `podman run -it mcp-course-module2`

### Module 3: Advanced Concepts
- **Container**: `mcp-course-module3`
- **Purpose**: Advanced MCP patterns and async operations
- **Includes**: Async libraries, streaming tools, mock APIs
- **Usage**: `podman run -it mcp-course-module3`

### Module 4: Basic Server Development
- **Container**: `mcp-course-module4`
- **Purpose**: Server development with file operations
- **Includes**: File system access, JSON storage, testing tools
- **Usage**: `podman run -it mcp-course-module4`

### Module 5: Client Integration
- **Container**: `mcp-course-module5`
- **Purpose**: MCP client development and AI integration
- **Includes**: Client libraries, mock AI services, integration tools
- **Usage**: `podman run -it mcp-course-module5`

### Module 6: Real-World Applications
- **Container**: `mcp-course-module6`
- **Purpose**: Real-world MCP applications
- **Includes**: Database connections, external APIs, business logic
- **Usage**: `podman run -it mcp-course-module6`

### Module 7: Practical Projects
- **Container**: `mcp-course-module7`
- **Purpose**: Complete project development
- **Includes**: Full-stack tools, project templates, deployment tools
- **Usage**: `podman run -it mcp-course-module7`

### Module 8: Production and Deployment
- **Container**: `mcp-course-module8`
- **Purpose**: Production deployment and monitoring
- **Includes**: Docker, Kubernetes tools, monitoring stack
- **Usage**: `podman run -it mcp-course-module8`

## 🚀 Quick Start

### Prerequisites
- Podman or Docker installed
- Git (for cloning the course)

### Build All Containers
```bash
# Build all containers
./build-all.sh

# Or build individual containers
./build-module1.sh
./build-module2.sh
# ... etc
```

### Run a Module Container
```bash
# Run module 1 container
podman run -it --rm -v $(pwd):/workspace mcp-course-module1

# Run module 4 with file system access
podman run -it --rm -v $(pwd):/workspace -v /tmp:/tmp mcp-course-module4
```

## 📁 Container Structure

Each container includes:
- **Python 3.11** with MCP SDK
- **Module-specific dependencies**
- **Pre-configured environment**
- **Example data and tools**
- **Testing frameworks**
- **Development tools**

## 🔧 Customization

### Adding Custom Tools
```bash
# Mount your custom tools
podman run -it --rm -v $(pwd)/my-tools:/app/tools mcp-course-module4
```

### Environment Variables
```bash
# Set custom environment variables
podman run -it --rm -e MCP_DEBUG=true -e LOG_LEVEL=DEBUG mcp-course-module4
```

## 📖 Usage Examples

### Module 1: Basic Calculator
```bash
podman run -it --rm mcp-course-module1
# Inside container:
python examples/calculator_server.py
```

### Module 4: File Manager
```bash
podman run -it --rm -v $(pwd):/workspace mcp-course-module4
# Inside container:
python examples/file_manager_server.py
```

### Module 6: Database Integration
```bash
podman run -it --rm -p 5432:5432 mcp-course-module6
# Container includes PostgreSQL pre-configured
```

## 🛠️ Development

### Building Custom Containers
```dockerfile
# Example: Custom module container
FROM mcp-course-module4:latest

# Add your custom tools
COPY my-tools/ /app/tools/
RUN pip install -r /app/tools/requirements.txt

# Set custom environment
ENV MCP_SERVER_NAME=my-custom-server
```

### Contributing
1. Fork the repository
2. Create your custom container
3. Test with the course modules
4. Submit a pull request

## 📚 Documentation

- [Container Specifications](specs/)
- [Build Scripts](scripts/)
- [Example Configurations](examples/)
- [Troubleshooting Guide](troubleshooting.md)

---

**Note**: These containers are designed to work with Podman (preferred) or Docker. The course assumes Podman is available based on the project setup.
