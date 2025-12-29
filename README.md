# Penguin Code

```
    ____                        _          ______          __
   / __ \___  ____  ____ ___  _(_)___     / ____/___  ____/ /__
  / /_/ / _ \/ __ \/ __ `/ / / / / __ \   / /   / __ \/ __  / _ \
 / ____/  __/ / / / /_/ / /_/ / / / / /  / /___/ /_/ / /_/ /  __/
/_/    \___/_/ /_/\__, /\__,_/_/_/ /_/   \____/\____/\__,_/\___/
                 /____/
```

**AI-powered coding assistant CLI and VS Code extension using Ollama**

[![CI](https://github.com/penguintechinc/penguin-code/workflows/CI/badge.svg)](https://github.com/penguintechinc/penguin-code/actions)
[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)

*Penguin Tech Inc © 2025*

## Features

- 🤖 **Multi-Agent System** - ChatAgent orchestrates specialized Explorer/Executor agents
- 🔍 **Multi-Engine Research** - 5 search engines with MCP protocol support
- 🧠 **Persistent Memory** - mem0 integration for context across sessions
- 📚 **Documentation RAG** - Auto-indexes docs for your project's languages and libraries
- 🔌 **MCP Integration** - Extend with N8N, Flowise, and custom MCP servers
- 🌐 **Client-Server Mode** - gRPC server for remote Ollama and team deployments
- ⚡ **GPU Optimized** - Smart model switching for RTX 4060 Ti (8GB VRAM) or higher
- 🐧 **Cross-Platform** - Works on Linux, macOS, and Windows

### Supported Languages

| Language | Detection | Doc Sources |
|----------|-----------|-------------|
| Python | `pyproject.toml`, `requirements.txt`, `*.py` | Official docs + PyPI libraries |
| JavaScript/TypeScript | `package.json`, `tsconfig.json` | MDN, npm packages |
| Go | `go.mod`, `*.go` | go.dev, pkg.go.dev |
| Rust | `Cargo.toml`, `*.rs` | docs.rs, crates.io |
| OpenTofu/Terraform | `*.tf`, `*.tofu`, `.terraform.lock.hcl` | OpenTofu docs, provider registries |
| Ansible | `ansible.cfg`, `playbook.yml`, `requirements.yml` | Ansible docs, Galaxy collections |

## Quick Start

```bash
# Install
pip install -e .
penguincode setup

# Pull required models
ollama pull llama3.2:3b qwen2.5-coder:7b nomic-embed-text

# Run
penguincode chat
```

**VS Code Extension**: Download VSIX from [Releases](https://github.com/penguintechinc/penguin-code/releases)

### Server Mode (Team Deployment)

```bash
# Start gRPC server (connects to local Ollama)
python -m penguincode.server.main

# Or use Docker
docker compose up -d

# Connect from client
penguincode chat --server localhost:50051
```

See [Architecture Documentation](docs/ARCHITECTURE.md) for remote deployment with TLS and authentication.

## Documentation

- **[Usage Guide](docs/USAGE.md)** - Installation, configuration, and usage
- **[Architecture](docs/ARCHITECTURE.md)** - Client-server architecture and deployment modes
- **[Agent Architecture](docs/AGENTS.md)** - ChatAgent, Explorer, Executor, Planner
- **[Tool Support](docs/TOOL_SUPPORT.md)** - Ollama models with native tool calling
- **[MCP Integration](docs/MCP.md)** - Extend with N8N, Flowise, and custom servers
- **[Memory](docs/MEMORY.md)** - Persistent memory with mem0 integration
- **[Documentation RAG](docs/DOCS_RAG.md)** - Project-aware documentation indexing
- **[Security](docs/SECURITY.md)** - Authentication, TLS, and secure code generation
- **[Contributing](docs/CONTRIBUTING.md)** - How to contribute

## License

AGPL-3.0 - See [LICENSE](LICENSE) for details

**Support**: [support.penguintech.io](https://support.penguintech.io) | **Homepage**: [www.penguintech.io](https://www.penguintech.io)
