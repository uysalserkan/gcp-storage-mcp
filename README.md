# 🚀 Professional GCP Storage API

A comprehensive, enterprise-grade FastAPI server providing REST endpoints for Google Cloud Storage operations. Built with production-ready features including authentication, rate limiting, monitoring, caching, and structured logging.

![Logo](imgs/logo.png)

## ✨ Features

### 🔧 **Core API Features**
- **Complete GCS Operations**: Full REST API for buckets, blobs, and storage management
- **Batch Operations**: Efficient bulk upload/delete operations
- **Storage Analytics**: Comprehensive storage usage and performance metrics
- **Search & Filter**: Advanced blob search and size-based filtering
- **Signed URLs**: Temporary access URL generation with configurable expiration
- **Project Management**: Multi-project support and project switching

### 🔒 **Security & Authentication**
- **API Key Authentication**: Secure API key-based authentication system
- **Rate Limiting**: Configurable rate limiting with Redis support
- **CORS Protection**: Configurable cross-origin resource sharing
- **Security Headers**: Comprehensive security headers (CSP, HSTS, etc.)
- **Input Validation**: Extensive request validation and sanitization

### ⚡ **Performance & Monitoring**
- **Intelligent Caching**: In-memory caching with TTL and automatic invalidation
- **Request Tracking**: Unique request IDs for complete request tracing
- **Metrics Collection**: Built-in performance and usage metrics
- **Structured Logging**: JSON-based structured logging with request context
- **Health Checks**: Advanced health monitoring with dependency status

### 🛡️ **Production Ready**
- **Configuration Management**: Environment-based configuration system
- **Error Handling**: Comprehensive error responses with detailed context
- **Graceful Shutdown**: Proper application lifecycle management
- **Documentation**: Auto-generated OpenAPI documentation with examples
- **Monitoring Endpoints**: Built-in metrics and health check endpoints

## 📖 Usage

GCP Storage MCP can be used with Claude Desktop, Cursor, and more.

### 📦 Claude Desktop
**claude_desktop_config.json**
```json
"mcpServers": {
    "gcp-storage-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/uysalserkan/gcp-storage-mcp",
        "gcp-storage-mcp",
        "--credential_path",
        "your-credential-path.json"
      ]
    },
    ...
}
```

### 📦 Cursor


**.cursor/mcp.json**
```json
"mcpServers": {
    "gcp-storage-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/uysalserkan/gcp-storage-mcp",
        "gcp-storage-mcp",
        "--credential_path",
        "your-credential-path.json"
      ]
    },
    ...
}
```

## 📦 Quick Start FastAPI

### Prerequisites
- Python 3.11+
- Google Cloud SDK installed and configured
- Active Google Cloud Project with Cloud Storage API enabled
- Service Account with appropriate permissions

### Installation

1. **Clone and Setup**
```bash
git clone https://github.com/your-username/gcp-storage-mcp.git
cd gcp-storage-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configure GCP Authentication**
```bash
# Option 1: Service Account Key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Option 2: Application Default Credentials
gcloud auth application-default login
```

3. **Start the Server**
```bash
# Development mode
python gcp-storage-mcp/api.py

# Production mode with configuration
GCP_STORAGE_API_LOG_LEVEL=INFO python gcp-storage-mcp/api.py
```

4. **Access the API**
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

## 📖 API Documentation

### Core Endpoints

#### Health & Monitoring
- `GET /health` - Advanced health check with dependencies
- `GET /metrics` - System performance metrics
- `GET /` - API information and navigation

#### Project Management
- `GET /projects` - List all accessible GCP projects
- `GET /projects/current` - Get current project ID

#### Bucket Operations
- `GET /buckets` - List all buckets
- `POST /buckets/{bucket_name}` - Create bucket
- `DELETE /buckets/{bucket_name}` - Delete bucket
- `GET /buckets/{bucket_name}` - Get bucket information
- `GET /buckets/{bucket_name}/exists` - Check bucket existence

#### Blob Operations
- `GET /buckets/{bucket_name}/blobs` - List blobs with prefix filtering
- `POST /buckets/{bucket_name}/blobs/{blob_name}/upload` - Upload blob
- `POST /buckets/{bucket_name}/blobs/{blob_name}/download` - Download blob
- `DELETE /buckets/{bucket_name}/blobs/{blob_name}` - Delete blob
- `GET /buckets/{bucket_name}/blobs/{blob_name}` - Get blob information
- `GET /buckets/{bucket_name}/blobs/{blob_name}/url` - Get public URL
- `POST /buckets/{bucket_name}/blobs/{blob_name}/signed-url` - Generate signed URL

#### Advanced Operations
- `POST /buckets/{bucket_name}/blobs/{blob_name}/copy` - Copy blob
- `POST /buckets/{bucket_name}/blobs/{blob_name}/move` - Move blob
- `POST /buckets/{bucket_name}/batch/upload` - Batch upload
- `POST /buckets/{bucket_name}/batch/delete` - Batch delete

#### Analytics & Search
- `GET /buckets/{bucket_name}/analytics/storage` - Storage analytics
- `POST /buckets/{bucket_name}/search` - Search blobs by pattern
- `POST /buckets/{bucket_name}/filter` - Filter blobs by size

## 🔧 Usage Examples

### Basic Operations

#### Authentication with API Key
```bash
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     http://localhost:8000/health
```

#### List Buckets
```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/buckets
```

#### Upload a File
```bash
curl -X POST \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "/path/to/local/file.txt"}' \
     http://localhost:8000/buckets/my-bucket/blobs/path/to/file.txt/upload
```

#### Generate Signed URL
```bash
curl -X POST \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"expiration_hours": 24, "method": "GET"}' \
     http://localhost:8000/buckets/my-bucket/blobs/file.txt/signed-url
```

### Python Client Example

```python
import httpx
import asyncio

class GCPStorageClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}

    async def list_buckets(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/buckets",
                headers=self.headers
            )
            return response.json()

    async def upload_file(self, bucket: str, blob_name: str, file_path: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/buckets/{bucket}/blobs/{blob_name}/upload",
                headers=self.headers,
                json={"file_path": file_path}
            )
            return response.json()

# Usage
async def main():
    client = GCPStorageClient("http://localhost:8000", "your-api-key")

    # List buckets
    buckets = await client.list_buckets()
    print(f"Found {buckets['count']} buckets")

    # Upload file
    result = await client.upload_file(
        "my-bucket",
        "documents/file.pdf",
        "/local/path/file.pdf"
    )
    print(f"Upload completed: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📊 Monitoring & Observability

### Health Monitoring
The API provides comprehensive health checks:

```bash
# Basic health check
curl http://localhost:8000/health

# Response includes:
# - Overall status
# - GCP connection status
# - Dependency health
# - Performance metrics
# - Uptime information
```

### Metrics Collection
Built-in metrics endpoint provides:

```bash
curl http://localhost:8000/metrics

# Metrics include:
# - Request counts by endpoint
# - Response times
# - Error rates
# - Active requests
# - Cache hit rates
```

### Structured Logging
All requests are logged with structured JSON:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "GET",
  "path": "/buckets",
  "status_code": 200,
  "duration_ms": 45.2,
  "client_ip": "192.168.1.100"
}
```

## 🔒 Security Best Practices

### Authentication
- Use strong, randomly generated API keys
- Rotate API keys regularly
- Store keys securely (environment variables, secret managers)

### CORS Configuration
```bash
# Restrict origins in production
GCP_STORAGE_API_ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
```

### Rate Limiting
```bash
# Configure appropriate limits
GCP_STORAGE_API_RATE_LIMIT_DEFAULT="1000/hour"
GCP_STORAGE_API_RATE_LIMIT_STORAGE="redis://secure-redis:6379"
```

### GCP Permissions
Minimum required IAM roles:
- `roles/storage.objectViewer` - Read operations
- `roles/storage.objectCreator` - Upload operations  
- `roles/storage.admin` - Full management (production)

## 📈 Performance Features

### Intelligent Caching
- **Bucket Lists**: Cached for 5 minutes
- **Bucket Info**: Cached for 10 minutes  
- **Blob URLs**: Cached for 1 hour
- **Storage Analytics**: Cached for 30 minutes
- **Automatic Invalidation**: Cache cleared on data modifications

### Request Optimization
- **Parallel Processing**: Concurrent operations where possible
- **Batch Operations**: Efficient bulk operations
- **Connection Pooling**: Optimized GCP client connections
- **Request Tracking**: Complete request lifecycle monitoring

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/

# Run with coverage
pytest --cov=gcp-storage-mcp tests/

# Integration tests (requires GCP setup)
GCP_STORAGE_API_TEST_BUCKET="test-bucket" pytest tests/integration/
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new features
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- **API Documentation**: `/docs` endpoint (Swagger UI)
- **Google Cloud Storage**: [Official Documentation](https://cloud.google.com/storage/docs)
- **FastAPI**: [Official Documentation](https://fastapi.tiangolo.com/)
- **Rate Limiting**: [SlowAPI Documentation](https://github.com/laurentS/slowapi)

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/uysalserkan/gcp-storage-mcp/issues)
- 📧 **Email**: [uysalserkan08@gmail.com](mailto:uysalserkan08@gmail.com)

---

**Status**: ✅ Production Ready | **Version**: 1.0.0 | **Last Updated**: January 2025
