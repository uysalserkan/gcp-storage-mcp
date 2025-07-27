"""GCP Storage FastAPI Server.

A professional FastAPI server that provides REST endpoints for Google Cloud Storage operations.
Exposes comprehensive GCS functionality through a clean, documented REST API with enterprise-grade
security, monitoring, and performance optimizations.
"""

import hashlib
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from models import (
    ProjectsResponse, CurrentProjectResponse, BucketsResponse, BucketCreateResponse,
    BucketDeleteResponse, BucketInfoResponse, BucketExistsResponse, BlobsResponse, FilterResponse,
    ErrorResponse, BucketCreateRequest, BucketDeleteRequest, FoldersResponse, BlobOperationResponse,
    BlobInfoResponse, BlobUrlResponse, SignedUrlResponse, BatchResponse, StorageInfoResponse,
    SearchResponse, BlobUploadRequest, BlobDownloadRequest, BlobCopyRequest, BlobMoveRequest,
    SignedUrlRequest, BatchUploadRequest, BatchDeleteRequest, SearchBlobsRequest, FilterBlobsRequest,
)

# Add the current directory to the path to import our provider
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from provider import (
    GCPStorageProvider,
    GCPProjectProvider,
    ProviderConfig,
    ValidationError,
    OperationError,
    GCPStorageError
)


# ========== CONFIGURATION ==========

class APISettings(BaseSettings):
    """Application configuration with environment variable support."""

    # Server Configuration
    app_name: str = "GCP Storage API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Security Configuration
    api_key: Optional[str] = None
    allowed_origins: List[str] = ["*"]
    allowed_hosts: List[str] = ["*"]
    cors_allow_credentials: bool = True
    auth_required: bool = False

    # Rate Limiting
    rate_limit_enabled: bool = False
    rate_limit_default: str = "100/minute"
    rate_limit_storage: str = "redis://localhost:6379"

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
    log_file: Optional[str] = None
    request_logging: bool = True

    # Performance Configuration
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_request_size: int = 100 * 1024 * 1024  # 100MB

    # Monitoring
    metrics_enabled: bool = True
    health_check_interval: int = 30

    # GCP Configuration
    gcp_project_id: Optional[str] = None
    gcp_credentials_path: Optional[str] = None

    @field_validator('allowed_origins')
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    @field_validator('allowed_hosts')
    def parse_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',')]
        return v

    class Config:
        env_prefix = "GCP_STORAGE_API_"
        env_file = ".env"
        case_sensitive = False


# Initialize settings
settings = APISettings()

# ========== STRUCTURED LOGGING ==========

def setup_logging():
    """Configure structured logging."""
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
        filename=settings.log_file
    )

    return structlog.get_logger()


logger = setup_logging()

# ========== METRICS AND MONITORING ==========

class MetricsCollector:
    """Simple in-memory metrics collector."""

    def __init__(self):
        self.request_count = defaultdict(int)
        self.request_duration = defaultdict(list)
        self.error_count = defaultdict(int)
        self.active_requests = 0
        self.start_time = time.time()

    def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record request metrics."""
        key = f"{method}:{path}"
        self.request_count[key] += 1
        self.request_duration[key].append(duration)

        if status_code >= 400:
            self.error_count[key] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        total_requests = sum(self.request_count.values())
        total_errors = sum(self.error_count.values())

        avg_durations = {}
        for key, durations in self.request_duration.items():
            if durations:
                avg_durations[key] = sum(durations) / len(durations)

        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "active_requests": self.active_requests,
            "request_count_by_endpoint": dict(self.request_count),
            "average_response_time": avg_durations,
        }


metrics = MetricsCollector() if settings.metrics_enabled else None

# ========== AUTHENTICATION & SECURITY ==========

# Security schemes
security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> Optional[str]:
    """Verify API key authentication."""
    if not settings.auth_required:
        return None

    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication not configured"
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    # Simple API key validation (in production, use proper key management)
    if api_key != settings.api_key:
        logger.warning("Invalid API key attempt", api_key_hash=hashlib.sha256(api_key.encode()).hexdigest()[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key


async def verify_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    api_key: Optional[str] = Depends(verify_api_key)
) -> Optional[str]:
    """Combined authentication verification."""
    # For now, return the API key if present, otherwise allow
    return api_key


# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.rate_limit_storage if settings.rate_limit_enabled else "memory://",
    enabled=settings.rate_limit_enabled
)

# ========== MIDDLEWARE ==========

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for request tracking, logging, and metrics."""

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()
        request.state.start_time = start_time

        # Increment active requests
        if metrics:
            metrics.active_requests += 1

        # Enhanced request logging
        if settings.request_logging:
            logger.info(
                "Request started",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                query_params=dict(request.query_params),
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            if metrics:
                metrics.record_request(
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration
                )
                metrics.active_requests -= 1

            # Enhanced response logging
            if settings.request_logging:
                logger.info(
                    "Request completed",
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=round(duration * 1000, 2),
                )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Record error metrics
            if metrics:
                metrics.record_request(
                    request.method,
                    request.url.path,
                    500,
                    duration
                )
                metrics.active_requests -= 1

            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
                exc_info=True
            )

            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        })

        return response


# ========== CACHING ==========

class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not settings.cache_enabled:
            return None

        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry["expires"]:
                return entry["value"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if not settings.cache_enabled:
            return

        ttl = ttl or self.default_ttl
        self.cache[key] = {
            "value": value,
            "expires": time.time() + ttl
        }

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


cache = SimpleCache(settings.cache_ttl)

# Global provider instances
storage_provider: Optional[GCPStorageProvider] = None
project_provider: Optional[GCPProjectProvider] = None

# ========== LIFECYCLE MANAGEMENT ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with enhanced startup/shutdown."""
    # Startup
    logger.info("Starting GCP Storage FastAPI Server", version=settings.app_version)

    try:
        await initialize_providers()
        logger.info("Server startup complete",
                   host=settings.host,
                   port=settings.port,
                   debug=settings.debug)
    except Exception as e:
        logger.error("Failed to start server", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down GCP Storage FastAPI Server...")

    # Clear cache
    if cache:
        cache.clear()

    logger.info("Server shutdown complete")


# ========== APPLICATION SETUP ==========

app = FastAPI(
    title=settings.app_name,
    description="""
    🚀 **Professional GCP Storage API**

    A comprehensive, enterprise-grade REST API for Google Cloud Storage operations with:

    ## 🔧 **Core Features**
    - **Bucket Management**: Create, delete, list, and manage GCS buckets
    - **Blob Operations**: Upload, download, delete, copy, and move files
    - **Batch Operations**: Bulk upload/delete for efficient processing
    - **Analytics & Search**: Storage analytics and advanced blob search
    - **Signed URLs**: Temporary access URL generation
    - **Project Management**: Multi-project support

    ## 🔒 **Security & Performance**
    - API key authentication with rate limiting
    - CORS and security headers
    - Request/response logging with structured JSON
    - Caching for improved performance
    - Comprehensive error handling
    - Request tracing with unique IDs

    ## 📊 **Monitoring & Observability**
    - Built-in metrics collection
    - Health checks and system status
    - Performance monitoring
    - Structured logging

    ## 🔗 **Quick Start**
    1. Ensure GCP credentials are configured
    2. Set environment variables (see documentation)
    3. Start making requests to the endpoints below

    All endpoints include detailed error responses, validation, and comprehensive documentation.
    """,
    version=settings.app_version,
    contact={
        "name": "GCP Storage MCP",
        "email": "uysalserkan08@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    debug=settings.debug,
)

# Add middleware (order matters!)
if settings.rate_limit_enabled:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestTrackingMiddleware)

# ========== UTILITY FUNCTIONS ==========

async def initialize_providers(config: Optional[ProviderConfig] = None) -> None:
    """Initialize the GCP Storage and Project providers with enhanced error handling."""
    global storage_provider, project_provider

    try:
        logger.info("Initializing GCP Storage providers...")

        # Create provider config with settings
        if not config:
            config = ProviderConfig()
            if settings.gcp_project_id:
                config.project_id = settings.gcp_project_id
            if settings.gcp_credentials_path:
                config.credentials_path = settings.gcp_credentials_path

        storage_provider = GCPStorageProvider(config)
        project_provider = GCPProjectProvider()

        # Verify connection
        current_project = storage_provider.get_current_project_id()

        logger.info("Providers initialized successfully",
                   project_id=current_project,
                   auth_type="service_account" if settings.gcp_credentials_path else "default")

    except Exception as e:
        logger.error("Failed to initialize providers", error=str(e), exc_info=True)
        raise


def get_storage_provider() -> GCPStorageProvider:
    """Dependency to get storage provider with enhanced error handling."""
    if storage_provider is None:
        logger.error("Storage provider not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Storage provider not initialized",
                "error_type": "service_unavailable",
                "suggestion": "Server may be starting up or misconfigured"
            }
        )
    return storage_provider


def get_project_provider() -> GCPProjectProvider:
    """Dependency to get project provider with enhanced error handling."""
    if project_provider is None:
        logger.error("Project provider not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Project provider not initialized",
                "error_type": "service_unavailable",
                "suggestion": "Server may be starting up or misconfigured"
            }
        )
    return project_provider


def handle_provider_exception(e: Exception, operation: str, request_id: Optional[str] = None) -> HTTPException:
    """Enhanced provider exception handling with request tracking."""
    error_context = {
        "operation": operation,
        "error": str(e),
        "request_id": request_id
    }

    if isinstance(e, ValidationError):
        logger.error("Validation error", **error_context, error_type="validation")
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "error_type": "validation",
                "operation": operation,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    elif isinstance(e, OperationError):
        logger.error("Operation error", **error_context, error_type="operation")
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": str(e),
                "error_type": "operation",
                "operation": operation,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    elif isinstance(e, GCPStorageError):
        logger.error("GCP storage error", **error_context, error_type="gcp_storage")
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": str(e),
                "error_type": "gcp_storage",
                "operation": operation,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    else:
        logger.error("Unexpected error", **error_context, error_type="unexpected", exc_info=True)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "An unexpected error occurred",
                "error_type": "unexpected",
                "operation": operation,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, 'request_id', 'unknown')


def cache_key(operation: str, *args) -> str:
    """Generate cache key for operation."""
    key_parts = [operation] + [str(arg) for arg in args]
    return ":".join(key_parts)


# ========== HEALTH CHECK & MONITORING ENDPOINTS ==========

class AdvancedHealthResponse(BaseModel):
    """Enhanced health response model."""
    status: str = Field(..., description="Overall health status")
    project_id: str = Field(..., description="Current GCP project ID")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    dependencies: Dict[str, str] = Field(..., description="Dependency health status")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")


@app.get(
    "/health",
    response_model=AdvancedHealthResponse,
    summary="🏥 Advanced Health Check",
    description="Comprehensive health check including dependencies and performance metrics",
    tags=["Health & Monitoring"]
)
@limiter.limit(settings.rate_limit_default)
async def health_check(
    request: Request,
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> AdvancedHealthResponse:
    """Enhanced health check endpoint with dependency verification."""
    try:
        request_id = get_request_id(request)
        start_time = time.time()

        # Check GCP connection
        project_id = storage.get_current_project_id()

        # Check dependencies
        dependencies = {
            "gcp_storage": "healthy",
            "project_provider": "healthy" if project_provider else "unhealthy"
        }

        # Performance metrics
        check_duration = time.time() - start_time
        performance = {
            "health_check_duration_ms": round(check_duration * 1000, 2),
            "cache_enabled": settings.cache_enabled,
            "rate_limiting_enabled": settings.rate_limit_enabled
        }

        if metrics:
            performance.update(metrics.get_metrics())

        response = AdvancedHealthResponse(
            status="healthy",
            project_id=project_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=settings.app_version,
            uptime_seconds=time.time() - (metrics.start_time if metrics else start_time),
            dependencies=dependencies,
            performance=performance
        )

        logger.info("Health check completed",
                   request_id=request_id,
                   status="healthy",
                   project_id=project_id,
                   duration_ms=performance["health_check_duration_ms"])

        return response

    except Exception as e:
        request_id = get_request_id(request)
        logger.error("Health check failed",
                    request_id=request_id,
                    error=str(e),
                    exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@app.get(
    "/metrics",
    summary="📊 System Metrics",
    description="Get detailed system performance and usage metrics",
    tags=["Health & Monitoring"]
)
@limiter.limit("20/minute")
async def get_metrics(
    request: Request,
    _: Optional[str] = Depends(verify_auth)
):
    """Get system metrics endpoint."""
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Metrics collection is disabled"
        )

    return {
        "metrics": metrics.get_metrics(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": get_request_id(request)
    }


@app.get(
    "/",
    summary="🏠 API Root",
    description="Welcome message and API information with quick navigation",
    tags=["General"]
)
async def root():
    """Enhanced root endpoint with comprehensive API information."""
    return {
        "message": f"🚀 Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "projects": "/projects",
            "buckets": "/buckets"
        },
        "features": [
            "🔧 Comprehensive GCS operations",
            "🔒 API key authentication",
            "📊 Built-in monitoring",
            "⚡ Optimized performance",
            "🛡️ Security headers",
            "📝 Structured logging"
        ]
    }


# ========== PROJECT ENDPOINTS ==========

@app.get(
    "/projects",
    response_model=ProjectsResponse,
    summary="📁 List Projects",
    description="List all GCP projects accessible to the current credentials with caching",
    tags=["Projects"]
)
@limiter.limit(settings.rate_limit_default)
async def list_projects(
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    project: GCPProjectProvider = Depends(get_project_provider)
) -> ProjectsResponse:
    """List all accessible GCP projects with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("projects")
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Projects retrieved from cache", request_id=request_id)
            return ProjectsResponse(**cached_result)

        # Fetch from provider
        projects = project.list_projects()
        result = {"projects": projects, "count": len(projects)}

        # Cache the result
        cache.set(cache_key_str, result, ttl=600)  # Cache for 10 minutes

        logger.info("Projects listed successfully",
                   request_id=request_id,
                   count=len(projects))

        return ProjectsResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "list_projects", get_request_id(request))


@app.get(
    "/projects/current",
    response_model=CurrentProjectResponse,
    summary="🎯 Get Current Project",
    description="Get the current GCP project ID with caching",
    tags=["Projects"]
)
@limiter.limit(settings.rate_limit_default)
async def get_current_project(
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> CurrentProjectResponse:
    """Get the current GCP project ID with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("current_project")
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Current project retrieved from cache", request_id=request_id)
            return CurrentProjectResponse(**cached_result)

        # Fetch from provider
        project_id = storage.get_current_project_id()
        result = {"project_id": project_id}

        # Cache the result
        cache.set(cache_key_str, result, ttl=3600)  # Cache for 1 hour

        logger.info("Current project retrieved",
                   request_id=request_id,
                   project_id=project_id)

        return CurrentProjectResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "get_current_project", get_request_id(request))


# ========== BUCKET ENDPOINTS ==========

@app.get(
    "/buckets",
    response_model=BucketsResponse,
    summary="🪣 List Buckets",
    description="List all buckets in the current project with caching",
    tags=["Buckets"]
)
@limiter.limit(settings.rate_limit_default)
async def list_buckets(
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BucketsResponse:
    """List all buckets in the current project with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("buckets")
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Buckets retrieved from cache", request_id=request_id)
            return BucketsResponse(**cached_result)

        # Fetch from provider
        buckets = storage.list_buckets()
        result = {"buckets": buckets, "count": len(buckets)}

        # Cache the result
        cache.set(cache_key_str, result, ttl=300)  # Cache for 5 minutes

        logger.info("Buckets listed successfully",
                   request_id=request_id,
                   count=len(buckets))

        return BucketsResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "list_buckets", get_request_id(request))


@app.post(
    "/buckets/{bucket_name}",
    response_model=BucketCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="➕ Create Bucket",
    description="Create a new GCS bucket with validation and cache invalidation",
    tags=["Buckets"]
)
@limiter.limit("10/minute")
async def create_bucket(
    bucket_name: str,
    request_obj: BucketCreateRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BucketCreateResponse:
    """Create a new GCS bucket with enhanced validation."""
    try:
        request_id = get_request_id(request)

        # Validate bucket name
        if not bucket_name or len(bucket_name) < 3:
            raise ValidationError("Bucket name must be at least 3 characters")

        success = storage.create_bucket(
            bucket_name,
            request_obj.location,
            request_obj.storage_class
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": f"Bucket '{bucket_name}' already exists",
                    "error_type": "conflict",
                    "bucket_name": bucket_name,
                    "request_id": request_id
                }
            )

        # Invalidate cache
        cache.delete(cache_key("buckets"))

        result = BucketCreateResponse(
            bucket_name=bucket_name,
            location=request_obj.location or storage.config.DEFAULT_LOCATION,
            storage_class=request_obj.storage_class or storage.config.DEFAULT_STORAGE_CLASS
        )

        logger.info("Bucket created successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   location=result.location,
                   storage_class=result.storage_class)

        return result

    except Exception as e:
        raise handle_provider_exception(e, "create_bucket", get_request_id(request))


@app.delete(
    "/buckets/{bucket_name}",
    response_model=BucketDeleteResponse,
    summary="🗑️ Delete Bucket",
    description="Delete a GCS bucket with force option and cache invalidation",
    tags=["Buckets"]
)
@limiter.limit("10/minute")
async def delete_bucket(
    bucket_name: str,
    request_obj: BucketDeleteRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BucketDeleteResponse:
    """Delete a GCS bucket with enhanced validation."""
    try:
        request_id = get_request_id(request)

        success = storage.delete_bucket(bucket_name, request_obj.force)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Bucket '{bucket_name}' not found",
                    "error_type": "not_found",
                    "bucket_name": bucket_name,
                    "request_id": request_id
                }
            )

        # Invalidate cache
        cache.delete(cache_key("buckets"))
        cache.delete(cache_key("bucket_info", bucket_name))

        result = BucketDeleteResponse(
            bucket_name=bucket_name,
            force_deleted=request_obj.force
        )

        logger.info("Bucket deleted successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   force=request_obj.force)

        return result

    except Exception as e:
        raise handle_provider_exception(e, "delete_bucket", get_request_id(request))


@app.get(
    "/buckets/{bucket_name}",
    response_model=BucketInfoResponse,
    summary="ℹ️ Get Bucket Info",
    description="Get detailed information about a bucket with caching",
    tags=["Buckets"]
)
@limiter.limit(settings.rate_limit_default)
async def get_bucket_info(
    bucket_name: str,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BucketInfoResponse:
    """Get detailed information about a bucket with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("bucket_info", bucket_name)
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Bucket info retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name)
            return BucketInfoResponse(**cached_result)

        # Fetch from provider
        bucket_info = storage.get_bucket_info(bucket_name)
        result = {"bucket_info": bucket_info}

        # Cache the result
        cache.set(cache_key_str, result, ttl=600)  # Cache for 10 minutes

        logger.info("Bucket info retrieved successfully",
                   request_id=request_id,
                   bucket_name=bucket_name)

        return BucketInfoResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "get_bucket_info", get_request_id(request))


@app.head(
    "/buckets/{bucket_name}",
    summary="🔍 Check Bucket Exists (HEAD)",
    description="Check if a bucket exists using HEAD request",
    tags=["Buckets"]
)
@limiter.limit(settings.rate_limit_default)
async def bucket_exists_head(
    bucket_name: str,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
):
    """Check if a bucket exists (HEAD request) with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("bucket_exists", bucket_name)
        cached_result = cache.get(cache_key_str)
        if cached_result is not None:
            if not cached_result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Bucket '{bucket_name}' not found"
                )
            return {}

        # Check existence
        exists = storage.bucket_exists(bucket_name)

        # Cache the result
        cache.set(cache_key_str, exists, ttl=300)  # Cache for 5 minutes

        if not exists:
            logger.info("Bucket not found",
                       request_id=request_id,
                       bucket_name=bucket_name)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Bucket '{bucket_name}' not found",
                    "error_type": "not_found",
                    "bucket_name": bucket_name,
                    "request_id": request_id
                }
            )

        logger.info("Bucket exists check completed",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   exists=True)

        return {}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_provider_exception(e, "bucket_exists", get_request_id(request))


@app.get(
    "/buckets/{bucket_name}/exists",
    response_model=BucketExistsResponse,
    summary="✅ Check Bucket Exists",
    description="Check if a bucket exists with detailed response",
    tags=["Buckets"]
)
@limiter.limit(settings.rate_limit_default)
async def bucket_exists(
    bucket_name: str,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BucketExistsResponse:
    """Check if a bucket exists with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("bucket_exists", bucket_name)
        cached_result = cache.get(cache_key_str)
        if cached_result is not None:
            logger.info("Bucket existence retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name,
                       exists=cached_result)
            return BucketExistsResponse(bucket_name=bucket_name, exists=cached_result)

        # Check existence
        exists = storage.bucket_exists(bucket_name)

        # Cache the result
        cache.set(cache_key_str, exists, ttl=300)  # Cache for 5 minutes

        logger.info("Bucket existence checked",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   exists=exists)

        return BucketExistsResponse(bucket_name=bucket_name, exists=exists)

    except Exception as e:
        raise handle_provider_exception(e, "bucket_exists", get_request_id(request))


# ========== BLOB ENDPOINTS ==========

@app.get(
    "/buckets/{bucket_name}/blobs",
    response_model=BlobsResponse,
    summary="📄 List Blobs",
    description="List all blobs in a bucket with optional prefix filtering and caching",
    tags=["Blobs"]
)
@limiter.limit(settings.rate_limit_default)
async def list_blobs(
    bucket_name: str,
    request: Request,
    prefix: str = "",
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobsResponse:
    """List all blobs in a bucket with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("blobs", bucket_name, prefix)
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Blobs retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name,
                       prefix=prefix)
            return BlobsResponse(**cached_result)

        # Fetch from provider
        blobs = storage.list_blobs(bucket_name, prefix)
        result = {
            "bucket_name": bucket_name,
            "prefix": prefix,
            "blobs": blobs,
            "count": len(blobs)
        }

        # Cache the result
        cache.set(cache_key_str, result, ttl=180)  # Cache for 3 minutes

        logger.info("Blobs listed successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   prefix=prefix,
                   count=len(blobs))

        return BlobsResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "list_blobs", get_request_id(request))


@app.get(
    "/buckets/{bucket_name}/folders",
    response_model=FoldersResponse,
    summary="📁 List Folders",
    description="List all folders in a bucket with optional prefix filtering and caching",
    tags=["Blobs"]
)
@limiter.limit(settings.rate_limit_default)
async def list_folders(
    bucket_name: str,
    request: Request,
    prefix: str = "",
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> FoldersResponse:
    """List all folders in a bucket with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("folders", bucket_name, prefix)
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Folders retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name,
                       prefix=prefix)
            return FoldersResponse(**cached_result)

        # Fetch from provider
        folders = storage.list_folders(bucket_name, prefix)
        result = {
            "bucket_name": bucket_name,
            "prefix": prefix,
            "folders": folders,
            "count": len(folders)
        }

        # Cache the result
        cache.set(cache_key_str, result, ttl=300)  # Cache for 5 minutes

        logger.info("Folders listed successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   prefix=prefix,
                   count=len(folders))

        return FoldersResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "list_folders", get_request_id(request))


@app.post(
    "/buckets/{bucket_name}/blobs/{blob_name:path}/upload",
    response_model=BlobOperationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="⬆️ Upload Blob",
    description="Upload a file to a bucket as a blob with validation and cache invalidation",
    tags=["Blobs"]
)
@limiter.limit("20/minute")
async def upload_blob(
    bucket_name: str,
    blob_name: str,
    request_obj: BlobUploadRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobOperationResponse:
    """Upload a file to a bucket as a blob with enhanced validation."""
    try:
        request_id = get_request_id(request)

        # Validate inputs
        if not blob_name.strip():
            raise ValidationError("Blob name cannot be empty")

        if not os.path.exists(request_obj.file_path):
            raise ValidationError(f"File not found: {request_obj.file_path}")

        # Check file size
        file_size = os.path.getsize(request_obj.file_path)
        if file_size > settings.max_request_size:
            raise ValidationError(f"File too large: {file_size} bytes (max: {settings.max_request_size})")

        success = storage.upload_blob(bucket_name, blob_name, request_obj.file_path)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Upload failed",
                    "error_type": "operation",
                    "bucket_name": bucket_name,
                    "blob_name": blob_name,
                    "request_id": request_id
                }
            )

        # Invalidate related caches
        cache.delete(cache_key("blobs", bucket_name, ""))
        cache.delete(cache_key("blob_info", bucket_name, blob_name))

        logger.info("Blob uploaded successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   blob_name=blob_name,
                   file_size=file_size)

        return BlobOperationResponse(bucket_name=bucket_name, blob_name=blob_name)

    except Exception as e:
        raise handle_provider_exception(e, "upload_blob", get_request_id(request))


@app.post(
    "/buckets/{bucket_name}/blobs/{blob_name:path}/download",
    response_model=BlobOperationResponse,
    summary="⬇️ Download Blob",
    description="Download a blob from a bucket to a local file with validation",
    tags=["Blobs"]
)
@limiter.limit("20/minute")
async def download_blob(
    bucket_name: str,
    blob_name: str,
    request_obj: BlobDownloadRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobOperationResponse:
    """Download a blob from a bucket to a local file with enhanced validation."""
    try:
        request_id = get_request_id(request)

        # Validate download path
        download_dir = os.path.dirname(request_obj.file_path)
        if download_dir and not os.path.exists(download_dir):
            raise ValidationError(f"Download directory does not exist: {download_dir}")

        success = storage.download_blob(bucket_name, blob_name, request_obj.file_path)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Download failed",
                    "error_type": "operation",
                    "bucket_name": bucket_name,
                    "blob_name": blob_name,
                    "request_id": request_id
                }
            )

        # Get downloaded file size for logging
        file_size = os.path.getsize(request_obj.file_path) if os.path.exists(request_obj.file_path) else 0

        logger.info("Blob downloaded successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   blob_name=blob_name,
                   file_path=request_obj.file_path,
                   file_size=file_size)

        return BlobOperationResponse(bucket_name=bucket_name, blob_name=blob_name)

    except Exception as e:
        raise handle_provider_exception(e, "download_blob", get_request_id(request))


@app.delete(
    "/buckets/{bucket_name}/blobs/{blob_name:path}",
    response_model=BlobOperationResponse,
    summary="🗑️ Delete Blob",
    description="Delete a blob from a bucket with cache invalidation",
    tags=["Blobs"]
)
@limiter.limit("30/minute")
async def delete_blob(
    bucket_name: str,
    blob_name: str,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobOperationResponse:
    """Delete a blob from a bucket with enhanced validation."""
    try:
        request_id = get_request_id(request)

        success = storage.delete_blob(bucket_name, blob_name)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Blob '{blob_name}' not found",
                    "error_type": "not_found",
                    "bucket_name": bucket_name,
                    "blob_name": blob_name,
                    "request_id": request_id
                }
            )

        # Invalidate related caches
        cache.delete(cache_key("blobs", bucket_name, ""))
        cache.delete(cache_key("blob_info", bucket_name, blob_name))

        logger.info("Blob deleted successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   blob_name=blob_name)

        return BlobOperationResponse(bucket_name=bucket_name, blob_name=blob_name)

    except Exception as e:
        raise handle_provider_exception(e, "delete_blob", get_request_id(request))


@app.get(
    "/buckets/{bucket_name}/blobs/{blob_name:path}",
    response_model=BlobInfoResponse,
    summary="ℹ️ Get Blob Info",
    description="Get detailed information about a blob with caching",
    tags=["Blobs"]
)
@limiter.limit(settings.rate_limit_default)
async def get_blob_info(
    bucket_name: str,
    blob_name: str,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobInfoResponse:
    """Get detailed information about a blob with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("blob_info", bucket_name, blob_name)
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Blob info retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name,
                       blob_name=blob_name)
            return BlobInfoResponse(**cached_result)

        # Fetch from provider
        blob = storage.get_blob(bucket_name, blob_name)

        if blob is None:
            result = {
                "bucket_name": bucket_name,
                "blob_name": blob_name,
                "exists": False,
                "blob_info": None
            }
        else:
            blob_info = {
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "storage_class": blob.storage_class,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "metadata": blob.metadata or {},
                "md5_hash": blob.md5_hash,
                "crc32c": blob.crc32c,
            }

            result = {
                "bucket_name": bucket_name,
                "blob_name": blob_name,
                "exists": True,
                "blob_info": blob_info
            }

        # Cache the result
        cache.set(cache_key_str, result, ttl=300)  # Cache for 5 minutes

        logger.info("Blob info retrieved successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   blob_name=blob_name,
                   exists=result["exists"])

        return BlobInfoResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "get_blob_info", get_request_id(request))


@app.get(
    "/buckets/{bucket_name}/blobs/{blob_name:path}/url",
    response_model=BlobUrlResponse,
    summary="🔗 Get Blob URL",
    description="Get the public URL of a blob with caching",
    tags=["Blobs"]
)
@limiter.limit(settings.rate_limit_default)
async def get_blob_url(
    bucket_name: str,
    blob_name: str,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobUrlResponse:
    """Get the public URL of a blob with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("blob_url", bucket_name, blob_name)
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Blob URL retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name,
                       blob_name=blob_name)
            return BlobUrlResponse(**cached_result)

        # Get URL from provider
        url = storage.get_blob_url(bucket_name, blob_name)
        result = {
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "url": url
        }

        # Cache the result
        cache.set(cache_key_str, result, ttl=3600)  # Cache for 1 hour

        logger.info("Blob URL retrieved successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   blob_name=blob_name)

        return BlobUrlResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "get_blob_url", get_request_id(request))


@app.post(
    "/buckets/{bucket_name}/blobs/{blob_name:path}/signed-url",
    response_model=SignedUrlResponse,
    summary="🔐 Generate Signed URL",
    description="Generate a signed URL for temporary access to a blob with validation",
    tags=["Blobs"]
)
@limiter.limit("50/minute")
async def generate_signed_url(
    bucket_name: str,
    blob_name: str,
    request_obj: SignedUrlRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> SignedUrlResponse:
    """Generate a signed URL for temporary access to a blob with validation."""
    try:
        request_id = get_request_id(request)

        # Validate expiration hours
        if request_obj.expiration_hours and (request_obj.expiration_hours < 1 or request_obj.expiration_hours > 168):
            raise ValidationError("Expiration hours must be between 1 and 168 (1 week)")

        url = storage.generate_signed_url(
            bucket_name,
            blob_name,
            request_obj.expiration_hours,
            request_obj.method
        )

        result = SignedUrlResponse(
            bucket_name=bucket_name,
            blob_name=blob_name,
            method=request_obj.method,
            expiration_hours=request_obj.expiration_hours or storage.config.DEFAULT_SIGNED_URL_EXPIRATION_HOURS,
            signed_url=url
        )

        logger.info("Signed URL generated successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   blob_name=blob_name,
                   method=request_obj.method,
                   expiration_hours=result.expiration_hours)

        return result

    except Exception as e:
        raise handle_provider_exception(e, "generate_signed_url", get_request_id(request))


# ========== COPY/MOVE ENDPOINTS ==========

@app.post(
    "/buckets/{bucket_name}/blobs/{blob_name:path}/copy",
    response_model=BlobOperationResponse,
    summary="📋 Copy Blob",
    description="Copy a blob from one location to another with validation and cache invalidation",
    tags=["Blobs"]
)
@limiter.limit("20/minute")
async def copy_blob(
    bucket_name: str,
    blob_name: str,
    request_obj: BlobCopyRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobOperationResponse:
    """Copy a blob from one location to another with enhanced validation."""
    try:
        request_id = get_request_id(request)

        # Validate that source and destination are different
        if (request_obj.source_bucket == request_obj.dest_bucket and
            request_obj.source_blob == request_obj.dest_blob):
            raise ValidationError("Source and destination must be different")

        success = storage.copy_blob(
            request_obj.source_bucket,
            request_obj.source_blob,
            request_obj.dest_bucket,
            request_obj.dest_blob
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Copy operation failed",
                    "error_type": "operation",
                    "source_bucket": request_obj.source_bucket,
                    "source_blob": request_obj.source_blob,
                    "dest_bucket": request_obj.dest_bucket,
                    "dest_blob": request_obj.dest_blob,
                    "request_id": request_id
                }
            )

        # Invalidate related caches
        cache.delete(cache_key("blobs", request_obj.dest_bucket, ""))
        cache.delete(cache_key("blob_info", request_obj.dest_bucket, request_obj.dest_blob))

        logger.info("Blob copied successfully",
                   request_id=request_id,
                   source_bucket=request_obj.source_bucket,
                   source_blob=request_obj.source_blob,
                   dest_bucket=request_obj.dest_bucket,
                   dest_blob=request_obj.dest_blob)

        return BlobOperationResponse(
            bucket_name=request_obj.dest_bucket,
            blob_name=request_obj.dest_blob
        )

    except Exception as e:
        raise handle_provider_exception(e, "copy_blob", get_request_id(request))


@app.post(
    "/buckets/{bucket_name}/blobs/{blob_name:path}/move",
    response_model=BlobOperationResponse,
    summary="✂️ Move Blob",
    description="Move a blob from one location to another with validation and cache invalidation",
    tags=["Blobs"]
)
@limiter.limit("20/minute")
async def move_blob(
    bucket_name: str,
    blob_name: str,
    request_obj: BlobMoveRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BlobOperationResponse:
    """Move a blob from one location to another with enhanced validation."""
    try:
        request_id = get_request_id(request)

        # Validate that source and destination are different
        if (request_obj.source_bucket == request_obj.dest_bucket and
            request_obj.source_blob == request_obj.dest_blob):
            raise ValidationError("Source and destination must be different")

        success = storage.move_blob(
            request_obj.source_bucket,
            request_obj.source_blob,
            request_obj.dest_bucket,
            request_obj.dest_blob
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Move operation failed",
                    "error_type": "operation",
                    "source_bucket": request_obj.source_bucket,
                    "source_blob": request_obj.source_blob,
                    "dest_bucket": request_obj.dest_bucket,
                    "dest_blob": request_obj.dest_blob,
                    "request_id": request_id
                }
            )

        # Invalidate related caches
        cache.delete(cache_key("blobs", request_obj.source_bucket, ""))
        cache.delete(cache_key("blobs", request_obj.dest_bucket, ""))
        cache.delete(cache_key("blob_info", request_obj.source_bucket, request_obj.source_blob))
        cache.delete(cache_key("blob_info", request_obj.dest_bucket, request_obj.dest_blob))

        logger.info("Blob moved successfully",
                   request_id=request_id,
                   source_bucket=request_obj.source_bucket,
                   source_blob=request_obj.source_blob,
                   dest_bucket=request_obj.dest_bucket,
                   dest_blob=request_obj.dest_blob)

        return BlobOperationResponse(
            bucket_name=request_obj.dest_bucket,
            blob_name=request_obj.dest_blob
        )

    except Exception as e:
        raise handle_provider_exception(e, "move_blob", get_request_id(request))


# ========== BATCH ENDPOINTS ==========

@app.post(
    "/buckets/{bucket_name}/batch/upload",
    response_model=BatchResponse,
    summary="⬆️ Batch Upload",
    description="Upload multiple files to a bucket with validation and progress tracking",
    tags=["Batch Operations"]
)
@limiter.limit("5/minute")
async def batch_upload(
    bucket_name: str,
    request_obj: BatchUploadRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BatchResponse:
    """Upload multiple files to a bucket with enhanced validation."""
    try:
        request_id = get_request_id(request)

        # Validate file paths
        missing_files = [fp for fp in request_obj.file_paths if not os.path.exists(fp)]
        if missing_files:
            raise ValidationError(f"Files not found: {missing_files}")

        # Check total size
        total_size = sum(os.path.getsize(fp) for fp in request_obj.file_paths)
        if total_size > settings.max_request_size * len(request_obj.file_paths):
            raise ValidationError(f"Total size too large: {total_size} bytes")

        results = storage.batch_upload(
            bucket_name,
            request_obj.file_paths,
            request_obj.prefix
        )

        successful_count = sum(results.values())
        failed_count = len(request_obj.file_paths) - successful_count

        # Invalidate related caches
        cache.delete(cache_key("blobs", bucket_name, ""))
        if request_obj.prefix:
            cache.delete(cache_key("blobs", bucket_name, request_obj.prefix))

        result = BatchResponse(
            bucket_name=bucket_name,
            total_items=len(request_obj.file_paths),
            successful_operations=successful_count,
            failed_operations=failed_count,
            results=results
        )

        logger.info("Batch upload completed",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   prefix=request_obj.prefix,
                   total=len(request_obj.file_paths),
                   successful=successful_count,
                   failed=failed_count,
                   total_size=total_size)

        return result

    except Exception as e:
        raise handle_provider_exception(e, "batch_upload", get_request_id(request))


@app.post(
    "/buckets/{bucket_name}/batch/delete",
    response_model=BatchResponse,
    summary="🗑️ Batch Delete",
    description="Delete multiple blobs from a bucket with progress tracking",
    tags=["Batch Operations"]
)
@limiter.limit("10/minute")
async def batch_delete(
    bucket_name: str,
    request_obj: BatchDeleteRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> BatchResponse:
    """Delete multiple blobs from a bucket with enhanced validation."""
    try:
        request_id = get_request_id(request)

        # Validate blob names
        if not request_obj.blob_names:
            raise ValidationError("At least one blob name is required")

        results = storage.batch_delete(bucket_name, request_obj.blob_names)
        successful_count = sum(results.values())
        failed_count = len(request_obj.blob_names) - successful_count

        # Invalidate related caches
        cache.delete(cache_key("blobs", bucket_name, ""))
        for blob_name in request_obj.blob_names:
            cache.delete(cache_key("blob_info", bucket_name, blob_name))

        result = BatchResponse(
            bucket_name=bucket_name,
            total_items=len(request_obj.blob_names),
            successful_operations=successful_count,
            failed_operations=failed_count,
            results=results
        )

        logger.info("Batch delete completed",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   total=len(request_obj.blob_names),
                   successful=successful_count,
                   failed=failed_count)

        return result

    except Exception as e:
        raise handle_provider_exception(e, "batch_delete", get_request_id(request))


# ========== ANALYTICS ENDPOINTS ==========

@app.get(
    "/buckets/{bucket_name}/analytics/storage",
    response_model=StorageInfoResponse,
    summary="📊 Get Storage Analytics",
    description="Get storage usage information for a bucket with caching",
    tags=["Analytics"]
)
@limiter.limit("30/minute")
async def get_bucket_storage_info(
    bucket_name: str,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> StorageInfoResponse:
    """Get storage usage information for a bucket with caching."""
    try:
        request_id = get_request_id(request)

        # Check cache first
        cache_key_str = cache_key("storage_info", bucket_name)
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Storage info retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name)
            return StorageInfoResponse(**cached_result)

        # Fetch from provider
        storage_info = storage.get_bucket_storage_info(bucket_name)
        result = {"storage_info": storage_info}

        # Cache the result for longer since storage info changes slowly
        cache.set(cache_key_str, result, ttl=1800)  # Cache for 30 minutes

        logger.info("Storage info retrieved successfully",
                   request_id=request_id,
                   bucket_name=bucket_name)

        return StorageInfoResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "get_bucket_storage_info", get_request_id(request))


# ========== SEARCH ENDPOINTS ==========

@app.post(
    "/buckets/{bucket_name}/search",
    response_model=SearchResponse,
    summary="🔍 Search Blobs",
    description="Search for blobs matching a pattern with validation and caching",
    tags=["Search"]
)
@limiter.limit("30/minute")
async def search_blobs(
    bucket_name: str,
    request_obj: SearchBlobsRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> SearchResponse:
    """Search for blobs matching a pattern with caching."""
    try:
        request_id = get_request_id(request)

        # Validate pattern
        if not request_obj.pattern.strip():
            raise ValidationError("Search pattern cannot be empty")

        # Check cache first
        cache_key_str = cache_key("search", bucket_name, request_obj.pattern, request_obj.prefix or "")
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Search results retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name,
                       pattern=request_obj.pattern)
            return SearchResponse(**cached_result)

        # Search from provider
        matching_blobs = storage.search_blobs(
            bucket_name,
            request_obj.pattern,
            request_obj.prefix
        )

        result = {
            "bucket_name": bucket_name,
            "pattern": request_obj.pattern,
            "prefix": request_obj.prefix,
            "matching_blobs": matching_blobs,
            "count": len(matching_blobs)
        }

        # Cache the result
        cache.set(cache_key_str, result, ttl=300)  # Cache for 5 minutes

        logger.info("Search completed successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   pattern=request_obj.pattern,
                   prefix=request_obj.prefix,
                   count=len(matching_blobs))

        return SearchResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "search_blobs", get_request_id(request))


@app.post(
    "/buckets/{bucket_name}/filter",
    response_model=FilterResponse,
    summary="🔎 Filter Blobs by Size",
    description="Filter blobs by size range with validation and caching",
    tags=["Search"]
)
@limiter.limit("30/minute")
async def filter_blobs_by_size(
    bucket_name: str,
    request_obj: FilterBlobsRequest,
    request: Request,
    _: Optional[str] = Depends(verify_auth),
    storage: GCPStorageProvider = Depends(get_storage_provider)
) -> FilterResponse:
    """Filter blobs by size range with caching."""
    try:
        request_id = get_request_id(request)

        # Validate size range
        if request_obj.min_size_mb < 0 or request_obj.max_size_mb < 0:
            raise ValidationError("Size values must be non-negative")

        if request_obj.min_size_mb > request_obj.max_size_mb:
            raise ValidationError("Minimum size cannot be greater than maximum size")

        # Check cache first
        cache_key_str = cache_key("filter", bucket_name, request_obj.min_size_mb, request_obj.max_size_mb)
        cached_result = cache.get(cache_key_str)
        if cached_result:
            logger.info("Filter results retrieved from cache",
                       request_id=request_id,
                       bucket_name=bucket_name,
                       min_size=request_obj.min_size_mb,
                       max_size=request_obj.max_size_mb)
            return FilterResponse(**cached_result)

        # Filter from provider
        matching_blobs = storage.filter_blobs_by_size(
            bucket_name,
            request_obj.min_size_mb,
            request_obj.max_size_mb
        )

        result = {
            "bucket_name": bucket_name,
            "min_size_mb": request_obj.min_size_mb,
            "max_size_mb": request_obj.max_size_mb,
            "matching_blobs": matching_blobs,
            "count": len(matching_blobs)
        }

        # Cache the result
        cache.set(cache_key_str, result, ttl=300)  # Cache for 5 minutes

        logger.info("Filter completed successfully",
                   request_id=request_id,
                   bucket_name=bucket_name,
                   min_size=request_obj.min_size_mb,
                   max_size=request_obj.max_size_mb,
                   count=len(matching_blobs))

        return FilterResponse(**result)

    except Exception as e:
        raise handle_provider_exception(e, "filter_blobs_by_size", get_request_id(request))


# ========== ERROR HANDLERS ==========

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Enhanced validation error handler."""
    request_id = get_request_id(request)
    logger.error("Validation error occurred",
                request_id=request_id,
                error=str(exc),
                path=request.url.path)

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            error_type="validation",
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).dict()
    )


@app.exception_handler(OperationError)
async def operation_error_handler(request: Request, exc: OperationError):
    """Enhanced operation error handler."""
    request_id = get_request_id(request)
    logger.error("Operation error occurred",
                request_id=request_id,
                error=str(exc),
                path=request.url.path)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=str(exc),
            error_type="operation",
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).dict()
    )


@app.exception_handler(GCPStorageError)
async def gcp_storage_error_handler(request: Request, exc: GCPStorageError):
    """Enhanced GCP storage error handler."""
    request_id = get_request_id(request)
    logger.error("GCP storage error occurred",
                request_id=request_id,
                error=str(exc),
                path=request.url.path)

    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorResponse(
            error=str(exc),
            error_type="gcp_storage",
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler."""
    request_id = get_request_id(request)
    logger.error("Unexpected error occurred",
                request_id=request_id,
                error=str(exc),
                path=request.url.path,
                exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An unexpected error occurred",
            error_type="unexpected",
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).dict()
    )


# ========== SERVER STARTUP ==========

if __name__ == "__main__":
    # Production-ready server configuration
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=settings.request_logging,
        use_colors=settings.log_format != "json",
        server_header=False,
        date_header=True,
        timeout_keep_alive=65,
        timeout_graceful_shutdown=10,
    )
