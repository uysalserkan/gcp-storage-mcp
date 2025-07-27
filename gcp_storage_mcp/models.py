"""This module contains the models for the GCP Storage MCP."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class GcpStorageConfig(BaseModel):
    """Configuration for the GCP Storage MCP."""

    project_id: str
    bucket_name: str
    credentials_path: str

# ========== REQUEST/RESPONSE MODELS ==========

class BaseResponse(BaseModel):
    """Base response model."""
    success: bool
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseResponse):
    """Health check response."""
    success: bool = True
    status: str
    project_id: Optional[str] = None
    timestamp: str


# Project Models
class ProjectsResponse(BaseResponse):
    """Projects list response."""
    success: bool = True
    projects: List[str]
    count: int


class CurrentProjectResponse(BaseResponse):
    """Current project response."""
    success: bool = True
    project_id: str


# Bucket Models
class BucketCreateRequest(BaseModel):
    """Create bucket request."""
    bucket_name: str = Field(..., description="Name of the bucket to create")
    location: Optional[str] = Field(None, description="Bucket location (e.g., 'US', 'EU', 'us-central1')")
    storage_class: Optional[str] = Field(None, description="Storage class ('STANDARD', 'NEARLINE', 'COLDLINE', 'ARCHIVE')")


class BucketDeleteRequest(BaseModel):
    """Delete bucket request."""
    force: bool = Field(False, description="Force delete by removing all objects first")


class BucketsResponse(BaseResponse):
    """Buckets list response."""
    success: bool = True
    buckets: List[str]
    count: int


class BucketCreateResponse(BaseResponse):
    """Bucket creation response."""
    success: bool = True
    bucket_name: str
    location: str
    storage_class: str


class BucketDeleteResponse(BaseResponse):
    """Bucket deletion response."""
    success: bool = True
    bucket_name: str
    force_deleted: bool


class BucketInfoResponse(BaseResponse):
    """Bucket info response."""
    success: bool = True
    bucket_info: Dict[str, Any]


class BucketExistsResponse(BaseResponse):
    """Bucket exists response."""
    success: bool = True
    bucket_name: str
    exists: bool


# Blob Models
class BlobUploadRequest(BaseModel):
    """Upload blob request."""
    blob_name: str = Field(..., description="Name of the blob")
    file_path: str = Field(..., description="Local file path to upload")


class BlobDownloadRequest(BaseModel):
    """Download blob request."""
    file_path: str = Field(..., description="Local file path to save to")


class BlobCopyRequest(BaseModel):
    """Copy blob request."""
    source_bucket: str = Field(..., description="Source bucket name")
    source_blob: str = Field(..., description="Source blob name")
    dest_bucket: str = Field(..., description="Destination bucket name")
    dest_blob: str = Field(..., description="Destination blob name")


class BlobMoveRequest(BaseModel):
    """Move blob request."""
    source_bucket: str = Field(..., description="Source bucket name")
    source_blob: str = Field(..., description="Source blob name")
    dest_bucket: str = Field(..., description="Destination bucket name")
    dest_blob: str = Field(..., description="Destination blob name")


class SignedUrlRequest(BaseModel):
    """Signed URL request."""
    expiration_hours: Optional[int] = Field(None, description="Hours until URL expires")
    method: str = Field("GET", description="HTTP method ('GET', 'PUT', 'POST', 'DELETE')")


class BatchUploadRequest(BaseModel):
    """Batch upload request."""
    file_paths: List[str] = Field(..., description="List of local file paths to upload")
    prefix: str = Field("", description="Optional prefix for blob names")


class BatchDeleteRequest(BaseModel):
    """Batch delete request."""
    blob_names: List[str] = Field(..., description="List of blob names to delete")


class SearchBlobsRequest(BaseModel):
    """Search blobs request."""
    pattern: str = Field(..., description="Pattern to search for in blob names")
    prefix: str = Field("", description="Optional prefix to limit search scope")


class FilterBlobsRequest(BaseModel):
    """Filter blobs request."""
    min_size_mb: float = Field(0, description="Minimum size in MB")
    max_size_mb: float = Field(float('inf'), description="Maximum size in MB")


class BlobsResponse(BaseResponse):
    """Blobs list response."""
    success: bool = True
    bucket_name: str
    prefix: str
    blobs: List[str]
    count: int


class FoldersResponse(BaseResponse):
    """Folders list response."""
    success: bool = True
    bucket_name: str
    prefix: str
    folders: List[str]
    count: int


class BlobOperationResponse(BaseResponse):
    """Generic blob operation response."""
    success: bool = True
    bucket_name: str
    blob_name: str


class BlobInfoResponse(BaseResponse):
    """Blob info response."""
    success: bool = True
    bucket_name: str
    blob_name: str
    exists: bool
    blob_info: Optional[Dict[str, Any]]


class BlobUrlResponse(BaseResponse):
    """Blob URL response."""
    success: bool = True
    bucket_name: str
    blob_name: str
    url: str


class SignedUrlResponse(BaseResponse):
    """Signed URL response."""
    success: bool = True
    bucket_name: str
    blob_name: str
    method: str
    expiration_hours: int
    signed_url: str


class BatchResponse(BaseResponse):
    """Batch operation response."""
    success: bool = True
    bucket_name: str
    total_items: int
    successful_operations: int
    failed_operations: int
    results: Dict[str, bool]


class StorageInfoResponse(BaseResponse):
    """Storage info response."""
    success: bool = True
    storage_info: Dict[str, Any]


class SearchResponse(BaseResponse):
    """Search response."""
    success: bool = True
    bucket_name: str
    pattern: str
    prefix: str
    matching_blobs: List[str]
    count: int


class FilterResponse(BaseResponse):
    """Filter response."""
    success: bool = True
    bucket_name: str
    min_size_mb: float
    max_size_mb: float
    matching_blobs: List[Dict[str, Any]]
    count: int
