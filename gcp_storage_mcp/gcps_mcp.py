"""GCP Storage MCP Server.

A Model Context Protocol server that provides Google Cloud Storage operations
for AI assistants to interact with GCS buckets and objects.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

# Add the current directory to the path to import our provider
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from provider import (
    GCPStorageProvider, GCPProjectProvider, ProviderConfig,
    ValidationError, OperationError, GCPStorageError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("gcp-storage-mcp")

# Global provider instances (initialized in main)
storage_provider: Optional[GCPStorageProvider] = None
project_provider: Optional[GCPProjectProvider] = None


# ========== UTILITY FUNCTIONS ==========

def handle_provider_error(func_name: str, error: Exception) -> Dict[str, Any]:
    """Handle provider errors and return standardized error response."""
    if isinstance(error, ValidationError):
        logger.error(f"{func_name} validation error: {error}")
        return {"success": False, "error": f"Validation error: {error}", "error_type": "validation"}
    elif isinstance(error, OperationError):
        logger.error(f"{func_name} operation error: {error}")
        return {"success": False, "error": f"Operation error: {error}", "error_type": "operation"}
    elif isinstance(error, GCPStorageError):
        logger.error(f"{func_name} GCP storage error: {error}")
        return {"success": False, "error": f"GCP Storage error: {error}", "error_type": "gcp_storage"}
    else:
        logger.error(f"{func_name} unexpected error: {error}")
        return {"success": False, "error": f"Unexpected error: {error}", "error_type": "unexpected"}


def ensure_providers():
    """Ensure providers are initialized."""
    if storage_provider is None or project_provider is None:
        raise RuntimeError("Providers not initialized. Call initialize_providers() first.")


# ========== PROJECT OPERATIONS ==========

@mcp.tool()
def list_projects() -> Dict[str, Any]:
    """List all GCP projects accessible to the current credentials.

    Returns:
        Dict containing success status and list of project IDs.
    """
    try:
        ensure_providers()
        projects = project_provider.list_projects()
        logger.info(f"Listed {len(projects)} projects")
        return {
            "success": True,
            "projects": projects,
            "count": len(projects)
        }
    except Exception as e:
        return handle_provider_error("list_projects", e)


@mcp.tool()
def get_current_project() -> Dict[str, Any]:
    """Get the current GCP project ID.

    Returns:
        Dict containing success status and current project ID.
    """
    try:
        ensure_providers()
        project_id = storage_provider.get_current_project_id()
        logger.info(f"Current project: {project_id}")
        return {
            "success": True,
            "project_id": project_id
        }
    except Exception as e:
        return handle_provider_error("get_current_project", e)


# ========== BUCKET OPERATIONS ==========

@mcp.tool()
def list_buckets() -> Dict[str, Any]:
    """List all buckets in the current project.

    Returns:
        Dict containing success status and list of bucket names.
    """
    try:
        ensure_providers()
        buckets = storage_provider.list_buckets()
        logger.info(f"Listed {len(buckets)} buckets")
        return {
            "success": True,
            "buckets": buckets,
            "count": len(buckets)
        }
    except Exception as e:
        return handle_provider_error("list_buckets", e)


@mcp.tool()
def create_bucket(bucket_name: str, location: Optional[str] = None, storage_class: Optional[str] = None) -> Dict[str, Any]:
    """Create a new GCS bucket.

    Args:
        bucket_name: Name of the bucket.
        location: Optional bucket location.
        storage_class: Optional storage class.

    Returns:
        Dict containing success status and creation details.
    """
    try:
        ensure_providers()
        success = storage_provider.create_bucket(
            bucket_name,
            location,
            storage_class
        )
        return {
            "success": success,
            "bucket_name": bucket_name,
            "location": location or storage_provider.config.DEFAULT_LOCATION,
            "storage_class": storage_class or storage_provider.config.DEFAULT_STORAGE_CLASS
        }
    except Exception as e:
        return handle_provider_error("create_bucket", e)


@mcp.tool()
def delete_bucket(bucket_name: str, force: bool = False) -> Dict[str, Any]:
    """Delete a GCS bucket.

    Args:
        bucket_name: Name of the bucket.
        force: Force delete by removing all objects first.

    Returns:
        Dict containing success status and deletion details.
    """
    try:
        ensure_providers()
        success = storage_provider.delete_bucket(bucket_name, force)
        return {
            "success": success,
            "bucket_name": bucket_name,
            "force_deleted": force
        }
    except Exception as e:
        return handle_provider_error("delete_bucket", e)


@mcp.tool()
def get_bucket_info(bucket_name: str) -> Dict[str, Any]:
    """Get detailed information about a bucket.

    Args:
        bucket_name: Name of the bucket.

    Returns:
        Dict containing success status and bucket information.
    """
    try:
        ensure_providers()
        bucket_info = storage_provider.get_bucket_info(bucket_name)
        return {
            "success": True,
            "bucket_info": bucket_info
        }
    except Exception as e:
        return handle_provider_error("get_bucket_info", e)


@mcp.tool()
def bucket_exists(bucket_name: str) -> Dict[str, Any]:
    """Check if a bucket exists.

    Args:
        bucket_name: Name of the bucket.

    Returns:
        Dict containing success status and existence check result.
    """
    try:
        ensure_providers()
        exists = storage_provider.bucket_exists(bucket_name)
        return {
            "success": True,
            "bucket_name": bucket_name,
            "exists": exists
        }
    except Exception as e:
        return handle_provider_error("bucket_exists", e)


# ========== BLOB OPERATIONS ==========

@mcp.tool()
def list_blobs(bucket_name: str, prefix: str = "") -> Dict[str, Any]:
    """List all blobs in a bucket.

    Args:
        bucket_name: Name of the bucket.
        prefix: Optional prefix to limit search scope.

    Returns:
        Dict containing success status and list of blob names.
    """
    try:
        ensure_providers()
        blobs = storage_provider.list_blobs(bucket_name, prefix)
        return {
            "success": True,
            "bucket_name": bucket_name,
            "prefix": prefix,
            "blobs": blobs,
            "count": len(blobs)
        }
    except Exception as e:
        return handle_provider_error("list_blobs", e)


@mcp.tool()
def list_folders(bucket_name: str, prefix: str = "") -> Dict[str, Any]:
    """List all folders in a bucket.

    Args:
        bucket_name: Name of the bucket.
        prefix: Optional prefix to limit search scope.

    Returns:
        Dict containing success status and list of folder paths.
    """
    try:
        ensure_providers()
        folders = storage_provider.list_folders(bucket_name, prefix)
        return {
            "success": True,
            "bucket_name": bucket_name,
            "prefix": prefix,
            "folders": folders,
            "count": len(folders)
        }
    except Exception as e:
        return handle_provider_error("list_folders", e)


@mcp.tool()
def upload_blob(bucket_name: str, blob_name: str, file_path: str) -> Dict[str, Any]:
    """Upload a file to a bucket as a blob.

    Args:
        bucket_name: Name of the bucket.
        blob_name: Name of the blob.
        file_path: Path to the file to upload.

    Returns:
        Dict containing success status and upload details.
    """
    try:
        ensure_providers()
        success = storage_provider.upload_blob(
            bucket_name,
            blob_name,
            file_path
        )
        return {
            "success": success,
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "file_path": file_path
        }
    except Exception as e:
        return handle_provider_error("upload_blob", e)


@mcp.tool()
def download_blob(bucket_name: str, blob_name: str, file_path: str) -> Dict[str, Any]:
    """Download a blob from a bucket to a local file.

    Args:
        bucket_name: Name of the bucket.
        blob_name: Name of the blob.
        file_path: Path to save the downloaded file.

    Returns:
        Dict containing success status and download details.
    """
    try:
        ensure_providers()
        success = storage_provider.download_blob(
            bucket_name,
            blob_name,
            file_path
        )
        return {
            "success": success,
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "file_path": file_path
        }
    except Exception as e:
        return handle_provider_error("download_blob", e)


@mcp.tool()
def delete_blob(bucket_name: str, blob_name: str) -> Dict[str, Any]:
    """Delete a blob from a bucket.

    Args:
        bucket_name: Name of the bucket.
        blob_name: Name of the blob.

    Returns:
        Dict containing success status and deletion details.
    """
    try:
        ensure_providers()
        success = storage_provider.delete_blob(bucket_name, blob_name)
        return {
            "success": success,
            "bucket_name": bucket_name,
            "blob_name": blob_name
        }
    except Exception as e:
        return handle_provider_error("delete_blob", e)


@mcp.tool()
def get_blob_info(bucket_name: str, blob_name: str) -> Dict[str, Any]:
    """Get detailed information about a blob.

    Args:
        bucket_name: Name of the bucket.
        blob_name: Name of the blob.

    Returns:
        Dict containing success status and blob information.
    """
    try:
        ensure_providers()
        blob = storage_provider.get_blob(bucket_name, blob_name)
        if blob is None:
            return {
                "success": True,
                "bucket_name": bucket_name,
                "blob_name": blob_name,
                "exists": False,
                "blob_info": None
            }

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

        return {
            "success": True,
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "exists": True,
            "blob_info": blob_info
        }
    except Exception as e:
        return handle_provider_error("get_blob_info", e)


@mcp.tool()
def get_blob_url(bucket_name: str, blob_name: str) -> Dict[str, Any]:
    """Get the public URL of a blob.

    Args:
        bucket_name: Name of the bucket.
        blob_name: Name of the blob.

    Returns:
        Dict containing success status and blob URL.
    """
    try:
        ensure_providers()
        url = storage_provider.get_blob_url(bucket_name, blob_name)
        return {
            "success": True,
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "url": url
        }
    except Exception as e:
        return handle_provider_error("get_blob_url", e)


@mcp.tool()
def blob_exists(bucket_name: str, blob_name: str) -> Dict[str, Any]:
    """Check if a blob exists.

    Args:
        bucket_name: Name of the bucket.
        blob_name: Name of the blob.

    Returns:
        Dict containing success status and existence check result.
    """
    try:
        ensure_providers()
        exists = storage_provider.blob_exists(bucket_name, blob_name)
        return {
            "success": True,
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "exists": exists
        }
    except Exception as e:
        return handle_provider_error("blob_exists", e)


# ========== COPY/MOVE OPERATIONS ==========

@mcp.tool()
def copy_blob(source_bucket: str, source_blob: str, dest_bucket: str, dest_blob: str) -> Dict[str, Any]:
    """Copy a blob from one location to another.

    Args:
        source_bucket: Source bucket name.
        source_blob: Source blob name.
        dest_bucket: Destination bucket name.
        dest_blob: Destination blob name.

    Returns:
        Dict containing success status and copy details.
    """
    try:
        ensure_providers()
        success = storage_provider.copy_blob(
            source_bucket,
            source_blob,
            dest_bucket,
            dest_blob
        )
        return {
            "success": success,
            "source_bucket": source_bucket,
            "source_blob": source_blob,
            "dest_bucket": dest_bucket,
            "dest_blob": dest_blob
        }
    except Exception as e:
        return handle_provider_error("copy_blob", e)


@mcp.tool()
def move_blob(source_bucket: str, source_blob: str, dest_bucket: str, dest_blob: str) -> Dict[str, Any]:
    """Move a blob from one location to another.

    Args:
        source_bucket: Source bucket name.
        source_blob: Source blob name.
        dest_bucket: Destination bucket name.
        dest_blob: Destination blob name.

    Returns:
        Dict containing success status and move details.
    """
    try:
        ensure_providers()
        success = storage_provider.move_blob(
            source_bucket,
            source_blob,
            dest_bucket,
            dest_blob
        )
        return {
            "success": success,
            "source_bucket": source_bucket,
            "source_blob": source_blob,
            "dest_bucket": dest_bucket,
            "dest_blob": dest_blob
        }
    except Exception as e:
        return handle_provider_error("move_blob", e)


# ========== SIGNED URL OPERATIONS ==========

@mcp.tool()
def generate_signed_url(bucket_name: str, blob_name: str, expiration_hours: Optional[int] = None, method: str = "GET") -> Dict[str, Any]:
    """Generate a signed URL for temporary access to a blob.

    Args:
        bucket_name: Name of the bucket.
        blob_name: Name of the blob.
        expiration_hours: Optional hours until URL expires.
        method: HTTP method ('GET', 'PUT', 'POST', 'DELETE').

    Returns:
        Dict containing success status and signed URL.
    """
    try:
        ensure_providers()
        url = storage_provider.generate_signed_url(
            bucket_name,
            blob_name,
            expiration_hours,
            method
        )
        return {
            "success": True,
            "bucket_name": bucket_name,
            "blob_name": blob_name,
            "method": method,
            "expiration_hours": expiration_hours or storage_provider.config.DEFAULT_SIGNED_URL_EXPIRATION_HOURS,
            "signed_url": url
        }
    except Exception as e:
        return handle_provider_error("generate_signed_url", e)


# ========== BATCH OPERATIONS ==========

@mcp.tool()
def batch_upload(bucket_name: str, file_paths: List[str], prefix: str = "") -> Dict[str, Any]:
    """Upload multiple files to a bucket.

    Args:
        bucket_name: Name of the bucket.
        file_paths: List of file paths to upload.
        prefix: Optional prefix for blob names.

    Returns:
        Dict containing success status and upload results.
    """
    try:
        ensure_providers()
        results = storage_provider.batch_upload(
            bucket_name,
            file_paths,
            prefix
        )
        successful_count = sum(results.values())
        return {
            "success": True,
            "bucket_name": bucket_name,
            "prefix": prefix,
            "total_files": len(file_paths),
            "successful_uploads": successful_count,
            "failed_uploads": len(file_paths) - successful_count,
            "results": results
        }
    except Exception as e:
        return handle_provider_error("batch_upload", e)


@mcp.tool()
def batch_delete(bucket_name: str, blob_names: List[str]) -> Dict[str, Any]:
    """Delete multiple blobs from a bucket.

    Args:
        bucket_name: Name of the bucket.
        blob_names: List of blob names to delete.

    Returns:
        Dict containing success status and deletion results.
    """
    try:
        ensure_providers()
        results = storage_provider.batch_delete(
            bucket_name,
            blob_names
        )
        successful_count = sum(results.values())
        return {
            "success": True,
            "bucket_name": bucket_name,
            "total_blobs": len(blob_names),
            "successful_deletions": successful_count,
            "failed_deletions": len(blob_names) - successful_count,
            "results": results
        }
    except Exception as e:
        return handle_provider_error("batch_delete", e)


# ========== ANALYTICS OPERATIONS ==========

@mcp.tool()
def get_bucket_storage_info(bucket_name: str) -> Dict[str, Any]:
    """Get storage usage information for a bucket.

    Args:
        bucket_name: Name of the bucket.

    Returns:
        Dict containing success status and storage information.
    """
    try:
        ensure_providers()
        storage_info = storage_provider.get_bucket_storage_info(bucket_name)
        return {
            "success": True,
            "storage_info": storage_info
        }
    except Exception as e:
        return handle_provider_error("get_bucket_storage_info", e)


# ========== SEARCH OPERATIONS ==========

@mcp.tool()
def search_blobs(bucket_name: str, pattern: str, prefix: str = "") -> Dict[str, Any]:
    """Search for blobs matching a pattern.

    Args:
        bucket_name: Name of the bucket.
        pattern: Pattern to search for in blob names.
        prefix: Optional prefix to limit search scope.

    Returns:
        Dict containing success status and search results.
    """
    try:
        ensure_providers()
        matching_blobs = storage_provider.search_blobs(
            bucket_name,
            pattern,
            prefix
        )
        return {
            "success": True,
            "bucket_name": bucket_name,
            "pattern": pattern,
            "prefix": prefix,
            "matching_blobs": matching_blobs,
            "count": len(matching_blobs)
        }
    except Exception as e:
        return handle_provider_error("search_blobs", e)


@mcp.tool()
def filter_blobs_by_size(bucket_name: str, min_size_mb: float = 0, max_size_mb: float = float('inf')) -> Dict[str, Any]:
    """Filter blobs by size range.

    Args:
        bucket_name: Name of the bucket.
        min_size_mb: Minimum size in MB.
        max_size_mb: Maximum size in MB.

    Returns:
        Dict containing success status and filtered results.
    """
    try:
        ensure_providers()
        matching_blobs = storage_provider.filter_blobs_by_size(
            bucket_name,
            min_size_mb,
            max_size_mb
        )
        return {
            "success": True,
            "bucket_name": bucket_name,
            "min_size_mb": min_size_mb,
            "max_size_mb": max_size_mb,
            "matching_blobs": matching_blobs,
            "count": len(matching_blobs)
        }
    except Exception as e:
        return handle_provider_error("filter_blobs_by_size", e)


# ========== SERVER INITIALIZATION ==========

def initialize_providers(config: Optional[ProviderConfig] = None) -> None:
    """Initialize the GCP Storage and Project providers.

    Args:
        config: Optional configuration for the providers.
    """
    global storage_provider, project_provider

    try:
        logger.info("Initializing GCP Storage MCP Server...")

        # Initialize providers
        storage_provider = GCPStorageProvider(config)
        project_provider = GCPProjectProvider()

        logger.info(f"Providers initialized successfully for project: {storage_provider.get_current_project_id()}")

    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")
        raise


def main():
    """Main function to run the MCP server."""
    try:
        # Initialize providers
        initialize_providers()

        # Start the MCP server
        logger.info("Starting GCP Storage MCP Server...")
        mcp.run()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


initialize_providers()

if __name__ == "__main__":
    # Run the server
    main()
