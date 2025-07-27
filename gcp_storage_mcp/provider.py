"""This module contains the provider for the GCP Storage MCP."""
import os
import datetime
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager

from google.cloud import storage
from google.cloud.exceptions import NotFound, Conflict, Forbidden
from googleapiclient.discovery import build


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for GCP Storage Provider."""

    # Default values
    DEFAULT_LOCATION: str = "US"
    DEFAULT_STORAGE_CLASS: str = "STANDARD"
    DEFAULT_SIGNED_URL_EXPIRATION_HOURS: int = 1
    DEFAULT_BATCH_SIZE: int = 100
    MAX_BLOB_NAME_LENGTH: int = 1024
    MAX_BUCKET_NAME_LENGTH: int = 63

    # Validation patterns
    BUCKET_NAME_PATTERN = re.compile(r'^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$')
    BLOB_NAME_PATTERN = re.compile(r'^[^#\[\]]*$')  # Basic validation for blob names


class GCPStorageError(Exception):
    """Base exception for GCP Storage operations."""
    pass


class ValidationError(GCPStorageError):
    """Raised when input validation fails."""
    pass


class OperationError(GCPStorageError):
    """Raised when a storage operation fails."""
    pass


class GCPProjectProvider:
    """Provider for the GCP Project MCP."""

    def __init__(self):
        """Initialize the GCP Project Provider."""
        try:
            self.project_client = build("cloudresourcemanager", "v1")
            logger.info("GCP Project Provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GCP Project Provider: {e}")
            raise OperationError(f"Failed to initialize project provider: {e}")

    def list_projects(self) -> List[str]:
        """List all projects in the organization.

        Returns:
            List[str]: A list of project IDs.

        Raises:
            OperationError: If the operation fails.
        """
        try:
            projects = self.project_client.projects().list().execute()
            project_ids = [project["projectId"] for project in projects.get("projects", [])]
            logger.info(f"Retrieved {len(project_ids)} projects")
            return project_ids
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            raise OperationError(f"Failed to list projects: {e}")


class GCPStorageProvider:
    """Provider for the GCP Storage MCP with professional standards."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize the GCP Storage Provider.

        Args:
            config: Configuration object for the provider.

        Raises:
            OperationError: If initialization fails.
        """
        self.config = config or ProviderConfig()
        try:
            self.storage_client = storage.Client()
            logger.info(f"GCP Storage Provider initialized for project: {self.storage_client.project}")
        except Exception as e:
            logger.error(f"Failed to initialize GCP Storage Provider: {e}")
            raise OperationError(f"Failed to initialize storage provider: {e}")

    def _validate_bucket_name(self, bucket_name: str) -> None:
        """Validate bucket name according to GCS rules.

        Args:
            bucket_name: The bucket name to validate.

        Raises:
            ValidationError: If the bucket name is invalid.
        """
        if not bucket_name:
            raise ValidationError("Bucket name cannot be empty")

        if len(bucket_name) > self.config.MAX_BUCKET_NAME_LENGTH:
            raise ValidationError(f"Bucket name too long (max {self.config.MAX_BUCKET_NAME_LENGTH} chars)")

        if not self.config.BUCKET_NAME_PATTERN.match(bucket_name):
            raise ValidationError(
                "Bucket name must start and end with alphanumeric characters, "
                "and contain only lowercase letters, numbers, hyphens, periods, and underscores"
            )

    def _validate_blob_name(self, blob_name: str) -> None:
        """Validate blob name according to GCS rules.

        Args:
            blob_name: The blob name to validate.

        Raises:
            ValidationError: If the blob name is invalid.
        """
        if not blob_name:
            raise ValidationError("Blob name cannot be empty")

        if len(blob_name) > self.config.MAX_BLOB_NAME_LENGTH:
            raise ValidationError(f"Blob name too long (max {self.config.MAX_BLOB_NAME_LENGTH} chars)")

        if not self.config.BLOB_NAME_PATTERN.match(blob_name):
            raise ValidationError("Blob name contains invalid characters")

    @contextmanager
    def _get_bucket(self, bucket_name: str):
        """Context manager to get a bucket with validation.

        Args:
            bucket_name: The bucket name.

        Yields:
            storage.Bucket: The bucket object.

        Raises:
            ValidationError: If bucket name is invalid.
            OperationError: If bucket access fails.
        """
        self._validate_bucket_name(bucket_name)
        try:
            bucket = self.storage_client.bucket(bucket_name)
            yield bucket
        except NotFound:
            raise OperationError(f"Bucket '{bucket_name}' not found")
        except Forbidden:
            raise OperationError(f"Access denied to bucket '{bucket_name}'")
        except Exception as e:
            raise OperationError(f"Failed to access bucket '{bucket_name}': {e}")

    @contextmanager
    def _get_blob(self, bucket_name: str, blob_name: str):
        """Context manager to get a blob with validation.

        Args:
            bucket_name: The bucket name.
            blob_name: The blob name.

        Yields:
            storage.Blob: The blob object.

        Raises:
            ValidationError: If names are invalid.
            OperationError: If blob access fails.
        """
        self._validate_blob_name(blob_name)
        with self._get_bucket(bucket_name) as bucket:
            try:
                blob = bucket.blob(blob_name)
                yield blob
            except Exception as e:
                raise OperationError(f"Failed to access blob '{blob_name}': {e}")

    def get_current_project_id(self) -> str:
        """Get the current project ID.

        Returns:
            str: The current project ID.
        """
        return self.storage_client.project

    def list_buckets(self) -> List[str]:
        """List all buckets in the project.

        Returns:
            List[str]: A list of bucket names.

        Raises:
            OperationError: If the operation fails.
        """
        try:
            buckets = list(self.storage_client.list_buckets())
            bucket_names = [bucket.name for bucket in buckets]
            logger.info(f"Retrieved {len(bucket_names)} buckets")
            return bucket_names
        except Exception as e:
            logger.error(f"Failed to list buckets: {e}")
            raise OperationError(f"Failed to list buckets: {e}")

    def list_blobs(self, bucket_name: str, prefix: str = "") -> List[str]:
        """List all blobs in a bucket.

        Args:
            bucket_name: The name of the bucket to list blobs from.
            prefix: The prefix to list blobs from.

        Returns:
            List[str]: A list of blob names.

        Raises:
            ValidationError: If bucket name is invalid.
            OperationError: If the operation fails.
        """
        try:
            with self._get_bucket(bucket_name) as bucket:
                blobs = list(bucket.list_blobs(prefix=prefix))
                blob_names = [blob.name for blob in blobs]
                logger.info(f"Retrieved {len(blob_names)} blobs from bucket '{bucket_name}'")
                return blob_names
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Failed to list blobs in bucket '{bucket_name}': {e}")
            raise OperationError(f"Failed to list blobs: {e}")

    def list_folders(self, bucket_name: str, prefix: str = "") -> List[str]:
        """List all folders in a bucket.

        Args:
            bucket_name: The name of the bucket to list folders from.
            prefix: The prefix to list folders from.

        Returns:
            List[str]: A list of folder paths.

        Raises:
            ValidationError: If bucket name is invalid.
            OperationError: If the operation fails.
        """
        try:
            with self._get_bucket(bucket_name) as bucket:
                iterator = bucket.list_blobs(prefix=prefix, delimiter="/")
                folders = set()
                for page in iterator.pages:
                    if hasattr(page, "prefixes"):
                        folders.update(page.prefixes)
                folder_list = sorted(folders)
                logger.info(f"Retrieved {len(folder_list)} folders from bucket '{bucket_name}'")
                return folder_list
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Failed to list folders in bucket '{bucket_name}': {e}")
            raise OperationError(f"Failed to list folders: {e}")

    # ========== BLOB OPERATIONS ==========

    def upload_blob(self, bucket_name: str, blob_name: str, file_path: str) -> bool:
        """Upload a file to a bucket.

        Args:
            bucket_name: The name of the bucket to upload the file to.
            blob_name: The name of the blob to upload the file to.
            file_path: The path of the file to upload.

        Returns:
            bool: True if upload successful.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the upload fails.
        """
        # Validate file exists
        if not os.path.isfile(file_path):
            raise ValidationError(f"File not found: {file_path}")

        try:
            with self._get_blob(bucket_name, blob_name) as blob:
                blob.upload_from_filename(file_path)
                logger.info(f"Uploaded '{file_path}' to '{bucket_name}/{blob_name}'")
                return True
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Failed to upload '{file_path}' to '{bucket_name}/{blob_name}': {e}")
            raise OperationError(f"Upload failed: {e}")

    def download_blob(self, bucket_name: str, blob_name: str, file_path: str) -> bool:
        """Download a blob from a bucket.

        Args:
            bucket_name: The name of the bucket to download the blob from.
            blob_name: The name of the blob to download.
            file_path: The path to save the downloaded blob to.

        Returns:
            bool: True if download successful.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the download fails.
        """
        try:
            with self._get_blob(bucket_name, blob_name) as blob:
                # Check if blob exists
                if not blob.exists():
                    raise OperationError(f"Blob '{blob_name}' does not exist in bucket '{bucket_name}'")

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

                blob.download_to_filename(file_path)
                logger.info(f"Downloaded '{bucket_name}/{blob_name}' to '{file_path}'")
                return True
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Failed to download '{bucket_name}/{blob_name}': {e}")
            raise OperationError(f"Download failed: {e}")

    def delete_blob(self, bucket_name: str, blob_name: str) -> bool:
        """Delete a blob from a bucket.

        Args:
            bucket_name: The name of the bucket to delete the blob from.
            blob_name: The name of the blob to delete.

        Returns:
            bool: True if deletion successful.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the deletion fails.
        """
        try:
            with self._get_blob(bucket_name, blob_name) as blob:
                if not blob.exists():
                    raise OperationError(f"Blob '{blob_name}' does not exist in bucket '{bucket_name}'")

                blob.delete()
                logger.info(f"Deleted blob '{bucket_name}/{blob_name}'")
                return True
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Failed to delete blob '{bucket_name}/{blob_name}': {e}")
            raise OperationError(f"Deletion failed: {e}")

    def get_blob(self, bucket_name: str, blob_name: str) -> Optional[storage.Blob]:
        """Get a blob from a bucket.

        Args:
            bucket_name: The name of the bucket to get the blob from.
            blob_name: The name of the blob to get.

        Returns:
            Optional[storage.Blob]: The blob if it exists, None otherwise.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the operation fails.
        """
        try:
            with self._get_blob(bucket_name, blob_name) as blob:
                if not blob.exists():
                    logger.warning(f"Blob '{blob_name}' does not exist in bucket '{bucket_name}'")
                    return None

                blob.reload()  # Load metadata
                logger.debug(f"Retrieved blob '{bucket_name}/{blob_name}'")
                return blob
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Failed to get blob '{bucket_name}/{blob_name}': {e}")
            raise OperationError(f"Failed to get blob: {e}")

    def get_blob_url(self, bucket_name: str, blob_name: str) -> str:
        """Get the URL of a blob.

        Args:
            bucket_name: The name of the bucket to get the blob from.
            blob_name: The name of the blob to get.

        Returns:
            str: The URL of the blob.
        """
        self._validate_bucket_name(bucket_name)
        self._validate_blob_name(blob_name)
        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"

    def get_blob_size(self, bucket_name: str, blob_name: str) -> Optional[int]:
        """Get the size of a blob.

        Args:
            bucket_name: The name of the bucket to get the blob from.
            blob_name: The name of the blob to get.

        Returns:
            Optional[int]: The size of the blob in bytes, None if blob doesn't exist.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the operation fails.
        """
        blob = self.get_blob(bucket_name, blob_name)
        return blob.size if blob else None

    def get_blob_metadata(self, bucket_name: str, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get the metadata of a blob.

        Args:
            bucket_name: The name of the bucket to get the blob from.
            blob_name: The name of the blob to get.

        Returns:
            Optional[Dict[str, Any]]: The metadata of the blob, None if blob doesn't exist.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the operation fails.
        """
        blob = self.get_blob(bucket_name, blob_name)
        return blob.metadata or {} if blob else None

    def get_blob_content(self, bucket_name: str, blob_name: str) -> Optional[bytes]:
        """Get the content of a blob.

        Args:
            bucket_name: The name of the bucket to get the blob from.
            blob_name: The name of the blob to get.

        Returns:
            Optional[bytes]: The content of the blob, None if blob doesn't exist.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the operation fails.
        """
        try:
            with self._get_blob(bucket_name, blob_name) as blob:
                if not blob.exists():
                    logger.warning(f"Blob '{blob_name}' does not exist in bucket '{bucket_name}'")
                    return None

                content = blob.download_as_bytes()
                logger.debug(f"Downloaded content for blob '{bucket_name}/{blob_name}' ({len(content)} bytes)")
                return content
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Failed to get content for blob '{bucket_name}/{blob_name}': {e}")
            raise OperationError(f"Failed to get blob content: {e}")

    def get_blob_content_type(self, bucket_name: str, blob_name: str) -> Optional[str]:
        """Get the content type of a blob.

        Args:
            bucket_name: The name of the bucket to get the blob from.
            blob_name: The name of the blob to get.

        Returns:
            Optional[str]: The content type of the blob, None if blob doesn't exist.

        Raises:
            ValidationError: If input validation fails.
            OperationError: If the operation fails.
        """
        blob = self.get_blob(bucket_name, blob_name)
        return blob.content_type if blob else None

    # ========== BUCKET MANAGEMENT ==========

    def create_bucket(self, bucket_name: str, location: str = "US", storage_class: str = "STANDARD") -> bool:
        """Create a new bucket.

        Args:
            bucket_name (str): The name of the bucket to create.
            location (str): The location for the bucket (e.g., 'US', 'EU', 'us-central1').
            storage_class (str): The storage class ('STANDARD', 'NEARLINE', 'COLDLINE', 'ARCHIVE').

        Returns:
            bool: True if bucket created successfully, False otherwise.
        """
        try:
            with self._get_bucket(bucket_name) as bucket:
                bucket.storage_class = storage_class
                bucket = self.storage_client.create_bucket(bucket, location=location)
                logger.info(f"Bucket {bucket_name} created in {location} with storage class {storage_class}")
                return True
        except Conflict:
            logger.warning(f"Bucket {bucket_name} already exists")
            return False
        except Exception as e:
            logger.error(f"Error creating bucket {bucket_name}: {e}")
            raise OperationError(f"Failed to create bucket: {e}")

    def delete_bucket(self, bucket_name: str, force: bool = False) -> bool:
        """Delete a bucket.

        Args:
            bucket_name (str): The name of the bucket to delete.
            force (bool): If True, delete all objects in bucket first.

        Returns:
            bool: True if bucket deleted successfully, False otherwise.
        """
        try:
            with self._get_bucket(bucket_name) as bucket:

                if force:
                    # Delete all objects first
                    blobs = bucket.list_blobs()
                    for blob in blobs:
                        blob.delete()
                    logger.info(f"Deleted all objects in bucket {bucket_name}")

                bucket.delete()
                logger.info(f"Bucket {bucket_name} deleted successfully")
                return True
        except NotFound:
            logger.warning(f"Bucket {bucket_name} not found")
            return False
        except Exception as e:
            logger.error(f"Error deleting bucket {bucket_name}: {e}")
            raise OperationError(f"Failed to delete bucket: {e}")

    def get_bucket_info(self, bucket_name: str) -> Dict[str, Any] | None:
        """Get detailed information about a bucket.

        Args:
            bucket_name (str): The name of the bucket.

        Returns:
            Dict[str, Any] | None: Bucket information or None if not found.
        """
        try:
            with self._get_bucket(bucket_name) as bucket:
                bucket.reload()

                return {
                    "name": bucket.name,
                    "location": bucket.location,
                    "storage_class": bucket.storage_class,
                    "created": bucket.time_created.isoformat() if bucket.time_created else None,
                    "updated": bucket.updated.isoformat() if bucket.updated else None,
                    "versioning_enabled": bucket.versioning_enabled,
                    "labels": bucket.labels or {},
                    "lifecycle_rules": len(bucket.lifecycle_rules) if bucket.lifecycle_rules else 0,
                }
        except NotFound:
            logger.warning(f"Bucket {bucket_name} not found")
            return None
        except Exception as e:
            logger.error(f"Error getting bucket info for {bucket_name}: {e}")
            raise OperationError(f"Failed to get bucket info: {e}")

    # ========== COPY/MOVE OPERATIONS ==========

    def copy_blob(self, source_bucket: str, source_blob: str, dest_bucket: str, dest_blob: str) -> bool:
        """Copy a blob from one location to another.

        Args:
            source_bucket (str): Source bucket name.
            source_blob (str): Source blob name.
            dest_bucket (str): Destination bucket name.
            dest_blob (str): Destination blob name.

        Returns:
            bool: True if copy successful, False otherwise.
        """
        try:
            with self._get_blob(source_bucket, source_blob) as source_blob_obj:
                if not source_blob_obj.exists():
                    raise OperationError(f"Source blob {source_bucket}/{source_blob} does not exist")

            with self._get_bucket(dest_bucket) as dest_bucket_obj:
                _ = source_blob_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)

            logger.info(f"Copied {source_bucket}/{source_blob} to {dest_bucket}/{dest_blob}")
            return True
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Error copying blob {source_bucket}/{source_blob} to {dest_bucket}/{dest_blob}: {e}")
            raise OperationError(f"Failed to copy blob: {e}")

    def move_blob(self, source_bucket: str, source_blob: str, dest_bucket: str, dest_blob: str) -> bool:
        """Move a blob from one location to another.

        Args:
            source_bucket (str): Source bucket name.
            source_blob (str): Source blob name.
            dest_bucket (str): Destination bucket name.
            dest_blob (str): Destination blob name.

        Returns:
            bool: True if move successful, False otherwise.
        """
        try:
            # First copy the blob
            if self.copy_blob(source_bucket, source_blob, dest_bucket, dest_blob):
                # Then delete the original
                if self.delete_blob(source_bucket, source_blob):
                    logger.info(f"Moved {source_bucket}/{source_blob} to {dest_bucket}/{dest_blob}")
                    return True
                else:
                    logger.warning(f"Copied but failed to delete source blob {source_bucket}/{source_blob}")
                    return False
            return False
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Error moving blob {source_bucket}/{source_blob} to {dest_bucket}/{dest_blob}: {e}")
            raise OperationError(f"Failed to move blob: {e}")

    # ========== SIGNED URLS ==========

    def generate_signed_url(self, bucket_name: str, blob_name: str,
                          expiration_hours: int = 1, method: str = "GET") -> str | None:
        """Generate a signed URL for temporary access to a blob.

        Args:
            bucket_name (str): The bucket name.
            blob_name (str): The blob name.
            expiration_hours (int): Hours until the URL expires.
            method (str): HTTP method ('GET', 'PUT', 'POST', 'DELETE').

        Returns:
            str | None: The signed URL or None if failed.
        """
        try:
            with self._get_blob(bucket_name, blob_name) as blob:
                # Calculate expiration time
                expiration = datetime.datetime.now() + datetime.timedelta(hours=expiration_hours)

                url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration,
                    method=method,
                )

                logger.info(f"Generated signed URL for {bucket_name}/{blob_name} (expires in {expiration_hours}h)")
                return url
        except (ValidationError, OperationError):
            raise
        except Exception as e:
            logger.error(f"Error generating signed URL for {bucket_name}/{blob_name}: {e}")
            raise OperationError(f"Failed to generate signed URL: {e}")

    # ========== BATCH OPERATIONS ==========

    def batch_upload(self, bucket_name: str, file_paths: List[str], prefix: str = "") -> Dict[str, bool]:
        """Upload multiple files to a bucket.

        Args:
            bucket_name (str): The bucket name.
            file_paths (List[str]): List of local file paths to upload.
            prefix (str): Optional prefix for blob names.

        Returns:
            Dict[str, bool]: Mapping of file paths to success status.
        """
        results = {}
        with self._get_bucket(bucket_name) as bucket:

            for file_path in file_paths:
                try:
                    if not os.path.exists(file_path):
                        logger.warning(f"File not found: {file_path}")
                        results[file_path] = False
                        continue

                    # Generate blob name from file path
                    blob_name = os.path.basename(file_path)
                    if prefix:
                        blob_name = f"{prefix.rstrip('/')}/{blob_name}"

                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(file_path)
                    results[file_path] = True
                    logger.info(f"Uploaded {file_path} to {bucket_name}/{blob_name}")

                except Exception as e:
                    logger.error(f"Error uploading {file_path}: {e}")
                    results[file_path] = False

        return results

    def batch_delete(self, bucket_name: str, blob_names: List[str]) -> Dict[str, bool]:
        """Delete multiple blobs from a bucket.

        Args:
            bucket_name (str): The bucket name.
            blob_names (List[str]): List of blob names to delete.

        Returns:
            Dict[str, bool]: Mapping of blob names to success status.
        """
        results = {}
        with self._get_bucket(bucket_name) as bucket:

            for blob_name in blob_names:
                try:
                    blob = bucket.blob(blob_name)
                    if blob.exists():
                        blob.delete()
                        results[blob_name] = True
                        logger.info(f"Deleted {bucket_name}/{blob_name}")
                    else:
                        logger.warning(f"Blob {blob_name} not found")
                        results[blob_name] = False
                except Exception as e:
                    logger.error(f"Error deleting {blob_name}: {e}")
                    results[blob_name] = False

        return results
