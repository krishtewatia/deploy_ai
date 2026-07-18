"""FastAPI routes for dataset uploads."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import logging
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder

from backend.app.upload.upload_service import UnsupportedFileTypeError, UploadService
from backend.app.upload.validator import DatasetValidationError


logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def get_upload_service() -> UploadService:
    """Provide an UploadService instance for request handling."""
    return UploadService()


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_dataset(
    file: UploadFile = File(...),
    upload_service: UploadService = Depends(get_upload_service),
) -> Dict[str, Any]:
    """Upload a dataset file, validate it, and return metadata with preview data."""
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        logger.warning("Rejected unsupported upload file type: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .csv and .xlsx files are supported.",
        )

    # Check Content-Length header first (fast path)
    content_length = file.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE:
                logger.warning(
                    "Rejected upload: Content-Length %s exceeds limit of %d bytes",
                    content_length,
                    MAX_FILE_SIZE,
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds the maximum limit of 50 MB.",
                )
        except ValueError:
            pass

    temporary_path: Path | None = None

    try:
        total_bytes = 0
        with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = Path(temporary_file.name)
            # Read in chunks to prevent loading large files into memory
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    logger.warning(
                        "Rejected upload: File content size exceeded limit of %d bytes",
                        MAX_FILE_SIZE,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File size exceeds the maximum limit of 50 MB.",
                    )
                temporary_file.write(chunk)

        result = upload_service.process(str(temporary_path), original_filename=file.filename)
        return jsonable_encoder(result)

    except HTTPException:
        raise
    except (UnsupportedFileTypeError, DatasetValidationError, ValueError) as exc:
        logger.warning("Dataset upload validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected server error while processing uploaded dataset.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the uploaded file.",
        ) from exc
    finally:
        await file.close()

        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
