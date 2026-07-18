"""FastAPI routes for dataset analysis."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.analysis.analysis_service import AnalysisService
from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.upload.csv_loader import CSVLoader
from backend.app.upload.excel_loader import ExcelLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def get_analysis_service() -> AnalysisService:
    """Provide an AnalysisService instance for request handling."""
    return AnalysisService()


@router.post("/analyze", response_model=DatasetAnalysisReport, status_code=status.HTTP_200_OK)
async def analyze_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = None,
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> DatasetAnalysisReport:
    """Analyze an uploaded dataset and return data profiling metrics."""
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        logger.warning("Rejected unsupported analysis file type: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .csv, .xlsx, and .xls files are supported.",
        )

    # Content-Length check (fast path)
    content_length = file.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE:
                logger.warning("Rejected upload size via header: %s bytes", content_length)
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
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunk
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    logger.warning("Rejected upload size during streaming: >50 MB")
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File size exceeds the maximum limit of 50 MB.",
                    )
                temporary_file.write(chunk)

        # Load DataFrame using existing upload loaders
        if suffix == ".csv":
            df = CSVLoader().load(str(temporary_path))
        else:
            df = ExcelLoader().load(str(temporary_path))

        # Perform analysis using analysis service
        report = analysis_service.analyze(df, target_column=target_column)
        return report

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.warning("File not found during loading: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.warning("Validation or value error during analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("Runtime error from file loader: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected server error while processing analysis.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the analysis.",
        ) from exc
    finally:
        await file.close()
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
