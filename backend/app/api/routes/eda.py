"""FastAPI routes for Exploratory Data Analysis (EDA) processing."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.ai_engine.groq_client import GroqClient
from backend.app.analysis.analysis_service import AnalysisService
from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.eda.chart_generator import ChartGenerator
from backend.app.eda.descriptive_analyzer import DescriptiveAnalyzer
from backend.app.eda.diagnostic_analyzer import DiagnosticAnalyzer
from backend.app.eda.eda_service import EDAService, EDAServiceError
from backend.app.eda.eda_service_schemas import EDAServiceReport
from backend.app.eda.insight_generator import InsightGenerator
from backend.app.eda.visualization_recommender import VisualizationRecommender
from backend.app.upload.csv_loader import CSVLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/eda", tags=["EDA"])

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ── Dependency Injection Providers ──────────────────────────────────────────


def get_analysis_service() -> AnalysisService:
    """Provide an AnalysisService instance."""
    return AnalysisService()


def get_groq_client() -> GroqClient:
    """Provide a GroqClient instance."""
    return GroqClient()


def get_insight_generator(
    groq_client: GroqClient = Depends(get_groq_client),
) -> InsightGenerator:
    """Provide an InsightGenerator instance."""
    return InsightGenerator(groq_client=groq_client)


def get_chart_generator() -> ChartGenerator:
    """Provide a ChartGenerator instance."""
    return ChartGenerator()


def get_eda_service(
    insight_generator: InsightGenerator = Depends(get_insight_generator),
    chart_generator: ChartGenerator = Depends(get_chart_generator),
) -> EDAService:
    """Provide an EDAService instance coordinating EDA sub-components."""
    return EDAService(
        descriptive_analyzer=DescriptiveAnalyzer(),
        diagnostic_analyzer=DiagnosticAnalyzer(),
        visualization_recommender=VisualizationRecommender(),
        chart_generator=chart_generator,
        insight_generator=insight_generator,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post(
    "/analyze",
    response_model=EDAServiceReport,
    status_code=status.HTTP_200_OK,
    summary="Conduct Exploratory Data Analysis on a CSV file",
    description=(
        "Upload a dataset in CSV format, run data quality analysis, "
        "generate visualization recommendations, save rendered charts to disk, "
        "and generate AI insights using the Groq LLM."
    ),
)
async def analyze_dataset(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> EDAServiceReport:
    """Expose the EDA pipeline execution via POST."""
    suffix = Path(file.filename or "").suffix.lower()

    if suffix != ".csv":
        logger.warning("Rejected unsupported file type: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .csv files are supported.",
        )

    # Content-Length check (fast path)
    content_length = file.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE:
                logger.warning("Rejected file size via header: %s bytes", content_length)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds the maximum limit of 50 MB.",
                )
        except ValueError:
            pass

    temporary_path: Path | None = None

    try:
        # Stream file to a temporary location to prevent loading large files in memory
        total_bytes = 0
        with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunk
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    logger.warning("Rejected file size during streaming: >50 MB")
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File size exceeds the maximum limit of 50 MB.",
                    )
                temporary_file.write(chunk)

        # Load CSV using loader helper (runs basic format validation checks)
        df = CSVLoader().load(str(temporary_path))

        # Perform analysis profiling
        analysis_report = analysis_service.analyze(df)

        # Resolve dataset identifier (use stem of filename plus random suffix if not provided)
        if not dataset_id or not dataset_id.strip():
            filename_stem = Path(file.filename or "dataset").stem
            clean_stem = "".join(c if c.isalnum() or c == "_" else "_" for c in filename_stem)
            dataset_id = f"{clean_stem}_{uuid.uuid4().hex[:8]}"

        # Run primary orchestrator service
        report = eda_service.run(dataset_id, df, analysis_report)
        return report

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions directly
        raise
    except EDAServiceError as exc:
        logger.warning("EDA Pipeline Execution Error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.warning("File validation or parsing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected server error while processing EDA route.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the analysis.",
        ) from exc
    finally:
        await file.close()
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
