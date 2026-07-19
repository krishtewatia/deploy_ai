"""Unit and integration tests for DeployAIEngine orchestrator."""

import os
import tempfile
import pandas as pd
import pytest

from backend.app.pipeline.engine import DeployAIEngine
from backend.app.pipeline.schemas import PipelineStage, PipelineStatus
from backend.app.ml_plan.orchestrator import PlanningMode


@pytest.fixture
def sample_csv_path():
    """Create a temporary CSV file with synthetic classification data."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10,
        "feature2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 10,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        path = tmp.name

    yield path

    if os.path.exists(path):
        os.unlink(path)


def test_deploy_ai_engine_full_pipeline_run(sample_csv_path):
    os.environ["GROQ_API_KEY"] = "mock_key"
    engine = DeployAIEngine()

    progress_stages = []

    def progress_cb(stage: PipelineStage, progress: float):
        progress_stages.append(stage)

    context = engine.run_pipeline(
        dataset_path=sample_csv_path,
        target_column="target",
        planning_mode=PlanningMode.DETERMINISTIC,
        progress_callback=progress_cb,
    )

    assert context.status == PipelineStatus.COMPLETED
    assert context.error_message is None
    assert len(context.completed_stages) == 11

    # Check key stage outputs populated
    assert context.dataset_metadata is not None
    assert context.dataset_metadata.rows == 100
    assert context.analysis_report is not None
    assert context.dataset_context is not None
    assert context.problem_definition is not None
    assert context.problem_definition.target_column == "target"
    assert context.hardware_profile is not None
    assert context.compute_capabilities is not None
    assert context.ml_plan is not None
    assert context.execution_result is not None
    assert context.champion_decision is not None
    assert context.champion_decision.winner_report.champion_summary.candidate_id is not None
    assert context.ai_explanation is not None
    assert context.executive_report is not None
    assert context.export_result is not None


def test_deploy_ai_engine_get_context(sample_csv_path):
    os.environ["GROQ_API_KEY"] = "mock_key"
    engine = DeployAIEngine()

    context = engine.run_pipeline(
        dataset_path=sample_csv_path,
        planning_mode=PlanningMode.DETERMINISTIC,
    )

    retrieved = engine.get_context(context.pipeline_id)
    assert retrieved is context
