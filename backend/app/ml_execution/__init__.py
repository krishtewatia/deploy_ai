"""ML Execution Schemas Package.

Defines schemas and enums used for pipeline execution metadata, progress,
artifacts, and metrics.
"""

from backend.app.ml_execution.schemas import (
    ExecutionStatus,
    ExecutionStage,
    ArtifactType,
    WarningSeverity,
    ExecutionArtifact,
    ExecutionWarning,
    TrainingMetrics,
    ExecutionProgress,
    ExecutionConstraintsSnapshot,
    ExecutionResult,
)

__all__ = [
    "ExecutionStatus",
    "ExecutionStage",
    "ArtifactType",
    "WarningSeverity",
    "ExecutionArtifact",
    "ExecutionWarning",
    "TrainingMetrics",
    "ExecutionProgress",
    "ExecutionConstraintsSnapshot",
    "ExecutionResult",
]

from backend.app.ml_execution.split_executor import (
    SplitExecutor,
    SplitExecutorError,
)
from backend.app.ml_execution.preprocessing_builder import (
    PreprocessingPipelineBuilder,
    PreprocessingPipelineBuilderError,
)
from backend.app.ml_execution.feature_engineering_executor import (
    FeatureEngineeringExecutor,
    FeatureEngineeringExecutorError,
    FeatureEngineeringResult,
)
from backend.app.ml_execution.feature_selection_executor import (
    FeatureSelectionExecutor,
    FeatureSelectionExecutorError,
    FeatureSelectionResult,
)
from backend.app.ml_execution.model_factory import (
    ModelFactory,
    ModelFactoryError,
    ModelFactoryResult,
)
from backend.app.ml_execution.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    HyperparameterOptimizerError,
    HyperparameterOptimizationResult,
)
from backend.app.ml_execution.training_executor import (
    TrainingExecutor,
    TrainingExecutorError,
    TrainingResult,
)
from backend.app.ml_execution.evaluation_engine import (
    EvaluationEngine,
    EvaluationEngineError,
    EvaluationResult,
)
from backend.app.ml_execution.orchestrator import (
    MLExecutionOrchestrator,
    MLExecutionOrchestratorError,
    MLExecutionResult,
)
from backend.app.ml_execution.execution_report import (
    ExecutionReport,
    ExecutionReportBuilder,
    ExecutionReportBuilderError,
)

__all__.extend([
    "SplitExecutor",
    "SplitExecutorError",
    "PreprocessingPipelineBuilder",
    "PreprocessingPipelineBuilderError",
    "FeatureEngineeringExecutor",
    "FeatureEngineeringExecutorError",
    "FeatureEngineeringResult",
    "FeatureSelectionExecutor",
    "FeatureSelectionExecutorError",
    "FeatureSelectionResult",
    "ModelFactory",
    "ModelFactoryError",
    "ModelFactoryResult",
    "HyperparameterOptimizer",
    "HyperparameterOptimizerError",
    "HyperparameterOptimizationResult",
    "TrainingExecutor",
    "TrainingExecutorError",
    "TrainingResult",
    "EvaluationEngine",
    "EvaluationEngineError",
    "EvaluationResult",
    "MLExecutionOrchestrator",
    "MLExecutionOrchestratorError",
    "MLExecutionResult",
    "ExecutionReport",
    "ExecutionReportBuilder",
    "ExecutionReportBuilderError",
])

