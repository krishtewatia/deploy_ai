# DeployAI — AI-Powered MLOps & Auto-ML Platform

DeployAI is a production-grade, end-to-end MLOps backend platform that automates the complete machine learning lifecycle — from raw dataset ingestion and intelligent preprocessing through AI-assisted planning, multi-candidate model training, hyperparameter optimization, evaluation, AI critique reviews, deterministic model governance, and executive reporting.

Built on **FastAPI**, the backend is architected as a modular, layered pipeline of 26 purpose-built services that communicate through strict Pydantic schema contracts, ensuring full traceability, reproducibility, and auditability at every stage.

---

## 📐 High-Level Architecture

```
                         ┌─────────────────────────────────────┐
                         │           FastAPI Gateway           │
                         │   /upload  /analysis  /training     │
                         │   /preprocessing  /models  /reports │
                         └──────────────┬──────────────────────┘
                                        │
          ┌─────────────────────────────┼───────────────────────────┐
          │                             │                           │
   ┌──────▼──────┐              ┌───────▼───────┐           ┌───────▼────────┐
   │   Upload    │              │   Analysis    │           │   Workspace    │
   │   Module    │              │   Module      │           │   Manager      │
   └──────┬──────┘              └──────┬────────┘           └───────┬────────┘
          │                            │                            │
          ▼                            ▼                            ▼
   ┌─────────────┐            ┌────────────────┐           ┌───────────────┐
   │  Dataset    │            │  Preprocessing │           │  Recent       │
   │  Intellig.  │            │  Engine        │           │  Registry     │
   └──────┬──────┘            └───────┬────────┘           └───────────────┘
          │                           │
          ▼                           ▼
   ┌─────────────┐            ┌────────────────┐
   │  Problem    │            │  Hardware      │
   │  Resolver   │            │  Detector      │
   └──────┬──────┘            └───────┬────────┘
          │                           │
          ├───────────────────────────┘
          ▼
   ┌─────────────────────────────────────────┐
   │         ML Planning Orchestrator        │
   │  ┌─────────────┐  ┌──────────────────┐  │
   │  │  Baseline   │  │  AI Decision     │  │
   │  │  Planner    │  │  Service (LLM)   │  │
   │  └──────┬──────┘  └───────┬──────────┘  │
   │         └────────┬────────┘             │
   │           ┌──────▼──────┐               │
   │           │  Plan       │               │
   │           │  Validator  │               │
   │           └─────────────┘               │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │        ML Execution Orchestrator        │
   │                                         │
   │  Split → Preprocess → Feature Eng.      │
   │  → Feature Select → Model Factory       │
   │  → Hyperparameter Opt → Train → Eval    │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │          Execution Report Builder       │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │          AI Model Critic (LLM)          │
   │  Context → Prompt → Generate → Parse    │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │          AI Model Optimizer             │
   │  Recommendation Mapper → Plan Optimizer │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │          Retraining Engine              │
   │  Validate → Execute → Report            │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │          Champion Comparator            │
   │  Baseline vs Retrained Governance       │
   └──────────────────┬──────────────────────┘
                      ▼
   ┌─────────────────────────────────────────┐
   │     Executive Report Generator          │
   │  Final Consolidated Training Report     │
   └─────────────────────────────────────────┘
```

---

## 🏗️ Backend Module Architecture

The backend is organized into **26 service modules** under `backend/app/`, each with its own schemas, services, and domain logic.

---

### 1. Core (`backend/app/core/`)

The foundational configuration layer for the entire platform.

| File | Purpose |
|------|---------|
| `config.py` | Centralized settings via `pydantic-settings`. Loads `.env` automatically. Validates `GROQ_API_KEY`, CORS origins, environment, and app name at import-time. |
| `logger.py` | Logging configuration bootstrapper. |

**Key settings**:
- `GROQ_API_KEY` — Required API key for Groq LLM service
- `cors_origins` — Configurable CORS whitelist (comma-separated or JSON list)
- `environment` — Runtime mode (`development` / `staging` / `production`)

---

### 2. API Gateway (`backend/app/api/`)

Six FastAPI routers that form the HTTP surface of the platform.

| Router | Prefix | Responsibility |
|--------|--------|----------------|
| `upload.py` | `/upload` | File upload endpoint — accepts CSV/XLSX, validates, parses, and returns structured metadata + data preview |
| `analysis.py` | `/analysis` | Dataset analysis endpoint — profiles columns, detects missing values, duplicates, class imbalance, and summary statistics |
| `preprocessing.py` | `/preprocessing` | Preprocessing pipeline configuration endpoint |
| `training.py` | `/training` | Model training trigger and status endpoint |
| `models.py` | `/models` | Trained model registry and retrieval endpoint |
| `reports.py` | `/reports` | Execution and executive report retrieval endpoint |

All routers are registered in `main.py` with CORS middleware configured for cross-origin access.

---

### 3. Upload Module (`backend/app/upload/`)

Handles dataset ingestion with format-specific loaders and validation.

| File | Purpose |
|------|---------|
| `csv_loader.py` | Parses CSV files into pandas DataFrames with encoding detection |
| `excel_loader.py` | Parses XLSX/XLS files with multi-sheet support |
| `validator.py` | Validates file size, structure, column types, and data integrity |
| `metadata_extractor.py` | Extracts shape, column types, null counts, and basic stats |
| `preview_generator.py` | Generates first-N-rows previews for frontend display |
| `upload_service.py` | Orchestrates the full upload pipeline: load → validate → extract → preview |
| `schemas.py` | Pydantic models for upload requests and responses |

---

### 4. Analysis Module (`backend/app/analysis/`)

Produces comprehensive data profiling reports from uploaded datasets.

| File | Purpose |
|------|---------|
| `analysis_service.py` | Orchestrates column profiling, missing value analysis, duplicate detection, imbalance checks, and statistics generation |
| `column_profiler.py` | Per-column statistical profiling (distributions, cardinality, nulls) |
| `missing_values.py` | Detects and quantifies missing data patterns |
| `duplicates.py` | Identifies duplicate rows and columns |
| `imbalance.py` | Detects class distribution imbalance in target columns |
| `statistics.py` | Computes dataset-wide aggregated statistics |
| `report_generator.py` | Compiles analysis results into a structured `DatasetAnalysisReport` |
| `schemas.py` | Pydantic models for analysis outputs |

---

### 5. Dataset Intelligence (`backend/app/dataset_intelligence/`)

Builds a rich `DatasetContext` profile used by all downstream planning and execution systems.

| File | Purpose |
|------|---------|
| `context_builder.py` | Constructs the `DatasetContext` object — a comprehensive metadata profile containing column statistics, type inference, cardinality analysis, and feature quality signals |
| `schemas.py` | `DatasetContext`, `BasicInfo`, `ColumnProfile`, and related Pydantic models |

---

### 6. Problem Definition (`backend/app/problem_definition/`)

Resolves user intent into a formal ML problem specification.

| File | Purpose |
|------|---------|
| `resolver.py` | `ProblemResolver` — infers problem type (classification/regression), identifies target column, selects feature columns, chooses primary metric, and resolves ambiguities |
| `schemas.py` | `ProblemDefinition`, `ProblemType`, `ResolutionStatus`, `TargetSource` enums and models |

**Resolution logic**:
- Auto-detects classification vs. regression from target cardinality and dtype
- Selects appropriate default metrics (`accuracy` for classification, `rmse` for regression)
- Excludes high-cardinality identifiers and constant columns from features

---

### 7. ML Request (`backend/app/ml_request/`)

Captures and validates user-facing ML training requests.

| File | Purpose |
|------|---------|
| `schemas.py` | `UserMLRequest` — the structured user intent (goal, constraints, preferences, target column hints) |

---

### 8. Compute Capabilities (`backend/app/compute_capabilities/`)

Profiles the hardware environment to constrain planning decisions.

| File | Purpose |
|------|---------|
| `analyzer.py` | `ComputeCapabilityAnalyzer` — detects CPU cores, RAM, GPU availability, and derives `ComputeTier` |
| `schemas.py` | `ComputeCapabilities`, `ComputeTier`, `AcceleratorType` enums |

**Compute tiers**: `minimal` → `standard` → `performance` → `high_performance`
**Accelerator types**: `cpu`, `cuda`, `rocm`, `mps`, `tpu`

---

### 9. Hardware Detection (`backend/app/hardware/`)

Low-level hardware discovery layer powered by `psutil`.

| File | Purpose |
|------|---------|
| `detector.py` | `HardwareDetector` — discovers CPU model, core count, RAM, disk, GPU vendor/VRAM via system introspection |
| `schemas.py` | `HardwareProfile`, `CPUInfo`, `GPUInfo`, `MemoryInfo` models |

---

### 10. ML Plan (`backend/app/ml_plan/`)

The core planning layer — generates, validates, and manages ML execution plans.

| File | Purpose |
|------|---------|
| `baseline_planner.py` | `BaselineMLPlanner` — deterministic rule-based planner that generates a complete `MLPlan` from `DatasetContext`, `ProblemDefinition`, and `ComputeCapabilities`. Selects model families, preprocessing steps, feature engineering, split strategy, and evaluation plan |
| `orchestrator.py` | `MLPlanningOrchestrator` — high-level coordinator that chains: Problem Resolution → Baseline Planning → (optional) AI-Assisted Planning → Plan Validation |
| `validator.py` | `MLPlanValidator` — comprehensive safety-gate validator checking schema integrity, column existence, model compatibility, metric validity, and compute feasibility |
| `schemas.py` | `MLPlan`, `ModelCandidate`, `SplitPlan`, `EvaluationPlan`, `FeatureSelectionPlan`, `ModelFamily`, `SearchStrategy` and 20+ related models |

**Planning modes**:
- **`DETERMINISTIC`** — Pure rule-based planning (no AI)
- **`AI_ASSISTED`** — Baseline plan enhanced by LLM proposals via `AIDecisionService`

---

### 11. AI Planning (`backend/app/ai_planning/`)

The AI-assisted planning pipeline that enhances baseline plans using LLM intelligence.

| File | Purpose |
|------|---------|
| `decision_service.py` | `AIDecisionService` — end-to-end orchestrator: Context → Prompt → LLM Generate → Parse → Merge → Validate |
| `context_builder.py` | `AIPlanningContextBuilder` — serializes `DatasetContext`, `ProblemDefinition`, `ComputeCapabilities`, and baseline plan into structured context for the LLM |
| `prompt_builder.py` | `AIPlanningPromptBuilder` — constructs system and user prompts with expert ML instructions |
| `response_parser.py` | `AIResponseParser` — extracts and validates structured `AIDecisionProposal` from raw LLM text |
| `proposal_merger.py` | `ProposalMerger` — safely merges AI proposals into the baseline plan with validation guards |
| `schemas.py` | `AIDecisionProposal`, `AIAssistedPlanningResult`, and proposal sub-schemas |
| `providers/base.py` | Abstract `AIProvider` interface with `generate(system_prompt, user_prompt)` contract |

---

### 12. AI Providers (`backend/app/ai_providers/`)

Pluggable LLM provider management with multi-backend support.

| File | Purpose |
|------|---------|
| `factory.py` | `AIProviderFactory` — creates provider instances by type (Groq, OpenAI-compatible, Ollama) |
| `presets.py` | Pre-configured provider profiles for popular LLM services |
| `schemas.py` | Provider configuration models, endpoint schemas, authentication types |
| `secret_store.py` | Secure API key management via OS keyring integration |
| `settings_store.py` | Persistent provider settings storage and retrieval |
| `validation.py` | Provider configuration validation (endpoint reachability, auth, model availability) |

---

### 13. AI Engine (`backend/app/ai_engine/`)

Legacy AI recommendation engine using Groq for ML plan suggestions.

| File | Purpose |
|------|---------|
| `groq_client.py` | Direct Groq API client wrapper with retry logic |
| `prompt_builder.py` | ML-specific prompt engineering for recommendation generation |
| `plan_parser.py` | Parses LLM responses into structured plan recommendations |
| `recommendation_service.py` | `RecommendationService` — end-to-end recommendation pipeline |
| `schemas.py` | Recommendation request/response models |

---

### 14. Preprocessing Engine (`backend/app/preprocessing_engine/`)

Configurable data preprocessing pipeline with modular handlers.

| File | Purpose |
|------|---------|
| `preprocessing_service.py` | Master service orchestrating the full preprocessing pipeline |
| `pipeline_executor.py` | Executes ordered preprocessing steps with rollback support |
| `missing_handler.py` | Missing value imputation strategies (mean, median, mode, drop, constant) |
| `duplicate_handler.py` | Duplicate row/column removal |
| `encoding_handler.py` | Categorical encoding (one-hot, label, ordinal, target) |
| `scaling_handler.py` | Feature scaling (standard, min-max, robust, max-abs) |
| `schemas.py` | `PreprocessingStep`, `PreprocessingPlan` models |
| `result_schemas.py` | Pipeline execution result models |

---

### 15. ML Execution (`backend/app/ml_execution/`)

The core execution engine — runs the complete training pipeline for all model candidates.

| File | Purpose |
|------|---------|
| `orchestrator.py` | `MLExecutionOrchestrator` — the 8-step deterministic pipeline: Split → Preprocess → Feature Engineer → Feature Select → Model Factory → Hyperparameter Optimize → Train → Evaluate |
| `split_executor.py` | `SplitExecutor` — train/test/validation splitting with stratification support |
| `preprocessing_builder.py` | `PreprocessingPipelineBuilder` — builds sklearn-compatible preprocessing pipelines from plan specs |
| `feature_engineering_executor.py` | `FeatureEngineeringExecutor` — creates polynomial, interaction, and derived features |
| `feature_selection_executor.py` | `FeatureSelectionExecutor` — mutual information, model-based, and variance threshold selection |
| `model_factory.py` | `ModelFactory` — instantiates sklearn estimators by `ModelFamily` with initial parameters |
| `hyperparameter_optimizer.py` | `HyperparameterOptimizer` — Grid Search, Random Search, and Bayesian optimization |
| `training_executor.py` | `TrainingExecutor` — fits estimators with timing and error handling |
| `evaluation_engine.py` | `EvaluationEngine` — computes metrics, confusion matrices, feature importance, and cross-validation scores |
| `execution_report.py` | `ExecutionReportBuilder` — consolidates all artifacts into an immutable `ExecutionReport` |
| `schemas.py` | `MLExecutionResult`, `EvaluationResult`, `TrainingResult`, `SplitResult` models |

**Execution pipeline** (per-candidate loop):
```
MLPlan
  │
  ├── 1. SplitExecutor         → train/test sets
  ├── 2. PreprocessingBuilder   → sklearn pipeline (fit on train, transform test)
  ├── 3. FeatureEngineeringExec → derived features
  ├── 4. FeatureSelectionExec   → column pruning
  ├── 5. ModelFactory           → instantiate estimators
  │
  └── Per Model Candidate:
      ├── 6. HyperparameterOpt  → optimized estimator
      ├── 7. TrainingExecutor   → fitted model
      └── 8. EvaluationEngine   → metrics + feature importance

Best Model Selection → by primary metric (higher/lower-is-better aware)
```

**Supported model families**: `random_forest`, `gradient_boosting`, `logistic_regression`, `linear_regression`, `svm`, `knn`, `decision_tree`, `extra_trees`, `ada_boost`, `naive_bayes`, `ridge`, `lasso`, `elastic_net`, `sgd`

**Supported metrics**: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `r2`, `mae`, `mse`, `rmse`

---

### 16. AI Model Critic (`backend/app/ai_model_critic/`)

Post-training AI review system that critiques execution results using LLM intelligence.

| File | Purpose |
|------|---------|
| `critic_service.py` | `AIModelCritic` — 5-step review pipeline: Validate → Build Context → Build Prompt → LLM Generate → Parse Critique |
| `context_builder.py` | Serializes `ExecutionReport` (metrics, candidates, champion, warnings) into structured JSON for LLM review |
| `prompt_builder.py` | Expert ML reviewer system/user prompt construction |
| `response_parser.py` | Extracts `ModelCritique` from raw LLM text with JSON validation |
| `schemas.py` | `ModelCritique` — overall grade, confidence, strengths, weaknesses, risks, and recommendations |

**Critique output**:
- Overall grade (A/B/C/D/F)
- Confidence score (0.0–1.0)
- Identified strengths, weaknesses, and risks
- Actionable recommendations

---

### 17. AI Model Optimizer (`backend/app/ai_model_optimizer/`)

Converts AI critique recommendations into deterministic plan modifications.

| File | Purpose |
|------|---------|
| `optimizer_service.py` | `AIModelOptimizer` — orchestrates: Map Recommendations → Optimize Plan → Return Result |
| `recommendation_mapper.py` | `AIRecommendationMapper` — translates natural-language critique items into typed `OptimizationAction` objects |
| `plan_optimizer.py` | `AIPlanOptimizer` — applies mapped actions to the baseline `MLPlan` (add candidates, adjust hyperparameters, change search strategy) |
| `retraining_engine.py` | `RetrainingEngine` — validates identity consistency, re-validates the optimized plan, executes full ML pipeline, and builds an execution report |
| `schemas.py` | `OptimizationResult`, `OptimizationAction` models |

**Optimization actions**: `ADD_CANDIDATE`, `CHANGE_SEARCH_STRATEGY`, `ADJUST_HYPERPARAMETERS`, `MODIFY_PREPROCESSING`, `NO_ACTION`

---

### 18. Model Governance (`backend/app/model_governance/`)

Deterministic champion model selection through baseline vs. retrained comparison.

| File | Purpose |
|------|---------|
| `comparator.py` | `ChampionComparator` — compares baseline and retrained `ExecutionReport` objects, validates identity consistency (dataset, problem, target), computes relative improvement, and selects winner |
| `schemas.py` | `ChampionDecision`, `Winner` enum (`BASELINE`, `RETRAINED`, `TIE`) |

**Decision logic**:
- Identity validation (dataset_id, problem_definition_id, target_column must match)
- Metric-direction-aware comparison (higher-is-better vs. lower-is-better)
- Tie detection (< 1e-9 absolute difference)
- Relative improvement calculation
- Production readiness flag propagation

---

### 19. Reporting (`backend/app/reporting/`)

Generates the final consolidated executive report.

| File | Purpose |
|------|---------|
| `report_generator.py` | `ExecutiveReportGenerator` — merges `ExecutionReport`, `ModelCritique`, `OptimizationResult`, and `ChampionDecision` into a single `ExecutiveReport` |
| `schemas.py` | `ExecutiveReport` — problem summary, dataset summary, pipeline summary, model comparisons, champion details, optimization actions, AI review, governance decision, and deployment recommendations |

**Report sections**:
- Problem & Dataset Summary
- Pipeline Configuration Summary
- Per-Model Candidate Results
- Champion Model Details
- Optimization Actions Applied
- AI Critique Review (grade, confidence, risks)
- Governance Decision (winner, improvement %)
- Deployment Readiness & Monitoring Recommendations

---

### 20. AutoML (`backend/app/automl/`)

Reserved module for higher-level automated ML orchestration.

| File | Purpose |
|------|---------|
| `evaluation.py` | Evaluation utilities (stub) |
| `hyperparameter_tuning.py` | Tuning utilities (stub) |
| `model_selector.py` | Model selection utilities (stub) |
| `trainer.py` | Training utilities (stub) |

---

### 21. Workspace (`backend/app/workspace/`)

Project workspace lifecycle management.

| File | Purpose |
|------|---------|
| `orchestrator.py` | `WorkspaceOrchestrator` — high-level project operations: create, open, delete, validate, pin/unpin |
| `manager.py` | `WorkspaceManager` — filesystem-level workspace CRUD operations |
| `recent_registry.py` | `RecentProjectsRegistry` — tracks recently opened projects with pinning support |
| `schemas.py` | `ProjectWorkspace`, `ProjectMetadata`, `RecentProjectEntry` models |

---

### 22. Exports (`backend/app/exports/`)

Trained model serialization in multiple formats.

| File | Purpose |
|------|---------|
| `pickle_export.py` | Python pickle serialization |
| `onnx_export.py` | ONNX format export |
| `keras_export.py` | Keras/TF SavedModel export |

---

### 23. Explainability (`backend/app/explainability/`)

Reserved module for model interpretability features (SHAP, LIME, etc.).

---

### 24. Optimization (`backend/app/optimization/`)

Reserved module for deployment optimization utilities.

---

### 25. Detector (`backend/app/detector/`)

Reserved module for data quality anomaly detection.

---

### 26. Utils (`backend/app/utils/`)

Shared utility functions used across modules.

---

## 🔄 End-to-End Training Pipeline

The complete model training lifecycle executes as a **14-stage pipeline**:

```
Stage  1  │  Upload & Ingest        │  CSV/XLSX → pandas DataFrame
Stage  2  │  Dataset Analysis       │  Profiling, missing values, duplicates, imbalance
Stage  3  │  Dataset Intelligence   │  Build DatasetContext metadata profile
Stage  4  │  Hardware Detection     │  CPU, RAM, GPU → ComputeCapabilities
Stage  5  │  Problem Resolution     │  Infer problem type, target, features, metric
Stage  6  │  Baseline Planning      │  Deterministic MLPlan generation
Stage  7  │  AI-Assisted Planning   │  LLM-enhanced plan proposals (optional)
Stage  8  │  Plan Validation        │  Safety-gate schema & feasibility checks
Stage  9  │  ML Execution           │  Split → Preprocess → FE → FS → Train → Eval
Stage 10  │  Execution Report       │  Consolidate all artifacts into immutable report
Stage 11  │  AI Model Critique      │  LLM-powered expert review of results
Stage 12  │  Plan Optimization      │  Map recommendations → optimize plan → retrain
Stage 13  │  Champion Governance    │  Baseline vs. Retrained comparison & selection
Stage 14  │  Executive Report       │  Final consolidated report with deployment guidance
```

---

## 📁 Project Structure

```
deploy_ai/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── core/                      # Config, logging
│   │   ├── api/                       # 6 REST routers
│   │   ├── upload/                    # File ingestion (CSV, Excel)
│   │   ├── analysis/                  # Dataset profiling & statistics
│   │   ├── dataset_intelligence/      # DatasetContext builder
│   │   ├── problem_definition/        # Problem type resolver
│   │   ├── ml_request/                # User ML request schemas
│   │   ├── compute_capabilities/      # Hardware profiling
│   │   ├── hardware/                  # Low-level hardware detection
│   │   ├── preprocessing_engine/      # Data cleaning pipeline
│   │   ├── ml_plan/                   # Plan generation & validation
│   │   ├── ai_planning/              # AI-assisted planning (LLM)
│   │   ├── ai_providers/             # Multi-LLM provider management
│   │   ├── ai_engine/                # Legacy AI recommendation engine
│   │   ├── ml_execution/             # Training pipeline orchestrator
│   │   ├── ai_model_critic/          # Post-training AI review
│   │   ├── ai_model_optimizer/       # Critique → plan optimization
│   │   ├── model_governance/         # Champion model selection
│   │   ├── reporting/                # Executive report generation
│   │   ├── automl/                   # AutoML utilities (reserved)
│   │   ├── workspace/               # Project workspace management
│   │   ├── exports/                  # Model serialization (pkl, ONNX, Keras)
│   │   ├── explainability/           # Interpretability (reserved)
│   │   ├── optimization/             # Deployment optimization (reserved)
│   │   ├── detector/                 # Anomaly detection (reserved)
│   │   └── utils/                    # Shared utilities
│   └── tests/                        # 88+ test files, 2090+ test cases
├── datasets/
│   ├── raw/                           # Original uploaded datasets
│   ├── processed/                     # Preprocessed datasets
│   └── samples/                       # Sample datasets for testing
├── models/
│   ├── trained/                       # Serialized trained models
│   ├── checkpoints/                   # Training checkpoints
│   └── experiments/                   # Experiment tracking artifacts
├── reports/
│   ├── evaluation/                    # Evaluation metric reports
│   └── feature_importance/            # Feature importance analysis
├── docs/
│   ├── api_docs/                      # API documentation
│   ├── architecture/                  # Architecture documentation
│   └── deployment/                    # Deployment guides
├── docker/                            # Docker configuration files
├── scripts/                           # Utility scripts
├── docker-compose.yml                 # Container orchestration
├── requirements.txt                   # Python dependencies
├── tester.py                          # Quick validation script
└── LICENSE                            # MIT License
```

---

## 🛠️ Prerequisites

- **Python 3.10+**
- **Groq API Key** — set in `.env` file as `GROQ_API_KEY=your_key_here`
- **pip** or **uv** package manager

---

## 🚀 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/krishtewatia/deploy_ai.git
   cd deploy_ai
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv

   # Linux/macOS
   source venv/bin/activate

   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   ```bash
   # Create .env file in project root
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

---

## ▶️ Running the Backend Server

Launch the FastAPI backend API server (runs on `http://127.0.0.1:8000` by default):

```bash
uvicorn backend.app.main:app --reload
```

**Available endpoints**:
- `GET  /health` — Health check
- `POST /upload/upload` — Upload dataset file
- `POST /analysis/analyze` — Analyze dataset
- `POST /preprocessing/...` — Preprocessing operations
- `POST /training/...` — Trigger model training
- `GET  /models/...` — Retrieve trained models
- `GET  /reports/...` — Retrieve reports

Interactive API docs available at `http://127.0.0.1:8000/docs` (Swagger UI).

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

This builds and starts the backend service on port `8000` using the `docker/Dockerfile.backend` configuration.

---

## 🧪 Testing

The project includes **88+ test files** with **2,090+ unit and integration tests** covering every module.

```bash
# Run the entire test suite
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage report
python -m pytest --cov=backend

# Run specific module tests
python -m pytest backend/tests/test_execution_orchestrator.py -v
python -m pytest backend/tests/test_evaluation_engine.py -v
python -m pytest backend/tests/test_ai_model_critic_service.py -v
python -m pytest backend/tests/test_champion_comparator.py -v
python -m pytest backend/tests/test_report_generator.py -v
```

**Key test areas**:
| Test File | Module Covered |
|-----------|---------------|
| `test_execution_orchestrator.py` | Full ML pipeline execution |
| `test_ml_planning_orchestrator.py` | Planning orchestration |
| `test_baseline_ml_planner.py` | Deterministic plan generation |
| `test_ml_plan_validator.py` | Plan validation rules |
| `test_ai_decision_service.py` | AI-assisted planning |
| `test_evaluation_engine.py` | Metric computation & CV |
| `test_ai_model_critic_service.py` | AI critique pipeline |
| `test_optimizer_service.py` | Plan optimization |
| `test_retraining_engine.py` | Retrained model execution |
| `test_champion_comparator.py` | Governance comparison |
| `test_report_generator.py` | Executive report assembly |
| `test_training_results.py` | Training results dashboard |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Async web framework |
| `uvicorn[standard]` | ASGI server |
| `pandas` | Data manipulation |
| `scikit-learn` | ML models, preprocessing, metrics |
| `pydantic` / `pydantic-settings` | Schema validation & config |
| `python-multipart` | File upload support |
| `openpyxl` | Excel file parsing |
| `psutil` | Hardware detection |
| `keyring` | Secure API key storage |
| `groq` | Groq LLM API client |
| `httpx` | Async HTTP client (testing) |
| `pytest` / `pytest-cov` | Testing framework |

**Optional dependencies** (commented in `requirements.txt`):
- `SQLAlchemy` + `psycopg2-binary` — PostgreSQL persistence
- `Celery` + `redis` — Background task queue
- `tensorflow` — Deep learning model support
- `xgboost` — XGBoost model support
- `gcsfs` — Google Cloud Storage integration

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 upanya
