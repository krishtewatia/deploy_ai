User Uploads Dataset
        │
        ▼
Upload Service
        │
        ▼
Dataset Validation
        │
        ▼
Dataset Intelligence
        │
        ▼
Problem Definition
        │
        ▼
AI Configuration
        │
        ▼
ML Plan Generation
        │
        ▼
ML Execution
        │
        ├── Split
        ├── Preprocessing
        ├── Feature Engineering
        ├── Feature Selection
        ├── Model Training
        ├── Hyperparameter Optimization
        └── Evaluation
        │
        ▼
Champion Model Selection
        │
        ▼
AI Explanation
        │
        ▼
PDF Report Generation
        │
        ▼
Export

| Workflow Step        | Current Package              |
| -------------------- | ---------------------------- |
| Upload Service       | `upload/`                    |
| Dataset Validation   | `upload/validator.py`        |
| Dataset Intelligence | `dataset_intelligence/`      |
| Problem Definition   | `problem_definition/`        |
| AI Configuration     | `ai_providers/`              |
| ML Plan Generation   | `ml_plan/` + `ai_planning/`  |
| ML Execution         | `ml_execution/`              |
| Champion Selection   | `model_governance/`          |
| AI Explanation       | *(To build)*                 |
| PDF Report           | `reporting/` (or new module) |
| Export               | `exports/`                   |
    