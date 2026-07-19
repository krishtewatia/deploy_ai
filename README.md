# DeployAI — AI-Powered AutoML & Model Training Platform

DeployAI is a terminal-based, end-to-end Auto-ML platform that automates the complete machine learning lifecycle. It operates purely as an in-memory Python backend with zero web-server or FastAPI dependencies. It is driven by an interactive command-line interface (CLI) and leverages local LLMs (via Ollama) or cloud providers (Groq/OpenAI) for intelligent, self-optimizing pipelines.

---

## 📐 Model Training Pipeline Architecture

DeployAI's execution engine orchestrates a strictly sequenced **11-Stage AutoML Pipeline** over in-memory domain models. Each stage is handled by dedicated Python service modules:

```
[1. Upload] ──> [2. Validation] ──> [3. Dataset Intelligence] ──> [4. Problem Definition] 
                                                                           │
┌──────────────────────────────────────────────────────────────────────────┘
│
▼
[5. AI Configuration] ──> [6. Planning] ──> [7. Execution] ──> [8. Champion Selection]
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
│
▼
[9. AI Explanation] ──> [10. PDF Generation] ──> [11. Export]
```

### The 11 Core Stages:

1. **Upload (`upload/`)**
   - Ingests single or multiple local dataset files (`.csv`, `.xlsx`, `.xls`).
   - **Dataset Merging**: If multiple comma-separated files are provided, the engine verifies that schemas match (same context) and concatenates them automatically.
   - **Inconsistency Correction**: Scans the loaded DataFrame for duplicate identical columns (same content/data values) and drops them.
2. **Validation (`upload/validator.py`)**
   - Validates that the loaded dataset is valid, checks column datatypes, detects missing entries, and ensures target label columns are valid.
3. **Dataset Intelligence (`dataset_intelligence/`)**
   - Constructs a structured profile (`DatasetContext`) cataloging feature types, statistics (mean, min, max, standard deviation), and value distributions.
4. **Problem Definition (`problem_definition/`)**
   - Automatically determines if the target column calls for a binary classification, multi-class classification, or regression task based on target column values and cardinality.
5. **AI Configuration (`ai_providers/`)**
   - Auto-detects local host compute capabilities (CPU cores, memory size) and identifies available AI model providers.
6. **Planning (`ml_plan/` & `ai_planning/`)**
   - Generates the preprocessing, feature engineering, and model training plan.
   - **Deterministic Mode**: Automatically selects standard preprocessors (Imputer, Scaler, Encoder) and basic model candidates.
   - **AI-Assisted Mode**: Leverages an LLM provider to optimize preprocessing choices, model candidates, and search parameters.
7. **Execution (`ml_execution/`)**
   - Performs a stratified train-test split, fits pipeline preprocessors, runs hyperparameter search (GridSearch/RandomSearch), and trains candidate classifiers or regressors.
   - **Tuning Injection**: Automatically injects hyperparameter search spaces for baseline models (e.g. Random Forest, Gradient Boosting, SVM, Logistic Regression, Decision Tree, KNN) to prevent underfitting and overfitting.
   - **Multi-class Scoring Optimization**: Dynamically maps classification scorers (e.g., `f1`, `precision`, `recall`, `roc_auc`) to their macro/OVR counterparts (e.g., `f1_macro`, `roc_auc_ovr`) on multi-class tasks, resolving scikit-learn search failures.
8. **Champion Selection (`model_governance/`)**
   - Executes deterministic model governance comparing candidates to select the champion model based on primary metric scores (F1, Accuracy, MSE).
9. **AI Explanation (`ai_model_critic/`)**
   - Leverages the LLM provider to review the final execution metrics and write a natural language critique detailing model strengths, weaknesses, and potential deployment risks.
10. **PDF Generation (`reporting/`)**
    - Compiles dataset profiles, candidate benchmark tables, governance results, and AI critiques into a professional multi-page PDF binary file (`reports/exec_report_*.pdf`).
11. **Export (`exports/`)**
    - Serializes the trained champion model binary to disk in the user's chosen format (`models/model_002_*.pkl`, `*.joblib`, `*.onnx`) and outputs the metadata path.

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.com/) running locally if you want to use local AI models.

### Installation
1. Clone the project repository and navigate to its root directory.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 1. Interactive CLI Wizard (Recommended)
To run the setup wizard and train a model interactively, run the CLI without arguments:
```bash
python cli.py
```
The interactive wizard will guide you through:
1. **Dataset Selection**: 
   - Scans the `datasets/` folder and lists available datasets.
   - You can choose a file from the list or **specify a custom path**.
   - You can pass **multiple custom paths separated by commas** (e.g., `datasets/iris.csv,datasets/Iris.csv`). The engine will automatically verify schemas and merge them.
2. **Target column input**: Ask for target column, or press Enter to auto-detect.
3. **Model Export Format Selection**:
   - Select your preferred model serialization format:
     1. Pickle (`.pkl`) - Standard Python serialization
     2. Joblib (`.joblib`) - Optimized for large numpy arrays
     3. ONNX (`.onnx`) - Open Neural Network Exchange format (falls back to Pickle if `skl2onnx` is not installed)
4. **AI Provider Select & Pull Menu**:
   - **Option 1**: Detect and select an existing local Ollama model.
   - **Option 2**: Download (pull) a new local model (e.g. `llama3.1`, `phi3`) dynamically via the command line.
   - **Option 3**: Configure a cloud provider (Groq / OpenAI) by entering your API key.
   - **Option 4**: Skip AI configuration and run in deterministic baseline mode.

### 2. Headless execution
To run the CLI directly with arguments:
```bash
python cli.py --dataset datasets/iris.csv,datasets/Iris.csv --target target --mode deterministic --format joblib
```

Arguments:
- `--dataset` - Path to local dataset file(s). Support multiple comma-separated paths.
- `--target` - Column name to predict (optional, will be inferred if omitted).
- `--mode` - `deterministic` (default) or `ai_assisted`.
- `--format` - `pickle` (default), `joblib`, or `onnx`.
- `--verbose` or `-v` - Enable verbose debug print logging.

---

## 🧪 Testing & Verification

### Unit Tests
Execute unit tests for the pipeline engine, CLI parser, and model serializers:
```bash
pytest backend/tests/test_deploy_ai_engine.py backend/tests/test_engine_cli.py
```

### Functional Integration Suite
To run end-to-end integration tests over all datasets (`iris.csv`, `Iris.csv`, `sample.csv`):
```bash
python tester.py
```
Upon completion, test logs will print to the console and generated JSON/PDF reports will be output to the `reports/` folder.
