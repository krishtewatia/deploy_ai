# DeployAI — AI-Powered MLOps & Auto-ML Platform

DeployAI is an end-to-end MLOps platform featuring automated data preprocessing, ML model training, evaluation, AI critique reviews, model governance auditing, and experiment tracking.

The application comprises a **FastAPI backend** that hosts model evaluation and analysis algorithms, and a native **PySide6 desktop frontend dashboard** that guides users through importing datasets, configuring ML plans, tracking executions, and reviewing dashboard reports.

---

## 🛠️ Prerequisites

Make sure you have python 3.10+ and standard tools installed.

---

## 🚀 Installation & Setup

1. **Clone or Navigate to the Directory**:
   ```powershell
   cd c:\Users\hp\Downloads\deploy_ai
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended)**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   pip install PySide6
   ```

---

## 🖥️ Running the Application

DeployAI runs as a decoupled backend API and a desktop GUI frontend.

### 1. Start the FastAPI Backend
To launch the backend API server (runs on `http://127.0.0.1:8000` by default):
```powershell
uvicorn backend.app.main:app --reload
```

### 2. Start the PySide6 Frontend GUI
To run the PySide6 desktop interface, run the helper runner script from the root:
```powershell
python run_app.py
```

---

## 🧪 Running tests

You can run the full test suite (over 2,090 unit and integration tests) using pytest:

```powershell
# Run the entire test suite
python -m pytest

# Run only training results dashboard tests
python -m pytest tests/test_training_results.py -v
```
