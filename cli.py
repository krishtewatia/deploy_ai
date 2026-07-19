#!/usr/bin/env python3
"""DeployAI Command Line Interface (CLI).

Enables running full end-to-end machine learning model training pipelines
directly from the terminal. Operates purely in-memory over Python backend packages
and supports interactive dataset selection, target detection, and AI provider/model management.
"""

from __future__ import annotations

import argparse
import sys
import os
import glob
import uuid
import logging
import subprocess
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from backend.app.pipeline.engine import DeployAIEngine
from backend.app.pipeline.schemas import PipelineStage, PipelineStatus
from backend.app.ml_plan.orchestrator import PlanningMode
from backend.app.ml_request.schemas import UserMLRequest, ProblemTypePreference
from backend.app.ai_providers.factory import (
    OllamaAIProvider, 
    OpenAICompatibleAIProvider, 
    detect_ai_provider
)
from backend.app.ai_providers.schemas import (
    OllamaProviderConfig, 
    OpenAICompatibleProviderConfig
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DeployAI CLI - Train ML models directly from your terminal."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the local input dataset file (.csv, .xlsx, .xls), support multiple comma-separated files.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name to predict (optional, will be inferred if omitted)",
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "ai_assisted"],
        default=None,
        help="Planning mode (deterministic or ai_assisted)",
    )
    parser.add_argument(
        "--format",
        choices=["pickle", "joblib", "onnx"],
        default="pickle",
        help="Desired export format for the champion model (default: pickle)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Force interactive setup wizard",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging output",
    )
    return parser


def run_interactive_wizard() -> tuple[str, Optional[str], PlanningMode, Optional[any], str, UserMLRequest]:
    """Runs interactive command line questionnaire to guide the user."""
    print("=" * 65)
    print("      DeployAI Model Training Interactive Setup Wizard")
    print("=" * 65)

    # 1. Dataset Selection
    dataset_dir = Path("datasets")
    discovered_files = []
    if dataset_dir.exists():
        discovered_files = sorted(
            list(dataset_dir.glob("*.csv")) +
            list(dataset_dir.glob("*.xlsx")) +
            list(dataset_dir.glob("*.xls")) +
            list(dataset_dir.glob("samples/*.csv")) +
            list(dataset_dir.glob("samples/*.xlsx"))
        )

    if discovered_files:
        print("\nDiscovered datasets in workspace:")
        for idx, file in enumerate(discovered_files, 1):
            print(f"  {idx}) {file}")
        print(f"  {len(discovered_files) + 1}) Specify custom file path(s) (separate multiple files with commas)...")
        
        while True:
            try:
                ds_choice = input(f"Choose a dataset (1-{len(discovered_files) + 1}): ").strip()
                choice_idx = int(ds_choice)
                if 1 <= choice_idx <= len(discovered_files):
                    dataset_path = str(discovered_files[choice_idx - 1])
                    break
                elif choice_idx == len(discovered_files) + 1:
                    dataset_path = input("Enter custom dataset file path(s) (separate multiple files with commas to merge): ").strip()
                    break
            except ValueError:
                pass
            print("Invalid selection. Try again.")
    else:
        dataset_path = input("\nEnter custom dataset file path(s) (separate multiple files with commas to merge): ").strip()

    # Helper function to check existence of multiple paths
    def verify_paths(paths_str: str) -> bool:
        parts = [p.strip() for p in paths_str.split(",")]
        return all(Path(p).exists() for p in parts)

    # Verify dataset exists
    while not verify_paths(dataset_path):
        print(f"One or more files not found in: {dataset_path}")
        dataset_path = input("Enter custom dataset file path(s) (separate multiple files with commas to merge): ").strip()

    # 2. Preview Discovered Columns
    try:
        parts = [p.strip() for p in dataset_path.split(",")]
        first_path = Path(parts[0]).resolve()
        suffix = first_path.suffix.lower()
        if suffix == ".csv":
            df_temp = pd.read_csv(first_path, nrows=1)
        elif suffix in [".xlsx", ".xls"]:
            df_temp = pd.read_excel(first_path, nrows=1)
        else:
            df_temp = None

        if df_temp is not None:
            cols = list(df_temp.columns)
            print("\nDiscovered columns in dataset:")
            for c_idx, col_name in enumerate(cols, 1):
                print(f"  {c_idx:3d}) {col_name}")
        else:
            cols = []
    except Exception as e:
        print(f"Warning: Could not preview columns: {e}")
        cols = []

    # 3. Target column
    target_input = input("\nEnter target column name or index number (press Enter to auto-detect): ").strip()
    if not target_input:
        target_column = None
    elif target_input.isdigit():
        idx_val = int(target_input)
        if cols and 1 <= idx_val <= len(cols):
            target_column = cols[idx_val - 1]
            print(f"-> Selected target column: '{target_column}'")
        else:
            target_column = target_input
    else:
        target_column = target_input

    # 4. User Request / AI Goal Context
    print("\n" + "-" * 60)
    print(" User Goal & AI Context Setup")
    print("-" * 60)
    user_goal = input("Describe your ML goal (e.g., 'Classify news as fake/true', 'Detect spam'): ").strip()
    if not user_goal:
        user_goal = "Optimize model performance and accuracy"

    additional_context = input("Enter any additional instructions or context (press Enter to skip): ").strip()
    if not additional_context:
        additional_context = None

    # Build Pydantic UserMLRequest
    user_request = UserMLRequest(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        goal=user_goal,
        target_column=target_column,
        additional_context=additional_context,
    )

    # 5. Model Export Format Selection
    print("\n" + "-" * 60)
    print(" Model Export Format Selection")
    print("-" * 60)
    print("Choose the format in which the final model should be serialized:")
    print("  1) Pickle (.pkl) - Default Python serialization")
    print("  2) Joblib (.joblib) - Optimized for large numpy arrays")
    print("  3) ONNX (.onnx) - Open Neural Network Exchange format")
    print("-" * 60)
    
    export_format = "PICKLE"
    while True:
        fmt_choice = input("Select format choice (1-3) [Default 1]: ").strip()
        if not fmt_choice or fmt_choice == "1":
            export_format = "PICKLE"
            break
        elif fmt_choice == "2":
            export_format = "JOBLIB"
            break
        elif fmt_choice == "3":
            export_format = "ONNX"
            break
        print("Invalid selection. Please choose 1, 2, or 3.")

    # 6. AI Provider Selection
    print("\n" + "-" * 60)
    print(" AI Provider & Model Configuration")
    print("-" * 60)
    print("DeployAI leverages AI providers to generate optimized preprocessing plans,")
    print("validate configurations, critique candidate models, and output explanations.")
    print("Choose your preferred setup:")
    print("  1) Use an existing local Ollama model")
    print("  2) Download/Pull a new Ollama model")
    print("  3) Configure a Cloud API Provider (Groq / OpenAI)")
    print("  4) Run in pure deterministic baseline mode (No AI)")
    print("-" * 60)

    ai_provider = None
    planning_mode = PlanningMode.DETERMINISTIC

    while True:
        choice = input("Select AI setup option (1-4): ").strip()
        if choice == "1":
            # List local models
            try:
                res = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
                if res.status_code == 200:
                    models = res.json().get("models", [])
                    if not models:
                        print("Ollama is running, but no local models are installed.")
                        print("Use option 2 to download a model.")
                        continue
                    print("\nInstalled local Ollama models:")
                    for idx, m in enumerate(models, 1):
                        print(f"  {idx}) {m['name']}")
                    while True:
                        try:
                            m_choice = input(f"Select model (1-{len(models)}): ").strip()
                            m_idx = int(m_choice)
                            if 1 <= m_idx <= len(models):
                                model_name = models[m_idx - 1]["name"]
                                break
                        except ValueError:
                            pass
                        print("Invalid selection.")
                    
                    config = OllamaProviderConfig(
                        config_id="cli-ollama",
                        display_name="CLI Ollama",
                        model_name=model_name,
                        base_url="http://localhost:11434",
                        request_timeout_seconds=600.0
                    )
                    ai_provider = OllamaAIProvider(config)
                    planning_mode = PlanningMode.AI_ASSISTED
                    break
            except Exception as e:
                print("Could not connect to local Ollama service on http://localhost:11434")
                print("Make sure the Ollama application is running and try again.")
                continue

        elif choice == "2":
            # Pull model
            model_name = input("Enter Ollama model name to pull (e.g., llama3.1, phi3, gemma2, qwen2.5): ").strip()
            if not model_name:
                print("Model name cannot be empty.")
                continue
            
            print(f"\nPulling Ollama model '{model_name}'... This may take several minutes.")
            try:
                # Runs command interactively showing standard download progress
                subprocess.run(["ollama", "pull", model_name], check=True)
                config = OllamaProviderConfig(
                    config_id="cli-ollama-pulled",
                    display_name="CLI Ollama Pulled",
                    model_name=model_name,
                    base_url="http://localhost:11434",
                    request_timeout_seconds=600.0
                )
                ai_provider = OllamaAIProvider(config)
                planning_mode = PlanningMode.AI_ASSISTED
                break
            except FileNotFoundError:
                print("Error: 'ollama' command line tool was not found on your system PATH.")
                print("Make sure Ollama is installed from https://ollama.com")
            except subprocess.CalledProcessError as e:
                print(f"Error pulling model: {e}")
            continue

        elif choice == "3":
            print("\nSelect Cloud Provider:")
            print("  1) Groq (Default: llama-3.1-8b-instant)")
            print("  2) OpenAI (Default: gpt-4o-mini)")
            while True:
                prov_choice = input("Choice (1-2): ").strip()
                if prov_choice in ["1", "2"]:
                    break
                print("Invalid input.")

            api_key = input("Enter your API Key: ").strip()
            if not api_key:
                print("An API key is required for the selected cloud provider.")
                continue
            if prov_choice == "1":
                os.environ["GROQ_API_KEY"] = api_key
                config = OpenAICompatibleProviderConfig(
                    config_id="cli-groq",
                    display_name="CLI Groq",
                    model_name="llama-3.1-8b-instant",
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key
                )
            else:
                os.environ["OPENAI_API_KEY"] = api_key
                config = OpenAICompatibleProviderConfig(
                    config_id="cli-openai",
                    display_name="CLI OpenAI",
                    model_name="gpt-4o-mini",
                    base_url="https://api.openai.com/v1",
                    api_key=api_key
                )
            ai_provider = OpenAICompatibleAIProvider(config)
            planning_mode = PlanningMode.AI_ASSISTED
            break

        elif choice == "4":
            planning_mode = PlanningMode.DETERMINISTIC
            ai_provider = None
            break
        else:
            print("Invalid option selected. Please select between 1 and 4.")

    return dataset_path, target_column, planning_mode, ai_provider, export_format, user_request


def main(args_list: Optional[list[str]] = None) -> int:
    """Main CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(args_list)

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine if we run in interactive wizard mode
    interactive_mode = args.interactive or (args.dataset is None)

    if interactive_mode:
        dataset_path, target_column, mode, ai_provider, export_format, user_request = run_interactive_wizard()
    else:
        dataset_path = args.dataset
        target_column = args.target
        mode_val = args.mode or "deterministic"
        mode = PlanningMode.AI_ASSISTED if mode_val == "ai_assisted" else PlanningMode.DETERMINISTIC
        ai_provider = detect_ai_provider() if mode == PlanningMode.AI_ASSISTED else None
        export_format = args.format.upper() if args.format else "PICKLE"
        user_request = UserMLRequest(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            goal="Optimize model performance",
            target_column=target_column,
        )

    print("\n" + "=" * 65)
    print(" DeployAI Desktop Model Training Platform — CLI Execution")
    print("=" * 65)
    print(f"Dataset:       {dataset_path}")
    print(f"Target Column: {target_column or 'AUTO-DETECT'}")
    print(f"Planning Mode: {mode.value}")
    print(f"Export Format: {export_format}")
    if ai_provider:
        print(f"AI Provider:   {ai_provider.config.display_name} ({ai_provider.config.model_name})")
    print("-" * 65)

    engine = DeployAIEngine()

    def progress_callback(stage: PipelineStage, progress: float) -> None:
        percent = int(progress * 100)
        print(f"[{percent:3d}%] Running Stage: {stage.value}")

    context = engine.run_pipeline(
        dataset_path=dataset_path,
        target_column=target_column,
        user_request=user_request,
        planning_mode=mode,
        ai_provider=ai_provider,
        export_format=export_format,
        progress_callback=progress_callback,
    )

    print("-" * 65)
    if context.status == PipelineStatus.COMPLETED:
        print(" SUCCESS: Pipeline completed successfully!")
        print(f"Pipeline ID: {context.pipeline_id}")
        
        if context.champion_decision and context.execution_report:
            champ = context.execution_report.champion_summary
            print(f"Champion Model:     {champ.model_family}")
            print(f"Primary Metric ({champ.primary_metric}): {champ.primary_metric_value:.4f}")
            print(f"Selection Reason:    {context.champion_decision.decision_reason}")
            
        if context.pdf_report_path:
            print(f"PDF Report Saved:   {Path(context.pdf_report_path).resolve()}")
        if context.export_result:
            print(f"Model Artifact:     {context.export_result.artifact_path}")
            
        print("=" * 65)
        return 0
    else:
        print(" FAILED: Pipeline execution failed!")
        print(f"Error: {context.error_message}")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    sys.exit(main())
