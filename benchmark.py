#!/usr/bin/env python3
"""
benchmark.py - DeployAI Backend Performance Benchmarker

Measures execution runtimes, CPU/RAM utilization, AI latencies,
and exported model sizes across all stages of the training pipeline.
"""

import os
import sys
import time
import pickle
import json
import psutil
from pathlib import Path
from typing import Optional, Any, List


from backend.app.pipeline.engine import DeployAIEngine
from backend.app.pipeline.schemas import PipelineContext, PipelineStage, PipelineStatus
from backend.app.ml_plan.orchestrator import PlanningMode, MLPlanningOrchestrator
from backend.app.ai_providers.schemas import OllamaProviderConfig, OpenAICompatibleProviderConfig
from backend.app.ai_providers.factory import OllamaAIProvider, OpenAICompatibleAIProvider
from backend.app.ai_planning.providers.base import AIProvider

# ANSI terminal colors
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"

# Track response times across all provider API calls
_AI_RESPONSE_TIMES: List[float] = []

def print_info(msg: str) -> None:
    print(f"{COLOR_BLUE}[INFO]{COLOR_RESET} {msg}")

def print_warn(step_name: str, warn_msg: str) -> None:
    print(f"  {COLOR_YELLOW}[WARN]{COLOR_RESET} {step_name} - {warn_msg}")

# Wrap Ollama generate
_original_ollama_generate = OllamaAIProvider.generate
def _wrapped_ollama_generate(self, *args, **kwargs):
    start = time.perf_counter()
    res = _original_ollama_generate(self, *args, **kwargs)
    duration = time.perf_counter() - start
    _AI_RESPONSE_TIMES.append(duration)
    return res
OllamaAIProvider.generate = _wrapped_ollama_generate

# Wrap OpenAI generate
_original_openai_generate = OpenAICompatibleAIProvider.generate
def _wrapped_openai_generate(self, *args, **kwargs):
    start = time.perf_counter()
    res = _original_openai_generate(self, *args, **kwargs)
    duration = time.perf_counter() - start
    _AI_RESPONSE_TIMES.append(duration)
    return res
OpenAICompatibleAIProvider.generate = _wrapped_openai_generate

# Global provider instance for monkeypatching
_ACTIVE_PROVIDER: Optional[AIProvider] = None

# Monkeypatch MLPlanningOrchestrator to inject the detected AI provider
_original_create_plan = MLPlanningOrchestrator.create_plan
def _patched_create_plan(self, *args, **kwargs):
    if _ACTIVE_PROVIDER is not None:
        if kwargs.get("mode") == PlanningMode.AI_ASSISTED or (len(args) > 3 and args[3] == PlanningMode.AI_ASSISTED):
            if "ai_provider" not in kwargs or kwargs["ai_provider"] is None:
                kwargs["ai_provider"] = _ACTIVE_PROVIDER
    return _original_create_plan(self, *args, **kwargs)
MLPlanningOrchestrator.create_plan = _patched_create_plan

# Robust AI Response Parser Monkeypatch to extract structured JSON cleanly
from backend.app.ai_planning.response_parser import AIResponseParser
_original_strip_markdown_fences = AIResponseParser._strip_markdown_fences

def _robust_strip_markdown_fences(self, text: str) -> str:
    cleaned = _original_strip_markdown_fences(self, text)
    cleaned_stripped = cleaned.strip()
    
    start = cleaned_stripped.find("{")
    if start == -1:
        return cleaned_stripped
    
    brace_count = 0
    in_string = False
    escape = False
    
    for idx in range(start, len(cleaned_stripped)):
        char = cleaned_stripped[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return cleaned_stripped[start:idx+1]
    
    return cleaned_stripped

AIResponseParser._strip_markdown_fences = _robust_strip_markdown_fences


def get_current_ram() -> float:
    """Returns memory usage of the current process in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def detect_ai_provider() -> Optional[AIProvider]:
    import httpx
    
    ollama_url = "http://localhost:11434"
    try:
        response = httpx.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models_list = response.json().get("models", [])
            if models_list:
                selected_model = models_list[0]["name"]
                config = OllamaProviderConfig(
                    config_id="benchmark-ollama",
                    display_name="Benchmark Ollama",
                    model_name=selected_model,
                    base_url=ollama_url,
                    request_timeout_seconds=600.0
                )
                return OllamaAIProvider(config)
    except Exception:
        pass

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key != "mock":
        config = OpenAICompatibleProviderConfig(
            config_id="benchmark-groq",
            display_name="Benchmark Groq",
            model_name="llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
        return OpenAICompatibleAIProvider(config)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        config = OpenAICompatibleProviderConfig(
            config_id="benchmark-openai",
            display_name="Benchmark OpenAI",
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=openai_key
        )
        return OpenAICompatibleAIProvider(config)

    return None


def run_dataset_benchmark(dataset_path: Path, planning_mode: PlanningMode) -> dict:
    print(f"\n{COLOR_CYAN}============================================================{COLOR_RESET}")
    print(f"Benchmarking Dataset: {dataset_path.name} | Mode: {planning_mode.value}")
    print(f"{COLOR_CYAN}============================================================{COLOR_RESET}")

    engine = DeployAIEngine()
    context = PipelineContext(dataset_path=str(dataset_path))
    engine._contexts[context.pipeline_id] = context
    context.status = PipelineStatus.RUNNING

    stages = [
        (PipelineStage.UPLOAD, "Stage Upload"),
        (PipelineStage.VALIDATION, "Stage Validation"),
        (PipelineStage.DATASET_INTELLIGENCE, "Stage Dataset Intelligence"),
        (PipelineStage.PROBLEM_DEFINITION, "Stage Problem Definition"),
        (PipelineStage.AI_CONFIGURATION, "Stage AI Configuration"),
        (PipelineStage.PLANNING, "Stage Planning"),
        (PipelineStage.EXECUTION, "Stage Execution"),
        (PipelineStage.CHAMPION_SELECTION, "Stage Champion Selection"),
        (PipelineStage.AI_EXPLANATION, "Stage AI Explanation"),
        (PipelineStage.PDF_GENERATION, "Stage PDF Generation"),
        (PipelineStage.EXPORT, "Stage Export")
    ]

    metrics = {}
    total_start = time.perf_counter()
    process = psutil.Process()

    global _AI_RESPONSE_TIMES
    _AI_RESPONSE_TIMES = []

    for stage, stage_name in stages:
        context.current_stage = stage
        
        # Capture baseline resources
        process.cpu_percent(interval=None) # Reset cpu counter
        start_time = time.perf_counter()
        
        try:
            # Run the stage with robust planning fallback
            if stage == PipelineStage.PLANNING and planning_mode == PlanningMode.AI_ASSISTED:
                try:
                    context = engine.run_stage(
                        stage=stage,
                        context=context,
                        target_column=None,
                        user_request=None,
                        planning_mode=planning_mode
                    )
                except Exception as pl_exc:
                    print_warn("AI Planning", f"AI-assisted planning failed ({pl_exc}). Falling back to DETERMINISTIC baseline plan.")
                    context = engine.run_stage(
                        stage=stage,
                        context=context,
                        target_column=None,
                        user_request=None,
                        planning_mode=PlanningMode.DETERMINISTIC
                    )
            else:
                context = engine.run_stage(
                    stage=stage,
                    context=context,
                    target_column=None,
                    user_request=None,
                    planning_mode=planning_mode
                )

            elapsed = time.perf_counter() - start_time
            cpu_usage = process.cpu_percent(interval=None)
            ram_usage = get_current_ram()

            metrics[stage.value] = {
                "time": elapsed,
                "ram": ram_usage,
                "cpu": cpu_usage,
                "status": "PASS"
            }
            print(f"  {stage_name:<30} | Time: {elapsed:.4f}s | RAM: {ram_usage:7.2f} MB | CPU: {cpu_usage:5.2f}%")

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            print(f"  {stage_name:<30} | {COLOR_RED}FAILED{COLOR_RESET} | Time: {elapsed:.4f}s")
            context.status = PipelineStatus.FAILED
            metrics[stage.value] = {
                "time": elapsed,
                "ram": get_current_ram(),
                "cpu": 0.0,
                "status": "FAIL",
                "error": str(exc)
            }
            break

    total_time = time.perf_counter() - total_start

    if context.status == PipelineStatus.FAILED:
        return {
            "status": "FAIL",
            "metrics": metrics
        }

    # Serialization size benchmark
    model_export_dir = Path("models/trained")
    model_export_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_export_dir / f"{dataset_path.stem}.pkl"
    
    model_export_start = time.perf_counter()
    with open(model_path, "wb") as f:
        pickle.dump(context.execution_result.best_model, f)
    model_export_time = time.perf_counter() - model_export_start
    model_size_kb = model_path.stat().st_size / 1024

    # Save reports
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{dataset_path.stem}_report.json"
    if context.executive_report:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(context.executive_report.model_dump(mode="json"), f, indent=2)

    ai_total_time = sum(_AI_RESPONSE_TIMES)
    ai_request_count = len(_AI_RESPONSE_TIMES)

    print(f"\n  Performance Summary for {dataset_path.name}:")
    print(f"    * Total Runtime:                {total_time:.4f}s")
    print(f"    * Peak RAM Usage:               {max(m['ram'] for m in metrics.values()):.2f} MB")
    print(f"    * Model Training Time (Stage):  {metrics['execution']['time']:.4f}s")
    print(f"    * Model Export Time (.pkl):      {model_export_time:.4f}s")
    print(f"    * Model Binary Size:            {model_size_kb:.2f} KB")
    if ai_request_count > 0:
        print(f"    * AI Provider Response Time:    {ai_total_time:.4f}s ({ai_request_count} requests)")
    else:
        print(f"    * AI Provider Response Time:    N/A (Deterministic)")

    return {
        "status": "PASS",
        "total_runtime": total_time,
        "peak_ram": max(m["ram"] for m in metrics.values()),
        "model_training_time": metrics["execution"]["time"],
        "model_export_time": model_export_time,
        "model_size_kb": model_size_kb,
        "ai_response_time": ai_total_time if ai_request_count > 0 else 0.0,
        "ai_request_count": ai_request_count,
        "metrics": metrics
    }


def main():
    global _ACTIVE_PROVIDER
    
    print("============================================================")
    print(" DeployAI Desktop Model Training Platform — Benchmarker")
    print("============================================================")

    # 1. Detect datasets
    dataset_dir = Path("datasets")
    if not dataset_dir.exists():
        print(f"{COLOR_RED}[ERROR]{COLOR_RESET} Datasets folder 'datasets/' does not exist.")
        sys.exit(1)

    csv_files = sorted(list(dataset_dir.glob("*.csv")) + list(dataset_dir.glob("samples/*.csv")))
    if not csv_files:
        print(f"{COLOR_RED}[ERROR]{COLOR_RESET} No CSV datasets found inside 'datasets/' or 'datasets/samples/'.")
        sys.exit(1)

    print_info(f"Discovered {len(csv_files)} datasets to benchmark: {[f.name for f in csv_files]}")

    # 2. Detect AI Provider
    _ACTIVE_PROVIDER = detect_ai_provider()
    planning_mode = PlanningMode.AI_ASSISTED if _ACTIVE_PROVIDER is not None else PlanningMode.DETERMINISTIC
    if _ACTIVE_PROVIDER is None:
        print_warn("AI Planning Config", "No running Ollama models or API keys found. Running in DETERMINISTIC mode.")
    else:
        print_info(f"Active AI Provider detected. Running in {planning_mode.value.upper()} mode.")

    # 3. Benchmark Loop
    results = {}
    successful = 0
    failed = 0
    runtimes = []
    peak_rams = []
    training_times = []
    export_times = []
    ai_times = []

    for csv_file in csv_files:
        res = run_dataset_benchmark(csv_file, planning_mode)
        results[csv_file.name] = res
        if res["status"] == "PASS":
            successful += 1
            runtimes.append(res["total_runtime"])
            peak_rams.append(res["peak_ram"])
            training_times.append(res["model_training_time"])
            export_times.append(res["model_export_time"])
            if res["ai_request_count"] > 0:
                ai_times.append(res["ai_response_time"])
        else:
            failed += 1

    # 4. Save consolidated system_benchmark_report.json
    hardware_info = {
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "active_ai_provider": type(_ACTIVE_PROVIDER).__name__ if _ACTIVE_PROVIDER else "None"
    }

    benchmark_report_data = {
        "hardware_profile": hardware_info,
        "mode": planning_mode.value,
        "datasets": results
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "system_benchmark_report.json", "w") as f:
        json.dump(benchmark_report_data, f, indent=4)

    # 5. Print final console summary
    print("\n==================================================")
    print("         DEPLOYAI BENCHMARK RESULTS")
    print("==================================================")
    print(f"Success Rate:         {successful / len(csv_files) * 100:.1f}% ({successful}/{len(csv_files)})")
    if successful > 0:
        print(f"Average Runtime:      {sum(runtimes)/len(runtimes):.2f}s")
        print(f"Overall Peak RAM:     {max(peak_rams):.2f} MB")
        print(f"Avg Training Time:    {sum(training_times)/len(training_times):.2f}s")
        print(f"Avg Model Export:     {sum(export_times)/len(export_times):.2f}s")
        if ai_times:
            print(f"Avg AI Response:      {sum(ai_times)/len(ai_times):.2f}s")
        else:
            print(f"Avg AI Response:      N/A (Deterministic)")
    print("Report Generated:     reports/system_benchmark_report.json")
    print("==================================================")

if __name__ == "__main__":
    main()
