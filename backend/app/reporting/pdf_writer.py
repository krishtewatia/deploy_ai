"""PDF writer converting ExecutiveReport schemas into styled PDF files using fpdf2."""

from __future__ import annotations

import os
from fpdf import FPDF
from backend.app.reporting.schemas import ExecutiveReport


class PDFReportWriter:
    """Consolidates AutoML pipeline execution summaries into a professional PDF report."""

    def write_pdf(self, report: ExecutiveReport, output_path: str) -> None:
        """Render the ExecutiveReport instance to a PDF file on disk.

        Args:
            report: The populated ExecutiveReport model schema.
            output_path: Absolute target path to write the PDF binary file.
        """
        # Ensure target folder directory exists
        target_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(target_dir, exist_ok=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # 1. Header Styling Accent block
        pdf.set_fill_color(26, 54, 93)  # Dark Slate Blue Accent
        pdf.rect(0, 0, 210, 40, "F")
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(pdf.epw, 15, "DeployAI AutoML Executive Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", "I", 10)
        pdf.cell(
            pdf.epw, 
            5, 
            f"Report ID: {report.report_id}  |  Generated: {report.generated_timestamp}", 
            new_x="LMARGIN", 
            new_y="NEXT", 
            align="C"
        )
        
        pdf.ln(20)
        
        # Reset default dark text color
        pdf.set_text_color(33, 37, 41)
        
        # 2. Executive Summary Block
        pdf.set_font("helvetica", "B", 12)
        pdf.set_fill_color(240, 244, 248)  # Soft blue-gray fill
        pdf.cell(pdf.epw, 8, " Executive Summary", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("helvetica", "", 10)
        pdf.ln(2)
        summary_clean = str(report.executive_summary).encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(pdf.epw, 5, summary_clean)
        pdf.ln(8)
        
        # 3. Problem & Dataset Metadata
        pdf.set_font("helvetica", "B", 12)
        pdf.set_fill_color(240, 244, 248)
        pdf.cell(pdf.epw, 8, " Dataset & Problem Definition", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(2)
        pdf.set_font("helvetica", "", 10)
        
        prob_type = report.problem_summary.get("problem_type", "N/A")
        target_col = report.problem_summary.get("target_column", "N/A")
        dataset_id = report.dataset_summary.get("dataset_id", "N/A")
        
        rows = report.dataset_summary.get("rows", "N/A")
        cols = report.dataset_summary.get("columns", "N/A")
        if cols == "N/A":
            features = report.dataset_summary.get("features", [])
            cols = len(features) if features else report.dataset_summary.get("feature_count", "N/A")
            
        pdf.cell(95, 6, f"Problem Type:  {prob_type}")
        pdf.cell(95, 6, f"Dataset Name:  {dataset_id}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(95, 6, f"Target Column: {target_col}")
        pdf.cell(95, 6, f"Features:      {cols} features detected", new_x="LMARGIN", new_y="NEXT")
        if rows != "N/A":
            pdf.cell(95, 6, f"Dataset Size:  {rows} rows", new_x="LMARGIN", new_y="NEXT")
            
        pdf.ln(8)
        
        # 4. Candidate Model Table
        pdf.set_font("helvetica", "B", 12)
        pdf.set_fill_color(240, 244, 248)
        pdf.cell(pdf.epw, 8, " Model Performance Benchmarks", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(3)
        
        # Table headers
        pdf.set_font("helvetica", "B", 10)
        pdf.set_fill_color(220, 227, 235)
        pdf.cell(60, 8, "Candidate ID", border=1, align="C", fill=True)
        pdf.cell(60, 8, "Algorithm", border=1, align="C", fill=True)
        pdf.cell(40, 8, "Metric", border=1, align="C", fill=True)
        pdf.cell(30, 8, "Value", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "", 10)
        for model in report.models_summary:
            cand_id = model.get("candidate_id", model.get("family", "N/A"))
            algo = model.get("algorithm", model.get("family", "N/A"))
            metric = model.get("primary_metric", model.get("metric", "N/A"))
            val_raw = model.get("metric_value", model.get("score", 0.0))
            val = f"{val_raw:.4f}" if isinstance(val_raw, (int, float)) else str(val_raw)
            
            pdf.cell(60, 8, str(cand_id), border=1, align="C")
            pdf.cell(60, 8, str(algo), border=1, align="C")
            pdf.cell(40, 8, str(metric), border=1, align="C")
            pdf.cell(30, 8, val, border=1, align="C", new_x="LMARGIN", new_y="NEXT")
            
        pdf.ln(8)
        
        # 5. Champion Governance Card
        pdf.set_font("helvetica", "B", 12)
        pdf.set_fill_color(240, 244, 248)
        pdf.cell(pdf.epw, 8, " Champion Selection Governance", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(2)
        
        pdf.set_font("helvetica", "", 10)
        champ_id = report.champion_summary.get("candidate_id", "N/A")
        champ_algo = report.champion_summary.get("algorithm", report.champion_summary.get("model_family", "N/A"))
        winner_cfg = report.champion_summary.get("winner_configuration", report.governance_summary.get("winner", "N/A"))
        ready = report.deployment_summary.get("production_ready", "N/A")
        
        pdf.cell(95, 6, f"Selected Champion ID: {champ_id}")
        pdf.cell(95, 6, f"Algorithm Family:     {champ_algo}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(95, 6, f"Winner Governance:    {winner_cfg}")
        pdf.cell(95, 6, f"Production Ready:     {ready}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(2)
        reason_clean = report.governance_summary.get("decision_reason", "N/A").encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(pdf.epw, 5, f"Governance Reason: {reason_clean}")
        pdf.ln(8)
        
        # 6. AI Model Review Card
        pdf.set_font("helvetica", "B", 12)
        pdf.set_fill_color(240, 244, 248)
        pdf.cell(pdf.epw, 8, " AI Critique & Recommendations", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(2)
        
        pdf.set_font("helvetica", "", 10)
        critique = report.ai_review.get("critique")
        if critique:
            critique_clean = str(critique).encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(pdf.epw, 5, critique_clean)
        else:
            grade = report.ai_review.get("overall_grade", "N/A")
            conf = report.ai_review.get("confidence", "N/A")
            summary = report.ai_review.get("summary", "N/A")
            
            pdf.cell(95, 6, f"Overall AI Grade: {grade}")
            pdf.cell(95, 6, f"AI Confidence Score: {conf}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            if summary and summary != "N/A":
                summary_clean = str(summary).encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(pdf.epw, 5, f"AI Summary: {summary_clean}")
                
            strengths = report.ai_review.get("strengths", [])
            weaknesses = report.ai_review.get("weaknesses", [])
            if strengths or weaknesses:
                pdf.ln(2)
                pdf.set_font("helvetica", "B", 10)
                pdf.cell(pdf.epw, 6, "Identified Strengths & Weaknesses:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("helvetica", "", 10)
                for s in strengths[:3]:
                    pdf.cell(
                        pdf.epw, 
                        5, 
                        f"- Strength: {s}".encode("latin-1", "replace").decode("latin-1"), 
                        new_x="LMARGIN", 
                        new_y="NEXT"
                    )
                for w in weaknesses[:3]:
                    pdf.cell(
                        pdf.epw, 
                        5, 
                        f"- Weakness: {w}".encode("latin-1", "replace").decode("latin-1"), 
                        new_x="LMARGIN", 
                        new_y="NEXT"
                    )
                    
        # 7. Warnings Card
        if report.warnings:
            pdf.ln(6)
            pdf.set_font("helvetica", "B", 12)
            pdf.set_fill_color(255, 235, 235)  # Warn light red fill
            pdf.cell(pdf.epw, 8, " Execution Warnings", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(2)
            pdf.set_font("helvetica", "", 10)
            for warn in report.warnings:
                warn_clean = str(warn).encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(pdf.epw, 5, f"[WARN] {warn_clean}")
                
        # 8. Output PDF Binary
        pdf.output(output_path)
