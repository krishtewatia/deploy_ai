import pandas as pd

from backend.app.analysis.analysis_service import AnalysisService

df = pd.DataFrame({
    "name": ["Krish", "Rahul", None],
    "age": [20, None, 22],
    "salary": [50000, 60000, None]
})

report = AnalysisService().analyze(df)

print(report.model_dump())