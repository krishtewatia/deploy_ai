import pandas as pd
from pprint import pprint

from backend.app.analysis.analysis_service import AnalysisService

df = pd.DataFrame(
    {
        "age": [20, 21, None, 22],
        "department": ["AI", "ML", "AI", "DS"],
    }
)

service = AnalysisService()

report = service.analyze(df)

print("=" * 80)
print("REPORT TYPE")
print("=" * 80)

print(type(report))

print("\n")

print("=" * 80)
print("COLUMN PROFILES")
print("=" * 80)

pprint(report.column_profiles)

print("\n")

print("=" * 80)
print("AGE PROFILE")
print("=" * 80)

pprint(report.column_profiles["age"])