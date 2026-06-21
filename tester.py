import pandas as pd

from backend.app.preprocessing_engine.preprocessing_service import (
    PreprocessingService,
)

df = pd.DataFrame(
    {
        "department": ["AI", "AI", "ML"],
        "salary": [50000, None, 70000],
    }
)

result = PreprocessingService().process(df)

print(result.model_dump_json(indent=2))