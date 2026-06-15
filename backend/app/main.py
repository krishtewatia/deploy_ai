from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.models import router as models_router
from backend.app.api.preprocessing import router as preprocessing_router
from backend.app.api.reports import router as reports_router
from backend.app.api.training import router as training_router
from backend.app.api.upload import router as upload_router
from backend.app.api.analysis import router as analysis_router
from backend.app.core.config import settings

app = FastAPI(title="AI Project Builder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router)
app.include_router(preprocessing_router)
app.include_router(reports_router)
app.include_router(training_router)
app.include_router(upload_router)
app.include_router(analysis_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}

