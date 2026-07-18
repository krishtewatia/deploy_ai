"""AI Model Critic package.

Provides structured critiques and grade cards for AutoML execution results using
LLM-based review analysis.
"""

from backend.app.ai_model_critic.schemas import CritiqueGrade, ModelCritique
from backend.app.ai_model_critic.critic_service import (
    AIModelCritic,
    AIModelCriticError,
)

__all__ = [
    "CritiqueGrade",
    "ModelCritique",
    "AIModelCritic",
    "AIModelCriticError",
]
