"""AI Providers configuration and factory package.

Provides configuration data contracts and factory mapping functionality
for Ollama and OpenAI-compatible AI services.
"""

from backend.app.ai_providers.schemas import (
    AIProviderType,
    AIProviderStatus,
    BaseAIProviderConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
    AIProviderConfig,
    AIProviderSettings,
)

from backend.app.ai_providers.factory import (
    AIProviderFactory,
    AIProviderFactoryError,
    OllamaAIProvider,
    OpenAICompatibleAIProvider,
)

from backend.app.ai_providers.presets import (
    AIProviderPresetId,
    AIProviderPreset,
    AIProviderPresetRegistry,
    AIProviderPresetRegistryError,
)

from backend.app.ai_providers.secret_store import (
    SecretStore,
    SecretStoreError,
    SystemSecretStore,
    InMemorySecretStore,
)

from backend.app.ai_providers.settings_store import (
    AIProviderSettingsStore,
    AIProviderSettingsStoreError,
)

from backend.app.ai_providers.validation import (
    AIProviderValidationStatus,
    AIProviderValidationIssue,
    AIProviderValidationResult,
    AIProviderConnectionValidator,
    AIProviderConnectionValidatorError,
)

__all__ = [
    "AIProviderType",
    "AIProviderStatus",
    "BaseAIProviderConfig",
    "OllamaProviderConfig",
    "OpenAICompatibleProviderConfig",
    "AIProviderConfig",
    "AIProviderSettings",
    "AIProviderFactory",
    "AIProviderFactoryError",
    "OllamaAIProvider",
    "OpenAICompatibleAIProvider",
    "AIProviderPresetId",
    "AIProviderPreset",
    "AIProviderPresetRegistry",
    "AIProviderPresetRegistryError",
    "SecretStore",
    "SecretStoreError",
    "SystemSecretStore",
    "InMemorySecretStore",
    "AIProviderSettingsStore",
    "AIProviderSettingsStoreError",
    "AIProviderValidationStatus",
    "AIProviderValidationIssue",
    "AIProviderValidationResult",
    "AIProviderConnectionValidator",
    "AIProviderConnectionValidatorError",
]
