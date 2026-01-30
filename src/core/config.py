"""
Application Configuration
Centralized configuration for the entire application
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """Global application configuration"""

    # Application Info
    APP_NAME: str = "Crypto AI Predictor"
    VERSION: str = "3.0.0"

    # Data Configuration
    DATASET_DIR: str = "dataset"
    DEFAULT_INTERVAL: str = "1h"
    DEFAULT_MAX_CANDLES: int = 10000
    QUOTE_ASSET: str = "USDT"

    # API Configuration
    BINANCE_BASE_URL: str = "https://api.binance.com/api/v3"
    API_TIMEOUT: int = 10
    REQUEST_DELAY_MS: int = 200

    # Data Freshness
    FRESHNESS_THRESHOLD_HOURS: int = 24

    # GUI Configuration
    WINDOW_WIDTH: int = 1600
    WINDOW_HEIGHT: int = 900

    # Colors (Dark Theme)
    COLOR_BG_DARK: str = "#1e1e1e"
    COLOR_BG_MEDIUM: str = "#2d2d2d"
    COLOR_BG_LIGHT: str = "#3d3d3d"
    COLOR_PRIMARY: str = "#0d47a1"
    COLOR_SUCCESS: str = "#2e7d32"
    COLOR_DANGER: str = "#c62828"
    COLOR_TEXT_PRIMARY: str = "#e0e0e0"
    COLOR_TEXT_SECONDARY: str = "#9e9e9e"
    COLOR_ACCENT: str = "#4fc3f7"

    # Fonts
    FONT_FAMILY: str = "Arial"
    FONT_MONO: str = "Consolas"
    FONT_SIZE_SMALL: int = 9
    FONT_SIZE_NORMAL: int = 10
    FONT_SIZE_LARGE: int = 12
    FONT_SIZE_HEADER: int = 16

    # ML Configuration
    MODEL_DIR: str = "models"
    CHECKPOINT_DIR: str = "models/checkpoints"
    PRODUCTION_MODEL_DIR: str = "models/production"
    EXPERIMENTS_DIR: str = "models/experiments"

    # ML Training Defaults (aligned with ver11_gpt RankNet hyperparameters)
    ML_DEFAULT_HIDDEN_SIZE: int = 128
    ML_DEFAULT_NUM_LAYERS: int = 3
    ML_DEFAULT_DROPOUT: float = 0.1
    ML_DEFAULT_BATCH_SIZE: int = 32
    ML_DEFAULT_EPOCHS: int = 20
    ML_DEFAULT_LR: float = 0.001
    ML_CONTEXT_LENGTH: int = 40000  # 40k hours of history
    ML_PREDICTION_HORIZON: int = 30  # 30 candles ahead
    ML_TRAIN_SPLIT: float = 0.70
    ML_VAL_SPLIT: float = 0.15
    ML_TEST_SPLIT: float = 0.15

    @classmethod
    def get_dataset_path(cls) -> Path:
        """Get dataset directory path"""
        return Path(cls.DATASET_DIR)

    @classmethod
    def get_model_path(cls) -> Path:
        """Get model directory path"""
        return Path(cls.MODEL_DIR)

    @classmethod
    def get_checkpoint_path(cls) -> Path:
        """Get checkpoint directory path"""
        return Path(cls.CHECKPOINT_DIR)
