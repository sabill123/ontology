# validation/__init__.py
"""
Validation module for FK Detection

다중 데이터셋 검증 및 일반화 성능 측정
"""

from .cross_dataset_validator import (
    CrossDatasetValidator,
    DatasetConfig,
    ValidationResult,
    GroundTruthFK,
)

__all__ = [
    "CrossDatasetValidator",
    "DatasetConfig",
    "ValidationResult",
    "GroundTruthFK",
]
