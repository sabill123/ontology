"""
Sync Module (v16.0)

Real-time synchronization with CDC (Change Data Capture).
"""

from .cdc_engine import (
    CDCEngine,
    ChangeEvent,
    ChangeType,
    ConflictStrategy,
    SyncStatus,
    SyncTask,
    Subscription,
    ConflictError,
    create_cdc_engine_with_context,
)

__all__ = [
    "CDCEngine",
    "ChangeEvent",
    "ChangeType",
    "ConflictStrategy",
    "SyncStatus",
    "SyncTask",
    "Subscription",
    "ConflictError",
    "create_cdc_engine_with_context",
]
