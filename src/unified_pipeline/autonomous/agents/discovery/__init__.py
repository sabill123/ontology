"""
Discovery Phase Autonomous Agents Package

v4.2에서 discovery_utils.py로 유틸리티를 분리하고,
v25.1에서 단일 discovery.py를 패키지 구조로 재분리.

Re-export all agent classes for backward compatibility:
  from .agents.discovery import DataAnalystAutonomousAgent  # 기존 import 유지
"""

from .data_analyst import DataAnalystAutonomousAgent
from .tda_expert import TDAExpertAutonomousAgent
from .schema_analyst import SchemaAnalystAutonomousAgent
from .value_matcher import ValueMatcherAutonomousAgent
from .entity_classifier import EntityClassifierAutonomousAgent
from .relationship_detector import RelationshipDetectorAutonomousAgent

__all__ = [
    "DataAnalystAutonomousAgent",
    "TDAExpertAutonomousAgent",
    "SchemaAnalystAutonomousAgent",
    "ValueMatcherAutonomousAgent",
    "EntityClassifierAutonomousAgent",
    "RelationshipDetectorAutonomousAgent",
]
