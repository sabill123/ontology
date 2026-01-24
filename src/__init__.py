"""
Ontoloty - 데이터 사일로 통합 플랫폼 (v3.0 - Autonomous Multi-Agent)

Architecture (Autonomous Pipeline):
- Discovery Stage: 데이터 발견, TDA, Homeomorphism 탐지 (자율 에이전트)
- Refinement Stage: 온톨로지 정제, 충돌 해결, 품질 평가 (자율 에이전트)
- Governance Stage: 거버넌스 결정, 액션 백로그, 정책 생성 (자율 에이전트)

핵심 원칙:
1. Todo 기반 작업 관리 (DAG 의존성)
2. 자율 에이전트 (Work Loop: claim → execute → complete)
3. Phase 자연 진행 (의존성 기반)
4. 실시간 이벤트 스트리밍 (FE 모니터링)

Components:
- unified_pipeline: 자율 멀티에이전트 파이프라인 (Active)
- common: 공유 유틸리티, LLM Judge
- _legacy: 이전 버전 코드 (참조용)
"""

__version__ = "3.0.0"

# Primary entry point - Autonomous Pipeline
from .unified_pipeline import (
    AutonomousPipelineOrchestrator,
    CompatiblePipelineOrchestrator,
    SharedContext,
)

__all__ = [
    "AutonomousPipelineOrchestrator",
    "CompatiblePipelineOrchestrator",
    "SharedContext",
]
