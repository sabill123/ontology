"""
Unified Multi-Agent Pipeline (v3.0 - Autonomous)

자율 멀티에이전트 파이프라인 - Todo 기반 자율 실행

핵심 원칙:
1. Todo 기반 작업 관리 (DAG 의존성)
2. 자율 에이전트 (Work Loop)
3. Phase 자연 진행 (의존성 기반)
4. 실시간 이벤트 스트리밍 (FE 모니터링)

아키텍처:
┌─────────────────────────────────────────────────────────────────┐
│              Autonomous Multi-Agent Pipeline (v3.0)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Todo Manager (DAG)                       │   │
│  │  • PipelineTodo with dependencies                         │   │
│  │  • Status: PENDING → READY → IN_PROGRESS → COMPLETED      │   │
│  │  • Phase별 Todo 템플릿                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓↑                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Agent Orchestrator                           │   │
│  │  • 에이전트 풀 관리                                       │   │
│  │  • Todo 할당 및 분배                                      │   │
│  │  • 병렬 실행                                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓↑                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Autonomous Agents (13개)                     │   │
│  │  • Work Loop: claim_todo → execute → complete             │   │
│  │  • Phase별 전문 에이전트                                   │   │
│  │  • 자율적 작업 가져오기                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓↑                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Shared Context Memory                        │   │
│  │  • Tables, TDA Signatures, Homeomorphisms                 │   │
│  │  • Ontology, Governance Decisions                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Event Emitter (실시간 스트리밍)              │   │
│  │  • WebSocket/SSE 지원                                     │   │
│  │  • FE 모니터링 대시보드                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

# === Core Components ===
from .shared_context import SharedContext, StageResult

# === Autonomous System ===
from .autonomous_pipeline import (
    AutonomousPipelineOrchestrator,
    CompatiblePipelineOrchestrator,
)

# === Unified Main System (v3.0 통합 진입점) ===
from .unified_main import (
    OntologyPlatform,
    PipelineConfig,
    PipelineResult,
)

# === Alias for compatibility ===
UnifiedPipelineOrchestrator = AutonomousPipelineOrchestrator

# === Todo System ===
from .todo import (
    TodoManager,
    TodoEventEmitter,
    PipelineTodo,
    TodoStatus,
    TodoPriority,
    TodoResult,
    TodoEvent,
    TodoEventType,
)

# === Autonomous Agents ===
from .autonomous import (
    AutonomousAgent,
    AgentState,
    AgentOrchestrator,
    TodoContinuationEnforcer,
)

# === Model Configuration ===
from .model_config import (
    ModelType,
    MODELS,
    get_model,
    get_agent_model,
    get_agent_config,
    UNIFIED_AGENT_CONFIGS,
    print_model_assignments,
)

__all__ = [
    # Core
    'SharedContext',
    'StageResult',

    # === Unified Main System (v3.0 권장 진입점) ===
    'OntologyPlatform',  # 사용자가 데이터만 업로드하면 자동 실행
    'PipelineConfig',
    'PipelineResult',

    # Autonomous Pipeline
    'AutonomousPipelineOrchestrator',
    'CompatiblePipelineOrchestrator',
    'UnifiedPipelineOrchestrator',  # Alias

    # Todo System
    'TodoManager',
    'TodoEventEmitter',
    'PipelineTodo',
    'TodoStatus',
    'TodoPriority',
    'TodoResult',
    'TodoEvent',
    'TodoEventType',

    # Autonomous Agents
    'AutonomousAgent',
    'AgentState',
    'AgentOrchestrator',
    'TodoContinuationEnforcer',

    # Model configuration
    'ModelType',
    'MODELS',
    'get_model',
    'get_agent_model',
    'get_agent_config',
    'UNIFIED_AGENT_CONFIGS',
    'print_model_assignments',
]
