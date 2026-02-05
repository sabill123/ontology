"""
OSDK - Ontology SDK (v2.1)

팔란티어 스타일 온톨로지 SDK:

v1.0 (기존):
- ObjectType: 객체 유형 정의 (엔티티)
- LinkType: 관계 유형 정의 (FK 기반)
- Property: 속성 정의
- QueryBuilder: 타입 안전 쿼리 빌드

v2.0 (신규):
- BehaviorDefinition: 객체 행동 정의 (Object → Action)
- BehaviorRegistry: 타입별 행동 레지스트리
- BehaviorExecutor: 행동 실행기 (Python/LLM/OQL)
- StateMachine: 명시적 상태 전이

v2.1 (신규):
- DataAdapter: 데이터 소스 추상화 레이어
- SharedContextAdapter: SharedContext 어댑터
- CompositeAdapter: 복합 데이터 소스 통합

팔란티어 검증 기준:
✅ 비즈니스 개념이 1급 객체 (ObjectType + Behavior)
✅ Object → Action 구조 (BehaviorExecutor)
✅ AI/Agent가 개념 단위로 reasoning (LLM 모드)
✅ 데이터 소스 변경이 로직을 깨지 않음 (DataAdapter)

사용 예시:
    # v1.0 방식 (기존 유지)
    client = OSDKClient(context)
    customer = client.object_type("Customer")

    # v2.0 방식 (신규)
    from osdk.behavior import GlobalBehaviorRegistry, BehaviorDefinition, BehaviorType

    # 행동 정의
    registry = GlobalBehaviorRegistry.register_type("Order")
    registry.register(BehaviorDefinition(
        behavior_id="order.evaluate_risk",
        name="evaluateRisk",
        behavior_type=BehaviorType.COMPUTATION,
        python_code="result = {'risk_score': self.get('amount') * 0.001}"
    ))

    # 에이전트에서 행동 호출
    result = await self.call_behavior(order, "evaluateRisk")
"""

from .models import (
    ObjectType,
    LinkType,
    Property,
    PropertyType,
    Cardinality,
    ObjectInstance,
    TypeRegistry,
)
from .query_builder import QueryBuilder, QueryResult
from .client import OSDKClient

# v2.0 Behavior Layer
from .behavior import (
    BehaviorType,
    ExecutionMode,
    BehaviorParameter,
    BehaviorDefinition,
    BehaviorRegistry,
    GlobalBehaviorRegistry,
    create_query_behavior,
    create_mutation_behavior,
    create_computation_behavior,
    create_validation_behavior,
    create_llm_behavior,
)
from .executor import (
    ExecutionResult,
    ExecutionErrorType,
    BehaviorExecutor,
)

# v2.0 State Machine Layer
from .state_machine import (
    StateDefinition,
    TransitionDefinition,
    TransitionResult,
    TransitionResultStatus,
    StateHistory,
    StateMachine,
    StateMachineExecutor,
    GlobalStateMachineRegistry,
)

# v2.1 Data Adapter Layer
from .adapter import (
    DataSourceType,
    ObjectProxy,
    TypeMapping,
    DataAdapter,
    SharedContextAdapter,
    SupabaseAdapter,
    CompositeAdapter,
    GlobalAdapterRegistry,
    create_adapter_for_context,
    create_supabase_adapter,
    create_hybrid_adapter,
)

__all__ = [
    # Models (v1.0)
    "ObjectType",
    "LinkType",
    "Property",
    "PropertyType",
    "Cardinality",
    "ObjectInstance",
    "TypeRegistry",
    # Query (v1.0)
    "QueryBuilder",
    "QueryResult",
    # Client (v1.0)
    "OSDKClient",
    # Behavior (v2.0)
    "BehaviorType",
    "ExecutionMode",
    "BehaviorParameter",
    "BehaviorDefinition",
    "BehaviorRegistry",
    "GlobalBehaviorRegistry",
    "create_query_behavior",
    "create_mutation_behavior",
    "create_computation_behavior",
    "create_validation_behavior",
    "create_llm_behavior",
    # Executor (v2.0)
    "ExecutionResult",
    "ExecutionErrorType",
    "BehaviorExecutor",
    # State Machine (v2.0)
    "StateDefinition",
    "TransitionDefinition",
    "TransitionResult",
    "TransitionResultStatus",
    "StateHistory",
    "StateMachine",
    "StateMachineExecutor",
    "GlobalStateMachineRegistry",
    # Data Adapter (v2.1)
    "DataSourceType",
    "ObjectProxy",
    "TypeMapping",
    "DataAdapter",
    "SharedContextAdapter",
    "SupabaseAdapter",
    "CompositeAdapter",
    "GlobalAdapterRegistry",
    "create_adapter_for_context",
    "create_supabase_adapter",
    "create_hybrid_adapter",
]
