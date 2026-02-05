"""
Object Behavior Layer (v2.0)

팔란티어 스타일 온톨로지의 핵심 - "개념이 행동한다"

핵심 검증 기준:
✅ 비즈니스 개념이 1급 객체 - Behavior는 ObjectType에 귀속
✅ Object → Action 구조 - SQL 쿼리가 아닌 객체 메서드 호출
✅ 데이터 소스와 개념 분리 - Behavior는 ObjectInstance 추상화로 동작

사용 예시:
    # 행동 정의
    evaluate_risk = BehaviorDefinition(
        behavior_id="order.evaluate_risk",
        name="evaluateRisk",
        behavior_type=BehaviorType.COMPUTATION,
        execution_mode=ExecutionMode.PYTHON,
        python_code='''
            score = self.get("amount") * 0.001 + self.get("customer_risk_score", 0) * 0.5
            result = {"risk_score": min(score, 1.0), "factors": ["amount", "customer_history"]}
        ''',
        return_type="object"
    )

    # 레지스트리에 등록
    registry = GlobalBehaviorRegistry.register_type("Order")
    registry.register(evaluate_risk)

    # 에이전트에서 호출 (v2.0 패턴)
    result = await self.call_behavior(order, "evaluateRisk")
    if result.success and result.result["risk_score"] > 0.9:
        await self.transition(order, "block")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import logging

if TYPE_CHECKING:
    from .models import ObjectInstance, ObjectType

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """
    행동 유형

    팔란티어 AIP의 Function 유형과 매핑:
    - QUERY: 읽기 전용 조회 (부수효과 없음)
    - MUTATION: 상태 변경 (write-back, update)
    - VALIDATION: 데이터 검증 (SHACL 스타일)
    - COMPUTATION: 계산/추론 (리스크 스코어, 집계 등)
    - ACTION: 외부 시스템 호출 (알림, 동기화 등)
    """
    QUERY = "query"              # 조회 (부수효과 없음)
    MUTATION = "mutation"        # 상태 변경
    VALIDATION = "validation"    # 검증
    COMPUTATION = "computation"  # 계산/추론
    ACTION = "action"            # 외부 시스템 호출


class ExecutionMode(Enum):
    """
    실행 방식

    팔란티어 Functions의 실행 모드:
    - PYTHON: 안전한 샌드박스에서 Python 코드 실행
    - LLM: LLM 프롬프트 실행 (AIP Agent 스타일)
    - OQL: OQL 쿼리 실행 (Ontology Query Language)
    - HYBRID: Python 데이터 준비 + LLM 추론
    """
    PYTHON = "python"            # Python 코드 직접 실행
    LLM = "llm"                  # LLM 프롬프트 실행
    OQL = "oql"                  # OQL 쿼리 실행
    HYBRID = "hybrid"            # Python + LLM 조합


@dataclass
class BehaviorParameter:
    """
    행동 파라미터 정의

    SHACL 스타일 제약조건 지원:
    - 타입 검증
    - 필수 여부
    - 커스텀 검증 규칙
    """
    name: str
    param_type: str  # "string", "number", "boolean", "entity_ref", "list", "object"
    required: bool = True
    default: Any = None
    description: str = ""

    # SHACL 스타일 검증 규칙 (Python 표현식)
    validation_rule: Optional[str] = None  # e.g., "value > 0", "len(value) <= 100"

    # 엔티티 참조인 경우 대상 타입
    reference_type: Optional[str] = None  # e.g., "Customer", "Order"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.param_type,
            "required": self.required,
            "default": self.default,
            "description": self.description,
            "validation_rule": self.validation_rule,
            "reference_type": self.reference_type,
        }


@dataclass
class BehaviorDefinition:
    """
    행동 정의

    팔란티어 Ontology의 핵심 - "개념이 행동을 가진다"

    검증 기준:
    ✅ 로직이 "객체의 행동"으로 표현됨
    ✅ AIP Logic/Functions로 캡슐화됨
    ✅ 데이터 소스 독립적 (ObjectInstance 추상화)

    예시:
    - order.calculateTotal() -> number
    - order.block(reason: string) -> BlockResult
    - customer.getActiveOrders() -> List[Order]
    - order.evaluateRisk() -> RiskAssessment
    """
    # 식별자
    behavior_id: str  # "order.calculate_total", "customer.evaluate_risk"
    name: str         # "calculateTotal", "evaluateRisk" (메서드명 스타일)

    # 유형
    behavior_type: BehaviorType
    description: str = ""

    # 실행 모드
    execution_mode: ExecutionMode = ExecutionMode.PYTHON

    # 입출력 정의
    parameters: List[BehaviorParameter] = field(default_factory=list)
    return_type: Optional[str] = None  # "number", "boolean", "object", "List[Order]"

    # 실행 내용 (execution_mode에 따라 하나만 사용)
    python_code: Optional[str] = None         # ExecutionMode.PYTHON
    llm_prompt_template: Optional[str] = None # ExecutionMode.LLM
    oql_query: Optional[str] = None           # ExecutionMode.OQL

    # 제약 조건 (Design-by-Contract)
    preconditions: List[str] = field(default_factory=list)   # 사전 조건 (Python 표현식)
    postconditions: List[str] = field(default_factory=list)  # 사후 조건 (Python 표현식)

    # 권한 (Governance)
    required_permissions: List[str] = field(default_factory=list)

    # 트리거 (성공/실패 시 연쇄 행동)
    triggers_on_success: List[str] = field(default_factory=list)  # action_id 목록
    triggers_on_failure: List[str] = field(default_factory=list)

    # 실행 설정
    is_async: bool = True
    timeout_seconds: int = 30
    retry_count: int = 0
    cache_ttl_seconds: Optional[int] = None  # 결과 캐싱 (QUERY 타입에 유용)

    # 상태 전이 연동 (State Machine과 연결)
    triggers_transition: Optional[str] = None  # 성공 시 자동 전이할 상태

    # 메타데이터
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "name": self.name,
            "behavior_type": self.behavior_type.value,
            "execution_mode": self.execution_mode.value,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "required_permissions": self.required_permissions,
            "triggers_on_success": self.triggers_on_success,
            "triggers_on_failure": self.triggers_on_failure,
            "timeout_seconds": self.timeout_seconds,
            "version": self.version,
        }

    def validate_preconditions(self, instance: "ObjectInstance") -> Optional[str]:
        """
        사전 조건 검증

        Returns:
            실패 시 오류 메시지, 성공 시 None
        """
        for cond in self.preconditions:
            try:
                # self는 ObjectInstance를 참조
                local_vars = {"self": instance}
                if not eval(cond, {"__builtins__": {}}, local_vars):
                    return f"Precondition failed: {cond}"
            except Exception as e:
                return f"Precondition error: {cond} - {str(e)}"
        return None


@dataclass
class BehaviorRegistry:
    """
    ObjectType별 행동 레지스트리

    각 ObjectType은 자신만의 행동 집합을 가짐
    - Order는 calculateTotal, block, evaluateRisk 등
    - Customer는 getActiveOrders, updateCreditScore 등
    """
    object_type_id: str
    behaviors: Dict[str, BehaviorDefinition] = field(default_factory=dict)

    def register(self, behavior: BehaviorDefinition) -> "BehaviorRegistry":
        """
        행동 등록 (fluent API)

        예:
            registry.register(calculate_total).register(evaluate_risk)
        """
        self.behaviors[behavior.behavior_id] = behavior
        logger.debug(f"[BehaviorRegistry] Registered {behavior.behavior_id} for {self.object_type_id}")
        return self

    def unregister(self, behavior_id: str) -> bool:
        """행동 등록 해제"""
        if behavior_id in self.behaviors:
            del self.behaviors[behavior_id]
            return True
        return False

    def get(self, behavior_id: str) -> Optional[BehaviorDefinition]:
        """행동 ID로 조회"""
        return self.behaviors.get(behavior_id)

    def get_by_name(self, name: str) -> Optional[BehaviorDefinition]:
        """메서드명으로 조회 (e.g., "evaluateRisk")"""
        for b in self.behaviors.values():
            if b.name == name:
                return b
        return None

    def list_by_type(self, behavior_type: BehaviorType) -> List[BehaviorDefinition]:
        """타입별 행동 목록"""
        return [b for b in self.behaviors.values() if b.behavior_type == behavior_type]

    def list_mutations(self) -> List[BehaviorDefinition]:
        """상태 변경 행동 목록 (MUTATION 타입)"""
        return self.list_by_type(BehaviorType.MUTATION)

    def list_queries(self) -> List[BehaviorDefinition]:
        """조회 행동 목록 (QUERY 타입)"""
        return self.list_by_type(BehaviorType.QUERY)

    def list_computations(self) -> List[BehaviorDefinition]:
        """계산 행동 목록 (COMPUTATION 타입)"""
        return self.list_by_type(BehaviorType.COMPUTATION)

    def list_validations(self) -> List[BehaviorDefinition]:
        """검증 행동 목록 (VALIDATION 타입)"""
        return self.list_by_type(BehaviorType.VALIDATION)

    def list_all(self) -> List[BehaviorDefinition]:
        """모든 행동 목록"""
        return list(self.behaviors.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_type_id": self.object_type_id,
            "behaviors": {k: v.to_dict() for k, v in self.behaviors.items()},
            "behavior_count": len(self.behaviors),
        }

    def get_schema_for_llm(self) -> str:
        """
        LLM에 제공할 스키마 정보

        AIP Agent가 객체의 행동을 이해하고 호출할 수 있도록
        구조화된 정보 제공
        """
        lines = [f"ObjectType: {self.object_type_id}", "Available Behaviors:"]

        for b in self.behaviors.values():
            params = ", ".join([f"{p.name}: {p.param_type}" for p in b.parameters])
            lines.append(f"  - {b.name}({params}) -> {b.return_type or 'void'}")
            lines.append(f"    Type: {b.behavior_type.value}")
            if b.description:
                lines.append(f"    Description: {b.description}")

        return "\n".join(lines)


class GlobalBehaviorRegistry:
    """
    전역 행동 레지스트리

    모든 ObjectType의 행동을 중앙에서 관리

    팔란티어 검증 기준:
    ✅ 비즈니스 개념(ObjectType)별로 행동이 정의됨
    ✅ 에이전트가 개념 단위로 행동을 호출할 수 있음
    """
    _registries: Dict[str, BehaviorRegistry] = {}

    @classmethod
    def register_type(cls, object_type_id: str) -> BehaviorRegistry:
        """
        ObjectType용 레지스트리 생성/반환

        예:
            order_registry = GlobalBehaviorRegistry.register_type("Order")
            order_registry.register(calculate_total_behavior)
        """
        if object_type_id not in cls._registries:
            cls._registries[object_type_id] = BehaviorRegistry(object_type_id)
            logger.info(f"[GlobalBehaviorRegistry] Created registry for {object_type_id}")
        return cls._registries[object_type_id]

    @classmethod
    def get_registry(cls, object_type_id: str) -> Optional[BehaviorRegistry]:
        """ObjectType의 레지스트리 조회"""
        return cls._registries.get(object_type_id)

    @classmethod
    def get_behavior(cls, object_type_id: str, behavior_id: str) -> Optional[BehaviorDefinition]:
        """특정 행동 조회"""
        registry = cls._registries.get(object_type_id)
        if registry:
            return registry.get(behavior_id)
        return None

    @classmethod
    def get_behavior_by_name(cls, object_type_id: str, name: str) -> Optional[BehaviorDefinition]:
        """메서드명으로 행동 조회"""
        registry = cls._registries.get(object_type_id)
        if registry:
            return registry.get_by_name(name)
        return None

    @classmethod
    def list_all_types(cls) -> List[str]:
        """등록된 모든 ObjectType ID 목록"""
        return list(cls._registries.keys())

    @classmethod
    def get_all_behaviors(cls) -> Dict[str, List[BehaviorDefinition]]:
        """모든 타입의 모든 행동"""
        return {
            type_id: registry.list_all()
            for type_id, registry in cls._registries.items()
        }

    @classmethod
    def clear(cls) -> None:
        """모든 레지스트리 초기화 (테스트용)"""
        cls._registries.clear()

    @classmethod
    def get_schema_for_llm(cls) -> str:
        """
        LLM에 제공할 전체 스키마

        AIP Agent가 전체 온톨로지의 행동을 이해하고
        적절한 행동을 선택할 수 있도록 정보 제공
        """
        lines = ["=== Ontology Behavior Schema ===", ""]

        for type_id, registry in cls._registries.items():
            lines.append(registry.get_schema_for_llm())
            lines.append("")

        return "\n".join(lines)


# 빌트인 행동 생성 헬퍼

def create_query_behavior(
    object_type_id: str,
    name: str,
    python_code: str,
    return_type: str = "object",
    description: str = "",
    cache_ttl_seconds: Optional[int] = 60,
) -> BehaviorDefinition:
    """조회 행동 생성 헬퍼"""
    return BehaviorDefinition(
        behavior_id=f"{object_type_id.lower()}.{name}",
        name=name,
        behavior_type=BehaviorType.QUERY,
        execution_mode=ExecutionMode.PYTHON,
        description=description,
        python_code=python_code,
        return_type=return_type,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def create_mutation_behavior(
    object_type_id: str,
    name: str,
    python_code: str,
    parameters: List[BehaviorParameter] = None,
    description: str = "",
    required_permissions: List[str] = None,
) -> BehaviorDefinition:
    """상태 변경 행동 생성 헬퍼"""
    return BehaviorDefinition(
        behavior_id=f"{object_type_id.lower()}.{name}",
        name=name,
        behavior_type=BehaviorType.MUTATION,
        execution_mode=ExecutionMode.PYTHON,
        description=description,
        python_code=python_code,
        parameters=parameters or [],
        required_permissions=required_permissions or [],
    )


def create_computation_behavior(
    object_type_id: str,
    name: str,
    python_code: str,
    return_type: str = "number",
    description: str = "",
) -> BehaviorDefinition:
    """계산 행동 생성 헬퍼"""
    return BehaviorDefinition(
        behavior_id=f"{object_type_id.lower()}.{name}",
        name=name,
        behavior_type=BehaviorType.COMPUTATION,
        execution_mode=ExecutionMode.PYTHON,
        description=description,
        python_code=python_code,
        return_type=return_type,
    )


def create_validation_behavior(
    object_type_id: str,
    name: str,
    python_code: str,
    description: str = "",
) -> BehaviorDefinition:
    """검증 행동 생성 헬퍼"""
    return BehaviorDefinition(
        behavior_id=f"{object_type_id.lower()}.{name}",
        name=name,
        behavior_type=BehaviorType.VALIDATION,
        execution_mode=ExecutionMode.PYTHON,
        description=description,
        python_code=python_code,
        return_type="boolean",
    )


def create_llm_behavior(
    object_type_id: str,
    name: str,
    prompt_template: str,
    behavior_type: BehaviorType = BehaviorType.COMPUTATION,
    return_type: str = "object",
    description: str = "",
) -> BehaviorDefinition:
    """LLM 기반 행동 생성 헬퍼"""
    return BehaviorDefinition(
        behavior_id=f"{object_type_id.lower()}.{name}",
        name=name,
        behavior_type=behavior_type,
        execution_mode=ExecutionMode.LLM,
        description=description,
        llm_prompt_template=prompt_template,
        return_type=return_type,
    )


__all__ = [
    # Enums
    "BehaviorType",
    "ExecutionMode",
    # Core Classes
    "BehaviorParameter",
    "BehaviorDefinition",
    "BehaviorRegistry",
    "GlobalBehaviorRegistry",
    # Helpers
    "create_query_behavior",
    "create_mutation_behavior",
    "create_computation_behavior",
    "create_validation_behavior",
    "create_llm_behavior",
]
