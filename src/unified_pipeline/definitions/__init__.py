"""
Definitions - 도메인별 행동/상태 정의 (v2.0)

Ontoloty 시스템의 비즈니스 개념에 대한 행동과 상태 머신 정의:
- OntologyConcept: 온톨로지 개념 (Entity, Relationship)
- Order: 주문 (예시)
- Customer: 고객 (예시)

팔란티어 검증 기준:
✅ 비즈니스 개념이 1급 객체로 정의됨
✅ 각 개념이 고유한 행동(Behavior)을 가짐
✅ 상태 전이가 명시적 (StateMachine)

사용 예시:
    from definitions import register_all_behaviors, get_state_machine

    # 모든 행동 등록
    register_all_behaviors()

    # 상태 머신 조회
    order_sm = get_state_machine("Order")
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..osdk.state_machine import StateMachine

# 상태 머신 레지스트리 (v2.0-B에서 채워짐)
_state_machines = {}


def get_state_machine(object_type_id: str) -> Optional["StateMachine"]:
    """ObjectType의 상태 머신 조회"""
    return _state_machines.get(object_type_id)


def register_state_machine(object_type_id: str, state_machine: "StateMachine") -> None:
    """상태 머신 등록"""
    _state_machines[object_type_id] = state_machine


def register_all_behaviors() -> None:
    """
    모든 도메인별 행동 등록

    시스템 시작 시 호출하여 모든 행동을 GlobalBehaviorRegistry에 등록
    """
    from .ontology_concept import register_ontology_concept_behaviors

    # OntologyConcept 행동 등록
    register_ontology_concept_behaviors()

    # 추후 추가될 도메인별 행동
    # from .order import register_order_behaviors
    # from .customer import register_customer_behaviors


__all__ = [
    "get_state_machine",
    "register_state_machine",
    "register_all_behaviors",
]
