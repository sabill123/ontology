"""
Writeback Module (v17.0)

온톨로지 기반 데이터 수정:
- 객체 생성/수정/삭제
- 검증 및 승인 워크플로우
- 롤백 지원

사용 예시:
    engine = WritebackEngine(context)

    # 객체 생성
    action = WritebackAction.create("Customer", {"name": "New Customer"})
    result = await engine.execute(action)

    # 롤백
    await engine.rollback(result.action_id)
"""

from .writeback_engine import (
    WritebackEngine,
    WritebackAction,
    WritebackResult,
    ActionType,
)

__all__ = [
    "WritebackEngine",
    "WritebackAction",
    "WritebackResult",
    "ActionType",
]
