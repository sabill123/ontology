"""
Writeback Engine (v17.0)

온톨로지 기반 데이터 수정 엔진:
- CRUD 작업 실행
- 검증 및 제약조건 확인
- 롤백 지원
- Lineage 추적 연동

사용 예시:
    engine = WritebackEngine(context)

    # 생성
    action = WritebackAction.create("Customer", {"name": "Acme Corp"})
    result = await engine.execute(action)

    # 수정
    action = WritebackAction.update("Customer", "cust_123", {"status": "active"})
    result = await engine.execute(action)

    # 삭제
    action = WritebackAction.delete("Customer", "cust_123")
    result = await engine.execute(action)

    # 롤백
    await engine.rollback(result.action_id)
"""

import logging
import uuid
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """작업 유형"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"
    BATCH = "batch"


class ActionStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ValidationRule:
    """검증 규칙"""
    name: str
    validator: Callable[[Dict[str, Any]], bool]
    error_message: str
    severity: str = "error"  # error, warning

    def validate(self, data: Dict[str, Any]) -> tuple:
        """검증 실행"""
        try:
            is_valid = self.validator(data)
            return is_valid, None if is_valid else self.error_message
        except Exception as e:
            return False, f"{self.error_message}: {e}"


@dataclass
class WritebackAction:
    """Writeback 작업"""
    action_id: str
    action_type: ActionType
    object_type: str
    object_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    # 검증
    skip_validation: bool = False
    validation_errors: List[str] = field(default_factory=list)

    # 롤백 정보
    rollback_data: Optional[Dict[str, Any]] = None
    is_rollbackable: bool = True

    # 메타데이터
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"
    description: str = ""

    @classmethod
    def create(
        cls,
        object_type: str,
        data: Dict[str, Any],
        **kwargs,
    ) -> "WritebackAction":
        """생성 작업"""
        return cls(
            action_id=f"wb_{uuid.uuid4().hex[:8]}",
            action_type=ActionType.CREATE,
            object_type=object_type,
            data=data,
            **kwargs,
        )

    @classmethod
    def update(
        cls,
        object_type: str,
        object_id: str,
        data: Dict[str, Any],
        **kwargs,
    ) -> "WritebackAction":
        """수정 작업"""
        return cls(
            action_id=f"wb_{uuid.uuid4().hex[:8]}",
            action_type=ActionType.UPDATE,
            object_type=object_type,
            object_id=object_id,
            data=data,
            **kwargs,
        )

    @classmethod
    def delete(
        cls,
        object_type: str,
        object_id: str,
        **kwargs,
    ) -> "WritebackAction":
        """삭제 작업"""
        return cls(
            action_id=f"wb_{uuid.uuid4().hex[:8]}",
            action_type=ActionType.DELETE,
            object_type=object_type,
            object_id=object_id,
            **kwargs,
        )

    @classmethod
    def upsert(
        cls,
        object_type: str,
        object_id: str,
        data: Dict[str, Any],
        **kwargs,
    ) -> "WritebackAction":
        """Upsert 작업 (존재하면 수정, 없으면 생성)"""
        return cls(
            action_id=f"wb_{uuid.uuid4().hex[:8]}",
            action_type=ActionType.UPSERT,
            object_type=object_type,
            object_id=object_id,
            data=data,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "data": self.data,
            "validation_errors": self.validation_errors,
            "is_rollbackable": self.is_rollbackable,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "description": self.description,
        }


@dataclass
class WritebackResult:
    """Writeback 결과"""
    action_id: str
    status: ActionStatus
    success: bool
    object_id: Optional[str] = None

    # 결과 데이터
    result_data: Optional[Dict[str, Any]] = None
    affected_rows: int = 0

    # 에러 정보
    error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)

    # 타이밍
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "success": self.success,
            "object_id": self.object_id,
            "result_data": self.result_data,
            "affected_rows": self.affected_rows,
            "error": self.error,
            "validation_errors": self.validation_errors,
            "execution_time_ms": self.execution_time_ms,
        }


class WritebackEngine:
    """
    Writeback 실행 엔진

    기능:
    - CRUD 작업 실행
    - 검증 규칙 적용
    - 롤백 지원
    - Lineage 추적
    - Evidence Chain 기록
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
    ):
        """
        Args:
            context: SharedContext
        """
        self.context = context

        # 검증 규칙
        self.validation_rules: Dict[str, List[ValidationRule]] = {}

        # 작업 히스토리 (롤백용)
        self.action_history: Dict[str, WritebackAction] = {}

        # 통계
        self.stats = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "rolled_back_actions": 0,
        }

        logger.info("WritebackEngine initialized")

    def add_validation_rule(
        self,
        object_type: str,
        rule: ValidationRule,
    ) -> None:
        """검증 규칙 추가"""
        if object_type not in self.validation_rules:
            self.validation_rules[object_type] = []
        self.validation_rules[object_type].append(rule)

    async def execute(
        self,
        action: WritebackAction,
    ) -> WritebackResult:
        """
        Writeback 작업 실행

        Args:
            action: 실행할 작업

        Returns:
            WritebackResult: 실행 결과
        """
        start_time = datetime.now()
        result = WritebackResult(
            action_id=action.action_id,
            status=ActionStatus.PENDING,
            success=False,
            started_at=start_time.isoformat(),
        )

        try:
            # 1. 검증
            if not action.skip_validation:
                result.status = ActionStatus.VALIDATING
                validation_errors = self._validate(action)
                if validation_errors:
                    result.validation_errors = validation_errors
                    result.status = ActionStatus.FAILED
                    result.error = "Validation failed"
                    return result

            # 2. 롤백 데이터 저장
            if action.is_rollbackable:
                action.rollback_data = self._capture_rollback_data(action)

            # 3. 실행
            result.status = ActionStatus.EXECUTING

            if action.action_type == ActionType.CREATE:
                result = await self._execute_create(action, result)
            elif action.action_type == ActionType.UPDATE:
                result = await self._execute_update(action, result)
            elif action.action_type == ActionType.DELETE:
                result = await self._execute_delete(action, result)
            elif action.action_type == ActionType.UPSERT:
                result = await self._execute_upsert(action, result)

            # 4. 히스토리 저장
            if result.success:
                self.action_history[action.action_id] = action
                self.stats["successful_actions"] += 1
            else:
                self.stats["failed_actions"] += 1

            self.stats["total_actions"] += 1

        except Exception as e:
            result.status = ActionStatus.FAILED
            result.error = str(e)
            self.stats["failed_actions"] += 1
            self.stats["total_actions"] += 1
            logger.error(f"Writeback failed: {e}")

        # 타이밍
        end_time = datetime.now()
        result.completed_at = end_time.isoformat()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Evidence Chain 기록
        self._record_to_evidence_chain(action, result)

        # SharedContext 업데이트
        if self.context:
            self.context.writeback_history.append(result.to_dict())

        return result

    async def execute_batch(
        self,
        actions: List[WritebackAction],
        stop_on_error: bool = False,
    ) -> List[WritebackResult]:
        """
        배치 작업 실행

        Args:
            actions: 작업 목록
            stop_on_error: 에러 시 중단 여부

        Returns:
            결과 목록
        """
        results = []

        for action in actions:
            result = await self.execute(action)
            results.append(result)

            if not result.success and stop_on_error:
                # 이전 성공 작업 롤백
                for prev_result in reversed(results[:-1]):
                    if prev_result.success:
                        await self.rollback(prev_result.action_id)
                break

        return results

    async def rollback(
        self,
        action_id: str,
    ) -> WritebackResult:
        """
        작업 롤백

        Args:
            action_id: 롤백할 작업 ID

        Returns:
            WritebackResult: 롤백 결과
        """
        if action_id not in self.action_history:
            return WritebackResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                success=False,
                error=f"Action not found: {action_id}",
            )

        action = self.action_history[action_id]

        if not action.is_rollbackable or not action.rollback_data:
            return WritebackResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                success=False,
                error="Action is not rollbackable",
            )

        start_time = datetime.now()

        try:
            # 롤백 실행
            if action.action_type == ActionType.CREATE:
                # 생성 롤백 = 삭제
                result = await self._execute_delete(
                    WritebackAction.delete(
                        action.object_type,
                        action.rollback_data.get("created_id", ""),
                    ),
                    WritebackResult(
                        action_id=f"rollback_{action_id}",
                        status=ActionStatus.EXECUTING,
                        success=False,
                    ),
                )
            elif action.action_type == ActionType.UPDATE:
                # 수정 롤백 = 이전 값으로 수정
                result = await self._execute_update(
                    WritebackAction.update(
                        action.object_type,
                        action.object_id or "",
                        action.rollback_data.get("previous_data", {}),
                    ),
                    WritebackResult(
                        action_id=f"rollback_{action_id}",
                        status=ActionStatus.EXECUTING,
                        success=False,
                    ),
                )
            elif action.action_type == ActionType.DELETE:
                # 삭제 롤백 = 재생성
                result = await self._execute_create(
                    WritebackAction.create(
                        action.object_type,
                        action.rollback_data.get("deleted_data", {}),
                    ),
                    WritebackResult(
                        action_id=f"rollback_{action_id}",
                        status=ActionStatus.EXECUTING,
                        success=False,
                    ),
                )
            else:
                result = WritebackResult(
                    action_id=f"rollback_{action_id}",
                    status=ActionStatus.FAILED,
                    success=False,
                    error=f"Unsupported rollback for action type: {action.action_type}",
                )

            if result.success:
                result.status = ActionStatus.ROLLED_BACK
                self.stats["rolled_back_actions"] += 1

        except Exception as e:
            result = WritebackResult(
                action_id=f"rollback_{action_id}",
                status=ActionStatus.FAILED,
                success=False,
                error=str(e),
            )

        # 타이밍
        end_time = datetime.now()
        result.completed_at = end_time.isoformat()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(f"Rollback {'succeeded' if result.success else 'failed'}: {action_id}")

        return result

    def _validate(self, action: WritebackAction) -> List[str]:
        """작업 검증"""
        errors = []

        # 기본 검증
        if not action.object_type:
            errors.append("Object type is required")

        if action.action_type in (ActionType.UPDATE, ActionType.DELETE):
            if not action.object_id:
                errors.append("Object ID is required for update/delete")

        if action.action_type in (ActionType.CREATE, ActionType.UPDATE, ActionType.UPSERT):
            if not action.data:
                errors.append("Data is required for create/update")

        # 객체 타입별 검증 규칙
        rules = self.validation_rules.get(action.object_type, [])
        for rule in rules:
            is_valid, error = rule.validate(action.data)
            if not is_valid and error:
                if rule.severity == "error":
                    errors.append(error)
                else:
                    logger.warning(f"Validation warning: {error}")

        return errors

    def _capture_rollback_data(self, action: WritebackAction) -> Dict[str, Any]:
        """롤백 데이터 캡처"""
        rollback_data = {}

        if action.action_type in (ActionType.UPDATE, ActionType.DELETE):
            # 현재 데이터 캡처
            current_data = self._get_current_data(action.object_type, action.object_id)
            if current_data:
                rollback_data["previous_data"] = copy.deepcopy(current_data)
                rollback_data["deleted_data"] = copy.deepcopy(current_data)

        return rollback_data

    def _get_current_data(
        self,
        object_type: str,
        object_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """현재 데이터 조회"""
        if not self.context or not object_id:
            return None

        # 테이블에서 조회 (sample_data 기반)
        if object_type in self.context.tables:
            sample_data = self.context.tables[object_type].get("sample_data", [])
            for row in sample_data:
                # ID 컬럼 찾기
                if row.get("id") == object_id or row.get(f"{object_type}_id") == object_id:
                    return row

        return None

    async def _execute_create(
        self,
        action: WritebackAction,
        result: WritebackResult,
    ) -> WritebackResult:
        """생성 작업 실행"""
        # ID 생성
        new_id = action.data.get("id") or f"{action.object_type}_{uuid.uuid4().hex[:8]}"
        action.data["id"] = new_id

        # SharedContext에 추가 (sample_data에)
        if self.context and action.object_type in self.context.tables:
            sample_data = self.context.tables[action.object_type].get("sample_data", [])
            sample_data.append(action.data.copy())
            self.context.tables[action.object_type]["sample_data"] = sample_data

        # 롤백 데이터 업데이트
        if action.rollback_data is not None:
            action.rollback_data["created_id"] = new_id

        result.status = ActionStatus.COMPLETED
        result.success = True
        result.object_id = new_id
        result.result_data = action.data.copy()
        result.affected_rows = 1

        logger.info(f"Created {action.object_type}: {new_id}")

        return result

    async def _execute_update(
        self,
        action: WritebackAction,
        result: WritebackResult,
    ) -> WritebackResult:
        """수정 작업 실행"""
        if not self.context or action.object_type not in self.context.tables:
            result.status = ActionStatus.FAILED
            result.error = f"Table not found: {action.object_type}"
            return result

        sample_data = self.context.tables[action.object_type].get("sample_data", [])
        updated = False

        for i, row in enumerate(sample_data):
            if row.get("id") == action.object_id or row.get(f"{action.object_type}_id") == action.object_id:
                # 업데이트
                sample_data[i].update(action.data)
                updated = True
                result.result_data = sample_data[i].copy()
                break

        if updated:
            result.status = ActionStatus.COMPLETED
            result.success = True
            result.object_id = action.object_id
            result.affected_rows = 1
            logger.info(f"Updated {action.object_type}: {action.object_id}")
        else:
            result.status = ActionStatus.FAILED
            result.error = f"Object not found: {action.object_id}"

        return result

    async def _execute_delete(
        self,
        action: WritebackAction,
        result: WritebackResult,
    ) -> WritebackResult:
        """삭제 작업 실행"""
        if not self.context or action.object_type not in self.context.tables:
            result.status = ActionStatus.FAILED
            result.error = f"Table not found: {action.object_type}"
            return result

        sample_data = self.context.tables[action.object_type].get("sample_data", [])
        original_len = len(sample_data)

        # 삭제
        sample_data = [
            row for row in sample_data
            if row.get("id") != action.object_id and
               row.get(f"{action.object_type}_id") != action.object_id
        ]

        self.context.tables[action.object_type]["sample_data"] = sample_data

        deleted = original_len - len(sample_data)

        if deleted > 0:
            result.status = ActionStatus.COMPLETED
            result.success = True
            result.object_id = action.object_id
            result.affected_rows = deleted
            logger.info(f"Deleted {action.object_type}: {action.object_id}")
        else:
            result.status = ActionStatus.FAILED
            result.error = f"Object not found: {action.object_id}"

        return result

    async def _execute_upsert(
        self,
        action: WritebackAction,
        result: WritebackResult,
    ) -> WritebackResult:
        """Upsert 작업 실행"""
        # 존재 여부 확인
        current_data = self._get_current_data(action.object_type, action.object_id)

        if current_data:
            # 업데이트
            return await self._execute_update(action, result)
        else:
            # 생성
            action.data["id"] = action.object_id
            return await self._execute_create(action, result)

    def _record_to_evidence_chain(
        self,
        action: WritebackAction,
        result: WritebackResult,
    ) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ..autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "WRITEBACK_ACTION",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="writeback",
                agent="WritebackEngine",
                evidence_type=evidence_type,
                finding=f"{action.action_type.value.upper()} {action.object_type}",
                reasoning=action.description or f"Execute {action.action_type.value}",
                conclusion=(
                    f"Success: {result.object_id}"
                    if result.success
                    else f"Failed: {result.error}"
                ),
                metrics={
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "object_type": action.object_type,
                    "object_id": result.object_id,
                    "affected_rows": result.affected_rows,
                    "execution_time_ms": result.execution_time_ms,
                },
                confidence=0.95 if result.success else 0.3,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_writeback_summary(self) -> Dict[str, Any]:
        """Writeback 요약"""
        return {
            "stats": self.stats,
            "validation_rules_count": {
                obj_type: len(rules)
                for obj_type, rules in self.validation_rules.items()
            },
            "pending_rollbacks": len(self.action_history),
            "recent_actions": [
                action.to_dict()
                for action in list(self.action_history.values())[-5:]
            ],
        }
