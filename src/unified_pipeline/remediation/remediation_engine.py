"""
Remediation Engine (v17.0)

데이터 품질 자동 수정 엔진:
- 이상 탐지 기반 수정
- 다양한 수정 전략
- 영향 분석
- 승인 워크플로우

사용 예시:
    engine = RemediationEngine(context)

    # 수정 제안 생성
    suggestions = await engine.suggest_remediation("customers", "missing_values")

    # 수정 적용
    result = await engine.apply(suggestions[0])

    # 롤백
    await engine.rollback(result.action_id)
"""

import logging
import uuid
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


def _get_table_sample_data(table_info) -> list:
    """TableInfo(dataclass) 또는 dict에서 sample_data를 안전하게 추출"""
    if hasattr(table_info, 'sample_data'):
        return table_info.sample_data or []
    elif isinstance(table_info, dict):
        return table_info.get("sample_data", [])
    return []


def _set_table_sample_data(table_info, data):
    """TableInfo(dataclass) 또는 dict에 sample_data를 안전하게 설정"""
    if hasattr(table_info, 'sample_data'):
        table_info.sample_data = data
    elif isinstance(table_info, dict):
        table_info["sample_data"] = data


class IssueType(str, Enum):
    """이슈 유형"""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    FORMAT_ERRORS = "format_errors"
    INCONSISTENT_TYPES = "inconsistent_types"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    CONSTRAINT_VIOLATIONS = "constraint_violations"


class RemediationStrategy(str, Enum):
    """수정 전략"""
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_DEFAULT = "fill_default"
    REMOVE_ROW = "remove_row"
    CLIP_VALUES = "clip_values"
    DEDUPLICATE = "deduplicate"
    FIX_FORMAT = "fix_format"
    INFER_VALUE = "infer_value"  # LLM 기반 추론


class ApprovalStatus(str, Enum):
    """승인 상태"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


@dataclass
class AffectedEntity:
    """영향받는 엔티티"""
    table: str
    row_id: str
    column: str
    current_value: Any
    proposed_value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "row_id": self.row_id,
            "column": self.column,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
        }


@dataclass
class ImpactAnalysis:
    """영향 분석"""
    affected_rows: int
    affected_columns: List[str]
    downstream_tables: List[str]
    risk_level: str  # low, medium, high
    reversible: bool

    # 상세
    affected_entities: List[AffectedEntity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "affected_rows": self.affected_rows,
            "affected_columns": self.affected_columns,
            "downstream_tables": self.downstream_tables,
            "risk_level": self.risk_level,
            "reversible": self.reversible,
            "affected_entities": [e.to_dict() for e in self.affected_entities],
        }


@dataclass
class RemediationAction:
    """수정 작업"""
    action_id: str
    issue_type: IssueType
    strategy: RemediationStrategy
    table: str
    column: Optional[str] = None

    # 영향 분석
    impact: Optional[ImpactAnalysis] = None

    # 승인
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

    # 수정 상세
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    confidence: float = 0.0

    # 메타데이터
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "issue_type": self.issue_type.value,
            "strategy": self.strategy.value,
            "table": self.table,
            "column": self.column,
            "impact": self.impact.to_dict() if self.impact else None,
            "approval_status": self.approval_status.value,
            "parameters": self.parameters,
            "description": self.description,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


@dataclass
class RemediationResult:
    """수정 결과"""
    action_id: str
    success: bool
    rows_modified: int = 0
    error: Optional[str] = None

    # 롤백 정보
    rollback_data: Dict[str, Any] = field(default_factory=dict)

    # 타이밍
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "rows_modified": self.rows_modified,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class RemediationSummary:
    """전체 remediate 결과 요약"""
    table: str
    actions_taken: int = 0
    issues_fixed: int = 0
    rows_modified: int = 0
    success_rate: float = 0.0
    results: List[RemediationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "actions_taken": self.actions_taken,
            "issues_fixed": self.issues_fixed,
            "rows_modified": self.rows_modified,
            "success_rate": self.success_rate,
        }


class RemediationEngine:
    """
    데이터 품질 자동 수정 엔진 (v17.0 - LLM 통합)

    기능:
    - 이슈 탐지 및 수정 제안
    - 영향 분석
    - 승인 워크플로우
    - 수정 적용 및 롤백
    - LLM 기반 값 추론 (INFER_VALUE 전략)
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        auto_approve_threshold: float = 0.9,
        llm_client=None,
    ):
        """
        Args:
            context: SharedContext
            auto_approve_threshold: 자동 승인 신뢰도 임계값
            llm_client: LLM 클라이언트 (값 추론에 사용)
        """
        self.context = context
        self.auto_approve_threshold = auto_approve_threshold
        self.llm_client = llm_client

        # 수정 히스토리
        self.actions: Dict[str, RemediationAction] = {}
        self.results: Dict[str, RemediationResult] = {}

        # 통계
        self.stats = {
            "total_issues_detected": 0,
            "total_remediations": 0,
            "successful_remediations": 0,
            "rollbacks": 0,
        }

        logger.info(f"RemediationEngine initialized (LLM: {'enabled' if llm_client else 'disabled'})")

    async def detect_issues(
        self,
        table: str,
        issue_types: Optional[List[IssueType]] = None,
    ) -> Dict[IssueType, List[Dict[str, Any]]]:
        """
        이슈 탐지

        Args:
            table: 테이블 이름
            issue_types: 탐지할 이슈 유형 (None이면 전체)

        Returns:
            이슈 유형별 발견된 이슈 목록
        """
        if not self.context or table not in self.context.tables:
            logger.error(f"Table not found: {table}")
            return {}

        table_info = self.context.tables[table]
        # TableInfo가 dataclass인 경우와 dict인 경우 모두 지원
        if hasattr(table_info, 'sample_data'):
            sample_data = table_info.sample_data or []
        elif isinstance(table_info, dict):
            sample_data = table_info.get("sample_data", [])
        else:
            sample_data = []

        if not sample_data:
            return {}

        issue_types = issue_types or list(IssueType)
        detected = {}

        for issue_type in issue_types:
            issues = self._detect_issue_type(table, sample_data, issue_type)
            if issues:
                detected[issue_type] = issues
                self.stats["total_issues_detected"] += len(issues)

        return detected

    async def suggest_remediation(
        self,
        table: str,
        issue_type: IssueType,
    ) -> List[RemediationAction]:
        """
        수정 제안 생성

        Args:
            table: 테이블 이름
            issue_type: 이슈 유형

        Returns:
            수정 작업 목록
        """
        issues = await self.detect_issues(table, [issue_type])

        if issue_type not in issues:
            return []

        suggestions = []

        for issue in issues[issue_type]:
            action = self._create_remediation_action(table, issue_type, issue)
            if action:
                # 영향 분석
                action.impact = self._analyze_impact(action)

                # 자동 승인 여부
                if action.confidence >= self.auto_approve_threshold:
                    action.approval_status = ApprovalStatus.AUTO_APPROVED

                self.actions[action.action_id] = action
                suggestions.append(action)

        return suggestions

    async def apply(
        self,
        action: RemediationAction,
    ) -> RemediationResult:
        """
        수정 적용

        Args:
            action: 적용할 수정 작업

        Returns:
            수정 결과
        """
        start_time = datetime.now()

        # 승인 확인
        if action.approval_status not in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED):
            return RemediationResult(
                action_id=action.action_id,
                success=False,
                error="Action not approved",
            )

        result = RemediationResult(
            action_id=action.action_id,
            success=False,
            started_at=start_time.isoformat(),
        )

        try:
            # 롤백 데이터 저장
            result.rollback_data = self._capture_rollback_data(action)

            # 전략별 수정 적용
            if action.strategy == RemediationStrategy.FILL_MEAN:
                rows = self._apply_fill_mean(action)
            elif action.strategy == RemediationStrategy.FILL_MEDIAN:
                rows = self._apply_fill_median(action)
            elif action.strategy == RemediationStrategy.FILL_MODE:
                rows = self._apply_fill_mode(action)
            elif action.strategy == RemediationStrategy.FILL_DEFAULT:
                rows = self._apply_fill_default(action)
            elif action.strategy == RemediationStrategy.REMOVE_ROW:
                rows = self._apply_remove_rows(action)
            elif action.strategy == RemediationStrategy.CLIP_VALUES:
                rows = self._apply_clip_values(action)
            elif action.strategy == RemediationStrategy.DEDUPLICATE:
                rows = self._apply_deduplicate(action)
            elif action.strategy == RemediationStrategy.INFER_VALUE:
                rows = await self._apply_infer_value(action)
            else:
                rows = 0

            result.success = True
            result.rows_modified = rows
            self.stats["successful_remediations"] += 1

        except Exception as e:
            result.error = str(e)
            logger.error(f"Remediation failed: {e}")

        # 타이밍
        end_time = datetime.now()
        result.completed_at = end_time.isoformat()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        self.results[action.action_id] = result
        self.stats["total_remediations"] += 1

        # Evidence Chain 기록
        self._record_to_evidence_chain(action, result)

        return result

    async def rollback(self, action_id: str) -> bool:
        """
        수정 롤백

        Args:
            action_id: 롤백할 작업 ID

        Returns:
            성공 여부
        """
        if action_id not in self.results:
            logger.error(f"Result not found: {action_id}")
            return False

        result = self.results[action_id]

        if not result.rollback_data:
            logger.error(f"No rollback data for: {action_id}")
            return False

        try:
            # 롤백 데이터 복원
            action = self.actions.get(action_id)
            if not action or not self.context:
                return False

            table = action.table
            if table not in self.context.tables:
                return False

            # 원본 데이터 복원
            _set_table_sample_data(
                self.context.tables[table],
                result.rollback_data.get("original_data", [])
            )

            self.stats["rollbacks"] += 1
            logger.info(f"Rolled back action: {action_id}")

            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def approve(
        self,
        action_id: str,
        approved_by: str = "user",
    ) -> bool:
        """수정 승인"""
        if action_id not in self.actions:
            return False

        action = self.actions[action_id]
        action.approval_status = ApprovalStatus.APPROVED
        action.approved_by = approved_by
        action.approved_at = datetime.now().isoformat()

        return True

    def reject(self, action_id: str) -> bool:
        """수정 거부"""
        if action_id not in self.actions:
            return False

        self.actions[action_id].approval_status = ApprovalStatus.REJECTED
        return True

    async def remediate(
        self,
        table: str,
        auto_approve: bool = False,
        issue_types: Optional[List[IssueType]] = None,
    ) -> RemediationSummary:
        """
        테이블 데이터 품질 이슈 자동 수정 (편의 메서드)

        탐지 → 제안 → 승인 → 적용 전체 과정 실행

        Args:
            table: 테이블 이름
            auto_approve: 자동 승인 여부
            issue_types: 처리할 이슈 유형 (None이면 전체)

        Returns:
            RemediationSummary 결과 요약
        """
        summary = RemediationSummary(table=table)

        if not self.context or table not in self.context.tables:
            logger.warning(f"Table not found for remediation: {table}")
            return summary

        # 1. 이슈 탐지
        detected_issues = await self.detect_issues(table, issue_types)

        if not detected_issues:
            logger.info(f"No issues found in table: {table}")
            return summary

        # 2. 각 이슈 유형별 처리
        all_results = []
        for issue_type, issues in detected_issues.items():
            if not issues:
                continue

            # 3. 수정 제안 생성
            suggestions = await self.suggest_remediation(table, issue_type)

            for action in suggestions:
                # 4. 자동 승인 또는 신뢰도 기준 승인
                if auto_approve or action.confidence >= self.auto_approve_threshold:
                    self.approve(action.action_id, approved_by="auto")

                    # 5. 수정 적용
                    result = await self.apply(action)
                    all_results.append(result)

                    if result.success:
                        summary.issues_fixed += 1
                        summary.rows_modified += result.rows_modified

        summary.actions_taken = len(all_results)
        summary.results = all_results
        summary.success_rate = (
            summary.issues_fixed / summary.actions_taken
            if summary.actions_taken > 0
            else 0.0
        )

        logger.info(
            f"Remediation complete for {table}: "
            f"{summary.actions_taken} actions, "
            f"{summary.issues_fixed} fixed, "
            f"{summary.rows_modified} rows modified"
        )

        return summary

    def _detect_issue_type(
        self,
        table: str,
        data: List[Dict[str, Any]],
        issue_type: IssueType,
    ) -> List[Dict[str, Any]]:
        """이슈 유형별 탐지"""
        issues = []

        if issue_type == IssueType.MISSING_VALUES:
            issues = self._detect_missing_values(data)
        elif issue_type == IssueType.OUTLIERS:
            issues = self._detect_outliers(data)
        elif issue_type == IssueType.DUPLICATES:
            issues = self._detect_duplicates(data)

        return issues

    def _detect_missing_values(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """결측값 탐지"""
        issues = []

        if not data:
            return issues

        columns = data[0].keys()

        for col in columns:
            missing_rows = []
            for i, row in enumerate(data):
                value = row.get(col)
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    missing_rows.append(i)

            if missing_rows:
                issues.append({
                    "column": col,
                    "missing_count": len(missing_rows),
                    "missing_rows": missing_rows,
                    "total_rows": len(data),
                    "missing_rate": len(missing_rows) / len(data),
                })

        return issues

    def _detect_outliers(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """이상치 탐지 (Z-score 기반)"""
        issues = []

        if not data:
            return issues

        for col in data[0].keys():
            # 숫자 컬럼만
            values = []
            for row in data:
                try:
                    values.append(float(row.get(col, 0)))
                except (ValueError, TypeError):
                    break
            else:
                if len(values) > 3:
                    mean = statistics.mean(values)
                    std = statistics.stdev(values)

                    if std > 0:
                        outlier_rows = []
                        for i, v in enumerate(values):
                            z = abs(v - mean) / std
                            if z > 3:
                                outlier_rows.append({
                                    "row": i,
                                    "value": v,
                                    "z_score": round(z, 2),
                                })

                        if outlier_rows:
                            issues.append({
                                "column": col,
                                "outlier_count": len(outlier_rows),
                                "outliers": outlier_rows,
                                "mean": round(mean, 2),
                                "std": round(std, 2),
                            })

        return issues

    def _detect_duplicates(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """중복 탐지"""
        issues = []

        if not data:
            return issues

        # 전체 행 기준 중복
        seen = {}
        duplicates = []

        for i, row in enumerate(data):
            key = tuple(sorted(row.items()))
            if key in seen:
                duplicates.append({
                    "row": i,
                    "duplicate_of": seen[key],
                })
            else:
                seen[key] = i

        if duplicates:
            issues.append({
                "type": "full_row",
                "duplicate_count": len(duplicates),
                "duplicates": duplicates,
            })

        return issues

    def _create_remediation_action(
        self,
        table: str,
        issue_type: IssueType,
        issue: Dict[str, Any],
    ) -> Optional[RemediationAction]:
        """수정 작업 생성"""
        action_id = f"rem_{uuid.uuid4().hex[:8]}"

        if issue_type == IssueType.MISSING_VALUES:
            column = issue.get("column")
            missing_rate = issue.get("missing_rate", 0)

            # 전략 선택
            if missing_rate < 0.3:
                strategy = RemediationStrategy.FILL_MEAN
                confidence = 0.8
            else:
                strategy = RemediationStrategy.FILL_DEFAULT
                confidence = 0.6

            return RemediationAction(
                action_id=action_id,
                issue_type=issue_type,
                strategy=strategy,
                table=table,
                column=column,
                parameters={
                    "missing_rows": issue.get("missing_rows", []),
                },
                description=f"Fill {issue.get('missing_count')} missing values in {column}",
                confidence=confidence,
            )

        elif issue_type == IssueType.OUTLIERS:
            column = issue.get("column")

            return RemediationAction(
                action_id=action_id,
                issue_type=issue_type,
                strategy=RemediationStrategy.CLIP_VALUES,
                table=table,
                column=column,
                parameters={
                    "outliers": issue.get("outliers", []),
                    "mean": issue.get("mean"),
                    "std": issue.get("std"),
                },
                description=f"Clip {issue.get('outlier_count')} outliers in {column}",
                confidence=0.7,
            )

        elif issue_type == IssueType.DUPLICATES:
            return RemediationAction(
                action_id=action_id,
                issue_type=issue_type,
                strategy=RemediationStrategy.DEDUPLICATE,
                table=table,
                parameters={
                    "duplicates": issue.get("duplicates", []),
                },
                description=f"Remove {issue.get('duplicate_count')} duplicate rows",
                confidence=0.85,
            )

        return None

    def _analyze_impact(self, action: RemediationAction) -> ImpactAnalysis:
        """영향 분석"""
        affected_rows = 0
        affected_columns = []
        downstream_tables = []

        if action.column:
            affected_columns.append(action.column)

        if action.strategy == RemediationStrategy.REMOVE_ROW:
            affected_rows = len(action.parameters.get("rows_to_remove", []))
        elif action.strategy == RemediationStrategy.DEDUPLICATE:
            affected_rows = len(action.parameters.get("duplicates", []))
        else:
            affected_rows = len(action.parameters.get("missing_rows", []))
            affected_rows += len(action.parameters.get("outliers", []))

        # FK 기반 다운스트림 테이블 찾기
        if self.context and action.table in self.context.tables:
            table_info = self.context.tables[action.table]
            fk_list = table_info.foreign_keys if hasattr(table_info, 'foreign_keys') else table_info.get("foreign_keys", []) if isinstance(table_info, dict) else []
            downstream_tables = [fk.get("to_table", fk.get("target_table", "")) for fk in fk_list if isinstance(fk, dict)]

        # 리스크 레벨 결정
        if affected_rows > 100 or downstream_tables:
            risk_level = "high"
        elif affected_rows > 10:
            risk_level = "medium"
        else:
            risk_level = "low"

        return ImpactAnalysis(
            affected_rows=affected_rows,
            affected_columns=affected_columns,
            downstream_tables=downstream_tables,
            risk_level=risk_level,
            reversible=True,
        )

    def _capture_rollback_data(
        self,
        action: RemediationAction,
    ) -> Dict[str, Any]:
        """롤백 데이터 캡처"""
        if not self.context or action.table not in self.context.tables:
            return {}

        import copy
        return {
            "original_data": copy.deepcopy(
                _get_table_sample_data(self.context.tables[action.table])
            ),
        }

    def _apply_fill_mean(self, action: RemediationAction) -> int:
        """평균값으로 채우기"""
        if not self.context or action.table not in self.context.tables:
            return 0

        data = _get_table_sample_data(self.context.tables[action.table])
        column = action.column

        # 평균 계산
        values = []
        for row in data:
            try:
                v = float(row.get(column, 0))
                if v is not None:
                    values.append(v)
            except (ValueError, TypeError):
                pass

        if not values:
            return 0

        mean_val = statistics.mean(values)

        # 채우기
        modified = 0
        for row in data:
            if row.get(column) is None or row.get(column) == "":
                row[column] = mean_val
                modified += 1

        return modified

    def _apply_fill_median(self, action: RemediationAction) -> int:
        """중앙값으로 채우기"""
        if not self.context or action.table not in self.context.tables:
            return 0

        data = _get_table_sample_data(self.context.tables[action.table])
        column = action.column

        values = []
        for row in data:
            try:
                v = float(row.get(column, 0))
                if v is not None:
                    values.append(v)
            except (ValueError, TypeError):
                pass

        if not values:
            return 0

        median_val = statistics.median(values)

        modified = 0
        for row in data:
            if row.get(column) is None or row.get(column) == "":
                row[column] = median_val
                modified += 1

        return modified

    def _apply_fill_mode(self, action: RemediationAction) -> int:
        """최빈값으로 채우기"""
        if not self.context or action.table not in self.context.tables:
            return 0

        data = _get_table_sample_data(self.context.tables[action.table])
        column = action.column

        values = [row.get(column) for row in data if row.get(column)]

        if not values:
            return 0

        try:
            mode_val = statistics.mode(values)
        except statistics.StatisticsError:
            mode_val = values[0]

        modified = 0
        for row in data:
            if row.get(column) is None or row.get(column) == "":
                row[column] = mode_val
                modified += 1

        return modified

    def _apply_fill_default(self, action: RemediationAction) -> int:
        """기본값으로 채우기"""
        if not self.context or action.table not in self.context.tables:
            return 0

        data = _get_table_sample_data(self.context.tables[action.table])
        column = action.column
        default_value = action.parameters.get("default_value", "N/A")

        modified = 0
        for row in data:
            if row.get(column) is None or row.get(column) == "":
                row[column] = default_value
                modified += 1

        return modified

    def _apply_remove_rows(self, action: RemediationAction) -> int:
        """행 제거"""
        if not self.context or action.table not in self.context.tables:
            return 0

        rows_to_remove = set(action.parameters.get("rows_to_remove", []))
        data = _get_table_sample_data(self.context.tables[action.table])

        new_data = [
            row for i, row in enumerate(data)
            if i not in rows_to_remove
        ]

        removed = len(data) - len(new_data)
        _set_table_sample_data(self.context.tables[action.table], new_data)

        return removed

    def _apply_clip_values(self, action: RemediationAction) -> int:
        """이상치 클리핑"""
        if not self.context or action.table not in self.context.tables:
            return 0

        data = _get_table_sample_data(self.context.tables[action.table])
        column = action.column
        mean = action.parameters.get("mean", 0)
        std = action.parameters.get("std", 1)

        lower = mean - 3 * std
        upper = mean + 3 * std

        modified = 0
        for row in data:
            try:
                v = float(row.get(column, 0))
                if v < lower:
                    row[column] = lower
                    modified += 1
                elif v > upper:
                    row[column] = upper
                    modified += 1
            except (ValueError, TypeError):
                pass

        return modified

    def _apply_deduplicate(self, action: RemediationAction) -> int:
        """중복 제거"""
        if not self.context or action.table not in self.context.tables:
            return 0

        data = _get_table_sample_data(self.context.tables[action.table])

        seen = set()
        new_data = []

        for row in data:
            key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                new_data.append(row)

        removed = len(data) - len(new_data)
        _set_table_sample_data(self.context.tables[action.table], new_data)

        return removed

    async def _apply_infer_value(self, action: RemediationAction) -> int:
        """
        LLM 기반 값 추론

        컨텍스트를 분석하여 결측값을 지능적으로 추론합니다.
        LLM이 없으면 기본값으로 폴백합니다.
        """
        if not self.context or action.table not in self.context.tables:
            return 0

        data = _get_table_sample_data(self.context.tables[action.table])
        column = action.column

        if not data or not column:
            return 0

        # 결측 행 찾기
        missing_indices = []
        for i, row in enumerate(data):
            value = row.get(column)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing_indices.append(i)

        if not missing_indices:
            return 0

        # LLM이 없으면 통계 기반 폴백
        if not self.llm_client:
            logger.info("LLM not available, falling back to statistical imputation")
            return self._apply_fill_mode(action)

        try:
            # LLM으로 값 추론
            inferred_values = await self._infer_values_with_llm(
                table=action.table,
                column=column,
                data=data,
                missing_indices=missing_indices,
            )

            # 추론된 값 적용
            modified = 0
            for idx, value in inferred_values.items():
                if idx < len(data):
                    data[idx][column] = value
                    modified += 1

            return modified

        except Exception as e:
            logger.error(f"LLM inference failed: {e}, falling back to mode")
            return self._apply_fill_mode(action)

    async def _infer_values_with_llm(
        self,
        table: str,
        column: str,
        data: List[Dict[str, Any]],
        missing_indices: List[int],
    ) -> Dict[int, Any]:
        """
        LLM을 사용하여 결측값 추론 (AI Gateway 패턴)

        Args:
            table: 테이블 이름
            column: 대상 컬럼
            data: 전체 데이터
            missing_indices: 결측 행 인덱스

        Returns:
            {row_index: inferred_value} 딕셔너리
        """
        from ..model_config import get_v17_service_model
        from ...common.utils.llm import chat_completion

        model = get_v17_service_model("remediation_engine")

        # 컨텍스트 구성: 완전한 행 샘플 + 결측 행
        complete_rows = [
            row for i, row in enumerate(data)
            if i not in missing_indices
        ]

        missing_rows = [
            {"index": i, "row": data[i]}
            for i in missing_indices
        ]

        # 컬럼 메타데이터
        if self.context and table in self.context.tables:
            ti = self.context.tables[table]
            cols = ti.columns if hasattr(ti, 'columns') else ti.get("columns", {}) if isinstance(ti, dict) else {}
            col_info = cols.get(column, {}) if isinstance(cols, dict) else {}
            dtype = col_info.get("dtype", "unknown")
            sample_values = col_info.get("sample_values", [])
        else:
            dtype = "unknown"
            sample_values = list(set(
                row.get(column) for row in complete_rows
                if row.get(column) is not None
            ))

        prompt = f"""당신은 데이터 품질 전문가입니다. 결측값을 추론해주세요.

## 테이블: {table}
## 컬럼: {column}
## 데이터 타입: {dtype}
## 기존 값 예시: {sample_values}

## 완전한 행 샘플:
{self._format_rows_for_prompt(complete_rows)}

## 결측값이 있는 행:
{self._format_missing_rows_for_prompt(missing_rows, column)}

각 결측 행에 대해 다른 컬럼의 값과 데이터 패턴을 분석하여
{column} 컬럼의 적절한 값을 추론해주세요.

JSON 형식으로 응답해주세요:
{{
    "inferences": [
        {{"index": <행번호>, "value": <추론값>, "reasoning": "<추론 이유>"}},
        ...
    ]
}}
"""

        try:
            # v18.0: asyncio.to_thread로 블로킹 호출 방지
            import asyncio
            response = await asyncio.to_thread(
                chat_completion,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # 낮은 temperature로 일관성 유지
            )

            content = response  # chat_completion은 이미 문자열 반환

            # JSON 파싱
            import json
            import re

            # JSON 블록 추출
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                inferences = result.get("inferences", [])

                return {
                    inf["index"]: inf["value"]
                    for inf in inferences
                    if "index" in inf and "value" in inf
                }

        except Exception as e:
            logger.error(f"LLM inference parsing failed: {e}")

        return {}

    def _format_rows_for_prompt(self, rows: List[Dict[str, Any]]) -> str:
        """프롬프트용 행 포맷팅"""
        if not rows:
            return "없음"

        lines = []
        for i, row in enumerate(rows):
            row_str = ", ".join(f"{k}: {v}" for k, v in list(row.items())[:6])
            lines.append(f"  {i+1}. {row_str}")

        return "\n".join(lines)

    def _format_missing_rows_for_prompt(
        self,
        missing_rows: List[Dict[str, Any]],
        target_column: str,
    ) -> str:
        """결측 행 프롬프트 포맷팅"""
        if not missing_rows:
            return "없음"

        lines = []
        for item in missing_rows:
            idx = item["index"]
            row = item["row"]
            # 타겟 컬럼 제외한 나머지 컬럼
            other_cols = {k: v for k, v in row.items() if k != target_column}
            row_str = ", ".join(f"{k}: {v}" for k, v in list(other_cols.items())[:6])
            lines.append(f"  행 {idx}: {row_str} -> {target_column}: ???")

        return "\n".join(lines)

    def _record_to_evidence_chain(
        self,
        action: RemediationAction,
        result: RemediationResult,
    ) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ..autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "AUTO_REMEDIATION",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="remediation",
                agent="RemediationEngine",
                evidence_type=evidence_type,
                finding=action.description,
                reasoning=f"Applied {action.strategy.value} strategy",
                conclusion=(
                    f"Modified {result.rows_modified} rows"
                    if result.success
                    else f"Failed: {result.error}"
                ),
                metrics={
                    "action_id": action.action_id,
                    "issue_type": action.issue_type.value,
                    "strategy": action.strategy.value,
                    "rows_modified": result.rows_modified,
                },
                confidence=action.confidence,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_remediation_summary(self) -> Dict[str, Any]:
        """수정 요약"""
        pending = sum(
            1 for a in self.actions.values()
            if a.approval_status == ApprovalStatus.PENDING
        )

        return {
            "stats": self.stats,
            "pending_actions": pending,
            "total_actions": len(self.actions),
            "recent_actions": [
                a.to_dict()
                for a in list(self.actions.values())[-5:]
            ],
        }
