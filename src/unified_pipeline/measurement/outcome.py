"""
Outcome Measurement - 의사결정 결과 측정

팔란티어 검증 기준:
✅ 의사결정 결과가 측정되는가?
✅ Object → Action → Outcome 흐름 추적
✅ Feedback loop으로 개선

컴포넌트:
- DecisionRecord: 의사결정 기록 (어떤 객체에 어떤 액션을 왜 수행했는가)
- OutcomeRecord: 결과 기록 (예상 vs 실제)
- OutcomeTracker: 결과 추적기
- OutcomeMetrics: 메트릭 집계
- FeedbackLoop: 개선 피드백 루프
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json
import statistics


class DecisionType(str, Enum):
    """의사결정 유형"""
    BEHAVIOR_CALL = "behavior_call"      # 객체 행동 호출
    STATE_TRANSITION = "state_transition" # 상태 전이
    QUERY_RESULT = "query_result"        # 쿼리 기반 결정
    LLM_REASONING = "llm_reasoning"      # LLM reasoning 결과
    BATCH_OPERATION = "batch_operation"  # 배치 작업


class OutcomeStatus(str, Enum):
    """결과 상태"""
    PENDING = "pending"          # 결과 대기 중
    SUCCESS = "success"          # 예상대로 성공
    PARTIAL = "partial"          # 부분 성공
    FAILURE = "failure"          # 실패
    UNEXPECTED = "unexpected"    # 예상치 못한 결과
    TIMEOUT = "timeout"          # 시간 초과


@dataclass
class DecisionRecord:
    """
    의사결정 기록

    팔란티어 패턴:
    - 어떤 객체(Object)에
    - 어떤 액션(Action)을
    - 왜(Reason) 수행했는가
    """
    decision_id: str
    decision_type: DecisionType

    # Object 정보
    object_type: str
    object_id: str
    object_state_before: Dict[str, Any]

    # Action 정보
    action_name: str
    action_parameters: Dict[str, Any]

    # Context
    agent_id: str
    reason: Optional[str] = None
    expected_outcome: Optional[str] = None

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "object_state_before": self.object_state_before,
            "action_name": self.action_name,
            "action_parameters": self.action_parameters,
            "agent_id": self.agent_id,
            "reason": self.reason,
            "expected_outcome": self.expected_outcome,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


@dataclass
class OutcomeRecord:
    """
    결과 기록

    팔란티어 패턴:
    - 예상(Expected)과 실제(Actual) 비교
    - 결과 기반 피드백
    """
    outcome_id: str
    decision_id: str  # 관련 의사결정

    # 결과 정보
    status: OutcomeStatus
    actual_outcome: Dict[str, Any]
    object_state_after: Dict[str, Any]

    # 예상 vs 실제
    expected_outcome: Optional[str] = None
    deviation: Optional[str] = None  # 예상과의 차이
    deviation_score: float = 0.0     # 0.0 (일치) ~ 1.0 (완전 불일치)

    # 성과 지표
    execution_time_ms: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    # 메타데이터
    recorded_at: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.status in (OutcomeStatus.SUCCESS, OutcomeStatus.PARTIAL)

    @property
    def matches_expectation(self) -> bool:
        return self.deviation_score < 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "decision_id": self.decision_id,
            "status": self.status.value,
            "actual_outcome": self.actual_outcome,
            "object_state_after": self.object_state_after,
            "expected_outcome": self.expected_outcome,
            "deviation": self.deviation,
            "deviation_score": self.deviation_score,
            "execution_time_ms": self.execution_time_ms,
            "resource_usage": self.resource_usage,
            "recorded_at": self.recorded_at.isoformat(),
            "notes": self.notes,
            "is_successful": self.is_successful,
            "matches_expectation": self.matches_expectation,
        }


@dataclass
class OutcomeMetrics:
    """결과 메트릭 집계"""

    # 기본 통계
    total_decisions: int = 0
    successful_outcomes: int = 0
    failed_outcomes: int = 0
    pending_outcomes: int = 0

    # 성공률
    success_rate: float = 0.0

    # 예상 적중률
    expectation_match_rate: float = 0.0

    # 결정 유형별 통계
    by_decision_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 객체 유형별 통계
    by_object_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 에이전트별 통계
    by_agent: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 시간대별 통계
    by_hour: Dict[int, int] = field(default_factory=dict)

    # 성능 통계
    avg_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    p99_execution_time_ms: float = 0.0

    # 편차 통계
    avg_deviation_score: float = 0.0
    max_deviation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "successful_outcomes": self.successful_outcomes,
            "failed_outcomes": self.failed_outcomes,
            "pending_outcomes": self.pending_outcomes,
            "success_rate": round(self.success_rate, 4),
            "expectation_match_rate": round(self.expectation_match_rate, 4),
            "by_decision_type": self.by_decision_type,
            "by_object_type": self.by_object_type,
            "by_agent": self.by_agent,
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "p95_execution_time_ms": round(self.p95_execution_time_ms, 2),
            "p99_execution_time_ms": round(self.p99_execution_time_ms, 2),
            "avg_deviation_score": round(self.avg_deviation_score, 4),
            "max_deviation_score": round(self.max_deviation_score, 4),
        }


class OutcomeTracker:
    """
    결과 추적기

    팔란티어 검증:
    ✅ 의사결정 결과가 측정됨
    ✅ Object → Action → Outcome 흐름 추적
    ✅ 시계열 분석 가능
    """

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days

        # 저장소
        self._decisions: Dict[str, DecisionRecord] = {}
        self._outcomes: Dict[str, OutcomeRecord] = {}

        # 인덱스 (빠른 조회용)
        self._by_object: Dict[str, List[str]] = defaultdict(list)  # object_key -> decision_ids
        self._by_agent: Dict[str, List[str]] = defaultdict(list)   # agent_id -> decision_ids
        self._by_type: Dict[str, List[str]] = defaultdict(list)    # decision_type -> decision_ids
        self._pending: List[str] = []  # 결과 대기 중인 decision_ids

        # 콜백
        self._on_decision: List[Callable[[DecisionRecord], None]] = []
        self._on_outcome: List[Callable[[OutcomeRecord], None]] = []

    # =========================================================================
    # 의사결정 기록
    # =========================================================================

    def record_decision(
        self,
        decision_type: DecisionType,
        object_type: str,
        object_id: str,
        object_state: Dict[str, Any],
        action_name: str,
        action_parameters: Dict[str, Any],
        agent_id: str,
        reason: Optional[str] = None,
        expected_outcome: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> DecisionRecord:
        """
        의사결정 기록

        예시:
            decision = tracker.record_decision(
                decision_type=DecisionType.BEHAVIOR_CALL,
                object_type="Order",
                object_id="ORD-123",
                object_state={"status": "pending", "amount": 1500},
                action_name="evaluateRisk",
                action_parameters={},
                agent_id="discovery-agent",
                reason="고액 주문이므로 위험 평가 필요",
                expected_outcome="risk_score < 0.5"
            )
        """
        decision_id = f"DEC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        record = DecisionRecord(
            decision_id=decision_id,
            decision_type=decision_type,
            object_type=object_type,
            object_id=object_id,
            object_state_before=object_state.copy(),
            action_name=action_name,
            action_parameters=action_parameters,
            agent_id=agent_id,
            reason=reason,
            expected_outcome=expected_outcome,
            tags=tags or []
        )

        # 저장
        self._decisions[decision_id] = record

        # 인덱스 업데이트
        object_key = f"{object_type}:{object_id}"
        self._by_object[object_key].append(decision_id)
        self._by_agent[agent_id].append(decision_id)
        self._by_type[decision_type.value].append(decision_id)
        self._pending.append(decision_id)

        # 콜백 실행
        for callback in self._on_decision:
            try:
                callback(record)
            except Exception:
                pass

        return record

    # =========================================================================
    # 결과 기록
    # =========================================================================

    def record_outcome(
        self,
        decision_id: str,
        status: OutcomeStatus,
        actual_outcome: Dict[str, Any],
        object_state_after: Dict[str, Any],
        execution_time_ms: int = 0,
        deviation: Optional[str] = None,
        deviation_score: float = 0.0,
        notes: Optional[str] = None
    ) -> OutcomeRecord:
        """
        결과 기록

        예시:
            outcome = tracker.record_outcome(
                decision_id=decision.decision_id,
                status=OutcomeStatus.SUCCESS,
                actual_outcome={"risk_score": 0.23},
                object_state_after={"status": "approved", "amount": 1500},
                execution_time_ms=150,
                deviation_score=0.0  # 예상과 일치
            )
        """
        if decision_id not in self._decisions:
            raise ValueError(f"Decision {decision_id} not found")

        decision = self._decisions[decision_id]
        outcome_id = f"OUT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        record = OutcomeRecord(
            outcome_id=outcome_id,
            decision_id=decision_id,
            status=status,
            actual_outcome=actual_outcome,
            object_state_after=object_state_after,
            expected_outcome=decision.expected_outcome,
            deviation=deviation,
            deviation_score=deviation_score,
            execution_time_ms=execution_time_ms,
            notes=notes
        )

        # 저장
        self._outcomes[outcome_id] = record

        # pending에서 제거
        if decision_id in self._pending:
            self._pending.remove(decision_id)

        # 콜백 실행
        for callback in self._on_outcome:
            try:
                callback(record)
            except Exception:
                pass

        return record

    # =========================================================================
    # 조회
    # =========================================================================

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """의사결정 조회"""
        return self._decisions.get(decision_id)

    def get_outcome(self, decision_id: str) -> Optional[OutcomeRecord]:
        """결과 조회 (decision_id로)"""
        for outcome in self._outcomes.values():
            if outcome.decision_id == decision_id:
                return outcome
        return None

    def get_decisions_for_object(
        self,
        object_type: str,
        object_id: str,
        limit: int = 100
    ) -> List[DecisionRecord]:
        """객체의 의사결정 이력"""
        object_key = f"{object_type}:{object_id}"
        decision_ids = self._by_object.get(object_key, [])
        return [
            self._decisions[did]
            for did in decision_ids[-limit:]
            if did in self._decisions
        ]

    def get_decisions_by_agent(
        self,
        agent_id: str,
        limit: int = 100
    ) -> List[DecisionRecord]:
        """에이전트의 의사결정 이력"""
        decision_ids = self._by_agent.get(agent_id, [])
        return [
            self._decisions[did]
            for did in decision_ids[-limit:]
            if did in self._decisions
        ]

    def get_pending_decisions(self) -> List[DecisionRecord]:
        """결과 대기 중인 의사결정"""
        return [
            self._decisions[did]
            for did in self._pending
            if did in self._decisions
        ]

    # =========================================================================
    # 메트릭 집계
    # =========================================================================

    def compute_metrics(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        object_type: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> OutcomeMetrics:
        """
        메트릭 집계

        예시:
            # 최근 24시간 메트릭
            metrics = tracker.compute_metrics(
                since=datetime.now() - timedelta(hours=24)
            )

            # 특정 에이전트 메트릭
            metrics = tracker.compute_metrics(agent_id="discovery-agent")
        """
        metrics = OutcomeMetrics()

        # 필터링된 결정/결과 수집
        filtered_decisions = []
        filtered_outcomes = []
        execution_times = []
        deviation_scores = []

        for decision in self._decisions.values():
            # 시간 필터
            if since and decision.created_at < since:
                continue
            if until and decision.created_at > until:
                continue

            # 객체 타입 필터
            if object_type and decision.object_type != object_type:
                continue

            # 에이전트 필터
            if agent_id and decision.agent_id != agent_id:
                continue

            filtered_decisions.append(decision)

            # 관련 outcome 찾기
            outcome = self.get_outcome(decision.decision_id)
            if outcome:
                filtered_outcomes.append(outcome)
                execution_times.append(outcome.execution_time_ms)
                deviation_scores.append(outcome.deviation_score)

        # 기본 통계
        metrics.total_decisions = len(filtered_decisions)
        metrics.successful_outcomes = sum(
            1 for o in filtered_outcomes if o.is_successful
        )
        metrics.failed_outcomes = sum(
            1 for o in filtered_outcomes
            if o.status in (OutcomeStatus.FAILURE, OutcomeStatus.TIMEOUT)
        )
        metrics.pending_outcomes = metrics.total_decisions - len(filtered_outcomes)

        # 성공률
        if filtered_outcomes:
            metrics.success_rate = metrics.successful_outcomes / len(filtered_outcomes)

            # 예상 적중률
            matches = sum(1 for o in filtered_outcomes if o.matches_expectation)
            metrics.expectation_match_rate = matches / len(filtered_outcomes)

        # 결정 유형별 통계
        by_type = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
        for decision in filtered_decisions:
            dt = decision.decision_type.value
            by_type[dt]["total"] += 1

            outcome = self.get_outcome(decision.decision_id)
            if outcome:
                if outcome.is_successful:
                    by_type[dt]["success"] += 1
                elif outcome.status == OutcomeStatus.FAILURE:
                    by_type[dt]["failure"] += 1
        metrics.by_decision_type = dict(by_type)

        # 객체 유형별 통계
        by_object = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
        for decision in filtered_decisions:
            ot = decision.object_type
            by_object[ot]["total"] += 1

            outcome = self.get_outcome(decision.decision_id)
            if outcome:
                if outcome.is_successful:
                    by_object[ot]["success"] += 1
                elif outcome.status == OutcomeStatus.FAILURE:
                    by_object[ot]["failure"] += 1
        metrics.by_object_type = dict(by_object)

        # 에이전트별 통계
        by_agent = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
        for decision in filtered_decisions:
            aid = decision.agent_id
            by_agent[aid]["total"] += 1

            outcome = self.get_outcome(decision.decision_id)
            if outcome:
                if outcome.is_successful:
                    by_agent[aid]["success"] += 1
                elif outcome.status == OutcomeStatus.FAILURE:
                    by_agent[aid]["failure"] += 1
        metrics.by_agent = dict(by_agent)

        # 성능 통계
        if execution_times:
            sorted_times = sorted(execution_times)
            metrics.avg_execution_time_ms = statistics.mean(execution_times)
            metrics.p95_execution_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
            metrics.p99_execution_time_ms = sorted_times[int(len(sorted_times) * 0.99)]

        # 편차 통계
        if deviation_scores:
            metrics.avg_deviation_score = statistics.mean(deviation_scores)
            metrics.max_deviation_score = max(deviation_scores)

        return metrics

    # =========================================================================
    # 콜백 등록
    # =========================================================================

    def on_decision(self, callback: Callable[[DecisionRecord], None]):
        """의사결정 기록 시 콜백"""
        self._on_decision.append(callback)

    def on_outcome(self, callback: Callable[[OutcomeRecord], None]):
        """결과 기록 시 콜백"""
        self._on_outcome.append(callback)

    # =========================================================================
    # 정리
    # =========================================================================

    def cleanup_old_records(self):
        """오래된 레코드 정리"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        # 오래된 결정 ID 수집
        old_decision_ids = [
            did for did, decision in self._decisions.items()
            if decision.created_at < cutoff
        ]

        # 삭제
        for did in old_decision_ids:
            decision = self._decisions.pop(did, None)
            if decision:
                # 인덱스에서 제거
                object_key = f"{decision.object_type}:{decision.object_id}"
                if did in self._by_object.get(object_key, []):
                    self._by_object[object_key].remove(did)
                if did in self._by_agent.get(decision.agent_id, []):
                    self._by_agent[decision.agent_id].remove(did)
                if did in self._by_type.get(decision.decision_type.value, []):
                    self._by_type[decision.decision_type.value].remove(did)
                if did in self._pending:
                    self._pending.remove(did)

        # 관련 outcome 삭제
        old_outcome_ids = [
            oid for oid, outcome in self._outcomes.items()
            if outcome.decision_id in old_decision_ids
        ]
        for oid in old_outcome_ids:
            self._outcomes.pop(oid, None)

        return len(old_decision_ids)


class FeedbackLoop:
    """
    피드백 루프

    팔란티어 패턴:
    - 결과 분석 → 개선점 도출 → 행동 조정
    - 지속적 학습 및 개선
    """

    def __init__(self, tracker: OutcomeTracker):
        self.tracker = tracker
        self._improvement_suggestions: List[Dict[str, Any]] = []

    def analyze_failures(
        self,
        since: Optional[datetime] = None,
        min_failures: int = 3
    ) -> List[Dict[str, Any]]:
        """
        실패 패턴 분석

        반환:
            [
                {
                    "pattern": "Order.evaluateRisk 행동이 높은 실패율",
                    "failure_rate": 0.35,
                    "sample_decisions": [...],
                    "suggested_improvement": "위험 평가 임계값 조정 필요"
                }
            ]
        """
        metrics = self.tracker.compute_metrics(since=since)
        suggestions = []

        # 객체 유형별 분석
        for object_type, stats in metrics.by_object_type.items():
            total = stats["total"]
            failures = stats["failure"]

            if total >= min_failures and failures / total > 0.2:
                suggestions.append({
                    "pattern": f"{object_type} 관련 행동의 높은 실패율",
                    "failure_rate": failures / total,
                    "total_decisions": total,
                    "failures": failures,
                    "suggested_improvement": f"{object_type} 행동 로직 검토 필요"
                })

        # 에이전트별 분석
        for agent_id, stats in metrics.by_agent.items():
            total = stats["total"]
            failures = stats["failure"]

            if total >= min_failures and failures / total > 0.2:
                suggestions.append({
                    "pattern": f"{agent_id} 에이전트의 높은 실패율",
                    "failure_rate": failures / total,
                    "total_decisions": total,
                    "failures": failures,
                    "suggested_improvement": f"{agent_id} 에이전트 reasoning 로직 검토 필요"
                })

        # 편차 분석
        if metrics.avg_deviation_score > 0.3:
            suggestions.append({
                "pattern": "예상과 실제 결과의 높은 편차",
                "avg_deviation": metrics.avg_deviation_score,
                "max_deviation": metrics.max_deviation_score,
                "suggested_improvement": "예상 결과 설정 로직 개선 필요"
            })

        self._improvement_suggestions = suggestions
        return suggestions

    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """개선 제안 반환"""
        return self._improvement_suggestions

    def generate_report(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        종합 보고서 생성

        팔란티어 검증:
        ✅ 의사결정 결과 측정 보고서
        """
        metrics = self.tracker.compute_metrics(since=since)
        failures = self.analyze_failures(since=since)

        return {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "since": since.isoformat() if since else None,
                "until": datetime.now().isoformat()
            },
            "summary": {
                "total_decisions": metrics.total_decisions,
                "success_rate": f"{metrics.success_rate * 100:.1f}%",
                "expectation_match_rate": f"{metrics.expectation_match_rate * 100:.1f}%",
                "pending_outcomes": metrics.pending_outcomes
            },
            "performance": {
                "avg_execution_time_ms": metrics.avg_execution_time_ms,
                "p95_execution_time_ms": metrics.p95_execution_time_ms,
                "p99_execution_time_ms": metrics.p99_execution_time_ms
            },
            "by_object_type": metrics.by_object_type,
            "by_agent": metrics.by_agent,
            "by_decision_type": metrics.by_decision_type,
            "improvement_suggestions": failures,
            "palantir_validation": {
                "outcome_measurement": True,
                "feedback_loop": True,
                "decision_tracking": True
            }
        }


# 전역 Tracker 싱글톤
GlobalOutcomeTracker = OutcomeTracker()
