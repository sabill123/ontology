"""
Streaming Reasoner (v17.0)

스트리밍 데이터에 대한 실시간 LLM 추론:
- 이벤트 스트림 처리
- 트리거 기반 추론 실행
- 인사이트 생성 및 알림

사용 예시:
    reasoner = StreamingReasoner(context)

    # 트리거 추가
    reasoner.add_trigger(ThresholdTrigger("revenue", ">", 10000))

    # 이벤트 처리
    async for insight in reasoner.process_stream(events):
        print(f"Insight: {insight.summary}")
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, AsyncIterator, TYPE_CHECKING
from datetime import datetime
from enum import Enum

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .event_triggers import EventTrigger, TriggerResult, TriggerSeverity

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """인사이트 유형"""
    ALERT = "alert"
    TREND = "trend"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"
    SUMMARY = "summary"


@dataclass
class StreamEvent:
    """스트림 이벤트"""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class StreamInsight:
    """스트림 인사이트"""
    insight_id: str
    insight_type: InsightType
    summary: str
    details: str = ""
    severity: TriggerSeverity = TriggerSeverity.INFO

    # 관련 이벤트
    triggered_by: List[str] = field(default_factory=list)  # event_ids
    trigger_results: List[Dict[str, Any]] = field(default_factory=list)

    # 권장 조치
    recommendations: List[str] = field(default_factory=list)

    # 신뢰도
    confidence: float = 0.0

    # 메타데이터
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    generated_by: str = "StreamingReasoner"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type.value,
            "summary": self.summary,
            "details": self.details,
            "severity": self.severity.value,
            "triggered_by": self.triggered_by,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


# DSPy Signatures
if DSPY_AVAILABLE:
    class InsightGenerationSignature(dspy.Signature):
        """인사이트 생성 시그니처"""
        event_data: str = dspy.InputField(
            desc="Triggered event data"
        )
        trigger_info: str = dspy.InputField(
            desc="Trigger that fired and why"
        )
        historical_context: str = dspy.InputField(
            desc="Recent event history and patterns"
        )

        insight_summary: str = dspy.OutputField(
            desc="Concise insight summary (1-2 sentences)"
        )
        insight_details: str = dspy.OutputField(
            desc="Detailed explanation of the insight"
        )
        recommendations: str = dspy.OutputField(
            desc="Comma-separated recommended actions"
        )
        confidence: str = dspy.OutputField(
            desc="Confidence level (0.0-1.0)"
        )


class StreamingReasoner:
    """
    스트리밍 추론 엔진

    기능:
    - 이벤트 스트림 처리
    - 트리거 기반 추론 실행
    - LLM 인사이트 생성
    - Evidence Chain 기록
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        buffer_size: int = 100,
        debounce_ms: int = 1000,
    ):
        """
        Args:
            context: SharedContext
            buffer_size: 이벤트 버퍼 크기
            debounce_ms: 같은 트리거 디바운스 시간 (ms)
        """
        self.context = context
        self.buffer_size = buffer_size
        self.debounce_ms = debounce_ms

        # 트리거
        self.triggers: List[EventTrigger] = []

        # 이벤트 버퍼
        self.event_buffer: List[StreamEvent] = []

        # 인사이트 히스토리
        self.insights: List[StreamInsight] = []

        # DSPy 모듈
        if DSPY_AVAILABLE:
            self.insight_generator = dspy.ChainOfThought(InsightGenerationSignature)

        # 통계
        self.stats = {
            "events_processed": 0,
            "triggers_fired": 0,
            "insights_generated": 0,
        }

        # 디바운스 추적
        self._last_trigger_time: Dict[str, datetime] = {}

        logger.info("StreamingReasoner initialized")

    def add_trigger(self, trigger: EventTrigger) -> None:
        """트리거 추가"""
        self.triggers.append(trigger)
        logger.info(f"Added trigger: {trigger.name}")

    def remove_trigger(self, name: str) -> bool:
        """트리거 제거"""
        for i, t in enumerate(self.triggers):
            if t.name == name:
                self.triggers.pop(i)
                logger.info(f"Removed trigger: {name}")
                return True
        return False

    async def process_event(self, event: StreamEvent) -> List[StreamInsight]:
        """
        단일 이벤트 처리

        Args:
            event: 처리할 이벤트

        Returns:
            생성된 인사이트 목록
        """
        # 버퍼에 추가
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer.pop(0)

        self.stats["events_processed"] += 1

        # 트리거 검사
        triggered_results: List[TriggerResult] = []

        for trigger in self.triggers:
            if not trigger.enabled:
                continue

            result = trigger.check(event.data)

            if result.triggered:
                # 디바운스 확인
                if self._should_debounce(trigger.name):
                    continue

                triggered_results.append(result)
                self._last_trigger_time[trigger.name] = datetime.now()
                self.stats["triggers_fired"] += 1

        # 인사이트 생성
        insights = []
        if triggered_results:
            insight = await self._generate_insight(event, triggered_results)
            if insight:
                insights.append(insight)
                self.insights.append(insight)
                self.stats["insights_generated"] += 1

                # SharedContext 업데이트
                if self.context:
                    self.context.streaming_insights.append(insight.to_dict())

                # Evidence Chain 기록
                self._record_to_evidence_chain(insight)

        return insights

    async def process_stream(
        self,
        events: AsyncIterator[StreamEvent],
    ) -> AsyncIterator[StreamInsight]:
        """
        이벤트 스트림 처리

        Args:
            events: 이벤트 스트림

        Yields:
            생성된 인사이트
        """
        async for event in events:
            insights = await self.process_event(event)
            for insight in insights:
                yield insight

    async def process_batch(
        self,
        events: List[StreamEvent],
    ) -> List[StreamInsight]:
        """
        이벤트 배치 처리

        Args:
            events: 이벤트 목록

        Returns:
            생성된 인사이트 목록
        """
        all_insights = []

        for event in events:
            insights = await self.process_event(event)
            all_insights.extend(insights)

        return all_insights

    def _should_debounce(self, trigger_name: str) -> bool:
        """디바운스 여부 확인"""
        last_time = self._last_trigger_time.get(trigger_name)

        if not last_time:
            return False

        elapsed_ms = (datetime.now() - last_time).total_seconds() * 1000

        return elapsed_ms < self.debounce_ms

    async def _generate_insight(
        self,
        event: StreamEvent,
        trigger_results: List[TriggerResult],
    ) -> Optional[StreamInsight]:
        """LLM 인사이트 생성"""
        import uuid

        # 기본 인사이트 정보
        highest_severity = max(
            (r.severity for r in trigger_results),
            key=lambda s: ["info", "warning", "critical"].index(s.value),
        )

        insight_type = self._determine_insight_type(trigger_results)

        # LLM 생성
        if DSPY_AVAILABLE:
            try:
                # 컨텍스트 준비
                event_data = str(event.to_dict())
                trigger_info = "\n".join(
                    f"- {r.trigger_name}: {r.message}"
                    for r in trigger_results
                )
                historical_context = self._get_historical_context()

                # LLM 호출
                result = self.insight_generator(
                    event_data=event_data,
                    trigger_info=trigger_info,
                    historical_context=historical_context,
                )

                # 결과 파싱
                recommendations = [
                    r.strip()
                    for r in result.recommendations.split(",")
                    if r.strip()
                ]

                try:
                    confidence = float(result.confidence)
                except (ValueError, TypeError):
                    confidence = 0.7

                return StreamInsight(
                    insight_id=f"insight_{uuid.uuid4().hex[:8]}",
                    insight_type=insight_type,
                    summary=result.insight_summary,
                    details=result.insight_details,
                    severity=highest_severity,
                    triggered_by=[event.event_id],
                    trigger_results=[r.to_dict() for r in trigger_results],
                    recommendations=recommendations,
                    confidence=confidence,
                )

            except Exception as e:
                logger.warning(f"LLM insight generation failed: {e}")

        # 폴백: 기본 인사이트
        trigger_messages = [r.message for r in trigger_results if r.message]

        return StreamInsight(
            insight_id=f"insight_{uuid.uuid4().hex[:8]}",
            insight_type=insight_type,
            summary=trigger_messages[0] if trigger_messages else "Event triggered",
            details="\n".join(trigger_messages),
            severity=highest_severity,
            triggered_by=[event.event_id],
            trigger_results=[r.to_dict() for r in trigger_results],
            recommendations=self._generate_default_recommendations(trigger_results),
            confidence=0.6,
        )

    def _determine_insight_type(
        self,
        trigger_results: List[TriggerResult],
    ) -> InsightType:
        """인사이트 유형 결정"""
        # 트리거 이름 기반
        trigger_names = [r.trigger_name.lower() for r in trigger_results]

        if any("anomaly" in n for n in trigger_names):
            return InsightType.ANOMALY
        elif any("trend" in n for n in trigger_names):
            return InsightType.TREND
        elif any(r.severity == TriggerSeverity.CRITICAL for r in trigger_results):
            return InsightType.ALERT
        else:
            return InsightType.RECOMMENDATION

    def _get_historical_context(self) -> str:
        """최근 이벤트 컨텍스트"""
        recent_events = self.event_buffer[-10:]

        if not recent_events:
            return "No recent events"

        context_parts = []
        for event in recent_events:
            context_parts.append(
                f"- [{event.timestamp}] {event.event_type}: {event.data}"
            )

        return "\n".join(context_parts)

    def _generate_default_recommendations(
        self,
        trigger_results: List[TriggerResult],
    ) -> List[str]:
        """기본 권장 조치 생성"""
        recommendations = []

        for result in trigger_results:
            if result.severity == TriggerSeverity.CRITICAL:
                recommendations.append(
                    f"Immediate attention required for {result.field or result.trigger_name}"
                )
            elif result.severity == TriggerSeverity.WARNING:
                recommendations.append(
                    f"Monitor {result.field or result.trigger_name} closely"
                )

        if not recommendations:
            recommendations.append("Continue monitoring")

        return recommendations

    def _record_to_evidence_chain(self, insight: StreamInsight) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ...autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "STREAMING_INSIGHT",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="realtime",
                agent="StreamingReasoner",
                evidence_type=evidence_type,
                finding=insight.summary,
                reasoning=insight.details,
                conclusion=", ".join(insight.recommendations),
                metrics={
                    "insight_id": insight.insight_id,
                    "insight_type": insight.insight_type.value,
                    "severity": insight.severity.value,
                    "triggered_by": insight.triggered_by,
                },
                confidence=insight.confidence,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_recent_insights(
        self,
        limit: int = 10,
        insight_type: Optional[InsightType] = None,
        severity: Optional[TriggerSeverity] = None,
    ) -> List[StreamInsight]:
        """최근 인사이트 조회"""
        insights = self.insights

        if insight_type:
            insights = [i for i in insights if i.insight_type == insight_type]

        if severity:
            insights = [i for i in insights if i.severity == severity]

        return insights[-limit:]

    def get_reasoner_summary(self) -> Dict[str, Any]:
        """추론 엔진 요약"""
        return {
            "stats": self.stats,
            "triggers_count": len(self.triggers),
            "triggers": [
                {
                    "name": t.name,
                    "enabled": t.enabled,
                    "trigger_count": t.trigger_count,
                }
                for t in self.triggers
            ],
            "buffer_size": len(self.event_buffer),
            "recent_insights": [
                i.to_dict() for i in self.insights[-5:]
            ],
        }
