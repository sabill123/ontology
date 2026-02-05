"""
Real-time Reasoning Module (v17.0)

스트리밍 데이터에 대한 즉각적인 LLM 추론:
- 이벤트 트리거 기반 추론
- 실시간 인사이트 생성
- 임계값/패턴/이상치 감지

사용 예시:
    reasoner = StreamingReasoner(context)

    # 트리거 등록
    reasoner.add_trigger(ThresholdTrigger("revenue", ">", 10000))

    # 이벤트 처리
    async for insight in reasoner.process_stream(events):
        print(insight)
"""

from .streaming_reasoner import (
    StreamingReasoner,
    StreamEvent,
    StreamInsight,
)

from .event_triggers import (
    EventTrigger,
    ThresholdTrigger,
    PatternTrigger,
    AnomalyTrigger,
    TriggerResult,
)

__all__ = [
    "StreamingReasoner",
    "StreamEvent",
    "StreamInsight",
    "EventTrigger",
    "ThresholdTrigger",
    "PatternTrigger",
    "AnomalyTrigger",
    "TriggerResult",
]
