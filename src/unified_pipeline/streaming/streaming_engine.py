"""
Streaming Insight Engine

실시간 인사이트 스트리밍 엔진:
- Kafka/Redis 기반 이벤트 스트리밍
- 윈도우 기반 집계 (Tumbling, Sliding, Session)
- 증분 이상치 탐지
- 이벤트 기반 거버넌스 트리거

References:
- Carbone et al. "Apache Flink" (2015)
- Kreps "I Heart Logs" (2014)
- Akidau et al. "The Dataflow Model" (2015)
"""

import asyncio
import logging
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Union
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Optional imports with availability flags
KAFKA_AVAILABLE = False
REDIS_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    AIOKafkaConsumer = None
    AIOKafkaProducer = None

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None


class WindowType(str, Enum):
    """윈도우 타입"""
    TUMBLING = "tumbling"  # 고정 크기, 겹치지 않음
    SLIDING = "sliding"    # 고정 크기, 겹침
    SESSION = "session"    # 활동 기반, 동적 크기
    GLOBAL = "global"      # 전체 스트림


class AlertSeverity(str, Enum):
    """알림 심각도"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class StreamEvent:
    """스트림 이벤트"""
    event_id: str
    source: str
    timestamp: datetime
    payload: Dict[str, Any]
    event_type: str
    partition_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "event_type": self.event_type,
            "partition_key": self.partition_key,
            "headers": self.headers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamEvent":
        """딕셔너리에서 StreamEvent 생성"""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            source=data.get("source", "unknown"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.now(),
            payload=data.get("payload", {}),
            event_type=data.get("event_type", "unknown"),
            partition_key=data.get("partition_key"),
            headers=data.get("headers", {}),
        )


@dataclass
class StreamInsight:
    """스트림 인사이트"""
    insight_id: str
    title: str
    severity: AlertSeverity
    source_events: List[str]
    timestamp: datetime
    insight_type: str = "anomaly"
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommended_action: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: int = 3600  # Time-to-live

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "title": self.title,
            "severity": self.severity.value,
            "source_events": self.source_events,
            "timestamp": self.timestamp.isoformat(),
            "insight_type": self.insight_type,
            "metrics": self.metrics,
            "recommended_action": self.recommended_action,
            "context": self.context,
        }


@dataclass
class WindowState:
    """윈도우 상태"""
    window_id: str
    window_start: datetime
    window_end: datetime
    event_count: int = 0
    aggregates: Dict[str, float] = field(default_factory=dict)
    events: List[StreamEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """윈도우 지속 시간 (초)"""
        return (self.window_end - self.window_start).total_seconds()

    def is_expired(self, current_time: datetime) -> bool:
        """윈도우 만료 여부"""
        return current_time >= self.window_end


class IncrementalAnomalyDetector:
    """
    증분 이상치 탐지기

    스트리밍 데이터에서 실시간으로 이상치 탐지:
    - Welford's 알고리즘으로 온라인 통계 계산
    - 적응형 임계값 (시간에 따라 조정)
    - 메트릭별 독립 탐지
    """

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 3.0,
        min_samples: int = 10,
        adaptive: bool = True,
    ):
        """
        Args:
            window_size: 슬라이딩 윈도우 크기
            threshold: Z-score 임계값
            min_samples: 탐지 시작 최소 샘플 수
            adaptive: 적응형 임계값 사용 여부
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        self.adaptive = adaptive

        # 메트릭별 상태
        self.values: Dict[str, List[float]] = defaultdict(list)
        self.stats: Dict[str, Dict[str, float]] = {}

        # Welford's 온라인 통계
        self._n: Dict[str, int] = defaultdict(int)
        self._mean: Dict[str, float] = defaultdict(float)
        self._M2: Dict[str, float] = defaultdict(float)

    def update(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """
        값 업데이트 및 이상치 탐지

        Args:
            metric_name: 메트릭 이름
            value: 새 값

        Returns:
            이상치일 경우 상세 정보, 아니면 None
        """
        # 값 저장
        self.values[metric_name].append(value)

        # 윈도우 크기 유지
        if len(self.values[metric_name]) > self.window_size:
            self.values[metric_name].pop(0)

        # Welford's 알고리즘으로 온라인 통계 업데이트
        self._n[metric_name] += 1
        n = self._n[metric_name]
        delta = value - self._mean[metric_name]
        self._mean[metric_name] += delta / n
        delta2 = value - self._mean[metric_name]
        self._M2[metric_name] += delta * delta2

        # 최소 샘플 확인
        if len(self.values[metric_name]) < self.min_samples:
            return None

        # 통계 계산
        values = self.values[metric_name]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5 if variance > 0 else 0.0001

        self.stats[metric_name] = {"mean": mean, "std": std, "variance": variance}

        # Z-score 계산 및 이상치 판단
        if std > 0:
            z_score = abs(value - mean) / std

            # 적응형 임계값
            effective_threshold = self.threshold
            if self.adaptive:
                # 데이터가 많을수록 임계값 약간 상향
                effective_threshold = self.threshold * (1 + 0.1 * (n / 1000))

            if z_score > effective_threshold:
                return {
                    "metric": metric_name,
                    "value": value,
                    "z_score": round(z_score, 3),
                    "mean": round(mean, 3),
                    "std": round(std, 3),
                    "threshold": round(effective_threshold, 3),
                    "is_anomaly": True,
                    "direction": "high" if value > mean else "low",
                    "deviation_pct": round(abs(value - mean) / mean * 100, 1) if mean != 0 else 0,
                }

        return None

    def get_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """메트릭 통계 조회"""
        return self.stats.get(metric_name)

    def reset(self, metric_name: Optional[str] = None):
        """상태 초기화"""
        if metric_name:
            self.values.pop(metric_name, None)
            self.stats.pop(metric_name, None)
            self._n.pop(metric_name, None)
            self._mean.pop(metric_name, None)
            self._M2.pop(metric_name, None)
        else:
            self.values.clear()
            self.stats.clear()
            self._n.clear()
            self._mean.clear()
            self._M2.clear()


class WindowAggregator:
    """
    윈도우 집계기

    이벤트 스트림을 시간 기반 윈도우로 집계:
    - Tumbling: 고정 크기, 겹치지 않는 윈도우
    - Sliding: 고정 크기, 겹치는 윈도우
    - Session: 활동 기반 동적 윈도우
    """

    def __init__(
        self,
        window_type: WindowType = WindowType.TUMBLING,
        window_size: timedelta = timedelta(minutes=5),
        slide_interval: Optional[timedelta] = None,
        session_gap: Optional[timedelta] = None,
    ):
        """
        Args:
            window_type: 윈도우 타입
            window_size: 윈도우 크기
            slide_interval: 슬라이딩 간격 (SLIDING 전용)
            session_gap: 세션 갭 타임아웃 (SESSION 전용)
        """
        self.window_type = window_type
        self.window_size = window_size
        self.slide_interval = slide_interval or window_size
        self.session_gap = session_gap or timedelta(minutes=30)

        self.windows: Dict[str, WindowState] = {}
        self.completed_windows: List[WindowState] = []

    def get_window_key(self, timestamp: datetime, source: str) -> str:
        """윈도우 키 생성"""
        if self.window_type == WindowType.TUMBLING:
            # 윈도우 시작 시간으로 정규화
            window_seconds = int(self.window_size.total_seconds())
            epoch = timestamp.timestamp()
            window_start = int(epoch // window_seconds) * window_seconds
            return f"{source}:{window_start}"

        elif self.window_type == WindowType.SESSION:
            # 세션 윈도우는 source 기반
            return f"session:{source}"

        else:
            # SLIDING, GLOBAL
            return f"{source}:{timestamp.isoformat()}"

    def add_event(self, event: StreamEvent) -> Optional[WindowState]:
        """
        이벤트 추가 및 완료된 윈도우 반환

        Returns:
            완료된 WindowState 또는 None
        """
        key = self.get_window_key(event.timestamp, event.source)

        if key not in self.windows:
            window_start = self._get_window_start(event.timestamp)
            self.windows[key] = WindowState(
                window_id=key,
                window_start=window_start,
                window_end=window_start + self.window_size,
            )

        window = self.windows[key]
        window.event_count += 1
        window.events.append(event)

        # 숫자 값 집계
        for k, v in event.payload.items():
            if isinstance(v, (int, float)):
                if k not in window.aggregates:
                    window.aggregates[k] = 0.0
                window.aggregates[k] += v

        # 윈도우 완료 확인
        if event.timestamp >= window.window_end:
            completed = self.windows.pop(key)
            self.completed_windows.append(completed)
            return completed

        return None

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """윈도우 시작 시간 계산"""
        if self.window_type == WindowType.TUMBLING:
            window_seconds = int(self.window_size.total_seconds())
            epoch = timestamp.timestamp()
            window_start_epoch = int(epoch // window_seconds) * window_seconds
            return datetime.fromtimestamp(window_start_epoch)
        else:
            return timestamp

    def get_current_windows(self) -> Dict[str, WindowState]:
        """현재 활성 윈도우 조회"""
        return self.windows.copy()

    def flush_expired(self, current_time: datetime) -> List[WindowState]:
        """만료된 윈도우 플러시"""
        expired = []
        for key, window in list(self.windows.items()):
            if window.is_expired(current_time):
                expired.append(self.windows.pop(key))
        return expired


class InsightHandler(ABC):
    """인사이트 핸들러 추상 클래스"""

    @abstractmethod
    async def handle(self, insight: StreamInsight) -> None:
        """인사이트 처리"""
        pass


class LoggingHandler(InsightHandler):
    """로깅 핸들러"""

    async def handle(self, insight: StreamInsight) -> None:
        logger.info(f"[{insight.severity.value.upper()}] {insight.title}: {insight.metrics}")


class WebhookHandler(InsightHandler):
    """웹훅 핸들러"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def handle(self, insight: StreamInsight) -> None:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=insight.to_dict(),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Webhook failed: {response.status}")
        except ImportError:
            logger.warning("aiohttp not available for webhook")
        except Exception as e:
            logger.error(f"Webhook error: {e}")


class StreamingInsightEngine:
    """
    스트리밍 인사이트 엔진

    실시간 데이터 스트림에서 인사이트 탐지:
    - Kafka 소비자로 이벤트 수신
    - 윈도우 기반 집계
    - 증분 이상치 탐지
    - 거버넌스 트리거
    """

    def __init__(
        self,
        kafka_bootstrap: str = "localhost:9092",
        redis_url: str = "redis://localhost:6379",
        domain_context=None,
        window_type: WindowType = WindowType.TUMBLING,
        window_size: timedelta = timedelta(minutes=5),
    ):
        """
        Args:
            kafka_bootstrap: Kafka 브로커 주소
            redis_url: Redis 주소
            domain_context: DomainContext 객체
            window_type: 윈도우 타입
            window_size: 윈도우 크기
        """
        self.kafka_bootstrap = kafka_bootstrap
        self.redis_url = redis_url
        self.domain_context = domain_context

        # 컴포넌트 초기화
        self.anomaly_detector = IncrementalAnomalyDetector()
        self.window_aggregator = WindowAggregator(
            window_type=window_type,
            window_size=window_size,
        )

        # 핸들러
        self.insight_handlers: List[InsightHandler] = []

        # 상태
        self.running = False
        self.processed_count = 0
        self.insight_count = 0

        # Redis 클라이언트 (지연 초기화)
        self._redis = None

        # 도메인별 규칙
        self._domain_rules: List[Callable[[StreamEvent], Optional[StreamInsight]]] = []

    def register_handler(self, handler: InsightHandler):
        """인사이트 핸들러 등록"""
        self.insight_handlers.append(handler)

    def register_domain_rule(
        self,
        rule: Callable[[StreamEvent], Optional[StreamInsight]],
    ):
        """도메인 규칙 등록"""
        self._domain_rules.append(rule)

    async def _init_redis(self):
        """Redis 연결 초기화"""
        if REDIS_AVAILABLE and self._redis is None:
            try:
                self._redis = await aioredis.from_url(self.redis_url)
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")

    async def _cache_insight(self, insight: StreamInsight):
        """인사이트 Redis 캐시"""
        if self._redis:
            try:
                await self._redis.setex(
                    f"insight:{insight.insight_id}",
                    insight.ttl_seconds,
                    json.dumps(insight.to_dict()),
                )
            except Exception as e:
                logger.debug(f"Redis cache failed: {e}")

    async def _process_event(self, event: StreamEvent) -> List[StreamInsight]:
        """이벤트 처리"""
        insights = []
        self.processed_count += 1

        # 1. 증분 이상치 탐지
        for key, value in event.payload.items():
            if isinstance(value, (int, float)):
                metric_name = f"{event.source}.{key}"
                anomaly = self.anomaly_detector.update(metric_name, value)

                if anomaly:
                    insight = StreamInsight(
                        insight_id=f"STREAM-{uuid.uuid4().hex[:8]}",
                        title=f"Anomaly Detected: {key}",
                        severity=AlertSeverity.HIGH if anomaly["z_score"] > 4 else AlertSeverity.MEDIUM,
                        source_events=[event.event_id],
                        timestamp=event.timestamp,
                        insight_type="anomaly",
                        metrics=anomaly,
                        recommended_action="Investigate anomalous value immediately",
                        context={
                            "source": event.source,
                            "event_type": event.event_type,
                        },
                    )
                    insights.append(insight)

        # 2. 윈도우 집계
        completed_window = self.window_aggregator.add_event(event)
        if completed_window:
            window_insights = self._analyze_window(completed_window)
            insights.extend(window_insights)

        # 3. 도메인 규칙 적용
        for rule in self._domain_rules:
            try:
                rule_insight = rule(event)
                if rule_insight:
                    insights.append(rule_insight)
            except Exception as e:
                logger.debug(f"Domain rule failed: {e}")

        # 4. 도메인 컨텍스트 기반 분석
        if self.domain_context:
            domain_insights = self._apply_domain_context(event)
            insights.extend(domain_insights)

        return insights

    def _analyze_window(self, window: WindowState) -> List[StreamInsight]:
        """완료된 윈도우 분석"""
        insights = []

        # 이벤트 빈도 분석
        if window.event_count > 0:
            events_per_second = window.event_count / window.duration_seconds

            # 높은 이벤트 빈도 경고
            if events_per_second > 100:  # 임계값
                insight = StreamInsight(
                    insight_id=f"WINDOW-{uuid.uuid4().hex[:8]}",
                    title="High Event Rate Detected",
                    severity=AlertSeverity.MEDIUM,
                    source_events=[e.event_id for e in window.events[:10]],
                    timestamp=window.window_end,
                    insight_type="traffic",
                    metrics={
                        "events_per_second": round(events_per_second, 2),
                        "total_events": window.event_count,
                        "window_duration_seconds": window.duration_seconds,
                    },
                    recommended_action="Review system capacity",
                )
                insights.append(insight)

        # 집계값 분석
        for metric, total in window.aggregates.items():
            avg_value = total / window.event_count if window.event_count > 0 else 0

            # 이상 집계값 탐지 (도메인별 임계값 적용 가능)
            if self.domain_context:
                threshold = self._get_domain_threshold(metric)
                if threshold and avg_value > threshold:
                    insight = StreamInsight(
                        insight_id=f"AGG-{uuid.uuid4().hex[:8]}",
                        title=f"Aggregate Threshold Exceeded: {metric}",
                        severity=AlertSeverity.HIGH,
                        source_events=[e.event_id for e in window.events[:5]],
                        timestamp=window.window_end,
                        insight_type="threshold",
                        metrics={
                            "metric": metric,
                            "average_value": round(avg_value, 2),
                            "threshold": threshold,
                            "total": round(total, 2),
                        },
                        recommended_action=f"Investigate high {metric} values",
                    )
                    insights.append(insight)

        return insights

    def _apply_domain_context(self, event: StreamEvent) -> List[StreamInsight]:
        """도메인 컨텍스트 기반 분석"""
        insights = []

        if not self.domain_context:
            return insights

        # 도메인 KPI 확인
        kpis = getattr(self.domain_context, 'kpis', [])
        for kpi in kpis:
            kpi_name = kpi.get("name", "").lower()
            threshold = kpi.get("threshold")
            direction = kpi.get("direction", "higher_is_better")

            # 이벤트 payload에서 KPI 값 확인
            for key, value in event.payload.items():
                if kpi_name in key.lower() and isinstance(value, (int, float)):
                    is_violation = False
                    if direction == "higher_is_better" and value < threshold:
                        is_violation = True
                    elif direction == "lower_is_better" and value > threshold:
                        is_violation = True

                    if is_violation:
                        insight = StreamInsight(
                            insight_id=f"KPI-{uuid.uuid4().hex[:8]}",
                            title=f"KPI Violation: {kpi.get('name', key)}",
                            severity=AlertSeverity.HIGH,
                            source_events=[event.event_id],
                            timestamp=event.timestamp,
                            insight_type="kpi_violation",
                            metrics={
                                "kpi": kpi.get("name", key),
                                "value": value,
                                "threshold": threshold,
                                "direction": direction,
                            },
                            recommended_action=kpi.get("action", "Review KPI performance"),
                        )
                        insights.append(insight)

        return insights

    def _get_domain_threshold(self, metric: str) -> Optional[float]:
        """도메인별 메트릭 임계값 조회"""
        if not self.domain_context:
            return None

        # 도메인 컨텍스트에서 임계값 조회
        thresholds = getattr(self.domain_context, 'metric_thresholds', {})
        return thresholds.get(metric)

    async def consume_kafka(self, topics: List[str]) -> AsyncGenerator[StreamEvent, None]:
        """Kafka 이벤트 소비"""
        if not KAFKA_AVAILABLE:
            raise ImportError("aiokafka required for Kafka streaming")

        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self.kafka_bootstrap,
            group_id="ontoloty-streaming",
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
        )

        await consumer.start()
        logger.info(f"Kafka consumer started for topics: {topics}")

        try:
            async for msg in consumer:
                event = StreamEvent(
                    event_id=f"{msg.topic}-{msg.partition}-{msg.offset}",
                    source=msg.topic,
                    timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                    payload=msg.value if isinstance(msg.value, dict) else {"value": msg.value},
                    event_type="kafka",
                    partition_key=str(msg.key) if msg.key else None,
                )
                yield event
        finally:
            await consumer.stop()

    async def consume_memory(
        self,
        events: List[Union[StreamEvent, Dict[str, Any]]],
    ) -> AsyncGenerator[StreamEvent, None]:
        """메모리 기반 이벤트 소비 (테스트용)"""
        for event in events:
            if isinstance(event, dict):
                event = StreamEvent.from_dict(event)
            yield event
            await asyncio.sleep(0.001)  # 비동기 처리 허용

    async def run(
        self,
        topics: Optional[List[str]] = None,
        events: Optional[List[Union[StreamEvent, Dict]]] = None,
    ):
        """
        스트리밍 엔진 실행

        Args:
            topics: Kafka 토픽 리스트 (Kafka 모드)
            events: 이벤트 리스트 (메모리 모드, 테스트용)
        """
        self.running = True
        await self._init_redis()

        logger.info("Starting streaming engine")

        # 이벤트 소스 선택
        if events is not None:
            event_source = self.consume_memory(events)
        elif topics and KAFKA_AVAILABLE:
            event_source = self.consume_kafka(topics)
        else:
            raise ValueError("Either topics or events must be provided")

        async for event in event_source:
            if not self.running:
                break

            try:
                insights = await self._process_event(event)

                for insight in insights:
                    self.insight_count += 1

                    # 캐시
                    await self._cache_insight(insight)

                    # 핸들러 호출
                    for handler in self.insight_handlers:
                        try:
                            await handler.handle(insight)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")

            except Exception as e:
                logger.error(f"Event processing error: {e}")

        logger.info(f"Streaming engine stopped. Processed: {self.processed_count}, Insights: {self.insight_count}")

    def stop(self):
        """스트리밍 중지"""
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """엔진 통계 조회"""
        return {
            "running": self.running,
            "processed_count": self.processed_count,
            "insight_count": self.insight_count,
            "active_windows": len(self.window_aggregator.windows),
            "anomaly_metrics": len(self.anomaly_detector.stats),
        }


# Factory functions
def create_streaming_engine(
    kafka_bootstrap: str = "localhost:9092",
    domain_context=None,
    window_size_minutes: int = 5,
) -> StreamingInsightEngine:
    """팩토리 함수"""
    return StreamingInsightEngine(
        kafka_bootstrap=kafka_bootstrap,
        domain_context=domain_context,
        window_size=timedelta(minutes=window_size_minutes),
    )


def check_dependencies() -> Dict[str, bool]:
    """의존성 상태 확인"""
    return {
        "kafka": KAFKA_AVAILABLE,
        "redis": REDIS_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
    }
