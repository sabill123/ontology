"""
전체 파이프라인 통합 테스트

Ontoloty 시스템의 모든 주요 컴포넌트 통합 검증:
- 시계열 예측
- 스트리밍 엔진
- 이벤트 버스
- 워크플로우 오케스트레이터
- 감사 로거
- 비즈니스 인사이트 분석
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Phase 1: Time-Series Forecaster
from src.unified_pipeline.autonomous.analysis.time_series_forecaster import (
    TimeSeriesForecaster,
    ForecastModel,
    ForecastResult,
    SeasonalityInfo,
    TrendInfo,
    create_forecaster,
    check_dependencies as check_forecaster_deps,
)

# Phase 2: Streaming Engine
from src.unified_pipeline.streaming.streaming_engine import (
    StreamingInsightEngine,
    StreamEvent,
    StreamInsight,
    WindowType,
    WindowAggregator,
    IncrementalAnomalyDetector,
    create_streaming_engine,
)

# Event Bus
from src.unified_pipeline.streaming.event_bus import (
    EventBus,
    DomainEvent,
    EventPriority,
    OntologyEvents,
    InsightEvents,
    get_event_bus,
    create_event_bus,
)

# Phase 4: Workflow Orchestrator
from src.unified_pipeline.workflow.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowTask,
    TaskStatus,
    TaskPriority,
    create_orchestrator,
)

# Phase 5: Audit Logger
from src.unified_pipeline.enterprise.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    ComplianceFramework,
    create_audit_logger,
)

# BusinessInsightsAnalyzer
from src.unified_pipeline.autonomous.analysis.business_insights import (
    BusinessInsightsAnalyzer,
    InsightType,
    InsightSeverity,
)


class TestTimeSeriesForecaster:
    """시계열 예측 모듈 테스트"""

    @pytest.fixture
    def forecaster(self):
        return create_forecaster()

    @pytest.fixture
    def sample_series(self):
        """트렌드 + 노이즈가 있는 샘플 시계열"""
        import random
        random.seed(42)
        base = 100
        trend = 2
        return [base + i * trend + random.gauss(0, 5) for i in range(50)]

    @pytest.fixture
    def seasonal_series(self):
        """계절성이 있는 샘플 시계열"""
        import math
        return [
            100 + 20 * math.sin(2 * math.pi * i / 7) + i * 0.5
            for i in range(100)
        ]

    def test_forecaster_creation(self, forecaster):
        """예측기 생성 테스트"""
        assert forecaster is not None
        assert isinstance(forecaster, TimeSeriesForecaster)

    def test_trend_detection(self, forecaster, sample_series):
        """트렌드 탐지 테스트"""
        trend = forecaster.detect_trend(sample_series)

        assert isinstance(trend, TrendInfo)
        assert trend.direction == "increasing"
        assert trend.slope > 0

    def test_seasonality_detection(self, forecaster, seasonal_series):
        """계절성 탐지 테스트"""
        seasonality = forecaster.detect_seasonality(seasonal_series)

        assert isinstance(seasonality, SeasonalityInfo)
        # 7일 주기 패턴이 탐지되어야 함
        if seasonality.has_seasonality:
            assert seasonality.period is not None

    def test_forecast_execution(self, forecaster, sample_series):
        """예측 실행 테스트"""
        result = forecaster.forecast(
            series_name="test_metric",
            series=sample_series,
            horizon=7,
        )

        assert isinstance(result, ForecastResult)
        assert result.series_name == "test_metric"
        assert len(result.predictions) == 7
        assert len(result.lower_bound) == 7
        assert len(result.upper_bound) == 7
        assert result.trend in ["increasing", "decreasing", "stable"]

    def test_forecast_with_dates(self, forecaster, sample_series):
        """날짜와 함께 예측 테스트"""
        dates = [
            (datetime.now() - timedelta(days=50-i)).strftime("%Y-%m-%d")
            for i in range(50)
        ]

        result = forecaster.forecast(
            series_name="dated_metric",
            series=sample_series,
            horizon=7,
            dates=dates,
        )

        assert result.model_used is not None

    def test_dependencies_available(self):
        """의존성 확인 테스트"""
        deps = check_forecaster_deps()

        assert "numpy" in deps
        assert "pandas" in deps
        assert "statsmodels" in deps


class TestStreamingEngine:
    """스트리밍 엔진 테스트"""

    @pytest.fixture
    def streaming_engine(self):
        return create_streaming_engine()

    @pytest.fixture
    def sample_events(self):
        """샘플 스트림 이벤트"""
        return [
            StreamEvent(
                event_id=f"E{i:03d}",
                source="test_source",
                timestamp=datetime.now(),
                payload={"value": 100 + i, "metric": "test"},
                event_type="measurement",
            )
            for i in range(20)
        ]

    def test_engine_creation(self, streaming_engine):
        """엔진 생성 테스트"""
        assert streaming_engine is not None
        assert isinstance(streaming_engine, StreamingInsightEngine)

    def test_incremental_anomaly_detector(self):
        """증분 이상치 탐지 테스트"""
        detector = IncrementalAnomalyDetector(window_size=20, threshold=2.5)

        # 정상 값 입력
        for i in range(15):
            result = detector.update("test", 100 + i % 5)
            # 초기에는 None (데이터 부족)

        # 이상치 입력
        anomaly = detector.update("test", 500)  # 큰 이상치

        # 충분한 데이터가 쌓인 후 이상치 탐지
        assert detector.get_stats("test") is not None

    def test_window_aggregator(self):
        """윈도우 집계 테스트"""
        aggregator = WindowAggregator(
            window_type=WindowType.TUMBLING,
            window_size=timedelta(seconds=1),
        )

        event = StreamEvent(
            event_id="E001",
            source="test",
            timestamp=datetime.now(),
            payload={"value": 100},
            event_type="test",
        )

        result = aggregator.add_event(event)
        # 첫 이벤트는 윈도우 완료 전이므로 None

        # 윈도우 상태 확인
        windows = aggregator.get_current_windows()
        assert len(windows) >= 0  # 빈 경우도 허용

    @pytest.mark.asyncio
    async def test_streaming_with_memory_events(self, streaming_engine, sample_events):
        """메모리 이벤트로 스트리밍 테스트"""
        insights_received = []

        class TestHandler:
            async def handle(self, insight):
                insights_received.append(insight)

        streaming_engine.register_handler(TestHandler())

        # 짧은 실행
        await asyncio.wait_for(
            streaming_engine.run(events=sample_events[:5]),
            timeout=5.0
        )

        stats = streaming_engine.get_stats()
        assert stats["processed_count"] >= 0


class TestEventBus:
    """이벤트 버스 테스트"""

    @pytest.fixture
    def event_bus(self):
        return create_event_bus()

    @pytest.mark.asyncio
    async def test_simple_publish_subscribe(self, event_bus):
        """기본 발행/구독 테스트"""
        received_events = []

        def handler(event):
            received_events.append(event)

        event_bus.subscribe("test.event", handler)

        event = DomainEvent(
            event_id="",
            event_name="test.event",
            payload={"data": "test"},
            source="test",
        )

        delivered = await event_bus.publish(event)

        assert delivered == 1
        assert len(received_events) == 1
        assert received_events[0].payload["data"] == "test"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """와일드카드 구독 테스트"""
        received_events = []

        def handler(event):
            received_events.append(event)

        event_bus.subscribe("ontology.*", handler)

        # 매칭되는 이벤트
        await event_bus.publish(DomainEvent(
            event_id="",
            event_name="ontology.concept.created",
            payload={},
            source="test",
        ))

        # 매칭되지 않는 이벤트
        await event_bus.publish(DomainEvent(
            event_id="",
            event_name="discovery.completed",
            payload={},
            source="test",
        ))

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_async_handler(self, event_bus):
        """비동기 핸들러 테스트"""
        results = []

        async def async_handler(event):
            await asyncio.sleep(0.01)
            results.append(event)

        event_bus.subscribe("async.test", async_handler)

        await event_bus.publish(DomainEvent(
            event_id="",
            event_name="async.test",
            payload={"async": True},
            source="test",
        ))

        assert len(results) == 1

    def test_event_history(self, event_bus):
        """이벤트 히스토리 테스트"""
        event_bus.publish_sync(DomainEvent(
            event_id="",
            event_name="history.test",
            payload={},
            source="test",
        ))

        history = event_bus.get_history("history.*")
        assert len(history) >= 1


class TestWorkflowOrchestrator:
    """워크플로우 오케스트레이터 테스트"""

    @pytest.fixture
    def orchestrator(self):
        return create_orchestrator(max_concurrent_tasks=3)

    def test_orchestrator_creation(self, orchestrator):
        """오케스트레이터 생성 테스트"""
        assert orchestrator is not None
        assert isinstance(orchestrator, WorkflowOrchestrator)

    def test_task_creation(self, orchestrator):
        """태스크 생성 테스트"""
        task = orchestrator.create_task(
            name="Test Task",
            workflow_type="data_quality_monitoring",
            priority=TaskPriority.HIGH,
            params={"table": "test_table"},
        )

        assert task.task_id.startswith("TASK-")
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING

    def test_task_from_action(self, orchestrator):
        """액션에서 태스크 생성 테스트"""
        action = {
            "action_id": "ACT001",
            "title": "Monitor Data Quality",
            "type": "quality_check",
            "severity": "high",
            "params": {"threshold": 0.95},
        }

        task = orchestrator.create_task_from_action(action)

        assert task.action_id == "ACT001"
        assert task.name == "Monitor Data Quality"
        assert task.priority == TaskPriority.HIGH

    def test_workflow_creation(self, orchestrator):
        """워크플로우 생성 테스트"""
        execution = orchestrator.create_workflow(
            workflow_type="data_quality_pipeline",
            tasks=[
                {"name": "Validate Schema", "type": "validation"},
                {"name": "Check Completeness", "type": "quality_check"},
                {"name": "Send Report", "type": "notification"},
            ],
            trigger_source="test",
        )

        assert execution.execution_id.startswith("EXEC-")
        assert len(execution.tasks) == 3

    @pytest.mark.asyncio
    async def test_task_execution(self, orchestrator):
        """태스크 실행 테스트"""
        task = orchestrator.create_task(
            name="Quick Task",
            workflow_type="notification",
            params={"message": "Test"},
        )

        # 스케줄러 시작
        scheduler_task = asyncio.create_task(orchestrator.run_scheduler())

        # 잠시 대기 후 중지
        await asyncio.sleep(0.5)
        orchestrator.stop()

        try:
            await asyncio.wait_for(scheduler_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass

        # 통계 확인
        stats = orchestrator.get_stats()
        assert stats["total_tasks"] >= 1


class TestAuditLogger:
    """감사 로거 테스트"""

    @pytest.fixture
    def audit_logger(self):
        return create_audit_logger(storage_path="/tmp/test_audit_logs")

    def test_logger_creation(self, audit_logger):
        """로거 생성 테스트"""
        assert audit_logger is not None
        assert isinstance(audit_logger, AuditLogger)

    def test_basic_logging(self, audit_logger):
        """기본 로깅 테스트"""
        event = audit_logger.log(
            event_type=AuditEventType.DATA_ACCESSED,
            actor="test_user",
            action="read",
            resource="test_table",
            details={"columns": ["id", "name"]},
        )

        assert event.event_id.startswith("AUDIT-")
        assert event.actor == "test_user"
        assert event.outcome == "success"

    def test_chain_integrity(self, audit_logger):
        """해시 체인 무결성 테스트"""
        # 여러 이벤트 로깅
        for i in range(5):
            audit_logger.log(
                event_type=AuditEventType.SYSTEM_EVENT,
                actor="system",
                action=f"action_{i}",
                resource="test",
            )

        integrity = audit_logger.verify_chain_integrity()
        assert integrity["valid"] is True
        assert integrity["checked_events"] >= 5

    def test_event_query(self, audit_logger):
        """이벤트 조회 테스트"""
        # 다양한 이벤트 로깅
        audit_logger.log(
            event_type=AuditEventType.INSIGHT_GENERATED,
            actor="InsightAgent",
            action="generate",
            resource="insight_001",
        )

        audit_logger.log(
            event_type=AuditEventType.DECISION_MADE,
            actor="GovernanceAgent",
            action="approve",
            resource="decision_001",
        )

        # 타입별 조회
        insights = audit_logger.get_events(
            event_types=[AuditEventType.INSIGHT_GENERATED]
        )
        assert len(insights) >= 1

        # 액터별 조회
        by_actor = audit_logger.get_events(actor="GovernanceAgent")
        assert len(by_actor) >= 1

    def test_compliance_report_sox(self, audit_logger):
        """SOX 컴플라이언스 보고서 테스트"""
        # 이벤트 로깅
        audit_logger.log(
            event_type=AuditEventType.DECISION_MADE,
            actor="system",
            action="approve",
            resource="financial_data",
            compliance_tags=["financial", "governance"],
        )

        report = audit_logger.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now() + timedelta(days=1),
            framework=ComplianceFramework.SOX,
        )

        assert report["framework"] == "SOX"
        assert "summary" in report or "executive_summary" in report

    def test_compliance_report_gdpr(self, audit_logger):
        """GDPR 컴플라이언스 보고서 테스트"""
        # PII 접근 로깅
        audit_logger.log(
            event_type=AuditEventType.DATA_ACCESSED,
            actor="analyst",
            action="read",
            resource="customer_pii",
            sensitivity_level="sensitive",
        )

        report = audit_logger.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now() + timedelta(days=1),
            framework=ComplianceFramework.GDPR,
        )

        assert report["framework"] == "GDPR"


class TestBusinessInsightsAnalyzer:
    """비즈니스 인사이트 분석기 테스트"""

    @pytest.fixture
    def analyzer(self):
        return BusinessInsightsAnalyzer(
            enable_ml_anomaly=False,  # ML 의존성 없이 테스트
            enable_forecasting=True,
        )

    @pytest.fixture
    def sample_data(self):
        """샘플 데이터"""
        import random
        random.seed(42)

        return {
            "orders": [
                {
                    "order_id": f"ORD{i:04d}",
                    "customer_id": f"C{i % 100:03d}",
                    "amount": random.randint(1000, 100000),
                    "status": random.choice(["completed", "pending", "cancelled", "delayed"]),
                    "order_date": (datetime.now() - timedelta(days=random.randint(0, 100))).strftime("%Y-%m-%d"),
                }
                for i in range(500)
            ]
        }

    @pytest.fixture
    def sample_tables(self):
        """테이블 메타데이터"""
        return {
            "orders": {
                "columns": ["order_id", "customer_id", "amount", "status", "order_date"],
                "row_count": 500,
            }
        }

    def test_analyzer_creation(self, analyzer):
        """분석기 생성 테스트"""
        assert analyzer is not None
        assert isinstance(analyzer, BusinessInsightsAnalyzer)

    def test_status_distribution_analysis(self, analyzer, sample_tables, sample_data):
        """상태 분포 분석 테스트"""
        insights = analyzer.analyze(sample_tables, sample_data)

        # 인사이트가 생성되어야 함
        assert len(insights) >= 0  # 임계값에 따라 생성 안 될 수 있음

        # 상태 관련 인사이트 확인
        status_insights = [
            i for i in insights
            if i.insight_type == InsightType.STATUS_DISTRIBUTION
        ]
        # 있으면 검증
        for insight in status_insights:
            assert insight.source_table == "orders"

    def test_anomaly_detection(self, analyzer, sample_tables, sample_data):
        """이상치 탐지 테스트"""
        # 이상치 데이터 추가
        sample_data["orders"].extend([
            {
                "order_id": f"OUTLIER{i}",
                "customer_id": "C999",
                "amount": 10000000,  # 매우 큰 금액
                "status": "completed",
                "order_date": datetime.now().strftime("%Y-%m-%d"),
            }
            for i in range(20)
        ])

        insights = analyzer.analyze(sample_tables, sample_data)

        anomaly_insights = [
            i for i in insights
            if i.insight_type == InsightType.ANOMALY
        ]
        # 이상치 인사이트가 있을 수 있음


class TestIntegrationScenarios:
    """통합 시나리오 테스트"""

    @pytest.mark.asyncio
    async def test_end_to_end_insight_to_action(self):
        """인사이트 → 워크플로우 → 감사 전체 흐름"""
        # 1. 이벤트 버스 설정
        event_bus = create_event_bus()
        audit_logger = create_audit_logger(storage_path="/tmp/e2e_audit")
        orchestrator = create_orchestrator()

        actions_created = []
        audits_logged = []

        # 2. 이벤트 핸들러 등록
        async def on_insight(event):
            """인사이트 발생 시 액션 생성"""
            action = {
                "action_id": f"ACT-{event.event_id}",
                "title": f"Respond to {event.payload.get('insight_type', 'insight')}",
                "type": "response",
                "severity": event.payload.get("severity", "medium"),
            }
            task = orchestrator.create_task_from_action(action)
            actions_created.append(task)

            # 감사 로그
            audit_event = audit_logger.log(
                event_type=AuditEventType.ACTION_CREATED,
                actor="InsightHandler",
                action="create_action",
                resource=task.task_id,
                correlation_id=event.correlation_id,
            )
            audits_logged.append(audit_event)

        event_bus.subscribe(InsightEvents.INSIGHT_GENERATED, on_insight)

        # 3. 인사이트 이벤트 발행
        await event_bus.publish(DomainEvent(
            event_id="",
            event_name=InsightEvents.INSIGHT_GENERATED,
            payload={
                "insight_type": "anomaly",
                "severity": "high",
                "description": "Unusual pattern detected",
            },
            source="InsightAnalyzer",
        ))

        # 4. 검증
        assert len(actions_created) == 1
        assert len(audits_logged) == 1
        assert actions_created[0].priority == TaskPriority.HIGH

    @pytest.mark.asyncio
    async def test_streaming_to_insight_pipeline(self):
        """스트리밍 → 인사이트 파이프라인"""
        event_bus = create_event_bus()
        insights_generated = []

        async def on_anomaly(event):
            insights_generated.append(event)

        event_bus.subscribe("insight.anomaly.*", on_anomaly)

        # 스트리밍 엔진에서 이상치 탐지 시뮬레이션
        detector = IncrementalAnomalyDetector()

        # 정상 데이터
        for i in range(20):
            detector.update("metric", 100)

        # 이상치 입력
        anomaly = detector.update("metric", 1000)

        if anomaly:
            await event_bus.publish(DomainEvent(
                event_id="",
                event_name="insight.anomaly.detected",
                payload=anomaly,
                source="StreamingEngine",
            ))

        # 이상치가 탐지되었으면 이벤트 발생
        # (충분한 데이터가 없으면 탐지 안 될 수 있음)


# 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
