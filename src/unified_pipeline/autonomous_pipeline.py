"""
Autonomous Pipeline Orchestrator

Todo 기반 자율 멀티에이전트 파이프라인 오케스트레이터
- 기존 UnifiedPipelineOrchestrator 대체
- Todo 기반 작업 관리
- 에이전트 자율 실행
- 실시간 이벤트 스트리밍
- 자동 도메인 탐지 (DomainDetector 통합)
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, AsyncGenerator, Union

from .shared_context import SharedContext
from .autonomous.orchestrator import AgentOrchestrator, OrchestratorConfig
from .autonomous.continuation import TodoContinuationEnforcer, ContinuationConfig
from .todo.manager import TodoManager
from .todo.events import TodoEventEmitter, get_event_emitter
from .domain import DomainDetector, DomainContext, DEFAULT_DOMAIN_CONTEXT

logger = logging.getLogger(__name__)


class AutonomousPipelineOrchestrator:
    """
    자율 파이프라인 오케스트레이터

    기존 UnifiedPipelineOrchestrator를 대체하는 Todo 기반 자율 시스템

    핵심 특징:
    1. Todo 기반 작업 관리 - DAG 의존성 기반 실행
    2. 자율 에이전트 - 에이전트가 스스로 작업을 가져감
    3. Phase 자연 진행 - 의존성 완료 시 자동으로 다음 Phase
    4. 실시간 이벤트 - FE 모니터링 지원

    사용법:
    ```python
    orchestrator = AutonomousPipelineOrchestrator(
        scenario_name="healthcare_integration",
        domain_context="healthcare",
    )

    # 동기 실행 (이벤트 제너레이터)
    for event in orchestrator.run_sync(tables_data):
        print(event)

    # 비동기 실행
    async for event in orchestrator.run(tables_data):
        print(event)
    ```
    """

    def __init__(
        self,
        scenario_name: str,
        domain_context: Optional[Union[str, DomainContext]] = None,  # Now optional, auto-detected
        llm_client=None,
        output_dir: Optional[str] = None,
        enable_continuation: bool = True,
        max_concurrent_agents: int = 5,
        auto_detect_domain: bool = True,  # NEW: Enable auto domain detection
    ):
        """
        오케스트레이터 초기화

        Args:
            scenario_name: 시나리오 이름
            domain_context: 도메인 컨텍스트 (None이면 자동 탐지)
            llm_client: LLM 클라이언트
            output_dir: 결과 출력 디렉토리
            enable_continuation: Continuation 활성화
            max_concurrent_agents: 최대 동시 에이전트 수
            auto_detect_domain: 도메인 자동 탐지 활성화
        """
        self.scenario_name = scenario_name
        self.llm_client = llm_client
        self.output_dir = Path(output_dir) if output_dir else None
        self.enable_continuation = enable_continuation
        self.auto_detect_domain = auto_detect_domain

        # Domain detector
        self.domain_detector = DomainDetector(
            use_llm=llm_client is not None,
            llm_client=llm_client,
        ) if auto_detect_domain else None

        # Initial domain context (may be updated after data loading)
        if domain_context is None:
            self._initial_domain = DEFAULT_DOMAIN_CONTEXT
        elif isinstance(domain_context, str):
            self._initial_domain = DomainContext(industry=domain_context, is_auto_detected=False)
        else:
            self._initial_domain = domain_context

        # 이벤트 이미터
        self.event_emitter = TodoEventEmitter()

        # 공유 컨텍스트
        self.shared_context = SharedContext(
            scenario_name=scenario_name,
            domain_context=self._initial_domain,
        )

        # v11.0: Evidence Chain 초기화 (블록체인 스타일 근거 관리)
        self.shared_context.initialize_evidence_chain()

        # 에이전트 오케스트레이터
        self.agent_orchestrator = AgentOrchestrator(
            shared_context=self.shared_context,
            config=OrchestratorConfig(
                max_concurrent_agents=max_concurrent_agents,
                enable_parallel=True,
            ),
            llm_client=llm_client,
            event_emitter=self.event_emitter,
        )

        # Continuation Enforcer
        self.continuation_enforcer: Optional[TodoContinuationEnforcer] = None
        if enable_continuation:
            self.continuation_enforcer = TodoContinuationEnforcer(
                todo_manager=self.agent_orchestrator.todo_manager,
                resume_callback=self._resume_todo,
                config=ContinuationConfig(
                    idle_threshold_seconds=5.0,
                    enabled=True,
                ),
                event_emitter=self.event_emitter,
            )

        # 결과 저장
        self._results: Dict[str, Any] = {}

        logger.info(f"AutonomousPipelineOrchestrator initialized: {scenario_name}")

    async def _resume_todo(self, todo) -> None:
        """Todo 재개 콜백"""
        await self.agent_orchestrator._assign_agent_to_todo(todo)

    # === 비동기 실행 ===

    async def run(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        phases: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        비동기 파이프라인 실행

        Args:
            tables_data: 테이블 데이터
            phases: 실행할 Phase 목록

        Yields:
            진행 이벤트
        """
        # Step 0: Auto-detect domain from data (BEFORE pipeline starts)
        if self.auto_detect_domain and self.domain_detector:
            yield {
                "type": "domain_detection_start",
                "message": "Auto-detecting domain from data...",
                "timestamp": datetime.now().isoformat(),
            }

            # First load tables into context for domain detection
            self.agent_orchestrator._load_tables_to_context(tables_data)

            # Detect domain
            try:
                detected_domain = self.domain_detector.detect_domain(self.shared_context)
                self.shared_context.set_domain(detected_domain)

                yield {
                    "type": "domain_detection_complete",
                    "domain": detected_domain.industry,
                    "confidence": detected_domain.industry_confidence,
                    "key_entities": detected_domain.key_entities,
                    "key_metrics": detected_domain.key_metrics,
                    "recommended_workflows": detected_domain.recommended_workflows,
                    "recommended_dashboards": detected_domain.recommended_dashboards,
                    "insight_patterns_count": len(detected_domain.required_insight_patterns),
                    "timestamp": datetime.now().isoformat(),
                }

                logger.info(
                    f"Domain auto-detected: {detected_domain.industry} "
                    f"(confidence: {detected_domain.industry_confidence:.2f})"
                )

            except Exception as e:
                logger.error(f"Domain detection failed: {e}")
                yield {
                    "type": "domain_detection_error",
                    "error": str(e),
                    "fallback_domain": self._initial_domain.industry,
                    "timestamp": datetime.now().isoformat(),
                }

        # Continuation Enforcer 시작
        if self.continuation_enforcer:
            await self.continuation_enforcer.start()

        try:
            # 파이프라인 실행
            async for event in self.agent_orchestrator.run_pipeline(tables_data, phases):
                # 활동 기록
                if self.continuation_enforcer:
                    self.continuation_enforcer.record_activity()

                yield event

        finally:
            # Continuation Enforcer 중지
            if self.continuation_enforcer:
                await self.continuation_enforcer.stop()

            # 결과 저장
            self._save_results()

    # === 동기 실행 (제너레이터) ===

    def run_sync(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        phases: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        동기 파이프라인 실행

        Args:
            tables_data: 테이블 데이터
            phases: 실행할 Phase 목록

        Yields:
            진행 이벤트
        """
        # 이벤트 루프 생성/가져오기
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 비동기 제너레이터를 동기로 변환
        async_gen = self.run(tables_data, phases)

        try:
            while True:
                try:
                    event = loop.run_until_complete(async_gen.__anext__())
                    yield event
                except StopAsyncIteration:
                    break
        finally:
            pass

    # === 단일 Phase 실행 ===

    async def run_phase(
        self,
        phase: str,
        tables_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        단일 Phase 실행

        Args:
            phase: 실행할 Phase
            tables_data: 테이블 데이터 (Discovery 필요)

        Yields:
            진행 이벤트
        """
        # Discovery인 경우 테이블 데이터 로드
        if phase == "discovery" and tables_data:
            self.agent_orchestrator._load_tables_to_context(tables_data)

        async for event in self.agent_orchestrator._run_phase(phase):
            yield event

    # === 결과 저장 ===

    def _save_results(self) -> None:
        """결과 저장"""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 컨텍스트 저장
        context_path = self.output_dir / f"{self.scenario_name}_context.json"
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(self.shared_context.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        # 통계 저장
        stats_path = self.output_dir / f"{self.scenario_name}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                "orchestrator_stats": self.agent_orchestrator.stats.to_dict(),
                "context_stats": self.shared_context.get_statistics(),
                "continuation_stats": self.continuation_enforcer.stats.to_dict()
                    if self.continuation_enforcer else None,
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")

    # === 상태 조회 ===

    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        domain = self.shared_context.get_domain()
        return {
            "scenario_name": self.scenario_name,
            "domain_context": {
                "industry": domain.industry,
                "confidence": domain.industry_confidence,
                "is_auto_detected": domain.is_auto_detected,
                "key_entities": domain.key_entities[:5] if domain.key_entities else [],
                "key_metrics": domain.key_metrics[:5] if domain.key_metrics else [],
                "recommended_workflows": domain.recommended_workflows,
                "recommended_dashboards": domain.recommended_dashboards,
            },
            "orchestrator": self.agent_orchestrator.get_status(),
            "continuation": self.continuation_enforcer.get_status()
                if self.continuation_enforcer else None,
            "context_stats": self.shared_context.get_statistics(),
        }

    def get_context(self) -> SharedContext:
        """공유 컨텍스트 반환"""
        return self.shared_context

    def get_dag_visualization(self) -> str:
        """DAG 시각화"""
        return self.agent_orchestrator.get_dag_visualization()

    # === 이벤트 구독 ===

    def subscribe_events(
        self,
        subscriber_id: str,
        callback,
    ):
        """이벤트 구독"""
        return self.event_emitter.subscribe(
            subscriber_id=subscriber_id,
            callback=callback,
        )

    def unsubscribe_events(self, subscriber_id: str):
        """이벤트 구독 해제"""
        return self.event_emitter.unsubscribe(subscriber_id)


# 기존 UnifiedPipelineOrchestrator 호환 래퍼
class CompatiblePipelineOrchestrator(AutonomousPipelineOrchestrator):
    """
    기존 UnifiedPipelineOrchestrator와 호환되는 래퍼

    기존 코드와의 호환성을 위해 제공
    """

    def run_full_pipeline(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        phases: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        기존 인터페이스 호환 메서드

        Args:
            tables_data: 테이블 데이터
            phases: 실행할 Phase 목록

        Yields:
            진행 이벤트
        """
        for event in self.run_sync(tables_data, phases):
            yield event

    def get_results(self) -> Dict[str, Any]:
        """기존 인터페이스 호환 메서드"""
        return self.shared_context.to_dict()
