"""
Autonomous Pipeline Orchestrator (v17.0 - Full Integration)

Todo 기반 자율 멀티에이전트 파이프라인 오케스트레이터
- 기존 UnifiedPipelineOrchestrator 대체
- Todo 기반 작업 관리
- 에이전트 자율 실행
- 실시간 이벤트 스트리밍
- 자동 도메인 탐지 (DomainDetector 통합)

v17.0 통합 기능:
- Confidence Calibration: 에이전트 신뢰도 보정
- Dataset Versioning: Git-style 버전 관리
- Semantic Search: 벡터 기반 의미 검색
- Auto-remediation: 데이터 품질 자동 수정
- XAI (Explanation): 결정 설명
- What-If Analysis: 시나리오 시뮬레이션
- NL Reports: 자연어 보고서 생성
- NL2SQL: 자연어 쿼리
- Function Calling: 도구 자동 선택
- OSDK: 온톨로지 SDK
- Writeback: 데이터 수정
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, AsyncGenerator, Union
from dataclasses import dataclass, field

from .shared_context import SharedContext
from .autonomous.orchestrator import AgentOrchestrator, OrchestratorConfig
from .autonomous.continuation import TodoContinuationEnforcer, ContinuationConfig
from .todo.manager import TodoManager
from .todo.events import TodoEventEmitter, get_event_emitter
from .domain import DomainDetector, DomainContext, DEFAULT_DOMAIN_CONTEXT

# === v17.0 Module Imports ===
# Confidence Calibration
from .calibration import ConfidenceCalibrator, CalibrationConfig

# Dataset Versioning & Branching
from .versioning import VersionManager
from .branching import BranchManager

# OSDK & Writeback
from .osdk import OSDKClient
from .writeback import WritebackEngine

# Semantic Search
from .semantic_search import SemanticSearcher

# Auto-remediation
from .remediation import RemediationEngine

# XAI (Explanation)
from .explanation import DecisionExplainer

# What-If Analysis
from .what_if import WhatIfAnalyzer

# NL Reports
from .reporting import ReportGenerator, ReportType

# AIP: NL2SQL, Function Calling, Real-time
from .aip.nl2sql import NL2SQLEngine
from .aip.function_calling import ToolRegistry, ToolSelector, ChainExecutor
from .aip.realtime import StreamingReasoner, EventTrigger

logger = logging.getLogger(__name__)


@dataclass
class PipelineServices:
    """v17.0 통합 서비스 컨테이너"""
    # Core Services
    calibrator: Optional[ConfidenceCalibrator] = None
    version_manager: Optional[VersionManager] = None
    branch_manager: Optional[BranchManager] = None

    # Data Services
    osdk_client: Optional[OSDKClient] = None
    writeback_engine: Optional[WritebackEngine] = None
    semantic_searcher: Optional[SemanticSearcher] = None
    remediation_engine: Optional[RemediationEngine] = None

    # Analysis Services
    decision_explainer: Optional[DecisionExplainer] = None
    whatif_analyzer: Optional[WhatIfAnalyzer] = None
    report_generator: Optional[ReportGenerator] = None

    # AIP Services
    nl2sql_engine: Optional[NL2SQLEngine] = None
    tool_registry: Optional[ToolRegistry] = None
    tool_selector: Optional[ToolSelector] = None
    chain_executor: Optional[ChainExecutor] = None
    streaming_reasoner: Optional[StreamingReasoner] = None

    def to_dict(self) -> Dict[str, bool]:
        """서비스 활성화 상태"""
        return {
            "calibrator": self.calibrator is not None,
            "version_manager": self.version_manager is not None,
            "branch_manager": self.branch_manager is not None,
            "osdk_client": self.osdk_client is not None,
            "writeback_engine": self.writeback_engine is not None,
            "semantic_searcher": self.semantic_searcher is not None,
            "remediation_engine": self.remediation_engine is not None,
            "decision_explainer": self.decision_explainer is not None,
            "whatif_analyzer": self.whatif_analyzer is not None,
            "report_generator": self.report_generator is not None,
            "nl2sql_engine": self.nl2sql_engine is not None,
            "tool_registry": self.tool_registry is not None,
            "streaming_reasoner": self.streaming_reasoner is not None,
        }


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
        enable_v17_services: bool = True,  # v17.0: 서비스 활성화
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
            enable_v17_services: v17.0 서비스 활성화
        """
        self.scenario_name = scenario_name
        self.llm_client = llm_client
        self.output_dir = Path(output_dir) if output_dir else None
        self.enable_continuation = enable_continuation
        self.auto_detect_domain = auto_detect_domain
        self.enable_v17_services = enable_v17_services

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

        # === v17.0: 서비스 초기화 ===
        self.services = self._initialize_v17_services() if enable_v17_services else PipelineServices()

        # SharedContext에 서비스 참조 저장
        if enable_v17_services:
            self._attach_services_to_context()

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
        if enable_v17_services:
            active_services = [k for k, v in self.services.to_dict().items() if v]
            logger.info(f"v17.0 services activated: {len(active_services)} services")

    def _initialize_v17_services(self) -> PipelineServices:
        """v17.0 서비스 초기화"""
        services = PipelineServices()

        try:
            # 1. Confidence Calibrator
            services.calibrator = ConfidenceCalibrator(
                context=self.shared_context,
                config=CalibrationConfig(),
            )
            logger.debug("Initialized: ConfidenceCalibrator")

            # 2. Version Manager
            services.version_manager = VersionManager(
                context=self.shared_context,
            )
            logger.debug("Initialized: VersionManager")

            # 3. Branch Manager (needs VersionManager)
            services.branch_manager = BranchManager(
                version_manager=services.version_manager,
                context=self.shared_context,
            )
            logger.debug("Initialized: BranchManager")

            # 4. OSDK Client
            services.osdk_client = OSDKClient(
                context=self.shared_context,
            )
            logger.debug("Initialized: OSDKClient")

            # 5. Writeback Engine
            services.writeback_engine = WritebackEngine(
                context=self.shared_context,
            )
            logger.debug("Initialized: WritebackEngine")

            # 6. Semantic Searcher (with LLM for query expansion & summarization)
            services.semantic_searcher = SemanticSearcher(
                context=self.shared_context,
                llm_client=self.llm_client,
            )
            logger.debug("Initialized: SemanticSearcher (LLM: %s)", self.llm_client is not None)

            # 7. Remediation Engine (with LLM for value inference)
            services.remediation_engine = RemediationEngine(
                context=self.shared_context,
                llm_client=self.llm_client,
            )
            logger.debug("Initialized: RemediationEngine (LLM: %s)", self.llm_client is not None)

            # 8. Decision Explainer (XAI)
            services.decision_explainer = DecisionExplainer(
                context=self.shared_context,
            )
            logger.debug("Initialized: DecisionExplainer")

            # 9. What-If Analyzer (with LLM for recommendations)
            services.whatif_analyzer = WhatIfAnalyzer(
                context=self.shared_context,
                llm_client=self.llm_client,
            )
            logger.debug("Initialized: WhatIfAnalyzer (LLM: %s)", self.llm_client is not None)

            # 10. Report Generator
            services.report_generator = ReportGenerator(
                context=self.shared_context,
            )
            logger.debug("Initialized: ReportGenerator")

            # 11. NL2SQL Engine
            services.nl2sql_engine = NL2SQLEngine(
                context=self.shared_context,
            )
            logger.debug("Initialized: NL2SQLEngine")

            # 12. Tool Registry & Executor
            services.tool_registry = ToolRegistry()
            services.tool_selector = ToolSelector(
                registry=services.tool_registry,
            )
            services.chain_executor = ChainExecutor(
                registry=services.tool_registry,
            )
            logger.debug("Initialized: ToolRegistry, ToolSelector, ChainExecutor")

            # 13. Streaming Reasoner
            services.streaming_reasoner = StreamingReasoner(
                context=self.shared_context,
            )
            logger.debug("Initialized: StreamingReasoner")

        except Exception as e:
            logger.warning(f"Some v17.0 services failed to initialize: {e}")

        return services

    def _attach_services_to_context(self) -> None:
        """SharedContext에 서비스 참조 연결"""
        # Calibrator를 SharedContext에 연결
        if self.services.calibrator:
            self.shared_context.confidence_calibrator = self.services.calibrator

        # v17.0 서비스 참조를 dynamic_data에 저장
        self.shared_context.set_dynamic("_v17_services", {
            "calibrator": self.services.calibrator,
            "version_manager": self.services.version_manager,
            "branch_manager": self.services.branch_manager,
            "osdk_client": self.services.osdk_client,
            "writeback_engine": self.services.writeback_engine,
            "semantic_searcher": self.services.semantic_searcher,
            "remediation_engine": self.services.remediation_engine,
            "decision_explainer": self.services.decision_explainer,
            "whatif_analyzer": self.services.whatif_analyzer,
            "report_generator": self.services.report_generator,
            "nl2sql_engine": self.services.nl2sql_engine,
            "tool_registry": self.services.tool_registry,
            "streaming_reasoner": self.services.streaming_reasoner,
        })

    async def _resume_todo(self, todo) -> None:
        """Todo 재개 콜백"""
        await self.agent_orchestrator._assign_agent_to_todo(todo)

    # === 비동기 실행 ===

    async def run(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        phases: Optional[List[str]] = None,
        generate_report: bool = True,
        run_what_if: bool = False,
        auto_remediate: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        비동기 파이프라인 실행

        Args:
            tables_data: 테이블 데이터
            phases: 실행할 Phase 목록
            generate_report: 파이프라인 완료 후 보고서 생성
            run_what_if: 파이프라인 완료 후 What-If 분석 실행
            auto_remediate: 데이터 품질 이슈 자동 수정

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

        # === v17.0: Pre-pipeline 처리 ===
        if self.enable_v17_services:
            async for event in self._run_pre_pipeline_hooks(tables_data, auto_remediate):
                yield event

        # Continuation Enforcer 시작
        if self.continuation_enforcer:
            await self.continuation_enforcer.start()

        current_phase = None
        try:
            # 파이프라인 실행
            async for event in self.agent_orchestrator.run_pipeline(tables_data, phases):
                # 활동 기록
                if self.continuation_enforcer:
                    self.continuation_enforcer.record_activity()

                # Phase 변경 감지 및 v17.0 훅 실행
                if event.get("type") == "phase_complete" and self.enable_v17_services:
                    completed_phase = event.get("phase")
                    async for hook_event in self._run_post_phase_hooks(completed_phase):
                        yield hook_event

                yield event

        finally:
            # Continuation Enforcer 중지
            if self.continuation_enforcer:
                await self.continuation_enforcer.stop()

            # === v17.0: Post-pipeline 처리 ===
            if self.enable_v17_services:
                async for event in self._run_post_pipeline_hooks(generate_report, run_what_if):
                    yield event

            # 결과 저장
            self._save_results()

    async def _run_pre_pipeline_hooks(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        auto_remediate: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """v17.0: 파이프라인 시작 전 처리"""
        yield {
            "type": "v17_pre_pipeline_start",
            "message": "Running v17.0 pre-pipeline hooks...",
            "timestamp": datetime.now().isoformat(),
        }

        # 1. 초기 버전 커밋
        if self.services.version_manager:
            try:
                version_id = self.services.version_manager.commit(
                    message="Initial data load",
                    author="system",
                )
                yield {
                    "type": "v17_version_commit",
                    "version_id": version_id,
                    "message": "Initial data load committed",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Version commit failed: {e}")

        # 2. OSDK 동기화
        if self.services.osdk_client:
            try:
                self.services.osdk_client.sync_from_shared_context()
                yield {
                    "type": "v17_osdk_sync",
                    "message": "OSDK synchronized from context",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"OSDK sync failed: {e}")

        # 3. Semantic Search 인덱싱
        if self.services.semantic_searcher:
            try:
                total_indexed = 0
                for table_name in tables_data.keys():
                    # 텍스트 컬럼 자동 감지 (tables_data 직접 사용)
                    table_data = tables_data.get(table_name, {})
                    columns = table_data.get("columns", {})

                    # columns가 dict인 경우 (tables_data 형식)
                    if isinstance(columns, dict):
                        text_columns = [
                            col for col, info in columns.items()
                            if isinstance(info, dict) and info.get("dtype") in ["object", "string", "str"]
                        ]
                    # columns가 list인 경우 (TableInfo 형식)
                    elif isinstance(columns, list):
                        text_columns = [
                            col.get("name", col.get("column_name", ""))
                            for col in columns
                            if isinstance(col, dict) and col.get("dtype") in ["object", "string", "str"]
                        ]
                    else:
                        text_columns = []

                    if text_columns:
                        count = await self.services.semantic_searcher.index_table(
                            table_name, text_columns[:5]
                        )
                        total_indexed += count

                if total_indexed > 0:
                    yield {
                        "type": "v17_semantic_index",
                        "total_indexed": total_indexed,
                        "message": f"Indexed {total_indexed} entities for semantic search",
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Semantic indexing failed: {e}")

        # 4. 데이터 품질 체크 및 자동 수정
        if auto_remediate and self.services.remediation_engine:
            try:
                for table_name in tables_data.keys():
                    result = await self.services.remediation_engine.remediate(
                        table=table_name,
                        auto_approve=True,
                    )
                    if result.actions_taken > 0:
                        yield {
                            "type": "v17_remediation",
                            "table": table_name,
                            "actions_taken": result.actions_taken,
                            "issues_fixed": result.issues_fixed,
                            "timestamp": datetime.now().isoformat(),
                        }
            except Exception as e:
                logger.warning(f"Auto-remediation failed: {e}")

        yield {
            "type": "v17_pre_pipeline_complete",
            "message": "Pre-pipeline hooks completed",
            "timestamp": datetime.now().isoformat(),
        }

    async def _run_post_phase_hooks(self, phase: str) -> AsyncGenerator[Dict[str, Any], None]:
        """v17.0: Phase 완료 후 처리"""
        # 1. 버전 커밋
        if self.services.version_manager:
            try:
                version_id = self.services.version_manager.commit(
                    message=f"Phase {phase} completed",
                    author="system",
                )
                yield {
                    "type": "v17_phase_version_commit",
                    "phase": phase,
                    "version_id": version_id,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Phase version commit failed: {e}")

        # 2. 신뢰도 보정 업데이트
        if self.services.calibrator:
            try:
                # Phase 완료 시 calibration 히스토리 저장
                calibration_summary = self.services.calibrator.get_calibration_summary()
                self.shared_context.calibration_history.append({
                    "phase": phase,
                    "summary": calibration_summary,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.warning(f"Calibration update failed: {e}")

    async def _run_post_pipeline_hooks(
        self,
        generate_report: bool,
        run_what_if: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """v17.0: 파이프라인 완료 후 처리"""
        yield {
            "type": "v17_post_pipeline_start",
            "message": "Running v17.0 post-pipeline hooks...",
            "timestamp": datetime.now().isoformat(),
        }

        # 1. 거버넌스 결정 설명 생성 (XAI)
        if self.services.decision_explainer:
            try:
                decisions = self.shared_context.governance_decisions
                explanations = []
                for decision in decisions[:10]:  # 상위 10개만
                    decision_id = getattr(decision, "decision_id", None) or getattr(decision, "concept_id", None)
                    if decision_id:
                        explanation = await self.services.decision_explainer.explain_decision(
                            decision_id
                        )
                        explanations.append(explanation.to_dict())

                if explanations:
                    yield {
                        "type": "v17_xai_explanations",
                        "count": len(explanations),
                        "message": f"Generated {len(explanations)} decision explanations",
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.warning(f"XAI explanation failed: {e}")

        # 2. 보고서 생성
        if generate_report and self.services.report_generator:
            try:
                # Executive Summary
                exec_report = await self.services.report_generator.generate(
                    report_type=ReportType.EXECUTIVE_SUMMARY,
                    title=f"{self.scenario_name} Analysis Report",
                )

                # Pipeline Status Report
                status_report = await self.services.report_generator.generate(
                    report_type=ReportType.PIPELINE_STATUS,
                )

                yield {
                    "type": "v17_reports_generated",
                    "reports": [
                        {"id": exec_report.report_id, "type": "executive_summary"},
                        {"id": status_report.report_id, "type": "pipeline_status"},
                    ],
                    "message": "Reports generated successfully",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")

        # 3. What-If 분석 (선택적)
        if run_what_if and self.services.whatif_analyzer:
            try:
                # 기본 시나리오 생성 및 시뮬레이션
                scenario = self.services.whatif_analyzer.create_scenario(
                    name="baseline_scenario",
                    changes={},  # 기준선
                    description="Baseline scenario for comparison",
                )
                result = await self.services.whatif_analyzer.simulate(scenario)

                yield {
                    "type": "v17_whatif_analysis",
                    "scenario_id": scenario.scenario_id,
                    "risk_score": result.risk_score,
                    "opportunity_score": result.opportunity_score,
                    "impact_summary": result.impact_summary,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"What-If analysis failed: {e}")

        # 4. v22.0: Writeback — 거버넌스 결정을 OSDK/원본 시스템에 반영
        if self.services.writeback_engine:
            try:
                writeback_count = 0
                for decision in self.shared_context.governance_decisions[:20]:
                    decision_type = decision.decision_type if hasattr(decision, 'decision_type') else decision.get("decision_type", "")
                    concept_id = decision.concept_id if hasattr(decision, 'concept_id') else decision.get("concept_id", "")
                    if decision_type == "approve" and concept_id:
                        from .writeback.writeback_engine import WritebackAction
                        action = WritebackAction.upsert(
                            object_type="ontology_concept",
                            object_id=concept_id,
                            data={
                                "status": "approved",
                                "confidence": decision.confidence if hasattr(decision, 'confidence') else decision.get("confidence", 0),
                                "decision_type": decision_type,
                            },
                        )
                        wb_result = await self.services.writeback_engine.execute(action)
                        if wb_result.success:
                            writeback_count += 1

                if writeback_count > 0:
                    yield {
                        "type": "v17_writeback",
                        "actions_written": writeback_count,
                        "message": f"Wrote back {writeback_count} approved governance decisions",
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Writeback failed: {e}")

        # 5. v22.0: Branch — 파이프라인 결과를 브랜치로 저장
        if self.services.branch_manager:
            try:
                branch = self.services.branch_manager.create_branch(
                    name=f"pipeline_{self.scenario_name}",
                    description=f"Pipeline results for {self.scenario_name}",
                )
                yield {
                    "type": "v17_branch_created",
                    "branch_name": branch.name if hasattr(branch, 'name') else str(branch),
                    "message": f"Results branched as pipeline_{self.scenario_name}",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Branch creation failed: {e}")

        # 6. v22.0: StreamingReasoner — 파이프라인 이벤트를 실시간 인사이트로 변환
        if self.services.streaming_reasoner:
            try:
                from .aip.realtime.streaming_reasoner import StreamEvent
                # 주요 파이프라인 결과를 이벤트로 변환
                events = []
                for insight in (self.shared_context.business_insights or [])[:10]:
                    if isinstance(insight, dict) and insight.get("severity") in ("critical", "high"):
                        events.append(StreamEvent(
                            event_id=f"insight_{insight.get('insight_id', 'unknown')}",
                            event_type="business_insight",
                            data={"title": insight.get("title", ""), "severity": insight.get("severity", "")},
                            source="unified_pipeline",
                        ))

                if events:
                    stream_insights = await self.services.streaming_reasoner.process_batch(events)
                    if stream_insights:
                        yield {
                            "type": "v17_streaming_insights",
                            "insights_generated": len(stream_insights),
                            "message": f"StreamingReasoner generated {len(stream_insights)} real-time insights",
                            "timestamp": datetime.now().isoformat(),
                        }
            except Exception as e:
                logger.warning(f"Streaming reasoner failed: {e}")

        # 7. v22.0: ToolRegistry — 분석 도구 등록 요약
        if self.services.tool_registry:
            try:
                registry_summary = self.services.tool_registry.get_registry_summary()
                tool_count = registry_summary.get("total_tools", 0)
                if tool_count > 0:
                    yield {
                        "type": "v17_tool_registry",
                        "tools_registered": tool_count,
                        "summary": registry_summary,
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Tool registry summary failed: {e}")

        # 8. v22.0: ToolSelector + ChainExecutor — 파이프라인 결과 기반 자동 분석 체인 실행
        if self.services.tool_selector and self.services.chain_executor:
            try:
                # 파이프라인 결과 요약을 쿼리로 변환하여 관련 도구 자동 선택
                governance_decisions = self.shared_context.get_dynamic("governance_decisions", [])
                query = f"Analyze ontology pipeline results: {len(self.shared_context.ontology_concepts)} concepts, {len(governance_decisions)} governance decisions"
                selection_result = await self.services.tool_selector.select_tools(
                    query=query, max_tools=3, include_alternatives=False
                )
                if selection_result and selection_result.chain and selection_result.chain.calls:
                    exec_result = await self.services.chain_executor.execute(
                        chain=selection_result.chain,
                        initial_context={
                            "concepts_count": len(self.shared_context.ontology_concepts),
                            "governance_decisions": len(governance_decisions),
                            "scenario_name": self.shared_context.scenario_name,
                        },
                    )
                    yield {
                        "type": "v17_tool_chain_execution",
                        "tools_selected": selection_result.chain.tool_names,
                        "execution_success": exec_result.success,
                        "steps_completed": exec_result.completed_steps,
                        "timestamp": datetime.now().isoformat(),
                    }
                    logger.info(f"[v22.0] ToolSelector+ChainExecutor: {selection_result.chain.tool_names} → success={exec_result.success}")
            except Exception as e:
                logger.warning(f"Tool selector/chain executor failed: {e}")

        # 9. 최종 버전 커밋
        if self.services.version_manager:
            try:
                version_id = self.services.version_manager.commit(
                    message="Pipeline completed",
                    author="system",
                )
                self.services.version_manager.tag(version_id, "pipeline_complete")

                yield {
                    "type": "v17_final_version",
                    "version_id": version_id,
                    "tag": "pipeline_complete",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"Final version commit failed: {e}")

        yield {
            "type": "v17_post_pipeline_complete",
            "services_summary": self.services.to_dict(),
            "message": "Post-pipeline hooks completed",
            "timestamp": datetime.now().isoformat(),
        }

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

    # === v17.0 서비스 직접 접근 ===

    async def query_nl2sql(self, question: str) -> Dict[str, Any]:
        """
        자연어 쿼리 실행 (NL2SQL)

        Args:
            question: 자연어 질문

        Returns:
            쿼리 결과 (SQL, 데이터, 메타데이터)
        """
        if not self.services.nl2sql_engine:
            raise RuntimeError("NL2SQL engine not initialized")

        result = await self.services.nl2sql_engine.query(question)
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
        tables: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        시맨틱 검색

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수
            tables: 검색할 테이블 (None이면 전체)

        Returns:
            검색 결과 목록
        """
        if not self.services.semantic_searcher:
            raise RuntimeError("Semantic searcher not initialized")

        results = await self.services.semantic_searcher.search(query, limit, tables)
        return [r.to_dict() for r in results]

    async def run_whatif_scenario(
        self,
        name: str,
        changes: Dict[str, Any],
        description: str = "",
    ) -> Dict[str, Any]:
        """
        What-If 시나리오 실행

        Args:
            name: 시나리오 이름
            changes: 변경 사항 {target: value}
            description: 설명

        Returns:
            시뮬레이션 결과
        """
        if not self.services.whatif_analyzer:
            raise RuntimeError("What-If analyzer not initialized")

        scenario = self.services.whatif_analyzer.create_scenario(
            name=name,
            changes=changes,
            description=description,
        )
        result = await self.services.whatif_analyzer.simulate(scenario)
        return result.to_dict()

    async def generate_report(
        self,
        report_type: str = "executive_summary",
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        보고서 생성

        Args:
            report_type: 보고서 유형
            title: 보고서 제목

        Returns:
            생성된 보고서
        """
        if not self.services.report_generator:
            raise RuntimeError("Report generator not initialized")

        report = await self.services.report_generator.generate(
            report_type=ReportType(report_type),
            title=title or f"{self.scenario_name} Report",
        )
        return report.to_dict()

    async def explain_decision(self, decision_id: str) -> Dict[str, Any]:
        """
        거버넌스 결정 설명 (XAI)

        Args:
            decision_id: 결정 ID

        Returns:
            설명 결과
        """
        if not self.services.decision_explainer:
            raise RuntimeError("Decision explainer not initialized")

        explanation = await self.services.decision_explainer.explain_decision(decision_id)
        return explanation.to_dict()

    def get_version_history(self) -> List[Dict[str, Any]]:
        """버전 히스토리 조회"""
        if not self.services.version_manager:
            return []

        return self.services.version_manager.log()

    def checkout_version(self, version_id: str) -> None:
        """특정 버전으로 체크아웃"""
        if not self.services.version_manager:
            raise RuntimeError("Version manager not initialized")

        self.services.version_manager.checkout(version_id)

    def get_services_status(self) -> Dict[str, Any]:
        """v17.0 서비스 상태 조회"""
        return {
            "enabled": self.enable_v17_services,
            "services": self.services.to_dict(),
            "calibration_history_count": len(self.shared_context.calibration_history),
            "version_count": len(self.shared_context.dataset_versions),
            "streaming_insights_count": len(self.shared_context.streaming_insights),
        }


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
