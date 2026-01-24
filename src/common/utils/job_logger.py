"""
Job Logger - 파이프라인 실행 전체 기록 시스템 (v6.2)

블랙박스 방지를 위한 상세 기록 시스템
모든 단계의 산출물, 에이전트 대화, 판단 근거, LLM 답변을 Job ID 기반 폴더에 저장

Job ID 형식: {timestamp}_{domain}_{numbering}
예: 20260113_143052_retail_001

=== 저장 구조 ===

data/storage/                      # 상세 결과물 (블랙박스 방지)
└── {job_id}/
    ├── job_metadata.json          # Job 메타데이터
    ├── pipeline.log               # 이 Job의 상세 실행 로그
    ├── phase1_discovery/          # Phase 1 에이전트 상세
    │   ├── data_analyst.json      # 실행 결과 + 판단 근거
    │   ├── tda_expert.json
    │   ├── schema_analyst.json
    │   ├── value_matcher.json
    │   ├── entity_classifier.json
    │   └── relationship_detector.json
    ├── phase2_refinement/         # Phase 2 에이전트 상세
    │   ├── ontology_architect.json
    │   ├── conflict_resolver.json
    │   ├── quality_judge.json
    │   └── semantic_validator.json
    ├── phase3_governance/         # Phase 3 에이전트 상세
    │   ├── governance_strategist.json
    │   ├── action_prioritizer.json
    │   ├── risk_assessor.json
    │   └── policy_generator.json
    ├── llm_calls/
    │   └── llm_calls.jsonl        # 모든 LLM 호출 기록 (프롬프트 + 응답)
    ├── conversations/             # 에이전트 간 대화/합의 과정
    │   └── consensus_rounds.json
    ├── thinking/                  # 에이전트 사고 과정
    │   └── {agent}_thinking.jsonl
    ├── reasoning_chains/          # 추론 체인
    │   └── chain_*.json
    ├── artifacts/                 # 최종 결과물
    │   ├── ontology.json
    │   ├── insights.json
    │   ├── governance_decisions.json
    │   ├── knowledge_graph.json
    │   ├── evidence_chain.json
    │   └── predictive_insights.json
    └── summary.json               # 실행 요약

Logs/                              # 시스템 로그 (진행현황)
├── ontoloty.log                   # 메인 시스템 로그
└── pipeline_YYYYMMDD.log          # 일별 진행 로그
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PhaseType(str, Enum):
    DISCOVERY = "phase1_discovery"
    REFINEMENT = "phase2_refinement"
    GOVERNANCE = "phase3_governance"


class ThinkingType(str, Enum):
    """사고 유형"""
    ANALYSIS = "analysis"           # 데이터 분석
    REASONING = "reasoning"         # 추론/논리적 사고
    HYPOTHESIS = "hypothesis"       # 가설 수립
    EVALUATION = "evaluation"       # 평가/판단
    DECISION = "decision"           # 의사결정
    REFLECTION = "reflection"       # 반성/재검토
    PLANNING = "planning"           # 계획 수립
    INSIGHT = "insight"             # 인사이트 도출
    QUESTION = "question"           # 질문/의문점
    CONCLUSION = "conclusion"       # 결론 도출


@dataclass
class ThinkingRecord:
    """에이전트 사고 과정 기록"""
    timestamp: str
    thinking_type: str              # ThinkingType value
    content: str                    # 사고 내용
    context: Dict[str, Any] = field(default_factory=dict)  # 사고 컨텍스트
    confidence: Optional[float] = None  # 확신도 (0.0-1.0)
    evidence: List[str] = field(default_factory=list)  # 근거
    related_data: Dict[str, Any] = field(default_factory=dict)  # 관련 데이터


@dataclass
class ReasoningChain:
    """추론 체인 (여러 사고 단계의 연결)"""
    chain_id: str
    agent_name: str
    started_at: str
    goal: str                       # 추론 목표
    completed_at: Optional[str] = None
    steps: List[ThinkingRecord] = field(default_factory=list)
    final_conclusion: Optional[str] = None
    confidence: Optional[float] = None
    success: bool = True


@dataclass
class LLMCallRecord:
    """LLM 호출 기록 (v5.1 - 사고 과정 포함)"""
    timestamp: str
    agent_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    messages: List[Dict[str, str]]
    response: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # v5.1: 사고 과정 관련 필드
    purpose: Optional[str] = None           # LLM 호출 목적
    thinking_before: Optional[str] = None   # 호출 전 에이전트의 생각
    thinking_after: Optional[str] = None    # 호출 후 에이전트의 해석
    extracted_insights: List[str] = field(default_factory=list)  # 추출된 인사이트


@dataclass
class AgentExecutionRecord:
    """에이전트 실행 기록 (v5.1 - 사고 과정 포함)"""
    agent_name: str
    agent_type: str
    phase: str
    started_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    todo_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    llm_calls: List[str] = field(default_factory=list)  # LLM call IDs
    logs: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    # v5.1: 사고 과정 관련 필드
    thinking_process: List[ThinkingRecord] = field(default_factory=list)  # 사고 과정
    reasoning_chains: List[str] = field(default_factory=list)  # 추론 체인 IDs
    key_decisions: List[Dict[str, Any]] = field(default_factory=list)  # 주요 결정들
    analysis_summary: Optional[str] = None  # 분석 요약


@dataclass
class JobMetadata:
    """Job 메타데이터"""
    job_id: str
    domain: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    total_agents: int = 0
    completed_agents: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    input_tables: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class JobLogger:
    """
    Job 기반 로깅 시스템

    파이프라인 실행의 모든 산출물을 Job ID 폴더에 저장
    """

    _instance: Optional["JobLogger"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        base_dir: str = "./data/storage",
        domain: Optional[str] = None,
    ):
        # Singleton이므로 이미 초기화된 경우 skip
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.current_job_id: Optional[str] = None
        self.current_job_dir: Optional[Path] = None
        self.metadata: Optional[JobMetadata] = None

        self._llm_call_counter = 0
        self._agent_records: Dict[str, AgentExecutionRecord] = {}
        self._file_logger: Optional[logging.Logger] = None
        self._llm_file: Optional[Path] = None

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "JobLogger":
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """싱글톤 인스턴스 리셋 (테스트용)"""
        cls._instance = None

    def _generate_job_id(self, domain: str) -> str:
        """Job ID 생성: timestamp_domain_numbering"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 같은 날짜/도메인의 기존 job 수 확인
        pattern = f"{timestamp[:8]}_*_{domain}_*"
        existing = list(self.base_dir.glob(pattern))
        numbering = len(existing) + 1

        return f"{timestamp}_{domain}_{numbering:03d}"

    def start_job(
        self,
        domain: str,
        input_tables: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        새 Job 시작

        Args:
            domain: 산업 도메인 (retail, healthcare, manufacturing 등)
            input_tables: 입력 테이블 목록
            config: 파이프라인 설정

        Returns:
            job_id
        """
        # 기존 job이 있으면 종료
        if self.current_job_id:
            self.end_job()

        # Job ID 생성
        self.current_job_id = self._generate_job_id(domain)
        self.current_job_dir = self.base_dir / self.current_job_id

        # 디렉토리 구조 생성
        self._create_directory_structure()

        # 메타데이터 초기화
        self.metadata = JobMetadata(
            job_id=self.current_job_id,
            domain=domain,
            started_at=datetime.now().isoformat(),
            input_tables=input_tables,
            config=config or {},
        )
        self._save_metadata()

        # 파일 로거 설정
        self._setup_file_logger()

        # LLM 호출 파일 초기화
        self._llm_file = self.current_job_dir / "llm_calls" / "llm_calls.jsonl"

        self.log_info(f"Job started: {self.current_job_id}")
        self.log_info(f"Domain: {domain}")
        self.log_info(f"Input tables: {input_tables}")

        return self.current_job_id

    def _create_directory_structure(self):
        """Job 디렉토리 구조 생성"""
        dirs = [
            self.current_job_dir / "phase1_discovery",
            self.current_job_dir / "phase2_refinement",
            self.current_job_dir / "phase3_governance",
            self.current_job_dir / "llm_calls",
            self.current_job_dir / "artifacts",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_file_logger(self):
        """파일 로거 설정"""
        log_file = self.current_job_dir / "pipeline.log"

        self._file_logger = logging.getLogger(f"job.{self.current_job_id}")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.handlers = []

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self._file_logger.addHandler(file_handler)

    def _save_metadata(self):
        """메타데이터 저장"""
        if self.current_job_dir and self.metadata:
            metadata_file = self.current_job_dir / "job_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.metadata), f, indent=2, ensure_ascii=False, default=str)

    def end_job(self, status: str = "completed") -> Dict[str, Any]:
        """
        Job 종료

        Returns:
            Job 요약 정보
        """
        if not self.current_job_id:
            return {}

        self.log_info(f"Job ending: {self.current_job_id}")

        # 메타데이터 업데이트
        if self.metadata:
            self.metadata.completed_at = datetime.now().isoformat()
            self.metadata.status = status
            self._save_metadata()

        # 요약 생성
        summary = self._generate_summary()
        summary_file = self.current_job_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        self.log_info(f"Job completed: {status}")
        self.log_info(f"Total LLM calls: {self.metadata.total_llm_calls if self.metadata else 0}")
        self.log_info(f"Total tokens: {self.metadata.total_tokens if self.metadata else 0}")

        # 리셋
        result = {
            "job_id": self.current_job_id,
            "job_dir": str(self.current_job_dir),
            "summary": summary,
        }

        self.current_job_id = None
        self.current_job_dir = None
        self.metadata = None
        self._agent_records = {}
        self._llm_call_counter = 0

        return result

    def _generate_summary(self) -> Dict[str, Any]:
        """Job 요약 생성"""
        if not self.metadata:
            return {}

        # 에이전트별 통계
        agent_stats = {}
        for agent_name, record in self._agent_records.items():
            agent_stats[agent_name] = {
                "phase": record.phase,
                "success": record.success,
                "duration_ms": record.duration_ms,
                "llm_calls": len(record.llm_calls),
            }

        return {
            "job_id": self.metadata.job_id,
            "domain": self.metadata.domain,
            "started_at": self.metadata.started_at,
            "completed_at": self.metadata.completed_at,
            "duration_seconds": self._calculate_duration(),
            "status": self.metadata.status,
            "statistics": {
                "total_agents": self.metadata.total_agents,
                "completed_agents": self.metadata.completed_agents,
                "total_llm_calls": self.metadata.total_llm_calls,
                "total_tokens": self.metadata.total_tokens,
            },
            "agent_stats": agent_stats,
            "input_tables": self.metadata.input_tables,
        }

    def _calculate_duration(self) -> float:
        """실행 시간 계산 (초)"""
        if not self.metadata or not self.metadata.completed_at:
            return 0.0

        start = datetime.fromisoformat(self.metadata.started_at)
        end = datetime.fromisoformat(self.metadata.completed_at)
        return (end - start).total_seconds()

    # === 로깅 메서드 ===

    def log(self, level: LogLevel, message: str, **kwargs):
        """범용 로그"""
        if self._file_logger:
            extra = f" | {kwargs}" if kwargs else ""
            getattr(self._file_logger, level.lower())(f"{message}{extra}")

    def log_debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)

    def log_info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)

    def log_error(self, message: str, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)

    # === 에이전트 실행 기록 ===

    def start_agent_execution(
        self,
        agent_name: str,
        agent_type: str,
        phase: PhaseType,
        todo_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """에이전트 실행 시작 기록"""
        record = AgentExecutionRecord(
            agent_name=agent_name,
            agent_type=agent_type,
            phase=phase.value,
            started_at=datetime.now().isoformat(),
            todo_id=todo_id,
            input_data=input_data or {},
        )

        self._agent_records[agent_name] = record

        if self.metadata:
            self.metadata.total_agents += 1
            self._save_metadata()

        self.log_info(f"Agent started: {agent_name} ({agent_type})", phase=phase.value)

        return agent_name

    def end_agent_execution(
        self,
        agent_name: str,
        output_data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """에이전트 실행 종료 기록"""
        if agent_name not in self._agent_records:
            return

        record = self._agent_records[agent_name]
        record.completed_at = datetime.now().isoformat()
        record.output_data = output_data or {}
        record.success = success
        record.error = error

        # 실행 시간 계산
        start = datetime.fromisoformat(record.started_at)
        end = datetime.fromisoformat(record.completed_at)
        record.duration_ms = (end - start).total_seconds() * 1000

        # 파일로 저장
        self._save_agent_record(record)

        if self.metadata:
            self.metadata.completed_agents += 1
            self._save_metadata()

        status = "SUCCESS" if success else f"FAILED: {error}"
        self.log_info(
            f"Agent completed: {agent_name}",
            status=status,
            duration_ms=record.duration_ms
        )

    def _save_agent_record(self, record: AgentExecutionRecord):
        """에이전트 기록 파일로 저장"""
        if not self.current_job_dir:
            return

        phase_dir = self.current_job_dir / record.phase
        agent_file = phase_dir / f"{record.agent_type}.json"

        with open(agent_file, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2, ensure_ascii=False, default=str)

    def add_agent_log(self, agent_name: str, message: str, level: LogLevel = LogLevel.INFO):
        """에이전트 내부 로그 추가"""
        if agent_name in self._agent_records:
            self._agent_records[agent_name].logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": level.value,
                "message": message,
            })

        self.log(level, f"[{agent_name}] {message}")

    # === v5.1: 사고 과정 기록 ===

    def log_thinking(
        self,
        agent_name: str,
        thinking_type: ThinkingType,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        evidence: Optional[List[str]] = None,
        related_data: Optional[Dict[str, Any]] = None,
    ):
        """
        에이전트의 사고 과정 기록

        Args:
            agent_name: 에이전트 이름
            thinking_type: 사고 유형 (analysis, reasoning, decision 등)
            content: 사고 내용
            context: 사고 컨텍스트
            confidence: 확신도 (0.0-1.0)
            evidence: 근거 목록
            related_data: 관련 데이터
        """
        if agent_name not in self._agent_records:
            return

        thinking_record = ThinkingRecord(
            timestamp=datetime.now().isoformat(),
            thinking_type=thinking_type.value,
            content=content,
            context=context or {},
            confidence=confidence,
            evidence=evidence or [],
            related_data=related_data or {},
        )

        self._agent_records[agent_name].thinking_process.append(thinking_record)

        # 파일에도 기록
        self._save_thinking_to_file(agent_name, thinking_record)

        # 로그 출력
        conf_str = f" (confidence: {confidence:.2f})" if confidence else ""
        self.log_info(f"[{agent_name}] 💭 {thinking_type.value}: {content[:200]}...{conf_str}")

    def _save_thinking_to_file(self, agent_name: str, record: ThinkingRecord):
        """사고 과정을 별도 파일에 저장"""
        if not self.current_job_dir:
            return

        thinking_dir = self.current_job_dir / "thinking"
        thinking_dir.mkdir(exist_ok=True)

        thinking_file = thinking_dir / f"{agent_name}_thinking.jsonl"
        with open(thinking_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")

    def log_analysis(
        self,
        agent_name: str,
        analysis_target: str,
        findings: List[str],
        methodology: Optional[str] = None,
        data_examined: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ):
        """
        데이터 분석 과정 기록

        Args:
            agent_name: 에이전트 이름
            analysis_target: 분석 대상
            findings: 발견 사항 목록
            methodology: 분석 방법론
            data_examined: 분석한 데이터
            confidence: 분석 확신도
        """
        content = f"Analyzing '{analysis_target}': {', '.join(findings[:3])}"
        if len(findings) > 3:
            content += f" (+{len(findings)-3} more findings)"

        self.log_thinking(
            agent_name=agent_name,
            thinking_type=ThinkingType.ANALYSIS,
            content=content,
            context={
                "target": analysis_target,
                "methodology": methodology,
            },
            confidence=confidence,
            evidence=findings,
            related_data=data_examined or {},
        )

    def log_reasoning(
        self,
        agent_name: str,
        premise: str,
        logic: str,
        conclusion: str,
        confidence: Optional[float] = None,
        supporting_evidence: Optional[List[str]] = None,
    ):
        """
        추론 과정 기록

        Args:
            agent_name: 에이전트 이름
            premise: 전제
            logic: 논리적 추론
            conclusion: 결론
            confidence: 추론 확신도
            supporting_evidence: 지지 근거
        """
        content = f"Premise: {premise} → Logic: {logic} → Conclusion: {conclusion}"

        self.log_thinking(
            agent_name=agent_name,
            thinking_type=ThinkingType.REASONING,
            content=content,
            context={
                "premise": premise,
                "logic": logic,
                "conclusion": conclusion,
            },
            confidence=confidence,
            evidence=supporting_evidence or [],
        )

    def log_hypothesis(
        self,
        agent_name: str,
        hypothesis: str,
        basis: List[str],
        testable: bool = True,
        confidence: Optional[float] = None,
    ):
        """
        가설 수립 기록

        Args:
            agent_name: 에이전트 이름
            hypothesis: 가설 내용
            basis: 가설의 근거
            testable: 검증 가능 여부
            confidence: 가설 확신도
        """
        content = f"Hypothesis: {hypothesis} (testable: {testable})"

        self.log_thinking(
            agent_name=agent_name,
            thinking_type=ThinkingType.HYPOTHESIS,
            content=content,
            context={
                "testable": testable,
            },
            confidence=confidence,
            evidence=basis,
        )

    def log_decision(
        self,
        agent_name: str,
        decision: str,
        options_considered: List[str],
        rationale: str,
        confidence: Optional[float] = None,
        impact: Optional[str] = None,
    ):
        """
        의사결정 기록

        Args:
            agent_name: 에이전트 이름
            decision: 결정 내용
            options_considered: 고려한 옵션들
            rationale: 결정 이유
            confidence: 결정 확신도
            impact: 예상 영향
        """
        content = f"Decision: {decision} | Rationale: {rationale}"

        # 주요 결정으로도 기록
        if agent_name in self._agent_records:
            self._agent_records[agent_name].key_decisions.append({
                "timestamp": datetime.now().isoformat(),
                "decision": decision,
                "options": options_considered,
                "rationale": rationale,
                "confidence": confidence,
                "impact": impact,
            })

        self.log_thinking(
            agent_name=agent_name,
            thinking_type=ThinkingType.DECISION,
            content=content,
            context={
                "options_considered": options_considered,
                "rationale": rationale,
                "impact": impact,
            },
            confidence=confidence,
        )

    def log_insight(
        self,
        agent_name: str,
        insight: str,
        source: str,
        importance: str = "medium",
        actionable: bool = True,
        confidence: Optional[float] = None,
    ):
        """
        인사이트 도출 기록

        Args:
            agent_name: 에이전트 이름
            insight: 인사이트 내용
            source: 인사이트 출처
            importance: 중요도 (low, medium, high, critical)
            actionable: 실행 가능 여부
            confidence: 인사이트 확신도
        """
        content = f"💡 Insight [{importance}]: {insight} (from: {source})"

        self.log_thinking(
            agent_name=agent_name,
            thinking_type=ThinkingType.INSIGHT,
            content=content,
            context={
                "source": source,
                "importance": importance,
                "actionable": actionable,
            },
            confidence=confidence,
        )

    def log_evaluation(
        self,
        agent_name: str,
        subject: str,
        criteria: List[str],
        scores: Dict[str, float],
        overall_score: float,
        verdict: str,
    ):
        """
        평가 과정 기록

        Args:
            agent_name: 에이전트 이름
            subject: 평가 대상
            criteria: 평가 기준
            scores: 기준별 점수
            overall_score: 종합 점수
            verdict: 평가 결론
        """
        content = f"Evaluating '{subject}': Overall {overall_score:.2f} → {verdict}"

        self.log_thinking(
            agent_name=agent_name,
            thinking_type=ThinkingType.EVALUATION,
            content=content,
            context={
                "subject": subject,
                "criteria": criteria,
                "scores": scores,
                "verdict": verdict,
            },
            confidence=overall_score,
        )

    # === v5.1: 추론 체인 기록 ===

    _reasoning_chains: Dict[str, ReasoningChain] = {}

    def start_reasoning_chain(
        self,
        agent_name: str,
        goal: str,
    ) -> str:
        """
        추론 체인 시작

        Args:
            agent_name: 에이전트 이름
            goal: 추론 목표

        Returns:
            chain_id
        """
        chain_id = f"chain_{datetime.now().strftime('%H%M%S')}_{len(self._reasoning_chains)}"

        chain = ReasoningChain(
            chain_id=chain_id,
            agent_name=agent_name,
            started_at=datetime.now().isoformat(),
            goal=goal,
        )

        self._reasoning_chains[chain_id] = chain

        if agent_name in self._agent_records:
            self._agent_records[agent_name].reasoning_chains.append(chain_id)

        self.log_info(f"[{agent_name}] 🔗 Reasoning chain started: {goal}")

        return chain_id

    def add_reasoning_step(
        self,
        chain_id: str,
        thinking_type: ThinkingType,
        content: str,
        confidence: Optional[float] = None,
        evidence: Optional[List[str]] = None,
    ):
        """추론 체인에 단계 추가"""
        if chain_id not in self._reasoning_chains:
            return

        step = ThinkingRecord(
            timestamp=datetime.now().isoformat(),
            thinking_type=thinking_type.value,
            content=content,
            confidence=confidence,
            evidence=evidence or [],
        )

        self._reasoning_chains[chain_id].steps.append(step)

    def end_reasoning_chain(
        self,
        chain_id: str,
        conclusion: str,
        confidence: Optional[float] = None,
        success: bool = True,
    ):
        """추론 체인 종료"""
        if chain_id not in self._reasoning_chains:
            return

        chain = self._reasoning_chains[chain_id]
        chain.completed_at = datetime.now().isoformat()
        chain.final_conclusion = conclusion
        chain.confidence = confidence
        chain.success = success

        # 파일에 저장
        self._save_reasoning_chain(chain)

        self.log_info(
            f"[{chain.agent_name}] 🔗 Reasoning chain completed: {conclusion[:100]}",
            confidence=confidence,
            steps=len(chain.steps)
        )

    def _save_reasoning_chain(self, chain: ReasoningChain):
        """추론 체인을 파일에 저장"""
        if not self.current_job_dir:
            return

        chains_dir = self.current_job_dir / "reasoning_chains"
        chains_dir.mkdir(exist_ok=True)

        chain_file = chains_dir / f"{chain.chain_id}.json"
        with open(chain_file, "w", encoding="utf-8") as f:
            json.dump(asdict(chain), f, indent=2, ensure_ascii=False, default=str)

    def set_analysis_summary(self, agent_name: str, summary: str):
        """에이전트의 분석 요약 설정"""
        if agent_name in self._agent_records:
            self._agent_records[agent_name].analysis_summary = summary
            self.log_info(f"[{agent_name}] 📝 Analysis summary: {summary[:200]}...")

    # === LLM 호출 기록 ===

    def log_llm_call(
        self,
        agent_name: str,
        model: str,
        messages: List[Dict[str, str]],
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # v5.1: 사고 과정 관련 파라미터
        purpose: Optional[str] = None,
        thinking_before: Optional[str] = None,
        thinking_after: Optional[str] = None,
        extracted_insights: Optional[List[str]] = None,
    ) -> str:
        """
        LLM 호출 기록 (v5.1 - 사고 과정 포함)

        Args:
            agent_name: 에이전트 이름
            model: 사용된 모델
            messages: 대화 메시지
            response: LLM 응답
            prompt_tokens: 프롬프트 토큰 수
            completion_tokens: 완료 토큰 수
            latency_ms: 지연 시간 (ms)
            success: 성공 여부
            error: 에러 메시지
            metadata: 추가 메타데이터
            purpose: LLM 호출 목적
            thinking_before: 호출 전 에이전트의 생각
            thinking_after: 호출 후 에이전트의 해석
            extracted_insights: 응답에서 추출된 인사이트
        """
        self._llm_call_counter += 1
        call_id = f"llm_{self._llm_call_counter:06d}"

        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            messages=messages,
            response=response[:10000] if response else "",  # 응답 길이 제한
            success=success,
            error=error,
            metadata=metadata or {},
            # v5.1: 사고 과정
            purpose=purpose,
            thinking_before=thinking_before,
            thinking_after=thinking_after,
            extracted_insights=extracted_insights or [],
        )

        # JSONL 파일에 추가
        if self._llm_file:
            with open(self._llm_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")

        # 에이전트 레코드에 call ID 추가
        if agent_name in self._agent_records:
            self._agent_records[agent_name].llm_calls.append(call_id)

        # 메타데이터 업데이트
        if self.metadata:
            self.metadata.total_llm_calls += 1
            self.metadata.total_tokens += record.total_tokens

        # 사고 과정이 있으면 상세 로그
        if purpose or thinking_before or thinking_after:
            self.log_debug(
                f"LLM call: {call_id}",
                agent=agent_name,
                model=model,
                tokens=record.total_tokens,
                latency_ms=latency_ms,
                purpose=purpose,
            )
        else:
            self.log_debug(
                f"LLM call: {call_id}",
                agent=agent_name,
                model=model,
                tokens=record.total_tokens,
                latency_ms=latency_ms
            )

        return call_id

    # === 산출물 저장 ===

    def save_artifact(
        self,
        name: str,
        data: Any,
        artifact_type: str = "json",
    ):
        """산출물 저장"""
        if not self.current_job_dir:
            return

        artifacts_dir = self.current_job_dir / "artifacts"

        if artifact_type == "json":
            file_path = artifacts_dir / f"{name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        else:
            file_path = artifacts_dir / f"{name}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(data))

        self.log_info(f"Artifact saved: {name}", path=str(file_path))

    def save_ontology(self, ontology: Dict[str, Any]):
        """온톨로지 저장"""
        self.save_artifact("ontology", ontology)

    def save_insights(self, insights: List[Dict[str, Any]]):
        """인사이트 저장"""
        self.save_artifact("insights", insights)

    def save_governance_decisions(self, decisions: List[Dict[str, Any]]):
        """거버넌스 결정 저장"""
        self.save_artifact("governance_decisions", decisions)

    def save_knowledge_graph(self, kg: Dict[str, Any]):
        """Knowledge Graph 저장"""
        self.save_artifact("knowledge_graph", kg)

    def save_palantir_insights(self, insights: List[Dict[str, Any]]):
        """v10.0: 팔란티어 스타일 예측 인사이트 저장"""
        self.save_artifact("predictive_insights", insights)

    # === 유틸리티 ===

    def get_job_dir(self) -> Optional[Path]:
        """현재 Job 디렉토리 반환"""
        return self.current_job_dir

    def get_job_id(self) -> Optional[str]:
        """현재 Job ID 반환"""
        return self.current_job_id

    def is_active(self) -> bool:
        """Job이 활성화 상태인지 확인"""
        return self.current_job_id is not None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """모든 Job 목록 반환"""
        jobs = []
        for job_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if job_dir.is_dir():
                metadata_file = job_dir / "job_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        jobs.append(json.load(f))
        return jobs


# === 전역 함수 ===

def get_job_logger() -> JobLogger:
    """JobLogger 싱글톤 인스턴스 반환"""
    return JobLogger.get_instance()


def start_job(domain: str, input_tables: List[str], config: Optional[Dict[str, Any]] = None) -> str:
    """Job 시작 (편의 함수)"""
    return get_job_logger().start_job(domain, input_tables, config)


def end_job(status: str = "completed") -> Dict[str, Any]:
    """Job 종료 (편의 함수)"""
    return get_job_logger().end_job(status)
