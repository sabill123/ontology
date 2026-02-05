"""
Evidence Chain - Blockchain-inspired Evidence Management System (v11.0)

블록체인 구조를 차용한 근거 체인 시스템:
- 각 에이전트의 분석 결과를 블록으로 저장
- 이전 블록의 해시를 참조하여 근거 체인 형성
- Phase 간 근거 추적 및 무결성 검증
- Council 토론 시 근거 기반 주장 지원

핵심 개념:
1. EvidenceBlock: 개별 근거 블록 (에이전트 분석 결과)
2. EvidenceChain: 블록들의 연결 체인 (근거 추적)
3. EvidenceRegistry: 전역 근거 저장소
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class EvidenceType(str, Enum):
    """근거 유형"""
    # Phase 1: Discovery
    TDA_ANALYSIS = "tda_analysis"  # 위상수학적 분석 결과
    SCHEMA_ANALYSIS = "schema_analysis"  # 스키마 분석 결과
    VALUE_MATCHING = "value_matching"  # 값 매칭 결과
    ENTITY_CLASSIFICATION = "entity_classification"  # 엔티티 분류 결과
    RELATIONSHIP_DETECTION = "relationship_detection"  # 관계 탐지 결과

    # Phase 2: Refinement
    ONTOLOGY_DESIGN = "ontology_design"  # 온톨로지 설계 결과
    CONFLICT_RESOLUTION = "conflict_resolution"  # 충돌 해결 결과
    QUALITY_ASSESSMENT = "quality_assessment"  # 품질 평가 결과
    SEMANTIC_VALIDATION = "semantic_validation"  # 의미 검증 결과
    CAUSAL_INFERENCE = "causal_inference"  # 인과 추론 결과
    SIMULATION_RESULT = "simulation_result"  # 시뮬레이션 결과

    # Phase 3: Governance
    COUNCIL_VOTE = "council_vote"  # Council 투표 결과
    DEBATE_ARGUMENT = "debate_argument"  # 토론 주장
    FINAL_DECISION = "final_decision"  # 최종 결정
    GOVERNANCE_DECISION = "governance_decision"  # 거버넌스 결정

    # v17.0: Palantir-style Features
    # NL2SQL
    NL2SQL_TRANSLATION = "nl2sql_translation"  # 자연어→SQL 변환 결과
    NL2SQL_VALIDATION = "nl2sql_validation"  # SQL 검증 결과
    QUERY_EXECUTION = "query_execution"  # 쿼리 실행 결과

    # Function Calling
    TOOL_SELECTION = "tool_selection"  # 도구 선택 결과
    TOOL_EXECUTION = "tool_execution"  # 도구 실행 결과
    TOOL_CHAIN = "tool_chain"  # 도구 체인 실행 결과

    # Versioning & Branching
    VERSION_COMMIT = "version_commit"  # 버전 커밋
    VERSION_CHECKOUT = "version_checkout"  # 버전 체크아웃
    BRANCH_CREATE = "branch_create"  # 브랜치 생성
    MERGE_DECISION = "merge_decision"  # 머지 결정
    CONFLICT_RESOLUTION_VERSION = "conflict_resolution_version"  # 버전 충돌 해결

    # Writeback
    WRITEBACK_ACTION = "writeback_action"  # Writeback 액션
    WRITEBACK_VALIDATION = "writeback_validation"  # Writeback 검증
    WRITEBACK_ROLLBACK = "writeback_rollback"  # Writeback 롤백

    # Calibration & Uncertainty
    CALIBRATION = "calibration"  # 신뢰도 보정
    UNCERTAINTY_ESTIMATE = "uncertainty_estimate"  # 불확실성 추정

    # Real-time Reasoning
    STREAMING_INSIGHT = "streaming_insight"  # 스트리밍 인사이트
    EVENT_TRIGGER = "event_trigger"  # 이벤트 트리거 발동
    ANOMALY_DETECTION = "anomaly_detection"  # 이상 탐지

    # Semantic Search
    SEMANTIC_SEARCH = "semantic_search"  # 의미 검색 결과
    ENTITY_LINKING = "entity_linking"  # 엔티티 연결

    # Auto-remediation
    REMEDIATION_ACTION = "remediation_action"  # 자동 수정 액션
    REMEDIATION_VALIDATION = "remediation_validation"  # 수정 검증

    # XAI (Explanation)
    DECISION_EXPLANATION = "decision_explanation"  # 결정 설명
    FEATURE_ATTRIBUTION = "feature_attribution"  # Feature Attribution
    COUNTERFACTUAL = "counterfactual"  # 반사실적 설명

    # What-If Analysis
    SCENARIO_DEFINITION = "scenario_definition"  # 시나리오 정의
    SCENARIO_SIMULATION = "scenario_simulation"  # 시나리오 시뮬레이션
    IMPACT_PREDICTION = "impact_prediction"  # 영향 예측

    # NL Reports
    REPORT_GENERATION = "report_generation"  # 보고서 생성
    NARRATIVE_SYNTHESIS = "narrative_synthesis"  # 내러티브 합성


@dataclass
class EvidenceBlock:
    """
    근거 블록 - 블록체인의 블록에 해당

    각 에이전트의 분석 결과를 블록으로 저장하며,
    이전 블록의 해시를 참조하여 근거 체인을 형성
    """
    # 블록 식별
    block_id: str
    phase: str  # "phase1_discovery", "phase2_refinement", "phase3_governance"
    agent: str  # 생성한 에이전트 이름
    evidence_type: EvidenceType

    # 핵심 내용
    finding: str  # 발견 사항 (무엇을 발견했는가)
    reasoning: str  # 추론 근거 (왜 그렇게 판단했는가)
    conclusion: str  # 결론 (그래서 어떻다는 것인가)

    # 데이터 참조
    data_references: List[str] = field(default_factory=list)  # ["table.column", ...]
    metrics: Dict[str, Any] = field(default_factory=dict)  # {"confidence": 0.85, "p_value": 0.02, ...}

    # 근거 체인 (블록체인 구조)
    supporting_block_ids: List[str] = field(default_factory=list)  # 이 결론을 지지하는 이전 블록들
    contradicting_block_ids: List[str] = field(default_factory=list)  # 이 결론과 충돌하는 블록들
    previous_block_hash: str = ""  # 직전 블록의 해시 (체인 연결)

    # 메타데이터
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 블록 해시 (자동 계산)
    block_hash: str = field(default="", repr=False)

    def __post_init__(self):
        """블록 생성 후 해시 계산"""
        if not self.block_hash:
            self.block_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """블록 해시 계산 (SHA-256)"""
        content = {
            "block_id": self.block_id,
            "phase": self.phase,
            "agent": self.agent,
            "evidence_type": self.evidence_type.value if isinstance(self.evidence_type, EvidenceType) else self.evidence_type,
            "finding": self.finding,
            "reasoning": self.reasoning,
            "conclusion": self.conclusion,
            "data_references": self.data_references,
            "metrics": self.metrics,
            "supporting_block_ids": self.supporting_block_ids,
            "previous_block_hash": self.previous_block_hash,
            "timestamp": self.timestamp,
        }
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """블록 무결성 검증"""
        return self.block_hash == self._compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = asdict(self)
        if isinstance(self.evidence_type, EvidenceType):
            result["evidence_type"] = self.evidence_type.value
        return result

    def get_citation(self) -> str:
        """인용 형식으로 반환 (토론 시 사용)"""
        return f"[Block #{self.block_id}] {self.agent}: \"{self.finding}\" (confidence: {self.confidence:.0%})"


class EvidenceChain:
    """
    근거 체인 - 블록들의 연결 관리

    블록체인 원리 적용:
    1. 모든 블록은 이전 블록의 해시를 참조
    2. 체인 무결성 검증 가능
    3. 근거 추적 (어떤 근거가 어떤 결론을 지지하는지)
    """

    def __init__(self):
        self.blocks: Dict[str, EvidenceBlock] = {}  # block_id -> EvidenceBlock
        self.chain_order: List[str] = []  # 블록 생성 순서
        self.phase_blocks: Dict[str, List[str]] = {
            "phase1_discovery": [],
            "phase2_refinement": [],
            "phase3_governance": [],
        }
        self.agent_blocks: Dict[str, List[str]] = {}  # agent -> [block_ids]
        self._block_counter = 0

    def _generate_block_id(self) -> str:
        """블록 ID 생성"""
        self._block_counter += 1
        return f"EVD-{self._block_counter:04d}"

    def add_block(
        self,
        phase: str,
        agent: str,
        evidence_type: EvidenceType,
        finding: str,
        reasoning: str,
        conclusion: str,
        data_references: List[str] = None,
        metrics: Dict[str, Any] = None,
        supporting_block_ids: List[str] = None,
        contradicting_block_ids: List[str] = None,
        confidence: float = 0.0,
    ) -> EvidenceBlock:
        """
        새 근거 블록 추가

        Args:
            phase: 파이프라인 단계
            agent: 생성 에이전트
            evidence_type: 근거 유형
            finding: 발견 사항
            reasoning: 추론 근거
            conclusion: 결론
            data_references: 데이터 참조
            metrics: 측정 지표
            supporting_block_ids: 지지 블록 IDs
            contradicting_block_ids: 충돌 블록 IDs
            confidence: 신뢰도

        Returns:
            생성된 EvidenceBlock
        """
        block_id = self._generate_block_id()

        # 이전 블록 해시 (체인 연결)
        previous_hash = "0000000000000000"  # Genesis
        if self.chain_order:
            last_block = self.blocks[self.chain_order[-1]]
            previous_hash = last_block.block_hash

        block = EvidenceBlock(
            block_id=block_id,
            phase=phase,
            agent=agent,
            evidence_type=evidence_type,
            finding=finding,
            reasoning=reasoning,
            conclusion=conclusion,
            data_references=data_references or [],
            metrics=metrics or {},
            supporting_block_ids=supporting_block_ids or [],
            contradicting_block_ids=contradicting_block_ids or [],
            previous_block_hash=previous_hash,
            confidence=confidence,
        )

        # 저장
        self.blocks[block_id] = block
        self.chain_order.append(block_id)

        # Phase별 인덱싱
        if phase in self.phase_blocks:
            self.phase_blocks[phase].append(block_id)

        # Agent별 인덱싱
        if agent not in self.agent_blocks:
            self.agent_blocks[agent] = []
        self.agent_blocks[agent].append(block_id)

        logger.info(f"Evidence block added: {block_id} ({agent} - {evidence_type.value})")

        return block

    def get_block(self, block_id: str) -> Optional[EvidenceBlock]:
        """블록 조회"""
        return self.blocks.get(block_id)

    def get_all_blocks(self) -> List[EvidenceBlock]:
        """모든 블록을 생성 순서대로 반환"""
        return [self.blocks[bid] for bid in self.chain_order if bid in self.blocks]

    def get_blocks_by_phase(self, phase: str) -> List[EvidenceBlock]:
        """Phase별 블록 조회"""
        block_ids = self.phase_blocks.get(phase, [])
        return [self.blocks[bid] for bid in block_ids if bid in self.blocks]

    def get_blocks_by_agent(self, agent: str) -> List[EvidenceBlock]:
        """Agent별 블록 조회"""
        block_ids = self.agent_blocks.get(agent, [])
        return [self.blocks[bid] for bid in block_ids if bid in self.blocks]

    def get_supporting_chain(self, block_id: str) -> List[EvidenceBlock]:
        """
        지지 근거 체인 조회 (재귀적)

        특정 블록의 결론을 지지하는 모든 이전 블록들을 추적
        """
        block = self.blocks.get(block_id)
        if not block:
            return []

        chain = []
        visited = set()

        def traverse(bid: str):
            if bid in visited or bid not in self.blocks:
                return
            visited.add(bid)
            b = self.blocks[bid]
            chain.append(b)
            for supporting_id in b.supporting_block_ids:
                traverse(supporting_id)

        for supporting_id in block.supporting_block_ids:
            traverse(supporting_id)

        return chain

    def get_evidence_for_conclusion(self, conclusion_keywords: List[str]) -> List[EvidenceBlock]:
        """
        특정 결론과 관련된 근거 블록 검색

        Args:
            conclusion_keywords: 검색 키워드

        Returns:
            관련 블록 목록
        """
        results = []
        keywords_lower = [k.lower() for k in conclusion_keywords]

        for block in self.blocks.values():
            text = f"{block.finding} {block.conclusion}".lower()
            if any(kw in text for kw in keywords_lower):
                results.append(block)

        return results

    def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        전체 체인 무결성 검증

        블록체인 원리: 하나의 블록이라도 변조되면 전체 체인이 무효화
        """
        if not self.chain_order:
            return {"valid": True, "blocks_verified": 0, "errors": []}

        errors = []
        blocks_verified = 0

        for i, block_id in enumerate(self.chain_order):
            block = self.blocks.get(block_id)
            if not block:
                errors.append(f"Block {block_id} not found in chain")
                continue

            # 블록 자체 무결성 검증
            if not block.verify_integrity():
                errors.append(f"Block {block_id} hash mismatch (tampered)")
                continue

            # 체인 연결 검증 (이전 블록 해시)
            if i > 0:
                prev_block = self.blocks.get(self.chain_order[i - 1])
                if prev_block and block.previous_block_hash != prev_block.block_hash:
                    errors.append(f"Block {block_id} chain link broken (prev hash mismatch)")
                    continue

            blocks_verified += 1

        return {
            "valid": len(errors) == 0,
            "blocks_verified": blocks_verified,
            "total_blocks": len(self.chain_order),
            "errors": errors,
        }

    def format_evidence_summary(self, block_ids: List[str]) -> str:
        """
        토론용 근거 요약 생성

        Council 토론 시 에이전트가 인용할 수 있는 형식
        """
        lines = ["=== Evidence Summary ==="]

        for block_id in block_ids:
            block = self.blocks.get(block_id)
            if not block:
                continue

            lines.append(f"\n[{block.block_id}] {block.agent} ({block.phase})")
            lines.append(f"  Finding: {block.finding}")
            lines.append(f"  Reasoning: {block.reasoning}")
            lines.append(f"  Conclusion: {block.conclusion}")
            lines.append(f"  Confidence: {block.confidence:.0%}")

            if block.data_references:
                lines.append(f"  Data: {', '.join(block.data_references[:3])}")

            if block.supporting_block_ids:
                lines.append(f"  Supports: {', '.join(block.supporting_block_ids)}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        evidence_by_type = {}
        for block in self.blocks.values():
            type_name = block.evidence_type.value if hasattr(block.evidence_type, 'value') else str(block.evidence_type)
            evidence_by_type[type_name] = evidence_by_type.get(type_name, 0) + 1

        phase_counts = {phase: len(blocks) for phase, blocks in self.phase_blocks.items()}
        agent_counts = {agent: len(blocks) for agent, blocks in self.agent_blocks.items()}

        return {
            "total_blocks": len(self.blocks),
            "chain_length": len(self.chain_order),
            "evidence_by_type": evidence_by_type,
            "blocks_by_phase": phase_counts,
            "blocks_by_agent": agent_counts,
            "unique_agents": len(self.agent_blocks),
        }

    def to_dict(self) -> Dict[str, Any]:
        """전체 체인을 딕셔너리로 변환"""
        return {
            "chain_order": self.chain_order,
            "blocks": {bid: block.to_dict() for bid, block in self.blocks.items()},
            "phase_blocks": self.phase_blocks,
            "agent_blocks": self.agent_blocks,
            "integrity": self.verify_chain_integrity(),
        }


class EvidenceRegistry:
    """
    전역 근거 저장소

    파이프라인 전체에서 공유되는 싱글톤 패턴의 근거 관리자
    """

    _instance: Optional["EvidenceRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.chain = EvidenceChain()
        self._topic_index: Dict[str, Set[str]] = {}  # topic -> block_ids
        logger.info("EvidenceRegistry initialized")

    @classmethod
    def reset(cls):
        """레지스트리 리셋 (테스트용)"""
        cls._instance = None

    def record_evidence(
        self,
        phase: str,
        agent: str,
        evidence_type: EvidenceType,
        finding: str,
        reasoning: str,
        conclusion: str,
        topics: List[str] = None,
        **kwargs
    ) -> EvidenceBlock:
        """
        근거 기록 (에이전트 호출용 간편 메서드)

        Args:
            phase: 파이프라인 단계
            agent: 에이전트 이름
            evidence_type: 근거 유형
            finding: 발견 사항
            reasoning: 추론 근거
            conclusion: 결론
            topics: 검색용 토픽 태그
            **kwargs: 추가 옵션 (data_references, metrics, supporting_block_ids 등)

        Returns:
            생성된 EvidenceBlock
        """
        block = self.chain.add_block(
            phase=phase,
            agent=agent,
            evidence_type=evidence_type,
            finding=finding,
            reasoning=reasoning,
            conclusion=conclusion,
            **kwargs
        )

        # 토픽 인덱싱
        if topics:
            for topic in topics:
                topic_lower = topic.lower()
                if topic_lower not in self._topic_index:
                    self._topic_index[topic_lower] = set()
                self._topic_index[topic_lower].add(block.block_id)

        return block

    def find_evidence_by_topic(self, topic: str) -> List[EvidenceBlock]:
        """토픽으로 근거 검색"""
        block_ids = self._topic_index.get(topic.lower(), set())
        return [self.chain.blocks[bid] for bid in block_ids if bid in self.chain.blocks]

    def get_phase_evidence(self, phase: str) -> List[EvidenceBlock]:
        """Phase별 근거 조회"""
        return self.chain.get_blocks_by_phase(phase)

    def get_agent_evidence(self, agent: str) -> List[EvidenceBlock]:
        """Agent별 근거 조회"""
        return self.chain.get_blocks_by_agent(agent)

    def get_all_evidence(self) -> List[EvidenceBlock]:
        """전체 근거 조회 (시간순)"""
        return [self.chain.blocks[bid] for bid in self.chain.chain_order]

    def format_for_council_debate(
        self,
        proposal: str,
        relevant_phases: List[str] = None
    ) -> str:
        """
        Council 토론용 근거 문서 생성

        Args:
            proposal: 토론 안건
            relevant_phases: 관련 Phase (기본: 전체)

        Returns:
            토론용 근거 요약 문서
        """
        relevant_phases = relevant_phases or ["phase1_discovery", "phase2_refinement"]

        lines = [
            "=" * 60,
            "EVIDENCE DOCUMENT FOR COUNCIL DEBATE",
            "=" * 60,
            f"\nProposal: {proposal}\n",
            "-" * 60,
        ]

        for phase in relevant_phases:
            blocks = self.chain.get_blocks_by_phase(phase)
            if not blocks:
                continue

            phase_name = phase.replace("_", " ").title()
            lines.append(f"\n## {phase_name} Evidence\n")

            for block in blocks:
                lines.append(f"### [{block.block_id}] {block.agent}")
                lines.append(f"- **Finding**: {block.finding}")
                lines.append(f"- **Reasoning**: {block.reasoning}")
                lines.append(f"- **Conclusion**: {block.conclusion}")
                lines.append(f"- **Confidence**: {block.confidence:.0%}")

                if block.metrics:
                    metrics_str = ", ".join(f"{k}={v}" for k, v in list(block.metrics.items())[:3])
                    lines.append(f"- **Metrics**: {metrics_str}")

                if block.supporting_block_ids:
                    lines.append(f"- **Based on**: {', '.join(block.supporting_block_ids)}")

                lines.append("")

        lines.append("-" * 60)
        lines.append(f"Total Evidence Blocks: {len(self.chain.chain_order)}")

        integrity = self.chain.verify_chain_integrity()
        lines.append(f"Chain Integrity: {'VALID' if integrity['valid'] else 'INVALID'}")

        return "\n".join(lines)


# 전역 레지스트리 인스턴스 (편의용)
def get_evidence_registry() -> EvidenceRegistry:
    """전역 EvidenceRegistry 인스턴스 반환"""
    return EvidenceRegistry()
