"""
Governance Analysis Utilities

거버넌스 분석 알고리즘:
- 의사결정 매트릭스 (Decision Matrix)
- 리스크 평가 프레임워크 (Risk Assessment)
- 우선순위 계산 (Priority Calculation)
- 정책 규칙 생성 (Policy Generation)
- v6.0: LLM 동적 임계값 시스템 (Dynamic Threshold)
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


# === LLM Dynamic Threshold System (v6.0) ===

@dataclass
class DynamicThresholds:
    """
    LLM이 결정한 동적 임계값

    v6.0: 컨텍스트 기반 임계값 - LLM이 도메인, 데이터 특성,
    비즈니스 요구사항을 분석하여 최적의 임계값을 결정
    """
    approve_min_quality: float
    approve_min_evidence: float
    reject_max_quality: float
    escalate_range: Tuple[float, float]

    # LLM 판단 근거
    reasoning: str
    domain_context: str
    risk_tolerance: str  # conservative, balanced, aggressive

    # 메타데이터
    generated_at: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approve_min_quality": self.approve_min_quality,
            "approve_min_evidence": self.approve_min_evidence,
            "reject_max_quality": self.reject_max_quality,
            "escalate_range": list(self.escalate_range),
            "reasoning": self.reasoning,
            "domain_context": self.domain_context,
            "risk_tolerance": self.risk_tolerance,
            "generated_at": self.generated_at,
            "confidence": self.confidence,
        }


class DynamicThresholdGenerator:
    """
    LLM 기반 동적 임계값 생성기

    v6.0: 하드코딩된 임계값 대신 LLM이 컨텍스트를 분석하여
    적절한 임계값을 동적으로 결정

    고려 요소:
    - 도메인 특성 (Healthcare: 엄격, E-commerce: 유연)
    - 데이터 품질 현황 (전반적으로 낮으면 임계값 완화)
    - 비즈니스 요구사항 (빠른 결정 vs 정확한 결정)
    - 규제 환경 (HIPAA, GDPR 등)
    """

    # 기본 임계값 (LLM 실패 시 사용)
    DEFAULT_THRESHOLDS = DynamicThresholds(
        approve_min_quality=55,
        approve_min_evidence=40,
        reject_max_quality=25,
        escalate_range=(35, 55),
        reasoning="Default thresholds (LLM not available)",
        domain_context="general",
        risk_tolerance="balanced",
    )

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._cached_thresholds: Optional[DynamicThresholds] = None
        self._cache_context_hash: Optional[str] = None

    async def generate_thresholds(
        self,
        context: "SharedContext",
        force_refresh: bool = False,
    ) -> DynamicThresholds:
        """
        컨텍스트 기반 동적 임계값 생성

        Args:
            context: 파이프라인 공유 컨텍스트
            force_refresh: 캐시 무시하고 새로 생성

        Returns:
            DynamicThresholds: LLM이 결정한 임계값
        """
        # 컨텍스트 해시로 캐싱 (같은 컨텍스트면 재사용)
        context_hash = self._compute_context_hash(context)

        if not force_refresh and self._cached_thresholds and self._cache_context_hash == context_hash:
            logger.debug("Using cached dynamic thresholds")
            return self._cached_thresholds

        if not self.llm_client:
            logger.warning("LLM client not available, using default thresholds")
            return self.DEFAULT_THRESHOLDS

        try:
            thresholds = await self._generate_with_llm(context)
            self._cached_thresholds = thresholds
            self._cache_context_hash = context_hash
            return thresholds
        except Exception as e:
            logger.error(f"Failed to generate dynamic thresholds: {e}")
            return self.DEFAULT_THRESHOLDS

    def generate_thresholds_sync(
        self,
        context: "SharedContext",
    ) -> DynamicThresholds:
        """동기 버전 (async 환경이 아닐 때)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 async 컨텍스트 안이면 캐시 반환
                if self._cached_thresholds:
                    return self._cached_thresholds
                return self.DEFAULT_THRESHOLDS
            return loop.run_until_complete(self.generate_thresholds(context))
        except RuntimeError:
            return self.DEFAULT_THRESHOLDS

    async def _generate_with_llm(self, context: "SharedContext") -> DynamicThresholds:
        """LLM을 통한 임계값 생성"""
        from datetime import datetime

        # 컨텍스트 요약 생성
        context_summary = self._summarize_context(context)

        prompt = f"""You are a Data Governance Expert. Analyze the following context and determine optimal decision thresholds.

## Context Summary
{json.dumps(context_summary, indent=2, ensure_ascii=False)}

## Your Task
Based on the domain, data characteristics, and business requirements, determine the optimal governance decision thresholds.

Consider:
1. **Domain Risk**: Healthcare/Finance require stricter thresholds; E-commerce can be more flexible
2. **Data Quality**: If overall quality is low, be more lenient to avoid rejecting everything
3. **Business Speed**: Fast-moving domains need quicker approvals
4. **Regulatory Environment**: HIPAA/GDPR compliance requires conservative approach

## Response Format (JSON only)
```json
{{
    "approve_min_quality": <30-80, quality score threshold for approval>,
    "approve_min_evidence": <30-70, evidence strength threshold for approval>,
    "reject_max_quality": <15-40, quality below this is rejected>,
    "escalate_low": <25-50, lower bound for escalation range>,
    "escalate_high": <45-70, upper bound for escalation range>,
    "risk_tolerance": "<conservative|balanced|aggressive>",
    "reasoning": "<2-3 sentences explaining your threshold choices>"
}}
```

Respond with JSON only, no additional text."""

        response = self.llm_client.chat.completions.create(
            model=getattr(self, 'model_name', 'gpt-5.1'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        response = response.choices[0].message.content

        # JSON 파싱
        parsed = self._parse_llm_response(response)

        return DynamicThresholds(
            approve_min_quality=parsed.get("approve_min_quality", 55),
            approve_min_evidence=parsed.get("approve_min_evidence", 40),
            reject_max_quality=parsed.get("reject_max_quality", 25),
            escalate_range=(
                parsed.get("escalate_low", 35),
                parsed.get("escalate_high", 55)
            ),
            reasoning=parsed.get("reasoning", ""),
            domain_context=context_summary.get("domain", "general"),
            risk_tolerance=parsed.get("risk_tolerance", "balanced"),
            generated_at=datetime.now().isoformat(),
            confidence=0.85,
        )

    def _summarize_context(self, context: "SharedContext") -> Dict[str, Any]:
        """컨텍스트 요약 생성"""
        # 테이블 통계
        table_stats = {
            "count": len(context.tables),
            "total_rows": sum(t.row_count for t in context.tables.values()),
            "avg_columns": sum(len(t.columns) for t in context.tables.values()) / max(len(context.tables), 1),
        }

        # 개념 통계
        concepts = context.ontology_concepts if hasattr(context, 'ontology_concepts') else []
        concept_stats = {
            "count": len(concepts),
            "avg_confidence": sum(c.confidence for c in concepts) / max(len(concepts), 1) if concepts else 0,
            "status_distribution": {},
        }

        for c in concepts:
            status = c.status or "pending"
            concept_stats["status_distribution"][status] = concept_stats["status_distribution"].get(status, 0) + 1

        # 관계 통계
        relationships = context.homeomorphisms if hasattr(context, 'homeomorphisms') else []

        # 도메인 추론
        domain = self._infer_domain(context)

        return {
            "domain": domain,
            "tables": table_stats,
            "concepts": concept_stats,
            "relationships_count": len(relationships),
            "industry": context.get_industry() if hasattr(context, 'get_industry') else "general",
            "regulatory_hints": self._detect_regulatory_hints(context),
        }

    def _infer_domain(self, context: "SharedContext") -> str:
        """도메인 추론"""
        table_names = [name.lower() for name in context.tables.keys()]
        all_columns = []
        for t in context.tables.values():
            cols = t.columns if not isinstance(t.columns, dict) else list(t.columns.keys())
            all_columns.extend([str(c).lower() for c in cols])

        text = " ".join(table_names + all_columns)

        domain_keywords = {
            "healthcare": ["patient", "diagnosis", "medication", "encounter", "claim", "provider", "icd", "cpt"],
            "finance": ["account", "transaction", "balance", "payment", "invoice", "ledger", "credit"],
            "ecommerce": ["customer", "order", "product", "cart", "shipping", "inventory", "price"],
            "manufacturing": ["product", "inventory", "supplier", "warehouse", "batch", "quality"],
        }

        scores = {}
        for domain, keywords in domain_keywords.items():
            scores[domain] = sum(1 for kw in keywords if kw in text)

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "general"

    def _detect_regulatory_hints(self, context: "SharedContext") -> List[str]:
        """규제 힌트 탐지"""
        hints = []

        table_names = [name.lower() for name in context.tables.keys()]
        all_columns = []
        for t in context.tables.values():
            cols = t.columns if not isinstance(t.columns, dict) else list(t.columns.keys())
            all_columns.extend([str(c).lower() for c in cols])

        text = " ".join(table_names + all_columns)

        if any(kw in text for kw in ["patient", "diagnosis", "phi", "hipaa", "medical"]):
            hints.append("HIPAA")
        if any(kw in text for kw in ["gdpr", "consent", "personal_data", "eu_resident"]):
            hints.append("GDPR")
        if any(kw in text for kw in ["pci", "card_number", "cvv", "payment"]):
            hints.append("PCI-DSS")

        return hints

    def _compute_context_hash(self, context: "SharedContext") -> str:
        """컨텍스트 해시 계산 (캐싱용)"""
        import hashlib

        key_parts = [
            str(len(context.tables)),
            str(len(context.ontology_concepts) if hasattr(context, 'ontology_concepts') else 0),
            context.get_industry() if hasattr(context, 'get_industry') else "general",
        ]

        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        import re

        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 블록 없으면 전체를 JSON으로 시도
            json_str = response

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {}


# === Enums ===

class DecisionType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    SCHEDULE_REVIEW = "schedule_review"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# === Data Classes ===

@dataclass
class DecisionResult:
    """거버넌스 결정 결과"""
    concept_id: str
    decision_type: str
    confidence: float
    reasoning: str
    business_value: str  # high, medium, low
    risk_level: str  # high, medium, low
    quality_score: float
    evidence_strength: float


@dataclass
class RiskAssessment:
    """리스크 평가 결과"""
    item_id: str
    item_type: str  # concept, action
    category: str  # data_quality, security, compliance, operational, reputational
    likelihood: int  # 1-5
    impact: int  # 1-5
    risk_score: int  # 1-25
    risk_level: str  # low, medium, high, critical
    mitigation: str


@dataclass
class ActionItem:
    """액션 아이템"""
    action_id: str
    title: str
    action_type: str
    priority: str
    impact: str
    effort: str
    source_decision: str
    dependencies: List[str] = field(default_factory=list)
    priority_score: float = 0.0


@dataclass
class PolicyRule:
    """정책 규칙"""
    rule_id: str
    rule_name: str
    rule_type: str  # data_quality, access_control, monitoring
    target_concept: str
    condition: str
    severity: str
    action_on_violation: str


# === Decision Matrix ===

class DecisionMatrix:
    """
    거버넌스 의사결정 매트릭스 (v6.0)

    v6.0 변경사항:
    - LLM 동적 임계값 시스템 통합
    - 하드코딩된 임계값 제거 → LLM이 컨텍스트 기반 결정

    의사결정 기준:
    - Quality Score
    - Evidence Strength
    - Business Value
    - Risk Level
    """

    def __init__(self, llm_client=None):
        # v6.0: LLM 동적 임계값 생성기
        self.threshold_generator = DynamicThresholdGenerator(llm_client)
        self._dynamic_thresholds: Optional[DynamicThresholds] = None

        # v6.0: 기본 임계값 (LLM 실패 시 또는 초기화 전)
        self.thresholds = {
            "approve_min_quality": 55,
            "approve_min_evidence": 40,
            "reject_max_quality": 25,
            "escalate_range": (35, 55),
        }

        # 비즈니스 가치 판단 키워드
        self.high_value_keywords = [
            "patient", "claim", "encounter", "provider",
            "diagnosis", "procedure", "medication", "billing",
        ]

        self.low_value_keywords = [
            "temp", "test", "debug", "backup", "archive",
        ]

    async def initialize_thresholds(self, context: "SharedContext") -> DynamicThresholds:
        """
        v6.0: LLM을 통한 동적 임계값 초기화

        파이프라인 시작 시 호출하여 컨텍스트 기반 임계값 설정

        Args:
            context: 파이프라인 공유 컨텍스트

        Returns:
            DynamicThresholds: LLM이 결정한 임계값
        """
        self._dynamic_thresholds = await self.threshold_generator.generate_thresholds(context)

        # 내부 thresholds dict 업데이트
        self.thresholds = {
            "approve_min_quality": self._dynamic_thresholds.approve_min_quality,
            "approve_min_evidence": self._dynamic_thresholds.approve_min_evidence,
            "reject_max_quality": self._dynamic_thresholds.reject_max_quality,
            "escalate_range": self._dynamic_thresholds.escalate_range,
        }

        logger.info(
            f"Dynamic thresholds initialized: "
            f"domain={self._dynamic_thresholds.domain_context}, "
            f"risk_tolerance={self._dynamic_thresholds.risk_tolerance}, "
            f"approve_min_quality={self._dynamic_thresholds.approve_min_quality}"
        )

        return self._dynamic_thresholds

    def get_threshold_info(self) -> Dict[str, Any]:
        """현재 임계값 정보 반환 (디버깅/감사용)"""
        if self._dynamic_thresholds:
            return self._dynamic_thresholds.to_dict()
        return {
            "thresholds": self.thresholds,
            "source": "default",
            "dynamic": False,
        }

    def make_decision(
        self,
        concept: Dict[str, Any],
        assessments: Dict[str, Any] = None,
    ) -> DecisionResult:
        """
        개념에 대한 거버넌스 결정

        Args:
            concept: 개념 정보
            assessments: 이전 에이전트 평가 결과

        Returns:
            결정 결과
        """
        assessments = assessments or {}

        # 1. 품질 점수 계산
        quality_score = self._calculate_quality_score(concept, assessments)

        # 2. 증거 강도 계산
        evidence_strength = self._calculate_evidence_strength(concept)

        # 3. 비즈니스 가치 판단
        business_value = self._assess_business_value(concept)

        # 4. 리스크 수준 판단
        risk_level = self._assess_risk_level(concept, quality_score, evidence_strength)

        # 5. 결정 내리기
        decision_type, confidence, reasoning = self._determine_decision(
            quality_score, evidence_strength, business_value, risk_level
        )

        return DecisionResult(
            concept_id=concept.get("concept_id", ""),
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            business_value=business_value,
            risk_level=risk_level,
            quality_score=quality_score,
            evidence_strength=evidence_strength,
        )

    def _calculate_quality_score(
        self,
        concept: Dict[str, Any],
        assessments: Dict[str, Any],
    ) -> float:
        """품질 점수 계산"""
        # 기존 평가에서 점수 추출
        quality_judge = assessments.get("quality_judge", {})
        if isinstance(quality_judge, dict):
            overall = quality_judge.get("overall_score")
            if overall is not None:
                return float(overall)

        # 직접 계산
        score = 0.0
        total_weight = 0.0

        # 신뢰도 (30%)
        confidence = concept.get("confidence", 0)
        score += confidence * 100 * 0.3
        total_weight += 0.3

        # 소스 증거 (30%)
        source_evidence = concept.get("source_evidence", {})
        if isinstance(source_evidence, dict):
            match_rate = source_evidence.get("match_rate", 0)
            ev_confidence = source_evidence.get("confidence", 0)
            score += ((match_rate + ev_confidence) / 2) * 100 * 0.3
            total_weight += 0.3

        # v5.2: 상태 점수 조정 (30%) - pending에 더 높은 기본값
        status = concept.get("status", "pending")
        status_scores = {
            "approved": 95,
            "provisional": 75,
            "pending": 60,      # v5.2: 40 -> 60 (pending도 괜찮은 시작점)
            "rejected": 30,
        }
        score += status_scores.get(status, 60) * 0.3  # v5.2: 가중치 40% -> 30%
        total_weight += 0.3

        # v5.2: concept_type 보너스 (10%)
        concept_type = concept.get("concept_type", "")
        type_bonus = {
            "object_type": 80,      # 엔티티는 중요
            "link_type": 70,        # 관계도 중요
            "property": 60,
        }
        score += type_bonus.get(concept_type, 50) * 0.1
        total_weight += 0.1

        return score / total_weight if total_weight > 0 else 55

    def _calculate_evidence_strength(self, concept: Dict[str, Any]) -> float:
        """v5.2: 증거 강도 계산 - 더 관대한 기본값"""
        score = 0.0

        # v5.2: 소스 테이블 수 (더 높은 점수)
        source_tables = concept.get("source_tables", [])
        if len(source_tables) >= 3:
            score += 40
        elif len(source_tables) >= 2:
            score += 30
        elif len(source_tables) >= 1:
            score += 20  # v5.2: 테이블 1개라도 기본 20점

        # 소스 증거
        source_evidence = concept.get("source_evidence", {})
        if isinstance(source_evidence, dict):
            match_rate = source_evidence.get("match_rate", 0)
            score += match_rate * 30

            # v5.2: confidence도 활용
            ev_confidence = source_evidence.get("confidence", 0)
            score += ev_confidence * 20

            source_count = source_evidence.get("source_count", 0)
            score += min(source_count * 5, 20)
        else:
            # v5.2: source_evidence 없어도 기본 점수
            score += 15

        # v5.2: confidence 있으면 보너스
        concept_confidence = concept.get("confidence", 0)
        if concept_confidence > 0:
            score += concept_confidence * 15

        return min(100, max(score, 25))  # v5.2: 최소 25점 보장

    def _assess_business_value(self, concept: Dict[str, Any]) -> str:
        """비즈니스 가치 판단"""
        name = concept.get("name", "").lower()
        description = concept.get("description", "").lower()
        concept_type = concept.get("concept_type", "")

        text = f"{name} {description}"

        # 고가치 키워드
        high_count = sum(1 for kw in self.high_value_keywords if kw in text)
        low_count = sum(1 for kw in self.low_value_keywords if kw in text)

        # Object Type은 기본적으로 중요
        type_bonus = 2 if concept_type == "object_type" else 0

        if high_count + type_bonus >= 2:
            return "high"
        elif low_count >= 2:
            return "low"
        else:
            return "medium"

    def _assess_risk_level(
        self,
        concept: Dict[str, Any],
        quality_score: float,
        evidence_strength: float,
    ) -> str:
        """리스크 수준 판단"""
        # 낮은 품질 + 낮은 증거 = 높은 리스크
        combined = (quality_score + evidence_strength) / 2

        # 민감 데이터 확인
        name = concept.get("name", "").lower()
        sensitive_keywords = ["patient", "ssn", "phi", "hipaa", "medical", "diagnosis"]
        is_sensitive = any(kw in name for kw in sensitive_keywords)

        if combined < 40 or (is_sensitive and combined < 60):
            return "high"
        elif combined < 60:
            return "medium"
        else:
            return "low"

    def _determine_decision(
        self,
        quality_score: float,
        evidence_strength: float,
        business_value: str,
        risk_level: str,
    ) -> Tuple[str, float, str]:
        """최종 결정"""
        reasons = []

        # 승인 조건
        if (quality_score >= self.thresholds["approve_min_quality"] and
            evidence_strength >= self.thresholds["approve_min_evidence"] and
            risk_level != "high"):

            confidence = min(quality_score, evidence_strength) / 100
            reasons.append(f"High quality ({quality_score:.0f}%)")
            reasons.append(f"Strong evidence ({evidence_strength:.0f}%)")
            reasons.append(f"Acceptable risk ({risk_level})")

            return "approve", confidence, "; ".join(reasons)

        # 거부 조건
        if quality_score <= self.thresholds["reject_max_quality"]:
            confidence = (100 - quality_score) / 100
            reasons.append(f"Low quality ({quality_score:.0f}%)")
            reasons.append(f"Evidence: {evidence_strength:.0f}%")

            return "reject", confidence, "; ".join(reasons)

        # 에스컬레이션 조건
        low, high = self.thresholds["escalate_range"]
        if low <= quality_score <= high and business_value == "high":
            # v27.0: 데이터 기반 confidence (기존 하드코딩 0.6 제거)
            confidence = min(quality_score, evidence_strength) / 100 * 0.9
            confidence = max(0.4, min(0.7, confidence))
            reasons.append(f"Medium quality ({quality_score:.0f}%) with high business value")
            reasons.append("Requires human review")

            return "escalate", confidence, "; ".join(reasons)

        # 기본: 검토 예약
        # v27.0: 데이터 기반 confidence (기존 하드코딩 0.5 제거)
        confidence = (quality_score + evidence_strength) / 200
        confidence = max(0.3, min(0.7, confidence))
        reasons.append(f"Quality: {quality_score:.0f}%")
        reasons.append(f"Evidence: {evidence_strength:.0f}%")
        reasons.append(f"Risk: {risk_level}")
        reasons.append("Scheduled for periodic review")

        return "schedule_review", confidence, "; ".join(reasons)


# === Risk Assessment Framework ===

class RiskAssessor:
    """
    리스크 평가 프레임워크

    NIST Risk Management 기반:
    - Likelihood × Impact = Risk Score
    - Risk Score를 Risk Level로 매핑
    """

    def __init__(self, domain: str = "healthcare"):
        self.domain = domain

        # 도메인별 민감도 매핑
        self.sensitivity_map = {
            "healthcare": {
                "patient": 5,
                "diagnosis": 5,
                "medication": 4,
                "procedure": 4,
                "claim": 4,
                "provider": 3,
                "encounter": 3,
                "facility": 2,
            }
        }

        # 리스크 레벨 경계
        self.risk_boundaries = {
            "low": (1, 5),
            "medium": (6, 12),
            "high": (13, 20),
            "critical": (21, 25),
        }

    def assess_concept_risk(
        self,
        concept: Dict[str, Any],
    ) -> List[RiskAssessment]:
        """
        개념에 대한 리스크 평가

        Args:
            concept: 개념 정보

        Returns:
            리스크 평가 목록
        """
        risks = []
        concept_id = concept.get("concept_id", "")
        name = concept.get("name", "").lower()
        confidence = concept.get("confidence", 0.5)

        # 1. 데이터 품질 리스크
        quality_risk = self._assess_data_quality_risk(concept)
        if quality_risk:
            risks.append(RiskAssessment(
                item_id=concept_id,
                item_type="concept",
                category="data_quality",
                **quality_risk
            ))

        # 2. 컴플라이언스 리스크 (민감 데이터)
        compliance_risk = self._assess_compliance_risk(concept)
        if compliance_risk:
            risks.append(RiskAssessment(
                item_id=concept_id,
                item_type="concept",
                category="compliance",
                **compliance_risk
            ))

        # 3. 운영 리스크
        operational_risk = self._assess_operational_risk(concept)
        if operational_risk:
            risks.append(RiskAssessment(
                item_id=concept_id,
                item_type="concept",
                category="operational",
                **operational_risk
            ))

        return risks

    def _assess_data_quality_risk(self, concept: Dict[str, Any]) -> Optional[Dict]:
        """
        v5.2: 데이터 품질 리스크 평가 - 더 관대한 기본값
        """
        # v5.2: 기본 confidence를 0.6으로 상향 (데이터가 있으면 어느정도 신뢰)
        confidence = concept.get("confidence", 0.6)
        source_evidence = concept.get("source_evidence", {})
        match_rate = source_evidence.get("match_rate", 0.6) if isinstance(source_evidence, dict) else 0.6

        # v5.2: 더 관대한 likelihood 계산 (confidence 0.5에서 likelihood=2)
        # 기존: 5 - int(confidence * 4) = 5 - 2 = 3
        # 변경: 4 - int(confidence * 4) = 4 - 2 = 2
        likelihood = max(1, 4 - int(confidence * 4))  # 1-4 범위

        # v5.2: 더 관대한 impact 계산
        impact = max(1, 4 - int(match_rate * 4))  # 1-4 범위

        risk_score = likelihood * impact

        # v5.2: 더 높은 임계값 (8 이하는 무시)
        if risk_score <= 8:
            return None  # 낮은/중간 리스크는 생략

        return {
            "likelihood": likelihood,
            "impact": impact,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "mitigation": f"Improve data quality validation. Current confidence: {confidence:.0%}",
        }

    def _assess_compliance_risk(self, concept: Dict[str, Any]) -> Optional[Dict]:
        """컴플라이언스 리스크 평가"""
        name = concept.get("name", "").lower()
        sensitivity_map = self.sensitivity_map.get(self.domain, {})

        # 민감도 결정
        sensitivity = 1
        for keyword, level in sensitivity_map.items():
            if keyword in name:
                sensitivity = max(sensitivity, level)

        if sensitivity <= 2:
            return None  # 낮은 민감도는 생략

        # 민감 데이터는 기본적으로 높은 컴플라이언스 리스크
        likelihood = 3  # 규정 위반 가능성
        impact = sensitivity  # 민감도에 따른 영향

        risk_score = likelihood * impact

        return {
            "likelihood": likelihood,
            "impact": impact,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "mitigation": f"Ensure HIPAA compliance for {concept.get('name')}. Implement access controls and audit logging.",
        }

    def _assess_operational_risk(self, concept: Dict[str, Any]) -> Optional[Dict]:
        """
        v5.2: 운영 리스크 평가 - 소스 테이블 수에 덜 민감
        """
        source_tables = concept.get("source_tables", [])
        concept_type = concept.get("concept_type", "")

        # v5.2: 소스 테이블 2개까지는 정상으로 간주
        complexity = max(0, len(source_tables) - 1)  # 1개는 0, 2개는 1
        complexity = min(complexity, 4)

        if complexity <= 1:
            return None

        # v5.2: Object Type 배수 제거 (엔티티가 있는게 정상)
        likelihood = min(complexity, 4)  # 최대 4
        impact = 2  # v5.2: 운영 영향 감소 (3 -> 2)

        risk_score = likelihood * impact

        # v5.2: 임계값 상향 (10 이하 무시)
        if risk_score <= 10:
            return None

        return {
            "likelihood": likelihood,
            "impact": impact,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "mitigation": f"Monitor {len(source_tables)} source tables for changes. Implement data lineage tracking.",
        }

    def _get_risk_level(self, score: int) -> str:
        """리스크 점수를 레벨로 변환"""
        for level, (low, high) in self.risk_boundaries.items():
            if low <= score <= high:
                return level
        return "medium"


# === Priority Calculator ===

class PriorityCalculator:
    """
    우선순위 계산기

    Impact × Urgency / Effort = Priority Score
    """

    def __init__(self):
        # 가중치
        self.weights = {
            "impact": 0.4,
            "urgency": 0.3,
            "effort_inverse": 0.3,  # 낮은 노력이 높은 우선순위
        }

        # 레벨 점수 매핑
        self.level_scores = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
        }

    def calculate_priority(
        self,
        action: Dict[str, Any],
    ) -> Tuple[str, float]:
        """
        액션 우선순위 계산

        Args:
            action: 액션 정보

        Returns:
            (우선순위 레벨, 점수)
        """
        # 영향도
        impact = action.get("impact", "medium")
        impact_score = self.level_scores.get(impact, 3)

        # 노력 (역수)
        effort = action.get("effort", "medium")
        effort_score = 6 - self.level_scores.get(effort, 3)  # 역수 변환

        # 긴급도 (리스크 기반)
        risk = action.get("risk_if_not_done", "medium")
        urgency_score = self.level_scores.get(risk, 3)

        # 가중 점수
        priority_score = (
            impact_score * self.weights["impact"] +
            urgency_score * self.weights["urgency"] +
            effort_score * self.weights["effort_inverse"]
        )

        # 점수를 레벨로 변환
        if priority_score >= 4.0:
            return "critical", priority_score
        elif priority_score >= 3.2:
            return "high", priority_score
        elif priority_score >= 2.4:
            return "medium", priority_score
        else:
            return "low", priority_score

    def sort_by_priority(
        self,
        actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """우선순위로 정렬"""
        scored_actions = []
        for action in actions:
            priority_level, priority_score = self.calculate_priority(action)
            action["calculated_priority"] = priority_level
            action["priority_score"] = priority_score
            scored_actions.append(action)

        return sorted(scored_actions, key=lambda x: x["priority_score"], reverse=True)

    def identify_quick_wins(
        self,
        actions: List[Dict[str, Any]],
    ) -> List[str]:
        """퀵윈 식별 (높은 영향, 낮은 노력)"""
        quick_wins = []

        for action in actions:
            impact = action.get("impact", "medium")
            effort = action.get("effort", "medium")

            if impact in ["high", "critical"] and effort in ["low", "medium"]:
                quick_wins.append(action.get("action_id", ""))

        return quick_wins


# === Policy Generator ===

class PolicyGenerator:
    """
    정책 규칙 생성기

    개념 유형에 따른 표준 정책 템플릿 적용
    """

    def __init__(self, domain: str = "healthcare"):
        self.domain = domain

        # 정책 템플릿
        self.policy_templates = {
            "data_quality": [
                {
                    "pattern": "null_check",
                    "condition": "{field} IS NOT NULL",
                    "severity": "high",
                    "action": "Alert data quality team",
                },
                {
                    "pattern": "completeness",
                    "condition": "COUNT(DISTINCT {field}) > 0",
                    "severity": "medium",
                    "action": "Log warning",
                },
            ],
            "access_control": [
                {
                    "pattern": "phi_protection",
                    "keywords": ["patient", "ssn", "diagnosis", "medication"],
                    "access_level": "restricted",
                    "audit_required": True,
                },
            ],
            "monitoring": [
                {
                    "pattern": "row_count",
                    "metric": "row_count",
                    "threshold_type": "deviation",
                    "threshold_value": 20,  # 20% 변동
                },
            ],
        }

    def generate_policies(
        self,
        concept: Dict[str, Any],
    ) -> List[PolicyRule]:
        """
        개념에 대한 정책 생성

        Args:
            concept: 개념 정보

        Returns:
            정책 규칙 목록
        """
        policies = []
        concept_id = concept.get("concept_id", "")
        concept_type = concept.get("concept_type", "")
        name = concept.get("name", "")

        # 1. 데이터 품질 정책
        quality_policies = self._generate_quality_policies(concept)
        policies.extend(quality_policies)

        # 2. 접근 제어 정책
        access_policies = self._generate_access_policies(concept)
        policies.extend(access_policies)

        # 3. 모니터링 정책
        monitoring_policies = self._generate_monitoring_policies(concept)
        policies.extend(monitoring_policies)

        return policies

    def _generate_quality_policies(
        self,
        concept: Dict[str, Any],
    ) -> List[PolicyRule]:
        """데이터 품질 정책 생성"""
        policies = []
        concept_id = concept.get("concept_id", "")
        name = concept.get("name", "")
        definition = concept.get("definition", {})

        # 필수 필드에 대한 null 체크
        attributes = definition.get("attributes", []) if isinstance(definition, dict) else []

        for i, attr in enumerate(attributes[:5]):  # 상위 5개 속성만
            if isinstance(attr, str):
                attr_name = attr
            elif isinstance(attr, dict):
                attr_name = attr.get("name", f"attr_{i}")
            else:
                continue

            policies.append(PolicyRule(
                rule_id=f"dq_{concept_id}_{attr_name}_null",
                rule_name=f"Null check for {name}.{attr_name}",
                rule_type="data_quality",
                target_concept=concept_id,
                condition=f"{attr_name} IS NOT NULL",
                severity="high",
                action_on_violation=f"Alert: Null value detected in {attr_name}",
            ))

        return policies

    def _generate_access_policies(
        self,
        concept: Dict[str, Any],
    ) -> List[PolicyRule]:
        """접근 제어 정책 생성"""
        policies = []
        concept_id = concept.get("concept_id", "")
        name = concept.get("name", "").lower()

        # 민감 데이터 확인
        phi_keywords = ["patient", "ssn", "diagnosis", "medication", "medical"]
        is_phi = any(kw in name for kw in phi_keywords)

        if is_phi:
            policies.append(PolicyRule(
                rule_id=f"ac_{concept_id}_phi",
                rule_name=f"PHI protection for {concept.get('name')}",
                rule_type="access_control",
                target_concept=concept_id,
                condition="role IN ('physician', 'nurse', 'admin') AND has_hipaa_training = true",
                severity="critical",
                action_on_violation="Block access and log security event",
            ))

        return policies

    def _generate_monitoring_policies(
        self,
        concept: Dict[str, Any],
    ) -> List[PolicyRule]:
        """모니터링 정책 생성"""
        policies = []
        concept_id = concept.get("concept_id", "")
        concept_type = concept.get("concept_type", "")

        # Object Type에 대해서만 모니터링
        if concept_type == "object_type":
            policies.append(PolicyRule(
                rule_id=f"mon_{concept_id}_rowcount",
                rule_name=f"Row count monitoring for {concept.get('name')}",
                rule_type="monitoring",
                target_concept=concept_id,
                condition="ABS(current_count - previous_count) / previous_count > 0.2",
                severity="medium",
                action_on_violation="Alert: Significant row count change detected (>20%)",
            ))

        return policies
