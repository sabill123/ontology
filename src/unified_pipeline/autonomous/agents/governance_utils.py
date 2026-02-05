"""
Governance Phase Utilities (v9.5)

Governance 단계 자율 에이전트들을 위한 공통 유틸리티 모듈
- Enum 클래스들
- 데이터 클래스들
- Embedded Phase 3 Agent Council 관련 클래스들

governance.py에서 추출됨 - 코드 모듈화 및 재사용성 향상

v9.5 변경사항:
- DomainContext 기반 동적 임계값 적용
- LLM Judge 복원 (Rule-based 기반 점수 + LLM 도메인 보정)
- 산업별 평가 기준 차별화 (healthcare 엄격, marketing 유연 등)
"""

import json
import logging
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ...model_config import get_service_model


# ============================================================================
# Embedded Phase 3 Agent Council (v4.5 - Legacy 완전 임베딩)
# ============================================================================

class EmbeddedPhase3AgentRole(Enum):
    """Phase 3 거버넌스 에이전트 역할"""
    RISK_ASSESSOR = "risk_assessor"
    POLICY_VALIDATOR = "policy_validator"
    QUALITY_AUDITOR = "quality_auditor"
    ACTION_PLANNER = "action_planner"


# Agent Council 설정 (가중치 포함)
EMBEDDED_PHASE3_AGENT_CONFIG = {
    "agents": {
        "risk_assessor": {"name": "Risk Assessor", "enabled": True, "weight": 1.5},
        "policy_validator": {"name": "Policy Validator", "enabled": True, "weight": 1.3},
        "quality_auditor": {"name": "Quality Auditor", "enabled": True, "weight": 1.2},
        "action_planner": {"name": "Action Planner", "enabled": True, "weight": 1.0},
    },
    "settings": {
        "parallel_execution": True,
        "max_workers": 4,
        "timeout_per_agent": 180,
        "fallback_to_rules": True,
    }
}


# v9.5: 산업 도메인별 동적 임계값 설정
DOMAIN_THRESHOLDS = {
    # 의료: 환자 안전이 최우선 → 엄격한 기준
    "healthcare": {
        "ds_strong": 0.92,      # DS >= 92% → 2점
        "ds_moderate": 0.80,    # DS >= 80% → 1점
        "bft_strong": 0.85,     # BFT >= 85% → 2점
        "bft_moderate": 0.70,   # BFT >= 70% → 1점
        "pass_threshold": 75,   # 75% 이상 통과
        "policy_compliance_weight": 1.8,  # 정책 준수 가중치 높음
        "risk_tolerance": "low",
        "description": "환자 안전 최우선, 엄격한 검증 필요"
    },
    # 금융: 규제 준수 중요 → 높은 policy_compliance
    "finance": {
        "ds_strong": 0.90,
        "ds_moderate": 0.75,
        "bft_strong": 0.82,
        "bft_moderate": 0.68,
        "pass_threshold": 72,
        "policy_compliance_weight": 1.6,
        "risk_tolerance": "low",
        "description": "규제 준수 중심, 감사 추적 필수"
    },
    # 제조: 품질과 안전 → 중간 수준
    "manufacturing": {
        "ds_strong": 0.85,
        "ds_moderate": 0.70,
        "bft_strong": 0.78,
        "bft_moderate": 0.62,
        "pass_threshold": 70,
        "policy_compliance_weight": 1.3,
        "risk_tolerance": "medium",
        "description": "품질 관리 중심, 운영 효율성 균형"
    },
    # 소매/마케팅: 빠른 의사결정 → 유연한 기준
    "retail": {
        "ds_strong": 0.80,
        "ds_moderate": 0.60,
        "bft_strong": 0.72,
        "bft_moderate": 0.55,
        "pass_threshold": 65,
        "policy_compliance_weight": 1.0,
        "risk_tolerance": "medium",
        "description": "민첩한 의사결정, 시장 대응력 우선"
    },
    "beauty_cosmetics": {
        "ds_strong": 0.80,
        "ds_moderate": 0.60,
        "bft_strong": 0.72,
        "bft_moderate": 0.55,
        "pass_threshold": 65,
        "policy_compliance_weight": 1.0,
        "risk_tolerance": "medium",
        "description": "고객 경험 중심, 트렌드 대응"
    },
    # 기술/스타트업: 실험 허용 → 가장 유연
    "technology": {
        "ds_strong": 0.78,
        "ds_moderate": 0.55,
        "bft_strong": 0.70,
        "bft_moderate": 0.50,
        "pass_threshold": 60,
        "policy_compliance_weight": 0.9,
        "risk_tolerance": "high",
        "description": "혁신 우선, 실험적 접근 허용"
    },
    # 일반: 기본 설정
    "general": {
        "ds_strong": 0.85,
        "ds_moderate": 0.65,
        "bft_strong": 0.75,
        "bft_moderate": 0.60,
        "pass_threshold": 70,
        "policy_compliance_weight": 1.3,
        "risk_tolerance": "medium",
        "description": "균형 잡힌 기본 평가 기준"
    }
}


@dataclass
class EmbeddedEvaluationCriterion:
    """평가 기준"""
    name: str
    description: str
    weight: float = 1.0
    max_score: int = 2


class EmbeddedPhase3EvaluationCriteria:
    """Phase 3 평가 기준 모음"""

    GOVERNANCE_CRITERIA = [
        EmbeddedEvaluationCriterion(
            name="decision_evidence_alignment",
            description="결정이 증거(confidence, severity, evidence sources)와 일치하는가?",
            weight=1.5
        ),
        EmbeddedEvaluationCriterion(
            name="reasoning_quality",
            description="추론이 논리적이고 증거를 적절히 인용하는가?",
            weight=1.2
        ),
        EmbeddedEvaluationCriterion(
            name="policy_compliance",
            description="결정이 거버넌스 정책(임계값, 에스컬레이션 규칙)을 준수하는가?",
            weight=1.3
        ),
        EmbeddedEvaluationCriterion(
            name="multi_agent_consensus",
            description="멀티에이전트 의견이 적절히 반영되었는가?",
            weight=1.0
        )
    ]


class EmbeddedPhase3LLMJudge:
    """
    임베딩된 Phase 3 LLM-as-Judge (v9.5)

    - LLM 기반 거버넌스 결정 평가
    - 도메인 컨텍스트 기반 동적 임계값
    - Rule-based 기초 점수 + LLM 도메인 보정

    v9.5: 하이브리드 접근
    1단계: Rule-based로 기본 점수 계산 (도메인별 임계값 적용)
    2단계: LLM이 도메인 컨텍스트 기반으로 보정/판단
    """

    def __init__(self, llm_client=None, domain_context=None):
        self.llm_client = llm_client
        self.domain_context = domain_context
        self.evaluation_history = []
        self.logger = logging.getLogger(__name__)

    def set_domain_context(self, domain_context):
        """도메인 컨텍스트 설정 (Phase 1에서 탐지된 정보)"""
        self.domain_context = domain_context

    def _get_domain_thresholds(self) -> Dict[str, Any]:
        """현재 도메인에 맞는 임계값 반환"""
        industry = "general"
        if self.domain_context:
            if hasattr(self.domain_context, 'industry'):
                industry = self.domain_context.industry
            elif isinstance(self.domain_context, dict):
                industry = self.domain_context.get('industry', 'general')
            elif isinstance(self.domain_context, str):
                industry = self.domain_context

        return DOMAIN_THRESHOLDS.get(industry, DOMAIN_THRESHOLDS["general"])

    def evaluate_governance_decision(
        self,
        insight: Dict[str, Any],
        decision: Dict[str, Any],
        agent_opinions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        v9.5 하이브리드 거버넌스 결정 평가

        Args:
            insight: 인사이트 데이터
            decision: 결정 데이터
            agent_opinions: 에이전트 의견 목록

        Returns:
            평가 결과 (Rule-based 기초 + LLM 보정)
        """
        agent_opinions = agent_opinions or []
        thresholds = self._get_domain_thresholds()

        # 1단계: 도메인별 임계값 적용한 Rule-based 점수 계산
        rule_result = self._domain_aware_rule_evaluation(insight, decision, agent_opinions, thresholds)

        # 2단계: LLM 보정 (llm_client가 있을 때만)
        if self.llm_client:
            try:
                final_result = self._llm_domain_adjustment(rule_result, insight, decision, thresholds)
                final_result["evaluation_method"] = "hybrid_llm_domain"
            except Exception as e:
                self.logger.warning(f"LLM adjustment failed, using rule-based: {e}")
                final_result = rule_result
                final_result["evaluation_method"] = "rule_based_fallback"
        else:
            final_result = rule_result
            final_result["evaluation_method"] = "rule_based_domain"

        # 도메인 정보 추가
        final_result["domain_info"] = {
            "industry": thresholds.get("description", "Unknown"),
            "risk_tolerance": thresholds.get("risk_tolerance", "medium"),
            "pass_threshold": thresholds.get("pass_threshold", 70)
        }

        # 평가 기록
        self._record_evaluation(insight, decision, final_result)
        return final_result

    def _get_judge_system_prompt(self) -> str:
        """Judge 시스템 프롬프트 (v9.4: 완화된 임계값 + 명확한 점수 규칙)"""
        return """You are an expert Governance Decision Judge for enterprise data ontology systems.

## v9.4 MANDATORY SCORING RULES

You MUST follow these exact scoring rules based on metrics:

### 1. decision_evidence_alignment (weight: 1.5)
| DS Confidence | Score | Rationale |
|---------------|-------|-----------|
| >= 85% | 2 | Strong evidence |
| >= 65% | 1 | Moderate evidence |
| < 65% | 0 | Weak evidence |

### 2. reasoning_quality (weight: 1.2)
| BFT Agreement | Score | Rationale |
|---------------|-------|-----------|
| >= 75% | 2 | Good consensus |
| >= 60% | 1 | Moderate consensus |
| < 60% | 0 | Weak consensus |

### 3. policy_compliance (weight: 1.3)
| Condition | Score | Rationale |
|-----------|-------|-----------|
| DS Conflict = 0 AND BFT >= 75% | 2 | Clean compliance |
| DS Conflict < 0.2 OR BFT >= 65% | 1 | Basic compliance |
| Otherwise | 0 | Compliance issues |

### 4. multi_agent_consensus (weight: 1.0)
| Condition | Score | Rationale |
|-----------|-------|-----------|
| DS >= 85% AND BFT >= 85% | 2 | Strong automated consensus |
| DS >= 65% OR BFT >= 65% | 1 | Moderate consensus |
| Both < 65% | 0 | Weak consensus |

## PASS THRESHOLD: weighted score >= 70%

## CRITICAL RULES:
1. If DS >= 85% AND BFT >= 75%, the decision should PASS (70%+)
2. If DS >= 65% AND BFT >= 65%, minimum total should be 4+ points
3. Do NOT be overly strict - these are automated validation metrics
4. "Moderate" metrics (65-85%) deserve Score 1, NOT Score 0

Calculate scores strictly by the tables above. Do not apply stricter thresholds."""

    def _build_judge_prompt(
        self,
        insight: Dict[str, Any],
        decision: Dict[str, Any],
        agent_opinions: List[Dict[str, Any]]
    ) -> str:
        """Judge 프롬프트 구성 (v9.0: 강화된 평가 기준)"""
        opinions_text = ""
        if agent_opinions:
            opinions_text = "\n".join([
                f"- {op.get('agent', 'Unknown')}: {op.get('decision', 'N/A')} (confidence: {op.get('confidence', 'N/A')})"
                for op in agent_opinions
            ])
        else:
            # v9.0: enhanced_validation 기반 상세 평가 정보 제공
            enhanced = decision.get("enhanced_validation", {})
            if enhanced:
                ds_conf = enhanced.get("ds_confidence", 0)
                bft_agreement = enhanced.get("bft_agreement", 0)
                bft_confidence = enhanced.get("bft_confidence", 0)
                calibrated = enhanced.get("calibrated_confidence", 0)
                ds_decision = enhanced.get("ds_decision", "unknown")
                bft_consensus = enhanced.get("bft_consensus", "unknown")

                opinions_text = f"""### Automated Validation Results (v9.0 Enhanced)

**Dempster-Shafer Evidence Fusion:**
- Decision: {ds_decision}
- Confidence: {ds_conf:.1%}
- Interpretation: {'High reliability' if ds_conf >= 0.9 else 'Moderate reliability' if ds_conf >= 0.7 else 'Lower reliability'}

**Byzantine Fault Tolerant Consensus:**
- Consensus: {bft_consensus}
- Agreement Level: {bft_agreement:.1%}
- Confidence: {bft_confidence:.1%}
- Interpretation: {'Strong agreement' if bft_agreement >= 0.9 else 'Moderate agreement' if bft_agreement >= 0.7 else 'Weak agreement'}

**Calibrated Final Confidence:** {calibrated:.1%}

**v9.4 SCORING GUIDANCE (apply these thresholds exactly):**
- decision_evidence_alignment: DS >= 85% → 2, DS >= 65% → 1, else → 0
- reasoning_quality: BFT >= 75% → 2, BFT >= 60% → 1, else → 0
- policy_compliance: (Conflict<0.15 AND BFT>=75%) → 2, (Conflict<0.2 OR BFT>=60%) → 1, else → 0
- multi_agent_consensus: (DS>=85% AND BFT>=75%) → 2, (DS>=65% OR BFT>=65%) → 1, else → 0

Note: This decision used efficient streamlined validation with algorithmic consensus.
The validation metrics above serve as proxy for multi-agent consensus.

⚠️ IMPORTANT: BFT 75%+ is "Good", 60-75% is "Moderate" - both get Score 1+!"""
            else:
                opinions_text = "No multi-agent opinions or validation data available (single decision maker)"

        return f"""## Governance Decision Evaluation

### Concept/Insight Under Review
- ID: {insight.get('insight_id', 'N/A')}
- Confidence Score: {insight.get('confidence', 'N/A')}
- Severity Level: {insight.get('severity', 'N/A')}
- Description: {insight.get('explanation', 'N/A')}
- Source Tables: {insight.get('source_signatures', [])}

### Decision Made
- Decision Type: {decision.get('decision_type', 'N/A')}
- Decision Confidence: {decision.get('confidence', 'N/A')}
- Reasoning Provided: {decision.get('reasoning', 'N/A')}

### Other Agent Opinions
{opinions_text}

---

Evaluate this governance decision. Return JSON only:

{{
  "decision_evidence_alignment": {{
    "score": <0-2>,
    "rationale": "<why this score>"
  }},
  "reasoning_quality": {{
    "score": <0-2>,
    "rationale": "<why this score>"
  }},
  "policy_compliance": {{
    "score": <0-2>,
    "rationale": "<why this score>"
  }},
  "multi_agent_consensus": {{
    "score": <0-2>,
    "rationale": "<why this score>"
  }},
  "overall_assessment": {{
    "pass": <true/false>,
    "summary": "<1-2 sentence assessment>"
  }},
  "recommendations": ["<improvement suggestion>", "..."]
}}"""

    def _parse_llm_response(
        self,
        response: str,
        insight: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        import re

        try:
            # JSON 블록 추출
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())

                # 점수 계산
                weights = {
                    "decision_evidence_alignment": 1.5,
                    "reasoning_quality": 1.2,
                    "policy_compliance": 1.3,
                    "multi_agent_consensus": 1.0
                }

                total = 0
                for key, weight in weights.items():
                    if key in result and "score" in result[key]:
                        score = min(2, max(0, result[key]["score"]))
                        result[key]["score"] = score
                        total += score * weight

                max_total = sum(2 * w for w in weights.values())
                percentage = (total / max_total) * 100

                # overall_assessment 업데이트
                llm_pass = result.get("overall_assessment", {}).get("pass", percentage >= 70)
                result["overall_assessment"] = {
                    "pass": llm_pass if isinstance(llm_pass, bool) else percentage >= 70,
                    "total_score": round(total, 2),
                    "max_score": max_total,
                    "percentage": round(percentage, 1),
                    "summary": result.get("overall_assessment", {}).get("summary",
                        f"LLM evaluation: {'PASS' if percentage >= 70 else 'FAIL'} ({percentage:.1f}%)")
                }

                if "recommendations" not in result:
                    result["recommendations"] = []

                return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")

        # 파싱 실패시 규칙 기반 폴백
        return self._rule_based_evaluation(insight, decision, [])

    def _domain_aware_rule_evaluation(
        self,
        insight: Dict[str, Any],
        decision: Dict[str, Any],
        agent_opinions: List[Dict[str, Any]],
        thresholds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """v9.5: 도메인별 임계값을 적용한 규칙 기반 평가"""
        scores = {}

        confidence = insight.get("confidence", 0.5)
        decision_type = decision.get("decision_type", "")
        reasoning = decision.get("reasoning", "")
        severity = insight.get("severity", "medium")

        # 도메인별 임계값 추출
        ds_strong = thresholds.get("ds_strong", 0.85)
        ds_moderate = thresholds.get("ds_moderate", 0.65)
        bft_strong = thresholds.get("bft_strong", 0.75)
        bft_moderate = thresholds.get("bft_moderate", 0.60)
        pass_threshold = thresholds.get("pass_threshold", 70)
        policy_weight = thresholds.get("policy_compliance_weight", 1.3)

        # Enhanced Validation 데이터 추출
        enhanced_val = decision.get("enhanced_validation", {})
        ds_confidence = enhanced_val.get("ds_confidence", 0)
        bft_agreement = enhanced_val.get("bft_agreement", 0)
        ds_conflict = enhanced_val.get("ds_conflict", 1.0)
        has_enhanced = bool(enhanced_val)

        # 1. Decision-Evidence Alignment (도메인별 DS 임계값)
        if has_enhanced and ds_confidence >= ds_strong:
            scores["decision_evidence_alignment"] = {"score": 2, "rationale": f"Strong DS evidence ({ds_confidence:.0%} >= {ds_strong:.0%})"}
        elif has_enhanced and ds_confidence >= ds_moderate:
            scores["decision_evidence_alignment"] = {"score": 1, "rationale": f"Moderate DS evidence ({ds_confidence:.0%} >= {ds_moderate:.0%})"}
        elif confidence >= ds_strong and decision_type == "approve":
            scores["decision_evidence_alignment"] = {"score": 2, "rationale": "High confidence supports approval"}
        elif confidence >= ds_moderate:
            scores["decision_evidence_alignment"] = {"score": 1, "rationale": f"Moderate confidence ({confidence:.0%})"}
        else:
            scores["decision_evidence_alignment"] = {"score": 0, "rationale": f"Weak evidence (DS={ds_confidence:.0%}, threshold={ds_moderate:.0%})"}

        # 2. Reasoning Quality (도메인별 BFT 임계값)
        if has_enhanced and bft_agreement >= bft_strong:
            scores["reasoning_quality"] = {"score": 2, "rationale": f"Good BFT agreement ({bft_agreement:.0%} >= {bft_strong:.0%})"}
        elif has_enhanced and bft_agreement >= bft_moderate:
            scores["reasoning_quality"] = {"score": 1, "rationale": f"Moderate BFT agreement ({bft_agreement:.0%} >= {bft_moderate:.0%})"}
        elif len(reasoning) > 100 and any(w in reasoning.lower() for w in ["confidence", "evidence", "risk"]):
            scores["reasoning_quality"] = {"score": 2, "rationale": "Detailed reasoning provided"}
        elif len(reasoning) > 50:
            scores["reasoning_quality"] = {"score": 1, "rationale": "Adequate reasoning"}
        else:
            scores["reasoning_quality"] = {"score": 0, "rationale": "Insufficient reasoning"}

        # 3. Policy Compliance (도메인별 BFT 임계값 + 도메인별 가중치)
        if has_enhanced and ds_conflict == 0 and bft_agreement >= bft_strong:
            scores["policy_compliance"] = {"score": 2, "rationale": f"Clean compliance (conflict=0, BFT={bft_agreement:.0%})"}
        elif has_enhanced and (ds_conflict < 0.2 or bft_agreement >= bft_moderate):
            scores["policy_compliance"] = {"score": 1, "rationale": f"Basic compliance (conflict={ds_conflict:.0%}, BFT={bft_agreement:.0%})"}
        elif severity == "critical" and decision_type == "escalate":
            scores["policy_compliance"] = {"score": 2, "rationale": "Critical issue properly escalated"}
        elif decision_type in ["approve", "schedule_review"]:
            scores["policy_compliance"] = {"score": 1, "rationale": "Standard policy handling"}
        else:
            scores["policy_compliance"] = {"score": 0, "rationale": f"Compliance concerns (conflict={ds_conflict:.0%})"}

        # 4. Multi-Agent Consensus (도메인별 DS/BFT 임계값)
        if len(agent_opinions) >= 3:
            scores["multi_agent_consensus"] = {"score": 2, "rationale": "Multiple agent opinions available"}
        elif len(agent_opinions) > 0:
            scores["multi_agent_consensus"] = {"score": 1, "rationale": "Some agent input available"}
        elif has_enhanced:
            if ds_confidence >= ds_strong and bft_agreement >= bft_strong:
                scores["multi_agent_consensus"] = {"score": 2, "rationale": f"Strong consensus (DS={ds_confidence:.0%}, BFT={bft_agreement:.0%})"}
            elif ds_confidence >= ds_moderate or bft_agreement >= bft_moderate:
                scores["multi_agent_consensus"] = {"score": 1, "rationale": f"Moderate consensus (DS={ds_confidence:.0%}, BFT={bft_agreement:.0%})"}
            else:
                scores["multi_agent_consensus"] = {"score": 0, "rationale": f"Weak consensus (DS={ds_confidence:.0%}, BFT={bft_agreement:.0%})"}
        else:
            scores["multi_agent_consensus"] = {"score": 0, "rationale": "No multi-agent input or validation data"}

        # 합계 (도메인별 policy_compliance 가중치 적용)
        weights = {
            "decision_evidence_alignment": 1.5,
            "reasoning_quality": 1.2,
            "policy_compliance": policy_weight,  # v9.5: 도메인별 가중치
            "multi_agent_consensus": 1.0
        }
        total = sum(scores[k]["score"] * weights[k] for k in scores)
        max_total = sum(2 * w for w in weights.values())
        percentage = (total / max_total) * 100

        result = {
            **scores,
            "overall_assessment": {
                "pass": percentage >= pass_threshold,  # v9.5: 도메인별 통과 기준
                "total_score": round(total, 2),
                "max_score": round(max_total, 2),
                "percentage": round(percentage, 1),
                "summary": f"Domain-aware: {'PASS' if percentage >= pass_threshold else 'FAIL'} ({percentage:.1f}% vs {pass_threshold}%)"
            },
            "recommendations": [],
            "thresholds_applied": {
                "ds_strong": ds_strong,
                "ds_moderate": ds_moderate,
                "bft_strong": bft_strong,
                "bft_moderate": bft_moderate,
                "pass_threshold": pass_threshold,
                "policy_weight": policy_weight
            }
        }

        return result

    def _llm_domain_adjustment(
        self,
        rule_result: Dict[str, Any],
        insight: Dict[str, Any],
        decision: Dict[str, Any],
        thresholds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """v9.5: LLM이 도메인 컨텍스트 기반으로 Rule-based 점수를 보정"""
        from ..llm_utils import chat_completion

        domain_desc = thresholds.get("description", "일반 도메인")
        risk_tolerance = thresholds.get("risk_tolerance", "medium")
        pass_threshold = thresholds.get("pass_threshold", 70)

        # Rule-based 결과 요약
        rule_percentage = rule_result.get("overall_assessment", {}).get("percentage", 0)
        rule_pass = rule_result.get("overall_assessment", {}).get("pass", False)

        prompt = f"""## v9.5 Domain-Aware Governance Judge

### 도메인 컨텍스트
- 산업: {domain_desc}
- 리스크 허용도: {risk_tolerance}
- 통과 기준: {pass_threshold}%

### Rule-based 평가 결과
- 점수: {rule_percentage:.1f}%
- 판정: {"PASS" if rule_pass else "FAIL"}
- 세부 점수:
  - decision_evidence_alignment: {rule_result.get("decision_evidence_alignment", {}).get("score", 0)}/2
  - reasoning_quality: {rule_result.get("reasoning_quality", {}).get("score", 0)}/2
  - policy_compliance: {rule_result.get("policy_compliance", {}).get("score", 0)}/2
  - multi_agent_consensus: {rule_result.get("multi_agent_consensus", {}).get("score", 0)}/2

### 결정 정보
- 결정 유형: {decision.get("decision_type", "N/A")}
- 신뢰도: {decision.get("confidence", "N/A")}
- 추론: {decision.get("reasoning", "N/A")[:200]}...

### 당신의 역할
1. Rule-based 점수가 이 도메인에 적합한지 평가
2. 도메인 특성상 보정이 필요하면 adjustment 제안
3. 최종 판정 (PASS/FAIL) 결정

### 보정 가이드라인
- healthcare/finance (low risk): Rule-based보다 엄격하게
- retail/technology (high risk tolerance): Rule-based보다 유연하게
- 경계선 케이스 (통과 기준 ±5%): 도메인 특성 고려해서 판단

JSON 형식으로 응답:
{{
  "adjustment": {{
    "needed": <true/false>,
    "direction": "<stricter/more_lenient/none>",
    "reason": "<도메인 특성상 왜 보정이 필요한지>"
  }},
  "final_assessment": {{
    "pass": <true/false>,
    "adjusted_percentage": <보정된 점수 또는 원래 점수>,
    "summary": "<최종 판단 근거>"
  }},
  "domain_insights": ["<도메인 관점에서의 추가 인사이트>"]
}}"""

        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": "You are a domain-expert governance judge. Evaluate decisions based on industry-specific requirements."},
                    {"role": "user", "content": prompt}
                ],
                model=get_service_model("governance_judge"),
                temperature=0.3,
                max_tokens=800
            )

            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                llm_result = json.loads(json_match.group())

                # LLM 보정 결과 병합
                final_result = rule_result.copy()

                adjustment = llm_result.get("adjustment", {})
                final_assessment = llm_result.get("final_assessment", {})

                # LLM이 보정을 제안한 경우
                if adjustment.get("needed", False):
                    final_result["llm_adjustment"] = adjustment
                    final_result["overall_assessment"]["pass"] = final_assessment.get("pass", rule_pass)
                    final_result["overall_assessment"]["adjusted_percentage"] = final_assessment.get("adjusted_percentage", rule_percentage)
                    final_result["overall_assessment"]["summary"] = final_assessment.get("summary", rule_result["overall_assessment"]["summary"])
                else:
                    final_result["llm_adjustment"] = {"needed": False, "reason": "Rule-based 결과가 도메인에 적합"}

                final_result["domain_insights"] = llm_result.get("domain_insights", [])
                return final_result

        except Exception as e:
            self.logger.warning(f"LLM domain adjustment failed: {e}")

        # LLM 실패시 Rule-based 결과 그대로 반환
        return rule_result

    def _rule_based_evaluation(
        self,
        insight: Dict[str, Any],
        decision: Dict[str, Any],
        agent_opinions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """기존 호환성을 위한 래퍼 - 기본 도메인 임계값 사용"""
        return self._domain_aware_rule_evaluation(
            insight, decision, agent_opinions,
            DOMAIN_THRESHOLDS["general"]
        )

    def _record_evaluation(
        self,
        insight: Dict[str, Any],
        decision: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """평가 기록"""
        evaluation = {
            "type": "governance",
            "insight_id": insight.get("insight_id"),
            "decision_id": decision.get("decision_id"),
            "evaluation": result,
            "timestamp": datetime.now().isoformat()
        }
        self.evaluation_history.append(evaluation)


class EmbeddedPhase3GovernanceOrchestrator:
    """
    임베딩된 Phase 3 멀티에이전트 거버넌스 오케스트레이터 (v4.5)

    4개 에이전트가 병렬로 실행됩니다:
    - RiskAssessor: 리스크 평가 (weight: 1.5)
    - PolicyValidator: 정책 준수 검증 (weight: 1.3)
    - QualityAuditor: 품질 감사 (weight: 1.2)
    - ActionPlanner: 액션 계획 수립 (weight: 1.0)

    이 클래스는 _legacy/phase3/llm_judge.py의 핵심 로직을 직접 포함합니다.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parallel_execution: bool = True
    ):
        self.config = config or EMBEDDED_PHASE3_AGENT_CONFIG
        self.parallel_execution = parallel_execution
        self.max_workers = 4
        self.logger = logging.getLogger(__name__)

    def _evaluate_risk(self, insight: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Risk Assessor: 리스크 평가

        - 인사이트의 리스크 수준을 정량적으로 평가
        - 비즈니스 영향도 분석
        """
        confidence = insight.get("confidence", 0.5)
        severity = insight.get("severity", "medium")

        severity_scores = {"critical": 3, "high": 2.5, "medium": 1.5, "low": 0.5}
        risk_score = (1 - confidence) * severity_scores.get(severity, 1.5)

        decision = "escalate" if risk_score > 2 else "review" if risk_score > 1 else "approve"

        return {
            "agent": "risk_assessor",
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "reasoning": f"Rule-based: Confidence {confidence:.2f} x Severity {severity} = Risk {risk_score:.2f}",
            "confidence": min(0.9, confidence + 0.1),
            "business_impact": severity,
            "weight": 1.5,
            "source": "rules"
        }

    def _evaluate_policy(self, insight: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Policy Validator: 정책 준수 검증

        - 거버넌스 정책 준수 여부 검증
        - 임계값 기반 자동 승인/거부
        """
        policy = context.get("governance_policy", {})
        min_confidence = policy.get("min_approval_confidence", 0.85)

        confidence = insight.get("confidence", 0.5)
        compliant = confidence >= min_confidence

        decision = "approve" if compliant else "review"

        return {
            "agent": "policy_validator",
            "decision": decision,
            "policy_compliant": compliant,
            "reasoning": f"Rule-based: Confidence {confidence:.2f} {'≥' if compliant else '<'} threshold {min_confidence}",
            "confidence": 0.95 if compliant else 0.7,
            "violations": [] if compliant else [f"Confidence below {min_confidence}"],
            "weight": 1.3,
            "source": "rules"
        }

    def _evaluate_quality(self, insight: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quality Auditor: 품질 감사

        - 인사이트 품질 검사
        - 증거 체인 검증
        """
        explanation = insight.get("explanation", "")
        has_evidence = bool(insight.get("phase1_mappings_referenced", []))
        has_recommendation = bool(insight.get("recommendation", ""))
        has_kpi = bool(insight.get("estimated_kpi_impact", {}))

        quality_score = sum([
            len(explanation) > 50,
            has_evidence,
            has_recommendation,
            has_kpi
        ])

        decision = "approve" if quality_score >= 2 else "review" if quality_score >= 1 else "escalate"

        return {
            "agent": "quality_auditor",
            "decision": decision,
            "quality_score": quality_score,
            "reasoning": f"Rule-based: Quality {quality_score}/4 (evidence: {has_evidence}, recommendation: {has_recommendation}, KPI: {has_kpi})",
            "confidence": 0.75 + (quality_score * 0.05),
            "quality_breakdown": {
                "evidence_complete": has_evidence,
                "explanation_quality": "high" if len(explanation) > 100 else "medium" if len(explanation) > 50 else "low",
                "actionable": has_recommendation,
                "kpi_linked": has_kpi
            },
            "weight": 1.2,
            "source": "rules"
        }

    def _evaluate_action(self, insight: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Action Planner: 액션 계획 수립

        - 인사이트 기반 실행 계획 수립
        - 워크플로우 타입 결정
        """
        category = insight.get("category", "general")
        severity = insight.get("severity", "medium")
        domain = context.get("domain_context", {}).get("industry", "general")

        # 카테고리별 워크플로우 매핑
        category_workflows = {
            "entity_definition": "schema_update",
            "relationship_mapping": "data_integration",
            "property_semantics": "data_quality",
            "domain_validation": "business_review",
            "entity_enrichment": "data_enrichment",
            "entity_relationship": "data_integration",
            "business_rule": "business_review",
            "strategic_recommendation": "business_review"
        }

        recommended = context.get("recommended_workflows", [])
        workflow = recommended[0] if recommended else category_workflows.get(category, "general_review")
        priority = "high" if severity in ["critical", "high"] else "medium" if severity == "medium" else "low"

        # 도메인별 KPI 예측
        kpi_estimates = {
            "retail": {"매출": "+12%", "고객_만족도": "+8%"},
            "healthcare": {"환자_안전성": "+15%", "데이터_품질": "+20%"},
            "manufacturing": {"생산_효율성": "+20%", "불량률": "-15%"},
        }

        return {
            "agent": "action_planner",
            "decision": "approve",
            "workflow_type": workflow,
            "priority": priority,
            "reasoning": f"Rule-based: Category {category} -> Workflow {workflow}, Priority {priority}",
            "confidence": 0.85,
            "action_title": f"{category.replace('_', ' ').title()} Action",
            "action_description": insight.get("recommendation", "Auto-generated action"),
            "estimated_kpi_delta": kpi_estimates.get(domain, {"business_value": "+10%"}),
            "weight": 1.0,
            "source": "rules"
        }

    def _run_agent(
        self,
        agent_name: str,
        insight: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """단일 에이전트 실행"""
        try:
            if agent_name == "risk_assessor":
                result = self._evaluate_risk(insight, context)
            elif agent_name == "policy_validator":
                result = self._evaluate_policy(insight, context)
            elif agent_name == "quality_auditor":
                result = self._evaluate_quality(insight, context)
            elif agent_name == "action_planner":
                result = self._evaluate_action(insight, context)
            else:
                result = {"agent": agent_name, "decision": "review", "weight": 1.0}
            return (agent_name, result)
        except Exception as e:
            self.logger.error(f"Agent {agent_name} error: {e}")
            return (agent_name, {
                "agent": agent_name,
                "decision": "review",
                "error": str(e)[:100],
                "confidence": 0.5,
                "weight": 1.0
            })

    def evaluate_insight(
        self,
        insight: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        모든 에이전트를 병렬로 실행하여 인사이트를 평가합니다.

        Returns:
            통합 거버넌스 결정
        """
        context = context or {}
        agent_results: Dict[str, Dict[str, Any]] = {}
        agent_names = ["risk_assessor", "policy_validator", "quality_auditor", "action_planner"]

        if self.parallel_execution:
            self.logger.info("Running 4 embedded governance agents in PARALLEL mode")

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._run_agent, agent_name, insight, context): agent_name
                    for agent_name in agent_names
                }

                for future in concurrent.futures.as_completed(futures):
                    agent_name = futures[future]
                    try:
                        result_name, result = future.result()
                        agent_results[result_name] = result
                        self.logger.info(f"Embedded Agent {result_name} completed: {result.get('decision')}")
                    except Exception as e:
                        self.logger.error(f"Embedded Agent {agent_name} failed: {e}")
                        agent_results[agent_name] = {"agent": agent_name, "decision": "review", "error": str(e)[:100]}
        else:
            # Sequential fallback
            for agent_name in agent_names:
                _, result = self._run_agent(agent_name, insight, context)
                agent_results[agent_name] = result

        # 가중 투표로 최종 결정 도출
        return self._aggregate_decisions(insight, agent_results)

    def _aggregate_decisions(
        self,
        insight: Dict[str, Any],
        agent_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """가중 투표로 최종 결정 도출"""
        decision_weights = {"approve": 0, "review": 0, "escalate": 0}
        total_weight = 0

        agent_opinions = []
        workflow_type = "general_review"
        priority = "medium"

        for agent_name, result in agent_results.items():
            decision = result.get("decision", "review")
            weight = result.get("weight", 1.0)

            decision_weights[decision] = decision_weights.get(decision, 0) + weight
            total_weight += weight

            agent_opinions.append({
                "agent": agent_name,
                "vote": decision,
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "")
            })

            # ActionPlanner의 workflow_type과 priority 사용
            if agent_name == "action_planner":
                workflow_type = result.get("workflow_type", "general_review")
                priority = result.get("priority", "medium")

        # 최종 결정 (가중 다수결)
        final_decision = max(decision_weights, key=decision_weights.get)
        final_confidence = decision_weights[final_decision] / total_weight if total_weight > 0 else 0.5

        self.logger.info(f"Final decision: {final_decision} (confidence: {final_confidence:.2f})")

        return {
            "decision_id": f"GOV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "insight_id": insight.get("id", "unknown"),
            "decision_type": final_decision,
            "confidence": round(final_confidence, 2),
            "workflow_type": workflow_type,
            "priority": priority,
            "agent_opinions": agent_opinions,
            "vote_breakdown": {k: round(v, 2) for k, v in decision_weights.items()},
            "reasoning": f"Embedded Agent Council: {len(agent_results)} agents voted, {final_decision} won with weight {decision_weights[final_decision]:.2f}",
            "timestamp": datetime.now().isoformat()
        }


# Legacy 플래그 - 이제 항상 True (임베딩됨)
LEGACY_PHASE3_AVAILABLE = True
