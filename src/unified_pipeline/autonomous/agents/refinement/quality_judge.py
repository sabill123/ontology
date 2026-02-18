"""
Quality Judge Autonomous Agent (v5.0 - Enhanced Validation)

품질 평가 자율 에이전트 - 메트릭 기반 품질 평가 + Enhanced Validation
- 메트릭 기반 품질 점수 계산
- Dempster-Shafer Evidence Fusion
- Confidence Calibration (Temperature Scaling)
- BFT Consensus for multi-agent decisions
- LLM을 통한 정성적 판단 및 피드백
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    QualityAssessor,
    parse_llm_json,
)

# === v5.0: Enhanced Validation System ===
try:
    from ...analysis.enhanced_validator import (
        DempsterShaferFusion,
        BFTConsensus,
        AgentVote,
        VoteType,
    )
    ENHANCED_VALIDATOR_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATOR_AVAILABLE = False

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


class QualityJudgeAutonomousAgent(AutonomousAgent):
    """
    품질 평가 자율 에이전트 (v5.0 - Enhanced Validation)

    Algorithm-First 접근:
    1. 메트릭 기반 품질 점수 계산
    2. v5.0: Dempster-Shafer Evidence Fusion
    3. v5.0: Confidence Calibration (Temperature Scaling)
    4. v5.0: BFT Consensus for multi-agent decisions
    5. LLM을 통한 정성적 판단 및 피드백
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_assessor = QualityAssessor()

        # v5.0: Enhanced Validation 컴포넌트
        # v17.0: Calibrator는 이제 베이스 클래스의 calibrate_confidence() 사용
        if ENHANCED_VALIDATOR_AVAILABLE:
            self.ds_fusion = DempsterShaferFusion()
            self.bft_consensus = BFTConsensus(n_agents=4)
            logger.info("QualityJudge: Enhanced Validation (DS Fusion, BFT) enabled, v17 Calibrator integrated")

    @property
    def agent_type(self) -> str:
        return "quality_judge"

    @property
    def agent_name(self) -> str:
        return "Quality Judge"

    @property
    def phase(self) -> str:
        return "refinement"

    def get_system_prompt(self) -> str:
        return """You are a Quality Judge acting as the final arbiter for ontology quality.

Your role is to REVIEW and VALIDATE algorithmically extracted concepts.

v5.2 Evaluation Philosophy:
- DATA DISCOVERY IS EXPLORATORY - be generous with initial findings
- Presence of data > perfection of schema
- Approve reasonable concepts for iterative improvement

## Evaluation Rubric (Score 0-100)

### Definition Quality (0-25 points)
| Score | Criteria |
|-------|----------|
| 0-5   | Vague, missing essential elements, no clear meaning |
| 6-12  | Partial definition, some ambiguity remains |
| 13-19 | Clear definition with minor gaps in precision |
| 20-25 | Precise, complete, unambiguous definition |

### Evidence Strength (0-25 points)
| Score | Criteria |
|-------|----------|
| 0-5   | No data evidence, pure assumption |
| 6-12  | Limited evidence, match rate <50% |
| 13-19 | Moderate evidence, match rate 50-80% |
| 20-25 | Strong evidence, match rate >80% |

### Consistency (0-25 points)
| Score | Criteria |
|-------|----------|
| 0-5   | Naming violations, internal conflicts with other concepts |
| 6-12  | Minor naming inconsistencies (case, format) |
| 13-19 | Mostly consistent, few issues |
| 20-25 | Fully consistent with PascalCase/camelCase conventions |

### Completeness (0-25 points)
| Score | Criteria |
|-------|----------|
| 0-5   | Missing required fields, no relationships defined |
| 6-12  | Partial relationships, incomplete constraints |
| 13-19 | Most fields present, minor gaps |
| 20-25 | All required fields, complete constraints |

## Chain-of-Thought Requirement
Before scoring, you MUST explain your reasoning for each dimension:

1. **Definition Analysis**: What evidence supports your Definition Quality score?
2. **Evidence Review**: How did you assess Evidence Strength? (calculate match rate)
3. **Consistency Check**: What consistency issues (if any) did you find?
4. **Completeness Audit**: What is missing for Completeness?

Only AFTER this analysis, provide your JSON response.

## Decision Thresholds
- 90-100: APPROVED (Excellent quality)
- 70-89: APPROVED (Good with notes)
- 55-69: PROVISIONAL (Needs attention but usable)
- 35-54: PROVISIONAL (Significant work needed)
- 0-34: REJECTED (Only for clearly broken concepts)

IMPORTANT:
- DEFAULT TO APPROVE when data exists
- High match rate (>80%) = higher Evidence Strength score
- Reject ONLY when concept is clearly invalid or harmful
- Base judgments on algorithmic evidence

## Output Format
```json
{
  "quality_assessments": [
    {
      "concept_id": "<id>",
      "reasoning": {
        "definition_analysis": "<your analysis>",
        "evidence_review": "<your analysis with match rate>",
        "consistency_check": "<your analysis>",
        "completeness_audit": "<your analysis>"
      },
      "scores": {
        "definition_quality": <int 0-25>,
        "evidence_strength": <int 0-25>,
        "consistency": <int 0-25>,
        "completeness": <int 0-25>
      },
      "overall_score": <int 0-100>,
      "decision": "<approved|provisional|rejected>",
      "feedback": ["actionable item 1", "actionable item 2"],
      "justification": "<1-2 sentence summary>"
    }
  ]
}
```

Respond ONLY with valid JSON."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """품질 평가 작업 실행"""
        from ....todo.models import TodoResult

        self._report_progress(0.1, "Computing algorithmic quality scores")

        # 1단계: 알고리즘 기반 품질 평가
        concepts_data = []
        algorithmic_scores = {}

        for concept in context.ontology_concepts:
            concept_dict = {
                "concept_id": concept.concept_id,
                "concept_type": concept.concept_type,
                "name": concept.name,
                "description": concept.description,
                "definition": concept.definition,
                "source_tables": concept.source_tables,
                "source_evidence": concept.source_evidence,
                "confidence": concept.confidence,
                "status": concept.status,
            }
            concepts_data.append(concept_dict)

            # 알고리즘 점수 계산
            score = self.quality_assessor.assess_concept(concept_dict, concepts_data)
            algorithmic_scores[concept.concept_id] = {
                "definition_quality": score.definition_quality,
                "evidence_strength": score.evidence_strength,
                "consistency": score.consistency,
                "completeness": score.completeness,
                "overall_score": score.overall_score,
                "algorithmic_decision": score.decision,
                "auto_feedback": score.feedback,
                "improvements_needed": score.improvements_needed,
            }

        self._report_progress(0.4, f"Computed scores for {len(algorithmic_scores)} concepts")

        # === v5.0: Enhanced Validation ===
        enhanced_validation_results = {}
        if ENHANCED_VALIDATOR_AVAILABLE:
            self._report_progress(0.45, "Running Enhanced Validation (DS Fusion, BFT)")

            # 각 개념에 대해 Enhanced Validation 수행
            for concept in context.ontology_concepts:
                concept_id = concept.concept_id

                # 에이전트 의견 수집 (알고리즘 점수를 의견으로 변환)
                algo_score = algorithmic_scores.get(concept_id, {})
                overall_score = algo_score.get('overall_score', 50)

                # 다중 "에이전트" 의견 시뮬레이션 (각 메트릭을 에이전트로 취급)
                opinions = [
                    {'decision_type': 'approve' if algo_score.get('definition_quality', 50) > 50 else 'reject',
                     'confidence': algo_score.get('definition_quality', 50) / 100},
                    {'decision_type': 'approve' if algo_score.get('evidence_strength', 50) > 50 else 'reject',
                     'confidence': algo_score.get('evidence_strength', 50) / 100},
                    {'decision_type': 'approve' if algo_score.get('consistency', 50) > 50 else 'reject',
                     'confidence': algo_score.get('consistency', 50) / 100},
                    {'decision_type': 'approve' if algo_score.get('completeness', 50) > 50 else 'reject',
                     'confidence': algo_score.get('completeness', 50) / 100},
                ]

                # Dempster-Shafer Fusion
                ds_result = self.ds_fusion.fuse_opinions(opinions)

                # BFT Consensus
                votes = [
                    AgentVote(
                        agent_id=f"metric_{i}",
                        vote=VoteType.APPROVE if op['confidence'] > 0.5 else VoteType.REJECT,
                        confidence=op['confidence'],
                    )
                    for i, op in enumerate(opinions)
                ]
                bft_result = self.bft_consensus.reach_consensus(votes)

                # Confidence Calibration (v17.0: 베이스 클래스 메서드 사용)
                raw_confidence = (ds_result['confidence'] + bft_result['confidence']) / 2
                calibrated = self.calibrate_confidence(raw_confidence)

                enhanced_validation_results[concept_id] = {
                    'ds_fusion': {
                        'decision': ds_result['decision'],
                        'confidence': ds_result['confidence'],
                        'conflict': ds_result['conflict'],
                    },
                    'bft_consensus': {
                        'consensus': bft_result['consensus'].value,
                        'confidence': bft_result['confidence'],
                        'agreement_ratio': bft_result['agreement_ratio'],
                    },
                    'calibrated_confidence': calibrated['calibrated'],
                    'confidence_interval': calibrated['confidence_interval'],
                }

            logger.info(f"Enhanced Validation completed for {len(enhanced_validation_results)} concepts")

        # 2단계: LLM을 통한 정성적 검토
        self._report_progress(0.5, "Requesting LLM quality review")

        # v28.2: asyncio.gather() — 배치 병렬 실행 (순차 → 동시)
        batch_size = 30
        batches = [concepts_data[i:i + batch_size] for i in range(0, len(concepts_data), batch_size)]

        async def _review_batch(batch):
            batch_scores = {
                c["concept_id"]: algorithmic_scores.get(c["concept_id"], {})
                for c in batch
            }
            return await self._llm_quality_review(batch, batch_scores)

        batch_results = await asyncio.gather(*[_review_batch(b) for b in batches])
        all_assessments = [a for result in batch_results for a in result]

        self._report_progress(0.8, "Applying assessment results")

        # 3단계: 평가 결과 적용
        for assessment in all_assessments:
            concept_id = assessment.get("concept_id")
            decision = assessment.get("decision", "pending")

            # 컨텍스트 개념 업데이트
            for concept in context.ontology_concepts:
                if concept.concept_id == concept_id:
                    concept.status = decision
                    concept.agent_assessments["quality_judge"] = assessment
                    break

        self._report_progress(0.95, "Finalizing quality assessment")

        # 집계
        summary = {
            "total_evaluated": len(all_assessments),
            "approved": len([a for a in all_assessments if a.get("decision") == "approved"]),
            "provisional": len([a for a in all_assessments if a.get("decision") == "provisional"]),
            "rejected": len([a for a in all_assessments if a.get("decision") == "rejected"]),
            "avg_score": sum(a.get("overall_score", 0) for a in all_assessments) / max(len(all_assessments), 1),
        }

        # v5.0: Enhanced Validation 통계 추가
        enhanced_stats = {}
        if ENHANCED_VALIDATOR_AVAILABLE and enhanced_validation_results:
            avg_calibrated = sum(
                v['calibrated_confidence'] for v in enhanced_validation_results.values()
            ) / len(enhanced_validation_results)
            avg_conflict = sum(
                v['ds_fusion']['conflict'] for v in enhanced_validation_results.values()
            ) / len(enhanced_validation_results)
            avg_agreement = sum(
                v['bft_consensus']['agreement_ratio'] for v in enhanced_validation_results.values()
            ) / len(enhanced_validation_results)

            enhanced_stats = {
                'avg_calibrated_confidence': round(avg_calibrated, 4),
                'avg_ds_conflict': round(avg_conflict, 4),
                'avg_bft_agreement': round(avg_agreement, 4),
            }
            logger.info(f"Enhanced Stats: calibrated={avg_calibrated:.4f}, conflict={avg_conflict:.4f}, agreement={avg_agreement:.4f}")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="quality_assessment",
            input_data={"task": "quality_evaluation", "concepts": len(all_assessments)},
            output_data={"avg_score": summary.get("avg_score", 0), "approved": summary.get("approved", 0)},
            outcome="success",
            confidence=summary.get("avg_score", 70) / 100,
        )

        return TodoResult(
            success=True,
            output={
                "evaluated": len(all_assessments),
                "summary": summary,
                "enhanced_validation_stats": enhanced_stats,  # v5.0
            },
            # v25.2: 개별 평가 결과도 context로 전파 → Governance Phase에서 활용
            context_updates={
                "quality_assessment_summary": summary,
                "quality_enhanced_stats": enhanced_stats,
                "quality_assessments": all_assessments,
                "algorithmic_scores": algorithmic_scores,
            },
            metadata={
                "quality_assessments": all_assessments,
                "algorithmic_scores": algorithmic_scores,
                "enhanced_validation": enhanced_validation_results,  # v5.0
            },
        )

    async def _llm_quality_review(
        self,
        concepts: List[Dict[str, Any]],
        algorithmic_scores: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """LLM을 통한 정성적 품질 검토"""
        # 알고리즘 점수를 개념에 추가
        enriched_concepts = []
        for c in concepts:
            enriched = c.copy()
            enriched["algorithmic_assessment"] = algorithmic_scores.get(c["concept_id"], {})
            enriched_concepts.append(enriched)

        instruction = f"""Review these ontology concepts with their algorithmic quality scores.

Concepts with Algorithmic Scores:
{json.dumps(enriched_concepts, indent=2, ensure_ascii=False, default=str)}

For each concept:
1. Validate or adjust the algorithmic scores
2. Provide specific feedback
3. Make final decision (approved/provisional/rejected)

Return JSON:
```json
{{
  "quality_assessments": [
    {{
      "concept_id": "<id>",
      "scores": {{
        "definition_quality": <int 0-100>,
        "evidence_strength": <int 0-100>,
        "consistency": <int 0-100>,
        "completeness": <int 0-100>
      }},
      "overall_score": <int 0-100>,
      "decision": "<approved|provisional|rejected>",
      "feedback": ["feedback1", "feedback2"],
      "improvements_needed": ["improvement1"],
      "justification": "<why this decision>"
    }}
  ]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        result = parse_llm_json(response, key="quality_assessments", default=[])

        if not result:
            # LLM 실패 시 알고리즘 결과 사용
            return [
                {
                    "concept_id": c["concept_id"],
                    "scores": {
                        "definition_quality": algorithmic_scores.get(c["concept_id"], {}).get("definition_quality", 50),
                        "evidence_strength": algorithmic_scores.get(c["concept_id"], {}).get("evidence_strength", 50),
                        "consistency": algorithmic_scores.get(c["concept_id"], {}).get("consistency", 50),
                        "completeness": algorithmic_scores.get(c["concept_id"], {}).get("completeness", 50),
                    },
                    "overall_score": algorithmic_scores.get(c["concept_id"], {}).get("overall_score", 50),
                    "decision": algorithmic_scores.get(c["concept_id"], {}).get("algorithmic_decision", "provisional"),
                    "feedback": algorithmic_scores.get(c["concept_id"], {}).get("auto_feedback", []),
                    "improvements_needed": algorithmic_scores.get(c["concept_id"], {}).get("improvements_needed", []),
                    "justification": "Based on algorithmic assessment",
                }
                for c in concepts
            ]

        return result
