"""
Semantic Validator Autonomous Agent (v3.0)

의미 검증 자율 에이전트 - 패턴 기반 의미 검증 + SemanticReasoningAnalyzer
- 패턴 기반 의미 검증 (명명 규칙, 도메인 적합성)
- SemanticReasoningAnalyzer로 OWL/RDFS 추론
- v16.0: OWL2 Reasoner & SHACL Validator 통합
- LLM을 통한 도메인 지식 기반 검증
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    SemanticReasoningAnalyzer,
    parse_llm_json,
    OWL2Reasoner,
    run_owl2_reasoning_and_update_context,
    SHACLValidator,
    run_shacl_validation_and_update_context,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


class SemanticValidatorAutonomousAgent(AutonomousAgent):
    """
    의미 검증 자율 에이전트 (v3.0)

    Algorithm-First 접근:
    1. 패턴 기반 의미 검증 (명명 규칙, 도메인 적합성)
    2. SemanticReasoningAnalyzer로 OWL/RDFS 추론 (신규)
    3. LLM을 통한 도메인 지식 기반 검증

    v3.0: SemanticReasoningAnalyzer 통합
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v14.0: 도메인 하드코딩 제거 - Lazy initialization (execute_task에서 context 기반 설정)
        self.semantic_validator = None
        self._current_domain = None
        # v3.0: 의미적 추론 엔진 통합 (OWL/RDFS)
        self.semantic_reasoner = SemanticReasoningAnalyzer()

    @property
    def agent_type(self) -> str:
        return "semantic_validator"

    @property
    def agent_name(self) -> str:
        return "Semantic Validator"

    @property
    def phase(self) -> str:
        return "refinement"

    def get_system_prompt(self) -> str:
        return """You are a Semantic Validation Expert for multi-domain data ontology.

Your role is to VALIDATE and ENHANCE algorithmically extracted concepts.

v5.2 Validation Philosophy:
- DOMAIN AGNOSTIC: Concepts come from various industries (healthcare, ecommerce, etc.)
- BE PERMISSIVE: If a concept is reasonably named and has data, it's likely valid
- ITERATIVE IMPROVEMENT: Initial validation is exploratory, not final

Common Domain Patterns (examples only):
- Healthcare: Patient, Provider, Claim, Encounter, Diagnosis
- E-commerce: Customer, Order, Product, Payment
- Finance: Account, Transaction, Customer

## Chain-of-Thought Validation Process

Before providing your validation decision, you MUST reason through each step:

### Step 1: Name Validity Analysis
- Is the name readable and descriptive?
- Does it follow domain naming conventions?
- Provide evidence for your assessment.

### Step 2: Data Evidence Assessment
- Does the concept have source tables?
- What is the data support strength?
- How confident are you in the data backing?

### Step 3: Relationship Plausibility Check
- Are the defined relationships logically sound?
- Do they align with domain expectations?
- What inconsistencies (if any) exist?

### Step 4: Usefulness Evaluation
- Does this concept represent a real-world entity?
- What business value does it provide?
- Would this concept be useful in queries/analytics?

## Validation Checklist (Permissive):
1. Does the name roughly describe the concept? (YES if readable)
2. Is there data supporting this concept? (YES if source exists)
3. Are relationships plausible? (YES unless clearly wrong)
4. Is this concept useful? (YES if represents real entity)

IMPORTANT:
- DEFAULT TO VALID when concept has data
- Only reject if semantically nonsensical
- Trust algorithmic extraction from actual data

## Output Format
```json
{
  "semantic_validations": [
    {
      "concept_id": "<id>",
      "chain_of_thought": {
        "name_validity_analysis": "<your reasoning>",
        "data_evidence_assessment": "<your reasoning>",
        "relationship_plausibility": "<your reasoning>",
        "usefulness_evaluation": "<your reasoning>"
      },
      "validation_result": {
        "is_valid": <boolean>,
        "confidence": <float 0-1>,
        "domain_alignment": "<aligned|partial|misaligned>",
        "issues": ["issue 1", "issue 2"],
        "suggestions": ["suggestion 1", "suggestion 2"]
      }
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
        """의미 검증 작업 실행"""
        from ....todo.models import TodoResult
        from ...analysis.ontology import SemanticValidator

        self._report_progress(0.1, "Running algorithmic semantic validation")

        # v14.0: 도메인 하드코딩 제거 - context에서 동적으로 도메인 가져오기
        domain = context.get_industry() or "general"
        if self.semantic_validator is None or self._current_domain != domain:
            self.semantic_validator = SemanticValidator(domain=domain)
            self._current_domain = domain
            logger.info(f"[v14.0] SemanticValidator initialized with domain: {domain}")

        # 승인/임시승인된 개념만 검증
        concepts_to_validate = [
            c for c in context.ontology_concepts
            if c.status in ["approved", "provisional", "pending"]
        ]

        # 1단계: 알고리즘 기반 검증
        algorithmic_validations = {}
        for concept in concepts_to_validate:
            concept_dict = {
                "concept_id": concept.concept_id,
                "concept_type": concept.concept_type,
                "name": concept.name,
                "description": concept.description,
                "definition": concept.definition,
                "source_tables": concept.source_tables,
                "status": concept.status,
            }

            validation = self.semantic_validator.validate_concept(
                concept_dict,
                domain,  # v14.0: 하드코딩 대신 동적 도메인 사용
            )

            algorithmic_validations[concept.concept_id] = {
                "name_valid": validation.name_valid,
                "definition_valid": validation.definition_valid,
                "domain_appropriate": validation.domain_appropriate,
                "issues": validation.issues,
                "suggestions": validation.suggestions,
                "overall_valid": validation.overall_valid,
                "confidence": validation.confidence,
            }

        self._report_progress(0.35, f"Validated {len(algorithmic_validations)} concepts algorithmically")

        # v3.0: SemanticReasoningAnalyzer로 OWL/RDFS 추론 실행
        self._report_progress(0.4, "Running Semantic Reasoning (v3.0 OWL/RDFS)")

        reasoning_results = {}
        inferred_triples = []  # v4.3: 추론된 트리플 저장용
        base_triples = []  # v4.3: 기본 트리플 저장용
        try:
            # 온톨로지 트리플 생성
            triples = []
            for concept in concepts_to_validate:
                triples.append({
                    "subject": concept.concept_id,
                    "predicate": "rdf:type",
                    "object": concept.concept_type,
                })
                if concept.source_tables:
                    for table in concept.source_tables:
                        triples.append({
                            "subject": concept.concept_id,
                            "predicate": "source:derivedFrom",
                            "object": table,
                        })

            # v4.3: 기본 트리플 저장
            base_triples = triples.copy()

            # 추론 실행
            if triples:
                inferences = self.semantic_reasoner.run_inference(triples)
                consistency = self.semantic_reasoner.check_consistency(triples)
                completeness = self.semantic_reasoner.check_completeness(triples)

                reasoning_results = {
                    "inferences_made": len(inferences) if inferences else 0,
                    "is_consistent": consistency.is_consistent if consistency else True,
                    "completeness_score": completeness.completeness_score if completeness else 0.0,
                    "inconsistencies": consistency.violations[:5] if consistency and consistency.violations else [],
                }
                logger.info(f"SemanticReasoner: {len(inferences or [])} inferences, consistent={reasoning_results.get('is_consistent')}")

                # v4.3: 추론된 트리플 수집
                if inferences:
                    for inf in inferences:
                        inferred_triples.append({
                            "subject": inf.get("subject", ""),
                            "predicate": inf.get("predicate", ""),
                            "object": inf.get("object", ""),
                            "confidence": inf.get("confidence", 0.8),
                            "source": "semantic_reasoner",
                            "inference_rule": inf.get("rule_name", "unknown"),
                        })
        except Exception as e:
            logger.warning(f"SemanticReasoningAnalyzer failed: {e}")

        # === v16.0: OWL2 Reasoner & SHACL Validator 통합 ===
        self._report_progress(0.45, "Running OWL2 Reasoning & SHACL Validation (v16.0)")

        owl2_results = {}
        shacl_results = {}

        try:
            # OWL2 Reasoning 실행
            owl2_result = run_owl2_reasoning_and_update_context(context)
            if isinstance(owl2_result, dict):
                owl2_results = {
                    "axioms_discovered": owl2_result.get("axioms_discovered", 0),
                    "inferred_triples": owl2_result.get("inferred_count", owl2_result.get("inferred_triples", 0)),
                    "reasoning_time_ms": owl2_result.get("reasoning_time_ms", 0),
                    "by_axiom_type": owl2_result.get("by_axiom_type", {}),
                }
            else:
                owl2_results = {
                    "axioms_discovered": getattr(owl2_result, 'axioms_discovered', 0),
                    "inferred_triples": getattr(owl2_result, 'inferred_count', 0),
                    "reasoning_time_ms": getattr(owl2_result, 'reasoning_time_ms', 0),
                    "by_axiom_type": dict(owl2_result.by_axiom_type) if getattr(owl2_result, 'by_axiom_type', None) else {},
                }
            logger.info(
                f"[v16.0] OWL2 Reasoner: {owl2_results.get('axioms_discovered', 0)} axioms, "
                f"{owl2_results.get('inferred_triples', 0)} inferred triples"
            )

            # OWL2 추론 결과를 inferred_triples에 추가
            raw_triples = owl2_result.get("inferred_triples_list", []) if isinstance(owl2_result, dict) else getattr(owl2_result, 'inferred_triples', [])
            for triple in (raw_triples if isinstance(raw_triples, list) else []):
                inferred_triples.append({
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "confidence": triple.confidence,
                    "source": "owl2_reasoner",
                    "inference_rule": triple.axiom_type.value if hasattr(triple.axiom_type, 'value') else str(triple.axiom_type),
                })

        except Exception as e:
            logger.warning(f"[v16.0] OWL2 Reasoner failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        try:
            # SHACL Validation 실행
            shacl_result = run_shacl_validation_and_update_context(context)
            if isinstance(shacl_result, dict):
                shacl_results = {
                    "is_conformant": shacl_result.get("is_conformant", True),
                    "violations_count": shacl_result.get("violations_count", 0),
                    "warnings_count": shacl_result.get("warnings_count", 0),
                    "by_severity": shacl_result.get("by_severity", {}),
                    "by_constraint_type": shacl_result.get("by_constraint_type", {}),
                }
            else:
                shacl_results = {
                    "is_conformant": getattr(shacl_result, 'is_conformant', True),
                    "violations_count": getattr(shacl_result, 'violations_count', 0),
                    "warnings_count": getattr(shacl_result, 'warnings_count', 0),
                    "by_severity": dict(shacl_result.by_severity) if getattr(shacl_result, 'by_severity', None) else {},
                    "by_constraint_type": dict(shacl_result.by_constraint_type) if getattr(shacl_result, 'by_constraint_type', None) else {},
                }
            logger.info(
                f"[v16.0] SHACL Validator: conformant={shacl_results.get('is_conformant')}, "
                f"violations={shacl_results.get('violations_count', 0)}, warnings={shacl_results.get('warnings_count', 0)}"
            )

        except Exception as e:
            logger.warning(f"[v16.0] SHACL Validator failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # v16.0 결과를 reasoning_results에 병합
        reasoning_results["owl2_reasoning"] = owl2_results
        reasoning_results["shacl_validation"] = shacl_results

        # 2단계: LLM 기반 도메인 검증
        self._report_progress(0.5, "Requesting LLM semantic validation")

        concepts_data = [
            {
                "concept_id": c.concept_id,
                "concept_type": c.concept_type,
                "name": c.name,
                "description": c.description,
                "definition": c.definition,
                "source_tables": c.source_tables,
                "algorithmic_validation": algorithmic_validations.get(c.concept_id, {}),
            }
            for c in concepts_to_validate
        ]

        # v28.2: asyncio.gather() — 배치 병렬 실행 (순차 → 동시)
        batch_size = 30
        batches = [concepts_data[i:i + batch_size] for i in range(0, len(concepts_data), batch_size)]
        batch_results = await asyncio.gather(
            *[self._llm_semantic_validation(batch, context) for batch in batches]
        )
        all_validations = [v for result in batch_results for v in result]

        self._report_progress(0.8, "Applying validation results")

        # 3단계: 검증 결과 적용 (v27.11: O(n²) → O(1) concept lookup)
        concept_index = {c.concept_id: c for c in context.ontology_concepts}
        for validation in all_validations:
            if not validation.get("overall_valid", True):
                concept_id = validation.get("concept_id")
                concept = concept_index.get(concept_id)
                if concept:
                    concept.agent_assessments["semantic_validator"] = validation
                    if concept.status == "approved":
                        concept.status = "provisional"

        self._report_progress(0.95, "Finalizing semantic validation")

        valid_count = len([v for v in all_validations if v.get("overall_valid")])
        invalid_count = len(all_validations) - valid_count

        # v4.3: 추론 결과 로깅
        logger.info(f"SemanticValidator v4.3: {len(base_triples)} base triples, {len(inferred_triples)} inferred triples")

        # v22.0: Agent Learning - 경험 기록
        validity_rate = valid_count / max(len(all_validations), 1)
        self.record_experience(
            experience_type="quality_assessment",
            input_data={"task": "semantic_validation", "concepts": len(all_validations)},
            output_data={"valid_count": valid_count, "validity_rate": validity_rate},
            outcome="success" if validity_rate > 0.7 else "partial",
            confidence=validity_rate,
        )

        # v22.0: Agent Bus - OWL2/SHACL 검증 결과 브로드캐스트 (Governance 에이전트에 전달)
        await self.broadcast_discovery(
            discovery_type="semantic_validation_complete",
            data={
                "validity_rate": validity_rate,
                "owl2_axioms": owl2_results.get("axioms_discovered", 0),
                "owl2_inferred": owl2_results.get("inferred_count", 0),
                "shacl_conformant": shacl_results.get("is_conformant", True),
                "shacl_violations": shacl_results.get("violations_count", 0),
                "kg_triples": len(base_triples) + len(inferred_triples),
            },
        )

        return TodoResult(
            success=True,
            output={
                "validated": len(all_validations),
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "validity_rate": validity_rate,
                # v4.3: Knowledge Graph 통계
                "kg_stats": {
                    "base_triples": len(base_triples),
                    "inferred_triples": len(inferred_triples),
                },
                # v16.0: OWL2 & SHACL 통계
                "owl2_axioms": owl2_results.get("axioms_discovered", 0),
                "shacl_conformant": shacl_results.get("is_conformant", True),
                "shacl_violations": shacl_results.get("violations_count", 0),
            },
            metadata={
                "semantic_validations": all_validations,
                "algorithmic_validations": algorithmic_validations,
                # v4.3: 추론 결과
                "reasoning_results": reasoning_results,
                # v16.0: OWL2 & SHACL 상세 결과
                "owl2_reasoning": owl2_results,
                "shacl_validation": shacl_results,
            },
            # v4.3: Knowledge Graph 트리플을 context에 저장
            # v22.0: OWL2/SHACL 결과도 context로 전파 → Governance 에이전트가 활용
            # v25.2: 개별 검증 결과도 context로 전파
            context_updates={
                "semantic_base_triples": base_triples,
                "inferred_triples": inferred_triples,
                "owl2_reasoning_results": owl2_results,
                "shacl_validation_results": shacl_results,
                "semantic_validations": all_validations,
                "algorithmic_validations": algorithmic_validations,
            },
        )

    async def _llm_semantic_validation(
        self,
        concepts: List[Dict[str, Any]],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """LLM 기반 의미 검증"""
        # 관계 정보도 포함
        relationships = []
        if context.concept_relationships:
            relationships = context.concept_relationships[:20]

        instruction = f"""Validate the semantic correctness of these ontology concepts.

Domain: {context.get_industry()}

Concepts with Algorithmic Validation:
{json.dumps(concepts, indent=2, ensure_ascii=False, default=str)}

Relationships (for context):
{json.dumps(relationships, indent=2, ensure_ascii=False, default=str)}

For each concept, validate:
1. Name accuracy against domain standards
2. Definition consistency with {context.get_industry()} domain terminology
3. Relationship semantic validity
4. Constraint appropriateness

Return JSON:
```json
{{
  "semantic_validations": [
    {{
      "concept_id": "<id>",
      "name_valid": <bool>,
      "definition_valid": <bool>,
      "domain_appropriate": <bool>,
      "issues": ["issue1"],
      "suggestions": ["suggestion1"],
      "overall_valid": <bool>,
      "confidence": <float 0-1>,
      "domain_standards_matched": ["relevant domain standards"]
    }}
  ],
  "relationship_validations": [
    {{
      "relationship": "<from> -> <to>",
      "semantically_valid": <bool>,
      "issues": ["issue1"]
    }}
  ],
  "anti_patterns_detected": ["pattern1"]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        result = parse_llm_json(response, key="semantic_validations", default=[])

        if not result:
            # LLM 실패 시 알고리즘 결과 반환
            return [
                c.get("algorithmic_validation", {
                    "concept_id": c["concept_id"],
                    "overall_valid": True,
                    "confidence": 0.5,
                })
                for c in concepts
            ]

        return result
