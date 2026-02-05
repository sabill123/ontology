"""
OntologyConcept Behavior Definitions (v2.0)

Ontoloty 시스템의 핵심 개념인 OntologyConcept에 대한 행동 정의

OntologyConcept는 다음을 포함:
- Entity: 테이블/개체 (Customer, Order, Product 등)
- Relationship: FK 관계 (Order → Customer)
- Attribute: 속성 (name, price, status 등)

팔란티어 검증 기준:
✅ 비즈니스 개념이 1급 객체 - OntologyConcept가 행동을 가짐
✅ Object → Action 구조 - validate(), merge(), classify() 등
✅ 상태 전이가 명시적 - draft → validated → approved

핵심 행동:
1. validate() - 개념 유효성 검증
2. classify() - 개념 유형 분류
3. merge() - 중복 개념 병합
4. suggestRelationships() - 관계 추천
5. generateSHACL() - SHACL 제약조건 생성
6. computeQualityScore() - 품질 점수 계산
"""

from typing import Dict, List, Any
import logging

from ..osdk.behavior import (
    BehaviorDefinition,
    BehaviorType,
    ExecutionMode,
    BehaviorParameter,
    GlobalBehaviorRegistry,
    create_computation_behavior,
    create_validation_behavior,
    create_query_behavior,
    create_llm_behavior,
)

logger = logging.getLogger(__name__)


def register_ontology_concept_behaviors() -> None:
    """
    OntologyConcept의 모든 행동 등록

    이 함수는 시스템 시작 시 호출되어
    GlobalBehaviorRegistry에 행동들을 등록함
    """
    registry = GlobalBehaviorRegistry.register_type("OntologyConcept")

    # ============== 검증 행동 (VALIDATION) ==============

    # 1. validate - 개념 유효성 검증
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.validate",
        name="validate",
        behavior_type=BehaviorType.VALIDATION,
        execution_mode=ExecutionMode.PYTHON,
        description="개념의 구조적/의미적 유효성을 검증합니다",
        python_code="""
issues = []

# 필수 속성 검증
if not self.get("name"):
    issues.append({"type": "missing_name", "severity": "error"})

if not self.get("concept_type"):
    issues.append({"type": "missing_type", "severity": "error"})

# 명명 규칙 검증
name = self.get("name", "")
if name and not name[0].isupper():
    issues.append({"type": "naming_convention", "severity": "warning", "message": "개념명은 대문자로 시작해야 합니다"})

# 속성 수 검증
properties = self.get("properties", [])
if len(properties) == 0:
    issues.append({"type": "no_properties", "severity": "warning"})

# 관계 검증
relationships = self.get("relationships", [])
for rel in relationships:
    if not rel.get("target"):
        issues.append({"type": "orphan_relationship", "severity": "error", "relationship": rel.get("name")})

result = {
    "is_valid": len([i for i in issues if i["severity"] == "error"]) == 0,
    "issues": issues,
    "issue_count": len(issues),
    "error_count": len([i for i in issues if i["severity"] == "error"]),
    "warning_count": len([i for i in issues if i["severity"] == "warning"])
}
        """,
        return_type="object",
    ))

    # 2. validateSHACL - SHACL 제약조건 검증
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.validate_shacl",
        name="validateSHACL",
        behavior_type=BehaviorType.VALIDATION,
        execution_mode=ExecutionMode.PYTHON,
        description="SHACL 제약조건을 기반으로 데이터 유효성을 검증합니다",
        python_code="""
violations = []
properties = self.get("properties", [])

for prop in properties:
    prop_name = prop.get("name", "")
    value = self.get(prop_name)

    # minCount 검증
    if prop.get("required") and value is None:
        violations.append({
            "property": prop_name,
            "constraint": "sh:minCount",
            "expected": 1,
            "actual": 0
        })

    # datatype 검증
    expected_type = prop.get("type")
    if value is not None and expected_type:
        actual_type = type(value).__name__
        type_map = {"string": "str", "integer": "int", "float": "float", "boolean": "bool"}
        if type_map.get(expected_type) and actual_type != type_map[expected_type]:
            violations.append({
                "property": prop_name,
                "constraint": "sh:datatype",
                "expected": expected_type,
                "actual": actual_type
            })

result = {
    "conforms": len(violations) == 0,
    "violations": violations,
    "violation_count": len(violations)
}
        """,
        return_type="object",
    ))

    # ============== 계산 행동 (COMPUTATION) ==============

    # 3. computeQualityScore - 품질 점수 계산
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.compute_quality_score",
        name="computeQualityScore",
        behavior_type=BehaviorType.COMPUTATION,
        execution_mode=ExecutionMode.PYTHON,
        description="개념의 품질 점수를 계산합니다 (0.0 ~ 1.0)",
        python_code="""
score = 1.0
factors = {}

# 이름 품질 (20%)
name = self.get("name", "")
name_score = 1.0
if not name:
    name_score = 0.0
elif len(name) < 3:
    name_score = 0.5
elif not name[0].isupper():
    name_score = 0.8
factors["name"] = name_score

# 설명 품질 (20%)
description = self.get("description", "")
desc_score = min(len(description) / 100, 1.0) if description else 0.0
factors["description"] = desc_score

# 속성 품질 (30%)
properties = self.get("properties", [])
prop_score = min(len(properties) / 5, 1.0)  # 5개 이상이면 만점
factors["properties"] = prop_score

# 관계 품질 (20%)
relationships = self.get("relationships", [])
rel_score = min(len(relationships) / 3, 1.0)  # 3개 이상이면 만점
factors["relationships"] = rel_score

# 신뢰도 (10%)
confidence = self.get("confidence", 0.5)
factors["confidence"] = confidence

# 가중 평균
weights = {"name": 0.2, "description": 0.2, "properties": 0.3, "relationships": 0.2, "confidence": 0.1}
final_score = sum(factors[k] * weights[k] for k in factors)

result = {
    "quality_score": round(final_score, 3),
    "factors": factors,
    "grade": "A" if final_score >= 0.9 else "B" if final_score >= 0.7 else "C" if final_score >= 0.5 else "D"
}
        """,
        return_type="object",
    ))

    # 4. computeImportance - 중요도 계산
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.compute_importance",
        name="computeImportance",
        behavior_type=BehaviorType.COMPUTATION,
        execution_mode=ExecutionMode.PYTHON,
        description="온톨로지 내에서 개념의 중요도를 계산합니다",
        python_code="""
score = 0.0

# 관계 수 기반 (PageRank 스타일)
incoming = len(self.get("incoming_relationships", []))
outgoing = len(self.get("relationships", []))
rel_score = (incoming * 2 + outgoing) / 10  # incoming에 가중치

# 속성 수 기반
properties = len(self.get("properties", []))
prop_score = properties / 20

# 참조 빈도 기반
reference_count = self.get("reference_count", 0)
ref_score = min(reference_count / 100, 1.0)

# 가중 합계
importance = min(rel_score * 0.5 + prop_score * 0.3 + ref_score * 0.2, 1.0)

result = {
    "importance": round(importance, 3),
    "factors": {
        "relationship_score": rel_score,
        "property_score": prop_score,
        "reference_score": ref_score
    },
    "rank": "critical" if importance >= 0.8 else "high" if importance >= 0.6 else "medium" if importance >= 0.3 else "low"
}
        """,
        return_type="object",
    ))

    # ============== 조회 행동 (QUERY) ==============

    # 5. getRelatedConcepts - 관련 개념 조회
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.get_related_concepts",
        name="getRelatedConcepts",
        behavior_type=BehaviorType.QUERY,
        execution_mode=ExecutionMode.PYTHON,
        description="이 개념과 관련된 모든 개념을 조회합니다",
        python_code="""
related = []

# 나가는 관계
for rel in self.get("relationships", []):
    related.append({
        "concept": rel.get("target"),
        "relationship": rel.get("name"),
        "direction": "outgoing",
        "cardinality": rel.get("cardinality", "many_to_one")
    })

# 들어오는 관계
for rel in self.get("incoming_relationships", []):
    related.append({
        "concept": rel.get("source"),
        "relationship": rel.get("name"),
        "direction": "incoming",
        "cardinality": rel.get("cardinality", "one_to_many")
    })

result = {
    "related_concepts": related,
    "total_count": len(related),
    "outgoing_count": len(self.get("relationships", [])),
    "incoming_count": len(self.get("incoming_relationships", []))
}
        """,
        return_type="object",
        cache_ttl_seconds=60,  # 1분 캐싱
    ))

    # 6. getProperties - 속성 목록 조회
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.get_properties",
        name="getProperties",
        behavior_type=BehaviorType.QUERY,
        execution_mode=ExecutionMode.PYTHON,
        description="개념의 모든 속성을 조회합니다",
        python_code="""
properties = self.get("properties", [])

result = {
    "properties": properties,
    "count": len(properties),
    "required_properties": [p for p in properties if p.get("required")],
    "indexed_properties": [p for p in properties if p.get("indexed")],
    "derived_properties": [p for p in properties if p.get("is_derived")]
}
        """,
        return_type="object",
        cache_ttl_seconds=120,  # 2분 캐싱
    ))

    # ============== LLM 기반 행동 ==============

    # 7. classify - 개념 유형 분류 (LLM)
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.classify",
        name="classify",
        behavior_type=BehaviorType.COMPUTATION,
        execution_mode=ExecutionMode.LLM,
        description="LLM을 사용하여 개념의 비즈니스 유형을 분류합니다",
        llm_prompt_template="""
다음 온톨로지 개념을 분석하고 유형을 분류하세요.

개념 정보:
- 이름: {self.name}
- 설명: {self.description}
- 속성: {self.properties}

다음 유형 중 하나로 분류하고, 그 이유를 설명하세요:
1. master_data: 핵심 비즈니스 엔티티 (Customer, Product, Employee 등)
2. transaction: 비즈니스 이벤트 (Order, Payment, Shipment 등)
3. reference: 참조 데이터 (Country, Status, Category 등)
4. junction: 다대다 관계 (OrderItem, Enrollment 등)
5. audit: 감사/이력 데이터 (Log, History, Audit 등)

JSON 형식으로 응답:
{{
    "classified_type": "유형명",
    "confidence": 0.0~1.0,
    "reasoning": "분류 이유",
    "suggested_behaviors": ["추천 행동1", "추천 행동2"]
}}
        """,
        return_type="object",
    ))

    # 8. suggestRelationships - 관계 추천 (LLM)
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.suggest_relationships",
        name="suggestRelationships",
        behavior_type=BehaviorType.COMPUTATION,
        execution_mode=ExecutionMode.LLM,
        description="LLM을 사용하여 이 개념에 적합한 관계를 추천합니다",
        llm_prompt_template="""
다음 온톨로지 개념에 대해 가능한 관계를 추천하세요.

현재 개념:
- 이름: {self.name}
- 유형: {self.concept_type}
- 기존 관계: {self.relationships}

컨텍스트의 다른 개념들:
{other_concepts}

추천할 관계 유형:
1. BELONGS_TO / HAS: 계층 관계
2. REFERENCES / REFERENCED_BY: 참조 관계
3. DEPENDS_ON: 의존 관계
4. RELATED_TO: 연관 관계

JSON 형식으로 응답:
{{
    "suggested_relationships": [
        {{
            "target_concept": "대상 개념명",
            "relationship_type": "관계 유형",
            "relationship_name": "관계명",
            "confidence": 0.0~1.0,
            "reasoning": "추천 이유"
        }}
    ]
}}
        """,
        parameters=[
            BehaviorParameter(
                name="other_concepts",
                param_type="string",
                required=False,
                default="",
                description="컨텍스트의 다른 개념 목록"
            )
        ],
        return_type="object",
    ))

    # 9. generateDescription - 설명 생성 (LLM)
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.generate_description",
        name="generateDescription",
        behavior_type=BehaviorType.COMPUTATION,
        execution_mode=ExecutionMode.LLM,
        description="LLM을 사용하여 개념에 대한 비즈니스 설명을 생성합니다",
        llm_prompt_template="""
다음 온톨로지 개념에 대한 비즈니스 설명을 생성하세요.

개념 정보:
- 이름: {self.name}
- 유형: {self.concept_type}
- 속성: {self.properties}
- 관계: {self.relationships}
- 원본 테이블: {self.source_tables}

다음 내용을 포함하여 2-3문장으로 설명하세요:
1. 이 개념이 무엇을 나타내는지
2. 비즈니스에서 어떻게 사용되는지
3. 다른 개념과의 관계

JSON 형식으로 응답:
{{
    "description": "생성된 설명",
    "keywords": ["키워드1", "키워드2"],
    "business_domain": "비즈니스 도메인 (예: 영업, 재무, HR)"
}}
        """,
        return_type="object",
    ))

    # ============== 변경 행동 (MUTATION) ==============

    # 10. merge - 개념 병합
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.merge",
        name="merge",
        behavior_type=BehaviorType.MUTATION,
        execution_mode=ExecutionMode.PYTHON,
        description="다른 개념을 이 개념에 병합합니다 (중복 제거)",
        parameters=[
            BehaviorParameter(
                name="other_concept",
                param_type="object",
                required=True,
                description="병합할 다른 개념"
            ),
            BehaviorParameter(
                name="strategy",
                param_type="string",
                required=False,
                default="union",
                description="병합 전략: union, intersection, this_priority, other_priority"
            )
        ],
        python_code="""
other = params.get("other_concept", {})
strategy = params.get("strategy", "union")

# 속성 병합
current_props = self.get("properties", [])
other_props = other.get("properties", [])

if strategy == "union":
    # 모든 속성 포함 (중복 제거)
    prop_names = set(p.get("name") for p in current_props)
    for p in other_props:
        if p.get("name") not in prop_names:
            current_props.append(p)
elif strategy == "intersection":
    # 공통 속성만
    other_names = set(p.get("name") for p in other_props)
    current_props = [p for p in current_props if p.get("name") in other_names]

# 관계 병합
current_rels = self.get("relationships", [])
other_rels = other.get("relationships", [])
rel_targets = set(r.get("target") for r in current_rels)
for r in other_rels:
    if r.get("target") not in rel_targets:
        current_rels.append(r)

# 결과 저장
self.set("properties", current_props)
self.set("relationships", current_rels)

# 병합 이력 기록
merge_history = self.get("merge_history", [])
merge_history.append({
    "merged_from": other.get("name"),
    "strategy": strategy,
    "timestamp": str(__import__('datetime').datetime.now())
})
self.set("merge_history", merge_history)

result = {
    "success": True,
    "merged_properties": len(current_props),
    "merged_relationships": len(current_rels),
    "strategy": strategy
}
        """,
        return_type="object",
    ))

    # 11. updateConfidence - 신뢰도 업데이트
    registry.register(BehaviorDefinition(
        behavior_id="ontology_concept.update_confidence",
        name="updateConfidence",
        behavior_type=BehaviorType.MUTATION,
        execution_mode=ExecutionMode.PYTHON,
        description="에이전트 합의 결과에 따라 신뢰도를 업데이트합니다",
        parameters=[
            BehaviorParameter(
                name="agent_votes",
                param_type="list",
                required=True,
                description="에이전트 투표 결과 [{agent_id, vote, weight}]"
            )
        ],
        python_code="""
votes = params.get("agent_votes", [])

if not votes:
    result = {"success": False, "error": "No votes provided"}
else:
    # 가중 평균 계산
    total_weight = sum(v.get("weight", 1.0) for v in votes)
    weighted_sum = sum(v.get("vote", 0.5) * v.get("weight", 1.0) for v in votes)
    new_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

    # 기존 신뢰도와 블렌딩 (Bayesian update 스타일)
    old_confidence = self.get("confidence", 0.5)
    blended = old_confidence * 0.3 + new_confidence * 0.7

    self.set("confidence", round(blended, 3))
    self.set("last_confidence_update", str(__import__('datetime').datetime.now()))

    result = {
        "success": True,
        "old_confidence": old_confidence,
        "new_confidence": round(blended, 3),
        "vote_count": len(votes),
        "consensus_level": "high" if blended >= 0.8 else "medium" if blended >= 0.5 else "low"
    }
        """,
        return_type="object",
    ))

    logger.info("[OntologyConcept] Registered 11 behaviors for OntologyConcept")


# 편의 함수: 개별 행동 정의 접근
def get_ontology_concept_behaviors() -> Dict[str, BehaviorDefinition]:
    """등록된 모든 OntologyConcept 행동 반환"""
    registry = GlobalBehaviorRegistry.get_registry("OntologyConcept")
    if registry:
        return {b.name: b for b in registry.list_all()}
    return {}


__all__ = [
    "register_ontology_concept_behaviors",
    "get_ontology_concept_behaviors",
]
