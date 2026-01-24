"""
Ontology Analysis Utilities

온톨로지 분석 알고리즘:
- 충돌 탐지 (Naming, Definition, Type)
- 품질 평가 (메트릭 기반)
- 의미 검증 (패턴 및 표준)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# === Data Classes ===

@dataclass
class ConflictInfo:
    """충돌 정보"""
    conflict_id: str
    conflict_type: str  # naming, definition, relationship, type
    severity: str  # critical, high, medium, low
    items_involved: List[str]
    description: str
    evidence: str
    suggested_resolution: str
    confidence: float = 0.0


@dataclass
class QualityScore:
    """품질 점수"""
    definition_quality: int = 0
    evidence_strength: int = 0
    consistency: int = 0
    completeness: int = 0
    overall_score: int = 0
    decision: str = "pending"
    feedback: List[str] = field(default_factory=list)
    improvements_needed: List[str] = field(default_factory=list)


@dataclass
class SemanticValidation:
    """의미 검증 결과"""
    concept_id: str
    name_valid: bool = True
    definition_valid: bool = True
    domain_appropriate: bool = True
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    overall_valid: bool = True
    confidence: float = 0.0


# === Conflict Detection ===

class ConflictDetector:
    """
    충돌 탐지기

    알고리즘 기반 충돌 탐지:
    - 이름 유사도 분석 (Levenshtein)
    - 정의 중복 탐지
    - 타입 불일치 탐지
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

        # 알려진 동의어/이의어 매핑
        self.known_synonyms = {
            "patient": ["member", "beneficiary", "customer"],
            "provider": ["physician", "doctor", "practitioner"],
            "encounter": ["visit", "episode", "admission"],
            "claim": ["bill", "invoice", "charge"],
            "diagnosis": ["condition", "problem", "finding"],
            "procedure": ["service", "treatment", "intervention"],
            "medication": ["drug", "prescription", "rx"],
        }

        # 알려진 동음이의어
        self.known_homonyms = {
            "status": ["patient_status", "claim_status", "encounter_status"],
            "type": ["service_type", "provider_type", "claim_type"],
            "date": ["service_date", "birth_date", "admit_date"],
            "id": ["patient_id", "provider_id", "claim_id"],
            "code": ["diagnosis_code", "procedure_code", "ndc_code"],
        }

    def detect_all_conflicts(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[ConflictInfo]:
        """
        모든 유형의 충돌 탐지

        Args:
            concepts: 온톨로지 개념 리스트

        Returns:
            탐지된 충돌 리스트
        """
        conflicts = []

        # 1. 이름 충돌 탐지
        naming_conflicts = self._detect_naming_conflicts(concepts)
        conflicts.extend(naming_conflicts)

        # 2. 정의 충돌 탐지
        definition_conflicts = self._detect_definition_conflicts(concepts)
        conflicts.extend(definition_conflicts)

        # 3. 타입 충돌 탐지
        type_conflicts = self._detect_type_conflicts(concepts)
        conflicts.extend(type_conflicts)

        # 4. 관계 충돌 탐지
        relationship_conflicts = self._detect_relationship_conflicts(concepts)
        conflicts.extend(relationship_conflicts)

        return conflicts

    def _detect_naming_conflicts(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[ConflictInfo]:
        """이름 충돌 탐지 (동의어/동음이의어)"""
        conflicts = []
        conflict_id = 0

        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts[i + 1:], i + 1):
                name_a = concept_a.get("name", "").lower()
                name_b = concept_b.get("name", "").lower()

                # 1. 동일 이름 탐지
                if name_a == name_b:
                    conflict_id += 1
                    conflicts.append(ConflictInfo(
                        conflict_id=f"naming_{conflict_id}",
                        conflict_type="naming",
                        severity="critical",
                        items_involved=[
                            concept_a.get("concept_id", ""),
                            concept_b.get("concept_id", ""),
                        ],
                        description=f"Identical names: '{name_a}'",
                        evidence="Same name string detected",
                        suggested_resolution="Merge concepts or differentiate names",
                        confidence=1.0,
                    ))
                    continue

                # 2. 유사 이름 탐지 (Levenshtein 기반)
                similarity = self._string_similarity(name_a, name_b)
                if similarity >= self.similarity_threshold:
                    conflict_id += 1
                    conflicts.append(ConflictInfo(
                        conflict_id=f"naming_{conflict_id}",
                        conflict_type="naming",
                        severity="high",
                        items_involved=[
                            concept_a.get("concept_id", ""),
                            concept_b.get("concept_id", ""),
                        ],
                        description=f"Similar names: '{name_a}' vs '{name_b}' ({similarity:.0%} similar)",
                        evidence=f"String similarity: {similarity:.2f}",
                        suggested_resolution="Review if these represent the same concept",
                        confidence=similarity,
                    ))
                    continue

                # 3. 알려진 동의어 탐지
                synonym_match = self._check_synonyms(name_a, name_b)
                if synonym_match:
                    conflict_id += 1
                    conflicts.append(ConflictInfo(
                        conflict_id=f"naming_{conflict_id}",
                        conflict_type="naming",
                        severity="medium",
                        items_involved=[
                            concept_a.get("concept_id", ""),
                            concept_b.get("concept_id", ""),
                        ],
                        description=f"Potential synonyms: '{name_a}' and '{name_b}'",
                        evidence=f"Known synonym group: {synonym_match}",
                        suggested_resolution="Consider merging into canonical term",
                        confidence=0.75,
                    ))

        return conflicts

    def _detect_definition_conflicts(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[ConflictInfo]:
        """정의 충돌 탐지"""
        conflicts = []
        conflict_id = 0

        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts[i + 1:], i + 1):
                # 같은 타입의 개념들만 비교
                if concept_a.get("concept_type") != concept_b.get("concept_type"):
                    continue

                desc_a = concept_a.get("description", "").lower()
                desc_b = concept_b.get("description", "").lower()

                if not desc_a or not desc_b:
                    continue

                # 정의 유사도 계산
                similarity = self._string_similarity(desc_a, desc_b)

                if similarity >= 0.7:  # 70% 이상 유사
                    conflict_id += 1
                    conflicts.append(ConflictInfo(
                        conflict_id=f"definition_{conflict_id}",
                        conflict_type="definition",
                        severity="high" if similarity >= 0.85 else "medium",
                        items_involved=[
                            concept_a.get("concept_id", ""),
                            concept_b.get("concept_id", ""),
                        ],
                        description=f"Overlapping definitions ({similarity:.0%} similar)",
                        evidence=f"Description similarity: {similarity:.2f}",
                        suggested_resolution="Clarify distinction or merge concepts",
                        confidence=similarity,
                    ))

        return conflicts

    def _detect_type_conflicts(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[ConflictInfo]:
        """타입 충돌 탐지"""
        conflicts = []
        conflict_id = 0

        # 동일 이름이지만 다른 타입인 경우 탐지
        name_to_types: Dict[str, List[Dict]] = defaultdict(list)

        for concept in concepts:
            name = concept.get("name", "").lower()
            name_to_types[name].append(concept)

        for name, type_concepts in name_to_types.items():
            if len(type_concepts) > 1:
                types = set(c.get("concept_type") for c in type_concepts)
                if len(types) > 1:
                    conflict_id += 1
                    conflicts.append(ConflictInfo(
                        conflict_id=f"type_{conflict_id}",
                        conflict_type="type",
                        severity="critical",
                        items_involved=[c.get("concept_id", "") for c in type_concepts],
                        description=f"Same name '{name}' with different types: {types}",
                        evidence=f"Type inconsistency detected",
                        suggested_resolution="Standardize type or differentiate names",
                        confidence=1.0,
                    ))

        return conflicts

    def _detect_relationship_conflicts(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[ConflictInfo]:
        """관계 충돌 탐지"""
        conflicts = []
        conflict_id = 0

        # 정의 내 관계 정보 추출 및 비교
        for concept in concepts:
            definition = concept.get("definition", {})
            if not isinstance(definition, dict):
                continue

            relationships = definition.get("relationships", [])

            # 중복 관계 탐지
            seen_relationships = set()
            for rel in relationships:
                rel_key = (rel.get("source"), rel.get("target"), rel.get("type"))
                if rel_key in seen_relationships:
                    conflict_id += 1
                    conflicts.append(ConflictInfo(
                        conflict_id=f"relationship_{conflict_id}",
                        conflict_type="relationship",
                        severity="medium",
                        items_involved=[concept.get("concept_id", "")],
                        description=f"Duplicate relationship: {rel_key}",
                        evidence="Same relationship defined multiple times",
                        suggested_resolution="Remove duplicate relationship",
                        confidence=1.0,
                    ))
                seen_relationships.add(rel_key)

        return conflicts

    def _string_similarity(self, s1: str, s2: str) -> float:
        """문자열 유사도 계산 (SequenceMatcher 기반)"""
        return SequenceMatcher(None, s1, s2).ratio()

    def _check_synonyms(self, name_a: str, name_b: str) -> Optional[str]:
        """알려진 동의어 확인"""
        for canonical, synonyms in self.known_synonyms.items():
            all_terms = [canonical] + synonyms
            if name_a in all_terms and name_b in all_terms:
                return canonical
        return None


# === Quality Assessment ===

class QualityAssessor:
    """
    품질 평가기

    메트릭 기반 품질 평가:
    - 정의 품질 (완전성, 명확성)
    - 증거 강도 (매치율, 소스 수)
    - 일관성 (명명 규칙, 표준 준수)
    - 완전성 (필수 필드 존재)
    """

    def __init__(self):
        # 명명 규칙 패턴
        self.object_type_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')  # PascalCase
        self.property_pattern = re.compile(r'^[a-z][a-zA-Z0-9]*$')  # camelCase
        self.link_pattern = re.compile(r'^[a-z]+(_[a-z]+)*$')  # snake_case

        # 필수 필드
        self.required_fields = {
            "object_type": ["name", "description", "source_tables"],
            "property": ["name", "description", "data_type"],
            "link_type": ["name", "source", "target"],
            "constraint": ["name", "description", "rule"],
        }

    def assess_concept(
        self,
        concept: Dict[str, Any],
        all_concepts: Optional[List[Dict]] = None,
    ) -> QualityScore:
        """
        개념 품질 평가

        Args:
            concept: 평가할 개념
            all_concepts: 전체 개념 (일관성 평가용)

        Returns:
            품질 점수
        """
        score = QualityScore()

        # 1. 정의 품질 평가 (0-100)
        score.definition_quality = self._assess_definition_quality(concept)

        # 2. 증거 강도 평가 (0-100)
        score.evidence_strength = self._assess_evidence_strength(concept)

        # 3. 일관성 평가 (0-100)
        score.consistency = self._assess_consistency(concept, all_concepts)

        # 4. 완전성 평가 (0-100)
        score.completeness = self._assess_completeness(concept)

        # 전체 점수 계산 (가중 평균)
        weights = {
            "definition_quality": 0.25,
            "evidence_strength": 0.30,
            "consistency": 0.20,
            "completeness": 0.25,
        }

        score.overall_score = int(
            score.definition_quality * weights["definition_quality"] +
            score.evidence_strength * weights["evidence_strength"] +
            score.consistency * weights["consistency"] +
            score.completeness * weights["completeness"]
        )

        # 결정
        if score.overall_score >= 90:
            score.decision = "approved"
        elif score.overall_score >= 70:
            score.decision = "approved"
            score.feedback.append("Good quality with minor improvements possible")
        elif score.overall_score >= 50:
            score.decision = "provisional"
            score.feedback.append("Acceptable but requires attention")
        else:
            score.decision = "rejected"
            score.feedback.append("Quality below acceptable threshold")

        return score

    def _assess_definition_quality(self, concept: Dict[str, Any]) -> int:
        """정의 품질 평가"""
        score = 0
        max_score = 100

        name = concept.get("name", "")
        description = concept.get("description", "")

        # 이름 존재 및 길이 (20점)
        if name:
            score += 10
            if 3 <= len(name) <= 50:
                score += 10

        # 설명 존재 및 품질 (40점)
        if description:
            score += 10
            # 적절한 길이 (20-500자)
            if 20 <= len(description) <= 500:
                score += 15
            elif len(description) > 10:
                score += 5
            # 문장 구조 (마침표로 끝남)
            if description.strip().endswith('.'):
                score += 5
            # 구체성 (숫자나 구체적 용어 포함)
            if any(char.isdigit() for char in description) or \
               any(term in description.lower() for term in ["must", "should", "unique", "required"]):
                score += 10

        # 정의 상세 (40점)
        definition = concept.get("definition", {})
        if isinstance(definition, dict):
            if definition.get("attributes"):
                score += 15
            if definition.get("relationships"):
                score += 15
            if definition.get("constraints"):
                score += 10

        return min(score, max_score)

    def _assess_evidence_strength(self, concept: Dict[str, Any]) -> int:
        """증거 강도 평가"""
        score = 0
        max_score = 100

        # 소스 테이블 수 (30점)
        source_tables = concept.get("source_tables", [])
        if source_tables:
            score += min(len(source_tables) * 10, 30)

        # 소스 증거 (40점)
        source_evidence = concept.get("source_evidence", {})
        if isinstance(source_evidence, dict):
            # 매치율
            match_rate = source_evidence.get("match_rate", 0)
            score += int(match_rate * 20)

            # 신뢰도
            confidence = source_evidence.get("confidence", 0)
            score += int(confidence * 20)

        # 개념 자체 신뢰도 (30점)
        concept_confidence = concept.get("confidence", 0)
        score += int(concept_confidence * 30)

        return min(score, max_score)

    def _assess_consistency(
        self,
        concept: Dict[str, Any],
        all_concepts: Optional[List[Dict]] = None,
    ) -> int:
        """일관성 평가"""
        score = 100  # 시작점에서 감점

        name = concept.get("name", "")
        concept_type = concept.get("concept_type", "object_type")

        # 명명 규칙 검사
        if concept_type == "object_type":
            if not self.object_type_pattern.match(name):
                score -= 30
        elif concept_type == "property":
            if not self.property_pattern.match(name):
                score -= 30
        elif concept_type == "link_type":
            if not self.link_pattern.match(name):
                score -= 30

        # 전체 개념과의 일관성
        if all_concepts:
            # 유사 이름 개념과의 일관성 검사
            for other in all_concepts:
                if other.get("concept_id") == concept.get("concept_id"):
                    continue
                other_name = other.get("name", "").lower()
                if name.lower() == other_name:
                    score -= 20  # 중복 이름
                    break

        return max(0, score)

    def _assess_completeness(self, concept: Dict[str, Any]) -> int:
        """완전성 평가"""
        concept_type = concept.get("concept_type", "object_type")
        required = self.required_fields.get(concept_type, [])

        if not required:
            return 80  # 기본값

        present = 0
        for field in required:
            if field in concept and concept[field]:
                present += 1
            elif field in concept.get("definition", {}):
                present += 1

        return int((present / len(required)) * 100)


# === Semantic Validation ===

class SemanticValidator:
    """
    의미 검증기

    도메인 표준 기반 검증:
    - 의료 용어 표준 (ICD, CPT, SNOMED)
    - 명명 패턴 검증
    - 관계 의미 검증
    """

    def __init__(self, domain: str = "healthcare"):
        self.domain = domain

        # 도메인별 표준 용어
        self.domain_terms = {
            "healthcare": {
                # 핵심 엔티티
                "core_entities": [
                    "patient", "provider", "encounter", "claim",
                    "diagnosis", "procedure", "medication", "lab",
                    "facility", "payer", "plan", "member",
                ],
                # 표준 관계
                "standard_relationships": [
                    ("patient", "encounter", "has"),
                    ("encounter", "provider", "performed_by"),
                    ("encounter", "diagnosis", "has"),
                    ("claim", "service", "contains"),
                ],
                # 안티패턴
                "anti_patterns": [
                    ("*", "data", "link"),  # 너무 일반적인 이름
                    ("*", "info", "link"),
                    ("*", "misc", "link"),
                ],
            }
        }

        # 의미적으로 유효하지 않은 관계 패턴
        self.invalid_relationships = [
            ("patient", "patient", "is"),  # 자기 참조 (일반적으로 잘못)
            ("claim", "diagnosis", "treats"),  # 의미 오류
        ]

    def validate_concept(
        self,
        concept: Dict[str, Any],
        domain_context: str = "",
    ) -> SemanticValidation:
        """
        개념 의미 검증

        Args:
            concept: 검증할 개념
            domain_context: 도메인 컨텍스트

        Returns:
            검증 결과
        """
        validation = SemanticValidation(
            concept_id=concept.get("concept_id", ""),
        )

        # 1. 이름 검증
        validation.name_valid, name_issues = self._validate_name(concept)
        validation.issues.extend(name_issues)

        # 2. 정의 검증
        validation.definition_valid, def_issues = self._validate_definition(concept)
        validation.issues.extend(def_issues)

        # 3. 도메인 적합성 검증
        validation.domain_appropriate, domain_issues = self._validate_domain_fit(
            concept, domain_context
        )
        validation.issues.extend(domain_issues)

        # 전체 유효성
        validation.overall_valid = (
            validation.name_valid and
            validation.definition_valid and
            validation.domain_appropriate
        )

        # 신뢰도 계산
        valid_count = sum([
            validation.name_valid,
            validation.definition_valid,
            validation.domain_appropriate,
        ])
        validation.confidence = valid_count / 3.0

        # 개선 제안
        if not validation.overall_valid:
            validation.suggestions.extend(self._generate_suggestions(validation))

        return validation

    def _validate_name(self, concept: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """이름 검증"""
        issues = []
        name = concept.get("name", "")

        if not name:
            issues.append("Missing concept name")
            return False, issues

        # 너무 일반적인 이름
        generic_names = {"data", "info", "item", "thing", "object", "entity", "record"}
        if name.lower() in generic_names:
            issues.append(f"Name '{name}' is too generic")

        # 예약어 사용
        reserved = {"type", "class", "id", "name", "value", "null", "true", "false"}
        if name.lower() in reserved:
            issues.append(f"Name '{name}' is a reserved word")

        # 숫자로 시작
        if name and name[0].isdigit():
            issues.append(f"Name '{name}' should not start with a number")

        # 특수문자 포함
        if not name.replace("_", "").isalnum():
            issues.append(f"Name '{name}' contains invalid characters")

        return len(issues) == 0, issues

    def _validate_definition(self, concept: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """정의 검증"""
        issues = []

        description = concept.get("description", "")
        definition = concept.get("definition", {})

        # 설명 검증
        if not description:
            issues.append("Missing description")
        elif len(description) < 10:
            issues.append("Description is too short")

        # 순환 참조 검증
        if isinstance(definition, dict):
            relationships = definition.get("relationships", [])
            concept_name = concept.get("name", "").lower()

            for rel in relationships:
                if isinstance(rel, dict):
                    target = rel.get("target", "").lower()
                    if target == concept_name and rel.get("type") != "self_reference":
                        issues.append(f"Potential circular reference to self")

        return len(issues) == 0, issues

    def _validate_domain_fit(
        self,
        concept: Dict[str, Any],
        domain_context: str,
    ) -> Tuple[bool, List[str]]:
        """도메인 적합성 검증"""
        issues = []

        domain_terms = self.domain_terms.get(self.domain, {})
        core_entities = domain_terms.get("core_entities", [])
        anti_patterns = domain_terms.get("anti_patterns", [])

        name = concept.get("name", "").lower()

        # 안티패턴 체크
        for pattern in anti_patterns:
            _, term, _ = pattern
            if term in name:
                issues.append(f"Name contains anti-pattern term: '{term}'")

        # 도메인 용어와의 관련성 (권장사항)
        if core_entities:
            is_related = any(
                entity in name or name in entity
                for entity in core_entities
            )
            if not is_related and concept.get("concept_type") == "object_type":
                # 경고만, 오류는 아님
                pass

        return len(issues) == 0, issues

    def _generate_suggestions(self, validation: SemanticValidation) -> List[str]:
        """개선 제안 생성"""
        suggestions = []

        if not validation.name_valid:
            suggestions.append("Review and update the concept name to be more specific")

        if not validation.definition_valid:
            suggestions.append("Expand the definition with more detail")

        if not validation.domain_appropriate:
            suggestions.append("Ensure concept aligns with domain standards")

        return suggestions

    def validate_relationship(
        self,
        source_concept: str,
        target_concept: str,
        relationship_type: str,
    ) -> Tuple[bool, List[str]]:
        """관계 의미 검증"""
        issues = []

        source = source_concept.lower()
        target = target_concept.lower()
        rel_type = relationship_type.lower()

        # 자기 참조 검사
        if source == target and rel_type not in ["recursive", "self_reference", "hierarchy"]:
            issues.append(f"Self-reference relationship may be incorrect: {source} -> {target}")

        # 알려진 무효 관계 검사
        for invalid_source, invalid_target, invalid_type in self.invalid_relationships:
            if (source == invalid_source and
                target == invalid_target and
                rel_type == invalid_type):
                issues.append(f"Invalid relationship pattern detected")

        return len(issues) == 0, issues
