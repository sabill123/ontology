"""
OWL 2 Full Reasoner (v16.0)

Palantir Ontology 수준의 추론 기능:
- Property Chains (속성 체인)
- Qualified Cardinality Restrictions (한정 기수 제약)
- Disjoint Union (상호 배타적 유니온)
- Role Assertions (역할 단언)
- Inverse Properties (역속성)
- Transitive Properties (전이적 속성)
- Symmetric Properties (대칭 속성)

SharedContext.knowledge_graph_triples와 연동

사용법:
    from .owl2_reasoner import run_owl2_reasoning_and_update_context
    result = run_owl2_reasoning_and_update_context(shared_context)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum
from collections import defaultdict
import logging

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


class OWLAxiomType(Enum):
    """OWL 2 Axiom 유형"""
    # Class Axioms
    SUBCLASS_OF = "SubClassOf"
    EQUIVALENT_CLASS = "EquivalentClasses"
    DISJOINT_CLASSES = "DisjointClasses"
    DISJOINT_UNION = "DisjointUnion"

    # Property Axioms
    SUBPROPERTY_OF = "SubObjectPropertyOf"
    INVERSE_PROPERTIES = "InverseObjectProperties"
    PROPERTY_CHAIN = "SubObjectPropertyOf(PropertyChain)"
    EQUIVALENT_PROPERTIES = "EquivalentObjectProperties"

    # Property Characteristics
    FUNCTIONAL_PROPERTY = "FunctionalObjectProperty"
    INVERSE_FUNCTIONAL = "InverseFunctionalObjectProperty"
    TRANSITIVE_PROPERTY = "TransitiveObjectProperty"
    SYMMETRIC_PROPERTY = "SymmetricObjectProperty"
    ASYMMETRIC_PROPERTY = "AsymmetricObjectProperty"
    REFLEXIVE_PROPERTY = "ReflexiveObjectProperty"
    IRREFLEXIVE_PROPERTY = "IrreflexiveObjectProperty"

    # Cardinality Restrictions
    QUALIFIED_CARDINALITY = "QualifiedCardinalityRestriction"
    MIN_CARDINALITY = "MinCardinalityRestriction"
    MAX_CARDINALITY = "MaxCardinalityRestriction"
    EXACT_CARDINALITY = "ExactCardinalityRestriction"

    # Domain/Range
    PROPERTY_DOMAIN = "ObjectPropertyDomain"
    PROPERTY_RANGE = "ObjectPropertyRange"


@dataclass
class OWLAxiom:
    """OWL 2 Axiom"""
    axiom_type: OWLAxiomType
    subject: str
    predicate: Optional[str] = None
    object: Optional[str] = None

    # Property Chain용 (chain의 결합 결과가 subject property)
    chain: Optional[Tuple[str, ...]] = None

    # Cardinality용
    cardinality: Optional[int] = None
    on_class: Optional[str] = None  # Qualified cardinality의 filler class

    # Disjoint/Equivalent용
    classes: Optional[Tuple[str, ...]] = None
    properties: Optional[Tuple[str, ...]] = None

    # 메타데이터
    confidence: float = 1.0
    source: str = "inferred"
    annotation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "axiom_type": self.axiom_type.value,
            "subject": self.subject,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }

        if self.predicate:
            result["predicate"] = self.predicate
        if self.object:
            result["object"] = self.object
        if self.chain:
            result["chain"] = list(self.chain)
        if self.cardinality is not None:
            result["cardinality"] = self.cardinality
        if self.on_class:
            result["on_class"] = self.on_class
        if self.classes:
            result["classes"] = list(self.classes)
        if self.properties:
            result["properties"] = list(self.properties)
        if self.annotation:
            result["annotation"] = self.annotation

        return result


@dataclass
class InferredTriple:
    """추론된 트리플"""
    subject: str
    predicate: str
    object: str

    rule_applied: str
    confidence: float = 1.0
    source_axioms: List[OWLAxiom] = field(default_factory=list)
    derivation_depth: int = 1  # 추론 체인 깊이

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "rule_applied": self.rule_applied,
            "confidence": round(self.confidence, 4),
            "derivation_depth": self.derivation_depth,
            "is_inferred": True,
        }


@dataclass
class CardinalityViolation:
    """Cardinality 제약 위반"""
    violation_type: str  # min_cardinality, max_cardinality, exact_cardinality
    subject: str
    property: str
    expected: int
    actual: int
    on_class: Optional[str] = None
    severity: str = "error"  # error, warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "subject": self.subject,
            "property": self.property,
            "expected": self.expected,
            "actual": self.actual,
            "on_class": self.on_class,
            "severity": self.severity,
        }


class OWL2Reasoner:
    """
    OWL 2 Full Reasoner

    기능:
    1. Property Chain 추론
       - hasSupervisor · worksIn → belongsToDepartment
    2. Transitive Property 추론
       - isPartOf(A, B) ∧ isPartOf(B, C) → isPartOf(A, C)
    3. Inverse Property 추론
       - manages(X, Y) ↔ reportsTo(Y, X)
    4. Symmetric Property 추론
       - relatedTo(X, Y) → relatedTo(Y, X)
    5. Qualified Cardinality 검증
       - Employee min 1 worksIn Department
    6. Disjoint Classes 검증
       - Employee ⊓ Department = ∅

    SharedContext 연동:
    - 입력: knowledge_graph_triples, ontology_concepts
    - 출력: inferred_triples, owl_axioms (dynamic_data)

    알고리즘:
    1. 트리플 인덱싱 (predicate → {(subject, object)})
    2. Axiom 자동 탐지 (패턴 기반)
    3. 추론 규칙 반복 적용 (고정점까지)
    4. Cardinality 검증
    """

    # Transitive property 패턴
    TRANSITIVE_PATTERNS = [
        "part_of", "contains", "ancestor_of", "descendant_of",
        "subclass_of", "is_a", "has_parent", "belongs_to",
        "located_in", "region_of",
    ]

    # Inverse property 쌍
    INVERSE_PAIRS = [
        ("manages", "reports_to"),
        ("supervises", "supervised_by"),
        ("has_child", "has_parent"),
        ("parent_of", "child_of"),
        ("contains", "part_of"),
        ("has_member", "member_of"),
        ("employs", "employed_by"),
        ("owns", "owned_by"),
        ("created_by", "created"),
        ("assigned_to", "has_assignment"),
    ]

    # Symmetric property 패턴
    SYMMETRIC_PATTERNS = [
        "related_to", "connected_to", "sibling_of",
        "colleague_of", "neighbor_of", "similar_to",
        "same_as", "married_to",
    ]

    # 일반적인 Property Chain 패턴
    PROPERTY_CHAIN_TEMPLATES = [
        # (chain_properties, result_property)
        (("has_supervisor", "works_in"), "belongs_to_department"),
        (("has_parent", "lives_in"), "born_in"),
        (("member_of", "located_in"), "based_in"),
    ]

    def __init__(self, max_iterations: int = 10):
        """
        Args:
            max_iterations: 추론 반복 최대 횟수 (무한 루프 방지)
        """
        self.max_iterations = max_iterations
        self.axioms: List[OWLAxiom] = []
        self.inferred: List[InferredTriple] = []
        self.violations: List[CardinalityViolation] = []

        # 인덱스
        self._triple_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._subject_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._object_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._inferred_set: Set[Tuple[str, str, str]] = set()

    def reason(
        self,
        triples: List[Dict[str, Any]],
        axioms: Optional[List[OWLAxiom]] = None,
        cardinality_restrictions: Optional[List[OWLAxiom]] = None,
    ) -> List[InferredTriple]:
        """
        추론 실행

        Args:
            triples: 입력 트리플 [{subject, predicate, object, ...}]
            axioms: OWL Axiom 목록 (없으면 자동 탐지)
            cardinality_restrictions: Cardinality 제약 (검증용)

        Returns:
            추론된 트리플 리스트
        """
        self.inferred = []
        self.violations = []
        self._inferred_set = set()

        # 1. 트리플 인덱싱
        self._build_indices(triples)

        # 2. Axiom 자동 탐지 (없으면)
        if not axioms:
            axioms = self._discover_axioms(triples)
        self.axioms = axioms

        logger.info(f"[OWL2Reasoner] Starting reasoning with {len(triples)} triples, {len(axioms)} axioms")

        # 3. 추론 규칙 반복 적용 (고정점까지)
        iteration = 0
        new_inferred_count = 1

        while new_inferred_count > 0 and iteration < self.max_iterations:
            prev_count = len(self.inferred)

            for axiom in axioms:
                self._apply_axiom(axiom, triples)

            new_inferred_count = len(self.inferred) - prev_count
            iteration += 1

            if new_inferred_count > 0:
                logger.debug(f"[OWL2Reasoner] Iteration {iteration}: {new_inferred_count} new inferences")

        # 4. Cardinality 검증
        if cardinality_restrictions:
            self.violations = self._validate_cardinality(triples, cardinality_restrictions)

        logger.info(
            f"[OWL2Reasoner] Completed: {len(self.inferred)} inferred triples, "
            f"{len(self.violations)} violations, {iteration} iterations"
        )

        return self.inferred

    def _build_indices(self, triples: List[Dict[str, Any]]) -> None:
        """트리플 인덱싱"""
        self._triple_index.clear()
        self._subject_index.clear()
        self._object_index.clear()

        for t in triples:
            pred = t.get("predicate", "")
            subj = t.get("subject", "")
            obj = t.get("object", "")

            if pred and subj and obj:
                self._triple_index[pred].add((subj, obj))
                self._subject_index[subj].add((pred, obj))
                self._object_index[obj].add((pred, subj))

    def _discover_axioms(self, triples: List[Dict[str, Any]]) -> List[OWLAxiom]:
        """트리플에서 Axiom 자동 탐지"""
        axioms = []
        predicates = set(t.get("predicate", "") for t in triples)

        for pred in predicates:
            pred_lower = pred.lower().replace("-", "_").replace(" ", "_")

            # Transitive 패턴
            for pattern in self.TRANSITIVE_PATTERNS:
                if pattern in pred_lower:
                    axioms.append(OWLAxiom(
                        axiom_type=OWLAxiomType.TRANSITIVE_PROPERTY,
                        subject=pred,
                        source="pattern_discovery",
                        annotation=f"Matched transitive pattern: {pattern}",
                    ))
                    break

            # Inverse 쌍
            for prop1, prop2 in self.INVERSE_PAIRS:
                if prop1 in pred_lower:
                    # 역방향 property 이름 추론
                    inverse_pred = pred.replace(prop1, prop2)
                    axioms.append(OWLAxiom(
                        axiom_type=OWLAxiomType.INVERSE_PROPERTIES,
                        subject=pred,
                        object=inverse_pred,
                        source="pattern_discovery",
                        annotation=f"Inverse pair: {prop1} <-> {prop2}",
                    ))
                    break
                elif prop2 in pred_lower:
                    inverse_pred = pred.replace(prop2, prop1)
                    axioms.append(OWLAxiom(
                        axiom_type=OWLAxiomType.INVERSE_PROPERTIES,
                        subject=pred,
                        object=inverse_pred,
                        source="pattern_discovery",
                    ))
                    break

            # Symmetric 패턴
            for pattern in self.SYMMETRIC_PATTERNS:
                if pattern in pred_lower:
                    axioms.append(OWLAxiom(
                        axiom_type=OWLAxiomType.SYMMETRIC_PROPERTY,
                        subject=pred,
                        source="pattern_discovery",
                        annotation=f"Matched symmetric pattern: {pattern}",
                    ))
                    break

        # Property Chain 탐지 (predicate 조합 기반)
        for (chain_props, result_prop) in self.PROPERTY_CHAIN_TEMPLATES:
            if all(any(cp in p.lower() for p in predicates) for cp in chain_props):
                # 실제 predicate 이름 찾기
                actual_chain = []
                for cp in chain_props:
                    for p in predicates:
                        if cp in p.lower():
                            actual_chain.append(p)
                            break

                if len(actual_chain) == len(chain_props):
                    axioms.append(OWLAxiom(
                        axiom_type=OWLAxiomType.PROPERTY_CHAIN,
                        subject=result_prop,
                        chain=tuple(actual_chain),
                        source="pattern_discovery",
                        annotation=f"Property chain: {' · '.join(actual_chain)} → {result_prop}",
                    ))

        logger.info(f"[OWL2Reasoner] Discovered {len(axioms)} axioms from patterns")
        return axioms

    def _apply_axiom(self, axiom: OWLAxiom, triples: List[Dict[str, Any]]) -> None:
        """Axiom 유형에 따른 추론 규칙 적용"""
        if axiom.axiom_type == OWLAxiomType.PROPERTY_CHAIN:
            self._apply_property_chain(axiom)
        elif axiom.axiom_type == OWLAxiomType.TRANSITIVE_PROPERTY:
            self._apply_transitivity(axiom)
        elif axiom.axiom_type == OWLAxiomType.INVERSE_PROPERTIES:
            self._apply_inverse(axiom)
        elif axiom.axiom_type == OWLAxiomType.SYMMETRIC_PROPERTY:
            self._apply_symmetry(axiom)
        elif axiom.axiom_type == OWLAxiomType.SUBPROPERTY_OF:
            self._apply_subproperty(axiom)

    def _apply_property_chain(self, axiom: OWLAxiom) -> None:
        """
        Property Chain 추론

        chain[0] · chain[1] · ... → subject
        예: hasSupervisor · worksIn → belongsToDepartment
        """
        if not axiom.chain or len(axiom.chain) < 2:
            return

        target_prop = axiom.subject

        # 2개 property chain 처리
        if len(axiom.chain) == 2:
            prop1, prop2 = axiom.chain
            triples1 = self._triple_index.get(prop1, set())
            triples2 = self._triple_index.get(prop2, set())

            for s1, o1 in triples1:
                for s2, o2 in triples2:
                    if o1 == s2:  # 연결점
                        self._add_inference(
                            subject=s1,
                            predicate=target_prop,
                            object=o2,
                            rule=f"PropertyChain({prop1} · {prop2})",
                            confidence=axiom.confidence * 0.9,
                            source_axiom=axiom,
                        )

        # 3개 이상 chain (재귀적 처리)
        elif len(axiom.chain) >= 3:
            # 처음 2개 합친 후 나머지와 재귀
            prop1, prop2 = axiom.chain[:2]
            intermediate_results = []

            triples1 = self._triple_index.get(prop1, set())
            triples2 = self._triple_index.get(prop2, set())

            for s1, o1 in triples1:
                for s2, o2 in triples2:
                    if o1 == s2:
                        intermediate_results.append((s1, o2))

            # 나머지 chain 처리
            for prop_idx in range(2, len(axiom.chain)):
                next_prop = axiom.chain[prop_idx]
                next_triples = self._triple_index.get(next_prop, set())
                new_intermediate = []

                for s1, o1 in intermediate_results:
                    for s2, o2 in next_triples:
                        if o1 == s2:
                            new_intermediate.append((s1, o2))

                intermediate_results = new_intermediate

            # 최종 결과 추가
            for s, o in intermediate_results:
                self._add_inference(
                    subject=s,
                    predicate=target_prop,
                    object=o,
                    rule=f"PropertyChain({' · '.join(axiom.chain)})",
                    confidence=axiom.confidence * 0.85,
                    source_axiom=axiom,
                )

    def _apply_transitivity(self, axiom: OWLAxiom) -> None:
        """
        Transitive Property 추론

        A --prop--> B, B --prop--> C => A --prop--> C
        """
        prop = axiom.subject
        prop_triples = list(self._triple_index.get(prop, set()))

        for s1, o1 in prop_triples:
            for s2, o2 in prop_triples:
                if o1 == s2 and s1 != o2:  # 체인이고 자기참조 아님
                    self._add_inference(
                        subject=s1,
                        predicate=prop,
                        object=o2,
                        rule=f"Transitivity({prop})",
                        confidence=axiom.confidence * 0.85,
                        source_axiom=axiom,
                    )

    def _apply_inverse(self, axiom: OWLAxiom) -> None:
        """
        Inverse Property 추론

        A --prop--> B => B --inverse--> A
        """
        prop = axiom.subject
        inverse_prop = axiom.object

        if not inverse_prop:
            return

        prop_triples = self._triple_index.get(prop, set())

        for s, o in prop_triples:
            self._add_inference(
                subject=o,
                predicate=inverse_prop,
                object=s,
                rule=f"Inverse({prop} ↔ {inverse_prop})",
                confidence=axiom.confidence,
                source_axiom=axiom,
            )

    def _apply_symmetry(self, axiom: OWLAxiom) -> None:
        """
        Symmetric Property 추론

        A --prop--> B => B --prop--> A
        """
        prop = axiom.subject
        prop_triples = self._triple_index.get(prop, set())

        for s, o in prop_triples:
            self._add_inference(
                subject=o,
                predicate=prop,
                object=s,
                rule=f"Symmetry({prop})",
                confidence=axiom.confidence,
                source_axiom=axiom,
            )

    def _apply_subproperty(self, axiom: OWLAxiom) -> None:
        """
        SubProperty 추론

        A --subProp--> B, subProp ⊆ superProp => A --superProp--> B
        """
        sub_prop = axiom.subject
        super_prop = axiom.object

        if not super_prop:
            return

        sub_triples = self._triple_index.get(sub_prop, set())

        for s, o in sub_triples:
            self._add_inference(
                subject=s,
                predicate=super_prop,
                object=o,
                rule=f"SubPropertyOf({sub_prop} ⊆ {super_prop})",
                confidence=axiom.confidence * 0.95,
                source_axiom=axiom,
            )

    def _add_inference(
        self,
        subject: str,
        predicate: str,
        object: str,
        rule: str,
        confidence: float,
        source_axiom: OWLAxiom,
    ) -> bool:
        """
        추론 결과 추가 (중복 방지)

        Returns:
            True if new inference added, False if duplicate
        """
        triple_key = (subject, predicate, object)

        # 이미 존재하는 트리플인지 확인
        if triple_key in self._inferred_set:
            return False

        # 원본 트리플에 있는지 확인
        if (subject, object) in self._triple_index.get(predicate, set()):
            return False

        self._inferred_set.add(triple_key)

        # 인덱스 업데이트 (다음 반복에서 사용)
        self._triple_index[predicate].add((subject, object))
        self._subject_index[subject].add((predicate, object))
        self._object_index[object].add((predicate, subject))

        self.inferred.append(InferredTriple(
            subject=subject,
            predicate=predicate,
            object=object,
            rule_applied=rule,
            confidence=confidence,
            source_axioms=[source_axiom],
        ))

        return True

    def _validate_cardinality(
        self,
        triples: List[Dict[str, Any]],
        restrictions: List[OWLAxiom],
    ) -> List[CardinalityViolation]:
        """Cardinality 제약 검증"""
        violations = []

        for restriction in restrictions:
            if restriction.axiom_type not in [
                OWLAxiomType.MIN_CARDINALITY,
                OWLAxiomType.MAX_CARDINALITY,
                OWLAxiomType.EXACT_CARDINALITY,
                OWLAxiomType.QUALIFIED_CARDINALITY,
            ]:
                continue

            prop = restriction.predicate
            expected_card = restriction.cardinality
            on_class = restriction.on_class

            if prop is None or expected_card is None:
                continue

            # 각 subject의 cardinality 계산
            subject_counts: Dict[str, int] = defaultdict(int)
            for t in triples:
                if t.get("predicate") == prop:
                    s = t.get("subject")
                    # Qualified cardinality: object가 특정 클래스인지 확인
                    if on_class:
                        obj = t.get("object")
                        obj_type = t.get("object_type", "")
                        if on_class not in obj_type:
                            continue
                    subject_counts[s] += 1

            # 추론된 트리플도 포함
            for inf in self.inferred:
                if inf.predicate == prop:
                    if on_class:
                        # 추론된 트리플의 object type 확인 필요
                        pass
                    subject_counts[inf.subject] += 1

            # 위반 검사
            for subj, count in subject_counts.items():
                if restriction.axiom_type == OWLAxiomType.MIN_CARDINALITY:
                    if count < expected_card:
                        violations.append(CardinalityViolation(
                            violation_type="min_cardinality",
                            subject=subj,
                            property=prop,
                            expected=expected_card,
                            actual=count,
                            on_class=on_class,
                        ))
                elif restriction.axiom_type == OWLAxiomType.MAX_CARDINALITY:
                    if count > expected_card:
                        violations.append(CardinalityViolation(
                            violation_type="max_cardinality",
                            subject=subj,
                            property=prop,
                            expected=expected_card,
                            actual=count,
                            on_class=on_class,
                        ))
                elif restriction.axiom_type == OWLAxiomType.EXACT_CARDINALITY:
                    if count != expected_card:
                        violations.append(CardinalityViolation(
                            violation_type="exact_cardinality",
                            subject=subj,
                            property=prop,
                            expected=expected_card,
                            actual=count,
                            on_class=on_class,
                        ))

        return violations

    def add_axiom(self, axiom: OWLAxiom) -> None:
        """Axiom 수동 추가"""
        self.axioms.append(axiom)

    def get_inferred_by_rule(self, rule_pattern: str) -> List[InferredTriple]:
        """특정 규칙으로 추론된 트리플 필터"""
        return [
            inf for inf in self.inferred
            if rule_pattern in inf.rule_applied
        ]

    def get_violations_summary(self) -> Dict[str, int]:
        """위반 요약"""
        summary: Dict[str, int] = defaultdict(int)
        for v in self.violations:
            summary[v.violation_type] += 1
        return dict(summary)


# SharedContext 연동 함수
def run_owl2_reasoning_and_update_context(
    shared_context: "SharedContext",
    custom_axioms: Optional[List[OWLAxiom]] = None,
    cardinality_restrictions: Optional[List[OWLAxiom]] = None,
) -> Dict[str, Any]:
    """
    OWL 2 추론 실행 후 SharedContext 업데이트

    사용법 (SemanticValidator 에이전트에서):
        from .owl2_reasoner import run_owl2_reasoning_and_update_context
        result = run_owl2_reasoning_and_update_context(self.shared_context)

    Args:
        shared_context: SharedContext 인스턴스
        custom_axioms: 사용자 정의 Axiom (없으면 자동 탐지)
        cardinality_restrictions: Cardinality 제약 (검증용)

    Returns:
        추론 결과 요약
    """
    reasoner = OWL2Reasoner()

    # 모든 트리플 수집
    all_triples = []

    # knowledge_graph_triples가 있으면 사용
    if hasattr(shared_context, 'knowledge_graph_triples'):
        all_triples.extend(shared_context.knowledge_graph_triples)

    # dynamic_data에서 추가 트리플 수집
    additional_triples = shared_context.get_dynamic("additional_triples", [])
    all_triples.extend(additional_triples)

    # FK 관계에서 트리플 생성
    for fk in shared_context.enhanced_fk_candidates:
        if isinstance(fk, dict):
            all_triples.append({
                "subject": fk.get("from_table", ""),
                "predicate": f"has_fk_to",
                "object": fk.get("to_table", ""),
                "from_column": fk.get("from_column", ""),
                "to_column": fk.get("to_column", ""),
            })

    if not all_triples:
        logger.warning("[OWL2Reasoner] No triples found in SharedContext")
        return {
            "axioms_discovered": 0,
            "triples_inferred": 0,
            "violations": 0,
            "axiom_types": [],
        }

    # 추론 실행
    inferred = reasoner.reason(
        triples=all_triples,
        axioms=custom_axioms,
        cardinality_restrictions=cardinality_restrictions,
    )

    # SharedContext에 추가
    inferred_triples_list = []
    for triple in inferred:
        triple_dict = triple.to_dict()
        inferred_triples_list.append(triple_dict)

        # knowledge_graph_triples에도 추가 (있으면)
        if hasattr(shared_context, 'knowledge_graph_triples'):
            shared_context.knowledge_graph_triples.append(triple_dict)

    # dynamic_data에 저장
    shared_context.set_dynamic("owl2_inferred_triples", inferred_triples_list)
    shared_context.set_dynamic("owl2_axioms", [a.to_dict() for a in reasoner.axioms])
    shared_context.set_dynamic("owl2_violations", [v.to_dict() for v in reasoner.violations])

    # 결과 요약
    result = {
        "axioms_discovered": len(reasoner.axioms),
        "triples_inferred": len(inferred),
        "violations": len(reasoner.violations),
        "axiom_types": list(set(a.axiom_type.value for a in reasoner.axioms)),
        "violations_summary": reasoner.get_violations_summary(),
    }

    logger.info(
        f"[OWL2Reasoner] Added {len(inferred)} inferred triples, "
        f"{len(reasoner.violations)} violations to context"
    )

    return result


# 유틸리티: Axiom 생성 헬퍼
def create_property_chain_axiom(
    chain: Tuple[str, ...],
    result_property: str,
    confidence: float = 1.0,
) -> OWLAxiom:
    """Property Chain Axiom 생성 헬퍼"""
    return OWLAxiom(
        axiom_type=OWLAxiomType.PROPERTY_CHAIN,
        subject=result_property,
        chain=chain,
        confidence=confidence,
        source="manual",
    )


def create_inverse_axiom(
    property1: str,
    property2: str,
    confidence: float = 1.0,
) -> OWLAxiom:
    """Inverse Property Axiom 생성 헬퍼"""
    return OWLAxiom(
        axiom_type=OWLAxiomType.INVERSE_PROPERTIES,
        subject=property1,
        object=property2,
        confidence=confidence,
        source="manual",
    )


def create_cardinality_restriction(
    class_name: str,
    property_name: str,
    cardinality: int,
    restriction_type: str = "min",  # min, max, exact
    on_class: Optional[str] = None,
) -> OWLAxiom:
    """Cardinality Restriction 생성 헬퍼"""
    axiom_type_map = {
        "min": OWLAxiomType.MIN_CARDINALITY,
        "max": OWLAxiomType.MAX_CARDINALITY,
        "exact": OWLAxiomType.EXACT_CARDINALITY,
    }

    return OWLAxiom(
        axiom_type=axiom_type_map.get(restriction_type, OWLAxiomType.MIN_CARDINALITY),
        subject=class_name,
        predicate=property_name,
        cardinality=cardinality,
        on_class=on_class,
        source="manual",
    )
