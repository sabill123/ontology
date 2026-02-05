"""
Semantic Reasoner Module - 의미적 추론 엔진

온톨로지/스키마 기반 의미적 추론을 수행합니다.

주요 기능:
- 규칙 기반 추론 (전방/후방 체이닝)
- 온톨로지 기반 추론 (RDFS/OWL 스타일)
- 일관성 검사
- 완전성 검사
- 추론 체인 설명

References:
- Baader et al., "The Description Logic Handbook", Cambridge 2003
- Brachman & Levesque, "Knowledge Representation and Reasoning", 2004
- RDF 1.1 Semantics, W3C Recommendation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict, deque
import re


class ReasoningType(Enum):
    """추론 타입"""
    FORWARD_CHAINING = "forward_chaining"      # 전방 체이닝 (데이터 주도)
    BACKWARD_CHAINING = "backward_chaining"    # 후방 체이닝 (목표 주도)
    HYBRID = "hybrid"                          # 하이브리드


class RuleType(Enum):
    """규칙 타입"""
    RDFS_SUBCLASS = "rdfs_subclass"           # rdfs:subClassOf
    RDFS_SUBPROPERTY = "rdfs_subproperty"     # rdfs:subPropertyOf
    RDFS_DOMAIN = "rdfs_domain"               # rdfs:domain
    RDFS_RANGE = "rdfs_range"                 # rdfs:range
    OWL_INVERSE = "owl_inverse"               # owl:inverseOf
    OWL_TRANSITIVE = "owl_transitive"         # owl:TransitiveProperty
    OWL_SYMMETRIC = "owl_symmetric"           # owl:SymmetricProperty
    OWL_FUNCTIONAL = "owl_functional"         # owl:FunctionalProperty
    OWL_EQUIVALENT = "owl_equivalent"         # owl:equivalentClass
    CUSTOM = "custom"                          # 사용자 정의


class InferenceStatus(Enum):
    """추론 상태"""
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    INCONSISTENT = "inconsistent"


@dataclass
class Triple:
    """RDF 스타일 트리플 (주어, 술어, 목적어)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "asserted"  # asserted, inferred
    rule_applied: Optional[str] = None


@dataclass
class Rule:
    """추론 규칙"""
    id: str
    name: str
    rule_type: RuleType
    antecedent: List[Tuple[str, str, str]]  # [(s_pattern, p_pattern, o_pattern), ...]
    consequent: List[Tuple[str, str, str]]
    confidence: float = 1.0
    priority: int = 0
    enabled: bool = True


@dataclass
class InferenceChain:
    """추론 체인 (설명용)"""
    conclusion: Triple
    premises: List[Triple]
    rule: Rule
    depth: int
    confidence: float


@dataclass
class ConsistencyReport:
    """일관성 검사 보고서"""
    is_consistent: bool
    violations: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]


@dataclass
class CompletenessReport:
    """완전성 검사 보고서"""
    completeness_score: float
    missing_required: List[Dict[str, Any]]
    recommendations: List[str]


class KnowledgeBase:
    """지식 베이스 (트리플 저장소)"""

    def __init__(self):
        self.triples: Set[Tuple[str, str, str]] = set()
        self.triple_details: Dict[Tuple[str, str, str], Triple] = {}

        # 인덱스
        self.spo_index: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)  # subject
        self.pos_index: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)  # predicate
        self.osp_index: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)  # object

    def add_triple(self, triple: Triple):
        """트리플 추가"""
        key = (triple.subject, triple.predicate, triple.object)
        if key not in self.triples:
            self.triples.add(key)
            self.triple_details[key] = triple

            # 인덱스 업데이트
            self.spo_index[triple.subject].add(key)
            self.pos_index[triple.predicate].add(key)
            self.osp_index[triple.object].add(key)

    def query(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None
    ) -> List[Triple]:
        """패턴 매칭 쿼리"""
        # 가장 선택적인 인덱스 사용
        if subject and subject[0] != '?':
            candidates = self.spo_index.get(subject, set())
        elif predicate and predicate[0] != '?':
            candidates = self.pos_index.get(predicate, set())
        elif obj and obj[0] != '?':
            candidates = self.osp_index.get(obj, set())
        else:
            candidates = self.triples

        results = []
        for s, p, o in candidates:
            if (subject is None or subject[0] == '?' or s == subject) and \
               (predicate is None or predicate[0] == '?' or p == predicate) and \
               (obj is None or obj[0] == '?' or o == obj):
                results.append(self.triple_details[(s, p, o)])

        return results

    def contains(self, subject: str, predicate: str, obj: str) -> bool:
        """트리플 존재 여부"""
        return (subject, predicate, obj) in self.triples


class PatternMatcher:
    """패턴 매칭 엔진"""

    @staticmethod
    def match_pattern(
        pattern: Tuple[str, str, str],
        triple: Triple,
        bindings: Dict[str, str] = None
    ) -> Optional[Dict[str, str]]:
        """패턴과 트리플 매칭"""
        if bindings is None:
            bindings = {}

        s_pattern, p_pattern, o_pattern = pattern
        values = [
            (s_pattern, triple.subject),
            (p_pattern, triple.predicate),
            (o_pattern, triple.object)
        ]

        new_bindings = bindings.copy()

        for pattern, value in values:
            if pattern.startswith('?'):
                # 변수
                var_name = pattern[1:]
                if var_name in new_bindings:
                    if new_bindings[var_name] != value:
                        return None  # 바인딩 충돌
                else:
                    new_bindings[var_name] = value
            else:
                # 상수
                if pattern != value:
                    return None

        return new_bindings

    @staticmethod
    def apply_bindings(
        pattern: Tuple[str, str, str],
        bindings: Dict[str, str]
    ) -> Tuple[str, str, str]:
        """바인딩 적용"""
        result = []
        for p in pattern:
            if p.startswith('?') and p[1:] in bindings:
                result.append(bindings[p[1:]])
            else:
                result.append(p)
        return tuple(result)


class ForwardChainer:
    """전방 체이닝 추론 엔진"""

    def __init__(self, kb: KnowledgeBase, rules: List[Rule]):
        self.kb = kb
        self.rules = sorted(rules, key=lambda r: -r.priority)
        self.inference_chains: List[InferenceChain] = []

    def infer(self, max_iterations: int = 100) -> List[Triple]:
        """전방 체이닝 추론 수행"""
        new_triples = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            round_new = []

            for rule in self.rules:
                if not rule.enabled:
                    continue

                # 규칙 적용
                inferred = self._apply_rule(rule)
                round_new.extend(inferred)

            if not round_new:
                break  # 고정점 도달

            for triple in round_new:
                self.kb.add_triple(triple)
                new_triples.append(triple)

        return new_triples

    def _apply_rule(self, rule: Rule) -> List[Triple]:
        """규칙 적용"""
        if not rule.antecedent:
            return []

        # 첫 번째 전제로 시작
        first_pattern = rule.antecedent[0]
        candidates = self.kb.query(
            first_pattern[0] if first_pattern[0][0] != '?' else None,
            first_pattern[1] if first_pattern[1][0] != '?' else None,
            first_pattern[2] if first_pattern[2][0] != '?' else None
        )

        # 바인딩 집합 초기화
        all_bindings = []
        for triple in candidates:
            bindings = PatternMatcher.match_pattern(first_pattern, triple)
            if bindings:
                all_bindings.append((bindings, [triple]))

        # 나머지 전제 매칭
        for pattern in rule.antecedent[1:]:
            new_bindings = []
            for bindings, premises in all_bindings:
                # 현재 바인딩으로 패턴 구체화
                concrete = PatternMatcher.apply_bindings(pattern, bindings)

                matches = self.kb.query(
                    concrete[0] if concrete[0][0] != '?' else None,
                    concrete[1] if concrete[1][0] != '?' else None,
                    concrete[2] if concrete[2][0] != '?' else None
                )

                for match in matches:
                    extended = PatternMatcher.match_pattern(pattern, match, bindings)
                    if extended:
                        new_bindings.append((extended, premises + [match]))

            all_bindings = new_bindings

        # 결론 생성
        inferred = []
        for bindings, premises in all_bindings:
            for consequent in rule.consequent:
                conclusion = PatternMatcher.apply_bindings(consequent, bindings)

                # 새로운 트리플인지 확인
                if not self.kb.contains(conclusion[0], conclusion[1], conclusion[2]):
                    # 신뢰도 계산
                    premise_confidence = min(p.confidence for p in premises)
                    final_confidence = premise_confidence * rule.confidence

                    new_triple = Triple(
                        subject=conclusion[0],
                        predicate=conclusion[1],
                        object=conclusion[2],
                        confidence=final_confidence,
                        source="inferred",
                        rule_applied=rule.id
                    )
                    inferred.append(new_triple)

                    # 추론 체인 기록
                    self.inference_chains.append(InferenceChain(
                        conclusion=new_triple,
                        premises=premises,
                        rule=rule,
                        depth=1,
                        confidence=final_confidence
                    ))

        return inferred


class BackwardChainer:
    """후방 체이닝 추론 엔진"""

    def __init__(self, kb: KnowledgeBase, rules: List[Rule]):
        self.kb = kb
        self.rules = rules
        self.proof_tree: List[Dict] = []

    def prove(
        self,
        goal: Tuple[str, str, str],
        max_depth: int = 10
    ) -> Tuple[bool, List[Dict]]:
        """목표 증명 시도"""
        self.proof_tree = []
        result = self._prove_goal(goal, {}, max_depth, 0)
        return result, self.proof_tree

    def _prove_goal(
        self,
        goal: Tuple[str, str, str],
        bindings: Dict[str, str],
        max_depth: int,
        depth: int
    ) -> bool:
        """재귀적 목표 증명"""
        if depth > max_depth:
            return False

        # 바인딩 적용
        concrete_goal = PatternMatcher.apply_bindings(goal, bindings)

        # KB에서 직접 확인
        matches = self.kb.query(
            concrete_goal[0] if concrete_goal[0][0] != '?' else None,
            concrete_goal[1] if concrete_goal[1][0] != '?' else None,
            concrete_goal[2] if concrete_goal[2][0] != '?' else None
        )

        for match in matches:
            new_bindings = PatternMatcher.match_pattern(goal, match, bindings)
            if new_bindings:
                self.proof_tree.append({
                    "type": "fact",
                    "goal": concrete_goal,
                    "match": (match.subject, match.predicate, match.object),
                    "depth": depth
                })
                return True

        # 규칙 역적용
        for rule in self.rules:
            if not rule.enabled:
                continue

            for consequent in rule.consequent:
                # 결론이 목표와 매칭되는지 확인
                unified = self._unify(goal, consequent, bindings)
                if unified is not None:
                    # 모든 전제 증명 시도
                    all_premises_proved = True
                    current_bindings = unified

                    for antecedent in rule.antecedent:
                        if not self._prove_goal(antecedent, current_bindings, max_depth, depth + 1):
                            all_premises_proved = False
                            break

                    if all_premises_proved:
                        self.proof_tree.append({
                            "type": "rule",
                            "goal": concrete_goal,
                            "rule": rule.id,
                            "depth": depth
                        })
                        return True

        return False

    def _unify(
        self,
        pattern1: Tuple[str, str, str],
        pattern2: Tuple[str, str, str],
        bindings: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """두 패턴 통일"""
        new_bindings = bindings.copy()

        for p1, p2 in zip(pattern1, pattern2):
            # 변수-변수
            if p1.startswith('?') and p2.startswith('?'):
                # 같은 변수로 취급
                var1, var2 = p1[1:], p2[1:]
                if var1 in new_bindings:
                    if var2 in new_bindings:
                        if new_bindings[var1] != new_bindings[var2]:
                            return None
                    else:
                        new_bindings[var2] = new_bindings[var1]
                elif var2 in new_bindings:
                    new_bindings[var1] = new_bindings[var2]
                continue

            # 변수-상수
            if p1.startswith('?'):
                var = p1[1:]
                if var in new_bindings:
                    if new_bindings[var] != p2:
                        return None
                else:
                    new_bindings[var] = p2
                continue

            if p2.startswith('?'):
                var = p2[1:]
                if var in new_bindings:
                    if new_bindings[var] != p1:
                        return None
                else:
                    new_bindings[var] = p1
                continue

            # 상수-상수
            if p1 != p2:
                return None

        return new_bindings


class RDFSReasoner:
    """RDFS 추론기"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.rules = self._create_rdfs_rules()
        self.forward_chainer = ForwardChainer(kb, self.rules)

    def _create_rdfs_rules(self) -> List[Rule]:
        """RDFS 추론 규칙 생성"""
        rules = [
            # rdfs2: domain 추론
            Rule(
                id="rdfs2",
                name="Domain Inference",
                rule_type=RuleType.RDFS_DOMAIN,
                antecedent=[
                    ("?p", "rdfs:domain", "?c"),
                    ("?x", "?p", "?y")
                ],
                consequent=[("?x", "rdf:type", "?c")],
                priority=10
            ),

            # rdfs3: range 추론
            Rule(
                id="rdfs3",
                name="Range Inference",
                rule_type=RuleType.RDFS_RANGE,
                antecedent=[
                    ("?p", "rdfs:range", "?c"),
                    ("?x", "?p", "?y")
                ],
                consequent=[("?y", "rdf:type", "?c")],
                priority=10
            ),

            # rdfs5: subPropertyOf 전이성
            Rule(
                id="rdfs5",
                name="SubProperty Transitivity",
                rule_type=RuleType.RDFS_SUBPROPERTY,
                antecedent=[
                    ("?p", "rdfs:subPropertyOf", "?q"),
                    ("?q", "rdfs:subPropertyOf", "?r")
                ],
                consequent=[("?p", "rdfs:subPropertyOf", "?r")],
                priority=5
            ),

            # rdfs7: subPropertyOf 상속
            Rule(
                id="rdfs7",
                name="SubProperty Inheritance",
                rule_type=RuleType.RDFS_SUBPROPERTY,
                antecedent=[
                    ("?p", "rdfs:subPropertyOf", "?q"),
                    ("?x", "?p", "?y")
                ],
                consequent=[("?x", "?q", "?y")],
                priority=8
            ),

            # rdfs9: subClassOf 추론
            Rule(
                id="rdfs9",
                name="SubClass Type Inference",
                rule_type=RuleType.RDFS_SUBCLASS,
                antecedent=[
                    ("?c1", "rdfs:subClassOf", "?c2"),
                    ("?x", "rdf:type", "?c1")
                ],
                consequent=[("?x", "rdf:type", "?c2")],
                priority=8
            ),

            # rdfs11: subClassOf 전이성
            Rule(
                id="rdfs11",
                name="SubClass Transitivity",
                rule_type=RuleType.RDFS_SUBCLASS,
                antecedent=[
                    ("?c1", "rdfs:subClassOf", "?c2"),
                    ("?c2", "rdfs:subClassOf", "?c3")
                ],
                consequent=[("?c1", "rdfs:subClassOf", "?c3")],
                priority=5
            )
        ]

        return rules

    def infer(self) -> List[Triple]:
        """RDFS 추론 수행"""
        return self.forward_chainer.infer()


class OWLReasoner:
    """OWL 추론기 (일부)"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.rules = self._create_owl_rules()
        self.forward_chainer = ForwardChainer(kb, self.rules)

    def _create_owl_rules(self) -> List[Rule]:
        """OWL 추론 규칙 생성"""
        rules = [
            # owl:inverseOf
            Rule(
                id="owl_inverse",
                name="Inverse Property",
                rule_type=RuleType.OWL_INVERSE,
                antecedent=[
                    ("?p", "owl:inverseOf", "?q"),
                    ("?x", "?p", "?y")
                ],
                consequent=[("?y", "?q", "?x")],
                priority=10
            ),

            # owl:TransitiveProperty
            Rule(
                id="owl_transitive",
                name="Transitive Property",
                rule_type=RuleType.OWL_TRANSITIVE,
                antecedent=[
                    ("?p", "rdf:type", "owl:TransitiveProperty"),
                    ("?x", "?p", "?y"),
                    ("?y", "?p", "?z")
                ],
                consequent=[("?x", "?p", "?z")],
                priority=8
            ),

            # owl:SymmetricProperty
            Rule(
                id="owl_symmetric",
                name="Symmetric Property",
                rule_type=RuleType.OWL_SYMMETRIC,
                antecedent=[
                    ("?p", "rdf:type", "owl:SymmetricProperty"),
                    ("?x", "?p", "?y")
                ],
                consequent=[("?y", "?p", "?x")],
                priority=10
            ),

            # owl:equivalentClass
            Rule(
                id="owl_equiv_class",
                name="Equivalent Class",
                rule_type=RuleType.OWL_EQUIVALENT,
                antecedent=[
                    ("?c1", "owl:equivalentClass", "?c2"),
                    ("?x", "rdf:type", "?c1")
                ],
                consequent=[("?x", "rdf:type", "?c2")],
                priority=9
            ),

            # owl:sameAs 전이성
            Rule(
                id="owl_same_as_trans",
                name="SameAs Transitivity",
                rule_type=RuleType.CUSTOM,
                antecedent=[
                    ("?x", "owl:sameAs", "?y"),
                    ("?y", "owl:sameAs", "?z")
                ],
                consequent=[("?x", "owl:sameAs", "?z")],
                priority=7
            )
        ]

        return rules

    def infer(self) -> List[Triple]:
        """OWL 추론 수행"""
        return self.forward_chainer.infer()


class ConsistencyChecker:
    """일관성 검사기"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def check(self) -> ConsistencyReport:
        """일관성 검사"""
        violations = []
        warnings = []

        # 1. owl:FunctionalProperty 위반 검사
        functional_violations = self._check_functional_properties()
        violations.extend(functional_violations)

        # 2. owl:disjointWith 위반 검사
        disjoint_violations = self._check_disjoint_classes()
        violations.extend(disjoint_violations)

        # 3. 도메인/범위 일관성 검사
        domain_range_warnings = self._check_domain_range_consistency()
        warnings.extend(domain_range_warnings)

        # 4. 순환 참조 검사
        cycle_warnings = self._check_cycles()
        warnings.extend(cycle_warnings)

        return ConsistencyReport(
            is_consistent=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )

    def _check_functional_properties(self) -> List[Dict]:
        """함수적 속성 위반 검사"""
        violations = []

        # 함수적 속성 찾기
        functional_props = self.kb.query(predicate="rdf:type", obj="owl:FunctionalProperty")

        for prop_triple in functional_props:
            prop = prop_triple.subject

            # 해당 속성의 모든 트리플
            prop_triples = self.kb.query(predicate=prop)

            # 주어별 그룹화
            by_subject = defaultdict(list)
            for t in prop_triples:
                by_subject[t.subject].append(t.object)

            # 중복 체크
            for subj, objects in by_subject.items():
                if len(set(objects)) > 1:
                    violations.append({
                        "type": "functional_property_violation",
                        "property": prop,
                        "subject": subj,
                        "values": list(set(objects)),
                        "message": f"함수적 속성 '{prop}'이 주어 '{subj}'에 대해 여러 값을 가짐"
                    })

        return violations

    def _check_disjoint_classes(self) -> List[Dict]:
        """상호 배제 클래스 위반 검사"""
        violations = []

        # disjointWith 관계 찾기
        disjoint_triples = self.kb.query(predicate="owl:disjointWith")

        for dt in disjoint_triples:
            class1, class2 = dt.subject, dt.object

            # 두 클래스의 인스턴스
            instances1 = {t.subject for t in self.kb.query(predicate="rdf:type", obj=class1)}
            instances2 = {t.subject for t in self.kb.query(predicate="rdf:type", obj=class2)}

            # 교집합 체크
            overlap = instances1 & instances2
            if overlap:
                violations.append({
                    "type": "disjoint_class_violation",
                    "class1": class1,
                    "class2": class2,
                    "overlapping_instances": list(overlap),
                    "message": f"상호 배제 클래스 '{class1}'과 '{class2}'의 공통 인스턴스 발견"
                })

        return violations

    def _check_domain_range_consistency(self) -> List[Dict]:
        """도메인/범위 일관성 경고"""
        warnings = []

        # 도메인 정의 확인
        domain_defs = self.kb.query(predicate="rdfs:domain")

        for dd in domain_defs:
            prop, domain_class = dd.subject, dd.object

            # 해당 속성 사용
            prop_uses = self.kb.query(predicate=prop)

            for use in prop_uses:
                subject = use.subject
                # 주어의 타입 확인
                subject_types = {t.object for t in self.kb.query(subject=subject, predicate="rdf:type")}

                if domain_class not in subject_types:
                    # subClassOf 관계도 확인
                    is_subtype = False
                    for st in subject_types:
                        if self.kb.contains(st, "rdfs:subClassOf", domain_class):
                            is_subtype = True
                            break

                    if not is_subtype:
                        warnings.append({
                            "type": "domain_warning",
                            "property": prop,
                            "subject": subject,
                            "expected_domain": domain_class,
                            "actual_types": list(subject_types),
                            "message": f"속성 '{prop}'의 주어 '{subject}'가 도메인 '{domain_class}'의 인스턴스가 아닐 수 있음"
                        })

        return warnings

    def _check_cycles(self) -> List[Dict]:
        """순환 참조 검사"""
        warnings = []

        # subClassOf 순환 검사
        subclass_triples = self.kb.query(predicate="rdfs:subClassOf")

        graph = defaultdict(set)
        for t in subclass_triples:
            graph[t.subject].add(t.object)

        # DFS로 순환 탐지
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, path + [neighbor])
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    return path + [neighbor]

            rec_stack.remove(node)
            return None

        for node in graph:
            if node not in visited:
                cycle = dfs(node, [node])
                if cycle:
                    warnings.append({
                        "type": "cycle_warning",
                        "predicate": "rdfs:subClassOf",
                        "cycle": cycle,
                        "message": f"subClassOf 관계에 순환 발견: {' -> '.join(cycle)}"
                    })
                    break

        return warnings


class CompletenessChecker:
    """완전성 검사기"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def check(
        self,
        required_properties: Dict[str, List[str]] = None
    ) -> CompletenessReport:
        """완전성 검사"""
        missing = []
        recommendations = []

        total_checks = 0
        passed_checks = 0

        # 클래스별 필수 속성 검사
        if required_properties:
            for class_name, props in required_properties.items():
                # 해당 클래스의 인스턴스
                instances = self.kb.query(predicate="rdf:type", obj=class_name)

                for inst in instances:
                    for prop in props:
                        total_checks += 1
                        prop_values = self.kb.query(subject=inst.subject, predicate=prop)

                        if not prop_values:
                            missing.append({
                                "instance": inst.subject,
                                "class": class_name,
                                "missing_property": prop,
                                "message": f"인스턴스 '{inst.subject}'에 필수 속성 '{prop}' 누락"
                            })
                        else:
                            passed_checks += 1

        # 라벨/주석 완전성
        all_subjects = set()
        for t in self.kb.query():
            all_subjects.add(t.subject)

        for subj in all_subjects:
            labels = self.kb.query(subject=subj, predicate="rdfs:label")
            if not labels:
                recommendations.append(f"'{subj}'에 rdfs:label 추가 권장")

        # 완전성 점수
        completeness = passed_checks / total_checks if total_checks > 0 else 1.0

        return CompletenessReport(
            completeness_score=completeness,
            missing_required=missing,
            recommendations=recommendations[:10]  # 상위 10개만
        )


class SemanticReasoner:
    """통합 의미적 추론 엔진"""

    def __init__(self):
        self.kb = KnowledgeBase()
        self.rdfs_reasoner = RDFSReasoner(self.kb)
        self.owl_reasoner = OWLReasoner(self.kb)
        self.consistency_checker = ConsistencyChecker(self.kb)
        self.completeness_checker = CompletenessChecker(self.kb)
        self.custom_rules: List[Rule] = []

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0
    ):
        """트리플 추가"""
        triple = Triple(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source="asserted"
        )
        self.kb.add_triple(triple)

    def add_custom_rule(self, rule: Rule):
        """사용자 정의 규칙 추가"""
        self.custom_rules.append(rule)

    def infer(self) -> Dict[str, Any]:
        """추론 수행"""
        # RDFS 추론
        rdfs_inferred = self.rdfs_reasoner.infer()

        # OWL 추론
        owl_inferred = self.owl_reasoner.infer()

        # 사용자 정의 규칙
        custom_inferred = []
        if self.custom_rules:
            custom_chainer = ForwardChainer(self.kb, self.custom_rules)
            custom_inferred = custom_chainer.infer()

        return {
            "rdfs_inferred": [(t.subject, t.predicate, t.object) for t in rdfs_inferred],
            "owl_inferred": [(t.subject, t.predicate, t.object) for t in owl_inferred],
            "custom_inferred": [(t.subject, t.predicate, t.object) for t in custom_inferred],
            "total_inferred": len(rdfs_inferred) + len(owl_inferred) + len(custom_inferred),
            "total_triples": len(self.kb.triples)
        }

    def check_consistency(self) -> ConsistencyReport:
        """일관성 검사"""
        return self.consistency_checker.check()

    def check_completeness(
        self,
        required_properties: Dict[str, List[str]] = None
    ) -> CompletenessReport:
        """완전성 검사"""
        return self.completeness_checker.check(required_properties)

    def explain_inference(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> List[Dict]:
        """추론 설명"""
        explanations = []

        # KB에서 해당 트리플 찾기
        key = (subject, predicate, obj)
        if key not in self.kb.triple_details:
            return [{"message": "해당 트리플이 존재하지 않습니다"}]

        triple = self.kb.triple_details[key]

        if triple.source == "asserted":
            explanations.append({
                "type": "fact",
                "message": "명시적으로 선언된 사실입니다"
            })
        else:
            explanations.append({
                "type": "inferred",
                "rule": triple.rule_applied,
                "confidence": triple.confidence,
                "message": f"규칙 '{triple.rule_applied}'에 의해 추론되었습니다"
            })

            # 추론 체인 찾기 (ForwardChainer에서)
            for chainer in [self.rdfs_reasoner.forward_chainer, self.owl_reasoner.forward_chainer]:
                for chain in chainer.inference_chains:
                    if (chain.conclusion.subject, chain.conclusion.predicate, chain.conclusion.object) == key:
                        explanations.append({
                            "type": "chain",
                            "premises": [
                                (p.subject, p.predicate, p.object)
                                for p in chain.premises
                            ],
                            "rule": chain.rule.name,
                            "depth": chain.depth
                        })

        return explanations

    def query(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None
    ) -> List[Tuple[str, str, str]]:
        """쿼리"""
        results = self.kb.query(subject, predicate, obj)
        return [(t.subject, t.predicate, t.object) for t in results]


class SchemaReasoner:
    """스키마 기반 추론기"""

    def __init__(self):
        self.reasoner = SemanticReasoner()

    def load_schema(self, tables: Dict[str, Dict]):
        """스키마를 지식 베이스로 로드"""
        for table_name, table_info in tables.items():
            # 테이블을 클래스로
            self.reasoner.add_triple(
                f"table:{table_name}",
                "rdf:type",
                "rdfs:Class"
            )

            columns = table_info.get("columns", [])
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", col.get("column_name", ""))
                    col_type = col.get("type", col.get("data_type", "string"))
                else:
                    col_name = str(col)
                    col_type = "string"

                # 컬럼을 속성으로
                col_uri = f"column:{table_name}.{col_name}"
                self.reasoner.add_triple(col_uri, "rdf:type", "rdf:Property")
                self.reasoner.add_triple(col_uri, "rdfs:domain", f"table:{table_name}")
                self.reasoner.add_triple(col_uri, "rdfs:range", f"xsd:{col_type}")

        # 추론
        return self.reasoner.infer()

    def add_fk_relation(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str
    ):
        """FK 관계 추가"""
        source_col = f"column:{source_table}.{source_column}"
        target_col = f"column:{target_table}.{target_column}"

        # FK 관계 = 역 관계 쌍
        self.reasoner.add_triple(source_col, "schema:references", target_col)
        self.reasoner.add_triple(target_col, "schema:referencedBy", source_col)

        # 역관계 정의
        self.reasoner.add_triple("schema:references", "owl:inverseOf", "schema:referencedBy")

    def infer_relationships(self) -> Dict[str, Any]:
        """관계 추론"""
        # 추론 실행
        inferred = self.reasoner.infer()

        # 일관성 검사
        consistency = self.reasoner.check_consistency()

        # 완전성 검사
        completeness = self.reasoner.check_completeness()

        return {
            "inferred": inferred,
            "consistency": {
                "is_consistent": consistency.is_consistent,
                "violations": consistency.violations,
                "warnings": consistency.warnings
            },
            "completeness": {
                "score": completeness.completeness_score,
                "missing": completeness.missing_required,
                "recommendations": completeness.recommendations
            }
        }

    def query_relationships(
        self,
        table_name: str = None
    ) -> List[Dict]:
        """관계 쿼리"""
        results = []

        if table_name:
            # 특정 테이블의 관계
            refs = self.reasoner.query(
                subject=f"column:{table_name}.*",
                predicate="schema:references"
            )
            for ref in refs:
                results.append({
                    "type": "outgoing_fk",
                    "from": ref[0],
                    "to": ref[2]
                })

            refd = self.reasoner.query(
                subject=f"column:{table_name}.*",
                predicate="schema:referencedBy"
            )
            for ref in refd:
                results.append({
                    "type": "incoming_fk",
                    "from": ref[2],
                    "to": ref[0]
                })
        else:
            # 모든 FK 관계
            all_refs = self.reasoner.query(predicate="schema:references")
            for ref in all_refs:
                results.append({
                    "type": "fk",
                    "from": ref[0],
                    "to": ref[2]
                })

        return results


class SemanticReasoningAnalyzer:
    """통합 의미적 추론 분석기"""

    def __init__(self):
        self.schema_reasoner = SchemaReasoner()

    def analyze(
        self,
        tables: Dict[str, Dict],
        fk_relations: List[Dict] = None
    ) -> Dict[str, Any]:
        """스키마 의미적 추론 분석"""
        # 스키마 로드
        self.schema_reasoner.load_schema(tables)

        # FK 관계 추가
        if fk_relations:
            for fk in fk_relations:
                self.schema_reasoner.add_fk_relation(
                    fk.get("source_table", ""),
                    fk.get("source_column", ""),
                    fk.get("target_table", ""),
                    fk.get("target_column", "")
                )

        # 추론 및 검사
        result = self.schema_reasoner.infer_relationships()

        # 관계 쿼리
        relationships = self.schema_reasoner.query_relationships()

        return {
            "inference_result": result["inferred"],
            "consistency": result["consistency"],
            "completeness": result["completeness"],
            "discovered_relationships": relationships,
            "summary": {
                "total_triples": result["inferred"]["total_triples"],
                "total_inferred": result["inferred"]["total_inferred"],
                "is_consistent": result["consistency"]["is_consistent"],
                "completeness_score": result["completeness"]["score"],
                "num_relationships": len(relationships)
            }
        }

    # =========================================================================
    # 편의 메서드: refinement.py 에이전트에서 사용하는 API
    # =========================================================================

    def run_inference(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        트리플 기반 추론 실행 (에이전트 호출용)

        Args:
            triples: [{"subject": ..., "predicate": ..., "object": ...}, ...]

        Returns:
            추론된 트리플 리스트
        """
        # 트리플을 KnowledgeBase에 추가
        for triple in triples:
            self.schema_reasoner.reasoner.kb.add_triple(
                Triple(
                    subject=triple.get("subject", ""),
                    predicate=triple.get("predicate", ""),
                    object=triple.get("object", ""),
                )
            )

        # 추론 실행
        result = self.schema_reasoner.reasoner.infer()
        all_inferred = result.get("rdfs_inferred", []) + result.get("owl_inferred", []) + result.get("custom_inferred", [])

        return [
            {"subject": t[0], "predicate": t[1], "object": t[2]}
            for t in all_inferred
        ]

    def check_consistency(self, triples: List[Dict[str, Any]] = None) -> 'ConsistencyReport':
        """
        일관성 검사 (에이전트 호출용)

        Args:
            triples: 검사할 트리플 (None이면 현재 KB 사용)

        Returns:
            ConsistencyReport 객체
        """
        # 트리플이 제공되면 먼저 추가
        if triples:
            for triple in triples:
                self.schema_reasoner.reasoner.kb.add_triple(
                    Triple(
                        subject=triple.get("subject", ""),
                        predicate=triple.get("predicate", ""),
                        object=triple.get("object", ""),
                    )
                )

        return self.schema_reasoner.check_consistency()

    def check_completeness(self, triples: List[Dict[str, Any]] = None) -> 'CompletenessReport':
        """
        완전성 검사 (에이전트 호출용)

        Args:
            triples: 검사할 트리플 (None이면 현재 KB 사용)

        Returns:
            CompletenessReport 객체
        """
        # 트리플이 제공되면 먼저 추가
        if triples:
            for triple in triples:
                self.schema_reasoner.reasoner.kb.add_triple(
                    Triple(
                        subject=triple.get("subject", ""),
                        predicate=triple.get("predicate", ""),
                        object=triple.get("object", ""),
                    )
                )

        return self.schema_reasoner.check_completeness()
