"""
Knowledge Graph Quality Validator

논문: "Structural Quality Metrics to Evaluate Knowledge Graphs" (2022)

6가지 구조적 메트릭:
1. Class Instantiation
2. Property Completeness
3. Relationship Balance
4. Semantic Consistency
5. Hierarchy Depth
6. Link Validity
"""

import logging
from collections import defaultdict
from typing import Any, Dict

from .models import KGQualityScore

logger = logging.getLogger(__name__)


class KnowledgeGraphQualityValidator:
    """
    Knowledge Graph 품질 검증

    논문: "Structural Quality Metrics to Evaluate Knowledge Graphs" (2022)

    6가지 구조적 메트릭:
    1. Class Instantiation
    2. Property Completeness
    3. Relationship Balance
    4. Semantic Consistency
    5. Hierarchy Depth
    6. Link Validity
    """

    def __init__(self):
        self.weights = {
            'class_instantiation': 0.15,
            'property_completeness': 0.20,
            'relationship_balance': 0.15,
            'semantic_consistency': 0.25,
            'hierarchy_depth': 0.10,
            'link_validity': 0.15,
        }

    def validate(self, kg_data: Dict[str, Any]) -> Dict[str, KGQualityScore]:
        """KG 품질 종합 평가"""
        scores = {}

        # 1. Class Instantiation: 클래스당 인스턴스 수
        entities = kg_data.get('unified_entities', [])
        concepts = kg_data.get('ontology_concepts', [])

        if concepts:
            avg_instances = len(entities) / len(concepts) if concepts else 0
            # 이상적: 5-20개의 인스턴스
            ci_score = min(avg_instances / 10, 1.0)
            scores['class_instantiation'] = KGQualityScore(
                metric_name="Class Instantiation",
                score=ci_score,
                interpretation=f"평균 {avg_instances:.1f} 인스턴스/클래스"
            )

        # 2. Property Completeness: 필수 속성 완성도
        total_props = 0
        filled_props = 0
        required_props = {'entity_type', 'confidence', 'source_tables'}

        for entity in entities:
            for prop in required_props:
                total_props += 1
                if entity.get(prop):
                    filled_props += 1

        pc_score = filled_props / total_props if total_props > 0 else 0
        scores['property_completeness'] = KGQualityScore(
            metric_name="Property Completeness",
            score=pc_score,
            interpretation=f"{filled_props}/{total_props} 속성 완성"
        )

        # 3. Relationship Balance: in/out degree 균형
        triples = (kg_data.get('knowledge_graph_triples', []) +
                  kg_data.get('semantic_base_triples', []))

        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for triple in triples:
            subj = triple.get('subject', '')
            obj = triple.get('object', '')
            out_degree[subj] += 1
            in_degree[obj] += 1

        if out_degree and in_degree:
            avg_in = sum(in_degree.values()) / len(in_degree)
            avg_out = sum(out_degree.values()) / len(out_degree)
            balance = min(avg_in, avg_out) / max(avg_in, avg_out) if max(avg_in, avg_out) > 0 else 0
        else:
            balance = 0.5

        scores['relationship_balance'] = KGQualityScore(
            metric_name="Relationship Balance",
            score=balance,
            interpretation=f"In/Out 비율: {balance:.2f}"
        )

        # 4. Semantic Consistency: 동일 predicate 일관성
        predicates = defaultdict(list)
        for triple in triples:
            pred = triple.get('predicate', '')
            predicates[pred].append(triple)

        consistency_scores = []
        for pred, pred_triples in predicates.items():
            if len(pred_triples) > 1:
                # 동일 predicate의 subject/object 패턴 일관성
                subjects = [t.get('subject', '') for t in pred_triples]
                # 간단한 일관성: 유사한 prefix/suffix 비율
                if subjects:
                    prefixes = [s.split(':')[0] if ':' in s else s[:3] for s in subjects]
                    consistency = len(set(prefixes)) / len(prefixes)
                    consistency_scores.append(1 - consistency + 0.5)  # 다양성 ↓ = 일관성 ↑

        sc_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.7
        scores['semantic_consistency'] = KGQualityScore(
            metric_name="Semantic Consistency",
            score=min(sc_score, 1.0),
            interpretation=f"{len(predicates)} predicate 유형 분석"
        )

        # 5. Hierarchy Depth: rdfs:subClassOf 체인 깊이
        subclass_triples = [t for t in triples if 'subClassOf' in t.get('predicate', '')]
        depth = len(subclass_triples)
        ideal_depth = 3  # 보통 3단계 정도가 적절
        hd_score = 1 - abs(depth - ideal_depth) / max(depth, ideal_depth) if depth > 0 else 0.5

        scores['hierarchy_depth'] = KGQualityScore(
            metric_name="Hierarchy Depth",
            score=max(hd_score, 0),
            interpretation=f"계층 깊이: {depth}"
        )

        # 6. Link Validity: 참조 무결성
        all_subjects = {t.get('subject') for t in triples}
        all_objects = {t.get('object') for t in triples}
        all_nodes = all_subjects | all_objects

        # 리터럴이 아닌 객체 중 subject로도 등장하는 비율
        non_literal_objects = {o for o in all_objects if ':' in str(o) and not o.startswith('"')}
        valid_refs = len(non_literal_objects & all_subjects)
        total_refs = len(non_literal_objects)

        lv_score = valid_refs / total_refs if total_refs > 0 else 0.8
        scores['link_validity'] = KGQualityScore(
            metric_name="Link Validity",
            score=lv_score,
            interpretation=f"{valid_refs}/{total_refs} 유효 참조"
        )

        return scores

    def get_overall_score(self, scores: Dict[str, KGQualityScore]) -> float:
        """가중 평균 품질 점수"""
        total = 0.0
        weight_sum = 0.0

        for name, score in scores.items():
            w = self.weights.get(name, 0.1)
            total += w * score.score
            weight_sum += w

        return total / weight_sum if weight_sum > 0 else 0.5
