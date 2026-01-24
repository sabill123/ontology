"""
Virtual Entity Generator - 가상 시나리오 엔티티 생성기

Phase 2 Refinement를 위한 가상 엔티티 생성 모듈입니다.
실제 데이터에는 없지만 비즈니스 분석에 필요한 파생 엔티티를 생성합니다.

주요 기능:
- ChurnRisk: 이탈 위험도 엔티티
- CustomerLifetimeValue: 고객 생애 가치 엔티티
- PredictedDemand: 예측 수요 엔티티
- InventoryRisk: 재고 위험도 엔티티
- SupplyChainBottleneck: 공급망 병목 엔티티

References:
- Palantir Foundry Ontology 패턴
- Data Mesh의 Data Product 개념
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VirtualEntityType(Enum):
    """가상 엔티티 타입"""
    # 고객 관련
    CHURN_RISK = "churn_risk"                    # 이탈 위험도
    CUSTOMER_LIFETIME_VALUE = "clv"              # 고객 생애 가치
    CUSTOMER_SEGMENT = "customer_segment"        # 고객 세그먼트

    # 예측 관련
    PREDICTED_DEMAND = "predicted_demand"        # 예측 수요
    FORECASTED_REVENUE = "forecasted_revenue"    # 예측 매출

    # 위험/기회 관련
    INVENTORY_RISK = "inventory_risk"            # 재고 위험도
    SUPPLY_CHAIN_BOTTLENECK = "bottleneck"       # 공급망 병목
    OPPORTUNITY_SCORE = "opportunity"            # 기회 점수

    # 품질 관련
    DATA_QUALITY_SCORE = "dq_score"              # 데이터 품질 점수
    ENTITY_COMPLETENESS = "completeness"         # 엔티티 완전성


@dataclass
class VirtualEntityDefinition:
    """가상 엔티티 정의"""
    entity_type: VirtualEntityType
    name: str
    description: str
    source_entities: List[str]                   # 기반이 되는 실제 엔티티들
    required_attributes: List[Dict[str, Any]]    # 필요한 속성들
    computation_logic: str                       # 계산 로직 설명
    business_rules: List[str]                    # 비즈니스 규칙
    kpi_impact: Dict[str, float]                 # KPI 영향도
    confidence: float = 0.0


@dataclass
class GeneratedVirtualEntity:
    """생성된 가상 엔티티"""
    entity_id: str
    entity_type: VirtualEntityType
    name: str
    description: str
    definition: Dict[str, Any]
    source_tables: List[str]
    source_entities: List[str]
    attributes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    computation_formula: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VirtualEntityGenerator:
    """
    가상 엔티티 생성기

    기존 엔티티와 테이블 정보를 기반으로 비즈니스에 필요한
    가상/파생 엔티티를 자동으로 제안합니다.
    """

    # 도메인별 가상 엔티티 템플릿
    DOMAIN_TEMPLATES = {
        "healthcare": [
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.CHURN_RISK,
                name="PatientChurnRisk",
                description="환자 이탈 위험도를 예측하는 가상 엔티티",
                source_entities=["Patient", "Encounter", "Claim"],
                required_attributes=[
                    {"name": "risk_score", "type": "float", "description": "이탈 위험 점수 (0-1)"},
                    {"name": "risk_factors", "type": "list", "description": "위험 요인 목록"},
                    {"name": "last_visit_days", "type": "int", "description": "마지막 방문 이후 일수"},
                    {"name": "predicted_churn_date", "type": "date", "description": "예상 이탈 시점"},
                ],
                computation_logic="최근 방문 빈도, 클레임 패턴, 진료 연속성을 기반으로 이탈 확률 계산",
                business_rules=[
                    "마지막 방문 후 6개월 경과 시 고위험",
                    "예정된 추적 관찰 누락 시 위험도 증가",
                    "만성 질환자의 정기 방문 이탈 시 우선 관리",
                ],
                kpi_impact={"patient_retention": 0.3, "revenue": 0.2},
            ),
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.CUSTOMER_LIFETIME_VALUE,
                name="PatientLifetimeValue",
                description="환자의 예상 생애 가치 (의료비 기준)",
                source_entities=["Patient", "Claim", "Insurance"],
                required_attributes=[
                    {"name": "lifetime_value", "type": "float", "description": "예상 생애 가치 (USD)"},
                    {"name": "annual_value", "type": "float", "description": "연간 예상 가치"},
                    {"name": "value_segment", "type": "string", "description": "가치 세그먼트 (High/Medium/Low)"},
                ],
                computation_logic="과거 청구 이력과 인구통계학적 특성을 기반으로 향후 의료 지출 예측",
                business_rules=[
                    "만성 질환 환자는 높은 LTV 예상",
                    "보험 유형에 따른 상환율 반영",
                ],
                kpi_impact={"revenue": 0.4, "strategic_planning": 0.3},
            ),
        ],
        "supply_chain": [
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.INVENTORY_RISK,
                name="InventoryRisk",
                description="재고 부족/과잉 위험도를 평가하는 엔티티",
                source_entities=["Product", "Warehouse", "Order", "Supplier"],
                required_attributes=[
                    {"name": "stockout_risk", "type": "float", "description": "품절 위험도 (0-1)"},
                    {"name": "overstock_risk", "type": "float", "description": "과잉 재고 위험도 (0-1)"},
                    {"name": "optimal_stock_level", "type": "int", "description": "적정 재고 수준"},
                    {"name": "reorder_point", "type": "int", "description": "재주문 시점"},
                    {"name": "safety_stock", "type": "int", "description": "안전 재고"},
                ],
                computation_logic="수요 예측, 리드타임, 재고 회전율을 기반으로 위험도 계산",
                business_rules=[
                    "리드타임 대비 현재 재고가 부족하면 고위험",
                    "계절성 반영하여 적정 재고 산정",
                    "ABC 분석 기반 차등 관리",
                ],
                kpi_impact={"fulfillment_rate": 0.35, "working_capital": 0.25},
            ),
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.SUPPLY_CHAIN_BOTTLENECK,
                name="SupplyChainBottleneck",
                description="공급망 병목 지점을 식별하는 엔티티",
                source_entities=["Plant", "Warehouse", "Supplier", "Route"],
                required_attributes=[
                    {"name": "bottleneck_score", "type": "float", "description": "병목 심각도 (0-1)"},
                    {"name": "affected_products", "type": "list", "description": "영향받는 제품들"},
                    {"name": "delay_days", "type": "int", "description": "예상 지연 일수"},
                    {"name": "alternative_routes", "type": "list", "description": "대안 경로"},
                ],
                computation_logic="처리량 대비 수요, 공급자 리드타임, 재고 수준을 종합 분석",
                business_rules=[
                    "단일 공급자 의존도가 높으면 병목 위험",
                    "물류 허브 용량 초과 시 경고",
                    "리드타임 변동성이 큰 경로는 리스크",
                ],
                kpi_impact={"delivery_performance": 0.4, "cost_efficiency": 0.2},
            ),
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.PREDICTED_DEMAND,
                name="PredictedDemand",
                description="제품별/지역별 예측 수요 엔티티",
                source_entities=["Product", "Order", "Customer", "Region"],
                required_attributes=[
                    {"name": "predicted_quantity", "type": "int", "description": "예측 수량"},
                    {"name": "prediction_date", "type": "date", "description": "예측 시점"},
                    {"name": "confidence_interval", "type": "tuple", "description": "신뢰구간"},
                    {"name": "forecast_horizon", "type": "int", "description": "예측 기간 (일)"},
                    {"name": "seasonality_factor", "type": "float", "description": "계절성 지수"},
                ],
                computation_logic="과거 판매 데이터, 계절성, 트렌드를 분석하여 미래 수요 예측",
                business_rules=[
                    "최소 12개월 과거 데이터 필요",
                    "프로모션 기간은 별도 보정",
                    "신제품은 유사 제품 데이터 활용",
                ],
                kpi_impact={"forecast_accuracy": 0.5, "inventory_turns": 0.3},
            ),
        ],
        "ecommerce": [
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.CHURN_RISK,
                name="CustomerChurnRisk",
                description="고객 이탈 위험도 예측 엔티티",
                source_entities=["Customer", "Order", "Product", "Review"],
                required_attributes=[
                    {"name": "churn_probability", "type": "float", "description": "이탈 확률"},
                    {"name": "days_since_last_order", "type": "int", "description": "마지막 주문 이후 일수"},
                    {"name": "engagement_score", "type": "float", "description": "참여도 점수"},
                    {"name": "retention_actions", "type": "list", "description": "권장 유지 활동"},
                ],
                computation_logic="RFM 분석, 구매 주기, 사이트 활동 데이터를 종합",
                business_rules=[
                    "평균 구매 주기의 2배 경과 시 고위험",
                    "최근 부정적 리뷰 작성자는 위험",
                    "장바구니 이탈 빈도가 높으면 주의",
                ],
                kpi_impact={"customer_retention": 0.4, "revenue": 0.3},
            ),
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.CUSTOMER_LIFETIME_VALUE,
                name="CustomerLifetimeValue",
                description="고객 생애 가치 예측 엔티티",
                source_entities=["Customer", "Order", "Payment"],
                required_attributes=[
                    {"name": "predicted_clv", "type": "float", "description": "예측 CLV"},
                    {"name": "clv_segment", "type": "string", "description": "CLV 세그먼트"},
                    {"name": "acquisition_cost", "type": "float", "description": "획득 비용"},
                    {"name": "margin_ratio", "type": "float", "description": "마진율"},
                ],
                computation_logic="과거 구매 이력, 평균 주문 가치, 예상 유지 기간으로 CLV 계산",
                business_rules=[
                    "BG/NBD 모델 또는 간단한 휴리스틱 사용",
                    "할인율 적용하여 순현재가치 계산",
                ],
                kpi_impact={"customer_value": 0.5, "marketing_roi": 0.3},
            ),
        ],
        "general": [
            VirtualEntityDefinition(
                entity_type=VirtualEntityType.DATA_QUALITY_SCORE,
                name="DataQualityScore",
                description="데이터 품질 점수 엔티티",
                source_entities=["*"],
                required_attributes=[
                    {"name": "completeness", "type": "float", "description": "완전성 점수"},
                    {"name": "accuracy", "type": "float", "description": "정확성 점수"},
                    {"name": "consistency", "type": "float", "description": "일관성 점수"},
                    {"name": "timeliness", "type": "float", "description": "적시성 점수"},
                    {"name": "overall_score", "type": "float", "description": "종합 점수"},
                ],
                computation_logic="각 테이블/컬럼의 품질 메트릭을 수집하여 종합 점수 산출",
                business_rules=[
                    "NULL 비율로 완전성 측정",
                    "참조 무결성으로 정확성 측정",
                    "업데이트 주기로 적시성 측정",
                ],
                kpi_impact={"data_reliability": 0.6, "analytics_trust": 0.4},
            ),
        ],
    }

    def __init__(self, domain: str = "general"):
        """
        Args:
            domain: 도메인 (healthcare, supply_chain, ecommerce, general)
        """
        self.domain = domain.lower()
        self.templates = self._get_templates_for_domain()

    def _get_templates_for_domain(self) -> List[VirtualEntityDefinition]:
        """도메인에 맞는 템플릿 가져오기"""
        templates = list(self.DOMAIN_TEMPLATES.get("general", []))
        domain_specific = self.DOMAIN_TEMPLATES.get(self.domain, [])
        templates.extend(domain_specific)
        return templates

    def generate_virtual_entities(
        self,
        existing_entities: List[Dict[str, Any]],
        tables: Dict[str, Any],
        business_context: Optional[Dict[str, Any]] = None,
    ) -> List[GeneratedVirtualEntity]:
        """
        가상 엔티티 생성

        Args:
            existing_entities: 기존 엔티티 목록
            tables: 테이블 정보
            business_context: 비즈니스 컨텍스트 (선택)

        Returns:
            생성된 가상 엔티티 목록
        """
        generated = []
        existing_names = {e.get("name", "").lower() for e in existing_entities}
        existing_types = {e.get("entity_type", "").lower() for e in existing_entities}

        for template in self.templates:
            # 이미 존재하는 엔티티인지 확인
            if template.name.lower() in existing_names:
                logger.debug(f"Skipping {template.name}: already exists")
                continue

            # 필요한 소스 엔티티가 있는지 확인
            available_sources = self._find_matching_entities(
                template.source_entities,
                existing_entities,
                tables,
            )

            if not available_sources and template.source_entities != ["*"]:
                logger.debug(f"Skipping {template.name}: required sources not found")
                continue

            # 가상 엔티티 생성
            virtual_entity = self._create_virtual_entity(
                template,
                available_sources,
                tables,
                business_context,
            )

            if virtual_entity:
                generated.append(virtual_entity)
                logger.info(f"Generated virtual entity: {virtual_entity.name}")

        return generated

    def _find_matching_entities(
        self,
        required_entities: List[str],
        existing_entities: List[Dict[str, Any]],
        tables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """필요한 엔티티가 존재하는지 확인"""
        if required_entities == ["*"]:
            return {"all": list(tables.keys())}

        matches = {}
        for req in required_entities:
            req_lower = req.lower()

            # 기존 엔티티에서 찾기
            for entity in existing_entities:
                entity_name = entity.get("name", "").lower()
                entity_type = entity.get("entity_type", "").lower()

                if req_lower in entity_name or req_lower in entity_type:
                    matches[req] = entity
                    break

            # 테이블 이름에서 찾기
            if req not in matches:
                for table_name in tables.keys():
                    if req_lower in table_name.lower():
                        matches[req] = {"name": table_name, "source": "table"}
                        break

        return matches

    def _create_virtual_entity(
        self,
        template: VirtualEntityDefinition,
        available_sources: Dict[str, Any],
        tables: Dict[str, Any],
        business_context: Optional[Dict[str, Any]],
    ) -> Optional[GeneratedVirtualEntity]:
        """템플릿으로부터 가상 엔티티 생성"""
        # 소스 테이블 수집
        source_tables = []
        source_entities = []

        for source_name, source_info in available_sources.items():
            if isinstance(source_info, dict):
                if source_info.get("source") == "table":
                    source_tables.append(source_info.get("name", ""))
                else:
                    source_entities.append(source_info.get("name", source_name))
                    src_tables = source_info.get("source_tables", [])
                    source_tables.extend(src_tables)

        # 신뢰도 계산 (소스 엔티티 매칭률 기반)
        required_count = len(template.source_entities)
        matched_count = len(available_sources)
        confidence = matched_count / required_count if required_count > 0 else 0.5

        # 속성에 소스 정보 추가
        attributes = []
        for attr in template.required_attributes:
            attr_copy = dict(attr)
            attr_copy["derived_from"] = list(available_sources.keys())[:2]
            attributes.append(attr_copy)

        # 관계 정의
        relationships = []
        for source_name in available_sources.keys():
            relationships.append({
                "target_entity": source_name,
                "relationship_type": "derived_from",
                "cardinality": "many_to_one",
            })

        # 계산 수식 생성
        computation_formula = self._generate_computation_formula(
            template,
            available_sources,
        )

        return GeneratedVirtualEntity(
            entity_id=f"virtual_{template.entity_type.value}_{id(template) % 10000:04d}",
            entity_type=template.entity_type,
            name=template.name,
            description=template.description,
            definition={
                "computation_logic": template.computation_logic,
                "business_rules": template.business_rules,
                "kpi_impact": template.kpi_impact,
            },
            source_tables=list(set(source_tables)),
            source_entities=source_entities,
            attributes=attributes,
            relationships=relationships,
            computation_formula=computation_formula,
            confidence=confidence,
            metadata={
                "template_type": template.entity_type.value,
                "domain": self.domain,
                "business_context": business_context,
            },
        )

    def _generate_computation_formula(
        self,
        template: VirtualEntityDefinition,
        available_sources: Dict[str, Any],
    ) -> str:
        """계산 수식 생성"""
        # 간단한 템플릿 기반 수식 생성
        formulas = {
            VirtualEntityType.CHURN_RISK: (
                "risk_score = sigmoid(w1 * days_since_last + w2 * engagement + w3 * complaints)"
            ),
            VirtualEntityType.CUSTOMER_LIFETIME_VALUE: (
                "CLV = (avg_order_value * purchase_frequency * margin) / churn_rate"
            ),
            VirtualEntityType.INVENTORY_RISK: (
                "risk = max(0, 1 - (current_stock - safety_stock) / (reorder_point - safety_stock))"
            ),
            VirtualEntityType.PREDICTED_DEMAND: (
                "demand[t+h] = trend + seasonality[t+h] + noise"
            ),
            VirtualEntityType.SUPPLY_CHAIN_BOTTLENECK: (
                "bottleneck_score = throughput_utilization * dependency_factor * variability"
            ),
            VirtualEntityType.DATA_QUALITY_SCORE: (
                "score = 0.3*completeness + 0.3*accuracy + 0.2*consistency + 0.2*timeliness"
            ),
        }

        return formulas.get(
            template.entity_type,
            f"computed_value = f({', '.join(available_sources.keys())})"
        )

    def suggest_virtual_entities(
        self,
        existing_entities: List[Dict[str, Any]],
        business_goals: List[str],
    ) -> List[VirtualEntityDefinition]:
        """
        비즈니스 목표에 맞는 가상 엔티티 제안

        Args:
            existing_entities: 기존 엔티티 목록
            business_goals: 비즈니스 목표 목록

        Returns:
            추천 가상 엔티티 정의 목록
        """
        suggestions = []

        # 목표별 매핑
        goal_mappings = {
            "retention": [VirtualEntityType.CHURN_RISK],
            "revenue": [VirtualEntityType.CUSTOMER_LIFETIME_VALUE],
            "inventory": [VirtualEntityType.INVENTORY_RISK, VirtualEntityType.PREDICTED_DEMAND],
            "supply_chain": [VirtualEntityType.SUPPLY_CHAIN_BOTTLENECK],
            "forecast": [VirtualEntityType.PREDICTED_DEMAND, VirtualEntityType.FORECASTED_REVENUE],
            "quality": [VirtualEntityType.DATA_QUALITY_SCORE],
        }

        relevant_types = set()
        for goal in business_goals:
            goal_lower = goal.lower()
            for keyword, types in goal_mappings.items():
                if keyword in goal_lower:
                    relevant_types.update(types)

        for template in self.templates:
            if template.entity_type in relevant_types:
                suggestions.append(template)

        return suggestions


def generate_virtual_entities_for_context(
    context_tables: Dict[str, Any],
    context_entities: List[Dict[str, Any]],
    domain: str = "general",
) -> List[Dict[str, Any]]:
    """
    SharedContext용 헬퍼 함수

    Args:
        context_tables: context.tables
        context_entities: unified_entities를 dict로 변환한 목록
        domain: 도메인

    Returns:
        생성된 가상 엔티티들 (dict 형태)
    """
    generator = VirtualEntityGenerator(domain=domain)

    virtual_entities = generator.generate_virtual_entities(
        existing_entities=context_entities,
        tables=context_tables,
    )

    # dict로 변환하여 반환
    return [
        {
            "entity_id": ve.entity_id,
            "entity_type": ve.entity_type.value,
            "name": ve.name,
            "description": ve.description,
            "definition": ve.definition,
            "source_tables": ve.source_tables,
            "source_entities": ve.source_entities,
            "attributes": ve.attributes,
            "relationships": ve.relationships,
            "computation_formula": ve.computation_formula,
            "confidence": ve.confidence,
            "is_virtual": True,
            "metadata": ve.metadata,
        }
        for ve in virtual_entities
    ]
