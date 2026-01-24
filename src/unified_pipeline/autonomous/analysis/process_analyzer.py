"""
Process Analyzer Module

비즈니스 프로세스 분석 모듈:
- 프로세스 컨텍스트 식별
- 워크플로우 매핑
- 통합 복잡도 평가
- 자동화 기회 탐지
- SLA 추적

Legacy ProcessArchitectAgent의 핵심 기능을 분석 유틸리티로 재구현
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ProcessArea(str, Enum):
    """비즈니스 프로세스 영역"""
    SALES = "sales"
    ORDER_MANAGEMENT = "order_management"
    MANUFACTURING = "manufacturing"
    SUPPLY_CHAIN = "supply_chain"
    MARKETING = "marketing"
    HUMAN_RESOURCES = "human_resources"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    LOGISTICS = "logistics"
    UNKNOWN = "unknown"


class IntegrationComplexity(str, Enum):
    """통합 복잡도 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProcessContext:
    """프로세스 컨텍스트"""
    process_area: ProcessArea
    likely_processes: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    automation_candidates: List[str] = field(default_factory=list)
    sla_considerations: List[str] = field(default_factory=list)
    cross_functional_impacts: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_area": self.process_area.value,
            "likely_processes": self.likely_processes,
            "integration_points": self.integration_points,
            "automation_candidates": self.automation_candidates,
            "sla_considerations": self.sla_considerations,
            "cross_functional_impacts": self.cross_functional_impacts,
            "confidence": self.confidence,
        }


@dataclass
class WorkflowMapping:
    """워크플로우 매핑 결과"""
    entity_name: str
    lifecycle_stages: List[str]
    state_transitions: List[Dict[str, str]]
    trigger_events: List[str]
    automation_opportunities: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "lifecycle_stages": self.lifecycle_stages,
            "state_transitions": self.state_transitions,
            "trigger_events": self.trigger_events,
            "automation_opportunities": self.automation_opportunities,
        }


@dataclass
class IntegrationAssessment:
    """통합 평가 결과"""
    complexity: IntegrationComplexity
    source_count: int
    integration_points: List[str]
    api_requirements: List[str]
    event_patterns: List[str]
    estimated_effort: str  # "days", "weeks", "months"
    risks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "source_count": self.source_count,
            "integration_points": self.integration_points,
            "api_requirements": self.api_requirements,
            "event_patterns": self.event_patterns,
            "estimated_effort": self.estimated_effort,
            "risks": self.risks,
        }


class ProcessAnalyzer:
    """
    비즈니스 프로세스 분석기

    기능:
    1. 프로세스 컨텍스트 식별 - 엔티티가 속하는 비즈니스 프로세스 영역 식별
    2. 워크플로우 매핑 - 엔티티 라이프사이클 및 상태 전이 분석
    3. 통합 복잡도 평가 - 시스템 통합 난이도 및 노력 추정
    4. 자동화 기회 탐지 - 자동화 가능한 프로세스 식별
    """

    # 도메인별 프로세스 매핑
    PROCESS_PATTERNS = {
        ProcessArea.SALES: {
            "keywords": ["customer", "account", "opportunity", "lead", "sales", "deal", "contract", "quote"],
            "processes": ["lead_to_cash", "opportunity_management", "account_management", "quote_to_order"],
            "automation": ["lead_scoring", "deal_routing", "follow_up_scheduling", "contract_generation"],
            "sla": ["response_time", "conversion_rate", "deal_closure_time"],
        },
        ProcessArea.ORDER_MANAGEMENT: {
            "keywords": ["order", "fulfillment", "delivery", "shipment", "invoice", "purchase", "receipt"],
            "processes": ["order_to_cash", "fulfillment", "invoicing", "returns_management"],
            "automation": ["order_validation", "inventory_check", "shipping_notification", "invoice_generation"],
            "sla": ["order_processing_time", "delivery_accuracy", "invoice_accuracy"],
        },
        ProcessArea.MANUFACTURING: {
            "keywords": ["production", "manufacturing", "quality", "inspection", "yield", "assembly", "work_order"],
            "processes": ["production_planning", "quality_control", "maintenance", "capacity_planning"],
            "automation": ["defect_detection", "maintenance_scheduling", "yield_optimization", "batch_tracking"],
            "sla": ["production_cycle_time", "quality_rate", "equipment_uptime"],
        },
        ProcessArea.SUPPLY_CHAIN: {
            "keywords": ["inventory", "supply", "warehouse", "stock", "procurement", "vendor", "supplier"],
            "processes": ["procure_to_pay", "inventory_management", "demand_planning", "supplier_management"],
            "automation": ["reorder_triggers", "demand_forecasting", "supplier_evaluation", "stock_alerts"],
            "sla": ["inventory_accuracy", "supplier_lead_time", "stock_availability"],
        },
        ProcessArea.MARKETING: {
            "keywords": ["campaign", "marketing", "engagement", "channel", "content", "audience", "segment"],
            "processes": ["campaign_management", "lead_generation", "customer_engagement", "content_marketing"],
            "automation": ["campaign_triggering", "audience_segmentation", "content_personalization", "a_b_testing"],
            "sla": ["campaign_roi", "engagement_rate", "conversion_rate"],
        },
        ProcessArea.HUMAN_RESOURCES: {
            "keywords": ["employee", "hr", "payroll", "benefit", "recruitment", "performance", "leave"],
            "processes": ["hire_to_retire", "payroll", "performance_management", "talent_acquisition"],
            "automation": ["onboarding_workflow", "leave_approval", "performance_review", "expense_reimbursement"],
            "sla": ["time_to_hire", "payroll_accuracy", "employee_satisfaction"],
        },
        ProcessArea.FINANCE: {
            "keywords": ["finance", "accounting", "budget", "expense", "revenue", "payment", "transaction"],
            "processes": ["record_to_report", "budget_planning", "expense_management", "revenue_recognition"],
            "automation": ["expense_approval", "budget_alerts", "reconciliation", "payment_processing"],
            "sla": ["close_cycle_time", "payment_accuracy", "compliance_rate"],
        },
        ProcessArea.HEALTHCARE: {
            "keywords": ["patient", "diagnosis", "treatment", "medication", "encounter", "claim", "provider"],
            "processes": ["patient_care", "claims_processing", "medication_management", "care_coordination"],
            "automation": ["appointment_scheduling", "claims_validation", "medication_alerts", "care_reminders"],
            "sla": ["wait_time", "claim_processing_time", "care_quality_metrics"],
        },
        ProcessArea.LOGISTICS: {
            "keywords": ["shipping", "transport", "route", "fleet", "delivery", "carrier", "tracking"],
            "processes": ["route_optimization", "fleet_management", "last_mile_delivery", "carrier_management"],
            "automation": ["route_planning", "delivery_tracking", "capacity_planning", "exception_handling"],
            "sla": ["on_time_delivery", "cost_per_delivery", "fleet_utilization"],
        },
    }

    # 라이프사이클 패턴
    LIFECYCLE_PATTERNS = {
        "status": ["created", "pending", "active", "completed", "cancelled", "archived"],
        "order": ["draft", "submitted", "approved", "processing", "shipped", "delivered", "closed"],
        "case": ["new", "open", "in_progress", "pending_customer", "resolved", "closed"],
        "approval": ["draft", "submitted", "under_review", "approved", "rejected"],
        "payment": ["pending", "authorized", "captured", "refunded", "failed"],
    }

    def __init__(self, domain_context: Optional[str] = None):
        """
        Args:
            domain_context: 도메인 컨텍스트 힌트 (예: "healthcare", "manufacturing")
        """
        self.domain_context = domain_context

    def identify_process_context(
        self,
        entity_name: str,
        description: str = "",
        properties: List[str] = None,
        source_tables: List[str] = None,
    ) -> ProcessContext:
        """
        비즈니스 프로세스 컨텍스트 식별

        Args:
            entity_name: 엔티티 이름
            description: 엔티티 설명
            properties: 속성 목록
            source_tables: 소스 테이블 목록

        Returns:
            ProcessContext
        """
        # 검색 텍스트 구성
        check_text = f"{entity_name} {description} {' '.join(properties or [])} {' '.join(source_tables or [])}".lower()

        best_match = ProcessArea.UNKNOWN
        best_score = 0
        best_pattern = None

        # 각 프로세스 영역과 매칭
        for area, pattern in self.PROCESS_PATTERNS.items():
            score = sum(1 for kw in pattern["keywords"] if kw in check_text)
            if score > best_score:
                best_score = score
                best_match = area
                best_pattern = pattern

        # 도메인 컨텍스트 힌트 적용
        if self.domain_context:
            for area in ProcessArea:
                if self.domain_context.lower() in area.value:
                    if best_score == 0:  # 매칭이 없을 때만 힌트 사용
                        best_match = area
                        best_pattern = self.PROCESS_PATTERNS.get(area, {})
                    break

        # 결과 생성
        confidence = min(1.0, best_score / 3) if best_score > 0 else 0.0

        return ProcessContext(
            process_area=best_match,
            likely_processes=best_pattern.get("processes", []) if best_pattern else [],
            automation_candidates=best_pattern.get("automation", []) if best_pattern else [],
            sla_considerations=best_pattern.get("sla", []) if best_pattern else [],
            confidence=confidence,
        )

    def map_workflow(
        self,
        entity_name: str,
        status_column: Optional[str] = None,
        sample_values: List[str] = None,
    ) -> WorkflowMapping:
        """
        워크플로우 매핑

        Args:
            entity_name: 엔티티 이름
            status_column: 상태 컬럼명
            sample_values: 샘플 상태 값들

        Returns:
            WorkflowMapping
        """
        # 라이프사이클 패턴 매칭
        lifecycle_stages = []
        pattern_type = "status"  # 기본

        if sample_values:
            sample_lower = [v.lower() for v in sample_values if isinstance(v, str)]

            for pattern_name, stages in self.LIFECYCLE_PATTERNS.items():
                match_count = sum(1 for s in sample_lower if any(st in s for st in stages))
                if match_count >= 2:
                    lifecycle_stages = stages
                    pattern_type = pattern_name
                    break

        if not lifecycle_stages:
            lifecycle_stages = self.LIFECYCLE_PATTERNS["status"]

        # 상태 전이 생성
        state_transitions = []
        for i in range(len(lifecycle_stages) - 1):
            state_transitions.append({
                "from": lifecycle_stages[i],
                "to": lifecycle_stages[i + 1],
                "trigger": f"{lifecycle_stages[i]}_to_{lifecycle_stages[i + 1]}",
            })

        # 트리거 이벤트
        trigger_events = [
            f"{entity_name.lower()}_created",
            f"{entity_name.lower()}_updated",
            f"{entity_name.lower()}_status_changed",
            f"{entity_name.lower()}_completed",
        ]

        # 자동화 기회
        automation_opportunities = [
            f"Auto-transition {entity_name} based on time rules",
            f"Notify stakeholders on {entity_name} status changes",
            f"Validate {entity_name} data before status change",
            f"Archive {entity_name} after completion",
        ]

        return WorkflowMapping(
            entity_name=entity_name,
            lifecycle_stages=lifecycle_stages,
            state_transitions=state_transitions,
            trigger_events=trigger_events,
            automation_opportunities=automation_opportunities,
        )

    def assess_integration_complexity(
        self,
        source_tables: List[str] = None,
        related_entities: List[str] = None,
        has_external_sources: bool = False,
    ) -> IntegrationAssessment:
        """
        통합 복잡도 평가

        Args:
            source_tables: 소스 테이블 목록
            related_entities: 관련 엔티티 목록
            has_external_sources: 외부 소스 여부

        Returns:
            IntegrationAssessment
        """
        source_count = len(source_tables or [])
        relation_count = len(related_entities or [])

        # 복잡도 계산
        complexity_score = source_count + (relation_count * 0.5) + (3 if has_external_sources else 0)

        if complexity_score <= 2:
            complexity = IntegrationComplexity.LOW
            effort = "days"
        elif complexity_score <= 5:
            complexity = IntegrationComplexity.MEDIUM
            effort = "weeks"
        elif complexity_score <= 10:
            complexity = IntegrationComplexity.HIGH
            effort = "weeks"
        else:
            complexity = IntegrationComplexity.CRITICAL
            effort = "months"

        # 통합 포인트
        integration_points = []
        if source_tables:
            integration_points.extend([f"ETL from {t}" for t in source_tables[:3]])
        if related_entities:
            integration_points.extend([f"Link to {e}" for e in related_entities[:3]])
        if has_external_sources:
            integration_points.append("External API integration")

        # API 요구사항
        api_requirements = []
        if complexity_score > 2:
            api_requirements = ["REST API for CRUD operations"]
        if complexity_score > 5:
            api_requirements.extend(["Batch import/export API", "Event webhook endpoints"])
        if has_external_sources:
            api_requirements.append("External system connectors")

        # 이벤트 패턴
        event_patterns = ["state_change_events"]
        if complexity_score > 3:
            event_patterns.extend(["data_sync_events", "validation_events"])
        if has_external_sources:
            event_patterns.append("external_sync_events")

        # 리스크
        risks = []
        if source_count > 3:
            risks.append("Data inconsistency across multiple sources")
        if has_external_sources:
            risks.append("External system availability dependency")
        if complexity_score > 8:
            risks.append("Complex rollback scenarios")
        if not risks:
            risks.append("Standard integration risks")

        return IntegrationAssessment(
            complexity=complexity,
            source_count=source_count,
            integration_points=integration_points,
            api_requirements=api_requirements,
            event_patterns=event_patterns,
            estimated_effort=effort,
            risks=risks,
        )

    def analyze_automation_potential(
        self,
        entity_name: str,
        process_context: ProcessContext,
        has_status_field: bool = True,
        has_date_fields: bool = True,
    ) -> Dict[str, Any]:
        """
        자동화 잠재력 분석

        Args:
            entity_name: 엔티티 이름
            process_context: 프로세스 컨텍스트
            has_status_field: 상태 필드 존재 여부
            has_date_fields: 날짜 필드 존재 여부

        Returns:
            자동화 분석 결과
        """
        automation_score = 0
        recommendations = []

        # 상태 기반 자동화
        if has_status_field:
            automation_score += 30
            recommendations.extend([
                f"Implement state machine for {entity_name} lifecycle",
                f"Add automatic notifications on {entity_name} status changes",
                f"Create validation rules for {entity_name} state transitions",
            ])

        # 시간 기반 자동화
        if has_date_fields:
            automation_score += 25
            recommendations.extend([
                f"Set up SLA monitoring for {entity_name}",
                f"Implement time-based escalation rules",
                f"Add aging alerts for stale {entity_name} records",
            ])

        # 프로세스 컨텍스트 기반 자동화
        if process_context.automation_candidates:
            automation_score += 30
            for candidate in process_context.automation_candidates[:3]:
                recommendations.append(f"Consider implementing {candidate}")

        # 통합 자동화
        if process_context.integration_points:
            automation_score += 15
            recommendations.append("Implement event-driven sync with related systems")

        return {
            "automation_score": min(100, automation_score),
            "automation_level": (
                "high" if automation_score >= 70 else
                "medium" if automation_score >= 40 else
                "low"
            ),
            "recommendations": recommendations,
            "quick_wins": recommendations[:2] if recommendations else [],
        }

    def generate_sla_recommendations(
        self,
        process_context: ProcessContext,
        entity_name: str,
    ) -> List[Dict[str, Any]]:
        """
        SLA 권장사항 생성

        Args:
            process_context: 프로세스 컨텍스트
            entity_name: 엔티티 이름

        Returns:
            SLA 권장사항 목록
        """
        slas = []

        for sla_type in process_context.sla_considerations:
            sla = {
                "name": f"{entity_name}_{sla_type}",
                "type": sla_type,
                "description": f"Track {sla_type.replace('_', ' ')} for {entity_name}",
                "suggested_threshold": self._get_suggested_threshold(sla_type),
                "measurement_frequency": "daily",
            }
            slas.append(sla)

        # 기본 SLA 추가
        if not slas:
            slas = [
                {
                    "name": f"{entity_name}_processing_time",
                    "type": "processing_time",
                    "description": f"Time to process {entity_name}",
                    "suggested_threshold": "24 hours",
                    "measurement_frequency": "hourly",
                },
                {
                    "name": f"{entity_name}_completion_rate",
                    "type": "completion_rate",
                    "description": f"Rate of completed {entity_name} records",
                    "suggested_threshold": "95%",
                    "measurement_frequency": "daily",
                },
            ]

        return slas

    def _get_suggested_threshold(self, sla_type: str) -> str:
        """SLA 유형별 권장 임계값"""
        thresholds = {
            "response_time": "4 hours",
            "processing_time": "24 hours",
            "delivery_accuracy": "98%",
            "quality_rate": "99%",
            "on_time_delivery": "95%",
            "claim_processing_time": "48 hours",
            "inventory_accuracy": "99%",
        }
        return thresholds.get(sla_type, "TBD")
