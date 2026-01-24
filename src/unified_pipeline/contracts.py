"""
Phase Contract Validation System

v6.0: Phase 간 데이터 계약 검증
- 각 Phase 출력이 다음 Phase 요구사항을 충족하는지 검증
- 조기 실패를 통한 파이프라인 안정성 확보
- 명시적 계약을 통한 인터페이스 문서화

사용 예:
    validator = PhaseContractValidator()

    # Discovery → Refinement 전환 시
    result = validator.validate_transition("discovery", "refinement", context)
    if not result.is_valid:
        raise PhaseContractViolation(
            f"Contract violated: {result.violations}",
            source_phase="discovery",
            target_phase="refinement",
            missing_fields=result.missing_fields
        )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from datetime import datetime
import logging

from .exceptions import PhaseContractViolation, DataIntegrityError

logger = logging.getLogger(__name__)


class FieldRequirement(str, Enum):
    """필드 요구사항 수준"""
    REQUIRED = "required"           # 필수 - 없으면 실패
    RECOMMENDED = "recommended"     # 권장 - 없으면 경고
    OPTIONAL = "optional"           # 선택 - 없어도 무방


class FieldType(str, Enum):
    """필드 데이터 타입"""
    LIST = "list"
    DICT = "dict"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ANY = "any"


@dataclass
class FieldContract:
    """개별 필드 계약"""
    name: str
    field_type: FieldType
    requirement: FieldRequirement = FieldRequirement.REQUIRED
    min_items: int = 0              # LIST 타입일 때 최소 아이템 수
    min_value: Optional[float] = None   # NUMBER 타입일 때 최소값
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None  # 커스텀 검증 함수


@dataclass
class PhaseContract:
    """Phase 출력 계약"""
    phase_name: str
    output_fields: List[FieldContract] = field(default_factory=list)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    description: str = ""

    def add_field(self, contract: FieldContract) -> "PhaseContract":
        """필드 계약 추가 (빌더 패턴)"""
        self.output_fields.append(contract)
        return self


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    source_phase: str
    target_phase: str

    # 위반 사항
    violations: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    type_mismatches: List[str] = field(default_factory=list)

    # 경고 (계속 진행 가능)
    warnings: List[str] = field(default_factory=list)

    # 메타데이터
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    fields_checked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "source_phase": self.source_phase,
            "target_phase": self.target_phase,
            "violations": self.violations,
            "missing_fields": self.missing_fields,
            "type_mismatches": self.type_mismatches,
            "warnings": self.warnings,
            "validated_at": self.validated_at,
            "fields_checked": self.fields_checked,
        }


class PhaseContractValidator:
    """
    Phase 계약 검증기

    각 Phase 전환 시 데이터 계약을 검증하여
    파이프라인 안정성을 보장합니다.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: True면 RECOMMENDED 필드도 경고 대신 실패 처리
        """
        self.strict_mode = strict_mode
        self._contracts = self._define_contracts()

    def _define_contracts(self) -> Dict[str, PhaseContract]:
        """Phase별 계약 정의"""
        contracts = {}

        # === Discovery Phase 출력 계약 ===
        contracts["discovery"] = PhaseContract(
            phase_name="discovery",
            description="데이터 발견 및 관계 탐지 단계",
            quality_thresholds={
                "min_tables_analyzed": 1,
                "min_confidence": 0.3,
            }
        ).add_field(FieldContract(
            name="tables",
            field_type=FieldType.DICT,
            requirement=FieldRequirement.REQUIRED,
            min_items=1,
            description="분석된 테이블 정보"
        )).add_field(FieldContract(
            name="homeomorphisms",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.REQUIRED,
            min_items=0,  # 관계가 없을 수도 있음
            description="탐지된 위상동형 관계"
        )).add_field(FieldContract(
            name="unified_entities",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.RECOMMENDED,
            description="통합된 엔티티 목록"
        )).add_field(FieldContract(
            name="value_overlaps",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.RECOMMENDED,
            description="값 겹침 분석 결과"
        )).add_field(FieldContract(
            name="schema_analysis",
            field_type=FieldType.DICT,
            requirement=FieldRequirement.RECOMMENDED,
            description="스키마 분석 결과"
        )).add_field(FieldContract(
            name="tda_signatures",
            field_type=FieldType.DICT,
            requirement=FieldRequirement.OPTIONAL,
            description="TDA 시그니처 (선택적)"
        ))

        # === Refinement Phase 출력 계약 ===
        contracts["refinement"] = PhaseContract(
            phase_name="refinement",
            description="온톨로지 정제 및 충돌 해결 단계",
            quality_thresholds={
                "min_concepts": 1,
                "min_avg_confidence": 0.4,
            }
        ).add_field(FieldContract(
            name="ontology_concepts",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.REQUIRED,
            min_items=1,
            description="정제된 온톨로지 개념",
            validator=lambda concepts: all(
                hasattr(c, 'concept_id') or isinstance(c, dict) and 'concept_id' in c
                for c in concepts
            ) if concepts else True
        )).add_field(FieldContract(
            name="concept_relationships",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.RECOMMENDED,
            description="개념 간 관계"
        )).add_field(FieldContract(
            name="conflicts_resolved",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.OPTIONAL,
            description="해결된 충돌 목록"
        )).add_field(FieldContract(
            name="knowledge_graph_triples",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.RECOMMENDED,
            description="지식 그래프 트리플"
        ))

        # === Governance Phase 출력 계약 ===
        contracts["governance"] = PhaseContract(
            phase_name="governance",
            description="거버넌스 결정 및 액션 생성 단계",
            quality_thresholds={
                "min_decisions": 0,  # 결정이 없을 수도 있음
                "min_decision_confidence": 0.3,
            }
        ).add_field(FieldContract(
            name="governance_decisions",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.REQUIRED,
            description="거버넌스 결정 목록"
        )).add_field(FieldContract(
            name="action_backlog",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.RECOMMENDED,
            description="액션 백로그"
        )).add_field(FieldContract(
            name="policy_rules",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.OPTIONAL,
            description="정책 규칙"
        )).add_field(FieldContract(
            name="governance_triples",
            field_type=FieldType.LIST,
            requirement=FieldRequirement.OPTIONAL,
            description="거버넌스 트리플"
        ))

        return contracts

    def validate_transition(
        self,
        source_phase: str,
        target_phase: str,
        context: Any,  # SharedContext
    ) -> ValidationResult:
        """
        Phase 전환 시 계약 검증

        Args:
            source_phase: 완료된 Phase
            target_phase: 시작할 Phase
            context: SharedContext 인스턴스

        Returns:
            ValidationResult
        """
        result = ValidationResult(
            is_valid=True,
            source_phase=source_phase,
            target_phase=target_phase,
        )

        # 계약 가져오기
        contract = self._contracts.get(source_phase)
        if not contract:
            result.warnings.append(f"No contract defined for phase: {source_phase}")
            return result

        # 필드 검증
        for field_contract in contract.output_fields:
            result.fields_checked += 1

            # 필드 값 가져오기
            field_value = getattr(context, field_contract.name, None)

            # 존재 여부 확인
            if field_value is None:
                if field_contract.requirement == FieldRequirement.REQUIRED:
                    result.is_valid = False
                    result.violations.append(
                        f"Required field '{field_contract.name}' is missing"
                    )
                    result.missing_fields.append(field_contract.name)
                elif field_contract.requirement == FieldRequirement.RECOMMENDED:
                    if self.strict_mode:
                        result.is_valid = False
                        result.violations.append(
                            f"Recommended field '{field_contract.name}' is missing (strict mode)"
                        )
                        result.missing_fields.append(field_contract.name)
                    else:
                        result.warnings.append(
                            f"Recommended field '{field_contract.name}' is missing"
                        )
                continue

            # 타입 검증
            type_valid = self._validate_type(field_value, field_contract.field_type)
            if not type_valid:
                result.is_valid = False
                result.violations.append(
                    f"Field '{field_contract.name}' has wrong type: "
                    f"expected {field_contract.field_type.value}, got {type(field_value).__name__}"
                )
                result.type_mismatches.append(field_contract.name)
                continue

            # 최소 아이템 수 검증 (LIST 타입)
            if field_contract.field_type == FieldType.LIST:
                if len(field_value) < field_contract.min_items:
                    if field_contract.requirement == FieldRequirement.REQUIRED:
                        result.is_valid = False
                        result.violations.append(
                            f"Field '{field_contract.name}' has {len(field_value)} items, "
                            f"minimum required: {field_contract.min_items}"
                        )
                    else:
                        result.warnings.append(
                            f"Field '{field_contract.name}' has fewer items than recommended"
                        )

            # 커스텀 검증기 실행
            if field_contract.validator:
                try:
                    is_valid = field_contract.validator(field_value)
                    if not is_valid:
                        result.is_valid = False
                        result.violations.append(
                            f"Field '{field_contract.name}' failed custom validation"
                        )
                except Exception as e:
                    result.warnings.append(
                        f"Custom validator error for '{field_contract.name}': {str(e)}"
                    )

        # 품질 임계값 검증
        self._validate_quality_thresholds(context, contract, result)

        # 로깅
        if result.is_valid:
            logger.info(
                f"Contract validation passed: {source_phase} → {target_phase} "
                f"({result.fields_checked} fields checked)"
            )
        else:
            logger.warning(
                f"Contract validation FAILED: {source_phase} → {target_phase} "
                f"Violations: {result.violations}"
            )

        return result

    def _validate_type(self, value: Any, expected_type: FieldType) -> bool:
        """타입 검증"""
        if expected_type == FieldType.ANY:
            return True
        if expected_type == FieldType.LIST:
            return isinstance(value, (list, tuple))
        if expected_type == FieldType.DICT:
            return isinstance(value, dict)
        if expected_type == FieldType.STRING:
            return isinstance(value, str)
        if expected_type == FieldType.NUMBER:
            return isinstance(value, (int, float))
        if expected_type == FieldType.BOOLEAN:
            return isinstance(value, bool)
        return False

    def _validate_quality_thresholds(
        self,
        context: Any,
        contract: PhaseContract,
        result: ValidationResult,
    ) -> None:
        """품질 임계값 검증"""
        thresholds = contract.quality_thresholds

        # min_tables_analyzed
        if "min_tables_analyzed" in thresholds:
            tables = getattr(context, "tables", {})
            if len(tables) < thresholds["min_tables_analyzed"]:
                result.warnings.append(
                    f"Tables analyzed ({len(tables)}) below threshold "
                    f"({thresholds['min_tables_analyzed']})"
                )

        # min_concepts
        if "min_concepts" in thresholds:
            concepts = getattr(context, "ontology_concepts", [])
            if len(concepts) < thresholds["min_concepts"]:
                result.is_valid = False
                result.violations.append(
                    f"Concepts created ({len(concepts)}) below minimum "
                    f"({thresholds['min_concepts']})"
                )

        # min_avg_confidence
        if "min_avg_confidence" in thresholds:
            concepts = getattr(context, "ontology_concepts", [])
            if concepts:
                confidences = [
                    getattr(c, 'confidence', 0) if hasattr(c, 'confidence')
                    else c.get('confidence', 0) if isinstance(c, dict)
                    else 0
                    for c in concepts
                ]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                if avg_confidence < thresholds["min_avg_confidence"]:
                    result.warnings.append(
                        f"Average concept confidence ({avg_confidence:.2f}) below threshold "
                        f"({thresholds['min_avg_confidence']})"
                    )

    def validate_phase_output(
        self,
        phase: str,
        context: Any,
    ) -> ValidationResult:
        """
        단일 Phase 출력 검증 (전환 없이)

        Args:
            phase: 검증할 Phase
            context: SharedContext

        Returns:
            ValidationResult
        """
        return self.validate_transition(phase, "validation_only", context)

    def get_contract(self, phase: str) -> Optional[PhaseContract]:
        """Phase 계약 조회"""
        return self._contracts.get(phase)

    def get_required_fields(self, phase: str) -> List[str]:
        """Phase의 필수 필드 목록"""
        contract = self._contracts.get(phase)
        if not contract:
            return []
        return [
            f.name for f in contract.output_fields
            if f.requirement == FieldRequirement.REQUIRED
        ]

    def describe_contract(self, phase: str) -> str:
        """계약 설명 문자열 생성"""
        contract = self._contracts.get(phase)
        if not contract:
            return f"No contract defined for phase: {phase}"

        lines = [
            f"=== {phase.upper()} Phase Contract ===",
            f"Description: {contract.description}",
            "",
            "Required Fields:",
        ]

        for f in contract.output_fields:
            if f.requirement == FieldRequirement.REQUIRED:
                lines.append(f"  - {f.name} ({f.field_type.value}): {f.description}")

        lines.append("")
        lines.append("Recommended Fields:")
        for f in contract.output_fields:
            if f.requirement == FieldRequirement.RECOMMENDED:
                lines.append(f"  - {f.name} ({f.field_type.value}): {f.description}")

        lines.append("")
        lines.append("Quality Thresholds:")
        for name, value in contract.quality_thresholds.items():
            lines.append(f"  - {name}: {value}")

        return "\n".join(lines)


# === 유틸리티 함수 ===

def validate_phase_transition(
    source_phase: str,
    target_phase: str,
    context: Any,
    strict: bool = True,
    raise_on_failure: bool = True,
) -> ValidationResult:
    """
    Phase 전환 검증 헬퍼 함수

    Args:
        source_phase: 완료된 Phase
        target_phase: 시작할 Phase
        context: SharedContext
        strict: strict_mode 활성화 여부
        raise_on_failure: 실패 시 예외 발생 여부

    Returns:
        ValidationResult

    Raises:
        PhaseContractViolation: raise_on_failure=True이고 검증 실패 시
    """
    validator = PhaseContractValidator(strict_mode=strict)
    result = validator.validate_transition(source_phase, target_phase, context)

    if not result.is_valid and raise_on_failure:
        raise PhaseContractViolation(
            message=f"Phase contract violated: {', '.join(result.violations)}",
            source_phase=source_phase,
            target_phase=target_phase,
            missing_fields=result.missing_fields,
        )

    return result


def get_phase_requirements(phase: str) -> Dict[str, Any]:
    """
    Phase 요구사항 조회

    Args:
        phase: Phase 이름

    Returns:
        요구사항 딕셔너리
    """
    validator = PhaseContractValidator()
    contract = validator.get_contract(phase)

    if not contract:
        return {}

    return {
        "phase": phase,
        "description": contract.description,
        "required_fields": [
            {"name": f.name, "type": f.field_type.value, "description": f.description}
            for f in contract.output_fields
            if f.requirement == FieldRequirement.REQUIRED
        ],
        "recommended_fields": [
            {"name": f.name, "type": f.field_type.value, "description": f.description}
            for f in contract.output_fields
            if f.requirement == FieldRequirement.RECOMMENDED
        ],
        "quality_thresholds": contract.quality_thresholds,
    }
