"""
SHACL Validator (v16.0)

Shapes Constraint Language 기반 데이터 품질 검증:
- Node Shapes (노드 제약)
- Property Shapes (속성 제약)
- Cardinality Constraints (기수 제약)
- Value Constraints (값 제약)
- Pattern Constraints (패턴 제약)
- Custom Rules (사용자 정의 규칙)

SharedContext와 연동하여 FK 관계 및 데이터 품질 검증

사용법:
    from .shacl_validator import run_shacl_validation_and_update_context
    result = run_shacl_validation_and_update_context(shared_context)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Callable, Union, Tuple, TYPE_CHECKING
from enum import Enum
from collections import defaultdict
import re
import logging

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


class SHACLSeverity(Enum):
    """SHACL 위반 심각도"""
    VIOLATION = "Violation"      # 필수 제약 위반
    WARNING = "Warning"          # 경고
    INFO = "Info"                # 정보성


class ConstraintType(Enum):
    """SHACL 제약 유형"""
    # Cardinality
    MIN_COUNT = "sh:minCount"
    MAX_COUNT = "sh:maxCount"

    # Value Type
    DATATYPE = "sh:datatype"
    NODE_KIND = "sh:nodeKind"
    CLASS = "sh:class"

    # Value Range
    MIN_INCLUSIVE = "sh:minInclusive"
    MAX_INCLUSIVE = "sh:maxInclusive"
    MIN_EXCLUSIVE = "sh:minExclusive"
    MAX_EXCLUSIVE = "sh:maxExclusive"

    # String
    MIN_LENGTH = "sh:minLength"
    MAX_LENGTH = "sh:maxLength"
    PATTERN = "sh:pattern"
    LANGUAGE_IN = "sh:languageIn"

    # Value Enumeration
    IN = "sh:in"
    HAS_VALUE = "sh:hasValue"

    # Logical
    NOT = "sh:not"
    AND = "sh:and"
    OR = "sh:or"
    XONE = "sh:xone"

    # Property Pair
    EQUALS = "sh:equals"
    DISJOINT = "sh:disjoint"
    LESS_THAN = "sh:lessThan"
    LESS_THAN_OR_EQUALS = "sh:lessThanOrEquals"

    # Uniqueness
    UNIQUE_LANG = "sh:uniqueLang"

    # Qualified
    QUALIFIED_MIN_COUNT = "sh:qualifiedMinCount"
    QUALIFIED_MAX_COUNT = "sh:qualifiedMaxCount"

    # Custom
    SPARQL = "sh:sparql"
    CUSTOM_FUNCTION = "custom:function"


@dataclass
class SHACLConstraint:
    """SHACL 제약 정의"""
    constraint_type: ConstraintType
    value: Any  # 제약 값 (숫자, 문자열, 리스트 등)
    message: Optional[str] = None
    severity: SHACLSeverity = SHACLSeverity.VIOLATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type.value,
            "value": self.value if not callable(self.value) else "custom_function",
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclass
class PropertyShape:
    """SHACL Property Shape"""
    path: str  # 속성 경로 (컬럼명)
    constraints: List[SHACLConstraint] = field(default_factory=list)

    name: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "constraints": [c.to_dict() for c in self.constraints],
        }


@dataclass
class NodeShape:
    """SHACL Node Shape"""
    target_class: str  # 대상 클래스 (테이블명)
    property_shapes: List[PropertyShape] = field(default_factory=list)

    name: Optional[str] = None
    description: Optional[str] = None
    is_closed: bool = False  # closed shape 여부

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_class": self.target_class,
            "name": self.name,
            "description": self.description,
            "is_closed": self.is_closed,
            "property_shapes": [ps.to_dict() for ps in self.property_shapes],
        }


@dataclass
class ValidationResult:
    """검증 결과"""
    conforms: bool
    focus_node: str  # 대상 노드 (테이블.컬럼 또는 행)
    result_path: Optional[str] = None  # 위반 속성 경로
    value: Any = None  # 위반 값

    constraint_type: Optional[ConstraintType] = None
    source_shape: Optional[str] = None
    message: str = ""
    severity: SHACLSeverity = SHACLSeverity.VIOLATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conforms": self.conforms,
            "focus_node": self.focus_node,
            "result_path": self.result_path,
            "value": str(self.value) if self.value is not None else None,
            "constraint_type": self.constraint_type.value if self.constraint_type else None,
            "source_shape": self.source_shape,
            "message": self.message,
            "severity": self.severity.value,
        }


class SHACLValidator:
    """
    SHACL Validator

    기능:
    1. Node Shape 검증 (테이블 구조)
    2. Property Shape 검증 (컬럼 제약)
    3. Cardinality 검증 (minCount, maxCount)
    4. Value Range 검증 (min/max)
    5. Pattern 검증 (정규식)
    6. FK 참조 무결성 검증
    7. 사용자 정의 규칙 검증

    SharedContext 연동:
    - 입력: tables, enhanced_fk_candidates
    - 출력: shacl_validation_results (dynamic_data)
    """

    def __init__(self):
        self.shapes: List[NodeShape] = []
        self.results: List[ValidationResult] = []
        self._custom_validators: Dict[str, Callable] = {}

    def validate(
        self,
        data: Dict[str, List[Dict[str, Any]]],  # {table_name: [row_dicts]}
        shapes: Optional[List[NodeShape]] = None,
        fk_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ValidationResult]:
        """
        SHACL 검증 실행

        Args:
            data: 테이블별 데이터 {table_name: [rows]}
            shapes: Node Shape 목록 (없으면 자동 생성)
            fk_candidates: FK 후보 (참조 무결성 검증용)

        Returns:
            검증 결과 리스트
        """
        self.results = []

        # Shape 자동 생성 (없으면)
        if not shapes:
            shapes = self._generate_default_shapes(data)
        self.shapes = shapes

        # 1. Node Shape 검증
        for shape in shapes:
            table_name = shape.target_class
            table_data = data.get(table_name, [])

            for row_idx, row in enumerate(table_data):
                focus_node = f"{table_name}[{row_idx}]"

                # Property Shape 검증
                for prop_shape in shape.property_shapes:
                    self._validate_property(
                        focus_node=focus_node,
                        property_shape=prop_shape,
                        row=row,
                        shape_name=shape.name or table_name,
                    )

                # Closed Shape 검증
                if shape.is_closed:
                    self._validate_closed_shape(
                        focus_node=focus_node,
                        shape=shape,
                        row=row,
                    )

        # 2. FK 참조 무결성 검증
        if fk_candidates:
            self._validate_fk_integrity(data, fk_candidates)

        logger.info(
            f"[SHACLValidator] Completed: {len(self.results)} results, "
            f"{sum(1 for r in self.results if not r.conforms)} violations"
        )

        return self.results

    def _generate_default_shapes(
        self,
        data: Dict[str, List[Dict[str, Any]]],
    ) -> List[NodeShape]:
        """데이터에서 기본 Shape 자동 생성"""
        shapes = []

        for table_name, rows in data.items():
            if not rows:
                continue

            # 컬럼 분석
            sample_row = rows[0]
            property_shapes = []

            for col_name, value in sample_row.items():
                constraints = []

                # 데이터 타입 기반 제약 추론
                if isinstance(value, int):
                    # 음수 불가 ID 패턴
                    if col_name.lower().endswith("_id") or col_name.lower() == "id":
                        constraints.append(SHACLConstraint(
                            constraint_type=ConstraintType.MIN_INCLUSIVE,
                            value=1,
                            message=f"{col_name} must be positive",
                        ))

                elif isinstance(value, str):
                    # 필수 컬럼 (이름 패턴)
                    if any(kw in col_name.lower() for kw in ["name", "title", "code"]):
                        constraints.append(SHACLConstraint(
                            constraint_type=ConstraintType.MIN_LENGTH,
                            value=1,
                            message=f"{col_name} cannot be empty",
                        ))

                    # 이메일 패턴
                    if "email" in col_name.lower():
                        constraints.append(SHACLConstraint(
                            constraint_type=ConstraintType.PATTERN,
                            value=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
                            message=f"{col_name} must be valid email format",
                            severity=SHACLSeverity.WARNING,
                        ))

                    # 전화번호 패턴
                    if any(kw in col_name.lower() for kw in ["phone", "tel", "mobile"]):
                        constraints.append(SHACLConstraint(
                            constraint_type=ConstraintType.PATTERN,
                            value=r"^[\d\-\+\s\(\)]+$",
                            message=f"{col_name} must be valid phone format",
                            severity=SHACLSeverity.WARNING,
                        ))

                if constraints:
                    property_shapes.append(PropertyShape(
                        path=col_name,
                        name=f"{table_name}.{col_name}",
                        constraints=constraints,
                    ))

            shapes.append(NodeShape(
                target_class=table_name,
                name=f"{table_name}Shape",
                property_shapes=property_shapes,
            ))

        return shapes

    def _validate_property(
        self,
        focus_node: str,
        property_shape: PropertyShape,
        row: Dict[str, Any],
        shape_name: str,
    ) -> None:
        """Property Shape 검증"""
        path = property_shape.path
        value = row.get(path)

        for constraint in property_shape.constraints:
            result = self._check_constraint(
                value=value,
                constraint=constraint,
                path=path,
                row=row,
            )

            self.results.append(ValidationResult(
                conforms=result,
                focus_node=focus_node,
                result_path=path,
                value=value,
                constraint_type=constraint.constraint_type,
                source_shape=shape_name,
                message=constraint.message or f"Constraint {constraint.constraint_type.value} violated",
                severity=constraint.severity if not result else SHACLSeverity.INFO,
            ))

    def _check_constraint(
        self,
        value: Any,
        constraint: SHACLConstraint,
        path: str,
        row: Dict[str, Any],
    ) -> bool:
        """개별 제약 검사"""
        ct = constraint.constraint_type
        cv = constraint.value

        # Null 처리
        if value is None:
            if ct == ConstraintType.MIN_COUNT:
                return 0 >= cv
            return True  # 대부분 제약은 값이 있을 때만 적용

        # Cardinality
        if ct == ConstraintType.MIN_COUNT:
            count = 1 if value is not None else 0
            return count >= cv
        elif ct == ConstraintType.MAX_COUNT:
            count = 1 if value is not None else 0
            return count <= cv

        # Value Range
        elif ct == ConstraintType.MIN_INCLUSIVE:
            try:
                return float(value) >= float(cv)
            except (ValueError, TypeError):
                return False
        elif ct == ConstraintType.MAX_INCLUSIVE:
            try:
                return float(value) <= float(cv)
            except (ValueError, TypeError):
                return False
        elif ct == ConstraintType.MIN_EXCLUSIVE:
            try:
                return float(value) > float(cv)
            except (ValueError, TypeError):
                return False
        elif ct == ConstraintType.MAX_EXCLUSIVE:
            try:
                return float(value) < float(cv)
            except (ValueError, TypeError):
                return False

        # String
        elif ct == ConstraintType.MIN_LENGTH:
            return len(str(value)) >= cv
        elif ct == ConstraintType.MAX_LENGTH:
            return len(str(value)) <= cv
        elif ct == ConstraintType.PATTERN:
            try:
                return bool(re.match(cv, str(value)))
            except re.error:
                return False

        # Value Enumeration
        elif ct == ConstraintType.IN:
            return value in cv
        elif ct == ConstraintType.HAS_VALUE:
            return value == cv

        # Property Pair (다른 컬럼과 비교)
        elif ct == ConstraintType.EQUALS:
            other_value = row.get(cv)
            return value == other_value
        elif ct == ConstraintType.DISJOINT:
            other_value = row.get(cv)
            return value != other_value
        elif ct == ConstraintType.LESS_THAN:
            other_value = row.get(cv)
            try:
                return float(value) < float(other_value)
            except (ValueError, TypeError):
                return False
        elif ct == ConstraintType.LESS_THAN_OR_EQUALS:
            other_value = row.get(cv)
            try:
                return float(value) <= float(other_value)
            except (ValueError, TypeError):
                return False

        # Custom Function
        elif ct == ConstraintType.CUSTOM_FUNCTION:
            if callable(cv):
                try:
                    return cv(value, row)
                except Exception:
                    return False

        return True

    def _validate_closed_shape(
        self,
        focus_node: str,
        shape: NodeShape,
        row: Dict[str, Any],
    ) -> None:
        """Closed Shape 검증 (허용된 속성만)"""
        allowed_props = {ps.path for ps in shape.property_shapes}
        actual_props = set(row.keys())

        extra_props = actual_props - allowed_props

        for extra in extra_props:
            self.results.append(ValidationResult(
                conforms=False,
                focus_node=focus_node,
                result_path=extra,
                message=f"Property {extra} not allowed in closed shape {shape.name}",
                severity=SHACLSeverity.WARNING,
            ))

    def _validate_fk_integrity(
        self,
        data: Dict[str, List[Dict[str, Any]]],
        fk_candidates: List[Dict[str, Any]],
    ) -> None:
        """FK 참조 무결성 검증"""
        for fk in fk_candidates:
            from_table = fk.get("from_table", "")
            from_column = fk.get("from_column", "")
            to_table = fk.get("to_table", "")
            to_column = fk.get("to_column", "")

            if not all([from_table, from_column, to_table, to_column]):
                continue

            from_data = data.get(from_table, [])
            to_data = data.get(to_table, [])

            # To 테이블의 참조 가능 값 집합
            valid_refs = set()
            for row in to_data:
                val = row.get(to_column)
                if val is not None:
                    valid_refs.add(str(val).lower())

            # From 테이블에서 무효 참조 검사
            for row_idx, row in enumerate(from_data):
                fk_value = row.get(from_column)

                # NULL은 허용 (optional FK)
                if fk_value is None or str(fk_value).lower() in ("", "null", "none"):
                    continue

                if str(fk_value).lower() not in valid_refs:
                    self.results.append(ValidationResult(
                        conforms=False,
                        focus_node=f"{from_table}[{row_idx}]",
                        result_path=from_column,
                        value=fk_value,
                        constraint_type=ConstraintType.CLASS,  # FK 참조를 Class 제약으로
                        message=f"FK violation: {from_column}={fk_value} not found in {to_table}.{to_column}",
                        severity=SHACLSeverity.VIOLATION,
                    ))

    def add_custom_validator(
        self,
        name: str,
        validator_fn: Callable[[Any, Dict], bool],
    ) -> None:
        """사용자 정의 검증 함수 추가"""
        self._custom_validators[name] = validator_fn

    def get_violations(self) -> List[ValidationResult]:
        """위반만 필터"""
        return [r for r in self.results if not r.conforms]

    def get_summary(self) -> Dict[str, Any]:
        """검증 결과 요약"""
        violations = self.get_violations()
        by_severity = defaultdict(int)
        by_constraint = defaultdict(int)

        for v in violations:
            by_severity[v.severity.value] += 1
            if v.constraint_type:
                by_constraint[v.constraint_type.value] += 1

        return {
            "total_checks": len(self.results),
            "violations": len(violations),
            "conforms": len(violations) == 0,
            "by_severity": dict(by_severity),
            "by_constraint_type": dict(by_constraint),
        }


# SharedContext 연동 함수
def run_shacl_validation_and_update_context(
    shared_context: "SharedContext",
    custom_shapes: Optional[List[NodeShape]] = None,
) -> Dict[str, Any]:
    """
    SHACL 검증 실행 후 SharedContext 업데이트

    사용법 (DataQualityAgent 에이전트에서):
        from .shacl_validator import run_shacl_validation_and_update_context
        result = run_shacl_validation_and_update_context(self.shared_context)

    Args:
        shared_context: SharedContext 인스턴스
        custom_shapes: 사용자 정의 Shape (없으면 자동 생성)

    Returns:
        검증 결과 요약
    """
    validator = SHACLValidator()

    # 테이블 데이터 수집
    data = {}
    for table_name, table_info in shared_context.tables.items():
        sample_data = (
            table_info.sample_data if hasattr(table_info, 'sample_data')
            else table_info.get('sample_data', [])
        )
        data[table_name] = sample_data

    if not data:
        logger.warning("[SHACLValidator] No data found in SharedContext")
        return {
            "total_checks": 0,
            "violations": 0,
            "conforms": True,
        }

    # FK 후보
    fk_candidates = shared_context.enhanced_fk_candidates

    # 검증 실행
    results = validator.validate(
        data=data,
        shapes=custom_shapes,
        fk_candidates=fk_candidates,
    )

    # 결과 요약
    summary = validator.get_summary()

    # SharedContext에 저장
    shared_context.set_dynamic("shacl_validation_results", [
        r.to_dict() for r in results
    ])
    shared_context.set_dynamic("shacl_violations", [
        r.to_dict() for r in validator.get_violations()
    ])
    shared_context.set_dynamic("shacl_shapes", [
        s.to_dict() for s in validator.shapes
    ])
    shared_context.set_dynamic("shacl_summary", summary)

    logger.info(
        f"[SHACLValidator] Completed: {summary['total_checks']} checks, "
        f"{summary['violations']} violations"
    )

    return summary


# Shape 생성 헬퍼 함수
def create_property_shape(
    path: str,
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
    datatype: Optional[str] = None,
    pattern: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    in_values: Optional[List[Any]] = None,
    name: Optional[str] = None,
) -> PropertyShape:
    """Property Shape 생성 헬퍼"""
    constraints = []

    if min_count is not None:
        constraints.append(SHACLConstraint(
            constraint_type=ConstraintType.MIN_COUNT,
            value=min_count,
        ))
    if max_count is not None:
        constraints.append(SHACLConstraint(
            constraint_type=ConstraintType.MAX_COUNT,
            value=max_count,
        ))
    if pattern is not None:
        constraints.append(SHACLConstraint(
            constraint_type=ConstraintType.PATTERN,
            value=pattern,
        ))
    if min_length is not None:
        constraints.append(SHACLConstraint(
            constraint_type=ConstraintType.MIN_LENGTH,
            value=min_length,
        ))
    if max_length is not None:
        constraints.append(SHACLConstraint(
            constraint_type=ConstraintType.MAX_LENGTH,
            value=max_length,
        ))
    if in_values is not None:
        constraints.append(SHACLConstraint(
            constraint_type=ConstraintType.IN,
            value=in_values,
        ))

    return PropertyShape(
        path=path,
        name=name or path,
        constraints=constraints,
    )


def create_node_shape(
    target_class: str,
    property_shapes: List[PropertyShape],
    name: Optional[str] = None,
    is_closed: bool = False,
) -> NodeShape:
    """Node Shape 생성 헬퍼"""
    return NodeShape(
        target_class=target_class,
        name=name or f"{target_class}Shape",
        property_shapes=property_shapes,
        is_closed=is_closed,
    )


# 자주 사용되는 사전 정의 Shape
def create_employee_shape() -> NodeShape:
    """Employee 테이블용 기본 Shape"""
    return create_node_shape(
        target_class="employees",
        property_shapes=[
            create_property_shape("emp_id", min_count=1),
            create_property_shape("name", min_length=1, max_length=100),
            create_property_shape("email", pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
        ],
    )


def create_order_shape() -> NodeShape:
    """Order 테이블용 기본 Shape"""
    return create_node_shape(
        target_class="orders",
        property_shapes=[
            create_property_shape("order_id", min_count=1),
            create_property_shape("status", in_values=["pending", "confirmed", "shipped", "delivered", "cancelled"]),
        ],
    )
