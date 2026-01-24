"""
LLM Schema Analyzer v1.0

LLM-First Discovery 아키텍처의 핵심 모듈.
Rule-based 패턴 매칭 전에 LLM이 전체 스키마를 분석하여 FK 후보를 먼저 발굴합니다.

문제점 해결:
1. Self-Reference FK: manager_id → emp_id, parent_dept → department_id
2. Single Noun FK: trainer → employees, participant → employees
3. Semantic Role FK: head_of_dept → employees, target_dept → departments

Architecture:
Step 1: LLM Schema Analyzer (NEW) - 전체 스키마 분석
Step 2: Rule-based Validation (Fast Filter)
Step 3: LLM FK Relationship Confirmation
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import Instructor
try:
    import instructor
    from pydantic import BaseModel, Field
    _INSTRUCTOR_AVAILABLE = True
except ImportError:
    _INSTRUCTOR_AVAILABLE = False
    logger.warning("Instructor not available for Schema Analyzer")

# Try to import DSPy
try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


class FKType(Enum):
    """FK 관계 유형"""
    STANDARD = "standard"           # 일반 FK (customer_id → customers)
    SELF_REFERENCE = "self_reference"  # 자기 참조 (manager_id → emp_id)
    SINGLE_NOUN = "single_noun"     # 단일 명사 (trainer → employees)
    SEMANTIC_ROLE = "semantic_role" # 시맨틱 역할 (head_of_dept → employees)
    HIERARCHICAL = "hierarchical"   # 계층 구조 (parent_dept → departments)


@dataclass
class DiscoveredFK:
    """LLM이 발굴한 FK 후보"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    fk_type: FKType
    confidence: float
    reasoning: str
    semantic_relationship: str = ""
    is_validated: bool = False


@dataclass
class TableAnalysis:
    """테이블 분석 결과"""
    table_name: str
    pk_column: str
    pk_confidence: float
    fk_columns: List[str]
    self_reference_columns: List[str]
    entity_type: str  # e.g., "person", "product", "transaction"
    is_hierarchical: bool


if _INSTRUCTOR_AVAILABLE:
    class SchemaFKDiscovery(BaseModel):
        """스키마 전체 FK 발굴 결과 (Instructor용)"""

        class DiscoveredRelation(BaseModel):
            source_table: str = Field(description="FK를 가진 테이블")
            source_column: str = Field(description="FK 컬럼")
            target_table: str = Field(description="참조되는 테이블")
            target_column: str = Field(description="참조되는 PK 컬럼")
            relationship_type: str = Field(
                description="관계 유형: standard, self_reference, single_noun, semantic_role, hierarchical"
            )
            confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
            semantic_meaning: str = Field(description="관계의 의미 설명")
            reasoning: str = Field(description="FK 판단 근거")

        discovered_fks: List[DiscoveredRelation] = Field(
            description="발굴된 모든 FK 관계"
        )
        self_reference_tables: List[str] = Field(
            description="자기 참조 구조를 가진 테이블 목록"
        )
        hierarchical_tables: List[str] = Field(
            description="계층 구조를 가진 테이블 목록 (parent_id, level 등)"
        )

    class SingleColumnAnalysis(BaseModel):
        """단일 컬럼 분석 결과"""
        is_fk: bool = Field(description="이 컬럼이 FK인지 여부")
        target_table: str = Field(description="참조 대상 테이블 (FK인 경우)")
        target_column: str = Field(description="참조 대상 컬럼 (FK인 경우)")
        fk_type: str = Field(description="FK 유형")
        confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
        reasoning: str = Field(description="판단 근거")

    class HierarchyAnalysis(BaseModel):
        """계층 구조 분석 결과"""
        is_hierarchical: bool = Field(description="계층 구조 여부")
        parent_column: str = Field(description="부모 참조 컬럼")
        child_column: str = Field(description="자식 ID 컬럼 (보통 PK)")
        level_column: Optional[str] = Field(default=None, description="레벨 컬럼")
        path_column: Optional[str] = Field(default=None, description="경로 컬럼")
        reasoning: str = Field(description="판단 근거")


if _DSPY_AVAILABLE:
    class SchemaAnalyzerSignature(dspy.Signature):
        """전체 스키마를 분석하여 FK 관계를 발굴합니다."""

        schema_info: str = dspy.InputField(
            desc="테이블별 컬럼 정보와 샘플 데이터"
        )
        domain_context: str = dspy.InputField(
            desc="도메인 컨텍스트 (e.g., hr, ecommerce, healthcare)"
        )

        discovered_fks: str = dspy.OutputField(
            desc="발굴된 FK 관계 목록 (JSON 형식)"
        )
        self_reference_fks: str = dspy.OutputField(
            desc="자기 참조 FK 목록"
        )
        reasoning: str = dspy.OutputField(
            desc="분석 과정 설명"
        )

    class HierarchyDetectorSignature(dspy.Signature):
        """테이블의 계층 구조를 탐지합니다."""

        table_name: str = dspy.InputField(desc="테이블명")
        columns: str = dspy.InputField(desc="컬럼 목록")
        sample_data: str = dspy.InputField(desc="샘플 데이터")

        is_hierarchical: bool = dspy.OutputField(
            desc="계층 구조 여부"
        )
        parent_column: str = dspy.OutputField(
            desc="부모 참조 컬럼 (있는 경우)"
        )
        reasoning: str = dspy.OutputField(
            desc="판단 근거"
        )


class LLMSchemaAnalyzer:
    """
    LLM 기반 스키마 분석기

    전체 스키마를 LLM으로 먼저 분석하여 Rule-based가 놓칠 수 있는
    FK 관계를 발굴합니다.

    Features:
    - Self-Reference Detection: manager_id, parent_dept 등
    - Single Noun FK Detection: trainer, participant 등
    - Semantic Role Detection: head_of_dept, target_dept 등
    - Hierarchy Structure Detection: parent-child 관계
    """

    def __init__(
        self,
        use_instructor: bool = True,
        use_dspy: bool = False,
        min_confidence: float = 0.6,
        llm_client = None,
    ):
        self.use_instructor = use_instructor and _INSTRUCTOR_AVAILABLE
        self.use_dspy = use_dspy and _DSPY_AVAILABLE
        self.min_confidence = min_confidence
        self.llm_client = llm_client

        # Instructor client
        if self.use_instructor:
            try:
                import openai
                if llm_client:
                    self._instructor_client = instructor.from_openai(llm_client)
                else:
                    client = openai.OpenAI()
                    self._instructor_client = instructor.from_openai(client)
            except Exception as e:
                logger.warning(f"Failed to initialize Instructor client: {e}")
                self._instructor_client = None
        else:
            self._instructor_client = None

        # DSPy modules
        if self.use_dspy:
            self._dspy_analyzer = dspy.ChainOfThought(SchemaAnalyzerSignature)
            self._dspy_hierarchy = dspy.ChainOfThought(HierarchyDetectorSignature)
        else:
            self._dspy_analyzer = None
            self._dspy_hierarchy = None

        logger.info(
            f"LLMSchemaAnalyzer initialized: "
            f"instructor={self.use_instructor}, dspy={self.use_dspy}"
        )

    def analyze_schema(
        self,
        tables_data: Dict[str, Any],
        domain_context: str = "",
    ) -> List[DiscoveredFK]:
        """
        전체 스키마를 분석하여 FK 후보를 발굴합니다.

        Args:
            tables_data: 테이블 데이터 (columns, sample_data 포함)
            domain_context: 도메인 컨텍스트

        Returns:
            발굴된 FK 후보 리스트
        """
        discovered_fks = []

        # Step 1: Heuristic-based self-reference detection (fast)
        self_ref_fks = self._detect_self_references_heuristic(tables_data)
        discovered_fks.extend(self_ref_fks)
        logger.info(f"Heuristic self-reference detection: {len(self_ref_fks)} found")

        # Step 2: Heuristic-based single noun FK detection (fast)
        single_noun_fks = self._detect_single_noun_fks_heuristic(tables_data)
        discovered_fks.extend(single_noun_fks)
        logger.info(f"Heuristic single-noun FK detection: {len(single_noun_fks)} found")

        # Step 3: LLM-based comprehensive analysis
        if self.use_instructor and self._instructor_client:
            llm_fks = self._analyze_with_instructor(tables_data, domain_context)

            # Merge with existing (avoid duplicates)
            existing_keys = {
                (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
                for fk in discovered_fks
            }
            for fk in llm_fks:
                key = (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
                if key not in existing_keys:
                    discovered_fks.append(fk)
                    existing_keys.add(key)

            logger.info(f"LLM comprehensive analysis: {len(llm_fks)} candidates")

        logger.info(f"Total discovered FKs: {len(discovered_fks)}")
        return discovered_fks

    def _detect_self_references_heuristic(
        self,
        tables_data: Dict[str, Any],
    ) -> List[DiscoveredFK]:
        """
        휴리스틱 기반 자기 참조 FK 탐지

        패턴:
        - manager_id, supervisor_id, parent_id, reports_to
        - parent_dept, parent_category, parent_node
        """
        discovered = []

        # Self-reference column patterns
        self_ref_patterns = {
            # Person hierarchy (employee → employee)
            'manager_id': ('employees', 'emp_id'),
            'manager': ('employees', 'emp_id'),
            'supervisor_id': ('employees', 'emp_id'),
            'supervisor': ('employees', 'emp_id'),
            'reports_to': ('employees', 'emp_id'),
            'reports_to_id': ('employees', 'emp_id'),

            # Department hierarchy
            'parent_dept': ('departments', 'department_id'),
            'parent_dept_id': ('departments', 'department_id'),
            'parent_department': ('departments', 'department_id'),
            'parent_department_id': ('departments', 'department_id'),

            # Category hierarchy
            'parent_category': ('categories', 'category_id'),
            'parent_category_id': ('categories', 'category_id'),
            'parent_cat': ('categories', 'category_id'),

            # Generic hierarchy
            'parent_id': (None, None),  # Will resolve to same table's PK
            'parent': (None, None),
        }

        for table_name, table_info in tables_data.items():
            columns = self._get_column_names(table_info)
            pk_column = self._find_pk_column(table_name, columns)

            for col in columns:
                col_lower = col.lower()

                # Check explicit patterns
                for pattern, (target_hint, target_col_hint) in self_ref_patterns.items():
                    if col_lower == pattern or col_lower.endswith(f"_{pattern}"):
                        # Determine target table
                        if target_hint is None:
                            # parent_id pattern - same table
                            target_table = table_name
                            target_column = pk_column
                        elif target_hint in tables_data:
                            target_table = target_hint
                            target_column = target_col_hint or self._find_pk_column(
                                target_hint, self._get_column_names(tables_data[target_hint])
                            )
                        else:
                            # Fallback: check if it's same table reference
                            target_table = table_name
                            target_column = pk_column

                        if target_column and col != target_column:
                            discovered.append(DiscoveredFK(
                                source_table=table_name,
                                source_column=col,
                                target_table=target_table,
                                target_column=target_column,
                                fk_type=FKType.SELF_REFERENCE if target_table == table_name else FKType.HIERARCHICAL,
                                confidence=0.85,
                                reasoning=f"Self-reference pattern '{pattern}' detected",
                                semantic_relationship="hierarchical_parent",
                            ))
                            break

        return discovered

    def _detect_single_noun_fks_heuristic(
        self,
        tables_data: Dict[str, Any],
    ) -> List[DiscoveredFK]:
        """
        휴리스틱 기반 단일 명사 FK 탐지

        패턴:
        - trainer → employees.emp_id (no _id suffix)
        - participant → employees.emp_id
        - reviewer → employees.emp_id
        - approver → employees.emp_id
        """
        discovered = []

        # Single noun patterns (column_name → target_table.pk_column)
        single_noun_hints = {
            # Person roles (→ employees/staff/users)
            'trainer': ['employees', 'staff', 'users'],
            'trainee': ['employees', 'staff', 'users'],
            'participant': ['employees', 'staff', 'users'],
            'attendee': ['employees', 'staff', 'users'],
            'reviewer': ['employees', 'staff', 'users'],
            'approver': ['employees', 'staff', 'users'],
            'assignee': ['employees', 'staff', 'users'],
            'requester': ['employees', 'staff', 'users'],
            'creator': ['employees', 'staff', 'users'],
            'modifier': ['employees', 'staff', 'users'],
            'owner': ['employees', 'staff', 'users'],
            'author': ['employees', 'staff', 'users'],

            # Healthcare roles
            'physician': ['doctors', 'physicians', 'staff'],
            'nurse': ['nurses', 'staff'],
            'patient': ['patients', 'members'],

            # Business entities
            'vendor': ['vendors', 'suppliers'],
            'supplier': ['vendors', 'suppliers'],
            'customer': ['customers', 'clients', 'users'],
            'buyer': ['customers', 'clients', 'users'],
            'seller': ['vendors', 'sellers'],
        }

        table_names = list(tables_data.keys())

        for table_name, table_info in tables_data.items():
            columns = self._get_column_names(table_info)

            for col in columns:
                col_lower = col.lower()

                # Skip if column already has _id suffix (handled by standard FK detection)
                if col_lower.endswith('_id') or col_lower.endswith('_code'):
                    continue

                # Check single noun patterns
                for pattern, target_hints in single_noun_hints.items():
                    if col_lower == pattern or col_lower.endswith(f"_{pattern}"):
                        # Find matching target table
                        for target_hint in target_hints:
                            for tn in table_names:
                                if target_hint in tn.lower() or tn.lower() in target_hint:
                                    target_columns = self._get_column_names(tables_data[tn])
                                    pk_col = self._find_pk_column(tn, target_columns)

                                    if pk_col and (table_name != tn or col != pk_col):
                                        discovered.append(DiscoveredFK(
                                            source_table=table_name,
                                            source_column=col,
                                            target_table=tn,
                                            target_column=pk_col,
                                            fk_type=FKType.SINGLE_NOUN,
                                            confidence=0.75,
                                            reasoning=f"Single noun pattern '{pattern}' → {tn}",
                                            semantic_relationship=f"{pattern}_role",
                                        ))
                                        break
                            else:
                                continue
                            break

        return discovered

    def _analyze_with_instructor(
        self,
        tables_data: Dict[str, Any],
        domain_context: str,
    ) -> List[DiscoveredFK]:
        """Instructor를 사용한 종합 스키마 분석"""
        if not self._instructor_client:
            return []

        # Build schema summary for LLM
        schema_summary = self._build_schema_summary(tables_data)

        prompt = f"""Analyze this database schema and discover ALL Foreign Key relationships.

## Domain Context
{domain_context if domain_context else "General database"}

## Schema
{schema_summary}

## Instructions
Find ALL FK relationships including:

1. **Standard FKs**: column_id → other_table.id pattern
2. **Self-Reference FKs**: manager_id → same_table.emp_id, parent_dept → departments.dept_id
3. **Single Noun FKs**: trainer → employees.emp_id (no _id suffix)
4. **Semantic Role FKs**: head_of_dept → employees.emp_id, reviewed_by → employees.emp_id
5. **Hierarchical FKs**: parent_category → categories.category_id

IMPORTANT:
- Look for columns that reference entities in other tables even without explicit _id suffix
- Check for self-referential hierarchies (employee-manager, category-subcategory)
- Consider domain semantics (e.g., in HR domain, 'trainer' likely references employees)

Return ALL discovered FK relationships with confidence scores and reasoning.
"""

        try:
            result = self._instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=SchemaFKDiscovery,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a database schema expert. Discover ALL foreign key relationships including non-obvious ones like self-references and semantic roles."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
            )

            # Convert to DiscoveredFK objects
            discovered = []
            for rel in result.discovered_fks:
                fk_type_map = {
                    'standard': FKType.STANDARD,
                    'self_reference': FKType.SELF_REFERENCE,
                    'single_noun': FKType.SINGLE_NOUN,
                    'semantic_role': FKType.SEMANTIC_ROLE,
                    'hierarchical': FKType.HIERARCHICAL,
                }
                fk_type = fk_type_map.get(rel.relationship_type.lower(), FKType.STANDARD)

                discovered.append(DiscoveredFK(
                    source_table=rel.source_table,
                    source_column=rel.source_column,
                    target_table=rel.target_table,
                    target_column=rel.target_column,
                    fk_type=fk_type,
                    confidence=rel.confidence,
                    reasoning=rel.reasoning,
                    semantic_relationship=rel.semantic_meaning,
                ))

            return discovered

        except Exception as e:
            logger.warning(f"Instructor schema analysis failed: {e}")
            return []

    def _build_schema_summary(self, tables_data: Dict[str, Any]) -> str:
        """LLM 분석을 위한 스키마 요약 생성"""
        lines = []

        for table_name, table_info in tables_data.items():
            columns = self._get_column_names(table_info)
            pk_col = self._find_pk_column(table_name, columns)

            lines.append(f"\n### Table: {table_name}")
            lines.append(f"Columns: {', '.join(columns)}")
            lines.append(f"Likely PK: {pk_col}")

            # Add sample data
            sample_data = self._get_sample_data(table_info)
            if sample_data:
                lines.append(f"Sample: {sample_data[:2]}")

        return '\n'.join(lines)

    def _get_column_names(self, table_info: Dict[str, Any]) -> List[str]:
        """테이블 정보에서 컬럼명 추출"""
        columns = table_info.get('columns', [])

        if isinstance(columns, list) and columns:
            if isinstance(columns[0], dict):
                return [c.get('name', '') for c in columns]
            else:
                return columns
        return []

    def _find_pk_column(self, table_name: str, columns: List[str]) -> Optional[str]:
        """테이블의 PK 컬럼 찾기"""
        table_lower = table_name.lower()
        table_singular = table_lower.rstrip('s')

        # Common PK patterns
        pk_patterns = [
            f"{table_singular}_id",
            f"{table_lower}_id",
            "id",
            f"{table_singular[:4]}_id" if len(table_singular) > 4 else None,
        ]
        pk_patterns = [p for p in pk_patterns if p]

        for pattern in pk_patterns:
            for col in columns:
                if col.lower() == pattern:
                    return col

        # Fallback: first column ending with _id
        for col in columns:
            if col.lower().endswith('_id'):
                return col

        return columns[0] if columns else None

    def _get_sample_data(self, table_info: Dict[str, Any]) -> List[Dict]:
        """샘플 데이터 추출"""
        # Try different data formats
        sample_data = table_info.get('sample_data', [])
        if sample_data:
            return sample_data[:3]

        data = table_info.get('data', [])
        columns = self._get_column_names(table_info)

        if data and columns:
            samples = []
            for row in data[:3]:
                if isinstance(row, (list, tuple)):
                    samples.append(dict(zip(columns, row)))
                elif isinstance(row, dict):
                    samples.append(row)
            return samples

        return []

    def detect_hierarchies(
        self,
        tables_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        모든 테이블의 계층 구조를 탐지합니다.

        Returns:
            테이블별 계층 구조 정보
        """
        hierarchies = {}

        for table_name, table_info in tables_data.items():
            columns = self._get_column_names(table_info)
            pk_col = self._find_pk_column(table_name, columns)

            # Check for hierarchy indicators
            hierarchy_cols = []
            parent_col = None
            level_col = None

            for col in columns:
                col_lower = col.lower()

                # Parent reference columns
                if any(p in col_lower for p in ['parent', 'manager', 'supervisor', 'reports_to']):
                    parent_col = col
                    hierarchy_cols.append(col)

                # Level/depth columns
                if any(l in col_lower for l in ['level', 'depth', 'hierarchy', 'tier']):
                    level_col = col
                    hierarchy_cols.append(col)

            if parent_col:
                hierarchies[table_name] = {
                    'is_hierarchical': True,
                    'pk_column': pk_col,
                    'parent_column': parent_col,
                    'level_column': level_col,
                    'hierarchy_columns': hierarchy_cols,
                }

        return hierarchies


def create_schema_analyzer(llm_client=None) -> LLMSchemaAnalyzer:
    """Schema Analyzer 생성 헬퍼"""
    return LLMSchemaAnalyzer(
        use_instructor=True,
        use_dspy=False,
        llm_client=llm_client,
    )
