# Phase 2: Refinement Engine 기술 명세

> **역할**: 지식 설계자 (Knowledge Architect)
> **입력**: Phase 1에서 검증된 엔티티-관계 (Integrated FK Detection)
> **출력**: 비즈니스 온톨로지, 지식 그래프, OWL2 추론 결과
> **최종 업데이트**: 2026-01-27
> **버전**: v1.0

---

## 1. 개요

Refinement Engine은 Phase 1의 물리적 발견(Physical Discovery)을 **비즈니스 지식(Business Knowledge)**으로 변환합니다.

### 핵심 기능
- **온톨로지 설계**: 엔티티, 속성, 관계를 정형화된 스키마로 구축
- **Cross-Entity 상관분석**: 팔란티어 스타일 예측 인사이트 생성
- **다중 에이전트 합의**: 품질 평가 및 의미 검증

### v1.0 핵심 기능
- **OWL 2 Full Reasoner**: Property Chain, Transitive, Inverse 추론
- **SHACL Validator**: 데이터 품질 제약조건 검증
- **Column-level Lineage**: 컬럼 수준 데이터 계보 추적
- **Agent Communication Bus**: 에이전트 간 협업 강화

---

## 2. 에이전트 구성 (4개)

### 2.1 OntologyArchitectAutonomousAgent

**역할**: 온톨로지 설계 및 인과관계 분석

```python
OntologyArchitectAutonomousAgent:
    - OntologyConcept 생성 (object_type, property, link_type)
    - 속성 정의 및 제약 조건 설정
    - Cross-Entity Correlation 분석 호출
    - Causal Inference 실행
    - OWL2Reasoner로 개념 추론
    - LineageTracker로 계보 구축
```

**Cross-Entity Correlation**:
```python
CrossEntityCorrelationAnalyzer:
    - compute_cross_table_correlations()  # Pearson 상관계수
    - detect_complementary_pairs()        # 합이 1인 쌍 탐지
    - detect_duplicate_columns()          # 동일 데이터 탐지
    - detect_threshold_effects()          # 임계값 비선형 효과
    - segment_analysis()                  # 세그먼트별 집중도 분석
```

### 2.2 ConflictResolverAutonomousAgent

**역할**: 온톨로지 충돌 해결

```python
ConflictResolverAutonomousAgent:
    - 명명 충돌 탐지 (같은 이름, 다른 의미)
    - 의미적 동등성 판단 (다른 이름, 같은 의미)
    - 중복 개념 병합
    - 대체 명칭 관리
    - AgentMemory로 충돌 패턴 학습
```

### 2.3 QualityJudgeAutonomousAgent

**역할**: 온톨로지 품질 평가

```python
QualityJudgeAutonomousAgent:
    - 개념 품질 점수 산출 (0-100%)
    - 상태 전이 결정: pending → provisional → approved
    - 증거 강도 평가
    - 신뢰도 보정 (Confidence Calibration)
    - SHACLValidator로 데이터 품질 검증
```

**Smart Consensus**:
```python
ConsensusMode:
    - full_skip: Phase 1 탐색 단계 (합의 생략)
    - lightweight: 소수 개념 (단일 LLM 호출)
    - full_consensus: 복잡한 결정 (다중 에이전트 합의)
```

### 2.4 SemanticValidatorAutonomousAgent

**역할**: 의미론적 검증

```python
SemanticValidatorAutonomousAgent:
    - 개념 간 일관성 검증
    - 제약 조건 위반 탐지
    - 비즈니스 규칙 준수 확인
    - Cross-reference 정합성 체크
    - OWL2Reasoner 추론 실행
    - SHACLValidator 검증 실행
    - AgentBus로 검증 결과 브로드캐스트
```

---

## 3. Semantic Reasoning 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│              Semantic Reasoning Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 1: Knowledge Graph Construction                       │    │
│  │  • RDF 트리플 생성                                            │    │
│  │  • Subject-Predicate-Object 구조화                           │    │
│  │  파일: analysis/knowledge_graph.py                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 2: OWL 2 Reasoning                                    │    │
│  │  • Property Chain 추론                                        │    │
│  │    hasSupervisor · worksIn → belongsToDepartment             │    │
│  │  • Transitive Property 추론                                   │    │
│  │    isPartOf(A,B) ∧ isPartOf(B,C) → isPartOf(A,C)             │    │
│  │  • Inverse Property 추론                                      │    │
│  │    manages(X,Y) ↔ reportsTo(Y,X)                             │    │
│  │  파일: autonomous/analysis/owl2_reasoner.py                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 3: SHACL Validation                                   │    │
│  │  • Cardinality 검증 (min/max count)                          │    │
│  │  • Datatype 검증 (xsd:string, xsd:integer 등)                │    │
│  │  • Pattern 검증 (정규식)                                      │    │
│  │  • Shape 자동 추론                                            │    │
│  │  파일: autonomous/analysis/shacl_validator.py                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 4: Lineage Tracking                                   │    │
│  │  • Column-level 데이터 계보                                   │    │
│  │  • Upstream/Downstream 추적                                   │    │
│  │  • Impact Analysis                                            │    │
│  │  파일: lineage/column_lineage.py                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. OWL 2 Reasoner 상세

### 4.1 지원 Axiom 유형

```python
class OWLAxiomType(Enum):
    # Class Axioms
    SUBCLASS_OF = "SubClassOf"
    EQUIVALENT_CLASS = "EquivalentClasses"
    DISJOINT_CLASSES = "DisjointClasses"
    DISJOINT_UNION = "DisjointUnion"

    # Property Axioms
    SUBPROPERTY_OF = "SubObjectPropertyOf"
    INVERSE_PROPERTIES = "InverseObjectProperties"
    PROPERTY_CHAIN = "SubObjectPropertyOf(PropertyChain)"

    # Property Characteristics
    FUNCTIONAL_PROPERTY = "FunctionalObjectProperty"
    TRANSITIVE_PROPERTY = "TransitiveObjectProperty"
    SYMMETRIC_PROPERTY = "SymmetricObjectProperty"

    # Cardinality Restrictions
    QUALIFIED_CARDINALITY = "QualifiedCardinalityRestriction"
    MIN_CARDINALITY = "MinCardinalityRestriction"
    MAX_CARDINALITY = "MaxCardinalityRestriction"
```

### 4.2 추론 예시

```
입력 트리플:
  - Employee emp1 hasSupervisor emp2
  - Employee emp2 worksIn dept1

Property Chain Axiom:
  - hasSupervisor · worksIn → belongsToDepartment

추론 결과:
  - Employee emp1 belongsToDepartment dept1
```

---

## 5. SHACL Validator 상세

### 5.1 지원 제약조건

```python
class ConstraintType(Enum):
    MIN_COUNT = "minCount"       # 최소 발생 횟수
    MAX_COUNT = "maxCount"       # 최대 발생 횟수
    DATATYPE = "datatype"        # 데이터타입
    PATTERN = "pattern"          # 정규식 패턴
    NODE_KIND = "nodeKind"       # IRI/Literal/BlankNode
    IN = "in"                    # 열거형 값
    CLASS = "class"              # 타입 제약
    MIN_LENGTH = "minLength"     # 최소 길이
    MAX_LENGTH = "maxLength"     # 최대 길이
```

### 5.2 Shape 자동 추론

```python
# 데이터에서 SHACL Shape 자동 생성
shapes = validator.infer_shapes(sample_data)

# 생성된 Shape 예시
NodeShape(
    name="CustomerShape",
    target_class="Customer",
    properties=[
        PropertyConstraint(
            path="customer_id",
            constraints={
                ConstraintType.MIN_COUNT: 1,
                ConstraintType.MAX_COUNT: 1,
                ConstraintType.DATATYPE: "xsd:string",
                ConstraintType.PATTERN: r"CUST-\d{6}"
            }
        )
    ]
)
```

---

## 6. Cross-Entity Correlation 분석

### 6.1 분석 유형

| 분석 유형 | 설명 | 출력 예시 |
|----------|------|----------|
| **상관관계** | 테이블 간 숫자 컬럼 Pearson 상관 | "Daily_Capacity ↔ Max_Weight: r=0.527" |
| **보완 관계** | 합이 1.0인 컬럼 쌍 탐지 | "신규비율 + 재방문비율 = 1.0 (중복)" |
| **동일 데이터** | 100% 일치 컬럼 탐지 | "모바일 ≡ 전체 (데이터 오류)" |
| **임계값 효과** | 비선형 변화 지점 발견 | "용량 > 31.5 시 중량 5544% 증가" |
| **세그먼트 집중도** | 카테고리별 분포 분석 | "상위 5개 채널이 92.5% 차지" |

### 6.2 출력: Predictive Insight

```json
{
  "insight_id": "PI_001",
  "insight_type": "simulation_based",
  "source_table": "warehouse_capacities",
  "source_column": "Daily_Capacity",
  "target_table": "freight_rates",
  "target_column": "Max_Weight_Quant",
  "correlation_coefficient": 0.527,
  "p_value": 0.020,
  "confidence": 0.7
}
```

---

## 7. 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Phase 2: Refinement                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1 Results                                                     │
│  (entities, relationships, evidence)                                 │
│       │                                                              │
│       ▼                                                              │
│  ┌───────────────────────────────────────────────┐                  │
│  │       Ontology Architect Agent                │                  │
│  │  ┌─────────────────────────────────────────┐  │                  │
│  │  │ Cross-Entity Correlation Analyzer       │  │                  │
│  │  │  - Pearson Correlation                  │  │                  │
│  │  │  - Threshold Effects                    │  │                  │
│  │  │  - Segment Analysis                     │  │                  │
│  │  └─────────────────────────────────────────┘  │                  │
│  │  ┌─────────────────────────────────────────┐  │                  │
│  │  │ OWL2Reasoner                            │  │                  │
│  │  │ LineageTracker                          │  │                  │
│  │  └─────────────────────────────────────────┘  │                  │
│  └───────────────────────────┬───────────────────┘                  │
│                              │                                       │
│       ┌──────────────────────┼──────────────────────┐               │
│       ▼                      ▼                      ▼               │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │
│  │  Conflict   │      │   Quality   │      │  Semantic   │         │
│  │  Resolver   │      │    Judge    │      │  Validator  │         │
│  │             │      │             │      │             │         │
│  │ AgentMemory │      │SHACLValid   │      │ OWL2Reason  │         │
│  └──────┬──────┘      └──────┬──────┘      │ SHACLValid  │         │
│         │                    │              │ AgentBus    │         │
│         │                    │              └──────┬──────┘         │
│         │                    │                     │                │
│         └────────────────────┴─────────────────────┘                │
│                              │                                       │
│                              ▼                                       │
│            ┌─────────────────────────────────┐                      │
│            │       Smart Consensus           │                      │
│            │  (lightweight/full) + AgentBus  │                      │
│            └───────────────┬─────────────────┘                      │
│                            │                                        │
│                            ▼                                        │
│            ┌─────────────────────────────────┐                      │
│            │        SharedContext            │                      │
│            │  • ontology_concepts            │                      │
│            │  • business_insights            │                      │
│            │  • inferred_triples              │                      │
│            │    (OWL2 추론 결과 저장)          │                      │
│            │  • dynamic_data                  │                      │
│            │    (SHACL 검증 결과 등 저장)      │                      │
│            │  • lineage_graph                │                      │
│            └─────────────────────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. 출력 데이터

### 8.1 Ontology Concept

```json
{
  "concept_id": "obj_customer_001",
  "concept_type": "object_type",
  "name": "Customer",
  "description": "비즈니스 고객 엔티티",
  "source_tables": ["vmi_customers", "order_list"],
  "status": "approved",
  "confidence": 0.92
}
```

### 8.2 OWL2 Inferred Triple

```json
{
  "subject": "employee_001",
  "predicate": "belongsToDepartment",
  "object": "dept_sales",
  "rule_applied": "PropertyChain(hasSupervisor, worksIn)",
  "confidence": 0.85
}
```

### 8.3 SHACL Validation Result

```json
{
  "conforms": false,
  "total_nodes": 150,
  "conforming_nodes": 147,
  "violations": [
    {
      "focus_node": "customer_123",
      "path": "email",
      "constraint": "pattern",
      "message": "Does not match email pattern"
    }
  ]
}
```

---

## 9. 핵심 분석 모듈

| 모듈 | 파일 | 기능 |
|------|------|------|
| OWL2 Reasoner | `autonomous/analysis/owl2_reasoner.py` | Property Chain, Transitive 추론 |
| SHACL Validator | `autonomous/analysis/shacl_validator.py` | 데이터 품질 검증 |
| Column Lineage | `lineage/column_lineage.py` | 컬럼 수준 계보 |
| Cross-Entity | `analysis/cross_entity_correlation.py` | 테이블 간 상관분석 |
| Knowledge Graph | `analysis/knowledge_graph.py` | RDF 트리플 구축 |
| Graph Embedding | `autonomous/analysis/owl2_reasoner.py` | OWL2 기반 그래프 추론 |
| Causal Impact | `analysis/causal_impact.py` | 인과관계 분석 |
| Business Insights | `analysis/business_insights.py` | KPI, 이상치 탐지 |

---

## 10. Phase Gate 검증

Refinement 완료 시 다음 조건 확인:

```python
Phase2Gate:
    - ontology_concepts >= 1   # 최소 1개 개념 생성
    - approved_ratio >= 0.5    # 50% 이상 승인
    - no critical conflicts    # 심각한 충돌 없음
    - completion_signal == [DONE]

    # 추가 검증
    - owl2_reasoning executed  # OWL2 추론 실행됨
    - shacl_validation executed # SHACL 검증 실행됨
    - lineage_graph built      # Lineage 그래프 구축됨
```

실패 시 자동 Fallback 적용 후 Phase 3 진행.

---

*Document Version: v1.0*
*Last Updated: 2026-01-27*
