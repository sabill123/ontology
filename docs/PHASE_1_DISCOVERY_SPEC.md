# Phase 1: Discovery Engine 기술 명세

> **역할**: 데이터 탐정 (Data Detective)
> **입력**: Raw 데이터 테이블 (CSV, JSON, Parquet, Excel)
> **출력**: 검증된 엔티티 관계 및 구조적 인사이트
> **최종 업데이트**: 2026-01-27
> **버전**: v1.0

---

## 1. 개요

Discovery Engine은 **도메인 지식 없이(Zero-knowledge)** raw 데이터의 구조적, 의미적 특징을 추출하고 엔티티 간의 숨겨진 연결 고리를 찾아냅니다.

### 핵심 특징
- **LLM-First 분석**: 전체 데이터셋을 LLM에 직접 전달하여 비즈니스 맥락 파악
- **위상수학(TDA) 기반 구조 분석**: Betti Numbers로 데이터 형태 파악
- **Integrated FK Detection**: 복합 FK + 계층 구조 + Temporal FK 통합 탐지

### v1.0 핵심 기능
- **Composite FK Detection**: 복합키 외래키 자동 탐지
- **Hierarchy Detection**: Self-Reference FK 100% 탐지 (조직도, 카테고리 트리)
- **Temporal FK Detection**: SCD Type 2, Bitemporal 관계 탐지
- **Agent Communication Bus**: 에이전트 간 실시간 통신
- **Agent Memory**: 학습 기반 패턴 인식

---

## 2. 에이전트 구성 (6개)

### 2.1 DataAnalystAutonomousAgent

**역할**: 데이터 프로파일링 및 LLM 기반 비즈니스 맥락 분석

```python
DataAnalystAutonomousAgent:
    - 전체 데이터셋을 LLM에 직접 전달
    - 비즈니스 도메인 자동 탐지
    - 컬럼별 비즈니스 의미 분류
    - 데이터 품질 초기 평가
    - 도메인 컨텍스트 생성
```

### 2.2 TDAExpertAutonomousAgent

**역할**: 위상 데이터 분석 (Topological Data Analysis)

```python
TDAExpertAutonomousAgent:
    - Betti Numbers 계산 (β0, β1, β2)
    - Persistence Diagram 생성
    - Simplicial Complex 구축
    - 테이블 간 위상적 동형성(Homeomorphism) 판단
    - AgentMemory로 패턴 학습
```

**Betti Numbers 의미**:
- β0: 연결 성분 수 (Connected Components)
- β1: 1차원 구멍/사이클 (Loops) - 순환 참조 탐지
- β2: 2차원 공동 (Voids)

### 2.3 SchemaAnalystAutonomousAgent

**역할**: 스키마 프로파일링 및 정규화 분석

```python
SchemaAnalystAutonomousAgent:
    - 컬럼 타입 추론 (Semantic Type Detection)
    - Null 비율, Unique 비율 분석
    - Primary Key 후보 식별
    - 정규화 위반 체크 (1NF, 2NF, 3NF)
    - LineageTracker로 컬럼 계보 추적
```

### 2.4 ValueMatcherAutonomousAgent

**역할**: 값 기반 테이블 간 매칭

```python
ValueMatcherAutonomousAgent:
    - Jaccard Similarity 계산
    - Containment Ratio 분석
    - 값 패턴 매칭
    - Cross-table 값 중복 탐지
    - AgentMemory로 매칭 패턴 학습
```

**핵심 수식**:
- Containment: C(A, B) = |A ∩ B| / |A|
- Jaccard: J(A, B) = |A ∩ B| / |A ∪ B|

### 2.5 EntityClassifierAutonomousAgent

**역할**: 엔티티 유형 분류

```python
EntityClassifierAutonomousAgent:
    - 비즈니스 엔티티 인식 (Customer, Product, Order 등)
    - Cross-table 엔티티 매칭
    - 통합 엔티티(UnifiedEntity) 생성
    - OWL2Reasoner 연동으로 개념 추론
```

### 2.6 RelationshipDetectorAutonomousAgent

**역할**: 테이블 간 관계 탐지

```python
RelationshipDetectorAutonomousAgent:
    - FK 관계 발견 (Integrated FK Detection)
    - Composite FK 탐지 (복합키)
    - Hierarchy 탐지 (Self-Reference)
    - Temporal FK 탐지 (SCD Type 2)
    - 관계 카디널리티 추론 (1:1, 1:N, N:M)
    - 의미론적 관계 분류
    - Join Path 추론
    - AgentBus로 발견 결과 브로드캐스트
```

---

## 3. FK Detection 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│              Integrated FK Detection Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 1: Single FK Detection                               │    │
│  │  • Multi-Signal Scoring                                     │    │
│  │  • Direction Analysis                                       │    │
│  │  • Semantic Resolution                                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 2: Composite FK Detection                            │    │
│  │  • 2-4개 컬럼 조합 탐지                                      │    │
│  │  • 복합 유니크 비율 계산                                     │    │
│  │  • Cross-table 복합 값 포함도                                │    │
│  │  파일: analysis/composite_fk_detector.py                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 3: Hierarchy Detection                               │    │
│  │  • Self-Reference 패턴 매칭                                  │    │
│  │  • manager_id → emp_id, parent_id → id                      │    │
│  │  • 트리 구조 검증 (루트, 깊이, 사이클)                       │    │
│  │  파일: analysis/hierarchy_detector.py                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 4: Temporal FK Detection                             │    │
│  │  • SCD Type 2 패턴 인식                                      │    │
│  │  • valid_from, valid_to 컬럼 식별                            │    │
│  │  • 이력 테이블 ↔ 메인 테이블 연결                            │    │
│  │  파일: analysis/temporal_fk_detector.py                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Stage 5: Agent Bus Broadcast                               │    │
│  │  • 발견 결과를 다른 에이전트에 즉시 공유                      │    │
│  │  • await bus.broadcast_discovery("fk_detected", results)    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 1: Discovery                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Tables (CSV/JSON)                                               │
│       │                                                              │
│       ├──────────────┬──────────────┬──────────────┐                │
│       ▼              ▼              ▼              ▼                │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐             │
│  │   TDA   │   │ Schema  │   │  Value  │                             │
│  │ Expert  │   │ Analyst │   │ Matcher │                             │
│  └────┬────┘   └────┬────┘   └────┬────┘                             │
│       │              │              │                                 │
│       └──────────────┴──────────────┘                                 │
│                          │                                           │
│            ┌─────────────┴─────────────┐                            │
│            ▼                           ▼                            │
│  ┌───────────────────┐    ┌───────────────────────┐                │
│  │ Entity Classifier │    │ Relationship Detector │                │
│  │                   │    │                       │                │
│  │ • OWL2 추론 연동  │    │ • Single FK           │                │
│  └─────────┬─────────┘    │ • Composite FK        │                │
│            │              │ • Hierarchy           │                │
│            │              │ • Temporal FK         │                │
│            │              └───────────┬───────────┘                │
│            │                          │                             │
│            └──────────────────────────┘                             │
│                          │                                          │
│                          ▼                                          │
│            ┌─────────────────────────────┐                         │
│            │      Agent Communication    │                         │
│            │      Bus (브로드캐스트)      │                         │
│            └─────────────┬───────────────┘                         │
│                          │                                          │
│                          ▼                                          │
│            ┌─────────────────────────────┐                         │
│            │       SharedContext         │                         │
│            │  • unified_entities         │                         │
│            │  • homeomorphisms           │                         │
│            │  • enhanced_fk_candidates   │                         │
│            │  • evidence_chain           │                         │
│            │  • lineage_graph            │                         │
│            └─────────────────────────────┘                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 출력 데이터

### 5.1 Unified Entity

```json
{
  "entity_id": "ENT_CUSTOMER",
  "entity_type": "Customer",
  "canonical_name": "Customer",
  "source_tables": ["vmi_customers", "order_list"],
  "key_mappings": {
    "vmi_customers": "Customers",
    "order_list": "Ship_To_Party"
  },
  "confidence": 0.92,
  "verified_by_values": true
}
```

### 5.2 Enhanced FK Candidate

```json
{
  "from_table": "orders",
  "from_column": "buyer_id",
  "to_table": "customers",
  "to_column": "customer_id",
  "confidence": 0.87,
  "fk_score": 0.85,
  "direction_certain": true,
  "is_composite": false,
  "is_self_reference": false,
  "is_temporal": false,
  "signal_breakdown": {
    "value_overlap": {"score": 0.24, "containment_fk_in_pk": 0.96},
    "cardinality": {"score": 0.18, "fk_unique": 200, "pk_unique": 200},
    "referential_integrity": {"score": 0.23, "orphan_ratio": 0.04},
    "naming": {"score": 0.12, "entity_match": 1.0},
    "structure": {"score": 0.13, "pk_is_unique": true}
  }
}
```

### 5.3 Composite FK Candidate

```json
{
  "from_table": "order_details",
  "from_columns": ["order_id", "line_no"],
  "to_table": "order_headers",
  "to_columns": ["order_id", "line_no"],
  "cardinality": "N:1",
  "composite_score": 0.92,
  "confidence": "high",
  "is_composite": true
}
```

### 5.4 Hierarchy Candidate

```json
{
  "table": "employees",
  "child_column": "manager_id",
  "parent_column": "emp_id",
  "hierarchy_type": "org_chart",
  "max_depth": 5,
  "root_count": 3,
  "confidence": 0.94,
  "is_self_reference": true
}
```

### 5.5 Temporal FK Candidate

```json
{
  "from_table": "employees_history",
  "from_column": "emp_id",
  "to_table": "employees",
  "to_column": "emp_id",
  "temporal_type": "scd_type_2",
  "valid_from_col": "valid_from",
  "valid_to_col": "valid_to",
  "confidence": 0.87,
  "is_temporal": true
}
```

### 5.6 TDA Signature

```json
{
  "table": "order_list",
  "betti_numbers": {
    "beta_0": 5,
    "beta_1": 2,
    "beta_2": 0
  },
  "structural_complexity": "medium",
  "cyclic_references": ["Order -> Product -> Order"]
}
```

---

## 6. 핵심 분석 모듈

| 모듈 | 파일 | 기능 |
|------|------|------|
| Integrated FK | `analysis/integrated_fk_detector.py` | 통합 FK 탐지 |
| Enhanced FK Detector | `analysis/enhanced_fk_detector.py` | Multi-Signal + Direction |
| FK Signal Scorer | `analysis/fk_signal_scorer.py` | 5가지 신호 기반 FK 점수 |
| FK Direction Analyzer | `analysis/fk_direction_analyzer.py` | 확률적 방향 분석 |
| Semantic Entity Resolver | `analysis/semantic_entity_resolver.py` | 의미론적 엔티티 해석 |
| Composite FK Detector | `analysis/composite_fk_detector.py` | 복합키 FK 탐지 |
| Hierarchy Detector | `analysis/hierarchy_detector.py` | Self-Reference FK 탐지 |
| Temporal FK Detector | `analysis/temporal_fk_detector.py` | SCD/Bitemporal FK |
| TDA | `analysis/tda.py` | Betti Numbers, Persistence |
| Advanced TDA | `analysis/advanced_tda.py` | Ripser 기반 고급 TDA |
| Schema | `analysis/schema.py` | 컬럼 프로파일링 |
| Similarity | `analysis/similarity.py` | Jaccard, 값 중복 |

---

## 7. Phase Gate 검증

Discovery 완료 시 다음 조건 확인:

```python
Phase1Gate:
    - unified_entities >= 1     # 최소 1개 엔티티 발견
    - evidence_chain not empty  # 증거 기록됨
    - no BLOCKED todos          # 블로킹 태스크 없음
    - completion_signal == [DONE]

    # 추가 검증
    - composite_fks analyzed    # 복합 FK 분석 완료
    - hierarchies analyzed      # 계층 구조 분석 완료
    - temporal_fks analyzed     # Temporal FK 분석 완료
```

실패 시 자동 Fallback 적용 후 Phase 2 진행.

---

*Document Version: v1.0*
*Last Updated: 2026-01-27*
