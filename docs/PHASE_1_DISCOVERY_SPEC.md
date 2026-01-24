# Phase 1: Discovery Engine 기술 명세

> **역할**: 데이터 탐정 (Data Detective)
> **입력**: Raw 데이터 테이블 (CSV, JSON, Parquet, Excel)
> **출력**: 검증된 엔티티 관계 및 구조적 인사이트
> **최종 업데이트**: 2026-01-22
> **버전**: v1

---

## 1. 개요

Discovery Engine은 **도메인 지식 없이(Zero-knowledge)** raw 데이터의 구조적, 의미적 특징을 추출하고 엔티티 간의 숨겨진 연결 고리를 찾아냅니다.

### 핵심 특징
- **LLM-First 분석**: 전체 데이터셋을 LLM에 직접 전달하여 비즈니스 맥락 파악
- **위상수학(TDA) 기반 구조 분석**: Betti Numbers로 데이터 형태 파악
- **Enhanced FK Detection v1**: Multi-Signal + Direction + Semantic (Cross-dataset F1 91.7%)

---

## 2. 에이전트 구성

### 2.1 DataAnalystAutonomousAgent

**역할**: LLM 기반 전체 데이터 분석

```python
DataAnalystAutonomousAgent:
    - 도메인 식별 (supply_chain, healthcare, retail 등)
    - 컬럼 비즈니스 특성 분류:
        - identifier: 고유 식별자
        - required_input: 필수 입력 필드
        - metric: 측정 지표
        - dimension: 분석 차원
        - timestamp: 시간 관련
    - 의미론적 데이터 품질 평가
```

### 2.2 TDAExpertAutonomousAgent

**역할**: 위상 데이터 분석 (Topological Data Analysis)

```python
TDAExpertAutonomousAgent:
    - Betti Numbers 계산 (β0, β1, β2)
    - Persistence Diagram 생성
    - Simplicial Complex 구축
    - 테이블 간 위상적 동형성(Homeomorphism) 판단
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
```

### 2.4 ValueMatcherAutonomousAgent

**역할**: 값 기반 테이블 간 매칭

```python
ValueMatcherAutonomousAgent:
    - Jaccard Similarity 계산
    - Containment Ratio 분석
    - 값 패턴 매칭
    - Cross-table 값 중복 탐지
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
```

### 2.6 RelationshipDetectorAutonomousAgent

**역할**: 테이블 간 관계 탐지

```python
RelationshipDetectorAutonomousAgent:
    - FK 관계 발견 (Enhanced FK Detection v1)
    - 관계 카디널리티 추론 (1:1, 1:N, N:M)
    - 의미론적 관계 분류
    - Join Path 추론
```

---

## 3. 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: Discovery                        │
├─────────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Tables (CSV/JSON)                                       │
│       │                                                      │
│       ├──────────────┬──────────────┬──────────────┐        │
│       ▼              ▼              ▼              ▼        │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │  Data   │   │   TDA   │   │ Schema  │   │  Value  │     │
│  │ Analyst │   │ Expert  │   │ Analyst │   │ Matcher │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │  Entity Classifier    │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │ Relationship Detector │                       │
│              │  (Enhanced FK v1)     │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │    SharedContext      │                       │
│              │  - unified_entities   │                       │
│              │  - homeomorphisms     │                       │
│              │  - evidence_chain     │                       │
│              └───────────────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 출력 데이터

### 4.1 Unified Entity

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

### 4.2 Enhanced FK Candidate (v1)

```json
{
  "from_table": "orders",
  "from_column": "buyer_id",
  "to_table": "customers",
  "to_column": "customer_id",
  "confidence": 0.87,
  "signal_confidence": 0.85,
  "direction_confidence": 0.82,
  "semantic_confidence": 0.90,
  "direction_certain": true,
  "signal_breakdown": {
    "value_overlap": {"score": 0.24, "containment_fk_in_pk": 0.96},
    "cardinality": {"score": 0.18, "fk_unique": 200, "pk_unique": 200},
    "referential_integrity": {"score": 0.23, "orphan_ratio": 0.04},
    "naming": {"score": 0.12, "entity_match": 1.0},
    "structure": {"score": 0.13, "pk_is_unique": true}
  },
  "semantic_analysis": {
    "fk_entity": "buyer",
    "pk_entity": "customer",
    "is_semantically_related": true,
    "relation_type": "synonym:person"
  }
}
```

### 4.3 TDA Signature

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

## 5. 핵심 분석 모듈

| 모듈 | 파일 | 기능 |
|------|------|------|
| **Enhanced FK Pipeline v1** | `analysis/enhanced_fk_pipeline.py` | **Multi-Signal + Direction + Semantic** |
| FK Signal Scorer | `analysis/fk_signal_scorer.py` | 5가지 신호 기반 FK 점수 |
| FK Direction Analyzer | `analysis/fk_direction_analyzer.py` | 확률적 방향 분석 |
| Semantic Entity Resolver | `analysis/semantic_entity_resolver.py` | 의미론적 엔티티 해석 |
| TDA | `analysis/tda.py` | Betti Numbers, Persistence |
| Advanced TDA | `analysis/advanced_tda.py` | Ripser 기반 고급 TDA |
| Schema | `analysis/schema.py` | 컬럼 프로파일링 |
| Similarity | `analysis/similarity.py` | Jaccard, 값 중복 |
| Entity Extractor | `analysis/entity_extractor.py` | 숨겨진 엔티티 발견 |

---

## 5.1 Enhanced FK Detection v1

**파일**: `src/unified_pipeline/autonomous/analysis/enhanced_fk_pipeline.py`

### 알고리즘 개요

```
┌─────────────────────────────────────────────────────────────────┐
│           Enhanced FK Detection Pipeline v1                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Solution 1: Multi-Signal FK Scoring (FKSignalScorer)     │  │
│  │                                                           │  │
│  │  Signal 1: Value Overlap (25%)                           │  │
│  │  ├── containment_fk_in_pk: FK값이 PK에 포함된 비율        │  │
│  │  └── jaccard: 전체 값 집합의 유사도                       │  │
│  │                                                           │  │
│  │  Signal 2: Cardinality Pattern (20%)                     │  │
│  │  ├── FK unique count <= PK unique count                  │  │
│  │  └── ratio가 낮을수록 높은 점수                           │  │
│  │                                                           │  │
│  │  Signal 3: Referential Integrity (25%)                   │  │
│  │  ├── orphan_count: FK에만 있는 값 수                      │  │
│  │  └── orphan_ratio 낮을수록 높은 점수                      │  │
│  │                                                           │  │
│  │  Signal 4: Naming Semantics (15%)                        │  │
│  │  ├── has_fk_suffix: _id, _code, _key 등                  │  │
│  │  ├── entity_match_score: FK 컬럼 ↔ PK 테이블             │  │
│  │  └── name_similarity: 컬럼명 유사도                       │  │
│  │                                                           │  │
│  │  Signal 5: Structural Pattern (15%)                      │  │
│  │  ├── pk_is_unique: PK 컬럼 unique 비율 > 95%             │  │
│  │  └── table_size_ratio: FK rows / PK rows >= 1 (N:1)      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Solution 2: Probabilistic Direction (FKDirectionAnalyzer)│  │
│  │                                                           │  │
│  │  Evidence 1: Uniqueness (strength 0.8)                   │  │
│  │  └── PK는 unique(>95%), FK는 중복 허용(<90%)              │  │
│  │                                                           │  │
│  │  Evidence 2: Containment (strength 0.7)                  │  │
│  │  └── FK ⊆ PK (FK 값이 PK에 포함)                          │  │
│  │                                                           │  │
│  │  Evidence 3: Row Count (strength 0.5)                    │  │
│  │  └── FK 테이블이 보통 더 큼 (N:1 관계)                    │  │
│  │                                                           │  │
│  │  Evidence 4: Column Naming (strength 0.6)                │  │
│  │  └── FK는 _id, _code suffix, PK는 테이블의 primary       │  │
│  │                                                           │  │
│  │  Evidence 5: Entity Reference (strength 0.7)             │  │
│  │  └── buyer_id → customers (synonym 포함)                  │  │
│  │                                                           │  │
│  │  Direction Confidence >= 0.3 이어야 방향 결정             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Solution 3: Semantic Resolution (SemanticEntityResolver) │  │
│  │                                                           │  │
│  │  Universal Synonym Groups:                                │  │
│  │  ├── person: {customer, buyer, user, client, member...}  │  │
│  │  ├── product: {product, item, goods, sku, prod...}       │  │
│  │  ├── order: {order, purchase, transaction, txn, sale}    │  │
│  │  ├── campaign: {campaign, cmp, promotion, marketing}     │  │
│  │  └── ...                                                 │  │
│  │                                                           │  │
│  │  Abbreviation Expansion:                                  │  │
│  │  ├── cust → customer, prod → product                     │  │
│  │  ├── cmp → campaign, emp → employee                      │  │
│  │  ├── txn → transaction, acct → account                   │  │
│  │  └── ...                                                 │  │
│  │                                                           │  │
│  │  Compound Entity Patterns:                                │  │
│  │  ├── attributed_cmp → campaign                           │  │
│  │  ├── ad_cmp → ad_campaign                                │  │
│  │  └── order_ref → order                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Final Confidence Calculation                             │  │
│  │                                                           │  │
│  │  final_confidence = signal_conf × 0.50                   │  │
│  │                   + direction_conf × 0.25                │  │
│  │                   + semantic_conf × 0.25                 │  │
│  │                                                           │  │
│  │  Bonus: 모든 신호 > 0.5 → +0.1                           │  │
│  │  Threshold: final_confidence >= 0.6 → confident FK       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Entity Synonym Groups

```python
# Domain-agnostic synonym groups
UNIVERSAL_SYNONYMS = {
    "person": {"customer", "buyer", "user", "client", "member", "cust", "consumer"},
    "product": {"product", "item", "goods", "sku", "article", "prod"},
    "order": {"order", "purchase", "transaction", "txn", "sale", "booking"},
    "campaign": {"campaign", "cmp", "promotion", "marketing"},
    "ad_campaign": {"ad_campaign", "ad_campaigns", "ad_cmp"},
    "organization": {"company", "org", "vendor", "supplier", "merchant"},
    "location": {"location", "loc", "address", "place", "site", "region"},
    "account": {"account", "acct", "profile", "registration"},
}
```

### 검증 결과

| 데이터셋 | Precision | Recall | F1 Score | 비고 |
|---------|-----------|--------|----------|------|
| **marketing_silo** | 100.0% | 100.0% | 100.0% | 12/12 관계 검출 |
| **airport_silo** | 90.9% | 76.9% | 83.3% | 10/13 관계 검출 |
| **Cross-dataset Avg** | **95.5%** | **88.5%** | **91.7%** | |

### 미검출 관계 분석 (airport_silo)

```
1. carrier → airlines (carrier ≠ airline 의미 연결 필요)
2. assigned_airline → airlines (assigned_airline ≠ airline)
3. flights.aircraft_reg → aircraft_registry (테이블명 차이)
```

---

## 6. Phase Gate 검증

Discovery 완료 시 다음 조건 확인:

```python
Phase1Gate:
    - unified_entities >= 1  # 최소 1개 엔티티 발견
    - evidence_chain not empty  # 증거 기록됨
    - no BLOCKED todos  # 블로킹 태스크 없음
    - completion_signal == [DONE]  # 명시적 완료 신호
```

실패 시 자동 Fallback 적용 후 Phase 2 진행.
