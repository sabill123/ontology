# Ontoloty 알고리즘 및 수학적 기반

> **버전**: v23.0
> **최종 업데이트**: 2026-02-09

---

## 1. 개요

Ontoloty 시스템은 **LLM의 범용 추론(Reasoning)**과 **엄밀한 수학적 알고리즘(Rigor)**을 결합한 하이브리드 인텔리전스 시스템입니다.

### 핵심 원칙
1. **Algorithmic Evidence**: 통계/수학적 증거로 주장 뒷받침
2. **LLM Reasoning**: 비즈니스 맥락 이해 및 자연어 해석
3. **Multi-Agent Consensus**: 다양한 관점의 검증 및 합의
4. **Semantic Validation**: OWL2/SHACL 기반 의미론적 검증

---

## 2. 위상 데이터 분석 (TDA)

### 2.1 이론적 배경

**Persistent Homology**: 데이터 간 연결성을 변화시키며 "구멍(Hole)"의 발생과 소멸을 추적합니다.

**Betti Numbers**:
- β₀: 연결 성분 수 (Connected Components)
- β₁: 1차원 구멍/사이클 (Loops) - 순환 참조 탐지
- β₂: 2차원 공동 (Voids)

### 2.2 프로세스

```
Schema/Data → Point Cloud → Rips Filtration → Simplicial Complex
     ↓
Boundary Matrix Reduction → Persistent Homology → Betti Numbers
     ↓
Homeomorphism Detection (위상적 동형성 판단)
```

### 2.3 활용

- **테이블 구조 비교**: 이름이 달라도 구조가 같으면 같은 엔티티로 간주
- **순환 참조 탐지**: β₁ > 0 이면 순환 의존성 존재
- **데이터 복잡도 평가**: Betti numbers 합으로 복잡도 측정

### 2.4 구현

```python
# src/unified_pipeline/autonomous/analysis/tda.py
# src/unified_pipeline/autonomous/analysis/advanced_tda.py

class TDAAnalyzer:
    def compute_betti_numbers(data) -> BettiNumbers
    def compute_persistence_diagram(data) -> PersistenceDiagram
    def is_homeomorphic(profile_a, profile_b) -> bool
```

### 2.5 v1.0 통합

```python
# TDA 결과 → Enhanced FK Pipeline 연동
tda_result = tda_analyzer.analyze(tables)

# Hierarchy Detector에 TDA 구조 정보 전달
hierarchy_detector.detect(
    tables,
    tda_structure=tda_result.topological_structure
)
```

---

## 3. Enhanced FK Detection v1.0

### 3.1 설계 원칙

v1.0은 기존 v1의 Multi-Signal 기반에 **Composite FK**, **Hierarchy**, **Temporal FK** 탐지를 추가했습니다.

| 원칙 | 설명 |
|------|------|
| **Multi-Signal** | 단일 지표 대신 5가지 독립 신호 종합 |
| **Evidence-Based Direction** | 단일 룰 대신 여러 증거로 방향 결정 |
| **Semantic Resolution** | Domain-agnostic synonym groups |
| **Composite FK (v1.0)** | 복합 키 (plant, product) → table 탐지 |
| **Hierarchy (v1.0)** | 계층 구조 plant → warehouse 탐지 |
| **Temporal FK (v1.0)** | 시간 범위 기반 FK 탐지 |

### 3.2 Multi-Signal FK Scoring (FKSignalScorer)

단일 지표(Jaccard/Containment)의 한계를 극복하기 위해 5가지 독립 신호를 종합합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Multi-Signal FK Scoring System v1.0                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Signal_Total = Σ Signal_i × Weight_i                           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Signal 1: Value Overlap (Weight = 0.25)                 │    │
│  │                                                         │    │
│  │ score = containment_fk_in_pk × 0.25                    │    │
│  │                                                         │    │
│  │ containment_fk_in_pk = |FK ∩ PK| / |FK|                │    │
│  │                                                         │    │
│  │ - FK 값이 PK에 포함되어야 함                            │    │
│  │ - 1.0에 가까울수록 좋음 (완벽한 참조 무결성)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Signal 2: Cardinality Pattern (Weight = 0.20)           │    │
│  │                                                         │    │
│  │ ratio = |FK_unique| / |PK_unique|                      │    │
│  │                                                         │    │
│  │ if ratio ≤ 1.0:                                        │    │
│  │     score = 0.20 × (1.0 - ratio × 0.5)                 │    │
│  │ else:                                                   │    │
│  │     score = 0.20 × max(0, 1.0 - (ratio - 1.0))         │    │
│  │                                                         │    │
│  │ - FK unique count ≤ PK unique count가 일반적           │    │
│  │ - ratio가 낮을수록 높은 점수                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Signal 3: Referential Integrity (Weight = 0.25)         │    │
│  │                                                         │    │
│  │ orphan_count = |FK - PK|  (FK에만 있는 값)             │    │
│  │ orphan_ratio = orphan_count / |FK|                     │    │
│  │                                                         │    │
│  │ score = (1.0 - orphan_ratio) × 0.25                    │    │
│  │                                                         │    │
│  │ - Orphan이 적을수록 참조 무결성 높음                    │    │
│  │ - orphan_ratio = 0 → 완벽한 무결성                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Signal 4: Naming Semantics (Weight = 0.15)              │    │
│  │                                                         │    │
│  │ score = fk_suffix_score + entity_match_score            │    │
│  │       + name_similarity_score                           │    │
│  │                                                         │    │
│  │ fk_suffix_score = 0.05 if has_fk_suffix else 0.0       │    │
│  │   - FK suffixes: _id, _code, _key, _no, _ref, _fk, _num│    │
│  │                                                         │    │
│  │ entity_match_score = entity_match × 0.05               │    │
│  │   - FK 컬럼명에서 추출한 entity와 PK 테이블 일치도      │    │
│  │                                                         │    │
│  │ name_similarity_score = name_sim × 0.05                │    │
│  │   - 동일: 1.0, 테이블명_id: 0.8, 토큰 공유: 0.5        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Signal 5: Structural Pattern (Weight = 0.15)            │    │
│  │                                                         │    │
│  │ pk_unique_ratio = |PK_unique| / |PK_rows|              │    │
│  │ pk_is_unique = pk_unique_ratio > 0.95                  │    │
│  │                                                         │    │
│  │ score = 0.0                                            │    │
│  │ if pk_is_unique: score += 0.10                         │    │
│  │ if FK_rows >= PK_rows: score += 0.05  (N:1 패턴)       │    │
│  │                                                         │    │
│  │ - PK가 실제로 unique해야 함                             │    │
│  │ - FK 테이블이 더 크면 N:1 관계 가능성 높음              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Total Score = min(1.0, Σ all signal scores)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Probabilistic Direction Analysis (FKDirectionAnalyzer)

단일 룰 대신 여러 증거를 종합하여 FK 방향을 결정합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Probabilistic FK Direction Analysis v1.0               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Direction = argmax(score_A→B, score_B→A)                       │
│  Confidence = |score_A→B - score_B→A| / (score_A→B + score_B→A) │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Evidence 1: Uniqueness (strength = 0.8)                 │    │
│  │                                                         │    │
│  │ if A_unique > 0.95 and B_unique < 0.90:                │    │
│  │     direction = "B→A", strength = 0.8                  │    │
│  │     reason = "A is unique, B has duplicates"           │    │
│  │                                                         │    │
│  │ - PK는 unique(>95%), FK는 중복 허용(<90%)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Evidence 2: Containment (strength = 0.7)                │    │
│  │                                                         │    │
│  │ a_in_b = |A ∩ B| / |A|                                 │    │
│  │ b_in_a = |A ∩ B| / |B|                                 │    │
│  │                                                         │    │
│  │ if a_in_b > 0.9 and b_in_a < 0.7:                      │    │
│  │     direction = "A→B", strength = 0.7                  │    │
│  │     reason = "A values contained in B"                 │    │
│  │                                                         │    │
│  │ - FK ⊆ PK (FK 값이 PK에 포함)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Evidence 3: Row Count (strength = 0.5)                  │    │
│  │                                                         │    │
│  │ if A_rows > B_rows × 2:                                │    │
│  │     direction = "A→B", strength = 0.5                  │    │
│  │     reason = "A has more rows (N:1 pattern)"           │    │
│  │                                                         │    │
│  │ - FK 테이블이 보통 더 큼 (N:1 관계)                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Evidence 4: Column Naming (strength = 0.6)              │    │
│  │                                                         │    │
│  │ a_has_fk = A ends with _id, _code, _key, etc.          │    │
│  │ b_is_pk = B matches table_id or id pattern             │    │
│  │                                                         │    │
│  │ if a_has_fk and b_is_pk:                               │    │
│  │     direction = "A→B", strength = 0.6                  │    │
│  │                                                         │    │
│  │ - FK는 보통 _id, _code suffix                           │    │
│  │ - PK는 테이블의 primary column (id, table_id)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Evidence 5: Entity Reference (strength = 0.7)           │    │
│  │                                                         │    │
│  │ a_refs_b = column_entity matches table_b (via synonym) │    │
│  │                                                         │    │
│  │ if a_refs_b and not b_refs_a:                          │    │
│  │     direction = "A→B", strength = 0.7                  │    │
│  │     reason = "Column A references table B"             │    │
│  │                                                         │    │
│  │ - buyer_id → customers (buyer ∈ person synonym group)   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Aggregation:                                                    │
│  - score_A→B = Σ strength (for evidences with "A→B")            │
│  - score_B→A = Σ strength (for evidences with "B→A")            │
│  - if confidence < 0.3: return "uncertain" (LLM에 위임)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Semantic Entity Resolution (SemanticEntityResolver)

Domain-agnostic한 방식으로 컬럼명/테이블명에서 entity를 추출하고 관계를 추론합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Semantic Entity Resolution v1.0                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Universal Synonym Groups                                 │    │
│  │                                                         │    │
│  │ person: {customer, buyer, user, client, member,         │    │
│  │          cust, consumer, subscriber, owner, creator}    │    │
│  │                                                         │    │
│  │ product: {product, item, goods, sku, article, prod,     │    │
│  │           merchandise, commodity}                       │    │
│  │                                                         │    │
│  │ order: {order, purchase, transaction, txn, sale,        │    │
│  │         booking}                                        │    │
│  │                                                         │    │
│  │ organization: {company, org, vendor, supplier,          │    │
│  │                merchant, business, firm, enterprise}    │    │
│  │                                                         │    │
│  │ campaign: {campaign, cmp, promotion, marketing,         │    │
│  │            advertisement, promo}                        │    │
│  │                                                         │    │
│  │ ad_campaign: {ad_campaign, ad_campaigns, ad_cmp,        │    │
│  │               advertising_campaign}                     │    │
│  │                                                         │    │
│  │ account: {account, acct, profile, registration}         │    │
│  │                                                         │    │
│  │ location: {location, loc, address, place, site,         │    │
│  │            region, area, zone, territory}               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Abbreviation Expansion                                   │    │
│  │                                                         │    │
│  │ cust → customer    prod → product    emp → employee     │    │
│  │ cmp → campaign     txn → transaction acct → account     │    │
│  │ org → organization loc → location    dept → department  │    │
│  │ no → number        num → number      ref → reference    │    │
│  │ conv → conversion  perf → performance ad → advertisement│    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Compound Entity Patterns                                 │    │
│  │                                                         │    │
│  │ attributed_cmp → campaign                               │    │
│  │ attributed_ad → ad_campaign                             │    │
│  │ marketing_campaign → campaign                           │    │
│  │ ad_cmp → ad_campaign                                    │    │
│  │ order_ref → order                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Relation Types                                           │    │
│  │                                                         │    │
│  │ identical:       entity_a == entity_b (conf: 1.0)       │    │
│  │ plural_singular: customers == customer (conf: 0.95)     │    │
│  │ synonym:         buyer ∈ person, customer ∈ person      │    │
│  │                  (conf: 0.9)                            │    │
│  │ partial_match:   "customer" in "customer_id" (conf: 0.7)│    │
│  │ fuzzy_match:     Jaccard similarity > 0.8 (conf: 0.64)  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 v1.0 Composite FK Detection

복합 키를 탐지하여 다중 컬럼 FK 관계를 식별합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Composite FK Detection v1.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  입력: tables = {products_per_plant, plants, products}          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Step 1: Candidate Generation                             │    │
│  │                                                         │    │
│  │ products_per_plant:                                     │    │
│  │   - plant_code (potential FK to plants)                 │    │
│  │   - product_id (potential FK to products)               │    │
│  │                                                         │    │
│  │ → Composite candidate: (plant_code, product_id)         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Step 2: Joint Uniqueness Check                           │    │
│  │                                                         │    │
│  │ uniqueness = |unique(plant_code, product_id)| / |rows|  │    │
│  │                                                         │    │
│  │ if uniqueness > 0.95:                                   │    │
│  │     composite_pk = True                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Step 3: Referential Integrity Verification               │    │
│  │                                                         │    │
│  │ for each (plant, product) in products_per_plant:        │    │
│  │     exists_plant = plant in plants.plant_code           │    │
│  │     exists_product = product in products.product_id     │    │
│  │                                                         │    │
│  │ integrity = count(both_exist) / total_rows              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  출력: CompositeFKCandidate(                                     │
│      fk_columns=["plant_code", "product_id"],                   │
│      pk_tables=["plants", "products"],                          │
│      confidence=0.92                                            │
│  )                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.6 v1.0 Hierarchy Detection

엔티티 간 계층 구조(plant → warehouse → region)를 탐지합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Hierarchy Detection v1.0                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Detection Signals                                        │    │
│  │                                                         │    │
│  │ 1. Cardinality Ratio:                                   │    │
│  │    |child| >> |parent|  (many-to-one)                   │    │
│  │                                                         │    │
│  │ 2. Value Containment:                                   │    │
│  │    child.parent_ref ⊆ parent.id                         │    │
│  │                                                         │    │
│  │ 3. Naming Pattern:                                      │    │
│  │    parent_id, parent_code → hierarchy indicator         │    │
│  │                                                         │    │
│  │ 4. Transitive Closure:                                  │    │
│  │    if A → B and B → C, then A →→ C (hierarchy)         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Hierarchy Types                                          │    │
│  │                                                         │    │
│  │ PART_OF:     component → assembly → product             │    │
│  │ LOCATED_IN:  store → city → region → country           │    │
│  │ REPORTS_TO:  employee → manager → director → CEO       │    │
│  │ BELONGS_TO:  product → category → department           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  출력: HierarchyResult(                                          │
│      type="LOCATED_IN",                                         │
│      levels=["store", "city", "region", "country"],             │
│      confidence=0.88                                            │
│  )                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.7 v1.0 Temporal FK Detection

시간 범위 기반 FK 관계를 탐지합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Temporal FK Detection v1.0                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Temporal Patterns                                        │    │
│  │                                                         │    │
│  │ 1. Point-in-Time Reference:                             │    │
│  │    order.created_at → price_history.effective_date      │    │
│  │    (가장 가까운 과거 시점의 가격 참조)                    │    │
│  │                                                         │    │
│  │ 2. Range Overlap:                                       │    │
│  │    booking.start_date - booking.end_date                │    │
│  │    ∩ availability.valid_from - availability.valid_to    │    │
│  │                                                         │    │
│  │ 3. Effective Dating:                                    │    │
│  │    employee.department_id WHERE                         │    │
│  │    now() BETWEEN effective_start AND effective_end      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Detection Algorithm                                      │    │
│  │                                                         │    │
│  │ 1. Identify date/time columns in both tables           │    │
│  │ 2. Check for temporal column naming patterns:          │    │
│  │    - effective_date, valid_from, valid_to              │    │
│  │    - start_date, end_date, created_at                  │    │
│  │ 3. Analyze value distribution overlap                  │    │
│  │ 4. Verify temporal referential integrity               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  출력: TemporalFKCandidate(                                      │
│      fk_column="order_date",                                    │
│      pk_table="price_history",                                  │
│      temporal_type="point_in_time",                             │
│      confidence=0.85                                            │
│  )                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.8 Final Confidence Calculation

세 가지 분석 결과를 종합하여 최종 confidence를 계산합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Final Confidence Calculation v1.0                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  final_confidence = signal_conf × 0.50                          │
│                   + direction_conf × 0.25                       │
│                   + semantic_conf × 0.25                        │
│                                                                  │
│  v1.0 Bonus:                                                    │
│  if composite_fk_detected: final_confidence += 0.05            │
│  if hierarchy_detected: final_confidence += 0.05               │
│  if temporal_fk_detected: final_confidence += 0.05             │
│                                                                  │
│  Thresholds:                                                     │
│  - min_confidence = 0.4 (조기 제외)                              │
│  - confident_threshold = 0.6 (confident FK)                     │
│                                                                  │
│  Bidirectional Resolution:                                       │
│  - 양방향 후보가 있으면 direction_certain인 것 우선              │
│  - 모두 uncertain이면 final_confidence로 결정                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.9 검증 결과

```
┌───────────────────┬───────────┬──────────┬──────────┐
│ Dataset           │ Precision │ Recall   │ F1 Score │
├───────────────────┼───────────┼──────────┼──────────┤
│ marketing_silo    │ 100.0%    │ 100.0%   │ 100.0%   │
│ airport_silo      │ 90.9%     │ 76.9%    │ 83.3%    │
├───────────────────┼───────────┼──────────┼──────────┤
│ Cross-dataset Avg │ 95.5%     │ 88.5%    │ 91.7%    │
└───────────────────┴───────────┴──────────┴──────────┘
```

---

## 4. 유사도 분석

### 4.1 Jaccard Similarity

두 집합의 유사도 측정:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

- 0.0: 완전히 다름
- 1.0: 완전히 동일

### 4.2 Containment Ratio

FK 관계 탐지에 사용:

```
C(A, B) = |A ∩ B| / |A|
```

- C(FK, PK) ≈ 1.0: FK의 모든 값이 PK에 존재 (참조 무결성)
- Parent(PK)는 Unique해야 함 (U > 0.95)
- Child(FK)는 중복 허용 (U < 0.95)

### 4.3 Pearson Correlation

Cross-Entity 상관관계 분석:

```
r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

- |r| > 0.3: 의미 있는 상관관계
- p-value < 0.05: 통계적 유의성

### 4.4 구현

```python
# src/unified_pipeline/autonomous/analysis/similarity.py
# src/unified_pipeline/autonomous/analysis/advanced_similarity.py

class SimilarityAnalyzer:
    def compute_jaccard(set_a, set_b) -> float
    def compute_containment(set_a, set_b) -> float

class AdvancedSimilarityAnalyzer:
    def minhash_similarity(set_a, set_b) -> float  # 근사 Jaccard
    def lsh_search(query, candidates) -> List[str]  # 근사 최근접 이웃
```

---

## 5. Cross-Entity Correlation 분석

### 5.1 분석 유형

| 유형 | 탐지 방법 | 비즈니스 의미 |
|------|----------|--------------|
| 상관관계 | Pearson r > 0.3 | 변수 간 연관성 |
| 보완 관계 | sum = 1.0 | 중복 컬럼 (신규 + 재방문 = 1) |
| 동일 데이터 | correlation = 1.0 | 데이터 수집 오류 |
| 임계값 효과 | 비선형 구간 탐지 | 비선형 인과관계 |

### 5.2 임계값 효과 탐지

특정 값을 기준으로 급격한 변화가 발생하는 지점 탐지:

```python
def detect_threshold_effects(x, y):
    # 1. 데이터를 구간별로 분할
    # 2. 각 구간에서 y의 평균 계산
    # 3. 구간 간 변화율 계산
    # 4. 변화율이 급격한 지점 = 임계값
```

**출력 예시**:
> "창고 용량이 31.5 임계값 초과 시 운송 중량 5544% 증가"

### 5.3 구현

```python
# src/unified_pipeline/autonomous/analysis/cross_entity_correlation.py

class CrossEntityCorrelationAnalyzer:
    def compute_cross_table_correlations(table_names, fk_relations=None) -> List[CorrelationResult]
    def _detect_mathematical_constraints(df, table_name) -> List[Dict]  # 보완관계
    def _detect_duplicate_columns(df, table_name) -> List[Dict]         # 동일데이터
    def _analyze_categorical_insights(df, table_name) -> List[Dict]     # 세그먼트
    def _filter_invalid_correlations(correlations) -> List[CorrelationResult]
    def _try_join_tables(table_a, df_a, table_b, df_b, fk_lookup) -> DataFrame  # FK join
    async def analyze(table_names, ...) -> List[PalantirInsight]        # async
    def analyze_sync(table_names, ...) -> List[PalantirInsight]         # sync
```

---

## 6. 증거 이론 (Dempster-Shafer)

### 6.1 이론적 배경

불확실한 상황에서 여러 소스의 증거를 수학적으로 결합:

- **Belief Function (Bel)**: 증거가 지지하는 하한 확률
- **Plausibility (Pl)**: 증거가 반박하지 않는 상한 확률

### 6.2 Dempster's Rule of Combination

독립적인 두 증거 m₁, m₂를 결합:

```
(m₁ ⊕ m₂)(A) = (1/(1-K)) × Σ m₁(B) × m₂(C)
                           B∩C=A

K = Σ m₁(B) × m₂(C)  (B∩C = ∅인 경우, 충돌하는 증거)
    B∩C=∅
```

- K가 1에 가까우면: 증거들이 심하게 충돌 → 판단 보류
- K가 0에 가까우면: 증거들이 일관성 있음 → 높은 신뢰도

### 6.3 활용

```python
# src/unified_pipeline/autonomous/analysis/enhanced_validator.py

class EnhancedValidator:
    def dempster_shafer_fusion(agent_opinions) -> DSResult:
        # 1. 각 에이전트 의견을 mass function으로 변환
        # 2. Dempster's rule로 순차적 결합
        # 3. 충돌도(K) 계산
        # 4. 최종 신뢰도 산출
```

---

## 7. BFT 합의 알고리즘

### 7.1 Byzantine Fault Tolerance

악의적이거나 오류가 있는 에이전트가 있어도 올바른 합의에 도달:

```
필요 조건: n ≥ 3f + 1
(n = 전체 에이전트, f = 장애 에이전트)
```

### 7.2 합의 프로세스

```
Round 1: Initial Vote
    - 각 에이전트가 approve/reject/review 투표
    - 만장일치 또는 >75% 동의 시 Fast-track

Round 2: Deep Debate (충돌 시)
    - 소수 의견 에이전트가 Challenge
    - 알고리즘 증거 제시
    - 재투표

Round 3+: Forced Decision
    - Max Rounds 도달 시 Soft Consensus
    - 신뢰도 하향 조정 후 결정
```

### 7.3 v1.0 AgentBus 통합

```python
# src/unified_pipeline/autonomous/consensus/engine.py
# src/unified_pipeline/autonomous/agent_bus.py

class ConsensusEngine:
    def __init__(self, agent_bus: AgentCommunicationBus):
        self.bus = agent_bus

    def run_consensus(concept, agents) -> ConsensusResult:
        # 1. AgentBus로 투표 요청 브로드캐스트
        self.bus.broadcast(Message(type=VOTE_REQUEST, data=concept))

        # 2. Pub/Sub으로 투표 수집
        votes = await self.bus.collect_responses(timeout=30)

        # 3. BFT 검증 (Byzantine 에이전트 탐지)
        valid_votes = self.validate_bft(votes)

        # 4. DS Fusion으로 신뢰도 결합
        result = self.dempster_shafer_fusion(valid_votes)

        # 5. 결과 publish
        self.bus.publish("consensus.result", result)
        return result
```

---

## 8. 그래프 분석

### 8.1 Node2Vec Embedding

그래프 구조를 벡터 공간으로 변환:

```
1. Random Walk: 그래프에서 임의 경로 생성
2. Skip-Gram: Word2Vec과 동일하게 벡터 학습
3. Result: 관계가 밀접한 노드는 공간상 가깝게 위치
```

**활용**:
- 엔티티 유사도 계산: Vector(Customer) ≈ Vector(Order)
- 커뮤니티 탐지: 밀접한 엔티티 그룹 식별

### 8.2 구현

```python
# src/unified_pipeline/autonomous/analysis/graph_embedding.py

class GraphEmbedding:
    def node2vec_embed(graph) -> Dict[str, np.ndarray]
    def detect_communities(embeddings) -> List[Community]
    def compute_similarity(node_a, node_b) -> float
```

---

## 9. 시계열 분석

### 9.1 계절성 탐지

FFT(Fast Fourier Transform) 또는 자기상관 분석으로 주기성 탐지:

```python
def detect_seasonality(series):
    # 1. 자기상관 함수(ACF) 계산
    # 2. 피크 지점 탐지
    # 3. 주기 및 강도 추출
```

**출력 예시**:
> "TPT_Day_Count에서 14일 주기 계절성 탐지 (57% 강도)"

### 9.2 구현

```python
# src/unified_pipeline/autonomous/analysis/time_series_forecaster.py

class TimeSeriesForecaster:
    def detect_seasonality(series) -> SeasonalityResult
    def forecast_arima(series, horizon) -> Forecast
```

---

## 10. 데이터 품질 분석

### 10.1 이상치 탐지

```python
def detect_outliers(series):
    # IQR 방법
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # 또는 Z-score 방법
    z_scores = (series - series.mean()) / series.std()
    outliers = abs(z_scores) > 3
```

### 10.2 결측치 분석

```python
def analyze_missing(df):
    # 1. 컬럼별 결측률 계산
    # 2. 결측 패턴 분석 (MCAR, MAR, MNAR)
    # 3. 비즈니스 영향도 평가
```

---

## 11. OWL2 Reasoning (v1.0)

### 11.1 이론적 배경

**OWL2 (Web Ontology Language 2)**: W3C 표준 온톨로지 언어로, 클래스 계층, 속성 관계, 제약조건을 정의합니다.

### 11.2 추론 유형

```
┌─────────────────────────────────────────────────────────────────┐
│          OWL2 Reasoning Types v1.0                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Class Subsumption (클래스 포함 관계)                         │
│     Customer ⊑ Person                                           │
│     → Customer의 모든 인스턴스는 Person                          │
│                                                                  │
│  2. Property Domain/Range (속성 정의역/치역)                     │
│     hasOrder: Customer → Order                                  │
│     → Customer만 Order를 가질 수 있음                            │
│                                                                  │
│  3. Inverse Properties (역속성)                                  │
│     hasOrder ≡ isOrderOf⁻¹                                      │
│     → 양방향 관계 추론                                           │
│                                                                  │
│  4. Transitive Properties (추이적 속성)                          │
│     locatedIn(a, b) ∧ locatedIn(b, c) → locatedIn(a, c)        │
│     → 계층 구조 추론                                             │
│                                                                  │
│  5. Disjoint Classes (배타적 클래스)                             │
│     Customer ⊓ Product = ∅                                      │
│     → 동일 인스턴스가 두 클래스에 속할 수 없음                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.3 구현

```python
# src/unified_pipeline/autonomous/analysis/owl2_reasoner.py

class OWL2Reasoner:
    def infer_class_hierarchy(entities) -> ClassHierarchy
    def infer_property_constraints(relations) -> PropertyConstraints
    def check_consistency(ontology) -> ConsistencyResult
    def materialize_inferences(ontology) -> MaterializedOntology
```

---

## 12. SHACL Validation (v1.0)

### 12.1 이론적 배경

**SHACL (Shapes Constraint Language)**: W3C 표준으로, RDF 그래프의 형태(Shape)를 정의하고 검증합니다.

### 12.2 제약조건 유형

```
┌─────────────────────────────────────────────────────────────────┐
│          SHACL Constraint Types v1.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Cardinality Constraints (기수 제약)                          │
│     sh:minCount 1  → 최소 1개 필수                              │
│     sh:maxCount 1  → 최대 1개만 허용                            │
│                                                                  │
│  2. Value Type Constraints (값 타입 제약)                        │
│     sh:datatype xsd:integer                                     │
│     sh:class :Customer                                          │
│                                                                  │
│  3. String Constraints (문자열 제약)                             │
│     sh:pattern "^[A-Z]{3}$"  → 정규식 패턴                      │
│     sh:minLength 5                                              │
│     sh:maxLength 100                                            │
│                                                                  │
│  4. Numeric Constraints (숫자 제약)                              │
│     sh:minInclusive 0                                           │
│     sh:maxExclusive 1000000                                     │
│                                                                  │
│  5. Logical Constraints (논리 제약)                              │
│     sh:or, sh:and, sh:not, sh:xone                             │
│                                                                  │
│  6. Property Path Constraints (속성 경로 제약)                   │
│     sh:path (hasOrder / hasProduct)  → 중첩 경로               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 구현

```python
# src/unified_pipeline/autonomous/analysis/shacl_validator.py

class SHACLValidator:
    def generate_shapes(entities, relations) -> SHACLShapes
    def validate(data_graph, shapes_graph) -> ValidationReport
    def infer_constraints_from_data(tables) -> InferredShapes
```

---

## 13. Column-level Lineage (v1.0)

### 13.1 이론적 배경

**Data Lineage**: 데이터의 출처, 변환, 이동 경로를 추적하여 데이터 품질과 거버넌스를 보장합니다.

### 13.2 Lineage 그래프 구조

```
┌─────────────────────────────────────────────────────────────────┐
│          Column Lineage Graph v1.0                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Node Types:                                                     │
│  - SOURCE: 원본 테이블 컬럼                                      │
│  - DERIVED: 계산/변환된 컬럼                                     │
│  - AGGREGATED: 집계된 컬럼                                       │
│  - JOINED: 조인으로 생성된 컬럼                                  │
│                                                                  │
│  Edge Types:                                                     │
│  - DIRECT: 직접 복사 (A → B)                                    │
│  - TRANSFORM: 변환 (f(A) → B)                                   │
│  - AGGREGATE: 집계 (SUM(A) → B)                                 │
│  - FK_REFERENCE: FK 참조 (A.fk → B.pk)                         │
│                                                                  │
│  Example:                                                        │
│  orders.customer_id ─FK_REF→ customers.id                       │
│       │                          │                               │
│       └─DERIVE→ report.customer_name ←─TRANSFORM                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 13.3 Impact Analysis

```python
def analyze_impact(column: str) -> ImpactAnalysis:
    """
    특정 컬럼 변경 시 영향 받는 모든 downstream 컬럼 분석

    Returns:
        - affected_columns: 영향 받는 컬럼 목록
        - affected_reports: 영향 받는 리포트
        - risk_score: 변경 위험도 (0-1)
    """
```

### 13.4 구현

```python
# src/unified_pipeline/lineage/column_lineage.py

class LineageTracker:
    def build_lineage_graph(context) -> LineageGraph
    def trace_upstream(column) -> List[ColumnNode]
    def trace_downstream(column) -> List[ColumnNode]
    def analyze_impact(column) -> ImpactAnalysis
    def export_to_openlineage() -> OpenLineageFormat
```

---

## 14. Agent Communication Bus (v1.0)

### 14.1 설계 원칙

에이전트 간 직접 통신을 가능하게 하는 메시지 버스 시스템입니다.

### 14.2 통신 패턴

```
┌─────────────────────────────────────────────────────────────────┐
│          Agent Communication Patterns v1.0                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Direct Messaging (1:1)                                       │
│     bus.send_to("schema_analyst", message)                      │
│     → 특정 에이전트에게 직접 메시지 전송                          │
│                                                                  │
│  2. Broadcast (1:N)                                              │
│     bus.broadcast(message)                                      │
│     → 모든 에이전트에게 메시지 전송                               │
│                                                                  │
│  3. Pub/Sub (Topic-based)                                        │
│     bus.subscribe("fk.detected", callback)                      │
│     bus.publish("fk.detected", data)                            │
│     → 토픽 기반 구독/발행                                        │
│                                                                  │
│  4. Request/Response (RPC-style)                                 │
│     response = await bus.request("validator", req)              │
│     → 동기식 요청/응답 패턴                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 14.3 Message Types

```python
class MessageType(Enum):
    INFO = "info"                    # 정보 공유
    DISCOVERY = "discovery"          # 발견 결과
    ALERT = "alert"                  # 경고/알림
    REQUEST = "request"              # 요청
    RESPONSE = "response"            # 응답
    COLLABORATION = "collaboration"  # 협업 요청
    VALIDATION = "validation"        # 검증 요청
    FEEDBACK = "feedback"            # 피드백
    HEARTBEAT = "heartbeat"          # 상태 확인
    STATUS = "status"                # 상태 보고
```

### 14.4 구현

```python
# src/unified_pipeline/autonomous/agent_bus.py

class AgentCommunicationBus:
    def register_agent(agent_id, handler) -> None
    def send_to(target, message) -> None
    def broadcast(message) -> None
    def subscribe(topic, callback) -> Subscription
    def publish(topic, data) -> None
    def request(target, request, timeout) -> Response
```

---

## 15. 인과 추론 (Causal Inference) — v23.0

### 15.1 Granger 인과성 분석

시계열 데이터에서 변수 X가 변수 Y의 예측에 도움이 되는지 통계적으로 검정합니다.

```
H₀: X가 Y를 Granger-cause 하지 않음
H₁: X의 과거 값이 Y 예측에 유의미한 기여

검정:
  F = ((RSS_restricted - RSS_unrestricted) / p) / (RSS_unrestricted / (n - 2p - 1))

  restricted:   Y_t = α + Σ β_i Y_{t-i}
  unrestricted: Y_t = α + Σ β_i Y_{t-i} + Σ γ_i X_{t-i}
```

### 15.2 ATE / CATE 추정

**ATE (Average Treatment Effect)**:
```
ATE = E[Y(1)] - E[Y(0)]
    ≈ (1/N) Σ (outcome_treated - outcome_control)
```

**CATE (Conditional Average Treatment Effect)** — 3가지 추정기:

| 추정기 | 방식 | 장점 |
|--------|------|------|
| S-Learner | 단일 모델에 treatment 변수 포함 | 단순, 데이터 효율적 |
| T-Learner | treatment/control 각각 별도 모델 | 이질적 효과 포착 |
| X-Learner | T-Learner + 교차 추정 + 가중 결합 | 비대칭 그룹 크기에 강건 |

### 15.3 Propensity Score Matching

```
e(x) = P(T=1 | X=x)  — 처리 확률 추정

Matching: 각 treated unit에 대해
  nearest_control = argmin |e(x_treated) - e(x_control)|

ATE_matched = (1/N) Σ (Y_treated - Y_matched_control)
```

### 15.4 Simpson's Paradox 탐지

```
sign(ATE_naive) ≠ sign(ATE_adjusted)  →  Simpson's Paradox 경고

탐지 방식:
  1. ATE_naive = E[Y|T=1] - E[Y|T=0]  (전체)
  2. ATE_adjusted = Σ_s P(S=s) × (E[Y|T=1,S=s] - E[Y|T=0,S=s])  (층화)
  3. 부호 반전 시 교란변수 경고 발생
```

### 15.5 구현

```python
# src/unified_pipeline/autonomous/analysis/causal_impact.py (125KB)

class CausalImpactAnalyzer:
    def compute_ate(treatment, outcome) -> ATEResult
    def compute_cate(treatment, outcome, moderators) -> CATEResult
    def granger_causality_test(x, y, max_lag) -> GrangerResult
    def propensity_score_matching(data, treatment) -> MatchedResult

class CounterfactualAnalyzer:
    def analyze_counterfactual(scenario) -> CounterfactualResult

class SensitivityAnalyzer:
    def rosenbaum_bounds(data) -> SensitivityResult
```

---

## 16. 통계적 가설 검정 (Statistical Hypothesis Testing) — v23.0

### 16.1 분포 비교

| 검정 | 용도 | 귀무가설 |
|------|------|---------|
| Kolmogorov-Smirnov (KS) | 두 분포 동일성 | F₁(x) = F₂(x) |
| Anderson-Darling (AD) | 정규성 검정 | 데이터가 정규분포를 따름 |

**KS 통계량**:
```
D = sup_x |F₁(x) - F₂(x)|
p-value < α → 두 분포가 유의미하게 다름
```

### 16.2 평균/분산 비교

| 검정 | 용도 | 특징 |
|------|------|------|
| Welch's t-test | 등분산 가정 없는 평균 비교 | 자유도 Satterthwaite 근사 |
| Levene's Test | 분산 동질성 검정 | 비정규 분포에도 강건 |

### 16.3 구현

```python
# src/unified_pipeline/autonomous/analysis/enhanced_validator.py

class EnhancedValidator:
    def ks_test(sample_a, sample_b) -> TestResult
    def welch_t_test(group_a, group_b) -> TestResult
    def levene_test(groups) -> TestResult
    def anderson_darling_test(sample) -> TestResult
```

---

## 17. 인과 그래프 탐색 (Causal Discovery) — v23.0

### 17.1 PC Algorithm

관측 데이터에서 인과 방향성 그래프(DAG)를 추정합니다.

```
1. 완전 비방향 그래프로 시작 (모든 변수 쌍 연결)
2. 조건부 독립성 검정으로 간선 제거:
   X ⊥ Y | Z  →  X—Y 간선 제거
3. V-structure 방향 결정:
   X → Z ← Y  (X와 Y가 독립이고 Z로 조건화 시 의존)
4. 방향 전파 규칙 적용
```

### 17.2 구현

```python
# src/unified_pipeline/autonomous/analysis/causal_impact.py

class CausalGraphDiscovery:
    def pc_algorithm(data, alpha) -> CausalGraph
    def orient_edges(skeleton, sep_sets) -> DAG
```

---

## 18. 알고리즘 파이프라인 요약 (v23.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: Discovery                            │
├─────────────────────────────────────────────────────────────────┤
│  TDA (Betti Numbers) + Enhanced FK Detection                    │
│  + Schema Profiling + Value Matching (Jaccard)                  │
│  + Multi-Signal Scoring + Direction Analysis                    │
│  + Semantic Entity Resolution                                   │
│  + Composite FK + Hierarchy + Temporal FK                       │
│  + MinHash/LSH/HyperLogLog (Advanced Similarity)               │
│  + Entity Resolution (ML-based)                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 2: Refinement                           │
├─────────────────────────────────────────────────────────────────┤
│  Cross-Entity Correlation (Pearson) + Threshold Effects        │
│  + Graph Embedding (Node2Vec) + Semantic Reasoning              │
│  + OWL2 Reasoning + SHACL Validation                            │
│  + Causal Impact (ATE/CATE/Granger/PC Algorithm)               │
│  + Statistical Hypothesis Testing (KS/Welch/Levene/AD)         │
│  + Simpson's Paradox Detection                                   │
│  + Anomaly Detection (Ensemble: IQR/Z-score/IForest/LOF)       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 3: Governance                           │
├─────────────────────────────────────────────────────────────────┤
│  Dempster-Shafer Fusion + BFT Consensus                        │
│  + Evidence-Based Debate + Confidence Calibration               │
│  + AgentBus Communication                                       │
│  + KPI Prediction + Time Series Forecasting                     │
│  + Simulation Engine + What-If Analysis                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Post-Processing (v17 Services)               │
├─────────────────────────────────────────────────────────────────┤
│  Actions Engine + CDC Sync Engine                               │
│  + Column-level Lineage + Agent Memory                          │
│  + Decision Explainer (XAI) + Report Generator                  │
│  + Version Manager + Remediation Engine                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 19. 참고 문헌

1. **TDA**: Carlsson, G. (2009). Topology and Data. Bulletin of the AMS.
2. **Dempster-Shafer**: Shafer, G. (1976). A Mathematical Theory of Evidence.
3. **Node2Vec**: Grover & Leskovec (2016). node2vec: Scalable Feature Learning for Networks.
4. **BFT**: Castro & Liskov (1999). Practical Byzantine Fault Tolerance.
5. **FK Discovery**: Papenbrock et al. (2015). A Hybrid Approach to Functional Dependency Discovery.
6. **Data Profiling**: Abedjan et al. (2015). Profiling Relational Data: A Survey.
7. **Schema Matching**: Koutras et al. (2021). Valentine: Evaluating Matching Techniques for Dataset Discovery.
8. **OWL2**: W3C (2012). OWL 2 Web Ontology Language Document Overview.
9. **SHACL**: W3C (2017). Shapes Constraint Language (SHACL).
10. **Granger Causality**: Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models.
11. **CATE**: Künzel et al. (2019). Metalearners for Estimating Heterogeneous Treatment Effects.
12. **PC Algorithm**: Spirtes, Glymour & Scheines (2000). Causation, Prediction, and Search.
13. **Propensity Score**: Rosenbaum & Rubin (1983). The Central Role of the Propensity Score.
14. **Simpson's Paradox**: Simpson, E. H. (1951). The Interpretation of Interaction in Contingency Tables.
10. **Data Lineage**: Herschel et al. (2017). A Survey on Provenance: What for? What form? What from?
