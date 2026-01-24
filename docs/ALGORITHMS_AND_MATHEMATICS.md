# Ontoloty 알고리즘 및 수학적 기반

> **버전**: v1
> **최종 업데이트**: 2026-01-22

---

## 1. 개요

Ontoloty 시스템은 **LLM의 범용 추론(Reasoning)**과 **엄밀한 수학적 알고리즘(Rigor)**을 결합한 하이브리드 인텔리전스 시스템입니다.

### 핵심 원칙
1. **Algorithmic Evidence**: 통계/수학적 증거로 주장 뒷받침
2. **LLM Reasoning**: 비즈니스 맥락 이해 및 자연어 해석
3. **Multi-Agent Consensus**: 다양한 관점의 검증 및 합의

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

---

## 3. Enhanced FK Detection v1

### 3.1 설계 원칙

v1은 **Overfitting 방지**와 **Cross-dataset Generalization**을 핵심 목표로 설계되었습니다.

| 원칙 | 설명 |
|------|------|
| **Multi-Signal** | 단일 지표 대신 5가지 독립 신호 종합 |
| **Evidence-Based Direction** | 단일 룰 대신 여러 증거로 방향 결정 |
| **Semantic Resolution** | Domain-agnostic synonym groups |
| **Cross-Dataset Validation** | 2개 이상 데이터셋에서 검증 |

### 3.2 Multi-Signal FK Scoring (FKSignalScorer)

단일 지표(Jaccard/Containment)의 한계를 극복하기 위해 5가지 독립 신호를 종합합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Multi-Signal FK Scoring System v1                       │
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
│          Probabilistic FK Direction Analysis v1                  │
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
│          Semantic Entity Resolution v1                           │
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

### 3.5 Final Confidence Calculation

세 가지 분석 결과를 종합하여 최종 confidence를 계산합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Final Confidence Calculation                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  final_confidence = signal_conf × 0.50                          │
│                   + direction_conf × 0.25                       │
│                   + semantic_conf × 0.25                        │
│                                                                  │
│  Bonus:                                                          │
│  if signal_conf > 0.5 and direction_conf > 0.5                  │
│     and semantic_conf > 0.5:                                    │
│      final_confidence = min(final_confidence + 0.1, 1.0)        │
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

### 3.6 검증 결과

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
    def compute_cross_table_correlations(tables) -> List[Correlation]
    def detect_complementary_pairs(df) -> List[Dict]
    def detect_duplicate_columns(df) -> List[Dict]
    def detect_threshold_effects(x, y) -> List[ThresholdEffect]
    def segment_analysis(df, segment_col, value_col) -> SegmentInsight
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

### 7.3 구현

```python
# src/unified_pipeline/autonomous/consensus/engine.py

class ConsensusEngine:
    def run_consensus(concept, agents) -> ConsensusResult:
        # 1. 각 에이전트 투표 수집
        # 2. BFT 검증 (Byzantine 에이전트 탐지)
        # 3. DS Fusion으로 신뢰도 결합
        # 4. 임계값 기반 최종 결정
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

## 11. 알고리즘 파이프라인 요약

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Discovery                        │
├─────────────────────────────────────────────────────────────┤
│  TDA (Betti Numbers) + Enhanced FK Detection v1             │
│  + Schema Profiling + Value Matching (Jaccard)              │
│  + Multi-Signal Scoring + Direction Analysis                │
│  + Semantic Entity Resolution                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Phase 2: Refinement                       │
├─────────────────────────────────────────────────────────────┤
│  Cross-Entity Correlation (Pearson) + Threshold Effects     │
│  + Graph Embedding (Node2Vec) + Semantic Reasoning          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Phase 3: Governance                       │
├─────────────────────────────────────────────────────────────┤
│  Dempster-Shafer Fusion + BFT Consensus                     │
│  + Evidence-Based Debate + Confidence Calibration           │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. 참고 문헌

1. **TDA**: Carlsson, G. (2009). Topology and Data. Bulletin of the AMS.
2. **Dempster-Shafer**: Shafer, G. (1976). A Mathematical Theory of Evidence.
3. **Node2Vec**: Grover & Leskovec (2016). node2vec: Scalable Feature Learning for Networks.
4. **BFT**: Castro & Liskov (1999). Practical Byzantine Fault Tolerance.
5. **FK Discovery**: Papenbrock et al. (2015). A Hybrid Approach to Functional Dependency Discovery.
6. **Data Profiling**: Abedjan et al. (2015). Profiling Relational Data: A Survey.
7. **Schema Matching**: Koutras et al. (2021). Valentine: Evaluating Matching Techniques for Dataset Discovery.
