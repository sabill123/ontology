# Phase 2: Refinement Engine 기술 명세

> **역할**: 지식 설계자 (Knowledge Architect)
> **입력**: Phase 1에서 검증된 엔티티-관계 (Enhanced FK Detection v1)
> **출력**: 비즈니스 온톨로지, 지식 그래프, Cross-Entity 인사이트
> **최종 업데이트**: 2026-01-22
> **버전**: v1

---

## 1. 개요

Refinement Engine은 Phase 1의 물리적 발견(Physical Discovery)을 **비즈니스 지식(Business Knowledge)**으로 변환합니다.

### 핵심 기능
- **온톨로지 설계**: 엔티티, 속성, 관계를 정형화된 스키마로 구축
- **Cross-Entity 상관분석 (v10.0)**: 팔란티어 스타일 예측 인사이트 생성
- **다중 에이전트 합의**: 품질 평가 및 의미 검증

---

## 2. 에이전트 구성

### 2.1 OntologyArchitectAutonomousAgent

**역할**: 온톨로지 설계 및 인과관계 분석

```python
OntologyArchitectAutonomousAgent:
    - OntologyConcept 생성 (object_type, property, link_type)
    - 속성 정의 및 제약 조건 설정
    - Cross-Entity Correlation 분석 호출
    - Causal Inference 실행
```

**Cross-Entity Correlation (v10.0)**:
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
```

### 2.3 QualityJudgeAutonomousAgent

**역할**: 온톨로지 품질 평가

```python
QualityJudgeAutonomousAgent:
    - 개념 품질 점수 산출 (0-100%)
    - 상태 전이 결정: pending → provisional → approved
    - 증거 강도 평가
    - 신뢰도 보정 (Confidence Calibration)
```

**Smart Consensus (v8.2)**:
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
```

---

## 3. Cross-Entity Correlation 분석 (v10.0)

### 3.1 분석 유형

| 분석 유형 | 설명 | 출력 예시 |
|----------|------|----------|
| **상관관계** | 테이블 간 숫자 컬럼 Pearson 상관 | "Daily_Capacity ↔ Max_Weight: r=0.527" |
| **보완 관계** | 합이 1.0인 컬럼 쌍 탐지 | "신규비율 + 재방문비율 = 1.0 (중복)" |
| **동일 데이터** | 100% 일치 컬럼 탐지 | "모바일 ≡ 전체 (데이터 오류)" |
| **임계값 효과** | 비선형 변화 지점 발견 | "용량 > 31.5 시 중량 5544% 증가" |
| **세그먼트 집중도** | 카테고리별 분포 분석 | "상위 5개 채널이 92.5% 차지" |

### 3.2 출력: Predictive Insight

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
  "confidence": 0.7,
  "business_interpretation": "창고 용량이 31.5 임계값 초과 시 운송 중량 5544% 증가",
  "recommended_action": ["창고 용량 임계값 초과 설비 개선", "소형 화물 운송 전략 검토"]
}
```

---

## 4. 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                   Phase 2: Refinement                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1 Results                                             │
│  (entities, relationships, evidence)                         │
│       │                                                      │
│       ▼                                                      │
│  ┌───────────────────────────────────────┐                  │
│  │       Ontology Architect Agent        │                  │
│  │  ┌─────────────────────────────────┐  │                  │
│  │  │ Cross-Entity Correlation Analyzer│  │                  │
│  │  │  - Pearson Correlation          │  │                  │
│  │  │  - Threshold Effects            │  │                  │
│  │  │  - Segment Analysis             │  │                  │
│  │  └─────────────────────────────────┘  │                  │
│  └───────────────────┬───────────────────┘                  │
│                      │                                       │
│       ┌──────────────┼──────────────┐                       │
│       ▼              ▼              ▼                       │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │Conflict │   │ Quality │   │Semantic │                   │
│  │Resolver │   │  Judge  │   │Validator│                   │
│  └────┬────┘   └────┬────┘   └────┬────┘                   │
│       │              │              │                        │
│       └──────────────┴──────────────┘                        │
│                      │                                       │
│                      ▼                                       │
│         ┌─────────────────────────┐                         │
│         │    Smart Consensus      │                         │
│         │  (lightweight/full)     │                         │
│         └───────────┬─────────────┘                         │
│                     │                                        │
│                     ▼                                        │
│         ┌─────────────────────────┐                         │
│         │    SharedContext        │                         │
│         │  - ontology_concepts    │                         │
│         │  - business_insights    │                         │
│         │  - palantir_insights    │                         │
│         └─────────────────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 출력 데이터

### 5.1 Ontology Concept

```json
{
  "concept_id": "obj_customer_001",
  "concept_type": "object_type",
  "name": "Customer",
  "description": "비즈니스 고객 엔티티",
  "source_tables": ["vmi_customers", "order_list"],
  "status": "approved",
  "confidence": 0.92,
  "agent_assessments": {
    "quality_judge": {"decision": "approve", "confidence": 0.95},
    "semantic_validator": {"decision": "approve", "confidence": 0.88}
  }
}
```

### 5.2 Business Insight

```json
{
  "insight_id": "INSIGHT_0001",
  "title": "High Dependency on Limited Plants",
  "description": "PLANT05가 67개 제품을 담당, 공급망 집중 리스크",
  "insight_type": "risk",
  "severity": "high",
  "source_table": "products_per_plant",
  "recommendations": [
    "다른 공장으로 제품 분산",
    "PLANT05 운영 효율성 평가"
  ],
  "business_impact": "공급망 중단 리스크 완화 가능"
}
```

---

## 6. 핵심 분석 모듈

| 모듈 | 파일 | 기능 |
|------|------|------|
| Cross-Entity | `analysis/cross_entity_correlation.py` | 테이블 간 상관분석 |
| Knowledge Graph | `analysis/knowledge_graph.py` | RDF 트리플 구축 |
| Graph Embedding | `analysis/graph_embedding.py` | Node2Vec 임베딩 |
| Semantic Reasoner | `analysis/semantic_reasoner.py` | OWL/RDFS 추론 |
| Causal Impact | `analysis/causal_impact.py` | 인과관계 분석 |
| Business Insights | `analysis/business_insights.py` | KPI, 이상치 탐지 |

---

## 7. Phase Gate 검증 (v14.0)

Refinement 완료 시 다음 조건 확인:

```python
Phase2Gate:
    - ontology_concepts >= 1  # 최소 1개 개념 생성
    - approved_ratio >= 0.5   # 50% 이상 승인
    - no critical conflicts   # 심각한 충돌 없음
    - completion_signal == [DONE]
```

실패 시 자동 Fallback 적용 후 Phase 3 진행.
