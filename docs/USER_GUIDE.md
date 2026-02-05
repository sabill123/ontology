# Ontoloty 사용 설명서

> **버전**: v1.0
> **최종 업데이트**: 2026-01-27

---

## 1. Ontoloty란?

**Ontoloty**는 데이터 사일로(Data Silo)에서 자동으로 온톨로지(Ontology)를 생성하고, 비즈니스 인사이트를 도출하는 **AI 기반 자율 데이터 분석 시스템**입니다.

### 1.1 핵심 가치

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ontoloty의 핵심 가치                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  데이터 이해 자동화                                           │
│     → 수동 스키마 분석 없이 데이터 구조를 자동으로 파악           │
│                                                                  │
│  관계 발견                                                    │
│     → 테이블 간 FK 관계를 자동으로 탐지 (95.5% 정확도)           │
│                                                                  │
│  비즈니스 인사이트                                            │
│     → 데이터에서 숨겨진 패턴과 상관관계 발견                      │
│                                                                  │
│  거버넌스 지원                                                │
│     → 데이터 품질 평가 및 거버넌스 정책 자동 생성                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 어떤 문제를 해결하나요?

### 2.1 데이터 사일로 문제

**문제**: 조직 내 여러 부서에서 각자의 데이터베이스를 운영하면서, 데이터 간 관계가 문서화되지 않고 단절됨

**Ontoloty 해결책**:
- CSV/JSON/Parquet 파일을 입력으로 받아 자동으로 테이블 간 관계 탐지
- 명시적 FK가 없어도 값 패턴, 이름 유사성, 구조적 특성으로 관계 추론
- 도메인에 관계없이 작동 (Supply Chain, Healthcare, Retail, Finance 등)

### 2.2 스키마 문서화 부재

**문제**: 레거시 시스템이나 인수합병된 데이터의 스키마가 문서화되어 있지 않음

**Ontoloty 해결책**:
- 데이터 프로파일링으로 각 컬럼의 역할 자동 분류 (PK, FK, Metric, Dimension 등)
- 온톨로지 형태로 데이터 구조 시각화
- OWL2/SHACL 표준으로 의미론적 정의 생성

### 2.3 숨겨진 인사이트 발견

**문제**: 대량의 데이터에서 유의미한 패턴이나 상관관계를 찾기 어려움

**Ontoloty 해결책**:
- Cross-Entity 상관관계 분석 (Pearson correlation)
- 임계값 효과 탐지 ("창고 용량 31.5 초과 시 운송량 5544% 증가")
- 계절성 패턴 탐지 ("14일 주기 계절성")
- 위상 데이터 분석(TDA)으로 데이터 구조 복잡도 평가

### 2.4 데이터 거버넌스

**문제**: 데이터 품질 문제와 거버넌스 정책 수립이 어려움

**Ontoloty 해결책**:
- 결측치, 이상치, 중복 데이터 자동 탐지
- SHACL 기반 데이터 제약조건 자동 생성
- 다중 에이전트 합의로 거버넌스 결정 도출

---

## 3. 입력 데이터

### 3.1 지원 형식

| 형식 | 확장자 | 설명 |
|------|--------|------|
| CSV | `.csv` | 쉼표로 구분된 값 |
| JSON | `.json` | JSON 배열 또는 객체 |
| Parquet | `.parquet` | Apache Parquet 형식 |

### 3.2 데이터 구조 요구사항

```
data/
├── scenario_name/           # 시나리오 폴더
│   ├── table1.csv          # 테이블 1
│   ├── table2.csv          # 테이블 2
│   ├── table3.json         # 테이블 3 (혼합 가능)
│   └── ...
```

### 3.3 입력 예시

**customers.csv**:
```csv
customer_id,name,email,region
C001,홍길동,hong@example.com,서울
C002,김철수,kim@example.com,부산
```

**orders.csv**:
```csv
order_id,buyer_id,product_code,order_date,amount
O001,C001,P100,2026-01-15,50000
O002,C002,P101,2026-01-16,75000
```

→ Ontoloty는 `buyer_id`가 `customer_id`를 참조한다는 것을 자동으로 탐지합니다.

---

## 4. 실행 방법

### 4.1 기본 실행

```bash
# 특정 시나리오 실행
python -m src.unified_pipeline.unified_main --scenario supply_chain_silo

# 모든 시나리오 실행
python -m src.unified_pipeline.unified_main --all
```

### 4.2 실행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--scenario` | 분석할 시나리오 이름 | - |
| `--all` | 모든 시나리오 실행 | False |
| `--output-dir` | 결과 출력 디렉토리 | `output/` |
| `--debug` | 디버그 모드 | False |

### 4.3 프로그래밍 방식 실행

```python
from src.unified_pipeline.autonomous import AgentOrchestrator
from src.unified_pipeline.autonomous.shared_context import SharedContext

# 컨텍스트 초기화
context = SharedContext()
context.load_tables_from_directory("data/my_scenario")

# 오케스트레이터 실행
orchestrator = AgentOrchestrator(context)
result = await orchestrator.run()

# 결과 저장
context.save_to_file("output/my_scenario_context.json")
```

---

## 5. 출력 결과

### 5.1 출력 파일 구조

```
output/{scenario_name}/
├── {scenario}_context.json      # 전체 분석 컨텍스트
├── {scenario}_entities.json     # 발견된 엔티티
├── {scenario}_relationships.json # 발견된 관계
├── {scenario}_insights.json     # 비즈니스 인사이트
├── {scenario}_ontology.json     # 온톨로지 정의
├── {scenario}_lineage.json      # 컬럼 리니지 (v1.0)
├── {scenario}_quality.json      # 데이터 품질 보고서
└── {scenario}_summary.json      # 실행 요약
```

### 5.2 주요 출력 상세

#### 5.2.1 발견된 관계 (relationships.json)

```json
{
  "relationships": [
    {
      "source_table": "orders",
      "source_column": "buyer_id",
      "target_table": "customers",
      "target_column": "customer_id",
      "relationship_type": "MANY_TO_ONE",
      "confidence": 0.92,
      "evidence": {
        "value_overlap": 1.0,
        "cardinality_ratio": 0.15,
        "naming_similarity": 0.85
      }
    }
  ]
}
```

#### 5.2.2 비즈니스 인사이트 (insights.json)

```json
{
  "insights": [
    {
      "type": "correlation",
      "description": "warehouse_capacity와 shipping_weight 간 강한 양의 상관관계 (r=0.87)",
      "tables": ["warehouses", "shipments"],
      "columns": ["capacity", "weight"],
      "confidence": 0.87,
      "business_implication": "창고 용량이 클수록 더 무거운 화물을 처리함"
    },
    {
      "type": "threshold_effect",
      "description": "창고 용량 31.5 초과 시 운송 중량 5544% 증가",
      "threshold_value": 31.5,
      "business_implication": "31.5 이상의 창고 용량이 대량 화물 처리의 임계점"
    },
    {
      "type": "seasonality",
      "description": "주문량에서 14일 주기 계절성 탐지 (강도: 57%)",
      "period": 14,
      "strength": 0.57,
      "business_implication": "2주 단위 프로모션 또는 급여 주기와 연관 가능성"
    }
  ]
}
```

#### 5.2.3 온톨로지 정의 (ontology.json)

```json
{
  "classes": [
    {
      "name": "Customer",
      "source_table": "customers",
      "properties": [
        {"name": "customer_id", "type": "identifier", "datatype": "string"},
        {"name": "name", "type": "attribute", "datatype": "string"},
        {"name": "region", "type": "dimension", "datatype": "string"}
      ]
    },
    {
      "name": "Order",
      "source_table": "orders",
      "properties": [
        {"name": "order_id", "type": "identifier", "datatype": "string"},
        {"name": "amount", "type": "metric", "datatype": "decimal"}
      ]
    }
  ],
  "object_properties": [
    {
      "name": "placedBy",
      "domain": "Order",
      "range": "Customer",
      "inverse": "hasOrder"
    }
  ]
}
```

#### 5.2.4 데이터 품질 보고서 (quality.json)

```json
{
  "tables": {
    "customers": {
      "row_count": 1000,
      "completeness": 0.98,
      "issues": [
        {
          "type": "missing_values",
          "column": "email",
          "count": 20,
          "percentage": 2.0
        }
      ]
    },
    "orders": {
      "row_count": 5000,
      "completeness": 1.0,
      "issues": []
    }
  },
  "overall_quality_score": 0.95,
  "recommendations": [
    "customers.email 컬럼의 결측값 2% 처리 권장"
  ]
}
```

#### 5.2.5 컬럼 리니지 (lineage.json) - v1.0

```json
{
  "nodes": [
    {"id": "customers.customer_id", "type": "SOURCE", "table": "customers"},
    {"id": "orders.buyer_id", "type": "SOURCE", "table": "orders"},
    {"id": "report.customer_name", "type": "DERIVED", "table": "report"}
  ],
  "edges": [
    {
      "source": "orders.buyer_id",
      "target": "customers.customer_id",
      "type": "FK_REFERENCE"
    },
    {
      "source": "customers.name",
      "target": "report.customer_name",
      "type": "TRANSFORM",
      "transformation": "COPY"
    }
  ]
}
```

---

## 6. 지원 도메인

Ontoloty는 **도메인에 구애받지 않는(Domain-Agnostic)** 설계로, 다양한 산업 분야에서 사용 가능합니다.

### 6.1 Supply Chain (공급망)

**탐지 가능한 엔티티**:
- Supplier, Warehouse, Plant, Shipment, Freight, Order

**인사이트 예시**:
- "창고 A와 B 간 운송 효율성 차이 분석"
- "공급업체별 리드타임 비교"
- "Plant별 제품 생산량 집중도"

### 6.2 Healthcare (의료)

**탐지 가능한 엔티티**:
- Patient, Diagnosis, Prescription, LabResult, Claim

**인사이트 예시**:
- "특정 진단과 처방 패턴 상관관계"
- "환자 재입원율 예측 요인"
- "보험 청구 거절 패턴"

### 6.3 Retail (소매)

**탐지 가능한 엔티티**:
- Product, Customer, Order, Cart, Promotion, Category

**인사이트 예시**:
- "장바구니 이탈률과 프로모션 효과"
- "고객 세그먼트별 구매 패턴"
- "상품 카테고리 간 교차 판매 기회"

### 6.4 Web Analytics (웹 분석)

**탐지 가능한 엔티티**:
- Page, Session, Visitor, Channel, Conversion

**인사이트 예시**:
- "채널별 전환율 비교"
- "세션 지속시간과 전환 상관관계"
- "신규 vs 재방문 사용자 행동 차이"

### 6.5 Finance (금융)

**탐지 가능한 엔티티**:
- Account, Transaction, Loan, Investment, RiskMetric

**인사이트 예시**:
- "대출 승인과 신용점수 임계값"
- "거래 패턴 이상 탐지"
- "포트폴리오 위험 지표 상관관계"

---

## 7. 분석 파이프라인

### 7.1 3단계 분석 프로세스

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: Discovery                            │
│                       (발견 단계)                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  수행 작업:                                                   │
│     • 데이터 로딩 및 프로파일링                                   │
│     • 위상 데이터 분석 (TDA)                                     │
│     • FK 관계 자동 탐지                                          │
│     • 엔티티 분류                                                │
│                                                                  │
│  담당 에이전트 (6개):                                         │
│     Data Analyst, TDA Expert, Schema Analyst,                  │
│     Value Matcher, Entity Classifier, Relationship Detector    │
│                                                                  │
│  출력:                                                        │
│     • 테이블 프로파일                                            │
│     • FK 후보 목록                                               │
│     • 엔티티 분류 결과                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 2: Refinement                           │
│                       (정제 단계)                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  수행 작업:                                                   │
│     • 온톨로지 구축                                              │
│     • Cross-Entity 상관관계 분석                                 │
│     • OWL2 추론 (v1.0)                                         │
│     • SHACL 검증 (v1.0)                                        │
│     • 데이터 품질 평가                                           │
│                                                                  │
│  담당 에이전트 (4개):                                         │
│     Ontology Architect, Conflict Resolver,                      │
│     Quality Judge, Semantic Validator                           │
│                                                                  │
│  출력:                                                        │
│     • 온톨로지 정의                                              │
│     • 비즈니스 인사이트                                          │
│     • 품질 보고서                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 3: Governance                           │
│                       (거버넌스 단계)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  수행 작업:                                                   │
│     • 다중 에이전트 합의 (BFT + Dempster-Shafer)                 │
│     • 거버넌스 정책 생성                                         │
│     • 우선순위 결정                                              │
│     • 최종 검증                                                  │
│                                                                  │
│  담당 에이전트 (4개):                                         │
│     Governance Strategist, Action Prioritizer,                  │
│     Risk Assessor, Policy Generator                             │
│                                                                  │
│  출력:                                                        │
│     • 최종 온톨로지                                              │
│     • 거버넌스 결정                                              │
│     • 실행 권고사항                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 14개 AI 에이전트

| 단계 | 에이전트 | 역할 |
|------|----------|------|
| Discovery | Data Analyst | 데이터 프로파일링 및 LLM 기반 분석 |
| Discovery | TDA Expert | 위상 데이터 분석 |
| Discovery | Schema Analyst | 스키마 구조 분석 |
| Discovery | Value Matcher | 값 패턴 매칭 |
| Discovery | Entity Classifier | 엔티티 분류 |
| Discovery | Relationship Detector | 관계 탐지 |
| Refinement | Ontology Architect | 온톨로지 설계 |
| Refinement | Conflict Resolver | 충돌 해결 |
| Refinement | Quality Judge | 품질 평가 |
| Refinement | Semantic Validator | 의미론적 검증 |
| Governance | Governance Strategist | 거버넌스 전략 |
| Governance | Action Prioritizer | 우선순위 결정 |
| Governance | Risk Assessor | 위험 평가 |
| Governance | Policy Generator | 정책 생성 |

---

## 8. 주요 기능 (v1.0)

### 8.1 Enhanced FK Detection

**기능**: 명시적 FK 없이도 테이블 간 관계를 자동 탐지

**방법**:
- Multi-Signal Scoring: 5가지 신호(값 오버랩, 카디널리티, 참조 무결성, 네이밍, 구조) 종합
- Direction Analysis: 어느 테이블이 Parent인지 자동 판단
- Semantic Resolution: 동의어 그룹으로 `buyer_id` → `customers` 매핑

**정확도**: Cross-dataset 평균 F1 91.7%

### 8.2 Composite FK Detection (v1.0)

**기능**: 복합 키 관계 탐지

**예시**:
```
products_per_plant(plant_code, product_id, quantity)
  ↓
(plant_code → plants.plant_code) + (product_id → products.product_id)
```

### 8.3 Hierarchy Detection (v1.0)

**기능**: 계층 구조 자동 탐지

**예시**:
```
store → city → region → country
employee → manager → director → CEO
```

### 8.4 Temporal FK Detection (v1.0)

**기능**: 시간 기반 FK 관계 탐지

**예시**:
```
order.created_at → price_history.effective_date
(주문 시점의 유효한 가격 참조)
```

### 8.5 OWL2 Reasoning (v1.0)

**기능**: W3C 표준 온톨로지 추론

**수행**:
- 클래스 계층 추론 (Customer ⊑ Person)
- 속성 도메인/치역 정의
- 역속성 추론
- 일관성 검증

### 8.6 SHACL Validation (v1.0)

**기능**: 데이터 형태 제약조건 자동 생성 및 검증

**생성되는 제약조건**:
- 기수 제약 (minCount, maxCount)
- 데이터타입 제약
- 정규식 패턴
- 숫자 범위

### 8.7 Column-level Lineage (v1.0)

**기능**: 컬럼 수준 데이터 리니지 추적

**활용**:
- Upstream 추적: 이 컬럼의 원본은?
- Downstream 추적: 이 컬럼이 영향 주는 곳은?
- Impact Analysis: 컬럼 변경 시 영향도 분석

### 8.8 Agent Communication Bus (v1.0)

**기능**: 에이전트 간 직접 통신

**패턴**:
- Direct Messaging (1:1)
- Broadcast (1:N)
- Pub/Sub (토픽 기반)
- Request/Response (RPC 스타일)

---

## 9. 사용 시나리오 예시

### 9.1 시나리오: 인수합병 후 데이터 통합

**상황**: 회사 A가 회사 B를 인수하여 두 회사의 데이터베이스를 통합해야 함

**문제**:
- 회사 B의 스키마 문서가 없음
- 테이블/컬럼 명명 규칙이 다름
- 어떤 테이블이 어떤 테이블과 연결되는지 모름

**Ontoloty 사용**:
```bash
# 두 회사 데이터를 하나의 시나리오로 구성
data/
└── merger_integration/
    ├── company_a_customers.csv
    ├── company_a_orders.csv
    ├── company_b_clients.csv      # "customers"와 동일
    ├── company_b_purchases.csv    # "orders"와 동일
    └── ...

# 분석 실행
python -m src.unified_pipeline.unified_main --scenario merger_integration
```

**결과**:
- `company_a_customers`와 `company_b_clients`가 동일한 엔티티임을 탐지
- 컬럼 매핑 자동 생성: `customer_id` ↔ `client_no`
- 통합 온톨로지 생성

### 9.2 시나리오: 레거시 시스템 문서화

**상황**: 10년 된 레거시 시스템의 데이터베이스를 현대화해야 함

**문제**:
- 원래 개발자가 퇴사함
- ERD나 문서가 없음
- 테이블이 100개 이상

**Ontoloty 사용**:
```bash
# 레거시 DB를 CSV로 덤프
pg_dump --table-data-only --csv legacy_db > data/legacy_system/

# 분석 실행
python -m src.unified_pipeline.unified_main --scenario legacy_system
```

**결과**:
- 모든 테이블 간 관계 자동 탐지
- 각 테이블의 역할 분류 (마스터, 트랜잭션, 룩업 등)
- OWL2 기반 온톨로지 문서 생성
- SHACL 제약조건으로 데이터 무결성 규칙 정의

### 9.3 시나리오: 비즈니스 인사이트 발굴

**상황**: 매출 데이터에서 성장 기회를 찾고 싶음

**Ontoloty 사용**:
```bash
python -m src.unified_pipeline.unified_main --scenario sales_analysis
```

**결과 예시**:
```json
{
  "insights": [
    {
      "type": "threshold_effect",
      "description": "고객 생애가치(LTV)가 $500 초과 시 재구매율 340% 증가",
      "recommendation": "$500 LTV 달성을 위한 타겟 마케팅 권장"
    },
    {
      "type": "correlation",
      "description": "무료배송과 평균 주문 금액 간 0.72 상관관계",
      "recommendation": "무료배송 임계값 조정으로 AOV 증가 가능"
    },
    {
      "type": "seasonality",
      "description": "매월 1일, 15일에 주문량 피크 (급여일 효과)",
      "recommendation": "해당 날짜에 프로모션 집중"
    }
  ]
}
```

---

## 10. 문제 해결 (Troubleshooting)

### 10.1 일반적인 문제

| 문제 | 원인 | 해결책 |
|------|------|--------|
| FK가 탐지되지 않음 | 값 오버랩 부족 | 샘플 데이터 확인, 참조 무결성 검증 |
| 잘못된 방향 탐지 | 카디널리티 불명확 | 수동 검토 후 재실행 |
| 메모리 부족 | 대용량 데이터 | 샘플링 또는 청크 처리 |
| 느린 실행 | 테이블 수 과다 | 병렬 처리 활성화 |

### 10.2 로그 확인

```bash
# 상세 로그 활성화
python -m src.unified_pipeline.unified_main --scenario my_data --debug

# 로그 파일 위치
output/my_data/logs/
├── discovery.log
├── refinement.log
└── governance.log
```

---

## 11. API 레퍼런스

### 11.1 주요 클래스

```python
from src.unified_pipeline.autonomous import (
    AgentOrchestrator,      # 전체 파이프라인 오케스트레이션
    SharedContext,          # 공유 컨텍스트
    AgentCommunicationBus,  # 에이전트 통신 버스 (v1.0)
)

from src.unified_pipeline.analysis import (
    EnhancedFKPipeline,     # FK 탐지 파이프라인
    CompositeFKDetector,    # 복합 FK 탐지 (v1.0)
    HierarchyDetector,      # 계층 탐지 (v1.0)
    TemporalFKDetector,     # 시간 FK 탐지 (v1.0)
)

from src.unified_pipeline.autonomous.analysis import (
    OWL2Reasoner,           # OWL2 추론 (v1.0)
    SHACLValidator,         # SHACL 검증 (v1.0)
)

from src.unified_pipeline.lineage import (
    LineageTracker,         # 리니지 추적 (v1.0)
)
```

### 11.2 기본 사용 예시

```python
import asyncio
from src.unified_pipeline.autonomous import AgentOrchestrator
from src.unified_pipeline.autonomous.shared_context import SharedContext

async def analyze_data():
    # 1. 컨텍스트 초기화
    context = SharedContext()
    context.load_tables_from_directory("data/my_scenario")

    # 2. 오케스트레이터 생성 및 실행
    orchestrator = AgentOrchestrator(context)
    result = await orchestrator.run()

    # 3. 결과 접근
    entities = context.get("entities")
    relationships = context.get("relationships")
    insights = context.get("insights")

    # 4. 결과 저장
    context.save_to_file("output/my_scenario_context.json")

    return result

# 실행
asyncio.run(analyze_data())
```

---

## 12. 성능 벤치마크

### 12.1 FK Detection 정확도

| 데이터셋 | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| marketing_silo (12 FK) | 100.0% | 100.0% | 100.0% |
| airport_silo (13 FK) | 90.9% | 76.9% | 83.3% |
| supply_chain (10 FK) | 95.0% | 90.0% | 92.4% |
| **평균** | **95.5%** | **88.5%** | **91.7%** |

### 12.2 처리 시간 (참고용)

| 테이블 수 | 평균 행 수 | 처리 시간 |
|-----------|------------|-----------|
| 5 | 1,000 | ~30초 |
| 10 | 10,000 | ~2분 |
| 20 | 100,000 | ~10분 |
| 50 | 500,000 | ~30분 |

*실제 시간은 데이터 복잡도와 하드웨어에 따라 다를 수 있습니다.*

---

## 13. 자주 묻는 질문 (FAQ)

### Q1: Ontoloty는 어떤 언어로 작성되었나요?
**A**: Python 3.10+ 기반입니다.

### Q2: 클라우드에서 실행할 수 있나요?
**A**: 네, Docker 컨테이너로 제공되며 AWS, GCP, Azure에서 실행 가능합니다.

### Q3: LLM(대규모 언어 모델)이 필요한가요?
**A**: 네, 비즈니스 맥락 이해와 자연어 해석에 LLM을 활용합니다. OpenAI API 또는 로컬 LLM을 지원합니다.

### Q4: 실시간 데이터 분석이 가능한가요?
**A**: v1.0의 CDC Sync Engine으로 증분 데이터 처리가 가능합니다.

### Q5: 결과를 BI 도구에 연결할 수 있나요?
**A**: 온톨로지를 JSON-LD, RDF/XML로 내보내기 가능하며, 그래프 데이터베이스(Neo4j 등)와 연동됩니다.

---

## 14. 다음 단계

1. **Quick Start**: `data/sample` 폴더의 예제 데이터로 시작
2. **사용자 데이터 분석**: 자체 데이터로 시나리오 생성
3. **결과 검토**: 생성된 온톨로지와 인사이트 검토
4. **커스터마이징**: 도메인별 동의어 그룹 추가

---

## 15. 지원 및 문의

- **문서**: `/docs` 폴더의 상세 기술 문서 참조
- **이슈 리포트**: GitHub Issues
- **이메일**: support@ontoloty.io

---

*이 문서는 Ontoloty v1.0 기준으로 작성되었습니다. (2026-01-27)*
