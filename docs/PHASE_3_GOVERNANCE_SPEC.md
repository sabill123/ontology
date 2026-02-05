# Phase 3: Governance Engine 기술 명세

> **역할**: 최종 심판관 (Supreme Court)
> **입력**: Phase 2에서 제안된 온톨로지 및 인사이트
> **출력**: 검증된 거버넌스 결정, 비즈니스 액션, 정책
> **최종 업데이트**: 2026-01-27
> **버전**: v1.0

---

## 1. 개요

Governance Engine은 시스템의 **최종 관문**입니다. Phase 1, 2에서 생성된 모든 인사이트는 **GovernanceStrategist 내부에 임베딩된 Agent Council(4개 페르소나)**의 검증을 통과해야 합니다. Agent Council은 별도 에이전트가 아니라 GovernanceStrategist가 LLM 호출 시 사용하는 4개의 관점(페르소나)입니다.

### 핵심 특징
- **BFT 기반 합의**: Byzantine Fault Tolerant 다중 에이전트 합의
- **Dempster-Shafer 융합**: 불확실한 증거의 수학적 결합
- **Evidence-Based Debate**: 증거 기반 토론 프로토콜

### v1.0 핵심 기능
- **Actions Engine**: 자동 액션 생성 및 실행
- **CDC Sync Engine**: 실시간 변경 데이터 캡처 및 동기화
- **Impact Analysis**: Lineage 기반 영향도 분석
- **Agent Communication Bus**: 합의 과정 실시간 브로드캐스트
- **Agent Learning**: 거버넌스 결정 패턴 학습

---

## 2. 에이전트 구성 (4개)

### 2.1 GovernanceStrategistAutonomousAgent

**역할**: 전략적 의사결정 프레임워크

```python
GovernanceStrategistAutonomousAgent:
    - 거버넌스 정책 프레임워크 설계
    - 리스크/베네핏 분석
    - 조직 정렬 전략 수립
    - 승인/거부/에스컬레이션 결정
    - LineageTracker로 영향도 분석
    - ActionsEngine으로 액션 생성
    - AgentBus로 결정 브로드캐스트

    # 내부 Agent Council (4개 페르소나, 별도 에이전트가 아님)
    # LLM 호출 시 다음 4가지 관점으로 평가:
    - Ontologist: 논리적 일관성 ("이 정의가 모순되지 않는가?")
    - Risk Assessor: 리스크 평가 ("이 인사이트에 따라 행동 시 위험은?")
    - Business Strategist: 비즈니스 가치 ("이것이 비즈니스 임팩트가 있는가?")
    - Data Steward: 데이터 품질 ("데이터 품질과 컴플라이언스 문제는?")
```

### 2.2 ActionPrioritizerAutonomousAgent

**역할**: 액션 우선순위 결정

```python
ActionPrioritizerAutonomousAgent:
    - 액션 우선순위화 (criticality, impact, effort)
    - 워크플로우 시퀀싱
    - 의존성 해결
    - KPI 예측
    - ActionsEngine 연동
    - AgentMemory로 우선순위 패턴 학습
```

### 2.3 RiskAssessorAutonomousAgent

**역할**: 리스크 식별 및 평가

```python
RiskAssessorAutonomousAgent:
    - 리스크 유형 식별 (데이터 품질, 컴플라이언스, 통합)
    - 심각도/가능성 평가
    - 완화 방안 제안
    - 리스크 집계
    - LineageTracker로 변경 영향 리스크 분석
```

### 2.4 PolicyGeneratorAutonomousAgent

**역할**: 정책 생성

```python
PolicyGeneratorAutonomousAgent:
    - 데이터 거버넌스 정책 초안
    - 접근 제어 규칙 생성
    - 데이터 품질 표준 정의
    - 컴플라이언스 요구사항 명세
    - SHACLValidator로 정책을 Shape로 변환
```

---

## 3. Enterprise Features 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│              Enterprise Features Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Actions Engine                                              │    │
│  │  • 액션 생성: create_action()                                │    │
│  │  • 액션 실행: execute_action()                               │    │
│  │  • 우선순위 정렬: prioritize_actions()                       │    │
│  │  • 스케줄링: schedule_action()                               │    │
│  │  파일: actions/engine.py                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  CDC Sync Engine                                             │    │
│  │  • 변경 감지: process_change()                               │    │
│  │  • 핸들러 등록: register_handler()                           │    │
│  │  • 타겟 동기화: sync_to_target()                             │    │
│  │  • Lineage 업데이트: _apply_to_lineage()                     │    │
│  │  파일: sync/cdc_engine.py                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Impact Analysis                                             │    │
│  │  • Upstream 추적: lineage.get_upstream()                     │    │
│  │  • Downstream 추적: lineage.get_downstream()                 │    │
│  │  • 영향도 분석: lineage.impact_analysis()                    │    │
│  │  파일: lineage/column_lineage.py                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Agent Learning System                                       │    │
│  │  • 결정 저장: memory.store()                                 │    │
│  │  • 패턴 검색: memory.recall()                                │    │
│  │  • 피드백 학습: memory.learn_from_feedback()                 │    │
│  │  파일: learning/agent_memory.py                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Actions Engine 상세

### 4.1 액션 상태 및 카테고리

```python
class ActionStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ActionPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class ActionCategory(Enum):
    DATA_QUALITY = "data_quality"
    SCHEMA_CHANGE = "schema_change"
    RELATIONSHIP_CREATE = "relationship_create"
    ONTOLOGY_UPDATE = "ontology_update"
    NOTIFICATION = "notification"
    EXPORT = "export"
    SYNC = "sync"
```

### 4.2 액션 생성 및 실행

```python
from unified_pipeline.actions import get_actions_engine

engine = get_actions_engine()

# 액션 생성
action = engine.create_action(
    category=ActionCategory.ONTOLOGY_UPDATE,
    title="Customer 엔티티 승인 후 적용",
    priority=ActionPriority.HIGH,
    payload={
        "concept_id": "obj_customer_001",
        "target_env": "production"
    }
)

# 액션 실행
result = await engine.execute_action(action.action_id)
```

---

## 5. CDC Sync Engine 상세

### 5.1 변경 유형

```python
class ChangeType(Enum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRUNCATE = "truncate"
```

### 5.2 변경 이벤트 처리

```python
from unified_pipeline.sync import get_cdc_engine

cdc = get_cdc_engine()

# 핸들러 등록
async def on_entity_change(event: ChangeEvent):
    # Lineage 업데이트
    # 관련 온톨로지 재검증
    pass

cdc.register_handler("entities", on_entity_change)

# 변경 처리
await cdc.process_change(ChangeEvent(
    table="customers",
    change_type=ChangeType.UPDATE,
    old_data={"status": "active"},
    new_data={"status": "inactive"}
))
```

---

## 6. Impact Analysis 상세

### 6.1 영향도 분석 결과

```python
@dataclass
class ImpactAnalysis:
    source_column: str
    affected_columns: List[str]       # 영향받는 컬럼 목록
    affected_tables: List[str]        # 영향받는 테이블 목록
    impact_paths: List[List[str]]     # 영향 경로
    severity: str                     # low, medium, high, critical
    recommended_actions: List[str]
```

### 6.2 사용 예시

```python
from unified_pipeline.lineage import build_lineage_from_context

lineage = build_lineage_from_context(self.shared_context)

# 컬럼 변경 시 영향도 분석
impact = lineage.impact_analysis(
    source_column="customers.customer_id"
)

# 결과: affected_tables=["orders", "invoices", "shipments"]
```

---

## 7. 합의 프로토콜

### 7.1 BFT-Inspired Consensus Engine

```python
ConsensusEngine:
    Round 1: Initial Vote
        - 각 에이전트가 approve/reject/review 투표
        - 만장일치 또는 >75% 동의 시 Fast-track 통과
        - AgentBus로 vote_cast 이벤트 발행

    Round 2: Deep Debate (충돌 시)
        - 반대 의견 에이전트에게 Challenge 발언권
        - 알고리즘 증거 제시
        - AgentBus로 challenge_issued 이벤트 발행
        - 재투표

    Final: Forced Decision
        - Max Rounds 도달 시 Soft Consensus
        - 신뢰도 하향 조정
        - AgentBus로 consensus_reached 이벤트 발행
```

### 7.2 Dempster-Shafer Fusion

```
(m₁ ⊕ m₂)(A) = (1/(1-K)) × Σ m₁(B) × m₂(C)
                          B∩C=A

K = 충돌하는 증거의 양
K가 너무 크면 "판단 보류" 선언
```

---

## 8. 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Phase 3: Governance                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 2 Results                                                     │
│  (ontology, insights, owl2_inferred, shacl_results)                 │
│       │                                                              │
│       ▼                                                              │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │              Agent Council (4 Personas)                    │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │      │
│  │  │Ontologist│ │  Risk    │ │ Business │ │  Data    │     │      │
│  │  │          │ │ Assessor │ │Strategist│ │ Steward  │     │      │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘     │      │
│  │       │            │            │            │            │      │
│  │       └────────────┴────────────┴────────────┘            │      │
│  │                         │                                 │      │
│  │                         ▼                                 │      │
│  │  ┌─────────────────────────────────────────────────┐     │      │
│  │  │    Consensus Engine + AgentBus                  │     │      │
│  │  │    (BFT + DS Fusion + Real-time Broadcast)      │     │      │
│  │  └─────────────────────────────────────────────────┘     │      │
│  └───────────────────────────┬───────────────────────────────┘      │
│                              │                                       │
│       ┌──────────────────────┼──────────────────────┐               │
│       ▼                      ▼                      ▼               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │  Governance  │     │    Action    │     │    Policy    │        │
│  │  Strategist  │     │  Prioritizer │     │  Generator   │        │
│  │              │     │              │     │              │        │
│  │ • Lineage    │     │ • Actions    │     │ • SHACL      │        │
│  │   Impact     │     │   Engine     │     │   Shapes     │        │
│  │ • Actions    │     │ • Priority   │     │              │        │
│  │ • AgentBus   │     │ • AgentMem   │     │              │        │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘        │
│         │                    │                     │                │
│         └────────────────────┴─────────────────────┘                │
│                              │                                       │
│            ┌─────────────────┼─────────────────┐                    │
│            ▼                 ▼                 ▼                    │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐               │
│     │ Approved │      │  Review  │      │ Rejected │               │
│     │ Actions  │      │  Queue   │      │   Log    │               │
│     └────┬─────┘      └──────────┘      └──────────┘               │
│          │                                                          │
│          ▼                                                          │
│  ┌───────────────────────────────────────────────────────┐         │
│  │              Actions Engine                            │         │
│  │  • execute_action()                                    │         │
│  │  • schedule_action()                                   │         │
│  └───────────────────────────────────────────────────────┘         │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────┐         │
│  │              CDC Sync Engine                           │         │
│  │  • 변경 전파                                            │         │
│  │  • Lineage 업데이트                                     │         │
│  └───────────────────────────────────────────────────────┘         │
│                              │                                       │
│                              ▼                                       │
│            ┌─────────────────────────────────┐                      │
│            │        SharedContext            │                      │
│            │  • governance_decisions         │                      │
│            │  • action_backlog               │                      │
│            │  • policy_rules                  │                      │
│            │  • lineage_graph                │                      │
│            └─────────────────────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 출력 데이터

### 9.1 Governance Decision

```json
{
  "decision_id": "decision_obj_customer_001",
  "concept_id": "obj_customer_001",
  "decision_type": "approve",
  "confidence": 0.87,
  "reasoning": "Customer는 핵심 비즈니스 엔티티이며, 두 소스 테이블에서 강한 증거 확인됨",
  "agent_opinions": {
    "ontologist": {"vote": "approve", "reasoning": "논리적으로 일관됨"},
    "risk_assessor": {"vote": "approve", "reasoning": "리스크 낮음"},
    "business_strategist": {"vote": "approve", "reasoning": "비즈니스 가치 높음"},
    "data_steward": {"vote": "approve", "reasoning": "데이터 품질 양호"}
  },
  "recommended_actions": [
    "Customer 360 통합 뷰 구축",
    "Cross-table 참조 무결성 검증"
  ],
  "made_at": "2026-01-27T10:30:00Z"
}
```

### 9.2 Action Item

```json
{
  "action_id": "ACT-001",
  "category": "ontology_update",
  "title": "Customer 360 통합 뷰 구축",
  "status": "scheduled",
  "priority": "critical",
  "dependencies": ["ACT-002"],
  "scheduled_at": "2026-01-26T10:00:00Z"
}
```

---

## 10. 핵심 분석 모듈

| 모듈 | 파일 | 기능 |
|------|------|------|
| Actions Engine | `actions/engine.py` | 액션 생성/실행/스케줄링 |
| CDC Sync Engine | `sync/cdc_engine.py` | 실시간 변경 동기화 |
| Impact Analysis | `lineage/column_lineage.py` | 영향도 분석 |
| Agent Memory | `learning/agent_memory.py` | 학습 시스템 |
| Consensus | `consensus/engine.py` | BFT 합의 엔진 |
| Enhanced Validator | `analysis/enhanced_validator.py` | DS 융합 검증 |

---

## 11. Evidence-Based Debate

### 토론 프로세스

```
1. Proposal: 온톨로지 개념 또는 인사이트 제출
       ↓
2. Initial Vote: 각 에이전트 투표 (approve/reject/review)
       ↓
3. Conflict Check: 충돌 감지
       ↓
4. Challenge Round: 반대 의견 에이전트가 증거 제시
       ↓
5. Algorithm Validation: 통계/알고리즘 근거 검증
       ↓
6. Re-vote: 새로운 증거 반영하여 재투표
       ↓
7. Final Consensus: 최종 결정
```

### 증거 유형

| 증거 유형 | 설명 | 예시 |
|----------|------|------|
| Statistical | 통계적 수치 | p-value < 0.05 |
| Algorithmic | 알고리즘 결과 | TDA Betti numbers, OWL2 추론 |
| Domain | 도메인 지식 | "헬스케어에서 Patient는 핵심 엔티티" |
| Historical | 과거 결정 | "이전 프로젝트에서 유사 패턴 승인" |
| Learned | 학습된 패턴 | AgentMemory에서 검색된 유사 결정 |

---

## 12. Phase Gate 검증

Governance 완료 시 다음 조건 확인:

```python
Phase3Gate:
    - governance_decisions not empty  # 결정 생성됨
    - approval_ratio >= 0.6           # 60% 이상 승인
    - no ESCALATED decisions          # 에스컬레이션 없음
    - completion_signal == [DONE]

    # 추가 검증
    - actions_created >= 1            # 최소 1개 액션 생성
    - impact_analysis_executed        # 영향도 분석 실행됨
    - agent_learning_stored           # 학습 데이터 저장됨
```

---

## 13. 최종 아티팩트 저장

```python
JobLogger.save_artifacts():
    - ontology.json              # 최종 온톨로지
    - insights.json              # 비즈니스 인사이트
    - governance_decisions.json  # 거버넌스 결정
    - actions.json               # 생성된 액션 목록
    - lineage_graph.json         # Lineage 그래프
    - agent_memories.json        # 에이전트 학습 데이터
    - summary.json               # 실행 요약
```

저장 위치: `output/{scenario_name}/`

---

*Document Version: v1.0*
*Last Updated: 2026-01-27*
