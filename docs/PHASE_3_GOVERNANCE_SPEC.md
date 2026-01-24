# Phase 3: Governance Engine 기술 명세

> **역할**: 최종 심판관 (Supreme Court)
> **입력**: Phase 2에서 제안된 온톨로지 및 인사이트
> **출력**: 검증된 거버넌스 결정 및 비즈니스 액션
> **최종 업데이트**: 2026-01-22
> **버전**: v1

---

## 1. 개요

Governance Engine은 시스템의 **최종 관문**입니다. Phase 1, 2에서 생성된 모든 인사이트는 **Agent Council(에이전트 위원회)**의 검증을 통과해야 합니다.

### 핵심 특징
- **BFT 기반 합의**: Byzantine Fault Tolerant 다중 에이전트 합의
- **Dempster-Shafer 융합**: 불확실한 증거의 수학적 결합
- **Evidence-Based Debate**: 증거 기반 토론 프로토콜

---

## 2. 에이전트 구성

### 2.1 Agent Council (4인 위원회)

| 페르소나 | 역할 | 검증 관점 |
|----------|------|----------|
| **Ontologist** | 논리적 일관성 | "이 정의가 모순되지 않는가?" |
| **Risk Assessor** | 리스크 평가 | "이 인사이트에 따라 행동 시 위험은?" |
| **Business Strategist** | 비즈니스 가치 | "이것이 비즈니스 임팩트가 있는가?" |
| **Data Steward** | 데이터 품질 | "데이터 품질과 컴플라이언스 문제는?" |

### 2.2 GovernanceStrategistAutonomousAgent

**역할**: 전략적 의사결정 프레임워크

```python
GovernanceStrategistAutonomousAgent:
    - 거버넌스 정책 프레임워크 설계
    - 리스크/베네핏 분석
    - 조직 정렬 전략 수립
    - 승인/거부/에스컬레이션 결정
```

### 2.3 ActionPrioritizerAutonomousAgent

**역할**: 액션 우선순위 결정

```python
ActionPrioritizerAutonomousAgent:
    - 액션 우선순위화 (criticality, impact, effort)
    - 워크플로우 시퀀싱
    - 의존성 해결
    - KPI 예측
```

### 2.4 RiskAssessorAutonomousAgent

**역할**: 리스크 식별 및 평가

```python
RiskAssessorAutonomousAgent:
    - 리스크 유형 식별 (데이터 품질, 컴플라이언스, 통합)
    - 심각도/가능성 평가
    - 완화 방안 제안
    - 리스크 집계
```

### 2.5 PolicyGeneratorAutonomousAgent

**역할**: 정책 생성

```python
PolicyGeneratorAutonomousAgent:
    - 데이터 거버넌스 정책 초안
    - 접근 제어 규칙 생성
    - 데이터 품질 표준 정의
    - 컴플라이언스 요구사항 명세
```

---

## 3. 합의 프로토콜

### 3.1 BFT-Inspired Consensus Engine (v7.0+)

```python
ConsensusEngine:
    Round 1: Initial Vote
        - 각 에이전트가 approve/reject/review 투표
        - 만장일치 또는 >75% 동의 시 Fast-track 통과

    Round 2: Deep Debate (충돌 시)
        - 반대 의견 에이전트에게 Challenge 발언권
        - 알고리즘 증거 제시
        - 재투표

    Final: Forced Decision
        - Max Rounds 도달 시 Soft Consensus
        - 신뢰도 하향 조정
```

### 3.2 Dempster-Shafer Fusion

불확실한 상황에서 여러 에이전트의 충돌하는 의견을 수학적으로 융합:

```
(m₁ ⊕ m₂)(A) = (1/(1-K)) × Σ m₁(B) × m₂(C)
                          B∩C=A

K = 충돌하는 증거의 양
K가 너무 크면 "판단 보류" 선언
```

### 3.3 Enhanced Validation

```python
EnhancedValidator:
    - DS Confidence: Dempster-Shafer 신뢰도 계산
    - BFT Agreement: Byzantine 합의 비율
    - Conflict Detection: 충돌 증거 탐지
    - Calibrated Confidence: 최종 보정 신뢰도
```

---

## 4. 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                   Phase 3: Governance                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 2 Results                                             │
│  (ontology, insights, predictive_insights)                   │
│       │                                                      │
│       ▼                                                      │
│  ┌───────────────────────────────────────────────────┐      │
│  │              Agent Council (4 Personas)            │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │      │
│  │  │Ontologist│ │  Risk    │ │ Business │ │ Data │ │      │
│  │  │          │ │ Assessor │ │Strategist│ │Steward│ │      │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──┬───┘ │      │
│  │       │            │            │           │      │      │
│  │       └────────────┴────────────┴───────────┘      │      │
│  │                         │                          │      │
│  │                         ▼                          │      │
│  │              ┌─────────────────────┐              │      │
│  │              │   Consensus Engine  │              │      │
│  │              │  (BFT + DS Fusion)  │              │      │
│  │              └──────────┬──────────┘              │      │
│  └─────────────────────────┼─────────────────────────┘      │
│                            │                                 │
│            ┌───────────────┼───────────────┐                │
│            ▼               ▼               ▼                │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│     │ Approved │    │  Review  │    │ Rejected │           │
│     │ Actions  │    │  Queue   │    │   Log    │           │
│     └──────────┘    └──────────┘    └──────────┘           │
│                                                              │
│                            │                                 │
│                            ▼                                 │
│              ┌─────────────────────────┐                    │
│              │    SharedContext        │                    │
│              │  - governance_decisions │                    │
│              │  - action_backlog       │                    │
│              │  - policies             │                    │
│              └─────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 출력 데이터

### 5.1 Governance Decision

```json
{
  "decision_id": "decision_obj_customer_001",
  "concept_id": "obj_customer_001",
  "decision_type": "approve",
  "confidence": 0.87,
  "reasoning": "Customer는 핵심 비즈니스 엔티티이며, 두 소스 테이블에서 강한 증거 확인됨",
  "agent_opinions": {
    "algorithmic": {
      "decision_type": "approve",
      "confidence": 0.72,
      "quality_score": 78.0
    },
    "llm_review": {
      "decision_type": "approve",
      "confidence": 0.85,
      "business_value": "high"
    },
    "enhanced_validation": {
      "ds_decision": "approve",
      "ds_confidence": 0.99,
      "bft_consensus": "approve",
      "bft_agreement": 1.0
    },
    "evidence_based_debate": {
      "verdict": "approve",
      "consensus_type": "unanimous",
      "vote_summary": {"approve": 4, "reject": 0}
    }
  },
  "recommended_actions": [
    {"action_type": "documentation", "priority": "high"},
    {"action_type": "monitoring", "priority": "medium"}
  ]
}
```

### 5.2 Action Item

```json
{
  "action_id": "ACT-001",
  "action_type": "implementation",
  "description": "Customer 360 통합 뷰 구축",
  "priority": "critical",
  "estimated_impact": "high",
  "dependencies": ["ACT-002"],
  "stakeholders": ["Data Engineering", "Analytics Team"]
}
```

---

## 6. 핵심 분석 모듈

| 모듈 | 파일 | 기능 |
|------|------|------|
| Consensus | `consensus/engine.py` | BFT 합의 엔진 |
| Enhanced Validator | `analysis/enhanced_validator.py` | DS 융합 검증 |
| Risk Assessment | `analysis/risk_assessment.py` | 리스크 평가 |
| Priority Calculator | `analysis/priority_calculator.py` | 우선순위 계산 |
| Policy Generator | `analysis/policy_generator.py` | 정책 생성 |
| Governance Matrix | `analysis/governance_decision_matrix.py` | 의사결정 매트릭스 |

---

## 7. Evidence-Based Debate (v11.0+)

### 토론 프로세스

```
1. Proposal: 온톨로지 개념 또는 인사이트 제출
       ↓
2. Initial Vote: 각 에이전트 투표 (approve/reject/review)
       ↓
3. Conflict Check: 충돌 감지 (예: 2 approve, 2 reject)
       ↓
4. Challenge Round: 반대 의견 에이전트가 증거 제시
       ↓
5. Algorithm Validation: 통계/알고리즘 근거 검증
       ↓
6. Re-vote: 새로운 증거 반영하여 재투표
       ↓
7. Final Consensus: 최종 결정 (또는 Soft Consensus)
```

### 증거 유형

| 증거 유형 | 설명 | 예시 |
|----------|------|------|
| Statistical | 통계적 수치 | p-value < 0.05 |
| Algorithmic | 알고리즘 결과 | TDA Betti numbers |
| Domain | 도메인 지식 | "헬스케어에서 Patient는 핵심 엔티티" |
| Historical | 과거 결정 | "이전 프로젝트에서 유사 패턴 승인" |

---

## 8. Phase Gate 검증 (v14.0)

Governance 완료 시 다음 조건 확인:

```python
Phase3Gate:
    - governance_decisions not empty  # 결정 생성됨
    - approval_ratio >= 0.6           # 60% 이상 승인
    - no ESCALATED decisions          # 에스컬레이션 없음
    - completion_signal == [DONE]
```

---

## 9. 최종 아티팩트 저장

```python
JobLogger.save_artifacts():
    - ontology.json              # 최종 온톨로지
    - insights.json              # 비즈니스 인사이트
    - predictive_insights.json   # Cross-Entity 예측 인사이트
    - governance_decisions.json  # 거버넌스 결정
    - summary.json               # 실행 요약
```

저장 위치: `Logs/{JOB_ID}/artifacts/`
