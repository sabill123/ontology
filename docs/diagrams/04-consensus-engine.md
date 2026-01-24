# 합의 엔진 다이어그램

> **버전**: v14.0
> **최종 업데이트**: 2026-01-19

## 1. 합의 엔진 개요 (v7.0+)

```mermaid
flowchart TB
    subgraph TRIGGER["Consensus Trigger"]
        T1{Requires<br/>Consensus?}
        T2[quality_assessment]
        T3[ontology_approval]
        T4[governance_decision]
    end

    subgraph SMART["Smart Consensus v8.2"]
        S1{Consensus Mode?}
        S2["full_skip<br/>Phase 1 Discovery"]
        S3["lightweight<br/>Few Concepts / High Quality"]
        S4["full_consensus<br/>Complex Cases"]
    end

    subgraph BFT["BFT Protocol"]
        F1["Round 1:<br/>Initial Vote"]
        F2{Confidence<br/>>= 75%?}
        F3["Fast-track<br/>Approval"]
        F4["Round 2+:<br/>Deep Debate"]
    end

    subgraph DS["Dempster-Shafer Fusion"]
        D1["Mass Function 변환"]
        D2["Dempster's Rule 적용"]
        D3["충돌도(K) 계산"]
        D4["최종 신뢰도 산출"]
    end

    subgraph RESULT["Consensus Result"]
        R1["APPROVED<br/>conf >= 0.75"]
        R2["PROVISIONAL<br/>conf >= 0.50"]
        R3["REJECTED<br/>conf < 0.30"]
        R4["ESCALATED<br/>Human Review"]
    end

    T1 -->|Yes| T2 & T3 & T4
    T2 & T3 & T4 --> SMART
    S1 -->|Phase 1| S2
    S1 -->|Simple| S3
    S1 -->|Complex| S4

    S2 --> R1
    S3 --> R1
    S4 --> BFT

    F1 --> F2
    F2 -->|Yes| F3
    F2 -->|No| F4
    F3 --> DS
    F4 --> DS

    D1 --> D2 --> D3 --> D4
    D4 --> RESULT

    style SMART fill:#e8f5e9
    style BFT fill:#fff3e0
    style DS fill:#e3f2fd
    style RESULT fill:#f3e5f5
```

## 2. BFT 합의 프로세스

```mermaid
sequenceDiagram
    participant Chair as Consensus Chair
    participant A1 as Ontologist
    participant A2 as Risk Assessor
    participant A3 as Business Strategist
    participant A4 as Data Steward
    participant DS as DS Fusion

    Note over Chair: Round 1: Initial Vote

    par Parallel Voting
        Chair->>A1: Request Vote
        Chair->>A2: Request Vote
        Chair->>A3: Request Vote
        Chair->>A4: Request Vote
    end

    A1-->>Chair: APPROVE (conf: 0.85)
    A2-->>Chair: REJECT (reason: Risk)
    A3-->>Chair: APPROVE (conf: 0.90)
    A4-->>Chair: APPROVE (conf: 0.75)

    Note over Chair: Conflict Detected (3:1)

    Chair->>A2: Challenge: Why REJECT?
    A2-->>Chair: Risk evidence presented

    Note over Chair: Round 2: Deep Debate

    Chair->>DS: Validate "Risk" claim
    DS-->>Chair: Risk Score: Low

    Chair->>A2: Challenge with evidence
    A2-->>Chair: APPROVE (revised, conf: 0.60)

    Note over Chair: Re-Vote Complete

    Chair->>DS: Dempster-Shafer Fusion
    DS->>DS: m₁ ⊕ m₂ ⊕ m₃ ⊕ m₄
    DS-->>Chair: Combined conf: 0.82

    Chair-->>Output: APPROVED (0.82)
```

## 3. Dempster-Shafer Fusion 상세

```mermaid
flowchart TB
    subgraph INPUT["Agent Opinions"]
        I1["Agent 1: conf=0.85, doubt=0.10"]
        I2["Agent 2: conf=0.60, doubt=0.30"]
        I3["Agent 3: conf=0.90, doubt=0.05"]
        I4["Agent 4: conf=0.75, doubt=0.15"]
    end

    subgraph MASS["Mass Function 변환"]
        M1["m₁(A) = 0.85<br/>m₁(Θ) = 0.15"]
        M2["m₂(A) = 0.60<br/>m₂(Θ) = 0.40"]
        M3["m₃(A) = 0.90<br/>m₃(Θ) = 0.10"]
        M4["m₄(A) = 0.75<br/>m₄(Θ) = 0.25"]
    end

    subgraph COMBINE["Dempster's Rule"]
        C1["(m₁ ⊕ m₂)(A) = ..."]
        C2["순차적 결합"]
        C3["충돌도 K 계산"]
    end

    subgraph OUTPUT["Final Result"]
        O1["Combined Belief: 0.87"]
        O2["Conflict K: 0.12"]
        O3["Decision: APPROVED"]
    end

    INPUT --> MASS
    M1 & M2 & M3 & M4 --> COMBINE
    C1 --> C2 --> C3
    C3 --> OUTPUT

    style MASS fill:#e3f2fd
    style COMBINE fill:#fff3e0
    style OUTPUT fill:#c8e6c9
```

## 4. Smart Consensus 모드 결정 (v8.2)

```mermaid
flowchart TB
    subgraph CHECK["Mode Decision"]
        C1{Current Phase?}
        C2{Data Quality?}
        C3{Concept Count?}
        C4{Complexity?}
    end

    subgraph MODES["Consensus Modes"]
        M1["full_skip<br/>합의 완전 생략<br/>Phase 1 탐색"]
        M2["lightweight<br/>단일 LLM 검증<br/>소수 개념"]
        M3["full_consensus<br/>다중 에이전트 합의<br/>복잡한 결정"]
    end

    subgraph CRITERIA["Selection Criteria"]
        CR1["Phase 1 → full_skip"]
        CR2["concepts <= 3 → lightweight"]
        CR3["quality >= 0.8 → lightweight"]
        CR4["conflicts > 0 → full_consensus"]
        CR5["concepts > 10 → full_consensus"]
    end

    C1 -->|Phase 1| M1
    C1 -->|Phase 2/3| C2
    C2 -->|High| C3
    C2 -->|Low| M3
    C3 -->|Few| M2
    C3 -->|Many| C4
    C4 -->|Simple| M2
    C4 -->|Complex| M3

    style M1 fill:#c8e6c9
    style M2 fill:#fff9c4
    style M3 fill:#ffcdd2
```

## 5. Evidence-Based Debate (v11.0)

```mermaid
flowchart TB
    subgraph PROPOSAL["Proposal"]
        P1["Ontology Concept 제출"]
        P2["Initial Evidence"]
    end

    subgraph VOTE1["Round 1: Initial Vote"]
        V1["Agent 1: APPROVE"]
        V2["Agent 2: REJECT"]
        V3["Agent 3: APPROVE"]
        V4["Agent 4: ABSTAIN"]
    end

    subgraph CONFLICT["Conflict Detection"]
        CF1{Conflict?}
        CF2["Yes: 충돌 감지"]
        CF3["No: Fast-track"]
    end

    subgraph DEBATE["Challenge Round"]
        D1["반대 의견 에이전트"]
        D2["Challenge 발언"]
        D3["Evidence 제시"]
    end

    subgraph EVIDENCE["Evidence Types"]
        E1["Statistical: p-value"]
        E2["Algorithmic: TDA, Betti"]
        E3["Domain: Business Rule"]
        E4["Historical: Past Decisions"]
    end

    subgraph REVOTE["Re-Vote"]
        R1["새 증거 반영"]
        R2["투표 변경"]
        R3["Convergence Check"]
    end

    subgraph FINAL["Final Decision"]
        F1["Consensus Reached"]
        F2["Soft Consensus"]
        F3["Escalated"]
    end

    PROPOSAL --> VOTE1
    V1 & V2 & V3 & V4 --> CONFLICT
    CF1 -->|Yes| DEBATE
    CF1 -->|No| F1
    D1 --> D2 --> D3
    D3 --> EVIDENCE
    E1 & E2 & E3 & E4 --> REVOTE
    R1 --> R2 --> R3
    R3 --> FINAL

    style DEBATE fill:#fff3e0
    style EVIDENCE fill:#e3f2fd
    style FINAL fill:#c8e6c9
```

## 6. Agent Opinion 데이터 구조

```mermaid
classDiagram
    class AgentOpinion {
        +String agent_id
        +String agent_type
        +String agent_name
        +Float confidence
        +Float uncertainty
        +Float doubt
        +Float evidence_weight
        +String reasoning
        +List key_observations
        +List concerns
        +List suggestions
        +Int round_number
        +VoteType vote
    }

    class VoteType {
        <<enumeration>>
        APPROVE
        REJECT
        ABSTAIN
    }

    class ConsensusResult {
        +Bool success
        +ConsensusType result_type
        +Float combined_confidence
        +Float agreement_level
        +Int total_rounds
        +String final_decision
        +String reasoning
        +List dissenting_views
        +Float conflict_k
    }

    class ConsensusType {
        <<enumeration>>
        APPROVED
        PROVISIONAL
        REJECTED
        ESCALATED
        NEEDS_REVIEW
    }

    AgentOpinion --> VoteType
    ConsensusResult --> ConsensusType
    ConsensusResult o-- AgentOpinion : contains
```

## 7. 합의 결과 임계값

```mermaid
flowchart LR
    subgraph INPUTS["Agent Confidences"]
        I1["Agent 1: 0.85"]
        I2["Agent 2: 0.60"]
        I3["Agent 3: 0.90"]
        I4["Agent 4: 0.75"]
    end

    subgraph CALC["Calculation"]
        C1["DS Fusion"]
        C2["Agreement Level"]
        C3["Combined: 0.82"]
    end

    subgraph THRESHOLDS["Decision Thresholds"]
        T1[">= 0.75: APPROVED"]
        T2[">= 0.50: PROVISIONAL"]
        T3[">= 0.30: NEEDS_REVIEW"]
        T4["< 0.30: REJECTED"]
    end

    subgraph RESULT["Result"]
        R1["APPROVED"]
        R2["Actions Generated"]
    end

    INPUTS --> CALC
    C1 --> C2 --> C3
    C3 --> THRESHOLDS
    T1 --> RESULT

    style CALC fill:#e3f2fd
    style THRESHOLDS fill:#fff3e0
    style RESULT fill:#c8e6c9
```

## 8. Convergence Detection

```mermaid
stateDiagram-v2
    [*] --> Round1: Start Consensus

    Round1 --> FastTrack: Confidence >= 75%
    Round1 --> DeepDebate: Confidence < 75%

    FastTrack --> Approved: Unanimous
    FastTrack --> DeepDebate: Conflict

    DeepDebate --> Challenge: Dissent Raised
    Challenge --> Evidence: Present Arguments
    Evidence --> ReVote: New Evidence
    ReVote --> ConvergenceCheck: Check Stability

    ConvergenceCheck --> EarlyStop: Vote Change < 10%
    ConvergenceCheck --> DeepDebate: Opinions Changed

    EarlyStop --> FinalDecision: Max 2 Stable Rounds

    FinalDecision --> Approved: conf >= 0.75
    FinalDecision --> Provisional: conf >= 0.50
    FinalDecision --> Rejected: conf < 0.30
    FinalDecision --> Escalated: Unresolved

    Approved --> [*]
    Provisional --> [*]
    Rejected --> [*]
    Escalated --> HumanReview
    HumanReview --> [*]
```

## 9. Agent Council 4인 위원회

```mermaid
graph TB
    subgraph COUNCIL["Agent Council"]
        A1["Ontologist<br/>논리적 일관성 검증"]
        A2["Risk Assessor<br/>리스크 평가"]
        A3["Business Strategist<br/>비즈니스 가치 판단"]
        A4["Data Steward<br/>데이터 품질 검증"]
    end

    subgraph QUESTIONS["검증 관점"]
        Q1["이 정의가 모순되지 않는가?"]
        Q2["이 인사이트에 따라 행동 시 위험은?"]
        Q3["이것이 비즈니스 임팩트가 있는가?"]
        Q4["데이터 품질과 컴플라이언스 문제는?"]
    end

    A1 --> Q1
    A2 --> Q2
    A3 --> Q3
    A4 --> Q4

    style A1 fill:#bbdefb
    style A2 fill:#ffcdd2
    style A3 fill:#c8e6c9
    style A4 fill:#fff9c4
```
