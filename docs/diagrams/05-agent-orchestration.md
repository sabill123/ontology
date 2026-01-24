# 에이전트 오케스트레이션 다이어그램

> **버전**: v14.0
> **최종 업데이트**: 2026-01-19

## 1. 에이전트 계층 구조

```mermaid
graph TB
    subgraph BASE["Base Agent"]
        BA[AutonomousAgent<br/>base.py]
    end

    subgraph PHASE1["Phase 1: Discovery Agents (6)"]
        P1A[Data Analyst<br/>v13.0 LLM-First]
        P1B[TDA Expert<br/>tda_expert]
        P1C[Schema Analyst<br/>schema_analyst]
        P1D[Value Matcher<br/>value_matcher]
        P1E[Entity Classifier<br/>entity_classifier]
        P1F[Relationship Detector<br/>relationship_detector]
    end

    subgraph PHASE2["Phase 2: Refinement Agents (4)"]
        P2A[Ontology Architect<br/>+ Cross-Entity Correlation]
        P2B[Conflict Resolver<br/>conflict_resolver]
        P2C[Quality Judge<br/>quality_judge]
        P2D[Semantic Validator<br/>semantic_validator]
    end

    subgraph PHASE3["Phase 3: Governance Agents (4)"]
        P3A[Governance Strategist<br/>governance_strategist]
        P3B[Action Prioritizer<br/>action_prioritizer]
        P3C[Risk Assessor<br/>risk_assessor]
        P3D[Policy Generator<br/>policy_generator]
    end

    BA --> P1A & P1B & P1C & P1D & P1E & P1F
    BA --> P2A & P2B & P2C & P2D
    BA --> P3A & P3B & P3C & P3D

    style PHASE1 fill:#bbdefb
    style PHASE2 fill:#fff9c4
    style PHASE3 fill:#e1bee7
```

## 2. 14개 에이전트 역할 맵

```mermaid
mindmap
    root((Agent<br/>Ecosystem<br/>v14.0))
        Phase 1 Discovery
            Data Analyst v13.0
                LLM-First Analysis
                Domain Detection
                Column Classification
            TDA Expert
                Betti Numbers
                Persistence Diagrams
                Structural Complexity
            Schema Analyst
                Column Type Analysis
                Key Detection
                Pattern Recognition
            Value Matcher
                Value Overlap
                Cross-Table Matching
                Jaccard Similarity
            Entity Classifier
                Table-to-Entity Mapping
                Type Inference
            Relationship Detector
                FK Discovery
                Semantic Relations
        Phase 2 Refinement
            Ontology Architect
                Hierarchy Design
                Property Schema
                Cross-Entity Correlation v10.0
            Quality Judge
                Completeness Check
                Confidence Assessment
            Semantic Validator
                Logical Consistency
                Inference Validity
            Conflict Resolver
                Naming Conflicts
                Merge/Split Decisions
        Phase 3 Governance
            Governance Strategist
                Business Impact
                Implementation Priority
            Risk Assessor
                Data Quality Risks
                Compliance Risks
            Action Prioritizer
                Action Generation
                Priority Scoring
            Policy Generator
                Quality Policies
                Access Policies
```

## 3. 에이전트 상태 머신

```mermaid
stateDiagram-v2
    [*] --> IDLE: Created

    IDLE --> ASSIGNED: Todo Assigned
    ASSIGNED --> RUNNING: Start Execution

    RUNNING --> WAITING_LLM: Call LLM
    WAITING_LLM --> RUNNING: LLM Response

    RUNNING --> WAITING_CONSENSUS: Join Consensus
    WAITING_CONSENSUS --> RUNNING: Consensus Complete

    RUNNING --> COMPLETED: Task Success
    RUNNING --> FAILED: Task Error

    COMPLETED --> IDLE: Ready for Next
    FAILED --> IDLE: Reset

    IDLE --> STOPPED: Stop Signal
    STOPPED --> [*]

    note right of RUNNING: AgentTerminateMode
    note right of COMPLETED: CompletionSignal = DONE
```

## 4. Todo 할당 및 실행 흐름

```mermaid
sequenceDiagram
    participant TM as Todo Manager
    participant AO as Agent Orchestrator
    participant AP as Agent Pool
    participant A1 as Agent 1
    participant A2 as Agent 2
    participant SC as Shared Context
    participant EC as Evidence Chain

    TM->>AO: get_ready_todos()
    AO->>TM: [Todo1, Todo2, Todo3]

    AO->>AO: Check Available Slots
    Note over AO: max_concurrent: 5<br/>active: 2<br/>available: 3

    AO->>AP: find_suitable_agent(Todo1)
    AP-->>AO: Agent1 (data_analyst)

    AO->>AP: find_suitable_agent(Todo2)
    AP-->>AO: Agent2 (tda_expert)

    par Parallel Execution
        AO->>A1: assign_todo(Todo1)
        AO->>A2: assign_todo(Todo2)
    end

    A1->>SC: Read Context
    A1->>A1: Execute LLM Analysis
    A1->>SC: Write Results
    A1->>EC: Add Evidence Block

    A2->>SC: Read Context
    A2->>A2: Execute TDA Analysis
    A2->>SC: Write Results
    A2->>EC: Add Evidence Block

    A1-->>AO: Todo1 Complete (DONE)
    A2-->>AO: Todo2 Complete (DONE)

    AO->>TM: complete_todo(Todo1)
    AO->>TM: complete_todo(Todo2)
```

## 5. 에이전트 풀 관리

```mermaid
flowchart TB
    subgraph CONFIG["Configuration"]
        C1["max_concurrent_agents: 5"]
        C2["agent_idle_timeout: 0"]
        C3["auto_scale: true"]
    end

    subgraph POOL["Agent Pool"]
        direction LR
        A1[Data Analyst<br/>RUNNING]
        A2[TDA Expert<br/>RUNNING]
        A3[Schema Analyst<br/>IDLE]
        A4[Value Matcher<br/>IDLE]
        A5[Entity Classifier<br/>WAITING]
    end

    subgraph QUEUE["Todo Queue (DAG)"]
        Q1[TDA Analysis<br/>Ready]
        Q2[Schema Analysis<br/>Ready]
        Q3[Entity Classification<br/>Blocked]
        Q4[Relationship Detection<br/>Blocked]
    end

    subgraph ASSIGN["Assignment Logic"]
        AS1{Agent<br/>Available?}
        AS2{Type<br/>Match?}
        AS3{Phase<br/>Match?}
        AS4[Assign Todo]
        AS5[Wait for<br/>Dependencies]
    end

    QUEUE --> AS1
    AS1 -->|Yes| AS2
    AS1 -->|No| AS5
    AS2 -->|Yes| AS4
    AS2 -->|No| AS3
    AS3 -->|Yes| AS4
    AS3 -->|No| AS5
    AS4 --> POOL

    style A1 fill:#ffecb3
    style A2 fill:#ffecb3
    style A3 fill:#c8e6c9
    style A4 fill:#c8e6c9
    style A5 fill:#bbdefb
```

## 6. Phase별 에이전트 배치

```mermaid
graph LR
    subgraph PHASES["Pipeline Phases"]
        P1[Phase 1<br/>Discovery]
        P2[Phase 2<br/>Refinement]
        P3[Phase 3<br/>Governance]
    end

    subgraph AGENTS1["Discovery Agents (6)"]
        A1[data_analyst]
        A2[tda_expert]
        A3[schema_analyst]
        A4[value_matcher]
        A5[entity_classifier]
        A6[relationship_detector]
    end

    subgraph AGENTS2["Refinement Agents (4)"]
        B1[ontology_architect]
        B2[conflict_resolver]
        B3[quality_judge]
        B4[semantic_validator]
    end

    subgraph AGENTS3["Governance Agents (4)"]
        C1[governance_strategist]
        C2[action_prioritizer]
        C3[risk_assessor]
        C4[policy_generator]
    end

    P1 --> AGENTS1
    P2 --> AGENTS2
    P3 --> AGENTS3

    style P1 fill:#bbdefb
    style P2 fill:#fff9c4
    style P3 fill:#e1bee7
```

## 7. 에이전트 간 통신 패턴

```mermaid
flowchart TB
    subgraph DIRECT["Direct Communication"]
        AG1[Agent] -->|call_llm| LLM[LLM Service]
        LLM -->|response| AG1
    end

    subgraph SHARED["Shared Context"]
        AG2[Agent A] -->|write| SC[(Shared Context)]
        SC -->|read| AG3[Agent B]
    end

    subgraph EVIDENCE["Evidence Chain"]
        AG4[Agent 1] -->|add_block| EC[Evidence Chain]
        EC -->|verify| AG5[Agent 2]
    end

    subgraph CONSENSUS["Consensus Communication"]
        AG6[Agent] -->|opinion| CE[Consensus Engine]
        CE -->|aggregate| CE
        CE -->|result| AG6
    end

    style DIRECT fill:#e3f2fd
    style SHARED fill:#e8f5e9
    style EVIDENCE fill:#fff3e0
    style CONSENSUS fill:#fce4ec
```

## 8. 에이전트 실행 타임라인

```mermaid
gantt
    title Agent Execution Timeline (v14.0)
    dateFormat  HH:mm:ss
    axisFormat %H:%M:%S

    section Phase 1
    Data Analyst (LLM)   :a1, 00:00:00, 40s
    TDA Expert           :a2, 00:00:05, 30s
    Schema Analyst       :a3, 00:00:10, 25s
    Value Matcher        :a4, after a2, 20s
    Entity Classifier    :a5, after a3 a4, 15s
    Relationship Detector:a6, after a5, 20s

    section Phase 2
    Ontology Architect   :b1, after a6, 35s
    Cross-Entity Corr    :b2, after b1, 30s
    Quality Judge        :b3, after b2, 25s
    Conflict Resolver    :b4, after b3, 20s
    Semantic Validator   :b5, after b4, 15s
    Smart Consensus      :b6, after b5, 30s

    section Phase 3
    Governance Strategist:c1, after b6, 25s
    Risk Assessor        :c2, after c1, 20s
    Action Prioritizer   :c3, after c2, 20s
    Policy Generator     :c4, after c3, 15s
    BFT Consensus        :c5, after c4, 35s
```

## 9. Agent Terminate Mode (v14.0)

```mermaid
flowchart TB
    subgraph MODES["AgentTerminateMode"]
        M1["PHASE_COMPLETE<br/>현재 Phase 완료"]
        M2["TASK_DONE<br/>단일 태스크 완료"]
        M3["CONSENSUS_REACHED<br/>합의 도달"]
        M4["ERROR_RECOVERY<br/>오류 복구"]
    end

    subgraph SIGNAL["CompletionSignal"]
        S1["[DONE]<br/>정상 완료"]
        S2["[BLOCKED]<br/>의존성 대기"]
        S3["[FALLBACK]<br/>폴백 적용"]
    end

    subgraph ACTION["Post-Terminate Action"]
        A1["Phase Gate 검증"]
        A2["다음 Phase 시작"]
        A3["Artifact 저장"]
    end

    M1 --> S1
    M2 --> S1
    M3 --> S1
    M4 --> S3

    S1 --> A1
    A1 --> A2
    A2 --> A3

    style MODES fill:#e3f2fd
    style SIGNAL fill:#fff3e0
    style ACTION fill:#c8e6c9
```

## 10. Data Analyst v13.0 상세

```mermaid
flowchart TB
    subgraph INPUT["Input"]
        I1["전체 테이블 데이터"]
        I2["샘플 데이터 (100행)"]
    end

    subgraph LLM_ANALYSIS["LLM-First Analysis"]
        L1["도메인 식별<br/>supply_chain, healthcare, etc."]
        L2["컬럼 비즈니스 특성 분류"]
        L3["의미론적 데이터 품질 평가"]
    end

    subgraph COLUMN_TYPES["Column Classification"]
        C1["identifier: 고유 식별자"]
        C2["required_input: 필수 입력"]
        C3["metric: 측정 지표"]
        C4["dimension: 분석 차원"]
        C5["timestamp: 시간 관련"]
    end

    subgraph OUTPUT["Output"]
        O1["domain_context"]
        O2["column_profiles"]
        O3["data_quality_assessment"]
    end

    INPUT --> LLM_ANALYSIS
    L1 --> L2 --> L3
    L2 --> COLUMN_TYPES
    C1 & C2 & C3 & C4 & C5 --> OUTPUT

    style LLM_ANALYSIS fill:#e3f2fd
    style COLUMN_TYPES fill:#fff3e0
    style OUTPUT fill:#c8e6c9
```
