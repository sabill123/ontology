# 에이전트 오케스트레이션 다이어그램

> **버전**: v1.0
> **최종 업데이트**: 2026-01-27

## 1. 에이전트 계층 구조 (v1.0)

```mermaid
graph TB
    subgraph BASE["Base Agent"]
        BA[AutonomousAgent<br/>base.py]
    end

    subgraph PHASE1["Phase 1: Discovery Agents (6)"]
        P1A[Data Analyst<br/>data_analyst]
        P1B[TDA Expert<br/>tda_expert]
        P1C[Schema Analyst<br/>schema_analyst]
        P1D[Value Matcher<br/>value_matcher]
        P1E[Entity Classifier<br/>entity_classifier]
        P1F[Relationship Detector<br/>v1.0 Enhanced FK]
    end

    subgraph PHASE2["Phase 2: Refinement Agents (4)"]
        P2A[Ontology Architect<br/>+ OWL2 Reasoner]
        P2B[Conflict Resolver<br/>conflict_resolver]
        P2C[Quality Judge<br/>+ SHACL Validator]
        P2D[Semantic Validator<br/>semantic_validator]
    end

    subgraph PHASE3["Phase 3: Governance Agents (4)"]
        P3A[Governance Strategist<br/>+ Actions Engine]
        P3B[Action Prioritizer<br/>+ Priority Scoring]
        P3C[Risk Assessor<br/>+ Impact Analysis]
        P3D[Policy Generator<br/>+ CDC Sync]
    end

    subgraph BUS["Agent Communication Bus v1.0"]
        AB[AgentCommunicationBus]
    end

    BA --> P1A & P1B & P1C & P1D & P1E & P1F
    BA --> P2A & P2B & P2C & P2D
    BA --> P3A & P3B & P3C & P3D

    P1F & P2A & P3A --> AB

    style PHASE1 fill:#bbdefb
    style PHASE2 fill:#fff9c4
    style PHASE3 fill:#e1bee7
    style BUS fill:#b2dfdb
```

## 2. 14개 에이전트 역할 맵 (v1.0)

```mermaid
mindmap
    root((Agent<br/>Ecosystem<br/>v1.0))
        Phase 1 Discovery
            Data Analyst
                LLM-First Profiling
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
            Relationship Detector v1.0
                Enhanced FK Pipeline
                Composite FK
                Hierarchy Detection
                Temporal FK
        Phase 2 Refinement
            Ontology Architect
                Hierarchy Design
                Property Schema
                OWL2 Reasoner v1.0
                Cross-Entity Correlation
            Quality Judge
                Completeness Check
                Confidence Assessment
                SHACL Validator v1.0
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
                Actions Engine v1.0
            Risk Assessor
                Data Quality Risks
                Compliance Risks
                Impact Analysis v1.0
            Action Prioritizer
                Action Generation
                Priority Scoring
            Policy Generator
                Quality Policies
                Access Policies
                CDC Sync v1.0
```

## 3. 에이전트 상태 머신

```mermaid
stateDiagram-v2
    [*] --> IDLE: Created

    IDLE --> ASSIGNED: Todo Assigned
    ASSIGNED --> RUNNING: Start Execution

    RUNNING --> WAITING_LLM: Call LLM
    WAITING_LLM --> RUNNING: LLM Response

    RUNNING --> WAITING_BUS: AgentBus Request v1.0
    WAITING_BUS --> RUNNING: Response Received

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
    note right of WAITING_BUS: AgentBus v1.0
```

## 4. Todo 할당 및 실행 흐름 (v1.0)

```mermaid
sequenceDiagram
    participant TM as Todo Manager
    participant AO as Agent Orchestrator
    participant BUS as AgentBus v1.0
    participant AP as Agent Pool
    participant A1 as Agent 1
    participant A2 as Agent 2
    participant SC as Shared Context
    participant LIN as Lineage Tracker
    participant EC as Evidence Chain

    TM->>AO: get_ready_todos()
    AO->>TM: [Todo1, Todo2, Todo3]

    AO->>AO: Check Available Slots
    Note over AO: max_concurrent: 5<br/>active: 2<br/>available: 3

    AO->>AP: find_suitable_agent(Todo1)
    AP-->>AO: Agent1 (tda_expert)

    AO->>AP: find_suitable_agent(Todo2)
    AP-->>AO: Agent2 (relationship_detector)

    par Parallel Execution
        AO->>A1: assign_todo(Todo1)
        AO->>A2: assign_todo(Todo2)
    end

    A1->>SC: Read Context
    A1->>A1: Execute TDA Analysis
    A1->>SC: Write Results
    A1->>EC: Add Evidence Block

    A2->>SC: Read Context
    A2->>A2: Execute FK Detection v1.0
    A2->>BUS: publish("discovery.fk_detected", candidates)
    A2->>SC: Write Results
    A2->>LIN: Track Column Lineage
    A2->>EC: Add Evidence Block

    A1-->>AO: Todo1 Complete (DONE)
    A2-->>AO: Todo2 Complete (DONE)

    AO->>TM: complete_todo(Todo1)
    AO->>TM: complete_todo(Todo2)
```

## 5. 에이전트 풀 관리 (v1.0)

```mermaid
flowchart TB
    subgraph CONFIG["Configuration"]
        C1["max_concurrent_agents: 5"]
        C2["agent_idle_timeout: 0"]
        C3["auto_scale: true"]
        C4["agentbus_enabled: true v1.0"]
    end

    subgraph POOL["Agent Pool"]
        direction LR
        A1[TDA Expert<br/>RUNNING]
        A2[Relationship Detector<br/>RUNNING<br/>v1.0 Enhanced]
        A3[Schema Analyst<br/>IDLE]
        A4[Ontology Architect<br/>IDLE<br/>+ OWL2]
        A5[Quality Judge<br/>WAITING<br/>+ SHACL]
    end

    subgraph QUEUE["Todo Queue (DAG)"]
        Q1[TDA Analysis<br/>Ready]
        Q2[Enhanced FK Detection<br/>Ready<br/>v1.0]
        Q3[OWL2 Reasoning<br/>Blocked<br/>v1.0]
        Q4[SHACL Validation<br/>Blocked<br/>v1.0]
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
    style A2 fill:#c8e6c9
    style A3 fill:#c8e6c9
    style A4 fill:#f3e5f5
    style A5 fill:#bbdefb
```

## 6. Phase별 에이전트 배치 (v1.0)

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
        A6[relationship_detector<br/>v1.0 Enhanced]
    end

    subgraph AGENTS2["Refinement Agents (4)"]
        B1[ontology_architect<br/>+ OWL2 Reasoner]
        B2[conflict_resolver]
        B3[quality_judge<br/>+ SHACL Validator]
        B4[semantic_validator]
    end

    subgraph AGENTS3["Governance Agents (4)"]
        C1[governance_strategist<br/>+ Actions Engine]
        C2[action_prioritizer]
        C3[risk_assessor<br/>+ Impact Analysis]
        C4[policy_generator<br/>+ CDC Sync]
    end

    P1 --> AGENTS1
    P2 --> AGENTS2
    P3 --> AGENTS3

    style P1 fill:#bbdefb
    style P2 fill:#fff9c4
    style P3 fill:#e1bee7
    style A6 fill:#c8e6c9
    style B1 fill:#f3e5f5
    style B3 fill:#f3e5f5
    style C1 fill:#e1bee7
```

## 7. 에이전트 간 통신 패턴 (v1.0 AgentBus)

```mermaid
flowchart TB
    subgraph DIRECT["Direct Messaging"]
        AG1[Agent A] -->|send_to| BUS1[AgentBus]
        BUS1 -->|deliver| AG2[Agent B]
    end

    subgraph BROADCAST["Broadcast"]
        AG3[Agent] -->|broadcast| BUS2[AgentBus]
        BUS2 -->|deliver| AG4[All Agents]
    end

    subgraph REQRES["Request/Response"]
        AG5[Agent A] -->|request| BUS3[AgentBus]
        BUS3 -->|deliver + wait| AG6[Agent B]
        AG6 -->|respond| BUS3
        BUS3 -->|return| AG5
    end

    subgraph PUBSUB["Pub/Sub Topics"]
        AG7[Publisher] -->|publish topic| BUS4[AgentBus]
        BUS4 -->|notify| AG8[Subscriber 1]
        BUS4 -->|notify| AG9[Subscriber 2]
    end

    subgraph SHARED["Shared Context (Legacy)"]
        AG10[Agent A] -->|write| SC[(Shared Context)]
        SC -->|read| AG11[Agent B]
    end

    style DIRECT fill:#e3f2fd
    style BROADCAST fill:#b2dfdb
    style REQRES fill:#fff9c4
    style PUBSUB fill:#c8e6c9
    style SHARED fill:#f3e5f5
```

## 8. AgentBus 통합 상세 (v1.0)

```mermaid
sequenceDiagram
    participant RD as Relationship Detector
    participant BUS as AgentBus
    participant OA as Ontology Architect
    participant QJ as Quality Judge
    participant CE as Consensus Engine

    Note over RD,CE: Topic Subscriptions
    OA->>BUS: subscribe("discovery.fk", callback)
    QJ->>BUS: subscribe("validation.request", callback)

    Note over RD,CE: Discovery → Refinement
    RD->>RD: detect_enhanced_fk()
    RD->>BUS: publish("discovery.fk", candidates)
    BUS->>OA: callback(message)

    Note over RD,CE: Refinement Request/Response
    OA->>OA: run_owl2_reasoner()
    OA->>BUS: request(QJ, validation_req, timeout=30)
    BUS->>QJ: deliver(requires_response=true)
    QJ->>QJ: run_shacl_validator()
    QJ->>BUS: respond(original_msg, result)
    BUS->>OA: return(response)

    Note over RD,CE: Governance Consensus
    CE->>BUS: broadcast(vote_request)
    RD->>BUS: send_to(CE, vote)
    OA->>BUS: send_to(CE, vote)
    QJ->>BUS: send_to(CE, vote)

    CE->>BUS: publish("consensus.result", decision)
```

## 9. 에이전트 실행 타임라인 (v1.0)

```mermaid
gantt
    title Agent Execution Timeline (v1.0)
    dateFormat  HH:mm:ss
    axisFormat %H:%M:%S

    section Phase 1
    TDA Expert           :a2, 00:00:00, 30s
    Schema Analyst       :a3, 00:00:05, 25s
    Value Matcher        :a4, after a2, 20s
    Entity Classifier    :a5, after a3 a4, 15s
    Relationship Detector v1.0 :a6, after a5, 25s

    section Phase 2
    Ontology Architect + OWL2 :b1, after a6, 40s
    SHACL Validation     :b2, after b1, 25s
    Cross-Entity Corr    :b3, after b2, 30s
    Quality Judge        :b4, after b3, 25s
    Conflict Resolver    :b5, after b4, 20s
    Smart Consensus      :b6, after b5, 30s

    section Phase 3
    Governance Strategist :c1, after b6, 25s
    Actions Engine v1.0 :c2, after c1, 20s
    Risk + Impact Analysis :c3, after c2, 25s
    CDC Sync v1.0       :c4, after c3, 20s
    Lineage Tracking     :c5, after c4, 15s
    BFT Consensus        :c6, after c5, 35s
```

## 10. Agent Terminate Mode (v1.0)

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

    subgraph ACTION["Post-Terminate Action v1.0"]
        A1["Phase Gate 검증"]
        A2["다음 Phase 시작"]
        A3["Actions Engine 실행"]
        A4["Lineage 업데이트"]
        A5["Artifact 저장"]
    end

    M1 --> S1
    M2 --> S1
    M3 --> S1
    M4 --> S3

    S1 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5

    style MODES fill:#e3f2fd
    style SIGNAL fill:#fff3e0
    style ACTION fill:#c8e6c9
```

## 11. v1.0 Agent-Module 연결 맵

```mermaid
graph TB
    subgraph DISCOVERY["Phase 1 Agents"]
        RD[Relationship Detector]
    end

    subgraph MODULES_P1["v1.0 Analysis Modules"]
        EFP[Enhanced FK Pipeline]
        CFK[Composite FK Detector]
        HFD[Hierarchy Detector]
        TFK[Temporal FK Detector]
    end

    subgraph REFINEMENT["Phase 2 Agents"]
        OA[Ontology Architect]
        QJ[Quality Judge]
    end

    subgraph MODULES_P2["v1.0 Semantic Modules"]
        OWL[OWL2 Reasoner]
        SHACL[SHACL Validator]
    end

    subgraph GOVERNANCE["Phase 3 Agents"]
        GS[Governance Strategist]
        AP[Action Prioritizer]
        RA[Risk Assessor]
        PG[Policy Generator]
    end

    subgraph MODULES_P3["v1.0 Enterprise Modules"]
        ACT[Actions Engine]
        CDC[CDC Sync Engine]
        LIN[Lineage Tracker]
        MEM[Agent Memory]
    end

    RD --> EFP --> CFK & HFD & TFK
    OA --> OWL
    QJ --> SHACL
    GS --> ACT
    AP --> ACT
    RA --> LIN
    PG --> CDC
    MEM --> RD & OA & GS

    style MODULES_P1 fill:#c8e6c9
    style MODULES_P2 fill:#f3e5f5
    style MODULES_P3 fill:#e1bee7
```

## 12. TDA Expert 에이전트 상세

```mermaid
flowchart TB
    subgraph INPUT["Input"]
        I1["전체 테이블 데이터"]
        I2["컬럼 스키마 정보"]
    end

    subgraph TDA_ANALYSIS["TDA Analysis"]
        L1["Betti Numbers 계산<br/>β₀, β₁, β₂"]
        L2["Persistence Diagram 생성"]
        L3["Structural Complexity 평가"]
    end

    subgraph HOMEOMORPHISM["Homeomorphism Detection"]
        C1["confidence: 위상적 동형 신뢰도"]
        C2["homeomorphism_type: 유형"]
        C3["structural_score: 구조 점수"]
        C4["value_score: 값 점수"]
        C5["semantic_score: 의미 점수"]
    end

    subgraph AGENTBUS["AgentBus Communication v1.0"]
        AB1["publish('discovery.tda_complete', signatures)"]
    end

    subgraph OUTPUT["Output"]
        O1["tda_signatures"]
        O2["homeomorphism_pairs"]
        O3["structural_complexity"]
    end

    INPUT --> TDA_ANALYSIS
    L1 --> L2 --> L3
    L3 --> HOMEOMORPHISM
    C1 & C2 & C3 & C4 & C5 --> AGENTBUS
    AGENTBUS --> OUTPUT

    style TDA_ANALYSIS fill:#e3f2fd
    style HOMEOMORPHISM fill:#fff3e0
    style AGENTBUS fill:#b2dfdb
    style OUTPUT fill:#c8e6c9
```
