# 데이터 흐름 다이어그램

> **버전**: v14.0
> **최종 업데이트**: 2026-01-19

## 1. 전체 데이터 흐름

```mermaid
flowchart TB
    subgraph INPUT["Input Layer"]
        direction LR
        T1[Table 1<br/>CSV/JSON]
        T2[Table 2<br/>CSV/JSON]
        T3[Table 3<br/>CSV/JSON]
        TN[Table N<br/>...]
    end

    subgraph PHASE1["Phase 1: Discovery"]
        direction TB
        P1A[Data Analyst<br/>LLM-First v13.0]
        P1B[TDA Signature<br/>Computation]
        P1C[Schema Analysis<br/>+ FK Detection]
        P1D[Entity<br/>Classification]
        P1E[Relationship<br/>Detection]

        P1A --> P1B
        P1A --> P1C
        P1B --> P1D
        P1C --> P1D
        P1D --> P1E
    end

    subgraph PHASE2["Phase 2: Refinement"]
        direction TB
        P2A[Ontology<br/>Architecture]
        P2B[Cross-Entity<br/>Correlation v10.0]
        P2C[Quality<br/>Assessment]
        P2D[Conflict<br/>Resolution]
        P2E[Smart Consensus<br/>v8.2]

        P2A --> P2B
        P2B --> P2C
        P2C --> P2D
        P2D --> P2E
    end

    subgraph PHASE3["Phase 3: Governance"]
        direction TB
        P3A[Agent Council<br/>4 Personas]
        P3B[BFT Consensus<br/>+ DS Fusion]
        P3C[Evidence-Based<br/>Debate]
        P3D[Final<br/>Decision]

        P3A --> P3B
        P3B --> P3C
        P3C --> P3D
    end

    subgraph OUTPUT["Output Layer"]
        direction LR
        O1[ontology.json]
        O2[insights.json]
        O3[predictive_insights.json]
        O4[governance_decisions.json]
    end

    INPUT --> PHASE1
    PHASE1 --> PHASE2
    PHASE2 --> PHASE3
    PHASE3 --> OUTPUT

    style PHASE1 fill:#bbdefb
    style PHASE2 fill:#fff9c4
    style PHASE3 fill:#e1bee7
```

## 2. SharedContext 데이터 흐름

```mermaid
flowchart LR
    subgraph SC["SharedContext"]
        direction TB
        Tables[(tables)]
        TDA[(tda_signatures)]
        FK[(fk_candidates)]
        Entities[(unified_entities)]
        Concepts[(ontology_concepts)]
        Insights[(business_insights)]
        Predictive[(palantir_insights)]
        Decisions[(governance_decisions)]
        Evidence[(evidence_chain)]
    end

    subgraph P1["Phase 1 Agents"]
        DA[Data Analyst]
        TA[TDA Expert]
        SA[Schema Analyst]
        VM[Value Matcher]
        EC[Entity Classifier]
        RD[Relationship Detector]
    end

    subgraph P2["Phase 2 Agents"]
        OA[Ontology Architect]
        CR[Conflict Resolver]
        QJ[Quality Judge]
        SV[Semantic Validator]
    end

    subgraph P3["Phase 3 Agents"]
        GS[Governance Strategist]
        AP[Action Prioritizer]
        RA[Risk Assessor]
        PG[Policy Generator]
    end

    Tables --> DA & TA & SA
    DA --> TDA
    TA --> TDA
    SA --> FK
    VM --> FK
    EC --> Entities
    RD --> Entities

    Entities --> OA
    OA --> Concepts & Predictive
    CR --> Concepts
    QJ --> Concepts
    SV --> Concepts

    Concepts --> GS
    GS --> Decisions
    AP --> Decisions
    RA --> Decisions
    PG --> Decisions

    Decisions --> Evidence

    style SC fill:#e8f5e9
```

## 3. 데이터 변환 파이프라인

```mermaid
flowchart TB
    subgraph RAW["Raw Table Data"]
        R1["columns: [id, name, ...]"]
        R2["sample_data: [{...}, ...]"]
        R3["row_count: 1000"]
    end

    subgraph TDA["TDA Analysis"]
        T1["Betti Numbers<br/>β₀=1, β₁=2, β₂=0"]
        T2["Persistence Pairs<br/>[(dim, birth, death), ...]"]
        T3["Structural Complexity<br/>low/medium/high"]
    end

    subgraph SCHEMA["Schema Analysis"]
        S1["Column Types<br/>integer, string, date"]
        S2["Key Candidates<br/>id, order_id"]
        S3["FK Relations<br/>supplier_id → suppliers.id"]
    end

    subgraph ENTITY["Entity Extraction"]
        E1["Entity Type<br/>CUSTOMER, ORDER"]
        E2["Properties<br/>{name, amount, ...}"]
        E3["Relationships<br/>ORDERS_FROM, SHIPS_TO"]
    end

    subgraph CONCEPT["Ontology Concept"]
        C1["concept_id: obj_customer_001"]
        C2["type: object_type"]
        C3["confidence: 0.92"]
        C4["status: approved"]
    end

    subgraph INSIGHT["Predictive Insight v10.0"]
        I1["correlation: 0.527"]
        I2["threshold_effect: detected"]
        I3["business_interpretation"]
        I4["recommended_action"]
    end

    RAW --> TDA
    RAW --> SCHEMA
    TDA & SCHEMA --> ENTITY
    ENTITY --> CONCEPT
    CONCEPT --> INSIGHT

    style RAW fill:#ffebee
    style TDA fill:#e3f2fd
    style SCHEMA fill:#fff3e0
    style ENTITY fill:#e8f5e9
    style CONCEPT fill:#f3e5f5
    style INSIGHT fill:#c8e6c9
```

## 4. Cross-Entity Correlation 데이터 흐름 (v10.0)

```mermaid
flowchart TB
    subgraph INPUT["Input Tables"]
        T1[Table A<br/>Numeric Columns]
        T2[Table B<br/>Numeric Columns]
    end

    subgraph ANALYSIS["Correlation Analysis"]
        PC[Pearson Correlation<br/>Matrix]
        CP[Complementary Pairs<br/>sum = 1.0]
        DC[Duplicate Columns<br/>r = 1.0]
        TE[Threshold Effects<br/>Non-linear Detection]
        SA[Segment Analysis<br/>Concentration]
    end

    subgraph OUTPUT["Predictive Insights"]
        PI1["상관관계 인사이트"]
        PI2["데이터 품질 이슈"]
        PI3["임계값 효과"]
        PI4["비즈니스 해석"]
    end

    T1 & T2 --> PC & CP & DC & TE & SA
    PC --> PI1
    CP & DC --> PI2
    TE --> PI3
    PI1 & PI2 & PI3 --> PI4

    style INPUT fill:#bbdefb
    style ANALYSIS fill:#fff9c4
    style OUTPUT fill:#c8e6c9
```

## 5. Evidence Chain 데이터 흐름 (v11.0)

```mermaid
flowchart LR
    subgraph BLOCKS["Evidence Blocks"]
        B1[Block 1<br/>Phase 1 Evidence]
        B2[Block 2<br/>Entity Discovery]
        B3[Block 3<br/>Relationship Detection]
        B4[Block 4<br/>Ontology Design]
        B5[Block 5<br/>Governance Decision]
    end

    subgraph CONTENT["Block Content"]
        C1["evidence_type"]
        C2["source_agent"]
        C3["data"]
        C4["timestamp"]
        C5["prev_hash"]
    end

    B1 -->|hash| B2 -->|hash| B3 -->|hash| B4 -->|hash| B5

    B1 --> C1 & C2 & C3 & C4 & C5

    style B1 fill:#bbdefb
    style B2 fill:#bbdefb
    style B3 fill:#bbdefb
    style B4 fill:#fff9c4
    style B5 fill:#e1bee7
```

## 6. 이벤트 스트림 흐름

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant TM as Todo Manager
    participant A as Agents
    participant SC as SharedContext
    participant CE as Consensus Engine
    participant JL as Job Logger

    P->>TM: initialize_phase(discovery)
    TM->>TM: create_dag_todos()

    loop For Each Todo (DAG Order)
        TM->>A: assign_todo
        A->>SC: read_context()
        A->>A: execute_task()
        A->>SC: write_results()
        A->>TM: complete_todo()
        TM->>JL: log_progress()
    end

    P->>CE: start_consensus()

    loop BFT Rounds
        CE->>A: request_opinion
        A->>CE: submit_vote
        CE->>CE: DS_fusion()
    end

    CE->>SC: save_decision()
    P->>JL: save_artifacts()
```

## 7. 아티팩트 출력 구조

```mermaid
flowchart TB
    subgraph LOGS["Logs/{JOB_ID}/"]
        LOG[pipeline.log]
        SUM[summary.json]

        subgraph ARTIFACTS["artifacts/"]
            ONT[ontology.json]
            INS[insights.json]
            PI[predictive_insights.json]
            GOV[governance_decisions.json]
        end
    end

    subgraph ONT_CONTENT["ontology.json"]
        OC1["entities: [...]"]
        OC2["relationships: [...]"]
        OC3["concepts: [...]"]
    end

    subgraph PI_CONTENT["predictive_insights.json"]
        PC1["correlations: [...]"]
        PC2["threshold_effects: [...]"]
        PC3["business_interpretations: [...]"]
    end

    subgraph GOV_CONTENT["governance_decisions.json"]
        GC1["decisions: [...]"]
        GC2["agent_opinions: {...}"]
        GC3["consensus_result: {...}"]
    end

    ONT --> ONT_CONTENT
    PI --> PI_CONTENT
    GOV --> GOV_CONTENT

    style ARTIFACTS fill:#e8f5e9
    style ONT_CONTENT fill:#bbdefb
    style PI_CONTENT fill:#c8e6c9
    style GOV_CONTENT fill:#e1bee7
```
