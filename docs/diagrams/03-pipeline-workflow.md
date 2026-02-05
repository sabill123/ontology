# 파이프라인 워크플로우 다이어그램

> **버전**: v1.0
> **최종 업데이트**: 2026-01-27

## 1. 전체 파이프라인 흐름

```mermaid
graph TB
    subgraph INIT["Initialization"]
        START([Start Pipeline])
        LOAD[Load Table Data<br/>CSV/JSON/Parquet]
        DOMAIN[Domain Detection<br/>LLM-First v13.0]
        CTX[Initialize SharedContext]
        BUS[Initialize AgentBus v1.0]
    end

    subgraph PHASE1["Phase 1: Discovery"]
        direction TB
        D1[Data Analysis<br/>LLM-First Profiling]
        D2[TDA Signature Computation]
        D3[Enhanced FK Pipeline v1.0]
        D4[Advanced FK Detection<br/>Composite/Hierarchy/Temporal]
        D5[Entity Classification]
        D6[Relationship Detection]

        D1 --> D2
        D2 --> D3
        D3 --> D4
        D2 & D4 --> D5
        D5 --> D6
    end

    subgraph PHASE2["Phase 2: Refinement"]
        direction TB
        R1[Ontology Architecture Design]
        R2[OWL2 Reasoning v1.0]
        R3[SHACL Validation v1.0]
        R4[Cross-Entity Correlation v10.0]
        R5{Quality Assessment}
        R6[Conflict Resolution]
        R7{Smart Consensus v8.2}

        R1 --> R2
        R2 --> R3
        R3 --> R4
        R4 --> R5
        R5 -->|Pass| R7
        R5 -->|Fail| R6
        R6 --> R7
    end

    subgraph PHASE3["Phase 3: Governance"]
        direction TB
        G1[Agent Council<br/>4 Personas]
        G2[BFT Consensus<br/>+ DS Fusion]
        G3[Actions Engine v1.0]
        G4[CDC Sync Engine v1.0]
        G5[Lineage Tracking v1.0]
        G6[Final Decision]
        G7[Save Artifacts]

        G1 --> G2
        G2 --> G3
        G3 --> G4
        G4 --> G5
        G5 --> G6
        G6 --> G7
    end

    subgraph OUTPUT["Output"]
        RESULT([Pipeline Complete])
        O_FK[(foreign_keys)]
        O_IFK[(indirect_fks)]
        O_ONT[(ontology)]
        O_TDA[(tda_analysis)]
        O_ENT[(entities)]
        O_INS[(insights)]
        O_COR[(correlations)]
        O_GOV[(governance)]
        O_EC[(evidence_chain)]
        O_KG[(knowledge_graph)]
        O_LLM[(llm_analysis)]
        O_DOM[(domain)]
    end

    START --> LOAD
    LOAD --> DOMAIN
    DOMAIN --> CTX
    CTX --> BUS
    BUS --> PHASE1
    PHASE1 --> PHASE2
    PHASE2 --> PHASE3
    PHASE3 --> O_FK & O_IFK & O_ONT & O_TDA & O_ENT & O_INS & O_COR & O_GOV & O_EC & O_KG & O_LLM & O_DOM
    O_FK & O_IFK & O_ONT & O_TDA & O_ENT & O_INS & O_COR & O_GOV & O_EC & O_KG & O_LLM & O_DOM --> RESULT

    style PHASE1 fill:#bbdefb
    style PHASE2 fill:#fff9c4
    style PHASE3 fill:#e1bee7
    style BUS fill:#b2dfdb
```

## 2. Phase 1: Discovery 상세 워크플로우 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["Input Tables"]
        T1[Table A]
        T2[Table B]
        T3[Table C]
    end

    subgraph TDA["TDA Analysis"]
        TDA1["Betti Numbers 계산<br/>β₀, β₁, β₂"]
        TDA2["Persistence Diagram"]
        TDA3["Structural Complexity"]
        TDA4["Homeomorphism Detection"]
    end

    subgraph FK["Enhanced FK Pipeline v1.0"]
        subgraph STAGE1["Multi-Signal Scoring"]
            FK1["Value Overlap (0.25)"]
            FK2["Cardinality (0.20)"]
            FK3["Referential Integrity (0.25)"]
            FK4["Naming Semantics (0.15)"]
            FK5["Structural Pattern (0.15)"]
        end

        subgraph STAGE2["Direction Analysis"]
            FK6["Uniqueness Evidence"]
            FK7["Containment Evidence"]
            FK8["Entity Reference Evidence"]
        end

        subgraph STAGE3["Semantic Resolution"]
            FK9["Synonym Groups"]
            FK10["Abbreviation Expansion"]
        end
    end

    subgraph ADVANCED["Advanced FK v1.0"]
        CFK["Composite FK Detector<br/>Multi-column keys"]
        HFD["Hierarchy Detector<br/>Self-referencing tables"]
        TFK["Temporal FK Detector<br/>Time-based relations"]
    end

    subgraph ENTITY["Entity Extraction"]
        E1["Table → Entity Mapping"]
        E2["Column → Property Mapping"]
        E3["FK → Relationship Mapping"]
        E4["Hierarchical Relationships"]
        E5["Temporal Relationships"]
        E6["Unified Entities"]
    end

    INPUT --> TDA & FK

    TDA1 --> TDA2 --> TDA3 --> TDA4
    FK1 & FK2 & FK3 & FK4 & FK5 --> FK6 & FK7 & FK8
    FK6 & FK7 & FK8 --> FK9 & FK10

    FK --> ADVANCED
    TDA4 & ADVANCED --> E1
    E1 --> E2 --> E3 --> E4 --> E5 --> E6

    style TDA fill:#fff3e0
    style FK fill:#c8e6c9
    style ADVANCED fill:#b2dfdb
    style ENTITY fill:#e8f5e9
```

## 3. Phase 2: Refinement 상세 워크플로우 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["From Phase 1"]
        UE[Unified Entities]
        FK[Enhanced FK Candidates<br/>Simple/Composite/Hierarchical/Temporal]
        TDA[TDA Signatures]
    end

    subgraph ONTO["Ontology Design"]
        O1["Design Concept Hierarchy"]
        O2["Define Property Schemas"]
        O3["Establish Relationship Types"]
        O4["Generate Concepts"]
    end

    subgraph OWL2["OWL2 Reasoning v1.0"]
        OWL1["SubClassOf Axioms"]
        OWL2["EquivalentClass Axioms"]
        OWL3["DisjointClasses Axioms"]
        OWL4["Property Domain/Range"]
        OWL5["Transitive/Inverse Properties"]
        OWL6["Axiom Inference"]
    end

    subgraph SHACL["SHACL Validation v1.0"]
        SH1["NodeShape Generation"]
        SH2["PropertyShape Generation"]
        SH3["Datatype Constraints"]
        SH4["Cardinality Constraints"]
        SH5["Pattern Constraints"]
        SH6["Validation Report"]
    end

    subgraph CEC["Cross-Entity Correlation v10.0"]
        C1["Pearson Correlation Matrix"]
        C2["Complementary Pairs Detection"]
        C3["Duplicate Columns Detection"]
        C4["Threshold Effects Analysis"]
        C5["Segment Concentration Analysis"]
        C6["Predictive Insights Generation"]
    end

    subgraph QUALITY["Quality Assessment"]
        Q1["Completeness Check"]
        Q2["Consistency Validation"]
        Q3["SHACL Conformance"]
        Q4{Score >= 0.7?}
    end

    subgraph CONFLICT["Conflict Resolution"]
        CF1["Naming Conflict Detection"]
        CF2["Semantic Overlap Detection"]
        CF3["LLM Reasoning Resolution"]
        CF4["Merge/Split Concepts"]
    end

    subgraph CONSENSUS["Smart Consensus v8.2"]
        CON1{Consensus Mode?}
        CON2["full_skip<br/>Phase 1"]
        CON3["lightweight<br/>Single LLM"]
        CON4["full_consensus<br/>Multi-Agent + AgentBus"]
        CON5["Consensus Result"]
    end

    INPUT --> ONTO
    O1 --> O2 --> O3 --> O4
    O4 --> OWL2
    OWL1 --> OWL2 --> OWL3 --> OWL4 --> OWL5 --> OWL6
    OWL6 --> SHACL
    SH1 --> SH2 --> SH3 --> SH4 --> SH5 --> SH6
    SH6 --> CEC
    C1 --> C2 --> C3 --> C4 --> C5 --> C6
    C6 --> QUALITY
    Q1 --> Q2 --> Q3 --> Q4
    Q4 -->|Yes| CONSENSUS
    Q4 -->|No| CONFLICT
    CF1 --> CF2 --> CF3 --> CF4
    CF4 --> CONSENSUS

    CON1 -->|Discovery| CON2
    CON1 -->|Few Concepts| CON3
    CON1 -->|Complex| CON4
    CON2 & CON3 & CON4 --> CON5

    style OWL2 fill:#f3e5f5
    style SHACL fill:#f3e5f5
    style CEC fill:#c8e6c9
    style ONTO fill:#e8f5e9
    style QUALITY fill:#fff3e0
    style CONFLICT fill:#ffebee
    style CONSENSUS fill:#fce4ec
```

## 4. Phase 3: Governance 상세 워크플로우 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["From Phase 2"]
        CONCEPTS[Approved Concepts<br/>+ OWL2 Axioms + SHACL Shapes]
        INSIGHTS[Business Insights]
        PI[Predictive Insights]
    end

    subgraph COUNCIL["Agent Council (4 Personas)"]
        A1["Ontologist<br/>논리적 일관성"]
        A2["Risk Assessor<br/>리스크 평가"]
        A3["Business Strategist<br/>비즈니스 가치"]
        A4["Data Steward<br/>데이터 품질"]
    end

    subgraph BFT["BFT Consensus + AgentBus"]
        B1["Round 1: Initial Vote<br/>via AgentBus broadcast"]
        B2{Confidence >= 75%?}
        B3["Fast-track Approval"]
        B4["Round 2: Deep Debate"]
        B5["Challenge + Evidence"]
        B6["Re-Vote"]
    end

    subgraph DS["Dempster-Shafer Fusion"]
        DS1["Convert to Mass Function"]
        DS2["Dempster's Rule"]
        DS3["Calculate Conflict (K)"]
        DS4["Final Confidence"]
    end

    subgraph ACTIONS["Actions Engine v1.0"]
        ACT1["Generate Actions<br/>DataAction/GovernanceAction/QualityAction"]
        ACT2["Priority Scoring<br/>Impact × Urgency × Dependency"]
        ACT3["Action Queue"]
    end

    subgraph CDC["CDC Sync v1.0"]
        CDC1["Change Detection"]
        CDC2["Delta Calculation"]
        CDC3["Schema Mapping"]
        CDC4["Sync Execution"]
    end

    subgraph LINEAGE["Lineage Tracking v1.0"]
        LIN1["Column Node Creation"]
        LIN2["Edge Creation<br/>DIRECT/DERIVED/AGGREGATED"]
        LIN3["Impact Analysis<br/>Upstream/Downstream"]
    end

    subgraph DECISION["Final Decision"]
        D1["APPROVED<br/>conf >= 0.75"]
        D2["PROVISIONAL<br/>conf >= 0.50"]
        D3["REJECTED<br/>conf < 0.30"]
        D4["ESCALATED<br/>Human Review"]
    end

    subgraph OUTPUT["Output"]
        OUT1[(governance_decisions.json)]
        OUT2[(actions.json)]
        OUT3[(lineage.json)]
    end

    INPUT --> COUNCIL
    A1 & A2 & A3 & A4 --> BFT
    B1 --> B2
    B2 -->|Yes| B3
    B2 -->|No| B4
    B4 --> B5 --> B6
    B3 & B6 --> DS
    DS1 --> DS2 --> DS3 --> DS4
    DS4 --> DECISION
    DECISION --> ACTIONS
    ACT1 --> ACT2 --> ACT3
    ACT3 --> CDC
    CDC1 --> CDC2 --> CDC3 --> CDC4
    CDC4 --> LINEAGE
    LIN1 --> LIN2 --> LIN3
    LIN3 --> OUTPUT
    D1 & D2 & D3 & D4 --> OUTPUT

    style COUNCIL fill:#e8f5e9
    style BFT fill:#fff3e0
    style DS fill:#e3f2fd
    style ACTIONS fill:#e1bee7
    style CDC fill:#ffccbc
    style LINEAGE fill:#b2dfdb
    style DECISION fill:#f3e5f5
```

## 5. Todo DAG 워크플로우 (v1.0)

```mermaid
graph TD
    T1[TDA Analysis]
    T2[Schema Analysis]
    T3[Value Matching]
    T4[Enhanced FK Detection v1.0]
    T5[Composite FK Detection]
    T6[Hierarchy Detection]
    T7[Temporal FK Detection]
    T8[Entity Classification]
    T9[Relationship Detection]
    T10[OWL2 Reasoning]
    T11[SHACL Validation]
    T12[Cross-Entity Correlation]
    T13[Quality Assessment]
    T14[Actions Generation]
    T15[CDC Sync]
    T16[Lineage Tracking]
    T17[Governance Decision]

    T1 --> T8
    T2 --> T4
    T3 --> T4
    T4 --> T5 & T6 & T7
    T5 & T6 & T7 --> T8
    T8 --> T9
    T9 --> T10
    T10 --> T11
    T11 --> T12
    T12 --> T13
    T13 --> T14
    T14 --> T15
    T15 --> T16
    T16 --> T17

    style T1 fill:#bbdefb
    style T2 fill:#bbdefb
    style T3 fill:#bbdefb
    style T4 fill:#c8e6c9
    style T5 fill:#b2dfdb
    style T6 fill:#b2dfdb
    style T7 fill:#b2dfdb
    style T8 fill:#bbdefb
    style T9 fill:#bbdefb
    style T10 fill:#f3e5f5
    style T11 fill:#f3e5f5
    style T12 fill:#c8e6c9
    style T13 fill:#fff9c4
    style T14 fill:#e1bee7
    style T15 fill:#ffccbc
    style T16 fill:#b2dfdb
    style T17 fill:#e1bee7
```

## 6. Phase Gate 검증 (v1.0)

```mermaid
flowchart TB
    subgraph P1_GATE["Phase 1 Gate"]
        P1G1{unified_entities >= 1?}
        P1G2{evidence_chain not empty?}
        P1G3{no BLOCKED todos?}
        P1G4{enhanced_fk_candidates validated?}
        P1G5{completion_signal == DONE?}
        P1G6["Pass → Phase 2"]
        P1G7["Fail → Fallback"]
    end

    subgraph P2_GATE["Phase 2 Gate"]
        P2G1{ontology_concepts >= 1?}
        P2G2{owl2_axioms generated?}
        P2G3{shacl_validation passed?}
        P2G4{approved_ratio >= 0.5?}
        P2G5{no critical conflicts?}
        P2G6{completion_signal == DONE?}
        P2G7["Pass → Phase 3"]
        P2G8["Fail → Fallback"]
    end

    subgraph P3_GATE["Phase 3 Gate"]
        P3G1{governance_decisions not empty?}
        P3G2{recommended_actions generated?}
        P3G3{lineage_tracked?}
        P3G4{approval_ratio >= 0.6?}
        P3G5{no ESCALATED decisions?}
        P3G6{completion_signal == DONE?}
        P3G7["Pass → Complete"]
        P3G8["Fail → Human Review"]
    end

    P1G1 --> P1G2 --> P1G3 --> P1G4 --> P1G5
    P1G5 -->|Yes| P1G6
    P1G5 -->|No| P1G7

    P2G1 --> P2G2 --> P2G3 --> P2G4 --> P2G5 --> P2G6
    P2G6 -->|Yes| P2G7
    P2G6 -->|No| P2G8

    P3G1 --> P3G2 --> P3G3 --> P3G4 --> P3G5 --> P3G6
    P3G6 -->|Yes| P3G7
    P3G6 -->|No| P3G8

    style P1_GATE fill:#bbdefb
    style P2_GATE fill:#fff9c4
    style P3_GATE fill:#e1bee7
```

## 7. 함수 호출 흐름 (v1.0)

```mermaid
graph TB
    subgraph DISCOVERY["Phase 1: Discovery"]
        D1[compute_tda_signatures]
        D2[analyze_schema]
        D3[detect_homeomorphisms]
        D4[run_enhanced_fk_pipeline]
        D5[detect_composite_fk]
        D6[detect_hierarchies]
        D7[detect_temporal_fk]
        D8[classify_entities]
        D9[detect_relationships]

        D1 --> D3
        D2 --> D4
        D4 --> D5 & D6 & D7
        D3 --> D8
        D5 & D6 & D7 --> D8
        D8 --> D9
    end

    subgraph REFINEMENT["Phase 2: Refinement"]
        R1[design_ontology]
        R2[run_owl2_reasoner]
        R3[run_shacl_validator]
        R4[run_cross_entity_correlation]
        R5[generate_concepts]
        R6[assess_quality]
        R7[resolve_conflicts]
        R8[run_consensus]

        D9 --> R1
        R1 --> R2
        R2 --> R3
        R3 --> R4
        R4 --> R5
        R5 --> R6
        R6 --> R7
        R7 --> R8
    end

    subgraph GOVERNANCE["Phase 3: Governance"]
        G1[create_governance_strategy]
        G2[run_bft_consensus]
        G3[dempster_shafer_fusion]
        G4[generate_actions]
        G5[run_cdc_sync]
        G6[track_lineage]
        G7[analyze_impact]
        G8[make_final_decision]

        R8 --> G1
        G1 --> G2
        G2 --> G3
        G3 --> G4
        G4 --> G5
        G5 --> G6
        G6 --> G7
        G7 --> G8
    end

    style D1 fill:#bbdefb
    style D4 fill:#c8e6c9
    style D5 fill:#b2dfdb
    style D6 fill:#b2dfdb
    style D7 fill:#b2dfdb
    style D9 fill:#bbdefb
    style R1 fill:#fff9c4
    style R2 fill:#f3e5f5
    style R3 fill:#f3e5f5
    style R4 fill:#c8e6c9
    style R8 fill:#fff9c4
    style G1 fill:#e1bee7
    style G4 fill:#e1bee7
    style G5 fill:#ffccbc
    style G6 fill:#b2dfdb
    style G8 fill:#e1bee7
```

## 8. AgentBus 통합 워크플로우 (v1.0)

```mermaid
sequenceDiagram
    participant AO as Agent Orchestrator
    participant BUS as AgentBus
    participant RD as Relationship Detector
    participant OA as Ontology Architect
    participant QJ as Quality Judge
    participant CE as Consensus Engine

    Note over AO,CE: Agent Registration
    AO->>BUS: register_agent("relationship_detector")
    AO->>BUS: register_agent("ontology_architect")
    AO->>BUS: register_agent("quality_judge")

    Note over AO,CE: Topic Subscriptions
    RD->>BUS: subscribe("validation.request", callback)
    OA->>BUS: subscribe("discovery.fk", callback)
    QJ->>BUS: subscribe("validation.request", callback)

    Note over AO,CE: Discovery Phase
    RD->>RD: detect_enhanced_fk_candidates()
    RD->>BUS: publish("discovery.fk", candidates)
    BUS->>OA: deliver(message)

    Note over AO,CE: Request/Response
    OA->>BUS: request("quality_judge", validation_req)
    BUS->>QJ: deliver(request)
    QJ->>QJ: validate()
    QJ->>BUS: respond(original_msg, result)
    BUS->>OA: deliver(response)

    Note over AO,CE: Consensus with AgentBus
    CE->>BUS: broadcast(vote_request)
    BUS->>RD: deliver
    BUS->>OA: deliver
    BUS->>QJ: deliver
    RD->>BUS: send_to("consensus", vote)
    OA->>BUS: send_to("consensus", vote)
    QJ->>BUS: send_to("consensus", vote)
    CE->>CE: aggregate_votes()
```

## 9. v1.0 Enterprise 워크플로우

```mermaid
flowchart TB
    subgraph ACTIONS["Actions Engine"]
        A1[Generate Action from Decision]
        A2[Classify Action Type<br/>Data/Governance/Quality]
        A3[Calculate Priority<br/>Impact × Urgency × Dependency]
        A4[Queue Action]
        A5[Execute Action]
    end

    subgraph CDC["CDC Sync Engine"]
        C1[Detect Changes<br/>Timestamp/Hash/Log]
        C2[Calculate Delta]
        C3[Map Schema]
        C4[Transform Values]
        C5[Apply to Target]
    end

    subgraph LINEAGE["Lineage Tracker"]
        L1[Create Column Nodes<br/>SOURCE/DERIVED/AGGREGATE]
        L2[Create Edges<br/>DIRECT/DERIVED/AGGREGATED/FILTERED]
        L3[Build Lineage Graph]
        L4[Analyze Upstream Impact]
        L5[Analyze Downstream Impact]
    end

    subgraph OUTPUT["Enterprise Outputs"]
        O1[actions.json]
        O2[sync_log.json]
        O3[lineage.json]
        O4[impact_report.json]
    end

    A1 --> A2 --> A3 --> A4 --> A5
    A5 --> C1
    C1 --> C2 --> C3 --> C4 --> C5
    C5 --> L1
    L1 --> L2 --> L3 --> L4 --> L5
    L5 --> OUTPUT

    style ACTIONS fill:#e1bee7
    style CDC fill:#ffccbc
    style LINEAGE fill:#b2dfdb
```
