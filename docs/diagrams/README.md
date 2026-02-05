# Ontoloty v1.0 다이어그램

> **최종 업데이트**: 2026-01-27
> **버전**: v1.0

## 1. 시스템 개요

```mermaid
graph LR
    subgraph Input
        DS[(Data Sources<br/>CSV/JSON/Parquet)]
    end

    subgraph Pipeline
        P1[Phase 1<br/>Discovery]
        P2[Phase 2<br/>Refinement]
        P3[Phase 3<br/>Governance]
    end

    subgraph V16_Modules["v1.0 Modules"]
        M1[Enhanced FK Pipeline]
        M2[OWL2 Reasoner]
        M3[SHACL Validator]
        M4[Actions Engine]
        M5[CDC Sync]
        M6[Column Lineage]
    end

    subgraph Communication["Agent Communication"]
        BUS[AgentBus v1.0]
    end

    subgraph Output
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

    DS --> P1 --> P2 --> P3 --> O_FK & O_IFK & O_ONT & O_TDA & O_ENT & O_INS & O_COR & O_GOV & O_EC & O_KG & O_LLM & O_DOM
    P1 --> M1
    P2 --> M2 & M3
    P3 --> M4 & M5 & M6
    BUS -.->|pub/sub| P1 & P2 & P3

    style P1 fill:#bbdefb
    style P2 fill:#fff9c4
    style P3 fill:#e1bee7
    style V16_Modules fill:#b2dfdb
    style BUS fill:#c8e6c9
```

## 2. 핵심 컴포넌트

```mermaid
graph TB
    subgraph Orchestration
        AO[Agent Orchestrator]
        TM[Todo Manager<br/>DAG]
        CE[Consensus Engine<br/>BFT + DS]
        BUS[AgentBus v1.0]
    end

    subgraph Analysis_V16["Analysis (v1.0 Enhanced)"]
        TDA[TDA Analyzer]
        FK[Enhanced FK v1.0<br/>Composite+Hierarchy+Temporal]
        CEC[Cross-Entity<br/>Correlation]
        BI[Business Insights]
    end

    subgraph Semantic_V16["Semantic (v1.0)"]
        OWL[OWL2 Reasoner]
        SHACL[SHACL Validator]
    end

    subgraph Enterprise_V16["Enterprise (v1.0)"]
        ACT[Actions Engine]
        CDC[CDC Sync Engine]
        LIN[Column Lineage]
        MEM[Agent Memory]
    end

    subgraph Storage
        SC[Shared Context]
        EC[Evidence Chain]
        KG[Knowledge Graph]
    end

    AO --> TM
    AO --> CE
    AO --> BUS
    TM --> Analysis_V16
    Analysis_V16 --> Semantic_V16
    Semantic_V16 --> Enterprise_V16
    Analysis_V16 --> SC
    SC --> EC
    SC --> KG

    style FK fill:#c8e6c9
    style BUS fill:#b2dfdb
    style Semantic_V16 fill:#f3e5f5
    style Enterprise_V16 fill:#e1bee7
```

## 3. 에이전트 구성 (14 Agents)

```mermaid
graph LR
    subgraph Phase1["Phase 1: Discovery (6 Agents)"]
        A1[Data Analyst]
        A2[TDA Expert]
        A3[Schema Analyst]
        A4[Value Matcher]
        A5[Entity Classifier]
        A6[Relationship Detector]
    end

    subgraph Phase2["Phase 2: Refinement (4 Agents)"]
        B1[Ontology Architect]
        B2[Conflict Resolver]
        B3[Quality Judge]
        B4[Semantic Validator]
    end

    subgraph Phase3["Phase 3: Governance (4 Agents)"]
        C1[Governance Strategist]
        C2[Action Prioritizer]
        C3[Risk Assessor]
        C4[Policy Generator]
    end

    subgraph Communication["AgentBus v1.0"]
        BUS[Direct Messaging<br/>Broadcast<br/>Pub/Sub<br/>Request/Response]
    end

    Phase1 <--> BUS
    Phase2 <--> BUS
    Phase3 <--> BUS

    style Phase1 fill:#bbdefb
    style Phase2 fill:#fff9c4
    style Phase3 fill:#e1bee7
    style Communication fill:#b2dfdb
```

## 4. Cross-Entity Correlation 분석

```mermaid
graph TD
    subgraph Input
        T1[Table A]
        T2[Table B]
    end

    subgraph Analysis
        PC[Pearson Correlation]
        CP[Complementary Pairs<br/>sum=1.0]
        DC[Duplicate Columns<br/>r=1.0]
        TE[Threshold Effects]
        SA[Segment Analysis]
    end

    subgraph V16_Integration["v1.0 Integration"]
        LIN[Column Lineage]
        SHACL[SHACL Constraints]
    end

    subgraph Output
        PI[Predictive Insights]
        DQ[Data Quality Issues]
    end

    T1 & T2 --> PC & CP & DC & TE & SA
    PC --> PI
    CP & DC --> DQ
    TE & SA --> PI
    PI --> LIN
    DQ --> SHACL

    style PI fill:#c8e6c9
    style DQ fill:#ffcdd2
    style V16_Integration fill:#b2dfdb
```

## 5. 다중 에이전트 합의 프로토콜 (v1.0)

```mermaid
sequenceDiagram
    participant Chair as Consensus Chair
    participant BUS as AgentBus v1.0
    participant A1 as Ontologist
    participant A2 as Risk Assessor
    participant A3 as Business Strategist
    participant A4 as Data Steward
    participant Alg as Validator Algorithm

    Chair->>BUS: broadcast("consensus.start")
    BUS->>A1: Request Vote
    BUS->>A2: Request Vote
    BUS->>A3: Request Vote
    BUS->>A4: Request Vote

    A1->>BUS: publish("vote", APPROVE 0.85)
    A2->>BUS: publish("vote", REJECT Risk)
    A3->>BUS: publish("vote", APPROVE 0.90)
    A4->>BUS: publish("vote", APPROVE 0.75)

    BUS->>Chair: Collect all votes

    Chair->>Alg: Validate "Risk" claim
    Alg-->>Chair: Risk Score: Low

    Chair->>BUS: send_to(A2, challenge)
    BUS->>A2: Challenge with evidence
    A2->>BUS: publish("vote", APPROVE revised)

    Chair->>Chair: Dempster-Shafer Fusion
    Chair->>BUS: publish("consensus.result", conf: 0.87)
```

## 6. Evidence Chain

```mermaid
graph LR
    B1[Block 1<br/>Phase 1 Evidence]
    B2[Block 2<br/>Entity Discovery]
    B3[Block 3<br/>Relationship Detection]
    B4[Block 4<br/>Ontology Design]
    B5[Block 5<br/>Governance Decision]
    B6[Block 6<br/>v1.0 Validation]

    B1 -->|hash| B2 -->|hash| B3 -->|hash| B4 -->|hash| B5 -->|hash| B6

    style B1 fill:#bbdefb
    style B2 fill:#bbdefb
    style B3 fill:#bbdefb
    style B4 fill:#fff9c4
    style B5 fill:#e1bee7
    style B6 fill:#b2dfdb
```

## 7. Todo 시스템 (DAG)

```mermaid
graph TD
    T1[TDA Analysis]
    T2[Schema Analysis]
    T3[Value Matching]
    T4[Entity Classification]
    T5[Relationship Detection]
    T6[Ontology Building]
    T7[Quality Assessment]
    T8[Governance Decision]
    T9[v1.0 Validation]
    T10[Lineage Tracking]

    T1 --> T4
    T2 --> T4
    T3 --> T4
    T4 --> T5
    T5 --> T6
    T6 --> T7
    T7 --> T8
    T8 --> T9
    T9 --> T10

    style T1 fill:#bbdefb
    style T2 fill:#bbdefb
    style T3 fill:#bbdefb
    style T4 fill:#bbdefb
    style T5 fill:#bbdefb
    style T6 fill:#fff9c4
    style T7 fill:#fff9c4
    style T8 fill:#e1bee7
    style T9 fill:#b2dfdb
    style T10 fill:#b2dfdb
```

## 8. Enhanced FK Detection v1.0

```mermaid
graph TD
    subgraph Input
        Tables[(Tables<br/>Schema + Data)]
    end

    subgraph Solution1["Solution 1: Multi-Signal Scoring"]
        S1[Signal 1: Value Overlap<br/>Weight=0.25]
        S2[Signal 2: Cardinality<br/>Weight=0.20]
        S3[Signal 3: Referential Integrity<br/>Weight=0.25]
        S4[Signal 4: Naming Semantics<br/>Weight=0.15]
        S5[Signal 5: Structural Pattern<br/>Weight=0.15]
        SC[Signal Confidence]
    end

    subgraph Solution2["Solution 2: Direction Analysis"]
        E1[Evidence 1: Uniqueness]
        E2[Evidence 2: Containment]
        E3[Evidence 3: Row Count]
        E4[Evidence 4: Naming]
        E5[Evidence 5: Entity Reference]
        DC[Direction Confidence]
    end

    subgraph Solution3["Solution 3: Semantic Resolution"]
        SYN[Universal Synonym Groups]
        ABB[Abbreviation Expansion]
        CMP[Compound Entity Patterns]
        SEC[Semantic Confidence]
    end

    subgraph V16_Modules["v1.0 Enhanced Detection"]
        COMP[Composite FK Detector<br/>(plant, product) → table]
        HIER[Hierarchy Detector<br/>plant → warehouse]
        TEMP[Temporal FK Detector<br/>order_date range]
    end

    subgraph FinalCalc["Final Confidence"]
        CALC[signal × 0.50<br/>+ direction × 0.25<br/>+ semantic × 0.25]
        THR[Threshold >= 0.6]
    end

    subgraph Output
        FKC[FK Candidates]
    end

    Tables --> S1 & S2 & S3 & S4 & S5
    S1 & S2 & S3 & S4 & S5 --> SC
    Tables --> E1 & E2 & E3 & E4 & E5
    E1 & E2 & E3 & E4 & E5 --> DC
    Tables --> SYN & ABB & CMP
    SYN & ABB & CMP --> SEC
    Tables --> V16_Modules
    SC & DC & SEC --> CALC --> THR --> FKC
    V16_Modules --> FKC

    style Solution1 fill:#c8e6c9
    style Solution2 fill:#fff9c4
    style Solution3 fill:#e1bee7
    style V16_Modules fill:#b2dfdb
```

**Cross-dataset Validation Results (v1.0)**:
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

## 9. AgentBus v1.0 통신 패턴

```mermaid
sequenceDiagram
    participant A1 as TDA Expert
    participant BUS as AgentBus v1.0
    participant A2 as Schema Analyst
    participant A3 as Relationship Detector

    Note over BUS: Direct Messaging
    A1->>BUS: send_to("schema_analyst", msg)
    BUS->>A2: deliver(msg)

    Note over BUS: Broadcast
    A2->>BUS: broadcast(discovery_result)
    BUS->>A1: notify
    BUS->>A3: notify

    Note over BUS: Pub/Sub
    A3->>BUS: subscribe("fk.detected")
    A1->>BUS: publish("fk.detected", data)
    BUS->>A3: notify(data)

    Note over BUS: Request/Response
    A1->>BUS: request(A2, "validate_schema")
    BUS->>A2: handle_request
    A2->>BUS: response(result)
    BUS->>A1: deliver(result)
```

---

## 10. 데이터 흐름 (v1.0)

```mermaid
graph TD
    subgraph Input
        CSV[CSV Files]
        JSON[JSON Files]
    end

    subgraph Phase1[Phase 1: Discovery]
        Load[Data Loading]
        TDA[TDA Analysis]
        FK[Enhanced FK v1.0]
        Entity[Entity Classification]
    end

    subgraph Phase2[Phase 2: Refinement]
        Onto[Ontology Building]
        OWL[OWL2 Reasoning]
        SHACL[SHACL Validation]
        CEC[Cross-Entity Correlation]
        Quality[Quality Assessment]
    end

    subgraph Phase3[Phase 3: Governance]
        Council[Agent Council]
        Consensus[BFT Consensus]
        Decision[Final Decision]
        Actions[Actions Engine]
    end

    subgraph V16_Post["v1.0 Post-Processing"]
        CDC[CDC Sync]
        Lineage[Column Lineage]
        Memory[Agent Memory]
    end

    subgraph Output
        Artifacts[Artifacts<br/>ontology.json<br/>insights.json<br/>predictive_insights.json]
    end

    CSV & JSON --> Load
    Load --> TDA & FK
    TDA & FK --> Entity
    Entity --> Onto
    Onto --> OWL --> SHACL
    SHACL --> CEC
    CEC --> Quality
    Quality --> Council
    Council --> Consensus
    Consensus --> Decision
    Decision --> Actions
    Actions --> V16_Post
    V16_Post --> Artifacts

    style Phase1 fill:#bbdefb
    style Phase2 fill:#fff9c4
    style Phase3 fill:#e1bee7
    style V16_Post fill:#b2dfdb
```

## 11. 파일 구조 (v1.0)

```
src/unified_pipeline/
├── unified_main.py              # Entry Point
├── autonomous/
│   ├── orchestrator.py          # Agent Orchestrator
│   ├── autonomous_pipeline.py   # Pipeline Orchestrator
│   ├── shared_context.py        # Shared Context
│   ├── agent_bus.py             # v1.0: AgentCommunicationBus
│   │
│   ├── agents/
│   │   ├── discovery.py         # Phase 1 Agents (6)
│   │   ├── refinement.py        # Phase 2 Agents (4)
│   │   └── governance.py        # Phase 3 Agents (4)
│   │
│   ├── analysis/
│   │   ├── enhanced_fk_pipeline.py    # Enhanced FK Pipeline
│   │   ├── fk_signal_scorer.py        # Multi-Signal Scoring
│   │   ├── fk_direction_analyzer.py   # Direction Analysis
│   │   ├── semantic_entity_resolver.py # Semantic Resolution
│   │   ├── composite_fk_detector.py   # v1.0: Composite FK
│   │   ├── hierarchy_detector.py      # v1.0: Hierarchy
│   │   ├── temporal_fk_detector.py    # v1.0: Temporal FK
│   │   ├── cross_entity_correlation.py
│   │   ├── tda.py
│   │   ├── business_insights.py
│   │   └── ...
│   │
│   ├── consensus/
│   │   └── engine.py            # BFT + DS Consensus
│   │
│   └── todo/
│       ├── models.py
│       └── manager.py
│
│   ├── autonomous/analysis/      # v1.0: Semantic Layer (in analysis/)
│   │   ├── owl2_reasoner.py     # OWL2 Reasoning
│   │   └── shacl_validator.py   # SHACL Validation
│
├── actions/                      # v1.0: Actions Engine
│   └── actions_engine.py
│
├── sync/                         # v1.0: CDC Sync
│   └── cdc_sync_engine.py
│
├── lineage/                      # v1.0: Column Lineage
│   └── column_lineage.py
│
└── learning/                     # v1.0: Agent Learning
    └── agent_memory.py

output/{scenario_name}/
├── *_context.json               # Full context with relationships
├── *_entities.json              # Discovered entities
├── *_summary.json               # Execution summary
├── *_lineage.json               # v1.0: Column lineage
└── *_stats.json                 # Statistics
```

---

## 12. v1.0 검증 결과 요약

### marketing_silo (12 FK Relationships)

```
✓ leads.cust_id → customers.customer_id
✓ orders.buyer_id → customers.customer_id
✓ orders.prod_code → products.product_id
✓ email_sends.cmp_id → campaigns.campaign_id
✓ email_sends.cust_no → customers.customer_id
✓ email_events.email_send_id → email_sends.send_id
✓ ad_campaigns.marketing_campaign_ref → campaigns.campaign_id
✓ ad_performance.ad_cmp_id → ad_campaigns.ad_campaign_id
✓ web_sessions.user_id → customers.customer_id
✓ conversions.order_ref → orders.order_id
✓ conversions.attributed_ad_id → ad_campaigns.ad_campaign_id
✓ conversions.attributed_cmp_id → campaigns.campaign_id

Precision: 100% | Recall: 100% | F1: 100%
```

### airport_silo (13 FK Relationships)

```
✓ flights.carrier_code → airlines.iata_code
✓ flights.gate_number → gates.gate_code
✓ passengers.flight_num → flights.flight_number
✓ baggage.booking_reference → passengers.booking_ref
✓ baggage.flt_number → flights.flight_number
✓ maintenance.aircraft_reg → aircraft.registration_no
✗ maintenance.mechanic_id → employees.emp_id (미검출)
✓ delay_records.flt_no → flights.flight_number
... (일부 미검출)

Precision: 90.9% | Recall: 76.9% | F1: 83.3%
```

### Cross-dataset Average

```
Precision: 95.5% | Recall: 88.5% | F1: 91.7%
```

---

## 13. 다이어그램 파일 목록

| 파일 | 설명 | 버전 |
|------|------|------|
| [01-system-architecture.md](01-system-architecture.md) | 시스템 아키텍처 | v1.0 |
| [02-data-flow.md](02-data-flow.md) | 데이터 흐름 | v1.0 |
| [03-pipeline-workflow.md](03-pipeline-workflow.md) | 파이프라인 워크플로우 | v1.0 |
| [04-consensus-engine.md](04-consensus-engine.md) | 합의 엔진 | v1.0 |
| [05-agent-orchestration.md](05-agent-orchestration.md) | 에이전트 오케스트레이션 | v1.0 |
| [06-knowledge-graph.md](06-knowledge-graph.md) | 지식 그래프 | v1.0 |
| [07-tda-analysis.md](07-tda-analysis.md) | TDA 분석 | v1.0 |
| [08-domain-detection.md](08-domain-detection.md) | 도메인 탐지 | v1.0 |
