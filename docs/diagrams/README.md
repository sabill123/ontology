# Ontoloty v1 다이어그램

> **최종 업데이트**: 2026-01-22
> **버전**: v1

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

    subgraph Output
        ONT[(Ontology)]
        INS[(Insights)]
        PI[(Predictive<br/>Insights)]
        GOV[(Governance<br/>Decisions)]
    end

    DS --> P1 --> P2 --> P3 --> ONT & INS & PI & GOV

    style P1 fill:#bbdefb
    style P2 fill:#fff9c4
    style P3 fill:#e1bee7
```

## 2. 핵심 컴포넌트

```mermaid
graph TB
    subgraph Orchestration
        AO[Agent Orchestrator]
        TM[Todo Manager<br/>DAG]
        CE[Consensus Engine<br/>BFT + DS]
    end

    subgraph Analysis
        TDA[TDA Analyzer]
        FK[Enhanced FK v1<br/>Multi-Signal]
        CEC[Cross-Entity<br/>Correlation]
        BI[Business Insights]
    end

    subgraph Storage
        SC[Shared Context]
        EC[Evidence Chain]
        KG[Knowledge Graph]
    end

    AO --> TM
    AO --> CE
    TM --> Analysis
    Analysis --> SC
    SC --> EC
    SC --> KG

    style FK fill:#c8e6c9
```

## 3. 에이전트 구성

```mermaid
graph LR
    subgraph Phase1["Phase 1: Discovery"]
        A1[Data Analyst]
        A2[TDA Expert]
        A3[Schema Analyst]
        A4[Value Matcher]
        A5[Entity Classifier]
        A6[Relationship Detector]
    end

    subgraph Phase2["Phase 2: Refinement"]
        B1[Ontology Architect]
        B2[Conflict Resolver]
        B3[Quality Judge]
        B4[Semantic Validator]
    end

    subgraph Phase3["Phase 3: Governance"]
        C1[Governance Strategist]
        C2[Action Prioritizer]
        C3[Risk Assessor]
        C4[Policy Generator]
    end

    style Phase1 fill:#bbdefb
    style Phase2 fill:#fff9c4
    style Phase3 fill:#e1bee7
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

    subgraph Output
        PI[Predictive Insights]
        DQ[Data Quality Issues]
    end

    T1 & T2 --> PC & CP & DC & TE & SA
    PC --> PI
    CP & DC --> DQ
    TE & SA --> PI

    style PI fill:#c8e6c9
    style DQ fill:#ffcdd2
```

## 5. 다중 에이전트 합의 프로토콜

```mermaid
sequenceDiagram
    participant Chair as Consensus Chair
    participant A1 as Ontologist
    participant A2 as Risk Assessor
    participant A3 as Business Strategist
    participant A4 as Data Steward
    participant Alg as Validator Algorithm

    Chair->>A1: Request Vote
    Chair->>A2: Request Vote
    Chair->>A3: Request Vote
    Chair->>A4: Request Vote

    A1->>Chair: APPROVE (conf: 0.85)
    A2->>Chair: REJECT (reason: Risk)
    A3->>Chair: APPROVE (conf: 0.90)
    A4->>Chair: APPROVE (conf: 0.75)

    Chair->>Alg: Validate "Risk" claim
    Alg-->>Chair: Risk Score: Low

    Chair->>A2: Challenge with evidence
    A2->>Chair: APPROVE (revised)

    Chair->>Chair: Dempster-Shafer Fusion
    Chair->>Output: Final Decision (conf: 0.87)
```

## 6. Evidence Chain

```mermaid
graph LR
    B1[Block 1<br/>Phase 1 Evidence]
    B2[Block 2<br/>Entity Discovery]
    B3[Block 3<br/>Relationship Detection]
    B4[Block 4<br/>Ontology Design]
    B5[Block 5<br/>Governance Decision]

    B1 -->|hash| B2 -->|hash| B3 -->|hash| B4 -->|hash| B5

    style B1 fill:#bbdefb
    style B2 fill:#bbdefb
    style B3 fill:#bbdefb
    style B4 fill:#fff9c4
    style B5 fill:#e1bee7
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

    T1 --> T4
    T2 --> T4
    T3 --> T4
    T4 --> T5
    T5 --> T6
    T6 --> T7
    T7 --> T8

    style T1 fill:#bbdefb
    style T2 fill:#bbdefb
    style T3 fill:#bbdefb
    style T4 fill:#bbdefb
    style T5 fill:#bbdefb
    style T6 fill:#fff9c4
    style T7 fill:#fff9c4
    style T8 fill:#e1bee7
```

## 8. Enhanced FK Detection v1

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
    SC & DC & SEC --> CALC --> THR --> FKC

    style Solution1 fill:#c8e6c9
    style Solution2 fill:#fff9c4
    style Solution3 fill:#e1bee7
```

**Cross-dataset Validation Results**:
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

## 9. 데이터 흐름

```mermaid
graph TD
    subgraph Input
        CSV[CSV Files]
        JSON[JSON Files]
    end

    subgraph Phase1[Phase 1: Discovery]
        Load[Data Loading]
        TDA[TDA Analysis]
        FK[Enhanced FK v1]
        Entity[Entity Classification]
    end

    subgraph Phase2[Phase 2: Refinement]
        Onto[Ontology Building]
        CEC[Cross-Entity Correlation]
        Quality[Quality Assessment]
    end

    subgraph Phase3[Phase 3: Governance]
        Council[Agent Council]
        Consensus[BFT Consensus]
        Decision[Final Decision]
    end

    subgraph Output
        Artifacts[Artifacts<br/>ontology.json<br/>insights.json<br/>predictive_insights.json]
    end

    CSV & JSON --> Load
    Load --> TDA & FK
    TDA & FK --> Entity
    Entity --> Onto
    Onto --> CEC
    CEC --> Quality
    Quality --> Council
    Council --> Consensus
    Consensus --> Decision
    Decision --> Artifacts

    style Phase1 fill:#bbdefb
    style Phase2 fill:#fff9c4
    style Phase3 fill:#e1bee7
```

## 10. 파일 구조

```
src/unified_pipeline/
├── unified_main.py              # Entry Point
├── autonomous/
│   ├── orchestrator.py          # Agent Orchestrator
│   ├── autonomous_pipeline.py   # Pipeline Orchestrator
│   ├── shared_context.py        # Shared Context
│   │
│   ├── agents/
│   │   ├── discovery.py         # Phase 1 Agents
│   │   ├── refinement.py        # Phase 2 Agents
│   │   └── governance.py        # Phase 3 Agents
│   │
│   ├── analysis/
│   │   ├── enhanced_fk_pipeline.py    # v1 Enhanced FK Pipeline
│   │   ├── fk_signal_scorer.py        # v1 Multi-Signal Scoring
│   │   ├── fk_direction_analyzer.py   # v1 Direction Analysis
│   │   ├── semantic_entity_resolver.py # v1 Semantic Resolution
│   │   ├── enhanced_fk_detector.py    # Legacy FK Detection
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

output/{scenario_name}/
├── *_context.json               # Full context with relationships
├── *_entities.json              # Discovered entities
├── *_summary.json               # Execution summary
└── *_stats.json                 # Statistics
```

---

## 11. v1 검증 결과 요약

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
