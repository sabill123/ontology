# 파이프라인 워크플로우 다이어그램

> **버전**: v14.0
> **최종 업데이트**: 2026-01-19

## 1. 전체 파이프라인 흐름

```mermaid
graph TB
    subgraph INIT["Initialization"]
        START([Start Pipeline])
        LOAD[Load Table Data<br/>CSV/JSON/Parquet]
        DOMAIN[Domain Detection<br/>LLM-First v13.0]
        CTX[Initialize SharedContext]
    end

    subgraph PHASE1["Phase 1: Discovery"]
        direction TB
        D1[Data Analyst<br/>LLM-First Analysis]
        D2[TDA Signature Computation]
        D3[Schema Analysis]
        D4[Universal FK Detection]
        D5[Entity Classification]
        D6[Relationship Detection]

        D1 --> D2 & D3
        D3 --> D4
        D2 & D4 --> D5
        D5 --> D6
    end

    subgraph PHASE2["Phase 2: Refinement"]
        direction TB
        R1[Ontology Architecture Design]
        R2[Cross-Entity Correlation v10.0]
        R3[Concept Generation]
        R4{Quality Assessment}
        R5[Conflict Resolution]
        R6{Smart Consensus v8.2}

        R1 --> R2
        R2 --> R3
        R3 --> R4
        R4 -->|Pass| R6
        R4 -->|Fail| R5
        R5 --> R6
    end

    subgraph PHASE3["Phase 3: Governance"]
        direction TB
        G1[Agent Council<br/>4 Personas]
        G2[BFT Consensus<br/>+ DS Fusion]
        G3[Evidence-Based Debate]
        G4[Final Decision]
        G5[Save Artifacts]

        G1 --> G2
        G2 --> G3
        G3 --> G4
        G4 --> G5
    end

    subgraph OUTPUT["Output"]
        RESULT([Pipeline Complete])
        ONT[(ontology.json)]
        INS[(insights.json)]
        PI[(predictive_insights.json)]
        GOV[(governance_decisions.json)]
    end

    START --> LOAD
    LOAD --> DOMAIN
    DOMAIN --> CTX
    CTX --> PHASE1
    PHASE1 --> PHASE2
    PHASE2 --> PHASE3
    PHASE3 --> ONT & INS & PI & GOV
    ONT & INS & PI & GOV --> RESULT

    style PHASE1 fill:#bbdefb
    style PHASE2 fill:#fff9c4
    style PHASE3 fill:#e1bee7
```

## 2. Phase 1: Discovery 상세 워크플로우

```mermaid
flowchart TB
    subgraph INPUT["Input Tables"]
        T1[Table A]
        T2[Table B]
        T3[Table C]
    end

    subgraph DA["Data Analyst v13.0"]
        DA1["전체 데이터 LLM 전달"]
        DA2["도메인 식별"]
        DA3["컬럼 비즈니스 특성 분류"]
        DA4["데이터 품질 평가"]
    end

    subgraph TDA["TDA Analysis"]
        TDA1["Betti Numbers 계산<br/>β₀, β₁, β₂"]
        TDA2["Persistence Diagram"]
        TDA3["Structural Complexity"]
        TDA4["Homeomorphism Detection"]
    end

    subgraph FK["FK Detection"]
        FK1["Key Column 식별"]
        FK2["Containment Ratio 계산"]
        FK3["Value Overlap 분석"]
        FK4["FK Candidates"]
    end

    subgraph ENTITY["Entity Extraction"]
        E1["Table → Entity Mapping"]
        E2["Column → Property Mapping"]
        E3["FK → Relationship Mapping"]
        E4["Unified Entities"]
    end

    INPUT --> DA
    DA1 --> DA2 --> DA3 --> DA4
    DA4 --> TDA & FK

    TDA1 --> TDA2 --> TDA3 --> TDA4
    FK1 --> FK2 --> FK3 --> FK4

    TDA4 & FK4 --> E1
    E1 --> E2 --> E3 --> E4

    style DA fill:#e3f2fd
    style TDA fill:#fff3e0
    style FK fill:#fce4ec
    style ENTITY fill:#e8f5e9
```

## 3. Phase 2: Refinement 상세 워크플로우

```mermaid
flowchart TB
    subgraph INPUT["From Phase 1"]
        UE[Unified Entities]
        FK[FK Candidates]
        TDA[TDA Signatures]
    end

    subgraph ONTO["Ontology Design"]
        O1["Design Concept Hierarchy"]
        O2["Define Property Schemas"]
        O3["Establish Relationship Types"]
        O4["Generate Concepts"]
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
        Q3["Confidence Scoring"]
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
        CON4["full_consensus<br/>Multi-Agent"]
        CON5["Consensus Result"]
    end

    INPUT --> ONTO
    O1 --> O2 --> O3 --> O4
    O4 --> CEC
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

    style CEC fill:#c8e6c9
    style ONTO fill:#e8f5e9
    style QUALITY fill:#fff3e0
    style CONFLICT fill:#ffebee
    style CONSENSUS fill:#f3e5f5
```

## 4. Phase 3: Governance 상세 워크플로우

```mermaid
flowchart TB
    subgraph INPUT["From Phase 2"]
        CONCEPTS[Approved Concepts]
        INSIGHTS[Business Insights]
        PI[Predictive Insights]
    end

    subgraph COUNCIL["Agent Council (4 Personas)"]
        A1["Ontologist<br/>논리적 일관성"]
        A2["Risk Assessor<br/>리스크 평가"]
        A3["Business Strategist<br/>비즈니스 가치"]
        A4["Data Steward<br/>데이터 품질"]
    end

    subgraph BFT["BFT Consensus"]
        B1["Round 1: Initial Vote"]
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

    subgraph DECISION["Final Decision"]
        D1["APPROVED<br/>conf >= 0.75"]
        D2["PROVISIONAL<br/>conf >= 0.50"]
        D3["REJECTED<br/>conf < 0.30"]
        D4["ESCALATED<br/>Human Review"]
    end

    subgraph OUTPUT["Output"]
        OUT1[(governance_decisions.json)]
        OUT2[(Action Items)]
        OUT3[(Policies)]
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
    D1 & D2 & D3 & D4 --> OUTPUT

    style COUNCIL fill:#e8f5e9
    style BFT fill:#fff3e0
    style DS fill:#e3f2fd
    style DECISION fill:#f3e5f5
```

## 5. Todo DAG 워크플로우

```mermaid
graph TD
    T1[TDA Analysis]
    T2[Schema Analysis]
    T3[Value Matching]
    T4[Entity Classification]
    T5[Relationship Detection]
    T6[Ontology Building]
    T7[Cross-Entity Correlation]
    T8[Quality Assessment]
    T9[Governance Decision]

    T1 --> T4
    T2 --> T4
    T3 --> T4
    T4 --> T5
    T5 --> T6
    T6 --> T7
    T7 --> T8
    T8 --> T9

    style T1 fill:#bbdefb
    style T2 fill:#bbdefb
    style T3 fill:#bbdefb
    style T4 fill:#bbdefb
    style T5 fill:#bbdefb
    style T6 fill:#fff9c4
    style T7 fill:#c8e6c9
    style T8 fill:#fff9c4
    style T9 fill:#e1bee7
```

## 6. Phase Gate 검증 (v14.0)

```mermaid
flowchart TB
    subgraph P1_GATE["Phase 1 Gate"]
        P1G1{unified_entities >= 1?}
        P1G2{evidence_chain not empty?}
        P1G3{no BLOCKED todos?}
        P1G4{completion_signal == DONE?}
        P1G5["Pass → Phase 2"]
        P1G6["Fail → Fallback"]
    end

    subgraph P2_GATE["Phase 2 Gate"]
        P2G1{ontology_concepts >= 1?}
        P2G2{approved_ratio >= 0.5?}
        P2G3{no critical conflicts?}
        P2G4{completion_signal == DONE?}
        P2G5["Pass → Phase 3"]
        P2G6["Fail → Fallback"]
    end

    subgraph P3_GATE["Phase 3 Gate"]
        P3G1{governance_decisions not empty?}
        P3G2{approval_ratio >= 0.6?}
        P3G3{no ESCALATED decisions?}
        P3G4{completion_signal == DONE?}
        P3G5["Pass → Complete"]
        P3G6["Fail → Human Review"]
    end

    P1G1 --> P1G2 --> P1G3 --> P1G4
    P1G4 -->|Yes| P1G5
    P1G4 -->|No| P1G6

    P2G1 --> P2G2 --> P2G3 --> P2G4
    P2G4 -->|Yes| P2G5
    P2G4 -->|No| P2G6

    P3G1 --> P3G2 --> P3G3 --> P3G4
    P3G4 -->|Yes| P3G5
    P3G4 -->|No| P3G6

    style P1_GATE fill:#bbdefb
    style P2_GATE fill:#fff9c4
    style P3_GATE fill:#e1bee7
```

## 7. 함수 호출 흐름

```mermaid
graph TB
    subgraph DISCOVERY["Phase 1: Discovery"]
        D1[compute_tda_signatures]
        D2[analyze_schema]
        D3[detect_homeomorphisms]
        D4[discover_fk_candidates]
        D5[classify_entities]
        D6[detect_relationships]

        D1 --> D3
        D2 --> D4
        D3 --> D5
        D4 --> D5
        D5 --> D6
    end

    subgraph REFINEMENT["Phase 2: Refinement"]
        R1[design_ontology]
        R2[run_cross_entity_correlation]
        R3[generate_concepts]
        R4[assess_quality]
        R5[resolve_conflicts]
        R6[run_consensus]

        D6 --> R1
        R1 --> R2
        R2 --> R3
        R3 --> R4
        R4 --> R5
        R5 --> R6
    end

    subgraph GOVERNANCE["Phase 3: Governance"]
        G1[create_governance_strategy]
        G2[run_bft_consensus]
        G3[dempster_shafer_fusion]
        G4[evidence_based_debate]
        G5[make_final_decision]

        R6 --> G1
        G1 --> G2
        G2 --> G3
        G3 --> G4
        G4 --> G5
    end

    style D1 fill:#bbdefb
    style D6 fill:#bbdefb
    style R1 fill:#fff9c4
    style R2 fill:#c8e6c9
    style R6 fill:#fff9c4
    style G1 fill:#e1bee7
    style G5 fill:#e1bee7
```
