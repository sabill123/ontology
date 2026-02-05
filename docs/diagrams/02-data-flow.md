# 데이터 흐름 다이어그램

> **버전**: v1.0
> **최종 업데이트**: 2026-01-27

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
        P1B[TDA Signature<br/>Computation]
        P1C[Enhanced FK Pipeline v1.0<br/>Multi-Signal + Direction]
        P1D[Advanced FK Detection<br/>Composite/Hierarchy/Temporal]
        P1E[Entity<br/>Classification]
        P1F[Relationship<br/>Detection]

        P1B --> P1C
        P1C --> P1D
        P1B --> P1E
        P1D --> P1E
        P1E --> P1F
    end

    subgraph PHASE2["Phase 2: Refinement"]
        direction TB
        P2A[OWL2 Reasoner<br/>Axiom Inference]
        P2B[SHACL Validator<br/>Constraint Checking]
        P2C[Cross-Entity<br/>Correlation v10.0]
        P2D[Conflict<br/>Resolution]
        P2E[Smart Consensus<br/>v8.2]

        P2A --> P2B
        P2B --> P2C
        P2C --> P2D
        P2D --> P2E
    end

    subgraph PHASE3["Phase 3: Governance"]
        direction TB
        P3A[Actions Engine<br/>Priority Scoring]
        P3B[CDC Sync Engine<br/>Change Detection]
        P3C[Lineage Tracker<br/>Impact Analysis]
        P3D[Final<br/>Decision]

        P3A --> P3B
        P3B --> P3C
        P3C --> P3D
    end

    subgraph OUTPUT["Output Layer"]
        direction LR
        O1[foreign_keys]
        O2[indirect_fks]
        O3[ontology]
        O4[tda_analysis]
        O5[entities]
        O6[insights]
        O7[correlations]
        O8[governance]
        O9[evidence_chain]
        O10[knowledge_graph]
        O11[llm_analysis]
        O12[domain]
    end

    INPUT --> PHASE1
    PHASE1 --> PHASE2
    PHASE2 --> PHASE3
    PHASE3 --> OUTPUT

    style PHASE1 fill:#bbdefb
    style PHASE2 fill:#fff9c4
    style PHASE3 fill:#e1bee7
```

## 2. SharedContext 데이터 흐름 (v1.0)

```mermaid
flowchart LR
    subgraph SC["SharedContext v1.0"]
        direction TB
        Tables[(tables)]
        TDA[(tda_signatures)]
        FK[(enhanced_fk_candidates)]
        CFK[(composite_fk)]
        HFD[(hierarchies)]
        TFK[(temporal_fk)]
        Entities[(unified_entities)]
        Concepts[(ontology_concepts)]
        OWL[(dynamic_data.owl2_axioms)]
        SHACL[(dynamic_data.shacl_shapes)]
        Insights[(business_insights)]
        Predictive[(palantir_insights)]
        Actions[(recommended_actions)]
        Lineage[(column_lineage)]
        Decisions[(governance_decisions)]
        Evidence[(evidence_chain)]
    end

    subgraph P1["Phase 1 Agents"]
        DA[Data Analyst]
        TA[TDA Expert]
        SA[Schema Analyst]
        VM[Value Matcher]
        EC[Entity Classifier]
        RD[Relationship Detector<br/>v1.0]
    end

    subgraph P2["Phase 2 Agents"]
        OA[Ontology Architect<br/>+ OWL2]
        CR[Conflict Resolver]
        QJ[Quality Judge<br/>+ SHACL]
        SV[Semantic Validator]
    end

    subgraph P3["Phase 3 Agents"]
        GS[Governance Strategist]
        AP[Action Prioritizer<br/>+ Actions Engine]
        RA[Risk Assessor<br/>+ Impact Analysis]
        PG[Policy Generator<br/>+ CDC Sync]
    end

    Tables --> DA & TA & SA
    TA --> TDA
    SA --> FK
    VM --> FK
    RD --> CFK & HFD & TFK
    EC --> Entities
    RD --> Entities

    Entities --> OA
    OA --> Concepts & OWL
    OA --> Predictive
    CR --> Concepts
    QJ --> Concepts & SHACL
    SV --> Concepts

    Concepts --> GS
    GS --> Decisions
    AP --> Actions
    RA --> Lineage
    PG --> Decisions

    Decisions --> Evidence

    style SC fill:#e8f5e9
    style CFK fill:#c8e6c9
    style HFD fill:#c8e6c9
    style TFK fill:#c8e6c9
    style OWL fill:#f3e5f5
    style SHACL fill:#f3e5f5
    style Lineage fill:#b2dfdb
```

## 3. 데이터 변환 파이프라인 (v1.0)

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

    subgraph SCHEMA["Enhanced FK Analysis v1.0"]
        S1["Multi-Signal Scoring<br/>5 signals weighted"]
        S2["Direction Analysis<br/>5 evidence types"]
        S3["Semantic Resolution<br/>Synonym/Abbreviation"]
        S4["Composite FK<br/>Multi-column keys"]
        S5["Hierarchy Detection<br/>Self-referencing"]
        S6["Temporal FK<br/>Valid time ranges"]
    end

    subgraph ENTITY["Entity Extraction"]
        E1["Entity Type<br/>CUSTOMER, ORDER"]
        E2["Properties<br/>{name, amount, ...}"]
        E3["Advanced Relations<br/>Hierarchical, Temporal"]
    end

    subgraph SEMANTIC["Semantic Reasoning v1.0"]
        SEM1["OWL2 Axioms<br/>SubClassOf, EquivalentClass"]
        SEM2["SHACL Shapes<br/>NodeShape, PropertyShape"]
        SEM3["Constraint Validation<br/>Pass/Fail + Violations"]
    end

    subgraph CONCEPT["Ontology Concept"]
        C1["concept_id: obj_customer_001"]
        C2["type: object_type"]
        C3["confidence: 0.92"]
        C4["owl2_axioms: [...]"]
        C5["shacl_shapes: [...]"]
    end

    subgraph INSIGHT["Predictive Insight v1.0"]
        I1["correlation: 0.527"]
        I2["threshold_effect: detected"]
        I3["lineage_impact: [columns]"]
        I4["recommended_action"]
    end

    subgraph LINEAGE["Column Lineage v1.0"]
        L1["source_columns: [...]"]
        L2["transformations: [...]"]
        L3["impact_analysis: [...]"]
    end

    RAW --> TDA
    RAW --> SCHEMA
    TDA & SCHEMA --> ENTITY
    ENTITY --> SEMANTIC
    SEMANTIC --> CONCEPT
    CONCEPT --> INSIGHT
    INSIGHT --> LINEAGE

    style RAW fill:#ffebee
    style TDA fill:#e3f2fd
    style SCHEMA fill:#c8e6c9
    style ENTITY fill:#e8f5e9
    style SEMANTIC fill:#f3e5f5
    style CONCEPT fill:#f3e5f5
    style INSIGHT fill:#c8e6c9
    style LINEAGE fill:#b2dfdb
```

## 4. Enhanced FK Detection 데이터 흐름 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["Input Tables"]
        T1[Table A<br/>Columns + Data]
        T2[Table B<br/>Columns + Data]
    end

    subgraph STAGE1["Stage 1: Multi-Signal Scoring"]
        S1[Value Overlap<br/>Weight: 0.25]
        S2[Cardinality Ratio<br/>Weight: 0.20]
        S3[Referential Integrity<br/>Weight: 0.25]
        S4[Naming Semantics<br/>Weight: 0.15]
        S5[Structural Pattern<br/>Weight: 0.15]
        SC[Signal Confidence]
    end

    subgraph STAGE2["Stage 2: Direction Analysis"]
        E1[Uniqueness Evidence]
        E2[Containment Evidence]
        E3[Row Count Evidence]
        E4[Naming Evidence]
        E5[Entity Reference Evidence]
        DC[Direction Confidence]
    end

    subgraph STAGE3["Stage 3: Semantic Resolution"]
        SYN[Universal Synonym Groups]
        ABB[Abbreviation Expansion]
        CMP[Compound Entity Patterns]
        SEC[Semantic Confidence]
    end

    subgraph STAGE4["Stage 4: Advanced Detection"]
        CFK[Composite FK Detector<br/>Multi-column keys]
        HFD[Hierarchy Detector<br/>Self-referencing]
        TFK[Temporal FK Detector<br/>Time-based relations]
    end

    subgraph OUTPUT["FK Candidates"]
        FK1["Simple FK<br/>(1:1 column mapping)"]
        FK2["Composite FK<br/>(multi-column)"]
        FK3["Hierarchical FK<br/>(self-referencing)"]
        FK4["Temporal FK<br/>(time-aware)"]
    end

    T1 & T2 --> S1 & S2 & S3 & S4 & S5
    S1 & S2 & S3 & S4 & S5 --> SC
    SC --> E1 & E2 & E3 & E4 & E5
    E1 & E2 & E3 & E4 & E5 --> DC
    DC --> SYN & ABB & CMP
    SYN & ABB & CMP --> SEC
    SEC --> CFK & HFD & TFK
    CFK --> FK2
    HFD --> FK3
    TFK --> FK4
    SEC --> FK1

    style STAGE1 fill:#c8e6c9
    style STAGE2 fill:#fff9c4
    style STAGE3 fill:#e1bee7
    style STAGE4 fill:#b2dfdb
```

## 5. OWL2 Reasoning 데이터 흐름 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["Input: Ontology Concepts"]
        I1["Entities"]
        I2["Relationships"]
        I3["Properties"]
    end

    subgraph OWL2["OWL2 Reasoner"]
        subgraph AXIOMS["Axiom Types"]
            A1[SubClassOf<br/>Class Hierarchy]
            A2[EquivalentClass<br/>Class Equivalence]
            A3[DisjointClasses<br/>Mutual Exclusion]
            A4[PropertyDomain<br/>Domain Constraints]
            A5[PropertyRange<br/>Range Constraints]
            A6[InverseProperties<br/>Bidirectional]
            A7[TransitiveProperty<br/>Transitive Closure]
        end

        subgraph INFERENCE["Inference Engine"]
            INF1[Subsumption Check]
            INF2[Consistency Check]
            INF3[Classification]
        end
    end

    subgraph OUTPUT["Inferred Knowledge"]
        O1["Inferred Axioms"]
        O2["Class Hierarchy"]
        O3["Property Constraints"]
        O4["Consistency Report"]
    end

    INPUT --> AXIOMS
    A1 & A2 & A3 & A4 & A5 & A6 & A7 --> INFERENCE
    INFERENCE --> OUTPUT

    style OWL2 fill:#f3e5f5
    style AXIOMS fill:#e1bee7
    style INFERENCE fill:#fff9c4
```

## 6. SHACL Validation 데이터 흐름 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["Input: Data Graph"]
        I1["Entities"]
        I2["Properties"]
        I3["Relationships"]
    end

    subgraph SHAPES["SHACL Shapes"]
        subgraph NODE["NodeShape"]
            N1[sh:targetClass]
            N2[sh:property]
            N3[sh:closed]
        end

        subgraph PROP["PropertyShape"]
            P1[sh:datatype]
            P2[sh:minCount/maxCount]
            P3[sh:pattern]
            P4[sh:minValue/maxValue]
            P5[sh:in]
        end
    end

    subgraph VALIDATION["Validation Engine"]
        V1[Shape Matching]
        V2[Constraint Checking]
        V3[Violation Detection]
    end

    subgraph OUTPUT["Validation Report"]
        O1["Conformance: Pass/Fail"]
        O2["Violations List"]
        O3["Severity Levels"]
        O4["Fix Suggestions"]
    end

    INPUT --> SHAPES
    NODE & PROP --> VALIDATION
    V1 --> V2 --> V3
    V3 --> OUTPUT

    style SHAPES fill:#f3e5f5
    style NODE fill:#e1bee7
    style PROP fill:#e1bee7
    style VALIDATION fill:#fff9c4
```

## 7. Column Lineage 데이터 흐름 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["Source Data"]
        S1[(Table A<br/>col_1, col_2)]
        S2[(Table B<br/>col_3, col_4)]
    end

    subgraph TRANSFORM["Transformations"]
        T1[JOIN<br/>A.col_1 = B.col_3]
        T2[AGGREGATE<br/>SUM(col_2)]
        T3[FILTER<br/>col_4 > 100]
    end

    subgraph LINEAGE["Lineage Graph"]
        subgraph NODES["Column Nodes"]
            CN1[A.col_1<br/>SOURCE]
            CN2[A.col_2<br/>SOURCE]
            CN3[B.col_3<br/>SOURCE]
            CN4[B.col_4<br/>SOURCE]
            CN5[joined.col_1<br/>DERIVED]
            CN6[result.total<br/>AGGREGATE]
        end

        subgraph EDGES["Lineage Edges"]
            E1[DIRECT<br/>Identity mapping]
            E2[AGGREGATED<br/>Aggregation]
            E3[FILTERED<br/>Subset]
            E4[DERIVED<br/>Calculation]
        end
    end

    subgraph OUTPUT["Impact Analysis"]
        O1["Upstream Impact<br/>What affects this?"]
        O2["Downstream Impact<br/>What does this affect?"]
        O3["Change Propagation<br/>Affected columns"]
    end

    S1 --> T1
    S2 --> T1
    T1 --> T2 --> T3

    CN1 & CN3 --> CN5
    CN2 --> CN6

    LINEAGE --> OUTPUT

    style LINEAGE fill:#b2dfdb
    style NODES fill:#c8e6c9
    style OUTPUT fill:#fff9c4
```

## 8. Agent Communication Bus 데이터 흐름 (v1.0)

```mermaid
sequenceDiagram
    participant A1 as Relationship Detector
    participant BUS as Agent Bus
    participant A2 as Ontology Architect
    participant A3 as Quality Judge
    participant CE as Consensus Engine

    Note over A1,CE: Direct Messaging
    A1->>BUS: send_to("ontology_architect", fk_result)
    BUS->>A2: deliver(message)

    Note over A1,CE: Pub/Sub Pattern
    A1->>BUS: publish("discovery.fk_detected", candidates)
    BUS->>A2: notify(subscription)
    BUS->>A3: notify(subscription)

    Note over A1,CE: Request/Response
    A2->>BUS: request("quality_judge", validation_req)
    BUS->>A3: deliver(request)
    A3->>BUS: respond(validation_result)
    BUS->>A2: deliver(response)

    Note over A1,CE: Consensus Integration
    CE->>BUS: broadcast(vote_request)
    BUS->>A1: notify
    BUS->>A2: notify
    BUS->>A3: notify
    A1->>BUS: send_to("consensus", vote)
    A2->>BUS: send_to("consensus", vote)
    A3->>BUS: send_to("consensus", vote)
```

## 9. CDC Sync 데이터 흐름 (v1.0)

```mermaid
flowchart TB
    subgraph SOURCE["Source Systems"]
        S1[(Database A)]
        S2[(Database B)]
        S3[(API Source)]
    end

    subgraph CDC["CDC Sync Engine"]
        subgraph MODES["Sync Modes"]
            FULL[Full Sync<br/>Complete reload]
            INC[Incremental<br/>Delta changes]
            STR[Streaming<br/>Real-time]
        end

        subgraph DETECT["Change Detection"]
            D1[Timestamp-based]
            D2[Hash-based]
            D3[Log-based]
        end

        subgraph TRANSFORM["Transformation"]
            T1[Schema Mapping]
            T2[Type Conversion]
            T3[Value Transform]
        end
    end

    subgraph TARGET["Target: Ontology"]
        TG1[Entities Update]
        TG2[Relations Update]
        TG3[Lineage Update]
    end

    subgraph AUDIT["Audit Trail"]
        A1[Change Log]
        A2[Sync History]
        A3[Error Log]
    end

    SOURCE --> MODES
    MODES --> DETECT
    DETECT --> TRANSFORM
    TRANSFORM --> TARGET
    TARGET --> AUDIT

    style CDC fill:#ffccbc
    style MODES fill:#fff9c4
    style DETECT fill:#e3f2fd
```

## 10. Actions Engine 데이터 흐름 (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["Governance Decisions"]
        I1["APPROVED decisions"]
        I2["PROVISIONAL decisions"]
        I3["Quality issues"]
    end

    subgraph ENGINE["Actions Engine"]
        subgraph TYPES["Action Types"]
            DA[DataAction<br/>Create/Update/Delete]
            GA[GovernanceAction<br/>Approval/Review]
            QA[QualityAction<br/>Fix/Validate]
        end

        subgraph PRIORITY["Priority Scoring"]
            P1[Impact Score]
            P2[Urgency Score]
            P3[Dependency Score]
            PS[Combined Priority]
        end

        subgraph EXEC["Execution"]
            E1[Synchronous]
            E2[Asynchronous]
            E3[Scheduled]
        end
    end

    subgraph OUTPUT["Action Results"]
        O1[Completed Actions]
        O2[Pending Actions]
        O3[Failed Actions]
        O4[Action Audit Log]
    end

    INPUT --> TYPES
    DA & GA & QA --> PRIORITY
    P1 & P2 & P3 --> PS
    PS --> EXEC
    EXEC --> OUTPUT

    style ENGINE fill:#e1bee7
    style TYPES fill:#f3e5f5
    style PRIORITY fill:#fff9c4
```

## 11. 아티팩트 출력 구조 (v1.0)

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
            LNG[lineage.json]
            ACT[actions.json]
        end
    end

    subgraph ONT_CONTENT["ontology.json v1.0"]
        OC1["entities: [...]"]
        OC2["relationships: [...]"]
        OC3["concepts: [...]"]
        OC4["owl2_axioms: [...]"]
        OC5["shacl_shapes: [...]"]
    end

    subgraph LNG_CONTENT["lineage.json v1.0"]
        LC1["column_nodes: [...]"]
        LC2["lineage_edges: [...]"]
        LC3["impact_analysis: [...]"]
        LC4["transformation_graph: [...]"]
    end

    subgraph GOV_CONTENT["governance_decisions.json v1.0"]
        GC1["concept_id: string"]
        GC2["decision_type: string"]
        GC3["agent_opinions: {...}"]
        GC4["recommended_actions: [...]"]
        GC5["made_at: ISO8601"]
    end

    ONT --> ONT_CONTENT
    LNG --> LNG_CONTENT
    GOV --> GOV_CONTENT

    style ARTIFACTS fill:#e8f5e9
    style ONT_CONTENT fill:#bbdefb
    style LNG_CONTENT fill:#b2dfdb
    style GOV_CONTENT fill:#e1bee7
```
