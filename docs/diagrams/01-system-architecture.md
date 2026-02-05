# 시스템 아키텍처 다이어그램

> **버전**: v1.0
> **최종 업데이트**: 2026-01-27

## 1. 전체 시스템 개요

```mermaid
graph TB
    subgraph "Data Sources"
        DS1[(CSV Files)]
        DS2[(JSON Files)]
        DS3[(Parquet Files)]
        DS4[(Excel Files)]
    end

    subgraph "Ontoloty Pipeline v1.0"
        subgraph "Entry Point"
            UP[Unified Pipeline<br/>unified_main.py]
            AP[Autonomous Pipeline<br/>autonomous_pipeline.py]
        end

        subgraph "Core Orchestration"
            AO[Agent Orchestrator<br/>orchestrator.py]
            TM[Todo Manager<br/>DAG-based]
            SC[Shared Context<br/>shared_context.py]
            EC[Evidence Chain<br/>Blockchain-style]
            AB[Agent Bus v1.0<br/>Direct/Pub-Sub]
        end

        subgraph "Consensus Engine v8.0+"
            CE[BFT Consensus]
            DS[Dempster-Shafer<br/>Fusion]
            EV[Enhanced Validator]
        end

        subgraph "Analysis Modules v1.0"
            TDA[TDA Analyzer<br/>tda.py]
            FK[Enhanced FK Pipeline v1.0<br/>Multi-Signal + Direction]
            CFK[Composite FK Detector]
            HFD[Hierarchy Detector]
            TFK[Temporal FK Detector]
            CEC[Cross-Entity Correlation<br/>v10.0]
            BI[Business Insights<br/>Generator]
        end

        subgraph "Semantic Reasoning v1.0"
            OWL[OWL2 Reasoner<br/>owl2_reasoner.py]
            SHACL[SHACL Validator<br/>shacl_validator.py]
        end

        subgraph "Enterprise Features v1.0"
            ACT[Actions Engine<br/>actions_engine.py]
            CDC[CDC Sync Engine<br/>cdc_sync.py]
            LIN[Lineage Tracker<br/>column_lineage.py]
            MEM[Agent Memory<br/>agent_memory.py]
        end
    end

    subgraph "Output Layer"
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

    subgraph "External Services"
        LLM[LLM Service<br/>Claude/OpenAI]
    end

    DS1 & DS2 & DS3 & DS4 --> UP
    UP --> AP
    AP --> AO
    AO --> TM
    AO --> SC
    AO --> CE
    AO --> AB
    TM --> TDA & FK & CFK & HFD & TFK & CEC
    CEC --> BI
    CE --> DS --> EV
    AO <--> LLM
    SC --> EC
    FK --> OWL --> SHACL
    SHACL --> ACT
    ACT --> CDC
    CDC --> LIN
    EC --> O_FK & O_IFK & O_ONT & O_TDA & O_ENT & O_INS & O_COR & O_GOV & O_EC & O_KG & O_LLM & O_DOM
    MEM --> AO

    style UP fill:#e1f5fe
    style AP fill:#e1f5fe
    style AO fill:#fff3e0
    style CE fill:#fce4ec
    style AB fill:#b2dfdb
    style CEC fill:#c8e6c9
    style OWL fill:#f3e5f5
    style SHACL fill:#f3e5f5
    style ACT fill:#e1bee7
    style LIN fill:#c8e6c9
    style LLM fill:#f3e5f5
```

## 2. 3단계 파이프라인 구조 (v1.0)

```mermaid
graph LR
    subgraph Phase1["Phase 1: Discovery"]
        P1A[Data Analyst]
        P1B[TDA Expert]
        P1C[Schema Analyst]
        P1D[Value Matcher]
        P1E[Entity Classifier]
        P1F[Relationship Detector<br/>v1.0 Enhanced]
    end

    subgraph Phase2["Phase 2: Refinement"]
        P2A[Ontology Architect<br/>+ OWL2 Reasoner]
        P2B[Conflict Resolver]
        P2C[Quality Judge<br/>+ SHACL Validator]
        P2D[Semantic Validator]
    end

    subgraph Phase3["Phase 3: Governance"]
        P3A[Governance Strategist<br/>+ Actions Engine]
        P3B[Action Prioritizer]
        P3C[Risk Assessor<br/>+ Impact Analysis]
        P3D[Policy Generator<br/>+ CDC Sync]
    end

    Phase1 --> Phase2 --> Phase3

    style Phase1 fill:#bbdefb
    style Phase2 fill:#fff9c4
    style Phase3 fill:#e1bee7
```

## 3. v1.0 모듈 아키텍처

```mermaid
graph TB
    subgraph "Enhanced FK Detection v1.0"
        direction TB
        EFP[Enhanced FK Pipeline]
        MSS[Multi-Signal Scorer<br/>5 Signals]
        FDA[Direction Analyzer<br/>5 Evidence Types]
        SER[Semantic Entity Resolver]

        subgraph "Advanced Detectors"
            CFK[Composite FK<br/>Multi-column Keys]
            HFD[Hierarchy Detector<br/>Self-referencing]
            TFK[Temporal FK<br/>Time-based Relations]
        end

        EFP --> MSS --> FDA --> SER
        SER --> CFK & HFD & TFK
    end

    subgraph "Semantic Reasoning v1.0"
        direction TB
        OWL[OWL2 Reasoner]

        subgraph "OWL2 Axioms"
            SUB[SubClassOf]
            EQV[EquivalentClass]
            DIS[DisjointClasses]
            DOM[PropertyDomain/Range]
        end

        SHACL[SHACL Validator]

        subgraph "SHACL Constraints"
            SHP[NodeShape]
            DT[Datatype Constraints]
            CARD[Cardinality]
            PAT[Pattern Matching]
        end

        OWL --> SUB & EQV & DIS & DOM
        SHACL --> SHP & DT & CARD & PAT
    end

    subgraph "Enterprise Features v1.0"
        direction TB
        ACT[Actions Engine]

        subgraph "Action Types"
            DA[DataAction]
            GA[GovernanceAction]
            QA[QualityAction]
        end

        CDC[CDC Sync Engine]

        subgraph "Sync Modes"
            FULL[Full Sync]
            INC[Incremental]
            STR[Streaming]
        end

        LIN[Lineage Tracker]

        subgraph "Lineage Types"
            COL[Column Lineage]
            TRF[Transformations]
            IMP[Impact Analysis]
        end

        ACT --> DA & GA & QA
        CDC --> FULL & INC & STR
        LIN --> COL & TRF & IMP
    end

    style EFP fill:#c8e6c9
    style OWL fill:#f3e5f5
    style SHACL fill:#f3e5f5
    style ACT fill:#e1bee7
    style CDC fill:#ffccbc
    style LIN fill:#b2dfdb
```

## 4. Agent Communication Bus (v1.0)

```mermaid
graph TB
    subgraph "Agent Communication Bus"
        direction TB
        BUS[AgentCommunicationBus<br/>Singleton]

        subgraph "Communication Patterns"
            DM[Direct Messaging<br/>send_to(agent_id, msg)]
            BC[Broadcast<br/>broadcast(msg)]
            RR[Request/Response<br/>request(agent_id, msg)]
            PS[Pub/Sub<br/>subscribe/publish(topic)]
        end

        subgraph "Message Types"
            INFO[INFO - 정보 공유]
            DISC[DISCOVERY - 발견 결과]
            REQ[REQUEST - 분석 요청]
            VAL[VALIDATION - 검증 요청]
            COL[COLLABORATION - 협업]
        end

        subgraph "Priority Levels"
            LOW[LOW - 0]
            NORMAL[NORMAL - 1]
            HIGH[HIGH - 2]
            URGENT[URGENT - 3]
        end

        BUS --> DM & BC & RR & PS
        DM & BC & RR & PS --> INFO & DISC & REQ & VAL & COL
    end

    subgraph "Agent Subscribers"
        A1[Discovery Agents<br/>6 agents]
        A2[Refinement Agents<br/>4 agents]
        A3[Governance Agents<br/>4 agents]
    end

    BUS --> A1 & A2 & A3

    style BUS fill:#b2dfdb
    style DM fill:#e3f2fd
    style PS fill:#fff9c4
```

## 5. 기술 스택 (v1.0)

```mermaid
graph LR
    subgraph "Backend"
        PY[Python 3.10+]
        ASYNC[Asyncio]
        PD[Pandas]
        NX[NetworkX]
    end

    subgraph "AI/ML"
        LLM[Claude/OpenAI API]
        TDA[TDA<br/>Betti Numbers]
        DS[Dempster-Shafer]
    end

    subgraph "Semantic Web"
        OWL[OWL2 Ontology]
        SHACL[SHACL Shapes]
        RDF[RDF Graph]
    end

    subgraph "Data Processing"
        JSON[(JSON Storage)]
        LOG[(Job Logger)]
        EC[(Evidence Chain)]
        LIN[(Lineage Graph)]
    end

    PY --> ASYNC
    PY --> PD & NX
    PY --> LLM
    LLM --> TDA & DS
    PD --> OWL --> SHACL
    SHACL --> JSON
    JSON --> LOG
    LOG --> EC --> LIN

    style PY fill:#3776ab,color:#fff
    style LLM fill:#412991,color:#fff
    style DS fill:#009688,color:#fff
    style OWL fill:#7b1fa2,color:#fff
    style SHACL fill:#7b1fa2,color:#fff
```

## 6. 디렉토리 구조 (v1.0)

```mermaid
graph TB
    subgraph ROOT["src/unified_pipeline/"]
        MAIN[unified_main.py<br/>Entry Point]

        subgraph AUTO["autonomous/"]
            ORCH[orchestrator.py]
            PIPE[autonomous_pipeline.py]
            CTX[shared_context.py]
            ABUS[agent_bus.py v1.0]

            subgraph AGENTS["agents/"]
                DISC[discovery.py<br/>6 Agents]
                REF[refinement.py<br/>4 Agents]
                GOV[governance.py<br/>4 Agents]
            end

            subgraph ANALYSIS["analysis/"]
                TDA_M[tda.py]
                EFP[enhanced_fk_pipeline.py]
                FSS[fk_signal_scorer.py]
                FDA[fk_direction_analyzer.py]
                SER[semantic_entity_resolver.py]
                CFK[composite_fk_detector.py]
                HFD[hierarchy_detector.py]
                TFK[temporal_fk_detector.py]
                CEC_M[cross_entity_correlation.py]
                BI_M[business_insights.py]
            end

            subgraph SEMANTIC["autonomous/analysis/"]
                OWL_M[owl2_reasoner.py]
                SHACL_M[shacl_validator.py]
            end

            subgraph ENTERPRISE["enterprise/"]
                ACT_M[actions_engine.py]
                CDC_M[cdc_sync.py]
            end

            subgraph LINEAGE["lineage/"]
                COL_LIN[column_lineage.py]
            end

            subgraph LEARNING["learning/"]
                MEM[agent_memory.py]
            end

            subgraph CONSENSUS["consensus/"]
                ENG[engine.py<br/>BFT + DS + AgentBus]
            end

            subgraph TODO["todo/"]
                MOD[models.py]
                MGR[manager.py<br/>DAG]
            end
        end
    end

    MAIN --> ORCH
    ORCH --> PIPE
    ORCH --> ABUS
    PIPE --> AGENTS
    ORCH --> ANALYSIS
    ORCH --> SEMANTIC
    ORCH --> ENTERPRISE
    ORCH --> LINEAGE
    ORCH --> LEARNING
    ORCH --> CONSENSUS
    ORCH --> CTX
    CTX --> TODO

    style MAIN fill:#e1f5fe
    style AGENTS fill:#bbdefb
    style ANALYSIS fill:#c8e6c9
    style SEMANTIC fill:#f3e5f5
    style ENTERPRISE fill:#e1bee7
    style LINEAGE fill:#b2dfdb
    style LEARNING fill:#fff9c4
    style CONSENSUS fill:#fce4ec
    style ABUS fill:#b2dfdb
```

## 7. 핵심 컴포넌트 상호작용 (v1.0)

```mermaid
sequenceDiagram
    participant U as User
    participant UP as Unified Pipeline
    participant AO as Agent Orchestrator
    participant AB as Agent Bus
    participant TM as Todo Manager
    participant AG as Agents (14)
    participant CE as Consensus Engine
    participant SC as Shared Context
    participant LIN as Lineage Tracker
    participant JL as Job Logger

    U->>UP: python unified_main.py data/
    UP->>AO: initialize_pipeline()
    AO->>AB: initialize_bus()
    AO->>TM: create_phase_todos(phase_1)
    AO->>SC: initialize_context()
    AO->>LIN: initialize_lineage()

    loop Phase 1-3
        TM->>AO: get_ready_todos()
        AO->>AG: assign_and_execute()

        par Agent Communication
            AG->>AB: publish("discovery.fk", result)
            AG->>AB: subscribe("validation.request")
        end

        AG->>SC: write_results()
        AG->>LIN: track_column_lineage()
        AG->>AO: report_completion()

        alt Consensus Required
            AO->>CE: run_consensus()
            CE->>AB: broadcast(vote_request)
            CE->>CE: BFT + DS Fusion
            CE->>SC: save_decision()
        end
    end

    AO->>JL: save_artifacts()
    JL->>U: Logs/{JOB_ID}/artifacts/
```

## 8. v1.0 주요 특징

```mermaid
mindmap
    root((Ontoloty v1.0))
        Phase Architecture
            Discovery (6 Agents)
            Refinement (4 Agents)
            Governance (4 Agents)
        Enhanced FK v1.0
            Multi-Signal Scoring
            Direction Analysis
            Semantic Resolution
            Composite FK
            Hierarchy Detection
            Temporal FK
        Agent Communication Bus
            Direct Messaging
            Broadcast
            Request/Response
            Pub/Sub Topics
        Semantic Reasoning
            OWL2 Reasoner
            SHACL Validator
            Axiom Inference
            Constraint Checking
        Enterprise Features
            Actions Engine
            CDC Sync Engine
            Column Lineage
            Impact Analysis
        Agent Learning
            AgentMemory
            Pattern Recognition
            Historical Analysis
        Consensus v8.0+
            BFT Protocol
            Dempster-Shafer
            AgentBus Integration
```

## 9. 모듈 의존성 그래프

```mermaid
graph TD
    subgraph "Phase 1"
        TDA[TDA Analyzer]
        FK[Enhanced FK Pipeline]
        CFK[Composite FK]
        HFD[Hierarchy Detector]
        TFK[Temporal FK]
    end

    subgraph "Phase 2"
        OWL[OWL2 Reasoner]
        SHACL[SHACL Validator]
        CEC[Cross-Entity Correlation]
    end

    subgraph "Phase 3"
        ACT[Actions Engine]
        CDC[CDC Sync]
        LIN[Lineage Tracker]
        MEM[Agent Memory]
    end

    subgraph "Core"
        AB[Agent Bus]
        SC[Shared Context]
        CE[Consensus Engine]
    end

    TDA --> FK
    FK --> CFK & HFD & TFK
    CFK & HFD & TFK --> OWL
    OWL --> SHACL
    SHACL --> CEC
    CEC --> ACT
    ACT --> CDC
    CDC --> LIN

    AB --> TDA & FK & OWL & ACT
    SC --> TDA & FK & OWL & ACT
    CE --> AB
    MEM --> SC

    style AB fill:#b2dfdb
    style OWL fill:#f3e5f5
    style SHACL fill:#f3e5f5
    style ACT fill:#e1bee7
    style LIN fill:#c8e6c9
```
