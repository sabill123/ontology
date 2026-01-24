# TDA (위상 데이터 분석) 다이어그램

> **버전**: v14.0
> **최종 업데이트**: 2026-01-19

## 1. TDA 분석 파이프라인

```mermaid
flowchart TB
    subgraph INPUT["Input: Table Schema"]
        I1["Columns<br/>[id, name, amount, ...]"]
        I2["Foreign Keys<br/>[supplier_id → suppliers.id]"]
        I3["Sample Data<br/>[{row1}, {row2}, ...]"]
    end

    subgraph GRAPH["Column Graph Construction"]
        G1["0-simplex:<br/>Columns as Vertices"]
        G2["1-simplex:<br/>Column Relationships"]
        G3["2-simplex:<br/>Triangle Relationships"]
    end

    subgraph EDGES["Edge Types"]
        E1["FK Relations<br/>weight: 1.0"]
        E2["Type Similarity<br/>weight: 0.3"]
        E3["Name Similarity<br/>weight: 0.5"]
        E4["Value Coexistence<br/>weight: 0.4"]
    end

    subgraph BETTI["Betti Numbers"]
        B0["β₀: Connected Components"]
        B1["β₁: 1D Holes (Cycles)"]
        B2["β₂: 2D Voids"]
        BC["χ = β₀ - β₁ + β₂<br/>Euler Characteristic"]
    end

    subgraph PERSISTENCE["Persistence Diagram"]
        P1["Birth-Death Pairs"]
        P2["Total Persistence"]
        P3["Max Persistence"]
    end

    subgraph OUTPUT["TDA Signature"]
        O1["betti_numbers"]
        O2["persistence_pairs"]
        O3["key_columns"]
        O4["structural_complexity"]
    end

    INPUT --> GRAPH
    G1 --> EDGES
    EDGES --> G2
    G2 --> G3
    G3 --> BETTI
    BETTI --> BC
    G3 --> PERSISTENCE
    BETTI & PERSISTENCE --> OUTPUT

    style BETTI fill:#e3f2fd
    style PERSISTENCE fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

## 2. Betti Numbers 계산

```mermaid
flowchart LR
    subgraph INPUT["Column Graph"]
        direction TB
        V1((col_a))
        V2((col_b))
        V3((col_c))
        V4((col_d))
        V5((col_e))

        V1 --- V2
        V2 --- V3
        V3 --- V1
        V4 --- V5
    end

    subgraph BETA0["β₀: Components"]
        C1["Component 1:<br/>{col_a, col_b, col_c}"]
        C2["Component 2:<br/>{col_d, col_e}"]
        B0R["β₀ = 2"]
    end

    subgraph BETA1["β₁: Cycles"]
        CY1["Cycle 1:<br/>col_a → col_b → col_c → col_a"]
        B1R["β₁ = 1"]
    end

    subgraph BETA2["β₂: Voids"]
        B2R["β₂ = 0<br/>(No 3D structures)"]
    end

    subgraph EULER["Euler Characteristic"]
        EC["χ = β₀ - β₁ + β₂<br/>χ = 2 - 1 + 0 = 1"]
    end

    INPUT --> BETA0
    INPUT --> BETA1
    INPUT --> BETA2
    BETA0 & BETA1 & BETA2 --> EULER

    style BETA0 fill:#bbdefb
    style BETA1 fill:#c8e6c9
    style BETA2 fill:#fff9c4
```

## 3. Persistence Diagram

```mermaid
flowchart TB
    subgraph FILTRATION["Edge Filtration"]
        F1["t=0.0: No edges"]
        F2["t=0.3: Type similar edges"]
        F3["t=0.5: Name similar edges"]
        F4["t=1.0: FK edges"]
    end

    subgraph PAIRS["Persistence Pairs"]
        P1["(dim=0, birth=0.0, death=0.3)"]
        P2["(dim=0, birth=0.0, death=0.5)"]
        P3["(dim=0, birth=0.0, death=∞)"]
        P4["(dim=1, birth=0.5, death=1.0)"]
    end

    subgraph METRICS["Persistence Metrics"]
        M1["Total Persistence:<br/>Σ(death - birth)"]
        M2["Max Persistence:<br/>max(death - birth)"]
        M3["Persistence Entropy"]
    end

    FILTRATION --> PAIRS
    PAIRS --> METRICS

    style FILTRATION fill:#e3f2fd
    style PAIRS fill:#fff3e0
    style METRICS fill:#e8f5e9
```

## 4. Homeomorphism Detection (위상적 동형성)

```mermaid
flowchart TB
    subgraph TABLE_A["Table A: Orders"]
        TA1["Columns: 8"]
        TA2["β₀=1, β₁=2, β₂=0"]
        TA3["Complexity: medium"]
    end

    subgraph TABLE_B["Table B: Purchases"]
        TB1["Columns: 7"]
        TB2["β₀=1, β₁=2, β₂=0"]
        TB3["Complexity: medium"]
    end

    subgraph COMPARE["Comparison"]
        C1["Betti Distance:<br/>√((1-1)² + (2-2)² + (0-0)²) = 0"]
        C2["Persistence Similarity:<br/>0.92"]
        C3["Column Ratio:<br/>7/8 = 0.875"]
        C4["Complexity Match:<br/>1.0"]
    end

    subgraph SCORE["Homeomorphism Score"]
        S1["structural_score: 1.0"]
        S2["persistence_similarity: 0.92"]
        S3["column_ratio: 0.875"]
        S4["complexity_match: 1.0"]
        S5["confidence: 0.95"]
        S6{is_homeomorphic?}
        S7["YES - 구조적으로 유사!<br/>같은 엔티티로 간주"]
    end

    TABLE_A --> COMPARE
    TABLE_B --> COMPARE
    COMPARE --> SCORE
    S1 & S2 & S3 & S4 --> S5
    S5 --> S6
    S6 -->|conf > 0.6| S7

    style TABLE_A fill:#bbdefb
    style TABLE_B fill:#c8e6c9
    style SCORE fill:#fff3e0
```

## 5. Structural Complexity 계산

```mermaid
flowchart TB
    subgraph INPUTS["Inputs"]
        I1["num_columns"]
        I2["betti_numbers"]
        I3["edge_density"]
    end

    subgraph COLUMN_SCORE["Column Score"]
        CS1{columns > 20?}
        CS2["+2 points"]
        CS3{columns > 10?}
        CS4["+1 point"]
        CS5["0 points"]
    end

    subgraph CYCLE_SCORE["Cycle Score (β₁)"]
        CY1{β₁ > 5?}
        CY2["+2 points"]
        CY3{β₁ > 2?}
        CY4["+1 point"]
        CY5["0 points"]
    end

    subgraph DENSITY_SCORE["Density Score"]
        DS1{density > 0.5?}
        DS2["+2 points"]
        DS3{density > 0.2?}
        DS4["+1 point"]
        DS5["0 points"]
    end

    subgraph RESULT["Complexity Result"]
        R1{total >= 4?}
        R2["HIGH"]
        R3{total >= 2?}
        R4["MEDIUM"]
        R5["LOW"]
    end

    I1 --> CS1
    CS1 -->|Yes| CS2
    CS1 -->|No| CS3
    CS3 -->|Yes| CS4
    CS3 -->|No| CS5

    I2 --> CY1
    CY1 -->|Yes| CY2
    CY1 -->|No| CY3
    CY3 -->|Yes| CY4
    CY3 -->|No| CY5

    I3 --> DS1
    DS1 -->|Yes| DS2
    DS1 -->|No| DS3
    DS3 -->|Yes| DS4
    DS3 -->|No| DS5

    CS2 & CS4 & CS5 & CY2 & CY4 & CY5 & DS2 & DS4 & DS5 --> R1
    R1 -->|Yes| R2
    R1 -->|No| R3
    R3 -->|Yes| R4
    R3 -->|No| R5

    style RESULT fill:#e8f5e9
```

## 6. TDA 데이터 구조

```mermaid
classDiagram
    class TDASignature {
        +String table_name
        +BettiNumbers betti_numbers
        +List persistence_pairs
        +List key_columns
        +String structural_complexity
        +Float total_persistence
        +Float max_persistence
        +Int column_count
        +Int relationship_count
        +to_dict() Dict
    }

    class BettiNumbers {
        +Int beta_0
        +Int beta_1
        +Int beta_2
        +euler_characteristic() Int
        +distance(other) Float
        +to_dict() Dict
    }

    class PersistencePair {
        +Int dimension
        +Float birth
        +Float death
        +persistence() Float
    }

    class TDAAnalyzer {
        +Dict signatures
        +compute_signature(table) TDASignature
        +compute_homeomorphism_score(sig_a, sig_b) Dict
        -_build_column_graph() List
        -_compute_betti_numbers() BettiNumbers
        -_compute_persistence() List
        -_assess_complexity() String
    }

    TDASignature --> BettiNumbers
    TDASignature --> PersistencePair
    TDAAnalyzer --> TDASignature
```

## 7. TDA 활용 사례

```mermaid
mindmap
    root((TDA in<br/>Ontoloty))
        Schema Similarity
            Same structure different names
            Cross-system mapping
            Data migration validation
        Data Silo Integration
            Find related tables
            Identify duplicates
            Merge candidates
        Quality Assessment
            Structural complexity
            Relationship density
            Anomaly detection
        Ontology Design
            Entity grouping
            Hierarchy inference
            Relationship discovery
        Cycle Detection
            β₁ > 0 = Circular dependencies
            FK loop detection
            Referential integrity
```

## 8. 순환 참조 탐지 (β₁ > 0)

```mermaid
flowchart TB
    subgraph TABLES["Tables with FK Relations"]
        T1[Orders]
        T2[Products]
        T3[Suppliers]
        T4[Inventory]
    end

    subgraph RELATIONS["FK Relations"]
        T1 -->|has_product| T2
        T2 -->|from_supplier| T3
        T3 -->|stocks_in| T4
        T4 -->|for_order| T1
    end

    subgraph TDA_RESULT["TDA Analysis"]
        B1["β₀ = 1 (Connected)"]
        B2["β₁ = 1 (One Cycle!)"]
        B3["β₂ = 0"]
    end

    subgraph INSIGHT["Insight"]
        I1["순환 참조 탐지됨!"]
        I2["Order → Product → Supplier → Inventory → Order"]
        I3["권장: 순환 의존성 해소 필요"]
    end

    RELATIONS --> TDA_RESULT
    TDA_RESULT --> INSIGHT

    style B2 fill:#ffcdd2
    style INSIGHT fill:#fff9c4
```

## 9. TDA Expert 에이전트 흐름

```mermaid
sequenceDiagram
    participant AO as Agent Orchestrator
    participant TDA as TDA Expert Agent
    participant SC as SharedContext
    participant AN as TDA Analyzer

    AO->>TDA: assign_todo(compute_tda_signatures)
    TDA->>SC: read tables
    SC-->>TDA: table data

    loop For Each Table
        TDA->>AN: compute_signature(table)
        AN->>AN: build_column_graph()
        AN->>AN: compute_betti_numbers()
        AN->>AN: compute_persistence()
        AN->>AN: assess_complexity()
        AN-->>TDA: TDASignature
    end

    TDA->>AN: detect_homeomorphisms()
    AN-->>TDA: homeomorphism_pairs

    TDA->>SC: write tda_signatures
    TDA->>SC: write homeomorphisms
    TDA-->>AO: task_complete(DONE)
```

## 10. Betti Numbers 의미 요약

```mermaid
graph TB
    subgraph B0["β₀: Connected Components"]
        B0D["독립적인 테이블 그룹 수"]
        B0E["β₀ = 1: 모든 테이블 연결됨"]
        B0F["β₀ > 1: 분리된 데이터 사일로"]
    end

    subgraph B1["β₁: 1차원 구멍 (Cycles)"]
        B1D["순환 참조/루프 수"]
        B1E["β₁ = 0: 순환 없음 (트리 구조)"]
        B1F["β₁ > 0: 순환 의존성 존재"]
    end

    subgraph B2["β₂: 2차원 공동 (Voids)"]
        B2D["고차원 구조적 복잡성"]
        B2E["β₂ = 0: 대부분의 경우"]
        B2F["β₂ > 0: 매우 복잡한 관계망"]
    end

    style B0 fill:#bbdefb
    style B1 fill:#c8e6c9
    style B2 fill:#fff9c4
```
