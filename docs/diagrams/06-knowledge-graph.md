# 지식 그래프 다이어그램

> **버전**: v14.0
> **최종 업데이트**: 2026-01-19

## 1. 지식 그래프 엔진 구조

```mermaid
graph TB
    subgraph KG["Knowledge Graph Engine"]
        subgraph ENTITIES["Entity Layer"]
            E1[KGEntity]
            E2[EntityType]
            E3[Properties]
        end

        subgraph RELATIONS["Relation Layer"]
            R1[KGRelation]
            R2[RelationType]
            R3[Weight/Confidence]
        end

        subgraph COMMUNITIES["Community Layer"]
            C1[KGCommunity]
            C2[Hierarchy]
            C3[Summary]
        end

        subgraph GRAPH["Graph Engine"]
            G1[NetworkX DiGraph]
            G2[Node Index]
            G3[Edge Index]
        end
    end

    subgraph OPERATIONS["Operations"]
        O1[Path Finding]
        O2[Subgraph Extraction]
        O3[Community Detection]
        O4[Graph Reasoning]
    end

    ENTITIES --> GRAPH
    RELATIONS --> GRAPH
    GRAPH --> COMMUNITIES
    GRAPH --> OPERATIONS

    style ENTITIES fill:#e8f5e9
    style RELATIONS fill:#e3f2fd
    style COMMUNITIES fill:#fff3e0
```

## 2. 엔티티 타입

```mermaid
graph TB
    subgraph SCHEMA["Schema Entities"]
        S1[TABLE]
        S2[COLUMN]
        S3[DATABASE]
        S4[SCHEMA]
    end

    subgraph BUSINESS["Business Entities"]
        B1[SUPPLIER]
        B2[CUSTOMER]
        B3[PRODUCT]
        B4[ORDER]
        B5[SHIPMENT]
        B6[WAREHOUSE]
        B7[INVENTORY]
        B8[INVOICE]
    end

    subgraph DIMENSIONAL["Dimensional Entities"]
        D1[LOCATION]
        D2[TIME_PERIOD]
        D3[CATEGORY]
        D4[STATUS]
    end

    subgraph INSIGHT["Insight Entities (v10.0)"]
        I1[CORRELATION]
        I2[THRESHOLD_EFFECT]
        I3[PREDICTIVE_INSIGHT]
    end

    style SCHEMA fill:#e3f2fd
    style BUSINESS fill:#e8f5e9
    style DIMENSIONAL fill:#fff3e0
    style INSIGHT fill:#c8e6c9
```

## 3. 관계 타입

```mermaid
graph LR
    subgraph SCHEMA_REL["Schema Relations"]
        SR1[HAS_ATTRIBUTE]
        SR2[HAS_COLUMN]
        SR3[HOMEOMORPHIC_TO]
        SR4[FK_REFERENCE]
        SR5[DEPENDS_ON]
    end

    subgraph TRANSACTIONAL["Transactional Relations"]
        TR1[SUPPLIES]
        TR2[ORDERS_FROM]
        TR3[SHIPS_TO]
        TR4[CONTAINS]
        TR5[PAYS_FOR]
    end

    subgraph CORRELATION["Correlation Relations (v10.0)"]
        CR1[CORRELATES_WITH]
        CR2[PREDICTS]
        CR3[IMPACTS]
        CR4[COMPLEMENTS]
    end

    subgraph LINEAGE["Data Lineage Relations"]
        LR1[DERIVED_FROM]
        LR2[MAPS_TO]
        LR3[EQUIVALENT_TO]
    end

    style SCHEMA_REL fill:#e3f2fd
    style TRANSACTIONAL fill:#e8f5e9
    style CORRELATION fill:#c8e6c9
    style LINEAGE fill:#fce4ec
```

## 4. Supply Chain 예시 그래프

```mermaid
graph TB
    subgraph SUPPLIERS["Suppliers"]
        SUP1((Supplier A<br/>SUPPLIER))
        SUP2((Supplier B<br/>SUPPLIER))
    end

    subgraph PRODUCTS["Products"]
        PRD1((Product X<br/>PRODUCT))
        PRD2((Product Y<br/>PRODUCT))
    end

    subgraph ORDERS["Orders"]
        ORD1((Order 001<br/>ORDER))
        ORD2((Order 002<br/>ORDER))
    end

    subgraph CUSTOMERS["Customers"]
        CUS1((Customer 1<br/>CUSTOMER))
        CUS2((Customer 2<br/>CUSTOMER))
    end

    subgraph WAREHOUSES["Warehouses"]
        WH1((Warehouse East<br/>WAREHOUSE))
        WH2((Warehouse West<br/>WAREHOUSE))
    end

    subgraph INSIGHTS["Predictive Insights (v10.0)"]
        PI1((Capacity-Weight<br/>Correlation))
    end

    SUP1 -->|SUPPLIES| PRD1
    SUP1 -->|SUPPLIES| PRD2
    SUP2 -->|SUPPLIES| PRD2

    ORD1 -->|CONTAINS| PRD1
    ORD2 -->|CONTAINS| PRD1
    ORD2 -->|CONTAINS| PRD2

    CUS1 -->|ORDERS_FROM| ORD1
    CUS2 -->|ORDERS_FROM| ORD2

    ORD1 -->|SHIPS_TO| WH1
    ORD2 -->|SHIPS_TO| WH2

    WH1 -->|CORRELATES_WITH| PI1
    PRD1 -->|IMPACTS| PI1

    style SUP1 fill:#bbdefb
    style SUP2 fill:#bbdefb
    style PRD1 fill:#c8e6c9
    style PRD2 fill:#c8e6c9
    style ORD1 fill:#fff9c4
    style ORD2 fill:#fff9c4
    style CUS1 fill:#e1bee7
    style CUS2 fill:#e1bee7
    style WH1 fill:#ffccbc
    style WH2 fill:#ffccbc
    style PI1 fill:#b2dfdb
```

## 5. 커뮤니티 탐지

```mermaid
graph TB
    subgraph FULL["Full Knowledge Graph"]
        N1((N1))
        N2((N2))
        N3((N3))
        N4((N4))
        N5((N5))
        N6((N6))
        N7((N7))
        N8((N8))

        N1 --- N2
        N1 --- N3
        N2 --- N3
        N4 --- N5
        N4 --- N6
        N5 --- N6
        N3 --- N4
        N7 --- N8
    end

    subgraph COMMUNITIES["Detected Communities"]
        subgraph C1["Community 0<br/>Supply Chain"]
            C1N1((N1))
            C1N2((N2))
            C1N3((N3))
        end

        subgraph C2["Community 1<br/>Logistics"]
            C2N4((N4))
            C2N5((N5))
            C2N6((N6))
        end

        subgraph C3["Community 2<br/>Finance"]
            C3N7((N7))
            C3N8((N8))
        end
    end

    FULL -->|Louvain<br/>Algorithm| COMMUNITIES

    style C1 fill:#bbdefb
    style C2 fill:#c8e6c9
    style C3 fill:#fff9c4
```

## 6. 경로 탐색

```mermaid
flowchart LR
    subgraph QUERY["Query: Supplier X → Warehouse Y?"]
        Q[Find Paths<br/>max_hops=3]
    end

    subgraph GRAPH["Graph Traversal"]
        A((Supplier X))
        B((Product A))
        C((Order 1))
        D((Warehouse Y))

        A -->|SUPPLIES| B
        B -->|IN_ORDER| C
        C -->|SHIPS_TO| D
    end

    subgraph RESULT["Found Paths"]
        P1["Path 1: Supplier X → Product A → Order 1 → Warehouse Y<br/>Hops: 3, Confidence: 0.85"]
    end

    Q --> GRAPH
    GRAPH --> RESULT

    style A fill:#bbdefb
    style D fill:#c8e6c9
```

## 7. 데이터 구조

```mermaid
classDiagram
    class KGEntity {
        +String id
        +EntityType entity_type
        +String name
        +Dict properties
        +String source_table
        +String source_column
        +Float confidence
        +String created_at
        +List embeddings
        +Int community_id
        +to_dict() Dict
        +from_dict(data) KGEntity
    }

    class KGRelation {
        +String source_id
        +String target_id
        +RelationType relation_type
        +String id
        +Dict properties
        +Float weight
        +Float confidence
        +String source_evidence
        +String created_at
        +to_dict() Dict
        +from_dict(data) KGRelation
    }

    class KGCommunity {
        +Int id
        +String name
        +String description
        +List entity_ids
        +String summary
        +List key_entities
        +List key_relations
        +Int level
        +Int parent_id
        +size() Int
        +to_dict() Dict
    }

    class KnowledgeGraph {
        +String name
        +DiGraph graph
        +Dict entities
        +Dict relations
        +Dict communities
        +add_entity(entity) String
        +add_relation(relation) String
        +find_paths(src, tgt) List
        +get_neighbors(id, hops) Set
        +extract_subgraph(ids) KnowledgeGraph
        +detect_communities() Dict
        +to_json() String
    }

    KnowledgeGraph "1" *-- "*" KGEntity
    KnowledgeGraph "1" *-- "*" KGRelation
    KnowledgeGraph "1" *-- "*" KGCommunity
```

## 8. Cross-Entity Correlation 통합 (v10.0)

```mermaid
flowchart TB
    subgraph INPUT["Correlation Analysis"]
        I1["Pearson Correlation"]
        I2["Threshold Effects"]
        I3["Segment Analysis"]
    end

    subgraph KG_INTEGRATION["Knowledge Graph Integration"]
        K1["Correlation Entity 생성"]
        K2["CORRELATES_WITH 관계 추가"]
        K3["IMPACTS 관계 추가"]
        K4["PREDICTS 관계 추가"]
    end

    subgraph OUTPUT["Enhanced Graph"]
        O1["Entity: Daily_Capacity"]
        O2["Entity: Max_Weight"]
        O3["Relation: CORRELATES_WITH<br/>r=0.527, p<0.05"]
        O4["Entity: Threshold_Effect<br/>31.5 threshold"]
    end

    I1 --> K1
    I2 --> K1
    I3 --> K1
    K1 --> K2 & K3 & K4
    K2 --> O1 & O2 & O3
    K3 --> O4

    style INPUT fill:#e3f2fd
    style KG_INTEGRATION fill:#fff3e0
    style OUTPUT fill:#c8e6c9
```

## 9. 그래프 직렬화

```mermaid
flowchart TB
    subgraph KG["Knowledge Graph Object"]
        KG1[entities: Dict]
        KG2[relations: Dict]
        KG3[communities: Dict]
        KG4[graph: NetworkX]
    end

    subgraph JSON["JSON Serialization"]
        J1["name: string"]
        J2["created_at: ISO8601"]
        J3["entities: [...]"]
        J4["relations: [...]"]
        J5["communities: [...]"]
        J6["statistics: {...}"]
    end

    subgraph STORAGE["Storage Options"]
        S1[(ontology.json)]
        S2[(Knowledge Graph DB)]
        S3[(Neo4j Export)]
    end

    KG --> |to_json| JSON
    JSON --> |from_json| KG
    JSON --> STORAGE

    style KG fill:#e8f5e9
    style JSON fill:#e3f2fd
```

## 10. 그래프 통계

```mermaid
pie title Entity Distribution
    "SUPPLIER" : 15
    "CUSTOMER" : 25
    "PRODUCT" : 40
    "ORDER" : 60
    "WAREHOUSE" : 8
    "CORRELATION" : 12
    "OTHER" : 20
```

```mermaid
pie title Relation Distribution
    "SUPPLIES" : 45
    "ORDERS_FROM" : 60
    "SHIPS_TO" : 35
    "CONTAINS" : 80
    "CORRELATES_WITH" : 25
    "IMPACTS" : 15
    "OTHER" : 30
```
