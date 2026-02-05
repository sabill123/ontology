# 지식 그래프 다이어그램

> **버전**: v1.0
> **최종 업데이트**: 2026-01-27

## 1. 지식 그래프 엔진 구조 (v1.0)

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

    subgraph SEMANTIC["Semantic Reasoning v1.0"]
        OWL[OWL2 Axioms]
        SHACL[SHACL Shapes]
    end

    subgraph LINEAGE["Column Lineage v1.0"]
        LIN[LineageTracker]
        IMP[Impact Analysis]
    end

    subgraph OPERATIONS["Operations"]
        O1[Path Finding]
        O2[Subgraph Extraction]
        O3[Community Detection]
        O4[Graph Reasoning]
        O5[OWL2 Inference v1.0]
        O6[SHACL Validation v1.0]
    end

    ENTITIES --> GRAPH
    RELATIONS --> GRAPH
    GRAPH --> COMMUNITIES
    GRAPH --> OPERATIONS
    GRAPH --> SEMANTIC
    GRAPH --> LINEAGE

    style ENTITIES fill:#e8f5e9
    style RELATIONS fill:#e3f2fd
    style COMMUNITIES fill:#fff3e0
    style SEMANTIC fill:#f3e5f5
    style LINEAGE fill:#b2dfdb
```

## 2. 엔티티 타입 (v1.0)

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

    subgraph SEMANTIC["Semantic Entities v1.0"]
        SE1[OWL_CLASS]
        SE2[OWL_PROPERTY]
        SE3[SHACL_SHAPE]
        SE4[AXIOM]
    end

    subgraph LINEAGE["Lineage Entities v1.0"]
        LE1[COLUMN_NODE]
        LE2[TRANSFORMATION]
        LE3[IMPACT_SCOPE]
    end

    style SCHEMA fill:#e3f2fd
    style BUSINESS fill:#e8f5e9
    style DIMENSIONAL fill:#fff3e0
    style INSIGHT fill:#c8e6c9
    style SEMANTIC fill:#f3e5f5
    style LINEAGE fill:#b2dfdb
```

## 3. 관계 타입 (v1.0)

```mermaid
graph LR
    subgraph SCHEMA_REL["Schema Relations"]
        SR1[HAS_ATTRIBUTE]
        SR2[HAS_COLUMN]
        SR3[HOMEOMORPHIC_TO]
        SR4[FK_REFERENCE]
        SR5[DEPENDS_ON]
    end

    subgraph ADVANCED_FK["Advanced FK Relations v1.0"]
        AF1[COMPOSITE_FK<br/>Multi-column]
        AF2[HIERARCHICAL_FK<br/>Self-referencing]
        AF3[TEMPORAL_FK<br/>Time-based]
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

    subgraph SEMANTIC_REL["Semantic Relations v1.0"]
        SER1[SUBCLASS_OF]
        SER2[EQUIVALENT_TO]
        SER3[DISJOINT_WITH]
        SER4[HAS_DOMAIN]
        SER5[HAS_RANGE]
        SER6[INVERSE_OF]
    end

    subgraph LINEAGE_REL["Lineage Relations v1.0"]
        LR1[DERIVED_FROM]
        LR2[AGGREGATED_FROM]
        LR3[FILTERED_FROM]
        LR4[TRANSFORMED_BY]
        LR5[AFFECTS]
    end

    style SCHEMA_REL fill:#e3f2fd
    style ADVANCED_FK fill:#c8e6c9
    style TRANSACTIONAL fill:#e8f5e9
    style CORRELATION fill:#c8e6c9
    style SEMANTIC_REL fill:#f3e5f5
    style LINEAGE_REL fill:#b2dfdb
```

## 4. OWL2 Ontology Structure (v1.0)

```mermaid
graph TB
    subgraph OWL2["OWL2 Ontology"]
        subgraph CLASSES["Class Hierarchy"]
            CLS1[Thing]
            CLS2[Entity]
            CLS3[Relationship]
            CLS4[Concept]

            CLS1 --> CLS2 & CLS3 & CLS4
        end

        subgraph AXIOMS["Axiom Types"]
            AX1[SubClassOf]
            AX2[EquivalentClass]
            AX3[DisjointClasses]
            AX4[ClassAssertion]
        end

        subgraph PROPERTIES["Property Types"]
            PR1[ObjectProperty]
            PR2[DataProperty]
            PR3[AnnotationProperty]
        end

        subgraph RESTRICTIONS["Property Restrictions"]
            RS1[Domain]
            RS2[Range]
            RS3[Cardinality]
            RS4[Transitivity]
            RS5[Inverse]
        end
    end

    CLASSES --> AXIOMS
    AXIOMS --> PROPERTIES
    PROPERTIES --> RESTRICTIONS

    style OWL2 fill:#f3e5f5
    style AXIOMS fill:#e1bee7
    style RESTRICTIONS fill:#fce4ec
```

## 5. SHACL Shapes Structure (v1.0)

```mermaid
graph TB
    subgraph SHACL["SHACL Shapes Graph"]
        subgraph NODE_SHAPES["NodeShape"]
            NS1[CustomerShape]
            NS2[OrderShape]
            NS3[ProductShape]
        end

        subgraph PROP_SHAPES["PropertyShape"]
            PS1[sh:datatype]
            PS2[sh:minCount]
            PS3[sh:maxCount]
            PS4[sh:pattern]
            PS5[sh:minValue]
            PS6[sh:maxValue]
            PS7[sh:in]
        end

        subgraph TARGETS["Target Types"]
            TG1[sh:targetClass]
            TG2[sh:targetNode]
            TG3[sh:targetSubjectsOf]
        end

        subgraph SEVERITY["Severity Levels"]
            SV1[sh:Violation]
            SV2[sh:Warning]
            SV3[sh:Info]
        end
    end

    NODE_SHAPES --> PROP_SHAPES
    NODE_SHAPES --> TARGETS
    PROP_SHAPES --> SEVERITY

    style SHACL fill:#f3e5f5
    style NODE_SHAPES fill:#e1bee7
    style PROP_SHAPES fill:#fce4ec
```

## 6. Column Lineage Graph (v1.0)

```mermaid
graph TB
    subgraph SOURCE["Source Columns"]
        S1((orders.customer_id<br/>SOURCE))
        S2((orders.amount<br/>SOURCE))
        S3((customers.name<br/>SOURCE))
    end

    subgraph DERIVED["Derived Columns"]
        D1((sales.customer_id<br/>DERIVED))
        D2((summary.total_amount<br/>AGGREGATE))
        D3((report.customer_name<br/>DIRECT))
    end

    subgraph EDGE_TYPES["Edge Types"]
        E1[DIRECT - Identity]
        E2[AGGREGATED - SUM/AVG/COUNT]
        E3[FILTERED - WHERE clause]
        E4[DERIVED - Calculation]
    end

    S1 -->|DIRECT| D1
    S2 -->|AGGREGATED| D2
    S3 -->|DIRECT| D3

    style SOURCE fill:#bbdefb
    style DERIVED fill:#c8e6c9
    style EDGE_TYPES fill:#fff9c4
```

## 7. Impact Analysis Flow (v1.0)

```mermaid
flowchart TB
    subgraph INPUT["Changed Column"]
        I1[customers.email<br/>Schema Change]
    end

    subgraph UPSTREAM["Upstream Impact"]
        U1["What affects this column?"]
        U2[Source Tables]
        U3[ETL Processes]
    end

    subgraph DOWNSTREAM["Downstream Impact"]
        D1["What does this column affect?"]
        D2[Derived Columns<br/>3 affected]
        D3[Reports<br/>2 affected]
        D4[API Endpoints<br/>1 affected]
    end

    subgraph ANALYSIS["Impact Score"]
        A1["Total Affected: 6"]
        A2["Critical: 2"]
        A3["High: 3"]
        A4["Low: 1"]
    end

    subgraph OUTPUT["Impact Report"]
        O1[affected_columns]
        O2[affected_reports]
        O3[recommended_actions]
    end

    INPUT --> UPSTREAM & DOWNSTREAM
    U1 --> U2 --> U3
    D1 --> D2 & D3 & D4
    D2 & D3 & D4 --> ANALYSIS
    ANALYSIS --> OUTPUT

    style INPUT fill:#ffebee
    style UPSTREAM fill:#e3f2fd
    style DOWNSTREAM fill:#fff3e0
    style ANALYSIS fill:#c8e6c9
```

## 8. Supply Chain 예시 그래프 (v1.0)

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

    subgraph OWL_AXIOMS["OWL2 Axioms v1.0"]
        OA1{{SubClassOf<br/>Order → Transaction}}
        OA2{{DisjointClasses<br/>Supplier ⊥ Customer}}
    end

    subgraph LINEAGE["Column Lineage v1.0"]
        LN1[orders.amount → summary.total]
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

    ORD1 -.->|INSTANCE_OF| OA1
    SUP1 -.->|INSTANCE_OF| OA2

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
    style OA1 fill:#f3e5f5
    style OA2 fill:#f3e5f5
```

## 9. 데이터 구조 (v1.0)

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
        +List owl2_axioms v1.0
        +String shacl_shape_id v1.0
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
        +String fk_type v1.0
        +to_dict() Dict
        +from_dict(data) KGRelation
    }

    class OWL2Axiom {
        +String axiom_id
        +AxiomType axiom_type
        +List subject_classes
        +List object_classes
        +String property
        +Float confidence
        +Dict restrictions
        +to_owl2_syntax() String
    }

    class SHACLShape {
        +String shape_id
        +String target_class
        +List property_shapes
        +Bool is_closed
        +List constraints
        +validate(data) ValidationReport
    }

    class ColumnNode {
        +String node_id
        +String table_name
        +String column_name
        +ColumnType column_type
        +List upstream_edges
        +List downstream_edges
        +get_impact_scope() ImpactAnalysis
    }

    class LineageEdge {
        +String edge_id
        +String source_node_id
        +String target_node_id
        +LineageEdgeType edge_type
        +String transformation
        +Float confidence
    }

    class KnowledgeGraph {
        +String name
        +DiGraph graph
        +Dict entities
        +Dict relations
        +Dict communities
        +Dict owl2_axioms v1.0
        +Dict shacl_shapes v1.0
        +LineageTracker lineage v1.0
        +add_entity(entity) String
        +add_relation(relation) String
        +add_axiom(axiom) String
        +add_shape(shape) String
        +find_paths(src, tgt) List
        +get_neighbors(id, hops) Set
        +extract_subgraph(ids) KnowledgeGraph
        +detect_communities() Dict
        +infer_axioms() List
        +validate_shapes() ValidationReport
        +to_json() String
    }

    KnowledgeGraph "1" *-- "*" KGEntity
    KnowledgeGraph "1" *-- "*" KGRelation
    KnowledgeGraph "1" *-- "*" OWL2Axiom
    KnowledgeGraph "1" *-- "*" SHACLShape
    KnowledgeGraph "1" *-- "1" LineageTracker
    LineageTracker "1" *-- "*" ColumnNode
    LineageTracker "1" *-- "*" LineageEdge
```

## 10. 그래프 통계 (v1.0)

```mermaid
pie title Entity Distribution v1.0
    "SUPPLIER" : 15
    "CUSTOMER" : 25
    "PRODUCT" : 40
    "ORDER" : 60
    "WAREHOUSE" : 8
    "CORRELATION" : 12
    "OWL_CLASS" : 18
    "SHACL_SHAPE" : 14
    "COLUMN_NODE" : 45
    "OTHER" : 20
```

```mermaid
pie title Relation Distribution v1.0
    "SUPPLIES" : 45
    "ORDERS_FROM" : 60
    "SHIPS_TO" : 35
    "CONTAINS" : 80
    "CORRELATES_WITH" : 25
    "IMPACTS" : 15
    "SUBCLASS_OF" : 30
    "DERIVED_FROM" : 55
    "OTHER" : 30
```

## 11. 그래프 직렬화 (v1.0)

```mermaid
flowchart TB
    subgraph KG["Knowledge Graph Object v1.0"]
        KG1[entities: Dict]
        KG2[relations: Dict]
        KG3[communities: Dict]
        KG4[graph: NetworkX]
        KG5[owl2_axioms: Dict]
        KG6[shacl_shapes: Dict]
        KG7[lineage: LineageTracker]
    end

    subgraph JSON["JSON Serialization"]
        J1["name: string"]
        J2["created_at: ISO8601"]
        J3["entities: [...]"]
        J4["relations: [...]"]
        J5["communities: [...]"]
        J6["owl2_axioms: [...]"]
        J7["shacl_shapes: [...]"]
        J8["lineage: {...}"]
        J9["statistics: {...}"]
    end

    subgraph STORAGE["Storage Options"]
        S1[(ontology.json)]
        S2[(lineage.json)]
        S3[(Knowledge Graph DB)]
        S4[(Neo4j Export)]
    end

    KG --> |to_json| JSON
    JSON --> |from_json| KG
    JSON --> STORAGE

    style KG fill:#e8f5e9
    style JSON fill:#e3f2fd
    style STORAGE fill:#fff3e0
```
