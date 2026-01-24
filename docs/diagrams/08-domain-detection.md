# 도메인 탐지 다이어그램

> **버전**: v14.0
> **최종 업데이트**: 2026-01-19

## 1. 도메인 탐지 파이프라인 (LLM-First v13.0)

```mermaid
flowchart TB
    subgraph INPUT["Input Data"]
        I1["Table Names"]
        I2["Column Names"]
        I3["Sample Values"]
        I4["FK Relationships"]
    end

    subgraph LLM_ANALYSIS["LLM-First Analysis (v13.0)"]
        L1["전체 데이터 LLM 전달"]
        L2["도메인 패턴 인식"]
        L3["비즈니스 맥락 이해"]
        L4["신뢰도 계산"]
    end

    subgraph SCORING["Domain Scoring"]
        S1["Supply Chain Score"]
        S2["Healthcare Score"]
        S3["Retail Score"]
        S4["Finance Score"]
        S5["Web Analytics Score"]
    end

    subgraph RESULT["Detection Result"]
        R1{Highest Score?}
        R2["Primary Domain"]
        R3["Secondary Domains"]
        R4["Confidence Level"]
    end

    subgraph CONFIG["Domain Config"]
        C1["Entity Mappings"]
        C2["KPI Templates"]
        C3["Business Rules"]
        C4["Insight Patterns"]
    end

    INPUT --> LLM_ANALYSIS
    L1 --> L2 --> L3 --> L4
    L4 --> SCORING
    S1 & S2 & S3 & S4 & S5 --> R1
    R1 --> R2 & R3 & R4
    R2 --> CONFIG

    style LLM_ANALYSIS fill:#e3f2fd
    style SCORING fill:#fff3e0
    style CONFIG fill:#c8e6c9
```

## 2. 지원 도메인

```mermaid
mindmap
    root((Industry<br/>Domains))
        Supply Chain
            Suppliers
            Purchase Orders
            Shipments
            Warehouses
            Logistics
            Freight Rates
        Healthcare
            Patient Records
            Medical Procedures
            Prescriptions
            Lab Results
            Insurance Claims
        Retail
            Products
            Customers
            Orders
            Inventory
            Promotions
        Finance
            Accounts
            Transactions
            Loans
            Investments
            Risk Metrics
        Web Analytics
            Page Views
            Sessions
            Visitors
            Channels
            Conversions
            Traffic Sources
```

## 3. 도메인별 키워드 패턴

```mermaid
graph TB
    subgraph SUPPLY_CHAIN["Supply Chain Keywords"]
        SC1["supplier"]
        SC2["shipment"]
        SC3["warehouse"]
        SC4["logistics"]
        SC5["freight"]
        SC6["plant"]
        SC7["carrier"]
    end

    subgraph HEALTHCARE["Healthcare Keywords"]
        H1["patient"]
        H2["diagnosis"]
        H3["prescription"]
        H4["icd_code"]
        H5["admission"]
        H6["discharge"]
    end

    subgraph RETAIL["Retail Keywords"]
        R1["product"]
        R2["sku"]
        R3["cart"]
        R4["checkout"]
        R5["promotion"]
        R6["category"]
    end

    subgraph WEB_ANALYTICS["Web Analytics Keywords"]
        W1["page_view"]
        W2["session"]
        W3["visitor"]
        W4["channel"]
        W5["conversion"]
        W6["bounce_rate"]
    end

    style SUPPLY_CHAIN fill:#ffccbc
    style HEALTHCARE fill:#e3f2fd
    style RETAIL fill:#c8e6c9
    style WEB_ANALYTICS fill:#e1bee7
```

## 4. Data Analyst v13.0 도메인 탐지

```mermaid
sequenceDiagram
    participant AO as Agent Orchestrator
    participant DA as Data Analyst Agent
    participant LLM as LLM Service
    participant SC as SharedContext

    AO->>DA: assign_todo(analyze_domain)
    DA->>SC: read table_data
    SC-->>DA: tables with samples

    DA->>DA: prepare_llm_prompt()
    Note over DA: 전체 테이블 정보 포함<br/>컬럼명, 샘플값, 통계

    DA->>LLM: analyze_domain(prompt)
    LLM-->>DA: domain_analysis_result

    Note over DA: Result includes:<br/>- primary_domain<br/>- confidence<br/>- column_classifications<br/>- business_context

    DA->>SC: write domain_context
    DA->>SC: write column_profiles
    DA-->>AO: task_complete(DONE)
```

## 5. Supply Chain 도메인 예시

```mermaid
graph TB
    subgraph TABLES["Detected Tables"]
        T1[freight_rates]
        T2[warehouse_capacities]
        T3[products_per_plant]
        T4[order_list]
        T5[vmi_customers]
    end

    subgraph ENTITIES["Supply Chain Entities"]
        E1[FREIGHT_RATE]
        E2[WAREHOUSE]
        E3[PLANT]
        E4[ORDER]
        E5[CUSTOMER]
    end

    subgraph RELATIONS["Supply Chain Relations"]
        R1[SHIPS_FROM]
        R2[STORES_IN]
        R3[PRODUCES_AT]
        R4[ORDERED_BY]
    end

    subgraph INSIGHTS["Domain Insights"]
        I1["창고 용량 ↔ 운송 중량 상관관계"]
        I2["Plant별 제품 집중도 분석"]
        I3["고객별 주문 패턴"]
    end

    T1 --> E1
    T2 --> E2
    T3 --> E3
    T4 --> E4
    T5 --> E5

    E1 --> R1 --> E2
    E3 --> R3 --> E4
    E5 --> R4 --> E4

    E1 & E2 & E3 --> INSIGHTS

    style TABLES fill:#ffccbc
    style ENTITIES fill:#c8e6c9
    style INSIGHTS fill:#e1bee7
```

## 6. Web Analytics 도메인 예시

```mermaid
graph TB
    subgraph TABLES["Detected Tables"]
        T1[page_channel_data]
        T2[session_metrics]
        T3[visitor_data]
    end

    subgraph ENTITIES["Web Analytics Entities"]
        E1[PAGE]
        E2[CHANNEL]
        E3[SESSION]
        E4[VISITOR]
        E5[METRIC]
    end

    subgraph RELATIONS["Analytics Relations"]
        R1[VIEWED_ON]
        R2[ATTRIBUTED_TO]
        R3[CONVERTED_FROM]
    end

    subgraph INSIGHTS["Domain Insights"]
        I1["채널별 전환율 상관관계"]
        I2["신규 vs 재방문 비율 분석"]
        I3["세션 지속시간 ↔ 전환 관계"]
    end

    T1 --> E1 & E2
    T2 --> E3
    T3 --> E4

    E1 --> R1 --> E2
    E3 --> R2 --> E2
    E4 --> R3 --> E5

    E1 & E2 & E3 --> INSIGHTS

    style TABLES fill:#e1bee7
    style ENTITIES fill:#c8e6c9
    style INSIGHTS fill:#fff9c4
```

## 7. 컬럼 비즈니스 특성 분류

```mermaid
flowchart TB
    subgraph COLUMN_TYPES["Column Business Types"]
        CT1["identifier<br/>고유 식별자"]
        CT2["required_input<br/>필수 입력 필드"]
        CT3["metric<br/>측정 지표"]
        CT4["dimension<br/>분석 차원"]
        CT5["timestamp<br/>시간 관련"]
    end

    subgraph EXAMPLES["Examples"]
        EX1["order_id, customer_id"]
        EX2["plant_code, carrier_code"]
        EX3["rate, weight, capacity"]
        EX4["channel, category, region"]
        EX5["created_at, order_date"]
    end

    subgraph USAGE["Usage in Analysis"]
        U1["PK/FK 관계 추론"]
        U2["엔티티 속성 정의"]
        U3["Cross-Entity Correlation"]
        U4["세그먼트 분석"]
        U5["시계열 분석"]
    end

    CT1 --> EX1 --> U1
    CT2 --> EX2 --> U2
    CT3 --> EX3 --> U3
    CT4 --> EX4 --> U4
    CT5 --> EX5 --> U5

    style COLUMN_TYPES fill:#e3f2fd
    style EXAMPLES fill:#fff3e0
    style USAGE fill:#c8e6c9
```

## 8. 도메인 신뢰도 계산

```mermaid
flowchart TB
    subgraph INPUT["Input Features"]
        I1["Table name matches: 5"]
        I2["Column name matches: 12"]
        I3["Value pattern matches: 8"]
        I4["Entity type matches: 4"]
    end

    subgraph WEIGHTS["Scoring Weights"]
        W1["Table: 0.3"]
        W2["Column: 0.25"]
        W3["Value: 0.25"]
        W4["Entity: 0.2"]
    end

    subgraph CALCULATION["Score Calculation"]
        C1["Supply Chain:<br/>5×0.3 + 12×0.25 + 8×0.25 + 4×0.2"]
        C2["= 1.5 + 3.0 + 2.0 + 0.8"]
        C3["= 7.3"]
    end

    subgraph NORMALIZE["Normalization"]
        N1["Max Score: 10"]
        N2["Confidence: 7.3/10 = 73%"]
    end

    subgraph RESULT["Result"]
        R1["Domain: Supply Chain"]
        R2["Confidence: 73%"]
    end

    INPUT --> WEIGHTS
    WEIGHTS --> CALCULATION
    CALCULATION --> NORMALIZE
    NORMALIZE --> RESULT

    style CALCULATION fill:#fff3e0
    style RESULT fill:#c8e6c9
```

## 9. 도메인별 KPI 템플릿

```mermaid
flowchart TB
    subgraph SUPPLY_CHAIN_KPI["Supply Chain KPIs"]
        SCK1["On-Time Delivery Rate"]
        SCK2["Inventory Turnover"]
        SCK3["Order Fulfillment Rate"]
        SCK4["Supplier Lead Time"]
        SCK5["Warehouse Utilization"]
    end

    subgraph WEB_ANALYTICS_KPI["Web Analytics KPIs"]
        WAK1["Conversion Rate"]
        WAK2["Bounce Rate"]
        WAK3["Session Duration"]
        WAK4["Pages per Session"]
        WAK5["New vs Returning Visitors"]
    end

    subgraph HEALTHCARE_KPI["Healthcare KPIs"]
        HK1["Readmission Rate"]
        HK2["Average Length of Stay"]
        HK3["Claim Denial Rate"]
        HK4["Patient Satisfaction"]
    end

    subgraph RETAIL_KPI["Retail KPIs"]
        RK1["Average Order Value"]
        RK2["Customer Lifetime Value"]
        RK3["Cart Abandonment Rate"]
        RK4["Product Return Rate"]
    end

    style SUPPLY_CHAIN_KPI fill:#ffccbc
    style WEB_ANALYTICS_KPI fill:#e1bee7
    style HEALTHCARE_KPI fill:#e3f2fd
    style RETAIL_KPI fill:#c8e6c9
```

## 10. 도메인 컨텍스트 데이터 구조

```mermaid
classDiagram
    class DomainContext {
        +String domain_name
        +Float confidence
        +List detected_entities
        +List detected_relations
        +Dict domain_config
        +List kpi_templates
        +get_entity_mapping() Dict
        +get_relation_mapping() Dict
    }

    class ColumnProfile {
        +String column_name
        +String table_name
        +String business_type
        +String semantic_type
        +Float importance_score
        +Dict statistics
    }

    class DomainDetector {
        +Dict keyword_patterns
        +Dict entity_patterns
        +detect(tables) DomainContext
        +score_domain(domain, data) Float
        +classify_columns(columns) List
    }

    class DataAnalystAgent {
        +DomainDetector detector
        +analyze_domain(tables) DomainContext
        +classify_columns(tables) List[ColumnProfile]
        +assess_data_quality() Dict
    }

    DataAnalystAgent --> DomainDetector
    DataAnalystAgent --> DomainContext
    DataAnalystAgent --> ColumnProfile
    DomainDetector --> DomainContext
```
