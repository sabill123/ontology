# Ontoloty v1.0 Database Design

## 1. Overview

Ontoloty 시스템의 Postgres/Supabase 데이터베이스 설계 문서입니다.

### 1.1 설계 원칙

1. **정규화**: 3NF(Third Normal Form) 준수
2. **확장성**: JSONB 활용으로 유연한 스키마 확장
3. **성능**: 적절한 인덱싱 및 파티셔닝
4. **추적성**: 모든 변경에 대한 audit trail
5. **참조 무결성**: Foreign Key로 데이터 일관성 보장
6. **Supabase Best Practices**: 공식 가이드라인 준수

### 1.2 Supabase Best Practices 적용 현황

| Category | Practice | Status |
|----------|----------|--------|
| **Query Performance** | FK 컬럼에 인덱스 | ✅ |
| **Query Performance** | JSONB에 GIN 인덱스 | ✅ |
| **Query Performance** | Partial Indexes | ✅ |
| **Query Performance** | Composite Indexes | ✅ |
| **Schema Design** | bigint identity PK | ✅ |
| **Schema Design** | TEXT (not VARCHAR) | ✅ |
| **Schema Design** | TIMESTAMPTZ | ✅ |
| **Schema Design** | lowercase snake_case | ✅ |
| **Security** | Row Level Security | ✅ |
| **Security** | Multi-tenant isolation | ✅ |
| **Advanced** | Full-text search (tsvector) | ✅ |
| **Advanced** | BRIN index for audit | ✅ |

---

## 2. Naming Conventions (명명 규칙)

### 2.1 테이블 명명 규칙

```
패턴: {domain}_{entity}_{suffix}
```

| 규칙 | 설명 | 예시 |
|------|------|------|
| snake_case | 모든 테이블명은 소문자 snake_case | `ontology_concepts` |
| 복수형 | 테이블명은 복수형 | `governance_decisions` |
| 접두사 | 도메인별 접두사 사용 | `disco_`, `refine_`, `govern_` |
| 조인 테이블 | 두 엔티티명 연결 | `concept_relationship_mappings` |

### 2.2 컬럼 명명 규칙

| 규칙 | 설명 | 예시 |
|------|------|------|
| snake_case | 모든 컬럼명은 소문자 snake_case | `created_at` |
| PK | `id` (bigint identity 사용) | `id` |
| FK | `{referenced_table}_id` | `pipeline_run_id` |
| Boolean | `is_`, `has_`, `can_` 접두사 | `is_approved`, `has_evidence` |
| Timestamp | `_at` 접미사 (TIMESTAMPTZ) | `created_at`, `updated_at` |
| Count | `_count` 접미사 | `row_count` |
| JSON | `_data`, `_config`, `_metadata` 접미사 | `sample_data`, `agent_config` |
| Search | `search_vector` (tsvector) | `search_vector` |

### 2.3 데이터 타입 규칙 (Supabase Best Practice)

| 데이터 | 권장 타입 | 비권장 | 이유 |
|--------|----------|--------|------|
| ID | `bigint GENERATED ALWAYS AS IDENTITY` | UUID (v4) | 인덱스 fragmentation 방지 |
| 문자열 | `text` | `varchar(n)` | 동일 성능, 불필요한 제약 제거 |
| 시간 | `timestamptz` | `timestamp` | 타임존 보존 |
| 금액 | `numeric(10,2)` | `float` | 정확한 소수점 연산 |
| 열거형 | `text + CHECK` | PostgreSQL ENUM | 유연한 변경 |

### 2.4 인덱스 명명 규칙

```
패턴: idx_{table}_{column(s)}_{type}
```

| 타입 | 접미사 | 예시 |
|------|--------|------|
| B-tree (기본) | 없음 | `idx_concepts_name` |
| Unique | `_uniq` | `idx_concepts_name_uniq` |
| GIN (JSON) | `_gin` | `idx_concepts_definition_gin` |
| GiST (지리) | `_gist` | `idx_locations_coords_gist` |
| BRIN (시계열) | `_brin` | `idx_audit_logs_created_brin` |
| 복합 | 컬럼 연결 | `idx_decisions_concept_status` |
| Partial | 조건 포함 | `idx_actions_active` (WHERE status='pending') |

### 2.5 Constraint 명명 규칙

```
패턴: {table}_{column(s)}_{constraint_type}
```

| Constraint | 접미사 | 예시 |
|------------|--------|------|
| Primary Key | `_pkey` | `ontology_concepts_pkey` |
| Foreign Key | `_fkey` | `decisions_concept_id_fkey` |
| Unique | `_uniq` | `concepts_name_uniq` |
| Check | `_check` | `decisions_confidence_check` |

---

## 3. Schema Design (스키마 설계)

### 3.1 스키마 구조

```
ontoloty (database)
├── public (default schema)
├── discovery      -- Phase 1: 데이터 발견
├── refinement     -- Phase 2: 온톨로지 정제
├── governance     -- Phase 3: 거버넌스
├── evidence       -- Cross-cutting: 근거 체인
├── versioning     -- Cross-cutting: 버전 관리
├── analytics      -- Post-pipeline: 분석
└── audit          -- 감사 로그
```

---

## 4. Table Definitions (테이블 정의)

### 4.1 Core Tables (핵심 테이블)

#### pipeline_runs (파이프라인 실행)
```sql
CREATE TABLE public.pipeline_runs (
    -- PK: bigint identity (Supabase Best Practice Rule 4.4)
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Text fields: TEXT not VARCHAR (Rule 4.1)
    scenario_name text NOT NULL,
    domain_context text DEFAULT 'general',
    status text DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),

    -- 설정
    config jsonb DEFAULT '{}',

    -- 결과 요약
    summary jsonb DEFAULT '{}',

    -- 메트릭
    total_tables integer DEFAULT 0,
    total_entities integer DEFAULT 0,
    total_concepts integer DEFAULT 0,
    total_decisions integer DEFAULT 0,

    -- 타임스탬프: TIMESTAMPTZ (Rule 4.1)
    started_at timestamptz DEFAULT now(),
    completed_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),

    -- Multi-tenant support (for RLS)
    tenant_id text DEFAULT 'default',

    -- Full-text search (Rule 8.2)
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(scenario_name, '') || ' ' || coalesce(domain_context, ''))
    ) STORED
);

-- Indexes
CREATE INDEX idx_pipeline_runs_status ON public.pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_tenant ON public.pipeline_runs(tenant_id);
-- Partial index for active runs (Rule 1.5)
CREATE INDEX idx_pipeline_runs_active ON public.pipeline_runs(created_at DESC)
    WHERE status = 'running';
-- GIN index for JSONB (Rule 8.1)
CREATE INDEX idx_pipeline_runs_config_gin ON public.pipeline_runs USING gin(config);
-- Full-text search index
CREATE INDEX idx_pipeline_runs_search ON public.pipeline_runs USING gin(search_vector);
```

#### source_tables (원본 테이블 정보)
```sql
CREATE TABLE discovery.source_tables (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 테이블 정보
    table_name text NOT NULL,
    source_type text DEFAULT 'csv',  -- csv, database, api
    source_path TEXT,

    -- 스키마 정보
    columns JSONB NOT NULL DEFAULT '[]',
    row_count INTEGER DEFAULT 0,

    -- 키 정보
    primary_keys TEXT[] DEFAULT '{}',
    foreign_keys JSONB DEFAULT '[]',

    -- TDA 분석 결과
    tda_signature JSONB,
    betti_numbers INTEGER[],

    -- 샘플 데이터
    sample_data JSONB DEFAULT '[]',

    -- 메타데이터
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT source_tables_pipeline_table_uniq UNIQUE(pipeline_run_id, table_name)
);

CREATE INDEX idx_source_tables_pipeline ON discovery.source_tables(pipeline_run_id);
CREATE INDEX idx_source_tables_name ON discovery.source_tables(table_name);
```

### 4.2 Discovery Phase Tables (Phase 1)

#### homeomorphism_pairs (위상동형 쌍)
```sql
CREATE TABLE discovery.homeomorphism_pairs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 테이블 쌍
    table_a_id bigint REFERENCES discovery.source_tables(id),
    table_b_id bigint REFERENCES discovery.source_tables(id),
    table_a_name text NOT NULL,
    table_b_name text NOT NULL,

    -- 분석 결과
    is_homeomorphic BOOLEAN DEFAULT FALSE,
    homeomorphism_type text,  -- structural, value_based, semantic, full
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- 상세 점수
    structural_score DECIMAL(5,4) DEFAULT 0.0,
    value_score DECIMAL(5,4) DEFAULT 0.0,
    semantic_score DECIMAL(5,4) DEFAULT 0.0,

    -- 조인 정보
    join_keys JSONB DEFAULT '[]',
    column_mappings JSONB DEFAULT '[]',

    -- 엔티티 타입
    entity_type text DEFAULT 'Unknown',

    -- 에이전트 분석
    agent_analysis JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_homeomorphisms_pipeline ON discovery.homeomorphism_pairs(pipeline_run_id);
CREATE INDEX idx_homeomorphisms_tables ON discovery.homeomorphism_pairs(table_a_name, table_b_name);
CREATE INDEX idx_homeomorphisms_confidence ON discovery.homeomorphism_pairs(confidence DESC);
```

#### unified_entities (통합 엔티티)
```sql
CREATE TABLE discovery.unified_entities (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 엔티티 정보
    entity_id text NOT NULL,
    entity_type text NOT NULL,
    canonical_name text NOT NULL,

    -- 소스 정보
    source_tables TEXT[] NOT NULL DEFAULT '{}',
    key_mappings JSONB NOT NULL DEFAULT '{}',  -- {table: key_column}

    -- 속성
    unified_attributes JSONB DEFAULT '[]',

    -- 검증
    confidence DECIMAL(5,4) DEFAULT 0.0,
    is_verified_by_values BOOLEAN DEFAULT FALSE,
    sample_shared_keys JSONB DEFAULT '[]',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT unified_entities_pipeline_entity_uniq UNIQUE(pipeline_run_id, entity_id)
);

CREATE INDEX idx_unified_entities_pipeline ON discovery.unified_entities(pipeline_run_id);
CREATE INDEX idx_unified_entities_type ON discovery.unified_entities(entity_type);
CREATE INDEX idx_unified_entities_sources_gin ON discovery.unified_entities USING GIN(source_tables);
```

#### enhanced_fk_candidates (FK 후보)
```sql
CREATE TABLE discovery.enhanced_fk_candidates (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- FK 정보
    from_table text NOT NULL,
    from_column text NOT NULL,
    to_table text NOT NULL,
    to_column text NOT NULL,

    -- 점수 및 신뢰도
    confidence DECIMAL(5,4) DEFAULT 0.0,
    fk_score DECIMAL(5,4) DEFAULT 0.0,

    -- 검출 방법
    detection_method text,  -- naming, value_overlap, semantic

    -- 증거
    evidence JSONB DEFAULT '{}',

    -- 간접 FK 정보 (Bridge Table 경유)
    is_indirect BOOLEAN DEFAULT FALSE,
    bridge_table text,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_enhanced_fk_candidates_pipeline ON discovery.enhanced_fk_candidates(pipeline_run_id);
CREATE INDEX idx_enhanced_fk_candidates_tables ON discovery.enhanced_fk_candidates(from_table, to_table);
```

#### column_semantics (컬럼 의미 분석)
```sql
CREATE TABLE discovery.column_semantics (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    source_table_id bigint REFERENCES discovery.source_tables(id),

    -- 컬럼 정보
    table_name text NOT NULL,
    column_name text NOT NULL,

    -- 의미 분석
    semantic_type text,  -- identifier, measure, dimension, timestamp
    business_role text,  -- primary_key, foreign_key, metric, attribute
    data_category text,  -- pii, financial, operational, reference

    -- 상세 분석
    importance text DEFAULT 'medium',  -- high, medium, low
    nullable_by_design BOOLEAN DEFAULT TRUE,

    -- LLM 분석 결과
    llm_analysis JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT column_semantics_uniq UNIQUE(pipeline_run_id, table_name, column_name)
);

CREATE INDEX idx_column_semantics_pipeline ON discovery.column_semantics(pipeline_run_id);
CREATE INDEX idx_column_semantics_table ON discovery.column_semantics(table_name);
```

### 4.3 Refinement Phase Tables (Phase 2)

#### ontology_concepts (온톨로지 개념)
```sql
CREATE TABLE refinement.ontology_concepts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 개념 정보
    concept_id text NOT NULL,
    concept_type text NOT NULL,  -- object_type, property, link_type, constraint
    name text NOT NULL,
    description TEXT,

    -- 정의
    definition JSONB NOT NULL DEFAULT '{}',

    -- 소스 정보
    source_tables TEXT[] DEFAULT '{}',
    source_evidence JSONB DEFAULT '{}',

    -- 상태
    status text DEFAULT 'pending',  -- pending, approved, rejected, provisional
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- 에이전트 평가
    agent_assessments JSONB DEFAULT '{}',

    -- 타임스탬프
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT ontology_concepts_pipeline_concept_uniq UNIQUE(pipeline_run_id, concept_id)
);

CREATE INDEX idx_concepts_pipeline ON refinement.ontology_concepts(pipeline_run_id);
CREATE INDEX idx_concepts_type ON refinement.ontology_concepts(concept_type);
CREATE INDEX idx_concepts_status ON refinement.ontology_concepts(status);
CREATE INDEX idx_concepts_name ON refinement.ontology_concepts(name);
CREATE INDEX idx_concepts_definition_gin ON refinement.ontology_concepts USING GIN(definition);
```

#### concept_relationships (개념 관계)
```sql
CREATE TABLE refinement.concept_relationships (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 관계 정보
    source_concept_id bigint REFERENCES refinement.ontology_concepts(id),
    target_concept_id bigint REFERENCES refinement.ontology_concepts(id),

    -- 관계 유형
    relationship_type text NOT NULL,  -- is_a, has_a, part_of, depends_on

    -- 속성
    cardinality text,  -- 1:1, 1:N, N:M
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- 메타데이터
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_concept_rels_pipeline ON refinement.concept_relationships(pipeline_run_id);
CREATE INDEX idx_concept_rels_source ON refinement.concept_relationships(source_concept_id);
CREATE INDEX idx_concept_rels_target ON refinement.concept_relationships(target_concept_id);
```

#### knowledge_graph_triples (지식 그래프 트리플)
```sql
CREATE TABLE refinement.knowledge_graph_triples (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 트리플 (SPO)
    subject text NOT NULL,
    predicate text NOT NULL,
    object text NOT NULL,

    -- 타입
    triple_type text DEFAULT 'base',  -- base, semantic, inferred, governance

    -- 출처
    source_agent text,
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- 메타데이터
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_triples_pipeline ON refinement.knowledge_graph_triples(pipeline_run_id);
CREATE INDEX idx_triples_subject ON refinement.knowledge_graph_triples(subject);
CREATE INDEX idx_triples_predicate ON refinement.knowledge_graph_triples(predicate);
CREATE INDEX idx_triples_type ON refinement.knowledge_graph_triples(triple_type);
```

#### causal_relationships (인과 관계)
```sql
CREATE TABLE refinement.causal_relationships (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 인과 관계
    cause_entity text NOT NULL,
    effect_entity text NOT NULL,

    -- 강도 및 신뢰도
    strength text,  -- strong, moderate, weak
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- 분석 결과
    analysis JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_causal_rels_pipeline ON refinement.causal_relationships(pipeline_run_id);
```

### 4.4 Governance Phase Tables (Phase 3)

#### governance_decisions (거버넌스 결정)
```sql
CREATE TABLE governance.governance_decisions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    concept_id bigint REFERENCES refinement.ontology_concepts(id),

    -- 결정 정보
    decision_id text NOT NULL,
    decision_type text NOT NULL,  -- approve, reject, escalate, schedule_review

    -- 결과
    confidence DECIMAL(5,4) DEFAULT 0.0,
    reasoning TEXT,

    -- 에이전트 의견
    agent_opinions JSONB DEFAULT '{}',

    -- 권장 액션
    recommended_actions JSONB DEFAULT '[]',

    -- 메타데이터
    made_by text DEFAULT 'UnifiedPipeline',
    made_at TIMESTAMPTZ DEFAULT NOW(),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT governance_decisions_pipeline_decision_uniq UNIQUE(pipeline_run_id, decision_id)
);

CREATE INDEX idx_decisions_pipeline ON governance.governance_decisions(pipeline_run_id);
CREATE INDEX idx_decisions_concept ON governance.governance_decisions(concept_id);
CREATE INDEX idx_decisions_type ON governance.governance_decisions(decision_type);
```

#### action_backlog (액션 백로그)
```sql
CREATE TABLE governance.action_backlog (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    decision_id bigint REFERENCES governance.governance_decisions(id),

    -- 액션 정보
    action_id text NOT NULL,
    action_type text NOT NULL,
    description TEXT,

    -- 우선순위
    priority text DEFAULT 'medium',  -- critical, high, medium, low
    priority_score DECIMAL(5,4) DEFAULT 0.0,

    -- 대상
    target_concept text,
    target_tables TEXT[] DEFAULT '{}',

    -- 상태
    status text DEFAULT 'pending',  -- pending, in_progress, completed, cancelled

    -- 영향도
    impact_score DECIMAL(5,4) DEFAULT 0.0,
    roi_estimate DECIMAL(10,2),

    -- 타임스탬프
    due_date DATE,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT action_backlog_pipeline_action_uniq UNIQUE(pipeline_run_id, action_id)
);

CREATE INDEX idx_actions_pipeline ON governance.action_backlog(pipeline_run_id);
CREATE INDEX idx_actions_priority ON governance.action_backlog(priority, priority_score DESC);
CREATE INDEX idx_actions_status ON governance.action_backlog(status);
```

#### policy_rules (정책 규칙)
```sql
CREATE TABLE governance.policy_rules (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 규칙 정보
    rule_id text NOT NULL,
    rule_name text NOT NULL,
    rule_type text,  -- validation, quality, security, compliance

    -- 정의
    condition JSONB NOT NULL,
    action JSONB NOT NULL,

    -- 적용 범위
    applies_to TEXT[] DEFAULT '{}',

    -- 상태
    is_active BOOLEAN DEFAULT TRUE,
    severity text DEFAULT 'warning',  -- error, warning, info

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_policy_rules_pipeline ON governance.policy_rules(pipeline_run_id);
CREATE INDEX idx_policy_rules_type ON governance.policy_rules(rule_type);
```

### 4.5 Evidence Chain Tables (Cross-cutting)

#### evidence_blocks (근거 블록)
```sql
CREATE TABLE evidence.evidence_blocks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 블록 정보
    block_id text NOT NULL,
    block_index INTEGER NOT NULL,

    -- 체인 연결 (블록체인 스타일)
    previous_hash text,
    current_hash text NOT NULL,

    -- 내용
    phase text NOT NULL,  -- discovery, refinement, governance
    agent text NOT NULL,
    evidence_type text NOT NULL,

    -- 발견 사항
    finding TEXT NOT NULL,
    reasoning TEXT,
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- 데이터 증거
    data_evidence JSONB DEFAULT '{}',

    -- 타임스탬프
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT evidence_blocks_pipeline_block_uniq UNIQUE(pipeline_run_id, block_id)
);

CREATE INDEX idx_evidence_blocks_pipeline ON evidence.evidence_blocks(pipeline_run_id);
CREATE INDEX idx_evidence_blocks_phase ON evidence.evidence_blocks(phase);
CREATE INDEX idx_evidence_blocks_agent ON evidence.evidence_blocks(agent);
CREATE INDEX idx_evidence_blocks_hash ON evidence.evidence_blocks(current_hash);
```

#### consensus_records (합의 기록)
```sql
CREATE TABLE evidence.consensus_records (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 토론 정보
    topic_id text NOT NULL,
    topic_name text,
    topic_type text,

    -- 결과
    result_type text,  -- accepted, rejected, provisional, escalated
    combined_confidence DECIMAL(5,4) DEFAULT 0.0,
    agreement_level DECIMAL(5,4) DEFAULT 0.0,

    -- 상세
    total_rounds INTEGER DEFAULT 0,
    final_decision TEXT,
    reasoning TEXT,
    dissenting_views JSONB DEFAULT '[]',

    -- 투표 기록
    votes JSONB DEFAULT '[]',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_consensus_pipeline ON evidence.consensus_records(pipeline_run_id);
CREATE INDEX idx_consensus_topic ON evidence.consensus_records(topic_id);
CREATE INDEX idx_consensus_result ON evidence.consensus_records(result_type);
```

### 4.6 Versioning Tables (Dataset Versioning)

#### dataset_versions (데이터셋 버전)
```sql
CREATE TABLE versioning.dataset_versions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 버전 정보
    version_id text NOT NULL,
    version_number INTEGER NOT NULL,

    -- 커밋 정보
    message TEXT,
    author text,

    -- 스냅샷
    snapshot_data JSONB,

    -- 부모 버전 (Git 스타일)
    parent_version_id bigint REFERENCES versioning.dataset_versions(id),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT dataset_versions_pipeline_version_uniq UNIQUE(pipeline_run_id, version_id)
);

CREATE INDEX idx_versions_pipeline ON versioning.dataset_versions(pipeline_run_id);
CREATE INDEX idx_versions_parent ON versioning.dataset_versions(parent_version_id);
```

#### branches (브랜치)
```sql
CREATE TABLE versioning.branches (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 브랜치 정보
    branch_name text NOT NULL,
    head_version_id bigint REFERENCES versioning.dataset_versions(id),

    -- 상태
    is_default BOOLEAN DEFAULT FALSE,
    is_protected BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT branches_pipeline_name_uniq UNIQUE(pipeline_run_id, branch_name)
);

CREATE INDEX idx_branches_pipeline ON versioning.branches(pipeline_run_id);
```

### 4.7 Analytics Tables (Post-pipeline)

#### simulation_scenarios (시뮬레이션 시나리오)
```sql
CREATE TABLE analytics.simulation_scenarios (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 시나리오 정보
    scenario_id text NOT NULL,
    name text NOT NULL,
    description TEXT,

    -- 변경 사항
    changes JSONB NOT NULL DEFAULT '[]',

    -- 설정
    num_simulations INTEGER DEFAULT 100,
    confidence_level DECIMAL(5,4) DEFAULT 0.95,

    -- 상태
    status text DEFAULT 'pending',  -- pending, running, completed, failed

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT scenarios_pipeline_scenario_uniq UNIQUE(pipeline_run_id, scenario_id)
);

CREATE INDEX idx_scenarios_pipeline ON analytics.simulation_scenarios(pipeline_run_id);
```

#### scenario_results (시나리오 결과)
```sql
CREATE TABLE analytics.scenario_results (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scenario_id bigint REFERENCES analytics.simulation_scenarios(id) ON DELETE CASCADE,

    -- 결과
    success BOOLEAN DEFAULT FALSE,
    num_iterations INTEGER DEFAULT 0,

    -- 영향 분석
    impacts JSONB DEFAULT '[]',
    mean_outcomes JSONB DEFAULT '{}',
    std_outcomes JSONB DEFAULT '{}',

    -- 리스크 분석
    risk_score DECIMAL(5,4) DEFAULT 0.0,
    opportunity_score DECIMAL(5,4) DEFAULT 0.0,

    -- 권장 사항
    recommendations JSONB DEFAULT '[]',
    impact_summary TEXT,

    -- 타이밍
    execution_time_ms DECIMAL(10,2),
    simulated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scenario_results_scenario ON analytics.scenario_results(scenario_id);
```

#### unified_insights (통합 인사이트)
```sql
CREATE TABLE analytics.unified_insights (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- 인사이트 정보
    insight_id text NOT NULL,
    insight_type text,  -- correlation, anomaly, pattern, prediction

    -- 내용
    title text,
    description TEXT,

    -- 신뢰도
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- 알고리즘 결과
    algorithm_source text,
    algorithm_result JSONB DEFAULT '{}',

    -- 시뮬레이션 결과
    simulation_result JSONB DEFAULT '{}',

    -- LLM 분석
    llm_analysis JSONB DEFAULT '{}',

    -- 연결된 컴포넌트
    related_concepts TEXT[] DEFAULT '{}',
    related_actions TEXT[] DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_insights_pipeline ON analytics.unified_insights(pipeline_run_id);
CREATE INDEX idx_insights_type ON analytics.unified_insights(insight_type);
```

### 4.8 Audit Tables (감사 로그)

#### audit_logs (감사 로그)
```sql
CREATE TABLE audit.audit_logs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- 대상
    table_name text NOT NULL,
    record_id UUID NOT NULL,

    -- 변경 정보
    action text NOT NULL,  -- INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    changed_fields TEXT[],

    -- 사용자 정보
    user_id UUID,
    user_agent TEXT,
    ip_address INET,

    -- 타임스탬프
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_table ON audit.audit_logs(table_name);
CREATE INDEX idx_audit_logs_record ON audit.audit_logs(record_id);
CREATE INDEX idx_audit_logs_action ON audit.audit_logs(action);
CREATE INDEX idx_audit_logs_created ON audit.audit_logs(created_at DESC);
```

---

## 5. ERD (Entity Relationship Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ONTOLOTY DATABASE ERD                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  pipeline_runs  │ ─────────────────────────────────────────────────────────────┐
│─────────────────│                                                              │
│ id (PK)         │                                                              │
│ scenario_name   │                                                              │
│ domain_context  │                                                              │
│ status          │                                                              │
│ config          │                                                              │
│ summary         │                                                              │
│ created_at      │                                                              │
└────────┬────────┘                                                              │
         │                                                                        │
         │ 1:N                                                                    │
         │                                                                        │
┌────────┴──────────────────────────────────────────────────────────────────────┐│
│                              DISCOVERY SCHEMA                                  ││
│ ┌─────────────────┐    ┌─────────────────────┐    ┌───────────────────┐      ││
│ │  source_tables  │    │ homeomorphism_pairs │    │  unified_entities │      ││
│ │─────────────────│    │─────────────────────│    │───────────────────│      ││
│ │ id (PK)         │◄───│ table_a_id (FK)     │    │ id (PK)           │      ││
│ │ pipeline_run_id │    │ table_b_id (FK)     │    │ pipeline_run_id   │──────┼┤
│ │ table_name      │    │ pipeline_run_id ────┼────│ entity_id         │      ││
│ │ columns         │    │ confidence          │    │ entity_type       │      ││
│ │ row_count       │    │ column_mappings     │    │ source_tables[]   │      ││
│ │ tda_signature   │    │ entity_type         │    │ key_mappings      │      ││
│ └─────────────────┘    └─────────────────────┘    └───────────────────┘      ││
│         │                                                                      ││
│         │ 1:N                                                                  ││
│ ┌───────┴─────────┐    ┌─────────────────────┐                                ││
│ │ column_semantics│    │enhanced_fk_candidates│                                ││
│ │─────────────────│    │─────────────────────│                                ││
│ │ id (PK)         │    │ id (PK)             │                                ││
│ │ source_table_id │    │ pipeline_run_id ────┼────────────────────────────────┼┤
│ │ column_name     │    │ from_table          │                                ││
│ │ semantic_type   │    │ to_table            │                                ││
│ │ business_role   │    │ confidence          │                                ││
│ └─────────────────┘    └─────────────────────┘                                ││
└───────────────────────────────────────────────────────────────────────────────┘│
                                                                                  │
┌─────────────────────────────────────────────────────────────────────────────────┤
│                              REFINEMENT SCHEMA                                  │
│ ┌───────────────────┐    ┌───────────────────────┐                             │
│ │ ontology_concepts │◄───│ concept_relationships │                             │
│ │───────────────────│    │───────────────────────│                             │
│ │ id (PK)           │    │ source_concept_id (FK)│                             │
│ │ pipeline_run_id ──┼────│ target_concept_id (FK)│                             │
│ │ concept_id        │    │ relationship_type     │                             │
│ │ concept_type      │    │ cardinality           │                             │
│ │ name              │    │ confidence            │                             │
│ │ status            │    └───────────────────────┘                             │
│ │ confidence        │                                                          │
│ └─────────┬─────────┘    ┌───────────────────────┐                             │
│           │              │ knowledge_graph_triples│                             │
│           │              │───────────────────────│                             │
│           │              │ subject               │                             │
│           │              │ predicate             │                             │
│           │              │ object                │                             │
│           │              │ triple_type           │                             │
│           │              └───────────────────────┘                             │
└───────────┼─────────────────────────────────────────────────────────────────────┤
            │                                                                     │
┌───────────┼─────────────────────────────────────────────────────────────────────┤
│           │              GOVERNANCE SCHEMA                                      │
│ ┌─────────▼─────────┐    ┌───────────────────┐    ┌───────────────────┐       │
│ │governance_decisions│───►│   action_backlog  │    │   policy_rules    │       │
│ │───────────────────│    │───────────────────│    │───────────────────│       │
│ │ id (PK)           │    │ id (PK)           │    │ id (PK)           │       │
│ │ pipeline_run_id ──┼────│ decision_id (FK)  │    │ pipeline_run_id ──┼───────┤
│ │ concept_id (FK)   │    │ action_type       │    │ rule_name         │       │
│ │ decision_type     │    │ priority          │    │ condition         │       │
│ │ confidence        │    │ status            │    │ action            │       │
│ │ reasoning         │    │ impact_score      │    │ severity          │       │
│ └───────────────────┘    └───────────────────┘    └───────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────────┤
                                                                                  │
┌─────────────────────────────────────────────────────────────────────────────────┤
│                              EVIDENCE SCHEMA (Cross-cutting)                    │
│ ┌───────────────────┐    ┌───────────────────┐                                 │
│ │  evidence_blocks  │    │ consensus_records │                                 │
│ │───────────────────│    │───────────────────│                                 │
│ │ id (PK)           │    │ id (PK)           │                                 │
│ │ pipeline_run_id ──┼────│ pipeline_run_id ──┼─────────────────────────────────┤
│ │ block_id          │    │ topic_id          │                                 │
│ │ previous_hash     │    │ result_type       │                                 │
│ │ current_hash      │    │ agreement_level   │                                 │
│ │ phase             │    │ votes             │                                 │
│ │ finding           │    └───────────────────┘                                 │
│ └───────────────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────────────┤
                                                                                  │
┌─────────────────────────────────────────────────────────────────────────────────┤
│                              ANALYTICS SCHEMA (Post-pipeline)                   │
│ ┌─────────────────────┐    ┌───────────────────┐    ┌───────────────────┐     │
│ │simulation_scenarios │───►│ scenario_results  │    │ unified_insights  │     │
│ │─────────────────────│    │───────────────────│    │───────────────────│     │
│ │ id (PK)             │    │ scenario_id (FK)  │    │ id (PK)           │     │
│ │ pipeline_run_id ────┼────│ risk_score        │    │ pipeline_run_id ──┼─────┤
│ │ name                │    │ opportunity_score │    │ insight_type      │     │
│ │ changes             │    │ recommendations   │    │ confidence        │     │
│ │ status              │    └───────────────────┘    │ algorithm_result  │     │
│ └─────────────────────┘                             │ llm_analysis      │     │
│                                                     └───────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Storage Rules (데이터 저장 규칙)

### 6.1 필수 필드 규칙 (v1.0 업데이트)

| 규칙 | 설명 | Best Practice |
|------|------|---------------|
| bigint identity PK | 모든 테이블은 `bigint GENERATED ALWAYS AS IDENTITY` 사용 | Rule 4.4 |
| UUID PK 마이그레이션 | 기존 `UUID` PK를 사용하는 테이블은 `bigint GENERATED ALWAYS AS IDENTITY`로 마이그레이션 권장. UUID는 인덱스 fragmentation을 유발하며 Supabase 공식 가이드에서 bigint identity를 권장합니다. | Rule 4.4 |
| pipeline_run_id | 파이프라인 결과 테이블은 반드시 포함 | - |
| tenant_id | 루트 테이블에 tenant_id 포함 (RLS용) | Rule 3.2 |
| created_at | 모든 테이블에 `timestamptz` 생성 시간 | Rule 4.1 |
| updated_at | 수정 가능한 테이블에 수정 시간 + 트리거 | - |
| search_vector | 검색 필요 테이블에 tsvector | Rule 8.2 |

### 6.2 JSONB 사용 규칙

```sql
-- 권장: 구조화된 데이터는 JSONB
config JSONB DEFAULT '{}'

-- 권장: 배열 데이터는 JSONB
impacts JSONB DEFAULT '[]'

-- 비권장: 단순 목록은 TEXT[]
source_tables TEXT[] DEFAULT '{}'
```

### 6.3 Confidence 필드 규칙

```sql
-- 모든 confidence 필드는 DECIMAL(5,4) 사용 (0.0000 ~ 1.0000)
confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1)
```

### 6.4 Status 필드 규칙

```sql
-- Enum 대신 VARCHAR + CHECK 사용 (유연성)
status text DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'completed', 'failed'))
```

### 6.5 Cascade 규칙

```sql
-- 파이프라인 삭제 시 관련 데이터 모두 삭제
REFERENCES public.pipeline_runs(id) ON DELETE CASCADE

-- 참조 무결성이 중요한 경우
REFERENCES refinement.ontology_concepts(id) ON DELETE RESTRICT
```

---

## 7. Indexes Strategy (인덱스 전략)

### 7.1 기본 인덱스

```sql
-- 모든 FK에 인덱스
CREATE INDEX idx_{table}_{fk_column} ON {table}({fk_column});

-- 자주 검색되는 컬럼에 인덱스
CREATE INDEX idx_{table}_{column} ON {table}({column});
```

### 7.2 JSONB 인덱스

```sql
-- JSONB 전체 검색용 GIN 인덱스
CREATE INDEX idx_{table}_{column}_gin ON {table} USING GIN({column});

-- 특정 키 검색용 인덱스
CREATE INDEX idx_{table}_{column}_key ON {table} (({column}->>'key'));
```

### 7.3 복합 인덱스

```sql
-- 자주 함께 검색되는 컬럼
CREATE INDEX idx_decisions_pipeline_type ON governance_decisions(pipeline_run_id, decision_type);
```

---

## 8. Supabase-Specific Features

### 8.1 Row Level Security (RLS) - Multi-Tenant Isolation

모든 테이블에 RLS 적용으로 테넌트 간 데이터 격리:

```sql
-- 1. 모든 테이블에 RLS 활성화
ALTER TABLE public.pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery.source_tables ENABLE ROW LEVEL SECURITY;
ALTER TABLE refinement.ontology_concepts ENABLE ROW LEVEL SECURITY;
-- ... (모든 테이블)

-- 2. 헬퍼 함수 (SECURITY DEFINER로 RLS 우회)
CREATE OR REPLACE FUNCTION get_pipeline_tenant(p_pipeline_run_id bigint)
RETURNS text
LANGUAGE sql
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT tenant_id FROM public.pipeline_runs WHERE id = p_pipeline_run_id
$$;

-- 3. 루트 테이블 정책 (Rule 3.3 - SELECT wrapper로 최적화)
CREATE POLICY pipeline_runs_tenant_policy ON public.pipeline_runs
    FOR ALL
    USING (tenant_id = (SELECT current_setting('app.tenant_id', true)));

-- 4. 자식 테이블 정책 (pipeline_run_id로 테넌트 상속)
CREATE POLICY source_tables_tenant_policy ON discovery.source_tables
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));
```

**사용법**:
```sql
-- 세션에서 테넌트 설정
SET app.tenant_id = 'tenant_123';

-- 이후 모든 쿼리는 자동으로 해당 테넌트 데이터만 반환
SELECT * FROM pipeline_runs;  -- tenant_123 데이터만 반환
```

### 8.2 Full-Text Search (Rule 8.2)

주요 테이블에 tsvector 컬럼 추가로 효율적인 검색:

```sql
-- Generated tsvector column
search_vector tsvector GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, ''))
) STORED;

-- GIN index for fast search
CREATE INDEX idx_concepts_search ON refinement.ontology_concepts USING gin(search_vector);

-- 검색 쿼리
SELECT * FROM refinement.ontology_concepts
WHERE search_vector @@ to_tsquery('english', 'patient & healthcare');

-- 랭킹 포함 검색
SELECT *, ts_rank(search_vector, query) as rank
FROM refinement.ontology_concepts, to_tsquery('english', 'patient') query
WHERE search_vector @@ query
ORDER BY rank DESC;
```

### 8.3 Cross-content Search Function

```sql
-- 모든 콘텐츠 타입에서 통합 검색
CREATE FUNCTION search_all_content(
    p_tenant_id text,
    p_query text,
    p_limit integer DEFAULT 50
) RETURNS TABLE (
    source_type text,
    source_id bigint,
    title text,
    snippet text,
    rank real
);
```

### 8.4 Realtime Subscriptions

```sql
-- 실시간 업데이트가 필요한 테이블
ALTER PUBLICATION supabase_realtime ADD TABLE public.pipeline_runs;
ALTER PUBLICATION supabase_realtime ADD TABLE governance.action_backlog;
```

### 8.5 Storage for Large Data

```sql
-- 대용량 데이터는 Supabase Storage 사용
-- sample_data가 큰 경우 storage에 저장하고 URL만 저장
sample_data_url TEXT  -- storage bucket URL
```

---

## 9. Migration Strategy

### 9.1 마이그레이션 순서

1. 스키마 생성 (`CREATE SCHEMA`)
2. 핵심 테이블 생성 (`pipeline_runs`)
3. Discovery 테이블 생성
4. Refinement 테이블 생성
5. Governance 테이블 생성
6. Evidence 테이블 생성
7. Analytics 테이블 생성
8. Audit 테이블 생성
9. 인덱스 생성
10. RLS 정책 설정

### 9.2 롤백 전략

```sql
-- 각 마이그레이션에 롤백 스크립트 포함
-- DOWN migration
DROP TABLE IF EXISTS {table} CASCADE;
```

---

## 10. Performance Considerations

### 10.1 파티셔닝

```sql
-- 대용량 테이블은 날짜별 파티셔닝
CREATE TABLE evidence.evidence_blocks (
    ...
) PARTITION BY RANGE (created_at);

-- 월별 파티션 생성
CREATE TABLE evidence.evidence_blocks_2024_01
PARTITION OF evidence.evidence_blocks
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 10.2 Vacuum 설정

```sql
-- 자주 업데이트되는 테이블
ALTER TABLE governance.action_backlog SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);
```

---

## 11. Migration Files

| 파일 | 설명 |
|------|------|
| `001_initial_schema.sql` | 초기 스키마 생성 (UUID PK, VARCHAR) |
| `002_supabase_best_practices.sql` | Supabase Best Practices 적용 (bigint identity, TEXT, RLS, tsvector) |

**권장 실행 순서**:
```bash
# Fresh install (권장)
psql -f migrations/002_supabase_best_practices.sql

# 또는 기존 데이터 보존 필요시
psql -f migrations/001_initial_schema.sql
# 이후 데이터 마이그레이션 후 002 적용
```

---

*Generated for Ontoloty v1.0*
*Last Updated: 2026-01*
*Supabase Postgres Best Practices v1.1.0 적용*
