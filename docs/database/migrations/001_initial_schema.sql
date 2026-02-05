-- ============================================================================
-- ONTOLOTY v17.1 - Initial Database Schema
-- ============================================================================
-- Supabase/Postgres DDL
-- Run this migration to set up the complete database structure
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 1. CREATE SCHEMAS
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS discovery;
CREATE SCHEMA IF NOT EXISTS refinement;
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS evidence;
CREATE SCHEMA IF NOT EXISTS versioning;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- ============================================================================
-- 2. CORE TABLES (public schema)
-- ============================================================================

-- Pipeline Runs: 파이프라인 실행 기록
CREATE TABLE IF NOT EXISTS public.pipeline_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_name VARCHAR(255) NOT NULL,
    domain_context VARCHAR(100) DEFAULT 'general',
    status VARCHAR(50) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),

    -- Configuration
    config JSONB DEFAULT '{}',

    -- Results Summary
    summary JSONB DEFAULT '{}',

    -- Metrics
    total_tables INTEGER DEFAULT 0,
    total_entities INTEGER DEFAULT 0,
    total_concepts INTEGER DEFAULT 0,
    total_decisions INTEGER DEFAULT 0,

    -- Timestamps
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE public.pipeline_runs IS '파이프라인 실행 기록 - 모든 분석의 루트 테이블';

CREATE INDEX idx_pipeline_runs_status ON public.pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_scenario ON public.pipeline_runs(scenario_name);
CREATE INDEX idx_pipeline_runs_created ON public.pipeline_runs(created_at DESC);

-- ============================================================================
-- 3. DISCOVERY SCHEMA TABLES (Phase 1)
-- ============================================================================

-- Source Tables: 분석 대상 테이블 정보
CREATE TABLE IF NOT EXISTS discovery.source_tables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Table Info
    table_name VARCHAR(255) NOT NULL,
    source_type VARCHAR(100) DEFAULT 'csv' CHECK (source_type IN ('csv', 'database', 'api', 'parquet', 'json')),
    source_path TEXT,

    -- Schema Info
    columns JSONB NOT NULL DEFAULT '[]',
    row_count INTEGER DEFAULT 0,

    -- Key Info
    primary_keys TEXT[] DEFAULT '{}',
    foreign_keys JSONB DEFAULT '[]',

    -- TDA Analysis Results
    tda_signature JSONB,
    betti_numbers INTEGER[],

    -- Sample Data
    sample_data JSONB DEFAULT '[]',

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT source_tables_pipeline_table_uniq UNIQUE(pipeline_run_id, table_name)
);

COMMENT ON TABLE discovery.source_tables IS '분석 대상 원본 테이블 메타데이터';

CREATE INDEX idx_source_tables_pipeline ON discovery.source_tables(pipeline_run_id);
CREATE INDEX idx_source_tables_name ON discovery.source_tables(table_name);

-- Homeomorphism Pairs: 위상동형 쌍
CREATE TABLE IF NOT EXISTS discovery.homeomorphism_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Table Pair
    table_a_id UUID REFERENCES discovery.source_tables(id) ON DELETE SET NULL,
    table_b_id UUID REFERENCES discovery.source_tables(id) ON DELETE SET NULL,
    table_a_name VARCHAR(255) NOT NULL,
    table_b_name VARCHAR(255) NOT NULL,

    -- Analysis Results
    is_homeomorphic BOOLEAN DEFAULT FALSE,
    homeomorphism_type VARCHAR(50) CHECK (homeomorphism_type IN ('structural', 'value_based', 'semantic', 'full', 'partial')),
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Detailed Scores
    structural_score DECIMAL(5,4) DEFAULT 0.0,
    value_score DECIMAL(5,4) DEFAULT 0.0,
    semantic_score DECIMAL(5,4) DEFAULT 0.0,
    similarity_score DECIMAL(5,4) DEFAULT 0.0,

    -- Join Info
    join_keys JSONB DEFAULT '[]',
    column_mappings JSONB DEFAULT '[]',

    -- Entity Type
    entity_type VARCHAR(100) DEFAULT 'Unknown',

    -- Agent Analysis
    agent_analysis JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE discovery.homeomorphism_pairs IS '테이블 간 위상동형(유사구조) 분석 결과';

CREATE INDEX idx_homeomorphisms_pipeline ON discovery.homeomorphism_pairs(pipeline_run_id);
CREATE INDEX idx_homeomorphisms_tables ON discovery.homeomorphism_pairs(table_a_name, table_b_name);
CREATE INDEX idx_homeomorphisms_confidence ON discovery.homeomorphism_pairs(confidence DESC);
CREATE INDEX idx_homeomorphisms_entity_type ON discovery.homeomorphism_pairs(entity_type);

-- Unified Entities: 통합 엔티티
CREATE TABLE IF NOT EXISTS discovery.unified_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Entity Info
    entity_id VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    canonical_name VARCHAR(255) NOT NULL,

    -- Source Info
    source_tables TEXT[] NOT NULL DEFAULT '{}',
    key_mappings JSONB NOT NULL DEFAULT '{}',

    -- Attributes
    unified_attributes JSONB DEFAULT '[]',

    -- Verification
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    is_verified_by_values BOOLEAN DEFAULT FALSE,
    sample_shared_keys JSONB DEFAULT '[]',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT unified_entities_pipeline_entity_uniq UNIQUE(pipeline_run_id, entity_id)
);

COMMENT ON TABLE discovery.unified_entities IS '여러 테이블에서 통합된 엔티티 정의';

CREATE INDEX idx_unified_entities_pipeline ON discovery.unified_entities(pipeline_run_id);
CREATE INDEX idx_unified_entities_type ON discovery.unified_entities(entity_type);
CREATE INDEX idx_unified_entities_sources_gin ON discovery.unified_entities USING GIN(source_tables);

-- FK Candidates: Foreign Key 후보
CREATE TABLE IF NOT EXISTS discovery.fk_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- FK Info
    from_table VARCHAR(255) NOT NULL,
    from_column VARCHAR(255) NOT NULL,
    to_table VARCHAR(255) NOT NULL,
    to_column VARCHAR(255) NOT NULL,

    -- Scores
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    fk_score DECIMAL(5,4) DEFAULT 0.0,

    -- Detection Method
    detection_method VARCHAR(100) CHECK (detection_method IN ('naming', 'value_overlap', 'semantic', 'pattern', 'hybrid')),

    -- Evidence
    evidence JSONB DEFAULT '{}',

    -- Indirect FK Info
    is_indirect BOOLEAN DEFAULT FALSE,
    bridge_table VARCHAR(255),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE discovery.fk_candidates IS 'Foreign Key 후보 - 자동 탐지된 테이블 간 관계';

CREATE INDEX idx_fk_candidates_pipeline ON discovery.fk_candidates(pipeline_run_id);
CREATE INDEX idx_fk_candidates_tables ON discovery.fk_candidates(from_table, to_table);
CREATE INDEX idx_fk_candidates_confidence ON discovery.fk_candidates(confidence DESC);

-- Column Semantics: 컬럼 의미 분석
CREATE TABLE IF NOT EXISTS discovery.column_semantics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    source_table_id UUID REFERENCES discovery.source_tables(id) ON DELETE CASCADE,

    -- Column Info
    table_name VARCHAR(255) NOT NULL,
    column_name VARCHAR(255) NOT NULL,

    -- Semantic Analysis
    semantic_type VARCHAR(100) CHECK (semantic_type IN ('identifier', 'measure', 'dimension', 'timestamp', 'text', 'categorical', 'boolean')),
    business_role VARCHAR(100) CHECK (business_role IN ('primary_key', 'foreign_key', 'metric', 'attribute', 'status', 'flag', 'date')),
    data_category VARCHAR(100) CHECK (data_category IN ('pii', 'financial', 'operational', 'reference', 'temporal', 'geographic')),

    -- Detailed Analysis
    importance VARCHAR(50) DEFAULT 'medium' CHECK (importance IN ('critical', 'high', 'medium', 'low')),
    nullable_by_design BOOLEAN DEFAULT TRUE,

    -- LLM Analysis Results
    llm_analysis JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT column_semantics_uniq UNIQUE(pipeline_run_id, table_name, column_name)
);

COMMENT ON TABLE discovery.column_semantics IS 'LLM 기반 컬럼 의미 분석 결과';

CREATE INDEX idx_column_semantics_pipeline ON discovery.column_semantics(pipeline_run_id);
CREATE INDEX idx_column_semantics_table ON discovery.column_semantics(table_name);
CREATE INDEX idx_column_semantics_type ON discovery.column_semantics(semantic_type);

-- Data Patterns: 데이터 패턴
CREATE TABLE IF NOT EXISTS discovery.data_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Pattern Info
    pattern_type VARCHAR(100) NOT NULL,
    pattern_name VARCHAR(255),
    description TEXT,

    -- Involved Columns
    table_name VARCHAR(255),
    columns TEXT[] DEFAULT '{}',

    -- Pattern Details
    pattern_details JSONB DEFAULT '{}',
    confidence DECIMAL(5,4) DEFAULT 0.0,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE discovery.data_patterns IS '데이터 패턴 (파생 컬럼, 계산 필드 등)';

CREATE INDEX idx_data_patterns_pipeline ON discovery.data_patterns(pipeline_run_id);
CREATE INDEX idx_data_patterns_type ON discovery.data_patterns(pattern_type);

-- ============================================================================
-- 4. REFINEMENT SCHEMA TABLES (Phase 2)
-- ============================================================================

-- Ontology Concepts: 온톨로지 개념
CREATE TABLE IF NOT EXISTS refinement.ontology_concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Concept Info
    concept_id VARCHAR(100) NOT NULL,
    concept_type VARCHAR(50) NOT NULL CHECK (concept_type IN ('object_type', 'property', 'link_type', 'constraint', 'enum', 'virtual')),
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Definition
    definition JSONB NOT NULL DEFAULT '{}',

    -- Source Info
    source_tables TEXT[] DEFAULT '{}',
    source_evidence JSONB DEFAULT '{}',

    -- Status
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'provisional', 'deprecated')),
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Agent Assessments
    agent_assessments JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT ontology_concepts_pipeline_concept_uniq UNIQUE(pipeline_run_id, concept_id)
);

COMMENT ON TABLE refinement.ontology_concepts IS '온톨로지 개념 정의 - 객체 타입, 속성, 링크 등';

CREATE INDEX idx_concepts_pipeline ON refinement.ontology_concepts(pipeline_run_id);
CREATE INDEX idx_concepts_type ON refinement.ontology_concepts(concept_type);
CREATE INDEX idx_concepts_status ON refinement.ontology_concepts(status);
CREATE INDEX idx_concepts_name ON refinement.ontology_concepts(name);
CREATE INDEX idx_concepts_definition_gin ON refinement.ontology_concepts USING GIN(definition);

-- Concept Relationships: 개념 간 관계
CREATE TABLE IF NOT EXISTS refinement.concept_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Relationship Info
    source_concept_id UUID REFERENCES refinement.ontology_concepts(id) ON DELETE CASCADE,
    target_concept_id UUID REFERENCES refinement.ontology_concepts(id) ON DELETE CASCADE,

    -- Relationship Type
    relationship_type VARCHAR(100) NOT NULL CHECK (relationship_type IN ('is_a', 'has_a', 'part_of', 'depends_on', 'references', 'extends', 'implements')),

    -- Properties
    cardinality VARCHAR(20) CHECK (cardinality IN ('1:1', '1:N', 'N:1', 'N:M')),
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE refinement.concept_relationships IS '온톨로지 개념 간 관계 정의';

CREATE INDEX idx_concept_rels_pipeline ON refinement.concept_relationships(pipeline_run_id);
CREATE INDEX idx_concept_rels_source ON refinement.concept_relationships(source_concept_id);
CREATE INDEX idx_concept_rels_target ON refinement.concept_relationships(target_concept_id);
CREATE INDEX idx_concept_rels_type ON refinement.concept_relationships(relationship_type);

-- Knowledge Graph Triples: 지식 그래프 트리플
CREATE TABLE IF NOT EXISTS refinement.knowledge_graph_triples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Triple (SPO)
    subject VARCHAR(500) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    object VARCHAR(500) NOT NULL,

    -- Type
    triple_type VARCHAR(50) DEFAULT 'base' CHECK (triple_type IN ('base', 'semantic', 'inferred', 'governance')),

    -- Source
    source_agent VARCHAR(100),
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE refinement.knowledge_graph_triples IS '지식 그래프 트리플 (Subject-Predicate-Object)';

CREATE INDEX idx_triples_pipeline ON refinement.knowledge_graph_triples(pipeline_run_id);
CREATE INDEX idx_triples_subject ON refinement.knowledge_graph_triples(subject);
CREATE INDEX idx_triples_predicate ON refinement.knowledge_graph_triples(predicate);
CREATE INDEX idx_triples_object ON refinement.knowledge_graph_triples(object);
CREATE INDEX idx_triples_type ON refinement.knowledge_graph_triples(triple_type);

-- Causal Relationships: 인과 관계
CREATE TABLE IF NOT EXISTS refinement.causal_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Causal Relationship
    cause_entity VARCHAR(255) NOT NULL,
    effect_entity VARCHAR(255) NOT NULL,

    -- Strength & Confidence
    strength VARCHAR(50) CHECK (strength IN ('strong', 'moderate', 'weak')),
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Analysis Results
    analysis JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE refinement.causal_relationships IS '엔티티 간 인과 관계 분석 결과';

CREATE INDEX idx_causal_rels_pipeline ON refinement.causal_relationships(pipeline_run_id);
CREATE INDEX idx_causal_rels_cause ON refinement.causal_relationships(cause_entity);
CREATE INDEX idx_causal_rels_effect ON refinement.causal_relationships(effect_entity);

-- Conflicts Resolved: 해결된 충돌
CREATE TABLE IF NOT EXISTS refinement.conflicts_resolved (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Conflict Info
    conflict_id VARCHAR(100) NOT NULL,
    conflict_type VARCHAR(100) NOT NULL,
    description TEXT,

    -- Involved Items
    involved_concepts TEXT[] DEFAULT '{}',
    involved_tables TEXT[] DEFAULT '{}',

    -- Resolution
    resolution_strategy VARCHAR(100),
    resolution_details JSONB DEFAULT '{}',
    resolved_by VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE refinement.conflicts_resolved IS '온톨로지 충돌 해결 기록';

CREATE INDEX idx_conflicts_pipeline ON refinement.conflicts_resolved(pipeline_run_id);
CREATE INDEX idx_conflicts_type ON refinement.conflicts_resolved(conflict_type);

-- ============================================================================
-- 5. GOVERNANCE SCHEMA TABLES (Phase 3)
-- ============================================================================

-- Governance Decisions: 거버넌스 결정
CREATE TABLE IF NOT EXISTS governance.governance_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    concept_id UUID REFERENCES refinement.ontology_concepts(id) ON DELETE SET NULL,

    -- Decision Info
    decision_id VARCHAR(100) NOT NULL,
    decision_type VARCHAR(50) NOT NULL CHECK (decision_type IN ('approve', 'reject', 'escalate', 'schedule_review', 'defer')),

    -- Results
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    reasoning TEXT,

    -- Agent Opinions
    agent_opinions JSONB DEFAULT '{}',

    -- Recommended Actions
    recommended_actions JSONB DEFAULT '[]',

    -- Metadata
    made_by VARCHAR(100) DEFAULT 'UnifiedPipeline',
    made_at TIMESTAMPTZ DEFAULT NOW(),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT governance_decisions_pipeline_decision_uniq UNIQUE(pipeline_run_id, decision_id)
);

COMMENT ON TABLE governance.governance_decisions IS '온톨로지 개념에 대한 거버넌스 결정';

CREATE INDEX idx_decisions_pipeline ON governance.governance_decisions(pipeline_run_id);
CREATE INDEX idx_decisions_concept ON governance.governance_decisions(concept_id);
CREATE INDEX idx_decisions_type ON governance.governance_decisions(decision_type);
CREATE INDEX idx_decisions_confidence ON governance.governance_decisions(confidence DESC);

-- Action Backlog: 액션 백로그
CREATE TABLE IF NOT EXISTS governance.action_backlog (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    decision_id UUID REFERENCES governance.governance_decisions(id) ON DELETE SET NULL,

    -- Action Info
    action_id VARCHAR(100) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    description TEXT,

    -- Priority
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    priority_score DECIMAL(5,4) DEFAULT 0.0,

    -- Target
    target_concept VARCHAR(255),
    target_tables TEXT[] DEFAULT '{}',

    -- Status
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled', 'blocked')),

    -- Impact
    impact_score DECIMAL(5,4) DEFAULT 0.0,
    roi_estimate DECIMAL(10,2),

    -- Timestamps
    due_date DATE,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT action_backlog_pipeline_action_uniq UNIQUE(pipeline_run_id, action_id)
);

COMMENT ON TABLE governance.action_backlog IS '실행 대기 중인 액션 항목';

CREATE INDEX idx_actions_pipeline ON governance.action_backlog(pipeline_run_id);
CREATE INDEX idx_actions_decision ON governance.action_backlog(decision_id);
CREATE INDEX idx_actions_priority ON governance.action_backlog(priority, priority_score DESC);
CREATE INDEX idx_actions_status ON governance.action_backlog(status);

-- Policy Rules: 정책 규칙
CREATE TABLE IF NOT EXISTS governance.policy_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Rule Info
    rule_id VARCHAR(100) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(100) CHECK (rule_type IN ('validation', 'quality', 'security', 'compliance', 'naming', 'structure')),

    -- Definition
    condition JSONB NOT NULL,
    action JSONB NOT NULL,

    -- Scope
    applies_to TEXT[] DEFAULT '{}',

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    severity VARCHAR(20) DEFAULT 'warning' CHECK (severity IN ('error', 'warning', 'info')),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE governance.policy_rules IS '데이터 거버넌스 정책 규칙';

CREATE INDEX idx_policy_rules_pipeline ON governance.policy_rules(pipeline_run_id);
CREATE INDEX idx_policy_rules_type ON governance.policy_rules(rule_type);
CREATE INDEX idx_policy_rules_active ON governance.policy_rules(is_active);

-- Business Insights: 비즈니스 인사이트
CREATE TABLE IF NOT EXISTS governance.business_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Insight Info
    insight_id VARCHAR(100) NOT NULL,
    insight_type VARCHAR(100),
    title VARCHAR(500),
    description TEXT,

    -- Value
    business_value VARCHAR(50) CHECK (business_value IN ('high', 'medium', 'low')),
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- Details
    details JSONB DEFAULT '{}',
    recommendations TEXT[],

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE governance.business_insights IS '데이터 분석을 통한 비즈니스 인사이트';

CREATE INDEX idx_business_insights_pipeline ON governance.business_insights(pipeline_run_id);
CREATE INDEX idx_business_insights_type ON governance.business_insights(insight_type);

-- ============================================================================
-- 6. EVIDENCE SCHEMA TABLES (Cross-cutting)
-- ============================================================================

-- Evidence Blocks: 근거 블록 (블록체인 스타일)
CREATE TABLE IF NOT EXISTS evidence.evidence_blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Block Info
    block_id VARCHAR(100) NOT NULL,
    block_index INTEGER NOT NULL,

    -- Chain Link (Blockchain-style)
    previous_hash VARCHAR(64),
    current_hash VARCHAR(64) NOT NULL,

    -- Content
    phase VARCHAR(50) NOT NULL CHECK (phase IN ('discovery', 'refinement', 'governance')),
    agent VARCHAR(100) NOT NULL,
    evidence_type VARCHAR(100) NOT NULL,

    -- Findings
    finding TEXT NOT NULL,
    reasoning TEXT,
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Data Evidence
    data_evidence JSONB DEFAULT '{}',

    -- Timestamp
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT evidence_blocks_pipeline_block_uniq UNIQUE(pipeline_run_id, block_id)
);

COMMENT ON TABLE evidence.evidence_blocks IS '블록체인 스타일 근거 체인 - 모든 분석의 근거 추적';

CREATE INDEX idx_evidence_blocks_pipeline ON evidence.evidence_blocks(pipeline_run_id);
CREATE INDEX idx_evidence_blocks_phase ON evidence.evidence_blocks(phase);
CREATE INDEX idx_evidence_blocks_agent ON evidence.evidence_blocks(agent);
CREATE INDEX idx_evidence_blocks_hash ON evidence.evidence_blocks(current_hash);
CREATE INDEX idx_evidence_blocks_index ON evidence.evidence_blocks(block_index);

-- Consensus Records: 합의 기록
CREATE TABLE IF NOT EXISTS evidence.consensus_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Discussion Info
    topic_id VARCHAR(100) NOT NULL,
    topic_name VARCHAR(255),
    topic_type VARCHAR(100),

    -- Results
    result_type VARCHAR(50) CHECK (result_type IN ('accepted', 'rejected', 'provisional', 'escalated')),
    combined_confidence DECIMAL(5,4) DEFAULT 0.0,
    agreement_level DECIMAL(5,4) DEFAULT 0.0,

    -- Details
    total_rounds INTEGER DEFAULT 0,
    final_decision TEXT,
    reasoning TEXT,
    dissenting_views JSONB DEFAULT '[]',

    -- Vote Records
    votes JSONB DEFAULT '[]',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE evidence.consensus_records IS '멀티에이전트 합의 토론 기록';

CREATE INDEX idx_consensus_pipeline ON evidence.consensus_records(pipeline_run_id);
CREATE INDEX idx_consensus_topic ON evidence.consensus_records(topic_id);
CREATE INDEX idx_consensus_result ON evidence.consensus_records(result_type);

-- Calibration History: 신뢰도 보정 이력
CREATE TABLE IF NOT EXISTS evidence.calibration_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Phase Info
    phase VARCHAR(50) NOT NULL,

    -- Calibration Summary
    summary JSONB DEFAULT '{}',

    -- Metrics
    mean_confidence DECIMAL(5,4),
    calibration_error DECIMAL(5,4),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE evidence.calibration_history IS 'LLM 신뢰도 보정 이력';

CREATE INDEX idx_calibration_pipeline ON evidence.calibration_history(pipeline_run_id);
CREATE INDEX idx_calibration_phase ON evidence.calibration_history(phase);

-- ============================================================================
-- 7. VERSIONING SCHEMA TABLES
-- ============================================================================

-- Dataset Versions: 데이터셋 버전
CREATE TABLE IF NOT EXISTS versioning.dataset_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Version Info
    version_id VARCHAR(100) NOT NULL,
    version_number INTEGER NOT NULL,

    -- Commit Info
    message TEXT,
    author VARCHAR(255),

    -- Snapshot
    snapshot_data JSONB,

    -- Parent Version (Git-style)
    parent_version_id UUID REFERENCES versioning.dataset_versions(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT dataset_versions_pipeline_version_uniq UNIQUE(pipeline_run_id, version_id)
);

COMMENT ON TABLE versioning.dataset_versions IS 'Git 스타일 데이터셋 버전 관리';

CREATE INDEX idx_versions_pipeline ON versioning.dataset_versions(pipeline_run_id);
CREATE INDEX idx_versions_parent ON versioning.dataset_versions(parent_version_id);
CREATE INDEX idx_versions_number ON versioning.dataset_versions(version_number);

-- Branches: 브랜치
CREATE TABLE IF NOT EXISTS versioning.branches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Branch Info
    branch_name VARCHAR(255) NOT NULL,
    head_version_id UUID REFERENCES versioning.dataset_versions(id) ON DELETE SET NULL,

    -- Status
    is_default BOOLEAN DEFAULT FALSE,
    is_protected BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT branches_pipeline_name_uniq UNIQUE(pipeline_run_id, branch_name)
);

COMMENT ON TABLE versioning.branches IS '데이터셋 브랜치';

CREATE INDEX idx_branches_pipeline ON versioning.branches(pipeline_run_id);
CREATE INDEX idx_branches_default ON versioning.branches(is_default) WHERE is_default = TRUE;

-- ============================================================================
-- 8. ANALYTICS SCHEMA TABLES (Post-pipeline)
-- ============================================================================

-- Simulation Scenarios: 시뮬레이션 시나리오
CREATE TABLE IF NOT EXISTS analytics.simulation_scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Scenario Info
    scenario_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Changes
    changes JSONB NOT NULL DEFAULT '[]',

    -- Settings
    num_simulations INTEGER DEFAULT 100,
    confidence_level DECIMAL(5,4) DEFAULT 0.95,

    -- Status
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT scenarios_pipeline_scenario_uniq UNIQUE(pipeline_run_id, scenario_id)
);

COMMENT ON TABLE analytics.simulation_scenarios IS 'What-If 분석 시나리오 정의';

CREATE INDEX idx_scenarios_pipeline ON analytics.simulation_scenarios(pipeline_run_id);
CREATE INDEX idx_scenarios_status ON analytics.simulation_scenarios(status);

-- Scenario Results: 시나리오 결과
CREATE TABLE IF NOT EXISTS analytics.scenario_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID NOT NULL REFERENCES analytics.simulation_scenarios(id) ON DELETE CASCADE,

    -- Results
    success BOOLEAN DEFAULT FALSE,
    num_iterations INTEGER DEFAULT 0,

    -- Impact Analysis
    impacts JSONB DEFAULT '[]',
    mean_outcomes JSONB DEFAULT '{}',
    std_outcomes JSONB DEFAULT '{}',

    -- Risk Analysis
    risk_score DECIMAL(5,4) DEFAULT 0.0 CHECK (risk_score >= 0 AND risk_score <= 1),
    opportunity_score DECIMAL(5,4) DEFAULT 0.0 CHECK (opportunity_score >= 0 AND opportunity_score <= 1),

    -- Recommendations
    recommendations JSONB DEFAULT '[]',
    impact_summary TEXT,

    -- Timing
    execution_time_ms DECIMAL(10,2),
    simulated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE analytics.scenario_results IS 'What-If 시뮬레이션 결과';

CREATE INDEX idx_scenario_results_scenario ON analytics.scenario_results(scenario_id);
CREATE INDEX idx_scenario_results_risk ON analytics.scenario_results(risk_score DESC);

-- Unified Insights: 통합 인사이트
CREATE TABLE IF NOT EXISTS analytics.unified_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Insight Info
    insight_id VARCHAR(100) NOT NULL,
    insight_type VARCHAR(100) CHECK (insight_type IN ('correlation', 'anomaly', 'pattern', 'prediction', 'recommendation')),

    -- Content
    title VARCHAR(500),
    description TEXT,

    -- Confidence
    confidence DECIMAL(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Algorithm Results
    algorithm_source VARCHAR(100),
    algorithm_result JSONB DEFAULT '{}',

    -- Simulation Results
    simulation_result JSONB DEFAULT '{}',

    -- LLM Analysis
    llm_analysis JSONB DEFAULT '{}',

    -- Related Components
    related_concepts TEXT[] DEFAULT '{}',
    related_actions TEXT[] DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE analytics.unified_insights IS 'Algorithm + Simulation + LLM 통합 인사이트';

CREATE INDEX idx_insights_pipeline ON analytics.unified_insights(pipeline_run_id);
CREATE INDEX idx_insights_type ON analytics.unified_insights(insight_type);
CREATE INDEX idx_insights_confidence ON analytics.unified_insights(confidence DESC);

-- NL2SQL History: 자연어 SQL 변환 이력
CREATE TABLE IF NOT EXISTS analytics.nl2sql_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    -- Query Info
    natural_query TEXT NOT NULL,
    generated_sql TEXT,

    -- Results
    success BOOLEAN DEFAULT FALSE,
    confidence DECIMAL(5,4) DEFAULT 0.0,

    -- Metadata
    tables_used TEXT[] DEFAULT '{}',
    execution_result JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE analytics.nl2sql_history IS '자연어 SQL 변환 이력';

CREATE INDEX idx_nl2sql_pipeline ON analytics.nl2sql_history(pipeline_run_id);
CREATE INDEX idx_nl2sql_success ON analytics.nl2sql_history(success);

-- ============================================================================
-- 9. AUDIT SCHEMA TABLES
-- ============================================================================

-- Audit Logs: 감사 로그
CREATE TABLE IF NOT EXISTS audit.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Target
    schema_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    record_id UUID NOT NULL,

    -- Change Info
    action VARCHAR(20) NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_data JSONB,
    new_data JSONB,
    changed_fields TEXT[],

    -- User Info
    user_id UUID,
    user_agent TEXT,
    ip_address INET,

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE audit.audit_logs IS '모든 테이블 변경에 대한 감사 로그';

CREATE INDEX idx_audit_logs_table ON audit.audit_logs(schema_name, table_name);
CREATE INDEX idx_audit_logs_record ON audit.audit_logs(record_id);
CREATE INDEX idx_audit_logs_action ON audit.audit_logs(action);
CREATE INDEX idx_audit_logs_created ON audit.audit_logs(created_at DESC);

-- ============================================================================
-- 10. TRIGGERS & FUNCTIONS
-- ============================================================================

-- Updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_pipeline_runs_updated_at
    BEFORE UPDATE ON public.pipeline_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ontology_concepts_updated_at
    BEFORE UPDATE ON refinement.ontology_concepts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_branches_updated_at
    BEFORE UPDATE ON versioning.branches
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit log trigger function
CREATE OR REPLACE FUNCTION audit_log_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit.audit_logs (schema_name, table_name, record_id, action, new_data)
        VALUES (TG_TABLE_SCHEMA, TG_TABLE_NAME, NEW.id, 'INSERT', to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit.audit_logs (schema_name, table_name, record_id, action, old_data, new_data)
        VALUES (TG_TABLE_SCHEMA, TG_TABLE_NAME, NEW.id, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit.audit_logs (schema_name, table_name, record_id, action, old_data)
        VALUES (TG_TABLE_SCHEMA, TG_TABLE_NAME, OLD.id, 'DELETE', to_jsonb(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to important tables (optional - uncomment to enable)
-- CREATE TRIGGER audit_governance_decisions
--     AFTER INSERT OR UPDATE OR DELETE ON governance.governance_decisions
--     FOR EACH ROW EXECUTE FUNCTION audit_log_trigger();

-- ============================================================================
-- 11. VIEWS
-- ============================================================================

-- Pipeline Summary View
CREATE OR REPLACE VIEW public.v_pipeline_summary AS
SELECT
    pr.id,
    pr.scenario_name,
    pr.domain_context,
    pr.status,
    pr.started_at,
    pr.completed_at,
    EXTRACT(EPOCH FROM (pr.completed_at - pr.started_at)) as duration_seconds,
    (SELECT COUNT(*) FROM discovery.source_tables st WHERE st.pipeline_run_id = pr.id) as table_count,
    (SELECT COUNT(*) FROM discovery.unified_entities ue WHERE ue.pipeline_run_id = pr.id) as entity_count,
    (SELECT COUNT(*) FROM refinement.ontology_concepts oc WHERE oc.pipeline_run_id = pr.id) as concept_count,
    (SELECT COUNT(*) FROM governance.governance_decisions gd WHERE gd.pipeline_run_id = pr.id) as decision_count,
    (SELECT COUNT(*) FROM governance.action_backlog ab WHERE ab.pipeline_run_id = pr.id) as action_count
FROM public.pipeline_runs pr;

COMMENT ON VIEW public.v_pipeline_summary IS '파이프라인 실행 요약 뷰';

-- Concept Status View
CREATE OR REPLACE VIEW refinement.v_concept_status AS
SELECT
    oc.pipeline_run_id,
    oc.status,
    COUNT(*) as count,
    AVG(oc.confidence) as avg_confidence
FROM refinement.ontology_concepts oc
GROUP BY oc.pipeline_run_id, oc.status;

COMMENT ON VIEW refinement.v_concept_status IS '온톨로지 개념 상태별 통계';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Record migration
INSERT INTO audit.audit_logs (schema_name, table_name, record_id, action, new_data)
VALUES ('system', 'migrations', gen_random_uuid(), 'INSERT',
    '{"migration": "001_initial_schema", "completed_at": "' || NOW()::TEXT || '"}'::JSONB);

COMMENT ON SCHEMA public IS 'Ontoloty v17.1 - Core schema';
COMMENT ON SCHEMA discovery IS 'Ontoloty v17.1 - Phase 1: Discovery';
COMMENT ON SCHEMA refinement IS 'Ontoloty v17.1 - Phase 2: Refinement';
COMMENT ON SCHEMA governance IS 'Ontoloty v17.1 - Phase 3: Governance';
COMMENT ON SCHEMA evidence IS 'Ontoloty v17.1 - Cross-cutting: Evidence Chain';
COMMENT ON SCHEMA versioning IS 'Ontoloty v17.1 - Cross-cutting: Version Control';
COMMENT ON SCHEMA analytics IS 'Ontoloty v17.1 - Post-pipeline: Analytics';
COMMENT ON SCHEMA audit IS 'Ontoloty v17.1 - Audit Logs';
