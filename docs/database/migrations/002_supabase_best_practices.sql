-- ============================================================================
-- ONTOLOTY v17.2 - Supabase Best Practices Migration
-- ============================================================================
-- This migration applies Supabase Postgres best practices to the existing schema:
-- 1. PK Strategy: bigint identity (sequential, no fragmentation)
-- 2. Data Types: TEXT instead of VARCHAR, TIMESTAMPTZ
-- 3. FK Indexes: All foreign key columns indexed
-- 4. JSONB GIN Indexes: For containment queries
-- 5. Row Level Security (RLS): Multi-tenant isolation
-- 6. Full-text Search: tsvector columns
-- 7. Partial Indexes: For filtered queries
-- 8. Connection pooling compatibility: Named statements handled
-- ============================================================================

-- ============================================================================
-- PART 1: EXTENSIONS
-- ============================================================================

-- Enable pg_stat_statements for query analysis (Rule 7.1)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Enable pg_trgm for fuzzy text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- PART 2: RECREATE TABLES WITH bigint IDENTITY PKs
-- ============================================================================
-- Note: This is a breaking change. For production, use a migration strategy
-- that preserves existing data. This script is for fresh installations.
-- ============================================================================

-- Drop existing tables in reverse dependency order
DROP VIEW IF EXISTS refinement.v_concept_status CASCADE;
DROP VIEW IF EXISTS public.v_pipeline_summary CASCADE;

-- Drop tables in correct order (leaf tables first)
DROP TABLE IF EXISTS audit.audit_logs CASCADE;
DROP TABLE IF EXISTS analytics.nl2sql_history CASCADE;
DROP TABLE IF EXISTS analytics.unified_insights CASCADE;
DROP TABLE IF EXISTS analytics.scenario_results CASCADE;
DROP TABLE IF EXISTS analytics.simulation_scenarios CASCADE;
DROP TABLE IF EXISTS versioning.branches CASCADE;
DROP TABLE IF EXISTS versioning.dataset_versions CASCADE;
DROP TABLE IF EXISTS evidence.calibration_history CASCADE;
DROP TABLE IF EXISTS evidence.consensus_records CASCADE;
DROP TABLE IF EXISTS evidence.evidence_blocks CASCADE;
DROP TABLE IF EXISTS governance.business_insights CASCADE;
DROP TABLE IF EXISTS governance.policy_rules CASCADE;
DROP TABLE IF EXISTS governance.action_backlog CASCADE;
DROP TABLE IF EXISTS governance.governance_decisions CASCADE;
DROP TABLE IF EXISTS refinement.conflicts_resolved CASCADE;
DROP TABLE IF EXISTS refinement.causal_relationships CASCADE;
DROP TABLE IF EXISTS refinement.knowledge_graph_triples CASCADE;
DROP TABLE IF EXISTS refinement.concept_relationships CASCADE;
DROP TABLE IF EXISTS refinement.ontology_concepts CASCADE;
DROP TABLE IF EXISTS discovery.data_patterns CASCADE;
DROP TABLE IF EXISTS discovery.column_semantics CASCADE;
DROP TABLE IF EXISTS discovery.fk_candidates CASCADE;
DROP TABLE IF EXISTS discovery.unified_entities CASCADE;
DROP TABLE IF EXISTS discovery.homeomorphism_pairs CASCADE;
DROP TABLE IF EXISTS discovery.source_tables CASCADE;
DROP TABLE IF EXISTS public.pipeline_runs CASCADE;

-- ============================================================================
-- CORE TABLE: pipeline_runs (Root table)
-- ============================================================================

CREATE TABLE public.pipeline_runs (
    -- PK: bigint identity (Rule 4.4 - best for single database)
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Text fields: use TEXT not VARCHAR (Rule 4.1)
    scenario_name text NOT NULL,
    domain_context text DEFAULT 'general',
    status text DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),

    -- JSONB for flexible schema
    config jsonb DEFAULT '{}',
    summary jsonb DEFAULT '{}',

    -- Metrics
    total_tables integer DEFAULT 0,
    total_entities integer DEFAULT 0,
    total_concepts integer DEFAULT 0,
    total_decisions integer DEFAULT 0,

    -- Timestamps: always use TIMESTAMPTZ (Rule 4.1)
    started_at timestamptz DEFAULT now(),
    completed_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),

    -- Multi-tenant support (for RLS)
    tenant_id text DEFAULT 'default',

    -- Full-text search vector (Rule 8.2)
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(scenario_name, '') || ' ' || coalesce(domain_context, ''))
    ) STORED
);

COMMENT ON TABLE public.pipeline_runs IS 'Pipeline execution records - root table for all analysis';

-- Indexes
CREATE INDEX idx_pipeline_runs_status ON public.pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_scenario ON public.pipeline_runs(scenario_name);
CREATE INDEX idx_pipeline_runs_created ON public.pipeline_runs(created_at DESC);
CREATE INDEX idx_pipeline_runs_tenant ON public.pipeline_runs(tenant_id);
-- Partial index for active runs (Rule 1.5)
CREATE INDEX idx_pipeline_runs_active ON public.pipeline_runs(created_at DESC)
    WHERE status = 'running';
-- GIN index for JSONB (Rule 8.1)
CREATE INDEX idx_pipeline_runs_config_gin ON public.pipeline_runs USING gin(config);
-- Full-text search index
CREATE INDEX idx_pipeline_runs_search ON public.pipeline_runs USING gin(search_vector);

-- ============================================================================
-- DISCOVERY SCHEMA
-- ============================================================================

CREATE TABLE discovery.source_tables (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    table_name text NOT NULL,
    source_type text DEFAULT 'csv' CHECK (source_type IN ('csv', 'database', 'api', 'parquet', 'json')),
    source_path text,

    columns jsonb NOT NULL DEFAULT '[]',
    row_count integer DEFAULT 0,

    primary_keys text[] DEFAULT '{}',
    foreign_keys jsonb DEFAULT '[]',

    tda_signature jsonb,
    betti_numbers integer[],
    sample_data jsonb DEFAULT '[]',
    metadata jsonb DEFAULT '{}',

    created_at timestamptz DEFAULT now(),

    CONSTRAINT source_tables_pipeline_table_uniq UNIQUE(pipeline_run_id, table_name)
);

COMMENT ON TABLE discovery.source_tables IS 'Source table metadata for analysis';

-- FK index (Rule 4.2)
CREATE INDEX idx_source_tables_pipeline ON discovery.source_tables(pipeline_run_id);
CREATE INDEX idx_source_tables_name ON discovery.source_tables(table_name);
-- GIN indexes for JSONB
CREATE INDEX idx_source_tables_columns_gin ON discovery.source_tables USING gin(columns);
CREATE INDEX idx_source_tables_fk_gin ON discovery.source_tables USING gin(foreign_keys);

-- ----------------------------------------
-- Homeomorphism Pairs
-- ----------------------------------------
CREATE TABLE discovery.homeomorphism_pairs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    table_a_id bigint REFERENCES discovery.source_tables(id) ON DELETE SET NULL,
    table_b_id bigint REFERENCES discovery.source_tables(id) ON DELETE SET NULL,
    table_a_name text NOT NULL,
    table_b_name text NOT NULL,

    is_homeomorphic boolean DEFAULT FALSE,
    homeomorphism_type text CHECK (homeomorphism_type IN ('structural', 'value_based', 'semantic', 'full', 'partial')),
    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    structural_score numeric(5,4) DEFAULT 0.0,
    value_score numeric(5,4) DEFAULT 0.0,
    semantic_score numeric(5,4) DEFAULT 0.0,
    similarity_score numeric(5,4) DEFAULT 0.0,

    join_keys jsonb DEFAULT '[]',
    column_mappings jsonb DEFAULT '[]',
    entity_type text DEFAULT 'Unknown',
    agent_analysis jsonb DEFAULT '{}',

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE discovery.homeomorphism_pairs IS 'Topological homeomorphism (structural similarity) analysis results';

CREATE INDEX idx_homeomorphisms_pipeline ON discovery.homeomorphism_pairs(pipeline_run_id);
-- FK indexes for nullable FKs (Rule 4.2)
CREATE INDEX idx_homeomorphisms_table_a ON discovery.homeomorphism_pairs(table_a_id) WHERE table_a_id IS NOT NULL;
CREATE INDEX idx_homeomorphisms_table_b ON discovery.homeomorphism_pairs(table_b_id) WHERE table_b_id IS NOT NULL;
CREATE INDEX idx_homeomorphisms_tables ON discovery.homeomorphism_pairs(table_a_name, table_b_name);
CREATE INDEX idx_homeomorphisms_confidence ON discovery.homeomorphism_pairs(confidence DESC);
CREATE INDEX idx_homeomorphisms_entity_type ON discovery.homeomorphism_pairs(entity_type);

-- ----------------------------------------
-- Unified Entities
-- ----------------------------------------
CREATE TABLE discovery.unified_entities (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    entity_id text NOT NULL,
    entity_type text NOT NULL,
    canonical_name text NOT NULL,

    source_tables text[] NOT NULL DEFAULT '{}',
    key_mappings jsonb NOT NULL DEFAULT '{}',
    unified_attributes jsonb DEFAULT '[]',

    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    is_verified_by_values boolean DEFAULT FALSE,
    sample_shared_keys jsonb DEFAULT '[]',

    created_at timestamptz DEFAULT now(),

    CONSTRAINT unified_entities_pipeline_entity_uniq UNIQUE(pipeline_run_id, entity_id)
);

COMMENT ON TABLE discovery.unified_entities IS 'Unified entity definitions across tables';

CREATE INDEX idx_unified_entities_pipeline ON discovery.unified_entities(pipeline_run_id);
CREATE INDEX idx_unified_entities_type ON discovery.unified_entities(entity_type);
CREATE INDEX idx_unified_entities_sources_gin ON discovery.unified_entities USING gin(source_tables);
CREATE INDEX idx_unified_entities_attrs_gin ON discovery.unified_entities USING gin(unified_attributes);

-- ----------------------------------------
-- FK Candidates
-- ----------------------------------------
CREATE TABLE discovery.fk_candidates (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    from_table text NOT NULL,
    from_column text NOT NULL,
    to_table text NOT NULL,
    to_column text NOT NULL,

    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    fk_score numeric(5,4) DEFAULT 0.0,

    detection_method text CHECK (detection_method IN ('naming', 'value_overlap', 'semantic', 'pattern', 'hybrid')),
    evidence jsonb DEFAULT '{}',

    is_indirect boolean DEFAULT FALSE,
    bridge_table text,

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE discovery.fk_candidates IS 'Auto-detected foreign key candidates';

CREATE INDEX idx_fk_candidates_pipeline ON discovery.fk_candidates(pipeline_run_id);
CREATE INDEX idx_fk_candidates_tables ON discovery.fk_candidates(from_table, to_table);
CREATE INDEX idx_fk_candidates_confidence ON discovery.fk_candidates(confidence DESC);
-- Partial index for high-confidence candidates
CREATE INDEX idx_fk_candidates_high_conf ON discovery.fk_candidates(pipeline_run_id, from_table, to_table)
    WHERE confidence >= 0.8;

-- ----------------------------------------
-- Column Semantics
-- ----------------------------------------
CREATE TABLE discovery.column_semantics (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    source_table_id bigint REFERENCES discovery.source_tables(id) ON DELETE CASCADE,

    table_name text NOT NULL,
    column_name text NOT NULL,

    semantic_type text CHECK (semantic_type IN ('identifier', 'measure', 'dimension', 'timestamp', 'text', 'categorical', 'boolean')),
    business_role text CHECK (business_role IN ('primary_key', 'foreign_key', 'metric', 'attribute', 'status', 'flag', 'date')),
    data_category text CHECK (data_category IN ('pii', 'financial', 'operational', 'reference', 'temporal', 'geographic')),

    importance text DEFAULT 'medium' CHECK (importance IN ('critical', 'high', 'medium', 'low')),
    nullable_by_design boolean DEFAULT TRUE,

    llm_analysis jsonb DEFAULT '{}',

    created_at timestamptz DEFAULT now(),

    CONSTRAINT column_semantics_uniq UNIQUE(pipeline_run_id, table_name, column_name)
);

COMMENT ON TABLE discovery.column_semantics IS 'LLM-based column semantic analysis';

CREATE INDEX idx_column_semantics_pipeline ON discovery.column_semantics(pipeline_run_id);
-- FK index (Rule 4.2)
CREATE INDEX idx_column_semantics_source_table ON discovery.column_semantics(source_table_id) WHERE source_table_id IS NOT NULL;
CREATE INDEX idx_column_semantics_table ON discovery.column_semantics(table_name);
CREATE INDEX idx_column_semantics_type ON discovery.column_semantics(semantic_type);
-- Partial index for PII columns (security-critical)
CREATE INDEX idx_column_semantics_pii ON discovery.column_semantics(pipeline_run_id, table_name)
    WHERE data_category = 'pii';

-- ----------------------------------------
-- Data Patterns
-- ----------------------------------------
CREATE TABLE discovery.data_patterns (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    pattern_type text NOT NULL,
    pattern_name text,
    description text,

    table_name text,
    columns text[] DEFAULT '{}',

    pattern_details jsonb DEFAULT '{}',
    confidence numeric(5,4) DEFAULT 0.0,

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE discovery.data_patterns IS 'Data patterns (derived columns, calculated fields)';

CREATE INDEX idx_data_patterns_pipeline ON discovery.data_patterns(pipeline_run_id);
CREATE INDEX idx_data_patterns_type ON discovery.data_patterns(pattern_type);
CREATE INDEX idx_data_patterns_details_gin ON discovery.data_patterns USING gin(pattern_details);

-- ============================================================================
-- REFINEMENT SCHEMA
-- ============================================================================

CREATE TABLE refinement.ontology_concepts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    concept_id text NOT NULL,
    concept_type text NOT NULL CHECK (concept_type IN ('object_type', 'property', 'link_type', 'constraint', 'enum', 'virtual')),
    name text NOT NULL,
    description text,

    definition jsonb NOT NULL DEFAULT '{}',

    source_tables text[] DEFAULT '{}',
    source_evidence jsonb DEFAULT '{}',

    status text DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'provisional', 'deprecated')),
    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    agent_assessments jsonb DEFAULT '{}',

    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),

    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, ''))
    ) STORED,

    CONSTRAINT ontology_concepts_pipeline_concept_uniq UNIQUE(pipeline_run_id, concept_id)
);

COMMENT ON TABLE refinement.ontology_concepts IS 'Ontology concept definitions';

CREATE INDEX idx_concepts_pipeline ON refinement.ontology_concepts(pipeline_run_id);
CREATE INDEX idx_concepts_type ON refinement.ontology_concepts(concept_type);
CREATE INDEX idx_concepts_status ON refinement.ontology_concepts(status);
CREATE INDEX idx_concepts_name ON refinement.ontology_concepts(name);
CREATE INDEX idx_concepts_definition_gin ON refinement.ontology_concepts USING gin(definition);
CREATE INDEX idx_concepts_search ON refinement.ontology_concepts USING gin(search_vector);
-- Partial index for approved concepts
CREATE INDEX idx_concepts_approved ON refinement.ontology_concepts(pipeline_run_id, concept_type)
    WHERE status = 'approved';

-- ----------------------------------------
-- Concept Relationships
-- ----------------------------------------
CREATE TABLE refinement.concept_relationships (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    source_concept_id bigint REFERENCES refinement.ontology_concepts(id) ON DELETE CASCADE,
    target_concept_id bigint REFERENCES refinement.ontology_concepts(id) ON DELETE CASCADE,

    relationship_type text NOT NULL CHECK (relationship_type IN ('is_a', 'has_a', 'part_of', 'depends_on', 'references', 'extends', 'implements')),

    cardinality text CHECK (cardinality IN ('1:1', '1:N', 'N:1', 'N:M')),
    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    metadata jsonb DEFAULT '{}',

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE refinement.concept_relationships IS 'Ontology concept relationships';

CREATE INDEX idx_concept_rels_pipeline ON refinement.concept_relationships(pipeline_run_id);
CREATE INDEX idx_concept_rels_source ON refinement.concept_relationships(source_concept_id);
CREATE INDEX idx_concept_rels_target ON refinement.concept_relationships(target_concept_id);
CREATE INDEX idx_concept_rels_type ON refinement.concept_relationships(relationship_type);

-- ----------------------------------------
-- Knowledge Graph Triples
-- ----------------------------------------
CREATE TABLE refinement.knowledge_graph_triples (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    subject text NOT NULL,
    predicate text NOT NULL,
    object text NOT NULL,

    triple_type text DEFAULT 'base' CHECK (triple_type IN ('base', 'semantic', 'inferred', 'governance')),

    source_agent text,
    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    metadata jsonb DEFAULT '{}',

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE refinement.knowledge_graph_triples IS 'Knowledge graph triples (SPO)';

CREATE INDEX idx_triples_pipeline ON refinement.knowledge_graph_triples(pipeline_run_id);
CREATE INDEX idx_triples_subject ON refinement.knowledge_graph_triples(subject);
CREATE INDEX idx_triples_predicate ON refinement.knowledge_graph_triples(predicate);
CREATE INDEX idx_triples_object ON refinement.knowledge_graph_triples(object);
CREATE INDEX idx_triples_type ON refinement.knowledge_graph_triples(triple_type);
-- Composite index for SPO lookups
CREATE INDEX idx_triples_spo ON refinement.knowledge_graph_triples(subject, predicate, object);

-- ----------------------------------------
-- Causal Relationships
-- ----------------------------------------
CREATE TABLE refinement.causal_relationships (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    cause_entity text NOT NULL,
    effect_entity text NOT NULL,

    strength text CHECK (strength IN ('strong', 'moderate', 'weak')),
    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    analysis jsonb DEFAULT '{}',

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE refinement.causal_relationships IS 'Causal relationships between entities';

CREATE INDEX idx_causal_rels_pipeline ON refinement.causal_relationships(pipeline_run_id);
CREATE INDEX idx_causal_rels_cause ON refinement.causal_relationships(cause_entity);
CREATE INDEX idx_causal_rels_effect ON refinement.causal_relationships(effect_entity);

-- ----------------------------------------
-- Conflicts Resolved
-- ----------------------------------------
CREATE TABLE refinement.conflicts_resolved (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    conflict_id text NOT NULL,
    conflict_type text NOT NULL,
    description text,

    involved_concepts text[] DEFAULT '{}',
    involved_tables text[] DEFAULT '{}',

    resolution_strategy text,
    resolution_details jsonb DEFAULT '{}',
    resolved_by text,

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE refinement.conflicts_resolved IS 'Ontology conflict resolution records';

CREATE INDEX idx_conflicts_pipeline ON refinement.conflicts_resolved(pipeline_run_id);
CREATE INDEX idx_conflicts_type ON refinement.conflicts_resolved(conflict_type);

-- ============================================================================
-- GOVERNANCE SCHEMA
-- ============================================================================

CREATE TABLE governance.governance_decisions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    concept_id bigint REFERENCES refinement.ontology_concepts(id) ON DELETE SET NULL,

    decision_id text NOT NULL,
    decision_type text NOT NULL CHECK (decision_type IN ('approve', 'reject', 'escalate', 'schedule_review', 'defer')),

    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    reasoning text,

    agent_opinions jsonb DEFAULT '{}',
    recommended_actions jsonb DEFAULT '[]',

    made_by text DEFAULT 'UnifiedPipeline',
    made_at timestamptz DEFAULT now(),

    created_at timestamptz DEFAULT now(),

    -- Full-text search for reasoning
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(reasoning, ''))
    ) STORED,

    CONSTRAINT governance_decisions_pipeline_decision_uniq UNIQUE(pipeline_run_id, decision_id)
);

COMMENT ON TABLE governance.governance_decisions IS 'Governance decisions for ontology concepts';

CREATE INDEX idx_decisions_pipeline ON governance.governance_decisions(pipeline_run_id);
CREATE INDEX idx_decisions_concept ON governance.governance_decisions(concept_id) WHERE concept_id IS NOT NULL;
CREATE INDEX idx_decisions_type ON governance.governance_decisions(decision_type);
CREATE INDEX idx_decisions_confidence ON governance.governance_decisions(confidence DESC);
CREATE INDEX idx_decisions_search ON governance.governance_decisions USING gin(search_vector);

-- ----------------------------------------
-- Action Backlog
-- ----------------------------------------
CREATE TABLE governance.action_backlog (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,
    decision_id bigint REFERENCES governance.governance_decisions(id) ON DELETE SET NULL,

    action_id text NOT NULL,
    action_type text NOT NULL,
    description text,

    priority text DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    priority_score numeric(5,4) DEFAULT 0.0,

    target_concept text,
    target_tables text[] DEFAULT '{}',

    status text DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled', 'blocked')),

    impact_score numeric(5,4) DEFAULT 0.0,
    roi_estimate numeric(10,2),

    due_date date,
    completed_at timestamptz,
    created_at timestamptz DEFAULT now(),

    CONSTRAINT action_backlog_pipeline_action_uniq UNIQUE(pipeline_run_id, action_id)
);

COMMENT ON TABLE governance.action_backlog IS 'Pending action items';

CREATE INDEX idx_actions_pipeline ON governance.action_backlog(pipeline_run_id);
CREATE INDEX idx_actions_decision ON governance.action_backlog(decision_id) WHERE decision_id IS NOT NULL;
-- Composite index for priority ordering (Rule 1.3)
CREATE INDEX idx_actions_priority ON governance.action_backlog(priority, priority_score DESC);
CREATE INDEX idx_actions_status ON governance.action_backlog(status);
-- Partial index for pending/in_progress actions
CREATE INDEX idx_actions_active ON governance.action_backlog(pipeline_run_id, priority, priority_score DESC)
    WHERE status IN ('pending', 'in_progress');

-- ----------------------------------------
-- Policy Rules
-- ----------------------------------------
CREATE TABLE governance.policy_rules (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    rule_id text NOT NULL,
    rule_name text NOT NULL,
    rule_type text CHECK (rule_type IN ('validation', 'quality', 'security', 'compliance', 'naming', 'structure')),

    condition jsonb NOT NULL,
    action jsonb NOT NULL,

    applies_to text[] DEFAULT '{}',

    is_active boolean DEFAULT TRUE,
    severity text DEFAULT 'warning' CHECK (severity IN ('error', 'warning', 'info')),

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE governance.policy_rules IS 'Data governance policy rules';

CREATE INDEX idx_policy_rules_pipeline ON governance.policy_rules(pipeline_run_id);
CREATE INDEX idx_policy_rules_type ON governance.policy_rules(rule_type);
-- Partial index for active rules (Rule 1.5)
CREATE INDEX idx_policy_rules_active ON governance.policy_rules(pipeline_run_id, rule_type)
    WHERE is_active = TRUE;

-- ----------------------------------------
-- Business Insights
-- ----------------------------------------
CREATE TABLE governance.business_insights (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    insight_id text NOT NULL,
    insight_type text,
    title text,
    description text,

    business_value text CHECK (business_value IN ('high', 'medium', 'low')),
    confidence numeric(5,4) DEFAULT 0.0,

    details jsonb DEFAULT '{}',
    recommendations text[],

    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, ''))
    ) STORED,

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE governance.business_insights IS 'Business insights from data analysis';

CREATE INDEX idx_business_insights_pipeline ON governance.business_insights(pipeline_run_id);
CREATE INDEX idx_business_insights_type ON governance.business_insights(insight_type);
CREATE INDEX idx_business_insights_search ON governance.business_insights USING gin(search_vector);

-- ============================================================================
-- EVIDENCE SCHEMA
-- ============================================================================

CREATE TABLE evidence.evidence_blocks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    block_id text NOT NULL,
    block_index integer NOT NULL,

    previous_hash text,
    current_hash text NOT NULL,

    phase text NOT NULL CHECK (phase IN ('discovery', 'refinement', 'governance')),
    agent text NOT NULL,
    evidence_type text NOT NULL,

    finding text NOT NULL,
    reasoning text,
    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    data_evidence jsonb DEFAULT '{}',

    timestamp timestamptz DEFAULT now(),

    -- Full-text search on findings
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(finding, '') || ' ' || coalesce(reasoning, ''))
    ) STORED,

    CONSTRAINT evidence_blocks_pipeline_block_uniq UNIQUE(pipeline_run_id, block_id)
);

COMMENT ON TABLE evidence.evidence_blocks IS 'Blockchain-style evidence chain';

CREATE INDEX idx_evidence_blocks_pipeline ON evidence.evidence_blocks(pipeline_run_id);
CREATE INDEX idx_evidence_blocks_phase ON evidence.evidence_blocks(phase);
CREATE INDEX idx_evidence_blocks_agent ON evidence.evidence_blocks(agent);
CREATE INDEX idx_evidence_blocks_hash ON evidence.evidence_blocks(current_hash);
CREATE INDEX idx_evidence_blocks_index ON evidence.evidence_blocks(block_index);
CREATE INDEX idx_evidence_blocks_search ON evidence.evidence_blocks USING gin(search_vector);
-- GIN for data evidence queries
CREATE INDEX idx_evidence_blocks_data_gin ON evidence.evidence_blocks USING gin(data_evidence);

-- ----------------------------------------
-- Consensus Records
-- ----------------------------------------
CREATE TABLE evidence.consensus_records (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    topic_id text NOT NULL,
    topic_name text,
    topic_type text,

    result_type text CHECK (result_type IN ('accepted', 'rejected', 'provisional', 'escalated')),
    combined_confidence numeric(5,4) DEFAULT 0.0,
    agreement_level numeric(5,4) DEFAULT 0.0,

    total_rounds integer DEFAULT 0,
    final_decision text,
    reasoning text,
    dissenting_views jsonb DEFAULT '[]',

    votes jsonb DEFAULT '[]',

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE evidence.consensus_records IS 'Multi-agent consensus discussion records';

CREATE INDEX idx_consensus_pipeline ON evidence.consensus_records(pipeline_run_id);
CREATE INDEX idx_consensus_topic ON evidence.consensus_records(topic_id);
CREATE INDEX idx_consensus_result ON evidence.consensus_records(result_type);
CREATE INDEX idx_consensus_votes_gin ON evidence.consensus_records USING gin(votes);

-- ----------------------------------------
-- Calibration History
-- ----------------------------------------
CREATE TABLE evidence.calibration_history (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    phase text NOT NULL,

    summary jsonb DEFAULT '{}',

    mean_confidence numeric(5,4),
    calibration_error numeric(5,4),

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE evidence.calibration_history IS 'LLM confidence calibration history';

CREATE INDEX idx_calibration_pipeline ON evidence.calibration_history(pipeline_run_id);
CREATE INDEX idx_calibration_phase ON evidence.calibration_history(phase);

-- ============================================================================
-- VERSIONING SCHEMA
-- ============================================================================

CREATE TABLE versioning.dataset_versions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    version_id text NOT NULL,
    version_number integer NOT NULL,

    message text,
    author text,

    snapshot_data jsonb,

    parent_version_id bigint REFERENCES versioning.dataset_versions(id) ON DELETE SET NULL,

    created_at timestamptz DEFAULT now(),

    CONSTRAINT dataset_versions_pipeline_version_uniq UNIQUE(pipeline_run_id, version_id)
);

COMMENT ON TABLE versioning.dataset_versions IS 'Git-style dataset version control';

CREATE INDEX idx_versions_pipeline ON versioning.dataset_versions(pipeline_run_id);
CREATE INDEX idx_versions_parent ON versioning.dataset_versions(parent_version_id) WHERE parent_version_id IS NOT NULL;
CREATE INDEX idx_versions_number ON versioning.dataset_versions(version_number);

-- ----------------------------------------
-- Branches
-- ----------------------------------------
CREATE TABLE versioning.branches (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    branch_name text NOT NULL,
    head_version_id bigint REFERENCES versioning.dataset_versions(id) ON DELETE SET NULL,

    is_default boolean DEFAULT FALSE,
    is_protected boolean DEFAULT FALSE,

    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),

    CONSTRAINT branches_pipeline_name_uniq UNIQUE(pipeline_run_id, branch_name)
);

COMMENT ON TABLE versioning.branches IS 'Dataset branches';

CREATE INDEX idx_branches_pipeline ON versioning.branches(pipeline_run_id);
CREATE INDEX idx_branches_head ON versioning.branches(head_version_id) WHERE head_version_id IS NOT NULL;
-- Partial index for default branches
CREATE INDEX idx_branches_default ON versioning.branches(pipeline_run_id) WHERE is_default = TRUE;

-- ============================================================================
-- ANALYTICS SCHEMA
-- ============================================================================

CREATE TABLE analytics.simulation_scenarios (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    scenario_id text NOT NULL,
    name text NOT NULL,
    description text,

    changes jsonb NOT NULL DEFAULT '[]',

    num_simulations integer DEFAULT 100,
    confidence_level numeric(5,4) DEFAULT 0.95,

    status text DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    created_at timestamptz DEFAULT now(),

    CONSTRAINT scenarios_pipeline_scenario_uniq UNIQUE(pipeline_run_id, scenario_id)
);

COMMENT ON TABLE analytics.simulation_scenarios IS 'What-If analysis scenario definitions';

CREATE INDEX idx_scenarios_pipeline ON analytics.simulation_scenarios(pipeline_run_id);
CREATE INDEX idx_scenarios_status ON analytics.simulation_scenarios(status);
-- Partial index for pending scenarios
CREATE INDEX idx_scenarios_pending ON analytics.simulation_scenarios(pipeline_run_id, created_at DESC)
    WHERE status = 'pending';

-- ----------------------------------------
-- Scenario Results
-- ----------------------------------------
CREATE TABLE analytics.scenario_results (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scenario_id bigint NOT NULL REFERENCES analytics.simulation_scenarios(id) ON DELETE CASCADE,

    success boolean DEFAULT FALSE,
    num_iterations integer DEFAULT 0,

    impacts jsonb DEFAULT '[]',
    mean_outcomes jsonb DEFAULT '{}',
    std_outcomes jsonb DEFAULT '{}',

    risk_score numeric(5,4) DEFAULT 0.0 CHECK (risk_score >= 0 AND risk_score <= 1),
    opportunity_score numeric(5,4) DEFAULT 0.0 CHECK (opportunity_score >= 0 AND opportunity_score <= 1),

    recommendations jsonb DEFAULT '[]',
    impact_summary text,

    execution_time_ms numeric(10,2),
    simulated_at timestamptz DEFAULT now()
);

COMMENT ON TABLE analytics.scenario_results IS 'What-If simulation results';

CREATE INDEX idx_scenario_results_scenario ON analytics.scenario_results(scenario_id);
CREATE INDEX idx_scenario_results_risk ON analytics.scenario_results(risk_score DESC);
CREATE INDEX idx_scenario_results_impacts_gin ON analytics.scenario_results USING gin(impacts);

-- ----------------------------------------
-- Unified Insights
-- ----------------------------------------
CREATE TABLE analytics.unified_insights (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    insight_id text NOT NULL,
    insight_type text CHECK (insight_type IN ('correlation', 'anomaly', 'pattern', 'prediction', 'recommendation')),

    title text,
    description text,

    confidence numeric(5,4) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),

    algorithm_source text,
    algorithm_result jsonb DEFAULT '{}',

    simulation_result jsonb DEFAULT '{}',
    llm_analysis jsonb DEFAULT '{}',

    related_concepts text[] DEFAULT '{}',
    related_actions text[] DEFAULT '{}',

    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, ''))
    ) STORED,

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE analytics.unified_insights IS 'Algorithm + Simulation + LLM unified insights';

CREATE INDEX idx_insights_pipeline ON analytics.unified_insights(pipeline_run_id);
CREATE INDEX idx_insights_type ON analytics.unified_insights(insight_type);
CREATE INDEX idx_insights_confidence ON analytics.unified_insights(confidence DESC);
CREATE INDEX idx_insights_search ON analytics.unified_insights USING gin(search_vector);
CREATE INDEX idx_insights_algorithm_gin ON analytics.unified_insights USING gin(algorithm_result);

-- ----------------------------------------
-- NL2SQL History
-- ----------------------------------------
CREATE TABLE analytics.nl2sql_history (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id bigint NOT NULL REFERENCES public.pipeline_runs(id) ON DELETE CASCADE,

    natural_query text NOT NULL,
    generated_sql text,

    success boolean DEFAULT FALSE,
    confidence numeric(5,4) DEFAULT 0.0,

    tables_used text[] DEFAULT '{}',
    execution_result jsonb,

    -- Full-text search on natural query
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(natural_query, ''))
    ) STORED,

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE analytics.nl2sql_history IS 'Natural language to SQL conversion history';

CREATE INDEX idx_nl2sql_pipeline ON analytics.nl2sql_history(pipeline_run_id);
CREATE INDEX idx_nl2sql_success ON analytics.nl2sql_history(success);
CREATE INDEX idx_nl2sql_search ON analytics.nl2sql_history USING gin(search_vector);
-- Partial index for successful queries
CREATE INDEX idx_nl2sql_successful ON analytics.nl2sql_history(pipeline_run_id, created_at DESC)
    WHERE success = TRUE;

-- ============================================================================
-- AUDIT SCHEMA
-- ============================================================================

CREATE TABLE audit.audit_logs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    schema_name text NOT NULL,
    table_name text NOT NULL,
    record_id bigint NOT NULL,

    action text NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_data jsonb,
    new_data jsonb,
    changed_fields text[],

    user_id text,
    user_agent text,
    ip_address inet,

    created_at timestamptz DEFAULT now()
);

COMMENT ON TABLE audit.audit_logs IS 'Audit log for all table changes';

CREATE INDEX idx_audit_logs_table ON audit.audit_logs(schema_name, table_name);
CREATE INDEX idx_audit_logs_record ON audit.audit_logs(record_id);
CREATE INDEX idx_audit_logs_action ON audit.audit_logs(action);
-- BRIN index for time-series data (Rule 1.2 - efficient for large tables)
CREATE INDEX idx_audit_logs_created_brin ON audit.audit_logs USING brin(created_at);

-- ============================================================================
-- TRIGGERS & FUNCTIONS
-- ============================================================================

-- Updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
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

-- Audit log trigger function (updated for bigint PKs)
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

-- ============================================================================
-- ROW LEVEL SECURITY (Rule 3.2)
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE public.pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery.source_tables ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery.homeomorphism_pairs ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery.unified_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery.fk_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery.column_semantics ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery.data_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE refinement.ontology_concepts ENABLE ROW LEVEL SECURITY;
ALTER TABLE refinement.concept_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE refinement.knowledge_graph_triples ENABLE ROW LEVEL SECURITY;
ALTER TABLE refinement.causal_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE refinement.conflicts_resolved ENABLE ROW LEVEL SECURITY;
ALTER TABLE governance.governance_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE governance.action_backlog ENABLE ROW LEVEL SECURITY;
ALTER TABLE governance.policy_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE governance.business_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence.evidence_blocks ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence.consensus_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence.calibration_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE versioning.dataset_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE versioning.branches ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.simulation_scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.scenario_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.unified_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.nl2sql_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies for pipeline_runs (Rule 3.3 - optimized with SELECT wrapper)
CREATE POLICY pipeline_runs_tenant_policy ON public.pipeline_runs
    FOR ALL
    USING (tenant_id = (SELECT current_setting('app.tenant_id', true)));

-- Helper function to get pipeline tenant (Rule 3.3 - security definer)
CREATE OR REPLACE FUNCTION get_pipeline_tenant(p_pipeline_run_id bigint)
RETURNS text
LANGUAGE sql
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT tenant_id FROM public.pipeline_runs WHERE id = p_pipeline_run_id
$$;

-- RLS Policies for child tables (inherit from pipeline_runs)
CREATE POLICY source_tables_tenant_policy ON discovery.source_tables
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY homeomorphism_pairs_tenant_policy ON discovery.homeomorphism_pairs
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY unified_entities_tenant_policy ON discovery.unified_entities
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY fk_candidates_tenant_policy ON discovery.fk_candidates
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY column_semantics_tenant_policy ON discovery.column_semantics
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY data_patterns_tenant_policy ON discovery.data_patterns
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY ontology_concepts_tenant_policy ON refinement.ontology_concepts
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY concept_relationships_tenant_policy ON refinement.concept_relationships
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY knowledge_graph_triples_tenant_policy ON refinement.knowledge_graph_triples
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY causal_relationships_tenant_policy ON refinement.causal_relationships
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY conflicts_resolved_tenant_policy ON refinement.conflicts_resolved
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY governance_decisions_tenant_policy ON governance.governance_decisions
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY action_backlog_tenant_policy ON governance.action_backlog
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY policy_rules_tenant_policy ON governance.policy_rules
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY business_insights_tenant_policy ON governance.business_insights
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY evidence_blocks_tenant_policy ON evidence.evidence_blocks
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY consensus_records_tenant_policy ON evidence.consensus_records
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY calibration_history_tenant_policy ON evidence.calibration_history
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY dataset_versions_tenant_policy ON versioning.dataset_versions
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY branches_tenant_policy ON versioning.branches
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY simulation_scenarios_tenant_policy ON analytics.simulation_scenarios
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY unified_insights_tenant_policy ON analytics.unified_insights
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

CREATE POLICY nl2sql_history_tenant_policy ON analytics.nl2sql_history
    FOR ALL
    USING ((SELECT get_pipeline_tenant(pipeline_run_id)) = (SELECT current_setting('app.tenant_id', true)));

-- scenario_results inherits from simulation_scenarios
CREATE POLICY scenario_results_tenant_policy ON analytics.scenario_results
    FOR ALL
    USING (
        (SELECT get_pipeline_tenant(ss.pipeline_run_id)
         FROM analytics.simulation_scenarios ss
         WHERE ss.id = scenario_id) = (SELECT current_setting('app.tenant_id', true))
    );

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW public.v_pipeline_summary AS
SELECT
    pr.id,
    pr.scenario_name,
    pr.domain_context,
    pr.status,
    pr.tenant_id,
    pr.started_at,
    pr.completed_at,
    EXTRACT(EPOCH FROM (pr.completed_at - pr.started_at)) as duration_seconds,
    (SELECT COUNT(*) FROM discovery.source_tables st WHERE st.pipeline_run_id = pr.id) as table_count,
    (SELECT COUNT(*) FROM discovery.unified_entities ue WHERE ue.pipeline_run_id = pr.id) as entity_count,
    (SELECT COUNT(*) FROM refinement.ontology_concepts oc WHERE oc.pipeline_run_id = pr.id) as concept_count,
    (SELECT COUNT(*) FROM governance.governance_decisions gd WHERE gd.pipeline_run_id = pr.id) as decision_count,
    (SELECT COUNT(*) FROM governance.action_backlog ab WHERE ab.pipeline_run_id = pr.id) as action_count
FROM public.pipeline_runs pr;

COMMENT ON VIEW public.v_pipeline_summary IS 'Pipeline execution summary view';

CREATE OR REPLACE VIEW refinement.v_concept_status AS
SELECT
    oc.pipeline_run_id,
    oc.status,
    COUNT(*) as count,
    AVG(oc.confidence) as avg_confidence
FROM refinement.ontology_concepts oc
GROUP BY oc.pipeline_run_id, oc.status;

COMMENT ON VIEW refinement.v_concept_status IS 'Ontology concept status statistics';

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Full-text search function
CREATE OR REPLACE FUNCTION search_all_content(
    p_tenant_id text,
    p_query text,
    p_limit integer DEFAULT 50
)
RETURNS TABLE (
    source_type text,
    source_id bigint,
    title text,
    snippet text,
    rank real
)
LANGUAGE sql
STABLE
AS $$
    WITH all_content AS (
        SELECT
            'pipeline_run' as source_type,
            id as source_id,
            scenario_name as title,
            domain_context as snippet,
            search_vector
        FROM public.pipeline_runs
        WHERE tenant_id = p_tenant_id

        UNION ALL

        SELECT
            'concept' as source_type,
            oc.id as source_id,
            oc.name as title,
            oc.description as snippet,
            oc.search_vector
        FROM refinement.ontology_concepts oc
        JOIN public.pipeline_runs pr ON pr.id = oc.pipeline_run_id
        WHERE pr.tenant_id = p_tenant_id

        UNION ALL

        SELECT
            'insight' as source_type,
            ui.id as source_id,
            ui.title,
            ui.description as snippet,
            ui.search_vector
        FROM analytics.unified_insights ui
        JOIN public.pipeline_runs pr ON pr.id = ui.pipeline_run_id
        WHERE pr.tenant_id = p_tenant_id

        UNION ALL

        SELECT
            'evidence' as source_type,
            eb.id as source_id,
            eb.finding as title,
            eb.reasoning as snippet,
            eb.search_vector
        FROM evidence.evidence_blocks eb
        JOIN public.pipeline_runs pr ON pr.id = eb.pipeline_run_id
        WHERE pr.tenant_id = p_tenant_id
    )
    SELECT
        source_type,
        source_id,
        title,
        snippet,
        ts_rank(search_vector, websearch_to_tsquery('english', p_query)) as rank
    FROM all_content
    WHERE search_vector @@ websearch_to_tsquery('english', p_query)
    ORDER BY rank DESC
    LIMIT p_limit;
$$;

COMMENT ON FUNCTION search_all_content IS 'Full-text search across all content types';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

INSERT INTO audit.audit_logs (schema_name, table_name, record_id, action, new_data)
VALUES ('system', 'migrations', 2, 'INSERT',
    jsonb_build_object(
        'migration', '002_supabase_best_practices',
        'completed_at', now()::text,
        'changes', ARRAY[
            'bigint identity PKs',
            'TEXT instead of VARCHAR',
            'All FK indexes added',
            'GIN indexes for JSONB',
            'RLS policies enabled',
            'tsvector full-text search',
            'Partial indexes for common filters',
            'BRIN index for audit logs'
        ]
    ));

COMMENT ON SCHEMA public IS 'Ontoloty v17.2 - Core schema (Supabase optimized)';
COMMENT ON SCHEMA discovery IS 'Ontoloty v17.2 - Phase 1: Discovery';
COMMENT ON SCHEMA refinement IS 'Ontoloty v17.2 - Phase 2: Refinement';
COMMENT ON SCHEMA governance IS 'Ontoloty v17.2 - Phase 3: Governance';
COMMENT ON SCHEMA evidence IS 'Ontoloty v17.2 - Cross-cutting: Evidence Chain';
COMMENT ON SCHEMA versioning IS 'Ontoloty v17.2 - Cross-cutting: Version Control';
COMMENT ON SCHEMA analytics IS 'Ontoloty v17.2 - Post-pipeline: Analytics';
COMMENT ON SCHEMA audit IS 'Ontoloty v17.2 - Audit Logs';
