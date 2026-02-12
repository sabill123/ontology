#!/usr/bin/env python3
"""
Detailed Pipeline Report - Shows ALL results from each phase
"""
import asyncio
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.unified_pipeline.autonomous_pipeline import AutonomousPipelineOrchestrator


def df_to_tables_data(df: pd.DataFrame, table_name: str = "shipments") -> dict:
    """Convert DataFrame to orchestrator-compatible tables_data format."""
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype:
            col_type = "integer"
        elif "float" in dtype:
            col_type = "float"
        elif "datetime" in dtype or "date" in dtype.lower():
            col_type = "datetime"
        elif "bool" in dtype:
            col_type = "boolean"
        else:
            col_type = "string"

        columns_info.append({
            "name": col,
            "type": col_type,
            "nullable": df[col].isnull().any(),
            "unique_count": df[col].nunique(),
            "null_count": int(df[col].isnull().sum()),
            "sample_values": df[col].dropna().head(5).tolist(),
        })

    return {
        table_name: {
            "columns": columns_info,
            "row_count": len(df),
            "sample_data": df.head(10).to_dict(orient="records"),
            "dataframe": df,
        }
    }


def print_separator(title: str, char: str = "="):
    width = 80
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_subsection(title: str):
    print(f"\n--- {title} ---")


def safe_get(obj, key, default=None):
    """Safely get attribute or dict key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def obj_to_dict(obj):
    """Convert object to dict representation."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '__dict__'):
        return vars(obj)
    if hasattr(obj, '_asdict'):
        return obj._asdict()
    return {"value": str(obj)}


async def run_pipeline_and_report(csv_path: str):
    """Run pipeline and generate detailed report."""

    print_separator("ONTOLOTY DETAILED PIPELINE REPORT", "█")
    print(f"\nInput file: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Initialize pipeline with output directory
    import uuid
    scenario_id = f"shipments_v22_{uuid.uuid4().hex[:8]}"
    output_dir = f"data/output/{scenario_id}"

    pipeline = AutonomousPipelineOrchestrator(
        scenario_name=scenario_id,
        output_dir=output_dir,
        enable_v17_services=True,
    )
    print(f"Output directory: {output_dir}")

    # Convert to tables_data format
    tables_data = df_to_tables_data(df, "shipments")

    # Run pipeline
    print_separator("RUNNING PIPELINE", "-")
    events = []
    async for event in pipeline.run(tables_data=tables_data):
        event_type = event.get("type", "unknown")
        if event_type == "phase_complete":
            phase = event.get("phase", "unknown")
            print(f"  ✓ Phase completed: {phase}")
        elif event_type == "task_complete":
            task = event.get("task", event.get("todo_description", "unknown"))
            print(f"    - Task: {task[:60]}...")
        elif event_type == "domain_detection_complete":
            print(f"  ✓ Domain detected: {event.get('domain', 'unknown')}")
        elif event_type == "pipeline_complete":
            print("  ✓ Pipeline complete!")
        events.append(event)

    print(f"\nTotal events: {len(events)}")

    # Get shared context from pipeline
    ctx = pipeline.shared_context

    # =========================================================================
    # PHASE 1: DISCOVERY RESULTS
    # =========================================================================
    print_separator("PHASE 1: DISCOVERY RESULTS", "=")

    # 1.1 Domain Detection
    print_subsection("1.1 Domain Detection")
    domain = ctx.domain_context if hasattr(ctx, 'domain_context') else None
    detected_domain = ctx._detected_domain if hasattr(ctx, '_detected_domain') else None
    if detected_domain:
        print(f"  Primary Domain: {safe_get(detected_domain, 'industry', 'N/A')}")
        print(f"  Sub-domain: {safe_get(detected_domain, 'sub_industry', 'N/A')}")
        print(f"  Confidence: {safe_get(detected_domain, 'industry_confidence', 'N/A')}")
        print(f"  Key Entities: {safe_get(detected_domain, 'key_entities', [])}")
        print(f"  Key Metrics: {safe_get(detected_domain, 'key_metrics', [])}")
    elif domain:
        if isinstance(domain, str):
            print(f"  Domain: {domain}")
        else:
            print(f"  Domain: {safe_get(domain, 'industry', str(domain))}")
    else:
        print("  No domain detection data")

    # 1.2 Tables Info
    print_subsection("1.2 Tables Info")
    tables = ctx.tables if hasattr(ctx, 'tables') else {}
    if tables:
        print(f"  Total tables: {len(tables)}")
        for tbl_name, tbl_info in tables.items():
            print(f"\n  Table: {tbl_name}")
            if hasattr(tbl_info, 'columns'):
                print(f"    Columns: {len(tbl_info.columns)}")
                for col in list(tbl_info.columns)[:10]:
                    col_info = obj_to_dict(col)
                    print(f"      - {safe_get(col_info, 'name', col)}: {safe_get(col_info, 'dtype', 'unknown')}")
            if hasattr(tbl_info, 'row_count'):
                print(f"    Rows: {tbl_info.row_count}")
    else:
        print("  No tables info")

    # 1.3 Schema Analysis
    print_subsection("1.3 Schema Analysis")
    schema = ctx.schema_analysis if hasattr(ctx, 'schema_analysis') else {}
    dynamic_schema = ctx.dynamic_data.get('schema_analysis', {}) if hasattr(ctx, 'dynamic_data') else {}
    schema = schema or dynamic_schema
    if schema:
        for tbl_name, analysis in schema.items() if isinstance(schema, dict) else [("main", schema)]:
            print(f"\n  Table: {tbl_name}")
            analysis_dict = obj_to_dict(analysis)
            print(f"    Primary Key Candidates: {analysis_dict.get('primary_key_candidates', [])}")
            print(f"    Foreign Key Candidates: {analysis_dict.get('fk_candidates', [])}")
            if analysis_dict.get('data_patterns'):
                print(f"    Data Patterns: {analysis_dict.get('data_patterns')}")
    else:
        print("  No schema analysis data")

    # 1.4 Data Understanding (from dynamic_data)
    print_subsection("1.4 Data Understanding (LLM Analysis)")
    data_understanding = ctx.dynamic_data.get('data_understanding', {}) if hasattr(ctx, 'dynamic_data') else {}
    if data_understanding:
        print(f"  Domain: {data_understanding.get('domain', 'N/A')}")
        print(f"  Summary: {str(data_understanding.get('summary', 'N/A'))[:300]}")
        if data_understanding.get('key_entities'):
            print(f"  Key Entities: {data_understanding.get('key_entities')}")
        if data_understanding.get('key_relationships'):
            print(f"  Key Relationships: {data_understanding.get('key_relationships')}")
        if data_understanding.get('column_semantics'):
            print("  Column Semantics:")
            for col, sem in list(data_understanding.get('column_semantics', {}).items())[:15]:
                print(f"    - {col}: {sem}")
    else:
        print("  No data understanding results")

    # 1.5 TDA Signatures
    print_subsection("1.5 TDA (Topological Data Analysis) Signatures")
    tda = ctx.tda_signatures if hasattr(ctx, 'tda_signatures') else {}
    dynamic_tda = ctx.dynamic_data.get('tda_analysis', {}) if hasattr(ctx, 'dynamic_data') else {}
    tda = tda or dynamic_tda
    if tda:
        print(f"  Betti Numbers: {tda.get('betti_numbers', 'N/A')}")
        print(f"  Persistence Diagrams: {len(tda.get('persistence_diagrams', []))} computed")
        print(f"  Data Topology: {tda.get('data_topology', tda.get('topological_summary', 'N/A'))}")
        if tda.get('clusters'):
            print(f"  Clusters detected: {len(tda.get('clusters', []))}")
        if tda.get('anomalies'):
            print(f"  Anomalies: {tda.get('anomalies')}")
    else:
        print("  No TDA signatures computed")

    # 1.6 Value Overlaps
    print_subsection("1.6 Value Overlaps Analysis")
    overlaps = ctx.value_overlaps if hasattr(ctx, 'value_overlaps') else []
    dynamic_overlaps = ctx.dynamic_data.get('value_overlaps', []) if hasattr(ctx, 'dynamic_data') else []
    overlaps = overlaps or dynamic_overlaps
    if overlaps:
        print(f"  Total overlaps detected: {len(overlaps)}")
        for i, overlap in enumerate(overlaps[:10], 1):
            overlap_dict = obj_to_dict(overlap)
            print(f"\n  Overlap #{i}:")
            print(f"    Columns: {overlap_dict.get('column1', 'N/A')} <-> {overlap_dict.get('column2', '')}")
            print(f"    Overlap Ratio: {overlap_dict.get('overlap_ratio', overlap_dict.get('similarity', 'N/A'))}")
            print(f"    Type: {overlap_dict.get('overlap_type', 'N/A')}")
    else:
        print("  No value overlaps detected")

    # 1.7 Unified Entities
    print_subsection("1.7 Unified Entities (Entity Classification)")
    entities = ctx.unified_entities if hasattr(ctx, 'unified_entities') else []
    if entities:
        print(f"  Total entities discovered: {len(entities)}")
        for i, entity in enumerate(entities, 1):
            ent_dict = obj_to_dict(entity)
            print(f"\n  Entity #{i}:")
            print(f"    Name: {ent_dict.get('canonical_name', ent_dict.get('name', 'N/A'))}")
            print(f"    Type: {ent_dict.get('entity_type', ent_dict.get('type', 'N/A'))}")
            print(f"    Confidence: {ent_dict.get('confidence', 'N/A')}")
            if ent_dict.get('source_columns'):
                print(f"    Source Columns: {ent_dict.get('source_columns')}")
            if ent_dict.get('attributes'):
                attrs = ent_dict.get('attributes', [])
                print(f"    Attributes: {attrs[:5]}{'...' if len(attrs) > 5 else ''}")
    else:
        print("  No unified entities found")

    # 1.8 FK Candidates
    print_subsection("1.8 Foreign Key Candidates")
    fk_candidates = ctx.enhanced_fk_candidates if hasattr(ctx, 'enhanced_fk_candidates') else []
    if fk_candidates:
        print(f"  Total FK candidates: {len(fk_candidates)}")
        for i, fk in enumerate(fk_candidates, 1):
            fk_dict = obj_to_dict(fk)
            print(f"\n  FK #{i}:")
            print(f"    From: {fk_dict.get('source_table', 'N/A')}.{fk_dict.get('source_column', 'N/A')}")
            print(f"    To: {fk_dict.get('target_table', 'N/A')}.{fk_dict.get('target_column', 'N/A')}")
            print(f"    Confidence: {fk_dict.get('confidence', 'N/A')}")
            print(f"    Evidence: {fk_dict.get('evidence', fk_dict.get('reason', 'N/A'))}")
    else:
        print("  No FK candidates found")

    # 1.9 Concept Relationships
    print_subsection("1.9 Concept Relationships (Discovery Phase)")
    relationships = ctx.concept_relationships if hasattr(ctx, 'concept_relationships') else []
    if relationships:
        print(f"  Total relationships: {len(relationships)}")
        for i, rel in enumerate(relationships[:15], 1):
            rel_dict = obj_to_dict(rel)
            print(f"  #{i}: {rel_dict.get('source', 'N/A')} --[{rel_dict.get('relationship_type', rel_dict.get('type', 'N/A'))}]--> {rel_dict.get('target', 'N/A')}")
    else:
        print("  No concept relationships found")

    # 1.10 Homeomorphisms
    print_subsection("1.10 Homeomorphisms")
    homeos = ctx.homeomorphisms if hasattr(ctx, 'homeomorphisms') else []
    if homeos:
        print(f"  Total homeomorphism pairs: {len(homeos)}")
        for i, h in enumerate(homeos[:10], 1):
            h_dict = obj_to_dict(h)
            print(f"  #{i}: {h_dict.get('source', 'N/A')} ↔ {h_dict.get('target', 'N/A')} (similarity: {h_dict.get('similarity', 'N/A')})")
    else:
        print("  No homeomorphisms detected")

    # =========================================================================
    # PHASE 2: REFINEMENT RESULTS
    # =========================================================================
    print_separator("PHASE 2: REFINEMENT RESULTS", "=")

    # 2.1 Ontology Concepts
    print_subsection("2.1 Ontology Concepts")
    concepts = ctx.ontology_concepts if hasattr(ctx, 'ontology_concepts') else []
    if concepts:
        print(f"  Total concepts: {len(concepts)}")

        # Group by concept type
        object_types = [c for c in concepts if safe_get(obj_to_dict(c), 'concept_type', '') == 'object_type']
        properties = [c for c in concepts if safe_get(obj_to_dict(c), 'concept_type', '') == 'property']
        link_types = [c for c in concepts if safe_get(obj_to_dict(c), 'concept_type', '') == 'link_type']

        print(f"\n  Object Types ({len(object_types)}):")
        for ot in object_types[:15]:
            ot_dict = obj_to_dict(ot)
            print(f"    - {ot_dict.get('name', 'N/A')}")
            if ot_dict.get('description'):
                print(f"      Description: {str(ot_dict.get('description'))[:100]}")
            if ot_dict.get('properties'):
                print(f"      Properties: {ot_dict.get('properties')[:5]}")

        print(f"\n  Properties ({len(properties)}):")
        for p in properties[:15]:
            p_dict = obj_to_dict(p)
            print(f"    - {p_dict.get('name', 'N/A')}: {p_dict.get('data_type', p_dict.get('type', 'N/A'))}")
            if p_dict.get('domain'):
                print(f"      Domain: {p_dict.get('domain')}")

        print(f"\n  Link Types ({len(link_types)}):")
        for lt in link_types[:15]:
            lt_dict = obj_to_dict(lt)
            print(f"    - {lt_dict.get('name', 'N/A')}: {lt_dict.get('source', 'N/A')} -> {lt_dict.get('target', 'N/A')}")
    else:
        print("  No ontology concepts generated")

    # 2.2 Conflicts Resolved
    print_subsection("2.2 Conflicts Resolved")
    resolutions = ctx.conflicts_resolved if hasattr(ctx, 'conflicts_resolved') else []
    if resolutions:
        print(f"  Total resolutions: {len(resolutions)}")
        for i, res in enumerate(resolutions, 1):
            res_dict = obj_to_dict(res)
            print(f"\n  Resolution #{i}:")
            print(f"    Conflict ID: {res_dict.get('conflict_id', 'N/A')}")
            print(f"    Resolution Type: {res_dict.get('resolution_type', res_dict.get('type', 'N/A'))}")
            print(f"    Action: {str(res_dict.get('action', res_dict.get('resolution', 'N/A')))[:150]}")
    else:
        print("  No conflicts resolved")

    # 2.3 Knowledge Graph Triples
    print_subsection("2.3 Knowledge Graph Triples")
    kg_triples = ctx.knowledge_graph_triples if hasattr(ctx, 'knowledge_graph_triples') else []
    base_triples = ctx.semantic_base_triples if hasattr(ctx, 'semantic_base_triples') else []
    inferred = ctx.inferred_triples if hasattr(ctx, 'inferred_triples') else []

    all_triples = kg_triples + base_triples + inferred
    if all_triples:
        print(f"  Total triples: {len(all_triples)}")
        print(f"    - Knowledge graph: {len(kg_triples)}")
        print(f"    - Semantic base: {len(base_triples)}")
        print(f"    - Inferred: {len(inferred)}")
        print("\n  Sample triples:")
        for i, triple in enumerate(all_triples[:15], 1):
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                print(f"    #{i}: ({triple[0]}, {triple[1]}, {triple[2]})")
            elif isinstance(triple, dict):
                print(f"    #{i}: ({triple.get('subject', 'N/A')}, {triple.get('predicate', 'N/A')}, {triple.get('object', 'N/A')})")
            else:
                print(f"    #{i}: {triple}")
    else:
        print("  No knowledge graph triples")

    # 2.4 Causal Relationships
    print_subsection("2.4 Causal Relationships")
    causal = ctx.causal_relationships if hasattr(ctx, 'causal_relationships') else []
    if causal:
        print(f"  Total causal relationships: {len(causal)}")
        for i, cr in enumerate(causal[:10], 1):
            cr_dict = obj_to_dict(cr)
            print(f"  #{i}: {cr_dict.get('cause', 'N/A')} -> {cr_dict.get('effect', 'N/A')} (strength: {cr_dict.get('strength', 'N/A')})")
    else:
        print("  No causal relationships detected")

    # =========================================================================
    # PHASE 3: GOVERNANCE RESULTS
    # =========================================================================
    print_separator("PHASE 3: GOVERNANCE RESULTS", "=")

    # 3.1 Governance Decisions
    print_subsection("3.1 Governance Decisions")
    decisions = ctx.governance_decisions if hasattr(ctx, 'governance_decisions') else []
    if decisions:
        print(f"  Total decisions: {len(decisions)}")
        for i, decision in enumerate(decisions, 1):
            dec_dict = obj_to_dict(decision)
            print(f"\n  Decision #{i}:")
            print(f"    ID: {dec_dict.get('decision_id', dec_dict.get('id', f'DEC-{i}'))}")
            print(f"    Type: {dec_dict.get('decision_type', dec_dict.get('type', 'N/A'))}")
            print(f"    Status: {dec_dict.get('status', 'N/A')}")
            print(f"    Title: {dec_dict.get('title', dec_dict.get('name', 'N/A'))}")
            if dec_dict.get('description'):
                print(f"    Description: {str(dec_dict.get('description'))[:150]}")
            if dec_dict.get('rationale'):
                print(f"    Rationale: {str(dec_dict.get('rationale'))[:100]}")
    else:
        print("  No governance decisions")

    # 3.2 Policy Rules
    print_subsection("3.2 Policy Rules Generated")
    policies = ctx.policy_rules if hasattr(ctx, 'policy_rules') else []
    if policies:
        print(f"  Total policy rules: {len(policies)}")
        for i, policy in enumerate(policies[:25], 1):
            pol_dict = obj_to_dict(policy)
            print(f"\n  Policy #{i}:")
            print(f"    ID: {pol_dict.get('policy_id', pol_dict.get('id', f'POL-{i}'))}")
            print(f"    Name: {pol_dict.get('name', pol_dict.get('title', 'N/A'))}")
            print(f"    Type: {pol_dict.get('policy_type', pol_dict.get('type', 'N/A'))}")
            print(f"    Priority: {pol_dict.get('priority', 'N/A')}")
            if pol_dict.get('description'):
                print(f"    Description: {str(pol_dict.get('description'))[:150]}")
    else:
        print("  No policy rules generated")

    # 3.3 Action Backlog
    print_subsection("3.3 Action Backlog")
    actions = ctx.action_backlog if hasattr(ctx, 'action_backlog') else []
    if actions:
        print(f"  Total actions: {len(actions)}")
        for i, action in enumerate(actions, 1):
            act_dict = obj_to_dict(action)
            print(f"\n  Action #{i}:")
            print(f"    Title: {act_dict.get('title', act_dict.get('name', act_dict.get('action', 'N/A')))}")
            print(f"    Priority: {act_dict.get('priority', 'N/A')}")
            print(f"    Status: {act_dict.get('status', 'pending')}")
            if act_dict.get('description'):
                print(f"    Description: {str(act_dict.get('description'))[:100]}")
    else:
        print("  No action backlog items")

    # 3.4 Business Insights
    print_subsection("3.4 Business Insights")
    insights = ctx.business_insights if hasattr(ctx, 'business_insights') else []
    if insights:
        print(f"  Total insights: {len(insights)}")
        for i, ins in enumerate(insights[:10], 1):
            ins_dict = obj_to_dict(ins)
            print(f"\n  Insight #{i}:")
            print(f"    Title: {ins_dict.get('title', ins_dict.get('name', 'N/A'))}")
            print(f"    Type: {ins_dict.get('insight_type', ins_dict.get('type', 'N/A'))}")
            if ins_dict.get('description'):
                print(f"    Description: {str(ins_dict.get('description'))[:150]}")
    else:
        print("  No business insights")

    # 3.5 Palantir Insights
    print_subsection("3.5 Palantir-style Predictive Insights")
    palantir = ctx.palantir_insights if hasattr(ctx, 'palantir_insights') else []
    if palantir:
        print(f"  Total Palantir insights: {len(palantir)}")
        for i, pi in enumerate(palantir[:10], 1):
            pi_dict = obj_to_dict(pi)
            print(f"\n  Insight #{i}:")
            print(f"    Title: {pi_dict.get('title', pi_dict.get('insight', 'N/A'))}")
            print(f"    Type: {pi_dict.get('insight_type', 'N/A')}")
            print(f"    Confidence: {pi_dict.get('confidence', 'N/A')}")
            if pi_dict.get('prediction'):
                print(f"    Prediction: {str(pi_dict.get('prediction'))[:100]}")
            if pi_dict.get('recommendation'):
                print(f"    Recommendation: {str(pi_dict.get('recommendation'))[:100]}")
    else:
        print("  No Palantir insights")

    # =========================================================================
    # EVIDENCE CHAIN & CONSENSUS
    # =========================================================================
    print_separator("EVIDENCE CHAIN & CONSENSUS", "=")

    # Evidence Chain
    print_subsection("Evidence Chain Blocks")
    evidence_chain = ctx.evidence_chain if hasattr(ctx, 'evidence_chain') else None
    if evidence_chain and hasattr(evidence_chain, 'get_all_blocks'):
        blocks = evidence_chain.get_all_blocks()
        print(f"  Total evidence blocks: {len(blocks)}")
        for i, block in enumerate(blocks[:15], 1):
            blk_dict = obj_to_dict(block)
            print(f"\n  Block #{i}:")
            print(f"    ID: {blk_dict.get('block_id', blk_dict.get('id', 'N/A'))}")
            print(f"    Agent: {blk_dict.get('agent_name', blk_dict.get('agent', 'N/A'))}")
            print(f"    Phase: {blk_dict.get('phase', 'N/A')}")
            print(f"    Type: {blk_dict.get('evidence_type', blk_dict.get('type', 'N/A'))}")
            finding = blk_dict.get('finding', blk_dict.get('content', 'N/A'))
            print(f"    Finding: {str(finding)[:100]}")
            print(f"    Confidence: {blk_dict.get('confidence', 'N/A')}")
    else:
        print("  No evidence chain available")

    # Debate Results
    print_subsection("Debate Results")
    debate = ctx.debate_results if hasattr(ctx, 'debate_results') else {}
    ev_debate = ctx.evidence_based_debate_results if hasattr(ctx, 'evidence_based_debate_results') else {}
    debate = debate or ev_debate
    if debate:
        print(f"  Topic: {debate.get('topic', 'N/A')}")
        print(f"  Outcome: {debate.get('outcome', debate.get('conclusion', 'N/A'))}")
        if debate.get('votes'):
            print(f"  Votes: {debate.get('votes')}")
        if debate.get('arguments'):
            print(f"  Arguments recorded: {len(debate.get('arguments', []))}")
    else:
        print("  No debate results")

    # =========================================================================
    # DYNAMIC DATA (LLM Analysis Results)
    # =========================================================================
    print_separator("DYNAMIC DATA (LLM Analysis Results)", "=")

    dynamic = ctx.dynamic_data if hasattr(ctx, 'dynamic_data') else {}
    if dynamic:
        print(f"  Keys stored: {list(dynamic.keys())}")
        for key, value in dynamic.items():
            print_subsection(f"Dynamic: {key}")
            if isinstance(value, dict):
                for k, v in list(value.items())[:10]:
                    print(f"    {k}: {str(v)[:100]}")
            elif isinstance(value, list):
                print(f"    Count: {len(value)}")
                for item in value[:5]:
                    print(f"    - {str(item)[:100]}")
            else:
                print(f"    {str(value)[:200]}")
    else:
        print("  No dynamic data stored")

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print_separator("PIPELINE SUMMARY STATISTICS", "=")

    concepts = ctx.ontology_concepts if hasattr(ctx, 'ontology_concepts') else []
    object_types = [c for c in concepts if safe_get(obj_to_dict(c), 'concept_type', '') == 'object_type']
    properties = [c for c in concepts if safe_get(obj_to_dict(c), 'concept_type', '') == 'property']
    link_types = [c for c in concepts if safe_get(obj_to_dict(c), 'concept_type', '') == 'link_type']

    stats = {
        "Tables loaded": len(ctx.tables) if hasattr(ctx, 'tables') else 0,
        "Entities discovered": len(ctx.unified_entities) if hasattr(ctx, 'unified_entities') else 0,
        "FK candidates": len(ctx.enhanced_fk_candidates) if hasattr(ctx, 'enhanced_fk_candidates') else 0,
        "Value overlaps": len(ctx.value_overlaps) if hasattr(ctx, 'value_overlaps') else 0,
        "Homeomorphisms": len(ctx.homeomorphisms) if hasattr(ctx, 'homeomorphisms') else 0,
        "Concept relationships": len(ctx.concept_relationships) if hasattr(ctx, 'concept_relationships') else 0,
        "Ontology concepts (total)": len(concepts),
        "  - Object types": len(object_types),
        "  - Properties": len(properties),
        "  - Link types": len(link_types),
        "Knowledge graph triples": len(ctx.knowledge_graph_triples) if hasattr(ctx, 'knowledge_graph_triples') else 0,
        "Causal relationships": len(ctx.causal_relationships) if hasattr(ctx, 'causal_relationships') else 0,
        "Conflicts resolved": len(ctx.conflicts_resolved) if hasattr(ctx, 'conflicts_resolved') else 0,
        "Governance decisions": len(ctx.governance_decisions) if hasattr(ctx, 'governance_decisions') else 0,
        "Policy rules": len(ctx.policy_rules) if hasattr(ctx, 'policy_rules') else 0,
        "Action items": len(ctx.action_backlog) if hasattr(ctx, 'action_backlog') else 0,
        "Business insights": len(ctx.business_insights) if hasattr(ctx, 'business_insights') else 0,
        "Palantir insights": len(ctx.palantir_insights) if hasattr(ctx, 'palantir_insights') else 0,
        "Dynamic data keys": len(ctx.dynamic_data) if hasattr(ctx, 'dynamic_data') else 0,
    }

    for key, value in stats.items():
        print(f"  {key}: {value}")

    print_separator("END OF REPORT", "█")


if __name__ == "__main__":
    csv_path = "/Users/jaeseokhan/Desktop/Work/ontoloty/data/sample_csv/shipments.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    asyncio.run(run_pipeline_and_report(csv_path))
