#!/usr/bin/env python3
"""
Test healthcare_silo FK detection with LLM Semantic Enhancer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path

def load_silo_data(silo_path: str):
    """Load all CSV files from silo directory into tables_data format."""
    path = Path(silo_path)
    tables_data = {}

    for csv_file in path.glob("*.csv"):
        table_name = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            tables_data[table_name] = {
                'columns': list(df.columns),
                'data': df.values.tolist(),
                'sample_data': df.head(50).to_dict('records'),
            }
            print(f"  Loaded {table_name}: {len(df)} rows, {len(df.columns)} cols")
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")

    return tables_data


def test_llm_detection():
    print("=" * 60)
    print("HEALTHCARE SILO LLM FK DETECTION TEST")
    print("=" * 60)
    print()

    # Load data
    silo_path = "/Users/jaeseokhan/Desktop/Work/ontoloty/data/healthcare_silo"
    print("[1] Loading healthcare_silo dataset...")
    tables_data = load_silo_data(silo_path)
    print(f"Total: {len(tables_data)} tables")
    print()

    # Show columns with semantic role patterns
    print("[2] Columns with semantic role patterns (potential LLM targets):")
    semantic_patterns = ["_by", "_doc", "_physician", "_nurse", "_staff", "assigned_", "primary_"]

    for table_name, table_info in tables_data.items():
        cols = table_info.get('columns', [])
        for col in cols:
            col_lower = col.lower()
            if any(p in col_lower for p in semantic_patterns):
                # Get sample values
                data = table_info.get('data', [])
                columns = table_info.get('columns', [])
                col_idx = columns.index(col) if col in columns else -1

                samples = []
                if col_idx >= 0:
                    for row in data[:10]:
                        if col_idx < len(row) and row[col_idx]:
                            samples.append(str(row[col_idx]))

                print(f"  {table_name}.{col}: {samples[:5]}")
    print()

    # Initialize SemanticFKDetector with LLM
    print("[3] Initializing SemanticFKDetector with LLM enhancement...")
    from src.unified_pipeline.autonomous.analysis.semantic_fk_detector import SemanticFKDetector

    detector = SemanticFKDetector(
        tables_data=tables_data,
        enable_llm=True,
        llm_min_confidence=0.6,
        domain_context="healthcare",
    )

    print(f"  LLM enabled: {detector.enable_llm}")
    print(f"  LLM enhancer available: {detector._llm_enhancer is not None}")
    print()

    # Run analysis
    print("[4] Running semantic FK analysis...")
    results = detector.analyze()
    print()

    # Show results
    print("[5] RESULTS")
    print("-" * 60)

    # Semantic candidates
    print(f"\n  Semantic Candidates: {len(results['semantic_candidates'])}")
    for cand in results['semantic_candidates'][:10]:
        print(f"    {cand.source_table}.{cand.source_column} → {cand.target_table}.{cand.target_column} (conf: {cand.confidence:.2f})")

    # LLM enhanced FKs
    print(f"\n  LLM Enhanced FKs: {len(results['llm_enhanced_fks'])}")
    for fk in results['llm_enhanced_fks']:
        print(f"    ✅ {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}")
        print(f"       Confidence: {fk['confidence']:.2f}")
        print(f"       Semantic: {fk.get('semantic_relationship', 'N/A')}")
        print(f"       Reasoning: {fk.get('reasoning', 'N/A')[:100]}...")

    # Check if missed FKs are now detected
    print("\n" + "=" * 60)
    print("CHECK: Previously Missed FKs")
    print("=" * 60)

    missed_fks = [
        ("diagnoses", "diagnosed_by", "doctors", "doctor_id"),
        ("prescriptions", "prescribing_doc", "doctors", "doctor_id"),
        ("lab_results", "ordering_physician", "doctors", "doctor_id"),
        ("medical_records", "attending_doc", "doctors", "doctor_id"),
    ]

    detected_count = 0
    for src_table, src_col, tgt_table, tgt_col in missed_fks:
        # Check in LLM enhanced
        found = any(
            fk['source_table'] == src_table and fk['source_column'] == src_col
            for fk in results['llm_enhanced_fks']
        )

        # Also check in semantic candidates
        if not found:
            found = any(
                c.source_table == src_table and c.source_column == src_col
                for c in results['semantic_candidates']
                if c.confidence >= 0.6
            )

        status = "✅ DETECTED" if found else "❌ MISSED"
        if found:
            detected_count += 1
        print(f"  {src_table}.{src_col} → {tgt_table}.{tgt_col}: {status}")

    print(f"\n  Detection rate: {detected_count}/{len(missed_fks)} ({detected_count/len(missed_fks)*100:.0f}%)")
    print()


if __name__ == "__main__":
    test_llm_detection()
