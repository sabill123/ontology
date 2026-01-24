#!/usr/bin/env python3
"""
LLM Semantic Enhancer Test

Healthcare_silo에서 놓친 FK들에 대해 LLM Enhancer 테스트
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test cases: Healthcare missed FKs
MISSED_FKS = [
    {
        "fk_table": "diagnoses",
        "fk_column": "diagnosed_by",
        "pk_table": "doctors",
        "pk_column": "doctor_id",
        "description": "진단한 의사 (semantic role: verb + by)",
    },
    {
        "fk_table": "prescriptions",
        "fk_column": "prescribing_doc",
        "pk_table": "doctors",
        "pk_column": "doctor_id",
        "description": "처방한 의사 (semantic role: verb + doc)",
    },
    {
        "fk_table": "medical_records",
        "fk_column": "attending_doc",
        "pk_table": "doctors",
        "pk_column": "doctor_id",
        "description": "담당 의사 (semantic role: role + doc)",
    },
    {
        "fk_table": "lab_results",
        "fk_column": "ordering_physician",
        "pk_table": "doctors",
        "pk_column": "doctor_id",
        "description": "오더한 의사 (semantic role: verb + physician)",
    },
]


def load_healthcare_samples():
    """Load sample data from healthcare_silo"""
    import pandas as pd

    data_dir = PROJECT_ROOT / 'data' / 'healthcare_silo'
    samples = {}

    for csv_file in data_dir.glob('*.csv'):
        table_name = csv_file.stem
        df = pd.read_csv(csv_file)
        samples[table_name] = {}

        for col in df.columns:
            samples[table_name][col] = df[col].dropna().head(15).tolist()

    return samples


def test_llm_enhancer():
    """Test LLM Semantic Enhancer on missed FKs"""
    print('='*70)
    print('LLM SEMANTIC ENHANCER TEST')
    print('Testing on healthcare_silo missed FKs')
    print('='*70)
    print()

    # Import enhancer
    try:
        from src.unified_pipeline.autonomous.analysis.llm_semantic_enhancer import (
            LLMSemanticEnhancer,
            analyze_semantic_fk,
        )
        print('[OK] LLM Semantic Enhancer imported')
    except ImportError as e:
        print(f'[FAIL] Import error: {e}')
        return

    # Load sample data
    print('\nLoading healthcare_silo samples...')
    samples = load_healthcare_samples()
    print(f'Loaded {len(samples)} tables')

    # Initialize enhancer
    print('\nInitializing LLM Enhancer...')
    try:
        enhancer = LLMSemanticEnhancer(
            use_instructor=True,
            use_dspy=False,
            min_confidence=0.5,
        )
        print('[OK] Enhancer initialized')
    except Exception as e:
        print(f'[FAIL] Enhancer init failed: {e}')
        return

    # Test each missed FK
    print('\n' + '-'*70)
    print('TESTING MISSED FKs')
    print('-'*70)

    results = []

    for i, fk_case in enumerate(MISSED_FKS, 1):
        fk_table = fk_case["fk_table"]
        fk_column = fk_case["fk_column"]
        pk_table = fk_case["pk_table"]
        pk_column = fk_case["pk_column"]
        description = fk_case["description"]

        print(f'\n[{i}] {fk_table}.{fk_column} → {pk_table}.{pk_column}')
        print(f'    Description: {description}')

        # Get samples
        fk_samples = samples.get(fk_table, {}).get(fk_column, [])
        pk_samples = samples.get(pk_table, {}).get(pk_column, [])

        print(f'    FK samples: {fk_samples[:5]}')
        print(f'    PK samples: {pk_samples[:5]}')

        # Calculate overlap
        fk_set = set(str(v) for v in fk_samples if v)
        pk_set = set(str(v) for v in pk_samples if v)
        overlap = len(fk_set & pk_set) / len(fk_set) if fk_set else 0.0
        print(f'    Overlap: {overlap:.2%}')

        # Run LLM analysis
        try:
            result = enhancer.analyze_semantic_fk(
                fk_column=fk_column,
                fk_table=fk_table,
                pk_column=pk_column,
                pk_table=pk_table,
                fk_samples=[str(v) for v in fk_samples],
                pk_samples=[str(v) for v in pk_samples],
                overlap_ratio=overlap,
                domain_context="healthcare",
            )

            print(f'    Result: is_relationship={result.is_relationship}, '
                  f'confidence={result.confidence:.2f}')
            print(f'    Reasoning: {result.reasoning[:100]}...' if len(result.reasoning) > 100 else f'    Reasoning: {result.reasoning}')

            results.append({
                "case": f"{fk_table}.{fk_column} → {pk_table}.{pk_column}",
                "is_relationship": result.is_relationship,
                "confidence": result.confidence,
                "success": result.is_relationship and result.confidence >= 0.5,
            })

        except Exception as e:
            print(f'    [ERROR] Analysis failed: {e}')
            results.append({
                "case": f"{fk_table}.{fk_column} → {pk_table}.{pk_column}",
                "is_relationship": False,
                "confidence": 0.0,
                "success": False,
                "error": str(e),
            })

    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)

    success_count = sum(1 for r in results if r.get("success", False))
    total = len(results)

    print(f'\nTotal: {total} FK cases')
    print(f'Success (detected by LLM): {success_count}')
    print(f'Failed: {total - success_count}')
    print(f'Success Rate: {success_count/total*100:.1f}%')

    print('\nDetails:')
    for r in results:
        status = "✓" if r.get("success") else "✗"
        print(f'  {status} {r["case"]} (conf: {r["confidence"]:.2f})')

    return results


def test_should_use_llm():
    """Test the should_use_llm decision logic"""
    print('\n' + '='*70)
    print('TESTING should_use_llm DECISION LOGIC')
    print('='*70)

    from src.unified_pipeline.autonomous.analysis.llm_semantic_enhancer import LLMSemanticEnhancer

    enhancer = LLMSemanticEnhancer()

    test_cases = [
        # (column_name, rule_confidence, expected_use_llm)
        ("customer_id", 0.9, False),  # High confidence - no LLM
        ("diagnosed_by", 0.3, True),  # Low confidence + semantic pattern
        ("prescribing_doc", 0.4, True),  # Low confidence + _doc pattern
        ("ordering_physician", 0.2, True),  # Low confidence + _physician
        ("created_at", 0.5, False),  # Medium confidence, no pattern
        ("assigned_airline", 0.3, True),  # Low confidence + assigned_ pattern
        ("carrier", 0.4, True),  # Low confidence + carrier pattern
        ("primary_doctor_id", 0.8, False),  # High confidence
        ("member_id", 0.4, True),  # Low confidence + member pattern
    ]

    print('\nColumn Name          | Rule Conf | Expected | Actual | Match')
    print('-'*70)

    for col, rule_conf, expected in test_cases:
        actual = enhancer.should_use_llm(col, rule_conf)
        match = "✓" if actual == expected else "✗"
        print(f'{col:20} | {rule_conf:9.2f} | {str(expected):8} | {str(actual):6} | {match}')


if __name__ == '__main__':
    # Test decision logic first
    test_should_use_llm()

    # Then test actual LLM analysis
    print('\n')
    results = test_llm_enhancer()
