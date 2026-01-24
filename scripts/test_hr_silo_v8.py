#!/usr/bin/env python
"""
HR Silo Test Script for v8.0 (LLM-First Discovery)

Tests the new LLM-First Discovery architecture with HR data:
- Self-Reference FK Detection: manager_id → emp_id
- Single Noun FK: trainer → employees
- Hierarchical: parent_dept → departments
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.unified_pipeline.autonomous.analysis.semantic_fk_detector import SemanticFKDetector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_hr_silo_data(data_dir: Path) -> dict:
    """HR Silo CSV 파일들을 로드"""
    tables_data = {}

    csv_files = list(data_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        if csv_file.name == "ANSWER_KEY.md":
            continue

        table_name = csv_file.stem
        df = pd.read_csv(csv_file)

        # Convert to standard format
        columns = df.columns.tolist()
        data = df.values.tolist()
        sample_data = df.head(10).to_dict('records')

        tables_data[table_name] = {
            'columns': columns,
            'data': data,
            'sample_data': sample_data,
        }

        logger.info(f"  Loaded {table_name}: {len(df)} rows, {len(columns)} columns")
        logger.info(f"    Columns: {columns}")

    return tables_data


def main():
    logger.info("=" * 60)
    logger.info("HR SILO TEST - v8.0 LLM-First Discovery")
    logger.info("=" * 60)

    # 데이터 로드
    data_dir = project_root / "data" / "hr_silo"
    tables_data = load_hr_silo_data(data_dir)

    logger.info(f"\nLoaded {len(tables_data)} tables: {list(tables_data.keys())}")

    # SemanticFKDetector v8.0 실행
    logger.info("\n" + "=" * 60)
    logger.info("Running SemanticFKDetector v8.0 (LLM-First Discovery)")
    logger.info("=" * 60)

    detector = SemanticFKDetector(
        tables_data=tables_data,
        enable_llm=True,
        llm_min_confidence=0.6,
        domain_context="hr",  # HR 도메인 컨텍스트
    )

    results = detector.analyze()

    # 결과 출력
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    # v8.0 LLM-First Discovery Results
    llm_discovered = results.get('llm_discovered_fks', [])
    logger.info(f"\n[v8.0] LLM-First Discovery: {len(llm_discovered)} FKs")

    self_ref_fks = [fk for fk in llm_discovered if fk.get('fk_type') == 'self_reference']
    single_noun_fks = [fk for fk in llm_discovered if fk.get('fk_type') == 'single_noun']
    hierarchical_fks = [fk for fk in llm_discovered if fk.get('fk_type') == 'hierarchical']

    if self_ref_fks:
        logger.info(f"\n  Self-Reference FKs ({len(self_ref_fks)}):")
        for fk in self_ref_fks:
            logger.info(f"    ✓ {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}")
            logger.info(f"      Confidence: {fk['confidence']:.2f}, Reasoning: {fk['reasoning'][:60]}...")

    if single_noun_fks:
        logger.info(f"\n  Single Noun FKs ({len(single_noun_fks)}):")
        for fk in single_noun_fks:
            logger.info(f"    ✓ {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}")
            logger.info(f"      Confidence: {fk['confidence']:.2f}, Reasoning: {fk['reasoning'][:60]}...")

    if hierarchical_fks:
        logger.info(f"\n  Hierarchical FKs ({len(hierarchical_fks)}):")
        for fk in hierarchical_fks:
            logger.info(f"    ✓ {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}")
            logger.info(f"      Confidence: {fk['confidence']:.2f}, Reasoning: {fk['reasoning'][:60]}...")

    # Hierarchy Tables
    hierarchy_tables = results.get('hierarchy_tables', {})
    if hierarchy_tables:
        logger.info(f"\n[v8.0] Hierarchical Tables Detected: {len(hierarchy_tables)}")
        for table_name, info in hierarchy_tables.items():
            logger.info(f"  {table_name}:")
            logger.info(f"    PK: {info.get('pk_column')}")
            logger.info(f"    Parent Col: {info.get('parent_column')}")

    # LLM Enhanced FKs (v7.3)
    llm_enhanced = results.get('llm_enhanced_fks', [])
    logger.info(f"\n[v7.3] LLM Semantic Enhancer: {len(llm_enhanced)} FKs")
    for fk in llm_enhanced[:5]:
        logger.info(f"  → {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}")

    # Verified FKs (v7.4)
    verified = results.get('verified_fks', [])
    logger.info(f"\n[v7.4] Verified FKs: {len(verified)}")

    corrections = [fk for fk in verified if fk.get('was_corrected')]
    if corrections:
        logger.info(f"  Corrections made: {len(corrections)}")
        for fk in corrections:
            logger.info(f"    ✓ {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}")
            logger.info(f"      Reason: {fk.get('correction_reason', 'N/A')[:60]}...")

    # Mandatory FKs
    mandatory = results.get('mandatory_fks', [])
    logger.info(f"\n[Total] Mandatory FKs: {len(mandatory)}")

    # Print all mandatory FKs organized by type
    by_type = {}
    for fk in mandatory:
        fk_type = fk.get('type', 'unknown')
        if fk_type not in by_type:
            by_type[fk_type] = []
        by_type[fk_type].append(fk)

    for fk_type, fks in by_type.items():
        logger.info(f"\n  [{fk_type}] ({len(fks)} FKs):")
        for fk in fks[:10]:  # Show up to 10
            logger.info(f"    {fk['source']} → {fk['target']} ({fk['priority']})")

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
