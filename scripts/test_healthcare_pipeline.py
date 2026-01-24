#!/usr/bin/env python3
"""
Healthcare Silo Pipeline Test
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.unified_pipeline import AutonomousPipelineOrchestrator

# LLM
try:
    from src.common.utils.llm import get_openai_client
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False


def load_healthcare_data(data_dir):
    """Load healthcare_silo dataset"""
    datasets = {}
    for f in sorted(data_dir.glob('*.csv')):
        name = f.stem
        datasets[name] = pd.read_csv(f)
        print(f"  Loaded {name}: {len(datasets[name])} rows, {len(datasets[name].columns)} cols", flush=True)
    return datasets


def convert_to_tables_data(datasets, data_dir):
    """Convert to agent format"""
    tables_data = {}
    for name, df in datasets.items():
        records = df.head(500).to_dict(orient='records')
        columns = [{'name': col, 'dtype': str(df[col].dtype)} for col in df.columns]
        tables_data[name] = {
            'table_name': name,
            'sample_data': records,
            'columns': columns,
            'row_count': len(df),
            'column_count': len(df.columns),
            'source': str(data_dir / f'{name}.csv'),
        }
    return tables_data


def run_pipeline():
    print('='*60, flush=True)
    print('HEALTHCARE SILO FULL PIPELINE TEST', flush=True)
    print('='*60, flush=True)

    # Load data
    DATA_DIR = PROJECT_ROOT / 'data' / 'healthcare_silo'
    print('\n[1] Loading healthcare_silo dataset...', flush=True)
    datasets = load_healthcare_data(DATA_DIR)
    print(f'Total: {len(datasets)} tables', flush=True)

    # Convert
    print('\n[2] Converting to agent format...', flush=True)
    tables_data = convert_to_tables_data(datasets, DATA_DIR)

    # LLM client
    llm_client = None
    if LLM_AVAILABLE:
        try:
            llm_client = get_openai_client()
            print('\n[3] LLM client initialized', flush=True)
        except Exception as e:
            print(f'\n[3] LLM warning: {e}', flush=True)
    else:
        print('\n[3] No LLM client', flush=True)

    # Create orchestrator
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = f'{timestamp}_healthcare_silo'
    output_dir = PROJECT_ROOT / 'data' / 'storage' / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n[4] Job ID: {job_id}', flush=True)
    print(f'    Output: {output_dir}', flush=True)

    orchestrator = AutonomousPipelineOrchestrator(
        scenario_name=job_id,
        domain_context='healthcare',
        llm_client=llm_client,
        output_dir=str(output_dir),
        enable_continuation=True,
        max_concurrent_agents=5,
        auto_detect_domain=True,
    )

    # Set data directory on shared context for DataAnalyst
    orchestrator.agent_orchestrator.shared_context.data_directory = str(DATA_DIR)

    # Run pipeline (synchronous)
    print('\n[5] Running 3-Phase Pipeline (sync)...', flush=True)
    print('-'*60, flush=True)

    event_count = 0

    for event in orchestrator.run_sync(tables_data):
        event_count += 1
        event_type = event.get('event_type', 'unknown')

        if event_type == 'phase_started':
            phase = event.get('phase', '')
            print(f">> PHASE {phase} STARTED", flush=True)
        elif event_type == 'phase_completed':
            phase = event.get('phase', '')
            print(f"<< PHASE {phase} COMPLETED", flush=True)
        elif event_type == 'agent_completed':
            agent = event.get('agent_name', '')
            status = event.get('status', '')
            duration = event.get('duration_ms', 0)
            print(f"   Agent [{agent}] -> {status} ({duration:.0f}ms)", flush=True)
        elif event_type == 'pipeline_completed':
            print(f"\n  PIPELINE COMPLETED!", flush=True)
        elif event_type == 'error':
            print(f"  ERROR: {event.get('error', '')}", flush=True)

    print('-'*60, flush=True)
    print(f'Total events: {event_count}', flush=True)

    return job_id


if __name__ == '__main__':
    job_id = run_pipeline()
    print(f'\nJob completed: {job_id}', flush=True)
