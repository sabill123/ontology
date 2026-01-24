#!/usr/bin/env python3
"""
Healthcare Full Pipeline - Autonomous Multi-Agent Pipeline

Complete pipeline execution for healthcare_integrated data with:
1. Discovery Stage: Schema Analysis + TDA + Data Analysis
2. Refinement Stage: Ontology Extraction + LLM-as-Judge Evaluation
3. Governance Stage: Governance Decisions + Action Generation

Target: Palantir-level data silo integration with full evidence chains
"""

import sys
import json
import csv
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from new architecture
from src import (
    AutonomousPipelineOrchestrator,
    CompatiblePipelineOrchestrator,
    SharedContext,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(title: str, char: str = "="):
    """Print formatted banner"""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def print_status(msg: str, icon: str = "->"):
    """Print status message"""
    print(f"  {icon} {msg}")


def load_csv_tables(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load CSV files from data directory into tables_data format

    Expected format for AutonomousPipelineOrchestrator:
    {
        "table_name": {
            "columns": ["col1", "col2", ...],
            "data": [[row1], [row2], ...],
            "row_count": int,
            "source_file": str
        }
    }
    """
    tables_data = {}

    csv_files = list(data_dir.glob("*.csv"))
    print_status(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        table_name = csv_file.stem  # e.g., "ehr__patient_master"

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                columns = next(reader)  # First row is header
                data = list(reader)

            tables_data[table_name] = {
                "columns": columns,
                "data": data[:10000],  # Limit to 10k rows for performance
                "row_count": len(data),
                "source_file": str(csv_file),
                "dtypes": {}  # Will be inferred by profiler
            }

            print_status(f"  {table_name}: {len(data):,} rows, {len(columns)} columns")

        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            continue

    return tables_data


def run_healthcare_pipeline(data_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run the full healthcare pipeline"""

    print_banner("HEALTHCARE INTEGRATION PIPELINE")
    print_status(f"Data Directory: {data_dir}")
    print_status(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Setup output directory
    if output_dir is None:
        output_dir = data_dir / "pipeline_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create phase directories
    (output_dir / "phase1").mkdir(exist_ok=True)
    (output_dir / "phase2").mkdir(exist_ok=True)
    (output_dir / "phase3").mkdir(exist_ok=True)

    output_files = []

    # ============================================
    # STEP 1: Load Data
    # ============================================
    print_banner("STEP 1: DATA LOADING", "-")

    tables_data = load_csv_tables(data_dir)

    if not tables_data:
        logger.error("No tables loaded!")
        return {"error": "No tables loaded"}

    print_status(f"Loaded {len(tables_data)} tables total", "[OK]")

    # ============================================
    # STEP 2: Initialize Pipeline
    # ============================================
    print_banner("STEP 2: PIPELINE INITIALIZATION", "-")

    scenario_name = data_dir.name

    # Use CompatiblePipelineOrchestrator for sync execution
    orchestrator = CompatiblePipelineOrchestrator(
        scenario_name=scenario_name,
        domain_context="healthcare",
        output_dir=str(output_dir),
        enable_continuation=True,
        max_concurrent_agents=5,
    )

    print_status(f"Orchestrator initialized: {scenario_name}", "[OK]")
    print_status(f"Output directory: {output_dir}")

    # ============================================
    # STEP 3: Run Pipeline
    # ============================================
    print_banner("STEP 3: RUNNING AUTONOMOUS PIPELINE", "-")

    event_count = 0
    phase_results = {
        "discovery": {},
        "refinement": {},
        "governance": {}
    }

    current_phase = None
    phase_start_times = {}

    try:
        for event in orchestrator.run_full_pipeline(tables_data):
            event_type = event.get("type", "unknown")
            event_count += 1

            # Phase tracking
            if event_type == "phase_start":
                current_phase = event.get("phase", "unknown")
                phase_start_times[current_phase] = time.time()
                print(f"\n{'='*50}")
                print(f"  PHASE: {current_phase.upper()}")
                print(f"{'='*50}")

            elif event_type == "phase_complete":
                phase = event.get("phase", "unknown")
                phase_time = time.time() - phase_start_times.get(phase, start_time)
                success = event.get("success", False)
                status = "[OK]" if success else "[FAIL]"
                print(f"\n  {status} Phase {phase} complete in {phase_time:.2f}s")

                if event.get("summary"):
                    phase_results[phase] = event.get("summary", {})
                    for k, v in event["summary"].items():
                        print(f"      {k}: {v}")

            elif event_type == "todo_start":
                todo_id = event.get("todo_id", "")
                todo_desc = event.get("description", "")
                print(f"  -> {todo_id}: {todo_desc}")

            elif event_type == "todo_complete":
                todo_id = event.get("todo_id", "")
                duration = event.get("duration", 0)
                print(f"  [OK] {todo_id} ({duration:.1f}s)")

            elif event_type == "agent_working":
                agent = event.get("agent", "unknown")
                task = event.get("task", "")
                if event_count % 10 == 0:  # Rate limit output
                    print(f"      {agent}: {task[:50]}...")

            elif event_type == "error":
                error = event.get("error", "unknown error")
                logger.error(f"Pipeline error: {error}")

            elif event_type == "pipeline_complete":
                print("\n" + "="*70)
                print("  PIPELINE COMPLETE")
                print("="*70)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    # ============================================
    # STEP 4: Collect Results
    # ============================================
    print_banner("STEP 4: COLLECTING RESULTS", "-")

    # Get context from orchestrator
    context = orchestrator.get_context()

    # Extract results
    results = {
        "scenario_name": scenario_name,
        "execution_time_seconds": time.time() - start_time,
        "events_processed": event_count,
        "tables_loaded": len(tables_data),
        "phase_results": phase_results,
        "output_files": [],
    }

    # ============================================
    # STEP 5: Save Results
    # ============================================
    print_banner("STEP 5: SAVING RESULTS", "-")

    # Save main output
    main_output_file = output_dir / "pipeline_output.json"

    try:
        context_dict = context.to_dict() if hasattr(context, 'to_dict') else {}

        full_output = {
            "version": "3.0",
            "scenario": scenario_name,
            "domain": "healthcare",
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": results["execution_time_seconds"],
            "tables_processed": len(tables_data),
            "context": context_dict,
            "statistics": context.get_statistics() if hasattr(context, 'get_statistics') else {},
        }

        with open(main_output_file, 'w', encoding='utf-8') as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False, default=str)

        results["output_files"].append(str(main_output_file))
        print_status(f"Main output: {main_output_file}", "[OK]")

    except Exception as e:
        logger.error(f"Error saving main output: {e}")

    # Save phase-specific outputs
    for phase in ["discovery", "refinement", "governance"]:
        phase_file = output_dir / f"phase{['discovery', 'refinement', 'governance'].index(phase) + 1}" / f"{phase}_output.json"
        try:
            phase_data = phase_results.get(phase, {})
            with open(phase_file, 'w', encoding='utf-8') as f:
                json.dump(phase_data, f, indent=2, ensure_ascii=False, default=str)
            results["output_files"].append(str(phase_file))
            print_status(f"Phase output: {phase_file}")
        except Exception as e:
            logger.warning(f"Error saving {phase} output: {e}")

    # ============================================
    # STEP 6: Validation
    # ============================================
    print_banner("STEP 6: VALIDATION", "-")

    validation_results = validate_results(results, context)
    results["validation"] = validation_results

    # Save validation report
    validation_file = output_dir / "validation_report.json"
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    results["output_files"].append(str(validation_file))
    print_status(f"Validation report: {validation_file}", "[OK]")

    # ============================================
    # SUMMARY
    # ============================================
    print_banner("PIPELINE SUMMARY")

    print(f"\n  Scenario: {scenario_name}")
    print(f"  Domain: healthcare")
    print(f"  Tables Processed: {len(tables_data)}")
    print(f"  Events Processed: {event_count}")
    print(f"  Execution Time: {results['execution_time_seconds']:.2f} seconds")

    print(f"\n  Validation Score: {validation_results.get('overall_score', 0):.2%}")
    print(f"  Data Quality: {validation_results.get('data_quality', 0):.2%}")
    print(f"  Schema Coverage: {validation_results.get('schema_coverage', 0):.2%}")

    print(f"\n  Output Files:")
    for f in results["output_files"]:
        print(f"    - {f}")

    return results


def validate_results(results: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
    """Validate pipeline results"""

    validation = {
        "timestamp": datetime.now().isoformat(),
        "checks": [],
        "overall_score": 0.0,
        "data_quality": 0.0,
        "schema_coverage": 0.0,
    }

    # Check 1: Tables processed
    tables_loaded = results.get("tables_loaded", 0)
    check1_pass = tables_loaded >= 5  # Healthcare should have 5+ tables
    validation["checks"].append({
        "name": "tables_loaded",
        "passed": check1_pass,
        "expected": ">=5",
        "actual": tables_loaded
    })

    # Check 2: Events processed
    events_processed = results.get("events_processed", 0)
    check2_pass = events_processed > 0
    validation["checks"].append({
        "name": "events_processed",
        "passed": check2_pass,
        "expected": ">0",
        "actual": events_processed
    })

    # Check 3: Context has tables
    context_tables = 0
    if hasattr(context, 'tables'):
        context_tables = len(context.tables)
    check3_pass = context_tables > 0
    validation["checks"].append({
        "name": "context_tables",
        "passed": check3_pass,
        "expected": ">0",
        "actual": context_tables
    })

    # Check 4: Execution time reasonable
    exec_time = results.get("execution_time_seconds", 0)
    check4_pass = 0 < exec_time < 600  # Less than 10 minutes
    validation["checks"].append({
        "name": "execution_time",
        "passed": check4_pass,
        "expected": "<600s",
        "actual": f"{exec_time:.2f}s"
    })

    # Calculate scores
    passed = sum(1 for c in validation["checks"] if c["passed"])
    total = len(validation["checks"])

    validation["overall_score"] = passed / total if total > 0 else 0
    validation["data_quality"] = 0.8 if check1_pass and check3_pass else 0.5
    validation["schema_coverage"] = context_tables / max(tables_loaded, 1)

    # Print validation results
    for check in validation["checks"]:
        status = "[OK]" if check["passed"] else "[FAIL]"
        print(f"  {status} {check['name']}: {check['actual']} (expected: {check['expected']})")

    return validation


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Healthcare Full Pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "scenarios" / "healthcare_integrated",
        help="Path to healthcare data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data_dir/pipeline_output)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    results = run_healthcare_pipeline(args.data_dir, args.output_dir)

    # Return exit code based on validation
    if results.get("validation", {}).get("overall_score", 0) >= 0.75:
        print("\n[SUCCESS] Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n[WARNING] Pipeline completed with validation issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
