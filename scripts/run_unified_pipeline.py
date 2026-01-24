#!/usr/bin/env python
"""
Unified Multi-Agent Pipeline Runner

통합 멀티에이전트 파이프라인 실행 스크립트

사용법:
    python scripts/run_unified_pipeline.py --scenario healthcare_integrated --use-llm
    python scripts/run_unified_pipeline.py --scenario healthcare_integrated --no-llm
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.unified_pipeline import UnifiedPipelineOrchestrator
from src.unified_pipeline.shared_context import StageType

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            project_root / "data" / "logs" / f"unified_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Unified Multi-Agent Pipeline")

    parser.add_argument(
        "--scenario",
        type=str,
        default="healthcare_integrated",
        help="Scenario name (default: healthcare_integrated)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="healthcare",
        help="Domain context (default: healthcare)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use LLM for agent analysis (default: True)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM (rule-based only)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        choices=["discovery", "refinement", "governance"],
        help="Stages to run (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: scenario_dir/unified_output)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # 로깅 레벨 조정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 경로 설정
    scenario_dir = project_root / "data" / "scenarios" / args.scenario
    if not scenario_dir.exists():
        logger.error(f"Scenario directory not found: {scenario_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else scenario_dir / "unified_output"

    # LLM 설정
    use_llm = not args.no_llm if args.no_llm else args.use_llm

    # 스테이지 설정
    stages_to_run = None
    if args.stages:
        stage_map = {
            "discovery": StageType.DISCOVERY,
            "refinement": StageType.REFINEMENT,
            "governance": StageType.GOVERNANCE,
        }
        stages_to_run = [stage_map[s] for s in args.stages]

    logger.info("=" * 60)
    logger.info("UNIFIED MULTI-AGENT PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Use LLM: {use_llm}")
    logger.info(f"Stages: {args.stages if args.stages else 'all'}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # 오케스트레이터 초기화
    orchestrator = UnifiedPipelineOrchestrator(
        scenario_name=args.scenario,
        domain_context=args.domain,
        use_llm=use_llm,
        output_dir=output_dir,
    )

    # 테이블 로드
    logger.info("\nLoading tables...")
    tables_data = orchestrator.load_tables(scenario_dir)
    logger.info(f"Loaded {len(tables_data)} tables")

    # 파이프라인 실행
    logger.info("\nRunning pipeline...")
    event_count = 0

    for event in orchestrator.run(tables_data, stages_to_run=stages_to_run):
        event_type = event.get("type", "unknown")
        event_count += 1

        if event_type == "stage_start":
            logger.info(f"\n{'='*40}")
            logger.info(f"STAGE: {event.get('stage', 'unknown').upper()}")
            logger.info(f"{'='*40}")

        elif event_type == "step_start":
            step = event.get("step", "unknown")
            desc = event.get("description", "")
            logger.info(f"  → {step}: {desc}")

        elif event_type == "step_complete":
            step = event.get("step", "unknown")
            results = event.get("results", "")
            logger.info(f"  ✓ {step}: {results}")

        elif event_type == "stage_complete":
            stage = event.get("stage", "unknown")
            exec_time = event.get("execution_time", 0)
            success = event.get("success", False)
            status = "✓" if success else "✗"
            logger.info(f"\n{status} Stage {stage} complete in {exec_time:.2f}s")
            if event.get("summary"):
                for k, v in event["summary"].items():
                    logger.info(f"    {k}: {v}")

        elif event_type == "analyzing_pair":
            if args.verbose:
                pair = event.get("pair", [])
                progress = event.get("progress", 0)
                logger.debug(f"    Analyzing: {pair[0]} ↔ {pair[1]} ({progress:.1%})")

        elif event_type == "pipeline_complete":
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)

        elif event_type == "orchestrator_error":
            logger.error(f"Error: {event.get('error')}")

    # 결과 저장
    logger.info("\nSaving results...")
    saved_files = orchestrator.save_results()
    for name, path in saved_files.items():
        logger.info(f"  Saved {name}: {path}")

    # 요약 출력
    orchestrator.print_summary()

    # 통계 반환
    stats = orchestrator.get_statistics()
    logger.info(f"\nTotal events processed: {event_count}")
    logger.info(f"Total execution time: {stats.get('execution_time', 0):.2f} seconds")

    return orchestrator.context


if __name__ == "__main__":
    context = main()
