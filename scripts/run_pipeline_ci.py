"""
CI Pipeline Runner — GitHub Actions에서 파이프라인 실행용

Usage:
    python scripts/run_pipeline_ci.py --data data/sample_csv/patients.csv --domain healthcare
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ci_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ontoloty pipeline for CI")
    parser.add_argument("--data", required=True, help="CSV 파일 경로")
    parser.add_argument("--domain", default="", help="도메인 힌트")
    parser.add_argument("--name", default="ci_run", help="시나리오 이름")
    return parser.parse_args()


def run_pipeline(data_path: str, domain_hint: str, scenario_name: str):
    from src.unified_pipeline.unified_main import OntologyPlatform, PipelineConfig

    output_dir = str(PROJECT_ROOT / "data" / "output")
    os.makedirs(output_dir, exist_ok=True)

    config = PipelineConfig(
        phases=["discovery", "refinement", "governance"],
        max_concurrent_agents=2,
        output_dir=output_dir,
        use_llm=True,
    )

    platform = OntologyPlatform(config=config)

    logger.info(f"=== Pipeline Start ===")
    logger.info(f"  data: {data_path}")
    logger.info(f"  domain: {domain_hint or '(auto-detect)'}")
    logger.info(f"  scenario: {scenario_name}")

    start_time = time.time()

    result = platform.run(
        data_source=data_path,
        scenario_name=scenario_name,
        domain_hint=domain_hint or None,
    )

    elapsed = time.time() - start_time

    logger.info(f"=== Pipeline Done ({elapsed:.1f}s) ===")
    logger.info(f"  success: {result.success}")
    logger.info(f"  domain: {result.domain_detected} ({result.domain_confidence:.2f})")
    logger.info(f"  tables: {result.tables_processed}")
    logger.info(f"  entities: {result.entities_found}")
    logger.info(f"  relationships: {result.relationships_found}")
    logger.info(f"  insights: {result.insights_generated}")
    logger.info(f"  governance: {result.governance_actions}")

    if result.errors:
        logger.warning(f"  errors: {result.errors}")

    # 결과 저장
    ci_prefix = f"ci_{scenario_name}"

    # 1) context.json
    context_path = os.path.join(output_dir, f"{ci_prefix}_context.json")
    if result.context:
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(result.context, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Context saved: {context_path} ({os.path.getsize(context_path):,} bytes)")

    # 2) summary.md — GitHub Step Summary로도 사용
    summary_path = os.path.join(output_dir, "ci_result_summary.md")
    summary = _build_summary(result, elapsed, data_path, domain_hint)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Summary saved: {summary_path}")

    # 콘솔에도 출력
    print("\n" + summary)

    return result


def _build_summary(result, elapsed: float, data_path: str, domain_hint: str) -> str:
    """결과 요약 마크다운 생성"""
    status = "SUCCESS" if result.success else "FAIL"
    lines = [
        f"### Pipeline Result: **{status}**",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Data | `{data_path}` |",
        f"| Domain | {result.domain_detected} ({result.domain_confidence:.2f}) |",
        f"| Tables | {result.tables_processed} |",
        f"| Entities | {result.entities_found} |",
        f"| Relationships | {result.relationships_found} |",
        f"| Insights | {result.insights_generated} |",
        f"| Governance Actions | {result.governance_actions} |",
        f"| Execution Time | {elapsed:.1f}s |",
    ]

    if result.errors:
        lines.append("")
        lines.append("#### Errors")
        for err in result.errors:
            lines.append(f"- {err}")

    # context에서 주요 정보 추출
    ctx = result.context or {}

    # 엔티티 목록
    entities = ctx.get("unified_entities") or ctx.get("entities", [])
    if entities:
        lines.append("")
        lines.append("#### Entities")
        for e in entities[:20]:
            if isinstance(e, dict):
                name = e.get("canonical_name") or e.get("name", "?")
                tables = e.get("source_tables", [])
                conf = e.get("confidence", 0)
                lines.append(f"- **{name}** (tables: {tables}, confidence: {conf:.2f})")

    # FK 관계
    fk_rels = ctx.get("foreign_key_relationships") or ctx.get("fk_relationships", [])
    if fk_rels:
        lines.append("")
        lines.append("#### Foreign Key Relationships")
        for fk in fk_rels[:20]:
            if isinstance(fk, dict):
                src = f"{fk.get('source_table','?')}.{fk.get('source_column','?')}"
                tgt = f"{fk.get('target_table','?')}.{fk.get('target_column','?')}"
                conf = fk.get("confidence", 0)
                lines.append(f"- {src} → {tgt} (confidence: {conf:.2f})")

    # Governance Decisions
    gov_decisions = ctx.get("governance_decisions", [])
    if gov_decisions:
        lines.append("")
        lines.append("#### Governance Decisions")
        for gd in gov_decisions[:10]:
            if isinstance(gd, dict):
                topic = gd.get("topic") or gd.get("decision_topic", "?")
                decision = gd.get("decision", "?")
                lines.append(f"- **{topic}**: {decision}")

    # Precomputed Stats
    stats = ctx.get("precomputed_statistics", {})
    if stats:
        lines.append("")
        lines.append("#### Precomputed Statistics (pandas, v22.1)")
        for table_name, table_stats in stats.items():
            if isinstance(table_stats, dict):
                total_rows = table_stats.get("total_rows", "?")
                lines.append(f"- **{table_name}**: {total_rows} rows")
                numeric = table_stats.get("numeric_stats", {})
                for col, col_stats in list(numeric.items())[:5]:
                    if isinstance(col_stats, dict):
                        mean = col_stats.get("mean", "?")
                        total = col_stats.get("sum", "?")
                        lines.append(f"  - {col}: mean={mean}, sum={total}")

    return "\n".join(lines)


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args.data, args.domain, args.name)
    sys.exit(0 if result.success else 1)
