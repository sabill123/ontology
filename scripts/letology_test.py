#!/usr/bin/env python3
"""
Letology Full System Test Script

전체 시스템 테스트 (Phase 1, 2, 3 모두 실행)
- 단일 CSV 파일 또는 폴더 경로 지원
- 폴더 입력 시 하위 모든 CSV 파일 읽기
- 결과를 팔란티어 스타일로 상세 출력

Usage:
    python scripts/letology_test.py <input_path>
    python scripts/letology_test.py data/letsur_test_data/download_utf8.csv
    python scripts/letology_test.py data/letsur_test_data/
"""

import argparse
import asyncio
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class LetologyTestRunner:
    """Letology 전체 시스템 테스트 러너"""

    def __init__(self, input_path: str, output_dir: Optional[str] = None):
        self.input_path = Path(input_path).resolve()
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "test_results"
        self.start_time = None
        self.end_time = None
        self.results = {}

    def prepare_data_directory(self) -> Path:
        """
        입력 경로를 데이터 디렉토리로 준비
        - 단일 파일: 임시 디렉토리에 복사
        - 폴더: 그대로 사용
        """
        if self.input_path.is_file():
            # 단일 파일인 경우 임시 디렉토리 생성
            temp_dir = self.output_dir / "temp_data"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # 파일 복사
            dest = temp_dir / self.input_path.name
            shutil.copy2(self.input_path, dest)
            print(f"📁 Single file detected: {self.input_path.name}")
            print(f"   Copied to: {temp_dir}")
            return temp_dir

        elif self.input_path.is_dir():
            # 폴더인 경우 CSV 파일 목록 출력
            csv_files = list(self.input_path.glob("*.csv"))
            print(f"📁 Directory detected: {self.input_path}")
            print(f"   Found {len(csv_files)} CSV files:")
            for f in csv_files[:5]:
                print(f"   - {f.name}")
            if len(csv_files) > 5:
                print(f"   ... and {len(csv_files) - 5} more")
            return self.input_path

        else:
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

    def run_pipeline(self, data_dir: Path) -> Dict[str, Any]:
        """전체 파이프라인 실행 (Phase 1, 2, 3) - subprocess 방식"""
        import subprocess

        print("\n" + "=" * 70)
        print("🚀 LETOLOGY FULL SYSTEM TEST")
        print("=" * 70)
        print(f"📂 Data Source: {data_dir}")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        self.start_time = datetime.now()

        # Phase 1: Discovery
        print("\n" + "-" * 70)
        print("📊 PHASE 1: DISCOVERY")
        print("-" * 70)
        print("   - Data Understanding (LLM-First Analysis)")
        print("   - TDA Analysis (Topological Data Analysis)")
        print("   - Schema Analysis")
        print("   - Value Matching")
        print("   - Entity Classification")
        print("   - Relationship Detection")

        # Phase 2: Refinement
        print("\n" + "-" * 70)
        print("🔧 PHASE 2: REFINEMENT")
        print("-" * 70)
        print("   - Ontology Architecture")
        print("   - Cross-Entity Correlation Analysis")
        print("   - Conflict Resolution")
        print("   - Quality Assessment")
        print("   - Semantic Validation")

        # Phase 3: Governance
        print("\n" + "-" * 70)
        print("📋 PHASE 3: GOVERNANCE")
        print("-" * 70)
        print("   - Governance Strategy")
        print("   - Action Prioritization")
        print("   - Risk Assessment")
        print("   - Policy Generation")

        print("\n⏳ Running pipeline... (this may take several minutes)")
        print("-" * 70)

        # subprocess로 실행
        cmd = [
            sys.executable,
            "-m", "src.unified_pipeline.unified_main",
            str(data_dir),
        ]

        process = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
        )

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"📅 Completed: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        return {"returncode": process.returncode}

    def load_results(self) -> Dict[str, Any]:
        """결과 파일들 로드"""
        results = {}

        # 최신 로그 디렉토리 찾기
        logs_dir = PROJECT_ROOT / "Logs"
        if logs_dir.exists():
            log_dirs = sorted(logs_dir.iterdir(), key=lambda x: x.name, reverse=True)
            if log_dirs:
                latest_log = log_dirs[0]
                print(f"\n📂 Results from: {latest_log}")

                # artifacts 로드
                artifacts_dir = latest_log / "artifacts"
                if artifacts_dir.exists():
                    for artifact in artifacts_dir.glob("*.json"):
                        with open(artifact, "r", encoding="utf-8") as f:
                            results[artifact.stem] = json.load(f)

                # summary 로드
                summary_file = latest_log / "summary.json"
                if summary_file.exists():
                    with open(summary_file, "r", encoding="utf-8") as f:
                        results["summary"] = json.load(f)

        return results

    def print_palantir_style_report(self, results: Dict[str, Any]):
        """팔란티어 스타일로 결과 출력"""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "🔮 LETOLOGY ANALYSIS REPORT" + " " * 30 + "║")
        print("║" + " " * 15 + "Palantir Foundry Style Insights" + " " * 31 + "║")
        print("╚" + "═" * 78 + "╝")

        # 1. Executive Summary
        self._print_executive_summary(results)

        # 2. Data Quality Assessment
        self._print_data_quality_section(results)

        # 3. Key Insights (Segment Analysis)
        self._print_key_insights_section(results)

        # 4. Discovered Entities & Relationships
        self._print_ontology_section(results)

        # 5. Recommended Actions
        self._print_actions_section(results)

        # 6. Risk Assessment
        self._print_risk_section(results)

    def _print_executive_summary(self, results: Dict[str, Any]):
        """Executive Summary 출력"""
        print("\n")
        print("┌" + "─" * 78 + "┐")
        print("│" + " 📊 EXECUTIVE SUMMARY".ljust(78) + "│")
        print("└" + "─" * 78 + "┘")

        summary = results.get("summary", {})
        insights = results.get("predictive_insights", [])
        ontology = results.get("ontology", {})

        # 기본 통계
        print(f"""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Domain Detected    : {summary.get('domain', 'Unknown'):<52} │
  │  Execution Time     : {summary.get('duration_seconds', 0):.1f} seconds{' ' * 42}│
  │  Total Agents       : {summary.get('statistics', {}).get('total_agents', 0):<52} │
  │  LLM Calls          : {summary.get('statistics', {}).get('total_llm_calls', 0):<52} │
  │  Total Tokens       : {summary.get('statistics', {}).get('total_tokens', 0):,}{' ' * 43}│
  └─────────────────────────────────────────────────────────────────────────┘
""")

        # 인사이트 요약
        segment_insights = [i for i in insights if i.get('insight_type') == 'segment_analysis']
        quality_insights = [i for i in insights if i.get('insight_type') == 'data_quality_issue']

        print(f"  📈 Total Insights Generated: {len(insights)}")
        print(f"     - Segment Analysis: {len(segment_insights)}")
        print(f"     - Data Quality Issues: {len(quality_insights)}")

        # 엔티티 요약
        entities = ontology.get("entities", [])
        relationships = ontology.get("relationships", [])
        print(f"\n  🔗 Ontology Summary:")
        print(f"     - Entities Discovered: {len(entities)}")
        print(f"     - Relationships Found: {len(relationships)}")

    def _print_data_quality_section(self, results: Dict[str, Any]):
        """Data Quality 섹션 출력"""
        print("\n")
        print("┌" + "─" * 78 + "┐")
        print("│" + " ⚠️  DATA QUALITY ASSESSMENT".ljust(78) + "│")
        print("└" + "─" * 78 + "┘")

        insights = results.get("predictive_insights", [])
        quality_insights = [i for i in insights if i.get('insight_type') == 'data_quality_issue']

        if not quality_insights:
            print("\n  ✅ No significant data quality issues detected.")
            return

        print(f"\n  Found {len(quality_insights)} data quality issues:\n")

        for i, insight in enumerate(quality_insights[:5], 1):
            source_col = insight.get('source_column', 'Unknown')
            target_col = insight.get('target_column', '')
            condition = insight.get('source_condition', '')
            impact = insight.get('predicted_impact', '')

            # LLM 탐지 여부
            is_llm = "LLM 탐지" in condition or "[LLM 분석]" in insight.get('business_interpretation', '')
            tag = "🤖 LLM" if is_llm else "📊 STAT"

            print(f"  {tag} Issue #{i}: {source_col}")
            print(f"  ├─ Condition: {condition[:70]}{'...' if len(condition) > 70 else ''}")
            print(f"  ├─ Impact: {impact[:70]}{'...' if len(impact) > 70 else ''}")
            if target_col:
                print(f"  └─ Related Column: {target_col}")
            else:
                print(f"  └─ Severity: {insight.get('risk_assessment', 'Unknown')}")
            print()

    def _print_key_insights_section(self, results: Dict[str, Any]):
        """Key Insights (Segment Analysis) 섹션 출력"""
        print("\n")
        print("┌" + "─" * 78 + "┐")
        print("│" + " 💡 KEY BUSINESS INSIGHTS".ljust(78) + "│")
        print("└" + "─" * 78 + "┘")

        insights = results.get("predictive_insights", [])
        segment_insights = [i for i in insights if i.get('insight_type') == 'segment_analysis']

        if not segment_insights:
            print("\n  ℹ️  No segment analysis insights available.")
            return

        print(f"\n  📊 Segment Analysis Results ({len(segment_insights)} insights):\n")

        # 그룹화: metric별로 정리
        metrics_seen = set()
        for insight in segment_insights:
            metric = insight.get('target_column', '')
            if metric in metrics_seen:
                continue
            metrics_seen.add(metric)

            interpretation = insight.get('business_interpretation', '')
            impact = insight.get('predicted_impact', '')
            recommendation = insight.get('recommended_action', '')
            risk = insight.get('risk_assessment', '')
            opportunity = insight.get('opportunity_assessment', '')
            concentration = insight.get('correlation_coefficient', 0) * 100

            print(f"  ╔═══════════════════════════════════════════════════════════════════════════╗")
            print(f"  ║ 📈 Metric: {metric:<64} ║")
            print(f"  ╠═══════════════════════════════════════════════════════════════════════════╣")

            # 상위 채널 파싱
            if "상위 유입경로:" in interpretation:
                channels_part = interpretation.split("상위 유입경로:")[1].split(".")[0].strip()
                print(f"  ║ Top Channels: {channels_part:<60} ║")

            print(f"  ║ Concentration: {concentration:.1f}% ({impact.split('(')[1] if '(' in impact else ''}".ljust(75) + "║")
            print(f"  ╟───────────────────────────────────────────────────────────────────────────╢")
            print(f"  ║ 🎯 Recommendation: {recommendation[:55]}{'...' if len(recommendation) > 55 else '':<3} ║")
            print(f"  ║ ⚠️  Risk: {risk:<65} ║")
            print(f"  ║ 💰 Opportunity: {opportunity:<58} ║")
            print(f"  ╚═══════════════════════════════════════════════════════════════════════════╝")
            print()

    def _print_ontology_section(self, results: Dict[str, Any]):
        """Ontology 섹션 출력"""
        print("\n")
        print("┌" + "─" * 78 + "┐")
        print("│" + " 🔗 DISCOVERED ONTOLOGY".ljust(78) + "│")
        print("└" + "─" * 78 + "┘")

        ontology = results.get("ontology", {})
        entities = ontology.get("entities", [])
        relationships = ontology.get("relationships", [])

        print(f"\n  Entities ({len(entities)}):")
        for entity in entities[:10]:
            # entity가 dict인 경우와 string인 경우 모두 처리
            if isinstance(entity, dict):
                name = entity.get('name', entity.get('canonical_name', 'Unknown'))
                etype = entity.get('type', entity.get('entity_type', 'Unknown'))
                source = entity.get('source_tables', [])
                print(f"    • {name} ({etype})")
                if source:
                    print(f"      Source: {', '.join(source[:3])}")
            elif isinstance(entity, str):
                # 문자열 표현에서 정보 추출
                import re
                name_match = re.search(r"canonical_name='([^']*)'", entity)
                type_match = re.search(r"entity_type='([^']*)'", entity)
                source_match = re.search(r"source_tables=\[([^\]]*)\]", entity)

                name = name_match.group(1) if name_match else 'Unknown'
                etype = type_match.group(1) if type_match else 'Unknown'
                source_str = source_match.group(1) if source_match else ''

                print(f"    • {name} ({etype})")
                if source_str:
                    print(f"      Source: {source_str}")

        if relationships:
            print(f"\n  Relationships ({len(relationships)}):")
            for rel in relationships[:5]:
                print(f"    • {rel.get('source', '?')} → {rel.get('target', '?')} ({rel.get('type', '?')})")

    def _print_actions_section(self, results: Dict[str, Any]):
        """Recommended Actions 섹션 출력"""
        print("\n")
        print("┌" + "─" * 78 + "┐")
        print("│" + " 📋 RECOMMENDED ACTIONS".ljust(78) + "│")
        print("└" + "─" * 78 + "┘")

        governance = results.get("governance_decisions", {})
        actions = governance.get("action_backlog", []) if isinstance(governance, dict) else []

        if not actions:
            print("\n  ℹ️  No specific actions recommended.")
            return

        print(f"\n  Prioritized Action Items ({len(actions)}):\n")

        for i, action in enumerate(actions[:7], 1):
            title = action.get('title', action.get('action', 'Unknown'))
            priority = action.get('priority', 'Medium')
            impact = action.get('expected_impact', action.get('impact', 'Unknown'))

            priority_icon = "🔴" if priority == "High" else "🟡" if priority == "Medium" else "🟢"

            print(f"  {i}. {priority_icon} [{priority}] {title}")
            if impact and impact != 'Unknown':
                print(f"     Expected Impact: {impact[:60]}{'...' if len(str(impact)) > 60 else ''}")
            print()

    def _print_risk_section(self, results: Dict[str, Any]):
        """Risk Assessment 섹션 출력"""
        print("\n")
        print("┌" + "─" * 78 + "┐")
        print("│" + " ⚠️  RISK ASSESSMENT".ljust(78) + "│")
        print("└" + "─" * 78 + "┘")

        insights = results.get("predictive_insights", [])

        # 리스크 수집
        risks = []
        for insight in insights:
            risk = insight.get('risk_assessment', '')
            if risk and '리스크' in risk or 'risk' in risk.lower():
                risks.append({
                    'source': insight.get('source_column', 'Unknown'),
                    'risk': risk
                })

        if not risks:
            print("\n  ✅ No significant risks identified.")
            return

        print(f"\n  Identified Risks ({len(risks)}):\n")

        seen = set()
        for r in risks:
            if r['risk'] in seen:
                continue
            seen.add(r['risk'])
            print(f"  ⚠️  {r['risk']}")
            print(f"     Related: {r['source']}")
            print()

    def run(self):
        """전체 테스트 실행"""
        try:
            # 1. 데이터 준비
            data_dir = self.prepare_data_directory()

            # 2. 파이프라인 실행
            self.run_pipeline(data_dir)

            # 3. 결과 로드
            results = self.load_results()
            self.results = results

            # 4. 팔란티어 스타일 리포트 출력
            self.print_palantir_style_report(results)

            # 5. 완료 메시지
            print("\n" + "=" * 80)
            print("🎉 LETOLOGY TEST COMPLETE!")
            print("=" * 80)
            print(f"\n📂 Full results saved to: {PROJECT_ROOT / 'Logs'}")
            print(f"📊 Pipeline results saved to: {self.output_dir}")

            return results

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Letology Full System Test - Run complete pipeline on any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with single CSV file
  python scripts/letology_test.py data/letsur_test_data/download_utf8.csv

  # Test with directory (all CSV files)
  python scripts/letology_test.py data/letsur_test_data/

  # Test with custom output directory
  python scripts/letology_test.py data/mydata.csv -o ./my_results
        """
    )

    parser.add_argument(
        "input_path",
        help="Path to CSV file or directory containing CSV files"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for results (default: ./test_results)",
        default=None
    )

    args = parser.parse_args()

    # 입력 경로 검증
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ Error: Input path not found: {args.input_path}")
        sys.exit(1)

    # 테스트 실행
    runner = LetologyTestRunner(args.input_path, args.output)
    runner.run()


if __name__ == "__main__":
    main()
