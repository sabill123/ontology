#!/usr/bin/env python3
"""
Palantir Dataset Full Pipeline Analysis (v2.0 - Agent-Based)

에이전트 기반 전체 파이프라인 실행:
1. 데이터 로드 및 에이전트 포맷 변환
2. AutonomousPipelineOrchestrator를 통한 3-Phase 실행
   - Discovery: TDA, Schema, Value Matching, Entity, Relationship
   - Refinement: Ontology, Conflict Resolution, Quality, Validation
   - Governance: Strategy, Prioritization, Risk, Policy
3. 결과 수집 및 저장

Usage:
    python scripts/run_palantir_pipeline.py
    python scripts/run_palantir_pipeline.py --no-llm
    python scripts/run_palantir_pipeline.py --phases discovery refinement
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
import logging

warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Ontoloty Agent System 임포트
from src.unified_pipeline import (
    AutonomousPipelineOrchestrator,
    SharedContext,
)
from src.unified_pipeline.streaming.event_bus import (
    EventBus, DomainEvent, InsightEvents, create_event_bus
)
from src.unified_pipeline.enterprise.audit_logger import (
    AuditLogger, AuditEventType, ComplianceFramework, create_audit_logger
)

# LLM 클라이언트 (Optional)
try:
    from src.common.utils.llm import get_openai_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    get_openai_client = None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 데이터 로드 및 변환
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "palantir_test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """모든 데이터셋 로드"""
    datasets = {}

    # General 데이터셋
    general_files = ["titanic.csv", "iris.csv", "flights.csv", "nyc_taxis.csv",
                     "diamonds.csv", "restaurant_tips.csv"]
    for f in general_files:
        path = DATA_DIR / f
        if path.exists():
            datasets[f.replace(".csv", "")] = pd.read_csv(path)
            print(f"  ✓ Loaded {f}: {len(datasets[f.replace('.csv', '')])} rows")

    # Healthcare
    healthcare_files = ["hospitals.csv", "patients.csv"]
    for f in healthcare_files:
        path = DATA_DIR / "healthcare" / f
        if path.exists():
            key = f"healthcare_{f.replace('.csv', '')}"
            datasets[key] = pd.read_csv(path)
            print(f"  ✓ Loaded healthcare/{f}: {len(datasets[key])} rows")

    # Manufacturing
    mfg_files = ["predictive_maintenance.csv", "iot_sensors.csv"]
    for f in mfg_files:
        path = DATA_DIR / "manufacturing" / f
        if path.exists():
            key = f"manufacturing_{f.replace('.csv', '')}"
            datasets[key] = pd.read_csv(path)
            print(f"  ✓ Loaded manufacturing/{f}: {len(datasets[key])} rows")

    # Supply Chain
    sc_files = ["order_list.csv", "freight_rates.csv", "products_per_plant.csv",
                "plant_ports.csv", "warehouse_capacities.csv", "warehouse_costs.csv"]
    for f in sc_files:
        path = DATA_DIR / "supply_chain" / f
        if path.exists():
            key = f"supply_chain_{f.replace('.csv', '')}"
            datasets[key] = pd.read_csv(path)
            print(f"  ✓ Loaded supply_chain/{f}: {len(datasets[key])} rows")

    return datasets


def convert_to_agent_format(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    DataFrame을 에이전트 시스템이 요구하는 포맷으로 변환

    Required format:
    {
        "table_name": {
            "source": "palantir_test",
            "columns": [{"name": "col", "type": "int64", "nullable": True}, ...],
            "row_count": 1000,
            "sample_data": [{...}, ...],
            "metadata": {...},
            "primary_keys": [...],
            "foreign_keys": [...]
        }
    }
    """
    tables_data = {}

    for table_name, df in datasets.items():
        # 컬럼 정보 추출
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "nullable": bool(df[col].isnull().any()),
            }

            # 추가 메타데이터
            if df[col].dtype in ['int64', 'float64']:
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            elif df[col].dtype == 'object':
                unique_count = df[col].nunique()
                col_info["unique_count"] = unique_count
                if unique_count <= 20:
                    col_info["sample_values"] = df[col].dropna().unique()[:10].tolist()

            columns.append(col_info)

        # Primary Key 추론 (간단한 휴리스틱)
        primary_keys = []
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'id' or col_lower.endswith('_id'):
                if df[col].nunique() == len(df):
                    primary_keys.append(col)
                    break

        # Foreign Key 추론 (패턴 기반)
        foreign_keys = []
        for col in df.columns:
            col_lower = col.lower()
            # FK 패턴 감지
            if '_id' in col_lower and col not in primary_keys:
                # 참조 테이블 추론
                base_name = col_lower.replace('_id', '')
                for other_table in datasets.keys():
                    if base_name in other_table.lower():
                        foreign_keys.append({
                            "source_column": col,
                            "target_table": other_table,
                            "target_column": "id",  # 추정
                            "confidence": 0.7
                        })
                        break
            # Code 패턴
            elif '_code' in col_lower:
                base_name = col_lower.replace('_code', '')
                for other_table in datasets.keys():
                    if base_name in other_table.lower():
                        foreign_keys.append({
                            "source_column": col,
                            "target_table": other_table,
                            "target_column": col,  # 같은 컬럼명
                            "confidence": 0.6
                        })
                        break

        # 도메인 추론
        domain = "general"
        if "healthcare" in table_name:
            domain = "healthcare"
        elif "manufacturing" in table_name:
            domain = "manufacturing"
        elif "supply_chain" in table_name:
            domain = "supply_chain"

        tables_data[table_name] = {
            "source": "palantir_test",
            "domain": domain,
            "columns": columns,
            "row_count": len(df),
            "sample_data": df.head(500).to_dict('records'),
            "metadata": {
                "file_path": str(DATA_DIR / f"{table_name}.csv"),
                "loaded_at": datetime.now().isoformat(),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            },
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }

    return tables_data


# ============================================================
# 파이프라인 실행
# ============================================================

class PalantirAgentPipeline:
    """Palantir 데이터셋을 위한 에이전트 기반 파이프라인"""

    def __init__(self, use_llm: bool = True, phases: Optional[List[str]] = None):
        """
        Args:
            use_llm: LLM 사용 여부 (False면 rule-based only)
            phases: 실행할 Phase 목록 (None이면 전체)
        """
        self.use_llm = use_llm and LLM_AVAILABLE
        self.phases = phases or ["discovery", "refinement", "governance"]

        # LLM 클라이언트 초기화
        self.llm_client = None
        if self.use_llm:
            try:
                self.llm_client = get_openai_client()
                logger.info("LLM client initialized (Letsur Gateway)")
            except Exception as e:
                logger.warning(f"LLM client initialization failed: {e}")
                self.use_llm = False

        # 오케스트레이터 (데이터 로드 후 초기화)
        self.orchestrator: Optional[AutonomousPipelineOrchestrator] = None

        # 감사 로거
        self.audit = create_audit_logger(
            storage_path=str(OUTPUT_DIR / "audit_logs")
        )

        # 이벤트 버스
        self.event_bus = create_event_bus()

        # 결과 저장
        self.results = {
            "execution_time": None,
            "datasets_analyzed": 0,
            "total_rows": 0,
            "phases_completed": [],
            "agents_used": [],
            "insights": [],
            "relationships": [],
            "entities": [],
            "governance_actions": [],
        }

    def run(self) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        start_time = datetime.now()

        print("=" * 70)
        print("ONTOLOTY - Agent-Based Data Intelligence Pipeline (v2.0)")
        print("=" * 70)
        print(f"LLM Enabled: {self.use_llm}")
        print(f"Phases: {', '.join(self.phases)}")
        print("=" * 70)

        # 1. 데이터 로드
        print("\n[Phase 0] Loading Datasets...")
        datasets = load_all_datasets()
        self.results["datasets_analyzed"] = len(datasets)
        self.results["total_rows"] = sum(len(df) for df in datasets.values())

        # 2. 에이전트 포맷으로 변환
        print("\n[Phase 0.5] Converting to Agent Format...")
        tables_data = convert_to_agent_format(datasets)
        print(f"  ✓ Converted {len(tables_data)} tables with metadata")

        # 3. 오케스트레이터 초기화
        print("\n[Phase 1] Initializing Agent Orchestrator...")
        self.orchestrator = AutonomousPipelineOrchestrator(
            scenario_name="palantir_analysis",
            domain_context=None,  # 자동 탐지
            llm_client=self.llm_client,
            output_dir=str(OUTPUT_DIR),
            enable_continuation=True,
            max_concurrent_agents=5,
            auto_detect_domain=True,
        )
        print(f"  ✓ Orchestrator initialized with {5} max concurrent agents")

        # 4. 감사 로그 시작
        self.audit.log(
            AuditEventType.SYSTEM_START,
            actor="PalantirAgentPipeline",
            action="pipeline_start",
            resource="palantir_test",
            details={
                "datasets": list(tables_data.keys()),
                "use_llm": self.use_llm,
                "phases": self.phases,
            }
        )

        # 5. 파이프라인 실행 (동기 모드)
        print("\n[Phase 2-4] Running Agent Pipeline...")
        print("-" * 70)

        event_count = 0
        for event in self.orchestrator.run_sync(tables_data, phases=self.phases):
            event_type = event.get("type", "unknown")
            event_count += 1

            # 이벤트 처리
            self._handle_event(event)

        print("-" * 70)
        print(f"  ✓ Processed {event_count} events")

        # 6. 결과 수집
        print("\n[Phase 5] Collecting Results...")
        self._collect_results()

        # 7. 컴플라이언스 리포트 생성
        print("\n[Phase 6] Generating Compliance Reports...")
        self._generate_compliance_reports()

        # 8. 결과 저장
        print("\n[Phase 7] Saving Results...")
        self._save_results()

        end_time = datetime.now()
        self.results["execution_time"] = str(end_time - start_time)

        # 9. 요약 출력
        self._print_summary()

        return self.results

    def _handle_event(self, event: Dict[str, Any]) -> None:
        """이벤트 핸들러"""
        event_type = event.get("type", "unknown")

        if event_type == "domain_detection_start":
            print(f"  🔍 Auto-detecting domain...")

        elif event_type == "domain_detection_complete":
            domain = event.get("domain", "unknown")
            confidence = event.get("confidence", 0)
            print(f"  ✓ Domain detected: {domain} (confidence: {confidence:.2%})")
            print(f"    Key entities: {event.get('key_entities', [])[:5]}")
            print(f"    Key metrics: {event.get('key_metrics', [])[:5]}")

        elif event_type == "phase_starting":
            phase = event.get("phase", "unknown")
            print(f"\n  ▶ Starting {phase.upper()} phase...")

        elif event_type == "phase_completed":
            phase = event.get("phase", "unknown")
            progress = event.get("progress", 0)
            print(f"  ✓ {phase.upper()} phase completed ({progress:.0%})")
            self.results["phases_completed"].append(phase)

        elif event_type == "todo_started":
            todo_name = event.get("name", "unknown")
            agent = event.get("agent_type", "unknown")
            print(f"    → [{agent}] {todo_name}")
            if agent not in self.results["agents_used"]:
                self.results["agents_used"].append(agent)

        elif event_type == "todo_completed":
            todo_name = event.get("name", "unknown")
            duration = event.get("duration", 0)
            print(f"    ✓ {todo_name} ({duration:.2f}s)")

        elif event_type == "todo_failed":
            todo_name = event.get("name", "unknown")
            error = event.get("error", "unknown")
            print(f"    ✗ {todo_name}: {error}")

        elif event_type == "pipeline_completed":
            stats = event.get("stats", {})
            print(f"\n  ✓ Pipeline completed!")
            print(f"    Total agents created: {stats.get('total_agents_created', 0)}")
            print(f"    Total todos completed: {stats.get('total_todos_completed', 0)}")

        elif event_type == "pipeline_error":
            error = event.get("error", "unknown")
            print(f"\n  ✗ Pipeline error: {error}")

    def _collect_results(self) -> None:
        """SharedContext에서 결과 수집"""
        if not self.orchestrator:
            return

        context = self.orchestrator.get_context()

        # TDA 서명
        if hasattr(context, 'tda_signatures'):
            self.results["tda_signatures"] = context.tda_signatures

        # 엔티티
        if hasattr(context, 'unified_entities'):
            self.results["entities"] = [
                e.to_dict() if hasattr(e, 'to_dict') else str(e)
                for e in context.unified_entities
            ]

        # 관계
        if hasattr(context, 'homeomorphisms'):
            self.results["relationships"] = [
                h.to_dict() if hasattr(h, 'to_dict') else str(h)
                for h in context.homeomorphisms
            ]

        # 온톨로지 개념
        if hasattr(context, 'ontology_concepts'):
            self.results["ontology_concepts"] = [
                c.to_dict() if hasattr(c, 'to_dict') else str(c)
                for c in context.ontology_concepts
            ]

        # 거버넌스 결정
        if hasattr(context, 'governance_decisions'):
            self.results["governance_decisions"] = [
                d.to_dict() if hasattr(d, 'to_dict') else str(d)
                for d in context.governance_decisions
            ]

        # 액션 백로그
        if hasattr(context, 'action_backlog'):
            self.results["governance_actions"] = context.action_backlog

        # 인사이트 (에이전트가 생성한)
        if hasattr(context, 'insights'):
            self.results["insights"] = context.insights

        print(f"  ✓ Collected {len(self.results.get('entities', []))} entities")
        print(f"  ✓ Collected {len(self.results.get('relationships', []))} relationships")
        print(f"  ✓ Collected {len(self.results.get('governance_actions', []))} actions")

    def _generate_compliance_reports(self) -> None:
        """컴플라이언스 리포트 생성"""
        # SOX 리포트
        sox_report = self.audit.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now() + timedelta(days=1),
            framework=ComplianceFramework.SOX
        )

        # GDPR 리포트
        gdpr_report = self.audit.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now() + timedelta(days=1),
            framework=ComplianceFramework.GDPR
        )

        # 리포트 저장
        with open(OUTPUT_DIR / "sox_compliance_report.json", "w") as f:
            json.dump(sox_report, f, indent=2, default=str)

        with open(OUTPUT_DIR / "gdpr_compliance_report.json", "w") as f:
            json.dump(gdpr_report, f, indent=2, default=str)

        print(f"  ✓ SOX Report: {sox_report.get('executive_summary', {}).get('total_events', 0)} events")
        print(f"  ✓ GDPR Report generated")

    def _save_results(self) -> None:
        """결과 저장"""
        # 전체 결과
        with open(OUTPUT_DIR / "pipeline_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # 오케스트레이터 컨텍스트 (이미 저장됨)
        print(f"  ✓ Results saved to {OUTPUT_DIR}")

    def _print_summary(self) -> None:
        """최종 요약 출력"""
        print("\n" + "=" * 70)
        print("AGENT PIPELINE EXECUTION SUMMARY")
        print("=" * 70)

        print(f"\n📊 Data Processing:")
        print(f"   • Datasets Analyzed: {self.results['datasets_analyzed']}")
        print(f"   • Total Records: {self.results['total_rows']:,}")
        print(f"   • Execution Time: {self.results['execution_time']}")

        print(f"\n🤖 Agents Used ({len(self.results['agents_used'])}):")
        for agent in self.results['agents_used']:
            print(f"   • {agent}")

        print(f"\n📋 Phases Completed ({len(self.results['phases_completed'])}):")
        for phase in self.results['phases_completed']:
            print(f"   • {phase.upper()}")

        print(f"\n🔍 Discovery Results:")
        print(f"   • Entities: {len(self.results.get('entities', []))}")
        print(f"   • Relationships: {len(self.results.get('relationships', []))}")

        print(f"\n🎯 Governance Results:")
        print(f"   • Decisions: {len(self.results.get('governance_decisions', []))}")
        print(f"   • Actions: {len(self.results.get('governance_actions', []))}")

        print(f"\n📁 Output Files:")
        for f in OUTPUT_DIR.glob("*.json"):
            size = f.stat().st_size
            print(f"   • {f.name} ({size:,} bytes)")

        print("\n" + "=" * 70)
        print("✅ Agent Pipeline completed successfully!")
        print("=" * 70)


# ============================================================
# 메인 실행
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Palantir Agent Pipeline")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (rule-based only)")
    parser.add_argument("--phases", nargs="+",
                       choices=["discovery", "refinement", "governance"],
                       help="Phases to run (default: all)")

    args = parser.parse_args()

    use_llm = not args.no_llm
    phases = args.phases

    pipeline = PalantirAgentPipeline(use_llm=use_llm, phases=phases)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    results = main()
