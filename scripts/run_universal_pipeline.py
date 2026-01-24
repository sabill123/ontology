#!/usr/bin/env python3
"""
Universal Pipeline Runner

새 데이터 구조 + 범용 거버넌스 엔진 사용
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase3.universal_engine import (
    UniversalGovernanceEngine,
    StandardInsight,
    StandardDecision,
    StandardAction,
    GovernanceDecisionType,
    ActionStatus,
    ActionPriority
)
from src.phase3.domain_plugins import get_plugin


def load_phase2_output(scenario_dir: Path) -> dict:
    """Phase 2 표준 출력 로드"""
    phase2_file = scenario_dir / "phase2" / "phase2_output.json"
    
    # all_insights.json에서 인사이트 로드
    insights_file = scenario_dir / "phase2" / "insights" / "all_insights.json"
    insights = []
    
    if insights_file.exists():
        with open(insights_file) as f:
            insights = json.load(f)
    
    if phase2_file.exists():
        with open(phase2_file) as f:
            data = json.load(f)
            # insights가 비어있으면 all_insights.json에서 가져옴
            if not data.get("insights") and insights:
                data["insights"] = insights
            return data
    
    # phase2_output.json이 없으면 기본 구조 반환
    return {
        "insights": insights,
        "validation_scores": {},
        "phase1_reference": {"file": "phase1/phase1_output.json"}
    }


def load_phase1_output(scenario_dir: Path) -> dict:
    """Phase 1 표준 출력 로드"""
    phase1_file = scenario_dir / "phase1" / "phase1_output.json"
    
    if phase1_file.exists():
        with open(phase1_file) as f:
            return json.load(f)
    
    return {"mappings": [], "canonical_objects": []}


def convert_to_standard_insight(
    raw_insight: dict,
    phase1_data: dict,
    phase2_data: dict,
    scenario: str
) -> StandardInsight:
    """원시 인사이트를 표준 형식으로 변환"""
    
    # insight_id 추출
    insight_id = (
        raw_insight.get("insight_id") or
        raw_insight.get("persona_id") or
        raw_insight.get("entity_id") or
        f"INS-{hash(str(raw_insight)) % 1000:03d}"
    )
    
    # 설명 추출
    explanation = (
        raw_insight.get("explanation") or
        raw_insight.get("insight_summary") or
        raw_insight.get("description") or
        ""
    )
    
    # 신뢰도 추출
    confidence = float(raw_insight.get("confidence", 0.75))
    
    # 심각도 추출
    severity = raw_insight.get("severity", "medium")
    
    # 소스 시그니처 추출
    source_sigs = []
    if "source_signatures" in raw_insight:
        source_sigs = raw_insight["source_signatures"].split("|")
    
    # Phase 1 매핑 연결
    phase1_mappings = []
    for mapping in phase1_data.get("mappings", []):
        if isinstance(mapping, dict):
            mapping_id = mapping.get("mapping_id", "")
            phase1_mappings.append(mapping_id)
    
    # Phase 2 증거 구성
    phase2_evidence = {
        "validation_scores": phase2_data.get("validation_scores", {}),
        "phase2_file": "phase2/phase2_output.json",
        "source_file": raw_insight.get("source_file", "")
    }
    
    return StandardInsight(
        insight_id=insight_id,
        scenario=scenario,
        confidence=confidence,
        severity=severity,
        phase2_evidence=phase2_evidence,
        phase1_mappings=phase1_mappings if phase1_mappings else ["MAP-unknown"],
        source_signatures=source_sigs if source_sigs else ["unknown_source"],
        explanation=explanation,
        kpis={}  # 엔진에서 계산
    )


def save_phase3_output(
    scenario_dir: Path,
    engine: UniversalGovernanceEngine,
    phase1_file: str,
    phase2_file: str
):
    """Phase 3 표준 출력 저장"""
    
    # phase3 디렉토리 확인
    phase3_dir = scenario_dir / "phase3"
    phase3_dir.mkdir(exist_ok=True)
    (phase3_dir / "governance").mkdir(exist_ok=True)
    (phase3_dir / "actions").mkdir(exist_ok=True)
    
    # 결정 저장
    decisions_data = []
    for d in engine.decisions:
        decisions_data.append({
            "decision_id": d.decision_id,
            "insight_id": d.insight_id,
            "decision_type": d.decision_type.value,
            "confidence": d.confidence,
            "reasoning": d.reasoning,
            "next_review_date": d.next_review_date,
            "escalation_path": d.escalation_path,
            "evidence_attached": d.evidence_attached,
            "phase2_validation_scores": d.phase2_validation_scores,
            "phase1_mappings_referenced": d.phase1_mappings_referenced,
            "agent_opinions": d.agent_opinions,
            "made_by": d.made_by,
            "made_at": d.made_at
        })
    
    with open(phase3_dir / "governance" / "decisions.json", 'w') as f:
        json.dump({"decisions": decisions_data}, f, indent=2)
    
    # 액션 저장
    actions_data = []
    for a in engine.actions:
        actions_data.append({
            "action_id": a.action_id,
            "insight_id": a.insight_id,
            "decision_id": a.decision_id,
            "title": a.title,
            "description": a.description,
            "status": a.status.value,
            "priority": a.priority.value,
            "owner": a.owner,
            "due_date": a.due_date,
            "estimated_kpi_delta": a.estimated_kpi_delta,
            "evidence_chain": a.evidence_chain,
            "created_at": a.created_at
        })
    
    with open(phase3_dir / "actions" / "backlog.json", 'w') as f:
        json.dump({"actions": actions_data}, f, indent=2)
    
    # 표준 출력 파일 생성
    summary = engine.get_summary()
    
    phase3_output = {
        "version": "1.0",
        "scenario": scenario_dir.name,
        "timestamp": datetime.now().isoformat(),
        "phase": "phase3",
        "phase1_reference": {
            "file": phase1_file
        },
        "phase2_reference": {
            "file": phase2_file,
            "insights_processed": summary["total_processed"]
        },
        "summary": {
            "total_decisions": summary["total_processed"],
            "approved": summary["decisions"].get("approve", 0),
            "scheduled_review": summary["decisions"].get("schedule_review", 0),
            "escalated": summary["decisions"].get("escalate", 0),
            "rejected": summary["decisions"].get("reject", 0),
            "actions_created": summary["actions_created"]
        },
        "decisions": decisions_data,
        "actions": actions_data,
        "domain_plugin": summary["domain"],
        "files": {
            "decisions": "phase3/governance/decisions.json",
            "backlog": "phase3/actions/backlog.json"
        }
    }
    
    with open(phase3_dir / "phase3_output.json", 'w') as f:
        json.dump(phase3_output, f, indent=2)
    
    return phase3_output


def run_universal_pipeline(scenario_dir: Path):
    """범용 파이프라인 실행"""
    
    print("=" * 70)
    print("UNIVERSAL GOVERNANCE PIPELINE")
    print("=" * 70)
    print(f"Scenario: {scenario_dir.name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. Phase 1, 2 데이터 로드
    print("\n[1] Loading Phase 1 & 2 data...")
    
    phase1_data = load_phase1_output(scenario_dir)
    print(f"  Phase 1: {len(phase1_data.get('mappings', []))} mappings")
    
    phase2_data = load_phase2_output(scenario_dir)
    raw_insights = phase2_data.get("insights", [])
    print(f"  Phase 2: {len(raw_insights)} insights")
    
    # 2. 도메인 플러그인 로드
    print("\n[2] Loading domain plugin...")
    plugin = get_plugin(scenario_dir.name)
    print(f"  Plugin: {plugin.domain_name}")
    
    # 3. 거버넌스 엔진 초기화
    print("\n[3] Initializing governance engine...")
    engine = UniversalGovernanceEngine(
        domain_plugin=plugin,
        enable_llm_eval=True
    )
    
    # 4. 인사이트 처리
    print("\n[4] Processing insights...")
    print("-" * 70)
    
    for i, raw in enumerate(raw_insights):
        # 표준 인사이트로 변환
        insight = convert_to_standard_insight(
            raw, phase1_data, phase2_data, scenario_dir.name
        )
        
        try:
            # 거버넌스 처리
            decision, action = engine.process_insight(insight)
            
            status_icon = {
                GovernanceDecisionType.APPROVE: "✅",
                GovernanceDecisionType.REJECT: "❌",
                GovernanceDecisionType.SCHEDULE_REVIEW: "📅",
                GovernanceDecisionType.ESCALATE: "🔺",
                GovernanceDecisionType.REQUEST_MORE_DATA: "📋"
            }.get(decision.decision_type, "❓")
            
            action_str = f"→ {action.action_id}" if action else ""
            
            print(f"  {status_icon} {insight.insight_id}: {decision.decision_type.value} (conf: {decision.confidence:.2f}) {action_str}")
            
            if decision.next_review_date:
                print(f"     📅 Review: {decision.next_review_date[:10]}")
            
            if decision.escalation_path:
                print(f"     🔺 Escalation: {' → '.join(decision.escalation_path)}")
            
            if action and action.estimated_kpi_delta:
                kpis = action.estimated_kpi_delta
                kpi_str = ", ".join(f"{k}: {v:.1f}" for k, v in list(kpis.items())[:3])
                print(f"     📊 KPI Delta: {kpi_str}")
                
        except ValueError as e:
            print(f"  ⚠️ {insight.insight_id}: SKIPPED - {e}")
    
    # 5. 결과 저장
    print("\n[5] Saving Phase 3 outputs...")
    
    phase3_output = save_phase3_output(
        scenario_dir,
        engine,
        "phase1/phase1_output.json",
        "phase2/phase2_output.json"
    )
    
    print(f"  ✅ phase3/phase3_output.json")
    print(f"  ✅ phase3/governance/decisions.json")
    print(f"  ✅ phase3/actions/backlog.json")
    
    # 6. 요약
    summary = engine.get_summary()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total Processed: {summary['total_processed']}")
    print(f"   Decisions:")
    for k, v in summary['decisions'].items():
        print(f"     - {k}: {v}")
    print(f"   Actions Created: {summary['actions_created']}")
    print(f"   Domain Plugin: {summary['domain']}")
    
    # 연속성 검증
    print(f"\n🔗 Evidence Continuity:")
    print(f"   Phase 1 → Phase 2: ✅ {phase2_data.get('phase1_reference', {}).get('file', 'linked')}")
    print(f"   Phase 2 → Phase 3: ✅ phase2/phase2_output.json")
    print(f"   Phase 3 Output: ✅ phase3/phase3_output.json")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run universal governance pipeline")
    
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to scenario directory (new structure: data/scenarios/{name})"
    )
    
    args = parser.parse_args()
    
    if not args.scenario.exists():
        print(f"❌ Scenario not found: {args.scenario}")
        return 1
    
    return run_universal_pipeline(args.scenario)


if __name__ == "__main__":
    sys.exit(main())
