"""
Conversation Logger for Pipeline Consensus Engine (v2.0)

에이전트 간 대화 내용을 파일로 저장하여 감사 추적 및 디버깅에 활용합니다.
- 세션별 대화 기록
- 라운드별 의견 추적
- 합의 결과 저장
- Markdown 및 JSON 형식 지원
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import (
    ConsensusState,
    ConsensusResult,
    AgentOpinion,
    DiscussionMessage,
    DiscussionRound,
    ConsensusType,
)


class ConversationLogger:
    """에이전트 대화 로깅 클래스"""

    def __init__(self, log_dir: str = "./data/logs/conversations"):
        """
        Args:
            log_dir: 로그 파일 저장 디렉토리
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[str] = None
        self.current_log: Dict[str, Any] = {}
        self._enabled = True

    def enable(self) -> None:
        """로깅 활성화"""
        self._enabled = True

    def disable(self) -> None:
        """로깅 비활성화"""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """로깅 활성화 여부"""
        return self._enabled

    def start_session(self, session_id: str) -> None:
        """
        새 세션 시작

        Args:
            session_id: 세션 식별자
        """
        if not self._enabled:
            return

        self.current_session = session_id
        self.current_log = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "discussions": [],
            "summary": None,
        }

    def start_discussion(
        self,
        topic_id: str,
        topic_name: str,
        topic_type: str,
        topic_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        토론 시작

        Args:
            topic_id: 토론 주제 ID
            topic_name: 토론 주제명
            topic_type: 주제 유형
            topic_data: 토론 데이터

        Returns:
            토론 로그 딕셔너리
        """
        if not self._enabled:
            return {}

        discussion_log = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_type": topic_type,
            "started_at": datetime.now().isoformat(),
            "topic_data_summary": self._summarize_topic_data(topic_data),
            "rounds": [],
            "final_result": None,
            "evidence_bundle_summary": None,
        }
        return discussion_log

    def _summarize_topic_data(self, topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """토론 데이터 요약"""
        summary = {}
        for key, value in topic_data.items():
            if isinstance(value, str):
                summary[key] = value[:200] + "..." if len(value) > 200 else value
            elif isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = f"[{len(value)} items]"
            elif isinstance(value, dict):
                summary[key] = f"{{...{len(value)} keys}}"
            else:
                summary[key] = str(type(value).__name__)
        return summary

    def log_round_start(
        self,
        discussion_log: Dict[str, Any],
        round_num: int,
    ) -> None:
        """
        라운드 시작 로깅

        Args:
            discussion_log: 토론 로그
            round_num: 라운드 번호
        """
        if not self._enabled:
            return

        round_log = {
            "round_number": round_num,
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "consensus_check": None,
        }
        discussion_log["rounds"].append(round_log)

    def log_agent_opinion(
        self,
        discussion_log: Dict[str, Any],
        opinion: AgentOpinion,
        response_text: str,
    ) -> None:
        """
        에이전트 의견 로깅

        Args:
            discussion_log: 토론 로그
            opinion: 에이전트 의견
            response_text: 원본 응답 텍스트
        """
        if not self._enabled:
            return

        if not discussion_log.get("rounds"):
            return

        message_log = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": opinion.agent_id,
            "agent_name": opinion.agent_name,
            "agent_type": opinion.agent_type,
            "confidence": opinion.confidence,
            "uncertainty": opinion.uncertainty,
            "doubt": opinion.doubt,
            "evidence_weight": opinion.evidence_weight,
            "adjusted_confidence": opinion.adjusted_confidence,
            "vote": opinion.legacy_vote,
            "key_observations": opinion.key_observations,
            "concerns": opinion.concerns,
            "suggestions": opinion.suggestions,
            "references_to": opinion.references_to,
            "agrees_with": opinion.agrees_with,
            "disagrees_with": opinion.disagrees_with,
            "reasoning_summary": opinion.reasoning[:500] if opinion.reasoning else "",
            "full_response": response_text[:2000] if response_text else "",
        }

        discussion_log["rounds"][-1]["messages"].append(message_log)

    def log_consensus_check(
        self,
        discussion_log: Dict[str, Any],
        combined_confidence: float,
        agreement_level: float,
        result_type: ConsensusType,
        reached: bool,
    ) -> None:
        """
        합의 체크 결과 로깅

        Args:
            discussion_log: 토론 로그
            combined_confidence: 종합 신뢰도
            agreement_level: 동의 수준
            result_type: 결과 유형
            reached: 합의 도달 여부
        """
        if not self._enabled:
            return

        if discussion_log.get("rounds"):
            discussion_log["rounds"][-1]["consensus_check"] = {
                "timestamp": datetime.now().isoformat(),
                "combined_confidence": combined_confidence,
                "agreement_level": agreement_level,
                "result_type": result_type.value,
                "reached": reached,
            }

    def finish_discussion(
        self,
        discussion_log: Dict[str, Any],
        result: ConsensusResult,
    ) -> str:
        """
        토론 종료 및 저장

        Args:
            discussion_log: 토론 로그
            result: 합의 결과

        Returns:
            저장된 파일 경로
        """
        if not self._enabled:
            return ""

        discussion_log["finished_at"] = datetime.now().isoformat()
        discussion_log["final_result"] = {
            "success": result.success,
            "result_type": result.result_type.value,
            "final_decision": result.final_decision,
            "combined_confidence": result.combined_confidence,
            "agreement_level": result.agreement_level,
            "total_rounds": result.total_rounds,
            "reasoning": result.reasoning,
            "dissenting_views_count": len(result.dissenting_views),
        }

        # EvidenceBundle 요약 추가
        if result.evidence_bundle:
            try:
                discussion_log["evidence_bundle_summary"] = result.evidence_bundle.to_summary()
            except Exception:
                discussion_log["evidence_bundle_summary"] = {"available": True}

        # 세션 로그에 추가
        if "discussions" in self.current_log:
            self.current_log["discussions"].append(discussion_log)

        # 개별 토론 로그 저장
        filepath = self._save_discussion_log(discussion_log)
        return filepath

    def _save_discussion_log(self, discussion_log: Dict[str, Any]) -> str:
        """개별 토론 로그 저장"""
        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in discussion_log.get("topic_name", "unknown")
        )
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(discussion_log, f, ensure_ascii=False, indent=2, default=str)

        return str(filepath)

    def save_session_log(self) -> str:
        """전체 세션 로그 저장"""
        if not self._enabled or not self.current_session:
            return ""

        self.current_log["finished_at"] = datetime.now().isoformat()

        # 요약 생성
        discussions = self.current_log.get("discussions", [])
        self.current_log["summary"] = {
            "total_discussions": len(discussions),
            "accepted": sum(
                1 for d in discussions
                if d.get("final_result", {}).get("result_type") == "accepted"
            ),
            "rejected": sum(
                1 for d in discussions
                if d.get("final_result", {}).get("result_type") == "rejected"
            ),
            "provisional": sum(
                1 for d in discussions
                if d.get("final_result", {}).get("result_type") == "provisional"
            ),
            "escalated": sum(
                1 for d in discussions
                if d.get("final_result", {}).get("result_type") == "escalated"
            ),
            "average_confidence": (
                sum(d.get("final_result", {}).get("combined_confidence", 0) for d in discussions)
                / len(discussions)
                if discussions else 0
            ),
        }

        filename = f"session_{self.current_session}.json"
        filepath = self.log_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_log, f, ensure_ascii=False, indent=2, default=str)

        return str(filepath)

    def format_discussion_markdown(self, discussion_log: Dict[str, Any]) -> str:
        """
        토론 내용을 Markdown 형식으로 포맷팅

        Args:
            discussion_log: 토론 로그

        Returns:
            Markdown 형식 문자열
        """
        lines = []
        lines.append(f"# {discussion_log.get('topic_name', 'Unknown')} 토론 기록")
        lines.append(f"\n**Topic ID**: {discussion_log.get('topic_id', 'N/A')}")
        lines.append(f"**Topic Type**: {discussion_log.get('topic_type', 'N/A')}")
        lines.append(f"**시작 시간**: {discussion_log.get('started_at', 'N/A')}")
        lines.append("")

        # Topic Data Summary
        if discussion_log.get("topic_data_summary"):
            lines.append("## Topic Data Summary")
            lines.append("```json")
            lines.append(json.dumps(discussion_log["topic_data_summary"], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

        # Rounds
        for round_info in discussion_log.get("rounds", []):
            lines.append(f"## 라운드 {round_info.get('round_number', '?')}")
            lines.append("")

            for msg in round_info.get("messages", []):
                lines.append(f"### {msg.get('agent_name', 'Unknown')} ({msg.get('agent_type', 'N/A')})")
                lines.append(f"- **Confidence**: {msg.get('confidence', 0):.2f}")
                lines.append(f"- **Uncertainty**: {msg.get('uncertainty', 0):.2f}")
                lines.append(f"- **Evidence Weight**: {msg.get('evidence_weight', 0):.2f}")
                lines.append(f"- **Vote**: {msg.get('vote', 'N/A')}")

                if msg.get("key_observations"):
                    lines.append("\n**Key Observations:**")
                    for obs in msg["key_observations"]:
                        lines.append(f"  - {obs}")

                if msg.get("concerns"):
                    lines.append("\n**Concerns:**")
                    for concern in msg["concerns"]:
                        lines.append(f"  - {concern}")

                if msg.get("agrees_with"):
                    lines.append(f"\n**Agrees with**: {msg['agrees_with']}")
                if msg.get("disagrees_with"):
                    lines.append(f"\n**Disagrees with**: {msg['disagrees_with']}")

                lines.append("")
                lines.append("---")
                lines.append("")

            # Consensus check
            if round_info.get("consensus_check"):
                check = round_info["consensus_check"]
                lines.append("### 합의 체크")
                lines.append(f"- **Combined Confidence**: {check.get('combined_confidence', 0):.2f}")
                lines.append(f"- **Agreement Level**: {check.get('agreement_level', 0):.2f}")
                lines.append(f"- **Result**: {check.get('result_type', 'N/A')}")
                lines.append(f"- **Reached**: {check.get('reached', False)}")
                lines.append("")

        # Final result
        if discussion_log.get("final_result"):
            result = discussion_log["final_result"]
            lines.append("## 최종 결과")
            lines.append(f"- **Success**: {result.get('success', False)}")
            lines.append(f"- **Result Type**: {result.get('result_type', 'N/A')}")
            lines.append(f"- **Decision**: {result.get('final_decision', 'N/A')}")
            lines.append(f"- **Combined Confidence**: {result.get('combined_confidence', 0):.2f}")
            lines.append(f"- **Agreement Level**: {result.get('agreement_level', 0):.2f}")
            lines.append(f"- **Total Rounds**: {result.get('total_rounds', 0)}")
            lines.append("")

            if result.get("reasoning"):
                lines.append("### Reasoning")
                lines.append(result["reasoning"])
                lines.append("")

        # Evidence Bundle Summary
        if discussion_log.get("evidence_bundle_summary"):
            lines.append("## Evidence Bundle Summary")
            lines.append("```json")
            lines.append(json.dumps(discussion_log["evidence_bundle_summary"], ensure_ascii=False, indent=2, default=str))
            lines.append("```")

        return "\n".join(lines)


# Global logger instance
_logger: Optional[ConversationLogger] = None


def get_conversation_logger() -> ConversationLogger:
    """전역 로거 인스턴스 반환"""
    global _logger
    if _logger is None:
        _logger = ConversationLogger()
    return _logger


def enable_conversation_logging(log_dir: str = "./data/logs/conversations") -> ConversationLogger:
    """
    대화 로깅 활성화

    Args:
        log_dir: 로그 저장 디렉토리

    Returns:
        ConversationLogger 인스턴스
    """
    global _logger
    _logger = ConversationLogger(log_dir)
    _logger.enable()
    return _logger


def disable_conversation_logging() -> None:
    """대화 로깅 비활성화"""
    global _logger
    if _logger:
        _logger.disable()
