"""
Agent Mixins (v6.0)

에이전트 공통 기능을 위한 Mixin 클래스들
- 코드 중복 제거
- 일관된 인터페이스 제공
- 테스트 용이성 향상

사용 예:
    class MyAgent(AutonomousAgent, LLMAnalysisMixin, AuditTrailMixin):
        pass
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..shared_context import SharedContext
    from ..enterprise.audit_logger import (
        AuditLogger, DecisionJustification, EvidenceItem
    )

logger = logging.getLogger(__name__)


# === LLM Analysis Mixin ===

class LLMAnalysisMixin:
    """
    LLM 분석 공통 기능 Mixin

    - LLM 응답 파싱
    - 신뢰도 추출
    - 구조화된 출력 처리
    """

    def parse_llm_response(
        self,
        response: str,
        expected_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        LLM 응답을 구조화된 딕셔너리로 파싱

        Args:
            response: LLM 원시 응답
            expected_keys: 기대하는 키 목록

        Returns:
            파싱된 딕셔너리
        """
        result = {
            "raw_response": response,
            "confidence": 0.5,
            "analysis": "",
            "key_findings": [],
            "recommendations": [],
            "output_data": {},
        }

        if not response:
            return result

        # 신뢰도 추출
        if "[CONFIDENCE:" in response:
            try:
                conf_start = response.index("[CONFIDENCE:") + 12
                conf_end = response.index("]", conf_start)
                conf_str = response[conf_start:conf_end].strip()
                result["confidence"] = float(conf_str)
            except (ValueError, IndexError):
                pass

        # 분석 섹션 추출
        if "[ANALYSIS]" in response:
            try:
                start = response.index("[ANALYSIS]") + 10
                end = response.index("[", start) if "[" in response[start:] else len(response)
                result["analysis"] = response[start:end].strip()
            except (ValueError, IndexError):
                pass

        # KEY_FINDINGS 추출
        if "[KEY_FINDINGS]" in response:
            try:
                start = response.index("[KEY_FINDINGS]") + 14
                end = response.index("[", start) if "[" in response[start:] else len(response)
                findings_text = response[start:end].strip()
                result["key_findings"] = [
                    line.strip().lstrip("-").strip()
                    for line in findings_text.split("\n")
                    if line.strip().startswith("-")
                ]
            except (ValueError, IndexError):
                pass

        # RECOMMENDATIONS 추출
        if "[RECOMMENDATIONS]" in response:
            try:
                start = response.index("[RECOMMENDATIONS]") + 17
                end = response.index("[", start) if "[" in response[start:] else len(response)
                rec_text = response[start:end].strip()
                result["recommendations"] = [
                    line.strip().lstrip("-").strip()
                    for line in rec_text.split("\n")
                    if line.strip().startswith("-")
                ]
            except (ValueError, IndexError):
                pass

        # OUTPUT_DATA JSON 추출
        if "[OUTPUT_DATA]" in response:
            try:
                start = response.index("[OUTPUT_DATA]") + 13
                json_start = response.index("```json", start) + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
                result["output_data"] = json.loads(json_str)
            except (ValueError, IndexError, json.JSONDecodeError):
                # JSON 블록 없이 직접 시도
                try:
                    json_start = response.index("{", start)
                    json_end = response.rindex("}") + 1
                    json_str = response[json_start:json_end]
                    result["output_data"] = json.loads(json_str)
                except (ValueError, json.JSONDecodeError):
                    pass

        # 기대 키 검증
        if expected_keys:
            missing = [k for k in expected_keys if k not in result["output_data"]]
            if missing:
                result["missing_keys"] = missing

        return result

    def extract_confidence_from_response(self, response: str) -> float:
        """응답에서 신뢰도 추출"""
        parsed = self.parse_llm_response(response)
        return parsed.get("confidence", 0.5)

    def format_analysis_prompt(
        self,
        instruction: str,
        context_data: Dict[str, Any],
        output_schema: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        분석 프롬프트 포맷팅

        Args:
            instruction: 분석 지시사항
            context_data: 컨텍스트 데이터
            output_schema: 출력 스키마 (선택)

        Returns:
            포맷된 프롬프트
        """
        # 컨텍스트 포맷팅
        context_parts = []
        for key, value in context_data.items():
            if isinstance(value, (dict, list)):
                formatted = json.dumps(value, indent=2, ensure_ascii=False, default=str)
                context_parts.append(f"### {key}\n```json\n{formatted}\n```")
            else:
                context_parts.append(f"### {key}\n{value}")

        context_str = "\n\n".join(context_parts)

        # 출력 스키마
        schema_str = ""
        if output_schema:
            schema_str = "\n\n## Expected Output Schema\n```json\n"
            schema_str += json.dumps(output_schema, indent=2)
            schema_str += "\n```"

        return f"""## Context
{context_str}

---

## Task
{instruction}
{schema_str}

## Response Format
Please provide your analysis in a structured format:

[CONFIDENCE: 0.XX] - Your confidence level (0.0-1.0)

[ANALYSIS]
Your detailed analysis here...

[KEY_FINDINGS]
- Finding 1
- Finding 2
...

[RECOMMENDATIONS]
- Recommendation 1
- Recommendation 2
...

[OUTPUT_DATA]
```json
{{
    "key": "value",
    ...
}}
```
"""


# === Audit Trail Mixin ===

class AuditTrailMixin:
    """
    Audit Trail 공통 기능 Mixin

    - Decision Justification 생성
    - 증거 추적
    - 감사 로깅 통합
    """

    _audit_logger: Optional["AuditLogger"] = None

    def set_audit_logger(self, logger: "AuditLogger") -> None:
        """Audit Logger 설정"""
        self._audit_logger = logger

    def create_justification(
        self,
        decision_id: str,
        decision_type: str,
        decision_target: str,
        confidence: float,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> "DecisionJustification":
        """
        Decision Justification 생성

        Args:
            decision_id: 결정 ID
            decision_type: 결정 타입
            decision_target: 결정 대상
            confidence: 신뢰도
            agent_id: 에이전트 ID
            agent_type: 에이전트 타입

        Returns:
            DecisionJustification 인스턴스
        """
        from ..enterprise.audit_logger import DecisionJustification

        return DecisionJustification(
            justification_id=f"JUST-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            decision_id=decision_id,
            agent_id=agent_id or getattr(self, 'agent_id', 'unknown'),
            agent_type=agent_type or getattr(self, 'agent_type', 'unknown'),
            decision_type=decision_type,
            decision_target=decision_target,
            final_confidence=confidence,
        )

    def add_evidence(
        self,
        justification: "DecisionJustification",
        evidence_type: str,
        source: str,
        description: str,
        weight: float = 1.0,
        confidence: float = 0.0,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> "DecisionJustification":
        """
        증거 추가

        Args:
            justification: 대상 Justification
            evidence_type: 증거 타입
            source: 소스
            description: 설명
            weight: 가중치
            confidence: 신뢰도
            raw_data: 원본 데이터

        Returns:
            업데이트된 Justification
        """
        from ..enterprise.audit_logger import EvidenceItem

        evidence = EvidenceItem(
            evidence_id=f"EV-{uuid.uuid4().hex[:12]}",
            evidence_type=evidence_type,
            source=source,
            description=description,
            weight=weight,
            confidence=confidence,
            raw_data=raw_data,
        )
        justification.evidence_chain.append(evidence)
        return justification

    def log_decision(
        self,
        justification: "DecisionJustification",
        correlation_id: Optional[str] = None,
    ) -> None:
        """결정 로깅"""
        if self._audit_logger:
            self._audit_logger.log_agent_decision(
                justification=justification,
                correlation_id=correlation_id,
            )


# === Context Update Mixin ===

class ContextUpdateMixin:
    """
    Context 업데이트 공통 기능 Mixin

    - 타입 안전 업데이트
    - 병합 전략
    - 변경 추적
    """

    def update_context_list(
        self,
        context: "SharedContext",
        field_name: str,
        items: List[Any],
        dedup_key: Optional[str] = None,
    ) -> int:
        """
        컨텍스트 리스트 필드 업데이트

        Args:
            context: SharedContext
            field_name: 필드 이름
            items: 추가할 아이템들
            dedup_key: 중복 제거 키 (선택)

        Returns:
            추가된 아이템 수
        """
        if not hasattr(context, field_name):
            logger.warning(f"Context has no field: {field_name}")
            return 0

        current = getattr(context, field_name)
        if not isinstance(current, list):
            logger.warning(f"Field {field_name} is not a list")
            return 0

        added = 0
        for item in items:
            # 중복 체크
            if dedup_key:
                item_key = (
                    getattr(item, dedup_key, None)
                    if hasattr(item, dedup_key)
                    else item.get(dedup_key) if isinstance(item, dict) else None
                )
                existing_keys = [
                    getattr(x, dedup_key, None) if hasattr(x, dedup_key)
                    else x.get(dedup_key) if isinstance(x, dict) else None
                    for x in current
                ]
                if item_key in existing_keys:
                    continue

            current.append(item)
            added += 1

        return added

    def update_context_dict(
        self,
        context: "SharedContext",
        field_name: str,
        updates: Dict[str, Any],
        merge_strategy: str = "update",  # update, replace, deep_merge
    ) -> int:
        """
        컨텍스트 딕셔너리 필드 업데이트

        Args:
            context: SharedContext
            field_name: 필드 이름
            updates: 업데이트할 딕셔너리
            merge_strategy: 병합 전략

        Returns:
            업데이트된 키 수
        """
        if not hasattr(context, field_name):
            logger.warning(f"Context has no field: {field_name}")
            return 0

        current = getattr(context, field_name)
        if not isinstance(current, dict):
            logger.warning(f"Field {field_name} is not a dict")
            return 0

        if merge_strategy == "replace":
            setattr(context, field_name, updates)
            return len(updates)
        elif merge_strategy == "deep_merge":
            self._deep_merge(current, updates)
            return len(updates)
        else:  # update
            current.update(updates)
            return len(updates)

    def _deep_merge(self, base: Dict, updates: Dict) -> None:
        """딥 머지"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# === Analysis Result Mixin ===

class AnalysisResultMixin:
    """
    분석 결과 처리 공통 기능 Mixin

    - 결과 통합
    - 신뢰도 계산
    - 결과 검증
    """

    def combine_analysis_results(
        self,
        results: List[Dict[str, Any]],
        confidence_key: str = "confidence",
        weight_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        여러 분석 결과 통합

        Args:
            results: 분석 결과 목록
            confidence_key: 신뢰도 키
            weight_key: 가중치 키 (선택)

        Returns:
            통합된 결과
        """
        if not results:
            return {"combined_confidence": 0.0, "results_count": 0}

        # 가중 평균 신뢰도 계산
        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            confidence = result.get(confidence_key, 0.5)
            weight = result.get(weight_key, 1.0) if weight_key else 1.0

            weighted_sum += confidence * weight
            total_weight += weight

        combined_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        # 키 파인딩 병합
        all_findings = []
        for result in results:
            findings = result.get("key_findings", [])
            all_findings.extend(findings)

        # 권장사항 병합
        all_recommendations = []
        for result in results:
            recs = result.get("recommendations", [])
            all_recommendations.extend(recs)

        return {
            "combined_confidence": combined_confidence,
            "results_count": len(results),
            "key_findings": list(set(all_findings)),  # 중복 제거
            "recommendations": list(set(all_recommendations)),
            "individual_results": results,
        }

    def calculate_weighted_confidence(
        self,
        confidences: List[Tuple[float, float]],  # (confidence, weight)
    ) -> float:
        """
        가중 평균 신뢰도 계산

        Args:
            confidences: (신뢰도, 가중치) 튜플 목록

        Returns:
            가중 평균 신뢰도
        """
        if not confidences:
            return 0.0

        total_weight = sum(w for _, w in confidences)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(c * w for c, w in confidences)
        return weighted_sum / total_weight

    def validate_result_completeness(
        self,
        result: Dict[str, Any],
        required_keys: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        결과 완전성 검증

        Args:
            result: 검증할 결과
            required_keys: 필수 키 목록

        Returns:
            (유효여부, 누락된 키 목록)
        """
        missing = [k for k in required_keys if k not in result]
        return len(missing) == 0, missing


# === Progress Reporting Mixin ===

class ProgressReportingMixin:
    """
    진행률 보고 공통 기능 Mixin

    - 단계별 진행률
    - 추정 완료 시간
    """

    _progress_steps: List[Tuple[float, str]] = []
    _current_step: int = 0
    _step_start_time: Optional[float] = None

    def init_progress_steps(self, steps: List[str]) -> None:
        """
        진행 단계 초기화

        Args:
            steps: 단계 이름 목록
        """
        if not steps:
            return

        step_size = 1.0 / len(steps)
        self._progress_steps = [
            (i * step_size, step) for i, step in enumerate(steps)
        ]
        self._current_step = 0

    def advance_step(self, step_name: Optional[str] = None) -> float:
        """
        다음 단계로 진행

        Args:
            step_name: 단계 이름 (선택, 로깅용)

        Returns:
            현재 진행률
        """
        import time

        if not self._progress_steps:
            return 0.0

        self._current_step = min(self._current_step + 1, len(self._progress_steps))
        self._step_start_time = time.time()

        if self._current_step < len(self._progress_steps):
            return self._progress_steps[self._current_step][0]
        return 1.0

    def get_current_progress(self) -> float:
        """현재 진행률 반환"""
        if not self._progress_steps or self._current_step >= len(self._progress_steps):
            return 1.0
        return self._progress_steps[self._current_step][0]


# === Combined Agent Mixin ===

class EnhancedAgentMixin(
    LLMAnalysisMixin,
    AuditTrailMixin,
    ContextUpdateMixin,
    AnalysisResultMixin,
    ProgressReportingMixin,
):
    """
    향상된 에이전트 Mixin

    모든 공통 기능을 통합한 Mixin
    """
    pass
