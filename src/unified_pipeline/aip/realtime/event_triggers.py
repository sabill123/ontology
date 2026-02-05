"""
Event Triggers (v17.0)

실시간 이벤트 트리거:
- 임계값 트리거 (ThresholdTrigger)
- 패턴 트리거 (PatternTrigger)
- 이상치 트리거 (AnomalyTrigger)

사용 예시:
    # 임계값 트리거
    trigger = ThresholdTrigger("revenue", ">", 10000)

    # 패턴 트리거
    trigger = PatternTrigger("status", ["pending", "processing", "failed"])

    # 이상치 트리거
    trigger = AnomalyTrigger("temperature", z_threshold=3.0)
"""

import logging
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Pattern
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class TriggerSeverity(str, Enum):
    """트리거 심각도"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TriggerResult:
    """트리거 결과"""
    triggered: bool
    trigger_name: str
    severity: TriggerSeverity
    message: str = ""
    field_name: str = ""  # Renamed from 'field' to avoid shadowing dataclass.field
    value: Any = None
    threshold: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 추가 컨텍스트
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "trigger_name": self.trigger_name,
            "severity": self.severity.value,
            "message": self.message,
            "field_name": self.field_name,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "context": self.context,
        }


class EventTrigger(ABC):
    """이벤트 트리거 기본 클래스"""

    def __init__(
        self,
        name: str,
        severity: TriggerSeverity = TriggerSeverity.WARNING,
        enabled: bool = True,
    ):
        self.name = name
        self.severity = severity
        self.enabled = enabled
        self.trigger_count = 0
        self.last_triggered: Optional[str] = None

    @abstractmethod
    def check(self, event: Dict[str, Any]) -> TriggerResult:
        """이벤트 검사"""
        pass

    def _create_result(
        self,
        triggered: bool,
        message: str = "",
        **kwargs,
    ) -> TriggerResult:
        """결과 생성 헬퍼"""
        if triggered:
            self.trigger_count += 1
            self.last_triggered = datetime.now().isoformat()

        return TriggerResult(
            triggered=triggered,
            trigger_name=self.name,
            severity=self.severity,
            message=message,
            **kwargs,
        )


class ThresholdTrigger(EventTrigger):
    """
    임계값 트리거

    특정 필드가 임계값을 초과/미만/동일할 때 트리거
    """

    OPERATORS = {
        ">": lambda v, t: v > t,
        ">=": lambda v, t: v >= t,
        "<": lambda v, t: v < t,
        "<=": lambda v, t: v <= t,
        "==": lambda v, t: v == t,
        "!=": lambda v, t: v != t,
    }

    def __init__(
        self,
        field: str,
        operator: str,
        threshold: float,
        name: Optional[str] = None,
        severity: TriggerSeverity = TriggerSeverity.WARNING,
    ):
        super().__init__(
            name=name or f"threshold_{field}_{operator}_{threshold}",
            severity=severity,
        )
        self.field = field
        self.operator = operator
        self.threshold = threshold

        if operator not in self.OPERATORS:
            raise ValueError(f"Invalid operator: {operator}")

    def check(self, event: Dict[str, Any]) -> TriggerResult:
        """임계값 검사"""
        value = event.get(self.field)

        if value is None:
            return self._create_result(
                triggered=False,
                message=f"Field '{self.field}' not found",
            )

        try:
            value = float(value)
        except (ValueError, TypeError):
            return self._create_result(
                triggered=False,
                message=f"Cannot convert '{value}' to number",
            )

        triggered = self.OPERATORS[self.operator](value, self.threshold)

        return self._create_result(
            triggered=triggered,
            message=(
                f"{self.field} ({value}) {self.operator} {self.threshold}"
                if triggered else ""
            ),
            field_name=self.field,
            value=value,
            threshold=self.threshold,
        )


class PatternTrigger(EventTrigger):
    """
    패턴 트리거

    특정 필드가 패턴(시퀀스)을 따르거나 특정 값일 때 트리거
    """

    def __init__(
        self,
        field: str,
        patterns: List[str],
        match_type: str = "any",  # "any", "all", "sequence"
        name: Optional[str] = None,
        severity: TriggerSeverity = TriggerSeverity.WARNING,
        use_regex: bool = False,
    ):
        super().__init__(
            name=name or f"pattern_{field}",
            severity=severity,
        )
        self.field = field
        self.patterns = patterns
        self.match_type = match_type
        self.use_regex = use_regex

        # 시퀀스 매칭용
        self.sequence_buffer: deque = deque(maxlen=len(patterns))

        # 정규식 컴파일
        if use_regex:
            self.compiled_patterns: List[Pattern] = [
                re.compile(p) for p in patterns
            ]

    def check(self, event: Dict[str, Any]) -> TriggerResult:
        """패턴 검사"""
        value = event.get(self.field)

        if value is None:
            return self._create_result(
                triggered=False,
                message=f"Field '{self.field}' not found",
            )

        value_str = str(value)

        if self.match_type == "sequence":
            return self._check_sequence(value_str)
        elif self.match_type == "all":
            return self._check_all(value_str)
        else:  # any
            return self._check_any(value_str)

    def _check_any(self, value: str) -> TriggerResult:
        """값이 패턴 중 하나와 일치"""
        matched = False
        matched_pattern = None

        if self.use_regex:
            for i, pattern in enumerate(self.compiled_patterns):
                if pattern.search(value):
                    matched = True
                    matched_pattern = self.patterns[i]
                    break
        else:
            if value in self.patterns:
                matched = True
                matched_pattern = value

        return self._create_result(
            triggered=matched,
            message=(
                f"{self.field}='{value}' matches pattern '{matched_pattern}'"
                if matched else ""
            ),
            field_name=self.field,
            value=value,
            context={"matched_pattern": matched_pattern},
        )

    def _check_all(self, value: str) -> TriggerResult:
        """값이 모든 패턴과 일치 (정규식용)"""
        if not self.use_regex:
            return self._check_any(value)

        all_matched = all(p.search(value) for p in self.compiled_patterns)

        return self._create_result(
            triggered=all_matched,
            message=(
                f"{self.field}='{value}' matches all patterns"
                if all_matched else ""
            ),
            field_name=self.field,
            value=value,
        )

    def _check_sequence(self, value: str) -> TriggerResult:
        """값 시퀀스가 패턴과 일치"""
        self.sequence_buffer.append(value)

        if len(self.sequence_buffer) < len(self.patterns):
            return self._create_result(
                triggered=False,
                message="Sequence not complete",
            )

        matched = list(self.sequence_buffer) == self.patterns

        return self._create_result(
            triggered=matched,
            message=(
                f"Sequence detected: {list(self.sequence_buffer)}"
                if matched else ""
            ),
            field_name=self.field,
            value=value,
            context={"sequence": list(self.sequence_buffer)},
        )


class AnomalyTrigger(EventTrigger):
    """
    이상치 트리거

    통계적 이상치 (Z-score, IQR 기반) 감지
    """

    def __init__(
        self,
        field: str,
        z_threshold: float = 3.0,
        window_size: int = 100,
        method: str = "zscore",  # "zscore", "iqr"
        name: Optional[str] = None,
        severity: TriggerSeverity = TriggerSeverity.WARNING,
    ):
        super().__init__(
            name=name or f"anomaly_{field}",
            severity=severity,
        )
        self.field = field
        self.z_threshold = z_threshold
        self.window_size = window_size
        self.method = method

        # 통계용 버퍼
        self.value_buffer: deque = deque(maxlen=window_size)

    def check(self, event: Dict[str, Any]) -> TriggerResult:
        """이상치 검사"""
        value = event.get(self.field)

        if value is None:
            return self._create_result(
                triggered=False,
                message=f"Field '{self.field}' not found",
            )

        try:
            value = float(value)
        except (ValueError, TypeError):
            return self._create_result(
                triggered=False,
                message=f"Cannot convert '{value}' to number",
            )

        # 버퍼에 추가
        self.value_buffer.append(value)

        # 최소 데이터 필요
        if len(self.value_buffer) < 10:
            return self._create_result(
                triggered=False,
                message="Insufficient data for anomaly detection",
            )

        # 이상치 검사
        if self.method == "zscore":
            is_anomaly, z_score = self._check_zscore(value)
            context = {"z_score": z_score}
        else:  # iqr
            is_anomaly, iqr_bounds = self._check_iqr(value)
            context = {"iqr_bounds": iqr_bounds}

        return self._create_result(
            triggered=is_anomaly,
            message=(
                f"Anomaly detected: {self.field}={value}"
                if is_anomaly else ""
            ),
            field_name=self.field,
            value=value,
            threshold=self.z_threshold,
            context=context,
        )

    def _check_zscore(self, value: float) -> tuple:
        """Z-score 기반 이상치 검사"""
        values = list(self.value_buffer)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0

        if std == 0:
            return False, 0.0

        z_score = abs(value - mean) / std
        is_anomaly = z_score > self.z_threshold

        return is_anomaly, round(z_score, 2)

    def _check_iqr(self, value: float) -> tuple:
        """IQR 기반 이상치 검사"""
        values = sorted(self.value_buffer)
        n = len(values)

        q1 = values[n // 4]
        q3 = values[3 * n // 4]
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        is_anomaly = value < lower or value > upper

        return is_anomaly, {"lower": lower, "upper": upper}


class CompositeTrigger(EventTrigger):
    """
    복합 트리거

    여러 트리거를 AND/OR 조합
    """

    def __init__(
        self,
        triggers: List[EventTrigger],
        operator: str = "and",  # "and", "or"
        name: Optional[str] = None,
        severity: TriggerSeverity = TriggerSeverity.WARNING,
    ):
        super().__init__(
            name=name or f"composite_{operator}",
            severity=severity,
        )
        self.triggers = triggers
        self.operator = operator

    def check(self, event: Dict[str, Any]) -> TriggerResult:
        """복합 검사"""
        results = [t.check(event) for t in self.triggers]
        triggered_results = [r for r in results if r.triggered]

        if self.operator == "and":
            triggered = all(r.triggered for r in results)
        else:  # or
            triggered = any(r.triggered for r in results)

        messages = [r.message for r in triggered_results if r.message]

        return self._create_result(
            triggered=triggered,
            message=" | ".join(messages) if messages else "",
            context={
                "sub_results": [r.to_dict() for r in results],
                "triggered_count": len(triggered_results),
            },
        )


class CustomTrigger(EventTrigger):
    """
    커스텀 트리거

    사용자 정의 함수 기반
    """

    def __init__(
        self,
        check_fn: Callable[[Dict[str, Any]], bool],
        message_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        name: str = "custom",
        severity: TriggerSeverity = TriggerSeverity.WARNING,
    ):
        super().__init__(name=name, severity=severity)
        self.check_fn = check_fn
        self.message_fn = message_fn

    def check(self, event: Dict[str, Any]) -> TriggerResult:
        """커스텀 검사"""
        try:
            triggered = self.check_fn(event)
            message = ""
            if triggered and self.message_fn:
                message = self.message_fn(event)

            return self._create_result(
                triggered=triggered,
                message=message,
                context={"event": event},
            )
        except Exception as e:
            return self._create_result(
                triggered=False,
                message=f"Check failed: {e}",
            )
