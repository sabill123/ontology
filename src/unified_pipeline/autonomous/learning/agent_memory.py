"""
Agent Learning System (v16.0)

에이전트 학습 및 메모리 시스템:
- Experience Replay Buffer
- Pattern Learning (성공/실패 패턴)
- Few-shot Learning 지원
- Adaptive Prompting
- 피드백 기반 개선

SharedContext와 연동하여 FK 탐지 성능 지속 개선

사용법:
    from .agent_memory import AgentMemory, Experience
    memory = AgentMemory()
    memory.store_experience(experience)
    similar_experiences = memory.recall_similar(context)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, TYPE_CHECKING
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json
import hashlib
import logging
import random

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


class ExperienceType(Enum):
    """경험 유형"""
    FK_DETECTION = "fk_detection"
    ENTITY_MAPPING = "entity_mapping"
    RELATIONSHIP_INFERENCE = "relationship_inference"
    ONTOLOGY_GENERATION = "ontology_generation"
    VALIDATION = "validation"
    AGENT_DECISION = "agent_decision"


class Outcome(Enum):
    """결과 유형"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class Experience:
    """학습 경험"""
    experience_id: str
    experience_type: ExperienceType
    timestamp: datetime

    # 입력 컨텍스트
    input_context: Dict[str, Any]  # 입력 데이터/상황

    # 에이전트 행동
    agent_action: str  # 수행한 작업

    # 아래부터 기본값 있는 필드들
    input_hash: str = ""  # 유사성 검색용
    agent_reasoning: Optional[str] = None  # 추론 과정

    # 결과
    outcome: Outcome = Outcome.UNKNOWN
    output_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0

    # 피드백
    feedback: Optional[str] = None
    feedback_score: float = 0.0  # -1.0 ~ 1.0

    # 메타데이터
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.input_hash:
            self.input_hash = self._compute_hash(self.input_context)

    @staticmethod
    def _compute_hash(data: Dict[str, Any]) -> str:
        """컨텍스트 해시 계산"""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "experience_type": self.experience_type.value,
            "timestamp": self.timestamp.isoformat(),
            "input_context": self.input_context,
            "agent_action": self.agent_action,
            "outcome": self.outcome.value,
            "confidence_score": round(self.confidence_score, 4),
            "feedback_score": round(self.feedback_score, 4),
            "tags": self.tags,
        }


@dataclass
class Pattern:
    """학습된 패턴"""
    pattern_id: str
    pattern_type: str  # "success_pattern", "failure_pattern", "decision_pattern"

    # 패턴 정의
    condition: Dict[str, Any]  # 패턴 조건
    action: str  # 권장 행동
    expected_outcome: Outcome

    # 통계
    occurrence_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None

    # 신뢰도
    confidence: float = 0.0  # success_count / occurrence_count

    def update_stats(self, was_successful: bool) -> None:
        """통계 업데이트"""
        self.occurrence_count += 1
        if was_successful:
            self.success_count += 1
        self.confidence = self.success_count / self.occurrence_count if self.occurrence_count > 0 else 0.0
        self.last_used = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "condition": self.condition,
            "action": self.action,
            "expected_outcome": self.expected_outcome.value,
            "occurrence_count": self.occurrence_count,
            "success_count": self.success_count,
            "confidence": round(self.confidence, 4),
        }


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer

    강화학습 스타일의 경험 재생 버퍼:
    - 경험 저장 및 샘플링
    - 우선순위 기반 샘플링
    - 시간 기반 decay
    - 유사 경험 검색
    """

    def __init__(self, max_size: int = 10000, priority_decay: float = 0.99):
        self.max_size = max_size
        self.priority_decay = priority_decay

        self._buffer: List[Experience] = []
        self._priorities: Dict[str, float] = {}  # experience_id -> priority
        self._hash_index: Dict[str, List[str]] = defaultdict(list)  # hash -> experience_ids

    def store(self, experience: Experience, priority: float = 1.0) -> None:
        """경험 저장"""
        # 버퍼 크기 제한
        if len(self._buffer) >= self.max_size:
            self._evict_oldest()

        self._buffer.append(experience)
        self._priorities[experience.experience_id] = priority
        self._hash_index[experience.input_hash].append(experience.experience_id)

        logger.debug(f"[ReplayBuffer] Stored experience: {experience.experience_id}")

    def sample(self, n: int = 1, prioritized: bool = True) -> List[Experience]:
        """경험 샘플링"""
        if not self._buffer:
            return []

        if prioritized:
            # 우선순위 기반 샘플링
            weights = [self._priorities.get(e.experience_id, 1.0) for e in self._buffer]
            total = sum(weights)
            probs = [w / total for w in weights]

            indices = random.choices(range(len(self._buffer)), weights=probs, k=min(n, len(self._buffer)))
            return [self._buffer[i] for i in indices]
        else:
            # 균등 샘플링
            return random.sample(self._buffer, min(n, len(self._buffer)))

    def find_similar(
        self,
        context: Dict[str, Any],
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[Tuple[Experience, float]]:
        """유사 경험 검색"""
        target_hash = Experience._compute_hash(context)

        # 해시 기반 빠른 검색
        exact_matches = []
        for exp_id in self._hash_index.get(target_hash, []):
            for exp in self._buffer:
                if exp.experience_id == exp_id:
                    exact_matches.append((exp, 1.0))
                    break

        if exact_matches:
            return exact_matches[:k]

        # 구조적 유사도 계산
        similarities = []
        for exp in self._buffer:
            sim = self._calculate_similarity(context, exp.input_context)
            if sim >= min_similarity:
                similarities.append((exp, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _calculate_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any],
    ) -> float:
        """컨텍스트 유사도 계산"""
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())

        # 키 유사도
        key_overlap = len(keys1 & keys2) / max(len(keys1 | keys2), 1)

        # 값 유사도 (공통 키에 대해)
        common_keys = keys1 & keys2
        value_matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                value_matches += 1

        value_overlap = value_matches / max(len(common_keys), 1)

        return (key_overlap + value_overlap) / 2

    def update_priority(self, experience_id: str, delta: float) -> None:
        """우선순위 업데이트"""
        if experience_id in self._priorities:
            self._priorities[experience_id] = max(0, self._priorities[experience_id] + delta)

    def decay_priorities(self) -> None:
        """모든 우선순위 감쇠"""
        for exp_id in self._priorities:
            self._priorities[exp_id] *= self.priority_decay

    def _evict_oldest(self) -> None:
        """가장 오래된 경험 제거"""
        if self._buffer:
            oldest = self._buffer.pop(0)
            self._priorities.pop(oldest.experience_id, None)
            if oldest.experience_id in self._hash_index.get(oldest.input_hash, []):
                self._hash_index[oldest.input_hash].remove(oldest.experience_id)

    def get_stats(self) -> Dict[str, Any]:
        """버퍼 통계"""
        by_type = defaultdict(int)
        by_outcome = defaultdict(int)

        for exp in self._buffer:
            by_type[exp.experience_type.value] += 1
            by_outcome[exp.outcome.value] += 1

        return {
            "total_experiences": len(self._buffer),
            "max_size": self.max_size,
            "by_type": dict(by_type),
            "by_outcome": dict(by_outcome),
        }


class PatternLearner:
    """
    Pattern Learner

    성공/실패 패턴 학습:
    - 패턴 추출
    - 패턴 매칭
    - 적응형 추천
    """

    def __init__(self, min_occurrences: int = 3, min_confidence: float = 0.6):
        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence

        self.patterns: Dict[str, Pattern] = {}  # pattern_id -> Pattern
        self._experience_patterns: Dict[str, List[str]] = defaultdict(list)  # exp_id -> pattern_ids

    def learn_from_experiences(self, experiences: List[Experience]) -> int:
        """경험들에서 패턴 학습"""
        new_patterns = 0

        # 유형별 그룹화
        by_type = defaultdict(list)
        for exp in experiences:
            by_type[exp.experience_type].append(exp)

        # 각 유형에서 패턴 추출
        for exp_type, exps in by_type.items():
            patterns = self._extract_patterns(exps)
            for pattern in patterns:
                if pattern.pattern_id not in self.patterns:
                    self.patterns[pattern.pattern_id] = pattern
                    new_patterns += 1
                else:
                    # 기존 패턴 업데이트
                    existing = self.patterns[pattern.pattern_id]
                    existing.occurrence_count += pattern.occurrence_count
                    existing.success_count += pattern.success_count
                    existing.confidence = (
                        existing.success_count / existing.occurrence_count
                        if existing.occurrence_count > 0 else 0
                    )

        logger.info(f"[PatternLearner] Learned {new_patterns} new patterns from {len(experiences)} experiences")
        return new_patterns

    def _extract_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """경험들에서 패턴 추출"""
        patterns = []

        # 성공 경험 패턴
        success_exps = [e for e in experiences if e.outcome == Outcome.SUCCESS]
        if len(success_exps) >= self.min_occurrences:
            common_features = self._find_common_features([e.input_context for e in success_exps])
            if common_features:
                pattern_id = f"success_{hashlib.md5(str(common_features).encode()).hexdigest()[:8]}"
                patterns.append(Pattern(
                    pattern_id=pattern_id,
                    pattern_type="success_pattern",
                    condition=common_features,
                    action=success_exps[0].agent_action,
                    expected_outcome=Outcome.SUCCESS,
                    occurrence_count=len(success_exps),
                    success_count=len(success_exps),
                    confidence=1.0,
                ))

        # 실패 경험 패턴
        failure_exps = [e for e in experiences if e.outcome == Outcome.FAILURE]
        if len(failure_exps) >= self.min_occurrences:
            common_features = self._find_common_features([e.input_context for e in failure_exps])
            if common_features:
                pattern_id = f"failure_{hashlib.md5(str(common_features).encode()).hexdigest()[:8]}"
                patterns.append(Pattern(
                    pattern_id=pattern_id,
                    pattern_type="failure_pattern",
                    condition=common_features,
                    action="avoid:" + failure_exps[0].agent_action,
                    expected_outcome=Outcome.FAILURE,
                    occurrence_count=len(failure_exps),
                    success_count=0,
                    confidence=0.0,
                ))

        return patterns

    def _find_common_features(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """공통 특징 추출"""
        if not contexts:
            return {}

        # 모든 컨텍스트에 존재하는 키
        common_keys = set(contexts[0].keys())
        for ctx in contexts[1:]:
            common_keys &= set(ctx.keys())

        # 동일한 값을 가진 키
        common_features = {}
        for key in common_keys:
            values = [ctx.get(key) for ctx in contexts]
            # 모든 값이 동일하면 공통 특징
            if len(set(str(v) for v in values)) == 1:
                common_features[key] = values[0]
            # 대부분 동일하면 (80% 이상)
            elif values:
                most_common = max(set(str(v) for v in values), key=lambda x: [str(v) for v in values].count(x))
                if [str(v) for v in values].count(most_common) / len(values) >= 0.8:
                    common_features[key] = most_common

        return common_features

    def match_patterns(
        self,
        context: Dict[str, Any],
        pattern_type: Optional[str] = None,
    ) -> List[Tuple[Pattern, float]]:
        """컨텍스트에 맞는 패턴 찾기"""
        matches = []

        for pattern in self.patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue

            match_score = self._calculate_match_score(context, pattern.condition)
            if match_score > 0.5:  # 50% 이상 매칭
                matches.append((pattern, match_score))

        matches.sort(key=lambda x: (x[0].confidence * x[1]), reverse=True)
        return matches

    def _calculate_match_score(
        self,
        context: Dict[str, Any],
        condition: Dict[str, Any],
    ) -> float:
        """매칭 점수 계산"""
        if not condition:
            return 0.0

        matched = 0
        for key, expected_value in condition.items():
            if key in context and str(context[key]) == str(expected_value):
                matched += 1

        return matched / len(condition)

    def get_recommendation(
        self,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """추천 행동 반환"""
        # 성공 패턴 우선
        success_matches = self.match_patterns(context, "success_pattern")
        if success_matches:
            best_pattern, score = success_matches[0]
            if best_pattern.confidence >= self.min_confidence:
                return {
                    "action": best_pattern.action,
                    "pattern_id": best_pattern.pattern_id,
                    "confidence": best_pattern.confidence,
                    "match_score": score,
                    "expected_outcome": best_pattern.expected_outcome.value,
                }

        # 실패 패턴 경고
        failure_matches = self.match_patterns(context, "failure_pattern")
        if failure_matches:
            pattern, score = failure_matches[0]
            return {
                "warning": f"Context matches failure pattern",
                "avoid_action": pattern.action.replace("avoid:", ""),
                "pattern_id": pattern.pattern_id,
                "match_score": score,
            }

        return None


class AgentMemory:
    """
    Agent Memory System

    통합 에이전트 메모리:
    - Experience Replay Buffer
    - Pattern Learning
    - Few-shot Learning 지원
    - Adaptive Prompting
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        min_pattern_occurrences: int = 3,
    ):
        self.replay_buffer = ExperienceReplayBuffer(max_size=buffer_size)
        self.pattern_learner = PatternLearner(min_occurrences=min_pattern_occurrences)

        self._feedback_history: List[Dict[str, Any]] = []

    def store_experience(
        self,
        experience_type: ExperienceType,
        input_context: Dict[str, Any],
        agent_action: str,
        outcome: Outcome = Outcome.UNKNOWN,
        output_data: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        priority: float = 1.0,
    ) -> Experience:
        """경험 저장"""
        import uuid

        experience = Experience(
            experience_id=str(uuid.uuid4()),
            experience_type=experience_type,
            timestamp=datetime.now(),
            input_context=input_context,
            agent_action=agent_action,
            outcome=outcome,
            output_data=output_data,
            confidence_score=confidence,
        )

        # 우선순위 조정 (성공 경험에 높은 가중치)
        if outcome == Outcome.SUCCESS:
            priority *= 1.5
        elif outcome == Outcome.FAILURE:
            priority *= 0.8

        self.replay_buffer.store(experience, priority)

        return experience

    def add_feedback(
        self,
        experience_id: str,
        feedback: str,
        score: float,
    ) -> None:
        """피드백 추가"""
        for exp in self.replay_buffer._buffer:
            if exp.experience_id == experience_id:
                exp.feedback = feedback
                exp.feedback_score = score

                # 피드백에 따라 우선순위 업데이트
                self.replay_buffer.update_priority(experience_id, score * 0.5)

                self._feedback_history.append({
                    "experience_id": experience_id,
                    "feedback": feedback,
                    "score": score,
                    "timestamp": datetime.now().isoformat(),
                })
                break

    def recall_similar(
        self,
        context: Dict[str, Any],
        k: int = 5,
    ) -> List[Experience]:
        """유사 경험 recall"""
        similar = self.replay_buffer.find_similar(context, k=k)
        return [exp for exp, _ in similar]

    def get_few_shot_examples(
        self,
        experience_type: ExperienceType,
        n: int = 3,
        only_success: bool = True,
    ) -> List[Experience]:
        """Few-shot 학습용 예제 반환"""
        candidates = []
        for exp in self.replay_buffer._buffer:
            if exp.experience_type == experience_type:
                if only_success and exp.outcome != Outcome.SUCCESS:
                    continue
                candidates.append(exp)

        # 우선순위 높은 것 선택
        candidates.sort(
            key=lambda e: self.replay_buffer._priorities.get(e.experience_id, 0),
            reverse=True
        )

        return candidates[:n]

    def get_recommendation(
        self,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """패턴 기반 추천"""
        return self.pattern_learner.get_recommendation(context)

    def learn_patterns(self) -> int:
        """현재 버퍼에서 패턴 학습"""
        experiences = self.replay_buffer._buffer
        return self.pattern_learner.learn_from_experiences(experiences)

    def get_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        return {
            "replay_buffer": self.replay_buffer.get_stats(),
            "patterns": {
                "total": len(self.pattern_learner.patterns),
                "success_patterns": sum(
                    1 for p in self.pattern_learner.patterns.values()
                    if p.pattern_type == "success_pattern"
                ),
                "failure_patterns": sum(
                    1 for p in self.pattern_learner.patterns.values()
                    if p.pattern_type == "failure_pattern"
                ),
            },
            "feedback_count": len(self._feedback_history),
        }

    def export_memory(self) -> Dict[str, Any]:
        """메모리 내보내기"""
        return {
            "experiences": [e.to_dict() for e in self.replay_buffer._buffer],
            "patterns": [p.to_dict() for p in self.pattern_learner.patterns.values()],
            "stats": self.get_stats(),
        }


# SharedContext 연동 함수
def create_agent_memory_with_context(
    shared_context: "SharedContext",
) -> AgentMemory:
    """
    SharedContext와 연동된 Agent Memory 생성

    사용법:
        from .agent_memory import create_agent_memory_with_context
        memory = create_agent_memory_with_context(shared_context)
    """
    memory = AgentMemory()

    # 기존 FK 탐지 결과를 경험으로 변환
    for fk in shared_context.enhanced_fk_candidates:
        if isinstance(fk, dict):
            context = {
                "from_table": fk.get("from_table"),
                "from_column": fk.get("from_column"),
                "to_table": fk.get("to_table"),
                "to_column": fk.get("to_column"),
            }

            confidence = fk.get("fk_score", 0)
            outcome = Outcome.SUCCESS if confidence >= 0.7 else Outcome.PARTIAL_SUCCESS

            memory.store_experience(
                experience_type=ExperienceType.FK_DETECTION,
                input_context=context,
                agent_action="detect_fk",
                outcome=outcome,
                output_data=fk,
                confidence=confidence,
            )

    # 패턴 학습
    memory.learn_patterns()

    logger.info(f"[AgentMemory] Initialized with {memory.get_stats()['replay_buffer']['total_experiences']} experiences")

    return memory


# 모듈 초기화
__all__ = [
    "AgentMemory",
    "Experience",
    "ExperienceType",
    "Outcome",
    "Pattern",
    "ExperienceReplayBuffer",
    "PatternLearner",
    "create_agent_memory_with_context",
]
