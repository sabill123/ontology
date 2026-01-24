# DSPy + Instructor 도입 설계서

## 개요

이 문서는 Ontoloty 시스템에 DSPy와 Instructor를 도입하기 위한 설계서입니다.

---

## 1. 사전 준비 (사용자 액션)

### 1.1 필요한 작업

| 항목 | 설명 | 필요 여부 |
|------|------|----------|
| **API 키 발급** | 불필요 - 기존 Letsur Gateway API 키 사용 | ❌ |
| **사이트 가입** | 불필요 - 오픈소스 라이브러리 | ❌ |
| **패키지 설치** | `pip install` 필요 | ✅ |

### 1.2 설치 명령어

```bash
# 프로젝트 루트에서 실행
cd /Users/jaeseokhan/Desktop/Work/ontoloty

# 가상환경 활성화 (있다면)
source venv/bin/activate  # 또는 conda activate ontoloty

# 패키지 설치
pip install dspy-ai instructor

# requirements.txt 업데이트 (권장)
echo "dspy-ai>=2.4.0" >> requirements.txt
echo "instructor>=1.0.0" >> requirements.txt
```

### 1.3 버전 호환성

```
Python: >= 3.9
dspy-ai: >= 2.4.0
instructor: >= 1.0.0
pydantic: >= 2.0 (이미 설치됨)
openai: >= 1.0 (이미 설치됨)
```

---

## 2. 현재 시스템 분석

### 2.1 LLM 호출 패턴 (현재)

```
src/common/utils/llm.py
├── chat_completion()              # 일반 LLM 호출
├── chat_completion_with_json()    # JSON 응답 요청
└── extract_json_from_response()   # JSON 파싱 (오류 발생 가능)

src/unified_pipeline/autonomous/base.py
├── call_llm()                     # 에이전트 LLM 호출
└── call_llm_with_context()        # 컨텍스트 포함 호출
```

### 2.2 문제점

1. **JSON 파싱 불안정**: `extract_json_from_response()`가 regex 기반으로 불안정
2. **프롬프트 하드코딩**: 모든 프롬프트가 f-string으로 하드코딩
3. **재시도 로직 없음**: 파싱 실패 시 빈 dict 반환
4. **타입 안전성 부족**: Dict[str, Any] 반환으로 런타임 에러 위험

---

## 3. Instructor 도입 설계

### 3.1 통합 아키텍처

```
[기존]
Agent → call_llm() → chat_completion_with_json() → extract_json_from_response() → Dict

[개선]
Agent → call_llm_structured() → instructor.patch(client) → Pydantic Model
```

### 3.2 신규 파일: `src/common/utils/llm_instructor.py`

```python
"""
Instructor 기반 구조화된 LLM 호출
- Pydantic 모델로 타입 안전한 응답 보장
- 자동 재시도 및 검증
"""

import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import Type, TypeVar, Optional, List, Dict, Any
import logging

from .llm import LETSUR_BASE_URL, LETSUR_API_KEY, DEFAULT_MODEL

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def get_instructor_client() -> instructor.Instructor:
    """Instructor로 패치된 OpenAI 클라이언트 반환"""
    base_client = OpenAI(
        base_url=LETSUR_BASE_URL,
        api_key=LETSUR_API_KEY,
    )
    return instructor.from_openai(base_client)


def call_llm_structured(
    response_model: Type[T],
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    client: Optional[instructor.Instructor] = None,
) -> T:
    """
    구조화된 LLM 호출 - Pydantic 모델로 응답 보장

    Args:
        response_model: 응답 Pydantic 모델 클래스
        messages: 메시지 리스트
        model: 모델명 (기본값: DEFAULT_MODEL)
        temperature: 온도
        max_retries: 검증 실패 시 재시도 횟수
        client: Instructor 클라이언트 (선택)

    Returns:
        response_model 인스턴스

    Example:
        class FKAnalysis(BaseModel):
            is_relationship: bool
            confidence: float
            reasoning: str

        result = call_llm_structured(
            response_model=FKAnalysis,
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": "Analyze this column..."}
            ]
        )
        # result.is_relationship, result.confidence 등 타입 안전하게 접근
    """
    if client is None:
        client = get_instructor_client()

    return client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_retries=max_retries,
    )


def call_llm_structured_with_context(
    response_model: Type[T],
    instruction: str,
    context_data: Dict[str, Any],
    system_prompt: str = "You are a data analysis expert.",
    model: str = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> T:
    """
    컨텍스트와 함께 구조화된 LLM 호출

    Args:
        response_model: 응답 Pydantic 모델
        instruction: 지시사항
        context_data: 컨텍스트 데이터
        system_prompt: 시스템 프롬프트
        model: 모델명
        temperature: 온도
        max_retries: 재시도 횟수

    Returns:
        response_model 인스턴스
    """
    import json

    # 컨텍스트 포맷팅
    context_str = json.dumps(context_data, ensure_ascii=False, indent=2, default=str)

    user_prompt = f"""## Context
{context_str}

## Task
{instruction}

Analyze carefully and provide your response."""

    return call_llm_structured(
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )
```

### 3.3 응답 모델 정의: `src/unified_pipeline/llm_schemas.py`

```python
"""
LLM 응답용 Pydantic 스키마
- 기존 models.py의 모델 확장
- LLM 응답 검증에 특화
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# === FK Detection ===

class FKRelationshipType(str, Enum):
    FOREIGN_KEY = "foreign_key"
    CROSS_SILO_REFERENCE = "cross_silo_reference"
    SEMANTIC_LINK = "semantic_link"
    NONE = "none"


class Cardinality(str, Enum):
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"


class FKAnalysisResult(BaseModel):
    """FK 관계 분석 결과"""
    is_relationship: bool = Field(description="FK 관계 존재 여부")
    relationship_type: FKRelationshipType = Field(description="관계 유형")
    confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
    cardinality: Cardinality = Field(description="카디널리티")
    semantic_relationship: str = Field(description="의미론적 관계 설명")
    reasoning: str = Field(description="판단 근거")


# === Column Analysis ===

class SemanticRole(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    DIMENSION = "DIMENSION"
    MEASURE = "MEASURE"
    TEMPORAL = "TEMPORAL"
    DESCRIPTOR = "DESCRIPTOR"
    FLAG = "FLAG"
    REFERENCE = "REFERENCE"
    UNKNOWN = "UNKNOWN"


class ColumnTypeAnalysis(BaseModel):
    """컬럼 타입 분석 결과"""
    inferred_type: str = Field(description="추론된 데이터 타입")
    semantic_type: str = Field(description="의미론적 타입")
    should_preserve_as_string: bool = Field(description="문자열 유지 필요 여부")
    is_identifier: bool = Field(description="식별자 여부")
    is_primary_key_candidate: bool = Field(description="PK 후보 여부")
    is_foreign_key_candidate: bool = Field(description="FK 후보 여부")
    referenced_entity: Optional[str] = Field(None, description="참조 엔티티")
    pattern_description: str = Field(description="패턴 설명")
    confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")


class SemanticRoleClassification(BaseModel):
    """컬럼 의미역 분류 결과"""
    semantic_role: SemanticRole = Field(description="의미역")
    confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
    business_name: str = Field(description="비즈니스 이름")
    business_description: str = Field(description="비즈니스 설명")
    domain_category: Optional[str] = Field(None, description="도메인 카테고리")
    reasoning: str = Field(description="분류 근거")


# === Entity Analysis ===

class EntityType(str, Enum):
    MASTER_DATA = "master_data"
    TRANSACTION = "transaction"
    REFERENCE = "reference"
    JUNCTION = "junction"


class EntitySemantics(BaseModel):
    """엔티티 의미론 분석 결과"""
    entity_name: str = Field(description="엔티티 이름 (PascalCase)")
    display_name: str = Field(description="표시 이름")
    description: str = Field(description="엔티티 설명")
    entity_type: EntityType = Field(description="엔티티 유형")
    business_domain: str = Field(description="비즈니스 도메인")
    key_attributes: List[str] = Field(default_factory=list, description="핵심 속성")
    relationships_hint: List[str] = Field(default_factory=list, description="관계 힌트")


# === Consensus ===

class VoteDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class AgentVote(BaseModel):
    """에이전트 투표 결과"""
    decision: VoteDecision = Field(description="투표 결정")
    confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
    key_observations: List[str] = Field(default_factory=list, description="핵심 관찰")
    concerns: List[str] = Field(default_factory=list, description="우려사항")
    reasoning: str = Field(description="투표 근거")


# === Dynamic Threshold ===

class ThresholdRecommendation(BaseModel):
    """동적 임계값 추천 결과"""
    recommended_value: float = Field(ge=0.0, le=1.0, description="추천 임계값")
    reasoning: str = Field(description="추천 근거")
    context_factors: List[str] = Field(default_factory=list, description="고려 요소")
    confidence: float = Field(ge=0.0, le=1.0, description="추천 신뢰도")
    alternative_values: Dict[str, float] = Field(
        default_factory=dict,
        description="대안 값 (conservative, moderate, aggressive)"
    )
```

### 3.4 기존 함수 마이그레이션

**Before** (`llm.py`):
```python
def detect_foreign_key_relationship(...) -> Dict[str, Any]:
    response = chat_completion_with_json(...)
    return extract_json_from_response(response)  # 실패 가능
```

**After** (`llm_instructor.py`):
```python
def detect_foreign_key_relationship(...) -> FKAnalysisResult:
    return call_llm_structured(
        response_model=FKAnalysisResult,
        messages=[...],
        max_retries=3  # 자동 재시도
    )
    # 항상 유효한 FKAnalysisResult 반환
```

---

## 4. DSPy 도입 설계

### 4.1 통합 아키텍처

```
[기존]
Agent → 하드코딩된 프롬프트 → LLM → 응답

[개선]
Agent → DSPy Signature → DSPy Module → LLM → 구조화된 응답
                 ↓
        프롬프트 자동 최적화 (Compile)
```

### 4.2 신규 파일: `src/unified_pipeline/dspy_modules/config.py`

```python
"""
DSPy 설정 및 LM 초기화
"""

import dspy
import os
from dotenv import load_dotenv

load_dotenv()

LETSUR_BASE_URL = "https://gateway.letsur.ai/v1"
LETSUR_API_KEY = os.environ.get("LETSUR_API_KEY", "")


def get_dspy_lm(model: str = "gpt-5.1", temperature: float = 0.2) -> dspy.LM:
    """DSPy Language Model 초기화"""
    return dspy.LM(
        model=f"openai/{model}",
        api_base=LETSUR_BASE_URL,
        api_key=LETSUR_API_KEY,
        temperature=temperature,
    )


def configure_dspy(model: str = "gpt-5.1"):
    """DSPy 전역 설정"""
    lm = get_dspy_lm(model)
    dspy.configure(lm=lm)
    return lm
```

### 4.3 신규 파일: `src/unified_pipeline/dspy_modules/signatures.py`

```python
"""
DSPy Signatures - 프롬프트를 선언적으로 정의
"""

import dspy
from typing import List, Optional


class FKRelationshipSignature(dspy.Signature):
    """Analyze if there's a foreign key relationship between two columns."""

    # 입력 필드
    source_table: str = dspy.InputField(desc="Source table name")
    source_column: str = dspy.InputField(desc="Source column name")
    source_samples: List[str] = dspy.InputField(desc="Sample values from source")
    target_table: str = dspy.InputField(desc="Target table name")
    target_column: str = dspy.InputField(desc="Target column name")
    target_samples: List[str] = dspy.InputField(desc="Sample values from target")
    overlap_ratio: float = dspy.InputField(desc="Value overlap ratio (0-1)")

    # 출력 필드
    is_relationship: bool = dspy.OutputField(desc="Whether FK relationship exists")
    relationship_type: str = dspy.OutputField(desc="foreign_key|cross_silo|semantic|none")
    cardinality: str = dspy.OutputField(desc="ONE_TO_ONE|ONE_TO_MANY|MANY_TO_ONE|MANY_TO_MANY")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Explanation of the decision")


class ColumnSemanticSignature(dspy.Signature):
    """Classify column's semantic role in business context."""

    column_name: str = dspy.InputField(desc="Column name")
    table_name: str = dspy.InputField(desc="Table name for context")
    data_type: str = dspy.InputField(desc="Data type of column")
    sample_values: List[str] = dspy.InputField(desc="Sample values")
    distinct_ratio: Optional[float] = dspy.InputField(desc="Unique values ratio")

    semantic_role: str = dspy.OutputField(
        desc="IDENTIFIER|DIMENSION|MEASURE|TEMPORAL|DESCRIPTOR|FLAG|REFERENCE"
    )
    business_name: str = dspy.OutputField(desc="Human readable business name")
    domain_category: str = dspy.OutputField(desc="Business domain category")
    confidence: float = dspy.OutputField(desc="Classification confidence 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Why this classification was chosen")


class EntityAnalysisSignature(dspy.Signature):
    """Infer semantic meaning of a database table/entity."""

    table_name: str = dspy.InputField(desc="Table name")
    columns: List[str] = dspy.InputField(desc="Column names")
    sample_data: str = dspy.InputField(desc="JSON sample data")
    source_context: str = dspy.InputField(desc="Data source context")

    entity_name: str = dspy.OutputField(desc="PascalCase entity name")
    display_name: str = dspy.OutputField(desc="Human readable display name")
    description: str = dspy.OutputField(desc="Entity description")
    entity_type: str = dspy.OutputField(desc="master_data|transaction|reference|junction")
    business_domain: str = dspy.OutputField(desc="Business domain")
    key_attributes: List[str] = dspy.OutputField(desc="Key attribute columns")


class ThresholdDecisionSignature(dspy.Signature):
    """Determine optimal threshold for data analysis."""

    threshold_type: str = dspy.InputField(desc="Type of threshold needed")
    data_characteristics: str = dspy.InputField(desc="JSON data characteristics")
    domain_context: str = dspy.InputField(desc="Domain context information")

    recommended_value: float = dspy.OutputField(desc="Recommended threshold 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Why this value was chosen")
    confidence: float = dspy.OutputField(desc="Recommendation confidence")


class ConsensusVoteSignature(dspy.Signature):
    """Vote on ontology proposal from specific role perspective."""

    proposal: str = dspy.InputField(desc="The proposal to vote on")
    evidence: str = dspy.InputField(desc="Supporting evidence")
    role: str = dspy.InputField(desc="Voter role perspective")
    previous_votes: str = dspy.InputField(desc="Other agents' votes if any")

    vote: str = dspy.OutputField(desc="approve|reject|abstain")
    confidence: float = dspy.OutputField(desc="Vote confidence 0.0-1.0")
    key_observations: List[str] = dspy.OutputField(desc="Key observations")
    concerns: List[str] = dspy.OutputField(desc="Any concerns")
    reasoning: str = dspy.OutputField(desc="Vote reasoning")
```

### 4.4 신규 파일: `src/unified_pipeline/dspy_modules/modules.py`

```python
"""
DSPy Modules - 재사용 가능한 LLM 호출 모듈
"""

import dspy
from typing import List, Dict, Any, Optional
import json

from .signatures import (
    FKRelationshipSignature,
    ColumnSemanticSignature,
    EntityAnalysisSignature,
    ThresholdDecisionSignature,
    ConsensusVoteSignature,
)


class FKRelationshipAnalyzer(dspy.Module):
    """FK 관계 분석 모듈"""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(FKRelationshipSignature)

    def forward(
        self,
        source_table: str,
        source_column: str,
        source_samples: List[str],
        target_table: str,
        target_column: str,
        target_samples: List[str],
        overlap_ratio: float,
    ):
        return self.analyze(
            source_table=source_table,
            source_column=source_column,
            source_samples=source_samples,
            target_table=target_table,
            target_column=target_column,
            target_samples=target_samples,
            overlap_ratio=overlap_ratio,
        )


class ColumnSemanticClassifier(dspy.Module):
    """컬럼 의미역 분류 모듈"""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ColumnSemanticSignature)

    def forward(
        self,
        column_name: str,
        table_name: str,
        data_type: str,
        sample_values: List[str],
        distinct_ratio: Optional[float] = None,
    ):
        return self.classify(
            column_name=column_name,
            table_name=table_name,
            data_type=data_type,
            sample_values=sample_values,
            distinct_ratio=distinct_ratio,
        )


class EntityAnalyzer(dspy.Module):
    """엔티티 분석 모듈"""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(EntityAnalysisSignature)

    def forward(
        self,
        table_name: str,
        columns: List[str],
        sample_data: List[Dict],
        source_context: str,
    ):
        return self.analyze(
            table_name=table_name,
            columns=columns,
            sample_data=json.dumps(sample_data[:5], default=str),
            source_context=source_context,
        )


class DynamicThresholdDecider(dspy.Module):
    """동적 임계값 결정 모듈"""

    def __init__(self):
        super().__init__()
        self.decide = dspy.ChainOfThought(ThresholdDecisionSignature)

    def forward(
        self,
        threshold_type: str,
        data_characteristics: Dict[str, Any],
        domain_context: Dict[str, Any] = None,
    ):
        return self.decide(
            threshold_type=threshold_type,
            data_characteristics=json.dumps(data_characteristics, default=str),
            domain_context=json.dumps(domain_context or {}, default=str),
        )


class ConsensusVoter(dspy.Module):
    """컨센서스 투표 모듈"""

    def __init__(self):
        super().__init__()
        self.vote = dspy.ChainOfThought(ConsensusVoteSignature)

    def forward(
        self,
        proposal: str,
        evidence: str,
        role: str,
        previous_votes: List[Dict] = None,
    ):
        return self.vote(
            proposal=proposal,
            evidence=evidence,
            role=role,
            previous_votes=json.dumps(previous_votes or [], default=str),
        )


# === 프롬프트 최적화용 헬퍼 ===

def compile_module(
    module: dspy.Module,
    trainset: List[dspy.Example],
    metric=None,
) -> dspy.Module:
    """
    모듈 프롬프트 최적화 (Few-shot 예시 자동 선택)

    Args:
        module: 최적화할 DSPy 모듈
        trainset: 훈련 예시 리스트
        metric: 평가 함수 (선택)

    Returns:
        최적화된 모듈
    """
    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(metric=metric)
    return optimizer.compile(module, trainset=trainset)


def save_optimized_module(module: dspy.Module, path: str):
    """최적화된 모듈 저장"""
    module.save(path)


def load_optimized_module(module_class, path: str) -> dspy.Module:
    """최적화된 모듈 로드"""
    module = module_class()
    module.load(path)
    return module
```

---

## 5. 통합 사용 예시

### 5.1 Instructor 사용 예시

```python
# src/unified_pipeline/autonomous/analysis/fk_detector_v2.py

from src.common.utils.llm_instructor import call_llm_structured
from src.unified_pipeline.llm_schemas import FKAnalysisResult

def detect_fk_relationship(
    source_col: str,
    target_col: str,
    source_samples: list,
    target_samples: list,
) -> FKAnalysisResult:
    """FK 관계 감지 - Instructor 버전"""

    messages = [
        {
            "role": "system",
            "content": "You are a database relationship expert. Analyze FK relationships."
        },
        {
            "role": "user",
            "content": f"""
Analyze FK relationship:
- Source: {source_col} with values {source_samples[:10]}
- Target: {target_col} with values {target_samples[:10]}
"""
        }
    ]

    # 항상 FKAnalysisResult 반환 (검증 실패 시 자동 재시도)
    result = call_llm_structured(
        response_model=FKAnalysisResult,
        messages=messages,
        max_retries=3,
    )

    return result  # 타입 안전: result.is_relationship, result.confidence 등
```

### 5.2 DSPy 사용 예시

```python
# src/unified_pipeline/autonomous/analysis/semantic_classifier_v2.py

from src.unified_pipeline.dspy_modules.config import configure_dspy
from src.unified_pipeline.dspy_modules.modules import ColumnSemanticClassifier

# 초기화 (앱 시작 시 1회)
configure_dspy(model="gpt-5.1")

# 모듈 인스턴스
classifier = ColumnSemanticClassifier()

def classify_column_semantic(
    column_name: str,
    table_name: str,
    samples: list,
    data_type: str = "string",
) -> dict:
    """컬럼 의미역 분류 - DSPy 버전"""

    result = classifier(
        column_name=column_name,
        table_name=table_name,
        data_type=data_type,
        sample_values=[str(s) for s in samples[:15]],
        distinct_ratio=None,
    )

    return {
        "semantic_role": result.semantic_role,
        "business_name": result.business_name,
        "domain_category": result.domain_category,
        "confidence": float(result.confidence),
        "reasoning": result.reasoning,
    }
```

### 5.3 기존 에이전트에서 사용

```python
# src/unified_pipeline/autonomous/agents/discovery.py 수정 예시

from src.common.utils.llm_instructor import call_llm_structured
from src.unified_pipeline.llm_schemas import FKAnalysisResult, ColumnTypeAnalysis

class DiscoveryAgent(AutonomousAgent):
    """Discovery 에이전트 - Instructor/DSPy 통합"""

    async def analyze_column_type(self, column_name: str, samples: list) -> ColumnTypeAnalysis:
        """컬럼 타입 분석 (Instructor 사용)"""
        return call_llm_structured(
            response_model=ColumnTypeAnalysis,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": f"Analyze column {column_name}: {samples[:20]}"}
            ],
            max_retries=2,
        )

    async def detect_fk(self, source: dict, target: dict) -> FKAnalysisResult:
        """FK 감지 (Instructor 사용)"""
        return call_llm_structured(
            response_model=FKAnalysisResult,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": f"Analyze FK: {source} -> {target}"}
            ],
        )
```

---

## 6. 마이그레이션 계획

### Phase 1: Instructor 도입 (1-2일)

1. `src/common/utils/llm_instructor.py` 생성
2. `src/unified_pipeline/llm_schemas.py` 생성
3. 기존 `extract_json_from_response()` 호출부 점진적 교체

### Phase 2: DSPy 도입 (2-3일)

1. `src/unified_pipeline/dspy_modules/` 디렉토리 생성
2. config.py, signatures.py, modules.py 생성
3. 핵심 분석 함수 DSPy 모듈로 교체

### Phase 3: 프롬프트 최적화 (선택)

1. 훈련 데이터 수집 (성공 케이스)
2. BootstrapFewShot으로 모듈 최적화
3. 최적화된 모듈 저장 및 로드

---

## 7. 테스트

### 7.1 Instructor 테스트

```python
# tests/test_instructor_integration.py

import pytest
from src.common.utils.llm_instructor import call_llm_structured
from src.unified_pipeline.llm_schemas import FKAnalysisResult

def test_fk_analysis_structured():
    """FK 분석이 올바른 타입으로 반환되는지 테스트"""
    result = call_llm_structured(
        response_model=FKAnalysisResult,
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": "Is customer_id in orders a FK to customers.id?"}
        ],
    )

    assert isinstance(result, FKAnalysisResult)
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning  # 비어있지 않음
```

### 7.2 DSPy 테스트

```python
# tests/test_dspy_modules.py

import pytest
from src.unified_pipeline.dspy_modules.config import configure_dspy
from src.unified_pipeline.dspy_modules.modules import ColumnSemanticClassifier

def test_column_classification():
    """컬럼 분류 모듈 테스트"""
    configure_dspy()
    classifier = ColumnSemanticClassifier()

    result = classifier(
        column_name="customer_id",
        table_name="orders",
        data_type="integer",
        sample_values=["1001", "1002", "1003"],
    )

    assert result.semantic_role in ["IDENTIFIER", "REFERENCE"]
    assert result.confidence > 0.5
```

---

## 8. 예상 효과

| 항목 | 현재 | 도입 후 |
|------|------|---------|
| JSON 파싱 성공률 | ~85% | 99%+ |
| 타입 안전성 | Dict[str, Any] | Pydantic Model |
| 재시도 로직 | 수동 구현 | 자동 (max_retries) |
| 프롬프트 관리 | f-string 하드코딩 | 선언적 Signature |
| 프롬프트 최적화 | 수동 튜닝 | 자동 컴파일 |

---

## 9. 참고 자료

- [Instructor Documentation](https://python.useinstructor.com/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Pydantic V2 Docs](https://docs.pydantic.dev/latest/)
