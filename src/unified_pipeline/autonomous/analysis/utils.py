"""
Analysis Utilities

공통 유틸리티 함수
- JSON 파싱
- 재시도 로직
- 데이터 변환
"""

import asyncio
import json
import logging
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _repair_truncated_json(json_str: str) -> str:
    """잘린 JSON 문자열 복구: 열린 문자열/괄호를 닫음"""
    # 1. 마지막 완전한 값 이후의 불완전한 부분을 제거
    # 열린 문자열 닫기: 이스케이프되지 않은 따옴표 수가 홀수이면 닫기
    in_string = False
    last_complete_idx = 0
    i = 0
    while i < len(json_str):
        c = json_str[i]
        if c == '\\' and in_string:
            i += 2  # 이스케이프 문자 건너뛰기
            continue
        if c == '"':
            if in_string:
                in_string = False
                last_complete_idx = i
            else:
                in_string = True
        elif not in_string and c in '{}[]:,':
            last_complete_idx = i
        i += 1

    if in_string:
        # 문자열 중간에서 잘림 → 마지막 열린 따옴표 이전까지의 완전한 항목 찾기
        # 현재 열린 문자열을 닫고 이후 괄호 추가
        json_str = json_str[:i] + '"'

    # 2. 마지막 불완전한 key-value 쌍 제거 (trailing comma 처리)
    json_str = re.sub(r',\s*$', '', json_str.rstrip())

    # 3. 누락된 닫는 괄호 추가
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')
    json_str += '}' * max(0, open_braces)
    json_str += ']' * max(0, open_brackets)

    return json_str


def parse_llm_json(
    response: str,
    key: Optional[str] = None,
    default: Any = None,
) -> Any:
    """
    LLM 응답에서 JSON 파싱

    다양한 형태의 LLM 응답 처리:
    - ```json ... ``` 블록
    - 직접 JSON
    - 마크다운 내 JSON

    Args:
        response: LLM 응답 문자열
        key: 추출할 키 (None이면 전체 반환)
        default: 파싱 실패 시 기본값

    Returns:
        파싱된 데이터
    """
    if not response:
        return default

    # 1. ```json ... ``` 블록 추출
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_block_pattern, response)

    for match in matches:
        try:
            data = json.loads(match.strip())
            return _ensure_return_type(data, default, key)
        except json.JSONDecodeError:
            continue

    # 2. 중괄호로 시작하는 JSON 직접 파싱
    try:
        # 첫 번째 { 와 마지막 } 사이 추출
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response[start:end + 1]
            data = json.loads(json_str)
            return _ensure_return_type(data, default, key)
    except json.JSONDecodeError:
        pass

    # 2.5. 불완전한 JSON 복구 시도 (잘린 응답 처리)
    try:
        start = response.find('{')
        if start != -1:
            json_str = _repair_truncated_json(response[start:])
            data = json.loads(json_str)
            logger.info("Recovered truncated JSON object via repair")
            return _ensure_return_type(data, default, key)
    except json.JSONDecodeError:
        pass

    # 3. 배열 형태 시도
    try:
        start = response.find('[')
        end = response.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = response[start:end + 1]
            data = json.loads(json_str)
            return _ensure_return_type(data, default, key)
    except json.JSONDecodeError:
        pass

    # 3.5. 잘린 배열 형태 복구 시도
    try:
        start = response.find('[')
        if start != -1:
            json_str = _repair_truncated_json(response[start:])
            data = json.loads(json_str)
            logger.info("Recovered truncated JSON array via repair")
            return _ensure_return_type(data, default, key)
    except json.JSONDecodeError:
        pass

    # 4. 전체 문자열 시도
    try:
        data = json.loads(response.strip())
        return _ensure_return_type(data, default, key)
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to parse JSON from response: {response[:200]}...")
    return default


def _ensure_return_type(data: Any, default: Any, key: Optional[str]) -> Any:
    """parse_llm_json 반환값의 타입을 default와 일치시킴"""
    if key:
        if isinstance(data, dict):
            return data.get(key, default)
        return default
    # default가 dict인데 파싱 결과가 dict가 아니면 default 반환
    if isinstance(default, dict) and not isinstance(data, dict):
        logger.warning(f"Expected dict but got {type(data).__name__}, returning default")
        return default
    return data


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    재시도 데코레이터 (지수 백오프)

    Args:
        max_retries: 최대 재시도 횟수
        base_delay: 기본 지연 시간 (초)
        max_delay: 최대 지연 시간 (초)
        exponential_base: 지수 기반
        exceptions: 재시도할 예외 타입

    Usage:
        @retry_with_backoff(max_retries=3)
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def merge_dicts(base: Dict, update: Dict, deep: bool = True) -> Dict:
    """
    딕셔너리 병합

    Args:
        base: 기본 딕셔너리
        update: 업데이트 딕셔너리
        deep: 깊은 병합 여부

    Returns:
        병합된 딕셔너리
    """
    result = base.copy()

    for key, value in update.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        elif deep and key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value

    return result


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    리스트를 청크로 분할

    Args:
        lst: 분할할 리스트
        chunk_size: 청크 크기

    Returns:
        청크 리스트
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_json_serialize(obj: Any, default: str = "null") -> str:
    """
    안전한 JSON 직렬화

    Args:
        obj: 직렬화할 객체
        default: 실패 시 기본값

    Returns:
        JSON 문자열
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"JSON serialization failed: {e}")
        return default


def truncate_for_llm(
    text: str,
    max_chars: int = 50000,
    suffix: str = "\n... (truncated)",
) -> str:
    """
    LLM 입력용 텍스트 자르기

    Args:
        text: 원본 텍스트
        max_chars: 최대 문자 수
        suffix: 자른 경우 추가할 접미사

    Returns:
        잘린 텍스트
    """
    if len(text) <= max_chars:
        return text

    return text[:max_chars - len(suffix)] + suffix


def extract_confidence(response: str) -> float:
    """
    LLM 응답에서 신뢰도 추출

    [CONFIDENCE: 0.XX] 패턴 찾기

    Args:
        response: LLM 응답

    Returns:
        신뢰도 (0.0-1.0)
    """
    pattern = r'\[CONFIDENCE:\s*([0-9.]+)\]'
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        try:
            confidence = float(match.group(1))
            return min(1.0, max(0.0, confidence))
        except ValueError:
            pass

    return 0.5  # 기본값


def format_analysis_results(
    results: Dict[str, Any],
    format_type: str = "summary",
) -> str:
    """
    분석 결과 포맷팅

    Args:
        results: 분석 결과
        format_type: "summary" | "detailed" | "json"

    Returns:
        포맷팅된 문자열
    """
    if format_type == "json":
        return json.dumps(results, indent=2, ensure_ascii=False, default=str)

    if format_type == "detailed":
        lines = []
        for key, value in results.items():
            if isinstance(value, dict):
                lines.append(f"## {key}")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"## {key} ({len(value)} items)")
                for i, item in enumerate(value[:5]):
                    lines.append(f"  {i + 1}. {item}")
                if len(value) > 5:
                    lines.append(f"  ... and {len(value) - 5} more")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    # Summary
    lines = []
    for key, value in results.items():
        if isinstance(value, (dict, list)):
            count = len(value) if isinstance(value, list) else len(value.keys())
            lines.append(f"{key}: {count} items")
        else:
            lines.append(f"{key}: {value}")
    return " | ".join(lines)


class AnalysisContext:
    """
    분석 컨텍스트 관리

    분석 작업 간 공유 상태 관리
    """

    def __init__(self):
        self.start_time: float = time.time()
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metrics: Dict[str, float] = {}

    def add_result(self, key: str, value: Any):
        """결과 추가"""
        self.results[key] = value

    def add_error(self, error: str):
        """에러 추가"""
        self.errors.append(error)
        logger.error(error)

    def add_warning(self, warning: str):
        """경고 추가"""
        self.warnings.append(warning)
        logger.warning(warning)

    def record_metric(self, name: str, value: float):
        """메트릭 기록"""
        self.metrics[name] = value

    def elapsed_time(self) -> float:
        """경과 시간"""
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "elapsed_time_seconds": round(self.elapsed_time(), 2),
        }


# === v14.0: Entity 이름 정규화 유틸리티 ===

def normalize_entity_name(name: str) -> str:
    """
    v14.0: Entity 이름을 정규화하여 매칭 용이하게 변환

    다양한 케이스 변환 처리:
    - snake_case: order_list → orderlist
    - PascalCase: OrderList → orderlist
    - camelCase: orderList → orderlist
    - kebab-case: order-list → orderlist
    - spaces: Order List → orderlist

    Args:
        name: 원본 Entity 이름

    Returns:
        정규화된 소문자 이름 (구분자 제거)
    """
    if not name:
        return ""

    # 소문자 변환
    result = name.lower()

    # 구분자 제거 (_, -, 공백)
    result = result.replace("_", "").replace("-", "").replace(" ", "")

    return result


def extract_entity_base_name(name: str) -> str:
    """
    v14.0: Entity 이름에서 핵심 단어 추출

    복합 이름에서 핵심 단어만 추출:
    - order_list → order
    - OrderList → order
    - customer_profiles → customer

    Args:
        name: 원본 Entity 이름

    Returns:
        핵심 단어 (첫 번째 의미있는 단어)
    """
    if not name:
        return ""

    # snake_case 분리
    if "_" in name:
        parts = name.split("_")
        return parts[0].lower() if parts else ""

    # PascalCase/camelCase 분리
    # 대문자 앞에 공백 삽입 후 분리
    import re
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)

    if words:
        return words[0].lower()

    return name.lower()


def entity_names_match(name1: str, name2: str, strict: bool = False) -> bool:
    """
    v14.0: 두 Entity 이름이 동일한 Entity를 가리키는지 판단

    다양한 표기법을 고려한 유연한 매칭:
    - "Order" == "order_list" → True (핵심 단어 매칭)
    - "OrderList" == "order_list" → True (정규화 매칭)
    - "Plant" == "PlantCode" → True (부분 매칭)

    Args:
        name1: 첫 번째 이름
        name2: 두 번째 이름
        strict: True면 정확 매칭만, False면 유연 매칭

    Returns:
        매칭 여부
    """
    if not name1 or not name2:
        return False

    # 정규화된 이름 비교
    norm1 = normalize_entity_name(name1)
    norm2 = normalize_entity_name(name2)

    if norm1 == norm2:
        return True

    if strict:
        return False

    # 핵심 단어 비교
    base1 = extract_entity_base_name(name1)
    base2 = extract_entity_base_name(name2)

    if base1 == base2:
        return True

    # 부분 매칭 (한쪽이 다른 쪽에 포함)
    if norm1 in norm2 or norm2 in norm1:
        return True

    if base1 in norm2 or base2 in norm1:
        return True

    return False


def find_matching_entity(
    target_name: str,
    entity_names: List[str],
    strict: bool = False,
) -> Optional[str]:
    """
    v14.0: Entity 목록에서 매칭되는 Entity 찾기

    Args:
        target_name: 찾으려는 Entity 이름
        entity_names: Entity 이름 목록
        strict: 엄격 매칭 여부

    Returns:
        매칭된 Entity 이름 또는 None
    """
    for entity_name in entity_names:
        if entity_names_match(target_name, entity_name, strict):
            return entity_name
    return None
