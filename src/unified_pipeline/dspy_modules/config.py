"""
DSPy 설정 및 LM 초기화

Letsur Gateway를 통한 OpenAI 호환 LM 설정입니다.

사용 예시:
    from src.unified_pipeline.dspy_modules.config import configure_dspy

    # 앱 시작 시 1회 호출
    configure_dspy(model="gpt-5.1")

    # 이후 DSPy 모듈 사용 가능
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Letsur AI Gateway 설정
LETSUR_BASE_URL = "https://gateway.letsur.ai/v1"
LETSUR_API_KEY = os.environ.get("LETSUR_API_KEY", "")

# 기본 모델
DEFAULT_DSPY_MODEL = "gpt-5.1"

# 전역 설정 상태
_dspy_configured = False


def get_dspy_lm(model: str = None, temperature: float = 0.2):
    """
    DSPy Language Model 인스턴스 생성

    Args:
        model: 모델명 (기본값: gpt-5.1)
        temperature: 온도 (기본값: 0.2)

    Returns:
        dspy.LM 인스턴스
    """
    try:
        import dspy
    except ImportError:
        logger.error("dspy-ai 패키지가 설치되지 않았습니다. pip install dspy-ai")
        raise

    model_name = model or DEFAULT_DSPY_MODEL

    # OpenAI 호환 API 사용
    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=LETSUR_BASE_URL,
        api_key=LETSUR_API_KEY,
        temperature=temperature,
    )

    logger.debug(f"DSPy LM created: {model_name}")
    return lm


def configure_dspy(
    model: str = None,
    temperature: float = 0.2,
    force: bool = False,
) -> None:
    """
    DSPy 전역 설정

    앱 시작 시 1회 호출하여 DSPy를 설정합니다.
    이후 모든 DSPy 모듈에서 이 설정을 사용합니다.

    Args:
        model: 사용할 모델 (기본값: gpt-5.1)
        temperature: 온도 (기본값: 0.2)
        force: 이미 설정되어 있어도 재설정할지 여부

    Example:
        # 앱 시작 시
        configure_dspy(model="gpt-5.1")

        # 이후 어디서든 DSPy 모듈 사용 가능
        from dspy_modules import ColumnSemanticClassifier
        classifier = ColumnSemanticClassifier()
    """
    global _dspy_configured

    if _dspy_configured and not force:
        logger.debug("DSPy already configured, skipping")
        return

    try:
        import dspy
    except ImportError:
        logger.error("dspy-ai 패키지가 설치되지 않았습니다. pip install dspy-ai")
        raise

    lm = get_dspy_lm(model, temperature)
    dspy.configure(lm=lm)

    _dspy_configured = True
    logger.info(f"DSPy configured with model: {model or DEFAULT_DSPY_MODEL}")


def is_dspy_configured() -> bool:
    """DSPy 설정 여부 확인"""
    return _dspy_configured


def reset_dspy_config() -> None:
    """DSPy 설정 리셋 (테스트용)"""
    global _dspy_configured
    _dspy_configured = False
    logger.debug("DSPy configuration reset")


def get_current_lm():
    """
    현재 설정된 DSPy LM 반환

    Returns:
        현재 설정된 dspy.LM 인스턴스

    Raises:
        RuntimeError: DSPy가 설정되지 않은 경우
    """
    import dspy

    if not _dspy_configured:
        raise RuntimeError("DSPy not configured. Call configure_dspy() first.")

    return dspy.settings.lm
