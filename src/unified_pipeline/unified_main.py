"""
Ontoloty - Unified Main System (v5.1)

통합된 메인 시스템 진입점
- 사용자는 데이터 업로드 + 시작 버튼만 누르면 됨
- 자동으로 전체 파이프라인 실행
- 실시간 이벤트 스트리밍
- 결과 자동 저장

출력 경로 구조:
- Logs/               → 시스템 로그, 진행현황 (간략한 상태 추적)
- data/storage/       → 상세 결과물 (JobLogger 관리)
  └── {job_id}/
      ├── artifacts/        → 최종 산출물 (ontology.json, knowledge_graph.json 등)
      ├── phase1_discovery/ → 에이전트 대화, 판단근거, LLM 응답
      ├── phase2_refinement/
      ├── phase3_governance/
      └── llm_calls/        → 모든 LLM 호출 기록

v5.1 변경사항:
- 출력 경로 구조 명확화 (Logs/ vs data/storage/)
- JobLogger가 data/storage/에 블랙박스 방지용 상세 기록 저장
- Job ID 형식: timestamp_domain_numbering (예: 20260113_143052_retail_001)

Usage:
    # Python API
    from src.unified_pipeline.unified_main import OntologyPlatform

    platform = OntologyPlatform()
    results = platform.run("./data/my_data/")

    # CLI
    python -m src.unified_pipeline.unified_main ./data/my_data/
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Generator, AsyncGenerator
from dataclasses import dataclass, field
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .autonomous_pipeline import AutonomousPipelineOrchestrator
from .shared_context import SharedContext

logger = logging.getLogger(__name__)


# === v5.0: JobLogger 헬퍼 함수 ===
def _get_job_logger():
    """JobLogger 싱글톤 인스턴스 반환"""
    try:
        from ..common.utils.job_logger import get_job_logger
        return get_job_logger()
    except ImportError:
        return None


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 실행 설정
    phases: List[str] = field(default_factory=lambda: ["discovery", "refinement", "governance"])
    max_concurrent_agents: int = 5
    enable_continuation: bool = True
    auto_detect_domain: bool = True

    # 출력 설정 (참고: 주요 산출물은 JobLogger가 data/storage/{job_id}/ 에 저장)
    # output_dir는 레거시 호환 및 추가 리포트용
    output_dir: str = "./data/storage"
    save_intermediate: bool = True
    generate_reports: bool = True

    # LLM 설정
    use_llm: bool = True
    llm_model: str = ""  # v21.0: empty → model_config default

    def __post_init__(self):
        if not self.llm_model:
            from .model_config import get_model, ModelType
            self.llm_model = get_model(ModelType.BALANCED)

    # 데이터 설정
    encoding: str = "utf-8"


@dataclass
class PipelineResult:
    """파이프라인 결과"""
    success: bool
    scenario_name: str
    domain_detected: str
    domain_confidence: float

    # 결과 통계
    tables_processed: int
    entities_found: int
    relationships_found: int
    insights_generated: int
    governance_actions: int

    # 상세 결과
    context: Optional[Dict[str, Any]] = None
    reports: Optional[Dict[str, str]] = None
    execution_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class OntologyPlatform:
    """
    Ontoloty 통합 플랫폼

    사용자가 데이터만 업로드하면 자동으로 전체 파이프라인을 실행합니다.

    Usage:
        platform = OntologyPlatform()

        # 방법 1: 디렉토리 경로로 실행
        results = platform.run("./data/my_data/")

        # 방법 2: DataFrame 딕셔너리로 실행
        results = platform.run({
            "customers": df_customers,
            "orders": df_orders,
        })

        # 방법 3: 비동기 실행 (실시간 이벤트)
        async for event in platform.run_async("./data/my_data/"):
            print(event)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        llm_client=None,
    ):
        """
        플랫폼 초기화

        Args:
            config: 파이프라인 설정
            llm_client: LLM 클라이언트 (None이면 자동 생성)
        """
        self.config = config or PipelineConfig()
        self.llm_client = llm_client

        # LLM 클라이언트 자동 생성
        if self.config.use_llm and self.llm_client is None:
            self.llm_client = self._create_llm_client()

        self._orchestrator: Optional[AutonomousPipelineOrchestrator] = None
        self._last_result: Optional[PipelineResult] = None

        logger.info("OntologyPlatform initialized")

    def _create_llm_client(self):
        """LLM 클라이언트 생성 (Letsur AI Gateway 사용)"""
        try:
            from ..common.utils.llm import get_openai_client
            return get_openai_client()
        except Exception as e:
            logger.warning(f"Failed to create LLM client: {e}")
            return None

    # === 메인 실행 메서드 ===

    def run(
        self,
        data_source: Union[str, Path, Dict[str, pd.DataFrame]],
        scenario_name: Optional[str] = None,
        domain_hint: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> PipelineResult:
        """
        파이프라인 동기 실행

        Args:
            data_source: 데이터 소스 (경로 또는 DataFrame 딕셔너리)
            scenario_name: 시나리오 이름 (자동 생성 가능)
            domain_hint: 도메인 힌트 (자동 탐지 가능)
            callback: 이벤트 콜백 함수 (optional)

        Returns:
            PipelineResult: 실행 결과
        """
        # 이벤트 루프 생성/가져오기
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 비동기 실행
        return loop.run_until_complete(
            self._run_async_internal(data_source, scenario_name, domain_hint, callback)
        )

    async def run_async(
        self,
        data_source: Union[str, Path, Dict[str, pd.DataFrame]],
        scenario_name: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        파이프라인 비동기 실행 (이벤트 스트리밍)

        Args:
            data_source: 데이터 소스
            scenario_name: 시나리오 이름
            domain_hint: 도메인 힌트

        Yields:
            실시간 이벤트
        """
        start_time = datetime.now()
        job_id = None
        job_logger = _get_job_logger()

        # 데이터 로드
        yield {"type": "loading_data", "message": "Loading data...", "timestamp": datetime.now().isoformat()}
        tables_data = self._load_data(data_source)

        yield {
            "type": "data_loaded",
            "tables": list(tables_data.keys()),
            "total_rows": sum(t.get("row_count", 0) for t in tables_data.values()),
            "timestamp": datetime.now().isoformat(),
        }

        # 시나리오 이름 생성
        if scenario_name is None:
            scenario_name = self._generate_scenario_name(data_source)

        # v5.0: Job 시작
        domain_for_job = domain_hint or "unknown"
        if job_logger:
            job_id = job_logger.start_job(
                domain=domain_for_job,
                input_tables=list(tables_data.keys()),
                config={
                    "scenario_name": scenario_name,
                    "phases": self.config.phases,
                    "max_concurrent_agents": self.config.max_concurrent_agents,
                    "use_llm": self.config.use_llm,
                    "llm_model": self.config.llm_model,
                },
            )
            yield {
                "type": "job_started",
                "job_id": job_id,
                "message": f"Job started: {job_id}",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # 오케스트레이터 생성
            self._orchestrator = AutonomousPipelineOrchestrator(
                scenario_name=scenario_name,
                domain_context=domain_hint,
                llm_client=self.llm_client,
                output_dir=self.config.output_dir,
                enable_continuation=self.config.enable_continuation,
                max_concurrent_agents=self.config.max_concurrent_agents,
                auto_detect_domain=self.config.auto_detect_domain,
            )

            # v10.0: 데이터 디렉토리 설정 (Cross-Entity Correlation 분석용)
            data_source_path = self._resolve_data_path(data_source)
            if data_source_path:
                self._orchestrator.shared_context.set_data_directory(str(data_source_path))
                logger.info(f"v10.0: Data directory set to {data_source_path}")

            # 파이프라인 실행
            async for event in self._orchestrator.run(tables_data, self.config.phases):
                yield event

            # 결과 생성
            execution_time = (datetime.now() - start_time).total_seconds()
            context = self._orchestrator.shared_context
            domain = context.get_domain()

            result = PipelineResult(
                success=True,
                scenario_name=scenario_name,
                domain_detected=domain.industry,
                domain_confidence=domain.industry_confidence,
                tables_processed=len(context.tables),
                entities_found=len([c for c in context.ontology_concepts if c.concept_type == "object_type"]) or len(context.unified_entities),
                relationships_found=len([c for c in context.ontology_concepts if c.concept_type == "link_type"]) or len(context.concept_relationships or []),
                insights_generated=len(context.unified_insights_pipeline or []),
                governance_actions=len(context.governance_decisions or []),
                context=context.to_dict() if self.config.save_intermediate else None,
                execution_time_seconds=execution_time,
            )

            self._last_result = result

            # 리포트 생성
            if self.config.generate_reports:
                reports = self._generate_reports(context, result)
                result.reports = reports

            # v5.0: Job 산출물 저장
            if job_logger and job_logger.is_active():
                # 온톨로지 저장
                if context.unified_entities:
                    job_logger.save_ontology({
                        "entities": [e.to_dict() if hasattr(e, 'to_dict') else e for e in context.unified_entities],
                        "relationships": context.concept_relationships or [],
                    })

                # 인사이트 저장
                if context.business_insights:
                    job_logger.save_insights(
                        [i.to_dict() if hasattr(i, 'to_dict') else i for i in context.business_insights]
                    )

                # 거버넌스 결정 저장
                if context.governance_decisions:
                    job_logger.save_governance_decisions(
                        [d.to_dict() if hasattr(d, 'to_dict') else d for d in context.governance_decisions]
                    )

                # Knowledge Graph 저장
                if hasattr(context, 'knowledge_graph') and context.knowledge_graph:
                    job_logger.save_knowledge_graph(
                        context.knowledge_graph.to_dict() if hasattr(context.knowledge_graph, 'to_dict')
                        else context.knowledge_graph
                    )

                # v10.0: Palantir-Style Predictive Insights 저장
                if hasattr(context, 'palantir_insights') and context.palantir_insights:
                    job_logger.save_palantir_insights(context.palantir_insights)

            yield {
                "type": "pipeline_complete",
                "result": {
                    "success": result.success,
                    "domain": result.domain_detected,
                    "entities": result.entities_found,
                    "relationships": result.relationships_found,
                    "insights": result.insights_generated,
                    "actions": result.governance_actions,
                    "execution_time": result.execution_time_seconds,
                    "job_id": job_id,
                },
                "timestamp": datetime.now().isoformat(),
            }

            # v5.0: Job 종료 (성공)
            if job_logger and job_logger.is_active():
                job_summary = job_logger.end_job(status="completed")
                yield {
                    "type": "job_completed",
                    "job_id": job_id,
                    "job_dir": job_summary.get("job_dir"),
                    "message": f"Job completed: {job_id}",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            # v5.0: Job 종료 (실패)
            if job_logger and job_logger.is_active():
                job_logger.log_error(f"Pipeline failed: {e}")
                job_summary = job_logger.end_job(status="failed")
                yield {
                    "type": "job_failed",
                    "job_id": job_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            raise

    async def _run_async_internal(
        self,
        data_source: Union[str, Path, Dict[str, pd.DataFrame]],
        scenario_name: Optional[str] = None,
        domain_hint: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> PipelineResult:
        """내부 비동기 실행"""
        result = None

        async for event in self.run_async(data_source, scenario_name, domain_hint):
            # 콜백 호출
            if callback:
                callback(event)

            # 로깅
            event_type = event.get("type", "unknown")
            if event_type == "pipeline_complete":
                result = event.get("result", {})
            elif event_type in ("phase_complete", "todo_complete"):
                logger.info(f"[{event_type}] {event.get('message', '')}")

        return self._last_result

    # === 데이터 로드 ===

    def _load_data(
        self,
        data_source: Union[str, Path, Dict[str, pd.DataFrame]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        데이터 소스에서 데이터 로드

        Args:
            data_source: 데이터 소스

        Returns:
            에이전트 시스템용 테이블 데이터
        """
        if isinstance(data_source, dict):
            # DataFrame 딕셔너리
            return self._convert_dataframes(data_source)

        path = Path(data_source)

        if not path.exists():
            raise ValueError(f"Data source not found: {path}")

        if path.is_file():
            # 단일 파일
            return self._load_single_file(path)
        elif path.is_dir():
            # 디렉토리 (여러 파일)
            return self._load_directory(path)
        else:
            raise ValueError(f"Invalid data source: {path}")

    def _load_single_file(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """단일 파일 로드"""
        df = self._read_file(path)
        if df is None:
            return {}

        table_name = path.stem
        return self._convert_dataframes({table_name: df})

    def _load_directory(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """디렉토리에서 모든 데이터 파일 로드"""
        supported_extensions = {'.csv', '.json', '.parquet', '.xlsx', '.xls'}

        dataframes = {}

        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                df = self._read_file(file_path)
                if df is not None:
                    dataframes[file_path.stem] = df

        # 하위 디렉토리도 탐색
        for subdir in path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                for file_path in subdir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        df = self._read_file(file_path)
                        if df is not None:
                            # 도메인/테이블 형태로 이름 지정
                            table_name = f"{subdir.name}/{file_path.stem}"
                            dataframes[table_name] = df

        return self._convert_dataframes(dataframes)

    def _read_file(self, path: Path) -> Optional[pd.DataFrame]:
        """파일 읽기 (v6.1: 주석/복잡한 헤더 자동 처리)"""
        try:
            suffix = path.suffix.lower()

            if suffix == '.csv':
                df = self._read_csv_smart(path)
            elif suffix == '.json':
                df = pd.read_json(path, encoding=self.config.encoding)
            elif suffix == '.parquet':
                df = pd.read_parquet(path)
            elif suffix in ('.xlsx', '.xls'):
                df = pd.read_excel(path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None

            # v28.11: nullable dtypes 근본 제거 + Phase 1 전처리
            if df is not None:
                df = self._preprocess_dataframe(df, path.stem)

            return df

        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return None

    def _read_csv_smart(self, path: Path) -> Optional[pd.DataFrame]:
        """
        v6.1: 스마트 CSV 읽기

        주석 행(#), 복잡한 헤더, 불규칙한 컬럼 수, 다양한 인코딩 등을 자동 처리합니다.
        """
        # v6.2: 인코딩 자동 감지 (한국어 포함)
        encoding = self._detect_encoding(path)

        # 먼저 주석/메타데이터 행 감지
        skip_rows = self._detect_header_start(path, encoding)

        # 1차 시도: skiprows로 주석 건너뛰고 읽기
        if skip_rows > 0:
            try:
                df = pd.read_csv(path, encoding=encoding, skiprows=skip_rows, on_bad_lines='skip')
                if len(df) > 0 and len(df.columns) > 1:  # 최소 2개 이상 컬럼
                    df = self._flatten_columns(df)
                    logger.info(f"CSV loaded with skiprows={skip_rows}: {path.name}")
                    return df
            except Exception:
                pass

        # 2차 시도: 기본 읽기
        try:
            df = pd.read_csv(path, encoding=encoding)
            if len(df) > 0 and len(df.columns) > 1:
                return df
        except Exception:
            pass

        # 3차 시도: comment='#' 사용
        try:
            df = pd.read_csv(path, encoding=encoding, comment='#')
            if len(df) > 0 and len(df.columns) > 1:
                logger.info(f"CSV loaded with comment='#': {path.name}")
                return df
        except Exception:
            pass

        # 4차 시도: on_bad_lines='skip'으로 불규칙한 행 무시
        try:
            df = pd.read_csv(path, encoding=encoding, on_bad_lines='skip')
            if len(df) > 0 and len(df.columns) > 1:
                logger.info(f"CSV loaded with on_bad_lines='skip': {path.name}")
                return df
        except Exception:
            pass

        # 5차 시도: skiprows + on_bad_lines 조합 (더 공격적으로)
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                skiprows=skip_rows,
                on_bad_lines='skip'
            )
            if len(df) > 0:
                df = self._flatten_columns(df)
                logger.info(f"CSV loaded with combined strategy: {path.name}")
                return df
        except Exception as e:
            logger.warning(f"All CSV read strategies failed for {path.name}: {e}")

        return None

    def _preprocess_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        v28.11: DataFrame 전처리 — nullable dtypes 제거 + Phase 1 데이터 검증/수정

        목적:
        1. pandas nullable dtypes (Int64, Float64, pd.NA) 제거 → 'numerator' 에러 근본 해결
        2. engagement_rate 재계산 (> 100% 오류 수정)
        3. Negative values 제거
        4. Hard rules 적용

        Args:
            df: 원본 DataFrame
            table_name: 테이블 이름 (로깅용)

        Returns:
            전처리된 DataFrame
        """
        import numpy as np

        original_rows = len(df)
        changes_log = []

        # ======================================================================
        # Stage 1: nullable dtypes 제거 (v28.14 개선 - hasattr 기반 감지)
        # ======================================================================
        # pandas nullable dtypes (Int64, Float64, pd.NA) → standard dtypes (int64, float64, None)
        # v28.14: str() 비교 대신 hasattr(dtype, 'na_value')로 정확히 감지
        for col in df.columns:
            col_dtype = df[col].dtype

            # Nullable dtypes는 na_value 속성을 가짐
            if hasattr(col_dtype, 'na_value'):
                dtype_name = str(col_dtype)

                # Integer/UInt nullable → float64 (NaN 보존)
                if col_dtype.kind in ('i', 'u'):
                    df[col] = df[col].astype('float64')
                    changes_log.append(f"  {col}: {dtype_name} → float64")

                # Float nullable → float64
                elif col_dtype.kind == 'f':
                    df[col] = df[col].astype('float64')
                    changes_log.append(f"  {col}: {dtype_name} → float64")

                # Boolean nullable → float64 (0/1)
                elif col_dtype.kind == 'b':
                    df[col] = df[col].astype('float64')
                    changes_log.append(f"  {col}: {dtype_name} → float64")

                # String nullable → object
                else:
                    df[col] = df[col].astype('object')
                    changes_log.append(f"  {col}: {dtype_name} → object")

        # pd.NA → None, np.nan 유지
        df = df.replace({pd.NA: None})

        # ======================================================================
        # Stage 2: Hard Rules — Auto Correction (Phase 1)
        # ======================================================================

        # Rule 1: engagement_rate > 100% → 재계산
        # 원인: like_rate (%) + comment_rate (%) 잘못된 합산
        # 수정: (likes + comments) / views * 100
        if 'engagement_rate' in df.columns:
            # engagement_rate가 100 초과인 행 찾기
            invalid_mask = df['engagement_rate'] > 100
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                # 재계산 가능한지 확인 (likes, comments, views 컬럼 존재)
                can_recalc = all(col in df.columns for col in ['likes', 'comments', 'views'])

                if can_recalc:
                    # 올바른 engagement_rate 재계산
                    df.loc[invalid_mask, 'engagement_rate'] = (
                        (df.loc[invalid_mask, 'likes'] + df.loc[invalid_mask, 'comments'])
                        / df.loc[invalid_mask, 'views'] * 100
                    )
                    changes_log.append(f"  ✓ engagement_rate > 100% 재계산: {invalid_count}행")
                else:
                    # 재계산 불가능 → 100으로 cap
                    df.loc[invalid_mask, 'engagement_rate'] = 100.0
                    changes_log.append(f"  ⚠ engagement_rate > 100% → 100 cap: {invalid_count}행 (재계산 불가)")

        # Rule 2: Negative values → NaN 처리 (수치형 컬럼만)
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # 의미상 음수가 불가능한 컬럼들 (counts, rates, duration 등)
            if any(keyword in col.lower() for keyword in ['count', 'views', 'likes', 'comment', 'duration', 'rate', 'score', 'length']):
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    df.loc[negative_mask, col] = np.nan
                    changes_log.append(f"  ✓ {col} 음수 → NaN: {negative_count}행")

        # Rule 3: views=0인데 engagement > 0 → engagement=0 (봇 조작 가능성)
        if 'views' in df.columns and 'engagement_rate' in df.columns:
            suspicious_mask = (df['views'] == 0) & (df['engagement_rate'] > 0)
            suspicious_count = suspicious_mask.sum()
            if suspicious_count > 0:
                df.loc[suspicious_mask, 'engagement_rate'] = 0.0
                changes_log.append(f"  ⚠ views=0 but engagement>0 → 0: {suspicious_count}행 (봇 조작 의심)")

        # ======================================================================
        # Stage 3: Extreme Outliers — Statistical Validation (선택적)
        # ======================================================================
        # TODO (Phase 2): LLM semantic judgment로 "봇" vs "바이럴" 구분
        # - High views + zero engagement → SUSPICIOUS (LLM 판단)
        # - High views + high engagement → VIRAL (KEEP)

        # ======================================================================
        # Logging
        # ======================================================================
        rows_removed = original_rows - len(df)
        if changes_log:
            logger.info(f"[v28.11 Preprocess] {table_name}")
            for log in changes_log:
                logger.info(log)
            if rows_removed > 0:
                logger.info(f"  Rows removed: {rows_removed} ({rows_removed/original_rows*100:.1f}%)")

        return df

    def _detect_encoding(self, path: Path) -> str:
        """
        v6.2: 파일 인코딩 자동 감지

        한국어(EUC-KR, CP949), UTF-8, UTF-8-BOM 등 다양한 인코딩 처리
        """
        # 시도할 인코딩 순서 (한국어 우선)
        encodings_to_try = [
            'utf-8',
            'utf-8-sig',  # UTF-8 with BOM
            'cp949',      # 한국어 Windows
            'euc-kr',     # 한국어 Legacy
            'latin-1',    # ISO-8859-1 fallback
        ]

        # 파일 시작 부분 읽어서 인코딩 테스트
        for encoding in encodings_to_try:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    # 처음 5줄 읽어서 한국어 포함 여부 확인
                    sample = f.read(2000)
                    # 읽기 성공하면 이 인코딩 사용
                    logger.debug(f"Encoding detected for {path.name}: {encoding}")
                    return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                continue

        # 모두 실패하면 설정값 사용
        logger.warning(f"Encoding detection failed for {path.name}, using default: {self.config.encoding}")
        return self.config.encoding

    def _detect_header_start(self, path: Path, encoding: str) -> int:
        """주석/메타데이터 행을 건너뛰고 실제 헤더 위치 감지"""
        skip_rows = 0

        try:
            with open(path, 'r', encoding=encoding) as f:
                lines = []
                for i, line in enumerate(f):
                    if i > 30:  # 최대 30행까지만 검사
                        break
                    lines.append(line)

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # 빈 행 건너뛰기
                    if not stripped:
                        skip_rows = i + 1
                        continue

                    # 주석 행 건너뛰기 (# // -- 로 시작)
                    if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('--'):
                        skip_rows = i + 1
                        continue

                    # CSV 헤더로 보이는지 확인 (쉼표가 있고 여러 필드가 있음)
                    comma_count = stripped.count(',')
                    if comma_count >= 2:
                        # 이게 헤더일 가능성이 높음 - 여기서 멈춤
                        break
                    elif comma_count == 0 and i < 15:
                        # 쉼표가 없으면 메타데이터/제목일 가능성
                        skip_rows = i + 1
                        continue
                    else:
                        # 쉼표가 1개면 일단 헤더로 간주
                        break

        except Exception:
            pass

        return skip_rows

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """멀티레벨 컬럼 또는 중복 컬럼명 정리"""
        # 멀티레벨 인덱스인 경우
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]

        # 첫 행이 실제 헤더인지 확인 (두 번째 헤더 행 처리)
        # 첫 행의 값들이 문자열이고 컬럼명과 다르면 헤더로 간주
        if len(df) > 0:
            first_row = df.iloc[0]
            # 첫 행이 모두 문자열이고, 숫자가 아닌 경우
            is_header_row = True
            non_empty_count = 0
            for val in first_row:
                if pd.notna(val):
                    val_str = str(val)
                    # 숫자로 변환 가능하면 데이터 행
                    try:
                        float(val_str.replace(',', ''))
                        is_header_row = False
                        break
                    except ValueError:
                        non_empty_count += 1

            # 첫 행이 헤더로 보이면 컬럼명으로 사용하고 제거
            if is_header_row and non_empty_count >= 2:
                # 첫 행의 값들을 새 컬럼명으로
                new_cols = []
                for i, (old_col, new_col) in enumerate(zip(df.columns, first_row)):
                    if pd.notna(new_col) and str(new_col).strip():
                        new_cols.append(str(new_col).strip())
                    else:
                        new_cols.append(str(old_col))
                df.columns = new_cols
                df = df.iloc[1:].reset_index(drop=True)

        # 중복 컬럼명 처리
        cols = df.columns.tolist()
        seen = {}
        new_cols = []
        for col in cols:
            col_str = str(col)
            if col_str in seen:
                seen[col_str] += 1
                new_cols.append(f"{col_str}_{seen[col_str]}")
            else:
                seen[col_str] = 0
                new_cols.append(col_str)
        df.columns = new_cols

        # 집계 행 제거 (첫 행이 총합/합계인 경우)
        if len(df) > 0:
            first_val = str(df.iloc[0, 0]) if pd.notna(df.iloc[0, 0]) else ''
            if '총' in first_val or '합계' in first_val or first_val == '':
                # 두 번째 컬럼도 확인
                second_val = str(df.iloc[0, 1]) if len(df.columns) > 1 and pd.notna(df.iloc[0, 1]) else ''
                if second_val == '' or '총' in second_val:
                    df = df.iloc[1:].reset_index(drop=True)

        return df

    def _convert_dataframes(
        self,
        dataframes: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, Any]]:
        """DataFrame을 에이전트 시스템 포맷으로 변환"""
        tables_data = {}

        for table_name, df in dataframes.items():
            # 도메인 추론
            domain = self._infer_domain(table_name, df)

            # 컬럼 정보 추출 (v27.11: 캐싱으로 중복 연산 제거)
            columns = []
            null_mask = df.isnull()  # 1회만 계산
            for col in df.columns:
                try:
                    col_series = df[col]
                    col_null = null_mask[col]
                    non_null = col_series[~col_null]

                    # dict/list 타입은 nunique 계산 불가
                    sample_val = non_null.iloc[0] if len(non_null) > 0 else None
                    is_complex_type = isinstance(sample_val, (dict, list))

                    if is_complex_type:
                        unique_count = -1  # 복잡 타입
                        sample_values = []
                    else:
                        # v27.12: 고카디널리티 컬럼 최적화 (ID 등)
                        # 샘플 100개로 유니크 비율 체크 → >95%면 전체 계산 스킵
                        if len(non_null) > 100:
                            sample_unique_ratio = non_null.head(100).nunique() / 100
                            if sample_unique_ratio > 0.95:
                                unique_count = len(non_null)  # 거의 유니크로 추정
                            else:
                                unique_count = int(col_series.nunique())
                        else:
                            unique_count = int(col_series.nunique())
                        sample_values = non_null.head(5).tolist()

                    col_info = {
                        "name": col,
                        "type": str(col_series.dtype),
                        "nullable": bool(col_null.any()),
                        "unique_count": unique_count,
                        "null_count": int(col_null.sum()),
                        "sample_values": sample_values,
                    }
                    columns.append(col_info)
                except Exception as e:
                    # 안전하게 처리
                    col_info = {
                        "name": col,
                        "type": str(df[col].dtype),
                        "nullable": True,
                        "unique_count": -1,
                        "null_count": int(df[col].isnull().sum()),
                        "sample_values": [],
                    }
                    columns.append(col_info)

            # Primary Key 추론
            primary_keys = self._infer_primary_keys(df, columns)

            # Foreign Key 추론
            foreign_keys = self._infer_foreign_keys(table_name, df, columns)

            # v28.0: Stratified Sampling — 카테고리 분포 보존 (Palantir 방식)
            # 기존: df.head(1000) 무작위 선두 1000행 → 카테고리 편향 가능
            # 변경: 카테고리 비율 보존 + 이상치 필수 포함 → 정확도 유지
            MAX_SAMPLE_ROWS = 1000
            if len(df) <= MAX_SAMPLE_ROWS:
                df_sample = df
            else:
                try:
                    # Step 1: 이상치 인덱스 수집 (수치형 컬럼 상하위 0.5%)
                    outlier_idx = set()
                    for _ocol in df.select_dtypes(include=['number']).columns[:5]:
                        _ovals = df[_ocol].dropna()
                        if len(_ovals) > 100:
                            _ql, _qh = _ovals.quantile(0.005), _ovals.quantile(0.995)
                            outlier_idx.update(df[(_ocol < _ql) | (_ocol > _qh) if False else df[_ocol].between(_ql, _qh) == False].index.tolist())
                    # Step 2: 카테고리 컬럼 기준 stratified sample
                    _cat_col = None
                    for _cc in df.columns:
                        _nu = df[_cc].nunique()
                        if 2 <= _nu <= 50 and df[_cc].count() > len(df) * 0.5:
                            _cat_col = _cc
                            break
                    if _cat_col:
                        df_sample = (
                            df.groupby(_cat_col, group_keys=False)
                            .apply(lambda g: g.sample(
                                min(len(g), max(5, int(MAX_SAMPLE_ROWS * len(g) / len(df)))),
                                random_state=42
                            ))
                        ).head(MAX_SAMPLE_ROWS)
                    else:
                        df_sample = df.sample(MAX_SAMPLE_ROWS, random_state=42)
                    # Step 3: 이상치 병합 (누락분만)
                    _missing = list(outlier_idx - set(df_sample.index))
                    if _missing:
                        df_sample = pd.concat([df_sample, df.loc[_missing[:50]]])
                except Exception:
                    df_sample = df.head(MAX_SAMPLE_ROWS)
            sample_data = df_sample.where(pd.notnull(df_sample), None).to_dict('records')

            tables_data[table_name] = {
                "source": str(table_name),
                "domain": domain,
                "columns": columns,
                "row_count": len(df),
                "sample_data": sample_data,
                "metadata": {
                    "loaded_at": datetime.now().isoformat(),
                    "column_count": len(columns),
                },
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            }

        return tables_data

    def _infer_domain(self, table_name: str, df: pd.DataFrame) -> str:
        """테이블에서 도메인 추론"""
        name_lower = table_name.lower()
        cols_lower = [c.lower() for c in df.columns]

        # 도메인별 키워드
        domain_keywords = {
            "healthcare": ["patient", "hospital", "diagnosis", "medication", "encounter", "provider"],
            "manufacturing": ["machine", "sensor", "maintenance", "failure", "iot", "production"],
            "supply_chain": ["order", "warehouse", "freight", "carrier", "port", "plant", "product"],
            "finance": ["transaction", "account", "balance", "payment", "invoice", "credit"],
            "retail": ["customer", "sale", "store", "inventory", "price", "discount"],
        }

        for domain, keywords in domain_keywords.items():
            # 테이블 이름 매칭
            if any(kw in name_lower for kw in keywords):
                return domain
            # 컬럼 이름 매칭
            if any(any(kw in col for kw in keywords) for col in cols_lower):
                return domain

        return "general"

    def _infer_primary_keys(
        self,
        df: pd.DataFrame,
        columns: List[Dict],
    ) -> List[str]:
        """Primary Key 추론"""
        primary_keys = []

        for col_info in columns:
            col_name = col_info["name"]
            col_lower = col_name.lower()

            # ID 패턴
            if col_lower.endswith("_id") or col_lower == "id":
                # 고유성 체크
                if col_info["unique_count"] == df[col_name].count():
                    primary_keys.append(col_name)
                    break

        return primary_keys

    def _infer_foreign_keys(
        self,
        table_name: str,
        df: pd.DataFrame,
        columns: List[Dict],
    ) -> List[Dict]:
        """Foreign Key 추론"""
        foreign_keys = []

        for col_info in columns:
            col_name = col_info["name"]
            col_lower = col_name.lower()

            # FK 패턴: xxx_id, xxx_code
            if (col_lower.endswith("_id") or col_lower.endswith("_code")) and col_lower not in ("id", f"{table_name.lower()}_id"):
                # 대상 테이블 추론
                parts = col_lower.replace("_id", "").replace("_code", "")

                foreign_keys.append({
                    "column": col_name,
                    "references_table": parts,
                    "references_column": col_name,
                    "inferred": True,
                })

        return foreign_keys

    def _generate_scenario_name(self, data_source) -> str:
        """시나리오 이름 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            return f"{path.stem}_{timestamp}"
        else:
            return f"analysis_{timestamp}"

    def _resolve_data_path(self, data_source) -> Optional[Path]:
        """
        v10.0: 데이터 소스에서 실제 데이터 디렉토리 경로 추출

        Cross-Entity Correlation 분석에서 실제 CSV 파일을 로드하기 위해 사용
        """
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if path.exists():
                if path.is_dir():
                    return path
                elif path.is_file():
                    return path.parent
        return None

    # === 리포트 생성 ===

    def _generate_reports(
        self,
        context: SharedContext,
        result: PipelineResult,
    ) -> Dict[str, str]:
        """결과 리포트 생성"""
        reports = {}
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Summary Report (JSON)
        summary_path = output_dir / f"{result.scenario_name}_summary.json"
        summary = {
            "scenario_name": result.scenario_name,
            "domain": result.domain_detected,
            "domain_confidence": result.domain_confidence,
            "execution_time_seconds": result.execution_time_seconds,
            "statistics": {
                "tables_processed": result.tables_processed,
                "entities_found": result.entities_found,
                "relationships_found": result.relationships_found,
                "insights_generated": result.insights_generated,
                "governance_actions": result.governance_actions,
            },
            "generated_at": datetime.now().isoformat(),
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        reports["summary"] = str(summary_path)

        # 2. Full Context (JSON)
        if result.context:
            context_path = output_dir / f"{result.scenario_name}_context.json"
            with open(context_path, 'w', encoding='utf-8') as f:
                json.dump(result.context, f, indent=2, ensure_ascii=False, default=str)
            reports["context"] = str(context_path)

        # 3. Entities Report
        if context.unified_entities:
            entities_path = output_dir / f"{result.scenario_name}_entities.json"
            entities_data = [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "canonical_name": e.canonical_name,
                    "source_tables": e.source_tables,
                    "confidence": e.confidence,
                }
                for e in context.unified_entities
            ]
            with open(entities_path, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)
            reports["entities"] = str(entities_path)

        # 4. Governance Actions Report
        if context.action_backlog:
            actions_path = output_dir / f"{result.scenario_name}_actions.json"
            with open(actions_path, 'w', encoding='utf-8') as f:
                json.dump(context.action_backlog, f, indent=2, ensure_ascii=False, default=str)
            reports["actions"] = str(actions_path)

        logger.info(f"Reports generated in {output_dir}")
        return reports

    # === 편의 메서드 ===

    def get_last_result(self) -> Optional[PipelineResult]:
        """마지막 실행 결과 반환"""
        return self._last_result

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        if self._orchestrator:
            return self._orchestrator.get_status()
        return {"status": "idle"}


# === CLI 진입점 ===

def main():
    """CLI 메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ontoloty - Enterprise Ontology Construction Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 디렉토리의 모든 데이터 처리
    python -m src.unified_pipeline.unified_main ./data/my_data/

    # 특정 도메인 지정
    python -m src.unified_pipeline.unified_main ./data/healthcare/ --domain healthcare

    # LLM 없이 실행
    python -m src.unified_pipeline.unified_main ./data/my_data/ --no-llm
        """
    )

    parser.add_argument(
        "data_source",
        help="데이터 소스 경로 (파일 또는 디렉토리)",
    )
    parser.add_argument(
        "--domain", "-d",
        help="도메인 힌트 (자동 탐지 가능)",
    )
    parser.add_argument(
        "--name", "-n",
        help="시나리오 이름",
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/pipeline_results",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="LLM 비활성화",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["discovery", "refinement", "governance"],
        help="실행할 Phase 목록",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력",
    )

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 콘솔 핸들러
    logging.basicConfig(
        level=log_level,
        format=log_format,
    )

    # 파일 핸들러 추가 (Logs/ 폴더에 시스템 로그 저장)
    logs_dir = Path("./Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(logs_dir / log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"System logs → {logs_dir / log_filename}")
    logger.info(f"Detailed outputs → data/storage/{{job_id}}/")

    # 설정 생성
    config = PipelineConfig(
        phases=args.phases,
        output_dir=args.output,
        use_llm=not args.no_llm,
    )

    # 플랫폼 생성 및 실행
    platform = OntologyPlatform(config=config)

    print(f"\n{'='*60}")
    print("Ontoloty - Enterprise Ontology Construction Platform")
    print(f"{'='*60}\n")
    print(f"Data Source: {args.data_source}")
    print(f"Domain Hint: {args.domain or 'Auto-detect'}")
    print(f"System Logs: ./Logs/")
    print(f"Detailed Outputs: ./data/storage/{{job_id}}/")
    print(f"Phases: {', '.join(args.phases)}")
    print(f"LLM: {'Enabled' if config.use_llm else 'Disabled'}")
    print(f"\n{'='*60}\n")

    def event_callback(event):
        event_type = event.get("type", "unknown")

        if event_type == "domain_detection_complete":
            print(f"✓ Domain Detected: {event.get('domain')} (confidence: {event.get('confidence', 0):.2f})")
        elif event_type == "phase_start":
            print(f"\n▶ Starting Phase: {event.get('phase', '').upper()}")
        elif event_type == "phase_complete":
            print(f"✓ Phase Complete: {event.get('phase', '').upper()}")
        elif event_type == "todo_complete":
            print(f"  • {event.get('agent', 'Agent')}: {event.get('task_type', 'task')}")
        elif event_type == "pipeline_complete":
            result = event.get("result", {})
            print(f"\n{'='*60}")
            print("Pipeline Complete!")
            print(f"{'='*60}")
            print(f"  Domain: {result.get('domain', 'Unknown')}")
            print(f"  Entities: {result.get('entities', 0)}")
            print(f"  Relationships: {result.get('relationships', 0)}")
            print(f"  Insights: {result.get('insights', 0)}")
            print(f"  Actions: {result.get('actions', 0)}")
            print(f"  Execution Time: {result.get('execution_time', 0):.2f}s")
        elif event_type == "error":
            print(f"✗ Error: {event.get('message', 'Unknown error')}")

    try:
        result = platform.run(
            data_source=args.data_source,
            scenario_name=args.name,
            domain_hint=args.domain,
            callback=event_callback,
        )

        if result and result.success:
            print(f"\n✓ Results saved to: {args.output}")
            return 0
        else:
            print(f"\n✗ Pipeline failed")
            return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
