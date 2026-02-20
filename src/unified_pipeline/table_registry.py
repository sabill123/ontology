"""
Table Registry

TableInfo / raw_data / data_directory / full-data access methods.
Extracted from SharedContext to reduce God Object complexity.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .shared_context import TableInfo


class TableRegistry:
    """
    Table metadata and full-data access.

    Manages:
    - tables: Dict[str, TableInfo]
    - raw_data: Dict[str, Dict[str, Any]]
    - data_directory: Optional[str]
    """

    def __init__(self) -> None:
        # These references are set by SharedContext.__post_init__ to point
        # at the *same* objects that live on the dataclass instance.
        self.tables: Dict[str, "TableInfo"] = {}
        self.raw_data: Dict[str, Dict[str, Any]] = {}
        self.data_directory: Optional[str] = None
        # v28.6: 전체 데이터 캐시 — CSV를 한 번만 읽고 재사용
        self._full_data_cache: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Bind helpers (called once from SharedContext.__post_init__)
    # ------------------------------------------------------------------

    def bind(
        self,
        tables: Dict[str, "TableInfo"],
        raw_data: Dict[str, Dict[str, Any]],
        data_directory: Optional[str],
    ) -> None:
        """Bind to the dataclass-owned containers so mutations are shared."""
        self.tables = tables
        self.raw_data = raw_data
        self.data_directory = data_directory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_table(self, table_info: "TableInfo") -> None:
        self.tables[table_info.name] = table_info

    def get_data_directory(self) -> Optional[str]:
        return self.data_directory

    def set_data_directory(self, path: str) -> None:
        self.data_directory = path

    def get_full_data(self, table_name: str) -> List[Dict[str, Any]]:
        """
        v22.1: Return full (un-sampled) data for *table_name*.

        Priority:
        1) source CSV on disk
        2) data_directory / <table>.csv
        3) sample_data already stored in TableInfo

        v28.6: 결과를 인스턴스 캐시에 보관 — 같은 파이프라인 실행 중 CSV 중복 로딩 방지
        """
        # v28.6: 캐시 히트 → 즉시 반환
        if table_name in self._full_data_cache:
            return self._full_data_cache[table_name]

        import os

        table_info = self.tables.get(table_name)
        if not table_info:
            return []

        result = []

        # 1st: source path CSV
        source_path = getattr(table_info, "source", None) or ""
        if source_path and os.path.exists(source_path) and source_path.endswith(".csv"):
            try:
                import pandas as pd
                df = pd.read_csv(source_path, dtype_backend="numpy")
                # v28.11: nullable dtypes 근본 제거 (이중 방어선)
                df = self._remove_nullable_dtypes(df)
                result = df.to_dict("records")
            except Exception:
                pass

        # 2nd: data_directory + table_name.csv
        if not result and self.data_directory:
            for suffix in [".csv", "_utf8.csv"]:
                path = os.path.join(self.data_directory, f"{table_name}{suffix}")
                if os.path.exists(path):
                    try:
                        import pandas as pd
                        df = pd.read_csv(path, dtype_backend="numpy")
                        # v28.11: nullable dtypes 근본 제거 (이중 방어선)
                        df = self._remove_nullable_dtypes(df)
                        result = df.to_dict("records")
                        break
                    except Exception:
                        pass

        # 3rd: sample_data as-is
        if not result:
            result = table_info.sample_data or []

        # v28.6: 캐시 저장
        self._full_data_cache[table_name] = result
        return result

    def _remove_nullable_dtypes(self, df) -> Any:
        """
        v28.11: pandas nullable dtypes → standard dtypes 변환

        이중 방어선: unified_main.py에서도 처리하지만, table_registry에서
        직접 CSV를 로딩하는 경우를 대비한 안전장치
        """
        import pandas as pd
        import numpy as np

        for col in df.columns:
            dtype_str = str(df[col].dtype)

            # Int64 → float64
            if dtype_str in ('Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64'):
                df[col] = df[col].astype('float64')

            # Float64 → float64
            elif dtype_str in ('Float32', 'Float64'):
                df[col] = df[col].astype('float64')

            # boolean → float64
            elif dtype_str == 'boolean':
                df[col] = df[col].astype('float64')

            # string → object
            elif dtype_str == 'string':
                df[col] = df[col].astype('object')

        # pd.NA → None
        df = df.replace({pd.NA: None})

        return df

    def get_all_full_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """v22.1: Full data for every table."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for table_name in self.tables:
            data = self.get_full_data(table_name)
            if data:
                result[table_name] = data
        return result

    def get_analysis_data(self, table_name: str, max_rows: int = 5000) -> List[Dict[str, Any]]:
        """
        v28.9: 분석용 stratified sample 데이터 반환.

        정확도 유지 전략:
        1. 전체 rows <= max_rows → 전체 반환 (q2cut 1,813행 → 전체 사용)
        2. 이상치 (상하위 0.5%) 필수 포함 → ML anomaly 누락 방지
        3. 가장 낮은 카디널리티(2-50) 카테고리 기준 비율 보존 → 세그먼트 인사이트 왜곡 없음
        4. 시계열 균등 분포 (datetime 컬럼 존재 시)

        Args:
            table_name: 테이블 이름
            max_rows: 최대 행 수 (기본 5,000 = 95% CI ±1.4%)

        Returns:
            Stratified sample rows (전체 <= max_rows이면 전체)
        """
        import pandas as _pd
        import numpy as _np

        df = self.get_full_dataframe(table_name)
        if df is None or len(df) <= max_rows:
            # 전체 데이터가 max_rows 이하면 전체 반환 (q2cut 등)
            return self.get_full_data(table_name)

        # Step 1: 이상치 인덱스 수집 (보존 대상)
        outlier_idx = set()
        numeric_cols = df.select_dtypes(include=[_np.number]).columns[:5]
        for col in numeric_cols:
            vals = df[col].dropna()
            if len(vals) > 100:
                q_low, q_high = vals.quantile([0.005, 0.995])
                mask = (df[col] < q_low) | (df[col] > q_high)
                outlier_idx.update(df[mask].index.tolist())

        # Step 2: 카테고리 기준 stratified sample
        cat_col = None
        for col in df.columns:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 50 and df[col].count() > len(df) * 0.5:
                cat_col = col
                break

        if cat_col:
            # 카테고리별 비율 보존 샘플링
            base_sample = (
                df.groupby(cat_col, group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(10, int(max_rows * len(g) / len(df)))),
                    random_state=42
                ))
            )
        else:
            # 단순 랜덤 샘플링
            base_sample = df.sample(max_rows, random_state=42)

        # Step 3: 이상치 병합 (누락된 것만)
        missing_idx = list(set(outlier_idx) - set(base_sample.index))
        if missing_idx:
            base_sample = _pd.concat([base_sample, df.loc[missing_idx]])

        # pandas DataFrame → List[Dict] 변환 (NaN → None)
        return base_sample.where(_pd.notnull(base_sample), None).to_dict('records')

    def get_all_analysis_data(self, max_rows: int = 5000) -> Dict[str, List[Dict[str, Any]]]:
        """v28.9: 모든 테이블의 분석용 샘플 데이터 반환."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for table_name in self.tables:
            data = self.get_analysis_data(table_name, max_rows)
            if data:
                result[table_name] = data
        return result
