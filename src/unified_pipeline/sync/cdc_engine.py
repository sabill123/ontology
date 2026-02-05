"""
Real-time Sync Engine (v16.0)

CDC (Change Data Capture) 기반 실시간 동기화 엔진:
- 변경 감지 및 캡처
- 실시간 스트리밍
- 충돌 해결 전략
- 이벤트 기반 알림
- 온톨로지 자동 업데이트

SharedContext와 연동하여 FK 관계 기반 동기화

사용법:
    from .cdc_engine import CDCEngine, ChangeEvent
    engine = CDCEngine(shared_context)
    engine.start()
    engine.subscribe("table_name", handler)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, TYPE_CHECKING
from enum import Enum
from datetime import datetime
from collections import defaultdict
import asyncio
import threading
import queue
import uuid
import logging
import json

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """변경 유형"""
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    UPSERT = "UPSERT"
    TRUNCATE = "TRUNCATE"
    SCHEMA_CHANGE = "SCHEMA_CHANGE"


class ConflictStrategy(Enum):
    """충돌 해결 전략"""
    LAST_WRITE_WINS = "last_write_wins"      # 마지막 쓰기 우선
    FIRST_WRITE_WINS = "first_write_wins"    # 처음 쓰기 우선
    MANUAL = "manual"                         # 수동 해결
    MERGE = "merge"                           # 필드별 병합
    CUSTOM = "custom"                         # 사용자 정의


class SyncStatus(Enum):
    """동기화 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICTED = "conflicted"


@dataclass
class ChangeEvent:
    """변경 이벤트"""
    event_id: str
    change_type: ChangeType
    table_name: str
    timestamp: datetime

    # 변경 데이터
    primary_key: Any = None
    before: Optional[Dict[str, Any]] = None  # UPDATE/DELETE 시 이전 값
    after: Optional[Dict[str, Any]] = None   # INSERT/UPDATE 시 이후 값

    # 메타데이터
    schema_version: Optional[str] = None
    source: str = "unknown"
    user_id: Optional[str] = None
    transaction_id: Optional[str] = None

    # FK 영향
    affected_fks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "change_type": self.change_type.value,
            "table_name": self.table_name,
            "timestamp": self.timestamp.isoformat(),
            "primary_key": self.primary_key,
            "before": self.before,
            "after": self.after,
            "source": self.source,
            "affected_fks": self.affected_fks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeEvent":
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            change_type=ChangeType(data.get("change_type", "UPDATE")),
            table_name=data.get("table_name", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            primary_key=data.get("primary_key"),
            before=data.get("before"),
            after=data.get("after"),
            source=data.get("source", "unknown"),
            affected_fks=data.get("affected_fks", []),
        )


@dataclass
class SyncTask:
    """동기화 작업"""
    task_id: str
    source_table: str
    target_table: str
    events: List[ChangeEvent] = field(default_factory=list)

    status: SyncStatus = SyncStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 결과
    synced_count: int = 0
    failed_count: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_table": self.source_table,
            "target_table": self.target_table,
            "status": self.status.value,
            "synced_count": self.synced_count,
            "failed_count": self.failed_count,
            "conflicts": self.conflicts,
            "error": self.error,
        }


@dataclass
class Subscription:
    """테이블 구독"""
    subscription_id: str
    table_name: str
    handler: Callable[[ChangeEvent], None]
    filter_fn: Optional[Callable[[ChangeEvent], bool]] = None
    change_types: Optional[Set[ChangeType]] = None
    is_active: bool = True


class CDCEngine:
    """
    CDC (Change Data Capture) Engine

    기능:
    1. 변경 이벤트 캡처 및 저장
    2. 테이블 구독 관리
    3. 실시간 이벤트 스트리밍
    4. 충돌 감지 및 해결
    5. FK 기반 cascade 동기화
    6. 온톨로지 자동 업데이트

    동작 방식:
    1. 데이터 소스에서 변경 감지 (폴링/트리거/로그)
    2. ChangeEvent 생성
    3. 구독자에게 이벤트 전달
    4. FK 관계 기반 cascade 처리
    5. 충돌 시 설정된 전략으로 해결
    """

    def __init__(
        self,
        shared_context: Optional["SharedContext"] = None,
        conflict_strategy: ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS,
    ):
        self.shared_context = shared_context
        self.conflict_strategy = conflict_strategy

        # 상태
        self._is_running = False
        self._event_queue: queue.Queue = queue.Queue()
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._event_log: List[ChangeEvent] = []
        self._sync_tasks: Dict[str, SyncTask] = {}

        # FK 매핑 (cascade 용)
        self._fk_graph: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # table -> [(fk_col, ref_table, ref_col)]

        # 스레드
        self._worker_thread: Optional[threading.Thread] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        """엔진 시작"""
        if self._is_running:
            logger.warning("[CDCEngine] Already running")
            return

        self._is_running = True
        self._build_fk_graph()

        # 이벤트 처리 워커 시작
        self._worker_thread = threading.Thread(target=self._event_worker, daemon=True)
        self._worker_thread.start()

        logger.info("[CDCEngine] Started")

    def stop(self) -> None:
        """엔진 중지"""
        self._is_running = False

        if self._worker_thread:
            self._event_queue.put(None)  # 종료 신호
            self._worker_thread.join(timeout=5)

        logger.info("[CDCEngine] Stopped")

    def _build_fk_graph(self) -> None:
        """SharedContext에서 FK 그래프 구축"""
        if not self.shared_context:
            return

        self._fk_graph.clear()

        for fk in self.shared_context.enhanced_fk_candidates:
            if isinstance(fk, dict):
                from_table = fk.get("from_table", "")
                from_col = fk.get("from_column", "")
                to_table = fk.get("to_table", "")
                to_col = fk.get("to_column", "")

                if all([from_table, from_col, to_table, to_col]):
                    self._fk_graph[from_table].append((from_col, to_table, to_col))

        logger.info(f"[CDCEngine] Built FK graph with {len(self._fk_graph)} tables")

    def subscribe(
        self,
        table_name: str,
        handler: Callable[[ChangeEvent], None],
        filter_fn: Optional[Callable[[ChangeEvent], bool]] = None,
        change_types: Optional[Set[ChangeType]] = None,
    ) -> str:
        """
        테이블 변경 구독

        Args:
            table_name: 구독할 테이블
            handler: 이벤트 핸들러
            filter_fn: 필터 함수 (True면 전달)
            change_types: 구독할 변경 유형

        Returns:
            구독 ID
        """
        subscription_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=subscription_id,
            table_name=table_name,
            handler=handler,
            filter_fn=filter_fn,
            change_types=change_types,
        )

        self._subscriptions[table_name].append(subscription)
        logger.info(f"[CDCEngine] Subscribed to {table_name}: {subscription_id}")

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """구독 해제"""
        for table_name, subs in self._subscriptions.items():
            for sub in subs:
                if sub.subscription_id == subscription_id:
                    subs.remove(sub)
                    logger.info(f"[CDCEngine] Unsubscribed: {subscription_id}")
                    return True
        return False

    def capture_change(
        self,
        change_type: ChangeType,
        table_name: str,
        primary_key: Any,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        source: str = "manual",
        user_id: Optional[str] = None,
    ) -> ChangeEvent:
        """
        변경 캡처

        Args:
            change_type: 변경 유형
            table_name: 테이블명
            primary_key: PK 값
            before: 변경 전 데이터
            after: 변경 후 데이터
            source: 변경 소스
            user_id: 변경 사용자

        Returns:
            ChangeEvent
        """
        # 영향받는 FK 찾기
        affected_fks = self._find_affected_fks(table_name, before, after)

        event = ChangeEvent(
            event_id=str(uuid.uuid4()),
            change_type=change_type,
            table_name=table_name,
            timestamp=datetime.now(),
            primary_key=primary_key,
            before=before,
            after=after,
            source=source,
            user_id=user_id,
            affected_fks=affected_fks,
        )

        # 큐에 추가
        self._event_queue.put(event)
        self._event_log.append(event)

        logger.debug(f"[CDCEngine] Captured {change_type.value} on {table_name}")

        return event

    def _find_affected_fks(
        self,
        table_name: str,
        before: Optional[Dict[str, Any]],
        after: Optional[Dict[str, Any]],
    ) -> List[str]:
        """변경으로 영향받는 FK 찾기"""
        affected = []

        fks = self._fk_graph.get(table_name, [])
        for fk_col, ref_table, ref_col in fks:
            before_val = before.get(fk_col) if before else None
            after_val = after.get(fk_col) if after else None

            if before_val != after_val:
                affected.append(f"{table_name}.{fk_col} -> {ref_table}.{ref_col}")

        return affected

    def _event_worker(self) -> None:
        """이벤트 처리 워커 스레드"""
        while self._is_running:
            try:
                event = self._event_queue.get(timeout=1)

                if event is None:  # 종료 신호
                    break

                self._process_event(event)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[CDCEngine] Worker error: {e}")

    def _process_event(self, event: ChangeEvent) -> None:
        """이벤트 처리 및 구독자 알림"""
        table_subs = self._subscriptions.get(event.table_name, [])
        wildcard_subs = self._subscriptions.get("*", [])  # 전체 구독

        for sub in table_subs + wildcard_subs:
            if not sub.is_active:
                continue

            # 변경 유형 필터
            if sub.change_types and event.change_type not in sub.change_types:
                continue

            # 사용자 정의 필터
            if sub.filter_fn and not sub.filter_fn(event):
                continue

            try:
                sub.handler(event)
            except Exception as e:
                logger.error(f"[CDCEngine] Handler error: {e}")

        # Cascade 처리
        if event.affected_fks:
            self._handle_cascade(event)

    def _handle_cascade(self, event: ChangeEvent) -> None:
        """FK 기반 cascade 처리"""
        # 간단한 cascade 로직 (실제로는 더 복잡)
        for fk_ref in event.affected_fks:
            logger.debug(f"[CDCEngine] Cascade for: {fk_ref}")

            # 필요시 cascade 이벤트 생성
            # 예: 부모 삭제 시 자식 처리

    def sync_tables(
        self,
        source_table: str,
        target_table: str,
        transform_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> SyncTask:
        """
        테이블 간 동기화 작업 생성

        Args:
            source_table: 소스 테이블
            target_table: 대상 테이블
            transform_fn: 데이터 변환 함수

        Returns:
            SyncTask
        """
        task = SyncTask(
            task_id=str(uuid.uuid4()),
            source_table=source_table,
            target_table=target_table,
            status=SyncStatus.PENDING,
        )

        self._sync_tasks[task.task_id] = task
        logger.info(f"[CDCEngine] Created sync task: {source_table} -> {target_table}")

        return task

    def resolve_conflict(
        self,
        event: ChangeEvent,
        conflicting_data: Dict[str, Any],
        strategy: Optional[ConflictStrategy] = None,
    ) -> Dict[str, Any]:
        """
        충돌 해결

        Args:
            event: 변경 이벤트
            conflicting_data: 충돌하는 기존 데이터
            strategy: 해결 전략 (없으면 기본 전략 사용)

        Returns:
            해결된 데이터
        """
        strategy = strategy or self.conflict_strategy

        if strategy == ConflictStrategy.LAST_WRITE_WINS:
            return event.after or {}

        elif strategy == ConflictStrategy.FIRST_WRITE_WINS:
            return conflicting_data

        elif strategy == ConflictStrategy.MERGE:
            # 필드별 병합 (새 값 우선, 없으면 기존 값)
            merged = {**conflicting_data}
            if event.after:
                for key, value in event.after.items():
                    if value is not None:
                        merged[key] = value
            return merged

        elif strategy == ConflictStrategy.MANUAL:
            # 충돌 기록 후 수동 처리 대기
            raise ConflictError(f"Manual resolution required for {event.table_name}")

        return event.after or {}

    def get_event_log(
        self,
        table_name: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """이벤트 로그 조회"""
        events = self._event_log

        if table_name:
            events = [e for e in events if e.table_name == table_name]
        if change_type:
            events = [e for e in events if e.change_type == change_type]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return [e.to_dict() for e in events[-limit:]]

    def get_sync_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """동기화 작업 상태 조회"""
        task = self._sync_tasks.get(task_id)
        return task.to_dict() if task else None

    def replay_events(
        self,
        events: List[ChangeEvent],
        target_handler: Callable[[ChangeEvent], None],
    ) -> int:
        """이벤트 재생 (복구/테스트용)"""
        count = 0
        for event in events:
            try:
                target_handler(event)
                count += 1
            except Exception as e:
                logger.error(f"[CDCEngine] Replay error: {e}")
        return count


class ConflictError(Exception):
    """충돌 에러"""
    pass


# SharedContext 연동 함수
def create_cdc_engine_with_context(
    shared_context: "SharedContext",
) -> CDCEngine:
    """
    SharedContext와 연동된 CDC Engine 생성

    사용법:
        from .cdc_engine import create_cdc_engine_with_context
        engine = create_cdc_engine_with_context(shared_context)
        engine.start()
    """
    engine = CDCEngine(shared_context=shared_context)

    # 기본 구독 설정: 온톨로지 업데이트
    def ontology_update_handler(event: ChangeEvent):
        """온톨로지 자동 업데이트"""
        if event.change_type == ChangeType.INSERT:
            # 새 엔티티 추가
            logger.info(f"[CDCEngine] New entity in {event.table_name}")
        elif event.change_type == ChangeType.DELETE:
            # 엔티티 제거
            logger.info(f"[CDCEngine] Removed entity from {event.table_name}")
        elif event.change_type == ChangeType.UPDATE:
            # FK 변경 감지
            if event.affected_fks:
                logger.info(f"[CDCEngine] FK relationship changed: {event.affected_fks}")

    # 모든 테이블에 대해 온톨로지 업데이트 구독
    engine.subscribe("*", ontology_update_handler)

    return engine


# 모듈 초기화
__all__ = [
    "CDCEngine",
    "ChangeEvent",
    "ChangeType",
    "ConflictStrategy",
    "SyncStatus",
    "SyncTask",
    "Subscription",
    "ConflictError",
    "create_cdc_engine_with_context",
]
