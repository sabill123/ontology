# Ontoloty v1 시스템 아키텍처

> **최종 업데이트**: 2026-01-22
> **버전**: v1

## 개요

Ontoloty는 **엔터프라이즈 데이터 통합 및 온톨로지 자동 구축 플랫폼**입니다. 분산된 데이터 사일로에서 자동으로 엔티티를 발견하고, 관계를 추론하며, 비즈니스 인사이트를 생성합니다.

### 핵심 기능

- **자동 도메인 탐지**: 데이터 특성 기반 산업 도메인 자동 인식
- **Cross-Entity 상관관계 분석**: 팔란티어 스타일 예측 인사이트 생성
- **다중 에이전트 합의**: Byzantine Fault Tolerant 의사결정
- **Evidence Chain**: 블록체인 스타일 감사 추적
- **Enhanced FK Detection v1**: Multi-Signal + Direction + Semantic (Cross-dataset F1 91.7%)

---

## 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                     OntologyPlatform (Entry Point)               │
│                        unified_main.py                           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              AutonomousPipelineOrchestrator                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Phase 1   │  │   Phase 2   │  │   Phase 3   │              │
│  │  Discovery  │──▶│ Refinement │──▶│ Governance │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SharedContext                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Tables   │ │ Entities │ │ Concepts │ │ Insights │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │ Evidence │ │Knowledge │ │Governance│                        │
│  │  Chain   │ │  Graph   │ │Decisions │                        │
│  └──────────┘ └──────────┘ └──────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 디렉토리 구조

```
src/unified_pipeline/
├── unified_main.py              # 메인 엔트리포인트
├── autonomous/
│   ├── orchestrator.py          # 에이전트 오케스트레이터
│   ├── autonomous_pipeline.py   # 파이프라인 오케스트레이터
│   ├── shared_context.py        # 공유 컨텍스트 (메모리)
│   │
│   ├── agents/                  # 자율 에이전트들
│   │   ├── discovery.py         # Phase 1 에이전트
│   │   ├── refinement.py        # Phase 2 에이전트
│   │   └── governance.py        # Phase 3 에이전트
│   │
│   ├── analysis/                # 분석 모듈
│   │   ├── enhanced_fk_pipeline.py       # v1 Enhanced FK Pipeline
│   │   ├── fk_signal_scorer.py           # v1 Multi-Signal Scoring
│   │   ├── fk_direction_analyzer.py      # v1 Probabilistic Direction
│   │   ├── semantic_entity_resolver.py   # v1 Semantic Entity Resolution
│   │   ├── enhanced_fk_detector.py       # Legacy FK 탐지
│   │   ├── cross_entity_correlation.py   # Cross-Entity 상관분석
│   │   ├── business_insights.py          # 비즈니스 인사이트
│   │   ├── causal_impact.py              # 인과관계 분석
│   │   ├── anomaly_detection.py          # 이상 탐지
│   │   ├── tda.py                        # 위상 데이터 분석
│   │   └── ...
│   │
│   └── todo/                    # Todo 기반 태스크 관리
│       ├── models.py            # Todo 데이터 모델
│       └── manager.py           # Todo 매니저
│
├── common/
│   ├── utils/
│   │   └── job_logger.py        # 작업 로깅 및 아티팩트 저장
│   └── llm/
│       └── client.py            # LLM 클라이언트
│
└── domain/                      # 도메인 탐지
    └── detector.py              # 자동 도메인 탐지기
```

---

## 파이프라인 실행 흐름

### 1. 초기화

```python
# CLI 실행
python -m src.unified_pipeline.unified_main ./data/your_data_folder/

# 프로그래매틱 실행
from src.unified_pipeline.unified_main import OntologyPlatform

platform = OntologyPlatform()
result = platform.run("./data/your_data_folder/")
```

### 2. 3단계 파이프라인

```
Phase 1: Discovery (데이터 이해)
    ├── DataAnalystAgent: LLM 기반 전체 데이터 분석
    ├── TDAExpertAgent: 위상 데이터 분석 (Betti Numbers)
    ├── SchemaAnalystAgent: 스키마 프로파일링
    ├── ValueMatcherAgent: 값 기반 매칭
    ├── EntityClassifierAgent: 엔티티 분류
    └── RelationshipDetectorAgent: 관계 탐지 (Enhanced FK Detection v1)

Phase 2: Refinement (온톨로지 정제)
    ├── OntologyArchitectAgent: 온톨로지 설계
    ├── ConflictResolverAgent: 충돌 해결
    ├── QualityJudgeAgent: 품질 평가
    └── SemanticValidatorAgent: 의미 검증

Phase 3: Governance (거버넌스)
    ├── GovernanceStrategistAgent: 전략 수립
    ├── ActionPrioritizerAgent: 액션 우선순위화
    ├── RiskAssessorAgent: 리스크 평가
    └── PolicyGeneratorAgent: 정책 생성
```

### 3. 출력 결과

```
output/{scenario_name}/
├── *_context.json               # Full context with relationships
├── *_entities.json              # Discovered entities
├── *_summary.json               # Execution summary
└── *_stats.json                 # Statistics
```

---

## 핵심 컴포넌트

### SharedContext (공유 메모리)

모든 에이전트가 공유하는 중앙 메모리 시스템:

```python
SharedContext:
    tables: Dict[str, TableInfo]              # 테이블 메타데이터
    unified_entities: List[UnifiedEntity]     # 통합 엔티티
    ontology_concepts: Dict[str, OntologyConcept]  # 온톨로지 개념
    concept_relationships: List[ConceptRelationship]
    knowledge_graph: KnowledgeGraph           # RDF 스타일 지식 그래프
    business_insights: List[BusinessInsight]  # 비즈니스 인사이트
    governance_decisions: List[GovernanceDecision]
    evidence_chain: List[EvidenceBlock]       # 감사 추적
    domain_context: DomainContext             # 자동 탐지된 도메인
```

### Evidence Chain

블록체인 스타일의 감사 추적 시스템:

```python
EvidenceBlock:
    block_id: str
    previous_hash: str
    agent_id: str
    phase: str
    evidence_type: str
    content: Dict
    timestamp: datetime
    hash: str
```

### Todo 시스템

DAG 기반 태스크 의존성 관리:

```python
PipelineTodo:
    todo_id: str
    phase: str
    name: str
    status: TodoStatus  # PENDING → READY → IN_PROGRESS → COMPLETED
    priority: Priority  # CRITICAL > HIGH > MEDIUM > LOW
    dependencies: List[str]
    assigned_agent_id: Optional[str]
```

---

## 에이전트 시스템

### 에이전트 종료 모드

```python
AgentTerminateMode:
    GOAL          # 정상 완료 + 증거 기록됨
    NO_EVIDENCE   # 완료했으나 증거 미기록 (Silent Failure)
    TIMEOUT       # 시간 초과
    MAX_TURNS     # 반복 한도 도달
    ERROR         # 예외 발생
    CONSENSUS_FAILED  # 합의 실패
    BLOCKED       # 의존성 미충족
    ESCALATED     # 사람 검토 필요
```

### 완료 시그널

```python
CompletionSignal:
    [DONE]        # 분석 완료, 확신 있음
    [NEEDS_MORE]  # 추가 분석 필요
    [BLOCKED]     # 진행 불가 (데이터 부족)
    [ESCALATE]    # 사람 검토 권장
```

### 다중 에이전트 합의

- **Dempster-Shafer 융합**: 다중 에이전트 신뢰도 결합
- **Byzantine Fault Tolerant**: 악의적/오류 에이전트 감지
- **Smart Consensus**: 데이터 복잡도에 따른 합의 레벨 선택

---

## 분석 모듈

### Enhanced FK Detection v1

Multi-Signal + Direction + Semantic 통합 파이프라인:

```
┌─────────────────────────────────────────────────────────────────┐
│              Enhanced FK Detection Pipeline v1                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Solution 1: Multi-Signal FK Scoring (FKSignalScorer)     │  │
│  │  ├── Signal 1: Value Overlap (25%)                        │  │
│  │  ├── Signal 2: Cardinality Pattern (20%)                  │  │
│  │  ├── Signal 3: Referential Integrity (25%)                │  │
│  │  ├── Signal 4: Naming Semantics (15%)                     │  │
│  │  └── Signal 5: Structural Pattern (15%)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Solution 2: Probabilistic Direction (FKDirectionAnalyzer)│  │
│  │  ├── Evidence 1: Uniqueness                               │  │
│  │  ├── Evidence 2: Containment                              │  │
│  │  ├── Evidence 3: Row Count                                │  │
│  │  ├── Evidence 4: Column Naming                            │  │
│  │  └── Evidence 5: Entity Reference                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Solution 3: Semantic Resolution (SemanticEntityResolver) │  │
│  │  ├── Universal Synonym Groups                             │  │
│  │  ├── Abbreviation Expansion                               │  │
│  │  └── Compound Entity Patterns                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Final: EnhancedFKPipeline                                │  │
│  │  - Signal Weight: 50%                                     │  │
│  │  - Direction Weight: 25%                                  │  │
│  │  - Semantic Weight: 25%                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Cross-dataset Validation Results:
┌───────────────────┬───────────┬──────────┬──────────┐
│ Dataset           │ Precision │ Recall   │ F1 Score │
├───────────────────┼───────────┼──────────┼──────────┤
│ marketing_silo    │ 100.0%    │ 100.0%   │ 100.0%   │
│ airport_silo      │ 90.9%     │ 76.9%    │ 83.3%    │
├───────────────────┼───────────┼──────────┼──────────┤
│ Cross-dataset Avg │ 95.5%     │ 88.5%    │ 91.7%    │
└───────────────────┴───────────┴──────────┴──────────┘
```

### Cross-Entity Correlation

팔란티어 스타일의 Cross-Entity 예측 인사이트 생성:

```python
CrossEntityCorrelationAnalyzer:
    - compute_cross_table_correlations()  # 테이블 간 상관계수
    - detect_threshold_effects()          # 임계값 효과 탐지
    - generate_palantir_insights()        # 자연어 인사이트 생성
```

### 분석 모듈 목록

| 모듈 | 파일 | 기능 |
|------|------|------|
| **Enhanced FK Pipeline v1** | `analysis/enhanced_fk_pipeline.py` | **Multi-Signal + Direction + Semantic** |
| FK Signal Scorer v1 | `analysis/fk_signal_scorer.py` | 5가지 신호 기반 FK 점수 |
| FK Direction Analyzer v1 | `analysis/fk_direction_analyzer.py` | 확률적 방향 분석 |
| Semantic Entity Resolver v1 | `analysis/semantic_entity_resolver.py` | 의미론적 엔티티 해석 |
| Enhanced FK Detector | `analysis/enhanced_fk_detector.py` | Legacy FK 탐지 |
| TDA | `analysis/tda.py` | 위상 데이터 분석 (Betti Numbers, Persistence) |
| Anomaly Detection | `analysis/anomaly_detection.py` | 통계/ML 기반 이상 탐지 |
| Causal Impact | `analysis/causal_impact.py` | 인과관계 분석 |
| Business Insights | `analysis/business_insights.py` | KPI, 패턴, 이상치 인사이트 |
| Semantic Reasoner | `analysis/semantic_reasoner.py` | RDFS/OWL 추론 |
| Graph Embedding | `analysis/graph_embedding.py` | Node2Vec, 커뮤니티 탐지 |
| Cross-Entity Correlation | `analysis/cross_entity_correlation.py` | Cross-Entity 상관분석 |

---

## 설정

### PipelineConfig

```python
PipelineConfig:
    phases: ["discovery", "refinement", "governance"]
    max_concurrent_agents: 5
    enable_continuation: True
    auto_detect_domain: True
    llm_model: "gpt-4o-mini"  # 또는 "claude-opus-4-5-20251101"
```

### 환경 변수

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

---

## 참고 문서

- [Phase 1: Discovery 명세](./PHASE_1_DISCOVERY_SPEC.md)
- [Phase 2: Refinement 명세](./PHASE_2_REFINEMENT_SPEC.md)
- [Phase 3: Governance 명세](./PHASE_3_GOVERNANCE_SPEC.md)
- [알고리즘 및 수학적 기반](./ALGORITHMS_AND_MATHEMATICS.md)
