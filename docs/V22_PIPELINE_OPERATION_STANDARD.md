# v22.0 파이프라인 정상 동작 기준서

## 작성일: 2026-02-04
## 버전: v22.0

---

## 1. 모듈 사용 현황 요약

### ✅ 정상 동작 모듈 (14/15)

| # | 모듈명 | 클래스 | 호출 위치 | 상태 |
|---|--------|--------|----------|------|
| 1 | version_manager | VersionManager | pre/post hooks | ✅ 활성 |
| 2 | osdk_client | OSDKClient | pre hooks | ✅ 활성 |
| 3 | semantic_searcher | SemanticSearcher | pre hooks + agents | ✅ 활성 |
| 4 | remediation_engine | RemediationEngine | pre hooks | ✅ 활성 |
| 5 | decision_explainer | DecisionExplainer | post hooks | ✅ 활성 |
| 6 | report_generator | ReportGenerator | post hooks | ✅ 활성 |
| 7 | whatif_analyzer | WhatIfAnalyzer | post hooks + agents | ✅ 활성 |
| 8 | writeback_engine | WritebackEngine | post hooks | ✅ 활성 |
| 9 | branch_manager | BranchManager | post hooks | ✅ 활성 |
| 10 | streaming_reasoner | StreamingReasoner | post hooks | ✅ 활성 |
| 11 | tool_registry | ToolRegistry | post hooks | ✅ 활성 |
| 12 | nl2sql_engine | NL2SQLEngine | on-demand | ✅ 활성 |
| 13 | tool_selector | ToolSelector | post hooks | ✅ 활성 |
| 14 | chain_executor | ChainExecutor | post hooks | ✅ 활성 |

### ✅ 추가 활성 모듈 (1/15)

| # | 모듈명 | 클래스 | 호출 위치 | 상태 |
|---|--------|--------|----------|------|
| 15 | calibrator | ConfidenceCalibrator | base.py:calibrate_confidence() → discovery.py:3116, refinement.py:2768, governance.py:492,577 | ✅ 활성 |

**Note**: ConfidenceCalibrator는 `base.py:852-884`의 `calibrate_confidence()` 메서드를 통해 간접 호출됩니다.
에이전트가 `calibrate_confidence(raw_confidence)` 호출 시 내부적으로 calibrator 인스턴스를 사용합니다.

---

## 2. 파이프라인 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           v22.0 파이프라인 실행 흐름                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ [0] 초기화 (Initialization)                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ • AutonomousPipelineOrchestrator 생성                                        │
│ • SharedContext 초기화                                                       │
│ • v17 서비스 13개 등록 (enable_v17_services=True)                            │
│ • AgentOrchestrator 생성                                                     │
│ • EvidenceChain 초기화                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ [1] 도메인 감지 (Domain Detection) - Optional                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ • DomainDetector.detect_domain()                                             │
│ • 테이블/컬럼명 기반 키워드 매칭                                               │
│ • 도메인별 워크플로우/대시보드 추천                                            │
│                                                                              │
│ 출력:                                                                        │
│   - detected_domain: string (healthcare, manufacturing, finance, etc.)       │
│   - domain_confidence: float (0.0 ~ 1.0)                                     │
│   - recommended_workflows: list                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ [2] PRE-PIPELINE HOOKS (_run_pre_pipeline_hooks)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 2-1. VersionManager.commit()                                            │  │
│ │      "Initial data load" 버전 생성                                       │  │
│ │      테이블별 체크섬, 행 수 스냅샷                                         │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 2-2. OSDKClient.sync_from_shared_context()                              │  │
│ │      SharedContext → Palantir OSDK 동기화                                │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 2-3. SemanticSearcher.index_table()                                     │  │
│ │      텍스트 컬럼 임베딩 생성 및 인덱싱                                     │  │
│ │      검색 가능한 의미 검색 준비                                           │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 2-4. RemediationEngine.remediate()                                      │  │
│ │      데이터 품질 이슈 자동 교정                                           │  │
│ │      NULL 처리, 형식 표준화 등                                            │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ [3] CORE PIPELINE EXECUTION (agent_orchestrator.run_pipeline)                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ PHASE 1: DISCOVERY (발견 단계)                                          │  │
│ ├─────────────────────────────────────────────────────────────────────────┤  │
│ │                                                                         │  │
│ │ 에이전트 7개:                                                            │  │
│ │ ┌─────────────────┬─────────────────┬─────────────────────────────────┐ │  │
│ │ │ Agent           │ Type            │ 역할                             │ │  │
│ │ ├─────────────────┼─────────────────┼─────────────────────────────────┤ │  │
│ │ │ Data Analyst    │ data_analyst    │ 데이터 이해 및 패턴 분석          │ │  │
│ │ │ TDA Expert      │ tda_expert      │ 위상 데이터 분석 (Betti numbers) │ │  │
│ │ │ Schema Analyst  │ schema_analyst  │ 스키마 구조 분석                  │ │  │
│ │ │ Value Matcher   │ value_matcher   │ 값 중복/오버랩 분석               │ │  │
│ │ │ Homeomorphism   │ homeo_detector  │ 위상동형 스키마 매핑              │ │  │
│ │ │ Relationship    │ rel_detector    │ 엔티티 간 관계 추론               │ │  │
│ │ │ Entity Class.   │ entity_class    │ 엔티티 유형 분류                  │ │  │
│ │ └─────────────────┴─────────────────┴─────────────────────────────────┘ │  │
│ │                                                                         │  │
│ │ 분석 모듈:                                                               │  │
│ │ • TDAAnalyzer - Betti numbers, persistence entropy 계산                 │  │
│ │ • SchemaAnalyzer - 컬럼/테이블 구조 분석                                 │  │
│ │ • SimilarityAnalyzer - 값 유사도 분석                                   │  │
│ │ • UniversalFKDetector - FK 후보 탐지                                    │  │
│ │ • SemanticFKDetector - 의미 기반 FK 탐지                                │  │
│ │ • GraphEmbeddingAnalyzer - Node2Vec 임베딩                              │  │
│ │                                                                         │  │
│ │ 출력:                                                                    │  │
│ │ • tda_signatures: {table: {betti_numbers, persistence_entropy}}         │  │
│ │ • schema_analysis: {tables: [...], columns: [...]}                      │  │
│ │ • homeomorphisms: [{source, target, similarity}]                        │  │
│ │ • fk_candidates: [{from, to, confidence}]                               │  │
│ │ • entity_classifications: [{entity, type, confidence}]                  │  │
│ │                                                                         │  │
│ │ Consensus Mode: full_skip (Discovery는 합의 생략)                        │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ PHASE 2: REFINEMENT (정제 단계)                                         │  │
│ ├─────────────────────────────────────────────────────────────────────────┤  │
│ │                                                                         │  │
│ │ 에이전트 4개:                                                            │  │
│ │ ┌─────────────────┬─────────────────┬─────────────────────────────────┐ │  │
│ │ │ Agent           │ Type            │ 역할                             │ │  │
│ │ ├─────────────────┼─────────────────┼─────────────────────────────────┤ │  │
│ │ │ Ontology Arch.  │ ontology_arch   │ 온톨로지 개념 설계               │ │  │
│ │ │ Quality Judge   │ quality_judge   │ 데이터 품질 평가                  │ │  │
│ │ │ Conflict Resol. │ conflict_res    │ 충돌 해결                        │ │  │
│ │ │ Semantic Enrich │ semantic_enrich │ 의미 보강                        │ │  │
│ │ └─────────────────┴─────────────────┴─────────────────────────────────┘ │  │
│ │                                                                         │  │
│ │ OWL2/SHACL 처리:                                                        │  │
│ │ • OWL2Reasoner - 온톨로지 추론, axiom 발견                              │  │
│ │ • SHACLValidator - SHACL 제약 검증                                     │  │
│ │                                                                         │  │
│ │ 출력:                                                                    │  │
│ │ • ontology_concepts: [{name, type, definition}]                         │  │
│ │ • concept_relationships: [{source, target, type}]                       │  │
│ │ • owl2_reasoning_results: {axioms_discovered, inferences}               │  │
│ │ • shacl_validation_results: {is_conformant, violations}                 │  │
│ │ • quality_scores: {table: score}                                        │  │
│ │                                                                         │  │
│ │ Consensus Mode: selective (품질/온톨로지 결정에만 합의)                   │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ PHASE 3: GOVERNANCE (거버넌스 단계)                                     │  │
│ ├─────────────────────────────────────────────────────────────────────────┤  │
│ │                                                                         │  │
│ │ 에이전트 4개:                                                            │  │
│ │ ┌─────────────────┬─────────────────┬─────────────────────────────────┐ │  │
│ │ │ Agent           │ Type            │ 역할                             │ │  │
│ │ ├─────────────────┼─────────────────┼─────────────────────────────────┤ │  │
│ │ │ Gov. Strategist │ gov_strategist  │ 거버넌스 전략 수립               │ │  │
│ │ │ Action Prior.   │ action_prior    │ 액션 우선순위 결정               │ │  │
│ │ │ Risk Assessor   │ risk_assessor   │ 리스크 평가                      │ │  │
│ │ │ Final Approver  │ final_approver  │ 최종 승인                        │ │  │
│ │ └─────────────────┴─────────────────┴─────────────────────────────────┘ │  │
│ │                                                                         │  │
│ │ 비즈니스 인사이트 분석:                                                  │  │
│ │ • BusinessInsightsAnalyzer - KPI/리스크/이상탐지/운영 패턴              │  │
│ │ • CausalImpactAnalyzer - 인과관계 분석                                  │  │
│ │ • CrossEntityCorrelationAnalyzer - 엔티티 간 상관관계                   │  │
│ │ • AnomalyDetector - 앙상블 이상탐지 (IsolationForest, LOF, Z-score)    │  │
│ │                                                                         │  │
│ │ v17 서비스 활용:                                                         │  │
│ │ • SemanticSearcher - 관련 엔티티 검색                                   │  │
│ │ • WhatIfAnalyzer - 시나리오 시뮬레이션                                  │  │
│ │ • Calibrator - 신뢰도 보정 (✅ calibrate_confidence()로 활용)          │  │
│ │                                                                         │  │
│ │ 출력:                                                                    │  │
│ │ • governance_decisions: [{decision_type, status, subject}]              │  │
│ │ • business_insights: [{type, title, priority, confidence}]              │  │
│ │ • pending_actions: [{action_type, priority, description}]               │  │
│ │ • risk_assessments: [{risk_type, severity, mitigation}]                 │  │
│ │                                                                         │  │
│ │ Consensus Mode: selective + debate (필요시 구조화된 토론)                │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ [4] POST-PIPELINE HOOKS (_run_post_pipeline_hooks)                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-1. DecisionExplainer.explain_decision()                               │  │
│ │      거버넌스 결정에 대한 XAI 설명 생성                                   │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-2. ReportGenerator.generate()                                         │  │
│ │      - Executive Summary 리포트                                          │  │
│ │      - Pipeline Status 리포트                                            │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-3. WhatIfAnalyzer.create_scenario() + simulate()                      │  │
│ │      베이스라인 시나리오 시뮬레이션                                       │  │
│ │      (⚠️ ML 예측이 아닌 규칙 기반 시뮬레이션)                            │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-4. WritebackEngine.execute()                                          │  │
│ │      승인된 거버넌스 결정을 소스 시스템에 쓰기                            │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-5. BranchManager.create_branch()                                      │  │
│ │      파이프라인 결과 브랜치 생성                                          │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-6. StreamingReasoner.process_batch()                                  │  │
│ │      실시간 인사이트 생성                                                 │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-7. ToolRegistry + ToolSelector + ChainExecutor                        │  │
│ │      - 사용 가능한 분석 도구 조회                                         │  │
│ │      - 관련 도구 자동 선택                                               │  │
│ │      - 도구 체인 실행                                                    │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ 4-8. VersionManager.commit() + tag()                                    │  │
│ │      최종 버전 커밋 및 태깅 ("Pipeline completed")                        │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ [5] 결과 출력                                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ • SharedContext에 모든 결과 저장                                             │
│ • output_dir에 JSON 파일로 저장:                                             │
│   - shared_context.json                                                      │
│   - summary.json                                                             │
│   - evidence_chain.json                                                      │
│   - phase_results.json                                                       │
│   - agent_results.json                                                       │
│   - council_records.json                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 정상 동작 체크리스트

### 3.1 도메인 감지 (Domain Detection)

| 체크 항목 | 기대값 | 확인 방법 |
|----------|--------|----------|
| 도메인 감지 완료 | event: `domain_detection_complete` | 로그 확인 |
| 신뢰도 > 0.5 | confidence >= 0.5 | `detected_domain_confidence` |
| 키워드 매칭 | 최소 3개 이상 | 도메인별 키워드 목록 |

### 3.2 Discovery Phase

| 체크 항목 | 기대값 | 확인 방법 |
|----------|--------|----------|
| 7개 에이전트 실행 | 모두 SUCCESS | `[AGENT_TASK] SUCCESS` 로그 |
| TDA 시그니처 생성 | `tda_signatures` 비어있지 않음 | SharedContext |
| FK 후보 탐지 | `enhanced_fk_candidates` 비어있지 않음 | SharedContext |
| 엔티티 분류 | `entity_classifications` 비어있지 않음 | SharedContext |
| 위상동형 매핑 | `homeomorphisms` (다중 테이블 시) | SharedContext |

### 3.3 Refinement Phase

| 체크 항목 | 기대값 | 확인 방법 |
|----------|--------|----------|
| 4개 에이전트 실행 | 모두 SUCCESS | `[AGENT_TASK] SUCCESS` 로그 |
| 온톨로지 개념 생성 | `ontology_concepts` 비어있지 않음 | SharedContext |
| 개념 관계 생성 | `concept_relationships` 비어있지 않음 | SharedContext |
| OWL2 추론 결과 | `owl2_reasoning_results` in context | SharedContext |
| SHACL 검증 결과 | `shacl_validation_results` in context | SharedContext |

### 3.4 Governance Phase

| 체크 항목 | 기대값 | 확인 방법 |
|----------|--------|----------|
| 4개 에이전트 실행 | 모두 SUCCESS | `[AGENT_TASK] SUCCESS` 로그 |
| 거버넌스 결정 | `governance_decisions` 비어있지 않음 | SharedContext |
| 비즈니스 인사이트 | `business_insights` 비어있지 않음 | SharedContext |
| 승인/거부 결정 | approve 또는 reject 상태 존재 | decisions 확인 |

### 3.5 v17 서비스

| 서비스 | 체크 항목 | 확인 방법 |
|--------|----------|----------|
| VersionManager | 버전 2개 이상 생성 | `v_*` 버전 ID |
| SemanticSearcher | 인덱싱 완료 | 로그 확인 |
| WhatIfAnalyzer | 시뮬레이션 실행 | `baseline_scenario` 생성 |
| ReportGenerator | 리포트 2개 생성 | executive_summary, pipeline_status |
| WritebackEngine | 쓰기 실행 (승인 시) | writeback 로그 |

### 3.6 에이전트 학습 시스템

| 체크 항목 | 기대값 | 확인 방법 |
|----------|--------|----------|
| record_experience 호출 | discovery: 7회, refinement: 5회, governance: 4회 | 소스 코드 |
| 경험 유형 | entity_mapping, fk_detection, relationship_inference 등 | 로그 |

### 3.7 증거 체인 (Evidence Chain)

| 체크 항목 | 기대값 | 확인 방법 |
|----------|--------|----------|
| 증거 블록 생성 | 최소 20개 이상 | evidence_chain.json |
| 해시 체인 유효 | previous_block_hash 연결 | 체인 검증 |
| Phase별 블록 | versioning, discovery, refinement, governance | 블록 분류 |

---

## 4. 비즈니스 인사이트 추출 방법

### 4.1 인사이트 유형

| 유형 | 분석 방법 | 예시 |
|------|----------|------|
| **kpi** | 수치 컬럼 통계 분석 | "High Average Total Charges" |
| **risk** | 집중도/의존성 분석 | "Dependency on Limited Manufacturers" |
| **anomaly** | IsolationForest, LOF, Z-score | "Incomplete Lab Result Units" |
| **operational** | 시계열/분포 분석 | "Concentration of Appointments" |
| **status_distribution** | 상태 컬럼 분포 | "Pending Status Distribution" |

### 4.2 인사이트 검증 단계

```
1. 패턴 탐지 (BusinessInsightsAnalyzer)
   └→ 통계적 이상치, 분포 편향 감지

2. 인과관계 분석 (CausalImpactAnalyzer)
   └→ 원인-결과 관계 추론

3. 교차 엔티티 상관 (CrossEntityCorrelationAnalyzer)
   └→ 다중 테이블 간 상관관계

4. 시뮬레이션 검증 (WhatIfAnalyzer)
   └→ 시나리오 기반 영향도 평가

5. 거버넌스 승인 (Consensus Engine)
   └→ 멀티에이전트 합의
```

---

## 5. 멀티에이전트 합의 구조

### 5.1 Consensus Engine 동작

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consensus Engine Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                               │
│   │  negotiate()│ ◄── 기본 합의 방식                             │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼ 실패 시                                               │
│   ┌─────────────────────────┐                                   │
│   │ evidence_based_vote()   │ ◄── 증거 기반 투표                 │
│   └──────────┬──────────────┘                                   │
│              │                                                  │
│              ▼ 실패 시                                           │
│   ┌─────────────────────────┐                                   │
│   │ structured_debate()     │ ◄── 구조화된 토론 (에스컬레이션)    │
│   └──────────┬──────────────┘                                   │
│              │                                                  │
│              ▼                                                  │
│   ┌─────────────────────────┐                                   │
│   │ Final Decision          │                                   │
│   └─────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Phase별 합의 모드

| Phase | Mode | 설명 |
|-------|------|------|
| Discovery | `full_skip` | 합의 생략 (탐색 단계이므로) |
| Refinement | `selective` | 품질/온톨로지 결정에만 합의 |
| Governance | `selective + debate` | 거버넌스 결정에 합의 + 필요시 토론 |

---

## 6. ML 검증 및 시뮬레이션 기능 현황

### 6.1 신뢰도 보정 (✅ 구현 완료)

| 모듈 | 위치 | 상태 | 설명 |
|------|------|------|------|
| ConfidenceCalibrator | calibration/confidence.py | ✅ 활성 | base.py:852-884 `calibrate_confidence()` 통해 간접 호출 |

**호출 경로**: 에이전트 → `calibrate_confidence(raw_confidence)` → calibrator.calibrate()

### 6.2 ML 기반 검증 (✅ 구현 완료)

| 기능 | 위치 | 상태 | 메트릭 |
|------|------|------|--------|
| 예측 정확도 평가 | time_series_forecaster.py:1553-1598 | ✅ 구현 | MAE, RMSE, MAPE, SMAPE |
| Bootstrap 검증 | simulation_engine.py:421-515 | ✅ 구현 | 500회 반복 리샘플링 |
| Cross-dataset 검증 | cross_dataset_validator.py | ✅ 구현 | Precision, Recall, F1-Score |

### 6.3 시뮬레이션 (✅ 구현 완료)

| 기능 | 위치 | 상태 | 설명 |
|------|------|------|------|
| What-If 시나리오 | what_if/analyzer.py | ✅ 구현 | 규칙 기반 + 에이전트 연동 |
| Bootstrap 시뮬레이션 | simulation_engine.py:421-515 | ✅ 구현 | 500회 반복 |
| 민감도 분석 | simulation_engine.py | ✅ 구현 | 파라미터별 영향도 분석 |

### 6.4 v22.0 신규 기능 (Agent Bus 통합)

| 기능 | 위치 | 상태 | 설명 |
|------|------|------|------|
| Agent Communication Bus | agent_bus.py | ✅ 구현 | 에이전트간 메시지 브로드캐스트 |
| Discovery Broadcast | discovery.py | ✅ 활성 | 도메인/TDA/FK/위상동형 결과 공유 |
| Refinement Broadcast | refinement.py | ✅ 활성 | OWL2/SHACL/충돌해결 결과 공유 |
| Governance Broadcast | governance.py | ✅ 활성 | 거버넌스 결정 결과 공유 |

---

## 7. 정상 동작 확인 명령어

```bash
# 1. Core 로직 테스트 (LLM 없이)
python scripts/test_v22_core_only.py

# 2. 상세 통합 리포트
python scripts/test_v22_final_report.py

# 3. Full Pipeline 테스트 (LLM 필요, 시간 소요)
python scripts/test_v22_full_pipeline.py

# 4. v17 서비스 확인
python -c "
from src.unified_pipeline.autonomous_pipeline import AutonomousPipelineOrchestrator
orch = AutonomousPipelineOrchestrator(scenario_name='test', enable_v17_services=True)
v17 = orch.shared_context.get_dynamic('_v17_services', {})
print(f'v17 Services: {len(v17)}')
for name in v17:
    print(f'  - {name}')
"
```

---

## 8. 정상 동작 판정 기준 요약

### ✅ 정상 동작으로 판정하려면:

1. **도메인 감지**: 신뢰도 >= 0.5
2. **Discovery**: 7개 에이전트 모두 SUCCESS
3. **Refinement**: 4개 에이전트 모두 SUCCESS, 온톨로지 개념 생성
4. **Governance**: 4개 에이전트 모두 SUCCESS, 거버넌스 결정 생성
5. **v17 서비스**: 13개 중 12개 이상 정상 호출
6. **에이전트 학습**: record_experience 16회 호출
7. **증거 체인**: 20개 이상 블록 생성

### ⚠️ 부분 동작:

1. 일부 에이전트 실패 (재시도 필요)
2. 합의 실패 → 토론 에스컬레이션
3. 시간 제한으로 중단

### ❌ 비정상:

1. 도메인 감지 실패 (신뢰도 < 0.1)
2. v17 서비스 5개 이상 미동작
3. 핵심 에이전트 (TDA, Schema, Entity) 실패
4. 증거 체인 생성 실패

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-02-04 | v22.0 | 최초 작성 |
