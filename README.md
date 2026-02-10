# Ontoloty v25.2

> **Multi-Agent Autonomous Ontology Discovery & Data Intelligence Platform**
>
> 분산 데이터 사일로에서 자동으로 엔티티를 발견하고, 관계를 추론하며, 비즈니스 인사이트를 생성합니다.

[![Version](https://img.shields.io/badge/version-25.2-blue)]()
[![Python](https://img.shields.io/badge/python-3.11+-green)]()
[![Status](https://img.shields.io/badge/status-production-brightgreen)]()
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF)]()

---

## Overview

Ontoloty는 **16개의 자율 에이전트**가 3단계 파이프라인을 통해 데이터 온톨로지를 자동 구축하고, 팔란티어 스타일의 처방적 비즈니스 인사이트를 생성하는 플랫폼입니다.

### 핵심 기능

- **자동 도메인 탐지**: LLM-First 분석으로 10개 산업 도메인 자동 인식 (신뢰도 92-100%)
- **ML 앙상블 이상 탐지**: 수치형 컬럼별 이상치 자동 탐지 및 리스크 분류
- **팔란티어 스타일 처방적 인사이트**: 가설 → 근거 → 검증 → 처방 구조의 Unified Insight Pipeline
- **위상 데이터 분석(TDA)**: Betti Numbers 기반 데이터 구조적 복잡도 측정 (Ripser)
- **BFT 합의 엔진**: 4-Agent Council 의사결정 (Ontologist, Risk Assessor, Business Strategist, Data Steward)
- **Dempster-Shafer 증거 융합**: 다중 소스 불확실성 결합
- **Evidence Chain**: 블록체인 스타일 감사 추적 (SHA256 해시 체인)
- **OWL2 추론 + SHACL 검증**: 시맨틱 웹 표준 기반 온톨로지 검증
- **Knowledge Graph 트리플**: RDF 스타일 거버넌스 트리플 자동 생성

### 3-Phase Pipeline

```
Phase 1 (Discovery)  →  Phase 2 (Refinement)  →  Phase 3 (Governance)
     6 Agents                 4 Agents                 4+2 Agents
   + Analysis Modules       + Enhanced Validator       + Unified Insight Pipeline
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- LLM Gateway API Key (Letsur AI)

### Installation

```bash
git clone <repo-url>
cd ontoloty

pip install -r requirements.txt

# Set environment variables
export LETSUR_API_KEY="your-api-key"
```

### Run Pipeline (Local)

```bash
# Single CSV
python -m src.unified_pipeline.unified_main data/sample_csv/patients.csv

# Multi-table Silo
python -m src.unified_pipeline.unified_main data/healthcare_silo/

# With domain hint
python -m src.unified_pipeline.unified_main data/sample_csv/sales.csv --domain retail
```

### Run Pipeline (CI)

```bash
# GitHub Actions workflow dispatch
gh workflow run "Run Ontoloty Pipeline" --ref main -f dataset=sample_csv/patients.csv

# Custom dataset
gh workflow run "Run Ontoloty Pipeline" --ref main -f dataset=q2cut_metadata_extended_20260209_210925.csv
```

---

## System Architecture

### 16 Autonomous Agents

| Phase | Agent | Role |
|-------|-------|------|
| **Phase 1: Discovery** | | **데이터 탐색 및 엔티티 발견** |
| | DataAnalystAgent | LLM-First 도메인 분석, 컬럼 시맨틱 해석 |
| | TDAExpertAgent | Betti Numbers, Persistent Homology, 구조적 복잡도 |
| | SchemaAnalystAgent | 컬럼 프로파일링, PK/FK 후보 탐지 |
| | ValueMatcherAgent | Cross-table 값 매칭, 외래키 추론 |
| | EntityClassifierAgent | 테이블 → 비즈니스 엔티티 매핑 |
| | RelationshipDetectorAgent | FK 탐지, 의미적 관계 발견 |
| **Phase 2: Refinement** | | **온톨로지 설계 및 품질 검증** |
| | OntologyArchitectAgent | 온톨로지 구조 설계, TDA 기반 confidence 보정 |
| | ConflictResolverAgent | 충돌 탐지/해결, clean concept 생성 |
| | QualityJudgeAgent | 품질 평가, 알고리즘 점수 산정 |
| | SemanticValidatorAgent | OWL2 추론, SHACL 검증, 시맨틱 일관성 |
| **Phase 3: Governance** | | **합의 및 비즈니스 의사결정** |
| | GovernanceStrategistAgent | 4-Agent Council BFT 합의, Evidence-based 투표 |
| | ActionPrioritizerAgent | 액션 우선순위화 (impact/effort 매트릭스) |
| | RiskAssessorAgent | 리스크 평가 (인사이트 기반) |
| | PolicyGeneratorAgent | 거버넌스 정책/규칙 생성 |
| **Analysis** | | **인사이트 생성** |
| | BusinessInsightsAgent | LLM 기반 비즈니스 인사이트 |
| | UnifiedInsightPipeline | Algorithm → Simulation → LLM 3단계 파이프라인 |

### Key Algorithms

| Algorithm | Purpose | Module |
|-----------|---------|--------|
| **TDA (Topological Data Analysis)** | 데이터 구조 복잡도 측정 | `analysis/tda.py`, `analysis/advanced_tda.py` |
| **Persistent Homology** | Betti Numbers 계산 (Ripser) | `analysis/advanced_tda.py` |
| **ML Ensemble Anomaly Detection** | 수치형 이상치 탐지 | `analysis/anomaly_detection.py` |
| **Cross-Entity Correlation** | 테이블 간 상관관계 분석 | `analysis/cross_entity_correlation.py` |
| **Dempster-Shafer Fusion** | 불확실한 증거 결합 | `analysis/enhanced_validator/` |
| **BFT Consensus** | 다중 에이전트 합의 (n >= 3f+1) | `autonomous/consensus_coordinator.py` |
| **OWL2 Reasoning** | 5가지 공리 유형 추론 | `analysis/owl2_reasoner.py` |
| **SHACL Validation** | 6가지 제약 조건 검증 | `analysis/shacl_validator.py` |
| **Unified Insight Pipeline** | 알고리즘 → 시뮬레이션 → LLM 처방적 분석 | `analysis/unified_insight_pipeline.py` |
| **Causal Impact Analysis** | 인과 관계 추론 | `analysis/causal_impact/` |
| **Time Series Forecasting** | 시즌성 탐지, 시계열 예측 | `analysis/time_series_forecaster.py` |

---

## Project Structure

```
ontoloty/
├── src/unified_pipeline/
│   ├── unified_main.py                    # Entry point
│   ├── shared_context.py                  # Central shared state (OntologyConcept, etc.)
│   ├── model_config.py                    # LLM model configuration
│   │
│   └── autonomous/
│       ├── orchestrator.py                # Pipeline orchestrator
│       ├── base.py                        # AutonomousAgent base class
│       ├── continuation.py                # Todo continuation enforcer
│       ├── consensus_coordinator.py       # Smart consensus (full_skip/lightweight/full)
│       ├── agent_bus.py                   # Agent communication bus
│       ├── evidence_chain.py              # Blockchain-style audit trail
│       ├── debate_protocol.py             # BFT debate with CouncilAgent
│       │
│       ├── agents/
│       │   ├── discovery/                 # Phase 1: 6 agents
│       │   │   ├── data_analyst.py
│       │   │   ├── tda_expert.py
│       │   │   ├── schema_analyst.py
│       │   │   ├── value_matcher.py
│       │   │   ├── entity_classifier.py
│       │   │   └── relationship_detector.py
│       │   ├── refinement/                # Phase 2: 4 agents
│       │   │   ├── ontology_architect.py
│       │   │   ├── conflict_resolver.py
│       │   │   ├── quality_judge.py
│       │   │   └── semantic_validator.py
│       │   └── governance/                # Phase 3: 4 agents
│       │       ├── governance_strategist.py
│       │       ├── action_prioritizer.py
│       │       ├── risk_assessor.py
│       │       └── policy_generator.py
│       │
│       ├── analysis/                      # 49+ analysis modules
│       │   ├── business_insights/         # LLM business insights (package)
│       │   ├── causal_impact/             # Causal inference (package)
│       │   ├── enhanced_validator/        # DS + Statistical + BFT (package)
│       │   ├── unified_insight_pipeline.py
│       │   ├── anomaly_detection.py       # ML ensemble anomaly detection
│       │   ├── tda.py                     # Topological data analysis
│       │   ├── advanced_tda.py            # Ripser persistent homology
│       │   ├── cross_entity_correlation.py
│       │   ├── owl2_reasoner.py           # OWL 2 reasoning
│       │   ├── shacl_validator.py         # SHACL validation
│       │   ├── knowledge_graph.py         # RDF triple construction
│       │   ├── simulation_engine.py       # Scenario simulation
│       │   ├── kpi_predictor.py           # Time series + ROI
│       │   ├── time_series_forecaster.py  # Temporal prediction
│       │   ├── integrated_fk_detector.py  # 5-stage FK pipeline
│       │   └── ...                        # 35+ additional modules
│       │
│       ├── consensus/                     # Consensus engine
│       ├── learning/                      # Agent experience replay
│       └── todo/                          # Todo DAG management
│
├── scripts/
│   ├── run_pipeline_ci.py                 # CI runner script
│   └── detailed_pipeline_report.py        # Report generator
│
├── data/
│   ├── sample_csv/                        # Single-table test datasets
│   │   ├── patients.csv
│   │   ├── shipments.csv
│   │   ├── employees.csv
│   │   ├── production.csv
│   │   ├── sales.csv
│   │   └── students.csv
│   ├── healthcare_silo/                   # Multi-table silos
│   ├── ecommerce_silo/
│   ├── marketing_silo/
│   ├── hr_silo/
│   ├── airport_silo/
│   └── ...
│
├── docs/
│   ├── ARCHITECTURE.md                    # System architecture
│   ├── PHASE_1_DISCOVERY_SPEC.md
│   ├── PHASE_2_REFINEMENT_SPEC.md
│   ├── PHASE_3_GOVERNANCE_SPEC.md
│   ├── ALGORITHMS_AND_MATHEMATICS.md
│   ├── USER_GUIDE.md
│   └── diagrams/                          # Mermaid diagrams
│
├── .github/workflows/
│   └── run-pipeline.yml                   # CI pipeline workflow
│
└── requirements.txt
```

---

## Output Artifacts

파이프라인 실행 후 생성되는 파일 (CI: `data/output/`, 로컬: `data/storage/`):

| File | Description |
|------|-------------|
| `*_context.json` | 전체 SharedContext — 온톨로지, 인사이트, 증거 체인 포함 |
| `*_entities.json` | 발견된 엔티티 목록 (confidence, source_agent 포함) |
| `*_stats.json` | 실행 통계 — 에이전트 수, 완료 Todo, 합의 결과 |
| `*_actions.json` | 거버넌스 액션 플랜 (priority, effort, estimated_hours) |
| `*_summary.json` | 실행 요약 (entities, relationships, insights, governance) |
| `ci_result_summary.md` | GitHub Step Summary용 마크다운 |

### Context JSON 주요 필드

| 필드 | 설명 |
|------|------|
| `ontology_concepts` | OntologyConcept 리스트 (concept_type, confidence, source_agent) |
| `business_insights` | 비즈니스 인사이트 (LLM+ML 생성) |
| `palantir_insights` | 팔란티어 스타일 처방적 인사이트 (가설/근거/조치) |
| `unified_insights_pipeline` | Algorithm → Simulation → LLM 통합 인사이트 |
| `governance_decisions` | BFT Council 투표 결과 (agent_opinions, vote_summary) |
| `evidence_chain_summary` | 증거 블록 수, 체인 무결성 상태 |
| `evidence_based_debate_results` | 증거 기반 토론 결과 (cited_evidence) |
| `governance_triples` | RDF 스타일 거버넌스 트리플 |
| `tda_signatures` | Betti Numbers, Euler Characteristic, Persistence |
| `dynamic_data` | OWL2, SHACL, quality_assessments, conflicts 등 |

---

## Supported Domains

| Domain | Silo | Tables | Keywords |
|--------|------|--------|----------|
| **Healthcare** | `healthcare_silo` | 10 | patient, diagnosis, prescription |
| **E-Commerce** | `ecommerce_silo` | 11 | customer, order, product |
| **Marketing** | `marketing_silo` | 11 | campaign, channel, conversion |
| **HR** | `hr_silo` | 11 | employee, department, salary |
| **Airport** | `airport_silo` | 9 | flight, passenger, terminal |
| **Finance** | `finance_silo` | 10 | account, transaction, loan |
| **Education** | `education_silo` | 10 | student, course, enrollment |
| **Logistics** | `logistics_silo` | 10 | shipment, warehouse, carrier |
| **Manufacturing** | `manufacturing_silo` | 10 | plant, work_order, equipment |
| **Media/Entertainment** | single CSV | 1+ | video, creator, platform, viral |

---

## Pipeline Results (Recent)

### patients.csv (v25.2.4)
| Metric | Value |
|--------|-------|
| Entities | 38 (31 ontology_architect + 7 fallback) |
| Relationships | 16 |
| Insights | 12 |
| Governance Decisions | 7 |
| Agent Assignments | 16/16 SUCCESS |
| Data Flow Checks | 18/18 PASS |

### q2cut_metadata_extended (v25.2.4)
| Metric | Value |
|--------|-------|
| Records | 1,812 |
| Entities | 1 (VideoMetadata, conf=0.95) |
| Business Insights | 46 |
| Palantir Insights | 4 (CRITICAL 1, HIGH 2, MEDIUM 1) |
| ML Anomaly Detection | 19 columns analyzed |
| Evidence Blocks | 63 (chain valid) |
| BFT Council | 4:0 unanimous approve |
| Execution Time | 16m 57s |

---

## Version History

| Version | Key Features |
|---------|--------------|
| **v25.2** | Phase 간 데이터 흐름 수정 (context_updates 전파), TDA confidence boost, source_agent 추적 |
| **v25.1** | Agent 패키지 구조 재분리 (discovery/, refinement/, governance/) |
| **v24.0** | CI/로컬 전체 버그 수정 + 안정화 (15개 이슈) |
| **v23.0** | Evidence-based vote 통합, adjusted_confidence 감쇠 수정 |
| **v17.0** | Enterprise Services Framework (Actions, CDC, Lineage, Learning) |
| **v16.0** | Agent Communication Bus (Direct, Broadcast, Pub/Sub, RPC) |
| **v14.0** | AgentTerminateMode, Phase Gate validation |
| **v13.0** | LLM-First DataAnalyst agent |
| **v11.0** | Evidence Chain (blockchain-style audit) |
| **v10.0** | Cross-Entity Correlation Analysis |
| **v8.2** | Smart Consensus (full_skip, lightweight, full) |
| **v7.0** | BFT Consensus + Dempster-Shafer Fusion |

---

## Documentation

- [System Architecture](./docs/ARCHITECTURE.md) - 전체 시스템 구조
- [Phase 1: Discovery](./docs/PHASE_1_DISCOVERY_SPEC.md) - Discovery Engine 명세
- [Phase 2: Refinement](./docs/PHASE_2_REFINEMENT_SPEC.md) - Refinement Engine 명세
- [Phase 3: Governance](./docs/PHASE_3_GOVERNANCE_SPEC.md) - Governance Engine 명세
- [Algorithms](./docs/ALGORITHMS_AND_MATHEMATICS.md) - 수학적 기반
- [Diagrams](./docs/diagrams/README.md) - Mermaid 다이어그램 모음
- [User Guide](./docs/USER_GUIDE.md) - 사용 가이드

---

## Configuration

```python
# Environment Variables
LETSUR_API_KEY=...                          # LLM Gateway API Key
HF_TOKEN=...                               # HuggingFace Token (optional)

# LLM Gateway
LLM_GATEWAY_URL=https://gateway.letsur.ai/v1

# Model Configuration (CI)
MODEL_FAST=gpt-5.1
MODEL_BALANCED=claude-opus-4-5-20251101
MODEL_CREATIVE=gemini-3-pro-preview
MODEL_HIGH_CONTEXT=gemini-3-pro-preview
```

---

## License

MIT License

---

**Built for Enterprise Data Integration**
