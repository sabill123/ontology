# Ontoloty v14.0

> **Multi-Agent Autonomous Ontology Discovery Platform**
>
> 분산 데이터 사일로에서 자동으로 엔티티를 발견하고, 관계를 추론하며, 비즈니스 인사이트를 생성합니다.

[![Version](https://img.shields.io/badge/version-14.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.11+-green)]()
[![Status](https://img.shields.io/badge/status-production-brightgreen)]()

---

## Overview

Ontoloty는 **14개의 자율 에이전트**가 협력하여 데이터 온톨로지를 자동 구축하는 플랫폼입니다.

### 핵심 기능

- **자동 도메인 탐지**: LLM-First 분석으로 Supply Chain, Healthcare, Retail 등 산업 도메인 자동 인식
- **Cross-Entity 상관관계 분석**: 팔란티어 스타일 예측 인사이트 생성
- **위상 데이터 분석(TDA)**: Betti Numbers 기반 구조적 유사성 탐지
- **BFT 합의 엔진**: Byzantine Fault Tolerant 다중 에이전트 의사결정
- **Evidence Chain**: 블록체인 스타일 감사 추적

### 3-Phase Pipeline

```
Phase 1 (Discovery)  →  Phase 2 (Refinement)  →  Phase 3 (Governance)
     6 Agents                 4 Agents                 4 Agents
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key

### Installation

```bash
# Clone repository
git clone <repo-url>
cd ontoloty

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Run Pipeline

```bash
# Run with supply chain test data
python -m src.unified_pipeline.unified_main data/palantir_test/supply_chain

# Results are saved in Logs/<timestamp>/artifacts/
```

---

## System Architecture

### 14 Autonomous Agents

| Phase | Agents | Role |
|-------|--------|------|
| **Phase 1: Discovery** | 6 agents | 데이터 탐색 및 엔티티 발견 |
| | DataAnalystAgent | LLM-First 도메인 분석 (v13.0) |
| | TDAExpertAgent | Betti Numbers, 구조적 복잡도 |
| | SchemaAnalystAgent | 컬럼 타입 및 키 탐지 |
| | ValueMatcherAgent | Cross-table 값 매칭 |
| | EntityClassifierAgent | 테이블→엔티티 매핑 |
| | RelationshipDetectorAgent | FK 및 의미적 관계 발견 |
| **Phase 2: Refinement** | 4 agents | 온톨로지 설계 및 품질 검증 |
| | OntologyArchitectAgent | 온톨로지 구조 설계, Cross-Entity Correlation (v10.0) |
| | ConflictResolverAgent | 충돌 해결 및 병합 |
| | QualityJudgeAgent | 품질 평가 및 신뢰도 산정 |
| | SemanticValidatorAgent | 논리적 일관성 검증 |
| **Phase 3: Governance** | 4 agents | 합의 및 비즈니스 의사결정 |
| | GovernanceStrategistAgent | 전략적 의사결정 |
| | ActionPrioritizerAgent | 액션 우선순위화 |
| | RiskAssessorAgent | 리스크 평가 |
| | PolicyGeneratorAgent | 정책 생성 |

### Key Algorithms

| Algorithm | Purpose | Module |
|-----------|---------|--------|
| **TDA (Topological Data Analysis)** | 테이블 구조 유사성 탐지 | `analysis/tda.py` |
| **Betti Numbers** | 순환 참조 탐지 (beta1 > 0) | `analysis/advanced_tda.py` |
| **Cross-Entity Correlation** | 테이블 간 상관관계 분석 | `analysis/cross_entity_correlation.py` |
| **Dempster-Shafer Fusion** | 불확실한 증거 결합 | `consensus/dempster_shafer.py` |
| **BFT Consensus** | 다중 에이전트 합의 | `consensus/engine.py` |

---

## Project Structure

```
ontoloty/
├── src/unified_pipeline/
│   ├── unified_main.py              # Entry point
│   └── autonomous/
│       ├── orchestrator.py          # Pipeline orchestrator
│       ├── shared_context.py        # Agent shared state
│       ├── agents/
│       │   ├── base.py              # AutonomousAgent base
│       │   ├── discovery.py         # Phase 1 agents (6)
│       │   ├── refinement.py        # Phase 2 agents (4)
│       │   └── governance.py        # Phase 3 agents (4)
│       ├── analysis/
│       │   ├── tda.py               # TDA analyzer
│       │   ├── advanced_tda.py      # Betti numbers
│       │   └── cross_entity_correlation.py  # v10.0
│       ├── consensus/
│       │   ├── engine.py            # Consensus engine
│       │   └── dempster_shafer.py   # DS fusion
│       └── common/
│           ├── todo_manager.py      # Todo DAG management
│           └── evidence_chain.py    # v11.0 audit trail
├── data/
│   └── palantir_test/               # Test datasets
│       ├── supply_chain/            # 7 tables
│       └── healthcare/              # 2 tables
├── docs/
│   ├── ARCHITECTURE.md              # System architecture
│   ├── PHASE_1_DISCOVERY_SPEC.md    # Phase 1 specification
│   ├── PHASE_2_REFINEMENT_SPEC.md   # Phase 2 specification
│   ├── PHASE_3_GOVERNANCE_SPEC.md   # Phase 3 specification
│   ├── ALGORITHMS_AND_MATHEMATICS.md # Algorithm details
│   └── diagrams/                    # Mermaid diagrams
└── Logs/                            # Pipeline outputs
```

---

## Output Artifacts

파이프라인 실행 후 `Logs/<timestamp>/artifacts/`에 생성되는 파일:

| File | Description |
|------|-------------|
| `discovery_report.json` | Phase 1 엔티티 발견 결과 |
| `ontology.json` | 온톨로지 스키마 (concepts, properties, links) |
| `knowledge_graph.json` | Knowledge Graph (entities, relations, communities) |
| `governance_decisions.json` | Phase 3 거버넌스 결정 및 액션 |
| `evidence_chain.json` | 전체 분석의 감사 추적 (v11.0) |
| `cross_entity_insights.json` | Cross-Entity 상관관계 인사이트 (v10.0) |

---

## Supported Domains

| Domain | Keywords | Example Tables |
|--------|----------|---------------|
| **Supply Chain** | supplier, shipment, warehouse, freight, carrier | order_list, freight_rates, warehouse_capacities |
| **Healthcare** | patient, diagnosis, prescription, hospital | patients, hospitals |
| **Retail** | product, sku, cart, checkout, promotion | products, orders, customers |
| **Finance** | account, transaction, loan, investment | transactions, accounts |
| **Web Analytics** | page_view, session, visitor, conversion | sessions, page_views |

---

## Version History

| Version | Key Features |
|---------|--------------|
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

---

## License

MIT License

---

**Built for Enterprise Data Integration**
