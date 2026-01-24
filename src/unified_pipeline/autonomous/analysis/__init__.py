"""
Analysis Utilities for Autonomous Agents

실제 알고리즘 기반 분석 유틸리티

기본 분석:
- TDA: 위상수학적 데이터 분석
- Similarity: Jaccard, 값 겹침 분석
- Advanced Similarity: MinHash, LSH, HyperLogLog
- Graph: 그래프 분석, 관계 탐지
- Schema: 스키마 분석
- Enhanced FK: 강화된 FK 탐지
- Entity Extraction: 숨은 엔티티 추출
- Business Insights: 비즈니스 인사이트 도출
- Ontology: 온톨로지 충돌/품질/의미 분석
- Governance: 거버넌스 의사결정/리스크/우선순위

고급 분석 (새로 추가):
- Advanced TDA: Ripser 기반 진짜 위상수학적 분석
- Entity Resolution: ML 기반 엔티티 매칭/통합
- Graph Embedding: Node2Vec, 커뮤니티 탐지
- Anomaly Detection: 앙상블 이상 탐지
- Causal Impact: 인과 추론 및 영향 분석
- KPI Predictor: 시계열 예측 및 ROI 분석
- Ontology Alignment: 온톨로지 정렬 및 매칭
- Semantic Reasoner: 의미적 추론 엔진
"""

from .tda import TDAAnalyzer, TDASignature, BettiNumbers
from .similarity import SimilarityAnalyzer, ValueOverlap, JaccardResult
from .advanced_similarity import (
    AdvancedSimilarityAnalyzer,
    MinHash,
    LSH,
    HyperLogLog,
    SimilarityResult,
)
from .graph import GraphAnalyzer, RelationshipGraph, EdgeType
from .schema import SchemaAnalyzer, ColumnProfile, KeyCandidate
from .enhanced_fk_detector import EnhancedFKDetector, FKCandidate
from .llm_fk_validator import (
    LLMFKValidator,
    FalsePositiveFilter,
    CrossPlatformFKDetector,
    FKValidationResult,
    FKRelationshipType,
)
# v6.0: Universal FK Detection (Domain-Agnostic)
from .universal_fk_detector import (
    UniversalFKDetector,
    NamingNormalizer,
    ValueBasedFKValidator,
    FKCandidate as UniversalFKCandidate,
    NormalizedName,
    TableGroup,
)
from .insight_standardizer import InsightStandardizer, standardize_insights
from .entity_extractor import EntityExtractor, ExtractedEntity, EntityType
from .business_insights import BusinessInsightsAnalyzer, BusinessInsight, InsightType
from .ontology import (
    ConflictDetector,
    ConflictInfo,
    QualityAssessor,
    QualityScore,
    SemanticValidator,
    SemanticValidation,
)
from .governance import (
    DecisionMatrix,
    DecisionResult,
    RiskAssessor as GovernanceRiskAssessor,
    RiskAssessment,
    PriorityCalculator,
    PolicyGenerator,
    PolicyRule,
)
from .utils import parse_llm_json, retry_with_backoff

# Advanced Analysis Modules (새로 추가)
from .advanced_tda import (
    AdvancedTDAAnalyzer,
    AdvancedTDASignature,
    PersistenceDiagram,
    TDADistance,
)
from .entity_resolution import (
    EntityResolver,
    EntityMatch,
    EntityCluster,
    EntityCandidate,
    MinHashBlocker,
    SimilarityCalculator,
)
from .graph_embedding import (
    GraphEmbeddingAnalyzer,
    SchemaGraph,
    GraphNode,
    GraphEdge,
    GraphEmbedding,
    Node2VecEmbedder,
    CommunityDetector,
    Community,
    CentralityAnalyzer,
    CentralityMetrics,
    NodeType,
    EdgeType as GraphEdgeType,
)
from .anomaly_detection import (
    AnomalyDetector,
    Anomaly,
    AnomalyReport,
    AnomalyType,
    AnomalySeverity,
    StatisticalDetector,
    MLDetector,
    DataQualityDetector,
)
from .causal_impact import (
    CausalImpactAnalyzer,
    CausalGraph,
    CausalNode,
    CausalEdge,
    TreatmentEffect,
    CounterfactualResult,
    SensitivityResult,
    CausalImpactResult,
    ATEEstimator,
    CATEEstimator,
    PropensityScoreEstimator,
    # v6.1: Granger Causality for dynamic causal inference
    GrangerCausalityAnalyzer,
    GrangerCausalityResult,
    # v6.1: Counterfactual Reasoning
    CounterfactualAnalyzer,
)
from .kpi_predictor import (
    KPIPredictor,
    IntegrationKPIAnalyzer,
    KPIForecast,
    Forecast,
    KPIDefinition,
    KPIType,
    ScenarioAnalysis,
    ScenarioType,
    ROIProjection,
    ROICalculator,
    SensitivityAnalysis,
)
from .ontology_alignment import (
    OntologyAlignmentAnalyzer,
    OntologyAligner,
    Alignment,
    AlignmentSet,
    AlignmentType,
    OntologyElement,
    StringMatcher,
    StructuralMatcher,
    SemanticMatcher,
    InstanceMatcher,
    MatchingMetrics,
    ConflictResolver,
)
from .semantic_reasoner import (
    SemanticReasoningAnalyzer,
    SemanticReasoner,
    SchemaReasoner,
    KnowledgeBase,
    Triple,
    Rule,
    RuleType,
    InferenceChain,
    ForwardChainer,
    BackwardChainer,
    RDFSReasoner,
    OWLReasoner,
    ConsistencyChecker,
    CompletenessChecker,
    ConsistencyReport,
    CompletenessReport,
)
from .time_series_forecaster import (
    TimeSeriesForecaster,
    ForecastResult,
    ForecastModel,
    SeasonalityInfo,
    TrendInfo,
)
from .process_analyzer import (
    ProcessAnalyzer,
    ProcessContext,
    ProcessArea,
    WorkflowMapping,
    IntegrationAssessment,
    IntegrationComplexity,
)

# v6.1: Virtual Entity Generator
from .virtual_entity_generator import (
    VirtualEntityGenerator,
    VirtualEntityType,
    VirtualEntityDefinition,
    GeneratedVirtualEntity,
    generate_virtual_entities_for_context,
)

# v6.1: Structured Debate for Governance
from .structured_debate import (
    StructuredDebateEngine,
    DebateRole,
    DebatePhase,
    VerdictType,
    DebateArgument,
    DebateRound,
    DebateOutcome,
    create_debate_summary,
)

# v7.0: Value Pattern Analyzer (Bridge Table Detection)
from .value_pattern_analyzer import (
    EnhancedValuePatternAnalysis,
    ValuePatternAnalyzer,
    SameTableMappingDetector,
    CrossTablePatternMatcher,
    PatternAnalysisContextGenerator,
    ValuePattern,
    SameTableMapping,
    CrossTablePatternMatch,
)

# v7.0: Column Normalizer (Semantic Column Matching)
from .column_normalizer import (
    ColumnNormalizer,
    NormalizedColumn,
    ColumnMatch,
    normalize_column,
    find_column_matches,
)

# v7.0: Data Quality Analyzer (Status/Enum/Boolean Detection)
from .data_quality_analyzer import (
    DataQualityAnalyzer,
    StatusEnumAnalyzer,
    BooleanFlagDetector,
    DataQualityReport,
    StatusValueAnalysis,
    BooleanFlagAnalysis,
    OutlierAnalysis,
    StatusCategory,
)

# v7.1/v7.2: Semantic FK Detector (Comprehensive FK Detection)
from .semantic_fk_detector import (
    SemanticFKDetector,
    ColumnSemanticTokenizer,
    BridgeTableDetector,
    DynamicAbbreviationLearner,
    ValuePatternCorrelator,
    SemanticToken,
    ColumnSemantics,
    BridgeTable,
    LearnedAbbreviation,
    ValueCorrelation,
    SemanticFKCandidate,
    create_semantic_fk_detector,
    analyze_fk_semantics,
    # v7.2: New exports
    EntitySynonym,
    IndirectRelationship,
)

# v8.0: Enhanced FK Detection Pipeline (Multi-Signal + Direction + Semantic)
from .fk_signal_scorer import (
    FKSignalScorer,
    FKSignals,
)
from .fk_direction_analyzer import (
    FKDirectionAnalyzer,
    DirectionEvidence,
)
from .semantic_entity_resolver import (
    SemanticEntityResolver,
    quick_semantic_check,
)
from .enhanced_fk_pipeline import (
    EnhancedFKPipeline,
    EnhancedFKCandidate,
    create_enhanced_detector,
    detect_fks,
)

# v5.0: Enhanced Validation System
from .enhanced_validator import (
    EnhancedValidator,
    StatisticalValidator,
    StatisticalTestResult,
    DempsterShaferFusion,
    BPA,
    ConfidenceCalibrator,
    KnowledgeGraphQualityValidator,
    KGQualityScore,
    BFTConsensus,
    AgentVote,
    VoteType,
)

__all__ = [
    # TDA
    'TDAAnalyzer',
    'TDASignature',
    'BettiNumbers',
    # Similarity
    'SimilarityAnalyzer',
    'ValueOverlap',
    'JaccardResult',
    # Advanced Similarity
    'AdvancedSimilarityAnalyzer',
    'MinHash',
    'LSH',
    'HyperLogLog',
    'SimilarityResult',
    # Graph
    'GraphAnalyzer',
    'RelationshipGraph',
    'EdgeType',
    # Schema
    'SchemaAnalyzer',
    'ColumnProfile',
    'KeyCandidate',
    # Enhanced FK
    'EnhancedFKDetector',
    'FKCandidate',
    # Entity Extraction
    'EntityExtractor',
    'ExtractedEntity',
    'EntityType',
    # Business Insights
    'BusinessInsightsAnalyzer',
    'BusinessInsight',
    'InsightType',
    # Ontology
    'ConflictDetector',
    'ConflictInfo',
    'QualityAssessor',
    'QualityScore',
    'SemanticValidator',
    'SemanticValidation',
    # Governance
    'DecisionMatrix',
    'DecisionResult',
    'GovernanceRiskAssessor',
    'RiskAssessment',
    'PriorityCalculator',
    'PolicyGenerator',
    'PolicyRule',
    # Utils
    'parse_llm_json',
    'retry_with_backoff',

    # ===== Advanced Analysis Modules (새로 추가) =====

    # Advanced TDA (Ripser 기반)
    'AdvancedTDAAnalyzer',
    'AdvancedTDASignature',
    'PersistenceDiagram',
    'TDADistance',

    # Entity Resolution (ML 기반)
    'EntityResolver',
    'EntityMatch',
    'EntityCluster',
    'EntityCandidate',
    'MinHashBlocker',
    'SimilarityCalculator',

    # Graph Embedding (Node2Vec, Community Detection)
    'GraphEmbeddingAnalyzer',
    'SchemaGraph',
    'GraphNode',
    'GraphEdge',
    'GraphEmbedding',
    'Node2VecEmbedder',
    'CommunityDetector',
    'Community',
    'CentralityAnalyzer',
    'CentralityMetrics',
    'NodeType',
    'GraphEdgeType',

    # Anomaly Detection (앙상블)
    'AnomalyDetector',
    'Anomaly',
    'AnomalyReport',
    'AnomalyType',
    'AnomalySeverity',
    'StatisticalDetector',
    'MLDetector',
    'DataQualityDetector',

    # Causal Impact (인과 추론)
    'CausalImpactAnalyzer',
    'CausalGraph',
    'CausalNode',
    'CausalEdge',
    'TreatmentEffect',
    'CounterfactualResult',
    'SensitivityResult',
    'CausalImpactResult',
    'ATEEstimator',
    'CATEEstimator',
    'PropensityScoreEstimator',
    # v6.1: Granger Causality
    'GrangerCausalityAnalyzer',
    'GrangerCausalityResult',
    # v6.1: Counterfactual Reasoning
    'CounterfactualAnalyzer',

    # KPI Predictor (시계열 예측)
    'KPIPredictor',
    'IntegrationKPIAnalyzer',
    'KPIForecast',
    'Forecast',
    'KPIDefinition',
    'KPIType',
    'ScenarioAnalysis',
    'ScenarioType',
    'ROIProjection',
    'ROICalculator',
    'SensitivityAnalysis',

    # Ontology Alignment (온톨로지 정렬)
    'OntologyAlignmentAnalyzer',
    'OntologyAligner',
    'Alignment',
    'AlignmentSet',
    'AlignmentType',
    'OntologyElement',
    'StringMatcher',
    'StructuralMatcher',
    'SemanticMatcher',
    'InstanceMatcher',
    'MatchingMetrics',
    'ConflictResolver',

    # Semantic Reasoner (의미적 추론)
    'SemanticReasoningAnalyzer',
    'SemanticReasoner',
    'SchemaReasoner',
    'KnowledgeBase',
    'Triple',
    'Rule',
    'RuleType',
    'InferenceChain',
    'ForwardChainer',
    'BackwardChainer',
    'RDFSReasoner',
    'OWLReasoner',
    'ConsistencyChecker',
    'CompletenessChecker',
    'ConsistencyReport',
    'CompletenessReport',

    # Time Series Forecaster (시계열 예측)
    'TimeSeriesForecaster',
    'ForecastResult',
    'ForecastModel',
    'SeasonalityInfo',
    'TrendInfo',

    # Process Analyzer (프로세스 분석)
    'ProcessAnalyzer',
    'ProcessContext',
    'ProcessArea',
    'WorkflowMapping',
    'IntegrationAssessment',
    'IntegrationComplexity',

    # v5.0: Enhanced Validation System
    'EnhancedValidator',
    'StatisticalValidator',
    'StatisticalTestResult',
    'DempsterShaferFusion',
    'BPA',
    'ConfidenceCalibrator',
    'KnowledgeGraphQualityValidator',
    'KGQualityScore',
    'BFTConsensus',
    'AgentVote',
    'VoteType',

    # v6.0: Universal FK Detection (Domain-Agnostic)
    'UniversalFKDetector',
    'NamingNormalizer',
    'ValueBasedFKValidator',
    'UniversalFKCandidate',
    'NormalizedName',
    'TableGroup',

    # v6.1: Virtual Entity Generator
    'VirtualEntityGenerator',
    'VirtualEntityType',
    'VirtualEntityDefinition',
    'GeneratedVirtualEntity',
    'generate_virtual_entities_for_context',

    # v6.1: Structured Debate
    'StructuredDebateEngine',
    'DebateRole',
    'DebatePhase',
    'VerdictType',
    'DebateArgument',
    'DebateRound',
    'DebateOutcome',
    'create_debate_summary',

    # v7.0: Value Pattern Analyzer (Bridge Table Detection)
    'EnhancedValuePatternAnalysis',
    'ValuePatternAnalyzer',
    'SameTableMappingDetector',
    'CrossTablePatternMatcher',
    'PatternAnalysisContextGenerator',
    'ValuePattern',
    'SameTableMapping',
    'CrossTablePatternMatch',

    # v7.0: Column Normalizer (Semantic Column Matching)
    'ColumnNormalizer',
    'NormalizedColumn',
    'ColumnMatch',
    'normalize_column',
    'find_column_matches',

    # v7.0: Data Quality Analyzer (Status/Enum/Boolean Detection)
    'DataQualityAnalyzer',
    'StatusEnumAnalyzer',
    'BooleanFlagDetector',
    'DataQualityReport',
    'StatusValueAnalysis',
    'BooleanFlagAnalysis',
    'OutlierAnalysis',
    'StatusCategory',

    # v7.1/v7.2: Semantic FK Detector (Comprehensive FK Detection)
    'SemanticFKDetector',
    'ColumnSemanticTokenizer',
    'BridgeTableDetector',
    'DynamicAbbreviationLearner',
    'ValuePatternCorrelator',
    'SemanticToken',
    'ColumnSemantics',
    'BridgeTable',
    'LearnedAbbreviation',
    'ValueCorrelation',
    'SemanticFKCandidate',
    'create_semantic_fk_detector',
    'analyze_fk_semantics',
    # v7.2: New exports
    'EntitySynonym',
    'IndirectRelationship',

    # v8.0: Enhanced FK Detection Pipeline
    'FKSignalScorer',
    'FKSignals',
    'FKDirectionAnalyzer',
    'DirectionEvidence',
    'SemanticEntityResolver',
    'quick_semantic_check',
    'EnhancedFKPipeline',
    'EnhancedFKCandidate',
    'create_enhanced_detector',
    'detect_fks',
]
