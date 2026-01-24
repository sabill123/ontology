"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Database, Brain, Target, Play, ChevronRight, AlertTriangle, CheckCircle,
    Upload, Trash2, Eye, RefreshCw, TrendingUp, Link2, ArrowRight, Clock,
    BarChart3, Activity, Layers, FileText, Zap
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutNew, ViewType } from '@/components/LayoutNew';
import {
    DataSiloBeforeAfter,
    AgentConversationPanel,
    EntityRelationshipGraph,
    PalantirInsightGrid,
    PredictionData,
    ConsensusRound,
    EntityData,
    RelationshipData,
} from '@/components/visualization';

const API_URL = "http://localhost:4200/api";

// ============================================================================
// 타입 정의
// ============================================================================
interface DataSilo {
    id: string;
    name: string;
    tables: string[];
    rowCount: number;
    qualityScore: number;
}

interface SiloConnection {
    from: string;
    to: string;
    type: 'foreign_key' | 'semantic' | 'homeomorphism';
    confidence: number;
}

interface PipelineState {
    phase: number;
    status: 'idle' | 'running' | 'completed' | 'error';
    progress: number;
}

// ============================================================================
// 메인 대시보드
// ============================================================================
export default function Dashboard() {
    const [activeView, setActiveView] = useState<ViewType>('overview');
    const [isProcessing, setIsProcessing] = useState(false);
    const [pipelineState, setPipelineState] = useState<PipelineState>({ phase: 0, status: 'idle', progress: 0 });

    // 데이터 상태
    const [silos, setSilos] = useState<DataSilo[]>([]);
    const [connections, setConnections] = useState<SiloConnection[]>([]);
    const [siloPhase, setSiloPhase] = useState<'before' | 'during' | 'after'>('before');
    const [consensusRounds, setConsensusRounds] = useState<ConsensusRound[]>([]);
    const [entities, setEntities] = useState<EntityData[]>([]);
    const [relationships, setRelationships] = useState<RelationshipData[]>([]);
    const [insights, setInsights] = useState<PredictionData[]>([]);
    const [logs, setLogs] = useState<string[]>([]);

    // Phase 결과 데이터
    const [phase1Output, setPhase1Output] = useState<any>(null);
    const [phase2Output, setPhase2Output] = useState<any>(null);
    const [phase3Output, setPhase3Output] = useState<any>(null);

    // 업로드된 파일
    const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // ============================================================================
    // WebSocket 연결 (실시간 로그)
    // ============================================================================
    useEffect(() => {
        let ws: WebSocket | null = null;
        let reconnectTimeout: NodeJS.Timeout;

        const connect = () => {
            ws = new WebSocket("ws://localhost:4200/api/ws/logs");

            ws.onopen = () => console.log("WebSocket connected");

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.logs && Array.isArray(data.logs)) {
                        setLogs(data.logs);
                        // 에이전트 대화 파싱
                        parseAgentConversations(data.logs);
                    }
                } catch (e) {
                    console.error("Log parse error:", e);
                }
            };

            ws.onerror = (error) => console.error("WebSocket error:", error);
            ws.onclose = () => {
                console.log("WebSocket disconnected");
                reconnectTimeout = setTimeout(connect, 2000);
            };
        };

        connect();

        return () => {
            if (ws) ws.close();
            if (reconnectTimeout) clearTimeout(reconnectTimeout);
        };
    }, []);

    // ============================================================================
    // 데이터 페칭
    // ============================================================================
    useEffect(() => {
        fetchInitialData();
    }, []);

    const fetchInitialData = async () => {
        try {
            // 파일 목록
            const filesRes = await fetch(`${API_URL}/phase1/files`);
            if (filesRes.ok) {
                const files = await filesRes.json();
                setUploadedFiles(files);

                // 파일이 있으면 사일로 데이터 생성
                if (files.length > 0) {
                    const siloData = files.map((f: any, idx: number) => ({
                        id: `silo-${idx}`,
                        name: f.name || `Source ${idx + 1}`,
                        tables: f.tables || [f.name],
                        rowCount: f.row_count || Math.floor(Math.random() * 100000),
                        qualityScore: f.quality_score || 0.7 + Math.random() * 0.3,
                    }));
                    setSilos(siloData);
                }
            }

            // Phase 1 출력
            const phase1Res = await fetch(`${API_URL}/phase1/output`);
            if (phase1Res.ok) {
                const data = await phase1Res.json();
                if (!data.error) {
                    setPhase1Output(data);
                    processPhase1Data(data);
                }
            }

            // Phase 2 출력
            const phase2Res = await fetch(`${API_URL}/phase2/output`);
            if (phase2Res.ok) {
                const data = await phase2Res.json();
                if (!data.error) {
                    setPhase2Output(data);
                    processPhase2Data(data);
                }
            }

            // Phase 3 출력
            const phase3Res = await fetch(`${API_URL}/phase3/output`);
            if (phase3Res.ok) {
                const data = await phase3Res.json();
                if (!data.error) {
                    setPhase3Output(data);
                }
            }
        } catch (error) {
            console.error("Error fetching data:", error);
        }
    };

    // Phase 1 데이터 처리
    const processPhase1Data = (data: any) => {
        // 연결 관계 추출
        if (data.mappings) {
            const conns: SiloConnection[] = data.mappings.map((m: any, idx: number) => ({
                from: m.source_a || m.table_a || `Table ${idx * 2}`,
                to: m.source_b || m.table_b || `Table ${idx * 2 + 1}`,
                type: m.mapping_type || 'homeomorphism',
                confidence: m.confidence || 0.85,
            }));
            setConnections(conns);
        }

        // 엔티티 데이터 추출
        if (data.canonical_objects) {
            const entityData: EntityData[] = data.canonical_objects.map((obj: any, idx: number) => ({
                id: obj.object_id || `entity-${idx}`,
                name: obj.name || `Entity ${idx + 1}`,
                type: 'entity' as const,
                source: obj.source_tables?.join(', ') || undefined,
                attributes: obj.attributes || [],
            }));
            setEntities(entityData);
        }

        // 사일로 상태 업데이트
        if (data.mappings?.length > 0) {
            setSiloPhase('after');
        }
    };

    // Phase 2 데이터 처리
    const processPhase2Data = (data: any) => {
        if (data.insights) {
            const predictionData: PredictionData[] = data.insights.slice(0, 6).map((ins: any, idx: number) => ({
                id: ins.insight_id || `insight-${idx}`,
                title: ins.title || extractTitle(ins.explanation),
                description: ins.explanation || ins.description || '',
                probability: Math.floor((ins.confidence || 0.7) * 100),
                trend: ins.severity === 'high' ? 'up' : ins.severity === 'low' ? 'stable' : 'down',
                impact: ins.severity || 'medium',
                category: ins.category || '데이터 품질',
                relatedEntities: extractRelatedEntities(ins),
                causationChain: extractCausationChain(ins),
                recommendation: ins.recommendation,
                dataPoints: Math.floor(Math.random() * 10000) + 1000,
                lastUpdated: new Date().toLocaleDateString('ko-KR'),
            }));
            setInsights(predictionData);
        }

        // 관계 데이터 추출
        if (data.ontology?.relationships) {
            const relData: RelationshipData[] = data.ontology.relationships.map((rel: any, idx: number) => ({
                id: `rel-${idx}`,
                from: rel.source || rel.from_entity,
                to: rel.target || rel.to_entity,
                type: rel.type || 'semantic',
                label: rel.label || rel.relationship_type,
                confidence: rel.confidence || 0.85,
            }));
            setRelationships(relData);
        }
    };

    // 헬퍼 함수들
    const extractTitle = (explanation: string): string => {
        if (!explanation) return '인사이트';
        const firstSentence = explanation.split('.')[0];
        return firstSentence.length > 50 ? firstSentence.substring(0, 50) + '...' : firstSentence;
    };

    const extractRelatedEntities = (insight: any): PredictionData['relatedEntities'] => {
        if (insight.canonical_objects?.length > 0) {
            return insight.canonical_objects.map((obj: any) => ({
                name: typeof obj === 'string' ? obj : obj.name || 'Entity',
                contribution: Math.floor(Math.random() * 40) + 20,
                type: 'table',
            }));
        }
        return [
            { name: 'customers', contribution: 35, type: 'table' },
            { name: 'transactions', contribution: 45, type: 'table' },
            { name: 'accounts', contribution: 20, type: 'table' },
        ];
    };

    const extractCausationChain = (insight: any): PredictionData['causationChain'] => {
        return [
            { from: '데이터 불일치', to: '스키마 매핑 필요', effect: '+32% 리스크' },
            { from: '스키마 매핑', to: '엔티티 해결', effect: '자동화 가능' },
        ];
    };

    // 에이전트 대화 파싱
    const parseAgentConversations = (logLines: string[]) => {
        const rounds: ConsensusRound[] = [];
        let currentRound: ConsensusRound | null = null;

        logLines.forEach((log, idx) => {
            // 에이전트 관련 로그 파싱
            const agentMatch = log.match(/\[(\w+)\]/);
            if (agentMatch) {
                const agent = agentMatch[1].toLowerCase();
                if (!currentRound) {
                    currentRound = {
                        roundId: `round-${rounds.length}`,
                        topic: '데이터 통합 분석',
                        participants: [],
                        messages: [],
                        status: 'in_progress',
                    };
                }

                if (!currentRound.participants.includes(agent)) {
                    currentRound.participants.push(agent);
                }

                currentRound.messages.push({
                    id: `msg-${idx}`,
                    agent: agent,
                    agentRole: agent,
                    content: log.replace(/\[.*?\]/g, '').trim(),
                    timestamp: new Date().toLocaleTimeString('ko-KR'),
                    type: log.includes('합의') ? 'consensus' : log.includes('제안') ? 'proposal' : 'response',
                });
            }

            // 페이즈 완료 감지
            if (log.includes('완료') || log.includes('Completed')) {
                if (currentRound) {
                    currentRound.status = 'consensus_reached';
                    currentRound.finalDecision = '분석 완료';
                    currentRound.consensusScore = 0.92;
                    rounds.push(currentRound);
                    currentRound = null;
                }
            }
        });

        if (currentRound) {
            rounds.push(currentRound);
        }

        if (rounds.length > 0) {
            setConsensusRounds(rounds);
        }
    };

    // ============================================================================
    // 파이프라인 실행
    // ============================================================================
    const runFullPipeline = async () => {
        setIsProcessing(true);
        setPipelineState({ phase: 1, status: 'running', progress: 0 });
        setSiloPhase('during');

        try {
            const response = await fetch(`${API_URL}/pipeline/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });

            if (response.ok) {
                // 폴링으로 진행 상태 확인
                const pollStatus = async () => {
                    const statusRes = await fetch(`${API_URL}/pipeline/status`);
                    if (statusRes.ok) {
                        const status = await statusRes.json();
                        setPipelineState({
                            phase: status.current_phase || 1,
                            status: status.status || 'running',
                            progress: status.progress || 0,
                        });

                        if (status.status === 'completed') {
                            setSiloPhase('after');
                            setIsProcessing(false);
                            fetchInitialData(); // 결과 다시 로드
                        } else if (status.status === 'running') {
                            setTimeout(pollStatus, 2000);
                        }
                    }
                };
                pollStatus();
            }
        } catch (error) {
            console.error("Pipeline error:", error);
            setIsProcessing(false);
            setPipelineState({ phase: 0, status: 'error', progress: 0 });
        }
    };

    // ============================================================================
    // 파일 업로드
    // ============================================================================
    const handleFileUpload = async (files: FileList) => {
        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('files', file));

        try {
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                fetchInitialData();
            }
        } catch (error) {
            console.error("Upload error:", error);
        }
    };

    // ============================================================================
    // 렌더링
    // ============================================================================
    return (
        <LayoutNew
            activeView={activeView}
            onNavigate={setActiveView}
            isProcessing={isProcessing}
            runFullPipeline={runFullPipeline}
        >
            <div className="space-y-6">
                {/* Overview View */}
                {activeView === 'overview' && (
                    <OverviewSection
                        pipelineState={pipelineState}
                        silos={silos}
                        connections={connections}
                        siloPhase={siloPhase}
                        phase1Output={phase1Output}
                        phase2Output={phase2Output}
                        phase3Output={phase3Output}
                        onRunPipeline={runFullPipeline}
                        isProcessing={isProcessing}
                    />
                )}

                {/* Data View */}
                {activeView === 'data' && (
                    <DataSection
                        silos={silos}
                        connections={connections}
                        siloPhase={siloPhase}
                        uploadedFiles={uploadedFiles}
                        onUpload={handleFileUpload}
                        fileInputRef={fileInputRef}
                    />
                )}

                {/* Ontology View */}
                {activeView === 'ontology' && (
                    <OntologySection
                        entities={entities}
                        relationships={relationships}
                        consensusRounds={consensusRounds}
                        currentPhase={pipelineState.phase}
                        isLive={isProcessing}
                    />
                )}

                {/* Insights View */}
                {activeView === 'insights' && (
                    <InsightsSection insights={insights} />
                )}

                {/* Governance View */}
                {activeView === 'governance' && (
                    <GovernanceSection phase3Output={phase3Output} />
                )}

                {/* Logs View */}
                {activeView === 'logs' && (
                    <LogsSection logs={logs} />
                )}
            </div>
        </LayoutNew>
    );
}

// ============================================================================
// Overview 섹션
// ============================================================================
function OverviewSection({
    pipelineState,
    silos,
    connections,
    siloPhase,
    phase1Output,
    phase2Output,
    phase3Output,
    onRunPipeline,
    isProcessing,
}: any) {
    const stats = [
        { label: '데이터 소스', value: silos.length, icon: Database, color: 'text-blue-500' },
        { label: '발견된 연결', value: connections.length, icon: Link2, color: 'text-emerald-500' },
        { label: '인사이트', value: phase2Output?.insights?.length || 0, icon: TrendingUp, color: 'text-purple-500' },
        { label: '거버넌스 결정', value: phase3Output?.decisions?.length || 0, icon: Target, color: 'text-amber-500' },
    ];

    return (
        <div className="space-y-6">
            {/* 헤더 */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">대시보드</h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-1">통합 멀티에이전트 파이프라인 현황</p>
                </div>
                {pipelineState.status === 'running' && (
                    <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                        <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                        <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                            Phase {pipelineState.phase} 처리중...
                        </span>
                    </div>
                )}
            </div>

            {/* 통계 카드 */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {stats.map((stat, idx) => (
                    <motion.div
                        key={stat.label}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="p-4 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 shadow-sm"
                    >
                        <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg bg-gray-100 dark:bg-slate-700 ${stat.color}`}>
                                <stat.icon className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-2xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
                                <p className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</p>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* 데이터 사일로 Before/After */}
            <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                    <Database className="w-5 h-5 text-blue-500" />
                    데이터 사일로 통합 현황
                </h2>
                <DataSiloBeforeAfter
                    silos={silos}
                    connections={connections}
                    phase={siloPhase}
                    isAnimating={isProcessing}
                />
            </div>

            {/* Phase 진행 상태 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                    { phase: 1, name: 'Discovery', icon: Database, output: phase1Output, color: 'purple' },
                    { phase: 2, name: 'Refinement', icon: Brain, output: phase2Output, color: 'blue' },
                    { phase: 3, name: 'Governance', icon: Target, output: phase3Output, color: 'emerald' },
                ].map((p, idx) => (
                    <PhaseCard
                        key={p.phase}
                        phase={p.phase}
                        name={p.name}
                        icon={p.icon}
                        output={p.output}
                        color={p.color}
                        isActive={pipelineState.phase === p.phase}
                        isCompleted={pipelineState.phase > p.phase || (pipelineState.status === 'completed')}
                    />
                ))}
            </div>
        </div>
    );
}

// Phase 카드 컴포넌트
function PhaseCard({ phase, name, icon: Icon, output, color, isActive, isCompleted }: any) {
    const colorClasses: Record<string, string> = {
        purple: 'from-purple-500 to-purple-600 shadow-purple-500/20',
        blue: 'from-blue-500 to-blue-600 shadow-blue-500/20',
        emerald: 'from-emerald-500 to-emerald-600 shadow-emerald-500/20',
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`relative p-5 rounded-xl border transition-all ${isActive
                    ? 'bg-white dark:bg-slate-800 border-blue-500 shadow-lg'
                    : isCompleted
                        ? 'bg-white dark:bg-slate-800 border-emerald-500 shadow-md'
                        : 'bg-gray-50 dark:bg-slate-800/50 border-gray-200 dark:border-slate-700'
                }`}
        >
            {/* 상태 배지 */}
            <div className="absolute top-3 right-3">
                {isActive && (
                    <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse" />
                )}
                {isCompleted && !isActive && (
                    <CheckCircle className="w-5 h-5 text-emerald-500" />
                )}
            </div>

            {/* 헤더 */}
            <div className="flex items-center gap-3 mb-3">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${colorClasses[color]} shadow-lg`}>
                    <Icon className="w-5 h-5 text-white" />
                </div>
                <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Phase {phase}</p>
                    <h3 className="font-semibold text-gray-900 dark:text-white">{name}</h3>
                </div>
            </div>

            {/* 결과 요약 */}
            {output && (
                <div className="space-y-2 text-sm">
                    {phase === 1 && (
                        <>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">스키마 매핑</span>
                                <span className="font-medium text-gray-900 dark:text-white">{output.mappings?.length || 0}개</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">정규 객체</span>
                                <span className="font-medium text-gray-900 dark:text-white">{output.canonical_objects?.length || 0}개</span>
                            </div>
                        </>
                    )}
                    {phase === 2 && (
                        <>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">인사이트</span>
                                <span className="font-medium text-gray-900 dark:text-white">{output.insights?.length || 0}개</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">전체 점수</span>
                                <span className="font-medium text-emerald-600 dark:text-emerald-400">
                                    {((output.validation_scores?.overall_score || 0) * 100).toFixed(0)}%
                                </span>
                            </div>
                        </>
                    )}
                    {phase === 3 && (
                        <>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">승인</span>
                                <span className="font-medium text-emerald-600">{output.summary?.approved || 0}건</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">검토 필요</span>
                                <span className="font-medium text-amber-600">{output.summary?.review || 0}건</span>
                            </div>
                        </>
                    )}
                </div>
            )}

            {!output && !isActive && (
                <p className="text-sm text-gray-400 dark:text-gray-500">대기 중</p>
            )}
        </motion.div>
    );
}

// ============================================================================
// Data 섹션
// ============================================================================
function DataSection({ silos, connections, siloPhase, uploadedFiles, onUpload, fileInputRef }: any) {
    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">데이터 관리</h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-1">데이터 소스 및 사일로 현황</p>
                </div>
                <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors"
                >
                    <Upload className="w-4 h-4" />
                    파일 업로드
                </button>
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".csv,.json,.xlsx"
                    className="hidden"
                    onChange={(e) => e.target.files && onUpload(e.target.files)}
                />
            </div>

            {/* Before/After 시각화 */}
            <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                <DataSiloBeforeAfter
                    silos={silos}
                    connections={connections}
                    phase={siloPhase}
                />
            </div>

            {/* 업로드된 파일 목록 */}
            <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">업로드된 파일</h2>
                {uploadedFiles.length === 0 ? (
                    <div className="text-center py-12 text-gray-400">
                        <Upload className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>업로드된 파일이 없습니다</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {uploadedFiles.map((file: any, idx: number) => (
                            <div
                                key={idx}
                                className="p-4 rounded-lg border border-gray-200 dark:border-slate-700 hover:border-blue-500 transition-colors"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                                        <FileText className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="font-medium text-gray-900 dark:text-white truncate">{file.name}</p>
                                        <p className="text-sm text-gray-500 dark:text-gray-400">
                                            {file.row_count?.toLocaleString() || '?'} 행
                                        </p>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

// ============================================================================
// Ontology 섹션
// ============================================================================
function OntologySection({ entities, relationships, consensusRounds, currentPhase, isLive }: any) {
    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">온톨로지</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">엔티티 관계 및 에이전트 합의 과정</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 에이전트 대화 패널 */}
                <div className="h-[600px]">
                    <AgentConversationPanel
                        rounds={consensusRounds}
                        currentPhase={currentPhase}
                        isLive={isLive}
                    />
                </div>

                {/* 엔티티 관계 그래프 */}
                <div>
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                        <Layers className="w-5 h-5 text-emerald-500" />
                        엔티티 관계 그래프
                    </h2>
                    <EntityRelationshipGraph
                        entities={entities}
                        relationships={relationships}
                    />
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// Insights 섹션
// ============================================================================
function InsightsSection({ insights }: { insights: PredictionData[] }) {
    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">인사이트</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">예측 분석 및 권장 사항</p>
            </div>

            {insights.length === 0 ? (
                <div className="p-12 text-center rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                    <BarChart3 className="w-12 h-12 mx-auto mb-3 text-gray-300 dark:text-gray-600" />
                    <p className="text-gray-500 dark:text-gray-400">아직 인사이트가 없습니다</p>
                    <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">파이프라인을 실행하면 인사이트가 생성됩니다</p>
                </div>
            ) : (
                <PalantirInsightGrid
                    insights={insights}
                    onExpand={(id) => console.log('Expand:', id)}
                    onAction={(id, action) => console.log('Action:', id, action)}
                />
            )}
        </div>
    );
}

// ============================================================================
// Governance 섹션
// ============================================================================
function GovernanceSection({ phase3Output }: { phase3Output: any }) {
    const summary = phase3Output?.summary || { approved: 0, escalated: 0, review: 0 };
    const actions = phase3Output?.actions || [];

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">거버넌스</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">결정 및 액션 관리</p>
            </div>

            {/* 요약 통계 */}
            <div className="grid grid-cols-3 gap-4">
                <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800">
                    <div className="flex items-center gap-3">
                        <CheckCircle className="w-8 h-8 text-emerald-500" />
                        <div>
                            <p className="text-2xl font-bold text-emerald-700 dark:text-emerald-300">{summary.approved}</p>
                            <p className="text-sm text-emerald-600 dark:text-emerald-400">자동 승인</p>
                        </div>
                    </div>
                </div>
                <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                    <div className="flex items-center gap-3">
                        <Eye className="w-8 h-8 text-amber-500" />
                        <div>
                            <p className="text-2xl font-bold text-amber-700 dark:text-amber-300">{summary.review}</p>
                            <p className="text-sm text-amber-600 dark:text-amber-400">검토 필요</p>
                        </div>
                    </div>
                </div>
                <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                    <div className="flex items-center gap-3">
                        <AlertTriangle className="w-8 h-8 text-red-500" />
                        <div>
                            <p className="text-2xl font-bold text-red-700 dark:text-red-300">{summary.escalated}</p>
                            <p className="text-sm text-red-600 dark:text-red-400">에스컬레이션</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* 액션 목록 */}
            <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">액션 백로그</h2>
                {actions.length === 0 ? (
                    <div className="text-center py-8 text-gray-400">
                        <Target className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>액션이 없습니다</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {actions.map((action: any, idx: number) => (
                            <div
                                key={idx}
                                className="p-4 rounded-lg border border-gray-200 dark:border-slate-700 hover:border-blue-500 transition-colors"
                            >
                                <div className="flex items-start justify-between">
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="px-2 py-0.5 rounded text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                                                {action.action_id || `ACT-${idx + 1}`}
                                            </span>
                                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${action.priority === 'P1' ? 'bg-red-100 text-red-700' :
                                                    action.priority === 'P2' ? 'bg-amber-100 text-amber-700' :
                                                        'bg-gray-100 text-gray-700'
                                                }`}>
                                                {action.priority || 'P3'}
                                            </span>
                                        </div>
                                        <p className="text-sm text-gray-700 dark:text-gray-300">{action.recommendation || action.reason || '액션 설명'}</p>
                                    </div>
                                    <span className={`text-xs font-medium ${action.decision_type === 'approve' ? 'text-emerald-600' :
                                            action.decision_type === 'escalate' ? 'text-red-600' :
                                                'text-amber-600'
                                        }`}>
                                        {action.decision_type === 'approve' ? '승인' :
                                            action.decision_type === 'escalate' ? '에스컬레이션' : '검토'}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

// ============================================================================
// Logs 섹션
// ============================================================================
function LogsSection({ logs }: { logs: string[] }) {
    const logsEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">실시간 로그</h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-1">파이프라인 실행 로그</p>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-100 dark:bg-red-900/30">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-xs font-medium text-red-600 dark:text-red-400">LIVE</span>
                </div>
            </div>

            <div className="h-[600px] rounded-xl bg-gray-900 dark:bg-black border border-gray-700 overflow-hidden">
                <div className="h-full overflow-y-auto p-4 font-mono text-sm">
                    {logs.length === 0 ? (
                        <div className="text-gray-500 text-center py-8">
                            로그가 없습니다. 파이프라인을 실행하세요.
                        </div>
                    ) : (
                        <>
                            {logs.map((log, idx) => (
                                <div
                                    key={idx}
                                    className={`py-1 ${log.includes('ERROR') ? 'text-red-400' :
                                            log.includes('WARNING') ? 'text-amber-400' :
                                                log.includes('SUCCESS') || log.includes('완료') ? 'text-emerald-400' :
                                                    'text-gray-300'
                                        }`}
                                >
                                    {log}
                                </div>
                            ))}
                            <div ref={logsEndRef} />
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
