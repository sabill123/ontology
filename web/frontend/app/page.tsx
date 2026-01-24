"use client";

import React, { useState, useEffect, useRef } from 'react';
import {
    Database, Brain, Target, Play, ChevronRight, AlertTriangle, CheckCircle,
    Upload, Eye, TrendingUp, Link2, ArrowRight,
    BarChart3, Layers, FileText, Sun, Moon
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTheme } from '@/contexts/ThemeContext';

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

type ViewType = 'overview' | 'data' | 'ontology' | 'insights' | 'governance' | 'logs';

// ============================================================================
// 테마 토글 버튼
// ============================================================================
function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className="relative flex items-center justify-center w-10 h-10 rounded-full
                       bg-gray-100 dark:bg-slate-800
                       hover:bg-gray-200 dark:hover:bg-slate-700
                       border border-gray-200 dark:border-slate-700
                       transition-all duration-300 group"
            aria-label={theme === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환'}
        >
            <Sun
                className={`absolute w-5 h-5 text-amber-500 transition-all duration-300
                           ${theme === 'light' ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 rotate-90 scale-0'}`}
            />
            <Moon
                className={`absolute w-5 h-5 text-blue-400 transition-all duration-300
                           ${theme === 'dark' ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-0'}`}
            />
        </button>
    );
}

// ============================================================================
// 레이아웃 컴포넌트
// ============================================================================
function Layout({ children, activeView, onNavigate, isProcessing, runFullPipeline }: {
    children: React.ReactNode;
    activeView: ViewType;
    onNavigate: (view: ViewType) => void;
    isProcessing?: boolean;
    runFullPipeline?: () => void;
}) {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => setIsScrolled(window.scrollY > 20);
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navLinks: { id: ViewType; label: string; icon: React.ElementType }[] = [
        { id: 'overview', label: '개요', icon: BarChart3 },
        { id: 'data', label: '데이터', icon: Database },
        { id: 'ontology', label: '온톨로지', icon: Brain },
        { id: 'insights', label: '인사이트', icon: TrendingUp },
        { id: 'governance', label: '거버넌스', icon: Target },
        { id: 'logs', label: '로그', icon: FileText },
    ];

    return (
        <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-slate-900 text-gray-900 dark:text-white transition-colors duration-300">
            {/* Navbar */}
            <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300
                ${isScrolled ? 'bg-white/90 dark:bg-slate-900/90 backdrop-blur-lg shadow-sm border-b border-gray-200 dark:border-slate-800' : 'bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-slate-800'}`}>
                <div className="max-w-7xl mx-auto px-4 md:px-6">
                    <div className="flex items-center justify-between h-16">
                        {/* Logo & Nav */}
                        <div className="flex items-center gap-8">
                            <div className="flex items-center gap-2 cursor-pointer" onClick={() => onNavigate('overview')}>
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
                                    <Brain className="w-5 h-5 text-white" />
                                </div>
                                <span className="text-lg font-bold tracking-tight">
                                    <span className="text-emerald-600 dark:text-emerald-400">LET</span>
                                    <span className="text-gray-700 dark:text-gray-300">OLOGY</span>
                                </span>
                            </div>

                            {/* Desktop Nav */}
                            <div className="hidden md:flex items-center gap-1">
                                {navLinks.map(link => (
                                    <button
                                        key={link.id}
                                        className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all
                                            ${activeView === link.id
                                                ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
                                                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-800'}`}
                                        onClick={() => onNavigate(link.id)}
                                    >
                                        <link.icon className="w-4 h-4" />
                                        {link.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Right Actions */}
                        <div className="flex items-center gap-3">
                            {runFullPipeline && (
                                <button
                                    onClick={runFullPipeline}
                                    disabled={isProcessing}
                                    className="hidden md:flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                                               bg-gradient-to-r from-emerald-500 to-cyan-500 text-white
                                               hover:from-emerald-600 hover:to-cyan-600
                                               disabled:opacity-50 shadow-lg shadow-emerald-500/20
                                               transition-all active:scale-95"
                                >
                                    {isProcessing ? (
                                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    ) : (
                                        <Play className="w-4 h-4" fill="currentColor" />
                                    )}
                                    {isProcessing ? '처리중...' : '파이프라인 실행'}
                                </button>
                            )}

                            {/* Status */}
                            <div className="hidden md:flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                                <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'}`} />
                                <span className="text-xs text-gray-600 dark:text-gray-400">
                                    {isProcessing ? '처리 중' : '온라인'}
                                </span>
                            </div>

                            {/* Theme Toggle */}
                            <ThemeToggle />

                            {/* Mobile Menu */}
                            <button
                                className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-800"
                                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                            >
                                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    {isMobileMenuOpen ? (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    ) : (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                                    )}
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mobile Menu */}
                {isMobileMenuOpen && (
                    <div className="md:hidden bg-white dark:bg-slate-900 border-t border-gray-200 dark:border-slate-800 p-4">
                        {navLinks.map(link => (
                            <button
                                key={link.id}
                                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-1 ${activeView === link.id
                                    ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
                                    : 'text-gray-600 dark:text-gray-400'}`}
                                onClick={() => { onNavigate(link.id); setIsMobileMenuOpen(false); }}
                            >
                                <link.icon className="w-5 h-5" />
                                {link.label}
                            </button>
                        ))}
                    </div>
                )}
            </nav>

            {/* Main */}
            <main className="flex-grow pt-20 pb-8 min-h-screen">
                <div className="max-w-7xl mx-auto px-4 md:px-6">
                    {children}
                </div>
            </main>

            {/* Footer */}
            <footer className="bg-white dark:bg-slate-900 border-t border-gray-200 dark:border-slate-800 py-6">
                <div className="max-w-7xl mx-auto px-4 md:px-6 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-md bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
                            <Brain className="w-4 h-4 text-white" />
                        </div>
                        <span className="text-sm font-semibold text-gray-400">LETOLOGY</span>
                    </div>
                    <span className="text-xs text-gray-400">© 2025 Letology Inc. v2.0.0</span>
                </div>
            </footer>
        </div>
    );
}

// ============================================================================
// 데이터 사일로 Before/After 시각화
// ============================================================================
function DataSiloVisualization({ silos, connections, phase }: {
    silos: DataSilo[];
    connections: SiloConnection[];
    phase: 'before' | 'after';
}) {
    return (
        <div className="w-full">
            {/* 상태 표시 */}
            <div className="flex items-center gap-4 mb-6">
                <div className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all
                    ${phase === 'before' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' : 'bg-gray-100 text-gray-500 dark:bg-slate-800'}`}>
                    <AlertTriangle className="w-4 h-4" />
                    <span className="text-sm font-medium">사일로 상태</span>
                </div>
                <ArrowRight className="w-5 h-5 text-gray-400" />
                <div className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all
                    ${phase === 'after' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300' : 'bg-gray-100 text-gray-500 dark:bg-slate-800'}`}>
                    <Layers className="w-4 h-4" />
                    <span className="text-sm font-medium">통합 완료</span>
                </div>
                <div className="ml-auto text-sm text-gray-500 dark:text-gray-400">
                    <span className="font-semibold text-gray-900 dark:text-white">{silos.length}</span> 데이터 소스 |{' '}
                    <span className="font-semibold text-emerald-600">{connections.length}</span> 연결
                </div>
            </div>

            {/* 사일로 그리드 */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {silos.map((silo, idx) => (
                    <motion.div
                        key={silo.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className={`relative p-4 rounded-xl border-2 transition-all
                            ${phase === 'before'
                                ? 'bg-white dark:bg-slate-800 border-red-200 dark:border-red-800 shadow-lg shadow-red-100 dark:shadow-red-900/20'
                                : 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700 shadow-lg shadow-emerald-100 dark:shadow-emerald-900/20'}`}
                    >
                        <div className="absolute -top-2 -right-2">
                            {phase === 'before' ? (
                                <div className="w-6 h-6 rounded-full bg-red-100 dark:bg-red-900 flex items-center justify-center">
                                    <AlertTriangle className="w-4 h-4 text-red-500" />
                                </div>
                            ) : (
                                <div className="w-6 h-6 rounded-full bg-emerald-100 dark:bg-emerald-900 flex items-center justify-center">
                                    <CheckCircle className="w-4 h-4 text-emerald-500" />
                                </div>
                            )}
                        </div>

                        <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${phase === 'before' ? 'bg-red-100 dark:bg-red-900/50' : 'bg-emerald-100 dark:bg-emerald-900/50'}`}>
                                <Database className={`w-5 h-5 ${phase === 'before' ? 'text-red-500' : 'text-emerald-500'}`} />
                            </div>
                            <div>
                                <h4 className="font-semibold text-gray-900 dark:text-white">{silo.name}</h4>
                                <p className="text-xs text-gray-500 dark:text-gray-400">{silo.tables.length}개 테이블</p>
                            </div>
                        </div>

                        <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">행 수</span>
                                <span className="font-medium text-gray-900 dark:text-white">{silo.rowCount.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">품질</span>
                                <span className={`font-medium ${silo.qualityScore >= 0.8 ? 'text-emerald-500' : silo.qualityScore >= 0.6 ? 'text-amber-500' : 'text-red-500'}`}>
                                    {(silo.qualityScore * 100).toFixed(0)}%
                                </span>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* 통합 허브 (After 상태) */}
            {phase === 'after' && connections.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="mt-6 p-4 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800"
                >
                    <h4 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3 flex items-center gap-2">
                        <Link2 className="w-4 h-4" />
                        발견된 연결 관계 ({connections.length}개)
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                        {connections.slice(0, 6).map((conn, idx) => (
                            <div key={idx} className="flex items-center gap-2 p-2 rounded-lg bg-white dark:bg-slate-800 text-sm">
                                <span className="font-medium text-gray-900 dark:text-white">{conn.from}</span>
                                <ArrowRight className="w-3 h-3 text-emerald-500" />
                                <span className="font-medium text-gray-900 dark:text-white">{conn.to}</span>
                                <span className={`ml-auto text-xs px-2 py-0.5 rounded-full
                                    ${conn.type === 'homeomorphism' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300' :
                                        conn.type === 'semantic' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300' :
                                            'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'}`}>
                                    {conn.type === 'homeomorphism' ? '위상동형' : conn.type === 'semantic' ? '의미적' : '외래키'}
                                </span>
                            </div>
                        ))}
                    </div>
                </motion.div>
            )}
        </div>
    );
}

// ============================================================================
// Phase 카드
// ============================================================================
function PhaseCard({ phase, name, icon: Icon, output, color, isActive, isCompleted }: {
    phase: number;
    name: string;
    icon: React.ElementType;
    output: any;
    color: string;
    isActive: boolean;
    isCompleted: boolean;
}) {
    const colorClasses: Record<string, string> = {
        purple: 'from-purple-500 to-purple-600',
        blue: 'from-blue-500 to-blue-600',
        emerald: 'from-emerald-500 to-emerald-600',
    };

    return (
        <div className={`relative p-5 rounded-xl border transition-all ${isActive
            ? 'bg-white dark:bg-slate-800 border-blue-500 shadow-lg'
            : isCompleted
                ? 'bg-white dark:bg-slate-800 border-emerald-500 shadow-md'
                : 'bg-gray-50 dark:bg-slate-800/50 border-gray-200 dark:border-slate-700'}`}>
            <div className="absolute top-3 right-3">
                {isActive && <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse" />}
                {isCompleted && !isActive && <CheckCircle className="w-5 h-5 text-emerald-500" />}
            </div>

            <div className="flex items-center gap-3 mb-3">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${colorClasses[color]} shadow-lg`}>
                    <Icon className="w-5 h-5 text-white" />
                </div>
                <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Phase {phase}</p>
                    <h3 className="font-semibold text-gray-900 dark:text-white">{name}</h3>
                </div>
            </div>

            {output ? (
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
                                <span className="font-medium text-emerald-600">{((output.validation_scores?.overall_score || 0) * 100).toFixed(0)}%</span>
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
            ) : (
                <p className="text-sm text-gray-400 dark:text-gray-500">대기 중</p>
            )}
        </div>
    );
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
    const [siloPhase, setSiloPhase] = useState<'before' | 'after'>('before');
    const [logs, setLogs] = useState<string[]>([]);

    // Phase 결과
    const [phase1Output, setPhase1Output] = useState<any>(null);
    const [phase2Output, setPhase2Output] = useState<any>(null);
    const [phase3Output, setPhase3Output] = useState<any>(null);

    // 업로드
    const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // WebSocket for logs
    useEffect(() => {
        let ws: WebSocket | null = null;
        let reconnectTimeout: NodeJS.Timeout;

        const connect = () => {
            ws = new WebSocket("ws://localhost:4200/api/ws/logs");
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.logs) setLogs(data.logs);
                } catch { }
            };
            ws.onclose = () => { reconnectTimeout = setTimeout(connect, 2000); };
        };
        connect();

        return () => { ws?.close(); clearTimeout(reconnectTimeout); };
    }, []);

    // Fetch data
    useEffect(() => {
        fetchInitialData();
    }, []);

    const fetchInitialData = async () => {
        try {
            // Files
            const filesRes = await fetch(`${API_URL}/phase1/files`);
            if (filesRes.ok) {
                const files = await filesRes.json();
                setUploadedFiles(files);
                if (files.length > 0) {
                    setSilos(files.map((f: any, idx: number) => ({
                        id: `silo-${idx}`,
                        name: f.name || `Source ${idx + 1}`,
                        tables: f.tables || [f.name],
                        rowCount: f.row_count || Math.floor(Math.random() * 100000),
                        qualityScore: f.quality_score || 0.7 + Math.random() * 0.3,
                    })));
                }
            }

            // Phase outputs
            const [p1, p2, p3] = await Promise.all([
                fetch(`${API_URL}/phase1/output`).then(r => r.json()).catch(() => null),
                fetch(`${API_URL}/phase2/output`).then(r => r.json()).catch(() => null),
                fetch(`${API_URL}/phase3/output`).then(r => r.json()).catch(() => null),
            ]);

            if (p1 && !p1.error) {
                setPhase1Output(p1);
                if (p1.mappings?.length > 0) {
                    setConnections(p1.mappings.map((m: any, idx: number) => ({
                        from: m.source_a || m.table_a || `Table ${idx * 2}`,
                        to: m.source_b || m.table_b || `Table ${idx * 2 + 1}`,
                        type: m.mapping_type || 'homeomorphism',
                        confidence: m.confidence || 0.85,
                    })));
                    setSiloPhase('after');
                }
            }
            if (p2 && !p2.error) setPhase2Output(p2);
            if (p3 && !p3.error) setPhase3Output(p3);
        } catch (e) {
            console.error("Fetch error:", e);
        }
    };

    // Run pipeline
    const runFullPipeline = async () => {
        setIsProcessing(true);
        setPipelineState({ phase: 1, status: 'running', progress: 0 });

        try {
            const response = await fetch(`${API_URL}/pipeline/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });

            if (response.ok) {
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
                            fetchInitialData();
                        } else if (status.status === 'running') {
                            setTimeout(pollStatus, 2000);
                        }
                    }
                };
                pollStatus();
            }
        } catch (e) {
            console.error("Pipeline error:", e);
            setIsProcessing(false);
        }
    };

    // File upload
    const handleFileUpload = async (files: FileList) => {
        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('files', file));
        try {
            const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
            if (res.ok) fetchInitialData();
        } catch (e) { console.error("Upload error:", e); }
    };

    // Stats
    const stats = [
        { label: '데이터 소스', value: silos.length, icon: Database, color: 'text-blue-500' },
        { label: '발견된 연결', value: connections.length, icon: Link2, color: 'text-emerald-500' },
        { label: '인사이트', value: phase2Output?.insights?.length || 0, icon: TrendingUp, color: 'text-purple-500' },
        { label: '거버넌스 결정', value: phase3Output?.decisions?.length || 0, icon: Target, color: 'text-amber-500' },
    ];

    return (
        <Layout
            activeView={activeView}
            onNavigate={setActiveView}
            isProcessing={isProcessing}
            runFullPipeline={runFullPipeline}
        >
            <div className="space-y-6">
                {/* Overview */}
                {activeView === 'overview' && (
                    <>
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                            <div>
                                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">대시보드</h1>
                                <p className="text-gray-500 dark:text-gray-400 mt-1">통합 멀티에이전트 파이프라인 현황</p>
                            </div>
                            {pipelineState.status === 'running' && (
                                <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                                    <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                                    <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Phase {pipelineState.phase} 처리중...</span>
                                </div>
                            )}
                        </div>

                        {/* Stats */}
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

                        {/* Silo Visualization */}
                        <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                <Database className="w-5 h-5 text-blue-500" />
                                데이터 사일로 통합 현황
                            </h2>
                            <DataSiloVisualization silos={silos} connections={connections} phase={siloPhase} />
                        </div>

                        {/* Phase Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <PhaseCard phase={1} name="Discovery" icon={Database} output={phase1Output} color="purple" isActive={pipelineState.phase === 1} isCompleted={pipelineState.phase > 1 || pipelineState.status === 'completed'} />
                            <PhaseCard phase={2} name="Refinement" icon={Brain} output={phase2Output} color="blue" isActive={pipelineState.phase === 2} isCompleted={pipelineState.phase > 2 || pipelineState.status === 'completed'} />
                            <PhaseCard phase={3} name="Governance" icon={Target} output={phase3Output} color="emerald" isActive={pipelineState.phase === 3} isCompleted={pipelineState.status === 'completed'} />
                        </div>
                    </>
                )}

                {/* Data View */}
                {activeView === 'data' && (
                    <>
                        <div className="flex items-center justify-between">
                            <div>
                                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">데이터 관리</h1>
                                <p className="text-gray-500 dark:text-gray-400 mt-1">데이터 소스 및 사일로 현황</p>
                            </div>
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600"
                            >
                                <Upload className="w-4 h-4" />
                                파일 업로드
                            </button>
                            <input ref={fileInputRef} type="file" multiple accept=".csv,.json,.xlsx" className="hidden" onChange={(e) => e.target.files && handleFileUpload(e.target.files)} />
                        </div>

                        <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                            <DataSiloVisualization silos={silos} connections={connections} phase={siloPhase} />
                        </div>

                        <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">업로드된 파일</h2>
                            {uploadedFiles.length === 0 ? (
                                <div className="text-center py-12 text-gray-400">
                                    <Upload className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                    <p>업로드된 파일이 없습니다</p>
                                </div>
                            ) : (
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {uploadedFiles.map((file, idx) => (
                                        <div key={idx} className="p-4 rounded-lg border border-gray-200 dark:border-slate-700 hover:border-blue-500">
                                            <div className="flex items-center gap-3">
                                                <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                                                    <FileText className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                                                </div>
                                                <div>
                                                    <p className="font-medium text-gray-900 dark:text-white truncate">{file.name}</p>
                                                    <p className="text-sm text-gray-500 dark:text-gray-400">{file.row_count?.toLocaleString() || '?'} 행</p>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </>
                )}

                {/* Insights View */}
                {activeView === 'insights' && (
                    <>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">인사이트</h1>
                        <p className="text-gray-500 dark:text-gray-400 mb-6">예측 분석 및 권장 사항</p>

                        {!phase2Output?.insights?.length ? (
                            <div className="p-12 text-center rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                                <BarChart3 className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                                <p className="text-gray-500">파이프라인을 실행하면 인사이트가 생성됩니다</p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {phase2Output.insights.slice(0, 6).map((ins: any, idx: number) => (
                                    <div key={idx} className="p-4 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                                        <div className="flex items-start gap-3">
                                            <div className={`px-2 py-1 rounded text-xs font-medium
                                                ${ins.severity === 'high' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' :
                                                    ins.severity === 'low' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' :
                                                        'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'}`}>
                                                {ins.severity === 'high' ? '높음' : ins.severity === 'low' ? '낮음' : '보통'}
                                            </div>
                                            <div className="flex-1">
                                                <p className="text-sm text-gray-900 dark:text-white mb-1">{ins.explanation?.slice(0, 150) || '설명 없음'}...</p>
                                                <p className="text-xs text-gray-500">신뢰도: {((ins.confidence || 0) * 100).toFixed(0)}%</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}

                {/* Governance View */}
                {activeView === 'governance' && (
                    <>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">거버넌스</h1>
                        <p className="text-gray-500 dark:text-gray-400 mb-6">결정 및 액션 관리</p>

                        <div className="grid grid-cols-3 gap-4 mb-6">
                            <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800">
                                <CheckCircle className="w-8 h-8 text-emerald-500 mb-2" />
                                <p className="text-2xl font-bold text-emerald-700 dark:text-emerald-300">{phase3Output?.summary?.approved || 0}</p>
                                <p className="text-sm text-emerald-600 dark:text-emerald-400">자동 승인</p>
                            </div>
                            <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                                <Eye className="w-8 h-8 text-amber-500 mb-2" />
                                <p className="text-2xl font-bold text-amber-700 dark:text-amber-300">{phase3Output?.summary?.review || 0}</p>
                                <p className="text-sm text-amber-600 dark:text-amber-400">검토 필요</p>
                            </div>
                            <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                                <AlertTriangle className="w-8 h-8 text-red-500 mb-2" />
                                <p className="text-2xl font-bold text-red-700 dark:text-red-300">{phase3Output?.summary?.escalated || 0}</p>
                                <p className="text-sm text-red-600 dark:text-red-400">에스컬레이션</p>
                            </div>
                        </div>

                        <div className="p-6 rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">액션 백로그</h2>
                            {!phase3Output?.actions?.length ? (
                                <div className="text-center py-8 text-gray-400">
                                    <Target className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                    <p>액션이 없습니다</p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {phase3Output.actions.slice(0, 5).map((action: any, idx: number) => (
                                        <div key={idx} className="p-3 rounded-lg border border-gray-200 dark:border-slate-700">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="text-xs px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                                                    {action.action_id || `ACT-${idx + 1}`}
                                                </span>
                                                <span className={`text-xs px-2 py-0.5 rounded ${action.priority === 'P1' ? 'bg-red-100 text-red-700' : action.priority === 'P2' ? 'bg-amber-100 text-amber-700' : 'bg-gray-100 text-gray-700'}`}>
                                                    {action.priority || 'P3'}
                                                </span>
                                            </div>
                                            <p className="text-sm text-gray-700 dark:text-gray-300">{action.recommendation || action.reason || '설명 없음'}</p>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </>
                )}

                {/* Logs View */}
                {activeView === 'logs' && (
                    <>
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">실시간 로그</h1>
                                <p className="text-gray-500 dark:text-gray-400">파이프라인 실행 로그</p>
                            </div>
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-100 dark:bg-red-900/30">
                                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                <span className="text-xs font-medium text-red-600 dark:text-red-400">LIVE</span>
                            </div>
                        </div>

                        <div className="h-[600px] rounded-xl bg-gray-900 border border-gray-700 overflow-hidden">
                            <div className="h-full overflow-y-auto p-4 font-mono text-sm">
                                {logs.length === 0 ? (
                                    <div className="text-gray-500 text-center py-8">파이프라인을 실행하세요</div>
                                ) : (
                                    logs.map((log, idx) => (
                                        <div key={idx} className={`py-1 ${log.includes('ERROR') ? 'text-red-400' : log.includes('WARNING') ? 'text-amber-400' : log.includes('완료') ? 'text-emerald-400' : 'text-gray-300'}`}>
                                            {log}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </>
                )}

                {/* Ontology View - placeholder */}
                {activeView === 'ontology' && (
                    <>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">온톨로지</h1>
                        <p className="text-gray-500 dark:text-gray-400 mb-6">엔티티 관계 및 에이전트 합의</p>
                        <div className="p-12 text-center rounded-xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                            <Brain className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                            <p className="text-gray-500">온톨로지 그래프가 여기에 표시됩니다</p>
                        </div>
                    </>
                )}
            </div>
        </Layout>
    );
}
