"use client";

import React, { useState } from 'react';
import { Database, ArrowRight, Link2, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

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

interface DataSiloBeforeAfterProps {
    silos: DataSilo[];
    connections: SiloConnection[];
    phase: 'before' | 'during' | 'after';
    isAnimating?: boolean;
}

// 개별 데이터 사일로 카드
function SiloCard({ silo, isConnected, phase }: { silo: DataSilo; isConnected: boolean; phase: string }) {
    const qualityColor = silo.qualityScore >= 0.8 ? 'text-emerald-500' : silo.qualityScore >= 0.6 ? 'text-amber-500' : 'text-red-500';

    return (
        <motion.div
            layout
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className={`
                relative p-4 rounded-xl border-2 transition-all duration-500
                ${phase === 'before'
                    ? 'bg-white dark:bg-slate-800 border-red-200 dark:border-red-800 shadow-lg shadow-red-100 dark:shadow-red-900/20'
                    : isConnected
                        ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700 shadow-lg shadow-emerald-100 dark:shadow-emerald-900/20'
                        : 'bg-white dark:bg-slate-800 border-gray-200 dark:border-slate-700'}
            `}
        >
            {/* 상태 아이콘 */}
            <div className="absolute -top-2 -right-2">
                {phase === 'before' ? (
                    <div className="w-6 h-6 rounded-full bg-red-100 dark:bg-red-900 flex items-center justify-center">
                        <AlertTriangle className="w-4 h-4 text-red-500" />
                    </div>
                ) : isConnected ? (
                    <div className="w-6 h-6 rounded-full bg-emerald-100 dark:bg-emerald-900 flex items-center justify-center">
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    </div>
                ) : null}
            </div>

            {/* 헤더 */}
            <div className="flex items-center gap-3 mb-3">
                <div className={`p-2 rounded-lg ${phase === 'before' ? 'bg-red-100 dark:bg-red-900/50' : 'bg-emerald-100 dark:bg-emerald-900/50'}`}>
                    <Database className={`w-5 h-5 ${phase === 'before' ? 'text-red-500' : 'text-emerald-500'}`} />
                </div>
                <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white">{silo.name}</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400">{silo.tables.length}개 테이블</p>
                </div>
            </div>

            {/* 통계 */}
            <div className="space-y-2">
                <div className="flex justify-between text-sm">
                    <span className="text-gray-500 dark:text-gray-400">행 수</span>
                    <span className="font-medium text-gray-900 dark:text-white">{silo.rowCount.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                    <span className="text-gray-500 dark:text-gray-400">품질 점수</span>
                    <span className={`font-medium ${qualityColor}`}>{(silo.qualityScore * 100).toFixed(0)}%</span>
                </div>
            </div>

            {/* 테이블 목록 */}
            <div className="mt-3 pt-3 border-t border-gray-100 dark:border-slate-700">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">테이블:</p>
                <div className="flex flex-wrap gap-1">
                    {silo.tables.slice(0, 3).map(table => (
                        <span
                            key={table}
                            className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300"
                        >
                            {table}
                        </span>
                    ))}
                    {silo.tables.length > 3 && (
                        <span className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-slate-700 text-gray-500">
                            +{silo.tables.length - 3}
                        </span>
                    )}
                </div>
            </div>
        </motion.div>
    );
}

// 연결 애니메이션
function ConnectionLine({ from, to, type, confidence, phase }: SiloConnection & { phase: string }) {
    if (phase === 'before') return null;

    const typeColors = {
        foreign_key: 'stroke-blue-500',
        semantic: 'stroke-purple-500',
        homeomorphism: 'stroke-emerald-500',
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`
                absolute z-10 flex items-center justify-center
                px-2 py-1 rounded-full text-xs font-medium
                ${type === 'homeomorphism' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300' :
                    type === 'semantic' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300' :
                        'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'}
            `}
            style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}
        >
            <Link2 className="w-3 h-3 mr-1" />
            {(confidence * 100).toFixed(0)}%
        </motion.div>
    );
}

export function DataSiloBeforeAfter({ silos, connections, phase, isAnimating }: DataSiloBeforeAfterProps) {
    const [selectedSilo, setSelectedSilo] = useState<string | null>(null);

    // 연결된 사일로 ID 목록
    const connectedSilos = new Set(connections.flatMap(c => [c.from, c.to]));

    return (
        <div className="w-full">
            {/* 상태 헤더 */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all
                        ${phase === 'before'
                            ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                            : 'bg-gray-100 text-gray-500 dark:bg-slate-800 dark:text-gray-400'}`}>
                        <AlertTriangle className="w-4 h-4" />
                        <span className="text-sm font-medium">사일로 상태</span>
                    </div>
                    <ArrowRight className="w-5 h-5 text-gray-400" />
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all
                        ${phase === 'after'
                            ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
                            : 'bg-gray-100 text-gray-500 dark:bg-slate-800 dark:text-gray-400'}`}>
                        <Layers className="w-4 h-4" />
                        <span className="text-sm font-medium">통합 완료</span>
                    </div>
                </div>

                {/* 통계 요약 */}
                <div className="flex items-center gap-4 text-sm">
                    <div className="text-gray-500 dark:text-gray-400">
                        <span className="font-semibold text-gray-900 dark:text-white">{silos.length}</span> 데이터 소스
                    </div>
                    <div className="text-gray-500 dark:text-gray-400">
                        <span className="font-semibold text-emerald-600 dark:text-emerald-400">{connections.length}</span> 연결 발견
                    </div>
                </div>
            </div>

            {/* 사일로 그리드 */}
            <div className="relative">
                {/* Before: 분산된 사일로 */}
                <AnimatePresence mode="wait">
                    {phase === 'before' ? (
                        <motion.div
                            key="before"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6"
                        >
                            {silos.map((silo, index) => (
                                <motion.div
                                    key={silo.id}
                                    initial={{ y: 20, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    transition={{ delay: index * 0.1 }}
                                >
                                    <SiloCard silo={silo} isConnected={false} phase="before" />
                                </motion.div>
                            ))}
                        </motion.div>
                    ) : (
                        <motion.div
                            key="after"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="relative"
                        >
                            {/* 통합 허브 */}
                            <div className="flex flex-col items-center mb-8">
                                <motion.div
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className="relative w-32 h-32 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-500
                                               flex items-center justify-center shadow-2xl shadow-emerald-500/30"
                                >
                                    <div className="absolute inset-2 rounded-full bg-white dark:bg-slate-900 flex items-center justify-center">
                                        <div className="text-center">
                                            <Layers className="w-8 h-8 text-emerald-500 mx-auto mb-1" />
                                            <span className="text-xs font-semibold text-gray-900 dark:text-white">통합 온톨로지</span>
                                        </div>
                                    </div>
                                    {/* 펄스 애니메이션 */}
                                    <div className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-20" />
                                </motion.div>
                            </div>

                            {/* 연결된 사일로들 */}
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                                {silos.map((silo, index) => (
                                    <motion.div
                                        key={silo.id}
                                        initial={{ y: 50, opacity: 0 }}
                                        animate={{ y: 0, opacity: 1 }}
                                        transition={{ delay: 0.3 + index * 0.1 }}
                                    >
                                        <SiloCard
                                            silo={silo}
                                            isConnected={connectedSilos.has(silo.id)}
                                            phase="after"
                                        />
                                    </motion.div>
                                ))}
                            </div>

                            {/* 연결 정보 패널 */}
                            <motion.div
                                initial={{ y: 20, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                transition={{ delay: 0.5 }}
                                className="mt-6 p-4 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800"
                            >
                                <h4 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3 flex items-center gap-2">
                                    <Link2 className="w-4 h-4" />
                                    발견된 연결 관계
                                </h4>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                                    {connections.map((conn, idx) => (
                                        <div
                                            key={idx}
                                            className="flex items-center gap-2 p-2 rounded-lg bg-white dark:bg-slate-800 text-sm"
                                        >
                                            <span className="font-medium text-gray-900 dark:text-white">{conn.from}</span>
                                            <ArrowRight className="w-3 h-3 text-emerald-500" />
                                            <span className="font-medium text-gray-900 dark:text-white">{conn.to}</span>
                                            <span className={`ml-auto text-xs px-2 py-0.5 rounded-full
                                                ${conn.type === 'homeomorphism' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300' :
                                                    conn.type === 'semantic' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300' :
                                                        'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'}`}>
                                                {conn.type === 'homeomorphism' ? '위상동형' :
                                                    conn.type === 'semantic' ? '의미적' : '외래키'}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}

export default DataSiloBeforeAfter;
