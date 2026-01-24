"use client";

import React, { useState } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, Info, ChevronRight, Lightbulb, ArrowRight, ExternalLink, Database, BarChart3, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// =====================================================
// 타입 정의
// =====================================================
export interface PredictionData {
    id: string;
    title: string;
    description: string;
    probability: number;
    trend: 'up' | 'down' | 'stable';
    impact: 'high' | 'medium' | 'low';
    category: string;
    relatedEntities: Array<{
        name: string;
        contribution: number;
        type: string;
    }>;
    causationChain: Array<{
        from: string;
        to: string;
        effect: string;
    }>;
    recommendation?: string;
    dataPoints?: number;
    lastUpdated?: string;
}

export interface InsightCardProps {
    insight: PredictionData;
    onExpand?: (id: string) => void;
    onAction?: (id: string, action: string) => void;
    className?: string;
}

// =====================================================
// 서브 컴포넌트들
// =====================================================

// 확률 게이지
function ProbabilityGauge({ probability, size = 80 }: { probability: number; size?: number }) {
    const radius = (size - 8) / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (probability / 100) * circumference;

    const getColor = (prob: number) => {
        if (prob >= 80) return { stroke: '#ef4444', glow: 'rgba(239, 68, 68, 0.3)' };
        if (prob >= 60) return { stroke: '#f59e0b', glow: 'rgba(245, 158, 11, 0.3)' };
        if (prob >= 40) return { stroke: '#3b82f6', glow: 'rgba(59, 130, 246, 0.3)' };
        return { stroke: '#10b981', glow: 'rgba(16, 185, 129, 0.3)' };
    };

    const color = getColor(probability);

    return (
        <div className="relative" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="transform -rotate-90">
                {/* 배경 링 */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="4"
                    className="text-gray-200 dark:text-slate-700"
                />
                {/* 확률 링 */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={color.stroke}
                    strokeWidth="4"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    className="transition-all duration-700"
                    style={{ filter: `drop-shadow(0 0 6px ${color.glow})` }}
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-xl font-bold text-gray-900 dark:text-white">{probability}%</span>
                <span className="text-xs text-gray-500 dark:text-gray-400">확률</span>
            </div>
        </div>
    );
}

// 인과관계 체인 시각화
function CausationChain({ chain }: { chain: PredictionData['causationChain'] }) {
    return (
        <div className="space-y-2">
            {chain.map((link, idx) => (
                <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="flex items-center gap-2 p-2 rounded-lg bg-gray-50 dark:bg-slate-800"
                >
                    <div className="flex-shrink-0 w-2 h-2 rounded-full bg-blue-500" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{link.from}</span>
                    <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{link.to}</span>
                    <span className="ml-auto text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-slate-700 px-2 py-0.5 rounded">
                        {link.effect}
                    </span>
                </motion.div>
            ))}
        </div>
    );
}

// 관련 엔티티 기여도 바
function EntityContribution({ entities }: { entities: PredictionData['relatedEntities'] }) {
    const maxContribution = Math.max(...entities.map(e => e.contribution));

    return (
        <div className="space-y-2">
            {entities.map((entity, idx) => (
                <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2">
                            <Database className="w-3 h-3 text-blue-500" />
                            <span className="font-medium text-gray-700 dark:text-gray-300">{entity.name}</span>
                            <span className="text-gray-400 dark:text-gray-500">({entity.type})</span>
                        </div>
                        <span className="font-semibold text-gray-900 dark:text-white">{entity.contribution}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-100 dark:bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${(entity.contribution / maxContribution) * 100}%` }}
                            transition={{ duration: 0.5, delay: idx * 0.1 }}
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"
                        />
                    </div>
                </div>
            ))}
        </div>
    );
}

// =====================================================
// 메인 컴포넌트
// =====================================================
export function PalantirInsightCard({ insight, onExpand, onAction, className = '' }: InsightCardProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    const impactColors = {
        high: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300 border-red-200 dark:border-red-800',
        medium: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300 border-amber-200 dark:border-amber-800',
        low: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
    };

    const trendIcons = {
        up: <TrendingUp className="w-4 h-4 text-red-500" />,
        down: <TrendingDown className="w-4 h-4 text-emerald-500" />,
        stable: <Activity className="w-4 h-4 text-blue-500" />,
    };

    const handleExpand = () => {
        setIsExpanded(!isExpanded);
        if (onExpand) onExpand(insight.id);
    };

    return (
        <div className={`
            rounded-xl border border-gray-200 dark:border-slate-700
            bg-white dark:bg-slate-800 shadow-sm hover:shadow-lg
            transition-all duration-300 overflow-hidden
            ${className}
        `}>
            {/* 헤더 */}
            <div className="p-4">
                <div className="flex items-start justify-between gap-4">
                    {/* 좌측: 확률 게이지 */}
                    <ProbabilityGauge probability={insight.probability} />

                    {/* 우측: 정보 */}
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                            <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${impactColors[insight.impact]}`}>
                                {insight.impact === 'high' ? '높은 영향' :
                                    insight.impact === 'medium' ? '중간 영향' : '낮은 영향'}
                            </span>
                            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300">
                                {insight.category}
                            </span>
                            {trendIcons[insight.trend]}
                        </div>

                        <h3 className="font-semibold text-gray-900 dark:text-white text-lg leading-tight mb-1">
                            {insight.title}
                        </h3>

                        <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                            {insight.description}
                        </p>

                        {/* 메타 정보 */}
                        <div className="flex items-center gap-3 mt-2 text-xs text-gray-500 dark:text-gray-400">
                            {insight.dataPoints && (
                                <span className="flex items-center gap-1">
                                    <BarChart3 className="w-3 h-3" />
                                    {insight.dataPoints.toLocaleString()} 데이터 포인트
                                </span>
                            )}
                            {insight.lastUpdated && (
                                <span>업데이트: {insight.lastUpdated}</span>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* 확장 버튼 */}
            <button
                onClick={handleExpand}
                className="w-full px-4 py-2 flex items-center justify-center gap-2
                           border-t border-gray-100 dark:border-slate-700
                           text-sm text-gray-500 dark:text-gray-400
                           hover:bg-gray-50 dark:hover:bg-slate-700/50 transition-colors"
            >
                {isExpanded ? '접기' : '상세 보기'}
                <ChevronRight className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
            </button>

            {/* 확장 영역 */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="px-4 pb-4 space-y-4">
                            {/* 인과관계 체인 */}
                            {insight.causationChain.length > 0 && (
                                <div>
                                    <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                                        <ArrowRight className="w-4 h-4 text-blue-500" />
                                        인과관계 체인
                                    </h4>
                                    <CausationChain chain={insight.causationChain} />
                                </div>
                            )}

                            {/* 관련 엔티티 기여도 */}
                            {insight.relatedEntities.length > 0 && (
                                <div>
                                    <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                                        <Database className="w-4 h-4 text-purple-500" />
                                        관련 데이터 소스 기여도
                                    </h4>
                                    <EntityContribution entities={insight.relatedEntities} />
                                </div>
                            )}

                            {/* 권장 사항 */}
                            {insight.recommendation && (
                                <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                                    <h4 className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-1 flex items-center gap-2">
                                        <Lightbulb className="w-4 h-4" />
                                        권장 조치
                                    </h4>
                                    <p className="text-sm text-blue-600 dark:text-blue-400">
                                        {insight.recommendation}
                                    </p>
                                </div>
                            )}

                            {/* 액션 버튼 */}
                            <div className="flex items-center gap-2 pt-2">
                                <button
                                    onClick={() => onAction?.(insight.id, 'investigate')}
                                    className="flex-1 px-3 py-2 rounded-lg text-sm font-medium
                                               bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-gray-300
                                               hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors
                                               flex items-center justify-center gap-2"
                                >
                                    <Info className="w-4 h-4" />
                                    상세 조사
                                </button>
                                <button
                                    onClick={() => onAction?.(insight.id, 'action')}
                                    className="flex-1 px-3 py-2 rounded-lg text-sm font-medium
                                               bg-blue-500 text-white hover:bg-blue-600 transition-colors
                                               flex items-center justify-center gap-2"
                                >
                                    조치 생성
                                    <ExternalLink className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

// =====================================================
// 그리드 컴포넌트
// =====================================================
export function PalantirInsightGrid({ insights, onExpand, onAction }: {
    insights: PredictionData[];
    onExpand?: (id: string) => void;
    onAction?: (id: string, action: string) => void;
}) {
    // 영향도별 정렬 (높음 > 중간 > 낮음)
    const sortedInsights = [...insights].sort((a, b) => {
        const order = { high: 0, medium: 1, low: 2 };
        return order[a.impact] - order[b.impact] || b.probability - a.probability;
    });

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {sortedInsights.map((insight, idx) => (
                <motion.div
                    key={insight.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.1 }}
                >
                    <PalantirInsightCard
                        insight={insight}
                        onExpand={onExpand}
                        onAction={onAction}
                    />
                </motion.div>
            ))}
        </div>
    );
}

export default PalantirInsightCard;
