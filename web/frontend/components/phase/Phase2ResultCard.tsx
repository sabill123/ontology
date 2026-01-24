"use client";

import React from 'react';
import { Brain, Sparkles, TrendingUp, Users, AlertCircle, CheckCircle2, Lightbulb } from 'lucide-react';

interface Phase2ResultProps {
    data: {
        insights: any[];
        validation_scores: {
            topology_fidelity: number;
            semantic_faithfulness: number;
            insight_soundness: number;
            action_alignment: number;
            overall_score: number;
        };
    } | null;
    isLoading?: boolean;
}

// Agent badge colors and Korean labels
const agentConfig: Record<string, { color: string; label: string }> = {
    entity_ontologist: { color: 'bg-purple-500/20 text-purple-300 border-purple-500/30', label: '엔티티 분석' },
    relationship_architect: { color: 'bg-blue-500/20 text-blue-300 border-blue-500/30', label: '관계 분석' },
    property_engineer: { color: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30', label: '속성 분석' },
    domain_expert: { color: 'bg-green-500/20 text-green-300 border-green-500/30', label: '도메인 검증' },
};

// Category Korean labels
const categoryLabels: Record<string, string> = {
    entity_definition: '🏢 엔티티 정의',
    relationship_mapping: '🔗 관계 매핑',
    property_semantics: '📊 속성 분석',
    domain_validation: '✅ 도메인 검증',
    business_rule: '📋 비즈니스 규칙',
    strategic_recommendation: '💡 전략 권장',
    data_quality: '🔍 데이터 품질',
};

// Severity Korean labels
const severityLabels: Record<string, { label: string; color: string }> = {
    high: { label: '🔴 높음', color: 'severity-high' },
    medium: { label: '🟡 보통', color: 'severity-medium' },
    low: { label: '🟢 낮음', color: 'severity-low' },
};

// Score bar component
function ScoreBar({ label, score, color }: { label: string; score: number; color: string }) {
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs">
                <span className="text-zinc-400">{label}</span>
                <span className="font-medium" style={{ color }}>{(score * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${score * 100}%`, background: color }}
                />
            </div>
        </div>
    );
}

// Parse insight explanation to extract structured parts
function parseInsightExplanation(explanation: string) {
    // Try to extract structured parts like [현상], [의미], [영향], [조치]
    const parts: { icon: string; title: string; content: string }[] = [];

    // Check for emoji-based structure
    const patterns = [
        { regex: /📊\s*\[현상\]\s*([^\n��⚠✅]+)/g, icon: '📊', title: '현상' },
        { regex: /💡\s*\[의미\]\s*([^\n📊⚠✅]+)/g, icon: '💡', title: '의미' },
        { regex: /⚠️?\s*\[영향\]\s*([^\n📊💡✅]+)/g, icon: '⚠️', title: '영향' },
        { regex: /✅\s*\[조치\]\s*([^\n📊💡⚠]+)/g, icon: '✅', title: '조치' },
    ];

    for (const pattern of patterns) {
        const match = pattern.regex.exec(explanation);
        if (match) {
            parts.push({ icon: pattern.icon, title: pattern.title, content: match[1].trim() });
        }
    }

    // If no structured parts found, return the whole explanation
    if (parts.length === 0) {
        return null;
    }

    return parts;
}

// Intuitive Insight Card Component
function InsightCard({ insight, index }: { insight: any; index: number }) {
    const severity = severityLabels[insight.severity] || severityLabels.medium;
    const category = categoryLabels[insight.category] || insight.category;
    const agent = agentConfig[insight.agent] || { color: 'bg-zinc-500/20 text-zinc-300 border-zinc-500/30', label: insight.agent };

    const explanationText = insight.explanation || insight.description || "";
    const structuredParts = parseInsightExplanation(explanationText);

    return (
        <div className="insight-card group">
            {/* Header */}
            <div className="flex items-start justify-between gap-2 mb-3">
                <div className="flex items-center gap-2 flex-wrap">
                    <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 text-xs font-mono">
                        {insight.insight_id || insight.id}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium border ${agent.color}`}>
                        {agent.label}
                    </span>
                    <span className="text-xs text-zinc-500">{category}</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${severity.color}`}>
                        {severity.label}
                    </span>
                </div>
            </div>

            {/* Content - Structured or Plain */}
            {structuredParts ? (
                <div className="space-y-2">
                    {structuredParts.map((part, i) => (
                        <div key={i} className="flex items-start gap-2">
                            <span className="flex-shrink-0 text-sm">{part.icon}</span>
                            <div className="flex-1">
                                <span className="text-xs text-zinc-500 font-medium">{part.title}: </span>
                                <span className="text-sm text-zinc-200">{part.content}</span>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <p className="text-sm text-zinc-300 line-clamp-3">{explanationText}</p>
            )}

            {/* Footer */}
            <div className="mt-3 pt-2 border-t border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                    <span className="flex items-center gap-1">
                        <TrendingUp className="w-3 h-3" />
                        신뢰도: {((insight.confidence || 0) * 100).toFixed(0)}%
                    </span>
                    {insight.canonical_objects?.length > 0 && (
                        <span className="text-purple-400">• {insight.canonical_objects.length}개 객체 연결됨</span>
                    )}
                </div>
                {insight.recommendation && (
                    <span className="text-xs text-green-400 flex items-center gap-1">
                        <Lightbulb className="w-3 h-3" />
                        권장사항 있음
                    </span>
                )}
            </div>
        </div>
    );
}

export function Phase2ResultCard({ data, isLoading }: Phase2ResultProps) {
    if (isLoading) {
        return (
            <div className="phase-card phase-2-gradient glow-blue animate-pulse">
                <div className="phase-card-header">
                    <div className="h-6 bg-white/10 rounded w-48" />
                </div>
                <div className="p-6 space-y-4">
                    <div className="h-32 bg-white/5 rounded-xl" />
                </div>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="phase-card phase-2-gradient">
                <div className="phase-card-header flex items-center gap-3">
                    <Brain className="w-5 h-5 text-blue-400" />
                    <h3 className="text-lg font-semibold gradient-text-blue">Phase 2: 온톨로지 인사이트</h3>
                </div>
                <div className="p-6 text-center text-zinc-400">
                    <p>Phase 2를 실행하여 인사이트를 생성하세요</p>
                </div>
            </div>
        );
    }

    const insights = data.insights || [];
    const scores = data.validation_scores || {};

    // Group insights by agent
    const agentCounts: Record<string, number> = {};
    insights.forEach((ins: any) => {
        const agent = ins.agent || 'unknown';
        agentCounts[agent] = (agentCounts[agent] || 0) + 1;
    });

    return (
        <div className="phase-card phase-2-gradient glow-blue">
            {/* Header */}
            <div className="phase-card-header flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-blue-500/20 animate-float">
                        <Brain className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold gradient-text-blue">Phase 2: 온톨로지 인사이트</h3>
                        <p className="text-xs text-zinc-400">멀티 에이전트 분석 결과</p>
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-3xl font-bold text-white">{insights.length}</div>
                    <div className="text-xs text-zinc-400">인사이트</div>
                </div>
            </div>

            {/* Validation Scores */}
            <div className="px-4 py-3 border-b border-white/5">
                <h4 className="text-sm font-medium text-zinc-300 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-blue-400" />
                    검증 점수
                </h4>
                <div className="grid grid-cols-2 gap-3">
                    <ScoreBar label="토폴로지 정합성" score={scores.topology_fidelity || 0} color="#8B5CF6" />
                    <ScoreBar label="의미론적 충실도" score={scores.semantic_faithfulness || 0} color="#3B82F6" />
                    <ScoreBar label="인사이트 건전성" score={scores.insight_soundness || 0} color="#06B6D4" />
                    <ScoreBar label="액션 정렬도" score={scores.action_alignment || 0} color="#10B981" />
                </div>
                <div className="mt-3 p-2 rounded-lg bg-white/5 flex items-center justify-between">
                    <span className="text-sm text-zinc-300">전체 점수</span>
                    <span className="text-xl font-bold gradient-text-blue">{((scores.overall_score || 0) * 100).toFixed(0)}%</span>
                </div>
            </div>

            {/* Agent Distribution */}
            <div className="px-4 py-3 border-b border-white/5">
                <h4 className="text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                    <Users className="w-4 h-4 text-blue-400" />
                    에이전트 기여도
                </h4>
                <div className="flex flex-wrap gap-2">
                    {Object.entries(agentCounts).map(([agent, count]) => {
                        const config = agentConfig[agent] || { color: 'bg-zinc-500/20 text-zinc-300 border-zinc-500/30', label: agent };
                        return (
                            <span
                                key={agent}
                                className={`px-2 py-1 rounded-full text-xs font-medium border ${config.color}`}
                            >
                                {config.label}: {count}개
                            </span>
                        );
                    })}
                </div>
            </div>

            {/* Insights List */}
            <div className="p-4">
                <h4 className="text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    주요 인사이트
                </h4>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                    {insights.slice(0, 5).map((ins: any, i: number) => (
                        <InsightCard key={i} insight={ins} index={i} />
                    ))}
                </div>
            </div>
        </div>
    );
}

export default Phase2ResultCard;
