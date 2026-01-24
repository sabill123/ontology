"use client";

import React from 'react';
import { Target, CheckCircle, AlertTriangle, Eye, Link, ArrowRight, Clock, User } from 'lucide-react';

interface Phase3ResultProps {
    data: {
        decisions: any[];
        actions: any[];
        summary: {
            approved: number;
            escalated: number;
            review: number;
            monitor?: number;
            total_decisions?: number;
            total_actions?: number;
        };
        approval_rate?: number;
        escalation_rate?: number;
        evidence_chain_summary?: {
            phase1_mappings_count: number;
            phase2_insights_count: number;
            actions_with_evidence: number;
        };
    } | null;
    isLoading?: boolean;
}

// 결정 유형 한글 레이블
const decisionTypeLabels: Record<string, { label: string; color: string; icon: any }> = {
    approve: { label: '✅ 자동 승인', color: 'text-green-400', icon: CheckCircle },
    escalate: { label: '🔴 에스컬레이션', color: 'text-red-400', icon: AlertTriangle },
    review: { label: '🟡 검토 필요', color: 'text-amber-400', icon: Eye },
    monitor: { label: '👀 모니터링', color: 'text-blue-400', icon: Eye },
};

// 우선순위 한글 레이블
const priorityLabels: Record<string, { label: string; color: string }> = {
    P1: { label: '긴급', color: 'priority-p1' },
    P2: { label: '높음', color: 'priority-p2' },
    P3: { label: '보통', color: 'priority-p3' },
    P4: { label: '낮음', color: 'bg-zinc-500 text-white' },
};

// 워크플로우 타입 한글 레이블
const workflowTypeLabels: Record<string, string> = {
    schema_fix: '스키마 수정',
    ontology_update: '온톨로지 업데이트',
    data_integration: '데이터 통합',
    relationship_fix: '관계 수정',
    property_standardization: '속성 표준화',
    dashboard_update: '대시보드 업데이트',
    kpi_dashboard: 'KPI 대시보드',
    alert_setup: '알림 설정',
};

// 도넛 차트 컴포넌트
function DonutChart({ data }: { data: { label: string; value: number; color: string }[] }) {
    const total = data.reduce((sum, d) => sum + d.value, 0);
    if (total === 0) {
        return (
            <div className="relative w-28 h-28 flex items-center justify-center">
                <div className="text-zinc-500 text-sm text-center">데이터 없음</div>
            </div>
        );
    }

    let currentAngle = -90; // 상단에서 시작

    return (
        <div className="relative w-28 h-28">
            <svg viewBox="0 0 100 100" className="w-full h-full">
                {data.map((d, i) => {
                    if (d.value === 0) return null;
                    const angle = (d.value / total) * 360;
                    const startAngle = currentAngle;
                    currentAngle += angle;

                    const startRad = (startAngle * Math.PI) / 180;
                    const endRad = ((startAngle + angle) * Math.PI) / 180;

                    const x1 = 50 + 35 * Math.cos(startRad);
                    const y1 = 50 + 35 * Math.sin(startRad);
                    const x2 = 50 + 35 * Math.cos(endRad);
                    const y2 = 50 + 35 * Math.sin(endRad);

                    const largeArc = angle > 180 ? 1 : 0;

                    return (
                        <path
                            key={i}
                            d={`M 50 50 L ${x1} ${y1} A 35 35 0 ${largeArc} 1 ${x2} ${y2} Z`}
                            fill={d.color}
                            className="transition-all duration-500"
                            style={{ filter: `drop-shadow(0 0 4px ${d.color})` }}
                        />
                    );
                })}
                {/* 중앙 구멍 */}
                <circle cx="50" cy="50" r="22" fill="#0f172a" />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                    <div className="text-lg font-bold text-white">{total}</div>
                    <div className="text-[10px] text-zinc-400">총 결정</div>
                </div>
            </div>
        </div>
    );
}

export function Phase3ResultCard({ data, isLoading }: Phase3ResultProps) {
    if (isLoading) {
        return (
            <div className="phase-card phase-3-gradient glow-green animate-pulse">
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
            <div className="phase-card phase-3-gradient">
                <div className="phase-card-header flex items-center gap-3">
                    <Target className="w-5 h-5 text-green-400" />
                    <h3 className="text-lg font-semibold gradient-text-green">Phase 3: 거버넌스 액션</h3>
                </div>
                <div className="p-6 text-center text-zinc-400">
                    <p>Phase 3를 실행하여 거버넌스 결정을 생성하세요</p>
                </div>
            </div>
        );
    }

    const summary = data.summary || {};
    const actions = data.actions || [];
    const decisions = data.decisions || [];
    const evidenceChain: { phase1_mappings_count?: number; phase2_insights_count?: number; actions_with_evidence?: number } = data.evidence_chain_summary || {};

    const chartData = [
        { label: '승인', value: summary.approved || 0, color: '#10B981' },
        { label: '에스컬레이션', value: summary.escalated || 0, color: '#EF4444' },
        { label: '검토', value: summary.review || 0, color: '#F59E0B' },
    ];

    // 승인율/에스컬레이션률 계산
    const total = (summary.approved || 0) + (summary.escalated || 0) + (summary.review || 0);
    const approvalRate = total > 0 ? ((summary.approved || 0) / total * 100).toFixed(0) : 0;
    const escalationRate = total > 0 ? ((summary.escalated || 0) / total * 100).toFixed(0) : 0;

    return (
        <div className="phase-card phase-3-gradient glow-green">
            {/* 헤더 */}
            <div className="phase-card-header flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-green-500/20 animate-float">
                        <Target className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold gradient-text-green">Phase 3: 거버넌스 액션</h3>
                        <p className="text-xs text-zinc-400">결정 및 실행 엔진</p>
                    </div>
                </div>
                <div className="text-right text-xs text-zinc-400">
                    <div>승인율: <span className="text-green-400 font-bold">{approvalRate}%</span></div>
                    <div>에스컬레이션: <span className="text-red-400 font-bold">{escalationRate}%</span></div>
                </div>
            </div>

            {/* 분포 차트 & 통계 */}
            <div className="p-4 flex items-center gap-6 border-b border-white/5">
                <DonutChart data={chartData} />
                <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span className="text-sm text-zinc-300">자동 승인</span>
                        <span className="ml-auto font-bold text-green-400">{summary.approved || 0}건</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-sm text-zinc-300">에스컬레이션</span>
                        <span className="ml-auto font-bold text-red-400">{summary.escalated || 0}건</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Eye className="w-4 h-4 text-amber-400" />
                        <span className="text-sm text-zinc-300">검토 필요</span>
                        <span className="ml-auto font-bold text-amber-400">{summary.review || 0}건</span>
                    </div>
                </div>
            </div>

            {/* 증거 체인 요약 */}
            <div className="px-4 py-3 border-b border-white/5">
                <h4 className="text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                    <Link className="w-4 h-4 text-green-400" />
                    증거 체인 (데이터 추적성)
                </h4>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-white/5 text-xs">
                    <div className="flex items-center gap-1">
                        <span className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">Phase 1</span>
                        <span className="text-zinc-400">{evidenceChain.phase1_mappings_count || 0}개 매핑</span>
                    </div>
                    <ArrowRight className="w-4 h-4 text-zinc-500" />
                    <div className="flex items-center gap-1">
                        <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300">Phase 2</span>
                        <span className="text-zinc-400">{evidenceChain.phase2_insights_count || 0}개 인사이트</span>
                    </div>
                    <ArrowRight className="w-4 h-4 text-zinc-500" />
                    <div className="flex items-center gap-1">
                        <span className="px-2 py-0.5 rounded bg-green-500/20 text-green-300">Phase 3</span>
                        <span className="text-zinc-400">{evidenceChain.actions_with_evidence || actions.length}개 연결됨</span>
                    </div>
                </div>
            </div>

            {/* 액션 리스트 */}
            <div className="p-4">
                <h4 className="text-sm font-medium text-zinc-300 mb-2">최근 액션</h4>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                    {actions.length === 0 && decisions.length === 0 ? (
                        <p className="text-xs text-zinc-500 text-center py-2">액션이 없습니다</p>
                    ) : (
                        (actions.length > 0 ? actions : decisions).slice(0, 4).map((act: any, i: number) => {
                            const decisionType = act.decision_type || act.status || 'review';
                            const priority = act.priority || 'P3';
                            const priorityConfig = priorityLabels[priority] || priorityLabels.P3;
                            const workflowType = act.workflow_type || act.category || 'general';

                            const statusClass =
                                decisionType === 'approve' || decisionType === 'approved' ? 'approved' :
                                    decisionType === 'escalate' || decisionType === 'escalated' ? 'escalated' : 'review';

                            return (
                                <div key={i} className={`action-card ${statusClass}`}>
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                                                <span className="px-2 py-0.5 rounded bg-green-500/20 text-green-300 text-xs font-mono">
                                                    {act.action_id || act.decision_id || `ACT-${i + 1}`}
                                                </span>
                                                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${priorityConfig.color}`}>
                                                    {priorityConfig.label}
                                                </span>
                                                <span className="text-[10px] text-zinc-500">
                                                    {workflowTypeLabels[workflowType] || workflowType.replace(/_/g, ' ')}
                                                </span>
                                            </div>
                                            <p className="text-xs text-zinc-400 line-clamp-1">
                                                {act.recommendation || act.reason || '설명 없음'}
                                            </p>
                                        </div>
                                        <div className="flex flex-col items-end gap-1">
                                            <span className={`text-xs ${decisionTypeLabels[decisionType]?.color || 'text-zinc-400'}`}>
                                                {decisionTypeLabels[decisionType]?.label || decisionType}
                                            </span>
                                            {act.owner && (
                                                <span className="text-[10px] text-zinc-500 flex items-center gap-1">
                                                    <User className="w-3 h-3" />
                                                    {act.owner}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    {act.evidence_chain && (
                                        <div className="mt-2 flex items-center gap-1 text-[10px] text-purple-400">
                                            <Link className="w-3 h-3" />
                                            {act.evidence_chain.phase2_insight || act.insight_id} 기반
                                            {act.evidence_chain.phase1_mappings?.length > 0 &&
                                                ` → ${act.evidence_chain.phase1_mappings.length}개 매핑 참조`
                                            }
                                        </div>
                                    )}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>
        </div>
    );
}

export default Phase3ResultCard;
