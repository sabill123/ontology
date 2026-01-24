"use client";

import React from 'react';
import { Database, GitBranch, Layers, Link2, TrendingUp } from 'lucide-react';

interface Phase1ResultProps {
    data: {
        mappings: any[];
        canonical_objects: any[];
        topology_score: number;
        data_sources: any[];
        cross_silo_links?: any[];
    } | null;
    isLoading?: boolean;
}

// 원형 점수 게이지 컴포넌트
function ScoreGauge({ score, size = 80, color = "#8B5CF6" }: { score: number; size?: number; color?: string }) {
    const radius = (size - 8) / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score * circumference);

    return (
        <div className="relative" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="transform -rotate-90">
                {/* 배경 링 */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="rgba(255,255,255,0.1)"
                    strokeWidth="4"
                />
                {/* 점수 링 */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth="4"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    className="score-gauge-ring"
                    style={{ filter: `drop-shadow(0 0 6px ${color})` }}
                />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold text-white">{(score * 100).toFixed(0)}%</span>
            </div>
        </div>
    );
}

// 매핑 타입 한글 레이블
const mappingTypeLabels: Record<string, string> = {
    'foreign_key': '외래키 관계',
    'semantic': '의미론적 연결',
    'cross_silo': '크로스 사일로',
    'structural': '구조적 유사성',
};

export function Phase1ResultCard({ data, isLoading }: Phase1ResultProps) {
    if (isLoading) {
        return (
            <div className="phase-card phase-1-gradient glow-purple animate-pulse">
                <div className="phase-card-header">
                    <div className="h-6 bg-white/10 rounded w-48" />
                </div>
                <div className="p-6 space-y-4">
                    <div className="h-24 bg-white/5 rounded-xl" />
                </div>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="phase-card phase-1-gradient">
                <div className="phase-card-header flex items-center gap-3">
                    <Database className="w-5 h-5 text-purple-400" />
                    <h3 className="text-lg font-semibold gradient-text-purple">Phase 1: 데이터 토폴로지</h3>
                </div>
                <div className="p-6 text-center text-zinc-400">
                    <p>데이터를 업로드하고 Phase 1을 실행하세요</p>
                </div>
            </div>
        );
    }

    const mappings = data.mappings || [];
    const canonicalObjects = data.canonical_objects || [];
    const score = data.topology_score || 0;
    const crossSiloLinks = data.cross_silo_links || [];

    return (
        <div className="phase-card phase-1-gradient glow-purple">
            {/* 헤더 */}
            <div className="phase-card-header flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-purple-500/20 animate-float">
                        <Database className="w-5 h-5 text-purple-400" />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold gradient-text-purple">Phase 1: 데이터 토폴로지</h3>
                        <p className="text-xs text-zinc-400">위상동형 스키마 분석</p>
                    </div>
                </div>
                <ScoreGauge score={score} color="#8B5CF6" />
            </div>

            {/* 지표 그리드 */}
            <div className="p-4 grid grid-cols-3 gap-3">
                <div className="metric-card text-center">
                    <GitBranch className="w-5 h-5 text-purple-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-white">{mappings.length}</div>
                    <div className="text-xs text-zinc-400">스키마 매핑</div>
                </div>
                <div className="metric-card text-center">
                    <Layers className="w-5 h-5 text-blue-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-white">{canonicalObjects.length}</div>
                    <div className="text-xs text-zinc-400">정규 객체</div>
                </div>
                <div className="metric-card text-center">
                    <Link2 className="w-5 h-5 text-cyan-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-white">{crossSiloLinks.length || mappings.length}</div>
                    <div className="text-xs text-zinc-400">크로스 사일로 연결</div>
                </div>
            </div>

            {/* 매핑 리스트 */}
            <div className="px-4 pb-4">
                <h4 className="text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-purple-400" />
                    위상동형 매핑
                </h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                    {mappings.slice(0, 4).map((m: any, i: number) => {
                        const mappingId = m.mapping_id || `MAP-${String(i + 1).padStart(3, '0')}`;
                        const sourceA = m.source_a || m.table_a || '테이블A';
                        const sourceB = m.source_b || m.table_b || '테이블B';
                        const confidence = m.confidence || m.match_score || 0;
                        const mappingType = m.mapping_type || 'cross_silo';

                        return (
                            <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-white/5 text-sm">
                                <span className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 text-xs font-mono">
                                    {mappingId}
                                </span>
                                <span className="text-zinc-300 truncate flex-1">
                                    {sourceA.split('.').pop()} → {sourceB.split('.').pop()}
                                </span>
                                <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-300">
                                    {mappingTypeLabels[mappingType] || mappingType}
                                </span>
                                <span className="text-xs text-zinc-500">
                                    {confidence ? `${(confidence * 100).toFixed(0)}%` : ''}
                                </span>
                            </div>
                        );
                    })}
                    {mappings.length === 0 && (
                        <p className="text-xs text-zinc-500 text-center py-2">매핑 발견 안됨</p>
                    )}
                </div>
            </div>

            {/* 정규 객체 요약 */}
            {canonicalObjects.length > 0 && (
                <div className="px-4 pb-4">
                    <h4 className="text-sm font-medium text-zinc-300 mb-2">감지된 정규 객체</h4>
                    <div className="flex flex-wrap gap-2">
                        {canonicalObjects.slice(0, 5).map((obj: any, i: number) => (
                            <span
                                key={i}
                                className="px-2 py-1 rounded-full text-xs font-medium bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                            >
                                {obj.name || obj.object_id || `객체-${i + 1}`}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Phase1ResultCard;
