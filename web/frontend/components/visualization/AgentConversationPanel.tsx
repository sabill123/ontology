"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Bot, User, MessageCircle, CheckCircle, AlertCircle, Clock, ChevronDown, ChevronUp, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export interface AgentMessage {
    id: string;
    agent: string;
    agentRole: string;
    content: string;
    timestamp: string;
    type: 'proposal' | 'response' | 'consensus' | 'validation' | 'decision';
    confidence?: number;
    references?: string[];
    status?: 'pending' | 'agreed' | 'disagreed' | 'escalated';
}

export interface ConsensusRound {
    roundId: string;
    topic: string;
    participants: string[];
    messages: AgentMessage[];
    status: 'in_progress' | 'consensus_reached' | 'escalated';
    finalDecision?: string;
    consensusScore?: number;
}

interface AgentConversationPanelProps {
    rounds: ConsensusRound[];
    currentPhase: number;
    isLive?: boolean;
    onExpandRound?: (roundId: string) => void;
}

// 에이전트 설정
const agentConfig: Record<string, { color: string; bgColor: string; icon: string; label: string }> = {
    tda_expert: { color: 'text-purple-600 dark:text-purple-400', bgColor: 'bg-purple-100 dark:bg-purple-900/50', icon: '🔬', label: 'TDA 전문가' },
    schema_analyst: { color: 'text-blue-600 dark:text-blue-400', bgColor: 'bg-blue-100 dark:bg-blue-900/50', icon: '📊', label: '스키마 분석가' },
    value_matcher: { color: 'text-cyan-600 dark:text-cyan-400', bgColor: 'bg-cyan-100 dark:bg-cyan-900/50', icon: '🔗', label: '값 매처' },
    entity_classifier: { color: 'text-emerald-600 dark:text-emerald-400', bgColor: 'bg-emerald-100 dark:bg-emerald-900/50', icon: '🏷️', label: '엔티티 분류기' },
    relationship_detector: { color: 'text-amber-600 dark:text-amber-400', bgColor: 'bg-amber-100 dark:bg-amber-900/50', icon: '🔀', label: '관계 탐지기' },
    ontology_architect: { color: 'text-indigo-600 dark:text-indigo-400', bgColor: 'bg-indigo-100 dark:bg-indigo-900/50', icon: '🏗️', label: '온톨로지 설계자' },
    conflict_resolver: { color: 'text-rose-600 dark:text-rose-400', bgColor: 'bg-rose-100 dark:bg-rose-900/50', icon: '⚖️', label: '충돌 해결사' },
    quality_judge: { color: 'text-teal-600 dark:text-teal-400', bgColor: 'bg-teal-100 dark:bg-teal-900/50', icon: '✅', label: '품질 심사관' },
    semantic_validator: { color: 'text-violet-600 dark:text-violet-400', bgColor: 'bg-violet-100 dark:bg-violet-900/50', icon: '🧠', label: '의미 검증자' },
    governance_strategist: { color: 'text-orange-600 dark:text-orange-400', bgColor: 'bg-orange-100 dark:bg-orange-900/50', icon: '📋', label: '거버넌스 전략가' },
    action_prioritizer: { color: 'text-pink-600 dark:text-pink-400', bgColor: 'bg-pink-100 dark:bg-pink-900/50', icon: '🎯', label: '액션 우선순위자' },
    risk_assessor: { color: 'text-red-600 dark:text-red-400', bgColor: 'bg-red-100 dark:bg-red-900/50', icon: '⚠️', label: '리스크 평가자' },
    policy_generator: { color: 'text-sky-600 dark:text-sky-400', bgColor: 'bg-sky-100 dark:bg-sky-900/50', icon: '📜', label: '정책 생성자' },
};

// 메시지 타입 아이콘
const messageTypeIcons: Record<string, { icon: React.ElementType; color: string }> = {
    proposal: { icon: MessageCircle, color: 'text-blue-500' },
    response: { icon: MessageCircle, color: 'text-gray-500' },
    consensus: { icon: CheckCircle, color: 'text-emerald-500' },
    validation: { icon: AlertCircle, color: 'text-amber-500' },
    decision: { icon: Sparkles, color: 'text-purple-500' },
};

// 개별 메시지 컴포넌트
function MessageBubble({ message, isLatest }: { message: AgentMessage; isLatest: boolean }) {
    const agent = agentConfig[message.agent] || {
        color: 'text-gray-600 dark:text-gray-400',
        bgColor: 'bg-gray-100 dark:bg-gray-800',
        icon: '🤖',
        label: message.agentRole
    };
    const typeConfig = messageTypeIcons[message.type] || messageTypeIcons.response;
    const TypeIcon = typeConfig.icon;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`relative ${isLatest ? 'animate-pulse-subtle' : ''}`}
        >
            <div className="flex items-start gap-3">
                {/* 에이전트 아바타 */}
                <div className={`flex-shrink-0 w-10 h-10 rounded-full ${agent.bgColor} flex items-center justify-center text-lg`}>
                    {agent.icon}
                </div>

                {/* 메시지 콘텐츠 */}
                <div className="flex-1 min-w-0">
                    {/* 헤더 */}
                    <div className="flex items-center gap-2 mb-1">
                        <span className={`font-semibold ${agent.color}`}>{agent.label}</span>
                        <span className={`text-xs ${typeConfig.color} flex items-center gap-1`}>
                            <TypeIcon className="w-3 h-3" />
                            {message.type === 'proposal' ? '제안' :
                                message.type === 'response' ? '응답' :
                                    message.type === 'consensus' ? '합의' :
                                        message.type === 'validation' ? '검증' : '결정'}
                        </span>
                        <span className="text-xs text-gray-400 dark:text-gray-500">
                            {message.timestamp}
                        </span>
                        {message.confidence && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400">
                                {(message.confidence * 100).toFixed(0)}% 신뢰도
                            </span>
                        )}
                    </div>

                    {/* 메시지 본문 */}
                    <div className={`p-3 rounded-xl rounded-tl-none
                                    ${message.type === 'consensus'
                            ? 'bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800'
                            : message.type === 'decision'
                                ? 'bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800'
                                : 'bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700'}`}>
                        <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                            {message.content}
                        </p>

                        {/* 참조된 항목 */}
                        {message.references && message.references.length > 0 && (
                            <div className="mt-2 pt-2 border-t border-gray-200 dark:border-slate-700">
                                <span className="text-xs text-gray-500 dark:text-gray-400">참조: </span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                    {message.references.map((ref, idx) => (
                                        <span
                                            key={idx}
                                            className="text-xs px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400"
                                        >
                                            {ref}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* 상태 표시 */}
                    {message.status && (
                        <div className="mt-1 flex items-center gap-1">
                            {message.status === 'agreed' && (
                                <span className="text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                                    <CheckCircle className="w-3 h-3" /> 동의함
                                </span>
                            )}
                            {message.status === 'disagreed' && (
                                <span className="text-xs text-red-600 dark:text-red-400 flex items-center gap-1">
                                    <AlertCircle className="w-3 h-3" /> 반대
                                </span>
                            )}
                            {message.status === 'escalated' && (
                                <span className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
                                    <Clock className="w-3 h-3" /> 에스컬레이션
                                </span>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    );
}

// 합의 라운드 카드
function ConsensusRoundCard({ round, isExpanded, onToggle }: { round: ConsensusRound; isExpanded: boolean; onToggle: () => void }) {
    const statusConfig = {
        in_progress: { color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300', label: '진행 중', icon: Clock },
        consensus_reached: { color: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300', label: '합의 완료', icon: CheckCircle },
        escalated: { color: 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300', label: '에스컬레이션', icon: AlertCircle },
    };
    const status = statusConfig[round.status];
    const StatusIcon = status.icon;

    return (
        <div className="border border-gray-200 dark:border-slate-700 rounded-xl overflow-hidden bg-white dark:bg-slate-800/50">
            {/* 헤더 */}
            <button
                onClick={onToggle}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-slate-800 transition-colors"
            >
                <div className="flex items-center gap-3">
                    <div className={`px-2.5 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${status.color}`}>
                        <StatusIcon className="w-3 h-3" />
                        {status.label}
                    </div>
                    <h4 className="font-medium text-gray-900 dark:text-white">{round.topic}</h4>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex -space-x-2">
                        {round.participants.slice(0, 4).map((p, idx) => (
                            <div
                                key={idx}
                                className={`w-6 h-6 rounded-full border-2 border-white dark:border-slate-800
                                           ${agentConfig[p]?.bgColor || 'bg-gray-200'} flex items-center justify-center text-xs`}
                                title={agentConfig[p]?.label || p}
                            >
                                {agentConfig[p]?.icon || '🤖'}
                            </div>
                        ))}
                        {round.participants.length > 4 && (
                            <div className="w-6 h-6 rounded-full border-2 border-white dark:border-slate-800 bg-gray-200 dark:bg-slate-700 flex items-center justify-center text-xs text-gray-600 dark:text-gray-400">
                                +{round.participants.length - 4}
                            </div>
                        )}
                    </div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                        {round.messages.length}개 메시지
                    </span>
                    {isExpanded ? (
                        <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                        <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                </div>
            </button>

            {/* 확장된 콘텐츠 */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="px-4 py-3 border-t border-gray-200 dark:border-slate-700 space-y-4 max-h-96 overflow-y-auto">
                            {round.messages.map((msg, idx) => (
                                <MessageBubble
                                    key={msg.id}
                                    message={msg}
                                    isLatest={idx === round.messages.length - 1}
                                />
                            ))}
                        </div>

                        {/* 합의 결과 */}
                        {round.status === 'consensus_reached' && round.finalDecision && (
                            <div className="px-4 py-3 bg-emerald-50 dark:bg-emerald-900/20 border-t border-emerald-200 dark:border-emerald-800">
                                <div className="flex items-start gap-2">
                                    <Sparkles className="w-5 h-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
                                    <div>
                                        <h5 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-1">최종 합의</h5>
                                        <p className="text-sm text-emerald-700 dark:text-emerald-300">{round.finalDecision}</p>
                                        {round.consensusScore && (
                                            <span className="inline-block mt-2 text-xs px-2 py-1 rounded-full bg-emerald-200 dark:bg-emerald-800 text-emerald-800 dark:text-emerald-200">
                                                합의 점수: {(round.consensusScore * 100).toFixed(0)}%
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export function AgentConversationPanel({ rounds, currentPhase, isLive, onExpandRound }: AgentConversationPanelProps) {
    const [expandedRounds, setExpandedRounds] = useState<Set<string>>(new Set());
    const containerRef = useRef<HTMLDivElement>(null);

    // 새 라운드가 추가되면 자동 스크롤
    useEffect(() => {
        if (isLive && containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [rounds, isLive]);

    const toggleRound = (roundId: string) => {
        setExpandedRounds(prev => {
            const next = new Set(prev);
            if (next.has(roundId)) {
                next.delete(roundId);
            } else {
                next.add(roundId);
            }
            return next;
        });
        onExpandRound?.(roundId);
    };

    const phaseLabels = ['', 'Discovery', 'Refinement', 'Governance'];

    return (
        <div className="h-full flex flex-col bg-gray-50 dark:bg-slate-900 rounded-xl border border-gray-200 dark:border-slate-700 overflow-hidden">
            {/* 헤더 */}
            <div className="px-4 py-3 border-b border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/50">
                        <Bot className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white">에이전트 합의 과정</h3>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Phase {currentPhase}: {phaseLabels[currentPhase]}</p>
                    </div>
                </div>
                {isLive && (
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-100 dark:bg-red-900/50">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                        <span className="text-xs font-medium text-red-600 dark:text-red-400">LIVE</span>
                    </div>
                )}
            </div>

            {/* 합의 라운드 목록 */}
            <div ref={containerRef} className="flex-1 overflow-y-auto p-4 space-y-3">
                {rounds.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400 dark:text-gray-500">
                        <Bot className="w-12 h-12 mb-3 opacity-50" />
                        <p className="text-sm">아직 대화가 없습니다</p>
                        <p className="text-xs mt-1">파이프라인 실행 시 에이전트 대화가 여기에 표시됩니다</p>
                    </div>
                ) : (
                    rounds.map(round => (
                        <ConsensusRoundCard
                            key={round.roundId}
                            round={round}
                            isExpanded={expandedRounds.has(round.roundId)}
                            onToggle={() => toggleRound(round.roundId)}
                        />
                    ))
                )}
            </div>

            {/* 통계 푸터 */}
            <div className="px-4 py-2 border-t border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>총 {rounds.length}개 토론</span>
                <span>{rounds.filter(r => r.status === 'consensus_reached').length}개 합의 완료</span>
            </div>
        </div>
    );
}

export default AgentConversationPanel;
