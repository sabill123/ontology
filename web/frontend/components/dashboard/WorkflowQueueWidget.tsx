"use client";

import { useState, useEffect } from "react";

interface WorkflowAction {
    action_id: string;
    insight_id: string;
    decision_type: "escalate" | "review" | "approve" | "monitor";
    workflow_type: string;
    status: "pending_approval" | "approved" | "rejected" | "in_progress" | "completed";
    owner: string;
    reason: string;
    created_at: string;
    evidence_chain?: {
        phase1?: {
            mappings: string[];
            canonical_objects: string[];
            source_signatures: string[];
        };
        phase2?: {
            confidence: number;
            severity: string;
            explanation: string;
        };
    };
}

interface WorkflowQueueWidgetProps {
    actions: WorkflowAction[];
    onApprove?: (actionId: string) => Promise<void>;
    onReject?: (actionId: string) => Promise<void>;
    onViewDetails?: (action: WorkflowAction) => void;
}

export default function WorkflowQueueWidget({
    actions,
    onApprove,
    onReject,
    onViewDetails,
}: WorkflowQueueWidgetProps) {
    const [expandedId, setExpandedId] = useState<string | null>(null);
    const [processingId, setProcessingId] = useState<string | null>(null);

    const getStatusColor = (status: string) => {
        switch (status) {
            case "approved":
                return "bg-green-500/20 text-green-400 border-green-500/50";
            case "rejected":
                return "bg-red-500/20 text-red-400 border-red-500/50";
            case "pending_approval":
                return "bg-yellow-500/20 text-yellow-400 border-yellow-500/50";
            case "in_progress":
                return "bg-blue-500/20 text-blue-400 border-blue-500/50";
            case "completed":
                return "bg-slate-500/20 text-slate-400 border-slate-500/50";
            default:
                return "bg-gray-500/20 text-gray-400 border-gray-500/50";
        }
    };

    const getDecisionIcon = (type: string) => {
        switch (type) {
            case "escalate":
                return "🚨";
            case "review":
                return "👀";
            case "approve":
                return "✅";
            case "monitor":
                return "📊";
            default:
                return "📋";
        }
    };

    const getWorkflowBadge = (type: string) => {
        const colors: Record<string, string> = {
            preventive_maintenance: "bg-orange-500/20 text-orange-400",
            quality_control: "bg-purple-500/20 text-purple-400",
            performance_tracking: "bg-cyan-500/20 text-cyan-400",
            anomaly_detection: "bg-pink-500/20 text-pink-400",
            general_monitoring: "bg-gray-500/20 text-gray-400",
        };
        return colors[type] || "bg-slate-500/20 text-slate-400";
    };

    const handleApprove = async (actionId: string) => {
        if (!onApprove) return;
        setProcessingId(actionId);
        try {
            await onApprove(actionId);
        } finally {
            setProcessingId(null);
        }
    };

    const handleReject = async (actionId: string) => {
        if (!onReject) return;
        setProcessingId(actionId);
        try {
            await onReject(actionId);
        } finally {
            setProcessingId(null);
        }
    };

    const pendingCount = actions.filter(a => a.status === "pending_approval").length;
    const approvedCount = actions.filter(a => a.status === "approved").length;

    return (
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold text-white">Workflow Queue</h3>
                    <span className="px-2 py-0.5 text-xs font-medium bg-blue-500/20 text-blue-400 rounded-full">
                        {actions.length} total
                    </span>
                </div>
                <div className="flex gap-2 text-sm">
                    <span className="text-yellow-400">⏳ {pendingCount}</span>
                    <span className="text-green-400">✓ {approvedCount}</span>
                </div>
            </div>

            {/* Action List */}
            <div className="space-y-3 max-h-[400px] overflow-y-auto">
                {actions.length === 0 ? (
                    <div className="text-center py-8 text-slate-400">
                        No workflow actions in queue
                    </div>
                ) : (
                    actions.map((action) => (
                        <div
                            key={action.action_id}
                            className="bg-slate-900/50 rounded-lg border border-slate-700/50 overflow-hidden"
                        >
                            {/* Action Header */}
                            <div
                                className="p-4 cursor-pointer hover:bg-slate-700/30 transition-colors"
                                onClick={() =>
                                    setExpandedId(expandedId === action.action_id ? null : action.action_id)
                                }
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <span className="text-xl">{getDecisionIcon(action.decision_type)}</span>
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium text-white">
                                                    {action.action_id}
                                                </span>
                                                <span
                                                    className={`px-2 py-0.5 text-xs rounded-full ${getStatusColor(
                                                        action.status
                                                    )}`}
                                                >
                                                    {action.status.replace("_", " ")}
                                                </span>
                                            </div>
                                            <div className="text-sm text-slate-400 mt-1">
                                                {action.reason}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span
                                            className={`px-2 py-1 text-xs rounded-md ${getWorkflowBadge(
                                                action.workflow_type
                                            )}`}
                                        >
                                            {action.workflow_type.replace("_", " ")}
                                        </span>
                                        <span className="text-slate-500">
                                            {expandedId === action.action_id ? "▲" : "▼"}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Expanded Details */}
                            {expandedId === action.action_id && (
                                <div className="px-4 pb-4 border-t border-slate-700/50">
                                    <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                                        <div>
                                            <span className="text-slate-400">Insight:</span>
                                            <span className="ml-2 text-white">{action.insight_id}</span>
                                        </div>
                                        <div>
                                            <span className="text-slate-400">Owner:</span>
                                            <span className="ml-2 text-white">{action.owner}</span>
                                        </div>
                                        <div>
                                            <span className="text-slate-400">Created:</span>
                                            <span className="ml-2 text-white">
                                                {new Date(action.created_at).toLocaleString()}
                                            </span>
                                        </div>
                                        {action.evidence_chain?.phase2 && (
                                            <div>
                                                <span className="text-slate-400">Confidence:</span>
                                                <span className="ml-2 text-white">
                                                    {(action.evidence_chain.phase2.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        )}
                                    </div>

                                    {/* Evidence */}
                                    {action.evidence_chain?.phase2?.explanation && (
                                        <div className="mt-3 p-3 bg-slate-800/50 rounded-md text-sm text-slate-300">
                                            {action.evidence_chain.phase2.explanation}
                                        </div>
                                    )}

                                    {/* Actions */}
                                    {action.status === "pending_approval" && (
                                        <div className="flex gap-2 mt-4">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleApprove(action.action_id);
                                                }}
                                                disabled={processingId === action.action_id}
                                                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
                                            >
                                                {processingId === action.action_id ? "Processing..." : "Approve"}
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleReject(action.action_id);
                                                }}
                                                disabled={processingId === action.action_id}
                                                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
                                            >
                                                Reject
                                            </button>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
