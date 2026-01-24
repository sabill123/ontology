"use client";

import { useState, useEffect } from "react";

interface Alert {
    id: string;
    title: string;
    message: string;
    severity: "critical" | "high" | "medium" | "low";
    source: string;
    timestamp: string;
    acknowledged?: boolean;
    metadata?: Record<string, any>;
}

interface AlertListWidgetProps {
    alerts: Alert[];
    onAcknowledge?: (alertId: string) => Promise<void>;
    onDismiss?: (alertId: string) => void;
    onViewDetails?: (alert: Alert) => void;
    maxVisible?: number;
}

export default function AlertListWidget({
    alerts,
    onAcknowledge,
    onDismiss,
    onViewDetails,
    maxVisible = 5,
}: AlertListWidgetProps) {
    const [showAll, setShowAll] = useState(false);
    const [processingId, setProcessingId] = useState<string | null>(null);

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case "critical":
                return {
                    bg: "bg-red-500/20",
                    border: "border-red-500/50",
                    text: "text-red-400",
                    icon: "🔴",
                    pulse: true,
                };
            case "high":
                return {
                    bg: "bg-orange-500/20",
                    border: "border-orange-500/50",
                    text: "text-orange-400",
                    icon: "🟠",
                    pulse: false,
                };
            case "medium":
                return {
                    bg: "bg-yellow-500/20",
                    border: "border-yellow-500/50",
                    text: "text-yellow-400",
                    icon: "🟡",
                    pulse: false,
                };
            case "low":
                return {
                    bg: "bg-blue-500/20",
                    border: "border-blue-500/50",
                    text: "text-blue-400",
                    icon: "🔵",
                    pulse: false,
                };
            default:
                return {
                    bg: "bg-gray-500/20",
                    border: "border-gray-500/50",
                    text: "text-gray-400",
                    icon: "⚪",
                    pulse: false,
                };
        }
    };

    const handleAcknowledge = async (alertId: string) => {
        if (!onAcknowledge) return;
        setProcessingId(alertId);
        try {
            await onAcknowledge(alertId);
        } finally {
            setProcessingId(null);
        }
    };

    const formatTimeAgo = (timestamp: string) => {
        const now = new Date();
        const then = new Date(timestamp);
        const diffMs = now.getTime() - then.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return "Just now";
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        return `${diffDays}d ago`;
    };

    // Count by severity
    const criticalCount = alerts.filter(a => a.severity === "critical" && !a.acknowledged).length;
    const highCount = alerts.filter(a => a.severity === "high" && !a.acknowledged).length;
    const unacknowledgedCount = alerts.filter(a => !a.acknowledged).length;

    // Sort and limit
    const sortedAlerts = [...alerts].sort((a, b) => {
        const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
        if (a.acknowledged !== b.acknowledged) return a.acknowledged ? 1 : -1;
        return severityOrder[a.severity] - severityOrder[b.severity];
    });

    const visibleAlerts = showAll ? sortedAlerts : sortedAlerts.slice(0, maxVisible);

    return (
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold text-white">Alerts</h3>
                    {criticalCount > 0 && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-red-500 text-white rounded-full animate-pulse">
                            {criticalCount} Critical
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-4 text-sm">
                    <span className="text-orange-400">{highCount} High</span>
                    <span className="text-slate-400">{unacknowledgedCount} Pending</span>
                </div>
            </div>

            {/* Alert List */}
            <div className="space-y-2">
                {visibleAlerts.length === 0 ? (
                    <div className="text-center py-8 text-slate-400">
                        <span className="text-3xl">✅</span>
                        <p className="mt-2">No alerts at this time</p>
                    </div>
                ) : (
                    visibleAlerts.map((alert) => {
                        const style = getSeverityColor(alert.severity);
                        return (
                            <div
                                key={alert.id}
                                className={`p-4 rounded-lg border ${style.bg} ${style.border} ${alert.acknowledged ? "opacity-60" : ""
                                    } ${style.pulse ? "animate-pulse" : ""}`}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex items-start gap-3 flex-1">
                                        <span className="text-lg">{style.icon}</span>
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2">
                                                <span className={`font-medium ${style.text}`}>
                                                    {alert.title}
                                                </span>
                                                {alert.acknowledged && (
                                                    <span className="text-xs text-slate-500">✓ Acknowledged</span>
                                                )}
                                            </div>
                                            <p className="text-sm text-slate-300 mt-1">{alert.message}</p>
                                            <div className="flex items-center gap-4 mt-2 text-xs text-slate-400">
                                                <span>Source: {alert.source}</span>
                                                <span>{formatTimeAgo(alert.timestamp)}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex gap-1 ml-2">
                                        {!alert.acknowledged && onAcknowledge && (
                                            <button
                                                onClick={() => handleAcknowledge(alert.id)}
                                                disabled={processingId === alert.id}
                                                className="p-1.5 bg-slate-600 hover:bg-slate-500 rounded-md transition-colors disabled:opacity-50"
                                                title="Acknowledge"
                                            >
                                                ✓
                                            </button>
                                        )}
                                        {onViewDetails && (
                                            <button
                                                onClick={() => onViewDetails(alert)}
                                                className="p-1.5 bg-slate-600 hover:bg-slate-500 rounded-md transition-colors"
                                                title="View Details"
                                            >
                                                🔍
                                            </button>
                                        )}
                                        {onDismiss && alert.acknowledged && (
                                            <button
                                                onClick={() => onDismiss(alert.id)}
                                                className="p-1.5 bg-slate-600 hover:bg-slate-500 rounded-md transition-colors"
                                                title="Dismiss"
                                            >
                                                ×
                                            </button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* Show More */}
            {alerts.length > maxVisible && (
                <button
                    onClick={() => setShowAll(!showAll)}
                    className="w-full mt-4 py-2 text-sm text-slate-400 hover:text-white transition-colors"
                >
                    {showAll ? "Show Less" : `Show ${alerts.length - maxVisible} More`}
                </button>
            )}
        </div>
    );
}
