"use client";

import { useState } from "react";

interface TimelineEvent {
    id: string;
    timestamp: string;
    title: string;
    description?: string;
    type: "phase1" | "phase2" | "phase3" | "insight" | "action" | "alert";
    status?: "success" | "warning" | "error" | "info";
    metadata?: Record<string, any>;
}

interface TimelineWidgetProps {
    events: TimelineEvent[];
    title?: string;
    maxHeight?: string;
    onEventClick?: (event: TimelineEvent) => void;
}

export default function TimelineWidget({
    events,
    title = "Activity Timeline",
    maxHeight = "400px",
    onEventClick,
}: TimelineWidgetProps) {
    const [filter, setFilter] = useState<string>("all");

    const getTypeColor = (type: string) => {
        switch (type) {
            case "phase1":
                return "bg-blue-500";
            case "phase2":
                return "bg-purple-500";
            case "phase3":
                return "bg-green-500";
            case "insight":
                return "bg-yellow-500";
            case "action":
                return "bg-orange-500";
            case "alert":
                return "bg-red-500";
            default:
                return "bg-gray-500";
        }
    };

    const getStatusIcon = (status?: string) => {
        switch (status) {
            case "success":
                return "✅";
            case "warning":
                return "⚠️";
            case "error":
                return "❌";
            case "info":
                return "ℹ️";
            default:
                return "📋";
        }
    };

    const getTypeIcon = (type: string) => {
        switch (type) {
            case "phase1":
                return "📊";
            case "phase2":
                return "💡";
            case "phase3":
                return "⚙️";
            case "insight":
                return "🔍";
            case "action":
                return "🎯";
            case "alert":
                return "🚨";
            default:
                return "📝";
        }
    };

    const filterOptions = [
        { value: "all", label: "All" },
        { value: "phase1", label: "Phase 1" },
        { value: "phase2", label: "Phase 2" },
        { value: "phase3", label: "Phase 3" },
        { value: "insight", label: "Insights" },
        { value: "action", label: "Actions" },
        { value: "alert", label: "Alerts" },
    ];

    const filteredEvents =
        filter === "all"
            ? events
            : events.filter((event) => event.type === filter);

    const formatTime = (timestamp: string) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    };

    const formatDate = (timestamp: string) => {
        const date = new Date(timestamp);
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);

        if (date.toDateString() === today.toDateString()) {
            return "Today";
        } else if (date.toDateString() === yesterday.toDateString()) {
            return "Yesterday";
        } else {
            return date.toLocaleDateString();
        }
    };

    // Group events by date
    const groupedEvents = filteredEvents.reduce((acc, event) => {
        const dateKey = formatDate(event.timestamp);
        if (!acc[dateKey]) {
            acc[dateKey] = [];
        }
        acc[dateKey].push(event);
        return acc;
    }, {} as Record<string, TimelineEvent[]>);

    return (
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">{title}</h3>
                <span className="text-sm text-slate-400">
                    {filteredEvents.length} events
                </span>
            </div>

            {/* Filter Tabs */}
            <div className="flex flex-wrap gap-2 mb-4">
                {filterOptions.map((option) => (
                    <button
                        key={option.value}
                        onClick={() => setFilter(option.value)}
                        className={`px-3 py-1 text-xs rounded-full transition-colors ${filter === option.value
                                ? "bg-blue-500 text-white"
                                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                            }`}
                    >
                        {option.label}
                    </button>
                ))}
            </div>

            {/* Timeline */}
            <div
                className="overflow-y-auto space-y-6"
                style={{ maxHeight }}
            >
                {Object.keys(groupedEvents).length === 0 ? (
                    <div className="text-center py-8 text-slate-400">
                        No events to display
                    </div>
                ) : (
                    Object.entries(groupedEvents).map(([date, dateEvents]) => (
                        <div key={date}>
                            {/* Date Header */}
                            <div className="flex items-center gap-2 mb-3">
                                <span className="text-sm font-medium text-slate-400">{date}</span>
                                <div className="flex-1 h-px bg-slate-700"></div>
                            </div>

                            {/* Events */}
                            <div className="relative">
                                {/* Timeline Line */}
                                <div className="absolute left-3 top-0 bottom-0 w-0.5 bg-slate-700"></div>

                                <div className="space-y-3">
                                    {dateEvents.map((event, index) => (
                                        <div
                                            key={event.id}
                                            className={`relative pl-8 cursor-pointer group ${onEventClick ? "hover:bg-slate-700/30 rounded-lg p-2 -ml-2" : ""
                                                }`}
                                            onClick={() => onEventClick?.(event)}
                                        >
                                            {/* Timeline Dot */}
                                            <div
                                                className={`absolute left-1.5 top-1 w-3 h-3 rounded-full ${getTypeColor(
                                                    event.type
                                                )} ring-2 ring-slate-800`}
                                            ></div>

                                            {/* Event Content */}
                                            <div className="flex items-start justify-between">
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-lg">{getTypeIcon(event.type)}</span>
                                                        <span className="font-medium text-white group-hover:text-blue-400 transition-colors">
                                                            {event.title}
                                                        </span>
                                                        {event.status && (
                                                            <span className="text-sm">
                                                                {getStatusIcon(event.status)}
                                                            </span>
                                                        )}
                                                    </div>
                                                    {event.description && (
                                                        <p className="text-sm text-slate-400 mt-1 ml-7">
                                                            {event.description}
                                                        </p>
                                                    )}
                                                </div>
                                                <span className="text-xs text-slate-500">
                                                    {formatTime(event.timestamp)}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
