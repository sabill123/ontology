"use client";

import { useState, useEffect } from "react";

interface KPIData {
    label: string;
    value: string | number;
    change?: number;
    changeLabel?: string;
    icon?: string;
    color?: "green" | "red" | "blue" | "yellow" | "purple";
}

interface KPICardWidgetProps {
    title: string;
    kpis: KPIData[];
    refreshInterval?: number;
    onRefresh?: () => Promise<KPIData[]>;
}

export default function KPICardWidget({
    title,
    kpis: initialKpis,
    refreshInterval = 30000,
    onRefresh,
}: KPICardWidgetProps) {
    const [kpis, setKpis] = useState<KPIData[]>(initialKpis);
    const [isLoading, setIsLoading] = useState(false);
    const [lastUpdated, setLastUpdated] = useState(new Date());

    useEffect(() => {
        if (onRefresh && refreshInterval > 0) {
            const interval = setInterval(async () => {
                setIsLoading(true);
                try {
                    const newKpis = await onRefresh();
                    setKpis(newKpis);
                    setLastUpdated(new Date());
                } catch (error) {
                    console.error("Failed to refresh KPIs:", error);
                } finally {
                    setIsLoading(false);
                }
            }, refreshInterval);

            return () => clearInterval(interval);
        }
    }, [onRefresh, refreshInterval]);

    const getColorClasses = (color?: string) => {
        switch (color) {
            case "green":
                return "bg-green-500/20 text-green-400 border-green-500/30";
            case "red":
                return "bg-red-500/20 text-red-400 border-red-500/30";
            case "blue":
                return "bg-blue-500/20 text-blue-400 border-blue-500/30";
            case "yellow":
                return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
            case "purple":
                return "bg-purple-500/20 text-purple-400 border-purple-500/30";
            default:
                return "bg-slate-500/20 text-slate-400 border-slate-500/30";
        }
    };

    const getChangeColor = (change?: number) => {
        if (!change) return "text-gray-400";
        return change > 0 ? "text-green-400" : "text-red-400";
    };

    const getChangeIcon = (change?: number) => {
        if (!change) return "";
        return change > 0 ? "↑" : "↓";
    };

    return (
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">{title}</h3>
                <div className="flex items-center gap-2">
                    {isLoading && (
                        <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                    )}
                    <span className="text-xs text-slate-400">
                        {lastUpdated.toLocaleTimeString()}
                    </span>
                </div>
            </div>

            {/* KPI Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {kpis.map((kpi, index) => (
                    <div
                        key={index}
                        className={`p-4 rounded-lg border ${getColorClasses(kpi.color)}`}
                    >
                        <div className="flex items-center gap-2 mb-2">
                            {kpi.icon && <span className="text-lg">{kpi.icon}</span>}
                            <span className="text-sm font-medium opacity-80">{kpi.label}</span>
                        </div>
                        <div className="text-2xl font-bold">{kpi.value}</div>
                        {kpi.change !== undefined && (
                            <div className={`text-sm mt-1 ${getChangeColor(kpi.change)}`}>
                                {getChangeIcon(kpi.change)} {Math.abs(kpi.change)}%{" "}
                                {kpi.changeLabel || ""}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
