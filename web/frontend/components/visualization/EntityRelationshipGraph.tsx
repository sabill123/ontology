"use client";

import React, { useCallback, useMemo, useState } from 'react';
import {
    ReactFlow,
    Node,
    Edge,
    Controls,
    Background,
    MiniMap,
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    MarkerType,
    Panel,
    NodeProps,
    Handle,
    Position,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Database, Layers, Link2, Table2, Key, Hash, Calendar, Type, ArrowRight } from 'lucide-react';

// =====================================================
// 타입 정의
// =====================================================
export interface EntityData {
    id: string;
    name: string;
    type: 'table' | 'entity' | 'concept' | 'attribute';
    source?: string;
    attributes?: Array<{
        name: string;
        type: string;
        isPrimary?: boolean;
        isForeign?: boolean;
    }>;
    metadata?: Record<string, any>;
}

export interface RelationshipData {
    id: string;
    from: string;
    to: string;
    type: 'foreign_key' | 'semantic' | 'homeomorphism' | 'inheritance' | 'composition';
    label?: string;
    confidence?: number;
}

interface EntityRelationshipGraphProps {
    entities: EntityData[];
    relationships: RelationshipData[];
    onNodeClick?: (entity: EntityData) => void;
    onEdgeClick?: (relationship: RelationshipData) => void;
    className?: string;
}

// =====================================================
// 커스텀 노드 컴포넌트들
// =====================================================

// 테이블 노드
function TableNode({ data, selected }: NodeProps) {
    const nodeData = data as unknown as EntityData & { label: string };
    const attributes = nodeData.attributes || [];

    return (
        <div className={`
            min-w-[200px] rounded-xl overflow-hidden shadow-lg
            border-2 transition-all duration-200
            ${selected
                ? 'border-blue-500 shadow-blue-200 dark:shadow-blue-900/50'
                : 'border-gray-200 dark:border-slate-600'}
            bg-white dark:bg-slate-800
        `}>
            {/* 헤더 */}
            <div className="px-3 py-2 bg-gradient-to-r from-blue-500 to-blue-600 flex items-center gap-2">
                <Table2 className="w-4 h-4 text-white" />
                <span className="font-semibold text-white text-sm">{nodeData.label}</span>
            </div>

            {/* 속성 목록 */}
            {attributes.length > 0 && (
                <div className="p-2 space-y-1">
                    {attributes.slice(0, 5).map((attr, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-xs px-2 py-1 rounded bg-gray-50 dark:bg-slate-700">
                            {attr.isPrimary ? (
                                <Key className="w-3 h-3 text-amber-500" />
                            ) : attr.isForeign ? (
                                <Link2 className="w-3 h-3 text-blue-500" />
                            ) : attr.type === 'string' ? (
                                <Type className="w-3 h-3 text-gray-400" />
                            ) : attr.type === 'number' || attr.type === 'integer' ? (
                                <Hash className="w-3 h-3 text-gray-400" />
                            ) : attr.type === 'date' || attr.type === 'datetime' ? (
                                <Calendar className="w-3 h-3 text-gray-400" />
                            ) : (
                                <div className="w-3 h-3" />
                            )}
                            <span className="text-gray-700 dark:text-gray-300 flex-1 truncate">{attr.name}</span>
                            <span className="text-gray-400 dark:text-gray-500">{attr.type}</span>
                        </div>
                    ))}
                    {attributes.length > 5 && (
                        <div className="text-xs text-gray-400 text-center py-1">
                            +{attributes.length - 5} more
                        </div>
                    )}
                </div>
            )}

            {/* 소스 표시 */}
            {nodeData.source && (
                <div className="px-3 py-1.5 border-t border-gray-100 dark:border-slate-700 bg-gray-50 dark:bg-slate-900">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                        Source: <span className="font-medium">{nodeData.source}</span>
                    </span>
                </div>
            )}

            {/* Handles */}
            <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-blue-500 !border-2 !border-white" />
            <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-blue-500 !border-2 !border-white" />
        </div>
    );
}

// 엔티티(개념) 노드
function EntityNode({ data, selected }: NodeProps) {
    const nodeData = data as unknown as EntityData & { label: string };

    return (
        <div className={`
            px-4 py-3 rounded-xl shadow-lg
            border-2 transition-all duration-200
            ${selected
                ? 'border-emerald-500 shadow-emerald-200 dark:shadow-emerald-900/50'
                : 'border-emerald-200 dark:border-emerald-800'}
            bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-900/30 dark:to-emerald-800/30
        `}>
            <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-emerald-500">
                    <Layers className="w-4 h-4 text-white" />
                </div>
                <div>
                    <span className="font-semibold text-emerald-800 dark:text-emerald-200 text-sm">{nodeData.label}</span>
                    <p className="text-xs text-emerald-600 dark:text-emerald-400">Canonical Entity</p>
                </div>
            </div>

            <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-emerald-500 !border-2 !border-white" />
            <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-emerald-500 !border-2 !border-white" />
        </div>
    );
}

// 컨셉 노드
function ConceptNode({ data, selected }: NodeProps) {
    const nodeData = data as unknown as EntityData & { label: string };

    return (
        <div className={`
            px-4 py-3 rounded-full shadow-lg
            border-2 transition-all duration-200
            ${selected
                ? 'border-purple-500 shadow-purple-200 dark:shadow-purple-900/50'
                : 'border-purple-200 dark:border-purple-800'}
            bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30
        `}>
            <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <span className="font-semibold text-purple-800 dark:text-purple-200 text-sm">{nodeData.label}</span>
            </div>

            <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-purple-500 !border-2 !border-white" />
            <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-purple-500 !border-2 !border-white" />
        </div>
    );
}

// =====================================================
// 노드 타입 등록
// =====================================================
const nodeTypes = {
    table: TableNode,
    entity: EntityNode,
    concept: ConceptNode,
    attribute: ConceptNode, // attribute도 concept 스타일 사용
};

// =====================================================
// 엣지 스타일
// =====================================================
const edgeStyles: Record<string, { stroke: string; strokeWidth: number; animated: boolean }> = {
    foreign_key: { stroke: '#3b82f6', strokeWidth: 2, animated: false },
    semantic: { stroke: '#8b5cf6', strokeWidth: 2, animated: true },
    homeomorphism: { stroke: '#10b981', strokeWidth: 3, animated: true },
    inheritance: { stroke: '#f59e0b', strokeWidth: 2, animated: false },
    composition: { stroke: '#ef4444', strokeWidth: 2, animated: false },
};

// =====================================================
// 메인 컴포넌트
// =====================================================
export function EntityRelationshipGraph({
    entities,
    relationships,
    onNodeClick,
    onEdgeClick,
    className = '',
}: EntityRelationshipGraphProps) {
    const [selectedNode, setSelectedNode] = useState<string | null>(null);

    // 노드 생성
    const initialNodes: Node[] = useMemo(() => {
        // 그리드 레이아웃 계산
        const cols = Math.ceil(Math.sqrt(entities.length));
        const spacing = { x: 300, y: 200 };

        return entities.map((entity, index) => ({
            id: entity.id,
            type: entity.type,
            position: {
                x: (index % cols) * spacing.x + Math.random() * 50,
                y: Math.floor(index / cols) * spacing.y + Math.random() * 50,
            },
            data: {
                ...entity,
                label: entity.name,
            },
        }));
    }, [entities]);

    // 엣지 생성
    const initialEdges = useMemo(() => {
        return relationships.map((rel) => {
            const style = edgeStyles[rel.type] || edgeStyles.semantic;
            return {
                id: rel.id,
                source: rel.from,
                target: rel.to,
                type: 'smoothstep',
                animated: style.animated,
                style: {
                    stroke: style.stroke,
                    strokeWidth: style.strokeWidth,
                },
                markerEnd: {
                    type: MarkerType.ArrowClosed,
                    color: style.stroke,
                },
                label: rel.label || (rel.confidence ? `${(rel.confidence * 100).toFixed(0)}%` : ''),
                labelStyle: {
                    fill: '#64748b',
                    fontSize: 11,
                    fontWeight: 500,
                },
                labelBgStyle: {
                    fill: '#fff',
                    stroke: '#e2e8f0',
                },
            } as Edge;
        });
    }, [relationships]);

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        [setEdges]
    );

    const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
        setSelectedNode(node.id);
        const entity = entities.find(e => e.id === node.id);
        if (entity && onNodeClick) {
            onNodeClick(entity);
        }
    }, [entities, onNodeClick]);

    const handleEdgeClick = useCallback((_: React.MouseEvent, edge: Edge) => {
        const relationship = relationships.find(r => r.id === edge.id);
        if (relationship && onEdgeClick) {
            onEdgeClick(relationship);
        }
    }, [relationships, onEdgeClick]);

    return (
        <div className={`w-full h-[600px] rounded-xl overflow-hidden border border-gray-200 dark:border-slate-700 ${className}`}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={handleNodeClick}
                onEdgeClick={handleEdgeClick}
                nodeTypes={nodeTypes}
                fitView
                className="bg-gray-50 dark:bg-slate-900"
            >
                <Controls
                    className="!bg-white dark:!bg-slate-800 !border-gray-200 dark:!border-slate-700 !shadow-lg"
                />
                <MiniMap
                    className="!bg-white dark:!bg-slate-800 !border-gray-200 dark:!border-slate-700"
                    nodeColor={(node) => {
                        switch (node.type) {
                            case 'table': return '#3b82f6';
                            case 'entity': return '#10b981';
                            case 'concept': return '#8b5cf6';
                            default: return '#94a3b8';
                        }
                    }}
                />
                <Background color="#94a3b8" gap={20} size={1} />

                {/* 범례 패널 */}
                <Panel position="top-left" className="!bg-white dark:!bg-slate-800 rounded-lg shadow-lg p-3 border border-gray-200 dark:border-slate-700">
                    <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">범례</h4>
                    <div className="space-y-1.5">
                        {/* 노드 타입 */}
                        <div className="flex items-center gap-2 text-xs">
                            <div className="w-3 h-3 rounded bg-blue-500" />
                            <span className="text-gray-600 dark:text-gray-400">테이블</span>
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                            <div className="w-3 h-3 rounded bg-emerald-500" />
                            <span className="text-gray-600 dark:text-gray-400">엔티티</span>
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                            <div className="w-3 h-3 rounded-full bg-purple-500" />
                            <span className="text-gray-600 dark:text-gray-400">컨셉</span>
                        </div>
                        {/* 엣지 타입 */}
                        <div className="pt-1.5 border-t border-gray-200 dark:border-slate-700 mt-1.5">
                            <div className="flex items-center gap-2 text-xs">
                                <div className="w-4 h-0.5 bg-blue-500" />
                                <span className="text-gray-600 dark:text-gray-400">외래키</span>
                            </div>
                            <div className="flex items-center gap-2 text-xs">
                                <div className="w-4 h-0.5 bg-purple-500" style={{ background: 'repeating-linear-gradient(90deg, #8b5cf6, #8b5cf6 4px, transparent 4px, transparent 8px)' }} />
                                <span className="text-gray-600 dark:text-gray-400">의미적</span>
                            </div>
                            <div className="flex items-center gap-2 text-xs">
                                <div className="w-4 h-1 bg-emerald-500" />
                                <span className="text-gray-600 dark:text-gray-400">위상동형</span>
                            </div>
                        </div>
                    </div>
                </Panel>

                {/* 통계 패널 */}
                <Panel position="top-right" className="!bg-white dark:!bg-slate-800 rounded-lg shadow-lg p-3 border border-gray-200 dark:border-slate-700">
                    <div className="flex items-center gap-4 text-xs">
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900 dark:text-white">{entities.length}</div>
                            <div className="text-gray-500 dark:text-gray-400">엔티티</div>
                        </div>
                        <div className="w-px h-8 bg-gray-200 dark:bg-slate-700" />
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900 dark:text-white">{relationships.length}</div>
                            <div className="text-gray-500 dark:text-gray-400">관계</div>
                        </div>
                    </div>
                </Panel>
            </ReactFlow>
        </div>
    );
}

export default EntityRelationshipGraph;
