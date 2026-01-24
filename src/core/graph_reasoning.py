"""
Graph Reasoning Agent

arXiv:2410.05130 기반 Multi-Agent Graph Reasoning 구현.
Knowledge Graph에서 multi-hop 추론을 수행하여 인사이트 발견.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json

from src.core.knowledge_graph import (
    KnowledgeGraph,
    KGEntity,
    KGRelation,
    EntityType,
    RelationType
)

# LLM helper function
def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM for reasoning explanations."""
    try:
        from src.common.utils.llm import chat_completion, get_openai_client
        client = get_openai_client()
        response = chat_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response
    except Exception:
        return ""


class ReasoningType(str, Enum):
    """Types of graph reasoning."""
    PATH_FINDING = "path_finding"
    PATTERN_MATCHING = "pattern_matching"
    COMMUNITY_ANALYSIS = "community_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CAUSAL_INFERENCE = "causal_inference"
    IMPACT_ANALYSIS = "impact_analysis"


@dataclass
class ReasoningQuery:
    """Query for graph reasoning."""
    query_type: ReasoningType
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None
    entity_types: List[EntityType] = field(default_factory=list)
    relation_types: List[RelationType] = field(default_factory=list)
    max_hops: int = 3
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result from graph reasoning."""
    query_type: ReasoningType
    success: bool
    findings: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GraphReasoningAgent:
    """
    Graph Reasoning Agent for multi-hop reasoning on Knowledge Graph.

    Implements:
    1. Multi-hop path finding
    2. Pattern matching
    3. Community-based reasoning
    4. Causal inference
    5. Impact analysis
    """

    def __init__(self, kg: KnowledgeGraph, use_llm: bool = True):
        """
        Initialize Graph Reasoning Agent.

        Args:
            kg: Knowledge Graph to reason over
            use_llm: Whether to use LLM for enhanced reasoning
        """
        self.kg = kg
        self.use_llm = use_llm
        self.reasoning_cache: Dict[str, ReasoningResult] = {}

    def reason(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Execute reasoning query on the knowledge graph.

        Args:
            query: Reasoning query

        Returns:
            Reasoning result
        """
        if query.query_type == ReasoningType.PATH_FINDING:
            return self._path_finding(query)
        elif query.query_type == ReasoningType.PATTERN_MATCHING:
            return self._pattern_matching(query)
        elif query.query_type == ReasoningType.COMMUNITY_ANALYSIS:
            return self._community_analysis(query)
        elif query.query_type == ReasoningType.ANOMALY_DETECTION:
            return self._anomaly_detection(query)
        elif query.query_type == ReasoningType.CAUSAL_INFERENCE:
            return self._causal_inference(query)
        elif query.query_type == ReasoningType.IMPACT_ANALYSIS:
            return self._impact_analysis(query)
        else:
            return ReasoningResult(
                query_type=query.query_type,
                success=False,
                explanation=f"Unknown reasoning type: {query.query_type}"
            )

    def _path_finding(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Find paths between entities in the knowledge graph.

        Implements BFS-based multi-hop path finding.
        """
        if not query.source_entity or not query.target_entity:
            return ReasoningResult(
                query_type=ReasoningType.PATH_FINDING,
                success=False,
                explanation="Source and target entities required"
            )

        # Find paths using KG's find_paths method
        paths = self.kg.find_paths(
            source_id=query.source_entity,
            target_id=query.target_entity,
            max_hops=query.max_hops
        )

        if not paths:
            return ReasoningResult(
                query_type=ReasoningType.PATH_FINDING,
                success=False,
                paths=[],
                explanation=f"No paths found between {query.source_entity} and {query.target_entity}"
            )

        # Analyze paths
        findings = []
        for path in paths:
            path_info = self._analyze_path(path)
            findings.append(path_info)

        # Calculate confidence based on path length and relation types
        avg_path_length = sum(len(p) for p in paths) / len(paths)
        confidence = max(0.5, 1.0 - (avg_path_length - 1) * 0.15)

        explanation = f"Found {len(paths)} paths between entities"
        if self.use_llm:
            explanation = self._llm_explain_paths(paths, findings)

        return ReasoningResult(
            query_type=ReasoningType.PATH_FINDING,
            success=True,
            findings=findings,
            paths=paths,
            confidence=confidence,
            explanation=explanation,
            evidence={"path_count": len(paths), "avg_length": avg_path_length}
        )

    def _analyze_path(self, path: List[str]) -> Dict[str, Any]:
        """Analyze a single path."""
        entities = []
        relations = []

        for i, entity_id in enumerate(path):
            entity = self.kg.entities.get(entity_id)
            if entity:
                entities.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type.value
                })

            # Get relation to next entity
            if i < len(path) - 1:
                next_id = path[i + 1]
                for rel in self.kg.relations.values():
                    if rel.source_id == entity_id and rel.target_id == next_id:
                        relations.append({
                            "type": rel.relation_type.value,
                            "confidence": rel.confidence
                        })
                        break

        return {
            "path_length": len(path),
            "entities": entities,
            "relations": relations,
            "semantic_meaning": self._infer_path_meaning(entities, relations)
        }

    def _infer_path_meaning(self, entities: List[Dict], relations: List[Dict]) -> str:
        """Infer semantic meaning of a path."""
        if not entities:
            return "Empty path"

        # Build path description
        parts = []
        for i, entity in enumerate(entities):
            parts.append(f"{entity['name']} ({entity['type']})")
            if i < len(relations):
                parts.append(f"--[{relations[i]['type']}]-->")

        return " ".join(parts)

    def _pattern_matching(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Find patterns in the knowledge graph.

        Looks for common structural patterns:
        - Star patterns (hub nodes)
        - Chain patterns
        - Cycle patterns
        """
        patterns = []

        # 1. Find hub nodes (entities with many connections)
        hub_threshold = 3
        hub_nodes = []
        for entity_id in self.kg.entities:
            neighbors = self.kg.get_neighbors(entity_id)
            if len(neighbors) >= hub_threshold:
                entity = self.kg.entities[entity_id]
                hub_nodes.append({
                    "entity_id": entity_id,
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "connection_count": len(neighbors),
                    "pattern_type": "hub"
                })

        if hub_nodes:
            patterns.append({
                "pattern_type": "hub_nodes",
                "instances": hub_nodes,
                "description": f"Found {len(hub_nodes)} hub entities with many connections"
            })

        # 2. Find entity type clusters
        type_clusters = {}
        for entity in self.kg.entities.values():
            etype = entity.entity_type.value
            if etype not in type_clusters:
                type_clusters[etype] = []
            type_clusters[etype].append(entity.id)

        patterns.append({
            "pattern_type": "type_distribution",
            "instances": {k: len(v) for k, v in type_clusters.items()},
            "description": f"Entity distribution across {len(type_clusters)} types"
        })

        # 3. Find relation type patterns
        relation_counts = {}
        for rel in self.kg.relations.values():
            rtype = rel.relation_type.value
            relation_counts[rtype] = relation_counts.get(rtype, 0) + 1

        if relation_counts:
            patterns.append({
                "pattern_type": "relation_distribution",
                "instances": relation_counts,
                "description": f"Relation distribution across {len(relation_counts)} types"
            })

        # 4. Find isolated entities
        isolated = []
        for entity_id in self.kg.entities:
            neighbors = self.kg.get_neighbors(entity_id)
            if len(neighbors) == 0:
                entity = self.kg.entities[entity_id]
                isolated.append({
                    "entity_id": entity_id,
                    "name": entity.name,
                    "type": entity.entity_type.value
                })

        if isolated:
            patterns.append({
                "pattern_type": "isolated_entities",
                "instances": isolated,
                "description": f"Found {len(isolated)} isolated entities (potential data silos)"
            })

        confidence = 0.8 if patterns else 0.5

        return ReasoningResult(
            query_type=ReasoningType.PATTERN_MATCHING,
            success=True,
            patterns=patterns,
            confidence=confidence,
            explanation=f"Identified {len(patterns)} structural patterns in the knowledge graph",
            evidence={"pattern_count": len(patterns)}
        )

    def _community_analysis(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Analyze communities in the knowledge graph.

        Uses Louvain algorithm and provides insights about each community.
        """
        # Detect communities if not already done
        communities = self.kg.detect_communities()

        if not communities:
            return ReasoningResult(
                query_type=ReasoningType.COMMUNITY_ANALYSIS,
                success=False,
                explanation="No communities detected (graph may be too small)"
            )

        findings = []
        # communities is a Dict[int, KGCommunity], iterate over values
        for community in communities.values():
            # Analyze community composition
            entity_types = {}
            for entity_id in community.entity_ids:
                entity = self.kg.entities.get(entity_id)
                if entity:
                    etype = entity.entity_type.value
                    entity_types[etype] = entity_types.get(etype, 0) + 1

            # Find dominant type
            dominant_type = max(entity_types.items(), key=lambda x: x[1]) if entity_types else ("unknown", 0)

            # Get inter-community relations
            external_relations = 0
            internal_relations = 0
            for rel in self.kg.relations.values():
                source_in = rel.source_id in community.entity_ids
                target_in = rel.target_id in community.entity_ids
                if source_in and target_in:
                    internal_relations += 1
                elif source_in or target_in:
                    external_relations += 1

            finding = {
                "community_id": community.id,
                "size": community.size,
                "entity_type_distribution": entity_types,
                "dominant_type": dominant_type[0],
                "internal_relations": internal_relations,
                "external_relations": external_relations,
                "cohesion": internal_relations / (internal_relations + external_relations + 1),
                "summary": community.summary
            }
            findings.append(finding)

        # Sort by size
        findings.sort(key=lambda x: x["size"], reverse=True)

        confidence = min(0.9, 0.5 + len(communities) * 0.1)

        explanation = f"Analyzed {len(communities)} communities"
        if self.use_llm:
            explanation = self._llm_explain_communities(findings)

        return ReasoningResult(
            query_type=ReasoningType.COMMUNITY_ANALYSIS,
            success=True,
            findings=findings,
            confidence=confidence,
            explanation=explanation,
            evidence={"community_count": len(communities)}
        )

    def _anomaly_detection(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Detect anomalies in the knowledge graph structure.

        Looks for:
        - Orphan entities
        - Unusual relation patterns
        - Inconsistent entity properties
        """
        anomalies = []

        # 1. Find orphan entities (no relations)
        for entity_id, entity in self.kg.entities.items():
            neighbors = self.kg.get_neighbors(entity_id)
            if len(neighbors) == 0:
                anomalies.append({
                    "type": "orphan_entity",
                    "entity_id": entity_id,
                    "entity_name": entity.name,
                    "severity": "medium",
                    "description": f"Entity '{entity.name}' has no connections"
                })

        # 2. Find entities with unusual connection patterns
        connection_counts = [len(self.kg.get_neighbors(eid)) for eid in self.kg.entities]
        if connection_counts:
            avg_connections = sum(connection_counts) / len(connection_counts)
            std_threshold = avg_connections * 2

            for entity_id, entity in self.kg.entities.items():
                conn_count = len(self.kg.get_neighbors(entity_id))
                if conn_count > std_threshold and conn_count > 5:
                    anomalies.append({
                        "type": "high_connectivity",
                        "entity_id": entity_id,
                        "entity_name": entity.name,
                        "connection_count": conn_count,
                        "average": avg_connections,
                        "severity": "low",
                        "description": f"Entity '{entity.name}' has unusually high connectivity ({conn_count} vs avg {avg_connections:.1f})"
                    })

        # 3. Find low-confidence relations
        for rel_id, rel in self.kg.relations.items():
            if rel.confidence < 0.5:
                source = self.kg.entities.get(rel.source_id)
                target = self.kg.entities.get(rel.target_id)
                anomalies.append({
                    "type": "low_confidence_relation",
                    "relation_id": rel_id,
                    "source": source.name if source else rel.source_id,
                    "target": target.name if target else rel.target_id,
                    "confidence": rel.confidence,
                    "severity": "medium",
                    "description": f"Low confidence relation ({rel.confidence:.2f}) between entities"
                })

        confidence = 0.75 if anomalies else 0.9

        return ReasoningResult(
            query_type=ReasoningType.ANOMALY_DETECTION,
            success=True,
            findings=anomalies,
            confidence=confidence,
            explanation=f"Detected {len(anomalies)} structural anomalies",
            evidence={"anomaly_count": len(anomalies)}
        )

    def _causal_inference(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Infer causal relationships from the knowledge graph.

        Uses relation semantics and path analysis to identify potential causality.
        """
        causal_chains = []

        # Look for chains of DEPENDS_ON, INFLUENCES, DERIVED_FROM relations
        causal_relation_types = {
            RelationType.DEPENDS_ON,
            RelationType.DERIVED_FROM,
            RelationType.HOMEOMORPHIC_TO
        }

        # Build adjacency for causal relations only
        causal_edges = {}
        for rel in self.kg.relations.values():
            if rel.relation_type in causal_relation_types:
                if rel.source_id not in causal_edges:
                    causal_edges[rel.source_id] = []
                causal_edges[rel.source_id].append({
                    "target": rel.target_id,
                    "type": rel.relation_type.value,
                    "confidence": rel.confidence
                })

        # Find causal chains using DFS
        for start_entity in causal_edges:
            chains = self._find_causal_chains(start_entity, causal_edges, max_length=4)
            for chain in chains:
                if len(chain) > 1:
                    chain_info = {
                        "start": start_entity,
                        "chain": chain,
                        "length": len(chain),
                        "entities": [self.kg.entities.get(eid, {}).name if self.kg.entities.get(eid) else eid for eid in chain]
                    }
                    causal_chains.append(chain_info)

        # Deduplicate and sort by length
        unique_chains = []
        seen = set()
        for chain in causal_chains:
            chain_key = tuple(chain["chain"])
            if chain_key not in seen:
                seen.add(chain_key)
                unique_chains.append(chain)

        unique_chains.sort(key=lambda x: x["length"], reverse=True)

        confidence = 0.6 if unique_chains else 0.4

        return ReasoningResult(
            query_type=ReasoningType.CAUSAL_INFERENCE,
            success=True,
            findings=unique_chains[:10],  # Top 10 chains
            confidence=confidence,
            explanation=f"Identified {len(unique_chains)} potential causal chains",
            evidence={"chain_count": len(unique_chains)}
        )

    def _find_causal_chains(
        self,
        start: str,
        edges: Dict[str, List[Dict]],
        max_length: int
    ) -> List[List[str]]:
        """Find all causal chains starting from an entity."""
        chains = []

        def dfs(current: str, path: List[str], depth: int):
            if depth >= max_length:
                return
            if current in edges:
                for edge in edges[current]:
                    target = edge["target"]
                    if target not in path:  # Avoid cycles
                        new_path = path + [target]
                        chains.append(new_path)
                        dfs(target, new_path, depth + 1)

        dfs(start, [start], 0)
        return chains

    def _impact_analysis(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Analyze impact of changes to an entity.

        Calculates which entities would be affected by changes to the source entity.
        """
        if not query.source_entity:
            return ReasoningResult(
                query_type=ReasoningType.IMPACT_ANALYSIS,
                success=False,
                explanation="Source entity required for impact analysis"
            )

        source_entity = self.kg.entities.get(query.source_entity)
        if not source_entity:
            return ReasoningResult(
                query_type=ReasoningType.IMPACT_ANALYSIS,
                success=False,
                explanation=f"Entity {query.source_entity} not found"
            )

        # BFS to find all reachable entities
        impacted = {}
        visited = set()
        queue = [(query.source_entity, 0)]  # (entity_id, distance)

        while queue:
            current_id, distance = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != query.source_entity:
                entity = self.kg.entities.get(current_id)
                impacted[current_id] = {
                    "entity_id": current_id,
                    "name": entity.name if entity else current_id,
                    "type": entity.entity_type.value if entity else "unknown",
                    "distance": distance,
                    "impact_score": 1.0 / (distance + 1)  # Closer = higher impact
                }

            if distance < query.max_hops:
                neighbors = self.kg.get_neighbors(current_id)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

        # Sort by impact score
        impact_list = sorted(impacted.values(), key=lambda x: x["impact_score"], reverse=True)

        # Group by distance
        by_distance = {}
        for imp in impact_list:
            dist = imp["distance"]
            if dist not in by_distance:
                by_distance[dist] = []
            by_distance[dist].append(imp)

        findings = {
            "source_entity": {
                "id": source_entity.id,
                "name": source_entity.name,
                "type": source_entity.entity_type.value
            },
            "total_impacted": len(impact_list),
            "by_distance": by_distance,
            "high_impact_entities": impact_list[:5]  # Top 5
        }

        confidence = min(0.9, 0.5 + len(impact_list) * 0.05)

        return ReasoningResult(
            query_type=ReasoningType.IMPACT_ANALYSIS,
            success=True,
            findings=[findings],
            confidence=confidence,
            explanation=f"Changes to '{source_entity.name}' would impact {len(impact_list)} entities",
            evidence={"impacted_count": len(impact_list)}
        )

    def _llm_explain_paths(self, paths: List[List[str]], findings: List[Dict]) -> str:
        """Use LLM to explain path findings."""
        try:
            prompt = f"""Analyze the following paths found in a knowledge graph:

Paths found: {len(paths)}
Path details:
{json.dumps(findings[:3], indent=2)}

Provide a concise explanation (2-3 sentences) of what these paths reveal about the data relationships."""

            response = call_llm(prompt, model="gpt-4o-mini")
            return response.strip()
        except Exception:
            return f"Found {len(paths)} paths connecting the entities"

    def _llm_explain_communities(self, findings: List[Dict]) -> str:
        """Use LLM to explain community findings."""
        try:
            prompt = f"""Analyze the following community structure in a knowledge graph:

Communities found: {len(findings)}
Community details:
{json.dumps(findings[:5], indent=2)}

Provide a concise explanation (2-3 sentences) of what these communities reveal about the data organization."""

            response = call_llm(prompt, model="gpt-4o-mini")
            return response.strip()
        except Exception:
            return f"Identified {len(findings)} distinct communities in the knowledge graph"

    def comprehensive_analysis(self) -> Dict[str, ReasoningResult]:
        """
        Run all reasoning types and return comprehensive analysis.

        Returns:
            Dict mapping reasoning type to results
        """
        results = {}

        # Pattern matching
        results["patterns"] = self.reason(ReasoningQuery(
            query_type=ReasoningType.PATTERN_MATCHING
        ))

        # Community analysis
        results["communities"] = self.reason(ReasoningQuery(
            query_type=ReasoningType.COMMUNITY_ANALYSIS
        ))

        # Anomaly detection
        results["anomalies"] = self.reason(ReasoningQuery(
            query_type=ReasoningType.ANOMALY_DETECTION
        ))

        # Causal inference
        results["causal"] = self.reason(ReasoningQuery(
            query_type=ReasoningType.CAUSAL_INFERENCE
        ))

        return results

    def generate_insights(self) -> List[Dict[str, Any]]:
        """
        Generate insights from comprehensive graph analysis.

        Returns:
            List of insights derived from graph reasoning
        """
        analysis = self.comprehensive_analysis()
        insights = []

        # Insights from patterns
        if analysis["patterns"].success:
            for pattern in analysis["patterns"].patterns:
                if pattern["pattern_type"] == "hub_nodes":
                    for hub in pattern.get("instances", [])[:3]:
                        insights.append({
                            "id": f"GR-HUB-{len(insights)+1:03d}",
                            "type": "graph_pattern",
                            "subtype": "hub_entity",
                            "description": f"'{hub['name']}' is a key connecting entity with {hub['connection_count']} connections",
                            "confidence": analysis["patterns"].confidence,
                            "evidence": hub
                        })

                elif pattern["pattern_type"] == "isolated_entities":
                    if pattern.get("instances"):
                        insights.append({
                            "id": f"GR-ISO-{len(insights)+1:03d}",
                            "type": "graph_pattern",
                            "subtype": "data_silo",
                            "description": f"Found {len(pattern['instances'])} isolated data entities (potential silos)",
                            "confidence": 0.8,
                            "severity": "medium",
                            "evidence": {"isolated_count": len(pattern["instances"])}
                        })

        # Insights from communities
        if analysis["communities"].success:
            for finding in analysis["communities"].findings[:3]:
                if finding["size"] > 2:
                    insights.append({
                        "id": f"GR-COM-{len(insights)+1:03d}",
                        "type": "community_insight",
                        "description": f"Data cluster identified with {finding['size']} entities, dominated by {finding['dominant_type']} types",
                        "confidence": finding["cohesion"],
                        "evidence": finding
                    })

        # Insights from anomalies
        if analysis["anomalies"].success:
            for anomaly in analysis["anomalies"].findings[:5]:
                insights.append({
                    "id": f"GR-ANO-{len(insights)+1:03d}",
                    "type": "anomaly",
                    "subtype": anomaly["type"],
                    "description": anomaly["description"],
                    "confidence": 0.75,
                    "severity": anomaly.get("severity", "medium"),
                    "evidence": anomaly
                })

        # Insights from causal chains
        if analysis["causal"].success:
            for chain in analysis["causal"].findings[:3]:
                if chain["length"] >= 3:
                    insights.append({
                        "id": f"GR-CAU-{len(insights)+1:03d}",
                        "type": "causal_chain",
                        "description": f"Dependency chain: {' -> '.join(chain['entities'])}",
                        "confidence": analysis["causal"].confidence,
                        "evidence": chain
                    })

        return insights
