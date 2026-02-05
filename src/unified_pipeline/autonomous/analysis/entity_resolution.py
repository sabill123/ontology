"""
Entity Resolution Module

ML 기반 엔티티 매칭 및 레코드 링키지
- Blocking (MinHash LSH)
- Similarity Functions (Embedding 기반)
- Classification (ML 기반 매칭 결정)
- Clustering (Transitive closure)

References:
- https://github.com/dedupeio/dedupe
- https://github.com/moj-analytical-services/splink
- Paper: "Active in-context learning for cross-domain entity resolution" (2024)
"""

import logging
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    logger.warning("datasketch not available. Install with: pip install datasketch")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available for semantic matching")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EntityCandidate:
    """엔티티 후보"""
    record_id: str
    source_table: str
    key_value: Any
    attributes: Dict[str, Any]
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_table": self.source_table,
            "key_value": self.key_value,
            "attributes": self.attributes,
        }


@dataclass
class EntityMatch:
    """엔티티 매칭 결과"""
    entity_a: EntityCandidate
    entity_b: EntityCandidate
    similarity_score: float
    match_confidence: float
    match_type: str  # "exact", "fuzzy", "semantic"
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_a": self.entity_a.to_dict(),
            "entity_b": self.entity_b.to_dict(),
            "similarity_score": self.similarity_score,
            "match_confidence": self.match_confidence,
            "match_type": self.match_type,
            "evidence": self.evidence,
        }


@dataclass
class EntityCluster:
    """동일 엔티티 클러스터"""
    cluster_id: str
    canonical_id: str
    members: List[EntityCandidate]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "canonical_id": self.canonical_id,
            "members": [m.to_dict() for m in self.members],
            "member_count": len(self.members),
            "source_tables": list(set(m.source_table for m in self.members)),
            "confidence": self.confidence,
        }


@dataclass
class ActiveLearningState:
    """Active Learning 상태 관리"""
    labeled_pairs: Dict[Tuple[str, str], bool] = field(default_factory=dict)  # (id_a, id_b) -> is_match
    uncertain_pairs: List[Tuple[str, str, float]] = field(default_factory=list)  # uncertainty 순
    iteration: int = 0
    model_accuracy: float = 0.0

    def add_label(self, id_a: str, id_b: str, is_match: bool):
        """라벨 추가"""
        pair = tuple(sorted([id_a, id_b]))
        self.labeled_pairs[pair] = is_match

    def get_label(self, id_a: str, id_b: str) -> Optional[bool]:
        """라벨 조회"""
        pair = tuple(sorted([id_a, id_b]))
        return self.labeled_pairs.get(pair)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_labels": len(self.labeled_pairs),
            "num_positive": sum(1 for v in self.labeled_pairs.values() if v),
            "num_negative": sum(1 for v in self.labeled_pairs.values() if not v),
            "iteration": self.iteration,
            "model_accuracy": self.model_accuracy,
        }


class ActiveLearningClassifier:
    """
    Active Learning 기반 매칭 분류기

    Query Strategies:
    - Uncertainty Sampling: 가장 불확실한 쌍 선택
    - Query-by-Committee: 위원회 투표 불일치 기반
    - Expected Model Change: 모델 변화 최대화

    References:
    - Settles, "Active Learning Literature Survey", 2009
    - Christen, "Data Matching", 2012
    - Dedupe.io Active Learning implementation
    """

    def __init__(
        self,
        n_committee: int = 5,
        uncertainty_threshold: float = 0.3,
        min_labels: int = 10,
        query_strategy: str = "uncertainty",  # uncertainty, qbc, random
    ):
        self.n_committee = n_committee
        self.uncertainty_threshold = uncertainty_threshold
        self.min_labels = min_labels
        self.query_strategy = query_strategy
        self.committee_weights: List[Any] = []
        self.feature_dim = 0
        self.is_trained = False

    def _extract_features(
        self,
        entity_a: EntityCandidate,
        entity_b: EntityCandidate,
        similarity_calc: 'SimilarityCalculator',
    ) -> List[float]:
        """
        엔티티 쌍에서 특성 추출 (8-dimensional)

        Features:
        1. Key Jaro-Winkler similarity
        2. Key exact match (0/1)
        3. Attribute avg similarity
        4. Attribute max similarity
        5. Attribute min similarity
        6. Common attribute ratio
        7. Cross-table indicator
        8. Levenshtein similarity
        """
        features = []

        # 1-2. Key similarity
        key_a = str(entity_a.key_value or "")
        key_b = str(entity_b.key_value or "")

        jw_sim = similarity_calc.jaro_winkler_similarity(key_a, key_b) if key_a and key_b else 0.0
        features.append(jw_sim)
        features.append(1.0 if key_a == key_b and key_a else 0.0)

        # 3-5. Attribute similarities
        common_attrs = set(entity_a.attributes.keys()) & set(entity_b.attributes.keys())
        attr_sims = []

        for attr in common_attrs:
            val_a = str(entity_a.attributes[attr] or "")
            val_b = str(entity_b.attributes[attr] or "")
            if val_a and val_b:
                attr_sims.append(similarity_calc.jaro_winkler_similarity(val_a, val_b))

        if attr_sims:
            features.append(float(np.mean(attr_sims)) if NUMPY_AVAILABLE else sum(attr_sims) / len(attr_sims))
            features.append(max(attr_sims))
            features.append(min(attr_sims))
        else:
            features.extend([0.0, 0.0, 0.0])

        # 6. Common attribute ratio
        all_attrs = set(entity_a.attributes.keys()) | set(entity_b.attributes.keys())
        features.append(len(common_attrs) / len(all_attrs) if all_attrs else 0.0)

        # 7. Cross-table indicator (다른 테이블이면 1)
        features.append(1.0 if entity_a.source_table != entity_b.source_table else 0.0)

        # 8. Levenshtein similarity
        lev_sim = similarity_calc.levenshtein_similarity(key_a, key_b) if key_a and key_b else 0.0
        features.append(lev_sim)

        return features

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with clipping"""
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + np.exp(-x)) if NUMPY_AVAILABLE else 1.0 / (1.0 + 2.718281828 ** (-x))

    def _train_single_model(self, X: List[List[float]], y: List[float]) -> List[float]:
        """단일 로지스틱 회귀 모델 학습"""
        if not NUMPY_AVAILABLE:
            return [0.0] * (len(X[0]) + 1)

        X_arr = np.array(X)
        y_arr = np.array(y)

        # Add bias term
        X_bias = np.column_stack([np.ones(len(X_arr)), X_arr])
        weights = np.zeros(X_bias.shape[1])

        # Gradient descent with L2 regularization
        lr = 0.1
        reg = 0.01

        for _ in range(200):
            logits = X_bias @ weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            gradient = X_bias.T @ (probs - y_arr) / len(y_arr) + reg * weights
            weights -= lr * gradient

        return weights.tolist()

    def fit(
        self,
        candidate_pairs: List[Tuple[EntityCandidate, EntityCandidate]],
        labeled_pairs: Dict[Tuple[str, str], bool],
        similarity_calc: 'SimilarityCalculator',
    ) -> bool:
        """
        라벨된 데이터로 Committee 모델 학습

        Returns:
            학습 성공 여부
        """
        if len(labeled_pairs) < self.min_labels:
            logger.info(f"Not enough labels ({len(labeled_pairs)}/{self.min_labels}), skipping training")
            return False

        # 특성 및 라벨 준비
        X_list = []
        y_list = []

        for entity_a, entity_b in candidate_pairs:
            pair_key = tuple(sorted([entity_a.record_id, entity_b.record_id]))

            if pair_key in labeled_pairs:
                feat = self._extract_features(entity_a, entity_b, similarity_calc)
                X_list.append(feat)
                y_list.append(1.0 if labeled_pairs[pair_key] else 0.0)

        if len(X_list) < self.min_labels:
            return False

        # Committee 학습 (Bootstrap Aggregating)
        self.committee_weights = []
        n = len(X_list)

        for _ in range(self.n_committee):
            # Bootstrap 샘플링
            if NUMPY_AVAILABLE:
                idx = np.random.choice(n, n, replace=True).tolist()
            else:
                import random
                idx = [random.randint(0, n - 1) for _ in range(n)]

            X_boot = [X_list[i] for i in idx]
            y_boot = [y_list[i] for i in idx]

            weights = self._train_single_model(X_boot, y_boot)
            self.committee_weights.append(weights)

        self.feature_dim = len(X_list[0])
        self.is_trained = True
        logger.info(f"Active Learning classifier trained with {len(X_list)} labeled pairs, {self.n_committee} committee members")

        return True

    def predict_proba(self, features: List[float]) -> float:
        """Committee 평균 확률 예측"""
        if not self.is_trained or not self.committee_weights:
            return 0.5

        if not NUMPY_AVAILABLE:
            return 0.5

        feat_arr = np.array([1.0] + features)  # Add bias
        probs = []

        for weights in self.committee_weights:
            w = np.array(weights)
            logit = feat_arr @ w
            prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
            probs.append(prob)

        return float(np.mean(probs))

    def compute_uncertainty(self, features: List[float]) -> float:
        """
        불확실성 계산 (Query-by-Committee Vote Entropy)

        H(v) = -sum(p_k * log2(p_k))
        """
        if not self.is_trained or not self.committee_weights:
            return 1.0

        if not NUMPY_AVAILABLE:
            return 1.0

        feat_arr = np.array([1.0] + features)
        votes = []

        for weights in self.committee_weights:
            w = np.array(weights)
            logit = feat_arr @ w
            prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
            votes.append(1 if prob > 0.5 else 0)

        # Vote entropy
        vote_pos = sum(votes) / len(votes)
        vote_neg = 1 - vote_pos

        if vote_pos < 1e-10 or vote_neg < 1e-10:
            return 0.0

        entropy = -(vote_pos * np.log2(vote_pos) + vote_neg * np.log2(vote_neg))
        return float(entropy)

    def select_queries(
        self,
        candidate_pairs: List[Tuple[EntityCandidate, EntityCandidate]],
        similarity_calc: 'SimilarityCalculator',
        labeled_pairs: Dict[Tuple[str, str], bool],
        n_queries: int = 10,
    ) -> List[Tuple[EntityCandidate, EntityCandidate, float]]:
        """
        Active Learning 쿼리 선택

        Returns:
            [(entity_a, entity_b, uncertainty_score), ...] - 가장 불확실한 쌍들
        """
        unlabeled = []

        for entity_a, entity_b in candidate_pairs:
            pair_key = tuple(sorted([entity_a.record_id, entity_b.record_id]))

            if pair_key not in labeled_pairs:
                features = self._extract_features(entity_a, entity_b, similarity_calc)

                if self.query_strategy == "uncertainty":
                    # 확률이 0.5에 가까울수록 불확실
                    prob = self.predict_proba(features)
                    uncertainty = 1.0 - 2 * abs(prob - 0.5)
                elif self.query_strategy == "qbc":
                    uncertainty = self.compute_uncertainty(features)
                else:  # random
                    uncertainty = np.random.random() if NUMPY_AVAILABLE else 0.5

                unlabeled.append((entity_a, entity_b, uncertainty))

        # 불확실성 높은 순 정렬
        unlabeled.sort(key=lambda x: -x[2])

        return unlabeled[:n_queries]

    def predict(
        self,
        entity_a: EntityCandidate,
        entity_b: EntityCandidate,
        similarity_calc: 'SimilarityCalculator',
    ) -> Tuple[float, float, str]:
        """
        매칭 예측

        Returns:
            (match_probability, uncertainty, match_type)
        """
        features = self._extract_features(entity_a, entity_b, similarity_calc)
        prob = self.predict_proba(features)
        uncertainty = self.compute_uncertainty(features)

        if prob > 0.9:
            match_type = "high_confidence"
        elif prob > 0.7:
            match_type = "medium_confidence"
        elif prob > 0.5:
            match_type = "low_confidence"
        else:
            match_type = "non_match"

        return prob, uncertainty, match_type


class MinHashBlocker:
    """MinHash LSH 기반 블로킹"""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.5,
        n_gram: int = 3,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.n_gram = n_gram
        self.lsh = None
        self.minhashes: Dict[str, MinHash] = {}

    def _tokenize(self, text: str) -> Set[str]:
        """텍스트를 n-gram 토큰으로 분할"""
        if not text:
            return set()

        text = str(text).lower().strip()
        # n-gram 생성
        tokens = set()
        for i in range(len(text) - self.n_gram + 1):
            tokens.add(text[i:i + self.n_gram])
        return tokens

    def _create_minhash(self, tokens: Set[str]) -> 'MinHash':
        """토큰 집합에서 MinHash 생성"""
        if not DATASKETCH_AVAILABLE:
            return None

        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode('utf8'))
        return m

    def build_index(self, entities: List[EntityCandidate], key_attr: str = "key_value"):
        """LSH 인덱스 구축"""
        if not DATASKETCH_AVAILABLE:
            logger.warning("datasketch not available, using fallback blocking")
            return

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        for entity in entities:
            # 키 값 기반 토큰화
            key_val = str(getattr(entity, key_attr, "") or entity.attributes.get(key_attr, ""))
            tokens = self._tokenize(key_val)

            # 추가 속성도 포함
            for attr_name, attr_val in entity.attributes.items():
                if isinstance(attr_val, str) and len(attr_val) < 100:
                    tokens.update(self._tokenize(attr_val))

            if tokens:
                minhash = self._create_minhash(tokens)
                self.minhashes[entity.record_id] = minhash
                self.lsh.insert(entity.record_id, minhash)

    def query_candidates(self, entity: EntityCandidate, key_attr: str = "key_value") -> List[str]:
        """유사 후보 쿼리"""
        if not DATASKETCH_AVAILABLE or self.lsh is None:
            return []

        key_val = str(getattr(entity, key_attr, "") or entity.attributes.get(key_attr, ""))
        tokens = self._tokenize(key_val)

        for attr_name, attr_val in entity.attributes.items():
            if isinstance(attr_val, str) and len(attr_val) < 100:
                tokens.update(self._tokenize(attr_val))

        if not tokens:
            return []

        minhash = self._create_minhash(tokens)
        return self.lsh.query(minhash)


class SimilarityCalculator:
    """다양한 유사도 계산"""

    def __init__(self, use_embeddings: bool = False):  # v7.5: 기본값 False로 변경 (메모리 이슈 방지)
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self._embedding_model = None  # Lazy loading
        self._model_loaded = False

    @property
    def embedding_model(self):
        """Lazy load embedding model only when needed"""
        if self.use_embeddings and not self._model_loaded:
            self._model_loaded = True
            try:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence-transformers model (lazy)")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.use_embeddings = False
                self._embedding_model = None
        return self._embedding_model

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """레벤슈타인 거리"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def levenshtein_similarity(self, s1: str, s2: str) -> float:
        """레벤슈타인 유사도 (0~1)"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        max_len = max(len(s1), len(s2))
        dist = self.levenshtein_distance(s1, s2)
        return 1.0 - (dist / max_len)

    def jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Jaro-Winkler 유사도"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        s1, s2 = s1.lower(), s2.lower()

        # Jaro similarity
        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1
        match_distance = max(0, match_distance)

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

        # Winkler adjustment
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + 0.1 * prefix * (1 - jaro)

    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Jaccard 유사도"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def embedding_similarity(self, text1: str, text2: str) -> float:
        """Embedding 기반 의미적 유사도"""
        if not self.use_embeddings or not self.embedding_model:
            return 0.0

        try:
            embeddings = self.embedding_model.encode([text1, text2])
            if SKLEARN_AVAILABLE:
                sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            else:
                # 수동 코사인 유사도
                dot = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
                norm1 = sum(a * a for a in embeddings[0]) ** 0.5
                norm2 = sum(b * b for b in embeddings[1]) ** 0.5
                sim = dot / (norm1 * norm2) if norm1 and norm2 else 0.0
            return float(sim)
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")
            return 0.0

    def compute_entity_similarity(
        self,
        entity_a: EntityCandidate,
        entity_b: EntityCandidate,
        key_weight: float = 0.4,
        attr_weight: float = 0.3,
        semantic_weight: float = 0.3,
    ) -> Tuple[float, str, List[str]]:
        """
        두 엔티티 간 종합 유사도 계산

        Returns:
            (similarity, match_type, evidence)
        """
        evidence = []
        scores = []

        # 1. 키 값 비교
        key_a = str(entity_a.key_value or "")
        key_b = str(entity_b.key_value or "")

        if key_a and key_b:
            if key_a == key_b:
                evidence.append(f"Exact key match: {key_a}")
                return (1.0, "exact", evidence)

            key_sim = self.jaro_winkler_similarity(key_a, key_b)
            scores.append(("key", key_sim, key_weight))
            if key_sim > 0.8:
                evidence.append(f"High key similarity: {key_sim:.2f}")

        # 2. 속성 비교
        common_attrs = set(entity_a.attributes.keys()) & set(entity_b.attributes.keys())
        if common_attrs:
            attr_sims = []
            for attr in common_attrs:
                val_a = str(entity_a.attributes[attr] or "")
                val_b = str(entity_b.attributes[attr] or "")
                if val_a and val_b:
                    sim = self.jaro_winkler_similarity(val_a, val_b)
                    attr_sims.append(sim)
                    if sim > 0.9:
                        evidence.append(f"Attribute match ({attr}): {sim:.2f}")

            if attr_sims:
                avg_attr_sim = sum(attr_sims) / len(attr_sims)
                scores.append(("attr", avg_attr_sim, attr_weight))

        # 3. 의미적 유사도
        if self.use_embeddings:
            # 엔티티 설명 생성
            desc_a = f"{key_a} " + " ".join(str(v) for v in entity_a.attributes.values() if v)
            desc_b = f"{key_b} " + " ".join(str(v) for v in entity_b.attributes.values() if v)

            semantic_sim = self.embedding_similarity(desc_a, desc_b)
            scores.append(("semantic", semantic_sim, semantic_weight))
            if semantic_sim > 0.8:
                evidence.append(f"Semantic similarity: {semantic_sim:.2f}")

        # 가중 평균
        if scores:
            total_weight = sum(w for _, _, w in scores)
            weighted_sum = sum(s * w for _, s, w in scores)
            final_sim = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            final_sim = 0.0

        # 매칭 타입 결정
        if final_sim > 0.9:
            match_type = "exact"
        elif final_sim > 0.7:
            match_type = "fuzzy"
        else:
            match_type = "semantic"

        return (final_sim, match_type, evidence)


class EntityResolver:
    """
    엔티티 해결 메인 클래스

    Workflow:
    1. Blocking: 후보 쌍 축소
    2. Comparison: 상세 유사도 계산
    3. Classification: 매칭 결정 (Threshold 또는 Active Learning)
    4. Clustering: 동일 엔티티 그룹화

    Active Learning Mode:
    - Query-by-Committee 기반 불확실성 샘플링
    - 사용자/오라클 라벨링으로 점진적 모델 개선
    - Threshold 방식 대비 더 정확한 매칭 결정
    """

    def __init__(
        self,
        match_threshold: float = 0.7,
        use_blocking: bool = True,
        use_embeddings: bool = False,  # v7.5: 기본값 False (메모리 이슈 방지)
        use_active_learning: bool = False,
        active_learning_config: Optional[Dict[str, Any]] = None,
    ):
        self.match_threshold = match_threshold
        self.use_blocking = use_blocking
        self.use_active_learning = use_active_learning

        self.blocker = MinHashBlocker() if use_blocking and DATASKETCH_AVAILABLE else None
        self.similarity = SimilarityCalculator(use_embeddings=use_embeddings)

        self.entities: Dict[str, EntityCandidate] = {}
        self.matches: List[EntityMatch] = []
        self.clusters: List[EntityCluster] = []

        # Active Learning 컴포넌트
        if use_active_learning:
            al_config = active_learning_config or {}
            self.al_classifier = ActiveLearningClassifier(
                n_committee=al_config.get("n_committee", 5),
                uncertainty_threshold=al_config.get("uncertainty_threshold", 0.3),
                min_labels=al_config.get("min_labels", 10),
                query_strategy=al_config.get("query_strategy", "qbc"),
            )
            self.al_state = ActiveLearningState()
        else:
            self.al_classifier = None
            self.al_state = None

    def add_entities(
        self,
        records: List[Dict[str, Any]],
        source_table: str,
        key_column: str,
        attribute_columns: Optional[List[str]] = None,
    ):
        """
        레코드를 엔티티로 추가

        Args:
            records: 레코드 딕셔너리 리스트
            source_table: 소스 테이블 이름
            key_column: 키 컬럼 이름
            attribute_columns: 속성으로 사용할 컬럼들
        """
        for i, record in enumerate(records):
            record_id = f"{source_table}_{i}"
            key_value = record.get(key_column)

            # 속성 추출
            if attribute_columns:
                attributes = {col: record.get(col) for col in attribute_columns}
            else:
                attributes = {k: v for k, v in record.items() if k != key_column}

            entity = EntityCandidate(
                record_id=record_id,
                source_table=source_table,
                key_value=key_value,
                attributes=attributes,
            )
            self.entities[record_id] = entity

    def resolve(self, oracle_callback: Optional[Callable[[EntityCandidate, EntityCandidate], bool]] = None) -> List[EntityMatch]:
        """
        엔티티 해결 실행

        Args:
            oracle_callback: Active Learning용 오라클 함수 (entity_a, entity_b) -> is_match
                             None이면 자동 라벨링 (높은 유사도=match, 낮은 유사도=non-match)

        Returns:
            매칭된 엔티티 쌍 리스트
        """
        entities_list = list(self.entities.values())

        if not entities_list:
            return []

        # 1. 블로킹 (선택적)
        candidate_pairs = self._generate_candidate_pairs(entities_list)

        # 2. 분류 방법 선택
        if self.use_active_learning and self.al_classifier:
            return self._resolve_with_active_learning(candidate_pairs, oracle_callback)
        else:
            return self._resolve_with_threshold(candidate_pairs)

    def _generate_candidate_pairs(self, entities_list: List[EntityCandidate]) -> Set[Tuple[str, str]]:
        """후보 쌍 생성 (블로킹 적용)"""
        candidate_pairs = set()

        if self.blocker and self.use_blocking:
            self.blocker.build_index(entities_list)

            for entity in entities_list:
                candidates = self.blocker.query_candidates(entity)
                for cand_id in candidates:
                    if cand_id != entity.record_id:
                        pair = tuple(sorted([entity.record_id, cand_id]))
                        candidate_pairs.add(pair)

            logger.info(f"Blocking reduced to {len(candidate_pairs)} candidate pairs")
        else:
            # 브루트포스 (모든 쌍)
            for i, e1 in enumerate(entities_list):
                for e2 in entities_list[i+1:]:
                    if e1.source_table != e2.source_table:
                        candidate_pairs.add((e1.record_id, e2.record_id))

        return candidate_pairs

    def _resolve_with_threshold(self, candidate_pairs: Set[Tuple[str, str]]) -> List[EntityMatch]:
        """Threshold 기반 매칭 (기존 방식)"""
        self.matches = []

        for id_a, id_b in candidate_pairs:
            entity_a = self.entities[id_a]
            entity_b = self.entities[id_b]

            sim, match_type, evidence = self.similarity.compute_entity_similarity(
                entity_a, entity_b
            )

            if sim >= self.match_threshold:
                match = EntityMatch(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    similarity_score=sim,
                    match_confidence=min(sim, 1.0),
                    match_type=match_type,
                    evidence=evidence,
                )
                self.matches.append(match)

        logger.info(f"[Threshold] Found {len(self.matches)} entity matches")
        return self.matches

    def _resolve_with_active_learning(
        self,
        candidate_pairs: Set[Tuple[str, str]],
        oracle_callback: Optional[Callable] = None,
    ) -> List[EntityMatch]:
        """
        Active Learning 기반 매칭

        1. Bootstrap: 높은/낮은 유사도 쌍으로 초기 라벨 생성
        2. Train: Committee 모델 학습
        3. Query: 불확실한 쌍 선택
        4. Label: 오라클/콜백으로 라벨 획득
        5. Iterate: 수렴까지 반복
        6. Predict: 전체 쌍에 대해 예측
        """
        # 후보 쌍 준비
        pair_entities = []
        for id_a, id_b in candidate_pairs:
            pair_entities.append((self.entities[id_a], self.entities[id_b]))

        # Step 1: Bootstrap - 자동 라벨링으로 초기 학습 데이터 생성
        self._bootstrap_labels(pair_entities)
        logger.info(f"[Active Learning] Bootstrap: {len(self.al_state.labeled_pairs)} initial labels")

        # Step 2-5: Active Learning 반복
        max_iterations = 10
        queries_per_iteration = 5

        for iteration in range(max_iterations):
            self.al_state.iteration = iteration

            # 모델 학습
            trained = self.al_classifier.fit(
                pair_entities,
                self.al_state.labeled_pairs,
                self.similarity
            )

            if not trained:
                logger.info(f"[Active Learning] Not enough labels for training, using threshold fallback")
                return self._resolve_with_threshold(candidate_pairs)

            # 쿼리 선택
            queries = self.al_classifier.select_queries(
                pair_entities,
                self.similarity,
                self.al_state.labeled_pairs,
                n_queries=queries_per_iteration
            )

            if not queries:
                logger.info(f"[Active Learning] No more uncertain pairs, stopping at iteration {iteration}")
                break

            # 라벨링
            new_labels = 0
            for entity_a, entity_b, uncertainty in queries:
                # 오라클 콜백이 있으면 사용, 없으면 자동 라벨링
                if oracle_callback:
                    is_match = oracle_callback(entity_a, entity_b)
                else:
                    # 자동 라벨링: 유사도 기반 (0.85 이상 = match, 0.3 이하 = non-match)
                    sim, _, _ = self.similarity.compute_entity_similarity(entity_a, entity_b)
                    if sim >= 0.85:
                        is_match = True
                    elif sim <= 0.3:
                        is_match = False
                    else:
                        # 중간 영역은 스킵 (진짜 오라클이 필요한 부분)
                        continue

                self.al_state.add_label(entity_a.record_id, entity_b.record_id, is_match)
                new_labels += 1

            logger.info(f"[Active Learning] Iteration {iteration}: added {new_labels} labels, total: {len(self.al_state.labeled_pairs)}")

            # 새 라벨이 없으면 종료
            if new_labels == 0:
                break

        # Step 6: 최종 예측
        self.matches = []

        for entity_a, entity_b in pair_entities:
            pair_key = tuple(sorted([entity_a.record_id, entity_b.record_id]))

            # 이미 라벨이 있으면 그대로 사용
            existing_label = self.al_state.get_label(entity_a.record_id, entity_b.record_id)
            if existing_label is not None:
                if existing_label:
                    sim, match_type, evidence = self.similarity.compute_entity_similarity(entity_a, entity_b)
                    match = EntityMatch(
                        entity_a=entity_a,
                        entity_b=entity_b,
                        similarity_score=sim,
                        match_confidence=0.95,  # 라벨된 데이터는 높은 신뢰도
                        match_type="active_learning_labeled",
                        evidence=evidence + ["Manually labeled as match"],
                    )
                    self.matches.append(match)
                continue

            # 모델 예측
            if self.al_classifier.is_trained:
                prob, uncertainty, match_type = self.al_classifier.predict(
                    entity_a, entity_b, self.similarity
                )

                if prob > 0.5:  # 매칭으로 예측
                    sim, _, evidence = self.similarity.compute_entity_similarity(entity_a, entity_b)
                    match = EntityMatch(
                        entity_a=entity_a,
                        entity_b=entity_b,
                        similarity_score=sim,
                        match_confidence=prob,
                        match_type=f"active_learning_{match_type}",
                        evidence=evidence + [f"AL probability: {prob:.3f}, uncertainty: {uncertainty:.3f}"],
                    )
                    self.matches.append(match)

        logger.info(f"[Active Learning] Final: {len(self.matches)} entity matches")
        return self.matches

    def _bootstrap_labels(self, pair_entities: List[Tuple[EntityCandidate, EntityCandidate]]):
        """Bootstrap: 명확한 쌍에 대해 자동 라벨링"""
        for entity_a, entity_b in pair_entities:
            sim, _, _ = self.similarity.compute_entity_similarity(entity_a, entity_b)

            # 매우 높은 유사도 = positive label
            if sim >= 0.9:
                self.al_state.add_label(entity_a.record_id, entity_b.record_id, True)
            # 매우 낮은 유사도 = negative label
            elif sim <= 0.2:
                self.al_state.add_label(entity_a.record_id, entity_b.record_id, False)

            # 충분한 bootstrap 라벨이 모이면 중단
            if len(self.al_state.labeled_pairs) >= self.al_classifier.min_labels * 2:
                break

    def get_active_learning_state(self) -> Optional[Dict[str, Any]]:
        """Active Learning 상태 반환"""
        if self.al_state:
            return self.al_state.to_dict()
        return None

    def add_oracle_label(self, entity_id_a: str, entity_id_b: str, is_match: bool):
        """외부에서 오라클 라벨 추가"""
        if self.al_state:
            self.al_state.add_label(entity_id_a, entity_id_b, is_match)

    def cluster_entities(self) -> List[EntityCluster]:
        """
        매칭된 엔티티를 클러스터로 그룹화 (Transitive closure)
        """
        if not self.matches:
            return []

        # Union-Find 구조
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 매칭 관계로 Union
        for match in self.matches:
            union(match.entity_a.record_id, match.entity_b.record_id)

        # 클러스터 구성
        cluster_members = defaultdict(list)
        for entity_id, entity in self.entities.items():
            root = find(entity_id)
            cluster_members[root].append(entity)

        # 2개 이상 멤버를 가진 클러스터만
        self.clusters = []
        for i, (root, members) in enumerate(cluster_members.items()):
            if len(members) > 1:
                # 신뢰도: 해당 클러스터 내 매칭의 평균 신뢰도
                cluster_matches = [
                    m for m in self.matches
                    if find(m.entity_a.record_id) == root
                ]
                avg_confidence = (
                    sum(m.match_confidence for m in cluster_matches) / len(cluster_matches)
                    if cluster_matches else 0.5
                )

                cluster = EntityCluster(
                    cluster_id=f"CLUSTER_{i:04d}",
                    canonical_id=root,
                    members=members,
                    confidence=avg_confidence,
                )
                self.clusters.append(cluster)

        logger.info(f"Created {len(self.clusters)} entity clusters")
        return self.clusters

    def get_cross_table_mappings(self) -> Dict[str, List[Dict]]:
        """
        테이블 간 매핑 관계 추출

        Returns:
            {
                "table_a->table_b": [
                    {"key_a": "...", "key_b": "...", "confidence": 0.9},
                    ...
                ]
            }
        """
        mappings = defaultdict(list)

        for match in self.matches:
            if match.entity_a.source_table != match.entity_b.source_table:
                key = f"{match.entity_a.source_table}->{match.entity_b.source_table}"
                mappings[key].append({
                    "key_a": match.entity_a.key_value,
                    "key_b": match.entity_b.key_value,
                    "confidence": match.match_confidence,
                    "match_type": match.match_type,
                })

        return dict(mappings)

    # =========================================================================
    # 편의 메서드: discovery.py 에이전트에서 사용하는 API
    # =========================================================================

    def resolve_entities(
        self,
        entity_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        엔티티 후보 리스트에서 중복 해결 (에이전트 호출용)

        Args:
            entity_candidates: [{
                "entity_id": ...,
                "name": ...,
                "source_tables": [...],
                "attributes": {...}
            }, ...]

        Returns:
            해결된 엔티티 리스트 [{
                "cluster_id": ...,
                "canonical_name": ...,
                "members": [...],
                "confidence": ...
            }, ...]
        """
        if not entity_candidates:
            return []

        # 엔티티를 내부 형식으로 변환하여 추가
        for i, cand in enumerate(entity_candidates):
            record_id = cand.get("entity_id", f"entity_{i}")
            source_tables = cand.get("source_tables", [])
            source_table = source_tables[0] if source_tables else "unknown"

            entity = EntityCandidate(
                record_id=record_id,
                source_table=source_table,
                key_value=cand.get("name", ""),
                attributes=cand.get("attributes", {}),
            )
            self.entities[record_id] = entity

        # 매칭 실행
        self.resolve()

        # 클러스터링
        clusters = self.cluster_entities()

        # 결과 변환
        results = []
        for cluster in clusters:
            results.append({
                "cluster_id": cluster.cluster_id,
                "canonical_name": cluster.members[0].key_value if cluster.members else "",
                "canonical_id": cluster.canonical_id,
                "members": [
                    {
                        "entity_id": m.record_id,
                        "name": m.key_value,
                        "source_table": m.source_table,
                        "attributes": m.attributes,
                    }
                    for m in cluster.members
                ],
                "member_count": len(cluster.members),
                "confidence": cluster.confidence,
            })

        return results


# Convenience function
def resolve_entities(
    tables_data: Dict[str, Dict[str, Any]],
    key_columns: Dict[str, str],
    match_threshold: float = 0.7,
    use_active_learning: bool = False,
    active_learning_config: Optional[Dict[str, Any]] = None,
    oracle_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    여러 테이블에서 엔티티 해결

    Args:
        tables_data: {table_name: {"columns": [...], "data": [...]}}
        key_columns: {table_name: "key_column_name"}
        match_threshold: 매칭 임계값 (threshold 모드에서만 사용)
        use_active_learning: Active Learning 사용 여부
        active_learning_config: AL 설정 {
            "n_committee": 5,        # Query-by-Committee 위원 수
            "min_labels": 10,        # 최소 라벨 수
            "query_strategy": "qbc", # qbc | uncertainty | random
        }
        oracle_callback: (entity_a, entity_b) -> bool 오라클 함수

    Returns:
        {
            "matches": [...],
            "clusters": [...],
            "cross_table_mappings": {...},
            "statistics": {...},
            "active_learning_state": {...}  # AL 모드에서만
        }
    """
    resolver = EntityResolver(
        match_threshold=match_threshold,
        use_active_learning=use_active_learning,
        active_learning_config=active_learning_config,
    )

    for table_name, table_info in tables_data.items():
        key_col = key_columns.get(table_name)
        if not key_col:
            continue

        columns = table_info.get("columns", [])
        data = table_info.get("data", [])

        # 데이터를 딕셔너리로 변환
        records = []
        for row in data:
            if isinstance(row, dict):
                records.append(row)
            elif isinstance(row, (list, tuple)) and columns:
                records.append(dict(zip(columns, row)))

        if records:
            resolver.add_entities(
                records=records,
                source_table=table_name,
                key_column=key_col,
            )

    matches = resolver.resolve(oracle_callback=oracle_callback)
    clusters = resolver.cluster_entities()
    mappings = resolver.get_cross_table_mappings()

    result = {
        "matches": [m.to_dict() for m in matches],
        "clusters": [c.to_dict() for c in clusters],
        "cross_table_mappings": mappings,
        "statistics": {
            "total_entities": len(resolver.entities),
            "total_matches": len(matches),
            "total_clusters": len(clusters),
            "method": "active_learning" if use_active_learning else "threshold",
        }
    }

    # Active Learning 상태 추가
    if use_active_learning:
        al_state = resolver.get_active_learning_state()
        if al_state:
            result["active_learning_state"] = al_state

    return result
