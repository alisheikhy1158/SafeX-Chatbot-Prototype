"""
Retrieval layer over the SafeX knowledge base, using dense sentence
embeddings (sentence-transformers, all-MiniLM-L6-v2) + cosine similarity.

"""

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from knowledge_base import KNOWLEDGE_BASE

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  


@dataclass
class RetrievedChunk:
    id: str
    topic: str
    text: str
    score: float


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


class KnowledgeRetriever:
    def __init__(self, knowledge_base=None, model_name: str = EMBEDDING_MODEL_NAME):
        self.kb = knowledge_base or KNOWLEDGE_BASE
        self._corpus = [f"{item['topic']}. {item['text']}" for item in self.kb]

        self.model = SentenceTransformer(model_name)
        self._matrix = self.model.encode(
            self._corpus, normalize_embeddings=True, show_progress_bar=False
        )

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.25) -> List[RetrievedChunk]:
        query_vec = self.model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        scores = _cosine_similarity(query_vec, self._matrix)
        ranked_idx = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_idx:
            if scores[idx] >= min_score:
                item = self.kb[idx]
                results.append(
                    RetrievedChunk(
                        id=item["id"],
                        topic=item["topic"],
                        text=item["text"],
                        score=float(scores[idx]),
                    )
                )
        return results
