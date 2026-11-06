"""
================================================================================
파일명: server/main.py
설명: FAISS Index + Multi-Query + Hybrid Search(FAISS+BM25) + RAG Fusion(RRF)
      기반의 고도화 RAG 파이프라인 FastAPI 백엔드 서버
연동: Google Apps Script (Code.gs / index.html) 및 Google Cloud Run 배포 호환
================================================================================
"""

import base64
import hashlib
import io
import json
import math
import os
import re
import time
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Google Generative AI (선택적 연동: API 키 미설정 시 로컬 결정론적 임베딩/확장 폴백 동작)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ==============================================================================
# 1. 실시간 스트리밍 및 상태/복잡도 추적 핸들러 (StreamHandler)
# ==============================================================================
class StreamHandler:
    """
    파이프라인의 각 단계별 진행 상황(Progress), 소요 시간(ms), 동작 상태(Status),
    그리고 알고리즘 시간/공간 복잡도(Complexity)를 실시간 기록 및 스트리밍하는 핸들러입니다.
    - SSE(Server-Sent Events) 스트리밍과 GAS(UrlFetchApp)용 JSON 일괄 반환을 모두 지원합니다.
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.start_time = time.perf_counter()
        self.events: List[Dict[str, Any]] = []

    def emit(
        self,
        step: str,
        message: str,
        status: str = "RUNNING",
        progress: int = 0,
        time_complexity: str = "O(1)",
        space_complexity: str = "O(1)",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """단일 파이프라인 단계 이벤트를 기록하고 직렬화 가능한 딕셔너리로 반환합니다."""
        elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        event = {
            "pipeline": self.pipeline_name,
            "step": step,
            "message": message,
            "status": status,  # RUNNING | SUCCESS | WARNING | ERROR
            "progress": max(0, min(100, progress)),
            "elapsed_ms": elapsed_ms,
            "complexity": {
                "time": time_complexity,
                "space": space_complexity,
            },
            "details": details or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.events.append(event)
        return event

    def to_sse(self, event: Dict[str, Any]) -> str:
        """이벤트를 SSE(Server-Sent Events) 포맷 문자열로 변환합니다."""
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def get_summary(self) -> Dict[str, Any]:
        """전체 파이프라인 실행 요약 및 총 소요 시간을 반환합니다."""
        total_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        return {
            "pipeline": self.pipeline_name,
            "total_elapsed_ms": total_ms,
            "step_count": len(self.events),
            "events": self.events,
        }


# ==============================================================================
# 2. 텍스트 전처리, 토크나이저 및 임베딩 유틸리티
# ==============================================================================
def tokenize_ko_en(text: str) -> List[str]:
    """
    한국어 및 영어 하이브리드 BM25 검색을 위한 토크나이저입니다.
    - 어절 단위 토큰과 2-gram(바이그램) 서브워드를 함께 추출하여 한국어 조사 결합 시에도
      높은 키워드 재현율(Recall)을 보장합니다.
    - 시간 복잡도: O(L), L은 문자열 길이
    """
    if not text:
        return []
    cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
    words = [w for w in cleaned.split() if len(w) >= 1]
    tokens: List[str] = list(words)
    # 한국어 복합명사 및 조사 분리를 위한 문자 2-gram 추가
    for word in words:
        if len(word) >= 2:
            for i in range(len(word) - 1):
                tokens.append(word[i : i + 2])
    return tokens


def compute_embedding(
    texts: List[str],
    dim: int = 768,
    api_key: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """
    문서 또는 쿼리 리스트를 고차원 Dense Vector(float32)로 임베딩합니다.
    1순위: Gemini Embedding API (models/text-embedding-004) 사용
    2순위(폴백): API 키가 없거나 오프라인일 때 결정론적 N-gram Feature Hashing 임베딩 수행
    - 반환값: (shape=(N, dim)의 L2 정규화된 float32 배열, 사용된 임베딩 엔진 이름)
    """
    effective_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if GENAI_AVAILABLE and effective_key:
        try:
            genai.configure(api_key=effective_key)
            vectors = []
            for t in texts:
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=t if t.strip() else "empty",
                )
                vec = np.array(res["embedding"], dtype=np.float32)
                # 차원 맞춤 (필요시 패딩 또는 슬라이싱)
                if len(vec) != dim:
                    resized = np.zeros(dim, dtype=np.float32)
                    copy_len = min(len(vec), dim)
                    resized[:copy_len] = vec[:copy_len]
                    vec = resized
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec)
            return np.vstack(vectors).astype(np.float32), "Gemini (text-embedding-004)"
        except Exception:
            # API 호출 실패 시 로컬 결정론적 Feature Hashing 임베딩으로 안전하게 폴백
            pass

    # 결정론적 Feature Hashing 기반 고차원 임베딩 (API 키 없이도 의미/어근 유사도 보존)
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = tokenize_ko_en(text)
        if not tokens:
            matrix[row_idx, 0] = 1.0
            continue
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx1 = int.from_bytes(digest[:4], "little") % dim
            idx2 = int.from_bytes(digest[4:8], "little") % dim
            sign = 1.0 if (digest[8] % 2 == 0) else -1.0
            weight = 1.5 if len(token) > 2 else 1.0
            matrix[row_idx, idx1] += sign * weight
            matrix[row_idx, idx2] += sign * (weight * 0.5)
        norm = np.linalg.norm(matrix[row_idx])
        if norm > 0:
            matrix[row_idx] /= norm
    return matrix, "Local Feature-Hashing Dense Vector (768d)"


# ==============================================================================
# 3. 다중 쿼리 확장 계층 (Multi-Query Layer)
# ==============================================================================
class MultiQueryExpander:
    """
    [다중 쿼리 확장 계층 (Multi-Query Layer)]
    사용자의 단일 입력 쿼리를 서로 다른 관점(정의/원리, 핵심 키워드/동의어, 실무/적용 관점)의
    파생 쿼리들로 확장하여 단일 쿼리 검색의 어휘 불일치(Vocabulary Mismatch) 및 누락을 방지합니다.
    - 시간 복잡도: O(M * L_tokens), M은 생성 쿼리 수
    """

    @staticmethod
    def expand_queries(
        original_query: str,
        num_queries: int = 3,
        api_key: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        원본 쿼리를 포함하여 총 (1 + num_queries)개의 다중 관점 쿼리를 생성합니다.
        반환 형식: [{"perspective": 관점명, "query": 확장된 쿼리}, ...], 생성 엔진 설명
        """
        base_query = original_query.strip()
        results: List[Dict[str, str]] = [
            {"perspective": "원본 질의 (Original)", "query": base_query}
        ]

        effective_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if GENAI_AVAILABLE and effective_key:
            try:
                genai.configure(api_key=effective_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    "당신은 RAG 검색 품질을 극대화하는 Multi-Query 확장 전문가입니다.\n"
                    f"사용자 질문: '{base_query}'\n"
                    f"위 질문에 대해 검색 재현율(Recall)을 높일 수 있도록 서로 다른 관점의 파생 검색 쿼리 {num_queries}개를 생성하세요.\n"
                    "반드시 줄바꿈으로만 구분하여 쿼리 문장만 출력하세요 (번호나 불릿 기호 제외)."
                )
                response = model.generate_content(prompt)
                lines = [
                    line.strip().lstrip("0123456789.-*) ")
                    for line in (response.text or "").splitlines()
                    if line.strip()
                ]
                perspectives = [
                    "관점 1 (개념/정의 확장)",
                    "관점 2 (핵심 키워드/동의어)",
                    "관점 3 (구조/실무 적용)",
                    "관점 4 (심층 문맥 확장)",
                ]
                for i, q_text in enumerate(lines[:num_queries]):
                    if q_text and q_text != base_query:
                        results.append(
                            {
                                "perspective": perspectives[i % len(perspectives)],
                                "query": q_text,
                            }
                        )
                if len(results) > 1:
                    return results, "Gemini LLM Multi-Query Expander"
            except Exception:
                pass

        # LLM API 미사용 시 규칙 기반 다각도 쿼리 확장 (항상 안정적으로 동작)
        core_tokens = [
            w
            for w in re.sub(r"[^\w\s가-힣]", " ", base_query).split()
            if len(w) >= 2
            and w not in {"무엇", "어떻게", "설명", "알려줘", "해줘", "대해", "이란", "인가요"}
        ]
        keyword_str = " ".join(core_tokens) if core_tokens else base_query

        rule_expansions = [
            {
                "perspective": "관점 1 (핵심 개념 및 정의)",
                "query": f"{keyword_str} 핵심 개념 정의 원리 및 주요 특징",
            },
            {
                "perspective": "관점 2 (키워드 밀집 매칭)",
                "query": f"{keyword_str} 주요 구성 요소 아키텍처 장단점 비교",
            },
            {
                "perspective": "관점 3 (구체적 절차 및 활용)",
                "query": f"{keyword_str} 동작 과정 적용 사례 및 세부 내용 요약",
            },
        ]
        results.extend(rule_expansions[:num_queries])
        return results, "Heuristic Multi-Perspective Query Expander"


# ==============================================================================
# 4. 하이브리드 검색 & FAISS 인덱스 + RRF 순위 융합 계층
# ==============================================================================
class AdvancedRAGStore:
    """
    [하이브리드 검색 및 FAISS 인덱스 계층 + RRF 순위 융합 계층]
    1) FAISS Dense Vector Index (`IndexFlatL2` 또는 `IndexHNSWFlat`)
    2) BM25Okapi Sparse Keyword Index (`rank_bm25`)
    3) Reciprocal Rank Fusion (RRF) 재정렬 알고리즘
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim: int = embedding_dim
        self.index_type: str = "IndexFlatL2"  # "IndexFlatL2" | "IndexHNSWFlat"
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.source_files: List[str] = []
        self.last_updated: Optional[str] = None
        self.embedding_backend: str = "Not Initialized"

    def _create_faiss_index(self, index_type: str) -> faiss.Index:
        """
        요청된 인덱스 타입에 따라 FAISS 인덱스를 생성합니다.
        - IndexFlatL2: 정확한 전수 L2 거리 계산, 검색 복잡도 O(N * d)
        - IndexHNSWFlat: 계층적 그래프 기반 근사 최근접 이웃(ANN), 검색 복잡도 O(d * log N)
        """
        if index_type == "IndexHNSWFlat":
            # M=32(그래프 연결 수), efConstruction=64, efSearch=32 설정
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
            return index
        return faiss.IndexFlatL2(self.embedding_dim)

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        index_type: str = "IndexFlatL2",
        reset: bool = True,
        api_key: Optional[str] = None,
        stream: Optional[StreamHandler] = None,
    ) -> Dict[str, Any]:
        """
        문서 청크 리스트로부터 FAISS Dense 인덱스와 BM25 Sparse 인덱스를 동시에 구축합니다.
        """
        if not chunks:
            raise ValueError("인덱싱할 텍스트 청크가 비어 있습니다.")

        if reset or self.faiss_index is None or self.index_type != index_type:
            self.index_type = index_type
            self.faiss_index = self._create_faiss_index(index_type)
            self.documents = []
            self.tokenized_corpus = []
            self.source_files = []

        if stream:
            stream.emit(
                step="EMBEDDING_GENERATION",
                message=f"총 {len(chunks)}개 문서 청크에 대한 {self.embedding_dim}차원 Dense 임베딩 생성 중...",
                status="RUNNING",
                progress=45,
                time_complexity="O(N * d)",
                space_complexity="O(N * d)",
                details={"chunk_count": len(chunks), "dimension": self.embedding_dim},
            )

        texts = [c["text"] for c in chunks]
        vectors, emb_backend = compute_embedding(
            texts, dim=self.embedding_dim, api_key=api_key
        )
        self.embedding_backend = emb_backend

        # 1) FAISS 인덱스에 Dense 벡터 적재
        self.faiss_index.add(vectors)

        if stream:
            faiss_build_tc = (
                "O(N * d)" if index_type == "IndexFlatL2" else "O(N * log(N) * d)"
            )
            stream.emit(
                step="FAISS_INDEX_BUILD",
                message=f"FAISS [{index_type}] 인덱스에 {len(chunks)}개 벡터 적재 완료 (총 {self.faiss_index.ntotal}개)",
                status="RUNNING",
                progress=70,
                time_complexity=faiss_build_tc,
                space_complexity="O(N * d)",
                details={
                    "index_type": index_type,
                    "ntotal": int(self.faiss_index.ntotal),
                    "embedding_backend": emb_backend,
                },
            )

        # 2) 문서 메타데이터 저장 및 BM25 역색인(Inverted Index) 구축
        start_id = len(self.documents)
        for idx, chunk in enumerate(chunks):
            doc_id = start_id + idx
            doc_record = {
                "doc_id": doc_id,
                "text": chunk["text"],
                "source": chunk.get("source", "uploaded_document.pdf"),
                "page": chunk.get("page", 1),
                "chunk_index": chunk.get("chunk_index", idx),
                "char_length": len(chunk["text"]),
            }
            self.documents.append(doc_record)
            self.tokenized_corpus.append(tokenize_ko_en(chunk["text"]))
            if doc_record["source"] not in self.source_files:
                self.source_files.append(doc_record["source"])

        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

        if stream:
            stream.emit(
                step="BM25_SPARSE_INDEX_BUILD",
                message=f"BM25Okapi 키워드 역색인 구축 완료 (고유 문서 {len(self.documents)}개)",
                status="SUCCESS",
                progress=100,
                time_complexity="O(N * L_tokens)",
                space_complexity="O(V + N * L_unique)",
                details={
                    "total_documents": len(self.documents),
                    "source_files": self.source_files,
                },
            )

        return {
            "index_type": self.index_type,
            "total_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "embedding_backend": self.embedding_backend,
            "source_files": self.source_files,
            "last_updated": self.last_updated,
        }

    def hybrid_search_with_rrf(
        self,
        expanded_queries: List[Dict[str, str]],
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.55,
        sparse_weight: float = 0.45,
        api_key: Optional[str] = None,
        stream: Optional[StreamHandler] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        [Hybrid Search + RAG Fusion (Reciprocal Rank Fusion) 핵심 파이프라인]
        - 입력된 다중 쿼리(M개) 각각에 대하여:
          1) FAISS Dense Vector Search 수행 -> 의미적 유사도 순위 산출
          2) BM25 Sparse Keyword Search 수행 -> 정확한 키워드 매칭 순위 산출
        - 수집된 2 * M개의 순위 리스트를 RRF 공식으로 통합하여 최종 상위 Top-K 문서를 반환합니다.
          공식: RRF_Score(d) = Σ_q [ w_dense / (k + rank_dense(q, d)) + w_sparse / (k + rank_sparse(q, d)) ]
        """
        if (
            self.faiss_index is None
            or self.bm25_index is None
            or len(self.documents) == 0
        ):
            raise RuntimeError(
                "벡터 DB가 아직 생성되지 않았습니다. 먼저 PDF 파일을 선택하고 [벡터 DB 생성]을 실행해 주세요."
            )

        num_docs = len(self.documents)
        candidate_k = min(num_docs, max(top_k * 3, 10))
        query_texts = [q["query"] for q in expanded_queries]

        # 1. 확장된 모든 쿼리 벡터 임베딩
        query_vectors, _ = compute_embedding(
            query_texts, dim=self.embedding_dim, api_key=api_key
        )

        # 2. FAISS Dense Search 일괄 수행
        distances_matrix, indices_matrix = self.faiss_index.search(
            query_vectors, candidate_k
        )

        if stream:
            dense_tc = (
                "O(M * N * d)"
                if self.index_type == "IndexFlatL2"
                else "O(M * d * log N)"
            )
            stream.emit(
                step="HYBRID_SEARCH_EXECUTION",
                message=f"{len(expanded_queries)}개 확장 쿼리 × [FAISS({self.index_type}) + BM25] 하이브리드 검색 완료",
                status="RUNNING",
                progress=60,
                time_complexity=f"Dense: {dense_tc} | Sparse(BM25): O(M * |Q| * N)",
                space_complexity="O(M * K)",
                details={
                    "num_expanded_queries": len(expanded_queries),
                    "candidate_pool_per_query": candidate_k,
                },
            )

        # 3. 문서별 RRF 점수 및 상세 검색 메타데이터 누적 구조체
        fusion_map: Dict[int, Dict[str, Any]] = {}

        for q_idx, q_info in enumerate(expanded_queries):
            q_str = q_info["query"]
            q_perspective = q_info["perspective"]

            # --- (A) FAISS Dense 순위 반영 ---
            for rank_zero, doc_id in enumerate(indices_matrix[q_idx]):
                if doc_id < 0 or doc_id >= num_docs:
                    continue
                rank = rank_zero + 1  # 1-based rank
                l2_dist = float(distances_matrix[q_idx][rank_zero])
                # L2 거리를 0~1 범위의 직관적 코사인/유사도 점수로 변환: sim = 1 / (1 + L2)
                dense_sim = round(1.0 / (1.0 + max(0.0, l2_dist)), 4)
                rrf_contrib = dense_weight * (1.0 / (rrf_k + rank))

                entry = fusion_map.setdefault(
                    int(doc_id),
                    {
                        "doc_id": int(doc_id),
                        "rrf_score": 0.0,
                        "dense_rrf_score": 0.0,
                        "sparse_rrf_score": 0.0,
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "best_dense_similarity": 0.0,
                        "best_l2_distance": None,
                        "best_bm25_score": 0.0,
                        "matched_queries": [],
                        "rank_breakdown": [],
                    },
                )
                entry["rrf_score"] += rrf_contrib
                entry["dense_rrf_score"] += rrf_contrib
                if entry["best_dense_rank"] is None or rank < entry["best_dense_rank"]:
                    entry["best_dense_rank"] = rank
                    entry["best_l2_distance"] = round(l2_dist, 4)
                    entry["best_dense_similarity"] = dense_sim
                if q_perspective not in entry["matched_queries"]:
                    entry["matched_queries"].append(q_perspective)
                entry["rank_breakdown"].append(
                    {
                        "query_perspective": q_perspective,
                        "channel": "FAISS_DENSE",
                        "rank": rank,
                        "metric": f"L2={l2_dist:.4f} (Sim={dense_sim:.3f})",
                        "rrf_delta": round(rrf_contrib, 6),
                    }
                )

            # --- (B) BM25 Sparse 키워드 순위 반영 ---
            q_tokens = tokenize_ko_en(q_str)
            bm25_scores = self.bm25_index.get_scores(q_tokens)
            ranked_bm25_ids = np.argsort(bm25_scores)[::-1][:candidate_k]

            for rank_zero, doc_id in enumerate(ranked_bm25_ids):
                bm25_val = float(bm25_scores[doc_id])
                # 점수가 0 이하더라도 후보군이 부족할 경우 상위 일부 반영
                if bm25_val <= 0.0 and rank_zero >= top_k:
                    continue
                rank = rank_zero + 1
                rrf_contrib = sparse_weight * (1.0 / (rrf_k + rank))

                entry = fusion_map.setdefault(
                    int(doc_id),
                    {
                        "doc_id": int(doc_id),
                        "rrf_score": 0.0,
                        "dense_rrf_score": 0.0,
                        "sparse_rrf_score": 0.0,
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "best_dense_similarity": 0.0,
                        "best_l2_distance": None,
                        "best_bm25_score": 0.0,
                        "matched_queries": [],
                        "rank_breakdown": [],
                    },
                )
                entry["rrf_score"] += rrf_contrib
                entry["sparse_rrf_score"] += rrf_contrib
                if (
                    entry["best_sparse_rank"] is None
                    or rank < entry["best_sparse_rank"]
                ):
                    entry["best_sparse_rank"] = rank
                if bm25_val > entry["best_bm25_score"]:
                    entry["best_bm25_score"] = round(bm25_val, 4)
                if q_perspective not in entry["matched_queries"]:
                    entry["matched_queries"].append(q_perspective)
                entry["rank_breakdown"].append(
                    {
                        "query_perspective": q_perspective,
                        "channel": "BM25_SPARSE",
                        "rank": rank,
                        "metric": f"BM25={bm25_val:.4f}",
                        "rrf_delta": round(rrf_contrib, 6),
                    }
                )

        # 4. RRF 합산 점수 기준 내림차순 재정렬 (Re-ranking)
        sorted_entries = sorted(
            fusion_map.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:top_k]

        results: List[Dict[str, Any]] = []
        for final_rank, item in enumerate(sorted_entries, start=1):
            doc_meta = self.documents[item["doc_id"]]
            results.append(
                {
                    "final_rank": final_rank,
                    "doc_id": item["doc_id"],
                    "text": doc_meta["text"],
                    "source": doc_meta["source"],
                    "page": doc_meta["page"],
                    "chunk_index": doc_meta["chunk_index"],
                    "rrf_score": round(item["rrf_score"], 6),
                    "dense_rrf_score": round(item["dense_rrf_score"], 6),
                    "sparse_rrf_score": round(item["sparse_rrf_score"], 6),
                    "best_dense_rank": item["best_dense_rank"],
                    "best_sparse_rank": item["best_sparse_rank"],
                    "best_dense_similarity": item["best_dense_similarity"],
                    "best_l2_distance": item["best_l2_distance"],
                    "best_bm25_score": item["best_bm25_score"],
                    "matched_queries": item["matched_queries"],
                    "rank_breakdown": item["rank_breakdown"][:6],
                    "citation": f"[{doc_meta['source']} - p.{doc_meta['page']} (Chunk #{doc_meta['chunk_index']})]",
                }
            )

        if stream:
            stream.emit(
                step="RRF_RANK_FUSION",
                message=f"RRF(k={rrf_k}) 상호 순위 융합 완료: 후보 {len(fusion_map)}개 중 상위 {len(results)}개 문서 재정렬",
                status="RUNNING",
                progress=80,
                time_complexity="O(M * K * log(M * K))",
                space_complexity="O(M * K)",
                details={
                    "rrf_constant_k": rrf_k,
                    "total_unique_candidates": len(fusion_map),
                    "selected_top_k": len(results),
                },
            )

        fusion_stats = {
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "total_candidates_evaluated": len(fusion_map),
            "returned_top_k": len(results),
        }
        return results, fusion_stats


# 전역 싱글톤 RAG 스토어 인스턴스
rag_store = AdvancedRAGStore(embedding_dim=768)


# ==============================================================================
# 5. PDF 텍스트 추출 및 슬라이딩 윈도우 청킹 유틸리티
# ==============================================================================
def extract_and_chunk_pdf(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    PDF 바이트 스트림에서 페이지별로 텍스트를 추출하고 슬라이딩 윈도우 방식으로 청킹합니다.
    - 시간 복잡도: O(P * C), P는 페이지 수, C는 페이지당 문자 수
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    num_pages = len(reader.pages)
    chunks: List[Dict[str, Any]] = []
    global_chunk_idx = 1

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
        if not cleaned_text:
            continue

        start = 0
        text_len = len(cleaned_text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            snippet = cleaned_text[start:end].strip()
            if len(snippet) >= 20:
                chunks.append(
                    {
                        "text": snippet,
                        "source": filename,
                        "page": page_num,
                        "chunk_index": global_chunk_idx,
                    }
                )
                global_chunk_idx += 1
            if end >= text_len:
                break
            start += max(1, chunk_size - chunk_overlap)

    return chunks, num_pages


def synthesize_rag_answer(
    query: str,
    expanded_queries: List[Dict[str, str]],
    ranked_docs: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    RRF로 재정렬된 상위 문서 스니펫들을 컨텍스트로 활용하여 출처가 포함된 최종 답변을 생성합니다.
    - Gemini API 키가 설정되어 있으면 LLM 생성 답변을 반환하고,
      없을 경우 RRF 상위 문서 기반의 구조화된 추출형(Extractive) 종합 답변을 생성합니다.
    """
    if not ranked_docs:
        return (
            "검색된 관련 문서가 없습니다. 다른 키워드로 질문하거나 PDF 문서를 다시 인덱싱해 주세요.",
            "No Context",
        )

    context_blocks = []
    for doc in ranked_docs:
        context_blocks.append(
            f"[문서 #{doc['final_rank']} | 출처: {doc['source']} p.{doc['page']} (Chunk #{doc['chunk_index']}) | RRF 점수: {doc['rrf_score']}]\n"
            f"{doc['text']}"
        )
    joined_context = "\n\n".join(context_blocks)

    effective_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if GENAI_AVAILABLE and effective_key:
        try:
            genai.configure(api_key=effective_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "당신은 신뢰할 수 있는 AI 어시스턴트입니다. 아래 [RRF 상위 검색 문서]만을 근거로 사용자의 질문에 한국어로 명확하고 체계적으로 답변하세요.\n"
                "반드시 답변 본문 중에 참고한 문서의 출처를 `[출처: 파일명 p.페이지]` 형태로 명시하세요.\n\n"
                f"[사용자 질문]\n{query}\n\n"
                f"[RRF 상위 검색 문서]\n{joined_context}\n"
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip(), "Gemini-1.5-Flash (RAG Synthesis)"
        except Exception:
            pass

    # API 키 미설정 시 RRF 상위 문서를 종합하여 출처가 명시된 고품질 요약 답변 구성
    lines = [
        f"### 질문에 대한 RAG Fusion 종합 답변",
        f"**질의 내용**: \"{query}\" (다중 쿼리 {len(expanded_queries)}개 관점 확장 및 Hybrid RRF 재정렬 결과)\n",
        "#### 핵심 내용 요약",
    ]
    for doc in ranked_docs[:3]:
        snippet = doc["text"]
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "..."
        lines.append(
            f"- **[순위 #{doc['final_rank']} | RRF `{doc['rrf_score']:.5f}`]** {snippet} "
            f"**{doc['citation']}**"
        )

    lines.append("\n#### 참고 출처 (Sources)")
    seen_sources = []
    for doc in ranked_docs:
        cite = f"{doc['source']} (p.{doc['page']}, Chunk #{doc['chunk_index']})"
        if cite not in seen_sources:
            seen_sources.append(cite)
            lines.append(
                f"- 📄 **{cite}** — Dense 순위: `{doc['best_dense_rank'] or '-'}`, "
                f"BM25 순위: `{doc['best_sparse_rank'] or '-'}`, RRF 점수: `{doc['rrf_score']:.5f}`"
            )

    return "\n".join(lines), "Extractive RRF Synthesizer (Local Mode)"


# ==============================================================================
# 6. FastAPI 애플리케이션 및 요청/응답 스키마 정의
# ==============================================================================
app = FastAPI(
    title="Advanced RAG Pipeline API (FAISS + Multi-Query + Hybrid Search + RRF)",
    description="FAISS Dense Vector + BM25 Sparse + Multi-Query + Reciprocal Rank Fusion 백엔드",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    filename: str = Field(default="sample_document.pdf", description="원본 PDF 파일명")
    pdf_base64: Optional[str] = Field(
        default=None, description="Base64 인코딩된 PDF 파일 바이너리"
    )
    raw_text: Optional[str] = Field(
        default=None, description="PDF 대신 직접 전달할 텍스트(테스트/데모용)"
    )
    index_type: str = Field(
        default="IndexFlatL2",
        description="FAISS 인덱스 종류 (IndexFlatL2 또는 IndexHNSWFlat)",
    )
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)
    reset_index: bool = Field(default=True, description="기존 인덱스 초기화 여부")
    api_key: Optional[str] = Field(default=None, description="선택적 Gemini API Key")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="사용자 검색 질의")
    top_k: int = Field(default=4, ge=1, le=20, description="반환할 RRF 상위 문서 수")
    num_expanded_queries: int = Field(
        default=3, ge=1, le=5, description="Multi-Query 파생 쿼리 생성 개수"
    )
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF 상수 k (기본값 60)")
    dense_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    api_key: Optional[str] = Field(default=None, description="선택적 Gemini API Key")


# ==============================================================================
# 7. API 엔드포인트 구현
# ==============================================================================
@app.get("/")
@app.get("/health")
@app.get("/status")
async def get_server_status():
    """
    서버 연동 상태, 현재 적재된 FAISS/BM25 인덱스 현황 및 알고리즘 복잡도 정보를 반환합니다.
    """
    return {
        "status": "ONLINE",
        "server_name": "FastAPI Advanced RAG Server (FAISS + Hybrid + RRF)",
        "index_ready": len(rag_store.documents) > 0,
        "index_type": rag_store.index_type,
        "total_chunks": len(rag_store.documents),
        "embedding_dim": rag_store.embedding_dim,
        "embedding_backend": rag_store.embedding_backend,
        "source_files": rag_store.source_files,
        "last_updated": rag_store.last_updated,
        "algorithm_complexity": {
            "multi_query": {"time": "O(M * L_tokens)", "space": "O(M)"},
            "faiss_flat_l2": {"time": "O(M * N * d)", "space": "O(N * d)"},
            "faiss_hnsw": {"time": "O(M * d * log N)", "space": "O(N * (d + M_graph))"},
            "bm25_sparse": {"time": "O(M * |Q| * N_df)", "space": "O(V + N * L)"},
            "rrf_fusion": {"time": "O(M * K * log(M * K))", "space": "O(M * K)"},
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/ingest")
async def ingest_document(req: IngestRequest):
    """
    [벡터 DB 생성 엔드포인트]
    PDF(Base64) 또는 텍스트를 받아 청킹 후 FAISS(Dense) + BM25(Sparse) 인덱스를 구축하고,
    StreamHandler의 단계별 진행 로그와 알고리즘 복잡도를 함께 반환합니다.
    """
    stream = StreamHandler(pipeline_name="VECTOR_DB_INGESTION")
    try:
        stream.emit(
            step="REQUEST_RECEIVED",
            message=f"'{req.filename}' 파일 수신 완료. 텍스트 추출 및 청킹을 시작합니다.",
            status="RUNNING",
            progress=15,
            time_complexity="O(B)",
            space_complexity="O(B)",
            details={"filename": req.filename, "index_type": req.index_type},
        )

        chunks: List[Dict[str, Any]] = []
        num_pages = 1

        if req.pdf_base64:
            # Data URL prefix 제거 처리 (예: data:application/pdf;base64,...)
            b64_clean = (
                req.pdf_base64.split(",")[-1]
                if "," in req.pdf_base64
                else req.pdf_base64
            )
            pdf_bytes = base64.b64decode(b64_clean)
            chunks, num_pages = extract_and_chunk_pdf(
                pdf_bytes=pdf_bytes,
                filename=req.filename,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
        elif req.raw_text and req.raw_text.strip():
            cleaned = re.sub(r"\s+", " ", req.raw_text).strip()
            start = 0
            idx = 1
            while start < len(cleaned):
                end = min(start + req.chunk_size, len(cleaned))
                chunks.append(
                    {
                        "text": cleaned[start:end],
                        "source": req.filename,
                        "page": 1,
                        "chunk_index": idx,
                    }
                )
                idx += 1
                if end >= len(cleaned):
                    break
                start += max(1, req.chunk_size - req.chunk_overlap)
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "EMPTY_PAYLOAD",
                    "message": "PDF 데이터(pdf_base64) 또는 텍스트(raw_text)가 제공되지 않았습니다.",
                },
            )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "NO_TEXT_EXTRACTED",
                    "message": f"'{req.filename}'에서 추출 가능한 텍스트가 없습니다. 스캔된 이미지 PDF인지 확인해 주세요.",
                },
            )

        stream.emit(
            step="PDF_CHUNKING_COMPLETE",
            message=f"총 {num_pages}페이지에서 {len(chunks)}개의 문맥 청크 분할 완료 (크기={req.chunk_size}, 중첩={req.chunk_overlap})",
            status="RUNNING",
            progress=30,
            time_complexity="O(P * C)",
            space_complexity="O(N * chunk_size)",
            details={"pages": num_pages, "chunks": len(chunks)},
        )

        build_info = rag_store.build_index(
            chunks=chunks,
            index_type=req.index_type,
            reset=req.reset_index,
            api_key=req.api_key,
            stream=stream,
        )

        return {
            "status": "SUCCESS",
            "message": f"'{req.filename}' 기반 하이브리드 벡터 DB(FAISS + BM25) 구축이 완료되었습니다.",
            "file_info": {
                "filename": req.filename,
                "pages": num_pages,
                "chunks_created": len(chunks),
            },
            "index_info": build_info,
            "complexity_metrics": {
                "chunking": "O(P * C)",
                "embedding": "O(N * d)",
                "faiss_index": (
                    "O(N * d)"
                    if req.index_type == "IndexFlatL2"
                    else "O(N * log(N) * d)"
                ),
                "bm25_index": "O(N * L_tokens)",
                "parameters": f"N={len(chunks)}, d={rag_store.embedding_dim}, P={num_pages}",
            },
            "stream_logs": stream.get_summary(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        stream.emit(
            step="INGEST_ERROR",
            message=f"벡터 DB 생성 중 오류 발생: {str(exc)}",
            status="ERROR",
            progress=100,
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error_code": "INGESTION_FAILED",
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "stream_logs": stream.get_summary(),
            },
        )


@app.post("/query")
async def query_rag_pipeline(req: QueryRequest):
    """
    [Multi-Query + Hybrid Search (FAISS + BM25) + RAG Fusion (RRF) 질의 엔드포인트]
    1단계: 단일 사용자 질의를 다중 관점 쿼리로 확장 (Multi-Query Layer)
    2단계: 각 쿼리별 FAISS Dense 검색 + BM25 Sparse 키워드 검색 병행 (Hybrid Search Layer)
    3단계: Reciprocal Rank Fusion(RRF)으로 순위 통합 및 재정렬 (RAG Fusion Layer)
    4단계: RRF 상위 문서와 출처를 기반으로 LLM 최종 답변 합성
    """
    stream = StreamHandler(pipeline_name="MULTI_QUERY_HYBRID_RRF_SEARCH")
    try:
        stream.emit(
            step="QUERY_RECEIVED",
            message=f"질의 수신: '{req.query}'",
            status="RUNNING",
            progress=10,
            time_complexity="O(1)",
            space_complexity="O(1)",
        )

        # 1. 다중 쿼리 확장 계층 (Multi-Query Layer)
        expanded_queries, expander_backend = MultiQueryExpander.expand_queries(
            original_query=req.query,
            num_queries=req.num_expanded_queries,
            api_key=req.api_key,
        )
        stream.emit(
            step="MULTI_QUERY_EXPANSION",
            message=f"단일 질의를 {len(expanded_queries)}개의 다중 관점 쿼리로 확장 완료 ({expander_backend})",
            status="RUNNING",
            progress=30,
            time_complexity="O(M * L_tokens)",
            space_complexity="O(M)",
            details={
                "expander_backend": expander_backend,
                "expanded_queries": expanded_queries,
            },
        )

        # 2 & 3. 하이브리드 검색(FAISS + BM25) 및 RRF 순위 융합 계층
        ranked_docs, fusion_stats = rag_store.hybrid_search_with_rrf(
            expanded_queries=expanded_queries,
            top_k=req.top_k,
            rrf_k=req.rrf_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
            api_key=req.api_key,
            stream=stream,
        )

        # 4. LLM 답변 합성 및 출처 표기
        llm_answer, answer_backend = synthesize_rag_answer(
            query=req.query,
            expanded_queries=expanded_queries,
            ranked_docs=ranked_docs,
            api_key=req.api_key,
        )
        stream.emit(
            step="ANSWER_SYNTHESIS_COMPLETE",
            message=f"RRF 상위 {len(ranked_docs)}개 문서 기반 답변 생성 및 출처 매핑 완료 ({answer_backend})",
            status="SUCCESS",
            progress=100,
            time_complexity="O(K * L_context)",
            space_complexity="O(L_answer)",
            details={"answer_backend": answer_backend},
        )

        n_docs = len(rag_store.documents)
        m_queries = len(expanded_queries)
        return {
            "status": "SUCCESS",
            "query": req.query,
            "expanded_queries": expanded_queries,
            "rag_results": ranked_docs,
            "llm_answer": llm_answer,
            "sources": [
                {
                    "rank": d["final_rank"],
                    "source": d["source"],
                    "page": d["page"],
                    "chunk_index": d["chunk_index"],
                    "rrf_score": d["rrf_score"],
                    "citation": d["citation"],
                }
                for d in ranked_docs
            ],
            "fusion_metadata": {
                **fusion_stats,
                "index_type": rag_store.index_type,
                "expander_backend": expander_backend,
                "answer_backend": answer_backend,
            },
            "complexity_metrics": {
                "multi_query_layer": f"O(M * L_tokens) [M={m_queries}]",
                "hybrid_faiss_dense": (
                    f"O(M * N * d) [M={m_queries}, N={n_docs}, d={rag_store.embedding_dim}]"
                    if rag_store.index_type == "IndexFlatL2"
                    else f"O(M * d * log N) [M={m_queries}, N={n_docs}, d={rag_store.embedding_dim}]"
                ),
                "hybrid_bm25_sparse": f"O(M * |Q| * N_df) [M={m_queries}, N={n_docs}]",
                "rrf_fusion_layer": f"O(M * K * log(M * K)) [k={req.rrf_k}, Top-K={req.top_k}]",
                "overall_space": f"O(N * d + M * K) [총 문서 청크 N={n_docs}]",
            },
            "stream_logs": stream.get_summary(),
        }

    except RuntimeError as r_err:
        stream.emit(
            step="INDEX_NOT_READY",
            message=str(r_err),
            status="ERROR",
            progress=100,
        )
        return JSONResponse(
            status_code=400,
            content={
                "status": "ERROR",
                "error_code": "INDEX_NOT_INITIALIZED",
                "message": str(r_err),
                "stream_logs": stream.get_summary(),
            },
        )
    except Exception as exc:
        stream.emit(
            step="QUERY_ERROR",
            message=f"질의 처리 중 서버 오류 발생: {str(exc)}",
            status="ERROR",
            progress=100,
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error_code": "QUERY_PIPELINE_FAILED",
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "stream_logs": stream.get_summary(),
            },
        )


@app.post("/query/stream")
async def query_rag_pipeline_sse(req: QueryRequest):
    """
    실시간 SSE(Server-Sent Events) 스트리밍 방식으로 각 계층(Multi-Query -> Hybrid -> RRF -> Answer)의
    진행 상태와 최종 결과를 순차 전송하는 엔드포인트입니다.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        stream = StreamHandler(pipeline_name="SSE_RAG_FUSION_STREAM")
        try:
            ev1 = stream.emit(
                step="QUERY_RECEIVED",
                message=f"실시간 스트리밍 질의 시작: '{req.query}'",
                status="RUNNING",
                progress=10,
                time_complexity="O(1)",
            )
            yield stream.to_sse(ev1)

            expanded_queries, expander_backend = MultiQueryExpander.expand_queries(
                original_query=req.query,
                num_queries=req.num_expanded_queries,
                api_key=req.api_key,
            )
            ev2 = stream.emit(
                step="MULTI_QUERY_EXPANSION",
                message=f"다중 쿼리 {len(expanded_queries)}개 확장 완료",
                status="RUNNING",
                progress=35,
                time_complexity="O(M * L_tokens)",
                details={"expanded_queries": expanded_queries},
            )
            yield stream.to_sse(ev2)

            ranked_docs, fusion_stats = rag_store.hybrid_search_with_rrf(
                expanded_queries=expanded_queries,
                top_k=req.top_k,
                rrf_k=req.rrf_k,
                dense_weight=req.dense_weight,
                sparse_weight=req.sparse_weight,
                api_key=req.api_key,
                stream=stream,
            )
            for ev in stream.events[2:]:
                yield stream.to_sse(ev)

            llm_answer, answer_backend = synthesize_rag_answer(
                query=req.query,
                expanded_queries=expanded_queries,
                ranked_docs=ranked_docs,
                api_key=req.api_key,
            )
            final_ev = stream.emit(
                step="COMPLETE",
                message="RAG Fusion 검색 및 답변 생성 완료",
                status="SUCCESS",
                progress=100,
                time_complexity="O(M * K * log(M * K))",
                details={
                    "expanded_queries": expanded_queries,
                    "rag_results": ranked_docs,
                    "llm_answer": llm_answer,
                    "fusion_metadata": fusion_stats,
                    "answer_backend": answer_backend,
                },
            )
            yield stream.to_sse(final_ev)
        except Exception as exc:
            err_ev = stream.emit(
                step="ERROR",
                message=str(exc),
                status="ERROR",
                progress=100,
            )
            yield stream.to_sse(err_ev)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

