# ============================================================
# Module: Vector Store (vector_store.py)
# 模块：向量索引存储
#
# Semantic embedding + cosine-similarity search for memory buckets,
# backed by local SQLite. Embedding provider is entirely env-var
# configured — no keys, URLs or model names are hardcoded here.
# When unconfigured or a call fails, callers degrade to keyword-only
# search (see BucketManager.search).
# 基于本地 SQLite 的记忆桶语义向量检索。Embedding 服务完全由环境变量
# 配置——密钥、地址、模型名都不写死在代码里。未配置或调用失败时，
# 调用方会自动降级为纯关键词检索（见 BucketManager.search）。
#
# Env vars / 环境变量：
#   OMBRE_EMBEDDING_API_KEY
#   OMBRE_EMBEDDING_BASE_URL   (any OpenAI-compatible embeddings endpoint)
#   OMBRE_EMBEDDING_MODEL
# ============================================================

import os
import sqlite3
import logging
from array import array
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.vector")


class VectorStore:
    """
    Stores one embedding per bucket in local SQLite and answers
    cosine-similarity queries against a candidate ID set.
    在本地 SQLite 中为每个记忆桶存一条向量，对给定候选桶集合做
    余弦相似度检索。
    """

    def __init__(self, config: dict):
        db_path = os.path.join(config["buckets_dir"], "vectors.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

        # --- Embedding provider config: env vars only, no defaults ---
        # --- Embedding 服务配置：只认环境变量，代码里没有默认值 ---
        self.api_key = os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
        self.base_url = os.environ.get("OMBRE_EMBEDDING_BASE_URL", "")
        self.model = os.environ.get("OMBRE_EMBEDDING_MODEL", "")

        self.available = bool(self.api_key and self.base_url and self.model)
        self.client = (
            AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)
            if self.available
            else None
        )
        if not self.available:
            logger.info(
                "Embedding 未配置（缺少 OMBRE_EMBEDDING_API_KEY/BASE_URL/MODEL），"
                "语义检索关闭，自动降级为纯关键词匹配。"
            )

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors ("
                "bucket_id TEXT PRIMARY KEY, "
                "embedding BLOB NOT NULL, "
                "updated_at TEXT NOT NULL)"
            )
            conn.commit()
        finally:
            conn.close()

    def is_available(self) -> bool:
        return self.available

    # ---------------------------------------------------------
    # Embed a piece of text via the configured API.
    # 调用配置好的 API 对一段文本做 embedding。
    # Returns None on any failure — callers must treat that as
    # "semantic channel unavailable this time", not an error.
    # 任何失败都返回 None——调用方应视为"这次没有语义结果"，而非报错。
    # ---------------------------------------------------------
    async def embed_text(self, text: str) -> Optional[list]:
        if not self.available or not text or not text.strip():
            return None
        try:
            resp = await self.client.embeddings.create(
                model=self.model, input=text[:8000]
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding 请求失败，本次跳过语义检索: {e}")
            return None

    # ---------------------------------------------------------
    # Create or refresh a bucket's stored embedding.
    # 创建或刷新某个桶的向量。
    # ---------------------------------------------------------
    async def upsert(self, bucket_id: str, text: str) -> None:
        if not self.available:
            return
        vec = await self.embed_text(text)
        if vec is None:
            return
        blob = array("f", vec).tobytes()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO vectors (bucket_id, embedding, updated_at) "
                "VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(bucket_id) DO UPDATE SET "
                "embedding=excluded.embedding, updated_at=excluded.updated_at",
                (bucket_id, blob),
            )
            conn.commit()
        finally:
            conn.close()

    def delete(self, bucket_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM vectors WHERE bucket_id = ?", (bucket_id,))
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------
    # Semantic search restricted to candidate_ids (already domain/
    # archive-filtered upstream by BucketManager.search).
    # 语义检索，只在 candidate_ids 范围内找（上游 BucketManager.search
    # 已经做过主题域/归档范围过滤）。
    # Returns {bucket_id: cosine_similarity clipped to 0~1}.
    # ---------------------------------------------------------
    async def search(self, query: str, candidate_ids: set) -> dict:
        if not self.available or not candidate_ids:
            return {}
        query_vec = await self.embed_text(query)
        if query_vec is None:
            return {}

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return {}

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ",".join("?" for _ in candidate_ids)
            rows = conn.execute(
                f"SELECT bucket_id, embedding FROM vectors WHERE bucket_id IN ({placeholders})",
                tuple(candidate_ids),
            ).fetchall()
        finally:
            conn.close()

        scores = {}
        for bucket_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            v_norm = np.linalg.norm(vec)
            if v_norm == 0:
                continue
            cosine = float(np.dot(q, vec) / (q_norm * v_norm))
            scores[bucket_id] = max(0.0, cosine)
        return scores
