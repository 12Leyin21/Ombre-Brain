# ============================================================
# Script: One-time vector backfill (reindex_vectors.py)
# 脚本：为存量记忆桶补建向量索引
#
# Run once after enabling OMBRE_EMBEDDING_* env vars, to embed every
# existing bucket (including archived ones) that predates the vector
# index. New buckets are embedded automatically going forward.
# 在配置好 OMBRE_EMBEDDING_* 环境变量后跑一次，把所有已存在的记忆桶
# （含归档）补建向量索引。之后新建的桶会自动向量化，无需再跑。
#
# Usage: python reindex_vectors.py
# ============================================================

import asyncio

from utils import load_config
from bucket_manager import BucketManager
from vector_store import VectorStore


async def main():
    config = load_config()
    vector_store = VectorStore(config)

    if not vector_store.is_available():
        print("Embedding 未配置（OMBRE_EMBEDDING_API_KEY/BASE_URL/MODEL 缺一不可），无法建索引。")
        return

    bucket_mgr = BucketManager(config, vector_store=vector_store)
    buckets = await bucket_mgr.list_all(include_archive=True)
    print(f"共 {len(buckets)} 个记忆桶，开始建索引...")

    ok, fail = 0, 0
    for b in buckets:
        try:
            await vector_store.upsert(b["id"], b["content"])
            ok += 1
        except Exception as e:
            print(f"  失败 [{b['id']}]: {e}")
            fail += 1

    print(f"完成：成功 {ok} 个，失败 {fail} 个。")


if __name__ == "__main__":
    asyncio.run(main())
