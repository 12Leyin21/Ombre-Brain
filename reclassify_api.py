#!/usr/bin/env python3
"""
用 API 重新打标未分类记忆桶，修正 domain/tags/name，移动到正确目录。
复用 server.py 同一套配置和 Dehydrator，保证跟实际部署配置一致。
用法: python3 reclassify_api.py
"""
import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import frontmatter

from utils import load_config
from dehydrator import Dehydrator


def sanitize(name):
    name = re.sub(r'[<>:"/\\|?*\n\r]', '', name).strip()
    return name[:20] if name else "未命名"


async def reclassify():
    config = load_config()
    dehydrator = Dehydrator(config)

    data_dir = config["buckets_dir"]
    unclass_dir = os.path.join(data_dir, "未分类")

    import glob
    files = sorted(glob.glob(os.path.join(unclass_dir, "*.md")))
    print(f"未分类目录: {unclass_dir}")
    print(f"找到 {len(files)} 个未分类文件\n")

    for fpath in files:
        basename = os.path.basename(fpath)
        post = frontmatter.load(fpath)
        content = post.content.strip()
        name = post.metadata.get("name", "")

        try:
            result = await dehydrator.analyze(content)
        except Exception as e:
            print(f"  X API失败 {basename}: {e}")
            continue

        new_domain = result.get("domain", ["未分类"])[:3]
        new_tags = result.get("tags", [])[:5]
        new_name = sanitize(result.get("suggested_name", "") or name)
        new_valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
        new_arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))

        post.metadata["domain"] = new_domain
        post.metadata["tags"] = new_tags
        post.metadata["valence"] = new_valence
        post.metadata["arousal"] = new_arousal
        if new_name:
            post.metadata["name"] = new_name

        # 写回文件
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        # 移动到正确目录
        primary = sanitize(new_domain[0]) if new_domain else "未分类"
        target_dir = os.path.join(data_dir, primary)
        os.makedirs(target_dir, exist_ok=True)

        bid = post.metadata.get("id", "")
        new_filename = f"{new_name}_{bid}.md" if new_name and new_name != bid else basename
        dest = os.path.join(target_dir, new_filename)

        if dest != fpath:
            os.rename(fpath, dest)

        print(f"  OK {basename}")
        print(f"     -> {primary}/{new_filename}")
        print(f"     domain={new_domain} tags={new_tags} V={new_valence} A={new_arousal}")
        print()


if __name__ == "__main__":
    asyncio.run(reclassify())
