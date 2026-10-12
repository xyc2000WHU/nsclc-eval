# -*- coding: utf-8 -*-
"""
下载本地嵌入模型（模型名与目标目录从 config.yaml 读取，一次下完永久可用）。
用途：把文献片段变成向量，用于检索。本地运行，不消耗任何 API 费用。

当前配置：BAAI/bge-m3（多语言，约 2.3GB）——因为语料是**英文文献**、提问是**中文**。
想换模型：改 config.yaml 的 embedding_hub / embedding_local，然后**重建知识库**
（换嵌入模型 = 知识库变了 → 知识库指纹变化 → 历史答题记录与陷阱题验证全部失效）。

用法：python src\1a_get_embedding_model.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 必须在使用 huggingface_hub 之前设置：镜像 + 把缓存也放在项目内（不污染用户目录，也避免权限问题）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".hfcache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".hfcache" / "hub"))

import yaml  # noqa: E402

_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
REPO = _cfg["embedding_hub"]
TARGET = ROOT / _cfg["embedding_local"]


def main():
    print(f"镜像    : {os.environ['HF_ENDPOINT']}")
    print(f"模型    : {REPO}")
    print(f"目标目录: {TARGET}")

    if (TARGET / "config.json").exists() and (
            list(TARGET.glob("*.safetensors")) or list(TARGET.glob("*.bin"))):
        print("[OK] 模型已存在，跳过下载。")
    else:
        from huggingface_hub import snapshot_download
        print(f"开始下载 {REPO} ……（视模型与网络，1~20 分钟）")
        try:
            # 只下 safetensors：很多仓库同时提供 .bin 和 .safetensors，两个都下体积翻倍
            p = snapshot_download(repo_id=REPO, local_dir=str(TARGET), max_workers=4,
                                  allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
            if not list(TARGET.glob("*.safetensors")) and not list(TARGET.glob("*.bin")):
                print("  该仓库没有 safetensors，改用 pytorch_model.bin 重试……")
                p = snapshot_download(repo_id=REPO, local_dir=str(TARGET), max_workers=4,
                                      allow_patterns=["*.json", "*.txt", "*.bin", "*.model"])
            print(f"[OK] 下载完成: {p}")
        except Exception as e:
            print(f"[X] 下载失败: {type(e).__name__}: {str(e)[:300]}")
            print("\n可选办法：")
            print("  1) 确认镜像：$env:HF_ENDPOINT='https://hf-mirror.com' 后重跑本脚本")
            print(f"  2) 浏览器打开 https://hf-mirror.com/{REPO}/tree/main 手动下载：")
            print("     config.json / *.safetensors / tokenizer.json / tokenizer_config.json / *.txt")
            print(f"     放进同一目录：{TARGET}")
            raise SystemExit(1)

    # 验证模型真的能加载
    print("验证能否加载 ……")
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(TARGET))
    v = m.encode(["非小细胞肺癌的驱动基因检测意义是什么？"], normalize_embeddings=True)
    print(f"[OK] 加载成功，向量维度 = {v.shape[1]}")
    print("[!] 换过嵌入模型就必须重建知识库：python src\\1_build_kb.py")


if __name__ == "__main__":
    main()
