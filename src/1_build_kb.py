# -*- coding: utf-8 -*-
"""
第 1 步：把 pdfs/ 里的 10 篇文献变成可检索的知识库。

做三件事：
  1. PDF → 纯文本（pypdf）
  2. 文本 → 小块（每块约 700 字，按段落边界切，避免把一句话切断）
  3. 每块 → 向量（本地 bge-small-zh 模型，免费不消耗 API）

产出：
  kb/chunks.jsonl     每行一个文献片段（后面检索、出题、判分都用它）
  kb/emb.npy          向量矩阵
  kb/doc_index.json   每篇文献切了多少块（写报告时要列文献清单）

用法：
  python src\1_build_kb.py
  python src\1_build_kb.py --search        # 建完后进入检索自测，人工确认"检索对不对"
"""
import json
import os
import re
import sys

import numpy as np

from common import (CFG, ROOT, compute_kb_fp, dirpath, encoder, file_sha16, path,
                    retrieve, write_jsonl)

CHUNK_SIZE = int(CFG["chunk_size"])
MIN_CHARS = int(CFG["min_chunk_chars"])
NCCN_2025_FILE = "02_NCCN_NSCLC_v4.2025.pdf"


def pdf_pages(p):
    """优先用 PyMuPDF（对中文 PDF 的文字层更靠谱），没装就退回 pypdf。"""
    try:
        try:
            import pymupdf as fitz                          # PyMuPDF 新包名（避免弃用警告）
        except ImportError:
            import fitz                                     # 旧包名
        with fitz.open(str(p)) as doc:
            return [pg.get_text() for pg in doc]
    except ImportError:
        from pypdf import PdfReader
        return [pg.extract_text() or "" for pg in PdfReader(str(p)).pages]


def clean(t: str) -> str:
    t = t.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    t = t.replace("-\n", "")                 # 英文跨行断词
    t = t.replace("\n", " \n")               # 保留段落感
    import re
    t = re.sub(r"[ \t\u3000]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def strip_nccn_boilerplate(pages: list[str]) -> list[str]:
    """去掉每页重复的 NCCN 页眉、导航和版权行，保留正文与脚注。"""
    repeated = {"NCCN Guidelines Version 4.2025", "Non-Small Cell Lung Cancer",
                "NCCN Guidelines Index", "Table of Contents", "Discussion"}
    return ["\n".join(line for line in page.splitlines()
                      if line.strip() not in repeated
                      and not line.strip().startswith("Version 4.2025, 05/23/25 ©"))
            for page in pages]


# ============ 参考文献/引用段过滤（两层，经真实数据校准）============
# 打分与章节截断的实现放在 common.py，供建库、检索降权、验收探针共用，避免逻辑漂移。
import common   # noqa: E402

REF_HEAD = common.REF_HEAD
REF_SIGNALS = common.REF_SIGNALS
strip_reference_section = common.strip_reference_section   # 已弃用（保留供对照）
ref_section_offset = common.ref_section_offset
ref_threshold = common.ref_threshold
ref_score = common.ref_score
REF_THRESHOLD = common.ref_threshold()


def is_reference_dump(chunk: str) -> bool:
    """块级兜底：判断这块是不是"文献条目堆"（抓章节截断漏掉的中间子节与无标题文献表）。

    注意：不能按行数判断——split_chunks() 把行用空格拼成了一个块，块内没有换行。
    """
    return ref_score(chunk) >= REF_THRESHOLD


def split_chunks(text: str, threshold: int = None, reason: str = "global"):
    """按段落聚合成块；超长段落再硬切。

    返回 (有效块列表, 被丢弃块列表)。丢弃项带上**得分与原因**，便于逐块留档审计。
    threshold=None 用全局阈值；参考文献章节内的块传章节阈值（更严）。
    """
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 2]
    blocks, cur = [], ""
    for line in lines:
        if len(cur) + len(line) + 1 <= CHUNK_SIZE:
            cur = (cur + " " + line).strip()
        else:
            if cur:
                blocks.append(cur)
            cur = line
    if cur:
        blocks.append(cur)

    chunks = []
    for b in blocks:
        if len(b) <= CHUNK_SIZE * 1.5:
            chunks.append(b)
        else:
            step = CHUNK_SIZE - 100
            for i in range(0, len(b), step):
                chunks.append(b[i:i + CHUNK_SIZE])

    th = REF_THRESHOLD if threshold is None else threshold
    kept, dropped = [], []
    for c in chunks:
        if len(c) < MIN_CHARS:
            continue
        s = ref_score(c)
        if s >= th:
            dropped.append({"text": c, "score": s, "reason": reason, "threshold": th})
            continue
        kept.append(c)
    return kept, dropped


def main():
    pdf_dir = ROOT / CFG["paths"]["pdfs"]
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"没找到 PDF。请把 10 篇文献放到：{pdf_dir}")

    print(f"找到 {len(pdfs)} 篇 PDF，开始解析……")
    rows, doc_index, dropped_rows = [], [], []
    for n, p in enumerate(pdfs, 1):
        doc_id = f"D{n:02d}"
        pages = pdf_pages(p)
        pages_used = pages
        if p.name == NCCN_2025_FILE:
            # 4.2025 版第 1-120 页是流程与原则；第 121-286 页的 Discussion
            # 明确标有 "Update in Progress"，不把这部分当作已更新的推荐依据。
            if len(pages) < 121 or "CAT-1" not in pages[119] or "MS-1" not in pages[120]:
                sys.exit("NCCN PDF 的页数或章节边界与已核对的 4.2025 版不符，请重新检查。")
            pages_used = strip_nccn_boilerplate(pages[:120])
        text = clean("\n".join(pages_used))
        # 参考文献处理：**不再整段截断**（会误伤 ST 分期表等正文，见 common.ref_section_offset 说明），
        # 改为"章节内按块过滤（更严阈值）+ 逐块留档"。
        off = ref_section_offset(text)
        if off is None:
            cs, dropped = split_chunks(text, REF_THRESHOLD, "global")
            sec_note = "无 References 标题"
            sec_chunks = 0
        else:
            cs_h, drop_h = split_chunks(text[:off], REF_THRESHOLD, "global")
            cs_t, drop_t = split_chunks(text[off:], ref_threshold(True), "in_reference_section")
            cs, dropped = cs_h + cs_t, drop_h + drop_t
            sec_chunks = len(cs_t)
            sec_note = f"章节内 {sec_chunks} 块按阈值 {ref_threshold(True)} 过滤"
        for j, c in enumerate(cs, 1):
            rows.append({"chunk_id": f"{doc_id}-C{j:03d}", "doc_id": doc_id,
                         "file": p.name, "text": c})
        for d in dropped:      # 逐块留档（带上 chunk_id，否则抽查打印会 KeyError）
            dropped_rows.append({"chunk_id": f"{doc_id}-J{len(dropped_rows) + 1:03d}",
                                 "doc_id": doc_id, "file": p.name, "text": d["text"],
                                 "score": d["score"], "reason": d["reason"],
                                 "threshold": d["threshold"]})
        doc_index.append({"doc_id": doc_id, "file": p.name, "chars": len(text),
                          "chunks": len(cs), "junk_dropped": len(dropped),
                          "ref_section_chunks": sec_chunks,
                          "pages_total": len(pages), "pages_used": len(pages_used)})
        print(f"  {doc_id} {p.name[:52]:<54} {len(text):>7} 字 → {len(cs):>3} 块"
              f"（过滤 {len(dropped)}｜{sec_note}）")

    if not rows:
        sys.exit("解析出来是空的：可能是扫描版 PDF（图片型）。换一篇有文字层的，或先用 OCR。")

    # ---------- 先在临时文件里生成全部产物，成功后再一次性替换 ----------
    # 目的：避免"新片段 + 旧向量"或"新片段 + 旧指纹"这种半成品状态被下游读到。
    import datetime
    kb_dir = path("chunks").parent
    tmp_chunks = kb_dir / "chunks.tmp.jsonl"
    tmp_emb = kb_dir / "emb.tmp.npy"          # 必须以 .npy 结尾，否则 np.save 会再加一个后缀
    tmp_ver = kb_dir / "kb_version.tmp.json"
    tmp_index = kb_dir / "doc_index.tmp.json"
    tmp_dropped = kb_dir / "dropped_chunks.tmp.jsonl"

    write_jsonl(tmp_chunks, rows)
    tmp_index.write_text(json.dumps(doc_index, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(tmp_dropped, dropped_rows)    # 被过滤的内容留档，便于人工抽查

    print(f"\n共 {len(rows)} 块，开始生成向量（首次会下载嵌入模型：bge-m3 约 2.3GB）……")
    emb = encoder().encode([r["text"] for r in rows], normalize_embeddings=True,
                           batch_size=16, show_progress_bar=True)
    np.save(tmp_emb, np.asarray(emb, dtype="float32"))

    # ★ 用**新片段**算指纹（不能用 kb_fingerprint()——它会读回上一次的 kb_version.json）
    fp = compute_kb_fp(rows)
    tmp_ver.write_text(json.dumps({
        "kb_fp": fp,
        "built_at": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
        "n_docs": len(doc_index),
        "n_chunks": len(rows),
        "chunk_size": CHUNK_SIZE,
        "min_chunk_chars": MIN_CHARS,
        "junk_dropped": sum(d.get("junk_dropped", 0) for d in doc_index),
        "embedding": str(ROOT / CFG["embedding_local"]) if (ROOT / CFG["embedding_local"]).exists()
        else CFG["embedding_hub"],
        "files": [d["file"] for d in doc_index],
        # 片段文件哈希：kb_fingerprint() 用它校验"版本记录与片段是否一致"，
        # 防止上次重建中断留下"新片段 + 旧版本记录"。
        "chunks_sha256": file_sha16(tmp_chunks),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 全部产物就绪 → 原子替换（kb_version.json 最后写，它是"提交标记"）
    os.replace(tmp_emb, path("emb"))
    os.replace(tmp_index, path("doc_index"))
    os.replace(tmp_dropped, kb_dir / "dropped_chunks.jsonl")
    os.replace(tmp_chunks, path("chunks"))
    os.replace(tmp_ver, kb_dir / "kb_version.json")

    print(f"[OK] 知识库完成：{len(rows)} 块，向量维度 {emb.shape[1]}，"
          f"过滤引用段 {sum(d.get('junk_dropped', 0) for d in doc_index)} 块")
    print(f"   {path('chunks')}")
    print(f"   {path('emb')}")
    print(f"   知识库版本指纹 kb-{fp}（已与片段文件哈希绑定，不一致会自动识别）")
    print("   [!] 知识库一变，历史答题记录与陷阱题验证都会自动失效，必须全量重跑。")

    # 过滤器抽查：看看被块级过滤丢掉的到底是什么，防止误删正文
    if dropped_rows:
        n_show = int((CFG.get("chunk_filter") or {}).get("audit_samples", 5))
        print(f"\n块级过滤抽查（共丢 {len(dropped_rows)} 块，明细在 kb/dropped_chunks.jsonl）：")
        for r in dropped_rows[:n_show]:
            print(f"  [{r['chunk_id']} 得分{r['score']} {r['reason']}] "
                  f"{r['text'][:130].replace(chr(10), ' ')}……")
        print("  → 请确认这些都是文献条目。若看到正文被误删，调高 config.yaml 里对应的阈值"
              f"（章节外 reference_score_threshold={REF_THRESHOLD}，"
              f"章节内 reference_score_threshold_in_section={ref_threshold(True)}）后重建；"
              "也可以用 _selftest\\simulate_build.py 先秒级预演一次。")

    # 视觉检查：随机看 3 块，确认没有乱码/页眉页脚噪声
    print("\n随机抽 3 块看看质量（如果全是页码/乱码，说明 PDF 解析有问题）：")
    for r in rows[:: max(1, len(rows) // 3)][:3]:
        print(f"  [{r['chunk_id']}] {r['text'][:110]}……")


def search_mode():
    print("\n检索自测：输入一个患者会问的问题，看检索出来的片段是否真的相关（输入 q 退出）")
    while True:
        q = input("\n问题> ").strip()
        if q.lower() in {"q", "quit", "exit", ""}:
            break
        for r in retrieve(q):
            print(f"  {r['score']:.3f} [{r['chunk_id']}] {r['text'][:120]}……")
    print("如果检索结果不相关：把 top_k 调大、chunk_size 调小，或换 embedding 模型（bge-base-zh / m3e-base）。")


if __name__ == "__main__":
    main()
    if "--search" in sys.argv:
        search_mode()
