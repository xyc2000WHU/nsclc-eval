# -*- coding: utf-8 -*-
r"""
正式集检索名次分布：逐题取 top-50，看登记的"来源片段"实际排第几。
用途：判断 26.7% 的严格命中率是"差一点"（前 20 名内 → 重排可救）还是"完全没对上"（50 名外）。
产物：outputs\检索名次分布.md
用法：python _selftest\retrieval_report_official.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import CFG, dirpath, path, read_jsonl, retrieve   # noqa: E402


def bucket(r):
    if r is None:
        return "50 名以外"
    if r <= 6:
        return "1-6（进上下文）"
    if r <= 10:
        return "7-10"
    if r <= 20:
        return "11-20"
    return "21-50"


def main():
    qa_all = read_jsonl(path("qa_set"))
    # [!] 检索统计只算**有登记证据的可答题**：10 道陷阱题本来就没有"来源片段"，
    #     把它们算进分母会把命中率人为拉低（曾把 30% 显示成 26.7%、33.8% 显示成 41%）。
    qa = [it for it in qa_all if it.get("answerable")]
    n_trap = len(qa_all) - len(qa)
    top_k = int(CFG["top_k"])
    rows, buckets, by_doc = [], Counter(), Counter()
    hit_doc = Counter()
    print(f"逐题检索（可答题 {len(qa)} 道；陷阱题 {n_trap} 道无登记证据已排除；候选池 50）……")
    for i, it in enumerate(qa, 1):
        hits = retrieve(it["question"], top_k=50)
        ids = [h["chunk_id"] for h in hits]
        sc = it.get("source_chunk") or ""
        rank = (ids.index(sc) + 1) if sc and sc in ids else None
        rows.append((it["question_id"], it.get("type"), it.get("source_doc") or sc.split("-")[0],
                     sc, rank))
        buckets[bucket(rank)] += 1
        by_doc[it.get("source_doc") or sc.split("-")[0]] += 1
        if rank and rank <= top_k:
            hit_doc[it.get("source_doc") or sc.split("-")[0]] += 1
        if i % 20 == 0:
            print(f"  {i}/{len(qa)}")

    hit = sum(1 for r in rows if r[4] and r[4] <= top_k)
    n = len(qa)
    md = [f"# 正式集检索名次分布（可答题 {n} 道｜top_k={top_k}）", "",
          f"- 金标准片段进 top-{top_k}：**{hit}/{n} = {hit / n * 100:.1f}%**；"
          f"**未进上下文 {(n - hit)}/{n} = {(n - hit) / n * 100:.1f}%**",
          f"- 另有 {n_trap} 道陷阱题**没有登记证据**（设计如此），不参与本统计", "",
          "| 名次区间 | 题数 | 占比 |", "|---|---|---|"]
    for b in ["1-6（进上下文）", "7-10", "11-20", "21-50", "50 名以外"]:
        c = buckets.get(b, 0)
        md.append(f"| {b} | {c} | {c / n * 100:.1f}% |")
    md += ["", "| 来源文献 | 题数 | 其中进 top-k |", "|---|---|---|"]
    for d in sorted(by_doc):
        md.append(f"| {d} | {by_doc[d]} | {hit_doc.get(d, 0)} |")
    md += ["", "## 逐题名次", "", "| 题目ID | 题型 | 来源 | 来源片段 | 名次 |", "|---|---|---|---|---|"]
    for qid, t, d, sc, r in rows:
        md.append(f"| {qid} | {t} | {d} | {sc} | {r if r else '>50'} |")
    out = dirpath("outputs") / "检索名次分布.md"
    out.write_text("\n".join(md), encoding="utf-8")

    print(f"\n金标准进 top-{top_k}：{hit}/{len(qa)} = {hit / len(qa) * 100:.1f}%")
    print("名次分布：", {b: buckets.get(b, 0) for b in
                     ["1-6（进上下文）", "7-10", "11-20", "21-50", "50 名以外"]})
    print(f"逐题明细已写入 {out}")


if __name__ == "__main__":
    main()
