# -*- coding: utf-8 -*-
r"""
为人工抽检中「成因」未填/填「不确定」的条目自动给出**候选成因**（本地检索，不调用任何模型）。

判据（**先查完整知识库，再谈检索名次**）：
  1) 该记录的登记证据已在 top-6 片段里（gold_hit=真）      → 候选：**模型没读出证据**
  2) 不在 top-6，但**登记片段本身就在 kb/chunks.jsonl 里** → 候选：**库内存在但未进上下文**
     （客观现象：证据在库内、未进入模型收到的片段）
  3) 登记片段**确实不在库里**（source_chunk 为空或查无此块）→ 候选：**登记片段不在库内**
     （陷阱题属设计如此：库内无该信息 → 拒答是正确行为）

[!] 为什么把旧的第 3 类拆开：旧实现把"宽检索（top-50）也找不到登记片段"直接写成
    **「知识库或题目缺」**，这是**逻辑错误**——登记的 source_chunk 大多就在 kb/chunks.jsonl 里
    （实测 9/9 全部在库内），只是没进 top-50。把"没检索到"读成"库里没有"，会得出
    "问题出在题库/知识库"的错误结论，等于**替检索链路开脱**。
    现在先按 source_chunk 查**完整知识库**，只记录"在不在库内"这个客观事实，
    不推断"检索失败"或"模型无责"。

[!] 免责限定（写进产物）：**登记片段未命中 ≠ 其它片段无等效证据**。
    因此本表只给"候选成因"，不能据此为模型免责，也不能断定检索一定失败；
    要分开"检索没给到"与"模型没读出"，需要固定回答条件后补"提供充分证据"的对照实验（本次未做）。

用法：python _selftest/suggest_cause.py    产物：qa\成因待确认.csv
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import dirpath, path, read_jsonl   # noqa: E402

_s = importlib.util.spec_from_file_location("rr", ROOT / "_selftest" / "retrieval_report_official.py")
rr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rr)

COLS = ["题号", "模型", "人工判定", "错误类型", "原成因", "候选成因(自动)", "依据",
        "登记片段是否在库内(客观)", "确认成因(填:模型没读出证据/检索没给到/库内存在但未进上下文/不确定)"]

CAVEAT = ("登记片段未命中 ≠ 其它片段无等效证据；本列只是客观现象，"
          "不能据此为模型免责、也不能断定检索失败（对照实验未做）")


def read_any(p):
    raw = Path(p).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "utf-16"):
        try:
            return list(csv.DictReader(raw.decode(enc).splitlines()))
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise SystemExit(f"无法解码 {p}")


def pick(row, *names):
    for c in row:
        if c and any(c.startswith(n) for n in names):
            return (row[c] or "").strip()
    return ""


def kb_chunk_ids() -> set:
    """**完整知识库**的片段 ID 集合（kb/chunks.jsonl）。

    [!] 这是本脚本唯一权威的"库内有没有"判据：检索函数只给 top-k，
        拿它当"库里有没有"用，就会把"没检索到"误判成"库里没有"。
    """
    p = path("chunks")
    if not p.exists():
        raise SystemExit(f"[X] 找不到知识库 {p} → 无法判断登记片段是否在库内，不生成产物")
    return {json.loads(l)["chunk_id"] for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()}


def classify(gold_hit: bool, source_chunk: str, in_kb: bool, answerable: bool):
    """三分法判据（**先查完整知识库**，可单测）。返回 (候选成因, 依据前缀)。

    [!] 判据顺序是本次修复的要点：**先看 source_chunk 在不在完整知识库里**，
        而不是先看"宽检索有没有找到"。旧实现先看检索名次，于是"top-50 没有"被写成
        "知识库或题目缺" —— 而那 9 条登记的片段其实**都在库内**。
    """
    sc = (source_chunk or "").strip()
    if gold_hit:
        return "模型没读出证据", "登记证据已在模型收到的 top-6 片段内（gold_hit=真）"
    if in_kb:
        return "库内存在但未进上下文", (f"登记片段 {sc} **存在于库内**（kb/chunks.jsonl 已核对），"
                                    f"但未进入模型上下文")
    if not answerable:
        return "登记片段不在库内", "陷阱题（设计上库内无此信息）→ 拒答属正确行为"
    return "登记片段不在库内", f"登记片段（source_chunk={sc or '空'}）在 kb/chunks.jsonl 中查无此块"


def main():
    qa = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    recs = {}
    for f in sorted(dirpath("runs").glob("*_open_full.jsonl")):
        for r in read_jsonl(f):
            recs[(r["model"], r["question_id"])] = r
    audit = []
    for p in ("人工抽检表.csv", "人工抽检表_追加60.csv"):
        f = ROOT / "qa" / p
        if f.exists():
            audit += read_any(f)
    todo = [r for r in audit
            if pick(r, "人工判定").startswith(("部分一致", "不一致"))
            and pick(r, "成因") in ("", "不确定")]
    kb_ids = kb_chunk_ids()
    print(f"需要判定成因的条目：{len(todo)} 条（完整知识库 {len(kb_ids)} 块）\n")

    out = []
    for r in todo:
        m, qid = pick(r, "模型"), pick(r, "题目ID")
        rec = recs.get((m, qid), {})
        item = qa.get(qid, {})
        gold = bool(rec.get("gold_hit"))
        sc = (item.get("source_chunk") or "").strip()
        in_kb = bool(sc) and sc in kb_ids
        cand, why = classify(gold, sc, in_kb, bool(item.get("answerable", True)))
        if in_kb and not gold:
            # 只在"库内存在但未进上下文"这一支才去跑检索：既省算力，也避免把
            # "库内没有该块"的题拿去问检索名次（那没有意义）。
            hits = rr.retrieve(item.get("question", ""), top_k=50)
            ids = [h["chunk_id"] for h in hits]
            rank = f"第 {ids.index(sc) + 1} 名" if sc in ids else "top-50 之外"
            why += f"（严格 top-6 未命中，实测名次：{rank}）。{CAVEAT}"
        out.append({"题号": qid, "模型": m, "人工判定": pick(r, "人工判定"),
                    "错误类型": pick(r, "错误类型"), "原成因": pick(r, "成因") or "（空）",
                    "候选成因(自动)": cand, "依据": why,
                    "登记片段是否在库内(客观)": ("是（" + sc + "）") if in_kb else "否",
                    "确认成因(填:模型没读出证据/检索没给到/库内存在但未进上下文/不确定)": ""})
        print(f"  {qid} {m:<12}{pick(r, '人工判定'):<6}→ {cand}｜{why[:60]}…")

    dst = path("qa_set").parent / "成因待确认.csv"
    with open(dst, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    import collections
    dist = collections.Counter(r["候选成因(自动)"] for r in out)
    print(f"\n[OK] → {dst}")
    print("   候选分布：", dict(dist))
    print(f"   [!] 口径：{CAVEAT}")
    print("   [!] 旧版把『宽检索找不到登记片段』写成『知识库或题目缺』属逻辑错误，"
          "已改为先按 source_chunk 查完整知识库；库内存在的一律记『库内存在但未进上下文』。")


if __name__ == "__main__":
    main()
