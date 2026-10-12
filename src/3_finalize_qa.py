# -*- coding: utf-8 -*-
"""
第 3 步：把你审核过的 审核表.csv 变成正式评测集。

★ 入库门槛（v3，两条都是硬门槛，不通过就不进评测集）：

  1. 普通题：「保留」=1 且「医学正确性」=「对」。
     证据句命中只证明引文能在原文找到，**不能代替医学正确性判断**；填「错」「需改」或留空一律拦下。

  2. 陷阱题（「答案能否从文献得到」=「否」）：必须有**仍然有效**的验证标记。
     验证标记由 2_gen_qa.py 写入「陷阱验证」列，格式：
         `已验证(kb-<知识库指纹>,top-12,<问题文本短哈希>)`
     因此下面两种情况都会被拦下：
       · 没有标记（例如你手工把一道普通题改成"否"）；
       · **知识库重建过**（kb 指纹对不上）→ "能否回答"这个判断本身依赖当时的语料，必须重验；
       · 陷阱题的问题文本被改过（哈希对不上）。
     处理办法：跑 `python src\\2_gen_qa.py --recheck-traps` 重新验证。

★ 陷阱题不进入"医学正确性确认"统计：它的正确性来自"宽检索 + 裁判判定不可答"，
  而不是临床医生对某个答案的确认，两者不能混为一谈。

产出：
  qa/qa_set.jsonl          正式评测集（含证据句、medically_verified、trap_verified/trap_verify_k）
  qa/mcq/nsclc_val.csv     选择题子集（EvalScope general_mcq 直接用）

用法：python src\3_finalize_qa.py
"""
import csv
import hashlib
import re
import sys
from collections import Counter

from common import CFG, DEV, dirpath, g_col, kb_fingerprint, path, read_csv_any, write_jsonl

KEEP_TRUE = {"1", "1.0", "是", "y", "yes", "true", "t", "保留"}
# 陷阱题验证标记：已验证(kb-<知识库指纹>,top-<检索宽度>,<问题文本哈希>)
MARKER_RE = re.compile(
    r"已验证\s*\(\s*kb-([0-9a-fA-F]{6,16})\s*,\s*top-(\d+)\s*,\s*([0-9a-fA-F]{6,16})\s*\)")


def qhash(q: str) -> str:
    """问题文本的短哈希：用于判断"验证过的题"是否被改过。必须与 2_gen_qa.py 一致。"""
    return hashlib.md5((q or "").strip().encode("utf-8")).hexdigest()[:8]


def read_review(p):
    """读审核表。多编码兜底已抽到 common.read_csv_any（7_report/7d/7b 共用同一份实现）。

    [!] 保留这个薄封装只为向后兼容（regression_test 等外部脚本按名字调用它）。
    """
    rows = read_csv_any(p)
    if not rows:
        sys.exit(f"审核表读不出来或为空：{p}\n"
                 f"    请在 Excel 里「另存为 → CSV UTF-8(逗号分隔)」再试。")
    return rows


def g(row, *names):
    """按列名取值，容忍列名被 Excel 改过（多余空格等）→ 见 common.g_col。"""
    return g_col(row, *names)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true",
                    help="开发模式：允许尚未通过医学复核的题入库，产物写入 dev/ 目录（绝不进正式榜单）")
    args = ap.parse_args()
    if args.dev:
        import common
        common.enable_dev()
        print("[DEV] 「医学正确性」未填的题会被收进 dev 版评测集，并标记 dev=true / medically_verified=false")

    src = path("review_csv")
    if not src.exists():
        sys.exit(f"找不到 {src} —— 先跑 python src\\2a_find_chunk.py --template 生成出题表，"
                 f"按《出题指南.md》填写")
    rows = read_review(src)
    kb_now = kb_fingerprint()          # 当前知识库版本：陷阱题验证必须与它一致

    final, dropped = [], 0
    blocked = []          # (题目ID, 原因)
    dev_unverified = []   # 开发模式下"未复核但放行"的题目
    for r in rows:
        if g(r, "保留(1=要,0=删)", "保留") not in KEEP_TRUE:
            dropped += 1
            continue
        q = g(r, "修改后的问题(留空=不改)", "修改后的问题") or g(r, "问题")
        a = g(r, "修改后的答案(留空=不改)", "修改后的答案") or g(r, "标准答案")
        if not q or not a:
            dropped += 1
            continue
        qid = g(r, "题目ID")
        declared_trap = g(r, "答案能否从文献得到") == "否"
        med = g(r, "医学正确性(填:对/错/需改)", "医学正确性")
        marker = g(r, "陷阱验证")
        m_mark = MARKER_RE.search(marker)

        is_verified_trap = False
        verify_k = None
        if declared_trap:
            if not m_mark:
                blocked.append((qid, "陷阱题缺少有效验证标记（手工改成「否」的题不算陷阱题）"))
                continue
            mk_kb = m_mark.group(1).lower()
            mk_top = int(m_mark.group(2))
            mk_q = m_mark.group(3).lower()
            if mk_kb != kb_now:
                blocked.append((qid, f"知识库已变更（验证时 kb-{mk_kb}，当前 kb-{kb_now}）"
                                     f"→ 陷阱题需重新验证"))
                continue
            if mk_q != qhash(q):
                blocked.append((qid, "陷阱题的问题文本被修改过 → 原验证失效，需重新验证"))
                continue
            is_verified_trap = True
            verify_k = mk_top
        else:
            if med != "对":
                if not DEV["on"]:
                    blocked.append((qid, f"医学正确性 = {med or '（留空）'}，未确认为「对」"))
                    continue
                dev_unverified.append(qid)     # 开发模式：放行，但下面会标记为未复核

        item = {
            "question_id": qid,
            "type": "trap" if is_verified_trap else (g(r, "题型") or "fact"),
            "answerable": not is_verified_trap,
            "question": q,
            "reference_answer": a,
            "evidence_quote": g(r, "原文证据句"),
            "source_chunk": g(r, "来源片段"),
            "source_doc": g(r, "来源文献"),
            "options": {k: g(r, k) for k in "ABCD" if g(r, k)},
            "correct_option": g(r, "正确答案"),
            "review_note": g(r, "备注(为什么删/要改)", "备注"),
            "medically_verified": (not is_verified_trap) and (med == "对"),
            "dev": DEV["on"],
            "medical_check": med,
            "trap_verified": is_verified_trap,
            "trap_verify_k": verify_k,
            "trap_kb_fp": kb_now if is_verified_trap else None,
            "trap_question_hash": qhash(q) if is_verified_trap else None,
        }
        final.append(item)

    if blocked:
        print(f"[!] 有 {len(blocked)} 道题没通过入库门槛，**未进入评测集**：")
        for qid, why in blocked[:25]:
            print(f"      {qid}: {why}")
        if len(blocked) > 25:
            print(f"      …… 其余 {len(blocked) - 25} 条同理")
        print("      处理：① 普通题 → 在「医学正确性」列填「对」（或把「保留」改成 0）；")
        print("            ② 陷阱题 → 跑 python src\\2c_verify_traps.py 逐题人工确认后写入验证标记。")

    if not final:
        sys.exit("没有任何题目通过入库门槛。检查「保留」「医学正确性」「陷阱验证」三列后重跑。")

    # 重新编号，保证 ID 连续（报告和抽检表都要用）
    prefix = CFG["question_prefix"]
    for i, it in enumerate(final, 1):
        it["question_id"] = f"{prefix}-{i:03d}"
    write_jsonl(path("qa_set"), final)

    trap_n = sum(1 for it in final if it["trap_verified"])
    verifiable = [it for it in final if not it["trap_verified"]]
    n_med = sum(1 for it in verifiable if it["medically_verified"])
    n_ev = sum(1 for it in final if it["evidence_quote"])

    print(f"[OK] 正式评测集：{len(final)} 题（审核删除 {dropped} 题）→ {path('qa_set')}")
    print("   题型分布：", dict(Counter(it["type"] for it in final)))
    print(f"   可答题 {len(verifiable)} 道 + 已验证陷阱题 {trap_n} 道"
          f"（陷阱题验证检索宽度 top-{sorted({it['trap_verify_k'] for it in final if it['trap_verify_k']}) or '-'}）")
    print(f"   有原文证据句：{n_ev}/{len(final)} 题（这个比例越高，评测集越可信）")
    print(f"   医学正确性：可答题 {n_med}/{len(verifiable)} 道已由你确认「对」")
    if verifiable and n_med < len(verifiable):
        if DEV["on"]:
            print(f"   [DEV] 其中 {len(verifiable) - n_med} 道可答题**未经医学复核**：已收入 dev 版并标记 "
                  f"medically_verified=false（正式模式会拦下它们，不得作为正式结果引用）。")
        else:
            print(f"   [!] 仍有 {len(verifiable) - n_med} 道可答题未确认——它们**没有**进入评测集。")

    # ---- 导出选择题 CSV（EvalScope 格式）----
    mcq = [it for it in final if it["options"] and it["correct_option"] in "ABCD"]
    if mcq:
        d = dirpath("mcq_dir")
        out = d / f"{CFG['mcq_subset']}_val.csv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "question", "A", "B", "C", "D", "answer"])
            for it in mcq:
                w.writerow([it["question_id"], it["question"], it["options"].get("A", ""),
                            it["options"].get("B", ""), it["options"].get("C", ""),
                            it["options"].get("D", ""), it["correct_option"]])
        print(f"[OK] 选择题子集：{len(mcq)} 题 → {out}（EvalScope 的 general_mcq 直接用这个文件）")
    else:
        print("[!] 没有一道题带选项。至少手工给 20-30 道题加上 A/B/C/D 和正确答案，"
              "选择题是最稳的指标（不需要裁判模型）。")

    print("\n下一步：python src\\4_rag_answer.py --workers 4（或先 --limit 3 试跑）")
    if DEV["on"]:
        print("\n" + "!" * 76)
        print("[DEV] 这是**开发版**评测集：产物在 dev/ 目录，题目标记 dev=true。")
        if dev_unverified:
            print(f"[DEV] 其中 {len(dev_unverified)} 道题**未经医学复核**，"
                  f"因此本批结果不得写进简历或正式报告：")
            print("      " + "、".join(dev_unverified[:12]) + ("…" if len(dev_unverified) > 12 else ""))
        print("[DEV] 你自己（或医生）复核后，把「医学正确性」填「对」，再用**正式模式**重跑：")
        print("[DEV]   python src\\3_finalize_qa.py           # 不加 --dev")
        print("[DEV]   python src\\4_rag_answer.py --workers 4")
        print("!" * 76)


if __name__ == "__main__":
    main()
