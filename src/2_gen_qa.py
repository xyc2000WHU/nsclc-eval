# -*- coding: utf-8 -*-
"""
⚠ 已弃用：本项目改为**人工出题**，见项目根目录《出题指南.md》。

弃用原因（实测记录，不是猜测）：拿 40 个真实片段让裁判模型出题，**40/40 全部"解析失败"**
（模型返回的 JSON 结构与解析器不匹配），一道题都没产出，且每次都要花 API 钱。
与其修生成器，不如由有临床背景的人直接出题：质量更高、更省钱。

现在请用：
  python src\\2a_find_chunk.py --template    # 生成空出题表（含 3 条示例）
  python src\\2a_find_chunk.py 关键词         # 找素材、拿「来源片段」ID、--show 看全文
  python src\\2b_check_questions.py          # 自检手写题（证据句是否逐字、选项是否成套…）
  python src\\2c_verify_traps.py             # 陷阱题人工确认 → 写入验证标记

本文件保留：候选题导出格式、证据句自动校验、陷阱题验证标记的**历史实现**（供对照，勿再使用）。
--------------------------------------------------------------------------------------------
原说明：

第 2 步：让强模型按文献出题，产出"候选题目"。

★ 常规题：**按文献分层抽样**后逐段出题（默认每篇 4 段 × 10 篇 = 40 段 → 约 80 道题）。
  分层抽样的原因：NCCN 指南占了知识库 1/3 的块，若全局随机抽，题目会被它主导；
  分层抽样保证 10 篇文献都被覆盖，报告里也能给出"每篇文献贡献多少题"。
★ 陷阱题（v2 修正，重要）：
    之前按"单个片段"判断"这段答不了"，但模型答题时能看到 top-4 片段，
    甚至答案就在另一篇文献里 —— 这样陷阱题可能其实可答，会把"答对"误判成"过度自信"。
    现在改成两步，缺一不可：
      ① 全语料级生成：把语料主题清单给裁判模型，让它出"看起来相关、但答案不在这些文献里"的问题；
      ② 强制验证：对该问题在知识库做一次**比答题更宽的检索**（默认 top-12），
         把检索到的片段交裁判模型判断"能否回答"；只有判定"不能回答"的才保留，
         无法验证的一律丢弃（宁可少，不可错）。
         [!] 这是"检索 + 模型判定"，**不是穷尽证明**。报告里要写"经检索与裁判判定为不可答"，
             不要写成"客观不可答 / 全库已证明不可答"。

★ 证据句自动校验：只检查"引文能否在原文里找到"（省你逐条 Ctrl+F）。
    [i] 注意：这**不等于医学正确性**。医学正确性必须你（项目负责人）逐条判定，
        审核表里有专门一列「医学正确性(填:对/错/需改)」，定稿时会统计确认数量。

产出：
  qa/candidates.jsonl   候选题目（机器可读）
  qa/审核表.csv          你的审核表（UTF-8-BOM，Excel 直接打开）

用法：
  python src\2_gen_qa.py                          # 每篇 4 段 × 10 篇 ≈ 80 道常规题 + 10 道陷阱题
  python src\2_gen_qa.py --per-doc 6 --traps 12   # 规模加大
  python src\2_gen_qa.py --traps 0                # 先不出陷阱题（不推荐，陷阱题是报告亮点）
  python src\2_gen_qa.py --recheck-traps          # 只重验陷阱题（改了题或重建了知识库时）
"""
import argparse
import csv
import hashlib
import json
import random
import re

from common import (CFG, chat, extract_json_array, kb_fingerprint, path, read_csv_any,
                    read_jsonl, retrieve, write_jsonl)

CHUNK_PROMPT = """你是肿瘤学命题专家。请**只依据**下面这段文献原文出题，禁止使用文献之外的知识。

出 {n} 道题，要求：
1. 至少 1 道针对具体数值或阈值（剂量、比例、生存率、HR 及 95%CI、样本量、分期阈值）。
2. 至少 1 道是"患者或年轻医生真的会问的问题"（治疗选择、预后、不良反应）。
3. 若题目适合做单选题，额外给出 4 个选项（options）和正确选项字母（correct_option）；
   干扰项要基于**常见的临床误解**，不能随便凑。不适合时 options 填 null。
4. 语言要求：**question 与 reference_answer 用中文；evidence_quote 必须是英文原文逐字复制**
   （本项目是"中文提问 → 英文文献"的跨语言检索评测，证据句必须能在原文里逐字找到）。

严格输出 JSON 数组，不要任何解释文字，每个元素字段固定为：
question, reference_answer, evidence_quote, answerable, type, options, correct_option

- question：**中文**问题（患者或年轻医生会问的口吻）
- reference_answer：**中文**答案，1-3 句，严格依据原文；数值（剂量/比例/生存率/HR/95%CI）必须原样保留
- evidence_quote：**英文原句**，必须从下面原文中**逐字复制**（不要翻译、不要改写、不要合并两句）
- answerable：true
- type：fact（事实）/ number（数值）/ guideline（指南推荐）/ reasoning（需推理）
- options：{{"A":"…","B":"…","C":"…","D":"…"}} 或 null（选项内容用中文）
- correct_option："A"/"B"/"C"/"D" 或 null

文献原文（来源：{doc}，片段号 {cid}）：
\"\"\"
{chunk}
\"\"\"
"""

TRAP_GEN_PROMPT = """下面是一个医学知识库的文献片段清单（文件名 + 片段开头）。

请出 {n} 道"陷阱题"，要求：
- 问题看起来与这些文献的主题**高度相关**（同一癌种或同一治疗领域里真的会被问到的问题）；
- 但答案**不在**这些文献里（例如：这些文献没报道过的数值、没讨论的亚组、
  其他癌种的数据、或文献发表之后才有的新结论）；
- 不要出"一看就荒谬"的问题（那测不出东西）；要出**看起来该有人能回答、实际这些文献没覆盖**的问题。
- 问题用**中文**（本项目是"中文提问、英文文献"的跨语言场景）。

严格输出 JSON 数组，不要任何解释文字，每个元素字段固定为：
question, topic, why_absent

- question：中文问题
- topic：你想考察的领域
- why_absent：一句话说明为什么这些文献答不了它

文献片段清单：
{outline}
"""

VERIFY_PROMPT = """请判断：**基于下面这些文献片段，能否回答这个问题？**

严格标准：
- 只有片段中确实含有回答问题所需的关键事实（结论、数值、适应症、推荐等），才算"能"；
- 只是主题相关、但不足以回答的，算"不能"。

只输出 JSON：{{"answerable": "能" 或 "不能", "reason": "一句话理由"}}

【问题】
{question}

【检索到的片段】
{contexts}
"""


def guard_prompt():
    """启动自检：提示词里的 JSON 示例必须写成 {{ }}（转义），否则 .format() 抛异常。"""
    try:
        CHUNK_PROMPT.format(n=1, doc="d", cid="c", chunk="t")
        TRAP_GEN_PROMPT.format(n=1, outline="o")
        VERIFY_PROMPT.format(question="q", contexts="c")
    except Exception as e:
        raise SystemExit(f"[X] 2_gen_qa 提示词模板的转义有误（JSON 示例的花括号要写成 {{{{ }}}}）：{e}")


# ---------------- 证据句自动校验（不等于医学正确）----------------
def norm(s: str) -> str:
    return re.sub(r"[\s\u3000()（）\[\]【】{}，,。.；;：:、\-—_/\\\"'’‘“”]", "", s or "")


def qhash(q: str) -> str:
    """问题文本的短哈希（必须与 3_finalize_qa.py 一致）。
    写进「陷阱验证」标记里，这样一旦你改了这道陷阱题的问题文本，验证就自动失效。"""
    return hashlib.md5((q or "").strip().encode("utf-8")).hexdigest()[:8]


def trap_marker(q: str, verify_k: int) -> str:
    """陷阱题验证标记：绑定**知识库版本** + 检索宽度 + 问题文本哈希。
    知识库一改，标记就失效（因为"能否回答"这件事本身依赖当时的语料）。
    必须与 3_finalize_qa.py 的解析正则一致。"""
    return f"已验证(kb-{kb_fp()},top-{verify_k},{qhash(q)})"


_KBF = {"v": None}


def kb_fp() -> str:
    if _KBF["v"] is None:
        _KBF["v"] = kb_fingerprint()
    return _KBF["v"]


def check_evidence(quote: str, chunk_text: str) -> float:
    """返回 0~1：证据句在原文里的可核验程度（1.0 = 逐字命中）。"""
    nq, nc = norm(quote), norm(chunk_text)
    if len(nq) < 8:
        return 0.0
    if nq in nc:
        return 1.0
    grams = [nq[i:i + 12] for i in range(0, max(1, len(nq) - 12) + 1, 6)]
    grams = [g for g in grams if len(g) == 12]
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in nc) / len(grams)


def build_outline(chunks, per_doc=3, head=110) -> str:
    docs = {}
    for c in chunks:
        docs.setdefault(c["file"], []).append(c)
    lines = []
    for f, cs in docs.items():
        lines.append(f"【{f}】共 {len(cs)} 段，开头示例：")
        for c in cs[:per_doc]:
            lines.append(f"  - {c['text'][:head]}…")
    return "\n".join(lines)


def make_traps(m, chunks, n) -> list:
    """全语料级生成陷阱题，并用"宽检索 + 裁判模型"验证它确实不可答。"""
    if n <= 0:
        return []
    # 比答题用的 top-k 更宽：宁可多给片段，也别让"其实可答"的题漏进来
    verify_k = max(int(CFG["top_k"]) * 3, 12)
    print(f"\n生成陷阱题（全语料级）：先让模型出 {n} 道候选，再逐条验证"
          f"（检索 top-{verify_k} + 裁判判定）……")
    outline = build_outline(chunks)
    text, _ = chat(m, [{"role": "user", "content": TRAP_GEN_PROMPT.format(n=n, outline=outline)}],
                   temperature=0.5, max_tokens=2500)
    cands = extract_json_array(text)
    if not cands:
        print("  [X] 候选生成失败（模型没按 JSON 输出），本次不出陷阱题")
        return []

    kept = []
    for i, c in enumerate(cands, 1):
        q = str(c.get("question", "")).strip()
        if not q:
            continue
        verdict, hits = verify_one(m, q, verify_k)
        ok = verdict == "不能"
        print(f"  陷阱题 {i}/{len(cands)}：裁判判定「{verdict or '无法解析'}」 → {'保留' if ok else '丢弃'}")
        if ok:
            kept.append({
                "question": q, "reference_answer": "所提供的文献中没有相关内容",
                "evidence_quote": "", "answerable": False, "type": "trap",
                "options": None, "correct_option": None,
                "trap_topic": str(c.get("topic", "")), "trap_verified": True,
                "trap_verify_k": verify_k, "trap_kb_fp": kb_fp(),
                "trap_reason": str(c.get("why_absent", "")),
                "verify_hits": hits,          # verify_one 返回的就是片段 ID 列表
            })
    print(f"  陷阱题结果：候选 {len(cands)} → 验证通过 {len(kept)}（丢弃 {len(cands) - len(kept)}）")
    if len(kept) < n:
        print(f"  [!] 通过数少于目标 {n}：可重跑本步，或用 --traps {n} 再试一次")
    return kept


def verify_one(m, q, verify_k):
    """对一个候选问题做"宽检索 + 裁判判定"，返回 (判定, 命中的片段 ID)。"""
    hits = retrieve(q, verify_k)
    vtext, _ = chat(m, [{"role": "user", "content": VERIFY_PROMPT.format(
        question=q,
        contexts="\n\n".join(f"[片段{j}] {h['text']}" for j, h in enumerate(hits, 1)))}],
        temperature=0.0, max_tokens=300)
    try:
        verdict = str(json.loads(vtext[vtext.find("{"): vtext.rfind("}") + 1]).get("answerable", "")).strip()
    except Exception:
        verdict = ""
    return verdict, [h["chunk_id"] for h in hits]


def read_review(p):
    """读审核表 → 统一走 common.read_csv_any（多编码兜底只保留一份实现）。"""
    rows = read_csv_any(p)
    if not rows:
        raise SystemExit(f"审核表读不出来或为空：{p}\n"
                         f"    请在 Excel 里「另存为 → CSV UTF-8(逗号分隔)」再试。")
    return rows


def recheck_traps(m, verify_k):
    """不重新出题：只对审核表里标为陷阱题的行重新验证，并刷新「陷阱验证」列。

    什么时候要跑它：你改了某道陷阱题的问题文本。验证标记里存的是"当时验证的问题文本"的哈希，
    改过之后哈希对不上，3_finalize_qa 会把它拦下（这是设计如此，防止绕过验证）。
    """
    p = path("review_csv")
    if not p.exists():
        raise SystemExit(f"找不到 {p} —— 先跑 python src\\2_gen_qa.py 生成审核表")
    rows = read_review(p)
    if not rows:
        raise SystemExit("审核表是空的")
    fields = list(rows[0].keys())
    if "陷阱验证" not in fields:
        fields.append("陷阱验证")
        for r in rows:
            r.setdefault("陷阱验证", "")

    n_ok = n_bad = n_skip = 0
    for r in rows:
        if (r.get("答案能否从文献得到") or "").strip() != "否":
            n_skip += 1
            continue
        q = (r.get("修改后的问题(留空=不改)") or "").strip() or (r.get("问题") or "").strip()
        if not q:
            continue
        verdict, _ = verify_one(m, q, verify_k)
        if verdict == "不能":
            r["陷阱验证"] = trap_marker(q, verify_k)
            n_ok += 1
            print(f"  [OK] 通过：{q[:44]}……")
        else:
            r["陷阱验证"] = f"验证失败({verdict or '无法解析'})"
            n_bad += 1
            print(f"  [X] 不通过（裁判判定「{verdict or '无法解析'}」）：{q[:44]}……")

    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n[OK] 重新验证完成：通过 {n_ok}，不通过 {n_bad}（普通题跳过 {n_skip}）→ {p}")
    print("     「不通过」的题在 3_finalize_qa 会被拦下：把它们改回普通题并填「医学正确性」，或把「保留」改成 0。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=4,
                    help="每篇文献抽几个片段出题（默认 4 → 10 篇约 40 个片段）")
    ap.add_argument("--per-chunk", type=int, default=2, help="每个片段出几道常规题（默认 2）")
    ap.add_argument("--traps", type=int, default=10, help="陷阱题目标数量（默认 10，验证后可能更少）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recheck-traps", action="store_true",
                    help="不重新出题，只对审核表里的陷阱题重新验证并刷新「陷阱验证」列")
    args = ap.parse_args()

    guard_prompt()
    if args.recheck_traps:
        verify_k = max(int(CFG["top_k"]) * 3, 12)
        print(f"重新验证审核表里的陷阱题（宽检索 top-{verify_k} + 裁判判定）……")
        recheck_traps(CFG["judge"], verify_k)
        return
    chunks = read_jsonl(path("chunks"))
    if not chunks:
        raise SystemExit("kb/chunks.jsonl 是空的 —— 先跑 python src\\1_build_kb.py")

    rng = random.Random(args.seed)
    body = [c for c in chunks if len(c["text"]) > 400]
    if not body:
        raise SystemExit("没有足够长的文献片段（>400 字）—— 检查 PDF 解析结果。")

    # ★ 按文献分层抽样：保证 10 篇文献都被覆盖，而不是被最大的那篇（NCCN 有 486 块）主导。
    by_doc = {}
    for c in body:
        by_doc.setdefault(c["doc_id"], []).append(c)
    picked = []
    for doc_id in sorted(by_doc):
        pool = by_doc[doc_id][:]
        rng.shuffle(pool)
        got = pool[: args.per_doc]
        picked += got
        label = got[0]["file"][:46] if got else "(无可用片段)"
        print(f"  {doc_id} {label:<48} 抽 {len(got)} 段（该文献共 {len(pool)} 段可用）")
    rng.shuffle(picked)

    m = CFG["judge"]
    print(f"\n用 {m['display']}({m['name']}) 出题：{len(picked)} 个片段 × {args.per_chunk} 道常规题"
          f" ≈ {len(picked) * args.per_chunk} 道，另加最多 {args.traps} 道陷阱题\n")

    by_id = {c["chunk_id"]: c["text"] for c in chunks}
    rows, fail = [], 0
    for i, c in enumerate(picked, 1):
        text, _ = chat(m, [{"role": "user", "content": CHUNK_PROMPT.format(
            n=args.per_chunk, doc=c["file"], cid=c["chunk_id"], chunk=c["text"])}],
            temperature=0.4, max_tokens=2600)
        items = extract_json_array(text)
        if not items:
            fail += 1
            print(f"  [{i}/{len(picked)}] {c['chunk_id']} 解析失败")
            continue
        for it in items:
            it = dict(it)
            it["source_chunk"] = c["chunk_id"]
            it["source_doc"] = c["file"]
            it["evidence_score"] = round(
                check_evidence(str(it.get("evidence_quote", "")), by_id.get(c["chunk_id"], "")), 3)
            rows.append(it)
        print(f"  [{i}/{len(picked)}] {c['chunk_id']} → {len(items)} 题（累计 {len(rows)}）")

    # 陷阱题（全语料级 + 验证）
    rows += make_traps(m, chunks, args.traps)

    seen, uniq = set(), []
    for r in rows:
        q = str(r.get("question", "")).strip()
        if q and q not in seen:
            seen.add(q)
            uniq.append(r)

    uniq.sort(key=lambda r: (0 if r.get("answerable", True) else 1, -(r.get("evidence_score") or 0)))
    write_jsonl(path("candidates"), uniq)

    ok = sum(1 for r in uniq if (r.get("evidence_score") or 0) >= 0.8)
    warn = sum(1 for r in uniq if 0 < (r.get("evidence_score") or 0) < 0.8)
    bad = sum(1 for r in uniq if r.get("evidence_score") == 0.0)
    traps = sum(1 for r in uniq if not r.get("answerable", True))
    print(f"\n候选 {len(uniq)} 道（片段解析失败 {fail} 个）")
    print(f"  证据句：命中 {ok} ｜ 需确认 {warn} ｜ 未命中 {bad} ｜ "
          f"陷阱题 {traps}（经检索+裁判判定为不可答）")
    print(f"  → {path('candidates')}")

    # ---------------- 导出审核表 ----------------
    cols = ["保留(1=要,0=删)", "自动校验", "陷阱验证", "医学正确性(填:对/错/需改)", "题目ID", "题型",
            "答案能否从文献得到", "问题", "标准答案", "原文证据句", "A", "B", "C", "D", "正确答案",
            "来源片段", "来源文献", "修改后的问题(留空=不改)", "修改后的答案(留空=不改)",
            "备注(为什么删/要改)"]
    out = path("review_csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(uniq, 1):
            sc = r.get("evidence_score")
            if not r.get("answerable", True):
                flag = "陷阱题(检索+裁判判定不可答)"
            elif sc >= 0.8:
                flag = "证据句命中"
            elif sc > 0:
                flag = f"证据句{sc:.0%}需确认"
            else:
                flag = "证据句未命中"
            opt = r.get("options") or {}
            opt = opt if isinstance(opt, dict) else {}
            tk = r.get("trap_verify_k")
            marker = (trap_marker(r.get("question", ""), int(tk))
                      if (not r.get("answerable", True)) and tk else "")
            w.writerow([1, flag, marker, "", f"C{i:03d}", r.get("type", ""),
                        "否" if not r.get("answerable", True) else "是",
                        r.get("question", ""), r.get("reference_answer", ""),
                        r.get("evidence_quote", ""),
                        opt.get("A", ""), opt.get("B", ""), opt.get("C", ""), opt.get("D", ""),
                        r.get("correct_option", "") or "",
                        r.get("source_chunk", ""), str(r.get("source_doc", ""))[:60],
                        "", "", ""])
    print(f"审核表 → {out}")
    print("\n★ 审核时请记住两件事：")
    print("  1.「自动校验」只说明**引文能否在原文找到**，不代表答案在医学上正确 ——"
          " 医学正确性请逐条填「医学正确性」列。")
    print("  2. 陷阱题已用全库检索 + 裁判模型验证过「确实不可答」，但你仍可复核"
          "（重点看它是不是真的问不到，而不是你觉得太难）。")
    print("  目标：留 80 题（事实 35 / 数值 15 / 指南 12 / 推理 8 / 陷阱 10），带选项的 ≥30 题。")
    print("  另存为 CSV UTF-8 → python src\\3_finalize_qa.py")


if __name__ == "__main__":
    main()
