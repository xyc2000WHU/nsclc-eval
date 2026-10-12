# -*- coding: utf-8 -*-
r"""
拆分上下文充分率：可答题 vs 陷阱题。

- 可答题：充分率越高越好（低说明检索没给到证据）；
- 陷阱题：预期全部被判"不充分"。

[!] 口径（重要）：这是裁判对**检索到的 top-6 片段**是否足以回答的判定，
    属于「**对检索上下文不足的补充核验**」——裁判只看到 top-6 片段，
    **不能证明整个知识库都答不了**，也不能称为"陷阱集的独立验证"，
    更不等同于独立医生审核。

[!] 汇总前的**题号集合闸门**（第三方复验新确认的收尾点 1，2026-10-12 修）：
    旧实现把"日志少一道题"当成正常情况照样汇总，于是打出 `正式集 80 − 解析失败 0` 却报
    分母 79 这类**自相矛盾**的说明；日志里出现**未知题号**时还会直接在
    `is_trap()` 里 `KeyError` → traceback。
    现在：**汇总之前**先断言「日志结果行解析到的题号集合 == 当前 `qa/qa_set.jsonl` 题号集合」，
    不等就**明确失败**并逐条列出「缺失题号 / 多出的未知题号」，**不输出任何比率**、
    也不写产物；分母构成一律写成**可核对的等式**（结果行 = 可答 + 陷阱 + 解析失败；缺失 0；未知 0）。

[!] 解析口径 = **严格**（第三方复验新确认的 3 类问题之二，2026-10-12 修）：
    日志行必须给出**唯一且明确**的「上下文充分: 是/否」，策略与
    `src\5b_judge_faithfulness.py` 完全一致——**只认带字段名的标签**、
    取全部匹配后**全部一致才接受**，**冲突就判「解析失败」，绝不猜**。
    冲突 / 无效（缺字段、字段后不是是/否）/ 题号缺失或重复 / 题号不在正式集
    → 一律记「解析失败」并**排除出分母**，同时打印「排除 N 条（解析失败）」，
    产物里写明**分母构成**（可答题/陷阱题各是多少、排除了多少）。

    改动前的宽松实现是"整段文本正则 findall"：同一题若出现两次（例如重跑追加了一行），
    两次的标签会**同时进分母**——冲突的一方若恰好是「是」，结果就被算成"充分"；
    而**完全无法解析的行则静默消失**（既不算充分也不算不充分，分母被悄悄缩小）。

用法：python _selftest\split_sufficiency.py
"""
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import path, read_jsonl   # noqa: E402

# [!] 字段名 + 标签：与 5b 的容错写法同源（只认带字段名的标签，不接受"裸 是/否"）
RE_GOLD = re.compile(r"金标准片段在\s*top-(\d+)\s*[:：]\s*(是|否)")
# [!] 允许日志把判定显式写成「解析失败」（`check_context_sufficiency.py` 的严格解析在
#     冲突/无效输出时就会这么写）——这种条目**计入解析失败并排除出分母**，绝不折算成"否"。
RE_SUFF = re.compile(r"上下文充分\s*[:：]\s*(是|否|解析失败)")
# [!] **JSON 形式**（收尾点 3，2026-10-12 修）：`{"上下文充分": "是"}` / `{"结论":"yes"}`
#     这类输出此前一律记「解析失败」。现在字段名（上下文充分/上下文充分性/结论/判定/judge/verdict）、
#     键值两侧的引号、英文值 yes/no 都容错，但值一律**归一为 是/否**；
#     英文 yes/no 同样要求后面不是字母（避免 "no" 命中 "none"）。
#     同一行里既有 是 又有 否（无论文本还是 JSON）→ 仍然「解析失败」，绝不猜。
RE_SUFF_JSON = re.compile(
    r"[\"'\u201c\u201d\u300c\u300d]?\s*"
    r"(?:上下文充分|上下文充分性|结论|判定|judge|verdict)\s*"
    r"[\"'\u201c\u201d\u300c\u300d]?\s*[:：]\s*"
    r"[\"'\u201c\u201d\u300c\u300d]?\s*(是|否|yes|no|解析失败)(?![\u4e00-\u9fffA-Za-z])",
    re.IGNORECASE)
RE_QID = re.compile(r"NSCLC-\d+")
# 英文值 → 中文归一（**只认 yes/no**，不接受 true/false —— 那是另一套语义，不猜）
_NORM_VAL = {"是": "是", "否": "否", "YES": "是", "NO": "否", "解析失败": "解析失败"}


def norm_val(v: str) -> str:
    """把取到的判定值归一为 是/否/解析失败（英文 yes/no → 是/否）。"""
    return _NORM_VAL.get((v or "").upper(), v)


def suff_tags(line: str):
    """取出**一行里全部**「上下文充分」判定标签（文本形式 + JSON 形式），英文值已归一。"""
    tags = [norm_val(v) for v in RE_SUFF.findall(line)]
    tags += [norm_val(v) for v in RE_SUFF_JSON.findall(line)]
    return tags

# 改动前的**宽松**实现（整段 findall）——**已废弃**，仅保留给「改前 → 改后」对照与
# 验收脚本 `verify_sufficiency_strict.py` 使用；**产物数字一律走严格解析**。
LOOSE_RE = (r"(NSCLC-\d+)｜金标准片段在 top-\d+: (是|否)｜上下文充分: (是|否)")


def parse_loose(txt: str, qa_ids):
    """复刻改动前的宽松解析（整段 `re.findall`），**只用于对照**。

    两个已知隐患（正是本次要修的）：
      ① 同一题在日志里出现两次时，两条记录**都进分母**；`rate()` 数的是 "== 是" 的条数，
         所以"冲突"会偏向「是」；
      ② 无法解析的行**完全不进 rows**，既不算失败也不计数 → 分母被静默缩小。
    """
    rows = re.findall(LOOSE_RE, txt)
    rows = [r for r in rows if r[0] in qa_ids]
    return rows


def read_text_any(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8", "utf-16", "utf-16-le", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"无法解码 {p}")


def parse_strict(txt: str, qa_ids):
    """严格解析逐题结果。

    返回 (ok, failed, non_result)：
      · ok        : {题号: {"gold": 是/否, "suff": 是/否, "line": 行号, "n": 出现次数}}
      · failed    : [{"line": 行号, "qid": 题号或"", "reason": 原因}]
      · non_result: [行号]（表头/空行/说明行——**不当作解析失败**，单独计数）
    """
    ok, failed, non_result = {}, [], []
    per_qid = collections.defaultdict(list)      # 题号 -> [(行号, gold, suff)]
    for ln, raw in enumerate(txt.splitlines(), 1):
        line = raw.strip()
        if not line:
            non_result.append(ln)
            continue
        qids = sorted(set(RE_QID.findall(line)))
        golds = RE_GOLD.findall(line)
        suffs = suff_tags(line)
        if not qids and not golds and not suffs:
            non_result.append(ln)                 # 表头/说明行：没有题号也没有结果字段
            continue
        # ---- 结果行：必须唯一且明确 ----
        if len(qids) != 1:
            failed.append({"line": ln, "qid": "" if not qids else "/".join(qids),
                           "reason": f"题号不唯一或缺失（找到 {len(qids)} 个）"})
            continue
        qid = qids[0]
        if qid not in qa_ids:
            failed.append({"line": ln, "qid": qid, "reason": "题号不在当前正式评测集里"})
            continue
        if not suffs:
            failed.append({"line": ln, "qid": qid,
                           "reason": "无效输出：整行找不到带字段名的判定标签"
                                     "（文本形式「上下文充分: 是/否」、"
                                     "JSON 形式 {\"上下文充分\"/\"结论\": \"是\"/\"否\"} 都没有）"})
            continue
        u_suff = sorted(set(suffs))
        if len(u_suff) > 1:
            failed.append({"line": ln, "qid": qid,
                           "reason": f"标签冲突（同一行出现 {u_suff}）→ 解析失败，不猜"})
            continue
        if u_suff[0] == "解析失败":
            # 上游（check_context_sufficiency.py 的严格解析）已判定"解析失败" → 直接排除出分母
            failed.append({"line": ln, "qid": qid,
                           "reason": "日志显式标注「上下文充分: 解析失败」（冲突/无效输出）→ 排除出分母"})
            continue
        if not golds:
            failed.append({"line": ln, "qid": qid,
                           "reason": "无效输出：缺少「金标准片段在 top-N: 是/否」字段"})
            continue
        u_gold = sorted({g for _, g in golds})
        if len(u_gold) > 1:
            failed.append({"line": ln, "qid": qid,
                           "reason": f"金标准字段冲突（{u_gold}）→ 解析失败，不猜"})
            continue
        per_qid[qid].append((ln, u_gold[0], u_suff[0]))

    for qid, hits in per_qid.items():
        u_suff = sorted({s for _, _, s in hits})
        u_gold = sorted({g for _, g, _ in hits})
        if len(u_suff) > 1 or len(u_gold) > 1:
            failed.append({"line": hits[0][0], "qid": qid,
                           "reason": f"同一题号出现多次且结果冲突"
                                     f"（上下文充分={u_suff}，金标准在top6={u_gold}）→ 解析失败"})
            continue
        ok[qid] = {"gold": u_gold[0], "suff": u_suff[0], "line": hits[0][0], "n": len(hits)}
    return ok, failed, non_result


def collect(txt: str, qa: dict):
    """严格解析 + **题号集合闸门**（汇总之前必须通过）。

    返回 dict：
      · ok / failed / non_result：同 `parse_strict`
      · official / seen / missing / unknown：题号集合对账（missing/unknown 都必须为空）
      · error：非空 = 闸门未通过 → 调用方必须**终止、不输出任何比率、不写产物**
    """
    official = set(qa)
    ok, failed, non_result = parse_strict(txt, official)
    # 「日志中解析到的题号集合」= 解析成功的题号 ∪ 被记为解析失败的题号
    # （一行里出现多个题号时 parse_strict 会记成 "A/B"，这里按 "/" 拆开逐个计入）
    seen = set(ok)
    for f in failed:
        for q in (f.get("qid") or "").split("/"):
            if RE_QID.fullmatch(q):
                seen.add(q)
    unknown = sorted(seen - official)
    missing = sorted(official - seen)
    err = []
    if missing or unknown:
        err = ["[X] 题号集合闸门未通过 → **拒绝汇总，不输出任何比率**"
               "（缺题或出现未知题号时，分母/分子都不可信）。",
               f"    日志结果行解析到的题号 **{len(seen)}** 个"
               f"｜当前正式集 `qa/qa_set.jsonl` 题号 **{len(official)}** 个"]
        if missing:
            err.append(f"    缺失题号（正式集里有、日志结果行里没有）**{len(missing)}** 个：")
            err += [f"      - {q}：日志里没有该题号的结果行" for q in missing]
        if unknown:
            err.append(f"    多出的未知题号（日志里有、正式集里没有）**{len(unknown)}** 个：")
            err += [f"      - {q}：该题号**不在当前正式集** `qa/qa_set.jsonl` 里" for q in unknown]
        err.append("    → 请补全日志（或删除未知题号行）后重跑；"
                   "本次**不写出** `outputs/上下文充分率.md`，也没有任何比率可用。")
    return {"ok": ok, "failed": failed, "non_result": non_result,
            "official": official, "seen": seen, "missing": missing, "unknown": unknown,
            "error": err}


def main():
    qa = {i["question_id"]: i for i in read_jsonl(path("qa_set"))}
    fp = ROOT / "_selftest" / "suff_official.txt"
    txt = read_text_any(fp)
    # ---- 汇总之前先过题号集合闸门（缺题 / 未知题号都不得给出任何比率）----
    col = collect(txt, qa)
    if col["error"]:
        for ln in col["error"]:
            print(ln, file=sys.stderr)
        return 2
    ok, failed, non_result = col["ok"], col["failed"], col["non_result"]
    missing, unknown = col["missing"], col["unknown"]
    if not ok:
        sys.exit("严格解析后一条有效结果都没有，检查 _selftest\\suff_official.txt 的格式")

    def is_trap(q):
        # [!] 用 .get 而不是 qa[q]：未知题号曾在这里 KeyError → traceback。
        #     现在闸门已在前面拦下未知题号，这里再兜一层，保证任何情况下都不会抛栈。
        it = qa.get(q)
        return bool(it and it.get("trap_verified"))

    ans = {q: v for q, v in ok.items() if not is_trap(q)}
    tr = {q: v for q, v in ok.items() if is_trap(q)}
    fail_ans = [f for f in failed if f["qid"] and not is_trap(f["qid"])]
    fail_tr = [f for f in failed if f["qid"] and is_trap(f["qid"])]
    fail_other = [f for f in failed if not f["qid"] or f["qid"] not in qa]
    # 涉及到的**题号**（去重）：分母等式按题号算，同一题的多条失败行只扣一次
    fail_ans_q = sorted({f["qid"] for f in fail_ans})
    fail_tr_q = sorted({f["qid"] for f in fail_tr})
    n_all_q = len(qa)
    n_trap_all = sum(1 for q in qa if is_trap(q))
    n_ans_all = n_all_q - n_trap_all

    def rate(rs, want="是"):
        y = sum(1 for v in rs.values() if v["suff"] == want)
        return y, len(rs), (y / len(rs) * 100 if rs else 0.0)

    ya, na, pa = rate(ans)
    yt, nt, pt = rate(tr)
    gold = sum(1 for v in ans.values() if v["gold"] == "是")

    # ---- 分母等式必须自洽：任一项对不上就直接失败，绝不输出自相矛盾的说明 ----
    R, S, F = len(ok) + len(failed), len(ok), len(failed)
    bad_eq = []
    if R != S + F:
        bad_eq.append(f"结果行 {R} ≠ 解析成功 {S} + 解析失败 {F}")
    if S != na + nt:
        bad_eq.append(f"解析成功 {S} ≠ 可答题 {na} + 陷阱题 {nt}")
    if F != len(fail_ans) + len(fail_tr) + len(fail_other):
        bad_eq.append(f"解析失败 {F} ≠ 可答题 {len(fail_ans)} + 陷阱题 {len(fail_tr)}"
                      f" + 无题号 {len(fail_other)}")
    if na != n_ans_all - len(fail_ans_q):
        bad_eq.append(f"可答题分母 {na} ≠ 正式集可答题 {n_ans_all} − 解析失败涉及题号 {len(fail_ans_q)}")
    if nt != n_trap_all - len(fail_tr_q):
        bad_eq.append(f"陷阱题分母 {nt} ≠ 正式集陷阱题 {n_trap_all} − 解析失败涉及题号 {len(fail_tr_q)}")
    if missing or unknown:
        bad_eq.append(f"缺失 {len(missing)}、未知 {len(unknown)}（闸门已保证为 0）")
    if bad_eq:
        print("[X] 分母等式不自洽 → 拒绝输出比率（说明模板与数据必须能对上）：", file=sys.stderr)
        for ln in bad_eq:
            print("    - " + ln, file=sys.stderr)
        return 3

    # ---- 改前（宽松）对照：只用于在产物里写清「改前 → 改后」，不参与任何产物数字 ----
    lo = parse_loose(txt, set(qa))
    lo_ans = [r for r in lo if not is_trap(r[0])]
    lo_tr = [r for r in lo if is_trap(r[0])]
    lo_ya = sum(1 for r in lo_ans if r[2] == "是")
    lo_yt = sum(1 for r in lo_tr if r[2] == "是")
    lo_pa = lo_ya / len(lo_ans) * 100 if lo_ans else 0.0
    lo_dup = len(lo) - len({r[0] for r in lo})

    print(f"严格解析：结果行 {R} 行（解析成功 {S}｜解析失败 **{F}**）"
          f"；另有非结果行（表头/空行）{len(non_result)} 行")
    print(f"排除 {F} 条（解析失败）：可答题 {len(fail_ans)} 条（{len(fail_ans_q)} 个题号）"
          f"｜陷阱题 {len(fail_tr)} 条（{len(fail_tr_q)} 个题号）"
          f"｜无题号 {len(fail_other)} 条（不在正式集/无法归属的未知题号由闸门拦截）")
    for f in failed[:5]:
        print(f"   [!] 第 {f['line']} 行 {f['qid'] or '（无题号）'}：{f['reason']}")
    print(f"分母构成：结果行 {R} = 解析成功 {S}（可答 {na} + 陷阱 {nt}）+ 解析失败 {F}；"
          f"缺失 {len(missing)}；未知 {len(unknown)}")
    print(f"    可答题分母 {na} = 正式集可答题 {n_ans_all} − 解析失败涉及题号 {len(fail_ans_q)}；"
          f"陷阱题分母 {nt} = 正式集陷阱题 {n_trap_all} − 解析失败涉及题号 {len(fail_tr_q)}")
    print(f"改前（宽松整段 findall）：可答题 {lo_ya}/{len(lo_ans)} = {lo_pa:.1f}%"
          f"｜陷阱题 {lo_yt}/{len(lo_tr)}｜重复计入 {lo_dup} 条｜排除条数**不统计**（无法解析的行静默消失）")
    print(f"可答题 {na} 道：上下文充分 {ya} 道 = **{pa:.1f}%**")
    print(f"   其中金标准片段进了 top-6 的 {gold} 道 = {gold / na * 100:.1f}%（严格命中的下界）")
    print(f"陷阱题 {nt} 道：被判「充分」的 {yt} 道 = {pt:.1f}%（预期为 0）")

    bad = sorted(q for q, v in ans.items() if v["suff"] == "否")
    print(f"\n可答题中上下文**不充分**的 {len(bad)} 道（模型很可能答不出，报告里要单列）：")
    print("   " + ("、".join(bad) if bad else "（无）"))

    bt = sorted(q for q, v in tr.items() if v["suff"] == "是")
    print(f"\n陷阱题中被判「充分」的 {len(bt)} 道（非 0 则需复核该陷阱）：")
    print("   " + ("、".join(bt) if bt else
                   "（无 → 10 道陷阱的检索上下文均被判为不足，与出题方判断一致）"))

    print("\n[!] 口径提醒：本检查是对「检索上下文是否足够」的补充核验——"
          "裁判只看到 top-6 片段，不能证明整个知识库都答不了，"
          "因此不能称为「陷阱集的独立验证」。")

    out = ROOT / "outputs" / "上下文充分率.md"
    lines = [f"# 上下文充分率（严格解析｜{len(ok)} 题｜裁判判定）", "",
             "## 解析口径与分母构成（2026-10-12 起为**严格**解析）", "",
             "- **严格解析**：每行必须给出**唯一且明确**的「上下文充分: 是/否」标签；"
             "策略与 `src/5b_judge_faithfulness.py` 相同——只认**带字段名**的标签，"
             "取全部匹配后**全部一致才接受**，**冲突即「解析失败」，绝不猜**。"
             "**JSON 形式**（收尾点 3）也接受：字段名 `上下文充分`/`结论`/`判定`/`judge`/`verdict`，"
             "值 `是`/`否`/`yes`/`no`（英文值内部归一为「是/否」）；"
             "同一条里既有「是」又有「否」（含 JSON 与文本互相矛盾）→ 仍记「解析失败」。",
             f"- **分母构成**（可核对的等式）：结果行 **{R}** = 解析成功 **{S}**"
             f"（可答题 {na} + 陷阱题 {nt}）+ **解析失败排除 {F}**"
             f"（可答题 {len(fail_ans)} 条/{len(fail_ans_q)} 个题号｜陷阱题 {len(fail_tr)} 条/"
             f"{len(fail_tr_q)} 个题号｜无题号 {len(fail_other)} 条）；"
             f"**缺失题号 {len(missing)}**；**未知题号 {len(unknown)}**；"
             f"另有非结果行（表头/空行）**{len(non_result)}** 行，不计入分母。",
             f"- **题号集合闸门**（2026-10-12 起，收尾点 1）：汇总前先断言「日志结果行解析到的题号集合 "
             f"== `qa/qa_set.jsonl` 题号集合」（{len(col['official'])} 个）；"
             f"**缺题或出现未知题号 → 明确失败并逐条列出，且不输出任何比率**"
             f"（旧实现会照样汇总，还会打出「80 − 0」却报 79 这类自相矛盾的说明）。",
             f"- 可答题分母 **{na}** = 正式集可答题 {n_ans_all} 道 − 解析失败涉及题号 {len(fail_ans_q)}；"
             f"陷阱题分母 **{nt}** = 正式集陷阱题 {n_trap_all} 道 − 解析失败涉及题号 {len(fail_tr_q)}。",
             f"- 改前（宽松整段 findall，**已废弃**）→ 改后（严格逐行）："
             f"**可答题 {lo_ya}/{len(lo_ans)} = {lo_pa:.1f}% → {ya}/{na} = {pa:.1f}%**；"
             f"**陷阱题 {lo_yt}/{len(lo_tr)} → {yt}/{nt}**；"
             f"**排除条数：宽松实现不统计（无法解析的行静默消失）→ 严格 {F} 条**"
             f"（本次真实日志下重复计入 {lo_dup} 条）。"
             "宽松实现的两处隐患：同一题重复出现时冲突标签会**同时进分母**"
             "（若其中一个是「是」，结果被算成「充分」）；完全无法解析的行会**静默消失**"
             "（分母被悄悄缩小、既不报错也不计数）。严格解析把这两类都显式记为「解析失败」并排除。",
             "",
             f"- 历史轨迹（**口径不同，不要混算**）：可答题充分率 **47/80 = 58.75%**（旧口径：裁判题面"
             f"不带 A~D 选项）→ **50/80 = 62.5%**（2026-10-11 17:50 一号全量重判后，题面含 A~D 选项）；"
             f"同期一号补选项重判与二号 v1→v3 版本变化**多因素同时发生**，"
             f"不能把这 3.75pp 单因素归因于「补选项」（详见 `outputs/项目改动记录_2026-10-11.md`）。",
             f"- 可答题 **{na}** 道：上下文充分 **{ya} 道 = {pa:.1f}%**；"
             f"其中金标准片段进 top-6 的 {gold} 道 = {gold / na * 100:.1f}%（严格命中的下界）",
             f"- 陷阱题 **{nt}** 道：被判「充分」的 **{yt} 道 = {pt:.1f}%**（预期为 0）", "",
             "[!] **口径**：这是裁判对**检索到的 top-6 片段**是否足以回答的判定，属"
             "「对检索上下文不足的补充核验」；它**不能证明整个知识库不可答**，"
             "也不等同于独立人工或临床审核。", "",
             f"## 可答题中上下文不充分的题（{len(bad)} 道）", ""]
    lines += [f"- {x}" for x in bad] if bad else ["- （无）"]
    lines += ["", "## 陷阱题判定（金标准片段是否进 top-6 / 上下文是否被判充分）", ""]
    for qid in sorted(tr):
        v = tr[qid]
        lines.append(f"- {qid}：进 top-6={v['gold']}｜上下文充分={v['suff']}")
    if failed:
        lines += ["", f"## 解析失败被排除的条目（{len(failed)} 条，不计入分母）", ""]
        lines += [f"- 第 {f['line']} 行｜{f['qid'] or '（无题号）'}｜{f['reason']}" for f in failed]
    lines += ["", "> 说明：这是**模型裁判**的判定，属辅助证据，不等于临床或人工结论。"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n记录已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
