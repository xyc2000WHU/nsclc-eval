# -*- coding: utf-8 -*-
"""
第 4 步：让每个模型把每道题答两遍（闭卷 / 开卷 RAG）。

★ 加速点：
  1. 检索只算一次：所有题目先统一检索好，各模型共用同一批片段（快数倍，且保证公平）
  2. --workers 多线程并发（默认 4）

★ 断点续跑的判据（v4）：
  每条回答有一个**输入指纹** = md5(完整提示词 + 标准答案 + 模型名 + 生成参数 + 提示词版本
  + **知识库版本指纹**)。
  开卷模式下提示词里含**全部检索片段正文**，所以片段改一个字指纹都会变；
  知识库版本进指纹则保证：**只要知识库变了（增删文献、重新切块、换嵌入模型），
  所有历史回答一并失效**——不可能靠 --limit 试跑把新旧数据拼成一张榜单。
  只有指纹一致、且上次没失败的记录才复用；否则重答。
      · 题目/标准答案被改过  · 知识库重建或片段正文变化  · 提示词或参数变化  · 上次失败

★ 三种"安全性"约定（v4，都是为了避免毁数据或把旧数据混进来）：
  1. `--limit` 试跑：只清点本次选中的题目，范围外的既有记录**一律保留**。
  2. `--mode open|closed` 单模式：只在本次跑的模式内清理；**不碰另一个模式的记录**。
     （v3 会在清理阶段去查未选模式的指纹 → KeyError 直接崩。）
  3. 传给 EvalScope 的产物只收**指纹有效**的记录：哪怕某个模式的记录是旧的，
     也不会被写进 `runs/*__open.json` 或 `qa/closed/*.jsonl`，只会被排除并计数。

产出：
  runs/{模型}__open.json        EvalScope RAGEval 需要的格式（只含有效记录）
  runs/{模型}__open_full.jsonl  全部原始记录（含指纹/命中/延迟/token）
  runs/{模型}__closed_full.jsonl
  qa/closed/{模型}.jsonl        EvalScope general_qa 需要的格式（只含有效记录）

用法：
  python src\4_rag_answer.py --limit 3 --model deepseek --workers 1   # 试跑（不动已有结果）
  python src\4_rag_answer.py --workers 4                              # 全量
  python src\4_rag_answer.py --mode open                              # 只补开卷
"""
import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (CFG, chat, dirpath, est_cost, is_substantive, kb_fingerprint, models_with_key, path,
                    read_jsonl, retrieve, write_jsonl)

PROMPT_VER = "v3"        # v3：选择题把选项写进题面（开卷/闭卷同格式）；v2：top_k 4→6 + 引用条目降权
TEMPERATURE = float(CFG.get("answer_temperature", 0.2))
# 额度必须给足：推理型模型（deepseek-flash / kimi-k3）会把额度用于"思考"，
# 太小会返回空内容（实测 700 时大量空答）。common.chat 还有"空内容自动翻倍"兜底。
MAX_TOKENS = int(CFG.get("answer_max_tokens", 2000))

OPEN_PROMPT = """你是肿瘤科医生，正在回答患者或年轻医生的问题。
请**严格依据**下面提供的文献片段回答，并遵守规则：

1. 只使用文献片段中的信息，不要引入片段之外的任何知识。
2. 如果片段中没有足够信息回答问题，请直接回答"所提供的文献中没有相关内容"，**不要猜测**。
3. 涉及数值（剂量、比例、生存率、HR、95%CI、样本量）必须与原文完全一致。
4. 先给结论，再给依据（依据要指向片段内容）。全文不超过 200 字。

【文献片段】
{context}

【问题】
{question}
"""

CLOSED_PROMPT = """你是肿瘤科医生，正在回答患者或年轻医生的问题。
请直接回答，不超过 200 字。如果把握不足，请明确说明"不确定"或"需要查阅文献"，
不要为了给出答案而编造具体数值或结论。

【问题】
{question}
"""


def slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "-", name).strip("-")


def empty_ctx():
    return {"ids": [], "texts": [], "context": ""}


_KB_FP = {"v": None}


def kb_fp() -> str:
    """知识库版本指纹（缓存一次）。它进指纹，意味着**知识库一变，所有历史回答全部失效**。"""
    if _KB_FP["v"] is None:
        _KB_FP["v"] = kb_fingerprint()
    return _KB_FP["v"]


def ctx_from_record(rec: dict) -> dict:
    """把一个已有记录里保存的检索片段还原成 ctx（用于给范围外记录复核指纹）。"""
    texts = rec.get("retrieved_contexts", []) or []
    ids = rec.get("retrieved_ids", []) or []
    return {"ids": ids, "texts": texts,
            "context": "\n\n".join(f"[片段{j}] {t}" for j, t in enumerate(texts, 1))}


INCLUDE_OPTIONS = bool(CFG.get("include_options_in_prompt", True))


def question_block(item: dict) -> str:
    """题面（选择题把 A~D 选项一并带上）。

    [!] 为什么必须带选项：原提示词只传问题，导致"依赖选项才能理解"的选择题在问答链路里
    无法与选择题链路比较（交付说明里已指出）。开卷与闭卷**共用同一个 question_block**，
    这样"有/无检索片段"两种条件才是同一套题面、同一套作答格式。
    """
    q = item["question"]
    opts = item.get("options") or {}
    if INCLUDE_OPTIONS and opts:
        lines = [q] + [f"{k}. {opts[k]}" for k in "ABCD" if opts.get(k)]
        lines.append("")
        lines.append("（这是选择题：请在回答**第一行**写「答案：X」（X 为 A/B/C/D），随后简要说理。）")
        return "\n".join(lines)
    return q


def build_prompt(mode: str, item: dict, ctx: dict) -> str:
    if mode == "open":
        return OPEN_PROMPT.format(context=ctx["context"], question=question_block(item))
    return CLOSED_PROMPT.format(question=question_block(item))


def fp_of(m, mode, item, ctx, ctx_override=None, budget=None):
    """返回 (指纹, 提示词)。指纹覆盖完整输入，不只是前一截。

    [!] 额度按**记录**走（budget）：提高某条的额度重答时，不应让其它记录失效。
    """
    if ctx_override is not None:
        ctx = ctx_override
    prompt = build_prompt(mode, item, ctx)
    raw = json.dumps({
        "v": PROMPT_VER, "model": m["name"], "prompt": prompt,
        "reference": item["reference_answer"],          # 标准答案不进 prompt，但变了也要重答
        "temperature": TEMPERATURE, "max_tokens": budget or MAX_TOKENS,
        "extra": m.get("extra", {}),
        "kb": kb_fp(),        # ★ 知识库版本：换库/重切块/换嵌入模型 → 全部重答
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12], prompt


def retrieval_fp(ctx: dict) -> str:
    """检索输入的指纹（片段 ID + 片段正文）。

    [!] 为什么额度之外还需要它：`fp` 里含**额度**（提高额度重答时不能让别的记录失效），
        所以 fp 相同**既可能**是"输入完全没变"，**也可能**是"输入变了但恰好也算出同一个 fp"。
        单靠 fp 无法区分这两者。把它单独存进记录，就能在**不依赖当前配置**的前提下
        证明"这条记录当初用的就是现在这批片段"。
    """
    raw = json.dumps({"ids": ctx.get("ids") or [], "texts": ctx.get("texts") or []},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def rec_fp_budgets(rec: dict) -> list:
    """这条记录的 fp 可能是用**哪个额度**算出来的（按可能性从高到低）。

    [!] 为什么需要它：额度会进指纹（见 fp_of），而"提高额度重答"的记录按自己的额度算指纹。
        `max_tokens_used` 存的是**最终成功那次**的额度，但指纹是按**最初请求**的额度算的
        （重答时预算可能被 chat() 内部翻倍），两者在重试场景下未必相等 →
        只认一个额度会把有效记录误判为"输入已变更"，进而**安排重答并退回默认额度**。
        实测已有 3 条 6000 额度的有效记录处于该状态。
    """
    out = []
    for b in (rec.get("attempt_budgets") or []):
        if isinstance(b, int) and b > 0:
            out.append(b)
    mt = rec.get("max_tokens_used")
    if isinstance(mt, int) and mt > 0:
        out.append(mt)
    out.append(MAX_TOKENS)
    seen, uniq = set(), []
    for b in out:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def rec_own_budget(rec: dict) -> int:
    """这条记录**自己要求过的最高额度**（用于重答时"绝不降级"）。

    [!] 一条有效的 6000 额度记录过去会被按默认 2000 重新安排作答 —— 既白花钱，
        又把已经答完整的回答退回可能被截断的额度（审计已复现）。
    """
    return max(rec_fp_budgets(rec)) if rec else MAX_TOKENS


def rec_own_fp(rec: dict, m, mode, item, ctx, want_fp) -> bool:
    """用**该记录自己的额度**复核它是不是当前输入算出来的指纹。"""
    for b in rec_fp_budgets(rec):
        if rec.get("fp") == fp_of(m, mode, item, ctx, budget=b)[0]:
            return True
    key = (mode, item["question_id"])
    return key in want_fp and rec.get("fp") == want_fp[key]


def record_is_current(m, mode, item, rec, want_fp, cur_ctx=None) -> bool:
    """这条记录是否还能用（指纹有效、没失败、**作答非空**，且检索输入未变）。

    [!] 判据只有这一处实现：main() 的任务规划、清理、导出都必须调用它，
        不能再各自复制一份简化版 —— 两份实现漂移正是"6000 额度有效记录被安排重答
        并退回 2000"的根因。
    [!] 必须把"空作答"也判为不可用：曾经有 16 条记录 response="" 却 error=false，
        于是被当成已完成，永远不会重答（静默缺数据）。
    [!] 检索输入必须与**当前**上下文核对（`cur_ctx`），不能只拿记录自带的片段自我核对：
        记录里的 `fp` 与 `retrieval_fp` 是同一次作答留下的，两者天然自洽 ——
        自我核对永远返回"未变"，于是**片段正文被改过也不会重答**（静默复用旧回答）。
        传了 cur_ctx 时：
          · 新记录用 `retrieval_fp` 与当前上下文比对；
          · 旧记录（缺字段）退回"自带片段与当前上下文逐字一致"。
        `cur_ctx is None`（如 7_report 无当前上下文）只核对记录内部自洽。
    """
    if not rec or rec.get("error"):
        return False
    ans = (rec.get("response") or "").strip()
    if not ans or not is_substantive(ans):
        return False                      # 空作答 / "答案："这类实质空答：都不算完成
    own_ctx = ctx_from_record(rec) if mode == "open" else empty_ctx()
    if not rec_own_fp(rec, m, mode, item, own_ctx, want_fp):
        return False
    if cur_ctx is None:
        return True
    rf = rec.get("retrieval_fp")
    if rf:
        # [!] 必须与**当前**上下文比对：这是"检索片段变了要重答"的唯一保障
        return rf == retrieval_fp(cur_ctx)
    # 旧记录（无 retrieval_fp）：要求自带片段与当前输入逐字一致
    return (own_ctx["ids"], own_ctx["texts"]) == (cur_ctx["ids"], cur_ctx["texts"])


def answer_one(m, mode, item, ctx, budget=None):
    # [!] 绝不把额度降到该记录已经用过的水平以下：main() 现在会把"记录自己的额度"传进来
    #     （见 rec_own_budget），这里再兜一层，防止别的调用方（如重答脚本）传小额度。
    budget = int(budget or MAX_TOKENS)
    request_budget = budget                 # 本次**请求**的额度：指纹按它算，保持稳定
    fp, prompt = fp_of(m, mode, item, ctx, budget=request_budget)
    t0 = time.time()
    ans, usage = chat(m, [{"role": "user", "content": prompt}],
                      temperature=TEMPERATURE, max_tokens=request_budget, **m.get("extra", {}))
    attempts = list(usage.get("attempts") or [])
    # [!] "实质空答"（例如只输出"答案："）既不是空串也不是失败，曾被当成成功作答。
    #     这里显式再答一次（额度翻倍），仍不合格就记 error，绝不静默当成功。
    retried = False
    retry_budget = min(request_budget * 2, 8192)
    if not is_substantive(ans):
        ans2, usage2 = chat(m, [{"role": "user", "content": prompt}],
                            temperature=TEMPERATURE, max_tokens=retry_budget,
                            **m.get("extra", {}))
        retried = True
        attempts += list(usage2.get("attempts") or [])   # [!] 两次尝试都要留档
        if is_substantive(ans2) or not (ans or "").strip():
            ans, usage = ans2, usage2
    ok = is_substantive(ans) and not (ans or "").startswith("[调用失败]")
    # [!] 额度与用量必须记**最终成功那次尝试**的，并把每次尝试留档：
    #     过去无论重试几次都写 max_tokens_used=最初请求值，也不记 finish_reason/attempts，
    #     于是"第一次 2000 被思考耗尽、第二次 4000 成功"在记录里完全看不出来，
    #     事后既无法复盘额度是否够用，也无法据此判断回答有没有被截断。
    attempt_budgets = []
    for att in attempts:
        b = att.get("requested_max_tokens") if isinstance(att, dict) else None
        if isinstance(b, int) and b > 0 and b not in attempt_budgets:
            attempt_budgets.append(b)
    if request_budget not in attempt_budgets:
        attempt_budgets.append(request_budget)
    if retried and retry_budget not in attempt_budgets:
        attempt_budgets.append(retry_budget)
    attempt_budgets.sort()
    # 最终成功那次的额度：取最后一次"有内容且有额度记录"的尝试；都没有就退回请求额度
    won = [att for att in attempts if isinstance(att, dict) and att.get("ok")]
    max_tokens_used = int(won[-1]["requested_max_tokens"]) if won else request_budget
    rec = {
        "model": m["display"], "model_name": m["name"], "mode": mode,
        "question_id": item["question_id"], "type": item["type"],
        "answerable": item["answerable"], "question": item["question"],
        "reference": item["reference_answer"], "response": ans,
        # 记录题面要素，便于后续判分：选择题的选项与正确项、是否把选项放进了提示词
        "options": item.get("options") or {},
        "correct_option": item.get("correct_option") or "",
        "options_in_prompt": bool(INCLUDE_OPTIONS and (item.get("options") or {})),
        "error": not ok,
        "unfinished": (not ok) or retried and not is_substantive(ans),
        "max_tokens_used": max_tokens_used,   # 最终成功那次的额度（不是最初请求值）
        "finish_reason": usage.get("finish_reason", ""),
        "attempts": attempts,                 # 每次尝试：请求额度/finish_reason/用量
        "attempt_budgets": attempt_budgets,   # 供截断扫描与续跑判定读（缺字段的历史记录也能用）
        "prompt_ver": PROMPT_VER, "fp": fp, "kb_fp": kb_fp(),
        "latency_s": round(time.time() - t0, 2),
        "in_tokens": usage.get("in_tokens", 0), "out_tokens": usage.get("out_tokens", 0),
    }
    if mode == "open":
        rec.update({"retrieved_ids": ctx["ids"], "retrieved_contexts": ctx["texts"],
                    "gold_hit": item["source_chunk"] in ctx["ids"]})
    # [!] 存下检索输入指纹：没有它就无法把"片段正文被改过"与"额度不同"区分开，
    #     续跑时会静默复用基于旧片段的回答。
    rec["retrieval_fp"] = retrieval_fp(ctx)
    rec["cost"] = round(est_cost(m, rec["in_tokens"], rec["out_tokens"]), 4)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true",
                    help="开发模式：读 dev/qa_set.jsonl，产物写入 dev/（题目未通过医学复核）")
    ap.add_argument("--model", default=None, help="只跑名字里含这个字符串的模型（试跑用）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（试跑用，不影响范围外记录）")
    ap.add_argument("--mode", default="both", choices=["both", "open", "closed"])
    ap.add_argument("--workers", type=int, default=4, help="并发数，被限流就改成 1（默认 4）")
    args = ap.parse_args()
    if args.dev:
        import common
        common.enable_dev()
    modes_this_run = {"open", "closed"} if args.mode == "both" else {args.mode}

    qa_all = read_jsonl(path("qa_set"))          # 完整题库（用于判断"题目是否真的被删了"）
    if not qa_all:
        raise SystemExit("qa/qa_set.jsonl 是空的 —— 先跑 3_finalize_qa.py")
    all_qa_ids = {it["question_id"] for it in qa_all}
    all_by_id = {it["question_id"]: it for it in qa_all}

    qa = qa_all[: args.limit] if args.limit else qa_all
    scope_ids = {it["question_id"] for it in qa}

    ms = models_with_key()
    if args.model:
        ms = [m for m in ms if args.model.lower() in (m["name"] + m["display"]).lower()]
    if not ms:
        raise SystemExit("没有可跑的模型（Key 未配置或 --model 没匹配上）")

    top_k = int(CFG["top_k"])

    # ---------- ① 检索只算一次（所有模型共用，保证公平）----------
    print(f"先统一检索：{len(qa)} 道题 × top-{top_k}（所有模型共用同一批片段）……")
    ctx_cache = {}
    for i, item in enumerate(qa, 1):
        hits = retrieve(item["question"], top_k)
        ctx_cache[item["question_id"]] = {
            "ids": [h["chunk_id"] for h in hits],
            "texts": [h["text"] for h in hits],
            "context": "\n\n".join(f"[片段{j}] {h['text']}" for j, h in enumerate(hits, 1)),
        }
        if i % 20 == 0 or i == len(qa):
            print(f"  检索 {i}/{len(qa)}")
    hit = sum(1 for it in qa if it["source_chunk"] in ctx_cache[it["question_id"]]["ids"])
    print(f"  检索命中（金标准片段进了 top-{top_k}）：{hit}/{len(qa)} = {hit / len(qa) * 100:.1f}%")
    if args.limit:
        print(f"  [!] 试跑模式（--limit {args.limit}）：范围外的既有记录会原样保留。")
    if args.mode != "both":
        print(f"  [!] 单模式（--mode {args.mode}）：另一个模式的记录不会被清理，也不会写进评测产物。")
    print()

    print(f"开始答题：{len(ms)} 个模型 × {len(qa)} 题 × {args.mode} 模式，并发 {args.workers}\n")

    total_cost, total_answered, total_excluded = 0.0, 0, 0
    total_cost_open, total_cost_closed = 0.0, 0.0
    for m in ms:
        s = slug(m["name"])
        runs_dir = dirpath("runs")
        f_open_full = runs_dir / f"{s}__open_full.jsonl"
        f_closed = runs_dir / f"{s}__closed_full.jsonl"

        done_open = {r["question_id"]: r for r in read_jsonl(f_open_full)}
        done_closed = {r["question_id"]: r for r in read_jsonl(f_closed)}

        # ---------- ② 指纹：两种模式都算（便宜），但只有本次跑的模式会重答/清理 ----------
        # [!] 任务规划**必须**与 record_is_current 用同一套判定：旧代码在这里另写了一份
        #     `old.get("fp") != want_fp[(mode, qid)]`，而 want_fp 是**按默认额度**算的，
        #     于是"按 6000 额度重答过的有效记录"被判为"输入已变更"→ 安排重答并**退回 2000 额度**
        #     （实测已有 3 条 6000 额度记录被这样安排）。现在统一调用 record_is_current，
        #     并用该记录自己的额度复核指纹。
        want_fp = {}
        for item in qa:
            qid = item["question_id"]
            want_fp[("open", qid)] = fp_of(m, "open", item, ctx_cache[qid])[0]
            want_fp[("closed", qid)] = fp_of(m, "closed", item, empty_ctx())[0]

        reasons = {"新增": 0, "上次失败重试": 0, "输入已变更重答": 0}
        tasks = []
        for item in qa:
            qid = item["question_id"]
            for mode in ("open", "closed"):
                if mode not in modes_this_run:
                    continue
                store = done_open if mode == "open" else done_closed
                ctx = ctx_cache[qid] if mode == "open" else empty_ctx()
                old = store.get(qid)
                if old is None:
                    reasons["新增"] += 1
                    tasks.append((mode, item, ctx, MAX_TOKENS))
                    continue
                if record_is_current(m, mode, item, old, want_fp, ctx):
                    continue          # [!] 有效记录（含高额度记录）一律不动，绝不降级重答
                if old.get("error"):
                    reasons["上次失败重试"] += 1
                else:
                    reasons["输入已变更重答"] += 1
                # [!] 重答额度只增不减：内容真的变了也要用 ≥ 该记录已用过的额度，
                #     否则会把一条已经答完整的回答换成可能被截断的短额度版本。
                tasks.append((mode, item, ctx, max(rec_own_budget(old), MAX_TOKENS)))

        print(f"=== {m['display']} ({m['name']}) 已有 开卷{len(done_open)} / 闭卷{len(done_closed)}；"
              f"本次待答 {len(tasks)} 条（新增 {reasons['新增']}｜上次失败重试 {reasons['上次失败重试']}｜"
              f"输入已变更重答 {reasons['输入已变更重答']}）===")

        def save():
            write_jsonl(f_open_full, [done_open[k] for k in sorted(done_open)])
            write_jsonl(f_closed, [done_closed[k] for k in sorted(done_closed)])

        if tasks:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futs = {ex.submit(answer_one, m, mode, item, ctx, budget): (mode, item)
                        for mode, item, ctx, budget in tasks}
                for k, fut in enumerate(as_completed(futs), 1):
                    mode, item = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:            # 兜底：任何异常都不让整批停掉
                        rec = {"model": m["display"], "model_name": m["name"], "mode": mode,
                               "question_id": item["question_id"], "type": item["type"],
                               "answerable": item["answerable"], "question": item["question"],
                               "reference": item["reference_answer"],
                               "response": f"[异常] {type(e).__name__}: {e}", "error": True,
                               # [!] 异常分支也要写额度/attempts：缺了它们，
                               #     记录额度扫描会把这条当成"默认额度"，续跑判定也会漂移。
                               "max_tokens_used": MAX_TOKENS, "attempts": [],
                               "attempt_budgets": [MAX_TOKENS], "finish_reason": "",
                               "prompt_ver": PROMPT_VER, "fp": want_fp.get((mode, item["question_id"]), ""),
                               "latency_s": 0, "in_tokens": 0, "out_tokens": 0, "cost": 0}
                    (done_open if mode == "open" else done_closed)[item["question_id"]] = rec
                    if k % 10 == 0 or k == len(tasks):
                        save()
                        print(f"  {k}/{len(tasks)} 完成")
            total_answered += len(tasks)

        # ---------- ③ 清理：只清"本次选中 + 本次跑的模式"里指纹失效的记录 ----------
        # [!] 同样复用 record_is_current：清理用的判据必须与规划、导出**完全一致**，
        #     否则会出现"规划说有效、清理把它删掉"这种自相矛盾（旧代码正是两套判据）。
        dropped = 0
        for store, mode in ((done_open, "open"), (done_closed, "closed")):
            for qid in list(store):
                if qid not in all_qa_ids:            # 题目确实已从题库删除
                    del store[qid]
                    dropped += 1
                    continue
                if qid not in scope_ids or mode not in modes_this_run:
                    continue                          # 范围外 / 非本次模式：一律不动
                if not record_is_current(m, mode, all_by_id[qid], store[qid], want_fp,
                                         ctx_cache[qid] if mode == "open" else empty_ctx()):
                    del store[qid]
                    dropped += 1
        save()

        open_rows = [done_open[k] for k in sorted(done_open)]
        closed_rows = [done_closed[k] for k in sorted(done_closed)]

        # ---------- ④ 产物只收"指纹有效"的记录（旧记录不进评测输入）----------
        # [!] 传入**当前**检索上下文：产物是给评测用的，必须与当前输入一致，
        #     否则会把"基于旧片段"的回答写进榜单。
        # [!] 必须容忍"范围外记录"：--limit 试跑时 ctx_cache 只装本次选中的题，
        #     范围外的记录仍留在 store 里（按安全约定不删除），不能用 ctx_cache[qid] 直接取键。
        ok_open = [r for r in open_rows
                   if record_is_current(m, "open", all_by_id[r["question_id"]], r, want_fp,
                                        ctx_cache.get(r["question_id"]))]
        ok_closed = [r for r in closed_rows
                     if record_is_current(m, "closed", all_by_id[r["question_id"]], r, want_fp,
                                          empty_ctx())]
        excl = (len(open_rows) - len(ok_open)) + (len(closed_rows) - len(ok_closed))
        total_excluded += excl
        old_kbs = sorted({r.get("kb_fp") for r in open_rows + closed_rows
                          if r.get("kb_fp") and r.get("kb_fp") != kb_fp()})
        kb_stale = sum(1 for r in open_rows + closed_rows
                       if not r.get("error") and r.get("kb_fp") and r.get("kb_fp") != kb_fp())

        rag_rows = [{"user_input": r["question"], "retrieved_contexts": r["retrieved_contexts"],
                     "response": r["response"], "reference": r["reference"]}
                    for r in ok_open if r["response"]]
        (runs_dir / f"{s}__open.json").write_text(
            json.dumps(rag_rows, ensure_ascii=False, indent=1), encoding="utf-8")

        cd = path("qa_set").parent / "closed"
        cd.mkdir(parents=True, exist_ok=True)
        # [!] 字段名必须与 EvalScope 的 general_qa 适配器对齐：它取
        #     query = record['question'] or record['query']；answer = record['answer'] or record['response']
        #     并把 answer 当作**标准答案（target）**。
        #     早期写成 {"response": 模型回答, "reference": 标准答案} → EvalScope 会把**模型旧回答当标准答案**
        #     （等于自己评自己），是一条必然触发的字段映射错误。
        #     现在只写语义明确的 question/answer；模型回答的权威记录在 runs/*_closed_full.jsonl。
        write_jsonl(cd / f"{s}.jsonl",
                    [{"question": r["question"], "answer": r["reference"]}
                     for r in ok_closed])

        # 费用要把**开卷与闭卷都算上**：旧版只统计 open_rows，导致总花费被少算一半
        c_open = sum(r.get("cost", 0) for r in open_rows)
        c_closed = sum(r.get("cost", 0) for r in closed_rows)
        c = c_open + c_closed
        total_cost += c
        total_cost_open += c_open
        total_cost_closed += c_closed
        err = sum(1 for r in open_rows + closed_rows if r["error"])
        blank = sum(1 for r in open_rows + closed_rows
                    if not (r.get("response") or "").strip())
        print(f"  [OK] {m['display']}：开卷有效 {len(ok_open)}/{len(open_rows)} 条，"
              f"闭卷有效 {len(ok_closed)}/{len(closed_rows)} 条，失败 {err} 条"
              + (f"（其中空作答 {blank} 条）" if blank else "") + "，"
              f"清理失效记录 {dropped} 条，排除未复核记录 {excl} 条，"
              f"估算花费 {c:.3f} 元（开卷 {c_open:.3f} + 闭卷 {c_closed:.3f}）")
        if excl:
            print(f"       [!] 有 {excl} 条记录未通过指纹复核（改了题/换了提示词/重建了知识库），"
                  f"未写入评测产物。")
        if kb_stale:
            print(f"       [!!] 其中 {kb_stale} 条来自**旧知识库版本**"
                  f"（{'、'.join('kb-' + k for k in old_kbs[:3])}），当前知识库是 kb-{kb_fp()}。")
            print(f"       [!!] 知识库已变更 → 这些历史答案与新库**不可比**。")
            print(f"       [!!] 请**不加 --limit / --mode 全量重跑**后再出榜单，否则榜单会混入旧数据。")
        print()

    print(f"本次共答 {total_answered} 条，估算总花费 {total_cost:.3f} 元"
          f"（开卷 {total_cost_open:.3f} + 闭卷 {total_cost_closed:.3f}）")
    print("说明：费用 = config.yaml 里的单价 × 实际 token 用量，是**估算**，不是账单金额。")
    if total_answered == 0:
        print("提示：本次没有需要重答的记录（全部命中指纹缓存）。")
    print("下一步：.\\run-eval.ps1 src\\6_eval_evalscope.py --only mcq（先拿最稳的分数）")


if __name__ == "__main__":
    main()
