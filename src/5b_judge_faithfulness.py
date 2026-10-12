# -*- coding: utf-8 -*-
"""
第 5b 步：忠实性评测（不依赖 RAGAS / torch 的轻量方案）

为什么需要它：
  EvalScope 的 RAGEval(RAGAS) 后端要拉 langchain + sentence-transformers + torch（约 1GB），
  是 3 天冲刺里最容易失败的一环。这个脚本用"裁判模型逐条判定"实现同类指标：
  每条开卷回答都问裁判一句"你的事实性内容是否都有文献依据"。
  成本约 2~4 元，而且**逐条可解释**（哪句话没依据会被写出来），比黑箱指标更好讲。

产出：
  outputs/faithfulness.jsonl   逐条判定明细（可复现、可深挖）
  outputs/忠实性汇总.md         按模型 + 按题型的忠实率表（直接贴进报告）

用法：
  .\run-eval.ps1 src\5b_judge_faithfulness.py                # 全部模型
  .\run-eval.ps1 src\5b_judge_faithfulness.py --model 豆包 --limit 5   # 先小样本试
"""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import CFG, ROOT, chat, dirpath, kb_fingerprint, path, read_jsonl, write_jsonl

PROMPT = """你是严格的医学评测裁判。请判断【模型回答】里的事实性内容是否**都有**文献依据。

判定规则：
- 忠实：回答中的结论、数值、适用范围都能在下面片段里找到依据（表述不同但意思一致，算忠实）。
- 部分无依据：主要结论有依据，但夹带了片段中没有的数值、结论或治疗推荐。
- 无依据：核心内容在片段里找不到支撑，属于编造。
- 特殊情况 1：若回答明确表示"文献中没有相关内容 / 无法回答"，而片段确实没有该信息 → 判"忠实"。
- 特殊情况 2（选择题，重要）：题目带 A~D 选项，模型可能只回答"答案：X"。
  **字母本身不是事实性内容，绝不能因为"片段里没出现字母 X"就扣分**；
  此时请按该选项的**文字内容**是否有片段依据来判断：有→忠实；部分有→部分无依据；完全没有→无依据。
  若回答只给了字母、没有任何说理，而该选项内容确有依据 → 判"忠实"，并在 unsupported 里注明"仅给出选项字母"。

只输出一个 JSON 对象，不要任何解释文字：
{{"judge": "忠实" 或 "部分无依据" 或 "无依据", "unsupported": "指出哪句没有依据；没有则填空字符串"}}

【问题】
{question}

【题目选项】（非选择题则为空）
{options}

【检索到的文献片段】
{contexts}

【模型回答】
{answer}
"""

def guard_prompt():
    """启动自检：提示词里的 JSON 示例必须写成 {{ }}（转义）。
    否则 .format() 会抛异常，所有判定会静默变成"解析失败"——这种错误很难发现。"""
    try:
        PROMPT.format(question="q", options="o", contexts="c", answer="a")
    except Exception as e:
        raise SystemExit(f"[X] 5b 提示词模板的转义有误（JSON 示例的花括号要写成 {{{{ }}}}）：{e}")


LABELS = ["忠实", "部分无依据", "无依据"]
VER = "v2"        # v2：把选择题选项传给裁判（此前只给字母 → 16 条被误判扣分）；改提示词时 +1


def build_prompt(r) -> str:
    ctx = "\n\n".join(f"[片段{i}] {t}" for i, t in enumerate(r.get("retrieved_contexts", []), 1))
    opts = r.get("options") or {}
    opt_txt = "\n".join(f"{k}. {opts[k]}" for k in "ABCD" if opts.get(k)) or "（本题没有选项）"
    return PROMPT.format(question=r["question"], options=opt_txt,
                         contexts=ctx, answer=r["response"])


def fp_of(r, judge_name: str) -> str:
    """判定指纹 = md5(完整判定提示词 + 裁判模型名 + 版本)。

    提示词里已经包含问题、**全部片段正文**、模型回答，所以任一改变都会重判；
    裁判模型换了也会重判（不同裁判的判定不能混在同一张表里）。
    """
    raw = json.dumps({"v": VER, "judge": judge_name, "prompt": build_prompt(r)},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _rag_answer_module():
    """加载 `4_rag_answer`，复用它的**作答输入指纹**定义（与 7_report.load_all 同一份实现）。

    [!] 为什么不能用简化版校验：本函数此前只查 kb_fp + 题号在正式集 + 回答非空，
        于是改题干/选项/标准答案/作答参数/模型配置之后，**旧实验回答仍会被忠实性流程接受**
        （第三方复验补充检查 A1-1 已复现：一条 fp 过期记录在 5b 里照样"输入版本校验通过"，
        而同一份数据送进 `7_report.load_all` 会被明确排除）。判据只有一处实现，不复制第二份。
    """
    import importlib.util
    p = ROOT / "src" / "4_rag_answer.py"
    spec = importlib.util.spec_from_file_location("faith_rag_answer_check", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def judge_one(j, r):
    prompt = build_prompt(r)
    text, usage = chat(j, [{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=3000)
    # [!] 实测：忠实性判定很容易把 1500 额度全用于"思考"并返回空内容，兜底重试会白烧一遍 token。
    #     所以这里直接给 3000（common.chat 对空内容仍有翻倍兜底）。
    verdict, reason, parse_mode = "解析失败", "", "json"
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        v = str(obj.get("judge", "")).strip()
        if v in LABELS:
            verdict, reason = v, str(obj.get("unsupported", ""))[:200]
        else:
            parse_mode = "json-标签无效"
    except Exception:
        parse_mode = "json-解析异常"
    if verdict not in LABELS:
        # [!] 容错**只认「带判定字段名的标签」**，且必须**唯一**：
        #     · 收集全部判定字段（不能只取第一个匹配）；
        #     · 全部一致才接受；**存在冲突就返回"解析失败"**（下次运行会自动重判），绝不猜。
        #     反例 1："判定：部分无依据，不能判为忠实。" → 只有字段后那个标签有效（=部分无依据）
        #     反例 2："判定：忠实。判定：无依据。" → 冲突 → 解析失败
        found = re.findall(r"(?:judge|判定|结论)\s*[:：]\s*[\"'「『]?\s*(忠实|部分无依据|无依据)(?!\w)",
                           text or "")
        uniq = sorted(set(found))
        if len(uniq) == 1:
            verdict = uniq[0]
            reason = reason or (text or "").strip()[:200]
            parse_mode = f"文本字段匹配(唯一{len(found)}处)"
        elif uniq:
            parse_mode = f"文本字段冲突({uniq})"
    return {"model": r["model"], "question_id": r["question_id"], "type": r["type"],
            "answerable": r["answerable"], "verdict": verdict, "unsupported": reason,
            # 只要不是走的严格 JSON 路径，就保存原始输出与解析方式，保证可追溯
            "parse_mode": parse_mode,
            "raw_head": (text or "")[:200] if parse_mode != "json" else "",
            "fp": fp_of(r, j["name"]),
            "in_tokens": usage.get("in_tokens", 0), "out_tokens": usage.get("out_tokens", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true",
                    help="开发模式：判定 dev/runs 下的答题记录，产物写入 dev/")
    ap.add_argument("--model", default=None, help="只评名字里含这个字符串的模型")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 条（试跑用）")
    ap.add_argument("--workers", type=int, default=4, help="并发数（默认 4，被限流就改 1）")
    args = ap.parse_args()
    if args.dev:
        import common
        common.enable_dev()

    rows_all = []
    # [!] 版本校验（原缺口）：只判**当前配置下有效**的回答记录。
    #     不做这一步，旧知识库/旧题集的历史记录会被判分并混进汇总。
    #     [!] 2026-10-12 补齐（第三方复验补充检查 A1-1）：判据与 `7_report.load_all()` **完全同源**——
    #         除 kb_fp / 题号 / 非空之外，还要**模型在当前 config 内**且
    #         **用 4_rag_answer.record_is_current 复算作答输入指纹**（覆盖题干、A~D 选项、
    #         标准答案、提示词版本、temperature/max_tokens、检索片段、知识库指纹）。
    #         原实现缺后两项 → 改题干/选项/标准答案/作答参数/模型配置后，
    #         旧实验回答会被忠实性流程静默接受（同源缺口，已实测）。
    kb_now = kb_fingerprint()
    qa_items = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    models = {m["name"]: m for m in CFG["models"]}
    try:
        ra = _rag_answer_module()
    except Exception as e:
        # [!] 校验能力缺失同样要**终止**，不做宽松回退（宽松回退会把旧记录当新记录判分）。
        raise SystemExit(f"[X] 无法加载 4_rag_answer 做作答输入指纹复核（{type(e).__name__}: {e}）"
                         f"→ 终止忠实性判定，不做宽松回退。")
    why = {}
    for f in sorted(dirpath("runs").glob("*_full.jsonl")):
        for r in read_jsonl(f):
            if r.get("mode") != "open":
                continue
            if r.get("error") or not (r.get("response") or "").strip():
                why["失败或空作答"] = why.get("失败或空作答", 0) + 1
                continue
            if r.get("kb_fp") != kb_now:
                why["知识库指纹缺失或不是当前库"] = why.get("知识库指纹缺失或不是当前库", 0) + 1
                continue
            item = qa_items.get(r.get("question_id"))
            if item is None:
                why["题目不在当前正式集里"] = why.get("题目不在当前正式集里", 0) + 1
                continue
            m = models.get(r.get("model_name"))
            if m is None:
                why["模型不在当前 config 里"] = why.get("模型不在当前 config 里", 0) + 1
                continue
            try:
                ok = ra.record_is_current(m, "open", item, r, {})
            except Exception:
                ok = False
            if not ok:
                why["作答输入指纹与当前题干/选项/提示词/参数不一致"] = \
                    why.get("作答输入指纹与当前题干/选项/提示词/参数不一致", 0) + 1
                continue
            rows_all.append(r)
    if why:
        print(f"[!] 5b 跳过 {sum(why.values())} 条与当前配置不一致的记录（不判分、不混入汇总）：")
        for k, v in why.items():
            print(f"    - {k}：{v} 条")
    else:
        # [!] 显式断言：0 条被跳过时也要打印，否则日志无法区分"校验通过"与"校验没跑"
        print(f"[OK] 输入版本校验通过：{len(rows_all)} 条（知识库指纹 kb-{kb_now}、题号全部在当前正式集、"
              f"模型在当前 config、**作答输入指纹逐条复算一致**）")
    if args.model:
        rows_all = [r for r in rows_all
                    if args.model.lower() in (r["model"] + r.get("model_name", "")).lower()]
    # --limit 只影响"本次要判哪些"，绝不影响"已有判定是否保留"
    rows = rows_all[: args.limit] if args.limit else rows_all
    if not rows:
        raise SystemExit("runs/ 里没有开卷回答 —— 先跑 4_rag_answer.py")

    guard_prompt()
    j = CFG["judge"]
    out_file = ROOT / CFG["paths"]["outputs"] / "faithfulness.jsonl"
    done = {(r["model"], r["question_id"]): r for r in read_jsonl(out_file)}
    row_by = {(r["model"], r["question_id"]): r for r in rows}          # 本次范围
    keys_all = {(r["model"], r["question_id"]) for r in rows_all}       # 同模型的全量范围
    todo, dropped = [], 0
    for key, r in row_by.items():
        old = done.get(key)
        if old is None:
            todo.append(r)
        elif old.get("verdict") not in LABELS or old.get("fp") != fp_of(r, j["name"]):
            del done[key]                     # 判定失效：回答/片段/裁判变了
            dropped += 1
            todo.append(r)
    # 只清理"同模型全量范围内、但已不在 runs 里"的记录；范围外的记录一律保留
    for key in list(done):
        if key in row_by or key in keys_all:
            continue
        if key[0] in {r["model"] for r in rows_all}:
            del done[key]
            dropped += 1

    j = CFG["judge"]
    print(f"裁判模型：{j['display']}({j['name']})｜待判 {len(todo)} 条"
          f"（有效缓存 {len(done)} 条，作废重判 {dropped} 条，并发 {args.workers}）\n")

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(judge_one, j, r): r for r in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    rec = fut.result()
                except Exception as e:
                    r = futs[fut]
                    rec = {"model": r["model"], "question_id": r["question_id"], "type": r["type"],
                           "answerable": r["answerable"], "verdict": "解析失败",
                           "unsupported": f"{type(e).__name__}: {e}"[:200],
                           "in_tokens": 0, "out_tokens": 0}
                done[(rec["model"], rec["question_id"])] = rec
                if k % 10 == 0 or k == len(todo):
                    write_jsonl(out_file, [done[key] for key in sorted(done)])
                    print(f"  {k}/{len(todo)} 完成")
    write_jsonl(out_file, [done[key] for key in sorted(done)])
    allrows = list(done.values())

    # ---------- 汇总 ----------
    models = sorted({r["model"] for r in allrows})
    lines = ["# 忠实性汇总（裁判模型逐条判定）", "",
             f"- 裁判模型：{j['display']}（{j['name']}）",
             f"- 判定条目：{len(allrows)} 条；判定口径：忠实 / 部分无依据 / 无依据",
             "- 说明：这是 RAGEval-RAGAS 忠实性指标的轻量替代实现（逐条可解释）", "",
             "## 按模型", "", "| 模型 | 条数 | 忠实率 | 部分无依据 | 无依据 | 解析失败 |",
             "|---|---|---|---|---|---|"]
    print(f"\n{'模型':<18}{'条数':>5}{'忠实率':>9}{'部分无依据':>11}{'无依据':>8}")
    for m in models:
        rs = [r for r in allrows if r["model"] == m]
        n = len(rs)
        cnt = {lab: sum(1 for r in rs if r["verdict"] == lab) for lab in LABELS}
        fail = sum(1 for r in rs if r["verdict"] not in LABELS)
        ok = cnt["忠实"] / n * 100 if n else 0
        lines.append(f"| {m} | {n} | **{ok:.1f}%** | {cnt['部分无依据']} | {cnt['无依据']} | {fail} |")
        print(f"{m:<18}{n:>5}{ok:>8.1f}%{cnt['部分无依据']:>11}{cnt['无依据']:>8}")

    # 按题型（数值题最容易编，单列出来是报告亮点）
    lines += ["", "## 按题型（数值题最容易编造，单列出来看）", "",
              "| 题型 | 条数 | 忠实率 | 无依据 |", "|---|---|---|---|"]
    for t in sorted({r["type"] for r in allrows}):
        rs = [r for r in allrows if r["type"] == t]
        n = len(rs)
        ok = sum(1 for r in rs if r["verdict"] == "忠实") / n * 100 if n else 0
        bad = sum(1 for r in rs if r["verdict"] == "无依据")
        lines.append(f"| {t} | {n} | {ok:.1f}% | {bad} |")

    # 典型"无依据"例子（面试讲案例用）
    bads = [r for r in allrows if r["verdict"] == "无依据" and r.get("unsupported")]
    if bads:
        lines += ["", "## 典型“无依据”案例（可直接在面试里讲）", ""]
        for r in bads[:8]:
            lines.append(f"- **{r['model']}** / {r['question_id']}（{r['type']}）：{r['unsupported']}")

    out_md = ROOT / CFG["paths"]["outputs"] / "忠实性汇总.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] 明细 → {out_file}")
    print(f"[OK] 汇总 → {out_md}")
    print("\n下一步：把它并进报告（或先跑 EvalScope 的 6_eval_evalscope.py 拿准确率）")


if __name__ == "__main__":
    main()
