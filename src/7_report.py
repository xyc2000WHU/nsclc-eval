# -*- coding: utf-8 -*-
"""
第 7 步：汇总出榜单 + 生成"人工抽检表"（这是整个项目最有说服力的一步）。

做五件事：
  1. 从 runs/ 里算出可客观统计的指标：检索命中率、正确拒答率、过度自信率、延迟、成本
  2. 抓取 outputs/ 里 EvalScope 的分数（并生成手工填报表，你核对后填进去）
  3. 分层抽 30 条回答 → 生成 qa/人工抽检表.csv，并用裁判模型预填"自动判定"
  4. 你填完"人工判定"后再跑一次 → 自动算 人工 vs 自动 一致率与 Cohen's Kappa
  5. 并入 outputs/faithfulness.jsonl（第 5b 步的忠实性判定），并统计医学正确性确认数

[!] 原则：**报告只写实际跑过的东西**。开头"实际使用的评测方法"是根据 outputs/ 里的
    真实产物动态生成的；没检测到 RAGEval/RAGAS 的产物，报告就会明确写"本次未使用
    RAGEval/RAGAS"，不会默认声称用了它。

用法：
  python src\7_report.py                # 生成榜单 + 抽检表
  python src\7_report.py --no-judge     # 不调用裁判模型（省几分钱）
  python src\7_report.py                # 你填完抽检表后再跑一次，会自动算出 Kappa
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

from common import (CFG, ROOT, chat, dirpath, full_question, g_col, kappa_of,
                    kb_fingerprint, path, read_csv_any, read_jsonl)

REFUSAL_WORDS = ["没有相关", "未提及", "没有提到", "无法回答", "不能回答", "无法确定",
                 "没有足够", "未提供", "不包含", "无法从", "文献中没有", "不确定",
                 "无法判断", "没有明确", "未给出"]

JUDGE_PROMPT = """你是医学评测裁判。请判断【模型回答】与【标准答案】在医学语义上是否一致。

判定边界（**严格按这三条**，2026-10-10 项目负责人确定，见 docs/评分口径_正确性.md）：
- 一致：核心结论正确，覆盖题目要求的必要要点；允许简短、同义表达；**不因缺少题目没问的扩展信息扣分**。
- 部分一致：核心结论正确且无重大错误，但**遗漏题目明确要求的次要要点**，或存在**不影响核心判断的局部错误**。
- 不一致：核心结论错误、**遗漏决定答案成立的关键条件**、出现重大错误，或**没有实质回答**。

补充规则：
- 选择题若只给选项字母（如"答案：B"）：按该选项的**文字内容**与标准答案比较；字母正确即算答对，
  除非其解释与标准答案矛盾。
- 陷阱题：若标准答案说"所提供的文献中没有相关内容"，而模型也明确表示无法回答/文献未提及 → 判"一致"；
  模型给出具体编造内容 → 判"不一致"；既说资料不足又顺带给出具体方案的，按上面三条边界判。

只输出一个 JSON 对象，不要任何解释文字：
{{"judge": "一致" 或 "部分一致" 或 "不一致", "reason": "一句话理由"}}

【问题】
{question}

【标准答案】
{reference}

【模型回答】
{response}
"""


def guard_prompt():
    """启动自检：判定提示词里的 JSON 示例必须写成 {{ }}（转义），
    否则 .format() 抛异常 → 抽检表的"自动判定"会全部变成解析失败，Kappa 永远算不出来。"""
    try:
        JUDGE_PROMPT.format(question="q", reference="r", response="s")
    except Exception as e:
        raise SystemExit(f"[X] 7_report 判定提示词的转义有误（JSON 示例的花括号要写成 {{{{ }}}}）：{e}")


# ---------------- 指标计算 ----------------
def is_refusal(text: str) -> bool:
    return any(w in (text or "") for w in REFUSAL_WORDS)


def _rag_answer_module():
    """加载 4_rag_answer 模块，复用它的指纹定义（避免两处实现漂移）。"""
    import importlib.util
    p = ROOT / "src" / "4_rag_answer.py"
    spec = importlib.util.spec_from_file_location("rag_answer_for_check", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_all(model_glob="*_full.jsonl"):
    """读取 runs/ 下的原始记录，并做**完整的版本一致性校验**。

    [!] 为什么必须校验：汇总脚本会把目录下所有记录算进去，此前跑过的题集/知识库/提示词版本
        都会静默混进榜单。这里逐条要求：
          ① `kb_fp` **存在且等于**当前知识库指纹（缺失也算不合规）；
          ② 题目 ID 在当前正式评测集里；
          ③ 模型在当前 config 里；
          ④ **用与 4_rag_answer 完全相同的定义复算输入指纹**（覆盖题干、选择题选项、标准答案、
             提示词版本 PROMPT_VER、temperature/max_tokens、检索片段、知识库指纹）。
        这样"同题号但改了题干/选项/提示词"的旧记录也会被拦下。
        被排除的记录**打印条数与原因**，绝不静默丢弃。
    """
    rows_all = []
    for f in sorted(dirpath("runs").glob(model_glob)):
        rows_all += read_jsonl(f)

    kb_now = kb_fingerprint()
    qa_items = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    models = {m["name"]: m for m in CFG["models"]}
    try:
        ra = _rag_answer_module()
    except Exception as e:
        # [!] 模块加载失败同样是"校验能力缺失"→ 终止，不做宽松回退（宽松回退会把旧记录当新记录）。
        raise SystemExit(f"[X] 无法加载 4_rag_answer 做指纹复核（{type(e).__name__}: {e}）"
                         f"→ 终止汇总，不做宽松回退。")

    keep, why = [], Counter()
    for r in rows_all:
        if r.get("kb_fp") != kb_now:
            why["知识库指纹缺失或不是当前库"] += 1
            continue
        item = qa_items.get(r.get("question_id"))
        if item is None:
            why["题目不在当前正式评测集里"] += 1
            continue
        m = models.get(r.get("model_name"))
        if m is None:
            why["模型不在当前 config 里"] += 1
            continue
        if ra is not None:
            mode = r.get("mode") or "open"
            try:
                ok = ra.record_is_current(m, mode, item, r, {})
            except Exception:
                ok = False
            if not ok:
                why["输入指纹与当前题干/选项/提示词/参数不一致"] += 1
                continue
        keep.append(r)

    if why and not keep and rows_all:
        # [!] **绝不自动放宽校验**：全部记录过期往往是"合法地全过期"（例如改了全局提示词、
        #     生成参数或重建了知识库）。此时放宽就会把**旧实验当成当前实验**汇总。
        #     正确做法：终止本次正式报告生成（原报告保留），只打印诊断。
        print("[X] 所有记录都未通过版本校验 → 终止本次正式报告生成（原榜单不会被覆盖）。")
        for k, v in why.items():
            print(f"     - {k}：{v} 条")
        raise SystemExit(
            "    这通常意味着题干/选项/提示词/参数/知识库已变更：\n"
            "    请先重跑 python src\\4_rag_answer.py（全量），再汇总。\n"
            "    诊断已在上方打印，未写入任何榜单。")
    if why:
        print(f"[!] 汇总时排除了 {sum(why.values())} 条与当前配置不一致的历史记录（未混入榜单）：")
        for k, v in why.items():
            print(f"    - {k}：{v} 条")
    return keep


def summarize(rows):
    """每个模型的客观指标。"""
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    out = {}
    for model, rs in by.items():
        op = [r for r in rs if r["mode"] == "open" and not r.get("error")]
        cl = [r for r in rs if r["mode"] == "closed" and not r.get("error")]
        traps = [r for r in op if not r["answerable"]]
        # [!] 检索命中率的分母只算**有登记证据的可答题**：10 道陷阱题没有"来源片段"，
        #     把它们算进分母会把命中率人为拉低（曾把 30% 显示成 26.7%）。
        ans = [r for r in op if r["answerable"]]
        out[model] = {
            "题数(开卷)": len(op),
            "题数(闭卷)": len(cl),
            "检索命中率@topk(可答题)": round(sum(1 for r in ans if r.get("gold_hit")) / len(ans) * 100, 1) if ans else 0,
            "陷阱题数": len(traps),
            "正确拒答率": round(sum(1 for r in traps if is_refusal(r["response"])) / len(traps) * 100, 1) if traps else 0,
            "过度自信率": round(sum(1 for r in traps if not is_refusal(r["response"])) / len(traps) * 100, 1) if traps else 0,
            "平均耗时(秒)": round(sum(r.get("latency_s", 0) for r in op) / len(op), 1) if op else 0,
            "平均输出token": round(sum(r.get("out_tokens", 0) for r in op) / len(op)) if op else 0,
            "估算成本(元)": round(sum(r.get("cost", 0) for r in rs), 2),
        }
    return out


# ---------------- 抓 EvalScope 的分数 ----------------
def harvest_scores():
    """在 outputs/ 里递归找形如 {'metric': ..., 'score': 数字} 的记录。"""
    found = []
    out_dir = ROOT / CFG["paths"]["outputs"]
    if not out_dir.exists():
        return found

    def visit(obj, ctx=None):
        if isinstance(obj, dict):
            score = None
            for k, v in obj.items():
                if k.lower() in ("score", "metric_score", "average", "avg") and isinstance(v, (int, float)):
                    score = float(v)
                    break
            if score is not None:
                name = None
                for k in ("metric", "name", "dataset", "dataset_name", "subset", "model"):
                    if isinstance(obj.get(k), str):
                        name = obj[k]
                        break
                found.append({"来源": ctx or "", "指标": name or "未知", "分数": round(score, 4)})
            for k, v in obj.items():
                visit(v, k)
        elif isinstance(obj, list):
            for v in obj:
                visit(v, ctx)

    for f in out_dir.rglob("*.json"):
        try:
            if f.stat().st_size > 6_000_000:
                continue
            visit(json.loads(f.read_text(encoding="utf-8")), f.parent.name)
        except Exception:
            continue
    return found[:400]


def manual_scores_file():
    """手工汇总表：从 EvalScope 终端表格抄进来，最可靠。"""
    f = ROOT / CFG["paths"]["outputs"] / "手工汇总.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "dataset", "metric", "score"])
            # 只写表头。把 EvalScope 终端打印的表格抄成下面这种行（一行一个分数）：
            # DeepSeek,RAG开卷,Faithfulness,0.85
        print(f"[i] 已创建空白汇总表：{f}\n  请把 EvalScope 打印的分数按 model,dataset,metric,score 四列抄进去")
        return None
    rows = read_csv_any(f)
    # [!] 兼容旧列名：这里吃的是**用户手工填的**表，列名可能带空格/大小写差异 → 统一走 g_col
    out = []
    for r in rows:
        model = g_col(r, "model")
        score = g_col(r, "score")
        if model and score:
            try:
                out.append((model, g_col(r, "dataset"), g_col(r, "metric"), float(score)))
            except ValueError:
                pass
    return out or None


# ---------------- 人工抽检 ----------------
def kappa(a_labels, b_labels):
    """Cohen's Kappa（委托 common.kappa_of，全项目统一退化约定）。

    [!] 原实现 pe>=1 时返回 1.0，而 7b/7d 在同一情形返回 0.0 —— 同一个退化样本
        （双方判定完全同分布）会在两份产物里给出**相反**的结论。现统一：解析失败等
        非标签判定一律排除；pe>=1 时 po==1 → 1.0，否则 0.0。
    """
    k, _, _ = kappa_of(list(zip(a_labels, b_labels)), CORR_LABELS)
    return k


CORR_LABELS = ["一致", "部分一致", "不一致"]


def make_spot_check(use_judge=True, n_each=10):
    spot = path("qa_set").parent / "人工抽检表.csv"
    meta = spot.with_suffix(".meta.json")
    if spot.exists():
        # [!] 版本绑定（原缺口）：抽检表是"某一版数据 + 某一版裁判"下生成的。若知识库或裁判换了，
        #     重新汇总会把旧表的判定当作当前配置的结果 → 必须显式警告，不能静默复用。
        try:
            old = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        except Exception:
            old = {}
        cur_kb, cur_j = kb_fingerprint(), CFG["judge"]["name"]
        if not old:
            meta.write_text(json.dumps({"kb_fp": cur_kb, "judge": cur_j,
                                        "note": "补写的绑定信息（原表生成时未记录）"},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[i] 已补写抽检表绑定信息 → {meta.name}")
        elif old.get("kb_fp") != cur_kb or old.get("judge") != cur_j:
            print(f"[!] 抽检表是**旧配置**下生成的（表内 kb_fp/judge = "
                  f"{old.get('kb_fp')}/{old.get('judge')}；当前 = {cur_kb}/{cur_j}）"
                  f"→ 它算出的 Kappa 不能代表当前配置；请改名或删除后重新生成。")
        return spot                      # 已存在就别覆盖（你可能已经填了一部分）

    rows = [r for r in load_all() if r["mode"] == "open" and not r.get("error") and r["response"]]
    if not rows:
        return None

    rng = random.Random(7)
    def pick(pred, k):
        pool = [r for r in rows if pred(r)]
        rng.shuffle(pool)
        return pool[:k]

    sample = (pick(lambda r: r["type"] == "number", n_each)
              + pick(lambda r: not r["answerable"], n_each)
              + pick(lambda r: r["type"] in ("fact", "guideline", "reasoning"), n_each))
    # 去重
    seen, s = set(), []
    for r in sample:
        key = (r["model"], r["question_id"])
        if key not in seen:
            seen.add(key)
            s.append(r)

    j = CFG["judge"]
    print(f"抽 {len(s)} 条回答做人工抽检" + ("，并用裁判模型预填自动判定……" if use_judge else "……"))
    out = path("qa_set").parent / "人工抽检表.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "题目ID", "模型", "题型", "是否可答", "问题", "标准答案",
                    "原文证据句", "模型回答", "自动判定", "人工判定(填:一致/部分一致/不一致)",
                    "错误类型(填:数值错误/结论错误/推理错误/回答不完整/该答却拒答/超出文献/证据不足/人群范围错配/剂量单位错误/其他)",
                    "成因(填:模型没读出证据/检索没给到/知识库或题目缺/不确定)", "备注"])
        for i, r in enumerate(s, 1):
            auto = ""
            if use_judge:
                # [!] 必须传**完整题面（含选项文字）**：提示词里写明"选择题若只给选项字母，
                #     按该选项的文字内容比较"，只传裸题干时裁判看不到选项，只能猜。
                qa_item = next((x for x in read_jsonl(path("qa_set"))
                                if x["question_id"] == r["question_id"]), {})
                text, _ = chat(j, [{"role": "user", "content": JUDGE_PROMPT.format(
                    question=full_question(qa_item or r), reference=r["reference"],
                    response=r["response"][:1500])}],
                    temperature=0.0, max_tokens=1500)   # 推理型裁判额度要够（common.chat 有兜底重试）
                try:
                    auto = json.loads(text[text.find("{"): text.rfind("}") + 1]).get("judge", "")
                except Exception:
                    auto = "解析失败"
            qa_row = next((x for x in read_jsonl(path("qa_set")) if x["question_id"] == r["question_id"]), {})
            w.writerow([i, r["question_id"], r["model"], r["type"],
                        "是" if r["answerable"] else "否", full_question(qa_row or r), r["reference"],
                        qa_row.get("evidence_quote", ""), r["response"][:800],
                        auto, "", "", ""])
    print(f"[OK] 抽检表 → {out}")
    return out


def read_spot_check():
    """读回人工填写的抽检表（含 GBK 兜底与旧列名兼容）。"""
    f = path("qa_set").parent / "人工抽检表.csv"
    if not f.exists():
        return None
    # [!] 用户的表是 Excel 存的 GBK（实测首字节 0xd0 0xf2）：只认 utf-8-sig 会 UnicodeDecodeError 崩溃。
    rows = read_csv_any(f)
    got = []
    for r in rows:
        # [!] 「错误类型」必须按**前缀**匹配：用户的表用的是旧列名
        #     「错误类型(填:数值误读/结论外推/…)」，而代码写死了新列名 →
        #     精确匹配会把这 7 条已填的错误类型**静默丢弃**（报告里那一节凭空消失）。
        got.append((g_col(r, "自动判定"), g_col(r, "人工判定(填:一致/部分一致/不一致)"),
                    g_col(r, "错误类型", prefix=True)))
    got = [g for g in got if g[1]]          # 只要填了「人工判定」就算有效
    return got or None


# ---------------- 忠实性 & 实际使用的评测方法 ----------------
def load_faithfulness():
    """读第 5b 步的忠实性判定（outputs/faithfulness.jsonl）；没有就返回 None。

    [!] 版本校验（原缺口）：每条判定必须与**当前回答 + 当前裁判提示词**复算的指纹一致，
        否则改了回答、换了提示词版本或重建知识库后，旧判定会混进报告。
    """
    rows = read_jsonl(ROOT / CFG["paths"]["outputs"] / "faithfulness.jsonl")
    if not rows:
        return None
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            "j5bchk", ROOT / "src" / "5b_judge_faithfulness.py")
        fb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fb)
    except Exception as e:
        raise SystemExit(f"[X] 无法加载 5b 做忠实性指纹复核（{type(e).__name__}: {e}）"
                         f"→ 终止汇总，不做宽松回退。")
    recs = {}
    for f in sorted(dirpath("runs").glob("*_open_full.jsonl")):
        for r in read_jsonl(f):
            recs[(r["model"], r["question_id"])] = r
    jname = CFG["judge"]["name"]
    keep, dropped = [], 0
    for v in rows:
        rec = recs.get((v.get("model"), v.get("question_id")))
        cur = None
        if rec is not None:
            try:
                cur = fb.fp_of(rec, jname)
            except Exception:
                cur = None
        if not cur or v.get("fp") != cur:
            dropped += 1
            continue
        keep.append(v)
    if dropped:
        print(f"[!] 忠实性判定：排除 {dropped} 条指纹与当前回答/裁判提示词不一致的历史记录"
              f"（剩余 {len(keep)} 条）")
    return keep or None


def faithfulness_md(rows):
    lines = ["## 4. 忠实性（LLM 裁判逐条判定，第 5b 步）\n",
             "> 这是 RAGEval/RAGAS 忠实性指标的轻量替代实现：逐条问裁判"
             "「这段回答的事实性内容是否都有文献依据」，因此逐条可解释。\n",
             "| 模型 | 条数 | 忠实率 | 部分无依据 | 无依据 |", "|---|---|---|---|---|"]
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        n = len(rs) or 1
        ok = sum(1 for r in rs if r["verdict"] == "忠实")
        part = sum(1 for r in rs if r["verdict"] == "部分无依据")
        bad = sum(1 for r in rs if r["verdict"] == "无依据")
        lines.append(f"| {m} | {len(rs)} | **{ok / n * 100:.1f}%** | {part} | {bad} |")
    types = sorted({r["type"] for r in rows})
    if len(types) > 1:
        lines += ["", "按题型拆分（数值题最容易编造，单列出来是报告亮点）：", "",
                  "| 题型 | 条数 | 忠实率 |", "|---|---|---|"]
        for t in types:
            rs = [r for r in rows if r["type"] == t]
            ok = sum(1 for r in rs if r["verdict"] == "忠实")
            lines.append(f"| {t} | {len(rs)} | {ok / (len(rs) or 1) * 100:.1f}% |")
    return lines


RAGAS_MARKERS = ("rageval", "ragas", "faithfulness", "answercorrectness", "contextrecall",
                 "answer_relevancy", "context_precision", "answer_correctness")


def ragas_candidates():
    """列出 outputs/ 里"路径像 RAGEval 产物"的文件（用于诊断：为什么没判定为已运行）。"""
    out = ROOT / CFG["paths"]["outputs"]
    hits = []
    if not out.exists():
        return hits
    for p in out.rglob("*"):
        if not p.is_file() or p.name == "手工汇总.csv":
            continue
        rel = str(p.relative_to(out)).lower()
        if "ragas" in rel or "rageval" in rel:
            hits.append(p)
    return hits


def has_ragas_artifacts() -> bool:
    """RAGEval/RAGAS 是否**真的跑过**。

    三个条件都要满足，避免误判：
      1. 文件所在路径含 ragas / rageval；
      2. 文件非空；
      3. 文件内容里出现 RAGEval/RAGAS 字样，或三个指标名之一。
    只凭手工汇总表里的指标名、或一个内容为 `{}` 的演示文件，都不算跑过。
    """
    for p in ragas_candidates():
        try:
            if p.stat().st_size == 0:
                continue
            blob = p.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
        except Exception:
            continue
        if any(mk in blob for mk in RAGAS_MARKERS):
            return True
    return False


def has_evalscope_report(fragment: str) -> bool:
    """outputs/*/reports/*/ 下是否**真的存在**某个数据集的 EvalScope 报告文件。

    [!] 为什么不能用"标签里含某关键词"来判断：手工汇总表的标签里只要出现"闭卷"，
        就会被误判成"跑过 general_qa"，于是在报告里**虚报一个没跑过的方法**。
        方法清单必须由**真实产物**决定。
    """
    out_dir = ROOT / CFG["paths"]["outputs"]
    if not out_dir.exists():
        return False
    return any(fragment in p.name.lower() for p in out_dir.glob("*/reports/*/*.json"))


def detect_pipeline(scores, harvested):
    """根据 outputs/ 里的真实产物判断"实际用了哪些评测方法"，避免报告与实际流程不符。

    返回 (方法清单, 是否真的跑过 RAGEval/RAGAS, 是否存在"可疑的 RAGAS 指标名")。
    """
    mets = {k for _, _, k, _ in (scores or [])}
    comps = []
    if has_evalscope_report("general_mcq"):
        comps.append("EvalScope `general_mcq`（选择题准确率，客观判分）")
    if has_evalscope_report("general_qa"):
        comps.append("EvalScope `general_qa`（闭卷问答；其主指标为 BLEU/Rouge，本项目正式结论不采用）")
    ragas = has_ragas_artifacts()
    if ragas:
        comps.append("EvalScope **RAGEval / RAGAS**（已检测到 outputs/ 中的对应产物）")
    suspicious = any(k in mets for k in ("Faithfulness", "AnswerCorrectness",
                                         "ContextRecall", "AnswerRelevancy"))
    if load_faithfulness():
        comps.append("LLM 裁判逐条忠实性判定（轻量方案，不依赖 RAGAS）")
    comps.append("本项目脚本直接统计的客观指标（检索命中率 / 拒答率 / 成本 / 延迟）")
    return comps, ragas, suspicious


# ---------------- 输出榜单 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true",
                    help="开发模式：只汇总 dev/ 下的结果，产物写入 dev/outputs")
    ap.add_argument("--no-judge", action="store_true", help="不调用裁判模型预填抽检表")
    args = ap.parse_args()
    if args.dev:
        import common
        common.enable_dev()

    rows = load_all()
    if not rows:
        raise SystemExit("runs/ 里没有数据 —— 先跑 python src\\4_rag_answer.py")

    summ = summarize(rows)
    scores = manual_scores_file()
    harvested = harvest_scores() if scores is None else []
    guard_prompt()
    spot = make_spot_check(use_judge=not args.no_judge)
    manual = read_spot_check()

    md = []
    md.append("# NSCLC 文献问答 · 大模型评测榜单\n")
    qa_all = read_jsonl(path("qa_set"))
    n_trap = sum(1 for x in qa_all if not x.get("answerable", True))
    n_verifiable = len(qa_all) - n_trap
    n_trap_verified = sum(1 for x in qa_all if x.get("trap_verified"))
    n_med = sum(1 for x in qa_all if x.get("medically_verified"))
    md.append(f"- 评测集：{len(qa_all)} 题 = 可答题 {n_verifiable} 道 + 陷阱题 {n_trap} 道"
              f"（其中 {n_trap_verified} 道带有效验证标记；判定依据是宽检索 + 裁判模型，非穷尽证明）")
    # [!] 披露口径（2026-10-11 修正）：这 n_med 道是**项目负责人本人**逐条确认的，
    #     不是独立临床医生审核。原句写成"经临床医生逐条确认"属过度声称，
    #     与本项目所有其它文档（README / docs/推送前检查清单.md / 正式评测结果）的口径矛盾。
    #     只改**措辞**，不动 n_med 的取数口径（仍是 medically_verified 计数）。
    md.append(f"- 其中可答题有 **{n_med} 道由项目负责人逐条确认医学正确性**"
              f"（**非独立医生审核**；未确认的题不会进入评测集）")
    if n_trap and n_trap_verified < n_trap:
        md.append(f"- [!] 有 {n_trap - n_trap_verified} 道陷阱题缺少有效验证标记，请检查入库流程")
    md.append(f"- 被测模型：{len(summ)} 个；每题检索 top-{CFG['top_k']} 个文献片段")
    comps, ragas_used, ragas_suspicious = detect_pipeline(scores, harvested)
    md.append("- 实际使用的评测方法（按 outputs/ 里的真实产物生成，未检测到的不写）：")
    for c in comps:
        md.append(f"    - {c}")
    if not ragas_used:
        md.append("    - [!] 本次**未使用** RAGEval/RAGAS —— 忠实性由 LLM 裁判逐条判定替代")
        if ragas_suspicious:
            md.append("    - [!] 但手工汇总表里出现了 RAGAS 类指标名，而 outputs/ 中未找到对应运行产物；"
                      "引用这些分数前请先核实来源")
        cands = ragas_candidates()
        if cands:
            md.append("    - [!] outputs/ 里有路径像 RAGEval 产物的文件，但内容中不含指标字样，"
                      "因此**不判定为已运行**（请核实这些文件是否真的来自 RAGEval 运行）：")
            for c in cands[:5]:
                md.append(f"        - `{c.relative_to(ROOT / CFG['paths']['outputs'])}`")
    import datetime
    md.append(f"- 报告生成日期：{datetime.date.today()}（模型会更新，报告必须带日期）\n")

    md.append("## 1. 可靠指标（由本项目脚本直接统计，无解析风险）\n")
    cols = ["题数(开卷)", "检索命中率@topk(可答题)", "陷阱题数", "正确拒答率", "过度自信率",
            "平均耗时(秒)", "平均输出token", "估算成本(元)"]
    # [!] 成本列的口径必须写在列名上：这一列是 **开卷 90 条＋闭卷 90 条＝180 条记录的合计**
    #     （summarize() 里对 rs 求和），而耗时/token 只算开卷 90 条。原列名只写"估算成本(元)"，
    #     读者会误当成"单次开卷 90 条的成本"。改的是**表头文案**，字典键不动。
    head = {"估算成本(元)": "估算成本(元·开卷+闭卷)"}
    md.append("> 检索命中率的分母是**有登记证据的可答题**（80 道）；陷阱题没有来源片段，不参与该统计。\n")
    md.append("> 「正确拒答率/过度自信率」由**关键词**判定（快速但粗）。裁判逐条复核见 `outputs/拒答率复核.md`："
              "50 条陷阱回答中 49 条为「明确拒答」（98%），关键词与裁判一致率 98%。\n")
    md.append("| 模型 | " + " | ".join(head.get(c, c) for c in cols) + " |")
    md.append("|" + "---|" * (len(cols) + 1))
    for model, d in summ.items():
        md.append("| " + model + " | " + " | ".join(str(d.get(c, "")) for c in cols) + " |")
    md.append("\n> **口径**：估算成本＝**开卷 90 条＋闭卷 90 条＝180 条记录合计**（不是单次开卷的成本；"
              "按各家价目 × 实际 token 估算）；平均耗时/平均输出 token 只统计**开卷 90 条**。\n")
    md.append("> 正确拒答率＝陷阱题里模型老实回答「文献里没有」的比例（越高越好）；"
              "过度自信率＝陷阱题里硬编答案的比例（越低越好）。\n")

    md.append("## 2. EvalScope 分数\n")
    if scores:
        md.append("| 模型 | 数据集 | 指标 | 分数 |")
        md.append("|---|---|---|---|")
        for m_, d_, k_, v_ in scores:
            md.append(f"| {m_} | {d_} | {k_} | {v_:.4f} |")
        md.append("\n（来自 `outputs/手工汇总.csv`，你从 EvalScope 终端表格抄进来的）\n")
    else:
        md.append(f"还没填 `outputs/手工汇总.csv`。脚本自动从 outputs/ 抓到 {len(harvested)} 条疑似分数，"
                  "**请核对后再用**（前 40 条）：\n")
        md.append("| 来源 | 指标 | 分数 |")
        md.append("|---|---|---|")
        for h in harvested[:40]:
            md.append(f"| {h['来源']} | {h['指标']} | {h['分数']} |")
        md.append("\n> 建议做法：把 EvalScope 打印的终端表格抄到 `outputs/手工汇总.csv`，"
                  "> 数据来自 `outputs/手工汇总.csv`（EvalScope 终端表格或经独立复算核对的预测文件）。\n")

    md.append("## 3. 人工抽检与一致性\n")
    md.append("> 更完整的统计口径（**分层一致率 / 判定交叉表 / 加权估计 / 与抽样设计匹配的分层 bootstrap 区间**）"
              "由 `python src\\7b_spot_stats.py` 生成，见 `outputs/抽检统计.md`。\n")
    if manual:
        # [!] 「解析失败」等非标签判定**排除**在一致率/Kappa 的分子分母之外（全项目统一口径），
        #     并把排除条数写进产物：不写明的话，读者会以为 30 条全都进了统计。
        both = [(a, h, e) for a, h, e in manual if a in CORR_LABELS]
        n_parse_fail = sum(1 for a, _, _ in manual if a and a not in CORR_LABELS)
        hum = [h for _, h, _ in manual]
        md.append(f"- 抽检条数：{len(manual)}")
        if n_parse_fail:
            md.append(f"- 排除「解析失败」{n_parse_fail} 条（裁判输出无法解析为 一致/部分一致/不一致，"
                      f"既不进分子也不进分母）")
        md.append(f"- 人工判定分布：{dict(Counter(hum))}")
        md.append(f"- 人工判定为「一致」的比例："
                  f"**{sum(1 for h in hum if h == '一致') / len(hum) * 100:.1f}%**")
        errs = Counter(e for _, _, e in manual if e)
        if errs:
            md.append(f"- 人工确认的错误类型分布：{dict(errs)}")
        if both:
            auto = [a for a, _, _ in both]
            hum2 = [h for _, h, _ in both]
            agree = sum(1 for a, h in zip(auto, hum2) if a == h) / len(both) * 100
            k = kappa(auto, hum2)
            md.append(f"- 自动判定 vs 人工判定（{len(both)} 条可比）：完全一致率 **{agree:.1f}%**，"
                      f"Cohen's Kappa **{k:.3f}**（>0.6 算一致性较好，>0.8 很好）")
        else:
            md.append("- [!]「自动判定」列是空的、或全部为「解析失败」，所以算不出一致率/Kappa。"
                      + (f"（已填人工判定的 {len(manual)} 条里，自动判定全部不可用）" if manual else "")
                      + "把 EvalScope 的逐条判分结果填进这一列，或重跑本脚本时不要加 `--no-judge`。")
        md.append("\n> 这一节是整份报告最值钱的部分：它说明「自动判分能不能信」。\n")
    else:
        md.append("还没填人工判定。打开 `qa/人工抽检表.csv`，逐条填「人工判定」和「错误类型」，"
                  "再重跑 `python src\\7_report.py`，这一节会自动出现一致率与 Kappa。\n")

    faith = load_faithfulness()
    if faith:
        md += faithfulness_md(faith)
    else:
        md.append("## 4. 忠实性\n")
        md.append("未发现 `outputs/faithfulness.jsonl`。跑完 "
                  "`.\run-eval.ps1 src\\5b_judge_faithfulness.py` 后重跑本脚本，这一节会自动出现。\n")

    out = ROOT / CFG["paths"]["outputs"] / "榜单.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[OK] 榜单 → {out}")
    print(f"[OK] 抽检表 → {spot}")
    print("\n下一步：填完抽检表重跑一次；然后写项目报告、推到 GitHub。")


if __name__ == "__main__":
    main()
