# -*- coding: utf-8 -*-
r"""
回归测试（桩模块 + 假 API；**全部在 _selftest/sandbox/ 内运行**）。

覆盖：
  1. 2_gen_qa：陷阱题"宽检索 + 裁判判定"；审核表写入带问题哈希的「陷阱验证」标记
  2. 3_finalize_qa 入库门槛：
     · 普通题未确认医学正确性 → 拦下
     · 手工把普通题改成"否"（无验证标记）→ 拦下
     · 已验证的陷阱题被改了问题文本（哈希失效）→ 拦下
  3. 4_rag_answer：
     · 指纹覆盖完整提示词（只改片段后半段也要重答）
     · --limit 不动范围外记录；--mode open 单模式不再 KeyError、且不删另一个模式
     · 失效记录不进评测产物
  4. 5b：判定指纹 + --limit 不删范围外判定
  5. 7_report：RAGEval/RAGAS 只在有**真实产物**（非空 + 含指标字样）时才声称
  6. 7c：导出的「问题」列含完整题面（A~D 选项），与裁判提示词同一拼法
  7. 裁判价目：config.yaml 的 judge 块有 price_in/price_out，common.est_cost() 对裁判不为 0
  8. （复验 R1）`.gitignore` 规则真实生效：**跑真实 `git check-ignore`**（临时仓库）断言
     报告/图表放行、`.env`/`dev/`/非公开 outputs JSONL 排除；**无 git 的环境优雅跳过并打印原因**
  9. （复验 R2）7c 重导出**不清空人工列**：回答未变原样保留、回答变更则备注标注待重新审核、覆盖前自动备份
  10. （复验 R3）7c 判定输入过**版本闸门**：过期 kb_fp/回答 fp 的记录被拦下且**0 次裁判调用**
  11. （复验 R5）`runs/*__open.json` 与 `*__open_full.jsonl` **逐题一致**（回答/参考答案/上下文）
  12. （复验 R4）成因候选**先查完整知识库**：库内存在但未进上下文 ≠ 知识库或题目缺

结尾自检：真实 kb/qa/runs/outputs 的**文件内容**（大小 + MD5）前后完全一致。
运行：.\.venv\Scripts\python.exe _selftest\regression_test.py
[i] 本测试只验证流程逻辑，不代表真实评测结果。
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_RELEASE = "--public" in sys.argv
sys.path.insert(0, str(ROOT / "src"))
os.environ["DEEPSEEK_API_KEY"] = "test-key"

SB = "_selftest/sandbox"

CFG = {
    "models": [{"display": "DeepSeek", "name": "deepseek-flash",
                "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY",
                "price_in": 1.0, "price_out": 2.0}],
    "judge": {"display": "裁判", "name": "deepseek-v4-pro",
              "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
    "embedding_hub": "BAAI/bge-small-zh-v1.5", "embedding_local": "models/bge-small-zh",
    "ragas_metrics": ["Faithfulness"], "ragas_batch_size": 6,
    "top_k": 2, "chunk_size": 700, "min_chunk_chars": 80,
    "paths": {
        "pdfs": f"{SB}/pdfs", "chunks": f"{SB}/kb/chunks.jsonl", "emb": f"{SB}/kb/emb.npy",
        "doc_index": f"{SB}/kb/doc_index.json", "candidates": f"{SB}/qa/candidates.jsonl",
        "review_csv": f"{SB}/qa/审核表.csv", "qa_set": f"{SB}/qa/qa_set.jsonl",
        "mcq_dir": f"{SB}/qa/mcq", "runs": f"{SB}/runs", "outputs": f"{SB}/outputs",
    },
    "mcq_subset": "nsclc", "question_prefix": "NSCLC",
}

GOOD = "PACIFIC 研究中位 PFS 为 16.8 个月"
CTX = {"text": GOOD + "，HR 0.52（95%CI 0.42-0.65）。" * 6}
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK] " if cond else "  [X] ") + name + (f"\n        {extra}" if extra and not cond else ""))


def install_stubs():
    np = types.ModuleType("numpy")
    np.random = types.SimpleNamespace(RandomState=lambda *a: types.SimpleNamespace(randn=lambda *x: []))
    np.save = lambda *a, **k: None
    np.asarray = lambda x, **k: x
    np.load = lambda *a, **k: None
    np.argsort = lambda x: x
    sys.modules["numpy"] = np
    y = types.ModuleType("yaml")
    y.safe_load = lambda s: CFG
    sys.modules["yaml"] = y
    d = types.ModuleType("dotenv")
    d.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = d
    o = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, **k):
            pass

    o.OpenAI = OpenAI
    sys.modules["openai"] = o


def fake_chat(m, messages, **kw):
    p = messages[-1]["content"]
    if "文献片段清单" in p:
        return json.dumps([
            {"question": "（假）这些文献报告的乳腺癌 5 年生存率是多少？", "topic": "乳腺癌", "why_absent": "癌种不同"},
            {"question": "（假）这些文献里 IL-6 的推荐阈值是多少？", "topic": "标志物", "why_absent": "未涉及"},
        ], ensure_ascii=False), {"in_tokens": 10, "out_tokens": 10}
    if "能否回答这个问题" in p:
        return '{"answerable": "不能", "reason": "片段中没有相关数据"}', {"in_tokens": 10, "out_tokens": 5}
    if "命题专家" in p:
        return json.dumps([
            {"question": "（假）PACIFIC 研究的中位 PFS 是多少？", "reference_answer": "16.8 个月",
             "evidence_quote": GOOD, "answerable": True, "type": "number",
             "options": {"A": "16.8 个月", "B": "5.6 个月", "C": "9.2 个月", "D": "24.1 个月"},
             "correct_option": "A"},
        ], ensure_ascii=False), {"in_tokens": 10, "out_tokens": 10}
    if "忠实" in p:
        return '{"judge": "忠实", "unsupported": ""}', {"in_tokens": 8, "out_tokens": 4}
    if "乳腺癌" in p:
        return "所提供的文献中没有相关内容。", {"in_tokens": 5, "out_tokens": 5}
    return "16.8 个月（片段1）。", {"in_tokens": 5, "out_tokens": 5}


def fake_retrieve(q, top_k=None):
    return [{"chunk_id": "D01-C001", "text": CTX["text"], "score": 0.9}]


def load(name):
    # [!] 优先 src/，其次 _selftest/：本测试也要能加载 _selftest 下的脚本
    #     （如 reanswer_truncated.py）来做"入口级"断言。
    p = ROOT / "src" / f"{name}.py"
    if not p.exists():
        p = ROOT / "_selftest" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(mod, argv):
    """在**捕获 stdout+stderr** 的情况下跑 `mod.main()`，返回合并后的输出。

    [!] 为什么必须连 stderr 一起捕获：PowerShell 下未捕获的 UnicodeEncodeError 会写到
        stderr，只捕获 stdout 会把它变成"看起来一切正常"的假通过（已踩过）。
        另外 `SystemExit` 的消息**不在** stdout 里（本函数不捕获异常，由调用方接），
        所以调用方判断"是否被拦住"必须看**异常/退出码**，不能只看输出文本。
    """
    old = sys.argv
    sys.argv = argv
    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            mod.main()
    finally:
        sys.argv = old
    return out.getvalue() + err.getvalue()


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_json_array(p):
    """读 JSON 数组文件（如 runs/xxx__open.json，它不是 jsonl）。"""
    p = Path(p)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def csv_sha16(p):
    """整表 sha256 前 16 位（含人工列）——用于断言"某次运行没有改动这张表"。"""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def manual_cols_sha16(p):
    """只取**人工三列**的 sha256 前 16 位（用于断言人工填写不被清空/改写）。

    为什么单列一个人工列哈希：整表哈希会因为"自动判定"列刷新而变，
    但人工列的保全与否是**独立**要断言的事，两者不能混为一谈。
    """
    cols = ("人工判定(填:一致/部分一致/不一致)",
            "错误类型(填:数值错误/结论错误/证据不足/超出文献/其他)", "备注")
    rows = read_csv(p)
    raw = json.dumps([[r.get(c, "") for c in cols] for r in rows], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def classify_check_ignore(quiet_exit, verbose_exit, verbose_out=""):
    """把 `git check-ignore` 的**两次调用结果**解释成实际忽略状态。

    [!] 为什么不能用 `-v` 的返回码：verbose 输出命中**否定规则**（`!outputs/*.md`）时
        返回码同样是 0，于是"已放行的报告/图表"会被当成"仍被忽略"（复验第 3 项：2 处假失败）。
        判定必须用 `git check-ignore -q` 的返回码：**0 = 被忽略；1 = 未被忽略（放行）**。
    [!] 返回码 **> 1 单独视为执行错误**，绝不能当成"被忽略/未忽略"——否则 git 自己出错时，
        测试会给出一个看似合理的结论（这一类误判最难发现）。
    返回 (状态, 说明)：状态 ∈ {"ignored", "not-ignored", "exec-error"}。
    """
    if quiet_exit == 0:
        return "ignored", f"-q exit=0（被忽略）；-v exit={verbose_exit} {verbose_out.strip()}"
    if quiet_exit == 1:
        return "not-ignored", f"-q exit=1（未被忽略/已放行）；-v exit={verbose_exit} {verbose_out.strip()}"
    return "exec-error", (f"-q exit={quiet_exit} > 1 → git 执行错误（不是『被忽略/未忽略』的答案）；"
                          f"-v exit={verbose_exit} {verbose_out.strip()}")


def real_snapshot():
    snap = {}
    for d in ("kb", "qa", "runs", "outputs"):
        p = ROOT / d
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    h = hashlib.md5(f.read_bytes()).hexdigest()
                except Exception:
                    h = "unreadable"
                snap[str(f.relative_to(ROOT))] = (f.stat().st_size, h)
    return snap


# =====================================================================
# 专项断言：缓存键 / 指纹**缺维度**（这一类问题已经复发两次）
# =====================================================================
def _sandbox_path(key):
    """把逻辑键映射到 sandbox 下（供 7c/7d 模块使用）。"""
    p = ROOT / CFG["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _sandbox_dir(key):
    p = ROOT / CFG["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _js(verdict, reason=""):
    """假裁判输出（合法 JSON）。"""
    return json.dumps({"judge": verdict, "reason": reason}, ensure_ascii=False)


def _mk_item():
    return {"question_id": "NSCLC-001", "question": "PACIFIC 研究的中位 PFS 是多少？",
            "reference_answer": "16.8 个月", "type": "number", "answerable": True,
            "source_chunk": "D01-C001", "options": {"A": "16.8 个月", "B": "5.6 个月",
                                                    "C": "9.2 个月", "D": "24.1 个月"},
            "correct_option": "A", "evidence_quote": GOOD}


def _mk_rec(qid="NSCLC-001", model="DeepSeek", name="deepseek-flash",
            resp="16.8 个月（片段1）。", ctx=None):
    return {"model": model, "model_name": name, "mode": "open", "question_id": qid,
            "type": "number", "answerable": True, "question": "PACIFIC 研究的中位 PFS 是多少？",
            "reference": "16.8 个月", "response": resp, "error": False,
            "retrieved_ids": ["D01-C001"], "retrieved_contexts": [ctx if ctx is not None else CTX["text"]],
            "gold_hit": True, "max_tokens_used": 2000, "prompt_ver": "v3", "fp": "x", "kb_fp": "x"}


def check_cache_key_dimensions():
    """断言：判定缓存的**输入指纹**必须覆盖每一个会改变判定的维度。

    背景（同一根因犯过两次）：
      · 第一次：缓存键只有 (裁判名, 题号) → 5 个模型的回答互相覆盖；
      · 第二次：缓存键只有 (被评模型, 题号) → 改回答/改选项/换裁判/换知识库都**静默复用旧判定**。
    这里逐个维度验证 fp 会变，并验证"fp 不一致就当未判、重新调用裁判"。
    """
    m7c = load("7c_export_full_review")
    m7c.path, m7c.dirpath = _sandbox_path, _sandbox_dir
    item, rec = _mk_item(), _mk_rec()
    j1 = dict(CFG["judge"])
    j2 = dict(CFG["judge"], name="judge-b", base_url="https://other.example/v1")

    base = m7c.fp_of(item, rec, j1)
    check("fp 覆盖**模型回答**：回答改一个字 → fp 变",
          m7c.fp_of(item, dict(rec, response="16.8 个月（改写过的回答）。"), j1) != base)
    check("fp 覆盖**题面含 A~D 选项**：选项文字变 → fp 变",
          m7c.fp_of(dict(item, options={"A": "18.6 个月", "B": "5.6 个月"}), rec, j1) != base)
    check("fp 覆盖**标准答案**：标准答案变 → fp 变",
          m7c.fp_of(dict(item, reference_answer="17.0 个月"), rec, j1) != base)
    check("fp 覆盖**裁判名**：换裁判模型 → fp 变", m7c.fp_of(item, rec, j2) != base)
    check("fp 覆盖**base_url**：同名模型换网关 → fp 变",
          m7c.fp_of(item, rec, dict(j1, base_url="https://mirror.example/v1")) != base)
    check("fp 覆盖**提示词/口径版本**：JUDGE_INPUT_VER 变 → fp 变",
          m7c.fp_of(item, rec, j1, ver="v999") != base)

    # ---- 行为断言：fp 不一致必须视为"未判"并重新调用裁判 ----
    calls = []

    def fake_judge(m, messages, **kw):
        calls.append(messages[-1]["content"])
        return _js("一致", "假理由"), {"in_tokens": 1, "out_tokens": 1}

    m7c.chat = fake_judge
    item, rec = _mk_item(), _mk_rec()
    cur = m7c.fp_of(item, rec, j1)

    cache = {("DeepSeek", "NSCLC-001"): {"model": "DeepSeek", "judge_name": j1["name"],
                                         "question_id": "NSCLC-001", "judge": "一致",
                                         "fp": cur, "prompt_ver": m7c.JUDGE_INPUT_VER}}
    v, hit = m7c.judge_verdict(j1, item, rec, cache)
    check("fp 一致 → 命中缓存，不调用裁判", hit and v == "一致" and not calls)

    stale = {("DeepSeek", "NSCLC-001"): dict(cache[("DeepSeek", "NSCLC-001")], fp="deadbeef0000")}
    v, hit = m7c.judge_verdict(j1, item, rec, stale)
    check("fp 不一致 → 视为未判并重新判", (not hit) and v == "一致" and len(calls) == 1)

    fail = {("DeepSeek", "NSCLC-001"): dict(cache[("DeepSeek", "NSCLC-001")],
                                            judge="解析失败", fp=cur)}
    v, hit = m7c.judge_verdict(j1, item, rec, fail)
    check("上次「解析失败」→ 视为未判并重试", (not hit) and v == "一致" and len(calls) == 2)

    # ---- 落盘行必须带齐可追溯字段 ----
    row = stale[("DeepSeek", "NSCLC-001")]
    for col in ("model", "model_name", "judge_name", "question_id", "judge", "reason",
                "raw_head", "fp", "prompt_ver"):
        check(f"判定缓存行含字段 {col}", col in row)
    check("落盘行同时含显示名与模型ID",
          row.get("model") == "DeepSeek" and row.get("model_name") == "deepseek-flash")

    # ---- 7d：只能读**指定裁判**的缓存，不得被历史留档覆盖 ----
    # [!] 7d 现在还要**逐条复核判分材料指纹**（正确性用 7c.fp_of、忠实性用 5b.fp_of）：
    #     桩缓存行必须带上"用**当前沙箱回答**复算出来的 fp"，否则它会被正确地判为
    #     "旧判定配新回答"而排除出配对（那是对的行为，但会让本节断言失去意义）。
    #     [!] 这里特意用 **NSCLC-002**（沙箱里当前有效的记录）：NSCLC-001 的开卷记录在本测试
    #     流程中已被 3_finalize_qa 重新生成的题集文本判为"输入指纹不一致"，会被正确排除。
    #     [!] 顺带把断言加强：若 7d 又去 glob 合并 `.OLD-RUBRIC` 留档（那份没有 fp），
    #     NSCLC-002 的键会被它覆盖 → 材料指纹复核不过 → 可比条数变 0，断言立刻失败。
    outd = _sandbox_dir("outputs")
    _qa_now = {x["question_id"]: x for x in read_jsonl(ROOT / CFG["paths"]["qa_set"])}
    _rec_now = {}
    for _f in sorted((ROOT / CFG["paths"]["runs"]).glob("*_open_full.jsonl")):
        for _r in read_jsonl(_f):
            if _r.get("mode") == "open" and _r.get("model") == "DeepSeek":
                _rec_now[_r["question_id"]] = _r
    _qid = "NSCLC-002" if "NSCLC-002" in _rec_now else sorted(_rec_now)[0]
    _fp_corr = m7c.fp_of(_qa_now[_qid], _rec_now[_qid], j1)
    (outd / f"auto_verdicts_{j1['name']}.jsonl").write_text(
        json.dumps({"model": "DeepSeek", "question_id": _qid, "judge": "一致",
                    "judge_name": j1["name"], "fp": _fp_corr}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    # 旧口径留档：按字典序 ".OLD-…" **排在 ".jsonl" 之后**，glob 合并时它会覆盖新判定
    (outd / f"auto_verdicts_{j1['name']}.OLD-RUBRIC-v1.jsonl").write_text(
        json.dumps({"model": "DeepSeek", "question_id": _qid, "judge": "不一致",
                    "judge_name": j1["name"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    blind = _sandbox_path("qa_set").parent / "双裁判盲评"
    blind.mkdir(parents=True, exist_ok=True)
    (blind / "DeepSeek.csv").write_text(
        "题目ID,模型,正确性判定(填:一致/部分一致/不一致),忠实性判定(填:忠实/部分无依据/无依据)\n"
        f"{_qid},DeepSeek,一致,\n", encoding="utf-8-sig")
    m7d = load("7d_dual_judge")
    m7d.path, m7d.dirpath = _sandbox_path, _sandbox_dir
    old_argv = sys.argv
    sys.argv = ["7d_dual_judge.py", "--compare"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            m7d.do_compare()
    finally:
        sys.argv = old_argv
    cmp_md = (outd / "双裁判比对.md").read_text(encoding="utf-8")
    check("7d 只读指定裁判的缓存（不被 .OLD-RUBRIC 留档覆盖 27 条口径）"
          "，且缓存行必须通过**判分材料指纹复核**才进配对",
          "可比条数：**1**" in cmp_md
          and "正确性 一致 **1** / 不一致 **0** / 缺 **0**" in cmp_md, cmp_md[:900])
    check("7d 产物写明了缓存来源与排除口径",
          f"auto_verdicts_{j1['name']}.jsonl" in cmp_md and "解析失败" in cmp_md)
    bad = outd / f"auto_verdicts_{j1['name']}.jsonl"
    bad.write_text(json.dumps({"model": "DeepSeek", "question_id": "NSCLC-001",
                               "judge": "一致", "judge_name": "别的裁判"}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m7d.do_compare()
        check("7d 遇到 judge_name 串用必须报错退出", False, "竟然继续跑完了")
    except SystemExit:
        check("7d 遇到 judge_name 串用直接退出（不静默算错 Kappa）", True)
    (outd / f"auto_verdicts_{j1['name']}.jsonl").write_text(
        json.dumps({"model": "DeepSeek", "question_id": "NSCLC-001", "judge": "一致",
                    "judge_name": j1["name"], "fp": _fp_corr}, ensure_ascii=False) + "\n",
        encoding="utf-8")


def check_budget_no_downgrade():
    """断言：提高过额度的**有效**记录在全量续跑时零调用、且额度不被退回。

    背景：`record_is_current()` 能按记录自身额度复核，但 `main()` 选任务时用的是**默认额度**
    算出的 want_fp → 一条有效的 6000 额度记录会被安排重答并**退回 2000**（实测 3 条）。
    """
    m4 = load("4_rag_answer")
    m4.path, m4.dirpath = _sandbox_path, _sandbox_dir
    m4.retrieve = fake_retrieve
    item = _mk_item()
    ctx = {"ids": ["D01-C001"], "texts": [CTX["text"]],
           "context": "\n\n".join(f"[片段{j}] {t}" for j, t in enumerate([CTX["text"]], 1))}
    calls = []

    def fake_chat_budget(m, messages, **kw):
        calls.append(kw.get("max_tokens"))
        b = kw.get("max_tokens", 2000)
        return "16.8 个月（片段1）。", {"in_tokens": 5, "out_tokens": 5, "finish_reason": "stop",
                                       "attempts": [{"requested_max_tokens": b,
                                                     "finish_reason": "stop", "ok": True,
                                                     "in_tokens": 5, "out_tokens": 5}]}

    m4.chat = fake_chat_budget
    (ROOT / CFG["paths"]["qa_set"]).write_text(
        json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    runs = _sandbox_dir("runs")
    for mode in ("open", "closed"):
        c = ctx if mode == "open" else m4.empty_ctx()
        fp, _ = m4.fp_of(CFG["models"][0], mode, item, c, budget=6000)
        rec = dict(_mk_rec(resp="16.8 个月（已按 6000 额度答完整）。"),
                   mode=mode, max_tokens_used=6000, attempt_budgets=[6000],
                   fp=fp, fp_budget=6000, retrieval_fp=m4.retrieval_fp(c),
                   retrieved_ids=c["ids"], retrieved_contexts=c["texts"])
        (runs / f"deepseek-flash__{mode}_full.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")

    calls.clear()
    out = run(m4, ["4_rag_answer.py", "--workers", "1"])
    check("6000 额度有效记录：全量续跑**零调用**", not calls, f"实际调用 {calls}")
    check("6000 额度有效记录：待答 0 条", "本次待答 0 条" in out, out[-500:])
    check("6000 额度有效记录：未被退回默认额度",
          "本次待答 0 条" in out and not calls, out[-300:])
    rows = read_jsonl(runs / "deepseek-flash__open_full.jsonl")
    check("记录额度仍为 6000（未被 2000 覆盖）", rows[0]["max_tokens_used"] == 6000,
          f"实际 {rows[0]['max_tokens_used']}")

    # 反向断言：内容真的变了，重答额度也只能 ≥ 原额度（绝不降级）
    item2 = dict(item, question=item["question"] + "（改过）")
    (ROOT / CFG["paths"]["qa_set"]).write_text(
        json.dumps(item2, ensure_ascii=False) + "\n", encoding="utf-8")
    calls.clear()
    run(m4, ["4_rag_answer.py", "--workers", "1"])
    check("输入真的变更时：重答额度 ≥ 记录已用额度（不降级到 2000）",
          bool(calls) and min(calls) >= 6000, f"实际请求额度 {calls}")

    # ---- 重试元数据留档：第一次空答、第二次成功 → 记最终额度 + attempts ----
    seq = {"n": 0}

    def fake_chat_retry(m, messages, **kw):
        seq["n"] += 1
        b = kw.get("max_tokens", 2000)
        if seq["n"] == 1:                       # 第一次：实质空答（只有前缀）
            return "答案：", {"in_tokens": 7, "out_tokens": 0, "finish_reason": "length",
                             "attempts": [{"requested_max_tokens": b, "finish_reason": "length",
                                           "ok": False, "in_tokens": 7, "out_tokens": 0}]}
        return "16.8 个月（片段1）。", {"in_tokens": 9, "out_tokens": 30,
                                     "finish_reason": "stop",
                                     "attempts": [{"requested_max_tokens": b,
                                                   "finish_reason": "stop", "ok": True,
                                                   "in_tokens": 9, "out_tokens": 30}]}

    m4.chat = fake_chat_retry
    (ROOT / CFG["paths"]["qa_set"]).write_text(
        json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    (runs / "deepseek-flash__open_full.jsonl").unlink()
    (runs / "deepseek-flash__closed_full.jsonl").unlink()
    run(m4, ["4_rag_answer.py", "--mode", "open", "--workers", "1"])
    r = read_jsonl(runs / "deepseek-flash__open_full.jsonl")[0]
    check("重试留档：max_tokens_used = 最终成功那次的额度（4000 而非 2000）",
          r.get("max_tokens_used") == 4000, f"实际 {r.get('max_tokens_used')}")
    check("重试留档：记录了 finish_reason", r.get("finish_reason") == "stop",
          f"实际 {r.get('finish_reason')}")
    check("重试留档：attempts 含两次尝试（含被丢弃的空答）",
          len(r.get("attempts") or []) == 2, f"实际 {r.get('attempts')}")
    check("重试留档：attempt_budgets 含 2000 与 4000",
          sorted(r.get("attempt_budgets") or []) == [2000, 4000],
          f"实际 {r.get('attempt_budgets')}")

    # 历史记录（缺 attempts/finish_reason/max_tokens_used）必须被容忍
    legacy = {"model": "DeepSeek", "model_name": "deepseek-flash", "mode": "open",
              "question_id": "NSCLC-001", "type": "number", "answerable": True,
              "question": item["question"], "reference": item["reference_answer"],
              "response": "16.8 个月。", "error": False, "fp": "whatever"}
    check("读取方容忍历史记录缺 attempts/finish_reason/max_tokens_used",
          m4.record_is_current(CFG["models"][0], "open", item, legacy, {}) is False
          and m4.rec_own_budget(legacy) == m4.MAX_TOKENS)


def check_reanswer_cache_invalidation():
    """断言：重答后只失效**当前裁判**缓存里受影响的条目，且用**显示名**比对。

    背景：`reanswer_truncated.py` 用缓存行的显示名 `"DeepSeek"` 去比 run 记录的
    `model_name` `"deepseek-flash"` → 实测交集 0/450，整个失效逻辑是**空操作**；
    而且它 `glob("auto_verdicts_*.jsonl")`，会把 `.OLD-RUBRIC-*`、`.BAD-*`、`.bak-*`
    这些历史留档一并就地截断（破坏证据）。
    """
    mra = load("reanswer_truncated")
    mra.dirpath = _sandbox_dir
    outd = _sandbox_dir("outputs")
    jname = CFG["judge"]["name"]
    keep_rows = [{"model": "DeepSeek", "question_id": "NSCLC-001", "judge": "一致",
                  "judge_name": jname},
                 {"model": "DeepSeek", "question_id": "NSCLC-002", "judge": "一致",
                  "judge_name": jname},
                 {"model": "豆包", "question_id": "NSCLC-001", "judge": "不一致",
                  "judge_name": jname},
                 # 模型 ID 出现在 model 字段是**异常**数据：显示名比对下不应被误删
                 {"model": "deepseek-flash", "question_id": "NSCLC-003", "judge": "一致",
                  "judge_name": jname}]
    cur = outd / f"auto_verdicts_{jname}.jsonl"
    cur.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in keep_rows) + "\n",
                   encoding="utf-8")
    old = outd / f"auto_verdicts_{jname}.OLD-RUBRIC-v1.jsonl"
    old_txt = json.dumps({"model": "DeepSeek", "question_id": "NSCLC-001", "judge": "不一致",
                          "judge_name": jname}, ensure_ascii=False) + "\n"
    old.write_text(old_txt, encoding="utf-8")
    bad = outd / f"auto_verdicts_{jname}.BAD-缓存键错误.jsonl"
    bad.write_text(old_txt, encoding="utf-8")

    # 只失效 (显示名 DeepSeek, NSCLC-001)
    dropped = mra.invalidate_verdict_cache({("DeepSeek", "NSCLC-001")}, jname)
    check("重答失效：按**显示名**命中（旧代码用 model_name → 交集 0，空操作）", dropped == 1,
          f"实际失效 {dropped} 条")
    left = read_jsonl(cur)
    check("重答失效：只删受影响的那一条",
          len(left) == 3 and all(not (x["model"] == "DeepSeek" and x["question_id"] == "NSCLC-001")
                                 for x in left), str(left))
    check("重答失效：同题号的其他模型判定不受影响",
          any(x["model"] == "豆包" and x["question_id"] == "NSCLC-001" for x in left))
    check("重答失效：绝不改写 .OLD-RUBRIC 历史留档",
          old.read_text(encoding="utf-8") == old_txt)
    check("重答失效：绝不改写 .BAD 历史留档",
          bad.read_text(encoding="utf-8") == old_txt)
    # 截断判据：读记录自身额度（6000 不在 {2000,4000,8000} 里，旧代码会漏报）
    check("截断判据读记录自身额度（6000 额度记录不被漏报）",
          mra.record_hit_budget_cap({"out_tokens": 6000, "max_tokens_used": 6000,
                                     "attempt_budgets": [6000], "error": False}))
    check("截断判据不误报：额度翻倍后正常答完（out=5079, cap={2000,4000}）",
          not mra.record_hit_budget_cap({"out_tokens": 5079, "max_tokens_used": 4000,
                                         "attempt_budgets": [2000, 4000], "error": False}))
    cur.unlink()
    old.unlink()
    bad.unlink()


def check_export_question_column():
    """断言：7c 导出的「问题」列是**完整题面（含 A~D 选项）**，与裁判提示词用同一个拼法。

    背景：判定提示词早就用 `common.full_question()` 带了选项，但导出的「问题」列一直只写裸题干
    —— 人工复核时看不到选项文字，没法核对模型回答里的字母（同一根因的另一半）。
    这里构造一道带选项的题，走一遍 7c 的导出入口，直接检查产物那一列。
    """
    m7c = load("7c_export_full_review")
    m7c.path, m7c.dirpath = _sandbox_path, _sandbox_dir
    m7c.chat = lambda m, msgs, **kw: (_js("一致", "假理由"), {"in_tokens": 1, "out_tokens": 1})
    item = _mk_item()
    qa = _sandbox_path("qa_set")
    qa.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    runs = _sandbox_dir("runs")
    for f in runs.glob("*_open_full.jsonl"):        # 只留这一条，保证 --limit 1 取到它
        f.unlink()
    # [!] 7c 现在要先过 7_report.load_all 的**版本闸门**（复验 R3）。该闸门用
    #     `4_rag_answer.record_is_current` 复算**作答输入指纹**（与 7c 自己的判定指纹
    #     `fp_of` 不是同一个东西，别混），所以桩记录必须写成"真的由 4_rag_answer 作答产生"的样子。
    rec = _mk_rec(qid=item["question_id"])
    m4 = load("4_rag_answer")
    ctx = m4.ctx_from_record(rec)
    rec["fp"] = m4.fp_of(CFG["models"][0], "open", item, ctx)[0]
    rec["retrieval_fp"] = m4.retrieval_fp(ctx)
    rec["kb_fp"] = m4.kb_fp()
    (runs / "exportcheck__open_full.jsonl").write_text(
        json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    run(m7c, ["7c_export_full_review.py", "--limit", "1"])
    out = qa.parent / "人工抽检表_全量450_测试.csv"
    rows = read_csv(out)
    got = rows[0]["问题"] if rows else ""
    check("7c 导出「问题」列仍含裸题干", "PACIFIC 研究的中位 PFS 是多少？" in got, got)
    check("7c 导出「问题」列含 A~D 选项（A. / D.）",
          "A." in got and "D." in got
          and all(f"{k}. {v}" in got for k, v in item["options"].items()), got)
    out.unlink()


def judge_prices_from_config():
    """从 config.yaml 文本里抽出 `judge:` 块的 price_in / price_out。

    [!] 不能用 yaml 库：本测试把 sys.modules["yaml"] 换成了桩（safe_load 一律返回测试 CFG），
        所以这里只做最小文本解析，读的才是**真实的 config.yaml**。
    """
    out, inside = {}, False
    for line in (ROOT / "config.yaml").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):                     # 顶格 = 新的一段
            inside = line.split(":")[0].strip() == "judge"
            continue
        if inside:
            k, _, v = line.strip().partition(":")
            if k in ("price_in", "price_out"):
                out[k] = float(v.split("#")[0].strip())
    return out


def check_judge_price():
    """断言：config.yaml 的 judge 块有价目，且 common.est_cost() 对裁判**不返回 0**。

    背景：`judge:` 块过去没有 price_in/price_out，`est_cost()` 里 `m.get("price_in", 0)` 默认 0
    → 裁判花费永远估成 0（裁判是花钱最多的一环）。
    """
    prices = judge_prices_from_config()
    check("config.yaml 的 judge 块填了 price_in / price_out",
          set(prices) == {"price_in", "price_out"}, str(prices))
    cmn = sys.modules.get("common") or load("common")
    judge = {"price_in": prices.get("price_in", 0), "price_out": prices.get("price_out", 0)}
    cost = cmn.est_cost(judge, 1_000_000, 1_000_000)     # 1M 输入 + 1M 输出
    check("est_cost() 对裁判返回非 0（1M in + 1M out）", cost > 0, f"cost={cost}")
    check("est_cost() 按元/百万 token 换算（1.0 + 2.0 = 3.0 元）",
          abs(cost - (prices.get("price_in", 0) + prices.get("price_out", 0))) < 1e-9,
          f"cost={cost} 价目={prices}")


# =====================================================================
# 12) 复验残留项的回归断言（R1 .gitignore / R2 人工列 / R3 版本闸门）
# =====================================================================
def check_gitignore_rules():
    """断言：`.gitignore` 的放行/排除规则**真的生效**（跑真实 `git check-ignore`）。

    背景（复验 R1）：`outputs/* #说明`、`!outputs/ #说明`、`dev/ #说明` 把注释写在同一行，
    而 gitignore 的注释**只在行首 `#`** 才成立 → 整行（含空格与说明文字）成了模式，
    匹配不到任何文件，于是仍由第 8 行 `outputs/` 排除整个目录，正式报告与图表**推不上去**。

    [!] 这里**必须跑真实 git**，不能只断言"文件里有 `!outputs/` 这个字符串"——
    那正是当初误判为"已修好"的原因。无 git 的环境**优雅跳过并打印原因**（不静默通过）。

    [!] 判定必须用 `git check-ignore -q` 的**返回码**（0=忽略 / 1=放行）；`-v` 只用于打印诊断。
    用 `-v` 的返回码判"是否被忽略"是**错的**：命中否定规则（`!outputs/*.md`）时 `-v` 同样返回 0
    → 已放行的报告/图表被当成"仍被忽略"（复验第 3 项：2 处假失败，已实测）。
    返回码 **> 1 一并视为执行错误**，绝不折算成"忽略/放行"。
    """
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    # ① 静态部分：不允许出现"规则 + 行尾 # 注释"的行（注释必须独占一行）
    bad = [ln for ln in gi.splitlines() if ln.strip() and not ln.startswith("#") and "#" in ln]
    check("`.gitignore` 里没有『规则与注释同行』的行（注释必须独占一行）", not bad, str(bad))
    for pat in ("!outputs/", "!outputs/*.md", "!outputs/*.csv", "!outputs/图表/",
                "!outputs/图表/*.png", "dev/"):
        check(f"`.gitignore` 保留纯模式 {pat}（行尾无注释）",
              any(ln.strip() == pat for ln in gi.splitlines()), pat)

    # ② 判据本身的单元断言（**无 git 也必须跑**）：否则"返回码用错了"这类缺陷
    #    在没有 git 的机器上永远不会被发现——正是本项的原始根因。
    st, _ = classify_check_ignore(0, 0, ".gitignore:45:!outputs/*.md")
    check("check-ignore 判据：-q 返回 0 → 被忽略（否定规则命中时 -v 也是 0，不能用 -v 判）",
          st == "ignored", st)
    st, _ = classify_check_ignore(1, 0, ".gitignore:45:!outputs/*.md")
    check("check-ignore 判据：-q 返回 1（-v 返回 0，命中否定规则）→ 放行，不得判成被忽略",
          st == "not-ignored", st)
    st, _ = classify_check_ignore(128, 128, "fatal: not a git repository")
    check("check-ignore 判据：返回码 > 1 → 执行错误（不得当成忽略/放行）",
          st == "exec-error", st)

    # ③ 动态部分：真实 git check-ignore（临时仓库，复制项目 .gitignore）
    git = shutil.which("git")
    if not git:
        print("  [skip] 未找到 git 可执行文件 → 跳过 5 条『真实 git check-ignore』断言。"
              "原因：本机 PATH 与常见安装位置都没有 git.exe；"
              "该项需在有 git 的环境（或 CI）复跑才算通过（跳过不静默计为通过）。")
        return
    # [!] 临时仓库目录**不用 tempfile.mkdtemp**：在受限/沙箱环境里 mkdtemp 目录可能不可写，
    #     会让 5 条真实断言**整体报错**（而不是给出真实的忽略状态），把环境问题伪装成代码缺陷。
    #     这里改用工作区内固定路径（每次运行前清空重建），并放在 .tmp/ 下——它不在
    #     real_snapshot 的 kb/qa/runs/outputs 四个目录里，所以不会影响"真实数据未被改动"的自检。
    tmp = ROOT / ".tmp" / "regression_gitignore"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "outputs" / "图表").mkdir(parents=True, exist_ok=True)
    (tmp / "dev").mkdir(parents=True, exist_ok=True)
    try:
        (tmp / ".gitignore").write_text(gi, encoding="utf-8")
        expect = {"outputs/正式评测结果_2026-10-10.md": False,
                  "outputs/图表/图1_模型对照.png": False,
                  ".env": True, "dev/example.json": True,
                  "outputs/auto_verdicts_deepseek-v4-pro.jsonl": True}
        for rel in expect:
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x", encoding="utf-8")
        subprocess.run([git, "init", "-q"], cwd=str(tmp), capture_output=True)
        for rel, should in expect.items():
            # -q：**只看返回码**（0=忽略 / 1=放行）；>1 = 执行错误
            pq = subprocess.run([git, "check-ignore", "-q", "--", rel], cwd=str(tmp),
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace")
            # -v：**只用于打印诊断**（命中否定规则时它也返回 0，不能用它判定）
            pv = subprocess.run([git, "check-ignore", "-v", "--", rel], cwd=str(tmp),
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace")
            state, detail = classify_check_ignore(pq.returncode, pv.returncode, pv.stdout or "")
            got = {"ignored": True, "not-ignored": False}.get(state)
            check(f"git check-ignore：{'忽略' if should else '放行'} {rel}",
                  got is not None and got == should, f"state={state} {detail}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_7c_manual_columns_and_gate():
    """断言（复验 R2/R3）：

      · 已填人工列的表**再次导出后人工列仍在**（原实现每次写空 → 静默清空）；
      · 回答变更时**不静默清空**，而是备注标注"回答已变更，需重新审核"；
      · 覆盖前**自动备份**；
      · 判定输入过**版本闸门**：过期 kb_fp / 过期回答 fp 的记录被拦下，且**0 次裁判调用**。
    """
    m7c = load("7c_export_full_review")
    m7c.path, m7c.dirpath = _sandbox_path, _sandbox_dir
    sb_qa = ROOT / CFG["paths"]["qa_set"]
    qa = _sandbox_path("qa_set")
    runs = _sandbox_dir("runs")
    calls = []

    def fake_judge(m, msgs, **kw):
        calls.append(m.get("name"))
        return _js("一致", "假理由"), {"in_tokens": 1, "out_tokens": 1}

    m7c.chat = fake_judge
    item = _mk_item()
    qa.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    for f in runs.glob("*_open_full.jsonl"):
        f.unlink()

    # ---- 版本闸门：过期 kb_fp / 过期 fp 的记录**不得**进裁判 ----
    expired = dict(_mk_rec(), kb_fp="old-kb", fp="outdated-answer-fp")
    (runs / "gate__open_full.jsonl").write_text(
        json.dumps(expired, ensure_ascii=False) + "\n", encoding="utf-8")
    calls.clear()
    try:
        out = run(m7c, ["7c_export_full_review.py"])
        blocked_ok = False
    except SystemExit as e:
        out, blocked_ok = str(e), True
    check("7c：过期回答（kb_fp/fp 不符）被版本闸门拦下",
          blocked_ok and ("版本校验" in out or "版本闸门" in out), out[:400])
    check("7c：过期回答**一次裁判都没有调用**", not calls, f"实际调用 {calls}")

    # ---- 人工列保护：先填一条，再导出两次 ----
    # 让这条记录**真的通过**版本闸门：按 4_rag_answer 的口径复算作答输入指纹
    # （和 7c 自己的判定指纹 fp_of 不是一个东西）。
    def _gate_ok_rec(resp="16.8 个月（片段1）。"):
        r = _mk_rec(qid="NSCLC-001", model="DeepSeek", name="deepseek-flash", resp=resp)
        r["fp"] = m4.fp_of(CFG["models"][0], "open", item, m4.ctx_from_record(r))[0]
        r["retrieval_fp"] = m4.retrieval_fp(m4.ctx_from_record(r))
        r["kb_fp"] = m4.kb_fp()
        return r

    (runs / "gate__open_full.jsonl").write_text(
        json.dumps(_gate_ok_rec(), ensure_ascii=False) + "\n", encoding="utf-8")
    calls.clear()
    run(m7c, ["7c_export_full_review.py"])                 # 第一次：正常生成
    csvp = qa.parent / "人工抽检表_全量450.csv"
    rows = read_csv(csvp)
    check("7c：通过版本闸门的有效记录能正常导出（闸门不是无差别全拦）",
          bool(rows), f"导出 {len(rows)} 行")
    rows[0]["人工判定(填:一致/部分一致/不一致)"] = "部分一致"
    rows[0]["错误类型(填:数值错误/结论错误/证据不足/超出文献/其他)"] = "数值错误"
    rows[0]["备注"] = "回归测试填写标记 REG-MARK"
    write_csv(csvp, rows)
    calls.clear()
    run(m7c, ["7c_export_full_review.py"])                 # 第二次：回答未变
    after = read_csv(csvp)
    check("7c：已填表再次导出后「人工判定」仍在（不再清空）",
          after[0]["人工判定(填:一致/部分一致/不一致)"] == "部分一致",
          str(after[0].get("人工判定(填:一致/部分一致/不一致)")))
    check("7c：已填表再次导出后「错误类型」「备注」仍在",
          after[0]["错误类型(填:数值错误/结论错误/证据不足/超出文献/其他)"] == "数值错误"
          and "REG-MARK" in (after[0]["备注"] or ""), str(after[0].get("备注")))
    check("7c：第二次导出零裁判调用（判定缓存命中）", not calls, f"实际调用 {calls}")
    check("7c：覆盖前已生成带时间戳的自动备份",
          bool(list(qa.parent.glob("人工抽检表_全量450.csv.bak-*"))))
    # 回答真的变了 → 保留填写内容，但备注必须标注待重新审核
    (runs / "gate__open_full.jsonl").write_text(
        json.dumps(_gate_ok_rec("16.8 个月（回归测试改写过的回答）。"), ensure_ascii=False) + "\n",
        encoding="utf-8")
    run(m7c, ["7c_export_full_review.py"])
    marked = read_csv(csvp)
    note = marked[0]["备注"] or ""
    check("7c：回答变更时人工列不被清空、而是标注『回答已变更，需重新审核』",
          marked[0]["人工判定(填:一致/部分一致/不一致)"] == "部分一致"
          and "回答已变更" in note and "重新审核" in note
          and "REG-MARK" in note, note)

    # ---- 三类入口断言：① 缓存缺失 ② 回答部分过期 ③ 真正零调用 ----
    # [!] 为什么必须**真的把缓存删掉**再跑：上一版这个用例是**假通过**——它用
    #     `"--offline" in out_off` 判定，而启动横幅本身就含 `--offline`；更糟的是它只删了
    #     **真实** outputs/ 下的缓存，沙箱缓存仍然命中 → 这条断言在"闸门完全失效"时也会显示通过
    #     （已实测：闸门失效时假 chat 被调用 1 次、退出码 0，用例照样 OK）。
    #     现在断言三件事：**退出码非 0** + **假 chat 调用计数 = 0** + **CSV 哈希前后一致**。
    sb_jsonl = m7c.dirpath("outputs") / f"auto_verdicts_{CFG['judge']['name']}.jsonl"   # 沙箱缓存
    sb_csv = qa.parent / "人工抽检表_全量450.csv"
    assert ROOT / "_selftest" in sb_csv.parents or "_selftest" in str(sb_csv), \
        f"离线用例必须只操作沙箱内的表，实际 {sb_csv}"
    rows_in = read_csv(sb_csv)
    rows_in[0]["人工判定(填:一致/部分一致/不一致)"] = "部分一致"
    rows_in[0]["错误类型(填:数值错误/结论错误/证据不足/超出文献/其他)"] = "数值错误"
    rows_in[0]["备注"] = "REGRESSION OFFLINE CANARY"
    write_csv(sb_csv, rows_in)
    (runs / "gate__open_full.jsonl").write_text(
        json.dumps(_gate_ok_rec(), ensure_ascii=False) + "\n", encoding="utf-8")

    # ① 缓存缺失 + --offline → 必须非零退出、零调用、CSV 不变
    if sb_jsonl.exists():
        sb_jsonl.unlink()
    sha_before, man_before = csv_sha16(sb_csv), manual_cols_sha16(sb_csv)
    calls.clear()
    try:
        out_off = run(m7c, ["7c_export_full_review.py", "--offline"])
        exit_off, exc_off = 0, None
    except SystemExit as e:
        out_off, exit_off, exc_off = str(e), (e.code if isinstance(e.code, int) else 1), e
    check("7c --offline：**缓存缺失**时非零退出（不再只看输出里有没有 `--offline`）",
          exit_off != 0, f"exit={exit_off} exc={exc_off!r}")
    check("7c --offline：缓存缺失时**假 chat 调用计数 = 0**（闸门真的在调用裁判之前拦住）",
          not calls, f"实际调用 {len(calls)} 次 {calls}")
    check("7c --offline：缓存缺失时「人工抽检表_全量450.csv」整表哈希前后一致（未被覆盖）",
          csv_sha16(sb_csv) == sha_before,
          f"{sha_before} → {csv_sha16(sb_csv)}")
    check("7c --offline：缓存缺失时人工三列哈希前后一致（人工填写未被清空/改写）",
          manual_cols_sha16(sb_csv) == man_before,
          f"{man_before} → {manual_cols_sha16(sb_csv)}")
    after_off = read_csv(sb_csv)
    check("7c --offline 失败退出后，CSV 人工列仍原样保留",
          after_off[0]["人工判定(填:一致/部分一致/不一致)"] == "部分一致"
          and "REGRESSION OFFLINE CANARY" in (after_off[0]["备注"] or ""),
          str(after_off[0].get("备注")))

    # ③ 真正零调用：先用**正常模式**把缓存建起来（1 次调用），再跑 --offline → 必须成功且零调用
    calls.clear()
    run(m7c, ["7c_export_full_review.py"])
    n_warm = len(calls)
    sha_cached = csv_sha16(sb_csv)
    calls.clear()
    try:
        out_hit = run(m7c, ["7c_export_full_review.py", "--offline"])
        exit_hit = 0
    except SystemExit as e:
        out_hit, exit_hit = str(e), (e.code if isinstance(e.code, int) else 1)
    check("7c --offline：**有效缓存命中**时正常导出（退出码 0，不误拦）",
          exit_hit == 0 and "已导出 1 行" in out_hit, f"exit={exit_hit} out={out_hit[:200]}")
    check("7c --offline：缓存命中时是**真正零调用**（假 chat 计数 = 0）",
          not calls, f"预热用了 {n_warm} 次，离线命中又调用 {len(calls)} 次 {calls}")
    check("7c --offline：缓存命中导出后 CSV 行数仍为 1（条数不变）",
          len(read_csv(sb_csv)) == 1, f"实际 {len(read_csv(sb_csv))} 行")

    # ② 回答部分过期：1 条有效 + 1 条 kb_fp 过期 → 过期那条**不得**进裁判（也进不了分母）
    stale = _gate_ok_rec()
    stale["kb_fp"] = "old-kb"
    stale["question_id"] = "NSCLC-002"
    stale["model_name"] = "deepseek-flash"
    (runs / "gate__open_full.jsonl").write_text(
        json.dumps(_gate_ok_rec(), ensure_ascii=False) + "\n"
        + json.dumps(stale, ensure_ascii=False) + "\n", encoding="utf-8")
    calls.clear()
    try:
        out_mix = run(m7c, ["7c_export_full_review.py", "--offline"])
        exit_mix = 0
    except SystemExit as e:
        out_mix, exit_mix = str(e), (e.code if isinstance(e.code, int) else 1)
    check("7c 回答**部分过期**：过期记录被排除且不进裁判（零调用）",
          not calls and "排除了 1 条" in out_mix, f"exit={exit_mix} calls={calls} out={out_mix[:300]}")
    check("7c 回答部分过期：导出行数 = 1（不是把过期的那条也算进去）",
          len(read_csv(sb_csv)) == 1, f"实际 {len(read_csv(sb_csv))} 行")
    m7c._OFFLINE["on"] = False          # 复位模块级开关，避免污染后续用例


def check_suggest_cause_classification():
    """断言（复验 R4）：成因候选**先查完整知识库**，不再把"top-50 未命中"写成"库/题缺"。

    背景：9 条被标成「知识库或题目缺」的登记片段**全部存在于 `kb/chunks.jsonl`**，
    真实原因是"库内存在但没进模型上下文"。把这两者混为一谈，会得出"问题出在题库/知识库"的错误结论。
    """
    msc = load("suggest_cause")
    c = msc.classify
    check("成因判据：登记片段在库内但没进 top-6 → 「库内存在但未进上下文」",
          c(False, "D07-C047", True, True)[0] == "库内存在但未进上下文", c(False, "D07-C047", True, True))
    check("成因判据：登记证据已进 top-6 → 「模型没读出证据」",
          c(True, "D01-C001", True, True)[0] == "模型没读出证据")
    check("成因判据：登记片段确实不在库内（可答题）→ 「登记片段不在库内」（不再写『知识库或题目缺』）",
          c(False, "D99-C999", False, True)[0] == "登记片段不在库内")
    check("成因判据：陷阱题没有登记片段 → 「登记片段不在库内」（属设计如此，非缺陷）",
          c(False, "", False, False)[0] == "登记片段不在库内")
    check("成因判据：库内是否存在**优先于**检索名次（不再有『知识库或题目缺』这个结论）",
          all(c(g, s, k, a)[0] != "知识库或题目缺"
              for g in (True, False) for s in ("D01-C001", "") for k in (True, False)
              for a in (True, False)))
    check("成因判据的免责限定写进「依据」（登记片段未命中 ≠ 其它片段无等效证据）",
          "等效证据" in msc.CAVEAT and "免责" in msc.CAVEAT, msc.CAVEAT)
    # Public snapshot omits the private production table; do not count these as passes.
    if PUBLIC_RELEASE:
        print("  [SKIP] --public：未分发生产成因表，跳过 2 项生产数据断言；上面的模拟判据仍执行。")
        return
    # 生产候选表：不能还剩「知识库或题目缺」
    rows = read_csv(ROOT / "qa" / "成因待确认.csv")
    left = [r for r in rows if "知识库或题目缺" in (r.get("候选成因(自动)") or "")]
    check("生产 qa/成因待确认.csv 已无『知识库或题目缺』标注", not left,
          f"仍有 {len(left)} 条：{[r['题号'] for r in left][:5]}")
    check("生产 qa/成因待确认.csv 新增「登记片段是否在库内(客观)」列",
          bool(rows) and "登记片段是否在库内(客观)" in rows[0])
    if rows:
        in_kb = sum(1 for r in rows if (r.get("登记片段是否在库内(客观)") or "").startswith("是"))
        print(f"  [i] 生产候选表 {len(rows)} 条，其中登记片段在库内 {in_kb} 条；"
              f"候选分布 {dict((k, sum(1 for r in rows if r['候选成因(自动)'] == k)) for k in {r['候选成因(自动)'] for r in rows})}")


def _open_export_vs_full_bad(runs_dir):
    """逐题比对 `*__open.json` 与 `*__open_full.jsonl`；返回 {slug: [问题...]}。

    比对三项：模型回答 / 参考答案 / 检索上下文。只看**共有的题号**——
    范围外的题（`--limit` 试跑留下的）本来就不该出现在导出里。
    """
    out = {}
    for f in sorted(Path(runs_dir).glob("*__open.json")):
        slug = f.name.replace("__open.json", "")
        full = Path(runs_dir) / f"{slug}__open_full.jsonl"
        if not full.exists():
            continue
        try:
            js = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            out[slug] = [f"导出文件无法解析：{type(e).__name__}"]
            continue
        idx = {}
        for e in js:
            idx.setdefault((e.get("user_input"), e.get("reference")), []).append(e)
        bad = []
        for line in full.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cand = idx.get((r["question"], r["reference"]))
            if not cand:
                bad.append(f"{r['question_id']}:导出缺该题")
                continue
            e = cand[0]
            if e.get("response") != r.get("response"):
                bad.append(f"{r['question_id']}:回答不一致")
            if list(e.get("retrieved_contexts") or []) != list(r.get("retrieved_contexts") or []):
                bad.append(f"{r['question_id']}:上下文不一致")
        out[slug] = bad
    return out


def check_open_export_consistency():
    """断言（复验 R5）：`runs/*__open.json` 与 `*__open_full.jsonl` **逐题一致**（隔离副本）。

    背景：`NSCLC-027`/`NSCLC-032` 在 `deepseek-flash__open.json` 里是**旧回答**，
    而 `5_eval_rageval.py` 读的正是这个导出文件 → 继续走该入口仍会评到旧回答。
    这里在**沙箱里**构造"导出有旧回答"和"导出已同步"两种情形，证明该断言真的能抓出问题。
    """
    runs = _sandbox_dir("runs")
    for f in runs.glob("*__open*"):
        f.unlink()
    rec = _mk_rec(qid="NSCLC-001")
    (runs / "m1__open_full.jsonl").write_text(
        json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    (runs / "m1__open.json").write_text(
        json.dumps([{"user_input": rec["question"], "reference": rec["reference"],
                     "response": "旧回答（未同步）", "retrieved_contexts": rec["retrieved_contexts"]}],
                   ensure_ascii=False), encoding="utf-8")
    bad = _open_export_vs_full_bad(runs)
    check("runs/*__open.json 与 full 不一致时能被抓出（回答不一致）",
          bool(bad.get("m1")), str(bad)[:300])
    (runs / "m1__open.json").write_text(
        json.dumps([{"user_input": rec["question"], "reference": rec["reference"],
                     "response": rec["response"], "retrieved_contexts": rec["retrieved_contexts"]}],
                   ensure_ascii=False), encoding="utf-8")
    bad2 = _open_export_vs_full_bad(runs)
    check("同步后 runs/*__open.json 与 full 逐题一致",
          bad2.get("m1") == [], str(bad2)[:300])

    if PUBLIC_RELEASE:
        print("  [SKIP] --public：未分发真实回答与上下文；跳过生产导出检查，模拟同步/不同步断言仍执行。")
        return
    # 真实数据也算一遍（只读；真实目录不在沙箱快照里，所以不会被"未修改真实数据"的自检误伤）
    real = _open_export_vs_full_bad(ROOT / "runs")
    for slug, problems in real.items():
        check(f"真实 runs/{slug}：__open.json 与 full 逐题一致", not problems,
              f"{len(problems)} 处：{problems[:5]}")
    check("真实 runs/ 至少有一个 __open.json 被检查（否则断言空转）", bool(real),
          f"实际 {len(real)} 个")


if __name__ == "__main__":
    if (ROOT / SB).exists():
        shutil.rmtree(ROOT / SB)
    (ROOT / SB).mkdir(parents=True, exist_ok=True)
    before = real_snapshot()

    os.chdir(ROOT)
    install_stubs()

    kb = ROOT / CFG["paths"]["chunks"]
    kb.parent.mkdir(parents=True, exist_ok=True)
    with open(kb, "w", encoding="utf-8") as f:
        for i in range(1, 5):
            f.write(json.dumps({"chunk_id": f"D01-C{i:03d}", "doc_id": "D01", "file": "01_fake.pdf",
                                "text": f"（假文献）第 {i} 段。{GOOD}，HR 0.52。" * 20},
                               ensure_ascii=False) + "\n")

    review = ROOT / CFG["paths"]["review_csv"]

    print("=" * 16, "1) 2_gen_qa：出题 + 陷阱验证标记", "=" * 16)
    m2 = load("2_gen_qa")
    m2.chat = fake_chat
    m2.retrieve = fake_retrieve
    out = run(m2, ["2_gen_qa.py", "--per-doc", "2", "--per-chunk", "1", "--traps", "2"])
    rows = read_csv(review)
    check("审核表含「陷阱验证」列", "陷阱验证" in rows[0])
    trap_rows = [r for r in rows if r["答案能否从文献得到"] == "否"]
    check("陷阱题写入带知识库指纹的验证标记",
          len(trap_rows) == 2 and all(
              re.search(r"已验证\(kb-[0-9a-f]{6,16},top-\d+,[0-9a-f]{8}\)", r["陷阱验证"])
              for r in trap_rows),
          str([r["陷阱验证"] for r in trap_rows]))

    print("=" * 16, "2) 3_finalize_qa：入库门槛", "=" * 16)
    for r in rows:
        r["医学正确性(填:对/错/需改)"] = "对"
    write_csv(review, rows)
    m3 = load("3_finalize_qa")
    out = run(m3, ["3_finalize_qa.py"])
    qa = read_jsonl(ROOT / CFG["paths"]["qa_set"])
    check("全部合规时 3 题入库", len(qa) == 3, f"n={len(qa)}")
    check("陷阱题带 trap_verified/trap_verify_k",
          all(x["trap_verified"] and x["trap_verify_k"] for x in qa if not x["answerable"]))
    check("陷阱题不计入医学正确性确认", all(not x["medically_verified"] for x in qa if not x["answerable"]))

    # ① 手工把普通题改成"否"（没有验证标记）→ 必须拦下
    rows[0]["答案能否从文献得到"] = "否"
    rows[0]["陷阱验证"] = ""
    write_csv(review, rows)
    out = run(m3, ["3_finalize_qa.py"])
    check("手工改成「否」的题被拦下（不能绕过验证）",
          "缺少有效验证标记" in out, out[-600:])
    check("此时只有 2 道真陷阱题入库", len(read_jsonl(ROOT / CFG["paths"]["qa_set"])) == 2)

    # ② 已验证的陷阱题被改问题文本 → 验证失效 → 拦下
    rows[0]["答案能否从文献得到"] = "是"
    write_csv(review, rows)
    trap_rows = [r for r in rows if r["答案能否从文献得到"] == "否"]
    trap_rows[0]["修改后的问题(留空=不改)"] = "（假）被改过的陷阱问题，哈希应当失效？"
    write_csv(review, rows)
    out = run(m3, ["3_finalize_qa.py"])
    check("改过问题文本的陷阱题被拦下（验证失效）", "原验证失效" in out, out[-600:])
    check("此时入库 2 题（1 道陷阱被拦）", len(read_jsonl(ROOT / CFG["paths"]["qa_set"])) == 2)

    # ③ 修复后重新验证（--recheck-traps）
    trap_rows[0]["修改后的问题(留空=不改)"] = ""
    write_csv(review, rows)
    run(m2, ["2_gen_qa.py", "--recheck-traps"])
    rows2 = read_csv(review)
    check("--recheck-traps 刷新了验证标记",
          all(re.search(r"已验证\(kb-[0-9a-f]{6,16},top-\d+,[0-9a-f]{8}\)", r["陷阱验证"])
              for r in rows2 if r["答案能否从文献得到"] == "否"))
    run(m3, ["3_finalize_qa.py"])
    qa = read_jsonl(ROOT / CFG["paths"]["qa_set"])
    check("修复后 3 题全部入库", len(qa) == 3, f"n={len(qa)}")
    n = len(qa)

    print("=" * 16, "3) 4_rag_answer：指纹 / --limit / 单模式", "=" * 16)
    m4 = load("4_rag_answer")
    m4.chat = fake_chat
    m4.retrieve = fake_retrieve
    f_open = ROOT / CFG["paths"]["runs"] / "deepseek-flash__open_full.jsonl"
    f_closed = ROOT / CFG["paths"]["runs"] / "deepseek-flash__closed_full.jsonl"
    closed_artifact = (ROOT / CFG["paths"]["qa_set"]).parent / "closed" / "deepseek-flash.jsonl"

    out1 = run(m4, ["4_rag_answer.py", "--workers", "2"])
    check("首次运行：全部作答", f"本次待答 {n * 2} 条" in out1, out1[-400:])
    check("无变更重跑：零调用", "本次待答 0 条" in run(m4, ["4_rag_answer.py", "--workers", "2"]))

    CTX["text"] = CTX["text"] + "【尾部新增内容，前 80 字符未变】"
    out3 = run(m4, ["4_rag_answer.py", "--workers", "2"])
    check("只改片段后半段也会重答", f"本次待答 {n} 条" in out3, out3[-400:])

    _PROMPT_VER_PROD = m4.PROMPT_VER
    m4.PROMPT_VER = "v2-test"
    out4 = run(m4, ["4_rag_answer.py", "--workers", "2"])
    check("提示词版本变化会重答", f"本次待答 {n * 2} 条" in out4, out4[-400:])

    # [!] 必须恢复生产提示词版本再重跑：汇总脚本（7_report）会**重新加载** 4_rag_answer 做指纹复核，
    #     它看到的是源文件默认版本。若不恢复，测试生成的记录会因"提示词版本不一致"被正式校验拦下
    #     ——那是**正确行为**，但会让后续汇总测试失败。所以这里恢复并重答，让桩数据与生产配置一致。
    m4.PROMPT_VER = _PROMPT_VER_PROD
    out4b = run(m4, ["4_rag_answer.py", "--workers", "2"])
    check("恢复生产提示词版本并重答（使测试数据与生产配置一致，不改动生产校验）",
          f"本次待答 {n * 2} 条" in out4b, out4b[-400:])

    out5 = run(m4, ["4_rag_answer.py", "--limit", "1", "--workers", "2"])
    check("--limit 试跑后范围外记录仍完整",
          len(read_jsonl(f_open)) == n and len(read_jsonl(f_closed)) == n and "本次待答 0 条" in out5,
          f"open={len(read_jsonl(f_open))} closed={len(read_jsonl(f_closed))} 期望={n}")

    out6 = run(m4, ["4_rag_answer.py", "--mode", "open", "--workers", "2"])
    check("单模式 --mode open 不报错（不再 KeyError）", "本次待答 0 条" in out6, out6[-500:])
    check("单模式不动另一个模式的记录", len(read_jsonl(f_closed)) == n, f"closed={len(read_jsonl(f_closed))}")

    recs = read_jsonl(f_open)
    recs[0]["error"] = True
    recs[0]["response"] = "[调用失败] 模拟"
    f_open.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    out7 = run(m4, ["4_rag_answer.py", "--workers", "2"])
    check("失败记录会被重试", "上次失败重试 1" in out7, out7[-400:])

    # 改一道题后只跑 open：closed 记录保留但**不进产物**
    qa_now = read_jsonl(ROOT / CFG["paths"]["qa_set"])
    qa_now[0]["question"] = qa_now[0]["question"] + "（改过）"
    (ROOT / CFG["paths"]["qa_set"]).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in qa_now) + "\n", encoding="utf-8")
    out8 = run(m4, ["4_rag_answer.py", "--mode", "open", "--workers", "2"])
    check("改题后单模式只重答该题（开卷 1 条）", "本次待答 1 条" in out8, out8[-400:])
    check("closed 记录未被删除", len(read_jsonl(f_closed)) == n, f"closed={len(read_jsonl(f_closed))}")
    check("失效的 closed 记录被排除出评测产物",
          len(read_jsonl(closed_artifact)) == n - 1,
          f"产物={len(read_jsonl(closed_artifact))} 期望={n - 1}")

    print("=" * 16, "4) 5b：判定指纹 + --limit 安全", "=" * 16)
    m5 = load("5b_judge_faithfulness")
    m5.chat = fake_chat
    run(m5, ["5b_judge_faithfulness.py", "--workers", "2"])
    faith = read_jsonl(ROOT / CFG["paths"]["outputs"] / "faithfulness.jsonl")
    check("首次判定覆盖全部开卷记录", len(faith) == n, f"faith={len(faith)} 期望={n}")
    out9 = run(m5, ["5b_judge_faithfulness.py", "--limit", "1", "--workers", "2"])
    check("5b --limit 不删范围外判定",
          len(read_jsonl(ROOT / CFG["paths"]["outputs"] / "faithfulness.jsonl")) == n
          and "待判 0 条" in out9)

    print("=" * 16, "5) 7_report：RAGAS 只认真实产物", "=" * 16)
    m7 = load("7_report")
    outdir = ROOT / CFG["paths"]["outputs"]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "手工汇总.csv").write_text(
        "model,dataset,metric,score\nDeepSeek,RAG开卷,Faithfulness,0.85\n", encoding="utf-8-sig")
    demo = outdir / "rageval_demo"
    demo.mkdir(exist_ok=True)
    (demo / "x.json").write_text("{}", encoding="utf-8")          # 用户指出的假信号
    run(m7, ["7_report.py", "--no-judge"])
    md = (outdir / "榜单.md").read_text(encoding="utf-8")
    check("内容为 {} 的演示文件不会被判为已运行 RAGEval",
          "未使用** RAGEval/RAGAS" in md and "已检测到 outputs/ 中的对应产物" not in md, md[:900])
    check("会列出可疑文件供核实", "不判定为已运行" in md, md[:1200])

    (demo / "report.json").write_text('{"metric": "Faithfulness", "score": 0.81}', encoding="utf-8")
    run(m7, ["7_report.py", "--no-judge"])
    md2 = (outdir / "榜单.md").read_text(encoding="utf-8")
    check("有含指标字样的真实产物才声称 RAGEval/RAGAS",
          "已检测到 outputs/ 中的对应产物" in md2, md2[:900])
    check("报告含忠实性第 4 节", "## 4. 忠实性" in md2)

    print("=" * 16, "6) 知识库变更的影响（两条铁律）", "=" * 16)
    # 重写 chunks.jsonl = 模拟"重建知识库"（指纹按 chunks 现算，因为沙箱里没有 kb_version.json）
    with open(kb, "w", encoding="utf-8") as f:
        for i in range(1, 5):
            f.write(json.dumps({"chunk_id": f"D01-C{i:03d}", "doc_id": "D01", "file": "01_fake.pdf",
                                "text": f"（假文献·已重建）第 {i} 段。{GOOD}。" * 20},
                               ensure_ascii=False) + "\n")
    m4._KB_FP["v"] = None          # 真实场景每次都是新进程；测试里手动清缓存
    m2._KBF["v"] = None
    out_kb = run(m4, ["4_rag_answer.py", "--limit", "1", "--workers", "2"])
    check("知识库变更后，旧记录被识别为旧库版本并告警",
          "旧知识库版本" in out_kb and "全量重跑" in out_kb, out_kb[-700:])
    check("旧库记录被排除出评测产物（试跑不能拼出榜单）",
          len(read_json_array(ROOT / CFG["paths"]["runs"] / "deepseek-flash__open.json")) == 1,
          f"产物={len(read_json_array(ROOT / CFG['paths']['runs'] / 'deepseek-flash__open.json'))} 期望=1")
    out_kb2 = run(m4, ["4_rag_answer.py", "--workers", "2"])
    check("全量重跑会补齐所有旧库记录", f"本次待答 {(n - 1) * 2} 条" in out_kb2, out_kb2[-500:])
    check("全量重跑后产物恢复完整",
          len(read_json_array(ROOT / CFG["paths"]["runs"] / "deepseek-flash__open.json")) == n
          and len(read_jsonl(closed_artifact)) == n,
          f"open={len(read_json_array(ROOT / CFG['paths']['runs'] / 'deepseek-flash__open.json'))} "
          f"closed={len(read_jsonl(closed_artifact))}")

    out_f = run(m3, ["3_finalize_qa.py"])
    check("知识库变更后陷阱题验证失效被拦下",
          "知识库已变更" in out_f, out_f[-700:])
    check("此时只有 1 道可答题入库", len(read_jsonl(ROOT / CFG["paths"]["qa_set"])) == 1)
    run(m2, ["2_gen_qa.py", "--recheck-traps"])
    run(m3, ["3_finalize_qa.py"])
    check("重验后 3 题重新入库", len(read_jsonl(ROOT / CFG["paths"]["qa_set"])) == 3)

    print("=" * 16, "7) 缓存键/指纹**缺维度**（已复发两次，专项断言）", "=" * 16)
    check_cache_key_dimensions()

    print("=" * 16, "8) 续跑额度不降级（6000 额度有效记录零调用）", "=" * 16)
    check_budget_no_downgrade()

    print("=" * 16, "9) 重答后的判定缓存失效键（显示名 vs 模型ID）", "=" * 16)
    check_reanswer_cache_invalidation()

    print("=" * 16, "10) 7c 导出的「问题」列含 A~D 选项", "=" * 16)
    check_export_question_column()

    print("=" * 16, "11) 裁判价目：est_cost() 对裁判不为 0", "=" * 16)
    check_judge_price()

    print("=" * 16, "12) `.gitignore` 放行规则真实生效（跑真实 git check-ignore）", "=" * 16)
    check_gitignore_rules()

    print("=" * 16, "13) 7c 人工列保护 + 判定输入版本闸门（复验 R2/R3）", "=" * 16)
    check_7c_manual_columns_and_gate()

    print("=" * 16, "14) runs/*__open.json 与 full 记录逐题一致（复验 R5）", "=" * 16)
    check_open_export_consistency()

    print("=" * 16, "15) 成因候选：先查完整知识库（复验 R4）", "=" * 16)
    check_suggest_cause_classification()

    print("\n" + "=" * 60)
    after = real_snapshot()
    check("真实数据目录的文件内容前后完全一致", before == after,
          f"新增={sorted(set(after) - set(before))} 减少={sorted(set(before) - set(after))} "
          f"变化={[k for k in set(before) & set(after) if before[k] != after[k]]}")
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for x in FAIL:
            print("  -", x)
        sys.exit(1)
    print("[OK] 全部通过（桩数据下的流程验证；正式结果以真实跑批为准）")
