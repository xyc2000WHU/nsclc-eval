# -*- coding: utf-8 -*-
r"""
全量 450 条人工复核表导出（含裁判自动判定，可加第二位裁判）。

用途：把 30 条抽检表扩展为**全量 450 条**的复核表，其中
  · 「自动判定」= 裁判模型对"模型回答 vs 标准答案"的判定（缓存到 outputs/auto_verdicts_*.jsonl）
  · 「人工判定」「错误类型」「备注」留空给你填
  · 可用 --judge2 追加**第二位裁判**的判定列，用于多裁判一致性比较

用法：
  python src\7c_export_full_review.py                       # 只跑配置里的裁判（第 1 位）
  python src\7c_export_full_review.py --judge2 <模型名>      # 追加第二位裁判（同一批题、同一提示词）
  python src\7c_export_full_review.py --limit 10            # 先小样本试
产物：qa\人工抽检表_全量450.csv（或 _测试 后缀）、outputs\auto_verdicts_*.jsonl

[!] 已导出过的 qa\人工抽检表_全量450.csv 是**旧版**：那一版「问题」列只有裸题干（没有 A~D 选项）。
    本文件已改成 full_question(item)（含选项），但**没有重跑导出**（避免任何 API 花费）；
    下次运行时生效——下一版 CSV 的「问题」列会与裁判提示词、7d 盲评包用同一个拼法（含选项）。
    判分输入（fp）本来就含选项，所以这次改动**不会**让判定缓存失效、不需要重判。
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import (CFG, chat, dirpath, full_question, g_col, kb_fingerprint, path,
                    read_csv_any, read_jsonl)   # noqa: E402

_spec = importlib.util.spec_from_file_location("r7", ROOT / "src" / "7_report.py")
r7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r7)
PROMPT = r7.JUDGE_PROMPT          # 复用抽检表用的同一提示词，保证与 30 条口径一致
LABELS = ("一致", "部分一致", "不一致")
# [!] 判分**输入**格式版本：改提示词、改题面拼法（如加入 A~D 选项）时必须 +1，
#     否则历史判定会被当成"仍然有效"而静默复用（本文件此前踩过的正是这一类坑）。
#     它同时进入 fp，所以提升它 = 强制重判，这是刻意的。
JUDGE_INPUT_VER = "v1-含选项题面"
COLS = ["序号", "题目ID", "模型", "题型", "是否可答", "问题", "标准答案", "原文证据句",
        "模型回答", "自动判定", "自动判定2", "人工判定(填:一致/部分一致/不一致)",
        "错误类型(填:数值错误/结论错误/证据不足/超出文献/其他)", "备注"]

_KB_FP = {"v": None}
# [!] 离线开关（`--offline`）。为什么要有这个模块级开关，而不是只传参数：
#     ① 传参数是主闸门（`judge_verdict(offline=True)`）——它不依赖任何赋值顺序；
#     ② 这个开关是**纵深防御**：万一将来有人在别处**误用模块级 `chat`**，
#        `_guarded_chat()` 也会挡住，不会静默产生费用。
#     [!] 注意：**不要**再用 `chat = _no_api` 那种写法。`main()` 里的裸赋值只创建**局部变量**，
#     而 `judge_verdict()` 读的是模块级 `chat` → 闸门静默失效（已实测：缓存未命中仍真实调用裁判、
#     不退出，日志却还在打印"保证零 API 花费"）。这正是复验第 1 项残留的根因。
_OFFLINE = {"on": False}


def kb_fp() -> str:
    """知识库指纹（缓存一次）。进判定指纹：知识库变了，旧判定就不再代表当前语料。"""
    if _KB_FP["v"] is None:
        _KB_FP["v"] = kb_fingerprint()
    return _KB_FP["v"]


def build_prompt(item, rec) -> str:
    """判定提示词（**含 A~D 选项的完整题面**）。

    [!] 提示词要求"选择题若只给选项字母，按该选项的文字内容比较"，但过去只传裸题干，
        裁判看不到选项文字 → 只能猜。抽 common.full_question 保证与 7_report 的抽检表
        （以及 7d 的盲评包列 m̃「问题（含选项）」）是同一个拼法。
    """
    return PROMPT.format(question=full_question(item), reference=item["reference_answer"],
                         response=(rec.get("response") or "")[:1500])


def fp_of(item, rec, judge, ver=None) -> str:
    """判定指纹 = md5(完整判定提示词 + 裁判名/base_url + 输入格式版本 + 知识库指纹)。

    参考 5b_judge_faithfulness.fp_of 的写法。为什么必须这么细：
      · 完整题面（含 A~D 选项）：选项文字变了，判定依据就变了；
      · 标准答案：题干/答案被改过必须重判；
      · 模型回答**全文**（截断前）：回答改一个字也要重判；
      · 裁判名 + base_url：同一模型名挂在不同网关背后是不同裁判，判定不可混；
      · 提示词/口径版本（JUDGE_INPUT_VER）：判分规则变了必须重判；
      · kb_fp：知识库换了，陷阱题的标准答案含义可能变了。
    缓存键只有 (被评模型, 题号) 时，上面任何一项变化都会**静默复用旧判定**（曾经发生）。
    `ver` 只为测试/对照留的口子（默认取 JUDGE_INPUT_VER）。
    """
    raw = json.dumps({
        "ver": ver if ver is not None else JUDGE_INPUT_VER,
        "judge": judge.get("name"), "base_url": judge.get("base_url"),
        "kb": kb_fp(), "prompt": build_prompt(item, rec),
        "reference": item["reference_answer"],
        "response": rec.get("response") or "",
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _parse_verdict(text) -> str:
    """把裁判输出解析成三个标签之一；解析不出来返回"解析失败"。"""
    v = "解析失败"
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        v = str(obj.get("judge", "")).strip() or "解析失败"
    except Exception:
        pass
    if v not in LABELS:
        found = set(re.findall(r"(?:judge|判定|结论)\s*[:：]\s*[\"「『]?\s*(一致|部分一致|不一致)", text or ""))
        v = found.pop() if len(found) == 1 else "解析失败"
    return v


def flush(cache, name):
    """**逐条增量落盘**：任何暂停/中断都不再浪费已经花掉的裁判调用。"""
    if not name:
        return
    with open(dirpath("outputs") / f"auto_verdicts_{name}.jsonl", "w", encoding="utf-8") as fh:
        for v in cache.values():
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")


# ---------------- 人工列保护（覆盖前备份 + 按 (模型,题号,回答指纹) 保留）----------------
MAN_COLS = ("人工判定(填:一致/部分一致/不一致)",
            "错误类型(填:数值错误/结论错误/证据不足/超出文献/其他)", "备注")
CHANGED_NOTE = "回答已变更，需重新审核"
_row_fp = {}


def row_fp(row) -> str:
    """被评回答 + 题面 + 标准答案的内容指纹（**不含**人工列）。

    [!] 为什么按它判断"回答是否已变更"：只有回答/题面/标准答案变了，已填的人工判定
        才可能不再成立。用整行哈希会把"自动判定刷新"也判成变更，从而把有效人工列标成待复核。
    """
    key = id(row)
    if key not in _row_fp:
        raw = json.dumps([row.get("问题", ""), row.get("标准答案", ""), row.get("模型回答", "")],
                         ensure_ascii=False, sort_keys=True)
        _row_fp[key] = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return _row_fp[key]


def backup_file(p: Path) -> Path:
    """覆盖前自动备份（带时间戳）。**不依赖调用方记得备份**。

    命名冲突时追加序号；备份失败 → 抛异常并放弃覆盖（宁可不写，也不能毁掉人工填写）。
    """
    if not Path(p).exists() or not Path(p).read_bytes().strip():
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    dst = Path(f"{p}.bak-{ts}")
    n = 1
    while dst.exists():
        n += 1
        dst = Path(f"{p}.bak-{ts}-{n}")
    try:
        shutil.copy2(p, dst)
    except Exception as e:
        raise SystemExit(f"[X] 备份失败（{p} → {dst}）：{type(e).__name__}: {e}\n"
                         f"    为避免清空已填的人工列，本次导出已终止。请先手工备份后重试。")
    return dst


def load_prev_manual(csv_path: Path):
    """读回上一版导出表的**人工三列**，键为 (模型, 题号)。

    用 common.read_csv_any（多编码兜底）：这张表是用户在 Excel 里填的，
    Excel 存 GBK 的概率很高，只认 utf-8-sig 会把已填内容读成乱码或直接崩溃。
    """
    prev = {}
    for r in read_csv_any(csv_path):
        m, q = g_col(r, "模型"), g_col(r, "题目ID")
        if not m or not q:
            continue
        prev[(m, q)] = {
            "人工判定": g_col(r, "人工判定", prefix=True),
            "错误类型": g_col(r, "错误类型", prefix=True),
            "备注": g_col(r, "备注"),
            "回答": r.get("模型回答") or "",
            "row_fp": row_fp(r),
        }
    return prev


def apply_manual_keep(rows, prev):
    """把上一版**已填**的人工三列并回新行。

    关键行为（原实现每次把三列写空，等于静默清空人工结果）：
      · 上一版没填 → 新行保持空白；
      · 上一版填了、回答指纹未变 → **原样保留**（含备注原文）；
      · 上一版填了、回答指纹已变 → **保留填写内容**，并在备注前标注
        「回答已变更，需重新审核」（绝不静默丢弃，也不静默当成仍然有效）。
    返回 (保留条数, 标注待复核条数)。
    """
    kept = flagged = 0
    for row in rows:
        old = prev.get((row["模型"], row["题目ID"]))
        if not old:
            continue
        v, e, note = old["人工判定"], old["错误类型"], old["备注"]
        if not (v or e or note):
            continue                       # 上一版也没填 → 保持空白
        row[MAN_COLS[0]] = v
        row[MAN_COLS[1]] = e
        if old["row_fp"] == row_fp(row):
            row[MAN_COLS[2]] = note
            kept += 1
        else:
            row[MAN_COLS[2]] = f"[{CHANGED_NOTE}] " + note
            kept += 1
            flagged += 1
    return kept, flagged


def _offline_block_reason(m, rec, old, cur_fp) -> str:
    """缓存未命中时，给出**人话原因**（用于 --offline 的报错与日志）。"""
    if old is None:
        return "该 (被评模型, 题号) 在判定缓存里完全没有记录"
    if old.get("fp") != cur_fp:
        return (f"缓存 fp={str(old.get('fp'))[:12]} ≠ 当前输入复算 fp={cur_fp}"
                f"（题面/选项/回答/标准答案/裁判/知识库/口径任一变化都会如此）")
    if old.get("judge") not in LABELS:
        return f"缓存里是「解析失败」（judge={old.get('judge')!r}），需重判"
    if old.get("judge_name") != m["name"]:
        return f"缓存行的 judge_name={old.get('judge_name')!r} 不是当前裁判 {m['name']!r}"
    return "缓存校验未通过（原因未归类）"


def _offline_hard_stop(m, rec, old, cur_fp):
    """--offline 硬闸门：**缓存未命中之后、调用裁判之前**拒绝执行。

    为什么单列一个函数：报错文本与退出码只有一处定义，`judge_verdict()` 与 `_guarded_chat()`
    都用它，避免两处漂移。退出码固定为 2（非 0），调用方能明确判定"被拦住"。
    """
    reason = _offline_block_reason(m, rec, old, cur_fp)
    print(f"[X] --offline：{rec.get('model')}/{rec.get('question_id')} 需要真实调用裁判"
          f"（{m.get('name')}）→ 拒绝执行，未发起任何请求。原因：{reason}")
    raise SystemExit(2)


def _guarded_chat(m, messages, **kw):
    """模块级 `chat` 的安全包装：offline 打开时拒绝**任何**真实调用（纵深防御）。

    正常情况下 offline 已经在 `judge_verdict()` 里被拦住（**显式参数**），走不到这里；
    这里兜底的是"别处直接调用模块级 chat"的情况。
    """
    if _OFFLINE["on"]:
        raise SystemExit(2)
    return chat(m, messages, **kw)


def judge_verdict(m, item, rec, cache, stats=None, offline=False):
    """取判定：缓存命中就复用，否则调用裁判。返回 (判定, 是否命中缓存)。

    命中条件（三者缺一不可，原实现只查键 → 改回答/改口径会静默复用旧判定）：
      ① 缓存里的 `fp` 与当前输入复算的 fp 一致；
      ② 缓存的 `judge` 在合法标签集里（"解析失败"视为未判，下次重试）；
      ③ 裁判名一致。

    [!] `offline=True` = `--offline` 硬闸门：**只在缓存未命中之后、调用裁判之前**拒绝执行。
        为什么必须是**显式参数**而不是靠模块级 `chat`：`main()` 里写 `chat = _no_api`
        只创建了一个**局部变量**，而本函数调用的是**模块级 `chat`** → 闸门形同不存在，
        缓存未命中时照样真实请求裁判且**不退出**，日志却还在打印"保证零 API 花费"（已实测复现）。
        所以这里把 offline 当参数传进来，闸门本身不依赖任何外部赋值顺序。
    """
    key = (rec["model"], rec["question_id"])
    cur_fp = fp_of(item, rec, m)
    old = cache.get(key)
    if old is not None and old.get("fp") == cur_fp \
            and old.get("judge") in LABELS and old.get("judge_name") == m["name"]:
        return old.get("judge", ""), True
    # ---- 缓存未命中：先看 offline 闸门，绝不再往下走 ----
    # [!] 主闸门 = **显式参数**（不依赖任何外部赋值顺序）；模块级开关只在没传参时才兜底。
    #     成功走完一次 --offline 后 `main()` 会复位模块级开关（见 main 尾部），
    #     所以"离线模式"不会泄漏到同进程里的正常调用（回归测试里踩过这个坑）。
    if offline:
        _offline_hard_stop(m, rec, old, cur_fp)
    if stats is not None and old is not None:
        if old.get("fp") != cur_fp:
            stats["输入已变更重判"] = stats.get("输入已变更重判", 0) + 1
        elif old.get("judge") not in LABELS:
            stats["上次解析失败重判"] = stats.get("上次解析失败重判", 0) + 1
    text, usage = _guarded_chat(m, [{"role": "user", "content": build_prompt(item, rec)}],
                                temperature=0.0, max_tokens=3000)
    v = _parse_verdict(text)
    # [!] 落盘的行必须带**被评模型**：早期存的是裁判名，读回来键就退化成 (裁判名, 题号)，
    #     同一题号下 5 个模型的判定会互相覆盖（同一根因犯过两次，这次把两者都写清楚）。
    # [!] 同时写 `model_name`（模型 ID）：缓存行的 model 是显示名，而 runs 记录里
    #     比对/失效逻辑要用 ID；两个名字都存下来，读取方不必再靠模糊匹配。
    reason = ""
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        reason = str(obj.get("reason", "") or "")[:200]
    except Exception:
        pass
    cache[key] = {"model": rec["model"], "model_name": rec.get("model_name", ""),
                  "judge_name": m["name"], "question_id": rec["question_id"], "judge": v,
                  "reason": reason,
                  "raw_head": (text or "")[:200] if v not in LABELS else "",
                  "fp": cur_fp, "prompt_ver": JUDGE_INPUT_VER,
                  "in_tokens": usage.get("in_tokens", 0), "out_tokens": usage.get("out_tokens", 0)}
    return v, False


def main():
    """入口：**离线开关的生命周期只包在 main() 内**（成功或异常退出都会复位）。

    [!] 为什么必须这样包：`python src\\7c...py` 每次都是新进程，开关"泄漏"看不出来；但同进程里
        连续调用 `main()`（回归测试、交互式调试、别的脚本 import 本模块）时，
        一次 `--offline` 会让**后续的正常调用**被兜底闸门误拦成退出码 2（实测踩过）。
        finally 里的复位只是"进程内状态卫生"，不改变任何产物。
    """
    try:
        _main_impl()
    finally:
        _OFFLINE["on"] = False


def _main_impl():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge2", default=None, help="第二位裁判的模型名（同一家或另一家均可）")
    ap.add_argument("--base-url2", default=None, help="第二位裁判的 base_url（不给则沿用第 1 位）")
    ap.add_argument("--key-env2", default=None, help="第二位裁判的 Key 环境变量名")
    ap.add_argument("--limit", type=int, default=0, help="只导出前 N 条（试跑用）")
    # [!] --offline：把裁判调用变成**硬失败**而不是真去请求。
    #     用途：只想重新导出 CSV（例如刷新「问题」列、或验证"全量命中缓存、零花费"）时，
    #     用它保证一旦有记录需要重判就立刻报错退出，**绝不会悄悄产生 API 费用**。
    ap.add_argument("--offline", action="store_true",
                    help="离线模式：任何需要调用裁判的条目都直接报错退出（保证零 API 花费）")
    a = ap.parse_args()
    if a.offline:
        # [!] 这里**不再**靠 `chat = _no_api` 拦调用（那只改了局部变量，`judge_verdict()` 读的是
        #     模块级 `chat` → 闸门静默失效，实测缓存未命中时仍真实调用裁判且不退出）。
        #     主闸门是把 offline **显式传给 judge_verdict()**（见该函数的 offline 参数）；
        #     这里再打开模块级开关，给 `_guarded_chat()` 兜底（纵深防御，退出码同样非 0）。
        _OFFLINE["on"] = True
        print("[i] 离线模式（--offline）：任何需要裁判调用的条目都会立即报错退出（退出码 ≠ 0），"
              "保证零 API 花费。")

    qa = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    # [!] 判定的输入必须与**正式汇总口径**一致：过去这里只查 `error=false` + 回答非空，
    #     于是改了题干/选项/知识库/提示词之后，旧回答仍会被包成"新判定"导出
    #     （已复现：kb_fp 改成 old-kb 的记录照样被送进裁判）。
    #     现在统一走 7_report.load_all()：它逐条校验 kb_fp、题目在正式集内、模型在当前 config 内，
    #     并用与 4_rag_answer 完全相同的定义**复算输入指纹**（题干/选项/标准答案/提示词版本/
    #     参数/检索片段）。过期记录被排除并打印原因，绝不静默混入。
    try:
        valid = r7.load_all("*_open_full.jsonl")
    except SystemExit as e:
        raise SystemExit(f"[X] 回答未通过版本校验 → 终止导出（不调用任何裁判）：\n{e}")
    recs = [r for r in valid if r.get("mode") == "open" and (r.get("response") or "").strip()]
    if not recs:
        raise SystemExit("[X] 没有任何通过当前版本校验的开卷回答 → 终止导出（不调用任何裁判）。\n"
                         "    这通常意味着题干/选项/提示词/知识库已变更，请先重跑 "
                         "python src\\4_rag_answer.py。")
    print(f"待判定 {len(recs)} 条（开卷有效作答；已通过 7_report.load_all 的版本校验）")
    if a.limit:
        recs = recs[:a.limit]

    j1 = CFG["judge"]
    j2 = None
    if a.judge2:
        j2 = dict(j1)
        j2["name"] = a.judge2
        j2["display"] = f"{a.judge2}(第2裁判)"
        if a.base_url2:
            j2["base_url"] = a.base_url2
        if a.key_env2:
            j2["api_key_env"] = a.key_env2

    cache1 = {(r["model"], r["question_id"]): r for r in
              read_jsonl(dirpath("outputs") / f"auto_verdicts_{j1['name']}.jsonl")}
    cache2 = ({(r["model"], r["question_id"]): r for r in
               read_jsonl(dirpath("outputs") / f"auto_verdicts_{j2['name']}.jsonl")} if j2 else {})
    stats1, stats2 = {}, {}

    rows = []
    for i, rec in enumerate(recs, 1):
        item = qa.get(rec["question_id"])
        if item is None:
            continue
        # [!] offline 必须**显式传参**（不要依赖模块级 chat 被替换）：见 judge_verdict 的注释。
        v1, c1 = judge_verdict(j1, item, rec, cache1, stats1, offline=a.offline)
        v2 = ""
        if j2:
            v2, _ = judge_verdict(j2, item, rec, cache2, stats2, offline=a.offline)
        rows.append({"序号": i, "题目ID": rec["question_id"], "模型": rec["model"],
                     "题型": rec.get("type", ""), "是否可答": "是" if rec.get("answerable") else "否",
                     # [!] 这一列必须是**完整题面（含 A~D 选项）**：判定提示词早就用 full_question()
                     #     带了选项，但导出的「问题」列一直只写裸题干 —— 人工复核时看不到选项文字，
                     #     没法核对模型回答里的字母（同一根因的另一半）。改这里**不影响判分输入与
                     #     判定缓存指纹**（fp 由 build_prompt 算，本来就含选项），所以不需要重判、零花费。
                     "问题": full_question(item), "标准答案": item["reference_answer"],
                     "原文证据句": item.get("evidence_quote", ""),
                     "模型回答": (rec.get("response") or "")[:4000],
                     "自动判定": v1, "自动判定2": v2,
                     "人工判定(填:一致/部分一致/不一致)": "",
                     "错误类型(填:数值错误/结论错误/证据不足/超出文献/其他)": "", "备注": ""})
        if i % 20 == 0:
            flush(cache1, j1["name"])
            if j2:
                flush(cache2, j2["name"])
            print(f"  {i}/{len(recs)}（缓存已增量落盘）")

    for nm, cch in ((j1["name"], cache1), (j2["name"] if j2 else None, cache2)):
        if not nm:
            continue
        with open(dirpath("outputs") / f"auto_verdicts_{nm}.jsonl", "w", encoding="utf-8") as fh:
            for v in cch.values():
                fh.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"  判定缓存 → outputs/auto_verdicts_{nm}.jsonl（{len(cch)} 条）")

    suffix = "_测试" if a.limit else ""
    out = path("qa_set").parent / f"人工抽检表_全量450{suffix}.csv"
    # [!] 覆盖前先备份，再按 (模型, 题号, 回答指纹) 并回人工三列 —— 这两步缺一不可：
    #     原实现每次把「人工判定/错误类型/备注」写空并以 w 覆盖，等于**静默清空**用户的填写。
    prev = load_prev_manual(out) if not a.limit else {}
    kept = flagged = 0
    if not a.limit:
        kept, flagged = apply_manual_keep(rows, prev)
        if prev:
            bak = backup_file(out)
            print(f"  覆盖前备份 → {bak.name if bak else '（原表为空/不存在，无需备份）'}")
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if not a.limit:
        print(f"  人工列保护：并回已填 {kept} 条"
              + (f"，其中 {flagged} 条回答已变更 → 备注已标注「{CHANGED_NOTE}」" if flagged else "")
              + ("（上一版没有已填的人工列）" if not prev else ""))
    from collections import Counter
    dist = Counter(r["自动判定"] for r in rows)
    print(f"\n[OK] 已导出 {len(rows)} 行 → {out}")
    print(f"  裁判1（{j1['name']}）判定分布：{dict(dist)}")
    if stats1:
        # [!] 显式打印"为什么重判"：只报总数的话，改了回答/口径导致的大批重判会被当成正常命中。
        print(f"  裁判1 缓存失效重判：{sum(stats1.values())} 条 {dict(stats1)}"
              f"（提示词/题面拼法/回答/标准答案/裁判/知识库任一变化都会失效）")
    n_fail = sum(1 for r in rows if r["自动判定"] not in LABELS)
    if n_fail:
        print(f"  [!] 裁判1 仍有 {n_fail} 条「解析失败」（已记 raw_head，下次运行会自动重判）")
    if j2:
        print(f"  裁判2（{j2['name']}）判定分布：{dict(Counter(r['自动判定2'] for r in rows))}")
        both = [(r["自动判定"], r["自动判定2"]) for r in rows]
        ag = sum(1 for x, y in both if x == y)
        print(f"  两裁判一致：{ag}/{len(both)} = {ag / len(both) * 100:.1f}%")


if __name__ == "__main__":
    main()

