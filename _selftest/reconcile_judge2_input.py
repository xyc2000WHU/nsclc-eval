# -*- coding: utf-8 -*-
r"""二号裁判材料对账（**离线**，零 LLM / 零 API 调用）。

为什么需要这个脚本（第三方复验新确认的缺口）
------------------------------------------------
二号裁判（你方交付）的 450 条判定是**建立在某一份回答之上**的。交付里自带一份
`当前原始回答快照.jsonl`，但此前**没有任何脚本证明**这份快照 == 我们当前
`runs\*_open_full.jsonl` 里的 900 条作答（450 条开卷）。于是出现这一类风险：

  · 回答被改过（例如重答 DeepSeek 的 NSCLC-027/032）、或题面/选项/标准答案/知识库变过，
    而二号表的判定仍是照着**旧回答**做的 → 拿旧判定配新回答，比对数字照样算得出来。

本脚本把两侧**逐条对照**，并把结论落成**机器可读**的 `outputs\二号材料对账.json`：

  ① 快照 ↔ 当前记录：逐条比较 题面 / 选项 / 标准答案 / 模型回答 / 检索上下文 / 回答指纹
     （任一项不同即列出该条）；
  ② 二号表 ↔ 当前记录：`回答指纹` 逐条相等（**强指纹**，回答一改即不等）、模型回答
     （二号表写表时**去掉了换行**，故按「去换行」归一化后再比）、标准答案、知识库指纹；
  ②b（收尾点 2，2026-10-12 修）**二号表自身的题面**：把二号表的
     **「问题」+「选项」两列拼出的完整题面**与 `qa/qa_set.jsonl` 的
     `common.full_question(item)` **逐条对照**（`标准答案` 列存在时同样纳入）——
     旧实现只比了回答侧与 `回答指纹`，**二号表里的题干/选项被改掉也照样通过**，
     于是可以拿"改了题面的二号判定"去配我们的回答。现在**任一不同 → 对账失败**
     （列出题号/模型与差异摘要）→ 该条**不得配对**，全量正式比较终止且**不覆盖产物**；
  ③ 只有**两侧都通过**的键才允许进入双裁判配对；不一致的条目**不得配对**。

退化策略（快照文件不存在时）
------------------------------------------------
若**本次被比较材料自己的目录**下没有快照文件（v2 交付就没有），则**退化为"按回答哈希配对"**：
只用二号表自身的 `回答指纹` / 模型回答哈希与当前记录比对，并在 stdout 与产物里
**明确打印所用策略**（`策略 = 按回答哈希配对（快照文件缺失）`），绝不静默假装做过快照对照。
（其它版本交付目录里存在快照时只作为信息列出，**不借用**——否则"这份材料自带快照"
与"借用别的版本快照"会被混为一谈。）

用法
------------------------------------------------
    python _selftest\reconcile_judge2_input.py                 # 对 v3 默认材料
    python _selftest\reconcile_judge2_input.py --csv <路径>     # 对指定二号表
    python _selftest\reconcile_judge2_input.py --records <jsonl>  # 用指定记录集（默认走 runs/）
产物
------------------------------------------------
    outputs\二号材料对账.json
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import dirpath, full_question, path, read_jsonl   # noqa: E402

SNAP_NAME = "当前原始回答快照.jsonl"
JUDGE2_CSV = "450条双项复评_人工复核表.csv"
# 二号裁判各版本交付目录（v3 = 当前正式口径）
DELIVERY = {
    "v3": ROOT / "outputs" / "二号裁判全量复评_2026-10-11_v3",
    "v2": ROOT / "outputs" / "二号裁判全量复评_2026-10-10_v2",
    "v1": ROOT / "outputs" / "二号裁判全量复评_2026-10-10",
}
# 本机外部交付快照（旧脚本曾硬编码；这里只作为**兜底候选**，找不到不影响流程）
EXTERNAL = ROOT / "outputs" / "external_judge2"

STRAT_SNAP = "快照逐条对照"
STRAT_HASH = "按回答哈希配对（快照文件缺失）"
STRAT_NONE = "无法对账（二号表既无『回答指纹』也无『模型回答』列）"

# 快照 ↔ 当前记录 的比较字段：(字段名, 中文名)
SNAP_FIELDS = (("question", "题面"), ("options", "选项"), ("reference", "标准答案"),
               ("response", "模型回答"), ("retrieved_contexts", "检索上下文"),
               ("fp", "回答指纹"))


def _load(name, filename):
    """按文件路径加载 src/ 下的模块（与 7c／compare_dual_full 同一写法）。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def canon(v) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def norm_text(s) -> str:
    """二号表写表时把回答里的换行**整体去掉**了（实测：450/450 在"去换行"下相等）。

    所以文本比较必须用这个归一化；但**判定配对是否安全要靠 `回答指纹`**（精确、独立），
    文本比较只作为可读的旁证，不单独放行任何条目。
    """
    return (s or "").replace("\r\n", "").replace("\n", "").replace("\r", "")


def answer_hash(rec) -> str:
    """按回答哈希配对用的内容指纹（题面/选项/标准答案/回答/检索上下文）。"""
    raw = canon({f: rec.get(f) for f, _ in SNAP_FIELDS})
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# 二号表里承载「题面」的两列（收尾点 2）
Q_COL, OPT_COL, REF_COL = "问题", "选项", "标准答案"


def _cut(s, n=60) -> str:
    """给差异摘要用的一小段文本（不含换行，便于一行展示）。"""
    t = "" if s is None else str(s)
    t = t.replace("\r", " ").replace("\n", "⏎")
    return t[:n] + ("…" if len(t) > n else "")


def full_question_from_csv(row, qa_item):
    """用二号表自己的「问题」+「选项」拼出完整题面，再与 `common.full_question(qa_set)` 对照。

    [!] 为什么必须拼出**完整题面**再比：二号表把题干与选项分成两列，
      只比其中一列都可能漏（改题干、改某个选项文字、增删选项都要拦住）。
      拼法与 `common.full_question()` **逐字节同源**（`A. …` 逐行、只用 ABCD、
      只有非空选项才出行），所以真实数据下 450/450 应当完全相等。

    返回 (是否一致, 差异原因列表)。
    """
    why = []
    if Q_COL not in row:
        return False, ["二号表缺「问题」列 → 无法核对题面"]
    if OPT_COL not in row:
        return False, ["二号表缺「选项」列 → 无法核对题面"]
    q_txt = row.get(Q_COL) or ""
    raw = (row.get(OPT_COL) or "").strip()
    opts = {}
    if raw:
        try:
            opts = json.loads(raw)
        except Exception as e:
            return False, [f"「选项」列不是合法 JSON（{type(e).__name__}）→ 无法核对题面"]
        if not isinstance(opts, dict):
            return False, [f"「选项」列不是 JSON 对象（{type(opts).__name__}）→ 无法核对题面"]
    lines = [f"{k}. {opts[k]}" for k in "ABCD" if opts.get(k)]
    got = q_txt + ("\n" + "\n".join(lines) if lines else "")
    want = full_question(qa_item)
    if got == want:
        return True, []
    # 不一致 → 逐段定位，给出可读摘要（题干 / 选项分别指出）
    if q_txt != (qa_item.get("question") or ""):
        why.append(f"题干不同（表={_cut(q_txt)} ≠ qa_set={_cut(qa_item.get('question'))}）")
    qa_opts = qa_item.get("options") or {}
    if opts != qa_opts:
        keys = sorted(set(opts) | set(qa_opts))
        det = [f"{k}:表={_cut(opts.get(k), 24)}≠qa={_cut(qa_opts.get(k), 24)}"
               for k in keys if opts.get(k) != qa_opts.get(k)]
        why.append("选项不同（" + "；".join(det[:4]) + ("…" if len(det) > 4 else "") + "）")
    if not why:   # 只可能是拼法差异（理论上不该发生）——给出完整题面摘要，绝不静默放过
        why.append(f"完整题面不同（表={_cut(got, 80)} ≠ qa_set={_cut(want, 80)}）")
    return False, why


def key_of(rec, model_col="model", qid_col="question_id"):
    return ((rec.get(model_col) or "").strip(), (rec.get(qid_col) or "").strip())


def read_csv_rows(p: Path):
    return list(csv.DictReader(Path(p).read_bytes().decode("utf-8-sig").splitlines()))


def find_snapshot(csv_path: Path, ver: str):
    """找**本次被比较材料自己的**快照。返回 (路径 or None, 已尝试候选, 其它版本的备选快照)。

    [!] 为什么只用"与被比较的二号表**同目录**"的那一份快照：二号材料对账要证明的是
        「**这份**二号判定所依据的回答 == 我们当前的回答」。拿**别的版本**交付目录里的
        快照来顶替，会让"这份材料自带快照"与"借用他人快照"混为一谈。
        因此同目录找不到快照时 → 明确退化为「按回答哈希配对」（策略名会打印出来），
        其它版本目录里存在快照只作为**信息**列出，不参与判定。
    """
    own = (Path(csv_path).parent / SNAP_NAME) if csv_path is not None else None
    tried = []
    if own is not None:
        tried.append(str(own))
        try:
            if own.is_file() and own.stat().st_size > 0:
                return own, tried, []
        except OSError:
            pass
    others = []
    for k, dd in sorted(DELIVERY.items()):
        p = dd / SNAP_NAME
        try:
            if p.is_file() and p.stat().st_size > 0 and (own is None or p.resolve() != own.resolve()):
                others.append(f"{k}:{p}")
        except OSError:
            continue
    p = EXTERNAL / SNAP_NAME
    try:
        if p.is_file() and p.stat().st_size > 0:
            others.append(f"外部:{p}")
    except OSError:
        pass
    return None, tried, others


def reconcile(csv_rows, cur_by, snap_rows=None, snap_path=None, cur_label="runs/*_open_full.jsonl",
              qa_by=None):
    """逐条对照二号材料与我们当前记录。**纯函数**（只读输入，不写任何文件）。

    csv_rows   ：二号表行（DictReader 出来的 dict 列表）
    cur_by     ：{(模型, 题号): 当前记录}
    snap_rows  ：快照行（None/[] = 快照缺失 → 退化策略）
    snap_path  ：快照路径（仅用于记录）
    qa_by      ：{题号: qa_set 条目}——用于把二号表的「问题」+「选项」（+「标准答案」）
                 与 `common.full_question()` 逐条对照（收尾点 2）。**None = 该检查不执行**
                 （会在产物里显式写明，绝不假装做过）。
    返回机器可读的对账结果 dict。
    """
    res = {
        "生成时间": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "策略": STRAT_SNAP if snap_rows else STRAT_HASH,
        "快照": {"文件": str(snap_path) if snap_path else None,
                 "条数": len(snap_rows or []),
                 "sha256": (hashlib.sha256(Path(snap_path).read_bytes()).hexdigest()
                            if snap_path and Path(snap_path).is_file() else None)},
        "当前记录": {"来源": cur_label, "条数": len(cur_by)},
        "逐条一致": 0, "不一致条数": 0, "不一致明细": [],
        "只在一侧": {"只在快照": [], "只在当前": []},
        "二号表": {"条数": len(csv_rows), "回答指纹一致": 0, "回答文本(去换行)一致": 0,
                   "回答文本(逐字节)一致": 0, "标准答案一致": 0, "知识库指纹一致": 0,
                   "题面(问题+选项)对账": ("已执行：问题+选项 拼出的完整题面 == qa_set 的 full_question()"
                                       if qa_by is not None else "未执行（未提供 qa_set）"),
                   "题面(问题+选项)一致": 0, "题干不同": 0, "选项不同": 0,
                   "未通过条数": 0, "未通过明细": []},
        "已验证键数": 0, "已排除键": [],
    }
    # ---------------- ① 快照 ↔ 当前记录 ----------------
    snap_ok = set()
    if snap_rows:
        snap_by, dup = {}, []
        for r in snap_rows:
            k = key_of(r)
            if k in snap_by:
                dup.append(k)
            snap_by[k] = r
        only_snap = sorted(set(snap_by) - set(cur_by))
        only_cur = sorted(set(cur_by) - set(snap_by))
        res["只在一侧"] = {"只在快照": [list(k) for k in only_snap],
                           "只在当前": [list(k) for k in only_cur]}
        if dup:
            res["快照重复键"] = [list(k) for k in dup]
        for k, sr in snap_by.items():
            cr = cur_by.get(k)
            if cr is None:
                continue
            diffs = []
            for f, cn in SNAP_FIELDS:
                a, b = sr.get(f), cr.get(f)
                same = (list(a) == list(b)) if isinstance(a, list) or isinstance(b, list) else (a == b)
                if not same:
                    diffs.append({"字段": cn, "快照": canon(a)[:120], "当前": canon(b)[:120]})
            if diffs:
                res["不一致明细"].append({"题号": k[1], "模型": k[0], "差异": diffs})
            else:
                snap_ok.add(k)
        res["逐条一致"] = len(snap_ok)
        res["不一致条数"] = len(res["不一致明细"])

    # ---------------- ② 二号表 ↔ 当前记录 ----------------
    cols = set(csv_rows[0].keys()) if csv_rows else set()
    has_fp = "回答指纹" in cols
    has_resp = "模型回答" in cols
    has_kb = "知识库指纹" in cols
    has_ref = REF_COL in cols
    has_q = Q_COL in cols and OPT_COL in cols
    csv_ok, csv_bad = set(), []
    if not has_fp and not has_resp:
        res["策略"] = STRAT_NONE
    for r in csv_rows:
        k = key_of(r, "模型", "题号")
        cr = cur_by.get(k)
        if cr is None:
            csv_bad.append({"题号": k[1], "模型": k[0], "原因": "当前记录里没有这条回答"})
            continue
        why = []
        if has_fp:
            if (r.get("回答指纹") or "").strip() == (cr.get("fp") or ""):
                res["二号表"]["回答指纹一致"] += 1
            else:
                why.append(f"回答指纹不同（表={r.get('回答指纹')!r} ≠ 当前={cr.get('fp')!r}）")
        if has_resp:
            a, b = r.get("模型回答") or "", cr.get("response") or ""
            if norm_text(a) == norm_text(b):
                res["二号表"]["回答文本(去换行)一致"] += 1
            else:
                why.append("回答文本不同（去换行归一化后仍不等）")
            if a == b:
                res["二号表"]["回答文本(逐字节)一致"] += 1
        # ---- ②b 二号表自身的**题面**（问题 + A~D 选项）↔ qa_set 的 full_question() ----
        # [!] 这是收尾点 2：只比回答侧/回答指纹时，**只改二号表里的题干或选项**照样能通过。
        if qa_by is not None:
            item = qa_by.get(k[1])
            if item is None:
                why.append(f"题号 {k[1]} 不在当前正式集 qa_set.jsonl 里 → 无法核对题面")
            elif has_q:
                same, diffs = full_question_from_csv(r, item)
                if same:
                    res["二号表"]["题面(问题+选项)一致"] += 1
                else:
                    why += diffs
                    if any(x.startswith("题干不同") for x in diffs):
                        res["二号表"]["题干不同"] += 1
                    if any(x.startswith("选项不同") for x in diffs):
                        res["二号表"]["选项不同"] += 1
            else:
                why.append("二号表缺「问题」或「选项」列 → 无法核对题面")
        # ---- 标准答案（表中有该列才比；有 qa_set 时以 qa_set 为准）----
        if has_ref:
            want_ref = (qa_by.get(k[1], {}).get("reference_answer")
                        if qa_by is not None else cr.get("reference")) or ""
            if (r.get(REF_COL) or "") == want_ref:
                res["二号表"]["标准答案一致"] += 1
            else:
                why.append(f"标准答案不同（表={_cut(r.get(REF_COL))} ≠ qa_set={_cut(want_ref)}）")
        if has_kb:
            if (r.get("知识库指纹") or "").strip() == (cr.get("kb_fp") or ""):
                res["二号表"]["知识库指纹一致"] += 1
            else:
                why.append(f"知识库指纹不同（表={r.get('知识库指纹')!r} ≠ 当前={cr.get('kb_fp')!r}）")
        if why:
            csv_bad.append({"题号": k[1], "模型": k[0], "原因": why})
        else:
            csv_ok.add(k)
    res["二号表"]["未通过条数"] = len(csv_bad)
    res["二号表"]["未通过明细"] = csv_bad[:50]

    # ---------------- ③ 只有两侧都通过才允许配对 ----------------
    ok2 = (snap_ok & csv_ok) if snap_rows else csv_ok
    res["已验证键数"] = len(ok2)
    # 「已排除」= 二号表里有、但两侧证据不足以证明"判定所依据的回答 == 当前回答"的条目
    res["已排除键"] = [[k[0], k[1]] for k in sorted({key_of(r, "模型", "题号") for r in csv_rows} - ok2)]
    res["配对口径"] = ("不一致的条目**不得配对**；全量正式比较遇任一条未通过 → 终止且不覆盖产物；"
                       "--allow-partial 时只从已验证键配对并写明实际分母/排除数")
    res["结论"] = (f"两侧回答逐条一致（快照 {res['逐条一致']}/{res['快照']['条数']}，"
                   f"二号表回答指纹一致 {res['二号表']['回答指纹一致']}/{res['二号表']['条数']}，"
                   f"二号表题面(问题+选项)一致 {res['二号表']['题面(问题+选项)一致']}/"
                   f"{res['二号表']['条数']}）"
                   f"→ 二号判定可与当前回答配对（已验证键 {len(ok2)}）"
                   if res["已验证键数"] == len(csv_rows) else
                   f"[!] 有 {len(csv_rows) - res['已验证键数']} 条无法证明「二号判定所依据的回答」"
                   f"就是当前回答（或题面被改过）→ 这些条目不得配对")
    return res


def current_records(records_path=None, label=None):
    """当前记录（默认走 7_report.load_all 的**完整版本校验**，返回 {(模型,题号): 记录}）。"""
    if records_path:
        rows = read_jsonl(records_path)
        lab = label or str(records_path)
    else:
        r7 = _load("recon_r7", "7_report.py")
        rows = r7.load_all("*_open_full.jsonl")
        lab = label or "runs/*_open_full.jsonl（经 7_report.load_all 版本校验）"
    return {key_of(r): r for r in rows}, lab


def main():
    ap = argparse.ArgumentParser(description="二号裁判材料对账（离线）")
    ap.add_argument("--csv", default=None, help=f"二号表路径（默认 v3 的 {JUDGE2_CSV}）")
    ap.add_argument("--ver", default="v3", choices=sorted(DELIVERY), help="二号裁判版本（决定交付目录）")
    ap.add_argument("--records", default=None, help="当前记录 jsonl（默认走 runs/ + 版本校验）")
    ap.add_argument("--out", default=None, help="产物路径（默认 outputs\\二号材料对账.json）")
    a = ap.parse_args()

    csv_path = Path(a.csv) if a.csv else DELIVERY[a.ver] / JUDGE2_CSV
    if not csv_path.is_file():
        sys.exit(f"[X] 找不到二号表：{csv_path}")
    csv_rows = read_csv_rows(csv_path)
    cur_by, lab = current_records(a.records)
    # qa_set 条目：用来把二号表的「问题」+「选项」与 full_question() 逐条对照（收尾点 2）
    qa_by = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    snap_path, tried, others = find_snapshot(csv_path, a.ver)
    snap_rows = read_jsonl(snap_path) if snap_path else []
    if snap_path is None:
        print(f"[!] 本次材料的同目录下**没有**二号材料快照（{SNAP_NAME}）"
              f"→ **退化为策略：{STRAT_HASH}**（只用二号表自带的『回答指纹』/模型回答哈希配对）。")
        print(f"    已尝试：{tried}")
        if others:
            print(f"    [i] 其它版本目录里存在快照（**不参与本次判定**，仅供核对）：{'；'.join(others)}")
    res = reconcile(csv_rows, cur_by, snap_rows, snap_path, lab, qa_by=qa_by)
    res["二号表路径"] = str(csv_path)
    res["快照候选"] = tried
    res["其它版本快照（未采用）"] = others

    print(f"二号表：{csv_path}（{len(csv_rows)} 行）")
    print(f"策略：**{res['策略']}**｜快照：{res['快照']['文件']}")
    print(f"① 快照 ↔ 当前记录：逐条一致 **{res['逐条一致']}/{res['快照']['条数']}**｜"
          f"不一致 **{res['不一致条数']}** 条")
    if res["不一致明细"]:
        for d in res["不一致明细"][:5]:
            print(f"   [!] {d['模型']}/{d['题号']}：" +
                  "；".join(x["字段"] for x in d["差异"]))
    print(f"   只在快照 {len(res['只在一侧']['只在快照'])} 条｜只在当前 {len(res['只在一侧']['只在当前'])} 条")
    t = res["二号表"]
    print(f"② 二号表 ↔ 当前记录：回答指纹一致 **{t['回答指纹一致']}/{t['条数']}**｜"
          f"回答文本(去换行)一致 {t['回答文本(去换行)一致']}/{t['条数']}｜"
          f"标准答案一致 {t['标准答案一致']}｜知识库指纹一致 {t['知识库指纹一致']}｜"
          f"未通过 **{t['未通过条数']}** 条")
    print(f"②b 二号表**题面**（问题+A~D选项 拼出的完整题面 ↔ qa_set 的 full_question()）："
          f"一致 **{t['题面(问题+选项)一致']}/{t['条数']}**"
          f"｜题干不同 {t['题干不同']} 条｜选项不同 {t['选项不同']} 条"
          f"｜核查方式：{t['题面(问题+选项)对账']}")
    for d in t["未通过明细"][:5]:
        print(f"   [!] {d['模型']}/{d['题号']}：{'；'.join(d['原因'])}")
    print(f"③ 已验证键 **{res['已验证键数']}/{len(csv_rows)}**｜已排除 {len(res['已排除键'])} 条")
    print(f"结论：{res['结论']}")

    out = Path(a.out) if a.out else dirpath("outputs") / "二号材料对账.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 机器可读对账结果 → {out}")
    return 0 if res["已验证键数"] == len(csv_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
