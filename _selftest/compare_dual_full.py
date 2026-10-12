# -*- coding: utf-8 -*-
r"""
全量双裁判比对（正确性 + 忠实性，各 450 条）。

数据源：
  裁判2（你方交付）：**默认 v3** → outputs\二号裁判全量复评_2026-10-11_v3\450条双项复评_人工复核表.csv
                     （v3 不存在时才回退到 v2：outputs\二号裁判全量复评_2026-10-10_v2\450条双项复评_人工复核表.csv）
  裁判1（本项目）：  outputs\auto_verdicts_deepseek-v4-pro.jsonl（正确性，题面**含 A~D 选项**）
                     outputs\faithfulness.jsonl（忠实性）

[!] 口径敏感性：v2 与 v3 是**两套不同尺度**，同一批回答会得到不同的条数，
    两个分母/分子**不可互相换算**。本脚本把「裁判2 版本 = v?（路径…）」同时写进
    stdout 与产物 `outputs\双裁判比对_全量.md`，避免把 v2 的数字当 v3 引用。

用法：python _selftest\compare_dual_full.py [裁判2的CSV路径]
      python _selftest\compare_dual_full.py --allow-partial [裁判2的CSV路径]
产物：outputs\双裁判比对_全量.md（机器可读对账结果另见 outputs\二号材料对账.json）

[!] 本文件是**判分输入闸门的唯一实现**（`_selftest\verify_dual_judge.py` 通过
    `analyze()` 复用同一套闸门 = 单一实现、两处调用）。四道闸门：
    ① **完整性/唯一性**（行数 == 唯一键数 == 期望条数）；② **fp 复核**（正确性 + 忠实性，
    各用与 7c / 5b 同一个 fp_of）；③ **二号材料对账**（快照 ↔ 当前回答、二号表题面/标准答案/
    回答指纹 ↔ 当前记录）；④ **标签合法性**（正确性 {一致,部分一致,不一致}／
    忠实性 {忠实,部分无依据,无依据}）。

[!] 全量正式比较（默认）遇到 **fp 缺失 / 对应回答缺失 / fp 不一致 / 二号表缺列 /
    二号材料对账不一致 / 模型不在 config** → **终止**，并且**不覆盖**已有产物。
    旧实现只 `fp_missing += 1; continue`，却仍从原始 j1c 与完整二号表组对 →
    打印"排除 1 条"却照样报 450 分母（复验第 2 项残留，已实测）。
    确实需要部分集比较时用 `--allow-partial`：**只从已验证键组成配对**，并在 stdout 与产物里
    写明「实际分母 N / 排除 M」。

[!] **标签合法性（第三方复验新确认的第 2 项）**：一号与二号的标签都必须落在各自的合法集合内。
    非法/缺失（含解析失败、空串）一律**显式排除**、不参与配对；因为 `kappa()` 只统计合法标签，
    旧实现让 `agree`（按全部配对算）与 `n`（按合法标签算）来自**两批不同的配对**，
    于是报告同时出现 450 与 449 两个互相矛盾的分母（已实测：把一条一号标签改成"解析失败"即复现）。
    现在 `block()` 只接受**已经过合法性筛选**的配对，并在内部再断言"筛掉 0 条"；
    分母一律写成**可核对等式**：`期望 T = 可比 P + 材料/版本校验排除 G + 非法/解析失败标签 I`，
    非法 > 0 时在产物**顶部**显著标注。非法标签**不终止**（仍出产物，但分母按有效标签算）。

[!] **完整性/唯一性（第三方复验新确认的第 1 项）**：判定缓存与二号表都必须
    `行数 == 唯一键数 == 期望条数`（期望条数 = 经 `7_report.load_all()` 版本校验后的有效回答数）。
    旧实现只按字典聚合键值，于是**缺一行**（450→449）、**重复一行**（451 行）、
    甚至**450 行但只有 449 个唯一记录（缺 1 + 重复 1）** 都能照常输出 450 分母的结论。
    现在三处（裁判1 正确性缓存、裁判1 忠实性缓存、二号表）**各自**断言，
    不符 → **非零退出、逐条列出缺失键/重复键/多余键、不覆盖产物**，
    并把「期望 N / 实际行 / 唯一键 / 缺失 / 重复」写进 stdout 与产物。
    该闸门与"二号表缺列"闸门同一纪律：**结构与唯一性不成立时没有"部分集"可谈**，
    因此 `--allow-partial` **也不放宽**。

[!] 版本校验贯穿**全部**判分输入：
    ① **忠实性侧**：`outputs\\faithfulness.jsonl` 的 `fp` 也必须用**与 `src\\5b_judge_faithfulness.py`
       同一个 `fp_of`** 逐条复核（此前只核对了正确性侧 j1c）。
    ② **二号裁判材料**：二号表是照着**某一份回答**判的。本脚本先跑
       `_selftest\\reconcile_judge2_input.py` 的对账（v3 交付自带的 `当前原始回答快照.jsonl`
       ↔ 我们当前 `runs/` 记录，逐条比 题面/选项/标准答案/回答/检索上下文/回答指纹；
       二号表自身的 `回答指纹` 也逐条比对；**二号表自己的「问题」+「选项」两列拼出的完整题面**
       也要与 `qa/qa_set.jsonl` 的 `common.full_question()` 逐条对照），
       **只有两侧证据都通过的键才允许配对**，不一致的条目**不得配对**（全量正式比较下即终止）。
       对账结论落盘 `outputs\\二号材料对账.json`。快照文件不存在时退化为「按回答哈希配对」，
       脚本会明确打印所用策略。
"""
import argparse
import collections
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import dirpath, read_jsonl   # noqa: E402


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_selftest(name, filename):
    """加载 `_selftest/` 下的模块（本文件自己就在 _selftest 里）。

    [!] 不能写成裸 `import xxx`：本文件在回归/验收脚本里是用 spec_from_file_location 加载的，
        解释器**不会**把 `_selftest/` 加进 sys.path，裸 import 会 ModuleNotFoundError。
    """
    p = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


r7 = _load("cmp_r7", "7_report.py")
c7 = _load("cmp_7c", "7c_export_full_review.py")
# [!] 忠实性指纹必须用**与 5b 同一个实现**（不再另写一份，避免两处漂移）
m5b = _load("cmp_5b", "5b_judge_faithfulness.py")
recon = _load_selftest("cmp_reconcile", "reconcile_judge2_input.py")

JUDGE2_CSV = "450条双项复评_人工复核表.csv"
V2 = ROOT / "outputs" / "二号裁判全量复评_2026-10-10_v2" / JUDGE2_CSV
V3 = ROOT / "outputs" / "二号裁判全量复评_2026-10-11_v3" / JUDGE2_CSV
# [!] 默认口径＝当前正式口径 v3。**只有** v3 文件不存在时才回退 v2（回退会在 stdout 与产物里告警）。
DEF = V3 if V3.exists() else V2
C = ["一致", "部分一致", "不一致"]
F = ["忠实", "部分无依据", "无依据"]

# 排除类别（分母等式里的两个"被减项"，必须可区分：修法完全不同）
EX_GATE = "材料/版本校验排除"
EX_ILLEGAL = "非法/解析失败标签"
OUT_NAME = "双裁判比对_全量.md"
J1C_NAME = "auto_verdicts_deepseek-v4-pro.jsonl"
J1F_NAME = "faithfulness.jsonl"


class GateFailure(Exception):
    """判分输入闸门未通过：调用方负责打印原因并非零退出（**绝不写比对产物**）。

    [!] `evidence` 用于携带**只读的对账中间结果**（如 `二号材料对账.json` 的内容）：
        比对产物 `outputs/双裁判比对_全量.md` 在闸门未通过时**必须保持原样**，
        但"二号材料对账"本身是**诊断证据**——旧实现就是在闸门之前落盘它的，
        验收脚本（`verify_judge2_reconcile.py` / `verify_judge2_question_block.py`）
        依赖"即使比较被拦下、对账 JSON 也反映了本次的实际不一致"。
        因此由 `main()` 在退出前落盘它；`verify_dual_judge.py` 不落盘（该入口全程只读）。
    """

    def __init__(self, title, problems, detail=(), exit_msg="[X] 拒绝输出可能与输入不一致的比对数字。",
                 evidence=None):
        self.title = title
        self.problems = list(problems)
        self.detail = list(detail)
        self.exit_msg = exit_msg
        self.evidence = dict(evidence or {})
        super().__init__(title + "｜" + "；".join(self.problems))

    def lines(self, indent="   "):
        out = []
        for d in self.detail:
            out.append(d)
        for p in self.problems:
            out.append(f"{indent}- {p}")
        return out


def judge2_ver(p: Path) -> str:
    """按**解析后的绝对路径**判定二号裁判版本，避免靠目录名猜（v2/v3 目录名相似）。"""
    rp = Path(p).resolve()
    if rp == V3.resolve():
        return "v3"
    if rp == V2.resolve():
        return "v2"
    return "未识别（自定义路径）"


# ---------------- 完整性 / 唯一性闸门 ----------------
def key_of_j1(r):
    """裁判1 判定缓存（正确性 / 忠实性）的键：**(模型显示名, 题号)**。"""
    return ((r.get("model") or "").strip(), (r.get("question_id") or "").strip())


def key_of_j2(r):
    """二号表的键：**(模型, 题号)**（列名是「题号」，历史文件可能写成「题目ID」）。"""
    return ((r.get("模型") or "").strip(), (r.get("题号") or r.get("题目ID") or "").strip())


def _key_text(keys, limit=50):
    shown = "、".join(f"{m}/{q}" for m, q in keys[:limit])
    if len(keys) > limit:
        shown += f"……（共 {len(keys)} 个，此处只列前 {limit} 个）"
    return shown


def key_stat(rows, key_fn, expected, label, keyset_name):
    """统计"行数 / 唯一键 / 缺失 / 重复 / 多余"，并判定 `行数 == 唯一键 == 期望条数`。

    [!] 为什么必须**逐键**比而不是只看条数：实测"450 行但只有 449 个唯一记录"（缺 1 + 重复 1）
        用条数判据完全看不出来，而旧实现连条数都不比 → 缺行、重复行、缺+重复都静默通过。
    """
    seen = collections.Counter(key_fn(r) for r in rows)
    uniq = set(seen)
    missing = sorted(expected - uniq)
    extra = sorted(uniq - expected)
    dups = sorted(k for k, c in seen.items() if c > 1)
    passed = (len(rows) == len(uniq) == len(expected)) and not missing and not extra and not dups
    return {
        "来源": label, "键": keyset_name,
        "期望": len(expected), "实际行": len(rows), "唯一键": len(uniq),
        "缺失": len(missing), "重复": len(dups), "多余": len(extra),
        "缺失键": [list(k) for k in missing], "重复键": [list(k) for k in dups],
        "多余键": [list(k) for k in extra],
        "通过": passed,
        "摘要": (f"{label}：期望 N={len(expected)} / 实际行={len(rows)} / 唯一键={len(uniq)}"
                 f" / 缺失={len(missing)} / 重复={len(dups)} / 多余={len(extra)}"),
    }


def completeness_gate(stats):
    """三条完整性统计 → GateFailure（不通过时**逐条列出**缺失键/重复键/多余键）。"""
    problems, detail = [], []
    for s in stats:
        if s["通过"]:
            continue
        reasons = []
        if s["实际行"] != s["期望"]:
            reasons.append(f"实际行 {s['实际行']} ≠ 期望 {s['期望']}")
        if s["唯一键"] != s["期望"]:
            reasons.append(f"唯一键 {s['唯一键']} ≠ 期望 {s['期望']}")
        if s["缺失"]:
            reasons.append(f"缺失 {s['缺失']} 个键")
        if s["重复"]:
            reasons.append(f"重复 {s['重复']} 个键（同一键出现多次）")
        if s["多余"]:
            reasons.append(f"多余 {s['多余']} 个键（不在当前有效回答集内）")
        problems.append(f"{s['来源']}：{s['键']} 行数/唯一键/期望条数不一致——" + "；".join(reasons))
        detail.append(f"   [!] {s['摘要']}")
        if s["缺失键"]:
            detail.append(f"       缺失键：{_key_text([tuple(k) for k in s['缺失键']])}")
        if s["重复键"]:
            detail.append(f"       重复键：{_key_text([tuple(k) for k in s['重复键']])}")
        if s["多余键"]:
            detail.append(f"       多余键（当前 `runs/` 里没有对应的有效回答：可能是回答过期/"
                          f"被 `7_report.load_all()` 版本校验排除，或模型不在当前 config）："
                          f"{_key_text([tuple(k) for k in s['多余键']])}")
    if not problems:
        return None
    return GateFailure(
        f"输入完整性/唯一性闸门未通过 → 终止，**不覆盖**已有产物"
        f"（{dirpath('outputs') / OUT_NAME} 保持原样）：",
        problems, detail,
        "[X] 拒绝输出可能与输入不一致的比对数字（判定记录存在缺行/重复："
        "三处的「行数 == 唯一键数 == 期望条数」必须逐一成立）。")


def kappa(pairs, labels):
    pairs = [(a, b) for a, b in pairs if a in labels and b in labels]
    n = len(pairs)
    if not n:
        return None, None, None, 0
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0, po, pe, n


def boot(pairs, labels, boot=2000):
    rnd = random.Random(13)
    v = [(a, b) for a, b in pairs if a in labels and b in labels]
    ks, ps = [], []
    for _ in range(boot):
        s = [v[rnd.randrange(len(v))] for _ in range(len(v))]
        k, po, _, _ = kappa(s, labels)
        if k is not None:
            ks.append(k)
            ps.append(po)
    ks.sort()
    ps.sort()
    return (ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks)) - 1]), \
           (ps[int(0.025 * len(ps))], ps[int(0.975 * len(ps)) - 1])


def block(name, pairs, labels):
    """一个维度的统计块（分子与分母必须来自**同一批**合法配对）。

    [!] 内部再筛一次并用断言证明"筛掉 0 条"：`kappa()` 只统计合法标签，而旧实现的
        `agree` 是按**全部**配对算的 → 把一条一号标签改成"解析失败"后，
        报告会同时出现 450（块标题）与 449（可比条数）两个互相矛盾的分母（已实测复现）。
        现在配对在进入本函数前就已按合法性筛过，这里的断言是"防止将来又有人绕过去"的兜底。
    """
    legal = [(a, b) for a, b in pairs if a in labels and b in labels]
    if len(legal) != len(pairs):
        raise GateFailure(
            "配对里出现非法标签 → 终止，**不覆盖**已有产物：",
            [f"{name}：有 {len(pairs) - len(legal)} 条配对的标签不在合法集合 {labels} 内"
             f"（分子分母将来自两批不同的配对）"],
            [], "[X] 内部自检失败：拒绝输出自相矛盾的分母。")
    k, po, pe, n = kappa(legal, labels)
    agree = sum(1 for a, b in legal if a == b)
    L = [f"## {name}", "",
         f"- 可比条数 **{n}**；完全一致 **{agree}/{n} = {agree / n * 100:.1f}%**"
         f"（bootstrap 95% 区间 {boot(legal, labels)[1][0] * 100:.1f}%~{boot(legal, labels)[1][1] * 100:.1f}%）",
         f"- Cohen's κ = **{k:.3f}**（95% 区间 {boot(legal, labels)[0][0]:.3f}~{boot(legal, labels)[0][1]:.3f}）"
         f"；po={po:.3f}，pe={pe:.3f}",
         f"- 边缘分布：裁判1 {dict(collections.Counter(a for a, _ in legal if a in labels))}"
         f"｜裁判2 {dict(collections.Counter(b for _, b in legal if b in labels))}",
         "", "| 裁判1＼裁判2 | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    ct = collections.defaultdict(collections.Counter)
    for a, b in legal:
        ct[a][b] += 1
    for a in labels:
        L.append(f"| **{a}** | " + " | ".join(str(ct[a].get(b, 0)) for b in labels) + " |")
    return L, ct, n, agree


def _why_excluded(key, ok2, verified, verdict, dim) -> str:
    """给出"这条为什么进不了配对"的**人话原因**（对账失败 / 左侧 fp 未过 / 标签非法）。

    [!] 必须把「二号材料对账未通过」与「左侧判定 fp 未过」分开报：
        两者的修法完全不同（前者要重跑/重取其判定，后者要重判一号），
        混成一句"未通过校验"会让人修错地方。
    """
    if key not in ok2:
        return "二号材料对账未通过（无法证明二号判定所依据的回答 == 当前回答）"
    if key not in verified:
        return f"左侧 {dim} 判定 fp 未通过校验（缺回答/缺 fp/与当前回答不一致）"
    return f"二号表{dim}列不是合法标签（{verdict!r}）"


def rows_for_known_models(rows, known_models, label):
    """只保留「模型名出现在当前 config」的二号表行；被丢掉的行**计数并打印**（绝不静默丢）。

    [!] 为什么必须显式丢掉：二号表里可能混有非当前 config 模型的整批行。这种行在左侧
        （裁判1 的 runs/缓存）**根本没有对手方**——旧实现却照样把它们算进分母，
        于是出现"排除 1 条、分母仍是 450"这一类自相矛盾的结果。
        [!] 2026-10-12（第三方复验第 1 项）：这种行现在会先被**完整性/唯一性闸门**当成
        「多余键」拦下（行数/键集合与期望条数不一致），修法与原因都会逐条列出来。
    """
    keep = [r for r in rows if r.get("模型") in known_models]
    dropped = len(rows) - len(keep)
    if dropped:
        names = sorted({(r.get("模型") or "").strip() for r in rows} - set(known_models))
        print(f"[!] 二号表（{label}）中有 {dropped} 行的模型不在当前 config（{names}）→ "
              f"不纳入本次比对（左侧没有对应回答）。")
    return keep, dropped


def _denom_eq(name, total, n_pairs, n_gate, n_ill, n_ill1, n_ill2):
    """分母等式（**可核对**）：期望 = 可比 + 材料/版本校验排除 + 非法/解析失败标签。"""
    return (f"{name} 期望 {total} = 可比 {n_pairs} + 材料/版本校验排除 {n_gate}"
            f" + 非法/解析失败标签 {n_ill}；非法明细：一号 {n_ill1} 条 / 二号 {n_ill2} 条")


def analyze(src_path, allow_partial: bool = False) -> dict:
    """读取全部判分输入 → 跑四道闸门 → 返回配对与统计（**不写任何产物**）。

    [!] 产物（`outputs/双裁判比对_全量.md` 与 `outputs/二号材料对账.json`）一律由
        `main()` 写：`_selftest/verify_dual_judge.py` 也调用本函数，必须做到"只读复核"。
    [!] 硬闸门（完整性/唯一性；全量正式模式下的 fp/材料问题）通过 `GateFailure` 抛出，
        调用方负责打印原因并非零退出 —— 单一实现、两处调用。
    """
    SRC = Path(src_path)
    if not SRC.exists():
        raise GateFailure("找不到二号裁判表：", [f"路径不存在：{SRC}"], [],
                          f"[X] 找不到 {SRC}")
    VER = judge2_ver(SRC)
    if VER != "v3":
        print(f"[!] 警告：本次**不是**当前正式口径 v3，而是 {VER}——"
              f"数字与 v3 尺度不可互相换算。")
    if SRC.resolve() == V2.resolve() and V3.exists():
        print(f"[!] 警告：默认应为 v3（{V3}），本次却用了 v2——请确认是否为有意指定。")
    rows = list(csv.DictReader(SRC.read_bytes().decode("utf-8-sig").splitlines()))
    j1c_path = dirpath("outputs") / J1C_NAME
    j1f_path = dirpath("outputs") / J1F_NAME
    j1c_rows = read_jsonl(j1c_path)
    j1c = {key_of_j1(r): (r.get("judge") or "") for r in j1c_rows}
    j1f_rows = read_jsonl(j1f_path)
    j1f = {key_of_j1(r): (r.get("verdict") or "") for r in j1f_rows}
    print(f"裁判2 CSV {len(rows)} 行；裁判1 正确性 {len(j1c)} 条、忠实性 {len(j1f)} 条")

    # —— 裁判缓存 fp 与被比较回答的一致性校验 ——
    # [!] 只核对"裁判名"不够：缓存行里有 fp，但它必须与**本次被比较的那条回答**复算出的 fp
    #     相等，否则这份比对可能拿旧回答的判定去对当前回答（改题面/选项/知识库/回答后尤甚）。
    #     这里用与 7c 完全相同的 fp_of 复算（同一实现，不再另写一份）。
    qa_now = {x["question_id"]: x for x in read_jsonl(r7.path("qa_set"))}
    _by_model_qid = {(r["model"], r["question_id"]): r
                     for r in r7.load_all("*_open_full.jsonl")}
    CFG_JUDGE = r7.CFG["judge"]
    # [!] 二号表列名同样要过闸门：缺列会让整列变成空串 → 静默少算。缺列直接终止（不再静默）。
    need_cols = ("题号", "模型", "正确性_二号裁判", "忠实性_二号裁判")
    miss_cols = [c for c in need_cols if not rows or c not in rows[0]]
    if miss_cols:
        raise GateFailure("二号表缺少必需列 → 终止（原产物保留）：",
                          [f"缺少必需列 {miss_cols}；实际列："
                           f"{list(rows[0].keys()) if rows else '（空表）'}"], [],
                          "[X] 拒绝在缺列（整列会变成空串）的情况下比对。")

    # ---- 闸门①：完整性 / 唯一性（三处各自断言；与"缺列"同一纪律，--allow-partial 也不放宽）----
    expected = set(_by_model_qid)
    comp = [
        key_stat(j1c_rows, key_of_j1, expected,
                 f"裁判1 正确性缓存（outputs/{J1C_NAME}）", "(模型, 题号)"),
        key_stat(j1f_rows, key_of_j1, expected,
                 f"裁判1 忠实性缓存（outputs/{J1F_NAME}）", "(模型, 题号)"),
        key_stat(rows, key_of_j2, expected,
                 f"二号表（{SRC.parent.name}/{SRC.name}）", "(模型, 题号)"),
    ]
    print("输入完整性/唯一性闸门（行数 == 唯一键数 == 期望条数；期望条数 = "
          f"经 7_report.load_all() 版本校验后的有效回答数 {len(expected)}）：")
    for s in comp:
        print(f"  - {s['摘要']}")
    gate = completeness_gate(comp)
    if gate:
        raise gate

    fp_ok = fp_missing = fp_mismatch = 0
    fp_bad, fp_miss = [], []
    verified = set()          # 左侧通过 fp 校验的键（只有这些键允许进入配对）
    for r in j1c_rows:
        qid = r.get("question_id")
        key = key_of_j1(r)
        item = qa_now.get(qid)
        rec = _by_model_qid.get(key)
        if item is None or rec is None or not r.get("fp"):
            fp_missing += 1
            fp_miss.append((key, "题目不在正式集内" if item is None else
                            ("当前回答不在 runs/（或未通过版本校验）" if rec is None else "缓存行缺 fp")))
            continue
        try:
            want = c7.fp_of(item, rec, CFG_JUDGE)
        except Exception as e:
            fp_missing += 1
            fp_miss.append((key, f"复算 fp 失败：{type(e).__name__}"))
            continue
        if want == r.get("fp"):
            fp_ok += 1
            verified.add(key)
        else:
            fp_mismatch += 1
            fp_bad.append((qid, key[0], r.get("fp"), want))
    print(f"裁判1 缓存 fp 复核：一致 {fp_ok}｜不一致 {fp_mismatch}｜缺回答/缺 fp {fp_missing}")

    # —— 忠实性侧 fp 复核（**与 5b 同一个 fp_of**）——
    # [!] 原缺口：正确性侧核了 fp，忠实性侧直接读 faithfulness.jsonl 的 verdict 就用。
    #     回答/片段/裁判任一变化都会让那份判定失效，却会被拿来配当前回答 → 静默错配。
    #     策略与裁判1 侧完全一致：缺 fp / 不符 → 全量正式比较终止、不覆盖产物；
    #     --allow-partial 时只从**已验证键**配对，并写明实际分母与排除数。
    #     [!] 标签合法性单独在组对阶段判定：`faith_verified` 只表示"fp 与当前回答一致"，
    #         不再顺带要求标签合法 —— 否则非法标签会被误报成"左侧 fp 未过"（原因错误）。
    f_ok = f_missing = f_mismatch = 0
    f_bad, f_miss = [], []
    faith_verified = set()
    for r in j1f_rows:
        qid = r.get("question_id")
        key = key_of_j1(r)
        item = qa_now.get(qid)
        rec = _by_model_qid.get(key)
        if item is None or rec is None or not r.get("fp"):
            f_missing += 1
            f_miss.append((key, "题目不在正式集内" if item is None else
                           ("当前回答不在 runs/（或未通过版本校验）" if rec is None else "判定行缺 fp")))
            continue
        try:
            want = m5b.fp_of(rec, CFG_JUDGE["name"])
        except Exception as e:
            f_missing += 1
            f_miss.append((key, f"复算 fp 失败：{type(e).__name__}"))
            continue
        if want == r.get("fp"):
            f_ok += 1
            faith_verified.add(key)
        else:
            f_mismatch += 1
            f_bad.append((qid, key[0], r.get("fp"), want))
    print(f"裁判1 忠实性 fp 复核（与 5b 同一 fp_of）：一致 {f_ok}｜不一致 {f_mismatch}"
          f"｜缺回答/缺 fp {f_missing}")

    rows_known, rows_dropped = rows_for_known_models(
        rows, {m["name"] for m in r7.CFG["models"]} | {m["display"] for m in r7.CFG["models"]},
        VER)

    # —— 二号裁判材料对账（快照 ↔ 当前记录；二号表回答指纹 ↔ 当前记录）——
    # [!] 二号表是照着**某一份回答**判的：只有能证明"那份回答 == 当前回答"的键才允许配对。
    #     快照缺失时退化为"按回答哈希配对"，策略名会打印出来（绝不静默假装核对过）。
    recon_path, recon_tried, recon_others = recon.find_snapshot(SRC, VER)
    recon_rows = read_jsonl(recon_path) if recon_path else []
    if recon_path is None:
        print(f"[!] {SRC.parent.name} 目录下没有二号材料快照（{recon.SNAP_NAME}）→ "
              f"**退化为策略：{recon.STRAT_HASH}**")
        print(f"    已尝试：{recon_tried}")
        if recon_others:
            print(f"    [i] 其它版本目录里存在快照（**不参与本次判定**）：{'；'.join(recon_others)}")
    REC = recon.reconcile(rows_known, _by_model_qid, recon_rows, recon_path,
                          "runs/*_open_full.jsonl（经 7_report.load_all 版本校验）",
                          qa_by=qa_now)
    REC["二号表路径"] = str(SRC)
    REC["快照候选"] = recon_tried
    REC["其它版本快照（未采用）"] = recon_others
    ok2 = {key_of_j2(r) for r in rows_known} - {tuple(k) for k in REC["已排除键"]}
    print(f"二号材料对账：策略 **{REC['策略']}**｜快照逐条一致 {REC['逐条一致']}/{REC['快照']['条数']}"
          f"｜不一致 {REC['不一致条数']}｜二号表回答指纹一致 "
          f"{REC['二号表']['回答指纹一致']}/{REC['二号表']['条数']}"
          f"｜二号表题面(问题+选项)一致 {REC['二号表']['题面(问题+选项)一致']}/{REC['二号表']['条数']}"
          f"（题干不同 {REC['二号表']['题干不同']}、选项不同 {REC['二号表']['选项不同']}）"
          f"｜已验证键 {len(ok2)}｜排除 {len(REC['已排除键'])}")

    # —— 组对：**只从已验证键组成配对**，并按"标签是否合法"分两类排除 ——
    # [!] 非法/缺失标签（解析失败等）**显式排除**、不参与配对；它们与"材料/fp 排除"分开计数，
    #     这样分母等式才有意义：期望 = 可比 + 材料/版本校验排除 + 非法/解析失败标签。
    pairs_c, excluded_c = [], []
    pairs_f, excluded_f = [], []
    for r in rows_known:
        key = key_of_j2(r)
        v1c = (j1c.get(key) or "").strip()
        v2c = (r.get("正确性_二号裁判") or "").strip()
        v1f = (j1f.get(key) or "").strip()
        v2f = (r.get("忠实性_二号裁判") or "").strip()
        if key in ok2 and key in verified:
            if v1c in C and v2c in C:
                pairs_c.append((v1c, v2c))
            else:
                bad = []
                if v1c not in C:
                    bad.append(f"一号正确性标签={v1c!r}")
                if v2c not in C:
                    bad.append(f"二号正确性标签={v2c!r}")
                excluded_c.append({"键": key, "类别": EX_ILLEGAL,
                                   "一号非法": v1c not in C, "二号非法": v2c not in C,
                                   "理由": "、".join(bad)})
        else:
            excluded_c.append({"键": key, "类别": EX_GATE,
                               "一号非法": False, "二号非法": False,
                               "理由": _why_excluded(key, ok2, verified, v2c, "正确性")})
        if key in ok2 and key in faith_verified:
            if v1f in F and v2f in F:
                pairs_f.append((v1f, v2f))
            else:
                bad = []
                if v1f not in F:
                    bad.append(f"一号忠实性标签={v1f!r}")
                if v2f not in F:
                    bad.append(f"二号忠实性标签={v2f!r}")
                excluded_f.append({"键": key, "类别": EX_ILLEGAL,
                                   "一号非法": v1f not in F, "二号非法": v2f not in F,
                                   "理由": "、".join(bad)})
        else:
            excluded_f.append({"键": key, "类别": EX_GATE,
                               "一号非法": False, "二号非法": False,
                               "理由": _why_excluded(key, ok2, faith_verified, v2f, "忠实性")})

    def _dn(name, pairs, excl):
        g = [e for e in excl if e["类别"] == EX_GATE]
        ill = [e for e in excl if e["类别"] == EX_ILLEGAL]
        d = {"期望": len(expected), "可比": len(pairs), "gate排除": len(g), "非法": len(ill),
             "非法一号": sum(1 for e in ill if e["一号非法"]),
             "非法二号": sum(1 for e in ill if e["二号非法"]),
             "排除合计": len(excl)}
        d["等式"] = _denom_eq(name, d["期望"], d["可比"], d["gate排除"], d["非法"],
                            d["非法一号"], d["非法二号"])
        # 等式自检：不自洽 → 终止（绝不输出自相矛盾的分母）
        if d["可比"] + d["gate排除"] + d["非法"] != d["期望"]:
            raise GateFailure(
                f"分母等式**不自洽**（{name}）→ 终止，**不覆盖**已有产物：",
                [f"{d['等式']}（左 {d['期望']} ≠ 右 "
                 f"{d['可比'] + d['gate排除'] + d['非法']}）"], [],
                "[X] 内部自检失败：拒绝输出自相矛盾的分母。")
        return d

    dc = _dn("正确性", pairs_c, excluded_c)
    df = _dn("忠实性", pairs_f, excluded_f)

    if not allow_partial:
        # ---- 全量正式比较：任何缺失/不一致都**终止**，且**不覆盖**已有产物 ----
        # [!] 非法/缺失标签**不进这里**：它们已被显式排除，分母按有效标签算（产物顶部标注）。
        problems, detail = [], []
        if fp_mismatch:
            for b in fp_bad[:5]:
                detail.append(f"   [!] {b[0]} {b[1]}：缓存 fp={b[2]} ≠ 当前回答复算 fp={b[3]}")
            problems.append(f"裁判1 正确性缓存有 {fp_mismatch} 条 fp 与当前回答不一致")
        if fp_missing:
            for k, why in fp_miss[:5]:
                detail.append(f"   [!] {k[0]}/{k[1]}：{why}")
            problems.append(f"有 {fp_missing} 条判定缓存的对应回答缺失或缺 fp（例：{fp_miss[0][1]}）")
        # —— 忠实性侧 fp（与 5b 同一 fp_of）：缺 / 不符同样终止 ——
        if f_mismatch:
            for b in f_bad[:5]:
                detail.append(f"   [!] 忠实性 {b[0]} {b[1]}：缓存 fp={b[2]} ≠ 当前回答复算 fp={b[3]}")
            problems.append(f"裁判1 **忠实性**缓存有 {f_mismatch} 条 fp 与当前回答不一致"
                            f"（会拿旧忠实性判定配新回答）")
        if f_missing:
            for k, why in f_miss[:5]:
                detail.append(f"   [!] 忠实性 {k[0]}/{k[1]}：{why}")
            problems.append(f"忠实性有 {f_missing} 条判定缺 fp 或对应回答缺失（例：{f_miss[0][1]}）")
        # —— 二号裁判材料对账：不一致的条目不得配对 ——
        if REC["不一致条数"]:
            for d in REC["不一致明细"][:5]:
                detail.append(f"   [!] 二号材料快照 {d['模型']}/{d['题号']} 与当前记录不同："
                              + "；".join(x["字段"] for x in d["差异"]))
            problems.append(f"二号材料快照与当前回答有 {REC['不一致条数']} 条不同"
                            f"（题面/选项/标准答案/回答/检索上下文/回答指纹任一不同）")
        if REC["二号表"]["未通过条数"]:
            for d in REC["二号表"]["未通过明细"][:5]:
                detail.append(f"   [!] 二号表 {d['模型']}/{d['题号']}：{'；'.join(d['原因'])}")
            problems.append(f"二号表有 {REC['二号表']['未通过条数']} 条的题面（问题+选项）"
                            f"/标准答案 或 回答指纹/回答文本与当前记录不一致"
                            f"（无法证明判定所依据的材料 == 当前材料）")
        if REC["只在一侧"]["只在快照"] or REC["只在一侧"]["只在当前"]:
            problems.append(f"二号材料快照与当前记录键集合不同"
                            f"（只在快照 {len(REC['只在一侧']['只在快照'])}、"
                            f"只在当前 {len(REC['只在一侧']['只在当前'])}）")
        if REC["策略"] == recon.STRAT_NONE:
            problems.append("二号表既无「回答指纹」也无「模型回答」列 → 无法对账，拒绝比对")
        if rows_dropped:
            problems.append(f"二号表有 {rows_dropped} 行的模型不在当前 config（左侧没有对应回答）")
        # [!] 只有「材料/版本校验排除」才算问题；非法/缺失标签已显式排除并计入分母等式。
        gate_excl_c = [e for e in excluded_c if e["类别"] == EX_GATE]
        gate_excl_f = [e for e in excluded_f if e["类别"] == EX_GATE]
        if gate_excl_c or gate_excl_f:
            for e in (gate_excl_c + gate_excl_f)[:5]:
                detail.append(f"   [!] {e['键'][0]}/{e['键'][1]}：{e['理由']}")
            problems.append(f"正确性有 {len(gate_excl_c)} 条、忠实性有 {len(gate_excl_f)} 条无法组成"
                            f"已验证配对")
        if problems:
            raise GateFailure(
                f"全量正式比较未通过输入版本闸门 → 终止，**不覆盖**已有产物"
                f"（{dirpath('outputs') / OUT_NAME} 保持原样）：",
                problems, detail,
                "[X] 拒绝输出可能与输入不一致的比对数字。"
                "若确实要做部分集比较，请显式加 --allow-partial（产物会写明实际分母与排除数）。",
                evidence={"REC": REC})

    if dc["非法"] or df["非法"]:
        print(f"[!] 存在非法/解析失败标签：正确性 {dc['非法']} 条（一号 {dc['非法一号']} / "
              f"二号 {dc['非法二号']}）、忠实性 {df['非法']} 条（一号 {df['非法一号']} / "
              f"二号 {df['非法二号']}）→ **显式排除、不参与配对**；产物顶部已标注，"
              f"分母按有效标签算。")
        print(f"    分母等式：{dc['等式']}")
        print(f"    分母等式：{df['等式']}")
    print(f"配对：正确性 {len(pairs_c)} 条（排除 {dc['排除合计']}）"
          f"｜忠实性 {len(pairs_f)} 条（排除 {df['排除合计']}）"
          f"｜二号表被丢弃行 {rows_dropped}"
          f"｜模式={'部分集(--allow-partial)' if allow_partial else '全量正式'}")
    print(f"分母等式：{dc['等式']}")
    print(f"分母等式：{df['等式']}")

    return {
        "SRC": SRC, "VER": VER, "rows": rows, "rows_known": rows_known,
        "rows_dropped": rows_dropped, "j1c": j1c, "j1f": j1f, "qa_now": qa_now,
        "by_model_qid": _by_model_qid, "completeness": comp, "expected": expected,
        "ok2": ok2, "REC": REC,
        "fp": {"ok": fp_ok, "mismatch": fp_mismatch, "missing": fp_missing,
               "f_ok": f_ok, "f_mismatch": f_mismatch, "f_missing": f_missing},
        "verified": verified, "faith_verified": faith_verified,
        "pairs_c": pairs_c, "pairs_f": pairs_f,
        "excluded_c": excluded_c, "excluded_f": excluded_f,
        "denom_c": dc, "denom_f": df, "allow_partial": allow_partial,
    }


def render(R) -> str:
    """把分析结果渲染成 `outputs/双裁判比对_全量.md` 的正文。"""
    SRC, VER, REC = R["SRC"], R["VER"], R["REC"]
    dc, df = R["denom_c"], R["denom_f"]
    fp = R["fp"]
    nc, nf = dc["可比"], df["可比"]
    partial = R["allow_partial"]
    L = ["# 全量双裁判比对（正确性与忠实性分开）", ""]
    # —— 非法标签：**产物顶部**显著标注（第三方复验第 2 项）——
    if dc["非法"] or df["非法"]:
        L += [f"- [!] **警告：本次存在非法/解析失败标签 → 已从配对中显式排除（既不进分子也不进分母）**："
              f"正确性 {dc['非法']} 条（一号 {dc['非法一号']} / 二号 {dc['非法二号']}）、"
              f"忠实性 {df['非法']} 条（一号 {df['非法一号']} / 二号 {df['非法二号']}）。"
              f"**本文件的分母必须按下述等式读**（不存在第二个分母）："
              f"{dc['等式']}；{df['等式']}。"]
    if VER != "v3":
        L += [f"- [!] **告警：本次用的是 {VER}，不是当前正式口径 v3**——"
              f"引用前请先统一到 v3（`二号裁判全量复评_2026-10-11_v3`）。"]
    L += [f"- **裁判2 版本 = {VER}**（路径 `{SRC}`）｜裁判1 正确性题面**含 A~D 选项**；"
          f"口径＝一号(含选项) vs 二号 {VER}",
          f"- 裁判1：`deepseek-v4-pro`（本项目脚本）｜裁判2：你方交付（{SRC.parent.name}）",
          "- [!] 这是**一致性**，不是医学准确性；不能证明任一裁判正确。",
          f"- [!] v2 与 v3 是**两套不同尺度**（同一批回答条数不同），**不可互相换算**；"
          f"本文件全部数字均为 **{VER}** 口径。",
          # —— 完整性/唯一性（第三方复验第 1 项）——
          "- [!] 输入**完整性/唯一性**闸门（**行数 == 唯一键数 == 期望条数**；期望条数 = 经 "
          "`7_report.load_all()` 版本校验后的有效回答数）："
          + "；".join(s["摘要"].replace("：", " ") for s in R["completeness"])
          + "。**任一不符 → 终止且不覆盖本文件**；缺行/重复行/“450 行但 449 个唯一记录”都会被逐条"
            "列出（缺失键 / 重复键 / 多余键）。",
          f"- [!] 输入版本闸门·**正确性侧**：裁判1 缓存 fp 与**当前被比较回答**复算的 fp 逐条核对 → "
          f"一致 **{fp['ok']}** / 不一致 **{fp['mismatch']}** / 缺回答或缺 fp **{fp['missing']}**",
          f"- [!] 输入版本闸门·**忠实性 fp 复核**（与 `src/5b_judge_faithfulness.py` 同一个 `fp_of`）："
          f"`outputs/faithfulness.jsonl` 的 fp 逐条复核 → 一致 **{fp['f_ok']}** / 不一致 "
          f"**{fp['f_mismatch']}** / 缺回答或缺 fp **{fp['f_missing']}**；缺 fp / 不符的条目"
          f"**不进入配对**（全量正式比较下即终止、不覆盖本文件）。",
          f"- [!] **二号裁判材料对账**（`_selftest/reconcile_judge2_input.py`）：策略＝**{REC['策略']}**"
          f"｜二号材料快照 ↔ 当前记录（题面/选项/标准答案/模型回答/检索上下文/回答指纹）"
          f"逐条一致 **{REC['逐条一致']}/{REC['快照']['条数']}**、不一致 **{REC['不一致条数']}**"
          f"｜二号表 `回答指纹` ↔ 当前记录一致 **{REC['二号表']['回答指纹一致']}/{REC['二号表']['条数']}**"
          f"、未通过 **{REC['二号表']['未通过条数']}**"
          f"（二号表写表时去掉了回答里的换行 → 文本比对按「去换行」归一化，"
          f"配对放行以**精确的回答指纹**为准）"
          f"｜二号表**题面**（「问题」+「选项」两列拼出的完整题面、含「标准答案」）"
          f"↔ `qa/qa_set.jsonl` 的 `common.full_question()` 逐条一致 "
          f"**{REC['二号表']['题面(问题+选项)一致']}/{REC['二号表']['条数']}**"
          f"（题干不同 **{REC['二号表']['题干不同']}**、选项不同 **{REC['二号表']['选项不同']}**；"
          f"核查方式：{REC['二号表']['题面(问题+选项)对账']}）"
          f"｜**已验证键 {len(R['ok2'])} / {len(R['rows_known'])}**、"
          f"排除 **{len(REC['已排除键'])}**；**不一致的条目一律不得配对**。"
          f"机器可读结果：`outputs/二号材料对账.json`。",
          f"- [!] 配对口径：**只从「两侧证据都通过」的键组成配对**。正确性实际分母 **{nc}**"
          f"（排除 **{dc['排除合计']}**）｜忠实性实际分母 **{nf}**（排除 **{df['排除合计']}**）"
          f"｜二号表被丢弃行 **{R['rows_dropped']}**（模型不在当前 config）"
          f"｜模式＝**{'部分集（--allow-partial）' if partial else '全量正式'}**。",
          # —— 分母等式：唯一权威的分母表述（可核对）——
          f"- [!] **分母等式（可核对，不存在第二个分母）**：{dc['等式']}；{df['等式']}。"
          f"其中「材料/版本校验排除」= fp 复核未过 / 缺回答 / 二号材料对账未过；"
          f"「非法/解析失败标签」= 标签不在 {C} / {F} 内（含空串与解析失败），"
          f"**一律不参与配对**（既不进分子也不进分母）。",
          f"- 被比较回答来源：`runs/*_open_full.jsonl` 且**通过 `7_report.load_all()` 的版本校验**"
          f"（kb_fp / 题目在正式集内 / 模型在 config 内 / 输入指纹复算一致），"
          f"共 {len(R['by_model_qid'])} 条。", ""]
    bc, ctc, nc2, ac = block(f"正确性（实际分母 {nc} 条）", R["pairs_c"], C)
    L += bc + [""]
    bf, ctf, nf2, af = block(f"忠实性（实际分母 {nf} 条）", R["pairs_f"], F)
    L += bf + [""]
    for nm, pairs, labels, col, left, allowed in (
            ("正确性", R["pairs_c"], C, "正确性_二号裁判", R["j1c"],
             set(R["verified"]) & R["ok2"]),
            ("忠实性", R["pairs_f"], F, "忠实性_二号裁判", R["j1f"],
             set(R["faith_verified"]) & R["ok2"])):
        # [!] 分歧清单也只能从**已验证配对**里取（旧实现从原始 j1c + 完整二号表取，
        #     会把未通过校验的记录也列出来）；并且双方标签都必须是合法标签。
        dis = [(key_of_j2(r)[1], key_of_j2(r)[0], left.get(key_of_j2(r), ""),
                (r.get(col) or "").strip())
               for r in R["rows_known"] if key_of_j2(r) in allowed]
        dis = [d for d in dis if d[2] in labels and d[3] in labels and d[2] != d[3]]
        L += [f"## {nm}分歧清单（{len(dis)} 条）", ""]
        L += [f"- {q}｜{m}｜裁判1={a}｜裁判2={b}" for q, m, a, b in dis[:80]]
        if len(dis) > 80:
            L.append(f"- …… 其余 {len(dis) - 80} 条见二分表")
        L.append("")
    return "\n".join(L), (nc2, ac, nf2, af)


def write_recon(REC):
    """落盘机器可读的对账结果，并返回路径。

    [!] 它是**诊断证据**而不是"比对结论"：闸门未通过时也要落盘（否则验收脚本从
        `outputs/二号材料对账.json` 里读到的还是上一次的旧结论，"本次哪里不一致"就丢了）。
        比对产物 `outputs/双裁判比对_全量.md` 则相反：闸门未通过时**必须保持原样**。
    """
    p = dirpath("outputs") / "二号材料对账.json"
    p.write_text(json.dumps(REC, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser(description="全量双裁判比对（正确性 + 忠实性）")
    # [!] 默认 = **全量正式比较**：任何一条缺 fp / 缺回答 / fp 不一致 / 二号表缺列 / 材料对账不一致
    #     → **直接终止**并且**不覆盖**已有产物（原报告保留）。只有显式加 --allow-partial 才做
    #     部分集比较，此时**只从已验证键组成配对**，并把「实际分母 N / 排除 M」写进 stdout 与产物。
    #     [!] 完整性/唯一性闸门（缺行/重复行）**不受 --allow-partial 影响**：结构不成立时
    #         没有"部分集"可谈（与"二号表缺列"同一纪律）。
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许部分集比较：只从已验证键配对，并在产物写明实际分母与排除数")
    ap.add_argument("csv", nargs="?", default=None, help="二号裁判 CSV 路径（默认 v3，缺失时回退 v2）")
    a = ap.parse_args()
    src = Path(a.csv) if a.csv else DEF
    VER = judge2_ver(src)
    # —— 版本断言/打印：无论用 v2 还是 v3，都必须把版本与路径写在 stdout 与产物里 ——
    print(f"裁判2 版本 = {VER}（路径 {src}）")
    try:
        R = analyze(src, a.allow_partial)
    except GateFailure as e:
        # [!] 闸门未通过：比对产物保持原样，但对账证据要落盘（供验收脚本读"本次哪里不一致"）
        if e.evidence.get("REC") is not None:
            print(f"  机器可读对账结果（本次不一致情况）→ {write_recon(e.evidence['REC'])}")
        print(f"\n[X] {e.title}")
        for line in e.lines():
            print(line)
        sys.exit(e.exit_msg)
    nc, ac, nf, af = [None] * 4
    try:
        text, (nc, ac, nf, af) = render(R)
    except GateFailure as e:
        print(f"\n[X] {e.title}")
        for line in e.lines():
            print(line)
        sys.exit(e.exit_msg)
    print(f"  机器可读对账结果 → {write_recon(R['REC'])}")
    if nc:
        print(f"正确性：{ac}/{nc} = {ac / nc * 100:.1f}%")
    if nf:
        print(f"忠实性：{af}/{nf} = {af / nf * 100:.1f}%")
    out = dirpath("outputs") / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(f"\n[OK] → {out}")


if __name__ == "__main__":
    main()
