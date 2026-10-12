# -*- coding: utf-8 -*-
r"""
人工抽检的统计口径（独立模块，供 7_report 引用）。

为什么单独写：抽检是**按题型分层抽样**（数值/陷阱/其他各 10 条），
所以不能只报一个"总体一致率"就算完——必须给出：
  1. 实际一致条数（分子/分母）
  2. **分层一致率**（每层 n、一致数、比率）
  3. **判定交叉表**（自动判定 × 人工判定 的 3×3）
  4. Cohen's Kappa（多分类未加权）
  5. 若要推断到全部 450 条：**按总体分层占比加权的估计** + **与抽样设计匹配的区间**（分层 bootstrap）
  6. 抽样设计说明（分层富集样本 ≠ 无偏随机样本）

用法：
  python src\7b_spot_stats.py               # 读 qa\人工抽检表.csv 出统计
  python src\7b_spot_stats.py --simulate    # 用合成人工判定自测算法（不写真实结果）
产物：outputs\抽检统计.md
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import (CFG, dirpath, g_col, kappa_of, path, read_csv_any,
                    read_jsonl)   # noqa: E402

LABELS = ["一致", "部分一致", "不一致"]
HUMAN_COL = "人工判定(填:一致/部分一致/不一致)"
AUTO_COL = "自动判定"
BOOT = 2000


def g(row, *names):
    """按列名取值 → 见 common.g_col。"""
    return g_col(row, *names)


def norm(x):
    x = (x or "").strip().replace(" ", "")
    for lab in LABELS:                     # 容忍"部分一致（理由…）"这类写法
        if x.startswith(lab):
            return lab
    return x


def stratum_of(r):
    """与 make_spot_check 的分层一致：陷阱 / 数值 / 其他。"""
    if g(r, "是否可答") == "否":
        return "陷阱题"
    return "数值题" if g(r, "题型") == "number" else "其他题型"


def kappa(pairs):
    """多分类未加权 Cohen's Kappa，pairs = [(auto, human), ...]。

    [!] 委托 common.kappa_of，与其他报告统一：
        · 「解析失败」等非标签判定**排除**（既不进分子也不进分母）；
        · pe>=1 的退化情形全项目统一（po==1 → 1.0，否则 0.0），
          过去本文件一律返回 0.0 而 7_report 一律返回 1.0，同一份样本两份产物互相矛盾。
    """
    return kappa_of(pairs, LABELS)


def weighted_rate(per_stratum, pop_w):
    tot = sum(pop_w.get(h, 0) for h in per_stratum)
    if not tot:
        return None
    return sum(pop_w.get(h, 0) / tot * v["rate"] for h, v in per_stratum.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true",
                    help="用合成人工判定自测算法（结果不写入真实产物）")
    a = ap.parse_args()

    sheet = path("qa_set").parent / "人工抽检表.csv"
    if not sheet.exists():
        sys.exit(f"找不到 {sheet} —— 先跑 python src\\7_report.py 生成抽检表")
    # [!] 统一走 common.read_csv_any：编码判定是**打分制**（不是"不抛错就采用"）——
    #     GBK 的汉字字节串常常也是合法 UTF-8，只试解码会把 GBK 表静默解成乱码。
    rows = read_csv_any(sheet)
    # [!] 「错误类型」等列名可能被 Excel 改过 → g() 用前缀/精确双匹配（见 common.g_col）。
    for r in rows:
        r["_auto"] = norm(g(r, AUTO_COL))
        r["_human"] = norm(g(r, HUMAN_COL))
        r["_stratum"] = stratum_of(r)
    if a.simulate:
        rnd = random.Random(20261010)
        for r in rows:
            r["_human"] = r["_auto"] if rnd.random() < 0.9 else \
                rnd.choice([x for x in LABELS if x != r["_auto"]])
        print("[simulate] 已用合成人工判定（约 90% 与自动一致）自测算法\n")

    filled = [r for r in rows if r["_human"] in LABELS]
    if not filled:
        sys.exit("抽检表里还没有填写「人工判定」，无法统计。")
    # [!] 进入统计的行必须**两边都是合法标签**：自动判定是"解析失败"时既算不出一致，
    #     也不该算进分层一致率的分母（那会把一致率人为压低）。排除条数在产物里写明。
    n_filled = len(filled)
    n_parse_fail = sum(1 for r in filled if r["_auto"] not in LABELS)
    ok_rows = [r for r in filled if r["_auto"] in LABELS]

    # 总体分层占比（用 450 条开卷记录的定义，与抽检表同口径）
    pop = Counter()
    for f in sorted(dirpath("runs").glob("*_open_full.jsonl")):
        for x in read_jsonl(f):
            if not x.get("error") and x.get("response"):
                pop["陷阱题" if not x.get("answerable")
                    else ("数值题" if x.get("type") == "number" else "其他题型")] += 1
    pop_w = {h: pop.get(h, 0) for h in ["数值题", "陷阱题", "其他题型"]}

    per = {}
    for h in ["数值题", "陷阱题", "其他题型"]:
        rs = [r for r in ok_rows if r["_stratum"] == h]
        ag = sum(1 for r in rs if r["_auto"] == r["_human"])
        per[h] = {"n": len(rs), "agree": ag, "rate": (ag / len(rs)) if rs else 0.0}

    pairs = [(r["_auto"], r["_human"]) for r in ok_rows]
    n_all = len(pairs)
    agree_all = sum(1 for a_, h_ in pairs if a_ == h_)
    k, po, pe = kappa(pairs)
    rate_w = weighted_rate(per, pop_w)

    # 与抽样设计匹配的区间：**分层 bootstrap**（每层内重抽，保持分层样本量）
    rnd = random.Random(42)
    by_h = {h: [r for r in ok_rows if r["_stratum"] == h] for h in per}
    bw, bk, ba = [], [], []
    for _ in range(BOOT):
        samp = []
        for h, rs in by_h.items():
            if rs:
                samp += [rs[rnd.randrange(len(rs))] for _ in range(len(rs))]
        pr = {}
        for h in per:
            rs = [r for r in samp if r["_stratum"] == h]
            ag = sum(1 for r in rs if r["_auto"] == r["_human"])
            pr[h] = {"n": len(rs), "agree": ag, "rate": (ag / len(rs)) if rs else 0.0}
        bw.append(weighted_rate(pr, pop_w) or 0.0)
        ba.append(sum(1 for r in samp if r["_auto"] == r["_human"]) / len(samp))
        kk = kappa([(r["_auto"], r["_human"]) for r in samp])[0]
        bk.append(kk if kk is not None else 0.0)

    def ci(v):
        v = sorted(v)
        return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]

    cross = defaultdict(Counter)
    for a_, h_ in pairs:
        cross[a_][h_] += 1

    L = ["# 人工抽检统计（分层设计）", "",
         f"- 抽检表：`qa/人工抽检表.csv`｜已填「人工判定」**{n_filled} / {len(rows)}** 条",
         f"- 排除「解析失败」**{n_parse_fail}** 条（自动判定不可解析为 一致/部分一致/不一致，"
         f"既不进分子也不进分母）；进入统计 **{n_all}** 条",
         f"- **实际一致条数：{agree_all} / {n_all} = {agree_all / n_all * 100:.1f}%**"
         f"（分层 bootstrap 95% 区间 {ci(ba)[0] * 100:.1f}%~{ci(ba)[1] * 100:.1f}%）",
         f"- Cohen's Kappa：**{k:.3f}**（95% 区间 {ci(bk)[0]:.3f}~{ci(bk)[1]:.3f}）"
         f"；观测一致率 po={po:.3f}，期望偶然一致 pe={pe:.3f}" if k is not None else "- Kappa：无法计算",
         "", "## 分层一致率", "",
         "| 层 | 抽检条数 | 一致条数 | 一致率 | 总体占比（450 条） |", "|---|---|---|---|---|"]
    for h in ["数值题", "陷阱题", "其他题型"]:
        v = per[h]
        L.append(f"| {h} | {v['n']} | {v['agree']} | {v['rate'] * 100:.0f}% | "
                 f"{pop_w[h] / sum(pop_w.values()) * 100:.0f}% |")

    L += ["", "## 判定交叉表（行=自动判定，列=人工判定）", "",
          "| 自动＼人工 | " + " | ".join(LABELS) + " | 合计 |",
          "|---|" + "---|" * (len(LABELS) + 1)]
    for a_ in LABELS:
        rowsum = sum(cross[a_].values())
        L.append(f"| **{a_}** | " + " | ".join(str(cross[a_].get(h_, 0)) for h_ in LABELS)
                 + f" | {rowsum} |")
    L.append("| 合计 | " + " | ".join(str(sum(cross[a_].get(h_, 0) for a_ in LABELS))
                                      for h_ in LABELS) + f" | {n_all} |")

    if rate_w is not None:
        wci = ci(bw)
        L += ["", "## 推断到全部 450 条（按总体分层占比加权）", "",
              f"- **加权一致率 = {rate_w * 100:.1f}%**（分层 bootstrap 95% 区间 "
              f"{wci[0] * 100:.1f}%~{wci[1] * 100:.1f}%）",
              f"- 未加权（原始样本）一致率 = {agree_all / n_all * 100:.1f}%"
              f" —— 两者差异来自**分层富集**（陷阱题在抽检里占 "
              f"{per['陷阱题']['n'] / n_all * 100:.0f}%，在总体里只占 "
              f"{pop_w['陷阱题'] / sum(pop_w.values()) * 100:.0f}%）"]

    L += ["", "## 抽样设计与口径说明", "",
          "1. 抽检**按题型分层**（数值 10 / 陷阱 10 / 其他 10），并非简单随机抽样 →",
          "   原始一致率**不是**全部 450 条的无偏估计；据此推断时须用上面的**加权**结果与**分层 bootstrap** 区间。",
          "2. 区间由**分层 bootstrap**（每层内重抽、保持各层样本量，2000 次）得到，与抽样设计匹配。",
          "3. Kappa 为多分类未加权；样本量小时区间较宽，只宜定性解释（>0.6 较好、>0.8 很好）。",
          "4. 各模型在抽检里只有 5~7 条，**不足以**给出 per-model 的 Kappa。",
          "5. 若判定口径（评分规则）需要修改：**先改规则，再另抽新样本验证**，不要用同一批样本反复调参。",
          "6. 「解析失败」等不可解析的自动判定**排除**在一致率/Kappa 的分子分母之外（全项目统一口径）。",
          "7. Kappa 退化约定（全项目统一）：pe>=1（双方判定完全同分布）时，po==1 记 1.0，否则记 0.0。"]

    out = ROOT / "outputs" / ("抽检统计_simulate.md" if a.simulate else "抽检统计.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:14]))
    print(f"\n…（完整内容见 {out}）")


if __name__ == "__main__":
    main()
