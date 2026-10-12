# -*- coding: utf-8 -*-
r"""旧比对入口（`_selftest/verify_dual_judge.py`）：**复用正式入口的同一套闸门**。

它为什么还在
------------------------------------------------
本脚本最初是"独立复算二号裁判交付摘要里的 84.9% / κ / 交叉表"的一次性核对脚本，
用的是**另一套口径**（默认指向 v1 交付目录、忠实性分母把"无法判定"算进去 = 449、
正确性只在 30 条重叠子集上比）。第三方复验新确认的第 3 个问题就是它：

  · **它从不核对二号回答材料**：二号表是照着某一份回答判的，把二号表里的一条
    `模型回答` 改掉之后，本入口照样输出 `标签一致：382/450 = 84.9%`（已实测复现），
    而正式入口 `_selftest/compare_dual_full.py` 会正确拦下；
  · 它还**默认指向已废弃的 v1 目录**（`二号裁判全量复评_2026-10-10`），
    与当前正式口径 v3 不是同一套尺度。

改法（二选一里的 **(a) 单一实现、两处调用**）
------------------------------------------------
> 为什么不选 (b)「标记废弃、非零退出」：本入口是**对外可复现的一部分**（交付说明里写了
> "独立复算"，面试里也会被问到"你怎么证明不是自己算给自己看的"）。把它变成一个只会报错的壳，
> 等于删掉一个真实的交叉核对位；而 (a) 让它继续可用、并且**不可能再与正式入口漂移**
> （判分输入闸门、标签合法性、分母等式都只有一处实现）。

现在它：
  ① 调用 `compare_dual_full.analyze()` —— 与正式入口**同一套闸门**：
     完整性/唯一性断言（行数 == 唯一键数 == 期望条数）、正确性 fp 复核、忠实性 fp 复核
     （与 `5b_judge_faithfulness.py` 同一个 `fp_of`）、二号材料对账（快照 ↔ 当前回答、
     二号表题面/标准答案/回答指纹 ↔ 当前记录）、标签合法性；
  ② 闸门未通过 → 打印**与正式入口相同的原因**并**非零退出**，**不输出任何比对数字**
     （因此不会再出现"回答被改过、本入口仍报 382/450"）；
  ③ 闸门通过 → 打印**与正式入口一致**的数字（忠实性一致率/κ、正确性一致率/κ、交叉表）；
  ④ 全程**只读**：不写 `outputs/` 下任何文件（产物一律由 `compare_dual_full.py` 写）。

用法：python _selftest\verify_dual_judge.py [二号表.csv]
      （默认与正式入口相同：v3；v3 不存在时才回退 v2）
退出码：闸门未通过 → 1；正常 → 0
"""
import argparse
import collections
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# [!] Windows 控制台默认 GBK：闸门报错会把子进程/输入里的原文打出来，
#     未做 errors=replace 时 print 自身可能抛 UnicodeEncodeError，把真正的原因吞掉。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

C = ["一致", "部分一致", "不一致"]
F = ["忠实", "部分无依据", "无依据"]


def _load_selftest(name, filename):
    """按文件路径加载 `_selftest/` 下的模块（裸 import 会 ModuleNotFoundError）。"""
    p = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# [!] 单一实现、两处调用：正式入口的分析/闸门只此一份
cmp = _load_selftest("vdj_cmp", "compare_dual_full.py")


def _rate(pairs, labels):
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    return agree, n, (agree / n * 100 if n else 0.0)


def main():
    ap = argparse.ArgumentParser(
        description="旧比对入口：与正式入口共用同一套输入闸门（只读复核）")
    ap.add_argument("csv", nargs="?", default=None,
                    help="二号裁判 CSV 路径（默认与正式入口一致：v3，缺失时回退 v2）")
    a = ap.parse_args()
    src = Path(a.csv) if a.csv else cmp.DEF
    print(f"二号裁判 CSV：{src}（版本 {cmp.judge2_ver(src)}）")
    print("[i] 本入口已改为**复用** `_selftest/compare_dual_full.py` 的同一套闸门"
          "（完整性/唯一性 + 正确性 fp 复核 + 忠实性 fp 复核 + 二号材料对账 + 标签合法性）"
          "：**单一实现、两处调用**。")
    print("    旧实现的缺口：默认指向已废弃的 v1 目录，且**从不核对二号回答材料** → "
          "改一条二号表回答后，本入口照样输出与正式入口相同的一致率结论、**不做任何拦截**（已实测）。")
    try:
        R = cmp.analyze(src, False)
    except cmp.GateFailure as e:
        print(f"\n[X] {e.title}")
        for line in e.lines():
            print(line)
        print(e.exit_msg)
        print("[X] 本入口与正式入口走**同一套闸门**：输入无法证明一致时，"
              "绝不输出任何比对数字。")
        return 1

    dc, df = R["denom_c"], R["denom_f"]
    pairs_c, pairs_f = R["pairs_c"], R["pairs_f"]
    rows = R["rows"]
    j2f = collections.Counter((r.get("忠实性_二号裁判") or "").strip() for r in rows)
    j2c = collections.Counter((r.get("正确性_二号裁判") or "").strip() for r in rows)
    print(f"\n① 二号裁判忠实性边际分布（原表 {len(rows)} 行，仅描述性统计、不参与配对口径）："
          f"{dict(j2f)}")
    print(f"② 二号裁判正确性边际分布（原表 {len(rows)} 行）：{dict(j2c)}")

    af, nf, pf = _rate(pairs_f, F)
    kf, pof, pef, nf2 = cmp.kappa(pairs_f, F)
    print(f"\n③ 忠实性双裁判比对（**与正式入口同一口径**：只从两侧证据都通过、且标签合法的键配对）"
          f"\n   分母等式：{df['等式']}")
    print(f"   标签一致：{af}/{nf} = {pf:.1f}%")
    print(f"   忠实性：{af}/{nf} = {pf:.1f}%")
    if kf is not None:
        print(f"   Cohen's κ = {kf:.3f}（po={pof:.3f}, pe={pef:.3f}）")

    ac, nc, pc = _rate(pairs_c, C)
    kc, poc, pec, nc2 = cmp.kappa(pairs_c, C)
    print(f"\n④ 正确性双裁判比对（同一口径）\n   分母等式：{dc['等式']}")
    print(f"   标签一致：{ac}/{nc} = {pc:.1f}%")
    print(f"   正确性：{ac}/{nc} = {pc:.1f}%")
    if kc is not None:
        print(f"   Cohen's κ = {kc:.3f}（po={poc:.3f}, pe={pec:.3f}）")

    ct = collections.defaultdict(collections.Counter)
    for x, y in pairs_f:
        ct[x][y] += 1
    print("\n⑤ 忠实性交叉表（行=一号，列=二号；只含合法标签）：")
    for x in F:
        if sum(ct[x].values()):
            print(f"   {x:<8}" + "  ".join(f"{y}:{ct[x].get(y, 0):>3}" for y in F))
    print(f"\n⑥ 一号忠实性边际分布："
          f"{dict(collections.Counter(x for x, _ in pairs_f if x in F))}")
    print(f"\n[OK] 与正式入口 `_selftest/compare_dual_full.py` 数字一致："
          f"正确性 {ac}/{nc} = {pc:.1f}%、忠实性 {af}/{nf} = {pf:.1f}%"
          f"（正式产物：outputs/双裁判比对_全量.md）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
