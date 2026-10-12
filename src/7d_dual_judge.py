# -*- coding: utf-8 -*-
r"""
双裁判协议支持（盲评 + 分歧比对）。两个维度分开记录：**正确性**（回答 vs 标准答案）与
**忠实性**（回答 vs 检索片段）。

--export   导出**盲评包**：按模型分 5 个文件（每模型 90 条），含两个维度的判分材料，
           **不含任何裁判判定**（避免锚定），列内预置两个待填判定列。
--compare  比对：裁判1 vs 裁判2（vs 人工，若已填），逐维度给一致率/Kappa/交叉表 + 分歧清单。

用法：
  python src\7d_dual_judge.py --export                 # 生成盲评包（不调用任何模型）
  python src\7d_dual_judge.py --compare                # 裁判2 填好后运行
产物：
  qa\双裁判盲评\<模型>.csv      （盲评包，5 个文件）
  outputs\双裁判比对.md          （--compare 的输出）
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import (CFG, dirpath, full_question, g_col, kappa_of, path, read_csv_any,
                    read_jsonl)   # noqa: E402

BLIND_DIR = "双裁判盲评"
CORR_LABELS = ["一致", "部分一致", "不一致"]
FAITH_LABELS = ["忠实", "部分无依据", "无依据"]
COLS = ["序号", "题目ID", "模型", "题型", "是否可答", "问题（含选项）", "标准答案", "原文证据句",
        "检索片段（忠实性判定用）", "模型回答",
        "正确性判定(填:一致/部分一致/不一致)", "正确性-错误类型",
        "忠实性判定(填:忠实/部分无依据/无依据)", "忠实性-无依据的具体句子", "备注"]
# 盲评包里**由裁判手工填写**的列：重新导出时绝不能清空（原实现每次写空 = 静默毁掉裁判的判定）。
MAN_COLS = COLS[10:]
CHANGED_NOTE = "回答已变更，需重新审核"
_row_fp = {}


def row_fp(row) -> str:
    """判分材料指纹（题面 + 标准答案 + 检索片段 + 模型回答），不含人工列。"""
    key = id(row)
    if key not in _row_fp:
        raw = json.dumps([row.get("问题（含选项）", ""), row.get("标准答案", ""),
                          row.get("检索片段（忠实性判定用）", ""), row.get("模型回答", "")],
                         ensure_ascii=False, sort_keys=True)
        _row_fp[key] = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return _row_fp[key]


def load_prev_manual(csv_path: Path):
    """读回上一版盲评包里已填的判定列（按 题目ID）。多编码兜底：裁判用 Excel 填表。"""
    prev = {}
    for r in read_csv_any(csv_path):
        q = g_col(r, "题目ID")
        if not q:
            continue
        filled = {c: (r.get(c) or "").strip() for c in MAN_COLS if (r.get(c) or "").strip()}
        if filled:
            prev[q] = {"filled": filled, "row_fp": row_fp(r)}
    return prev


def apply_manual_keep(rows, prev):
    """并回已填的判定列；回答/材料变更时保留内容并在备注标注待重新审核。"""
    kept = flagged = 0
    for row in rows:
        old = prev.get(row["题目ID"])
        if not old:
            continue
        for c, v in old["filled"].items():
            if c in row:
                row[c] = v
        kept += 1
        if old["row_fp"] != row_fp(row):
            note = row.get("备注") or ""
            if CHANGED_NOTE not in note:
                row["备注"] = f"[{CHANGED_NOTE}] " + note
            flagged += 1
    return kept, flagged


def backup_path(p: Path):
    """覆盖前自动备份（带时间戳，冲突追加序号）；失败即放弃覆盖。"""
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
                         f"    为避免清空裁判已填的判定列，本次导出已终止。")
    return dst



def _r7():
    spec = importlib.util.spec_from_file_location("r7x", ROOT / "src" / "7_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _by_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _valid_records():
    """当前**有效**的开卷记录（判据与榜单/7c 完全同源）。

    [!] 2026-10-12 补齐（第三方复验补充检查 A1 修复要求第 5 条）：本入口此前直接 glob
        `runs/*_open_full.jsonl` 并把题号相同的记录当成"同一份输入"——改题干/选项/标准答案/
        作答参数/知识库之后，盲评包里会混进**旧回答**。现在统一走 `7_report.load_all()`：
        kb_fp、题目在正式集内、模型在当前 config、**作答输入指纹逐条复算**，全部通过才采用；
        过期记录被排除并打印原因，绝不静默混入。
    """
    r7 = _r7()
    try:
        rows = r7.load_all("*_open_full.jsonl")
    except SystemExit as e:
        raise SystemExit(f"[X] 回答未通过版本校验 → 终止本入口（不使用历史数据）：\n{e}")
    return [r for r in rows if r.get("mode") == "open" and (r.get("response") or "").strip()]


def _verify_judge1(rec_by_key, j_name):
    """逐条复核裁判1 的**判分材料指纹**，返回 (已验证的正确性缓存, 已验证的忠实性判定, 统计)。

    为什么两个维度都要核（同一根因）：只核对"裁判名"或"题号存在"是不够的——
    缓存行里的 `fp` 必须等于**用当前被比较回答复算出来的 fp**，
    否则就是拿旧回答的判定去配新回答（改题面/选项/标准答案/知识库/回答后尤甚）。
    正确性用 `7c.fp_of`，忠实性用 `5b.fp_of`（各自判分链路的原始定义，不另写一份）。
    """
    c7, m5b = _by_file("d7c", "7c_export_full_review.py"), _by_file("d5b", "5b_judge_faithfulness.py")
    judge_cfg = CFG["judge"]
    qa_now = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    j1c, j1f = {}, {}
    st = {"corr": {"ok": 0, "bad": [], "miss": []}, "faith": {"ok": 0, "bad": [], "miss": []}}
    for r in read_jsonl(dirpath("outputs") / f"auto_verdicts_{j_name}.jsonl"):
        got = r.get("judge_name")
        if got and got != j_name:
            sys.exit(f"[X] auto_verdicts_{j_name}.jsonl 里出现 judge_name={got!r} 的判定"
                     f"（当前裁判是 {j_name!r}）。\n"
                     f"    该文件被手工改名/串用过，混用两套口径算出的 Kappa 无效；请重新导出。")
        key = (r["model"], r["question_id"])
        rec = rec_by_key.get(key)
        item = qa_now.get(r["question_id"])
        if item is None or rec is None or not r.get("fp"):
            st["corr"]["miss"].append((key, "题目不在正式集/当前回答缺失/缓存行缺 fp"))
            continue
        try:
            want = c7.fp_of(item, rec, judge_cfg)
        except Exception as e:
            st["corr"]["miss"].append((key, f"复算失败 {type(e).__name__}"))
            continue
        if want != r.get("fp"):
            st["corr"]["bad"].append((key, r.get("fp"), want))
            continue
        st["corr"]["ok"] += 1
        j1c[key] = r.get("judge", "")
    for r in read_jsonl(dirpath("outputs") / "faithfulness.jsonl"):
        key = (r["model"], r["question_id"])
        rec = rec_by_key.get(key)
        if rec is None or not r.get("fp"):
            st["faith"]["miss"].append((key, "当前回答缺失/判定行缺 fp"))
            continue
        try:
            want = m5b.fp_of(rec, j_name)
        except Exception as e:
            st["faith"]["miss"].append((key, f"复算失败 {type(e).__name__}"))
            continue
        if want != r.get("fp"):
            st["faith"]["bad"].append((key, r.get("fp"), want))
            continue
        st["faith"]["ok"] += 1
        j1f[key] = r.get("verdict", "")
    return j1c, j1f, st


def do_export():
    qa = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    recs = collections.defaultdict(list)
    for x in _valid_records():
        recs[x["model"]].append(x)
    out_dir = path("qa_set").parent / BLIND_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # [!] 盲评包是**裁判手工填的**：重新导出前必须备份整个目录，并按 (题目ID, 材料指纹) 并回已填判定。
    #     原实现直接以 w 覆盖同名 CSV → 已填的判定被静默清空（与 7c 同一根因）。
    existing = sorted(out_dir.glob("*.csv"))
    if existing:
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = out_dir.parent / f"{BLIND_DIR}.bak-{ts}"
        n = 1
        while dst.exists():
            n += 1
            dst = out_dir.parent / f"{BLIND_DIR}.bak-{ts}-{n}"
        shutil.copytree(out_dir, dst)
        print(f"  覆盖前备份 → {dst.name}（{len(existing)} 个文件）")
    total = 0
    kept_all = flagged_all = 0
    for model, rs in sorted(recs.items()):
        rows = []
        for i, rec in enumerate(sorted(rs, key=lambda r: r["question_id"]), 1):
            item = qa[rec["question_id"]]
            # [!] 抽公共 full_question：题面拼法必须与 7_report / 7c 的判分输入**完全一致**，
            #     否则同一道题在两处看到的题面不同，判定与比对就不可比。
            q = full_question(item)
            ctx = "\n\n".join(f"[片段{j}] {t}" for j, t in
                              enumerate(rec.get("retrieved_contexts", []), 1))
            rows.append({"序号": i, "题目ID": rec["question_id"], "模型": rec["model"],
                         "题型": rec.get("type", ""),
                         "是否可答": "是" if rec.get("answerable") else "否",
                         "问题（含选项）": q, "标准答案": item["reference_answer"],
                         "原文证据句": item.get("evidence_quote", ""),
                         "检索片段（忠实性判定用）": ctx[:9000],
                         "模型回答": (rec.get("response") or "")[:4000],
                         "正确性判定(填:一致/部分一致/不一致)": "",
                         "正确性-错误类型": "",
                         "忠实性判定(填:忠实/部分无依据/无依据)": "",
                         "忠实性-无依据的具体句子": "", "备注": ""})
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in model)
        f = out_dir / f"{safe}.csv"
        kept, flagged = apply_manual_keep(rows, load_prev_manual(f))
        kept_all += kept
        flagged_all += flagged
        with open(f, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        total += len(rows)
        extra = ""
        if kept:
            extra = f"（并回已填 {kept} 条" + (f"，{flagged} 条待重新审核）" if flagged else "）")
        print(f"  {model:<34} {len(rows)} 条 → {f.name}{extra}")
    print(f"\n[OK] 盲评包已生成：{len(recs)} 个文件、共 {total} 条 → {out_dir}")
    if kept_all:
        print(f"     人工列保护：并回已填 {kept_all} 条"
              + (f"，其中 {flagged_all} 条材料已变更 → 备注标注「{CHANGED_NOTE}」" if flagged_all else ""))
    print("     [!] 盲评包内**不含任何裁判判定**；请第二位裁判逐模型独立评分（先正确性、再忠实性）。")
    print("     [!] 判完后把两个判定列填回同名文件，再运行 --compare。")


def _kappa(pairs, labels):
    """委托 common.kappa_of（全项目统一退化约定与"解析失败"排除口径）。"""
    return kappa_of(pairs, labels)


def _boot_ci(pairs, labels, boot=2000):
    rnd = random.Random(7)
    ks, ps = [], []
    for _ in range(boot):
        s = [pairs[rnd.randrange(len(pairs))] for _ in range(len(pairs))]
        k, po, _ = _kappa(s, labels)
        if k is not None:
            ks.append(k)
            ps.append(po)
    ks.sort()
    ps.sort()
    if not ks:
        return None, None
    return (ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks)) - 1]), \
           (ps[int(0.025 * len(ps))], ps[int(0.975 * len(ps)) - 1])


def do_compare():
    out_dir = path("qa_set").parent / BLIND_DIR
    files = sorted(out_dir.glob("*.csv"))
    if not files:
        sys.exit(f"没有盲评包：{out_dir}（先 --export）")
    # 裁判1 的判定：正确性来自 auto_verdicts_*.jsonl（需先跑导出器），忠实性来自 faithfulness.jsonl
    # [!] 只读**指定裁判自己的**缓存文件，绝不用 glob("auto_verdicts_*.jsonl") 合并全目录：
    #     目录里还有被手工改名的旧口径留档（如 .OLD-RUBRIC-v1.jsonl），而 ".OLD-…" 在字典序上
    #     **排在 ".jsonl" 之后**，于是它的旧口径判定会覆盖新判定 ——
    #     实测已污染 27 条「裁判1」标签（同一批题号、两套口径静默混用）。
    #     同时逐行断言 judge_name 一致：缓存文件被改名/串用时要立刻报错，而不是算出一份错的 Kappa。
    j_name = CFG["judge"]["name"]
    cache_file = dirpath("outputs") / f"auto_verdicts_{j_name}.jsonl"
    # [!] 判分材料必须逐条绑定**当前回答**（同上）：正确性用 7c.fp_of、忠实性用 5b.fp_of 复算，
    #     不一致/缺失的一律**排除出配对**（绝不拿旧判定配新回答），并把计数写进产物。
    rec_by_key = {(r["model"], r["question_id"]): r for r in _valid_records()}
    j1c_v, j1f_v, vst = _verify_judge1(rec_by_key, j_name)
    print(f"裁判1 判分材料指纹复核：正确性 一致 {vst['corr']['ok']}｜不一致 {len(vst['corr']['bad'])}"
          f"｜缺 {len(vst['corr']['miss'])}；忠实性 一致 {vst['faith']['ok']}"
          f"｜不一致 {len(vst['faith']['bad'])}｜缺 {len(vst['faith']['miss'])}")
    for dim in ("corr", "faith"):
        for k, a, b in vst[dim]["bad"][:3]:
            print(f"   [!] {dim} {k[0]}/{k[1]}：缓存 fp={a} ≠ 当前回答复算 fp={b}")
        for k, why in vst[dim]["miss"][:3]:
            print(f"   [!] {dim} {k[0]}/{k[1]}：{why}")
    j1 = {}
    has_corr_j1 = False
    if not cache_file.exists():
        print(f"[!] 找不到裁判1 的正确性缓存 {cache_file.name} —— 先跑 python src\\7c_export_full_review.py")
    for key, v in j1c_v.items():
        j1[(key[0], key[1], "corr")] = v
        has_corr_j1 = True
    for key, v in j1f_v.items():
        j1[(key[0], key[1], "faith")] = v

    rows = []
    for f in files:
        # [!] 盲评包是第二位裁判在 Excel 里回填的，编码可能是 GBK → 统一走 common.read_csv_any
        #     （打分制编码判定；"不抛错就采用"会把 GBK 静默解成乱码）。
        rows += read_csv_any(f)
    print(f"盲评包 {len(files)} 个文件、{len(rows)} 条")

    md = ["# 双裁判比对（正确性与忠实性分开）", "",
          f"- 样本：{len(rows)} 条开卷回答（{len(files)} 个模型文件）",
          f"- 裁判1 正确性判定："
          f"{'已加载并通过材料指纹复核 ' + str(vst['corr']['ok']) + ' 条' if j1c_v else ('**缺失**（先跑 7c_export_full_review.py）' if not cache_file.exists() else '**0 条可用**（全部未通过材料指纹复核）')}"
          f"｜来源：`outputs/auto_verdicts_{j_name}.jsonl`（**只读指定裁判的缓存**，"
          f"不合并 .OLD-RUBRIC/.BAD/.bak 等历史留档）",
          f"- [!] **判分材料指纹逐条复核**（第三方复验补充检查 A1 修复要求第 5 条）："
          f"正确性用 `7c.fp_of`、忠实性用 `5b.fp_of` 与**当前被比较回答**复算比对 →"
          f" 正确性 一致 **{vst['corr']['ok']}** / 不一致 **{len(vst['corr']['bad'])}** / 缺 **{len(vst['corr']['miss'])}**；"
          f"忠实性 一致 **{vst['faith']['ok']}** / 不一致 **{len(vst['faith']['bad'])}** / 缺 **{len(vst['faith']['miss'])}**。"
          f"**未通过复核的条目一律不进入配对**（拿旧判定配新回答会算出错的 Kappa）。",
          "- 口径：「解析失败」等不可解析的判定**排除**在一致率/Kappa 的分子分母之外；"
          "pe>=1 时 po==1 记 1.0、否则记 0.0（全项目统一）。", ""]
    for dim, labels, col2, key in (
            ("正确性", CORR_LABELS, "正确性判定(填:一致/部分一致/不一致)", "corr"),
            ("忠实性", FAITH_LABELS, "忠实性判定(填:忠实/部分无依据/无依据)", "faith")):
        pairs = []
        for r in rows:
            v2 = (r.get(col2) or "").strip()
            v1 = j1.get((r["模型"], r["题目ID"], key), "")
            if v2 in labels and v1 in labels:
                pairs.append((v1, v2))
        md += [f"## {dim}", ""]
        if not pairs:
            md += [f"- 尚无可比数据（裁判2 的「{col2}」为空，或裁判1 该维度缺失）", ""]
            continue
        k, po, pe = _kappa(pairs, labels)
        ci_k, ci_p = _boot_ci(pairs, labels)
        agree = sum(1 for a, b in pairs if a == b)
        md += [f"- 可比条数：**{len(pairs)}**；两位裁判完全一致：**{agree}/{len(pairs)} = "
               f"{agree / len(pairs) * 100:.1f}%**（bootstrap 95% 区间 "
               f"{ci_p[0] * 100:.1f}%~{ci_p[1] * 100:.1f}%）",
               f"- Cohen's Kappa：**{k:.3f}**（95% 区间 {ci_k[0]:.3f}~{ci_k[1]:.3f}）；po={po:.3f}，pe={pe:.3f}",
               "", "| 裁判1＼裁判2 | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        ct = collections.defaultdict(collections.Counter)
        for a, b in pairs:
            ct[a][b] += 1
        for a in labels:
            md.append(f"| **{a}** | " + " | ".join(str(ct[a].get(b, 0)) for b in labels) + " |")
        dis = [(r["题目ID"], r["模型"], j1.get((r["模型"], r["题目ID"], key), ""),
                (r.get(col2) or "").strip()) for r in rows
               if (r.get(col2) or "").strip() in labels
               and j1.get((r["模型"], r["题目ID"], key), "") in labels
               and (r.get(col2) or "").strip() != j1.get((r["模型"], r["题目ID"], key), "")]
        md += ["", f"分歧 {len(dis)} 条（题目ID｜模型｜裁判1｜裁判2）：", ""]
        md += [f"- {q}｜{m}｜{a}｜{b}" for q, m, a, b in dis[:60]]
        if len(dis) > 60:
            md.append(f"- …… 其余 {len(dis) - 60} 条见盲评包")
        md.append("")
    out = dirpath("outputs") / "双裁判比对.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:16]))
    print(f"\n…完整内容见 {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.export or not a.compare:
        do_export()
    if a.compare:
        do_compare()
