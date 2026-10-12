# -*- coding: utf-8 -*-
r"""
重答"把输出额度用尽、可能被截断"的记录（B 类）。

背景：扫描 900 条发现 68 条 out_tokens 已达上限（2000）→ 回答可能被截断，而忠实性/正确性
判定都建立在这段文本上。这里**只重答这些记录**，用更大额度；其余记录不受影响
（额度按记录保存在 max_tokens_used，指纹也随之按记录计算）。

用法：
  python _selftest\reanswer_truncated.py --dry-run     # 只看会重答哪些
  python _selftest\reanswer_truncated.py --budget 6000 # 真跑（写回 runs/*_full.jsonl，先备份）
"""
import argparse
import importlib.util
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import (CFG, dirpath, path, read_jsonl, record_budgets,
                    record_hit_budget_cap)   # noqa: E402
# [!] 显式再导出：本模块的截断判据与扫描脚本共用同一实现（common.record_hit_budget_cap），
#     回归测试直接按 mra.record_hit_budget_cap 断言，避免两处判据再次漂移。
__all__ = ["main", "invalidate_verdict_cache", "record_hit_budget_cap", "record_budgets"]

_s = importlib.util.spec_from_file_location("ra", ROOT / "src" / "4_rag_answer.py")
ra = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ra)


def invalidate_verdict_cache(affected, jname):
    """让**当前裁判**判定缓存里受影响的条目失效；返回失效条数。

    [!] 本函数是第 4 项修复的核心，抽出来是为了能被回归测试**直接断言**：
        · 缓存行的 `model` 是**显示名**，run 记录的 `model_name` 是模型 ID ——
          旧代码用 `r["model_name"]` 去比 `o["model"]`，**交集 0/450，整个失效逻辑是空操作**；
        · 只允许改 `auto_verdicts_{jname}.jsonl`，绝不 glob 命中 `.OLD-RUBRIC-*`、
          `.BAD-*`、`.bak-*` 这些历史留档（旧代码会把它们一并就地截断 = 破坏证据）。
    `affected` 是 {(显示名, 题号), ...}。
    """
    cache_file = dirpath("outputs") / f"auto_verdicts_{jname}.jsonl"
    if not cache_file.exists():
        return 0
    keep, dropped = [], 0
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        # [!] 只丢当前裁判的条目：混入别的 judge_name 属于数据串用，
        #     保留下来由 7d 的断言报出来，而不是在这里静默删除。
        if o.get("judge_name", jname) == jname \
                and (o.get("model"), o.get("question_id")) in affected:
            dropped += 1
            continue
        keep.append(line)
    cache_file.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    qa = {x["question_id"]: x for x in read_jsonl(path("qa_set"))}
    models = {m["name"]: m for m in CFG["models"]}
    files = sorted(dirpath("runs").glob("*_full.jsonl"))
    print(f"扫描 {len(files)} 个记录文件")

    total = hit = 0
    plan = []
    for f in files:
        rows = read_jsonl(f)
        total += len(rows)
        for r in rows:
            # [!] 判据统一走 common.record_hit_budget_cap：它读**这条记录自己**试过的全部额度
            #     （max_tokens_used + attempts + attempt_budgets），不再写死 2000*k。
            #     只有"恰好停在某个额度边界"才可疑：额度翻倍后**正常答完**的记录
            #     （如 out=5079、额度={2000,4000}）不能被误判成截断，否则白花钱重答。
            if record_hit_budget_cap(r, ra.MAX_TOKENS):
                hit += 1
                plan.append((f, r))
    print(f"记录 {total} 条；**额度用尽（可能截断）{hit} 条**")
    caps = sorted({b for _, r in plan for b in record_budgets(r, ra.MAX_TOKENS)})
    print(f"  判定用的额度 cap（读自记录自身）：{caps or '无'}｜默认额度 {ra.MAX_TOKENS}")
    print("  按模型：", dict(Counter(r["model"] for _, r in plan)))
    print("  按模式：", dict(Counter(r.get("mode") for _, r in plan)))
    if a.dry_run:
        for f, r in plan[:12]:
            print(f"    {r['model']:<12}{r['question_id']} {r.get('mode')} "
                  f"out={r.get('out_tokens')}/{record_budgets(r, ra.MAX_TOKENS)}")
        print("  （--dry-run：未真正重答）")
        return

    by_file = {}
    still_capped = 0
    for i, (f, r) in enumerate(plan, 1):
        m = models.get(r.get("model_name"))
        item = qa.get(r["question_id"])
        if not m or not item:
            print(f"  [!] 跳过 {r['question_id']}（模型或题目缺失）")
            continue
        # ctx 必须是库自己构造的字典（含 context/chunks），不能传拼接字符串
        ctx = ra.ctx_from_record(r) if r.get("mode") == "open" else ra.empty_ctx()
        new = ra.answer_one(m, r.get("mode") or "open", item, ctx, budget=a.budget)
        # [!] 判"仍达上限"也要用**记录自身试过的额度**（answer_one 现在会把最终成功那次
        #     的额度写进 max_tokens_used，并留 attempts 明细）——只看 a.budget 会漏判
        #     "额度被 chat() 再翻倍后仍停在上限"的情况。
        if record_hit_budget_cap(new, a.budget) and not new.get("error"):
            still_capped += 1
        by_file.setdefault(f, {})[(r.get("mode"), r["question_id"])] = new
        print(f"  {i}/{len(plan)} {r['model']:<12}{r['question_id']} {r.get('mode'):<6} "
              f"{r.get('out_tokens')}→{new.get('out_tokens')} token"
              f"（额度 {record_budgets(new, a.budget)}，finish={new.get('finish_reason') or '-'}）"
              f"{'  [仍达上限]' if record_hit_budget_cap(new, a.budget) else ''}")
        if i % 20 == 0:
            _write_back(files, by_file, dry=False)
            print("     （已增量写回）")

    _write_back(files, by_file, dry=False)

    # 让受影响记录的正确性判定缓存失效（忠实性缓存按指纹自动失效）
    # [!] 统一用**显示名**比对（见 invalidate_verdict_cache 的说明），且只动当前裁判的缓存。
    jname = CFG["judge"]["name"]
    affected = {(r["model"], r["question_id"]) for _, r in plan}
    dropped = invalidate_verdict_cache(affected, jname)
    if not (dirpath("outputs") / f"auto_verdicts_{jname}.jsonl").exists():
        print(f"[i] 没有 auto_verdicts_{jname}.jsonl，跳过正确性缓存失效"
              f"（先跑 7c_export_full_review.py）")
    print(f"\n[OK] 重答 {len(plan)} 条；其中重答后**仍达上限 {still_capped} 条**"
          f"（这些应作为'未完成'单列）")
    print(f"     已让 {dropped} 条正确性判定缓存失效（auto_verdicts_{jname}.jsonl，"
          f"下次导出会重判这些）；忠实性缓存按指纹自动失效")
    print("     [!] `runs/*__open.json`（route B 用）未同步更新，正式链路不使用它。")


def _write_back(files, by_file, dry):
    for f in files:
        if f not in by_file:
            continue
        rows = read_jsonl(f)
        # [!] 留档文件名带**时间戳**：旧代码固定叫 *.bak-before-reanswer，第二次重答时
        #     因为"已存在就不覆盖"，留档仍是**第一次**重答前的版本，会被误当成最新备份。
        bak = f.parent / (f.name + ".bak-before-reanswer-" + time.strftime("%Y%m%d-%H%M%S"))
        if not bak.exists():
            shutil.copy2(f, bak)
        out = [by_file[f].get((r.get("mode"), r["question_id"]), r) for r in rows]
        if not dry:
            with open(f, "w", encoding="utf-8", newline="\n") as fh:
                for r in out:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
