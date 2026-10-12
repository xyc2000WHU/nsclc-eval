# -*- coding: utf-8 -*-
"""
第 6 步：用 EvalScope 跑两件事（这一步最稳，几乎不会失败，一定要跑通）

  A. 选择题准确率（general_mcq）：不需要裁判模型，纯客观，是榜单的"保底指标"
  B. 闭卷问答正确率（general_qa + LLM 裁判）：衡量模型"自己知道多少"

用法：
  python src\6_eval_evalscope.py --only mcq        # 先跑这个，5 分钟内有结果
  python src\6_eval_evalscope.py --only qa
  python src\6_eval_evalscope.py
"""
import argparse
import json          # [!] 少了它 --only qa 一进来就 NameError → schema 闸门整个失效
import os
from pathlib import Path

from common import (CFG, add_dev_flag, dirpath, enable_dev, eval_workdir, models_with_key,
                    path)

from evalscope import TaskConfig, run_task


def judge_cfg():
    j = CFG["judge"]
    return {
        "strategy": "llm",
        "models": {                       # 若报参数错误，按 EvalScope 最新文档调整这一段的写法
            "model_id": j["name"],
            "api_url": j["base_url"],
            "api_key": os.getenv(j["api_key_env"], "").strip(),
            "generation_config": {"temperature": 0.0, "max_tokens": 1024},
        },
        "contract": {"score_type": "pattern"},
    }


def run_mcq(ms):
    mcq_dir = path("mcq_dir")
    csv_file = mcq_dir / f"{CFG['mcq_subset']}_val.csv"
    if not csv_file.exists():
        print(f"[!] 没有选择题文件 {csv_file} —— 跳过（回第 3 步给题目加选项）")
        return
    for m in ms:
        print("=" * 70)
        print(f"[选择题] {m['display']} ({m['name']})")
        cfg = TaskConfig(
            model=m["name"],
            api_url=m["base_url"],
            api_key=os.getenv(m["api_key_env"], "").strip(),
            eval_type="openai_api",
            datasets=["general_mcq"],
            dataset_args={"general_mcq": {"local_path": str(mcq_dir),
                                          "subset_list": [CFG["mcq_subset"]]}},
            work_dir=eval_workdir(),
            eval_batch_size=4,
        )
        run_task(task_cfg=cfg)


def closed_files_schema_check(closed_dir) -> int:
    """schema 闸门：校验闭卷目录下的 *.jsonl 是否符合 EvalScope `general_qa` 规范。

    [!] EvalScope 把 `record['answer'] or record['response']` 当**标准答案**。
        若文件里出现 "response"（模型回答）字段，说明是旧格式 → 会把**模型旧回答当标准答案**
        （自评自己）。因此这里**直接拒绝运行**，而不是算出一份错的分数。
    [!] 抽成独立函数是为了能被**不调用模型的入口测试**直接调用（原来这段逻辑内联在 run_qa 里，
        加上本文件顶层 `from evalscope import ...`，没有 evalscope 的环境根本无法验证闸门）。
        返回校验通过的文件数；任何不合规都 raise SystemExit。
    """
    d = Path(closed_dir)
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    n = 0
    for f in files:
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                raise SystemExit(f"[X] {f.name} 第 {ln} 行不是合法 JSON（{type(e).__name__}: {e}）\n"
                                 f"    schema 闸门无法判定字段是否符合 general_qa 规范，拒绝运行。")
            if "response" in r or "answer" not in r:
                raise SystemExit(
                    f"[X] {f.name} 的字段不符合 general_qa 规范（发现 {sorted(r)}）。\n"
                    f"    EvalScope 会把 answer/response 当作**标准答案**，旧格式会让模型旧回答充当标准答案。\n"
                    f"    先运行：python _selftest\\migrate_closed_files.py（或重跑 4_rag_answer.py 重新导出）")
        n += 1
    return n


def run_qa(ms):
    """[不推荐] 闭卷问答走 EvalScope 的 general_qa。

    [!] 两个已知问题（评估这条路之前必须知道）：
      1. **字段映射**：EvalScope 的 general_qa 把 `answer = record['answer'] or record['response']` 当作
         **标准答案**。若数据文件写成 {"response": 模型回答} → 会把模型旧回答当标准答案（自评自己）。
         现已改为只写 {"question", "answer"}（见 4_rag_answer.py），并由 closed_files_schema_check 把门。
      2. **指标不合适**：general_qa 的主指标是 BLEU / Rouge（词面重叠），
         **不能衡量医学正确性**；中文医学问答应以本项目的
         「4_rag_answer（开卷/闭卷）+ 裁判判分/忠实性判定」为准。
         所以本分支默认不跑，除非你明确需要 BLEU/Rouge 这组参考数字。
    """
    closed_dir = path("qa_set").parent / "closed"
    if not closed_dir.exists() or not list(closed_dir.glob("*.jsonl")):
        print(f"[!] 没有闭卷问答文件 {closed_dir} —— 先跑 4_rag_answer.py")
        return
    n_ok = closed_files_schema_check(closed_dir)
    print(f"[OK] schema 闸门通过：{n_ok} 个闭卷文件字段合规（只含 question/answer，无 response）")
    import re
    print("[!] 注意：general_qa 用 BLEU/Rouge（词面重叠）评分，**不衡量医学正确性**；"
          "正式结论请用 4_rag_answer 的答案 + 裁判判分/忠实性。")
    for m in ms:
        slug = re.sub(r"[^0-9A-Za-z._-]", "-", m["name"]).strip("-")
        if not (closed_dir / f"{slug}.jsonl").exists():
            continue
        print("=" * 70)
        print(f"[闭卷问答 + LLM裁判] {m['display']} ({m['name']})")
        cfg = TaskConfig(
            model=m["name"],
            api_url=m["base_url"],
            api_key=os.getenv(m["api_key_env"], "").strip(),
            eval_type="openai_api",
            datasets=["general_qa"],
            dataset_args={"general_qa": {"local_path": str(closed_dir), "subset_list": [slug]}},
            judge=judge_cfg(),
            work_dir=eval_workdir(),
            eval_batch_size=4,
        )
        run_task(task_cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    add_dev_flag(ap)
    ap.add_argument("--only", default="mcq", choices=["both", "mcq", "qa"],
                    help="默认只跑 mcq（客观、无字段陷阱）。qa 分支不推荐：其主指标是 BLEU/Rouge，"
                         "不衡量医学正确性；确需时显式 --only qa")
    ap.add_argument("--model", default=None,
                    help="只评名字里含这个字符串的模型（后补一家模型时用，避免全部重跑）")
    args = ap.parse_args()
    if args.dev:
        enable_dev()

    ms = models_with_key()
    if args.model:
        ms = [m for m in ms if args.model.lower() in (m["name"] + m["display"]).lower()]
    if not ms:
        raise SystemExit("没有可用的模型（Key 未配置，或 --model 没匹配上）")
    if args.only in ("both", "mcq"):
        run_mcq(ms)
    if args.only in ("both", "qa"):
        run_qa(ms)
    print(f"\n结果在：{eval_workdir()}")
    print("下一步：python src\\7_report.py")


if __name__ == "__main__":
    main()
