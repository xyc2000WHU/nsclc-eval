# -*- coding: utf-8 -*-
"""
第 5 步：用 EvalScope 的 RAGEval(RAGAS) 后端给"开卷"回答打分。

指标含义（面试必背）：
  Faithfulness      忠实性：答案里的每句话是否都能在检索到的片段里找到依据 → 有没有编
  AnswerCorrectness 答案正确性：和标准答案在语义上是否一致
  ContextRecall     上下文召回：回答所需的信息有没有被检索出来（测你的检索器）

裁判模型（critic_llm）负责"读懂答案并打分"，所以这一步会消耗 API（比答题便宜，但也不免费）。
嵌入模型用本地的 bge-small-zh，不花钱。

用法：
  python src\5_eval_rageval.py                 # 所有模型
  python src\5_eval_rageval.py --model deepseek
"""
import argparse
import json
import os

from common import CFG, ROOT, dirpath, eval_workdir
from common import read_jsonl  # noqa: F401  (保持导入风格一致)

from evalscope.run import run_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="只评名字里含这个字符串的模型")
    ap.add_argument("--dev", action="store_true",
                    help="开发模式：只评 dev/runs 下的答题记录，产物写入 dev/")
    args = ap.parse_args()
    if args.dev:
        import common
        common.enable_dev()

    j = CFG["judge"]
    key = os.getenv(j["api_key_env"], "").strip()
    if not key:
        raise SystemExit(f"缺少裁判模型的 Key：{j['api_key_env']}")

    files = sorted(dirpath("runs").glob("*__open.json"))
    if args.model:
        files = [f for f in files if args.model.lower() in f.name.lower()]
    if not files:
        raise SystemExit("runs/ 下没有 *__open.json —— 先跑 python src\\4_rag_answer.py")

    emb_path = ROOT / CFG["embedding_local"]
    if not emb_path.exists():
        print(f"[!] 没找到本地嵌入模型 {emb_path}，会尝试联网下载（可能很慢）。")
        print("  建议先执行：modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir models/bge-small-zh")

    for f in files:
        name = f.name.replace("__open.json", "")
        n = len(json.loads(f.read_text(encoding="utf-8")))
        print("=" * 70)
        print(f"评测 {name}（{n} 条）—— 指标：{CFG['ragas_metrics']}")
        print("=" * 70)

        task_cfg = {
            "eval_backend": "RAGEval",
            "work_dir": eval_workdir(),
            "eval_config": {
                "tool": "RAGAS",
                "eval": {
                    "testset_file": str(f),
                    "critic_llm": {                       # 裁判模型：负责打分
                        "model_name": j["name"],
                        "api_base": j["base_url"],
                        "api_key": key,
                    },
                    "embeddings": {                       # 嵌入模型：负责算语义相似
                        "model_name_or_path": str(emb_path) if emb_path.exists()
                        else CFG["embedding_hub"],
                    },
                    "metrics": CFG["ragas_metrics"],
                    "language": "chinese",
                    "batch_size": int(CFG.get("ragas_batch_size", 4)),
                },
            },
        }
        run_task(task_cfg=task_cfg)
        print(f"[OK] {name} 完成\n")

    print(f"结果在：{eval_workdir()}")
    print("下一步：python src\\6_eval_evalscope.py（跑选择题 + 闭卷问答）")


if __name__ == "__main__":
    main()
