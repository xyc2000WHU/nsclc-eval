# 复现说明

## 1. 已公开结果的离线复算

Python 3.10+，不安装第三方包：

```bash
python scripts/verify_public_release.py
```

所有输入来自 data/，不访问网络、不读 .env、不重跑裁判。成功输出为公开标签与元数据的复算，不能替代原始证据的医学复核。

该脚本复算：90 题/900 条作答/450 条判定的结构与去重、文件指纹、两位裁判的正确性与忠实性一致率、人工抽检样本内一致率（裁定前快照 82/90）、检索与上下文指标，以及裁定层——人工裁定分布（56/10/9，P0 16/2/0）、裁定后 450 条最终分布（334/19/97，375 条为两裁判一致的推定值）、"谁更准"（一号 30/75 = 40.0%、二号 v3 44/75 = 58.7% 的描述性命中数与比率）、人工覆盖关系（16 / 149 / 74 / 301）与成因裁定（18/18 在检索/上下文侧）。**它只做描述性复算，不做显著性判读**：两位裁判评同一批回答（配对）且同题回答相关，样本又按分歧筛选，因此不能用区间重叠之类的直观比较推断差异是否显著（见 RESULTS.md 脚注）。包内不存在的数据项会打印"未复算"并说明原因，不会当成通过。

## 2. 流程回归

```bash
python _selftest/regression_test.py --public
```

此测试使用模拟 API 和临时数据。公开版不含私有生产产物，针对缺失生产产物的检查会明确跳过，项数可能与内部 130 项不同。需要 Git 才能执行实际忽略规则测试。本地已运行结果见 RELEASE_NOTES.md。生成的 _selftest/sandbox/ 不纳入发布快照。

## 3. 重新运行实验

以下操作需要自行准备文献和模型 API，并可能产生费用。请在新的工作副本中执行。

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

把 .env.example 复制为 .env，填入自己新签发的 key。config.yaml 是当时的配置，包含备选模型；原实验实际有输出的 5 个模型以 data/answer_metadata.jsonl 为准。请核对服务端模型 ID、接入路径和价格；requirements-historical.txt 仅记录当时环境，并非跨平台安装保证。

1. 自行取得所需文献，按 data/source_documents.json 的本地文件名放入 pdfs/。
2. 运行 `python src/1a_get_embedding_model.py` 下载嵌入模型，再运行 `python src/1_build_kb.py` 建库。
3. 用 `python src/2a_find_chunk.py --template` 建立审核表，按原始文献、题库与实际重建片段填写证据和审核；不要把 data/questions.jsonl 直接当成完整 qa/qa_set.jsonl。
4. 按 `2b_check_questions.py`、`2c_verify_traps.py`、`3_finalize_qa.py` 完成自检、陷阱核验和正式入库。原陷阱标记不能直接迁移到新库。
5. 运行 `python src/4_rag_answer.py --limit 3` 先试跑，再决定是否全量。该步骤会调用 API。
6. `5b_judge_faithfulness.py`、`7c_export_full_review.py` 用于裁判；`7_report.py` 生成本地工作报告。发布报告应保持两批实验和判定版本分开。
7. EvalScope 为可选扩展：安装 requirements-rag.txt 后，使用 `python src/6_eval_evalscope.py --only mcq`。general_qa 的 BLEU/ROUGE 不代表医学正确性。

精选 _selftest 比较工具需要完整私有回答、判定缓存和快照；public data/ 的无正文元数据不满足这些输入要求。旧 verify_dual_judge 复用同一正式比较实现，是另一调用入口，不能称为独立算法交叉验证。
