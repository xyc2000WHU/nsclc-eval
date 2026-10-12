# 上传 GitHub

此目录是独立发布副本，没有沿用原项目 .git 历史。先完成 key 轮换，再公开仓库；本副本未携带 .env，但打包不能撤销此前已暴露的凭据。

## 本地最后检查

```bash
python scripts/verify_public_release.py
git init
git add .
git status --short
git diff --cached --stat
```

`verify_public_release.py` 应当打印 `PASS` 并以退出码 0 结束；它会逐项复算基础标签指标、人工裁定分布（56/10/9，P0 16/2/0）、裁定后 450 条分布（334/19/97）、"谁更准"（一号 30/75 = 40.0%、二号 v3 44/75 = 58.7%，仅描述性）、人工覆盖关系（16 / 149 / 74 / 301）与成因 18/18。**它不做显著性判读**：配对结构与同题相关性未处理，区间重叠不能当作"差异不显著"。任何数据不在包内的项目会明确打印"未复算"，不会静默通过。

确认待提交列表只有此公开版本：不应有 .env、PDF、kb/、models/、runs/、个人审计日志或备份。不要在原始实验目录执行这些发布命令。

本快照已是 `post-adjudication`。请**不要**把 `data/release_status.json` 改回待裁定状态，也不要在文档里写成"75 条待裁定"；同日期（2026-10-12）的 pre-adjudication 目录与 zip 是历史快照，保留但不要一并上传。

## 新建空仓库并推送

在 GitHub 创建空仓库，建议名称 `nsclc-eval`，描述可用：

> Evidence-grounded NSCLC QA evaluation: 5 models, open/closed-book experiments, dual AI judges and human spot checks.

建议 topics：`medical-ai`, `llm-evaluation`, `rag`, `nsclc`, `llm-as-a-judge`。

```bash
git commit -m "Prepare NSCLC evaluation public snapshot"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

本次整理没有替你创建仓库、提交或推送，也没有检查/清洗原项目 Git 历史。请从此副本创建全新历史。暂未选择许可证；如需开放再使用权限，再单独确定代码和数据的许可范围。

## 人工裁定已完成：如何继续

75/75 已裁定完毕，`status` 为 `post-adjudication`、`adjudication_pending` 为 0。如后续题库、知识库或输入指纹发生变化，需另加带日期与输入版本的裁定文件，并明确哪些旧判定失效；不要覆盖原始裁判标签。

改动任何数据或文档后，先审阅差异，再显式重新生成文件指纹：

```bash
python scripts/build_manifest.py
python scripts/verify_public_release.py
```

再打包时，zip 内应包含 `nsclc-eval/` 根、`上传前阅读.md` 与 `发布检查结果.json`；不要重新打包上一版 pre-adjudication 内容。
