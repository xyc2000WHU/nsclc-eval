# -*- coding: utf-8 -*-
r"""
上下文充分性检查：按正式流程检索 top-k 后，让裁判模型判断"这些片段是否足以回答该问题"。

为什么需要：小样的"金标准片段进 top-6"只有 50%，但那是**按片段 ID 严格匹配**的下界——
相邻片段或另一篇文献里可能有同样能回答问题的内容。这个检查直接回答真正重要的问题：
"模型拿到的上下文够不够用"，从而判断 50% 是"检索失败"还是"指标过严"。

[!] 判定解析口径 = **严格**（2026-10-12 修，第三方复验补充检查 A2；同日补「JSON 形式」兼容 = 收尾点 3）：
    旧实现是"字符串包含 `结论：是` 判真、其余非空输出判假"，实测有三个问题：
      · `结论：是` 与 `结论：否` **同时出现** → 算成"是"（冲突被当成充分）；
      · **没有任何结论字段**的普通文本 → 算成"否"（格式失败被当成"证据不足"，两者含义完全不同）；
      · `结论: 是`（**英文冒号**）→ 算成"否"（格式差异改变了判定）。
    现在：只认带字段名（结论 / 判定 / judge / verdict）的标签，取全部匹配后**全部一致才接受**，
    **冲突 / 缺失 / 空输出一律记 `解析失败`**（不塞进"否"），并保存裁判**原文**；
    **JSON 形式**（`{"结论":"是"}` / `{"verdict": "no"}`）同样接受，英文值 yes/no 内部归一为「是/否」
    （**不接受 true/false**）；JSON 与文本互相矛盾仍记「解析失败」，绝不猜。
    汇总同时报「有效 / 是 / 否 / 解析失败」四个数；**正式全量有失败时非零退出**（清楚标注不完整），
    重跑时失败项自然会被重判。同时留档裁判 token 用量与**判定材料指纹**（题面+片段+提示词版本+裁判名）。

用法：python _selftest\check_context_sufficiency.py [--dev]   # 默认检查正式集 qa/qa_set.jsonl
产物：outputs\上下文充分性_原始判定.jsonl（逐条原文 + 指纹 + 用量）
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import CFG, chat, dirpath, full_question, path, read_jsonl, write_jsonl   # noqa: E402

PROMPT = (Path(__file__).resolve().parent / "sufficiency_prompt.txt").read_text(encoding="utf-8")
MAXTOK = 1500   # 推理型裁判需要足够额度，否则返回空内容（曾因此得出错误的"充分率 10%"）
PROMPT_VER = "strict-v2"
LABELS = ("是", "否", "解析失败")

# [!] 只认**带字段名**的标签：`结论：是` / `结论: 是` / `判定：否` / `"结论": "是"` / `{"judge":"yes"}` 都接受
#     （全角/半角冒号、常见引号、**JSON 形式**（收尾点 3，2026-10-12 修）、英文值 yes/no 都容错
#     —— 格式差异不该改变判定），但"是否…"这类词不算标签
#     （用 (?![\u4e00-\u9fffA-Za-z]) 排除；英文 yes/no 后面也不能再跟字母，避免 "no" 命中 "none"）。
#     英文值只认 yes/no（**不接受 true/false**：那是另一套语义，不猜），并在内部归一为「是/否」。
LABELS = ("是", "否", "解析失败")
_Q = "\"'\u201c\u201d\u300c\u300d"                     # 容忍的引号集合（含 JSON 双引号、中文引号）
RE_FIELD = re.compile(
    rf"[{_Q}]?\s*(?:结论|判定|judge|verdict)\s*[{_Q}]?\s*[:：]\s*"
    rf"[{_Q}]?\s*(是|否|yes|no)(?![\u4e00-\u9fffA-Za-z])",
    re.IGNORECASE)
# 值归一：英文 yes/no → 是/否（其余原样，不会出现在这里）
_NORM_VAL = {"是": "是", "否": "否", "YES": "是", "NO": "否"}


def norm_verdict(v: str) -> str:
    """把解析到的值归一为「是/否」。"""
    return _NORM_VAL.get((v or "").upper(), v)


def parse_verdict(out):
    """严格解析裁判输出。返回 (判定, 理由, 解析方式)。

    判定 ∈ {"是", "否", "解析失败"}；**冲突或缺失一律"解析失败"，绝不猜**。
    支持的形式：`结论：是`、`判定: 否`、`judge: yes`、`{"结论":"是"}`、`{"verdict": "no"}`
    —— 英文 yes/no 归一为 是/否；同一条里既有 是 又有 否（含 JSON 与文本互相矛盾）→ 解析失败。
    """
    text = out or ""
    reason = ""
    for line in text.splitlines():
        s = line.strip().lstrip("*#").strip()
        if s.startswith("理由"):
            reason = s[2:].lstrip("：: 　").strip()
            break
    found = RE_FIELD.findall(text)
    uniq = sorted({norm_verdict(v) for v in found})
    if len(uniq) == 1:
        return uniq[0], reason, f"字段唯一({len(found)}处)"
    if uniq:
        return "解析失败", reason, f"字段冲突({uniq})"
    if not text.strip():
        return "解析失败", reason, "空输出"
    return "解析失败", reason, "无结论字段"


def material_fp(question, contexts, judge_name) -> str:
    """判定材料指纹：题面（含选项）+ 全部片段正文 + 提示词版本 + 裁判名。

    [!] 与 5b/7c 同一个思路：判定必须绑定**它当时看到的那份材料**。
        片段正文、题面或裁判任一变化，指纹就变，旧判定就不能再代表当前材料。
    """
    raw = json.dumps({"v": PROMPT_VER, "judge": judge_name, "prompt_ver": PROMPT_VER,
                      "question": question, "contexts": contexts},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="检查 dev 集（默认检查正式集）")
    a = ap.parse_args()
    if a.dev:
        import common
        common.enable_dev()
    qa = read_jsonl(path("qa_set"))
    m = CFG["judge"]
    yes = no = failed = 0
    records = []
    print(f"逐题检查上下文充分性（{m['name']}，top_k={CFG['top_k']}）\n")
    for it in qa:
        hits = retrieve_safe(it["question"])
        contexts = [h["text"][:900] for h in hits]
        ctx = "\n\n".join(f"[{h['chunk_id']}] {t}" for h, t in zip(hits, contexts))
        q_txt = full_question(it)
        out, usage = chat(m, [{"role": "user", "content": PROMPT.format(
            question=q_txt, contexts=ctx)}], max_tokens=MAXTOK)
        verdict, reason, parse_mode = parse_verdict(out)
        if verdict == "是":
            yes += 1
        elif verdict == "否":
            no += 1
        else:
            failed += 1
            print(f"{it['question_id']}｜[!] 判定**解析失败**（{parse_mode}）"
                  f"→ 计入「解析失败」、不进是/否；原文已留档（用量 {usage}）")
        gold_in = it.get("source_chunk") in [h["chunk_id"] for h in hits]
        # [!] 日志行格式保持不变（`金标准片段在 top-N: 是/否｜上下文充分: 是/否`）：
        #     `_selftest/split_sufficiency.py` 与既有留档都按这个格式读；
        #     解析失败时写 `解析失败`（该脚本会把它算作"排除出分母"，与这里的口径一致）。
        print(f"{it['question_id']}｜金标准片段在 top-{CFG['top_k']}: {'是' if gold_in else '否'}"
              f"｜上下文充分: {verdict}｜{reason[:60]}")
        records.append({"question_id": it["question_id"], "model": m["name"],
                        "gold_in_topk": gold_in, "verdict": verdict, "reason": reason,
                        "parse_mode": parse_mode,
                        "raw": (out or "")[:2000],
                        "fp": material_fp(q_txt, contexts, m["name"]),
                        "in_tokens": usage.get("in_tokens", 0),
                        "out_tokens": usage.get("out_tokens", 0),
                        "attempts": usage.get("attempts")})
    valid = yes + no
    n = valid + failed
    rate = (yes / valid * 100) if valid else 0.0
    print(f"\n上下文充分率（严格解析）：{yes}/{valid} = {rate:.0f}%"
          f"｜有效 {valid}｜是 {yes}｜否 {no}｜**解析失败 {failed}**（总 {n} 条）")
    out_file = dirpath("outputs") / "上下文充分性_原始判定.jsonl"
    write_jsonl(out_file, records)
    print(f"[i] 逐条原文 + 判定材料指纹 + 用量 → {out_file}")
    if failed:
        print(f"[X] 有 {failed} 条**解析失败** → 本次统计**不完整**（既不等于「否」也不等于「是」）。"
              f"原文已留档；重跑本脚本会重判这些条目（不会把失败当结论）。")
        return 2
    return 0


def retrieve_safe(q):
    """检索（单独抽一层：`--dev` 与测试里可以替换，避免误触真实向量库）。"""
    from common import retrieve
    return retrieve(q)


if __name__ == "__main__":
    sys.exit(main())
