# -*- coding: utf-8 -*-
r"""
出题表自检：把「格式错误」和「会被入库门槛拦下的行」在**出题阶段**就找出来，而不是等你写满 80 题。

检查项（硬错误 = 退出码 1）：
  · 必填字段为空（问题/标准答案/来源片段/原文证据句）
  · 来源片段在知识库里不存在
  · **原文证据句不是该片段里的逐字文本**（最关键：gold 命中率、忠实性判定都依赖它）
  · 选择题选项不成套、正确答案不在 A~D
  · 重复题目（同一问题文本出现两次）
  · 普通题「保留」未填 1 或「医学正确性」未填「对」→ 会被 3_finalize_qa 拦下（这里提前报警）
  · 陷阱题缺「陷阱验证」标记 → 提示先跑 2c_verify_traps.py

体检项（只提示，不拦）：
  · 题型分布、选择题数量（建议 ≥30）、正确答案位置分布（防止"答案总在 A"）
  · 正确选项是否总是最长（常见的出题偏差）
  · 问题是否大多为中文（本项目是"中文提问 → 英文文献"）

用法：python src\2b_check_questions.py
"""
import csv
import sys
from collections import Counter

from common import CFG, g_col, kb_fingerprint, path, read_csv_any, read_jsonl

KEEP_TRUE = {"1", "1.0", "是", "y", "yes", "true", "t", "保留"}
TYPES = {"fact", "number", "guideline", "reasoning", "trap"}


def g(row, *names):
    """按列名取值 → 见 common.g_col（容忍 Excel 改过列名/多余空格）。"""
    return g_col(row, *names)


def main():
    src = path("review_csv")
    if not src.exists():
        sys.exit(f"找不到 {src} —— 先跑 python src\\2a_find_chunk.py --template 生成模板")
    # [!] 统一走 common.read_csv_any：出题表是用户在 Excel 里存的，编码可能是 GBK
    #     （Excel 默认"CSV(逗号分隔)"），只认 utf-8-sig 必崩。
    rows = read_csv_any(src)
    if not rows:
        sys.exit(f"出题表读不出来或为空：{src}\n"
                 f"    请用 Excel 另存为 UTF-8 或 CSV UTF-8 后重试。")

    chunks = {c["chunk_id"]: c for c in read_jsonl(path("chunks"))}
    print(f"出题表：{src.name}｜{len(rows)} 行｜知识库 {len(chunks)} 块｜kb-{kb_fingerprint()}\n")

    errors, warns = [], []
    seen, types, sources, letters = {}, Counter(), Counter(), Counter()
    mcq_n = longest_right = 0

    for n, r in enumerate(rows, 1):
        qid = g(r, "题目ID") or f"第{n}行"
        q = g(r, "修改后的问题(留空=不改)", "修改后的问题") or g(r, "问题")
        a = g(r, "修改后的答案(留空=不改)", "修改后的答案") or g(r, "标准答案")
        ev = g(r, "原文证据句")
        sc = g(r, "来源片段")
        typ = g(r, "题型") or "fact"
        is_trap = g(r, "答案能否从文献得到") == "否"
        keep = g(r, "保留(1=要,0=删)", "保留")
        med = g(r, "医学正确性(填:对/错/需改)", "医学正确性")
        marker = g(r, "陷阱验证")

        if not q:
            errors.append((qid, "问题为空"))
            continue
        if q in seen:
            errors.append((qid, f"问题与 {seen[q]} 重复"))
        seen[q] = qid
        if not a:
            errors.append((qid, "标准答案为空"))
        if typ not in TYPES:
            errors.append((qid, f"题型「{typ}」不在 {sorted(TYPES)} 内"))

        if is_trap:
            types["trap"] += 1
            if not marker:
                errors.append((qid, "陷阱题缺「陷阱验证」标记 → 先跑 "
                                    "python src\\2c_verify_traps.py"))
            if any(g(r, k) for k in "ABCD"):
                warns.append((qid, "陷阱题给了选项（一般不该给：测的是「该说不知道」）"))
        else:
            types[typ] += 1
            if keep not in KEEP_TRUE:
                errors.append((qid, f"「保留」= 「{keep or '空'}」→ 会被 3_finalize_qa 丢掉"))
            if med != "对":
                errors.append((qid, f"「医学正确性」= 「{med or '空'}」→ 会被入库门槛拦下"))
            if not ev:
                errors.append((qid, "普通题缺「原文证据句」"))
            if not sc:
                errors.append((qid, "普通题缺「来源片段」"))
            elif sc not in chunks:
                errors.append((qid, f"来源片段 {sc} 在知识库里不存在（用 2a_find_chunk.py 核对）"))
            elif ev and ev not in chunks[sc]["text"]:
                errors.append((qid, f"原文证据句不是 {sc} 里的逐字文本"
                                    f"（可能翻译过/改写过/多空格）→ gold 命中与忠实性会判错"))
            if sc in chunks:
                sources[chunks[sc]["doc_id"]] += 1

        opts = {k: g(r, k) for k in "ABCD"}
        filled = [k for k, v in opts.items() if v]
        ans = g(r, "正确答案")
        if filled and len(filled) != 4:
            errors.append((qid, f"选项只填了 {filled}，要么 4 个都填、要么都不填"))
        if len(filled) == 4:
            mcq_n += 1
            if ans not in "ABCD" or not ans:
                errors.append((qid, f"正确答案「{ans}」不是 A/B/C/D 之一"))
            else:
                letters[ans] += 1
                if len(opts[ans]) == max(len(v) for v in opts.values()):
                    longest_right += 1
        elif ans:
            warns.append((qid, "没有选项却填了「正确答案」"))

        if q and sum(1 for ch in q if ch.isascii() and ch.isalpha()) > len(q) * 0.6:
            warns.append((qid, "问题看起来是英文（本项目设计为中文提问 → 英文文献）"))

    print(f"可答题 {sum(v for k, v in types.items() if k != 'trap')} 道｜陷阱题 {types['trap']} 道"
          f"｜选择题 {mcq_n} 道")
    print("  题型分布：", dict(types))
    print("  来源文献分布：", dict(sorted(sources.items())))
    if letters:
        print("  正确答案分布：", dict(letters),
              "（尽量均衡；若集中在某一个字母，说明有位置偏差）")
        if longest_right:
            print(f"  正确选项是最长选项的比例：{longest_right}/{mcq_n}"
                  f"（{longest_right / mcq_n:.0%}，超过 40% 就是明显的出题偏差）")

    if mcq_n < 30:
        warns.append(("整体", f"选择题只有 {mcq_n} 道，建议 ≥30 道（选择题指标最稳，不依赖裁判模型）"))
    if not sources and types["trap"] != sum(types.values()):
        warns.append(("整体", "没有一道题登记了来源文献 → 检查「来源片段」是否填了"))

    if warns:
        print(f"\n[!] {len(warns)} 条提示（不拦，但建议处理）：")
        for qid, w in warns[:25]:
            print(f"    {qid}: {w}")
    if errors:
        print(f"\n[X] {len(errors)} 条**硬错误**（必须改，否则这些题进不了评测集）：")
        for qid, e in errors[:40]:
            print(f"    {qid}: {e}")
        if len(errors) > 40:
            print(f"    …… 其余 {len(errors) - 40} 条同理")
        print("\n改完重跑本脚本；全绿后再跑 python src\\3_finalize_qa.py")
        sys.exit(1)

    print(f"\n[OK] 自检通过：{len(rows)} 行无硬错误")
    print("下一步：python src\\3_finalize_qa.py（生成正式评测集 + 选择题子集）")


if __name__ == "__main__":
    main()
