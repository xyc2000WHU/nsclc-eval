# -*- coding: utf-8 -*-
r"""
出题辅助工具：在知识库里找片段、看全文、生成出题表模板。

自己出题时你会需要三件事，这个脚本一次给全：
  1) 找素材：按关键词搜知识库，拿到**片段ID**（出题表要填「来源片段」）
  2) 看全文：打印某个片段的完整原文（方便你**逐字**抄「原文证据句」）
  3) 模板  ：生成 qa\出题表_模板.csv（含 3 条真实示例，字段与流水线完全对齐）
             [!] 绝不覆盖真实出题表 qa\审核表.csv：默认路径已存在就**拒绝**，
                 显式 --template-out 指向已有文件时也会先 shutil.copy2 备份。

用法：
  python src\2a_find_chunk.py 巩固治疗 durvalumab          # 按关键词搜
  python src\2a_find_chunk.py 免疫性心肌炎                 # 查某个主题库里有不有（0 命中=可做陷阱题）
  python src\2a_find_chunk.py --show D02-C124            # 看某片段全文
  python src\2a_find_chunk.py --template                 # 生成出题表模板
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from common import CFG, ROOT, kb_fingerprint, path, read_jsonl, ref_score


def load():
    return read_jsonl(path("chunks"))


def search(chunks, keys, limit=8):
    for k in keys:
        kl = k.lower()
        hits = [c for c in chunks if kl in c["text"].lower()]
        print("=" * 92)
        print(f"关键词「{k}」：命中 {len(hits)} 块"
              + ("   ← 库里没有，可考虑做成陷阱题" if not hits else ""))
        for c in hits[:limit]:
            i = c["text"].lower().find(kl)
            seg = c["text"][max(0, i - 70): i + 130].replace("\n", " ")
            print(f"  [{c['chunk_id']}] {c['file'][:40]:<42} 文献条目得分 {ref_score(c['text'])}")
            print(f"      …{seg}…")
        if len(hits) > limit:
            print(f"  …… 还有 {len(hits) - limit} 块，用更具体的关键词缩小范围")


def show(chunks, cid):
    for c in chunks:
        if c["chunk_id"] == cid:
            print(f"[{c['chunk_id']}] {c['file']}（{len(c['text'])} 字，"
                  f"文献条目得分 {ref_score(c['text'])}）\n")
            print(c["text"])
            print("\n提示：「原文证据句」必须从上面**逐字复制**（英文原句，不翻译、不改写）。")
            return
    sys.exit(f"知识库里有没 {cid}。用关键词搜一下确认片段ID。")


def make_template(chunks, out=None, explicit_out=False):
    """用真实片段生成 3 条示例行，字段与 2_gen_qa/3_finalize_qa 完全对齐。

    [!] 为什么**不再**默认写到 `qa/审核表.csv`：那正是你手写/被流水线写过的**真实出题表**，
        旧实现直接覆盖、无备份 → 误跑一次 `--template` 就等于删掉全部手工内容（不可恢复）。
        现在的安全约定：
          · 默认落 `qa/出题表_模板.csv`（另一个文件），且**目标已存在就拒绝**；
          · 若你**显式**用 --template-out 指向已有文件（例如要覆盖 qa/审核表.csv），
            则先 `shutil.copy2` 备份（带时间戳）再写。

    [!] 两道防线都必须在**真正写文件之前**生效：旧写法把"已存在就拒绝"放在写之前、
        把"先备份"放在写之前但它前面已经 return/exit 了 → 备份分支永远走不到。
    """
    cols = ["保留(1=要,0=删)", "自动校验", "陷阱验证", "医学正确性(填:对/错/需改)",
            "题目ID", "题型", "答案能否从文献得到", "问题", "标准答案", "原文证据句",
            "A", "B", "C", "D", "正确答案", "来源片段", "来源文献",
            "修改后的问题(留空=不改)", "修改后的答案(留空=不改)", "备注(为什么删/要改)"]

    def pick(must, avoid=()):
        for c in chunks:
            t = c["text"]
            if all(m.lower() in t.lower() for m in must) and \
               not any(a.lower() in t.lower() for a in avoid) and ref_score(t) < 6:
                return c
        return None

    def quote(c, key):
        """从片段里**切片**出一句含关键词的英文证据句——必须逐字，不能拼接（拼接会改掉原文）。"""
        t = c["text"]
        i = t.lower().find(key.lower())
        if i < 0:
            i = 0
        start = t.rfind(". ", 0, max(0, i)) + 2
        end = t.find(". ", i)
        seg = t[start:end + 1] if end > 0 else t[start:start + 300]
        return seg.strip()

    rows = []
    c1 = pick(["alectinib", "hazard ratio", "disease-free"])
    if c1:
        rows.append({
            "保留(1=要,0=删)": "1", "自动校验": "", "陷阱验证": "",
            "医学正确性(填:对/错/需改)": "（填 对/错/需改）", "题目ID": "T001", "题型": "number",
            "答案能否从文献得到": "是",
            "问题": "【示例，请删除】ALK 阳性的完全切除 NSCLC 患者，术后辅助阿来替尼相比含铂化疗的无病生存获益如何？",
            "标准答案": "【示例】辅助阿来替尼组的无病生存显著优于含铂化疗；确切风险比请按右列证据句核对后替换本行文字。",
            "原文证据句": quote(c1, "hazard ratio"),
            "A": "辅助阿来替尼", "B": "含铂化疗", "C": "仅随访观察",
            "D": "单纯局部放疗后随访观察", "正确答案": "A",
            "来源片段": c1["chunk_id"], "来源文献": c1["file"],
            "修改后的问题(留空=不改)": "", "修改后的答案(留空=不改)": "",
            "备注(为什么删/要改)": "示例行：确认无误后请删除",
        })
    c2 = pick(["consolidation", "durvalumab"])
    if c2:
        rows.append({
            "保留(1=要,0=删)": "1", "自动校验": "", "陷阱验证": "",
            "医学正确性(填:对/错/需改)": "（填 对/错/需改）", "题目ID": "T002", "题型": "guideline",
            "答案能否从文献得到": "是",
            "问题": "【示例，请删除】不可手术的 III 期 NSCLC，同步放化疗后推荐什么巩固治疗？",
            "标准答案": "【示例】同步放化疗后推荐度伐利尤单抗（durvalumab）巩固治疗。",
            "原文证据句": quote(c2, "durvalumab"),
            "A": "度伐利尤单抗", "B": "继续诱导化疗", "C": "立即手术",
            "D": "继续诱导化疗并加用胸腔放疗", "正确答案": "A",
            "来源片段": c2["chunk_id"], "来源文献": c2["file"],
            "修改后的问题(留空=不改)": "", "修改后的答案(留空=不改)": "",
            "备注(为什么删/要改)": "示例行：确认无误后请删除",
        })
    rows.append({
        "保留(1=要,0=删)": "1", "自动校验": "", "陷阱验证": "",
        "医学正确性(填:对/错/需改)": "（陷阱题不用填这列）", "题目ID": "T003", "题型": "trap",
        "答案能否从文献得到": "否",
        "问题": "【示例，请删除】PD-1 抑制剂治疗期间出现 3 级免疫性心肌炎，具体的处理流程和激素剂量怎么定？",
        "标准答案": "【示例】本题为陷阱题：这 10 篇文献只覆盖免疫性肺炎，未覆盖免疫性心肌炎的处理流程，"
                    "正确答案应是「文献不足以回答」。",
        "原文证据句": "", "A": "", "B": "", "C": "", "D": "", "正确答案": "",
        "来源片段": "", "来源文献": "",
        "修改后的问题(留空=不改)": "", "修改后的答案(留空=不改)": "",
        "备注(为什么删/要改)": "示例行：陷阱题必须跑 2c_verify_traps.py 生成验证标记才会入库",
    })

    # [!] 默认落到独立的模板文件，绝不覆盖真实出题表 qa/审核表.csv
    if out is None:
        out = path("qa_set").parent / "出题表_模板.csv"
    out = Path(out)
    if out.exists():
        if not explicit_out:
            # 默认路径已存在 → 拒绝而不是静默覆盖（用户可能已经往模板里填了题）
            sys.exit(f"[X] {out} 已存在，拒绝覆盖。\n"
                     f"    如需重新生成：先改名/删除该文件，或用 --template-out 指定另一个路径。")
        # 显式指定的路径也先备份：它可能指向真实出题表，覆盖前必须留档（带时间戳，可重复运行）
        import shutil
        import time
        bak = out.parent / (out.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(out, bak)
        print(f"[i] 已先备份原文件 → {bak.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] 已生成出题表模板：{out}")
    print(f"   含 {len(rows)} 条示例（请按示例格式填写后删掉示例行）")
    print("   字段说明见 出题指南.md；填完后跑 python src\\2b_check_questions.py 自检")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keywords", nargs="*", help="要搜的关键词")
    ap.add_argument("--show", help="打印某个片段的全文（用于抄证据句）")
    ap.add_argument("--template", action="store_true", help="生成出题表模板")
    ap.add_argument("--template-out", default=None,
                    help="模板输出路径（默认 qa\\出题表_模板.csv；已存在则拒绝覆盖）")
    args = ap.parse_args()

    chunks = load()
    print(f"知识库 {len(chunks)} 块｜指纹 kb-{kb_fingerprint()}｜top_k={CFG['top_k']}\n")
    if args.template:
        make_template(chunks, out=args.template_out, explicit_out=bool(args.template_out))
    elif args.show:
        show(chunks, args.show)
    elif args.keywords:
        search(chunks, args.keywords)
    else:
        # 默认给一份"可出题素材分布"，帮助决定题目配比
        per = Counter(c["doc_id"] for c in chunks)
        print("各文献片段数（决定每题能从哪篇取材）：")
        for d in sorted(per):
            f = next(c["file"] for c in chunks if c["doc_id"] == d)
            print(f"  {d} {f[:44]:<46} {per[d]:>4} 块")
        print("\n用法：给个关键词搜素材，例如 python src\\2a_find_chunk.py 巩固治疗")


if __name__ == "__main__":
    main()
