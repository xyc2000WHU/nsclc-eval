# -*- coding: utf-8 -*-
r"""
陷阱题验证（人工确认版，不用裁判模型）。

陷阱题测的是"模型该不该承认文献答不了"，所以必须证明"这些文献确实答不了"。
这个脚本会把该问题在知识库里做一次**比答题更宽的检索**（默认 top-18，答题只用 top-6），
把命中的片段逐条摆给你看，由**你**判断有没有哪条其实能回答：
  · 你确认都答不了 → 写入验证标记 `已验证(kb-<指纹>,top-<宽度>,<问题短哈希>)`
  · 知识库一改或问题文本一改，标记自动失效（3_finalize_qa 会拦下，重跑本脚本即可）

用法：
  python src\2c_verify_traps.py              # 逐题交互确认（推荐）
  python src\2c_verify_traps.py --show-only  # 只看检索结果，不写标记
  python src\2c_verify_traps.py --yes        # 全部直接标记（仅当你已逐题确认过、只是刷新标记时用）
"""
import argparse
import csv
import hashlib
import shutil
import sys
import time

from common import CFG, ROOT, kb_fingerprint, path, read_csv_any, retrieve


def qhash(q: str) -> str:
    """必须与 2_gen_qa.py / 3_finalize_qa.py 完全一致，否则标记校验不过。"""
    return hashlib.md5((q or "").strip().encode("utf-8")).hexdigest()[:8]


def g(row, *names):
    for k in row:
        if k and k.strip() in names:
            return (row[k] or "").strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-only", action="store_true", help="只打印检索结果，不写标记")
    ap.add_argument("--yes", action="store_true", help="不逐题询问，直接写标记")
    args = ap.parse_args()

    src = path("review_csv")
    if not src.exists():
        sys.exit(f"找不到 {src}")
    # [!] 统一走 common.read_csv_any（多编码兜底）：出题表可能是 Excel 存的 GBK。
    rows = read_csv_any(src)
    if not rows:
        sys.exit(f"出题表读不出来或为空：{src}")
    cols = list(rows[0].keys())

    verify_k = max(int(CFG["top_k"]) * 3, 12)
    fp = kb_fingerprint()
    traps = [r for r in rows if g(r, "答案能否从文献得到") == "否"]
    print(f"出题表 {len(rows)} 行，其中陷阱题 {len(traps)} 道")
    print(f"知识库指纹 kb-{fp}｜验证检索宽度 top-{verify_k}（答题只用 top-{CFG['top_k']}）\n")
    if not traps:
        print("没有陷阱题（「答案能否从文献得到」= 否 的行）。")
        return

    changed = 0
    for i, r in enumerate(traps, 1):
        q = g(r, "修改后的问题(留空=不改)", "修改后的问题") or g(r, "问题")
        qid = g(r, "题目ID") or f"第{i}道"
        old = g(r, "陷阱验证")
        print("=" * 92)
        print(f"[{i}/{len(traps)}] {qid}：{q}")
        if old:
            print(f"  已有标记：{old}" + ("（问题文本已改 → 原验证失效）"
                                        if f",{qhash(q)})" not in old else "（仍然有效）"))
        hits = retrieve(q, top_k=verify_k)
        print(f"  下面这 {len(hits)} 个片段是知识库里与该问题最相关的内容：")
        for j, h in enumerate(hits, 1):
            print(f"    {j:>2}. {h['chunk_id']:<12} {h['doc_id']} {h['file'][:34]:<36}")
            print(f"        {h['text'][:150].replace(chr(10), ' ')}……")
        if args.show_only:
            continue

        if args.yes:
            ok = True
        else:
            ans = input("  → 这些片段**都无法回答**该问题吗？(y=确认不可答 / n=跳过，稍后改题): ")
            ok = ans.strip().lower() in ("y", "yes", "是")
        if not ok:
            print("  已跳过：请改题（或用 2a_find_chunk.py 确认库里确实没有该内容）后重跑。")
            continue
        r["陷阱验证"] = f"已验证(kb-{fp},top-{verify_k},{qhash(q)})"
        changed += 1
        print(f"  [OK] 已记录验证：{r['陷阱验证']}")

    if args.show_only:
        print("\n（--show-only：未写入任何标记）")
        return
    if not changed:
        print("\n没有写入任何标记。")
        return

    bak = src.with_suffix(src.suffix + f".bak-{time.strftime('%H%M%S')}")
    shutil.copy2(src, bak)
    with open(src, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\n[OK] 已写入 {changed} 道陷阱题的验证标记 → {src.name}")
    print(f"   原文件已备份：{bak.name}")
    print("   下一步：python src\\2b_check_questions.py → python src\\3_finalize_qa.py")


if __name__ == "__main__":
    main()
