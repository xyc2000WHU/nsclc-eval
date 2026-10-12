# -*- coding: utf-8 -*-
"""
检查 pdfs/ 里每篇 PDF 的"文字层质量"——在建知识库之前做的一道闸门。

为什么要这一步：
  扫描件（图片型 PDF）或 CID 编码的 PDF 抽出来是乱码/空白，会让知识库和题目全是垃圾，
  而且等到出题、答题、评测时才发现，返工成本极高。

判断标准（脚本会给出结论）：
  · 中文字符占比、总字符数、每页字符数
  · 抽取结果里是否出现大量替换符/控制符
  · 建议：字符数 < 500 或每页 < 100 → 该篇基本不可用，换文献

用法：python src\1b_check_pdfs.py
"""
import sys
from pathlib import Path

from common import CFG, ROOT

MIN_CHARS = 500          # 整篇最少字符数
MIN_PER_PAGE = 100       # 平均每页最少字符数


def extract(p):
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        with fitz.open(str(p)) as doc:
            pages = [pg.get_text() for pg in doc]
        return len(pages), "".join(pages), "PyMuPDF"
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return len(reader.pages), "".join((pg.extract_text() or "") for pg in reader.pages), "pypdf"


def main():
    pdf_dir = ROOT / CFG["paths"]["pdfs"]
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"pdfs/ 里没有 PDF —— 放到：{pdf_dir}")

    print(f"共 {len(pdfs)} 篇，逐篇检查文字层……\n")
    print(f"{'文件':<34}{'页数':>5}{'字符':>9}{'每页':>7}{'中文占比':>9}  结论")
    print("-" * 92)

    bad = []
    for p in pdfs:
        try:
            n_pages, txt, how = extract(p)
        except Exception as e:
            print(f"{p.name[:33]:<34}{'—':>5}{'—':>9}{'—':>7}{'—':>9}  [X] {type(e).__name__}: {str(e)[:30]}")
            bad.append((p.name, "无法解析"))
            continue
        t = txt.strip()
        n = len(t)
        cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
        latin = sum(1 for ch in t if ch.isascii() and ch.isalpha())
        weird = sum(1 for ch in t if ch in "\ufffd\u0000")
        per_page = n / n_pages if n_pages else 0
        cjk_ratio = cjk / n if n else 0

        if n < MIN_CHARS or per_page < MIN_PER_PAGE:
            verdict = "[X] 几乎没文字层（很可能是扫描件）→ 换文献"
            bad.append((p.name, "无文字层"))
        elif weird > n * 0.02:
            verdict = "[X] 大量乱码（编码问题）→ 换文献"
            bad.append((p.name, "乱码"))
        elif n < 3000 and latin > cjk:
            verdict = "[!] 文字偏少（可能是图文摘要/短篇）→ 可留但注意覆盖度"
        else:
            verdict = f"[OK] 可用（{how}）"
        print(f"{p.name[:33]:<34}{n_pages:>5}{n:>9}{per_page:>7.0f}{cjk_ratio:>8.0%}  {verdict}")

    print("-" * 92)
    if bad:
        print(f"\n[!] 有 {len(bad)} 篇需要处理：")
        for name, why in bad:
            print(f"    {name} —— {why}")
        print("    换成有文字层的版本（能选中复制的 PDF），或先用 OCR 处理；否则不要建库。")
    else:
        print("\n[OK] 全部通过：所有 PDF 都有可用文字层，可以建库。")
    print("\n提示：中文占比低说明这篇是英文文献——混合中英文是好事（能顺带测跨语言检索）。")


if __name__ == "__main__":
    main()
