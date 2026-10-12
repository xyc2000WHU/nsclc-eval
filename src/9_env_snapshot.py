# -*- coding: utf-8 -*-
"""
环境快照：记录 Python 版本、关键包版本、完整锁定版本清单。
用途：报告里"可复现性"一节要用（面试官一定会问"别人怎么复现你的结果"）。

产出：
  outputs/环境快照.md        给人看的表格
  requirements-lock.txt      给别人照装的精确版本（pip freeze）

用法：python src/9_env_snapshot.py
"""
import datetime
import importlib.metadata as md
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

KEY = ["numpy", "pandas", "openai", "python-dotenv", "PyYAML", "pypdf", "PyMuPDF",
       "sentence-transformers", "torch", "transformers", "huggingface-hub",
       "modelscope", "evalscope", "ragas", "openpyxl"]


def ver(p: str) -> str:
    try:
        return md.version(p)
    except Exception:
        return "（未安装）"


def main():
    lines = [
        "# 环境快照",
        f"- 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- Python：{sys.version.split()[0]}（{platform.architecture()[0]}）",
        f"- 解释器：{sys.executable}",
        f"- 操作系统：{platform.platform()}",
        "",
        "## 关键包版本",
        "",
        "| 包 | 版本 |",
        "|---|---|",
    ]
    for p in KEY:
        lines.append(f"| {p} | {ver(p)} |")

    # 完整锁定清单（pip freeze）。若沙箱/权限不允许捕获子进程输出，就跳过。
    freeze = ""
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        freeze = (r.stdout or "").strip()
    except Exception as e:
        freeze = f"（自动获取失败：{type(e).__name__}，请手动运行 pip freeze > requirements-lock.txt）"

    if freeze:
        lines += ["", "## pip freeze（完整锁定版本，别人照这个装就能复现）", "", "```", freeze, "```"]
        (ROOT / "requirements-lock.txt").write_text(freeze + "\n", encoding="utf-8")

    (OUT / "环境快照.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:16]))
    print(f"\n[OK] 已写出 {OUT / '环境快照.md'}")
    if freeze and not freeze.startswith("（"):
        print(f"[OK] 已写出 {ROOT / 'requirements-lock.txt'}（{len(freeze.splitlines())} 个包）")


if __name__ == "__main__":
    main()
