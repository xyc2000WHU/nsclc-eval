# -*- coding: utf-8 -*-
"""
Day 0 用：验证「Key 对不对、网络通不通、模型名对不对」。
跑通这个脚本，说明后面所有脚本的调用方式都是通的了。

用法：python src\0_hello_api.py
"""
from common import CFG, chat, models_with_key

TEST_Q = "用一句话解释：什么是非小细胞肺癌的驱动基因突变？"


def main():
    print("=" * 60)
    print("被测模型连通性测试（每家问一个问题，成本≈0.01元）")
    print("=" * 60)

    ms = models_with_key()
    if not ms:
        raise SystemExit("一个 Key 都没配。先把 .env.example 复制成 .env 并填入至少一个 Key。")

    ok, bad = [], []
    for m in ms:
        text, usage = chat(m, [{"role": "user", "content": TEST_Q}], max_tokens=200)
        if text.startswith("[调用失败]"):
            print(f"[X] {m['display']:<16} {text[:110]}")
            bad.append(m["display"])
        else:
            print(f"[OK] {m['display']:<16} {text[:60]}...  (tokens: {usage})")
            ok.append(m["display"])

    # 顺手测一下裁判模型
    j = CFG["judge"]
    print("-" * 60)
    text, _ = chat(j, [{"role": "user", "content": "只回答一个词：好"}], max_tokens=10)
    print(("[OK] " if not text.startswith("[调用失败]") else "[X] ") + f"裁判模型 {j['name']}: {text[:60]}")

    print("=" * 60)
    print(f"成功 {len(ok)}/{len(ms)} 家：{', '.join(ok) or '无'}")
    if bad:
        print(f"失败：{', '.join(bad)}")
        print("排查顺序：① .env 里 Key 有没有粘全/多空格 ② config.yaml 里 model 名是否与控制台一致")
        print("          ③ 账户是否欠费/未实名 ④ base_url 是否有多余斜杠")
    print("把这一屏截图存档，作为「环境搭建完成」的证据。")


if __name__ == "__main__":
    main()
