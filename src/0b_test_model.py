# -*- coding: utf-8 -*-
"""
试某个模型能不能通 —— 不用改 config.yaml，专门用来"找正确的 model 值"。

为什么需要它：各家控制台的模型名经常变（火山方舟要 ep-xxx 接入点 ID 或完整模型 ID），
改一次 config.yaml 试一次太慢。这个脚本让你直接在命令行里换 model 试。

用法（Key 从 .env 里读，先确认已填好）：
  python src\0b_test_model.py --env ARK_API_KEY --url https://ark.cn-beijing.volces.com/api/v3 --model 你复制的模型ID
  python src\0b_test_model.py --env DEEPSEEK_API_KEY --url https://api.deepseek.com/v1 --model deepseek-chat
  python src\0b_test_model.py --env DASHSCOPE_API_KEY --url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen-plus

通了就把这一行照抄进 config.yaml 的 models 里。
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def diagnose(err: str) -> str:
    s = err.lower()
    if "401" in s or "invalid_api_key" in s or "authenticationerror" in s or "unauthorized" in s:
        return "Key 不对：没粘全 / 首尾有空格 / .env 里变量名写错 / 账号未实名认证"
    if "model" in s and ("not found" in s or "notfound" in s or "invalid" in s or "does not exist" in s):
        return "model 值不对：去控制台复制准确的模型 ID，或改用接入点 ID（火山方舟形如 ep-2026xxxx）"
    if "accessdenied" in s or "not activated" in s or "未开通" in s or "forbidden" in s or "403" in s:
        return "该模型服务未开通：去控制台「开通管理」开通这个模型"
    if "quota" in s or "balance" in s or "insufficient" in s or "欠费" in s or "429" in s:
        return "余额/免费额度用完，或被限流（限流的话等一会儿重试）"
    if "timeout" in s or "timed out" in s or "connection" in s:
        return "网络问题：检查是否开了代理/公司网络限制，或稍后重试"
    if "model_not_supported" in s or "unsupported" in s:
        return "该模型不支持这种调用方式，换一个模型档位"
    return "看上面的原始报错信息"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help=".env 里的变量名，如 ARK_API_KEY")
    ap.add_argument("--url", required=True, help="base_url，如 https://ark.cn-beijing.volces.com/api/v3")
    ap.add_argument("--model", help="模型 ID / 接入点 ID；不知道填什么就先加 --list")
    ap.add_argument("--list", action="store_true", help="列出该平台当前可用的模型 ID（省得猜名字）")
    ap.add_argument("--prompt", default="用一句话说明：非小细胞肺癌做驱动基因检测的意义是什么？")
    a = ap.parse_args()

    key = os.getenv(a.env, "").strip()
    if not key:
        raise SystemExit(f"[X] .env 里没有 {a.env}（检查变量名是否拼错、Key 是否粘贴成功）")
    print(f"平台 {a.url}｜Key 长度 {len(key)}，前缀 {key[:10]}…（确认没有多余空格）\n")

    client = OpenAI(api_key=key, base_url=a.url, timeout=90)

    # --list：先把可用模型名列出来，避免反复猜
    if a.list:
        print("该账号可用的模型：")
        try:
            ids = [m.id for m in client.models.list().data]
            for i in ids:
                print("  " + i)
            if not ids:
                print("  （返回为空：可能这个平台不支持 /models 接口，看控制台里的模型列表）")
            print("\n把上面某个名字填进 --model 再跑一次即可。")
        except Exception as e:
            print(f"  [X] 列不出来：{str(e)[:300]}")
            print(f"  -> 判断：{diagnose(str(e))}")
        return

    if not a.model:
        raise SystemExit("要么给 --model 指定模型名，要么加 --list 先列出可用模型。")

    try:
        r = client.chat.completions.create(
            model=a.model, messages=[{"role": "user", "content": a.prompt}],
            temperature=0.2, max_tokens=300)
        text = (r.choices[0].message.content or "").strip()
        u = getattr(r, "usage", None)
        print("[OK] 调用成功，模型回答：")
        print("   " + text[:300])
        if u:
            print(f"\n   tokens: 输入 {u.prompt_tokens} / 输出 {u.completion_tokens}")
        print("\n把这一行抄进 config.yaml 的 models 里：")
        print(f"""  - display: {a.model[:20]}
    name: {a.model}
    base_url: {a.url}
    api_key_env: {a.env}
    price_in: 1.0
    price_out: 2.0""")
    except Exception as e:
        print(f"[X] 调用失败：{type(e).__name__}")
        print(f"   原始报错：{str(e)[:400]}")
        print(f"\n-> 判断：{diagnose(str(e))}")


if __name__ == "__main__":
    main()
