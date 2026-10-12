# -*- coding: utf-8 -*-
"""
公共工具：所有脚本都 import 这个文件。
你不需要改这里，除非要改检索方式。
"""
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI

# Windows 控制台默认是 GBK 编码：万一某行输出含它无法表示的字符，
# 让它降级成 "?" 而不是让整个脚本崩溃。
try:
    import sys as _sys
    _sys.stdout.reconfigure(errors="replace")
    _sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent      # 项目根目录 nsclc-eval/
load_dotenv(ROOT / ".env")
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


# ---------------- 路径 ----------------
# 「开发模式」(DEV)：题目还没通过医学复核时，用它先把流程跑通，但产物一律落到顶层 dev/ 目录，
# 绝不与正式结果混在一起——正式榜单只认 qa/qa_set.jsonl + runs/ + outputs/。
DEV = {"on": False}
_DEV_KEYS = {"qa_set", "mcq_dir", "runs", "outputs"}
# 不隔离的：kb/*（知识库，dev 不重新建库）、pdfs/、review_csv（你手写的出题表，两种模式读同一张）。
# 注意：DEV 不放松任何门槛，只是把"未复核的产物"隔离出来并打上 dev 标记。


def enable_dev(why: str = ""):
    """打开开发模式（幂等）。"""
    if not DEV["on"]:
        DEV["on"] = True
        print("[DEV] 开发模式已开启：题目未通过医学复核 → 产物写入顶层 dev/ 目录")
        print("[DEV] 这些结果**不得**写进简历或正式报告；复核后请用正式模式重跑。")
    if why:
        print(f"[DEV] 原因：{why}")


def _resolve(key: str) -> Path:
    p = ROOT / CFG["paths"][key]
    if DEV["on"] and key in _DEV_KEYS:
        p = ROOT / "dev" / CFG["paths"][key]
    return p


def path(key: str) -> Path:
    """返回配置里的文件路径，并自动创建它的上级目录。"""
    p = _resolve(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dirpath(key: str) -> Path:
    """返回配置里的目录路径，并自动创建它。"""
    p = _resolve(key)
    p.mkdir(parents=True, exist_ok=True)
    return p


# EvalScope 的结果目录。默认写到项目的 outputs/；
# 如需自定义 EvalScope 输出目录，可设置 EVALSCOPE_WORKDIR。
EVAL_WORKDIR = os.getenv("EVALSCOPE_WORKDIR") or str(ROOT / CFG["paths"]["outputs"])


def eval_workdir() -> str:
    """EvalScope 的结果目录（dev 模式下自动指向 dev/outputs，避免与正式结果混在一起）。"""
    return os.getenv("EVALSCOPE_WORKDIR") or str(dirpath("outputs"))


def add_dev_flag(ap):
    """给脚本加统一的 --dev 开关（开发模式：题目未复核，产物隔离到 dev/）。"""
    ap.add_argument("--dev", action="store_true",
                    help="开发模式：题目未通过医学复核；产物写入 dev/ 目录，绝不进正式榜单")


# ---------------- 读写 jsonl ----------------
def read_jsonl(p) -> list:
    p = Path(p)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- 读人工填写的 CSV（多编码兜底）----------------
# [!] 为什么不能"解码不抛错就采用"：**GBK 的汉字字节串常常恰好是合法的 UTF-8**
#     （如 "序号,模型" 的 GBK 字节以 \x90\xf2 / \xa3\xac 结尾，落在合法 UTF-8 区间），
#     于是先试 utf-8 会"成功地"解出乱码，永远轮不到 gbk。因此必须**给每个候选编码打分**
#     再取最优，而不是取第一个不抛异常的。
# [!] 为什么需要 gbk：Windows 上 Excel 默认"CSV(逗号分隔)"存的是本地编码（简体中文 = GBK），
#     实测 qa/人工抽检表.csv 首字节 0xd0 0xf2 → 只认 utf-8-sig 会直接 UnicodeDecodeError 崩溃。
# 顺序只影响同分时的取舍：utf-8-sig（Excel 的"CSV UTF-8"带 BOM）优先。
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "gbk", "gb18030")

_CTRL_BAD = set(range(0, 9)) | {11, 12} | set(range(14, 32))


def _text_plausibility(text: str, raw: bytes = b""):
    """中文文本的合理性打分（越大越像"真的解对了"）；None 表示这个解码明显是错的。

    · 出现 U+FFFD（替换字符）→ 直接淘汰：说明字节流在这个编码下并不合法；
    · **解出 BOM 但原始字节没有 BOM** → 直接淘汰：UTF-8 的 CSV 若被当成 UTF-16 解码，
      开头几个字节会变成 U+FEFF（例如 b"\\xef\\xbb\\xbf" 按 UTF-16LE 解出 U+FEFF + U+BBEF），
      这种"无中生有的 BOM"是选错编码的确证；
    · CJK 汉字 +2、中文标点 +1.5、ASCII 可打印 +1、控制字符 -5 ——
      乱码会把汉字解成稀有/符号字符，分数会明显低于正确解码。
    """
    if not text:
        return 0.0
    if "\ufffd" in text:
        return None
    if text[0] == "\ufeff" and raw[:2] not in (b"\xff\xfe", b"\xfe\xff") \
            and raw[:3] != b"\xef\xbb\xbf" and raw[:4] not in (
                b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return None
    score, cjk, bad = 0.0, 0, 0
    for ch in text[:20000]:
        o = ord(ch)
        if 0x4e00 <= o <= 0x9fff:                   # CJK 统一汉字
            score += 2.0
            cjk += 1
        elif 0x3000 <= o <= 0x303f or 0xff00 <= o <= 0xffef:   # 中文标点 / 全角
            score += 1.5
        elif ch in "\r\n\t":
            score += 0.1
        elif 32 <= o < 127:
            score += 1.0
        elif o in _CTRL_BAD:
            score -= 5.0
            bad += 1
        else:
            score += 0.5
    return score


def detect_csv_encoding(raw: bytes) -> str:
    """判断一段 CSV 字节流最可能的编码；判不出来抛 SystemExit。

    [!] 与 read_csv_any 共用同一套打分：调用方要自己 splitlines() 时也必须用这个结果，
        否则又会出现"两处编码判据不一致"的老问题。
    [!] BOM 是**确证**，必须先判：UTF-8-BOM 的文件用 plain "utf-8" 解码会**留下** U+FEFF，
        于是表头第一列变成 '\ufeff"保留(1=要' —— 列名再也匹配不上，
        `g_col` 取不到值，所有行被静默丢弃（实测导致 3 题全部不入库）。
    """
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"                           # UTF-8 BOM 确证
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw[:4] in (
            b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return "utf-16"                              # UTF-16 BOM 确证
    best, best_score = None, None
    for enc in CSV_ENCODINGS:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        sc = _text_plausibility(text, raw)
        if sc is None:
            continue
        if best_score is None or sc > best_score:
            best, best_score = enc, sc
    if best is None:
        raise SystemExit(f"[X] CSV 编码无法识别（试过 {'、'.join(CSV_ENCODINGS)}）\n"
                         f"    请在 Excel 里「另存为 → CSV UTF-8(逗号分隔)」后重试。")
    return best


def read_csv_any(p) -> list:
    """用多编码兜底读 CSV，返回 list[dict]（表头为键）；文件不存在返回 []。

    [!] 所有"读人工填写的 CSV"的地方都必须走这里：这些文件是用户在 Excel 里存的，
        编码取决于 Excel 而非我们，只认一种编码必崩（GBK 表已复现过）。
    """
    p = Path(p)
    if not p.exists():
        return []
    raw = p.read_bytes()
    if not raw.strip():
        return []
    enc = detect_csv_encoding(raw)
    text = raw.decode(enc)
    # 用 io.StringIO 交给 csv 模块：直接 splitlines() 会丢掉字段内的换行（多行回答很常见）
    import io
    return list(csv.DictReader(io.StringIO(text, newline="")))


def g_col(row, *names, prefix=False):
    """按列名取值，容忍 Excel 改过列名（多余空格、改了括号里的示例文字）。

    prefix=True 时按**前缀**匹配：用户的表常是旧列名（如
    「错误类型(填:数值误读/结论外推/…)」而代码写死了新列名），精确匹配会把已填的值**静默丢弃**。
    """
    for k in row:
        if not k:
            continue
        ks = k.strip()
        if ks in names or (prefix and any(ks.startswith(n) for n in names)):
            return (row[k] or "").strip()
    return ""


# ---------------- 题面（含 A~D 选项）----------------
def full_question(item: dict) -> str:
    """完整题面 = 题干 + 选项行（``A. … B. …``）。

    [!] 为什么要有这个公共函数：正确性/充分性提示词都写着"选择题若只给选项字母，
        按该选项的**文字内容**比较"，但过去只把裸题干传给裁判 —— 裁判既看不到选项文字，
        也没法核对模型回答的字母，判定只能靠猜。四处调用必须用**同一个**拼法，
        否则判定口径又会分叉。
    [!] 注意：改这里会**改变判分输入指纹**（7c 的 fp 已把完整题面纳入），
        因此历史判定缓存会按设计失效并重判 —— 这是刻意的，不是 bug。
    """
    q = item.get("question") or ""
    opts = item.get("options") or {}
    lines = [f"{k}. {opts[k]}" for k in "ABCD" if opts.get(k)]
    return q + ("\n" + "\n".join(lines) if lines else "")


# ---------------- Kappa 退化约定（全项目统一）----------------
def kappa_of(pairs, labels):
    """多分类未加权 Cohen's Kappa，返回 (k, po, pe)；无可比数据返回 (None, None, None)。

    退化约定（**全项目统一，已写进产物**）：
      · 解析失败等**不在 labels 里的判定一律排除**（既不进分子也不进分母）——
        把它们算成"不一致"会人为压低 Kappa，算成"一致"更荒谬；
      · pe >= 1（双方判定完全同分布且只有一个类别）时分母为 0：
        po == 1 → 1.0（完全一致）；否则 0.0（拿不到任何超过偶然的信息）。
        过去 7_report 一律返回 1.0、7b/7d 一律返回 0.0 —— 同一个退化情形给出
        两个相反的结论，报告之间会自相矛盾。
    """
    from collections import Counter
    pairs = [(a, b) for a, b in pairs if a in labels and b in labels]
    n = len(pairs)
    if not n:
        return None, None, None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe >= 1:
        return (1.0 if po >= 1 else 0.0), po, pe
    return (po - pe) / (1 - pe), po, pe


# ---------------- 知识库版本指纹 ----------------
def compute_kb_fp(chunks: list) -> str:
    """按**给定的片段列表**算知识库指纹（不读 kb_version.json）。

    建库脚本必须用这个：如果用 kb_fingerprint()，而 kb/kb_version.json 还存在（上次建库留下的），
    就会取回**旧指纹**——新旧数据就混在一起了。
    """
    raw = json.dumps({
        "chunks": [[c.get("chunk_id"), c.get("text")] for c in chunks],
        "chunk_size": CFG.get("chunk_size"),
        "min_chunk_chars": CFG.get("min_chunk_chars"),
        "embedding": CFG.get("embedding_local") or CFG.get("embedding_hub"),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def file_sha16(p) -> str:
    """文件内容的短哈希（用于校验"版本记录"与"片段文件"是否配套）。"""
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


# ---------------- 答题记录的"额度"口径（4_rag_answer / 扫描脚本共用）----------------
DEFAULT_ANSWER_MAX_TOKENS = 2000      # 与 4_rag_answer.MAX_TOKENS 的默认值一致


def record_budgets(rec: dict, default: int = DEFAULT_ANSWER_MAX_TOKENS) -> list:
    """一条答题记录**尝试过的全部输出额度**（升序去重），用于判断"是否被额度截断"。

    [!] 为什么不能只看 `max_tokens_used`：common.chat 遇到"空内容 + finish_reason=length"会
        **自动把额度翻倍**重试，4_rag_answer 还会再补一次 2×budget 的重答。于是
        "第一次 2000 被思考耗尽、第二次 4000 成功"的记录，out_tokens 可能正好=4000，
        而 `max_tokens_used` 只记了 2000 —— 只看它会把这条**正常答完**的回答误判成截断
        （曾把 3 条误报成 68 条，反向也漏报过真实截断）。
    [!] 字段缺失一律容忍：历史记录没有 `attempts`/`finish_reason`/`max_tokens_used`。
    """
    budgets = set()
    mt = rec.get("max_tokens_used")
    if isinstance(mt, int) and mt > 0:
        budgets.add(mt)
    for att in (rec.get("attempts") or []):
        if not isinstance(att, dict):
            continue
        b = att.get("requested_max_tokens")
        if isinstance(b, int) and b > 0:
            budgets.add(b)
    for b in (rec.get("attempt_budgets") or []):
        if isinstance(b, int) and b > 0:
            budgets.add(b)
    if not budgets:
        budgets.add(int(default))
    return sorted(budgets)


def record_hit_budget_cap(rec: dict, default: int = DEFAULT_ANSWER_MAX_TOKENS) -> bool:
    """这条记录的 out_tokens 是否**恰好停在**它自己试过的某个额度上（= 可能被截断）。

    与 `out_tokens >= 2000` 不同：像 out=5079 而额度是 2000/4000/8000 的记录，
    说明是"额度翻倍后正常答完"，不算截断。
    """
    ot = int(rec.get("out_tokens") or 0)
    if ot <= 0 or rec.get("error"):
        return False
    if ot >= 8192:
        return True
    return ot in set(record_budgets(rec, default))


def kb_fingerprint() -> str:
    """知识库版本指纹（供答题、陷阱题验证使用）。

    优先读 kb/kb_version.json，但会**校验它与 chunks.jsonl 是否一致**（比对片段文件哈希）：
    如果上次重建中途失败，会留下"新片段 + 旧版本记录"，此时必须按当前片段现算，否则指纹是错的。
    """
    vf = path("chunks").parent / "kb_version.json"
    if vf.exists():
        try:
            meta = json.loads(vf.read_text(encoding="utf-8"))
            if meta.get("kb_fp") and meta.get("chunks_sha256") == file_sha16(path("chunks")):
                return str(meta["kb_fp"])
            print("[!] kb_version.json 与 chunks.jsonl 不一致（可能是上次重建中断）"
                  "→ 按当前片段现算指纹")
        except Exception:
            pass
    return compute_kb_fp(read_jsonl(path("chunks")))


# ---------------- 调用大模型 ----------------
_clients = {}


PLACEHOLDER = "在这里粘贴"


def key_of(env_name: str) -> str:
    """读 Key，并把 .env 里的占位文字当成"没配置"——
    否则会把中文占位符当 Key 去发请求，报出很难懂的 UnicodeEncodeError。"""
    v = (os.getenv(env_name) or "").strip()
    return "" if (not v or PLACEHOLDER in v) else v


def client_for(m: dict) -> OpenAI:
    """同一个 base_url 复用连接；Key 从环境变量读，不写进代码。"""
    if m["base_url"] not in _clients:
        key = key_of(m["api_key_env"])
        if not key:
            raise SystemExit(f"环境变量 {m['api_key_env']} 没设置 —— 检查 .env 文件")
        _clients[m["base_url"]] = OpenAI(api_key=key, base_url=m["base_url"], timeout=120)
    return _clients[m["base_url"]]


_NO_TEMP = set()      # 记下哪些模型不接受 temperature（如推理型模型 kimi-k3）

_PREFIX_ONLY = re.compile(r"^答案\s*[:：]?\s*[。．.,，、;；:：\s]*$")


def is_substantive(text, min_chars: int = 4) -> bool:
    """作答是否有实质内容。

    [!] 为什么需要：`answer_one` 过去只把**空串**当失败，于是 "答案："（只有前缀、没有选项字母）
        这类实质空答被当成"成功作答"——DeepSeek/NSCLC-027 正是额度耗尽后只输出了前缀。
    """
    t = (text or "").strip()
    if not t or t.startswith("[调用失败]"):
        return False
    if len(t) < min_chars:
        return False
    return not _PREFIX_ONLY.match(t)


def chat(m: dict, messages: list, temperature: float = 0.2, max_tokens: int = 900,
         retries: int = 4, **kw):
    """调一次模型，失败自动重试。

    兼容性处理：
      · 部分推理型模型（如百炼上的 kimi-k3）不接受 temperature → 自动去掉该参数重试并记住；
      · **推理型模型会把 max_tokens 全用在"思考"上，导致可见内容为空**
        （实测 deepseek-v4-pro：max_tokens=300 时返回空字符串，判定会被静默记成"解析失败"）。
        因此内容为空且 finish_reason=length 时自动把额度翻倍重试（上限 8192）；
        仍为空则返回明确的失败文本——**绝不返回空串让上层误判**。
    返回 (文本, 用量字典)。

    [!] 用量字典同时含 `attempts`（每次尝试的请求额度/结束原因/用量）与 `finish_reason`：
        过去只返回**最后一次**尝试的用量，"第一次 2000 额度实质空答、第二次 4000 成功"
        这种情况在记录里完全看不出来，事后无法复盘额度到底够不够。
    [!] 兼容性：`{"in_tokens":…, "out_tokens":…}` 两个键的位置与含义**未变**，
        所有 `text, usage = chat(...)` 的老调用点不用改。
    """
    last_err = None
    budget = int(max_tokens)
    attempts = []                     # 每次尝试：请求额度 / finish_reason / 用量（含被丢弃的尝试）
    for i in range(retries):
        try:
            payload = {"model": m["name"], "messages": messages, "max_tokens": budget, **kw}
            if m["name"] not in _NO_TEMP:
                payload["temperature"] = temperature
            r = client_for(m).chat.completions.create(**payload)
            text = (r.choices[0].message.content or "").strip()
            fin = getattr(r.choices[0], "finish_reason", "")
            att = {"requested_max_tokens": budget, "finish_reason": fin, "ok": bool(text)}
            if getattr(r, "usage", None):
                att["in_tokens"] = r.usage.prompt_tokens
                att["out_tokens"] = r.usage.completion_tokens
            attempts.append(att)
            usage = {}
            if getattr(r, "usage", None):
                usage = {"in_tokens": r.usage.prompt_tokens, "out_tokens": r.usage.completion_tokens}
            if not text:
                if fin == "length" and budget < 8192:
                    print(f"  [!] {m['name']} 返回空内容（max_tokens={budget} 被思考耗尽）"
                          f"→ 提高额度重试")
                    budget = min(budget * 2, 8192)
                    continue
                last_err = RuntimeError(f"模型返回空内容（finish_reason={fin}）")
                break
            # 成功：把"这次成功用的是多少额度"和全部尝试一并交给上层留档
            usage["attempts"] = attempts
            usage["finish_reason"] = fin
            return text, usage
        except Exception as e:                      # 网络抖动 / 限流 / 参数不支持 / 偶发 5xx
            msg = str(e)
            if "temperature" in msg.lower() and m["name"] not in _NO_TEMP:
                _NO_TEMP.add(m["name"])
                print(f"  [i] {m['name']} 不支持 temperature 参数，已自动去掉后重试")
                continue
            last_err = e
            time.sleep(2 * (i + 1))
    if last_err is None:
        last_err = RuntimeError("模型持续返回空内容")
    # [!] 失败路径也带上 attempts：否则"到底试了几次、每次给了多少额度"这条线索就丢了
    return (f"[调用失败] {type(last_err).__name__}: {last_err}",
            {"attempts": attempts, "finish_reason": ""})


def est_cost(m: dict, in_tokens: int, out_tokens: int) -> float:
    """按 config.yaml 里填的价格估算花费（元）。"""
    return in_tokens / 1e6 * float(m.get("price_in", 0)) + out_tokens / 1e6 * float(m.get("price_out", 0))


def models_with_key() -> list:
    """只返回 .env 里真的配了 Key 的模型，没配的自动跳过。"""
    out = []
    for m in CFG["models"]:
        if key_of(m["api_key_env"]):
            out.append(m)
        else:
            print(f"  · 跳过 {m['display']}（{m['api_key_env']} 未配置）")
    return out


# ---------------- 检索 ----------------
_KB = None
_ENC = None


def encoder():
    """嵌入模型：优先用本地目录（不联网），没有就按 hub 名下载。"""
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        local = ROOT / CFG["embedding_local"]
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        src = str(local) if local.exists() else CFG["embedding_hub"]
        print(f"  加载嵌入模型：{src}")
        _ENC = SentenceTransformer(src)
    return _ENC


def load_kb():
    global _KB
    if _KB is None:
        chunks = read_jsonl(path("chunks"))
        emb = np.load(path("emb"))
        _KB = (chunks, emb)
    return _KB


# ---------------- 参考文献/引用条目识别（建库过滤与检索降权共用）----------------
# 为什么需要：参考文献条目**术语密集**（"unresectable stage III … concurrent … consolidation"），
# 稠密检索很容易把它们排到正文前面。实测一个"III 期巩固治疗"问题：前 8 名全是引用条目，
# 而正确正文排在第 9 名（0.602）。所以既要建库时过滤，也要在检索时降权。
REF_HEAD = re.compile(r"^\s*(references|reference list|bibliography|literature cited)"
                      r"[\s:.\-–—]*$", re.I)
_CITE = re.compile(r"\[\d{1,3}\]")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ETAL = re.compile(r"\bet al\b", re.I)
# 「姓 + 首字母缩写」是参考文献条目的强特征（Lim E, / Batchelor TJP,）；正文几乎不出现
_AUTHORTOK = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z]{1,3}\b")
_VOLISSUE = re.compile(r"\d{4}\s*;\s*\d{1,3}\s*\(\s*\d{1,4}\s*\)\s*[:：]\s*\d+")
_PAGES = re.compile(r"\b\d{1,4}\s*[-–]\s*\d{1,4}\b")
_JOURNALS = re.compile(
    r"\b(Ann Oncol|N Engl J Med|Lancet Oncol|Lancet|J Clin Oncol|Eur Respir J|"
    r"Int J Radiat Oncol|Radiother Oncol|Cancer Discov|J Thorac Oncol|Cancer|"
    r"Oncologist|Clin Cancer Res|Eur J Cancer|JAMA|Chest|Thorax)\b")
_DOIISH = re.compile(r"(\bdoi\s*:|Epub|PMID|https?://)", re.I)

REF_SIGNALS = [
    ("[n]≥2", lambda t: len(_CITE.findall(t)) >= 2, 2),
    ("et al", lambda t: len(_ETAL.findall(t)) >= 1, 2),
    ("作者缩写≥3", lambda t: len(_AUTHORTOK.findall(t)) >= 3, 2),
    ("作者缩写≥6", lambda t: len(_AUTHORTOK.findall(t)) >= 6, 1),
    ("年≥2", lambda t: len(_YEAR.findall(t)) >= 2, 1),
    ("年≥4", lambda t: len(_YEAR.findall(t)) >= 4, 1),
    ("卷(期):页", lambda t: len(_VOLISSUE.findall(t)) >= 1, 3),
    ("卷(期):页≥2", lambda t: len(_VOLISSUE.findall(t)) >= 2, 2),
    ("期刊", lambda t: len(_JOURNALS.findall(t)) >= 1, 2),
    ("doi/Epub", lambda t: len(_DOIISH.findall(t)) >= 1, 2),
    ("页码≥2", lambda t: len(_PAGES.findall(t)) >= 2, 1),
    ("数字>7%", lambda t: sum(1 for c in t if c.isdigit()) / max(1, len(t)) > 0.07, 1),
    ("分号≥4", lambda t: t.count(";") >= 4, 1),
]


def ref_score(t: str) -> int:
    """文献条目得分：越高越像参考文献条目（正文一般 0~2，引用条目 7~9）。"""
    return sum(w for _, fn, w in REF_SIGNALS if fn(t))


def ref_threshold(in_section: bool = False) -> int:
    """建库过滤阈值。

    章节外取 10：实测 8 会误伤正文（LUSTRE 的结论得 8、NCCN 流程文字得 9）。
    章节内（最后一个 References 标题之后）取 7：那里的块即便"混合型"也以引用条目为主，
    但实测**章节内的 ST-1/ST-2/ST-3 分期表等正文得分只有 0~5**，所以 7 仍能保住它们。
    """
    cf = CFG.get("chunk_filter") or {}
    if in_section:
        return int(cf.get("reference_score_threshold_in_section", 7))
    return int(cf.get("reference_score_threshold", 10))


def ref_section_offset(text: str):
    """返回**最后一个** References 标题所在的字符偏移（没有则 None）。

    只用来给"章节内"的块换一个更严的阈值，**不再整段截断**——
    实测 NCCN 最后一个 References 标题之后还有 ST-1/ST-2 分期表等正文
    （18,616 字符切成 22 块，其中 17 块是正文，整段截掉就是误伤）。
    """
    off, found = 0, None
    for line in text.split("\n"):
        if REF_HEAD.match(line):
            found = off
        off += len(line) + 1
    return found


def strip_reference_section(text: str):
    """[已弃用，不再被建库调用] 整段截掉末尾参考文献章节。

    为什么弃用：NCCN 的最后一个 References 标题之后仍有正文表格（ST-1/ST-2 分期表等），
    整段截断会连带丢掉它们且无法逐块留档。现改为"章节内按块过滤 + 逐块留档"。
    保留此函数只为向后兼容与对照。
    """
    lines = text.split("\n")
    idxs = [i for i, l in enumerate(lines) if REF_HEAD.match(l)]
    if not idxs:
        return text, 0, None
    cut = idxs[-1]
    kept = "\n".join(lines[:cut])
    return kept, len(text) - len(kept), (cut, len(lines))


def retrieve(query: str, top_k: int = None) -> list:
    """返回与问题最相似的若干文献片段（每项是 chunks.jsonl 里的一行，含 score / rank_score）。

    排序做了**引用条目降权**：候选池取大一些（4 倍），再按
        rank_score = 稠密相似度 − ref_penalty × 文献条目得分
    重排。实测依据：不降权时某题的 top-8 全是引用条目、正确正文排第 9；
    降权 0.01/分即可把它提到前面，且不会打乱"正文 vs 正文"的相对次序。
    """
    top_k = top_k or int(CFG["top_k"])
    penalty = float((CFG.get("chunk_filter") or {}).get("ref_penalty", 0.0))
    chunks, emb = load_kb()
    q = encoder().encode([query], normalize_embeddings=True)[0]
    sims = emb @ q                                  # 向量都已归一化 → 点积即余弦相似度
    idx = np.argsort(-sims)[: max(top_k * 4, 24) if penalty > 0 else top_k]
    out = []
    for i in idx:
        row = dict(chunks[int(i)])
        row["score"] = float(sims[int(i)])
        row["ref_score"] = ref_score(row["text"])
        row["rank_score"] = row["score"] - penalty * row["ref_score"]
        out.append(row)
    if penalty > 0:
        out.sort(key=lambda r: -r["rank_score"])
    return out[:top_k]


def extract_json_array(text: str) -> list:
    """从模型输出里抠出 JSON 数组（模型常会多写几行解释或加 ```json 围栏）。"""
    if not text:
        return []
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for seg in parts:
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if seg.startswith("["):
                t = seg
                break
    i, j = t.find("["), t.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return []
    try:
        data = json.loads(t[i:j + 1])
    except Exception:
        return []
    return data if isinstance(data, list) else []
