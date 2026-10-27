"""Full run, Stage 1 (CPU only): load Bangla-Instruct, sample the 100K pool + 500 held-out rows,
normalise, heuristic filters, dedup, decontamination, token counts.

Generated from the bangla-filtering-selection notebook (cells 2-16, RUN_MODE="full"), so the
filters are byte-identical to the pilot. Invisible characters in regexes are written as escapes.

Kaggle: CPU kernel, Internet ON, no inputs. Outputs under /kaggle/working/bangla_filtering/full/:
    pool/pool_clean.parquet   clean training pool (arm A), with instr_tokens / resp_tokens
    pool/heldout.parquet      held-out rows (never trained on)
    pool/near_dup_pairs.json  MinHash near-dup pairs, for the embedding sanity check in Stage 2
    reports/filter_report.csv, reports/stage1_manifest.json
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "datasets", "huggingface_hub", "ftfy", "rapidfuzz", "datasketch", "pyarrow"])


# ====================================================================================================
# Setup and CONFIG  (notebook cell 2)
# ====================================================================================================
import os, re, io, json, math, time, random, hashlib, unicodedata, html
import importlib.util
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch touches CUDA

# --- platform: Kaggle / Colab / local ----------------------------------------------------------------
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = os.path.exists("/content") and not ON_KAGGLE

def hf_login():
    """Read HF_TOKEN from Kaggle Secrets / Colab secrets (needed for the gated meta-llama weights)."""
    token = os.environ.get("HF_TOKEN")
    try:
        if not token and ON_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        elif not token and ON_COLAB:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
    except Exception as e:
        print("no HF_TOKEN secret found:", type(e).__name__)
    if token:
        os.environ["HF_TOKEN"] = token
        from huggingface_hub import login
        login(token=token)

hf_login()

CONFIG = {
    "RUN_MODE":        "full" ,      # "pilot" (Task 4, ~1K rows) | "full"
    "PILOT_SIZE":      1_000,
    "POOL_SIZE":       100_000,      # the paper's size; the HF file actually holds 342,391 rows
    "POOL_SAMPLING":   "stratified", # stratified by domain (keeps the file's mix exactly) | "uniform"
    "HELDOUT_SIZE":    500,          # removed BEFORE any filtering/selection; never trained on
    "SUBSET_SIZE":     40_000,       # arms B–E
    "SEED":            42,

    "INSTRUCT_REPO":   "md-nishat-008/Bangla-Instruct",   # licence on the Hub: MIT (not cc-by-4.0)
    "INCLUDE_SECONDARY": False,      # iteration02 open-QnA/Wikipedia rows. Keep False for the A–E study:
    "SECONDARY_PATH":  "/content/bangla_instruct_30k/raw/secondary_raw.jsonl",  # mixing sources adds a 2nd variable

    # --- Stage 3 heuristics ---
    "MIN_INSTR_CHARS": 8,
    "MIN_RESP_CHARS":  20,           # relaxed to 2 for domains where short answers are correct
    "MAX_RESP_CHARS":  12_000,
    "MIN_BN_PROSE_RATIO": 0.60,      # Bengali share of letters AFTER removing code / LaTeX / URLs
    "MAX_DUP_LINE_FRAC":  0.50,      # Gopher-style repetition limits, loosened: at Gopher's 0.15–0.30 step-by-step
    "MAX_TOP2GRAM_FRAC":  0.20,      # answers that quote the question get flagged. Not applied to math, whose
    "MAX_DUP5GRAM_FRAC":  0.50,      # LaTeX-stripped skeleton ("সমাধান করি: …") is repetitive by design.
    "MAX_LATIN_RUN":      8,         # ≥8 consecutive English words in Bengali prose = untranslated text
    "DROP_TRANSLATIONESE": True,

    # --- Stage 4 dedup ---
    "MINHASH_PERM": 128, "MINHASH_THRESHOLD": 0.80, "SHINGLE": 5,

    # --- Stage 5 decontamination ---
    "DECON_N": 13, "DECON_SHORT_N": 8,
    "DECON_TEMPLATE_DF": 10,         # an n-gram shared by more pool rows than this is a prompt template, not a leak
    "MHUMANEVAL_BN_URL": "https://raw.githubusercontent.com/mraihan-gmu/mHumanEval-Benchmark/main/mHumanEval-max/mHumanEval-ben_Beng-python.csv",
    # BEnQA, BanglaQuaD and PangBench-bn are not on the HF Hub under those names. When you obtain
    # them, add (name, path_or_url, [text columns]) here so they are decontaminated too.
    "EXTRA_BENCH_FILES": [],

    # --- Stages 6–8 (GPU) ---
    "BASE_MODEL":          "meta-llama/Llama-3.2-1B",   # gated: accept the licence + set HF_TOKEN
    "BASE_MODEL_FALLBACK": "unsloth/Llama-3.2-1B",      # ungated mirror of the same weights
    "RUN_PPL_IFD":   True,
    # "vllm" (fast) when installed, else "hf". Not vLLM on Kaggle: installing it needs a runtime restart,
    # which a pushed (batch) kernel cannot do.
    "JUDGE_BACKEND": "vllm" if (importlib.util.find_spec("vllm") and not ON_KAGGLE) else "hf",
    "JUDGE_MODEL":   "Qwen/Qwen2.5-7B-Instruct",  # open-weight on purpose: Bangla-Instruct was generated by
                                                  # GPT-4o + Claude-3.5, a same-family judge would be circular.
                                                  # Compare e.g. google/gemma-2-9b-it in the pilot.
    "JUDGE_MAX_INSTR_CHARS": 1_200,
    "JUDGE_MAX_RESP_CHARS":  2_000,
    "JUDGE_MIN_MASS":        0.5,    # probability on digits 1–6 below this = judge ignored the format
    "EMBED_MODEL":   "BAAI/bge-m3",
    "TAU_SIM":       0.90,           # DEITA: add a pair only if cos-sim to its nearest selected pair < τ

    "USE_DRIVE": False,              # True = write outputs to Google Drive (survives Colab disconnects)
    "OUT_DIR":   "/kaggle/working/bangla_filtering" if ON_KAGGLE else "/content/bangla_filtering",
}

FULL_POOL_SIZE = CONFIG["POOL_SIZE"]          # used to extrapolate pilot cost
if CONFIG["RUN_MODE"] == "pilot":
    CONFIG.update(POOL_SIZE=CONFIG["PILOT_SIZE"], HELDOUT_SIZE=50,
                  SUBSET_SIZE=int(CONFIG["PILOT_SIZE"] * CONFIG["SUBSET_SIZE"] / FULL_POOL_SIZE))

if CONFIG["USE_DRIVE"] and ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    CONFIG["OUT_DIR"] = "/content/drive/MyDrive/bangla_filtering"

random.seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"])
OUT = Path(CONFIG["OUT_DIR"]) / CONFIG["RUN_MODE"]
for sub in ["pool", "scores", "arms", "reports"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

TIMINGS = {}
@contextmanager
def stage(name, n_rows=None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    TIMINGS[name] = {"seconds": round(dt, 1), "rows": n_rows}
    print(f"[{name}] {dt:,.1f}s" + (f" for {n_rows:,} rows" if n_rows else ""))

print(json.dumps({k: CONFIG[k] for k in ["RUN_MODE", "POOL_SIZE", "HELDOUT_SIZE", "SUBSET_SIZE"]}), "->", OUT)

# ====================================================================================================
# Stage 1: load  (notebook cell 4)
# ====================================================================================================
from datasets import load_dataset

with stage("load"):
    raw = load_dataset(CONFIG["INSTRUCT_REPO"], split="train").to_pandas()
raw = raw.rename(columns={"response": "output"})
raw["orig_idx"] = np.arange(len(raw))          # file order == sub-corpus order
raw = raw.dropna(subset=["instruction", "output"])
raw["instruction"] = raw["instruction"].astype(str)
raw["output"] = raw["output"].astype(str)
raw["source_type"] = "bangla_instruct"
raw["id"] = ["bi" + str(i).zfill(6) for i in raw["orig_idx"]]

if CONFIG["INCLUDE_SECONDARY"] and Path(CONFIG["SECONDARY_PATH"]).exists():
    sec = pd.read_json(CONFIG["SECONDARY_PATH"], lines=True)
    sec["instruction"] = sec["instruction"] + np.where(sec["input"].str.len() > 0, "\n\n" + sec["input"], "")
    sec["orig_idx"] = -1
    sec["id"] = ["sec" + str(i).zfill(6) for i in range(len(sec))]
    raw = pd.concat([raw, sec[["id", "instruction", "output", "source_type", "orig_idx"]]], ignore_index=True)

print(f"{len(raw):,} rows")

# ====================================================================================================
# Stage 1: domain tags  (notebook cell 5)
# ====================================================================================================
DOMAIN_RULES = [   # (name, pattern, where) — first match wins
    ("math",        r"\\boxed|\\frac|\\sqrt|\\\(|\\\[|\$[^$\n]{1,80}\$|সমীকরণ|গণনা কর|সমাধান কর|মান নির্ণয়", "both"),
    ("code",        r"```|\bdef |\bimport |#include|console\.log|\bSELECT\b|public static|কোড|প্রোগ্রাম|ফাংশন|স্ক্রিপ্ট|রেজেক্স|\b(?:python|Python|Java|SQL|HTML|CSS|JavaScript)\b", "both"),
    ("translation", r"অনুবাদ|\b[Tt]ranslate\b|ইংরেজিতে|বাংলায় রূপান্তর", "instr"),
    ("mcq_classification", r"Choices:|Options:|OPTIONS|বিকল্প(?:গুলো|সমূহ)?\s*:|\n\s*[কখগঘ][\)\.]|\n\s*[A-D][\)\.]\s|হ্যাঁ বা না|সত্য না মিথ্যা|শ্রেণ[ীি]বদ্ধ|শনাক্ত কর|চিহ্নিত কর", "instr"),
    ("summarization", r"সারসংক্ষেপ|সারাংশ|সংক্ষেপে|সংক্ষিপ্তসার", "instr"),
    ("creative",    r"গল্প|কবিতা|রচনা|চিঠি|ছড়া|শিরোনাম|স্লোগান|সংলাপ", "instr"),
    ("reading_comp", None, "instr"),
    ("list_brainstorm", r"তালিকা|উপায়|টিপস|পরামর্শ|উদাহরণ দা|উদাহরণ দি", "instr"),
    ("open_qa",     r"\?|কী|কি |কেন|কীভাবে|কিভাবে|কোথায়|কখন|কত|কোন|কে ", "instr"),
]
_DOMAIN_RX = [(n, re.compile(p) if p else None, w) for n, p, w in DOMAIN_RULES]
SHORT_ANSWER_DOMAINS = {"mcq_classification", "reading_comp", "open_qa", "translation"}

def tag_domain(instr, resp):
    both = instr + "\n" + resp[:2000]
    for name, rx, where in _DOMAIN_RX:
        if rx is None:                                   # reading comprehension: long passage + question
            if len(instr) > 500 and ("?" in instr[-300:] or "প্রশ্ন" in instr):
                return name
            continue
        if rx.search(both if where == "both" else instr):
            return name
    return "other"

with stage("domain_tag", len(raw)):
    raw["domain"] = [tag_domain(i, o) for i, o in zip(raw["instruction"], raw["output"])]
raw["segment"] = pd.cut(raw["orig_idx"], 10, labels=[f"s{i}" for i in range(10)])  # position in file

print(raw["domain"].value_counts().to_string())
print(pd.crosstab(raw["segment"], raw["domain"], normalize="index").round(2))

# ====================================================================================================
# Stage 1: sample pool + held-out  (notebook cell 6)
# ====================================================================================================
n_take = min(CONFIG["POOL_SIZE"] + CONFIG["HELDOUT_SIZE"], len(raw))
if CONFIG["POOL_SAMPLING"] == "stratified":
    sampled = raw.groupby("domain").sample(frac=n_take / len(raw), random_state=CONFIG["SEED"])
else:
    sampled = raw.sample(n=n_take, random_state=CONFIG["SEED"])
sampled = sampled.sample(frac=1, random_state=CONFIG["SEED"]).reset_index(drop=True)

heldout = sampled.iloc[:CONFIG["HELDOUT_SIZE"]].copy().reset_index(drop=True)
pool = sampled.iloc[CONFIG["HELDOUT_SIZE"]:].copy().reset_index(drop=True)
pool.to_parquet(OUT / "pool" / "pool_raw.parquet", index=False)
print(f"pool {len(pool):,} | held-out {len(heldout):,}")

# ====================================================================================================
# Stage 2: normalisation  (notebook cell 8)
# ====================================================================================================
import ftfy

CODE_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
ZW_DROP    = re.compile(r"[\u200b\u2060\ufeff\u00ad]")
ZWJ_STRAY  = re.compile(r"(?<![ঀ-৿])[\u200c\u200d]|[\u200c\u200d](?![ঀ-৿])")
KHANDA_TA  = re.compile("ত্\u200d")
HTML_KNOWN = re.compile(r"</?(?:p|br|div|span|b|i|u|a|ul|ol|li|strong|em|font|table|tr|td|th|h[1-6])(?:\s[^<>]*)?/?>", re.I)
INNER_WS   = re.compile(r"(?<=\S)[ \t\u00a0\u3000]{2,}")
TRAIL_WS   = re.compile(r"[ \t]+\n")

def _clean_prose(t):
    t = html.unescape(t)
    t = HTML_KNOWN.sub(" ", t)
    t = INNER_WS.sub(" ", t)
    return TRAIL_WS.sub("\n", t)

def normalize_bn(s):
    if not isinstance(s, str):
        return ""
    s = ftfy.fix_text(s, normalization="NFC", uncurl_quotes=False)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = KHANDA_TA.sub("ৎ", s)
    s = ZW_DROP.sub("", s)
    s = ZWJ_STRAY.sub("", s)
    parts, last = [], 0
    for m in CODE_FENCE.finditer(s):                 # leave fenced code byte-identical
        parts.append(_clean_prose(s[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_clean_prose(s[last:]))
    s = "".join(parts)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def normalize_frame(frame):
    frame = frame.copy()
    for col in ["instruction", "output"]:
        frame[col] = [normalize_bn(x) for x in frame[col]]
    return frame

with stage("normalize", len(pool)):
    changed = pool[["instruction", "output"]].copy()
    pool = normalize_frame(pool)
    heldout = normalize_frame(heldout)
print("rows changed by normalisation:",
      f"instruction {(changed['instruction'] != pool['instruction']).mean():.2%},",
      f"output {(changed['output'] != pool['output']).mean():.2%}")
del changed

# ====================================================================================================
# Stage 3: heuristic filters  (notebook cell 10)
# ====================================================================================================
from rapidfuzz import fuzz

BN_CH     = re.compile(r"[ঀ-৿]")
LATIN_CH  = re.compile(r"[A-Za-z]")
WORD      = re.compile(r"[ঀ-৿\u200c\u200dA-Za-z0-9]+")
MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\(.*?\\\)|\\\[.*?\\\]|\\[A-Za-z]+(?:\{[^{}]*\})*", re.S)
INLINE_CODE = re.compile(r"`[^`\n]+`")
URL_RE    = re.compile(r"https?://\S+|www\.\S+")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*[.,;:!?]?")

MOJIBAKE  = re.compile(r"�|à¦|à§|Ã[\x80-\xbf]|â€")
ARTIFACT  = re.compile(r"নো\s?ইনপুট|<\s*no\s*_?input\s*>|\bno\s?input\b|(?:কোন|কোনো)\s+ইনপুট\s+নেই|ইনপুট\s*:\s*(?:নেই|N/?A)?\s*\Z"
                       r"|###\s*(?:Instruction|Response|Input)\b|<\|(?:im_start|im_end|endoftext|eot_id|start_header_id)\|>", re.I | re.M)
IDENTITY  = re.compile(r"(?:একটি|একজন)\s+(?:এআই|AI|কৃত্রিম\s+বুদ্ধিমত্তা(?:র)?)\s*(?:ভাষা\s*)?(?:মডেল|সহকারী|চ্যাটবট)\s*(?:হিসেবে|হিসাবে)"
                       r"|\bas an ai\b|\bi(?:'m| am) (?:sorry|unable|not able)\b|(?:আমি|আমাকে)\s+(?:ChatGPT|GPT-?4o?|Claude)\b"
                       r"|(?:OpenAI|Anthropic)\s+(?:আমাকে|আমায়)"
                       r"|আমি\s+দুঃখিত[,]?\s*(?:কিন্তু|তবে)?.{0,40}?(?:পারব\s+না|পারি\s+না|সম্ভব\s+নয়|সক্ষম\s+নই)", re.I)
EN_TEMPLATE = re.compile(r"\b(?:Choices|Options|OPTIONS|Answer|Question|Premise|Hypothesis|Passage|Context)\s*:")
CHAR_SPAM = re.compile(r"([^\s\-=_*#.~০-৯0-9|])\1{9,}")
NO_SPACE_WALL = re.compile(r"[ঀ-৿\u200c\u200d]{50,}")
END_OK    = tuple("।॥?!.…\"'”’)]}>*`|:;$%০১২৩৪৫৬৭৮৯0123456789")

def prose_only(s):
    for rx in (CODE_FENCE, MATH_SPAN, INLINE_CODE, URL_RE):
        s = rx.sub(" ", s)
    return s

def bn_prose_ratio(s):
    p = prose_only(s)
    b, l = len(BN_CH.findall(p)), len(LATIN_CH.findall(p))
    return b / (b + l) if (b + l) else 1.0          # pure math / code: nothing to judge

def max_latin_run(s):
    best = run = 0
    for tok in prose_only(s).split():
        run = run + 1 if LATIN_WORD.fullmatch(tok) else 0
        best = max(best, run)
    return best

def repetition_stats(text):
    p = prose_only(text)
    lines = [l.strip() for l in p.split("\n") if l.strip()]
    dup_line = 1 - len(set(lines)) / len(lines) if len(lines) >= 4 else 0.0
    w = WORD.findall(p)
    if len(w) < 50:
        return dup_line, 0.0, 0.0
    total = sum(map(len, w))
    (b1, b2), cnt = Counter(zip(w, w[1:])).most_common(1)[0]
    top2 = cnt * (len(b1) + len(b2)) / total
    grams = [tuple(w[i:i + 5]) for i in range(len(w) - 4)]
    c5 = Counter(grams)
    dup = np.zeros(len(w), bool)
    for i, g in enumerate(grams):
        if c5[g] > 1:
            dup[i:i + 5] = True
    dup5 = sum(len(x) for x, d in zip(w, dup) if d) / total
    return dup_line, top2, dup5

def looks_truncated(resp):
    t = resp.rstrip()
    if t.count("```") % 2 == 1:
        return True
    return len(t) >= 60 and not t.endswith(END_OK)

SENT_END = re.compile(r"[।॥?!.](?=\s|$)")

def repair_truncated(resp, keep_frac=0.6):
    """Cut a response that stops mid-sentence back to its last complete sentence.
    Returns the repaired text, or None when that would lose too much (or a code fence is open)."""
    t = resp.rstrip()
    if t.count("```") % 2 == 1:
        return None
    ends = [m.end() for m in SENT_END.finditer(t)]
    if not ends or ends[-1] < keep_frac * len(t):
        return None
    return t[:ends[-1]]

FILTER_ORDER = ["length", "mojibake", "template_artifact", "identity_refusal", "language",
                "translationese", "echo", "truncated", "char_spam", "no_space_wall", "repetition"]

def heuristic_flags(instr, resp, domain):
    C = CONFIG
    min_resp = 2 if domain in SHORT_ANSWER_DOMAINS else C["MIN_RESP_CHARS"]
    ri, ro = bn_prose_ratio(instr), bn_prose_ratio(resp)
    lang_ok = (max(ri, ro) if domain == "translation" else min(ri, ro)) >= C["MIN_BN_PROSE_RATIO"]
    latin_run = 0 if domain in ("translation", "code") else max(max_latin_run(instr), max_latin_run(resp))
    dup_line, top2, dup5 = repetition_stats(resp) if domain != "math" else (0.0, 0.0, 0.0)
    f = {
        "length":            len(instr) < C["MIN_INSTR_CHARS"] or not (min_resp <= len(resp) <= C["MAX_RESP_CHARS"]),
        "mojibake":          bool(MOJIBAKE.search(instr) or MOJIBAKE.search(resp)),
        "template_artifact": bool(ARTIFACT.search(instr) or ARTIFACT.search(resp)),
        "identity_refusal":  bool(IDENTITY.search(resp)),
        "language":          not lang_ok,
        "translationese":    C["DROP_TRANSLATIONESE"] and (latin_run >= C["MAX_LATIN_RUN"]
                                                           or (domain != "translation" and bool(EN_TEMPLATE.search(instr)))),
        "echo":              len(resp) > 30 and fuzz.ratio(instr, resp) > 90,
        "truncated":         looks_truncated(resp),
        "char_spam":         bool(CHAR_SPAM.search(resp)),
        "no_space_wall":     bool(NO_SPACE_WALL.search(prose_only(resp))),
        "repetition":        dup_line > C["MAX_DUP_LINE_FRAC"] or top2 > C["MAX_TOP2GRAM_FRAC"] or dup5 > C["MAX_DUP5GRAM_FRAC"],
    }
    f["drop_reason"] = next((k for k in FILTER_ORDER if f[k]), "")
    f["bn_ratio_instr"], f["bn_ratio_resp"] = round(ri, 4), round(ro, 4)
    return f

def apply_heuristics(frame):
    # repair first: truncation sits mostly in long-form answers (code, lists), so dropping
    # every cut-off row would bias the pool towards short answers
    frame = frame.copy()
    frame["repaired"] = False
    for ix in frame.index[[looks_truncated(o) for o in frame["output"]]]:
        fixed = repair_truncated(frame.at[ix, "output"])
        if fixed is not None:
            frame.at[ix, "output"], frame.at[ix, "repaired"] = fixed, True
    flags = pd.DataFrame([heuristic_flags(i, o, d) for i, o, d in
                          tqdm(zip(frame["instruction"], frame["output"], frame["domain"]), total=len(frame), desc="heuristics")],
                         index=frame.index)
    return pd.concat([frame, flags], axis=1)

with stage("heuristics", len(pool)):
    pool = apply_heuristics(pool)
    heldout = apply_heuristics(heldout)
heldout = heldout[heldout["drop_reason"] == ""].reset_index(drop=True)

# ====================================================================================================
# Stage 3: filter report  (notebook cell 11)
# ====================================================================================================
# Reason counts + the bias check: does any filter hit one domain much harder than the rest?
dropped = pool[pool["drop_reason"] != ""]
reason_by_domain = pd.crosstab(pool["domain"], pool["drop_reason"].replace("", "kept"))
drop_rate = (pool["drop_reason"] != "").groupby(pool["domain"]).mean().rename("drop_rate")
report = reason_by_domain.join(drop_rate).sort_values("drop_rate", ascending=False)
report.to_csv(OUT / "reports" / "filter_report.csv")

overall = (pool["drop_reason"] != "").mean()
print(f"heuristics drop {len(dropped):,} / {len(pool):,} ({overall:.2%}) | truncated responses repaired: {int(pool.repaired.sum()):,}")
print(dropped["drop_reason"].value_counts().to_string())
print(report)
biased = drop_rate[drop_rate > 2 * overall]
if len(biased):
    print("⚠ filters hit these domains > 2x the average rate — inspect before trusting the filter:", biased.round(3).to_dict())

# one example per reason, for a quick manual look
for r in FILTER_ORDER:
    ex = dropped[dropped["drop_reason"] == r].head(1)
    if len(ex):
        print(f"\n--- {r} ---\nI: {ex.iloc[0]['instruction'][:200]}\nR: {ex.iloc[0]['output'][-200:]}")

# ====================================================================================================
# Stage 3: PII masking  (notebook cell 12)
# ====================================================================================================
# PII is masked, not dropped (rare in synthetic data, but cheap to guarantee)
PII = [(re.compile(r"(?<!\d)(?:\+?88)?01[3-9]\d{8}(?!\d)"), "<PHONE>"),
       (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>")]
pii_hits = 0
for col in ["instruction", "output"]:
    for rx, tag in PII:
        hit = pool[col].str.contains(rx)
        pii_hits += int(hit.sum())
        pool.loc[hit, col] = pool.loc[hit, col].str.replace(rx, tag, regex=True)
print("PII masks applied:", pii_hits)

pool = pool[pool["drop_reason"] == ""].reset_index(drop=True)

# ====================================================================================================
# Stage 4: dedup  (notebook cell 14)
# ====================================================================================================
from datasketch import MinHash, MinHashLSH

PUNCT_WS = re.compile(r"[\s।॥!-/:-@\[-`{-~“”‘’…–—]+")

def dedup_key(*texts):
    s = unicodedata.normalize("NFC", " ".join(texts)).lower()
    return hashlib.md5(PUNCT_WS.sub("", s).encode("utf-8")).hexdigest()

def shingles(text, k):
    t = re.sub(r"\s+", " ", text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}

with stage("dedup", len(pool)):
    n0 = len(pool)
    pool["_k_pair"] = [dedup_key(i, o) for i, o in zip(pool["instruction"], pool["output"])]
    pool["_k_instr"] = [dedup_key(i) for i in pool["instruction"]]
    pool = pool.drop_duplicates("_k_pair").drop_duplicates("_k_instr")
    n_exact = n0 - len(pool)

    lsh = MinHashLSH(threshold=CONFIG["MINHASH_THRESHOLD"], num_perm=CONFIG["MINHASH_PERM"])
    keep, dup_pairs = [], []
    texts = dict(zip(pool["id"], pool["instruction"] + " || " + pool["output"].str[:1000]))
    for rid, text in tqdm(texts.items(), desc="minhash"):
        m = MinHash(num_perm=CONFIG["MINHASH_PERM"], seed=CONFIG["SEED"])
        m.update_batch([s.encode("utf-8") for s in shingles(text, CONFIG["SHINGLE"])])
        hits = lsh.query(m)
        if hits:
            dup_pairs.append((hits[0], rid))
            continue
        lsh.insert(rid, m)
        keep.append(rid)
    near_dup_texts = {rid: texts[rid] for pair in dup_pairs[:300] for rid in pair}   # for the embedding sanity check
    pool = pool[pool["id"].isin(set(keep))].drop(columns=["_k_pair", "_k_instr"]).reset_index(drop=True)
    del lsh, texts

print(f"exact dups: {n_exact:,} | near dups: {len(dup_pairs):,} | pool now {len(pool):,}")
for a, b in dup_pairs[:3]:
    print("\nKEPT:   ", near_dup_texts[a][:150].replace("\n", " "), "\nDROPPED:", near_dup_texts[b][:150].replace("\n", " "))

# ====================================================================================================
# Stage 5: decontamination  (notebook cell 16)
# ====================================================================================================
from huggingface_hub import hf_hub_download

def tokens(s):
    return [t for t in WORD.findall(unicodedata.normalize("NFC", prose_only(s)).lower()) if not t.isdigit()]

def load_benchmarks():
    b = {}
    mm = pd.read_csv(hf_hub_download("openai/MMMLU", "test/mmlu_BN-BD.csv", repo_type="dataset"))
    b["mmlu_bn"] = (mm["Question"].astype(str) + " " + mm[["A", "B", "C", "D"]].astype(str).agg(" ".join, axis=1)).tolist()
    rqa = json.load(open(hf_hub_download("sartajekram/BanglaRQA", "Test.json", repo_type="dataset"), encoding="utf-8"))["data"]
    b["banglarqa_question"] = [q["question_text"] for p in rqa for q in p["qas"]]
    b["banglarqa_context"] = [p["context"] for p in rqa]
    b["mhumaneval_bn"] = pd.read_csv(CONFIG["MHUMANEVAL_BN_URL"])["prompt"].astype(str).tolist()
    for name, path, cols in CONFIG["EXTRA_BENCH_FILES"]:
        ext = pd.read_json(path, lines=path.endswith("l")) if ".json" in path else pd.read_csv(path)
        b[name] = ext[cols].astype(str).agg(" ".join, axis=1).tolist()
    b["heldout"] = heldout["instruction"].tolist()
    return b

def build_ngram_index(bench):
    """n -> {hash: benchmark}; n-grams found in more than 3 benchmark items are templates and skipped."""
    idx = {CONFIG["DECON_N"]: {}, CONFIG["DECON_SHORT_N"]: {}}
    item_df = Counter()
    for name, texts in bench.items():
        for t in texts:
            tk = tokens(t)
            n = CONFIG["DECON_N"] if len(tk) >= CONFIG["DECON_N"] else CONFIG["DECON_SHORT_N"]
            hs = {hash(tuple(tk[i:i + n])) for i in range(len(tk) - n + 1)}
            item_df.update(hs)
            for h in hs:
                idx[n].setdefault(h, name)
    return {n: {h: b for h, b in table.items() if item_df[h] <= 3} for n, table in idx.items()}

def matched_ngrams(text, idx):
    tk, hits = tokens(text), set()
    for n, table in idx.items():
        for i in range(len(tk) - n + 1):
            h = hash(tuple(tk[i:i + n]))
            if h in table:
                hits.add(h)
    return hits

with stage("decontaminate", len(pool)):
    BENCH = load_benchmarks()
    IDX = build_ngram_index(BENCH)
    lookup = {h: b for table in IDX.values() for h, b in table.items()}
    row_hits = [matched_ngrams(i + " " + o, IDX)
                for i, o in tqdm(zip(pool["instruction"], pool["output"]), total=len(pool), desc="decon")]
    pool_df = Counter(h for hs in row_hits for h in hs)          # pool rows sharing each matched n-gram
    pool["contaminated_by"] = [next((lookup[h] for h in hs if pool_df[h] <= CONFIG["DECON_TEMPLATE_DF"]), "")
                               for hs in row_hits]
    n_template = sum(c > CONFIG["DECON_TEMPLATE_DF"] for c in pool_df.values())
print({k: len(v) for k, v in BENCH.items()}, "| template n-grams ignored:", n_template)
print(pool["contaminated_by"].replace("", "clean").value_counts().to_string())

pool = pool[pool["contaminated_by"] == ""].drop(columns=["contaminated_by"]).reset_index(drop=True)
pool.to_parquet(OUT / "pool" / "pool_clean.parquet", index=False)
heldout.to_parquet(OUT / "pool" / "heldout.parquet", index=False)
print(f"CLEAN POOL: {len(pool):,} rows  (arm A = all of these)")


# ====================================================================================================
# Stage 1 extras: token counts (notebook cell 18, CPU part), near-dup pairs, manifest
# ====================================================================================================
from transformers import AutoTokenizer

def load_base_tokenizer():
    for name in (CONFIG["BASE_MODEL"], CONFIG["BASE_MODEL_FALLBACK"]):
        try:
            return name, AutoTokenizer.from_pretrained(name)
        except Exception as e:
            print(f"cannot load {name}: {type(e).__name__}")
    raise RuntimeError("no base tokenizer available")

with stage("token_counts", len(pool)):
    BASE_NAME, base_tok = load_base_tokenizer()
    pool["instr_tokens"] = [len(x) for x in base_tok(pool["instruction"].tolist(), add_special_tokens=False)["input_ids"]]
    pool["resp_tokens"]  = [len(x) for x in base_tok(pool["output"].tolist(), add_special_tokens=False)["input_ids"]]
pool.to_parquet(OUT / "pool" / "pool_clean.parquet", index=False)
print(BASE_NAME, "| total tokens in pool:", f"{int(pool.instr_tokens.sum() + pool.resp_tokens.sum()):,}")

(OUT / "pool" / "near_dup_pairs.json").write_text(
    json.dumps({"pairs": dup_pairs[:300], "texts": near_dup_texts}, ensure_ascii=False), encoding="utf-8")

manifest = {
    "config": CONFIG, "base_model": BASE_NAME,
    "raw_rows": len(raw), "pool_rows_sampled": int(CONFIG["POOL_SIZE"]),
    "pool_rows_clean": len(pool), "heldout_rows": len(heldout),
    "heuristic_drops": dropped["drop_reason"].value_counts().to_dict(),
    "exact_dups": n_exact, "near_dup_pairs": len(dup_pairs),
    "pool_domains": pool["domain"].value_counts().to_dict(),
    "heldout_domains": heldout["domain"].value_counts().to_dict(),
    "pool_tokens": int(pool.instr_tokens.sum() + pool.resp_tokens.sum()),
    "timings": TIMINGS,
}
(OUT / "reports" / "stage1_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2, ensure_ascii=False, default=str))
