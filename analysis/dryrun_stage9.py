"""Local dry-run of the Stage 9 kernel: everything except the real vLLM scoring calls.

Mocks the vLLM worker so the input loading, the human/pool matching, the two passes
(Qwen judge + independent Llama rater), the output assembly and the summary can be checked
before spending GPU quota.

Run:  .venv\\Scripts\\python.exe analysis\\dryrun_stage9.py
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle/full_run/s09_judge_human/s09_judge_human.py"
DATASET = ROOT / "outputs/validation_dataset"

tmp = Path(tempfile.mkdtemp(prefix="s09_dryrun_"))
input_dir = tmp / "input"
out_dir = tmp / "out"
input_dir.mkdir(parents=True)

for name in ("human_annotation_100.jsonl", "spotcheck_blind.csv", "pool_matches.csv"):
    shutil.copy2(DATASET / name, input_dir / name)

print(f"dry-run input : {input_dir}")
print(f"dry-run output: {out_dir}\n")

# ---- fake vllm module ----------------------------------------------------------------------
fake = types.ModuleType("vllm")


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")

    def get_tokenizer(self):
        class Tok:
            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
                return msgs[0]["content"]

            def decode(self, ids):
                return str(ids[0])

        return Tok()

    def generate(self, prompts, params, use_tqdm=False):
        class LP:
            def __init__(self, tok):
                self.decoded_token = tok
                self.logprob = -0.1

        class Out:
            def __init__(self):
                self.logprobs = [{1: LP(" 4"), 2: LP(" 5")}]

        class O:
            def __init__(self):
                self.outputs = [Out()]

        return [O() for _ in prompts]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        pass


fake.LLM = FakeLLM
fake.SamplingParams = FakeSamplingParams
sys.modules["vllm"] = fake

# ---- load the kernel module ----------------------------------------------------------------
os.environ["S09_INPUT"] = str(input_dir)
os.environ["S09_OUT"] = str(out_dir)
os.environ["DEADLINE"] = str(time.time() + 3600)

spec = importlib.util.spec_from_file_location("s09_judge_human", KERNEL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# neutralise the pip install / vllm version probe, and run the worker in-process
class FakeCompleted:
    def __init__(self, code=0):
        self.returncode = code


def fake_run(args, *rest, **kwargs):
    if args and str(args[-1]) == "worker":
        os.environ.update(kwargs.get("env", {}))
        mod.worker()
        return FakeCompleted(0)
    return FakeCompleted(0)


mod.subprocess.check_call = lambda *a, **k: 0
mod.subprocess.run = fake_run
mod.subprocess.TimeoutExpired = TimeoutError

print("running the kernel with a mocked vLLM backend...\n")
mod.main()

# ---- verify the outputs --------------------------------------------------------------------
human_out = out_dir / "judge_human_100.jsonl"
sc_out = out_dir / "spotcheck_judge.csv"
summary_out = out_dir / "judge_human_summary.json"

assert human_out.exists(), "judge_human_100.jsonl missing"
assert sc_out.exists(), "spotcheck_judge.csv missing"
assert summary_out.exists(), "judge_human_summary.json missing"

rows = [json.loads(line) for line in human_out.read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.DataFrame(rows)
print(f"\njudge_human_100.jsonl: {len(df)} rows")
print(df["source"].value_counts().to_string())
assert len(df) == 100, f"expected 100 human rows, got {len(df)}"
assert df["llm_complexity"].notna().all(), "some rows have no complexity score"
assert df["llm_quality"].notna().all(), "some rows have no quality score"
assert df["llama_usefulness"].notna().all(), "some rows have no independent rating"

sc = pd.read_csv(sc_out)
print(f"\nspotcheck_judge.csv: {len(sc)} rows")
assert len(sc) == 100, f"expected 100 spot-check rows, got {len(sc)}"
assert sc["llama_usefulness"].notna().all(), "some spot-check rows have no independent rating"

summary = json.loads(summary_out.read_text(encoding="utf-8"))
print("\nsummary:")
print(json.dumps(summary, indent=2))
assert summary["complete"], "summary says incomplete"
assert summary["human_rows_from_stage3_pool"] == 29, summary
assert summary["human_rows_fresh"] == 71, summary
assert summary["human_rows_with_llama"] == 100, summary
assert summary["spotcheck_rows_with_llama"] == 100, summary

print("\nDRY RUN OK")
shutil.rmtree(tmp, ignore_errors=True)
