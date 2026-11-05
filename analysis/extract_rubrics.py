"""Inject the judge rubric into the Stage 9 kernel script, straight from the pilot notebook.

The notebook (`kaggle/notebooks/bangla-filtering-selection.ipynb`, cell c022) is the single
source of truth for the rubric, and HANDOFF requires the Stage 3 judge and any later judge
to use byte-identical prompts. Hand-copying the rubric risks drift, so this script rewrites
the marked block in `kaggle/full_run/s09_judge_human/s09_judge_human.py` from the notebook.

Also writes `rubrics.py` next to the kernel, so local analysis can import the same rubric.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "kaggle/notebooks/bangla-filtering-selection.ipynb"
KERNEL = ROOT / "kaggle/full_run/s09_judge_human/s09_judge_human.py"
RUBRICS_PY = ROOT / "kaggle/full_run/s09_judge_human/rubrics.py"

BEGIN = "# --- BEGIN GENERATED RUBRIC (analysis/extract_rubrics.py) ---"
END = "# --- END GENERATED RUBRIC ---"

nb = json.loads(NB.read_text(encoding="utf-8"))

source = None
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    text = "".join(cell["source"])
    if text.startswith("RUBRIC = {"):
        source = text
        break

if source is None:
    raise SystemExit("RUBRIC dict not found in the notebook")

rubric_src = source.split("\n\ndef clip", 1)[0].rstrip()
if not rubric_src.rstrip().endswith("}"):
    raise SystemExit("unexpected RUBRIC source shape")

# --- kernel script ---------------------------------------------------------------------------
kernel_text = KERNEL.read_text(encoding="utf-8")
if BEGIN not in kernel_text or END not in kernel_text:
    raise SystemExit(f"markers not found in {KERNEL}")

new_block = f"{BEGIN}\n{rubric_src}\n{END}"
start = kernel_text.index(BEGIN)
end = kernel_text.index(END) + len(END)
kernel_text = kernel_text[:start] + new_block + kernel_text[end:]
KERNEL.write_text(kernel_text, encoding="utf-8")
print(f"injected rubric into {KERNEL.name} ({len(rubric_src)} chars)")
