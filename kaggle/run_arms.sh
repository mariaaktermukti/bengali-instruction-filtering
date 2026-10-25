#!/bin/bash
# Train one LoRA adapter per selection arm on Kaggle, one arm after another.
#
# For each arm this creates a private script kernel habiburrahman153/bengali-lora-arm-<x>
# (T4 GPU, Internet on, the bangla-filtering-selection notebook output attached),
# waits for it to finish and saves its training_report.json under results/lora_arms/.
#
# Usage (from the repo root, with the kaggle CLI logged in):
#     bash kaggle/run_arms.sh b c d f a

set -u

KAGGLE_USER="${KAGGLE_USER:-habiburrahman153}"
SELECTION_KERNEL="$KAGGLE_USER/bangla-filtering-selection"
WORK_DIR="${WORK_DIR:-outputs/kaggle_arms}"
RESULTS_DIR="results/lora_arms"

export PYTHONUTF8=1   # the kaggle CLI writes Bengali logs; cp1252 breaks on Windows

mkdir -p "$WORK_DIR" "$RESULTS_DIR"

for l in "$@"; do

    ARM_UPPER=$(echo "$l" | tr a-z A-Z)
    KERNEL="$KAGGLE_USER/bengali-lora-arm-$l"
    KDIR="$WORK_DIR/arm_$l"

    mkdir -p "$KDIR"

    sed "s/^ARM = \"C\"/ARM = \"$ARM_UPPER\"/" scripts/kaggle_lora_training.py > "$KDIR/lora_arm_$l.py"

    cat > "$KDIR/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL",
  "title": "bengali-lora-arm-$l",
  "code_file": "lora_arm_$l.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "kernel_sources": ["$SELECTION_KERNEL"],
  "competition_sources": [],
  "model_sources": [],
  "keywords": []
}
EOF

    echo "PUSH $l: $(kaggle kernels push -p "$KDIR" 2>&1 | tail -1)"

    sleep 20

    while true; do
        STATUS=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
        echo "$STATUS" | grep -qiE "complete|error|cancel" && break
        sleep 30
    done

    echo "DONE $l: $STATUS"

    kaggle kernels output "$KERNEL" -p "$KDIR/out" --force > /dev/null 2>&1

    REPORT="$KDIR/out/outputs/toy_lora_arm_$ARM_UPPER/training_report.json"

    if [ -f "$REPORT" ]; then
        cp "$REPORT" "$RESULTS_DIR/arm_${ARM_UPPER}_training_report.json"
        echo "SAVED $RESULTS_DIR/arm_${ARM_UPPER}_training_report.json"
    else
        echo "NO REPORT for arm $ARM_UPPER, see $KDIR/out/*.log"
    fi

done

python scripts/summarize_lora_arms.py
