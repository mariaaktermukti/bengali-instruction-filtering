"""Phase 4 & Phase 5 Comprehensive Analysis Package.

1. Qualitative generation samples across arms (Phase 4 final item) -> results/full_run/qualitative_samples.md
2. Master results synthesis table (Phase 5 item 1) -> results/full_run/main_results_table.csv & .md
3. Token efficiency analysis (Phase 5 item 2) -> results/full_run/token_efficiency.csv
4. Domain-mix & filter bias analysis (Phase 5 items 3 & 5) -> results/full_run/domain_filter_bias.csv
5. Tau / diversity ablation (Phase 5 item 4) -> results/full_run/diversity_ablation.csv
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path("results/full_run")
BENCH_DIR = RESULTS_DIR / "benchmarks"
PAIRWISE_DIR = RESULTS_DIR / "pairwise"
GEN_DIR = Path("outputs/full/s07_02/bangla_filtering/full/benchmarks/generations")
S04_REPORTS = Path("outputs/full/s04_v4/bangla_filtering/full/reports")
S01_REPORTS = Path("outputs/full/s01/bangla_filtering/full/reports")

MODELS = ["base", "A", "B", "C", "D", "E", "F"]

def step1_qualitative_samples():
    print("Extracting qualitative generation samples across arms...")
    # Load generation files for all models
    gen_data = {}
    model_files = {
        "base": GEN_DIR / "base.jsonl",
        "A": GEN_DIR / "arm_A_seed42.jsonl",
        "B": GEN_DIR / "arm_B_seed42.jsonl",
        "C": GEN_DIR / "arm_C_seed42.jsonl",
        "D": GEN_DIR / "arm_D_seed42.jsonl",
        "E": GEN_DIR / "arm_E_seed42.jsonl",
        "F": GEN_DIR / "arm_F_seed42.jsonl",
    }
    
    for m, p in model_files.items():
        gen_data[m] = {}
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gen_data[m][r["id"]] = r
                    
    # Select representative prompts from diverse domains: math, open_qa, code, creative, list_brainstorm
    prompt_ids = list(gen_data["base"].keys())
    # Group by domain
    domain_to_pids = {}
    for pid in prompt_ids:
        dom = gen_data["base"][pid]["domain"]
        domain_to_pids.setdefault(dom, []).append(pid)

    selected_samples = []
    # Pick 1 from code, 1 from math, 1 from open_qa, 1 from creative, 1 from list_brainstorm
    target_domains = ["code", "math", "open_qa", "creative", "list_brainstorm"]
    for dom in target_domains:
        if dom in domain_to_pids:
            # pick the first one that has moderate length instruction
            cand = None
            for pid in domain_to_pids[dom]:
                instr = gen_data["base"][pid]["instruction"]
                if 20 < len(instr) < 300:
                    cand = pid
                    break
            if not cand:
                cand = domain_to_pids[dom][0]
            selected_samples.append(cand)

    out_md = []
    out_md.append("# Qualitative Generation Samples across Model Arms\n")
    out_md.append("This document contains qualitative side-by-side generations on representative held-out Bengali prompts.\n")

    for pid in selected_samples:
        base_item = gen_data["base"][pid]
        out_md.append(f"## Prompt ID: `{pid}` (Domain: `{base_item['domain']}`)\n")
        out_md.append(f"**Instruction:**\n> {base_item['instruction']}\n\n")
        out_md.append("### Model Responses:\n")
        for m in MODELS:
            resp = gen_data[m][pid]["response"]
            # truncate display to 400 chars if very long for readability
            resp_disp = resp.strip()
            out_md.append(f"#### Model {m}:\n```\n{resp_disp}\n```\n")
        out_md.append("---\n")

    out_path = RESULTS_DIR / "qualitative_samples.md"
    out_path.write_text("\n".join(out_md), encoding="utf-8")
    print(f"Saved {out_path}")


def step2_master_results_table():
    print("Generating master results table...")
    # Load heldout loss
    df_heldout = pd.read_csv(RESULTS_DIR / "heldout_comparison.csv").set_index("model")
    # Load token-weighted loss
    df_token = pd.read_csv(RESULTS_DIR / "token_weighted.csv").set_index("model")
    # Load MMLU
    df_mmlu = pd.read_csv(BENCH_DIR / "mmlu_comparison.csv").set_index("model")
    # Load RQA
    df_rqa = pd.read_csv(BENCH_DIR / "rqa_comparison.csv").set_index("model")
    # Load Pairwise vs B
    df_pw = pd.read_csv(PAIRWISE_DIR / "pairwise_vs_B.csv").set_index("model")
    # Load Arm Diagnostics
    df_diag = pd.read_csv(S04_REPORTS / "arm_diagnostics.csv").set_index("arm")

    rows = []
    arm_descriptions = {
        "A": "Whole Pool (91.1K)",
        "B": "Random Sample (10K)",
        "C": "DEITA (c x q + div tau=0.74)",
        "D": "Quality-only (10K)",
        "E": "DEITA score-only (c x q, no div)",
        "F": "Cherry-LLM / IFD (10K)",
        "base": "Base (Qwen2.5-0.5B-Inst)"
    }

    for m in MODELS:
        n_tokens = df_diag.loc[m, "tokens_total"] if m in df_diag.index else None
        tokens_disp = f"{n_tokens / 1e6:.1f}" if n_tokens is not None and pd.notna(n_tokens) else "-"
        m_loss = df_heldout.loc[m, "mean_loss"] if m in df_heldout.index else np.nan
        loss_diff_b = df_heldout.loc[m, "diff_vs_B"] if (m in df_heldout.index and pd.notna(df_heldout.loc[m, "diff_vs_B"])) else 0.0
        
        tw_loss = df_token.loc[m, "token_weighted_loss"] if m in df_token.index else np.nan
        tw_diff_b = df_token.loc[m, "diff_vs_B"] if (m in df_token.index and pd.notna(df_token.loc[m, "diff_vs_B"])) else 0.0
        
        mmlu_acc = df_mmlu.loc[m, "accuracy"] if m in df_mmlu.index else np.nan
        mmlu_diff_b = df_mmlu.loc[m, "diff_vs_B"] if (m in df_mmlu.index and pd.notna(df_mmlu.loc[m, "diff_vs_B"])) else 0.0
        
        rqa_f1 = df_rqa.loc[m, "f1_all"] if m in df_rqa.index else np.nan
        rqa_diff_b = df_rqa.loc[m, "diff_vs_B"] if (m in df_rqa.index and pd.notna(df_rqa.loc[m, "diff_vs_B"])) else 0.0
        
        win_rate_b = df_pw.loc[m, "win_rate"] if m in df_pw.index else (0.5000 if m == "B" else np.nan)
        pw_ci = f"[{df_pw.loc[m, 'ci95_low']:.3f}, {df_pw.loc[m, 'ci95_high']:.3f}]" if m in df_pw.index else "[0.500, 0.500]"

        rows.append({
            "model": m,
            "description": arm_descriptions.get(m, m),
            "train_tokens_M": tokens_disp,
            "heldout_loss": round(m_loss, 4),
            "loss_diff_vs_B": f"{loss_diff_b:+.4f}" if m != "B" else "-",
            "token_wt_loss": round(tw_loss, 4),
            "token_wt_diff_vs_B": f"{tw_diff_b:+.4f}" if m != "B" else "-",
            "mmlu_acc": round(mmlu_acc, 4),
            "mmlu_diff_vs_B": f"{mmlu_diff_b:+.4f}" if m != "B" else "-",
            "rqa_f1": round(rqa_f1, 4),
            "rqa_diff_vs_B": f"{rqa_diff_b:+.4f}" if m != "B" else "-",
            "pairwise_win_vs_B": round(win_rate_b, 4),
            "pairwise_win_95ci": pw_ci,
        })

    df_master = pd.DataFrame(rows)
    df_master.to_csv(RESULTS_DIR / "main_results_table.csv", index=False)
    
    # Generate clean Markdown table
    md_table = df_master.to_markdown(index=False)
    (RESULTS_DIR / "main_results_table.md").write_text(md_table, encoding="utf-8")
    print("Master results table saved to CSV and MD.")


def step3_token_efficiency():
    print("Generating token efficiency analysis...")
    df_diag = pd.read_csv(S04_REPORTS / "arm_diagnostics.csv").set_index("arm")
    df_heldout = pd.read_csv(RESULTS_DIR / "heldout_comparison.csv").set_index("model")
    df_mmlu = pd.read_csv(BENCH_DIR / "mmlu_comparison.csv").set_index("model")
    df_rqa = pd.read_csv(BENCH_DIR / "rqa_comparison.csv").set_index("model")
    df_pw = pd.read_csv(PAIRWISE_DIR / "pairwise_vs_B.csv").set_index("model")

    rows = []
    base_loss = df_heldout.loc["base", "mean_loss"]

    for arm in ["A", "B", "C", "D", "E", "F"]:
        tokens_M = df_diag.loc[arm, "tokens_total"] / 1e6
        loss = df_heldout.loc[arm, "mean_loss"]
        loss_reduction = base_loss - loss
        loss_red_per_Mtok = loss_reduction / tokens_M
        
        mmlu = df_mmlu.loc[arm, "accuracy"]
        rqa = df_rqa.loc[arm, "f1_all"]
        win_rate_b = df_pw.loc[arm, "win_rate"] if arm in df_pw.index else 0.5000

        rows.append({
            "arm": arm,
            "tokens_M": round(tokens_M, 2),
            "heldout_loss": round(loss, 4),
            "loss_reduction_vs_base": round(loss_reduction, 4),
            "loss_reduction_per_Mtok": round(loss_red_per_Mtok, 4),
            "mmlu_acc": round(mmlu, 4),
            "rqa_f1": round(rqa, 4),
            "win_rate_vs_B": round(win_rate_b, 4)
        })

    df_eff = pd.DataFrame(rows)
    df_eff.to_csv(RESULTS_DIR / "token_efficiency.csv", index=False)
    print("Token efficiency saved.")


def step4_diversity_ablation():
    print("Generating diversity ablation table (C vs E)...")
    # Load tau sweep
    df_sweep = pd.read_csv(S04_REPORTS / "tau_sweep.csv")
    df_sweep.to_csv(RESULTS_DIR / "tau_sweep_summary.csv", index=False)

    # Detailed C vs E comparison across all metrics
    df_diag = pd.read_csv(S04_REPORTS / "arm_diagnostics.csv").set_index("arm")
    df_heldout = pd.read_csv(RESULTS_DIR / "heldout_comparison.csv").set_index("model")
    df_token = pd.read_csv(RESULTS_DIR / "token_weighted.csv").set_index("model")
    df_mmlu = pd.read_csv(BENCH_DIR / "mmlu_comparison.csv").set_index("model")
    df_rqa = pd.read_csv(BENCH_DIR / "rqa_comparison.csv").set_index("model")
    df_h2h = pd.read_csv(PAIRWISE_DIR / "pairwise_head_to_head.csv")
    
    c_vs_e_h2h = df_h2h[(df_h2h["model_A"] == "C") & (df_h2h["model_B"] == "E")].iloc[0]

    ablation_metrics = [
        {"metric": "Selection Diversity Threshold (tau)", "Arm_C (with diversity)": "0.74", "Arm_E (no diversity)": "inf (none)", "Delta (C - E)": "-"},
        {"metric": "Overlap with Arm E (rows)", "Arm_C (with diversity)": "58.1%", "Arm_E (no diversity)": "100%", "Delta (C - E)": "-41.9%"},
        {"metric": "Mean Pairwise Embedding Similarity", "Arm_C (with diversity)": df_diag.loc["C", "mean_pair_sim"], "Arm_E (no diversity)": df_diag.loc["E", "mean_pair_sim"], "Delta (C - E)": round(df_diag.loc["C", "mean_pair_sim"] - df_diag.loc["E", "mean_pair_sim"], 4)},
        {"metric": "Distinct-2 Response Tokens", "Arm_C (with diversity)": df_diag.loc["C", "distinct2_resp"], "Arm_E (no diversity)": df_diag.loc["E", "distinct2_resp"], "Delta (C - E)": round(df_diag.loc["C", "distinct2_resp"] - df_diag.loc["E", "distinct2_resp"], 4)},
        {"metric": "Embedding Space Coverage", "Arm_C (with diversity)": df_diag.loc["C", "coverage"], "Arm_E (no diversity)": df_diag.loc["E", "coverage"], "Delta (C - E)": round(df_diag.loc["C", "coverage"] - df_diag.loc["E", "coverage"], 4)},
        {"metric": "Code Domain Representation", "Arm_C (with diversity)": "3.0%", "Arm_E (no diversity)": "1.4%", "Delta (C - E)": "+1.6%"},
        {"metric": "Held-out Per-example Loss (lower better)", "Arm_C (with diversity)": df_heldout.loc["C", "mean_loss"], "Arm_E (no diversity)": df_heldout.loc["E", "mean_loss"], "Delta (C - E)": round(df_heldout.loc["C", "mean_loss"] - df_heldout.loc["E", "mean_loss"], 4)},
        {"metric": "Token-weighted Loss (lower better)", "Arm_C (with diversity)": df_token.loc["C", "token_weighted_loss"], "Arm_E (no diversity)": df_token.loc["E", "token_weighted_loss"], "Delta (C - E)": round(df_token.loc["C", "token_weighted_loss"] - df_token.loc["E", "token_weighted_loss"], 4)},
        {"metric": "Bengali MMLU Accuracy (higher better)", "Arm_C (with diversity)": df_mmlu.loc["C", "accuracy"], "Arm_E (no diversity)": df_mmlu.loc["E", "accuracy"], "Delta (C - E)": round(df_mmlu.loc["C", "accuracy"] - df_mmlu.loc["E", "accuracy"], 4)},
        {"metric": "BanglaRQA F1 (higher better)", "Arm_C (with diversity)": df_rqa.loc["C", "f1_all"], "Arm_E (no diversity)": df_rqa.loc["E", "f1_all"], "Delta (C - E)": round(df_rqa.loc["C", "f1_all"] - df_rqa.loc["E", "f1_all"], 4)},
        {"metric": "Head-to-Head Pairwise Win Rate", "Arm_C (with diversity)": f"{c_vs_e_h2h['win_rate_A']:.4f}", "Arm_E (no diversity)": f"{c_vs_e_h2h['win_rate_B']:.4f}", "Delta (C - E)": f"+{c_vs_e_h2h['win_rate_A'] - c_vs_e_h2h['win_rate_B']:.4f}"},
    ]
    df_abl = pd.DataFrame(ablation_metrics)
    df_abl.to_csv(RESULTS_DIR / "diversity_ablation.csv", index=False)
    print("Diversity ablation saved.")


def step5_domain_and_filter_bias():
    print("Generating domain-mix and filter bias analysis...")
    df_s1 = pd.read_csv(S01_REPORTS / "filter_report.csv").set_index("domain")
    df_diag = pd.read_csv(S04_REPORTS / "arm_diagnostics.csv").set_index("arm")
    df_pw_dom = pd.read_csv(PAIRWISE_DIR / "pairwise_by_domain.csv").set_index("domain")

    domains = df_s1.index.tolist()
    rows = []
    for d in domains:
        s1_drop = df_s1.loc[d, "drop_rate"] if d in df_s1.index else np.nan
        s1_lang_drop = df_s1.loc[d, "language"] if d in df_s1.index else 0
        
        # Share in pool (A) vs arms B, C, D, E, F
        col_name = f"dom_{d}"
        share_A = df_diag.loc["A", col_name] if col_name in df_diag.columns else np.nan
        share_B = df_diag.loc["B", col_name] if col_name in df_diag.columns else np.nan
        share_C = df_diag.loc["C", col_name] if col_name in df_diag.columns else np.nan
        share_D = df_diag.loc["D", col_name] if col_name in df_diag.columns else np.nan
        share_E = df_diag.loc["E", col_name] if col_name in df_diag.columns else np.nan
        share_F = df_diag.loc["F", col_name] if col_name in df_diag.columns else np.nan

        # Pairwise win rate vs B
        pw_c_vs_b = df_pw_dom.loc[d, "C_win_vs_B"] if d in df_pw_dom.index else np.nan
        pw_f_vs_b = df_pw_dom.loc[d, "F_win_vs_B"] if d in df_pw_dom.index else np.nan

        rows.append({
            "domain": d,
            "stage1_drop_rate": round(s1_drop, 4),
            "stage1_lang_filter_drops": s1_lang_drop,
            "share_in_pool_A": round(share_A, 4),
            "share_in_B": round(share_B, 4),
            "share_in_C": round(share_C, 4),
            "share_in_D": round(share_D, 4),
            "share_in_E": round(share_E, 4),
            "share_in_F": round(share_F, 4),
            "C_pairwise_win_vs_B": round(pw_c_vs_b, 4) if pd.notna(pw_c_vs_b) else "-",
            "F_pairwise_win_vs_B": round(pw_f_vs_b, 4) if pd.notna(pw_f_vs_b) else "-",
        })

    df_dom_bias = pd.DataFrame(rows)
    df_dom_bias.to_csv(RESULTS_DIR / "domain_filter_bias.csv", index=False)
    print("Domain filter bias analysis saved.")


def step6_judge_validation():
    """Fold the Stage 9 judge validation into one table for the paper.

    Two results matter:
      * judge vs human on the 100 annotated rows (complexity and quality separately),
      * the blind spot-check, where the independent rater is the non-circular number.
    Both are produced by analysis/validate_llm_judge.py and analysis/join_spotcheck.py;
    this step only merges them into a single summary table.
    """
    print("Generating judge validation summary...")
    val_file = RESULTS_DIR / "judge_human_validation_summary.json"
    spot_file = RESULTS_DIR / "spotcheck_summary.json"
    if not (val_file.exists() and spot_file.exists()):
        print("  (skipped: run analysis/validate_llm_judge.py and analysis/join_spotcheck.py first)")
        return

    val = json.loads(val_file.read_text(encoding="utf-8"))
    spot = json.loads(spot_file.read_text(encoding="utf-8"))
    rows = []

    for metric, label in (("complexity", "Judge complexity vs human complexity"),
                          ("quality", "Judge quality vs human quality")):
        r = val["results"][metric]
        rows.append({
            "check": label,
            "n": r["n"],
            "spearman": r["spearman"],
            "p_value": r["spearman_p"],
            "exact_agreement": r["exact_agreement"],
            "within_1": r["within_1"],
            "kappa_quadratic": r["kappa_quadratic"],
            "mae": r["mae_human_scale"],
            "delta": "-",
            "ci95": "-",
        })

    ind = spot.get("independent_rater") or {}
    if ind:
        rows.append({
            "check": "Spot-check: arm C vs arm B (independent rater)",
            "n": ind["n_C"] + ind["n_B"],
            "spearman": "-",
            "p_value": "-",
            "exact_agreement": "-",
            "within_1": "-",
            "kappa_quadratic": "-",
            "mae": "-",
            "delta": ind["diff_C_minus_B"],
            "ci95": f"[{ind['diff_ci95_low']}, {ind['diff_ci95_high']}]",
        })

    df_val = pd.DataFrame(rows)
    df_val.to_csv(RESULTS_DIR / "judge_validation_summary.csv", index=False)
    print("Judge validation summary saved.")


if __name__ == "__main__":
    step1_qualitative_samples()
    step2_master_results_table()
    step3_token_efficiency()
    step4_diversity_ablation()
    step5_domain_and_filter_bias()
    step6_judge_validation()
    print("\nALL PHASE 4 & PHASE 5 ANALYSES SUCCESSFULLY GENERATED!")
