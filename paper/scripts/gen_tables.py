#!/usr/bin/env python3
"""Generate Tables 1--2 from the accepted paper-data snapshot."""

from __future__ import annotations

import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
data = json.loads((PAPER / "data" / "accepted_results.json").read_text(encoding="utf-8"))


def interval(values: list[float]) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def main() -> None:
    rows = []
    for item in data["cells"]:
        spread = item["control_spread"]
        rows.append(
            f"{item['model']} & {item['concept']} & {item['module'].replace('_', r'\_')} & "
            f"{item['auroc']:.4f} {interval(item['auroc_ci95'])} & "
            f"{item['control_mean']:.4f} ({spread['p05']:.4f}--{spread['p95']:.4f}) & "
            f"{item['selectivity']:.4f} {interval(item['selectivity_ci95'])} \\\\"
        )
    table1 = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\caption{Controlled decodability at the exact final visual block. Intervals are 95\% patient-cluster bootstrap intervals over 2,000 draws. The control column gives the mean AUROC over 20 recurring-type random-label maps and their 5th--95th percentile spread.}
\label{tab:decoding}
\begin{tabular}{lllccc}
\toprule
Model & Concept & Consumed module & AUROC [95\% CI] & Control mean (5--95\%) & Selectivity [95\% CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""

    gate_rows = []
    for item in data["cells"]:
        p = item["primary"]
        random_bound = f"{p['random_p95']:.4f}"
        rung = "Clinical directions" if item["selective_cell"] else "Random/sham"
        gate_rows.append(
            f"{item['model']} & {item['concept']} & Original images & 200 & "
            f"{p['alpha']:+.2f} & {p['concept_consistent_change']:.4f} & {random_bound} & "
            f"{item['maximum_absolute_sham_effect']:.4f} & --- & "
            f"{'Clinical directions (next gate)' if item['selective_cell'] else rung} \\\\"
        )
    spec = data["specificity"]
    gate_rows.append(
        f"Qwen2.5-VL-7B & Effusion & Independent patients & {spec['n_eval']} & {spec['alpha']:+.2f} & "
        f"{spec['direction_effects']['concept']:.4f} & {spec['random_effect_p95']:.4f} & "
        f"{spec['absolute_sham_effect']:.4f} & {spec['primary']['margin']:.4f} "
        f"[{spec['primary']['ci95'][0]:.4f}, {spec['primary']['ci95'][1]:.4f}] & Clinical directions \\\\"
    )
    table2 = r"""\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\caption{Registered intervention outcomes. Original cohorts select the maximum concept-consistent effect over the six controlled nonzero doses; their clinical-direction endpoint was not prospectively familywise. The independent cohort locks $\alpha=+0.25$ and recomputes the maximum over five fixed clinical directions inside each of 5,000 patient-bootstrap replicates. ``Stopped at'' names the first evidentiary rung not cleared.}
\label{tab:gates}
\begin{tabular}{lllrrccccl}
\toprule
Model & Concept & Cohort unit & $N$ & $\alpha$ & Concept effect & Random p95 & $|$sham$|$ & Clinical margin [95\% CI] & Stopped at \\
\midrule
""" + "\n".join(gate_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (PAPER / "tables" / "table_decoding.tex").write_text(table1, encoding="utf-8")
    (PAPER / "tables" / "table_gates.tex").write_text(table2, encoding="utf-8")
    print(PAPER / "tables" / "table_decoding.tex")
    print(PAPER / "tables" / "table_gates.tex")


if __name__ == "__main__":
    main()
