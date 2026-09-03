#!/usr/bin/env python3
"""Generate Tables 1--2 from the accepted paper-data snapshot."""

from __future__ import annotations

import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
data = json.loads((PAPER / "data" / "accepted_results.json").read_text(encoding="utf-8"))


def interval(values: list[float]) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def stack(top: str, bottom: str) -> str:
    return rf"\shortstack{{{top} \\ \mbox{{{bottom}}}}}"


def main() -> None:
    locus_labels = {
        "model.vision_tower.encoder.layers.22": "CLIP L22",
        "model.visual.blocks.31": "Qwen B31",
    }
    rows = []
    for item in data["cells"]:
        spread = item["control_spread"]
        auroc = f"{item['auroc']:.4f}"
        control_mean = f"{item['control_mean']:.4f}"
        control_spread = f"({spread['p05']:.4f}--{spread['p95']:.4f})"
        selectivity = f"{item['selectivity']:.4f}"
        rows.append(
            f"{item['model']} & {item['concept']} & {locus_labels[item['module']]} & "
            f"{stack(auroc, interval(item['auroc_ci95']))} & "
            f"{stack(control_mean, control_spread)} & "
            f"{stack(selectivity, interval(item['selectivity_ci95']))} \\\\"
        )
    table1 = r"""\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Controlled decodability at the exact final visual block: CLIP \texttt{encoder.layers.22} for LLaVA and \texttt{model.visual.blocks.31} for Qwen. Intervals are 95\% patient-cluster bootstrap intervals over 2,000 draws. The control column gives the mean AUROC over 20 recurring-type random-label maps and their 5th--95th percentile spread.}
\label{tab:decoding}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllccc@{}}
\toprule
Model & Concept & Locus & \shortstack{AUROC \\ {[95\% CI]}} & \shortstack{Control mean \\ {(5--95\%)}} & \shortstack{Selectivity \\ {[95\% CI]}} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""

    gate_rows = []
    rung_rows = []
    for item in data["cells"]:
        p = item["primary"]
        random_bound = f"{p['random_p95']:.4f}"
        rung = "Clinical directions" if item["selective_cell"] else "Random/sham"
        gate_rows.append(
            f"{item['model']} & {item['concept']} & Original & 200 & "
            f"{p['alpha']:+.2f} & {p['concept_consistent_change']:.4f} & {random_bound} & "
            f"{item['maximum_absolute_sham_effect']:.4f} \\\\"
        )
        rung_rows.append(
            f"{item['model']} & {item['concept']} & Original & --- & {rung} \\\\"
        )

    spec = data["specificity"]
    gate_rows.append(
        f"Qwen2.5-VL-7B & Effusion & Independent & {spec['n_eval']} & {spec['alpha']:+.2f} & "
        f"{spec['direction_effects']['concept']:.4f} & {spec['random_effect_p95']:.4f} & "
        f"{spec['absolute_sham_effect']:.4f} \\\\"
    )
    rung_rows.append(
        f"Qwen2.5-VL-7B & Effusion & Independent & {spec['primary']['margin']:.4f} "
        f"[{spec['primary']['ci95'][0]:.4f}, {spec['primary']['ci95'][1]:.4f}] & "
        f"Clinical directions \\\\"
    )
    table2 = r"""\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\caption{Registered intervention outcomes. Original cohorts select the maximum concept-consistent effect over the six controlled nonzero doses; their clinical-direction endpoint was not prospectively familywise. The independent cohort locks $\alpha=+0.25$ and recomputes the maximum over five fixed clinical directions inside each of 5,000 patient-bootstrap replicates. ``Stopped at'' names the first evidentiary rung not cleared.}
\label{tab:gates}
\textit{Generic-control outcomes}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrccc@{}}
\toprule
Model & Concept & Cohort & $N$ & $\alpha$ & Target $\Delta$ & Random p95 & $|$sham$|$ \\
\midrule
""" + "\n".join(gate_rows) + r"""
\bottomrule
\end{tabular*}
\medskip

\textit{Clinical-direction outcomes}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllcl@{}}
\toprule
Model & Concept & Cohort & Clinical margin [95\% CI] & Stopped at \\
\midrule
""" + "\n".join(rung_rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    (PAPER / "tables" / "table_decoding.tex").write_text(table1, encoding="utf-8")
    (PAPER / "tables" / "table_gates.tex").write_text(table2, encoding="utf-8")
    print(PAPER / "tables" / "table_decoding.tex")
    print(PAPER / "tables" / "table_gates.tex")


if __name__ == "__main__":
    main()
