# LLaVA-Med runtime feasibility: server assignment

## Run and ownership

Complete the CPU-only preparation in `docs/RESEARCH_PLAN.md#medical-domain-runtime-feasibility`, using `docs/MEDICAL_MODEL_FEASIBILITY.md` as the verified metadata baseline. After clean pushed synchronization and launcher acceptance, server Codex owns `research/llava-med-runtime-feasibility`; the local project task supervises read-only. The paper repository stays with its independent task.

## Observation to establish

Read the pinned official model/index/tokenizer configuration and official implementation, and inspect the fixed runtime's relevant classes. Identify an exact viable loading path or the concrete interface preventing it. Trace every learned tensor family, image preprocessing, patch selection, image-token insertion, Mistral conversation and first-answer scoring. Treat source inspection as feasibility evidence; proposed equivalence tests remain planned until a separately authorized implementation stage.

## Gate decision and delivery

Use at most one hour of CPU-only preparation from task start, small public source/configuration/tokenizer/index files and existing runtime introspection. Preserve the fixed environment and accepted artifacts. This assignment includes no model-weight staging, package installation, adapter implementation, model instantiation, GPU use or patient evaluation.

Write `docs/LLAVA_MED_RUNTIME_FEASIBILITY.md` with the source-grounded loading path, unresolved mappings, a small executable verification design and staging estimate, or a concrete `BLOCKED` disposition. Obtain independent internal read-only review; at a clean pushed checkpoint use the pinned read-only reviewer for the preparation report. Preserve its actual identity, effort and unchanged-checkout receipt. This gate assesses implementation feasibility, not clinical capability.

Record the result and one next question in `docs/RESEARCH_STATE.md`. Integrate owned documentation into main, push, remove the completed temporary branch and return a clean checkout. On a time or source-access limit, preserve the partial findings and exact remaining decision. All server files and processes stay below `/home/qingchan/`; project source synchronization uses GitHub.

## Next step

Can the accepted loading path support a prospectively specified asset and validation-only measurement gate? Return the preparation for local registration; this assignment ends before that execution.
