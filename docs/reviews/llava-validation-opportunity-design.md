# LLaVA validation opportunity design review

Run: independent native review of `docs/RESEARCH_PLAN.md#llava-validation-intervention-opportunity`, its preparation note, the research-writing contract and the relevant readout/intervention implementation. Read-only metadata feasibility is recorded in `docs/LLAVA_VALIDATION_COHORT_FEASIBILITY.md`.

Observation: the 700-patient calibration and 100-patient write allocation is feasible and separated from implementation preflight and the accepted test cohort. The accepted clinical coefficients support AUROC ranking without refitting. Twenty shared train-only type controls supply validation selectivity. Fixed questions, doses, random/sham generation, qualification-dependent complete grids and exploratory routing define an implementable screen. The maximum grid is 31,200 outcomes; the allocation budget is two GPU-hours and two wall-clock hours.

Gate decision: design-to-implementation `PASS`. This gate authorizes implementation and local verification. Formal pinned registration review follows executable-source and test evidence before GPU dispatch.

Next step: implement exact allocation and joint bootstrap replay, reader/capability qualification, controlled bidirectional scoring and the persisted route decision; verify with synthetic and real-interface fixtures.
