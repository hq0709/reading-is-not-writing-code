# LLaVA readout diagnostic design review

Run: prospective design in `docs/RESEARCH_PLAN.md#llava-effusion-readout-diagnostic`, supported by metadata-only allocation checks in `docs/LLAVA_READOUT_DIAGNOSTIC_FEASIBILITY.md`.

Observation: independent scientist reviewer `01a07340-34b7-7cc0-9057-6067780a98bd` passed the design-to-implementation gate. The 700 independent index-donor pairs, six fixed prompt conditions, index-label image-advantage estimands, separate frozen-reader replication, exact scoring, preflight and resource limits address the registered question. The reviewer recommended the unstudentized maximum absolute centered bootstrap deviation as a single simultaneous-interval radius; this construction is fixed in the prospective protocol. It retains informative comparisons when another contrast is exactly constant, at the cost of sensitivity when a noisier contrast sets the radius.

Gate decision: design-to-implementation `PASS`. Executable, pinned registration and terminal result reviews remain separate gates. No new model outcome or activation was used in design.

Next step: can the executable reproduce the fixed allocation, complete paired score grid and six joint intervals? Implement the diagnostic, test it with independent reference fixtures and obtain pinned registration review before evaluating the new images.
