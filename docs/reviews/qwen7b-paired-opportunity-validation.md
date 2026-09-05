# Paired opportunity mapping validation

Run: review of a post-preflight-failure, pre-scientific-outcome validation amendment for the Consolidation opportunity gate. The run and exact response are recorded in `docs/EXPERIMENT_REGISTRY.md` under `20260905T052025Z-9bc918b414ff-paired-opportunity`.

Observation: the image-free target-statement construction combines a disease assertion with a question about an unprovided radiograph. It yielded one wrong-sign margin. Its yes/no token identities match the accepted Mass validation, raw and semantic margins agree, and probabilities match their sigmoid. The rejection follows the registered validator; no implementation defect was established. The cohort was bound, but neither the scientific CSV nor its temporary file exists.

The validation standard uses the pre-existing four generic statements from `qwen_answer_encoding.STATEMENTS`, the question `Is the finding present?`, and the accepted explicit answer instructions from `qwen_mass_prompt_specificity.instruction`. Consolidation uses `yes_no` validation instructions while scientific scoring remains `standard`. Mass uses the two registered A/B mappings. Every case must have the expected strict sign. Token identity, exact clean repeat, throughput, source/model and terminal checks remain in force.

This construction tests generic output-symbol semantics directly. Its use is failure-informed but precedes all paired scientific outcomes; it is a separately established measurement standard, not independent statistical confirmation. Disease-specific behavior is measured by the fixed paired-image endpoint. The original text-only response remains a valid recorded observation about that construction.

The scientific prompt, clinical score, fixed ordered 100-pair cohort, selection seed, endpoint, bootstrap and opportunity threshold remain unchanged. Recovery binds patient and image order and complete source records to the original selection receipt; its execution receives separate immutable provenance.

Gate decision: independent internal design and implementation review `PASS`. All 30 focused CPU tests passed, including exact mapping parity, reference-cohort identity and full metadata drift rejection, CPU-only imports and both mocked concept runs. Pinned amendment review precedes dispatch.

The first pinned review accepted the design and implementation and requested tracked provenance for this registration and the failed-run ledger entry. Its closure review evaluates those exact records. The semantic-mapping criterion remains strict sign agreement on the four pre-existing cases; the recorded reference margins are provenance for the established test, while exact-repeat verification is performed on the clean validation image.

Next step: does the fixed Consolidation cohort have positive paired behavioral opportunity under a validated output mapping? Execute the registered image comparison after amendment acceptance.

Evidence: original run `artifacts/preflight.json`, `artifacts/registered-pairs.json` and `metadata.env`; accepted semantic mapping in Mass run `20260905T015725Z-1c9820d7b576-mass-prompts`; protocol `docs/RESEARCH_PLAN.md#qwen-paired-behavioral-opportunity`.
