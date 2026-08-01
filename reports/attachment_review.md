# Attachment review

> **Scope revision, 2026-08-01:** attachment findings and scientific blockers
> remain historical evidence, but later user direction superseded the
> administrative requirement for a full-root inventory, per-file hash ledger,
> CAS/bundle authority rollover and repeated environment audit. Use
> `reports/data_inventory/summary.md` and `datasets/registry/` for current data
> status.

Status: **semantic and visual review complete; hardened execution-provenance supplement terminally
failed/incomplete on the unchanged renderer path; scientific conflicts and external submission
blockers recorded**

The source attachments were opened read-only through Slurm job `918740` in the registered `eeg2025` environment. That job produced the manifest, performed archive safety checks, and extracted review material into hash-versioned directories. The first PDF path was unavailable because that environment contains neither Poppler utilities nor `pypdf`; the failure was retained rather than hidden. Supplemental Slurm job `918742` used the already registered `icml` environment's audited PyMuPDF 1.26.5 on a CPU allocation, without installing or changing anything, to extract all PDF text, links, metadata, annotations, and page images.

After the semantic review, an independent control-plane audit found that the first-generation
submit/runtime/attachment scripts did not fully bind the exact request, environment audit, source
snapshot, allocation, and no-replace artifact closure now required. Jobs `918740` and `918742`
therefore remain useful read-only extraction provenance but are **provenance-incomplete** for the
strict execution contract. The hashes, full manual reading, and visual inspection below remain
valid observations. The later hardened chain was executed and its canary failed closed as recorded
below; the unchanged renderer path is now terminally incomplete rather than awaiting another
identical retry. No scientific work is allowed to treat that supplement as successful.

## Attachment manifest and read status

| Attachment | Bytes | SHA-256 | Media type | Final review state |
|---|---:|---|---|---|
| `ARTIFACTS.md` | 2,441 | `d2fb2be71d7b4974d2d7772c8b09f72daccb03a627473f85bd677d1baa7f9a11` | text/plain | fully read |
| `CSPD_TAAS_manuscript_source.zip` | 1,144,789 | `e377315852679a7b1412b1acf5dd27072cd50de7b84c5228c6d8823a22a505fb` | application/zip | safely extracted; all text and relevant visuals read |
| `pdf/CSPD_TAAS_review_draft.pdf` | 1,018,848 | `e86263106bec8e27ad589c8e745e83c75e0054e21388d39526beb4398d14c332` | application/pdf | all 54 pages read and visually inspected; zero annotations observed |

No separate task-upload directory was present under `/mnt`, `/mnt/data`, or an obvious `/tmp` upload path. Older June documents in the worktree are repository history rather than current task attachments unless referenced by the three files above; they are covered by the subsequent repository audit, not silently promoted to attachment authority.

The ZIP contained 39 members. Its total compression ratio was about 1.304; it contained no absolute member path, traversal component, symbolic-link member, compression-bomb trigger, or nested archive. Extraction was performed by writing each validated regular member into a new `.partial-918740` directory and atomically publishing the hash-versioned review directory; no repository source file was overwritten.

The PDF is unencrypted, has 54 pages, and has no embedded files. PyMuPDF produced 54 page-text files and 54 110-DPI renderings. Every page, table, displayed formula, three algorithms, reference section, header/footer, page number, and the five generated figures was inspected. No gross clipping or unreadable overlap was observed. The visible red `TBD` fields, empty result panels, pending author metadata, and large intentional whitespace on the final reference page are not empirical evidence. All 54 page annotation lists were empty.

## Authority boundary

The bundle's `notes/README.md` states that the current LaTeX manuscript and `notes/server_codex_experiment_instruction.md` jointly form the executable scientific specification; disagreement must be recorded and the affected downstream work stopped. Other dated notes are provenance only and must not restore the old `B_omega A` notation, old mask semantics, an unconditional `rho=0`, a six-question ladder, or immediate backup selection.

`SUBMISSION_CHECKLIST.md` is a release checklist, not proof of completion. The manuscript contains 230 explicit `\TBD{...}` tokens. It describes a falsification protocol and empty result contracts, not completed experiments. The empirical, authorship, ethics, editorial, archive, and provenance fields remain open; the manuscript must not be submitted or populated with illustrative, inherited, or manually transcribed values.

`ARTIFACTS.md` contains a producer-supplied validation record. This review independently confirmed the two attachment sizes/hashes, PDF page count, visible five figures, visible placeholder-only result tables, and page readability. It did **not** independently rebuild the manuscript, verify every bibliography source against a primary source, reproduce the stated citation/static-token audits, obtain editor approval, or close any scientific gate. Those claims remain attachment provenance rather than server experiment evidence.

The PowerShell/local-`pip --target` build instructions describe the attachment's Windows build workspace. They are not executable server instructions. On this server all build/audit/test payloads remain Slurm-only and may use only the two registered environments; no package installation or third environment is authorized. Likewise, the checklist's archival language authorizes local preparation only, not a remote push/upload.

## Executable attachment requirements

The full mapping is in `reports/requirement_traceability.csv`. Principal attachment-specific requirements are:

- Treat all empirical numbers, figures, tables, abstract/results/limitations/conclusion text as admissible only when generated from the same provenance-bound records. Never inherit old code outputs or fill placeholders illustratively.
- Complete access, license, original-consent, local secondary-use, raw inventory/hash, participant/session, exclusion, channel-map, support/query, ethics, AI-use, authorship, editorial, and archive ledgers before the corresponding claim or submission action.
- Freeze all five gate thresholds and every data/model/operator/attenuation/guidance/sampler/seed/decoder margin before any outer-test result is inspected.
- Implement `NULL` as an explicit same-query population-posterior short circuit before context-dependent computation; an unconditional prior is inadmissible.
- Keep B1–B6 disabled; never use held-out query targets to select a family; native SGEYESUB and oracle-span subtraction must remain distinct; all strong information/task-matched and operator-source controls are claim blockers when absent.
- Literature cited for lag, filter-bank/covariance, graph, CCA/GED, Wiener, robust, or SGEYESUB designs is motivation only. RPCA remains exploratory and no citation proves the proposed backup works.

## Scientific conflicts that block downstream mechanisms

These are not silently resolved. They block Stage B, P0, G1–G5, and any backup execution while data/repository audits and non-scientific scaffolding may continue.

### CONFLICT-SCI-001: population energy versus NULL/mask isolation

The manuscript's `04_method.tex:92-118,143-149` defines the same query-derived attenuation `a_tau` for both population and context precision/energy (`j in {0,C}`). Yet `04_method.tex:219` and Algorithm 3 require `NULL/rho=0` to short-circuit before obtaining attenuation or constructing context state, matching the server contract's zero calls to attenuation/mask/calibration components. The manuscript does not specify an auditable construction of `E0` that both uses the shared `a_tau` formula and satisfies the no-attenuation NULL call-count rule. Choosing either interpretation would change scientific semantics. Required resolution: explicitly separate any population-base observation mechanism from individualized `a_tau`/mask calls, or amend the NULL contract and its tests; no implementation choice is made here.

### CONFLICT-SCI-002: real-EEG evidence enters the ordered gates at different points

The manuscript's `06_experiments.tex:9-25,55-75` places semi-simulated oracle mechanism/specificity in RQ1/RQ2 and introduces natural-recording preservation mainly in RQ4. The server instruction requires G1 to combine the semi-simulated mechanism with real-EEG attenuation/preservation/task evidence, G2 personalization to use all frozen real-EEG outer folds, and G3 diffusion necessity to use complete real-EEG outer evaluation. The manuscript could therefore label RQ1/RQ2/RQ3 passed before evidence the server contract makes mandatory for G1/G2/G3. Gate thresholds, estimands, and data roles cannot be frozen until the two documents are reconciled.

### CONFLICT-SCI-003: backup activation order

The manuscript's `04_method.tex:217`, `06_experiments.tex:24-25,94-98`, Appendix E, and the checklist allow language that opens one prospectively amended backup after P0 passes the oracle mechanism and specificity gates and exhibits a diagnosed failure. The server instruction requires **all G1–G5** to pass before any B1–B6 fit/search/comparison. Some attachment phrases can be read as necessary but not sufficient conditions, but other passages describe RQ1/RQ2 as the opening condition. The implementation therefore applies the safe common subset—every backup stays a disabled stub—but cannot authorize a backup until the authority conflict is resolved.

## Non-scientific conflicts and resolved handling

- The Windows PowerShell/local-install recipe conflicts with server execution constraints. Resolution: do not execute it; use Slurm and existing registered environments only.
- The requested final archive could imply an external upload, while this task forbids push/upload without explicit authority. Resolution: prepare local manifests only; external archive is blocked.
- The worktree's old README/PLAN/result claims describe a different SADDPM/BCI-IV/V100 workflow. They are not attachment authority and are not accepted as evidence; the repository audit records their incompatibility separately.
- Backup literature search credentials were unavailable when the attachment was made. That is search provenance, not an experiment blocker and not evidence of backup efficacy.

## Open blockers

- Resolve the three scientific conflicts above before population-base semantics, preregistered gates, or backup routing are implemented or run.
- Finish the scheduled full data-root inventory, data access/license/consent audit, sample readability checks, real-EEG field audit, and immutable outer/support/query split manifests.
- Freeze every `TBD-PREREG` threshold, sample minimum, confidence rule, multiplicity family, seed set, training budget, checkpoint rule, and operating margin without looking at outer-test results.
- Obtain human/editorial decisions for author order, contributions, funding, conflicts, ethics/consent, CCS, ScholarOne format, ACM AI disclosure, and revision-track status before submission.
- Independently audit primary bibliography sources and reproduce the manuscript build/static checks on an authorized server path if/when submission work is in scope.
- Do not push, publish, release, upload, download restricted data, or modify the shared environments under the present authority.

## Hardened execution-provenance supplement status

The prospective shared renderer audit chain completed in jobs `918899`–`918902`, and fresh parent
job `918903` closed the current three attachment sources. Its first registered top-level PDF canary
`918904` failed closed after publishing byte-matching page 1–8 text and renders, before page 9
publication. Because its v1 failure record could not uniquely identify the final operation, a
bounded diagnostic bundle was audited in jobs `918907`–`918915`; fresh parent `918916` then closed
the same three source hashes. Diagnostic canary `918917` proved that PyMuPDF emitted a one-line,
40-byte warning immediately after extracting page 8 text. It had fully validated pages 1–7, found
no registered credential pattern, retained no raw warning, and pinned its failure record as
`0e0eba84…7e145`. It has no COMPLETE marker. The five sibling embedded PDFs were not submitted,
and the warning rule was not weakened or ignored.

This failed hardened canary does not supersede the earlier 54-page Slurm semantic/visual review;
the parent ZIP's equivalent PNG/SVG figure members were also available to that review. It does mean
that the hardened execution-provenance supplement remains explicitly incomplete and cannot
authorize downstream scientific claims. The failure is terminal for the unchanged renderer path;
read-only data inventory may proceed independently, while population base and every scientific
gate remain blocked for the separately recorded semantic and data reasons.
