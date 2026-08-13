"""Claim, scope, editor-consultation, and reviewer maps for V31."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def markdown_table(rows: Iterable[Mapping[str, Any]], columns: list[str]) -> str:
    rows = list(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ") for column in columns) + " |")
    return "\n".join(lines)


def claim_rows() -> list[dict[str, str]]:
    """Return the complete, explicitly narrowed master claim registry."""
    return [
        {
            "claim_id": "C01", "claim_text": "Query-disjoint support contains transferable ocular-corruption information.",
            "evidence_source": "V20 natural support-to-query operator transfer",
            "supporting_result": "N_P=+0.178803; N_W=+0.174162; 15/15 positive; p=1/100001.",
            "contradicting_result": "Transfer of an operator does not establish useful EEG-only temporal estimation or denoising.",
            "scientific_status": "supported",
            "allowed_wording": "Support contains transferable corruption-related information under the registered development protocol.",
            "forbidden_wording": "Support guarantees subject-specific denoising benefit.",
            "manuscript_location": "Introduction; Evaluation protocol; Discussion",
            "reviewer_relevance": "subject mechanism and transductive-setting clarification",
        },
        {
            "claim_id": "C02", "claim_text": "Support improves paired denoising versus a strong population route.",
            "evidence_source": "V25/V26 and V30 common-panel replay",
            "supporting_result": "V30 V25 and V26 correct support improve over population and mean-wrong on paired development.",
            "contradicting_result": "Effects are development-only and do not identify the correct donor reliably.",
            "scientific_status": "partially_supported",
            "allowed_wording": "Some frozen support-conditioned routes improve paired development risk relative to population controls.",
            "forbidden_wording": "Personalization universally improves denoising.",
            "manuscript_location": "Paired utility results",
            "reviewer_relevance": "subject-component ablation and statistics",
        },
        {
            "claim_id": "C03", "claim_text": "The correct support donor is specifically useful.",
            "evidence_source": "V30 15x15 all-donor and lag/shuffle falsification",
            "supporting_result": "Correct support beats mean wrong for V25/V26.",
            "contradicting_result": "Correct top-1 is 1/15; lagged/shuffled controls slightly outperform correct support.",
            "scientific_status": "mixed",
            "allowed_wording": "Correct-context specificity is mixed and not identified by the current controls.",
            "forbidden_wording": "Participant-specific calibration specificity is established.",
            "manuscript_location": "Specificity audit; Limitations",
            "reviewer_relevance": "subject mechanism and negative controls",
        },
        {
            "claim_id": "C04", "claim_text": "Support-aware diffusion is competitive with matched deterministic estimation.",
            "evidence_source": "V26/V27/V29 matched comparisons",
            "supporting_result": "V27 EnergySDEdit and EnergyDET are essentially equivalent; V29 CDM and DET occupy the same narrow range.",
            "contradicting_result": "V26 SDEdit is slightly worse than matched one-step and V25 latent diffusion is uniformly worse.",
            "scientific_status": "partially_supported",
            "allowed_wording": "Selected diffusion and deterministic controls occupy a comparable development range.",
            "forbidden_wording": "Diffusion consistently outperforms deterministic denoisers.",
            "manuscript_location": "Competitive positioning",
            "reviewer_relevance": "strong deterministic baselines",
        },
        {
            "claim_id": "C05", "claim_text": "Diffusion is superior to deterministic estimation.",
            "evidence_source": "V25-V30 frozen comparisons",
            "supporting_result": "No robust supporting result.",
            "contradicting_result": "V25 DIFF-DET is negative for 15/15; V26 one-step is better; V27 is equivalent.",
            "scientific_status": "unsupported",
            "allowed_wording": "Diffusion is evaluated as a competitive stochastic formulation, not a superior estimator.",
            "forbidden_wording": "Diffusion provides superior denoising accuracy.",
            "manuscript_location": "Abstract claim removed; Discussion",
            "reviewer_relevance": "subject-agnostic DDPM and method comparison",
        },
        {
            "claim_id": "C06", "claim_text": "A frozen candidate produces absolute natural ocular-artifact attenuation.",
            "evidence_source": "V30 corrected natural common panel",
            "supporting_result": "V25/V26/V27 have remaining ratio below 1; V27-L0.5 remaining=0.929094.",
            "contradicting_result": "Attenuation is coupled to observation-retention and PSD costs; V29 remaining is above 1.",
            "scientific_status": "partially_supported",
            "allowed_wording": "Some operating points attenuate the evaluator-defined natural ocular reference, with explicit trade-offs.",
            "forbidden_wording": "Natural EEG is cleanly recovered.",
            "manuscript_location": "Natural attenuation-retention trade-off",
            "reviewer_relevance": "natural validity",
        },
        {
            "claim_id": "C07", "claim_text": "The method retains the natural observation in low-EOG intervals.",
            "evidence_source": "V28-corrected metric and V30 natural panel",
            "supporting_result": "V29 retention=0.993272; V27 energy exposes a tunable retention endpoint.",
            "contradicting_result": "The more attenuating V25/V26/V27-L0.5 candidates show substantial correction/PSD cost.",
            "scientific_status": "mixed",
            "allowed_wording": "Observation retention and attenuation form a measured Pareto trade-off.",
            "forbidden_wording": "Low-EOG retention proves neural preservation.",
            "manuscript_location": "Natural trade-off; Metric definitions",
            "reviewer_relevance": "preservation semantics",
        },
        {
            "claim_id": "C08", "claim_text": "ERP, SSVEP, or other physiological/task information is preserved.",
            "evidence_source": "V28 event/task inventory",
            "supporting_result": "None; endpoints are unavailable.",
            "contradicting_result": "Earlier ERP/SSVEP aliases were invalid and removed.",
            "scientific_status": "unavailable",
            "allowed_wording": "Task-valid physiological preservation was not evaluable.",
            "forbidden_wording": "ERP/SSVEP preservation is established.",
            "manuscript_location": "Limitations",
            "reviewer_relevance": "downstream/task-valid preservation",
        },
        {
            "claim_id": "C09", "claim_text": "Support acquisition burden is characterized.",
            "evidence_source": "V31 exact 0/5/10/30/120 s repair",
            "supporting_result": "Exact non-overlapping prefix curves report acquisition span, effective exposure, and encoding cost.",
            "contradicting_result": "V30 duration evidence is superseded because its contract used overlap/future normalization/subsampling.",
            "scientific_status": "supported",
            "allowed_wording": "Support-duration sensitivity is reported under the exact V31 prefix contract.",
            "forbidden_wording": "The invalid V30 duration curve establishes sample efficiency.",
            "manuscript_location": "Computational cost and support burden",
            "reviewer_relevance": "target data amount",
        },
        {
            "claim_id": "C10", "claim_text": "Sampler latency and memory are characterized.",
            "evidence_source": "V30 fixed A100 5/10/25-step benchmark",
            "supporting_result": "Batch-1 DDIM10: V26 37.56 ms, V29 55.41 ms; fixed model footprints and memory reported.",
            "contradicting_result": "Measurements are hardware- and implementation-specific.",
            "scientific_status": "supported",
            "allowed_wording": "Registered A100 latency/memory and step curves are reported.",
            "forbidden_wording": "Deployment latency is universally guaranteed.",
            "manuscript_location": "Computational cost",
            "reviewer_relevance": "steps/latency/memory",
        },
        {
            "claim_id": "C11", "claim_text": "Stored support states create participant-linkage risk.",
            "evidence_source": "V30 development linkage diagnostic",
            "supporting_result": "Context+projector top-1=0.836 and same/different AUROC=0.962.",
            "contradicting_result": "This is not a formal privacy attack or external-cohort audit.",
            "scientific_status": "supported",
            "allowed_wording": "Stored states show substantial development participant-linkage risk.",
            "forbidden_wording": "The representation is anonymous or privacy-safe.",
            "manuscript_location": "Privacy-linkage audit",
            "reviewer_relevance": "privacy risk",
        },
        {
            "claim_id": "C12", "claim_text": "The method is safe for deployment.",
            "evidence_source": "No deployment study",
            "supporting_result": "None.",
            "contradicting_result": "Mixed specificity, retention concern, linkage risk, and no confirmation.",
            "scientific_status": "unsupported",
            "allowed_wording": "No deployment claim is made.",
            "forbidden_wording": "Safe deployment or clinical readiness.",
            "manuscript_location": "Limitations",
            "reviewer_relevance": "scope boundary",
        },
        {
            "claim_id": "C13", "claim_text": "Support effects are stable across sessions.",
            "evidence_source": "Within-session development protocol only",
            "supporting_result": "None beyond registered within-session support-to-later-query separation.",
            "contradicting_result": "Cross-session permanence was not tested.",
            "scientific_status": "unavailable",
            "allowed_wording": "Evidence is limited to the registered within-session setting.",
            "forbidden_wording": "Cross-session stability or permanence.",
            "manuscript_location": "Limitations",
            "reviewer_relevance": "transductive/support setting",
        },
        {
            "claim_id": "C14", "claim_text": "The method generalizes across montages or acquisition protocols.",
            "evidence_source": "Fixed-montage development evidence",
            "supporting_result": "None.",
            "contradicting_result": "No valid additional-montage confirmation has been opened.",
            "scientific_status": "unavailable",
            "allowed_wording": "Cross-montage generalization remains future work.",
            "forbidden_wording": "Cross-montage or universal acquisition generalization.",
            "manuscript_location": "Limitations; Future work",
            "reviewer_relevance": "additional dataset/montage",
        },
    ]


def scope_comparison() -> dict[str, Any]:
    return {
        "recommended_for_AE_consultation": "Scope A",
        "automatic_selection": False,
        "scope_A": {
            "id": "A_audit_centric",
            "title": "Subject-Aware Diffusion for EEG Denoising: Utility, Specificity, Trade-offs, and Privacy under Query-Disjoint Support",
            "scientific_support": "strongest alignment with the complete positive and negative evidence",
            "continuity_with_original_submission": "retains subject-aware diffusion and EEG denoising while withdrawing universal benefit",
            "reviewer_coverage": "high: baselines, controls, statistics, support, steps, latency, and privacy",
            "risk_of_overclaim": "lowest of the two scopes",
            "amount_of_rewrite": "substantial",
            "need_for_additional_experiment": "exact duration repair complete; task-valid physiology and extra montage remain unavailable",
            "TAAS_fit": "requires AE guidance because the contribution becomes an audit plus bounded method evidence",
            "acceptance_risk": "material but evidence-aligned",
        },
        "scope_B": {
            "id": "B_method_centric",
            "title": "Query-Disjoint Support-Conditioned Diffusion for Ocular Artifact Removal in EEG",
            "primary_operating_point": "V27 EnergySDEdit lambda_y=0.5",
            "scientific_support": "absolute attenuation and competitive paired development performance",
            "continuity_with_original_submission": "closer method-centric continuity",
            "reviewer_coverage": "moderate; correct-context specificity remains mixed",
            "risk_of_overclaim": "higher because the operating point trades away observation retention and PSD stability",
            "amount_of_rewrite": "substantial but narrower than Scope A",
            "need_for_additional_experiment": "editorial acceptance of mixed specificity and missing task-valid physiology",
            "TAAS_fit": "plausible only with explicit Pareto and limitation framing",
            "acceptance_risk": "higher than Scope A",
        },
    }


def reviewer_rows() -> list[dict[str, str]]:
    """Map ledger-summarized concerns; original verbatim reviews are unavailable."""
    rows = [
        ("AE", "major reconstruction and acceptable revised scope", "Old dual-method evidence was not independently reproducible.", "Clean-room V20-V31 evidence chain and two explicit scopes.", "Replace old method/results with the selected evidence-aligned scope after AE guidance.", "We completed a clean-room reconstruction and seek guidance on whether the audit-centric scope remains suitable as a major revision.", "requires_AE_guidance"),
        ("Reviewer 1", "strong denoising baselines", "Original comparisons were too weak.", "EEGDfus, STANDARD, strong population DET, matched DET and subject-agnostic diffusion are included.", "Add common-panel absolute tables and distinguish matched from strongest baseline.", "We added strong deterministic and diffusion controls and report absolute, not only relative, outcomes.", "resolved"),
        ("Reviewer 1", "subject-agnostic DDPM", "No clean subject-agnostic diffusion comparator.", "V26 PopSDEdit and V28/V29 population diffusion routes.", "Add explicit population-diffusion rows.", "We now compare support-aware diffusion with frozen subject-agnostic diffusion under matched data and sampling.", "resolved"),
        ("Reviewer 1", "RAW / STANDARD", "Preprocessing contribution was unclear.", "V30 common panel reports RAW/STANDARD references.", "Expose observation and conventional preprocessing references.", "RAW/STANDARD references are now separated from learned methods.", "resolved"),
        ("Reviewer 2", "subject-component ablation", "Subject contribution was not isolated.", "MATCH/POP/all-wrong/lagged/shuffled/mean-context controls.", "Add all-donor and falsification sections.", "The expanded controls show mixed, not established, correct-context specificity.", "resolved"),
        ("Reviewer 2", "statistics and confidence intervals", "Window-level evidence overstated biological sample size.", "Participant-first aggregation and bootstrap CIs.", "Describe participants as biological units and windows/seeds as repeated measurements.", "We replaced window-level interpretation with participant-first summaries.", "resolved"),
        ("Reviewer 2", "support amount", "Calibration burden was not quantified.", "V31 exact 0/5/10/30/120 s prefix repair.", "Replace superseded V30 duration curve with V31 exact contract.", "We corrected overlap and future-normalization errors and report acquisition and effective exposure separately.", "resolved"),
        ("Reviewer 2", "sampling steps and latency", "Diffusion cost was missing.", "V30 5/10/25 steps, A100 latency/memory, batch 1/16.", "Add quality-latency and footprint table.", "We report sampler-step sensitivity and fixed-hardware latency/memory.", "resolved"),
        ("Reviewer 3", "privacy risk", "Subject state may act as an identifier.", "V30 context/projector linkage top-1 and AUROC.", "Add linkage-risk audit and storage size; withdraw anonymity language.", "The states are strongly linkable in development data; we do not claim privacy safety.", "resolved"),
        ("Reviewer 3", "transductive limitation", "Target data usage and deployment setting were unclear.", "Strict query-disjoint early support and EEG-only later query inference.", "Reframe as within-session query-disjoint calibration and state its limitation.", "We use only a disjoint support bank at inference and make no cross-session claim.", "resolved"),
        ("Reviewer 3", "additional dataset or montage", "Generalization evidence was insufficient.", "Natural SGE development complements paired data, but sealed and new montage remain unopened.", "Withdraw cross-montage/generalization claim and ask AE whether this is acceptable.", "We cannot support cross-montage generalization and explicitly withdraw it.", "requires_AE_guidance"),
        ("Reviewer 3", "task-valid physiology", "Artifact attenuation alone cannot establish preservation.", "Observation retention, PSD and covariance are reported; ERP/SSVEP/ERD endpoints unavailable.", "Remove proxy aliases and state N/A.", "Task-valid physiological preservation is unavailable and is not claimed.", "unresolved_but_claim_withdrawn"),
    ]
    columns = ("reviewer", "comment", "original_weakness", "new_evidence", "proposed_manuscript_change", "response_wording", "status")
    return [dict(zip(columns, row)) for row in rows]


def write_scope_and_consultation_reports(report_root: Path, result_root: Path) -> None:
    claims = claim_rows()
    scopes = scope_comparison()
    reviewers = reviewer_rows()
    report_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    claim_columns = ["claim_id", "claim_text", "evidence_source", "supporting_result", "contradicting_result", "scientific_status", "allowed_wording", "forbidden_wording", "manuscript_location", "reviewer_relevance"]
    (report_root / "v31_claim_evidence_matrix.md").write_text(
        "# V31 claim–evidence matrix\n\n" + markdown_table(claims, claim_columns) + "\n"
    )

    scope_a = scopes["scope_A"]
    scope_b = scopes["scope_B"]
    (report_root / "v31_scope_A_audit_centric.md").write_text(
        "# Scope A — audit-centric (recommended for AE consultation)\n\n"
        f"Provisional title: **{scope_a['title']}**\n\n"
        "The revision centers on technical viability plus an explicit audit of utility, mixed specificity, attenuation–retention trade-offs, computational burden, and linkage risk. Contributions are the clean-room query-disjoint protocol; matched population/deterministic controls; all-donor and falsification tests; Pareto analysis; exact support-duration and latency curves; privacy-linkage audit; and claim narrowing. It does not claim universal personalization, unique participant-operator recovery, physiological preservation, or safe deployment.\n"
    )
    (report_root / "v31_scope_B_method_centric.md").write_text(
        "# Scope B — method-centric (alternative)\n\n"
        f"Provisional title: **{scope_b['title']}**\n\n"
        "The operational method is V27 EnergySDEdit at lambda_y=0.5. It provides absolute evaluator-defined attenuation, while the full lambda curve exposes observation-retention and PSD costs. Correct-context specificity is mixed secondary evidence, not a primary claim. This scope has higher acceptance and overclaim risk than Scope A.\n"
    )

    email = """# Draft AE consultation email — NOT SENT

Subject: TAAS-26-0171 major-revision scope consultation

Dear Associate Editor,

Thank you again for allowing a substantial reconstruction of our manuscript while retaining its core topic of subject-aware diffusion for EEG denoising. We have now completed a clean-room, query-disjoint development evaluation rather than carrying forward the original unreproducible evidence.

Five findings define the revised evidence: (1) disjoint support contains transferable ocular-corruption information; (2) several frozen support-conditioned routes provide paired-development gains over population controls; (3) correct-donor specificity is mixed—correct support rarely ranks first and lagged/shuffled controls do not validate unique synchronous coupling; (4) some operating points achieve absolute evaluator-defined natural attenuation, but attenuation trades against low-EOG observation retention and PSD stability; and (5) stored support context/projectors are strongly participant-linkable in our development diagnostic.

We do not ask that these negative or mixed findings be discounted. Instead, we propose an audit-centric major revision titled “Subject-Aware Diffusion for EEG Denoising: Utility, Specificity, Trade-offs, and Privacy under Query-Disjoint Support.” It would retain the subject-aware diffusion topic while explicitly narrowing claims and reporting the full controls, computational burden, and privacy risk.

Would this audit-centric scope be acceptable as the major revision of TAAS-26-0171? If not, would you prefer the narrower method-centric presentation around one transparent attenuation–retention operating point, or should the reconstructed work be treated as a new submission?

We can provide a concise evidence table if helpful. No confirmation cohort has been opened while awaiting scope guidance.

Sincerely,

[Authors]
"""
    (report_root / "v31_AE_consultation_email.md").write_text(email)
    (report_root / "v31_AE_one_page_evidence_summary.md").write_text(
        "# AE one-page evidence summary\n\n"
        "## Original claim\n\nClosed-set subject-aware diffusion was presented as generally superior for cross-subject EEG denoising.\n\n"
        "## New evidence\n\nA clean-room query-disjoint support protocol establishes transferable support information and some paired utility, but correct-donor specificity is mixed. Natural attenuation exists only at operating points with measurable retention/PSD costs. Support states are strongly linkable.\n\n"
        "## What survives\n\nTechnical viability; query-disjoint support information; bounded paired utility; competitive diffusion positioning; transparent attenuation–retention and cost curves.\n\n"
        "## What is withdrawn\n\nUniversal personalization, diffusion superiority, established donor specificity, physiological/task preservation, privacy safety, cross-session permanence, cross-montage generalization, and deployment claims.\n\n"
        "## Reviewer requirements completed\n\nStrong deterministic and diffusion controls; RAW/STANDARD; subject ablations and falsification; participant-first statistics; exact support amount; sampler steps; latency/memory; and linkage-risk audit.\n\n"
        "## Remaining missing evidence\n\nTask-valid physiological endpoints, independent montage/generalization evidence, and sealed confirmation (intentionally unopened).\n\n"
        "## Proposed scope\n\nScope A: an audit-centric revision focused on utility, mixed specificity, attenuation–retention trade-offs, computational burden, privacy risk, and explicit claim boundaries.\n"
    )

    reviewer_columns = ["reviewer", "comment", "original_weakness", "new_evidence", "proposed_manuscript_change", "response_wording", "status"]
    (report_root / "v31_reviewer_response_map.md").write_text(
        "# V31 reviewer-response architecture\n\nThe repository does not contain the verbatim decision letter. Reviewer attribution below is a transparent reconstruction from the project ledger’s summarized requirements, not a quotation.\n\n"
        + markdown_table(reviewers, reviewer_columns) + "\n"
    )

    (report_root / "v31_scope_A_manuscript_blueprint.md").write_text(
        "# Scope A manuscript blueprint\n\n"
        "1. Introduction — narrowed question and withdrawn original claims.\n"
        "2. Related Work — EEG diffusion, calibration, falsification, privacy.\n"
        "3. Query-Disjoint Support-Conditioned Diffusion — frozen clean-room candidates and inference boundary.\n"
        "4. Evaluation Protocol and Falsification Controls — common panel, all donors, lag/shuffle/null, participant-first statistics.\n"
        "5. Paired Utility and Correct-Context Specificity — absolute metrics before contrasts.\n"
        "6. Natural Attenuation–Retention Trade-off — no physiological-preservation alias.\n"
        "7. Computational Cost and Support Burden — V31 exact durations and V30 step/latency data.\n"
        "8. Privacy-Linkage Audit — linkage risk without anonymity claims.\n"
        "9. Limitations and Claim Boundary — within-session, fixed montage, no task-valid physiology or confirmation.\n"
        "10. Conclusion — viable but mixed evidence; no deployment claim.\n"
    )
    (report_root / "v31_scope_B_manuscript_blueprint.md").write_text(
        "# Scope B manuscript blueprint\n\n"
        "1. Introduction — narrow ocular-removal operating-point claim.\n"
        "2. Related Work — conditional diffusion and EEG artifact removal.\n"
        "3. Energy-Controlled Support-Conditioned Diffusion — V27-L0.5 and fixed controls.\n"
        "4. Experiments — common panel, population/DET/EEGDfus, participant-first statistics.\n"
        "5. Attenuation–Retention Pareto — full lambda curve and absolute metrics.\n"
        "6. Specificity and Privacy Limitations — mixed donor evidence and linkage risk.\n"
        "7. Conclusion — one transparent development operating point; no confirmation/generalization claim.\n"
    )


__all__ = [
    "claim_rows",
    "markdown_table",
    "reviewer_rows",
    "scope_comparison",
    "write_scope_and_consultation_reports",
]
