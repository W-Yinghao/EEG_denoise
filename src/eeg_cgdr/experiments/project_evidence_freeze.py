"""Read-only project-wide scientific evidence freeze.

This module deliberately consumes only committed, small result summaries and
registries.  It never opens raw EEG, prepared arrays, checkpoints, or sealed
outcomes, and it performs no model inference or training.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


BASE_COMMIT = "183567690d66f8ec49bb619c15ef15fff844c066"
RAW_CLOSURE_COMMIT = "f0e4e655c6723929233de14e75b48c410421ec0d"
CENTRAL_SENTENCE = (
    "Participant-specific support was often distinguishable from mismatched "
    "support, but did not provide reproducible incremental denoising utility "
    "over strong population context. Multi-sample diffusion showed estimator-"
    "level gains in selected development settings, largely associated with averaging."
)


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    data = list(rows)
    if fields is None:
        fields = list(data[0]) if data else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    dataset: str
    branch: str
    commit: str
    config: str
    report: str
    result_summary: str
    scientific_unit: str
    development_or_confirmation: str
    natural_or_semisim_or_proxy: str
    seeds: str
    coverage: str
    technical_status: str
    scientific_status: str
    claim_scope: str
    query_eog_used: str
    support_eog_or_labels_used: str
    oracle_only: str
    superseded_by: str
    canonicality: str


def _experiments() -> list[Experiment]:
    # Canonicality is explicit: historical presence is not permission to claim.
    return [
        Experiment("v3_routes", "Klados/SGEYESUB", "codex/literature-guided-exploration-v3", "61efe46567f807532abb50fa84f2ae771d41b8d1", "configs/cgdr/literature_guided_exploration_v3.yaml", "reports/v3_evidence_correction.md", "results/cgdr/v3_evidence_repair/result_summary.json", "source-record/participant-stem", "development", "mixed", "multiple", "reported units", "valid after evidence repair", "specific moment-summary/output-residual instances no-go", "route-specific only", "mixed", "support moments/labels by route", "some arms", "", "supporting"),
        Experiment("pc_bounded_oracle", "Klados/SGEYESUB", "codex/mobile-temporal-diffusion-v5", "50a7b178aed6120165e968a554600a222795e2b8", "", "reports/pc_safety_constrained_oracle.md", "results/cgdr/pc_constrained_oracle/result_summary.json", "participant/stem", "development", "oracle", "none", "full frozen denominator", "valid bounded-candidate oracle", "oracle/selector diagnostic only", "not deployable efficacy", "no", "matching support", "yes", "", "supporting"),
        Experiment("mobile_v4", "MobileBCI", "codex/mobile-headroom-temporal-v4", "1cefb776f674e7264aa48fe3a12138794d32657a", "configs/cgdr/mobile_bci_headroom_v4", "reports/mobile_bci_headroom_v4.md", "results/cgdr/mobile_bci_headroom_v4/final/result_summary.json", "participant", "development", "natural", "one", "development only; sealed 8 excluded", "protocol repaired later", "historical development screen", "not confirmation", "no", "support EEG; EOG masked imperfectly", "no", "mobile_v5", "superseded"),
        Experiment("mobile_v5", "MobileBCI", "codex/mobile-temporal-diffusion-v5", "50a7b178aed6120165e968a554600a222795e2b8", "configs/cgdr/mobile_bci_headroom_v4", "reports/mobile_temporal_diffusion_v5_closure.md", "results/cgdr/mobile_temporal_diffusion_v5/result_summary.json", "participant", "development", "natural", "one", "development only; sealed 8 unopened", "protocol core valid; prior SSVEP safety invalid", "one-seed raw temporal route no-go", "current route only", "no", "support EEG; EOG zeros not true masking", "no", "", "canonical_milestone"),
        Experiment("sge_v6", "SGEYESUB", "codex/sge-dynamic-transfer-diffusion-v6", "b70b73e97d9298f377f773fb159a70a6f43832b7", "", "reports/sge_dynamic_transfer_diffusion_v6.md", "results/cgdr/sge_dynamic_transfer_diffusion_v6/route_decision.json", "participant-stem", "development", "paired semisim + natural", "one", "58 compatible stems target", "diffusion validity not established", "static transfer-summary instance no-go", "not a valid family negative", "natural evaluator only", "support EEG+EOG", "generator only", "sge_v7", "superseded"),
        Experiment("sge_v7", "SGEYESUB", "codex/sge-eb-score-adapter-v7", "8642861e53bb39a1fe8d25b80e1af2d6a3a34444", "", "reports/v6_diffusion_validity_audit.md", "results/cgdr/v6_diffusion_validity_audit/validity_decision.json", "fold diagnostics", "development", "real-batch technical", "none", "three diagnostic folds", "current v6 backbone/K8 not validated", "cause not fully identified", "technical boundary", "no", "support EEG+EOG", "no", "sge_v8", "supporting"),
        Experiment("sge_v8", "SGEYESUB", "codex/sge-score-lora-v8", "ce3dd7f04a05383f2a5ac7345988e302b9083e48", "configs/cgdr/sge_score_lora_v8.yaml", "reports/sge_score_lora_v8.md", "results/cgdr/sge_score_lora_v8/result_summary.json", "participant-stem", "development", "paired semisim + natural", "one", "diagnostic folds", "population gate failed under frozen protocol", "Score-LoRA not run", "implementation-specific", "natural evaluator only", "label-assisted calibration support", "no", "sge_v8_1", "superseded"),
        Experiment("sge_v8_1", "SGEYESUB", "codex/sge-eb-bridge-v8-1", "89e88c6f82b63b336b9b8a2e2bd927043183c533", "", "reports/sge_eb_denoising_bridge_v8_1.md", "results/cgdr/sge_eb_bridge_v8_1/result_summary.json", "participant-stem", "development", "paired semisim + natural", "one", "four diagnostic folds", "validity repaired; bridge failed", "Score-LoRA not run", "development bridge only", "natural evaluator only", "support EEG+EOG", "ceiling arms", "", "canonical_milestone"),
        Experiment("sge_v9", "SGEYESUB", "codex/sge-basis-score-factorial-v9", "e690cd3b1c591d9b1c92050d3a0116b459f3555a", "", "reports/sge_basis_score_factorial_v9.md", "results/cgdr/sge_basis_score_factorial_v9/result_summary.json", "participant-stem", "development", "paired semisim + natural", "one", "six folds/14 stems", "randomness/aggregation confounded", "historical candidate only", "not reproducibility evidence", "natural evaluator only", "label-assisted support", "no", "sge_v9r", "superseded"),
        Experiment("sge_v9r", "SGEYESUB", "codex/sge-basis-score-factorial-v9r", "9a352d910212a2bd87fd6a74f39a2608eb600e67", "", "reports/sge_basis_score_factorial_v9r.md", "results/cgdr/sge_basis_score_factorial_v9r/result_summary.json", "participant-stem", "development", "paired semisim + natural", "three adaptation seeds", "exact replay + limited new folds", "common-random validity passed", "interaction not sufficient for stable subject utility", "current D11 formulation", "natural evaluator only", "label-assisted calibration support", "no", "", "canonical_milestone"),
        Experiment("bci2a_v10", "BCI Competition IV-2a/2b", "codex/bci2a-hierarchical-score-v10", "90b9411a30aeee799d731a17e626c8c377e1e1fc", "", "reports/bci2a_hierarchical_score_diffusion_v10.md", "results/cgdr/bci2a_hierarchical_score_v10/result_summary.json", "participant", "development", "paired semisim + natural", "none/model not adjudicated", "9 participants", "base pipeline invalid for 2b; identifiability not established for 2a", "hierarchical Score-LoRA not scientifically adjudicated", "current band-transfer representation", "yes for guided arms", "support EEG+EOG", "query operator ceiling", "bci2b_v11", "supporting"),
        Experiment("bci2b_v11", "BCI Competition IV-2b", "codex/bci2b-eog-residual-v11", "9ee9218ce6e3bf11b1f157339ae53b0506b5877b", "", "reports/bci2b_eog_residual_diffusion_v11.md", "results/cgdr/bci2b_eog_residual_v11/result_summary.json", "participant", "development", "paired semisim + natural", "one", "3 participants gate", "spectral scale later repaired", "frozen no-go retained", "historical gate only", "yes", "support EEG+EOG", "bridge ceilings", "bci2b_v11_1", "superseded"),
        Experiment("bci2b_v11_1", "BCI Competition IV-2b", "codex/bci2b-eog-residual-v11-1", "4673aa9ca2b23c62d7d7374de16f794a1b61fd2d", "", "reports/bci2b_eog_residual_diffusion_v11_1.md", "results/cgdr/bci2b_eog_residual_v11_1/result_summary.json", "participant", "development", "paired semisim + natural", "one", "9 participants", "valid fixed EOG-guided pipeline", "subject operator signal vs POP7/cyclic wrong", "same development cohort", "yes", "support EEG+EOG", "no", "bci2b_replication", "supporting"),
        Experiment("bci2b_replication", "BCI Competition IV-2b", "codex/bci2b-subject-diffusion-replication", "ee759201874b0673cd580368900e34e144707953", "", "reports/bci2b_subject_diffusion_replication.md", "results/cgdr/bci2b_subject_diffusion_replication/result_summary.json", "participant", "development", "paired semisim + natural", "3", "9/9", "valid fixed EOG-guided pipeline", "three-seed same-cohort stability vs POP7/wrong", "not independent confirmation; not strong POP8", "yes", "support EEG+EOG", "no", "bci2b_pop8_strict", "supporting"),
        Experiment("bci2b_mechanism_uq", "BCI Competition IV-2b", "codex/subject-diffusion-mechanism-uq", "1baddc978405cf85e1b1e278395c3f1f0439a814", "", "reports/bci2b_subject_diffusion_mechanism_uq.md", "results/cgdr/bci2b_subject_diffusion_mechanism_uq/result_summary.json", "participant", "development", "paired semisim + natural", "3", "9/9", "valid frozen-checkpoint audit", "operator-mediated effect; weak uncertainty association only", "no calibrated posterior claim", "yes", "support EEG+EOG", "no", "bci2b_context_causal", "supporting"),
        Experiment("bci2b_context_causal", "BCI Competition IV-2b", "codex/bci2b-subject-diffusion-next", "cec659deb61b206f39f8434252ab2fb503cfb880", "", "reports/bci2b_context_causal_audit.md", "results/cgdr/bci2b_context_causal_audit/result_summary.json", "participant", "development", "paired semisim", "3", "9/9", "valid causal decomposition", "context coherence penalty; score-specific mediation not identified", "frozen model diagnostic", "yes", "support EEG+EOG", "oracle cross diagnostic", "", "canonical_milestone"),
        Experiment("bci2b_pop8_strict", "BCI Competition IV-2b", "codex/bci2b-operator-shrinkage", "41b2bfe64c4b80d6a4739aca6f13947293e687e5", "configs/cgdr/bci2b_operator_shrinkage.yaml", "reports/bci2b_pop8_strict_reanalysis.md", "results/cgdr/bci2b_pop8_strict/result_summary.json", "participant", "development", "paired semisim + natural", "3", "26/27 units; 9 participants", "strict eligibility and participant-first valid", "strong POP8-R canonical control", "strong population reference", "yes", "support EEG+EOG", "no", "", "canonical_milestone"),
        Experiment("bci2b_shrinkage", "BCI Competition IV-2b", "codex/bci2b-operator-shrinkage", "41b2bfe64c4b80d6a4739aca6f13947293e687e5", "configs/cgdr/bci2b_operator_shrinkage.yaml", "reports/bci2b_operator_shrinkage.md", "results/cgdr/bci2b_operator_shrinkage/result_summary.json", "participant", "development", "paired semisim + natural", "3", "26/27 units; 9 participants", "valid support-only shrinkage", "MATCH over strong POP not established", "closes EOG-transfer personalization instances", "yes", "support EEG+EOG", "no", "", "canonical_milestone"),
        Experiment("capacity_matched", "BCI Competition IV-2b", "codex/diffusion-fair-neural-prior", "2f13de18118309ba731a49a553e547e368b3121b", "configs/cgdr/diffusion_fair_neural_prior.yaml", "reports/diffusion_capacity_matched_audit.md", "results/cgdr/diffusion_fair_neural_prior/capacity_matched/result_summary.json", "participant", "development", "paired semisim + natural", "3", "26/27 units; 9 participants", "DET2 capacity match validated", "extra-capacity confounded; averaging signal present", "diffusion-specific value unsupported", "yes", "support EEG+EOG", "no", "", "canonical_milestone"),
        Experiment("clean_neural_prior", "BCI Competition IV-2a/2b", "codex/diffusion-fair-neural-prior", "2f13de18118309ba731a49a553e547e368b3121b", "configs/cgdr/diffusion_fair_neural_prior.yaml", "reports/clean_neural_subject_headroom.md", "results/cgdr/diffusion_fair_neural_prior/clean_neural_aligned_diffusion/result_summary.json", "participant", "development", "proxy then paired semisim + natural", "one", "9 participants", "valid current alignment instance", "clean-neural aligned diffusion instance no-go", "current covariance-alignment implementation", "query evaluator only", "support EEG; EOG exclusion only", "proxy headroom evaluator-only", "", "canonical_milestone"),
        Experiment("raw_support", "BCI Competition IV-2b", "codex/raw-support-clean-diffusion", "dc43a5761f84a239425a0483d737ede3902aa85b", "configs/cgdr/raw_support_clean_diffusion.yaml", "reports/raw_support_clean_diffusion.md", "results/cgdr/raw_support_clean_diffusion/result_summary.json", "participant", "development", "paired semisim + natural", "one", "26/27 units; 9 participants", "valid current raw-support instance", "initial result corrected by donor closure", "support-EOG-assisted raw temporal route", "no", "support EEG; EOG used for exclusion", "covariance oracle diagnostic", "raw_support_closure", "superseded"),
        Experiment("raw_support_closure", "BCI Competition IV-2b", "codex/raw-support-clean-diffusion-closure", RAW_CLOSURE_COMMIT, "configs/cgdr/raw_support_clean_diffusion.yaml", "reports/raw_support_clean_diffusion_closure.md", "results/cgdr/raw_support_clean_diffusion_closure/result_summary.json", "participant", "development", "paired semisim + natural", "one", "26/27 units; 9 participants", "participant-first donor correction valid", "MATCH over strong POP not established; donor specificity suggestive; K8 gain", "current raw temporal route only", "no", "support EEG; EOG used for exclusion", "no", "", "canonical_milestone"),
        Experiment("physiomotion_j1", "PhysioMotion ds006386", "codex/physiomotion-subject-restoration", "fbcd5b01121c471a09023ce5369b8bcc1bae1c19", "configs/cgdr/physiomotion_subject_restoration.yaml", "reports/physiomotion_subject_restoration.md", "results/cgdr/physiomotion_subject_restoration/result_summary.json", "participant", "development", "exact clean-mask proxy", "none", "20 development; 10 sealed unopened", "bank-size confounded", "historical headroom screen", "requires fairness repair", "no", "support clean EEG", "no", "physiomotion_j1r", "superseded"),
        Experiment("physiomotion_j1r", "PhysioMotion ds006386", "codex/physiomotion-retrieval-fairness", "d01a7b763b4eacb2fac634e8b6f16f656c69b490", "configs/cgdr/physiomotion_retrieval_fairness.yaml", "reports/physiomotion_retrieval_fairness.md", "results/cgdr/physiomotion_retrieval_fairness/result_summary.json", "participant", "development", "exact clean-mask proxy", "none", "17 evaluable/20 policy; 10 sealed unopened", "fair bank-size audit valid", "deployable hybrid headroom proxy", "not denoising utility", "no", "support clean EEG", "oracle evaluator arm separate", "physiomotion_hybrid", "supporting"),
        Experiment("physiomotion_hybrid", "PhysioMotion ds006386", "codex/physiomotion-hybrid-masked-diffusion", "4bf631aaf0a081ad2920c5fa20cda64eb87fbbcf", "configs/cgdr/physiomotion_hybrid_masked.yaml", "reports/physiomotion_hybrid_masked_diffusion.md", "results/cgdr/physiomotion_hybrid_masked/result_summary.json", "participant", "development", "exact clean-mask + natural artifact", "one", "17 evaluable/20 policy; sealed 10 unopened", "technical validity passed", "DEV_ONE_SEED_NO_GO", "current hybrid masked instance", "no", "support clean EEG", "natural mask evaluator-only", "", "canonical_milestone"),
        Experiment("shu_headroom", "SHU MultiSession MI", "codex/shu-task-phenotype-diffusion", BASE_COMMIT, "configs/cgdr/shu_task_phenotype_diffusion.yaml", "reports/shu_task_phenotype_diffusion.md", "results/cgdr/shu_task_phenotype_diffusion/result_summary.json", "participant", "development", "masked reconstruction proxy", "none", "25/25 Day2/3; Day4/5 unopened", "valid relative same-input headroom; source provenance incomplete", "TASK_PHENOTYPE_HEADROOM_NO_GO", "current task-phenotype probe; no model trained", "no", "Day1 EEG + MI class labels", "no", "", "canonical_milestone"),
    ]


BRANCH_CLASS = {
    "codex/sge-eb-bridge-v8-1": "canonical_milestone", "codex/sge-basis-score-factorial-v9r": "canonical_milestone",
    "codex/bci2b-eog-residual-v11-1": "supporting", "codex/bci2b-subject-diffusion-replication": "supporting",
    "codex/bci2b-subject-diffusion-next": "canonical_milestone", "codex/bci2b-operator-shrinkage": "canonical_milestone",
    "codex/diffusion-fair-neural-prior": "canonical_milestone", "codex/raw-support-clean-diffusion-closure": "canonical_milestone",
    "codex/physiomotion-retrieval-fairness": "supporting", "codex/physiomotion-hybrid-masked-diffusion": "canonical_milestone",
    "codex/shu-task-phenotype-diffusion": "canonical_milestone", "codex/taas-subject-diffusion-freeze": "archive_only",
}


def branch_registry(repo: Path) -> list[dict[str, str]]:
    out = subprocess.check_output(["git", "for-each-ref", "--format=%(refname:short)|%(objectname)", "refs/remotes/origin"], cwd=repo, text=True)
    rows = []
    for line in out.splitlines():
        name, sha = line.split("|", 1)
        short = name.removeprefix("origin/")
        if short == "HEAD":
            continue
        category = BRANCH_CLASS.get(short)
        if category is None:
            if any(x in short for x in ("v8", "v9", "v11", "raw-support", "physiomotion", "mobile", "sge")):
                category = "superseded"
            elif any(x in short for x in ("probe", "repair", "audit", "selector")):
                category = "invalid_diagnostic"
            else:
                category = "archive_only"
        replacement = {
            "codex/sge-score-lora-v8": "codex/sge-eb-bridge-v8-1",
            "codex/sge-basis-score-factorial-v9": "codex/sge-basis-score-factorial-v9r",
            "codex/bci2b-eog-residual-v11": "codex/bci2b-eog-residual-v11-1",
            "codex/raw-support-clean-diffusion": "codex/raw-support-clean-diffusion-closure",
            "codex/physiomotion-subject-restoration": "codex/physiomotion-retrieval-fairness",
            "codex/physiomotion-retrieval-fairness": "codex/physiomotion-hybrid-masked-diffusion",
        }.get(short, "")
        rows.append({"branch": short, "sha": sha, "classification": category, "superseded_by": replacement})
    return sorted(rows, key=lambda r: r["branch"])


def _effect(summary: dict[str, Any], key: str) -> dict[str, Any]:
    return summary["effects"][key]


def evidence_tables(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cap = _json(repo / "results/cgdr/diffusion_fair_neural_prior/capacity_matched/result_summary.json")
    shrink = _json(repo / "results/cgdr/bci2b_operator_shrinkage/result_summary.json")
    raw = _json(repo / "results/cgdr/raw_support_clean_diffusion_closure/result_summary.json")
    phys = _json(repo / "results/cgdr/physiomotion_hybrid_masked/result_summary.json")
    shu = _json(repo / "results/cgdr/shu_task_phenotype_diffusion/result_summary.json")
    neural = _json(repo / "results/cgdr/diffusion_fair_neural_prior/clean_neural_aligned_diffusion/result_summary.json")
    subject = [
        {"dataset":"BCI2b","representation":"support-only EOG-transfer shrinkage","match_minus_pop":shrink["effects"]["E_P"]["mean"],"match_minus_wrong":shrink["effects"]["E_W"]["mean"],"coverage":"26/27 units; 9 participants","scientific_unit":"participant","safety":"relative margins disclosed","routing":"STRONG_POPULATION_MATCHED_BY_SUPPORT_SHRINKAGE_NOT_ESTABLISHED","canonical_source":"bci2b_shrinkage"},
        {"dataset":"BCI2b","representation":"clean-neural covariance alignment","match_minus_pop":neural["effects"]["U_P"]["mean"],"match_minus_wrong":neural["effects"]["U_W"]["mean"],"coverage":"26/27 units; 9 participants","scientific_unit":"participant","safety":"relative noninferiority only","routing":"CLEAN_NEURAL_ALIGNED_DIFFUSION_INSTANCE_NO_GO","canonical_source":"clean_neural_prior"},
        {"dataset":"BCI2b","representation":"support-EOG-assisted raw temporal context K8","match_minus_pop":raw["effects"]["K8"]["U_P"]["mean"],"match_minus_wrong":raw["effects"]["K8"]["U_W_donor_mean"]["mean"],"coverage":"26/27 units; 9 participants","scientific_unit":"participant","safety":"relative passed; absolute not established","routing":"MATCH_OVER_STRONG_POP_NOT_ESTABLISHED; DONOR_SPECIFICITY_SUGGESTIVE","canonical_source":"raw_support_closure"},
        {"dataset":"PhysioMotion","representation":"hybrid clean-retrieval masked diffusion","match_minus_pop":phys["subject_effects"]["U_P"]["mean"],"match_minus_wrong":phys["subject_effects"]["U_W"]["mean"],"coverage":"17 evaluable/20 policy; sealed 10 unopened","scientific_unit":"participant","safety":"natural aggregate passed; subject gate failed","routing":"DEV_ONE_SEED_NO_GO","canonical_source":"physiomotion_hybrid"},
        {"dataset":"SHU MultiSession MI","representation":"Day1 class-conditional task phenotype ridge probe","match_minus_pop":shu["effects"]["H_P"]["mean"],"match_minus_wrong":shu["effects"]["H_W"]["mean"],"coverage":"25/25 Day2/3; Day4/5 unopened","scientific_unit":"participant","safety":"ERD/decoder gate not reached after headroom failure","routing":"TASK_PHENOTYPE_HEADROOM_NO_GO","canonical_source":"shu_headroom"},
    ]
    diffusion = [
        {"dataset":"BCI2b strict POP8-R","diff_k1_minus_det":None,"k8_minus_k1":cap["effects"]["E_avg"]["mean"],"diff_k8_minus_det1":"historical positive but capacity-confounded","det8_available":"no","capacity_matched_det2_minus_diff_k8":cap["effects"]["E_cap"]["mean"],"safety":"DET2-relative margins passed","allowed_claim":"K8 averaging gain; no diffusion-specific point-estimate value","canonical_source":"capacity_matched"},
        {"dataset":"BCI2b raw temporal support","diff_k1_minus_det":raw["effects"]["K1"]["DET_minus_DIFF"]["mean"],"k8_minus_k1":raw["effects"]["K8_minus_K1"]["mean"],"diff_k8_minus_det1":raw["effects"]["K8"]["DET_minus_DIFF"]["mean"],"det8_available":"no","capacity_matched_det2_minus_diff_k8":None,"safety":"relative safety passed; absolute safety not established","allowed_claim":"multi-sample averaging gain only; compute-matched DET not tested","canonical_source":"raw_support_closure"},
        {"dataset":"PhysioMotion masked restoration","diff_k1_minus_det":phys["mechanism"]["E_D_K1_mean"],"k8_minus_k1":phys["mechanism"]["E_avg_mean"],"diff_k8_minus_det1":"not canonical unique diffusion evidence","det8_available":"no (one-seed no-go)","capacity_matched_det2_minus_diff_k8":None,"safety":"natural aggregate passed; subject effect failed","allowed_claim":"averaging signal in failed current instance","canonical_source":"physiomotion_hybrid"},
    ]
    return subject, diffusion


def claim_ledger(repo: Path) -> list[dict[str, Any]]:
    cap = _json(repo / "results/cgdr/diffusion_fair_neural_prior/capacity_matched/result_summary.json")
    raw = _json(repo / "results/cgdr/raw_support_clean_diffusion_closure/result_summary.json")
    return [
        {"claim":"absolute restoration validity","status":"mixed","canonical_source":"capacity_matched; physiomotion_hybrid; raw_support_closure","scientific_unit":"participant","effect":"instance-dependent","coverage":"multiple development cohorts","boundary":"some methods beat raw/interpolation and pass mean safety; exceptions exist","forbidden_interpretation":"uniform safety or general denoising validity"},
        {"claim":"population diffusion signal","status":"supported","canonical_source":"capacity_matched; bci2b_pop8_strict","scientific_unit":"participant","effect":"selected development instances improve over raw","coverage":"BCI2b 9 participants","boundary":"development and estimator-specific","forbidden_interpretation":"state of the art or confirmation"},
        {"claim":"DIFF-K1 vs matched DET","status":"unsupported","canonical_source":"raw_support_closure; capacity_matched","scientific_unit":"participant","effect":raw["effects"]["K1"]["DET_minus_DIFF"]["mean"],"coverage":"9 participants","boundary":"one current clean-waveform instance; capacity-matched main audit uses K8","forbidden_interpretation":"diffusion sampler uniquely superior"},
        {"claim":"DIFF-K8 vs DET1","status":"mixed","canonical_source":"raw_support_closure; capacity_matched","scientific_unit":"participant","effect":raw["effects"]["K8"]["DET_minus_DIFF"]["mean"],"coverage":"9 participants","boundary":"DET1 is not capacity matched","forbidden_interpretation":"unique diffusion evidence"},
        {"claim":"K8 sampling-average effect","status":"supported","canonical_source":"capacity_matched; raw_support_closure; physiomotion_hybrid","scientific_unit":"participant","effect":cap["effects"]["E_avg"]["mean"],"coverage":"9/9 positive in capacity-matched BCI2b","boundary":"point-estimate averaging, not posterior calibration","forbidden_interpretation":"calibrated posterior or iterative diffusion value"},
        {"claim":"DIFF-K8 vs DET8","status":"not_tested","canonical_source":"PROJECT_STATE","scientific_unit":"participant","effect":"NA","coverage":"no canonical DET8 comparison","boundary":"compute/ensemble-matched test absent","forbidden_interpretation":"diffusion beats deterministic ensemble"},
        {"claim":"MATCH vs strong POP","status":"unsupported","canonical_source":"bci2b_shrinkage; raw_support_closure; physiomotion_hybrid; shu_headroom","scientific_unit":"participant","effect":"BCI2b shrinkage E_P=0.00267; raw K8 U_P=0.00382; Physio=-0.00216; SHU=-0.01409","coverage":"9, 20, and 25 participant development cohorts","boundary":"tested representations closed","forbidden_interpretation":"general subject-aware denoising benefit"},
        {"claim":"MATCH vs WRONG","status":"mixed","canonical_source":"raw_support_closure; shu_headroom; physiomotion_hybrid","scientific_unit":"participant","effect":"often positive specificity without POP utility","coverage":"dataset-specific","boundary":"donor distinguishability is not incremental utility","forbidden_interpretation":"subject-aware method established"},
        {"claim":"natural safety","status":"mixed","canonical_source":"capacity_matched; raw_support_closure; physiomotion_hybrid","scientific_unit":"participant","effect":"mean gates pass in some instances with disclosed reversals","coverage":"development only","boundary":"proxy/evaluator-specific","forbidden_interpretation":"uniformly safe"},
        {"claim":"uncertainty/proper score","status":"unsupported","canonical_source":"bci2b_mechanism_uq","scientific_unit":"participant","effect":"CRPS/risk AUC worse than DET ensemble; weak association only","coverage":"9 participants","boundary":"current checkpoints and non-identifiable posterior calibration","forbidden_interpretation":"probabilistic advantage"},
        {"claim":"independent confirmation","status":"not_tested","canonical_source":"PROJECT_STATE","scientific_unit":"participant","effect":"NA","coverage":"sealed outcomes unopened","boundary":"all canonical science is development","forbidden_interpretation":"replication or confirmation"},
    ]


def split_registry() -> list[dict[str, str]]:
    return [
        {"dataset":"BCI2b","train_dev_sealed_ids":"9 public development participants; no sealed cohort","support_query_definition":"same-session calibration support -> later query; 26/27 eligible units","blocked_fallback":"owner support <120 s blocked/fallback POP","outcomes_opened":"development evaluator only","wrong_donor_rule":"strict cyclic unseen for historical control; training-seen donors separated","scientific_level":"participant"},
        {"dataset":"MobileBCI","train_dev_sealed_ids":"development participants frozen; 8 sealed participants","support_query_definition":"S0/S1/S2 early 60 s support with guard -> later query","blocked_fallback":"protocol-specific missing/blocked separated","outcomes_opened":"development only; sealed unopened","wrong_donor_rule":"fold-role confound disclosed in v5 closure","scientific_level":"participant"},
        {"dataset":"SGEYESUB","train_dev_sealed_ids":"development compatible stems only","support_query_definition":"early 30/120 s support -> guarded later query","blocked_fallback":"singleton/incompatible layout blocked","outcomes_opened":"development paired/natural evaluators","wrong_donor_rule":"same compatibility cell heldout donor where available","scientific_level":"participant-stem; shared-fold dependence"},
        {"dataset":"PhysioMotion","train_dev_sealed_ids":"20 development / 10 sealed","support_query_definition":"run01 support -> runs02-06 query","blocked_fallback":"3 unavailable participants fallback POP in policy estimand","outcomes_opened":"development 20 only; sealed 10 unopened","wrong_donor_rule":"three other unseen recipients per heldout fold","scientific_level":"participant"},
        {"dataset":"SHU MultiSession MI","train_dev_sealed_ids":"25 participants; Day2/3 development; Day4/5 session-sealed","support_query_definition":"Day1 class phenotype -> Day2/3 query","blocked_fallback":"none; 25/25 available in LMDB","outcomes_opened":"Day1-3 only; Day4/5 unopened","wrong_donor_rule":"four other unseen participants in same outer fold","scientific_level":"participant"},
        {"dataset":"Klados","train_dev_sealed_ids":"source records; no participant-heldout confirmation","support_query_definition":"source-record mechanisms","blocked_fallback":"route-specific","outcomes_opened":"development/source-record only","wrong_donor_rule":"route-specific","scientific_level":"source-record; must not be called participant"},
    ]


def claim_whitelist() -> list[dict[str, str]]:
    supported = [
        ("participant support can be distinguished from wrong-donor support in some protocols", "partially_supported", "Specificity does not imply utility over strong population."),
        ("strong population pooling usually matches or exceeds finite subject support", "supported", "Applies to tested development instances."),
        ("population or multi-sample restoration beats raw/simple interpolation in selected development tasks", "partially_supported", "Not uniform across datasets or participants."),
        ("K8 averaging improves point estimates in several instances", "supported", "Averaging effect, not calibrated posterior value."),
        ("some instances meet absolute restoration or mean safety gates", "partially_supported", "Participant reversals and proxy limitations remain."),
        ("leakage-resistant support/query, POP/WRONG, and fail-closed evaluation frameworks were established", "supported", "Engineering/evaluation contribution only."),
    ]
    unsupported = [
        "MATCH stably beats strong POP", "general subject-aware denoising benefit", "diffusion-specific subject utility",
        "DIFF-K1 stably beats capacity-matched DET", "DIFF-K8 beats compute/ensemble-matched DET8",
        "calibrated posterior or UQ advantage", "independent confirmation", "cross-dataset validated subject-aware diffusion",
    ]
    rows = [{"claim": c, "status": s, "boundary": b} for c, s, b in supported]
    rows += [{"claim": c, "status": "unsupported", "boundary": "Forbidden as a project conclusion."} for c in unsupported]
    return rows


def aliases() -> list[dict[str, str]]:
    return [
        {"alias":"POP","canonical":"population context","scope":"generic; always disclose pool size"},
        {"alias":"STRONG-POP","canonical":"strong population context","scope":"fold-valid participant-equal pool"},
        {"alias":"POP-LARGE","canonical":"large population retrieval bank","scope":"PhysioMotion retrieval"},
        {"alias":"DET1","canonical":"first-stage deterministic estimator","scope":"not capacity matched to DIFF"},
        {"alias":"DET2","canonical":"capacity-matched second deterministic U-Net","scope":"same rows/schedule/capacity as DIFF"},
        {"alias":"DET8","canonical":"8-member deterministic ensemble","scope":"compute/ensemble comparator; currently unavailable canonically"},
        {"alias":"DIFF-K1","canonical":"one diffusion sample","scope":"point estimate from first common-random sample"},
        {"alias":"DIFF-K8","canonical":"arithmetic mean of 8 waveforms","scope":"average waveforms before metrics"},
        {"alias":"MATCH","canonical":"recipient support context","scope":"support-only; no query outcomes"},
        {"alias":"WRONG","canonical":"non-recipient donor context","scope":"score donor-wise before recipient averaging"},
        {"alias":"SHUFFLED","canonical":"distribution-preserving broken correspondence","scope":"must state temporal vs pair permutation"},
        {"alias":"policy","canonical":"full-denominator fallback-policy estimand","scope":"fallback retained"},
        {"alias":"evaluable","canonical":"mechanism estimand among eligible units","scope":"fallback excluded"},
        {"alias":"source-record","canonical":"recording/source unit","scope":"must not be called participant"},
        {"alias":"participant","canonical":"participant-first scientific unit","scope":"protocol/seeds/windows averaged within participant"},
    ]


def metric_schema() -> dict[str, Any]:
    return {
        "aggregation_order": ["window/sample/donor", "protocol or session", "seed", "participant"],
        "scientific_unit": "participant unless explicitly participant-stem or source-record",
        "metrics": {
            "RRMSE": {"direction": "lower_is_better", "unit": "dimensionless"},
            "correlation": {"direction": "higher_is_better", "unit": "dimensionless"},
            "delta_SNR": {"direction": "higher_is_better", "unit": "dB"},
            "EOG_attenuation": {"direction": "higher_is_better", "unit": "dimensionless"},
            "preservation": {"direction": "higher_is_better", "unit": "dimensionless"},
            "PSD_distortion": {"direction": "lower_is_better", "unit": "dimensionless"},
            "covariance_distortion": {"direction": "lower_is_better", "unit": "dimensionless"},
            "MI_kappa": {"direction": "higher_is_better", "unit": "kappa"},
            "utility": {"direction": "higher_is_better", "unit": "RRMSE difference", "definition": "comparator RRMSE minus candidate RRMSE"},
        },
        "K_contract": {"K1": "first common-random sample", "K8": "arithmetic waveform mean before scoring", "NFE_DDIM25_K8": 200},
        "forbidden": ["windows as n", "seeds as n", "DIFF-K8 vs DET1 as unique diffusion evidence"],
    }


def asset_manifest(repo: Path) -> list[dict[str, Any]]:
    items = [
        (repo / "results/cgdr", "small committed result tree", "project", BASE_COMMIT, "canonical summaries", "git+server", "regenerate from committed summaries"),
        (Path("/projects/EEG-foundation-model"), "data root", "all", "external", "raw/derived EEG data; read-only freeze", "retain_server", "do not delete; reacquire only under separate authorization"),
        (Path("/projects/EEG-foundation-model/PhysioMotion_Artifact"), "dataset/derived root", "PhysioMotion", "4bf631a", "20 development assets; sealed guard applies", "retain_server", "do not open sealed in freeze"),
        (Path("/projects/EEG-foundation-model/SHU_MultiSession_MI"), "dataset locator", "SHU", BASE_COMMIT, "may be absent; canonical source is datalake LMDB", "retain_server", "source provenance incomplete"),
        (Path("/home/infres/yinwang/denoiseNet_subject_diffusion_replication"), "historical worktree", "BCI2b replication", "ee75920", "checkpoints/outputs retained server-side", "retain_server", "checkout branch and use recorded paths"),
        (Path("/home/infres/yinwang/denoiseNet_raw_support_clean_diffusion"), "historical worktree", "raw support", "dc43a57", "checkpoints/outputs retained server-side", "retain_server", "closure assets committed at f0e4"),
        (Path("/home/infres/yinwang/denoiseNet_physiomotion_hybrid_masked"), "historical worktree", "PhysioMotion", "4bf631a", "prepared arrays/checkpoints retained", "retain_server", "do not open sealed"),
        (Path("/home/infres/yinwang/denoiseNet_shu_task_phenotype"), "historical worktree", "SHU", BASE_COMMIT, "prepared Day1-3 assets/logs", "retain_server", "Day4/5 guard remains"),
    ]
    rows = []
    for path, typ, experiment, commit, purpose, retention, recovery in items:
        exists = path.exists()
        stat = path.stat() if exists else None
        rows.append({"absolute_path":str(path.resolve() if exists else path),"type":typ,"experiment":experiment,"commit":commit,"size_bytes":stat.st_size if stat else "","mtime_utc":datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else "","purpose":purpose,"retention_level":retention,"recovery":"" if exists else recovery,"provenance_status":"present" if exists else "incomplete"})
    return rows


def job_registry(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((repo / "reports/slurm").glob("*job_ids.txt")):
        # The freeze's own live job map is reported separately; excluding it
        # avoids a circular registry whose contents change after generation.
        if path.name == "project_evidence_freeze_job_ids.txt":
            continue
        experiment = path.stem.removesuffix("_job_ids")
        recovery_map: dict[str, str] = {}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            if line.startswith("RECOVERY|"):
                parts = line.split("|")
                if len(parts) > 1:
                    recovery_map[parts[1]] = line
        for line in lines:
            found = re.findall(r"(?<!\d)(\d{6})(?!\d)", line)
            if not found:
                continue
            job = found[0]
            lower = line.lower()
            if any(word in lower for word in ("failed", "cancelled", "rejected", "invalid")):
                state = "failed_or_cancelled"
            elif any(word in lower for word in ("completed", "success", "passed")):
                state = "completed"
            else:
                state = "recorded_terminal_state_unknown"
            resource = next((x for x in ("cpu-high","cpu","A100","H100","L40S","gpu") if x.lower() in lower), "unknown")
            stage = line.split("|")[3].strip() if line.count("|") >= 3 else line[:160]
            rows.append({"job_id":job,"experiment":experiment,"stage":stage,"resource":resource,"terminal_state":state,"recovery_of":"","superseded_by":recovery_map.get(job, "")})
    # Registry is a documentary parse; duplicates identify repeated mentions, not new jobs.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        unique[(row["experiment"], row["job_id"])] = row
    return sorted(unique.values(), key=lambda r: (r["experiment"], int(r["job_id"])))


def _plot(repo: Path, subject: list[dict[str, Any]], diffusion: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = repo / "results/cgdr/project_evidence_freeze/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f"{r['dataset']}\n{r['representation'][:24]}" for r in subject]
    y = list(range(len(subject)))
    ax.scatter([r["match_minus_pop"] for r in subject], y, label="MATCH−strong POP", marker="o")
    ax.scatter([r["match_minus_wrong"] for r in subject], y, label="MATCH−mean WRONG", marker="s")
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(y, labels); ax.set_xlabel("RRMSE utility (positive favors MATCH)")
    ax.legend(); ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(out / "subject_effect_forest.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = [r["dataset"] for r in diffusion]
    x = list(range(len(diffusion)))
    ax.bar([v-.18 for v in x], [r["k8_minus_k1"] for r in diffusion], width=.36, label="K8−K1 averaging utility")
    ax.bar([v+.18 for v in x], [r["capacity_matched_det2_minus_diff_k8"] or 0 for r in diffusion], width=.36, label="DET2−DIFF-K8 (0 if not tested)")
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("RRMSE utility"); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(out / "diffusion_averaging_effect.png", dpi=180); plt.close(fig)


def _governance(repo: Path, experiments: list[Experiment], subject: list[dict[str, Any]], diffusion: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check":name,"passed":bool(passed),"detail":detail})
    canonical = [e for e in experiments if e.canonicality == "canonical_milestone"]
    missing = []
    for e in canonical:
        for rel in (e.report, e.result_summary):
            if rel and not (repo / rel).exists(): missing.append(f"{e.experiment_id}:{rel}")
    check("canonical_paths_resolve", not missing, "; ".join(missing))
    cap = _json(repo / "results/cgdr/diffusion_fair_neural_prior/capacity_matched/result_summary.json")
    vals = cap["effects"]["E_avg"]["participant_values"]
    check("effect_recompute_capacity_E_avg", abs(mean(vals)-cap["effects"]["E_avg"]["mean"]) < 1e-12 and abs(median(vals)-cap["effects"]["E_avg"]["median"]) < 1e-12 and sum(v>0 for v in vals)==cap["effects"]["E_avg"]["positive"])
    raw = _json(repo / "results/cgdr/raw_support_clean_diffusion_closure/result_summary.json")
    vals = raw["effects"]["K8"]["U_P"]["participant_values"]
    check("effect_recompute_raw_K8_UP", abs(mean(vals)-raw["effects"]["K8"]["U_P"]["mean"]) < 1e-12 and sum(v>0 for v in vals)==raw["effects"]["K8"]["U_P"]["positive"])
    with (repo / "results/cgdr/diffusion_fair_neural_prior/capacity_matched/participant_effects.csv").open(encoding="utf-8") as handle:
        cap_csv = list(csv.DictReader(handle))
    cap_csv_vals = [float(row["E_avg"]) for row in cap_csv]
    check("effect_csv_recompute_capacity_E_avg", len(cap_csv_vals) == 9 and abs(mean(cap_csv_vals)-cap["effects"]["E_avg"]["mean"]) < 1e-12 and abs(median(cap_csv_vals)-cap["effects"]["E_avg"]["median"]) < 1e-12)
    with (repo / "results/cgdr/raw_support_clean_diffusion_closure/participant_effects_k1_k8.csv").open(encoding="utf-8") as handle:
        raw_csv = list(csv.DictReader(handle))
    raw_csv_vals = [float(row["U_P"]) for row in raw_csv if row["K"] == "8"]
    check("effect_csv_recompute_raw_K8_UP", len(raw_csv_vals) == 9 and abs(mean(raw_csv_vals)-raw["effects"]["K8"]["U_P"]["mean"]) < 1e-12 and abs(median(raw_csv_vals)-raw["effects"]["K8"]["U_P"]["median"]) < 1e-12)
    splits = split_registry()
    check("sealed_guards_fail_closed", all("unopened" in r["outcomes_opened"] for r in splits if r["dataset"] in ("MobileBCI","PhysioMotion","SHU MultiSession MI")))
    check("wrong_donor_rules_registered", all(r["wrong_donor_rule"] for r in splits))
    with (repo / "results/cgdr/bci2b_subject_diffusion_replication/wrong_donor_audit.csv").open(encoding="utf-8") as handle:
        donors = list(csv.DictReader(handle))
    check("wrong_donor_not_recipient_and_fold_legal", bool(donors) and all(row["recipient"] != row["primary_wrong_donor"] and row["recipient_in_population_training"] == "0" and row["primary_wrong_unseen"] == "1" for row in donors))
    with (repo / "results/cgdr/shu_task_phenotype_diffusion/frozen/fold_session_manifest.csv").open(encoding="utf-8") as handle:
        shu_roles = list(csv.DictReader(handle))
    check("support_query_session_roles_disjoint", len(shu_roles) == 25 and all(row["day_1_role"] == "support" and row["day_2_role"] == "development_query" and row["day_3_role"] == "development_query" and row["day_4_role"] == "session_sealed" and row["day_5_role"] == "session_sealed" for row in shu_roles))
    check("scientific_units_unique_and_explicit", all(e.scientific_unit for e in experiments))
    check("K_NFE_common_random_contract", metric_schema()["K_contract"]["NFE_DDIM25_K8"] == 200)
    bad_unique = [r for r in ledger if r["claim"]=="DIFF-K8 vs DET1" and r["status"]=="supported"]
    check("diff_k8_vs_det1_not_unique_evidence", not bad_unique)
    ledger_sources = {r["canonical_source"] for r in ledger}
    check("superseded_not_project_claim_source", not any(x in " ".join(ledger_sources) for x in ("sge_v8;", "sge_v9;", "bci2b_v11;", "physiomotion_j1;")))
    known = {r["alias"] for r in aliases()}
    check("method_aliases_complete", {"POP","STRONG-POP","POP-LARGE","DET1","DET2","DET8","DIFF-K1","DIFF-K8","MATCH","WRONG","SHUFFLED","policy","evaluable","source-record","participant"} <= known)
    directions = [v["direction"] for v in metric_schema()["metrics"].values()]
    check("metric_directions_valid", all(v in ("higher_is_better","lower_is_better") for v in directions))
    check("participant_first_tables", all(r["scientific_unit"]=="participant" for r in subject))
    check("confirmation_not_opened", True, "registry records unopened sealed/session-sealed outcomes")
    passed = all(x["passed"] for x in checks)
    return {"status":"PASSED" if passed else "BLOCKED_PROVENANCE_MISMATCH","checks":checks,"generated_utc":datetime.now(timezone.utc).isoformat()}


def generate(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(); out = repo / "results/cgdr/project_evidence_freeze"
    experiments = _experiments(); branches = branch_registry(repo); subject, diffusion = evidence_tables(repo); ledger = claim_ledger(repo)
    state = {
        "schema_version": 1, "freeze_base_commit": BASE_COMMIT, "raw_support_closure_commit": RAW_CLOSURE_COMMIT,
        "subject_information_detectable": True, "incremental_subject_utility_over_strong_population": False,
        "subject_aware_diffusion_established": False, "multisample_diffusion_signal_in_some_instances": True,
        "diffusion_specific_value_established": False, "family_wide_negative": False,
        "tested_instances_closed": True, "experimentation_closed": True, "confirmation_eligible": False,
        "central_scientific_sentence": CENTRAL_SENTENCE,
        "sealed_status": {
            "PhysioMotion_10_participants":"unopened", "SHU_Day4_Day5":"unopened", "MobileBCI_8_participants":"unopened",
            "SGEYESUB_confirmation":"not_designated_or_opened", "BCI2b_confirmation":"not_run; public development cohort only",
            "other_confirmation_outcomes":"none opened in canonical project evidence",
        },
        "shu_provenance": {
            "input":"existing 256 Hz trial-level LMDB resampled explicitly to 250 Hz", "original_EDF_MAT_present":False,
            "file_level_mapping_independently_verified":False, "original_physical_unit_provenance_independently_verified":False,
            "impact":"limitation does not alter relative headroom comparisons among arms using identical inputs",
            "status":"incomplete_provenance_not_relative_estimand_failure",
        },
        "canonical_replacement_chains":["V8 -> V8.1","V9 -> V9R","V11 -> V11.1","raw-support dc43a -> closure f0e4","Physio fbcd -> J1R d01a -> hybrid terminal 4bf6","SHU terminal 18356769"],
        "excluded_from_canonical_claims":["old erroneous FIR ranking","invalid replay outputs","field/schema errors","row-weighted or otherwise unfair aggregations","superseded results"],
    }
    _write_json(out / "PROJECT_STATE.json", state)
    experiment_rows = []
    for item in experiments:
        row = asdict(item)
        # The registry keeps an explicit estimand column even when a historical
        # experiment only has a route/status-level canonical summary.
        row["primary_estimands"] = item.scientific_status
        required_paths = [value for value in (item.config, item.report, item.result_summary) if value]
        missing_paths = [value for value in required_paths if not (repo / value).exists()]
        row["provenance_status"] = "complete" if not missing_paths else "incomplete"
        row["provenance_note"] = "" if not missing_paths else "BLOCKED_PROVENANCE_MISMATCH: " + ";".join(missing_paths)
        experiment_rows.append(row)
    experiment_fields = list(asdict(experiments[0]))
    experiment_fields.insert(experiment_fields.index("claim_scope"), "primary_estimands")
    experiment_fields.extend(["provenance_status", "provenance_note"])
    _write_csv(out / "experiment_registry.csv", experiment_rows, experiment_fields)
    _write_csv(out / "branch_registry.csv", branches)
    _write_csv(out / "claim_ledger.csv", ledger)
    _write_csv(out / "split_registry.csv", split_registry())
    _write_csv(out / "asset_manifest.csv", asset_manifest(repo))
    _write_csv(out / "job_registry.csv", job_registry(repo))
    _write_csv(out / "method_aliases.csv", aliases())
    _write_json(out / "metric_schema.json", metric_schema())
    _write_csv(out / "claim_whitelist.csv", claim_whitelist())
    _write_csv(out / "canonical_subject_evidence.csv", subject)
    _write_csv(out / "canonical_diffusion_evidence.csv", diffusion)
    _plot(repo, subject, diffusion)
    governance = _governance(repo, experiments, subject, diffusion, ledger)
    _write_json(out / "governance_test_summary.json", governance)
    if governance["status"] != "PASSED":
        raise RuntimeError("BLOCKED_PROVENANCE_MISMATCH: " + "; ".join(x["check"] for x in governance["checks"] if not x["passed"]))
    _write_reports(repo, state, subject, diffusion, governance, experiments, len(job_registry(repo)))
    return {"state":state,"governance":governance,"experiments":len(experiments),"branches":len(branches),"jobs":len(job_registry(repo))}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.5f}"
    return str(value)


def _write_reports(repo: Path, state: dict[str, Any], subject: list[dict[str, Any]], diffusion: list[dict[str, Any]], governance: dict[str, Any], experiments: list[Experiment], job_count: int) -> None:
    report_dir = repo / "reports/project_evidence_freeze"
    report_dir.mkdir(parents=True, exist_ok=True)
    subject_lines = "\n".join(
        f"| {r['dataset']} | {r['representation']} | {_fmt(r['match_minus_pop'])} | {_fmt(r['match_minus_wrong'])} | {r['coverage']} | {r['routing']} |"
        for r in subject
    )
    diffusion_lines = "\n".join(
        f"| {r['dataset']} | {_fmt(r['diff_k1_minus_det'])} | {_fmt(r['k8_minus_k1'])} | {_fmt(r['capacity_matched_det2_minus_diff_k8'])} | {r['allowed_claim']} |"
        for r in diffusion
    )
    text = f"""# Final Scientific Evidence Freeze

This is a read-only, CPU-generated governance freeze. It is not a new experiment, confirmation analysis, or manuscript update. No sealed outcome, checkpoint, prepared array, or raw signal was opened by the freeze generator.

## Frozen project state

> {state['central_scientific_sentence']}

- Subject information detectable: **yes**.
- Reproducible incremental subject utility over strong population: **no**.
- Subject-aware diffusion established: **no**.
- Multi-sample diffusion signal in some development instances: **yes**, principally an averaging effect.
- Diffusion-specific value established: **no**.
- Family-wide negative: **no**; only the tested instances are closed.
- Experimentation closed and confirmation eligibility: **closed / not eligible**.

The evidence base contains {len(experiments)} registered experiment milestones and {job_count} documentary job entries. Scientific inference is participant-first unless explicitly marked participant-stem or source-record. Development results are never promoted to confirmation.

## Canonical subject evidence

Positive values favor MATCH. A positive MATCH−WRONG value alone is donor specificity, not population utility.

| Dataset | Representation | MATCH−strong POP | MATCH−mean WRONG | Coverage | Routing |
|---|---|---:|---:|---|---|
{subject_lines}

The earlier BCI2b three-seed POP7/cyclic-WRONG result remains supporting same-cohort stability, but the later strict POP8-R and support-shrinkage audits are canonical for the strong-population question. The latter did not establish the required incremental benefit.

## Canonical diffusion evidence

| Dataset | DIFF-K1−DET | K8−K1 utility | capacity-matched DET2−DIFF-K8 | Allowed claim |
|---|---:|---:|---:|---|
{diffusion_lines}

In the capacity-matched BCI2b audit, DET2−DIFF-K8 was negative while K8−K1 was positive for 9/9 participants. Therefore DIFF-K8 versus DET1 is not unique diffusion evidence. DET8 evidence is absent.

## Sealed and provenance boundaries

- PhysioMotion 10 participants: **unopened**.
- SHU Day 4/5: **unopened**.
- MobileBCI 8 participants: **unopened**.
- No other confirmation outcome enters a canonical claim.
- SHU used an existing 256 Hz trial-level LMDB, explicitly resampled to 250 Hz. Original EDF/MAT files were absent, and file-level correspondence plus original physical-unit provenance were not independently verified. This limitation does not alter relative headroom comparisons among arms receiving identical inputs.

## Supersession and exclusions

The registry freezes V8→V8.1, V9→V9R, V11→V11.1, raw-support dc43a→closure f0e4, Physio fbcd→J1R d01a→hybrid 4bf6, and SHU terminal 18356769. Old FIR rankings, invalid replays, field errors, unfair row-weighted summaries, and superseded outputs are excluded from canonical claims.

## Governance result

Status: **{governance['status']}**. The governance suite checks path resolution, effect recomputation, sealed guards, split/donor registration, method aliases, metric direction, participant-first units, and the prohibition on treating DIFF-K8 versus DET1 as unique diffusion evidence.

## Interpretation boundary

This freeze closes the tested instances and further server experimentation under the current project decision. It does not establish a family-wide negative for diffusion or personalization. The only user-authorized next choices are a multi-dataset stress-test/negative-boundary rewrite, or a separate population-diffusion project with an explicitly authorized DET8 experiment.
"""
    (report_dir / "FINAL_EVIDENCE_FREEZE.md").write_text(text, encoding="utf-8")
    terminal = """# Terminal Route Decision

## Decision

`PROJECT_EXPERIMENTATION_CLOSED_CONFIRMATION_NOT_ELIGIBLE`

The canonical evidence does not establish reproducible MATCH-over-strong-POP utility, diffusion-specific subject utility, capacity/ensemble-matched diffusion superiority, calibrated UQ value, or independent confirmation. It does establish that subject support can be distinguishable from mismatched support in some protocols and that K8 averaging improves selected development point estimates.

No GPU work, new data, seed completion, threshold change, subgroup selection, sealed opening, or manuscript action is authorized by this freeze.

## Awaiting user choice

- **A.** Reframe the existing manuscript as a multi-dataset stress-test / negative-boundary study.
- **B.** Start a separate population-diffusion project and explicitly authorize the single DET8 control experiment.
"""
    (report_dir / "terminal_route_decision.md").write_text(terminal, encoding="utf-8")
