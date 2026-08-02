# Klados Stage-3 v4 exploratory result

Status: `completed_descriptive_no_broad_classifier`

Slurm jobs 919854 (three operator-scope-specific U-Net checkpoints), 919857
(eight development source records), and 919876 (aggregation) completed.  This
is a development-set diagnostic on sim31--sim36, sim44, and sim45.  These are
source records, not verified independent participants.  The result is
exploratory, is not fresh confirmatory evidence, and is neither formal G1 nor
formal G3.

## Eligibility and fallback coverage

The checkpoint manifests use one common matching-P0 eligibility rule.  Of 30
requested training source records, 15 were eligible (sim04, sim06--sim10,
sim12, sim15--sim16, sim19, sim24--sim26, sim28, and sim30).  Of the eight
development records, six were eligible (sim31, sim33, sim34, sim36, sim44,
and sim45); sim32 and sim35 were ineligible and used population-projector
fallback under the all-requested deployment-policy estimand.

All 144 aggregated method-by-record rows completed successfully, with no
retained or per-seed failures.  Population and query-derived-oracle scopes
cover 8/8 records.  Matching operator-effect summaries below cover only the
common 6/8 eligible records and contain no fallback rows.  The separate
all-requested matching-policy summary retains all 8/8 records, including its
2/8 population fallbacks; it must not be interpreted as a pure matching-P0
effect.

## Exact aggregate medians

The values below are copied from
`operator_effect_eligible_summary.csv`.  Lower is favorable for `e_parallel`,
`e_perp`, RRMSE, and PSD distortion; higher is favorable for correlation.
`deterministic_Qy` denotes hard-Q consistency with the operator named in the
scope column.  Only its query-derived-oracle row uses query clean truth and is
therefore the non-deployable oracle mechanism diagnostic.

| Operator scope | Method | e_parallel | e_perp | RRMSE | Correlation | PSD distortion |
|---|---|---:|---:|---:|---:|---:|
| population, 8/8 | M1 warm start | 1.2882429964413218 | 0.2885846159191456 | 0.8360691555999437 | 0.8546603319918951 | 0.894657526563567 |
| population, 8/8 | M2 final hard-Q | 1.9414856786003998 | 0.035889377086679525 | 1.2395556592185342 | 0.7788886138574047 | 0.4066296079168399 |
| population, 8/8 | M4 stepwise soft proximal | 5.665473391447796 | 0.06605206841533068 | 3.340256304059113 | 0.8092080677833112 | 8.25476943680393 |
| population, 8/8 | deterministic hard-Q | 0.9975819633700774 | 0.026525931441050345 | 0.6203157432586724 | 0.9300462244542075 | 0.4325505705710805 |
| population, 8/8 | deterministic soft proximal | 0.8358753629483102 | 0.013262965673008668 | 0.5236674118077405 | 0.9482156895620155 | 0.2252071797927247 |
| population, 8/8 | paired-supervised U-Net | 0.4376192388352601 | 0.006839864675065077 | 0.2504793537706736 | 0.9887506605822598 | 0.13182858643706108 |
| matching P0 eligible-only, 6/8 | M1 warm start | 1.4297580492574546 | 0.3021323604083431 | 0.9062943650234252 | 0.8395620753459181 | 1.0231820150929025 |
| matching P0 eligible-only, 6/8 | M2 final hard-Q | 2.082083311506814 | 0.2512213257282543 | 1.3338880000530022 | 0.7401360702411522 | 0.4685423213914329 |
| matching P0 eligible-only, 6/8 | M4 stepwise soft proximal | 5.5703344630885585 | 0.6229333294213587 | 3.3208111008877754 | 0.7852141015242519 | 9.175757996271246 |
| matching P0 eligible-only, 6/8 | deterministic hard-Q | 0.9990782840139942 | 0.1821969791364359 | 0.6212914935815534 | 0.9114767399075974 | 0.4253084958757284 |
| matching P0 eligible-only, 6/8 | deterministic soft proximal | 0.8974729081916435 | 0.09109848958718475 | 0.5504801377910311 | 0.9406878218459442 | 0.23694866618637342 |
| matching P0 eligible-only, 6/8 | paired-supervised U-Net | 0.4272797895333628 | 0.0073332385595930825 | 0.2510562858327994 | 0.9887425477638275 | 0.1096285651602383 |
| query-derived oracle, 8/8 | M1 warm start | 1.2882431394894613 | 0.2885845978459516 | 0.8360692397848787 | 0.8546603695405479 | 0.8946575254797253 |
| query-derived oracle, 8/8 | current M2 final hard-Q | 1.9381322230877687 | 1.7178741158471522e-07 | 1.2370667488240048 | 0.7737107825861899 | 0.41329253668453025 |
| query-derived oracle, 8/8 | M4 stepwise soft proximal | 5.688949699686007 | 0.012159296651998087 | 3.3535329076072133 | 0.8039797570524303 | 8.349044236681898 |
| query-derived oracle, 8/8 | deterministic oracle Qy | 1.0 | 5.660542433861966e-08 | 0.6212796382374934 | 0.9284673592970738 | 0.4361911826322292 |
| query-derived oracle, 8/8 | deterministic oracle soft proximal | 0.83650159140475 | 5.6605424337668837e-08 | 0.5239680918487174 | 0.9499363329783013 | 0.22819042792025662 |
| query-derived oracle, 8/8 | paired-supervised U-Net | 0.4352338427628272 | 0.007104709875892776 | 0.24943344492128816 | 0.9891250936118754 | 0.13215327513389669 |

## Interpretation boundary

Within this development diagnostic, M1 is measurably different from current
M2: under both population and query-derived-oracle geometry it has lower
median `e_parallel` and RRMSE and higher correlation, while incurring much
larger `e_perp`.  M4 performs poorly in the frozen configuration.  The
paired-supervised U-Net is a strong reference, but it uses contaminated-to-clean
supervision whereas M1/M2/M4 use the existing unconditional clean prior, so
this table is not a fair same-supervision diffusion-necessity comparison.

Accordingly, these rows preserve the existing instance-level status
`current_M2_no_incremental_value`; they do not establish a conclusion about
conditional diffusion, dual-prior diffusion, alternative objectives or
samplers, or the diffusion family.  The diffusion-family status remains
`not_tested`.  Query-derived oracle geometry is a non-deployable mechanism
upper bound, and no population-level, participant-level, formal G1, or formal
G3 claim is made from these development source records.

Machine-readable outputs remain under
`results/cgdr/klados_stage3_deterministic_scope_isolated_v4/development/`.
