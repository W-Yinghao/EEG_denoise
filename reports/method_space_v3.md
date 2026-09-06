# Method space v3

The earlier route-level no-go covers analytic full-C/projector/FIR bridges and
the tested guidance/SDEdit instances. It does not cover raw calibration sets,
support-only parameter adaptation, or selective deployment of independently
trained population and matching cleaners.

| Route | Subject information carrier | What changes at inference | Main falsification |
|---|---|---|---|
| P-A raw-support tokens | Query-disjoint calibration EEG/EOG windows | Shared prompt/cross-attention or FiLM context in one checkpoint | MATCH fails to beat POP and three compatible WRONG donors |
| P-B direct support adapter | Calibration loss gradients | Small zero-initialized LoRA/adapter on frozen population diffusion; each diffusion/deterministic adapter is retained only when the disjoint support-validation half improves its own matched target | Direct adaptation upper bound fails before hypernetwork development |
| P-C selective diffusion | Support quantiles plus query-EEG-only activity/disagreement/dispersion | Discrete POP/MATCH/abstain action | Hindsight ceiling has no utility or deployable selector loses preservation |
| P-D support-stat control | Support channel statistics | Fixed EA/DSBN/ReVIN-style normalization | Cheap statistics explain the same apparent subject effect |

Every route uses the same frozen participant/source splits, no query-time EOG,
the same query windows, one checkpoint for MATCH/POP/WRONG interventions where
applicable, and participant/source-record aggregation. Negative routes remain
valid scientific results and do not cancel independent routes.

Ranking reports five separate axes: absolute validity, diffusion increment,
subject utility, wrong-donor specificity, and neural safety. No family-wide
negative claim is permitted.
