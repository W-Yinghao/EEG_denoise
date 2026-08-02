# Invalid pre-scope-isolation run

Jobs `919785`--`919787` are not scientific evidence.  Job `919785` selected a
single deterministic U-Net checkpoint using validation cells that mixed
population, matching-calibration, and query-clean-derived oracle projector
conditions.  This did not cross source-record splits, but the oracle condition
could influence checkpoint selection for the deployable population and
matching arms, violating the same-information comparison.

The completed training log and checkpoint are retained for audit only.  The
development array `919786` and dependent aggregate `919787` were cancelled
after this issue was found; any partial per-record outputs are invalid.  The
replacement protocol uses separate population, matching, and non-deployable
oracle U-Net checkpoints, each trained and selected only within its own
operator-information scope and written under a new result root.
