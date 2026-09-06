# V25 project transition

The authoritative ledger v1.1 was read before V25 execution. It records V24 as the
latest completed stage, V25 as the only active route, and all confirmation payloads
as unopened. V25 is consistent with the project mainline: query-disjoint raw EEG+EOG
support, a strong population anchor, a learned support-conditioned residual, and
low-dimensional residual diffusion.

V24 provenance is reconciled without modifying V24: coordinate audit
`bf840a72229622ff1a311dfa5b2686d46444cd69`, core implementation
`2e5b0caf8108201e6c6e177ab95908f7b7075a71`, packaging/latency
`2750552e85263ba262c4276cb83cc6c6a0ec6e1f`, result-producing
`55a078214eddcc35bec7046441784eedcdf673ab`, and terminal
`8dadb508fd2d50a089246c4e11c83b7b7628fa42`. The V24 terminal manifest's
`implementation_commit` label points to packaging/latency; this note supplies the
precise role without altering history.

Canonical V24 natural utilities are artifact `-0.373885` and preservation
`-0.232186`. V25 does not recompute or rewrite them. Its sole scientific change is to
replace the harmful fixed analytic deviation decoder with a raw support-set learned
context and spatial decoder. Sealed data, A-track, and `taas_submission/**` remain
read-only.

## Terminal ledger transition

After V25 evaluation, the authoritative ledger was upgraded without deleting v1.1 history to v1.2 at commit `b4948537123535ff46acdfb190fd3e6725fe3040`. It records the small deterministic raw-support development signal, the uniformly negative residual-diffusion increment, the failed natural trade-off, and the decision not to open confirmation.
