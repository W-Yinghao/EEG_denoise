# runs/ — SLURM run records

One directory per submitted job (`<experiment>_<jobid>`), holding the resolved config, result summary and per-run metadata the launchers wrote. This is the audit trail behind `results/`; nothing reads it programmatically. Large intermediates were never tracked.
