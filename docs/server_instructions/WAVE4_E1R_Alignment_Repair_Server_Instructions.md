# WAVE4-E1R — One registered repair of the optical-correspondence instrument
### Server execution instructions (Claude Code)

E1's verdict stands as banked (0/3 → 0/26 at the frozen gate; run-1 defects disclosed and
corrected without touching gate constants — exemplary). This is the ONE registered repair
round (v19→v20 pattern): the diagnosis says the failure is substantially
instrument-definition (mis-centered lag window vs known blink kinematics; a flanking
clause that rejects real blinks under 59% tracker invalidity) and gate-miscalibration
(one gate sized to the most demanding customer killed customers that need less). If the
repaired instrument also fails its weakest tier, the optical instrument is CLOSED on
this panel permanently and the four WAVE3 customers are documented as
instrument-limited-unserved. No second repair.

---

# 1. Workspace

```text
continue : branch codex/wave4-optical (tip = the STOP-1 commit), CPU only
frozen   : the clock layer (drift fits, 26/35 clock-valid set, the 4 discontinuity and
           5 no-sync exclusions) is UNTOUCHED — it passed; only the correspondence
           instrument and the E2 gate structure are repaired
```

# 2. Preregistration addendum (`reports/wave4_e1r_preregistration.md`, commit BEFORE running)

```text
REPAIR 1 — PHYSIOLOGICAL LAG WINDOW (frozen from literature, not from the banked lags):
  the tracker loses the eye at lid-closure onset; the VEOG peak occurs near maximal
  closure; blink kinematics put the expected tracking-loss→VEOG-peak lag at ~+30..+100 ms.
  Frozen window: [−20, +120] ms. The banked 30–55 ms matched-lag median is cited as
  consistent-with, never as the source of, this window.

REPAIR 2 — FRAGMENTATION-ROBUST BLINK SEGMENTATION: merge invalid runs separated by
  ≤40 ms of valid samples; blink candidate = merged run with duration ∈ [50, 500] ms;
  pupil-diameter collapse (if exported) as auxiliary evidence, never required. The ≥50 ms
  flanking clause is replaced by the duration+merge criterion (the clause was correct for
  a clean tracker and wrong at 59% invalidity — stated in the report).

REPAIR 3 — REVERSE INSTRUMENT (co-primary, segmentation-free): VEOG-anchored
  correspondence — for each VEOG-detected blink, the Tobii invalidity fraction inside
  the frozen window must be elevated vs a circular-shift null (per recording:
  mean elevation with bootstrap CI; null = ≥200 circular shifts). This measures
  correspondence without segmenting Tobii blinks at all.

REPAIR 4 — ELIGIBILITY GATING: recording-level Tobii validity fraction ≥ 0.40 (frozen;
  full sweep 0.30–0.60 reported, verdict on the frozen value); excluded recordings
  counted, never repaired. Clock exclusions unchanged.

TIERED E2 GATES (the structural fix — gates sized per customer):
  TIER-S (segment-level; unlocks M2, M4, and the segment-level variant of M3):
    reverse-instrument elevation CI-low > 0 in ≥60% of eligible recordings,
    pooled elevation CI-low > 0.
  TIER-E (event-level; unlocks M1 and full M3):
    forward blink match (repaired segmentation, physiological window) median ≥ 0.70
    across eligible recordings AND per-recording circular-shift null p < 0.01 in ≥60%
    of them. (The old 0.80-at-±50ms gate is retired with its rationale stated.)
  CLOSE RULE: Tier-S fail ⇒ optical instrument CLOSED on this panel; all four customers
  documented instrument-limited; no further repairs, no re-download, no new panels
  this wave.

Statistics: per-recording first, then subject-level aggregation; sweeps reported,
verdicts on frozen values; no gate constant may move after this commit. The banked E1
verdict is never edited — E1R is a new instrument row beside it.
```

# 3. Execution

```text
1. commit the addendum
2. rebuild blink segmentation + reverse instrument (CPU, reuse the aligned streams)
3. run E1R on all 26 clock-valid recordings; apply eligibility; produce the
   per-recording correspondence table (forward match, reverse elevation, validity,
   eligibility, exclusion reasons)
4. apply the tiered gates → decision JSON {tier_s, tier_e, eligible_n, close_fired}
5. STOP and report the table + verdicts verbatim. E2 measurements (which of M1–M4 run,
   and in which variant) are authorized by the tier verdicts but WAIT for the operator
   read before running.
```

Deliverables: `reports/wave4_e1r_{preregistration, report}.md`,
`results/wave4_optical/e1r/{correspondence_table.csv, decision.json}`, one ledger line.

# 4. Kickoff prompt

```text
Read WAVE4_E1R_Alignment_Repair_Server_Instructions.md in full and execute it. Continue
on codex/wave4-optical. Commit the preregistration addendum FIRST (physiological lag
window frozen from blink kinematics, fragmentation-robust segmentation, the reverse
VEOG-anchored instrument as co-primary, eligibility gating, tiered E2 gates, and the
close rule). Rerun the correspondence layer only — the clock layer is frozen and
untouched. Apply the tiered gates, write the decision JSON and the per-recording table,
commit, push, and STOP before running any E2 measurement. CPU only; the banked E1
verdict and all WAVE3 numbers are read-only.
```
