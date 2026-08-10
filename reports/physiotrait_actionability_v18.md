# PhysioTrait-Actionability Gate v18

CPU-only, no-network development gate. The immutable BrainID Gate-01 and Gate-01R failures remain unchanged. No identity verifier endpoint was used.

## Protocol

- Data: DATA_PROTOCOL_VALID; 15/15 longitudinal participants; Day-1 support, Day-7 restoration query, Day-80 independent physiological gallery.
- Official metadata fixed event 1 as target/client-photo ERP and event 2 as non-target/generated-photo VEP before Day-7/Day-80 signals were evaluated.
- Official code defines the complete 0–600 ms post-stimulus epoch but no independent prestimulus baseline field. The preregistration therefore discloses the fixed −50–0 ms trial baseline rather than attributing it to an official field.
- All 57 common channels and the full post-stimulus interval entered the primary trait. Trait blocks were equally weighted after outer-fold scaling.
- Day-200, PhysioMotion sealed, and SHU Day-4/5 remained unopened.

## Cross-day trait headroom

- Decision: `CROSS_DAY_PHYSIOTRAIT_HEADROOM_NO_GO`.
- Primary H_P mean/median/positive/p: 0.819764 / 0.373587 / 13/15 / 0.013184; descriptive CI 0.211750–1.476538.
- Primary H_W mean/median/positive/p: 1.050408 / 1.048907 / 12/15 / 0.001617; descriptive CI 0.500149–1.606559.
- All three primary blocks were directionally positive (morphology: H_P=2.0606, H_W=2.5121; topography: H_P=0.1644, H_W=0.2492; dynamics: H_P=0.2343, H_W=0.3899). Gain-normalized sensitivity stayed positive, and PRESTIM/HF_ART/LABEL_SHUFFLE did not explain primary H_P.
- Hard failure: TIME_SHUFFLE H_W remained stable after post-stimulus temporal order was destroyed (mean 0.046003, median 0.040955, 12/15 positive, p=0.002502). This leaves a channel/spatial-statistics shortcut compatible with the observed donor separation.
- Failed criteria: time_shuffle_H_W_not_stable.

## Conditional restoration actionability

- NOT_RUN after the preceding hard gate failed.

## Final route

- `PASS_TRAIT=false`; headroom=FAIL; actionability=NOT_RUN.
- This constrains only the frozen v18 longitudinal ERP trait instance and is not a family-wide negative.
