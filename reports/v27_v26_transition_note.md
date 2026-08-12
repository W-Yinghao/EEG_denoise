# V27 transition from frozen V26

- Base/terminal: `7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c`
- V26 implementation: `8257bf0`
- V26 Round A: `c3eeb4a`
- V26 Round B/Natural/Ledger: `9a1c469`
- Frozen checkpoints: 62
- Targeted tests: 19 passed
- Clean-archive tests: 19 passed
- Query auxiliary inference reads: 0
- Sealed reads: 0

V26 terminal manifest's `push_status=pending_terminal_commit` is a self-referential packaging field. It does not affect science; V26 local/remote parity was verified at `7af5a007…`. V27 will report remote/local parity after its terminal commit without rewriting V26.

V27 does not treat `DIFF > DET` as a retention threshold. It evaluates the same energy on deterministic and diffusion candidates, with natural artifact–preservation validity prioritized.
