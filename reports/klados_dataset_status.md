# Klados–Bamidis dataset status

## Outcome

- Version 1 is unusable: official UnRAR 7.23 can list three members, then reports `Unexpected end of archive`. The archive is 46,757,186 bytes, while its listed packed members already total 52,420,525 bytes.
- Version 4 is available at `/projects/EEG-foundation-model/klados_bamidis/v4`: four direct MAT files, 53,659,298 bytes total.
- No dataset hashes were computed. Version 4 was checked only by official file size and `scipy.io.whosmat` directory reads.

## Slurm evidence

| Job | Action | Result |
|---|---|---|
| 919343 | Check installed archive tools | No system UnRAR/7-Zip found |
| 919351 | Fetch official UnRAR 7.23 and first listing attempt | Tool installed privately; command mismatch fixed |
| 919356 | List v1 with official UnRAR 7.23 | Failed: truncated archive |
| 919357 | Plan official v4 direct files | 4 files, 53,659,298 bytes |
| 919359 | Download and inspect v4 MAT directories | Completed in 7 seconds |
| 919362 | Recheck frozen v4 file IDs/sizes and MAT directories | Completed; already present |

The UnRAR binary and license are kept under ignored `runs/tools/`; they are not committed or copied into the data root. The corrupt v1 archive is retained unchanged as evidence.
