# Bootstrap state

> **Scope revision, 2026-08-01:** this file preserves the initial bootstrap
> snapshot. Its full-root inventory/download embargo was later superseded by
> the user's lightweight private-project workflow. Inventory `918918` was
> cancelled; current data status is in `reports/data_inventory/summary.md` and
> `datasets/registry/`.

Recorded: 2026-08-01 (Europe/Paris)

## Fixed roots

- Code root: `/home/infres/yinwang/denoiseNet`
  - `realpath`: `/home/infres/yinwang/denoiseNet`
  - owner/group/mode: `yinwang:comelec`, `0755`
  - mount: `/home/infres`, NFS (`ssd.enst.fr:/data/ir800`), read/write
  - observed capacity: 84 TiB total, 28 TiB available, 67% used
- Data root: `/projects/EEG-foundation-model`
  - `realpath`: `/projects/EEG-foundation-model`
  - owner/group/mode: `root:part_dig_eeg`, `2770`
  - mount: `/projects/EEG-foundation-model`, NFS4 (`nodenvme03.enst.fr:/zdata/EEG-foundation-model`), read/write
  - observed capacity: 30 TiB total, 1.5 TiB available, 96% used
  - one top-level symlink was observed: `tuh_eeg_abnormal` resolves within the fixed data root to `datalake/raw/tuh/tuh_eeg_abnormal`

The root checks were lightweight control-plane operations only. No recursive data scan, hashing, parsing, or write under the data root was performed on the login node.

## Conflict with stated initial condition

The execution instruction describes the known initial state as an empty local code directory and an empty remote repository, subject to re-verification. Re-verification found:

- The remote repository returned no heads or tags (`git ls-remote --heads --tags` exit code `0`, empty stdout), consistent with an empty remote.
- The local code directory is **not empty** and is an existing Git worktree on branch `master`.
- Local HEAD is `2f408f3cdc4895347b6a567b913159b1c60a0b50` (`Improve conditional denoiser: x0-prediction + conditional-SDEdit sampling`, authored 2026-06-04).
- The worktree already contains many tracked modifications and untracked source, result, manuscript, figure, archive, and dataset-reference files. These are treated as pre-existing/concurrent user work and will not be removed, reset, overwritten, or reformatted.
- No Git remote was configured. After verifying that the specified remote had no refs, `origin` was added as `https://github.com/W-Yinghao/EEG_denoise.git`. No fetch content existed, no push was attempted, and no branch/default-branch/release operation was performed.

Raw remote-ref observation:

```text
$ git ls-remote --heads --tags https://github.com/W-Yinghao/EEG_denoise.git
<empty stdout>
exit_code=0
```

## Preservation decision

The empty-repository bootstrap path is not applicable. Work will proceed by auditing and extending the existing repository in place. Before any model implementation, the required attachment review, repository audit, requirement traceability, and minimal change map will be produced. Existing experiment outputs are not accepted as evidence until their provenance, execution environment, data, splits, and scientific semantics have been audited against the current instruction.

The data root is already heavily populated and at 96% utilization. No dataset will be classified as `missing`, downloaded, unpacked, moved, or published until the scheduled full-root read-only inventory and license/source review are complete. Disk headroom must be estimated before any approved derived-data write.
