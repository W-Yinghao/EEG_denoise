# V36P training-exemplar exposure

All 16 registered releases per query were retained. Exact-copy is defined as bytewise float32 row
membership in the outer-training fiber bank. The initial primary-array distance implementation
used `pairwise_distances`, which returns roughly `1.3e-7` even for byte-identical rows; those
threshold rows remain in immutable fold JSON, while the aggregate exact-copy field supersedes them
with the structural bytewise definition.

| Method | Exact copy | Near copy | Nearest training distance | Membership/exposure probability |
|---|---:|---:|---:|---:|
| Fiber-Gaussian | 0.000 | 0.000 | 6.8087 | 0.000090 |
| Fiber-SANDiff | 0.000 | 0.000 | 5.6884 | 0.002392 |
| Fiber-Stratified-Resample | 1.000 | 1.000 | numerical ~1.37e-7 / structural 0 | 0.998454 |

Participant bootstrap intervals were exactly [0,0] for Gaussian/SANDiff exact and near copies and
[1,1] for Resample. SANDiff therefore sharply reduces direct training-exemplar exposure relative
to bank resampling. Gaussian does so as well and has both larger nearest-training distance and a
lower registered exposure score than SANDiff.

The membership score is one registered nearest-training-distance attack. Zero copies do not imply
formal membership privacy, and these results do not establish anonymity.
