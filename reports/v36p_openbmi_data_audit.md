# V36P OpenBMI data audit

The frozen source is the existing project datalake cache
`/projects/EEG-foundation-model/datalake/processed/4704743c/Lee2019_MI`, produced through
MOABB's `Lee2019_MI` dataset adapter. No download was performed for V36P.

The cache contains 108 recordings: 54 participants × two sessions, one run and 100 registered
left/right motor-imagery trials per participant/session. The producer contract is EEG-only,
62 channels, 200 Hz, 0.5–45 Hz, with registered trial interval 0–4 s. V36P freezes that recipe and
uses the 800 samples beginning at each registered event. Each trial/channel is centered and
standardized only from that trial before EEGNet input. Event code 1 maps to right hand and code 2
to left hand. No channel, band, interval, or preprocessing recipe will be changed after outcomes
are observed.

The machine-readable dataset inventory binds `metadata.parquet`, `events.parquet`, and
`infos.json` by SHA256, records original subject/session mappings and every recording path, and
validates all 10,800 events. The six-fold manifest proves a 36/9/9 train/validation/test split and
counts every participant exactly once in outer test.

License/provenance status: Lee et al. (2019) OpenBMI/Lee2019-MI, accessed from the existing local
MOABB cache and its producer metadata. This repository does not redistribute raw or processed
EEG arrays.
