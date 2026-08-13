# EEGDfus multichannel architecture delta

The local `EEGDfus-MC` port retains the official algorithmic skeleton:

- two streams for noised clean state and contaminated condition;
- one initial two-convolution block and three encoder layers per stream;
- four noise-level FiLM bridges from the noisy stream into the condition stream;
- epsilon prediction, linear 500-step schedule, beta 1e-4 to 0.02, Adam family;
- conditional input at every reverse step.

Registered differences are limited to the admitted waveform contract:

| Component | Official | V40R port | Reason |
|---|---:|---:|---|
| signal channels | 1 | 46 | frozen SGEYESUB montage |
| output channels | 1 | 46 | multichannel clean estimate |
| window samples / attention model dimension | 512 | 256 | existing audited query window |
| sampling | 500-step ancestral | 25-step DDIM | registered V40R development sampler |
| support | none | two zero-initialized FiLM sites | V40R incremental method |

No energy guidance, posterior correction, routing, operator portfolio, privacy module, or alternate
diffusion objective is introduced. The multichannel port is not labelled unchanged official code.
