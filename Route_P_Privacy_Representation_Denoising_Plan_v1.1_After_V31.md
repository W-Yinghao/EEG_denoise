# Route P v1.1 — Privacy-Aware EEG Representation Denoising
## Literature Frontier, Route Comparison, Method Scope, and Experimental Plan

**Status:** revised after V31 exact-duration reconciliation  
**Primary recommendation:** retain Route P, but separate stable identity from session/acquisition context and revise the role of support  
**Backup:** retain V27 as the waveform-level denoising route  

---

# 1. Executive decision

The preferred primary paper is:

> **Denoise the Identity, Preserve the Task: Subject-Aware Diffusion for Privacy-Preserving EEG Representations**

The paper should not call a person or their physiology “noise.” It should define a task-conditional nuisance:

```text
subject-linked information that is unnecessary for the declared task,
creates shortcut/bias across subjects,
and enables identity linkage.
```

The denoising target is therefore a representation, not necessarily the raw waveform.

---


# 2. V31 reconciliation: what changes and what does not

## 2.1 Main decision does not change

V31 does not weaken Route P. It reinforces the pivot away from waveform personalization:

```text
no waveform candidate qualified for sealed confirmation;
more ocular support did not resolve the natural attenuation–retention conflict;
V29 saturated quickly while remaining near identity.
```

The privacy route remains stronger because the support-derived state is highly linkable even when correct-context denoising specificity is mixed.

## 2.2 V31 duration result changes the support interpretation

The repaired protocol showed:

```text
V25/V26 paired fidelity improves with longer support;
natural remaining ratio is non-monotonic;
V29 is essentially saturated by about 5 s.
```

Therefore, Route P should treat support duration as:

```text
privacy attack budget
+
private-signature estimation budget
+
state-storage burden
```

not as an assumption that longer support necessarily improves utility.

Use the exact V31 contract:

```text
0 / 5 / 10 / 30 / 120 s
2 s non-overlapping chronological windows
prefix-only normalization
0 s exact no-support route
no repeated or future samples
```

## 2.3 Stable identity must be separated from acquisition context

The V30/V31 support state may encode:

```text
stable subject physiology
session state
cap placement
reference
impedance
montage and acquisition practice
```

Route P must report separate attacks for:

```text
within-session identity
cross-session identity
cross-task identity
session/acquisition classification
```

A high within-session linkage score is not sufficient evidence of a stable biometric factor.

## 2.4 Support no longer defines the primary private subspace

Primary:

```text
global private subspace learned from training participants
and validated across sessions
```

Query-disjoint support supplies an ephemeral source prototype for negative privacy guidance.

Secondary only:

```text
support-specific subspace correction
```

This prevents Route P from inheriting Route A's unsupported assumption that the matched ocular support identifies the correct local operator.

## 2.5 Lagged and shuffled support have a different meaning in privacy

For ocular transfer, time shuffling can falsify synchronization-based specificity.

For identity privacy, spatial, spectral, covariance and acquisition signatures may survive temporal shuffling. Therefore lag/shuffle are diagnostic tools for locating the private signal, not automatic failure criteria.

---

# 3. Why this route is scientifically stronger now

## 2.1 Route A failed at the point where Route P has headroom

Route A needs a support-derived operator that is useful for removing ocular artifact from the correct query context.

V30 showed:

```text
correct support > mean wrong support for V25/V26
but correct donor rarely ranks first
lagged/shuffled support does not degrade performance
```

That is weak evidence for a correct ocular operator.

However, the same support representation has strong participant linkage:

```text
context + projector top-1 = 0.836
AUROC = 0.962
```

For privacy denoising this is not a nuisance result; it is evidence that the private factor is measurable.

## 2.2 Existing internal bias results reinforce this direction

The current internal evidence suggests:

```text
raw subject identification ≈ 0.998
simple residual x − x_ref ≈ 0.019
some learned transformed representations reintroduce identity leakage ≈ 0.4–0.55
```

This creates a sharper research question than waveform artifact removal:

> Why is identity easy to erase linearly at one representation, yet reappears after nonlinear task-oriented transformations, and can a stochastic diffusion channel suppress it more robustly against adaptive attackers?

---

# 4. Route A versus Route P

| Dimension | Route A — waveform CSPD | Route P — privacy representation denoising |
|---|---|---|
| Primary nuisance | Ocular artifact propagation | Subject identity / acquisition shortcut |
| Output | Cleaned EEG waveform | Sanitized EEG representation |
| Subject-aware role | Estimate the correct ocular operator | Estimate what subject-linked information must be removed |
| Strongest current evidence | V27 absolute attenuation | V30 linkage + internal identity-removal headroom |
| Main current weakness | Mixed correct-donor specificity; no joint natural winner | Must beat strong EEG privacy baselines; “denoising” must be defined precisely |
| Natural evaluation | Needs attenuation plus physiological preservation | Direct task utility plus privacy attacks; cleaner endpoint |
| Closest prior art | SGEYESUB, EEGDfus, D4PM, posterior inverse solvers | ID-RemovalNet, user-wise perturbations, FHVAE, LEACE/INLP/DANN, EEG anonymizers |
| Role of diffusion | Reconstruct waveform overlap | Stochastic replacement of private coordinates and controllable privacy–utility path |
| Data burden | Synchronized EEG/EOG and natural physiological endpoints | Standard task EEG with subject/session labels; no perfect clean waveform needed |
| Fit to original title | Very high | Medium-high if title explicitly says representation denoising |
| Feasibility after V30 | Low–medium | High |
| Recommended status | Backup / geometric template | Primary |

---

# 5. Literature frontier

## 4.1 Direct EEG identity/privacy methods

### ID-RemovalNet — IJCAI 2025

The strongest direct comparator. It decomposes EEG features into task and identity components with decorrelation, adversarial separation and feature enhancement. It reports four datasets across motor-imagery and ERP paradigms.

Implication:

```text
“remove EEG identity while preserving task” is no longer novel by itself.
```

The new method must contribute nonlinear/open-set privacy, stochastic control, cross-backbone transfer, or a better privacy–utility frontier.

### User-wise perturbations — Journal of Neural Engineering 2025

Adds user-specific random, synthetic, error-minimization or error-maximization perturbations to make identity unlearnable while retaining BCI task information.

Implication:

```text
perturbation is a mandatory baseline.
```

### Disguising Personal Identity Information in EEG — CycleGAN, 2020

Transforms EEG into dummy identities while preserving selected features.

Implication:

```text
generative anonymization is prior art, even before diffusion.
```

### Subject-invariant FHVAE — 2022

Separates subject and content latents in speech-evoked EEG; subject classification is high in the subject latent and near chance in the content latent.

Implication:

```text
disentanglement must be compared directly.
```

### Transformer EEG anonymization — 2025 preprint

Generates anonymized EEG for sleep staging and evaluates re-identification versus task utility.

Implication:

```text
sleep staging is a useful external task and autoencoder anonymization is a relevant generative baseline.
```

## 4.2 EEG foundation models and subject bias

FMScope / “The Identity Trap in EEG Foundation Models” reports large subject-dominant variance across LaBraM, CBraMod and REVE, and shows that linear subject-axis erasure may improve label decoding when labels vary within subject.

Implication:

```text
subject identity is not merely a privacy issue;
it is also a representation shortcut and cross-subject bias.
```

The proposed method should include at least one frozen EEG foundation-model representation.

## 4.3 Generic concept-erasure baselines

Mandatory deterministic representation baselines:

```text
DANN / gradient reversal
INLP
LEACE
Gaussian noise / random projection
variational privacy funnel
matched deterministic sanitizer
```

LEACE is particularly important because it gives the minimally distorted representation that defeats every linear classifier for the erased concept.

## 4.4 Diffusion-based privacy and representation learning

### PrivDiff-Net — 2026

Uses latent diffusion with selective attribute suppression and privacy guidance to hide identity while preserving pathology in chest X-rays.

It is the closest cross-modal methodological comparator.

### InfoDiffusion — ICML 2023

Shows that mutual-information regularization can make diffusion latent variables semantically meaningful and disentangled.

It provides a representation-learning basis for the proposed latent diffusion.

### Conditioning can leak identity — ICCV Workshop 2025

Conditioned latent diffusion anonymization can remain re-identifiable under black-box contrastive attacks because the conditioning itself preserves identity-linked structure.

Implication:

```text
diffusion is not privacy by default.
Every conditioning path and support state must be attacked.
```

### Adversarial privacy may be illusory

Work in speech representation privacy shows that reducing closed-set identity classification may not improve open-set verification.

Implication:

```text
closed-set subject accuracy alone is insufficient.
```

---

# 6. Literature gap

Within the reviewed literature, no directly matching peer-reviewed EEG method was found that jointly provides:

```text
query-disjoint support-conditioned private-factor estimation
+
selective diffusion only in a private EEG representation subspace
+
open-set and adaptive identity attacks
+
cross-subject task utility
+
privacy–utility control
+
foundation-model representation evaluation
```

This conjunction is the novelty target.

The claim should be phrased as:

> A support-aware diffusion channel selectively resamples subject-linked representation coordinates while anchoring task-relevant coordinates, producing a controllable privacy–utility frontier for unseen-subject EEG decoding.

---


# 7. Proposed method

## 7.1 Representation and hierarchical private factors

\[
Z=E(X),
\]

where:

```text
X = EEG input
Z = frozen or jointly learned representation
Y = task variable
S = stable subject identity
R = session/acquisition context
C_s = query-disjoint support
```

Target:

\[
\max I(\widetilde Z;Y)
-\lambda_S I(\widetilde Z;S\mid Y)
-\lambda_R I(\widetilde Z;R\mid Y,S).
\]

The two privacy terms are reported separately. This avoids claiming that a session-specific cap or reference signature is equivalent to stable identity.

## 7.2 Global private subspace

Learn from training participants:

```text
P_id:
cross-session subject-linked subspace

P_ctx:
session/acquisition-linked subspace
```

Primary union:

\[
P_{priv}=\operatorname{span}(P_{id},P_{ctx}),
\qquad Q=I-P_{priv}.
\]

Initial implementation:

```text
LEACE / INLP / supervised subject probes
cross-session validation
no large support network
```

## 7.3 Query-disjoint support as negative prototype

Support produces:

\[
c_s=A(C_s),
\]

an ephemeral source identity/context prototype.

It is used to discourage the sanitized output from remaining linkable to the source:

\[
E_{priv}=\operatorname{sim}(g(\widehat Z_0),c_s).
\]

It does not define the primary private subspace in the first version.

A local correction:

\[
P_s=P_{priv}+\Delta P(C_s)
\]

is a secondary ablation only.

## 7.4 Selective forward diffusion

Noise only private coordinates:

\[
Z_t
=
QZ
+
\sqrt{\bar\alpha_t}P_{priv}Z
+
\sqrt{1-\bar\alpha_t}P_{priv}\epsilon.
\]

Task complement remains anchored.

## 7.5 Reverse process

\[
\widehat Z_0
=
D_\theta(Z_t,QZ,c_Y,t),
\]

where \(c_Y\) is a task label, frozen teacher distribution or task prototype.

Task consistency:

\[
E_{task}
=
D_{KL}(h(Z)\|h(\widehat Z_0)).
\]

Output:

\[
\widetilde Z=QZ+P_{priv}\widehat Z_0.
\]

Support state is ephemeral and deleted after sanitization.

## 7.6 Why diffusion may help

1. nonlinear replacement beyond linear erasure;
2. stochastic many-to-one anonymization;
3. continuous privacy-strength control;
4. population-conditioned replacement instead of zeroing;
5. transfer across heterogeneous encoders.

A matched one-step sanitizer remains mandatory, but not a hard retention gate.


# 8. Experimental design

## 8.1 Primary datasets

### Dataset 1 — MI4C / BCI Competition IV-2a

```text
continuity with original TAAS submission
two sessions
subject and task labels
cross-session linkage
```

### Dataset 2 — P300 or ERN

```text
different paradigm
direct comparison with ID-RemovalNet-style methods
within-subject label variation
```

### External setting

```text
sleep staging dataset
or
one EEG foundation-model cohort
```

## 8.2 Encoders

```text
EEGNet
one frozen EEG foundation model: LaBraM or CBraMod
```

Waveform conditions:

```text
RAW
STANDARD / ICA
V27-L0.5
```

## 8.3 Hierarchical privacy attacks

Mandatory:

```text
within-session closed-set subject classification
cross-session open-set verification
cross-task retrieval/linkage
session/acquisition classification
adaptive attacker retrained on sanitized outputs
metric-learning attacker
membership inference
```

Report stable identity and acquisition leakage separately.

## 8.4 Exact support-duration privacy curve

Use:

```text
0 / 5 / 10 / 30 / 120 s
non-overlapping chronological prefixes
prefix-only normalization
no repeated/future samples
```

For each duration report:

```text
identity leakage
session/acquisition leakage
task utility
privacy–utility frontier
support encoding latency
stored-state bytes
```

## 8.5 Utility outcomes

```text
fixed task head
retrained task head
balanced accuracy / AUROC
calibration error
worst-subject accuracy
between-subject performance variance
cross-session generalization
```

## 8.6 Privacy–utility summaries

```text
privacy–utility curve
Pareto frontier
hypervolume / area under privacy–utility curve
latency–privacy–utility curve
state byte size
```

# 9. Required baselines

## Primary table

1. RAW representation
2. Gaussian perturbation
3. User-wise perturbation
4. DANN / GRL
5. INLP
6. LEACE
7. FHVAE-style disentanglement
8. ID-RemovalNet or source-faithful reproduction
9. Matched deterministic privacy adapter
10. Proposed subject-aware privacy diffusion

## Ablations

```text
no support
wrong support
shuffled support
no private-subspace restriction
full-space diffusion
no privacy guidance
no task consistency
one-step versus diffusion
fixed versus adaptive attacker
```

## Waveform preprocessing conditions

```text
RAW waveform
standard / ICA waveform
V27 waveform-cleaned EEG
```

This directly tests whether ordinary denoising reduces or amplifies identity leakage.

---

# 10. How V27 and Route A remain useful

## V27 as backup paper

V27 remains the strongest waveform attenuation route and can still support a separate method-centric manuscript if the privacy route fails.

## V27 as comparator in Route P

It enables a novel experiment:

> Does waveform artifact denoising reduce identity leakage, leave it unchanged, or make stable brainprints easier to recover?

## Route A as the geometric template

Reuse:

```text
preserve complement
resample uncertain/private subspace
query-disjoint support
population prior
wrong/shuffled interventions
```

Remove:

```text
large routing system
rollback architecture
operator zoo
natural physiological-cleaning claim
```

---


# 11. V32P execution plan

## Round 0 — identity-versus-context decomposition

```text
MI4C + P300/ERN
EEGNet + one foundation model
within-session identity attack
cross-session verification
cross-task retrieval
session/acquisition classifier
```

Determine whether the dominant linkable factor is stable identity, acquisition context, or both.

## Round 1 — deterministic privacy frontier

```text
RAW
Gaussian perturbation
DANN
INLP
LEACE
User-wise perturbation
ID-RemovalNet-style
matched deterministic sanitizer
```

Use fixed and adaptive attackers plus fixed/retrained task heads.

## Round 2 — exact support-duration privacy curve

```text
0 / 5 / 10 / 30 / 120 s
V31 exact prefix-only contract
```

Measure privacy leakage, utility, context stability, latency and stored-state size.

## Round 3 — selective diffusion pilot

```text
global private subspace
query-disjoint negative identity prototype
K=1
5 / 10 / 25 steps
one privacy-strength axis
matched one-step sanitizer
2 folds × 2 seeds
```

No support-specific subspace correction in the primary pilot.

## Round 4 — waveform preprocessing interaction

```text
RAW waveform
STANDARD / ICA
V27-L0.5
```

Apply the same encoder, attacker and sanitizer to determine whether waveform denoising hides or exposes stable identity.

## Interpretation

The paper remains viable if diffusion:

```text
A. improves the adaptive privacy–utility frontier;
B. matches deterministic sanitization but improves cross-session/open-set privacy;
C. provides a controllable stochastic family with comparable utility.
```

A route that reduces only within-session classification but not cross-session adaptive verification is not sufficient.

# 12. Recommended final positioning

## Primary paper

```text
Denoise the Identity, Preserve the Task:
Subject-Aware Diffusion for Privacy-Preserving EEG Representations
```

## One-sentence novelty

> We introduce a query-disjoint, subject-aware diffusion channel that selectively resamples identity-linked EEG representation coordinates while anchoring task-relevant coordinates, and evaluate it against adaptive closed- and open-set privacy attacks across conventional and foundation-model EEG representations.

## Claim boundary

Use:

```text
empirical identity obfuscation
privacy–utility control
representation denoising
cross-subject bias reduction
```

Do not use without formal mechanisms:

```text
anonymous
differentially private
irreversible
identity-free
```

---

# 13. Literature shortlist

## Direct EEG privacy

- Wang et al., “ID-RemovalNet: Identity Removal Network for EEG Privacy Protection with Enhancing Decoding Tasks,” IJCAI 2025.
- Chen et al., “User-wise perturbations for user identity protection in EEG-based BCIs,” Journal of Neural Engineering, 2025.
- Liu et al., “Disguising Personal Identity Information in EEG Signals,” 2020.
- Bollens et al., “Learning Subject-Invariant Representations from Speech-Evoked EEG Using Variational Autoencoders,” 2022.
- Fuhrmeister et al., “Bridging Privacy and Utility: Synthesizing anonymized EEG with constraining utility functions,” 2025 preprint.
- Lin et al., “The Identity Trap in EEG Foundation Models: A Diagnostic Audit,” 2026 preprint.

## Concept erasure and privacy representation

- Ganin et al., “Domain-Adversarial Training of Neural Networks,” JMLR 2016.
- Ravfogel et al., “Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection,” ACL 2020.
- Belrose et al., “LEACE: Perfect Linear Concept Erasure in Closed Form,” NeurIPS 2023.
- Razeghi et al., “Deep Variational Privacy Funnel,” 2024.

## Diffusion privacy and representation

- Akhter et al., “Hide Identity, Preserve Pathology: Diffusion-Based Anonymization for Chest X-rays,” PMLR 2026.
- Wang et al., “InfoDiffusion: Representation Learning Using Information Maximizing Diffusion Models,” ICML 2023.
- Lorenz et al., “On the Importance of Conditioning for Privacy-Preserving Data Augmentation,” ICCV Workshops 2025.
- Srivastava et al., “Privacy-Preserving Adversarial Representation Learning in ASR: Reality or Illusion?” 2019.
