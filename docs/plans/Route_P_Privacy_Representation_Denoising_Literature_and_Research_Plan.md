# Route P — Privacy-Aware EEG Representation Denoising
## Literature Frontier, Route Comparison, Method Scope, and Experimental Plan

**Status:** strategic design after V30  
**Primary recommendation:** replace the audit-centric route with a method-centric representation privacy paper  
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

# 2. Why this route is scientifically stronger now

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

# 3. Route A versus Route P

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

# 4. Literature frontier

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

# 5. Literature gap

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

# 6. Proposed method

## 6.1 Representation and variables

\[
Z=E(X),
\]

where:

```text
X = EEG input
Z = frozen or jointly learned representation
Y = task variable
S = subject/private variable
C_s = query-disjoint subject support
```

Target:

\[
\max I(\widetilde Z;Y)-\lambda I(\widetilde Z;S\mid Y).
\]

Conditional privacy is preferred over unconditional privacy because removing all subject information can erase task-relevant physiology when subject and task are correlated.

## 6.2 Private subspace

Estimate a global private subspace from training subjects and an optional low-rank local correction from support:

\[
P_s=P_0+\Delta P(C_s),
\qquad
Q_s=I-P_s.
\]

The first implementation should keep this simple:

```text
P0 from LEACE or a subject probe
support prototype c_s from a frozen identity encoder
no large support network
```

## 6.3 Selective forward diffusion

Noise only the private coordinates:

\[
Z_t
=
Q_sZ
+
\sqrt{\bar\alpha_t}P_sZ
+
\sqrt{1-\bar\alpha_t}P_s\epsilon.
\]

The task complement remains anchored exactly in the forward process.

## 6.4 Reverse process

Predict the sanitized private coordinates under a task-conditioned population prior:

\[
\widehat Z_0
=
D_\theta(Z_t,Q_sZ,c_Y,t).
\]

Privacy guidance discourages linkage to the support identity prototype:

\[
E_{priv}
=
\operatorname{sim}(g(\widehat Z_0),c_s),
\]

and task consistency preserves a frozen task head or teacher distribution:

\[
E_{task}
=
D_{KL}(h(Z)\|h(\widehat Z_0)).
\]

Output:

\[
\widetilde Z=Q_sZ+P_s\widehat Z_0.
\]

Support state is ephemeral and deleted after sanitization.

## 6.5 Why diffusion may help

Diffusion is not justified by lower ideal squared error. Its plausible roles are:

1. nonlinear replacement of private coordinates beyond linear erasure;
2. stochastic many-to-one anonymization that frustrates deterministic inverse mapping;
3. a continuous privacy-strength path through noise level and guidance;
4. sampling from a task-conditioned population representation instead of setting erased coordinates to zero;
5. transfer across heterogeneous encoders.

A matched one-step sanitizer remains mandatory.

---

# 7. Experimental design

## 7.1 Primary datasets

### Dataset 1 — MI4C / BCI Competition IV-2a

Reasons:

```text
continuity with original TAAS submission
two sessions
subject and task labels
shared with ID-RemovalNet
cross-session linkage possible
```

### Dataset 2 — P300 or ERN

Reasons:

```text
different paradigm
direct comparison with ID-RemovalNet
within-subject label variation
```

### External dataset

Choose one:

```text
sleep staging dataset
or
one cohort already used in the EEG foundation-model identity project
```

## 7.2 Encoders

Primary:

```text
EEGNet
one frozen EEG foundation model: LaBraM or CBraMod
```

Optional:

```text
V27 waveform denoiser followed by the same encoder
```

## 7.3 Privacy attacks

Mandatory:

```text
closed-set subject identification
open-set same/different verification
cross-session retrieval/linkage
adaptive attacker retrained on sanitized outputs
kNN / metric-learning attacker
membership inference
```

Secondary:

```text
gender / BCI experience / site or session inference
```

## 7.4 Utility outcomes

```text
fixed task head
retrained task head
balanced accuracy / AUROC
calibration error
worst-subject accuracy
between-subject performance variance
cross-session generalization
```

## 7.5 Privacy–utility summaries

Do not select one arbitrary operating point only.

Report:

```text
privacy–utility curve
Pareto frontier
hypervolume / area under privacy–utility curve
latency–privacy–utility curve
state byte size
```

---

# 8. Required baselines

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

# 9. How V27 and Route A remain useful

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

# 10. V31R execution plan

## Round 0 — benchmark and threat model

```text
MI4C + P300/ERN
EEGNet + one foundation model
RAW / LEACE / DANN / perturbation / ID-RemovalNet-style
closed-set + open-set attackers
fixed + retrained task heads
```

## Round 1 — deterministic ceiling

Determine whether simple methods already achieve:

```text
strong privacy
acceptable task utility
cross-session robustness
```

This is not a gate that closes diffusion. It defines the baseline frontier diffusion must improve or complement.

## Round 2 — diffusion pilot

```text
private-subspace selective diffusion
K=1
5 / 10 / 25 steps
one privacy-strength axis
matched one-step sanitizer
2 folds × 2 seeds
```

## Round 3 — full development

```text
all subjects
3 seeds
all primary baselines
adaptive attacks
V27 preprocessing condition
privacy–utility and latency curves
```

## Interpretation

The paper remains viable under any of these outcomes:

```text
A. diffusion improves the privacy–utility frontier
B. diffusion matches deterministic sanitization but gives stronger open-set privacy or stochastic diversity
C. diffusion provides a controllable family of sanitized representations while deterministic methods supply endpoints
```

Only a model that fails to reduce adaptive identity leakage at any usable utility level should be abandoned.

---

# 11. Recommended final positioning

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

# 12. Literature shortlist

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
