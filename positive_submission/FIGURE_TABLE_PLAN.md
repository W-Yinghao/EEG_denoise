# Figure and Table Plan

The figures are deliberately represented by design placeholders in the manuscript. Quantitative plots and EEG examples must be generated from stored evaluator outputs; AI image generation is restricted to a non-data-bearing method sketch.

## Figure 1: Graphical abstract

A four-stage horizontal summary: separate EEG--EOG calibration, estimation and shrinkage of the EOG-to-EEG propagation matrix, matched EOG guidance of the shared conditional diffusion model, and restored EEG with predictive intervals. Blue denotes population-shared components, orange denotes calibration-derived quantities, green denotes matched guidance, and gray denotes population and zero-anchor comparison modes.

## Figure 2: Method overview

Four panels show the non-overlapping calibration/evaluation timeline, ridge estimation after robust EOG scaling, empirical Bayes shrinkage, construction of the ocular-artifact guide, and the DDIM restoration path. Runtime and calibration-only information flows use solid and dashed arrows, respectively. The figure must show that EOG is recorded during evaluation, that the clean EEG target is never an input, and that the zero-anchor comparator sets only the waveform guide to zero while retaining the compact calibration state.

## Figure 3: Paired denoising and participant effects

Panel A uses evaluator outputs selected without looking at method ranking: the lowest-identifier development participant with complete outputs, the earliest eligible ocular event, and fixed scalp channels. It shows EOG, contaminated EEG, paired reference, linear regression, zero-anchor diffusion, and matched-calibration diffusion at identical scale. The final caption records participant, event, and channel identifiers. Panels B and C use every participant in the development and held-out cohorts. Panel D shows cohort-specific effect intervals without pooling the cohorts.

## Figure 4: Natural recordings and calibration controls

The main panel plots low-EOG retention against EOG attenuation for matched, population, zero-anchor, mismatched, and shuffled conditions. Further panels show participant-level effects for attenuation, retention, and EOG--EEG coherence. Power spectra and scalp topographies are included only if their evaluator arrays are available; they follow the same lowest-identifier/earliest-event selection rule and identical scales across variants.

## Figure 5: Calibration representation and predictive intervals

The left half shows matched-calibration gain over training-pool size and the matched/zero-embedding/mismatched control at the largest pool. It is labeled as a controlled injected deterministic study with 15 evaluation participants and a plausible-range analysis of 14. The right half shows nominal versus empirical interval coverage, CRPS, and risk--coverage area on the 15-participant development data.

## Tables

Table 1 defines cohorts, inputs, and the purpose of each experimental panel. Table 2 lists comparison variants and their information inputs. Table 3 reports development and held-out paired restoration separately. Table 4 reports natural attenuation and retention. Table 5 presents the controlled calibration-representation study and the development predictive-interval analysis in separate blocks, with their distinct sample sizes and metrics.
