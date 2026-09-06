# PhysioMotion hybrid masked-diffusion development screen

## Frozen scope

This was the single preregistered development screen of retrieval-conditioned, artifact-localized masked clean restoration. It used 20 frozen development participants in five outer folds; the 10 sealed participants remained unopened. Manual annotations supplied mask localization, so artifact detection is not a contribution.

The contexts had identical shapes: POP used `(r_P, 0)`, HYBRID-MATCH used `(r_P, r_M-r_P)`, and each HYBRID-WRONG donor used `(r_P, r_W-r_P)`. `r_P` came only from the 16 outer-training participants; MATCH came only from recipient run-01; WRONG donors were the other three unseen recipients in the same outer fold. Retrieval used only the frozen J1R observable selector.

## Technical validity

The technical fold passed: DET RRMSE/correlation 0.00398/0.99999; DIFF-K8 0.07104/0.99748. Mask-exterior change was 0.0e+00; common-noise replay and raw/optimizer/EMA interrupted continuation were exact. DET and DIFF each had 362,114 trainable parameters and all trainable tensors received finite nonzero gradients.

## One-seed participant-first result

Decision: **DEV_ONE_SEED_NO_GO**. Mechanism coverage was 17/20; participants 9, 10, and 11 followed the frozen POP fallback policy. U_P mean/median was -0.00216/-0.00177, with 2/17 positive and one-sided exact p=0.999954. U_W was -0.00170/+0.00044, with 10/17 positive and p=0.744080. The 20-person policy effects were U_P=-0.00149, U_W=-0.00016.

All five artifact-family U_P effects were negative, so the subject and family gates failed. K8 averaging itself was beneficial (E_avg=+0.26920), while DIFF-K1 did not beat the matched DET point estimate (E_D_K1=-0.24169). DeltaSA was -0.00071.

## Absolute restoration and safety

DIFF-K8-HYBRID-MATCH beat masked-zero and temporal interpolation, preserved the mask exterior exactly, and improved the outer-trained natural artifact-detector aggregate. Relative spectral/topographic/covariance safety passed and severe reversal fraction was 0.000. These checks do not rescue failed subject utility.

## Routing boundary

The frozen outcome is `DEV_ONE_SEED_NO_GO`. No additional DIFF seeds, DET8 members, final development models, or sealed evaluation were run. This constrains this hybrid retrieval-context masked restoration instance; it is not a family-wide conclusion about diffusion, masked restoration, or personalization. Per the terminal instruction, no further PhysioMotion structural variants are authorized.
