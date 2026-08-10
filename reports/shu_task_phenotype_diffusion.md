# SHU task-phenotype subject-aware restoration exploration

Final routing decision: `TASK_PHENOTYPE_HEADROOM_NO_GO`.

## Data and protocol validity

The datalake contains 25 participants × five sessions and 11,988 real processed MI trials. The available asset is a 256 Hz trial-level LMDB derivative; it was deterministically resampled to the frozen nominal 250 Hz/1000-sample protocol. The source EDF/MAT containers are not directly present, so EDF–MAT file-level correspondence could not be independently re-audited. Trial keys, sessions, labels, shape, finiteness, and 25×5 coverage were audited. Day-4/5 payloads were never deserialized.

PhysioMotion contributed only channel-count/duration/adjacency mask geometry from its already-opened development-20 annotations. No PhysioMotion waveform or sealed participant was read. The mapped library contains 36 masks including the fixed electrode-dropout controls; the scientific J1 used the five frozen nonocular artifact families.

## Day-1 task-phenotype headroom

Class-conditional Day-1 support used equal trial counts across every MATCH, POP, and WRONG operator. POP was an equal-participant outer-fold mean; WRONG donors were the other four unseen participants in the same held-out fold. Day-2/3 query results were aggregated trial/mask → class → day → participant (n=25).

H_P mean/median was -0.014089/-0.023080, with 10/25 positive and one-sided exact p=0.803934. H_W was +0.053166/+0.048368, with 19/25 positive and p=0.001588.

Day effects: {'2': {'H_P': -0.006480116645759122, 'H_W': 0.062339058942872894}, '3': {'H_P': -0.02169750140711085, 'H_W': 0.043993172103970545}}. Class effects: {'0': {'H_P': -0.014644847588937405, 'H_W': 0.05209129067408795}, '1': {'H_P': -0.013532770463932565, 'H_W': 0.0542409403727555}}. Mean ERD-distortion margin versus POP was +0.025091; frozen Day-1 CSP-LDA accuracy margin was +0.001834.

## Routing boundary

The preregistered task-phenotype headroom gate failed. No DET/diffusion model training, scientific GPU screen, extra seeds, or Day-4/5 evaluation was run. Under the terminal instruction, subject-specific EEG denoising experimentation stops here; this is a failure of this fixed task-phenotype representation/probe, not a family-wide mathematical claim about diffusion or personalization.
