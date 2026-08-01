# Klados v4 pairing decision

Klados v4 contains 54 aligned clean EEG, contaminated EEG, VEOG and HEOG
records.  Slurm job `919373` verified the mixture relation to relative residual
below `1.1e-7`, but the participant-dependent transfer coefficients do not
recover a unique 54-to-27 participant pairing: only six mutual-nearest pairs
were found and their distances were not separated from non-pairs.

The second and final targeted check used the official Data in Brief article and
the v4 release description.  They state 27 participants and 54 records, but do
not publish a record-to-participant table.  File order is therefore not used as
participant identity.

Decision: the Klados participant-held-out route is blocked.  `sim45` is retained
as a paired, held-out **source-record** mechanism fold because it alone supports
30 s calibration, a 1 s guard, and about 11 s of non-overlapping query.  Its
population clean prior is trained on the independent EEGdenoiseNet clean-EEG
collection, not on Klados.  Any comparison using other Klados records is marked
source-level and cannot establish participant-level specificity.  Eye-BCI is
the participant-held-out natural-EEG route.
