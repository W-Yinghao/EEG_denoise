# V26 natural development

Inference used query EEG, frozen V25 anchors, and query-disjoint support only. Fifteen output bundles were digested and frozen before the evaluator opened development query EOG/reference information. Inference reads were: query EOG 0, query operator 0, events 0, sealed 0.

CalibSDEdit-MATCH produced remaining ratio 1.00771, artifact attenuation 1.36311 dB, preservation 0.80336, PSD distortion 0.38653, and covariance distortion 0.23111. Relative to PopSDEdit, artifact utility was +0.009705 (12/15), while preservation utility was -0.007522 (5/15; bootstrap interval entirely below zero).

Relative to matched one-step, SDEdit was worse on artifact utility (-0.019899, 1/15 positive) and preservation (-0.008492, 1/15 positive). The resulting classification is `preservation_concern`.

Natural artifact–preservation validity has higher interpretive priority than a strict paired `DIFF > DET` ordering. The evidence supports testing a lightweight energy refinement focused on overlap/preservation; it does not support confirmation, a safety claim, or a large routing system.
