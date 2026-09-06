# SCAD V22 natural development evaluation

Natural SGEYESUB results are reported as an attenuation–preservation trade-off. Query EOG and query operators were absent from inference and opened only after the 15-file output freeze. Natural data have no paired clean counterfactual, so attenuation alone is not interpreted as successful denoising.

SCAD-MATCH had held-out EOG remaining ratio 0.9061, preservation 0.8424, PSD distortion 0.2099, and covariance distortion 0.1185. DET-MATCH remaining ratio was 0.8396. Relative to SCAD-POP, MATCH slightly improved attenuation (+0.00333) but reduced preservation (-0.00212) and did not improve PSD/covariance on average. The resulting label is `preservation_concern`.
