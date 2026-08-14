# V42R native single-channel x0 sanity

The clean-room x0 code path completed 10,000 updates on the admitted source-record-disjoint
EEGdenoiseNet EOG materialization. Its best full 50-step-DDIM validation temporal RRMSE was 0.355391.
On the complete registered test materialization it achieved temporal RRMSE 0.354282, spectral RRMSE
0.105862, and correlation 0.930160, versus noisy-input values 1.067589, 0.264224, and 0.696453.

The separately frozen official EEGDfus native reproduction remains stronger in temporal RRMSE
(0.296527) and correlation (0.953041), while using 208,000 optimizer updates and its official
epsilon-prediction code. These protocols are positioning evidence, not interchangeable training runs.

Classification: `valid_cleanroom_x0_sanity`. No multichannel hyperparameter was selected from the
native test result, and no engineering repair was used.
