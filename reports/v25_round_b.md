# V25 Round B

Five development folds and three fixed seeds were evaluated participant-first. DeepSets was frozen from Round A using validation joint error (24.0767 versus 24.2959 for Set Transformer). Round B trained 15 DET cells and 15 diffusion cells; no seed or participant was dropped.

```json
{
  "DET_MATCH_POP": {
    "mean": 0.007147799208957136,
    "median": 0.0029593362264629386,
    "positive": 10,
    "participants": 15,
    "bootstrap_low": 0.001177543135070455,
    "bootstrap_high": 0.013916224231739632
  },
  "DET_MATCH_WRONG": {
    "mean": 0.016730774263641664,
    "median": 0.007147395995339922,
    "positive": 14,
    "participants": 15,
    "bootstrap_low": 0.007803781728365395,
    "bootstrap_high": 0.026910183436618818
  },
  "DIFF_MATCH_POP": {
    "mean": -0.04988596318179542,
    "median": -0.045047443736295545,
    "positive": 0,
    "participants": 15,
    "bootstrap_low": -0.06119710254395704,
    "bootstrap_high": -0.03896836149839303
  },
  "DIFF_MATCH_WRONG": {
    "mean": 0.026051251203328348,
    "median": 0.02794918588790668,
    "positive": 12,
    "participants": 15,
    "bootstrap_low": 0.012658797319522564,
    "bootstrap_high": 0.03922494457562572
  },
  "DIFF_DET": {
    "mean": -0.05703376239075254,
    "median": -0.05826845805648079,
    "positive": 0,
    "participants": 15,
    "bootstrap_low": -0.06431298844861341,
    "bootstrap_high": -0.050080189028283
  }
}
```

The deterministic MATCH increment is small but consistent in participant-bootstrap summaries. Diffusion is uniformly worse than DET: its participant utility is negative for all 15 participants and in every preregistered severity stratum (mild −0.06793, medium −0.05238, severe −0.03977).
