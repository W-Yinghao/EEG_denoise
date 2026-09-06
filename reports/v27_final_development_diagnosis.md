# V27 final development diagnosis

{
  "engineering": "valid",
  "energy_effect": "improves_preservation_only",
  "subject_mechanism": "natural_signal_remains_mixed",
  "paired_subject_signal": "paired_signal_preserved",
  "diffusion_positioning": "competitive_with_one_step",
  "natural_tradeoff": "both_failed",
  "next_route": "C. standard clean-signal conditional diffusion bridge",
  "optional_energy_finetune_authorized": false,
  "paired": {
    "energy_det_effect": {
      "mean": -0.006263231097428993,
      "median": -0.0037027707080070638,
      "positive": 6,
      "participants": 15,
      "bootstrap_low": -0.01598209807372951,
      "bootstrap_high": 0.0028239075176174225
    },
    "energy_diff_effect": {
      "mean": -0.0031691235857137456,
      "median": -0.002908012134738902,
      "positive": 7,
      "participants": 15,
      "bootstrap_low": -0.013209941994858311,
      "bootstrap_high": 0.006504469542184965
    },
    "energy_diff_vs_det": {
      "mean": -2.7854465624260748e-05,
      "median": -0.0002662425921602374,
      "positive": 5,
      "participants": 15,
      "bootstrap_low": -0.0004667230781207871,
      "bootstrap_high": 0.0004591606651261204
    },
    "energy_diff_support": {
      "mean": 0.00920586190579723,
      "median": 0.002637398237961164,
      "positive": 10,
      "participants": 15,
      "bootstrap_low": 0.0017800844071479366,
      "bootstrap_high": 0.019195997492069954
    }
  },
  "natural": {
    "energy_diff_effect_artifact": {
      "mean": -0.031430153183122526,
      "median": -0.0624490028373037,
      "positive": 4,
      "participants": 15,
      "bootstrap_low": -0.07193724126946131,
      "bootstrap_high": 0.012813477288891673
    },
    "energy_diff_effect_preservation": {
      "mean": 0.12485319783617592,
      "median": 0.13059202236119127,
      "positive": 15,
      "participants": 15,
      "bootstrap_low": 0.1113988548574137,
      "bootstrap_high": 0.1380297138875582
    },
    "energy_diff_support_artifact": {
      "mean": -0.007747043241597362,
      "median": 0.00810522211143394,
      "positive": 10,
      "participants": 15,
      "bootstrap_low": -0.04979164683731273,
      "bootstrap_high": 0.02793972916557477
    },
    "energy_diff_support_preservation": {
      "mean": -0.0075154451960388535,
      "median": -0.0020945303774884128,
      "positive": 5,
      "participants": 15,
      "bootstrap_low": -0.014690958892710956,
      "bootstrap_high": -0.00165331766282129
    }
  },
  "interpretive_priority": "natural_artifact_preservation_validity_over_strict_diffusion_superiority",
  "development_only": true,
  "query_EOG_inference_reads": 0,
  "query_operator_inference_reads": 0,
  "event_inference_reads": 0,
  "sealed_reads": 0
}

The scientific interpretation prioritizes natural artifact–preservation validity over strict diffusion superiority. Matched DET remains a competitive control and mechanism comparator. Because final-only—not stepwise—was selected, the registered condition for energy-aware fine-tuning was not met.
