from __future__ import annotations
from torch import Tensor,nn
from .eegdus_backbone import ArtifactBackbone


class DeterministicArtifactEstimator(nn.Module):
    visible_fields=("contaminated_EEG","support_operator_context")
    forbidden_fields=("query_EOG","query_operator","query_event","clean_target_at_inference")
    def __init__(self,channels:int=46,base_channels:int=32,context_input_dim:int=189,context_hidden_dim:int=256,context_dim:int=128)->None:
        super().__init__();self.channels=channels;self.backbone=ArtifactBackbone(channels,channels,base_channels,context_input_dim,context_hidden_dim,context_dim,time_conditioned=False)
    def forward(self,y:Tensor,context:Tensor)->Tensor:return self.backbone(y,context,None)
    def clean(self,y:Tensor,context:Tensor)->Tensor:return y-self(y,context)

