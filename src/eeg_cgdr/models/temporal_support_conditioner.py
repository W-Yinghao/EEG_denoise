"""Compact raw-temporal support conditioner for MobileBCI development.

Unlike v3 moment summaries, this encoder retains ordered waveform patches and
modality identity.  The query network reads the tokens through cross-attention
at two temporal resolutions.  Participant identifiers are not represented.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalSupportEncoder(nn.Module):
    def __init__(self, eeg_channels: int = 46, imu_channels: int = 27, eog_channels: int = 4,
                 width: int = 96, patch_stride: int = 32) -> None:
        super().__init__()
        self.counts=(eeg_channels,imu_channels,eog_channels); self.width=width
        self.projections=nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels,width,kernel_size=patch_stride,stride=patch_stride),nn.SiLU(),
                          nn.Conv1d(width,width,3,padding=1),nn.SiLU())
            for channels in self.counts
        ])
        self.modality=nn.Parameter(torch.randn(3,1,width)*0.02)
        self.position=nn.Parameter(torch.randn(1,512,width)*0.01)
        layer=nn.TransformerEncoderLayer(width,4,4*width,dropout=0.0,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.null_token=nn.Parameter(torch.zeros(1,1,width))

    def forward(self,support_eeg:Tensor,support_imu:Tensor,support_eog:Tensor,
                modality_present:Tensor,context_present:Tensor)->Tensor:
        values=(support_eeg,support_imu,support_eog); tokens=[]
        for index,(value,projection) in enumerate(zip(values,self.projections)):
            if value.ndim!=3 or value.shape[1]!=self.counts[index]: raise ValueError("support modality shape mismatch")
            token=projection(value).transpose(1,2)+self.modality[index]
            token=token*modality_present[:,index,None,None].to(token.dtype); tokens.append(token)
        result=torch.cat(tokens,dim=1)
        if result.shape[1]>self.position.shape[1]:
            # Deterministic temporal subsampling retains order while bounding memory.
            indices=torch.linspace(0,result.shape[1]-1,self.position.shape[1],device=result.device).round().long()
            result=result[:,indices]
        result=result+self.position[:,:result.shape[1]]
        result=self.encoder(result)
        present=context_present[:,None,None].to(result.dtype)
        return result*present+self.null_token.expand(result.shape[0],result.shape[1],-1)*(1-present)


class CrossAttentionBlock(nn.Module):
    def __init__(self,width:int)->None:
        super().__init__(); self.norm=nn.GroupNorm(8,width); self.conv=nn.Conv1d(width,width,3,padding=1)
        self.query_norm=nn.LayerNorm(width); self.attention=nn.MultiheadAttention(width,4,batch_first=True)
        self.feed=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,4*width),nn.SiLU(),nn.Linear(4*width,width))
    def forward(self,value:Tensor,tokens:Tensor)->Tensor:
        hidden=self.conv(torch.nn.functional.silu(self.norm(value))); query=self.query_norm(hidden.transpose(1,2))
        attended,_=self.attention(query,tokens,tokens,need_weights=False); query=query+attended; query=query+self.feed(query)
        return hidden+query.transpose(1,2)


class TemporalSupportCleaner(nn.Module):
    """One-step EEG-space correction estimator with ordered support tokens."""
    forbidden_inputs=("query_EOG","query_IMU","query_event_label","participant_ID","query_outcome")
    def __init__(self,eeg_channels:int=46,imu_channels:int=27,eog_channels:int=4,width:int=96)->None:
        super().__init__(); self.support_encoder=TemporalSupportEncoder(eeg_channels,imu_channels,eog_channels,width)
        self.input=nn.Conv1d(eeg_channels,width,7,padding=3)
        self.block1=CrossAttentionBlock(width); self.down=nn.Conv1d(width,width,4,stride=2,padding=1)
        self.block2=CrossAttentionBlock(width); self.up=nn.ConvTranspose1d(width,width,4,stride=2,padding=1)
        self.output=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv1d(width,eeg_channels,3,padding=1))
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)
    def forward(self,query_eeg:Tensor,*,support_eeg:Tensor,support_imu:Tensor,support_eog:Tensor,
                modality_present:Tensor,context_present:Tensor)->Tensor:
        tokens=self.support_encoder(support_eeg,support_imu,support_eog,modality_present,context_present)
        first=self.block1(self.input(query_eeg),tokens); second=self.block2(self.down(first),tokens)
        up=self.up(second)
        if up.shape[-1]!=first.shape[-1]: up=torch.nn.functional.interpolate(up,size=first.shape[-1],mode="linear",align_corners=False)
        return self.output(first+up)


class PopulationCleaner(nn.Module):
    def __init__(self,eeg_channels:int=46,width:int=96)->None:
        super().__init__(); self.network=nn.Sequential(nn.Conv1d(eeg_channels,width,7,padding=3),nn.SiLU(),
            nn.Conv1d(width,width,5,padding=2),nn.SiLU(),nn.Conv1d(width,width,5,padding=2),nn.SiLU(),
            nn.Conv1d(width,eeg_channels,3,padding=1)); nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
    def forward(self,query_eeg:Tensor)->Tensor: return self.network(query_eeg)

