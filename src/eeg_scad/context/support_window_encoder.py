from __future__ import annotations

import torch
from torch import Tensor, nn


class _Branch(nn.Module):
    def __init__(self, inputs:int, width:int, blocks:int) -> None:
        super().__init__();layers=[nn.Conv1d(inputs,width,7,padding=3),nn.SiLU()]
        for index in range(blocks):
            dilation=2**index;layers.extend([nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),nn.GroupNorm(8,width),nn.SiLU()])
        self.net=nn.Sequential(*layers);self.attention=nn.Conv1d(width,1,1)
    def forward(self,value:Tensor)->Tensor:
        hidden=self.net(value);weights=torch.softmax(self.attention(hidden),dim=-1);return (hidden*weights).sum(-1)


class SupportWindowEncoder(nn.Module):
    def __init__(self,eeg_channels:int=46,eog_channels:int=4,token_dimension:int=128)->None:
        super().__init__();self.eeg=_Branch(eeg_channels,96,4);self.eog=_Branch(eog_channels,32,3);self.cross=nn.Sequential(nn.Linear(eeg_channels*eog_channels,32),nn.SiLU());self.psd=nn.Sequential(nn.Linear(eeg_channels+eog_channels,16),nn.SiLU());self.out=nn.Sequential(nn.Linear(176,token_dimension),nn.SiLU(),nn.LayerNorm(token_dimension))
    def forward(self,eeg:Tensor,eog:Tensor)->Tensor:
        # B,N,C,T -> B*N,C,T
        b,n,ce,t=eeg.shape;flat_eeg=eeg.reshape(b*n,ce,t);flat_eog=eog.reshape(b*n,eog.shape[2],t)
        covariance=torch.einsum("bct,bdt->bcd",flat_eeg-flat_eeg.mean(-1,keepdim=True),flat_eog-flat_eog.mean(-1,keepdim=True))/max(t-1,1)
        spectrum_eeg=torch.fft.rfft(flat_eeg,dim=-1).abs().square().mean(-1).log1p();spectrum_eog=torch.fft.rfft(flat_eog,dim=-1).abs().square().mean(-1).log1p()
        token=torch.cat((self.eeg(flat_eeg),self.eog(flat_eog),self.cross(covariance.flatten(1)),self.psd(torch.cat((spectrum_eeg,spectrum_eog),1))),1)
        return self.out(token).reshape(b,n,-1)


__all__=["SupportWindowEncoder"]
