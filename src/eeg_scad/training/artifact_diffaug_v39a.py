"""Matched generator/augmentation experiment for V39A."""
from __future__ import annotations
import json,random
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import torch
from scipy import signal
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from eeg_scad.data.artifact_diffaug_v39a import context_bank,sample_targets,sha256
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.models.artifact_generators_v39a import ArtifactCritic,ArtifactGenerator,ConditionalArtifactDiffusion,ConditionalArtifactGaussian,SpatialArtifactCodec,SupportDenoiserV39


GENS=("Empirical-Resample","Conditional-Gaussian","Conditional-WGAN-GP","Conditional-Artifact-Diffusion")
ARMS=("No-Augmentation","Real-Artifact-Augmentation","Gaussian-Augmentation","WGAN-Augmentation","Diffusion-Augmentation")


def seed_all(seed:int):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
def _state(m):return {k:v.detach().cpu() for k,v in m.state_dict().items()}
def _loader(*arrays,batch=64,shuffle=True,seed=0):
    ds=TensorDataset(*[torch.from_numpy(np.ascontiguousarray(x)).float() for x in arrays]);return DataLoader(ds,batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed))


def severity_contract(train:dict[str,Any]):
    values=np.asarray([m["severity"] for m in train["meta"]]);cuts=np.quantile(values,[1/3,2/3]);return cuts,float(values.mean()),float(max(values.std(),1e-6))
def conditions(bank:dict[str,Any],cuts:np.ndarray,mean:float,std:float):
    stream=np.asarray([[1,0] if m["stream"]=="paired" else [0,1] for m in bank["meta"]],np.float32);sev=np.asarray([(m["severity"]-mean)/std for m in bank["meta"]],np.float32)[:,None];return np.concatenate((bank["context"],stream,sev),1).astype(np.float32),np.digitize([m["severity"] for m in bank["meta"]],cuts)


class EmpiricalGenerator:
    def __init__(self,artifact,context,bins):self.artifact=artifact;self.context=context;self.bins=bins
    def sample(self,condition,bins,seed):
        rng=np.random.default_rng(seed);out=[]
        for c,b in zip(condition,bins):
            pool=np.flatnonzero(self.bins==b);pool=pool if len(pool) else np.arange(len(self.artifact));dist=np.linalg.norm(self.context[pool,:128]-c[:128],axis=1);near=pool[np.argsort(dist)[:min(32,len(pool))]];out.append(self.artifact[int(rng.choice(near))])
        return np.asarray(out,np.float32)


def train_wgan(latent,condition,device,seed,epochs=30):
    seed_all(seed);g=ArtifactGenerator().to(device);d=ArtifactCritic().to(device);go=torch.optim.Adam(g.parameters(),1e-4,betas=(0,.9));do=torch.optim.Adam(d.parameters(),1e-4,betas=(0,.9));curve=[]
    for epoch in range(epochs):
        for real,c in _loader(latent,condition,batch=64,seed=seed+epoch):
            real,c=real.to(device),c.to(device)
            for _ in range(3):
                fake=g(torch.randn(len(real),16,real.shape[-1],device=device),c).detach();alpha=torch.rand(len(real),1,1,device=device);mix=(alpha*real+(1-alpha)*fake).requires_grad_(True);score=d(mix,c);grad=torch.autograd.grad(score.sum(),mix,create_graph=True)[0];gp=((grad.flatten(1).norm(2,1)-1)**2).mean();loss_d=d(fake,c).mean()-d(real,c).mean()+10*gp;do.zero_grad();loss_d.backward();do.step()
            fake=g(torch.randn(len(real),16,real.shape[-1],device=device),c);loss_g=-d(fake,c).mean();go.zero_grad();loss_g.backward();go.step()
        curve.append({"epoch":epoch+1,"generator_loss":float(loss_g.detach()),"critic_loss":float(loss_d.detach())})
    return g,curve


def train_diffusion(latent,condition,device,seed,epochs=30,condition_strength=1.0):
    seed_all(seed);model=ConditionalArtifactDiffusion().to(device);opt=torch.optim.AdamW(model.parameters(),2e-4,weight_decay=1e-4);curve=[]
    for epoch in range(epochs):
        losses=[]
        for x,c in _loader(latent,condition,batch=64,seed=seed+epoch):
            x,c=x.to(device),c.to(device);t=torch.randint(0,len(model.alpha_bar),(len(x),),device=device);pred=model(model.q_sample(x,t,torch.randn_like(x)),condition_strength*c,t);loss=torch.nn.functional.mse_loss(pred,x);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        curve.append({"epoch":epoch+1,"x0_loss":float(np.mean(losses))})
    return model,curve


@torch.no_grad()
def generator_sample(name,count,condition,bins,codec,empirical,gaussian,wgan,diffusion,device,seed,condition_strength=1.0):
    n=len(condition);values=[]
    for draw in range(count):
        if name=="Empirical-Resample":artifact=empirical.sample(condition,bins,seed+draw)
        elif name=="Conditional-Gaussian":artifact=codec.decode(gaussian.sample(condition,bins,seed+draw))
        elif name=="Conditional-WGAN-GP":
            c=torch.from_numpy(condition).to(device);z=wgan(torch.randn(n,16,256,device=device,generator=torch.Generator(device=device).manual_seed(seed+draw)),c).cpu().numpy();artifact=codec.decode(z)
        elif name=="Conditional-Artifact-Diffusion":
            c=torch.from_numpy(condition).to(device);noise=torch.randn(n,8,256,device=device,generator=torch.Generator(device=device).manual_seed(seed+draw));artifact=codec.decode(diffusion.sample(condition_strength*c,noise,10).cpu().numpy())
        else:raise ValueError(name)
        values.append(artifact)
    return np.stack(values)


def _fidelity(method,generated,target,participant,seed):
    # Participant/context-first: mean per-target discrepancy, then participant aggregate downstream.
    mean=generated.mean(0);flat_target=target.reshape(len(target),-1);flat_mean=mean.reshape(len(mean),-1);temporal=[];spectral=[];topo=[]
    for a,b in zip(mean,target):
        temporal.append(float(np.mean(np.abs(np.correlate(a.mean(0),a.mean(0),"same")-np.correlate(b.mean(0),b.mean(0),"same")))/max(np.mean(np.abs(np.correlate(b.mean(0),b.mean(0),"same"))),1e-8)))
        f,pa=signal.welch(a,fs=100,nperseg=128,axis=-1);_,pb=signal.welch(b,fs=100,nperseg=128,axis=-1);spectral.append(float(np.mean(np.abs(np.log(pa+1e-8)-np.log(pb+1e-8)))))
        topo.append(float(np.linalg.norm(np.cov(a)-np.cov(b))/max(np.linalg.norm(np.cov(b)),1e-8)))
    rng=np.random.default_rng(seed);idx=rng.choice(len(flat_target),min(128,len(flat_target)),replace=False);x=flat_target[idx];y=flat_mean[idx];scale=np.std(x,axis=0);keep=scale>1e-6;x=x[:,keep]/scale[keep];y=y[:,keep]/scale[keep];dist=pairwise_distances(x,y);xx=pairwise_distances(x,x);yy=pairwise_distances(y,y);energy=float(2*dist.mean()-xx.mean()-yy.mean());sq=pairwise_distances(np.concatenate((x,y)),squared=True);bw=np.median(sq[sq>0]);k=np.exp(-sq/max(2*bw,1e-8));n=len(x);mmd=float(k[:n,:n].mean()+k[n:,n:].mean()-2*k[:n,n:].mean());within=float(np.var(generated,axis=0,ddof=1).mean());nearest=pairwise_distances(generated.reshape(-1,generated.shape[2]*generated.shape[3])[:min(256,generated.shape[0]*generated.shape[1])],flat_target).min(1)
    return {"method":method,"temporal_autocorrelation_distance":float(np.mean(temporal)),"welch_band_power_error":float(np.mean(spectral)),"channel_covariance_topography_error":float(np.mean(topo)),"amplitude_error":float(np.mean(np.abs(np.sqrt(np.mean(mean*mean,axis=(1,2)))-np.sqrt(np.mean(target*target,axis=(1,2)))))),"energy_distance":energy,"mmd":mmd,"within_context_diversity":within,"between_context_separation":float(np.var(mean,axis=0).mean()),"nearest_training_artifact_distance":float(nearest.mean()),"exact_copy_rate":float(np.mean(nearest<1e-7)),"near_copy_rate":float(np.mean(nearest<.1*np.median(pairwise_distances(flat_target[:min(128,len(flat_target))]))))}


def train_denoiser(arm,train,condition,bins,generators,device,seed,epochs=20):
    seed_all(seed);model=SupportDenoiserV39().to(device);opt=torch.optim.AdamW(model.parameters(),2e-4,weight_decay=1e-4);n=512;idx=np.random.default_rng(seed).choice(len(train["clean"]),n,replace=len(train["clean"])<n);clean=train["clean"][idx];context=train["context"][idx];c=condition[idx];b=bins[idx]
    if arm=="No-Augmentation":art=np.zeros((8,n,46,256),np.float32)
    elif arm=="Real-Artifact-Augmentation":art=np.stack([train["artifact"][np.random.default_rng(seed+d).choice(len(train["artifact"]),n)] for d in range(8)])
    else:art=generator_sample({"Gaussian-Augmentation":"Conditional-Gaussian","WGAN-Augmentation":"Conditional-WGAN-GP","Diffusion-Augmentation":"Conditional-Artifact-Diffusion"}[arm],8,c,b,*generators,device,seed+1000)
    x=np.tile(clean,(8,1,1,1)).reshape(8*n,46,256);ctx=np.tile(context,(8,1)).reshape(8*n,128);y=x+art.reshape(8*n,46,256);curve=[]
    for epoch in range(epochs):
        losses=[]
        for yy,cc,xx in _loader(y,ctx,x,batch=64,seed=seed+epoch):
            yy,cc,xx=yy.to(device),cc.to(device),xx.to(device);pred=model(yy,cc);loss=torch.nn.functional.smooth_l1_loss(pred,xx);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        curve.append({"epoch":epoch+1,"loss":float(np.mean(losses))})
    return model,curve,{"arm":arm,"clean_carriers":n,"artifacts_per_carrier":8,"training_rows":8*n,"epochs":epochs,"updates":epochs*int(np.ceil(8*n/64)),"architecture":"SupportDenoiserV39_width64"}


@torch.no_grad()
def evaluate_denoiser(model,bank,device,arm,fold,seed,natural=False):
    pred=[]
    for y,c in _loader(bank["y"],bank["context"],batch=64,shuffle=False):pred.append(model(y.to(device),c.to(device)).cpu().numpy())
    pred=np.concatenate(pred);rows=[]
    for i,(clean,y,a,meta) in enumerate(zip(bank["clean"],bank["y"],bank["artifact"],bank["meta"])):
        estimate=y-pred[i]
        if not natural:rows.append({"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":arm,"severity":meta["severity"],**paired_metrics(clean,y,a,estimate)})
        else:
            energy=np.sqrt(np.mean(bank["latent"][i]**2,axis=0));low=energy<=np.quantile(energy,.3);high=energy>=np.quantile(energy,.7);remaining=float(np.linalg.norm((a-estimate)[:,high])/max(np.linalg.norm(a[:,high]),1e-8));retention=1-float(np.linalg.norm(estimate[:,low])/max(np.linalg.norm(y[:,low]),1e-8));f,p0=signal.welch(y[:,low],fs=100,nperseg=min(128,low.sum()),axis=-1);_,p1=signal.welch(pred[i,:,low],fs=100,nperseg=min(128,low.sum()),axis=-1);keep=(f>=1)&(f<=15);rows.append({"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":arm,"heldout_eog_remaining_ratio":remaining,"artifact_attenuation_db":float(-20*np.log10(max(remaining,1e-8))),"low_eog_observation_retention":retention,"psd_distortion":float(np.mean(np.abs(np.log(p0[:,keep]+1e-8)-np.log(p1[:,keep]+1e-8)))),"covariance_distortion":float(np.linalg.norm(np.cov(pred[i,:,low])-np.cov(y[:,low]))/max(np.linalg.norm(np.cov(y[:,low])),1e-8)),"output_input_rms":float(np.sqrt(np.mean(pred[i]**2))/max(np.sqrt(np.mean(y**2)),1e-8)),"query_eog_inference_reads":0})
    return rows


def run_fold(result_root:Path,data:Mapping[str,Any],fold_cfg:Mapping[str,Any],seed:int,device:torch.device):
    fold=int(fold_cfg["fold"]);runtime=result_root/"runtime"/f"fold_{fold}_seed_{seed}";runtime.mkdir(parents=True,exist_ok=True);contexts,support_rows,support_binding=context_bank(data,fold_cfg,20260825,device)
    train=sample_targets(data,fold_cfg,"train",seed+1,768,256,contexts);val=sample_targets(data,fold_cfg,"validation",seed+2,192,64,contexts);test=sample_targets(data,fold_cfg,"test",seed+3,384,0,contexts);natural=sample_targets(data,fold_cfg,"test",seed+4,0,192,contexts)
    cuts,mean,std=severity_contract(train);train_c,train_b=conditions(train,cuts,mean,std);val_c,val_b=conditions(val,cuts,mean,std);codec=SpatialArtifactCodec.fit(train["artifact"]);latent=codec.encode(train["artifact"]);emp=EmpiricalGenerator(train["artifact"],train_c,train_b);gauss=ConditionalArtifactGaussian.fit(latent,train_c,train_b)
    wgan,wcurve=train_wgan(latent,train_c,device,seed+10);diff,dcurve=train_diffusion(latent,train_c,device,seed+20);torch.save({"model":_state(wgan),"curve":wcurve},runtime/"wgan.pt");torch.save({"model":_state(diff),"curve":dcurve},runtime/"diffusion.pt")
    generators=(codec,emp,gauss,wgan,diff);fidelity=[];diversity=[];exposure=[]
    for name in GENS:
        generated=generator_sample(name,8,val_c,val_b,*generators,device,seed+100);row=_fidelity(name,generated,val["artifact"],np.asarray([m["participant"] for m in val["meta"]]),seed);fidelity.append({"fold":fold,"seed":seed,**row});diversity.append({k:v for k,v in fidelity[-1].items() if k in ("fold","seed","method","within_context_diversity","between_context_separation")});exposure.append({k:v for k,v in fidelity[-1].items() if k in ("fold","seed","method","nearest_training_artifact_distance","exact_copy_rate","near_copy_rate")})
    denoiser_rows=[];paired=[];natural_rows=[];bindings=[support_binding,{"fold":fold,"seed":seed,"model":"Conditional-WGAN-GP","path":str(runtime/"wgan.pt"),"sha256":sha256(runtime/"wgan.pt")},{"fold":fold,"seed":seed,"model":"Conditional-Artifact-Diffusion","path":str(runtime/"diffusion.pt"),"sha256":sha256(runtime/"diffusion.pt")}]
    for index,arm in enumerate(ARMS):
        model,curve,manifest=train_denoiser(arm,train,train_c,train_b,generators,device,seed+1000+index);path=runtime/f"denoiser_{index}.pt";torch.save({"model":_state(model),"curve":curve,"manifest":manifest},path);bindings.append({"fold":fold,"seed":seed,"model":arm,"path":str(path),"sha256":sha256(path)});denoiser_rows.append({"fold":fold,"seed":seed,**manifest});paired+=evaluate_denoiser(model,test,device,arm,fold,seed);natural_rows+=evaluate_denoiser(model,natural,device,arm,fold,seed,True)
    target_rows=[]
    for split_name,bank in (("train",train),("validation",val),("test_paired",test),("test_natural",natural)):
        for i,m in enumerate(bank["meta"]):target_rows.append({"fold":fold,"seed":seed,"split":split_name,"row":i,**m})
    payload={"fold":fold,"seed":seed,"support_manifest":support_rows,"artifact_target_manifest":target_rows,"checkpoint_binding":bindings,"generator_fidelity":fidelity,"generator_diversity":diversity,"training_exposure":exposure,"denoiser_training_manifest":denoiser_rows,"paired":paired,"natural":natural_rows,"repair_used":False,"sealed_reads":0,"query_eog_inference_reads":0}
    (runtime/"result.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");return payload


__all__=["ARMS","GENS","run_fold"]
