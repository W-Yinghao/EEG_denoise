# TAAS-26-0171 项目总纲更新附录
## v4.4 — V41R 后的 Clean-room Calib-SADDPM-Cond 路线

> 服务器应将本节置于仓库现有 v4.3 ledger 顶部，保留 v4.3 及更早 provenance 不变。

## 1. V41R 最终判决

```text
final positioning:
D — base model not established

closed implementation:
official-semantics channel-wise epsilon-prediction EEGDfus
+
explicit artifact-transfer FiLM
```

V41R 的 population model：

```text
contaminated temporal RRMSE:
0.550124

POP temporal RRMSE:
0.693946

POP SNR improvement:
-8.097552 dB
```

因此 `MATCH−POP` 不能作为 artifact-transfer conditioning 的有效性检验。

```text
MATCH−POP:
-0.001974
95% CI [-0.009502, 0.005585]

MATCH−WRONG:
+0.025006
12/15 positive

MATCH−SHUFFLED:
-0.000678
```

V41R 仅关闭该 channel-wise backbone；不关闭 subject/context conditioning。

## 2. 失败原因的当前诊断

### 2.1 训练预算不足以匹配 official anchor

```text
official-native EEGDfus EOG:
208,000 optimizer updates

V41R:
12,000 updates
```

V41R 的 checkpoint 又按 epsilon L1 选择，而不是最终 clean-waveform fidelity。

### 2.2 Prediction target 与 paired denoising 不匹配

V41R沿用：

```text
epsilon prediction
```

原 SADDPM-Cond 方法规范已经指出：

```text
x0 prediction was the main lever for paired artifact removal
```

V42R因此不再把epsilon-prediction official semantics强制迁移到subject-specific paired setting。

### 2.3 Channel-wise model丢失多通道artifact topography

V41R将46通道拆成独立single-channel samples。Transfer row虽作为条件输入，但模型无法联合利用：

```text
frontal topography
cross-channel covariance
shared ocular waveform
relative channel propagation
```

原SADDPM-Cond的joint multichannel setting正是为了利用这类空间结构。

### 2.4 Full generation caused attenuation/shrinkage

V41R POP output/input RMS q99约为`0.779`，说明输出存在系统性幅度收缩。下一实现使用：

```text
x0 prediction
+
observation-centered residual parameterization
```

使模型初始行为接近identity，而不是从一开始重造整段波形。

## 3. 下一路线

```text
V42R — Clean-room Calib-SADDPM-Cond
```

目标是一个窄的related-work增量：

> Does query-disjoint support-estimated ocular transfer improve a valid joint multichannel conditional
> diffusion denoiser over its identical population-conditioned route?

V42R不是V41R调参repair，而是根据原稿已声明的方法差异，预先冻结：

```text
joint multichannel conditional diffusion
x0 prediction
observation-centered output
explicit transfer conditioning
joint training of population and transfer residual branches
```

## 4. 方法所有权

Primary proposed method：

```text
Calib-SADDPM-Cond
```

External official positioning：

```text
EEGDfus official reproduction
EEGdenoiseNet official CNNs
DeepSeparator official source
D4PM related work only
```

Primary support comparison：

```text
MATCH vs POP
```

Wrong/shuffled/oracle是机制控制，不是diffusion生死门。

## 5. Subject-aware边界

```text
subject-aware:
support-estimated corruption transfer

not identity-conditioned:
no subject ID
no persistent learned identity embedding
no target-gradient adaptation
```

## 6. 项目纪律

V42R只允许：

```text
one clean-room joint architecture
one x0 objective
one transfer-conditioning mechanism
one fixed training budget
one optional engineering repair
```

禁止：

```text
return to generic support embedding
new diffusion task
fiber/privacy route
operator zoo
energy/routing system
post-test architecture search
```

若V42R population model仍未优于contaminated input，结论只能是：

```text
paired diffusion base not reproduced
```

若population model有效，则以MATCH−POP判断transfer conditioning。
