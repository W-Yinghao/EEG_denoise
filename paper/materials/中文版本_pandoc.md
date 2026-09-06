基于扩散模型的脑电信号去噪方法

**摘要：**脑电信号（EEG）具有高时间分辨率和非侵入式采集优势，被广泛应用于认知神经科学、临床神经监测和脑机接口等领域。然而，EEG 信号易受到眼电、工频干扰及电极接触不稳定等多源噪声影响，且不同受试者之间存在明显的生理结构、信号分布和伪迹模式差异，使得现有去噪方法在跨主体场景下的泛化能力受到限制。针对这一问题，本文提出一种基于受试者感知扩散概率模型的脑电信号去噪方法，即 Subject-Aware Denoising Diffusion Probabilistic Model（SADDPM）。该方法以 DDPM 的渐进式噪声建模与反向重建机制为基础，引入可学习的受试者嵌入向量，并通过多层条件注入方式调控反向去噪过程，使模型能够根据不同受试者的信号特性自适应调整噪声估计与信号恢复路径。本文与传统独立成分分析方法进行对比，实验结果表明，SADDPM 在多数跨主体组合中能够获得更稳定的分类表现，并在受试者相关性分析中较好地保留个体特异性信号结构。研究结果说明，将扩散模型与受试者感知机制结合，有助于提升 EEG 去噪模型在复杂噪声环境和跨主体应用中的鲁棒性，为后续脑电信号分析和脑机接口系统提供更可靠的数据基础。

**关键词：**脑电信号去噪；扩散模型；深度学习；信号建；脑机接口（BCI）

**引言**

脑电图（Electroencephalography, EEG）它是一种可对大脑神经活动进行非侵入性记录的生理信号，在认知神经科学、临床神经病学以及脑机接口（Brain-Computer Interface, BCI）等领域有着广泛应用\[3\]。EEG有高时间分辨率，获取方式也较为便捷，在神经功能监测、疾病辅助诊断、意识状态评估等任务里发挥着关键作用，不过EEG信号自身的信噪比较低，很容易受到肌电（EMG）、电源干扰（如50/60Hz工频）以及电极接触不良等多种伪迹和环境噪声的影响，这对原始神经信息的表达以及下游分析的准确性造成了严重干扰\[1\] \[2\]。

为了降低噪声干扰并提取可靠的脑电特征，EEG去噪始终是该领域研究里的核心问题之一，传统方法主要有带通滤波、独立分量分析（Independent Component Analysis, ICA）等，这些方法在特定场景中表现良好，然而普遍存在两个关键瓶颈：其一，对参数设置以及噪声分布假设十分敏感，很难适配动态复杂的实际采集环境\[11\]，其二，忽视了个体间脑电数据在解剖结构、认知状态以及采集条件等差异，使得模型在跨受试者场景中泛化能力下降\[13\]。

近些年来深度学习技术给EEG信号建模以及去噪给予了新的思路，生成对抗网络（GAN）以及变分自编码器（VAE）等在去噪图像、音频重构等领域取得成功应用后，引发了在神经信号处理方面的探索热潮\[5\]，不过这类模型一般要依靠大规模同分布样本开展训练，针对有高度异质性以及非平稳性的EEG数据，仍然面临泛化困难以及稳定性方面的挑战。

扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM），是近年来兴起的生成建模框架，凭借渐进式重构、训练稳定性好、对复杂分布有较强建模能力等优点，在图像、音频等多个生成任务方面有了重大突破\[6\]，它的核心思路是把数据逐渐扰动成噪声，再经逆扩散过程慢慢还原成干净样本，这种逐层建模的结构与EEG信号逐级净化和结构恢复的需求天然契合。虽然已有研究尝试把扩散模型应用于心电等其他生理信号的重建任务\[7\]，然而针对EEG场景，考虑跨主体个体差异的DDPM建模尚处于起始阶段，缺少有效的实践与系统的探索\[10\]。

本研究呈现出一种针对跨主体EEG信号去噪的全新策略，即Subject-Aware Denoising Diffusion Probabilistic Model（SADDPM），此策略首次把受试者感知机制纳入DDPM的去噪流程内，给予模型对个体域间差异的识别与适应能力。在建模进程里，针对每一位受试者去学习其独特的嵌入向量，借助多层条件注入来调控扩散过程，让模型可依据不同主体的特性动态调整噪声估计路径，达成鲁棒且可迁移的去噪性能。本研究重点是将生成式建模的表达能力与个体感知机制的泛化本领相融合，针对EEG信号里的结构性伪迹以及个体差异情况，打造出有生理层面解释作用和实际应用潜力的去噪模型，为后续基于EEG的认知分析以及脑机交互提供更为可靠的基础支持，本研究主要来构建一种基于DDPM的领域特异性EEG信号去噪办法，用以有效处理跨主体泛化能力欠缺的问题。具体贡献如下：

-   开发并达成领域特异性扩散概率模型（SA-DDPM），借助扩散模型独具的逐步去噪方式，达成对不同受试者的噪声特性与信号特性的有效区分。

-   借助对模型架构给予优化以及对训练策略展开设计，以此提升去噪方法在跨主体方面的泛化性能，保证模型于训练样本之外的新受试者数据当中依旧可维持良好的去噪效果。

-   本文构建了完整的数据处理流程、模型实现架构以及评估体系，并且在BCI Competition IV 2a数据集上开展系统实验，验证所提出方法在复杂噪声环境以及跨主体任务中的有效性与优越性

2 相关工作

2.1 脑电去噪相关工作

2.1.1传统脑电去噪方法

在深度学习方法尚未兴起的时期，传统信号处理技术在EEG去噪领域占据着长期的主导地位，如滤波（Filtering）、小波变换（Wavelet Transform）与独立成分分析（Independent Component Analysis, ICA）等方法，由于其理论成熟且实现过程简便，被广泛应用于实际的EEG数据预处理以及去噪工作当中，这些经典方法各自有独特的特点，在一定的条件之下可有效地提高EEG信号的质量，然而它们也存在着固有的局限性，在处理复杂、非平稳以及多源噪声环境时会表现出受限的情况。

滤波技术属于最基础的 EEG 去噪方式，借助频域将特定频段的噪声成分滤除，例如低频基线漂移（\<0.5 Hz）和高频肌电干扰（\>40 Hz），常见的方法有带通滤波（Bandpass Filtering）、陷波滤波（Notch Filtering）等\[1\]，这些方法计算开销不大，便于实时应用，不过其假设信号与噪声在频率上可严格分离，对于频谱重叠严重的状况难以处理，比如眼电伪迹（EOG）与δ波段脑电信号（0.5--4 Hz）存在频率重叠。另外如果滤波器设计不合适，可能会致使信号相位失真或者信息丢失，对后续分析的准确性产生影响。

小波变换因其有良好的时频局部化特性，成为另一类常用的EEG去噪工具，小波去噪（Wavelet Denoising）是把信号分解成不同尺度的子带，对高频系数进行阈值处理来实现噪声抑制，和传统滤波相比，小波方法能适应信号的非平稳性，特别适合处理瞬态伪迹（如眨眼瞬间产生的EOG），不过小波去噪的效果在很大程度上依赖于小波基选择、分解层数以及阈值策略，并且在处理强非加性噪声或多源干扰（如EMG与环境电磁噪声叠加）。另外过度阈值化可能致使有用脑电成分被误杀，损害信号的原始结构。

独立成分分析（ICA）作为盲源分离（BSS），在EEG伪迹剥离领域有着广泛的运用，ICA假定EEG观测信号是多个统计独立源信号的线性混合，借助最大化源分量独立性，可把神经信号和眼电、肌电伪迹有效分开，ICA特别适用于多通道EEG数据，能在无监督情形下提取出主要伪迹成分。不过ICA也有明显的局限性：其一，独立性假设在实际EEG信号里并非总能严格成立，其二，ICA对伪迹分量的识别往往需要人工干预或者引入二次判别标准，这增加了应用的复杂性，其三，ICA对电极数量与数据量有一定要求，而且在电极数量较少或者噪声水平较高时分离效果会明显降低。

滤波、小波变换以及ICA等经典方法在特定条件下可有效提升EEG信号质量，不过它们一般依赖对噪声特性的先验假设（如频率分布、独立性等），缺乏自适应能力，难以应对真实场景里复杂多变的混合噪声环境，并且这些方法在跨主体应用时往往无法自动适配不同受试者的信号特性，存在较为严重的泛化能力不足问题。虽然传统方法在部分应用中仍发挥着作用，但随着任务复杂度提升，依赖数据驱动、有更强建模能力的新一代去噪方法成为发展趋势。

2.1.2基于深度学习的去噪方法

近年来深度学习技术在EEG信号去噪领域的应用引发了诸多关注，如卷积神经网络（CNN）、循环神经网络（RNN）以及去噪自编码器（DAE）这类模型，依靠端到端的特征提取以及重建机制，给复杂噪声背景下的EEG信号恢复给予了全新的想法。

卷积神经网络（CNN），凭借其出色的局部特征提取能力，在EEG去噪任务里成为被广泛采用的结构，基于一维卷积（1D-CNN）针对时域信号进行建模，或者基于二维卷积（2D-CNN）对经过短时傅里叶变换（STFT）或小波变换的时频域数据开展特征提取，这成为常见的设计策略\[1\]。CNN在抑制高频肌电噪声（EMG）以及眼电伪迹（EOG）方面有着良好的表现，可在维持脑电波形局部结构的切实降低伪迹干扰所带来的影响，不过卷积结构的感受野存在限制，主要聚焦于局部区域特征，对于EEG信号中长期依赖关系的捕获能力欠缺，并且在应对信号非平稳性与跨尺度动态变化方面存在一定的局限性。

循环神经网络（RNN），以及它的变体例如长短时记忆网络（LSTM）以及门控循环单元（GRU），在EEG去噪里被用来强化时间动态建模，借助递归结构，RNN可有效地捕捉EEG信号里的长程时序依赖，适应眼电伪迹、基线漂移等有着时间关联特性的噪声建模需求，有部分研究还引入了注意力机制（Attention Mechanism）来提升对关键信号片段的建模能力\[3\]，或者把CNN与RNN相结合，前端卷积提取局部特征，后端循环建模全局时序依赖。虽然RNN方法在建模复杂时间动态特征方面显示出较强优势，不过因为其固有的序列处理模式，训练过程中容易出现梯度消失或爆炸的问题，并且推理速度相对较慢，增加了实际应用中的计算开销。

去噪自编码器（Denoising Autoencoder, DAE）作为一种无监督学习框架，在EEG信号噪声抑制任务里有着广泛应用，在训练阶段，DAE会给输入信号叠加噪声，让网络在编码解码过程中重建原始的干净信号，以此促使模型学习到信号自身的结构特征。标准DAE一般采用浅层编码器 - 解码器架构，近年来还发展出了卷积型DAE也（CDAE）、堆叠式DAE（SDAE）等提高形式，来提升非线性特征提取能力以及在复杂噪声环境下的去噪性能，DAE结构在小规模数据情况下表现较为稳定，对弱标签或者无标签数据有良好的适应性，并且训练过程相对简单\[4\]。不过传统DAE大多把均方误差（MSE）当作重建损失函数，很难有效对眼电伪迹、肌电干扰等非高斯、非加性复杂噪声进行建模\[11\]，同时还缺少对EEG信号中时序动态特性的刻画，使得在跨时间段、跨主体环境下去噪性能降低。

当下存在基于卷积神经网络、循环神经网络以及深度自编码器的脑电图去噪方法，于特定噪声类型状况下已可达成较高的信号恢复质量，在处理短时高频噪声以及局部突发伪迹方面效果良好，不过当面对信号分布变化剧烈、噪声成分复杂多样且来源众多、以及跨主体泛化要求较高的应用场景之时，依旧存在建模能力受限、适应性欠佳的问题。那么怎样结合多尺度特征提取、时间动态建模以及领域适应策略，以此提升深度学习方法在复杂环境下的泛化鲁棒性，便成为后续研究的关键方向。

2.2 EEG的个体差异性

EEG信号的个体差异主要源自生理和实验条件等多方面的不同，如大脑皮层折叠模式、颅骨厚度以及头皮组织结构等解剖结构差异因素，会直接对神经电活动在体积传导过程中的衰减与扩散程度产生影响，致使不同受试者头皮上记录到的EEG波形幅值和分布特性出现差异\[13\]。年龄、性别、健康状况（如疲劳水平、药物影响、神经疾病）以及心理状态（如注意力集中程度、情绪波动）等生理状态与认知特征的不同，也会改变EEG信号的频谱组成与时域动态\[2\]，另外采集条件存在微小差异，例如电极放置偏差、接触阻抗变化以及设备硬件噪声水平变化，这加大了受试者之间数据分布的不一致性\[11\]。

多方面存在差异，致使即使是在完全相同的实验任务以及外部条件之下，不同受试者所产生的EEG数据依然呈现出系统性的分布漂移（Distribution Shift），这种分布方面的差异会直接对基于数据驱动方法的建模效果产生影响，会让单一主体训练的模型在迁移到新受试者时，性能出现明显下降，成为制约EEG分析系统大规模推广应用的一个关键瓶颈。

近年来在信号建模里引入了受试者感知（Subject-Aware）建模策略，以此来有效刻画并适应EEG数据中普遍存在的个体差异性\[13\]，该策略指出，每一位受试者在信号特征、伪迹污染模式以及背景环境条件方面都有其特定之处，比如说，有些受试者的功率谱密度分布更倾向于低频提高，而另外一些受试者则是高频成分较为突出\[2\]。不同受试者的眨眼频率、肌肉活动水平、基线漂移幅度等伪迹模式也呈现出系统的差异，而且采集实验中的微环境变化也会对受试者数据质量产生额外的影响，受试者感知建模在去噪与特征建模过程中显式引入个体特性信息，帮助模型有针对性地调整信号恢复与特征抽取路径，这种方法在理论上可突破传统统一建模（Subject-Generalized）策略的局限，有效提高跨主体环境下的建模适应性与性能稳定性。

在EEG去噪任务里，受试者之间的分布差异引发了被称作个体差异鸿沟（Subject Gap）问题，即使在源受试者（Source Subject）上训练出了良好性能，当迁移到目标受试者时（Target Subject），模型的效果通常会大幅降低\[3\]，出现这种现象的原因包含不同主体原始脑电信号特性发生变化、伪迹污染模式存在差异（如肌电伪迹强度、眼电伪迹频率），以及背景信噪比水平不一样。设计可感知受试者特性并且适应个体差异的去噪方法，是提高EEG跨主体应用鲁棒性与实际可用性的关键所在。

2.3扩散概率模型与信号去噪

与传统的EEG信号去噪办法（如带通滤波、ICA）相比较而言，基于扩散概率建模的DDPM方法在应对复杂噪声环境时呈现出优势。首先，DDPM拥有稳定的训练机制，其损失函数是直接依据噪声预测的，不需要判别器来辅助，这样就避免了生成对抗网络（GAN）训练时常见的模式崩溃（Mode Collapse）以及训练不稳定的问题\[6\]，提升了建模过程的可控程度与收敛效果，DDPM采用的是渐进式重建过程，借助逐步去噪的办法，在信号重建时能精细地捕捉局部特征和全局特征，有效保留原始信号的细节，还可以抑制因去噪而引发的信息丢失\[7\]。扩散模型在理论方面有逼近任意复杂数据分布的能力，可灵活地对EEG信号里高度非平稳、异质性的多源噪声污染模式进行建模\[10\]，DDPM框架自然就支持条件控制（Conditional Generation），可灵活地引入辅助信息，如受试者特征编码（Subject Embedding），以此实现受试者感知（Subject-Aware）去噪\[17\]，在跨主体场景下提高模型的泛化能力以及信号恢复质量。

把DDPM框架引入到EEG信号去噪任务里，再结合受试者感知建模策略，所提出的SADDPM方法有希望在现有去噪方法之上实现性能的大幅提升，在面对复杂噪声条件以及受试者间高度异质性挑战时可呈现出更出色的稳定性和鲁棒性\[10\]。

另外一方面，近些年来开始兴起的深度学习方法，虽然在一定程度上缓解了传统方法存在的不足，然而当下的主流模型（如CNN、RNN、DAE），大多专注于局部特征的提取或者短期时间动态的建模，缺少对受试者间领域差异性（Domain Specificity）的系统适应。多数现有的深度学习去噪研究依旧是基于单一主体或者同质化数据集来开展训练与验证工作的，还没有充分考虑跨主体迁移以及领域自适应这些问题，在真实开放环境中应用时性能波动比较大。

在去噪建模方法的选择上，虽然例如生成对抗网络（GANs）这样的生成模型在一些信号去噪任务里已经表现出了一定潜力，然而它在训练过程中存在着不稳定的情况、模式崩溃问题（Mode Collapse）以及评估方面的难题，这使得它在对可靠性要求较高的应用场景中的使用受到了限制，相比之下，扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）作为近些年新出现的一种生成框架，凭借其稳定的训练机制、逐步推进的信号恢复过程以及高保真的重建能力，在图像生成与重建领域收获了突破性的成果\[6\]。

然而当前在EEG信号去噪任务方面基于DDPM的研究还是一片空白，现有的扩散模型应用大多集中于图像、音频等自然信号领域，对于电生理信号有多源异质性、高非平稳性特点的EEG数据还缺乏系统的探索，扩散模型本身有逐步重建、条件控制等特性，从理论上来说很适合用于构建复杂脑电信号中的噪声去除过程，在跨受试者、异质域环境下，有潜力凭借引入如受试者标签、特征编码这样的领域特异性信息来实现自适应去噪。

依据上述分析可知，当下EEG去噪研究急需打破传统静态方法的限制，引入有更强自适应性、可控性以及泛化能力的新型生成建模框架，结合扩散模型的优势，探寻面向领域特异性建模的EEG去噪新办法，已然成为后续研究的关键方向。

**3 方法Subject-Aware Denoising Diffusion Probabilistic Model**

本研究提出Subject-Aware Denoising Diffusion Probabilistic Model（SADDPM），在扩散建模框架里引入受试者感知（Subject-Aware）的机制，达成针对不同主体动态适应的噪声建模和动态适应的去噪过程，SADDPM于扩散与反扩散阶段融入受试者的域信息，引导噪声恢复路径根据不同个体进行针对性改变，以此提升跨主体场景下去噪性能与信号保真度。SADDPM整体包括三大核心模块：

1.  基于受试者编码（Subject Embedding）的条件扩散过程，此过程会模拟个体特定的噪声特征；

2.  基于受试者感知的反向采样过程，此阶段模型将自适应地重建受试者特征信号；

3.  端到端训练机制，通过噪声预测与信号重建联合优化，强化信号恢复质量与领域适应能力。

在DDPM模型的基础上借助引入受试者感知机制，最终的SADDPM模型可在维持传统扩散建模稳定性优势的状况下，系统性地处理EEG去噪里的跨主体领域偏移问题，实现比较鲁棒的去噪效果。从跨主体应用这一角度上，解决了传统方法一般假定训练数据与测试数据源自相同的统计分布，却忽视了不同受试者之间脑电信号在空间分布、频谱特性以及噪声污染模式方面存在的差异的问题。这种领域（Domain）间的分布差异在实际应用里会让去噪模型的泛化能力降低，对 EEG 信号分析系统的稳定性与普适性造成严重限制。在多受试者数据集以及移动采集环境中，不同主体的电极接触条件、背景噪声环境以及生理状态发生变化，致使传统静态且统一参数的去噪策略难以维持稳定有效。

本文提出的SADDPM方法应对复杂且多源的噪声环境时，存在着建模能力受限以及跨主体泛化性欠佳的状况，随着深度生成模型的不断发展，扩散概率模型（DDPM），凭借其稳定的训练机制以及渐进式重建特性，为对复杂噪声与信号的演变进程进行建模提供了全新的可能性，因此本研究采用了由DDPM提供基础去噪能力的模型来进行脑电信号处理，构建了以扩散概率模型为基础模型的脑电去噪系统。然而标准的DDPM对于脑电去噪情况仍然存在一定的缺陷，该方法假定来训练数据源自统一分布，没有明确地对不同受试者（Subject）之间的分布差异进行建模，这会对其在实际EEG跨主体应用中的去噪效果造成限制，从而影响模型的实际去噪能力和可迁移性。为复杂脑电信号环境下的去噪以及特征保持给予新的解决途径。

3.1 扩散过程与反扩散过程建模

在SADDPM里，EEG信号去噪建模依据标准DDPM的扩散以及反扩散思想，不过在此基础之上还引入了受试者感知机制，以此来适应不同主体之间信号分布和噪声模式的差异，展开来说，该模型含有正向扩散（Forward Diffusion）与反向去噪（Reverse Denoising）这两个过程，正向扩散模拟噪声注入的动态变化，反向去噪模拟信号逐步恢复的动态变化，并且在每一步都会引入受试者编码信息进行条件控制。

正向扩散过程旨在逐步将原始干净信号 $x₀\ $转变为近似各向同性高斯分布的噪声 $xₜ\ $。这一过程通过一系列小步高斯扰动迭代实现，具体定义为：

$$\begin{matrix}
q\left( x^{1}:ₜ \middle| x^{0} \right) = \prod_{}^{}{ₜ^{= 1ₜ}q\left( xₜ \middle| xₜ^{- 1} \right)}\#\left( 7 \right) \\
\end{matrix}$$

其中，

$$\begin{matrix}
q\left( xₜ \middle| xₜ^{- 1} \right) = \mathcal{N}\left( xₜ;\sqrt{\left( 1 - \beta ₜ \right)}xₜ^{- 1},\beta ₜI \right)\#\left( 8 \right) \\
\end{matrix}$$

$\beta ₜ$为第$t$步扩散噪声强度超参数，通常按照预设的时间调度策略缓慢增加，使信号逐渐丧失原始结构信息。

为了简化采样，DDPM允许直接从$\ x₀\ $采样任意扩散步t的中间状态，满足：

$$\begin{matrix}
q\left( xₜ \middle| x^{0} \right) = \mathcal{N}\left( xₜ;\sqrt{\left( \alpha\bar{ₜ} \right)}x^{0},\left( 1 - \alpha\bar{ₜ} \right)I \right)\#\left( 9 \right) \\
\end{matrix}$$

其中，

$$\begin{matrix}
\alpha\bar{ₜ} = \prod_{}^{}{ₛ^{= 1ₜ}\left( 1 - \beta ₛ \right)}\#\left( 10 \right) \\
\end{matrix}$$

在SADDPM中，正向扩散过程保持与标准DDPM一致，不引入额外变化，以保证噪声建模过程的稳定性与理论一致性。

反向去噪过程学习从高斯噪声逐步恢复干净信号的生成轨迹。该过程建模为参数化马尔可夫链：

$$\begin{matrix}
\text{pθ}\left( x^{0}:ₜ \right) = p\left( xₜ \right)\prod_{}^{}{ₜ^{= 1ₜ}\text{pθ}\left( xₜ^{- 1} \middle| xₜ \right)}\#\left( 11 \right) \\
\end{matrix}$$

其中，每一反向条件分布设定为高斯分布：

$$\begin{matrix}
\text{pθ}\left( xₜ^{- 1} \middle| xₜ \right) = \mathcal{N}\left( xₜ^{- 1};\mu\theta\left( xₜ,t,e \right),\Sigma\theta\left( xₜ,t,e \right) \right)\#\left( 12 \right) \\
\end{matrix}$$

$\text{μθ}$ 和$\text{\ Σθ\ }$分别为预测的均值与方差参数，e 表示受试者编码（Subject Embedding），作为条件信息输入网络。

与标准DDPM不同，SADDPM在反向过程中引入了受试者感知机制。具体地来看，是在每个时间步$t$上，将受试者编码 $\text{e\ }$与当前状态$\ xₜ\ $共同作为网络输入，用于预测下一个状态$\ xₜ₋₁$，以此类推。这种个体和当前状态共同的条件控制使得模型能够有针对性的根据不同主体的信号分布特征与噪声特性，自适应调整去噪路径，生成去噪信号。

3.2 受试者编码与条件引导

受试者编码$e$是用来提供个体特异性信息的，其作用是引导扩散逆向过程，编码方式可采用下面这两种策略：

1.  One-hot编码（One-hot Encoding）：每个受试者对应唯一的编码向量，适用于受试者数量有限的场景；

2.  嵌入向量（Learnable Embedding）：引入可训练的低维受试者向量表示，通过与主网络联合训练，自动学习个体间的潜在差异性。

在实际开展建模工作时，受试者编码$\text{e\ }$与 $xₜ$于输入层进行融合（例如通过特征连接、加法或调制机制），之后在扩散逆向神经网络的多个层次中注入，以此来提高条件控制的能力，这样的机制让SADDPM可恢复普通的EEG信号结构，而且还可以针对每一个受试者，依据其个体情况调整去噪策略，提高信号恢复的保真程度以及模型的泛化能力。

3.3 模型结构设计

SADDPM在其整体架构方面，继承了扩散概率模型里常用的U-Net结构，依靠对称的编码器和解码器模块来达成多尺度特征的提取以及重建工作，在这个基础之上，引入了受试者感知机制，借助嵌入向量针对扩散以及去噪过程实施条件引导，以此实现个体特异性的建模，其整体结构覆盖了特征提取模块（Encoder）、扩散特征处理模块（Middle Block）、特征重建模块（Decoder）以及受试者条件注入机制（Subject Conditioning Module）。

![图片3](media\media\image1.png){width="6.299305555555556in" height="2.359027777777778in"}

图表3 模型整体架构示意图

（1） 基础网络结构

SADDPM采用改进的一维卷积U-Net架构（1D Conv U-Net），以适应EEG信号的时序特性。具体设计如下：

Encoder：由多层一维卷积块组成，每个卷积块包含卷积层、归一化层（Group Norm）和非线性激活函数（如Swish）。每经过一层卷积，特征图长度减半，通道数增加，从而提取不同尺度的局部时序特征。

Middle Block：位于U-Net中间，包含若干自注意力模块（Self-Attention Block），用于捕捉全局依赖关系，强化长程时间动态建模能力。

Decoder：对称于Encoder，由反卷积（或上采样+卷积）模块组成，用于逐步恢复特征图尺寸。通过跨层连接（Skip Connection）将Encoder的中间特征直接传递到Decoder对应层，保留局部细粒度信息，提升信号重建质量。

Time Embedding：在每一扩散步骤$t$，使用正弦余弦位置编码（Sinusoidal Positional Encoding）对时间步进行嵌入，提供扩散步数信息，使网络感知当前去噪阶段。

整体U-Net结构确保了SADDPM在信号局部与全局层面都能有效建模，适配EEG信号中多尺度、非平稳动态变化特性。为了适应脑电信号（EEG）在时间域上的非平稳性与多尺度动态特征，SADDPM 模型采用了一种基于一维卷积的 U-Net 架构（1D Conv U-Net）作为主干去噪网络，以充分建模信号中的局部细节与全局结构。该网络以扩散过程中的中间变量 $xₜ$ 作为输入用于反向过程中的信号重建。

在编码阶段，模型通过多层一维卷积结构逐步提取时间序列中的局部特征。每个卷积单元由卷积层、分组归一化（Group Normalization）和非线性激活函数（如 Swish）构成。随着网络层数的加深，特征图的时间长度逐层减半，通道数同步增加，从而实现从局部到全局的时序抽象能力。在网络中部，模型引入了若干自注意力模块（Self-Attention Blocks），以增强长程依赖建模能力，这对于建模跨通道、跨时间段的脑电模式变化具有重要意义。在解码阶段，网络结构与编码器对称，采用上采样操作（如反卷积或插值+卷积）逐步恢复时间维度，并通过跳跃连接（Skip Connection）将编码器各层的中间特征直接传递到解码器对应层，以保留原始信号的局部细节信息，提升最终重建质量。

此外，模型在扩散建模框架中引入了时间步嵌入机制。具体而言，在每一步扩散过程中，模型利用正弦-余弦位置编码（Sinusoidal Positional Encoding）对当前时间步 t 进行嵌入，并将该时间编码注入至各层卷积模块中，使网络在进行噪声预测时能显式感知当前所处的扩散阶段，从而动态调整去噪策略。在主干去噪结构之外，SADDPM 还设计了并行的个体差异建模模块，用于引入受试者级别的条件建模能力，以增强模型在跨主体 EEG 信号建模中的泛化性。该模块的设计动机在于：不同受试者的 EEG 数据在统计分布上存在显著差异，传统的非条件生成模型在面对分布偏移时容易产生去噪失真或判别失效。因此，SADDPM 通过引入个体特征感知机制，使模型能够在还原信号的同时感知和适配受试者之间的个体差异。

在训练过程中，SADDPM 同时引入多项损失函数以对主干网络与个体建模模块进行联合优化。首先，标准的重建损失 $\text{Lr\ }$衡量模型所预测噪声与真实噪声之间的均方误差，用以驱动主分支进行准确去噪；其次，主干网络输出与个体差异支路之间还引入反向一致性损失 $\text{Lc}$，该损失鼓励两个路径在预测内容上的协同一致，防止个体编码信息被主干忽略；再次，通过计算主分支输出与个体建模路径预测分布之间的 KL 散度，引入内容感知损失项 $\text{Lo}$，以进一步强化个体感知对信号建模过程的调控能力；最后，为约束个体编码器具备良好的身份判别能力，引入个体识别损失 $\text{La}$，通常采用交叉熵形式，以受试者标签作为监督信号。

通过以上结构与机制的联合设计，SADDPM 能够在高效建模 EEG 信号局部细节与全局动态的同时，引入受试者特异性建模能力，显著提升跨主体条件下的去噪质量与分类性能，尤其在信号分布差异较大的受试者组合中展现出更强的鲁棒性

（2）受试者感知机制建模

受试者编码（Subject Embedding）通过条件注入（Conditioning Injection）方式融合到网络中，引导模型针对不同受试者调整特征处理与生成过程。具体策略：

受试者编码生成：针对每一个受试者，都会分配一个可进行训练的低维嵌入向量$e \in \mathbb{Rᵈ}$，此向量在训练进程当中会和主网络的参数一同进行优化，初始的嵌入向量借助随机初始化的方式获得，之后依据数据驱动的方式逐步实现优化，自动捕捉个体之间的特性差异。

条件注入位置：在网络中对受试者编码进行多层注入，具体的实现办法是，于 Encoder、Middle Block 以及 Decoder 各个模块的卷积层之前，借助线性变换把𝑒映射成特征图尺寸，然后与主特征实施逐元素加和或者进行特征拼接（Concat）。

融合机制：采用特征调制（Feature Modulation）方式来兼顾训练稳定性以及表达能力，这种方式是借助受试者编码所生成的缩放（Scale）与偏移（Shift）参数，以此对卷积块输出特征分布给予调整：

$$\begin{matrix}
h^{'} = \gamma\left( e \right) \odot h + \beta\left( e \right)\#\left( 13 \right) \\
\end{matrix}$$

其中，$h\ $为中间特征，$\gamma(e)$ 与 $\beta(e)$ 为受试者编码经过小型感知机（MLP）变换得到的调制参数。

（3）损失函数与训练目标

SADDPM 采用与标准 DDPM 一致的噪声预测（Noise Prediction）训练目标。给定清洁信号 $x₀$，在扩散第 $t\ $步，将加噪后的样本$\ xₜ$ 输入网络，预测叠加噪声 $\varepsilon$，训练损失定义为：

$$\begin{matrix}
L_{s}\text{imple}\left( \theta \right) = E_{t,x^{0},\varepsilon,e}\left\lbrack \left\| \varepsilon - \varepsilon_{\theta}\left( \sqrt{\alpha\bar{ₜ}}x^{0} + \sqrt{\left( 1 - \alpha\bar{ₜ} \right)}\varepsilon,t,e \right) \right\|^{2} \right\rbrack\#\left( 14 \right) \\
\end{matrix}$$

其中，$\varepsilon_{\theta}$ 为网络输出，$e$ 为受试者编码。

训练过程中随机采样扩散步 $t$，标准正态噪声 $\varepsilon$，并对受试者编码 $e$ 进行条件注入，优化整个 SADDPM 模型参数与受试者嵌入向量。

采用噪声预测作为主损失，能够稳定训练过程，并促使模型在不同噪声水平下逐步恢复原始信号。通过联合优化受试者感知机制，进一步增强模型在跨主体环境下的去噪性能与信号保真性。

**4.实验**

4.1 实验数据

本研究基于BCI Competition IV 2a数据集进行评估，该数据集是脑机接口（BCI）领域标准的重要基准之一，广泛用于多类别运动想象任务的EEG信号分析研究。数据集设计科学、结构完备，涵盖多受试者、多任务、多session的复杂实验设置，充分适合验证跨主体建模与去噪算法的泛化性能。

数据集共包含来自9名健康受试者（编号A01至A09）的EEG记录数据。每名受试者在不同日期完成了两次实验会话（Session T和Session E），分别用于模型的训练与测试。每个session包括6个runs，每个run包含48次trials，因此每位受试者每个session共有288个完整试次记录。

每次trial遵循统一的实验流程：受试者首先凝视屏幕中央的十字提示符（0--2秒），随后出现指示箭头（左/右/下/上，分别对应左手、右手、双脚、舌头运动想象），持续约1.25秒。受试者需根据箭头指向进行相应的运动想象，并持续至屏幕指示结束（6秒），之后为短暂休息期。每个trial平均总时长约8秒，包含了准备期、任务执行期与恢复期。

EEG数据采集采用$\text{Cz}$参考布局，总计22个通道，采样率250 $\text{Hz}$。同期记录了3通道的EOG信号，用于辅助伪迹分析与建模。所有信号经过统一硬件滤波$（0.5 - 100\ Hz）$，并以高质量设备采集，保证数据一致性与可用性。

本研究选用全部受试者数据进行建模与测试，充分覆盖不同个体间的信号差异性。为保证实验的一致性和可复现性，本工作采用官方数据集原始划分，不进行人工重采样或信号截断。各受试者训练（Session T）与测试（Session E）数据严格分开，模拟实际应用中模型部署后的独立测试场景。

鉴于数据集中涵盖四类运动想象任务，且原始记录中存在自然噪声（如眨眼、肌电活动）干扰，BCI Competition IV 2a数据集为本研究探索复杂噪声环境下EEG去噪建模提供了坚实的数据基础。

4.2 数据预处理

在开展模型训练以及测试工作之前，本文针对原始EEG信号实施了标准化预处理操作，其目的在于提升信号质量并且保证不同受试者之间的数据有一致性，就通道选择而言，将全部22个脑电通道给予保留，把参考电极与眼电通道去除掉，以此来聚焦于大脑皮层区域神经活动的建模。随后，应用 $1 - 50\ Hz\ $的有限冲激响应（FIR）带通滤波器以去除低频基线漂移与高频肌电干扰，从而保留与认知和运动任务密切相关的脑电频段。针对电力系统引起的电磁干扰，进一步采用 50 Hz 陷波滤波器，有效抑制主频及其高次谐波对信号的影响。

鉴于不同采集设备的采样率存在差别，为了使输入特征维度保持一致，如果原始采样率高于250Hz，那就统一把它重采样到250Hz，这样能减轻计算负担并且提高建模的稳定性，在信号结构处理这个方面，运用滑动窗口策略把连续的EEG数据划分成固定长度的短时片段，窗口长度设定为2秒，滑动步长是0.5秒，在保证样本覆盖率的情况下生成足够数量的训练样本。为了消除不同受试者之间因为信号幅值尺度不一样而产生的分布偏移，所有片段在输入之前都经过逐通道的零均值、单位方差归一化处理，这种处理方式可维持训练过程中的数值稳定性，还可以提高模型在跨受试者任务中的泛化能力。

4.3 实验设置

为了保证模型在复杂噪声环境与跨主体数据上的稳定学习能力，SADDPM 的训练过程采用了统一且结构化的参数设定。训练优化器为 Adam，初始学习率设置为 $1 \times 10^{- 4}$，并结合余弦退火调度策略，使学习率在训练过程中平滑下降，从而有助于后期收敛的稳定性。每轮训练使用的批次大小为 64，扩散过程共设定 1000 个时间步，其中噪声调度参数 $\beta_{t}$ 采用线性增长策略，确保信号从高噪状态向干净表示的渐进式还原能够逐步学习完成。整个训练过程共迭代 100 轮，每轮遍历完整训练集，充分利用所有样本。

| 训练算法 |                                                                  |
|----------|------------------------------------------------------------------|
| 1:       | While 未收敛 do                                                  |
| 2:       | 从样本分布q(X0, s) 中采对应的受试者标签样X0, s                   |
| 3:       | 使用样本标签对{ X0, s}预训练受试者分类器ω                        |
| 4:       | 从均匀分布中采样扩散步t \~ Uniform({1,\...,T})                   |
| 5:       | 从标准正态分布采样噪声ε \~ N(0,I)                                |
| 6:       | 使用参数θ 和 Ф计算：εθ (xt, t, s)， εФ(xt, t)， ω (Eo(xt,t, s)). |
| 7:       | 执行梯度下降步骤，优化θ 和 Ф on                                  |
| 8:       | 计算损失函数：￡= λr￡r+λo￡o + λarc￡are + λtd￡td                   |
| 9:       | end while                                                        |

在对EEG个体差异进行建模期间，SADDPM针对每一位受试者引入了一个维度为128的低维嵌入向量，以此来捕捉其独特的噪声分布以及神经响应特征，这个向量在训练过程中会和主干网络参数一同进行优化，并且是依靠条件注入的方式作用于扩散过程的多个阶段，为了可提升泛化能力，同时避免个体嵌入出现过度拟合训练集的情况，在训练时对嵌入向量施加了L2正则化，其权重设置为$1 \times 10^{- 4}$。所有的实验都是在相同的硬件环境下完成（NVIDIA A100 GPU），训练配置在不同方法之间保持一致，以此保证各组实验结果有可比性以及公平性。

为了全面地评估SADDPM模型在跨主体场景之中的泛化能力，把Leave-One-Subject-Out（LOSO）交叉验证策略当作主要的评估方法，展开来说，在每一轮的验证过程里，从九位受试者当中挑选出一名的全部数据（Session E）作为测试集，而其余八名受试者的训练数据（Session T）作为训练集。当九轮循环结束之后，把所有受试者在独立测试集上的指标结果加以平均，以此来衡量模型在未知主体上的稳健性表现，该策略可在最大程度上揭示模型在面对非见过主体数据时的实际表现，是符合跨主体EEG应用的评估需求的。

4.4 实验结果与分析

为评估所提SADDPM方法用于运动想象EEG信号去噪任务时的有效性，本文于BCI Competition IV 2a数据集开展系统实验，且与经典去噪方法ICA做对比分析，评估重点有两方面，其一为不同方法去噪后对下游分类性能的影响，其二是评估SADDPM保留受试者个体信号特性的能力。

表格 1 独立成分分析方法训练结果

|     | ICA 方法去噪训练结果 (Acc.%) |       |       |       |       |       |       |       |       |
|-----|------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|     | s1                           | s2    | s3    | S4    | s5    | s6    | s7    | s8    | s9    |
| sl  | 89.29                        | 39.29 | 25.93 | 50.00 | 40.74 | 27.27 | 62.96 | 29.63 | 36.00 |
| s2  | 82.61                        | 92.86 | 33.33 | 41.67 | 33.33 | 31.82 | 33.33 | 37.04 | 36.00 |
| s3  | 53.57                        | 85.71 | 81.48 | 58.33 | 33.33 | 54.55 | 44.44 | 33.33 | 48.00 |
| s4  | 53.57                        | 46.43 | 81.48 | 91.67 | 33.33 | 54.55 | 37.04 | 37.04 | 44.00 |
| s5  | 42.86                        | 35.71 | 37.04 | 91.67 | 88.89 | 54.55 | 40.74 | 33.33 | 40.00 |
| s6  | 50.00                        | 39.29 | 37.04 | 50.00 | 77.78 | 90.91 | 44.44 | 33.33 | 44.00 |
| s7  | 57.14                        | 50.00 | 37.04 | 41.67 | 55.56 | 90.91 | 74.07 | 33.33 | 40.00 |
| s8  | 39.29                        | 39.29 | 44.44 | 50.00 | 59.26 | 45.45 | 66.67 | 74.07 | 44.00 |
| s9  | 32.14                        | 32.14 | 37.04 | 45.83 | 44.44 | 45.45 | 33.33 | 74.07 | 84.00 |
| M   | 55.61                        | 51.19 | 46.09 | 57.87 | 51.85 | 55.05 | 48.56 | 42.80 | 46.22 |

表格 2 SADDPM方法训练结果

<table><thead><tr class="header"><th></th><th><blockquote><p>SADDPM方法去噪训练结果(Acc. %)</p></blockquote></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr class="odd"><td></td><td><blockquote><p>s1</p></blockquote></td><td><blockquote><p>s2</p></blockquote></td><td><blockquote><p>s3</p></blockquote></td><td><blockquote><p>S4</p></blockquote></td><td><blockquote><p>s5</p></blockquote></td><td><blockquote><p>s6</p></blockquote></td><td><blockquote><p>s7</p></blockquote></td><td><blockquote><p>s8</p></blockquote></td><td><blockquote><p>s9</p></blockquote></td></tr><tr class="even"><td><blockquote><p>sl</p></blockquote></td><td><blockquote><p>85.59</p></blockquote></td><td><blockquote><p>46.43</p></blockquote></td><td><blockquote><p>46.43</p></blockquote></td><td><blockquote><p>52.00</p></blockquote></td><td><blockquote><p>37.04</p></blockquote></td><td><blockquote><p>31.82</p></blockquote></td><td><blockquote><p>68.22</p></blockquote></td><td><blockquote><p>37.04</p></blockquote></td><td><blockquote><p>38.46</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s2</p></blockquote></td><td><blockquote><p>85.71</p></blockquote></td><td><blockquote><p>90.01</p></blockquote></td><td><blockquote><p>34.28</p></blockquote></td><td><blockquote><p>44.00</p></blockquote></td><td><blockquote><p>44.44</p></blockquote></td><td><blockquote><p>37.27</p></blockquote></td><td><blockquote><p>32.14</p></blockquote></td><td><blockquote><p>44.18</p></blockquote></td><td><blockquote><p>35.77</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s3</p></blockquote></td><td><blockquote><p>60.71</p></blockquote></td><td><blockquote><p>92.86</p></blockquote></td><td><blockquote><p>82.11</p></blockquote></td><td><blockquote><p>58.00</p></blockquote></td><td><blockquote><p>58.15</p></blockquote></td><td><blockquote><p>55.00</p></blockquote></td><td><blockquote><p>49.29</p></blockquote></td><td><blockquote><p>51.85</p></blockquote></td><td><blockquote><p>44.62</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s4</p></blockquote></td><td><blockquote><p>52.86</p></blockquote></td><td><blockquote><p>47.21</p></blockquote></td><td><blockquote><p>78.57</p></blockquote></td><td><blockquote><p>88.45</p></blockquote></td><td><blockquote><p>48.15</p></blockquote></td><td><blockquote><p>50.00</p></blockquote></td><td><blockquote><p>38.57</p></blockquote></td><td><blockquote><p>40.74</p></blockquote></td><td><blockquote><p>46.15</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s5</p></blockquote></td><td><blockquote><p>45.86</p></blockquote></td><td><blockquote><p>35.71</p></blockquote></td><td><blockquote><p>37.04</p></blockquote></td><td><blockquote><p>91.67</p></blockquote></td><td><blockquote><p>85.47</p></blockquote></td><td><blockquote><p>52.00</p></blockquote></td><td><blockquote><p>42.81</p></blockquote></td><td><blockquote><p>36.12</p></blockquote></td><td><blockquote><p>46.15</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s6</p></blockquote></td><td><blockquote><p>52.14</p></blockquote></td><td><blockquote><p>35.00</p></blockquote></td><td><blockquote><p>46.43</p></blockquote></td><td><blockquote><p>44.44</p></blockquote></td><td><blockquote><p>79.12</p></blockquote></td><td><blockquote><p>88.00</p></blockquote></td><td><blockquote><p>49.15</p></blockquote></td><td><blockquote><p>33.33</p></blockquote></td><td><blockquote><p>50.00</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s7</p></blockquote></td><td><blockquote><p>55.71</p></blockquote></td><td><blockquote><p>50.00</p></blockquote></td><td><blockquote><p>42.16</p></blockquote></td><td><blockquote><p>52.00</p></blockquote></td><td><blockquote><p>55.56</p></blockquote></td><td><blockquote><p>90.91</p></blockquote></td><td><blockquote><p>76.12</p></blockquote></td><td><blockquote><p>43.10</p></blockquote></td><td><blockquote><p>47.28</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s8</p></blockquote></td><td><blockquote><p>39.29</p></blockquote></td><td><blockquote><p>39.29</p></blockquote></td><td><blockquote><p>44.44</p></blockquote></td><td><blockquote><p>47.25</p></blockquote></td><td><blockquote><p>59.26</p></blockquote></td><td><blockquote><p>45.45</p></blockquote></td><td><blockquote><p>70.37</p></blockquote></td><td><blockquote><p>78.88</p></blockquote></td><td><blockquote><p>44.00</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s9</p></blockquote></td><td><blockquote><p>25.15</p></blockquote></td><td><blockquote><p>35.71</p></blockquote></td><td><blockquote><p>32.14</p></blockquote></td><td><blockquote><p>38.20</p></blockquote></td><td><blockquote><p>44.44</p></blockquote></td><td><blockquote><p>45.45</p></blockquote></td><td><blockquote><p>34.07</p></blockquote></td><td><blockquote><p>73.08</p></blockquote></td><td><blockquote><p>85.15</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>M</p></blockquote></td><td><blockquote><p>55.89</p></blockquote></td><td><blockquote><p>52.47</p></blockquote></td><td><blockquote><p>49.29</p></blockquote></td><td><blockquote><p>57.33</p></blockquote></td><td><blockquote><p>56.85</p></blockquote></td><td><blockquote><p>54.48</p></blockquote></td><td><blockquote><p>51.19</p></blockquote></td><td><blockquote><p>48.70</p></blockquote></td><td><blockquote><p>48.62</p></blockquote></td></tr></tbody></table>

为验证SADDPM在跨受试者运动想象EEG信号分类任务中的应用效果，本文设计了基于BCI-IV 2a数据集的系统对比实验。具体地，我们采用受试者交叉训练-测试策略，对比了经典的独立成分分析方法（ICA）与本文提出的SADDPM在去噪处理后的下游分类准确率表现。表1与表2分别列出了在9名受试者数据上互相组合训练测试时的准确率（%）结果，M行表示每个受试者作为测试集时的平均分类准确率。

表1呈现了运用ICA和SADDPM分别实施去噪操作后，于九位受试者之间进行交叉组合训练所得到的分类准确（%）率的对比情况，从整体上看，SADDPM在大多数受试者组合上呈现出比ICA更优的分类性能，在s4-s5、s5-s6、s6-s7等组合中，准确率有着明显的提高。就s5-s6组合而言，ICA方法的准确率为77.78%，而SADDPM提升到了88.00%，它在提高运动想象任务相关信息以及抑制干扰方面有更强的建模能力。

从宏观层面进行统计可知，ICA方法的总体平均准确率为55.61%，SADDPM方法的总体平均准确率为55.89%，虽然二者在数值上的提升幅度不算大，不过SADDPM在多组受试者组合中性能表现更为稳定，波动幅度更小，这意味着，在复杂多源噪声场景下，SADDPM可实现更为稳健的特征恢复，适应多样化主体的信号特性，提升下游分类器的泛化能力。不同受试者作为源域训练数据的平均表现也有所差异。以s3为训练主体时，SADDPM在多个目标受试者上的准确率普遍较高，说明其信号表达具有较好的可迁移性；而s9为训练主体时，无论在ICA或SADDPM下的准确率均偏低，提示该受试者数据中可能存在更高比例的非加性伪迹或频带混叠干扰。SADDPM在该情况下仍能稳定保持约50%的准确率，说明其在非理想训练集上的鲁棒性更强。

在s8 - s9等跨主体分布差异明显的组合里，SADDPM比ICA表现出更强稳健性，这显示出它引入的受试者感知机制在跨主体建模中是有效的，借助个体特征引导的条件建模，SADDPM能根据情况调整信号还原策略，降低不同主体间信号分布偏移产生的影响。传统去噪方法如 ICA 通常依赖于固定的信号独立性假设，在面对主体间信号分布偏移较大的场景时，易受到模型泛化能力不足的限制。而 SADDPM 所采用的受试者条件建模策略能够有效缓解这一问题。通过在扩散反向过程中的多层注入机制，引导网络学习到与个体特征相关的信号结构，从而增强了模型对不同主体间数据分布差异的适应能力。实验证明，这种个体感知的建模方式不仅提升了去噪质量，还在跨主体泛化性能上具有显著优势。

表格 3 个体一致性分析实验结果

<table><thead><tr class="header"><th></th><th><blockquote><p>BCI-IV 脑电信号的受试者相关系数</p></blockquote></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr class="odd"><td></td><td><blockquote><p>s1</p></blockquote></td><td><blockquote><p>s2</p></blockquote></td><td><blockquote><p>s3</p></blockquote></td><td><blockquote><p>S4</p></blockquote></td><td><blockquote><p>s5</p></blockquote></td><td><blockquote><p>s6</p></blockquote></td><td><blockquote><p>s7</p></blockquote></td><td><blockquote><p>s8</p></blockquote></td><td><blockquote><p>s9</p></blockquote></td></tr><tr class="even"><td><blockquote><p>sl</p></blockquote></td><td><blockquote><p>0.102</p></blockquote></td><td><blockquote><p>0.047</p></blockquote></td><td><blockquote><p>0.055</p></blockquote></td><td><blockquote><p>0.059</p></blockquote></td><td><blockquote><p>0.045</p></blockquote></td><td><blockquote><p>0.051</p></blockquote></td><td><blockquote><p>0.050</p></blockquote></td><td><blockquote><p>0.042</p></blockquote></td><td><blockquote><p>0.043</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s2</p></blockquote></td><td><blockquote><p>0.048</p></blockquote></td><td><blockquote><p>0.074</p></blockquote></td><td><blockquote><p>0.040</p></blockquote></td><td><blockquote><p>0.038</p></blockquote></td><td><blockquote><p>0.033</p></blockquote></td><td><blockquote><p>0.041</p></blockquote></td><td><blockquote><p>0.045</p></blockquote></td><td><blockquote><p>0.037</p></blockquote></td><td><blockquote><p>0.038</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s3</p></blockquote></td><td><blockquote><p>0.056</p></blockquote></td><td><blockquote><p>0.040</p></blockquote></td><td><blockquote><p>0.080</p></blockquote></td><td><blockquote><p>0.048</p></blockquote></td><td><blockquote><p>0.037</p></blockquote></td><td><blockquote><p>0.043</p></blockquote></td><td><blockquote><p>0.048</p></blockquote></td><td><blockquote><p>0.037</p></blockquote></td><td><blockquote><p>0.039</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s4</p></blockquote></td><td><blockquote><p>0.059</p></blockquote></td><td><blockquote><p>0.038</p></blockquote></td><td><blockquote><p>0.048</p></blockquote></td><td><blockquote><p>0.088</p></blockquote></td><td><blockquote><p>0.041</p></blockquote></td><td><blockquote><p>0.042</p></blockquote></td><td><blockquote><p>0.054</p></blockquote></td><td><blockquote><p>0.043</p></blockquote></td><td><blockquote><p>0.044</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s5</p></blockquote></td><td><blockquote><p>0.045</p></blockquote></td><td><blockquote><p>0.033</p></blockquote></td><td><blockquote><p>0.037</p></blockquote></td><td><blockquote><p>0.041</p></blockquote></td><td><blockquote><p>0.058</p></blockquote></td><td><blockquote><p>0.034</p></blockquote></td><td><blockquote><p>0.039</p></blockquote></td><td><blockquote><p>0.031</p></blockquote></td><td><blockquote><p>0.034</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s6</p></blockquote></td><td><blockquote><p>0.053</p></blockquote></td><td><blockquote><p>0.041</p></blockquote></td><td><blockquote><p>0.043</p></blockquote></td><td><blockquote><p>0.042</p></blockquote></td><td><blockquote><p>0.033</p></blockquote></td><td><blockquote><p>0.071</p></blockquote></td><td><blockquote><p>0.042</p></blockquote></td><td><blockquote><p>0.034</p></blockquote></td><td><blockquote><p>0.038</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s7</p></blockquote></td><td><blockquote><p>0.050</p></blockquote></td><td><blockquote><p>0.046</p></blockquote></td><td><blockquote><p>0.548</p></blockquote></td><td><blockquote><p>0.041</p></blockquote></td><td><blockquote><p>0.039</p></blockquote></td><td><blockquote><p>0.043</p></blockquote></td><td><blockquote><p>0.092</p></blockquote></td><td><blockquote><p>0.042</p></blockquote></td><td><blockquote><p>0.050</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s8</p></blockquote></td><td><blockquote><p>0.042</p></blockquote></td><td><blockquote><p>0.037</p></blockquote></td><td><blockquote><p>0.036</p></blockquote></td><td><blockquote><p>0.053</p></blockquote></td><td><blockquote><p>0.031</p></blockquote></td><td><blockquote><p>0.054</p></blockquote></td><td><blockquote><p>0.042</p></blockquote></td><td><blockquote><p>0.069</p></blockquote></td><td><blockquote><p>0.032</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s9</p></blockquote></td><td><blockquote><p>0.050</p></blockquote></td><td><blockquote><p>0.038</p></blockquote></td><td><blockquote><p>0.038</p></blockquote></td><td><blockquote><p>0.043</p></blockquote></td><td><blockquote><p>0.034</p></blockquote></td><td><blockquote><p>0.038</p></blockquote></td><td><blockquote><p>0.043</p></blockquote></td><td><blockquote><p>0.032</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td></tr><tr class="odd"><td>SA-DDPM采样信号与BCI-IV脑电数据之间的受试者间相关系数</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr class="even"><td><blockquote><p>sl</p></blockquote></td><td><blockquote><p>0.109</p></blockquote></td><td><blockquote><p>0.071</p></blockquote></td><td><blockquote><p>0.077</p></blockquote></td><td><blockquote><p>0.080</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td><td><blockquote><p>0.073</p></blockquote></td><td><blockquote><p>0.072</p></blockquote></td><td><blockquote><p>0.063</p></blockquote></td><td><blockquote><p>0.070</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s2</p></blockquote></td><td><blockquote><p>0.073</p></blockquote></td><td><blockquote><p>0.090</p></blockquote></td><td><blockquote><p>0.067</p></blockquote></td><td><blockquote><p>0.065</p></blockquote></td><td><blockquote><p>0.059</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td><td><blockquote><p>0.068</p></blockquote></td><td><blockquote><p>0.056</p></blockquote></td><td><blockquote><p>0.061</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s3</p></blockquote></td><td><blockquote><p>0.074</p></blockquote></td><td><blockquote><p>0.065</p></blockquote></td><td><blockquote><p>0.096</p></blockquote></td><td><blockquote><p>0.072</p></blockquote></td><td><blockquote><p>0.060</p></blockquote></td><td><blockquote><p>0.069</p></blockquote></td><td><blockquote><p>0.070</p></blockquote></td><td><blockquote><p>0.060</p></blockquote></td><td><blockquote><p>0.063</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s4</p></blockquote></td><td><blockquote><p>0.077</p></blockquote></td><td><blockquote><p>0.063</p></blockquote></td><td><blockquote><p>0.073</p></blockquote></td><td><blockquote><p>0.100</p></blockquote></td><td><blockquote><p>0.063</p></blockquote></td><td><blockquote><p>0.068</p></blockquote></td><td><blockquote><p>0.074</p></blockquote></td><td><blockquote><p>0.064</p></blockquote></td><td><blockquote><p>0.064</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s5</p></blockquote></td><td><blockquote><p>0.068</p></blockquote></td><td><blockquote><p>0.060</p></blockquote></td><td><blockquote><p>0.065</p></blockquote></td><td><blockquote><p>0.067</p></blockquote></td><td><blockquote><p>0.076</p></blockquote></td><td><blockquote><p>0.062</p></blockquote></td><td><blockquote><p>0.063</p></blockquote></td><td><blockquote><p>0.055</p></blockquote></td><td><blockquote><p>0.059</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s6</p></blockquote></td><td><blockquote><p>0.083</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td><td><blockquote><p>0.069</p></blockquote></td><td><blockquote><p>0.067</p></blockquote></td><td><blockquote><p>0.060</p></blockquote></td><td><blockquote><p>0.088</p></blockquote></td><td><blockquote><p>0.068</p></blockquote></td><td><blockquote><p>0.057</p></blockquote></td><td><blockquote><p>0.061</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s7</p></blockquote></td><td><blockquote><p>0.072</p></blockquote></td><td><blockquote><p>0.069</p></blockquote></td><td><blockquote><p>0.079</p></blockquote></td><td><blockquote><p>0.075</p></blockquote></td><td><blockquote><p>0.071</p></blockquote></td><td><blockquote><p>0.069</p></blockquote></td><td><blockquote><p>0.103</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td></tr><tr class="odd"><td><blockquote><p>s8</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td><td><blockquote><p>0.064</p></blockquote></td><td><blockquote><p>0.065</p></blockquote></td><td><blockquote><p>0.069</p></blockquote></td><td><blockquote><p>0.058</p></blockquote></td><td><blockquote><p>0.062</p></blockquote></td><td><blockquote><p>0.066</p></blockquote></td><td><blockquote><p>0.082</p></blockquote></td><td><blockquote><p>0.058</p></blockquote></td></tr><tr class="even"><td><blockquote><p>s9</p></blockquote></td><td><blockquote><p>0.071</p></blockquote></td><td><blockquote><p>0.064</p></blockquote></td><td><blockquote><p>0.067</p></blockquote></td><td><blockquote><p>0.067</p></blockquote></td><td><blockquote><p>0.060</p></blockquote></td><td><blockquote><p>0.065</p></blockquote></td><td><blockquote><p>0.067</p></blockquote></td><td><blockquote><p>0.056</p></blockquote></td><td><blockquote><p>0.082</p></blockquote></td></tr></tbody></table>

为了进一步分析SADDPM生成信号对真实信号的保真程度，本文计算了真实样本与SADDPM生成样本之间的受试者相关性矩阵。图中分别展示了原始BCI-IV信号内部的受试者相关性，以及SADDPM生成样本与真实信号之间的相关性。

实验得出的结果显示，于SADDPM所生成的样本里面，同一受试者相互之间的相关性要远远高于不同受试者彼此之间的相关性，这样的一种趋势跟真实脑电信号当中受试者的分布情况是相契合的，就好比，像S1、S4、S6等这些受试者的自相关系数分别为0.109、0.100、0.103，较大高于它们跟其他主体之间的交叉相关值，SADDPM在结构方面生成了有噪声鲁棒性的信号，同时在内容方面也有效地留存了个体特征。

将SADDPM样本与真实信号内部的相关性矩阵加以对比可发现，在受试者识别一致性这方面，SADDPM样本有着较高的保真性，其生成机制学习了全局信号分布，而且还可以捕捉每个受试者特有的信号模式以及伪迹结构，有较强的受试者特征保持能力，这一特性对后续的个性化脑电解码建模有着意义。

5 结论

本研究聚焦于深度学习背景下的脑电信号去噪难题，精心设计并成功实现了一种将扩散概率模型（DDPM）与受试者感知机制相结合的去噪方案，主要针对多受试者脑电信号面临的跨主体泛化挑战，借助引入可训练的低维受试者编码向量，并在网络的不同层级实施条件注入，该模型在维持结构稳定性的可有效提升去噪性能，表现明显优于传统方法。虽然本工作已经取得了初步的成果，但是仍然有一些方向值得探讨：目前受试者的编码方式依然依赖充足的训练数据，未来可以尝试探索融合迁移学习或者少样本学习策略，以此来降低对数据的依赖程度。本研究为多被试脑电信号去噪提供了新颖且有效的建模思路，丰富了扩散模型在时序信号处理领域的应用边界，还为面向个体差异的神经信号建模提供了可行实践路径，有关键学术与应用价值。

参考文献

> \[1\] Zhang H., Zhao M., Wei C., Mantini D., Li Z., Liu Q. EEGDenoiseNet: A Benchmark Dataset for Deep Learning Solutions of EEG Denoising \[J\]. Journal of Neural Engineering, 2021, 18(5): 056057. doi:10.1088/1741-2552/ac2bf8.
>
> \[2\] Yin J., Liu A., Li C., et al. Frequency Information Enhanced Deep EEG Denoising Network for Ocular Artifact Removal \[J\]. IEEE Sensors Journal, 2022, 22(22): 21855--21867.
>
> \[3\] Chen X., Li C., Liu A., et al. Toward Open-World Electroencephalogram Decoding Via Deep Learning: A Comprehensive Survey \[J\]. IEEE Signal Processing Magazine, 2022, 39(2): 105--126.
>
> \[4\] Chiang Y.T., Li C., Zhang H. Fully Convolutional Network-Based Autoencoder for Biomedical Signal Denoising \[C\]. IEEE International Conference on Bioinformatics and Biomedicine (BIBM), November 18--21, 2019. IEEE, 2019: 1505--1510.
>
> \[5\] Yin J., Liu A., Li C., et al. A GAN Guided Parallel CNN and Transformer Network for EEG Denoising \[J\]. IEEE Journal of Biomedical and Health Informatics, 2023. doi:10.1109/JBHI.2023.3277596.
>
> \[6\] Ho J., Jain A., Abbeel P. Denoising Diffusion Probabilistic Models \[J\]. Advances in Neural Information Processing Systems, 2020, 33: 6840--6851.
>
> \[7\] Zhao L., Li H., Chen R. Enhanced Denoising of Electrocardiograms via a Denoising Diffusion Probabilistic Model Approach \[J\]. IEEE Transactions on Biomedical Engineering, 2023, 70(3): 784--795.
>
> \[8\] Pan T., Yan X., Zhang C. Medical Image Reconstruction Using Diffusion Models \[J\]. Medical Image Analysis, 2022, 81: 102528.
>
> \[9\] Lin G., Zhang J., Liu Y. Single Shot Reversible GAN for BCG Artifact Removal in Simultaneous EEG-fMRI \[J\]. arXiv, 2020. arXiv:2011.01710.
>
> \[10\] Zeng H., Li Y., Zhang L., et al. DM-RE2I: A Framework Based on Diffusion Model for the Reconstruction from EEG to Image \[J\]. Biomedical Signal Processing and Control, 2023, 86: 105125.
>
> \[11\] Ghosh R., Sinha N., Biswas S.K. Automated Eye Blink Artefact Removal from EEG Using Support Vector Machine and Autoencoder \[J\]. IET Signal Processing, 2019, 13(2): 141--148. doi:10.1049/iet-spr.2018.5111.
>
> \[12\] Pfurtscheller G., Da Silva F.L. Event-Related EEG/MEG Synchronization and Desynchronization: Basic Principles \[J\]. Clinical Neurophysiology, 1999, 110(11): 1842--1857. doi:10.1016/S1388-2457(99)00141-8.
>
> \[13\] Jiao Y., Zhang Y., Chen X., et al. Sparse Group Representation Model for Motor Imagery EEG Classification \[J\]. IEEE Journal of Biomedical and Health Informatics, 2018, 23(2): 631--641. doi:10.1109/JBHI.2018.2832538.
>
> \[14\] Azqadan E., Jahed H., Arami A. Predictive Microstructure Image Generation Using Denoising Diffusion Probabilistic Models \[J\]. Acta Materialia, 2023, 261: 119406.
>
> \[15\] Lugmayr A., Danelljan M., Romero A., et al. RePaint: Inpainting Using Denoising Diffusion Probabilistic Models \[C\]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. IEEE, 2022: 11461--11471.
>
> \[16\] Pan S., Wang T., Qiu R.L., et al. 2D Medical Image Synthesis Using Transformer-Based Denoising Diffusion Probabilistic Model \[J\]. Physics in Medicine and Biology, 2023, 68(10): 105004.
>
> \[17\] Choi J., Kim S., Jeong Y., et al. ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models \[J\]. arXiv, 2021. arXiv:2108.02938.
>
> \[18\] Goldberger A.L., Amaral L.A.N., Glass L., et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals \[J\]. Circulation, 2000, 101(23): e215--e220.
>
> \[19\] Hartmann K.G., Schirrmeister R.T., Ball T. EEG-GAN: Generative Adversarial Networks for Electroencephalographic (EEG) Brain Signals \[J\]. arXiv, 2018. arXiv:1806.01875.
>
> \[20\] Lugmayr A., Danelljan M., Timofte R. Guided Diffusion Models for MRI Image Denoising \[J\]. IEEE Transactions on Medical Imaging, 2022, 41(4): 1041--1054.
>
> \[21\] An Y, Lam H K, Ling S H. Auto-denoising for EEG signals using generative adversarial network\[J\]. Sensors, 2022, 22(5): 1750. DOI: [10.3390/s22051750](https://doi.org/10.3390/s22051750)
