# Awesome-SOTA-FER

A curated list of facial expression recognition in both 7-emotion classification and affect estimation. In addition, this repository includes basic studies on FER and recent datasets.

👀 Update frequently!


## 🛎 Notice
⭐️: __Importance__ of FER method (from 1 to 3 points)

🏆: __Mandatory (research) paper__ to understand FER technology.


## 📋 What's New

- [Dec. 2024] Add ArXiv papers.
- [Nov. 2024] Add ECCV 2024 accepted papers.
- [Sep. 2024] Add Challenge and emotion-related papers.
- [Jul. 2024] Add Conference and Journal papers.
- [Jun. 2024] Add some ArXiv and conference accepted papers.
- [Apr. 2024] Make `Multi-modal, EEG-based Emotion Recognition` section. Update papers.
- [Mar. 2024] Update FG papers.
- [Feb. 2024] Update AAAI and ArXiv papers.
- [Jan. 2024] Update few ArXiv papers.
- [Nov. 2023] Update NeurIPS 2023 / ArXiv papers.
- [Oct. 2023] Update ACII 2023 paper / Fix details.
- [Sep. 2023] Update BMVC 2023 paper / Add `Dataset` section.
- [Aug. 2023] Update ICCV 2023 / ACM MM 2023 papers.
- [Jul. 2023] Update AAAI 2023 paper.
- [Jun. 2023] Update ICML 2023 / ICASSP 2023 papers.
- [Apr. 2023] Update CVPR 2023 papers.
- [Mar. 2023] Initial update for FER papers.
- [Mar. 2023] Create awesome-SOTA-FER repository.

## 👥 Contributing
Please feel free to refer this repository for your FER research/development and send me [pull requests](https://github.com/kdhht2334/awesome-SOTA-FER/pulls) or email [Daeha Kim](kdhht5022@gmail.com) to add links.


## Table of Contents

- [7-Emotion Classification](#seven-emotion) <a id="seven-emotion"></a>
  - [2024](#2024-c)
  - [2023](#2023-c)
  - [2022](#2022-c)
  - [2021](#2021-c)
  - [2020](#2020-c)

- [Valence-arousal Affect Estimation and Analysis](#affect)

- [Facial Action Unit (AU) Detection (or Recognition)](#au)

- [About Facial Privacy](#privacy)

- [Multi-modal, EEG-based Emotion Recognition](#mm-er)
    
- [Facial Expression Manipulation and Synthesis](#fem)
    
- [Emotion Recognition, Facial Representations, and Others](#er-fr-o)

- [Datasets](#datasets)

- [Challenges](#challenges)
  
- [Tools](#tools)

- [Remarkable Papers (2019~)](#previous)



## 7-Emotion Classification <a id="seven-emotion"></a>

#### 2024 <a id="2024-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [Generalizable Facial Expression Recognition](https://arxiv.org/pdf/2408.10614) | ECCV | ⭐️⭐️ | [GitHub](https://github.com/zyh-uaiaaaa/Generalizable-FER) | 
| [Norface: Improving Facial Expression Analysis by Identity Normalization](https://arxiv.org/pdf/2407.15617) | ECCV | ⭐️⭐️ | [Project](https://norface-fea.github.io/) |
| [Bridging the Gaps: Utilizing Unlabeled Face Recognition Datasets to Boost Semi-Supervised Facial Expression Recognition](https://arxiv.org/pdf/2410.17622) | ArXiv | ⭐️ | [PyTorch](https://github.com/zhelishisongjie/SSFER) |
| [SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](https://arxiv.org/pdf/2410.09865) | ArXiv | ⭐️⭐️ | N/A |
| [ExpLLM: Towards Chain of Thought for Facial Expression Recognition](https://arxiv.org/pdf/2409.02828) | ArXiv | ⭐️⭐️ | [Site](https://starhiking.github.io/ExpLLM_Page/)
| [A Survey on Facial Expression Recognition of Static and Dynamic Emotions](https://arxiv.org/pdf/2408.15777) | TPAMI | ⭐️⭐️⭐️ | [GitHub](https://github.com/wangyanckxx/SurveyFER) |
| [Rethinking Affect Analysis: A Protocol for Ensuring Fairness and Consistency](https://arxiv.org/pdf/2408.02164) | ArXiv | 🏆 | N/A |
| [LLDif: Diffusion Models for Low-light Emotion Recognition](https://arxiv.org/pdf/2408.04235) | ICPR | ⭐️⭐️ | N/A |
| [From Macro to Micro: Boosting Micro-Expression Recognition via Pre-Training on Macro-Expression Videos](https://arxiv.org/pdf/2405.16451) | AriX | ⭐️ | N/A |
| [FacePhi: Light-Weight Multi-Modal Large Language Model for Facial Landmark Emotion Recognition](https://openreview.net/pdf?id=7Tf9JTGeLU) | ICLRW | ⭐️⭐️⭐️ | N/A |
| [Enhancing Zero-Shot Facial Expression Recognition by LLM Knowledge Transfer](https://arxiv.org/pdf/2405.19100) | ArXiv | ⭐️⭐️ | [PyTorch](https://github.com/zengqunzhao/Exp-CLIP) |
| [MSSTNET: A Multi-Scale Spatio-Temporal CNN-Transformer Network for Dynamic Facial Expression Recognition](https://arxiv.org/pdf/2404.08433) | ICASSP | ⭐️ | N/A |
| [Open-Set Facial Expression Recognition](https://arxiv.org/pdf/2401.12507.pdf) | AAAI | ⭐️⭐️⭐️ | [GitHub](https://github.com/zyh-uaiaaaa) |
| [MIDAS: Mixing Ambiguous Data with Soft Labels for Dynamic Facial Expression Recognition](https://openaccess.thecvf.com/content/WACV2024/papers/Kawamura_MIDAS_Mixing_Ambiguous_Data_With_Soft_Labels_for_Dynamic_Facial_WACV_2024_paper.pdf) | WACV | ⭐️ | N/A |
| [Hard Sample-aware Consistency for Low-resolution Facial Expression Recognition](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Hard_Sample-Aware_Consistency_for_Low-Resolution_Facial_Expression_Recognition_WACV_2024_paper.pdf) | WACV | ⭐️ | N/A |

#### 2023 <a id="2023-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition](https://arxiv.org/pdf/2204.04083.pdf) |  ICCV Workshop (AMFG) | ⭐️⭐️ | [Github](https://github.com/zczcwh/POSTER) |
| [GaFET: Learning Geometry-aware Facial Expression Translation from In-The-Wild Images](https://arxiv.org/pdf/2308.03413v1.pdf) | ICCV | ⭐️⭐️⭐️ | N/A |
| [LA-Net: Landmark-Aware Learning for Reliable Facial Expression Recognition under Label Noise](https://arxiv.org/pdf/2307.09023.pdf) | ICCV | ⭐️ | N/A |
| [Latent-OFER: Detect, Mask, and Reconstruct with Latent Vectors for Occluded Facial Expression Recognition](https://arxiv.org/pdf/2307.11404.pdf) | ICCV | ⭐️⭐️ | [PyTorch](https://github.com/leeisack/Latent-OFER) |
| [Prompting Visual-Language Models for Dynamic Facial Expression Recognition](https://arxiv.org/pdf/2308.13382.pdf) | BMVC | ⭐️⭐️⭐️ | [PyTorch](https://github.com/zengqunzhao/DFER-CLIP) |
| [MAE-DFER: Efficient Masked Autoencoder for Self-supervised Dynamic Facial Expression Recognition](https://arxiv.org/pdf/2307.02227.pdf) | ACM MM | ⭐️⭐️⭐️ | [PyTorch](https://github.com/sunlicai/MAE-DFER) |
| [Addressing Racial Bias in Facial Emotion Recognition](https://arxiv.org/pdf/2308.04674.pdf) | ArXiv | ⭐️ | N/A |
| [Learning Deep Hierarchical Features with Spatial Regularization for One-Class Facial Expression Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/25749) | AAAI | ⭐️ | N/A |
| [Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction](https://openreview.net/pdf?id=DH11pt7S2t) | ICML | ⭐️⭐️ | N/A |
| [Feature Representation Learning with Adaptive Displacement Generation and Transformer Fusion for Micro-Expression Recognition](https://arxiv.org/pdf/2304.04420.pdf) | CVPR | ⭐️⭐️ | N/A |
| [Rethinking the Learning Paradigm for Facial Expression Recognition](https://arxiv.org/pdf/2209.15402.pdf) | CVPR | ⭐️⭐️⭐️ | N/A |
| [Multi-Domain Norm-Referenced Encoding enables Data Efficient Transformer Learning of Facial Expression Recognition](https://arxiv.org/pdf/2304.02309v1.pdf) | ArXiv | ⭐️⭐️ | N/A |
| [Uncertainty-aware Label Distribution Learning for Facial Expression Recognition](https://openaccess.thecvf.com/content/WACV2023/papers/Le_Uncertainty-Aware_Label_Distribution_Learning_for_Facial_Expression_Recognition_WACV_2023_paper.pdf) | WACV | ⭐️⭐️ | [TensorFlow](https://github.com/minhnhatvt/label-distribution-learning-fer-tf/) |
| [POSTER V2: A simpler and stronger facial expression recognition network](https://arxiv.org/pdf/2301.12149.pdf) | ArXiv | ⭐️⭐️ | [PyTorch](https://github.com/Talented-Q/POSTER_V2) |


#### 2022 <a id="2022-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [Learn From All: Erasing Attention Consistency for Noisy Label Facial Expression Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860406.pdf) | ECCV | ⭐️⭐️⭐️ | [PyTorch](https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency) |
| [Learn-to-Decompose: Cascaded Decomposition Network for Cross-Domain Few-Shot Facial Expression Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790672.pdf) | ECCV | ⭐️ | [PyTorch](https://github.com/zouxinyi0625/CDNet) |
| [Teaching with Soft Label Smoothing for Mitigating Noisy Labels in Facial Expressions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720639.pdf) | ECCV | ⭐️ | [PyTorch](https://github.com/toharl/soft) |
| [Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Semi-Supervised_Deep_Facial_Expression_Recognition_With_an_Adaptive_Confidence_CVPR_2022_paper.pdf) | CVPR | ⭐️ | [PyTorch](https://github.com/hangyu94/Ada-CM/) |
| [Face2Exp: Combating Data Biases for Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Zeng_Face2Exp_Combating_Data_Biases_for_Facial_Expression_Recognition_CVPR_2022_paper.pdf) | CVPR | ⭐️⭐️⭐️ | [PyTorch](https://github.com/danzeng1990/Face2Exp) |
| [Facial Expression Recognition By Using a Disentangled Identity Invariant Expression Representation](https://ailb-web.ing.unimore.it/icpr/media/slides/12024.pdf) [[Poster]](https://ailb-web.ing.unimore.it/icpr/media/posters/12024.pdf) | ICPR | ⭐️⭐️ | N/A |
| [Vision Transformer Equipped with Neural Resizer on Facial Expression Recognition Task](https://arxiv.org/pdf/2204.02181) | ICASSP | ⭐️ | [PyTorch](https://github.com/hbin0701/VT_with_NR_for_FER) |
| [A Prototype-Oriented Contrastive Adaption Network For Cross domain Facial Expression Recognition](https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_A_Prototype-Oriented_Contrastive_Adaption_Network_For_Cross-domain_Facial_Expression_Recognition_ACCV_2022_paper.pdf) | ACCV | ⭐️ | N/A |
| [Soft Label Mining and Average Expression Anchoring for Facial Expression Recognition](https://openaccess.thecvf.com/content/ACCV2022/papers/Ming_Soft_Label_Mining_and_Average_Expression_Anchoring_for_Facial_Expression_ACCV_2022_paper.pdf) | ACCV | ⭐️ | [PyTorch](https://github.com/HaipengMing/SLM-AEA) |
| [Revisiting Self-Supervised Contrastive Learning for Facial Expression Recognition](https://arxiv.org/pdf/2210.03853.pdf) | BMVC | ⭐️ | [Site](https://claudiashu.github.io/SSLFER/) |
| [Cluster-level pseudo-labelling for source-free cross-domain facial expression recognition](https://arxiv.org/pdf/2210.05246.pdf) | BMVC | ⭐️⭐️ | [PyTorch](https://github.com/altndrr/clup) |
| [Analysis of Semi-Supervised Methods for Facial Expression Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9953876&casa_token=RptxgdB8SrEAAAAA:maVFJFxUaED8NbBbIFLjn4HCTNqQhXR6dSXN1NgWaUkQgXimVX9-8WLx-Ob9Zk0i4GL5eCEAeA) | ACII | ⭐️ | [GitHub](https://github.com/ShuvenduRoy/SSL) |
| [POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition](https://arxiv.org/pdf/2204.04083.pdf) | ArXiv | 2022 | ⭐️ | [GitHub](https://github.com/zczcwh/POSTER) |

#### 2021 <a id="2021-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [TransFER: Learning Relation-aware Facial Expression Representations with Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Xue_TransFER_Learning_Relation-Aware_Facial_Expression_Representations_With_Transformers_ICCV_2021_paper.pdf) | ICCV | ⭐️⭐️ | N/A |
| [Understanding and Mitigating Annotation Bias in Facial Expression Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Understanding_and_Mitigating_Annotation_Bias_in_Facial_Expression_Recognition_ICCV_2021_paper.pdf) | ICCV | ⭐️ | N/A |
| [Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty Estimation for Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/She_Dive_Into_Ambiguity_Latent_Distribution_Mining_and_Pairwise_Uncertainty_Estimation_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️⭐️ | [PyTorch](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DMUE) |
| [Affective Processes: stochastic modelling of temporal context for emotion and facial expression recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Sanchez_Affective_Processes_Stochastic_Modelling_of_Temporal_Context_for_Emotion_and_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️ | N/A |
| [Feature Decomposition and Reconstruction Learning for Effective Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Ruan_Feature_Decomposition_and_Reconstruction_Learning_for_Effective_Facial_Expression_Recognition_CVPR_2021_paper.pdf) | CVPR | ⭐️ | N/A |
| [Learning a Facial Expression Embedding Disentangled from Identity](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_a_Facial_Expression_Embedding_Disentangled_From_Identity_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️⭐️ | N/A |
| [Affect2MM: Affective Analysis of Multimedia Content Using Emotion Causality](https://openaccess.thecvf.com/content/CVPR2021/papers/Mittal_Affect2MM_Affective_Analysis_of_Multimedia_Content_Using_Emotion_Causality_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️⭐️ | [Site](https://gamma.umd.edu/researchdirections/affectivecomputing/emotionrecognition/affect2mm/) |
| [A Circular-Structured Representation for Visual Emotion Distribution Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_A_Circular-Structured_Representation_for_Visual_Emotion_Distribution_Learning_CVPR_2021_paper.pdf) | CVPR | ⭐️ | N/A |
| [Temporal Stochastic Softmax for 3D CNNs: An Application in Facial Expression Recognition](https://openaccess.thecvf.com/content/WACV2021/papers/Ayral_Temporal_Stochastic_Softmax_for_3D_CNNs_An_Application_in_Facial_WACV_2021_paper.pdf) | WACV | ⭐️ | [PyTorch](https://github.com/thayral/temporal-stochastic-softmax) |
| [Facial Expression Recognition in the Wild via Deep Attentive Center Loss](https://openaccess.thecvf.com/content/WACV2021/papers/Farzaneh_Facial_Expression_Recognition_in_the_Wild_via_Deep_Attentive_Center_WACV_2021_paper.pdf) | WACV | ⭐️⭐️ | [PyTorch](https://github.com/amirhfarzaneh/dacl) |
| [Identity-Aware Facial Expression Recognition Via Deep Metric Learning Based on Synthesized Images](https://ieeexplore.ieee.org/document/9479695) | IEEE TMM | ⭐️ | N/A |
| [Relative Uncertainty Learning for Facial Expression Recognition](https://proceedings.neurips.cc/paper/2021/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf) | NeurIPS | ⭐️⭐️⭐️ | [PyTorch](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning) |
| [Identity-Free Facial Expression Recognition using conditional Generative Adversarial Network](https://arxiv.org/pdf/1903.08051) | ICIP | ⭐️ | N/A | 
| [Domain Generalisation for Apparent Emotional Facial Expression Recognition across Age-Groups](https://arxiv.org/pdf/2110.09168) | Tech.</br>report | ⭐️⭐️⭐️ | N/A |



#### 2020 <a id="2020-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [Suppressing Uncertainties for Large-Scale Facial Expression Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Suppressing_Uncertainties_for_Large-Scale_Facial_Expression_Recognition_CVPR_2020_paper.pdf) | CVPR | ⭐️⭐️ | [PyTorch](https://github.com/kaiwang960112/Self-Cure-Network) |
| [Label Distribution Learning on Auxiliary Label Space Graphs for Facial Expression Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Label_Distribution_Learning_on_Auxiliary_Label_Space_Graphs_for_Facial_CVPR_2020_paper.pdf) | CVPR | ⭐️ | N/A |
| [Graph Neural Networks for Image Understanding Based on Multiple Cues: Group Emotion Recognition and Event Recognition as Use Cases](https://openaccess.thecvf.com/content_WACV_2020/papers/Guo_Graph_Neural_Networks_for_Image_Understanding_Based_on_Multiple_Cues_WACV_2020_paper.pdf) | WACV | ⭐️ | [TensorFlow](https://github.com/gxstudy/Graph-Neural-Networks-for-Image-Understanding-Based-on-Multiple-Cues) |
| [Detecting Face2Face Facial Reenactment in Videos](https://openaccess.thecvf.com/content_WACV_2020/papers/Kumar_Detecting_Face2Face_Facial_Reenactment_in_Videos_WACV_2020_paper.pdf) | WACV | ⭐️ | N/A |



### Valence-arousal Affect Estimation and Analysis [back-to-top](#seven-emotion) <a id="affect"></a>

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Ig3D: Integrating 3D Face Representations in Facial Expression Inference](https://arxiv.org/pdf/2408.16907) | ECCVW | 2024 | ⭐️⭐️ | [Site](https://dongludeeplearning.github.io/Ig3D.html) |
| [Detail-Enhanced Intra- and Inter-modal Interaction for Audio-Visual Emotion Recognition](https://arxiv.org/pdf/2405.16701) | ICPR | 2024 | ⭐️ | N/A |
| [Bridging the Gap: Protocol Towards Fair and Consistent Affect Analysis](https://arxiv.org/pdf/2405.06841) | FG | 2024 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/dkollias/Fair-Consistent-Affect-Analysis) |
| [Inconsistency-Aware Cross-Attention for Audio-Visual Fusion in Dimensional Emotion Recognition](https://arxiv.org/pdf/2405.12853) | ArXiv | 2024 | ⭐️ | N/A |
| [CAGE: Circumplex Affect Guided Expression Inference](https://arxiv.org/pdf/2404.14975) | CVPRW | 2024 | ⭐️⭐️ | [PyTorch](https://github.com/wagner-niklas/CAGE_expression_inference) |
| [Cross-Attention is Not Always Needed: Dynamic Cross-Attention for Audio-Visual Dimensional Emotion Recognition](https://arxiv.org/pdf/2403.19554) | ICME | 2024 | ⭐️⭐️ | N/A |
| [3DEmo: for Portrait Emotion Recognition with New Dataset](https://dl.acm.org/doi/pdf/10.1145/3631133) | ACM Journal on Computing and Cultural Heritage | 2023 | ⭐️⭐️ | None |
| [Optimal Transport-based Identity Matching for Identity-invariant Facial Expression Recognition](https://arxiv.org/pdf/2209.12172.pdf) | NeurIPS</br>(spotlight) | 2022 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/kdhht2334/ELIM_FER) |
| [Emotion-aware Multi-view Contrastive Learning for Faciel Emotion Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730181.pdf) | ECCV | 2022 | ⭐️ | [PyTorch](https://github.com/kdhht2334/AVCE_FER) |
| [Learning from Label Relationships in Human Affect](https://arxiv.org/pdf/2207.05577.pdf) | ACM MM | 2022 | ⭐️ | N/A |
| [Are 3D Face Shapes Expressive Enough for Recognising Continuous Emotions and Action Unit Intensities?](https://arxiv.org/pdf/2207.01113.pdf) | ArXiv | 2022 | ⭐️⭐️ | N/A |
| [Contrastive Adversarial Learning for Person Independent Facial Emotion Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/16743) | AAAI | 2021 | ⭐️ | [PyTorch](https://github.com/kdhht2334/Contrastive-Adversarial-Learning-FER) |
| [Estimating continuous affect with label uncertainty](https://ieeexplore.ieee.org/iel7/9597394/9597388/09597425.pdf?casa_token=0QyPEEV_UKwAAAAA:vXZLYYOLqO3kXHCwSzdYx8tAyABtJ-gzK6VBk79HUlNHzZK0__gLA7TdQd2AnrCyeKtUnCqoUg) | ACII | 2021 | ⭐️ | N/A |
| [Factorized Higher-Order CNNs with an Application to Spatio-Temporal Emotion Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kossaifi_Factorized_Higher-Order_CNNs_With_an_Application_to_Spatio-Temporal_Emotion_Estimation_CVPR_2020_paper.pdf) | CVPR | 2020 | ⭐️⭐️ | N/A |
| [BReG-NeXt: Facial affect computing using adaptive residual networks with bounded gradient](https://ieeexplore.ieee.org/iel7/5165369/5520654/09064942.pdf?casa_token=mmeTU4Eqv6IAAAAA:m6rR6FVdNZhuuxcTA5G8z5j2h28hnm5zE3FtgkEJOeXUfT818fM501SzCIoJ_aHmx6yGWHE78w) | IEEE TAC | 2020 | ⭐️⭐️ | [TensorFlow](https://github.com/behzadhsni/BReG-NeXt) |


### Facial Action Unit (AU) Detection (or Recognition) [back-to-top](#seven-emotion) <a id="au"></a>

  - This section also contains `landmark` detection.

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Trend-Aware Supervision: On Learning Invariance for Semi-supervised Facial Action Unit Intensity Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/27803) | AAAI | 2024 | ⭐️⭐️ | N/A |
| [FAN-Trans: Online Knowledge Distillation for Facial Action Unit Detection](https://openaccess.thecvf.com/content/WACV2023/papers/Yang_FAN-Trans_Online_Knowledge_Distillation_for_Facial_Action_Unit_Detection_WACV_2023_paper.pdf) | WACV | 2023 | ⭐️ | N/A |
| [Knowledge-Driven Self-Supervised Representation Learning for Facial Action Unit Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_Knowledge-Driven_Self-Supervised_Representation_Learning_for_Facial_Action_Unit_Recognition_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | N/A |
| [Towards Accurate Facial Landmark Detection via Cascaded Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Accurate_Facial_Landmark_Detection_via_Cascaded_Transformers_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | N/A |
| [Causal intervention for subject-deconfounded facial action unit recognition](https://ojs.aaai.org/index.php/AAAI/article/view/19914/19673) | AAAI</br>(Oral) | 2022 | ⭐️⭐️⭐️ | N/A |
| [PIAP-DF: Pixel-Interested and Anti Person-Specific Facial Action Unit Detection Net with Discrete Feedback Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_PIAP-DF_Pixel-Interested_and_Anti_Person-Specific_Facial_Action_Unit_Detection_Net_ICCV_2021_paper.pdf) | ICCV | 2021 | ⭐️ | N/A |


### About Facial Privacy [back-to-top](#seven-emotion) <a id="privacy"></a>

  - __Note__: Some face recognition studies are included.
  - You can get additional information [here](https://github.com/Tencent/TFace)

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Anonymization Prompt Learning for Facial Privacy-Preserving Text-to-Image Generation](https://arxiv.org/pdf/2405.16895) | ArXiv | 2024 | ⭐️⭐️ | Inside Paper |
| [ϵ-Mesh Attack: A Surface-based Adversarial Point Cloud Attack for Facial Expression Recognition](https://arxiv.org/pdf/2403.06661v1.pdf) | FG | 2024 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/batuceng/e-mesh-attack) |
| [Walk as you feel: Privacy preserving emotion recognition from gait patterns](https://www.sciencedirect.com/science/article/pii/S0952197623017499) | EAAI (ELSEVIER) | 2023 | ⭐️ | N/A |
| [GANonymization: A GAN-based Face Anonymization Framework for Preserving Emotional Expressions](https://arxiv.org/pdf/2305.02143.pdf) | ACM Transactions on MCCA | 2023 | ⭐️⭐️ | N/A |
| [Simulated adversarial testing of face recognition models](https://openaccess.thecvf.com/content/CVPR2022/papers/Ruiz_Simulated_Adversarial_Testing_of_Face_Recognition_Models_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | N/A |
| [Exploring frequency adversarial attacks for face forgery detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Jia_Exploring_Frequency_Adversarial_Attacks_for_Face_Forgery_Detection_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | N/A |
| [Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Protecting_Facial_Privacy_Generating_Adversarial_Identity_Masks_via_Style-Robust_Makeup_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/CGCL-codes/AMT-GAN) |
| [Privacy-Preserving Face Recognition with Learnable Privacy Budgets in Frequency Domain](https://arxiv.org/pdf/2207.07316.pdf) | ECCV | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/Tencent/TFace/tree/master/recognition/tasks/dctdp) |
| [DuetFace: Collaborative Privacy-Preserving Face Recognition via Channel Splitting in the Frequency Domain](https://dl.acm.org/doi/abs/10.1145/3503161.3548303) | ACM MM | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/Tencent/TFace/tree/master/recognition/tasks/duetface) |
| [AdverFacial: Privacy-Preserving Universal Adversarial Perturbation Against Facial Micro-Expression Leakages](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9746848&casa_token=8uyq2J9NQV0AAAAA:qfLmtAMVlhrEv7AX76W8tkVadwZUo1ZDZuTjWB4FOjCwpUf0qRaSg6LshZ-1AYmF_JMr6_bRBQ) | ICASSP | 2022 | ⭐️ | N/A |
| [Lie to me: shield your emotions from prying software](https://www.mdpi.com/1424-8220/22/3/967/pdf) | Sensors | 2022 | ⭐️ | N/A |
| [Point adversarial self-mining: A simple method for facial expression recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9471014&casa_token=iZsit6xNvL8AAAAA:qFwyHceOIXr5NO9bxZODojTiJy63zkjuELqUyAdsLvdDoxcVvdaqpBex7bC-42FW8oPzSd-mMA) | IEEE Transactions on Cybernetics | 2022 | ⭐️ | N/A |
| [Improving transferability of adversarial patches on face recognition with generative models](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_Improving_Transferability_of_Adversarial_Patches_on_Face_Recognition_With_Generative_CVPR_2021_paper.pdf) | CVPR | 2021 | ⭐️ | N/A |
| [Towards face encryption by generating adversarial identity masks](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Towards_Face_Encryption_by_Generating_Adversarial_Identity_Masks_ICCV_2021_paper.pdf) | ICCV | 2021 | ⭐️⭐️ | [PyTorch](https://github.com/ShawnXYang/TIP-IM) |
| [Disentangled Representation with Dual-stage Feature Learning for Face Anti-spoofing](https://openaccess.thecvf.com/content/WACV2022/papers/Wang_Disentangled_Representation_With_Dual-Stage_Feature_Learning_for_Face_Anti-Spoofing_WACV_2022_paper.pdf) | WACV | 2021 | ⭐️ | N/A |


### Multi-modal, EEG-based Emotion Recognition [back-to-top](#seven-emotion) <a id="mm-er"></a>
| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [A Comprehensive Survey on EEG-Based Emotion Recognition: A Graph-Based Perspective](https://arxiv.org/pdf/2408.06027) | ArXiv | 2024 | ⭐️⭐️⭐️ | N/A |
| [Beyond Mimicking Under-Represented Emotions: Deep Data Augmentation with Emotional Subspace Constraints for EEG-Based Emotion Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/28891) | AAAI | 2024 | ⭐️⭐️ | N/A |
| [A Brain-Inspired Way of Reducing the Network Complexity via Concept-Regularized Coding for Emotion Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/27811) | AAAI | 2024 | ⭐️⭐️ | [PyTorch](https://github.com/hanluyt/emotion-conceptual-knowledge) |


### Facial Expression Manipulation and Synthesis [back-to-top](#seven-emotion) <a id="fem"></a>

  - Uprising `Talking face generation task` is included here!

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [All You Need is Your Voice: Emotional Face Representation with Audio Perspective for Emotional Talking Face Generation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08116.pdf) | ECCV | 2024 | ⭐️⭐️ | [GitHub](https://github.com/sbde500/EAP) |
| [AnimateMe: 4D Facial Expressions via Diffusion Models](https://arxiv.org/pdf/2403.17213) | ECCV | 2024 | ⭐️⭐️ | N/A |
| [Towards Localized Fine-Grained Control for Facial Expression Generation](https://arxiv.org/pdf/2407.20175) | ArXiv | 2024 | ⭐️⭐️⭐️ | [GitHub](https://github.com/tvaranka/fineface) |
| [3D Facial Expressions through Analysis-by-Neural-Synthesis](https://arxiv.org/pdf/2404.04104) | CVPR | 2024 | ⭐️⭐️⭐️ | [Project](https://georgeretsi.github.io/smirk/) |
| [Learning Adaptive Spatial Coherent Correlations for Speech-Preserving Facial Expression Manipulation](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Learning_Adaptive_Spatial_Coherent_Correlations_for_Speech-Preserving_Facial_Expression_Manipulation_CVPR_2024_paper.pdf) | CVPR | 2024 | ⭐️⭐️ | [PyTorch](https://github.com/jianmanlincjx/ASCCL) |
| [FSRT: Facial Scene Representation Transformer for Face Reenactment from Factorized Appearance, Head-pose, and Facial Expression Features](https://arxiv.org/pdf/2404.09736) | CVPR | 2024 | ⭐️⭐️ | [Project](https://andrerochow.github.io/fsrt/) |
| [FlowVQTalker: High-Quality Emotional Talking Face Generation through Normalizing Flow and Quantization](https://arxiv.org/pdf/2403.06375) | CVPR | 2024 | ⭐️⭐️ | N/A |
| [EMOPortraits: Emotion-enhanced Multimodal One-shot Head Avatars](https://arxiv.org/pdf/2404.19110) | CVPR | 2024 | ⭐️⭐️ | [Project](https://neeek2303.github.io/EMOPortraits/) |
| [FG-EmoTalk: Talking Head Video Generation with Fine-Grained Controllable Facial Expressions](https://ojs.aaai.org/index.php/AAAI/article/view/28309) | AAAI | 2024 | ⭐️⭐️ | N/A |
| [EmoStyle: One-Shot Facial Expression Editing Using Continuous Emotion Parameters](https://openaccess.thecvf.com/content/WACV2024/papers/Azari_EmoStyle_One-Shot_Facial_Expression_Editing_Using_Continuous_Emotion_Parameters_WACV_2024_paper.pdf) | WACV | 2024 | ⭐️⭐️⭐️ | [Project](https://bihamta.github.io/emostyle/) |
| [EmoTalk: Speech-Driven Emotional Disentanglement for 3D Face Animation](https://browse.arxiv.org/pdf/2303.11089.pdf) | ICCV | 2023 | ⭐️⭐️⭐️ | [Project](https://ziqiaopeng.github.io/emotalk/) |
| [EMMN: Emotional Motion Memory Network for Audio-driven Emotional Talking Face Generation](https://openaccess.thecvf.com/content/ICCV2023/papers/Tan_EMMN_Emotional_Motion_Memory_Network_for_Audio-driven_Emotional_Talking_Face_ICCV_2023_paper.pdf) | ICCV | 2023 | ⭐️⭐️ | None |
| [Efficient Emotional Adaptation for Audio-Driven Talking-Head Generation](https://arxiv.org/pdf/2309.04946.pdf) | ICCV | 2023 | ⭐️⭐️ | [Project](https://yuangan.github.io/eat/) |
| [DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions](https://arxiv.org/pdf/2303.07697.pdf) | ICASSP | 2023 | ⭐️ | [Project](https://deepbrainai-research.github.io/discohead/) |
| [Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert](https://github.com/Sxjdwang/TalkLip) | CVPR | 2023 | ⭐️⭐️ | [GitHub](https://github.com/Sxjdwang/TalkLip) |
| [LipFormer: High-fidelity and Generalizable Talking Face Generation with A Pre-learned Facial Codebook](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_LipFormer_High-Fidelity_and_Generalizable_Talking_Face_Generation_With_a_Pre-Learned_CVPR_2023_paper.pdf) | CVPR | 2023 | ⭐️ | N/A |
| [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_SadTalker_Learning_Realistic_3D_Motion_Coefficients_for_Stylized_Audio-Driven_Single_CVPR_2023_paper.pdf) | CVPR | 2023 | 🏆 | [GitHub](https://github.com/OpenTalker/SadTalker) |
| [Identity-Preserving Talking Face Generation with Landmark and Appearance Priors](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhong_Identity-Preserving_Talking_Face_Generation_With_Landmark_and_Appearance_Priors_CVPR_2023_paper.pdf) | CVPR | 2023 | ⭐️ | N/A |
| [OTAvatar : One-shot Talking Face Avatar with Controllable Tri-plane Rendering](https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_OTAvatar_One-Shot_Talking_Face_Avatar_With_Controllable_Tri-Plane_Rendering_CVPR_2023_paper.pdf) | CVPR | 2023 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/theEricMa/OTAvatar) |
| [High-fidelity Generalized Emotional Talking Face Generation with Multi-modal Emotion Space Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_High-Fidelity_Generalized_Emotional_Talking_Face_Generation_With_Multi-Modal_Emotion_Space_CVPR_2023_paper.pdf) | CVPR | 2023 | ⭐️⭐️⭐️ | N/A |
| [4D Facial Expression Diffusion Model](https://arxiv.org/pdf/2303.16611v1.pdf) | ArXiv | 2023 | ⭐️⭐️⭐️ | [Github](https://github.com/ZOUKaifeng/4DFM) |
| [EMOCA: Emotion Driven Monocular Face Capture and Animation](https://openaccess.thecvf.com/content/CVPR2022/papers/Danecek_EMOCA_Emotion_Driven_Monocular_Face_Capture_and_Animation_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [Project](https://emoca.is.tue.mpg.de/) |
| [Sparse to Dense Dynamic 3D Facial Expression Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Otberdout_Sparse_to_Dense_Dynamic_3D_Facial_Expression_Generation_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | [GitHub](https://github.com/CRISTAL-3DSAM/Sparse2Dense) |
| [Neural Emotion Director: Speech-preserving semantic control of facial expressions in “in-the-wild” videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Papantoniou_Neural_Emotion_Director_Speech-Preserving_Semantic_Control_of_Facial_Expressions_in_CVPR_2022_paper.pdf) | CVPR</br>(best paper finalist) | 2022 | ⭐️⭐️⭐️ | [Site](https://foivospar.github.io/NED/) |
| [TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_TransEditor_Transformer-Based_Dual-Space_GAN_for_Highly_Controllable_Facial_Editing_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/BillyXYB/TransEditor) |
| [EAMM: One-Shot Emotional Talking Face via Audio-Based Emotion-Aware Motion Model](https://browse.arxiv.org/pdf/2205.15278.pdf) | SIGGRAPH | 2022 | ⭐️⭐️ | [Project](https://jixinya.github.io/projects/EAMM/) |
| [Information Bottlenecked Variational Autoencoder for Disentangled 3D Facial Expression Modelling](https://openaccess.thecvf.com/content/WACV2022/papers/Sun_Information_Bottlenecked_Variational_Autoencoder_for_Disentangled_3D_Facial_Expression_Modelling_WACV_2022_paper.pdf) | WACV | 2022 | ⭐️ | N/A |
| [Detection and Localization of Facial Expression Manipulations](https://openaccess.thecvf.com/content/WACV2022/papers/Mazaheri_Detection_and_Localization_of_Facial_Expression_Manipulations_WACV_2022_paper.pdf) | WACV | 2022 | ⭐️ | N/A |
| [Learning an Animatable Detailed 3D Face Model from In-The-Wild Images](https://arxiv.org/pdf/2012.04012) | SIGGRAPH | 2021 | ⭐️⭐️⭐️ | [Project](https://deca.is.tue.mpg.de/) |
| [Talk-to-Edit: Fine-Grained Facial Editing via Dialog](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf) | ICCV | 2021 | ⭐️⭐️ | [PyTorch](https://github.com/yumingj/Talk-to-Edit) |
| [Audio-Driven Emotional Video Portraits](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Audio-Driven_Emotional_Video_Portraits_CVPR_2021_paper.pdf) | CVPR | 2021 | ⭐️ | [Site](https://jixinya.github.io/projects/evp/) |
| [GANmut: Learning Interpretable Conditional Space for Gamut of Emotions](https://openaccess.thecvf.com/content/CVPR2021/papers/dApolito_GANmut_Learning_Interpretable_Conditional_Space_for_Gamut_of_Emotions_CVPR_2021_paper.pdf) | CVPR | 2021 | ⭐️⭐️ | [PyTorch](https://github.com/stefanodapolito/GANmut) |
| [3D Dense Geometry-Guided Facial Expression Synthesis by Adversarial Learning](https://openaccess.thecvf.com/content/WACV2021/papers/Bodur_3D_Dense_Geometry-Guided_Facial_Expression_Synthesis_by_Adversarial_Learning_WACV_2021_paper.pdf) | WACV | 2021 | ⭐️ | N/A |
| [FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_FACIAL_Synthesizing_Dynamic_Talking_Face_With_Implicit_Attribute_Learning_ICCV_2021_paper.pdf) | ICCV | 2021 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/zhangchenxu528/FACIAL) |
| [Cascade EF-GAN: Progressive Facial Expression Editing with Local Focuses](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Cascade_EF-GAN_Progressive_Facial_Expression_Editing_With_Local_Focuses_CVPR_2020_paper.pdf) | CVPR | 2020 | ⭐️ | N/A |
| [Interpreting the latent space of gans for semantic face editing](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing_CVPR_2020_paper.pdf) | CVPR | 2020 | ⭐️⭐️⭐️ | [TensorFlow](https://github.com/genforce/interfacegan) |


### Emotion Recognition, Facial Representations, and Others [back-to-top](#seven-emotion) <a id="er-fr-o"></a>

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [A Survey on Facial Expression Recognition of Static and Dynamic Emotions](https://arxiv.org/pdf/2408.15777) | ArXiv | 2024 | ⭐️⭐️⭐️ | [GitHub](https://github.com/wangyanckxx/SurveyFER) |
| [Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning](https://arxiv.org/pdf/2406.11161) | NeurIPS | 2024 | ⭐️⭐️⭐️ | [Project](https://github.com/ZebangCheng/Emotion-LLaMA) [Demo](https://huggingface.co/spaces/ZebangCheng/Emotion-LLaMA) |
| [Exploring Vision Language Models for Facial Attribute Recognition: Emotion, Race, Gender, and Age](https://arxiv.org/pdf/2410.24148) | ArXiv | 2024 | ⭐️ | N/A |
| [A survey on Graph Deep Representation Learning for Facial Expression Recognition](https://arxiv.org/pdf/2411.08472) | ArXiv | 2024 | ⭐️ | N/A |
| [Training A Small Emotional Vision Language Model for Visual Art Comprehension](https://arxiv.org/pdf/2403.11150) | ECCV | 2024 | ⭐️⭐️ | [GitHub](https://github.com/BetterZH/SEVLM-code) |
| [Affective Visual Dialog: A Large-Scale Benchmark for Emotional Reasoning Based on Visually Grounded Conversations](https://arxiv.org/pdf/2308.16349) | ECCV | 2024 | ⭐️⭐️⭐️ | [Project](https://affective-visual-dialog.github.io/) |
| [Generative Technology for Human Emotion Recognition: A Scope Review](https://arxiv.org/pdf/2407.03640) | ArXiv | 2024 | ⭐️⭐️ | N/A |
| [Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition](https://arxiv.org/pdf/2407.05374) | ACL | 2024 | ⭐️⭐️ | [PyTorch](https://github.com/zrguo/MPLMM) |
| [AffectGPT: Dataset and Framework for Explainable Multimodal Emotion Recognition](https://arxiv.org/pdf/2407.07653) | ArXiv | 2024 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/zeroQiaoba/AffectGPT) |
| [Towards Context-Aware Emotion Recognition Debiasing from a Causal Demystification Perspective via De-confounded Training](https://arxiv.org/pdf/2407.04963) | TPAMI | 2024 | ⭐️⭐️ | None |
| [EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models](https://arxiv.org/pdf/2401.04608) | CVPR | 2024 | ⭐️⭐️ | N/A |
| [A Unified and Interpretable Emotion Representation and Expression Generation](https://arxiv.org/pdf/2404.01243) | CVPR | 2024 | ⭐️⭐️⭐️ | [Project](https://emotion-diffusion.github.io/) |
| [Weakly-Supervised Emotion Transition Learning for Diverse 3D Co-speech Gesture Generation](https://arxiv.org/pdf/2311.17532) | CVPR | 2024 | ⭐️⭐️ | [Project](https://xingqunqi-lab.github.io/Emo-Transition-Gesture/) |
| [EmoVIT: Revolutionizing Emotion Insights with Visual Instruction Tuning](https://arxiv.org/pdf/2404.16670) | CVPR | 2024 | ⭐️⭐️⭐️ | [GitHub](https://github.com/aimmemotion/EmoVIT) |
| [Robust Emotion Recognition in Context Debiasing](https://arxiv.org/pdf/2403.05963) | CVPR | 2024 | ⭐️⭐️ | N/A |
| [Region-Based Emotion Recognition via Superpixel Feature Pooling](https://openreview.net/pdf?id=YTcu23qVUU) | CVPRW | 2024 | ⭐️⭐️ | N/A |
| [Emotion Recognition from the perspective of Activity Recognition](https://arxiv.org/abs/2403.16263) | ArXiv | 2024 | ⭐️ | N/A |
| [GPT as Psychologist? Preliminary Evaluations for GPT-4V on Visual Affective Computing](https://arxiv.org/abs/2403.05916) | ArXiv | 2024 | ⭐️⭐️⭐️ | [GitHub](https://github.com/EnVision-Research/GPT4Affectivity) |
| [The Strong Pull of Prior Knowledge in Large Language Models and Its Impact on Emotion Recognition](https://arxiv.org/pdf/2403.17125.pdf) | ArXiv | 2024 | ⭐️ | N/A |
| [DrFER: Learning Disentangled Representations for 3D Facial Expression Recognition](https://arxiv.org/pdf/2403.08318.pdf) | FG | 2024 | ⭐️⭐️ | N/A |
| [Distilling Privileged Multimodal Information for Expression Recognition using Optimal Transport](https://arxiv.org/pdf/2401.15489.pdf) | ArXiv | 2024 | ⭐️⭐️ | [PyTorch](https://github.com/haseebaslam95/PKDOT) |
| [Beyond Accuracy: Fairness, Scalability, and Uncertainty Considerations in Facial Emotion Recognition](https://openreview.net/pdf?id=HRj4VRUPtV) | NLDL | 2024 | ⭐️⭐️⭐️ | None |
| [Evaluating and Inducing Personality in Pre-trained Language Models](https://papers.nips.cc/paper_files/paper/2023/file/21f7b745f73ce0d1f9bcea7f40b1388e-Paper-Conference.pdf) | NeurIPS | 2024 | ⭐️⭐️⭐️ | [Site](https://sites.google.com/view/machinepersonality) |
| [Deep Imbalanced Learning for Multimodal Emotion Recognition in Conversations](https://arxiv.org/pdf/2312.06337.pdf) | ArXiv | 2023 | ⭐️⭐️ | None |
| [An Empirical Study of Super-resolution on Low-resolution Micro-expression Recognition](https://arxiv.org/pdf/2310.10022.pdf) | ArXiv | 2023 | ⭐️⭐️ | None |
| [EmoCLIP: A Vision-Language Method for Zero-Shot Video Facial Expression Recognition](https://arxiv.org/pdf/2310.16640.pdf) | ArXiv | 2023 | ⭐️⭐️ | [PyTorch](https://github.com/NickyFot/EmoCLIP) |
| [Towards affective computing that works for everyone](https://arxiv.org/abs/2309.10780) | ACII | 2023 | ⭐️⭐️⭐️ | None |
| [Emotional Listener Portrait: Realistic Listener Motion Simulation in Conversation](https://openaccess.thecvf.com/content/ICCV2023/papers/Song_Emotional_Listener_Portrait_Neural_Listener_Head_Generation_with_Emotion_ICCV_2023_paper.pdf) | ICCV | 2023 | ⭐️⭐️ | None |
| [Affective Image Filter: Reflecting Emotions from Text to Images](https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Affective_Image_Filter_Reflecting_Emotions_from_Text_to_Images_ICCV_2023_paper.pdf) | ICCV | 2023 | ⭐️⭐️ | None |
| [Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) | CVPR | 2023 | ⭐️ | [PyTorch](https://github.com/nku-zhichengzhang/WECL) |
| [Learning Emotion Representations from Verbal and Nonverbal Communication](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Learning_Emotion_Representations_From_Verbal_and_Nonverbal_Communication_CVPR_2023_paper.pdf) | CVPR | 2023 | ⭐️⭐️ | [GitHub](https://github.com/Xeaver/EmotionCLIP) |
| [Multivariate, Multi-frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Multivariate_Multi-Frequency_and_Multimodal_Rethinking_Graph_Neural_Networks_for_Emotion_CVPR_2023_paper.pdf) | CVPR | 2023 | ⭐️ | N/A |
| [How you feelin’? Learning Emotions and Mental States in Movie Scenes](https://arxiv.org/pdf/2304.05634.pdf) | CVPR | 2023 | ⭐️⭐️⭐️ | [Project](https://katha-ai.github.io/projects/emotx/) |
| [Decoupled Multimodal Distilling for Emotion Recognition](https://arxiv.org/pdf/2303.13802v1.pdf) | CVPR | 2023 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/mdswyz/dmd) |
| [Context De-confounded Emotion Recognition](https://arxiv.org/pdf/2303.11921.pdf) | CVPR | 2023 | ⭐️⭐️ | [PyTorch](https://github.com/ydk122024/CCIM) |
| [More is Better: A Database for Spontaneous Micro-Expression with High Frame Rates](https://arxiv.org/pdf/2301.00985.pdf) | ArXiv | 2023 | ⭐️ | N/A |
| [Pre-training strategies and datasets for facial representation learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730109.pdf) | ECCV | 2022 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/1adrianb/unsupervised-face-representation) |
| [Multi-Dimensional, Nuanced and Subjective – Measuring the Perception of Facial Expressions](https://openaccess.thecvf.com/content/CVPR2022/papers/Bryant_Multi-Dimensional_Nuanced_and_Subjective_-_Measuring_the_Perception_of_Facial_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️⭐️ | N/A |
| [General Facial Representation Learning in a Visual-Linguistic Manner](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_General_Facial_Representation_Learning_in_a_Visual-Linguistic_Manner_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/faceperceiver/farl) |
| [Robust Egocentric Photo-realistic Facial Expression Transfer for Virtual Reality](https://openaccess.thecvf.com/content/CVPR2022/papers/Jourabloo_Robust_Egocentric_Photo-Realistic_Facial_Expression_Transfer_for_Virtual_Reality_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️⭐️ | N/A |
| [Fair Contrastive Learning for Facial Attribute Classification](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Fair_Contrastive_Learning_for_Facial_Attribute_Classification_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/sungho-CoolG/FSCL) |
| [Quantified Facial Expressiveness for Affective Behavior Analytics](https://openaccess.thecvf.com/content/WACV2022/papers/Uddin_Quantified_Facial_Expressiveness_for_Affective_Behavior_Analytics_WACV_2022_paper.pdf) | WACV | 2022 | ⭐️ | N/A |
| [Deep facial expression recognition: A survey](https://ieeexplore.ieee.org/iel7/5165369/5520654/09039580.pdf?casa_token=CAh7bbilIRMAAAAA:EI5iTZcdsqualuSwzc1Zrk7DgNI8aHgJJ5MYZ2R9RM3r3CHQWkimHChANibA9olNRYthY2ShZg) | IEEE TAC | 2020 | 🏆 | N/A |
| [iMiGUE: An Identity-free Video Dataset for Micro-Gesture Understanding and Emotion Analysis](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_iMiGUE_An_Identity-Free_Video_Dataset_for_Micro-Gesture_Understanding_and_Emotion_CVPR_2021_paper.pdf) | CVPR | 2021 | ⭐️ | [Data](https://github.com/linuxsino/iMiGUE) |
| [Emotions as overlapping causal networks of emotion components: Implications and methodological approaches](https://journals.sagepub.com/doi/pdf/10.1177/1754073920988787) | Emotion Review | 2021 | ⭐️ | N/A |
| [Hidden Emotion Detection using Multi-modal Signals](https://dl.acm.org/doi/pdf/10.1145/3411763.3451721) | CHI | 2021 | ⭐️⭐️ | [Data](https://github.com/kdhht2334/Hidden_Emotion_Detection_using_MM_Signals) |
| [Latent to Latent: A Learned Mapper for Identity Preserving Editing of Multiple Face Attributes in StyleGAN-generated Images](https://openaccess.thecvf.com/content/WACV2022/papers/Khodadadeh_Latent_to_Latent_A_Learned_Mapper_for_Identity_Preserving_Editing_WACV_2022_paper.pdf) | WACV | 2021 | ⭐️ | [PyTorch](https://github.com/850552586/Latent-To-Latent) |
| [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1803.09179.pdf) | ArXiv | 2018 | ⭐️⭐️⭐️ | [Site](http://niessnerlab.org/projects/roessler2018faceforensics.html) |
| [Graph-Structured Referring Expression Reasoning in The Wild](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Graph-Structured_Referring_Expression_Reasoning_in_the_Wild_CVPR_2020_paper.pdf) | CVPR | 2020 | ⭐️ | [PyTorch](https://github.com/sibeiyang/sgmn) |
| [EmotiCon: Context-Aware Multimodal Emotion Recognition using Frege’s Principle](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mittal_EmotiCon_Context-Aware_Multimodal_Emotion_Recognition_Using_Freges_Principle_CVPR_2020_paper.pdf) | CVPR | 2020 | ⭐️ | [Site](https://gamma.umd.edu/researchdirections/affectivecomputing/emotionrecognition/emoticon/) |
| [Learning Visual Emotion Representations from Web Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Learning_Visual_Emotion_Representations_From_Web_Data_CVPR_2020_paper.pdf) | CVPR | 2020 | ⭐️ | [Site](https://github.com/cvlab-stonybrook/EmotionNet_CVPR2020) |
| [Computational Models of Emotion Inference in Theory of Mind: A Review and Roadmap](https://onlinelibrary.wiley.com/doi/epdf/10.1111/tops.12371) | Topics in cognitive science | 2019 | ⭐️⭐️⭐️ | N/A |
| [Putting feelings into words: Affect labeling as implicit emotion regulation](https://journals.sagepub.com/doi/10.1177/1754073917742706) | Emotion Review | 2018 | ⭐️⭐️⭐️ | N/A |
| [Affective cognition: Exploring lay theories of emotion](https://reader.elsevier.com/reader/sd/pii/S0010027715300196?token=5035CDA1C7A4252DE60FA657834E4BD568D820643E0D97128E504594DC5B0379E97E380A15E8D12031E97B737E62F68D&originRegion=us-east-1&originCreation=20230313150623) | Cognition</br>(ELSEVIER) | 2015 | ⭐️⭐️⭐️ | N/A |
| [Facial Expression Recognition: A Survey](https://reader.elsevier.com/reader/sd/pii/S1877050915021225?token=CE361276875BD44CA05330858CBB8A98AF346C512168EA04E34373DD30AEDFB05227F8A8B2540DCA3AF68A29F552F5C1&originRegion=us-east-1&originCreation=20230313160446) | Procedia Computer Science</br>(ELSEVIER) | 2015 | ⭐️⭐️ | N/A |
| [Norms of valence, arousal, and dominance for 13,915 English lemmas](https://link.springer.com/article/10.3758/s13428-012-0314-x) | Behavior Research Methods | 2013 | ⭐️⭐️ | N/A |
| [Facial expression and emotion](http://gruberpeplab.com/5131/5_Ekman_1993_Faicalexpressionemotion.pdf) | American Psychologist | 1993 | 🏆 | N/A |
| [Understanding face recognition](https://www.researchgate.net/profile/Louise_Hancock4/post/Are_there_any_research_to_reject_Bruce_and_Youngs_1986_theory_of_face_recognition/attachment/5f99527b7600090001f16eb1/AS%3A951570959183873%401603883604241/download/Bruce+and+Young+1986+Understanding+face+recognition.pdf) | British journal of psychology | 1986 | 🏆 | N/A |
| [A circumplex model of affect](https://d1wqtxts1xzle7.cloudfront.net/38425675/Russell1980-libre.pdf?1439132613=&response-content-disposition=inline%3B+filename%3DRussell1980.pdf&Expires=1678728266&Signature=JLK-DCUZNrH3iP-f3l5kB4uxUV~VUIhB04KfodmthXNX8n07xP1qkQ8ghjD0xtJR68zGUpp~19S2mOlPPBILqURiMV0iRcYUkqNoydOt~He463YsZAWMp105JjJfe40vGP-mmh~p5Ba~x3tTjtHx5fGPX~r15bnRhsjF7Q8~qC4L9m8DX1l3V0XCgQ97Ry5hhzGLTnKuDbHdMPkrkNRC598ibi4Pe54yrzYA0HoBaM-x4M1fak~tq6zt4lfMbVVeP2aQvVYzEWOLzO60J5zYqot9gdRyXuTl0lvqUB~BIspke1ZE7q2pm89~ZkoxYHGu7hg32PnfAXtj4fa6Q-NYMA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) | Journal of Personality and Social Psychology | 1980 | 🏆 | N/A |


### Datasets [back-to-top](#seven-emotion) <a id="datasets"></a>
| Name | Venue | Year | Impact | Site |
| :--- | :---: | :---:| :---:  | :---:|
| [FindingEmo: An Image Dataset for Emotion Recognition in the Wild](https://arxiv.org/pdf/2402.01355.pdf) | ArXiv | 2024 | ⭐️⭐️ | N/A |
| [VEATIC: Video-based Emotion and Affect Tracking in Context Dataset](https://arxiv.org/pdf/2309.06745v2.pdf) | WACV | 2024 | ⭐️ | [Project](https://veatic.github.io/) |
| [EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes](https://arxiv.org/pdf/2307.07961.pdf) | ICCV | 2023 | ⭐️⭐️ | [Project](https://vcc.tech/EmoSet) |
| [MimicME: A Large Scale Diverse 4D Database for Facial Expression Analysis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680457.pdf) | ECCV | 2022 | ⭐️⭐️ | [Github](https://github.com/apapaion/mimicme) |
| [CelebV-HQ: A Large-Scale Video Facial Attributes Dataset](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670641.pdf) | ECCV | 2022 | ⭐️⭐️⭐️ | [Project](https://celebv-hq.github.io/) [GitHub](https://github.com/CelebV-HQ/CelebV-HQ) [Demo](https://www.youtube.com/watch?v=Y0uxlUW4sW0) |
| [FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_FERV39k_A_Large-Scale_Multi-Scene_Dataset_for_Facial_Expression_Recognition_in_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️⭐️ | [Data](https://github.com/wangyanckxx/FERV39k) |
| [MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild](https://arxiv.org/pdf/2208.00847) | ACM MM | 2022 | ⭐️ | [Site](https://mafw-database.github.io/MAFW/) |
| [__Aff-wild2__: Extending the aff-wild database for affect recognition](https://arxiv.org/pdf/1811.07770) | ArXiv | 2018 | ⭐️⭐️ | [Project](https://ibug.doc.ic.ac.uk/resources/aff-wild2/) |
| [__Aff-wild__: valence and arousal 'In-the-Wild' challenge](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Zafeiriou_Aff-Wild_Valence_and_CVPR_2017_paper.pdf) | CVPRW | 2017 | ⭐️⭐️ | [Project](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) |


### Challenges [back-to-top](#seven-emotion) <a id="challenges"></a>

| Name | Venue | Year | Site |
| :--- | :---: | :---:| :---:|
| Facial Expression Recognition and Analysis Challenge | FG | 2015 | [Site](https://ibug.doc.ic.ac.uk/resources/FERA15/) |
| Emotion Recognition in the Wild Challenge (EmotiW) | ICMI | 2013-2018 | [Site](https://sites.google.com/view/emotiw2018) |
| Emotion Recognition in the Wild Challenge (EmotiW) | ICMI | 2023 | [Site](https://sites.google.com/view/emotiw2023) |
| Affect Recognition in-the-wild: Uni/Multi-Modal Analysis & VA-AU-Expression Challenges | FG | 2020 | [Site](https://ibug.doc.ic.ac.uk/resources/affect-recognition-wild-unimulti-modal-analysis-va/) |
| Affective Behavior Analysis In-the-Wild (1st) | FG | 2020 | [Site](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/) |
| Deepfake Detection Challenge | - | 2020 | [Site](https://ai.facebook.com/datasets/dfdc/) [Paper](https://arxiv.org/pdf/1910.08854.pdf) [GitHub](https://github.com/selimsef/dfdc_deepfake_challenge) | 
| Affective Behavior Analysis In-the-Wild (2nd) | ICCV | 2021 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) [Paper](https://openaccess.thecvf.com/ICCV2021_workshops/ABAW) |
| Affective Behavior Analysis In-the-Wild (3rd) | CVPR | 2022 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) [Paper](https://openaccess.thecvf.com/CVPR2022_workshops/ABAW) |
| Affective Behavior Analysis In-the-Wild (4th) | ECCV | 2022 | [Site](https://ibug.doc.ic.ac.uk/resources/eccv-2023-4th-abaw/) |
| The Multimodal Sentiment Analysis Challenge (MuSe) | ACM MM | 2022 | [Site](https://www.muse-challenge.org/) [Paper](https://arxiv.org/pdf/2207.05691.pdf) | 
| Affective Behavior Analysis In-the-Wild (5th) | CVPR | 2023 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2023-5th-abaw/) |
| Emotionally And Culturally Intelligent AI (1st) | ICCV | 2023 | [Site](https://iccv23-wecia.github.io/) |
| Affective Behavior Analysis in-the-wild (6th) | CVPR | 2024 | [Site](https://affective-behavior-analysis-in-the-wild.github.io/6th/) |
| Affective Behavior Analysis in-the-wild (7th) | ECCV | 2024 | [Site](https://affective-behavior-analysis-in-the-wild.github.io/7th/) |
| 1M-Deepfakes Detection Challenge | ACM MM | 2024 | [Paper](https://arxiv.org/pdf/2409.06991) [Site](https://deepfakes1m.github.io/)

### Tools [back-to-top](#seven-emotion) <a id="tools"></a>

| Name | Paper(/Feature) | Site |
| :--- | :---: | :---:|
| FLAME | [Learning a model of facial shape and expression from 4D scans](https://dl.acm.org/doi/pdf/10.1145/3130800.3130813) | [Project](https://flame.is.tue.mpg.de/) [TensorFlow](https://github.com/TimoBolkart/TF_FLAME) [PyTorch](https://github.com/HavenFeng/photometric_optimization)
| FaceNet | [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878.pdf) | [PyTorch](https://github.com/timesler/facenet-pytorch) [TensorFlow](https://github.com/davidsandberg/facenet) |
| Landmark detection | - | [PyTorch](https://github.com/cunjian/pytorch_face_landmark) |
| Age estimation | - | 
| DataGen | APIs all about human faces and bodies | [Web](https://docs.datagen.tech/en/latest/index.html) |


### Remarkable Papers (2019~) [back-to-top](#seven-emotion) <a id="previous"></a>

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [A Compact Embedding for Facial Expression Similarity](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vemulapalli_A_Compact_Embedding_for_Facial_Expression_Similarity_CVPR_2019_paper.pdf) | CVPR | 2019 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/AmirSh15/FECNet) |
| [A Personalized Affective Memory Model for Improving Emotion Recognition](http://proceedings.mlr.press/v97/barros19a/barros19a.pdf) | ICML | 2019 | ⭐️⭐️ | [TensorFlow](https://github.com/pablovin/P-AffMem) |
| [Facial Expression Recognition via Relation-based Conditional Generative Adversarial Network](https://dl.acm.org/doi/10.1145/3340555.3353753) | ICMI | 2019 | ⭐️ | N/A |
| [Facial Expression Recognition by De-expression Residue Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Facial_Expression_Recognition_CVPR_2018_paper.pdf) | CVPR | 2018 | ⭐️⭐️ | N/A |
| [Joint pose and expression modeling for facial expression recognition](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Joint_Pose_and_CVPR_2018_paper.pdf) | CVPR | 2018 | ⭐️⭐️ | [TensorFlow](https://github.com/FFZhang1231/Facial-expression-recognition) |
| [Identity-Aware Convolutional Neural Network for Facial Expression Recognition](https://ieeexplore.ieee.org/document/7961791) | FG | 2017 | ⭐️⭐️⭐️ | N/A |
| [Facenet2expnet: Regularizing a deep face recognition net for expression recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7961731&casa_token=pXkTiuN8h14AAAAA:VortCHqQThv1pMOSb5d1_yBtl8HjoncX90tvxPex2s06KZxxk-rHLOQxWQm0jFwlEMD1w4Mb9Q) | FG | 2017 | ⭐️⭐️ | N/A |
| [Facial expression recognition from near-infrared videos](https://pdf.sciencedirectassets.com/271526/1-s2.0-S0262885611X00069/1-s2.0-S0262885611000515/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAgaCXVzLWVhc3QtMSJHMEUCIFiRBdKTQnt2tJQdiIPUyZ5NkklwzXN7ndWpG%2FB3ldHkAiEAinrnqmBXx%2FQTu%2BW5TROEYUwe3fl4VM%2FiFluS8Xx1qDsqvAUIwf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDHRfTWGZGeIbh5ud5iqQBa8ed2uJybHSSbHh2WhcqRwxFXijTAzHE4P1rzbms7AgYXFr3DprlfTXICxhApQxvZjfvOXbTYcctW9%2FPTuhvs%2F4s6W5eTv6JJa1UpYpa1rHJpcImJw9m9A0idz0df6BcAZiV9iunoP0EgjTIJVUf%2ByY5cP8XM1bee03QyfaEBqn50uGLl6j%2Fz1HlRTJwjZwbmUeKieqibitqnougLPDd6gpEqoN9%2Bp8wq8AgwFb2FiBmLz7BAFzS6bqhfS6p14ZBuSo9gsjoNwKW9IbcT3BnXLoT6o86zmR%2BdutZuIGKfVy%2BQyExLJ2L7hjZWsbM%2BRNoV78L%2FgWyVC6e5acdTKbODBsIJxXHobOeizkrqR19bTADQPJCGd%2BfN%2FVGvk2g5zkeBM%2F7w4NJKbhhoRV8VnXIn1PiPajNptgBtOalhqaJ1ID1g4w7U8mi0rCWdJHwqCyir40zZ3atM2nr9C5ksSbrWWCrnWDdqXkGhZjSb05VG3zXtt%2FMgrj0XpBzBw6SIaUUL8Cgk2RcLPO9GFhuf8h2cmyby1SFi4uGL706Qydex5900rWTyMIBlUYy2UWeRL6iEYzN31YLTa5ePkVf6SU4TmPExChZ1RIIjphXd3OseK2owoN15U6A9qjdokkaNF%2BglU952eaZsLnd%2Ff4wAzKYvhGmIwUtyEiPP77VdUSgE9HSDQ1PtL0g%2FgngDw4J27lsrybEW%2B0%2BDdvvzM3GsV6U3SKil1fJE1DzuGwMA9EZCCi7RhTIDkCImjk5dvg%2Bgr7pzyfZ2Aeeg262on5FFvbGHdW7CnqwtgEj0lPMh5zcQm1I3ra1%2Bsbsh%2BBJ3g%2BUFrmDC8k9hAs78iKks3KkMoVcqbn2f4hrJoF%2F6dWIZiwqvePMOCBvaAGOrEBLkb%2Fg5fA3Sf66RiVyVy%2BBq3I07GiagBVf7KTDdCnNfbOKhmBhqDTlWNpmmAUnf38ndrAKBy%2F4I4EgFnA%2F0fYhQsqovwPkoTJmlW81aXBa4U4e28Xk1nUHTJXez3%2Be2qFDNAEiahPj2AVpgkmB7KlAR0Sco58hKCnWlvz7oUY60xSfhehYmrtwq6LD%2B%2FwZG4PPfSYWyW%2FDxcf9JMYNcps3zPIDnAYQaFTOf8Snv2DowMG&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230313T162109Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2IZH667A%2F20230313%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1c7e2184e360b5ffc3aa4f04b91337abcb3f8666a8d6c56a0522d816bc8db7be&hash=0f93b7e4a99cabf40ee6502e981b6bd9a94a1eb9e79d48a71ba784c4de8ed804&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0262885611000515&tid=spdf-d6a989c5-5c13-4e65-ba7e-f0595678bc29&sid=9c0d33e215a458404b8b7a19c5cb2654c657gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=11145707545057010d5150&rr=7a759bb74d4e9331&cc=kr) | Image and Vision Computing</br>(ELSEVIER) | 2011 | ⭐️ | N/A |
| [Facial expression decomposition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1238452&casa_token=MiF20iuuvgIAAAAA:vDTIx_7mvG5tLAQcyAXVBIaM-3HQcgxkwjWQ9CiLmPLybQBM89FgJaX33yY-VO7Gaz2mhu2pUg&tag=1) | ICCV | 2003 | 🏆 | N/A |


