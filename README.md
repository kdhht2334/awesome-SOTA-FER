# awesome-SOTA-FER
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of facial expression recognition in both 7-emotion classification and affect estimation. In addition, this repository includes basic studies on FER and recent datasets.

👀 Update frequently!


#### Notice
⭐️: __Importance__ of FER method (from 1 to 3 points)

🏆: __Mandatory (research) paper__ to understand FER technology.


## What's New

- [Mar. 2023] Initial update for FER papers.
- [Mar. 2023] Create awesome-SOTA-FER repository.

## Contributing
Please feel free to refer this repository for your FER research/development and send me [pull requests](https://github.com/kdhht2334/awesome-SOTA-FER/pulls) or email [Daeha Kim](kdhht5022@gmail.com) to add links.


## Table of Contents

- [7-Emotion Classification](#seven-emotion)
  - [2023](#2023-c)
  - [2022](#2022-c)
  - [2021](#2021-c)
  - [2020](#2020-c)

- [Valence-arousal Affect Estimation](#affect)

- [Facial Action Unit (AU) Detection / Recognition](#au)

- [Privacy-aware Facial Expression Recognition](#privacy)
    
- [Facial Expression Manipulation](#fem)
    
- [Emotion Recognition, Facial Representations, and Others](#er-fr-o)

- [Challenges](#challenges)
  
- [Tools](#tools)

- [Previous Papers (2019~)](#previous)



## 7-Emotion Classification <a id="seven-emotion"></a>

#### 2023 <a id="2023-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [RNAS-MER: A Refined Neural Architecture Search With Hybrid Spatiotemporal Operations for Micro-Expression Recognition]() | WACV | ⭐️ | N/A |
| [Uncertainty-aware Label Distribution Learning for Facial Expression Recognition]() | WACV | ⭐️⭐️ | [TensorFlow](https://github.com/minhnhatvt/label-distribution-learning-fer-tf/) |
| [POSTER V2: A simpler and stronger facial expression recognition network](https://arxiv.org/pdf/2301.12149.pdf) | ArXiv | ⭐️⭐️ | [PyTorch](https://github.com/Talented-Q/POSTER_V2) |


#### 2022 <a id="2022-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
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



### Valence-arousal Affect Estimation <a id="affect"></a>


| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Optimal Transport-based Identity Matching for Identity-invariant Facial Expression Recognition](https://arxiv.org/pdf/2209.12172.pdf) | NeurIPS</br>(spotlight) | 2022 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/kdhht2334/ELIM_FER) |
| [Emotion-aware Multi-view Contrastive Learning for Faciel Emotion Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730181.pdf) | ECCV | 2022 | ⭐️ | [PyTorch](https://github.com/kdhht2334/AVCE_FER) |
| [Learning from Label Relationships in Human Affect](https://arxiv.org/pdf/2207.05577.pdf) | ACM MM | 2022 | ⭐️ | N/A |
| [Are 3D Face Shapes Expressive Enough for Recognising Continuous Emotions and Action Unit Intensities?](https://arxiv.org/pdf/2207.01113.pdf) | ArXiv | 2022 | ⭐️⭐️ | N/A |
| [Contrastive Adversarial Learning for Person Independent Facial Emotion Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/16743) | AAAI | 2021 | ⭐️ | [PyTorch](https://github.com/kdhht2334/Contrastive-Adversarial-Learning-FER) |
| [Estimating continuous affect with label uncertainty](https://ieeexplore.ieee.org/iel7/9597394/9597388/09597425.pdf?casa_token=0QyPEEV_UKwAAAAA:vXZLYYOLqO3kXHCwSzdYx8tAyABtJ-gzK6VBk79HUlNHzZK0__gLA7TdQd2AnrCyeKtUnCqoUg) | ACII | 2021 | ⭐️ | N/A |
| [Factorized Higher-Order CNNs with an Application to Spatio-Temporal Emotion Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kossaifi_Factorized_Higher-Order_CNNs_With_an_Application_to_Spatio-Temporal_Emotion_Estimation_CVPR_2020_paper.pdf) | CVPR | 2020 | ⭐️⭐️ | N/A |
| [BReG-NeXt: Facial affect computing using adaptive residual networks with bounded gradient](https://ieeexplore.ieee.org/iel7/5165369/5520654/09064942.pdf?casa_token=mmeTU4Eqv6IAAAAA:m6rR6FVdNZhuuxcTA5G8z5j2h28hnm5zE3FtgkEJOeXUfT818fM501SzCIoJ_aHmx6yGWHE78w) | IEEE TAC | 2020 | ⭐️⭐️ | [TensorFlow](https://github.com/behzadhsni/BReG-NeXt) |


### Facial Action Unit (AU) Detection / Recognition <a id="au"></a>

  - This section also contains `landmark` detection.

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [FAN-Trans: Online Knowledge Distillation for Facial Action Unit Detection](https://openaccess.thecvf.com/content/WACV2023/papers/Yang_FAN-Trans_Online_Knowledge_Distillation_for_Facial_Action_Unit_Detection_WACV_2023_paper.pdf) | WACV | 2023 | ⭐️ | N/A |
| [Knowledge-Driven Self-Supervised Representation Learning for Facial Action Unit Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_Knowledge-Driven_Self-Supervised_Representation_Learning_for_Facial_Action_Unit_Recognition_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | N/A |
| [Towards Accurate Facial Landmark Detection via Cascaded Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Accurate_Facial_Landmark_Detection_via_Cascaded_Transformers_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | N/A |
| [Causal intervention for subject-deconfounded facial action unit recognition](https://ojs.aaai.org/index.php/AAAI/article/view/19914/19673) | AAAI</br>(Oral) | 2022 | ⭐️⭐️⭐️ | N/A |
| [PIAP-DF: Pixel-Interested and Anti Person-Specific Facial Action Unit Detection Net with Discrete Feedback Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_PIAP-DF_Pixel-Interested_and_Anti_Person-Specific_Facial_Action_Unit_Detection_Net_ICCV_2021_paper.pdf) | ICCV | 2021 | ⭐️ | N/A |


### Privacy-aware Facial Expression Recognition <a id="privacy"></a>

  - __Note__: Some face recognition studies are included.
  - You can get additional information [here](https://github.com/Tencent/TFace)

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Simulated adversarial testing of face recognition models](https://openaccess.thecvf.com/content/CVPR2022/papers/Ruiz_Simulated_Adversarial_Testing_of_Face_Recognition_Models_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | N/A|
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


### Facial Expression Manipulation / Generation <a id="fem"></a>

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [EMOCA: Emotion Driven Monocular Face Capture and Animation](https://openaccess.thecvf.com/content/CVPR2022/papers/Danecek_EMOCA_Emotion_Driven_Monocular_Face_Capture_and_Animation_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [Project](https://emoca.is.tue.mpg.de/) |
| [Sparse to Dense Dynamic 3D Facial Expression Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Otberdout_Sparse_to_Dense_Dynamic_3D_Facial_Expression_Generation_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | [GitHub](https://github.com/CRISTAL-3DSAM/Sparse2Dense) |
| [Neural Emotion Director: Speech-preserving semantic control of facial expressions in “in-the-wild” videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Papantoniou_Neural_Emotion_Director_Speech-Preserving_Semantic_Control_of_Facial_Expressions_in_CVPR_2022_paper.pdf) | CVPR</br>(best paper finalist) | 2022 | ⭐️⭐️⭐️ | [Site](https://foivospar.github.io/NED/) |
| [TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_TransEditor_Transformer-Based_Dual-Space_GAN_for_Highly_Controllable_Facial_Editing_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/BillyXYB/TransEditor) |
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


### Emotion Recognition, Facial Representations, and Others <a id="er-fr-o"></a>

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Decoupled Multimodal Distilling for Emotion Recognition](https://arxiv.org/pdf/2303.13802v1.pdf) | CVPR | 2023 | ⭐️⭐️⭐️ | [PyTorch](https://github.com/mdswyz/dmd) |
| [Context De-confounded Emotion Recognition](https://arxiv.org/pdf/2303.11921.pdf) | CVPR | 2023 | ⭐️⭐️ | [PyTorch](https://github.com/ydk122024/CCIM) |
| [More is Better: A Database for Spontaneous Micro-Expression with High Frame Rates](https://arxiv.org/pdf/2301.00985.pdf) | ArXiv | 2023 | ⭐️ | N/A |
| [FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_FERV39k_A_Large-Scale_Multi-Scene_Dataset_for_Facial_Expression_Recognition_in_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️⭐️ | [Data](https://github.com/wangyanckxx/FERV39k) |
| [Multi-Dimensional, Nuanced and Subjective – Measuring the Perception of Facial Expressions](https://openaccess.thecvf.com/content/CVPR2022/papers/Bryant_Multi-Dimensional_Nuanced_and_Subjective_-_Measuring_the_Perception_of_Facial_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️⭐️ | N/A |
| [General Facial Representation Learning in a Visual-Linguistic Manner](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_General_Facial_Representation_Learning_in_a_Visual-Linguistic_Manner_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/faceperceiver/farl) |
| [Robust Egocentric Photo-realistic Facial Expression Transfer for Virtual Reality](https://openaccess.thecvf.com/content/CVPR2022/papers/Jourabloo_Robust_Egocentric_Photo-Realistic_Facial_Expression_Transfer_for_Virtual_Reality_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️⭐️ | N/A |
| [Fair Contrastive Learning for Facial Attribute Classification](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Fair_Contrastive_Learning_for_Facial_Attribute_Classification_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️ | [PyTorch](https://github.com/sungho-CoolG/FSCL) |
| [MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild](https://arxiv.org/pdf/2208.00847) | ACM MM | 2022 | ⭐️ | [Site](https://mafw-database.github.io/MAFW/) |
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
| [__Aff-wild2__: Extending the aff-wild database for affect recognition](https://arxiv.org/pdf/1811.07770) | ArXiv | 2018 | ⭐️⭐️ | N/A |
| [Putting feelings into words: Affect labeling as implicit emotion regulation](https://journals.sagepub.com/doi/10.1177/1754073917742706) | Emotion Review | 2018 | ⭐️⭐️⭐️ | N/A |
| [__Aff-wild__: valence and arousal 'In-the-Wild' challenge](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Zafeiriou_Aff-Wild_Valence_and_CVPR_2017_paper.pdf) | CVPRW | 2017 | ⭐️⭐️ | N/A |
| [Affective cognition: Exploring lay theories of emotion](https://reader.elsevier.com/reader/sd/pii/S0010027715300196?token=5035CDA1C7A4252DE60FA657834E4BD568D820643E0D97128E504594DC5B0379E97E380A15E8D12031E97B737E62F68D&originRegion=us-east-1&originCreation=20230313150623) | Cognition</br>(ELSEVIER) | 2015 | ⭐️⭐️⭐️ | N/A |
| [Facial Expression Recognition: A Survey](https://reader.elsevier.com/reader/sd/pii/S1877050915021225?token=CE361276875BD44CA05330858CBB8A98AF346C512168EA04E34373DD30AEDFB05227F8A8B2540DCA3AF68A29F552F5C1&originRegion=us-east-1&originCreation=20230313160446) | Procedia Computer Science</br>(ELSEVIER) | 2015 | ⭐️⭐️ | N/A |
| [Norms of valence, arousal, and dominance for 13,915 English lemmas](https://link.springer.com/article/10.3758/s13428-012-0314-x) | Behavior Research Methods | 2013 | ⭐️⭐️ | N/A |
| [Facial expression and emotion](http://gruberpeplab.com/5131/5_Ekman_1993_Faicalexpressionemotion.pdf) | American Psychologist | 1993 | 🏆 | N/A |
| [Understanding face recognition](https://www.researchgate.net/profile/Louise_Hancock4/post/Are_there_any_research_to_reject_Bruce_and_Youngs_1986_theory_of_face_recognition/attachment/5f99527b7600090001f16eb1/AS%3A951570959183873%401603883604241/download/Bruce+and+Young+1986+Understanding+face+recognition.pdf) | British journal of psychology | 1986 | 🏆 | N/A |
| [A circumplex model of affect](https://d1wqtxts1xzle7.cloudfront.net/38425675/Russell1980-libre.pdf?1439132613=&response-content-disposition=inline%3B+filename%3DRussell1980.pdf&Expires=1678728266&Signature=JLK-DCUZNrH3iP-f3l5kB4uxUV~VUIhB04KfodmthXNX8n07xP1qkQ8ghjD0xtJR68zGUpp~19S2mOlPPBILqURiMV0iRcYUkqNoydOt~He463YsZAWMp105JjJfe40vGP-mmh~p5Ba~x3tTjtHx5fGPX~r15bnRhsjF7Q8~qC4L9m8DX1l3V0XCgQ97Ry5hhzGLTnKuDbHdMPkrkNRC598ibi4Pe54yrzYA0HoBaM-x4M1fak~tq6zt4lfMbVVeP2aQvVYzEWOLzO60J5zYqot9gdRyXuTl0lvqUB~BIspke1ZE7q2pm89~ZkoxYHGu7hg32PnfAXtj4fa6Q-NYMA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) | Journal of Personality and Social Psychology | 1980 | 🏆 | N/A |


### Challenges <a id="challenges"></a>

| Name | Venue | Year | Site |
| :--- | :---: | :---:| :---:|
| Facial Expression Recognition and Analysis Challenge | FG | 2015 | [Site](https://ibug.doc.ic.ac.uk/resources/FERA15/) |
| Emotion Recognition in the Wild Challenge (EmotiW) | ICMI | 2013-2018 | [Site](https://sites.google.com/view/emotiw2018) |
| Affect Recognition in-the-wild: Uni/Multi-Modal Analysis & VA-AU-Expression Challenges | FG | 2020 | [Site](https://ibug.doc.ic.ac.uk/resources/affect-recognition-wild-unimulti-modal-analysis-va/) |
| Affective Behavior Analysis In-the-Wild (1st) | FG | 2020 | [Site](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/) |
| Deepfake Detection Challenge | - | 2020 | [Site](https://ai.facebook.com/datasets/dfdc/) [Paper](https://arxiv.org/pdf/1910.08854.pdf) [GitHub](https://github.com/selimsef/dfdc_deepfake_challenge) | 
| Affective Behavior Analysis In-the-Wild (2nd) | ICCV | 2021 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) [Paper](https://openaccess.thecvf.com/ICCV2021_workshops/ABAW) |
| Affective Behavior Analysis In-the-Wild (3rd) | CVPR | 2022 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) [Paper](https://openaccess.thecvf.com/CVPR2022_workshops/ABAW) |
| Affective Behavior Analysis In-the-Wild (4th) | ECCV | 2022 | [Site](https://ibug.doc.ic.ac.uk/resources/eccv-2023-4th-abaw/) |
| The Multimodal Sentiment Analysis Challenge (MuSe) | ACM MM | 2022 | [Site](https://www.muse-challenge.org/) [Paper](https://arxiv.org/pdf/2207.05691.pdf) | 
| Affective Behavior Analysis In-the-Wild (5th) | CVPR | 2023</br>(not yet) | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2023-5th-abaw/) |


### Tools <a id="tools"></a>

| Name | Paper | Site |
| :--- | :---: | :---:|
| FLAME | [Learning a model of facial shape and expression from 4D scans](https://dl.acm.org/doi/pdf/10.1145/3130800.3130813) | [Project](https://flame.is.tue.mpg.de/) [TensorFlow](https://github.com/TimoBolkart/TF_FLAME) [PyTorch](https://github.com/HavenFeng/photometric_optimization)
| FaceNet | [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878.pdf) | [PyTorch](https://github.com/timesler/facenet-pytorch) [TensorFlow](https://github.com/davidsandberg/facenet) |
| Landmark detection | - | [PyTorch](https://github.com/cunjian/pytorch_face_landmark) |
| Age estimation | - | 


### Previous Papers (2019~) <a id="previous"></a>

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


