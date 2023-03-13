# awesome-SOTA-FER
A curated list of facial expression recognition in both 7-emotion classification and affect estimation.


## Contributing
Please feel free to refer this repository for your FER research/development and send me [pull requests](https://github.com/kdhht2334/awesome-SOTA-FER/pulls) or email [Daeha Kim](kdhht5022@gmail.com) to add links.


## Table of Contents

- [SOTA Papers](#sota-paper)
  - 7-Emotion Classification
    - [2023](#2023-c)
    - [2022](#2022-c)
    - [2021](#2021-c)
    - [2020](#2020-c)
  - Valence-arousal Affect Estimation
    - [2023](#2023-ae)
    - [2022](#2022-ae)
    - [2021](#2021-ae)
    - [2020](#2020-ae)
  - Facial Action Unit (AU) Detection
    - [2023](#2023-au)
    - [2022](#2022-au)
    - [2021](#2021-au)
    - [2020](#2020-au)
    
- [Facial Expression Manipulation](#fem)
    
- [Emotion Theory or Dataset](#th-db)

- [Challenges](#challenges)

- [Previous Papers](#previous_papers)
  - 7-Emotion Classification
  - Valence-arousal Affect Estimation
  - Facial Action Unit (AU) Detection
  
  
- [Bench-marking Results](#benchmarking)



## SOTA Papers <a id="sota-paper"></a>

### 7-Emotion Classification

#### 2023 <a id="2023-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [RNAS-MER: A Refined Neural Architecture Search With Hybrid Spatiotemporal Operations for Micro-Expression Recognition]() | WACV | ⭐️ | N/A |
| [Uncertainty-aware Label Distribution Learning for Facial Expression Recognition]() | WACV | ⭐️⭐️ | [TensorFlow](https://github.com/minhnhatvt/label-distribution-learning-fer-tf/) |


#### 2022 <a id="2022-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| [FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_FERV39k_A_Large-Scale_Multi-Scene_Dataset_for_Facial_Expression_Recognition_in_CVPR_2022_paper.pdf) | CVPR | ⭐️⭐️⭐️ | [Data](https://github.com/wangyanckxx/FERV39k) |
| [Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Semi-Supervised_Deep_Facial_Expression_Recognition_With_an_Adaptive_Confidence_CVPR_2022_paper.pdf) | CVPR | ⭐️ | [PyTorch](https://github.com/hangyu94/Ada-CM/) |
| [Face2Exp: Combating Data Biases for Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Zeng_Face2Exp_Combating_Data_Biases_for_Facial_Expression_Recognition_CVPR_2022_paper.pdf) | CVPR | ⭐️⭐️⭐️ | [PyTorch](https://github.com/danzeng1990/Face2Exp) |

#### 2021 <a id="2021-c"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
[TransFER: Learning Relation-aware Facial Expression Representations with Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Xue_TransFER_Learning_Relation-Aware_Facial_Expression_Representations_With_Transformers_ICCV_2021_paper.pdf) | ICCV | ⭐️⭐️ | N/A |
| [Understanding and Mitigating Annotation Bias in Facial Expression Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Understanding_and_Mitigating_Annotation_Bias_in_Facial_Expression_Recognition_ICCV_2021_paper.pdf) | ICCV | ⭐️ | N/A |
| [Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty Estimation for Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/She_Dive_Into_Ambiguity_Latent_Distribution_Mining_and_Pairwise_Uncertainty_Estimation_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️⭐️ | [PyTorch](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DMUE) |
| [Affective Processes: stochastic modelling of temporal context for emotion and facial expression recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Sanchez_Affective_Processes_Stochastic_Modelling_of_Temporal_Context_for_Emotion_and_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️ | N/A |
| [Feature Decomposition and Reconstruction Learning for Effective Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Ruan_Feature_Decomposition_and_Reconstruction_Learning_for_Effective_Facial_Expression_Recognition_CVPR_2021_paper.pdf) | CVPR | ⭐️ | N/A |
| [Learning a Facial Expression Embedding Disentangled from Identity](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_a_Facial_Expression_Embedding_Disentangled_From_Identity_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️⭐️ | N/A |
| [Affect2MM: Affective Analysis of Multimedia Content Using Emotion Causality](https://openaccess.thecvf.com/content/CVPR2021/papers/Mittal_Affect2MM_Affective_Analysis_of_Multimedia_Content_Using_Emotion_Causality_CVPR_2021_paper.pdf) | CVPR | ⭐️⭐️⭐️ | [Site](https://gamma.umd.edu/researchdirections/affectivecomputing/emotionrecognition/affect2mm/) |
| [A Circular-Structured Representation for Visual Emotion Distribution Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_A_Circular-Structured_Representation_for_Visual_Emotion_Distribution_Learning_CVPR_2021_paper.pdf) | CVPR | ⭐️ | N/A |


#### 2020 <a id="2020-c"></a>



### Valence-arousal Affect Estimation

#### 2023 <a id="2023-ae"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| Optimal Transport-based Identity Matching for Identity-invariant Facial Expression Recognition | NeurIPS | ⭐️⭐️⭐️ | [PyTorch](https://github.com/kdhht2334/ELIM_FER) |
| ddd


#### 2022 <a id="2022-ae"></a>

#### 2021 <a id="2021-ae"></a>

#### 2020 <a id="2020-ae"></a>


### Facial Action Unit (AU) Detection

#### 2023 <a id="2023-au"></a>

| Paper | Venue | Impact | Code |
| :---  | :---: | :---:  | :---:|
| FAN-Trans: Online Knowledge Distillation for Facial Action Unit Detection | WACV | ⭐️ | N/A |

#### 2022 <a id="2022-au"></a>

#### 2021 <a id="2021-au"></a>

#### 2020 <a id="2020-au"></a>


### Facial Expression Manipulation (or Generation) <a id="fem"></a>

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Sparse to Dense Dynamic 3D Facial Expression Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Otberdout_Sparse_to_Dense_Dynamic_3D_Facial_Expression_Generation_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️ | [GitHub](https://github.com/CRISTAL-3DSAM/Sparse2Dense) |
| [Neural Emotion Director: Speech-preserving semantic control of facial expressions in “in-the-wild” videos](https://openaccess.thecvf.com/content/CVPR2022/papers/Papantoniou_Neural_Emotion_Director_Speech-Preserving_Semantic_Control_of_Facial_Expressions_in_CVPR_2022_paper.pdf) | CVPR (best paper finalist) | 2022 | ⭐️⭐️⭐️ | [Site](https://foivospar.github.io/NED/) |
| [Information Bottlenecked Variational Autoencoder for Disentangled 3D Facial Expression Modelling](https://openaccess.thecvf.com/content/WACV2022/papers/Sun_Information_Bottlenecked_Variational_Autoencoder_for_Disentangled_3D_Facial_Expression_Modelling_WACV_2022_paper.pdf) | WACV | 2022 | ⭐️ | N/A |
| [Detection and Localization of Facial Expression Manipulations](https://openaccess.thecvf.com/content/WACV2022/papers/Mazaheri_Detection_and_Localization_of_Facial_Expression_Manipulations_WACV_2022_paper.pdf) | WACV | 2022 | ⭐️ | N/A |
| [Audio-Driven Emotional Video Portraits](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Audio-Driven_Emotional_Video_Portraits_CVPR_2021_paper.pdf) | CVPR | 2021 | ⭐️ | [Site](https://jixinya.github.io/projects/evp/) |
| [GANmut: Learning Interpretable Conditional Space for Gamut of Emotions](https://openaccess.thecvf.com/content/CVPR2021/papers/dApolito_GANmut_Learning_Interpretable_Conditional_Space_for_Gamut_of_Emotions_CVPR_2021_paper.pdf) | CVPR | 2021 | ⭐️⭐️ | [PyTorch](https://github.com/stefanodapolito/GANmut) |


### Emotion Theory or Dataset <a id="th-db"></a>

| Paper | Venue | Year | Impact | Code |
| :---  | :---: | :---:| :---:  | :---:|
| [Multi-Dimensional, Nuanced and Subjective – Measuring the Perception of Facial Expressions](https://openaccess.thecvf.com/content/CVPR2022/papers/Bryant_Multi-Dimensional_Nuanced_and_Subjective_-_Measuring_the_Perception_of_Facial_CVPR_2022_paper.pdf) | CVPR | 2022 | ⭐️⭐️⭐️ | N/A |
| [Quantified Facial Expressiveness for Affective Behavior Analytics](https://openaccess.thecvf.com/content/WACV2022/papers/Uddin_Quantified_Facial_Expressiveness_for_Affective_Behavior_Analytics_WACV_2022_paper.pdf) | WACV | 2022 | ⭐️ | N/A |
| [iMiGUE: An Identity-free Video Dataset for Micro-Gesture Understanding and Emotion Analysis](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_iMiGUE_An_Identity-Free_Video_Dataset_for_Micro-Gesture_Understanding_and_Emotion_CVPR_2021_paper.pdf) | CVPR | 2021 | ⭐️ | [Data](https://github.com/linuxsino/iMiGUE) |
| [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1803.09179.pdf) | ArXiv | 2018 | ⭐️⭐️⭐️ | [Site](http://niessnerlab.org/projects/roessler2018faceforensics.html) |

### Challenges <a id="challenges"></a>

| Name | Venue | Year | Site |
| :--- | :---: | :---:| :---:|
| Facial Expression Recognition and Analysis Challenge 2015 | FG | 2015 | [Site](https://ibug.doc.ic.ac.uk/resources/FERA15/) |
| Emotion Recognition in the Wild Challenge (EmotiW) | ICMI | 2013-2018 | [Site](https://sites.google.com/view/emotiw2018) |
| Affect Recognition in-the-wild: Uni/Multi-Modal Analysis & VA-AU-Expression Challenges | FG | 2020 | [Site](https://ibug.doc.ic.ac.uk/resources/affect-recognition-wild-unimulti-modal-analysis-va/) |
| Affective Behavior Analysis In-the-Wild (1st) | FG | 2020 | [Site](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/) |
| Affective Behavior Analysis In-the-Wild (2nd) | ICCV | 2021 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) [Papers](https://openaccess.thecvf.com/ICCV2021_workshops/ABAW) |
| Affective Behavior Analysis In-the-Wild (3rd) | CVPR | 2022 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) [Papers](https://openaccess.thecvf.com/CVPR2022_workshops/ABAW) |
| Affective Behavior Analysis In-the-Wild (4th) | ECCV | 2022 | [Site](https://ibug.doc.ic.ac.uk/resources/eccv-2023-4th-abaw/) |
| Affective Behavior Analysis In-the-Wild (5th) | CVPR | 2023 | [Site](https://ibug.doc.ic.ac.uk/resources/cvpr-2023-5th-abaw/) |
