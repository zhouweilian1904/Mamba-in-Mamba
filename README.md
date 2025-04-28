# MiM

Weilian Zhou, Sei-ichiro Kamata, Haipeng Wang, Man Sing Wong, Huiying (Cynthia) Hou,

Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral image classification,

Neurocomputing,

Volume 613,

2025,

128751,

ISSN 0925-2312,

https://doi.org/10.1016/j.neucom.2024.128751.

(https://www.sciencedirect.com/science/article/pii/S0925231224015224)

------
**Please kindly cite the paper if this code is useful and helpful for your research.** 

We are preparing this github step by step. Some codes are still modified, and you can also modify and improve them by yourself. Good Luck! 

Overall Structure
----------------------------------------
![image](https://github.com/zhouweilian1904/Mamba-in-Mamba/blob/main/whole_structure_3.jpg)

T-Mamba Encoder
----------------------------------------
![image](https://github.com/zhouweilian1904/Mamba-in-Mamba/blob/main/T_mamba_2.jpg)

**Abstract:**

Hyperspectral image (HSI) classification is pivotal in the remote sensing (RS) field, particularly with the advancement of deep learning techniques. Sequential models, such as Recurrent Neural Networks (RNNs) and Transformers, have been tailored to this task, offering unique viewpoints. However, they face several challenges: 1) RNNs struggle with aggregating central features and are sensitive to interfering pixels; 2) Transformers require extensive computational resources and often underperform with limited HSI training samples. To address these issues, recent advances have introduced State Space Models (SSM) and Mamba, known for their lightweight and parallel scanning capabilities within linear sequence processing, providing a balance between RNNs and Transformers. However, deploying Mamba as a backbone for HSI classification has not been fully explored. Although there are improved Mamba models for visual tasks, such as Vision Mamba (ViM) and Visual Mamba (VMamba), directly applying them to HSI classification encounters problems. For example, these models do not effectively handle land-cover semantic tokens with multi-scale perception for feature aggregation when implementing patch-wise HSI classifiers for central pixel prediction. Hence, the suitability of these models for this task remains an open question. In response, this study introduces the innovative Mamba-in-Mamba (MiM) architecture for HSI classification, marking the pioneering deployment of Mamba in this field. The MiM model includes: 1) A novel centralized Mamba-Cross-Scan (MCS) mechanism for transforming images into efficient-pair sequence data; 2) A Tokenized Mamba (T-Mamba) encoder that incorporates a Gaussian Decay Mask (GDM), a Semantic Token Learner (STL), and a Semantic Token Fuser (STF) for enhanced feature generation and concentration; and 3) A Weighted MCS Fusion (WMF) module coupled with a Multi-Scale Loss Design to improve model training efficiency. Experimental results from four public HSI datasets with fixed and disjoint training-testing samples demonstrate that our method outperforms existing baselines and is competitive with state-of-the-art approaches, highlighting its efficacy and potential in HSI applications.

--------------------------------
**Datasets:**

We have uploaded several datasets: https://drive.google.com/drive/folders/1IQxuz4jpwf6goB_2ZwVLEZz1ILScymdO?usp=share_link
1. Indian Pines, 
2. PaviaU, 
3. PaviaC, 
4. Botswana, 
5. Houston 2013, 
6. KSC, 
7. Mississippi_Gulfport, 
8. Salinas, 
9. Simulate_database, 
10. Augmented_IndianPines, 
11. Augmented_Mississippi_Gulfport, 
12. Augmented_PaviaU
13. The disjoint datasets (IP, PU, HU) can be referred in https://github.com/danfenghong/IEEE_TGRS_SpectralFormer.
14. WHU-Hi dataset: http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm

--------------------------------
**How to use:**

You can find and add some arguments in *main.py* for your own testing.

For example:

python main.py --model (MiM-v1, v2, v3) --dataset IndianPines --patch_size 7 --epoch 300 --cuda 0 --batch_size 64 --train_set Datasets/IndianPines/TRAIN_GT.mat --test_set Datasets/IndianPines/TEST_GT.mat

--------------------------------
**Models:**

In the *model.py*, we have implemented many types of different designs for HSI classification. You can try it with your debug because we are still modifying them. There may exist some mistakes. 
Some papers do not open-source their codes, so we code it by ourself.

--------------------------------
**Env settings:**

Pytorch:2.3

Cuda:12.2

Ubuntu 22.04

---------------------------------
**Many thanks**

DeepHyperX: https://github.com/nshaud/DeepHyperX.

VMamba: https://github.com/MzeroMiko/VMamba.

Tokenlearner: https://github.com/rish-16/tokenlearner-pytorch.

Mamba in pytorch: https://github.com/alxndrTL/mamba.py.

Vision Mamba: https://github.com/hustvl/Vim.





