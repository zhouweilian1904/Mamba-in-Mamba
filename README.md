# MiM

The repository for the paper "Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral Image Classification" (preprint version: https://arxiv.org/abs/2405.12003)

------

Weilian Zhou, Sei-Ichiro Kamata, Haipeng Wang, Man-Sing Wong, Huiying (Cynthia) Hou

Feel free to contact us if there is anything we can help. Thanks for your support!

zhouweilian1904@akane.waseda.jp

------
**Please kindly cite the paper if this code is useful and helpful for your research.** 

We are preparing this github one by one.

Overall Structure
----------------------------------------
![image](https://github.com/zhouweilian1904/Mamba-in-Mamba/blob/main/whole%20structure.png)

T-Mamba Encoder
----------------------------------------
![image](https://github.com/zhouweilian1904/Mamba-in-Mamba/blob/main/T_mamba_3.png)

**Abstract:**

Hyperspectral image (HSI) classification is pivotal in the remote sensing (RS) field, particularly with the advancement of deep learning techniques. Sequential models, adapted from the natural language processing (NLP) field—such as Recurrent Neural Networks (RNNs) and Transformers—have been tailored to this task, offering a unique viewpoint. However, several challenges persist: 1) RNNs struggle with centric feature aggregation and are sensitive to interfering pixels, 2) Transformers require significant computational resources and often underperform with limited HSI training samples, and 3) Current scanning methods for converting images into sequence-data are simplistic and inefficient. In response, this study introduces the innovative Mamba-in-Mamba (MiM) architecture for HSI classification, the first attempt of deploying State Space Model (SSM) in this task. The MiM model includes: 1) A novel centralized Mamba-Cross-Scan (MCS) mechanism for transforming images into sequence-data, 2) A Tokenized Mamba (T-Mamba) encoder that incorporates a Gaussian Decay Mask (GDM), a Semantic Token Learner (STL), and a Semantic Token Fuser (STF) for enhanced feature generation and concentration, and 3) A Weighted MCS Fusion (WMF) module coupled with a Multi-Scale Loss Design to improve decoding efficiency. Experimental results from three public HSI datasets with fixed and disjoint training-testing samples demonstrate that our method outperforms existing baselines and state-of-the-art approaches, highlighting its efficacy and potential in HSI applications.

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

--------------------------------
**How to use:**

You can find and add some arguments in *main.py* for your own testing.

For example:

python main.py --model MiM-v1 --dataset IndianPines --patch_size 7 --epoch 300 --cuda 0 --batch_size 64 --train_set Datasets/IndianPines/TRAIN_GT.mat --test_set Datasets/IndianPines/TEST_GT.mat

--------------------------------
**Models:**

In the *model.py*, we have implemented many types of different designs for HSI classification. You can try it with your debug because we are still modifying them. There may exist some mistakes. 
Some papers do not open-source their codes, so we code it by ourself.

--------------------------------
**Env settings:**

Pytorch:2.3

Cuda:12.2

Ubuntu 22.04
