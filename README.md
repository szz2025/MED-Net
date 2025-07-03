# MED-Net

## 0. Abstract

With the widespread application of the Kolmogorov-Arnold Network (KAN) framework in computer vision, KAN-based models have made remarkable progress in the field of medical image segmentation. However, these models do not always outperform models based on convolutional neural networks or Mamba. This limitation mainly stems from the fact that most existing KAN-based studies tend to focus on a single visual encoding strategy, while ignoring the advantages of integrating multiple visual encoding strategies. Secondly, most existing methods use max pooling and average pooling operations, which can effectively compress feature dimensions but are prone to losing rich contextual information due to only focusing on local extreme values or global averages. To address these challenges, we propose a new framework called MED-Net, which contains multiple visual encoding strategies and Manhattan distance-based pooling operations. This design makes full use of different encoding strategies and helps to extract comprehensive visual features. In addition, the proposed Manhattan-distance pooling uses reparameterized sorting and linear rank weighting to improve the contextual expression ability of the network without additional learnable parameters. Comprehensive experiments on three public benchmark datasets, BUSI, GlaS, and ISIC2017, show that our approach consistently outperforms the state-of-the-art methods on multiple evaluation metrics, demonstrating its robustness and effectiveness in addressing key segmentation challenges. For reproduction, the implementation codes can be checked out at https://github.com/szz2025/MED-Net.



## 1. Overview

<div align="center">
<img src="Figs/MED-Net.png" />
</div>



## 2. Main Environments

The environment installation process can be carried out as follows:

```
conda create -n MED-Net python=3.8
conda activate MED-Net
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```



## 3. Datasets

You can refer to [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) for processing datasets, but for the division of the PH2 dataset, please run the Prepare_PH2.py we provide to divide the training set, validation set, and test set. Then organize the .npy file into the following format:

'./datasets/'

- ISIC2017
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy
- ISIC2018
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy
- PH2
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy
- The resulted file structure is as follows. 
MED
├── inputs
│   ├── busi
│     ├── images
│           ├── malignant (1).png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── malignant (1)_mask.png
|           ├── ...
│   ├── GLAS
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
│   ├── ISIC2017
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...



## 4. Train the MED-Net

```
python train.py
```



## 5. Test the MED-Net 

First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.

```
python test.py
```



## 6. Comparison With State of the Arts

The performance of the proposed method is compared with the state-of-the-art models on the ISIC2017, ISIC2018, and $\text{PH}^2$ datasets, with the top two results highlighted in red and blue, respectively.

<div align="center">
<img src="Figs/Table1.png" />
</div>



## 7. Acknowledgement

Thanks to [Vim](https://github.com/hustvl/Vim), [U-KAN](https://github.com/Zhaoyi-Yan/U-KAN) and [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) for their outstanding works.
