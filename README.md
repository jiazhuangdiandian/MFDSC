# Multi-Angle Fusion Deep Subspace Clustering via Channel Selection
## Abstract
Subspace clustering networks based on convolutional autoencoders have achieved remarkable success in clustering tasks. However, not all channels in the convolutional layers contribute positively; some may learn noisy or irrelevant features, which degrade the quality of the self-representation coefficient matrix and consequently affect the clustering performance. To address this issue, this paper proposes a multi-angle fusion deep subspace clustering method via channel selection (MFDSC). The work effectively removes noisy channel features through a channel selection mechanism, thereby improving clustering performance. Additionally, we design a multi-angle fusion attention module that weights the self-representation coefficient matrix from three different perspectives, further enhancing its quality. Experimental results on five publicly available datasets fully demonstrate the effectiveness and superiority of MFDSC in clustering performance. Experimental results show that our approach presents significant performance advantages over current advanced baseline methods. Our code is available at https://github.com/jiazhuangdiandian/MFDSC.git.
## framework diagram
![]([https://github.com/jiazhuangdiandian/MFDSC/blob/main/img/network.jpg) "framework diagram")

![]([https://github.com/jiazhuangdiandian/MFDSC/blob/main/img/MA.jpg)
## Environment
1.Clonee thin repo:https://github.com/jiazhuangdiandian/ATGP-Net.git

2.Create a new conda environment and install dependencies:

pip:
```
-addict==2.4.0
-dataclasses=-0.8
-mmcv-full==-1.2.7
-numpy==1.19.5
-opencv-python==4.5.1.48
-perceptual=-0.1
-pillow==8.4.0
-scikit-image==0.17.2
-scipy==1.5.4
-tifffile==2020.9.3
-timm==0.3.2
-torch==1.7.1
-torchvision==0.8.2
-typing-extensions==4.0.0
-yapf==0.31.0
```
## Training & Test
Python train.py

Python val.py
## Dataset
Breast Ultrasound Dataset B: https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php

Breast Ultrasound Images (BUSI)：https://github.com/hugofigueiras/Breast-Cancer-Imaging-Datasets

BLUI:https://qamebi.com/breast-ultrasound-images-database/
