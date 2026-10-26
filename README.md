# Multi-Angle Fusion Deep Subspace Clustering via Channel Selection
## Abstract
Deep subspace clustering relies on informative feature representations and reliable sample relationships. However, convolutional autoencoders may retain noisy or irrelevant feature channels, compromising the quality of the self-representation coefficient matrices used for clustering. To address this limitation, we propose multi-angle fusion deep subspace clustering via channel selection (MFDSC), which integrates feature selection with relationship fusion. Specifically, a channel selection mechanism removes low-weight feature channels to reduce the influence of unreliable features on self-representation learning. A multi-angle fusion attention module subsequently refines and fuses the self-representation coefficient matrices by modeling inter-matrix dependencies, block-diagonal structures, and inter-sample relationships. The resulting coefficient matrix is used to construct an affinity matrix for spectral clustering. Experiments on five publicly available datasets demonstrate improved clustering performance over the compared methods. These results highlight the value of jointly refining feature channels and sample relationships for deep subspace clustering. Our code is available at https://github.com/jiazhuangdiandian/MFDSC.git.
## framework diagram
![](https://github.com/jiazhuangdiandian/MFDSC/blob/main/img/network.jpg "framework diagram")
## Multi-angle fusion attention mechanism module
![](https://github.com/jiazhuangdiandian/MFDSC/blob/main/img/MA.jpg "Multi-angle fusion attention mechanism module")
## Environment
1.Clonee thin repo:https://github.com/jiazhuangdiandian/MFDSC.git

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
## Training
Python main-[data].py

