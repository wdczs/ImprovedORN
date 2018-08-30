# IORN: An Effective Remote Sensing Image Scene Classification Framework

## Introduction
Improved orientated response network (IORN) is descriped in a [IEEE Geoscience and Remote Sensing Letters paper](https://ieeexplore.ieee.org/document/8434220/).

Remote sensing images captured by satellites, however, usually show varied orientations because of the earth’s rotation and camera angles. This variation increases the difficulties of recognizing the class of a scene. 

Based on orientated response network ([ORN](https://arxiv.org/abs/1701.01833)), we design Improved orientated response network (IORN).
1.  Based on active rotating filters (ARFs), we propose average active rotating filters (A-ARF), which leads to good convergence and high learning speed without requiring additional computation.
2.  We propose S-ORAlign, which is an improvement of ORAlign. The squeeze layer in S-ORAlign makes it possible for the model to address large-scale remote sensing images. 
<img src='pic/S-ORAlign.png' width='550'>

## Experimental result
We use the VGG16 model as our fundamental network. Then, we upgrade the original VGG16 with 3x3x4 A-ARFs and S-ORAlign to create the IOR4-VGG16 model.
IOR4-VGG16 are mainly tested on [NWPU-RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) and [UCM](http://weegee.vision.ucmerced.edu/datasets/landuse.html).
<img src='pic/result.png' width='850'>
