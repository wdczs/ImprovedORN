# IORN: An Effective Remote Sensing Image Scene Classification Framework

## Introduction
Improved orientated response network (IORN) is descriped in a [IEEE Geoscience and Remote Sensing Letters paper](https://ieeexplore.ieee.org/document/8434220/).

Remote sensing images captured by satellites, however, usually show varied orientations because of the earth’s rotation and camera angles. This variation increases the difficulties of recognizing the class of a scene. 

Based on orientated response network ([ORN](https://arxiv.org/abs/1701.01833)), we design Improved orientated response network (IORN).


## Experimental result
We use the VGG16 model as our fundamental network. Then, we upgrade the original VGG16 with 3x3x4 A-ARFs and S-ORAlign to create the IOR4-VGG16 model.
IOR4-VGG16 are mainly tested on [NWPU-RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) and [UCM](http://weegee.vision.ucmerced.edu/datasets/landuse.html).
<img src='pic/result.png' width='850'>

## Prerequisites
* Linux(Ubuntu 16.04)
* install dependencies
'''
pip install -r requirements.txt
'''
* install IORN
'''
cd ImprovedORN/IORN_install/install/
bash install.sh
'''
