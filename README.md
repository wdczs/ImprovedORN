# IORN: An Effective Remote Sensing Image Scene Classification Framework

## Introduction
Improved orientated response network (IORN) is descriped in a [IEEE Geoscience and Remote Sensing Letters paper](https://ieeexplore.ieee.org/document/8434220/).

Remote sensing images captured by satellites, however, usually show varied orientations because of the earth’s rotation and camera angles. This variation increases the difficulties of recognizing the class of a scene. 

Based on orientated response network ([ORN](https://arxiv.org/abs/1701.01833)), we design Improved orientated response network (IORN).
We use the VGG16 model as our fundamental network. Then, we upgrade the original VGG16 with 3x3x4 A-ARFs and S-ORAlign to create the IOR4-VGG16 model.

<img src='pic/arch.png' width='400'>

## Experimental result
IOR4-VGG16 are mainly tested on [NWPU-RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) and [UCM](http://weegee.vision.ucmerced.edu/datasets/landuse.html).
<img src='pic/result.png' width='850'>

## Prerequisites
* Linux(Ubuntu 16.04)
* install dependencies
	```
	pip install -r requirements.txt
	```
* install IORN
```
cd ImprovedORN/IORN_install/install/
bash install.sh
```

## Train IOR4-VGG16 on UCM
1. download pre-trained IOR4-VGG16 (90 epoches on Imagenet) from [OneDrive](https://1drv.ms/u/s!AseOni9i6qlubypHdgKxxcNA5s8) or [BaiduYun](https://pan.baidu.com/s/1e39zySQtMZ9kRcc9bSo-lA). Then move the pre-trained moedl to ImprovedORN/pretrained_model/
2. download [UCM](http://weegee.vision.ucmerced.edu/datasets/landuse.html), and make sure it looks like this:
```
.../ImprovedORN/datasets/UCMerced_LandUse/Images/
```
3. train IOR4-VGG16 on UCM
```
python main_vgg16_0.5.py
```
