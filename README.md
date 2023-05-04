# iColoriT, SIGGRAPH, ECCV Non-Official Implementation

This is the Non-official PyTorch implementation of the Papers:<br>
[iColoriT: Towards Propagating Local Hint to the Right Region in Interactive Colorization by Leveraging Vision Transformer](https://arxiv.org/abs/2207.06831)<br>
[SIGGRAPH: Real-Time User-Guided Image Colorization with Learned Deep Priors](https://arxiv.org/abs/1705.02999)<br>
[ECCV: Colorful Image Colorization](http://arxiv.org/abs/1603.08511)

<p align="center">
  <img width="90%" src="iColoriT/docs/iColoriT_demo.gif">
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/icolorit-towards-propagating-local-hint-to/point-interactive-image-colorization-on)](https://paperswithcode.com/sota/point-interactive-image-colorization-on?p=icolorit-towards-propagating-local-hint-to)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/icolorit-towards-propagating-local-hint-to/point-interactive-image-colorization-on-1)](https://paperswithcode.com/sota/point-interactive-image-colorization-on-1?p=icolorit-towards-propagating-local-hint-to)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/icolorit-towards-propagating-local-hint-to/point-interactive-image-colorization-on-cub)](https://paperswithcode.com/sota/point-interactive-image-colorization-on-cub?p=icolorit-towards-propagating-local-hint-to)  

> **iColoriT: Towards Propagating Local Hint to the Right Region in Interactive Colorization by Leveraging Vision Transformer**  
> Jooyeol Yun*, Sanghyeon Lee*, Minho Park*, and Jaegul Choo  
> KAIST  
> In WACV 2023. (* indicate equal contribution)

> Paper: https://arxiv.org/abs/2207.06831  
> Project page: https://pmh9960.github.io/research/iColoriT/

> **Abstract:** *Point-interactive image colorization aims to colorize grayscale images when a user provides the colors for specific locations. It is essential for point-interactive colorization methods to appropriately propagate user-provided colors (i.e., user hints) in the entire image to obtain a reasonably colorized image with minimal user effort. However, existing approaches often produce partially colorized results due to the inefficient design of stacking convolutional layers to propagate hints to distant relevant regions. To address this problem, we present iColoriT, a novel point-interactive colorization Vision Transformer capable of propagating user hints to relevant regions, leveraging the global receptive field of Transformers. The self-attention mechanism of Transformers enables iColoriT to selectively colorize relevant regions with only a few local hints. Our approach colorizes images in real-time by utilizing pixel shuffling, an efficient upsampling technique that replaces the decoder architecture. Also, in order to mitigate the artifacts caused by pixel shuffling with large upsampling ratios, we present the local stabilizing layer. Extensive quantitative and qualitative results demonstrate that our approach highly outperforms existing methods for point-interactive colorization, producing accurately colorized images with a user's minimal effort.*


## Demo 🎨

Try colorizing images yourself with the [demo software](https://github.com/pmh9960/iColoriT/tree/main/iColoriT_demo)!

## Pretrained iColoriT

Checkpoints for iColoriT models are available in the links below.

|  	| Backbone 	| Link 	|
|:---:	|:---:	|:---:	|
| iColoriT-ImageNet	| ViT-B 	| [iColoriT-ImageNet (Google Drive)](https://drive.google.com/file/d/16i9ulB4VRbFLbLlAa7UjIQR6J334BeKW/view?usp=sharing)	|
| iColoriT-CUB 	| ViT-B 	| [iColoriT-CUB (Google Drive)](https://drive.google.com/file/d/1yKwFTQGDBvr9B7NIyXhxQH0K-BNlCs4L/view?usp=sharing) 	|
| iColoriT-OxfordFlowers 	| ViT-B 	| [iColoriT-OxfordFlowers (Google Drive)](https://drive.google.com/file/d/1GMmjfxAoM95cABwlZD8555WxI7nmIZrR/view?usp=sharing)	|
| ECCV-CUB 	| CNN 	| [ECCV-CUB (Google Drive)](https://drive.google.com/file/d/1GMmjfxAoM95cABwlZD8555WxI7nmIZrR/view?usp=sharing)	|
| ECCV-OxfordFlowers 	| CNN 	| [ECCV-OxfordFlowers (Google Drive)](https://drive.google.com/file/d/1GMmjfxAoM95cABwlZD8555WxI7nmIZrR/view?usp=sharing)	|
| SIGGRAPH-CUB 	| CNN 	| [SIGGRAPH-CUB (Google Drive)](https://drive.google.com/file/d/1GMmjfxAoM95cABwlZD8555WxI7nmIZrR/view?usp=sharing)	|
| SIGGRAPH-OxfordFlowers 	| CNN 	| [SIGGRAPH-OxfordFlowers (Google Drive)](https://drive.google.com/file/d/1GMmjfxAoM95cABwlZD8555WxI7nmIZrR/view?usp=sharing)	|

## Testing

### Installation

Our code is implemented in Python 3.8, torch>=1.8.2
```
git clone https://github.com/yhsharaf/MBZUAI-AI702.git
pip install -r requirements.txt
```

### Testing iColoriT

You can generate colorization results when iColoriT is provided with randomly selected groundtruth hints from color images. 
Please fill in the path to the model checkpoints and validation directories in the scripts/infer.sh file.

```
bash scripts/infer.sh
```

Then, you can evaluate the results by running

```
bash scripts/eval.sh
```
Randomly sampled hints used in our paper is available in this [link](https://drive.google.com/file/d/1MVU5Ze5FbT0Kp14bJcfDANY17lgkNmwn/view?usp=sharing)

The codes used for randomly sampling hint locations can be seen in hint_generator.py

## Training

First prepare an official ImageNet dataset with the following structure. 

```
train
 └ id1
   └ image1.JPEG
   └ image2.JPEG
   └ ...
 └ id2
   └ image1.JPEG
   └ image2.JPEG
   └ ...     

```

Please fill in the train/evaluation directories in the scripts/train.sh file and execute

```
bash scripts/train.sh
```




## Citations

```
@InProceedings{Yun_2023_WACV,
    author    = {Yun, Jooyeol and Lee, Sanghyeon and Park, Minho and Choo, Jaegul},
    title     = {iColoriT: Towards Propagating Local Hints to the Right Region in Interactive Colorization by Leveraging Vision Transformer},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {1787-1796}
}
```
```
@article{DBLP:journals/corr/ZhangZIGLYE17,
  author       = {Richard Zhang and
                  Jun{-}Yan Zhu and
                  Phillip Isola and
                  Xinyang Geng and
                  Angela S. Lin and
                  Tianhe Yu and
                  Alexei A. Efros},
  title        = {Real-Time User-Guided Image Colorization with Learned Deep Priors},
  journal      = {CoRR},
  volume       = {abs/1705.02999},
  year         = {2017},
  url          = {http://arxiv.org/abs/1705.02999},
  eprinttype    = {arXiv},
  eprint       = {1705.02999},
  timestamp    = {Wed, 14 Aug 2019 08:23:33 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/ZhangZIGLYE17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```
@article{DBLP:journals/corr/ZhangIE16,
  author       = {Richard Zhang and
                  Phillip Isola and
                  Alexei A. Efros},
  title        = {Colorful Image Colorization},
  journal      = {CoRR},
  volume       = {abs/1603.08511},
  year         = {2016},
  url          = {http://arxiv.org/abs/1603.08511},
  eprinttype    = {arXiv},
  eprint       = {1603.08511},
  timestamp    = {Wed, 14 Aug 2019 08:23:33 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/ZhangIE16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
