## MsMorph: An Unsupervised pyramid learning network for brain image registration



Registration: Aligning two images to make them similar  

How would a human perform manual registration?  

Typically, they would repeatedly compare the images, identify areas of dissimilarity, and make adjustments to enhance their similarity. Based on this approach, we propose MsMorph:

<img src="https://github.com/GaodengFan/MsMorph/blob/main/img/network.png" alt="image-20241028120154973" style="zoom: 80%;" />

<img src="https://github.com/GaodengFan/MsMorph/blob/main/img/Differential%20Def.png" style="zoom: 80%;" />

### Datasets

LPBA [[link\]](https://resource.loni.usc.edu/resources/atlases-downloads/)

Mindboggle [[link\]](https://osf.io/yhkde/)

### Environment setup:

1. buildup on V100 and cuda11.4
2. related python libraries:

- torch ==1.12.1+cu113
- torchaudio==0.12.1+cu113
- torchvision==0.13.1+cu113
- matplotlib==3.8.4
