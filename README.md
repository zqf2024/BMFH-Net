# &nbsp;&nbsp; Bidirectional-Modulation Frequency-Heterogeneous  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Network for Remote Sensing Image Dehazing<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(IEEE TCSVT 2025)

## Abstract

Abstract—Recently, deep neural networks have been exten-sively explored in remote sensing image haze removal and achieved remarkable performance. However, existing methods
fail to effectively fuse the features extracted from Convolu-tional Neural Networks (CNNs) and Transformer networks, leading to performance degradation. Moreover, most dehazing methods lack further exploration of the distinct properties of high- and low-frequency features, which are crucial for texture restoration and haze removal. To address these issues, we propose a Bidirectional-Modulation Frequency-Heterogeneous Network (BMFH-Net). Speciffcally, we propose a Differential-Expert Guided Bidirectional Modulation (DGBM) module that incorporates Differential experts and physical inversion models to exploit the complementarity of CNN-Transformer features and extract their latent haze-related physical characteristics, thereby enabling more effective bidirectional alignment. Furthermore, a Wavelet Frequency Heterogeneous Enhancement (WFHE) Mod-ule is designed to capture the most representative high-frequency features to reffne image texture details, while enhancing the global perception of haze and reconstructing structural information during low-frequency processing. Experiments on challenging remote sensing image datasets demonstrate our BMFH-Net outperforms several state-of-the-art haze removal methods.

## Overall architecture
![main.jpg](images/main.jpg)

## Differential-Expert Guided Bidirectional Modulation module
![DGBM.jpg](images/DGBM.jpg)

## Wavelet Frequency Heterogeneous Enhancement module
![WFHE.jpg](images/WFHE.jpg)

## Quantitative results🔥
<span style="font-size:30px"><b>COMPARISON OF OUR METHOD AGAINST OTHERS ON THE STATEHAZE1K DATASET. ↑ INDICATES HIGHER IS BETTER. THE RED,
GREEN AND BLUE FONTS REPRESENT THE FIRST, SECOND AND THIRD RANKINGS IN TERMS OF PSNR OR SSIM.</b></span><br>
![table1.jpg](images/table1.jpg)

<span style="font-size:30px"><b>COMPARISON RESULTS OF OUR METHOD WITH OTHER ADVANCED METHODS ON THE RICE AND RSID DATASETS.↑ INDICATES HIGHER IS BETTER. THE RED,
GREEN AND BLUE FONTS REPRESENT THE FIRST, SECOND AND THIRD RANKINGS IN TERMS OF PSNR OR SSIM.</b></span><br>
![table2.jpg](images/table2.jpg)

## Qualitative results🔥

![thick.jpg](images/thick.jpg)
### Results on Haze1k-thick remote sensing Dehazing Challenge testing images  
<br>

![RICE1.jpg](images/RICE1.jpg)
### Results on RICE1 remote sensing Dehazing Challenge testing images  
<br>

![RICE2.jpg](images/RICE2.jpg)
### Results on RICE2 remote sensing Dehazing Challenge testing images  
<br>

![RISD.jpg](images/RSID.jpg)
### Results on RSID remote sensing Dehazing Challenge testing images

### Dependences

1.Pytorch 1.8.0  
2.python 3.8  
3.CUDA 11.7  
4.Ubuntu 18.04

### Datasets Preparation
>./dataset/dataset_name/train
>>clean<br>
>>hazy

>./dataset/dataset_name/test
>>clean<br>
>>hazy

>./output_result


### 1.Train 
<div style="display: flex; justify - content: center; align - items: center; height: 100vh;">
  <pre style="background - color: lightgray;"><code>
  python train.py --type 1 -train_batch_size 4 --gpus 0
  </code></pre>
</div>

### 2.Test 
<div style="display: flex; justify - content: center; align - items: center; height: 100vh;">
  <pre style="background - color: lightgray;"><code>
  python test.py --type 1  --gpus 0
  </code></pre>
</div>

### 3.Clone the repo
<div style="display: flex; justify - content: center; align - items: center; height: 100vh;">
  <pre style="background - color: lightgray;"><code>
  it clone https://github.com/zqf2024/BFMT-Net.git
  </code></pre>
</div>








