# Bidirectional-Modulation Frequency-Heterogeneous Network<br>       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for Remote Sensing Image Dehazing (IEEE TCSVT 2025)



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

# Visualization Results

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






