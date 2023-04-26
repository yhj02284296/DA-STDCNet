
 BiSeSTDC Network for Real-Time  Small Object Semantic Segmentation

## Overview

<p align="center">
  <img src="image/BiSeSTDC architecture.png" alt="overview-of-our-method" width="600"/></br>
  <span align="center">Speed-Accuracy performance comparison on the Cityscapes test set</span> 
</p>
We present BiSeSTDC-Seg, an mannully designed semantic segmentation network with not only state-of-the-art performance but also faster speed than current methods.

Highlights:

* **BiSeSTDC Net**: a novel and efficient network named BiSeSTDC(Bilateral Segmentation ShortTerm Dense Concatenate) to improve the performance of small object real-time semantic segmentation.
* **Small Object Sensitive**: Effectiveness of Small-Object Sensitive improvement.
* **SOTA**: BiSeSTDC achieves extremely fast speed  and maintains competitive accuracy.


<p align="center">
<img src="images/comparison-cityscapes.png" alt="Cityscapes" width="400"/></br>
</p>

## Methods


## Prerequisites

- Pytorch 1.1
- Python 3.5.6
- NVIDIA GPU
- TensorRT v5.1.5.0 (Only need for testing inference speed)
BIiSeSTDCNePre
This repository has been trained on Tesla V100. Configurations (e.g batch size, image patch size) may need to be changed on different platforms. Also, for fair competition, we test the inference speed on NVIDIA GTX 1080Ti.

## Installation

* Clone this repo:

```bash
git clone https://github.com/bearking79/BiSeSTDC
cd BiSeSTDC
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

* Install [PyCuda](https://wiki.tiker.net/PyCuda/Installation) which is a dependency of TensorRT.
* Install [TensorRT](https://github.com/NVIDIA/TensorRT) (v5.1.5.0): a library for high performance inference on NVIDIA GPUs with [Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html#python).

## Usage

### 0. Prepare the dataset

* Download the [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) from the Cityscapes.
* Link data to the  `data` dir.

  ```bash
  ln -s /path_to_data/cityscapes/gtFine data/gtFine
  ln -s /path_to_data/leftImg8bit data/leftImg8bit
  ```

### 1. Train BiSeSTDC-Seg

Note: Backbone BiSeSTDC denotes BiSeSTDC, STDCNet813 denotes STDC1, STDCNet1446 denotes STDC2.

* Train BiSeSTDC:

```bashBIiSeSTDCNePre
export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch \
--nproc_per_node=2 train.py \
--respath checkpoints/BiSeSTDC_seg/ \
--backbone BiSeSTDCNet \
--mode val \
--n_workers_train 2 \
--n_workers_val 1 \
--max_iter 160000 \
--n_img_per_gpu 6 \
--use_boundary_8 true \
--pretrain_path checkpoints/BiSeSTDCNePre.tar
```

We will save the model's params in model_maxmIOU50.pth for input resolution 512x1024，and model_maxmIOU75.pth for input resolution 768 x 1536.

ImageNet Pretrained STDCNet Weights for training and Cityscapes trained STDC-Seg weights for evaluation:

###

### 2. Evaluation

Here we use our pretrained BiSeSTDCSeg as an example for the evaluation.

* Choose the evaluation model in evaluation.py:

```python
#BiSeSTDC-Seg50 mIoU 0.746
evaluatev0('./checkpoints/BiSeSTDC/model_maxmIOU50.pth', dspth='./data', backbone='BiSeSTDC', scale=0.5, 
           use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

#STDC1-Seg75 mIoU 0.776
evaluatev0('./checkpoints/BiSeSTDC/model_maxmIOU75.pth', dspth='./data', backbone='BiSeSTDC', scale=0.75, 
           use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)


* Start the evaluation process:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

### 3. Latency

#### 3.0 Latency measurement tools

* If you have successfully installed [TensorRT](https://github.com/chenwydj/FasterSeg#installation), you will automatically use TensorRT for the following latency tests (see [function](https://github.com/chenwydj/FasterSeg/blob/master/tools/utils/darts_utils.py#L167) here).
* Otherwise you will be switched to use Pytorch for the latency tests  (see [function](https://github.com/chenwydj/FasterSeg/blob/master/tools/utils/darts_utils.py#L184) here).

#### 3.1 Measure the latency of the FasterSeg

* Choose the evaluation model in run_latency:

```python
# BiSeSTDC-50 234 FPS on NVIDIA GTX 1080Ti
backbone = 'BiSeSTDC'
methodName = 'BiSeSTDC'
inputSize = 512
inputScale = 50
inputDimension = (1, 3, 512, 1024)

# BiSeSTDC-75 136FPS on NVIDIA GTX 1080Ti
backbone = 'BiSeSTDC'
methodName = 'BiSeSTDC'
inputSize = 768
inputScale = 75
inputDimension = (1, 3, 768, 1536)

```

* Run the script:

```bash
CUDA_VISIBLE_DEVICES=0 python run_latency.py
```

## Citation

```

```

## Acknowledgement
