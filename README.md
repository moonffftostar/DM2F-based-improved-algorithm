# DM2F Based Improved Algorithm

This repo's baseline is
"[Deep Multi-Model Fusion for Single-Image Dehazing](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Deep_Multi-Model_Fusion_for_Single-Image_Dehazing_ICCV_2019_paper.pdf)"

By Zijun Deng, Lei Zhu, Xiaowei Hu, Chi-Wing Fu, Xuemiao Xu, Qing Zhang, Jing Qin, and Pheng-Ann Heng.

(ICCV 2019), written by Zijun Deng at the South China University of Technology.

## Results

The dehazing results can be found at 
[Google Drive](https://drive.google.com/drive/folders/1ZVBI_3Y2NthVLeK7ODMIB5vRjmN9payF?usp=sharing).

## Installation & Preparation

Make sure you have `Python>=3.7` installed on your machine.

**Environment setup:**

1. Create conda environment

       conda create -n dm2f
       conda activate dm2f

2. Install dependencies (test with PyTorch 1.8.0):

   1. Install pytorch==1.8.0 torchvision==0.9.0 (via conda, recommend).

   2. Install other dependencies

          pip install -r requirements.txt

* Prepare the dataset

   * Download the RESIDE dataset from the [official webpage](https://sites.google.com/site/boyilics/website-builder/reside).

   * Download the O-Haze dataset from the [official webpage](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

   * Make a directory `./data` and create a symbolic link for uncompressed data, e.g., `./data/RESIDE`.

## Training

1. Set the path of datasets in tools/config.py
2. Run by ```python train.py```

Use pretrained Swin_t from torchvision.

*Hyper-parameters* of training were set at the top of *train.py*, and you can conveniently
change them as you need.

Training a model on a single RTX3090 GPU takes about 4 hours.

## Testing

1. Set the path of five benchmark datasets in tools/config.py.
2. Put the trained model in `./ckpt/`.
2. Run by ```python test.py```

*Settings* of testing were set at the top of `test.py`, and you can conveniently
change them as you need.

