# NPF-200: A Multi-Modal Eye Fixation Dataset and Method for Non-Photorealistic Videos
<figure class="third">
    <img src="https://github.com/Yangziyu/NPF200/blob/main/img/teaser1.jpg">
    <img src="https://github.com/Yangziyu/NPF200/blob/main/img/teaser2.jpg">
    <img src="https://github.com/Yangziyu/NPF200/blob/main/img/teaser3.jpg">
</figure>


## Description
This is the official implementation of the ACM MM 2023 paper "NPF-200: A Multi-Modal Eye Fixation Dataset and Method for Non-Photorealistic Videos". Paper can be downloaded from [here](https://arxiv.org/pdf/2308.12163v1.pdf).

## NPF-200 Dataset
The dataset (NPF200) can be downloaded from [here](https://figshare.com/s/9b45d1bdc790db3ee843) .

## Some Results
<center>
    <img src="https://github.com/Yangziyu/NPF200/blob/main/img/results1.png">
</center>
<figure class="half">
    <img src="https://github.com/Yangziyu/NPF200/blob/main/img/results2.png">
    <img src="https://github.com/Yangziyu/NPF200/blob/main/img/results3.png">
</figure>


## Video Demo
Please see our [video demo](https://www.youtube.com/watch?v=r4XWogTQEzc) on Youtube.

## Requirements
- Linux
- Python 3
- pytorch >= 1.7.1 and torchvision >= 0.8.2

## Pretrained Models

## Train
```python
python train.py --dataset [myDataset/DHF1KDataset] --model_val_path YOUR_MODEL_SAVE_PATH --load_weight PRETRAINED_MODEL_PATH
```

## Test

## License
Software Copyright License for non-commercial scientific research purposes. Please read carefully the [terms and conditions](https://github.com/Yangziyu/NPF200/blob/main/LICENSE.md) in the LICENSE file and any accompanying documentation before you download and/or use the NPF-200 dataset, model and software, (the "Data & Software"), including code, images, videos, textures, software, scripts, and animations. By downloading and/or using the Data & Software (including downloading, cloning, installing, and any other use of the corresponding github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Data & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](https://github.com/Yangziyu/NPF200/blob/main/LICENSE.md).




