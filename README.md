# Say Anything with Any Style

This repository provides the official PyTorch implementation for the following paper:<br>
**[Say Anything with Any Style](https://arxiv.org/abs/2403.06363)**<br>
[Shuai Tan](https://scholar.google.com.hk/citations?user=9KjKwDwAAAAJ&hl=zh-CN), et al.<br>
In AAAI, 2024.<br>

![visualization](demo/teaser.jpg)

Given a source image and a style reference clip, SAAS generates stylized talking faces driven by audio. The lip motions are synchronized with the audio, while the speaking styles are controlled by the style clips. We also support video-driven style editing by inputting a source video. The pipeline of our SAAS is as follows:

![visualization](demo/pipeline.svg)

## Requirements
We train and test based on Python 3.8 and Pytorch. To install the dependencies run:
```
conda create -n SAAS python=3.8
conda activate SAAS
```

- python packages
```
pip install -r requirements.txt
```

## Inference

- Run the demo in audio-driven setting:
    ```bash
    python audio_driven/train_test/inference.py --img_path path/to/image --wav_path path/to/audio --img_3DMM_path path/to/img_3DMM --style_path path/to/style --save_path path/to/save
    ```
  The result will be stored in save_path.

- Run the demo in video-driven setting:
    ```bash
    python video_driven/inference.py --img_path path/to/image --wav_path path/to/audio --video_3DMM_path path/to/video_3DMM --style_path path/to/style --save_path path/to/save
    ```
  The result will be stored in save_path.

  img_path used should be first cropped using scripts [crop_image.py](data_preprocess/crop_image.py)

- Download [checkpoints](https://drive.google.com/file/d/1bZYQZF1Ftm_BDWvq899KY2MolSXktHYl/view?usp=drive_link) for video-driven setting and put them into ./checkpoints.
- Our audio encoder can be viewed as the combination of [SadTalker' Audio encoder](https://github.com/OpenTalker/SadTalker) and our video-encoder. You can download the checkpoint of [SadTalker' Audio encoder](https://github.com/OpenTalker/SadTalker) and our video-encoder to support audio-driven setting.

## Acknowledgement
Some code are borrowed from following projects:
* [Learning2Listen](https://github.com/evonneng/learning2listen)
* [PIRenderer](https://github.com/RenYurui/PIRender)
* [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
* [SadTalker](https://github.com/OpenTalker/SadTalker)
* [Style-ERD](https://github.com/tianxintao/Online-Motion-Style-Transfer)
* [GFPGAN](https://github.com/TencentARC/GFPGAN)
* [FOMM video preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing)

Thanks for their contributions!

* We would like to thank [Xinya Ji](https://scholar.google.com.hk/citations?user=sy_WtmcAAAAJ&hl=zh-CN&oi=ao), [Yifeng Ma](https://scholar.google.com.hk/citations?user=0SxgRqoAAAAJ&hl=zh-CN&oi=ao) and Zhiyao Sun for their generous help.

## Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@inproceedings{tan2024say,
  title={Say Anything with Any Style},
  author={Tan, Shuai and Ji, Bin and Ding, Yu and Pan, Ye},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={5088--5096},
  year={2024}
}
```