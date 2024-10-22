# Adaptive Selection based Referring Image Segmentation
This is an official PyTorch implementation of ASDA (accepted by ACMMM 2024).
## News
- [July 16, 2024] The paper is accepted by ACMMM 2024🎉.
- [Oct 22, 2024] Pytorch implementation of ASDA is released.

## Main Results

Main results on RefCOCO

Model | Backbone | val | test A | test B |
--- | ---- |:-------------:| :-----:|:-----:|
CRIS| ResNet101 | 70.47 | 73.18 | 66.10 |
ASDA| ViT-B | 75.06  | 77.14 | 71.36 |

Main results on RefCOCO+

Model | Backbone | val | test A | test B |
--- | ---- |:-------------:| :-----:|:-----:|
CRIS| ResNet101 | 62.27 | 68.08 | 53.68 |
ASDA| ViT-B | 66.84  | 71.13 | 57.83 |

Main results on G-Ref

Model | Backbone | val(U) | test(U) | val(G) |
--- | ---- |:-------------:| :-----:| :-----:|
CRIS| ResNet101 | 59.87 | 60.36 | - |
ASDA| ViT-B | 65.73  | 66.45 |  63.55

## Quick Start
### Environment preparation
```bash
conda create -n ASDA python=3.6 -y
conda activate ASDA
# install pytorch according to your cuda version
# don't change version of torch, or it may occur conflict
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge 

pip install -r requirements.txt 
```

### Dataset Preparation

#### 1. Download the COCO train2014 to ASDA/ln_data/images.
```bash
wget https://pjreddie.com/media/files/train2014.zip
```

#### 2. Download the RefCOCO, RefCOCO+, RefCOCOg to ASDA/ln_data.
```bash
mkdir ln_data && cd ln_data
# The original link bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip is no longer valid, we have uploaded it to Google Drive (https://drive.google.com/file/d/1AnNBSL1gc9uG1zcdPIMg4d9e0y4dDSho/view?usp=sharing)
wget 'https://drive.usercontent.google.com/download?id=1AnNBSL1gc9uG1zcdPIMg4d9e0y4dDSho&export=download&authuser=0&confirm=t&uuid=be656478-9669-4b58-ab23-39f196f88c07&at=AN_67v3n4xwkPBdEQ9pMlwonmhrH%3A1729591897703' -O refcoco_all.zip
unzip refcoco_all.zip
```

#### 3. Run data.sh to generate the annotations.
```bash
mkdir dataset && cd dataset
bash data.sh
```

### Training & Testing
```bash
bash train.sh 0,1
bash test.sh 0
```
## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Acknowledgement
Thanks for a lot of codes from [CRIS](https://github.com/DerrickWang005/CRIS.pytorch), [VLT](https://github.com/henghuiding/Vision-Language-Transformer), [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{yue2024adaptive,
  title={Adaptive Selection based Referring Image Segmentation},
  author={Yue, Pengfei and Lin, Jianghang and Zhang, Shengchuan and Hu, Jie and Lu, Yilin and Niu, Hongwei and Ding, Haixin and Zhang, Yan and JIANG, GUANNAN and Cao, Liujuan and others},
  booktitle={ACM Multimedia 2024}
}
```