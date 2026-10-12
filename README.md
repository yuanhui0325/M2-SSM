从你发来的截图来看，最大的问题**不是需要换“字体（Font）”**，而是缺少了 **Markdown 排版标记**。

目前你的主页看起来全是普通的平铺文字，没有层级感，特别是**“运行代码”和“普通文字”混在了一起**，这在开源项目中是大忌（别人没法一键复制你的代码）。

在 GitHub 上，我们不需要手动调字体和字号，而是用几个**简单的符号**就能让排版瞬间变得极其高级。

我把你截图里的内容用标准的 Markdown 语法重新排版了，**你直接点击右上角的“编辑（Edit / 铅笔图标）”，把下面这段代码全部复制进去覆盖掉你原来的内容**：

***

### 👇 请直接复制下面这个灰框里的所有代码：

```markdown
# M²-SSM: Mask-Modulated Mamba-based Quality Enhancement for JPEG Compressed Screen Content Image

## Introduction
M²-SSM is a lightweight Mamba-based network for quality enhancement of JPEG-compressed Screen Content Images (SCIs). It aims to suppress blocking and ringing artifacts while restoring sharp text edges and long-range geometric structures with efficient global modeling.

## Dataset Preparation
We use the large-scale SCI dataset SDU-SCICD10K for training and testing. Download the dataset from [SunshineSki/Screen-content-image-dataset](https://github.com/SunshineSki/Screen-content-image-dataset).

## Usage

### Training
```bash
cd enhancement 
python train.py \
  --train_dir ./dataset/SDU-SCICD10K/train \
  --val_dir ./dataset/SDU-SCICD10K/val \
  --batch_size 8 \
  --epochs 100 \
  --lr 2e-4
```

### Testing
```bash
cd enhancement 
python test.py \
  --test_dir ./dataset/SDU-SCICD10K/test \
  --checkpoint ./checkpoints/sci_best.pth \
  --qualities 10 20 30 40
```

## Citation
If M²-SSM is useful for your research, please consider citing:

```bibtex
@inproceedings{li2026m, 
  title={M2-SSM: Mask-Modulated Mamba-based Quality Enhancement for JPEG Compressed Screen Content Image}, 
  author={Li, Tuo and Li, Chaofei and Yang, Jingwen and Sang, Wusi and Yuan, Hui}, 
  booktitle={2026 International Symposium on Machine Learning and Media Computing (MLMC)}, 
  pages={1--6}, 
  year={2026}, 
  organization={IEEE} 
}
```
```
