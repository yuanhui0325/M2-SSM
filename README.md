M²-SSM: Mask-Modulated Mamba-based Quality Enhancement for JPEG Compressed Screen Content Image
Introduction
M²-SSM is a lightweight Mamba-based network for quality enhancement of JPEG-compressed Screen Content Images (SCIs). It aims to suppress blocking and ringing artifacts while restoring sharp text edges and long-range geometric structures with efficient global modeling.
Dataset Preparation
We use the large-scale SCI dataset SDU-SCICD10K for training and testing.
Download the dataset from SunshineSki/Screen-content-image-dataset.
Usage
Training
cd enhancement 
python train.py \ --train_dir ./dataset/SDU-SCICD10K/train \ --val_dir ./dataset/SDU-SCICD10K/val \ --batch_size 8 \ --epochs 100 \ --lr 2e-4
Testing
cd enhancement 
python test.py \ --test_dir ./dataset/SDU-SCICD10K/test \ --checkpoint ./checkpoints/sci_best.pth \ --qualities 10 20 30 40
Citation
If FD-SCU is useful for your research, please consider citing:
@inproceedings{li2026m,
  title={M 2-SSM: Mask-Modulated Mamba-based Quality Enhancement for JPEG Compressed Screen Content Image},
  author={Li, Tuo and Li, Chaofei and Yang, Jingwen and Sang, Wusi and Yuan, Hui},
  booktitle={2026 International Symposium on Machine Learning and Media Computing (MLMC)},
  pages={1--6},
  year={2026},
  organization={IEEE}
}
