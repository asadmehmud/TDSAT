'''
real world image denosing
'''
import argparse
from hsi_setup import Engine, train_options
from utility import *
from torchvision.transforms import Compose
import torch
from torch.utils.data import DataLoader
import warnings
import os

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    """Train Settings"""
    parser = argparse.ArgumentParser(description="Hyperspectral Image Denosing")
    opt = train_options(parser)
    opt.no_log = True
    opt.no_ropt = True
    cuda = not opt.no_cuda
    print(f'opt settings: {opt}')

    """Set Random Status"""
    seed_everywhere(opt.seed)

    """Setup Engine"""
    engine = Engine(opt)

    # Urban
    # mat_dataset = MatDataFromFolder('./data/real', fns=['Urban.mat'])
    # key = 'croppedData'

    # Artificial
    mat_dataset = MatDataFromFolder('/kaggle/input/hsi-denoising-artificial-dataset-mat-files', fns=['avcup97refcomb_coloredSigmaSImin4Times1Same_datamat3D_small.mat'])
    #mat_dataset = MatDataFromFolder('/content/drive/MyDrive/Colab/HSI/Denoising/Data/artificial/', fns=['avcup97refcomb_smaller_coloredSigmaSImin5Times100Same_datamat3D.mat'])
    
    key = 'signal_art_noisy3D'

    saveimgdir = f'./'

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatKey(key=key),
            lambda x: x[...][None],
            minmax_normalize,
        ])
    else:
        mat_transform = Compose([
            LoadMatKey(key=key),
            minmax_normalize,
        ])

    mat_dataset = TransformDataset(mat_dataset, mat_transform)
    mat_loader = DataLoader(
                    mat_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=cuda)


    engine.image_denosing(mat_loader, saveimgdir=saveimgdir)


