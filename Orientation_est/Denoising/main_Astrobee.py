import os
import torch
import src.learning as lr
import src.networks as sn
#import src.networks as sn

import src.losses as sl
import src.dataset as ds
import numpy as np
from pathlib import Path


def load_str_list(txt_path: str):
    p = Path(txt_path)
    text = p.read_text(encoding="utf-8").strip()

    items = []
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip().strip(',').strip('"').strip("'")
        if line:
            items.append(line)
    return items

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = '/home/mpil/Downloads/AstrobeePublic/'
predata_dir = '/home/mpil/miniconda3/envs/Denoising/data/AstrobeePublic'
train_list_txt = '/home/mpil/astrobee_dataset_revised/train.txt'
test_list_txt = '/home/mpil/astrobee_dataset_revised/test.txt'
val_list_txt = '/home/mpil/astrobee_dataset_revised/val.txt'
all_list_txt = '/home/mpil/astrobee_dataset_revised/all.txt'
# test a given network
# address = os.path.join(base_dir, 'results/EUROC/2020_02_18_16_52_55/')
# or test the last trained network
address = "last"
################################################################################
# Network parameters
################################################################################
net_class = sn.GyroNet
net_params = {
    'in_dim': 6,
    'out_dim': 3,
    'c0': 16,
    'dropout': 0.1,
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
    'gyro_std': [0.2*np.pi/180, 0.2*np.pi/180, 0.2*np.pi/180], # data augmentation
}
################################################################################
# Dataset parameters
################################################################################
dataset_class = ds.ASTROBEEDataset
dataset_params = {
    # where are raw data ?
    'data_dir': data_dir,
    # where record preloaded data ?
    #'predata_dir': os.path.join(base_dir, 'data/EUROC'),
    'predata_dir': predata_dir,
    # set train, val and test sequence
    # size of trajectory during training
    'N': 32 * 90 , # should be integer * 'max_train_freq' 32 * 500 4480
    'min_train_freq': 16,
    'max_train_freq': 32,
}
'''dataset_params['train_seqs'] = load_str_list(train_list_txt)
dataset_params['test_seqs'] = load_str_list(all_list_txt)
#dataset_params['test_seqs'] = load_str_list(test_list_txt)
dataset_params['val_seqs'] = load_str_list(val_list_txt)'''
publicdata_path = '/home/mpil/Downloads/AstrobeePublic/'

dataset_params['train_seqs'] = load_str_list('/home/mpil/Downloads/AstrobeePublic_top_level_folders_named.txt')
dataset_params['test_seqs'] = load_str_list('/home/mpil/Downloads/AstrobeePublic_top_level_folders_named.txt')
dataset_params['val_seqs'] = load_str_list('/home/mpil/Downloads/AstrobeePublic_top_level_folders_named.txt')
################################################################################
# Training parameters
################################################################################
train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,
        'weight_decay': 1e-1,
        'amsgrad': False,
    },
    'loss_class': sl.GyroLoss,
    'loss': {
        'min_N': int(np.log2(dataset_params['min_train_freq'])),
        'max_N': int(np.log2(dataset_params['max_train_freq'])),
        'w':  1e6,
        'target': 'rotation matrix', #target': 'rotation matrix
        'huber': 0.005,
        'dt': 0.005,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 120, #600
        'T_mult': 2,
        'eta_min': 1e-5,
    },
    'dataloader': {
        'batch_size': 8,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
    },
    # frequency of validation step
    'freq_val': 120,
    # total number of epochs
    'n_epochs': 360,
    # where record results ?
    'res_dir': os.path.join(base_dir, "results/AstrobeePublic"),
    # where record Tensorboard log ?
    'tb_dir': os.path.join(base_dir, "results/runs/AstrobeePublic"),
}
################################################################################
# Train on training data set
################################################################################
learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, net_params, None,
    train_params['loss']['dt'])
learning_process.train(dataset_class, dataset_params, train_params)
################################################################################
# Test on full data set
################################################################################
learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, net_params, address=address,
    dt=train_params['loss']['dt'])
learning_process.test(dataset_class, dataset_params, ['test'])

