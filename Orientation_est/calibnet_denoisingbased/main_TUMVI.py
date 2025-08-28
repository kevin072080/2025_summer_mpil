import os
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = '/home/mpil/miniconda3/envs/Denoising/TUMVI_dataset'
# test a given network
# address = os.path.join(base_dir, 'results/TUM/2020_02_18_16_26_33')
# or test the last trained network
address = 'last'
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
dataset_class = ds.TUMVIDataset
dataset_params = {
    # where are raw data ?
    'data_dir': data_dir,
    # where record preloaded data ?
    'predata_dir': os.path.join(base_dir, 'data/TUM'),
    # set train, val and test sequence
    'train_seqs': [
        'dataset-room1_512_16',
        'dataset-room3_512_16',
        'dataset-room5_512_16',
        ],
    'val_seqs': [
        'dataset-room2_512_16',
        'dataset-room4_512_16',
        'dataset-room6_512_16',
        ],
    'test_seqs': [
        'dataset-room2_512_16',
        'dataset-room4_512_16',
        'dataset-room6_512_16'
        ],
    # size of trajectory during training -> local window size X, 50s * 200Hz = 10000?
    'N': 9984, # should be integer * 'max_train_freq' 32 * 500 9984 (dataset 길이 자를 때 활용됨 -> train은 50초라고 명시됐으므로 9984가 맞는 듯)
    'min_train_freq': 16,
    'max_train_freq': 32,
}
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
        'target': 'rotation matrix mask',
        #'target': 'rotation matrix',
        'huber': 0.005,
        'dt': 0.005,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 600,
        'T_mult': 2,
        'eta_min': 1e-3,
    },
    'dataloader': {
        'batch_size': 8, # 총 타임스탬프는 정해져 있음 -> 마지막 부분 잘리지 않도록 타임스탬프의 약수로 배치 사이즈 지정
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
    },
    # frequency of validation step
    'freq_val': 600,
    # total number of epochs
    'n_epochs': 1800,
    # where record results ?
    'res_dir': os.path.join(base_dir, "results/TUM"),
    # where record Tensorboard log ?
    'tb_dir': os.path.join(base_dir, "results/runs/TUM"),
}
################################################################################
# Train on training data set
################################################################################
#learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
#    train_params['tb_dir'], net_class, net_params, None,
#    train_params['loss']['dt'])
#learning_process.train(dataset_class, dataset_params, train_params)
################################################################################
# Test on full data set
################################################################################
learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, net_params, address=address,
    dt=train_params['loss']['dt'])
learning_process.test(dataset_class, dataset_params, ['test'])