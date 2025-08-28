from src.utils import pdump, pload, bmtv, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import os
from os import path as osp
import torch
import sys
import json
from pathlib import Path
from typing import Union, List  # 3.8 호환

def save_str_list(path: Union[str, Path], items: List[str]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
    # 한 줄에 하나씩 기록(+ 마지막 줄도 개행)
    text = "\n".join(str(s).strip() for s in items) + "\n"
    p.write_text(text, encoding="utf-8")

class BaseDataset(Dataset):

    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
        min_train_freq=128, max_train_freq=512, dt=0.005):
        super().__init__()
        # where record pre loaded data
        self.predata_dir = predata_dir
        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')

        self.mode = mode
        # choose between training, validation or test sequences
        train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs,
            test_seqs) # 3, T, C에서 3은 배치가 아니라 시퀀스 -> 시퀀스 하나에 16000프레임
        # get and compute value for normalizing inputs
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)
        self.mode = mode  # train, val or test
        self._train = False
        self._val = False
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # self.imu_std = torch.Tensor([0, 0]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # self.imu_b0 = torch.Tensor([0, 0]).float()
        # IMU sampling time
        self.dt = dt # (s)
        # sequence size during training
        self.N = N # power of 2
        self.min_train_freq = min_train_freq
        self.max_train_freq = max_train_freq
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1),
            torch.ones(1))

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode"""
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict['train'], sequences_dict[self.mode]

    def __getitem__(self, i):
        #print(self.sequences)
        mondict = self.load_seq(i)
        mondict_gt = self.load_gt(i)
        N_max = mondict['xs'].shape[0]
        # print(mondict['xs'].shape,mondict['us'].shape) # torch.Size([28100, 4]) torch.Size([28116, 6])
        if self._train: # random start
            n0 = torch.randint(0, self.max_train_freq, (1, ))
            nend = n0 + self.N
        elif self._val: # end sequence
            n0 = self.max_train_freq + self.N
            nend = N_max - ((N_max - n0) % self.max_train_freq)
        else:  # full sequence
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)
        u = mondict['us'][n0: nend]
        x = mondict['xs'][n0: nend]
        t = mondict_gt['ts'][n0: nend]
        # print(u[0], u[-1])
        return u, x, t

    def __len__(self):
        return len(self.sequences)

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in run bias stability)
        b0 = self.uni.sample(u[:, 0].shape).cuda()
        b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]
        b0[:, :, 3:6] =  b0[:, :, 3:6] * self.imu_b0[1]
        u = u + noise + b0.transpose(1, 2)
        return u

    def init_train(self):
        self._train = True
        self._val = False

    def init_val(self):
        self._train = False
        self._val = True

    def length(self):
        return self._length

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    def init_normalize_factors(self, train_seqs):
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            # print(mondict['mean_u'].shape)
            return mondict['mean_u'], mondict['std_u']

        path = os.path.join(self.predata_dir, train_seqs[0] + '.p')
        if not os.path.exists(path):
            print("init_normalize_factors not computed")
            return 0, 0

        print('Start computing normalizing factors ...')
        cprint("Do it only on training sequences, it is vital!", 'yellow')
        # first compute mean
        num_data = 0

        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            sms = pickle_dict['xs']
            if i == 0:
                mean_u = us.sum(dim=0)
                num_positive = sms.sum(dim=0)
                num_negative = sms.shape[0] - sms.sum(dim=0)
            else:
                mean_u += us.sum(dim=0)
                num_positive += sms.sum(dim=0)
                num_negative += sms.shape[0] - sms.sum(dim=0)
            num_data += us.shape[0]
        mean_u = mean_u / num_data
        pos_weight = num_negative / num_positive

        # second compute standard deviation
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_u) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()
        normalize_factors = {
            'mean_u': mean_u,
            'std_u': std_u,
        }
        print('... ended computing normalizing factors')
        print('pos_weight:', pos_weight)
        print('This values most be a training parameters !')
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        print('num_data  :', num_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_u, std_u

    def read_data(self, data_dir):
        raise NotImplementedError

    @staticmethod
    def interpolate(x, t, t_int):
            """
            Interpolate ground truth at the sensor timestamps
            """

            # vector interpolation
            x_int = np.zeros((t_int.shape[0], x.shape[1]))
            for i in range(x.shape[1]):
                if i in [4, 5, 6, 7]:
                    continue
                x_int[:, i] = np.interp(t_int, t, x[:, i])
            # quaternion interpolation
            t_int = torch.Tensor(t_int - t[0])
            t = torch.Tensor(t - t[0])
            qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
            x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
            return x_int


class EUROCDataset(BaseDataset):
    """
        Dataloader for the EUROC Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq, dt)
        # convert raw data to pre loaded data
        self.read_data(data_dir)

    def read_data(self, data_dir):
        r"""Read the data from the dataset"""

        f = os.path.join(self.predata_dir, 'MH_01_easy.p')
        if True and os.path.exists(f): # == if os.path.exists(f):
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv")
            path_gt = os.path.join(data_dir, seq, "mav0", "state_groundtruth_estimate0", "data.csv")
            return path_imu, path_gt

        sequences = os.listdir(data_dir) # MH_01_easy, ...
        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence) # directory 안의 mav0/imu0/data.csv
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0]/1e9

            # interpolate
            gt = self.interpolate(gt, gt[:, 0]/1e9, ts)

            # take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = torch.Tensor(gt[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

            # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.tensor(gt[:, 8:11]).double()
            imu = torch.Tensor(imu[:, 1:]).double()

            # compute pre-integration factors for all training
            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij.cuda())
            dxi_ij = SO3.log(dRot_ij).cpu()

            # save for all training
            mondict = {
                'xs': dxi_ij.float(),
                'us': imu.float(),
            }
            pdump(mondict, self.predata_dir, sequence + ".p")
            # save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")

class RONINDataset(BaseDataset):
    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq, dt)
        # convert raw data to pre loaded data
        self.read_data(data_dir)


    def read_data(self, data_dir):
        r"""Read the data from the dataset"""
        #    'predata_dir': '/home/mpil/PycharmProjects/RoNIN/Data/seen_subjects_test_set', pickle 저장 경로
        # data_dir = '/home/mpil/PycharmProjects/RoNIN/Data/train_dataset_merge/'  hdf5 저장 경로
        p = '/home/mpil/PycharmProjects/RoNIN/Data/cache_localframe/'
        f = os.path.join(self.predata_dir, 'a059_3.p')
        if True and os.path.exists(f): # == if os.path.exists(f):
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            #path = osp.join(data_dir, seq+'.hdf5') global frame data
            path = osp.join(data_dir, seq) # local frame data
            return path

        sequences = os.listdir(data_dir) # MH_01_easy, ...
        # read each sequence
        for sequence in sequences:
            if sequence == 'config.json':
                continue
            print("\nSequence name: " + sequence)
            path = set_path(sequence)  # directory 안의 mav0/imu0/data.csv
            if osp.exists(path) == False:
                print('In the data_dir, Not list')
                continue
            with h5py.File(path) as f:
                print(path)
                feat = np.copy(f['feature'])
                targ = np.copy(f['target'])
                aux = np.copy(f['aux'])
            ts = aux[:,0]
            imu = feat
            q_gt = aux[:, 1:5] # global frame에서 본 ori gt
            p_gt = aux[:,5:8]
            v_gt = targ

            ts = ts/1e9

            q_gt = torch.Tensor(q_gt).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

            # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.tensor(v_gt).double()
            imu = torch.Tensor(imu).double()

            # compute pre-integration factors for all training
            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij.cuda())
            dxi_ij = SO3.log(dRot_ij).cpu()

            sequence = os.path.splitext(sequence)[0]

            # save for all training
            mondict = {
                'xs': dxi_ij.float(),
                'us': imu.float(),
            }
            pdump(mondict, self.predata_dir, sequence + ".p")
            # save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")

class ASTROBEEDataset(BaseDataset):
    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                 test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq, dt)
        self.Rimu2body = torch.tensor([[0, -1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1]], dtype=torch.float64)
        # convert raw data to pre loaded data
        self.read_data(data_dir)

    def read_data(self, data_dir):
        """Read the data from the dataset"""

        f = os.path.join(self.predata_dir, 'iva_badlocal_descend.p')
        # f2 = os.path.join(data_dir, 'train_revised.txt')
        if True and os.path.exists(f):
            return

        print("Start read_data, be patient please")

        def set_path(seq):
            #path_imu = os.path.join(data_dir, seq, "imu_filtered.txt")
            #path_gt = os.path.join(data_dir, seq, "vio.txt")
            path_imu = os.path.join(data_dir, seq, "imu.txt")
            path_gt = os.path.join(data_dir, seq, "groundtruth.txt")
            return path_imu, path_gt

        sequences = os.listdir(data_dir)
        not_imu_filtered = []
        not_gt = []
        train_revised = []
        # read each sequence
        for sequence in sequences: # a000 ...
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            if not Path(path_gt).is_file():
                not_gt.append(sequence)
                if not Path(path_imu).is_file():
                    not_imu_filtered.append(sequence)

            else:
                if not Path(path_imu).is_file():
                    not_imu_filtered.append(sequence)
                else:
                    imu = np.genfromtxt(path_imu, delimiter=" ", skip_header=0)
                    # timestamp wx wy wz ax ay az vx vy vz b_wx b_wy b_wz b_ax b_ay b_az
                    gt = np.genfromtxt(path_gt, delimiter=" ", skip_header=0)
                    # timestamp tx ty tz qx qy qz qw

                    # time synchronization between IMU and ground truth
                    t0 = np.max([gt[0, 0], imu[0, 0]])
                    t_end = np.min([gt[-1, 0], imu[-1, 0]])

                    # start index
                    idx0_imu = np.searchsorted(imu[:, 0], t0)
                    idx0_gt = np.searchsorted(gt[:, 0], t0)

                    # end index
                    idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
                    idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

                    # subsample
                    imu = imu[idx0_imu: idx_end_imu]
                    gt = gt[idx0_gt: idx_end_gt]
                    ts = imu[:, 0] / 1e9 # imu에서 잘랐으므로 ts까지 interp된 게 맞음

                    # interpolate
                    gt = self.interpolate(gt, gt[:, 0] / 1e9, ts)

                    # take ground truth position
                    p_gt = gt[:, 1:4]
                    p_gt = p_gt - p_gt[0]

                    # take ground true quaternion pose
                    q_gt = torch.Tensor(gt[:, 4:8]).double() # xyzw
                    q_gt = q_gt[:, [3,0,1,2]] / q_gt.norm(dim=1, keepdim=True)
                    Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

                    # convert from numpy
                    p_gt = torch.Tensor(p_gt).double()
                    # v_gt = torch.tensor(gt[:, 8:11]).double()
                    imu = torch.Tensor(imu[:, 1:7]).double()
                    '''gyro = imu[:, :3] @ self.Rimu2body.T
                    acc = imu[:, 3:] @ self.Rimu2body.T
                    imu = torch.cat([gyro, acc], dim=1) # CF alignment'''

                    # compute pre-integration factors for all training
                    mtf = self.min_train_freq
                    dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
                    dRot_ij = SO3.dnormalize(dRot_ij.cuda())
                    dxi_ij = SO3.log(dRot_ij).cpu()

                    # save for all training
                    mondict = {
                        'xs': dxi_ij.float(),
                        'us': imu.float(),
                    }
                    pdump(mondict, self.predata_dir, sequence + ".p")
                    # save ground truth
                    mondict = {
                        'ts': ts,
                        'qs': q_gt.float(), # wxyz
                        # 'vs': v_gt.float(),
                        'ps': p_gt.float(),
                    }
                    pdump(mondict, self.predata_dir, sequence + "_gt.p")
                    train_revised.append(sequence)
        not_data_dict = {
            'not_imu_filtered' : not_imu_filtered,
            'not_gt' : not_gt
        }
        save_str_list("/home/mpil/astrobee_dataset_revised/train_revised.txt", train_revised)
        with open("/home/mpil/miniconda3/envs/gyronet/missing_data.json", "w") as f:
            json.dump(not_data_dict, f, indent=4)


class TUMVIDataset(BaseDataset):
    """
        Dataloader for the TUM-VI Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
            min_train_freq, max_train_freq, dt)
        # convert raw data to pre loaded data
        self.read_data(data_dir)
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()

    def read_data(self, data_dir):
        r"""Read the data from the dataset"""

        f = os.path.join(self.predata_dir, 'dataset-room1_512_16_gt.p')
        if True and os.path.exists(f):
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            seq_in = os.listdir(data_dir + '/' + seq) # seq = sequences = room1, room2,
            for s in seq_in: # s = dataset_room1, dataset_room1_gt
                if 'gt' in s:
                    path_gt = os.path.join(data_dir, seq, s)
                    print("gt loaded ", s)
                else:
                    path_imu = os.path.join(data_dir, seq, s)
                    print("imu loaded ", s)

                if 'gt' in s:
                    s = s.removesuffix('_gt.csv')
                else:
                    s = s.removesuffix('.csv')
            return path_imu, path_gt, s

        sequences = os.listdir(data_dir)

        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            if 'room' not in sequence:
                continue

            path_imu, path_gt, s = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0]/1e9

            # interpolate
            t_gt = gt[:, 0]/1e9
            gt = self.interpolate(gt, t_gt, ts)

            # take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = SO3.qnorm(torch.Tensor(gt[:, 4:8]).double())
            Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

            # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.zeros_like(p_gt).double()
            v_gt[1:] = (p_gt[1:]-p_gt[:-1])/self.dt
            imu = torch.Tensor(imu[:, 1:]).double()

            # compute pre-integration factors for all training
            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:]) #batch matrix transpose matrix product
            dRot_ij = SO3.dnormalize(dRot_ij.cuda())
            dxi_ij = SO3.log(dRot_ij).cpu()

            # masks with 1 when ground truth is available, 0 otherwise
            masks = dxi_ij.new_ones(dxi_ij.shape[0])
            tmp = np.searchsorted(t_gt, ts[:-mtf])
            diff_t = ts[:-mtf] - t_gt[tmp]
            masks[np.abs(diff_t) > 0.01] = 0

            # save all the sequence
            mondict = {
                'xs': torch.cat((dxi_ij, masks.unsqueeze(1)), 1).float(),
                'us': imu.float(),
            }
            pdump(mondict, self.predata_dir, s + ".p") # sequence -> s.removesuffix('.csv')

            # save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(),
                'vs': v_gt.float(),
                'ps': p_gt.float(),
            }
            pdump(mondict, self.predata_dir, s + "_gt.p") # sequence -> s
