import json
import random
import matplotlib.pyplot as plt
import torch
from os import path as osp

import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences
from lie_algebra import SO3

from math_util import gyro_integration
#def gyro_integration(ts, gyro, init_q):

class GlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 3 # 2-> 3, StridedSequenceDataset에서 참조함, ronin_resnet에서 이미 갱신되긴 함
    aux_dim = 8

    def __init__(self, data_path=None, ori_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        if data_path is not None:
            self.load(data_path, ori_path)


    def load(self, data_path, ori_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        self.info['path'] = osp.split(data_path)[-1]
        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            gyro_uncalib = f['synced/gyro_uncalib']
            acce_uncalib = f['synced/acce']
            gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
            acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))
            ts = np.copy(f['synced/time'])
            tango_pos = np.copy(f['pose/tango_pos'])
            #print(*f['pose/tango_ori'].shape)
            tango_ori = quaternion.from_float_array(f['pose/tango_ori'])

        if ori_path is None:
            self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
                data_path, self.max_ori_error, self.grv_only)
        else:
            print(ori_path)
            ori = np.genfromtxt(ori_path, delimiter=" ", skip_header=0)[:,[4,1,2,3]]
            ts_ori = np.genfromtxt(ori_path, delimiter=" ", skip_header=0)[:, 0]
            ts_ori = ts_ori * 1e9
            self.info['ori_source'] = ori_path
            self.info['source_ori_error'] = 0
            #print(ts[0], ts_ori[0])
            t0 = np.max([ts[0], ts_ori[0]])
            t_end = np.min([ts[-1], ts_ori[-1]])

            # start index
            idx0_imu = np.searchsorted(ts, t0)
            idx0_ori = np.searchsorted(ts_ori, t0)

            # end index
            idx_end_imu = np.searchsorted(ts, t_end, 'right')
            idx_end_ori = np.searchsorted(ts_ori, t_end, 'right')
            #print(idx_end_ori - idx0_ori, idx_end_imu - idx0_imu)
            #print(ts[idx0_imu], ts_ori[idx0_ori], t0)
            #print(ts[idx_end_imu], ts_ori[idx_end_ori-1], t_end)


            # subsample
            interv_imu = idx_end_imu - idx0_imu
            interv_ori = idx_end_ori - idx0_ori
            if interv_ori > interv_imu:
                idx_end_ori = idx0_ori + interv_imu
            elif interv_ori < interv_imu:
                idx_end_imu = idx0_imu + interv_ori
            else:
                idx_end_ori -= 1
                idx_end_imu -= 1
            ts = ts[idx0_imu: idx_end_imu]
            gyro = gyro[idx0_imu: idx_end_imu]
            acce = acce[idx0_imu: idx_end_imu]
            tango_pos = tango_pos[idx0_imu: idx_end_imu]
            init_tango_ori = tango_ori[idx0_imu]
            #ori = ori[idx0_ori: idx0_ori + idx_end_imu - idx0_imu]
            ori = ori[idx0_ori: idx_end_ori]
            print(ori.shape, gyro.shape)
            tango_ori = tango_ori[idx0_ori: idx_end_ori]


        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration']) # global frame compensation (spatial alignment) x축 중심 180도 회전 -> global끼리 calib
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj() # 여기에 init_ori 곱하면 imu to tango init이 도출됨
        ori_q = init_rotor * ori_q # 윗줄에서 tango frame으로 정렬하는 rotation quaternion 구하여 other ts ori에 모두 곱함

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1)) # 쿼터니언 계산용 scalar concat
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        start_frame = self.info.get('start_frame', 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:] # body frame imu data
        #self.features = np.concatenate([gyro, acce], axis=1)[start_frame:] # imu frame imu data
        self.targets = glob_v[start_frame:, :3] # 2 -> 3
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:] # wxyz
        self.gt_pos = tango_pos[start_frame:]

        # print('Debug', ori_q.shape, self.orientations.shape) Debug (106000,) (91103, 4)
    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        print(self.gt_pos.shape)
        print(self.gt_pos[0])
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])

class AstrobeeGlobSpeedSequence(CompiledSequence):
    feature_dim = 6
    target_dim = 3  # 2-> 3, StridedSequenceDataset에서 참조함, ronin_resnet에서 이미 갱신되긴 함
    aux_dim = 8

    def __init__(self, data_path=None, ori_data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        self.Rimu2body = np.array([[0, -1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1]], dtype=np.float64)
        if data_path is not None:
            self.load(data_path, ori_data_path)

    @staticmethod
    def interpolate(x, t, t_int):
        """
        Interpolate ground truth at the sensor timestamps
        """

        # vector interpolation
        x_int = np.zeros((t_int.shape[0], x.shape[1]))
        if x.shape[1] == 8:
            print('interpolating pos and qs')
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
        elif x.shape[1] == 5:
            print('interpolating only qs')
            x_int[:, 0] = np.interp(t_int, t, x[:, 0])

            t_int = torch.Tensor(t_int - t[0])
            t = torch.Tensor(t - t[0])
            qs = SO3.qnorm(torch.Tensor(x[:, 1:5]))
            x_int[:, 1:5] = SO3.qinterp(qs, t, t_int).numpy()
            return x_int


    def quat2rotvec(self, q):
        w, x, y, z = q
        # 회전 각도 계산 (radian)
        theta = 2 * np.arccos(w)

        # 회전 축 계산 (단위 벡터)
        norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if norm < 1e-8:  # 회전 축이 없으면 (w = 1인 경우)
            return np.array([0.0, 0.0, 0.0])

        u = np.array([x, y, z]) / norm  # 단위 회전축

        # 회전 벡터 계산
        rotvec = theta * u
        return rotvec

    def process_data(self, q_gt):
        # q_gt가 N x 4 크기일 때
        q_gt = np.array(q_gt)  # N x 4 배열로 변환

        # 벡터화된 연산: 각 쿼터니언을 한 번에 처리
        w, x, y, z = q_gt.T  # q_gt를 각 행렬의 벡터로 분리
        theta = 2 * np.arccos(w)
        norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        u = np.stack([x, y, z], axis=-1) / norm[:, None]  # 회전 축
        u[norm < 1e-8] = 0  # 회전축이 없을 때는 0으로 처리

        # 회전 벡터 계산
        rotvecs = (theta[:, None] * u)  # N x 3 회전 벡터 배열

        # 속도 향상을 위한 벡터화 처리
        return rotvecs

    def load(self, data_path, ori_data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        self.data_path = data_path
        self.ori_path = ori_data_path

        #path_imu = osp.join(data_path, 'imu_filtered.txt') # astrobee_revised
        #path_gt = osp.join(data_path, 'vio.txt')
        path_imu = osp.join(data_path, 'imu.txt') # AstrobeePublic
        path_gt = osp.join(data_path, 'groundtruth.txt')
        path_ori = ori_data_path # load_cache에서 이미 ori_q까지 지정

        imu = np.genfromtxt(path_imu, delimiter=" ", skip_header=0)
        # timestamp wx wy wz ax ay az vx vy vz b_wx b_wy b_wz b_ax b_ay b_az
        gt = np.genfromtxt(path_gt, delimiter=" ", skip_header=0)
        # timestamp tx ty tz qx qy qz qw
        ori_explicit = np.genfromtxt(path_ori, delimiter=" ", skip_header=0) #xyzw
        ori_explicit[:,0] = ori_explicit[:,0] * 1e9


        # time synchronization between IMU and ground truth
        t0 = np.max([gt[0, 0], imu[0, 0], ori_explicit[0, 0]])
        t_end = np.min([gt[-1, 0], imu[-1, 0],ori_explicit[-1, 0]])

        # start index
        idx0_imu = np.searchsorted(imu[:, 0], t0)
        idx0_gt = np.searchsorted(gt[:, 0], t0)
        idx0_ori = np.searchsorted(ori_explicit[:, 0], t0)

        # end index
        idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
        idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')
        idx_end_ori = np.searchsorted(ori_explicit[:, 0], t_end, 'right')

        # subsample
        imu = imu[idx0_imu: idx_end_imu]
        gt = gt[idx0_gt: idx_end_gt]
        ori_explicit = ori_explicit[idx0_ori: idx_end_ori] # xyzw
        #print('idx 직후 ',ori_explicit[0:3])
        #print('idx 직후 ', gt[0:3])
        ts = imu[:, 0] / 1e9
        print('ts ', ts[0], ts[-1])
        print('gts ', gt[0,0]/1e9, gt[-1, 0]/1e9)
        print('ori ', ori_explicit[0,0]/1e9, ori_explicit[-1,0]/1e9)
        # interpolate
        print(ts.shape, gt.shape, ori_explicit.shape)
        gt = self.interpolate(gt, gt[:, 0] / 1e9, ts) # numpy 반환
        ori_explicit = self.interpolate(ori_explicit, ori_explicit[:, 0] / 1e9, ts)
        print(ts.shape, gt.shape, ori_explicit.shape)

        ori_explicit = ori_explicit[:, [4,1,2,3]] # wxyz
        ori_q = ori_explicit / np.linalg.norm(ori_explicit, axis=1, keepdims=True)
        ori_q = quaternion.from_float_array(ori_q)

        # take ground true quaternion pose
        q_gt = torch.Tensor(gt[:, 4:8]).double()  # xyzw
        q_gt = q_gt[:, [3, 0, 1, 2]] / q_gt.norm(dim=1, keepdim=True) # wxyz
        # Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

        ori_q = quaternion.from_float_array(q_gt.detach().cpu().numpy())[0] * ori_q[0].conj() * ori_q

        imu = torch.Tensor(imu[:, 1:7]).double()
        gyro = imu[:, :3]
        acc = imu[:, 3:]
        #gyro = gyro @ self.Rimu2body.T
        #acc = acc @ self.Rimu2body.T
        imu = torch.cat([gyro, acc], dim=1)  # CF alignment

        p_gt = gt[:, 1:4]
        p_gt = p_gt - p_gt[0]
        dt = ts[1:] - ts[:-1]
        #print(p_gt)
        #print('dt :', np.mean(dt))
        v_gt = (p_gt[1:,:] - p_gt[:-1,:]) / dt[:,None]

        ori_q = quaternion.from_float_array(gt[:, [7,4,5,6]])


        #init_q = q_gt[0, [1,2,3,0]].cpu().numpy() # wxyz -> xyzw
        #####################################################################################
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acc.shape[0], 1]), acc], axis=1))

        # world -> body가 아닌가?
        #glob_gyro = (ori_q.conj() * gyro_q * ori_q)[:, 1:] # body2world q
        #glob_acce = (ori_q.conj() * acce_q * ori_q)[:, 1:]

        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:] # world2body q
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
        #glob_gyro = glob_gyro @ self.Rimu2body.T
        #glob_acce = glob_acce @ self.Rimu2body.T
        #print(glob_acce * 0.01)
        #print(v_gt[1:]/0.01 - v_gt[:-1]/0.01)
        print((glob_acce[1:] - glob_acce[:-1]))
        print((v_gt[1:] - v_gt[:-1])/dt[:-1, None])
        diff_acce = (glob_acce[1:] - glob_acce[:-1]) * dt[:, None]  # (N-1, 3)
        diff_v = (v_gt[1:] - v_gt[:-1])   # (N-1, 3)
        print(np.cumsum(diff_v))
        print(np.cumsum(diff_acce))

        #print(v_gt)

        #####################################################################################
        self.ts = ts
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
        self.targets = v_gt
        self.orientations = quaternion.as_float_array(ori_q)  # wxyz
        #self.gt_pos = np.array([self.quat2rotvec(q) for q in q_gt])
        #self.gt_pos = self.process_data(q_gt)
        self.gt_pos = p_gt
        #print(ori_explicit.shape)
        #print(self.ts.shape, self.orientations.shape, self.gt_pos.shape)

        self.aux = np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)
        #print('ASTRO',self.features.shape, self.targets.shape, self.aux.shape)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        #print(self.ts.shape, self.orientations.shape, self.gt_pos.shape)
        # print(self.aux.shape) dim == 8
        return self.aux

    def get_meta(self):
        return 'Astrobee dataset, data_Path: {}, ori_path: {}'.format(self.data_path,self.ori_path)

class DenseSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, ori_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super().__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, ori_dir, data_list, cache_path, interval=1, **kwargs)


        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(window_size, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class StridedSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, ori_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)
        #print(data_list)
        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, ori_dir, data_list, cache_path, interval=self.interval, **kwargs)
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)
        # print(feat.shape, targ.shape, seq_id ,frame_id) # 200,6 3, _ _
        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=400,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, 5:8])

            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            bad_data = velocity > max_norm
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)
