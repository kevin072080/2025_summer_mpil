import json
import random
from os import path as osp

import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences


class GlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 3 # 2-> 3, StridedSequenceDataset에서 참조함, ronin_resnet에서 이미 갱신되긴 함
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        self.info['path'] = osp.split(data_path)[-1]

        self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
            data_path, self.max_ori_error, self.grv_only)

        start_frame = self.info.get('start_frame', 0)
        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            grav = np.copy(f['synced/grav'])
            gyro_uncalib = np.copy(f['synced/gyro_uncalib'])
            acce_uncalib = np.copy(f['synced/acce'])
            linacce_uncalib = np.copy(f['synced/linacce'])

            gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
            #acceg = np.array(self.info['imu_acce_scale']) * (acce_uncalib - grav - np.array(self.info['imu_acce_bias']))
            acceg = np.array(self.info['imu_acce_scale']) * (acce_uncalib - grav) # grav 및 acce 측정에 모두 같은 bias -> 이미 서로 상쇄 -> 실제로 metric 살짝 작음
            linacce = np.array(self.info['imu_acce_scale']) * (linacce_uncalib - np.array(self.info['imu_acce_bias']))
            ts = np.copy(f['synced/time'])
            tango_pos = np.copy(f['pose/tango_pos'])
            tango_ori = np.copy(f['pose/tango_ori'])
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
            self.startframe_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][start_frame])

        # Compute the Tango global v in the IMU body frame
        ori_q = quaternion.from_float_array(ori)  # recon_traj -> start_frame부터 step 10마다 읽어감
        tango_ori_q = quaternion.from_float_array(tango_ori)
        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])  # global frame compensation (spatial alignment), x축 중심으로 약 180도 회전
        rot_tango_to_imu = rot_imu_to_tango.conj()
        ori_body_q = ori_q.conj() * ori_q[0] * rot_tango_to_imu * tango_ori_q[0].conj()

        self.rot_imu_to_tango_startframe = tango_ori_q[start_frame].conj() * tango_ori_q[0] * rot_imu_to_tango * ori_q[0].conj() * ori_q[start_frame] # start frame index rotation q

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt  # n - w개의 v가 계산됨 , w = 200
        glob_v_q = quaternion.from_float_array(np.concatenate([np.zeros([glob_v.shape[0], 1]), glob_v], axis=1))
        body_v = quaternion.as_float_array(ori_body_q[self.w:] * glob_v_q * ori_body_q[self.w:].conj())
        body_v = body_v[:, 1:]

        '''ori_test = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        grav_q = quaternion.from_float_array(np.concatenate([np.zeros([grav.shape[0], 1]), grav], axis=1))
        print(quaternion.as_float_array((ori_test * ori_q) * grav_q * (ori_test * ori_q).conj())) 
        # 약 9.8k'''

        self.ts = ts[start_frame:]
        #self.features = np.concatenate([gyro, acceg], axis=1)[start_frame:]
        self.features = np.concatenate([gyro, linacce], axis=1)[start_frame:]
        # global - g 에서도 linacce 활용했으므로 여기서도 활용
        self.targets = body_v[start_frame:, :3] # 2 -> 3
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
        self.gt_pos = tango_pos[start_frame:]

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class DenseSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
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
            seq_type, root_dir, data_list, cache_path, interval=1, **kwargs)

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
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
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

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        self.seq_objects = [seq_type(self.data_path[i], **kwargs) for i in range(len(data_list))]

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
