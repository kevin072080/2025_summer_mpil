import argparse
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory


class Sequence(ABC):
    # Dictionary to keep track of subclasses
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls


class SeqeuncesDataset(Data.Dataset):
    """
    For the purpose of training and inferering
    1. Abandon the features of the last time frame, since there are no ground truth pose and dt
     to integrate the imu data of the last frame. So the length of the dataset is seq.get_length() - 1
    """
    def __init__(self, data_set_config, data_path = None, data_root = None, device= "cuda:0"):
        super(SeqeuncesDataset, self).__init__()
        (
            self.ts,
            self.dt,
            self.acc,
            self.gyro,
            self.gt_pos,
            self.gt_ori,
            self.gt_velo,
            self.index_map,
            self.seq_idx,
        ) = ([], [], [], [], [], [], [], [], 0)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        self.device = device
        self.conf = data_set_config
        # self.gravity = conf.gravity if "gravity" in conf.keys() else 9.81007
        self.gravity = 9.81007
        self.seq_id_to_name = {}


        self.DataClass = Sequence.subclasses
        print("Registered subclasses:", Sequence.subclasses)

        ## the design of datapath provide a quick way to revisit a specific sequence, but introduce some inconsistency
        # root : 한 번에, path : 개별 -> config에는 root만 존재
        if data_path is None:
            for conf in data_set_config.data_list: # data_set_config :
                print(conf, data_set_config.data_list)
                for path in conf.data_drive:
                    self.construct_index_map(conf, conf["data_root"], path, self.seq_idx)
                    self.seq_idx += 1
        ## the design of dataroot provide a quick way to introduce multiple sequences in eval set, but introduce some inconsistency
        elif data_root is None:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, conf["data_root"], data_path, self.seq_idx)
            self.seq_idx += 1
        else:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, data_root, data_path, self.seq_idx)
            self.seq_idx += 1


    def load_data(self, seq, start_frame, end_frame):
        #if "time" in seq.data.keys():
        #    self.ts.append(seq.data["time"][start_frame:end_frame])
        self.ts.append(seq.data["time"][start_frame:end_frame])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        # the groud truth state should include the init state and integrated state, thus has one more frame than imu data
        self.dt.append(seq.data["dt"][start_frame:end_frame +1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame:end_frame +1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame:end_frame +1])
        #self.gt_velo.append(seq.data["velocity"][start_frame:end_frame +1])

    # window로 안 쪼개도록 수정
    def construct_index_map(self, conf, data_root, data_name, seq_id):
        seq = self.DataClass[conf.name](data_root, data_name, intepolate=True, **self.conf)
        seq_len = seq.get_length() - 1  # 마지막 프레임 제외
        start_frame, end_frame = 0, seq_len

        # 하나의 시퀀스를 통으로 index_map에 추가
        if torch.all(seq.data["mask"][start_frame:end_frame]):
            self.index_map.append([seq_id, start_frame, end_frame])
        self.seq_id_to_name[seq_id] = data_name

        # 데이터는 그대로 로드
        self.load_data(seq, start_frame, end_frame)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id, end_frame_id = self.index_map[item][0], self.index_map[item][1], self.index_map[item][2]
        time = {
            'timestamp' : self.ts[seq_id][frame_id: end_frame_id]
        }

        data = {
            'dt': self.dt[seq_id][frame_id: end_frame_id],
            'acc': self.acc[seq_id][frame_id: end_frame_id],
            'gyro': self.gyro[seq_id][frame_id: end_frame_id],
            'rot': self.gt_ori[seq_id][frame_id: end_frame_id]# xyzw
        }
        init_state = {
            'init_rot': self.gt_ori[seq_id][frame_id][None, ...],
            'init_pos': self.gt_pos[seq_id][frame_id][None, ...],
            #'init_vel': self.gt_velo[seq_id][frame_id][None, ...],
        }
        label = {
            'gt_pos': self.gt_pos[seq_id][frame_id + 1: end_frame_id + 1],
            'gt_rot': self.gt_ori[seq_id][frame_id + 1: end_frame_id + 1],
            #'gt_vel': self.gt_velo[seq_id][frame_id + 1: end_frame_id + 1],
        }

        return {**time, **data, **init_state, **label}

    def get_dtype(self):
        return self.acc[0].dtype