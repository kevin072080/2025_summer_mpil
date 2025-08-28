import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from scipy.spatial.transform import Rotation as R, Slerp
import os
from ahrs.filters import Madgwick, Mahony
import json
from pyhocon import ConfigFactory
from datasets import SeqeuncesDataset
from EuRoCdataset import Euroc
from Astrobeedataset import Astrobee
def align_quaternions(q1, q2):
    """
    q1, q2: (N, 4) numpy arrays in wxyz format
    Return q2 with sign aligned to q1
    """
    q2_aligned = q2.copy()
    dot = np.sum(q1 * q2, axis=1)
    q2_aligned[dot < 0] *= -1
    return q2_aligned


def move_to(obj, device):
    if torch.is_tensor(obj):return obj.to(device)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj).to(device)
    else:
        raise TypeError("Invalid type for move_to", type(obj))

def AOE_tensor(est_R: R, rot_gt: R):
    #rot_gt = R.from_quat(quat_gt_tensor[:,[1,2,3,0]].numpy()) # R : xyzw 기준, gt : xyzw로 전달됨
    rel_rot = est_R.inv() * rot_gt
    rotvec = torch.tensor(rel_rot.as_rotvec())  # (N, 3)
    angles = torch.norm(rotvec, dim=1)  # (N,)

    # 회전 각이 π 초과면 반대 방향으로 변경 (최소 회전)
    angles = torch.where(angles > np.pi, 2 * np.pi - angles, angles) * 180 / np.pi
    count10 = (angles > 10).sum()
    count30 = (angles > 30).sum()
    count90 = (angles > 90).sum()
    '''print(count10, count30, count90)
    print('deg: ',angles)'''
    '''print(rotvec)
    print((torch.norm(rotvec,dim=1)**2))
    print(torch.mean((torch.norm(rotvec,dim=1)**2)))
    print(type(est_R), type(rot_gt), type(rel_rot))
    assert len(est_R) == len(rot_gt)'''

    return torch.sqrt(torch.mean(angles[:]**2))
    #return torch.sqrt(torch.mean(angles[1:] ** 2)) -> 1?
    #return torch.sqrt(torch.mean((torch.norm(rotvec,dim=1)**2))) * 180 / np.pi
    #return torch.norm(rotvec, dim=1).mean().item() * 180 / np.pi

def ROE_tensor(est_R: R, rot_gt: R):
    #rot_gt = R.from_quat(quat_gt_tensor[:,[1,2,3,0]].numpy())

    roe = []
    last_idx = 0

    accum_t = 0
    for i in range(len(est_R)):
        if accum_t > 5:  # ms 단위 기준 (50s마다 계산)
            drot_est = est_R[last_idx].inv() * est_R[i]
            drot_gt = rot_gt[last_idx].inv() * rot_gt[i]
            rel_rot = drot_est.inv() * drot_gt
            rotvec = rel_rot.as_rotvec()
            roe.append(torch.norm(torch.tensor(rotvec)).item() * 180 / np.pi)
            last_idx = i  # 기준점 업데이트
            accum_t = 0
        accum_t += 1/200 # 5ms per frame -> 5초마다 계산

    return torch.tensor(roe) # list of rotation angle errors (float)

def RPY(est_R: R, rot_gt : R):
    euler_est = est_R.as_euler('xyz', degrees= True)
    euler_gt = rot_gt.as_euler('xyz', degrees=True)

    roll_rmse = np.sqrt(np.mean((euler_est[:, 0] - euler_gt[:, 0]) ** 2))
    pitch_rmse = np.sqrt(np.mean((euler_est[:, 1] - euler_gt[:, 1]) ** 2))
    yaw_rmse = np.sqrt(np.mean((euler_est[:, 2] - euler_gt[:, 2]) ** 2))
    return roll_rmse, pitch_rmse, yaw_rmse

def metric(filename, est_R: R, rot_gt : R, show = False):
    roll_rmse, pitch_rmse, yaw_rmse = RPY(est_R, rot_gt)
    euler_est = est_R.as_euler('xyz', degrees=True)
    euler_gt = rot_gt.as_euler('xyz', degrees=True)
    if show:
        plt.figure(figsize=(10, 6))
        plt.title(filename)
        plt.subplot(3, 1, 1)
        plt.plot(euler_est[:, 0], label='Roll')
        plt.plot(euler_gt[:, 0], label='Roll_gt')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(euler_est[:, 1], label='P')
        plt.plot(euler_gt[:, 1], label='P_gt')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(euler_est[:, 2], label='Y')
        plt.plot(euler_gt[:, 2], label='Y_gt')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.title(filename)
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.title(f'{filename}, q{i}')
            plt.plot(est_R.as_quat()[:, [3, 0, 1, 2]][:, i], label='Estimated')
            plt.plot(rot_gt.as_quat()[:, [3, 0, 1, 2]][:, i], label='GT')
            plt.legend()
        plt.tight_layout()
        plt.show()

    aoe = AOE_tensor(est_R, rot_gt)
    roe = ROE_tensor(est_R, rot_gt)

    print('RPY (deg) : ', roll_rmse, pitch_rmse, yaw_rmse)

    print(f"File : {filename}, AOE (deg): {aoe:.6f}, ROE (deg): {roe.mean():.6f}")
    print("All ROE", roe, "\n")
    resultdict = {
        'name' : filename,
        'AOE (deg)': aoe.item(),
        'ROE_Mean (deg)': float(roe.mean()),
        'ROE': roe.tolist(),
        #'rot_compensated': est_R.as_euler('xyz', degrees=True).tolist(),
        #'rot_gt': rot_gt.as_euler('xyz', degrees=True).tolist()
    }
    return resultdict

def Madgwick_EuRoc(loader, confs, resultdict_madg, orientations_madg):
    for i, sample in enumerate(loader):
        seq_id = loader.dataset.index_map[i][0]
        filename = loader.dataset.seq_id_to_name[seq_id]

        ts = sample["timestamp"].squeeze(0)
        gyro = sample["gyro"].squeeze(0) # Euroc IMU
        acc = sample["acc"].squeeze(0) # Euroc IMU
        dt = sample["dt"].squeeze(0) # 다 0.0050 확인
        quats_gt = sample["gt_rot"].squeeze(0)  # xyzw, Euroc World

        R_imu_to_madg = torch.tensor([
            [0., 0., 1.],
            [0., 1., 0.],
            [-1., 0., 0.]
        ], dtype=torch.float64) # 가속도 벡터 변환 확인함 , 확정 (URF to FRD)
        R_gt_to_imu = torch.tensor([
            [0.,  0., 1.],
            [0., -1., 0.],
            [1.,  0., 0.]
        ], dtype=torch.float64)
        R_gt_to_madg = R_imu_to_madg @ R_gt_to_imu

        rot_gt = R.from_quat(quats_gt) # xyzw -> rot
        rot_gt_madglocal = R.from_matrix(R_gt_to_madg.cpu().numpy()) * rot_gt # Madg local -> Euroc GT

        quats = Madgwick(gyr=gyro, acc=acc, frequency= 200.0, q0=(rot_gt_madglocal[0].as_quat())[[3,0,1,2]]).Q # input : Eu local, output : Eu global -> Eu local #wxyz
        rot_est = R.from_quat(quats[:,[1,2,3,0]])
        rot_est_aligned = rot_gt_madglocal[0] * rot_est[0].inv() * rot_est

        final_rot_est = rot_gt[0] * rot_est_aligned[0].inv() * rot_est_aligned # rot_gt 좌표계에 맞춰 표기하기 위함


        #resultdict = metric(filename,  rot_est_aligned, rot_gt_madglocal, show=True)
        resultdict = metric(filename,  final_rot_est, rot_gt, show=True)
        orientations = {
            'name' : filename,
            'timestamp' : ts.cpu().tolist(),
            'qx' : R.as_quat(final_rot_est)[:,0].tolist(),
            'qy': R.as_quat(final_rot_est)[:, 1].tolist(),
            'qz': R.as_quat(final_rot_est)[:, 2].tolist(),
            'qw': R.as_quat(final_rot_est)[:, 3].tolist()
        }
        resultdict_madg.append(resultdict)
        orientations_madg.append(orientations)
    return resultdict_madg, orientations_madg

def Mahony_EuRoc(loader, confs, resultdict_maho, orientations_maho):
    for i, sample in enumerate(loader):
        seq_id = loader.dataset.index_map[i][0]
        filename = loader.dataset.seq_id_to_name[seq_id]

        ts = sample["timestamp"].squeeze(0)
        gyro = sample["gyro"].squeeze(0)  # Euroc IMU
        acc = sample["acc"].squeeze(0)  # Euroc IMU
        dt = sample["dt"].squeeze(0)  # 다 0.0050 확인
        quats_gt = sample["gt_rot"].squeeze(0)  # xyzw, Euroc World

        R_imu_to_madg = torch.tensor([
            [0., 0., 1.],
            [0., 1., 0.],
            [-1., 0., 0.]
        ], dtype=torch.float64)  # 가속도 벡터 변환 확인함 , 확정 (URF to FRD)
        R_gt_to_imu = torch.tensor([
            [0., 0., 1.],
            [0., -1., 0.],
            [1., 0., 0.]
        ], dtype=torch.float64)
        R_gt_to_madg = R_imu_to_madg @ R_gt_to_imu

        rot_gt = R.from_quat(quats_gt)
        rot_gt_madglocal = R.from_matrix(R_gt_to_madg.cpu().numpy()) * rot_gt# Madg local -> Euroc GT

        quats = Mahony(gyr=gyro, acc=acc, frequency=200.0, q0=rot_gt_madglocal[0].as_quat()[
            [3, 0, 1, 2]]).Q  # input : Eu local, output : Eu global -> Eu local
        rot_est = R.from_quat(quats[:, [1, 2, 3, 0]])
        rot_est_aligned = rot_gt_madglocal[0] * rot_est[0].inv() * rot_est

        final_rot_est = rot_gt[0] * rot_est_aligned[0].inv() * rot_est_aligned

        # resultdict = metric(filename,  rot_est_aligned, rot_gt_madglocal, show=True)
        resultdict = metric(filename, final_rot_est, rot_gt, show=True)
        orientations = {
            'name': filename,
            'timestamp': ts.cpu().tolist(),
            'qx': R.as_quat(final_rot_est)[:, 0].tolist(),
            'qy': R.as_quat(final_rot_est)[:, 1].tolist(),
            'qz': R.as_quat(final_rot_est)[:, 2].tolist(),
            'qw': R.as_quat(final_rot_est)[:, 3].tolist()
        }
        resultdict_maho.append(resultdict)
        orientations_maho.append(orientations)
    return resultdict_maho, orientations_maho

save_path_madg = "/home/mpil/miniconda3/envs/filter/results/Madgwick_astrobee"
save_path_maho = "/home/mpil/miniconda3/envs/filter/results/Mahony_astrobee"

conf = ConfigFactory.parse_file("/home/mpil/miniconda3/envs/filter/configs/Filterconf_astrobee.conf")
dataset_conf = conf.dataset['test']

resultdict_madg = []
orientation_madg = []

dataset_test = SeqeuncesDataset(data_set_config=conf.dataset.test) # xyzw
loader = Data.DataLoader(dataset=dataset_test)

results_madg, orientations_madg = Madgwick_EuRoc(loader, conf, resultdict_madg, orientation_madg)

file_path_metric = os.path.join(save_path_madg, "metric_Madgwick.json")
file_path_ori = os.path.join(save_path_madg, "orientation_Madgwick.json")
with open(file_path_metric, "w") as f:
    json.dump(resultdict_madg, f, indent=4)
with open(file_path_ori, "w") as f:
    json.dump(orientation_madg, f, indent=4)

###########################################################################
resultdict_maho = []
orientation_maho = []

dataset_test = SeqeuncesDataset(data_set_config=conf.dataset.test)
loader = Data.DataLoader(dataset=dataset_test)

results_maho, orientations_maho = Mahony_EuRoc(loader, conf, resultdict_maho, orientation_maho)

file_path_metric = os.path.join(save_path_maho, "metric_Mahony.json")
file_path_ori = os.path.join(save_path_maho, "orientation_Mahony.json")
with open(file_path_metric, "w") as f:
    json.dump(resultdict_maho, f, indent=4)
with open(file_path_ori, "w") as f:
    json.dump(orientation_maho, f, indent=4)
