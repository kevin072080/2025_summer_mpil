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
    resultdict = {}
    roll_rmse, pitch_rmse, yaw_rmse = RPY(est_R, rot_gt)
    euler_est = est_R.as_euler('xyz', degrees=True)
    euler_gt = rot_gt.as_euler('xyz', degrees=True)
    if show:
        plt.figure(figsize=(10, 6))
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
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(est_R.as_quat()[:, [3, 0, 1, 2]][:, i], label='Estimated')
            plt.plot(rot_gt.as_quat()[:, [3, 0, 1, 2]][:, i], label='GT')
            plt.title(f'{filename}, q{i}')
            plt.legend()
        plt.tight_layout()
        plt.show()

    aoe = AOE_tensor(est_R, rot_gt)
    roe = ROE_tensor(est_R, rot_gt)

    print(roll_rmse, pitch_rmse, yaw_rmse)
    # print(quats, '\n', rot_est_global.as_quat()[:,[3,0,1,2]], '\n', rot_gt.as_quat()[:,[3,0,1,2]])
    # print(rot_est_global.as_quat()[:, [3, 0, 1, 2]], '\n', rot_gt.as_quat()[:, [3, 0, 1, 2]])


    print(f"File : {filename}, AOE (deg): {aoe:.6f}, ROE (deg): {roe.mean():.6f}")
    print("All ROE", roe, "\n")
    resultdict.setdefault(filename, []).append({
        'AOE (deg)': aoe.item(),
        'ROE_Mean (deg)': float(roe.mean()),
        'ROE': roe.tolist(),
        'rot_compensated': est_R.as_euler('xyz', degrees=True).tolist(),
        'rot_gt': rot_gt.as_euler('xyz', degrees=True).tolist()
    })
    return resultdict

def Madgwick_EuRoc(loader, confs, resultdict):
    for i, sample in enumerate(loader):
        seq_id = loader.dataset.index_map[i][0]
        filename = loader.dataset.seq_id_to_name[seq_id]

        ts = sample["timestamp"].squeeze(0)
        gyro = sample["gyro"].squeeze(0) # Euroc IMU
        acc = sample["acc"].squeeze(0) # Euroc IMU
        dt = sample["dt"].squeeze(0) # 다 0.0050 확인
        quats_gt = sample["gt_rot"].squeeze(0)  # wxyz, Euroc World
        #print('acc\n', acc[:10, :])

        R_flip = torch.tensor([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.]
        ], dtype=torch.float64)
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

        gyro_madglocal = gyro @ R_imu_to_madg.T # Madg local 원래 imu_to_madg
        acc_madglocal = acc @ R_imu_to_madg.T # Madg local
        #gyro = gyro @ R_flip
        #acc = acc @ R_flip

        rot_gt = R.from_quat(quats_gt[:, [1, 2, 3, 0]]) # Euroc World -> Euroc GT
        rot_gt_madglocal = R.from_matrix(R_gt_to_madg.cpu().numpy()) * R.from_quat(quats_gt[:, [1, 2, 3, 0]]) # Madg local -> Euroc GT
        rot_gt_eulocal = rot_gt_madglocal * R.from_matrix(R_gt_to_imu.cpu().numpy())
        #rot_gt_madglocal = rot_gt_madglocal[10000:,:]

        quats = Madgwick(gyr=gyro, acc=acc, frequency= 200.0, q0=rot_gt_madglocal[0].as_quat()[[3,0,1,2]]).Q # input : Eu local, output : Eu global -> Eu local
        #quats = Madgwick(gyr=gyro, acc=acc, frequency=200.0).Q  # input : Eu local, output : Eu global -> Eu local
        quats_madglocal = Madgwick(gyr=gyro_madglocal, acc=acc_madglocal, frequency= 200.0, q0=rot_gt_madglocal[0].as_quat()[[3,0,1,2]]).Q # input : madglocal
        # quats는 지구 좌표계에 대한 센서 좌표계의 오리엔테이션 반환  q0는 지구 좌표계에서의 gt 초기값이어야 함

        #quats = Madgwick(gyr=gyro[10000:,:], acc=acc[10000:,:], frequency= 200.0, q0=rot_gt_madglocal[0].as_quat()[[3,0,1,2]]).Q # wxyz Madg World

        rot_est = R.from_quat(quats[:,[1,2,3,0]]) * R.from_matrix(R_gt_to_imu.cpu().numpy())  # xyzw, Eu global -> Eu local
        rot_est = R.from_quat(quats[:,[1,2,3,0]])
        rot_est_aligned = rot_gt_madglocal[0] * rot_est[0].inv() * rot_est

        rot_est_madg = R.from_quat(quats_madglocal[:,[1,2,3,0]])
        rot_est_madg_aligned = rot_gt_madglocal[0] * rot_est_madg[0].inv() * rot_est_madg

        '''rot_est_Euglobal = R_madg_to_euroc * rot_est # Euroc global
        # 정렬 행렬: Madgwick Earth → EuRoC world
        rot_est_aligned = rot_gt[0] * rot_est_Euglobal[0].inv() * rot_est_Euglobal  # now aligned with EuRoC GT'''

        resultdict = metric(filename,  rot_est_aligned, rot_gt_madglocal, show=True)
        orientations = {
            'timestamp' : ts.cpu().tolist(),
            'qx' : rot_gt_madg
        }
    return resultdict, orientations

def Mahony_EuRoc(loader, confs, resultdict):
    for i, sample in enumerate(loader):
        seq_id = loader.dataset.index_map[i][0]
        filename = loader.dataset.seq_id_to_name[seq_id]
        gyro = sample["gyro"].squeeze(0)  # Euroc IMU
        acc = sample["acc"].squeeze(0)  # Euroc IMU
        dt = sample["dt"].squeeze(0)  # 다 0.0050 확인
        quats_gt = sample["gt_rot"].squeeze(0)  # wxyz, Euroc World
        # print('acc\n', acc[:10, :])

        R_flip = torch.tensor([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.]
        ], dtype=torch.float64)
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

        R_gt_to_madg = R.from_euler('x', 180, degrees=True)
        R_imu_to_gt = R_gt_to_madg.inv() * R.from_matrix(R_imu_to_madg.cpu().numpy())

        R_mtx = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.]
        ])
        R_madg_to_euroc = R.from_matrix(R_mtx)  # 확정

        gyro_madglocal = gyro @ R_imu_to_madg.T  # Madg local
        acc_madglocal = acc @ R_imu_to_madg.T  # Madg local
        # gyro = gyro @ R_flip
        # acc = acc @ R_flip

        rot_gt = R.from_quat(quats_gt[:, [1, 2, 3, 0]])  # Euroc World
        rot_gt_madglocal = R_gt_to_madg * R.from_quat(quats_gt[:, [1, 2, 3, 0]])  # Madg local
        rot_gt_eulocal = R.from_matrix(R_gt_to_imu.cpu().numpy()) * rot_gt
        # rot_gt_madglocal = rot_gt_madglocal[10000:,:]

        quats = Mahony(gyr=gyro, acc=acc, frequency=200.0,
                         q0=rot_gt_madglocal[0].as_quat()[[3, 0, 1, 2]]).Q  # input : Eu local
        quats_madglocal = Mahony(gyr=gyro_madglocal, acc=acc_madglocal, frequency=200.0,
                                   q0=rot_gt_madglocal[0].as_quat()[[3, 0, 1, 2]]).Q  # input : madglocal
        # quats는 지구 좌표계에 대한 센서 좌표계의 오리엔테이션 반환  q0는 지구 좌표계에서의 gt 초기값이어야 함

        # quats = Madgwick(gyr=gyro[10000:,:], acc=acc[10000:,:], frequency= 200.0, q0=rot_gt_madglocal[0].as_quat()[[3,0,1,2]]).Q # wxyz Madg World

        rot_est = R.from_quat(quats[:, [1, 2, 3, 0]])  # xyzw, Madg global frame (Earth frame)
        rot_est_aligned = rot_gt_madglocal[0] * rot_est[0].inv() * rot_est

        # rot_est_aligned = rot_gt_madglocal[0] * rot_est[0].inv() * rot_est

        rot_est_madg = R.from_quat(quats_madglocal[:, [1, 2, 3, 0]])
        rot_est_madg_aligned = rot_gt_madglocal[0] * rot_est_madg[0].inv() * rot_est_madg
        '''rot_est_Euglobal = R_madg_to_euroc * rot_est # Euroc global
        # 정렬 행렬: Madgwick Earth → EuRoC world
        rot_est_aligned = rot_gt[0] * rot_est_Euglobal[0].inv() * rot_est_Euglobal  # now aligned with EuRoC GT'''

        resultdict = metric(filename, rot_est_aligned, rot_gt_madglocal)
    return resultdict

#base_dir = "/home/mpil/Astrobee_dataset"
#datelist = os.listdir(base_dir)
save_path_madg = "/home/mpil/miniconda3/envs/filter/results/Madgwick_EuRoC_0814"
save_path_maho = "/home/mpil/miniconda3/envs/filter/results/Mahony_EuRoC/result.json"

conf = ConfigFactory.parse_file("/home/mpil/miniconda3/envs/filter/configs/Filterconf.conf")
#dataset_test = SeqeuncesDataset(data_set_config=conf.dataset.test)
#loader = Data.DataLoader(dataset = dataset_test)
dataset_conf = conf.dataset['test']

resultdict_madg = []
orientation_madg = []
for data_conf in dataset_conf.data_list:
    for data_name in data_conf.data_drive:
        print(data_conf.data_root, data_name)
        print("data_conf.dataroot", data_conf.data_root)
        print("data_name", data_name)
        print("data_conf.name", data_conf.name)

        dataset_test = SeqeuncesDataset(data_set_config=conf.dataset.test)
        loader = Data.DataLoader(dataset=dataset_test)

        results_madg, orientations = Madgwick_EuRoc(loader, conf, resultdict_madg)
        resultdict_madg.append(results_madg)
        orientation_madg.append(orientations)

#resultdict_madg = {}
#results_madg = Madgwick_EuRoc(loader, conf, resultdict_madg)
file_path_metric = os.path.join(save_path_madg, "metric_Madgwick.json")
file_path_ori = os.path.join(save_path_madg, "orientation_Madgwick.json")
with open(file_path_metric, "w") as f:
    json.dump(resultdict_madg, f, indent=4)
with open(file_path_metric, "w") as f:
    json.dump(orientation_madg, f, indent=4)

'''resultdict_maho = {}
results_maho = Mahony_EuRoc(loader, conf, resultdict_maho)
with open(save_path_maho, "w") as f:
    json.dump(results_maho, f, indent=4)

'''