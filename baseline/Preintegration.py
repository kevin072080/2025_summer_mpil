import argparse
import os

from sympy.polys.matrices.dense import ddm_ilu

#import tqdm, wandb

from datasets import collate_fcs,SeqDataset, imu_seq_collate, SeqInfDataset

import numpy as np
import pypose as pp
import torch
import torch.utils.data as Data
from utils import integrate, CPU_Unpickler
from pyhocon import ConfigFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu, Default is cuda:0")
    parser.add_argument("--seqlen", type=int, default="200", help="the length of the integration sequence")
    parser.add_argument("--dataconf", type=str, default="Air_IO/configs/datasets/EuRoC/Euroc_global.conf",
                        help="the configuration of the dataset")
    parser.add_argument("--usegtrot", action="store_true", help="Use ground truth rotation for gravity compensation")
    parser.add_argument("--linacce", default=False, action="store_true", help="Use linacce, grav = 0")
    parser.add_argument("--exp", type=str, default=None,
                        help="the directory path where your network output pickle file is stored")

    args = parser.parse_args();
    print(("\n" * 3) + str(args) + ("\n" * 3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference

    if args.exp is not None:
        net_result_path = os.path.join(args.exp, 'net_output.pickle')
        if os.path.isfile(net_result_path):
            with open(net_result_path, 'rb') as handle:
                inference_state_load = CPU_Unpickler(handle).load()
                print('Inference pickle loaded from ', args.exp)
        else:
            raise Exception(f"Unable to load the network result: {net_result_path}")

    for data_conf in dataset_conf.data_list:
        for data_name in data_conf.data_drive:
            print(data_conf.data_root, data_name)
            print("Dataroot", data_conf.data_root)
            print("Sequence_name", data_name)
            print("Dataset_name", data_conf.name)

            if args.linacce == True:
                dataset_conf['remove_g'] = True
                print('g has to be removed')


            dataset = SeqDataset(data_conf.data_root, data_name, args.device, name=data_conf.name,
                                 duration=args.seqlen,
                                 step_size=args.seqlen, drop_last=False, conf=dataset_conf)
            # print('evaluate_state level key ',dataset[0].keys())
            loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False,
                                     drop_last=False)
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()

            if data_conf.name == 'Astrobee': # vel X
                # gravity = dataset.get_gravity()
                init_vel = (dataset.data['gt_translation'][1] - dataset.data['gt_translation'][0]) / dataset.data['dt'][0, None]
                init['vel'] = init_vel
                integrator_outstate = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'], init['vel'], gravity=0.,
                    reset=False
                ).to(args.device).double()
                integrator_reset = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'], init['vel'], gravity=0.,
                    reset=True
                ).to(args.device).double()  # 적분할 때마다 reset -> relative orientation

                outstate = integrate(
                    integrator_outstate, loader, init,
                    device=args.device, gtinit=True, save_full_traj=True,
                    use_gt_rot=True
                )  # integrated orietnation_full trajectory # xyzw
                relative_outstate = integrate(
                    integrator_reset, loader, init,
                    device=args.device, gtinit=True,
                    use_gt_rot=True
                )  # relative orientation_last 값들만 저장
                if args.exp is not None:  # linacc, corrected
                    inference_state = inference_state_load[data_name]
                    dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device=args.device,
                                                name=data_conf.name, duration=args.seqlen, step_size=args.seqlen,
                                                drop_last=False, conf=dataset_conf)
                    infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1,
                                                collate_fn=imu_seq_collate,
                                                shuffle=False, drop_last=True)
                    init_vel_inf = (dataset_inf.data['gt_translation'][1] - dataset_inf.data['gt_translation'][0]) / \
                               dataset_inf.data['dt'][0, None]
                    init['vel'] = init_vel_inf
                    integrator_infstate = pp.module.IMUPreintegrator(
                        init['pos'], init['rot'], init['vel'], gravity=gravity,
                        reset=False
                    ).to(args.device).double()

                    infstate = integrate(
                        integrator_infstate, infloader, init,
                        device=args.device, gtinit=True, save_full_traj=True,
                        use_gt_rot=args.usegtrot
                    )
            else: # vel O
                init_vel = (dataset.data['gt_translation'][1] - dataset.data['gt_translation'][0]) / dataset.data['dt'][
                    0, None]
                print(init_vel, init['vel'])
                integrator_outstate = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'],
                    init['vel'],
                    gravity=0.,
                    reset=False
                ).to(args.device).double()
                integrator_reset = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'],
                    init['vel'],
                    gravity=0.,
                    reset=True
                ).to(args.device).double()  # 적분할 때마다 reset -> relative orientation

                outstate = integrate(
                    integrator_outstate, loader, init,
                    device=args.device, gtinit=True, save_full_traj=True,
                    use_gt_rot=True
                )  # integrated orietnation_full trajectory # xyzw
                relative_outstate = integrate(
                    integrator_reset, loader, init,
                    device=args.device, gtinit=True,
                    use_gt_rot=True
                )  # relative orientation_last 값들만 저장
                if args.exp is not None: # linacc, corrected
                    inference_state = inference_state_load[data_name]
                    dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device=args.device,
                                                name=data_conf.name, duration=args.seqlen, step_size=args.seqlen,
                                                drop_last=False, conf=dataset_conf)
                    infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1,
                                                collate_fn=imu_seq_collate,
                                                shuffle=False, drop_last=True)

                    integrator_infstate = pp.module.IMUPreintegrator(
                        init['pos'], init['rot'], init['vel'], gravity=gravity,
                        reset=False
                    ).to(args.device).double()

                    infstate = integrate(
                        integrator_infstate, infloader, init,
                        device=args.device, gtinit=True, save_full_traj=True,
                        use_gt_rot=args.usegtrot
                    )



            '''else: # raw acc
                # raw, uncorrected
                dataset = SeqDataset(data_conf.data_root, data_name, args.device, name=data_conf.name,
                                     duration=args.seqlen,
                                     step_size=args.seqlen, drop_last=False, conf=dataset_conf)
                # print('evaluate_state level key ',dataset[0].keys())
                loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False,
                                         drop_last=False)
                init = dataset.get_init_value()
                gravity = dataset.get_gravity()

                integrator_outstate = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'],
                    init['vel'],
                    gravity=0.,
                    reset=False
                ).to(args.device).double()
                integrator_reset = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'],
                    init['vel'],
                    gravity=0.,
                    reset=True
                ).to(args.device).double()  # 적분할 때마다 reset -> relative orientation

                outstate = integrate(
                    integrator_outstate, loader, init,
                    device=args.device, gtinit=True, save_full_traj=True,
                    use_gt_rot=True
                )  # integrated orietnation_full trajectory # xyzw
                relative_outstate = integrate(
                    integrator_reset, loader, init,
                    device=args.device, gtinit=True,
                    use_gt_rot=True
                )  # relative orientation_last 값들만 저장

                if args.exp is not None:  # raw, corrected
                    inference_state = inference_state_load[data_name]
                    dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device=args.device,
                                                name=data_conf.name, duration=args.seqlen, step_size=args.seqlen,
                                                drop_last=False, conf=dataset_conf)
                    infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1,
                                                collate_fn=imu_seq_collate,
                                                shuffle=False, drop_last=True)

                    integrator_infstate = pp.module.IMUPreintegrator(
                        init['pos'], init['rot'], init['vel'], gravity=gravity,
                        reset=False
                    ).to(args.device).double()

                    infstate = integrate(
                        integrator_infstate, infloader, init,
                        device=args.device, gtinit=True, save_full_traj=True,
                        use_gt_rot=args.usegtrot
                    )
                    relative_infstate = integrate(
                        integrator_reset, infloader, init,
                        device=args.device, gtinit=True,
                        use_gt_rot=args.usegtrot
                    )'''



            if args.exp is not None:
                infstate['timestamp'] = torch.cat(
                    [infstate['timestamp'],
                     (infstate['timestamp'][0, -1] + 5e6).view(1, 1)],
                    dim=1
                )  # 차원 맞추기 용, 마지막에 값 추가

                # timestamp
                ts = infstate['timestamp'][0].cpu().tolist()

                # estimated poses (x, y, z)
                est_posx = infstate['poses'][0, :, 0].cpu().tolist()
                est_posy = infstate['poses'][0, :, 1].cpu().tolist()
                est_posz = infstate['poses'][0, :, 2].cpu().tolist()

                # estimated orientations (qx, qy, qz, qw)
                est_qx = infstate['orientations'][0, :, 0].cpu().tolist()
                est_qy = infstate['orientations'][0, :, 1].cpu().tolist()
                est_qz = infstate['orientations'][0, :, 2].cpu().tolist()
                est_qw = infstate['orientations'][0, :, 3].cpu().tolist()

                # ground truth poses (x, y, z)
                gt_posx = infstate['poses_gt'][0, :, 0].cpu().tolist()
                gt_posy = infstate['poses_gt'][0, :, 1].cpu().tolist()
                gt_posz = infstate['poses_gt'][0, :, 2].cpu().tolist()

                # ground truth orientations (qx, qy, qz, qw)
                gt_qx = infstate['orientations_gt'][0, :, 0].cpu().tolist()
                gt_qy = infstate['orientations_gt'][0, :, 1].cpu().tolist()
                gt_qz = infstate['orientations_gt'][0, :, 2].cpu().tolist()
                gt_qw = infstate['orientations_gt'][0, :, 3].cpu().tolist()
            else:
                outstate['timestamp'] = torch.cat(
                    [outstate['timestamp'],
                     (outstate['timestamp'][0, -1] + 5e6).view(1, 1)],
                    dim=1
                )  # 차원 맞추기 용, 마지막에 값 추가

                # timestamp
                ts = outstate['timestamp'][0].cpu().tolist()

                # estimated poses (x, y, z)
                est_posx = outstate['poses'][0, :, 0].cpu().tolist()
                est_posy = outstate['poses'][0, :, 1].cpu().tolist()
                est_posz = outstate['poses'][0, :, 2].cpu().tolist()

                # estimated orientations (qx, qy, qz, qw)
                est_qx = outstate['orientations'][0, :, 0].cpu().tolist()
                est_qy = outstate['orientations'][0, :, 1].cpu().tolist()
                est_qz = outstate['orientations'][0, :, 2].cpu().tolist()
                est_qw = outstate['orientations'][0, :, 3].cpu().tolist()

                # ground truth poses (x, y, z)
                gt_posx = outstate['poses_gt'][0, :, 0].cpu().tolist()
                gt_posy = outstate['poses_gt'][0, :, 1].cpu().tolist()
                gt_posz = outstate['poses_gt'][0, :, 2].cpu().tolist()

                # ground truth orientations (qx, qy, qz, qw)
                gt_qx = outstate['orientations_gt'][0, :, 0].cpu().tolist()
                gt_qy = outstate['orientations_gt'][0, :, 1].cpu().tolist()
                gt_qz = outstate['orientations_gt'][0, :, 2].cpu().tolist()
                gt_qw = outstate['orientations_gt'][0, :, 3].cpu().tolist()

            # save timestamp, qx, qy, qz, qw
            # print(outstate['timestamp'].shape, outstate['orientations'].shape, mask.shape, select_mask.shape, 't, o, m')


            dataset = data_conf.name
            if args.exp is not None:
                if args.linacce == True:
                    savedir = os.path.join('/home/mpil/miniconda3/envs/baseline/result_Preintegration_linacce', dataset, 'AirIMU',
                                           'AirIMU_linacce_' + dataset + '_' + data_name + '_preinte.npz')
                    os.makedirs(os.path.dirname(savedir), exist_ok=True)
                else:
                    savedir = os.path.join('/home/mpil/miniconda3/envs/baseline/result_Preintegration_acc', dataset, 'AirIMU',
                                           'AirIMU_acc_' + dataset + '_' + data_name + '_preinte.npz')
                    os.makedirs(os.path.dirname(savedir), exist_ok=True)
            else:
                if args.linacce == True:
                    savedir = os.path.join('/home/mpil/miniconda3/envs/baseline/result_Preintegration_linacce', dataset, 'Raw',
                                           'linacce_' + dataset + '_' + data_name + '_preinte.npz')
                    os.makedirs(os.path.dirname(savedir), exist_ok=True)
                else:
                    savedir = os.path.join('/home/mpil/miniconda3/envs/baseline/result_Preintegration_acc', dataset, 'Raw',
                                           'acc_' + dataset + '_' + data_name + '_preinte.npz')
                    os.makedirs(os.path.dirname(savedir), exist_ok=True)


            #print(outstate.keys())
            # print(outstate['timestamp'].shape, outstate['poses'].shape, outstate['orientations'].shape)


            np.savez(
                savedir,
                ts=ts,
                est_posx=est_posx,
                est_posy=est_posy,
                est_posz=est_posz,
                est_qx=est_qx,
                est_qy=est_qy,
                est_qz=est_qz,
                est_qw=est_qw,
                gt_posx=gt_posx,
                gt_posy=gt_posy,
                gt_posz=gt_posz,
                gt_qx=gt_qx,
                gt_qy=gt_qy,
                gt_qz=gt_qz,
                gt_qw=gt_qw,
            )

            # npz 파일 불러오기
            data = np.load(savedir)

            # 키 리스트 확인
            print("Keys:", data.files)

            # 각 키별 내용 확인
            for key in data.files:
                print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
                print(data[key])  # 실제 값까지 보고 싶으면
