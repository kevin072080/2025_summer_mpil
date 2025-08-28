import os
import time
from os import path as osp

import numpy as np
import quaternion
import torch
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_glob_speed import * # frame 바꿔가면서 test하려면 data_glob_speed_(bodyframe,bodyremoveg,globalremoveg)
from transformations import *
from metric import compute_ate_rte
from model_resnet1d import *

_input_channel, _output_channel = 6, 3 #output channel 2 -> 3
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}
# fc_dim은 hyperparam. -> 왜 resnet18은 1024가 아닌지는 모름
# in_dim = window_size // 32 + 1, window_size default == 200 -> in_dim default 7

def get_model(arch):
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1536 #512 units * 2D -> 512 * 3D
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1536 #512 units * 2D -> 512 * 3D
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config) # 잘 안 먹히면 basic 대신 bottleneck 해보기
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network


def run_test(network, data_loader, device, eval_mode=True): # 오직 val test만을 위한 model, back prop 통한 param update X
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, data in enumerate(data_loader):
        (feat, targ, _, _) = data
        # print('f, t ',feat.shape, targ.shape, len(data))
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def get_dataset(root_dir, ori_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train': # random shuffle 및 shift, transform 통해 섞어서 data num 증가
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2) #Random HACF -> 2D일 때와 3D일 때 다름
    elif mode == 'val': # shuffle만 적용하여 training이 그 세트에 대해 valid한지 여부 판단
        shuffle = True
    elif mode == 'test': #game rotation vector만 사용하여 시험
        shuffle = False
        grv_only = True
    if args.ori_dir == None:
        if args.dataset == 'ronin':
            seq_type = GlobSpeedSequence
        elif args.dataset == 'ridi':
            from data_ridi import RIDIGlobSpeedSequence
            seq_type = RIDIGlobSpeedSequence
        print(data_list)
        dataset = StridedSequenceDataset(
            seq_type, root_dir, None, data_list, args.cache_path, args.step_size, args.window_size,
            random_shift=random_shift, transform=transforms,
            shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)
            # feat = self.features[seq_id][frame_id:frame_id + self.window_size]
            # targ = self.targets[seq_id][frame_id]

    else:
        seq_type = GlobSpeedSequence
        print(args.ori_dir)
        dataset = StridedSequenceDataset(
            seq_type, root_dir, ori_dir, data_list, args.cache_path, args.step_size, args.window_size,
            random_shift=random_shift, transform=transforms,
            shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)
        # feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        # targ = self.targets[seq_id][frame_id]

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, ori_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, ori_dir, data_list, args, **kwargs)


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.root_dir, args.ori_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    end_t = time.time()
    print('Training set loaded. Feature size: {}, target size: {}. Time usage: {:.3f}s'.format(
        train_dataset.feature_dim, train_dataset.target_dim, end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir,args.ori_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch).to(device)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-12)

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    step = 0
    best_val_loss = np.inf

    print('Start from epoch {}'.format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device)
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        for epoch in range(start_epoch, args.epochs):
            start_t = time.time()
            network.train()
            train_outs, train_targets = [], []
            for batch_id, (feat, targ, _, _) in enumerate(train_loader):
                feat, targ = feat.to(device), targ.to(device)
                # feat = self.features[seq_id][frame_id:frame_id + self.window_size] window size 동안의 Gyro, IMU data
                # targ = self.targets[seq_id][frame_id] tango 활용한 GT
                optimizer.zero_grad()
                pred = network(feat) # feat 활용한 v 예측
                train_outs.append(pred.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                loss = criterion(pred, targ) #    criterion = torch.nn.MSELoss()
                loss = torch.mean(loss)
                loss.backward() #back propagation
                optimizer.step()   #grad 바탕으로 param. update by ADAM
                step += 1
            train_outs = np.concatenate(train_outs, axis=0) # 배치별 계산 결과 이어붙이기
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.average(train_losses)))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(network, val_loader, device)
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
            else:
                if args.out_dir is not None and osp.isdir(args.out_dir):
                    model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                    torch.save({'model_state_dict': network.state_dict(),
                                'epoch': epoch,
                                'optimizer_state_dict': optimizer.state_dict()}, model_path)
                    print('Model saved to ', model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all


def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id] # timestamp
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 3]) # 2-> 3, 첫 2는 시작 및 끝 보간용
    pos[0] = dataset.gt_pos[seq_id][0, :3]  # 2-> 3, 초기 위치를 gt 값에서 얻어 옴
    pos[1:-1] = np.cumsum(preds[:, :3] * dts, axis=0) + pos[0]  # 2-> 3, 변위 누적하는 부분임, 1에서 n-1까지
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0) # 0, n 변위 임의 추가

    pos = interp1d(ts_ext, pos, axis=0)(ts) # timestamp 맞춰서 정렬
    return pos


    # print('recon에서의 rotq :',startframe_tango_ori * rot_imu_to_tango_startframe * ori[0].conj() )
    # 아무 것도 안 한 ori를 쓰는 것이 가장 잘 나옴, 방향만 이상 -> 학습은 잘 됐고, recon 이상이라고 판단 -> ori가 0이 아닌 startframe index부터임을 확인 -> 고쳤으나 잘 안 됨
    # index 바꾸기 전 상태로 resnet101 학습 돌리는 중

def recon_traj_with_preds_body(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    imu body frame predict -> tango global frame
    """

    ts = dataset.ts[seq_id]  # timestamp, seq_id : data id
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=int)
    ori = quaternion.from_float_array(dataset.orientations[seq_id][ind]) # start_frame 부터
    rot_imu_to_tango_startframe = dataset.seq_objects[seq_id].rot_imu_to_tango_startframe
    startframe_tango_ori = dataset.seq_objects[seq_id].startframe_tango_ori # start_frame 부터
    #rot_imu_to_tango = quaternion.quaternion(*dataset.seq_objects[seq_id].info['start_calibration'])

    ori_modi = startframe_tango_ori * rot_imu_to_tango_startframe * ori[0].conj() * ori

    preds = quaternion.from_float_array(np.concatenate([np.zeros((preds.shape[0], 1)), preds], axis=1))
    preds = ori_modi * preds * ori_modi.conj()
    preds = quaternion.as_float_array(preds)
    preds = preds[:, 1:]

    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 3]) # 2-> 3, 첫 2는 시작 및 끝 보간용
    pos[0] = dataset.gt_pos[seq_id][0, :3]  # 2-> 3, 초기 위치를 gt 값에서 얻어 옴
    pos[1:-1] = np.cumsum(preds[:, :3] * dts, axis=0) + pos[0]  # 2-> 3, 변위 누적하는 부분임, 1에서 n-1까지 / quaternion에서 변환했으므로 1,2,3열
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0) # 0, n 변위 임의 추가
    pos = interp1d(ts_ext, pos, axis=0)(ts) # timestamp 맞춰서 정렬
    return pos


def test_sequence(args):
    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.root_dir
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, args.ori_dir,  [test_data_list[0]], args)

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1 # output data channel X, 한 channel 당 time-domain length

    network = get_model(args.arch)

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60

    for data in test_data_list:
        seq_dataset = get_dataset(root_dir, args.ori_dir, [data], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False) # batch size는 hyperparam. 너무 느린데 수정 필요?
        ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=int)

        targets, preds = run_test(network, seq_loader, device, True)
        losses = np.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :3] # 2 -> 3 / global frame
        #pos_pred = recon_traj_with_preds_body(seq_dataset, preds)[:, :3]  # 2 -> 3 / body frame
        pos_gt = seq_dataset.gt_pos[0][:, :3] # 2-> 3

        traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
        ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

        print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

        # Plot figures
        kp = preds.shape[1]
        if kp == 2:
            targ_names = ['vx', 'vy']
        elif kp == 3:
            targ_names = ['vx', 'vy', 'vz']

        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 3), (0, 1), rowspan=kp - 1)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.title('X-Y')
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])

        plt.subplot2grid((kp, 3), (0, 0), rowspan=kp)
        plt.plot(pos_pred[:, 0], pos_pred[:, 2])
        plt.plot(pos_gt[:, 0], pos_gt[:, 2])
        plt.title('X-Z')
        plt.legend(['Predicted', 'Ground truth'])

        plt.subplot2grid((kp, 3), (kp - 1, 1))
        plt.plot(pos_cum_error)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
        for i in range(kp):
            plt.subplot2grid((kp, 3), (i, 2))
            plt.plot(ind, preds[:, i])
            plt.plot(ind, targets[:, i])
            plt.legend(['Predicted', 'Ground truth'])
            plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
        plt.tight_layout()
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], label='trajectory', color='blue')
        ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], label='trajectory', color='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory')
        ax.legend()

        plt.show()
        '''

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                    np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
            plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

        plt.close('all')

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    # Export a csv file
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
            if losses_seq.shape[1] == 2:
                f.write('seq,vx,vy,avg,ate,rte\n')
            else:
                f.write('seq,vx,vy,vz,avg,ate,rte\n')
            for i in range(losses_seq.shape[0]):
                f.write('{},'.format(test_data_list[i]))
                for j in range(losses_seq.shape[1]):
                    f.write('{:.6f},'.format(losses_seq[i][j]))
                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

    print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
        np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
    return losses_avg


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--ori_dir', type=str, default=None, help='Path to estimated ori data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--arch', type=str, default='resnet101') # resnet18 -> resnet101
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_sequence(args)
    else:
        raise ValueError('Undefined mode')
