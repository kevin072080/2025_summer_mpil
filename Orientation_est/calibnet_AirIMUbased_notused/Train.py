import os
import torch
import numpy as np
import torch.utils.data as Data
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.spatial.transform import Rotation as R
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert
import argparse
import tqdm, wandb
import pypose as pp

from utils import move_to
from model import net_dict
from datasets import SeqeuncesDataset, collate_fcs, SeqDataset, imu_seq_collate
from Loss import multiscaleloss


#torch.autograd.set_detect_anomaly(True) # For Debugging

def axis_angle_to_quat(theta):  # theta: (B, 3)
    angle = torch.norm(theta, dim=1, keepdim=True) + 1e-8  # (B, 1), 회전 각도
    axis = theta / angle  # 회전 축 (B, 3)
    half_angle = angle * 0.5
    quat = torch.cat([
        torch.cos(half_angle),                 # w 성분
        axis * torch.sin(half_angle)          # x, y, z 성분
    ], dim=1)  # → shape: (B, 4)
    return quat

def quat_mul(q1, q2):  # (B, 4) * (B, 4)
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=1)  # (B, 4)


def train(network, loader, confs, epoch, optimizer):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses
    """
    network.train()
    losses = 0
    all_preds = []

    t_range = tqdm.tqdm(loader) #loader : SeqeuncesDataset(Data.Dataset) -> B, _, T 꼴로 나오도록 바꿔놓음
    for i, (data, init_state, label) in enumerate(t_range): # i 일정해도 window는 움직임, for loop 넘어가는 건 한 epoch 끝나는 시점
        data, init_state, label = move_to([data, init_state, label], confs.device)
        # data_dt, acc, gyro, rot / init_rot, pos, vel / label_gt_pos, gt_rot, gt_vel

        # imu = torch.cat((data['gyro'].permute(0,2,1), data['acc'].permute(0,2,1)), axis=1)
        # dt = data['dt'].permute(0,2,1)
        # gt_rot = label['gt_rot'].permute(0,2,1)

        imu = torch.cat((data['gyro'], data['acc']), axis=1)
        dt = data['dt']
        targ = label['gt_rot']
        init_rot = targ[:,:,0]

        pred_w = network(imu)

        B, _, T = pred_w.shape
        q_t = init_rot
        pred = torch.zeros((B,4,T), device=confs.device)
        pred[:,:,0] = init_rot
        for t in range(1,T):
            drot = pred_w[:,:,t] * dt[:,:,t]  # 미소 각속도와 시간 곱하기 (적분)
            dq = axis_angle_to_quat(drot)   # (B, 4), 각 변화량 쿼터니언 변환

            q_t = quat_mul(q_t, dq) # 이전 q_t에 변화량 dq 적용
            q_t = q_t.clone() if isinstance(q_t, pp.LieTensor) else q_t
            dq = dq.clone() if isinstance(dq, pp.LieTensor) else dq

            pred[:, :, t] = quat_mul(q_t, dq)

        loss_tensor = multiscaleloss(pred, targ, 2) # 각 batch의 loss값 갖는 tensor, tensor[128,]
        losses = loss_tensor.mean() # batch의 loss 평균, scalar

        t_range.set_description(f"[Train] Epoch {epoch} | Loss for each batch: {losses / (i + 1):.6f}")

        # t_range.refresh()
        optimizer.zero_grad() #pytorch에서는 기본적으로 grad가 누적 -> 매 학습 batch마다 초기화 필요
        losses.backward() # scalar losses -> back propagation
        optimizer.step() # grad에 저장된 gradient 바탕으로 param. update

    return {"loss": (losses / (i + 1))} # loss 다 더한 다음에 그 수로 나눠서 평균 구하기

def eval(network, loader, confs, data_name):
    network.eval()
    with torch.no_grad():
        AOE, ROE = 0, 0

        t_range = tqdm.tqdm(loader)  # loader : SeqeuncesDataset(Data.Dataset)
        for i, sample in enumerate(
                t_range):  # i 일정해도 window는 움직임, for loop 넘어가는 건 한 epoch 끝나는 시점
            sample = move_to([sample], confs.device)

            # imu = torch.cat((data['gyro'].permute(0,2,1), data['acc'].permute(0,2,1)), axis=1)
            imu = torch.cat((sample['gyro'], sample['acc']), axis=1)
            # dt = data['dt'].permute(0,2,1)
            # gt_rot = label['gt_rot'].permute(0,2,1)
            dt = sample['dt']
            gt_rot = sample['gt_rot']
            init_rot = gt_rot[:, :, 0]

            pred_w = network(imu)
            B, _, T = pred_w.shape
            q_t = init_rot

            pred = torch.zeros((B, 4, T), device=confs.device)
            pred[:, :, 0] = init_rot
            for t in range(T):
                drot = pred_w[:, :, t] * dt[:, :, t]
                # print(pred_w[:,:,t].shape, dt[:,:,t].shape, drot.shape)
                dq = axis_angle_to_quat(drot)  # (B, 4)
                # print(q_t.shape, dq.shape)
                if isinstance(q_t, pp.LieTensor):
                    q_t = q_t.tensor()

                if isinstance(dq, pp.LieTensor):
                    dq = dq.tensor()

                q_t = quat_mul(q_t, dq)
                pred[:, :, t] = q_t

            pred = pred.permute(0,2,1)
            pred = pred.reshape(-1,4).cpu().numpy()

            targ = sample['gt_rot']
            targ = targ.permute(0, 2, 1)
            targ = targ.reshape(-1,4).cpu().numpy()

            rot_est = R.from_quat(pred)
            rot_gt = R.from_quat(targ)

            rel_rot = rot_est.inv() * rot_gt
            rotvec = torch.tensor(rel_rot.as_rotvec())
            angles = torch.norm(rotvec, dim=1)
            AOE = torch.sqrt(torch.mean(angles**2)) * 180 / np.pi

            roe = []
            accum_t = 0
            last_idx = 0
            for i in range(len(rot_est)):
                if accum_t > 5:  # ms 단위 기준 (5s마다 계산)
                    drot_est = rot_est[last_idx].inv() * rot_est[i]
                    drot_gt = rot_gt[last_idx].inv() * rot_gt[i]
                    rel_rot = drot_est.inv() * drot_gt
                    rotvec = rel_rot.as_rotvec()
                    roe.append(torch.norm(torch.tensor(rotvec)).item() * 180 / np.pi)
                    last_idx = i  # 기준점 업데이트
                    accum_t = 0
                accum_t += 1 / 200  # 5ms per frame -> 5초마다 계산

            metricdict = {
                'name': data_name,
                'AOE (deg)' : AOE,
                'ROE (deg)' : roe.mean(),
                'All ROE (deg)' : roe.cpu().tolist()
            }
            oridict = {
                'name' : data_name,
                'timestamp' : sample['timestamp'].cpu().tolist(),
                'qx' : rot_est.as_quat()[:,0],
                'qy': rot_est.as_quat()[:, 1],
                'qz': rot_est.as_quat()[:, 2],
                'q2': rot_est.as_quat()[:, 3],
            }
    return metricdict, oridict

def test(network, loader, confs):
    network.eval()
    with torch.no_grad():
        losses = 0

        t_range = tqdm.tqdm(loader)  # loader : SeqeuncesDataset(Data.Dataset)
        for i, (data, init_state, label) in enumerate(
                t_range):  # i 일정해도 window는 움직임, for loop 넘어가는 건 한 epoch 끝나는 시점
            data, init_state, label = move_to([data, init_state, label], confs.device)

            # imu = torch.cat((data['gyro'].permute(0,2,1), data['acc'].permute(0,2,1)), axis=1)
            imu = torch.cat((data['gyro'], data['acc']), axis=1)
            # dt = data['dt'].permute(0,2,1)
            # gt_rot = label['gt_rot'].permute(0,2,1)
            dt = data['dt']
            gt_rot = label['gt_rot']
            init_rot = gt_rot[:, :, 0]

            pred_w = network(imu)
            B, _, T = pred_w.shape
            q_t = init_rot

            pred = torch.zeros((B, 4, T), device=confs.device)
            pred[:, :, 0] = init_rot
            for t in range(T):
                drot = pred_w[:, :, t] * dt[:, :, t]
                # print(pred_w[:,:,t].shape, dt[:,:,t].shape, drot.shape)
                dq = axis_angle_to_quat(drot)  # (B, 4)
                # print(q_t.shape, dq.shape)
                if isinstance(q_t, pp.LieTensor):
                    q_t = q_t.tensor()

                if isinstance(dq, pp.LieTensor):
                    dq = dq.tensor()

                q_t = quat_mul(q_t, dq)
                pred[:, :, t] = q_t

            targ = label['gt_rot']

            loss_tensor = multiscaleloss(pred, targ, 2)
            losses = loss_tensor.mean()

            t_range.set_description(f"[Test] Epoch {epoch} | Loss: {losses / (i + 1):.6f}")


    return {"loss": (losses / (i + 1))}  # loss 다 더한 다음에 그 수로 나눠서 평균 구하기

def write_wandb(header, objs, epoch_i): # wandb라는 웹 통해 모델 추이 확인
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                wandb.log({os.path.join(header, k): v}, epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)


def save_ckpt(network, optimizer, scheduler, epoch_i, test_loss, conf, save_best=False):
    # state_dict : parameter save
    if epoch_i % conf.train.save_freq == conf.train.save_freq - 1:  # epoch_i : epoch id -> save_freq 배수의 직전 epoch (icheckpoint같은 개념인 듯)
        torch.save({
            'epoch': epoch_i,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt" % epoch_i))

    if save_best:
        print("saving the best model", test_loss)
        torch.save({
            'epoch': epoch_i,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt"))

    torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
    }, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))  # 계속 갱신

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp/EuRoC/convnet.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")
    parser.add_argument('--load_ckpt', default=False, action="store_true", help="If True, try to load the newest.ckpt in the \
                                                                                exp_dir specificed in our config file.")
    parser.add_argument('--log', default=True, action="store_false", help="if True, save the meta data with wandb")
    parser.add_argument('--eval', default=False, action="store_true", help="if True, save the metric")
    args = parser.parse_args();
    print(args)
    conf = ConfigFactory.parse_file(args.config)
    # torch.cuda.set_device(args.device)

    # 경로 지정
    conf.train.device = args.device
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)

    if 'collate' in conf.dataset.keys(): # dataset에서 acc, gyro, vel 등끼리 묶어서 정리한 dict 만들기, 원래는 timestamp 별로 정리됨
        collate_fn = collate_fcs[conf.dataset.collate]
    else:
        collate_fn = collate_fcs['base']

    train_dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesDataset(data_set_config=conf.dataset.eval)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=conf.train.batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=4,             # 일반 CPU는 4~8 적절
    pin_memory=True,           # GPU로 전송 속도 향상
    persistent_workers=True )
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=conf.train.batch_size, shuffle=False,
                                  collate_fn=collate_fn)
    eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))



    ## optimizer
    print(train_dataset.get_dtype())  # 예상: torch.float32, torch.float64 등
    print(args.device)  # 예상: 'cuda:0' 또는 'cpu'

    network = net_dict[conf.train.network](conf.train).to(device=args.device, dtype=train_dataset.get_dtype())
    optimizer = torch.optim.Adam(network.parameters(), lr=conf.train.lr,
                                 weight_decay=conf.train.weight_decay)  # to use with ViTs
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=conf.train.factor, patience=conf.train.patience, min_lr=conf.train.min_lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=conf.train.T_0,  # 첫 번째 주기의 길이 (에폭 수)
                                            T_mult=conf.train.T_mult,  # 주기가 증가하는 비율 (예: 2배)
                                            eta_min=conf.train.min_lr)  # 최저 학습률
    best_loss = np.inf
    epoch = 0

    metricsdict = []
    orientations = []
    config = ConfigFactory.parse_file("configs/datasets/BaselineEuroc/Euroc_1000.conf")
    dataset_conf = config.inference

    # metric
    if args.eval:
        if os.path.isfile(os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")):
            print("Metrics evaluating...")
            for data_conf in dataset_conf.data_list:
                for data_name in data_conf.data_drive:
                    dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=200, step_size=200, drop_last=False, conf = dataset_conf)
                    loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False,
                                             drop_last=False)
                    checkpoint = torch.load(os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt"), map_location=args.device)
                    network.load_state_dict(checkpoint["model_state_dict"])
                    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    #scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    metricdict, oridict = eval(network, loader, conf.train, data_name)
                    metricsdict.append(metricdict)
                    orientations.append(oridict)
            print(f"Name : {data_name}, AOE (deg) : {metricdict['AOE (deg)']}, ROE (deg) : {metricdict['ROE (deg)']}\n All ROE = {metricdict['All ROE (deg)']}")
        else:
            print("Can't find the checkpoint")
    else:
        if not args.log:
            wandb.disabled = True
            print("wandb is disabled")
        else:
            wandb.init(project="CalibNet_" + exp_folder,
                       config=conf.train,
                       group=conf.train.network,
                       name=conf_name, )

        ## load the chkp if there exist
        if args.load_ckpt:
            if os.path.isfile(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")):
                checkpoint = torch.load(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"),
                                        map_location=args.device)
                network.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                print("loaded state dict %s best_loss %f" % (os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"),
                                                             best_loss))
            else:
                print("Can't find the checkpoint")
        for epoch_i in range(epoch, conf.train.max_epoches):
            train_loss = train(network, train_loader, conf.train, epoch_i, optimizer)
            test_loss = test(network, test_loader, conf.train)
            print("train loss: %f test loss: %f" % (train_loss["loss"], test_loss["loss"]))

            # save the training meta information
            if args.log:
                write_wandb("train", train_loss, epoch_i)
                write_wandb("test", test_loss, epoch_i)
                write_wandb("lr", scheduler.optimizer.param_groups[0]['lr'], epoch_i)

            '''if epoch_i % conf.train.eval_freq == conf.train.eval_freq - 1: # config file에서 정한 eval frequency 직전에 대해 eval 
                eval_state = evaluate(network=network, loader=eval_loader, confs=conf.train)
                
                if args.log:
                    #write_wandb('eval/pos_loss', eval_state['loss']['pos'].mean(), epoch_i)
                    write_wandb('eval/rot_loss', eval_state['loss']['rot'].mean(), epoch_i)
                    #write_wandb('eval/vel_loss', eval_state['loss']['vel'].mean(), epoch_i)
                    write_wandb('eval/rot_dist', eval_state['loss']['rot_dist'].mean(), epoch_i)
                    #write_wandb('eval/vel_dist', eval_state['loss']['vel_dist'].mean(), epoch_i)
                    #write_wandb('eval/pos_dist', eval_state['loss']['pos_dist'].mean(), epoch_i)

                print("eval rot: %f" % (eval_state['loss']['rot'].mean()))'''

            scheduler.step(test_loss['loss'])
            if test_loss['loss'] < best_loss:
                best_loss = test_loss['loss'];
                save_best = True
            else:
                save_best = False

            save_ckpt(network, optimizer, scheduler, epoch_i, best_loss, conf, save_best=save_best, )
    wandb.finish()