import numpy as np
import torch
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class BaseLoss(torch.nn.Module):

    def __init__(self, min_N, max_N, dt):
        super().__init__()
        # windows sizes
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N
        # sampling time
        self.dt = dt # (s)


class CalibLoss(BaseLoss):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, target, huber):
        super().__init__(min_N, max_N, dt)
        # weights on loss
        self.w = w
        self.sl = torch.nn.SmoothL1Loss(reduction= 'sum')
        if target == 'rotation matrix':
            self.forward = self.forward_with_rotation_matrices
        elif target == 'quaternion':
            self.forward = self.forward_with_quaternions
        elif target == 'rotation matrix mask':
            self.forward = self.forward_with_rotation_matrices_mask
        elif target == 'quaternion mask':
            self.forward = self.forward_with_quaternion_mask
        self.huber = huber
        self.weight16 = torch.ones(1, 1, 16).cuda()/16
        self.weight32 = torch.ones(1, 1, 32).cuda() / 32
        self.weight64 = torch.ones(1, 1, 64).cuda() / 64
        self.N0 = 5 # remove first N0 increment in loss due not account padding
        self.dt = 0.005

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double()*self.dt))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double()*self.dt))
        Rot0 = SO3.qnorm(self.gt['qs'][:2].cuda().double())
        imu_qs[0] = Rot0[0]
        net_qs[0] = Rot0[0]

        N = np.log2(imu_qs.shape[0])
        for i in range(int(N)):
            k = 2**i
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

        if int(N) < N:
            k = 2**int(N)
            k2 = imu_qs[k:].shape[0]
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float() #wxyz
        return net_qs.cpu(), imu_Rots, net_Rots

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w*self.sl(rs/self.huber, torch.zeros_like(rs))*(self.huber**2) # huber loss * 가중치 w
        return loss

    def forward_with_rotation_matrices(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]

        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]

        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss

    def forward_with_quaternions(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        Xs = SO3.qexp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss
    #     def integrate_with_quaternions_superfast(self, N, raw_us, net_us):
    def forward_with_rotation_matrices_mask(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        # print(xs.shape, hat_xs.shape)
        '''masks = xs[:, :, 3].unsqueeze(1)
        m16 = torch.nn.functional.conv1d(masks, self.weight16, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2) # self.weight : kernal -> 각 값에 1/freq 곱해서 더하여 평균 구하기
        m16[m16 < 1] = 0
        m32 = torch.nn.functional.conv1d(masks, self.weight32, bias=None,
                                         stride=32).double().transpose(1,2)  # self.weight : kernal -> 각 값에 1/freq 곱해서 더하여 평균 구하기
        m32[m32 < 1] = 0'''

        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()  # B*T, 3 reshape & dt 적분하여 각도 차이로 변환
        Omegas = SO3.exp(hat_xs[:, :3])  # rotvec -> rotmat / B*T, 3 -> B*T, 3, 3

        Xs16 = SO3.exp(xs[:, ::16, :3].reshape(-1, 3).double()) # self.min_train_freq == 16 -> 1/16 downsampling
        for k in range(4): # 1/16으로 downsampling
            Omegas = Omegas[::2].bmm(Omegas[1::2]) # 홀수 omega * 인접 짝수 omega -> dtheta1 + dtheta2 = dtheta (절반 down)
        Omegas16 = Omegas # 일단은 이렇게 하고, 문제 생기면 reshape 전에 omegas16, 32, 64 정의 !

        if Xs16.shape[0] % 2 == 0:
            Xs32 = Xs16[::2].bmm(Xs16[1::2])
            Omegas32 = Omegas[::2].bmm(Omegas[1::2])  # 1/32 downsampling
        else:
            print('16 downsampling에서 마지막 원소 반영 X')
            Xs32 = Xs16[:-1:2].bmm(Xs16[1::2])
            Omegas32 = Omegas[:-1:2].bmm(Omegas[1::2])  # 1/32 downsampling

        # print(xs.shape, Xs16.shape, Xs32.shape, Xs64.shape,Omegas.shape)
        # print(hat_xs.shape)
        rl16 = SO3.log(bmtm(Omegas16, Xs16)).reshape(N,-1,3)
        #rl16m = rl16[m16.squeeze(-1)][:, self.N0:]
        rloss16 = torch.log2(torch.cosh(torch.norm(rl16, dim=2)))

        rl32 = SO3.log(bmtm(Omegas32, Xs32)).reshape(N, -1, 3)
        #rl32m = rl32[m32][:, self.N0:]
        rloss32 = torch.log2(torch.cosh(torch.norm(rl32, dim=2)))
        rl = torch.sum(rloss16) + torch.sum(rloss32)
        #print('rl',rl, rl16.shape, rloss32.shape, rloss16.shape)

        return rl

    def forward_with_quaternion_mask(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.qexp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        rs = rs[masks[:, self.N0:].squeeze(2) == 1]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss


