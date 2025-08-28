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


class GyroLoss(BaseLoss):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, target, huber):
        super().__init__(min_N, max_N, dt)
        # weights on loss
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        if target == 'rotation matrix':
            self.forward = self.forward_with_rotation_matrices
        elif target == 'quaternion':
            self.forward = self.forward_with_quaternions
        elif target == 'rotation matrix mask':
            self.forward = self.forward_with_rotation_matrices_mask
        elif target == 'quaternion mask':
            self.forward = self.forward_with_quaternion_mask
        self.huber = huber
        self.weight = torch.ones(1, 1,
            self.min_train_freq).cuda()/self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding
        self.weight16 = torch.ones(1, 1, 16).cuda() / 16
        self.weight32 = torch.ones(1, 1, 32).cuda() / 32
        self.weight64 = torch.ones(1, 1, 64).cuda() / 64

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

    def forward_with_rotation_matrices_mask(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]

        masks = xs[:, :, 3].unsqueeze(1)
        m16 = torch.nn.functional.conv1d(masks, self.weight16, stride=self.min_train_freq)  # [B,1,T16]
        m32 = torch.nn.functional.conv1d(masks, self.weight32, stride=32)  # [B,1,T32]
        m64 = torch.nn.functional.conv1d(masks, self.weight64, stride=64)  # [B,1,T64]

        # 평균이 1 이상인 위치만 유효 (bool로 변환)
        m16 = (m16 >= 1).transpose(1, 2)  # [B,T16,1] → 후에 squeeze
        m32 = (m32 >= 1).transpose(1, 2)
        m64 = (m64 >= 1).transpose(1, 2)

        Xs16 = SO3.exp(xs[:, ::16, :3].reshape(-1, 3).double()) # self.min_train_freq == 16 -> 1/16 downsampling
        Xs32 = SO3.exp(xs[:, ::32, :3].reshape(-1, 3).double())
        Xs64 = SO3.exp(xs[:, ::64, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.double() # B, T, 3 reshape & dt 적분하여 각도 차이로 변환
        #print(xs.shape, Xs16.shape, Xs32.shape, Xs64.shape,  hat_xs.shape)

        Omegas = SO3.exp(hat_xs[:, :, :3]) # rotvec -> rotmat B, T, 3, 3
        # Omegas_all = Omegas.reshape(-1, 3) # B*T, 3 -> B*T, 3, 3
        for k in range(4): # 1/16으로 downsampling (decimation)
            Omegas = Omegas[:, ::2]@(Omegas[:, 1::2]) # 홀수 omega * 인접 짝수 omega -> dtheta1 + dtheta2 = dtheta (절반 down) B, T/16, 3, 3
        Omegas16 = Omegas.reshape(-1, 3, 3) # B*T/16, 3, 3

        if Omegas.shape[1] % 2 == 0:
            Omegas = Omegas[:, ::2]@(Omegas[:, 1::2])  # 1/32 downsampling
            Omegas32 = Omegas.reshape(-1,3,3)
        else:
            Omegas = Omegas[:, :-1:2]@(Omegas[:, 1::2])  # 1/32 downsampling
            Omegas32 = Omegas.reshape(-1, 3, 3)
            m64 = m64[:, 1:, :]
            m32 = m32[:, 1:, :]

        if Omegas.shape[1] % 2 == 0:
            Omegas64 = (Omegas[:, ::2] @ Omegas[:, 1::2]).reshape(-1, 3, 3) # B*T/64, 3, 3
            L_gt = Xs64[0] @ Xs64[1:]
        else:
            Omegas64 = (Omegas[:, :-1:2] @ Omegas[:, 1::2]).reshape(-1, 3, 3) # B*T/64, 3, 3, 처음 거 날림
            m64 = m64[:,1:,:]
            L_gt = Xs64[0] @ Xs64[1:-1]
        #print(Omegas_all.shape, Omegas16.shape, Omegas32.shape, Omegas64.shape)

        # compute increment at min_train_freq by decimation

        rl16 = SO3.log(bmtm(Omegas16, Xs16)).reshape(N, -1, 3)[:, self.N0:]
        rl16m = rl16[m16[:, self.N0:].squeeze(-1)]
        rloss16 = self.f_huber(rl16m)

        # Omegas32 = Omegas[::2].bmm(Omegas[1::2]) # 1/32 downsampling
        rl32 = SO3.log(bmtm(Omegas32, Xs32)).reshape(N, -1, 3)[:, self.N0:]
        rl32m = rl32[m32[:, self.N0:].squeeze(-1)]
        rloss32 = self.f_huber(rl32m)

        rl = rloss16 + 0.5 * rloss32
        # print('rloss', rloss32.shape, rl.shape)

        L = Omegas64[0].T @ Omegas64[1:]
        # L_gt = Xs64[0] @ Xs64[1:]
        al64 = SO3.log(bmtm(L_gt, L))[:-(L.shape[0] % 3)].reshape(N, -1, 3)[:, self.N0:]
        al64m = al64[m64[:,1+self.N0:,:].squeeze(-1)]
        # print('al', al64.shape)
        al = self.f_huber(al64m)

        loss = rl + 0.2 * al
        #print(m16.shape, rl16.shape, rl16m.shape)
        #print(rloss32, rloss32)
        return loss

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
