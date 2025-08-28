import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import bmtm, bmtv, bmmt, bbmv


class IFESBlock(nn.Module):
    """
    두 conv 분기 (k=7,9, 동일 dilation) -> GAP -> FC -> softmax(분기 가중치 2개)
    -> 분기 feature에 가중치 곱 후 합
    """
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        # 두 분기
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation,
                               padding=((7-1)//2)*dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=9, dilation=dilation,
                               padding=((9-1)//2)*dilation) #
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()

        # Global Average Pool
        self.gap = nn.AdaptiveAvgPool1d(1)

        # FC → softmax 로 분기 가중치 2개 생성 (스칼라 가중치)
        # 입력은 각 분기의 GAP 후 채널 평균하여 2차원 벡터 [w1_in, w2_in]
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, dim)

    def forward(self, x):                       # x: [B, C_in, T]
        y1 = self.act(self.bn1(self.conv1(x)))  # [B, dim, T]
        y2 = self.act(self.bn2(self.conv2(x)))  # [B, dim, T]
        y = torch.add(y1, y2) # [B, dim, T]

        s = self.gap(y).squeeze(-1) # [B, dim, 1] -> [B, dim]

        w = self.fc2(F.gelu(self.fc1(s)))
        ws = F.softmax(w, dim=1) # [B, dim]

        # 브로드캐스트용 reshape
        w1 = ws.unsqueeze(-1)                              # [B,dim,1]

        out = torch.add(w1*y1, w1*y2)                      # [B, dim, T]
        return out

class gyronet(nn.Module):
    def __init__(self, in_dim, out_dim, dims, kernalC, dropout, momentum, diaC, diaS, diaT, gyro_std):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        #'dims' : [16, 32, 64, 128],
        self.dimC_in = [6] + dims
        self.dimC_out = dims +[3]
        self.dimS = [6] + dims
        self.dimT_in = [6] + dims[:-1]
        self.dimT_out = dims

        self.ifc = nn.ModuleList([
            self.IFC(indim=self.dimC_in, outdim=self.dimC_out, idx=i, k=kernalC, d=diaC, dropout=dropout, momentum=momentum)
            for i in range(5)
        ])
        self.ifes = nn.ModuleList([
            self.IFES(dims=self.dimS, idx=i, dia=diaS)
            for i in range(5)
        ])
        self.tran = nn.ModuleList([
            self.Transition(indim=self.dimT_in, outdim=self.dimT_out, idx=i, dia=diaT)
            for i in range(4)
        ])

        self.mean_u = torch.nn.Parameter(torch.zeros(in_dim),
                                         requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.ones(in_dim),
                                        requires_grad=False)

        gyro_std = torch.Tensor(gyro_std)
        self.gyro_std = torch.nn.Parameter(gyro_std, requires_grad=False)

        gyro_Rot = 0.05 * torch.randn(3, 3).cuda()
        self.gyro_Rot = torch.nn.Parameter(gyro_Rot)
        self.Id3 = torch.eye(3).cuda()

    def IFC(self, indim, outdim, idx, k, d, dropout, momentum):
        layers = []

        if idx == 0:
            k0, k1, k2, k3 = k[:4]
            d0, d1, d2 = d[:3]
            p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) + d0 * d1 * d2 * (k3 - 1)
            #layers.append(nn.ReplicationPad1d((p0, 0)))  # padding at start
            '''if p0 > 0:
                layers.append(nn.ReplicationPad1d(((k0-1) * d0 // 2, 0))) # padding at start'''

        if idx != 4:
            layers.extend([
                nn.Conv1d(in_channels=indim[idx], out_channels= outdim[idx], kernel_size= k[idx], dilation=d[idx], padding=d[idx] * (k[idx]-1) // 2),
                nn.BatchNorm1d(num_features=outdim[idx], momentum=momentum),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        else:
            layers.extend([
                nn.Conv1d(in_channels=indim[idx], out_channels= outdim[idx], kernel_size= k[idx], dilation=d[idx], padding=d[idx] * (k[idx]-1) // 2),
                nn.ReplicationPad1d((0, 0)) # no padding at end, to fit an interface
            ])

        return nn.Sequential(*layers)

    def IFES(self, dims, idx, dia):
        dim = dims[idx] if isinstance(dims, (list, tuple)) else dia
        dil = dia[idx] if isinstance(dia, (list, tuple)) else dia
        return IFESBlock(dim, dilation=dil)

    def Transition(self, indim, outdim, idx, dia):
        layers = []
        if idx == 0:
            layers.extend([
                nn.Conv1d(in_channels=indim[idx], out_channels=outdim[idx], kernel_size=1, dilation= dia[idx], padding=dia[idx] * (1-1) // 2),
                nn.GELU()
            ])
        else:
            if idx != 3:
                layers.extend([
                    nn.Conv1d(in_channels=indim[idx], out_channels=outdim[idx], kernel_size=7, dilation= dia[idx], padding=(7-1)//2*dia[idx]),
                    nn.GELU()
                ])
            else:
                layers.extend([
                    nn.Conv1d(in_channels=indim[idx], out_channels=outdim[idx], kernel_size=7, dilation=dia[idx], padding=(7-1)//2*dia[idx]),
                    nn.ReLU()
                ])
        return nn.Sequential(*layers)


    def norm(self, us): # 편차를 표준편차로 나누어 정규화
        return (us-self.mean_u)/self.std_u

    def set_normalized_factors(self, mean_u, std_u):
        # 아마 mean_u가 채널별로 나와야 하는 것 같은데 일단 tensor화만 함
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)
        #self.mean_u = mean_u.detach().clone()
        #self.std_u = std_u.detach().clone()
        #self.std_u = torch.nn.Parameter(torch.tensor(std_u, dtype=torch.float32).cuda(), requires_grad=False)

    def forward(self, _input):
        imu = self.norm(_input).transpose(1,2)
        #print('input', imu.shape)
        outS0 = self.ifes[0](imu)
        #print('outS0',outS0.shape)
        outC0 = self.ifc[0](outS0)
        #print('outC0', outC0.shape)

        tran0 = self.tran[0](imu)
        #print('tran', tran0.shape)
        outS1 = self.ifes[1](torch.add(tran0, outC0))
        outC1 = self.ifc[1](torch.add(outS1, outC0))

        tran1 = self.tran[1](outC0)
        #print('tran', tran1.shape)
        outS2 = self.ifes[2](torch.add(tran1, outC1))
        outC2 = self.ifc[2](torch.add(outS2,outC1))

        tran2 = self.tran[2](outC1)
        #print('tran', tran2.shape)
        outS3 = self.ifes[3](torch.add(tran2, outC2))
        outC3 = self.ifc[3](torch.add(outS3, outC2))

        tran3 = self.tran[3](outC2)
        #print('tran', tran3.shape)
        outS4 = self.ifes[4](torch.add(tran3, outC3))
        outC4 = self.ifc[4](outS4)
        #print('output', outC4.shape)

        Rots = (self.Id3 + self.gyro_Rot).expand(_input.shape[0], _input.shape[1], 3, 3)  # 기하학적 보정 by rotation mat
        #print((self.gyro_std * outC4.transpose(1,2)).shape, (outC4.transpose(1,2)).shape)
        #Rots = (self.Id3 + self.gyro_Rot).expand(_input.shape[0], 3, 3)  # 기하학적 보정 by rotation mat
        Rot_us = bbmv(Rots, _input[:, :, :3])
        return self.gyro_std * outC4.transpose(1,2) + Rot_us