import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import bmtm, bmtv, bmmt, bbmv

class calibnet(nn.Module):
    def __init__(self, in_dim, out_dim, ker, dia, dims, dropout, momentum):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        #'dims' : [16, 32, 128, 32],
        self.dimC_in = [6] + dims[:-1]
        self.dimC_out = dims

        self.dconv = nn.ModuleList([
            self.Dconv(indim=self.dimC_in, outdim=self.dimC_out, idx=i, k=ker, d=dia, dropout=dropout, momentum=momentum)
            for i in range(4)
        ])
        self.fc = nn.Sequential(
            #nn.Flatten(start_dim=1),
            nn.Linear(32, 256),
            nn.Linear(256, 3)
        )

        self.mean_u = torch.nn.Parameter(torch.zeros(in_dim),
                                         requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.ones(in_dim),
                                        requires_grad=False)

        gyro_Rot = 0.05 * torch.randn(3, 3).cuda()
        self.gyro_Rot = torch.nn.Parameter(gyro_Rot)
        self.Id3 = torch.eye(3).cuda()

    def Dconv(self, indim, outdim, idx, k, d, dropout, momentum):
        layers = []

        if idx == 0:
            k0, k1, k2, k3 = k[:4]
            d0, d1, d2 = d[:3]
            p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) + d0 * d1 * d2 * (k3 - 1)
            #layers.append(nn.ReplicationPad1d((p0, 0)))  # padding at start
            '''if p0 > 0:
                layers.append(nn.ReplicationPad1d(((k0-1) * d0 // 2, 0))) # padding at start'''

        if idx != 3:
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

    def forward(self, _input):
        imu = _input.transpose(1,2)
        #print('input', imu.shape)
        out0 = self.dconv[0](imu)
        #print(out0.shape)
        out1 = self.dconv[1](out0)
        out2 = self.dconv[2](out1)
        out3 = self.dconv[3](out2)
        #print(out3.shape)
        calibrated_w = self.fc(torch.add(out1, out3).permute(0,2,1))
        Cali_w = calibrated_w.permute(0,2,1)

        Cw = (self.Id3 + self.gyro_Rot).expand(_input.shape[0], _input.shape[1], 3, 3)  # 기하학적 보정 by rotation mat
        # print((self.gyro_std * outC4.transpose(1,2)).shape, (outC4.transpose(1,2)).shape)
        # Rots = (self.Id3 + self.gyro_Rot).expand(_input.shape[0], 3, 3)  # 기하학적 보정 by rotation mat
        Cww = bbmv(Cw, _input[:, :, :3])
        return Cali_w.transpose(1, 2) + Cww