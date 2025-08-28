import torch
import torch.nn as nn


class DebugLayer(nn.Module):
    def forward(self, x):
        print(f"[DEBUG] output shape: {x.shape}")
        return x

class ConvNet(nn.Module): #nn.module이어야 torch.save -> state_dict로 param 저장 가능
    def __init__(self, num_classes=3):
        super(ConvNet, self).__init__()
        self.C = torch.nn.Parameter(torch.eye(3))

        self.conv_layers = nn.Sequential(
            # Dilation conv layer 1, 2
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=7, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #DebugLayer(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=1, dilation=4), # 12 -> 1, 448 -> 426
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #DebugLayer(),
        )
        self.residual = nn.Sequential(
            # Residual Block
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=7, padding=120, dilation=16), # 48 -> 120
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #DebugLayer(),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, padding=120, dilation=64), #192 -> 120
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # 일단 data 수 안 줄어들도록 padding 설정
        self.fc_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            #DebugLayer(),
            nn.Linear(32*422, 256),  # 448 -> 82
            nn.ReLU(),
            nn.Linear(256, 3)
        )


    def forward(self, x):
        gyro = x[:, :3, :]
        #print(gyro.shape, self.C.view(1, 3, 3).shape)
        imu = self.conv_layers(x)
        gyro = torch.matmul(self.C.view(1, 3, 3), gyro)  # (1, 3, 3) * (128, 3, 448)
        #print(gyro.shape, 'gyro')
        shortcut = imu + self.residual(imu)

        gyro_correction = self.fc_layers(shortcut.view(imu.size(0), -1))
        #print(imu.shape, "imu")
        '''if not hasattr(self, 'fc_initialized'):
            in_features = x.flatten(start_dim=1).size(1)
            self.fc_layers[1] = nn.Linear(in_features, 256)
            self.fc_initialized = True'''

        output = gyro_correction.unsqueeze(-1) + gyro #gyro_correction이 윈도우 내의 모든 t에 동일하게 작용한다고 생각, test, train step, window size 고려 필요
        # output -> calibrated w for 448 t, 128 batch
        return output

