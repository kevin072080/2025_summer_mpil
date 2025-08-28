# CalibNet: gyroscope compensation + orientation integration (PyTorch)
# - 1D dilated conv blocks: kernel=[7,7,7,7], dilation=[1,4,16,64],
#   channels out=[16,32,128,32]
# - residual shortcut: output_of_block2 (+) output_of_block4
# - FC1: 256, FC2: 3  (outputs Δω_t)
# Input:  IMU window of length k, shape [B, k, 6] where 6=[gyro(3),acc(3)]
# Output: Δω_t (B,3); and calibrated ω̂_t via calibrate_gyro(...)
import torch
import torch.nn as nn
from typing import Tuple, List

# ---------- small SO(3) helpers ----------
def hat(v: torch.Tensor) -> torch.Tensor:
    """v: (...,3) -> skew-symmetric (...,3,3)"""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([ O, -z,  y], dim=-1),
        torch.stack([ z,  O, -x], dim=-1),
        torch.stack([-y,  x,  O], dim=-1),
    ], dim=-2)

def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues: omega (...,3) -> R (...,3,3)"""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
    K = hat(omega / theta)
    s = torch.sin(theta)[..., None]  # (...,1,1)
    c = torch.cos(theta)[..., None]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape)
    return I + s * K + (1 - c) * (K @ K)

# ---------- network blocks ----------
class DilatedConv1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dilation: int = 1):
        super().__init__()
        # 'same' padding for odd kernel
        pad = (kernel // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        return self.net(x)

class CalibNet(nn.Module):
    """
    Implements CalibNet-style IMU calibration network producing Δω_t.
    - Input:  [B, k, 6] (gyro:3 + acc:3)
    - Output: Δω_t: [B,3]
    Also exposes:
      - calibrate_gyro(window): returns ω̂_t and Δω_t
      - integrate_orientation(seq, dt, k): returns list of rotation matrices
    """
    def __init__(self,
                 in_ch: int = 6,
                 kernel_sizes=(7, 7, 7, 7),
                 dilations=(1, 4, 16, 64),
                 out_chs=(16, 32, 128, 32),
                 fc_hidden: int = 256):
        super().__init__()
        assert len(kernel_sizes) == len(dilations) == len(out_chs) == 4

        # four dilated conv blocks
        self.conv1 = DilatedConv1DBlock(in_ch,        out_chs[0], kernel_sizes[0], dilations[0])
        self.conv2 = DilatedConv1DBlock(out_chs[0],   out_chs[1], kernel_sizes[1], dilations[1])
        self.conv3 = DilatedConv1DBlock(out_chs[1],   out_chs[2], kernel_sizes[2], dilations[2])
        self.conv4 = DilatedConv1DBlock(out_chs[2],   out_chs[3], kernel_sizes[3], dilations[3])

        # FC head: GAP -> 256 -> 3
        self.fc1 = nn.Linear(out_chs[3], fc_hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_hidden, 3)  # Δω_t

        # trainable C_omega (3x3), initialized to identity (as in the paper's model)
        self.C_omega = nn.Parameter(torch.eye(3))

    def forward_features(self, imu_window: torch.Tensor) -> torch.Tensor:
        """
        imu_window: [B, k, 6]  -> features: [B, C]
        """
        x = imu_window.transpose(1, 2)  # [B, 6, k]
        x = self.conv1(x)
        x2 = self.conv2(x)
        x = self.conv3(x2)
        x = self.conv4(x)       # 4th
        x = x + x2              # shortcut: block2 -> block4
        x = x.mean(dim=-1)      # global average pooling over time -> [B, C(=32)]
        return x

    def forward(self, imu_window: torch.Tensor) -> torch.Tensor:
        """Return Δω_t for the window's last time index t. Shape: [B,3]."""
        feat = self.forward_features(imu_window)
        h = self.act(self.fc1(feat))
        delta_w = self.fc2(h)
        return delta_w

    @torch.no_grad()
    def calibrate_gyro(self, imu_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        imu_window: [B, k, 6] -> returns:
          ω_hat_t: [B,3]  (calibrated gyro at the last step t)
          Δω_t:    [B,3]
        """
        delta_w = self.forward(imu_window)          # Δω_t
        w_t = imu_window[:, -1, :3]                 # raw gyro at time t
        w_cal = (w_t @ self.C_omega.T) + delta_w    # ω̂_t = Cω ω_t + Δω_t
        return w_cal, delta_w

    @torch.no_grad()
    def integrate_orientation(self,
                              imu_seq: torch.Tensor,
                              dt: float = 0.005,
                              k: int = 448,
                              R0: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Streaming orientation from a long IMU sequence (single batch).
        imu_seq: [N, 6]  (gyro(3), acc(3))
        returns: list of rotation matrices [len=N], where the first (k-1) are copies of R0
                 until the window is full; after that, calibrated updates are applied.
        """
        device = imu_seq.device
        N = imu_seq.shape[0]
        if R0 is None:
            R = torch.eye(3, device=device)
        else:
            R = R0.clone()
        Rs: List[torch.Tensor] = [R.clone() for _ in range(N)]

        if N < k:
            return Rs  # not enough data for a full window

        # warm-start: keep R0 for first k-1 samples
        for t in range(k-1, N):
            window = imu_seq[t - (k - 1): t + 1].unsqueeze(0)  # [1, k, 6]
            w_cal, _ = self.calibrate_gyro(window)             # [1,3]
            dR = so3_exp(w_cal[0] * dt)                        # [3,3]
            R = R @ dR
            Rs[t] = R.clone()
        return Rs

# ------------------- example usage -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, k = 4, 448
    # fake IMU window: [B, k, 6] = [gyro(3), acc(3)]
    x = torch.randn(B, k, 6)
    model = CalibNet()
    delta_w = model(x)                  # Δω_t
    w_cal, _ = model.calibrate_gyro(x)  # ω̂_t
    print("Δω_t shape:", delta_w.shape) # -> [B,3]
    print("ω̂_t shape:", w_cal.shape)   # -> [B,3]

    # integrate on a long sequence (single sample)
    N = 2000
    seq = torch.randn(N, 6)
    Rs = model.integrate_orientation(seq, dt=0.005, k=448)
    print("num rotations:", len(Rs), "R[1000] det ~", torch.det(Rs[1000]).item())
