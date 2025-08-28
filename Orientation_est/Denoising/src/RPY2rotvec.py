import torch
from scipy.spatial.transform import Rotation as R

# 예시 RPY값 (rad)
roll, pitch, yaw = 0.67, 0.64, 0.46

# 1) RPY -> Rotation 객체 (ZYX 순서)
r = R.from_euler('xyz', [roll, pitch, yaw])

# 2) Rotation -> rotvec
rotvec = r.as_rotvec()  # shape (3,)
rotvec_norm = torch.norm(torch.tensor(rotvec))

print("rotvec:", rotvec)
print("rotvec norm:", rotvec_norm)