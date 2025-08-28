import numpy as np
import matplotlib.pyplot as plt

# 시스템 설정 (1D 위치 + 속도)
dt = 1.0
A = np.array([[1, dt], [0, 1]])       # 상태 전이 행렬
H = np.array([[1, 0]])                # 위치만 측정

Q = np.array([[0.1, 0], [0, 0.1]])    # 프로세스 잡음 공분산
R = np.array([[1.0]])                # 측정 잡음 공분산

x = np.array([[0], [1]])             # 초기 상태 (위치 0, 속도 1)
P = np.eye(2) * 1.0                  # 초기 오차 공분산

measurements = [1.2, 2.1, 2.9, 4.05, 5.0]

pred_trace = []
upd_trace = []

for z in measurements:
    # === 예측 단계 ===
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q
    pred_trace.append(np.trace(P_pred))

    # === 갱신 단계 ===
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    z = np.array([[z]])
    x = x_pred + K @ (z - H @ x_pred)
    P = (np.eye(2) - K @ H) @ P_pred
    upd_trace.append(np.trace(P))

# === 시각화 ===
plt.figure(figsize=(8, 5))
plt.plot(pred_trace, label="Prediction Tr($P_{k|k-1}$)", marker='o')
plt.plot(upd_trace, label="Update Tr($P_{k|k}$)", marker='s')
plt.xlabel("Time Step")
plt.ylabel("Trace of Error Covariance")
plt.title("Trace of $P$ over Kalman Filter Iterations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
