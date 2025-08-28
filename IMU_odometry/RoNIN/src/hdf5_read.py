import h5py
import numpy as np
# 파일 열기
with h5py.File('/home/mpil/PycharmProjects/RoNIN/Data/train_dataset_1/a000_1/data.hdf5', 'r+') as f:  # 'r+'는 읽기/쓰기 모드
    # 파일의 최상위 키를 출력
    print(list(f.keys()))  # 예: ['pose', 'raw', 'synced']

    # 'pose' 그룹 안의 데이터셋을 출력
    print(list(f['pose'].keys()))  # 예: ['tango_pos', 'tango_ori']
    #print(list(f['pose/ekf_ori']))
    print(list(f['raw'].keys()))
    print(list(f['synced'].keys()))
    #print(list(f['synced/grav']))
    print(f['synced/grav'][-3:-1, ])
    print(f['synced/linacce'][-3:-1, ])
    #, f['synced/acce'][-3:-1, ] - f['synced/grav'][-3:-1, ]
    print(f['synced/acce'][-3:-1, ])
    print(f['synced/acce'].shape)


    q1 = np.array([0.240031, -0.673588, 0.202263, 0.669145])
    q2 = np.array([-0.673316, -0.239134, -0.669466, 0.203165])

    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    dot = np.abs(np.dot(q1, q2))
    theta_deg = 2 * np.arccos(dot) * 180 / np.pi

    print(f"Angle difference ≈ {theta_deg:.2f} degrees")
