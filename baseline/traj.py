import numpy as np
import matplotlib.pyplot as plt
import os



def plot_traj3d(ts, posx, posy, posz, title="Trajectory 3D"):
    posx = np.asarray(posx); posy = np.asarray(posy); posz = np.asarray(posz)
    assert posx.shape == posy.shape == posz.shape

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # 궤적 라인
    ax.plot(posx, posy, posz, linewidth=1.5)

    # 시작/끝 표시
    ax.scatter(posx[0],  posy[0],  posz[0],  s=40, marker='o', label='start')
    ax.scatter(posx[-1], posy[-1], posz[-1], s=60, marker='^', label='end')

    # 축/제목
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z'); ax.set_title(title)
    ax.legend(loc='best')

    # 등축 설정
    def set_equal_aspect(ax, X, Y, Z):
        x_range = X.max()-X.min(); y_range = Y.max()-Y.min(); z_range = Z.max()-Z.min()
        max_range = max(x_range, y_range, z_range)
        x_mid = (X.max()+X.min())/2; y_mid = (Y.max()+Y.min())/2; z_mid = (Z.max()+Z.min())/2
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    set_equal_aspect(ax, posx, posy, posz)

    plt.tight_layout()
    plt.show()

# 사용 예
# plot_traj3d(ts, posx, posy, posz)

def plot_traj3d_compare(ts, est_posx, est_posy, est_posz,
                             gt_posx, gt_posy, gt_posz):
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # est trajectory
    ax.plot(est_posx, est_posy, est_posz, label="est", color='blue')
    ax.scatter(est_posx[0], est_posy[0], est_posz[0], marker='o', c='blue')
    ax.scatter(est_posx[-1], est_posy[-1], est_posz[-1], marker='^', c='blue')

    # gt trajectory
    ax.plot(gt_posx, gt_posy, gt_posz, label="gt", color='red')
    ax.scatter(gt_posx[0], gt_posy[0], gt_posz[0], marker='o', c='red')
    ax.scatter(gt_posx[-1], gt_posy[-1], gt_posz[-1], marker='^', c='red')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

# 사용 예시


if __name__ == '__main__':
    input_acc_type = 'linacce'
    dataset = 'Euroc'
    corrected = 'AirIMU'
    data_name = 'MH_01_easy'
    #data_name = 'ff_return_journey_right'
    #data_name = 'ff_JEM2USL_dark'

    path = os.path.join('/home/mpil/miniconda3/envs/baseline/result_Preintegration_' + input_acc_type, dataset, corrected,
                                       input_acc_type + '_' + dataset + '_' + data_name + '_preinte.npy')

    if dataset == 'Blackbird':
        path = os.path.join('/home/mpil/miniconda3/envs/baseline/result_Preintegration_linacce/BlackBird/AirIMU/linacce_BlackBird_maxSpeed8p0_preinte.npy')
        #path = os.path.join(
        #    '/home/mpil/miniconda3/envs/baseline/result_Preintegration_acc/BlackBird/Raw/acc_BlackBird_maxSpeed8p0_preinte.npy')
    data = np.load(path)
    print(path)
    ts = data['ts']
    est_posx = data['est_posx']
    est_posy = data['est_posy']
    est_posz = data['est_posz']
    gt_posx = data['gt_posx']
    gt_posy = data['gt_posy']
    gt_posz = data['gt_posz']

    plot_traj3d_compare(ts, est_posx, est_posy, est_posz,
                        gt_posx, gt_posy, gt_posz)