import os,warnings
import torch
import numpy as np
import pypose as pp
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R, Slerp
from datasets import Sequence

def qinterp(qs: torch.Tensor, t, t_int):
    """
    qs: (N, 4) tensor [w, x, y, z]
    t:  (N,) time vector
    t_int: interpolated time points
    """
    qs_np = qs.clone().numpy()  # (N, 4)

    # ensure continuity in quaternion sign
    for i in range(1, len(qs_np)):
        if np.dot(qs_np[i - 1], qs_np[i]) < 0:
            qs_np[i] *= -1  # flip sign for continuity

    rot = R.from_quat(qs_np[:, [1, 2, 3, 0]])  # to [x, y, z, w]
    slerp = Slerp(t, rot)
    interp_rot = slerp(t_int).as_quat()  # [x, y, z, w]
    interp_quat = interp_rot[:, [3, 0, 1, 2]]  # to [w, x, y, z]

    return torch.tensor(interp_quat, dtype=torch.float64)

def lookAt(dir_vec, up=torch.tensor([0., 0., 1.], dtype=torch.float64),
           source=torch.tensor([0., 0., 0.], dtype=torch.float64)):
    '''
    dir_vec: the tensor shall be (1)
    return the rotation matrix of the
    '''
    if not isinstance(dir_vec, torch.Tensor):
        dir_vec = torch.tensor(dir_vec)

    def normalize(x):
        length = x.norm()
        if length < 0.005:
            length = 1
            warnings.warn('Normlization error that the norm is too small')
        return x / length

    zaxis = normalize(dir_vec - source)
    xaxis = normalize(torch.cross(zaxis, up))
    yaxis = torch.cross(xaxis, zaxis)

    m = torch.tensor([
        [xaxis[0], xaxis[1], xaxis[2]],
        [yaxis[0], yaxis[1], yaxis[2]],
        [zaxis[0], zaxis[1], zaxis[2]],
    ])

    return m

class Astrobee(Sequence):
    """
    Output:
    acce: the accelaration in **world frame**
    """
    def __init__(self, data_root, data_name, intepolate = True, calib = False, load_vicon = False, glob_coord=False, **kwargs):
        super(Astrobee, self).__init__()
        (   
            self.data_root, self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)

        self.gravity = torch.tensor([0., 0., 9.81007], dtype=torch.float64)

        self.Rimu2body = np.array([[0, -1, 0],
                                   [1, 0, 0],
                                   [0, 0, 1]], dtype=np.float64)
        data_path = os.path.join(data_root, data_name)
        self.load_imu(data_path)
        self.load_gt(data_path)
        
        # EUROC require an interpolation
        if intepolate:
            t_start = np.max([self.data['gt_time'][0], self.data['time'][0]])
            t_end = np.min([self.data['gt_time'][-1], self.data['time'][-1]])

            # find the index of the start and end
            idx_start_imu = np.searchsorted(self.data['time'], t_start)
            idx_start_gt = np.searchsorted(self.data['gt_time'], t_start)

            idx_end_imu = np.searchsorted(self.data['time'], t_end, 'right')
            idx_end_gt = np.searchsorted(self.data['gt_time'], t_end, 'right')

            for k in ['gt_time', 'pos', 'quat']: # dataset 불러오는 단계에서 이미 interpolated, idx 정렬까지
                self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

            for k in ['time', 'acc', 'gyro']:
                self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

            ## start interpotation
            self.data["gt_orientation"] = self.interp_rot(self.data['time'], self.data['gt_time'], self.data['quat'])
            self.data["gt_translation"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['pos'])
        
        else:
            self.data["gt_orientation"] = pp.SO3(torch.tensor(self.data['pose'][:,3:]))
            self.data['gt_translation'] = torch.tensor(self.data['pose'][:,:3])
        
        # move the time to torch
        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data['dt'] = (self.data["time"][1:] - self.data["time"][:-1])[:,None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

        #calibration already done
        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])

        # change the acc and gyro scope into the global coordinate.  
        if glob_coord:
            self.data['gyro'] = self.data["gt_orientation"] * self.data['gyro']
            self.data['acc'] = self.data["gt_orientation"] * self.data['acc']

        print("loaded: ", data_path, "interpolate: ", intepolate)

    def get_length(self):
        return self.data['time'].shape[0]

    def load_imu(self, folder): # folder :.../seq
        path_imu = os.path.join(folder, "imu_filtered.txt")
        imu_data = np.genfromtxt(path_imu, delimiter=" ", skip_header=0)
        self.data["time"] = imu_data[:,0] / 1e9
        self.data["gyro"] = imu_data[:,1:4] @ self.Rimu2body.T # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = imu_data[:,4:7] @ self.Rimu2body.T   # acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

    def load_gt(self, folder):
        path_imu = os.path.join(folder, "vio.txt")
        gt_data = np.genfromtxt(path_imu, delimiter=" ", skip_header=0)
        self.data["gt_time"] = gt_data[:,0] / 1e9
        self.data["pos"] = gt_data[:,1:4]
        self.data['quat'] = gt_data[:,[7,4,5,6]] # x, y, z, w -> wxyz
        #self.data["b_acc"] = gt_data[:,-3:]
        #self.data["b_gyro"] = gt_data[:,-6:-3]
        #self.data["velocity"] = gt_data[:,-9:-6]

    def interp_rot(self, time, opt_time, quat):
        # interpolation in the log space
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])

        quat = torch.tensor(quat)
        quat = qinterp(quat, gt_dt, imu_dt).double() #imu_dt를 gt_dt에 맞춰 보간
        self.data['rot_wxyz'] = quat
        rot = torch.zeros_like(quat)
        rot[:,3] = quat[:,0] # quat w,x,y,z -> rot x,y,z,w
        rot[:,:3] = quat[:,1:]

        return pp.SO3(rot)

    def interp_xyz(self, time, opt_time, xyz):
        
        intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
        intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
        intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])

        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

