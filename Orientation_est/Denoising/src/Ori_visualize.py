import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os


class ViewerSimple:
    def __init__(self, gt_data, est_data):
        self.vis = None
        self.first_update = True
        self.stop_update = False
        self.current_frame = 0
        self.length = 0

        self.gt_cf = None
        self.est_cf = None
        self.world_cf = None

        self.gt_cf_center = [-1., 0., 0.]
        self.est_cf_center = [1., 0., 0.]

        self.gt_orientation = None
        self.est_orientation = None

        self.gt_orientation_cache = np.eye(3, dtype=float)
        self.est_orientation_cache = np.eye(3, dtype=float)

        self.load_data(gt_data, est_data)
        #self.set_mesh()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis.close()
        self.vis = None
        self.first_update = True

    def connect(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Test", width=1280, height=720, visible=True)
        self.vis.register_key_callback(ord('Q'), self.callback_quit)
        self.vis.register_key_callback(ord(','), self.callback_prev_frame)
        self.vis.register_key_callback(ord('.'), self.callback_next_frame)

    def disconnect(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis.close()
        self.vis = None
        self.first_update = True

    def set_mesh(self):
        self.world_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
        self.gt_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=self.gt_cf_center)
        self.est_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=self.est_cf_center)

        self.world_cf.compute_vertex_normals()
        self.gt_cf.compute_vertex_normals()
        self.est_cf.compute_vertex_normals()

        self.vis.add_geometry(self.world_cf)
        self.vis.add_geometry(self.gt_cf)
        self.vis.add_geometry(self.est_cf)

    def load_data(self, gt_data, est_data):
        #gt_data = np.loadtxt(gt_directory)
        #est_data = np.loadtxt(est_directory)

        self.length = len(gt_data)

        # gt_quat = gt_data[:, 1:]  # xyzw
        # est_quat = est_data[:, 1:]

        '''gt_quat = gt_data[:,[1,2,3,0]]  # wxyz
        est_quat = est_data[:,[1,2,3,0]]

        self.gt_orientation = R.from_quat(gt_quat).as_matrix()
        self.est_orientation = R.from_quat(est_quat).as_matrix()'''
        self.gt_orientation = gt_data.cpu().numpy()
        self.est_orientation = est_data.cpu().numpy()

    def update(self, i):
        self.gt_cf.rotate(self.gt_orientation[i] @ self.gt_orientation_cache.T, center=self.gt_cf_center)
        self.est_cf.rotate(self.est_orientation[i] @ self.est_orientation_cache.T, center=self.est_cf_center)

        self.gt_orientation_cache = self.gt_orientation[i]
        self.est_orientation_cache = self.est_orientation[i]


        self.vis.update_geometry(self.gt_cf)
        self.vis.update_geometry(self.est_cf)
        #print('debug; before poll')
        self.vis.poll_events()
        #print('debug; before update renderer')
        self.vis.update_renderer()
        #print('debug; after update')
        self.first_update = False

    def pause(self):
        self.stop_update = False
        while not self.stop_update:
            self.vis.poll_events()
            self.vis.update_renderer()

    def callback_quit(self, vis):
        self.stop_update = True
        return False

    def callback_prev_frame(self, vis):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update(self.current_frame)
        return False

    def callback_next_frame(self, vis):
        if self.current_frame < self.length - 10:
            self.current_frame += 10
            self.update(self.current_frame)
        return False


if __name__ == '__main__':
    gt_directory = ""
    est_directory = ""

    with ViewerSimple(gt_directory=gt_directory, est_directory=est_directory) as viewer:
        viewer.update(0)
        viewer.pause()