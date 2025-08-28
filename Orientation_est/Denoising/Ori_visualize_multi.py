import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict, Optional, Union

Array = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: Array) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _read_ts_quat(
    path: str,
    quat_order: str = "xyzw",
    delimiter: Optional[str] = None,
    skip_header_try: Tuple[int, ...] = (0, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    읽기 유틸: 다양한 텍스트 헤더/구분자에 대해 ts와 quaternion을 파싱.
    반환: (ts:[T], quat:[T,4], xyzw 순서로 정규화)
    """
    last_err = None
    for sh in skip_header_try:
        try:
            data = np.genfromtxt(path, delimiter=delimiter, dtype=float, skip_header=sh)
            if data.ndim == 1:
                data = data[None, :]
            # ts 후보: 첫 컬럼
            ts = data[:, 0]
            if data.shape[1] < 5:
                raise ValueError(f"need at least 5 columns (ts + quat), got {data.shape[1]}")
            # 쿼터니언은 마지막 4개 컬럼로 가정
            q = data[:, -4:]
            # 정규화 & 순서 맞추기
            if quat_order.lower() == "wxyz":
                # 현재 q가 [w x y z]면 [x y z w]로 재배열
                q = q[:, [1, 2, 3, 0]]
            # 정규화(안정성)
            norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
            q = q / norm
            return ts, q
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read '{path}': {last_err}")

def load_tracks_from_files(
    file_paths: List[str],
    names: Optional[List[str]] = None,
    quat_order: str = "xyzw",
    centers: Optional[List[Tuple[float, float, float]]] = None,
    cf_size: float = 0.5,
) -> List[Dict]:
    """
    6개(또는 N개) 파일에서 ts, quat을 읽어 트랙 리스트를 만든다.
    - 모든 파일 ts 동일해야 하며, 검증 수행.
    - 각 트랙은 {'name', 'ts', 'R', 'center', 'size'} 로 구성.
    """
    if names is None:
        names = [f"track{i+1}" for i in range(len(file_paths))]
    assert len(names) == len(file_paths)

    # 기본 중심 좌표(최대 6개용): 가로 3 × 세로 2 격자
    default_centers = [(-2.0,  1.0, 0.0),
                       ( 0.0,  1.0, 0.0),
                       ( 2.0,  1.0, 0.0),
                       (-2.0, -1.0, 0.0),
                       ( 0.0, -1.0, 0.0),
                       ( 2.0, -1.0, 0.0)]
    if centers is None:
        centers = default_centers[:len(file_paths)]
    assert len(centers) == len(file_paths)

    tracks = []
    ref_ts = None
    for i, p in enumerate(file_paths):
        print(i, p)
        ts, q = _read_ts_quat(p, quat_order=quat_order)
        if ref_ts is None:
            ref_ts = ts
        elif(len(ref_ts) != len(ts)):
            ts_min = min(len(ref_ts), len(ts))
            print(p, len(ts))
            if ref_ts[0] == ts[0]:
                print('First idx correct ')
                ts = ts[:ts_min]
            else:
                print('First idx does not correct ')
                ts = ts[len(ts) - ts_min:]
        else:
            if not np.allclose(ts, ref_ts, rtol=0, atol=1e-9):
                raise ValueError(f"timestamps mismatch at '{p}'")
        print(len(ts))

        # SciPy는 [x,y,z,w] 순서를 기대
        Rmats = R.from_quat(q).as_matrix()  # (T,3,3)

        tracks.append({
            "name": names[i],
            "ts": ts,
            "R": Rmats,
            "center": np.asarray(centers[i], dtype=float),
            "size": cf_size,
        })
    return tracks

class ViewerMulti:
    """
    다중(예: 6개) 오리엔테이션 시퀀스를 Open3D로 동시 시각화.
    - '.' : 다음 프레임
    - ',' : 이전 프레임
    - 'Q' : 종료
    """
    def __init__(
        self,
        tracks: Optional[List[Dict]] = None,
        # 하위호환: 과거처럼 2개 트랙을 직접 넘기는 경우 (회전행렬 시퀀스)
        gt_data: Optional[Array] = None,
        est_data: Optional[Array] = None,
        gt_name: str = "gt",
        est_name: str = "est",
    ):
        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.first_update = True
        self.stop_update = False
        self.current_frame = 0
        self.length = 0

        self.world_cf = None
        self.track_cfs: List[o3d.geometry.TriangleMesh] = []
        self.track_cacheRs: List[np.ndarray] = []
        self.tracks: List[Dict] = []

        # 입력 경로 1) tracks(dict 리스트)
        if tracks is not None:
            self._init_from_tracks(tracks)
        # 입력 경로 2) 과거(2트랙) 하위호환
        elif gt_data is not None and est_data is not None:
            gt_R = _to_numpy(gt_data)   # (T,3,3) or (T,4) quat
            est_R = _to_numpy(est_data)
            gt_R = self._ensure_rotmats(gt_R)
            est_R = self._ensure_rotmats(est_R)
            T = min(gt_R.shape[0], est_R.shape[0])
            self.tracks = [
                {"name": gt_name,  "ts": np.arange(T), "R": gt_R[:T],  "center": np.array([-1.0, 0.0, 0.0]), "size": 0.5},
                {"name": est_name, "ts": np.arange(T), "R": est_R[:T], "center": np.array([ 1.0, 0.0, 0.0]), "size": 0.5},
            ]
        else:
            raise ValueError("Provide either 'tracks' or both 'gt_data' and 'est_data'.")

        # 길이 결정(동일 ts 전제이므로 첫 트랙 길이 사용)
        self.length = int(min(t["R"].shape[0] for t in self.tracks))
        # 캐시 초기화
        self.track_cacheRs = [np.eye(3, dtype=float) for _ in self.tracks]

    def _lock_camera_orthographic(self):
        vc = self.vis.get_view_control()

        # 0) 창/컨트롤 초기화 보장
        self.vis.poll_events()
        self.vis.update_renderer()

        # 1) 현재 카메라 파라미터를 받아와 "창과 일치하는" W,H를 얻는다
        params = vc.convert_to_pinhole_camera_parameters()
        intr = params.intrinsic
        W, H = intr.width, intr.height  # ★ 이 값을 반드시 사용

        # 2) 보기 고정(원근 감 영향 최소화용 예시 자세)
        vc.set_lookat([0.0, 0.0, 0.0])
        vc.set_front([0.0, 0.0, -1.0])
        vc.set_up([0.0, 1.0, 0.0])

        # 3) 직교 투영이 지원되면 켠다 (버전에 따라 없을 수도)
        try:
            vc.set_orthographic(True)
            self.vis.poll_events();
            self.vis.update_renderer()
            return
        except Exception:
            pass

    # === core API (이전 ViewerSimple과 동일한 호출 흐름 유지) ===
    def connect(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Multi-Orientation Viewer", width=1280, height=720, visible=True)
        self.vis.register_key_callback(ord('Q'), self._callback_quit)
        self.vis.register_key_callback(ord(','), self._callback_prev_frame)
        self.vis.register_key_callback(ord('.'), self._callback_next_frame)

        self._lock_camera_orthographic()
    def disconnect(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis.close()
        self.vis = None
        self.first_update = True

    def set_mesh(self):
        # World frame
        self.world_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
        self.world_cf.compute_vertex_normals()
        self.vis.add_geometry(self.world_cf)
        #self.vis.add_3d_label

        # Track frames
        self.track_cfs = []

        for t in self.tracks:
            cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(t["size"]), origin=t["center"].tolist())
            cf.compute_vertex_normals()
            self.track_cfs.append(cf)
            self.vis.add_geometry(cf)

    def update(self, i: int):
        # 각 트랙에 대해: ΔR = R[i] @ R_cache^T 로 상대 회전 적용
        for idx, t in enumerate(self.tracks):
            R_now = t["R"][i]
            R_prev = self.track_cacheRs[idx]
            dR = R_now @ R_prev.T
            self.track_cfs[idx].rotate(dR, center=t["center"])
            self.track_cacheRs[idx] = R_now

            # 갱신
            self.vis.update_geometry(self.track_cfs[idx])

        self.vis.poll_events()
        self.vis.update_renderer()
        self.first_update = False
        self.current_frame = i

    def pause(self):
        self.stop_update = False
        while not self.stop_update:
            self.vis.poll_events()
            self.vis.update_renderer()

    # === 키 콜백 ===
    def _callback_quit(self, vis):
        self.stop_update = True
        return False

    def _callback_prev_frame(self, vis):
        if self.current_frame > 0:
            self.update(self.current_frame - 1)
        return False

    def _callback_next_frame(self, vis):
        if self.current_frame < self.length - 100:
            self.update(self.current_frame + 100)
        return False

    # === 내부 유틸 ===
    def _init_from_tracks(self, tracks: List[Dict]):
        """사용자 tracks(dict) 입력을 정규화."""
        norm_tracks = []
        ref_ts = None
        for t in tracks:
            name = t.get("name", f"track{len(norm_tracks)+1}")
            center = np.asarray(t.get("center", (0.0, 0.0, 0.0)), dtype=float)
            size = float(t.get("size", 0.5))
            ts = _to_numpy(t["ts"]).reshape(-1)
            R_in = _to_numpy(t["R"])
            Rm = self._ensure_rotmats(R_in)
            if ref_ts is None:
                ref_ts = ts
            else:
                if ts.shape != ref_ts.shape or not np.allclose(ts, ref_ts, rtol=0, atol=1e-9):
                    raise ValueError(f"timestamps mismatch in track '{name}'")
            norm_tracks.append({"name": name, "ts": ts, "R": Rm, "center": center, "size": size})
        self.tracks = norm_tracks

    @staticmethod
    def _ensure_rotmats(x: np.ndarray) -> np.ndarray:
        """
        입력이 (T,3,3) 회전행렬이면 그대로, (T,4) quat이면 회전행렬로 변환.
        """
        if x.ndim == 3 and x.shape[-2:] == (3, 3):
            return x.astype(float)
        if x.ndim == 2 and x.shape[-1] == 4:
            # 입력이 xyzw라고 가정 (필요 시 wxyz → xyzw로 변환해서 넣어줄 것)
            q = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            return R.from_quat(q).as_matrix().astype(float)
        raise ValueError(f"Unsupported orientation array shape: {x.shape}")

# 과거 사용법의 이름을 유지하고 싶으면 VS로 alias
VS = ViewerMulti

# 샘플 실행(직접 실행 시)
if __name__ == "__main__":
    seq = 'a062'
    files = [
        "/home/mpil/astrobee_dataset_revised",
        "/home/mpil/miniconda3/envs/AirIMU/result/Astrobee",
        "/home/mpil/miniconda3/envs/Denoising/results/Astrobee/2025_08_19_12_26_03",
        "/home/mpil/miniconda3/envs/calibnet_denoisingbased/results/Astrobee/2025_08_19_12_43_26",
        "/home/mpil/miniconda3/envs/gyronet/results/Astrobee/2025_08_19_17_02_00"
    ]
    files[0] = os.path.join(files[0], seq, seq + '_gt_q.txt')
    files = [files[0]] + [os.path.join(f,seq,seq+"_ori_q.txt") for f in files[1:]]
    print(files)
    tracks = load_tracks_from_files(files, names=["GT","AirIMU","Denoising","Calibnet","Gyronet"], quat_order="xyzw")
    viewer = VS(tracks=tracks)
    viewer.connect()
    viewer.set_mesh()
    viewer.update(0)
    viewer.pause()
    viewer.disconnect()
