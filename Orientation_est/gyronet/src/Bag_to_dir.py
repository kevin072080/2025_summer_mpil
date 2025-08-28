import bagpy
from bagpy import bagreader
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np

# === 사용자 설정 ===
bag_path = '/home/mpil/EuRoc/MH_03_medium.bag'
output_dir = '/home/mpil/EuRoc/MH_03_medium'
os.makedirs(output_dir, exist_ok=True)

# === BAG 열기 ===
b = bagreader(bag_path)
print('[INFO] Topics:', b.topics)

# === IMU 추출 ===
imu_topic = '/imu0'  # 보통 EuRoC는 이 이름
imu_csv = b.message_by_topic(imu_topic)
print('[INFO] IMU saved to:', imu_csv)

# IMU CSV 리네이밍
os.makedirs(f'{output_dir}/mav0/imu0', exist_ok=True)
imu_dst = f'{output_dir}/mav0/imu0/data.csv'
pd.read_csv(imu_csv).to_csv(imu_dst, index=False)


# === 카메라 이미지 추출 ===
def extract_images(topic_name, cam_id):
    print(f'[INFO] Extracting images from {topic_name}')
    image_msgs = b.read_messages(topic_name)
    image_dir = f'{output_dir}/mav0/{cam_id}/data'
    os.makedirs(image_dir, exist_ok=True)

    csv_rows = []
    for msg in tqdm(image_msgs):
        t = msg.timestamp
        img = msg.message
        np_arr = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        fname = f'{int(t * 1e9)}.png'
        path = os.path.join(image_dir, fname)
        cv2.imwrite(path, np_arr)
        csv_rows.append([int(t * 1e9), fname])

    # 저장용 CSV
    cam_csv_path = f'{output_dir}/mav0/{cam_id}/data.csv'
    pd.DataFrame(csv_rows, columns=['#timestamp [ns]', 'filename']).to_csv(cam_csv_path, index=False)


# === 카메라 두 개 처리 (예: /cam0/image_raw, /cam1/image_raw) ===
extract_images('/cam0/image_raw', 'cam0')
extract_images('/cam1/image_raw', 'cam1')
