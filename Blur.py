import cv2
import numpy as np
from PIL import Image

# 이미지 읽기
image = cv2.imread('data/content/blub.png')
image = image.astype(np.float32) / 255.0  # 0~1로 정규화

# 가우시안 노이즈 생성
mean = 0
stddev = 0.5  # 노이즈 표준편차 (값이 클수록 노이즈가 큼)
gaussian_noise = np.random.normal(mean, stddev, image.shape)

# 이미지에 노이즈 추가
noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 1)  # 값 범위를 0~1로 클리핑

# (H, W, C) 형태를 uint8로 변환
noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)

# BGR(OpenCV) → RGB(PIL)
noisy_image_rgb = cv2.cvtColor(noisy_image_uint8, cv2.COLOR_BGR2RGB)

# PIL로 저장
Image.fromarray(noisy_image_rgb).save('data/content/blur/blub_0.5.png')
