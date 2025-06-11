import cv2
import numpy as np
import os
from PIL import Image  # Pillow 라이브러리 추가
from core.filter import GuidedFilter

# 입력 및 출력 디렉토리 설정
input_dir = 'btw'     # 입력 이미지 폴더 경로
output_enhanced_dir = 'Alpha_20'   # 디테일 향상 이미지 저장 폴더

# 출력 폴더 없으면 생성

os.makedirs(output_enhanced_dir, exist_ok=True)

# Guided Filter 파라미터
r = 16
eps = 0.01
alpha =20

# 디렉토리 내 이미지 파일 처리
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(input_dir, filename)

        # Pillow로 이미지 열기
        pil_image = Image.open(image_path)

        # ICC 프로파일 제거 (색상 프로파일을 RGB로 변환)
        pil_image = pil_image.convert("RGB")

        # Pillow에서 OpenCV 형식으로 변환
        I = np.array(pil_image).astype(np.float32) / 255.0
        p = I.copy()

        q = np.zeros_like(I)

        # 채널별 Guided Filter 적용
        for c in range(3):
            GF = GuidedFilter(I[:, :, c], r, eps)
            q[:, :, c] = GF.filter(p[:, :, c])

        # 디테일 향상
        I_enhanced = q + alpha * (I - q)
        I_enhanced = np.clip(I_enhanced, 0, 1)

        # 저장 경로 지정
        base_name = os.path.splitext(filename)[0]

        enhanced_path = os.path.join(output_enhanced_dir, f'{base_name}.jpg')

        # 이미지 저장

        cv2.imwrite(enhanced_path, cv2.cvtColor((I_enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        

        print(f'Processed: {filename}')
