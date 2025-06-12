import cv2
import numpy as np
from PIL import Image

# 텍스처 이미지 불러오기
img = cv2.imread("inter/blub_20/blub.png")  # BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 마스크 생성: 거의 검은색인 픽셀을 선택 (예: R,G,B < 10)
threshold = 10
mask = np.all(img_rgb < threshold, axis=2).astype(np.uint8) * 255  # shape: (H, W), dtype: uint8

# inpainting
inpainted = cv2.inpaint(img_rgb, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# 저장
Image.fromarray(inpainted).save("final/blub_20.png")

