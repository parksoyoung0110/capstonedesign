## 프로젝트 요약 (Project Summary)

본 프로젝트는 스타일 전이 과정에서 자주 손실되는 텍스처의 구조적 디테일을 효과적으로 보존하는 것을 목표로 하였으며, 이를 위해 다음과 같은 기술적 접근을 적용하였다.

1. Laplacian, Sobel, Canny Edge Detector를 활용하여 콘텐츠 이미지의 경계 정보를 추출하고, 스타일 전이 결과 이미지와 비교하여 디테일을 보존하는 Edge Loss를 설계하였다.

![alt text](image-4.png)

3. 기존 픽셀 기반 콘텐츠 손실 함수 대신 SSIM(Structural Similarity Index) 기반의 콘텐츠 손실 함수를 적용하여 시각적 유사성과 구조적 일관성을 유지하였다.

4. 스타일 전이 후 저주파 성분을 제거하고 고주파 성분을 더하기 위해 Guided Filter를 활용한 후처리 과정을 적용하였다.

5. 콘텐츠 이미지에 패턴이 부족하여 스타일 전이가 잘 이루어지지 않는 경우를 보완하고자 Gaussian 노이즈를 추가하여 스타일의 패턴이 더욱 효과적으로 전이되도록 유도하였다.

6. 2D UV 텍스처에서 발생하는 시각적 불일치 문제를 해결하기 위해 differentiable rendering 기반의 멀티뷰 스타일 전이 방식을 적용하여 3D 모델의 시각적 일관성을 확보하였다.

![alt text](image-5.png)

---
## Code Instruction
1. 로스 모듈을 수정한 스타일 전이(Edge Loss, SSIM Loss)
```bash
python TestArtistic.py
```
trainingOutput 폴더에 있는 .pth 파일을 지정하여 미리 학습 되어 있는 원하는 엣지 디텍터 기반 로스 모델과 SSIM Loss로 대체된 모델을 선택해 사용할 수 있다.

2. 후처리
```bash
cd PostProcessing
python detail.py
```
스타일이 전이된 텍스처 이미지에 대해 후처리를 진행한다.

3. 노이즈 추가
```bash
python Blur.py
```
패턴이 나타나지 않는 경우, 콘텐츠 이미지에 노이즈를 추가하여 해결할 수 있다.

4. 렌더링을 통한 스타일 전이
```bash
python Texture.py
```
원하는 obj와 그에 맞는 기존 텍스처 이미지를 지정하여 사용 가능하다.

---
## 실험 환경 (Windows)
- Python: 3.7.1
- PyTorch: 1.7.1 + cu110
- OpenCV: 4.11.0.86
- TensorFlow: 2.6.0
- PyOpenGL: 3.1.5
---
## Result
**Edge loss에 대한 결과**

![alt text](image.png)


**content loss를 ssim loss로 대체한 결과**

![alt text](image-1.png)


**Guided Filter를 통한 후처리 결과**

![alt text](image-2.png)


**렌더러를 이용한 스타일 전이 결과**

![alt text](image-3.png)

---

## 결론 및 향후 과제 (Conclusion and Future Work)
본 연구에서는 텍스처 스타일 전이에서 중요한 구조적 디테일을 효과적으로 보존하기 위해 Edge 기반 손실 함수와 SSIM 기반 Content Loss를 도입하였다. 이를 통해 기존 픽셀 단위 손실의 한계를 극복하고, 스타일 전이 과정에서 발생하는 색상 왜곡 문제를 완화할 수 있었다. 특히 Canny 엣지 디텍터를 활용한 엣지 손실은 시각적 디테일 보존에 효과적이었으며, SSIM 손실은 이미지의 구조적 유사성을 보다 정밀하게 유지하는 데 기여하였다. 또한, 후처리 기법을 통한 고주파 성분 강화 및 노이즈 추가를 통한 패턴 전이 개선, 3D 공간 기반 스타일 전이 방식 도입으로 UV 절개선 문제를 해소하는 등 다양한 기술적 진전을 이루었다.


향후 연구에서는 엣지 손실과 SSIM 손실의 가중치 조절을 보다 정교하게 최적화하여 스타일과 구조 보존 간의 균형을 강화할 필요가 있다. 또한, 다양한 스타일 이미지 및 복잡한 3D 모델에 대한 적용 범위를 넓혀 실용성을 검증하고, 실시간 스타일 전이 기술 개발을 위한 연산 효율 개선도 병행해야 할 것이다. 나아가 사용자 맞춤형 스타일 조절 인터페이스 개발과 함께, AR/VR, 게임, 디지털 콘텐츠 제작 등 다양한 분야에서의 응용 가능성을 확장하는 방향으로 연구를 지속하는 것을 제안한다.
