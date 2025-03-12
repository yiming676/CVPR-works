import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r'D:\CV\2-2-FeatureDetection\lena.jpg', cv2.IMREAD_GRAYSCALE)

# 1. 高斯滤波
sigma = 5
gaussian_blurred = cv2.GaussianBlur(image, (0, 0), sigma)

# 2. 一阶差分求梯度
grad_x = cv2.Sobel(gaussian_blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gaussian_blurred, cv2.CV_64F, 0, 1, ksize=3)

# 3. 求IxIx, IyIy, IxIy图像
IxIx = grad_x ** 2
IyIy = grad_y ** 2
IxIy = grad_x * grad_y

# 4. 高斯核滤波
IxIx_blurred = cv2.GaussianBlur(IxIx, (5, 5), sigma)
IyIy_blurred = cv2.GaussianBlur(IyIy, (5, 5), sigma)
IxIy_blurred = cv2.GaussianBlur(IxIy, (5, 5), sigma)

# 5. 构造二阶矩矩阵M
M = np.dstack((IxIx_blurred, IxIy_blurred, IyIy_blurred))

# 6. 构造Harris角点响应函数R
k = 0.04
R = M[:, :, 0] * M[:, :, 2] - M[:, :, 1] ** 2 - k * (M[:, :, 0] + M[:, :, 2]) ** 2

# 7. 利用正阈值T对响应函数R进行判断
T = 0.01
candidates = R > T

# 8. 利用非最大化抑制（NMS）求局部极值
nms = np.copy(candidates)
for i in range(1, R.shape[0] - 1):
    for j in range(1, R.shape[1] - 1):
        if candidates[i, j]:
            if R[i, j] < np.max([R[i-1, j], R[i+1, j], R[i, j-1], R[i, j+1]]):
                nms[i, j] = 0

# 显示结果
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(IxIx, cmap='gray'), plt.title('IxIx')
plt.subplot(232), plt.imshow(IyIy, cmap='gray'), plt.title('IyIy')
plt.subplot(233), plt.imshow(IxIy, cmap='gray'), plt.title('IxIy')
plt.subplot(234), plt.imshow(R, cmap='gray'), plt.title('Harris Response')
plt.subplot(235), plt.imshow(candidates, cmap='gray'), plt.title('Candidates')
plt.subplot(236), plt.imshow(nms, cmap='gray'), plt.title('NMS')
plt.show()