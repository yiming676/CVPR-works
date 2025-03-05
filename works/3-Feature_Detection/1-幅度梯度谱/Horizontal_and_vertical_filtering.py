import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def gaussian_kernel(size, sigma):
    assert size % 2 == 1, "核大小必须是奇数。"
    radius = (size - 1) // 2
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(x, y)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g, x, y

def gaussian_derivative_kernel(size, sigma):
    g, x, y = gaussian_kernel(size, sigma)
    Gx = -x * g / (sigma ** 2)
    Gy = -y * g / (sigma ** 2)
    return Gx, Gy

# 标准差
sigma = 5
# 核大小
size = 21
# 生成核
Gx, Gy = gaussian_derivative_kernel(size, sigma)

# 读取图像
image = cv2.imread(r'D:\\CV\\2-2-FeatureDetection\\lena.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("无法读取图像，请检查路径是否正确。")

# 确保核的尺寸不超过图像的尺寸
if size > min(image.shape):
    size = min(image.shape) - 1
    Gx, Gy = gaussian_derivative_kernel(size, sigma)

# 对图像进行填充以适应核的尺寸
padded_image = cv2.copyMakeBorder(image, size//2, size//2, size//2, size//2, cv2.BORDER_CONSTANT, value=0)

# 应用高斯一阶微分滤波器
Ix = cv2.filter2D(padded_image, -1, Gx)
Iy = cv2.filter2D(padded_image, -1, Gy)

# 裁剪填充的部分以恢复原始图像尺寸
Ix = Ix[size//2:-size//2, size//2:-size//2]
Iy = Iy[size//2:-size//2, size//2:-size//2]

# 显示梯度图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(Ix, cmap='gray')
plt.title(f'Horizontal Direction - Ix ($\sigma$={sigma})')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(Iy, cmap='gray')
plt.title(f'Vertical Direction - Iy ($\sigma$={sigma})')
plt.colorbar()

plt.show()