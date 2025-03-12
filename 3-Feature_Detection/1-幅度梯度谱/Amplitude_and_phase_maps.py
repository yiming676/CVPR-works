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
g, x, y = gaussian_kernel(size, sigma)
Gx = -x * g / (sigma ** 2)
Gy = -y * g / (sigma ** 2)

# 读取图像
image_path = r'D:\CV\2-2-FeatureDetection\lena.jpg'  # 替换为你的图像文件路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("无法读取图像，请检查路径是否正确。")

# 确保核的尺寸不超过图像的尺寸
if size > min(image.shape):
    size = min(image.shape) - 1
    g, x, y = gaussian_kernel(size, sigma)
    Gx = -x * g / (sigma ** 2)
    Gy = -y * g / (sigma ** 2)

# 对图像进行填充以适应核的尺寸
padded_image = cv2.copyMakeBorder(image, size//2, size//2, size//2, size//2, cv2.BORDER_CONSTANT, value=0)

# 应用高斯一阶微分滤波器
Ix = cv2.filter2D(padded_image, -1, Gx)
Iy = cv2.filter2D(padded_image, -1, Gy)

# 裁剪填充的部分以恢复原始图像尺寸
Ix = Ix[size//2:-size//2, size//2:-size//2]
Iy = Iy[size//2:-size//2, size//2:-size//2]

# 计算幅度图和方向图
I_M = np.sqrt(Ix**2 + Iy**2)
I_D = np.arctan2(Iy, Ix)

# # 将方向图转换为 HSV 颜色空间
# I_DHSV = np.zeros((I_D.shape[0], I_D.shape[1], 3), dtype=np.uint8)
# I_DHSV[..., 0] = ((np.degrees(I_D) + 360) % 360).astype(np.uint8)  # 色相
# I_DHSV[..., 1] = 255  # 饱和度
# I_DHSV[..., 2] = cv2.normalize(I_M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 值

# 显示幅度图和方向图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(I_M, cmap='gray')
plt.title('幅度图')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I_D)
plt.title('方向图')
plt.axis('off')  # 不显示坐标轴

plt.show()