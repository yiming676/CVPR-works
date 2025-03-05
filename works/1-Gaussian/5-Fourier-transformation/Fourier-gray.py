import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(image, title):
    # 计算傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 计算相位谱
    phase_spectrum = np.angle(fshift)

    # 绘制幅度谱s
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')

    # 绘制相位谱
    plt.subplot(1, 2, 2)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.title('Phase Spectrum')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 读取图像
img_name = r'D:\CV\1-Gaussian\lena.jpg'  # 确保文件名和路径正确
img_BGR = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像

if img_BGR is None:
    print(f"Image not found or incorrect path: {img_name}")
    exit()

# 将图像转换为 float32 类型
img_float = np.float32(img_BGR)

# 对图像进行傅里叶变换并显示幅度谱和相位谱
plot_spectrum(img_float, 'Fourier Spectrum')