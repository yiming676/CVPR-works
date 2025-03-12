import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(image, title):
    # 计算傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # 加1避免对数为负无穷

    # 计算相位谱
    phase_spectrum = np.angle(fshift)

    # 绘制幅度谱
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
img_name = r'D:\CV\1-Gaussian\lena_color.jpg'  # 确保文件名和路径正确
img_BGR = cv2.imread(img_name, cv2.IMREAD_COLOR)  # 读取彩色图像

if img_BGR is None:
    print(f"Image not found or incorrect path: {img_name}")
    exit()

# 分别处理 B, G, R 通道
plt.figure(figsize=(12, 12))
titles = ['Blue Channel', 'Green Channel', 'Red Channel']
for i, color in enumerate(['b', 'g', 'r']):
    img_channel = img_BGR[:, :, i]

    # 将图像转换为 float32 类型
    img_float = np.float32(img_channel)

    # 计算傅里叶变换
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)

    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # 加1避免对数为负无穷

    # 计算相位谱
    phase_spectrum = np.angle(fshift)

    # 绘制幅度谱
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'{titles[i]} Magnitude Spectrum')
    plt.axis('off')

    # 绘制相位谱
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.title(f'{titles[i]} Phase Spectrum')
    plt.axis('off')

    plt.tight_layout()
    plt.show()