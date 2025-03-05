import cv2
import matplotlib.pyplot as plt

def gaussian_pyramid(image, levels=3):
    gaussian_pyramid = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        next_image = cv2.pyrUp(gaussian_pyramid[i])
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], next_image)
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid[::-1]  # Reverse to match the Gaussian pyramid order

# 加载图像
image = cv2.imread(r'D:\CV\2-1-Image_Transform\23.jpg')

# 生成高斯金字塔
gaussian_pyramid = gaussian_pyramid(image, 5)

# 显示高斯金字塔的每一层
plt.figure(figsize=(12, 6))
for i, g_img in enumerate(gaussian_pyramid, 1):
    plt.subplot(2, len(gaussian_pyramid), i)
    plt.imshow(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Gaussian Layer {i}')
    plt.axis('off')  # 不显示坐标轴
plt.tight_layout()
plt.show()

# 根据高斯金字塔生成拉普拉斯金字塔
laplacian_pyramid = laplacian_pyramid(gaussian_pyramid)

# 显示拉普拉斯金字塔的每一层
plt.figure(figsize=(12, 6))
for i, l_img in enumerate(laplacian_pyramid, 1):
    plt.subplot(2, len(laplacian_pyramid), i)
    plt.imshow(cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Laplacian Layer {i}')
    plt.axis('off')  # 不显示坐标轴
plt.tight_layout()
plt.show()