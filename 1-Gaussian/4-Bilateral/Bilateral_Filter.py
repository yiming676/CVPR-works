# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def bilateral_filter(image, sigma_s, sigma_r):
#     # 使用 OpenCV 的双边滤波函数
#     filtered_image = cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)
#     return filtered_image
#
#
# # 读取图像
# img_name = r'D:\CV\1-Gaussian\lena.jpg'  # 确保文件名和路径正确
# img_BGR = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
#
# if img_BGR is None:
#     print(f"Image not found or incorrect path: {img_name}")
#     exit()
#
# # 定义三组不同的 sigma_s 和 sigma_r
# sigma_s_list = [5, 10, 20]  # 空间标准差
# sigma_r_list = [25, 50, 75]  # 强度标准差
#
# # 应用双边滤波并显示结果
# plt.figure(figsize=(12, 8))
#
# for i, (sigma_s, sigma_r) in enumerate(zip(sigma_s_list, sigma_r_list), 1):
#     filtered_image = bilateral_filter(img_BGR, sigma_s, sigma_r)
#
#     # 显示原图和滤波后的图像
#     plt.subplot(3, 2, 2 * i - 1)
#     plt.imshow(img_BGR, cmap='gray')
#     plt.title(f'Original Image')
#     plt.axis('off')
#
#     plt.subplot(3, 2, 2 * i)
#     plt.imshow(filtered_image, cmap='gray')
#     plt.title(f'Bilateral Filter - sigma_s={sigma_s}, sigma_r={sigma_r}')
#     plt.axis('off')
#
# plt.tight_layout()
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

def bilateral_filter(image, sigma_s, sigma_r):
    # 使用 OpenCV 的双边滤波函数
    filtered_image = cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    return filtered_image

# 读取图像
img_name = '23.jpg'  # 确保文件名和路径正确
img_BGR = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像

if img_BGR is None:
    print(f"Image not found or incorrect path: {img_name}")
    exit()

# 定义空间标准差和强度标准差
sigma_s_list = [5, 10, 20]  # 空间标准差
sigma_r_list = [25, 50, 75]  # 强度标准差

# 应用双边滤波并显示结果
plt.figure(figsize=(18, 6))  # 调整图像大小以适应9张图片

# 使用嵌套循环遍历所有可能的组合
index = 1
for sigma_s in sigma_s_list:
    for sigma_r in sigma_r_list:
        filtered_image = bilateral_filter(img_BGR, sigma_s, sigma_r)

        # 显示滤波后的图像
        plt.subplot(3, 3, index)  # 使用单一索引
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'sigma_s={sigma_s}, sigma_r={sigma_r}')
        plt.axis('off')
        index += 1

plt.tight_layout()
plt.show()
