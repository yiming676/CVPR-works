# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义高斯核生成函数
# def GaussianTemplate2D(sigma, kernel_size):
#     gus2D_kernel = np.zeros((kernel_size, kernel_size))
#     kernel_radius = kernel_size // 2
#     for i in range(kernel_size):
#         for j in range(kernel_size):
#             x2 = (i - kernel_radius) ** 2
#             y2 = (j - kernel_radius) ** 2
#             tmp = np.exp(-(x2 + y2) / (2 * sigma ** 2))
#             gus2D_kernel[i, j] = tmp
#     gus2D_kernel /= gus2D_kernel.sum()
#     return gus2D_kernel
#
# def apply_gaussian_filter(image, kernel):
#     return cv2.filter2D(image, -1, kernel)
#
# def main():
#     # 读取图片
#     img_name = 'D:/CV/1-Gaussian/lena.jpg'
#     img_BGR = cv2.imread(img_name, cv2.IMREAD_COLOR)  # 读取彩色图像
#
#     if img_BGR is None:
#         print("Image not found or incorrect path")
#         return
#
#     # 定义两个不同的标准差
#     sigma1 = 10
#     sigma2 = 50
#     kernel_size = int(np.ceil(6 * max(sigma1, sigma2)))
#     if kernel_size % 2 == 0:
#         kernel_size += 1
#
#     # 生成两个不同标准差的高斯核
#     gauss_kernel_1 = GaussianTemplate2D(sigma1, kernel_size)
#     gauss_kernel_2 = GaussianTemplate2D(sigma2, kernel_size)
#
#     # 方法1：先对高斯核作差，再滤波
#     kernel_diff = gauss_kernel_2 - gauss_kernel_1
#     img_filtered_diff = apply_gaussian_filter(img_BGR, kernel_diff)
#
#     # 方法2：先分别用两个高斯滤波处理图像，再对结果作差
#     img_filtered_1 = apply_gaussian_filter(img_BGR, gauss_kernel_1)
#     img_filtered_2 = apply_gaussian_filter(img_BGR, gauss_kernel_2)
#     img_diff = cv2.subtract(img_filtered_2, img_filtered_1)
#
#     # 显示结果
#     plt.figure(figsize=(12, 8))
#     plt.subplot(221), plt.imshow(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)), plt.title('Original'), plt.axis('off')
#     plt.subplot(222), plt.imshow(cv2.cvtColor(img_filtered_diff, cv2.COLOR_BGR2RGB)), plt.title('Filtered Diff'), plt.axis('off')
#     plt.subplot(223), plt.imshow(cv2.cvtColor(img_diff, cv2.COLOR_BGR2RGB)), plt.title('Diff of Filtered'), plt.axis('off')
#     plt.show()
#
#     # 保存结果
#     cv2.imwrite('filtered_diff.jpg', img_filtered_diff)
#     cv2.imwrite('diff_of_filtered.jpg', img_diff)
#
# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义高斯核生成函数
def GaussianTemplate2D(sigma, kernel_size):
    gus2D_kernel = np.zeros((kernel_size, kernel_size))
    kernel_radius = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x2 = (i - kernel_radius) ** 2
            y2 = (j - kernel_radius) ** 2
            tmp = np.exp(-(x2 + y2) / (2 * sigma ** 2))
            gus2D_kernel[i, j] = tmp
    gus2D_kernel /= gus2D_kernel.sum()
    return gus2D_kernel

def apply_gaussian_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def main():
    # 读取图片
    img_name = r'D:\CV\1-Gaussian\23.jpg'
    img_BGR = cv2.imread(img_name, cv2.IMREAD_COLOR)  # 读取彩色图像

    if img_BGR is None:
        print("Image not found or incorrect path")
        return

    # 定义两个不同的标准差
    sigma1 = 3
    sigma2 = 50
    kernel_size = int(np.ceil(6 * max(sigma1, sigma2)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 生成两个不同标准差的高斯核
    gauss_kernel_1 = GaussianTemplate2D(sigma1, kernel_size)
    gauss_kernel_2 = GaussianTemplate2D(sigma2, kernel_size)

    # 方法1：先对高斯核作差，再滤波
    kernel_diff = gauss_kernel_2 - gauss_kernel_1
    img_filtered_diff = apply_gaussian_filter(img_BGR, kernel_diff)

    # 方法2：先分别用两个高斯滤波处理图像，再对结果作差
    img_filtered_1 = apply_gaussian_filter(img_BGR, gauss_kernel_1)
    img_filtered_2 = apply_gaussian_filter(img_BGR, gauss_kernel_2)
    img_diff = cv2.subtract(img_filtered_2, img_filtered_1)

    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.subplot(221), plt.imshow(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)), plt.title('Original'), plt.axis('off')
    plt.subplot(222), plt.imshow(cv2.cvtColor(img_filtered_diff, cv2.COLOR_BGR2RGB)), plt.title('Filtered Diff'), plt.axis('off')
    plt.subplot(223), plt.imshow(cv2.cvtColor(img_diff, cv2.COLOR_BGR2RGB)), plt.title('Diff of Filtered'), plt.axis('off')
    plt.show()

    # 保存结果
    cv2.imwrite('filtered_diff.jpg', img_filtered_diff)
    cv2.imwrite('diff_of_filtered.jpg', img_diff)

if __name__ == "__main__":
    main()