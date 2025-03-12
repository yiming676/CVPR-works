import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# 标准差，窗口大小
sigma = 5
kernel_size = math.ceil(6 * sigma)
if kernel_size % 2 == 0:
    kernel_size = kernel_size + 1

def GaussianTemplate(ker_s, sig):
    gus2D_kernel = np.zeros([ker_s, ker_s])
    kernel_radius = math.floor(ker_s / 2)

    for i in range(0, ker_s):
        x2 = pow(i - kernel_radius, 2)
        for j in range(0, ker_s):
            y2 = pow(j - kernel_radius, 2)
            tmp = np.exp(-(x2 + y2) / (2 * sig * sig))
            gus2D_kernel[i, j] = tmp

    gus2D_kernel_nor = gus2D_kernel / np.sum(gus2D_kernel)
    return gus2D_kernel_nor

def pad_with_wrap(img, kernel_size):
    padded_img = np.zeros((img.shape[0] + kernel_size - 1, img.shape[1] + kernel_size - 1))
    kernel_radius = math.floor(kernel_size / 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            padded_img[i + kernel_radius, j + kernel_radius] = img[i, j]
            if i < kernel_radius:
                padded_img[i + kernel_radius + img.shape[0], j + kernel_radius] = img[i, j]
            if i >= img.shape[0] - kernel_radius:
                padded_img[i + kernel_radius - img.shape[0], j + kernel_radius] = img[i, j]
            if j < kernel_radius:
                padded_img[i + kernel_radius, j + kernel_radius + img.shape[1]] = img[i, j]
            if j >= img.shape[1] - kernel_radius:
                padded_img[i + kernel_radius, j + kernel_radius - img.shape[1]] = img[i, j]
    return padded_img

def main():
    img_name = 'lena.jpg'
    img_BGR = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if img_BGR is None:
        print("Image not found")
        return

    row, col = img_BGR.shape[:2]

    # 权重矩阵
    gus2D_kernel_nor = GaussianTemplate(kernel_size, sigma)

    img_GauBlur = np.zeros((row, col))

    # 边界环绕
    img_pool = pad_with_wrap(img_BGR, kernel_size)

    for i in range(0, row):
        for j in range(0, col):
            img_blk = img_pool[i:i + kernel_size, j:j + kernel_size]
            GauBlur_result = np.sum(np.multiply(gus2D_kernel_nor, img_blk))
            img_GauBlur[i, j] = GauBlur_result

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_BGR, 'gray')
    plt.axis('off')
    plt.title('source image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_GauBlur, 'gray')
    plt.axis('on')
    plt.title('GaussianBlur image')
    plt.show()

    img_out_name = img_name[0:4] + '_gaublur_sigma%.2f_py.bmp' % sigma
    cv2.imwrite(img_out_name, img_GauBlur)

if __name__ == "__main__":
    main()