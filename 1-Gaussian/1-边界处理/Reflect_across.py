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

def mirror_border(img, kernel_size):
    pad_width = kernel_size // 2
    img_padded = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width)), mode='reflect')
    return img_padded

def main():
    # 读取图片
    img_name = 'lena.jpg'
    img_BGR = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

    # 检查图像是否正确加载
    if img_BGR is None:
        print(f"Image not found: {img_name}")
        return

    # 获取图像的尺寸
    if len(img_BGR.shape) == 2:  # 灰度图像
        row, col = img_BGR.shape
        img_GauBlur = np.zeros(img_BGR.shape)
        img_padded = mirror_border(img_BGR, kernel_size)
    else:
        row, col, ch = img_BGR.shape
        img_GauBlur = np.zeros((row, col, ch))
        img_padded = np.dstack([mirror_border(img_BGR[:,:,i], kernel_size) for i in range(ch)])

    # 权重矩阵
    gus2D_kernel_nor = GaussianTemplate(kernel_size, sigma)

    if len(img_BGR.shape) == 2:  # 灰度图像
        for i in range(row):
            for j in range(col):
                img_blk = img_padded[i:i + kernel_size, j:j + kernel_size]
                GauBlur_result = np.sum(np.multiply(gus2D_kernel_nor, img_blk))
                img_GauBlur[i, j] = GauBlur_result
    else:
        for i in range(row):
            for j in range(col):
                for k in range(ch):
                    img_blk = img_padded[i:i + kernel_size, j:j + kernel_size, k]
                    GauBlur_result = np.sum(np.multiply(gus2D_kernel_nor, img_blk))
                    img_GauBlur[i, j, k] = GauBlur_result

    # 显示图像
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_BGR, cmap='gray')
    plt.axis('off')
    plt.title('source image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_GauBlur, cmap='gray')
    plt.axis('on')
    plt.title('GaussianBlur image')
    plt.show()

    img_out_name = img_name[0:4] + '_gaublur_sigma%.2f_py.bmp' % (sigma)
    cv2.imwrite(img_out_name, img_GauBlur )

if __name__ == "__main__":
    main()