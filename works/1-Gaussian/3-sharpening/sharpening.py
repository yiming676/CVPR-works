import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_kernel(sigma, kernel_size):
    """生成高斯核"""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel /= kernel.sum()
    return kernel

def apply_sharpening_filter(image, sigma):
    """应用锐化滤波算子"""
    kernel_size = int(6 * sigma)
    gaussian_kernel = generate_gaussian_kernel(sigma, kernel_size)
    blurred_image = cv2.filter2D(image, -1, gaussian_kernel)
    sharpened_image = cv2.add(image, cv2.subtract(image, blurred_image))
    return sharpened_image

def main():
    img_name = 'D:/CV/1-Gaussian/23.jpg'
    img_BGR = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img_BGR is None:
        print("Image not found or incorrect path")
        return

    sigma = 50  # 可以调整标准差来看不同的效果
    sharpened_img = apply_sharpening_filter(img_BGR, sigma)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Sharpened Image (sigma={sigma})')
    plt.axis('off')

    plt.show()
    cv2.imwrite('sharpening_sigma=50.jpg', sharpened_img)
if __name__ == "__main__":
    main()