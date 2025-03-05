import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. 读取彩色图像
image = cv2.imread('D:/CV/2-1-Image_Transform/23.jpg')
rows, cols, _ = image.shape

# 定义变换矩阵
def get_transformation_matrices():
    transformations = {
        'Translation': np.array([[1, 0, 100], [0, 1, 50]], dtype=np.float32),
        'Rotation': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0],
                              [np.sin(np.pi/4), np.cos(np.pi/4), 0]], dtype=np.float32),
        'Euclidean': np.array([[1, 0, 100], [0, 1, 50]], dtype=np.float32),
        'Similarity': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 100],
                                [np.sin(np.pi/4), np.cos(np.pi/4), 50]], dtype=np.float32),
        'Affine': np.array([[0.866, 0.5, 50], [-0.5, 0.866, 100]], dtype=np.float32)
    }
    return transformations

# 计算逆矩阵
def invert_transformation(transformation):
    # 将2x3矩阵转换为3x3矩阵
    matrix = np.vstack([transformation, [0, 0, 1]])
    # 计算逆矩阵
    inverse_matrix = np.linalg.inv(matrix).astype(np.float32)
    # 移除最后一行
    inverse_transformation = inverse_matrix[:2, :]
    return inverse_transformation

# 2. 应用几何变换及其逆变换
def apply_transformations(image, transformations):
    images = []
    titles = []
    interpolation_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
    for name, transformation in transformations.items():
        for method in interpolation_methods:
            # 应用正向变换
            transformed_image = cv2.warpAffine(image, transformation, (cols, rows), flags=method)
            images.append(transformed_image)
            titles.append(f'{name} - {method}')

            # 计算逆矩阵
            inverse_transformation = invert_transformation(transformation)
            # 应用逆向变换
            inverse_transformed_image = cv2.warpAffine(transformed_image, inverse_transformation, (cols, rows), flags=method)
            images.append(inverse_transformed_image)
            titles.append(f'Inverse {name} - {method}')
    return images, titles

# 3. 显示结果
def show_results(images, titles):
    n = len(images)
    for i in range(0, n, 4):
        plt.figure(figsize=(10, 5))
        for j in range(4):
            if i + j < n:
                plt.subplot(2, 2, j+1)
                plt.imshow(cv2.cvtColor(images[i+j], cv2.COLOR_BGR2RGB))
                plt.title(titles[i+j])
                plt.axis('off')
        plt.show()

# 主函数
def main():
    transformations = get_transformation_matrices()
    images, titles = apply_transformations(image, transformations)
    show_results(images, titles)

if __name__ == '__main__':
    main()