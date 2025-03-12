import os
import shutil
from sklearn.model_selection import train_test_split

# 定义源数据集路径
source_dir = r"E:\CVPR-works\4-3-images_classification\Dataset-caltech256\Datasets\ol"

# 定义目标数据集路径
train_dir = "E:/CVPR-works/4-3-images_classification/Dataset-caltech256/Datasets/train"
test_dir = "E:/CVPR-works/4-3-images_classification/Dataset-caltech256/Datasets/test"
val_dir = "E:/CVPR-works/4-3-images_classification/Dataset-caltech256/Datasets/val"

# 创建目标文件夹
for directory in [train_dir, test_dir, val_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 遍历每个类别的文件夹
for class_dir in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_dir)

    # 确保是文件夹
    if os.path.isdir(class_path):
        # 获取所有图像文件
        images = [os.path.join(class_path, f) for f in os.listdir(class_path) if
                  os.path.isfile(os.path.join(class_path, f))]

        # 分割图像为训练、测试和验证集
        train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
        test_images, val_images = train_test_split(temp_images, test_size=0.5, random_state=42)

        # 创建类别的目标文件夹
        for target_dir, image_list in [(train_dir, train_images), (test_dir, test_images), (val_dir, val_images)]:
            class_target_path = os.path.join(target_dir, class_dir)
            if not os.path.exists(class_target_path):
                os.makedirs(class_target_path)

            # 复制图像到目标文件夹
            for image in image_list:
                shutil.copy(image, class_target_path)

print("数据集分割完成！")