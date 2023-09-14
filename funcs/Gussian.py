import cv2
import numpy as np

# 读取图片
img = cv2.imread('D:/1-Python/sift_reid/query/query4.png')

# 定义金字塔层数和缩放比例
num_levels = 5
scale_factor = 0.5

# 创建一个空列表，用于存储每个金字塔图像
pyramid_images = []

# 添加原始图像到金字塔列表中
pyramid_images.append(img)

# 计算并添加金字塔上层图像
for i in range(num_levels - 1):
    img_copy = pyramid_images[i].copy()
    blurred = cv2.GaussianBlur(img_copy, (3, 3), 0)
    downsampled = cv2.resize(blurred, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    pyramid_images.append(downsampled)

# 绘制金字塔图像
output_image = None
for i in range(num_levels):
    img = pyramid_images[i]
    if output_image is None:
        # 如果是第一张图像，则创建输出图像
        output_image = img
    else:
        # 否则将当前图像拼接到输出图像右侧
        new_width = output_image.shape[1] + img.shape[1]
        new_height = max(output_image.shape[0], img.shape[0])
        temp_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        temp_image[:output_image.shape[0], :output_image.shape[1]] = output_image
        temp_image[:img.shape[0], output_image.shape[1]:] = img
        output_image = temp_image

# 保存输出图像到本地文件夹
cv2.imwrite('output.png', output_image)
