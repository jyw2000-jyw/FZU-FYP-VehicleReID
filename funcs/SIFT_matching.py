import cv2
import numpy as np

input_image = cv2.imread('D:/1-Python/sift_reid/query/query4.png')

# 创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 检测输入图像的特征点和描述符
keypoints, descriptors = sift.detectAndCompute(input_image, None)

# 绘制输入图像的特征点
input_keypoints = cv2.drawKeypoints(input_image, keypoints, None)
cv2.imwrite('input_keypoints.jpg', input_keypoints)

# 定义需要缩放的尺度
scales = [0.9, 0.8, 0.7, 0.6]

# 遍历不同的缩放比例
for i, scale in enumerate(scales):
    # 对输入图像进行缩放
    scaled_image = cv2.resize(input_image, None, fx=scale, fy=scale)

    # 检测缩放后图像的特征点和描述符
    scaled_keypoints, scaled_descriptors = sift.detectAndCompute(scaled_image, None)

    # 进行描述符匹配
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(descriptors, scaled_descriptors, k=2)

    # 选择出较好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:   # 设置匹配的阈值
            good_matches.append(m)

    # 绘制匹配的特征点
    matched_image = cv2.drawMatches(input_image, keypoints, scaled_image, scaled_keypoints,
                                    good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # 根据缩放比例命名并保存绘制的匹配结果
    output_filename = f'matched_{scale}.jpg'
    cv2.imwrite(output_filename, matched_image)
