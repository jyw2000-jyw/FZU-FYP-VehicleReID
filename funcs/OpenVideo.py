import cv2

# 打开视频文件
cap = cv2.VideoCapture('output/REID.mp4')

# 检查是否成功打开视频
if not cap.isOpened():
    print("Error opening video file")

# 循环读取每一帧并显示
while True:
    # 从视频文件中读取图像帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        break

    # 显示图像帧
    cv2.imshow('Video', frame)

    # 按下ESC键退出循环
    if cv2.waitKey(25) & 0xFF == 27:
        break

# 关闭视频文件和OpenCV窗口
cap.release()
cv2.destroyAllWindows()
