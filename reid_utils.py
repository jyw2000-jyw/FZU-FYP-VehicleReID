import streamlit as st
import torch
import numpy as np
import shutil
import os
import cv2  # 计算机视觉
import datetime
import tempfile  # 临时文件
import time
from scipy.spatial.distance import cdist
from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load


# 获取当前时间
def get_time():
    now_time = datetime.datetime.now()
    ymd = now_time.strftime('%Y-%m-%d')
    return ymd


# 音乐
def get_audio_bytes(music):
    audio_file = open(f'music/{music}.mp3', 'rb')
    audio_bytes = audio_file.read()
    audio_file.close()
    return audio_bytes


# 展示视频
def show_video(video_name):
    st.markdown('Reid Video:')
    image_placeholder = st.empty()
    while True:  # 循环展示视频
        video_file = open(f'output/{video_name}.mp4', 'rb')
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)

            if (cap.isOpened() == False):
                st.write("Error opening video stream or file")

            while (cap.isOpened()):
                success, frame = cap.read()
                if success:
                    to_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(to_show)
                else:
                    # 如果到达了视频的结尾，将游标指向开头使得视频循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break

            cap.release()
    return


# 读取query图片，并进行预处理
def load_and_resize_image(path):
    # 加载图片
    img = cv2.imread(path)
    # 调整大小为 256x256 像素
    resized_img = cv2.resize(img, (256, 256))
    return resized_img


# 车辆重识别
def vehicle_reid(video_path,
                 query_img_path,
                 weights_path='yolov5s.pt',
                 save_path='output/REID.mp4',
                 conf_thres=0.4,
                 iou_thres=0.45):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 删除 output 文件夹，清理之前的检测结果
    if os.path.exists('output'):
        shutil.rmtree('output')
    # 创建新的 output 文件夹
    os.makedirs('output')

    # 统计输入视频的帧数
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'输入视频的帧数为：{frame_count}')

    def preprocess_image(image):
        input_size = 640  # yolov5要求输入网络的图片大小为640
        h, w = image.shape[:2]
        aspect_ratio = input_size / max(h, w)  # 计算缩放比例
        resized_h, resized_w = int(h * aspect_ratio), int(w * aspect_ratio)  # 调整图片的尺寸
        image = cv2.resize(image, (resized_w, resized_h))

        pad_h = input_size - resized_h  # 填充图片
        pad_w = input_size - resized_w
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # 将处理后的图片转换为浮点数格式，并保持数值在0到1之间。变换维度，将通道维度放在前面。添加新维度，并且转换为pytorch张量
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).to(device)
        return image

    weights = weights_path
    model = attempt_load(weights)  # 读取训练好的模型
    model.to(device)

    # 初始化 SIFT 特征提取器
    sift = cv2.SIFT_create()

    # 读取 query 图片并预处理
    query_img = load_and_resize_image(query_img_path)

    # 提取 query 图片的 SIFT 特征
    _, query_des = sift.detectAndCompute(query_img, None)

    # 读取视频，并获取视频帧数和帧大小
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧大小
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    # 记录检测到的车辆信息和对应的特征点
    cars_info = []
    cars_des = []

    # 定义编解码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建VideoWriter对象
    outVid = cv2.VideoWriter(save_path, fourcc, 30.0, size)

    # 记录开始时间
    start_time = time.time()

    # 统计识别了多少帧
    finish_count = 0

    # 创建一个进度条对象，并设置初始值为0
    progress_bar = st.progress(0)

    # 创建一个文本对象来显示进度百分比
    progress_text = st.empty()

    # 循环播放输入视频中的每一帧
    for i in range(frame_count):
        # 从视频文件中读取下一帧
        ret, frame = cap.read()

        # 如果没有更多的帧了，就脱离循环
        if not ret:
            break

        # 预处理帧
        img = preprocess_image(frame)

        ratio = max(frame.shape[0], frame.shape[1]) / 640
        # 使用YOLOv5模型检测可能的车辆位置
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False,
                                   max_det=100)

        # 循环遍历每个车辆，在视频帧上裁剪并进行特征提取和相似度计算
        for j, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = det[:, :4].clamp(min=0, max=frame.shape[1])  # 检测框不能超出图像范围
                max_sim_total = 0.0  # 记录最大相似度分值
                max_sim_box = None  # 记录最大相似度分值对应的框
                results = []  # 记录每个框的信息
                for x1, y1, x2, y2, conf, cls in reversed(det):
                    if int(cls) in [2, 5, 7]:

                        x1 = int(x1 * ratio)
                        y1 = int(y1 * ratio)
                        x2 = int(x2 * ratio)
                        y2 = int(y2 * ratio)

                        # 裁剪出当前车辆 图像，并提取 SIFT 特征
                        input_img = frame[y1:y2, x1:x2]
                        input_img = cv2.resize(input_img, (256, 256))
                        _, input_des = sift.detectAndCompute(input_img, None)

                        if input_des is not None:
                            input_des = input_des.reshape(-1, input_des.shape[-1])
                        else:
                            # 如果没有检测到特征点，则将 input_des 赋值为空数组
                            break

                        # 计算当前车辆与 query 图片的相似度
                        if len(cars_info) == 0:
                            sim_total = 0.0
                        else:
                            # 使用欧几里得距离度量方法计算查询query图片和当前输入图片的各个特征描述符之间的距离。
                            dists = cdist(query_des, input_des, metric='euclidean')
                            # 取每个查询特征描述符的最近邻距离比率
                            nn_ratio = np.nanmin(dists, axis=0) / (np.sort(dists, axis=0)[1, :] + 1e-6)
                            # 生成一个掩码数组,其中值为 True 的索引表示相对应的特征描述符的最小距离比率低于阈值 0.8
                            mask = nn_ratio < 0.8
                            # 计算掩码数组中值为 True 的元素个数，除以总元素个数得到匹配比例，作为当前车辆与查询图片的相似度。
                            sim_total = np.sum(mask) / float(nn_ratio.shape[0])

                        results.append(
                            {'sim_total': sim_total, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'des': input_des})

                        if sim_total > max_sim_total:
                            max_sim_total = sim_total
                            max_sim_box = (x1, y1, x2, y2)

                # 根据最大相似度分值画蓝框和其他框的绿框
                for res in results:
                    if res['sim_total'] == max_sim_total:
                        cv2.rectangle(frame, (res['x1'], res['y1']), (res['x2'], res['y2']), (255, 0, 0), 2)
                        cars_info.append((res['x1'], res['y1'], res['x2'], res['y2']))
                        cars_des.append(res['des'])
                    else:
                        cv2.rectangle(frame, (res['x1'], res['y1']), (res['x2'], res['y2']), (0, 255, 0), 2)

        # 更新进度条与文本
        progress_percent = (i + 1) / frame_count
        progress_bar.progress(progress_percent)
        progress_text.text(f"Reid Progress: {progress_percent * 100:.2f}%")

        # 显示标注后的帧
        outVid.write(frame)
        # 计数
        finish_count += 1

        print(f'第{finish_count}帧重识别完成！')
        if finish_count == frame_count:
            # 记录结束时间
            end_time = time.time()
            # 计算代码运行时间
            elapsed_time = end_time - start_time
            print(f'重识别任务完成！总耗时：{elapsed_time:.2f}s，文件保存至：{save_path}。')
        else:
            continue


# 视频抽帧
def extract_frames_from_video(video_path: str, output_dir: str) -> int:
    """
    Extracts frames from a video file and saves them as individual image files.

    Args:
        video_path (str): The path of the input video file.
        output_dir (str): The path of the directory where the extracted frames will be saved.

    Returns:
        int: The total number of frames in the input video.
    """
    # Delete the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create the output directory
    os.makedirs(output_dir)

    # Open the input video file
    video = cv2.VideoCapture(video_path)

    # Loop over each frame in the video and save it as an image file
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f'frame{frame_count:04d}.jpg')
        cv2.imwrite(filename, frame)
        frame_count += 1

    # Print the total number of frames processed
    print(f'The input video has {frame_count} frames')

    # Return the total number of frames processed
    return frame_count


''' 已废弃
def get_audio_bytes(video_name):
    video_file = open(f'videos/{video_name}.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
'''