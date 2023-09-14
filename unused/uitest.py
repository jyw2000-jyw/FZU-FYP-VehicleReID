"""
-------------------------------------------------
Project Name: vehicle-reid-system-st
File Name: uitest.py
Author: YUWEI JIANG
Create Date: 2023/3/25
Description：ui for vehicle reid system
-------------------------------------------------
"""

# from track import * # 这个一导入就说get_time未定义
from reid_utils import *
import streamlit as st
import random
import torch
import cv2
import os
import shutil
import time
import numpy as np
from scipy.spatial.distance import cdist
from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load
from reid_utils import load_and_resize_image


if __name__ == '__main__':

    # 设置网页
    st.set_page_config(page_title="vehicle reid system",
                       page_icon=":car:",
                       layout="wide",
                       initial_sidebar_state='auto')

    st.title('Vehicle Re-Identification System')
    st.markdown('<h3 style="color: red"> based on YOLOv5, SIFT and Streamlit </h3', unsafe_allow_html=True)
    st.write("This is a vehicle re-identification system supporting video format\
        developed by ***Yuwei Jiang*** from ***Fuzhou University***")

    # 时间设置
    st.sidebar.header(f"Current Time：{get_time()}")

    # BGM音乐
    music = st.sidebar.radio('Choose a song：',
                             ['卡农', 'Summer'],
                             index=random.choice(range(2)))
    st.sidebar.write(f'Playing: {music}... :musical_note:')
    audio_bytes = get_audio_bytes(music)
    st.sidebar.audio(audio_bytes, format='audio/mp3')

    # upload image and video
    image_file_buffer = st.sidebar.file_uploader("Upload a image", type=["png", "jpg", "jpeg"])
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

    if image_file_buffer:
        st.sidebar.text('Input image')
        st.sidebar.image(image_file_buffer)
        # save image from streamlit into "images" folder for future detect
        with open(os.path.join('query', image_file_buffer.name), 'wb') as f:
            f.write(image_file_buffer.getbuffer())

    if video_file_buffer:
        st.sidebar.text('Input video')
        st.sidebar.video(video_file_buffer)
        # save video from streamlit into "videos" folder for future detect
        with open(os.path.join('videos', video_file_buffer.name), 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    st.sidebar.markdown('---')

    # file status
    image_status = st.empty()
    video_status = st.empty()
    stframe = st.empty()

    if image_file_buffer is None:
        image_status.markdown('<font size= "4"> **Image Status:** Waiting for input image</font>', unsafe_allow_html=True)
    else:
        image_status.markdown('<font size= "4"> **Image Status:** Ready </font>', unsafe_allow_html=True)

    if video_file_buffer is None:
        video_status.markdown('<font size= "4"> **Video Status:** Waiting for input video</font>', unsafe_allow_html=True)
    else:
        video_status.markdown('<font size= "4"> **Video Status:** Ready </font>', unsafe_allow_html=True)

    frames, fps, reid = st.columns(3)
    with frames:
        st.markdown('**FRAMES**')
        frames_text = st.markdown('__')

    with fps:
        st.markdown('**FPS**')
        fps_text = st.markdown('__')

    with reid:
        st.markdown('**REID VIDEO**')
        reid_text = st.markdown('__')

    track_button = st.sidebar.button('START REID')

    if track_button:
        video_status.markdown('<font size= "4"> **Video Status:** Running... </font>', unsafe_allow_html=True)

        if os.path.exists('output'):
            shutil.rmtree('output')  # 删除output文件夹，清理之前的检测结果
        os.makedirs('output')  # 创建新的output文件夹
        st.write('output文件夹已生成!')

        # 统计输入视频的帧数
        video_path = "videos/cars.mp4"
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f'输入视频的帧数为：{frame_count}')


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


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        weights = 'yolov5s.pt'
        model = attempt_load(weights)  # 读取训练好的模型
        model.to(device)

        # 设置阈值和IOU（非极大抑制）阈值
        conf_thres = 0.4
        iou_thres = 0.45

        # 初始化SIFT特征提取器
        sift = cv2.SIFT_create()

        # 读取query图片并预处理
        image_path = r"D:\1-Python\sift_reid\query\car1.jpg"  # 路径一定要英文，不然会报错，除非用numpy的方法处理
        query_img = load_and_resize_image(image_path)

        # 提取query图片的SIFT特征
        _, query_des = sift.detectAndCompute(query_img, None)

        # # 将特征矩阵转换为二维数组
        # query_des = query_des.reshape(-1, query_des.shape[-1])

        # 读取视频，并获取视频帧数和帧大小
        video_path = r"D:\1-Python\sift_reid\videos\cars.mp4"  # 路径一定要英文，不然会报错，除非用numpy的方法处理
        cap = cv2.VideoCapture(video_path)

        # 获取视频帧大小
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)
        # print(f'视频帧大小：{size}')

        # 记录检测到的车辆信息和对应的特征点
        cars_info = []
        cars_des = []

        # 定义编解码器并创建VideoWriter对象
        save_path = "output/video.mp4"
        fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outVid = cv2.VideoWriter(save_path, fourcc, fps, size)

        # 记录开始时间
        start_time = time.time()
        # 完成帧统计
        finish_count = 0
        # 循环遍历视频每一帧，进行车辆重识别并标注出每个车辆的位置
        while True:
            # 读取视频中的帧
            ret, frame = cap.read()  # ret返回值为True，若为False则表示帧已提取完成，跳出循环
            if not ret:
                break

            # 对帧进行预处理
            img = preprocess_image(frame)

            ratio = max(frame.shape[0], frame.shape[1]) / 640
            # 使用 yolov5s.pt 模型检测出可能的车辆位置
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False,
                                       max_det=100)  # 去除重叠框

            # 循环遍历每个车辆，在视频帧上裁剪并进行特征提取和相似度计算
            for i, det in enumerate(pred):
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

            # 显示标注后的帧
            outVid.write(frame)
            finish_count += 1
            st.write(f'第{finish_count}帧重识别完成！')
            if finish_count == frame_count:
                # 记录结束时间
                end_time = time.time()
                # 计算代码运行时间
                elapsed_time = end_time - start_time
                video_status.markdown('<font size= "4"> **Video Status:** Finished ! </font>', unsafe_allow_html=True)
                st.write(f'重识别任务完成！总耗时：{elapsed_time:.2f}s，文件保存至：{save_path}。')
                # st.video(video1, format='video/mp4')
            else:
                continue

        # 调用函数展示视频
        time.sleep(10)
        st.caption("REID VIDEO")
        # image_placeholder = st.empty()  # 创建空白块使得图片展示在同一位置
        # show_video('REID')
        # cap.release()

'''
        # reset ID and count from 0
        reset()
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = f'videos/{video_file_buffer.name}'

        video_status.markdown('<font size= "4"> **Status:** Running... </font>', unsafe_allow_html=True)
        with torch.no_grad():
            detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text, assigned_class_id)
        video_status.markdown('<font size= "4"> **Status:** Finished ! </font>', unsafe_allow_html=True)
'''
