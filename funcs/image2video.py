import cv2
import os
import shutil
import streamlit as st
from PIL import Image, ImageOps

if os.path.exists('output'):
    shutil.rmtree('output')  # 删除output文件夹，清理之前的检测结果
os.makedirs('output')        # 创建新的output文件夹

def image2video(save_path):
    # 得到图像路径
    files = os.listdir("frames/")
    # 对图像排序
    # files.sort(key = lambda x: int(x.split(".")[0]))
    # 获取图像宽高
    h, w, _ = cv2.imread("frames/" + files[0]).shape
    # 设置帧数
    fps = 30
    vid = []

    # 准备写入视频
    vid = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 写入
    for file in files:
        img = cv2.imread("frames/" + file)
        vid.write(img)

    print('视频已生成！')


if __name__ == "__main__":
    save_path = "output/video.mp4"		# 保存视频路径和名称 MP4格式
    image2video(save_path)

    # Streamlit应用程序标题
    st.title("展示视频示例代码")

    # 上传视频文件
    video_file = st.file_uploader("上传视频文件", type=["mp4"])

    # 检查是否选择了视频文件
    if video_file is not None:
        # 将视频文件读取为字节流
        video_bytes = video_file.read()
        # 在Streamlit应用程序中显示视频
        st.video(video_bytes)
    else:
        # 如果没有选择文件，则在应用程序中显示消息
        st.write("请选择一个MP4格式的视频文件")

