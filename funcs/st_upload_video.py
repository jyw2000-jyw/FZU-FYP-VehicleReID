import streamlit as st
from PIL import Image, ImageOps

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
