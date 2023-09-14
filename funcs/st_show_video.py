import streamlit as st
import tempfile
import cv2


image_placeholder = st.empty()  # 创建空白块使得图片展示在同一位置

def show_video(video_name):
    video_file = open(f'output/{video_name}.mp4', 'rb')
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)  # opencv打开文件

        if (cap.isOpened() == False):
            st.write("Error opening video stream or file")

        while (cap.isOpened()):
            success, frame = cap.read()
            if success:
                to_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(to_show, caption='Video')  # 将图片帧展示在同一位置得到视频效果
            else:
                break
        cap.release()


if __name__ == '__main__':
    video_name = 'REID'
    show_video(video_name)
