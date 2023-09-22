"""
-------------------------------------------------
Project Name: vehicle-reid-system-st
File Name: demo.py
Author: YUWEI JIANG
Create Date: 2023/4/25
Description：ui for vehicle reid system
-------------------------------------------------
"""

from reid_utils import *
import random


if __name__ == '__main__':
    # Web factors setting
    st.set_page_config(page_title="vehicle reid system",
                       page_icon=":car:",  # label
                       layout="wide",
                       initial_sidebar_state='auto')

    st.title('Vehicle Re-Identification System')
    st.markdown('<h3 style="color: red"> based on YOLOv5, SIFT and Streamlit </h3', unsafe_allow_html=True)
    st.write("This is a vehicle re-identification system supporting video format input\
            developed by ***Yuwei Jiang*** from ***Fuzhou University***")

    # Display time
    st.sidebar.header(f"Current Time：{get_time()}")

    # BGM
    music = st.sidebar.radio('Choose a song：',
                             ['一路向北', 'Rage your dream'],
                             index=random.choice(range(2)))
    st.sidebar.write(f'Playing: {music}... :musical_note:')
    audio_bytes = get_audio_bytes(music)
    st.sidebar.audio(audio_bytes, format='audio/mp3')

    # query图片和视频上传按钮
    image_file_buffer = st.sidebar.file_uploader("Upload a image", type=["png", "jpg", "jpeg"])
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

    if image_file_buffer:
        st.sidebar.text('Query Image')
        st.sidebar.image(image_file_buffer)
        image_name = image_file_buffer.name
        # save image from streamlit into "query" folder for future detect
        with open(os.path.join('query', image_name), 'wb') as f:
            # st.write(image_file_buffer)  # 返回值：UploadedFile(id=2, name='car1.jpg', type='image/jpeg', size=4876)
            # st.write(image_file_buffer.name)  # 返回值：xxx.jpg
            f.write(image_file_buffer.getbuffer())  # 返回值：<memory at 0x000001DAE3679D00>

    if video_file_buffer:
        st.sidebar.text('Original Video')
        st.sidebar.video(video_file_buffer)
        video_name = video_file_buffer.name
        # save video from streamlit into "videos" folder for future detect
        with open(os.path.join('videos', video_name), 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    st.sidebar.markdown('---')  # 分割线

    # 设置图片与视频状态
    image_status = st.empty()
    video_status = st.empty()

    if image_file_buffer is None:
        image_status.markdown('<font size= "4"> **Image Status:** Waiting for input image</font>', unsafe_allow_html=True)
    else:
        image_status.markdown('<font size= "4"> **Image Status:**  Ready </font>', unsafe_allow_html=True)

    if video_file_buffer is None:
        video_status.markdown('<font size= "4"> **Video Status:** Waiting for input video</font>', unsafe_allow_html=True)
    else:
        video_status.markdown('<font size= "4"> **Video Status:**  Ready </font>', unsafe_allow_html=True)

    # 设置超参数
    # conf_thres = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.4)
    # iou_thres = st.sidebar.number_input('IOU', min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    # st.sidebar.markdown('---')

    # 启动按钮
    track_button = st.sidebar.button('START REID')

    # 若按下启动则开始重识别
    if track_button:
        # 更改视频状态为running
        video_status.markdown('<font size= "4"> **Video Status:**  Running... </font>', unsafe_allow_html=True)
        # 设置query图片和视频路径，开始重识别和视频展示
        video_path = "D:/1-PythonProjects/sift_reid/videos/" + video_name  # todo
        query_img_path = "D:/1-PythonProjects/sift_reid/query/" + image_name # todo
        # 开始计时
        t0 = time.time()
        # 调用重识别函数
        vehicle_reid(video_path, query_img_path)
        tx = time.time()
        time_space = tx-t0
        # 更改视频状态为finished
        video_status.markdown(f'<font size= "4"> **Video Status:**   Finished! Time cost: {time_space:.2f}s. Check it below! </font>', unsafe_allow_html=True)
        # 调用展示视频函数
        show_video('REID')