import cv2
import os
import shutil


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

if __name__ == "__main__":
    video_path = "videos/cars.mp4"
    output_dir = "frames"
    extract_frames_from_video(video_path, output_dir)

'''
if os.path.exists('frames'):
    shutil.rmtree('frames')  # 删除output文件夹，清理之前的检测结果
os.makedirs('frames')        # 创建新的output文件夹

# 读取视频文件
video = cv2.VideoCapture(r"D:\1-Python\sift_reid\test\cars.mp4")

# 循环遍历视频的每一帧图像并保存到指定文件夹
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    # 生成帧图像的文件名
    filename = os.path.join('frames', f'frame{frame_count:04d}.jpg')
    # 保存帧图像
    cv2.imwrite(filename, frame)
    frame_count += 1

print(f'The input video has {frame_count} frames')
'''