import shutil

from imageai.Detection import VideoObjectDetection
from moviepy.video.io.VideoFileClip import VideoFileClip
from pytube import YouTube

import os

execution_path = os.getcwd()

if not os.path.exists('los-angeles-car-driving-cropped.mp4'):
    if not os.path.exists('los-angeles-car-driving.mp4'):
        # Завантаження відео з YouTube
        yt = YouTube('https://www.youtube.com/watch?v=Cw0d-nqSNE8', use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(file_extension='mp4').first()
        stream.download(output_path='./', filename='los-angeles-car-driving.mp4')

    video = VideoFileClip('los-angeles-car-driving.mp4').subclip(50, 60)
    video.write_videofile('los-angeles-car-driving-cropped.mp4', fps=20)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, 'pretrained/yolov3.pt'))
detector.loadModel()

output_path = os.path.join(execution_path, 'output')
shutil.rmtree(output_path, ignore_errors=True)
os.makedirs(output_path)

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, 'los-angeles-car-driving-cropped.mp4'),
    output_file_path=os.path.join(output_path, 'task2-output')
    , frames_per_second=20, log_progress=True)
print(video_path)
