"""
Author : Stéphane KPOVIESSI
"""

import os
import cv2

def read_videos(input_dir):
    """
    Reads all the video files in the input directory and returns the video capture objects and the total number of frames for each video.
    """
    video_caps = {}
    total_frames = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(input_dir, filename)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                continue
            total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_caps[filename] = cap
            total_frames[filename] = total_frames_count

    return video_caps, total_frames

def save_frames(video_caps, total_frames, output_dir):
    """
    Saves three frames per second from the video capture objects to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename, cap in video_caps.items():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 3)  # Save one frame every 1/3 second
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                output_path = os.path.join(output_dir, f"{filename}_frame_{frame_count // frame_interval}.jpg")
                cv2.imwrite(output_path, frame)

            frame_count += 1

        cap.release()