
from utils import read_videos, save_frames

# Set the input and output directories
input_dir = "whatvid"
output_dir = "whatvid_output"

# Read the videos and get the video capture objects and total frames
video_caps, total_frames = read_videos(input_dir)

# Save the frames from the video capture objects
save_frames(video_caps, total_frames, output_dir)
