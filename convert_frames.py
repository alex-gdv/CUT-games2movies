import cv2
import os
import re

# https://stackoverflow.com/a/44948030/16445870
image_folder = "./results_clip/games2movies_faces_hq/test_latest/images/fake_B/"
video_name = "video.avi"

images = [img for img in sorted(os.listdir(image_folder), key=lambda s: int(re.findall(r"\d+", s)[-1])) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()