import shutil
import os
import re

path_in = "./data/images_original/trainA/"
path_out = "./data/clip_frames/testA/"
os.makedirs(path_out, exist_ok=True)
os.makedirs("./data/clip_frames/testB/", exist_ok=True)
os.makedirs("./data/clip_frames/trainA/", exist_ok=True)
os.makedirs("./data/clip_frames/trainB/", exist_ok=True)

files = os.listdir(path_in)
files = sorted(files, key=lambda s: int(re.findall(r"\d+", s)[-1]))
for i in range(75269, 75570):
    shutil.copyfile(path_in + files[i], path_out + files[i])
