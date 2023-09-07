import os
import re

path_abs = os.getcwd()
path_images = "./data/images/"
SPLIT = 0.3

movies = ["TheGodfather", "TheSopranos", "TheIrishman"]
files = os.listdir(path_images + "./trainB/")
for movie in movies:
    r = re.compile(f"{movie}.")
    files2 = list(filter(r.match, files))
    num = int(len(files2) * SPLIT)
    files2 = sorted(files2, key=lambda s: int(re.findall(r"\d+", s)[-1]))
    files2 = files2[-num:]
    for f in files2:
        p = f"{path_abs}/data/images/testB/"
        os.makedirs(p, exist_ok=True)
        os.rename(f"{path_abs}/data/images/trainB/{f}", p + f)

files = os.listdir(path_images + "./trainA/")
num = int(len(files) * SPLIT)
files = sorted(files, key=lambda s: int(re.findall(r"\d+", s)[-1]))
files = files[-num:]
for f in files:
    p = f"{path_abs}/data/images/testA/"
    os.makedirs(p, exist_ok=True)
    os.rename(f"{path_abs}/data/images/trainA/{f}", p + f)
