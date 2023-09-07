import torchvision.transforms as T
import os
from PIL import Image

path = "./data/images/"
path_new = "./data/images_resized/"
dirs = ["trainA", "trainB"]
for dir in dirs:
    files = os.listdir(path + dir + "/")
    path_temp = path + dir + "/"
    os.makedirs(path_temp, exist_ok=True)
    for f in files:
        img = Image.open(path + dir + "/" + f)
        new_img = T.CenterCrop((512,512))(img)
        new_img.save(path_temp + f)
