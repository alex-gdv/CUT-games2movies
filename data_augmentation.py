import cv2
import albumentations as A
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import numpy as np

def save_images(images):
    width, height = 256, 256
    new_width = width * len(images)
    new_img = Image.new("RGB", (new_width, height))
    _, axarr = plt.subplots(1,len(images))
    for i in range(len(images)):
        axarr[i].imshow(images[i]) 
        axarr[i].axis("off")
        # new_img.paste(np.reshape(images[i], (3, width, height)), (i * width, 0, width*(i+1), height))
        new_img.paste(Image.fromarray(images[i]), (width * i, 0))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    new_img.save("./data/augmented.png")

# we tested all of Albumentations geometric and photometric transformations
# we include only the transformations that result in realistic images
transform = A.Compose([
    A.OneOf([
        A.AdvancedBlur(),
        A.GaussianBlur(),
        A.MedianBlur(),
        A.MotionBlur(),
    ]),
    A.OneOf([
        A.GaussNoise(),
        A.MultiplicativeNoise(),
        A.ISONoise(),
    ]),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.RandomToneCurve(),
    ]),

    A.Emboss(),
    A.RingingOvershoot(),
    A.Sharpen(),
    A.OneOf([
        A.CLAHE(),
        A.Equalize(),
    ]),
    A.FancyPCA(),
    A.HorizontalFlip(),
    A.OneOf([
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.ShiftScaleRotate(),
    ]),
])


path = "./data/faces_high_quality_bigger/trainA/"
# def augment_data(path, num_per_img=10):
files = os.listdir(path)
index = random.randint(0,len(files))
image = cv2.imread(path + files[index])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images = [image]
for _ in range(3):
    images.append(transform(image=image)["image"])
save_images(images)