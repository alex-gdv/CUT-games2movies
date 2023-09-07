import cv2
import albumentations as A
import os
import random
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
import shutil

# Source:
# https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py
def apply_make_up(image):
    face_landmarks_list = face_recognition.face_landmarks(image)
    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    return np.array(pil_image)

class MakeUp(A.ImageOnlyTransform):
    def apply(self, img, **params):
        return apply_make_up(img)

# We tested all of Albumentations geometric and photometric transformations
# We include only the transformations that result in realistic images
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

def create_augmented_dataset(path_in, path_out, augment_chance=0.3):
    files = os.listdir(path_in)
    os.makedirs(path_out, exist_ok=True)
    for f in files:
        shutil.copyfile(path_in + f, path_out + f)
        image = cv2.imread(path_in + f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augment = random.random() < augment_chance
        if augment:
            new_image = transform(image=image)["image"]
            Image.fromarray(new_image).save(path_out + f[:-4] + "_a.jpg")
            exit()

create_augmented_dataset("./data/faces_high_quality_bigger/trainA/", "./data/faces_augmented/trainA/")
create_augmented_dataset("./data/faces_high_quality_bigger/trainB/", "./data/faces_augmented/trainB/")
