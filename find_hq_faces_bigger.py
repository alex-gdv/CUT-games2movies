import os
import face_recognition
import numpy as np
from PIL import Image, ImageEnhance
import re
import cv2

path_images = "./data/images_original/"
path_faces_hq = "./data/faces_high_quality_bigger/"
dirs = ["trainA", "trainB"]
batch_size = 16

def detect_blur_fft(image, size=60, thresh=0):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean, mean <= thresh)

for d in dirs:
    path_input = path_images + d + "/"
    path_output_hq = path_faces_hq + d + "/"
    os.makedirs(path_output_hq, exist_ok=True)
    files = os.listdir(path_input)
    files = sorted(files, key=lambda s: int(re.findall(r"\d+", s)[-1]))
    images = []
    for index_files in range(0, len(files), 60):
        image = Image.open(path_input + files[index_files])
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)
        image = np.array(image)
        images.append(image)
        if len(images) == batch_size:
            face_locations_batch = face_recognition.batch_face_locations(images, batch_size=batch_size)
            for index_batch, face_locations in enumerate(face_locations_batch):
                index = index_files + 1 - batch_size + index_batch
                for index_face, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location
                    if abs(top - bottom) * abs(left - right) > 128*64:
                        increase_width = min((right - left) / 2, left, 1280 - right)
                        increase_height = min((bottom - top) / 2, top, 720 - bottom)
                        image_face_np = images[index_batch][int(top - increase_height):int(bottom + increase_height),
                                                            int(left - increase_width):int(right + increase_width)]
                        image_face = Image.fromarray(image_face_np)
                        enhancer = ImageEnhance.Brightness(image_face)
                        image_face = enhancer.enhance(0.75)
                        image_face = image_face.resize((256,256))
                        _, blurry = detect_blur_fft(cv2.cvtColor(image_face_np, cv2.COLOR_RGB2GRAY), size=128, thresh=-10)
                        if not blurry:
                            image_face.save(f"{path_output_hq}{files[index][:-4]}_{index_face}.jpg")
            images = []
