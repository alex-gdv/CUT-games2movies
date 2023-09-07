import os
import face_recognition
import numpy as np
from PIL import Image, ImageEnhance

path_images = "./data/images_original/"
path_faces = "./data/faces/"
path_faces_hq = "./data/faces_high_quality/"
dirs = ["trainA", "trainB"]
batch_size = 256



for d in dirs:
    path_input = path_images + d + "/"
    path_output = path_faces + d + "/"
    path_output_hq = path_faces_hq + d + "/"
    os.makedirs(path_output, exist_ok=True)
    os.makedirs(path_output_hq, exist_ok=True)
    files = os.listdir(path_input)
    images = []
    for index_files, f in enumerate(files):
        image = Image.open(path_input + f)
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
                    image_face = images[index_batch][top:bottom, left:right]
                    image_face = Image.fromarray(image_face)
                    enhancer = ImageEnhance.Brightness(image_face)
                    image_face = enhancer.enhance(0.75)
                    image_face = image_face.resize((64,64))
                    image_face.save(f"{path_output}{files[index][:-4]}_{index_face}.jpg")
                    if abs(top - bottom) * abs(left - right) > 4096:
                        image_face.save(f"{path_output_hq}{files[index][:-4]}_{index_face}.jpg")
            images = []
