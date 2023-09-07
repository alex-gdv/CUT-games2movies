import face_recognition
import os
import numpy as np
from PIL import Image
import random
import queue

path_train = "./data/images_faces/"
path_test = "./data/images_faces_test/"
dirs = ["trainA", "trainB"]

for d in dirs:
    path_train_temp = path_train + d + "/"
    files = os.listdir(path_train_temp)
    encodings = []
    for f in files:
        image = Image.open(path_train_temp + f)
        try:
            encoding = face_recognition.face_encodings(np.array(image), known_face_locations=[[0,128,128,0]])[0]
            encodings.append(encoding)
        except IndexError:
            print("Face not detected.")

    path_temp_test = path_test + d + "/"
    test_set_indices = []
    test_set_size = int(len(files) * 0.3)

    def find_close_faces(index, visited_face_indices, num):
        new_visited_face_indices = visited_face_indices
        new_visited_face_indices += [index]
        q = queue.Queue()
        q.put(index)
        while not q.empty() and len(new_visited_face_indices) < num:
            index = q.get()
            face_distances = face_recognition.face_distance(encodings[:index] + encodings[index+1:], encodings[index])
            list_indices = np.argsort(face_distances)
            i = 0
            while face_distances[list_indices[i]] <= 0.6 and len(new_visited_face_indices) < num:
                if list_indices[i] >= index and (list_indices[i] + 1) not in new_visited_face_indices:
                    q.put(list_indices[i]+1)
                    new_visited_face_indices += [list_indices[i] + 1]
                elif list_indices[i] < index and list_indices[i] not in new_visited_face_indices:
                    q.put(list_indices[i])
                    new_visited_face_indices += [list_indices[i]]
                i += 1
        return new_visited_face_indices

    while len(test_set_indices) < test_set_size:
        index = random.randint(0, len(encodings))
        while index in test_set_indices:
            index = random.randint(0, len(encodings))
        test_set_indices = find_close_faces(index, test_set_indices, test_set_size)
    
    for i in test_set_indices:
        os.makedirs(path_temp_test, exist_ok=True)
        os.rename(path_train_temp + files[i], path_temp_test + files[i])
