import os
import random
import shutil

def create_mini_set(path_in, path_out, num=1000):
    os.makedirs(path_out, exist_ok=True)
    files = os.listdir(path_in)
    for _ in range(num):
        index = random.randint(0, len(files)-1)
        shutil.copyfile(path_in + files[index], path_out + files[index])
        del files[index]

# create_mini_set("./data/images/testA/", "./data/mini_test_set_frames/testA/")
#ccreate_mini_set("./data/images/testB/", "./data/mini_test_set_frames/testB/")
# create_mini_set("./data/images_faces_test/testA/", "./data/mini_test_set_faces/testA/")
create_mini_set("./data/images_faces_test/testB/", "./data/mini_test_set_faces/testB/", num=794)
