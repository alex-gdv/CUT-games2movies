import os

path_files = "./data/faces_high_quality2/trainA/"
# path_files = "./data/test/"
path_output = "./AttGAN-PyTorch/data/"

files = os.listdir(path_files)
labels = open(path_output + "labels.txt", "w")
labels.write(f"{len(files)}\n")
labels.write("Bald Bangs Black_Hair Blond_Hair Brown_Hair Bushy_Eyebrows Eyeglasses Male Mouth_Slightly_Open Mustache No_Beard Pale_Skin Young\n")
for f in files:
    s = f
    for _ in range(13):
        s += "\t-1"
    labels.write(s + "\n")