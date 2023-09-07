import numpy as np
import imagehash
from PIL import Image
import os
import re

path = "./data/images_train/"

# Source:
# https://towardsdatascience.com/removing-duplicate-or-similar-images-in-python-93d447c1c3eb
def alpharemover(image):
    if image.mode != 'RGBA':
        return image
    canvas = Image.new('RGBA', image.size, (255,255,255,255))
    canvas.paste(image, mask=image)
    return canvas.convert('RGB')

def with_ztransform_preprocess(hashfunc, hash_size=8):
    def function(path):
        image = alpharemover(Image.open(path))
        image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
        data = image.getdata()
        quantiles = np.arange(100)
        quantiles_values = np.percentile(data, quantiles)
        zdata = (np.interp(data, quantiles_values, quantiles) / 100 * 255).astype(np.uint8)
        image.putdata(zdata)
        return hashfunc(image)
    return function

# def delete_copycats(path, clip_name=""):
#     files = os.listdir(path)
#     if clip_name != "":
#         r = re.compile(f"{clip_name}.")
#         files = list(filter(r.match, files))
#     files = sorted(files, key=lambda s: int(re.findall(r"\d+", s)[-1]))
#     for i in range(0, len(files), 1800):
#         j = 0
#         while j < 1800 and j < len(files):
#             pass

# dirs = ["trainA", "trainB"]
# for d in dirs:
#     files = os.listdir()
dhash_z_transformed = with_ztransform_preprocess(imagehash.dhash, hash_size = 8)
x = dhash_z_transformed("./data/faces2/TheGodfather_4992_0.jpg")
y = dhash_z_transformed("./data/faces2/TheGodfather_5015_0.jpg")
print(x-y)