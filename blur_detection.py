import numpy as np
import cv2
import imutils
import os
import random

# Source:
# https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
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


def delete_blurries(path, thresh):
    files = os.listdir(path)
    for f in files:
        img = cv2.imread(path + f)
        img = imutils.resize(img, width=500)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, blurry = detect_blur_fft(img, thresh=thresh)
        if blurry:
            os.remove(path + f)

# delete_blurries("./data/images_train/trainA/", 5)
delete_blurries("./data/faces2/", -10)

# for _ in range(1000):
#     num = random.randint(1,15000)
#     img = cv2.imread(f"./data/faces2/TheSopranos_{num}_0.jpg")
#     if img is not None:
#         img = imutils.resize(img, width=500)
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         mean, blurry = detect_blur_fft(img2)
#         print(mean)
#         cv2.imshow("", img)
#         cv2.waitKey(0)
