import cv2
import os

path = "./data/"  
path_images = f"{path}./images/"

def convert_video_to_frames(path_input, path_output, name):
    os.makedirs(path_output, exist_ok=True)
    cap = cv2.VideoCapture(path_input)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    while cap.isOpened():
        _, frame = cap.read()
        cv2.imwrite(f"{path_output}./{name}_{count+1}.jpg", frame)
        count = count + 1
        if (count > (video_length-1)):
            cap.release()
            break

if __name__=="__main__":
    dirs = [("game", "trainA"), ("movie", "trainB")]
    for d in dirs:
        path_curr = f"{path}./{d[0]}/"
        files = os.listdir(path_curr)
        for f in files:
            path_input = f"{path_curr}./{f}"
            path_output = f"{path_images}./{d[1]}/"
            print(f"Current: {path_input}")
            convert_video_to_frames(path_input, path_output, f[:-4])
