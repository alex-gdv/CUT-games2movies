import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

def compare(paths, filenames, name):
    num_rows = len(paths)
    num_cols = len(os.listdir(paths[0])) + 1
    fig = plt.figure(figsize=(num_cols,num_rows))
    ax = [fig.add_subplot(num_rows,num_cols,i+1) for i in range(num_rows*num_cols)]
    # f, axarr = plt.subplots(len(paths), len(os.listdir(paths[0])) + 1)
    for i, path in enumerate(paths):
        folders = os.listdir(path)
        image = Image.open(f"{path}{folders[0]}/test_latest/images/real_A/{filenames[i]}")
        images = [image]
        cols = ["Input"] + folders
        for folder in folders:
            image = Image.open(f"{path}{folder}/test_latest/images/fake_B/{filenames[i]}")
            images.append(image)
        for j in range(num_cols):
            ax[i * num_cols + j].imshow(images[j]) 
            ax[i * num_cols + j].axis("off")
            ax[i * num_cols + j].set_xticklabels([])
            ax[i * num_cols + j].set_yticklabels([])
            ax[i * num_cols + j].set_aspect("equal")
            if i == 0:
                ax[i * num_cols + j].set_title(cols[j], fontsize=5)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    fig.savefig(name, dpi=1000.0, bbox_inches="tight")

paths = ["./results/games2movies/", "./results/movies2games_/", 
        "./results_faces/games2movies/", "./results_faces/movies2games_/"]
filenames = ["MafiaVideogame_180640.png", "TheGodfather_10944.png", 
            "MafiaVideogame_117721_0.png", "TheGodfather_2094_0.png"]
compare(paths, filenames, "comparison.png")
