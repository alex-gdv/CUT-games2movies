from PIL import Image

# Source:
# https://stackoverflow.com/a/44231784/16445870
def make_bigger(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    print(x,y,size)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

test = Image.open("./data/test/TheGodfather_599_1.jpg")
test = make_bigger(test)
test = test.convert("RGB")
test.save("./AttGAN-PyTorch/data/custom2/stupid.jpg")