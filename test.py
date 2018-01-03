from PIL import Image
from io import BytesIO
import numpy as np

ipath = 'test_images/BNE_10127.jpg'

with open(ipath, 'rb') as f:
    binary = f.read()

print(type(binary))

img = Image.open(BytesIO(binary))

image = np.asarray(img)

print(type(img))
print(img)
print(type(image))
print(image.shape)