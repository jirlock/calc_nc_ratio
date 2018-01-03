from PIL import Image
from io import BytesIO
import numpy as np

ipath = 'test_images/BNE_10127.jpg'

with open(ipath, 'rb') as f:
    binary = f.read()

print(type(binary))

image = np.asarray(Image.open(BytesIO(binary)))

print(type(image))
print(image.shape)
