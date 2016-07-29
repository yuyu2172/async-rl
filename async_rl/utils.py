# There are a number of options for doing this:
# 1. opencv
# 2. PIL resize

try:
   import cv2
   imresize = cv2.resize
except:
   import numpy as np
   import PIL.Image

   def tmp(img,size):
        return np.array(PIL.Image.fromarray(img).resize(size,PIL.Image.BILINEAR))

   imresize = tmp
