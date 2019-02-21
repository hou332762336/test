import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

img=cv2.imread('20181111.png')
print(img)
##图像二值化
# cv2.THRESH_BINARY
ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)








# b,r,g=cv2.split(img)
#
# img_rgb=cv2.merge([r,g,b])
#
#
#
# cv2.imshow('img',img)
#
# cv2.imshow('img_rgb',img_rgb)
# plt.subplots(20);plt.imshow(img)
# plt.subplots(20);plt.imshow(img_rgb)
# # plt.xticks(([])),plt.yticks([])
# plt.show()
#
# k=cv2.waitKey(0)&0xFF
# if k ==0:
#     cv2.destroyAllWindows()
# elif k==ord('s'):
#     cv2.imwrite('messigray.jpg',img)
#     cv2.destroyallWindows()