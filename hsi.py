import cv2
import numpy as np
import math
from interval import Interval

def rgbtohsi(r,g,b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    den=np.sqrt((r-g)**2+(r-b)*(g-b))
    theta=np.arccos((0.5*(r-g+r-b)/den))
    if den==0:
        H=0
    elif g>=b:
        H=theta
    else:
        H=2*np.pi-theta
    min_rgb=min(min(b,g),r)

    min_rgb=min(b,g,r)
    sum=b+g+r
    if sum ==0:
        S=0
    else:
        S=1-3*min_rgb/sum
    # H=H/(2*np.pi)
    I=sum/3.0
    return H,S,I

def hsiToRGB(H,S,I):

    n=np.pi
    if H in Interval(0,2*n/3,upper_closed=False):
        B=I*(1-S)
        R=I*(1+S*(np.cos(H))/(np.cos(n/3-H)))
        G=3*I-R-B
    elif H in Interval(2*n/3,4*n/3,upper_closed=False):
        H-=2*n/3
        R=I*(1-S)
        G=I*(1+(S*(np.cos(H)))/(np.cos(n/3-H)))
        B=3*I-R-G
    else:
        H-=4*n/3
        G=I*(1-S)
        B=I*(1+(S*(np.cos(H)))/(np.cos(n/3-H)))
        R=3*I-G-B
    return int(R*255),int(G*255),int(B*255)






if __name__=='__main__':
    img=cv2.imread('D:/data/rgb_hsi/3.png')
    print(img.shape[:2])
    # hsi=[rgbtohsi(img.item(i,j,2),img.item(i,j,1),img.item(i, j, 0))for i in range(img.shape[0]) for j in range(img.shape[1])]
    # hsi=set(hsi)
    # print(hsi)
    H,S,I=rgbtohsi(255,0,0)
    print(H,S,I)
    b,g,r=cv2.split(img)
    R,G,B=hsiToRGB(0.0, 1.0,0.3333333333333333)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i,j]=B
            g[i, j]=G
            r[i,j]=R
    # R=img[:,:,2]
    # G=img[:,:1]
    # B=img[:,:,0]
    # zeros=np.zeros(img.shape[:3],dtype='uint8')
    cv2.imshow('img',img)
    img1=np.zeros(img.shape[:2],dtype='uint8')
    img1=cv2.merge([B, G, R])
    cv2.imshow("MERGE",img1 )
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
    # print(np.pi/3)
    # print(Interval(0,2*np.pi/3,upper_closed=False))



    # h_list=[]
    # s_list=[]
    # i_list=[]
    # hsi=[]
    # RGB=[[250,221,209],[250,211,209],[250,209,230],[244,182,156],[244,160,156],[244,156,200]]
    # for j in range(6):
    #     H,S,I =rgbtohsi(RGB[j][0],RGB[j][1],RGB[j][2])
    #     h_list.append(H)
    #     s_list.append(S)
    #     i_list.append(I)
    #     hsi.append((h_list[j],s_list[j],i_list[j]))
    # # print(h_list)
    # # print(i_list)
    # # print(s_list)
    # print(hsi)

    # rgb_lwpImg=cv2.imread('AAA.jpg')
    # hsi_lwpImg = rgbtohsi(rgb_lwpImg)
    #

