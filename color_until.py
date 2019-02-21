import numpy as np
import pandas as pd
import cv2
from  sklearn.cluster import KMeans
from  sklearn.decomposition import PCA




def rgbtohsi(r,g,b):

    # 归一化到[0,1]
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    den = np.sqrt((r-g)**2+(r-b)*(g-b))
    theta = np.arccos(0.5 * ((r-g)+(r-b))/den)
    if den == 0:
            H = 0
    elif b <= g:
        H = theta
    else:
        H = 2*(np.pi) - theta
    min_RGB = min(min(b, g), r)
    sum = b+g+r
    if sum == 0:
        S = 0
    else:
        S = 1 - 3*min_RGB/sum
    # H = H/(2*(np.pi))
    I = sum/3.0
    return H,S,I


df_color=pd.read_excel('D:/data/color.xlsx')
df=df_color.drop('RGB',axis=1).join(df_color['RGB'].str.split(' ',expand=True))
df.rename(columns={0:'R',1:'G',2:'B'},inplace=True)
df[['R','G','B']]=df[['R','G','B']].apply(pd.to_numeric).astype(float)


# color_dict={df.color_name:df[['R','G','B']]}
hsi_list=[ list(rgbtohsi(df.iloc[:,2].values[i],df.iloc[:,3].values[i],df.iloc[:,4].values[i]))for i in range(df.shape[0]) ]

X=np.array(hsi_list)
kmean=KMeans(n_clusters=15,init="k-means++",random_state=28)
kmean.fit(X)
print('所有样本距离簇中心点的总距离总和:',kmean.inertia_)
print('距离聚簇中心点的平均距离:',(kmean.inertia_/455))
cluster_center=kmean.cluster_centers_
# print



