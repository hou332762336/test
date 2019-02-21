#coding:utf-8

from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
import sys
import time

class RBF(object):
    """构造rbf类"""
    def __init__(self, indim, numCenters, outdim):
        """创建构造函数，numCenters为中心点"""
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, i) for i in range(numCenters)]
        
        self.beta = 100
        """高斯基函数修正参数"""
        self.W = random.random((self.numCenters, self.outdim))
        
    def _basisfunc(self, c, d):
        """c, d分别为样本点和中心点"""
        assert len(d) ==self.indim
        return exp(-self.beta*norm(c-d)**2)
    
    def _calcAct(self, X):
        """计算径向基函数矩阵，
        X为样本点
        """
        G = zeros((X.shape[0], self.numCenters),float)
        for ci, c in enumerate(self.centers) :
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G
    
    def train(self, X, Y):
        """开始训练，利用矩阵求逆的法则来计算W权重值，shuffle"""
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
        print("center", end="\n")
        print(self.centers)
        G = self._calcAct(X)
        print(G)
        self.W = dot(pinv(G), Y)
    
    def test(self, X):
        G = self._calcAct(X)   
        Y = dot(G, self.W)
        return Y


if __name__ == "__main__":
    n = 100
    x = mgrid[-1:1:complex(0, n)].reshape(n, 1)
    y = sin(3*(x+0.5)**3-1)
    
    rbf = RBF(1, 10, 1)
    """RBF类的实例化"""
    rbf.train(x, y)
    z = rbf.test(x)
    
    """开始展示，首先确定作图区域"""
    plt.figure(figsize = (12, 8))
    """绘制原来函数形态"""
    plt.plot(x, y, 'k-')
    """画出拟合函数"""
    plt.plot(x, z, 'r-', linewidth = 2)
    """绘制径向基函数点，即变换后的中心点"""
    
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')
    
    for c in rbf.centers:
        cx = arange(c-0.7, c+0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        
        plt.plot(cx, cy, "-", color = "gray", linewidth = 0.2)
    plt.xlim(-1.2, 1.2)
    plt.show()















                
        
        
        
        
        