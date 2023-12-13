## 定义超圆盘分类器类
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint  
from cvxopt import matrix,solvers 
solvers.options['show_progress'] = False

class nnhd(BaseEstimator, ClassifierMixin):
    def __init__(self,sigma=0) -> None:
        super().__init__()
        self.sigma = sigma
        self.classes = 0
        self.s_r = {}
        self.beta = {}
        self.accuracy = 0

    def gaussian_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (self.sigma ** 2)))
    
    def fit(self, X, y):
        # 拟合函数
        classes = np.unique(y)
        Num = np.size(classes)
        beta = np.zeros((Num, y.shape[0]))
        s_and_r = np.ones((Num, X.shape[1]+1+1))
        # 分类求解超圆盘对偶问题拉格朗日系数，以求解s和r
        for i, class_name in enumerate(classes):
            Xx = X[y==class_name, :]
            M = Xx.shape[0]
            P = [[self.gaussian_kernel(Xx[ii], Xx[jj]) for jj in range(M)] for ii in range(M)]
            P = 2*np.array(P)
            q = -np.diag(P)/2
            
            """ P = 2*Xx@Xx.T
            q = -np.diag(Xx@Xx.T) """
            G = np.concatenate((np.eye(M), -np.eye(M)), axis=0)
            h = np.concatenate((np.ones([M, 1]), -np.zeros([M, 1])), axis=0)
            P = matrix(np.array(P), tc='d')
            q = matrix(np.array(q), tc='d')
            G = matrix(np.array(G), tc='d')
            h = matrix(np.array(h), tc='d')
            A = matrix(np.ones([1, M]), tc='d')
            b = matrix(np.array([[1]]), tc='d')
            
            sol = solvers.qp(P, q, G, h, A, b)
            result1 = np.array(sol['x']).flatten()
            beta[i-1, :M] = result1                  
            s = np.sum(result1.reshape(-1, 1)*Xx, 0)
            
            r = np.max(np.linalg.norm(Xx-s, ord=2,axis=1))
            s_and_r[i-1, :] = np.concatenate([[int(classes[i])], s, [r]])
            
        self.s_r = s_and_r
        self.beta = beta
   
                
    def predict(self,X):
        y_pred = np.zeros([X.shape[0], 1])
        s_r = self.s_r
        for i in range(X.shape[0]):
            y_pred[i] = s_r[np.argmin(np.linalg.norm(X[i, :]-s_r[:, 1:-1], ord=2,axis=1)), 0]
        
        y_pred = np.array(y_pred, dtype=int)
        return  pd.Categorical(y_pred.flatten())
