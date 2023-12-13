## 定义超圆盘分类器类
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint  
import pickle
from cvxopt import matrix,solvers 
import cvxpy as cvx
import mosek

class hd(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=0) -> None:
        super().__init__()
        self.sigma = sigma
        self.classes = 0
        self.s_r = {}
        self.beta = {}
        self.classifiers = {}
        self.predictData = None
        self.accuracy = 0

    def sqp(self, paraP, paraq, paraG, parah):
        P = matrix(np.array(paraP), tc='d')
        q = matrix(np.array(paraq), tc='d')
        G = matrix(np.array(paraG), tc='d')
        h = matrix(np.array(parah), tc='d')
        # A = matrix(np.array(paraA), tc='d')
        # b = matrix(np.array(parab), tc='d')
        return solvers.qp(P, q, G, h)
    
    def gaussian_kernel(self, x, y):
        return np.exp(-np.linalg.norm(x-y, ord=2, axis=1)**2 / (2 * (self.sigma ** 2)))
    
    def jacabian1(self, beta, X):
        """返回超圆盘的对偶问题的目标函数的雅可比矩阵
        Args:
            beta (ndarray): 为该输入样本集合下的拉格朗日系数
            X (ndarray): X为输入样本矩阵, 行为样本个数, 列为样本特征
        """
        return np.sum(X*X, 1)-2*np.sum(X@(X.T*beta), 1)
    
    def Objective_Function1(self, beta, X):
        """返回超圆盘的对偶问题的目标函数，在某一拉格朗日系数下的值
        Args:
            beta (ndarray): 为该输入样本集合下的拉格朗日系数
            X (ndarray): X为输入样本矩阵, 行为样本个数, 列为样本特征
        """
        """ Q = X@X.T
        # print(X)
        return np.dot(beta, Q@beta)-np.dot(beta, np.diag(Q)) """ # 慢很多呀
        beta = beta.reshape(-1, 1)
        # print(X)
        return np.sum((beta*X)@((beta*X).T))-np.sum(beta*np.sum(X*X, 1).reshape(-1, 1))
    
    def Objective_Function2(self, alpha, X_plus, X_minus):
        M_plus  =X_plus.shape[0]
        alpha_plus = alpha[:M_plus]
        alpha_minus = alpha[M_plus:]
        M_minus = alpha_minus.shape[0]
        result = 0
        for i in range(M_plus):
            result = result + np.sum(alpha_plus[i]*alpha_plus*self.gaussian_kernel(X_plus[i, :], X_plus))
            - 2*np.sum(alpha_plus[i]*alpha_minus*self.gaussian_kernel(X_plus[i, :], X_minus))
        for i in range(M_minus):
            result = result + np.sum(alpha_minus[i]*alpha_minus*self.gaussian_kernel(X_minus[i, :], X_minus))
        return result
    
    def nonlcon(self, alpha, beta_plus, beta_minus, X_plus, X_minus, r_plus, r_minus):
        M_plus  =X_plus.shape[0]
        alpha_plus = alpha[:M_plus]
        alpha_minus = alpha[M_plus:]
        M_minus = alpha_minus.shape[0]
        
        c = np.zeros((2, 1))
        for i in range(M_plus):
            c[0] = c[0] + np.sum(alpha_plus[i]*alpha_plus*self.gaussian_kernel(X_plus[i, :],X_plus))
            - np.sum(alpha_plus[i]*beta_plus*self.gaussian_kernel(X_plus[i, :], X_plus))
            + np.sum(beta_plus[i]*beta_plus*self.gaussian_kernel(X_plus[i, :], X_plus))
        c[0] -= r_plus**2
        for i in range(M_minus):
            c[0] = c[0] + np.sum(alpha_minus[i]*alpha_minus*self.gaussian_kernel(X_minus[i, :],X_minus))
            - np.sum(alpha_minus[i]*beta_minus*self.gaussian_kernel(X_minus[i, :], X_minus))
            + np.sum(beta_minus[i]*beta_minus*self.gaussian_kernel(X_minus[i, :], X_minus))
        c[1] -= r_minus**2
        return c.flatten()

    def fit(self, X, y):
        # 拟合函数
        classes = np.unique(y)
        Num = np.size(classes)
        beta = np.zeros((Num, y.shape[0]))
        s_and_r = np.ones((Num, X.shape[1]+1))
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
        
        # 求解多个分类器的参数，即分类超平面参数，OVO策略
        Num_classifier = int(Num*(Num-1)/2)
        b_classifier = np.zeros((Num_classifier, 1))
        iter = 0
        for i in range(Num-1):
            for j in range(i+1, Num):
                iter += 1
                # 获取基本信息
                X_plus = X[y==classes[i], :]
                X_minus = X[y==classes[j], :]
                M_plus = X_plus.shape[0]
                M_minus = X_minus.shape[0]
                
                P = 2*X@X.T
                P[0:M_plus, M_plus:] = - P[:M_plus, M_plus:]
                P[M_plus:, :M_plus] = - P[M_plus:, : M_plus]

                # q = -np.diag(Xx@Xx.T)
                G = np.concatenate((np.eye(M), -np.eye(M)), axis=0)
                h = np.concatenate((np.ones([M, 1]), -np.zeros([M, 1])), axis=0)
                A = np.zeros([2, X.shape[0]])
                A[0,: M_plus] = 1
                A[1, M_plus:] = 1
                b = np.array([1, 1])
                sol = self.sqp(P, q, G, h)
                
                alpha = result2.x
                alpha_plus = alpha[:M_plus]
                alpha_minus = alpha[M_plus:]
                # 求解b
                for k in range(M_plus):
                    b_classifier[iter-1] += np.sum(alpha_plus[k]*alpha_plus*self.gaussian_kernel(X_plus[k, :], X_plus))
                for k in range(M_minus):
                    b_classifier[iter-1] -= np.sum(alpha_minus[k]*alpha_minus*self.gaussian_kernel(X_minus[k, :], X_minus))
                b_classifier[iter-1] = -1/2*b_classifier[iter-1]
                b = b_classifier[iter-1]
                self.classes = Num
                self.classifiers[(classes[i], classes[j])] = {'X+': X_plus,
                                                'X-': X_minus,
                                                'alpha+': alpha_plus,
                                                'alpha-': alpha_minus,
                                                'b': b}    
        # 保存分类器参数字典
        with open('classifiers.pkl', 'wb') as f:
            pickle.dump(self.classifiers, f)
            
        with open('classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)   
            
        with open('s_r.pkl', 'wb') as f:
            pickle.dump(self.s_r, f) 
            
        with open('predictData.pkl', 'wb') as f:
            pickle.dump(self.predictData, f)   
                             
        return 0  
                

    def predict(self,X):
        fx = np.zeros((X.shape[0], self.classes))
        indexclasses = 0
        fxij = np.zeros((X.shape[0], self.classes))
        for (i, j), classifier in self.classifiers.items():
            indexclasses += 1
            X_pl = classifier['X+']
            X_mi = classifier['X-']
            alpha_pl = classifier['alpha+']
            alpha_mi = classifier['alpha-']
            bb = classifier['b']
            
            M_plus = X_pl.shape[0]
            M_minus = X_mi.shape[0]
            
            for g in range(M_plus):
                fx[:, indexclasses-1] = fx[:, indexclasses-1] + alpha_pl[g]*self.gaussian_kernel(X_pl[g, :], X).reshape(-1)
            for g in range(M_minus):
                fx[:, indexclasses-1] = fx[:, indexclasses-1] - alpha_mi[g]*self.gaussian_kernel(X_mi[g, :], X).reshape(-1)
            
            fx[:, indexclasses-1] = np.sign(fx[:, indexclasses-1] + bb)
            fxij[:, indexclasses-1] = fx[:, indexclasses-1] 
            fx[:, indexclasses-1][fx[:, indexclasses-1]==1] = i
            fx[:, indexclasses-1][fx[:, indexclasses-1]==-1] = j
            fx[:, indexclasses-1][fx[:, indexclasses-1]==0] = 0


        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), axis=1, arr=fx) # axis表示假如你是n×m维度，axis=0，表示在n方向上，axis=1，表示在m方向上
        y_pred = y_pred.reshape(-1, 1)
        self.predictData = np.concatenate((X, y_pred, fxij, fx), axis=1)
        return y_pred
    
    def score(self, y_pred, y):
        y = y.reshape(-1, 1)
        accuray = np.sum(y_pred==y) / y.shape[0]
        self.accuracy = accuray
        return accuray