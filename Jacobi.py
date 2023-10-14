import numpy as np

A = None
def Jacobi(A=[[-4, 1, 1, 1, 1],
             [1, -4, 1, 1, 1],
             [1, 1, -4, 1, 1],
             [1, 1, 1, -4, 1]]):
    """ 雅可比迭代向量化
        返回值：方程组解
    Args:
        A (list): 增广矩阵
    """ 
    A = np.array(A, dtype=float)
    [M, N] = np.shape(A)
    
    aii = np.diag(A[:,:-1]).reshape(-1, 1) # reshape转换成列向量
    aij = A[:,:-1] - np.eye(M) * aii # 始终注意python索引从0开始。左闭右开
    
    x0 = np.zeros((M, 1)) # 双重(())
    xk = np.ones((M, 1))
    
    while np.max(np.abs(x0 - xk)) >= 0.00000001: # np.max 和 np.abs类型都为narray
        
        xk = x0
        x0 = (np.divide(1, aii)) * (A[:, -1].reshape(-1, 1) - np.dot(aij, xk))
        
    return x0

if __name__ == '__main__':

    A = [[20, 2, 3, 24],[1, 8, 1, 12],[2, -3, 15, 30]]
    print(Jacobi(A))