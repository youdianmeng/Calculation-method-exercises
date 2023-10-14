import numpy as np

def Gauss_Seidel(A=[[-4, 1, 1, 1, 1],
                    [1, -4, 1, 1, 1],
                    [1, 1, -4, 1, 1],
                    [1, 1, 1, -4, 1]]):
    """ 高斯-赛德尔迭代
        返回值：方程组解
    Args:
        A (list): 增广矩阵
    """
    
    A = np.array(A,dtype=float)
    [M, N] = np.shape(A)
    
    x0 = np.zeros((M, 1))
    xk = np.ones((M, 1))
    
    while np.max(np.abs(x0 - xk)) >= 0.0000001:
        
        xk = x0
        for i in range(0, M):
            if i == 1:
                x0[i] = (1/A[i, i]) * (A[i, -1] - np.dot(A[i, i+1:-1], xk[i+1:]))
            elif i == M - 1:
                x0[i] = (1/A[i, i]) * (A[i, -1] - np.dot(A[i, :i], x0[:i])) # np.dot 好像可以自动识别前后向量维度来计算
            else:
                x0[i] = (1/A[i, i]) * (A[i, -1] - np.dot(A[i, :i], x0[:i]) - np.dot(A[i, i+1:-1], xk[i+1:]))
            
    return x0

if __name__ == '__main__':

    A = [[20, 2, 3, 24],[1, 8, 1, 12],[2, -3, 15, 30]]
    print(Gauss_Seidel(A))