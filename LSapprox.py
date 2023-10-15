import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def LSapprox(x, y, n):
    """ 最小二乘拟合

    Args:
        x (float): 样本点横坐标组成的向量
        y (float): 样本点纵坐标组成的向量
        n (int): 拟合阶数
    """
    ## 检查输入
    if len(x) != len(y):
        raise ValueError("The lengths of x and y must be equal")
    if x==0 or y==0:
        raise ValueError("The inputs cannot be empty")
    
    x = np.array(x)
    y = np.array(y)
    
    plt.figure('Raw data and fitted images')
    plt.grid()
    plt.scatter(x, y, c='None',marker='o', edgecolors='b', label="原始数据")

    ## 构造矩阵x_ij和向量b
    x_ij = np.zeros((n+1, n+1))
    yk = np.zeros((n+1, 1))
    
    for i in range(n+1):
        for j in range(n+1):
            x_ij[i, j] = np.sum(x**(i+j))
        yk[i] = np.sum(x**(i) * y)
    
    a = np.linalg.solve(x_ij, yk) ## a_0在前,求解Ax = b
    fx = np.poly1d(a.flatten()[::-1]) # flatten()展开成 一维 的，np.poly1d只能接受一维输入
    print('多项式表达式：')
    print(fx)
    
    xx = np.linspace(np.min(x), np.max(x)).tolist()
    ## yy = np.polyval(reversed(a), xx) ## a_0在前，故需要翻转
    yy = np.polyval(a[::-1], xx)


    plt.plot(xx, yy,'r-', label='拟合数据')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
if __name__ == '__main__':
    x = [0, 0.25, 0.50, 0.75, 1.00]
    y = [1.0000, 1.2840, 1.6487, 2.1170, 2.7183]
    LSapprox(x, y, 2)