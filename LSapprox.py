import numpy as np
def LSapprox(x, y, n):
    """最小二乘拟合

    Args:
        x (float): 样本点横坐标组成的向量
        y (float): 样本点纵坐标组成的向量
        n (int): 拟合阶数
    """
    ## 检查输入
    if len(x) != len(y):
        raise ValueError("The lengths of x and y must be equal")
    if x.size==0 or y.size==0:
        raise ValueError("The inputs cannot be empty")
    
    ## 预处理
    x = x.reshape(1, -1)#改变维度为1行、d列 （-1表示列数自动计算，d= a*b /m ）
    y = y.reshape(1, -1)
    
    ## 构造矩阵A 和 向量b
    