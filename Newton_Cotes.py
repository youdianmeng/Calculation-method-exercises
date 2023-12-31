import numpy as np
def Newton_Cotes(a=0, b=1, n=7):
    """特定函数的newton-cotes求积实现

    Args:
        a (float): 积分开始值
        b (float): 积分结束值
        n (int): cotes求积区间份数
    """
    ## 建立n=1:7的cotes系数表，便于查询
    Ctable = np.array([[1/2, 1/2, 0, 0, 0, 0, 0, 0, 0],
                      [1/6, 2/3, 1/6, 0, 0, 0, 0, 0, 0],
                      [1/8, 3/8, 3/8, 1/8, 0, 0, 0, 0, 0],
                      [7/90, 16/45, 2/15, 16/45, 7/90, 0, 0, 0, 0],
                      [19/288, 25/96, 25/144, 25/144, 25/96, 19/288, 0, 0, 0],
                      [41/840, 9/35, 9/280, 34/105, 9/280, 9/35, 41/840, 0, 0],
                      [3577/17280, 3577/17280, 1323/17280, 2989/17280, 2989/17280, 1323/17280, 3577/17280, 3577/17280, 0]])
    
    ## 计算各节点及其函数值
    x = np.linspace(a, b, n + 1)
    # print(x)
    fx = 1 / (1 + x ** 2)
    # print(fx)
    # print(Ctable[n-1,:].shape)
    ## 增维fx，使其满足广播条件便于计算
    fx = np.append(fx, np.ones(Ctable.shape[1]-(n+1)))
    # print(fx.shape)
    ## 计算结果
    result = fx @ Ctable[n-1,:] * (b-a)
    
    return result

if __name__ == '__main__':
    ## 测试
    print(Newton_Cotes())