import numpy as np
import time

def adapt_integate(D=[0, 1], epsilon=10**-7):
    """ 变步长算法，梯形法求积分
        返回值: 输出T_n和T_2n
    Args:
        D (list): 积分区间
        epsolon (_type_, optional): _description_. Defaults to 10**-7.
    """
    a, b = D
    if a == 0:
        a = 0.000000001
    if b == 0:
        b = 0.000000001
    i = 2
    x0 = np.linspace(a,b,i)
    
    T_n = (b - a)/2 * (sum(np.sin(x0) / x0))
    T_2n = 0
    
    tic = time.time()
    print('计算中……')
    while abs(T_n - T_2n) >= epsilon:
        T_2n = T_n
        
        x = np.setdiff1d(np.linspace(a, b, 2*i-1), np.linspace(a, b, i)) # np.setdiff1d 返回在a中不在b中的元素
        T_n = T_2n/2 + (b-a)/(2*i-1) * np.sum(np.sin(x) / x)
        i = 2 * i - 1
        
        toc = time.time()
        if toc - tic > 10:
            break 
    
    print('计时10s计算的结果为：')
    print(f'T_2n = {T_2n}; T_n = {T_n}')

    return 0

if __name__ == '__main__':
    adapt_integate([0, 1])
