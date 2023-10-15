import numpy as np

def Newton_descent(x0=-0.99, epsilon=10**-5):
    """ 牛顿下山法求解 (x^3/x - x = 0)

    Args:
        x0 (float): 解的设定初值
        epsilon (float): 精度
    """
     
    lamb = list(map(lambda i: (1/2)**(i-1), range(1, 11))) # λ
    xk1 = x0
    xk = 0.0

    k = 0
    while abs(xk - xk1) >= epsilon:
        k = k + 1
        print(f'k = {k}')
        xk = xk1
        for i in range(10):
            xk1 = xk - lamb[i] * (xk**3/3 - xk)/(xk**2-1)
            print(f'lambda = {lamb[i]}')
            print(f'xk1 = {xk1}')
            
            if abs(xk1**3-xk1) < abs(xk**3/3-xk):
                break
    print(xk, xk1)
    return 0

if __name__ == '__main__':
    Newton_descent()