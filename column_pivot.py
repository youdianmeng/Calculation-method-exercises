import numpy as np

def column_pivot(A=[[-3, 2, 6, 4],
                    [10, -7, 0, 7],
                    [5, -1, 5, 6]]):
    """ 列主元消去法
        返回值：方程组解
    Args:
        A (list): 增广矩阵
    """
    
    A = np.array(A, dtype=float) ## 很关键！确保后续计算不会默认A为int型，计算时四舍五入了
    [N, M] = np.shape(A)
    x = np.zeros((N, 1))
    
    print('消元中……')
    for i in range(N-1):
        max_index = np.argmax(np.abs(A[i:,i:][:,0]))
        print(max_index)
        
        # print(A[i, :])
        # print(A[max_index + i, :])
        temp = np.copy(A[i, :]) 
        A[i,:] = A[max_index + i,:]
        A[max_index + i, :] = temp
        print(A)
        
        """ 
        A[i], A[max_index + i] = A[max_index + i], A[i] # python多重赋值特性不成立！！！
        因为多重赋值只针对与list类型,对于narray类型不起作用"""
        
        """ 
        temp = A[i, :] ## 这里会将A[i, :]命名成temp,在更改A[i, :]时，
        print(temp)         # temp也会变,故不能如此变换两行
        print('')
        A[i,:] = A[max_index + i,:]
        print(A[i,:])
        print(temp) 
        print('')
        A[max_index + i, :] = temp
        print(A) """
        
        if A[i, i] != 0:
            for j in range(i + 1, N):
                # print(A[j, i], A[i, i])
                # print(A[j, i]/A[i, i] * A[i, :])
                A[j, :] = A[j, :] - A[j, i]/A[i, i] * A[i, :] ##     A = np.array(A, dtype=float),定义dtype，防止int化
                # print(A[j, :])
                
                
        else:
            print('待解决……')
            break
        
    print('消元结果：')
    print(A)
        
    x[N-1] = A[-1, -1] / A[-1, -2]
    for i in reversed(range(N-1)):
        x[i] = A[i, -1]
        
        for j in range(i + 1, N):
            x[i] = x[i] - A[i, j] * x[j]
            
        x[i] = x[i] / A[i, i]
        
    print(f'求解得x={x}')
    
    return 0

if __name__ == '__main__':
    print(column_pivot())